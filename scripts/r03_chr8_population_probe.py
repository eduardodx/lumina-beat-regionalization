#!/usr/bin/env python3
"""Fase 0.5 -- sonda (b): a interface populacional do R03 ordena frequencia no chr8?

RODA NO HOST GPU DO R03 (precisa do checkpoint + torch + mamba). NAO treina nada.
Extracao SO backbone: para cada variante montamos a janela de referencia do hg38, passamos
pelo R03 e lemos a predicao por-alt da head populacional na posicao da variante. Depois
correlacionamos (Spearman tie-aware) esse score contra o log10(AF) observado.

POR QUE ESTE E UM GATE BARATO (plano R03, Fase 0.5)
---------------------------------------------------
O TECHNICAL.md do R03 (secao 6) avisa que o desempenho de population-AF cai MUITO no chr8,
com evidencia de overfitting aos cromossomos de treino. O eixo representacional do chr8 e o
NOSSO fallback de poder se o endpoint clinico (DiD de MCC) nao tiver n suficiente -- entao,
se a interface populacional ja estiver quebrada no chr8, o fallback esta furado pelo mesmo
buraco (a "ameaca dupla"). Este script mede isso ANTES de gastar qualquer job de treino.

COMO LER O RESULTADO
--------------------
Comparamos o Spearman do set de EVAL (default chr8, held-out) contra um cromossomo IN-DOMAIN
(default chr1). Se rho_chr8 colapsa para perto de 0 enquanto rho_indomain e claramente > 0,
a interface populacional do R03 degrada no chr8 e o eixo representacional perde forca --
levar ao Eduardo antes da Fase 2. Se rho_chr8 se sustenta, o fallback esta vivo.

Isto e SO backbone (nenhum adapter treinado ainda), entao mede o piso do R03 "as-is". A
comparacao M1 vs M2/M3/M4 da Tabela 5 do Eduardo vem depois, com os adapters treinados.

ENTRADA
-------
Um parquet com, no minimo, `variant_key` (formato "chr8:pos:ref:alt") e uma coluna de AF
observada (default `af_gnomad`). Servem tanto as slices de `build_regional_clinvar_eval_slices.py`
quanto o indice ABraOM v2 (17.8M variantes, tem `af_gnomad`) -- o indice da MUITO mais poder no
set representacional (o ponto do Eduardo). Passe `--variants` para qualquer um dos dois.

USO
---
    python scripts/r03_chr8_population_probe.py \\
        --checkpoint s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt \\
        --variants   ~/slices/br_only.parquet \\
        --af-col     af_gnomad \\
        --fasta      ~/hg38/hg38.fa \\
        --eval-chrom chr8 --indomain-chrom chr1 \\
        --context-size 4096 --batch-size 8 --max-variants 4000 \\
        --out ~/artifacts/fase05/r03_chr8_probe.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta

from lumina import batch_encode_dna, load_model_from_checkpoint
from lumina.constants import SNV_ALT_TO_INDEX

from eval.clinvar.variant_utils import extract_variant_window

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("r03_chr8_probe")

# Duas saidas populacionais do R03 (README/TECHNICAL): as duas sao [B, L, 4] sobre A/C/G/T.
POP_SCORE_KEYS = ("gnomad_af_pred", "gnomad_observed_logits")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="s3:// ou caminho local do checkpoint R03")
    p.add_argument("--variants", required=True, type=Path, help="parquet com variant_key + coluna de AF")
    p.add_argument("--af-col", default="af_gnomad", help="coluna de AF observada (default af_gnomad)")
    p.add_argument("--fasta", type=Path, default=Path("~/hg38/hg38.fa").expanduser(), help="hg38 fasta")
    p.add_argument("--eval-chrom", default="chr8", help="cromossomo held-out avaliado (default chr8)")
    p.add_argument("--indomain-chrom", default="chr1", help="cromossomo in-domain de comparacao (default chr1)")
    p.add_argument("--context-size", type=int, default=4096, help="tamanho da janela em bp (default 4096)")
    p.add_argument("--batch-size", type=int, default=8, help="batch de forward (default 8; gotcha #5)")
    p.add_argument("--max-variants", type=int, default=4000, help="teto de variantes por set (0 = sem teto)")
    p.add_argument("--bootstrap", type=int, default=2000, help="reamostras p/ IC 95%% do Spearman (0 = pula)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    p.add_argument("--out", type=Path, default=None, help="caminho do JSON de saida")
    return p.parse_args(argv)


def select_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman tie-aware via mid-ranks (pandas .rank(method='average')) + Pearson dos ranks.

    Sem dependencia de scipy. Empates recebem rank medio, entao a estatistica e deterministica."""
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(x: np.ndarray, y: np.ndarray, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    if resamples <= 0 or len(x) < 3:
        return (float("nan"), float("nan"))
    n = len(x)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        draws[i] = spearman(x[idx], y[idx])
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def ensure_variant_key(df: pd.DataFrame) -> pd.DataFrame:
    """Garante uma coluna `variant_key` ("chrN:pos:ref:alt"). Aceita dois schemas:

    - slices deste repo: ja tem `variant_key`;
    - indice ABraOM v2: tem colunas chrom/pos/ref/alt separadas -> sintetiza a chave.
    """
    if "variant_key" in df.columns:
        return df
    lower = {c.lower(): c for c in df.columns}
    need = ["chrom", "pos", "ref", "alt"]
    if not all(k in lower for k in need):
        raise SystemExit(
            f"parquet sem 'variant_key' e sem chrom/pos/ref/alt (colunas: {list(df.columns)})"
        )
    chrom = df[lower["chrom"]].astype(str)
    chrom = chrom.where(chrom.str.lower().str.startswith("chr"), "chr" + chrom)
    df = df.copy()
    df["variant_key"] = (
        chrom + ":" + df[lower["pos"]].astype(str) + ":" + df[lower["ref"]].astype(str) + ":" + df[lower["alt"]].astype(str)
    )
    return df


def read_variants(path: Path, af_col: str) -> pd.DataFrame:
    """Le so as colunas necessarias (o indice ABraOM tem 17.8M linhas -- nao carregar tudo)."""
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(str(path)).schema.names)
    except Exception:  # noqa: BLE001 -- fallback: le tudo se nao der pra inspecionar o schema
        available = None
    if available is not None:
        key_cols = ["variant_key"] if "variant_key" in available else [
            c for c in available if c.lower() in {"chrom", "pos", "ref", "alt"}
        ]
        wanted = [c for c in (*key_cols, af_col) if c in available]
        if af_col not in available:
            raise SystemExit(f"{path} nao tem a coluna de AF '{af_col}' (colunas: {sorted(available)})")
        return pd.read_parquet(path, columns=wanted or None)
    return pd.read_parquet(path)


def parse_variant_key(key: str) -> tuple[str, int, str, str] | None:
    """"chr8:12345:A:G" -> ("8", 12345, "A", "G"). chrom SEM prefixo (extract_variant_window espera assim)."""
    parts = key.split(":")
    if len(parts) != 4:
        return None
    chrom_raw, pos_raw, ref, alt = parts
    try:
        pos = int(pos_raw)
    except ValueError:
        return None
    chrom = chrom_raw[3:] if chrom_raw.lower().startswith("chr") else chrom_raw
    return chrom, pos, ref.upper(), alt.upper()


def load_chrom_subset(
    df: pd.DataFrame, af_col: str, chrom: str, max_variants: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Filtra o cromossomo pedido, SNV bialelico, AF positiva; amostra ate max_variants."""
    keyed = df["variant_key"].astype(str)
    chrom_prefixed = chrom if chrom.lower().startswith("chr") else f"chr{chrom}"
    sub = df[keyed.str.startswith(f"{chrom_prefixed}:")].copy()
    sub["_af"] = pd.to_numeric(sub[af_col], errors="coerce")
    sub = sub[sub["_af"] > 0]
    # SNV bialelico: ref/alt de 1 base. Usa is_snv se existir, senao deriva do variant_key.
    def _is_snv(key: str) -> bool:
        parsed = parse_variant_key(key)
        return parsed is not None and len(parsed[2]) == 1 and len(parsed[3]) == 1 and parsed[3] in SNV_ALT_TO_INDEX
    sub = sub[sub["variant_key"].astype(str).map(_is_snv)]
    sub = sub.drop_duplicates("variant_key").reset_index(drop=True)
    if max_variants and len(sub) > max_variants:
        sub = sub.sample(n=max_variants, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    return sub


@torch.inference_mode()
def score_variants(
    model: torch.nn.Module,
    fasta: Fasta,
    frame: pd.DataFrame,
    af_col: str,
    context_size: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Retorna {score_key: array} + 'log_af' alinhados, so p/ variantes cuja janela extraiu ok."""
    scores: dict[str, list[float]] = {k: [] for k in POP_SCORE_KEYS}
    log_af: list[float] = []
    skipped = 0

    pending: list[dict] = []
    eps = 0.5 * float(frame["_af"].min())  # metade da menor AF observavel do set (regra do Eduardo)

    def flush() -> None:
        if not pending:
            return
        ref_seqs = [item["ref_seq"] for item in pending]
        batch = batch_encode_dna(ref_seqs, pad_to=context_size, device=device)
        out = model(**batch)
        for key in POP_SCORE_KEYS:
            head = out[key]  # [B, L, 4]
            for i, item in enumerate(pending):
                scores[key].append(float(head[i, item["offset"], item["alt_idx"]].float().cpu()))
        for item in pending:
            log_af.append(item["log_af"])
        pending.clear()

    for _, row in frame.iterrows():
        parsed = parse_variant_key(str(row["variant_key"]))
        if parsed is None:
            skipped += 1
            continue
        chrom, pos, ref, alt = parsed
        window = extract_variant_window(fasta, chrom, pos, ref, alt, context_size)
        if window is None or window.status == "ref_mismatch":
            skipped += 1
            continue
        if len(window.ref_seq) != context_size:
            skipped += 1
            continue
        af = float(row["_af"])
        pending.append(
            {
                "ref_seq": window.ref_seq,
                "offset": int(window.variant_offset),
                "alt_idx": SNV_ALT_TO_INDEX[alt],
                "log_af": math.log10(af + eps),
            }
        )
        if len(pending) >= batch_size:
            flush()
    flush()

    log.info("Scored %d variants, skipped %d (ref_mismatch/parse/boundary)", len(log_af), skipped)
    result = {k: np.asarray(v, dtype=np.float64) for k, v in scores.items()}
    result["log_af"] = np.asarray(log_af, dtype=np.float64)
    result["_skipped"] = np.asarray([skipped], dtype=np.int64)
    return result


def evaluate_set(
    label: str, scored: dict[str, np.ndarray], resamples: int, rng: np.random.Generator
) -> dict:
    log_af = scored["log_af"]
    n = int(len(log_af))
    entry: dict = {"set": label, "n": n, "n_skipped": int(scored["_skipped"][0])}
    for key in POP_SCORE_KEYS:
        rho = spearman(scored[key], log_af)
        lo, hi = bootstrap_ci(scored[key], log_af, resamples, rng)
        entry[key] = {"spearman": rho, "ci95": [lo, hi]}
    return entry


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    log.info("device=%s checkpoint=%s", device, args.checkpoint)

    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)
    model = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    df = read_variants(args.variants, args.af_col)
    df = ensure_variant_key(df)

    results = []
    for label, chrom in (("eval", args.eval_chrom), ("indomain", args.indomain_chrom)):
        subset = load_chrom_subset(df, args.af_col, chrom, args.max_variants, rng)
        log.info("[%s] chrom=%s n_candidatas=%d", label, chrom, len(subset))
        if len(subset) < 3:
            results.append({"set": label, "chrom": chrom, "n": len(subset), "note": "poucas variantes"})
            continue
        scored = score_variants(model, fasta, subset, args.af_col, args.context_size, args.batch_size, device)
        entry = evaluate_set(label, scored, args.bootstrap, rng)
        entry["chrom"] = chrom
        results.append(entry)

    payload = {
        "checkpoint": str(args.checkpoint),
        "variants": str(args.variants),
        "af_col": args.af_col,
        "context_size": args.context_size,
        "eval_chrom": args.eval_chrom,
        "indomain_chrom": args.indomain_chrom,
        "score_keys": list(POP_SCORE_KEYS),
        "results": results,
    }

    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        log.info("Escrito %s", args.out)

    # Leitura rapida: gap eval-vs-indomain no gnomad_af_pred.
    by_set = {r["set"]: r for r in results if "gnomad_af_pred" in r}
    if "eval" in by_set and "indomain" in by_set:
        rho_eval = by_set["eval"]["gnomad_af_pred"]["spearman"]
        rho_indom = by_set["indomain"]["gnomad_af_pred"]["spearman"]
        log.info(
            "LEITURA: gnomad_af_pred Spearman  %s=%.3f  vs  %s=%.3f  (gap=%.3f)",
            args.eval_chrom, rho_eval, args.indomain_chrom, rho_indom, rho_indom - rho_eval,
        )
        log.info(
            "  gap grande + rho_%s ~0  => interface populacional degradada no chr8 (fallback fraco; "
            "levar ao Eduardo). rho_%s se sustenta => fallback vivo.", args.eval_chrom, args.eval_chrom,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
