#!/usr/bin/env python3
"""Fase 0 -- matcher 1:1 T_nonBR (spec Eduardo secao 6). RODA LOCAL (CPU, pandas+numpy).

Para cada variante em T_BR (br_only), seleciona UMA variante de controle no pool nonBR-only com:
  - match EXATO (6.3): gene, label binario, tipo de variante [, consequencia];
  - dentro do estrato, o candidato de MENOR distancia nas variaveis APROXIMADAS (6.3/6.5);
  - SEM reuso de candidato; se o estrato nao tem candidato livre -> `unmatched` (nao relaxar, 6.5);
  - NAO pareia por ABraOM/AMR/enriquecimento (6.7) -- essas colunas sao so PRESERVADAS.
Emite o par (BR<->nonBR) + relatorio de cobertura (6.6): n_BR_raw, n_BR_matched, coverage, e a
distribuicao de incluidas vs unmatched.

O QUE ESTA ATIVO vs FALTANDO (dado o que as slices carregam hoje)
-----------------------------------------------------------------
Exato: gene, label, variant_type ATIVOS. `consequencia` so entra se existir a coluna (hoje NAO ha
consequencia molecular real -- decisao B1 do Eduardo: anotar via VEP/MolecularConsequence, ou
aprovar tipo-como-proxy). Aproximado: af_gnomad + nº submitters + comprimento de indel ATIVOS;
`estrelas` (review_status_rank) e `ano` (LastEvaluated) FALTAM (so no master, que nao esta no disco)
-> documentados como gap, nao usados (sao aproximados, degradacao menor). Config via --exact-cols /
--approx-* deixa subir as variaveis que faltam sem reescrever.

CAVEATS
-------
- Pareamento GULOSO sem reuso (6.5), em ordem estavel por variant_key. Otimo bipartido fica p/ depois.
- Dedup por variante canonica (variant_key) antes de parear (unidade de analise = 6.2/5.2).
- Distancia = soma de |z| das features aproximadas (z-score com media/desvio do POOL nonBR).

USO
---
    python scripts/build_matched_nonbr.py \\
        --br ~/slices/br_only.parquet --nonbr ~/slices/nonbr_only.parquet \\
        --exclude-chrom 8 --out ~/artifacts/fase0/t_nonbr_matched.parquet \\
        --report ~/artifacts/fase0/matching_coverage.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("matcher_nonbr")

DEFAULT_EXACT = ["GeneSymbol", "label", "variant_type"]
# Colunas ABraOM/AMR/enriquecimento: PRESERVAR, nunca parear (6.7).
PRESERVE_COLS = ["af_abraom", "abraom_present", "specificity", "specificity_bin", "af_amr", "amr_present"]
# Bins de AF global do Eduardo (6.4), so p/ o relatorio de cobertura.
AF_BINS = [-np.inf, 0.0, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
AF_LABELS = ["ausente", "<1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", ">=1e-2"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--br", required=True, type=Path, help="parquet do br_only (T_BR)")
    p.add_argument("--nonbr", required=True, type=Path, help="parquet do nonbr_only (pool de controle)")
    p.add_argument("--exact-cols", nargs="+", default=DEFAULT_EXACT,
                   help="colunas de match EXATO (adicione 'consequence_col' quando existir)")
    p.add_argument("--af-col", default="af_gnomad", help="coluna de AF global p/ distancia + bins")
    p.add_argument("--submitter-col", default="regional_submission_rows", help="proxy de nº de submitters")
    p.add_argument("--exclude-chrom", default="8", help="cromossomo excluido do conjunto main (chr8 = holdout)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, required=True, help="parquet dos pares pareados")
    p.add_argument("--report", type=Path, default=None, help="json do relatorio de cobertura")
    return p.parse_args(argv)


def _norm_chrom(df):
    if "variant_key" in df.columns:
        c = df["variant_key"].astype(str).str.split(":").str[0]
    else:
        c = df["Chromosome"].astype(str)
    return c.str.lower().str.replace(r"\.0$", "", regex=True).str.replace("^chr", "", regex=True)


def _indel_length(df):
    ref = df["ReferenceAlleleVCF"].astype(str).str.len()
    alt = df["AlternateAlleleVCF"].astype(str).str.len()
    return (ref - alt).abs().astype(float)


def load_slice(path, exclude_chrom, exact_cols, af_col, submitter_col):
    df = pd.read_parquet(path)
    df = df.drop_duplicates("variant_key").reset_index(drop=True)  # unidade = variante canonica
    n_before_chr = len(df)
    chrom = _norm_chrom(df)
    df = df[chrom != str(exclude_chrom).lower().replace("chr", "")].reset_index(drop=True)
    # features aproximadas (config): af_log, submitters (log1p), indel_len
    af = pd.to_numeric(df[af_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["_af_log"] = np.log10(af + 1e-6)
    df["_af_bin"] = pd.cut(af, AF_BINS, labels=AF_LABELS, include_lowest=True).astype(str)
    submitters = df[submitter_col] if submitter_col in df.columns else pd.Series(0.0, index=df.index)
    df["_submitters"] = np.log1p(pd.to_numeric(submitters, errors="coerce").fillna(0.0))
    df["_indel_len"] = _indel_length(df)
    missing_exact = [c for c in exact_cols if c not in df.columns]
    if missing_exact:
        raise SystemExit(f"{path}: colunas de match exato ausentes: {missing_exact}")
    return df, n_before_chr


APPROX_FEATURES = ["_af_log", "_submitters", "_indel_len"]


def _standardize(br, nonbr):
    """z-score de cada feature aproximada usando media/desvio do POOL nonBR (mesma escala p/ os dois)."""
    stats = {}
    for f in APPROX_FEATURES:
        mu = float(nonbr[f].mean())
        sd = float(nonbr[f].std(ddof=0)) or 1.0
        stats[f] = (mu, sd)
        br[f + "_z"] = (br[f] - mu) / sd
        nonbr[f + "_z"] = (nonbr[f] - mu) / sd
    return stats


def match(br, nonbr, exact_cols, rng):
    """Guloso 1:1 sem reuso, por estrato exato. Retorna (pares_df, unmatched_df)."""
    zcols = [f + "_z" for f in APPROX_FEATURES]
    # estrato = tupla das colunas exatas
    br = br.assign(_stratum=list(zip(*[br[c] for c in exact_cols])))
    nonbr = nonbr.assign(_stratum=list(zip(*[nonbr[c] for c in exact_cols])))
    # pre-computa por estrato: array z (numpy), as linhas (posicional) e a mascara de uso
    pool = {}
    for k, g in nonbr.groupby("_stratum", sort=False):
        pool[k] = {"z": g[zcols].to_numpy(dtype=float),
                   "rows": g.reset_index(drop=True),
                   "used": np.zeros(len(g), dtype=bool)}

    matches, unmatched_idx = [], []
    br_sorted = br.sort_values("variant_key", kind="stable")  # ordem estavel + determinista
    br_z_all = br_sorted[zcols].to_numpy(dtype=float)
    for pos, (idx, row) in enumerate(br_sorted.iterrows()):
        entry = pool.get(row["_stratum"])
        if entry is None:
            unmatched_idx.append(idx)
            continue
        free = np.where(~entry["used"])[0]
        if free.size == 0:
            unmatched_idx.append(idx)
            continue
        dist = np.abs(entry["z"][free] - br_z_all[pos]).sum(axis=1)
        pick_local = int(np.argmin(dist))
        pick = free[pick_local]
        entry["used"][pick] = True
        matches.append((row, entry["rows"].iloc[pick], float(dist[pick_local])))

    pairs = _assemble_pairs(matches)
    unmatched = br.loc[unmatched_idx].copy()
    return pairs, unmatched


def _assemble_pairs(matches):
    rows = []
    carry = ["variant_key", "GeneSymbol", "label", "variant_type", "af_gnomad", "_af_bin"] + PRESERVE_COLS
    for i, (br_row, nb_row, dist) in enumerate(matches):
        rec = {"match_set_id": i, "match_distance": dist}
        for c in carry:
            if c in br_row:
                rec[f"br_{c}"] = br_row[c]
            if c in nb_row:
                rec[f"nonbr_{c}"] = nb_row[c]
        rows.append(rec)
    return pd.DataFrame(rows)


def coverage_report(br, pairs, unmatched, n_before_chr, exclude_chrom, exact_cols, af_col, submitter_col):
    n_raw = len(br)
    n_matched = len(pairs)

    def dist_by(frame, col):
        if col not in frame.columns:
            return {}
        return {str(k): int(v) for k, v in frame[col].value_counts().items()}

    return {
        "n_br_rows_deduped": n_before_chr,
        "n_br_main_raw": n_raw,
        "n_br_matched": n_matched,
        "n_br_unmatched": len(unmatched),
        "matching_coverage": (n_matched / n_raw) if n_raw else None,
        "exclude_chrom": str(exclude_chrom),
        "exact_cols_used": exact_cols,
        "approx_features_used": ["af_gnomad(log)", f"{submitter_col}(log1p)", "indel_length"],
        "missing_vs_spec": {
            "exact": ["consequencia_molecular (so variant_type disponivel -> decisao B1)"],
            "approx": ["estrelas/review_status_rank (so no master)", "ano/LastEvaluated (nao disponivel)"],
        },
        "not_matched_on_preserved_6_7": [c for c in PRESERVE_COLS if c in br.columns],
        "included_distribution": {
            "label": dist_by(pairs.rename(columns={"br_label": "label"}), "label"),
            "af_bin": dist_by(pairs.rename(columns={"br__af_bin": "af_bin"}), "af_bin"),
        },
        "unmatched_distribution": {
            "label": dist_by(unmatched, "label"),
            "af_bin": dist_by(unmatched, "_af_bin"),
        },
    }


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    br, n_br_dedup = load_slice(args.br, args.exclude_chrom, args.exact_cols, args.af_col, args.submitter_col)
    nonbr, _ = load_slice(args.nonbr, args.exclude_chrom, args.exact_cols, args.af_col, args.submitter_col)
    log.info("br_main=%d (dedup %d) | nonbr_pool_main=%d", len(br), n_br_dedup, len(nonbr))

    _standardize(br, nonbr)
    pairs, unmatched = match(br, nonbr, args.exact_cols, rng)

    report = coverage_report(br, pairs, unmatched, n_br_dedup, args.exclude_chrom,
                             args.exact_cols, args.af_col, args.submitter_col)
    log.info("MATCHED=%d / %d  cobertura=%.3f  (unmatched=%d)",
             report["n_br_matched"], report["n_br_main_raw"], report["matching_coverage"] or 0.0, report["n_br_unmatched"])
    log.info("distribuicao label incluidas=%s | unmatched=%s",
             report["included_distribution"]["label"], report["unmatched_distribution"]["label"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.out, index=False)
    log.info("Escrito %s (%d pares)", args.out, len(pairs))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        log.info("Escrito %s", args.report)

    if (report["matching_coverage"] or 0.0) < 0.8:
        log.warning("Cobertura < 0.80: o T_nonBR pareado representa so uma fracao do T_BR (6.6). "
                    "Comparar distribuicoes incluidas x unmatched antes de congelar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
