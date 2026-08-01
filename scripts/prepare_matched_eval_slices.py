#!/usr/bin/env python3
"""Materializa T_BR e T_nonBR (pareados 1:1) no esquema do cache de eval — item (b), parte 1.

Por que este passo existe: o T_nonBR congelado (`t_nonbr_matched_soft.parquet`) e um arquivo de
PARES (colunas `match_set_id`, `br_variant_key`, `nonbr_variant_key`, `br_label`, ...), NAO variantes
prontas. Ja o `eval.clinvar.dataset.build_variant_cache` (usado pelo evaluate_clinvar_finetuned_model)
exige por linha: `Chromosome`, `Start`, `ReferenceAlleleVCF`, `AlternateAlleleVCF`, `label` e SEMPRE a
coluna de split (`split_within_gene`). Entao materializamos as variantes a partir do MASTER (fonte
canonica, esquema conhecido = MASTER_COLS), preservando o `match_set_id` de cada par — que a Fase 4
usa para o bootstrap pareado por matched-set (§10 do Eduardo).

Saida (no --out-dir):
  * t_br_matched.parquet   — os `br_variant_key` dos pares (o BR do contraste)
  * t_nonbr_matched.parquet — os `nonbr_variant_key` dos pares (o controle nonBR pareado)
Ambos: MASTER_COLS + `match_set_id` + `split_within_gene="test"`. Mesma contagem, mesmo match_set_id
alinhando o par BR<->nonBR. Chr8 nao deve aparecer (holdout §11); validado e reportado.

NAO parear/filtrar por ABraOM/AMR aqui (§6.7): apenas materializamos os conjuntos ja congelados.

Rodar (no notebook):
    cd "$WORK" && PYTHONPATH="$WORK" "$PY" scripts/prepare_matched_eval_slices.py \
        --pairs ~/artifacts/fase0/t_nonbr_matched_soft.parquet \
        --master-uri s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet \
        --out-dir ~/artifacts/fase0/eval_slices
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_matched_eval_slices")

# Espelha scripts/build_clinvar_splits.py (mesma fonte, mesmo esquema).
MASTER_COLS = ["variant_key", "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
               "label", "GeneSymbol", "has_brazilian_submitter", "consequence_bucket"]


def read_master(uri: str) -> pd.DataFrame:
    """Le apenas MASTER_COLS do master (local ou s3://), via pyarrow (leitura colunar eficiente)."""
    import pyarrow.parquet as pq

    if uri.startswith("s3://"):
        from pyarrow import fs

        without = uri[len("s3://"):]
        bucket = without.split("/", 1)[0]
        s3 = fs.S3FileSystem(region=fs.resolve_s3_region(bucket))
        available = set(pq.ParquetFile(without, filesystem=s3).schema_arrow.names)
        cols = [c for c in MASTER_COLS if c in available]
        df = pq.read_table(without, columns=cols, filesystem=s3).to_pandas()
    else:
        available = set(pq.ParquetFile(uri).schema_arrow.names)
        cols = [c for c in MASTER_COLS if c in available]
        df = pd.read_parquet(uri, columns=cols)
    df["variant_key"] = df["variant_key"].astype(str)
    return df.drop_duplicates("variant_key").reset_index(drop=True)


def _norm_chrom(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.split(":").str[0].str.lower()
            .str.replace("^chr", "", regex=True).str.replace(r"\.0$", "", regex=True))


def _materialize(side: str, keys: pd.DataFrame, master: pd.DataFrame, split_value: str) -> pd.DataFrame:
    """keys: DataFrame com colunas [variant_key, match_set_id]. Junta com o master por variant_key."""
    merged = keys.merge(master, on="variant_key", how="left", validate="one_to_one")
    missing = merged["Chromosome"].isna()
    if missing.any():
        sample = merged.loc[missing, "variant_key"].head(10).tolist()
        raise SystemExit(
            f"[{side}] {int(missing.sum())} variant_key nao encontrados no master (ex.: {sample}). "
            "Confirme que o --master-uri e o mesmo usado na Fase 0."
        )
    merged["split_within_gene"] = split_value
    n_chr8 = int((_norm_chrom(merged["Chromosome"]) == "8").sum())
    if n_chr8:
        log.warning("[%s] %d variantes em chr8 (deveria ser holdout §11) — verifique a fonte.", side, n_chr8)
    log.info("[%s] n=%d | pos=%d neg=%d | genes=%d | chr8=%d",
             side, len(merged), int((merged["label"] == 1).sum()), int((merged["label"] == 0).sum()),
             merged["GeneSymbol"].nunique(), n_chr8)
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True, help="t_nonbr_matched_soft.parquet (pares)")
    parser.add_argument("--master-uri", required=True, help="s3://... ou caminho local do master")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-value", default="test", help="valor gravado em split_within_gene")
    args = parser.parse_args(argv)

    pairs = pd.read_parquet(args.pairs)
    for col in ("match_set_id", "br_variant_key", "nonbr_variant_key"):
        if col not in pairs.columns:
            raise SystemExit(f"Coluna {col!r} ausente em {args.pairs} (colunas: {list(pairs.columns)}).")
    log.info("Pares carregados: n=%d de %s", len(pairs), args.pairs)

    master = read_master(args.master_uri)
    log.info("Master carregado: n=%d variantes canonicas", len(master))

    br_keys = pd.DataFrame({
        "variant_key": pairs["br_variant_key"].astype(str),
        "match_set_id": pairs["match_set_id"].to_numpy(),
    })
    nonbr_keys = pd.DataFrame({
        "variant_key": pairs["nonbr_variant_key"].astype(str),
        "match_set_id": pairs["match_set_id"].to_numpy(),
    })

    t_br = _materialize("T_BR", br_keys, master, args.split_value)
    t_nonbr = _materialize("T_nonBR", nonbr_keys, master, args.split_value)

    # Sanity de pareamento: mesma contagem e mesmos match_set_id dos dois lados.
    if len(t_br) != len(t_nonbr) or set(t_br["match_set_id"]) != set(t_nonbr["match_set_id"]):
        raise SystemExit("Pareamento inconsistente: match_set_id divergente entre T_BR e T_nonBR.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    br_path = args.out_dir / "t_br_matched.parquet"
    nonbr_path = args.out_dir / "t_nonbr_matched.parquet"
    t_br.to_parquet(br_path, index=False)
    t_nonbr.to_parquet(nonbr_path, index=False)
    log.info("Escrito %s e %s (%d pares, prevalencia BR=%.3f nonBR=%.3f)",
             br_path, nonbr_path, len(t_br),
             float((t_br["label"] == 1).mean()), float((t_nonbr["label"] == 1).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
