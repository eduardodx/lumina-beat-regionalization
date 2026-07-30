#!/usr/bin/env python3
"""Fase 0 (§7) -- constroi os splits ClinVar train/val/cal + manifests com hash (congelamento).

RODA LOCAL (CPU, pandas+pyarrow). Ordem do Eduardo (§7): T_BR e T_nonBR ja congelados -> remover do
treino -> so entao construir train/val/cal. O ClinVar train/val/cal e (§7 + §4.2):
  - label in {0,1} (P/LP ou B/LB, sem VUS);
  - NAO brasileiro (has_brazilian_submitter == False);
  - NAO chr8 (holdout);
  - variant-disjoint dos test sets: `variant_key` fora de T_BR uniao T_nonBR (exclusao §4.2, unidade
    canonica GRCh38:chrom:pos:ref:alt -- cobre SCVs/aliases da MESMA variante);
  - split 80/10/10 POR VARIANTE CANONICA (nunca por submission).

Gera manifests com sha256 (contrato §2) de cada split + dos test sets, pra congelar as identidades.

Split deterministico por HASH do `variant_key` (reproduzivel, independe da ordem das linhas). Opcao
`--split-by gene` agrupa por gene (todas as variantes de um gene no mesmo split) -- mais conservador
contra leakage de gene, superset do variant-disjoint; default `variant` segue a letra do §7.

USO
---
    python scripts/build_clinvar_splits.py \\
        --master-uri s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet \\
        --t-br ~/slices_enriched/br_only.enriched.parquet \\
        --t-nonbr ~/artifacts/fase0/t_nonbr_matched_soft.parquet \\
        --exclude-chrom 8 --out-dir ~/artifacts/fase0/clinvar_splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow.fs as fs
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clinvar_splits")

MASTER_COLS = ["variant_key", "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
               "label", "GeneSymbol", "has_brazilian_submitter", "consequence_bucket"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master-uri", required=True, help="s3://... ou caminho local do master")
    p.add_argument("--t-br", required=True, type=Path, help="parquet do T_BR (br_only enriquecido)")
    p.add_argument("--t-nonbr", required=True, type=Path, help="parquet dos pares T_nonBR congelado (soft)")
    p.add_argument("--exclude-chrom", default="8")
    p.add_argument("--split-by", choices=["variant", "gene"], default="variant")
    p.add_argument("--fracs", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "CAL"))
    p.add_argument("--salt", default="r03-clinvar-splits-v1", help="sal do hash (muda = re-split)")
    p.add_argument("--out-dir", required=True, type=Path)
    return p.parse_args(argv)


def read_master(uri: str) -> pd.DataFrame:
    if uri.startswith("s3://"):
        without = uri[len("s3://"):]
        bucket = without.split("/", 1)[0]
        s3 = fs.S3FileSystem(region=fs.resolve_s3_region(bucket))
        available = set(pq.ParquetFile(without, filesystem=s3).schema_arrow.names)
        cols = [c for c in MASTER_COLS if c in available]
        return pq.read_table(without, columns=cols, filesystem=s3).to_pandas()
    available = set(pq.ParquetFile(uri).schema_arrow.names)
    return pd.read_parquet(uri, columns=[c for c in MASTER_COLS if c in available])


def _norm_chrom(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split(":").str[0].str.lower().str.replace("^chr", "", regex=True).str.replace(r"\.0$", "", regex=True)


def load_test_keys(t_br: Path, t_nonbr: Path) -> tuple[set[str], set[str]]:
    br_keys = set(pd.read_parquet(t_br, columns=["variant_key"])["variant_key"].astype(str))
    pairs = pd.read_parquet(t_nonbr)
    nonbr_keys: set[str] = set()
    for col in ("nonbr_variant_key", "br_variant_key"):
        if col in pairs.columns:
            nonbr_keys.update(pairs[col].astype(str))
    # br_variant_key (subset de T_BR) fica no T_BR; o T_nonBR sao os nonbr_variant_key.
    nonbr_only = {k for k in nonbr_keys if k not in br_keys}
    return br_keys, nonbr_only


def assign_split(keys: pd.Series, salt: str, fracs) -> pd.Series:
    """Bucket deterministico por hash md5(salt+key) -> uniforme [0,1) -> train/val/cal."""
    fr_train, fr_val, _ = fracs

    def bucket(k: str) -> str:
        u = int.from_bytes(hashlib.md5(f"{salt}:{k}".encode()).digest()[:8], "big") / 2**64
        return "train" if u < fr_train else ("validation" if u < fr_train + fr_val else "calibration")

    return keys.map(bucket)


def sha256_keys(keys) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, keys))).encode()).hexdigest()


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_master(args.master_uri)
    log.info("master lido: %d linhas", len(df))

    lab = pd.to_numeric(df["label"], errors="coerce")
    chrom = _norm_chrom(df["variant_key"] if "variant_key" in df.columns else df["Chromosome"])
    hasbr = (df["has_brazilian_submitter"].fillna(False).astype(bool) if "has_brazilian_submitter" in df.columns
             else pd.Series(False, index=df.index))
    pool = df[lab.isin([0, 1]) & (~hasbr) & (chrom != str(args.exclude_chrom).lower().replace("chr", ""))].copy()
    pool = pool.drop_duplicates("variant_key").reset_index(drop=True)
    log.info("pool nao-BR labeled nao-chr8 (dedup): %d", len(pool))

    br_keys, nonbr_keys = load_test_keys(args.t_br, args.t_nonbr)
    exclusion = br_keys | nonbr_keys
    before = len(pool)
    pool = pool[~pool["variant_key"].astype(str).isin(exclusion)].reset_index(drop=True)
    log.info("apos remover T_BR(%d) uniao T_nonBR(%d): %d (removidas do pool: %d)",
             len(br_keys), len(nonbr_keys), len(pool), before - len(pool))

    # chave de split: variante (default, §7) ou gene (mais conservador)
    if args.split_by == "gene" and "GeneSymbol" in pool.columns:
        gene = pool["GeneSymbol"].astype(str)
        split_key = gene.where(gene.str.len() > 0, pool["variant_key"].astype(str))  # sem gene -> por variante
    else:
        split_key = pool["variant_key"].astype(str)
    pool["split"] = assign_split(split_key, args.salt, args.fracs)

    manifest = {"source": args.master_uri, "split_by": args.split_by, "fracs": args.fracs,
                "salt": args.salt, "exclude_chrom": str(args.exclude_chrom), "splits": {}, "test_sets": {}}
    for name in ("train", "validation", "calibration"):
        sub = pool[pool["split"] == name]
        out_path = args.out_dir / f"clinvar_{name}.parquet"
        sub.drop(columns=["split"]).to_parquet(out_path, index=False)
        lab_sub = pd.to_numeric(sub["label"], errors="coerce")
        manifest["splits"][name] = {
            "path": str(out_path), "n": int(len(sub)),
            "n_pos": int((lab_sub == 1).sum()), "n_neg": int((lab_sub == 0).sum()),
            "sha256": sha256_keys(sub["variant_key"]),
        }
        log.info("  %-12s n=%-7d pos=%-7d neg=%-7d sha=%s",
                 name, len(sub), int((lab_sub == 1).sum()), int((lab_sub == 0).sum()),
                 manifest["splits"][name]["sha256"][:12])

    # Parquet COMBINADO com a coluna `split_within_gene` (o que o harness de treino le: dataset.py
    # DEFAULT_SPLIT_COLUMN). Valores train/validation/calibration -> train.py usa validation p/
    # early-stopping e calibration p/ scoring (Platt), via load_variant_cache_by_split.
    combined = pool.drop(columns=["split"]).copy()
    combined["split_within_gene"] = pool["split"].to_numpy()
    combined_path = args.out_dir / "clinvar_splits_combined.parquet"
    combined.to_parquet(combined_path, index=False)
    manifest["combined_dataset"] = {"path": str(combined_path), "n": int(len(combined)),
                                    "split_column": "split_within_gene"}
    log.info("Escrito combinado %s (n=%d, coluna split_within_gene)", combined_path, len(combined))

    manifest["test_sets"]["T_BR"] = {"n": len(br_keys), "sha256": sha256_keys(br_keys)}
    manifest["test_sets"]["T_nonBR_matched"] = {"n": len(nonbr_keys), "sha256": sha256_keys(nonbr_keys)}
    manifest["note"] = ("chr8 benchmarks + BRCA/BRCA2/TP53 sao fatias dos test sets ja congelados (§12.1); "
                        "manifests deles gerados na Fase 5/6. Overlap train/val/cal x test = 0 por construcao.")

    # gate: overlap zero entre os splits e com os test sets
    keysets = {n: set(pd.read_parquet(manifest["splits"][n]["path"], columns=["variant_key"])["variant_key"].astype(str))
               for n in ("train", "validation", "calibration")}
    overlaps = {}
    names = list(keysets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlaps[f"{names[i]}x{names[j]}"] = len(keysets[names[i]] & keysets[names[j]])
    for n in names:
        overlaps[f"{n}xTEST"] = len(keysets[n] & exclusion)
    manifest["overlap_check"] = overlaps
    all_zero = all(v == 0 for v in overlaps.values())
    log.info("overlap check (tem que ser tudo 0): %s", overlaps)

    (args.out_dir / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Escrito %s", args.out_dir / "splits_manifest.json")
    if not all_zero:
        log.error("GATE FALHOU: ha overlap entre splits/test. NAO congelar.")
        return 1
    log.info("GATE OK: sem overlap. Splits congelaveis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
