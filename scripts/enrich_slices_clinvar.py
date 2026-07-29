#!/usr/bin/env python3
"""Fase 0 -- enriquece as slices com CONSEQUENCIA molecular (VCF ClinVar) + ESTRELAS (master).

Fecha os gaps de pareamento do matcher (§6.3 do Eduardo): a consequencia (match EXATO) sai do campo
`MC` do VCF ClinVar, mapeada por severidade (tipo-VEP); as estrelas (aproximado) saem do
`max_review_status_rank_aggregate` do master. Junta por `variant_key` (chrom:pos:ref:alt, sem 'chr',
1-based) -- a MESMA chave das slices e do VCF NCBI.

O ano (LastEvaluated) NAO esta no VCF (so no XML VCV/RCV) -> fica como gap documentado (aproximado).

USO
---
    # 1) baixar o VCF (uma vez):
    aws s3 cp s3://ai4bio-lumina/benchmarks/mosaic/data/raw/clinvar/clinvar_20260606.vcf.gz ~/data/clinvar/
    # 2) enriquecer:
    python scripts/enrich_slices_clinvar.py \\
        --br ~/slices/br_only.parquet --nonbr ~/slices/nonbr_only.parquet \\
        --vcf ~/data/clinvar/clinvar_20260606.vcf.gz \\
        --master-uri s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet \\
        --out-dir ~/slices_enriched
"""

from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enrich_clinvar")

# Severidade tipo-VEP: (termo SO no MC do ClinVar) -> categoria. Ordem = mais severo primeiro; para
# uma variante com varios termos (varios transcritos), fica a categoria do termo mais severo.
CONSEQUENCE_SEVERITY: list[tuple[str, str]] = [
    ("stop_gained", "nonsense"),
    ("nonsense", "nonsense"),
    ("frameshift_variant", "frameshift"),
    ("splice_acceptor_variant", "splice"),
    ("splice_donor_variant", "splice"),
    ("start_lost", "start_stop_lost"),
    ("stop_lost", "start_stop_lost"),
    ("missense_variant", "missense"),
    ("inframe_insertion", "inframe_indel"),
    ("inframe_deletion", "inframe_indel"),
    ("inframe_variant", "inframe_indel"),
    ("protein_altering_variant", "protein_altering"),
    ("splice_region_variant", "splice_region"),
    ("synonymous_variant", "synonymous"),
    ("stop_retained_variant", "synonymous"),
    ("initiator_codon_variant", "synonymous"),
    ("5_prime_UTR_variant", "utr"),
    ("3_prime_UTR_variant", "utr"),
    ("non-coding_transcript_variant", "noncoding"),
    ("non_coding_transcript_variant", "noncoding"),
    ("intron_variant", "intron"),
    ("no_sequence_alteration", "other"),
    ("genic_downstream_transcript_variant", "other"),
    ("genic_upstream_transcript_variant", "other"),
]
_TERM_RANK = {term: i for i, (term, _cat) in enumerate(CONSEQUENCE_SEVERITY)}
_TERM_CAT = {term: cat for term, cat in CONSEQUENCE_SEVERITY}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--br", required=True, type=Path)
    p.add_argument("--nonbr", required=True, type=Path)
    p.add_argument("--vcf", required=True, type=Path, help="clinvar_*.vcf.gz local (baixe com aws s3 cp)")
    p.add_argument("--master-uri", required=True, help="s3://... ou caminho local do master (p/ estrelas)")
    p.add_argument("--stars-col", default="max_review_status_rank_aggregate")
    p.add_argument("--out-dir", required=True, type=Path)
    return p.parse_args(argv)


def load_target_keys(*slice_paths) -> set[str]:
    keys: set[str] = set()
    for pth in slice_paths:
        s = pd.read_parquet(pth, columns=["variant_key"])["variant_key"].astype(str)
        keys.update(s.tolist())
    return keys


def _most_severe(mc_value: str) -> str:
    """MC='SO:0001587|nonsense,SO:0001627|intron_variant' -> categoria do termo mais severo."""
    best_rank, best_cat = 10**9, "unknown"
    for token in mc_value.split(","):
        term = token.split("|")[-1].strip().lower() if "|" in token else token.strip().lower()
        rank = _TERM_RANK.get(term)
        if rank is not None and rank < best_rank:
            best_rank, best_cat = rank, _TERM_CAT[term]
    return best_cat


def parse_vcf_consequence(vcf_path: Path, target_keys: set[str]) -> dict[str, str]:
    """Stream do VCF gzip; extrai a consequencia so das variantes-alvo (chrom:pos:ref:alt sem 'chr')."""
    out: dict[str, str] = {}
    n_lines = 0
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_lines += 1
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            chrom, pos, ref, alts, info = f[0], f[1], f[3], f[4], f[7]
            for alt in alts.split(","):
                key = f"{chrom}:{pos}:{ref}:{alt}"
                if key not in target_keys:
                    continue
                mc = next((kv[3:] for kv in info.split(";") if kv.startswith("MC=")), "")
                out[key] = _most_severe(mc) if mc else "no_mc"
            if n_lines % 1_000_000 == 0:
                log.info("  ... %d linhas VCF lidas, %d alvos anotados", n_lines, len(out))
    log.info("VCF: %d linhas, %d/%d variantes-alvo com consequencia", n_lines, len(out), len(target_keys))
    return out


def load_stars(master_uri: str, stars_col: str) -> dict[str, float]:
    cols = ["variant_key", stars_col]
    if master_uri.startswith("s3://"):
        import pyarrow.fs as fs
        import pyarrow.parquet as pq

        without = master_uri[len("s3://"):]
        bucket = without.split("/", 1)[0]
        region = fs.resolve_s3_region(bucket)
        s3 = fs.S3FileSystem(region=region)
        tbl = pq.read_table(without, columns=cols, filesystem=s3)
        df = tbl.to_pandas()
    else:
        df = pd.read_parquet(master_uri, columns=cols)
    df = df.dropna(subset=["variant_key"]).drop_duplicates("variant_key")
    return dict(zip(df["variant_key"].astype(str), pd.to_numeric(df[stars_col], errors="coerce")))


def enrich(slice_path: Path, consequence_map, stars_map, out_dir: Path) -> dict:
    df = pd.read_parquet(slice_path)
    vk = df["variant_key"].astype(str)
    df["consequence"] = vk.map(consequence_map).fillna("unknown")
    df["review_star_rank"] = vk.map(stars_map)
    out_path = out_dir / slice_path.name.replace(".parquet", ".enriched.parquet")
    df.to_parquet(out_path, index=False)
    cons_cov = float((df["consequence"] != "unknown").mean())
    star_cov = float(df["review_star_rank"].notna().mean())
    log.info("%s -> %s | consequencia cobertura=%.3f | estrelas cobertura=%.3f | consequencias=%s",
             slice_path.name, out_path.name, cons_cov, star_cov,
             dict(df["consequence"].value_counts().head(12)))
    return {"slice": slice_path.name, "out": str(out_path), "n": len(df),
            "consequence_coverage": cons_cov, "star_coverage": star_cov,
            "consequence_counts": {str(k): int(v) for k, v in df["consequence"].value_counts().items()}}


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target = load_target_keys(args.br, args.nonbr)
    log.info("alvos (br+nonbr, unicos)=%d", len(target))

    consequence_map = parse_vcf_consequence(args.vcf, target)
    stars_map = load_stars(args.master_uri, args.stars_col)
    log.info("estrelas: %d variantes com rank no master", len(stars_map))

    for pth in (args.br, args.nonbr):
        enrich(pth, consequence_map, stars_map, args.out_dir)

    log.info("Enriquecimento pronto em %s. Rode o matcher com --exact-cols GeneSymbol label variant_type "
             "consequence (as slices enriquecidas ja tem 'consequence' e 'review_star_rank').", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
