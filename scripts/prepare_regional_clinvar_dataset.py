#!/usr/bin/env python3
"""Prepare a local ClinVar x ABRAOM regional dataset.

This script intentionally reuses local artifacts produced by earlier work:

* ClinVar pathogenicity splits from the legacy ``lumina`` repository.
* ClinVar Brazilian/non-Brazilian submitter slices from ``lumina-benchmarks``.
* ABRAOM v2 variant index from ``gen-abraom-seqs``.

The main output is a ClinVar parquet compatible with ``eval.clinvar`` plus
metadata columns that make regional and ABRAOM-slice evaluation possible.
No S3 reads or writes are performed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_WORKSPACE = Path("/home/sagemaker-user")
DEFAULT_LUMINA_CLINVAR_DIR = DEFAULT_WORKSPACE / "lumina/data/variants/clinvar/processed"
DEFAULT_REGIONAL_CLINVAR = (
    DEFAULT_WORKSPACE / "lumina-benchmarks/data/datasets/clinvar/processed/eval_all_enriched/eval_unified.parquet"
)
DEFAULT_ABRAOM_INDEX = DEFAULT_WORKSPACE / "gen-abraom-seqs/data/production_v2/abraom_index.v2.parquet"
DEFAULT_OUTPUT_DIR = Path("data/datasets/clinvar/regional_abraom")

CLINVAR_REQUIRED_COLUMNS = {
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "gene",
    "review_status",
    "variant_type",
}

REGIONAL_COLUMNS = [
    "VariationID",
    "cohort",
    "ClinicalSignificance",
    "Submitter",
    "ReviewStatus",
    "Chromosome",
    "Start",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "SubmittedGeneSymbol",
    "submission_match_status",
    "variant_match_status",
    "review_status_rank_submission",
    "review_status_rank_aggregate",
    "number_submitters",
    "variant_type_bucket",
]

ABRAOM_COLUMNS = [
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "af_abraom",
    "af_gnomad",
    "specificity",
]

SPECIFICITY_BINS = [-0.0000001, 0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
SPECIFICITY_LABELS = [
    "0",
    "(0,0.001]",
    "(0.001,0.005]",
    "(0.005,0.01]",
    "(0.01,0.05]",
    "(0.05,0.1]",
    "(0.1,0.5]",
    "(0.5,1]",
]


def normalize_chrom(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    if text in {"23", "x"}:
        return "X"
    if text in {"24", "y"}:
        return "Y"
    if text.lower() in {"m", "mt", "mitochondrial"}:
        return "M"
    return text.upper() if text.upper() in {"X", "Y", "M"} else text


def normalize_allele(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def variant_key(chrom: pd.Series, pos_1based: pd.Series, ref: pd.Series, alt: pd.Series) -> pd.Series:
    return (
        chrom.astype("string")
        + ":"
        + pos_1based.astype("Int64").astype("string")
        + ":"
        + ref.astype("string")
        + ":"
        + alt.astype("string")
    )


def _read_parquet_columns(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    selected = [column for column in columns if column in available]
    return pd.read_parquet(path, columns=selected)


def _require_columns(df: pd.DataFrame, required: set[str], *, path: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _load_lumina_clinvar_split(path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    _require_columns(df, CLINVAR_REQUIRED_COLUMNS, path=path)

    out = pd.DataFrame(
        {
            "Chromosome": df["chrom"].map(normalize_chrom),
            "Start": pd.to_numeric(df["pos"], errors="coerce").astype("Int64"),
            "ReferenceAlleleVCF": df["ref"].map(normalize_allele),
            "AlternateAlleleVCF": df["alt"].map(normalize_allele),
            "label": pd.to_numeric(df["label"], errors="coerce").astype("Int64"),
            "split_within_gene": split_name,
            "source_split": split_name,
            "source_variant_id": df["variant_id"].astype("string"),
            "GeneSymbol": df["gene"].fillna("").astype("string"),
            "gene_symbol": df["gene"].fillna("").astype("string"),
            "source_review_status": df["review_status"].fillna("").astype("string"),
            "variant_type": df["variant_type"].fillna("").astype("string"),
            "source_row_index": range(len(df)),
        }
    )
    if "gene_id" in df.columns:
        out["source_gene_id"] = df["gene_id"]
    if "sequence_ref" in df.columns:
        out["source_sequence_ref"] = df["sequence_ref"]
    if "sequence_alt" in df.columns:
        out["source_sequence_alt"] = df["sequence_alt"]

    out["pos_0_based"] = out["Start"].astype("Int64") - 1
    out["ref_len"] = out["ReferenceAlleleVCF"].str.len().astype("Int64")
    out["alt_len"] = out["AlternateAlleleVCF"].str.len().astype("Int64")
    out["is_snv"] = (
        (out["ref_len"] == 1)
        & (out["alt_len"] == 1)
        & out["ReferenceAlleleVCF"].isin(["A", "C", "G", "T"])
        & out["AlternateAlleleVCF"].isin(["A", "C", "G", "T"])
    )
    out["consequence_bucket"] = out["variant_type"].str.lower().map(
        {"snv": "snv", "mnv": "mnv", "indel": "indel"}
    )
    out["consequence_bucket"] = out["consequence_bucket"].fillna(
        out["is_snv"].map({True: "snv", False: "indel"})
    )
    out["variant_key"] = variant_key(
        out["Chromosome"],
        out["Start"],
        out["ReferenceAlleleVCF"],
        out["AlternateAlleleVCF"],
    )

    valid = (
        out["Chromosome"].ne("")
        & out["Start"].notna()
        & out["ReferenceAlleleVCF"].ne("")
        & out["AlternateAlleleVCF"].ne("")
        & out["label"].isin([0, 1])
    )
    return out.loc[valid].reset_index(drop=True)


def load_lumina_clinvar_splits(clinvar_dir: Path, splits: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = clinvar_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"ClinVar split not found: {path}")
        frames.append(_load_lumina_clinvar_split(path, split))
    return pd.concat(frames, ignore_index=True)


def _join_unique(values: pd.Series, limit: int = 20) -> str:
    unique = sorted({str(value) for value in values.dropna() if str(value).strip()})
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";...(+{len(unique) - limit})"
    return ";".join(unique)


def build_regional_annotation(regional_clinvar_path: Path) -> pd.DataFrame:
    regional = _read_parquet_columns(regional_clinvar_path, REGIONAL_COLUMNS)
    required = {"Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF", "cohort"}
    _require_columns(regional, required, path=regional_clinvar_path)

    regional = regional.copy()
    regional["Chromosome"] = regional["Chromosome"].map(normalize_chrom)
    regional["Start"] = pd.to_numeric(regional["Start"], errors="coerce").astype("Int64")
    regional["ReferenceAlleleVCF"] = regional["ReferenceAlleleVCF"].map(normalize_allele)
    regional["AlternateAlleleVCF"] = regional["AlternateAlleleVCF"].map(normalize_allele)
    regional["cohort"] = regional["cohort"].fillna("unknown").astype("string").str.lower()
    regional["variant_key"] = variant_key(
        regional["Chromosome"],
        regional["Start"],
        regional["ReferenceAlleleVCF"],
        regional["AlternateAlleleVCF"],
    )
    regional = regional.loc[
        regional["Chromosome"].ne("")
        & regional["Start"].notna()
        & regional["ReferenceAlleleVCF"].ne("")
        & regional["AlternateAlleleVCF"].ne("")
    ].reset_index(drop=True)

    grouped_rows: list[dict[str, Any]] = []
    for key, group in regional.groupby("variant_key", sort=False):
        cohort = group["cohort"].astype("string")
        brazilian_rows = int((cohort == "brazilian").sum())
        non_brazilian_rows = int((cohort == "non_brazilian").sum())
        if brazilian_rows and non_brazilian_rows:
            regional_cohort = "mixed"
        elif brazilian_rows:
            regional_cohort = "brazilian"
        elif non_brazilian_rows:
            regional_cohort = "non_brazilian"
        else:
            regional_cohort = "unknown"

        row: dict[str, Any] = {
            "variant_key": key,
            "clinvar_regional_cohort": regional_cohort,
            "has_brazilian_submitter": bool(brazilian_rows),
            "has_non_brazilian_submitter": bool(non_brazilian_rows),
            "brazilian_submission_rows": brazilian_rows,
            "non_brazilian_submission_rows": non_brazilian_rows,
            "regional_submission_rows": len(group),
        }
        if "VariationID" in group:
            row["clinvar_variation_ids"] = _join_unique(group["VariationID"])
            row["clinvar_variation_id_count"] = int(group["VariationID"].dropna().astype(str).nunique())
        if "Submitter" in group:
            row["regional_submitters"] = _join_unique(group["Submitter"])
        if "ClinicalSignificance" in group:
            row["regional_clinical_significance_values"] = _join_unique(group["ClinicalSignificance"])
        if "submission_match_status" in group:
            row["exact_submission_match_rows"] = int((group["submission_match_status"] == "exact").sum())
        if "variant_match_status" in group:
            row["exact_variant_match_rows"] = int((group["variant_match_status"] == "exact").sum())
        if "review_status_rank_submission" in group:
            row["max_review_status_rank_submission"] = pd.to_numeric(
                group["review_status_rank_submission"], errors="coerce"
            ).max()
        if "review_status_rank_aggregate" in group:
            row["max_review_status_rank_aggregate"] = pd.to_numeric(
                group["review_status_rank_aggregate"], errors="coerce"
            ).max()
        grouped_rows.append(row)

    return pd.DataFrame(grouped_rows)


def attach_regional_annotation(clinvar: pd.DataFrame, regional_annotation: pd.DataFrame) -> pd.DataFrame:
    enriched = clinvar.merge(regional_annotation, on="variant_key", how="left")
    count_columns = [
        "brazilian_submission_rows",
        "non_brazilian_submission_rows",
        "regional_submission_rows",
        "clinvar_variation_id_count",
        "exact_submission_match_rows",
        "exact_variant_match_rows",
    ]
    for column in count_columns:
        if column in enriched:
            enriched[column] = enriched[column].fillna(0).astype("int64")

    for column in ["has_brazilian_submitter", "has_non_brazilian_submitter"]:
        if column in enriched:
            enriched[column] = enriched[column].fillna(False).astype(bool)

    enriched["clinvar_regional_cohort"] = enriched["clinvar_regional_cohort"].fillna("unknown")
    enriched["matched_regional_clinvar_benchmark"] = enriched["regional_submission_rows"].fillna(0).astype(int) > 0
    return enriched


def _read_abraom_chromosome(abraom_index: Path, chrom: str) -> pd.DataFrame:
    table = pq.read_table(
        abraom_index,
        columns=ABRAOM_COLUMNS,
        filters=[("chrom", "=", f"chr{chrom}")],
    )
    df = table.to_pandas()
    if df.empty:
        return df
    df["Chromosome"] = df["chrom"].map(normalize_chrom)
    df["Start"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64") + 1
    df["ReferenceAlleleVCF"] = df["ref"].map(normalize_allele)
    df["AlternateAlleleVCF"] = df["alt"].map(normalize_allele)
    return df.rename(columns={"variant_id": "abraom_variant_id", "pos": "abraom_pos_0_based"})


def build_abraom_matches(clinvar: pd.DataFrame, abraom_index: Path) -> pd.DataFrame:
    snv_keys = clinvar.loc[
        clinvar["is_snv"],
        ["Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF", "variant_key"],
    ].drop_duplicates()
    if snv_keys.empty:
        return pd.DataFrame(columns=["variant_key"])

    matches: list[pd.DataFrame] = []
    for chrom in sorted(snv_keys["Chromosome"].dropna().unique(), key=str):
        chrom_keys = snv_keys.loc[snv_keys["Chromosome"] == chrom]
        if chrom in {"Y", "M"}:
            continue
        abraom_chrom = _read_abraom_chromosome(abraom_index, str(chrom))
        if abraom_chrom.empty:
            continue
        merged = chrom_keys.merge(
            abraom_chrom[
                [
                    "Chromosome",
                    "Start",
                    "ReferenceAlleleVCF",
                    "AlternateAlleleVCF",
                    "abraom_variant_id",
                    "abraom_pos_0_based",
                    "af_abraom",
                    "af_gnomad",
                    "specificity",
                ]
            ],
            on=["Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF"],
            how="inner",
        )
        if not merged.empty:
            matches.append(merged)

    if not matches:
        return pd.DataFrame(columns=["variant_key"])
    return pd.concat(matches, ignore_index=True).drop_duplicates("variant_key")


def attach_abraom(clinvar: pd.DataFrame, abraom_matches: pd.DataFrame) -> pd.DataFrame:
    if abraom_matches.empty:
        enriched = clinvar.copy()
        enriched["abraom_present"] = False
        enriched["abraom_variant_id"] = pd.NA
        enriched["abraom_pos_0_based"] = pd.NA
        enriched["af_abraom"] = pd.NA
        enriched["af_gnomad"] = pd.NA
        enriched["specificity"] = pd.NA
    else:
        keep = [
            "variant_key",
            "abraom_variant_id",
            "abraom_pos_0_based",
            "af_abraom",
            "af_gnomad",
            "specificity",
        ]
        enriched = clinvar.merge(abraom_matches[keep], on="variant_key", how="left")
        enriched["abraom_present"] = enriched["abraom_variant_id"].notna()

    for column in ["abraom_variant_id", "abraom_pos_0_based"]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").astype("Int64")

    specificity = pd.to_numeric(enriched["specificity"], errors="coerce")
    enriched["specificity_bin"] = pd.cut(
        specificity,
        bins=SPECIFICITY_BINS,
        labels=SPECIFICITY_LABELS,
        include_lowest=True,
    ).astype("string")
    enriched["specificity_bin"] = enriched["specificity_bin"].fillna("absent")
    return enriched


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _split_counts(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, group in df.groupby("split_within_gene", sort=True):
        result[str(split)] = {
            "rows": len(group),
            "positives": int((group["label"] == 1).sum()),
            "negatives": int((group["label"] == 0).sum()),
            "snv": int(group["is_snv"].sum()),
            "abraom_present": int(group["abraom_present"].sum()),
            "brazilian_submitter": int(group["has_brazilian_submitter"].sum()),
        }
    return result


def build_summary(
    *,
    enriched: pd.DataFrame,
    regional_annotation: pd.DataFrame,
    abraom_matches: pd.DataFrame,
    inputs: dict[str, str],
    outputs: dict[str, str],
) -> dict[str, Any]:
    abraom_present = enriched.loc[enriched["abraom_present"]]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "rows": len(enriched),
        "unique_variant_keys": int(enriched["variant_key"].nunique()),
        "by_split": _split_counts(enriched),
        "label_counts": _value_counts(enriched["label"]),
        "variant_type_counts": _value_counts(enriched["variant_type"]),
        "regional_cohort_counts": _value_counts(enriched["clinvar_regional_cohort"]),
        "matched_regional_clinvar_benchmark": int(enriched["matched_regional_clinvar_benchmark"].sum()),
        "regional_annotation_unique_variant_keys": int(regional_annotation["variant_key"].nunique())
        if not regional_annotation.empty
        else 0,
        "abraom": {
            "matched_rows": int(enriched["abraom_present"].sum()),
            "matched_unique_variant_keys": (
                int(abraom_matches["variant_key"].nunique()) if not abraom_matches.empty else 0
            ),
            "matched_brazilian_submitter_rows": int(
                (enriched["abraom_present"] & enriched["has_brazilian_submitter"]).sum()
            ),
            "matched_non_brazilian_submitter_rows": int(
                (enriched["abraom_present"] & enriched["has_non_brazilian_submitter"]).sum()
            ),
            "mean_af_abraom": float(pd.to_numeric(abraom_present["af_abraom"], errors="coerce").mean())
            if not abraom_present.empty
            else None,
            "mean_af_gnomad": float(pd.to_numeric(abraom_present["af_gnomad"], errors="coerce").mean())
            if not abraom_present.empty
            else None,
            "mean_specificity": float(pd.to_numeric(abraom_present["specificity"], errors="coerce").mean())
            if not abraom_present.empty
            else None,
            "specificity_bin_counts": _value_counts(enriched["specificity_bin"]),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ClinVar Regional ABRAOM Dataset",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Outputs",
        "",
    ]
    for name, output_path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{output_path}`")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Rows: `{summary['rows']}`",
            f"- Unique variant keys: `{summary['unique_variant_keys']}`",
            f"- Regional benchmark matches: `{summary['matched_regional_clinvar_benchmark']}`",
            f"- ABRAOM matched rows: `{summary['abraom']['matched_rows']}`",
            "",
            "## By Split",
            "",
            "| Split | Rows | Positives | Negatives | SNV | ABRAOM | Brazilian Submitter |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, payload in summary["by_split"].items():
        lines.append(
            (
                "| {split} | {rows} | {positives} | {negatives} | {snv} | "
                "{abraom_present} | {brazilian_submitter} |"
            ).format(
                split=split,
                **payload,
            )
        )
    lines.extend(["", "## Regional Cohorts", ""])
    for key, value in summary["regional_cohort_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Specificity Bins", ""])
    for key, value in summary["abraom"]["specificity_bin_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def prepare_regional_clinvar_dataset(
    *,
    lumina_clinvar_dir: Path,
    regional_clinvar_path: Path,
    abraom_index: Path,
    output_dir: Path,
    splits: tuple[str, ...] = ("train", "test", "holdout"),
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    clinvar = load_lumina_clinvar_splits(lumina_clinvar_dir, splits)
    regional_annotation = build_regional_annotation(regional_clinvar_path)
    enriched = attach_regional_annotation(clinvar, regional_annotation)
    abraom_matches = build_abraom_matches(enriched, abraom_index)
    enriched = attach_abraom(enriched, abraom_matches)

    master_path = output_dir / "clinvar_regional_abraom_master.parquet"
    train_test_path = output_dir / "clinvar_regional_abraom_train_test.parquet"
    holdout_path = output_dir / "clinvar_regional_abraom_holdout.parquet"
    regional_annotation_path = output_dir / "regional_annotation_by_variant.parquet"
    abraom_matches_path = output_dir / "abraom_matches.parquet"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "README.md"

    enriched.to_parquet(master_path, index=False)
    enriched.loc[enriched["split_within_gene"].isin(["train", "test"])].to_parquet(train_test_path, index=False)
    enriched.loc[enriched["split_within_gene"] == "holdout"].to_parquet(holdout_path, index=False)
    regional_annotation.to_parquet(regional_annotation_path, index=False)
    abraom_matches.to_parquet(abraom_matches_path, index=False)

    outputs = {
        "master": str(master_path),
        "train_test": str(train_test_path),
        "holdout": str(holdout_path),
        "regional_annotation_by_variant": str(regional_annotation_path),
        "abraom_matches": str(abraom_matches_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }
    inputs = {
        "lumina_clinvar_dir": str(lumina_clinvar_dir),
        "regional_clinvar_path": str(regional_clinvar_path),
        "abraom_index": str(abraom_index),
    }
    summary = build_summary(
        enriched=enriched,
        regional_annotation=regional_annotation,
        abraom_matches=abraom_matches,
        inputs=inputs,
        outputs=outputs,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(report_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lumina-clinvar-dir", type=Path, default=DEFAULT_LUMINA_CLINVAR_DIR)
    parser.add_argument("--regional-clinvar", type=Path, default=DEFAULT_REGIONAL_CLINVAR)
    parser.add_argument("--abraom-index", type=Path, default=DEFAULT_ABRAOM_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "test", "holdout"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = prepare_regional_clinvar_dataset(
        lumina_clinvar_dir=args.lumina_clinvar_dir,
        regional_clinvar_path=args.regional_clinvar,
        abraom_index=args.abraom_index,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
