#!/usr/bin/env python3
"""Build regional ClinVar evaluation slices from the local ClinVar x ABRAOM table."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_INPUT = Path("data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet")
DEFAULT_OUTPUT_DIR = Path("data/datasets/clinvar/regional_abraom/slices")

READ_COLUMNS = [
    "Chromosome",
    "Start",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "label",
    "split_within_gene",
    "source_variant_id",
    "GeneSymbol",
    "variant_type",
    "is_snv",
    "variant_key",
    "clinvar_regional_cohort",
    "has_brazilian_submitter",
    "has_non_brazilian_submitter",
    "brazilian_submission_rows",
    "non_brazilian_submission_rows",
    "regional_submission_rows",
    "matched_regional_clinvar_benchmark",
    "abraom_present",
    "abraom_variant_id",
    "af_abraom",
    "af_gnomad",
    "specificity",
    "specificity_bin",
]


def _require_columns(path: Path, required: list[str]) -> None:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _safe_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _slice_definitions(
    *,
    high_specificity_threshold: float,
    common_af_threshold: float,
) -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (
            "br_only",
            (
                "ClinVar rows with Brazilian submitter evidence and no non-Brazilian submitter "
                "evidence in the regional table."
            ),
            lambda df: _safe_bool(df["has_brazilian_submitter"]) & ~_safe_bool(df["has_non_brazilian_submitter"]),
        ),
        (
            "nonbr_only",
            (
                "ClinVar rows with non-Brazilian submitter evidence and no Brazilian submitter "
                "evidence in the regional table."
            ),
            lambda df: _safe_bool(df["has_non_brazilian_submitter"]) & ~_safe_bool(df["has_brazilian_submitter"]),
        ),
        (
            "mixed_br_nonbr",
            "ClinVar rows with both Brazilian and non-Brazilian submitter evidence in the regional table.",
            lambda df: _safe_bool(df["has_brazilian_submitter"]) & _safe_bool(df["has_non_brazilian_submitter"]),
        ),
        (
            "br_any",
            "ClinVar rows with any Brazilian submitter evidence in the regional table.",
            lambda df: _safe_bool(df["has_brazilian_submitter"]),
        ),
        (
            "regional_benchmark_any",
            "ClinVar rows matched to the enriched regional benchmark table.",
            lambda df: _safe_bool(df["matched_regional_clinvar_benchmark"]),
        ),
        (
            "abraom_present",
            "ClinVar SNVs present in the filtered ABRAOM v2 index.",
            lambda df: _safe_bool(df["abraom_present"]),
        ),
        (
            "abraom_high_specificity",
            f"ClinVar rows present in ABRAOM with specificity >= {high_specificity_threshold}.",
            lambda df: _safe_bool(df["abraom_present"])
            & (pd.to_numeric(df["specificity"], errors="coerce") >= high_specificity_threshold),
        ),
        (
            "abraom_common",
            f"ClinVar rows present in ABRAOM with AF_ABRAOM >= {common_af_threshold}.",
            lambda df: _safe_bool(df["abraom_present"])
            & (pd.to_numeric(df["af_abraom"], errors="coerce") >= common_af_threshold),
        ),
        (
            "abraom_common_benign",
            f"Benign/likely-benign ClinVar rows present in ABRAOM with AF_ABRAOM >= {common_af_threshold}.",
            lambda df: (pd.to_numeric(df["label"], errors="coerce") == 0)
            & _safe_bool(df["abraom_present"])
            & (pd.to_numeric(df["af_abraom"], errors="coerce") >= common_af_threshold),
        ),
        (
            "abraom_pathogenic_present",
            "Pathogenic/likely-pathogenic ClinVar rows present in ABRAOM; this is a do-not-suppress check.",
            lambda df: (pd.to_numeric(df["label"], errors="coerce") == 1) & _safe_bool(df["abraom_present"]),
        ),
        (
            "abraom_pathogenic_common",
            (
                "Pathogenic/likely-pathogenic ClinVar rows present in ABRAOM with "
                f"AF_ABRAOM >= {common_af_threshold}; this is a high-risk do-not-suppress check."
            ),
            lambda df: (pd.to_numeric(df["label"], errors="coerce") == 1)
            & _safe_bool(df["abraom_present"])
            & (pd.to_numeric(df["af_abraom"], errors="coerce") >= common_af_threshold),
        ),
        (
            "global_nonbr_no_abraom",
            "Non-Brazilian-only ClinVar rows not present in the filtered ABRAOM v2 index.",
            lambda df: _safe_bool(df["has_non_brazilian_submitter"])
            & ~_safe_bool(df["has_brazilian_submitter"])
            & ~_safe_bool(df["abraom_present"]),
        ),
    ]


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _summarize_slice(name: str, description: str, df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    label = pd.to_numeric(df["label"], errors="coerce")
    specificity = pd.to_numeric(df["specificity"], errors="coerce")
    af_abraom = pd.to_numeric(df["af_abraom"], errors="coerce")
    return {
        "slice": name,
        "description": description,
        "path": str(output_path),
        "rows": len(df),
        "unique_variant_keys": int(df["variant_key"].nunique()) if "variant_key" in df else len(df),
        "positives": int((label == 1).sum()),
        "negatives": int((label == 0).sum()),
        "by_split": _value_counts(df["split_within_gene"]),
        "variant_type_counts": _value_counts(df["variant_type"]),
        "abraom_present": int(_safe_bool(df["abraom_present"]).sum()),
        "mean_af_abraom": float(af_abraom.mean()) if af_abraom.notna().any() else None,
        "mean_specificity": float(specificity.mean()) if specificity.notna().any() else None,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Regional ClinVar Evaluation Slices",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Parameters",
        "",
        f"- High specificity threshold: `{summary['parameters']['high_specificity_threshold']}`",
        f"- Common AF threshold: `{summary['parameters']['common_af_threshold']}`",
        "",
        "## Slices",
        "",
        "| Slice | Rows | Positives | Negatives | ABRAOM present | Mean AF ABRAOM | Mean specificity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in summary["slices"]:
        mean_af = payload["mean_af_abraom"]
        mean_spec = payload["mean_specificity"]
        lines.append(
            "| {slice} | {rows} | {positives} | {negatives} | {abraom_present} | {mean_af} | {mean_spec} |".format(
                slice=payload["slice"],
                rows=payload["rows"],
                positives=payload["positives"],
                negatives=payload["negatives"],
                abraom_present=payload["abraom_present"],
                mean_af=f"{mean_af:.6f}" if mean_af is not None else "NA",
                mean_spec=f"{mean_spec:.6f}" if mean_spec is not None else "NA",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def build_regional_clinvar_eval_slices(
    *,
    input_path: Path,
    output_dir: Path,
    high_specificity_threshold: float = 0.05,
    common_af_threshold: float = 0.01,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    _require_columns(input_path, READ_COLUMNS)
    data = pd.read_parquet(input_path, columns=READ_COLUMNS)

    slice_payloads: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for name, description, mask_fn in _slice_definitions(
        high_specificity_threshold=high_specificity_threshold,
        common_af_threshold=common_af_threshold,
    ):
        mask = mask_fn(data).fillna(False).astype(bool)
        sliced = data.loc[mask].reset_index(drop=True)
        output_path = output_dir / f"{name}.parquet"
        sliced.to_parquet(output_path, index=False)
        payload = _summarize_slice(name, description, sliced, output_path)
        slice_payloads.append(payload)
        manifest_rows.append(
            {
                "slice": name,
                "description": description,
                "path": str(output_path),
                "rows": payload["rows"],
                "positives": payload["positives"],
                "negatives": payload["negatives"],
                "abraom_present": payload["abraom_present"],
                "mean_af_abraom": payload["mean_af_abraom"],
                "mean_specificity": payload["mean_specificity"],
            }
        )

    manifest_path = output_dir / "slice_manifest.parquet"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "README.md"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "outputs": {
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "parameters": {
            "high_specificity_threshold": high_specificity_threshold,
            "common_af_threshold": common_af_threshold,
        },
        "source_rows": len(data),
        "slices": slice_payloads,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_report(report_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--high-specificity-threshold", type=float, default=0.05)
    parser.add_argument("--common-af-threshold", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_regional_clinvar_eval_slices(
        input_path=args.input,
        output_dir=args.output_dir,
        high_specificity_threshold=args.high_specificity_threshold,
        common_af_threshold=args.common_af_threshold,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
