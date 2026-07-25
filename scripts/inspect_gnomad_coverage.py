#!/usr/bin/env python3
"""Is there unused gnomAD frequency for the 86% of Brazilian variants that currently have none?

Runs LOCAL on the notebook. No GPU, no job, no training.

THE QUESTION AND WHY IT DECIDES b-vs-c
--------------------------------------
Across every diagnostic, br_only performance turned out to be near a ceiling because 86% of
Brazilian variants carry NO allele frequency at all: `af_gnomad` is populated ONLY for variants
matched into the ABRAOM index (scripts/prepare_regional_clinvar_dataset.py:348-349), and the
ABRAOM index has a hard floor at AF ~= 0.5% in Brazil. So a variant absent from ABRAOM has no
gnomAD number in our pipeline -- even if it has one in real gnomAD.

The source ClinVar table is `.../clinvar/processed/eval_all_enriched/eval_unified.parquet`. The
loader (prepare_regional_clinvar_dataset.py:35-45) reads only 9 columns from it, but the file is
called "enriched" and may already carry a gnomAD frequency column we never used. If it does,
universal gnomAD coverage is sitting there for free, and lever (a) is worth pursuing (-> option c
or a coverage rebuild). If it does NOT -- or if the missing-frequency variants turn out to be
genuinely rare/absent in gnomAD too -- then frequency coverage cannot move br_only, and the
honest deliverable is the reframing (-> option b).

WHAT IT DOES
------------
[1] SCHEMA: prints every column of the enriched ClinVar parquet and flags anything that looks
    like an allele frequency (name contains af / gnomad / freq / faf / popmax / allele).

[2] If a frequency column is found, joins it onto the Brazilian slice variants by variant_key
    (rebuilt with the pipeline's own normalizers, so the key matches exactly) and reports:
      - coverage: of the variants that currently have NO af_gnomad, how many the enriched file
        could fill in;
      - among those newly-covered variants, the AF distribution -- are they actually common
        (frequency can discriminate) or rare (it cannot)?
      - the incremental value: AUROC on br_only using the enriched gnomAD AF vs the current
        pipeline coverage, tie-aware.

[3] VERDICT: whether unused, informative gnomAD coverage exists. Prints the recommendation
    (pursue coverage / reframe) with the numbers behind it.

USAGE
-----
    python scripts/inspect_gnomad_coverage.py \
        --clinvar-parquet ~/lumina-benchmarks/data/datasets/clinvar/processed/eval_all_enriched/eval_unified.parquet \
        --slice-dir ~/slices \
        --output-dir ~/v11eval/gnomad_coverage

If --clinvar-parquet is omitted the script tries a few standard locations and tells you what to
pass if none exist.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_regional_clinvar_dataset import (  # noqa: E402
    normalize_allele,
    normalize_chrom,
    variant_key,
)
from scripts.test_abraom_vs_gnomad_power import auroc  # noqa: E402 (tie-aware)

FREQ_HINT = re.compile(r"(af|gnomad|freq|faf|popmax|allele)", re.IGNORECASE)
# Columns that match the hint but are NOT population frequencies.
FREQ_EXCLUDE = re.compile(r"(alt|ref)$|allele[_-]?(vcf|id)|after|affected", re.IGNORECASE)

BRAZILIAN_SLICES = ["br_only", "br_any"]
DEFAULT_PARQUET_CANDIDATES = [
    "~/lumina-benchmarks/data/datasets/clinvar/processed/eval_all_enriched/eval_unified.parquet",
    "~/lumina/data/variants/clinvar/processed/eval_unified.parquet",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clinvar-parquet", type=Path, default=None,
                        help="the enriched ClinVar source; if omitted, standard locations are tried")
    parser.add_argument("--slice-dir", type=Path, default=home / "slices")
    parser.add_argument("--slices", nargs="*", default=BRAZILIAN_SLICES)
    parser.add_argument("--common-af", type=float, default=0.005,
                        help="AF at/above which frequency can actually discriminate (ABRAOM index floor)")
    parser.add_argument("--output-dir", type=Path, default=home / "v11eval/gnomad_coverage")
    return parser.parse_args(argv)


def resolve_parquet(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for candidate in DEFAULT_PARQUET_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path
    return None


def find_frequency_columns(columns: list[str]) -> list[str]:
    hits = []
    for column in columns:
        if FREQ_HINT.search(column) and not FREQ_EXCLUDE.search(column):
            hits.append(column)
    return hits


def build_key(frame: pd.DataFrame, chrom: str, pos: str, ref: str, alt: str) -> pd.Series:
    return variant_key(
        frame[chrom].map(normalize_chrom),
        pd.to_numeric(frame[pos], errors="coerce").astype("Int64"),
        frame[ref].map(normalize_allele),
        frame[alt].map(normalize_allele),
    )


def _num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def analyze_column(
    enriched: pd.DataFrame, freq_column: str, slice_dir: Path, dataset: str, common_af: float
) -> dict | None:
    slice_path = slice_dir / f"{dataset}.parquet"
    if not slice_path.is_file():
        return None
    sl = pd.read_parquet(slice_path)
    if "variant_key" not in sl.columns:
        sl["variant_key"] = build_key(sl, "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF")

    enriched_slim = enriched[["_variant_key", freq_column]].dropna(subset=["_variant_key"]).drop_duplicates("_variant_key")
    merged = sl.merge(enriched_slim, left_on="variant_key", right_on="_variant_key", how="left")

    current_af = _num(merged["af_gnomad"]) if "af_gnomad" in merged.columns else np.full(len(merged), np.nan)
    enriched_af = _num(merged[freq_column])
    labels = pd.to_numeric(merged["label"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)

    currently_missing = ~np.isfinite(current_af)
    newly_covered = currently_missing & np.isfinite(enriched_af)
    newly_common = newly_covered & (enriched_af >= common_af)

    result = {
        "dataset": dataset,
        "freq_column": freq_column,
        "n": int(len(merged)),
        "n_currently_missing_af": int(currently_missing.sum()),
        "n_newly_covered": int(newly_covered.sum()),
        "n_newly_common": int(newly_common.sum()),
        "pct_of_missing_now_covered": round(100.0 * newly_covered.sum() / max(currently_missing.sum(), 1), 1),
        "pct_of_newly_covered_that_are_common": round(100.0 * newly_common.sum() / max(newly_covered.sum(), 1), 1),
    }

    # incremental AUROC on the whole slice: current coverage (missing -> rare) vs enriched coverage
    def freq_score(af: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(af), af, 0.0)
        return 1.0 - filled

    combined_af = np.where(np.isfinite(current_af), current_af, enriched_af)
    valid = labels >= 0
    if np.unique(labels[valid]).size > 1:
        result["auroc_current_coverage"] = auroc(labels[valid], freq_score(current_af)[valid])
        result["auroc_enriched_coverage"] = auroc(labels[valid], freq_score(combined_af)[valid])
        result["auroc_gain"] = result["auroc_enriched_coverage"] - result["auroc_current_coverage"]
    return result


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 92)
    lines.append("UNUSED gnomAD COVERAGE? -- can we fill frequency for the 86% of Brazilian variants?")
    lines.append("=" * 92)

    lines.append("")
    lines.append(f"[1] Source: {payload.get('clinvar_parquet', 'NOT FOUND')}")
    if payload.get("error"):
        lines.append("")
        lines.append(f"    !! {payload['error']}")
        return "\n".join(lines)
    lines.append(f"    columns: {payload['n_columns']}")
    lines.append(f"    frequency-like columns found: {payload['frequency_columns'] or 'NONE'}")

    if payload.get("coverage"):
        lines.append("")
        lines.append("[2] Coverage the enriched file could add to the current pipeline")
        for col, rows in payload["coverage"].items():
            lines.append("")
            lines.append(f"    --- column: {col} ---")
            for row in rows:
                lines.append(
                    f"    {row['dataset']:<10} currently-missing AF: {row['n_currently_missing_af']:>4}/"
                    f"{row['n']:<4}  now covered: {row['n_newly_covered']:>4} "
                    f"({row['pct_of_missing_now_covered']}%)  of those COMMON: {row['n_newly_common']} "
                    f"({row['pct_of_newly_covered_that_are_common']}%)"
                )
                if "auroc_gain" in row:
                    lines.append(
                        f"    {'':<10} AUROC(freq) current {_fmt(row['auroc_current_coverage'])} -> "
                        f"enriched {_fmt(row['auroc_enriched_coverage'])}  "
                        f"gain {_fmt(row['auroc_gain'])}"
                    )

    verdict = payload.get("verdict", {})
    if verdict:
        lines.append("")
        lines.append("=" * 92)
        lines.append(f"VERDICT: {verdict.get('headline', '')}")
        for line in verdict.get("detail", []):
            lines.append(f"    {line}")
        lines.append("=" * 92)
    return "\n".join(lines)


def decide(payload: dict, common_af: float) -> dict:
    freq_cols = payload.get("frequency_columns", [])
    if not freq_cols:
        return {
            "headline": "NO frequency column in the enriched source -> lever (a) is dead. Recommend (b) reframe.",
            "detail": ["The enriched ClinVar parquet carries no gnomAD/AF column, so there is no unused",
                       "coverage to add. Frequency for the missing 86% would require joining an external",
                       "gnomAD release, which is a data-acquisition task, not a quick win."],
        }
    gains = [row.get("auroc_gain", float("nan")) for rows in payload.get("coverage", {}).values() for row in rows]
    commons = [row.get("pct_of_newly_covered_that_are_common", 0.0)
               for rows in payload.get("coverage", {}).values() for row in rows]
    max_gain = max([g for g in gains if np.isfinite(g)], default=float("nan"))
    max_common = max(commons, default=0.0)

    if np.isfinite(max_gain) and max_gain >= 0.02 and max_common >= 10.0:
        return {
            "headline": "Unused, INFORMATIVE gnomAD coverage EXISTS -> lever (a) is real. Recommend (c).",
            "detail": [f"Best AUROC gain from filling coverage: {max_gain:.3f}; up to {max_common:.0f}% of",
                       "newly-covered variants are common enough for frequency to discriminate.",
                       "Worth an M2/coverage run to convert this into br_only under the recall constraint."],
        }
    return {
        "headline": "A frequency column exists but adds little -> coverage is not the lever. Recommend (b).",
        "detail": [f"Best AUROC gain from filling coverage: {_fmt(max_gain)}; only {max_common:.0f}% of newly-",
                   f"covered variants reach AF >= {common_af}. The missing variants are missing because they",
                   "are rare, and frequency cannot discriminate among rare variants. This confirms the",
                   "ceiling is a data/label problem, not a coverage problem."],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    parquet = resolve_parquet(args.clinvar_parquet)

    payload: dict = {"generated_at": datetime.now(UTC).isoformat(), "common_af": args.common_af}

    if parquet is None:
        payload["error"] = ("enriched ClinVar parquet not found. Pass --clinvar-parquet <path>. Tried: "
                            + ", ".join(DEFAULT_PARQUET_CANDIDATES))
        report = render(payload)
        print(report)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "gnomad_coverage.txt").write_text(report, encoding="utf-8")
        return 1

    payload["clinvar_parquet"] = str(parquet)
    import pyarrow.parquet as pq
    schema_names = list(pq.ParquetFile(parquet).schema_arrow.names)
    payload["n_columns"] = len(schema_names)
    payload["all_columns"] = schema_names
    freq_columns = find_frequency_columns(schema_names)
    payload["frequency_columns"] = freq_columns

    if freq_columns:
        # need coords to rebuild the key + the freq columns themselves
        coord_columns = [c for c in ("chrom", "pos", "ref", "alt") if c in schema_names]
        enriched = pd.read_parquet(parquet, columns=list(dict.fromkeys(coord_columns + freq_columns)))
        if len(coord_columns) == 4:
            enriched["_variant_key"] = build_key(enriched, "chrom", "pos", "ref", "alt")
        else:
            enriched["_variant_key"] = pd.NA
            payload["warning"] = f"coordinate columns incomplete ({coord_columns}); cannot join by variant_key"

        coverage: dict[str, list] = {}
        for freq_column in freq_columns:
            rows = []
            for dataset in args.slices:
                row = analyze_column(enriched, freq_column, args.slice_dir, dataset, args.common_af)
                if row:
                    rows.append(row)
            if rows:
                coverage[freq_column] = rows
        payload["coverage"] = coverage

    payload["verdict"] = decide(payload, args.common_af)

    report = render(payload)
    print(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gnomad_coverage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "gnomad_coverage.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/gnomad_coverage.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
