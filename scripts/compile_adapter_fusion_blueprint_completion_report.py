#!/usr/bin/env python3
"""Compile a scientific completion report for the ABRAOM adapter-fusion blueprint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = Path("artifacts/adapter_fusion_blueprint_completion")
DEFAULT_BASELINE_CSV = Path(
    "artifacts/clinvar_regional_comparison/"
    "m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_regional_test_summary.csv"
)
DEFAULT_ABRAOM_ALIGNMENT = Path(
    "artifacts/abraom_frequency_adapter/alignment_comparison/ABRAOM_FREQUENCY_ALIGNMENT_REPORT.md"
)
DEFAULT_M5V2_FINAL = Path(
    "artifacts/clinvar_regional_eval/"
    "m5_v2_calibrated_holdout_tuned/final_analysis/M5_V2_CALIBRATED_FINAL_ANALYSIS.md"
)
DEFAULT_SENTINEL_MANIFEST = Path(
    "artifacts/clinvar_regional_eval/brazilian_plp_sentinel_panel/manifest.json"
)

REQUIRED_DATASETS = {
    "br_only",
    "br_any",
    "regional_benchmark_any",
    "abraom_common_benign",
    "abraom_pathogenic_present",
    "abraom_pathogenic_common",
    "global_nonbr_no_abraom",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--abraom-alignment-report", type=Path, default=DEFAULT_ABRAOM_ALIGNMENT)
    parser.add_argument("--m5-v2-final-report", type=Path, default=DEFAULT_M5V2_FINAL)
    parser.add_argument("--sentinel-manifest", type=Path, default=DEFAULT_SENTINEL_MANIFEST)
    parser.add_argument("--dynamic-summary-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def status(done: bool, note: str) -> dict[str, Any]:
    return {"status": "complete" if done else "pending", "note": note}


def read_models_and_datasets(path: Path) -> tuple[set[str], set[str], pd.DataFrame | None]:
    if not path.is_file():
        return set(), set(), None
    frame = pd.read_csv(path)
    models = set(frame["model"].astype(str)) if "model" in frame else set()
    datasets = set(frame["dataset"].astype(str)) if "dataset" in frame else set()
    return models, datasets, frame


def metric(frame: pd.DataFrame | None, model: str, dataset: str, field: str) -> float | None:
    if frame is None:
        return None
    match = frame.loc[(frame["model"].astype(str) == model) & (frame["dataset"].astype(str) == dataset)]
    if match.empty or field not in match:
        return None
    value = pd.to_numeric(match.iloc[0][field], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    baseline_models, baseline_datasets, baseline = read_models_and_datasets(args.baseline_csv)
    if args.dynamic_summary_csv:
        dynamic_models, _dynamic_datasets, dynamic = read_models_and_datasets(args.dynamic_summary_csv)
    else:
        dynamic_models, dynamic = set(), None

    sentinel: dict[str, Any] = {}
    if args.sentinel_manifest.is_file():
        sentinel = json.loads(args.sentinel_manifest.read_text(encoding="utf-8"))

    checklist = {
        "data_manifest_and_slices": status(
            REQUIRED_DATASETS.issubset(baseline_datasets),
            f"Baseline datasets present: {sorted(baseline_datasets)}",
        ),
        "abraom_frequency_alignment": status(
            args.abraom_alignment_report.is_file(),
            str(args.abraom_alignment_report),
        ),
        "baseline_ablation_table": status(
            {"M0", "M4", "M5", "M6", "M7_scrambled"}.issubset(baseline_models),
            f"Baseline models present: {sorted(baseline_models)}",
        ),
        "m5_v2_calibrated_final_analysis": status(
            args.m5_v2_final_report.is_file(),
            str(args.m5_v2_final_report),
        ),
        "sentinel_panel": status(
            bool(sentinel) and sentinel.get("rows", {}).get("sentinel_total", 0) > 0,
            str(args.sentinel_manifest),
        ),
        "dynamic_gate_experiments": status(
            {"M2_gnomad_only", "M4_dynamic_gated", "M5_dynamic_gated", "M7_dynamic_scrambled"}.issubset(dynamic_models),
            f"Dynamic models present: {sorted(dynamic_models)}",
        ),
        "dynamic_gate_diagnostics": status(
            dynamic is not None and {"gate_alpha_abraom", "gate_entropy"}.intersection(dynamic.columns),
            "Requires dynamic prediction summaries with alpha/gate columns.",
        ),
    }

    key_metrics = {
        "M0_br_only_mcc": metric(baseline, "M0", "br_only", "mcc"),
        "M5_v2_calibrated_br_only_mcc": metric(baseline, "M5_v2_calibrated", "br_only", "mcc"),
        "M0_abraom_common_benign_specificity": metric(baseline, "M0", "abraom_common_benign", "specificity"),
        "M5_v2_calibrated_abraom_common_benign_specificity": metric(
            baseline,
            "M5_v2_calibrated",
            "abraom_common_benign",
            "specificity",
        ),
        "M0_abraom_pathogenic_present_recall": metric(baseline, "M0", "abraom_pathogenic_present", "recall"),
        "M5_v2_calibrated_abraom_pathogenic_present_recall": metric(
            baseline,
            "M5_v2_calibrated",
            "abraom_pathogenic_present",
            "recall",
        ),
    }

    complete = all(item["status"] == "complete" for item in checklist.values())
    decision = "supported" if complete else "needs_more_validation"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": decision,
        "checklist": checklist,
        "key_metrics": key_metrics,
        "sentinel_rows": sentinel.get("rows", {}),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ABRAOM Adapter-Fusion Blueprint Completion Report",
        "",
        f"Generated at UTC: `{report['generated_at_utc']}`",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Checklist",
        "",
        "| Item | Status | Note |",
        "|---|---|---|",
    ]
    for name, item in report["checklist"].items():
        lines.append(f"| `{name}` | `{item['status']}` | {item['note']} |")
    lines.extend(["", "## Key Metrics", "", "| Metric | Value |", "|---|---:|"])
    for name, value in report["key_metrics"].items():
        rendered = "NA" if value is None else f"{value:.3f}"
        lines.append(f"| `{name}` | {rendered} |")
    lines.extend(["", "## Interpretation", ""])
    if report["decision"] == "supported":
        lines.append("All blueprint research gates are represented by artifacts.")
    else:
        lines.append(
            "The current package remains research-only and needs the pending checklist items before the "
            "blueprint is complete."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Pass --overwrite to replace outputs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    (args.output_dir / "completion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "ABRAOM_ADAPTER_FUSION_BLUEPRINT_COMPLETION_REPORT.md", report)
    print(json.dumps({"decision": report["decision"], "checklist": report["checklist"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
