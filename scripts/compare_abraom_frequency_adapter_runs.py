#!/usr/bin/env python3
"""Compare ABRAOM frequency-adapter run summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

OVERALL_KEYS = (
    "model_n",
    "model_nll",
    "gnomad_nll",
    "delta_nll_model_minus_gnomad",
    "model_brier",
    "gnomad_brier",
    "delta_brier_model_minus_gnomad",
    "model_mae",
    "gnomad_mae",
    "model_spearman",
    "gnomad_spearman",
    "model_pearson",
    "gnomad_pearson",
    "model_mean_pred",
    "gnomad_mean_pred",
    "model_mean_target",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as label=/path/to/run_dir_or_summary.json. Repeat for each run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _summary_path(path: Path) -> Path:
    return path if path.is_file() else path / "summary.json"


def _load_run(spec: str) -> tuple[str, dict[str, Any]]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be label=path, got {spec!r}.")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Run label is empty in spec {spec!r}.")
    path = _summary_path(Path(raw_path).expanduser())
    if not path.is_file():
        raise FileNotFoundError(f"summary.json not found for {label}: {path}")
    return label, json.loads(path.read_text())


def _row_for_split(label: str, summary: dict[str, Any], split: str) -> dict[str, Any]:
    metrics = summary.get(f"{split}_metrics")
    if not metrics:
        raise ValueError(f"{label} does not contain {split}_metrics.")
    config = summary["config"]
    overall = metrics["overall"]
    row: dict[str, Any] = {
        "run": label,
        "split": split,
        "target_column": config.get("target_column"),
        "metric_target_column": config.get("metric_target_column"),
        "row_sample_strategy": config.get("row_sample_strategy"),
        "max_train_rows": config.get("max_train_rows"),
        "max_steps": config.get("max_steps"),
        "best_step": summary.get("best_step"),
        "rows_loaded_train": summary.get("rows_loaded", {}).get("train"),
        "rows_loaded_eval": metrics.get("rows_loaded"),
        "rows_scored": metrics.get("rows_scored"),
    }
    for key in OVERALL_KEYS:
        row[key] = overall.get(key)
    return row


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "run",
        "split",
        "target_column",
        "rows_scored",
        "best_step",
        "model_nll",
        "gnomad_nll",
        "delta_nll_model_minus_gnomad",
        "model_brier",
        "gnomad_brier",
        "model_spearman",
        "gnomad_spearman",
    ]
    lines = [
        "# ABRAOM Frequency Adapter Comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in args.run:
        label, summary = _load_run(spec)
        rows.append(_row_for_split(label, summary, "val"))
        if summary.get("test_metrics"):
            rows.append(_row_for_split(label, summary, "test"))

    csv_path = args.output_dir / "abraom_frequency_adapter_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.output_dir / "abraom_frequency_adapter_comparison.md")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
