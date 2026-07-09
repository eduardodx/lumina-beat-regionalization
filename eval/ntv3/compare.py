from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["species"], row["datasets"], row["assay_type"])


def _row_metric(row: dict[str, str]) -> float:
    return float(row["Metric"])


def _mean_metric(rows: list[dict[str, str]]) -> float:
    if not rows:
        return float("nan")
    return fmean(_row_metric(row) for row in rows)


def _rows_by_subset(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    subsets: dict[str, list[dict[str, str]]] = {
        "overall": rows,
        "functional": [row for row in rows if row["assay_type"] != "Annotation"],
        "annotation": [row for row in rows if row["assay_type"] == "Annotation"],
    }
    for species_name in sorted({row["species"] for row in rows}):
        subsets[f"species:{species_name}"] = [row for row in rows if row["species"] == species_name]
    return subsets


def _rank_models(public_rows: list[dict[str, str]], subset_keys: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in public_rows:
        if _row_key(row) in subset_keys:
            rows_by_model[row["model_name"]].append(row)

    ranked: list[dict[str, Any]] = []
    for model_name, rows in rows_by_model.items():
        ranked.append(
            {
                "model_name": model_name,
                "num_rows": len(rows),
                "mean_metric": _mean_metric(rows),
            }
        )
    ranked.sort(key=lambda item: item["mean_metric"], reverse=True)
    return ranked


def build_comparison(public_csv: Path, our_csv: Path, *, top_k: int) -> dict[str, Any]:
    public_rows = _load_rows(public_csv)
    our_rows = _load_rows(our_csv)
    if not our_rows:
        raise ValueError(f"Our CSV has no rows: {our_csv}")

    our_subsets = _rows_by_subset(our_rows)
    public_subsets = _rows_by_subset(public_rows)

    summary: dict[str, Any] = {
        "public_csv": str(public_csv),
        "our_csv": str(our_csv),
        "our_model_name": our_rows[0]["model_name"],
        "our_run_id": our_rows[0]["run_id"],
        "num_public_rows": len(public_rows),
        "num_our_rows": len(our_rows),
        "subsets": {},
    }

    for subset_name, subset_rows in our_subsets.items():
        subset_keys = {_row_key(row) for row in subset_rows}
        ranked_public = _rank_models(public_rows, subset_keys)
        our_mean = _mean_metric(subset_rows)

        our_rank = 1
        for item in ranked_public:
            if item["mean_metric"] > our_mean:
                our_rank += 1

        summary["subsets"][subset_name] = {
            "num_rows": len(subset_rows),
            "our_mean_metric": our_mean,
            "our_hypothetical_rank": our_rank,
            "public_top_k": ranked_public[:top_k],
            "public_num_models": len(ranked_public),
            "public_reference_mean": _mean_metric(public_subsets.get(subset_name, [])),
        }

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a local NTv3 benchmark CSV against the public leaderboard CSV."
    )
    parser.add_argument("--public-csv", required=True)
    parser.add_argument("--our-csv", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = build_comparison(
        Path(args.public_csv).expanduser().resolve(),
        Path(args.our_csv).expanduser().resolve(),
        top_k=args.top_k,
    )
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
