#!/usr/bin/env python3
"""Compile a reproducible AlphaGenome/Lumina/NTv3 comparison packet.

The script does not run AlphaGenome, Lumina, or NTv3. It reads completed
artifacts, assigns each score to an explicit experimental regime, and writes a
small comparison bundle suitable for review and manuscript drafting.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_TRACK_ID = "ENCSR814RGG"
DEFAULT_ZERO_SHOT_METRICS = Path("artifacts/analysis/alphagenome_ntv3_track_allfolds_full/metrics.json")
DEFAULT_READOUT_METRICS = Path(
    "artifacts/analysis/alphagenome_ntv3_readout_allfolds_2048_512_fulltest/metrics.json"
)
DEFAULT_READOUT_WEIGHTS = Path(
    "artifacts/analysis/alphagenome_ntv3_readout_allfolds_2048_512_fulltest/readout_weights.json"
)
DEFAULT_ALPHA_TRACK_METADATA = Path(
    "artifacts/analysis/alphagenome_ntv3_track_allfolds_full/alphagenome_track_metadata.csv"
)
DEFAULT_HONEST_DELTAS = Path(
    "artifacts/analysis/ntv3_recent/best_vs_ntv3_8m_1page_honest/"
    "lumina_best_vs_ntv3_8m_honest_track_deltas.csv"
)
DEFAULT_PUBLIC_RESULTS = Path("artifacts/analysis/ntv3_recent/public_ntv3_space/ntv3_benchmark_results.csv")
DEFAULT_PUBLIC_BEST_SLICE = Path("artifacts/analysis/ntv3_recent/ntv3_latest_track_slice_analysis.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/alphagenome_ntv3_comparison")


@dataclass(frozen=True)
class ComparisonRow:
    track_id: str
    assay_type: str
    biosample_or_ontology: str
    model: str
    training_regime: str
    adaptation: str
    input_len: str
    train_windows: str
    val_windows: str
    test_windows: str
    test_positions: str
    pearson: float
    pearson_std: str
    n_runs: str
    metric_space: str
    source_artifact: str
    notes: str


@dataclass(frozen=True)
class TrackAuditRow:
    track_id: str
    ntv3_assay_type: str
    ntv3_track_name_clean: str
    alphagenome_output_type: str
    alphagenome_ontology_curie: str
    alphagenome_biosample_name: str
    alphagenome_assay_title: str
    overlap_class: str
    rationale: str
    metadata_source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile AlphaGenome/Lumina/NTv3 comparison tables and report.",
    )
    parser.add_argument("--track-id", default=DEFAULT_TRACK_ID)
    parser.add_argument("--zero-shot-metrics", type=Path, default=DEFAULT_ZERO_SHOT_METRICS)
    parser.add_argument("--readout-metrics", type=Path, default=DEFAULT_READOUT_METRICS)
    parser.add_argument("--readout-weights", type=Path, default=DEFAULT_READOUT_WEIGHTS)
    parser.add_argument("--alpha-track-metadata", type=Path, default=DEFAULT_ALPHA_TRACK_METADATA)
    parser.add_argument("--honest-deltas", type=Path, default=DEFAULT_HONEST_DELTAS)
    parser.add_argument("--public-results", type=Path, default=DEFAULT_PUBLIC_RESULTS)
    parser.add_argument("--public-best-slice", type=Path, default=DEFAULT_PUBLIC_BEST_SLICE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_float(value: str | int | float | None) -> float:
    if value is None:
        raise ValueError("Missing numeric value.")
    return float(value)


def optional_float(value: str | int | float | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def format_optional_int(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(int(value))


def find_track_row(rows: list[dict[str, str]], track_id: str, *, column: str) -> dict[str, str] | None:
    for row in rows:
        if row.get(column) == track_id:
            return row
    return None


def public_model_summary(
    rows: list[dict[str, str]],
    *,
    track_id: str,
    model_name: str,
) -> dict[str, Any] | None:
    scores: list[float] = []
    run_ids: list[str] = []
    assay_type = ""
    track_name_clean = ""
    for row in rows:
        if row.get("species") != "human":
            continue
        if row.get("datasets") != track_id:
            continue
        if row.get("model_name") != model_name:
            continue
        scores.append(to_float(row.get("Metric")))
        if row.get("run_id"):
            run_ids.append(row["run_id"])
        assay_type = row.get("assay_type", assay_type)
        track_name_clean = row.get("track_name_clean", track_name_clean)
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return {
        "model_name": model_name,
        "mean": mean,
        "std": math.sqrt(variance),
        "n_runs": len(scores),
        "run_ids": run_ids,
        "assay_type": assay_type,
        "track_name_clean": track_name_clean,
    }


def public_track_label(rows: list[dict[str, str]], *, track_id: str) -> str:
    for row in rows:
        if row.get("species") == "human" and row.get("datasets") == track_id:
            return row.get("track_name_clean", "")
    return ""


def alpha_metadata_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one AlphaGenome metadata row in {path}; found {len(rows)}.")
    return rows[0]


def selected_readout_metric(metrics: dict[str, Any]) -> tuple[float, str]:
    selected_lambda = metrics.get("selected_lambda")
    if selected_lambda is None:
        raise ValueError("Readout metrics must contain selected_lambda.")
    test_metrics = metrics.get("test_metrics")
    if not isinstance(test_metrics, dict):
        raise ValueError("Readout metrics must contain a test_metrics object.")
    key = f"readout_pearson_ntv3_scaled_lambda_{float(selected_lambda):g}"
    if key not in test_metrics:
        raise KeyError(f"Missing selected readout metric {key!r}.")
    return to_float(test_metrics[key]), key


def build_track_audit(
    *,
    track_id: str,
    zero_metrics: dict[str, Any],
    public_best_row: dict[str, str] | None,
    ntv3_track_name_clean: str,
    alpha_metadata: dict[str, str],
    alpha_metadata_path: Path,
) -> TrackAuditRow:
    config = zero_metrics.get("config")
    if not isinstance(config, dict):
        raise ValueError("Zero-shot metrics must contain a config object.")
    track_info = config.get("track_info")
    if not isinstance(track_info, dict):
        raise ValueError("Zero-shot metrics config must contain track_info.")

    ntv3_assay = str(track_info.get("assay") or (public_best_row or {}).get("assay_type") or "")
    ntv3_track_name = str(ntv3_track_name_clean or (public_best_row or {}).get("track_name_clean") or track_id)
    alpha_assay = alpha_metadata.get("Assay title", "")
    alpha_biosample = alpha_metadata.get("biosample_name", "")
    alpha_ontology = alpha_metadata.get("ontology_curie", "")
    output_type = str(config.get("alphagenome_output_type") or "")

    assay_match = ntv3_assay.lower().replace("-seq", "").startswith(alpha_assay.lower().replace("-seq", ""))
    tissue_match = alpha_biosample and alpha_biosample.lower() in ntv3_track_name.lower()
    if assay_match and tissue_match:
        overlap_class = "exact_assay_tissue_match"
        rationale = "AlphaGenome metadata matches the NTv3 assay and the NTv3 public track label tissue."
    elif assay_match:
        overlap_class = "assay_match_tissue_uncertain"
        rationale = "Assay matches, but the tissue/ontology match needs manual confirmation."
    else:
        overlap_class = "manual_review_required"
        rationale = "The AlphaGenome output metadata does not clearly match the NTv3 assay/tissue label."

    return TrackAuditRow(
        track_id=track_id,
        ntv3_assay_type=ntv3_assay,
        ntv3_track_name_clean=ntv3_track_name,
        alphagenome_output_type=output_type,
        alphagenome_ontology_curie=alpha_ontology,
        alphagenome_biosample_name=alpha_biosample,
        alphagenome_assay_title=alpha_assay,
        overlap_class=overlap_class,
        rationale=rationale,
        metadata_source=str(alpha_metadata_path),
    )


def build_comparison_rows(
    *,
    track_id: str,
    zero_metrics: dict[str, Any],
    readout_metrics: dict[str, Any],
    readout_weights: dict[str, Any],
    honest_row: dict[str, str] | None,
    public_summaries: list[dict[str, Any]],
    public_best_row: dict[str, str] | None,
    zero_path: Path,
    readout_path: Path,
    honest_path: Path,
    public_path: Path,
) -> list[ComparisonRow]:
    zero_config = zero_metrics["config"]
    readout_config = readout_metrics["config"]
    track_info = zero_config["track_info"]
    assay_type = str(track_info.get("assay") or (honest_row or {}).get("assay_type") or "")
    biosample = str(zero_config.get("ontology_term") or "")
    zero_metric = to_float(zero_metrics["alpha_vs_ntv3_pearson_ntv3_scaled_both"])
    selected_readout, readout_key = selected_readout_metric(readout_metrics)
    split_selected = readout_config.get("split_selected_windows")
    if not isinstance(split_selected, dict):
        raise ValueError("Readout config must contain split_selected_windows.")
    test_metrics = readout_metrics.get("test_metrics")
    if not isinstance(test_metrics, dict):
        raise ValueError("Readout metrics must contain test_metrics.")

    feature_names = readout_weights.get("feature_names")
    if not isinstance(feature_names, list):
        feature_names = []

    rows = [
        ComparisonRow(
            track_id=track_id,
            assay_type=assay_type,
            biosample_or_ontology=biosample,
            model="AlphaGenome",
            training_regime="native supervised genome-track model",
            adaptation="zero-shot native ATAC output, no NTv3 fitting",
            input_len=format_optional_int(zero_config.get("sequence_length")),
            train_windows="0",
            val_windows="0",
            test_windows=format_optional_int(zero_metrics.get("evaluated_windows")),
            test_positions=format_optional_int(zero_metrics.get("num_positions")),
            pearson=zero_metric,
            pearson_std="",
            n_runs="1",
            metric_space="NTv3 transformed target and prediction, Pearson",
            source_artifact=str(zero_path),
            notes="Contextual baseline; not equivalent to NTv3/Lumina training regime.",
        ),
        ComparisonRow(
            track_id=track_id,
            assay_type=assay_type,
            biosample_or_ontology=biosample,
            model="AlphaGenome",
            training_regime="frozen AlphaGenome plus NTv3 readout",
            adaptation=(
                f"ridge readout selected on val ({readout_key}); "
                f"features={','.join(str(name) for name in feature_names)}"
            ),
            input_len=format_optional_int(readout_config.get("sequence_length")),
            train_windows=format_optional_int(split_selected.get("train")),
            val_windows=format_optional_int(split_selected.get("val")),
            test_windows=format_optional_int(test_metrics.get("evaluated_windows")),
            test_positions=format_optional_int(test_metrics.get("num_positions")),
            pearson=selected_readout,
            pearson_std="",
            n_runs="1",
            metric_space="NTv3 transformed target, readout prediction, Pearson",
            source_artifact=str(readout_path),
            notes="Head-only calibration evidence; backbone remains frozen.",
        ),
    ]

    if honest_row is not None:
        rows.append(
            ComparisonRow(
                track_id=track_id,
                assay_type=honest_row.get("assay_type", assay_type),
                biosample_or_ontology="NTv3 benchmark track",
                model="Lumina beat-v7 context-pyramid",
                training_regime="Lumina NTv3 benchmark fine-tuning",
                adaptation="benchmark fine-tuned model/head",
                input_len="",
                train_windows="",
                val_windows="",
                test_windows="",
                test_positions="",
                pearson=to_float(honest_row.get("lumina")),
                pearson_std="",
                n_runs="1",
                metric_space="NTv3 benchmark Pearson",
                source_artifact=str(honest_path),
                notes=(
                    "Best honest same-track Lumina score from the existing Lumina-vs-NTv3 delta artifact; "
                    f"delta_vs_ntv3_8m={honest_row.get('delta', '')}."
                ),
            )
        )

    for summary in public_summaries:
        rows.append(
            ComparisonRow(
                track_id=track_id,
                assay_type=str(summary.get("assay_type") or assay_type),
                biosample_or_ontology="NTv3 benchmark track",
                model=str(summary["model_name"]),
                training_regime="public NTv3 benchmark model",
                adaptation="public benchmark score averaged across runs",
                input_len="",
                train_windows="",
                val_windows="",
                test_windows="",
                test_positions="",
                pearson=to_float(summary["mean"]),
                pearson_std=f"{to_float(summary['std']):.12g}",
                n_runs=str(summary["n_runs"]),
                metric_space="NTv3 benchmark Pearson",
                source_artifact=str(public_path),
                notes=f"run_ids={','.join(summary.get('run_ids', []))}",
            )
        )

    if public_best_row is not None:
        public_best_model = public_best_row.get("public_best_model", "")
        if public_best_model and all(row.model != public_best_model for row in rows):
            rows.append(
                ComparisonRow(
                    track_id=track_id,
                    assay_type=public_best_row.get("assay_type", assay_type),
                    biosample_or_ontology="NTv3 benchmark track",
                    model=public_best_model,
                    training_regime="public NTv3 benchmark best model",
                    adaptation="public best score context",
                    input_len="",
                    train_windows="",
                    val_windows="",
                    test_windows="",
                    test_positions="",
                    pearson=to_float(public_best_row.get("public_best_score")),
                    pearson_std="",
                    n_runs="",
                    metric_space="NTv3 benchmark Pearson",
                    source_artifact=str(public_path),
                    notes="Added from public best slice because no per-run summary was available.",
                )
            )

    return rows


def render_markdown(
    *,
    rows: list[ComparisonRow],
    audit: TrackAuditRow,
    summary: dict[str, Any],
) -> str:
    ordered = sorted(rows, key=lambda row: row.pearson, reverse=True)
    lines = [
        "# AlphaGenome, Lumina and NTv3 comparison",
        "",
        "## Scope",
        "",
        (
            "This report compiles existing artifacts into a reproducible comparison packet. "
            "It does not rerun model inference or training."
        ),
        "",
        "## Track audit",
        "",
        f"- Track: `{audit.track_id}`",
        f"- NTv3 label: `{audit.ntv3_track_name_clean}` / `{audit.ntv3_assay_type}`",
        (
            "- AlphaGenome request: "
            f"`{audit.alphagenome_output_type}` / `{audit.alphagenome_ontology_curie}` "
            f"({audit.alphagenome_biosample_name}, {audit.alphagenome_assay_title})"
        ),
        f"- Overlap class: `{audit.overlap_class}`",
        f"- Rationale: {audit.rationale}",
        "",
        "## Canonical scores",
        "",
        "| Rank | Model | Regime | Adaptation | Pearson | Runs | Notes |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            "| "
            f"{rank} | {row.model} | {row.training_regime} | {row.adaptation} | "
            f"{row.pearson:.6f} | {row.n_runs or 'NA'} | {row.notes} |"
        )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            (
                "- The primary benchmark-comparable result is Lumina vs NTv3, because both are evaluated "
                "inside the NTv3 protocol."
            ),
            (
                "- AlphaGenome zero-shot measures native alignment between its requested ATAC output and the "
                "NTv3 target."
            ),
            (
                "- AlphaGenome frozen + readout measures whether its frozen predictions carry calibratable "
                "signal for the NTv3 target; it should be presented as a head-only adaptation, not as an "
                "identical training regime to Lumina."
            ),
            "",
            "## Reproducibility metadata",
            "",
            f"- Generated at UTC: `{summary['generated_at_utc']}`",
            f"- Output directory: `{summary['output_dir']}`",
            f"- Canonical CSV: `{summary['canonical_results_csv']}`",
            f"- Track audit CSV: `{summary['track_audit_csv']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_html(markdown_text: str, rows: list[ComparisonRow], audit: TrackAuditRow) -> str:
    score_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row.model)}</td>"
        f"<td>{html.escape(row.training_regime)}</td>"
        f"<td>{html.escape(row.adaptation)}</td>"
        f"<td>{row.pearson:.6f}</td>"
        f"<td>{html.escape(row.n_runs or 'NA')}</td>"
        f"<td>{html.escape(row.notes)}</td>"
        "</tr>"
        for row in sorted(rows, key=lambda item: item.pearson, reverse=True)
    )
    escaped_markdown = html.escape(markdown_text)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AlphaGenome, Lumina and NTv3 comparison</title>
  <style>
    body {{
      color: #1f2933;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
      margin: 32px auto;
      max-width: 1120px;
      padding: 0 20px;
    }}
    h1, h2 {{ color: #111827; }}
    table {{
      border-collapse: collapse;
      font-size: 14px;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid #d8dee9;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f3f6fa; }}
    code, pre {{
      background: #f6f8fa;
      border-radius: 4px;
      padding: 2px 4px;
    }}
    pre {{
      overflow-x: auto;
      padding: 12px;
    }}
  </style>
</head>
<body>
  <h1>AlphaGenome, Lumina and NTv3 comparison</h1>
  <h2>Track audit</h2>
  <p>
    <strong>{html.escape(audit.track_id)}</strong>:
    NTv3 <code>{html.escape(audit.ntv3_track_name_clean)}</code> /
    <code>{html.escape(audit.ntv3_assay_type)}</code>;
    AlphaGenome <code>{html.escape(audit.alphagenome_output_type)}</code> /
    <code>{html.escape(audit.alphagenome_ontology_curie)}</code>.
    Overlap: <code>{html.escape(audit.overlap_class)}</code>.
  </p>
  <h2>Canonical scores</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Regime</th>
        <th>Adaptation</th>
        <th>Pearson</th>
        <th>Runs</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {score_rows}
    </tbody>
  </table>
  <h2>Markdown source</h2>
  <pre>{escaped_markdown}</pre>
</body>
</html>
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zero_metrics = load_json(args.zero_shot_metrics)
    readout_metrics = load_json(args.readout_metrics)
    readout_weights = load_json(args.readout_weights)
    alpha_metadata = alpha_metadata_row(args.alpha_track_metadata)
    honest_rows = read_csv_rows(args.honest_deltas)
    public_rows = read_csv_rows(args.public_results)
    public_best_rows = read_csv_rows(args.public_best_slice)

    honest_row = find_track_row(honest_rows, args.track_id, column="datasets")
    public_best_row = find_track_row(public_best_rows, args.track_id, column="dataset")
    audit = build_track_audit(
        track_id=args.track_id,
        zero_metrics=zero_metrics,
        public_best_row=public_best_row,
        ntv3_track_name_clean=public_track_label(public_rows, track_id=args.track_id),
        alpha_metadata=alpha_metadata,
        alpha_metadata_path=args.alpha_track_metadata,
    )

    public_model_names = [
        "NTv3 8M (pre)",
        "NTv3 100M (pre)",
        "NTv3 650M (pre)",
        "NTv3 650M (pos)",
    ]
    public_summaries = [
        summary
        for model_name in public_model_names
        if (
            summary := public_model_summary(
                public_rows,
                track_id=args.track_id,
                model_name=model_name,
            )
        )
        is not None
    ]

    rows = build_comparison_rows(
        track_id=args.track_id,
        zero_metrics=zero_metrics,
        readout_metrics=readout_metrics,
        readout_weights=readout_weights,
        honest_row=honest_row,
        public_summaries=public_summaries,
        public_best_row=public_best_row,
        zero_path=args.zero_shot_metrics,
        readout_path=args.readout_metrics,
        honest_path=args.honest_deltas,
        public_path=args.public_results,
    )

    canonical_csv = args.output_dir / "canonical_results.csv"
    audit_csv = args.output_dir / "track_audit.csv"
    summary_json = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "comparison_report.md"
    html_path = args.output_dir / "comparison_report.html"

    write_csv(canonical_csv, [asdict(row) for row in rows])
    write_csv(audit_csv, [asdict(audit)])

    best_row = max(rows, key=lambda row: row.pearson)
    lumina_row = next((row for row in rows if row.model.startswith("Lumina ")), None)
    readout_row = next(
        (row for row in rows if row.model == "AlphaGenome" and "readout" in row.training_regime),
        None,
    )
    zero_row = next(
        (row for row in rows if row.model == "AlphaGenome" and "zero-shot" in row.adaptation),
        None,
    )
    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "track_id": args.track_id,
        "output_dir": str(args.output_dir),
        "canonical_results_csv": str(canonical_csv),
        "track_audit_csv": str(audit_csv),
        "comparison_report_md": str(markdown_path),
        "comparison_report_html": str(html_path),
        "best_model": best_row.model,
        "best_pearson": best_row.pearson,
        "track_audit": asdict(audit),
        "artifact_inputs": {
            "zero_shot_metrics": str(args.zero_shot_metrics),
            "readout_metrics": str(args.readout_metrics),
            "readout_weights": str(args.readout_weights),
            "alpha_track_metadata": str(args.alpha_track_metadata),
            "honest_deltas": str(args.honest_deltas),
            "public_results": str(args.public_results),
            "public_best_slice": str(args.public_best_slice),
        },
        "key_deltas": {
            "alphagenome_readout_minus_zero_shot": (
                readout_row.pearson - zero_row.pearson if readout_row is not None and zero_row is not None else None
            ),
            "alphagenome_readout_minus_lumina": (
                readout_row.pearson - lumina_row.pearson if readout_row is not None and lumina_row is not None else None
            ),
            "lumina_minus_ntv3_8m": optional_float(honest_row.get("delta")) if honest_row is not None else None,
        },
    }
    write_json(summary_json, summary)

    markdown_text = render_markdown(rows=rows, audit=audit, summary=summary)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(render_html(markdown_text, rows=rows, audit=audit), encoding="utf-8")

    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
