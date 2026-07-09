#!/usr/bin/env python3
"""Render a static NTv3 human/functional leaderboard bar chart."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "ntv3" / "beat-v6-official-human-functional-seed0-r2"
DEFAULT_INPUT_CSV = DEFAULT_RESULTS_DIR / "official_plus_lumina_beat_v6_results.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "visualization"
DEFAULT_RUN_ID = "ntv3-beat-v6-official-human-functional-seed0-r2"
DEFAULT_MODEL_NAME = "Lumina beat-v6"
LUMINA_COLORS = {
    "Lumina beat-v6": "#f97316",
    "Lumina beat-v7": "#059669",
}


@dataclass(frozen=True)
class RunSummary:
    rank: int
    mean_metric: float
    run_id: str
    model_name: str
    n_tracks: int
    best_step: str
    training_tokens: str
    running_time: str
    is_target: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--target-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--species", default="human")
    parser.add_argument(
        "--max-bars",
        type=int,
        default=24,
        help="Number of leading bars to show. The target run is always included.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_value(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _target_tracks(rows: list[dict[str, str]], *, target_run_id: str) -> set[str]:
    tracks = {row["datasets"] for row in rows if row.get("run_id") == target_run_id and row.get("datasets")}
    if not tracks:
        raise ValueError(f"No rows found for target run_id={target_run_id!r}.")
    return tracks


def summarize_complete_runs(
    rows: list[dict[str, str]],
    *,
    species: str,
    target_run_id: str,
    target_model_name: str,
) -> list[RunSummary]:
    tracks = _target_tracks(rows, target_run_id=target_run_id)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("species") != species:
            continue
        if row.get("datasets") not in tracks:
            continue
        grouped[(row.get("run_id", ""), row.get("model_name", ""))].append(row)

    summaries: list[RunSummary] = []
    for (run_id, model_name), group_rows in grouped.items():
        seen_tracks = {row.get("datasets", "") for row in group_rows}
        if seen_tracks != tracks:
            continue
        metrics = [_float_value(row.get("Metric", "")) for row in group_rows]
        if any(math.isnan(metric) for metric in metrics):
            continue
        sample = group_rows[0]
        summaries.append(
            RunSummary(
                rank=0,
                mean_metric=mean(metrics),
                run_id=run_id,
                model_name=model_name,
                n_tracks=len(group_rows),
                best_step=sample.get("best_step", ""),
                training_tokens=sample.get("training_tokens", ""),
                running_time=sample.get("running_time", ""),
                is_target=run_id == target_run_id and model_name == target_model_name,
            )
        )

    summaries.sort(key=lambda item: item.mean_metric, reverse=True)
    return [
        RunSummary(
            rank=index,
            mean_metric=item.mean_metric,
            run_id=item.run_id,
            model_name=item.model_name,
            n_tracks=item.n_tracks,
            best_step=item.best_step,
            training_tokens=item.training_tokens,
            running_time=item.running_time,
            is_target=item.is_target,
        )
        for index, item in enumerate(summaries, start=1)
    ]


def select_chart_rows(summaries: list[RunSummary], *, max_bars: int) -> list[RunSummary]:
    if max_bars <= 0:
        raise ValueError("--max-bars must be positive.")
    selected = summaries[:max_bars]
    target = next((item for item in summaries if item.is_target), None)
    if target is not None and target not in selected:
        selected = [*summaries[: max(0, max_bars - 1)], target]
    return selected


def _bar_label(summary: RunSummary) -> str:
    return f"{summary.rank}. {summary.model_name}"


def _bar_color(summary: RunSummary) -> str:
    return LUMINA_COLORS.get(summary.model_name, "#334155")


def _label_color(summary: RunSummary) -> str:
    if summary.model_name == "Lumina beat-v6":
        return "#9a3412"
    if summary.model_name == "Lumina beat-v7":
        return "#065f46"
    return "#0f172a"


def _is_lumina(summary: RunSummary) -> bool:
    return summary.model_name.startswith("Lumina ")


def render_svg(
    summaries: list[RunSummary],
    *,
    target_run_id: str,
    n_tracks: int,
) -> str:
    chart_rows = list(reversed(summaries))
    width = 1280
    left = 350
    right = 160
    top = 104
    row_height = 30
    bottom = 92
    height = top + len(chart_rows) * row_height + bottom
    chart_width = width - left - right
    max_metric = max((row.mean_metric for row in summaries), default=0.1)
    x_max = max(0.1, math.ceil(max_metric / 0.05) * 0.05)
    tick_count = 7

    def x_pos(value: float) -> float:
        return left + (value / x_max) * chart_width

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>'
        ".title{font:700 30px Georgia,serif;fill:#0f172a}"
        ".subtitle{font:400 14px ui-sans-serif,system-ui,sans-serif;fill:#475569}"
        ".label{font:500 13px ui-sans-serif,system-ui,sans-serif;fill:#0f172a}"
        ".small{font:400 12px ui-sans-serif,system-ui,sans-serif;fill:#64748b}"
        ".value{font:700 13px ui-sans-serif,system-ui,sans-serif;fill:#0f172a}"
        ".axis{font:400 11px ui-sans-serif,system-ui,sans-serif;fill:#64748b}"
        "</style>",
        '<text x="40" y="48" class="title">NTv3 Benchmark - Human Functional</text>',
        (
            f'<text x="40" y="74" class="subtitle">Mean Pearson across {n_tracks} matching tracks. '
            f'Target run: {html.escape(target_run_id)}.</text>'
        ),
    ]

    for tick in range(tick_count + 1):
        value = x_max * tick / tick_count
        x = x_pos(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 20}" x2="{x:.1f}" '
            f'y2="{height - bottom + 12}" stroke="#e2e8f0"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom + 34}" '
            f'text-anchor="middle" class="axis">{value:.2f}</text>'
        )
    parts.append(
        f'<text x="{left + chart_width / 2:.1f}" y="{height - 24}" '
        'text-anchor="middle" class="small">Mean Pearson</text>'
    )

    for row_index, summary in enumerate(chart_rows):
        y = top + row_index * row_height
        bar_height = 20
        bar_y = y + 4
        bar_width = max(2.0, x_pos(summary.mean_metric) - left)
        color = _bar_color(summary)
        label_color = _label_color(summary)
        parts.append(
            f'<text x="{left - 14}" y="{bar_y + 14}" text-anchor="end" '
            f'class="label" fill="{label_color}">{html.escape(_bar_label(summary))}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{bar_y}" width="{bar_width:.1f}" '
            f'height="{bar_height}" rx="5" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 8:.1f}" y="{bar_y + 14}" '
            f'class="value">{summary.mean_metric:.3f}</text>'
        )
        if _is_lumina(summary):
            parts.append(
                f'<text x="{width - 40}" y="{bar_y + 14}" text-anchor="end" class="small">'
                f"{html.escape(summary.model_name.replace('Lumina ', ''))}</text>"
            )

    parts.append('<circle cx="40" cy="98" r="6" fill="#f97316"/>')
    parts.append('<text x="54" y="102" class="small">Lumina beat-v6</text>')
    parts.append('<circle cx="178" cy="98" r="6" fill="#059669"/>')
    parts.append('<text x="192" y="102" class="small">Lumina beat-v7</text>')
    parts.append('<circle cx="316" cy="98" r="6" fill="#334155"/>')
    parts.append('<text x="330" y="102" class="small">Public NTv3 Space rows</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_html(*, svg: str, summaries: list[RunSummary], target: RunSummary | None, n_tracks: int) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{summary.rank}</td>"
        f"<td>{html.escape(summary.model_name)}</td>"
        f"<td><code>{html.escape(summary.run_id)}</code></td>"
        f"<td>{summary.mean_metric:.6f}</td>"
        f"<td>{summary.n_tracks}</td>"
        f"<td>{html.escape(summary.training_tokens)}</td>"
        "</tr>"
        for summary in summaries
    )
    target_summary = ""
    if target is not None:
        target_summary = (
            f"<p><strong>Lumina rank:</strong> {target.rank}/{len(summaries)} complete runs. "
            f"<strong>Mean Pearson:</strong> {target.mean_metric:.6f}. "
            f"<strong>Tracks:</strong> {n_tracks}.</p>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NTv3 Human Functional - Lumina beat-v6 vs beat-v7</title>
  <style>
    body {{ margin: 0; background: #f8fafc; color: #0f172a; font-family: ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 28px; }}
    .panel {{
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    .chart {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #475569; font-weight: 700; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
<main>
  <section class="panel">
    <div class="chart">
{svg}
    </div>
    {target_summary}
    <table>
      <thead>
        <tr>
          <th>Rank</th><th>Model</th><th>Run</th>
          <th>Mean Pearson</th><th>Tracks</th><th>Training Tokens</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def write_summary_csv(path: Path, summaries: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "mean_metric",
                "run_id",
                "model_name",
                "n_tracks",
                "best_step",
                "training_tokens",
                "running_time",
                "is_target",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def main() -> int:
    args = parse_args()
    rows = _read_rows(args.input_csv)
    n_tracks = len(_target_tracks(rows, target_run_id=args.target_run_id))
    summaries = summarize_complete_runs(
        rows,
        species=args.species,
        target_run_id=args.target_run_id,
        target_model_name=args.target_model_name,
    )
    target = next((summary for summary in summaries if summary.is_target), None)
    chart_rows = select_chart_rows(summaries, max_bars=args.max_bars)
    svg = render_svg(chart_rows, target_run_id=args.target_run_id, n_tracks=n_tracks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "human_functional_bar_chart.svg"
    html_path = args.output_dir / "human_functional_bar_chart.html"
    summary_path = args.output_dir / "human_functional_summary.csv"
    svg_path.write_text(svg + "\n", encoding="utf-8")
    html_path.write_text(
        render_html(svg=svg, summaries=summaries, target=target, n_tracks=n_tracks),
        encoding="utf-8",
    )
    write_summary_csv(summary_path, summaries)

    if target is None:
        print("Warning: target run was not present in complete-run summaries.")
    else:
        print(f"Lumina rank: {target.rank}/{len(summaries)}")
        print(f"Lumina mean Pearson: {target.mean_metric:.6f}")
    print(f"Wrote {html_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
