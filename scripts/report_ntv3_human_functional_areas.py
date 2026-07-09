#!/usr/bin/env python3
"""Generate a detailed NTv3 human/functional area report for Lumina."""

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
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "visualization" / "area_report"
DEFAULT_TARGET_RUN_ID = "ntv3-beat-v6-official-human-functional-seed0-r2"
DEFAULT_TARGET_MODEL_NAME = "Lumina beat-v6"


@dataclass(frozen=True)
class RunKey:
    run_id: str
    model_name: str

    @property
    def label(self) -> str:
        return f"{self.model_name} ({self.run_id})"


@dataclass(frozen=True)
class AssaySummary:
    assay_type: str
    n_tracks: int
    lumina_mean: float
    lumina_rank: int
    complete_runs: int
    best_model: str
    best_run_id: str
    best_mean: float
    gap_to_best: float


@dataclass(frozen=True)
class TrackSummary:
    assay_type: str
    dataset: str
    lumina_metric: float
    lumina_rank: int
    complete_runs: int
    best_model: str
    best_run_id: str
    best_metric: float
    gap_to_best: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-run-id", default=DEFAULT_TARGET_RUN_ID)
    parser.add_argument("--target-model-name", default=DEFAULT_TARGET_MODEL_NAME)
    parser.add_argument("--species", default="human")
    parser.add_argument(
        "--assay-type",
        default=None,
        help="Optional single assay_type to focus the HTML report on.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(row: dict[str, str]) -> float:
    try:
        return float(row["Metric"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid Metric value in row: {row}") from exc


def _rank_descending(values: list[tuple[float, RunKey]], target: RunKey) -> int:
    ordered = sorted(values, key=lambda item: item[0], reverse=True)
    for index, (_value, key) in enumerate(ordered, start=1):
        if key == target:
            return index
    raise ValueError(f"Target run missing from ranking: {target}")


def _target_rows(rows: list[dict[str, str]], *, target: RunKey, species: str) -> list[dict[str, str]]:
    target_rows = [
        row
        for row in rows
        if row.get("species") == species
        and row.get("run_id") == target.run_id
        and row.get("model_name") == target.model_name
    ]
    if not target_rows:
        raise ValueError(f"No rows found for target run {target.label!r} in species={species!r}.")
    return target_rows


def _complete_run_rows(
    rows: list[dict[str, str]],
    *,
    species: str,
    target_tracks: set[str],
) -> dict[RunKey, list[dict[str, str]]]:
    grouped: dict[RunKey, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("species") != species:
            continue
        if row.get("datasets") not in target_tracks:
            continue
        grouped[RunKey(row.get("run_id", ""), row.get("model_name", ""))].append(row)

    complete: dict[RunKey, list[dict[str, str]]] = {}
    for key, group_rows in grouped.items():
        seen_tracks = {row.get("datasets", "") for row in group_rows}
        if seen_tracks == target_tracks:
            complete[key] = group_rows
    return complete


def _rows_by_dataset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["datasets"]: row for row in rows}


def build_assay_summaries(
    *,
    complete_runs: dict[RunKey, list[dict[str, str]]],
    target: RunKey,
    assay_tracks: dict[str, set[str]],
) -> list[AssaySummary]:
    summaries: list[AssaySummary] = []
    rows_by_run = {key: _rows_by_dataset(rows) for key, rows in complete_runs.items()}
    target_rows = rows_by_run[target]

    for assay_type, tracks in sorted(assay_tracks.items()):
        values: list[tuple[float, RunKey]] = []
        for key, rows_by_track in rows_by_run.items():
            metrics = [_metric(rows_by_track[track]) for track in tracks]
            values.append((mean(metrics), key))

        ordered = sorted(values, key=lambda item: item[0], reverse=True)
        best_mean, best_key = ordered[0]
        lumina_mean = mean(_metric(target_rows[track]) for track in tracks)
        summaries.append(
            AssaySummary(
                assay_type=assay_type,
                n_tracks=len(tracks),
                lumina_mean=lumina_mean,
                lumina_rank=_rank_descending(values, target),
                complete_runs=len(values),
                best_model=best_key.model_name,
                best_run_id=best_key.run_id,
                best_mean=best_mean,
                gap_to_best=best_mean - lumina_mean,
            )
        )
    return sorted(summaries, key=lambda item: item.lumina_mean, reverse=True)


def build_track_summaries(
    *,
    complete_runs: dict[RunKey, list[dict[str, str]]],
    target: RunKey,
    target_rows: list[dict[str, str]],
) -> list[TrackSummary]:
    rows_by_run = {key: _rows_by_dataset(rows) for key, rows in complete_runs.items()}
    target_by_track = _rows_by_dataset(target_rows)
    summaries: list[TrackSummary] = []

    for dataset, target_row in sorted(target_by_track.items()):
        values = [(_metric(rows_by_track[dataset]), key) for key, rows_by_track in rows_by_run.items()]
        ordered = sorted(values, key=lambda item: item[0], reverse=True)
        best_metric, best_key = ordered[0]
        lumina_metric = _metric(target_row)
        summaries.append(
            TrackSummary(
                assay_type=target_row.get("assay_type", ""),
                dataset=dataset,
                lumina_metric=lumina_metric,
                lumina_rank=_rank_descending(values, target),
                complete_runs=len(values),
                best_model=best_key.model_name,
                best_run_id=best_key.run_id,
                best_metric=best_metric,
                gap_to_best=best_metric - lumina_metric,
            )
        )
    return sorted(summaries, key=lambda item: (item.assay_type, item.lumina_rank, item.dataset))


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def write_assay_summary_csv(path: Path, summaries: list[AssaySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AssaySummary.__dataclass_fields__))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def write_track_summary_csv(path: Path, summaries: list[TrackSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TrackSummary.__dataclass_fields__))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def _bar_rows_for_assay(
    *,
    complete_runs: dict[RunKey, list[dict[str, str]]],
    target: RunKey,
    tracks: set[str],
) -> str:
    values: list[tuple[float, RunKey]] = []
    for key, rows in complete_runs.items():
        rows_by_track = _rows_by_dataset(rows)
        values.append((mean(_metric(rows_by_track[track]) for track in tracks), key))
    values.sort(key=lambda item: item[0], reverse=True)
    max_value = max(value for value, _key in values)
    parts: list[str] = []
    for rank, (value, key) in enumerate(values, start=1):
        width = 100.0 * value / max_value if max_value > 0 else 0.0
        target_class = " target" if key == target else ""
        parts.append(
            f'<div class="bar-row{target_class}">'
            f'<div class="bar-label">{rank}. {html.escape(key.model_name)}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.2f}%"></div>'
            "</div>"
            f'<div class="bar-value">{value:.3f}</div>'
            "</div>"
        )
    return "\n".join(parts)


def render_html(
    *,
    assay_summaries: list[AssaySummary],
    track_summaries: list[TrackSummary],
    complete_runs: dict[RunKey, list[dict[str, str]]],
    target: RunKey,
    assay_tracks: dict[str, set[str]],
    focused_assay: str | None,
) -> str:
    visible_assays = [
        summary
        for summary in assay_summaries
        if focused_assay is None or summary.assay_type == focused_assay
    ]
    if focused_assay is not None and not visible_assays:
        raise ValueError(f"assay_type={focused_assay!r} was not found.")

    assay_cards = []
    for summary in visible_assays:
        track_rows = [
            track
            for track in track_summaries
            if track.assay_type == summary.assay_type
        ]
        track_table = "\n".join(
            "<tr>"
            f"<td><code>{html.escape(track.dataset)}</code></td>"
            f"<td>{_format_float(track.lumina_metric)}</td>"
            f"<td>{track.lumina_rank}/{track.complete_runs}</td>"
            f"<td>{_format_float(track.best_metric)}</td>"
            f"<td>{html.escape(track.best_model)}</td>"
            f"<td>{_format_float(track.gap_to_best)}</td>"
            "</tr>"
            for track in track_rows
        )
        assay_cards.append(
            f"""
<section class="card">
  <div class="card-head">
    <div>
      <h2>{html.escape(summary.assay_type)}</h2>
      <p>{summary.n_tracks} tracks. Lumina rank {summary.lumina_rank}/{summary.complete_runs}.</p>
    </div>
    <div class="metric">
      <span>Lumina mean</span>
      <strong>{summary.lumina_mean:.3f}</strong>
    </div>
    <div class="metric muted">
      <span>Gap to best</span>
      <strong>{summary.gap_to_best:.3f}</strong>
    </div>
  </div>
  <div class="bars">
    {_bar_rows_for_assay(
        complete_runs=complete_runs,
        target=target,
        tracks=assay_tracks[summary.assay_type],
    )}
  </div>
  <table>
    <thead>
      <tr>
        <th>Track</th><th>Lumina</th><th>Rank</th><th>Best</th><th>Best model</th><th>Gap</th>
      </tr>
    </thead>
    <tbody>{track_table}</tbody>
  </table>
</section>
"""
        )

    summary_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(summary.assay_type)}</td>"
        f"<td>{summary.n_tracks}</td>"
        f"<td>{_format_float(summary.lumina_mean)}</td>"
        f"<td>{summary.lumina_rank}/{summary.complete_runs}</td>"
        f"<td>{_format_float(summary.best_mean)}</td>"
        f"<td>{html.escape(summary.best_model)}</td>"
        f"<td>{_format_float(summary.gap_to_best)}</td>"
        "</tr>"
        for summary in assay_summaries
    )
    track_count = sum(summary.n_tracks for summary in assay_summaries)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NTv3 Human Functional Area Report - Lumina beat-v6</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #64748b;
      --line: #e2e8f0;
      --paper: #fffaf0;
      --panel: #ffffff;
      --lumina: #f97316;
      --other: #334155;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #ffedd5 0, #f8fafc 380px);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, sans-serif;
    }}
    main {{ max-width: 1360px; margin: 0 auto; padding: 34px; }}
    h1 {{ font: 760 34px Georgia, serif; margin: 0 0 8px; }}
    h2 {{ margin: 0; font-size: 24px; }}
    p {{ color: var(--muted); }}
    .hero, .card {{
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.09);
      padding: 22px;
      margin-bottom: 24px;
    }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; }}
    .summary-box {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 16px; padding: 16px; }}
    .summary-box span, .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .summary-box strong, .metric strong {{ display: block; font-size: 28px; margin-top: 4px; }}
    .card-head {{ display: grid; grid-template-columns: 1fr 170px 170px; gap: 16px; align-items: start; }}
    .metric {{ background: #fff7ed; border-radius: 16px; border: 1px solid #fed7aa; padding: 14px; }}
    .metric.muted {{ background: #f8fafc; border-color: var(--line); }}
    .bars {{ margin: 18px 0 18px; }}
    .bar-row {{ display: grid; grid-template-columns: 250px 1fr 64px; gap: 10px; align-items: center; margin: 5px 0; }}
    .bar-label {{ font-size: 12px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .bar-track {{ height: 17px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--other); border-radius: 999px; }}
    .target .bar-fill {{ background: var(--lumina); }}
    .target .bar-label, .target .bar-value {{ color: #9a3412; font-weight: 800; }}
    .bar-value {{ font-size: 12px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; }}
    th {{ color: #475569; font-weight: 800; }}
    code {{ font-size: 12px; }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      .summary-grid, .card-head {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .bar-label {{ text-align: left; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>NTv3 Human Functional - Area Report</h1>
    <p>Run-level comparison on complete human/functional runs over Lumina's 34 tracks.</p>
    <div class="summary-grid">
      <div class="summary-box"><span>Target</span><strong>Lumina beat-v6</strong></div>
      <div class="summary-box"><span>Complete runs</span><strong>{len(complete_runs)}</strong></div>
      <div class="summary-box"><span>Assay groups</span><strong>{len(assay_summaries)}</strong></div>
      <div class="summary-box"><span>Tracks</span><strong>{track_count}</strong></div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Area</th><th>Tracks</th><th>Lumina mean</th><th>Lumina rank</th>
          <th>Best mean</th><th>Best model</th><th>Gap</th>
        </tr>
      </thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </section>
  {''.join(assay_cards)}
</main>
</body>
</html>
"""


def render_markdown(
    *,
    assay_summaries: list[AssaySummary],
    track_summaries: list[TrackSummary],
    complete_run_count: int,
) -> str:
    lines = [
        "# NTv3 Human Functional Area Report",
        "",
        f"Complete runs compared: `{complete_run_count}`.",
        "",
        "## Area Summary",
        "",
        "| Area | Tracks | Lumina mean | Rank | Best mean | Best model | Gap |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for summary in assay_summaries:
        lines.append(
            f"| {summary.assay_type} | {summary.n_tracks} | {_format_float(summary.lumina_mean)} | "
            f"{summary.lumina_rank}/{summary.complete_runs} | {_format_float(summary.best_mean)} | "
            f"{summary.best_model} | {_format_float(summary.gap_to_best)} |"
        )

    for summary in assay_summaries:
        lines.extend(
            [
                "",
                f"## {summary.assay_type}",
                "",
                "| Track | Lumina | Rank | Best | Best model | Gap |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for track in [item for item in track_summaries if item.assay_type == summary.assay_type]:
            lines.append(
                f"| `{track.dataset}` | {_format_float(track.lumina_metric)} | "
                f"{track.lumina_rank}/{track.complete_runs} | {_format_float(track.best_metric)} | "
                f"{track.best_model} | {_format_float(track.gap_to_best)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    target = RunKey(args.target_run_id, args.target_model_name)
    rows = _read_rows(args.input_csv)
    target_rows = _target_rows(rows, target=target, species=args.species)
    target_tracks = {row["datasets"] for row in target_rows}
    assay_tracks: dict[str, set[str]] = defaultdict(set)
    for row in target_rows:
        assay_tracks[row.get("assay_type", "")].add(row["datasets"])

    complete_runs = _complete_run_rows(rows, species=args.species, target_tracks=target_tracks)
    if target not in complete_runs:
        raise ValueError(f"Target run is not complete over {len(target_tracks)} tracks.")

    assay_summaries = build_assay_summaries(
        complete_runs=complete_runs,
        target=target,
        assay_tracks=assay_tracks,
    )
    track_summaries = build_track_summaries(
        complete_runs=complete_runs,
        target=target,
        target_rows=target_rows,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assay_csv = args.output_dir / "assay_summary.csv"
    track_csv = args.output_dir / "track_rankings.csv"
    html_path = args.output_dir / "human_functional_area_report.html"
    markdown_path = args.output_dir / "human_functional_area_report.md"

    write_assay_summary_csv(assay_csv, assay_summaries)
    write_track_summary_csv(track_csv, track_summaries)
    html_path.write_text(
        render_html(
            assay_summaries=assay_summaries,
            track_summaries=track_summaries,
            complete_runs=complete_runs,
            target=target,
            assay_tracks=assay_tracks,
            focused_assay=args.assay_type,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(
            assay_summaries=assay_summaries,
            track_summaries=track_summaries,
            complete_run_count=len(complete_runs),
        ),
        encoding="utf-8",
    )

    print(f"Complete runs: {len(complete_runs)}")
    for summary in assay_summaries:
        print(
            f"{summary.assay_type}: Lumina mean={summary.lumina_mean:.6f} "
            f"rank={summary.lumina_rank}/{summary.complete_runs} "
            f"gap={summary.gap_to_best:.6f}"
        )
    print(f"Wrote {html_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {assay_csv}")
    print(f"Wrote {track_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
