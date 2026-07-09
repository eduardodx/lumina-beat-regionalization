#!/usr/bin/env python3
"""Compare AlphaGenome predictions against one NTv3 functional track.

This script is intentionally scoped to one track at a time. It is meant to
make a narrow, auditable bridge between AlphaGenome genome-track predictions
and the local NTv3 human/functional protocol.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyBigWig
from pyfaidx import Fasta

DEFAULT_DATASET_ROOT = Path("data/datasets/ntv3")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/alphagenome_ntv3_track")
DEFAULT_TRACK_ID = "ENCSR814RGG"
DEFAULT_ONTOLOGY_TERM = "UBERON:0000056"
DEFAULT_OUTPUT_TYPE = "ATAC"
DEFAULT_SEQUENCE_LENGTH = 32_768
DEFAULT_KEEP_TARGET_CENTER_FRACTION = 0.375
DEFAULT_MAX_WINDOWS = 32
SUPPORTED_SEQUENCE_LENGTHS = {
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
}
OUTPUT_ATTRS = {
    "ATAC": "atac",
    "DNASE": "dnase",
    "RNA_SEQ": "rna_seq",
    "CAGE": "cage",
    "PROCAP": "procap",
    "CHIP_HISTONE": "chip_histone",
    "CHIP_TF": "chip_tf",
}


@dataclass(frozen=True)
class TrackInfo:
    track_id: str
    path: Path
    assay: str
    mean: float
    std: float | None


@dataclass(frozen=True)
class Window:
    chrom: str
    start: int
    end: int
    index: int


@dataclass
class PearsonAccumulator:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64).reshape(-1)
        y64 = np.asarray(y, dtype=np.float64).reshape(-1)
        if x64.shape != y64.shape:
            raise ValueError(f"Shape mismatch for Pearson update: {x64.shape=} {y64.shape=}")
        self.count += int(x64.size)
        self.sum_x += float(x64.sum())
        self.sum_y += float(y64.sum())
        self.sum_x2 += float(np.square(x64).sum())
        self.sum_y2 += float(np.square(y64).sum())
        self.sum_xy += float((x64 * y64).sum())

    def compute(self) -> float:
        n = float(self.count)
        if n <= 1:
            return 0.0
        numerator = n * self.sum_xy - self.sum_x * self.sum_y
        denom_x = n * self.sum_x2 - self.sum_x * self.sum_x
        denom_y = n * self.sum_y2 - self.sum_y * self.sum_y
        denominator = math.sqrt(max(denom_x * denom_y, 0.0))
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-track AlphaGenome vs NTv3 comparison.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--species", default="human")
    parser.add_argument("--split", default="test")
    parser.add_argument("--track-id", default=DEFAULT_TRACK_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument(
        "--keep-target-center-fraction",
        type=float,
        default=DEFAULT_KEEP_TARGET_CENTER_FRACTION,
    )
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=DEFAULT_MAX_WINDOWS,
        help="Maximum windows to evaluate. Use 0 to evaluate all NTv3 windows for the split.",
    )
    parser.add_argument(
        "--window-sampling",
        choices=("uniform", "first"),
        default="uniform",
        help="How to subsample windows when --max-windows is positive.",
    )
    parser.add_argument("--max-n-fraction", type=float, default=0.05)
    parser.add_argument("--ontology-term", default=DEFAULT_ONTOLOGY_TERM)
    parser.add_argument("--alphagenome-output-type", default=DEFAULT_OUTPUT_TYPE)
    parser.add_argument("--alphagenome-model-version", default="all_folds")
    parser.add_argument(
        "--alphagenome-source",
        choices=("huggingface", "kaggle"),
        default="huggingface",
    )
    parser.add_argument(
        "--alphagenome-multi-track-policy",
        choices=("error", "mean", "first", "index"),
        default="error",
        help="What to do if AlphaGenome returns multiple tracks for the ontology/output request.",
    )
    parser.add_argument("--alphagenome-track-index", type=int, default=0)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file used only to populate missing HF/Kaggle env vars.",
    )
    parser.add_argument(
        "--lumina-dataset-scores-path",
        type=Path,
        default=None,
        help="Optional Lumina dataset_scores.csv for contextual same-track score reporting.",
    )
    parser.add_argument(
        "--public-track-scores-path",
        type=Path,
        default=Path("artifacts/analysis/ntv3_recent/ntv3_latest_track_slice_analysis.csv"),
        help="Optional CSV with public NTv3 same-track score context.",
    )
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args(argv)


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def load_track_info(dataset_root: Path, species: str, track_id: str) -> TrackInfo:
    track_path = dataset_root / species / "functional_tracks" / f"{track_id}.bigwig"
    if not track_path.is_file():
        raise FileNotFoundError(f"NTv3 functional track not found: {track_path}")

    metadata_path = dataset_root / "benchmark_metadata.tsv"
    if not metadata_path.is_file():
        return TrackInfo(track_id=track_id, path=track_path, assay="", mean=1.0, std=None)

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("file_id") != track_id:
                continue
            return TrackInfo(
                track_id=track_id,
                path=track_path,
                assay=row.get("assay", ""),
                mean=max(float(row.get("mean") or 1.0), 1e-6),
                std=float(row["std"]) if row.get("std") else None,
            )
    return TrackInfo(track_id=track_id, path=track_path, assay="", mean=1.0, std=None)


def load_split_windows(
    splits_path: Path,
    *,
    split: str,
    sequence_length: int,
    overlap: float,
) -> list[Window]:
    if not splits_path.is_file():
        raise FileNotFoundError(f"NTv3 splits file not found: {splits_path}")
    stride = max(1, int((1.0 - overlap) * sequence_length))
    windows: list[Window] = []
    with splits_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 4 or row[3] != split:
                continue
            chrom = row[0]
            region_start = int(float(row[1]))
            region_end = int(float(row[2]))
            if region_end - region_start < sequence_length:
                continue
            num_samples = 1 + max(0, (region_end - region_start - sequence_length) // stride)
            for sample_index in range(num_samples):
                start = region_start + sample_index * stride
                windows.append(Window(chrom=chrom, start=start, end=start + sequence_length, index=len(windows)))
    if not windows:
        raise ValueError(f"No valid NTv3 windows found for split={split!r}.")
    return windows


def select_windows(
    windows: list[Window],
    *,
    max_windows: int,
    sampling: str,
) -> list[Window]:
    if max_windows <= 0 or max_windows >= len(windows):
        return windows
    if sampling == "first":
        return windows[:max_windows]
    if sampling != "uniform":
        raise ValueError(f"Unsupported window sampling: {sampling!r}")
    indices = np.linspace(0, len(windows) - 1, num=max_windows, dtype=int)
    return [windows[int(index)] for index in indices]


def resolve_chrom_name(chrom: str, names: set[str]) -> str:
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append(f"chr{chrom}")
    for candidate in candidates:
        if candidate in names:
            return candidate
    return chrom


def center_bounds(sequence_length: int, keep_fraction: float) -> tuple[int, int]:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("--keep-target-center-fraction must be in (0, 1].")
    offset = int(sequence_length * (1.0 - keep_fraction) // 2)
    return offset, sequence_length - offset


def ntv3_transform(values: np.ndarray, *, track_mean: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float32) / np.float32(track_mean)
    return np.where(scaled > 10.0, 2.0 * np.sqrt(scaled * 10.0) - 10.0, scaled)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    acc = PearsonAccumulator()
    acc.update(x, y)
    return acc.compute()


def read_sequence(fasta: Fasta, window: Window) -> str:
    chrom = resolve_chrom_name(window.chrom, set(fasta.keys()))
    return str(fasta[chrom][window.start : window.end]).upper()


def read_target(
    bigwig: pyBigWig.pyBigWig,
    window: Window,
    *,
    center_start: int,
    center_end: int,
) -> np.ndarray:
    chrom = resolve_chrom_name(window.chrom, set(bigwig.chroms().keys()))
    values = bigwig.values(chrom, window.start, window.end, numpy=True)
    target = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
    return target[center_start:center_end]


def create_alphagenome_model(*, source: str, model_version: str) -> Any:
    try:
        dna_model = importlib.import_module("alphagenome_research.model.dna_model")
    except ImportError as exc:
        raise RuntimeError(
            "alphagenome_research is not installed. Install it with:\n"
            "  uv pip install git+https://github.com/google-deepmind/alphagenome_research.git"
        ) from exc

    if source == "huggingface":
        return dna_model.create_from_huggingface(model_version)
    if source == "kaggle":
        return dna_model.create_from_kaggle(model_version)
    raise ValueError(f"Unsupported AlphaGenome source: {source!r}")


def predict_alphagenome(
    model: Any,
    sequence: str,
    *,
    interval: Any,
    output_type_name: str,
    ontology_term: str,
) -> tuple[np.ndarray, Any]:
    dna_model = importlib.import_module("alphagenome_research.model.dna_model")

    output_type_name = output_type_name.upper()
    if output_type_name not in OUTPUT_ATTRS:
        raise ValueError(f"Unsupported AlphaGenome output type: {output_type_name!r}")
    output_type = getattr(dna_model.OutputType, output_type_name)
    outputs = model.predict_sequence(
        sequence,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=[output_type],
        ontology_terms=[ontology_term],
        interval=interval,
    )
    track_data = getattr(outputs, OUTPUT_ATTRS[output_type_name])
    if track_data is None:
        raise RuntimeError(f"AlphaGenome returned no TrackData for {output_type_name}.")
    values = np.asarray(track_data.values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise RuntimeError(f"Expected 2D AlphaGenome track values, got shape={values.shape}.")
    return values, track_data


def select_alphagenome_prediction(
    values: np.ndarray,
    *,
    policy: str,
    track_index: int,
) -> np.ndarray:
    num_tracks = values.shape[1]
    if num_tracks == 1:
        return values[:, 0]
    if policy == "error":
        raise RuntimeError(
            f"AlphaGenome returned {num_tracks} tracks. Use --alphagenome-multi-track-policy "
            "mean, first, or index to make the selection explicit."
        )
    if policy == "mean":
        return values.mean(axis=1)
    if policy == "first":
        return values[:, 0]
    if policy == "index":
        if track_index < 0 or track_index >= num_tracks:
            raise ValueError(f"--alphagenome-track-index={track_index} outside [0, {num_tracks}).")
        return values[:, track_index]
    raise ValueError(f"Unsupported AlphaGenome multi-track policy: {policy!r}")


def make_interval(window: Window) -> Any:
    genome = importlib.import_module("alphagenome.data.genome")

    return genome.Interval(chromosome=window.chrom, start=window.start, end=window.end)


def row_for_context_score(path: Path | None, *, track_id: str, metric_column: str) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_track = row.get("datasets") or row.get("dataset") or row.get("track_name_clean")
            if row_track != track_id:
                continue
            value = row.get(metric_column) or row.get("Metric") or row.get("score")
            if value is None:
                return row
            result: dict[str, Any] = dict(row)
            result["metric_value"] = float(value)
            return result
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_window_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_track_metadata(path: Path, track_data: Any) -> None:
    metadata = getattr(track_data, "metadata", None)
    if metadata is not None and hasattr(metadata, "to_csv"):
        metadata.to_csv(path, index=False)


def summarize_target_only(
    *,
    fasta: Fasta,
    bigwig: pyBigWig.pyBigWig,
    windows: list[Window],
    center_start: int,
    center_end: int,
    max_n_fraction: float,
) -> dict[str, Any]:
    total_positions = 0
    total_target_sum = 0.0
    total_target_sq_sum = 0.0
    skipped_high_n = 0
    for window in windows:
        sequence = read_sequence(fasta, window)
        n_fraction = sequence.count("N") / max(len(sequence), 1)
        if n_fraction > max_n_fraction:
            skipped_high_n += 1
            continue
        target = read_target(bigwig, window, center_start=center_start, center_end=center_end)
        total_positions += int(target.size)
        total_target_sum += float(target.sum())
        total_target_sq_sum += float(np.square(target.astype(np.float64)).sum())
    mean = total_target_sum / total_positions if total_positions else 0.0
    variance = total_target_sq_sum / total_positions - mean * mean if total_positions else 0.0
    return {
        "evaluated_windows": len(windows) - skipped_high_n,
        "skipped_high_n_windows": skipped_high_n,
        "target_positions": total_positions,
        "target_raw_mean": mean,
        "target_raw_std": math.sqrt(max(variance, 0.0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sequence_length not in SUPPORTED_SEQUENCE_LENGTHS:
        raise ValueError(
            f"AlphaGenome supports these sequence lengths: {sorted(SUPPORTED_SEQUENCE_LENGTHS)}. "
            f"Got --sequence-length={args.sequence_length}."
        )
    load_env_file(args.env_file)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    species_root = args.dataset_root / args.species
    fasta_path = species_root / "genome.fasta"
    splits_path = species_root / "splits.bed"
    if not fasta_path.is_file():
        raise FileNotFoundError(f"NTv3 FASTA not found: {fasta_path}")

    track_info = load_track_info(args.dataset_root, args.species, args.track_id)
    all_windows = load_split_windows(
        splits_path,
        split=args.split,
        sequence_length=args.sequence_length,
        overlap=args.overlap,
    )
    windows = select_windows(
        all_windows,
        max_windows=args.max_windows,
        sampling=args.window_sampling,
    )
    center_start, center_end = center_bounds(args.sequence_length, args.keep_target_center_fraction)

    config = {
        "dataset_root": str(args.dataset_root),
        "species": args.species,
        "split": args.split,
        "track_info": {
            **asdict(track_info),
            "path": str(track_info.path),
        },
        "ontology_term": args.ontology_term,
        "alphagenome_output_type": args.alphagenome_output_type,
        "alphagenome_model_version": args.alphagenome_model_version,
        "alphagenome_source": args.alphagenome_source,
        "sequence_length": args.sequence_length,
        "keep_target_center_fraction": args.keep_target_center_fraction,
        "target_center_start": center_start,
        "target_center_end": center_end,
        "target_length": center_end - center_start,
        "total_split_windows": len(all_windows),
        "selected_windows": len(windows),
        "window_sampling": args.window_sampling,
        "dry_run": bool(args.dry_run),
    }
    write_json(output_dir / "config.json", config)

    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    bigwig = pyBigWig.open(str(track_info.path))
    if args.dry_run:
        summary = summarize_target_only(
            fasta=fasta,
            bigwig=bigwig,
            windows=windows,
            center_start=center_start,
            center_end=center_end,
            max_n_fraction=args.max_n_fraction,
        )
        result = {
            "status": "dry_run",
            "config": config,
            "target_summary": summary,
            "lumina_context": row_for_context_score(
                args.lumina_dataset_scores_path,
                track_id=args.track_id,
                metric_column="Metric",
            ),
            "public_ntv3_context": row_for_context_score(
                args.public_track_scores_path,
                track_id=args.track_id,
                metric_column="public_best_score",
            ),
        }
        write_json(output_dir / "metrics.json", result)
        return result

    model = create_alphagenome_model(
        source=args.alphagenome_source,
        model_version=args.alphagenome_model_version,
    )

    raw_acc = PearsonAccumulator()
    ntv3_scaled_both_acc = PearsonAccumulator()
    ntv3_target_only_acc = PearsonAccumulator()
    window_rows: list[dict[str, Any]] = []
    saved_predictions: list[np.ndarray] = []
    saved_targets: list[np.ndarray] = []
    last_track_data: Any | None = None
    skipped_high_n = 0

    for ordinal, window in enumerate(windows, start=1):
        sequence = read_sequence(fasta, window)
        n_fraction = sequence.count("N") / max(len(sequence), 1)
        if n_fraction > args.max_n_fraction:
            skipped_high_n += 1
            continue

        target_raw = read_target(bigwig, window, center_start=center_start, center_end=center_end)
        interval = make_interval(window)
        alpha_values, track_data = predict_alphagenome(
            model,
            sequence,
            interval=interval,
            output_type_name=args.alphagenome_output_type,
            ontology_term=args.ontology_term,
        )
        last_track_data = track_data
        alpha_pred_raw_full = select_alphagenome_prediction(
            alpha_values,
            policy=args.alphagenome_multi_track_policy,
            track_index=args.alphagenome_track_index,
        )
        if alpha_pred_raw_full.shape[0] != args.sequence_length:
            raise RuntimeError(
                "AlphaGenome prediction length does not match NTv3 window length: "
                f"{alpha_pred_raw_full.shape[0]} vs {args.sequence_length}."
            )
        alpha_pred_raw = alpha_pred_raw_full[center_start:center_end]
        alpha_pred_ntv3 = ntv3_transform(alpha_pred_raw, track_mean=track_info.mean)
        target_ntv3 = ntv3_transform(target_raw, track_mean=track_info.mean)

        raw_acc.update(alpha_pred_raw, target_raw)
        ntv3_scaled_both_acc.update(alpha_pred_ntv3, target_ntv3)
        ntv3_target_only_acc.update(alpha_pred_raw, target_ntv3)
        window_rows.append(
            {
                "window_ordinal": ordinal,
                "window_index": window.index,
                "chrom": window.chrom,
                "start": window.start,
                "end": window.end,
                "target_start": window.start + center_start,
                "target_end": window.start + center_end,
                "n_fraction": n_fraction,
                "num_alpha_tracks": int(alpha_values.shape[1]),
                "pearson_raw": pearson(alpha_pred_raw, target_raw),
                "pearson_ntv3_scaled_both": pearson(alpha_pred_ntv3, target_ntv3),
                "pearson_ntv3_target_only": pearson(alpha_pred_raw, target_ntv3),
                "target_raw_mean": float(target_raw.mean()),
                "alpha_raw_mean": float(alpha_pred_raw.mean()),
                "target_ntv3_mean": float(target_ntv3.mean()),
                "alpha_ntv3_mean": float(alpha_pred_ntv3.mean()),
            }
        )
        if args.save_arrays:
            saved_predictions.append(alpha_pred_raw.astype(np.float32))
            saved_targets.append(target_raw.astype(np.float32))
        if args.log_every > 0 and ordinal % args.log_every == 0:
            print(
                f"[{ordinal}/{len(windows)}] {window.chrom}:{window.start}-{window.end} "
                f"raw_pearson={raw_acc.compute():.6f} ntv3_scaled={ntv3_scaled_both_acc.compute():.6f}",
                flush=True,
            )

    write_window_rows(output_dir / "window_scores.csv", window_rows)
    if last_track_data is not None:
        write_track_metadata(output_dir / "alphagenome_track_metadata.csv", last_track_data)
    if args.save_arrays and saved_predictions:
        np.savez_compressed(
            output_dir / "alphagenome_ntv3_arrays.npz",
            alpha_prediction_raw=np.concatenate(saved_predictions),
            target_raw=np.concatenate(saved_targets),
        )

    metrics = {
        "status": "completed",
        "config": config,
        "evaluated_windows": len(window_rows),
        "skipped_high_n_windows": skipped_high_n,
        "num_positions": raw_acc.count,
        "alpha_vs_ntv3_pearson_raw": raw_acc.compute(),
        "alpha_vs_ntv3_pearson_ntv3_scaled_both": ntv3_scaled_both_acc.compute(),
        "alpha_vs_ntv3_pearson_ntv3_target_only": ntv3_target_only_acc.compute(),
        "mean_window_pearson_raw": float(np.mean([row["pearson_raw"] for row in window_rows]))
        if window_rows
        else 0.0,
        "mean_window_pearson_ntv3_scaled_both": float(
            np.mean([row["pearson_ntv3_scaled_both"] for row in window_rows])
        )
        if window_rows
        else 0.0,
        "lumina_context": row_for_context_score(
            args.lumina_dataset_scores_path,
            track_id=args.track_id,
            metric_column="Metric",
        ),
        "public_ntv3_context": row_for_context_score(
            args.public_track_scores_path,
            track_id=args.track_id,
            metric_column="public_best_score",
        ),
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metrics = run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
