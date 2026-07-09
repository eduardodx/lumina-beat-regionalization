#!/usr/bin/env python3
"""Train a frozen-AlphaGenome readout for one NTv3 functional track.

The AlphaGenome model remains frozen. This script streams AlphaGenome genome
track predictions, builds a small local feature bank from those predictions,
fits a ridge readout on NTv3 train windows, selects the ridge value on val
windows, and reports Pearson on test windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyBigWig
from pyfaidx import Fasta

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_alphagenome_ntv3_track import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_KEEP_TARGET_CENTER_FRACTION,
    DEFAULT_ONTOLOGY_TERM,
    DEFAULT_OUTPUT_TYPE,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TRACK_ID,
    PearsonAccumulator,
    Window,
    center_bounds,
    create_alphagenome_model,
    load_env_file,
    load_split_windows,
    load_track_info,
    make_interval,
    ntv3_transform,
    predict_alphagenome,
    read_sequence,
    read_target,
    row_for_context_score,
    select_alphagenome_prediction,
    select_windows,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/alphagenome_ntv3_readout")
DEFAULT_LAMBDAS = "0,1e-6,1e-4,1e-2,1,100"
DEFAULT_SMOOTH_WINDOWS = "25,101,501"


@dataclass(frozen=True)
class FeatureConfig:
    include_raw: bool
    include_sqrt: bool
    include_log1p: bool
    smooth_windows: tuple[int, ...]


@dataclass
class StreamingMoments:
    xtx: np.ndarray
    xty: np.ndarray
    count: int = 0

    @classmethod
    def create(cls, num_features: int) -> StreamingMoments:
        return cls(
            xtx=np.zeros((num_features, num_features), dtype=np.float64),
            xty=np.zeros(num_features, dtype=np.float64),
        )

    def update(self, features: np.ndarray, target: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D features, got {x.shape}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Feature/target mismatch: {x.shape[0]} vs {y.shape[0]}.")
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.count += int(x.shape[0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen AlphaGenome NTv3 readout.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--species", default="human")
    parser.add_argument("--track-id", default=DEFAULT_TRACK_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--keep-target-center-fraction", type=float, default=DEFAULT_KEEP_TARGET_CENTER_FRACTION)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--train-windows", type=int, default=512)
    parser.add_argument("--val-windows", type=int, default=256)
    parser.add_argument("--test-windows", type=int, default=1024)
    parser.add_argument("--window-sampling", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--max-n-fraction", type=float, default=0.05)
    parser.add_argument("--ontology-term", default=DEFAULT_ONTOLOGY_TERM)
    parser.add_argument("--alphagenome-output-type", default=DEFAULT_OUTPUT_TYPE)
    parser.add_argument("--alphagenome-model-version", default="all_folds")
    parser.add_argument("--alphagenome-source", choices=("huggingface", "kaggle"), default="huggingface")
    parser.add_argument(
        "--alphagenome-multi-track-policy",
        choices=("error", "mean", "first", "index"),
        default="error",
    )
    parser.add_argument("--alphagenome-track-index", type=int, default=0)
    parser.add_argument("--ridge-lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--smooth-windows", default=DEFAULT_SMOOTH_WINDOWS)
    parser.add_argument("--no-raw-feature", action="store_true")
    parser.add_argument("--no-sqrt-feature", action="store_true")
    parser.add_argument("--no-log1p-feature", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--save-val-arrays", action="store_true")
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument(
        "--lumina-dataset-scores-path",
        type=Path,
        default=Path(
            "artifacts/analysis/ntv3_recent/beat-v7-20k-aggressive-moderate-lr125-llrd085-seed0/"
            "human/functional/dataset_scores.csv"
        ),
    )
    parser.add_argument(
        "--public-track-scores-path",
        type=Path,
        default=Path("artifacts/analysis/ntv3_recent/ntv3_latest_track_slice_analysis.csv"),
    )
    return parser.parse_args(argv)


def parse_float_list(raw_value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw_value.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one ridge lambda.")
    return values


def parse_int_list(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw_value.split(",") if item.strip())
    for value in values:
        if value <= 1 or value % 2 == 0:
            raise ValueError("Smooth windows must be odd integers greater than 1.")
    return values


def feature_names(config: FeatureConfig) -> list[str]:
    names = ["intercept"]
    if config.include_raw:
        names.append("alpha_raw")
    if config.include_sqrt:
        names.append("sqrt_alpha")
    if config.include_log1p:
        names.append("log1p_alpha")
    names.extend(f"mean_{window}" for window in config.smooth_windows)
    return names


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    padded = np.pad(values.astype(np.float32), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def build_features(values: np.ndarray, config: FeatureConfig) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    nonnegative = np.maximum(x, 0.0)
    columns = [np.ones_like(x, dtype=np.float32)]
    if config.include_raw:
        columns.append(x)
    if config.include_sqrt:
        columns.append(np.sqrt(nonnegative).astype(np.float32))
    if config.include_log1p:
        columns.append(np.log1p(nonnegative).astype(np.float32))
    columns.extend(moving_average(x, window) for window in config.smooth_windows)
    return np.stack(columns, axis=1)


def solve_ridge(moments: StreamingMoments, ridge_lambda: float) -> np.ndarray:
    penalty = np.eye(moments.xtx.shape[0], dtype=np.float64) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    return np.linalg.solve(moments.xtx + penalty, moments.xty)


def load_selected_windows(
    dataset_root: Path,
    species: str,
    split: str,
    *,
    sequence_length: int,
    overlap: float,
    max_windows: int,
    sampling: str,
) -> tuple[int, list[Window]]:
    all_windows = load_split_windows(
        dataset_root / species / "splits.bed",
        split=split,
        sequence_length=sequence_length,
        overlap=overlap,
    )
    selected = select_windows(all_windows, max_windows=max_windows, sampling=sampling)
    return len(all_windows), selected


def alpha_center_prediction(
    model: Any,
    sequence: str,
    window: Window,
    *,
    center_start: int,
    center_end: int,
    output_type_name: str,
    ontology_term: str,
    multi_track_policy: str,
    track_index: int,
) -> np.ndarray:
    alpha_values, _track_data = predict_alphagenome(
        model,
        sequence,
        interval=make_interval(window),
        output_type_name=output_type_name,
        ontology_term=ontology_term,
    )
    alpha_full = select_alphagenome_prediction(
        alpha_values,
        policy=multi_track_policy,
        track_index=track_index,
    )
    return np.asarray(alpha_full[center_start:center_end], dtype=np.float32)


def stream_split(
    *,
    split_name: str,
    windows: list[Window],
    model: Any,
    fasta: Fasta,
    bigwig: pyBigWig.pyBigWig,
    track_mean: float,
    center_start: int,
    center_end: int,
    feature_config: FeatureConfig,
    args: argparse.Namespace,
    moments: StreamingMoments | None = None,
    weights_by_lambda: dict[float, np.ndarray] | None = None,
    collect_arrays: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[np.ndarray, np.ndarray] | None]:
    zero_raw_acc = PearsonAccumulator()
    zero_ntv3_acc = PearsonAccumulator()
    readout_accs = {ridge_lambda: PearsonAccumulator() for ridge_lambda in weights_by_lambda or {}}
    window_rows: list[dict[str, Any]] = []
    feature_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    skipped_high_n = 0

    for ordinal, window in enumerate(windows, start=1):
        sequence = read_sequence(fasta, window)
        n_fraction = sequence.count("N") / max(len(sequence), 1)
        if n_fraction > args.max_n_fraction:
            skipped_high_n += 1
            continue
        target_raw = read_target(bigwig, window, center_start=center_start, center_end=center_end)
        target_ntv3 = ntv3_transform(target_raw, track_mean=track_mean)
        alpha_raw = alpha_center_prediction(
            model,
            sequence,
            window,
            center_start=center_start,
            center_end=center_end,
            output_type_name=args.alphagenome_output_type,
            ontology_term=args.ontology_term,
            multi_track_policy=args.alphagenome_multi_track_policy,
            track_index=args.alphagenome_track_index,
        )
        alpha_ntv3 = ntv3_transform(alpha_raw, track_mean=track_mean)
        features = build_features(alpha_raw, feature_config)
        if moments is not None:
            moments.update(features, target_ntv3)
        if collect_arrays:
            feature_chunks.append(features.astype(np.float32))
            target_chunks.append(target_ntv3.astype(np.float32))
        zero_raw_acc.update(alpha_raw, target_raw)
        zero_ntv3_acc.update(alpha_ntv3, target_ntv3)

        row: dict[str, Any] = {
            "split": split_name,
            "window_ordinal": ordinal,
            "window_index": window.index,
            "chrom": window.chrom,
            "start": window.start,
            "end": window.end,
            "zero_shot_pearson_ntv3_scaled": PearsonAccumulator(),
        }
        row_zero = PearsonAccumulator()
        row_zero.update(alpha_ntv3, target_ntv3)
        row["zero_shot_pearson_ntv3_scaled"] = row_zero.compute()
        for ridge_lambda, weights in (weights_by_lambda or {}).items():
            prediction = features @ weights
            readout_accs[ridge_lambda].update(prediction, target_ntv3)
            row_acc = PearsonAccumulator()
            row_acc.update(prediction, target_ntv3)
            row[f"readout_pearson_lambda_{ridge_lambda:g}"] = row_acc.compute()
        window_rows.append(row)

        if args.log_every > 0 and ordinal % args.log_every == 0:
            message = (
                f"[{split_name} {ordinal}/{len(windows)}] "
                f"zero_ntv3={zero_ntv3_acc.compute():.6f}"
            )
            if readout_accs:
                best_current = max(acc.compute() for acc in readout_accs.values())
                message += f" best_readout={best_current:.6f}"
            print(message, flush=True)

    metrics: dict[str, Any] = {
        "split": split_name,
        "selected_windows": len(windows),
        "evaluated_windows": len(window_rows),
        "skipped_high_n_windows": skipped_high_n,
        "num_positions": zero_ntv3_acc.count,
        "zero_shot_pearson_raw": zero_raw_acc.compute(),
        "zero_shot_pearson_ntv3_scaled": zero_ntv3_acc.compute(),
    }
    for ridge_lambda, acc in readout_accs.items():
        metrics[f"readout_pearson_ntv3_scaled_lambda_{ridge_lambda:g}"] = acc.compute()

    arrays = None
    if collect_arrays and feature_chunks:
        arrays = (np.concatenate(feature_chunks, axis=0), np.concatenate(target_chunks, axis=0))
    return metrics, window_rows, arrays


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_config = FeatureConfig(
        include_raw=not args.no_raw_feature,
        include_sqrt=not args.no_sqrt_feature,
        include_log1p=not args.no_log1p_feature,
        smooth_windows=parse_int_list(args.smooth_windows),
    )
    names = feature_names(feature_config)
    ridge_lambdas = parse_float_list(args.ridge_lambdas)

    track_info = load_track_info(args.dataset_root, args.species, args.track_id)
    center_start, center_end = center_bounds(args.sequence_length, args.keep_target_center_fraction)
    split_counts: dict[str, int] = {}
    train_total, train_windows = load_selected_windows(
        args.dataset_root,
        args.species,
        "train",
        sequence_length=args.sequence_length,
        overlap=args.overlap,
        max_windows=args.train_windows,
        sampling=args.window_sampling,
    )
    val_total, val_windows = load_selected_windows(
        args.dataset_root,
        args.species,
        "val",
        sequence_length=args.sequence_length,
        overlap=args.overlap,
        max_windows=args.val_windows,
        sampling=args.window_sampling,
    )
    test_total, test_windows = load_selected_windows(
        args.dataset_root,
        args.species,
        "test",
        sequence_length=args.sequence_length,
        overlap=args.overlap,
        max_windows=args.test_windows,
        sampling=args.window_sampling,
    )
    split_counts.update(train=train_total, val=val_total, test=test_total)

    config: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "species": args.species,
        "track_info": {**asdict(track_info), "path": str(track_info.path)},
        "sequence_length": args.sequence_length,
        "keep_target_center_fraction": args.keep_target_center_fraction,
        "target_center_start": center_start,
        "target_center_end": center_end,
        "target_length": center_end - center_start,
        "ontology_term": args.ontology_term,
        "alphagenome_output_type": args.alphagenome_output_type,
        "alphagenome_model_version": args.alphagenome_model_version,
        "alphagenome_source": args.alphagenome_source,
        "feature_names": names,
        "ridge_lambdas": ridge_lambdas,
        "split_total_windows": split_counts,
        "split_selected_windows": {
            "train": len(train_windows),
            "val": len(val_windows),
            "test": len(test_windows),
        },
        "window_sampling": args.window_sampling,
    }
    write_json(args.output_dir / "config.json", config)

    fasta = Fasta(str(args.dataset_root / args.species / "genome.fasta"), as_raw=True, sequence_always_upper=True)
    bigwig = pyBigWig.open(str(track_info.path))
    model = create_alphagenome_model(source=args.alphagenome_source, model_version=args.alphagenome_model_version)

    train_moments = StreamingMoments.create(len(names))
    train_metrics, train_rows, _ = stream_split(
        split_name="train",
        windows=train_windows,
        model=model,
        fasta=fasta,
        bigwig=bigwig,
        track_mean=track_info.mean,
        center_start=center_start,
        center_end=center_end,
        feature_config=feature_config,
        args=args,
        moments=train_moments,
    )

    weights_by_lambda = {ridge_lambda: solve_ridge(train_moments, ridge_lambda) for ridge_lambda in ridge_lambdas}
    val_metrics, val_rows, val_arrays = stream_split(
        split_name="val",
        windows=val_windows,
        model=model,
        fasta=fasta,
        bigwig=bigwig,
        track_mean=track_info.mean,
        center_start=center_start,
        center_end=center_end,
        feature_config=feature_config,
        args=args,
        weights_by_lambda=weights_by_lambda,
        collect_arrays=args.save_val_arrays,
    )
    best_lambda = max(
        ridge_lambdas,
        key=lambda value: float(val_metrics[f"readout_pearson_ntv3_scaled_lambda_{value:g}"]),
    )
    best_weights = {best_lambda: weights_by_lambda[best_lambda]}
    test_metrics, test_rows, _ = stream_split(
        split_name="test",
        windows=test_windows,
        model=model,
        fasta=fasta,
        bigwig=bigwig,
        track_mean=track_info.mean,
        center_start=center_start,
        center_end=center_end,
        feature_config=feature_config,
        args=args,
        weights_by_lambda=best_weights,
    )

    if val_arrays is not None:
        np.savez_compressed(
            args.output_dir / "val_readout_arrays.npz",
            features=val_arrays[0].astype(np.float32),
            target=val_arrays[1].astype(np.float32),
        )

    weights_payload = {
        "feature_names": names,
        "selected_lambda": best_lambda,
        "weights_by_lambda": {
            f"{ridge_lambda:g}": weights.astype(float).tolist()
            for ridge_lambda, weights in weights_by_lambda.items()
        },
    }
    write_json(args.output_dir / "readout_weights.json", weights_payload)
    write_csv(args.output_dir / "window_scores.csv", [*train_rows, *val_rows, *test_rows])

    result = {
        "status": "completed",
        "config": config,
        "selected_lambda": best_lambda,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
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
    write_json(args.output_dir / "metrics.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
