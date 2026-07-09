from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.ntv3.dataset import (  # noqa: E402
    GenomeBigWigDataset,
    create_functional_targets_scaler,
    load_species_assets,
)
from eval.ntv3.heads import FunctionalTracksModel  # noqa: E402
from src.constants import DNA_VOCAB  # noqa: E402
from src.models.beat_v7.model import BeatV7Config, DNAFoundationBeatV7  # noqa: E402

DNA_BASE_IDS = tuple(DNA_VOCAB[base] for base in ("A", "C", "G", "T"))
N_ID = DNA_VOCAB["N"]


def _load_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected checkpoint bundle dict at {path}, got {type(bundle).__name__}.")
    if "model_state_dict" not in bundle:
        raise KeyError(f"Checkpoint bundle at {path} does not contain model_state_dict.")
    return bundle


def _build_model_from_checkpoint(
    *,
    state_dict: dict[str, torch.Tensor],
    num_tracks: int,
    checkpoint_config: dict[str, Any],
    device: torch.device,
) -> FunctionalTracksModel:
    backbone = DNAFoundationBeatV7(BeatV7Config(activation_checkpointing=False))
    model = FunctionalTracksModel(
        backbone,
        embed_dim=backbone.cfg.d_model,
        num_tracks=num_tracks,
        keep_target_center_fraction=float(checkpoint_config.get("keep_target_center_fraction", 0.375)),
        feature_source=str(checkpoint_config.get("feature_source", "hidden")),
        head_type=str(checkpoint_config.get("functional_head_type", "gated-hybrid")),
        head_hidden_dim=checkpoint_config.get("functional_head_hidden_dim"),
        head_dropout=float(checkpoint_config.get("functional_head_dropout", 0.05)),
        head_kernel_size=int(checkpoint_config.get("functional_head_kernel_size", 15)),
        head_aux_features=str(checkpoint_config.get("functional_head_aux_features", "phylo-structure")),
        head_aux_projection_dim=int(checkpoint_config.get("functional_head_aux_projection_dim", 16)),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "backbone.aa_head.",
        "backbone.codon_phylo_head.",
        "backbone.mutation_effect_head.",
        "backbone.codon_head.",
        "backbone.global_proj.",
    )
    disallowed_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if disallowed_missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={disallowed_missing}, unexpected={list(unexpected)}.")
    model.to(device)
    model.eval()
    return model


def _chunked(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for offset in range(0, len(items), chunk_size):
        yield items[offset : offset + chunk_size]


def _select_anchor_positions(
    *,
    input_ids: torch.Tensor,
    target_start: int,
    target_end: int,
    anchors_per_window: int,
    generator: torch.Generator,
    anchor_region: str,
) -> list[int]:
    seq_len = int(input_ids.numel())
    if anchor_region == "target":
        start, end = target_start, target_end
    elif anchor_region == "full":
        start, end = 0, seq_len
    else:
        raise ValueError(f"Unsupported anchor_region={anchor_region!r}.")

    candidate_ids = input_ids[start:end]
    valid_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
    for base_id in DNA_BASE_IDS:
        valid_mask |= candidate_ids == base_id
    candidates = torch.nonzero(valid_mask, as_tuple=False).flatten() + start
    if candidates.numel() == 0:
        return []
    if candidates.numel() <= anchors_per_window:
        return [int(value) for value in candidates.tolist()]
    permutation = torch.randperm(candidates.numel(), generator=generator)[:anchors_per_window]
    return sorted(int(value) for value in candidates[permutation].tolist())


def _mutation_ids(original_id: int, mode: str) -> list[int]:
    if mode == "n":
        return [N_ID]
    if mode == "alt3-mean":
        return [base_id for base_id in DNA_BASE_IDS if base_id != original_id]
    raise ValueError(f"Unsupported mutation_mode={mode!r}.")


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _first_crossing_distance(
    effects: list[float],
    *,
    threshold: float,
    bin_size: int,
    consecutive_bins: int = 3,
) -> int | None:
    if not effects:
        return None
    for index in range(1, len(effects) - consecutive_bins + 1):
        window = effects[index : index + consecutive_bins]
        if all(value <= threshold for value in window):
            return index * bin_size
    return None


def _last_above_distance(effects: list[float], *, threshold: float, bin_size: int) -> int | None:
    for index in range(len(effects) - 1, -1, -1):
        if effects[index] >= threshold:
            return index * bin_size
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _maybe_write_plot(path: Path, profile_rows: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return None

    by_assay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        by_assay[str(row["assay_type"])].append(row)

    plt.figure(figsize=(10, 6))
    for assay, rows in sorted(by_assay.items()):
        rows = sorted(rows, key=lambda row: int(row["distance_bin_start_bp"]))
        x = [int(row["distance_bin_center_bp"]) for row in rows]
        y = [float(row["effect_mean"]) for row in rows]
        plt.plot(x, y, label=assay, linewidth=1.8)
    plt.xlabel("Distance from mutated anchor (bp)")
    plt.ylabel("Mean absolute prediction change")
    plt.title("Beat-v7 NTv3 effective receptive field by assay")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets/ntv3"))
    parser.add_argument("--species", default="human")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--anchors-per-window", type=int, default=16)
    parser.add_argument("--mutation-batch-size", type=int, default=2)
    parser.add_argument("--mutation-mode", choices=("n", "alt3-mean"), default="n")
    parser.add_argument("--anchor-region", choices=("target", "full"), default="target")
    parser.add_argument("--distance-bin-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_windows <= 0:
        raise ValueError("--num-windows must be positive.")
    if args.anchors_per_window <= 0:
        raise ValueError("--anchors-per-window must be positive.")
    if args.mutation_batch_size <= 0:
        raise ValueError("--mutation-batch-size must be positive.")
    if args.distance_bin_size <= 0:
        raise ValueError("--distance-bin-size must be positive.")

    start_time = time.perf_counter()
    device = torch.device(args.device)
    bundle = _load_bundle(args.checkpoint)
    state_dict = bundle["model_state_dict"]
    checkpoint_config = bundle.get("config") or {}
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = asdict(checkpoint_config)

    assets = load_species_assets(args.dataset_root, args.species)
    track_infos = list(assets.functional_tracks)
    assay_to_track_indices: dict[str, list[int]] = defaultdict(list)
    for index, track in enumerate(track_infos):
        assay_to_track_indices[track.assay_type].append(index)

    sequence_length = int(checkpoint_config.get("sequence_length", 32768))
    keep_fraction = float(checkpoint_config.get("keep_target_center_fraction", 0.375))
    target_offset = int(sequence_length * (1.0 - keep_fraction) // 2)
    target_length = sequence_length - 2 * target_offset
    target_start = target_offset
    target_end = target_offset + target_length
    num_bins = math.ceil(sequence_length / args.distance_bin_size)

    dataset = GenomeBigWigDataset(
        fasta_path=assets.fasta_path,
        split_regions=assets.split_regions,
        track_infos=track_infos,
        split="val",
        sequence_length=sequence_length,
        transform_fn=create_functional_targets_scaler(track_infos),
        keep_target_center_fraction=keep_fraction,
        limit_num_samples=args.num_windows,
        seed=int(checkpoint_config.get("seed", args.seed)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = _build_model_from_checkpoint(
        state_dict=state_dict,
        num_tracks=len(track_infos),
        checkpoint_config=checkpoint_config,
        device=device,
    )

    assay_sums = {assay: torch.zeros(num_bins, dtype=torch.float64) for assay in assay_to_track_indices}
    assay_counts = {assay: torch.zeros(num_bins, dtype=torch.float64) for assay in assay_to_track_indices}
    track_sums = torch.zeros((len(track_infos), num_bins), dtype=torch.float64)
    track_counts = torch.zeros((len(track_infos), num_bins), dtype=torch.float64)
    anchor_records: list[dict[str, Any]] = []

    target_positions = torch.arange(target_start, target_end, device=device)
    generator = torch.Generator().manual_seed(args.seed)
    windows_seen = 0
    total_anchor_mutations = 0
    total_forward_batches = 0

    with torch.no_grad():
        for window_index, batch in enumerate(loader):
            if windows_seen >= args.num_windows:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            baseline = model(input_ids=input_ids, attention_mask=attention_mask).detach()
            anchor_positions = _select_anchor_positions(
                input_ids=input_ids[0].cpu(),
                target_start=target_start,
                target_end=target_end,
                anchors_per_window=args.anchors_per_window,
                generator=generator,
                anchor_region=args.anchor_region,
            )
            if not anchor_positions:
                continue
            windows_seen += 1

            for anchor_chunk in _chunked(anchor_positions, args.mutation_batch_size):
                mutated_inputs: list[torch.Tensor] = []
                anchor_for_mutation: list[int] = []
                for anchor_position in anchor_chunk:
                    original_id = int(input_ids[0, anchor_position].item())
                    for mutation_id in _mutation_ids(original_id, args.mutation_mode):
                        mutated = input_ids[0].clone()
                        mutated[anchor_position] = mutation_id
                        mutated_inputs.append(mutated)
                        anchor_for_mutation.append(anchor_position)
                mutated_batch = torch.stack(mutated_inputs, dim=0)
                mutated_attention_mask = attention_mask.expand(mutated_batch.shape[0], -1)
                mutated_predictions = model(input_ids=mutated_batch, attention_mask=mutated_attention_mask).detach()
                total_forward_batches += 1

                grouped_predictions: dict[int, list[torch.Tensor]] = defaultdict(list)
                for mutation_index, anchor_position in enumerate(anchor_for_mutation):
                    grouped_predictions[anchor_position].append(mutated_predictions[mutation_index])

                for anchor_position, predictions in grouped_predictions.items():
                    mean_mutated = torch.stack(predictions, dim=0).mean(dim=0, keepdim=True)
                    diff = (mean_mutated - baseline).abs()[0]
                    distances = (target_positions - anchor_position).abs()
                    bins = torch.div(distances, args.distance_bin_size, rounding_mode="floor").clamp(max=num_bins - 1)
                    bins_cpu = bins.cpu()

                    for track_index in range(len(track_infos)):
                        values = diff[:, track_index].to(torch.float64).cpu()
                        track_sums[track_index].scatter_add_(0, bins_cpu, values)
                        track_counts[track_index].scatter_add_(0, bins_cpu, torch.ones_like(values))

                    for assay, track_indices in assay_to_track_indices.items():
                        assay_values = diff[:, track_indices].mean(dim=-1).to(torch.float64).cpu()
                        assay_sums[assay].scatter_add_(0, bins_cpu, assay_values)
                        assay_counts[assay].scatter_add_(0, bins_cpu, torch.ones_like(assay_values))

                    anchor_records.append(
                        {
                            "window_index": window_index,
                            "anchor_position": anchor_position,
                            "original_id": int(input_ids[0, anchor_position].item()),
                            "num_mutations": len(predictions),
                        }
                    )
                    total_anchor_mutations += len(predictions)

    profile_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for assay in sorted(assay_to_track_indices):
        sums = assay_sums[assay]
        counts = assay_counts[assay]
        means = torch.where(counts > 0, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
        mean_values = [float(value) for value in means.tolist()]
        effect0 = mean_values[0] if mean_values else 0.0
        threshold10 = effect0 * 0.10
        threshold05 = effect0 * 0.05
        first10 = _first_crossing_distance(mean_values, threshold=threshold10, bin_size=args.distance_bin_size)
        first05 = _first_crossing_distance(mean_values, threshold=threshold05, bin_size=args.distance_bin_size)
        last10 = _last_above_distance(mean_values, threshold=threshold10, bin_size=args.distance_bin_size)
        last05 = _last_above_distance(mean_values, threshold=threshold05, bin_size=args.distance_bin_size)
        auc = sum(mean_values) * args.distance_bin_size
        summary_rows.append(
            {
                "assay_type": assay,
                "num_tracks": len(assay_to_track_indices[assay]),
                "effect_at_0": effect0,
                "threshold_10pct": threshold10,
                "threshold_5pct": threshold05,
                "rf_10pct_first_crossing_bp": first10,
                "rf_5pct_first_crossing_bp": first05,
                "rf_10pct_last_above_bp": last10,
                "rf_5pct_last_above_bp": last05,
                "effect_auc": auc,
            }
        )
        for bin_index, effect_mean in enumerate(mean_values):
            count = float(counts[bin_index].item())
            if count <= 0:
                continue
            bin_start = bin_index * args.distance_bin_size
            bin_end = min(sequence_length - 1, (bin_index + 1) * args.distance_bin_size - 1)
            profile_rows.append(
                {
                    "assay_type": assay,
                    "distance_bin_start_bp": bin_start,
                    "distance_bin_end_bp": bin_end,
                    "distance_bin_center_bp": (bin_start + bin_end) // 2,
                    "effect_mean": effect_mean,
                    "effect_relative_to_bin0": effect_mean / effect0 if effect0 > 0 else None,
                    "count": int(count),
                }
            )

    track_rows: list[dict[str, Any]] = []
    for track_index, track in enumerate(track_infos):
        means = torch.where(
            track_counts[track_index] > 0,
            track_sums[track_index] / track_counts[track_index].clamp_min(1.0),
            torch.zeros_like(track_sums[track_index]),
        )
        effect0 = float(means[0].item())
        for bin_index, effect_mean in enumerate(float(value) for value in means.tolist()):
            count = float(track_counts[track_index, bin_index].item())
            if count <= 0:
                continue
            bin_start = bin_index * args.distance_bin_size
            bin_end = min(sequence_length - 1, (bin_index + 1) * args.distance_bin_size - 1)
            track_rows.append(
                {
                    "track_id": track.dataset_id,
                    "assay_type": track.assay_type,
                    "distance_bin_start_bp": bin_start,
                    "distance_bin_end_bp": bin_end,
                    "distance_bin_center_bp": (bin_start + bin_end) // 2,
                    "effect_mean": effect_mean,
                    "effect_relative_to_bin0": effect_mean / effect0 if effect0 > 0 else None,
                    "count": int(count),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_csv = args.output_dir / "effective_rf_assay_profile.csv"
    summary_csv = args.output_dir / "effective_rf_assay_summary.csv"
    track_csv = args.output_dir / "effective_rf_track_profile.csv"
    anchors_csv = args.output_dir / "effective_rf_anchors.csv"
    plot_png = args.output_dir / "effective_rf_assay_profile.png"
    report_json = args.output_dir / "effective_rf_report.json"
    _write_csv(profile_csv, profile_rows)
    _write_csv(summary_csv, summary_rows)
    _write_csv(track_csv, track_rows)
    _write_csv(anchors_csv, anchor_records)
    plot_path = _maybe_write_plot(plot_png, profile_rows)

    elapsed_seconds = time.perf_counter() - start_time
    requested_anchor_count = args.num_windows * args.anchors_per_window
    payload = {
        "checkpoint": str(args.checkpoint),
        "species": args.species,
        "step": bundle.get("step"),
        "metric_value": bundle.get("metric_value"),
        "device": str(device),
        "num_windows_requested": args.num_windows,
        "num_windows_seen": windows_seen,
        "anchors_per_window": args.anchors_per_window,
        "anchor_region": args.anchor_region,
        "requested_anchor_count": requested_anchor_count,
        "observed_anchor_count": len(anchor_records),
        "total_anchor_mutations": total_anchor_mutations,
        "total_forward_batches": total_forward_batches,
        "mutation_mode": args.mutation_mode,
        "mutation_batch_size": args.mutation_batch_size,
        "distance_bin_size": args.distance_bin_size,
        "sequence_length": sequence_length,
        "target_start": target_start,
        "target_end": target_end,
        "target_length": target_length,
        "elapsed_seconds": elapsed_seconds,
        "estimated_seconds_for_64x128_same_settings": (
            elapsed_seconds * (64 * 128) / max(len(anchor_records), 1)
            if args.mutation_mode == "n"
            else elapsed_seconds * (64 * 128 * 3) / max(total_anchor_mutations, 1)
        ),
        "assay_profile_csv": str(profile_csv),
        "assay_summary_csv": str(summary_csv),
        "track_profile_csv": str(track_csv),
        "anchors_csv": str(anchors_csv),
        "plot_png": plot_path,
        "assay_summary": summary_rows,
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
