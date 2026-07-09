from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
from eval.ntv3.losses import poisson_multinomial_loss  # noqa: E402
from eval.ntv3.metrics import FunctionalTracksMetrics  # noqa: E402
from src.models.beat_v7.model import BeatV7Config, DNAFoundationBeatV7  # noqa: E402


def _load_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected checkpoint bundle dict at {path}, got {type(bundle).__name__}.")
    if "model_state_dict" not in bundle:
        raise KeyError(f"Checkpoint bundle at {path} does not contain model_state_dict.")
    return bundle


def _load_dataset_scores(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset_id = row.get("datasets")
            metric = row.get("Metric")
            if dataset_id and metric:
                scores[dataset_id] = float(metric)
    return scores


def _extract_gate_rows(
    *,
    state_dict: dict[str, torch.Tensor],
    track_ids: list[str],
    assay_by_track: dict[str, str],
    track_name_by_track: dict[str, str],
    test_scores: dict[str, float],
) -> list[dict[str, Any]]:
    gate_logits = state_dict.get("head.track_gate_logits")
    if gate_logits is None:
        raise KeyError("Could not find head.track_gate_logits in checkpoint state_dict.")
    if gate_logits.ndim != 1:
        raise ValueError(f"Expected 1-D gate logits, got shape={tuple(gate_logits.shape)}.")
    if gate_logits.numel() != len(track_ids):
        raise ValueError(f"Gate count {gate_logits.numel()} does not match track count {len(track_ids)}.")

    gate_values = torch.sigmoid(gate_logits.float()).tolist()
    rows: list[dict[str, Any]] = []
    for index, (track_id, gate_logit, gate_value) in enumerate(
        zip(track_ids, gate_logits.float().tolist(), gate_values, strict=True)
    ):
        rows.append(
            {
                "track_index": index,
                "track_id": track_id,
                "assay_type": assay_by_track[track_id],
                "track_name_clean": track_name_by_track[track_id],
                "gate_logit": gate_logit,
                "gate_value": gate_value,
                "test_pearson": test_scores.get(track_id),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _summarize_by_assay(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["assay_type"])].append(row)

    summaries: list[dict[str, Any]] = []
    for assay_type, assay_rows in sorted(grouped.items()):
        logits = [float(row["gate_logit"]) for row in assay_rows]
        gates = [float(row["gate_value"]) for row in assay_rows]
        pearsons = [float(row["test_pearson"]) for row in assay_rows if row["test_pearson"] is not None]
        summaries.append(
            {
                "assay_type": assay_type,
                "num_tracks": len(assay_rows),
                "gate_logit_mean": _mean(logits),
                "gate_logit_std": _std(logits),
                "gate_logit_min": min(logits),
                "gate_logit_max": max(logits),
                "gate_value_mean": _mean(gates),
                "gate_value_std": _std(gates),
                "gate_value_min": min(gates),
                "gate_value_max": max(gates),
                "test_pearson_mean": _mean(pearsons) if pearsons else None,
            }
        )
    return summaries


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
    unexpected_keys = list(unexpected)
    disallowed_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch: missing={disallowed_missing}, unexpected={unexpected_keys}."
        )
    model.to(device)
    model.eval()
    return model


def _run_validation_forward(
    *,
    checkpoint_config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    dataset_root: Path,
    species_name: str,
    num_batches: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if num_batches <= 0:
        return None

    assets = load_species_assets(dataset_root, species_name)
    track_infos = list(assets.functional_tracks)
    transform_fn = create_functional_targets_scaler(track_infos)
    dataset = GenomeBigWigDataset(
        fasta_path=assets.fasta_path,
        split_regions=assets.split_regions,
        track_infos=track_infos,
        split="val",
        sequence_length=int(checkpoint_config.get("sequence_length", 32768)),
        transform_fn=transform_fn,
        keep_target_center_fraction=float(checkpoint_config.get("keep_target_center_fraction", 0.375)),
        limit_num_samples=max(num_batches * batch_size, 1),
        seed=int(checkpoint_config.get("seed", 0)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _build_model_from_checkpoint(
        state_dict=state_dict,
        num_tracks=len(track_infos),
        checkpoint_config=checkpoint_config,
        device=device,
    )
    metrics = FunctionalTracksMetrics(track_ids=[track.dataset_id for track in track_infos])
    batches_seen = 0
    with torch.no_grad():
        for batch in loader:
            if batches_seen >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = poisson_multinomial_loss(predictions, targets)
            metrics.update(predictions, targets, float(loss.item()))
            batches_seen += 1
    computed = metrics.compute()
    return {
        "device": str(device),
        "batches": batches_seen,
        "batch_size": batch_size,
        "examples": batches_seen * batch_size,
        "mean_pearson": computed["mean/pearson"],
        "loss": computed["loss"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets/ntv3"))
    parser.add_argument("--species", default="human")
    parser.add_argument("--dataset-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forward-val-batches", type=int, default=0)
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = _load_bundle(args.checkpoint)
    state_dict = bundle["model_state_dict"]
    checkpoint_config = bundle.get("config") or {}
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = asdict(checkpoint_config)

    assets = load_species_assets(args.dataset_root, args.species)
    track_infos = list(assets.functional_tracks)
    track_ids = [track.dataset_id for track in track_infos]
    assay_by_track = {track.dataset_id: track.assay_type for track in track_infos}
    track_name_by_track = {track.dataset_id: track.track_name_clean for track in track_infos}

    gate_rows = _extract_gate_rows(
        state_dict=state_dict,
        track_ids=track_ids,
        assay_by_track=assay_by_track,
        track_name_by_track=track_name_by_track,
        test_scores=_load_dataset_scores(args.dataset_scores),
    )
    assay_rows = _summarize_by_assay(gate_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    track_csv = args.output_dir / "gated_hybrid_track_gates.csv"
    assay_csv = args.output_dir / "gated_hybrid_assay_gate_summary.csv"
    report_json = args.output_dir / "gated_hybrid_gate_report.json"
    _write_csv(track_csv, gate_rows)
    _write_csv(assay_csv, assay_rows)

    device = torch.device(args.device)
    forward_summary = _run_validation_forward(
        checkpoint_config=checkpoint_config,
        state_dict=state_dict,
        dataset_root=args.dataset_root,
        species_name=args.species,
        num_batches=args.forward_val_batches,
        batch_size=args.forward_batch_size,
        device=device,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "species": args.species,
        "step": bundle.get("step"),
        "metric_value": bundle.get("metric_value"),
        "num_tracks": len(gate_rows),
        "gate_value_mean": _mean([float(row["gate_value"]) for row in gate_rows]),
        "gate_value_std": _std([float(row["gate_value"]) for row in gate_rows]),
        "gate_value_min": min(float(row["gate_value"]) for row in gate_rows),
        "gate_value_max": max(float(row["gate_value"]) for row in gate_rows),
        "track_gates_csv": str(track_csv),
        "assay_summary_csv": str(assay_csv),
        "assay_summary": assay_rows,
        "validation_forward": forward_summary,
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
