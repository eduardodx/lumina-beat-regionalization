from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.constants import (
    AA_NON_CDS,
    CDS_PHASE_NONE,
    DEFAULT_CHROMOSOMES,
    DNA_VOCAB,
    MASK_ID,
    NUM_AA_CLASSES,
    NUM_STRUCTURE_CLASSES,
    PAD_ID,
    REGION_CDS,
    REGION_INTRON,
    REGION_UTR,
    STRUCT_BACKGROUND,
    STRUCT_SPLICE_CORE,
    VOCAB_SIZE,
)
from src.dataset import HG38SplicePhyloDataset, MultiTaskCollator
from src.defaults import (
    DEFAULT_FASTA_PATH,
    DEFAULT_GTF_PATH,
    DEFAULT_PHYLO100_BW_PATH,
    DEFAULT_PHYLO470_BW_PATH,
)
from src.encode_tracks import EncodeTrackSpec
from src.metrics import METRIC_STAT_KEYS, MetricAccumulator
from src.model_utils import count_parameters
from src.models import build_registered_model, registered_model_keys
from src.objectives import compute_multitask_loss
from src.train import (
    TrainConfig,
    load_yaml_train_config,
    normalize_train_config_overrides,
    resolve_encode_track_specs,
    resolve_train_config,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_stats(name: str, x: torch.Tensor, max_unique: int = 10) -> None:
    x_cpu = x.detach().cpu()
    msg = [f"{name}: shape={tuple(x_cpu.shape)} dtype={x_cpu.dtype}"]
    if x_cpu.numel() > 0:
        if x_cpu.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            msg.append(
                f"min={x_cpu.min().item():.4f} max={x_cpu.max().item():.4f} mean={x_cpu.float().mean().item():.4f}"
            )
        else:
            uniques = torch.unique(x_cpu)
            preview = uniques[:max_unique].tolist()
            msg.append(f"unique[:{max_unique}]={preview} n_unique={len(uniques)}")
    print(" | ".join(msg))


def discover_encode_track_specs(root: Path, limit: int | None = None) -> list[dict[str, str]]:
    candidates = sorted(
        list(root.rglob("*.bw")) + list(root.rglob("*.bigWig")),
        key=lambda path: str(path),
    )
    if limit is not None:
        candidates = candidates[:limit]

    specs: list[dict[str, str]] = []
    for path in candidates:
        specs.append(
            {
                "name": path.stem,
                "bw_path": str(path),
                "transform": "asinh",
                "normalize": "per_chromosome_zscore",
            }
        )
    return specs


def build_dataset(
    fasta_path: str,
    phylo100_bw_path: str,
    phylo470_bw_path: str,
    gtf_path: str,
    seq_len: int,
    encode_track_specs: list[EncodeTrackSpec] | None = None,
    clinvar_blocklist_bed_path: str | None = None,
) -> HG38SplicePhyloDataset:
    return HG38SplicePhyloDataset(
        fasta_path=fasta_path,
        phylo100_bw_path=phylo100_bw_path,
        phylo470_bw_path=phylo470_bw_path,
        gtf_path=gtf_path,
        seq_len=seq_len,
        chromosomes=DEFAULT_CHROMOSOMES,
        core_radius=2,
        region_radius=10,
        encode_track_specs=encode_track_specs,
        clinvar_blocklist_bed_path=clinvar_blocklist_bed_path,
    )


def build_loader(
    dataset: HG38SplicePhyloDataset,
    cfg: TrainConfig,
    device: torch.device,
    encode_track_specs: list[EncodeTrackSpec] | None = None,
) -> DataLoader:
    collator = MultiTaskCollator(
        mask_prob=cfg.mask_prob,
        mean_span_len=cfg.mean_span_len,
        conservation_mix=cfg.conservation_mix,
        hard_position_mix=cfg.hard_position_mix,
        counterfactual_fraction=cfg.counterfactual_fraction,
        encode_track_names=[spec.name for spec in encode_track_specs or []],
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )


def build_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    return build_registered_model(cfg.model, cfg.model_config).to(device)


def check_sample(dataset: HG38SplicePhyloDataset) -> None:
    print("\n=== DATASET SAMPLE CHECK ===")
    sample = dataset.sample()

    expected_keys = {
        "input_ids",
        "phylo100",
        "phylo470",
        "structure_labels",
        "region_labels",
        "aa_labels",
        "cds_phase",
        "codon_phylo_target",
        "chrom",
        "start",
        "end",
        "n_fraction",
        "splice_positive_fraction",
        "splice_core_fraction",
        "exon_fraction",
        "cds_fraction",
        "utr_fraction",
        "intron_fraction",
        "n_filter_fallback_used",
    }
    missing = expected_keys - set(sample.keys())
    assert not missing, f"Missing keys in dataset sample: {missing}"

    print(f"chrom={sample['chrom']} start={sample['start']} end={sample['end']}")
    tensor_stats("input_ids", sample["input_ids"])
    tensor_stats("phylo100", sample["phylo100"])
    tensor_stats("phylo470", sample["phylo470"])
    tensor_stats("structure_labels", sample["structure_labels"])
    tensor_stats("region_labels", sample["region_labels"])
    tensor_stats("aa_labels", sample["aa_labels"])
    tensor_stats("cds_phase", sample["cds_phase"])
    tensor_stats("codon_phylo_target", sample["codon_phylo_target"])

    seq_len = sample["input_ids"].shape[0]
    assert sample["phylo100"].shape == (seq_len,)
    assert sample["phylo470"].shape == (seq_len,)
    assert sample["structure_labels"].shape == (seq_len,)
    assert sample["region_labels"].shape == (seq_len,)
    assert sample["aa_labels"].shape == (seq_len,)
    assert sample["cds_phase"].shape == (seq_len,)
    assert sample["codon_phylo_target"].shape == (seq_len,)
    assert torch.all((sample["cds_phase"] >= CDS_PHASE_NONE) & (sample["cds_phase"] <= 2))
    assert torch.all(sample["aa_labels"] >= AA_NON_CDS)

    sample_n_fraction = float((sample["input_ids"] == DNA_VOCAB["N"]).float().mean().item())
    sample_splice_positive_fraction = float((sample["structure_labels"] != STRUCT_BACKGROUND).float().mean().item())
    sample_splice_core_fraction = float((sample["structure_labels"] == STRUCT_SPLICE_CORE).float().mean().item())
    sample_cds_fraction = float((sample["region_labels"] == REGION_CDS).float().mean().item())
    sample_utr_fraction = float((sample["region_labels"] == REGION_UTR).float().mean().item())
    sample_intron_fraction = float((sample["region_labels"] == REGION_INTRON).float().mean().item())

    assert abs(sample["n_fraction"] - sample_n_fraction) < 1e-6
    assert abs(sample["splice_positive_fraction"] - sample_splice_positive_fraction) < 1e-6
    assert abs(sample["splice_core_fraction"] - sample_splice_core_fraction) < 1e-6
    assert abs(sample["cds_fraction"] - sample_cds_fraction) < 1e-6
    assert abs(sample["utr_fraction"] - sample_utr_fraction) < 1e-6
    assert abs(sample["intron_fraction"] - sample_intron_fraction) < 1e-6
    assert 0.0 <= sample["exon_fraction"] <= 1.0
    assert isinstance(sample["n_filter_fallback_used"], bool)

    print(
        f"n_fraction={sample['n_fraction']:.4f} "
        f"splice_positive_fraction={sample['splice_positive_fraction']:.4f} "
        f"splice_core_fraction={sample['splice_core_fraction']:.4f} "
        f"exon_fraction={sample['exon_fraction']:.4f} "
        f"cds_fraction={sample['cds_fraction']:.4f} "
        f"utr_fraction={sample['utr_fraction']:.4f} "
        f"intron_fraction={sample['intron_fraction']:.4f} "
        f"n_filter_fallback_used={sample['n_filter_fallback_used']}"
    )


def check_batch(loader: DataLoader) -> dict[str, Any]:
    print("\n=== COLLATOR / BATCH CHECK ===")
    batch = next(iter(loader))

    expected_keys = {
        "input_ids",
        "alt_input_ids",
        "attention_mask",
        "alt_attention_mask",
        "mlm_labels",
        "mask_positions",
        "aux_valid_mask",
        "phylo100",
        "phylo470",
        "structure_labels",
        "region_labels",
        "aa_labels",
        "cds_phase",
        "codon_phylo_target",
        "rc_input_ids",
        "rc_attention_mask",
        "encode_track_names",
        "chroms",
        "n_fraction",
        "splice_positive_fraction",
        "splice_core_fraction",
        "exon_fraction",
        "cds_fraction",
        "utr_fraction",
        "intron_fraction",
        "n_filter_fallback_fraction",
        "mask_density_intergenic",
        "mask_density_intron",
        "mask_density_noncoding_exon",
        "mask_density_utr",
        "mask_density_cds",
    }
    missing = expected_keys - set(batch.keys())
    assert not missing, f"Missing keys in batch: {missing}"

    for key, value in batch.items():
        if torch.is_tensor(value):
            tensor_stats(key, value)

    batch_size, seq_len = batch["input_ids"].shape
    assert batch["attention_mask"].shape == (batch_size, seq_len)
    assert batch["alt_input_ids"].shape == (batch_size, seq_len)
    assert batch["alt_attention_mask"].shape == (batch_size, seq_len)
    assert batch["mlm_labels"].shape == (batch_size, seq_len)
    assert batch["mask_positions"].shape == (batch_size, seq_len)
    assert batch["aux_valid_mask"].shape == (batch_size, seq_len)
    assert batch["phylo100"].shape == (batch_size, seq_len)
    assert batch["phylo470"].shape == (batch_size, seq_len)
    assert batch["structure_labels"].shape == (batch_size, seq_len)
    assert batch["region_labels"].shape == (batch_size, seq_len)
    assert batch["aa_labels"].shape == (batch_size, seq_len)
    assert batch["cds_phase"].shape == (batch_size, seq_len)
    assert batch["codon_phylo_target"].shape == (batch_size, seq_len)
    assert batch["rc_input_ids"].shape == (batch_size, seq_len)
    assert batch["rc_attention_mask"].shape == (batch_size, seq_len)
    assert len(batch["chroms"]) == batch_size
    assert len(batch["encode_track_names"]) == int(batch["encode_targets"].shape[-1])

    masked_count = batch["mask_positions"].sum().item()
    supervised_count = (batch["mlm_labels"] != PAD_ID).sum().item()
    aux_valid_count = batch["aux_valid_mask"].sum().item()
    print(
        f"masked_count={masked_count} supervised_count={supervised_count} "
        f"aux_valid_count={aux_valid_count} chroms={batch['chroms']}"
    )
    assert masked_count == supervised_count, "mask_positions and mlm_labels supervision disagree"
    assert masked_count > 0, "No masked tokens found in batch"
    assert aux_valid_count > 0, "No valid auxiliary positions found in batch"

    masked_input_values = batch["input_ids"][batch["mask_positions"]]
    assert torch.all(masked_input_values == MASK_ID), "Masked positions are not MASK_ID in input_ids"
    assert torch.all(batch["mlm_labels"][~batch["mask_positions"]] == PAD_ID), (
        "Non-masked positions should be PAD_ID in mlm_labels"
    )
    assert torch.all(batch["aux_valid_mask"] <= batch["attention_mask"].bool()), (
        "aux_valid_mask cannot include padding positions"
    )
    assert torch.all((batch["cds_phase"] >= CDS_PHASE_NONE) & (batch["cds_phase"] <= 2))
    assert torch.all(batch["aa_labels"] >= AA_NON_CDS)
    assert torch.all(batch["aa_labels"][batch["cds_phase"] < 0] == AA_NON_CDS)

    masked_n = batch["mask_positions"] & (batch["mlm_labels"] == DNA_VOCAB["N"])
    visible_n = (~batch["mask_positions"]) & (batch["input_ids"] == DNA_VOCAB["N"])
    masked_valid = batch["mask_positions"] & (batch["mlm_labels"] != DNA_VOCAB["N"])

    if torch.any(masked_n):
        assert not torch.any(batch["aux_valid_mask"] & masked_n), "Masked N positions should not be aux-valid"
    if torch.any(visible_n):
        assert not torch.any(batch["aux_valid_mask"] & visible_n), "Visible N positions should not be aux-valid"
    if torch.any(masked_valid):
        assert torch.all(batch["aux_valid_mask"][masked_valid]), "Masked valid bases should remain aux-valid"

    expected_n_fraction = float((~batch["aux_valid_mask"]).float().mean().item())
    expected_splice_positive_fraction = float((batch["structure_labels"] != STRUCT_BACKGROUND).float().mean().item())
    expected_splice_core_fraction = float((batch["structure_labels"] == STRUCT_SPLICE_CORE).float().mean().item())

    assert abs(batch["n_fraction"] - expected_n_fraction) < 1e-6
    assert abs(batch["splice_positive_fraction"] - expected_splice_positive_fraction) < 1e-6
    assert abs(batch["splice_core_fraction"] - expected_splice_core_fraction) < 1e-6
    assert 0.0 <= batch["exon_fraction"] <= 1.0
    assert 0.0 <= batch["cds_fraction"] <= 1.0
    assert 0.0 <= batch["utr_fraction"] <= 1.0
    assert 0.0 <= batch["intron_fraction"] <= 1.0
    assert 0.0 <= batch["n_filter_fallback_fraction"] <= 1.0

    print(
        f"batch_n_fraction={batch['n_fraction']:.4f} "
        f"splice_positive_fraction={batch['splice_positive_fraction']:.4f} "
        f"splice_core_fraction={batch['splice_core_fraction']:.4f} "
        f"exon_fraction={batch['exon_fraction']:.4f} "
        f"cds_fraction={batch['cds_fraction']:.4f} "
        f"utr_fraction={batch['utr_fraction']:.4f} "
        f"intron_fraction={batch['intron_fraction']:.4f} "
        f"n_filter_fallback_fraction={batch['n_filter_fallback_fraction']:.4f} "
        f"masked_n={masked_n.sum().item()} visible_n={visible_n.sum().item()}"
    )

    return batch


def check_model_forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("\n=== MODEL FORWARD CHECK ===")
    model.eval()

    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        rc_outputs = model(input_ids=batch["rc_input_ids"], attention_mask=batch["rc_attention_mask"])

    expected_out_keys = {
        "hidden_states",
        "mlm_logits",
        "phylo100_pred",
        "phylo470_pred",
        "structure_logits",
    }
    missing = expected_out_keys - set(outputs.keys())
    assert not missing, f"Missing keys in model outputs: {missing}"

    for key, value in outputs.items():
        tensor_stats(f"outputs[{key}]", value)

    batch_size, seq_len = batch["input_ids"].shape
    model_cfg = getattr(model, "cfg", None)
    if model_cfg is None or not hasattr(model_cfg, "d_model"):
        raise AssertionError("Registered models must expose `.cfg.d_model` for shared sanity checks.")
    hidden_size = int(model_cfg.d_model)

    assert outputs["hidden_states"].shape == (batch_size, seq_len, hidden_size)
    assert outputs["mlm_logits"].shape == (batch_size, seq_len, VOCAB_SIZE)
    assert outputs["phylo100_pred"].shape == (batch_size, seq_len)
    assert outputs["phylo470_pred"].shape == (batch_size, seq_len)
    assert outputs["structure_logits"].shape == (batch_size, seq_len, NUM_STRUCTURE_CLASSES)
    if "sequence_embedding" in outputs:
        assert outputs["sequence_embedding"].shape == (batch_size, hidden_size)
        assert rc_outputs["sequence_embedding"].shape == (batch_size, hidden_size)

    num_region_classes = getattr(model_cfg, "num_region_classes", 0)
    if num_region_classes > 0:
        assert "region_logits" in outputs, "Model with num_region_classes > 0 must produce region_logits"
        assert outputs["region_logits"].shape == (batch_size, seq_len, num_region_classes)
    if hasattr(model, "aa_head") and hasattr(model, "codon_phylo_head"):
        assert "aa_logits" in outputs, "beat-v2 must produce aa_logits"
        assert "codon_phylo_pred" in outputs, "beat-v2 must produce codon_phylo_pred"
        assert outputs["aa_logits"].shape == (batch_size, seq_len, NUM_AA_CLASSES)
        assert outputs["codon_phylo_pred"].shape == (batch_size, seq_len)

    return outputs, rc_outputs


def check_loss_and_backward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> None:
    print("\n=== LOSS / BACKWARD CHECK ===")
    model.train()

    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    model_cfg = getattr(model, "cfg", None)
    has_region = getattr(model_cfg, "num_region_classes", 0) > 0
    has_sequence_embedding = hasattr(model, "global_proj") or hasattr(model, "pooled_embedding")

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_sequence_embedding=has_sequence_embedding,
    )
    rc_outputs = model(
        input_ids=batch["rc_input_ids"],
        attention_mask=batch["rc_attention_mask"],
        return_sequence_embedding=has_sequence_embedding,
    )

    w_rc = 0.05 if has_sequence_embedding and "sequence_embedding" in outputs else 0.0
    w_region = 0.25 if has_region else 0.0
    has_beat_v2_heads = "aa_logits" in outputs and "codon_phylo_pred" in outputs
    w_aa = 0.25 if has_beat_v2_heads else 0.0
    w_codon_phylo = 0.1 if has_beat_v2_heads else 0.0

    loss, stats = compute_multitask_loss(
        outputs=outputs,
        batch=batch,
        rc_outputs=rc_outputs,
        alt_outputs=alt_outputs,
        w_mlm=1.0,
        w_phylo100=0.25,
        w_phylo470=0.25,
        w_structure=0.25,
        w_rc=w_rc,
        w_region=w_region,
        w_aa=w_aa,
        w_codon_phylo=w_codon_phylo,
        rc_loss_type="cosine",
    )

    expected_stats = {
        "loss",
        "loss_mlm",
        "loss_phylo100",
        "loss_phylo470",
        "loss_structure",
        "loss_region",
        "loss_aa",
        "loss_codon_phylo",
        "loss_rc",
        "mlm_supervised_tokens",
        "aux_valid_tokens",
        "aa_cds_tokens",
        "splice_bg_tokens",
        "splice_core_tokens",
        "splice_region_tokens",
        "splice_weight_bg",
        "splice_weight_core",
        "splice_weight_region",
        "region_intergenic_tokens",
        "region_intron_tokens",
        "region_noncoding_exon_tokens",
        "region_utr_tokens",
        "region_cds_tokens",
        "region_weight_intergenic",
        "region_weight_intron",
        "region_weight_noncoding_exon",
        "region_weight_utr",
        "region_weight_cds",
    }
    missing = expected_stats - set(stats.keys())
    assert not missing, f"Missing expected loss stats: {missing}"

    print("loss stats:")
    for key, value in stats.items():
        print(f"  {key}: {float(value):.6f}")
        assert torch.isfinite(value).all(), f"{key} is not finite"

    metric_accumulator = MetricAccumulator()
    metric_accumulator.update_from_batch(outputs, batch)
    metric_stats = metric_accumulator.summary()

    print("observability metrics:")
    for key in METRIC_STAT_KEYS:
        value = metric_stats[key]
        print(f"  {key}: {value:.6f}")
        assert math.isfinite(value), f"{key} is not finite"
    for key in sorted(metric_stats):
        if key.startswith("encode_"):
            value = metric_stats[key]
            print(f"  {key}: {value:.6f}")
            assert math.isfinite(value), f"{key} is not finite"

    assert torch.isfinite(loss).all(), "Total loss is not finite"

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.data.norm().item()

    assert grad_norms, "No gradients found after backward"

    tracked_param_names = ["token_emb.weight", "mlm_head.weight"]
    blocks = getattr(model, "blocks", None)
    first_block = next(iter(blocks)) if isinstance(blocks, torch.nn.ModuleList) and len(blocks) > 0 else None
    if has_beat_v2_heads and first_block is not None and hasattr(first_block, "gate_proj"):
        tracked_param_names.append("blocks.0.gate_proj.weight")
    else:
        tracked_param_names.append("blocks.0.fuse.weight")

    for name in tracked_param_names:
        norm = grad_norms.get(name)
        if norm is None:
            print(f"grad_norm[{name}]=<not found>")
        else:
            print(f"grad_norm[{name}]={norm:.6f}")

    total_grad_norm = math.sqrt(sum(value * value for value in grad_norms.values()))
    print(f"total_grad_norm={total_grad_norm:.6f}")


def run_full_sanity_check(cfg: TrainConfig) -> None:
    print("=== START SANITY CHECK ===")
    device = get_device()
    print(f"device={device.type}")
    print(f"model={cfg.model}")
    encode_track_specs = resolve_encode_track_specs(cfg)

    dataset = build_dataset(
        fasta_path=cfg.fasta_path,
        phylo100_bw_path=cfg.phylo100_bw_path,
        phylo470_bw_path=cfg.phylo470_bw_path,
        gtf_path=cfg.gtf_path,
        seq_len=cfg.seq_len,
        encode_track_specs=encode_track_specs,
        clinvar_blocklist_bed_path=cfg.clinvar_blocklist_bed_path,
    )
    loader = build_loader(dataset, cfg=cfg, device=device, encode_track_specs=encode_track_specs)
    model = build_model(cfg, device)
    parameter_count = count_parameters(model)

    if cfg.model == "beat-v7":
        assert parameter_count <= 30_000_000, (
            f"beat-v7 parameter budget exceeded: {parameter_count:,} > 30,000,000"
        )

    print(f"trainable_params={parameter_count:,}")

    check_sample(dataset)
    batch = check_batch(loader)
    check_model_forward(model, batch, device)
    check_loss_and_backward(model, batch, device)

    print("\n=== SANITY CHECK PASSED ===")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Lumina end-to-end sanity check.")
    parser.add_argument("--config")
    parser.add_argument("--fasta-path", default=argparse.SUPPRESS)
    parser.add_argument("--phylo100-bw-path", default=argparse.SUPPRESS)
    parser.add_argument("--phylo470-bw-path", default=argparse.SUPPRESS)
    parser.add_argument("--gtf-path", default=argparse.SUPPRESS)
    parser.add_argument("--model", choices=registered_model_keys(), default=argparse.SUPPRESS)
    parser.add_argument("--seq-len", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--with-encode-tracks", action="store_true")
    parser.add_argument("--encode-root", default="data/encode")
    parser.add_argument("--encode-track-limit", type=int, default=6)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    raw_args = dict(vars(args))
    config_path = raw_args.pop("config", None)
    with_encode_tracks = bool(raw_args.pop("with_encode_tracks", False))
    encode_root = Path(str(raw_args.pop("encode_root", "data/encode"))).expanduser()
    encode_track_limit = int(raw_args.pop("encode_track_limit", 6))

    config_data = asdict(TrainConfig())
    config_data["fasta_path"] = DEFAULT_FASTA_PATH
    config_data["phylo100_bw_path"] = DEFAULT_PHYLO100_BW_PATH
    config_data["phylo470_bw_path"] = DEFAULT_PHYLO470_BW_PATH
    config_data["gtf_path"] = DEFAULT_GTF_PATH
    config_data["seq_len"] = 1024
    config_data["batch_size"] = 2
    config_data["num_workers"] = 0
    if config_path is not None:
        config_data.update(load_yaml_train_config(Path(config_path).expanduser()))

    if with_encode_tracks:
        discovered_specs = discover_encode_track_specs(encode_root, limit=encode_track_limit)
        if not discovered_specs:
            raise FileNotFoundError(
                f"No ENCODE tracks found under {encode_root}. "
                "Download them first or omit --with-encode-tracks."
            )
        config_data["encode_track_specs"] = discovered_specs
        model_config = dict(config_data.get("model_config", {}))
        model_config["num_encode_tracks"] = len(discovered_specs)
        config_data["model_config"] = model_config

    config_data.update(normalize_train_config_overrides(raw_args, source="command line"))
    return resolve_train_config(TrainConfig(**config_data))


def main() -> None:
    args = build_arg_parser().parse_args()
    run_full_sanity_check(config_from_args(args))


if __name__ == "__main__":
    main()
