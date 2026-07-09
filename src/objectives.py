import torch
import torch.nn.functional as F

from src.constants import (
    AA_NON_CDS,
    ALLELE_EFFECT_IGNORE_INDEX,
    CODON_IGNORE_INDEX,
    COUNTERFACTUAL_EFFECT_IGNORE_INDEX,
    DNA_VOCAB,
    MUTATION_EFFECT_IGNORE_INDEX,
    MUTATION_EFFECT_MISSENSE,
    MUTATION_EFFECT_STOP,
    NUM_AA_CLASSES,
    NUM_COUNTERFACTUAL_EFFECT_CLASSES,
    NUM_REGION_CLASSES,
    NUM_STRUCTURE_CLASSES,
    PAD_ID,
    REGION_CDS,
    REGION_INTERGENIC,
    REGION_INTRON,
    REGION_NONCODING_EXON,
    REGION_UTR,
    SNV_BASES,
    STRUCT_BACKGROUND,
    STRUCT_SPLICE_CORE,
    STRUCT_SPLICE_REGION,
)

MAX_SPLICE_CLASS_WEIGHT = 8.0
MAX_REGION_CLASS_WEIGHT = 4.0
MAX_AA_CLASS_WEIGHT = 4.0
COUNTERFACTUAL_EFFECT_CLASS_WEIGHTS = (
    0.2,
    1.0,
    2.0,
    2.5,
    5.0,
    5.0,
    5.0,
    8.0,
    8.0,
    2.5,
    2.0,
    1.0,
)
assert len(COUNTERFACTUAL_EFFECT_CLASS_WEIGHTS) == NUM_COUNTERFACTUAL_EFFECT_CLASSES


def _map_mlm_labels_for_logits(mlm_logits: torch.Tensor, mlm_labels: torch.Tensor) -> torch.Tensor:
    if mlm_logits.shape[-1] != len(SNV_BASES):
        return mlm_labels

    mapped = torch.full_like(mlm_labels, -100)
    for index, base in enumerate(SNV_BASES):
        mapped = torch.where(mlm_labels == DNA_VOCAB[base], torch.full_like(mapped, index), mapped)
    return mapped


def _mlm_supervised_mask(mlm_logits: torch.Tensor, mlm_labels: torch.Tensor) -> torch.Tensor:
    if mlm_logits.shape[-1] != len(SNV_BASES):
        return mlm_labels != PAD_ID
    return _map_mlm_labels_for_logits(mlm_logits, mlm_labels) != -100


def masked_ce_token(mlm_logits: torch.Tensor, mlm_labels: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, vocab_size = mlm_logits.shape
    labels = _map_mlm_labels_for_logits(mlm_logits, mlm_labels)
    ignore_index = -100 if mlm_logits.shape[-1] == len(SNV_BASES) else PAD_ID
    return F.cross_entropy(
        mlm_logits.reshape(batch_size * seq_len, vocab_size),
        labels.reshape(batch_size * seq_len),
        ignore_index=ignore_index,
    )


def phylo_weighted_ce_token(
    mlm_logits: torch.Tensor,
    mlm_labels: torch.Tensor,
    phylo_weights: torch.Tensor,
    boost: float = 2.0,
) -> torch.Tensor:
    """MLM cross-entropy with per-token PhyloP conservation weighting.

    Tokens at conserved positions (high PhyloP) receive higher loss weight,
    forcing the backbone to be especially accurate where mutations are most
    likely pathogenic.

    Weight formula: w_i = 1.0 + boost * max(0, phylo_i)
    Normalized by sum of weights over masked positions.
    """
    batch_size, seq_len, vocab_size = mlm_logits.shape
    labels = _map_mlm_labels_for_logits(mlm_logits, mlm_labels)
    ignore_index = -100 if mlm_logits.shape[-1] == len(SNV_BASES) else PAD_ID
    per_token = F.cross_entropy(
        mlm_logits.reshape(batch_size * seq_len, vocab_size),
        labels.reshape(batch_size * seq_len),
        ignore_index=ignore_index,
        reduction="none",
    )
    phylo_flat = torch.clamp(phylo_weights.reshape(batch_size * seq_len), min=0.0)
    weights = 1.0 + boost * phylo_flat
    valid = labels.reshape(batch_size * seq_len) != ignore_index
    weighted = per_token * weights
    return weighted.sum() / weights[valid].sum().clamp_min(1.0)


def valid_smooth_l1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.sum() == 0:
        # Gradient-connected zero: keeps DDP bucket alive when no valid positions.
        return (pred * 0.0).sum()
    return F.smooth_l1_loss(pred[valid_mask], target[valid_mask])


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    ignore_index: int = -100,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_mask = mask.to(dtype=torch.bool) & (labels != ignore_index)
    if valid_mask.sum() == 0:
        return (logits * 0.0).sum()
    return F.cross_entropy(
        logits[valid_mask].float(),
        labels[valid_mask].long(),
        weight=class_weights,
        ignore_index=ignore_index,
    )


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_mask = mask.to(dtype=torch.bool) & torch.isfinite(target)
    if valid_mask.sum() == 0:
        return (pred * 0.0).sum()
    return F.smooth_l1_loss(pred[valid_mask].float(), target[valid_mask].float())


def valid_smooth_l1_multitrack(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = torch.isfinite(target)
    if valid_mask.sum() == 0:
        zero = (pred * 0.0).sum()
        return zero, pred.new_zeros(pred.shape[-1] if pred.ndim > 0 else 0)

    overall = F.smooth_l1_loss(pred[valid_mask], target[valid_mask])
    per_track_losses: list[torch.Tensor] = []
    for track_index in range(pred.shape[-1]):
        track_mask = valid_mask[..., track_index]
        if track_mask.sum() == 0:
            per_track_losses.append((pred[..., track_index] * 0.0).sum())
        else:
            per_track_losses.append(
                F.smooth_l1_loss(pred[..., track_index][track_mask], target[..., track_index][track_mask])
            )
    return overall, torch.stack(per_track_losses)


def weighted_ce_structure(
    structure_logits: torch.Tensor,
    structure_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weight_cap: float = MAX_SPLICE_CLASS_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_counts = structure_logits.new_zeros(NUM_STRUCTURE_CLASSES)
    zero_weights = structure_logits.new_zeros(NUM_STRUCTURE_CLASSES)

    if valid_mask.sum() == 0:
        return (structure_logits * 0.0).sum(), zero_counts, zero_weights

    logits = structure_logits[valid_mask]
    labels = structure_labels[valid_mask]
    class_counts = torch.bincount(labels, minlength=NUM_STRUCTURE_CLASSES)

    logits_for_loss = logits.float()
    class_weights = structure_logits.new_zeros(NUM_STRUCTURE_CLASSES, dtype=torch.float32)
    present = class_counts > 0
    present_count = present.sum().to(dtype=torch.float32).clamp_min(1.0)
    class_weights[present] = labels.numel() / (present_count * class_counts[present].to(dtype=torch.float32))
    class_weights[present] = class_weights[present].clamp(max=class_weight_cap)

    loss = F.cross_entropy(logits_for_loss, labels, weight=class_weights)
    return loss, class_counts.to(dtype=torch.float32), class_weights


def weighted_ce_region(
    region_logits: torch.Tensor,
    region_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weight_cap: float = MAX_REGION_CLASS_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_counts = region_logits.new_zeros(NUM_REGION_CLASSES)
    zero_weights = region_logits.new_zeros(NUM_REGION_CLASSES)

    if valid_mask.sum() == 0:
        return (region_logits * 0.0).sum(), zero_counts, zero_weights

    logits = region_logits[valid_mask]
    labels = region_labels[valid_mask]
    class_counts = torch.bincount(labels, minlength=NUM_REGION_CLASSES)

    logits_for_loss = logits.float()
    class_weights = region_logits.new_zeros(NUM_REGION_CLASSES, dtype=torch.float32)
    present = class_counts > 0
    present_count = present.sum().to(dtype=torch.float32).clamp_min(1.0)
    class_weights[present] = labels.numel() / (present_count * class_counts[present].to(dtype=torch.float32))
    class_weights[present] = class_weights[present].clamp(max=class_weight_cap)

    loss = F.cross_entropy(logits_for_loss, labels, weight=class_weights)
    return loss, class_counts.to(dtype=torch.float32), class_weights


def weighted_ce_aa(
    aa_logits: torch.Tensor,
    aa_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weight_cap: float = MAX_AA_CLASS_WEIGHT,
) -> tuple[torch.Tensor, int]:
    supervised_mask = valid_mask & (aa_labels != AA_NON_CDS)
    cds_token_count = int(supervised_mask.sum().item())
    if cds_token_count == 0:
        return (aa_logits * 0.0).sum(), 0

    logits = aa_logits[supervised_mask].float()
    labels = aa_labels[supervised_mask]
    class_counts = torch.bincount(labels, minlength=NUM_AA_CLASSES)
    class_weights = aa_logits.new_zeros(NUM_AA_CLASSES, dtype=torch.float32)
    present = class_counts > 0
    present_count = present.sum().to(dtype=torch.float32).clamp_min(1.0)
    class_weights[present] = labels.numel() / (present_count * class_counts[present].to(dtype=torch.float32))
    class_weights[present] = class_weights[present].clamp(max=class_weight_cap)
    class_weights[AA_NON_CDS] = 0.0
    loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=AA_NON_CDS)
    return loss, cds_token_count


def weighted_ce_mutation_effect(
    mutation_effect_logits: torch.Tensor,
    mutation_effect_labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    supervised_mask = mutation_effect_labels != MUTATION_EFFECT_IGNORE_INDEX
    supervised_count = int(supervised_mask.sum().item())
    if supervised_count == 0:
        return (mutation_effect_logits * 0.0).sum(), 0

    logits = mutation_effect_logits.reshape(-1, mutation_effect_logits.shape[-1]).float()
    labels = mutation_effect_labels.reshape(-1)
    loss = F.cross_entropy(logits, labels, ignore_index=MUTATION_EFFECT_IGNORE_INDEX)
    return loss, supervised_count


def weighted_ce_codon(
    codon_logits: torch.Tensor,
    codon_labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    supervised_mask = codon_labels != CODON_IGNORE_INDEX
    supervised_count = int(supervised_mask.sum().item())
    if supervised_count == 0:
        return (codon_logits * 0.0).sum(), 0

    logits = codon_logits.reshape(-1, codon_logits.shape[-1]).float()
    labels = codon_labels.reshape(-1)
    loss = F.cross_entropy(logits, labels, ignore_index=CODON_IGNORE_INDEX)
    return loss, supervised_count


def mutation_effect_pairwise_ranking_loss(
    mutation_effect_logits: torch.Tensor,
    mutation_effect_labels: torch.Tensor,
    margin: float = 0.5,
    max_pairs_per_sample: int = 64,
) -> torch.Tensor:
    """Rank stop above missense above synonymous without treating all missense alike."""

    zero = (mutation_effect_logits * 0.0).sum()
    sample_losses: list[torch.Tensor] = []
    for sample_index in range(mutation_effect_logits.shape[0]):
        sample_logits = mutation_effect_logits[sample_index].reshape(-1, mutation_effect_logits.shape[-1]).float()
        sample_labels = mutation_effect_labels[sample_index].reshape(-1)
        valid_mask = sample_labels != MUTATION_EFFECT_IGNORE_INDEX
        if valid_mask.sum() == 0:
            continue

        sample_logits = sample_logits[valid_mask]
        sample_labels = sample_labels[valid_mask]
        rank_targets = sample_logits.new_zeros(sample_labels.shape)
        rank_targets = torch.where(
            sample_labels == MUTATION_EFFECT_MISSENSE,
            sample_logits.new_full(sample_labels.shape, 0.5),
            rank_targets,
        )
        rank_targets = torch.where(
            sample_labels == MUTATION_EFFECT_STOP,
            sample_logits.new_full(sample_labels.shape, 1.0),
            rank_targets,
        )
        severity_scores = sample_logits[:, MUTATION_EFFECT_STOP] + 0.5 * sample_logits[:, MUTATION_EFFECT_MISSENSE]
        higher, lower = torch.nonzero(
            rank_targets.unsqueeze(1) > rank_targets.unsqueeze(0),
            as_tuple=True,
        )
        if higher.numel() == 0:
            continue
        pair_count = min(int(higher.numel()), max_pairs_per_sample)
        pair_indices = torch.randperm(higher.numel(), device=mutation_effect_logits.device)[:pair_count]
        sample_losses.append(
            F.relu(margin - (severity_scores[higher[pair_indices]] - severity_scores[lower[pair_indices]])).mean()
        )

    if not sample_losses:
        return zero
    return torch.stack(sample_losses).mean()


def allele_same_locus_ranking_loss(
    severity_scores: torch.Tensor,
    severity_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    zero = (severity_scores * 0.0).sum()
    if not torch.any(valid_mask):
        return zero

    losses: list[torch.Tensor] = []
    for batch_index in range(severity_scores.shape[0]):
        valid = valid_mask[batch_index]
        if valid.sum() < 2:
            continue
        scores = severity_scores[batch_index][valid].float()
        targets = severity_targets[batch_index][valid].float()
        higher, lower = torch.nonzero(targets.unsqueeze(1) > targets.unsqueeze(0), as_tuple=True)
        if higher.numel() == 0:
            continue
        target_gap = (targets[higher] - targets[lower]).clamp_min(0.0)
        losses.append(F.relu(float(margin) * target_gap - (scores[higher] - scores[lower])).mean())
    if not losses:
        return zero
    return torch.stack(losses).mean()


def allele_directed_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_rank: float = 0.5,
    lambda_severity: float = 1.0,
    lambda_swap: float = 0.25,
    lambda_far: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = outputs["allele_effect_logits"]
    scores = outputs["allele_severity_score"]
    labels = batch["allele_effect_labels"]
    severity_targets = batch["allele_severity_targets"]
    valid_mask = batch["allele_valid_mask"] & (labels != ALLELE_EFFECT_IGNORE_INDEX)
    zero = (logits * 0.0).sum() + (scores * 0.0).sum()

    if torch.any(valid_mask):
        loss_effect = F.cross_entropy(
            logits[valid_mask].float(),
            labels[valid_mask].long(),
            ignore_index=ALLELE_EFFECT_IGNORE_INDEX,
        )
        loss_severity = F.smooth_l1_loss(scores[valid_mask].float(), severity_targets[valid_mask].float())
    else:
        loss_effect = zero
        loss_severity = zero

    loss_rank = allele_same_locus_ranking_loss(scores, severity_targets, valid_mask)
    swap_scores = outputs.get("allele_swap_severity_score")
    if swap_scores is None or not torch.any(valid_mask):
        loss_swap = zero
    else:
        loss_swap = F.smooth_l1_loss((scores + swap_scores)[valid_mask].float(), torch.zeros_like(scores[valid_mask]))
    far_distance = outputs.get("allele_far_distance")
    loss_far = zero if far_distance is None or not torch.any(valid_mask) else far_distance[valid_mask].float().mean()

    total = (
        loss_effect
        + float(lambda_severity) * loss_severity
        + float(lambda_rank) * loss_rank
        + float(lambda_swap) * loss_swap
        + float(lambda_far) * loss_far
    )
    valid_count = valid_mask.sum().to(dtype=logits.dtype)
    with torch.no_grad():
        if torch.any(valid_mask):
            pred = logits.argmax(dim=-1)
            accuracy = (pred[valid_mask] == labels[valid_mask]).float().mean()
            severity_mae = (scores[valid_mask] - severity_targets[valid_mask]).abs().float().mean()
        else:
            accuracy = logits.new_tensor(0.0)
            severity_mae = logits.new_tensor(0.0)
    return total, {
        "loss_allele_effect": loss_effect.detach(),
        "loss_allele_severity": loss_severity.detach(),
        "loss_allele_rank": loss_rank.detach(),
        "loss_allele_swap": loss_swap.detach(),
        "loss_allele_far": loss_far.detach(),
        "allele_valid_pairs": valid_count.detach(),
        "allele_effect_accuracy": accuracy.detach(),
        "allele_severity_mae": severity_mae.detach(),
    }


def counterfactual_effect_loss(
    effect_logits: torch.Tensor,
    effect_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if class_weights is None:
        class_weights = effect_logits.new_tensor(COUNTERFACTUAL_EFFECT_CLASS_WEIGHTS)
    return masked_cross_entropy(
        effect_logits,
        effect_labels,
        valid_mask,
        ignore_index=COUNTERFACTUAL_EFFECT_IGNORE_INDEX,
        class_weights=class_weights.to(device=effect_logits.device, dtype=torch.float32),
    )


def counterfactual_severity_loss(
    severity_pred: torch.Tensor,
    severity_targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return masked_huber(severity_pred, severity_targets, valid_mask)


def counterfactual_disagreement_loss(
    h_ref: torch.Tensor,
    h_alt: torch.Tensor,
    edit_position: torch.Tensor,
    cf_active: torch.Tensor,
    radius: int = 8,
    far_radius: int = 64,
    local_similarity_target: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if h_ref.shape != h_alt.shape:
        raise ValueError(f"h_ref and h_alt must have matching shapes, got {h_ref.shape} and {h_alt.shape}.")

    zero = (h_ref * 0.0).sum() + (h_alt * 0.0).sum()
    if cf_active.sum() == 0:
        return zero, zero, zero, zero

    positions = torch.arange(h_ref.shape[1], device=h_ref.device).unsqueeze(0)
    active_mask = cf_active.unsqueeze(1)
    local_mask = active_mask & ((positions - edit_position.unsqueeze(1)).abs() <= radius)
    far_mask = active_mask & ((positions - edit_position.unsqueeze(1)).abs() > far_radius)
    cosine_similarity = F.cosine_similarity(h_ref, h_alt, dim=-1)

    local_cosine = cosine_similarity[local_mask].mean() if local_mask.any() else zero
    local_margin_loss = F.relu(local_cosine - float(local_similarity_target)) if local_mask.any() else zero
    far_distance = (1.0 - cosine_similarity[far_mask]).mean() if far_mask.any() else zero
    return local_margin_loss, far_distance, local_cosine, far_distance


def masked_mlm_region_losses(
    mlm_logits: torch.Tensor,
    mlm_labels: torch.Tensor,
    region_labels: torch.Tensor,
) -> dict[int, torch.Tensor]:
    batch_size, seq_len, vocab_size = mlm_logits.shape
    labels = _map_mlm_labels_for_logits(mlm_logits, mlm_labels)
    ignore_index = -100 if mlm_logits.shape[-1] == len(SNV_BASES) else PAD_ID
    per_token = F.cross_entropy(
        mlm_logits.reshape(batch_size * seq_len, vocab_size),
        labels.reshape(batch_size * seq_len),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(batch_size, seq_len)
    supervised_mask = _mlm_supervised_mask(mlm_logits, mlm_labels)

    losses: dict[int, torch.Tensor] = {}
    for region_index in range(NUM_REGION_CLASSES):
        region_mask = supervised_mask & (region_labels == region_index)
        if region_mask.sum() == 0:
            losses[region_index] = (mlm_logits * 0.0).sum()
        else:
            losses[region_index] = per_token[region_mask].mean()
    return losses


def rc_embedding_loss(
    z: torch.Tensor,
    z_rc: torch.Tensor,
    loss_type: str = "cosine",
) -> torch.Tensor:
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(z, z_rc, dim=-1).mean()
    if loss_type == "mse":
        return F.mse_loss(z, z_rc)
    raise ValueError(f"Unknown RC loss type: {loss_type}")


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    rc_outputs: dict[str, torch.Tensor] | None = None,
    alt_outputs: dict[str, torch.Tensor] | None = None,
    w_mlm: float = 1.0,
    w_phylo100: float = 0.25,
    w_phylo470: float = 0.25,
    w_structure: float = 0.25,
    w_rc: float = 0.0,
    w_region: float = 0.0,
    w_aa: float = 0.0,
    w_codon_phylo: float = 0.0,
    w_mutation_effect: float = 0.0,
    w_counterfactual: float = 0.0,
    w_allele: float = 0.0,
    w_codon: float = 0.0,
    w_encode: float = 0.0,
    w_conservation_bin: float = 0.0,
    w_splice_distance: float = 0.0,
    w_codon_pos: float = 0.0,
    w_exon_phase: float = 0.0,
    w_counterfactual_snv: float = 0.0,
    w_counterfactual_severity: float = 0.0,
    rc_loss_type: str = "cosine",
    phylo_weighted_mlm: bool = False,
    phylo_mlm_boost: float = 2.0,
    lambda_mutation_effect_rank: float = 0.5,
    lambda_allele_rank: float = 0.5,
    lambda_allele_severity: float = 1.0,
    lambda_allele_swap: float = 0.25,
    lambda_allele_far: float = 0.1,
    splice_class_weight_cap: float = MAX_SPLICE_CLASS_WEIGHT,
    region_class_weight_cap: float = MAX_REGION_CLASS_WEIGHT,
    aa_class_weight_cap: float = MAX_AA_CLASS_WEIGHT,
    counterfactual_radius: int = 8,
    counterfactual_far_radius: int = 64,
    counterfactual_local_similarity_target: float = 0.8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    aux_valid_mask = batch["aux_valid_mask"]
    mlm_supervised_tokens = (batch["mlm_labels"] != PAD_ID).sum().to(dtype=outputs["mlm_logits"].dtype)
    aux_valid_tokens = aux_valid_mask.sum().to(dtype=outputs["mlm_logits"].dtype)
    zero = outputs["mlm_logits"].new_tensor(0.0)
    zero_splice_counts = outputs["mlm_logits"].new_zeros(NUM_STRUCTURE_CLASSES)
    zero_splice_weights = outputs["mlm_logits"].new_zeros(NUM_STRUCTURE_CLASSES)
    zero_region_counts = outputs["mlm_logits"].new_zeros(NUM_REGION_CLASSES)
    zero_region_weights = outputs["mlm_logits"].new_zeros(NUM_REGION_CLASSES)

    loss_mlm = zero
    if w_mlm != 0.0:
        if phylo_weighted_mlm:
            loss_mlm = phylo_weighted_ce_token(
                outputs["mlm_logits"], batch["mlm_labels"],
                batch["phylo100"], phylo_mlm_boost,
            )
        else:
            loss_mlm = masked_ce_token(outputs["mlm_logits"], batch["mlm_labels"])
    mlm_region_losses = masked_mlm_region_losses(
        outputs["mlm_logits"],
        batch["mlm_labels"],
        batch["region_labels"],
    )

    loss_phylo100 = zero
    if w_phylo100 != 0.0:
        loss_phylo100 = valid_smooth_l1(outputs["phylo100_pred"], batch["phylo100"], aux_valid_mask)

    loss_phylo470 = zero
    if w_phylo470 != 0.0:
        loss_phylo470 = valid_smooth_l1(outputs["phylo470_pred"], batch["phylo470"], aux_valid_mask)

    loss_structure = zero
    splice_class_counts = zero_splice_counts
    splice_class_weights = zero_splice_weights
    if w_structure != 0.0:
        loss_structure, splice_class_counts, splice_class_weights = weighted_ce_structure(
            outputs["structure_logits"],
            batch["structure_labels"],
            aux_valid_mask,
            class_weight_cap=splice_class_weight_cap,
        )

    loss_region = zero
    region_class_counts = zero_region_counts
    region_class_weights = zero_region_weights
    if w_region != 0.0 and "region_logits" in outputs:
        loss_region, region_class_counts, region_class_weights = weighted_ce_region(
            outputs["region_logits"],
            batch["region_labels"],
            aux_valid_mask,
            class_weight_cap=region_class_weight_cap,
        )

    loss_aa = zero
    aa_cds_tokens = zero
    aa_valid_mask = aux_valid_mask & (batch["aa_labels"] != AA_NON_CDS)
    if w_aa != 0.0 and "aa_logits" in outputs:
        loss_aa, aa_cds_count = weighted_ce_aa(
            outputs["aa_logits"],
            batch["aa_labels"],
            aux_valid_mask,
            class_weight_cap=aa_class_weight_cap,
        )
        aa_cds_tokens = outputs["mlm_logits"].new_tensor(float(aa_cds_count))

    loss_codon_phylo = zero
    if w_codon_phylo != 0.0 and "codon_phylo_pred" in outputs:
        loss_codon_phylo = valid_smooth_l1(
            outputs["codon_phylo_pred"],
            batch["codon_phylo_target"],
            aa_valid_mask,
        )

    loss_mutation_effect = zero
    loss_mutation_effect_rank = zero
    if w_mutation_effect != 0.0 and "mutation_effect_logits" in outputs and "mutation_effect_labels" in batch:
        loss_me_ce, _ = weighted_ce_mutation_effect(
            outputs["mutation_effect_logits"],
            batch["mutation_effect_labels"],
        )
        loss_mutation_effect_rank = mutation_effect_pairwise_ranking_loss(
            outputs["mutation_effect_logits"],
            batch["mutation_effect_labels"],
        )
        loss_mutation_effect = loss_me_ce + lambda_mutation_effect_rank * loss_mutation_effect_rank

    loss_counterfactual_local = zero
    loss_counterfactual_far = zero
    cf_local_cosine = zero
    cf_local_margin_loss = zero
    cf_far_distance = zero
    if (
        w_counterfactual != 0.0
        and alt_outputs is not None
        and "hidden_states" in alt_outputs
        and "edit_position" in batch
        and "cf_active" in batch
    ):
        assert alt_outputs is not None
        (
            loss_counterfactual_local,
            loss_counterfactual_far,
            cf_local_cosine,
            cf_far_distance,
        ) = counterfactual_disagreement_loss(
            outputs["hidden_states"],
            alt_outputs["hidden_states"],
            batch["edit_position"],
            batch["cf_active"],
            radius=counterfactual_radius,
            far_radius=counterfactual_far_radius,
            local_similarity_target=counterfactual_local_similarity_target,
        )
        cf_local_margin_loss = loss_counterfactual_local
        cf_far_distance = loss_counterfactual_far
    loss_counterfactual = loss_counterfactual_local + loss_counterfactual_far

    loss_allele = zero
    allele_stats: dict[str, torch.Tensor] = {
        "loss_allele_effect": zero.detach(),
        "loss_allele_severity": zero.detach(),
        "loss_allele_rank": zero.detach(),
        "loss_allele_swap": zero.detach(),
        "loss_allele_far": zero.detach(),
        "allele_valid_pairs": zero.detach(),
        "allele_effect_accuracy": zero.detach(),
        "allele_severity_mae": zero.detach(),
    }
    if (
        w_allele != 0.0
        and "allele_effect_logits" in outputs
        and "allele_severity_score" in outputs
        and "allele_effect_labels" in batch
    ):
        loss_allele, allele_stats = allele_directed_loss(
            outputs,
            batch,
            lambda_rank=lambda_allele_rank,
            lambda_severity=lambda_allele_severity,
            lambda_swap=lambda_allele_swap,
            lambda_far=lambda_allele_far,
        )

    loss_codon = zero
    if w_codon != 0.0 and "codon_logits" in outputs and "codon_labels" in batch:
        loss_codon, _ = weighted_ce_codon(outputs["codon_logits"], batch["codon_labels"])

    loss_encode = zero
    num_encode_tracks = batch["encode_targets"].shape[-1] if "encode_targets" in batch else 0
    encode_track_losses = outputs["mlm_logits"].new_zeros(num_encode_tracks)
    if (
        w_encode != 0.0
        and "encode_pred" in outputs
        and "encode_targets" in batch
        and batch["encode_targets"].shape[-1] > 0
    ):
        loss_encode, encode_track_losses = valid_smooth_l1_multitrack(
            outputs["encode_pred"],
            batch["encode_targets"],
        )

    loss_conservation_bin = zero
    if (
        w_conservation_bin != 0.0
        and "conservation_bin_logits" in outputs
        and "conservation_bin_labels" in batch
    ):
        loss_conservation_bin = masked_cross_entropy(
            outputs["conservation_bin_logits"],
            batch["conservation_bin_labels"],
            aux_valid_mask,
        )

    loss_splice_distance = zero
    if (
        w_splice_distance != 0.0
        and "donor_distance_logits" in outputs
        and "acceptor_distance_logits" in outputs
        and "donor_distance_labels" in batch
        and "acceptor_distance_labels" in batch
    ):
        loss_donor_distance = masked_cross_entropy(
            outputs["donor_distance_logits"],
            batch["donor_distance_labels"],
            aux_valid_mask,
        )
        loss_acceptor_distance = masked_cross_entropy(
            outputs["acceptor_distance_logits"],
            batch["acceptor_distance_labels"],
            aux_valid_mask,
        )
        loss_splice_distance = 0.5 * (loss_donor_distance + loss_acceptor_distance)

    loss_codon_pos = zero
    if w_codon_pos != 0.0 and "codon_pos_logits" in outputs and "codon_pos_labels" in batch:
        loss_codon_pos = masked_cross_entropy(
            outputs["codon_pos_logits"],
            batch["codon_pos_labels"],
            aux_valid_mask,
        )

    loss_exon_phase = zero
    if w_exon_phase != 0.0 and "exon_phase_logits" in outputs and "exon_phase_labels" in batch:
        loss_exon_phase = masked_cross_entropy(
            outputs["exon_phase_logits"],
            batch["exon_phase_labels"],
            aux_valid_mask,
        )

    loss_counterfactual_snv = zero
    loss_counterfactual_snv_severity = zero
    counterfactual_snv_valid = zero
    if (
        (w_counterfactual_snv != 0.0 or w_counterfactual_severity != 0.0)
        and "counterfactual_effect_logits" in outputs
        and "counterfactual_severity" in outputs
        and "counterfactual_effect_labels" in batch
        and "counterfactual_severity_targets" in batch
    ):
        cf_mask = batch.get("counterfactual_valid_mask")
        if not torch.is_tensor(cf_mask):
            cf_mask = batch["counterfactual_effect_labels"] != COUNTERFACTUAL_EFFECT_IGNORE_INDEX
        loss_counterfactual_snv = counterfactual_effect_loss(
            outputs["counterfactual_effect_logits"],
            batch["counterfactual_effect_labels"],
            cf_mask,
        )
        loss_counterfactual_snv_severity = counterfactual_severity_loss(
            outputs["counterfactual_severity"],
            batch["counterfactual_severity_targets"],
            cf_mask,
        )
        counterfactual_snv_valid = cf_mask.sum().to(dtype=outputs["mlm_logits"].dtype)

    loss_rc = zero
    if w_rc != 0.0 and rc_outputs is not None:
        loss_rc = rc_embedding_loss(
            outputs["sequence_embedding"],
            rc_outputs["sequence_embedding"],
            loss_type=rc_loss_type,
        )

    total = (
        w_mlm * loss_mlm
        + w_phylo100 * loss_phylo100
        + w_phylo470 * loss_phylo470
        + w_structure * loss_structure
        + w_region * loss_region
        + w_aa * loss_aa
        + w_codon_phylo * loss_codon_phylo
        + w_mutation_effect * loss_mutation_effect
        + w_counterfactual * loss_counterfactual
        + w_allele * loss_allele
        + w_codon * loss_codon
        + w_encode * loss_encode
        + w_conservation_bin * loss_conservation_bin
        + w_splice_distance * loss_splice_distance
        + w_codon_pos * loss_codon_pos
        + w_exon_phase * loss_exon_phase
        + w_counterfactual_snv * loss_counterfactual_snv
        + w_counterfactual_severity * loss_counterfactual_snv_severity
        + w_rc * loss_rc
    )

    stats = {
        "loss": total.detach(),
        "loss_mlm": loss_mlm.detach(),
        "loss_phylo100": loss_phylo100.detach(),
        "loss_phylo470": loss_phylo470.detach(),
        "loss_structure": loss_structure.detach(),
        "loss_region": loss_region.detach(),
        "loss_aa": loss_aa.detach(),
        "loss_codon_phylo": loss_codon_phylo.detach(),
        "loss_mutation_effect": loss_mutation_effect.detach(),
        "loss_mutation_effect_rank": loss_mutation_effect_rank.detach(),
        "loss_counterfactual": loss_counterfactual.detach(),
        "loss_counterfactual_local": loss_counterfactual_local.detach(),
        "loss_counterfactual_far": loss_counterfactual_far.detach(),
        "loss_allele": loss_allele.detach(),
        "loss_allele_effect": allele_stats["loss_allele_effect"],
        "loss_allele_severity": allele_stats["loss_allele_severity"],
        "loss_allele_rank": allele_stats["loss_allele_rank"],
        "loss_allele_swap": allele_stats["loss_allele_swap"],
        "loss_allele_far": allele_stats["loss_allele_far"],
        "cf_local_cosine": cf_local_cosine.detach(),
        "cf_local_margin_loss": cf_local_margin_loss.detach(),
        "cf_far_distance": cf_far_distance.detach(),
        "counterfactual_effective_weight": outputs["mlm_logits"].new_tensor(float(w_counterfactual)).detach(),
        "loss_codon": loss_codon.detach(),
        "loss_encode": loss_encode.detach(),
        "loss_conservation_bin": loss_conservation_bin.detach(),
        "loss_splice_distance": loss_splice_distance.detach(),
        "loss_codon_pos": loss_codon_pos.detach(),
        "loss_exon_phase": loss_exon_phase.detach(),
        "loss_counterfactual_snv": loss_counterfactual_snv.detach(),
        "loss_counterfactual_snv_severity": loss_counterfactual_snv_severity.detach(),
        "loss_rc": loss_rc.detach(),
        "mlm_supervised_tokens": mlm_supervised_tokens.detach(),
        "aux_valid_tokens": aux_valid_tokens.detach(),
        "aa_cds_tokens": aa_cds_tokens.detach(),
        "counterfactual_snv_valid": counterfactual_snv_valid.detach(),
        "allele_valid_pairs": allele_stats["allele_valid_pairs"],
        "allele_effect_accuracy": allele_stats["allele_effect_accuracy"],
        "allele_severity_mae": allele_stats["allele_severity_mae"],
        "splice_bg_tokens": splice_class_counts[STRUCT_BACKGROUND].detach(),
        "splice_core_tokens": splice_class_counts[STRUCT_SPLICE_CORE].detach(),
        "splice_region_tokens": splice_class_counts[STRUCT_SPLICE_REGION].detach(),
        "splice_weight_bg": splice_class_weights[STRUCT_BACKGROUND].detach(),
        "splice_weight_core": splice_class_weights[STRUCT_SPLICE_CORE].detach(),
        "splice_weight_region": splice_class_weights[STRUCT_SPLICE_REGION].detach(),
        "region_intergenic_tokens": region_class_counts[REGION_INTERGENIC].detach(),
        "region_intron_tokens": region_class_counts[REGION_INTRON].detach(),
        "region_noncoding_exon_tokens": region_class_counts[REGION_NONCODING_EXON].detach(),
        "region_utr_tokens": region_class_counts[REGION_UTR].detach(),
        "region_cds_tokens": region_class_counts[REGION_CDS].detach(),
        "region_weight_intergenic": region_class_weights[REGION_INTERGENIC].detach(),
        "region_weight_intron": region_class_weights[REGION_INTRON].detach(),
        "region_weight_noncoding_exon": region_class_weights[REGION_NONCODING_EXON].detach(),
        "region_weight_utr": region_class_weights[REGION_UTR].detach(),
        "region_weight_cds": region_class_weights[REGION_CDS].detach(),
        "mlm_region_loss_intergenic": mlm_region_losses[REGION_INTERGENIC].detach(),
        "mlm_region_loss_intron": mlm_region_losses[REGION_INTRON].detach(),
        "mlm_region_loss_noncoding_exon": mlm_region_losses[REGION_NONCODING_EXON].detach(),
        "mlm_region_loss_utr": mlm_region_losses[REGION_UTR].detach(),
        "mlm_region_loss_cds": mlm_region_losses[REGION_CDS].detach(),
    }
    track_names = batch.get("encode_track_names")
    if isinstance(track_names, list) and len(track_names) == int(encode_track_losses.shape[0]):
        for track_name, track_loss in zip(track_names, encode_track_losses, strict=False):
            stats[f"loss_encode_{track_name}"] = track_loss.detach()
    return total, stats


UNCERTAINTY_TASK_KEYS = (
    "mlm",
    "phylo100",
    "phylo470",
    "structure",
    "region",
    "aa",
    "codon_phylo",
    "mutation_effect",
    "counterfactual",
    "allele",
    "codon",
    "encode",
)


def compute_uncertainty_weighted_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    log_sigmas: dict[str, torch.Tensor],
    alt_outputs: dict[str, torch.Tensor] | None = None,
    phylo_weighted_mlm: bool = False,
    phylo_mlm_boost: float = 2.0,
    aux_scale: float = 1.0,
    lambda_mutation_effect_rank: float = 0.5,
    lambda_allele_rank: float = 0.5,
    lambda_allele_severity: float = 1.0,
    lambda_allele_swap: float = 0.25,
    lambda_allele_far: float = 0.1,
    splice_class_weight_cap: float = MAX_SPLICE_CLASS_WEIGHT,
    region_class_weight_cap: float = MAX_REGION_CLASS_WEIGHT,
    aa_class_weight_cap: float = MAX_AA_CLASS_WEIGHT,
    counterfactual_radius: int = 8,
    counterfactual_far_radius: int = 64,
    counterfactual_local_similarity_target: float = 0.8,
    fixed_counterfactual_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Kendall et al. (2018) uncertainty weighting for multitask losses.

    Each task loss is weighted by learned homoscedastic uncertainty:
        weighted_loss_t = loss_t * exp(-2 * log_sigma_t) / 2 + log_sigma_t
    """
    aux_valid_mask = batch["aux_valid_mask"]
    mlm_supervised_tokens = (batch["mlm_labels"] != PAD_ID).sum().to(dtype=outputs["mlm_logits"].dtype)
    aux_valid_tokens = aux_valid_mask.sum().to(dtype=outputs["mlm_logits"].dtype)
    zero = outputs["mlm_logits"].new_tensor(0.0)
    zero_region_counts = outputs["mlm_logits"].new_zeros(NUM_REGION_CLASSES)
    zero_region_weights = outputs["mlm_logits"].new_zeros(NUM_REGION_CLASSES)
    zero_count = outputs["mlm_logits"].new_tensor(0.0)

    if phylo_weighted_mlm:
        loss_mlm = phylo_weighted_ce_token(
            outputs["mlm_logits"], batch["mlm_labels"],
            batch["phylo100"], phylo_mlm_boost,
        )
    else:
        loss_mlm = masked_ce_token(outputs["mlm_logits"], batch["mlm_labels"])
    mlm_region_losses = masked_mlm_region_losses(
        outputs["mlm_logits"],
        batch["mlm_labels"],
        batch["region_labels"],
    )
    loss_phylo100 = valid_smooth_l1(outputs["phylo100_pred"], batch["phylo100"], aux_valid_mask)
    loss_phylo470 = valid_smooth_l1(outputs["phylo470_pred"], batch["phylo470"], aux_valid_mask)
    loss_structure, splice_class_counts, splice_class_weights = weighted_ce_structure(
        outputs["structure_logits"],
        batch["structure_labels"],
        aux_valid_mask,
        class_weight_cap=splice_class_weight_cap,
    )

    has_region = "region_logits" in outputs and "region" in log_sigmas
    loss_region = zero
    region_class_counts = zero_region_counts
    region_class_weights = zero_region_weights
    if has_region:
        loss_region, region_class_counts, region_class_weights = weighted_ce_region(
            outputs["region_logits"],
            batch["region_labels"],
            aux_valid_mask,
            class_weight_cap=region_class_weight_cap,
        )

    has_aa = "aa_logits" in outputs and "aa" in log_sigmas
    aa_valid_mask = aux_valid_mask & (batch["aa_labels"] != AA_NON_CDS)
    loss_aa = zero
    aa_cds_count = 0
    if has_aa:
        loss_aa, aa_cds_count = weighted_ce_aa(
            outputs["aa_logits"],
            batch["aa_labels"],
            aux_valid_mask,
            class_weight_cap=aa_class_weight_cap,
        )

    has_codon_phylo = "codon_phylo_pred" in outputs and "codon_phylo" in log_sigmas
    loss_codon_phylo = zero
    if has_codon_phylo:
        loss_codon_phylo = valid_smooth_l1(
            outputs["codon_phylo_pred"],
            batch["codon_phylo_target"],
            aa_valid_mask,
        )

    has_mutation_effect = (
        "mutation_effect_logits" in outputs
        and "mutation_effect" in log_sigmas
        and "mutation_effect_labels" in batch
    )
    loss_mutation_effect = zero
    loss_mutation_effect_rank = zero
    if has_mutation_effect:
        loss_me_ce, _ = weighted_ce_mutation_effect(
            outputs["mutation_effect_logits"],
            batch["mutation_effect_labels"],
        )
        loss_mutation_effect_rank = mutation_effect_pairwise_ranking_loss(
            outputs["mutation_effect_logits"],
            batch["mutation_effect_labels"],
        )
        loss_mutation_effect = loss_me_ce + lambda_mutation_effect_rank * loss_mutation_effect_rank

    has_counterfactual_outputs = (
        alt_outputs is not None
        and "hidden_states" in alt_outputs
        and "edit_position" in batch
        and "cf_active" in batch
    )
    has_active_counterfactual = (
        has_counterfactual_outputs
        and torch.is_tensor(batch["cf_active"])
        and bool(torch.any(batch["cf_active"]).item())
    )
    has_counterfactual = has_active_counterfactual and "counterfactual" in log_sigmas
    loss_counterfactual_local = zero
    loss_counterfactual_far = zero
    cf_local_cosine = zero
    cf_local_margin_loss = zero
    cf_far_distance = zero
    if has_counterfactual_outputs:
        assert alt_outputs is not None
        (
            loss_counterfactual_local,
            loss_counterfactual_far,
            cf_local_cosine,
            cf_far_distance,
        ) = counterfactual_disagreement_loss(
            outputs["hidden_states"],
            alt_outputs["hidden_states"],
            batch["edit_position"],
            batch["cf_active"],
            radius=counterfactual_radius,
            far_radius=counterfactual_far_radius,
            local_similarity_target=counterfactual_local_similarity_target,
        )
        cf_local_margin_loss = loss_counterfactual_local
    loss_counterfactual = loss_counterfactual_local + loss_counterfactual_far

    has_allele_outputs = (
        "allele" in log_sigmas
        and "allele_effect_logits" in outputs
        and "allele_severity_score" in outputs
        and "allele_effect_labels" in batch
        and "allele_valid_mask" in batch
    )
    has_active_allele = (
        has_allele_outputs
        and torch.is_tensor(batch["allele_valid_mask"])
        and bool(
            torch.any(
                batch["allele_valid_mask"] & (batch["allele_effect_labels"] != ALLELE_EFFECT_IGNORE_INDEX)
            ).item()
        )
    )
    has_allele = has_active_allele and "allele" in log_sigmas
    loss_allele = zero
    allele_stats: dict[str, torch.Tensor] = {
        "loss_allele_effect": zero.detach(),
        "loss_allele_severity": zero.detach(),
        "loss_allele_rank": zero.detach(),
        "loss_allele_swap": zero.detach(),
        "loss_allele_far": zero.detach(),
        "allele_valid_pairs": zero.detach(),
        "allele_effect_accuracy": zero.detach(),
        "allele_severity_mae": zero.detach(),
    }
    if has_allele:
        loss_allele, allele_stats = allele_directed_loss(
            outputs,
            batch,
            lambda_rank=lambda_allele_rank,
            lambda_severity=lambda_allele_severity,
            lambda_swap=lambda_allele_swap,
            lambda_far=lambda_allele_far,
        )

    has_codon = "codon_logits" in outputs and "codon" in log_sigmas and "codon_labels" in batch
    loss_codon = zero
    if has_codon:
        loss_codon, _ = weighted_ce_codon(outputs["codon_logits"], batch["codon_labels"])

    has_encode = (
        "encode_pred" in outputs
        and "encode" in log_sigmas
        and "encode_targets" in batch
        and batch["encode_targets"].shape[-1] > 0
    )
    loss_encode = zero
    num_encode_tracks = batch["encode_targets"].shape[-1] if "encode_targets" in batch else 0
    encode_track_losses = outputs["mlm_logits"].new_zeros(num_encode_tracks)
    if has_encode:
        loss_encode, encode_track_losses = valid_smooth_l1_multitrack(
            outputs["encode_pred"],
            batch["encode_targets"],
        )

    raw_losses: dict[str, torch.Tensor] = {
        "mlm": loss_mlm,
        "phylo100": loss_phylo100,
        "phylo470": loss_phylo470,
        "structure": loss_structure,
    }
    if has_region:
        raw_losses["region"] = loss_region
    if has_aa:
        raw_losses["aa"] = loss_aa
    if has_codon_phylo:
        raw_losses["codon_phylo"] = loss_codon_phylo
    if has_mutation_effect:
        raw_losses["mutation_effect"] = loss_mutation_effect
    if has_counterfactual:
        raw_losses["counterfactual"] = loss_counterfactual
    if has_allele:
        raw_losses["allele"] = loss_allele
    if has_codon:
        raw_losses["codon"] = loss_codon
    if has_encode:
        raw_losses["encode"] = loss_encode

    total = outputs["mlm_logits"].new_tensor(0.0)
    counterfactual_effective_weight = zero
    for key, raw_loss in raw_losses.items():
        log_s = log_sigmas[key]
        precision = torch.exp(-2 * log_s)
        effective_loss = raw_loss * aux_scale if key != "mlm" else raw_loss
        total = total + effective_loss * precision / 2 + log_s
        if key == "counterfactual":
            counterfactual_effective_weight = (aux_scale * precision / 2).detach()
    if "counterfactual" in log_sigmas and not has_counterfactual:
        total = total + log_sigmas["counterfactual"] * 0.0
    if "allele" in log_sigmas and not has_allele:
        total = total + log_sigmas["allele"] * 0.0
    if fixed_counterfactual_weight != 0.0 and not has_counterfactual:
        total = total + fixed_counterfactual_weight * aux_scale * loss_counterfactual
        counterfactual_effective_weight = outputs["mlm_logits"].new_tensor(
            float(fixed_counterfactual_weight * aux_scale)
        )

    stats: dict[str, torch.Tensor] = {
        "loss": total.detach(),
        "loss_mlm": loss_mlm.detach(),
        "loss_phylo100": loss_phylo100.detach(),
        "loss_phylo470": loss_phylo470.detach(),
        "loss_structure": loss_structure.detach(),
        "loss_region": loss_region.detach(),
        "loss_aa": loss_aa.detach(),
        "loss_codon_phylo": loss_codon_phylo.detach(),
        "loss_mutation_effect": loss_mutation_effect.detach(),
        "loss_mutation_effect_rank": loss_mutation_effect_rank.detach(),
        "loss_counterfactual": loss_counterfactual.detach(),
        "loss_counterfactual_local": loss_counterfactual_local.detach(),
        "loss_counterfactual_far": loss_counterfactual_far.detach(),
        "loss_allele": loss_allele.detach(),
        "loss_allele_effect": allele_stats["loss_allele_effect"],
        "loss_allele_severity": allele_stats["loss_allele_severity"],
        "loss_allele_rank": allele_stats["loss_allele_rank"],
        "loss_allele_swap": allele_stats["loss_allele_swap"],
        "loss_allele_far": allele_stats["loss_allele_far"],
        "cf_local_cosine": cf_local_cosine.detach(),
        "cf_local_margin_loss": cf_local_margin_loss.detach(),
        "cf_far_distance": cf_far_distance.detach(),
        "counterfactual_effective_weight": counterfactual_effective_weight.detach(),
        "loss_codon": loss_codon.detach(),
        "loss_encode": loss_encode.detach(),
        "loss_rc": zero,
        "mlm_supervised_tokens": mlm_supervised_tokens.detach(),
        "aux_valid_tokens": aux_valid_tokens.detach(),
        "aa_cds_tokens": outputs["mlm_logits"].new_tensor(float(aa_cds_count)).detach()
        if has_aa
        else zero_count.detach(),
        "allele_valid_pairs": allele_stats["allele_valid_pairs"],
        "allele_effect_accuracy": allele_stats["allele_effect_accuracy"],
        "allele_severity_mae": allele_stats["allele_severity_mae"],
        "splice_bg_tokens": splice_class_counts[STRUCT_BACKGROUND].detach(),
        "splice_core_tokens": splice_class_counts[STRUCT_SPLICE_CORE].detach(),
        "splice_region_tokens": splice_class_counts[STRUCT_SPLICE_REGION].detach(),
        "splice_weight_bg": splice_class_weights[STRUCT_BACKGROUND].detach(),
        "splice_weight_core": splice_class_weights[STRUCT_SPLICE_CORE].detach(),
        "splice_weight_region": splice_class_weights[STRUCT_SPLICE_REGION].detach(),
        "region_intergenic_tokens": region_class_counts[REGION_INTERGENIC].detach(),
        "region_intron_tokens": region_class_counts[REGION_INTRON].detach(),
        "region_noncoding_exon_tokens": region_class_counts[REGION_NONCODING_EXON].detach(),
        "region_utr_tokens": region_class_counts[REGION_UTR].detach(),
        "region_cds_tokens": region_class_counts[REGION_CDS].detach(),
        "region_weight_intergenic": region_class_weights[REGION_INTERGENIC].detach(),
        "region_weight_intron": region_class_weights[REGION_INTRON].detach(),
        "region_weight_noncoding_exon": region_class_weights[REGION_NONCODING_EXON].detach(),
        "region_weight_utr": region_class_weights[REGION_UTR].detach(),
        "region_weight_cds": region_class_weights[REGION_CDS].detach(),
        "mlm_region_loss_intergenic": mlm_region_losses[REGION_INTERGENIC].detach(),
        "mlm_region_loss_intron": mlm_region_losses[REGION_INTRON].detach(),
        "mlm_region_loss_noncoding_exon": mlm_region_losses[REGION_NONCODING_EXON].detach(),
        "mlm_region_loss_utr": mlm_region_losses[REGION_UTR].detach(),
        "mlm_region_loss_cds": mlm_region_losses[REGION_CDS].detach(),
        "sigma_mlm": torch.exp(log_sigmas["mlm"]).detach(),
        "sigma_phylo100": torch.exp(log_sigmas["phylo100"]).detach(),
        "sigma_phylo470": torch.exp(log_sigmas["phylo470"]).detach(),
        "sigma_structure": torch.exp(log_sigmas["structure"]).detach(),
    }
    if has_region:
        stats["sigma_region"] = torch.exp(log_sigmas["region"]).detach()
    if has_aa:
        stats["sigma_aa"] = torch.exp(log_sigmas["aa"]).detach()
    if has_codon_phylo:
        stats["sigma_codon_phylo"] = torch.exp(log_sigmas["codon_phylo"]).detach()
    if has_mutation_effect:
        stats["sigma_mutation_effect"] = torch.exp(log_sigmas["mutation_effect"]).detach()
    if "counterfactual" in log_sigmas:
        stats["sigma_counterfactual"] = torch.exp(log_sigmas["counterfactual"]).detach()
    if has_allele:
        stats["sigma_allele"] = torch.exp(log_sigmas["allele"]).detach()
    if has_codon:
        stats["sigma_codon"] = torch.exp(log_sigmas["codon"]).detach()
    if has_encode:
        stats["sigma_encode"] = torch.exp(log_sigmas["encode"]).detach()
    track_names = batch.get("encode_track_names")
    if isinstance(track_names, list) and len(track_names) == int(encode_track_losses.shape[0]):
        for track_name, track_loss in zip(track_names, encode_track_losses, strict=False):
            stats[f"loss_encode_{track_name}"] = track_loss.detach()
    return total, stats


class GradNormBalancer:
    """Loss-ratio approximation of GradNorm (Chen et al. 2018).

    Adjusts per-task loss weights based on relative training rate.
    Tasks converging fast get lower weight; lagging tasks get higher weight.
    Uses moving-average loss ratios as a proxy for gradient norms,
    avoiding expensive per-task backward passes.
    """

    def __init__(
        self,
        task_keys: tuple[str, ...] | list[str],
        alpha: float = 1.5,
        ema_decay: float = 0.99,
    ) -> None:
        self.task_keys = task_keys
        self.alpha = alpha
        self.ema_decay = ema_decay
        self._ema: dict[str, float] = {}
        self._initial: dict[str, float] = {}
        self._weights: dict[str, float] = {k: 1.0 for k in task_keys}

    def update(self, raw_losses: dict[str, float]) -> None:
        """Update EMA of per-task losses and recompute weights."""
        for key in self.task_keys:
            loss_key = f"loss_{key}"
            val = raw_losses.get(loss_key, 0.0)
            if val <= 0.0:
                continue

            if key not in self._initial:
                self._initial[key] = val
                self._ema[key] = val
            else:
                self._ema[key] = self.ema_decay * self._ema[key] + (1.0 - self.ema_decay) * val

        if not self._initial:
            return

        # Inverse training rate: tasks with loss still close to initial get higher weight.
        inv_rates: dict[str, float] = {}
        for key in self.task_keys:
            if key in self._initial and self._initial[key] > 0:
                inv_rates[key] = (self._ema[key] / self._initial[key]) ** self.alpha

        if not inv_rates:
            return

        mean_rate = sum(inv_rates.values()) / len(inv_rates)
        if mean_rate <= 0:
            return

        n = len(self.task_keys)
        for key in self.task_keys:
            if key in inv_rates:
                self._weights[key] = n * inv_rates[key] / (sum(inv_rates.values()))
            else:
                self._weights[key] = 1.0

    def get_weights(self) -> dict[str, float]:
        """Return current per-task weight dict."""
        return dict(self._weights)
