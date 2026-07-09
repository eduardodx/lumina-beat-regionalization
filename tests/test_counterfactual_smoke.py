from __future__ import annotations

from typing import Any

import pytest
import torch

from src.constants import (
    AA_NON_CDS,
    ALLELE_EFFECT_MISSENSE_NONCONSERVATIVE,
    ALLELE_EFFECT_SYNONYMOUS,
    DNA_VOCAB,
    MASK_ID,
    MUTATION_EFFECT_IGNORE_INDEX,
    NUM_ALLELE_EFFECT_CLASSES,
    PAD_ID,
    VOCAB_SIZE,
)
from src.dataset import _build_counterfactual_batch
from src.objectives import compute_multitask_loss, compute_uncertainty_weighted_loss, counterfactual_disagreement_loss
from src.precision import PrecisionPolicy
from src.train import (
    TrainConfig,
    prepare_allele_scoring_batch,
    resolve_auxiliary_compute_state,
    run_model_step,
)


def _test_precision() -> PrecisionPolicy:
    return PrecisionPolicy(requested="fp32", resolved="fp32", use_autocast=False)


def test_counterfactual_local_margin_is_nonnegative_and_bounded() -> None:
    h_ref = torch.zeros(1, 3, 2, dtype=torch.float32)
    h_ref[0, :, 0] = 1.0
    h_alt_high_similarity = h_ref.clone()
    h_alt_low_similarity = h_ref.clone()
    h_alt_low_similarity[0, 1] = torch.tensor([0.0, 1.0])
    edit_position = torch.tensor([1], dtype=torch.long)
    cf_active = torch.tensor([True], dtype=torch.bool)

    high_loss, high_far, high_cosine, high_distance = counterfactual_disagreement_loss(
        h_ref,
        h_alt_high_similarity,
        edit_position,
        cf_active,
        radius=0,
        far_radius=1,
        local_similarity_target=0.8,
    )
    low_loss, low_far, low_cosine, low_distance = counterfactual_disagreement_loss(
        h_ref,
        h_alt_low_similarity,
        edit_position,
        cf_active,
        radius=0,
        far_radius=1,
        local_similarity_target=0.8,
    )

    assert high_loss.item() == pytest.approx(0.2)
    assert low_loss.item() == 0.0
    assert low_loss <= high_loss
    assert high_far.item() == 0.0
    assert low_far.item() == 0.0
    assert high_cosine.item() == 1.0
    assert low_cosine.item() == 0.0
    assert high_distance.item() == 0.0
    assert low_distance.item() == 0.0


def test_counterfactual_local_margin_stops_below_target() -> None:
    h_ref = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    h_alt = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32, requires_grad=True)
    edit_position = torch.tensor([0], dtype=torch.long)
    cf_active = torch.tensor([True], dtype=torch.bool)

    local_loss, far_loss, _local_cosine, _far_distance = counterfactual_disagreement_loss(
        h_ref,
        h_alt,
        edit_position,
        cf_active,
        radius=0,
        far_radius=1,
        local_similarity_target=0.8,
    )
    (local_loss + far_loss).backward()

    assert local_loss.item() == 0.0
    assert h_alt.grad is not None
    assert torch.count_nonzero(h_alt.grad).item() == 0


def test_counterfactual_zero_active_loss_keeps_alt_graph_connected() -> None:
    h_ref = torch.randn(2, 16, 4, requires_grad=True)
    h_alt = torch.randn(2, 16, 4, requires_grad=True)
    edit_position = torch.tensor([3, 11], dtype=torch.long)
    cf_active = torch.tensor([False, False], dtype=torch.bool)

    local_loss, far_loss, local_cosine, far_distance = counterfactual_disagreement_loss(
        h_ref,
        h_alt,
        edit_position,
        cf_active,
    )
    loss = local_loss + far_loss
    loss.backward()

    assert h_ref.grad is not None
    assert h_alt.grad is not None
    assert local_cosine.item() == 0.0
    assert far_distance.item() == 0.0
    assert torch.count_nonzero(h_ref.grad).item() == 0
    assert torch.count_nonzero(h_alt.grad).item() == 0


def test_counterfactual_batch_edits_reference_not_masked_token() -> None:
    reference = torch.tensor([[DNA_VOCAB["A"], DNA_VOCAB["C"], DNA_VOCAB["N"], DNA_VOCAB["N"]]])
    corrupted = reference.clone()
    corrupted[0, 1] = MASK_ID
    mask_positions = torch.tensor([[False, True, False, False]])
    mutation_effect_labels = torch.full((1, 4, 4), MUTATION_EFFECT_IGNORE_INDEX, dtype=torch.long)

    alt_input_ids, _alt_attention_mask, edit_position, cf_active = _build_counterfactual_batch(
        reference_input_ids=reference,
        corrupted_input_ids=corrupted,
        attention_mask=torch.ones_like(reference),
        mask_positions=mask_positions,
        phylo100=torch.ones((1, 4), dtype=torch.float32),
        structure_labels=torch.zeros((1, 4), dtype=torch.long),
        mutation_effect_labels=mutation_effect_labels,
        counterfactual_fraction=1.0,
    )

    assert cf_active.tolist() == [True]
    assert edit_position.tolist() == [0]
    assert alt_input_ids[0, 0].item() != reference[0, 0].item()
    assert alt_input_ids[0, 1].item() == MASK_ID


def _allele_scoring_batch(batch_size: int = 4, seq_len: int = 12) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [[DNA_VOCAB["A"], DNA_VOCAB["C"], DNA_VOCAB["G"], DNA_VOCAB["T"]] * (seq_len // 4)]
        * batch_size,
        dtype=torch.long,
    )
    alt_input_ids = input_ids.unsqueeze(1).repeat(1, 3, 1)
    allele_position = torch.tensor([2, 4, 7, 9], dtype=torch.long)[:batch_size]
    allele_alt_ids = torch.tensor(
        [[DNA_VOCAB["A"], DNA_VOCAB["C"], DNA_VOCAB["T"]]] * batch_size,
        dtype=torch.long,
    )
    for row in range(batch_size):
        for alt_slot in range(3):
            alt_input_ids[row, alt_slot, allele_position[row]] = allele_alt_ids[row, alt_slot]

    labels = torch.full((batch_size, 3), -100, dtype=torch.long)
    severity = torch.zeros(batch_size, 3, dtype=torch.float32)
    valid = torch.zeros(batch_size, 3, dtype=torch.bool)
    if batch_size > 0:
        labels[0, 1] = ALLELE_EFFECT_SYNONYMOUS
        severity[0, 1] = 0.1
        valid[0, 1] = True
    if batch_size > 2:
        labels[2, 0] = ALLELE_EFFECT_MISSENSE_NONCONSERVATIVE
        severity[2, 0] = 0.8
        valid[2, 0] = True
    if batch_size > 3:
        labels[3, 0] = ALLELE_EFFECT_SYNONYMOUS
        labels[3, 2] = ALLELE_EFFECT_MISSENSE_NONCONSERVATIVE
        severity[3, 0] = 0.2
        severity[3, 2] = 0.9
        valid[3, 0] = True
        valid[3, 2] = True

    return {
        "allele_ref_input_ids": input_ids,
        "allele_alt_input_ids": alt_input_ids,
        "allele_position": allele_position,
        "allele_alt_ids": allele_alt_ids,
        "allele_effect_labels": labels,
        "allele_severity_targets": severity,
        "allele_valid_mask": valid,
        "allele_locus_ids": torch.arange(batch_size, dtype=torch.long).unsqueeze(1).repeat(1, 3),
    }


def test_prepare_allele_scoring_batch_filters_caps_samples_and_crops() -> None:
    cfg = TrainConfig(allele_max_rows_per_batch=2, allele_score_window=4)
    batch = _allele_scoring_batch()

    prepared = prepare_allele_scoring_batch(batch, cfg, rank_step=False)

    assert prepared is not None
    assert prepared.active_rows == 2
    assert prepared.scored_alts == 2
    assert prepared.score_seq_len == 4
    assert prepared.scorer_kwargs["ref_input_ids"].shape == (2, 4)
    assert prepared.scorer_kwargs["alt_input_ids"].shape == (2, 1, 4)
    assert prepared.loss_batch["allele_effect_labels"].shape == (2, 1)
    assert prepared.loss_batch["allele_position"].tolist() == [2, 2]


def test_prepare_allele_scoring_batch_keeps_all_alts_on_rank_step() -> None:
    cfg = TrainConfig(allele_max_rows_per_batch=1, allele_score_window=6)
    batch = _allele_scoring_batch()

    prepared = prepare_allele_scoring_batch(batch, cfg, rank_step=True)

    assert prepared is not None
    assert prepared.active_rows == 1
    assert prepared.scored_alts == 3
    assert prepared.scorer_kwargs["alt_input_ids"].shape == (1, 3, 6)
    assert prepared.loss_batch["allele_valid_mask"].tolist() == [[True, False, True]]


def test_prepare_allele_scoring_batch_returns_none_without_active_rows() -> None:
    cfg = TrainConfig(allele_max_rows_per_batch=2, allele_score_window=4)
    batch = _allele_scoring_batch()
    batch["allele_effect_labels"].fill_(-100)
    batch["allele_valid_mask"].fill_(False)

    assert prepare_allele_scoring_batch(batch, cfg, rank_step=False) is None


def test_alternating_auxiliary_schedule_train_and_eval_modes() -> None:
    cfg = TrainConfig(
        w_allele=0.35,
        w_counterfactual=0.10,
        auxiliary_schedule="alternate_allele_counterfactual",
        allele_rank_every_n_steps=4,
    )

    step1 = resolve_auxiliary_compute_state(cfg, 1, grad_enabled=True, has_allele=True, has_counterfactual=True)
    step2 = resolve_auxiliary_compute_state(cfg, 2, grad_enabled=True, has_allele=True, has_counterfactual=True)
    step7 = resolve_auxiliary_compute_state(cfg, 7, grad_enabled=True, has_allele=True, has_counterfactual=True)
    eval_state = resolve_auxiliary_compute_state(cfg, 2, grad_enabled=False, has_allele=True, has_counterfactual=True)
    allele_only = resolve_auxiliary_compute_state(cfg, 2, grad_enabled=True, has_allele=True, has_counterfactual=False)

    assert (step1.allele, step1.counterfactual, step1.allele_rank_step) == (True, False, False)
    assert (step2.allele, step2.counterfactual, step2.allele_rank_step) == (False, True, False)
    assert (step7.allele, step7.counterfactual, step7.allele_rank_step) == (True, False, True)
    assert (eval_state.allele, eval_state.counterfactual) == (True, True)
    assert (allele_only.allele, allele_only.counterfactual) == (True, False)


class _CountingAlleleModel(torch.nn.Module):
    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.hidden_dim = hidden_dim
        self.score_calls = 0
        self.counterfactual_calls = 0
        self.last_score_ref_shape: tuple[int, ...] | None = None
        self.last_score_alt_shape: tuple[int, ...] | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = False,
        return_hidden: bool = True,
        allele_ref_input_ids: torch.Tensor | None = None,
        allele_alt_input_ids: torch.Tensor | None = None,
        allele_position: torch.Tensor | None = None,
        allele_alt_ids: torch.Tensor | None = None,
        allele_attention_mask: torch.Tensor | None = None,
        allele_alt_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del attention_mask, return_hidden
        batch_size, seq_len = input_ids.shape
        base = self.weight * 0.0
        hidden = base + torch.zeros(batch_size, seq_len, self.hidden_dim, device=input_ids.device)
        outputs: dict[str, torch.Tensor] = {"hidden_states": hidden}
        if return_token_heads:
            outputs.update(
                {
                    "mlm_logits": base + torch.zeros(batch_size, seq_len, VOCAB_SIZE, device=input_ids.device),
                    "phylo100_pred": base + torch.zeros(batch_size, seq_len, device=input_ids.device),
                    "phylo470_pred": base + torch.zeros(batch_size, seq_len, device=input_ids.device),
                    "structure_logits": base + torch.zeros(batch_size, seq_len, 3, device=input_ids.device),
                }
            )
        else:
            self.counterfactual_calls += 1
        if return_sequence_embedding:
            outputs["sequence_embedding"] = hidden.mean(dim=1)
        if (
            allele_ref_input_ids is not None
            and allele_alt_input_ids is not None
            and allele_position is not None
            and allele_alt_ids is not None
            and allele_attention_mask is not None
            and allele_alt_attention_mask is not None
        ):
            outputs.update(
                self.score_alleles_from_ids(
                    ref_input_ids=allele_ref_input_ids,
                    alt_input_ids=allele_alt_input_ids,
                    allele_position=allele_position,
                    allele_alt_ids=allele_alt_ids,
                    attention_mask=allele_attention_mask,
                    alt_attention_mask=allele_alt_attention_mask,
                )
            )
        return outputs

    def score_alleles_from_ids(
        self,
        *,
        ref_input_ids: torch.Tensor,
        alt_input_ids: torch.Tensor,
        allele_position: torch.Tensor,
        allele_alt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        alt_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del allele_position, allele_alt_ids, attention_mask, alt_attention_mask
        self.score_calls += 1
        self.last_score_ref_shape = tuple(ref_input_ids.shape)
        self.last_score_alt_shape = tuple(alt_input_ids.shape)
        batch_size, num_alts, _seq_len = alt_input_ids.shape
        base = self.weight * 0.0
        return {
            "allele_effect_logits": base
            + torch.zeros(batch_size, num_alts, NUM_ALLELE_EFFECT_CLASSES, device=alt_input_ids.device),
            "allele_severity_score": base + torch.zeros(batch_size, num_alts, device=alt_input_ids.device),
            "allele_swap_severity_score": base + torch.zeros(batch_size, num_alts, device=alt_input_ids.device),
            "allele_far_distance": base + torch.zeros(batch_size, num_alts, device=alt_input_ids.device),
        }


def _run_model_step_batch() -> dict[str, Any]:
    allele_batch = _allele_scoring_batch()
    input_ids = allele_batch["allele_ref_input_ids"].clone()
    batch_size, seq_len = input_ids.shape
    return {
        **allele_batch,
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "alt_input_ids": input_ids.clone(),
        "alt_attention_mask": torch.ones_like(input_ids),
        "edit_position": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "cf_active": torch.tensor([True, True, True, True], dtype=torch.bool),
        "mlm_labels": torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long),
        "phylo100": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "phylo470": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "aux_valid_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "structure_labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "region_labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "aa_labels": torch.full((batch_size, seq_len), AA_NON_CDS, dtype=torch.long),
        "mutation_effect_labels": torch.full((batch_size, seq_len, 4), MUTATION_EFFECT_IGNORE_INDEX),
        "codon_labels": torch.full((batch_size, seq_len), -100, dtype=torch.long),
        "codon_phylo_target": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "encode_targets": torch.zeros(batch_size, seq_len, 0, dtype=torch.float32),
        "n_fraction": 0.0,
        "splice_positive_fraction": 0.0,
        "splice_core_fraction": 0.0,
        "exon_fraction": 0.0,
        "cds_fraction": 0.0,
        "utr_fraction": 0.0,
        "intron_fraction": 1.0,
        "n_filter_fallback_fraction": 0.0,
        "mask_density_intergenic": 0.0,
        "mask_density_intron": 0.0,
        "mask_density_noncoding_exon": 0.0,
        "mask_density_utr": 0.0,
        "mask_density_cds": 0.0,
    }


def test_run_model_step_caps_and_crops_allele_scorer_inputs() -> None:
    model = _CountingAlleleModel()
    cfg = TrainConfig(
        w_mlm=0.0,
        w_phylo100=0.0,
        w_phylo470=0.0,
        w_structure=0.0,
        w_allele=0.35,
        w_counterfactual=0.10,
        counterfactual_weighting="fixed",
        auxiliary_schedule="alternate_allele_counterfactual",
        allele_max_rows_per_batch=2,
        allele_score_window=4,
        allele_rank_every_n_steps=4,
        loss_balancing="fixed",
    )

    _loss, stats, _metrics = run_model_step(
        model,
        _run_model_step_batch(),
        torch.device("cpu"),
        cfg,
        _test_precision(),
        step=1,
    )

    assert model.score_calls == 1
    assert model.counterfactual_calls == 0
    assert model.last_score_ref_shape == (2, 4)
    assert model.last_score_alt_shape == (2, 1, 4)
    assert stats["allele_active_rows"] == pytest.approx(2.0)
    assert stats["allele_scored_alts"] == pytest.approx(2.0)
    assert stats["allele_score_seq_len"] == pytest.approx(4.0)
    assert stats["allele_compute_active"] == pytest.approx(1.0)
    assert stats["counterfactual_compute_active"] == pytest.approx(0.0)


def test_run_model_step_routes_v8_allele_scoring_through_forward() -> None:
    model = _CountingAlleleModel()
    cfg = TrainConfig(
        model="beat-v8",
        w_mlm=0.0,
        w_phylo100=0.0,
        w_phylo470=0.0,
        w_structure=0.0,
        w_allele=0.35,
        w_counterfactual=0.10,
        counterfactual_weighting="fixed",
        auxiliary_schedule="alternate_allele_counterfactual",
        allele_max_rows_per_batch=2,
        allele_score_window=4,
        loss_balancing="fixed",
    )

    _loss, stats, _metrics = run_model_step(
        model,
        _run_model_step_batch(),
        torch.device("cpu"),
        cfg,
        _test_precision(),
        step=1,
    )

    assert model.score_calls == 1
    assert model.last_score_ref_shape == (2, 4)
    assert model.last_score_alt_shape == (2, 1, 4)
    assert stats["allele_compute_active"] == pytest.approx(1.0)


def test_run_model_step_alternates_to_counterfactual_without_allele_scorer() -> None:
    model = _CountingAlleleModel()
    cfg = TrainConfig(
        w_mlm=0.0,
        w_phylo100=0.0,
        w_phylo470=0.0,
        w_structure=0.0,
        w_allele=0.35,
        w_counterfactual=0.10,
        counterfactual_weighting="fixed",
        auxiliary_schedule="alternate_allele_counterfactual",
        loss_balancing="fixed",
    )

    _loss, stats, _metrics = run_model_step(
        model,
        _run_model_step_batch(),
        torch.device("cpu"),
        cfg,
        _test_precision(),
        step=2,
    )

    assert model.score_calls == 0
    assert model.counterfactual_calls == 1
    assert stats["allele_compute_active"] == pytest.approx(0.0)
    assert stats["counterfactual_compute_active"] == pytest.approx(1.0)


def test_fixed_counterfactual_uncertainty_path_does_not_require_sigma_counterfactual() -> None:
    batch_size = 1
    seq_len = 3
    hidden_dim = 2
    log_sigmas: dict[str, torch.Tensor] = {
        key: torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        for key in ("mlm", "phylo100", "phylo470", "structure")
    }
    outputs = {
        "mlm_logits": torch.zeros(batch_size, seq_len, VOCAB_SIZE, dtype=torch.float32),
        "phylo100_pred": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "phylo470_pred": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "structure_logits": torch.zeros(batch_size, seq_len, 3, dtype=torch.float32),
        "hidden_states": torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32),
    }
    batch = {
        "mlm_labels": torch.tensor([[DNA_VOCAB["A"], PAD_ID, PAD_ID]], dtype=torch.long),
        "phylo100": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "phylo470": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "aux_valid_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "structure_labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "region_labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "aa_labels": torch.full((batch_size, seq_len), AA_NON_CDS, dtype=torch.long),
        "edit_position": torch.tensor([1], dtype=torch.long),
        "cf_active": torch.tensor([False], dtype=torch.bool),
    }

    loss, stats = compute_uncertainty_weighted_loss(
        outputs=outputs,
        batch=batch,
        log_sigmas=log_sigmas,
        alt_outputs={"hidden_states": outputs["hidden_states"].clone()},
        fixed_counterfactual_weight=0.25,
    )
    loss.backward()

    assert "sigma_counterfactual" not in stats
    assert stats["loss_counterfactual"].item() == 0.0
    assert stats["counterfactual_effective_weight"].item() == 0.25
    assert all(param.grad is not None for param in log_sigmas.values())


def _allele_loss_fixture() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch_size = 1
    seq_len = 2
    outputs = {
        "mlm_logits": torch.zeros(batch_size, seq_len, VOCAB_SIZE, dtype=torch.float32, requires_grad=True),
        "phylo100_pred": torch.zeros(batch_size, seq_len, dtype=torch.float32, requires_grad=True),
        "phylo470_pred": torch.zeros(batch_size, seq_len, dtype=torch.float32, requires_grad=True),
        "structure_logits": torch.zeros(batch_size, seq_len, 3, dtype=torch.float32, requires_grad=True),
        "allele_effect_logits": torch.tensor(
            [[[2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
              [-1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, -1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "allele_severity_score": torch.tensor([[0.1, 0.8]], dtype=torch.float32, requires_grad=True),
        "allele_swap_severity_score": torch.tensor([[-0.1, -0.8]], dtype=torch.float32, requires_grad=True),
        "allele_far_distance": torch.tensor([[0.01, 0.02]], dtype=torch.float32, requires_grad=True),
    }
    batch = {
        "mlm_labels": torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long),
        "phylo100": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "phylo470": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "aux_valid_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "structure_labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "region_labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "aa_labels": torch.full((batch_size, seq_len), AA_NON_CDS, dtype=torch.long),
        "allele_effect_labels": torch.tensor(
            [[ALLELE_EFFECT_SYNONYMOUS, ALLELE_EFFECT_MISSENSE_NONCONSERVATIVE]],
            dtype=torch.long,
        ),
        "allele_severity_targets": torch.tensor([[0.0, 0.9]], dtype=torch.float32),
        "allele_valid_mask": torch.tensor([[True, True]], dtype=torch.bool),
        "mutation_effect_labels": torch.full((batch_size, seq_len, 4), MUTATION_EFFECT_IGNORE_INDEX),
        "codon_labels": torch.full((batch_size, seq_len), -100, dtype=torch.long),
        "codon_phylo_target": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "encode_targets": torch.zeros(batch_size, seq_len, 0, dtype=torch.float32),
    }
    return outputs, batch


def test_allele_loss_contributes_to_fixed_multitask_loss() -> None:
    outputs, batch = _allele_loss_fixture()

    loss_without, _ = compute_multitask_loss(outputs, batch, w_mlm=0.0, w_structure=0.0, w_allele=0.0)
    loss_with, stats = compute_multitask_loss(outputs, batch, w_mlm=0.0, w_structure=0.0, w_allele=0.35)

    assert loss_with.item() > loss_without.item()
    assert stats["loss_allele"].item() > 0.0


def test_allele_loss_contributes_to_uncertainty_multitask_loss() -> None:
    outputs, batch = _allele_loss_fixture()
    log_sigmas: dict[str, torch.Tensor] = {
        key: torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        for key in ("mlm", "phylo100", "phylo470", "structure", "allele")
    }

    loss, stats = compute_uncertainty_weighted_loss(outputs, batch, log_sigmas)
    loss.backward()

    assert stats["loss_allele"].item() > 0.0
    assert "sigma_allele" in stats
    assert log_sigmas["allele"].grad is not None
