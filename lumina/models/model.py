"""Lumina DNA foundation model.

A distinct, focused variant of the Lumina family. It reuses the proven Lumina
hourglass backbone (multi-kernel stem + purity branch, 4x downsample, bidirectional
Mamba mid-stack with local/sparse attention, gated upsample) and exposes ONLY the
per-position Lumina supervised heads:

    MLM, conservation (scalar + bins + variant-delta), splice, region, counterfactual
    ref/alt, ESM-2 missense-severity, weak population-prior (AF + rank), ENCODE regulatory
    (regional bottleneck), and the Hi-C contact band (long-range, cell-conditioned FiLM).

The current architecture omits the pooled sequence-embedding head and every reverse-complement
representation objective (RC-Barlow / rc-consistency / sequence-summary / seq-contrastive /
VICReg): the pooled embedding rank-collapsed and all Lumina benchmarks read PER-POSITION
hidden states, so it carried no headline. The coupling and edit-localization heads (dead
objectives, offline AUROC below chance) are gone too. The mut/edit-mask plumbing is kept
(counterfactual + conservation-delta need it). Nothing here builds an unused parameter, so
DDP static_graph stays valid.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumina.constants import PAD_ID, SNV_BASES
from lumina.models.backbone import (
    DownStage,
    MultiKernelStem,
    SinusoidalPositionEmbedding,
    UpStage,
)
from lumina.models.losses import LUMINA_LOSS_NAMES, LossEMANormalizer
from lumina.models.midstack import LuminaBackboneConfig, LuminaMidStack


@dataclass(frozen=True)
class LuminaParameterPartitions:
    shared: tuple[nn.Parameter, ...]
    counterfactual_exclusive: tuple[nn.Parameter, ...]
    missense_exclusive: tuple[nn.Parameter, ...]
    other_head_exclusive: tuple[nn.Parameter, ...]

    @property
    def all_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.shared + self.counterfactual_exclusive + self.missense_exclusive + self.other_head_exclusive


@dataclass
class LuminaConfig(LuminaBackboneConfig):
    """Config for the Lumina model.

    Subclasses ``LuminaBackboneConfig`` so every existing Lumina ``model_config`` key
    validates (the registry rejects unknown keys), and adds the Lumina-only fields.
    """

    num_conservation_bins: int = 16

    # Shakeout: use constant main-phase loss weights (no warmup/ramp/polish) so the EMA-normalized
    # total loss + grad-norm stay flat over a short run and every objective is active from step 0.
    constant_loss_weights_enabled: bool = False

    # Build-out §4a-ii — Hi-C within-window contact head (Akita/Orca-style). Reads the mid (regional)
    # representation, pools it to 2 kb bins, and predicts the near-diagonal log-O/E contact band via a
    # bilinear (left·right) interaction → direct supervision pressure on the long-range pathway.
    # hic_head_enabled=False → head not constructed (backbone bit-identical); gated additionally by w_hic.
    hic_head_enabled: bool = False
    hic_resolution_bp: int = 2000  # contact-bin size (matches the data/derived/hic_2kb cache)
    hic_num_offsets: int = 16  # channel count: offsets 0..15; cache storage may be a declared superset
    hic_proj_dim: int = 64  # rank of the bilinear contact interaction

    # Backlog item 3 — ENCODE regulatory head bottleneck. The default expressive head
    # (LN→Linear(d_model,d_model)→GELU→Drop→Linear(d_model,tracks)) is expressive enough to absorb the
    # ENCODE gradient in the head itself (frozen probe flat at ~0.49); a low-rank bottleneck forces the
    # signal into the backbone. regulatory_head_rank: >0 → LN→Linear(d_model,r)→GELU→Linear(r,tracks);
    # 0 → relaxed expressive head (default, bit-identical); <0 → pure linear probe LN→Linear(d_model,tracks).
    regulatory_head_rank: int = 0

    # T2-3 — ESM-2 missense-severity regression head. Per-position, 4 scalars (one per SNV_BASES alt)
    # → [B, L, 4], supervised by the data/derived/esm2_missense cache via a masked Huber loss. Built
    # only when True so the baseline backbone is bit-identical when off; gated additionally by the
    # w_missense_severity loss/data weight (set both together — validate_train_config enforces it).
    missense_severity_head_enabled: bool = False

    # T1-2a — conservation variant-delta. When True the forward also exposes conservation_delta_pred =
    # cons(mut) - cons(ref) at the synthetic-edit positions (the conservation head is linear, so this is
    # F.linear(edit_delta, conservation_head.weight) — NO new parameters, head reused). Supervised by the
    # conservation_delta loss term (magnitude tracks reference phyloP). Off ⇒ no extra output, bit-identical
    # backbone + head_outputs view; gated additionally by w_conservation_delta (validate enforces pairing).
    conservation_delta_head_enabled: bool = False

    # T4-8b — per-loss gating (vs the EMA-floor's up-to-4x amplification of satisfied losses). Threaded
    # into LossEMANormalizer below. Default False ⇒ EMA normalization is byte-identical to the prior runs.
    per_loss_gating_enabled: bool = False

    # Item 5 — cell-type conditioning. Rides the Hi-C contact path (reads the mid representation pooled to
    # contact bins), so it requires hic_head_enabled. Default OFF ⇒ bit-identical to the pre-item-5 model.
    #
    # cell_conditioning_enabled applies a FiLM to the pooled, post-LayerNorm contact bins, conditioned on the
    # per-window sampled cell. current (F2): warm-started with a tiny nonzero init (see _cell_film below) so the
    # cell embedding gets gradient from step 0 instead of the old zero-init cold-start. The FiLM modules are
    # built only when the Hi-C head exists, so they stay inert on an unconditioned config.
    cell_conditioning_enabled: bool = False
    cell_cond_dim: int = 64
    n_cell_types: int = 4  # GM12878, K562, HepG2, IMR90 (vocab order = data.base.CELL_TYPES)

    # Step-1000 grad-explosion fix — Final RMSNorm on the trunk output H = cat(h_up, h_pure) (d_full).
    # The full-res token heads read `hidden` RAW (no per-head input norm), so an un-normalized H let the
    # readout/trunk gradients blow up (a data-order fluctuation → compounding escalation grad-clip could
    # only cap in magnitude, not direction). Normalizing H pins per-token ‖H‖ (RMS≈1) → conditioned
    # gradients. Default False ⇒ backbone bit-identical; True builds one nn.RMSNorm(d_full).
    trunk_final_norm_enabled: bool = False


class LuminaModel(nn.Module):
    def __init__(self, cfg: LuminaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = PAD_ID
        self.n_downsample_stages = cfg.resolved_downsample_stages()
        self.downsample_factor = 2**self.n_downsample_stages
        self.full_hidden_dim = cfg.d_model + cfg.d_pure
        if cfg.d_full != self.full_hidden_dim:
            raise ValueError(
                f"Lumina d_full must equal d_model + d_pure; got d_full={cfg.d_full} "
                f"and d_model + d_pure={self.full_hidden_dim}."
            )
        if cfg.resolved_position_encoding() != "sinusoidal":
            raise ValueError("Lumina supports sinusoidal position_encoding only.")

        # --- backbone (identical to Lumina) ---
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos_emb = SinusoidalPositionEmbedding(cfg.l_max, cfg.d_model)
        self.stem = MultiKernelStem(cfg.d_model, cfg.d_pure, cfg.conv_stem_kernels)
        self.down_stages = nn.ModuleList([DownStage(cfg.d_model) for _ in range(self.n_downsample_stages)])
        self.mid_stack = LuminaMidStack(cfg)
        self.up_stages = nn.ModuleList([UpStage(cfg.d_model) for _ in range(self.n_downsample_stages)])
        self.dropout = nn.Dropout(cfg.dropout)

        d_full = self.full_hidden_dim
        # Step-1000 grad-explosion fix: optional Final RMSNorm on the trunk output H = cat(h_up, h_pure).
        # The full-res token heads read `hidden` RAW, so an un-normed H let readout/trunk gradients blow up
        # (grad-clip bounds magnitude, not direction). Normalizing H pins ‖H‖ (per-token RMS≈1) so those
        # gradients stay conditioned. Default off ⇒ backbone bit-identical; the mid-reading heads
        # (regulatory/Hi-C) already self-normalize their input, so only `hidden` needs this.
        self.trunk_final_norm: nn.Module | None = nn.RMSNorm(d_full) if cfg.trunk_final_norm_enabled else None
        # --- Lumina heads only ---
        self.mlm_head = nn.Linear(d_full, len(SNV_BASES))
        self.conservation_scalar_head = nn.Linear(d_full, cfg.num_conservation_targets)
        self.conservation_bin_head = nn.Linear(d_full, cfg.num_conservation_bins)
        self.splice_class_head = nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, cfg.num_splice_classes))
        self.splice_distance_head = nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, 1))
        self.region_head = nn.Linear(d_full, cfg.num_region_classes)
        # Counterfactual ref/alt head: per-position SNV consequence over precomputed labels.
        self.counterfactual_snv_head = nn.Linear(d_full, len(SNV_BASES) * cfg.num_counterfactual_effect_classes)
        # T2-3 ESM-2 missense-severity regression head: per-position, one scalar per SNV_BASES alt → [B,L,4].
        # Built only when enabled (baseline backbone bit-identical when off); runs every forward when built
        # so its params stay in the DDP static graph (the loss emits a graph-connected zero on empty batches).
        # current: 2-layer MLP (hidden 64) instead of a bare Linear — the DNA→protein-severity map the ESM-2
        # distillation target teaches needs capacity a linear readout lacks. Mirrors the splice_class_head idiom.
        self.missense_severity_head: nn.Module | None = (
            nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, len(SNV_BASES)))
            if cfg.missense_severity_head_enabled
            else None
        )
        # Weak population-prior head (the single consolidated gnomAD head): observed + log-AF.
        self.population_af_head = nn.Linear(d_full, len(SNV_BASES))
        self.population_observed_head = nn.Linear(d_full, len(SNV_BASES))
        # --- Build-out §4b: ENCODE regulatory head (regional scale) ---
        # Reads the mid representation (d_model channels, 4 bp/token); pooled x2 in forward to the 8 bp
        # ENCODE grid, then predicts the 50 tracks. Constructed only when tracks are requested so the
        # baseline backbone is unchanged when off (num_regulatory_tracks=0).
        if cfg.num_regulatory_tracks > 0:
            r = int(cfg.regulatory_head_rank)
            if r > 0:
                # item 3: low-rank bottleneck — forces the ENCODE signal into the backbone.
                self.regulatory_head: nn.Module | None = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, r),
                    nn.GELU(),
                    nn.Linear(r, cfg.num_regulatory_tracks),
                )
            elif r < 0:
                # item 3: pure linear probe (strongest bottleneck — no head nonlinearity at all).
                self.regulatory_head = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.num_regulatory_tracks),
                )
            else:
                # relaxed expressive head (default; bit-identical to pre-item-3 runs).
                self.regulatory_head = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.d_model, cfg.num_regulatory_tracks),
                )
        else:
            self.regulatory_head = None

        # --- Build-out §4a-ii: Hi-C contact head (long-range scale) ---
        # Pools the mid representation to hic_resolution_bp bins and predicts the near-diagonal band
        # via a bilinear interaction band[d, k] = scale * (left[d] · right[d+k]) + offset_bias[k].
        if cfg.hic_head_enabled:
            self.hic_norm: nn.Module | None = nn.LayerNorm(cfg.d_model)
            self.hic_left: nn.Module | None = nn.Linear(cfg.d_model, cfg.hic_proj_dim)
            self.hic_right: nn.Module | None = nn.Linear(cfg.d_model, cfg.hic_proj_dim)
            self.hic_offset_bias: nn.Parameter | None = nn.Parameter(torch.zeros(cfg.hic_num_offsets))
            self._hic_scale = float(cfg.hic_proj_dim) ** -0.5
        else:
            self.hic_norm = None
            self.hic_left = None
            self.hic_right = None
            self.hic_offset_bias = None
            self._hic_scale = 1.0

        # --- Item 5: cell-type conditioning (FiLM) on the Hi-C contact bins ---
        # Both ride the Hi-C contact path. The FiLM is ZERO-INITIALIZED so gamma=beta=0 at step 0 ⇒
        # FiLM(x) = x·(1+gamma)+beta = x (identity): the Hi-C band output is bit-identical to the no-conditioning
        # model at init, and only diverges as the cell embedding learns. It conditions the POOLED,
        # post-LayerNorm bins (a LayerNorm AFTER the FiLM would cancel gamma/beta — see _contact_bins). Built only
        # when a consumer head exists so an unconditioned config is unchanged.
        self._cell_conditioning = bool(cfg.cell_conditioning_enabled) and cfg.hic_head_enabled
        if self._cell_conditioning:
            self.cell_embedding: nn.Module | None = nn.Embedding(int(cfg.n_cell_types), int(cfg.cell_cond_dim))
            _cell_film = nn.Linear(int(cfg.cell_cond_dim), 2 * cfg.d_model)
            # current (F2 "Fix E"): warm-start the FiLM near-identity with a tiny nonzero weight init instead of the
            # old zero-init cold-start (gamma=beta=0 gave the cell embedding no gradient until cell_film moved
            # off zero). The small std keeps the Hi-C band ~identical at init while letting the cell signal flow
            # from step 0; bias stays 0 so gamma≈beta≈0 ⇒ bins·(1+gamma)+beta ≈ bins.
            nn.init.normal_(_cell_film.weight, std=1e-3)
            nn.init.zeros_(_cell_film.bias)
            self.cell_film: nn.Module | None = _cell_film
        else:
            self.cell_embedding = None
            self.cell_film = None

        # EMA per-loss normalizer (plan §10). Lives in the model so its buffers are
        # checkpointed/restored automatically and survive spot resume. No parameters,
        # so it is inert for DDP gradient sync. T4-8b per-loss gating is threaded from the config
        # (default False ⇒ normalization byte-identical to the prior runs).
        self.loss_ema = LossEMANormalizer(
            LUMINA_LOSS_NAMES, per_loss_gating_enabled=bool(cfg.per_loss_gating_enabled)
        )

    def parameter_partitions(self) -> LuminaParameterPartitions:
        """Return a complete, disjoint partition based on owning modules."""

        def collect(modules: tuple[nn.Module | None, ...]) -> tuple[nn.Parameter, ...]:
            seen: set[int] = set()
            parameters: list[nn.Parameter] = []
            for module in modules:
                if module is None:
                    continue
                for parameter in module.parameters():
                    if parameter.requires_grad and id(parameter) not in seen:
                        seen.add(id(parameter))
                        parameters.append(parameter)
            return tuple(parameters)

        shared = collect(
            (
                self.token_emb,
                self.pos_emb,
                self.stem,
                self.down_stages,
                self.mid_stack,
                self.up_stages,
                self.trunk_final_norm,
            )
        )
        counterfactual = collect((self.counterfactual_snv_head,))
        missense = collect((self.missense_severity_head,))
        other = list(
            collect(
                (
                    self.mlm_head,
                    self.conservation_scalar_head,
                    self.conservation_bin_head,
                    self.splice_class_head,
                    self.splice_distance_head,
                    self.region_head,
                    self.population_af_head,
                    self.population_observed_head,
                    self.regulatory_head,
                    self.hic_norm,
                    self.hic_left,
                    self.hic_right,
                    self.cell_embedding,
                    self.cell_film,
                )
            )
        )
        if self.hic_offset_bias is not None and self.hic_offset_bias.requires_grad:
            other.append(self.hic_offset_bias)
        partitions = LuminaParameterPartitions(shared, counterfactual, missense, tuple(other))
        all_trainable = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        partition_ids = [id(parameter) for parameter in partitions.all_parameters]
        if len(partition_ids) != len(set(partition_ids)):
            raise RuntimeError("Lumina parameter partitions overlap.")
        if set(partition_ids) != {id(parameter) for parameter in all_trainable}:
            raise RuntimeError("Lumina parameter partitions do not cover every trainable parameter.")
        return partitions

    def _position_embeddings(self, seq_len: int, x: torch.Tensor) -> torch.Tensor:
        if seq_len > self.cfg.l_max:
            raise ValueError(f"Lumina seq_len={seq_len} exceeds l_max={self.cfg.l_max}.")
        return self.pos_emb(seq_len, device=x.device, dtype=x.dtype).unsqueeze(0)

    def _downsample_edit_mask(self, edit_mask: torch.Tensor, mid_len: int) -> torch.Tensor:
        import torch.nn.functional as F

        pooled = edit_mask.to(dtype=torch.float32).unsqueeze(1)
        for _ in range(self.n_downsample_stages):
            pooled = F.max_pool1d(pooled, kernel_size=2, stride=2, ceil_mode=True)
        if pooled.shape[-1] > mid_len:
            pooled = pooled[..., :mid_len]
        elif pooled.shape[-1] < mid_len:
            pooled = F.pad(pooled, (0, mid_len - pooled.shape[-1]))
        return pooled.squeeze(1).to(dtype=torch.bool)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        variant_edit_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        import torch.nn.functional as F  # noqa: F401  (parity with current; F used via UpStage)

        _batch_size, seq_len = input_ids.shape
        x = self.token_emb(input_ids)
        x = x + self._position_embeddings(seq_len, x)
        x = self.dropout(x)

        h_stem, h_pure = self.stem(x)
        skips: list[torch.Tensor] = [h_stem]
        h = h_stem
        for stage in self.down_stages:
            h = stage(h)
            skips.append(h)
        h_mid = h
        edit_mid_mask = (
            None if variant_edit_mask is None else self._downsample_edit_mask(variant_edit_mask, h_mid.shape[1])
        )
        h_mid = self.mid_stack(h_mid, edit_mid_mask=edit_mid_mask)

        h_up = h_mid
        for stage, skip in zip(self.up_stages, reversed(skips[:-1]), strict=True):
            h_up = stage(h_up, skip)
        if h_up.shape[1] != seq_len:
            h_up = UpStage._match_length(h_up, seq_len)

        # Always compute the variant-residual boost when a mask is present (zero-gated by torch.where: on
        # non-variant batches the mask is all-False => h_up returned unchanged, and no gradient flows through
        # `boost`). Dropping the `torch.any(...)` short-circuit keeps the autograd graph IDENTICAL across
        # iterations, so DDP static_graph=True is valid (the mut pass + edit-loc + conservation-delta heads
        # are already run every step). In training step.py always passes the (all-zero-on-non-variant) mask;
        # eval may pass None, which correctly skips this (no DDP there).
        if self.cfg.use_variant_token_residual and variant_edit_mask is not None:
            boost = self.cfg.variant_residual_gamma * h_stem
            h_up = torch.where(variant_edit_mask.unsqueeze(-1).to(dtype=torch.bool), h_up + boost, h_up)

        hidden = torch.cat([h_up, h_pure], dim=-1)
        # Step-1000 grad-explosion fix: bound ‖H‖ so the raw-reading full-res token heads get a normalized
        # readout (no-op when trunk_final_norm_enabled=False). `hidden` is a fresh tensor — normalizing it
        # here does not touch h_mid, the up-stack, or the (already self-normalizing) mid heads.
        if self.trunk_final_norm is not None:
            hidden = self.trunk_final_norm(hidden)
        encoded = {"last_hidden_state": hidden, "mid_hidden_state": h_mid}
        return encoded

    def _token_head_outputs(self, hidden: torch.Tensor, delta_hidden: torch.Tensor | None) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        outputs["mlm_logits"] = self.mlm_head(hidden)
        conservation = self.conservation_scalar_head(hidden)
        outputs["conservation_scalar_pred"] = conservation
        if conservation.shape[-1] > 0:
            outputs["phylo100_pred"] = conservation[..., 0]
        if conservation.shape[-1] > 1:
            outputs["zoo241_pred"] = conservation[..., 1]  # Zoonomia-241 (real, once zoo241_bw_path is fixed)
        if conservation.shape[-1] > 2:
            outputs["phylo470_pred"] = conservation[..., 2]
        outputs["conservation_bin_logits"] = self.conservation_bin_head(hidden)
        outputs["splice_class_logits"] = self.splice_class_head(hidden)
        outputs["splice_distance_pred"] = self.splice_distance_head(hidden).squeeze(-1)
        outputs["region_logits"] = self.region_head(hidden)
        cf_logits = self.counterfactual_snv_head(hidden)
        outputs["counterfactual_effect_logits"] = cf_logits.view(
            *cf_logits.shape[:2], len(SNV_BASES), self.cfg.num_counterfactual_effect_classes
        )
        if self.missense_severity_head is not None:
            outputs["missense_severity_pred"] = self.missense_severity_head(hidden)  # [B, L, 4]
        outputs["gnomad_af_pred"] = self.population_af_head(hidden)
        outputs["gnomad_observed_logits"] = self.population_observed_head(hidden)
        # Synthetic-edit delta (mutant-minus-reference hidden; zero on non-variant batches). Kept because the
        # conservation variant-delta below reuses it; computed every forward so the autograd graph is identical
        # across iterations (DDP static_graph safety). The edit-localization readout that used to consume it was
        # removed in current (dead objective, offline AUROC below chance).
        edit_delta = delta_hidden if delta_hidden is not None else torch.zeros_like(hidden)
        # T1-2a: predicted Δconservation under the variant. The conservation head is linear, so
        # cons(ref+delta) - cons(ref) = W·delta (bias cancels) — reuse its weight, no new params. Computed
        # every forward when enabled (edit_delta is zeros on non-variant batches) ⇒ DDP-static-graph-safe.
        if self.cfg.conservation_delta_head_enabled:
            outputs["conservation_delta_pred"] = F.linear(
                edit_delta, self.conservation_scalar_head.weight
            )  # [B, L, num_conservation_targets]
        return outputs

    def _contact_bins(
        self, h_mid: torch.Tensor, cell_id: torch.Tensor | None, window_start: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool the mid representation to hic_resolution_bp contact bins and (item 5) FiLM-condition them
        on the cell type. Returns [B, n_bins, C]. Read by the Hi-C band (item 5 supervision).

        current (F2a): the bins are ALIGNED TO THE GLOBAL 2 kb grid the target uses. The data reads contacts on
        the global grid (target bin d == global bin start//2000 + d), while the old model pooled 512-token
        (2048 bp) bins from window-bp-0 — a per-window RANDOM phase offset (start % 2000, avg half a bin) plus
        a 48 bp/bin scale drift that the model could not learn a correction for, forcing a blurred average and
        capping Pearson for every cell. Given the per-sample ``window_start`` we assign each mid token to its
        global 2 kb bin (res//downsample = 500 tokens/bin) and scatter-mean, so model bin d == target bin d.
        ``window_start=None`` (eval callers that don't thread it) falls back to the relaxed window-relative
        reshape-mean. The pool is done in fp32 (summing ~500 bf16 values loses precision).

        FiLM is applied AFTER the pool (current F2: warm-started near-identity — cell_film weight std 1e-3, bias 0,
        so gamma≈beta≈0 and the band is ~unchanged at init while the cell embedding gets gradient from step 0).
        """
        assert self.hic_norm is not None
        res = int(self.cfg.hic_resolution_bp)
        seq_len_bp = h_mid.shape[1] * self.downsample_factor
        n_bins = max(1, seq_len_bp // res)
        mid = self.hic_norm(h_mid)  # [B, mid_len, C]
        b, mid_len, c = mid.shape
        if window_start is None:
            # Evaluation fallback: window-relative reshape-mean from token 0 (res//downsample tokens per bin).
            k = max(1, mid_len // n_bins)
            bins = mid[:, : n_bins * k].reshape(b, n_bins, k, c).mean(dim=2)  # [B, n_bins, C]
        else:
            # Pool only tokens belonging to complete GLOBAL bins. Leading/trailing partial bins and
            # padded slots contribute zero; no token is clamped into the final prediction bin.
            ds = self.downsample_factor
            starts = window_start.to(mid.device).long().view(b, 1)
            first_complete_bp = ((starts + res - 1) // res) * res
            window_end_bp = starts + seq_len_bp
            exclusive_end_bp = torch.div(window_end_bp, res, rounding_mode="floor") * res
            valid_bins = torch.div(exclusive_end_bp - first_complete_bp, res, rounding_mode="floor").clamp(
                min=0, max=n_bins
            )
            token_bp = starts + torch.arange(mid_len, device=mid.device).view(1, mid_len) * ds
            rel_bp = token_bp - first_complete_bp
            raw_idx = torch.div(rel_bp, res, rounding_mode="floor")
            token_valid = (rel_bp >= 0) & (raw_idx >= 0) & (raw_idx < valid_bins)
            safe_idx = raw_idx.clamp(0, n_bins - 1)
            idx_c = safe_idx.unsqueeze(-1).expand(b, mid_len, c)
            weights = token_valid.unsqueeze(-1).to(torch.float32)
            sums = torch.zeros(b, n_bins, c, device=mid.device, dtype=torch.float32).scatter_add_(
                1, idx_c, mid.float() * weights
            )
            cnts = torch.zeros(b, n_bins, 1, device=mid.device, dtype=torch.float32).scatter_add_(
                1, safe_idx.unsqueeze(-1), weights
            )
            bins = (sums / cnts.clamp_min(1.0)).to(mid.dtype)
        if self._cell_conditioning and self.cell_film is not None and self.cell_embedding is not None:
            if cell_id is None:
                cell_id = bins.new_zeros(bins.shape[0], dtype=torch.long)  # default cell 0 (GM12878)
            cond = self.cell_film(self.cell_embedding(cell_id.to(torch.long)))  # [B, 2C]
            gamma, beta = cond.chunk(2, dim=-1)  # each [B, C]
            bins = bins * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # warm-started near-identity (F2)
        return bins

    def _hic_band(
        self, h_mid: torch.Tensor, cell_id: torch.Tensor | None = None, window_start: torch.Tensor | None = None
    ) -> torch.Tensor:
        """§4a-ii: predict the within-window near-diagonal contact band via a bilinear interaction
        band[d, k] = scale·(left[d]·right[d+k]) + offset_bias[k] over the (cell-conditioned) contact bins.
        n_bins is derived from the window length so the head is sequence-length agnostic."""
        assert self.hic_left is not None and self.hic_right is not None
        assert self.hic_offset_bias is not None
        n_off = int(self.cfg.hic_num_offsets)
        bins = self._contact_bins(h_mid, cell_id, window_start)  # [B, n_bins, C] (F2a: global-2kb-aligned)
        n_bins = bins.shape[1]
        left = self.hic_left(bins)  # [B, n_bins, r]
        right = self.hic_right(bins)  # [B, n_bins, r]
        contact = torch.einsum("bnr,bmr->bnm", left, right) * self._hic_scale  # [B, n_bins, n_bins]
        d_idx = torch.arange(n_bins, device=contact.device).view(n_bins, 1).expand(n_bins, n_off)
        k_idx = torch.arange(n_off, device=contact.device).view(1, n_off).expand(n_bins, n_off)
        e_idx = (d_idx + k_idx).clamp(max=n_bins - 1)  # within-window; out-of-band masked in loss
        band = contact[:, d_idx, e_idx]  # [B, n_bins, n_off]
        return band + self.hic_offset_bias.view(1, 1, n_off)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_hidden: bool = True,
        variant_edit_mask: torch.Tensor | None = None,
        delta_hidden: torch.Tensor | None = None,
        edit_delta_from_hidden_states: torch.Tensor | None = None,
        cell_id: torch.Tensor | None = None,
        hic_window_start: torch.Tensor | None = None,  # F2a: per-sample window start (bp) for global-2kb bin align
    ) -> dict[str, Any]:
        if attention_mask is None:
            attention_mask = input_ids.ne(PAD_ID)
        encoded = self.encode(input_ids, attention_mask=attention_mask, variant_edit_mask=variant_edit_mask)
        hidden = encoded["last_hidden_state"]
        h_mid = encoded["mid_hidden_state"]

        if delta_hidden is not None and edit_delta_from_hidden_states is not None:
            raise ValueError("Pass either delta_hidden or edit_delta_from_hidden_states, not both.")
        if edit_delta_from_hidden_states is not None:
            if edit_delta_from_hidden_states.shape != hidden.shape:
                raise ValueError(
                    "edit_delta_from_hidden_states must match hidden shape, got "
                    f"{tuple(edit_delta_from_hidden_states.shape)} and {tuple(hidden.shape)}."
                )
            delta_hidden = edit_delta_from_hidden_states - hidden

        outputs: dict[str, Any] = {
            "last_hidden_state": hidden,
            "mid_hidden_state": h_mid,
            "head_outputs": {},
        }
        if return_hidden:
            outputs["hidden_states"] = hidden
            outputs["mid_hidden_states"] = h_mid
        else:
            outputs["mid_hidden_states"] = h_mid

        if return_token_heads:
            head_outputs = self._token_head_outputs(hidden, delta_hidden)
            outputs.update(head_outputs)

        # Build-out upper-scale heads read the mid (regional) representation. Gated on return_token_heads
        # so they run on the main + rc passes (consistent across ranks) and are skipped on the mut pass.
        if return_token_heads and self.regulatory_head is not None:
            # §4b: mid is 4 bp/token; ENCODE targets are 8 bp → pool x2 to align before the per-track head.
            mid_t = h_mid.transpose(1, 2)
            pooled_mid = F.avg_pool1d(mid_t, kernel_size=2, stride=2).transpose(1, 2)
            outputs["regulatory_pred"] = self.regulatory_head(pooled_mid)
        if return_token_heads and self.hic_left is not None:
            outputs["hic_band_pred"] = self._hic_band(
                h_mid, cell_id, hic_window_start
            )  # §4a-ii: [B,n_bins,n_off] log-O/E band

        # Public plan-named head_outputs view.
        if return_token_heads:
            outputs["head_outputs"] = {
                "mlm": outputs.get("mlm_logits"),
                "conservation": {
                    "scalar": outputs.get("conservation_scalar_pred"),
                    "bins": outputs.get("conservation_bin_logits"),
                },
                "splice": {
                    "class": outputs.get("splice_class_logits"),
                    "distance": outputs.get("splice_distance_pred"),
                },
                "region": outputs.get("region_logits"),
                "counterfactual_variant": outputs.get("counterfactual_effect_logits"),
                "population_prior": {
                    "af": outputs.get("gnomad_af_pred"),
                    "observed": outputs.get("gnomad_observed_logits"),
                },
            }
            # T2-3: only exposed when the head is built, so the off-path head_outputs view is unchanged.
            if "missense_severity_pred" in outputs:
                outputs["head_outputs"]["missense_severity"] = outputs["missense_severity_pred"]
            # T1-2a: only exposed when enabled, so the off-path head_outputs view is unchanged.
            if "conservation_delta_pred" in outputs:
                outputs["head_outputs"]["conservation_delta"] = outputs["conservation_delta_pred"]
        return outputs


def build_lumina_model(cfg: LuminaConfig) -> LuminaModel:
    return LuminaModel(cfg)
