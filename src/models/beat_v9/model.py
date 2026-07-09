from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn

from src.constants import (
    COMPLEMENT_TABLE,
    NUM_AA_CLASSES,
    NUM_COUNTERFACTUAL_EFFECT_CLASSES,
    NUM_REGION_CLASSES,
    NUM_STRUCTURE_CLASSES,
    PAD_ID,
    VOCAB_SIZE,
)
from src.models.beat_shared import BeatBlock, resolve_chunk_size
from src.models.beat_v7.local_attn import LocalWindowAttention
from src.models.beat_v9.heads import (
    ClassificationHead,
    CounterfactualSNVHead,
    HiddenTapFusion,
    MLMHead,
    MultiKernelConvStem,
    MultiScaleContext,
    RegressionHead,
    SequenceEmbeddingHead,
)


@dataclass
class BeatV9Config:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    depth: int = 8
    n_layers: int | None = None
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 1.0
    chunk_size: int = 64
    is_mimo: bool = False
    mimo_rank: int = 4
    is_outproj_norm: bool = True
    dropout: float = 0.05
    use_rc_fusion: bool = True
    use_gated_fusion: bool | None = None
    activation_checkpointing: bool = False
    checkpoint_use_reentrant: bool = False

    use_local_attention: bool = True
    local_attention_every: int = 3
    local_attention_window: int = 256
    local_attention_heads: int = 4
    attention_every_n_blocks: int | None = None
    attention_window: int | None = None
    attention_n_heads: int | None = None

    use_local_conv_stem: bool = True
    conv_stem_kernels: list[int] = field(default_factory=lambda: [3, 6, 11, 21])
    conv_stem_dropout: float = 0.05

    hidden_tap_layers: list[int] = field(default_factory=lambda: [2, 4, 6, 8])

    use_counterfactual_snv_head: bool = True
    counterfactual_repr_dim: int = 128
    num_counterfactual_effect_classes: int = NUM_COUNTERFACTUAL_EFFECT_CLASSES
    counterfactual_context_radii: list[int] = field(default_factory=lambda: [3, 16, 64, 256])

    use_multiscale_context_head: bool = True
    multiscale_context_radii: list[int] = field(default_factory=lambda: [3, 16, 64, 256, 1024])

    use_sequence_embedding: bool = True
    sequence_embedding_dim: int = 256

    num_region_classes: int = NUM_REGION_CLASSES
    num_structure_classes: int = NUM_STRUCTURE_CLASSES
    num_aa_classes: int = NUM_AA_CLASSES
    num_conservation_bins: int = 7
    num_splice_dist_bins: int = 7
    num_codon_pos_classes: int = 5
    num_exon_phase_classes: int = 5
    num_encode_tracks: int = 0
    heads: dict[str, bool] = field(default_factory=dict)


class BeatV9Block(BeatBlock):
    pass


class DNAFoundationBeatV9(nn.Module):
    def __init__(self, cfg: BeatV9Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = PAD_ID

        n_layers = self._n_layers
        chunk_size = resolve_chunk_size(cfg)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        use_gated_fusion = cfg.use_rc_fusion if cfg.use_gated_fusion is None else cfg.use_gated_fusion
        self.blocks = nn.ModuleList(
            [
                BeatV9Block(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    expand=cfg.expand,
                    headdim=cfg.headdim,
                    ngroups=cfg.ngroups,
                    rope_fraction=cfg.rope_fraction,
                    chunk_size=chunk_size,
                    is_mimo=cfg.is_mimo,
                    mimo_rank=cfg.mimo_rank,
                    is_outproj_norm=cfg.is_outproj_norm,
                    dropout=cfg.dropout,
                    use_gated_fusion=use_gated_fusion,
                    activation_checkpointing=cfg.activation_checkpointing,
                    checkpoint_use_reentrant=cfg.checkpoint_use_reentrant,
                )
                for _ in range(n_layers)
            ]
        )

        self.local_conv_stem: MultiKernelConvStem | None = None
        if cfg.use_local_conv_stem:
            self.local_conv_stem = MultiKernelConvStem(
                cfg.d_model,
                kernels=cfg.conv_stem_kernels,
                dropout=cfg.conv_stem_dropout,
            )

        attention_every = self._attention_every
        num_attn_layers = 0
        if cfg.use_local_attention and attention_every > 0:
            num_attn_layers = max(0, (n_layers - 1) // attention_every)
        self.attn_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(num_attn_layers)])
        self.attn_layers = nn.ModuleList(
            [
                LocalWindowAttention(
                    d_model=cfg.d_model,
                    n_heads=self._attention_heads,
                    window=self._attention_window,
                    dropout=cfg.dropout,
                )
                for _ in range(num_attn_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.mlm_head = MLMHead(self.token_emb, cfg.d_model, cfg.vocab_size)
        self.phylo100_head = RegressionHead(cfg.d_model, cfg.dropout)
        self.phylo470_head = RegressionHead(cfg.d_model, cfg.dropout)
        self.conservation_bin_head = ClassificationHead(cfg.d_model, cfg.num_conservation_bins, cfg.dropout)
        self.region_head = ClassificationHead(cfg.d_model, cfg.num_region_classes, cfg.dropout)
        self.structure_head = ClassificationHead(cfg.d_model, cfg.num_structure_classes, cfg.dropout)
        self.donor_distance_head = ClassificationHead(cfg.d_model, cfg.num_splice_dist_bins, cfg.dropout)
        self.acceptor_distance_head = ClassificationHead(cfg.d_model, cfg.num_splice_dist_bins, cfg.dropout)
        self.aa_head = ClassificationHead(cfg.d_model, cfg.num_aa_classes, cfg.dropout)
        self.codon_head = nn.Linear(cfg.d_model, 64)
        self.codon_pos_head = ClassificationHead(cfg.d_model, cfg.num_codon_pos_classes, cfg.dropout)
        self.exon_phase_head = ClassificationHead(cfg.d_model, cfg.num_exon_phase_classes, cfg.dropout)
        self.codon_phylo_head = RegressionHead(cfg.d_model, cfg.dropout)

        self.hidden_tap_fusion = HiddenTapFusion(
            cfg.d_model,
            num_taps=len(cfg.hidden_tap_layers) + 1,
            dropout=cfg.dropout,
        )
        self.multi_scale_context: MultiScaleContext | nn.Identity
        if cfg.use_multiscale_context_head:
            self.multi_scale_context = MultiScaleContext(
                cfg.d_model,
                radii=cfg.multiscale_context_radii,
                dropout=cfg.dropout,
            )
        else:
            self.multi_scale_context = nn.Identity()
        self.counterfactual_snv_head: CounterfactualSNVHead | None = None
        if cfg.use_counterfactual_snv_head:
            self.counterfactual_snv_head = CounterfactualSNVHead(
                cfg.d_model,
                d_repr=cfg.counterfactual_repr_dim,
                num_effect_classes=cfg.num_counterfactual_effect_classes,
                dropout=cfg.dropout,
            )

        self.sequence_embedding_head: SequenceEmbeddingHead | None = None
        if cfg.use_sequence_embedding:
            self.sequence_embedding_head = SequenceEmbeddingHead(cfg.d_model, cfg.sequence_embedding_dim)

        self.encode_head: nn.Module | None = None
        if cfg.num_encode_tracks > 0:
            self.encode_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.num_encode_tracks),
            )

        self.register_buffer(
            "_complement_table",
            torch.tensor(COMPLEMENT_TABLE, dtype=torch.long),
            persistent=False,
        )

    @property
    def _n_layers(self) -> int:
        return int(self.cfg.n_layers if self.cfg.n_layers is not None else self.cfg.depth)

    @property
    def _attention_every(self) -> int:
        if self.cfg.attention_every_n_blocks is not None:
            return int(self.cfg.attention_every_n_blocks)
        return int(self.cfg.local_attention_every)

    @property
    def _attention_window(self) -> int:
        if self.cfg.attention_window is not None:
            return int(self.cfg.attention_window)
        return int(self.cfg.local_attention_window)

    @property
    def _attention_heads(self) -> int:
        if self.cfg.attention_n_heads is not None:
            return int(self.cfg.attention_n_heads)
        return int(self.cfg.local_attention_heads)

    def _head_enabled(self, name: str) -> bool:
        return bool(self.cfg.heads.get(name, True))

    def _reverse_complement_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        complement_table = cast(torch.Tensor, self._complement_table)
        return torch.flip(complement_table[input_ids], dims=[-1])

    def _embed_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_emb(input_ids)
        if self.local_conv_stem is not None:
            hidden = self.local_conv_stem(hidden)
        return hidden

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_hidden_taps: bool = False,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        rc_ids = self._reverse_complement_ids(input_ids)
        hidden = self._embed_inputs(input_ids)
        hidden_rc = self._embed_inputs(rc_ids)

        hidden_taps: dict[int, torch.Tensor] = {}
        tap_layers = set(int(layer) for layer in self.cfg.hidden_tap_layers)
        attention_every = self._attention_every
        for index, block in enumerate(self.blocks):
            layer_idx = index + 1
            hidden = block(hidden, hidden_rc)
            if (
                self.cfg.use_local_attention
                and attention_every > 0
                and layer_idx % attention_every == 0
                and index < len(self.blocks) - 1
            ):
                attn_index = index // attention_every
                hidden = hidden + self.attn_layers[attn_index](
                    self.attn_norms[attn_index](hidden),
                    attention_mask,
                )
            if layer_idx in tap_layers:
                hidden_taps[layer_idx] = hidden

        hidden = self.dropout(self.final_norm(hidden))
        return hidden, hidden_taps

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = False,
        return_hidden: bool = False,
        return_hidden_taps: bool = False,
        return_counterfactual: bool = True,
        return_counterfactual_repr: bool = True,
    ) -> dict[str, Any]:
        if attention_mask is None:
            attention_mask = input_ids.ne(PAD_ID)

        hidden, hidden_taps = self.encode(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_taps=return_hidden_taps,
        )

        outputs: dict[str, Any] = {}
        if return_hidden:
            outputs["hidden_states"] = hidden
        if return_hidden_taps:
            outputs["hidden_taps"] = hidden_taps

        if return_token_heads:
            if self._head_enabled("mlm"):
                outputs["mlm_logits"] = self.mlm_head(hidden)
            if self._head_enabled("phylo"):
                outputs["phylo100_pred"] = self.phylo100_head(hidden)
                outputs["phylo470_pred"] = self.phylo470_head(hidden)
                outputs["phylo_pred"] = outputs["phylo100_pred"]
                outputs["conservation_bin_logits"] = self.conservation_bin_head(hidden)
            if self._head_enabled("region"):
                outputs["region_logits"] = self.region_head(hidden)
            if self._head_enabled("structure"):
                outputs["structure_logits"] = self.structure_head(hidden)
                outputs["donor_distance_logits"] = self.donor_distance_head(hidden)
                outputs["acceptor_distance_logits"] = self.acceptor_distance_head(hidden)
            if self._head_enabled("aa"):
                outputs["aa_logits"] = self.aa_head(hidden)
            if self._head_enabled("codon"):
                outputs["codon_logits"] = self.codon_head(hidden)
            if self._head_enabled("codon_phylo"):
                outputs["codon_phylo_pred"] = self.codon_phylo_head(hidden)
            if self._head_enabled("transcript_frame"):
                outputs["codon_pos_logits"] = self.codon_pos_head(hidden)
                outputs["exon_phase_logits"] = self.exon_phase_head(hidden)
            if self.encode_head is not None and self._head_enabled("encode"):
                outputs["encode_pred"] = self.encode_head(hidden)

        if (
            return_counterfactual
            and self.counterfactual_snv_head is not None
            and self._head_enabled("counterfactual_snv")
        ):
            tap_list = [hidden_taps[layer] for layer in self.cfg.hidden_tap_layers if layer in hidden_taps]
            tap_list.append(hidden)
            mutation_hidden = self.hidden_tap_fusion(tap_list)
            if isinstance(self.multi_scale_context, MultiScaleContext):
                mutation_hidden = self.multi_scale_context(mutation_hidden, attention_mask=attention_mask)
            else:
                mutation_hidden = self.multi_scale_context(mutation_hidden)
            outputs.update(
                self.counterfactual_snv_head(
                    mutation_hidden,
                    input_ids=input_ids,
                    return_repr=return_counterfactual_repr,
                )
            )

        if return_sequence_embedding and self.sequence_embedding_head is not None:
            outputs["sequence_embedding"] = self.sequence_embedding_head(hidden, attention_mask=attention_mask)

        return outputs


def build_beat_v9_model(cfg: BeatV9Config) -> DNAFoundationBeatV9:
    return DNAFoundationBeatV9(cfg)
