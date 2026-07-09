from __future__ import annotations

from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_AUX_FEATURE_DIMS = {
    "phylo": 2,
    "structure": 3,
}


def crop_center(
    x: torch.Tensor | np.ndarray,
    keep_target_center_fraction: float = 0.375,
    *,
    sequence_axis: int = -2,
) -> Any:
    axis = sequence_axis % x.ndim
    seq_len = x.shape[axis]
    target_offset = int(seq_len * (1.0 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(target_offset, target_offset + target_length)
    return x[tuple(slices)]


def _backbone_hidden_states(
    backbone: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    requires_grad = any(parameter.requires_grad for parameter in backbone.parameters())
    context = nullcontext() if requires_grad else torch.no_grad()
    with context:
        encode = getattr(backbone, "encode", None)
        if callable(encode):
            encoded = encode(input_ids)
            if isinstance(encoded, torch.Tensor):
                return encoded
            if isinstance(encoded, dict) and isinstance(encoded.get("hidden_states"), torch.Tensor):
                return encoded["hidden_states"]
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_token_heads=False)
    return outputs["hidden_states"]


def _backbone_encoded_states(
    backbone: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    requires_grad = any(parameter.requires_grad for parameter in backbone.parameters())
    context = nullcontext() if requires_grad else torch.no_grad()
    with context:
        encode = getattr(backbone, "encode", None)
        if not callable(encode):
            raise RuntimeError("v10 functional heads require backbone.encode().")
        encoded = encode(input_ids, attention_mask=attention_mask)
    if not isinstance(encoded, dict):
        raise RuntimeError("v10 functional heads require encode() to return a dict of encoded states.")
    hidden_states = encoded.get("hidden_states")
    mid_hidden_states = encoded.get("mid_hidden_states")
    if not isinstance(hidden_states, torch.Tensor):
        raise RuntimeError("v10 functional heads require encode()['hidden_states'].")
    if not isinstance(mid_hidden_states, torch.Tensor):
        raise RuntimeError("v10 functional heads require encode()['mid_hidden_states'].")
    return {"hidden_states": hidden_states, "mid_hidden_states": mid_hidden_states}


def _backbone_feature_states(
    backbone: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    feature_source: str,
) -> torch.Tensor:
    if feature_source == "hidden":
        return _backbone_hidden_states(backbone, input_ids=input_ids, attention_mask=attention_mask)
    if feature_source != "decoder":
        raise ValueError(f"Unsupported NTv3 feature_source={feature_source!r}.")

    requires_grad = any(parameter.requires_grad for parameter in backbone.parameters())
    context = nullcontext() if requires_grad else torch.no_grad()
    with context:
        extract_sequence_features = getattr(backbone, "extract_sequence_features", None)
        if callable(extract_sequence_features):
            features = extract_sequence_features(input_ids)
            decoder_states = features.get("decoder_states") if isinstance(features, dict) else None
            if isinstance(decoder_states, torch.Tensor):
                return decoder_states
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_token_heads=False)
    decoder_states = outputs.get("decoder_states") if isinstance(outputs, dict) else None
    if isinstance(decoder_states, torch.Tensor):
        return decoder_states
    raise RuntimeError("feature_source='decoder' requires backbone decoder states.")


def _normalized_aux_feature_groups(aux_features: str) -> tuple[str, ...]:
    normalized = aux_features.lower().replace("_", "-").strip()
    if normalized in {"", "none"}:
        return ()
    if normalized == "phylo-structure":
        return ("phylo", "structure")
    groups = tuple(part for part in normalized.split("+") if part)
    allowed = set(_AUX_FEATURE_DIMS)
    unknown = [group for group in groups if group not in allowed]
    if unknown:
        raise ValueError(
            f"Unsupported functional_head_aux_features={aux_features!r}; "
            "expected 'none', 'phylo', 'structure', or 'phylo-structure'."
        )
    return groups


_V10_CONTEXT_DIM = 256
_V10_BIO_AUX_DIM = 125
_V10_MLM_AUX_DIM = 4
_V10_SEQUENCE_EMBED_DIM = 256
_V10_BIO_AUX_HEAD_NAMES = {
    "phylo100_subst_head",
    "zoo241_subst_head",
    "splice_class_head",
    "splice_distance_head",
    "region_head",
    "gnomad_af_head",
    "gnomad_observed_head",
    "counterfactual_snv_head",
    "regulatory_head",
}
_V10_STACK_HEAD_NAMES = _V10_BIO_AUX_HEAD_NAMES | {
    "mlm_head",
    "sequence_embedding_head",
}
_V10_BIO_GROUP_DIMS = {
    "sequence": _V10_MLM_AUX_DIM,
    "conservation": 8,
    "splice": 6,
    "region": 5,
    "population": 8,
    "counterfactual": 48,
    "regulatory": 50,
}


def functional_aux_required_head_names(aux_features: str) -> set[str]:
    groups = _normalized_aux_feature_groups(aux_features)
    required_names: set[str] = set()
    if "phylo" in groups:
        required_names.update({"phylo100_head", "phylo470_head"})
    if "structure" in groups:
        required_names.add("structure_head")
    return required_names


def functional_required_backbone_module_names(head_type: str, aux_features: str) -> set[str]:
    required_names = functional_aux_required_head_names(aux_features)
    normalized = head_type.lower().replace("_", "-")
    if normalized in {
        "v10-bio-aux-pyramid",
        "v10-assay-gated-bio-pyramid",
        "v10-profile-count-bioaux",
        "v10-profile-count-bioaux-rc-gated-residual",
        "v10-assay-rescue-hybrid",
    }:
        required_names.update(_V10_BIO_AUX_HEAD_NAMES)
    if normalized in {
        "v10-profile-count-bioaux",
        "v10-profile-count-bioaux-rc-gated-residual",
        "v10-assay-rescue-hybrid",
    }:
        required_names.add("sequence_embedding_head")
    if normalized in {"v10-bioprogram-stack", "v10-biocov-residual"}:
        required_names.update(_V10_STACK_HEAD_NAMES)
    return required_names


def _scalar_token_prediction(prediction: Any, *, name: str, hidden_states: torch.Tensor) -> torch.Tensor:
    if not isinstance(prediction, torch.Tensor):
        raise RuntimeError(f"{name} must return a tensor.")
    if prediction.ndim == hidden_states.ndim and prediction.shape[-1] == 1:
        return prediction.squeeze(-1)
    return prediction


def _functional_feature_bundle(
    backbone: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    feature_source: str,
    aux_features: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    groups = _normalized_aux_feature_groups(aux_features)
    if not groups:
        return _backbone_feature_states(
            backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_source=feature_source,
        ), None
    if feature_source != "hidden":
        raise RuntimeError("functional_head_aux_features requires feature_source='hidden'.")

    hidden_states = _backbone_hidden_states(backbone, input_ids=input_ids, attention_mask=attention_mask)

    # Auxiliary predictions are fixed readout features. The backbone still
    # trains through the direct hidden-state path used by the NTv3 head.
    feature_tensors: list[torch.Tensor] = []
    if "phylo" in groups:
        phylo100_head = getattr(backbone, "phylo100_head", None)
        phylo470_head = getattr(backbone, "phylo470_head", None)
        if not callable(phylo100_head) or not callable(phylo470_head):
            raise RuntimeError("functional_head_aux_features='phylo' requires phylo100_head and phylo470_head.")
        with torch.no_grad():
            phylo100 = _scalar_token_prediction(
                cast(Any, phylo100_head)(hidden_states),
                name="phylo100_head",
                hidden_states=hidden_states,
            )
            phylo470 = _scalar_token_prediction(
                cast(Any, phylo470_head)(hidden_states),
                name="phylo470_head",
                hidden_states=hidden_states,
            )
        feature_tensors.append(torch.stack([phylo100, phylo470], dim=-1))
    if "structure" in groups:
        structure_head = getattr(backbone, "structure_head", None)
        if not callable(structure_head):
            raise RuntimeError("functional_head_aux_features='structure' requires structure_head.")
        with torch.no_grad():
            structure_logits = structure_head(hidden_states)
        if not isinstance(structure_logits, torch.Tensor):
            raise RuntimeError("functional_head_aux_features='structure' requires structure_logits.")
        feature_tensors.append(structure_logits)

    return hidden_states, torch.cat(feature_tensors, dim=-1)


class FunctionalAuxFeatureProjector(nn.Module):
    def __init__(
        self,
        *,
        aux_features: str,
        projection_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("auxiliary feature projection_dim must be positive.")
        self.groups = _normalized_aux_feature_groups(aux_features)
        self.output_dim = len(self.groups) * projection_dim
        self.projections = nn.ModuleDict(
            {
                group: nn.Sequential(
                    nn.LayerNorm(_AUX_FEATURE_DIMS[group]),
                    nn.Linear(_AUX_FEATURE_DIMS[group], projection_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for group in self.groups
            }
        )

    def forward(self, aux_features: torch.Tensor | None) -> torch.Tensor | None:
        if not self.groups:
            return None
        if aux_features is None:
            raise RuntimeError("Auxiliary feature projector expected aux_features.")

        chunks: list[torch.Tensor] = []
        offset = 0
        for group in self.groups:
            width = _AUX_FEATURE_DIMS[group]
            chunks.append(self.projections[group](aux_features[..., offset : offset + width]))
            offset += width
        return torch.cat(chunks, dim=-1)


class FunctionalTracksHead(nn.Module):
    def __init__(self, embed_dim: int, num_tracks: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_tracks)

    def forward(self, hidden_states: torch.Tensor, aux_features: torch.Tensor | None = None) -> torch.Tensor:
        del aux_features
        return F.softplus(self.head(self.layer_norm(hidden_states)))


class FunctionalTracksMlpHead(nn.Module):
    """Non-linear functional head for NTv3 transfer experiments."""

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        aux_features: str = "none",
        aux_projection_dim: int = 16,
    ) -> None:
        super().__init__()
        self.aux_projector = FunctionalAuxFeatureProjector(
            aux_features=aux_features,
            projection_dim=aux_projection_dim,
            dropout=dropout,
        )
        input_dim = embed_dim + self.aux_projector.output_dim
        inner_dim = hidden_dim or embed_dim * 2
        self.layer_norm = nn.LayerNorm(input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, num_tracks),
        )

    def forward(self, hidden_states: torch.Tensor, aux_features: torch.Tensor | None = None) -> torch.Tensor:
        aux_projection = self.aux_projector(aux_features)
        if aux_projection is not None:
            hidden_states = torch.cat([hidden_states, aux_projection], dim=-1)
        return F.softplus(self.head(self.layer_norm(hidden_states)))


class FunctionalTracksGlobalContextHead(nn.Module):
    """Aux-aware MLP readout conditioned on a global window embedding.

    RNA and chromatin tracks can depend on broader locus state than a per-base
    readout exposes directly. This head broadcasts a pooled beat-v7 window
    embedding back to every target position, while keeping the proven MLP +
    phylo/structure readout path intact.
    """

    uses_global_context = True

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        aux_features: str = "none",
        aux_projection_dim: int = 16,
    ) -> None:
        super().__init__()
        self.aux_projector = FunctionalAuxFeatureProjector(
            aux_features=aux_features,
            projection_dim=aux_projection_dim,
            dropout=dropout,
        )
        self.global_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        input_dim = embed_dim + embed_dim + self.aux_projector.output_dim
        inner_dim = hidden_dim or embed_dim * 2
        self.layer_norm = nn.LayerNorm(input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, num_tracks),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if global_context is None:
            raise RuntimeError("FunctionalTracksGlobalContextHead requires global_context.")
        if global_context.ndim != 2 or global_context.shape[0] != hidden_states.shape[0]:
            raise RuntimeError("global_context must have shape [batch, embed_dim].")

        global_projection = self.global_projector(global_context).unsqueeze(1).expand(
            -1,
            hidden_states.shape[1],
            -1,
        )
        features = [hidden_states, global_projection]
        aux_projection = self.aux_projector(aux_features)
        if aux_projection is not None:
            features.append(aux_projection)
        x = torch.cat(features, dim=-1)
        return F.softplus(self.head(self.layer_norm(x)))


class FunctionalTracksLocalConvHead(nn.Module):
    """Non-linear functional head with explicit local sequence context."""

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("local-conv functional head requires a positive odd kernel_size.")
        inner_dim = hidden_dim or embed_dim * 2
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(inner_dim)
        self.output_proj = nn.Linear(inner_dim, num_tracks)

    def forward(self, hidden_states: torch.Tensor, aux_features: torch.Tensor | None = None) -> torch.Tensor:
        del aux_features
        x = self.input_proj(self.layer_norm(hidden_states))
        local_context = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(self.activation(local_context))
        return F.softplus(self.output_proj(self.output_norm(x)))


class FunctionalTracksGatedHybridHead(nn.Module):
    """MLP readout with a track-gated local residual branch.

    The base path matches the successful MLP readout. The local branch starts
    with a small per-track gate, so contextual smoothing is learned only where
    it helps instead of being forced onto every NTv3 track.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_features: str = "none",
        aux_projection_dim: int = 16,
        gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("gated-hybrid functional head requires a positive odd kernel_size.")
        self.aux_projector = FunctionalAuxFeatureProjector(
            aux_features=aux_features,
            projection_dim=aux_projection_dim,
            dropout=dropout,
        )
        input_dim = embed_dim + self.aux_projector.output_dim
        inner_dim = hidden_dim or embed_dim * 2
        self.layer_norm = nn.LayerNorm(input_dim)
        self.base_head = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, num_tracks),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim,
        )
        self.context_activation = nn.GELU()
        self.context_dropout = nn.Dropout(dropout)
        self.context_norm = nn.LayerNorm(inner_dim)
        self.context_out = nn.Linear(inner_dim, num_tracks)
        self.track_gate_logits = nn.Parameter(torch.full((num_tracks,), gate_init))

    def forward(self, hidden_states: torch.Tensor, aux_features: torch.Tensor | None = None) -> torch.Tensor:
        aux_projection = self.aux_projector(aux_features)
        if aux_projection is not None:
            hidden_states = torch.cat([hidden_states, aux_projection], dim=-1)

        x = self.layer_norm(hidden_states)
        base_logits = self.base_head(x)

        context = self.context_proj(x)
        local_context = self.context_conv(context.transpose(1, 2)).transpose(1, 2)
        context = context + self.context_dropout(self.context_activation(local_context))
        residual_logits = self.context_out(self.context_norm(context))

        gate = torch.sigmoid(self.track_gate_logits).view(1, 1, -1)
        return F.softplus(base_logits + gate * residual_logits)


class FunctionalTracksMultiScaleDilatedHead(nn.Module):
    """Aux-aware multi-scale dilated readout without learned branch gates."""

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_features: str = "none",
        aux_projection_dim: int = 16,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("multi-scale-dilated functional head requires a positive odd kernel_size.")
        self.aux_projector = FunctionalAuxFeatureProjector(
            aux_features=aux_features,
            projection_dim=aux_projection_dim,
            dropout=dropout,
        )
        input_dim = embed_dim + self.aux_projector.output_dim
        branch_dim = hidden_dim or max(32, embed_dim // 4)
        output_dim = branch_dim * 4
        self.layer_norm = nn.LayerNorm(input_dim)
        self.point_branch = nn.Sequential(
            nn.Conv1d(input_dim, branch_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dilated_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, branch_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(
                        branch_dim,
                        branch_dim,
                        kernel_size=kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                        groups=branch_dim,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dilation in (1, 8, 64)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_tracks),
        )

    def forward(self, hidden_states: torch.Tensor, aux_features: torch.Tensor | None = None) -> torch.Tensor:
        aux_projection = self.aux_projector(aux_features)
        if aux_projection is not None:
            hidden_states = torch.cat([hidden_states, aux_projection], dim=-1)
        x = self.layer_norm(hidden_states).transpose(1, 2)
        branches = [self.point_branch(x), *(branch(x) for branch in self.dilated_branches)]
        features = torch.cat(branches, dim=1).transpose(1, 2)
        return F.softplus(self.output_head(features))


class FunctionalTracksContextPyramidHead(nn.Module):
    """Global-conditioned multi-scale readout for dense NTv3 functional tracks.

    The head keeps NTv3-compliant inputs and outputs: it reads only Lumina's
    dense hidden state, optional frozen Lumina aux readouts, and the model's own
    pooled window embedding. It does not alter benchmark targets, splits, loss,
    or 1bp prediction resolution.
    """

    uses_global_context = True

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_features: str = "none",
        aux_projection_dim: int = 16,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("context-pyramid functional head requires a positive odd kernel_size.")
        self.aux_projector = FunctionalAuxFeatureProjector(
            aux_features=aux_features,
            projection_dim=aux_projection_dim,
            dropout=dropout,
        )
        input_dim = embed_dim + self.aux_projector.output_dim
        branch_dim = hidden_dim or max(32, embed_dim // 4)
        pyramid_dim = branch_dim * 4
        self.input_norm = nn.LayerNorm(input_dim)
        self.point_branch = nn.Sequential(
            nn.Conv1d(input_dim, branch_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dilated_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, branch_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(
                        branch_dim,
                        branch_dim,
                        kernel_size=kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                        groups=branch_dim,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dilation in (1, 64, 512)
            ]
        )
        self.global_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        head_input_dim = pyramid_dim + embed_dim
        head_hidden_dim = max(embed_dim * 2, pyramid_dim)
        self.output_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_tracks),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if global_context is None:
            raise RuntimeError("FunctionalTracksContextPyramidHead requires global_context.")
        if global_context.ndim != 2 or global_context.shape[0] != hidden_states.shape[0]:
            raise RuntimeError("global_context must have shape [batch, embed_dim].")

        aux_projection = self.aux_projector(aux_features)
        if aux_projection is not None:
            hidden_states = torch.cat([hidden_states, aux_projection], dim=-1)

        x = self.input_norm(hidden_states).transpose(1, 2)
        pyramid = torch.cat(
            [self.point_branch(x), *(branch(x) for branch in self.dilated_branches)],
            dim=1,
        ).transpose(1, 2)
        global_projection = self.global_projector(global_context).unsqueeze(1).expand(
            -1,
            pyramid.shape[1],
            -1,
        )
        features = torch.cat([pyramid, global_projection], dim=-1)
        return F.softplus(self.output_head(features))


def _resize_sequence_features(features: torch.Tensor, target_length: int) -> torch.Tensor:
    if features.shape[1] == target_length:
        return features
    return F.interpolate(
        features.transpose(1, 2),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def _split_v10_hidden(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.shape[-1] <= _V10_CONTEXT_DIM:
        raise RuntimeError(
            "v10 functional heads expect hidden_states with concatenated context and h_pure channels."
        )
    return hidden_states[..., :_V10_CONTEXT_DIM], hidden_states[..., _V10_CONTEXT_DIM:]


def _v10_bio_aux_features(
    backbone: nn.Module,
    *,
    hidden_states: torch.Tensor,
    mid_hidden_states: torch.Tensor,
) -> torch.Tensor:
    def _call_token_head(name: str, inputs: torch.Tensor) -> torch.Tensor:
        module = getattr(backbone, name, None)
        if not callable(module):
            raise RuntimeError(f"v10 biological auxiliary features require backbone.{name}.")
        output = cast(Any, module)(inputs)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"v10 biological auxiliary head {name} must return a tensor.")
        if output.ndim == hidden_states.ndim - 1:
            output = output.unsqueeze(-1)
        if output.ndim != hidden_states.ndim:
            shape = tuple(output.shape)
            raise RuntimeError(f"v10 biological auxiliary head {name} returned unsupported shape {shape}.")
        return output

    with torch.no_grad():
        feature_tensors = [
            _call_token_head("phylo100_subst_head", hidden_states),
            _call_token_head("zoo241_subst_head", hidden_states),
            _call_token_head("splice_class_head", hidden_states),
            _call_token_head("splice_distance_head", hidden_states),
            _call_token_head("region_head", hidden_states),
            _call_token_head("gnomad_af_head", hidden_states),
            _call_token_head("gnomad_observed_head", hidden_states),
            _call_token_head("counterfactual_snv_head", hidden_states),
        ]
        regulatory_head = getattr(backbone, "regulatory_head", None)
        if not callable(regulatory_head):
            raise RuntimeError("v10 biological auxiliary features require backbone.regulatory_head.")
        regulatory = cast(Any, regulatory_head)(mid_hidden_states)
        if not isinstance(regulatory, torch.Tensor):
            raise RuntimeError("v10 biological auxiliary head regulatory_head must return a tensor.")
        regulatory = _resize_sequence_features(regulatory, hidden_states.shape[1])
        feature_tensors.append(regulatory)
    aux = torch.cat(feature_tensors, dim=-1)
    if aux.shape[-1] != _V10_BIO_AUX_DIM:
        raise RuntimeError(f"Expected {_V10_BIO_AUX_DIM} v10 aux dims, got {aux.shape[-1]}.")
    return aux


def _v10_bio_aux_feature_groups(
    backbone: nn.Module,
    *,
    hidden_states: torch.Tensor,
    mid_hidden_states: torch.Tensor,
    detach: bool,
) -> dict[str, torch.Tensor]:
    def _call_token_head(name: str, inputs: torch.Tensor) -> torch.Tensor:
        module = getattr(backbone, name, None)
        if not callable(module):
            raise RuntimeError(f"v10 BioProgram Stack requires backbone.{name}.")
        output = cast(Any, module)(inputs)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"v10 BioProgram Stack head {name} must return a tensor.")
        if output.ndim == hidden_states.ndim - 1:
            output = output.unsqueeze(-1)
        if output.ndim != hidden_states.ndim:
            shape = tuple(output.shape)
            raise RuntimeError(f"v10 BioProgram Stack head {name} returned unsupported shape {shape}.")
        return output

    context = torch.no_grad() if detach else nullcontext()
    with context:
        mlm = _call_token_head("mlm_head", hidden_states)
        phylo100 = _call_token_head("phylo100_subst_head", hidden_states)
        zoo241 = _call_token_head("zoo241_subst_head", hidden_states)
        splice_class = _call_token_head("splice_class_head", hidden_states)
        splice_distance = _call_token_head("splice_distance_head", hidden_states)
        region = _call_token_head("region_head", hidden_states)
        gnomad_af = _call_token_head("gnomad_af_head", hidden_states)
        gnomad_observed = _call_token_head("gnomad_observed_head", hidden_states)
        counterfactual = _call_token_head("counterfactual_snv_head", hidden_states)

        regulatory_head = getattr(backbone, "regulatory_head", None)
        if not callable(regulatory_head):
            raise RuntimeError("v10 BioProgram Stack requires backbone.regulatory_head.")
        regulatory = cast(Any, regulatory_head)(mid_hidden_states)
        if not isinstance(regulatory, torch.Tensor):
            raise RuntimeError("v10 BioProgram Stack regulatory_head must return a tensor.")
        regulatory = _resize_sequence_features(regulatory, hidden_states.shape[1])

    groups = {
        "sequence": mlm,
        "conservation": torch.cat([phylo100, zoo241], dim=-1),
        "splice": torch.cat([splice_class, splice_distance], dim=-1),
        "region": region,
        "population": torch.cat([gnomad_af, gnomad_observed], dim=-1),
        "counterfactual": counterfactual,
        "regulatory": regulatory,
    }
    for group_name, expected_dim in _V10_BIO_GROUP_DIMS.items():
        actual_dim = groups[group_name].shape[-1]
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"Expected {expected_dim} dims for v10 BioProgram Stack group {group_name}, got {actual_dim}."
            )
    return groups


def _v10_sequence_embedding(
    backbone: nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    sequence_embedding_head = getattr(backbone, "sequence_embedding_head", None)
    if not callable(sequence_embedding_head):
        raise RuntimeError("v10 BioProgram Stack requires backbone.sequence_embedding_head.")
    output = cast(Any, sequence_embedding_head)(hidden_states, attention_mask=attention_mask)
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise RuntimeError("v10 sequence_embedding_head must return a [batch, dim] tensor.")
    return output


class FunctionalTracksV10RepresentationPyramidHead(nn.Module):
    """Context-pyramid readout that exposes beat-v10's local, full-resolution, and mid states."""

    uses_global_context = True
    uses_v10_encoded_features = True
    needs_v10_bio_aux = False

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_projection_dim: int = 16,
    ) -> None:
        del aux_projection_dim
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("v10 representation pyramid head requires a positive odd kernel_size.")
        pure_dim = embed_dim - _V10_CONTEXT_DIM
        if pure_dim <= 0:
            raise ValueError("v10 representation pyramid head requires embed_dim > 256.")
        pure_projection_dim = max(32, pure_dim * 2)
        input_dim = _V10_CONTEXT_DIM + pure_projection_dim + _V10_CONTEXT_DIM
        branch_dim = hidden_dim or max(64, embed_dim // 3)
        pyramid_dim = branch_dim * 4

        self.pure_projector = nn.Sequential(
            nn.LayerNorm(pure_dim),
            nn.Linear(pure_dim, pure_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mid_projector = nn.Sequential(
            nn.LayerNorm(_V10_CONTEXT_DIM),
            nn.Linear(_V10_CONTEXT_DIM, _V10_CONTEXT_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_norm = nn.LayerNorm(input_dim)
        self.point_branch = nn.Sequential(
            nn.Conv1d(input_dim, branch_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dilated_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, branch_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(
                        branch_dim,
                        branch_dim,
                        kernel_size=kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                        groups=branch_dim,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dilation in (1, 64, 512)
            ]
        )
        self.global_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        head_input_dim = pyramid_dim + embed_dim
        head_hidden_dim = max(embed_dim * 2, pyramid_dim)
        self.output_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_tracks),
        )

    def _context_logits(
        self,
        hidden_states: torch.Tensor,
        global_context: torch.Tensor,
        mid_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        h_context, h_pure = _split_v10_hidden(hidden_states)
        mid_up = _resize_sequence_features(mid_hidden_states, hidden_states.shape[1])
        x = torch.cat([h_context, self.pure_projector(h_pure), self.mid_projector(mid_up)], dim=-1)
        x = self.input_norm(x).transpose(1, 2)
        pyramid = torch.cat(
            [self.point_branch(x), *(branch(x) for branch in self.dilated_branches)],
            dim=1,
        ).transpose(1, 2)
        global_projection = self.global_projector(global_context).unsqueeze(1).expand(
            -1,
            pyramid.shape[1],
            -1,
        )
        return self.output_head(torch.cat([pyramid, global_projection], dim=-1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
        mid_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del aux_features
        if global_context is None or mid_hidden_states is None:
            raise RuntimeError("v10 representation pyramid head requires global_context and mid_hidden_states.")
        return F.softplus(self._context_logits(hidden_states, global_context, mid_hidden_states))


class FunctionalTracksV10BioAuxPyramidHead(FunctionalTracksV10RepresentationPyramidHead):
    """Beat-v10 pyramid readout with frozen biological readout features from the pretraining heads."""

    needs_v10_bio_aux = True

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_projection_dim: int = 128,
    ) -> None:
        super().__init__(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        self.bio_projector = nn.Sequential(
            nn.LayerNorm(_V10_BIO_AUX_DIM),
            nn.Linear(_V10_BIO_AUX_DIM, aux_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        old_final = self.output_head[-1]
        if not isinstance(old_final, nn.Linear):
            raise RuntimeError("Unexpected v10 representation output head shape.")
        old_input_dim = cast(nn.LayerNorm, self.output_head[0]).normalized_shape[0]
        head_hidden_dim = cast(nn.Linear, self.output_head[1]).out_features
        self.output_head = nn.Sequential(
            nn.LayerNorm(int(old_input_dim) + aux_projection_dim),
            nn.Linear(int(old_input_dim) + aux_projection_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, old_final.out_features),
        )

    def _context_logits(
        self,
        hidden_states: torch.Tensor,
        global_context: torch.Tensor,
        mid_hidden_states: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if aux_features is None:
            raise RuntimeError("v10 biological pyramid head requires aux_features.")
        h_context, h_pure = _split_v10_hidden(hidden_states)
        mid_up = _resize_sequence_features(mid_hidden_states, hidden_states.shape[1])
        x = torch.cat([h_context, self.pure_projector(h_pure), self.mid_projector(mid_up)], dim=-1)
        x = self.input_norm(x).transpose(1, 2)
        pyramid = torch.cat(
            [self.point_branch(x), *(branch(x) for branch in self.dilated_branches)],
            dim=1,
        ).transpose(1, 2)
        global_projection = self.global_projector(global_context).unsqueeze(1).expand(
            -1,
            pyramid.shape[1],
            -1,
        )
        bio_projection = self.bio_projector(aux_features)
        return self.output_head(torch.cat([pyramid, global_projection, bio_projection], dim=-1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
        mid_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if global_context is None or mid_hidden_states is None:
            raise RuntimeError("v10 biological pyramid head requires global_context and mid_hidden_states.")
        return F.softplus(self._context_logits(hidden_states, global_context, mid_hidden_states, aux_features))


class FunctionalTracksV10AssayGatedBioPyramidHead(FunctionalTracksV10RepresentationPyramidHead):
    """Per-track mixture of local, regional, and biological branches for beat-v10."""

    needs_v10_bio_aux = True

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_projection_dim: int = 128,
    ) -> None:
        super().__init__(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        pure_dim = embed_dim - _V10_CONTEXT_DIM
        local_dim = hidden_dim or max(64, embed_dim // 3)
        self.bio_projector = nn.Sequential(
            nn.LayerNorm(_V10_BIO_AUX_DIM),
            nn.Linear(_V10_BIO_AUX_DIM, aux_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_norm = nn.LayerNorm(pure_dim)
        self.local_projector = nn.Sequential(
            nn.Linear(pure_dim, local_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_conv = nn.Conv1d(
            local_dim,
            local_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=local_dim,
        )
        self.local_out_norm = nn.LayerNorm(local_dim)
        self.local_out = nn.Linear(local_dim, num_tracks)
        self.bio_head = nn.Sequential(
            nn.LayerNorm(embed_dim + aux_projection_dim),
            nn.Linear(embed_dim + aux_projection_dim, max(embed_dim * 2, aux_projection_dim * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(embed_dim * 2, aux_projection_dim * 2), num_tracks),
        )
        self.track_gate_logits = nn.Parameter(torch.zeros(num_tracks, 3))

    def _local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _h_context, h_pure = _split_v10_hidden(hidden_states)
        local = self.local_projector(self.local_norm(h_pure))
        local_context = self.local_conv(local.transpose(1, 2)).transpose(1, 2)
        local = local + F.gelu(local_context)
        return self.local_out(self.local_out_norm(local))

    def _bio_logits(self, hidden_states: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
        bio_projection = self.bio_projector(aux_features)
        return self.bio_head(torch.cat([hidden_states, bio_projection], dim=-1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
        mid_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if aux_features is None or global_context is None or mid_hidden_states is None:
            raise RuntimeError(
                "v10 assay-gated bio pyramid head requires aux_features, global_context, and mid states."
            )
        local_logits = self._local_logits(hidden_states)
        context_logits = self._context_logits(hidden_states, global_context, mid_hidden_states)
        bio_logits = self._bio_logits(hidden_states, aux_features)
        stacked = torch.stack([local_logits, context_logits, bio_logits], dim=-1)
        gates = torch.softmax(self.track_gate_logits, dim=-1).view(1, 1, -1, 3)
        return F.softplus((stacked * gates).sum(dim=-1))


class FunctionalTracksV10BioProgramStackHead(nn.Module):
    """Stacked beat-v10 readout over local, regional, global, and biological program features.

    This is intentionally not a pyramid head. It treats NTv3 tracks as mixtures
    of shared latent biological programs, while allowing each track to gate the
    frozen beat-v10 biological readout families differently.
    """

    uses_global_context = True
    uses_v10_encoded_features = True
    uses_v10_sequence_embedding = True
    needs_v10_all_bio_aux = True
    uses_v10_differentiable_aux = True

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_projection_dim: int = 256,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("v10 BioProgram Stack requires a positive odd kernel_size.")
        pure_dim = embed_dim - _V10_CONTEXT_DIM
        if pure_dim <= 0:
            raise ValueError("v10 BioProgram Stack requires embed_dim > 256.")
        program_dim = hidden_dim or 32
        if program_dim <= 0:
            raise ValueError("v10 BioProgram Stack hidden_dim/program count must be positive.")

        state_dim = max(64, embed_dim // 3)
        group_dim = max(16, aux_projection_dim // 8)
        self.group_names = tuple(_V10_BIO_GROUP_DIMS)

        self.context_projector = nn.Sequential(
            nn.LayerNorm(_V10_CONTEXT_DIM),
            nn.Linear(_V10_CONTEXT_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pure_projector = nn.Sequential(
            nn.LayerNorm(pure_dim),
            nn.Linear(pure_dim, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pure_local_conv = nn.Conv1d(
            state_dim,
            state_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=state_dim,
        )
        self.local_dropout = nn.Dropout(dropout)
        self.mid_projector = nn.Sequential(
            nn.LayerNorm(_V10_CONTEXT_DIM),
            nn.Linear(_V10_CONTEXT_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_projector = nn.Sequential(
            nn.LayerNorm(_V10_SEQUENCE_EMBED_DIM),
            nn.Linear(_V10_SEQUENCE_EMBED_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.group_projectors = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.LayerNorm(group_dim_in),
                    nn.Linear(group_dim_in, group_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for group_name, group_dim_in in _V10_BIO_GROUP_DIMS.items()
            }
        )

        bio_dim = group_dim * len(self.group_names)
        feature_dim = state_dim * 4 + bio_dim
        program_hidden_dim = max(embed_dim * 2, feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.program_head = nn.Sequential(
            nn.Linear(feature_dim, program_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(program_hidden_dim, program_dim),
        )
        self.program_to_tracks = nn.Linear(program_dim, num_tracks)

        self.local_out_norm = nn.LayerNorm(state_dim)
        self.local_out = nn.Linear(state_dim, num_tracks, bias=False)
        self.track_group_logits = nn.Parameter(torch.zeros(num_tracks, len(self.group_names)))
        self.bio_track_out = nn.Linear(group_dim, 1, bias=False)
        self.local_residual_logit = nn.Parameter(torch.tensor(-3.0))
        self.bio_residual_logit = nn.Parameter(torch.tensor(-2.0))

    def _encode_bio_groups(self, aux_features: object) -> torch.Tensor:
        if not isinstance(aux_features, dict):
            raise RuntimeError("v10 BioProgram Stack requires grouped v10 auxiliary features.")
        encoded_groups: list[torch.Tensor] = []
        for group_name in self.group_names:
            group_features = aux_features.get(group_name)
            if not isinstance(group_features, torch.Tensor):
                raise RuntimeError(f"v10 BioProgram Stack missing auxiliary group {group_name!r}.")
            encoded_groups.append(self.group_projectors[group_name](group_features))
        return torch.stack(encoded_groups, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | dict[str, torch.Tensor] | None = None,
        global_context: torch.Tensor | None = None,
        mid_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if aux_features is None or global_context is None or mid_hidden_states is None:
            raise RuntimeError("v10 BioProgram Stack requires aux_features, global_context, and mid states.")

        h_context, h_pure = _split_v10_hidden(hidden_states)
        mid_up = _resize_sequence_features(mid_hidden_states, hidden_states.shape[1])

        context_projection = self.context_projector(h_context)
        pure_projection = self.pure_projector(h_pure)
        local_context = self.pure_local_conv(pure_projection.transpose(1, 2)).transpose(1, 2)
        pure_local = pure_projection + self.local_dropout(F.gelu(local_context))
        mid_projection = self.mid_projector(mid_up)
        global_projection = self.global_projector(global_context).unsqueeze(1).expand(
            -1,
            hidden_states.shape[1],
            -1,
        )

        bio_groups = self._encode_bio_groups(aux_features)
        bio_concat = bio_groups.flatten(start_dim=2)
        features = torch.cat(
            [context_projection, pure_local, mid_projection, global_projection, bio_concat],
            dim=-1,
        )
        programs = self.program_head(self.feature_norm(features))
        program_logits = self.program_to_tracks(programs)

        local_logits = self.local_out(self.local_out_norm(pure_local))
        group_gates = torch.softmax(self.track_group_logits, dim=-1)
        track_bio_features = torch.einsum("blgd,tg->bltd", bio_groups, group_gates)
        bio_logits = self.bio_track_out(track_bio_features).squeeze(-1)

        logits = (
            program_logits
            + torch.sigmoid(self.local_residual_logit) * local_logits
            + torch.sigmoid(self.bio_residual_logit) * bio_logits
        )
        return F.softplus(logits)


class FunctionalTracksV10ProfileCountBioAuxHead(nn.Module):
    """Profile/count readout using beat-v10 sequence and biological auxiliary features.

    The profile branch learns position-specific shape while the count branch
    learns a per-track window-level amplitude. This mirrors the structure of
    functional genomics assays where local signal shape and total signal mass
    can carry different biological information.
    """

    uses_global_context = True
    uses_v10_encoded_features = True
    uses_v10_sequence_embedding = True
    needs_v10_bio_aux = True

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_projection_dim: int = 128,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("v10 profile/count head requires a positive odd kernel_size.")
        pure_dim = embed_dim - _V10_CONTEXT_DIM
        if pure_dim <= 0:
            raise ValueError("v10 profile/count head requires embed_dim > 256.")

        state_dim = hidden_dim or max(64, embed_dim // 3)
        self.state_dim = state_dim
        self.aux_projection_dim = aux_projection_dim
        profile_hidden_dim = max(embed_dim * 2, state_dim * 5 + aux_projection_dim)
        count_hidden_dim = max(embed_dim, state_dim * 3 + aux_projection_dim)

        self.context_projector = nn.Sequential(
            nn.LayerNorm(_V10_CONTEXT_DIM),
            nn.Linear(_V10_CONTEXT_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pure_projector = nn.Sequential(
            nn.LayerNorm(pure_dim),
            nn.Linear(pure_dim, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pure_local_conv = nn.Conv1d(
            state_dim,
            state_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=state_dim,
        )
        self.local_dropout = nn.Dropout(dropout)
        self.mid_projector = nn.Sequential(
            nn.LayerNorm(_V10_CONTEXT_DIM),
            nn.Linear(_V10_CONTEXT_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_projector = nn.Sequential(
            nn.LayerNorm(_V10_SEQUENCE_EMBED_DIM),
            nn.Linear(_V10_SEQUENCE_EMBED_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bio_projector = nn.Sequential(
            nn.LayerNorm(_V10_BIO_AUX_DIM),
            nn.Linear(_V10_BIO_AUX_DIM, aux_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        profile_input_dim = state_dim * 4 + aux_projection_dim
        self.profile_norm = nn.LayerNorm(profile_input_dim)
        self.profile_mlp = nn.Sequential(
            nn.Linear(profile_input_dim, profile_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.profile_out = nn.Linear(profile_hidden_dim, num_tracks)

        count_input_dim = state_dim * 3 + aux_projection_dim
        self.count_norm = nn.LayerNorm(count_input_dim)
        self.count_mlp = nn.Sequential(
            nn.Linear(count_input_dim, count_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.count_out = nn.Linear(count_hidden_dim, num_tracks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | dict[str, torch.Tensor] | None = None,
        global_context: torch.Tensor | None = None,
        mid_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(aux_features, torch.Tensor) or global_context is None or mid_hidden_states is None:
            raise RuntimeError("v10 profile/count head requires flat aux_features, global_context, and mid states.")

        h_context, h_pure = _split_v10_hidden(hidden_states)
        mid_up = _resize_sequence_features(mid_hidden_states, hidden_states.shape[1])

        context_projection = self.context_projector(h_context)
        pure_projection = self.pure_projector(h_pure)
        local_context = self.pure_local_conv(pure_projection.transpose(1, 2)).transpose(1, 2)
        pure_local = pure_projection + self.local_dropout(F.gelu(local_context))
        mid_projection = self.mid_projector(mid_up)
        global_vector = self.global_projector(global_context)
        global_projection = global_vector.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        bio_projection = self.bio_projector(aux_features)

        profile_features = torch.cat(
            [context_projection, pure_local, mid_projection, global_projection, bio_projection],
            dim=-1,
        )
        raw_profile = F.softplus(self.profile_out(self.profile_mlp(self.profile_norm(profile_features))))
        profile = raw_profile / raw_profile.mean(dim=1, keepdim=True).clamp_min(torch.finfo(raw_profile.dtype).eps)

        count_features = torch.cat(
            [
                global_vector,
                pure_local.mean(dim=1),
                mid_projection.mean(dim=1),
                bio_projection.mean(dim=1),
            ],
            dim=-1,
        )
        count = F.softplus(self.count_out(self.count_mlp(self.count_norm(count_features)))).unsqueeze(1)
        return count * profile


class FunctionalTracksV10ProfileCountBioAuxGatedResidualHead(FunctionalTracksV10ProfileCountBioAuxHead):
    """Profile/count beat-v10 readout with a small gated contextual residual.

    This preserves the best observed global mechanism while letting individual
    tracks learn a limited correction where profile/count underfit chromatin or
    eCLIP-style behavior. The residual gate starts nearly closed to keep the
    initial model close to the validated profile/count hypothesis.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_projection_dim: int = 128,
    ) -> None:
        super().__init__(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
        residual_input_dim = self.state_dim * 4 + self.aux_projection_dim
        residual_hidden_dim = max(embed_dim, residual_input_dim)
        self.residual_norm = nn.LayerNorm(residual_input_dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(residual_input_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_out = nn.Linear(residual_hidden_dim, num_tracks)
        self.track_residual_gate_logits = nn.Parameter(torch.full((num_tracks,), -3.5))
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | dict[str, torch.Tensor] | None = None,
        global_context: torch.Tensor | None = None,
        mid_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(aux_features, torch.Tensor) or global_context is None or mid_hidden_states is None:
            raise RuntimeError(
                "v10 profile/count gated residual head requires flat aux_features, global_context, and mid states."
            )

        h_context, h_pure = _split_v10_hidden(hidden_states)
        mid_up = _resize_sequence_features(mid_hidden_states, hidden_states.shape[1])

        context_projection = self.context_projector(h_context)
        pure_projection = self.pure_projector(h_pure)
        local_context = self.pure_local_conv(pure_projection.transpose(1, 2)).transpose(1, 2)
        pure_local = pure_projection + self.local_dropout(F.gelu(local_context))
        mid_projection = self.mid_projector(mid_up)
        global_vector = self.global_projector(global_context)
        global_projection = global_vector.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        bio_projection = self.bio_projector(aux_features)

        profile_features = torch.cat(
            [context_projection, pure_local, mid_projection, global_projection, bio_projection],
            dim=-1,
        )
        raw_profile = F.softplus(self.profile_out(self.profile_mlp(self.profile_norm(profile_features))))
        profile = raw_profile / raw_profile.mean(dim=1, keepdim=True).clamp_min(torch.finfo(raw_profile.dtype).eps)

        count_features = torch.cat(
            [
                global_vector,
                pure_local.mean(dim=1),
                mid_projection.mean(dim=1),
                bio_projection.mean(dim=1),
            ],
            dim=-1,
        )
        count = F.softplus(self.count_out(self.count_mlp(self.count_norm(count_features)))).unsqueeze(1)
        profile_count_prediction = count * profile

        residual_logits = self.residual_out(self.residual_mlp(self.residual_norm(profile_features)))
        residual_gate = torch.sigmoid(self.track_residual_gate_logits).view(1, 1, -1)
        logits = _inverse_softplus(profile_count_prediction) + residual_gate * residual_logits
        return F.softplus(logits)


class FunctionalTracksV10AssayRescueHybridHead(FunctionalTracksV10ProfileCountBioAuxGatedResidualHead):
    """Profile/count SOTA head with a small per-track assay rescue branch.

    The base path is the current beat-v10 profile/count BioAux residual readout.
    The rescue path is deliberately small and starts with a weak logit-space
    effect. It gives chromatin-like tracks a local/BioAux branch to learn from
    without forcing all assays away from the validated profile/count behavior.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_projection_dim: int = 128,
    ) -> None:
        super().__init__(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
        local_dim = self.state_dim
        regional_dim = self.state_dim * 3 + self.aux_projection_dim
        bio_dim = self.state_dim + self.aux_projection_dim
        rescue_hidden_dim = max(embed_dim, regional_dim)

        self.assay_local_norm = nn.LayerNorm(local_dim)
        self.assay_local_conv = nn.Conv1d(
            local_dim,
            local_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=local_dim,
        )
        self.assay_local_out_norm = nn.LayerNorm(local_dim)
        self.assay_local_out = nn.Linear(local_dim, num_tracks)

        self.assay_regional_norm = nn.LayerNorm(regional_dim)
        self.assay_regional_mlp = nn.Sequential(
            nn.Linear(regional_dim, rescue_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.assay_regional_out = nn.Linear(rescue_hidden_dim, num_tracks)

        self.assay_bio_norm = nn.LayerNorm(bio_dim)
        self.assay_bio_mlp = nn.Sequential(
            nn.Linear(bio_dim, rescue_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.assay_bio_out = nn.Linear(rescue_hidden_dim, num_tracks)

        self.track_assay_rescue_gate_logits = nn.Parameter(torch.full((num_tracks,), -2.2))
        self.track_assay_rescue_branch_logits = nn.Parameter(torch.zeros(num_tracks, 3))
        with torch.no_grad():
            self.track_assay_rescue_branch_logits[:, 0] = 1.25
            self.track_assay_rescue_branch_logits[:, 1] = -0.25
            self.track_assay_rescue_branch_logits[:, 2] = -0.25
        for layer in (self.assay_local_out, self.assay_regional_out, self.assay_bio_out):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | dict[str, torch.Tensor] | None = None,
        global_context: torch.Tensor | None = None,
        mid_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(aux_features, torch.Tensor) or global_context is None or mid_hidden_states is None:
            raise RuntimeError(
                "v10 assay-rescue hybrid head requires flat aux_features, global_context, and mid states."
            )

        h_context, h_pure = _split_v10_hidden(hidden_states)
        mid_up = _resize_sequence_features(mid_hidden_states, hidden_states.shape[1])

        context_projection = self.context_projector(h_context)
        pure_projection = self.pure_projector(h_pure)
        local_context = self.pure_local_conv(pure_projection.transpose(1, 2)).transpose(1, 2)
        pure_local = pure_projection + self.local_dropout(F.gelu(local_context))
        mid_projection = self.mid_projector(mid_up)
        global_vector = self.global_projector(global_context)
        global_projection = global_vector.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        bio_projection = self.bio_projector(aux_features)

        profile_features = torch.cat(
            [context_projection, pure_local, mid_projection, global_projection, bio_projection],
            dim=-1,
        )
        raw_profile = F.softplus(self.profile_out(self.profile_mlp(self.profile_norm(profile_features))))
        profile = raw_profile / raw_profile.mean(dim=1, keepdim=True).clamp_min(torch.finfo(raw_profile.dtype).eps)

        count_features = torch.cat(
            [
                global_vector,
                pure_local.mean(dim=1),
                mid_projection.mean(dim=1),
                bio_projection.mean(dim=1),
            ],
            dim=-1,
        )
        count = F.softplus(self.count_out(self.count_mlp(self.count_norm(count_features)))).unsqueeze(1)
        profile_count_prediction = count * profile

        residual_logits = self.residual_out(self.residual_mlp(self.residual_norm(profile_features)))
        residual_gate = torch.sigmoid(self.track_residual_gate_logits).view(1, 1, -1)
        base_logits = _inverse_softplus(profile_count_prediction) + residual_gate * residual_logits

        local_rescue = self.assay_local_norm(pure_local)
        local_rescue = local_rescue + F.gelu(
            self.assay_local_conv(local_rescue.transpose(1, 2)).transpose(1, 2)
        )
        local_logits = self.assay_local_out(self.assay_local_out_norm(local_rescue))

        regional_features = torch.cat([context_projection, pure_local, mid_projection, bio_projection], dim=-1)
        regional_logits = self.assay_regional_out(
            self.assay_regional_mlp(self.assay_regional_norm(regional_features))
        )

        bio_features = torch.cat([pure_local, bio_projection], dim=-1)
        bio_logits = self.assay_bio_out(self.assay_bio_mlp(self.assay_bio_norm(bio_features)))

        rescue_stack = torch.stack([local_logits, regional_logits, bio_logits], dim=-1)
        branch_weights = torch.softmax(self.track_assay_rescue_branch_logits, dim=-1).view(1, 1, -1, 3)
        rescue_logits = (rescue_stack * branch_weights).sum(dim=-1)
        rescue_gate = torch.sigmoid(self.track_assay_rescue_gate_logits).view(1, 1, -1)

        return F.softplus(base_logits + rescue_gate * rescue_logits)


class FunctionalTracksV10BioCovResidualHead(nn.Module):
    """Low-rank biological covariance residual over beat-v10 grouped auxiliary features.

    The direct branch preserves a conventional per-track readout. The residual
    branches test whether NTv3 tracks share covariance structure that is already
    represented by beat-v10 biological programs, without forcing every track
    through a large stacked program bottleneck.
    """

    uses_global_context = True
    uses_v10_encoded_features = True
    uses_v10_sequence_embedding = True
    needs_v10_all_bio_aux = True
    uses_v10_differentiable_aux = False

    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        kernel_size: int = 15,
        aux_projection_dim: int = 256,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("v10 BioCov residual head requires a positive odd kernel_size.")
        pure_dim = embed_dim - _V10_CONTEXT_DIM
        if pure_dim <= 0:
            raise ValueError("v10 BioCov residual head requires embed_dim > 256.")
        rank_dim = hidden_dim or 16
        if rank_dim <= 0:
            raise ValueError("v10 BioCov residual hidden_dim/rank must be positive.")

        state_dim = max(64, embed_dim // 3)
        group_dim = max(16, aux_projection_dim // 8)
        self.group_names = tuple(_V10_BIO_GROUP_DIMS)

        self.context_projector = nn.Sequential(
            nn.LayerNorm(_V10_CONTEXT_DIM),
            nn.Linear(_V10_CONTEXT_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pure_projector = nn.Sequential(
            nn.LayerNorm(pure_dim),
            nn.Linear(pure_dim, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pure_local_conv = nn.Conv1d(
            state_dim,
            state_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=state_dim,
        )
        self.local_dropout = nn.Dropout(dropout)
        self.mid_projector = nn.Sequential(
            nn.LayerNorm(_V10_CONTEXT_DIM),
            nn.Linear(_V10_CONTEXT_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_projector = nn.Sequential(
            nn.LayerNorm(_V10_SEQUENCE_EMBED_DIM),
            nn.Linear(_V10_SEQUENCE_EMBED_DIM, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.group_projectors = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.LayerNorm(group_dim_in),
                    nn.Linear(group_dim_in, group_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for group_name, group_dim_in in _V10_BIO_GROUP_DIMS.items()
            }
        )

        bio_dim = group_dim * len(self.group_names)
        feature_dim = state_dim * 4 + bio_dim
        hidden_feature_dim = max(embed_dim * 2, feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.direct_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.direct_out = nn.Linear(hidden_feature_dim, num_tracks)
        self.factor_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feature_dim, rank_dim),
        )
        self.factor_to_tracks = nn.Linear(rank_dim, num_tracks, bias=False)
        self.track_group_logits = nn.Parameter(torch.zeros(num_tracks, len(self.group_names)))
        self.bio_track_out = nn.Linear(group_dim, 1, bias=False)
        self.factor_residual_logit = nn.Parameter(torch.tensor(-2.0))
        self.bio_residual_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.factor_to_tracks.weight)
        nn.init.zeros_(self.bio_track_out.weight)

    def _encode_bio_groups(self, aux_features: object) -> torch.Tensor:
        if not isinstance(aux_features, dict):
            raise RuntimeError("v10 BioCov residual head requires grouped v10 auxiliary features.")
        encoded_groups: list[torch.Tensor] = []
        for group_name in self.group_names:
            group_features = aux_features.get(group_name)
            if not isinstance(group_features, torch.Tensor):
                raise RuntimeError(f"v10 BioCov residual head missing auxiliary group {group_name!r}.")
            encoded_groups.append(self.group_projectors[group_name](group_features))
        return torch.stack(encoded_groups, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_features: torch.Tensor | dict[str, torch.Tensor] | None = None,
        global_context: torch.Tensor | None = None,
        mid_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if aux_features is None or global_context is None or mid_hidden_states is None:
            raise RuntimeError("v10 BioCov residual head requires aux_features, global_context, and mid states.")

        h_context, h_pure = _split_v10_hidden(hidden_states)
        mid_up = _resize_sequence_features(mid_hidden_states, hidden_states.shape[1])

        context_projection = self.context_projector(h_context)
        pure_projection = self.pure_projector(h_pure)
        local_context = self.pure_local_conv(pure_projection.transpose(1, 2)).transpose(1, 2)
        pure_local = pure_projection + self.local_dropout(F.gelu(local_context))
        mid_projection = self.mid_projector(mid_up)
        global_projection = self.global_projector(global_context).unsqueeze(1).expand(
            -1,
            hidden_states.shape[1],
            -1,
        )
        bio_groups = self._encode_bio_groups(aux_features)
        bio_concat = bio_groups.flatten(start_dim=2)

        features = torch.cat(
            [context_projection, pure_local, mid_projection, global_projection, bio_concat],
            dim=-1,
        )
        x = self.feature_norm(features)
        direct_logits = self.direct_out(self.direct_mlp(x))
        factor_logits = self.factor_to_tracks(self.factor_mlp(x))

        group_gates = torch.softmax(self.track_group_logits, dim=-1)
        track_bio_features = torch.einsum("blgd,tg->bltd", bio_groups, group_gates)
        bio_logits = self.bio_track_out(track_bio_features).squeeze(-1)
        logits = (
            direct_logits
            + torch.sigmoid(self.factor_residual_logit) * factor_logits
            + torch.sigmoid(self.bio_residual_logit) * bio_logits
        )
        return F.softplus(logits)


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(torch.finfo(values.dtype).eps)
    return values + torch.log(-torch.expm1(-values))


def initialize_functional_head_output_bias(head: nn.Module, target_means: torch.Tensor) -> None:
    """Initialize final positive-output biases to match expected transformed target means."""
    bias_values = _inverse_softplus(target_means.detach().to(dtype=torch.float32))
    candidates: list[nn.Linear] = []
    if isinstance(head, FunctionalTracksHead):
        candidates.append(head.head)
    elif isinstance(head, (FunctionalTracksMlpHead, FunctionalTracksGlobalContextHead)):
        final = head.head[-1]
        if isinstance(final, nn.Linear):
            candidates.append(final)
    elif isinstance(head, FunctionalTracksLocalConvHead):
        candidates.append(head.output_proj)
    elif isinstance(head, FunctionalTracksGatedHybridHead):
        final = head.base_head[-1]
        if isinstance(final, nn.Linear):
            candidates.append(final)
    elif isinstance(
        head,
        (
            FunctionalTracksMultiScaleDilatedHead,
            FunctionalTracksContextPyramidHead,
            FunctionalTracksV10RepresentationPyramidHead,
            FunctionalTracksV10BioAuxPyramidHead,
        ),
    ):
        final = head.output_head[-1]
        if isinstance(final, nn.Linear):
            candidates.append(final)
    elif isinstance(head, FunctionalTracksV10BioProgramStackHead):
        candidates.append(head.program_to_tracks)
    elif isinstance(head, FunctionalTracksV10ProfileCountBioAuxHead):
        candidates.append(head.count_out)
    elif isinstance(head, FunctionalTracksV10BioCovResidualHead):
        candidates.append(head.direct_out)
    if isinstance(head, FunctionalTracksV10AssayGatedBioPyramidHead):
        candidates.extend([head.local_out])
        bio_final = head.bio_head[-1]
        if isinstance(bio_final, nn.Linear):
            candidates.append(bio_final)
    if not candidates:
        raise RuntimeError(f"Unsupported functional head for output bias initialization: {type(head).__name__}.")
    for layer in candidates:
        if layer.bias is None or layer.bias.numel() != bias_values.numel():
            raise RuntimeError("Functional output bias shape does not match number of tracks.")
        with torch.no_grad():
            layer.bias.copy_(bias_values.to(device=layer.bias.device, dtype=layer.bias.dtype))


def build_functional_tracks_head(
    *,
    head_type: str,
    embed_dim: int,
    num_tracks: int,
    hidden_dim: int | None = None,
    dropout: float = 0.05,
    kernel_size: int = 15,
    aux_features: str = "none",
    aux_projection_dim: int = 16,
) -> nn.Module:
    normalized = head_type.lower().replace("_", "-")
    if normalized == "linear":
        return FunctionalTracksHead(embed_dim, num_tracks)
    if normalized == "mlp":
        return FunctionalTracksMlpHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            aux_features=aux_features,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized in {"global-context", "global", "context"}:
        return FunctionalTracksGlobalContextHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            aux_features=aux_features,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized in {"local-conv", "conv"}:
        return FunctionalTracksLocalConvHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
        )
    if normalized in {"gated-hybrid", "hybrid", "gated"}:
        return FunctionalTracksGatedHybridHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_features=aux_features,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized in {"multi-scale-dilated", "multiscale-dilated", "multi-scale", "multiscale"}:
        return FunctionalTracksMultiScaleDilatedHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_features=aux_features,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized in {"context-pyramid", "pyramid-context", "global-pyramid"}:
        return FunctionalTracksContextPyramidHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_features=aux_features,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized == "v10-representation-pyramid":
        return FunctionalTracksV10RepresentationPyramidHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized == "v10-bio-aux-pyramid":
        return FunctionalTracksV10BioAuxPyramidHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized == "v10-assay-gated-bio-pyramid":
        return FunctionalTracksV10AssayGatedBioPyramidHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized == "v10-bioprogram-stack":
        return FunctionalTracksV10BioProgramStackHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized == "v10-profile-count-bioaux":
        return FunctionalTracksV10ProfileCountBioAuxHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized == "v10-profile-count-bioaux-rc-gated-residual":
        return FunctionalTracksV10ProfileCountBioAuxGatedResidualHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized == "v10-assay-rescue-hybrid":
        return FunctionalTracksV10AssayRescueHybridHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
    if normalized == "v10-biocov-residual":
        return FunctionalTracksV10BioCovResidualHead(
            embed_dim,
            num_tracks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            aux_projection_dim=aux_projection_dim,
        )
    raise ValueError(
        "Unsupported functional_head_type="
        f"{head_type!r}; expected 'linear', 'mlp', 'global-context', 'local-conv', 'gated-hybrid', "
        "'multi-scale-dilated', 'context-pyramid', or a 'v10-*' experimental head."
    )


class AnnotationHead(nn.Module):
    def __init__(self, embed_dim: int, num_elements: int) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_elements * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.layer_norm(hidden_states))
        batch_size, sequence_length, _ = logits.shape
        return logits.reshape(batch_size, sequence_length, self.num_elements, 2)


class FunctionalTracksModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_tracks: int,
        keep_target_center_fraction: float,
        feature_source: str = "hidden",
        head_type: str = "linear",
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.05,
        head_kernel_size: int = 15,
        head_aux_features: str = "none",
        head_aux_projection_dim: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_aux_features = head_aux_features
        self.head = build_functional_tracks_head(
            head_type=head_type,
            embed_dim=embed_dim,
            num_tracks=num_tracks,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            kernel_size=head_kernel_size,
            aux_features=head_aux_features,
            aux_projection_dim=head_aux_projection_dim,
        )
        self.keep_target_center_fraction = keep_target_center_fraction
        self.feature_source = feature_source

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        mid_hidden_states = None
        if bool(getattr(self.head, "uses_v10_encoded_features", False)):
            encoded = _backbone_encoded_states(
                self.backbone,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = encoded["hidden_states"]
            mid_hidden_states = encoded["mid_hidden_states"]
            aux_features = None
            if bool(getattr(self.head, "needs_v10_all_bio_aux", False)):
                aux_features = _v10_bio_aux_feature_groups(
                    self.backbone,
                    hidden_states=hidden_states,
                    mid_hidden_states=mid_hidden_states,
                    detach=not bool(getattr(self.head, "uses_v10_differentiable_aux", False)),
                )
            elif bool(getattr(self.head, "needs_v10_bio_aux", False)):
                aux_features = _v10_bio_aux_features(
                    self.backbone,
                    hidden_states=hidden_states,
                    mid_hidden_states=mid_hidden_states,
                )
        else:
            hidden_states, aux_features = _functional_feature_bundle(
                self.backbone,
                input_ids=input_ids,
                attention_mask=attention_mask,
                feature_source=self.feature_source,
                aux_features=self.head_aux_features,
            )

        global_context = None
        if bool(getattr(self.head, "uses_global_context", False)):
            pooled_embedding = getattr(self.backbone, "pooled_embedding", None)
            if bool(getattr(self.head, "uses_v10_sequence_embedding", False)):
                global_context = _v10_sequence_embedding(
                    self.backbone,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                )
            elif callable(pooled_embedding):
                global_context = pooled_embedding(hidden_states, attention_mask)
            elif attention_mask is None:
                global_context = hidden_states.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                global_context = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            if bool(getattr(self.head, "uses_v10_encoded_features", False)):
                logits = self.head(
                    hidden_states,
                    aux_features=aux_features,
                    global_context=global_context,
                    mid_hidden_states=mid_hidden_states,
                )
            else:
                logits = self.head(hidden_states, aux_features, global_context)
        else:
            logits = self.head(hidden_states, aux_features)
        return crop_center(logits, self.keep_target_center_fraction)


class AnnotationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_elements: int,
        keep_target_center_fraction: float,
        feature_source: str = "hidden",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = AnnotationHead(embed_dim, num_elements)
        self.keep_target_center_fraction = keep_target_center_fraction
        self.feature_source = feature_source

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = _backbone_feature_states(
            self.backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_source=self.feature_source,
        )
        logits = self.head(hidden_states)
        return crop_center(logits, self.keep_target_center_fraction, sequence_axis=-3)
