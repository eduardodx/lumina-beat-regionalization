"""Gradient-enabled model adapters for end-to-end ClinVar fine-tuning.

Each adapter wraps a DNA foundation model family with a unified interface
for tokenization, hidden-state extraction (with gradients), and
nucleotide-to-token position mapping.

All four families are kept in one file for ease of comparison review.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import Tensor, nn

from eval.clinvar.variant_utils import compute_pool_bounds
from src.constants import (
    NUM_AA_CLASSES,
    NUM_ALLELE_EFFECT_CLASSES,
    NUM_MUTATION_EFFECT_CLASSES,
    SNV_ALT_TO_INDEX,
    SNV_BASES,
)
from src.models.beat_shared import get_cuda_device_capability, normalize_mamba3_runtime_config
from src.precision import PrecisionPolicy

log = logging.getLogger(__name__)

TokenizedBatch = dict[str, Any]


NATIVE_FEATURE_HEAD_DIMS: dict[str, int] = {
    "mutation_effect": NUM_MUTATION_EFFECT_CLASSES,
    "aa": NUM_AA_CLASSES,
    "codon_phylo": 1,
    "phylo470": 1,
    "mlm_logit_ratio": 1,
    "is_snv": 1,
    "allele_repr": 0,
    "allele_effect_logits": NUM_ALLELE_EFFECT_CLASSES,
    "allele_severity_score": 1,
    "allele_swap_severity_score": 1,
    "allele_far_distance": 1,
    "allele_site_delta": 0,
    "allele_local_delta": 0,
}

# ``None`` marks a synthetic head: a feature derived from inputs alone with no
# backbone module to freeze or lock in eval mode (e.g. ``is_snv``).
NATIVE_HEAD_ATTRIBUTES: dict[str, str | None] = {
    "mutation_effect": "mutation_effect_head",
    "aa": "aa_head",
    "codon_phylo": "codon_phylo_head",
    "phylo470": "phylo470_head",
    "mlm_logit_ratio": "mlm_head",
    "is_snv": None,
    "allele_repr": "allele_scorer_head",
    "allele_effect_logits": "allele_scorer_head",
    "allele_severity_score": "allele_scorer_head",
    "allele_swap_severity_score": "allele_scorer_head",
    "allele_far_distance": "allele_scorer_head",
    "allele_site_delta": "allele_scorer_head",
    "allele_local_delta": "allele_scorer_head",
}

NATIVE_FEATURE_SUPPORTED_MODEL_KEYS: frozenset[str] = frozenset({"beat-v6", "beat-v7", "beat-v8"})


def compute_native_feature_dim(selection: tuple[str, ...] | list[str], d_model: int | None = None) -> int:
    total = 0
    for name in selection:
        if name in {"allele_repr", "allele_site_delta", "allele_local_delta"}:
            total += 256 if d_model is None else int(d_model)
        else:
            total += NATIVE_FEATURE_HEAD_DIMS[name]
    return total


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FineTuneAdapter(Protocol):
    """Unified adapter interface for ClinVar fine-tuning."""

    @property
    def d_model(self) -> int: ...

    @property
    def backbone(self) -> nn.Module: ...

    def tokenize(self, sequences: list[str]) -> TokenizedBatch: ...

    def forward_hidden_states(self, batch: TokenizedBatch) -> Tensor:
        """Return last-layer hidden states [B, L, D] with gradient tracking."""
        ...

    def nuc_window_to_token_bounds(
        self,
        batch: TokenizedBatch,
        batch_index: int,
        center_nuc: int,
        radius_bp: int,
    ) -> tuple[int, int]: ...

    def extract_variant_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]: ...


@runtime_checkable
class NativeVariantFeatureAdapter(Protocol):
    """Adapter that supplies a backbone-native variant representation.

    Implemented by backbones (currently beat-v6) whose pretraining heads
    can emit per-position variant-effect features from a single ref-only
    forward pass, avoiding the generic two-pass ref/alt subtraction.
    """

    @property
    def native_variant_head_selection(self) -> tuple[str, ...] | None: ...

    @property
    def native_variant_feature_dim(self) -> int | None: ...

    def forward_native_variant_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _token_seq_len(batch: TokenizedBatch, batch_index: int) -> int:
    attention_mask = batch["attention_mask"]
    if not isinstance(attention_mask, Tensor):
        raise TypeError("Tokenized batches must expose attention_mask as a torch.Tensor.")
    return int(attention_mask[batch_index].shape[0])


def _char_level_window_to_token_bounds(
    batch: TokenizedBatch,
    batch_index: int,
    center_nuc: int,
    radius_bp: int,
) -> tuple[int, int]:
    return compute_pool_bounds(center_nuc, radius_bp, _token_seq_len(batch, batch_index))


def _approximate_subword_window_to_token_bounds(
    batch: TokenizedBatch,
    batch_index: int,
    center_nuc: int,
    radius_bp: int,
    nt_per_token: int,
) -> tuple[int, int]:
    return compute_pool_bounds(
        center_nuc // nt_per_token,
        radius_bp // nt_per_token,
        _token_seq_len(batch, batch_index),
    )


def _normalize_offset_mapping(offset_mapping: Any) -> list[list[tuple[int, int]]] | None:
    if offset_mapping is None:
        return None
    rows = offset_mapping.detach().cpu().tolist() if isinstance(offset_mapping, Tensor) else offset_mapping
    try:
        return [[(int(start), int(end)) for start, end in row] for row in rows]
    except (TypeError, ValueError):
        return None


def _offset_window_to_token_bounds(
    offsets: list[tuple[int, int]],
    center_nuc: int,
    radius_bp: int,
) -> tuple[int, int] | None:
    window_start = max(0, center_nuc - radius_bp)
    window_end = max(window_start, center_nuc + radius_bp)
    overlapping = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > tok_start and tok_end > window_start and tok_start < window_end
    ]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1] + 1


def _masked_mean_pool_with_bounds(
    hidden_states: Tensor,
    attention_mask: Tensor,
    starts: Tensor,
    ends: Tensor,
) -> Tensor:
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    region_mask = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    valid_mask = (attention_mask > 0) & region_mask
    mask = valid_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _nuc_offset_to_token_index(
    adapter: FineTuneAdapter,
    batch: TokenizedBatch,
    batch_index: int,
    nuc_offset: int,
) -> int:
    start, end = adapter.nuc_window_to_token_bounds(batch, batch_index, nuc_offset, radius_bp=1)
    token_idx = (start + end) // 2
    seq_len = int(batch["attention_mask"][batch_index].shape[0])
    return min(max(token_idx, 0), seq_len - 1)


def _extract_paired_variant_features(
    adapter: FineTuneAdapter,
    ref_seqs: list[str],
    alt_seqs: list[str],
    variant_offsets: list[int],
) -> tuple[Tensor, Tensor, Tensor]:
    ref_batch = adapter.tokenize(ref_seqs)
    alt_batch = adapter.tokenize(alt_seqs)
    ref_hidden = adapter.forward_hidden_states(ref_batch)
    alt_hidden = adapter.forward_hidden_states(alt_batch)

    batch_size = ref_hidden.shape[0]
    device = ref_hidden.device
    batch_idx = torch.arange(batch_size, device=device)

    ref_token_indices = torch.tensor(
        [_nuc_offset_to_token_index(adapter, ref_batch, i, off) for i, off in enumerate(variant_offsets)],
        dtype=torch.long,
        device=device,
    )
    alt_token_indices = torch.tensor(
        [_nuc_offset_to_token_index(adapter, alt_batch, i, off) for i, off in enumerate(variant_offsets)],
        dtype=torch.long,
        device=device,
    )

    site_ref = ref_hidden[batch_idx, ref_token_indices]
    site_alt = alt_hidden[batch_idx, alt_token_indices]
    variant_repr = site_alt - site_ref

    local_bounds = [
        adapter.nuc_window_to_token_bounds(ref_batch, i, off, radius_bp=64)
        for i, off in enumerate(variant_offsets)
    ]
    local_starts = torch.tensor([s for s, _ in local_bounds], dtype=torch.long, device=device)
    local_ends = torch.tensor([e for _, e in local_bounds], dtype=torch.long, device=device)
    local_context = _masked_mean_pool_with_bounds(
        ref_hidden,
        ref_batch["attention_mask"],
        local_starts,
        local_ends,
    )
    return site_ref, variant_repr, local_context


def _load_torch_checkpoint(path: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path!r} must contain a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def normalize_finetune_model_family(model_family: str) -> str:
    """Normalize user-facing model-family aliases to the canonical ClinVar spelling."""
    family = model_family.strip().lower()
    if family == "dnabert-2":
        return "dnabert2"
    if family in {"beat_v11", "beat-v11-bioprime", "beat_v11_bioprime", "beatv11"}:
        return "beat-v11"
    return family


def _env_flag_is_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _huggingface_hub_kwargs(*, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    resolved_env = os.environ if env is None else env
    kwargs: dict[str, Any] = {}
    cache_dir = resolved_env.get("HF_HUB_CACHE")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if _env_flag_is_true(resolved_env.get("HF_HUB_OFFLINE")) or _env_flag_is_true(
        resolved_env.get("TRANSFORMERS_OFFLINE")
    ):
        kwargs["local_files_only"] = True
    return kwargs


def _resolve_caduceus_hidden_dim(config: Any) -> int:
    """Resolve the effective hidden width emitted by a Caduceus backbone."""
    d_model = int(config.d_model)
    # RCPS backbones expose forward and reverse-complement channels together.
    if bool(config.rcps):
        return d_model * 2
    return d_model


def _resolve_lumina_checkpoint_spec(
    requested_model_key: str,
    checkpoint_path: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from src.models import normalize_model_key

    checkpoint = _load_torch_checkpoint(checkpoint_path)
    raw_config = checkpoint.get("config")
    config = raw_config if isinstance(raw_config, dict) else {}

    checkpoint_model_key = config.get("model")
    if isinstance(checkpoint_model_key, str) and checkpoint_model_key.strip():
        normalized_ckpt = normalize_model_key(checkpoint_model_key)
    else:
        normalized_ckpt = normalize_model_key(requested_model_key)

    normalized_req = normalize_model_key(requested_model_key)
    if normalized_ckpt != normalized_req:
        raise ValueError(
            f"Checkpoint model mismatch: requested {normalized_req!r}, "
            f"but checkpoint specifies {normalized_ckpt!r}."
        )

    raw_model_config = config.get("model_config")
    model_config = raw_model_config if isinstance(raw_model_config, dict) else {}
    return normalized_ckpt, model_config, checkpoint


def _normalize_lumina_model_config_for_precision(
    model_key: str,
    model_config: Mapping[str, Any] | None,
    precision: PrecisionPolicy | None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    from src.models import resolve_model_config_dict

    resolved = resolve_model_config_dict(
        model_key,
        model_config,
        source="checkpoint['config']['model_config']",
    )
    normalized, notes = normalize_mamba3_runtime_config(
        model_key,
        resolved,
        uses_bf16_compute=(precision.uses_bf16_compute if precision is not None else False),
        cuda_device_capability=(get_cuda_device_capability(device) if device is not None else None),
    )
    for note in notes:
        log.info(
            "Adjusted Lumina runtime model config for ClinVar fine-tuning "
            "(model=%s resolved_precision=%s): %s",
            model_key,
            precision.resolved if precision is not None else "fp32",
            note,
        )
    return normalized


def _disable_dnabert2_triton_attention(model: nn.Module) -> bool:
    patched_any = False
    module_name = getattr(model.__class__, "__module__", "")
    module_prefix = module_name.rsplit(".", 1)[0] if "." in module_name else module_name

    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if module_name and not (name == module_name or (module_prefix and name.startswith(f"{module_prefix}."))):
            continue
        if hasattr(module, "flash_attn_qkvpacked_func"):
            cast(Any, module).flash_attn_qkvpacked_func = None
            patched_any = True

    for submodule in model.modules():
        if submodule.__class__.__name__ != "BertUnpadSelfAttention":
            continue
        p_dropout = getattr(submodule, "p_dropout", None)
        if isinstance(p_dropout, (int, float)):
            cast(Any, submodule).p_dropout = max(float(p_dropout), 1e-6)
            patched_any = True

    return patched_any


def _extract_dnabert2_hidden(outputs: Any) -> Tensor:
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if isinstance(last_hidden, torch.Tensor):
        return last_hidden
    if isinstance(outputs, dict):
        hidden = outputs.get("last_hidden_state")
        if isinstance(hidden, torch.Tensor):
            return hidden
    if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
        return outputs[0]
    raise RuntimeError("DNABERT-2 forward pass did not return a recognizable hidden-state tensor.")


def _extract_lumina_hidden(encoded: Any) -> Tensor:
    if isinstance(encoded, torch.Tensor):
        return encoded
    if isinstance(encoded, dict):
        hidden = encoded.get("hidden_states")
        if isinstance(hidden, torch.Tensor):
            return hidden
    if isinstance(encoded, (tuple, list)) and encoded and isinstance(encoded[0], torch.Tensor):
        return encoded[0]
    raise RuntimeError("Lumina encode() did not return a recognizable hidden-state tensor.")


def _resolve_lumina_hidden_dim(model: Any) -> int:
    full_hidden_dim = getattr(model, "full_hidden_dim", None)
    if full_hidden_dim is not None:
        return int(full_hidden_dim)
    cfg = getattr(model, "cfg", None)
    d_model = getattr(cfg, "d_model", None)
    if d_model is None:
        raise RuntimeError("Lumina backbone does not expose cfg.d_model or full_hidden_dim.")
    return int(d_model)


# ---------------------------------------------------------------------------
# Model repositories
# ---------------------------------------------------------------------------

NTV3_REPOS: dict[str, str] = {
    "8M_pre": "InstaDeepAI/NTv3_8M_pre",
    "100M_pre": "InstaDeepAI/NTv3_100M_pre",
    "650M_pre": "InstaDeepAI/NTv3_650M_pre",
}

CADUCEUS_REPOS: dict[str, str] = {
    "caduceus-ph": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    "caduceus-ps": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
}

DNABERT2_REPOS: dict[str, str] = {
    "117M": "zhihan1996/DNABERT-2-117M",
}


# ---------------------------------------------------------------------------
# Nucleotide Transformers v3
# ---------------------------------------------------------------------------


class FineTuneNTv3Adapter:
    """Gradient-enabled adapter for InstaDeep Nucleotide Transformers v3."""

    def __init__(
        self, repo: str, device: torch.device, precision: PrecisionPolicy | None = None,
    ) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        hub_kwargs = _huggingface_hub_kwargs()
        self._tokenizer: Any = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, **hub_kwargs)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True, **hub_kwargs}
        if precision is not None and precision.uses_bf16_compute:
            bf16_kwargs = {
                "stem_compute_dtype": "bfloat16",
                "down_convolution_compute_dtype": "bfloat16",
                "transformer_qkvo_compute_dtype": "bfloat16",
                "transformer_ffn_compute_dtype": "bfloat16",
                "up_convolution_compute_dtype": "bfloat16",
                "modulation_compute_dtype": "bfloat16",
            }
            try:
                self._model: nn.Module = AutoModelForMaskedLM.from_pretrained(repo, **model_kwargs, **bf16_kwargs)
            except Exception as exc:
                log.warning("NTv3 bf16 kwargs rejected for %s; falling back: %s", repo, exc)
                self._model = AutoModelForMaskedLM.from_pretrained(repo, **model_kwargs)
        else:
            self._model = AutoModelForMaskedLM.from_pretrained(repo, **model_kwargs)
        self._model.to(device)
        self._device = device
        self._d_model: int = self._model.config.embed_dim  # type: ignore[union-attr]

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._model

    def tokenize(self, sequences: list[str]) -> TokenizedBatch:
        encoded = self._tokenizer(
            sequences, add_special_tokens=False, padding=True,
            pad_to_multiple_of=128, return_tensors="pt",
        )
        result = {k: v.to(self._device) for k, v in encoded.items()}
        if "attention_mask" not in result:
            result["attention_mask"] = (result["input_ids"] != self._tokenizer.pad_token_id).long()
        return result

    def forward_hidden_states(self, batch: TokenizedBatch) -> Tensor:
        outputs = self._model(input_ids=batch["input_ids"], output_hidden_states=True)
        return outputs.hidden_states[-1]

    def nuc_window_to_token_bounds(
        self, batch: TokenizedBatch, batch_index: int, center_nuc: int, radius_bp: int,
    ) -> tuple[int, int]:
        return _char_level_window_to_token_bounds(batch, batch_index, center_nuc, radius_bp)

    def extract_variant_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        _ = (ref_alleles, alt_alleles)
        return _extract_paired_variant_features(self, ref_seqs, alt_seqs, variant_offsets)


# ---------------------------------------------------------------------------
# Caduceus
# ---------------------------------------------------------------------------


class FineTuneCaduceusAdapter:
    """Gradient-enabled adapter for Caduceus models."""

    def __init__(self, repo: str, device: torch.device) -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        hub_kwargs = _huggingface_hub_kwargs()
        self._tokenizer: Any = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, **hub_kwargs)
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True, **hub_kwargs)
        model = AutoModel.from_config(config, trust_remote_code=True)
        weights_path = hf_hub_download(repo, filename="model.safetensors", **hub_kwargs)
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        self._model: nn.Module = model.to(device)
        self._device = device
        self._d_model: int = _resolve_caduceus_hidden_dim(config)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._model

    def tokenize(self, sequences: list[str]) -> TokenizedBatch:
        encoded = self._tokenizer(sequences, padding=True, return_tensors="pt")
        result = {k: v.to(self._device) for k, v in encoded.items()}
        if "attention_mask" not in result:
            result["attention_mask"] = (result["input_ids"] != self._tokenizer.pad_token_id).long()
        return result

    def forward_hidden_states(self, batch: TokenizedBatch) -> Tensor:
        outputs = self._model(input_ids=batch["input_ids"], output_hidden_states=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        return outputs.last_hidden_state

    def nuc_window_to_token_bounds(
        self, batch: TokenizedBatch, batch_index: int, center_nuc: int, radius_bp: int,
    ) -> tuple[int, int]:
        return _char_level_window_to_token_bounds(batch, batch_index, center_nuc, radius_bp)

    def extract_variant_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        _ = (ref_alleles, alt_alleles)
        return _extract_paired_variant_features(self, ref_seqs, alt_seqs, variant_offsets)


# ---------------------------------------------------------------------------
# Lumina
# ---------------------------------------------------------------------------


class FineTuneLuminaAdapter:
    """Gradient-enabled adapter for internal Lumina models."""

    def __init__(
        self,
        model_key: str,
        device: torch.device,
        checkpoint_path: str | None = None,
        precision: PrecisionPolicy | None = None,
        native_feature_heads: list[str] | None = None,
    ) -> None:
        from src.constants import DNA_VOCAB, PAD_ID, UNK_ID
        from src.models.registry import build_registered_model

        self._vocab = DNA_VOCAB
        self._pad_id = PAD_ID
        self._unk_id = UNK_ID

        checkpoint: dict[str, Any] | None = None
        resolved_model_key = model_key
        resolved_model_config: dict[str, Any] | None = None
        if checkpoint_path is not None:
            resolved_model_key, resolved_model_config, checkpoint = (
                _resolve_lumina_checkpoint_spec(model_key, checkpoint_path)
            )

        effective_model_config = _normalize_lumina_model_config_for_precision(
            resolved_model_key,
            resolved_model_config,
            precision,
            device,
        )
        log.info(
            "Lumina ClinVar backbone config: model=%s chunk_size=%s is_mimo=%s mimo_rank=%s precision=%s",
            resolved_model_key,
            effective_model_config.get("chunk_size"),
            effective_model_config.get("is_mimo"),
            effective_model_config.get("mimo_rank"),
            precision.resolved if precision is not None else "fp32",
        )
        self._model: Any = build_registered_model(resolved_model_key, effective_model_config)

        if checkpoint is not None:
            model_state = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
            self._model.load_state_dict(model_state)

        self._model.to(device)
        self._device = device
        self._d_model = _resolve_lumina_hidden_dim(self._model)
        self._model_key = resolved_model_key
        self._requested_native_heads: tuple[str, ...] = (
            tuple(native_feature_heads) if native_feature_heads is not None else ()
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._model

    def _encode_seq(self, seq: str) -> list[int]:
        return [self._vocab.get(b.upper(), self._unk_id) for b in seq]

    def tokenize(self, sequences: list[str]) -> TokenizedBatch:
        max_len = max(len(s) for s in sequences)
        input_ids = []
        attention_mask = []
        for seq in sequences:
            ids = self._encode_seq(seq)
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self._pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self._device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self._device),
        }

    def encode_alleles(self, alleles: list[str]) -> torch.Tensor:
        max_len = max(max(len(allele), 1) for allele in alleles)
        encoded: list[list[int]] = []
        for allele in alleles:
            ids = self._encode_seq(allele) if allele else [self._pad_id]
            pad_len = max_len - len(ids)
            encoded.append(ids + [self._pad_id] * pad_len)
        return torch.tensor(encoded, dtype=torch.long, device=self._device)

    def forward_hidden_states(self, batch: TokenizedBatch) -> Tensor:
        encoded = self._model.encode(batch["input_ids"])
        return _extract_lumina_hidden(encoded)

    def forward_sequence_features(self, batch: TokenizedBatch) -> dict[str, Tensor]:
        return cast(Any, self._model).extract_sequence_features(batch["input_ids"])

    def nuc_window_to_token_bounds(
        self, batch: TokenizedBatch, batch_index: int, center_nuc: int, radius_bp: int,
    ) -> tuple[int, int]:
        return _char_level_window_to_token_bounds(batch, batch_index, center_nuc, radius_bp)

    def extract_variant_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        _ = (ref_alleles, alt_alleles)
        return _extract_paired_variant_features(self, ref_seqs, alt_seqs, variant_offsets)

    @property
    def native_variant_head_selection(self) -> tuple[str, ...] | None:
        """Resolve the native-head selection against backbone capability.

        Returns ``None`` when the new single-pass path is unavailable or
        explicitly disabled (caller passed ``["none"]``).
        """
        if self._model_key not in NATIVE_FEATURE_SUPPORTED_MODEL_KEYS:
            return None
        requested = self._requested_native_heads
        if not requested or requested == ("none",):
            return None
        cfg = getattr(self._model, "cfg", None)
        if getattr(cfg, "decoder_dim", None) is not None:
            return None
        resolved: list[str] = []
        for head_name in requested:
            if head_name not in NATIVE_HEAD_ATTRIBUTES:
                continue
            attr = NATIVE_HEAD_ATTRIBUTES[head_name]
            if attr is None:
                # Synthetic head — no backbone module required.
                resolved.append(head_name)
                continue
            module = getattr(self._model, attr, None)
            if isinstance(module, nn.Module):
                resolved.append(head_name)
        if not resolved:
            return None
        return tuple(resolved)

    @property
    def native_variant_feature_dim(self) -> int | None:
        selection = self.native_variant_head_selection
        if selection is None:
            return None
        return compute_native_feature_dim(selection, self._d_model)

    def forward_native_variant_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single-pass variant features using pretrained per-position heads.

        Returns ``(site_ref, variant_native, local_context)`` where
        ``variant_native`` concatenates the selected pretrained head outputs
        at the variant position (with the mutation_effect slot zeroed for
        non-SNV rows).
        """
        selection = self.native_variant_head_selection
        if selection is None:
            raise RuntimeError(
                "forward_native_variant_features called but backbone does not "
                "expose a native variant feature selection."
            )

        ref_batch = self.tokenize(ref_seqs)
        hidden = self.forward_hidden_states(ref_batch)
        batch_size = hidden.shape[0]
        device = hidden.device
        batch_idx = torch.arange(batch_size, device=device)

        ref_token_indices = torch.tensor(
            [_nuc_offset_to_token_index(self, ref_batch, i, off) for i, off in enumerate(variant_offsets)],
            dtype=torch.long,
            device=device,
        )

        site_hidden = hidden[batch_idx, ref_token_indices]

        local_bounds = [
            self.nuc_window_to_token_bounds(ref_batch, i, off, radius_bp=64)
            for i, off in enumerate(variant_offsets)
        ]
        local_starts = torch.tensor([s for s, _ in local_bounds], dtype=torch.long, device=device)
        local_ends = torch.tensor([e for _, e in local_bounds], dtype=torch.long, device=device)
        local_context = _masked_mean_pool_with_bounds(
            hidden, ref_batch["attention_mask"], local_starts, local_ends,
        )

        alt_batch: TokenizedBatch | None = None
        alt_hidden: Tensor | None = None
        alt_token_indices: Tensor | None = None

        def _ensure_alt_hidden() -> tuple[TokenizedBatch, Tensor, Tensor]:
            nonlocal alt_batch, alt_hidden, alt_token_indices
            if alt_batch is None:
                alt_batch = self.tokenize(alt_seqs)
            if alt_hidden is None:
                alt_hidden = self.forward_hidden_states(alt_batch)
            if alt_token_indices is None:
                alt_token_indices = torch.tensor(
                    [
                        _nuc_offset_to_token_index(self, alt_batch, i, off)
                        for i, off in enumerate(variant_offsets)
                    ],
                    dtype=torch.long,
                    device=device,
                )
            return alt_batch, alt_hidden, alt_token_indices

        def _site_delta_from_alt_hidden() -> Tensor:
            _, hidden_alt, token_indices = _ensure_alt_hidden()
            return hidden_alt[batch_idx, token_indices] - site_hidden

        def _local_delta_from_alt_hidden() -> Tensor:
            batch_alt, hidden_alt, _ = _ensure_alt_hidden()
            alt_bounds = [
                self.nuc_window_to_token_bounds(batch_alt, i, off, radius_bp=64)
                for i, off in enumerate(variant_offsets)
            ]
            alt_starts = torch.tensor([s for s, _ in alt_bounds], dtype=torch.long, device=device)
            alt_ends = torch.tensor([e for _, e in alt_bounds], dtype=torch.long, device=device)
            local_alt = _masked_mean_pool_with_bounds(
                hidden_alt, batch_alt["attention_mask"], alt_starts, alt_ends,
            )
            return local_alt - local_context

        features: list[Tensor] = []
        needs_allele_scorer = any(head_name.startswith("allele_") for head_name in selection)
        allele_outputs: dict[str, Tensor] | None = None
        if needs_allele_scorer:
            score_alleles = getattr(self._model, "score_alleles_from_ids", None)
            if not callable(score_alleles):
                raise RuntimeError("Selected allele scorer features, but the Lumina backbone does not expose them.")
            score_alleles = cast(Callable[..., dict[str, Tensor]], score_alleles)
            if alt_batch is None:
                alt_batch = self.tokenize(alt_seqs)
            alt_ids = torch.tensor(
                [
                    self._vocab.get(allele.upper(), self._pad_id) if len(allele) == 1 else self._pad_id
                    for allele in alt_alleles
                ],
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)
            allele_outputs = score_alleles(
                ref_input_ids=ref_batch["input_ids"],
                alt_input_ids=alt_batch["input_ids"].unsqueeze(1),
                allele_position=ref_token_indices,
                allele_alt_ids=alt_ids,
                attention_mask=ref_batch["attention_mask"],
                alt_attention_mask=alt_batch["attention_mask"].unsqueeze(1),
            )
            snv_mask = self._is_snv_mask(ref_alleles, alt_alleles, device=device)
            for key in (
                "allele_repr",
                "allele_effect_logits",
                "allele_severity_score",
                "allele_swap_severity_score",
            ):
                output = allele_outputs.get(key)
                if output is not None:
                    mask_shape = (snv_mask.shape[0],) + (1,) * (output.ndim - 1)
                    allele_outputs[key] = output * snv_mask.reshape(mask_shape).to(dtype=output.dtype)
        for head_name in selection:
            if head_name == "mutation_effect":
                features.append(
                    self._mutation_effect_variant_slot(site_hidden, ref_alleles, alt_alleles, device)
                )
            elif head_name == "aa":
                features.append(self._model.aa_head(site_hidden))
            elif head_name == "codon_phylo":
                features.append(self._model.codon_phylo_head(site_hidden))
            elif head_name == "phylo470":
                features.append(self._model.phylo470_head(site_hidden))
            elif head_name == "mlm_logit_ratio":
                features.append(
                    self._mlm_logit_ratio_feature(site_hidden, ref_alleles, alt_alleles, device)
                )
            elif head_name == "is_snv":
                features.append(
                    self._is_snv_feature(ref_alleles, alt_alleles, site_hidden.dtype, device)
                )
            elif head_name == "allele_repr":
                assert allele_outputs is not None
                features.append(allele_outputs["allele_repr"].squeeze(1))
            elif head_name == "allele_effect_logits":
                assert allele_outputs is not None
                features.append(allele_outputs["allele_effect_logits"].squeeze(1))
            elif head_name == "allele_severity_score":
                assert allele_outputs is not None
                features.append(allele_outputs["allele_severity_score"].squeeze(1).unsqueeze(-1))
            elif head_name == "allele_swap_severity_score":
                assert allele_outputs is not None
                features.append(allele_outputs["allele_swap_severity_score"].squeeze(1).unsqueeze(-1))
            elif head_name == "allele_far_distance":
                assert allele_outputs is not None
                features.append(allele_outputs["allele_far_distance"].squeeze(1).unsqueeze(-1))
            elif head_name == "allele_site_delta":
                assert allele_outputs is not None
                site_alt = allele_outputs.get("allele_site_alt")
                site_ref = allele_outputs.get("allele_site_ref")
                if site_alt is not None and site_ref is not None:
                    features.append(site_alt.squeeze(1) - site_ref)
                else:
                    features.append(_site_delta_from_alt_hidden())
            elif head_name == "allele_local_delta":
                assert allele_outputs is not None
                local_alt = allele_outputs.get("allele_local_alt")
                local_ref = allele_outputs.get("allele_local_context")
                if local_alt is not None and local_ref is not None:
                    features.append(local_alt.squeeze(1) - local_ref)
                else:
                    features.append(_local_delta_from_alt_hidden())
            else:
                raise ValueError(f"Unsupported native feature head {head_name!r}")

        variant_native = torch.cat(features, dim=-1)
        return site_hidden, variant_native, local_context

    def _apply_single_snv(self, ref_seq: str, variant_offset: int, alt_allele: str) -> str:
        if len(alt_allele) != 1 or not (0 <= variant_offset < len(ref_seq)):
            return ref_seq
        bases = list(ref_seq)
        bases[variant_offset] = alt_allele.upper()
        return "".join(bases)

    def _mutation_effect_variant_slot(
        self,
        site_hidden: Tensor,
        ref_alleles: list[str],
        alt_alleles: list[str],
        device: torch.device,
    ) -> Tensor:
        """mutation_effect_head logits at the variant position sliced on alt base.

        Zeroes the output for non-SNV rows (indels / MNVs / N / non-ACGT).
        """
        batch_size = site_hidden.shape[0]
        batch_idx = torch.arange(batch_size, device=device)

        alt_base_ids: list[int] = []
        is_snv_flags: list[bool] = []
        for ref_allele, alt_allele in zip(ref_alleles, alt_alleles, strict=True):
            is_snv = (
                len(ref_allele) == 1
                and len(alt_allele) == 1
                and alt_allele in SNV_ALT_TO_INDEX
                and ref_allele in SNV_ALT_TO_INDEX
            )
            is_snv_flags.append(is_snv)
            alt_base_ids.append(SNV_ALT_TO_INDEX[alt_allele] if is_snv else 0)

        alt_base_idx = torch.tensor(alt_base_ids, dtype=torch.long, device=device)
        logits = self._model.mutation_effect_head(site_hidden).reshape(
            batch_size, len(SNV_BASES), NUM_MUTATION_EFFECT_CLASSES,
        )
        selected = logits[batch_idx, alt_base_idx]
        is_snv_mask = torch.tensor(is_snv_flags, dtype=selected.dtype, device=device).unsqueeze(-1)
        return selected * is_snv_mask

    def _mlm_logit_ratio_feature(
        self,
        site_hidden: Tensor,
        ref_alleles: list[str],
        alt_alleles: list[str],
        device: torch.device,
    ) -> Tensor:
        """Log-odds ``mlm[alt] - mlm[ref]`` at the variant position for SNVs.

        Raw-logit difference equals the log-softmax difference (the partition
        function cancels in pairwise subtraction).  Zeroed for non-SNV rows to
        match the mutation_effect contract.
        """
        mlm_logits = self._model.mlm_head(site_hidden)
        batch_size = site_hidden.shape[0]
        batch_idx = torch.arange(batch_size, device=device)

        ref_ids: list[int] = []
        alt_ids: list[int] = []
        is_snv_flags: list[bool] = []
        for ref_allele, alt_allele in zip(ref_alleles, alt_alleles, strict=True):
            snv = (
                len(ref_allele) == 1
                and len(alt_allele) == 1
                and ref_allele in self._vocab
                and alt_allele in self._vocab
            )
            is_snv_flags.append(snv)
            ref_ids.append(self._vocab[ref_allele] if snv else self._pad_id)
            alt_ids.append(self._vocab[alt_allele] if snv else self._pad_id)

        ref_t = torch.tensor(ref_ids, dtype=torch.long, device=device)
        alt_t = torch.tensor(alt_ids, dtype=torch.long, device=device)
        ratio = mlm_logits[batch_idx, alt_t] - mlm_logits[batch_idx, ref_t]
        mask = torch.tensor(is_snv_flags, dtype=ratio.dtype, device=device)
        return (ratio * mask).unsqueeze(-1)

    def _is_snv_mask(
        self,
        ref_alleles: list[str],
        alt_alleles: list[str],
        device: torch.device,
    ) -> Tensor:
        flags = [
            len(ref_allele) == 1
            and len(alt_allele) == 1
            and ref_allele in SNV_ALT_TO_INDEX
            and alt_allele in SNV_ALT_TO_INDEX
            for ref_allele, alt_allele in zip(ref_alleles, alt_alleles, strict=True)
        ]
        return torch.tensor(flags, dtype=torch.bool, device=device)

    def _is_snv_feature(
        self,
        ref_alleles: list[str],
        alt_alleles: list[str],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Binary indicator ``1.0`` for SNVs and ``0.0`` otherwise."""
        return self._is_snv_mask(ref_alleles, alt_alleles, device=device).to(dtype=dtype).unsqueeze(-1)


# ---------------------------------------------------------------------------
# DNABERT-2
# ---------------------------------------------------------------------------


class FineTuneDNABERT2Adapter:
    """Gradient-enabled adapter for DNABERT-2 models."""

    def __init__(self, repo: str, device: torch.device) -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        hub_kwargs = _huggingface_hub_kwargs()
        self._tokenizer: Any = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, **hub_kwargs)
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True, **hub_kwargs)
        cast(Any, config).pad_token_id = cast(Any, self._tokenizer).pad_token_id

        model = AutoModel.from_config(config, trust_remote_code=True)
        try:
            weights_path = hf_hub_download(repo, filename="model.safetensors", **hub_kwargs)
            state_dict = load_file(weights_path)
        except Exception:
            weights_path = hf_hub_download(repo, filename="pytorch_model.bin", **hub_kwargs)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        triton_disabled = _disable_dnabert2_triton_attention(model)
        if triton_disabled:
            log.info("Disabled DNABERT-2 Triton attention; using PyTorch fallback")
        else:
            log.warning("Could not confirm DNABERT-2 Triton attention was disabled")

        self._model: nn.Module = model.to(device)
        self._device = device
        self._d_model: int = config.hidden_size
        self._warned_offset_fallback = False

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._model

    def tokenize(self, sequences: list[str]) -> TokenizedBatch:
        encoded_kwargs: dict[str, Any] = {"padding": True, "return_tensors": "pt"}
        offset_mapping: Any = None
        if getattr(self._tokenizer, "is_fast", False):
            try:
                encoded = self._tokenizer(sequences, return_offsets_mapping=True, **encoded_kwargs)
                offset_mapping = encoded.pop("offset_mapping", None)
            except (NotImplementedError, TypeError, ValueError):
                encoded = self._tokenizer(sequences, **encoded_kwargs)
        else:
            encoded = self._tokenizer(sequences, **encoded_kwargs)

        result: TokenizedBatch = {k: v.to(self._device) for k, v in encoded.items()}
        if offset_mapping is not None:
            if isinstance(offset_mapping, Tensor):
                result["offset_mapping"] = offset_mapping.to(dtype=torch.long)
            else:
                result["offset_mapping"] = torch.tensor(offset_mapping, dtype=torch.long)
        return result

    def forward_hidden_states(self, batch: TokenizedBatch) -> Tensor:
        model_inputs = {
            key: value for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"} and isinstance(value, Tensor)
        }
        outputs = self._model(**model_inputs)
        return _extract_dnabert2_hidden(outputs)

    def nuc_window_to_token_bounds(
        self, batch: TokenizedBatch, batch_index: int, center_nuc: int, radius_bp: int,
    ) -> tuple[int, int]:
        offset_mapping = _normalize_offset_mapping(batch.get("offset_mapping"))
        if offset_mapping is not None and batch_index < len(offset_mapping):
            bounds = _offset_window_to_token_bounds(offset_mapping[batch_index], center_nuc, radius_bp)
            if bounds is not None:
                return bounds
        if not self._warned_offset_fallback:
            log.warning("DNABERT-2 offsets unavailable; falling back to approximate bounds.")
            self._warned_offset_fallback = True
        return _approximate_subword_window_to_token_bounds(batch, batch_index, center_nuc, radius_bp, nt_per_token=3)

    def extract_variant_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        _ = (ref_alleles, alt_alleles)
        return _extract_paired_variant_features(self, ref_seqs, alt_seqs, variant_offsets)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_finetune_adapter(
    model_family: str,
    model_version: str,
    device: torch.device,
    checkpoint_path: str | None = None,
    precision: PrecisionPolicy | None = None,
    native_feature_heads: list[str] | None = None,
) -> FineTuneAdapter:
    """Build a gradient-enabled adapter for the given model family."""
    family = normalize_finetune_model_family(model_family)

    if family == "ntv3":
        repo = NTV3_REPOS.get(model_version)
        if repo is None:
            raise ValueError(f"Unknown NTv3 version {model_version!r}. Available: {list(NTV3_REPOS)}")
        return FineTuneNTv3Adapter(repo, device, precision=precision)  # type: ignore[return-value]

    if family == "caduceus":
        repo = CADUCEUS_REPOS.get(model_version)
        if repo is None:
            raise ValueError(f"Unknown Caduceus version {model_version!r}. Available: {list(CADUCEUS_REPOS)}")
        return FineTuneCaduceusAdapter(repo, device)  # type: ignore[return-value]

    if family == "lumina":
        return FineTuneLuminaAdapter(  # type: ignore[return-value]
            model_key=model_version,
            device=device,
            checkpoint_path=checkpoint_path,
            precision=precision,
            native_feature_heads=native_feature_heads,
        )

    if family == "beat-v11":
        # Standalone Beat-v11 BioPrime package (vendored ``lumina_beat_v11``). The checkpoint's own
        # config drives the architecture (e.g. r1: d_full=448), so ``model_version`` is advisory and
        # the registry / native-head paths do not apply -- it loads via the package loader. Compute
        # precision is left to the trainer's autocast, matching the other frozen-feature paths.
        if checkpoint_path is None:
            raise ValueError(
                "beat-v11 requires a checkpoint, e.g. "
                "s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt"
            )
        from eval.clinvar.beat_v11_adapter import FineTuneBeatV11Adapter

        return FineTuneBeatV11Adapter(checkpoint_path, device)  # type: ignore[return-value]

    if family == "dnabert2":
        repo = DNABERT2_REPOS.get(model_version)
        if repo is None:
            raise ValueError(f"Unknown DNABERT-2 version {model_version!r}. Available: {list(DNABERT2_REPOS)}")
        return FineTuneDNABERT2Adapter(repo, device)  # type: ignore[return-value]

    raise ValueError(
        f"Unknown model family {family!r}. Available: ntv3, caduceus, lumina, beat-v11, dnabert2"
    )
