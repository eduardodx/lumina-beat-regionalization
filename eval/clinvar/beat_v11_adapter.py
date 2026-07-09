"""Gradient-enabled adapter for the vendored Beat-v11 BioPrime package.

Phase-0 integration for the ABRAOM regionalization port (Beat-v10 -> Beat-v11).
Unlike ``FineTuneLuminaAdapter`` (which builds through the ``src/models`` registry),
this loads the standalone ``lumina_beat_v11`` package via its own checkpoint loader,
because:

  * the r1 release ships a 74-key ``beat_v11_bioprime`` config that the registry's
    dataclass normalizer (``_normalize_dataclass_config``) would reject;
  * the package loader strips ``module.``/``_orig_mod.`` prefixes, reads ``s3://``
    URIs, and loads the SISO r1 checkpoint with ``strict=True`` (verified live);
  * v11's ``encode()`` returns ``last_hidden_state`` -- Beat-v10 returned
    ``hidden_states`` (see ``src/models/beat_v10/model.py`` vs the vendored
    ``lumina_beat_v11/model.py``), so ``_extract_lumina_hidden`` (which keys on
    ``hidden_states``) would raise here; the key is pulled directly instead.

The DNA vocab is byte-identical to ``src.constants`` (PAD=0, MASK=6, UNK=7,
VOCAB_SIZE=8, SNV_BASES=ACGT), so tokenization matches the other Lumina adapters and
``_extract_paired_variant_features`` / ``_char_level_window_to_token_bounds`` are
reused unchanged.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)

TokenizedBatch = dict[str, Any]


def install_tilelang_fallback_shim() -> bool:
    """Force ``mamba_ssm``'s Mamba3 kernel onto its triton fallback path.

    The SageMaker image ships a tilelang/tvm build whose import raises a ``tvm_ffi``
    ``AttributeError`` under Python 3.12; ``mamba_ssm`` imports tilelang eagerly at
    import time, so ``import lumina_beat_v11`` fails before it can select a backend.
    Registering a ``None`` sentinel in ``sys.modules`` makes ``import tilelang`` raise
    ``ImportError``, which ``mamba_ssm`` catches to fall back to triton.

    MUST run before the first ``mamba_ssm`` import (i.e. before importing
    ``lumina_beat_v11``). Idempotent; a no-op once tilelang has imported for real.
    Returns True iff the sentinel was installed.

    TODO(env): replace with a real fix to the SageMaker image's tilelang/tvm build.
    This is a session workaround -- acceptable for inference/extraction, fragile for
    long training runs.
    """
    if isinstance(sys.modules.get("tilelang"), types.ModuleType):
        return False  # a real module already imported cleanly -> leave it alone
    sys.modules["tilelang"] = None  # poison: `import tilelang` -> ImportError -> triton
    return True


class FineTuneBeatV11Adapter:
    """Adapter exposing the Beat-v11 BioPrime backbone through the ``FineTuneAdapter``
    protocol used by the ClinVar/regionalization training + eval harness."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        install_tilelang_fallback_shim()
        # Lazy imports: the shim above must run before mamba_ssm is pulled in.
        from lumina_beat_v11 import load_model_from_checkpoint
        from src.constants import DNA_VOCAB, PAD_ID, UNK_ID

        self._vocab = DNA_VOCAB
        self._pad_id = int(PAD_ID)
        self._unk_id = int(UNK_ID)
        self._device = device
        self._model: Any = load_model_from_checkpoint(
            checkpoint_path, device=device, dtype=dtype, strict=True,
        )
        # ``full_hidden_dim`` == cfg.d_full == d_model(256) + d_pure(64) == 320.
        self._d_model = int(getattr(self._model, "full_hidden_dim", None) or self._model.cfg.d_full)
        log.info(
            "Loaded Beat-v11 BioPrime backbone (d_full=%d) from %s",
            self._d_model, checkpoint_path,
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
        # Char-level, right-padded -- identical scheme to FineTuneLuminaAdapter.
        max_len = max(len(s) for s in sequences)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for seq in sequences:
            ids = self._encode_seq(seq)
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self._pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self._device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self._device),
        }

    def forward_hidden_states(self, batch: TokenizedBatch) -> Tensor:
        # v11 encode() -> {"last_hidden_state": [B,L,320], "mid_hidden_state": [B,L/4,256], ...};
        # attention_mask is optional (encode() defaults it to input_ids != PAD).
        encoded = self._model.encode(batch["input_ids"])
        hidden = encoded.get("last_hidden_state") if isinstance(encoded, dict) else None
        if not isinstance(hidden, torch.Tensor):
            keys = list(encoded) if isinstance(encoded, dict) else type(encoded).__name__
            raise RuntimeError(
                f"Beat-v11 encode() did not return a 'last_hidden_state' tensor (got {keys})."
            )
        return hidden

    def nuc_window_to_token_bounds(
        self, batch: TokenizedBatch, batch_index: int, center_nuc: int, radius_bp: int,
    ) -> tuple[int, int]:
        from eval.clinvar.adapters import _char_level_window_to_token_bounds

        return _char_level_window_to_token_bounds(batch, batch_index, center_nuc, radius_bp)

    def extract_variant_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        from eval.clinvar.adapters import _extract_paired_variant_features

        _ = (ref_alleles, alt_alleles)  # two-tower path uses full ref/alt sequences
        return _extract_paired_variant_features(self, ref_seqs, alt_seqs, variant_offsets)
