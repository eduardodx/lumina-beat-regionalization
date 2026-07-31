"""Lumina — inference package.

Self-contained inference/review release for the Lumina DNA foundation model (bidirectional Mamba-3
hourglass, ~52M params, masked-LM + per-position genomic heads). The architecture is vendored verbatim
from the training repo at source revision ``f2f7f9d``; only the training loss/data/optimizer machinery
is omitted. Load a checkpoint and run a forward pass::

    import torch
    from lumina import batch_encode_dna, load_model_from_checkpoint

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model_from_checkpoint(
        "s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt",
        device=device,
    )
    batch = batch_encode_dna(["ACGT...N"], pad_to=4096, device=device)
    with torch.inference_mode():
        outputs = model(**batch)   # dict of hidden states + head logits

The Mamba-3 block is loaded lazily via ``require_mamba3()``: it imports the real CUDA ``mamba_ssm`` if
present, else falls back to the pure-PyTorch ``mamba3`` drop-in (CPU/MPS). No MIMO/TileLang kernel is
needed — the released checkpoint is SISO (``is_mimo=False``).
"""

from __future__ import annotations

from lumina.checkpoint import (
    build_model,
    config_from_checkpoint,
    default_config_dict,
    download_s3_checkpoint,
    extract_model_state_dict,
    is_s3_uri,
    load_model_from_checkpoint,
    normalize_config_dict,
    parse_s3_uri,
    strip_state_dict_prefixes,
    torch_load_checkpoint,
)
from lumina.constants import (
    COMPLEMENT_TABLE,
    DNA_VOCAB,
    MASK_ID,
    NUM_COUNTERFACTUAL_EFFECT_CLASSES,
    NUM_REGION_CLASSES,
    PAD_ID,
    SNV_ALT_TO_INDEX,
    SNV_BASES,
    UNK_ID,
    VOCAB_SIZE,
)
from lumina.models.mamba_runtime import require_mamba3
from lumina.models.model import LuminaConfig, LuminaModel, build_lumina_model
from lumina.tokenizer import (
    ID_TO_DNA,
    batch_encode_dna,
    decode_dna_ids,
    encode_dna_sequence,
    reverse_complement_sequence,
)

__all__ = [
    # constants
    "COMPLEMENT_TABLE",
    "DNA_VOCAB",
    "ID_TO_DNA",
    "MASK_ID",
    "NUM_COUNTERFACTUAL_EFFECT_CLASSES",
    "NUM_REGION_CLASSES",
    "PAD_ID",
    "SNV_ALT_TO_INDEX",
    "SNV_BASES",
    "UNK_ID",
    "VOCAB_SIZE",
    # model
    "LuminaConfig",
    "LuminaModel",
    # tokenizer
    "batch_encode_dna",
    "build_lumina_model",
    # checkpoint / loading
    "build_model",
    "config_from_checkpoint",
    "decode_dna_ids",
    "default_config_dict",
    "download_s3_checkpoint",
    "encode_dna_sequence",
    "extract_model_state_dict",
    "is_s3_uri",
    "load_model_from_checkpoint",
    "normalize_config_dict",
    "parse_s3_uri",
    "require_mamba3",
    "reverse_complement_sequence",
    "strip_state_dict_prefixes",
    "torch_load_checkpoint",
]
