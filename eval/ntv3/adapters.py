from __future__ import annotations

import torch
from torch import Tensor

from src.checkpoints import load_lumina_backbone_from_checkpoint
from src.constants import DNA_VOCAB, PAD_ID, UNK_ID
from src.precision import PrecisionPolicy


def _int_config_value(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _hidden_dim_from_backbone(
    backbone: torch.nn.Module,
    *,
    model_config: dict[str, object],
    checkpoint_config: dict[str, object],
) -> int:
    full_hidden_dim = getattr(backbone, "full_hidden_dim", None)
    if isinstance(full_hidden_dim, bool):
        return int(full_hidden_dim)
    if isinstance(full_hidden_dim, int):
        return full_hidden_dim

    d_model = _int_config_value(model_config, "d_model", _int_config_value(checkpoint_config, "d_model", 256))
    d_pure = _int_config_value(model_config, "d_pure", 0)
    return d_model + d_pure


def tokenize_dna_sequences(sequences: list[str], *, max_length: int) -> dict[str, Tensor]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    input_ids = torch.full((len(sequences), max_length), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        encoded = [DNA_VOCAB.get(base.upper(), UNK_ID) for base in sequence[:max_length]]
        if not encoded:
            continue
        input_ids[row_index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        attention_mask[row_index, : len(encoded)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class LuminaNTv3Adapter:
    def __init__(
        self,
        *,
        model_key: str,
        checkpoint_path: str,
        device: torch.device,
        precision: PrecisionPolicy | None = None,
    ) -> None:
        backbone, model_config, checkpoint_config = load_lumina_backbone_from_checkpoint(
            checkpoint_path,
            requested_model_key=model_key,
            precision=precision,
            device=device,
        )
        self.backbone = backbone
        self.model_key = model_key
        self.device = device
        self.model_config = model_config
        self.checkpoint_config = checkpoint_config
        self.d_model = _hidden_dim_from_backbone(
            backbone,
            model_config=model_config,
            checkpoint_config=checkpoint_config,
        )
        self.decoder_dim = _int_config_value(model_config, "decoder_dim", self.d_model)
        self.max_length = _int_config_value(checkpoint_config, "seq_len", 4096)

    def tokenize(self, sequences: list[str]) -> dict[str, Tensor]:
        return tokenize_dna_sequences(sequences, max_length=self.max_length)

    def forward_hidden_states(self, batch: dict[str, Tensor]) -> Tensor:
        input_ids = batch["input_ids"].to(self.device)
        encode = getattr(self.backbone, "encode", None)
        if callable(encode):
            encoded = encode(input_ids)
            if isinstance(encoded, torch.Tensor):
                return encoded
            if isinstance(encoded, dict) and isinstance(encoded.get("hidden_states"), torch.Tensor):
                return encoded["hidden_states"]

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=batch.get("attention_mask"),
            return_token_heads=False,
        )
        hidden_states = outputs.get("hidden_states") if isinstance(outputs, dict) else None
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError("Lumina backbone must return a tensor under 'hidden_states'.")
        return hidden_states


def build_ntv3_adapter(
    *,
    model_family: str,
    model_version: str,
    checkpoint_path: str,
    device: torch.device,
    precision: PrecisionPolicy | None = None,
) -> LuminaNTv3Adapter:
    if model_family != "lumina":
        raise ValueError(f"Unsupported model_family for NTv3 benchmark: {model_family!r}")
    return LuminaNTv3Adapter(
        model_key=model_version,
        checkpoint_path=checkpoint_path,
        device=device,
        precision=precision,
    )
