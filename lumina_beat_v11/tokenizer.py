from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from .constants import DNA_VOCAB, ID_TO_DNA, PAD_ID, UNK_ID

UnknownPolicy = Literal["n", "unk", "error"]


def _normalize_sequence(sequence: str) -> str:
    return "".join(sequence.upper().split()).replace("U", "T")


def encode_dna_sequence(
    sequence: str,
    *,
    max_length: int | None = None,
    unknown_policy: UnknownPolicy = "n",
) -> list[int]:
    """Encode a DNA string into Beat-v10 token ids.

    A, C, G, T, and N use the training vocabulary. Other IUPAC ambiguity codes
    default to N so real genomic windows remain easy to pass through inference.
    """

    if unknown_policy not in {"n", "unk", "error"}:
        raise ValueError("unknown_policy must be one of: 'n', 'unk', 'error'.")
    if max_length is not None and max_length <= 0:
        raise ValueError(f"max_length must be positive when set, got {max_length}.")

    ids: list[int] = []
    unknown_id = DNA_VOCAB["N"] if unknown_policy == "n" else UNK_ID
    for base in _normalize_sequence(sequence):
        token_id = DNA_VOCAB.get(base)
        if token_id is not None:
            ids.append(token_id)
        elif unknown_policy == "error":
            raise ValueError(f"Unsupported DNA character {base!r}.")
        else:
            ids.append(unknown_id)

    if max_length is not None:
        ids = ids[:max_length]
    return ids


def batch_encode_dna(
    sequences: Sequence[str],
    *,
    max_length: int | None = None,
    pad_to: int | None = None,
    device: torch.device | str | None = None,
    unknown_policy: UnknownPolicy = "n",
) -> dict[str, torch.Tensor]:
    """Encode and pad DNA strings for Beat-v10 inference."""

    if not sequences:
        raise ValueError("sequences must contain at least one DNA sequence.")
    encoded = [
        encode_dna_sequence(sequence, max_length=max_length, unknown_policy=unknown_policy)
        for sequence in sequences
    ]

    if pad_to is not None:
        if pad_to <= 0:
            raise ValueError(f"pad_to must be positive when set, got {pad_to}.")
        target_len = pad_to
    else:
        target_len = max(1, max(len(ids) for ids in encoded))

    input_ids = torch.full((len(encoded), target_len), PAD_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded), target_len), dtype=torch.long, device=device)
    for row, ids in enumerate(encoded):
        clipped = ids[:target_len]
        if clipped:
            values = torch.tensor(clipped, dtype=torch.long, device=device)
            input_ids[row, : len(clipped)] = values
            attention_mask[row, : len(clipped)] = 1

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def decode_dna_ids(token_ids: Sequence[int] | torch.Tensor, *, skip_special: bool = True) -> str:
    """Decode token ids back to a compact DNA-ish string."""

    if torch.is_tensor(token_ids):
        values = [int(value) for value in token_ids.detach().cpu().flatten().tolist()]
    else:
        values = [int(value) for value in token_ids]

    chars: list[str] = []
    for token_id in values:
        token = ID_TO_DNA.get(token_id, "<unk>")
        if skip_special and token.startswith("<"):
            continue
        chars.append(token)
    return "".join(chars)


def reverse_complement_sequence(sequence: str) -> str:
    complement = str.maketrans("ACGTNacgtnUu", "TGCANtgcanAa")
    return _normalize_sequence(sequence.translate(complement)[::-1])
