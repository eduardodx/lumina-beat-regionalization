"""Gradient-enabled adapter for the R03 Lumina backbone (``lumina-inference`` package).

Fase 1 da campanha de regionalizacao R03: expoe o backbone R03 (LUM-20260719-001-R03) atraves do
``FineTuneAdapter`` protocol usado pelo harness de treino/eval do ClinVar. Espelha o
``FineTuneBeatV11Adapter`` -- mas E ESPECIFICO PRO R03, nao uma copia. Diferencas conscientes vs v11:

  * **Loader/pacote:** carrega via ``from lumina import load_model_from_checkpoint, batch_encode_dna``
    (o pacote inference-only do repo lumina-inference), nao o ``lumina_beat_v11`` vendorizado nem o
    registry ``src/models``. O proprio checkpoint dita a arquitetura (strict=True, 630 tensores).
  * **d_full = 448** (R03: d_model=384 + d_pure=64), nao os 320 do v11. Lido de ``full_hidden_dim``.
  * **Tokenizacao:** usa o ``batch_encode_dna`` NATIVO do R03 (char-level A/C/G/T/N, right-padded,
    1 token por base, sem prefixo) -- compativel com os helpers char-level do harness.
  * **encode() sem ``variant_edit_mask``:** le a representacao da janela de REFERENCIA limpa (o
    variant-residual boost do R03 fica desligado); o efeito da variante vem do two-tower ref/alt,
    igual ao v11. As janelas ref/alt tem tamanho fixo (context_size) => sem padding no batch.
  * **SEM o shim de tilelang do v11:** aquele era workaround pra uma imagem SageMaker cuja
    tilelang/tvm quebrava no import. O R03 roda sob ``scripts/setup-gpu.sh`` (tilelang 0.1.11 fixado,
    mamba_ssm.Mamba3 construido do source). Envenenar ``tilelang`` aqui poderia MASCARAR um env
    quebrado, caindo num kernel de fallback que nao e o SISO contra o qual o R03 foi treinado.
    => M0 deve rodar sob ``source scripts/env.sh`` (NUNCA ``uv sync`` no host GPU).

O backbone fica CONGELADO (§4.1 do Eduardo): quem treina e a LoRA (aplicada pelo train.py, exclusion-
based, ja verificada no R03) + a classification head. Este adapter so expoe as primitivas.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)

TokenizedBatch = dict[str, Any]


class FineTuneR03Adapter:
    """Adapter expondo o backbone R03 (lumina-inference) pelo ``FineTuneAdapter`` protocol."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Import tardio: so puxa o lumina (e mamba_ssm/tilelang, sob setup-gpu.sh) quando de fato usado.
        from lumina import batch_encode_dna, load_model_from_checkpoint
        from lumina.constants import DNA_VOCAB, PAD_ID, UNK_ID

        self._batch_encode_dna = batch_encode_dna
        self._vocab = DNA_VOCAB
        self._pad_id = int(PAD_ID)
        self._unk_id = int(UNK_ID)
        self._device = device
        self._model: Any = load_model_from_checkpoint(
            checkpoint_path, device=device, dtype=dtype, strict=True,
        )
        # full_hidden_dim == cfg.d_full == d_model(384) + d_pure(64) == 448 no R03.
        self._d_model = int(getattr(self._model, "full_hidden_dim", None) or self._model.cfg.d_full)
        log.info("Loaded R03 Lumina backbone (d_full=%d) from %s", self._d_model, checkpoint_path)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._model

    def _encode_seq(self, seq: str) -> list[int]:
        return [self._vocab.get(b.upper(), self._unk_id) for b in seq]

    def tokenize(self, sequences: list[str]) -> TokenizedBatch:
        # batch_encode_dna nativo do R03: char-level A/C/G/T/N, right-padded ao maior do batch, 1
        # token/base, sem prefixo -> attention_mask.shape[0] == n_tokens == n_nucleotideos (os helpers
        # char-level do harness dependem desse 1:1). Janelas ref/alt tem tamanho fixo => sem padding.
        max_len = max(len(s) for s in sequences)
        return self._batch_encode_dna(sequences, pad_to=max_len, device=self._device)

    def forward_hidden_states(self, batch: TokenizedBatch) -> Tensor:
        # R03 encode() -> {"last_hidden_state": [B,L,448], "mid_hidden_state": [B,L/4,384]}.
        # Sem variant_edit_mask (default None) -> representacao de referencia limpa. Grad flui pela LoRA.
        encoded = self._model.encode(batch["input_ids"])
        hidden = encoded.get("last_hidden_state") if isinstance(encoded, dict) else None
        if not isinstance(hidden, torch.Tensor):
            keys = list(encoded) if isinstance(encoded, dict) else type(encoded).__name__
            raise RuntimeError(f"R03 encode() nao retornou 'last_hidden_state' (got {keys}).")
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

        _ = (ref_alleles, alt_alleles)  # two-tower usa as sequencias ref/alt completas
        return _extract_paired_variant_features(self, ref_seqs, alt_seqs, variant_offsets)

    @torch.no_grad()
    def extract_native_pathogenicity_features(
        self,
        ref_seqs: list[str],
        variant_offsets: list[int],
        alt_alleles: list[str],
    ) -> dict[str, list[float]]:
        """Conservacao + missense-severity das heads nativas FROZEN do R03, lidas no sitio da variante
        na janela de referencia (backbone-only, ortogonal a AF do ABraOM). Espelha o v11; as heads do
        R03 (conservation_scalar_head sempre; missense_severity_head condicional ao config) batem."""
        from eval.clinvar.adapters import _nuc_offset_to_token_index
        from lumina.constants import SNV_BASES  # (A, C, G, T) -- eixo alt da missense head

        base_to_idx = {b: i for i, b in enumerate(SNV_BASES)}
        model = self._model

        batch = self.tokenize(ref_seqs)
        hidden = self.forward_hidden_states(batch)  # [B, L, 448]
        conservation = model.conservation_scalar_head(hidden)  # [B, L, num_conservation_targets]
        missense = (
            model.missense_severity_head(hidden)  # [B, L, len(SNV_BASES)]
            if getattr(model, "missense_severity_head", None) is not None
            else None
        )

        device = hidden.device
        n = hidden.shape[0]
        rows = torch.arange(n, device=device)
        token_idx = torch.tensor(
            [_nuc_offset_to_token_index(self, batch, i, off) for i, off in enumerate(variant_offsets)],
            dtype=torch.long, device=device,
        )

        cons_at = conservation[rows, token_idx].float()  # [B, num_conservation_targets]
        names = ("phylo100", "zoo241", "phylo470")
        out: dict[str, list[float]] = {
            names[j]: cons_at[:, j].tolist()
            for j in range(min(cons_at.shape[-1], len(names)))
        }
        if missense is not None:
            alt_idx = torch.tensor(
                [base_to_idx.get((a or "").upper(), 0) for a in alt_alleles],
                dtype=torch.long, device=device,
            )
            miss_at = missense[rows, token_idx].float()  # [B, len(SNV_BASES)]
            out["missense_severity"] = miss_at[rows, alt_idx].tolist()
        return out
