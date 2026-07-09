import importlib
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from src.constants import COMPLEMENT_TABLE, NUM_STRUCTURE_CLASSES, PAD_ID, VOCAB_SIZE

try:
    _mamba_ssm = importlib.import_module("mamba_ssm")
except ImportError:
    _Mamba3 = None
else:
    _Mamba3 = getattr(_mamba_ssm, "Mamba3", None)
    if _Mamba3 is None:
        try:
            _Mamba3 = importlib.import_module("mamba_ssm.modules.mamba3").Mamba3
        except (AttributeError, ImportError):
            _Mamba3 = None


def _require_mamba3() -> type:
    if _Mamba3 is None:
        raise ImportError(
            "Mamba3 is not available in the installed mamba-ssm package. "
            "Install the latest mamba-ssm from source: "
            "MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation"
        )
    return _Mamba3


@dataclass
class BiMamba3RCConfig:
    d_model: int = 256
    n_layers: int = 8
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 1.0
    chunk_size: int = 16
    is_mimo: bool = True
    mimo_rank: int = 4
    is_outproj_norm: bool = True
    dropout: float = 0.1
    num_region_classes: int = 0


def _resolve_chunk_size(cfg: BiMamba3RCConfig) -> int:
    """Mamba3 MIMO with bf16 needs chunk_size = 64 / mimo_rank."""
    if cfg.is_mimo and cfg.chunk_size == 64:
        return max(1, 64 // cfg.mimo_rank)
    return cfg.chunk_size


class BiMamba3RCBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        chunk_size: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        is_outproj_norm: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_fwd = nn.LayerNorm(d_model)
        self.norm_rc = nn.LayerNorm(d_model)

        Mamba3 = _require_mamba3()

        mamba3_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            is_outproj_norm=is_outproj_norm,
        )

        self.fwd_mixer = Mamba3(**mamba3_kwargs)
        self.bwd_mixer = Mamba3(**mamba3_kwargs)

        self.fuse = nn.Linear(2 * d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_rc: torch.Tensor) -> torch.Tensor:
        h_fwd = self.fwd_mixer(self.norm_fwd(x))

        h_bwd = self.bwd_mixer(self.norm_rc(x_rc))
        h_bwd = torch.flip(h_bwd, dims=[1])

        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        h_out = self.fuse(h_cat)

        return x + self.dropout(h_out)


class DNAFoundationBiMamba3RC(nn.Module):
    def __init__(self, cfg: BiMamba3RCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = PAD_ID

        chunk_size = _resolve_chunk_size(cfg)

        self.token_emb = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.d_model,
            padding_idx=PAD_ID,
        )

        self.blocks = nn.ModuleList(
            [
                BiMamba3RCBlock(
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
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.mlm_head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.mlm_head.weight = self.token_emb.weight

        self.phylo100_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

        self.phylo470_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

        self.structure_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, NUM_STRUCTURE_CLASSES),
        )

        self.region_head: nn.Module | None = None
        if cfg.num_region_classes > 0:
            self.region_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.num_region_classes),
            )

        self.register_buffer(
            "_complement_table",
            torch.tensor(COMPLEMENT_TABLE, dtype=torch.long),
            persistent=False,
        )

    def _reverse_complement_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Complement A<->T, C<->G then reverse. [B, L] -> [B, L]."""
        complement_table = cast(torch.Tensor, self._complement_table)
        return torch.flip(complement_table[input_ids], dims=[-1])

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        rc_ids = self._reverse_complement_ids(input_ids)
        x = self.token_emb(input_ids)
        x_rc = self.token_emb(rc_ids)

        for block in self.blocks:
            x = block(x, x_rc)

        x = self.final_norm(x)
        x = self.dropout(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids)

        outputs: dict[str, torch.Tensor] = {"hidden_states": hidden}
        if return_token_heads:
            outputs["mlm_logits"] = self.mlm_head(hidden)
            outputs["phylo100_pred"] = self.phylo100_head(hidden).squeeze(-1)
            outputs["phylo470_pred"] = self.phylo470_head(hidden).squeeze(-1)
            outputs["structure_logits"] = self.structure_head(hidden)
            if self.region_head is not None:
                outputs["region_logits"] = self.region_head(hidden)
        return outputs


def build_bimamba3_rc_model(cfg: BiMamba3RCConfig) -> DNAFoundationBiMamba3RC:
    return DNAFoundationBiMamba3RC(cfg)
