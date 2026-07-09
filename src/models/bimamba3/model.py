import importlib
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.constants import NUM_STRUCTURE_CLASSES, PAD_ID, VOCAB_SIZE

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
class BiMamba3Config:
    d_model: int = 256
    n_layers: int = 8
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 0.5
    chunk_size: int = 64
    is_mimo: bool = False
    mimo_rank: int = 4
    is_outproj_norm: bool = False
    dropout: float = 0.1


def _resolve_chunk_size(cfg: BiMamba3Config) -> int:
    """Mamba3 MIMO with bf16 needs chunk_size = 64 / mimo_rank."""
    if cfg.is_mimo and cfg.chunk_size == 64:
        return max(1, 64 // cfg.mimo_rank)
    return cfg.chunk_size


class BiMamba3Block(nn.Module):
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
        self.norm = nn.LayerNorm(d_model)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)

        h_fwd = self.fwd_mixer(h)

        h_rev = torch.flip(h, dims=[1])
        h_bwd = self.bwd_mixer(h_rev)
        h_bwd = torch.flip(h_bwd, dims=[1])

        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        h_out = self.fuse(h_cat)

        return x + self.dropout(h_out)


class DNAFoundationBiMamba3(nn.Module):
    def __init__(self, cfg: BiMamba3Config) -> None:
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
                BiMamba3Block(
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

        self.global_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = self.dropout(x)
        return x

    def pooled_embedding(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return self.global_proj(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = True,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids)

        outputs: dict[str, torch.Tensor] = {"hidden_states": hidden}
        if return_token_heads:
            outputs["mlm_logits"] = self.mlm_head(hidden)
            outputs["phylo100_pred"] = self.phylo100_head(hidden).squeeze(-1)
            outputs["phylo470_pred"] = self.phylo470_head(hidden).squeeze(-1)
            outputs["structure_logits"] = self.structure_head(hidden)
        if return_sequence_embedding:
            outputs["sequence_embedding"] = self.pooled_embedding(hidden, attention_mask)
        return outputs


def build_bimamba3_model(cfg: BiMamba3Config) -> DNAFoundationBiMamba3:
    return DNAFoundationBiMamba3(cfg)
