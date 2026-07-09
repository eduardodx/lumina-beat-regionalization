import importlib
from dataclasses import dataclass

import torch
import torch.nn as nn
from mamba_ssm import Mamba2

from src.constants import NUM_STRUCTURE_CLASSES, PAD_ID, VOCAB_SIZE

try:
    mamba_ssd_combined = importlib.import_module("mamba_ssm.ops.triton.ssd_combined")
except ImportError:
    mamba_ssd_combined = None


@dataclass
class BiMambaConfig:
    d_model: int = 256
    n_layers: int = 8
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    use_mem_eff_path: bool | None = None


def default_use_mem_eff_path() -> bool:
    return (
        mamba_ssd_combined is not None
        and getattr(mamba_ssd_combined, "causal_conv1d_fwd_function", None) is not None
    )


class BiMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_mem_eff_path: bool | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        resolved_use_mem_eff_path = (
            default_use_mem_eff_path() if use_mem_eff_path is None else use_mem_eff_path
        )

        self.fwd_mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mem_eff_path=resolved_use_mem_eff_path,
        )
        self.bwd_mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mem_eff_path=resolved_use_mem_eff_path,
        )

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


class DNAFoundationBiMamba(nn.Module):
    def __init__(self, cfg: BiMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = PAD_ID

        self.token_emb = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.d_model,
            padding_idx=PAD_ID,
        )

        self.blocks = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    dropout=cfg.dropout,
                    use_mem_eff_path=cfg.use_mem_eff_path,
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


def build_bimamba_model(cfg: BiMambaConfig) -> DNAFoundationBiMamba:
    return DNAFoundationBiMamba(cfg)
