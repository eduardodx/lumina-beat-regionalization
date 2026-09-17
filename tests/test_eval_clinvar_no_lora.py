"""Caminho M0 da campanha de regionalizacao: backbone congelado, SEM LoRA clinico.

Antes deste guard, `apply_lora` embrulhava os Linear qualquer que fosse o rank e `LoRALinear` fazia alpha/rank,
entao rank=0 estourava com ZeroDivisionError -- nao havia como rodar M0 com so a cabeca treinando.
Precisa de torch: roda no notebook (`pytest tests/test_eval_clinvar_no_lora.py`), nao no Windows.
"""

from __future__ import annotations

import torch

from eval.clinvar.lora import LoRALinear, apply_lora


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.proj(x))


def test_rank_zero_nao_embrulha_nada_e_nao_estoura():
    backbone = _TinyBackbone()
    summary = apply_lora(backbone, rank=0, alpha=8.0, dropout=0.1)
    assert summary.module_count == 0 and summary.total_params == 0
    assert not any(isinstance(m, LoRALinear) for m in backbone.modules())


def test_rank_zero_preserva_a_saida_do_backbone():
    backbone = _TinyBackbone()
    x = torch.randn(2, 4)
    with torch.no_grad():
        antes = backbone(x).clone()
    apply_lora(backbone, rank=0, alpha=8.0, dropout=0.1)
    with torch.no_grad():
        depois = backbone(x)
    assert torch.allclose(antes, depois), "sem LoRA a representacao tem de ser identica"


def test_com_backbone_congelado_e_rank_zero_so_a_cabeca_recebe_gradiente():
    backbone = _TinyBackbone()
    for p in backbone.parameters():
        p.requires_grad_(False)
    apply_lora(backbone, rank=0, alpha=8.0, dropout=0.1)
    head = torch.nn.Linear(3, 1)

    saida = head(backbone(torch.randn(5, 4))).sum()
    saida.backward()

    assert all(p.grad is None for p in backbone.parameters()), "nenhum gradiente no backbone"
    assert all(p.grad is not None for p in head.parameters()), "a cabeca tem de treinar"
    assert [p for p in backbone.parameters() if p.requires_grad] == []


def test_rank_positivo_continua_embrulhando():
    backbone = _TinyBackbone()
    summary = apply_lora(backbone, rank=2, alpha=8.0, dropout=0.0)
    assert summary.module_count > 0 and summary.total_params > 0
    assert any(isinstance(m, LoRALinear) for m in backbone.modules())
