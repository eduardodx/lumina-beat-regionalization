"""Caminho M0 da campanha de regionalizacao: backbone congelado, SEM LoRA clinico.

Antes deste guard, `apply_lora` embrulhava os Linear qualquer que fosse o rank e `LoRALinear` fazia alpha/rank,
entao rank=0 estourava com ZeroDivisionError -- nao havia como rodar M0 com so a cabeca treinando.
Precisa de torch: roda no notebook (`pytest tests/test_eval_clinvar_no_lora.py`), nao no Windows.
"""

from __future__ import annotations

import torch

import pytest

from eval.clinvar.lora import (
    LoRALinear,
    apply_lora,
    assert_only_head_trains,
    freeze_backbone_in_eval,
)


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


def test_rank_negativo_e_erro_de_configuracao():
    with pytest.raises(ValueError):
        apply_lora(_TinyBackbone(), rank=-1, alpha=8.0, dropout=0.1)


def test_freeze_backbone_in_eval_resiste_a_model_train():
    """Congelar parametro nao desliga dropout: `model.train()` religaria o backbone."""
    backbone = _TinyBackbone()
    backbone.dropout = torch.nn.Dropout(0.5)
    freeze_backbone_in_eval(backbone)
    assert not backbone.training
    backbone.train()
    assert not backbone.training, "o backbone tem de ficar em eval mesmo depois de train()"

    modelo = torch.nn.Sequential(backbone, torch.nn.Linear(3, 1))
    modelo.train()
    assert not backbone.training, "train() no modelo inteiro tambem nao pode reativar o backbone"


def test_representacao_do_backbone_congelado_e_deterministica():
    backbone = _TinyBackbone()
    backbone.mlp = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.9))
    freeze_backbone_in_eval(backbone)
    x = torch.randn(4, 4)
    primeira, segunda = backbone(x), backbone(x)
    assert torch.allclose(primeira, segunda), "mesma entrada tem de dar a mesma representacao"


def test_assert_only_head_trains_reprova_backbone_treinavel():
    class _Modelo(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = torch.nn.Linear(3, 1)

        def forward(self, x):
            return self.head(self.backbone(x))

    backbone = _TinyBackbone()
    modelo = _Modelo(backbone)
    with pytest.raises(AssertionError):
        assert_only_head_trains(modelo)

    freeze_backbone_in_eval(backbone)
    apply_lora(backbone, rank=0, alpha=8.0, dropout=0.1)
    treinaveis = assert_only_head_trains(modelo)
    assert all(nome.startswith("head.") for nome in treinaveis) and treinaveis


def test_com_backbone_congelado_e_rank_zero_so_a_cabeca_recebe_gradiente():
    backbone = _TinyBackbone()
    freeze_backbone_in_eval(backbone)
    apply_lora(backbone, rank=0, alpha=8.0, dropout=0.1)
    head = torch.nn.Linear(3, 1)

    saida = head(backbone(torch.randn(5, 4))).sum()
    saida.backward()

    assert all(p.grad is None for p in backbone.parameters()), "nenhum gradiente no backbone"
    assert all(p.grad is not None for p in head.parameters()), "a cabeca tem de treinar"
    assert [p for p in backbone.parameters() if p.requires_grad] == []

    # E os pesos do backbone nao podem mudar depois de um passo do otimizador.
    antes = [p.clone() for p in backbone.parameters()]
    torch.optim.SGD(head.parameters(), lr=0.1).step()
    assert all(torch.equal(a, b) for a, b in zip(antes, backbone.parameters()))


def test_rank_positivo_continua_embrulhando():
    backbone = _TinyBackbone()
    summary = apply_lora(backbone, rank=2, alpha=8.0, dropout=0.0)
    assert summary.module_count > 0 and summary.total_params > 0
    assert any(isinstance(m, LoRALinear) for m in backbone.modules())
