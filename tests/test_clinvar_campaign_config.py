"""A configuracao M0/MR tem de ser imposta pela config, nao combinada a mao em quatro flags soltas.

Rodar o M0 com os defaults (`lora_rank=4`, `freeze_backbone=False`, sem congelar o backbone) produziria um sistema
diferente com o mesmo nome. Aqui a config falha alto em vez de corrigir em silencio.
Precisa de torch (a config importa `src.precision`): roda no notebook.
"""

from __future__ import annotations

import pytest

from eval.clinvar.config import FineTuneConfig

CONFLITOS = [
    {"lora_rank": 4},
    {"freeze_backbone": False},
    {"fusion_mode": "static_lora"},
    {"freeze_backbone_for_fusion": True},
    {"init_finetuned_checkpoint_path": "/tmp/checkpoint_clinico_antigo.pt"},
    {"fusion_adapter_paths": ["/tmp/adapter.pt"]},
    {"fusion_adapter_names": ["abraom"]},
]


def test_for_campaign_impoe_os_invariantes():
    cfg = FineTuneConfig.for_campaign("M0")
    assert cfg.campaign_system == "M0"
    assert cfg.lora_rank == 0
    assert cfg.freeze_backbone is True
    assert cfg.fusion_mode == "none"
    assert cfg.freeze_backbone_for_fusion is False
    assert cfg.init_finetuned_checkpoint_path is None
    assert cfg.fusion_adapter_paths == [] and cfg.fusion_adapter_names == []


def test_for_campaign_aceita_o_que_nao_e_invariante():
    cfg = FineTuneConfig.for_campaign("MR", seed=7, batch_size=8, max_epochs=1)
    assert (cfg.seed, cfg.batch_size, cfg.max_epochs) == (7, 8, 1)
    assert cfg.campaign_system == "MR" and cfg.lora_rank == 0


@pytest.mark.parametrize("conflito", CONFLITOS, ids=lambda c: next(iter(c)))
def test_cada_conflito_falha_e_nomeia_o_atributo(conflito):
    with pytest.raises(ValueError) as erro:
        FineTuneConfig.for_campaign("M0", **conflito)
    mensagem = str(erro.value)
    atributo = next(iter(conflito))
    assert "campaign_system" in mensagem
    assert atributo.split("_")[0] in mensagem, mensagem


def test_config_montada_a_mao_tambem_e_verificada():
    """Nao basta usar a fabrica: quem montar FineTuneConfig direto com campaign_system tambem e cobrado."""
    with pytest.raises(ValueError):
        FineTuneConfig(campaign_system="M0")  # defaults tem lora_rank=4 e freeze_backbone=False


def test_sem_campaign_system_nada_muda():
    """Compatibilidade: as configs antigas continuam validas."""
    cfg = FineTuneConfig()
    assert cfg.campaign_system == "" and cfg.lora_rank == 4 and cfg.freeze_backbone is False


def test_mensagem_de_erro_explica_o_porque():
    with pytest.raises(ValueError) as erro:
        FineTuneConfig.for_campaign("M0", lora_rank=4, freeze_backbone=False)
    mensagem = str(erro.value)
    assert "sem LoRA clinico" in mensagem and "backbone congelado" in mensagem, mensagem
