"""Prova o nucleo em torch do adapter: loss diferenciavel, invariantes de congelamento e checkpoint estrito.

Precisa de torch (roda no notebook; no Windows os testes PULAM, e o runner conta separado).
    PYTHONPATH=. REQUIRE_NO_SKIP=1 python3 tests/test_adapter_treino.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


try:
    import torch
    from torch import nn

    from eval.adapter import mlm, treino
    from eval.clinvar.lora import apply_lora
    TORCH = True
except ImportError as exc:  # pragma: no cover - no Windows
    TORCH = False
    MOTIVO = str(exc)


def _exige_torch():
    if not TORCH:
        raise Skip(f"sem torch: {MOTIVO}")


def _classes():
    """Definidas aqui dentro: no Windows nao ha torch, e `nn` nao existiria no escopo do modulo."""

    class ModeloDeBrinquedo(nn.Module):
        """Mesma superficie que o treinador usa do R03: `encode` -> last_hidden_state, e `mlm_head`."""

        def __init__(self, d: int = 8):
            super().__init__()
            self.emb = nn.Embedding(8, d)
            self.proj = nn.Linear(d, d)
            self.mlm_head = nn.Linear(d, 4)
            self.gnomad_af_head = nn.Linear(d, 4)  # cabeca nativa: nao pode ser embrulhada nem treinar

        def encode(self, input_ids):
            return {"last_hidden_state": self.proj(self.emb(input_ids))}

    class AdapterDeBrinquedo:
        def __init__(self, modelo):
            self._modelo = modelo

        @property
        def backbone(self):
            return self._modelo

        def forward_hidden_states(self, batch):
            return self._modelo.encode(batch["input_ids"])["last_hidden_state"]

    return ModeloDeBrinquedo, AdapterDeBrinquedo


def _monta(rank: int = 2, use_rslora: bool = True):
    ModeloDeBrinquedo, AdapterDeBrinquedo = _classes()
    modelo = ModeloDeBrinquedo()
    treino.congelar_tudo(modelo)
    resumo = apply_lora(modelo, rank=rank, alpha=4, dropout=0.0, use_rslora=use_rslora)
    return AdapterDeBrinquedo(modelo), modelo, resumo


def _exemplos(n: int = 2):
    alt_seq = "ACGT" * 4  # 16 bp
    saida = []
    for indice in range(n):
        saida.append(mlm.montar_exemplo(
            alt_seq, [(4, 7, mlm.TIPO_VARIANTE), (10, 13, mlm.TIPO_REFERENCIA)],
            variant_id=f"v{indice}", fonte="abraom" if indice % 2 else "global", focal_index=5))
    return saida


def test_apply_lora_nao_embrulha_a_cabeca_mlm():
    """`mlm_head` esta em _EXCLUDE_PATTERNS: ela fica congelada e no grafo, nao vira parametro do adapter."""
    _exige_torch()
    _, modelo, resumo = _monta()
    assert not any("mlm_head" in nome for nome in resumo.module_names), resumo.module_names
    assert not any("gnomad_af_head" in nome for nome in resumo.module_names), resumo.module_names
    assert not modelo.mlm_head.weight.requires_grad


def test_so_o_adapter_fica_treinavel():
    _exige_torch()
    _, modelo, _ = _monta()
    nomes = treino.assert_so_o_adapter_treina(modelo)
    assert nomes and all(treino.e_do_adapter(n) for n in nomes), nomes


def test_layernorm_descongelado_por_engano_e_pego():
    """Contar parametros nao bastaria: um LayerNorm solto mudaria o backbone que a campanha promete fixo."""
    _exige_torch()
    _, modelo, _ = _monta()
    modelo.proj.base.bias.requires_grad = True
    try:
        treino.assert_so_o_adapter_treina(modelo)
    except RuntimeError as exc:
        assert "fora do adapter" in str(exc)
        return
    raise AssertionError("parametro fora do adapter tinha de reprovar")


def test_logits_tem_quatro_classes():
    _exige_torch()
    adapter, _, _ = _monta()
    lote = treino.montar_lote(_exemplos())
    logits = treino.logits_mlm(adapter, lote.input_ids)
    assert logits.shape[0] == 2 and logits.shape[1] == 16
    assert logits.shape[-1] == len(mlm.SNV_BASES) == 4, logits.shape
    assert int(lote.alvo.max()) < 4 and int(lote.alvo.min()) >= 0


def test_loss_em_tensores_bate_com_o_nucleo_sem_torch():
    """A igualdade que liga os dois mundos: se as formulas divergirem, isto quebra."""
    _exige_torch()
    adapter, _, _ = _monta()
    lote = treino.montar_lote(_exemplos())
    logits = treino.logits_mlm(adapter, lote.input_ids)
    perda, decomposicao = treino.perda_do_lote(logits, lote, mlm.PESOS_INICIAIS)
    referencia = mlm.perda_ponderada(decomposicao, mlm.PESOS_INICIAIS)
    medido = float(perda.detach())  # sem detach, o torch avisa sobre converter tensor do grafo
    assert abs(medido - referencia) < 1e-5, (medido, referencia)


def test_loss_e_diferenciavel_e_o_relatorio_nao_segura_o_grafo():
    _exige_torch()
    adapter, modelo, _ = _monta()
    lote = treino.montar_lote(_exemplos())
    perda, decomposicao = treino.perda_do_lote(treino.logits_mlm(adapter, lote.input_ids), lote,
                                               mlm.PESOS_INICIAIS)
    assert perda.requires_grad, "a loss de treino tem de estar no grafo"
    assert all(isinstance(v["media"], float) for v in decomposicao.values())
    perda.backward()
    com_gradiente = [n for n, p in modelo.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert com_gradiente, "nenhum parametro recebeu sinal"
    assert all(treino.e_do_adapter(n) for n in com_gradiente), com_gradiente


def test_no_primeiro_passo_lora_a_pode_ter_gradiente_zero():
    """`lora_b` nasce em zeros, entao o gradiente de `lora_a` e zero por CONSTRUCAO no primeiro passo.

    Exigir gradiente nao nulo em todo tensor do adapter reprovaria um treino correto. O que se exige e que o
    adapter receba sinal e MUDE.
    """
    _exige_torch()
    adapter, modelo, _ = _monta()
    for nome, parametro in modelo.named_parameters():
        if nome.endswith("lora_b"):
            assert float(parametro.detach().abs().sum()) == 0.0, nome
    lote = treino.montar_lote(_exemplos())
    perda, _ = treino.perda_do_lote(treino.logits_mlm(adapter, lote.input_ids), lote, mlm.PESOS_INICIAIS)
    perda.backward()
    gradiente_de_a = [float(p.grad.abs().sum()) for n, p in modelo.named_parameters()
                      if n.endswith("lora_a") and p.grad is not None]
    gradiente_de_b = [float(p.grad.abs().sum()) for n, p in modelo.named_parameters()
                      if n.endswith("lora_b") and p.grad is not None]
    assert gradiente_de_a and all(v == 0.0 for v in gradiente_de_a), gradiente_de_a
    assert any(v > 0 for v in gradiente_de_b), gradiente_de_b


def test_um_passo_muda_o_adapter_e_deixa_o_congelado_identico():
    _exige_torch()
    adapter, modelo, _ = _monta()
    antes = treino.impressao_dos_congelados(modelo)
    parametros = list(treino.parametros_do_adapter(modelo).values())
    copia = [p.detach().clone() for p in parametros]
    otimizador = torch.optim.AdamW(parametros, lr=1e-2)

    lote = treino.montar_lote(_exemplos())
    perda, _ = treino.perda_do_lote(treino.logits_mlm(adapter, lote.input_ids), lote, mlm.PESOS_INICIAIS)
    perda.backward()
    otimizador.step()

    assert treino.impressao_dos_congelados(modelo) == antes, "o backbone congelado MUDOU"
    mudou = any(not torch.equal(p.detach(), c) for p, c in zip(parametros, copia))
    assert mudou, "o adapter nao mudou apos o passo"


def test_otimizador_recebe_cada_parametro_uma_vez():
    _exige_torch()
    _, modelo, _ = _monta()
    parametros = treino.parametros_do_adapter(modelo)
    identidades = [id(p) for p in parametros.values()]
    assert len(identidades) == len(set(identidades)), "parametro repetido no otimizador"


def _identidades():
    return {"r03": "f2983560", "plano": "c99e5dae"}


def test_salvar_e_recarregar_reproduz_as_predicoes():
    _exige_torch()
    adapter, modelo, resumo = _monta()
    lote = treino.montar_lote(_exemplos())
    # Mexe no adapter para o checkpoint nao ser a inicializacao.
    with torch.no_grad():
        for nome, parametro in modelo.named_parameters():
            if nome.endswith("lora_b"):
                parametro.add_(torch.randn_like(parametro) * 0.1)
    modelo.eval()
    antes = treino.logits_mlm(adapter, lote.input_ids).detach().clone()

    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "adapter.pt"
        treino.salvar_adapter(caminho, backbone=modelo, resumo_lora=resumo, config={"rank": 2},
                              identidades=_identidades(), metricas={}, passo=7)
        with torch.no_grad():
            for nome, parametro in modelo.named_parameters():
                if treino.e_do_adapter(nome):
                    parametro.zero_()
        assert not torch.allclose(treino.logits_mlm(adapter, lote.input_ids), antes), "zerar tinha de mudar"

        carga = treino.carregar_adapter(caminho, modelo)
        assert carga["passo"] == 7 and carga["identidades"] == _identidades()
        depois = treino.logits_mlm(adapter, lote.input_ids)
    assert torch.allclose(depois, antes, atol=1e-6), "recarregar nao reproduziu as predicoes"


def test_checkpoint_com_chave_removida_reprova():
    """O carregador antigo usava strict=False e chave AUSENTE passava em silencio."""
    _exige_torch()
    _, modelo, resumo = _monta()
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "adapter.pt"
        treino.salvar_adapter(caminho, backbone=modelo, resumo_lora=resumo, config={},
                              identidades=_identidades(), metricas={}, passo=1)
        carga = torch.load(caminho, map_location="cpu", weights_only=False)
        removida = sorted(carga["estado_do_adapter"])[0]
        del carga["estado_do_adapter"][removida]
        torch.save(carga, caminho)
        try:
            treino.carregar_adapter(caminho, modelo)
        except RuntimeError as exc:
            assert "faltando" in str(exc), str(exc)
            return
    raise AssertionError("chave ausente tinha de reprovar, nao ficar na inicializacao")


def test_checkpoint_com_chave_sobrando_reprova():
    _exige_torch()
    _, modelo, resumo = _monta()
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "adapter.pt"
        treino.salvar_adapter(caminho, backbone=modelo, resumo_lora=resumo, config={},
                              identidades=_identidades(), metricas={}, passo=1)
        carga = torch.load(caminho, map_location="cpu", weights_only=False)
        carga["estado_do_adapter"]["intruso.lora_a"] = torch.zeros(2, 2)
        torch.save(carga, caminho)
        try:
            treino.carregar_adapter(caminho, modelo)
        except RuntimeError as exc:
            assert "sobrando" in str(exc), str(exc)
            return
    raise AssertionError("chave extra tinha de reprovar")


def test_nao_salva_se_algo_fora_do_adapter_estiver_treinavel():
    _exige_torch()
    _, modelo, resumo = _monta()
    modelo.mlm_head.weight.requires_grad = True
    with tempfile.TemporaryDirectory() as tmp:
        try:
            treino.salvar_adapter(Path(tmp) / "a.pt", backbone=modelo, resumo_lora=resumo, config={},
                                  identidades={}, metricas={}, passo=0)
        except RuntimeError as exc:
            assert "fora do adapter" in str(exc)
            return
    raise AssertionError("checkpoint com peso extra treinavel tinha de reprovar")


def test_formato_nao_e_o_do_adapter_de_frequencia():
    _exige_torch()
    assert treino.FORMATO != "abraom_frequency_adapter_v1"
    assert "mlm" in treino.FORMATO


def test_decomposicao_por_fonte_separa_as_metades():
    _exige_torch()
    adapter, _, _ = _monta()
    lote = treino.montar_lote(_exemplos(n=2))
    logits = treino.logits_mlm(adapter, lote.input_ids)
    por_fonte = treino.decomposicao_por_fonte(logits, lote)
    assert set(por_fonte) == {"global", "abraom"}, set(por_fonte)
    for dados in por_fonte.values():
        assert dados[mlm.CATEGORIA_FOCAL]["posicoes"] == 1


def test_lote_com_janelas_de_larguras_diferentes_reprova():
    _exige_torch()
    curto = mlm.montar_exemplo("ACGT" * 2, [(0, 3, mlm.TIPO_VARIANTE)], variant_id="c", fonte="global",
                               focal_index=1)
    longo = _exemplos(1)[0]
    try:
        treino.montar_lote([curto, longo])
    except ValueError:
        return
    raise AssertionError("larguras diferentes tinham de reprovar")


def test_lora_nao_embrulha_linear_dentro_de_multihead_attention():
    """MEDIDO no R03 em 22/09: `nn.MultiheadAttention` passa `out_proj.weight` para
    `F.multi_head_attention_forward` em vez de chamar o modulo. Embrulhar aquele Linear e INERTE: o delta nunca e
    aplicado, os parametros ficam fora do grafo (`grad is None`) e mesmo assim o weight decay os move.
    """
    _exige_torch()

    class ComMHA(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.emb = nn.Embedding(8, d)
            self.mha = nn.MultiheadAttention(d, 2, batch_first=True)
            self.proj = nn.Linear(d, d)
            self.mlm_head = nn.Linear(d, 4)

        def encode(self, input_ids):
            x = self.emb(input_ids)
            atendido, _ = self.mha(x, x, x, need_weights=False)
            return {"last_hidden_state": self.proj(atendido)}

    modelo = ComMHA()
    treino.congelar_tudo(modelo)
    resumo = apply_lora(modelo, rank=2, alpha=4, dropout=0.0, use_rslora=True)
    assert "proj" in resumo.module_names, resumo.module_names
    assert not any(n.startswith("mha.") for n in resumo.module_names), resumo.module_names
    assert any("out_proj" in n for n in resumo.modulos_inertes_ignorados), resumo.modulos_inertes_ignorados

    # E o que sobrou entra todo no grafo: nenhum `grad is None`.
    class Env:
        def __init__(self, m):
            self._m = m

        @property
        def backbone(self):
            return self._m

        def forward_hidden_states(self, batch):
            return self._m.encode(batch["input_ids"])["last_hidden_state"]

    exemplo = mlm.montar_exemplo("ACGT" * 4, [(4, 7, mlm.TIPO_VARIANTE)], variant_id="v", fonte="global",
                                 focal_index=5)
    lote = treino.montar_lote([exemplo])
    perda, _ = treino.perda_do_lote(treino.logits_mlm(Env(modelo), lote.input_ids), lote, mlm.PESOS_INICIAIS)
    perda.backward()
    estado = treino.gradientes_do_adapter(modelo)
    assert not estado["sem_gradiente"], estado["sem_gradiente"]


def test_gradientes_separam_ausente_de_zero():
    _exige_torch()
    _, modelo, _ = _monta()
    lote = treino.montar_lote(_exemplos())
    adapter, modelo2, _ = _monta()
    perda, _ = treino.perda_do_lote(treino.logits_mlm(adapter, lote.input_ids), lote, mlm.PESOS_INICIAIS)
    perda.backward()
    estado = treino.gradientes_do_adapter(modelo2)
    assert "com_gradiente_zero" in estado and "sem_gradiente" in estado
    # lora_a no primeiro passo entra no grafo com gradiente ZERO, nao ausente.
    assert any(n.endswith("lora_a") for n in estado["com_gradiente_zero"]), estado["com_gradiente_zero"][:5]


def test_acumulacao_equivale_a_um_lote_unico():
    """O requisito da revisao: microlotes tem QUANTIDADES DIFERENTES de posicoes mascaradas.

    Dividir cada microlote pelo numero de microlotes daria peso igual a um com 3 posicoes e a outro com 30. O
    certo e somar `w_i * CE_i` em cada um e dividir os gradientes pelo peso TOTAL antes do passo.
    """
    _exige_torch()
    variados = [
        mlm.montar_exemplo("ACGT" * 4, [(4, 7, mlm.TIPO_VARIANTE)], variant_id="a", fonte="global",
                           focal_index=5),
        mlm.montar_exemplo("ACGT" * 4, [(4, 7, mlm.TIPO_VARIANTE), (9, 14, mlm.TIPO_REFERENCIA)],
                           variant_id="b", fonte="abraom", focal_index=5),
    ]
    assert len(variados[0].posicoes) != len(variados[1].posicoes), "o teste precisa de tamanhos diferentes"

    # (1) lote unico
    adapter, modelo, _ = _monta()
    torch.manual_seed(0)
    lote = treino.montar_lote(variados)
    perda, _ = treino.perda_do_lote(treino.logits_mlm(adapter, lote.input_ids), lote, mlm.PESOS_INICIAIS)
    perda.backward()
    referencia = {n: p.grad.detach().clone() for n, p in modelo.named_parameters() if p.grad is not None}

    # (2) acumulado em microlotes de 1
    adapter2, modelo2, _ = _monta()
    modelo2.load_state_dict(modelo.state_dict())
    peso_total = 0.0
    for exemplo in variados:
        micro = treino.montar_lote([exemplo])
        soma, peso, _ = treino.perda_somada_do_lote(treino.logits_mlm(adapter2, micro.input_ids), micro,
                                                    mlm.PESOS_INICIAIS)
        soma.backward()
        peso_total += peso
    treino.dividir_gradientes(list(treino.parametros_do_adapter(modelo2).values()), peso_total)

    obtido = {n: p.grad for n, p in modelo2.named_parameters() if p.grad is not None}
    assert set(obtido) == set(referencia), (sorted(obtido), sorted(referencia))
    for nome in referencia:
        assert torch.allclose(obtido[nome], referencia[nome], atol=1e-6), nome


def test_dividir_gradientes_recusa_divisor_invalido():
    _exige_torch()
    _, modelo, _ = _monta()
    try:
        treino.dividir_gradientes(list(treino.parametros_do_adapter(modelo).values()), 0.0)
    except ValueError:
        return
    raise AssertionError("divisor zero tinha de falhar")


def test_perda_somada_e_a_media_vezes_o_peso_total():
    _exige_torch()
    adapter, _, _ = _monta()
    lote = treino.montar_lote(_exemplos())
    logits = treino.logits_mlm(adapter, lote.input_ids)
    soma, peso_total, decomposicao = treino.perda_somada_do_lote(logits, lote, mlm.PESOS_INICIAIS)
    media, _ = treino.perda_do_lote(logits, lote, mlm.PESOS_INICIAIS)
    assert abs(float(soma.detach()) / peso_total - float(media.detach())) < 1e-6
    assert abs(float(media.detach()) - mlm.perda_ponderada(decomposicao, mlm.PESOS_INICIAIS)) < 1e-5


def test_diagnostico_separa_tirar_da_referencia_de_aprender_o_alelo():
    """A fracao `P(ALT)/(1-P(REF))` fica em ~1/3 quando so se tira massa da referencia, e sobe quando o modelo
    sabe qual alelo e. E o que desempata as duas explicacoes para uma queda da perda focal."""
    _exige_torch()
    exemplo = mlm.montar_exemplo("ACGT" * 4, [(4, 7, mlm.TIPO_VARIANTE)], variant_id="v", fonte="global",
                                 focal_index=5, ref="A")   # alt_seq[5] = "C" -> alvo 1; ref "A" -> 0
    lote = treino.montar_lote([exemplo])
    assert int(lote.focal_alvo[0]) == mlm.SNV_ALT_TO_INDEX["C"]
    assert int(lote.focal_ref[0]) == mlm.SNV_ALT_TO_INDEX["A"]

    def _logits(valores):
        saida = torch.zeros(1, 16, 4)
        saida[0, 5] = torch.tensor(valores)
        return saida

    # (a) massa toda na referencia: fracao entre as nao-ref e 1/3 (as tres empatadas)
    so_ref = treino.diagnostico_do_focal(_logits([4.0, 0.0, 0.0, 0.0]), lote)["global"]
    assert so_ref["p_ref"] > 0.9
    assert abs(so_ref["fracao_do_alt_entre_as_nao_ref"] - 1 / 3) < 1e-4, so_ref

    # (b) tirou da referencia SEM saber o alelo: p_ref cai, fracao continua em 1/3
    tirou = treino.diagnostico_do_focal(_logits([0.0, 0.0, 0.0, 0.0]), lote)["global"]
    assert tirou["p_ref"] < so_ref["p_ref"]
    assert abs(tirou["fracao_do_alt_entre_as_nao_ref"] - 1 / 3) < 1e-4, tirou

    # (c) aprendeu o alelo: a fracao SOBE
    aprendeu = treino.diagnostico_do_focal(_logits([0.0, 3.0, 0.0, 0.0]), lote)["global"]
    assert aprendeu["fracao_do_alt_entre_as_nao_ref"] > 0.9, aprendeu

    # (d) ACHATOU: a entropia vai para ln(4) e tudo anda para o uniforme. Foi o regime do piloto 4.
    assert so_ref["entropia"] < tirou["entropia"], (so_ref["entropia"], tirou["entropia"])
    assert abs(tirou["entropia"] - tirou["entropia_do_uniforme"]) < 1e-4, tirou
    assert aprendeu["entropia"] < tirou["entropia"], "concentrar no alelo BAIXA a entropia"


def test_diagnostico_ausente_quando_o_exemplo_nao_traz_a_referencia():
    _exige_torch()
    lote = treino.montar_lote(_exemplos())          # montados sem `ref=`
    assert lote.focal_ref is None
    assert "indisponivel" in treino.diagnostico_do_focal(torch.zeros(2, 16, 4), lote)


def test_juntar_diagnosticos_pondera_por_n():
    _exige_torch()
    um = {"global": {"p_ref": 0.4, "p_alt": 0.3, "fracao_do_alt_entre_as_nao_ref": 0.5, "n": 1}}
    outro = {"global": {"p_ref": 0.6, "p_alt": 0.1, "fracao_do_alt_entre_as_nao_ref": 0.25, "n": 3}}
    um["global"]["entropia"] = 1.0
    outro["global"]["entropia"] = 1.2
    junto = treino.juntar_diagnosticos([um, outro])["global"]
    assert junto["n"] == 4
    assert abs(junto["entropia"] - (1.0 + 1.2 * 3) / 4) < 1e-9
    assert abs(junto["p_ref"] - (0.4 + 0.6 * 3) / 4) < 1e-9


def test_decomposicao_massa_mais_escolha_e_a_perda_focal():
    """A identidade que torna a atribuicao exata: -log P(ALT) = -log(1-P(REF)) + -log(P(ALT)/(1-P(REF)))."""
    _exige_torch()
    exemplo = mlm.montar_exemplo("ACGT" * 4, [(4, 7, mlm.TIPO_VARIANTE)], variant_id="v", fonte="global",
                                 focal_index=5, ref="A")
    lote = treino.montar_lote([exemplo])
    logits = torch.zeros(1, 16, 4)
    logits[0, 5] = torch.tensor([1.5, 0.3, -0.7, 0.2])
    d = treino.diagnostico_do_focal(logits, lote)["global"]
    focal = -math.log(d["p_alt"])
    assert abs(d["termo_massa"] + d["termo_escolha"] - focal) < 1e-5, (d, focal)


def test_fracao_nao_fica_presa_em_um_terco_ao_tirar_massa_da_referencia():
    """O contraexemplo da revisao de 22/09: redistribuir PROPORCIONALMENTE preserva a fracao onde ela estiver.

    Minha regra ("tirar da referencia deixa a fracao em 1/3") so valia para redistribuicao uniforme.
    """
    _exige_torch()
    exemplo = mlm.montar_exemplo("ACGT" * 4, [(4, 7, mlm.TIPO_VARIANTE)], variant_id="v", fonte="global",
                                 focal_index=5, ref="A")
    lote = treino.montar_lote([exemplo])

    def _diag(probabilidades):
        saida = torch.zeros(1, 16, 4)
        saida[0, 5] = torch.tensor(probabilidades).log()
        return treino.diagnostico_do_focal(saida, lote)["global"]

    antes = _diag([0.60, 0.20, 0.12, 0.08])
    depois = _diag([0.40, 0.30, 0.18, 0.12])
    assert depois["p_ref"] < antes["p_ref"], "a referencia perdeu massa"
    assert abs(depois["fracao_do_alt_entre_as_nao_ref"] - antes["fracao_do_alt_entre_as_nao_ref"]) < 1e-5
    assert abs(antes["fracao_do_alt_entre_as_nao_ref"] - 0.5) < 1e-5, "ficou em 0,50, nao em 1/3"
    # E a queda da perda focal veio TODA do termo de massa.
    assert abs(depois["termo_escolha"] - antes["termo_escolha"]) < 1e-5
    assert depois["termo_massa"] < antes["termo_massa"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed, skipped = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Skip as exc:
            skipped.append(name)
            print(f"  SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    ran = len(tests) - failed - len(skipped)
    tail = f"  |  {len(skipped)} PULADO(S), sem cobertura: {', '.join(skipped)}" if skipped else ""
    print(f"\n{ran}/{len(tests) - len(skipped)} passaram{tail}")
    if skipped and os.environ.get("REQUIRE_NO_SKIP"):
        print("REQUIRE_NO_SKIP: teste pulado conta como falha")
        sys.exit(1)
    sys.exit(1 if failed else 0)
