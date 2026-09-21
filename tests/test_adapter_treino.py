"""Prova o nucleo em torch do adapter: loss diferenciavel, invariantes de congelamento e checkpoint estrito.

Precisa de torch (roda no notebook; no Windows os testes PULAM, e o runner conta separado).
    PYTHONPATH=. REQUIRE_NO_SKIP=1 python3 tests/test_adapter_treino.py
"""

from __future__ import annotations

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
            assert float(parametro.abs().sum()) == 0.0, nome
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
