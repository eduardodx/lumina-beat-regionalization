"""Prova que o plano de janelas segue a receita declarada, e nao uma versao conveniente dela.

As tres coisas que nao podem escorregar: a janela desliza em torno da variante (a coordenada genomica nao muda),
o span de variante cobre o focal e os de referencia nao o tocam, e a amostragem e estratificada por AF.
    PYTHONPATH=. python3 tests/test_build_adapter_window_plan.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.build_adapter_window_plan as gen  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _pool(n_por_bin=40):
    """Pool sintetico com variantes em todos os bins de AF."""
    linhas = []
    afs = [0.0005, 0.003, 0.008, 0.03, 0.08, 0.3, 0.8]
    for indice, af in enumerate(afs):
        for j in range(n_por_bin):
            linhas.append({"chrom": f"chr{(indice % 3) + 1}", "pos": 10_000 + indice * 1000 + j,
                           "ref": "A", "alt": "G", "af_abraom": af})
    return pd.DataFrame(linhas)


def test_janela_desliza_em_torno_da_variante():
    """window_start = pos-1 - focal_index: a variante cai no focal sem mudar de lugar no genoma."""
    rng = np.random.default_rng(7)
    amostras = _pool(2).head(5).copy()
    amostras["af_bin"] = gen.rotular_bins(amostras["af_abraom"])
    plano = gen.montar_plano(amostras, fonte=gen.FONTE_ABRAOM, rng=rng, window_bp=4096, margem=64,
                             span_min=3, span_max=10, spans_de_referencia=1)
    for linha in plano.itertuples():
        assert linha.window_start + linha.focal_index == linha.pos_1based - 1
    # O plano fala o vocabulario da campanha, senao o auditor de janelas nao o consome.
    for coluna in ("variant_id", "chrom", "pos_1based", "ref", "alt"):
        assert coluna in plano.columns, coluna
    assert plano["variant_id"].is_unique


def test_focal_respeita_a_margem_do_contexto_local():
    rng = np.random.default_rng(3)
    focais = gen.posicoes_focais(rng, quantidade=500, window_bp=4096, margem=64)
    assert focais.min() >= 64 and focais.max() < 4096 - 64


def test_janela_pequena_demais_para_a_margem_falha():
    rng = np.random.default_rng(1)
    try:
        gen.posicoes_focais(rng, quantidade=1, window_bp=100, margem=64)
    except ValueError:
        return
    raise AssertionError("janela menor que duas margens tinha de falhar")


def test_span_de_variante_cobre_o_focal_e_os_de_referencia_nao():
    rng = np.random.default_rng(11)
    for _ in range(200):
        focal = int(rng.integers(64, 4096 - 64))
        spans = gen.spans_da_janela(rng, focal_index=focal, window_bp=4096, span_min=3, span_max=10,
                                    spans_de_referencia=2, margem=64)
        de_variante = [s for s in spans if s[2] == gen.TIPO_VARIANTE]
        assert len(de_variante) == 1
        inicio, fim, _ = de_variante[0]
        assert inicio <= focal < fim, (inicio, focal, fim)
        for comeco, termino, tipo in spans:
            if tipo == gen.TIPO_REFERENCIA:
                assert not (comeco <= focal < termino), "span de referencia nao pode tocar o focal"
            assert 0 <= comeco < termino <= 4096


def test_spans_nao_se_sobrepoem():
    rng = np.random.default_rng(5)
    spans = gen.spans_da_janela(rng, focal_index=2000, window_bp=4096, span_min=5, span_max=10,
                                spans_de_referencia=3, margem=64)
    ordenados = sorted((inicio, fim) for inicio, fim, _ in spans)
    for (_, fim_anterior), (inicio, _) in zip(ordenados, ordenados[1:]):
        assert inicio >= fim_anterior, ordenados


def test_amostragem_estratificada_equilibra_quando_da():
    """Pool equilibrado: cada bin entrega a mesma fatia."""
    amostra = gen.amostrar_estratificado(_pool(40), n=70, rng=np.random.default_rng(2))
    bins = amostra["af_bin"].value_counts()
    assert len(bins) == 7 and bins.max() - bins.min() <= 1, bins
    assert len(amostra) == 70


def test_bin_curto_entrega_o_que_tem_e_o_resto_vai_para_quem_sobra():
    """Sem estratificar, 54% do pool real cairia no bin mais raro e o adapter veria so singletons. Mas um bin
    curto nao pode derrubar o total: o que ele nao entrega passa para os bins com folga."""
    rng = np.random.default_rng(2)
    desbalanceado = pd.concat([_pool(5), _pool(300).head(300)], ignore_index=True)
    amostra = gen.amostrar_estratificado(desbalanceado, n=70, rng=rng)
    assert len(amostra) == 70, "o total pedido tem de ser atingido"
    bins = amostra["af_bin"].value_counts()
    assert len(bins) == 7, bins
    curtos = [q for b, q in bins.items() if b != "(-0.001, 0.001]"]
    assert all(q == 5 for q in curtos), bins  # bins curtos entregaram tudo que tinham


def test_amostragem_e_deterministica_pela_seed():
    pool = _pool()
    uma = gen.amostrar_estratificado(pool, n=30, rng=np.random.default_rng(42))
    outra = gen.amostrar_estratificado(pool.sample(frac=1, random_state=9), n=30,
                                       rng=np.random.default_rng(42))
    assert list(uma["pos"]) == list(outra["pos"]), "a ordem do pool nao pode mudar a amostra"


def test_bin_curto_nao_impede_atingir_o_total():
    rng = np.random.default_rng(4)
    pool = _pool(3)
    amostra = gen.amostrar_estratificado(pool, n=15, rng=rng)
    assert len(amostra) == 15, len(amostra)


def _executa(tmp: Path, **extra):
    pool_path = tmp / "abraom.parquet"
    _pool().to_parquet(pool_path, index=False)
    saida = tmp / "out"
    argumentos = ["--abraom-pool", str(pool_path), "--n-janelas", "50", "--out-dir", str(saida)]
    for chave, valor in extra.items():
        argumentos += [f"--{chave.replace('_', '-')}", str(valor)]
    rc = gen.main(argumentos)
    manifesto = json.loads((saida / "manifesto_do_plano.json").read_text(encoding="utf-8"))
    plano = pd.read_parquet(saida / "plano_de_janelas.parquet")
    return rc, manifesto, plano


def test_pool_curto_reprova_em_vez_de_encolher_calado():
    """Pedir 60 e receber 2 mudaria a mistura sem ninguem decidir: tem de reprovar, nao publicar quieto."""
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        pool_path = raiz / "abraom.parquet"
        _pool(1).head(2).to_parquet(pool_path, index=False)
        saida = raiz / "out"
        rc = gen.main(["--abraom-pool", str(pool_path), "--n-janelas", "60", "--out-dir", str(saida)])
        assert rc == 2, rc
        manifesto = json.loads((saida / "manifesto_do_plano.json").read_text(encoding="utf-8"))
        assert manifesto["pronto_para_campanha"] is False
        assert manifesto["conferencia"]["janelas_produzidas"] < 60
        assert any("nao tinha variantes suficientes" in p for p in manifesto["pendencias"]), manifesto["pendencias"]


def test_mistura_conferida_nas_linhas_produzidas():
    """`pronto_para_campanha` nao pode significar so 'recebi um arquivo global'."""
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        global_path = raiz / "global.parquet"
        _pool(2).to_parquet(global_path, index=False)  # 14 linhas: nao sustenta 60% de 50
        rc, manifesto, _ = _executa(raiz, global_pool=global_path)
        assert rc == 2, rc
        assert manifesto["pronto_para_campanha"] is False
        assert manifesto["conferencia"]["fracao_global_efetiva_nas_linhas"] < 0.6


def test_sem_pool_global_o_plano_e_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        rc, manifesto, plano = _executa(Path(tmp))
        assert rc == 0
        assert manifesto["pronto_para_campanha"] is False and manifesto["pendencias"]
        assert manifesto["receita"]["fracao_global_efetiva"] == 0.0
        assert set(plano["fonte"]) == {gen.FONTE_ABRAOM}
        assert len(plano) == 50


def test_com_pool_global_a_mistura_sai_na_proporcao():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        global_path = raiz / "global.parquet"
        _pool().to_parquet(global_path, index=False)
        rc, manifesto, plano = _executa(raiz, global_pool=global_path)
        assert rc == 0 and manifesto["pronto_para_campanha"] is True
        assert manifesto["resumo"]["fracao_global"] == 0.6, manifesto["resumo"]
        assert manifesto["conferencia"]["janelas_produzidas"] == 50
        assert manifesto["conferencia"]["fracao_global_efetiva_nas_linhas"] == 0.6
        assert manifesto["falta_antes_de_treinar"], "auditar as janelas contra o FASTA continua faltando"
        assert manifesto["resumo"]["por_fonte"] == {gen.FONTE_GLOBAL: 30, gen.FONTE_ABRAOM: 20}


def test_fracao_invalida_reprova():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        pool_path = raiz / "abraom.parquet"
        _pool().to_parquet(pool_path, index=False)
        assert gen.main(["--abraom-pool", str(pool_path), "--n-janelas", "10",
                         "--fracao-global", "1.5", "--out-dir", str(raiz / "out")]) == 2


def test_manifesto_registra_a_receita_inteira():
    with tempfile.TemporaryDirectory() as tmp:
        _, manifesto, _ = _executa(Path(tmp))
        receita = manifesto["receita"]
        for chave in ("window_bp", "margem", "variantes_por_janela", "fracao_global_pedida", "span_min",
                      "span_max", "spans_de_referencia", "seed", "posicao_da_variante", "amostragem"):
            assert chave in receita, chave
        assert len(manifesto["saidas"]["plano_sha256"]) == 64


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
