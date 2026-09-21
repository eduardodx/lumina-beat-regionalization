"""Prova que a separacao do adapter e por LOCO, e que a disjuncao e verificada e nao assumida.

O caso que importa e o encadeamento: A cobre B, B cobre C, A nao cobre C. Se o script tratasse so pares, mandaria
A e C para lados diferentes e B contaminaria um deles.
    PYTHONPATH=. python3 tests/test_split_adapter_plan_by_locus.py
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

import scripts.split_adapter_plan_by_locus as sp  # noqa: E402

JANELA = 100


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _plano(linhas):
    """linhas = [(chrom, window_start, fonte, af_bin)]"""
    return pd.DataFrame([{
        "variant_id": f"{c}:{s}:A:G", "chrom": c, "pos_1based": s + 50, "focal_index": 50,
        "window_start": s, "fonte": f, "af_bin": b,
    } for c, s, f, b in linhas])


def test_janelas_que_se_sobrepoem_caem_no_mesmo_loco():
    plano = _plano([("chr1", 0, "global", "b1"), ("chr1", 50, "global", "b1"),
                    ("chr1", 500, "global", "b1")])
    ids = sp.atribuir_locus(plano, window_bp=JANELA)
    assert ids.iloc[0] == ids.iloc[1], "janelas [0,100) e [50,150) se sobrepoem"
    assert ids.iloc[2] != ids.iloc[0]


def test_encadeamento_liga_a_ponta_que_nao_se_toca():
    """A=[0,100), B=[80,180), C=[160,260): A nao cobre C, mas B cobre os dois -- os tres sao um loco so."""
    plano = _plano([("chr1", 0, "global", "b1"), ("chr1", 80, "global", "b1"),
                    ("chr1", 160, "global", "b1")])
    ids = sp.atribuir_locus(plano, window_bp=JANELA)
    assert ids.nunique() == 1, list(ids)


def test_cromossomos_diferentes_nunca_compartilham_loco():
    plano = _plano([("chr1", 0, "global", "b1"), ("chr2", 0, "global", "b1")])
    ids = sp.atribuir_locus(plano, window_bp=JANELA)
    assert ids.nunique() == 2


def test_folga_separa_o_que_so_encosta():
    plano = _plano([("chr1", 0, "global", "b1"), ("chr1", 100, "global", "b1")])
    assert sp.atribuir_locus(plano, window_bp=JANELA).nunique() == 2, "[0,100) e [100,200) nao se sobrepoem"
    assert sp.atribuir_locus(plano, window_bp=JANELA, folga_bp=50).nunique() == 1, "com folga, viram um loco"


def test_id_do_loco_nao_depende_da_ordem_do_arquivo():
    plano = _plano([("chr1", 500, "global", "b1"), ("chr1", 0, "global", "b1"),
                    ("chr1", 50, "global", "b1")])
    ids = sp.atribuir_locus(plano, window_bp=JANELA)
    embaralhado = plano.iloc[[2, 0, 1]].reset_index(drop=True)
    ids2 = sp.atribuir_locus(embaralhado, window_bp=JANELA)
    por_variante = dict(zip(plano["variant_id"], ids))
    por_variante2 = dict(zip(embaralhado["variant_id"], ids2))
    assert por_variante == por_variante2, (por_variante, por_variante2)


def test_violacao_e_detectada_quando_os_lados_se_tocam():
    treino = _plano([("chr1", 0, "global", "b1")]).assign(locus_id="chr1:0")
    validacao = _plano([("chr1", 50, "global", "b1")]).assign(locus_id="chr1:1")
    problemas = sp.violacoes_de_disjuncao(treino, validacao, window_bp=JANELA)
    assert len(problemas) == 1, problemas
    distante = _plano([("chr1", 5000, "global", "b1")]).assign(locus_id="chr1:9")
    assert sp.violacoes_de_disjuncao(treino, distante, window_bp=JANELA) == []


def test_violacao_ignora_sobreposicao_dentro_do_mesmo_lado():
    """Treino consigo mesmo se sobrepoe o tempo todo; isso nao e vazamento."""
    treino = _plano([("chr1", 0, "global", "b1"), ("chr1", 10, "global", "b1")]).assign(locus_id="chr1:0")
    validacao = _plano([("chr1", 9000, "global", "b1")]).assign(locus_id="chr1:5")
    assert sp.violacoes_de_disjuncao(treino, validacao, window_bp=JANELA) == []


def _plano_grande(n_por_celula=40):
    linhas = []
    passo = 10_000
    posicao = 0
    for fonte in ("global", "abraom"):
        for af_bin in ("b1", "b2"):
            for _ in range(n_por_celula):
                linhas.append(("chr1", posicao, fonte, af_bin))
                posicao += passo
    return _plano(linhas)


def test_escolha_preserva_a_composicao_em_cada_celula():
    plano = _plano_grande()
    plano["locus_id"] = sp.atribuir_locus(plano, window_bp=JANELA)
    escolhidos = sp.escolher_validacao(plano, fracao=0.25, rng=np.random.default_rng(3))
    validacao = plano[plano["locus_id"].isin(escolhidos)]
    for chave, grupo in plano.groupby(["fonte", "af_bin"], sort=True):
        na_validacao = len(validacao[(validacao["fonte"] == chave[0]) & (validacao["af_bin"] == chave[1])])
        assert abs(na_validacao / len(grupo) - 0.25) <= 0.05, (chave, na_validacao, len(grupo))


def _executa(tmp: Path, plano: pd.DataFrame, **extra):
    caminho = tmp / "plano.parquet"
    plano.to_parquet(caminho, index=False)
    saida = tmp / "out"
    argumentos = ["--plano", str(caminho), "--window-bp", str(JANELA), "--out-dir", str(saida)]
    for chave, valor in extra.items():
        argumentos += [f"--{chave.replace('_', '-')}", str(valor)]
    rc = sp.main(argumentos)
    relatorio = json.loads((saida / "separacao_por_locus.json").read_text(encoding="utf-8"))
    return rc, relatorio, saida


def test_end_to_end_publica_os_dois_recortes_com_disjuncao_verificada():
    with tempfile.TemporaryDirectory() as tmp:
        rc, relatorio, saida = _executa(Path(tmp), _plano_grande(), fracao_validacao=0.2)
        assert rc == 0, relatorio["pendencias"]
        assert relatorio["disjuncao_verificada"] is True and relatorio["violacoes"] == 0
        assert relatorio["treino"]["n"] + relatorio["validacao"]["n"] == 160
        assert (saida / "plano_treino.parquet").exists() and (saida / "plano_validacao.parquet").exists()
        assert len(relatorio["saidas"]["plano_treino_sha256"]) == 64
        # A mistura tem de sobreviver aos dois lados.
        assert abs(relatorio["treino"]["fracao_global"] - 0.5) < 0.05
        assert abs(relatorio["validacao"]["fracao_global"] - 0.5) < 0.05


def test_end_to_end_nao_publica_quando_um_recorte_fica_vazio():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        # Um loco so: nao ha como separar sem esvaziar um lado.
        plano = _plano([("chr1", 0, "global", "b1"), ("chr1", 10, "global", "b1")])
        rc, relatorio, saida = _executa(raiz, plano, fracao_validacao=0.5)
        assert rc == 2
        assert relatorio["pendencias"]
        assert not (saida / "plano_treino.parquet").exists()


def test_end_to_end_reprova_plano_sem_as_colunas():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        caminho = raiz / "plano.parquet"
        pd.DataFrame({"chrom": ["chr1"], "pos": [1]}).to_parquet(caminho, index=False)
        assert sp.main(["--plano", str(caminho), "--out-dir", str(raiz / "out")]) == 2


def test_fracao_invalida_reprova():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        caminho = raiz / "plano.parquet"
        _plano_grande().to_parquet(caminho, index=False)
        assert sp.main(["--plano", str(caminho), "--fracao-validacao", "1.5",
                        "--out-dir", str(raiz / "out")]) == 2


def test_estrutura_separa_as_duas_metades():
    """As metades chegam com estruturas diferentes: o relatorio tem de mostrar isso, nao so o total."""
    aglomerado = [("chr1", 0, "global", "b1"), ("chr1", 20, "global", "b1"), ("chr1", 40, "global", "b1")]
    espalhado = [("chr2", 10_000 * i, "abraom", "b1") for i in range(4)]
    plano = _plano(aglomerado + espalhado)
    plano["locus_id"] = sp.atribuir_locus(plano, window_bp=JANELA)
    estrutura = sp.estrutura_de_locos(plano)
    assert estrutura["por_fonte"]["global"]["locos_que_a_contem"] == 1
    assert estrutura["por_fonte"]["global"]["janelas_por_loco_mediana"] == 3.0
    assert estrutura["por_fonte"]["abraom"]["locos_que_a_contem"] == 4
    assert estrutura["por_fonte"]["abraom"]["locos_com_uma_janela"] == 4
    assert estrutura["locos_mistos"] == 0


def test_loco_misto_e_contado():
    plano = _plano([("chr1", 0, "global", "b1"), ("chr1", 20, "abraom", "b1")])
    plano["locus_id"] = sp.atribuir_locus(plano, window_bp=JANELA)
    assert sp.estrutura_de_locos(plano)["locos_mistos"] == 1


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
