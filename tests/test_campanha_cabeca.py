"""Cabeca sobre o cache: parada pela macro, Platt e determinismo. Precisa de torch.

    PYTHONPATH=. REQUIRE_NO_SKIP=1 python3 tests/test_campanha_cabeca.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS."""


try:
    import torch  # noqa: F401

    from eval.campanha import cabeca, metricas
    TORCH = True
except ImportError as exc:  # pragma: no cover - no Windows
    TORCH = False
    MOTIVO = str(exc)


def _exige_torch():
    if not TORCH:
        raise Skip(f"sem torch: {MOTIVO}")


def _problema(n=900, d=12, seed=0):
    """Rotulo depende de 2 das 12 colunas; paineis de discriminacao alternados."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)) > 0).astype(int)
    tabela = pd.DataFrame({"binary_label": y, "primary_panel": np.array(["missense", "splice", "noncoding"])[
        np.arange(n) % 3], "variant_id": [f"v{i}" for i in range(n)]})
    linhas = {"train": np.arange(0, 600), "validation": np.arange(600, 750), "selecao": np.arange(750, n)}
    return X, tabela, linhas


def test_aprende_um_problema_separavel_e_para_pela_macro():
    _exige_torch()
    X, tabela, linhas = _problema()
    rodada = cabeca.rodar_sementes(X, tabela, linhas, sementes=[11])[0]
    assert rodada["selecao"]["macro"] > 0.85, rodada["selecao"]["macro"]
    assert rodada["epoca"] % cabeca.RECEITA["avaliar_a_cada"] == 0
    assert 0.0 < rodada["limiar_de_mcc"]["limiar"] < 1.0


def test_platt_nao_muda_a_ordem():
    _exige_torch()
    X, tabela, linhas = _problema()
    treinada = cabeca.treinar(X[linhas["train"]], tabela["binary_label"].to_numpy()[linhas["train"]],
                              X[linhas["validation"]], tabela["binary_label"].to_numpy()[linhas["validation"]],
                              tabela["primary_panel"].to_numpy()[linhas["validation"]], semente=11)
    logits = cabeca.pontuar(treinada, X[linhas["selecao"]])
    y = tabela["binary_label"].to_numpy()[linhas["selecao"]]
    a, b = cabeca.platt(cabeca.pontuar(treinada, X[linhas["validation"]]),
                        tabela["binary_label"].to_numpy()[linhas["validation"]])
    assert a > 0, "Platt com inclinacao negativa inverteria a ordem"
    assert metricas.auroc(logits, y) == metricas.auroc(cabeca.calibrar(logits, a, b), y)


def test_mesma_semente_no_cpu_da_o_mesmo_resultado():
    _exige_torch()
    X, tabela, linhas = _problema()
    primeira = cabeca.rodar_sementes(X, tabela, linhas, sementes=[12])[0]
    segunda = cabeca.rodar_sementes(X, tabela, linhas, sementes=[12])[0]
    assert np.array_equal(primeira["prob_selecao"], segunda["prob_selecao"])


def test_papel_com_uma_classe_so_e_recusado():
    _exige_torch()
    X, tabela, linhas = _problema()
    tabela.loc[linhas["validation"], "binary_label"] = 1
    try:
        cabeca.rodar_sementes(X, tabela, linhas, sementes=[11])
    except ValueError:
        return
    raise AssertionError("validacao com uma classe so deveria ser recusada")



def _cache_sintetico(pasta, *, sistema, tabela, sinal, campanha):
    """Um cache no formato REAL do extrator (identidade, manifesto, tabela, fragmentos), com features sinteticas."""
    import json

    from eval.campanha import cache as cache_io
    from eval.campanha.recortes import adapter_congelado, hash_do_conteudo

    pasta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7 if sistema == "M0" else 8)
    y = tabela["binary_label"].to_numpy().astype(np.float32)
    matrizes = {}
    for nome, dims in (("cabecas_172", 172), ("leitura_antiga_1344", 1344)):
        base = rng.normal(size=(len(tabela), dims)).astype(np.float32)
        base[:, 0] += sinal * (2 * y - 1)
        matrizes[nome] = base
    a1 = adapter_congelado(campanha, 20260921)["sha256"]
    identidade = {"sistema": sistema, "tabela_sha256_conteudo": hash_do_conteudo(tabela),
                  "codigo": {"arquivos": {"x.py": "1"}}, "ambiente": {"gpu": "sintetica"},
                  "adapter_sha256": a1 if sistema == "MR" else None,
                  "semente_do_adapter": 20260921 if sistema == "MR" else None}
    (pasta / "identidade.json").write_text(json.dumps(identidade), encoding="utf-8")
    (pasta / "manifesto.json").write_text(json.dumps({"completo": True}), encoding="utf-8")
    tabela.to_parquet(pasta / "tabela.parquet", index=False)
    cache_io.gravar_fragmento(pasta, 0, variant_id=tabela["variant_id"].to_numpy().astype(str),
                              papel=tabela["papel"].to_numpy().astype(str), matrizes=matrizes)


def test_g5_e_comparador_de_ponta_a_ponta_num_cache_sintetico():
    """Roda os DOIS scripts inteiros, como no notebook, antes de gastar a rodada real."""
    _exige_torch()
    import json
    import tempfile

    from eval.campanha.recortes import carregar_campanha
    from scripts import comparar_m0_mr_desenvolvimento as comparador
    from scripts import g5_escolher_extracao_e_politica as g5_script

    raiz = Path(__file__).resolve().parents[1]
    campanha = carregar_campanha(raiz / "configs" / "campanha_r03_desenvolvimento.json")
    rng = np.random.default_rng(0)
    papeis = ["train"] * 360 + ["validation"] * 120 + ["selecao"] * 120
    tabela = pd.DataFrame({
        "variant_id": [f"v{i:04d}" for i in range(len(papeis))], "chrom": "chr1",
        "pos_1based": np.arange(len(papeis)) + 100, "ref": "A", "alt": "G",
        "binary_label": (rng.random(len(papeis)) < 0.4).astype(int),
        "primary_panel": np.array(["missense", "splice", "noncoding"])[np.arange(len(papeis)) % 3],
        "overlap_cluster_id": [f"c{i // 4}" for i in range(len(papeis))], "label_tier": "gold", "papel": papeis})
    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        _cache_sintetico(pasta / "M0", sistema="M0", tabela=tabela, sinal=1.0, campanha=campanha)
        _cache_sintetico(pasta / "MR", sistema="MR", tabela=tabela, sinal=1.2, campanha=campanha)
        treino = tabela.loc[tabela["papel"] == "train", "variant_id"].tolist()
        argumentos_de_snapshot = []
        for nome, fracao in (("nenhum", 1.0), ("janela2048", 0.7), ("janela4096", 0.5)):
            snap = pd.DataFrame({"variant_id": treino[: int(len(treino) * fracao)], "role": "train"})
            snap.to_parquet(pasta / f"{nome}.parquet", index=False)
            argumentos_de_snapshot += ["--snapshot", f"{nome}={pasta / (nome + '.parquet')}"]
        comum = ["--campanha", str(raiz / "configs" / "campanha_r03_desenvolvimento.json"), *argumentos_de_snapshot]
        assert g5_script.main(["--cache-m0", str(pasta / "M0"), *comum, "--out-dir", str(pasta / "g5")]) == 2, \
            "sem a confirmacao da regra da extracao o G5 tem de recusar"
        assert g5_script.main(["--cache-m0", str(pasta / "M0"), *comum, "--confirmo-a-regra-da-extracao",
                               "--out-dir", str(pasta / "g5")]) == 0
        decisao = json.loads((pasta / "g5" / "g5_decisao.json").read_text(encoding="utf-8"))
        assert decisao["decisao"]["politica"] in ("nenhum", "janela2048", "janela4096")
        assert g5_script.main(["--cache-m0", str(pasta / "MR"), *comum, "--confirmo-a-regra-da-extracao",
                               "--out-dir", str(pasta / "g5_mr")]) == 2, "o G5 nao aceita cache de MR"
        assert comparador.main(["--cache-m0", str(pasta / "M0"), "--cache-mr", str(pasta / "MR"),
                                "--decisao-g5", str(pasta / "g5" / "g5_decisao.json"), *comum,
                                "--replicas", "50", "--out-dir", str(pasta / "comparacao")]) == 0
        saida = json.loads((pasta / "comparacao" / "comparacao_m0_mr.json").read_text(encoding="utf-8"))
        assert len(saida["por_semente"]) == 3 and saida["bootstrap_da_media"]["natureza"] == "exploratoria"


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    falhas, pulados = 0, []
    for nome, funcao in testes:
        try:
            funcao()
            print(f"  PASS  {nome}")
        except Skip as exc:
            pulados.append(nome)
            print(f"  SKIP  {nome}: {exc}")
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"  FAIL  {nome}: {type(exc).__name__}: {exc}")
    print(f"\n{len(testes) - falhas - len(pulados)}/{len(testes) - len(pulados)} passaram"
          + (f"  |  {len(pulados)} PULADO(S)" if pulados else ""))
    if pulados and os.environ.get("REQUIRE_NO_SKIP"):
        print("REQUIRE_NO_SKIP: teste pulado conta como falha")
        sys.exit(1)
    sys.exit(1 if falhas else 0)
