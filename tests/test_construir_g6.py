"""Construtor do G6 de ponta a ponta sobre caches SINTETICOS: G5, os tres comparadores, as tres conferencias e o
construtor, na ordem do notebook. Precisa de torch.

    PYTHONPATH=. REQUIRE_NO_SKIP=1 python3 tests/test_construir_g6.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS."""


try:
    import torch  # noqa: F401

    TORCH = True
except ImportError as exc:  # pragma: no cover - no Windows
    TORCH = False
    MOTIVO = str(exc)

SEMENTES = (20260921, 20260922, 20260923)


def _exige_torch():
    if not TORCH:
        raise Skip(f"sem torch: {MOTIVO}")


def _tabela():
    rng = np.random.default_rng(0)
    papeis = ["train"] * 360 + ["validation"] * 120 + ["selecao"] * 120
    return pd.DataFrame({
        "variant_id": [f"v{i:04d}" for i in range(len(papeis))], "chrom": "chr1",
        "pos_1based": np.arange(len(papeis)) + 100, "ref": "A", "alt": "G",
        "binary_label": (rng.random(len(papeis)) < 0.4).astype(int),
        "primary_panel": np.array(["missense", "splice", "noncoding"])[np.arange(len(papeis)) % 3],
        "overlap_cluster_id": [f"c{i // 4}" for i in range(len(papeis))], "label_tier": "gold", "papel": papeis})


def _cache(pasta: Path, *, semente_do_adapter, tabela, sinal, campanha, rng_seed):
    """Um cache no formato REAL do extrator, com features sinteticas; M0 quando semente_do_adapter e None."""
    from eval.campanha import cache as cache_io
    from eval.campanha.recortes import adapter_congelado, hash_do_conteudo

    pasta.mkdir(parents=True)
    rng = np.random.default_rng(rng_seed)
    y = tabela["binary_label"].to_numpy().astype(np.float32)
    matrizes = {}
    for nome, dims in (("cabecas_172", 172), ("leitura_antiga_1344", 1344)):
        base = rng.normal(size=(len(tabela), dims)).astype(np.float32)
        base[:, 0] += sinal * (2 * y - 1)
        matrizes[nome] = base
    mr = semente_do_adapter is not None
    identidade = {"sistema": "MR" if mr else "M0", "tabela_sha256_conteudo": hash_do_conteudo(tabela),
                  "codigo": {"arquivos": {"x.py": "1"}}, "ambiente": {"gpu": "sintetica"},
                  "checkpoint_sha256": campanha["adapter_do_mr"]["referencia"]["checkpoint_sha256"],
                  "adapter_sha256": adapter_congelado(campanha, semente_do_adapter)["sha256"] if mr else None,
                  "semente_do_adapter": semente_do_adapter}
    (pasta / "identidade.json").write_text(json.dumps(identidade), encoding="utf-8")
    (pasta / "manifesto.json").write_text(json.dumps({"completo": True}), encoding="utf-8")
    tabela.to_parquet(pasta / "tabela.parquet", index=False)
    cache_io.gravar_fragmento(pasta, 0, variant_id=tabela["variant_id"].to_numpy().astype(str),
                              papel=tabela["papel"].to_numpy().astype(str), matrizes=matrizes)


def _regra(**extra):
    return {"estudos": ["br_clinical_evidence"], "metrica": "auroc", "estatistica": "p2_5", **extra}


def _campanha_do_teste(tabela, destino: Path, *, selecao_comum: Path, resolvida=False, abraom_sha=None) -> Path:
    """A declaracao real com as contagens e o hash da selecao sintetica; `resolvida` fecha margens, bootstrap e
    pendencias com CONTEUDO (exemplo sintetico, nao proposta de margem)."""
    campanha = json.loads((RAIZ / "configs" / "campanha_r03_desenvolvimento.json").read_text(encoding="utf-8"))
    selecao = tabela[tabela["papel"] == "selecao"]
    campanha["recortes"]["comparacao_de_desenvolvimento"].update(
        variantes=int(len(selecao)), clusters=int(selecao["overlap_cluster_id"].nunique()),
        sha256_prefixo=hashlib.sha256(selecao_comum.read_bytes()).hexdigest()[:8])
    campanha["recortes"]["parada_e_calibracao"]["variantes"] = int((tabela["papel"] == "validation").sum())
    if resolvida:
        campanha["g6"]["margens"] = {
            "estado": "DECLARADO (teste sintetico)",
            "melhoria_minima_no_coorte_br": _regra(delta="delta_br_full", limite=0.0),
            "regressao_maxima_no_controle": _regra(delta="delta_control", limite=-0.01),
            "paineis_com_regressao_inaceitavel": _regra(delta="delta_br_full", limite=-0.02, paineis=["missense"]),
            "beneficio_nao_explicado_por_um_painel": _regra(delta="delta_br_full", limite=0.0,
                                                            suporte_minimo_por_painel=10),
            "interacao": {"criterio_proprio": False, "motivo": "relatada com os absolutos (teste)"}}
        campanha["g6"]["bootstrap_da_interacao"] = {
            "estado": "DECLARADO (teste sintetico)", "unidade_principal": "cluster_conjunto",
            "unidade_de_sensibilidade": "par", "replicas": 1000, "seed": 20260901, "percentis": [2.5, 97.5]}
        for pendencia in campanha["g6"]["pendencias_antes_do_congelamento"]:
            pendencia.update(estado="FEITO", onde="teste sintetico")
    if abraom_sha:
        campanha["g6"]["proveniencia"]["abraom"]["sha256"] = abraom_sha
    destino.write_text(json.dumps(campanha), encoding="utf-8")
    return destino


def _montar(raiz: Path, tabela) -> dict:
    """Caches, snapshots, G5, os tres comparadores e as tres conferencias, como no notebook."""
    from eval.campanha.recortes import carregar_campanha
    from scripts import comparar_m0_mr_desenvolvimento as comparador
    from scripts import conferir_cabecas_salvas as conferencia
    from scripts import g5_escolher_extracao_e_politica as g5_script

    selecao_comum = raiz / "selecao_comum.parquet"
    tabela.loc[tabela["papel"] == "selecao", ["variant_id"]].to_parquet(selecao_comum, index=False)
    configuracao = _campanha_do_teste(tabela, raiz / "campanha.json", selecao_comum=selecao_comum)
    campanha = carregar_campanha(configuracao)
    caches = {"M0": raiz / "g3_cache" / "M0"}
    _cache(caches["M0"], semente_do_adapter=None, tabela=tabela, sinal=1.0, campanha=campanha, rng_seed=7)
    for i, semente in enumerate(SEMENTES, start=1):
        caches[semente] = raiz / "g3_cache" / f"MR_a{i}"
        _cache(caches[semente], semente_do_adapter=semente, tabela=tabela, sinal=1.0 + 0.1 * i, campanha=campanha,
               rng_seed=7 + i)
    treino = tabela.loc[tabela["papel"] == "train", "variant_id"].tolist()
    validacao = tabela.loc[tabela["papel"] == "validation", "variant_id"].tolist()
    snapshots, argumentos = {}, []
    for nome, fracao in (("nenhum", 1.0), ("janela2048", 0.7), ("janela4096", 0.5)):
        snapshots[nome] = raiz / f"{nome}.parquet"
        do_treino = treino[: int(len(treino) * fracao)]
        pd.DataFrame({"variant_id": do_treino + validacao,
                      "role": ["train"] * len(do_treino) + ["validation"] * len(validacao)}).to_parquet(
            snapshots[nome], index=False)
        argumentos += ["--snapshot", f"{nome}={snapshots[nome]}"]
    comum = ["--campanha", str(configuracao), *argumentos]
    decisao = raiz / "g5" / "g5_decisao.json"
    assert g5_script.main(["--cache-m0", str(caches["M0"]), *comum, "--confirmo-a-regra-da-extracao",
                           "--out-dir", str(raiz / "g5")]) == 0
    for i, semente in enumerate(SEMENTES, start=1):
        pasta = raiz / f"comparacao_dev_a{i}"
        assert comparador.main(["--cache-m0", str(caches["M0"]), "--cache-mr", str(caches[semente]),
                                "--decisao-g5", str(decisao), *comum, "--replicas", "20",
                                "--out-dir", str(pasta)]) == 0
        referencia = [] if i == 1 else ["--m0-de-referencia", str(raiz / "comparacao_dev_a1")]
        assert conferencia.main(["--comparacao", str(pasta), "--cache-m0", str(caches["M0"]),
                                 "--cache-mr", str(caches[semente]), "--decisao-g5", str(decisao),
                                 "--campanha", str(configuracao), *referencia]) == 0
    politica = json.loads(decisao.read_text(encoding="utf-8"))["decisao"]["politica"]
    return {"caches": caches, "decisao": decisao, "snapshot": snapshots[politica], "campanha": configuracao,
            "selecao_comum": selecao_comum}


def _argumentos(raiz: Path, montado: dict, caches: dict | None = None) -> list[str]:
    caches = caches or montado["caches"]
    return ["--raiz", str(raiz), "--cache-m0", str(caches["M0"]),
            *[a for s in SEMENTES for a in ("--cache-mr", f"{s}={caches[s]}")],
            "--decisao-g5", str(montado["decisao"]), "--snapshot-da-politica", str(montado["snapshot"]),
            "--selecao", str(montado["selecao_comum"]), "--replicas", "30"]


def test_construtor_de_ponta_a_ponta():
    _exige_torch()
    from eval.campanha import g6
    from scripts import construir_g6 as construtor

    tabela = _tabela()
    with tempfile.TemporaryDirectory() as pasta:
        raiz = Path(pasta)
        montado = _montar(raiz, tabela)
        args = [*_argumentos(raiz, montado), "--campanha", str(montado["campanha"])]

        # 1. A declaracao de hoje: conferencias passam, sai o RASCUNHO com bloqueios, nada congela.
        assert construtor.main([*args, "--out-dir", str(raiz / "g6_a")]) == 0
        construcao = json.loads((raiz / "g6_a" / "g6_construcao.json").read_text(encoding="utf-8"))
        assert construcao["passou"] and construcao["bloqueios"], construcao["problemas"]
        assert all(v == "identica" for n in construcao["m0_nos_comparadores"].values() for v in n.values())
        assert not (raiz / "g6_a" / g6.NOME_DO_MANIFESTO).exists()
        rascunho = json.loads((raiz / "g6_a" / g6.NOME_DO_RASCUNHO).read_text(encoding="utf-8"))
        assert rascunho["estado"] == g6.RASCUNHO and rascunho["bloqueios"] == construcao["bloqueios"]
        rotulos = [c["rotulo"] for c in rascunho["sistemas"]["regionalized"]["componentes"]]
        assert rotulos == [f"MR_a{s}_h{h}" for s, h in zip(SEMENTES, (11, 12, 13))], rotulos

        # 2. O ensemble e a media das tres colunas, e o limiar e a regra do Mosaic nessa media do fold 1.
        predicoes = pd.read_parquet(raiz / "g6_a" / "g6_predicoes.parquet")
        np.testing.assert_allclose(predicoes["ensemble_MR"], predicoes[rotulos].mean(axis=1), rtol=0, atol=1e-15)
        fold1 = predicoes[predicoes["papel"] == "validation"]
        y = tabela.set_index("variant_id").loc[fold1["variant_id"], "binary_label"].to_numpy()
        for sistema in ("M0", "MR"):
            refeito = g6.limiar_do_mosaic(y, fold1[f"ensemble_{sistema}"].to_numpy())
            assert refeito["threshold"] == construcao["limiares_do_ensemble"][sistema]["threshold"]
        assert set(construcao["desenvolvimento_do_ensemble_final"]["bootstrap"]) >= {"macro", "auroc", "auprc"}

        # 3. Pasta existente e --congelar com bloqueio sao recusados.
        assert construtor.main([*args, "--out-dir", str(raiz / "g6_a")]) == 2
        assert construtor.main([*args, "--congelar", "--out-dir", str(raiz / "g6_b")]) == 2
        assert not (raiz / "g6_b" / g6.NOME_DO_MANIFESTO).exists()

        # 4. Com a declaracao resolvida, o ABraOM reconferido e o codigo dado como pronto, congela.
        abraom = raiz / "SABE1171.Abraom.clean.tsv"
        abraom.write_text("chrom\tpos\tref\talt\taf_abraom\n", encoding="utf-8")
        resolvida = _campanha_do_teste(tabela, raiz / "campanha_resolvida.json",
                                       selecao_comum=montado["selecao_comum"], resolvida=True,
                                       abraom_sha=hashlib.sha256(abraom.read_bytes()).hexdigest())
        original = construtor.estado_do_codigo
        construtor.estado_do_codigo = lambda arquivos: {"revisao": "0" * 40, "ausentes": [], "nao_rastreados": [],
                                                        "modificados": [], "erro": None}
        try:
            assert construtor.main([*_argumentos(raiz, montado), "--campanha", str(resolvida), "--proveniencia",
                                    f"abraom={abraom}", "--congelar", "--out-dir", str(raiz / "g6_c")]) == 0
        finally:
            construtor.estado_do_codigo = original
        manifesto, sha = g6.ler_manifesto_congelado(raiz / "g6_c")
        assert manifesto["estado"] == g6.CONGELADO and not manifesto["bloqueios"]
        assert manifesto["campos_do_mosaic"]["abraom_snapshot_hash"]["reconferido_no_arquivo"] is True
        rascunho_c = json.loads((raiz / "g6_c" / g6.NOME_DO_RASCUNHO).read_text(encoding="utf-8"))
        assert {k: v for k, v in rascunho_c.items() if k != "estado"} == \
            {k: v for k, v in manifesto.items() if k != "estado"}, "rascunho e congelado so diferem no estado"
        assert manifesto["sistemas"]["base"]["limiar"]["threshold"] == construcao["limiares_do_ensemble"]["M0"][
            "threshold"], "mesmo limiar nas duas construcoes"

        # 4b. Um git que falha nao vale como `sem mudancas`: com todo o resto resolvido, o --congelar e recusado.
        import subprocess

        def git_que_falha(*argumentos):
            raise subprocess.CalledProcessError(128, ["git", *argumentos], stderr="fatal: not a git repository")

        original = construtor._git
        construtor._git = git_que_falha
        try:
            assert construtor.main([*_argumentos(raiz, montado), "--campanha", str(resolvida), "--proveniencia",
                                    f"abraom={abraom}", "--congelar", "--out-dir", str(raiz / "g6_c2")]) == 2
        finally:
            construtor._git = original
        construcao_c2 = json.loads((raiz / "g6_c2" / "g6_construcao.json").read_text(encoding="utf-8"))
        assert construcao_c2["bloqueios"] == ["estado do codigo nao conferido (git falhou: CalledProcessError: "
                                              "fatal: not a git repository)"], construcao_c2["bloqueios"]
        assert not (raiz / "g6_c2" / g6.NOME_DO_MANIFESTO).exists()

        # 5. Adulteracoes: cabeca trocada no comparador da a_2 e caches de adapter trocados.
        alvo = raiz / "comparacao_dev_a2" / "cabeca_MR_h12.pt"
        guardada = alvo.read_bytes()
        shutil.copy(raiz / "comparacao_dev_a2" / "cabeca_MR_h11.pt", alvo)
        assert construtor.main([*args, "--out-dir", str(raiz / "g6_d")]) == 2
        alvo.write_bytes(guardada)
        trocados = dict(montado["caches"])
        trocados[SEMENTES[1]], trocados[SEMENTES[2]] = trocados[SEMENTES[2]], trocados[SEMENTES[1]]
        assert construtor.main([*_argumentos(raiz, montado, trocados), "--campanha", str(montado["campanha"]),
                                "--out-dir", str(raiz / "g6_e")]) == 2
        # 6. Uma cabeca do M0 do comparador da a_3 diferente da da a_1 (so a epoca) reprova.
        m0_a3 = raiz / "comparacao_dev_a3" / "cabeca_M0_h13.pt"
        carga = torch.load(m0_a3, weights_only=False)
        torch.save(dict(carga, epoca=carga["epoca"] + 10), m0_a3)
        assert construtor.main([*args, "--out-dir", str(raiz / "g6_f")]) == 2
        # 7. Composicao declarada fora do pareamento: recusada antes de ler qualquer cache.
        errada = json.loads(Path(montado["campanha"]).read_text(encoding="utf-8"))
        errada = copy.deepcopy(errada)
        errada["g6"]["composicao_final"]["MR"][1]["cabeca"] = 11
        (raiz / "campanha_errada.json").write_text(json.dumps(errada), encoding="utf-8")
        assert construtor.main([*_argumentos(raiz, montado), "--campanha", str(raiz / "campanha_errada.json"),
                                "--out-dir", str(raiz / "g6_g")]) == 2
        assert not (raiz / "g6_g").exists()


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
            import traceback

            traceback.print_exc()
            print(f"  FAIL  {nome}: {type(exc).__name__}: {exc}")
    print(f"\n{len(testes) - falhas - len(pulados)}/{len(testes) - len(pulados)} passaram"
          + (f"  |  {len(pulados)} PULADO(S)" if pulados else ""))
    if pulados and os.environ.get("REQUIRE_NO_SKIP"):
        print("REQUIRE_NO_SKIP: teste pulado conta como falha")
        sys.exit(1)
    sys.exit(1 if falhas else 0)
