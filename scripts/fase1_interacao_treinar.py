#!/usr/bin/env python3
"""Fase 1, cabeca com interacao: treina H(E+F) e H(E+F+BR2) (desenvolvimento exploratorio, posterior ao teste).

Especificacao, congelada antes de rodar: docs/fase1_cabeca_interacao_especificacao.md. Tudo igual ao passo 3 e ao BR v2
(release, embeddings, cinco execucoes, purgas, linhas, padronizacao no treino, regressao logistica L2, grade de C,
escolha pela macro AUROC da validation e regra revisada de convergencia). Muda a cabeca:

- H(E+F)     = E + F + S + S x F
- H(E+F+BR2) = E + F + BR2 + S + S x F + S x BR2

S e o score do braco E do passo 3, o mesmo do S+F: no treino de cada execucao, o score fora da amostra pela particao
interna do passo 3 (`diagnosticos/s_fora_da_amostra.parquet`); na validation e no teste, o score do braco E. Nenhum
ajuste novo de E. S x X e o produto de S padronizado no treino da execucao por cada coluna de X padronizada no treino;
depois, como todas as colunas, a matriz inteira e padronizada no treino. O BR2 e o bloco do BR v2, sem mudanca.

Antes de treinar, confere que o S reaproveitado e das mesmas linhas, execucoes e purgas, e interrompe se nao for:
- em cada execucao, os ids do S de treino sao os das linhas de treino, na ordem, e o fold interno e o fold da linha;
  os de validation e teste sao os desses papeis;
- media, desvio e n de treino, validation e teste batem com o `s_resumo` do S+F no selecao.json do passo 3;
- a particao interna refeita tem os mesmos tamanhos e purgas do `s_interno` gravado, e os ajustes internos usaram o C
  do braco E da execucao.

USO (notebook, no .venv do Mosaic)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1_interacao_treinar.py \\
        --entrega ~/mosaic-v1-2026-09-30 --bracos ~/artifacts/mosaic_v1/bracos_<...> \\
        --cache ~/artifacts/redesenho/g3_cache/M0 --cache ~/artifacts/redesenho/g7_cache/M0 \\
        --cache ~/artifacts/mosaic_v1/cache_M0_complemento --out-dir ~/artifacts/mosaic_v1/interacao_treino_<...>

SAIDAS (em --out-dir, que nao pode existir; gravadas em <out-dir>.tmp e renomeadas so depois do contrato):
fase1-h-e-f/ e fase1-h-e-f-br2/ (predictions.parquet, system.yaml), modelos/, selecao.json, contagens.json e
fontes.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import metricas  # noqa: E402
from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from scripts import fase1_br2_treinar as br2  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from scripts.inventario_mosaic_v1 import K  # noqa: E402

FalhaDaFase1 = bracos.FalhaDaFase1
ESPECIFICACAO = RAIZ / "docs" / "fase1_cabeca_interacao_especificacao.md"
#: Os dois bracos novos. `base_do_sistema` e o braco do passo 3 cujo system.yaml serve de molde.
BRACOS_H: dict[str, dict[str, Any]] = {
    "H(E+F)": {"id": "fase1-h-e-f", "blocos": ("e", "f"), "interacoes": ("f",), "base_do_sistema": "E+F"},
    "H(E+F+BR2)": {"id": "fase1-h-e-f-br2", "blocos": ("e", "f", "br2"), "interacoes": ("f", "br2"),
                   "base_do_sistema": "E+F+BR"},
}


def ferramentas_do_mosaic() -> dict[str, Callable[..., Any]]:
    """As do BR v2 (passo 3 mais o limite inferior do Mosaic); os testes substituem esta funcao."""
    return br2.ferramentas_do_mosaic()


# ------------------------------------------------------------------------------------------------- S reaproveitado

def carregar_s(pasta3: Path, dados: bracos.Dados) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """O S de cada execucao (treino, validation e teste), conferido contra o que o passo 3 gravou."""
    selecao = json.loads((pasta3 / "selecao.json").read_text(encoding="utf-8"))
    s_mais_f = {int(r["run"]): r for r in selecao["bracos"]["S+F"]}
    braco_e = {int(r["run"]): r for r in selecao["bracos"]["E"]}
    caminho = pasta3 / "diagnosticos" / "s_fora_da_amostra.parquet"
    treino = pd.read_parquet(caminho)
    preds_e = pd.read_parquet(pasta3 / bracos.BRACOS["E"]["id"] / "predictions.parquet")
    preds_e = preds_e.assign(variant_id=preds_e["variant_id"].astype(str), run=preds_e["run"].astype(int))
    saida, relatorio = [], []
    for run in range(K):
        idx = dados.papeis[run]
        t = treino[treino["run"].astype(int) == run]
        if (len(t) != len(idx["treino"])
                or not np.array_equal(t["variant_id"].astype(str).to_numpy(), dados.ids[idx["treino"]])):
            raise FalhaDaFase1(f"run {run}: o S de treino nao tem as linhas de treino da execucao, na ordem")
        if not np.array_equal(t["fold_interno"].to_numpy(np.int64), dados.folds[idx["treino"]]):
            raise FalhaDaFase1(f"run {run}: o fold interno do S nao e o fold das linhas de treino")
        partes = {"treino": t["score_e"].to_numpy(dtype=np.float64)}
        for papel, nome in (("validation", "validation"), ("test", "teste")):
            v = preds_e[(preds_e["run"] == run) & (preds_e["role"] == papel)]
            serie = pd.Series(v["score"].to_numpy(dtype=np.float64), index=v["variant_id"].to_numpy())
            esperado = dados.ids[idx[nome]]
            if len(serie) != len(esperado) or set(serie.index) != set(esperado):
                raise FalhaDaFase1(f"run {run}: o S de {nome} nao tem as linhas de {nome} da execucao")
            partes[nome] = serie.reindex(esperado).to_numpy()
        for nome, valores in partes.items():
            if not np.isfinite(valores).all():
                raise FalhaDaFase1(f"run {run}: S de {nome} nao finito")
        gravado = s_mais_f[run]["s_resumo"]
        for chave, nome in (("treino_fora_da_amostra", "treino"), ("validation", "validation"), ("teste", "teste")):
            g, v = gravado[chave], partes[nome]
            if (int(g["n"]) != len(v) or not np.isclose(float(g["media"]), float(v.mean()), rtol=1e-9, atol=1e-12)
                    or not np.isclose(float(g["desvio"]), float(v.std()), rtol=1e-9, atol=1e-12)):
                raise FalhaDaFase1(f"run {run}: o S de {nome} nao bate com o s_resumo do passo 3")
        internos = {int(r["fold_interno"]): r for r in s_mais_f[run]["s_interno"]}
        particao = bracos.particao_interna(dados, run)
        if set(internos) != {int(p["fold"]) for p in particao}:
            raise FalhaDaFase1(f"run {run}: folds internos diferentes dos gravados no passo 3")
        for parte in particao:
            g = internos[int(parte["fold"])]
            refeito = (len(parte["treino"]), len(parte["teste"]), int(parte["purgadas"]))
            if (int(g["treino_interno"]), int(g["teste_interno"]), int(g["purgadas_internas"])) != refeito:
                raise FalhaDaFase1(f"run {run}, fold interno {parte['fold']}: tamanhos ou purgas diferentes dos "
                                   f"gravados no passo 3")
            if not g.get("convergiu", False) or not np.isclose(float(g["C"]), float(braco_e[run]["C"])):
                raise FalhaDaFase1(f"run {run}, fold interno {parte['fold']}: ajuste interno sem convergir ou com "
                                   f"outro C que o do braco E")
        saida.append(partes)
        relatorio.append({"run": run, "n": {k: int(len(v)) for k, v in partes.items()},
                          "C_do_braco_e": float(braco_e[run]["C"]),
                          "purgadas_internas": {str(int(p["fold"])): int(p["purgadas"]) for p in particao}})
    return saida, {"arquivo": str(caminho), "sha256": sha256_do_arquivo(caminho), "execucoes": relatorio}


# ------------------------------------------------------------------------------------------------------ cabeca H

def colunas_h(dados: bracos.Dados, cfg: dict[str, Any]) -> list[str]:
    return ([c for b in cfg["blocos"] for c in dados.colunas[b]] + ["s_e"]
            + [f"s_x_{c}" for b in cfg["interacoes"] for c in dados.colunas[b]])


def padronizacao_das_interacoes(dados: bracos.Dados, cfg: dict[str, Any], treino: np.ndarray,
                                s_treino: np.ndarray) -> dict[str, tuple[Any, Any]]:
    """Media e desvio, no treino da execucao, de S e das colunas que entram nos produtos."""
    desvio_s = float(np.std(s_treino))
    z: dict[str, tuple[Any, Any]] = {"s": (float(np.mean(s_treino)), desvio_s if desvio_s >= 1e-8 else 1.0)}
    for bloco in cfg["interacoes"]:
        z[bloco] = bracos.padronizador(dados.blocos[bloco][treino])
    return z


def montar_h(dados: bracos.Dados, cfg: dict[str, Any], indices: np.ndarray, s: np.ndarray,
             z: dict[str, tuple[Any, Any]]) -> np.ndarray:
    """Matriz float64 das linhas `indices`: os blocos, S e S x X, alocada uma vez e preenchida por partes."""
    s = np.asarray(s, dtype=np.float64)
    if len(s) != len(indices):
        raise FalhaDaFase1("S precisa de um score por linha")
    larguras = [dados.blocos[b].shape[1] for b in cfg["blocos"]]
    extra = sum(dados.blocos[b].shape[1] for b in cfg["interacoes"])
    X = np.empty((len(indices), sum(larguras) + 1 + extra), dtype=np.float64)
    inicio = 0
    for bloco, largura in zip(cfg["blocos"], larguras):
        fonte = dados.blocos[bloco]
        for a in range(0, len(indices), bracos.PASSO):
            X[a:a + bracos.PASSO, inicio:inicio + largura] = fonte[indices[a:a + bracos.PASSO]]
        inicio += largura
    X[:, inicio] = s
    inicio += 1
    s_padronizado = (s - z["s"][0]) / z["s"][1]
    for bloco in cfg["interacoes"]:
        media, desvio = z[bloco]
        largura = dados.blocos[bloco].shape[1]
        X[:, inicio:inicio + largura] = s_padronizado[:, None] * ((dados.blocos[bloco][indices] - media) / desvio)
        inicio += largura
    return X


def treinar_h(dados: bracos.Dados, run: int, cfg: dict[str, Any], s: dict[str, np.ndarray],
              ajustador: bracos.Ajustador) -> dict[str, Any]:
    """A logica de `bracos.treinar_braco` (padroniza no treino, ajusta a grade, escolhe C na validation, pontua
    validation e teste com o escolhido), com a matriz da cabeca H."""
    idx = dados.papeis[run]
    y_tr = dados.y[idx["treino"]]
    if len(np.unique(y_tr)) < 2:
        raise FalhaDaFase1(f"run {run}: treino sem as duas classes")
    z = padronizacao_das_interacoes(dados, cfg, idx["treino"], s["treino"])
    X = montar_h(dados, cfg, idx["treino"], s["treino"], z)
    media, desvio = bracos.padronizador(X)
    grade = ajustador(bracos.padronizar(X, media, desvio), y_tr, bracos.GRADE_C)
    del X
    Xva = bracos.padronizar(montar_h(dados, cfg, idx["validation"], s["validation"], z), media, desvio)
    y_va, p_va = dados.y[idx["validation"]], dados.paineis[idx["validation"]]
    for r in grade:
        if not r.get("convergiu", False):
            r["macro_validacao"] = None
            r["auroc_validacao_por_painel"] = {}
            r["motivo_de_exclusao"] = "ajuste nao convergiu"
            continue
        painel = metricas.por_painel(bracos.pontuar(Xva, r["coef"], r["intercepto"]), y_va, p_va)
        r["macro_validacao"] = metricas.macro(painel)
        r["auroc_validacao_por_painel"] = {nome: v["auroc"] for nome, v in painel.items()}
    escolhido = grade[bracos.escolher_c(grade)]
    Xte = bracos.padronizar(montar_h(dados, cfg, idx["teste"], s["teste"], z), media, desvio)
    return {"C": escolhido["C"], "macro_validacao": escolhido["macro_validacao"], "coef": escolhido["coef"],
            "intercepto": escolhido["intercepto"], "media": media, "desvio": desvio, "colunas": colunas_h(dados, cfg),
            "z": z, "score_validation": bracos.pontuar(Xva, escolhido["coef"], escolhido["intercepto"]),
            "score_teste": bracos.pontuar(Xte, escolhido["coef"], escolhido["intercepto"]),
            "grade": [{k: v for k, v in r.items() if k != "coef"} for r in grade]}


def sistema_h(braco: str, proveniencia: dict[str, Any]) -> dict[str, Any]:
    cfg = BRACOS_H[braco]
    conteudo = bracos.sistema(cfg["base_do_sistema"], proveniencia)
    conteudo["id"] = cfg["id"]
    fontes = ["gnomAD v4.1 joint: bloco de frequencia oficial do Mosaic (frequency_features)"]
    if "br2" in cfg["blocos"]:
        fontes.append("ABraOM WGS-1171 (variant-annotations), BR v2: estados, faixas de AC, AF = AC/AN, limite inferior "
                      "de Clopper-Pearson, FILTER PASS, AN e excesso sustentado sobre o AF_joint")
    conteudo["exposure"]["population_frequency"]["sources"] = fontes
    conteudo["fase1"].update({
        "braco": braco, "blocos": [*cfg["blocos"], "s", *(f"s_x_{b}" for b in cfg["interacoes"])],
        "natureza": "desenvolvimento exploratorio posterior ao teste da Fase 1",
        "s": ("S = score do braco E do passo 3: no treino, fora da amostra pela particao interna (tres folds, C do "
              "braco E, purga de 4.096 bp); na validation e no teste, o do modelo E da execucao"),
        "interacoes": "S padronizado no treino vezes cada coluna do bloco padronizada no treino",
        "especificacao_sha256": proveniencia["especificacao_sha256"]})
    return conteudo


# ---------------------------------------------------------------------------------------------------------- main

def preparar(entrega: Path, pasta3: Path, caches: list[Path],
             ferramentas: dict[str, Callable[..., Any]]) -> dict[str, Any]:
    """Release, linhas, blocos (com o BR2) e o S conferido: tudo o que o treino usa."""
    raiz = entrega / bracos.RELEASE
    identidade = ferramentas["verificar_identidade"](raiz, config_dir=entrega / "config")
    protocolo_sha256 = sha256_do_arquivo(entrega / bracos.PROTOCOLO_DE_ESTUDOS)
    bracos.conferir_entrega(identidade, protocolo_sha256)
    leituras.conferir_fontes(json.loads((pasta3 / "fontes.json").read_text(encoding="utf-8")), raiz)
    leituras.conferir_selecao(json.loads((pasta3 / "selecao.json").read_text(encoding="utf-8")))
    release = br2.carregar_release(raiz)
    por_run, contagens = bracos.linhas_por_execucao(release)
    if bracos.CONTAGENS_DO_GUIA_RUN0 is not None and contagens[0]["guia"] != bracos.CONTAGENS_DO_GUIA_RUN0:
        raise FalhaDaFase1(f"run 0 nao bate com o guia de splits: {contagens[0]['guia']}")
    elegiveis = release[release["sequence_eligible"]].reset_index(drop=True)
    conferencia_f = bracos.conferir_frequencia_oficial(elegiveis, ferramentas["frequencia_oficial"])
    contagens_br = br2.conferir_contagens(elegiveis)
    bloco = br2.bloco_br2(elegiveis, ferramentas["af_lower_bound"])
    matriz_e, fontes_dos_caches = bracos.carregar_leitura(caches, raiz, elegiveis["variant_id"].astype(str).to_numpy())
    dados = bracos.montar_dados(elegiveis, matriz_e, por_run)
    dados.blocos["br2"], dados.colunas["br2"] = bloco.to_numpy(np.float64), list(bloco.columns)
    s_por_run, conferencia_s = carregar_s(pasta3, dados)
    return {"raiz": raiz, "identidade": identidade, "protocolo_sha256": protocolo_sha256, "release": release,
            "contagens": contagens, "conferencia_f": conferencia_f, "contagens_br": contagens_br, "dados": dados,
            "fontes_dos_caches": fontes_dos_caches, "s": s_por_run, "conferencia_s": conferencia_s}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path)
    parser.add_argument("--bracos", required=True, type=Path, help="pasta do passo 3 (S, braco E, release e linhas)")
    parser.add_argument("--cache", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    entrega, pasta3, destino = args.entrega.expanduser(), args.bracos.expanduser(), args.out_dir.expanduser()
    temporaria = destino.with_name(destino.name + ".tmp")
    if destino.exists() or temporaria.exists():
        return bracos._falhar([f"{destino} ou {temporaria} ja existe; a cabeca H grava sempre numa pasta nova"])
    if not ESPECIFICACAO.is_file():
        return bracos._falhar([f"sem a especificacao {ESPECIFICACAO}: ela vem antes de rodar"])
    try:
        ferramentas = ferramentas_do_mosaic()
    except ImportError as exc:
        return bracos._falhar([f"sem o pacote mosaic ou o sklearn ({exc}); rodar no .venv do Mosaic (uv run --project)"])

    inicio = time.perf_counter()
    try:
        p = preparar(entrega, pasta3, [c.expanduser() for c in args.cache], ferramentas)
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return bracos._falhar([f"{type(exc).__name__}: {exc}"])
    dados = p["dados"]
    print(f"S conferido (ids, execucao, s_resumo, purgas e C internos): "
          + "; ".join(f"run {r['run']} {r['n']}" for r in p["conferencia_s"]["execucoes"]), flush=True)

    resultados: dict[str, list[dict[str, Any]]] = {braco: [] for braco in BRACOS_H}
    try:
        for braco, cfg in BRACOS_H.items():
            for run in range(K):
                t0 = time.perf_counter()
                r = treinar_h(dados, run, cfg, p["s"][run], bracos.AJUSTADOR)
                resultados[braco].append(r)
                iteracoes = ", ".join(f"{g['C']:g}:{g.get('n_iter')}" for g in r["grade"])
                print(f"[{braco} run {run}] C={r['C']:g} macro_validacao={r['macro_validacao']:.4f} (iteracoes "
                      f"{iteracoes}; {time.perf_counter() - t0:.1f} s)", flush=True)
        preds = {braco: bracos.predicoes(dados, rs) for braco, rs in resultados.items()}
        leituras.conferir_linhas(preds, p["release"])   # as mesmas linhas de validation e teste do passo 3
    except (FalhaDaFase1, ValueError) as exc:
        return bracos._falhar([f"{type(exc).__name__}: {exc}"])

    raiz, identidade = p["raiz"], p["identidade"]
    primeira = next(iter(p["fontes_dos_caches"].values()))["identidade"]
    proveniencia = {
        "entrega": str(entrega), "passo3": str(pasta3),
        "identidade_do_release": {k: identidade.get(k) for k in ("release_id", "version", "release_identity_hash",
                                                                  "protocol_hash")},
        "study_protocol_sha256": p["protocolo_sha256"],
        "arquivos_do_release": {nome: sha256_do_arquivo(raiz / nome) for nome in
                                ("clinical-variants.parquet", bracos.VISTA, "evaluation-panels.parquet",
                                 "variant-annotations.parquet")},
        "conferencia_do_bloco_f": p["conferencia_f"], "caches": p["fontes_dos_caches"], "s": p["conferencia_s"],
        "r03": {"extracao": bracos.EXTRACAO, "checkpoint_sha256": primeira.get("checkpoint_sha256"),
                "versao_do_extrator": primeira.get("versao_do_extrator"), "janela_bp": primeira.get("janela_bp")},
        "especificacao_sha256": sha256_do_arquivo(ESPECIFICACAO), "script_sha256": sha256_do_arquivo(Path(__file__)),
        "revisao": bracos.revisao_do_repositorio(), "versoes": bracos.versoes(), "receita": bracos.RECEITA,
        "segundos": round(time.perf_counter() - inicio, 1),
    }
    temporaria.mkdir(parents=True)
    try:
        (temporaria / "modelos").mkdir()
        selecao: dict[str, list[dict[str, Any]]] = {}
        for braco, cfg in BRACOS_H.items():
            pasta = temporaria / cfg["id"]
            pasta.mkdir()
            preds[braco].to_parquet(pasta / "predictions.parquet", index=False)
            bracos.escrever_sistema(pasta, sistema_h(braco, proveniencia))
            selecao[braco] = []
            for run, r in enumerate(resultados[braco]):
                padronizacao = {f"z_{nome}_media": np.asarray(m, dtype=np.float64) for nome, (m, _) in r["z"].items()}
                padronizacao.update({f"z_{nome}_desvio": np.asarray(d, dtype=np.float64)
                                     for nome, (_, d) in r["z"].items()})
                np.savez(temporaria / "modelos" / f"{cfg['id']}_run{run}.npz", coef=r["coef"],
                         intercepto=np.float64(r["intercepto"]), media=r["media"], desvio=r["desvio"],
                         C=np.float64(r["C"]), colunas=np.array(r["colunas"]), braco=np.array(braco),
                         run=np.int64(run), **padronizacao)
                selecao[braco].append({"run": run, "C": r["C"], "macro_validacao": r["macro_validacao"],
                                       "grade": r["grade"]})
        (temporaria / "selecao.json").write_text(json.dumps({"receita": bracos.RECEITA, "bracos": selecao},
                                                            indent=2, ensure_ascii=False), encoding="utf-8")
        (temporaria / "contagens.json").write_text(json.dumps(
            {"abraom": p["contagens_br"], "execucoes": p["contagens"], "s": p["conferencia_s"],
             "colunas": {braco: resultados[braco][0]["colunas"][-(1 + sum(dados.blocos[b].shape[1]
                                                                          for b in cfg["interacoes"])):]
                         for braco, cfg in BRACOS_H.items()}}, indent=2, ensure_ascii=False), encoding="utf-8")
        (temporaria / "fontes.json").write_text(json.dumps(proveniencia, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
        conferidas = {cfg["id"]: bracos.conferir_contrato(ferramentas, temporaria / cfg["id"], p["release"], identidade,
                                                          p["protocolo_sha256"]) for cfg in BRACOS_H.values()}
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return bracos._falhar([f"{type(exc).__name__}: {exc} (saida parcial deixada em {temporaria} para inspecao)"])
    temporaria.rename(destino)
    print("\nmacro da validation (a mesma usada para escolher C):")
    for braco, rs in resultados.items():
        print(f"  {braco:11s} " + "  ".join(f"run{i} {r['macro_validacao']:.4f} (C={r['C']:g})" for i, r in enumerate(rs)))
    print(f"PASSOU: {len(conferidas)} bracos no contrato do Mosaic ({conferidas}) | {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
