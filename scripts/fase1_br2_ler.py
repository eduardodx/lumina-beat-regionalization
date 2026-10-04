#!/usr/bin/env python3
"""Fase 1, BR v2: leituras dos bracos E+F, E+F+BR e E+F+BR2 (desenvolvimento exploratorio, posterior ao teste).

Especificacao: docs/fase1_br2_especificacao.md. Mede com o MESMO codigo do passo 4 e do diagnostico, nos pares
E+F -> E+F+BR (reproducao), E+F -> E+F+BR2 e E+F+BR -> E+F+BR2:

- nucleo (contraste oficial e AUROC por painel), proxies (com interacao), beneficio, P-BR no limiar MCC original
  (perdas e ganhos separados) e as 13 criticas;
- o diagnostico das perdas e ganhos, com a especificidade equivalente refeita na validation e a obtida por execucao;
- quantas das P-BR perdidas pelo BR original o BR2 recupera, quais perdas sao novas e em que faixa de AC estao;
- os coeficientes padronizados das features BR e BR2, por execucao (leitura, nao ablacao);
- AUXILIAR, sem bloquear: as saidas populacionais nativas do R03 em `cabecas_172`, com o layout conferido nos dados,
  comparadas com as frequencias observadas. Nao sao tratadas como AF calibrada.

Antes de medir, confere a selecao dos dois treinos (regra revisada de convergencia), as fontes, as linhas e os
limiares: os de E+F e E+F+BR tem de ser os do passo 4, e os de E+F+BR2, os do avaliador oficial.

USO (notebook, no .venv do Mosaic)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1_br2_ler.py \\
        --entrega ~/mosaic-v1-2026-09-30 --bracos ~/artifacts/mosaic_v1/bracos_<...> \\
        --br2 ~/artifacts/mosaic_v1/br2_<...> --leituras ~/artifacts/mosaic_v1/leituras_<...> \\
        --avaliacao-br2 ~/artifacts/mosaic_v1/avaliacao_br2_<...> --cache <3 caches> --out-dir <nova>

SAIDAS (em --out-dir, que nao pode existir): br2.json, br2.md e perdas_e_ganhos.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import metricas  # noqa: E402
from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_diagnostico as diag  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from scripts.fase1_br2_treinar import BANDAS, BRACO, ID  # noqa: E402
from scripts.inventario_mosaic_v1 import K, ler_criticas  # noqa: E402

FalhaDaFase1 = bracos.FalhaDaFase1
IDS = {"E+F": "fase1-e-f", "E+F+BR": "fase1-e-f-br", BRACO: ID}
PARES_BR2: tuple[tuple[str, str], ...] = (("E+F", "E+F+BR"), ("E+F", BRACO), ("E+F+BR", BRACO))
ORIGINAL, NOVO = leituras.nome_do_par("E+F", "E+F+BR"), leituras.nome_do_par("E+F", BRACO)

#: Layout de `cabecas_172` no R03 (eval/embedding_probe/rich.py e config/lumina_r03_base.json do lumina-inference):
#: W.Delta das cabecas lineares (68) | deltas das MLP (10) | valores na REF das lineares (68) e das MLP (10) | 16 do
#: one-hot (4 da base REF + 12 da substituicao). Conferido nos dados por `conferir_layout`.
LINEARES = (("mlm_head", 4), ("conservation_scalar_head", 3), ("conservation_bin_head", 16), ("region_head", 5),
            ("counterfactual_snv_head", 32), ("population_af_head", 4), ("population_observed_head", 4))
MLP_TOTAL, SUBST = 10, 16
BASES = ("A", "C", "G", "T")


def _inicio(nome: str) -> int:
    return sum(n for cabeca, n in LINEARES[:[c for c, _ in LINEARES].index(nome)])


LIN_TOTAL = sum(n for _, n in LINEARES)
REF_INICIO = LIN_TOTAL + MLP_TOTAL
SUBST_INICIO = REF_INICIO + LIN_TOTAL + MLP_TOTAL


def coeficientes(pasta: Path, ident: str, prefixo: str) -> dict[str, Any]:
    """Coeficientes (sobre features padronizadas) das colunas com `prefixo`, por execucao e a media."""
    por_run = {}
    for run in range(K):
        modelo = np.load(pasta / "modelos" / f"{ident}_run{run}.npz")
        colunas = [str(c) for c in modelo["colunas"]]
        por_run[str(run)] = {c: float(v) for c, v in zip(colunas, modelo["coef"]) if c.startswith(prefixo)}
    nomes = list(next(iter(por_run.values())))
    return {"por_execucao": por_run,
            "media": {c: float(np.mean([por_run[r][c] for r in por_run])) for c in nomes},
            "nota": "coeficientes da regressao logistica sobre features padronizadas no treino; positivo empurra para "
                    "patogenica"}


def recuperacao(p_br: dict[str, Any], equivalente: dict[str, Any], release: pd.DataFrame) -> dict[str, Any]:
    """As P-BR perdidas pelo BR original que o BR2 recupera, as perdas novas e a faixa de AC de cada grupo."""
    ac = release.set_index("variant_id")["abraom_ac"] if "abraom_ac" in release else None

    def faixas(ids: set[str]) -> dict[str, int]:
        if ac is None:
            return {}
        valores = ac.reindex(sorted(ids)).to_numpy(dtype=float)
        return {nome: int(((valores >= baixo) & (valores <= (alto if alto is not None else np.inf))).sum())
                for baixo, alto, nome in BANDAS}

    saida = {}
    for leitura, fonte in (("limiar_mcc_original", p_br["pares"]), ("especificidade_equivalente", equivalente)):
        original, novo = set(fonte[ORIGINAL]["ids_perdidas"]), set(fonte[NOVO]["ids_perdidas"])
        saida[leitura] = {"perdidas_pelo_br_original": len(original), "perdidas_pelo_br2": len(novo),
                          "recuperadas_pelo_br2": len(original - novo), "mantidas": len(original & novo),
                          "novas_no_br2": len(novo - original),
                          "faixas_de_ac": {"recuperadas": faixas(original - novo), "mantidas": faixas(original & novo),
                                           "novas": faixas(novo - original)},
                          "ids": {"recuperadas": sorted(original - novo), "novas": sorted(novo - original)}}
    return saida


# ---------------------------------------------------------------------------------------- cabecas nativas

def conferir_layout(cabecas: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """O one-hot da base REF tem de bater com a variante em toda linha; o argmax do MLM na REF e relatado."""
    if cabecas.shape[1] != SUBST_INICIO + SUBST:
        raise FalhaDaFase1(f"cabecas_172 com {cabecas.shape[1]} colunas, esperado {SUBST_INICIO + SUBST}")
    indice = np.array([BASES.index(b) for b in ref])
    one_hot = cabecas[:, SUBST_INICIO:SUBST_INICIO + 4]
    if not (np.isclose(one_hot.sum(axis=1), 1.0).all() and (one_hot.argmax(axis=1) == indice).all()):
        raise FalhaDaFase1("o one-hot da base REF nao bate com a variante: o layout de cabecas_172 nao e o esperado")
    mlm = cabecas[:, REF_INICIO + _inicio("mlm_head"):REF_INICIO + _inicio("mlm_head") + 4]
    return {"subst_ref_confere": True, "mlm_na_ref_aponta_a_base_ref": float((mlm.argmax(axis=1) == indice).mean())}


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def cabecas_nativas(cabecas: np.ndarray, elegiveis: pd.DataFrame) -> dict[str, Any]:
    """Saida das cabecas populacionais do R03 na REF, no alelo ALT, contra as frequencias observadas."""
    layout = conferir_layout(cabecas, elegiveis["ref"].astype(str).to_numpy())
    alt = np.array([BASES.index(b) for b in elegiveis["alt"].astype(str)])
    linhas = np.arange(len(alt))
    af_inicio = REF_INICIO + _inicio("population_af_head")
    obs_inicio = REF_INICIO + _inicio("population_observed_head")
    af_pred = cabecas[linhas, af_inicio + alt].astype(np.float64)
    obs_pred = cabecas[linhas, obs_inicio + alt].astype(np.float64)
    estado_g = elegiveis["gnomad_status"].fillna("unknown").to_numpy()
    af_g = pd.to_numeric(elegiveis["gnomad_v4_af"]).to_numpy(dtype=float)
    achada = ~np.isin(estado_g, ["not_found", "ac0", "unknown"]) & (af_g > 0)
    log_g = np.log10(np.where(achada, af_g, 1.0))
    inclinacao, intercepto = (np.polyfit(log_g[achada], af_pred[achada], 1) if achada.sum() >= 3 else (None, None))
    estado_a = elegiveis["abraom_status"].fillna("").astype(str).to_numpy()
    medida_a = np.isin(estado_a, ["present", "not_found"])
    rotulo_g = np.isin(estado_g, ["not_found"])
    medida_g = np.isin(estado_g, ["not_found"]) | achada
    return {
        "layout": layout,
        "colunas": {"population_af_head_ref": [af_inicio, af_inicio + 4], "population_observed_head_ref":
                    [obs_inicio, obs_inicio + 4], "alelo": "a saida do alelo ALT (A, C, G, T)"},
        "af_pred": {"quantis": {q: float(np.quantile(af_pred, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
                    "spearman_com_log10_af_gnomad": _spearman(af_pred[achada], log_g[achada]),
                    "reta_contra_log10_af_gnomad": {"inclinacao": None if inclinacao is None else float(inclinacao),
                                                    "intercepto": None if intercepto is None else float(intercepto)},
                    "n_achadas_no_gnomad": int(achada.sum())},
        "observed_pred": {
            "auroc_achada_no_gnomad": metricas.auroc(obs_pred[medida_g], (~rotulo_g[medida_g]).astype(int)),
            "auroc_presente_no_abraom": metricas.auroc(obs_pred[medida_a], (estado_a[medida_a] == "present").astype(int)),
            "auroc_af_pred_presente_no_abraom": metricas.auroc(af_pred[medida_a],
                                                               (estado_a[medida_a] == "present").astype(int))},
        "nota": "saidas nativas na REF do R03; escala e transformacao do alvo nao conferidas: nao e AF calibrada",
    }


# -------------------------------------------------------------------------------------------------- relatorio

def secao_br2(d: dict[str, Any]) -> str:
    L = ["# BR v2: recuperação das perdas, coeficientes e cabeças nativas", ""]
    titulos = {"limiar_mcc_original": "no limiar MCC original", "especificidade_equivalente":
               "na especificidade equivalente (posterior)"}
    for leitura, r in d["recuperacao"].items():
        L += [f"## Recuperação {titulos.get(leitura, leitura)}", "",
              f"- Perdidas pelo BR original: {r['perdidas_pelo_br_original']}; pelo BR2: {r['perdidas_pelo_br2']}",
              f"- Recuperadas pelo BR2: {r['recuperadas_pelo_br2']} (faixas de AC {r['faixas_de_ac']['recuperadas']})",
              f"- Mantidas: {r['mantidas']} (faixas {r['faixas_de_ac']['mantidas']}); novas no BR2: {r['novas_no_br2']} "
              f"(faixas {r['faixas_de_ac']['novas']})", ""]
    L += ["## Coeficientes médios (features padronizadas; positivo empurra para patogênica)", ""]
    for nome, bloco in d["coeficientes"].items():
        L.append(f"- **{nome}:** " + "; ".join(f"{c} {v:+.3f}" for c, v in bloco["media"].items()))
    nativas = d["cabecas_nativas"]
    L += ["", "## Cabeças populacionais nativas (auxiliar)", ""]
    if "falhou" in nativas:
        L.append(f"- Não calculado: {nativas['falhou']}")
    else:
        af, obs = nativas["af_pred"], nativas["observed_pred"]
        L += [f"- Layout conferido; o MLM na REF aponta a base REF em {nativas['layout']['mlm_na_ref_aponta_a_base_ref']:.3f}",
              f"- `population_af_head` × log10 AF do gnomAD: Spearman {af['spearman_com_log10_af_gnomad']}; reta "
              f"{af['reta_contra_log10_af_gnomad']} ({af['n_achadas_no_gnomad']} achadas)",
              f"- `population_observed_head`: AUROC achada no gnomAD {obs['auroc_achada_no_gnomad']}; presente no ABraOM "
              f"{obs['auroc_presente_no_abraom']}", f"- {nativas['nota']}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path)
    parser.add_argument("--bracos", required=True, type=Path, help="pasta do passo 3 (E+F e E+F+BR)")
    parser.add_argument("--br2", required=True, type=Path, help="pasta do treino do BR v2")
    parser.add_argument("--leituras", required=True, type=Path, help="pasta do passo 4 (limiares e perdas)")
    parser.add_argument("--avaliacao-br2", type=Path, help="--output-root do evaluate_candidate.py no braco novo")
    parser.add_argument("--cache", action="append", default=[], type=Path, help="para o diagnostico das cabecas")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--replicas", type=int, default=leituras.REPLICAS)
    args = parser.parse_args(argv)
    entrega, pasta3, pasta_br2 = args.entrega.expanduser(), args.bracos.expanduser(), args.br2.expanduser()
    pasta4, destino = args.leituras.expanduser(), args.out_dir.expanduser()
    if destino.exists():
        return leituras._falhar([f"{destino} ja existe; as leituras do BR v2 gravam sempre numa pasta nova"])
    try:
        ferramentas = leituras.ferramentas_do_mosaic()
    except ImportError as exc:
        return leituras._falhar([f"sem o pacote mosaic ({exc}); rodar no .venv do Mosaic (uv run --project)"])
    inicio = time.perf_counter()

    def avisar(texto: str) -> None:
        print(f"[{time.perf_counter() - inicio:7.1f} s] {texto}", flush=True)

    raiz = entrega / bracos.RELEASE
    try:
        identidade = ferramentas["verificar_identidade"](raiz, config_dir=entrega / "config")
        protocolo_sha256 = sha256_do_arquivo(entrega / bracos.PROTOCOLO_DE_ESTUDOS)
        bracos.conferir_entrega(identidade, protocolo_sha256)
        release = leituras.carregar_release(raiz)
        release = release.merge(pd.read_parquet(raiz / "variant-annotations.parquet", columns=["variant_id", "abraom_ac"]),
                                on="variant_id", how="left", validate="one_to_one")
        for pasta in (pasta3, pasta_br2):
            leituras.conferir_fontes(json.loads((pasta / "fontes.json").read_text(encoding="utf-8")), raiz)
            leituras.conferir_selecao(json.loads((pasta / "selecao.json").read_text(encoding="utf-8")))
        tres = leituras.carregar_predicoes(pasta3)
        novo = pd.read_parquet(pasta_br2 / ID / "predictions.parquet")
        preds = {"E+F": tres["E+F"], "E+F+BR": tres["E+F+BR"],
                 BRACO: novo.assign(variant_id=novo["variant_id"].astype(str), run=novo["run"].astype(int))}
        conferencia_das_linhas = leituras.conferir_linhas(preds, release)
        rotulo = release.set_index("variant_id")["binary_label"].astype(int)
        limiares = leituras.limiares(preds, rotulo, ferramentas["calibrate_threshold"])
        lidas4 = json.loads((pasta4 / "leituras.json").read_text(encoding="utf-8"))
        for braco in ("E+F", "E+F+BR"):
            gravado = {int(r): float(t) for r, t in lidas4["limiares"][braco].items()}
            if any(not np.isclose(gravado[r], limiares[braco][r], rtol=0, atol=1e-12) for r in range(K)):
                raise FalhaDaFase1(f"{braco}: limiares diferentes dos do passo 4")
        oficial = (leituras.avaliador_oficial(args.avaliacao_br2.expanduser(), {BRACO: limiares[BRACO]}, ids=IDS)
                   if args.avaliacao_br2 else None)
        membros = pd.read_parquet(raiz / leituras.MEMBERSHIP)
        lista_de_criticas = ler_criticas(entrega / leituras.CRITICAS)
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return leituras._falhar([f"{type(exc).__name__}: {exc}"])
    avisar(f"entradas conferidas: {conferencia_das_linhas['linhas_por_braco']:,} linhas por braco; limiares de E+F e "
           f"E+F+BR iguais aos do passo 4; {'limiar do BR2 conferido com o avaliador' if oficial else 'sem avaliador'}")

    chamadas = {braco: leituras.chamadas_de_teste(p, limiares[braco]) for braco, p in preds.items()}
    try:
        lido: dict[str, Any] = {
            "formato": "fase1_br2_v1",
            "declaracao": {"pares": [leituras.nome_do_par(b, n) for b, n in PARES_BR2], "delta": "novo - base",
                           "replicas": args.replicas, "seed": leituras.SEED,
                           "natureza": "desenvolvimento exploratorio posterior ao teste; especificacao em "
                                       "docs/fase1_br2_especificacao.md"},
            "limiares": {b: {str(r): t for r, t in v.items()} for b, v in limiares.items()},
            "avaliador_oficial": oficial,
        }
        lido["nucleo"] = leituras.nucleo(ferramentas, raiz, release, preds, replicas=args.replicas, seed=leituras.SEED,
                                         avisar=avisar, pares=PARES_BR2, descritivos=(), ids=IDS)
        lido["beneficio"] = leituras.beneficio(ferramentas, raiz, release, preds, chamadas, replicas=args.replicas,
                                               seed=leituras.SEED, avisar=avisar, pares=PARES_BR2, ids_dos_sistemas=IDS)
        lido["p_br"] = leituras.p_br(release, chamadas, lista_de_criticas, pares=PARES_BR2)
        reproduzido = lido["p_br"]["pares"][ORIGINAL]
        for chave in ("ids_perdidas", "ids_ganhas"):
            if set(reproduzido[chave]) != set(lidas4["p_br"]["pares"][ORIGINAL][chave]):
                raise FalhaDaFase1(f"{ORIGINAL}: {chave} nao reproduz o passo 4")
        lido["criticas"] = leituras.criticas_por_braco(lista_de_criticas, release, chamadas, pares=PARES_BR2)
        pbr = diag.carregar_p_br(raiz)
        benignas = diag.benignas_da_validacao(preds, rotulo)
        diagnostico, tabelas = diag.diagnosticar(pbr, chamadas, benignas, PARES_BR2, lido["p_br"]["pares"])
        lido["diagnostico"] = {"bracos": list(chamadas), "pares": diagnostico,
                               "criticas": diag.criticas_com_fpr(lista_de_criticas, raiz, chamadas, benignas)}
        lido["recuperacao"] = recuperacao(lido["p_br"], {n: p["especificidade_equivalente"]
                                                         for n, p in diagnostico.items()}, release)
        lido["coeficientes"] = {"E+F+BR": coeficientes(pasta3, IDS["E+F+BR"], "br_"),
                                BRACO: coeficientes(pasta_br2, ID, "br2_")}
        avisar("proxies (o mais demorado)")
        lido["proxies"] = leituras.proxies(membros, preds, replicas=args.replicas, seed=leituras.SEED, avisar=avisar,
                                           pares=PARES_BR2)
    except (FalhaDaFase1, leituras.estudos.EstudoInvalido, ValueError, KeyError) as exc:
        return leituras._falhar([f"{type(exc).__name__}: {exc}"])
    try:
        avisar("cabecas populacionais nativas (auxiliar)")
        elegiveis = release[release["sequence_eligible"]].reset_index(drop=True)
        cabecas, _ = bracos.carregar_leitura([p.expanduser() for p in args.cache], raiz,
                                             elegiveis["variant_id"].astype(str).to_numpy(), extracao="cabecas_172")
        lido["cabecas_nativas"] = cabecas_nativas(cabecas, elegiveis)
    except (FalhaDaFase1, ValueError, KeyError, OSError, IndexError) as exc:   # auxiliar: registra e segue
        lido["cabecas_nativas"] = {"falhou": f"{type(exc).__name__}: {exc}"}
    lido["proveniencia"] = {
        "entrega": str(entrega), "bracos": f"{pasta3} (E+F, E+F+BR) e {pasta_br2} ({BRACO})", "leituras": str(pasta4),
        "script_sha256": sha256_do_arquivo(Path(__file__)), "revisao": bracos.revisao_do_repositorio(),
        "versoes": bracos.versoes(), "segundos": round(time.perf_counter() - inicio, 1)}

    destino.mkdir(parents=True)
    (destino / "br2.json").write_text(json.dumps(lido, indent=2, ensure_ascii=False, default=leituras._json),
                                      encoding="utf-8")
    relido = json.loads((destino / "br2.json").read_text(encoding="utf-8"))
    texto = (leituras.resumo(relido, titulo="# Fase 1, BR v2: leituras (desenvolvimento, posterior ao teste)") + "\n"
             + diag.relatorio(relido["diagnostico"], titulo="# BR v2: diagnóstico das perdas e ganhos de P-BR") + "\n"
             + secao_br2(relido))
    (destino / "br2.md").write_text(texto, encoding="utf-8")
    pd.concat(tabelas, ignore_index=True).drop(columns=["core_purged_runs"], errors="ignore").to_csv(
        destino / "perdas_e_ganhos.csv", index=False)
    leituras._imprimir(texto)
    print(f"PASSOU: leituras do BR v2 em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
