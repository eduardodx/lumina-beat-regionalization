#!/usr/bin/env python3
"""Fase 1, passo 4: as leituras dos seis bracos do passo 3 (docs/fase1_r03_congelado.md §3 e passo 4).

So mede: nao treina, nao escolhe C e so calibra o limiar pela regra do avaliador. Tudo e desenvolvimento.

- Nucleo (teste gold de 4 kb, cinco execucoes): o contraste pareado OFICIAL do Mosaic (`contrasts.paired_contrast`:
  macro AUPRC por fold e MCC, bootstrap conjunto por cluster com o limiar refeito em cada replica) e, por painel,
  AUROC e AUPRC no teste das cinco execucoes juntas, com bootstrap por `overlap_cluster_id`.
- Proxies brasileiros (`studies/brazilian-proxies/membership.parquet`): as regras do track brazil pelo consumidor da
  campanha anterior (`eval/campanha/estudos.py`): coorte completo, pareados, controles e interacao, com
  `cluster_conjunto` como unidade principal e `par` como sensibilidade. Cada membro e pontuado pela execucao que o
  testa (cross-fitting); um membro pode ter treinado as cabecas de outras execucoes.
- Beneficio (teste gold de 4 kb presente no ABraOM, a coorte `regional_clinical` do Mosaic): o contraste oficial
  das chamadas (`contrasts.paired_call_contrast`) e AUROC/AUPRC descritivos.
- P-BR: a conta do `scripts/evaluate_safety.py`, com o braco base no lugar de R0 e o novo no de R1.
- As 13 criticas, uma a uma, inclusive as que nao estao no release ou no ABraOM.

Pares (delta = novo - base): F -> F+BR e E+F -> E+F+BR (a pergunta regional), F -> E+F (o R03 sobre o prior global)
e E -> E+F (a frequencia sobre a representacao). E+F -> S+F so no nucleo, como descricao.

Chamada de cada variante de teste: score >= limiar da sua execucao. O limiar e o de `stats.calibrate_threshold` do
Mosaic na validation da execucao (gold, sem purgas, todos os paineis), a regra do avaliador do candidato. Com
--avaliacao, ele e conferido contra o `validation-thresholds.parquet` gravado pelo avaliador.

Antes de medir, confere:
- que o passo 3 vale com a regra revisada de convergencia: o C escolhido e o mesmo quando os ajustes que nao
  convergiram ficam de fora, e todo S interno convergiu;
- que as predicoes sao do release em disco;
- que cada braco tem exatamente as linhas esperadas.

USO (notebook, no .venv do Mosaic)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1_leituras.py \\
        --entrega ~/mosaic-v1-2026-09-30 --bracos ~/artifacts/mosaic_v1/bracos_<...> \\
        --avaliacao ~/artifacts/mosaic_v1/avaliacao_<...> --out-dir ~/artifacts/mosaic_v1/leituras_<...>

SAIDAS (em --out-dir, que nao pode existir): leituras.json (tudo) e resumo.md (as tabelas principais).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import estudos, metricas  # noqa: E402
from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts.inventario_mosaic_v1 import K, ler_criticas  # noqa: E402
from scripts.inventario_mosaic_v1 import criticas as criticas_do_release  # noqa: E402

FalhaDaFase1 = bracos.FalhaDaFase1
MEMBERSHIP = "studies/brazilian-proxies/membership.parquet"
CRITICAS = "config/critical-variants-br.yaml"
JANELA_BP = bracos.JANELA_BP
#: (base, novo); delta = novo - base.
PARES: tuple[tuple[str, str], ...] = (("F", "F+BR"), ("E+F", "E+F+BR"), ("F", "E+F"), ("E", "E+F"))
PAR_DESCRITIVO_DO_NUCLEO = ("E+F", "S+F")
REPLICAS = 1000
SEED = 20260901
MARGEM_DA_PERDA = 0.01   # o padrao de scripts/evaluate_safety.py
UNIDADE_PRINCIPAL = "cluster_conjunto"
PAINEIS = metricas.PAINEIS_DE_DISCRIMINACAO + metricas.PAINEIS_DE_GUARDA
METRICAS_DO_AVALIADOR = {"auroc": "auroc", "auprc_average_precision": "auprc", "macro_auroc": "macro_auroc"}
IDS_DOS_BRACOS = {cfg["id"]: braco for braco, cfg in bracos.BRACOS.items()}


def nome_do_par(base: str, novo: str) -> str:
    return f"{base} -> {novo}"


# ----------------------------------------------------------------------------------------- entradas e conferencias

def carregar_release(raiz: Path) -> pd.DataFrame:
    """O release do passo 3, mais o grupo de gene da vista; `present_abraom` nulo vira False (como no Mosaic)."""
    df = bracos.carregar_release(raiz)
    grupos = pd.read_parquet(raiz / bracos.VISTA, columns=["variant_id", "gene_transfer_group_id"])
    df = df.merge(grupos, on="variant_id", validate="one_to_one")
    df["present_abraom"] = df["present_abraom"].fillna(False).astype(bool)
    return df


def carregar_predicoes(pasta: Path) -> dict[str, pd.DataFrame]:
    saida = {}
    for braco, cfg in bracos.BRACOS.items():
        p = pd.read_parquet(pasta / cfg["id"] / "predictions.parquet")
        saida[braco] = p.assign(variant_id=p["variant_id"].astype(str), run=p["run"].astype(int))
    return saida


def conferir_selecao(selecao: dict[str, Any]) -> dict[str, Any]:
    """O passo 3 vale sob a regra revisada (ef4b29d): o C escolhido e o que a regra escolhe sem os ajustes que nao
    convergiram, e todo S interno convergiu. Senao, o passo 3 precisa rodar de novo com o codigo revisado."""
    problemas, avisos, nao_convergidos, ajustes = [], set(), 0, 0
    for braco, por_run in selecao["bracos"].items():
        for registro in por_run:
            grade = registro["grade"]
            for g in grade:
                ajustes += 1
                avisos.update(g.get("avisos", []))
                nao_convergidos += not g.get("convergiu", False)
            validos = [(g["macro_validacao"], -i) for i, g in enumerate(grade)
                       if g.get("convergiu", False) and g.get("macro_validacao") is not None
                       and math.isfinite(g["macro_validacao"])]
            if not validos:
                problemas.append(f"{braco} run {registro['run']}: nenhum C convergido com macro finita")
            elif grade[-max(validos)[1]]["C"] != registro["C"]:
                problemas.append(f"{braco} run {registro['run']}: a regra revisada escolheria "
                                 f"C={grade[-max(validos)[1]]['C']}, nao {registro['C']}")
            for interno in registro.get("s_interno", []):
                ajustes += 1
                avisos.update(interno.get("avisos", []))
                if not interno.get("convergiu", False):
                    problemas.append(f"{braco} run {registro['run']}: S interno do fold {interno['fold_interno']} "
                                     f"nao convergiu")
    if problemas:
        raise FalhaDaFase1("o passo 3 desta pasta nao vale com a regra revisada; rodar o passo 3 de novo: "
                           + "; ".join(problemas[:5]))
    return {"ajustes": ajustes, "nao_convergidos_fora_da_escolha": nao_convergidos, "avisos": sorted(avisos)}


def conferir_fontes(fontes: dict[str, Any], raiz: Path) -> None:
    """As predicoes tem de ser do release em disco: mesma identidade e mesmos arquivos lidos no passo 3."""
    lida = fontes.get("identidade_do_release", {}).get("release_identity_hash")
    if lida != bracos.REFERENCIA_DO_BENCHMARK["release_identity_hash"]:
        raise FalhaDaFase1(f"o passo 3 foi feito com outro release: {lida}")
    for nome, sha in fontes.get("arquivos_do_release", {}).items():
        if sha256_do_arquivo(raiz / nome) != sha:
            raise FalhaDaFase1(f"{nome} mudou desde o passo 3")


def conferir_linhas(preds: dict[str, pd.DataFrame], release: pd.DataFrame) -> dict[str, int]:
    """Cada braco com exatamente as linhas do passo 3: validation e teste de cada execucao, score finito."""
    por_run, _ = bracos.linhas_por_execucao(release)
    ids = release.loc[release["sequence_eligible"], "variant_id"].astype(str).to_numpy()
    esperado: set[tuple[str, int, str]] = set()
    for run, idx in enumerate(por_run):
        esperado |= {(v, run, "validation") for v in ids[idx["validation"]]}
        esperado |= {(v, run, "test") for v in ids[idx["teste"]]}
    for braco, p in preds.items():
        chaves = set(zip(p["variant_id"], p["run"], p["role"].astype(str)))
        if len(chaves) != len(p) or chaves != esperado:
            raise FalhaDaFase1(f"{braco}: as predicoes nao sao as linhas esperadas do passo 3")
        if not np.isfinite(p["score"].to_numpy(dtype=float)).all():
            raise FalhaDaFase1(f"{braco}: score nao finito")
        if set(p["study"]) != {"core_locus"} or set(p["window_bp"].astype(int)) != {JANELA_BP}:
            raise FalhaDaFase1(f"{braco}: estudo ou janela fora do contrato")
    return {"linhas_por_braco": len(esperado)}


# --------------------------------------------------------------------------------------------- limiar e chamadas

def limiares(preds: dict[str, pd.DataFrame], rotulo: pd.Series, calibrar: Callable[..., Any]) -> dict[str, dict]:
    """O limiar de cada execucao pela regra do avaliador: MCC maximo na validation (todos os paineis)."""
    saida: dict[str, dict[int, float]] = {}
    for braco, p in preds.items():
        saida[braco] = {}
        for run in range(K):
            v = p[(p["run"] == run) & (p["role"] == "validation")]
            t = p[(p["run"] == run) & (p["role"] == "test")]
            ajuste = calibrar(rotulo.reindex(v["variant_id"]).to_numpy(), v["score"].to_numpy(dtype=float),
                              validation_ids=v["variant_id"], test_ids=t["variant_id"])
            if ajuste.threshold is None:
                raise FalhaDaFase1(f"{braco} run {run}: sem limiar na validation ({ajuste.status})")
            saida[braco][run] = float(ajuste.threshold)
    return saida


def chamadas_de_teste(p: pd.DataFrame, limiar: dict[int, float]) -> pd.DataFrame:
    """Score, limiar da execucao e chamada (positive/negative) de cada variante de teste, indexado por variant_id."""
    t = p[p["role"] == "test"]
    corte = t["run"].map(limiar).to_numpy(dtype=float)
    score = t["score"].to_numpy(dtype=float)
    return pd.DataFrame({"run": t["run"].to_numpy(), "score": score, "limiar": corte,
                         "chamada": np.where(score >= corte, "positive", "negative")},
                        index=pd.Index(t["variant_id"], name="variant_id"))


def scores_de_teste(p: pd.DataFrame) -> pd.Series:
    t = p[p["role"] == "test"]
    return pd.Series(t["score"].to_numpy(dtype=float), index=pd.Index(t["variant_id"], name="variant_id"))


def pontos(preds: dict[str, pd.DataFrame], base: str, novo: str) -> dict[str, pd.Series]:
    """Os dois sistemas no formato de `eval/campanha/estudos.py`: `regionalized` e o braco NOVO do par."""
    return {estudos.BASE: scores_de_teste(preds[base]), estudos.REGIONALIZADO: scores_de_teste(preds[novo])}


def endpoints_de_chamada(y: np.ndarray, chamada: np.ndarray) -> dict[str, float | None]:
    """Os de `contrasts.call_endpoints`: positivo e P/LP, negativo e B/LB, sem score conta como abstencao."""
    y = np.asarray(y, dtype=int)
    n_p, n_b = int((y == 1).sum()), int((y == 0).sum())
    positivo, negativo = chamada == "positive", chamada == "negative"
    correto = (positivo & (y == 1)) | (negativo & (y == 0))
    return {"coverage_correct": float(correto.mean()) if len(y) else None,
            "sensitivity": float((positivo & (y == 1)).sum() / n_p) if n_p else None,
            "false_positive_rate": float((positivo & (y == 0)).sum() / n_b) if n_b else None}


# ----------------------------------------------------------------------------------------------------- leituras

def _com_scores(frame: pd.DataFrame, preds: dict[str, pd.DataFrame], colunas: dict[str, str]) -> pd.DataFrame:
    """Score de cada braco por (variant_id, run, role) nas linhas de uma coorte do Mosaic."""
    frame = frame.copy()
    chave = pd.MultiIndex.from_arrays([frame["variant_id"].astype(str), frame["run"].astype(int),
                                       frame["role"].astype(str)])
    for braco, coluna in colunas.items():
        p = preds[braco].set_index(["variant_id", "run", "role"])["score"]
        frame[coluna] = p.reindex(chave).to_numpy(dtype=float)
    return frame


def nucleo(ferramentas: dict[str, Callable[..., Any]], raiz: Path, release: pd.DataFrame,
           preds: dict[str, pd.DataFrame], *, replicas: int, seed: int,
           avisar: Callable[[str], None]) -> dict[str, Any]:
    teste, validacao = ferramentas["core_cohort"](raiz, JANELA_BP)
    gold = release[(release["label_tier"] == "gold") & release["sequence_eligible"]]
    frame = gold[["variant_id", "binary_label", "overlap_cluster_id", "primary_panel"]].reset_index(drop=True)
    saida: dict[str, Any] = {"coorte": {"teste": int(len(teste)), "validation": int(len(validacao)),
                                        "teste_gold_elegivel": int(len(frame))}, "pares": {}}
    for base, novo in (*PARES, PAR_DESCRITIVO_DO_NUCLEO):
        nome = nome_do_par(base, novo)
        avisar(f"nucleo: {nome}")
        a, b = bracos.BRACOS[novo]["id"], bracos.BRACOS[base]["id"]
        t = _com_scores(teste, preds, {novo: a, base: b})
        v = _com_scores(validacao, preds, {novo: a, base: b})
        oficial = ferramentas["paired_contrast"](t, v, a=a, b=b, aggregate="fold_weighted", n_replicates=replicas,
                                                 seed=seed)
        saida["pares"][nome] = {
            "base": base, "novo": novo, "descritivo": (base, novo) == PAR_DESCRITIVO_DO_NUCLEO,
            "oficial": {"definicao": "contrasts.paired_contrast: a = novo, b = base; delta = a - b", "linhas": oficial},
            "por_painel": estudos.analise_do_coorte(frame, pontos(preds, base, novo), replicas=replicas, seed=seed,
                                                    por_painel=True)}
    return saida


def proxies(membros: pd.DataFrame, preds: dict[str, pd.DataFrame], *, replicas: int, seed: int,
            avisar: Callable[[str], None]) -> dict[str, Any]:
    saida: dict[str, Any] = {}
    for base, novo in PARES:
        nome = nome_do_par(base, novo)
        saida[nome] = {"base": base, "novo": novo}
        for estudo in estudos.ESTUDOS:
            avisar(f"proxies: {nome}, {estudo}")
            saida[nome][estudo] = estudos.avaliar_estudo(membros, estudo, pontos(preds, base, novo),
                                                         replicas=replicas, seed=seed,
                                                         unidade_principal=UNIDADE_PRINCIPAL)
    return saida


def beneficio(ferramentas: dict[str, Callable[..., Any]], raiz: Path, release: pd.DataFrame,
              preds: dict[str, pd.DataFrame], chamadas: dict[str, pd.DataFrame], *, replicas: int, seed: int,
              avisar: Callable[[str], None]) -> dict[str, Any]:
    teste, _ = ferramentas["regional_clinical_cohort"](raiz, JANELA_BP)
    teste = teste.reset_index(drop=True)
    ids = teste["variant_id"].astype(str)
    y = teste["y"].to_numpy(dtype=int)
    por_braco = {}
    for braco in bracos.BRACOS:
        c = chamadas[braco].reindex(ids)
        s, chamada = c["score"].to_numpy(dtype=float), c["chamada"].to_numpy()
        por_painel = {p: {"auroc": metricas.auroc(s[m], y[m]), "auprc": metricas.auprc(s[m], y[m]),
                          "n_P": int((y[m] == 1).sum()), "n_B": int((y[m] == 0).sum())}
                      for p in PAINEIS if (m := (teste["panel"] == p).to_numpy()).any()}
        por_braco[braco] = {"auroc": metricas.auroc(s, y), "auprc": metricas.auprc(s, y),
                            **endpoints_de_chamada(y, chamada), "por_painel": por_painel}
    linhas = release.set_index("variant_id").loc[ids]
    frame = linhas.reset_index()[["variant_id", "binary_label", "overlap_cluster_id", "primary_panel"]]
    pares: dict[str, Any] = {}
    for base, novo in PARES:
        nome = nome_do_par(base, novo)
        avisar(f"beneficio: {nome}")
        a, b = bracos.BRACOS[novo]["id"], bracos.BRACOS[base]["id"]
        t = teste.assign(**{a: chamadas[novo]["chamada"].reindex(ids).to_numpy(),
                            b: chamadas[base]["chamada"].reindex(ids).to_numpy()})
        pares[nome] = {
            "base": base, "novo": novo,
            "chamadas_oficial": {"definicao": "contrasts.paired_call_contrast: a = novo, b = base; delta = a - b",
                                 "linhas": ferramentas["paired_call_contrast"](t, a=a, b=b, n_replicates=replicas,
                                                                               seed=seed)},
            "continuas": estudos.analise_do_coorte(frame, pontos(preds, base, novo), replicas=replicas, seed=seed,
                                                   por_painel=True)}
    return {"coorte": {"n": int(len(teste)), "n_P": int((y == 1).sum()), "n_B": int((y == 0).sum()),
                       "unidades": int(teste["unit"].nunique())},
            "por_braco": por_braco, "pares": pares}


def limite_superior_cp(x: int, n: int, confianca: float = 0.95) -> float:
    """Limite superior unilateral de Clopper-Pearson: o p com P(X <= x | n, p) = 1 - confianca, o mesmo que
    `beta.ppf(confianca, x + 1, n - x)` do `evaluate_safety.py`, por bissecao (sem scipy)."""
    if x >= n:
        return 1.0
    alfa = 1.0 - confianca

    def cdf(p: float) -> float:
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        termos = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p)
                  + (n - k) * math.log1p(-p) for k in range(x + 1)]
        maior = max(termos)
        return math.exp(maior) * sum(math.exp(t - maior) for t in termos)

    baixo, alto = 0.0, 1.0
    for _ in range(200):
        meio = (baixo + alto) / 2
        if cdf(meio) > alfa:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2


def _chave(frame: pd.DataFrame) -> pd.Series:
    return frame["chrom"].astype(str) + ":" + frame["pos_1based"].astype(str) + ":" + frame["ref"] + ":" + frame["alt"]


def p_br(release: pd.DataFrame, chamadas: dict[str, pd.DataFrame], criticas: list[dict[str, Any]], *,
         margem: float = MARGEM_DA_PERDA) -> dict[str, Any]:
    """P/LP do release presentes no ABraOM (gold e consensus), cada uma chamada pela execucao que a testa."""
    pbr = release[(release["binary_label"] == 1) & release["present_abraom"]].reset_index(drop=True)
    ids = pbr["variant_id"].astype(str)
    n_grupos = int(pbr["gene_transfer_group_id"].nunique())
    chaves_criticas = {f"{c['grch38']['chrom']}:{c['grch38']['pos']}:{c['grch38']['ref']}:{c['grch38']['alt']}":
                       c.get("gene") for c in criticas}
    critica = _chave(pbr).isin(chaves_criticas).to_numpy()

    def positivas(braco: str) -> np.ndarray:
        return (chamadas[braco]["chamada"].reindex(ids) == "positive").to_numpy()

    por_braco = {}
    for braco in bracos.BRACOS:
        pos = positivas(braco)
        por_braco[braco] = {
            "positivas": int(pos.sum()), "sensibilidade": float(pos.mean()) if len(pos) else None,
            "sem_score": int(chamadas[braco]["chamada"].reindex(ids).isna().sum()),
            "por_painel": {str(p): {"n": int(len(g)), "positivas": int(pos[g.index].sum())}
                           for p, g in pbr.groupby("primary_panel", sort=True)},
            "por_tier": {str(t): {"n": int(len(g)), "positivas": int(pos[g.index].sum())}
                         for t, g in pbr.groupby("label_tier", sort=True)}}
    pares = {}
    for base, novo in PARES:
        r0, r1 = positivas(base), positivas(novo)
        perdidas, ganhas = r0 & ~r1, r1 & ~r0
        x = int(perdidas.sum())
        superior = limite_superior_cp(x, n_grupos)
        pares[nome_do_par(base, novo)] = {
            "base": base, "novo": novo, "reconhecidas_pela_base": int(r0.sum()), "perdidas": x, "ganhas": int(ganhas.sum()),
            "taxa_bruta_de_perda": x / len(pbr) if len(pbr) else None, "limite_superior_95": superior,
            "margem": margem, "criticas_perdidas": int((perdidas & critica).sum()),
            "seguranca_declarada": bool(superior < margem and not (perdidas & critica).any()),
            "perdidas_por_painel": {str(p): int(perdidas[g.index].sum()) for p, g in pbr.groupby("primary_panel")},
            "ids_perdidas": ids[perdidas].tolist(), "ids_ganhas": ids[ganhas].tolist()}
    return {"definicao": ("a de scripts/evaluate_safety.py: perda = P-BR chamada positiva pela base e nao pelo novo; "
                          "limite superior unilateral de Clopper-Pearson a 95% com n = grupos de gene com P-BR"),
            "n_p_br": int(len(pbr)), "n_grupos_de_gene": n_grupos,
            "por_tier": {str(t): int(n) for t, n in pbr["label_tier"].value_counts().sort_index().items()},
            "criticas_entre_as_p_br": int(critica.sum()), "por_braco": por_braco, "pares": pares}


def criticas_por_braco(lista: list[dict[str, Any]], release: pd.DataFrame,
                       chamadas: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """As 13 uma a uma: a conferencia que `evaluate_safety.py` nao faz para as ausentes do ABraOM."""
    saida = []
    for item in criticas_do_release(lista, release):
        if not item["no_release"]:
            saida.append({**item, "bracos": None, "nota": "fora do release: sem score nesta fase"})
            continue
        vid, por_braco = item["variant_id"], {}
        for braco in bracos.BRACOS:
            if vid in chamadas[braco].index:
                linha = chamadas[braco].loc[vid]
                por_braco[braco] = {"run": int(linha["run"]), "score": float(linha["score"]),
                                    "limiar": float(linha["limiar"]), "chamada": str(linha["chamada"])}
            else:
                por_braco[braco] = {"chamada": None, "nota": "sem score (nao elegivel em 4 kb)"}
        perdas = {nome_do_par(b, n): (por_braco[b]["chamada"] == "positive" and por_braco[n]["chamada"] != "positive")
                  for b, n in PARES}
        saida.append({**item, "bracos": por_braco, "perdida_no_par": perdas})
    return saida


def avaliador_oficial(pasta: Path, limiares_: dict[str, dict[int, float]]) -> dict[str, Any]:
    """As metricas agregadas por fold que o avaliador oficial gravou (4 kb) e a conferencia dos limiares."""
    tabela, conferidos = [], {}
    for k, (braco, cfg) in enumerate(bracos.BRACOS.items()):
        achados = sorted((pasta / cfg["id"]).glob("*/4kb/specialist-metrics.parquet"))
        if len(achados) != 1:
            raise FalhaDaFase1(f"{pasta / cfg['id']}: esperado um */4kb/specialist-metrics.parquet, achados "
                               f"{len(achados)}")
        m = pd.read_parquet(achados[0])
        agregadas = m[(m["evaluation_id"] == "core_locus") & m["run"].isna()
                      & m["metric"].isin(list(METRICAS_DO_AVALIADOR))]
        for _, linha in agregadas.iterrows():
            proprio = linha["comparator_id"] == cfg["id"]
            if proprio or (k == 0 and linha["comparator_id"] not in IDS_DOS_BRACOS):
                tabela.append({"sistema": braco if proprio else str(linha["comparator_id"]), "oficial": not proprio,
                               "painel": str(linha["panel"]), "metrica": METRICAS_DO_AVALIADOR[linha["metric"]],
                               "estimativa": _float(linha["estimate"]), "ic95": [_float(linha["ci95_low"]),
                                                                                  _float(linha["ci95_high"])],
                               "n_scored": _float(linha["n_scored"]), "coverage": _float(linha["coverage"])})
        gravados = pd.read_parquet(achados[0].with_name("validation-thresholds.parquet"))
        gravados = gravados[(gravados["comparator_id"] == cfg["id"]) & (gravados["track"] == "core_locus")]
        lidos = {int(r): float(t) for r, t in zip(gravados["run"], gravados["threshold"])}
        if set(lidos) != set(limiares_[braco]) or any(not np.isclose(lidos[r], limiares_[braco][r], rtol=0, atol=1e-12)
                                                       for r in lidos):
            raise FalhaDaFase1(f"{braco}: limiares do avaliador {lidos} diferem dos calculados {limiares_[braco]}")
        conferidos[braco] = {"pasta": str(achados[0].parent), "limiares": lidos}
    return {"metricas": tabela, "limiares_conferidos": conferidos}


def _float(valor: Any) -> float | None:
    return None if valor is None or pd.isna(valor) else float(valor)


# -------------------------------------------------------------------------------------------------------- resumo

def _f(valor: Any, casas: int = 4) -> str:
    return "—" if valor is None or (isinstance(valor, float) and not math.isfinite(valor)) else f"{valor:.{casas}f}"


def _ic(bloco: dict[str, Any] | None) -> str:
    if not bloco or bloco.get("estimativa") is None:
        return "—"
    return f"{bloco['estimativa']:+.4f} [{_f(bloco.get('p2_5'))}; {_f(bloco.get('p97_5'))}]"


def _linha_oficial(linhas: list[dict[str, Any]], endpoint: str, parte: str) -> dict[str, Any] | None:
    return next((r for r in linhas if r.get("endpoint") == endpoint and r.get("part") == parte), None)


def _ic_oficial(r: dict[str, Any] | None, sinal: bool = True) -> str:
    if not r or r.get("estimate") is None:
        return "—"
    valor = f"{r['estimate']:+.4f}" if sinal else f"{r['estimate']:.4f}"
    return f"{valor} [{_f(r.get('ci95_low'))}; {_f(r.get('ci95_high'))}]"


def resumo(leituras: dict[str, Any]) -> str:
    """As tabelas principais em Markdown. O JSON tem o resto (celulas sem painel, sensibilidades, ids)."""
    L = ["# Fase 1, passo 4: leituras dos braços (desenvolvimento)", "",
         f"Passo 3 em `{leituras['proveniencia']['bracos']}`. Delta = novo − base. IC 95% por bootstrap "
         f"({leituras['declaracao']['replicas']} réplicas, seed {leituras['declaracao']['seed']}). Nada aqui é "
         "confirmatório.", ""]
    nuc = leituras["nucleo"]
    L += ["## Núcleo: contraste pareado oficial (teste gold de 4 kb)", "",
          "| par | macro AUPRC base | macro AUPRC novo | Δ macro AUPRC [IC] | Δ MCC [IC] |", "|---|---|---|---|---|"]
    for nome, par in nuc["pares"].items():
        linhas = par["oficial"]["linhas"]
        base, novo = _linha_oficial(linhas, "macro_auprc", "b"), _linha_oficial(linhas, "macro_auprc", "a")
        L.append(f"| {nome}{' (descritivo)' if par['descritivo'] else ''} | {_f(base and base.get('estimate'))} | "
                 f"{_f(novo and novo.get('estimate'))} | {_ic_oficial(_linha_oficial(linhas, 'macro_auprc', 'delta'))}"
                 f" | {_ic_oficial(_linha_oficial(linhas, 'mcc', 'delta'))} |")
    L += ["", "## Núcleo: Δ AUROC por painel (teste gold das cinco execuções juntas)", "",
          "| par | coorte | " + " | ".join(metricas.PAINEIS_DE_DISCRIMINACAO) + " |",
          "|---|---|" + "---|" * len(metricas.PAINEIS_DE_DISCRIMINACAO)]
    for nome, par in nuc["pares"].items():
        celulas = par["por_painel"]
        L.append(f"| {nome} | {_ic(celulas['coorte']['delta']['auroc'])} | "
                 + " | ".join(_ic(celulas.get(f"painel:{p}", {}).get("delta", {}).get("auroc"))
                              for p in metricas.PAINEIS_DE_DISCRIMINACAO) + " |")
    if leituras.get("avaliador_oficial"):
        tabela = pd.DataFrame(leituras["avaliador_oficial"]["metricas"])
        auroc = tabela[tabela["metrica"].isin(["auroc", "macro_auroc"])]
        auroc = auroc.assign(painel=np.where(auroc["metrica"] == "macro_auroc", "macro", auroc["painel"]))
        colunas = ["macro", *PAINEIS]
        L += ["", "## Avaliador oficial: AUROC agregado por fold (teste gold de 4 kb)", "",
              "| sistema | " + " | ".join(colunas) + " |", "|---|" + "---|" * len(colunas)]
        for sistema, g in auroc.groupby("sistema", sort=False):
            valores = dict(zip(g["painel"], g["estimativa"]))
            L.append(f"| {sistema} | " + " | ".join(_f(valores.get(c)) for c in colunas) + " |")
    L += ["", "## Proxies brasileiros: Δ AUROC [IC]", "",
          "| par | estudo | coorte completo | pareados | controles | interação (cluster_conjunto) |",
          "|---|---|---|---|---|---|"]
    for nome, par in leituras["proxies"].items():
        for estudo in estudos.ESTUDOS:
            r = par[estudo]
            coortes = r["coortes"]
            L.append(f"| {nome} | {estudo} | {_ic(coortes['full_cohort']['coorte']['delta']['auroc'])} | "
                     f"{_ic(coortes['matched_cases']['coorte']['delta']['auroc'])} | "
                     f"{_ic(coortes['controls']['coorte']['delta']['auroc'])} | "
                     f"{_ic(r['interacao']['auroc'].get('interacao'))} |")
    ben = leituras["beneficio"]
    L += ["", f"## Benefício: teste gold de 4 kb presente no ABraOM ({ben['coorte']['n']}, "
              f"{ben['coorte']['n_P']} P)", "",
          "| braço | AUROC | AUPRC | acertos | sensibilidade | falso-positivo |", "|---|---|---|---|---|---|"]
    for braco, r in ben["por_braco"].items():
        L.append(f"| {braco} | {_f(r['auroc'])} | {_f(r['auprc'])} | {_f(r['coverage_correct'])} | "
                 f"{_f(r['sensitivity'])} | {_f(r['false_positive_rate'])} |")
    L += ["", "| par | Δ AUROC [IC] | Δ acertos [IC] | Δ sensibilidade [IC] | Δ falso-positivo [IC] |",
          "|---|---|---|---|---|"]
    for nome, par in ben["pares"].items():
        linhas = par["chamadas_oficial"]["linhas"]
        L.append(f"| {nome} | {_ic(par['continuas']['coorte']['delta']['auroc'])} | "
                 + " | ".join(_ic_oficial(_linha_oficial(linhas, e, "delta"))
                              for e in ("coverage_correct", "sensitivity", "false_positive_rate")) + " |")
    pbr = leituras["p_br"]
    L += ["", f"## P-BR ({pbr['n_p_br']}; {pbr['n_grupos_de_gene']} grupos de gene)", "",
          "| braço | positivas | sensibilidade |", "|---|---|---|"]
    for braco, r in pbr["por_braco"].items():
        L.append(f"| {braco} | {r['positivas']} | {_f(r['sensibilidade'])} |")
    L += ["", "| par | reconhecidas pela base | perdidas | ganhas | limite superior 95% | críticas perdidas | "
              "segurança (margem 0,01) |", "|---|---|---|---|---|---|---|"]
    for nome, r in pbr["pares"].items():
        L.append(f"| {nome} | {r['reconhecidas_pela_base']} | {r['perdidas']} | {r['ganhas']} | "
                 f"{_f(r['limite_superior_95'])} | {r['criticas_perdidas']} | {'sim' if r['seguranca_declarada'] else 'não'} |")
    L += ["", "## Críticas", "", "| gene | variante | ABraOM | tier | " + " | ".join(bracos.BRACOS) + " |",
          "|---|---|---|---|" + "---|" * len(bracos.BRACOS)]
    for c in leituras["criticas"]:
        if c["bracos"] is None:
            L.append(f"| {c['gene']} | {c['hgvs']} | — | — | " + " | ".join("fora do release" for _ in bracos.BRACOS)
                     + " |")
            continue
        L.append(f"| {c['gene']} | {c['hgvs']} | {'sim' if c['present_abraom'] else 'não'} | {c['tier']} | "
                 + " | ".join(str(c["bracos"][b]["chamada"]) for b in bracos.BRACOS) + " |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------- main

def ferramentas_do_mosaic() -> dict[str, Callable[..., Any]]:
    """O que vem do pacote `mosaic` da entrega; os testes substituem esta funcao."""
    from mosaic.comparator_eval.contrasts import paired_call_contrast, paired_contrast
    from mosaic.comparator_eval.primary_cohorts import core_cohort, regional_clinical_cohort
    from mosaic.comparator_eval.stats import calibrate_threshold
    from mosaic.identity import verify_release_identity

    return {"verificar_identidade": verify_release_identity, "calibrate_threshold": calibrate_threshold,
            "core_cohort": core_cohort, "regional_clinical_cohort": regional_clinical_cohort,
            "paired_contrast": paired_contrast, "paired_call_contrast": paired_call_contrast}


def _json(valor: Any) -> Any:
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    return str(valor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path)
    parser.add_argument("--bracos", required=True, type=Path, help="pasta de saida do passo 3")
    parser.add_argument("--avaliacao", type=Path, help="--output-root do evaluate_candidate.py dos seis bracos")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    args = parser.parse_args(argv)
    entrega, pasta_bracos, destino = args.entrega.expanduser(), args.bracos.expanduser(), args.out_dir.expanduser()
    temporaria = destino.with_name(destino.name + ".tmp")
    if destino.exists() or temporaria.exists():
        return _falhar([f"{destino} ou {temporaria} ja existe; as leituras gravam sempre numa pasta nova"])
    try:
        ferramentas = ferramentas_do_mosaic()
    except ImportError as exc:
        return _falhar([f"sem o pacote mosaic ({exc}); rodar no .venv do Mosaic (uv run --project)"])

    inicio = time.perf_counter()

    def avisar(texto: str) -> None:
        print(f"[{time.perf_counter() - inicio:7.1f} s] {texto}", flush=True)

    raiz = entrega / bracos.RELEASE
    try:
        identidade = ferramentas["verificar_identidade"](raiz, config_dir=entrega / "config")
        protocolo_sha256 = sha256_do_arquivo(entrega / bracos.PROTOCOLO_DE_ESTUDOS)
        bracos.conferir_entrega(identidade, protocolo_sha256)
        release = carregar_release(raiz)
        selecao = json.loads((pasta_bracos / "selecao.json").read_text(encoding="utf-8"))
        fontes = json.loads((pasta_bracos / "fontes.json").read_text(encoding="utf-8"))
        conferencia_da_selecao = conferir_selecao(selecao)
        conferir_fontes(fontes, raiz)
        preds = carregar_predicoes(pasta_bracos)
        conferencia_das_linhas = conferir_linhas(preds, release)
        rotulo = release.set_index("variant_id")["binary_label"].astype(int)
        limiares_ = limiares(preds, rotulo, ferramentas["calibrate_threshold"])
        oficial = avaliador_oficial(args.avaliacao.expanduser(), limiares_) if args.avaliacao else None
        membros = pd.read_parquet(raiz / MEMBERSHIP)
        lista_de_criticas = ler_criticas(entrega / CRITICAS)
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return _falhar([f"{type(exc).__name__}: {exc}"])
    avisar(f"entradas conferidas: {conferencia_das_linhas['linhas_por_braco']:,} linhas por braco; "
           f"{conferencia_da_selecao['ajustes']} ajustes do passo 3, "
           f"{conferencia_da_selecao['nao_convergidos_fora_da_escolha']} sem convergir (fora da escolha); "
           f"limiares {'conferidos com o avaliador' if oficial else 'sem --avaliacao para conferir'}")

    chamadas = {braco: chamadas_de_teste(p, limiares_[braco]) for braco, p in preds.items()}
    try:
        leituras = {
            "formato": "fase1_leituras_v1",
            "declaracao": {"pares": [nome_do_par(b, n) for b, n in PARES],
                           "par_descritivo_do_nucleo": nome_do_par(*PAR_DESCRITIVO_DO_NUCLEO), "delta": "novo - base",
                           "replicas": args.replicas, "seed": SEED, "unidade_principal_da_interacao": UNIDADE_PRINCIPAL,
                           "limiar": "calibrate_threshold do Mosaic na validation da execucao, todos os paineis",
                           "margem_da_perda_de_p_br": MARGEM_DA_PERDA,
                           "natureza": "desenvolvimento exploratorio; proxies cross-fitted, nao o par congelado"},
            "conferencias": {"selecao": conferencia_da_selecao, "linhas": conferencia_das_linhas},
            "limiares": {b: {str(r): t for r, t in v.items()} for b, v in limiares_.items()},
            "avaliador_oficial": oficial,
        }
        leituras["nucleo"] = nucleo(ferramentas, raiz, release, preds, replicas=args.replicas, seed=SEED,
                                    avisar=avisar)
        leituras["beneficio"] = beneficio(ferramentas, raiz, release, preds, chamadas, replicas=args.replicas,
                                          seed=SEED, avisar=avisar)
        leituras["p_br"] = p_br(release, chamadas, lista_de_criticas)
        leituras["criticas"] = criticas_por_braco(lista_de_criticas, release, chamadas)
        leituras["proxies"] = proxies(membros, preds, replicas=args.replicas, seed=SEED, avisar=avisar)
    except (FalhaDaFase1, estudos.EstudoInvalido, ValueError, KeyError) as exc:
        return _falhar([f"{type(exc).__name__}: {exc}"])
    leituras["proveniencia"] = {
        "entrega": str(entrega), "bracos": str(pasta_bracos), "revisao_do_passo3": fontes.get("revisao"),
        "avaliacao": str(args.avaliacao) if args.avaliacao else None,
        "script_sha256": sha256_do_arquivo(Path(__file__)), "revisao": bracos.revisao_do_repositorio(),
        "versoes": bracos.versoes(), "segundos": round(time.perf_counter() - inicio, 1)}

    temporaria.mkdir(parents=True)
    (temporaria / "leituras.json").write_text(json.dumps(leituras, indent=2, ensure_ascii=False, default=_json),
                                              encoding="utf-8")
    texto = resumo(json.loads((temporaria / "leituras.json").read_text(encoding="utf-8")))
    (temporaria / "resumo.md").write_text(texto, encoding="utf-8")
    temporaria.rename(destino)
    _imprimir("\n" + texto)
    print(f"PASSOU: leituras em {destino}")
    return 0


def _imprimir(texto: str) -> None:
    """O resumo vai em UTF-8 para o arquivo; no console, o que a codificacao local nao tiver vira '?'."""
    codificacao = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(texto.encode(codificacao, errors="replace").decode(codificacao, errors="replace"), flush=True)


def _falhar(problemas: list[str]) -> int:
    print(f"FALHOU: {len(problemas)} problema(s)")
    for problema in problemas:
        print(f"  - {problema}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
