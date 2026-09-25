"""G7, as regras puras da avaliacao unica: a tabela dos estudos para a extracao, a identidade dos caches dos
estudos contra os de desenvolvimento, os scores SINTETICOS do ensaio e a aplicacao das margens declaradas. Sem torch.

Nada aqui escolhe, ajusta ou calibra: os sistemas, os limiares e as regras de decisao vem do manifesto congelado do
G6. A aplicacao das margens e mecanica -- cada regra `estatistica >= limite` sobre o delta declarado, estudo por
estudo, nunca unindo os estudos.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eval.campanha.estudos import (
    BASE,
    CASOS_PAREADOS,
    CONTROLES,
    COORTE_COMPLETO,
    ESTUDOS,
    PAINEIS_DE_DISCRIMINACAO,
    REGIONALIZADO,
)
from eval.campanha.g6 import problemas_das_margens, problemas_do_bootstrap
from eval.campanha.recortes import COLUNAS

PAPEL_DOS_ESTUDOS = "estudo"
#: Campos da identidade de um cache que so dependem da tabela extraida (e o commit, que nao determina numero): o
#: cache dos estudos tem de ser igual ao de desenvolvimento do mesmo sistema em todo o resto.
CAMPOS_DA_TABELA = ("tabela_sha256_composicao", "tabela_sha256_conteudo", "papeis", "revisao_do_codigo")
DELTA_DO_COORTE = {"delta_br_full": COORTE_COMPLETO, "delta_br_matched": CASOS_PAREADOS, "delta_control": CONTROLES}
LEITURA_DO_CHR8 = ("membros no chr8 entram no G7: o Mosaic avalia o membership inteiro, a reserva do chr8 (decisao E) "
                   "e do treino da cabeca e das janelas do adapter, e o G7 e avaliacao unica, nao consulta repetida")


class G7Invalido(ValueError):
    """Entrada do G7 que nao confere com o que foi congelado."""


# ----------------------------------------------------------------------------------------------- extracao

def tabela_dos_estudos(membros: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por variante dos dois estudos (a que esta nos dois e extraida uma vez), com as colunas da extracao e
    papel `estudo`, na ordem canonica do extrator. Recusa variante cujas linhas discordem em coordenada, alelo,
    rotulo, painel, cluster ou tier. Pura."""
    faltando = [c for c in COLUNAS if c not in membros.columns]
    if faltando:
        raise G7Invalido(f"membros sem as colunas {faltando} (use a saida do G1, com coordenadas)")
    distintos = membros.groupby(membros["variant_id"].astype(str))[list(COLUNAS[1:])].nunique(dropna=False)
    divergentes = distintos.index[(distintos > 1).any(axis=1)]
    if len(divergentes):
        raise G7Invalido(f"{len(divergentes)} variantes com atributos diferentes entre estudos, ex.: "
                         f"{list(divergentes[:3])}")
    tabela = membros.drop_duplicates("variant_id")[list(COLUNAS)].copy()
    tabela["papel"] = PAPEL_DOS_ESTUDOS
    return tabela.sort_values(["papel", "variant_id"], kind="mergesort").reset_index(drop=True)


#: As colunas de sequencia vem do `pb_examples` do release; o resto, da membership. As duas com hash logico
#: conferido: a tabela extraida fica amarrada as tabelas oficiais, nao a um arquivo intermediario.
COLUNAS_DE_SEQUENCIA = ("chrom", "pos_1based", "ref", "alt")


def tabela_oficial(membership: pd.DataFrame, exemplos: pd.DataFrame) -> pd.DataFrame:
    """A tabela dos estudos reconstruida das tabelas OFICIAIS: membership + coordenadas e alelos do `pb_examples`
    (o mesmo casamento do G1). Recusa membro sem exemplo e rotulo ou tier que divirjam entre as duas. Pura."""
    faltando = [c for c in ("variant_id", *COLUNAS_DE_SEQUENCIA, "binary_label", "label_tier") if c not in exemplos]
    if faltando:
        raise G7Invalido(f"pb_examples sem as colunas {faltando}")
    if exemplos["variant_id"].duplicated().any():
        raise G7Invalido("pb_examples com variant_id repetido")
    juntos = membership.merge(exemplos[["variant_id", *COLUNAS_DE_SEQUENCIA, "binary_label", "label_tier"]],
                              on="variant_id", how="left", suffixes=("", "_exemplo"), validate="many_to_one")
    if juntos["pos_1based"].isna().any():
        raise G7Invalido(f"{int(juntos['pos_1based'].isna().sum())} membros sem linha no pb_examples")
    for coluna in ("binary_label", "label_tier"):
        divergentes = juntos[coluna].astype(str) != juntos[f"{coluna}_exemplo"].astype(str)
        if divergentes.any():
            raise G7Invalido(f"{int(divergentes.sum())} membros com {coluna} diferente do pb_examples")
    juntos["pos_1based"] = juntos["pos_1based"].astype("int64")
    return tabela_dos_estudos(juntos.drop(columns=["binary_label_exemplo", "label_tier_exemplo"]))


def diferencas_de_tabela(oficial: pd.DataFrame, outra: pd.DataFrame) -> list[str]:
    """Onde uma tabela dos estudos difere da oficial (ids, e em cada coluna da extracao). Pura."""
    if sorted(oficial["variant_id"].astype(str)) != sorted(outra["variant_id"].astype(str)):
        return ["variantes diferentes"]
    a = oficial.set_index(oficial["variant_id"].astype(str))
    b = outra.set_index(outra["variant_id"].astype(str)).loc[a.index]
    return [c for c in COLUNAS if (a[c].astype(str).to_numpy() != b[c].astype(str).to_numpy()).any()]


def diferencas_do_estudo(desenvolvimento: dict[str, Any], estudo: dict[str, Any]) -> list[str]:
    """Campos em que a identidade do cache dos estudos difere da do cache de desenvolvimento do MESMO sistema, alem
    dos que so dependem da tabela. Qualquer um invalida o cache: seria outro caminho numerico. Pura."""
    return sorted(k for k in set(desenvolvimento) | set(estudo)
                  if k not in CAMPOS_DA_TABELA and desenvolvimento.get(k) != estudo.get(k))


def limiares_do_manifesto(manifesto: dict[str, Any]) -> dict[str, float]:
    """Os limiares do ensemble congelados no G6, no formato do consumidor."""
    return {BASE: float(manifesto["sistemas"]["base"]["limiar"]["threshold"]),
            REGIONALIZADO: float(manifesto["sistemas"]["regionalized"]["limiar"]["threshold"])}


# ----------------------------------------------------------------------------------------------- ensaio

def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pontos_sinteticos(membros: pd.DataFrame, *, seed: int) -> dict[str, pd.Series]:
    """Probabilidades SINTETICAS para o ensaio: um sinal fixo no rotulo mais ruido com seed. Nao sao resultado de
    nenhum modelo; servem para exercitar a leitura dos artefatos reais antes da avaliacao unica. Pura."""
    unicos = membros.drop_duplicates("variant_id")
    ids = unicos["variant_id"].astype(str).to_numpy()
    y = unicos["binary_label"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    logito = 1.5 * (2 * y - 1) + rng.normal(size=len(y))
    return {BASE: pd.Series(_sigmoide(logito), index=ids),
            REGIONALIZADO: pd.Series(_sigmoide(logito + 0.3 * rng.normal(size=len(y))), index=ids)}


def baselines_sinteticas(membros: pd.DataFrame, nomes: list[str], *, seed: int) -> dict[str, pd.Series]:
    """Scores de baseline SINTETICOS e sem relacao com o rotulo, para o ensaio nao medir baseline real nenhuma. Pura."""
    ids = membros.drop_duplicates("variant_id")["variant_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    return {nome: pd.Series(rng.random(len(ids)), index=ids) for nome in nomes}


# ----------------------------------------------------------------------------------------------- margens

def _regra(valor: Any, item: dict[str, Any]) -> dict[str, Any]:
    atende = None if valor is None else bool(valor >= item["limite"])
    return {"valor": valor, "estatistica": item["estatistica"], "limite": item["limite"], "atende": atende}


def _celula(resultado: dict[str, Any], coorte: str, celula: str, metrica: str) -> dict[str, Any] | None:
    return ((resultado.get("coortes", {}).get(coorte) or {}).get(celula) or {}).get("delta", {}).get(metrica)


def _combinar(regras: list[dict[str, Any]]) -> bool | None:
    """Todas as regras aplicaveis: uma falha -> False; alguma indefinida -> None; senao True."""
    estados = [r["atende"] for r in regras if r.get("aplicavel", True)]
    if any(e is False for e in estados):
        return False
    if any(e is None for e in estados):
        return None
    return True


def avaliar_margens(resultados: dict[str, dict[str, Any]], margens: dict[str, Any],
                    bootstrap: dict[str, Any]) -> dict[str, Any]:
    """As regras declaradas (g6.margens) sobre os resultados do consumidor, estudo por estudo. Declaracao incompleta
    nao e avaliada: sai o motivo. Pura."""
    problemas = problemas_das_margens(margens) + problemas_do_bootstrap(bootstrap)
    if problemas:
        return {"avaliado": False, "motivo": "margens ou bootstrap nao declarados com conteudo", "problemas": problemas}
    saida: dict[str, Any] = {"avaliado": True, "regra": "cada regra: estatistica >= limite sobre o delta MR - M0 "
                                                        "declarado; estudos nunca unidos", "por_estudo": {}}
    for estudo in ESTUDOS:
        if estudo not in resultados:
            continue
        r, regras = resultados[estudo], {}
        for nome in ("melhoria_minima_no_coorte_br", "regressao_maxima_no_controle"):
            item = margens[nome]
            if estudo in item["estudos"]:
                celula = _celula(r, DELTA_DO_COORTE[item["delta"]], "coorte", item["metrica"])
                regras[nome] = {"delta": item["delta"], "metrica": item["metrica"],
                                **_regra((celula or {}).get(item["estatistica"]), item)}
        item = margens["paineis_com_regressao_inaceitavel"]
        if estudo in item.get("estudos", []) and item.get("paineis"):
            por_painel = {}
            for painel in item["paineis"]:
                celula = _celula(r, DELTA_DO_COORTE[item["delta"]], f"painel:{painel}", item["metrica"])
                por_painel[painel] = _regra((celula or {}).get(item["estatistica"]), item)
            regras["paineis_com_regressao_inaceitavel"] = {"delta": item["delta"], "metrica": item["metrica"],
                                                           "por_painel": por_painel,
                                                           "atende": _combinar(list(por_painel.values()))}
        item = margens["beneficio_nao_explicado_por_um_painel"]
        if estudo in item["estudos"]:
            coorte = DELTA_DO_COORTE[item["delta"]]
            minimo = item["suporte_minimo_por_painel"]
            com_suporte = []
            for painel in PAINEIS_DE_DISCRIMINACAO:
                intersecao = ((r.get("coortes", {}).get(coorte) or {}).get(f"painel:{painel}") or {}).get(
                    "intersecao") or {}
                if intersecao.get("n_P_scored", 0) >= minimo and intersecao.get("n_B_scored", 0) >= minimo:
                    com_suporte.append(painel)
            if len(com_suporte) < 2:
                regras["beneficio_nao_explicado_por_um_painel"] = {
                    "aplicavel": False, "atende": None, "paineis_com_suporte": com_suporte,
                    "motivo": "suporte em menos de dois paineis: a condicao 3 do Mosaic nao se aplica"}
            else:
                sem = {p: _regra((_celula(r, coorte, f"sem_painel:{p}", item["metrica"]) or {}).get(
                    item["estatistica"]), item) for p in com_suporte}
                regras["beneficio_nao_explicado_por_um_painel"] = {
                    "aplicavel": True, "delta": item["delta"], "metrica": item["metrica"],
                    "paineis_com_suporte": com_suporte, "sem_cada_painel": sem, "atende": _combinar(list(sem.values()))}
        interacao = margens["interacao"]
        if interacao["criterio_proprio"] and estudo in interacao["estudos"]:
            # O IC da unidade DECLARADA, lido em `por_unidade`: nao depende do que o consumidor pos no nivel de cima.
            celula = (((r.get("interacao") or {}).get(interacao["metrica"]) or {}).get("por_unidade") or {}).get(
                bootstrap["unidade_principal"], {}).get("interacao") or {}
            regras["interacao"] = {"metrica": interacao["metrica"], "unidade": bootstrap["unidade_principal"],
                                   **_regra(celula.get(interacao["estatistica"]), interacao)}
        if not regras:
            saida["por_estudo"][estudo] = {"regras": {}, "atende_todas": None,
                                           "nota": "nenhuma regra declarada para este estudo: nada a atender"}
            continue
        saida["por_estudo"][estudo] = {"regras": regras, "atende_todas": _combinar(list(regras.values()))}
    return saida
