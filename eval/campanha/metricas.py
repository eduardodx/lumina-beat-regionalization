"""Metricas da comparacao clinica de desenvolvimento. Sem torch.

AUROC e macro dos paineis PORTADAS da pesquisa de extracao (`eval/embedding_probe/stats.py` e `protocol.py` da
branch `embedding-probe-mosaic`): Mann-Whitney com empate valendo meio ponto, scores arredondados a 1e-12 relativo
antes de ranquear, e a macro NAO ponderada de missense, splice e noncoding -- a regra de selecao do Mosaic. Os
paineis de guarda (plof quase so P, synonymous quase so B) ficam fora da macro: separa-los premiaria reconhecer o
tipo de variante, nao discriminar dentro dele.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

PAINEIS_DE_DISCRIMINACAO = ("missense", "splice", "noncoding")
PAINEIS_DE_GUARDA = ("plof", "synonymous")


def arredondar(scores: Sequence[float]) -> np.ndarray:
    """1e-12 RELATIVO antes de ranquear. Sem isso, um preditor sem ordenamento (constante a menos de ruido de
    ponto flutuante) tem o ruido ordenado e a AUROC passeia em torno de 0,5 -- a pesquisa mediu 0,511 e 0,580 no
    mesmo dado antes desta correcao."""
    valores = np.asarray(scores, dtype=np.float64)
    escala = float(np.max(np.abs(valores))) if valores.size else 0.0
    return np.round(valores / (escala or 1.0), 12)


def postos_medios(valores: np.ndarray) -> np.ndarray:
    """Postos 1..n com empate recebendo a media dos postos (o `rankdata` da pesquisa)."""
    ordem = np.argsort(valores, kind="mergesort")
    ordenados = valores[ordem]
    postos = np.empty(len(valores), dtype=np.float64)
    inicio = 0
    while inicio < len(ordenados):
        fim = inicio
        while fim + 1 < len(ordenados) and ordenados[fim + 1] == ordenados[inicio]:
            fim += 1
        postos[ordem[inicio:fim + 1]] = (inicio + fim) / 2.0 + 1.0
        inicio = fim + 1
    return postos


def auroc(scores: Sequence[float], rotulos: Sequence[int]) -> float | None:
    """Mann-Whitney; empate vale meio ponto. None se faltar uma das classes."""
    s = arredondar(scores)
    y = np.asarray(rotulos, dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    postos = postos_medios(s)
    return float((postos[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(scores: Sequence[float], rotulos: Sequence[int]) -> float | None:
    """Precisao media (area sob precisao x revocacao em degraus), com empates tratados como um unico limiar."""
    s = arredondar(scores)
    y = np.asarray(rotulos, dtype=int)
    n_pos = int((y == 1).sum())
    if n_pos == 0 or n_pos == len(y):
        return None
    ordem = np.argsort(-s, kind="mergesort")
    s, y = s[ordem], y[ordem]
    limites = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]   # ultimo indice de cada grupo de empate
    verdadeiros = np.cumsum(y)[limites]
    previstos = limites + 1
    precisao = verdadeiros / previstos
    revocacao = verdadeiros / n_pos
    return float(np.sum(np.diff(np.r_[0.0, revocacao]) * precisao))


def por_painel(scores: Sequence[float], rotulos: Sequence[int], paineis: Sequence[str]) -> dict[str, Any]:
    """AUROC, n_P e n_B por painel. Painel sem as duas classes fica com AUROC None."""
    s, y, p = np.asarray(scores, dtype=np.float64), np.asarray(rotulos, dtype=int), np.asarray(paineis)
    saida = {}
    for painel in PAINEIS_DE_DISCRIMINACAO + PAINEIS_DE_GUARDA:
        mascara = p == painel
        saida[painel] = {"auroc": auroc(s[mascara], y[mascara]),
                         "n_pos": int((y[mascara] == 1).sum()), "n_neg": int((y[mascara] == 0).sum())}
    return saida


def macro(paineis: dict[str, Any]) -> float | None:
    """Macro NAO ponderada dos paineis de discriminacao. None se algum nao for avaliavel: melhor nao selecionar
    do que selecionar por uma macro incompleta, que muda de base entre configuracoes."""
    valores = [paineis[p]["auroc"] for p in PAINEIS_DE_DISCRIMINACAO]
    if any(v is None for v in valores):
        return None
    return float(sum(valores) / len(valores))


def resumo(scores: Sequence[float], rotulos: Sequence[int], paineis: Sequence[str]) -> dict[str, Any]:
    painel = por_painel(scores, rotulos, paineis)
    return {"macro": macro(painel), "auroc": auroc(scores, rotulos), "auprc": auprc(scores, rotulos),
            "por_painel": painel, "n": int(len(rotulos))}


def bootstrap_pareado_por_cluster(scores_a: Sequence[float], scores_b: Sequence[float], rotulos: Sequence[int],
                                  paineis: Sequence[str], clusters: Sequence[str], *, replicas: int = 1000,
                                  seed: int = 20260901) -> dict[str, Any]:
    """IC EXPLORATORIO de b - a reamostrando CLUSTERS, com os MESMOS sorteios para os dois sistemas.

    A unidade e o `overlap_cluster_id` (a do Mosaic, 1.000 replicas, seed 20260901): variantes do mesmo cluster
    nao sao independentes. Replica em que algum painel de discriminacao perde uma classe nao tem macro; ela e
    contada e fica fora so da macro.
    """
    a, b = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    y, p, c = np.asarray(rotulos, dtype=int), np.asarray(paineis), np.asarray(clusters).astype(str)
    if not (len(a) == len(b) == len(y) == len(p) == len(c)):
        raise ValueError("scores, rotulos, paineis e clusters precisam ter o mesmo tamanho")
    unicos = np.unique(c)
    linhas_do_cluster = {cluster: np.nonzero(c == cluster)[0] for cluster in unicos}
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {"macro": [], "auroc": [], "auprc": []}
    sem_macro = 0
    for _ in range(replicas):
        sorteio = rng.integers(0, len(unicos), size=len(unicos))
        linhas = np.concatenate([linhas_do_cluster[unicos[i]] for i in sorteio])
        ra, rb = resumo(a[linhas], y[linhas], p[linhas]), resumo(b[linhas], y[linhas], p[linhas])
        for chave in deltas:
            if ra[chave] is None or rb[chave] is None:
                if chave == "macro":
                    sem_macro += 1
                continue
            deltas[chave].append(rb[chave] - ra[chave])
    observado_a, observado_b = resumo(a, y, p), resumo(b, y, p)
    saida: dict[str, Any] = {"natureza": "exploratoria", "unidade": "overlap_cluster_id",
                             "clusters": int(len(unicos)), "replicas": replicas, "seed": seed,
                             "replicas_sem_macro": sem_macro}
    for chave, valores in deltas.items():
        pontual = (None if observado_a[chave] is None or observado_b[chave] is None
                   else observado_b[chave] - observado_a[chave])
        saida[chave] = {"delta": pontual,
                        "p2_5": float(np.percentile(valores, 2.5)) if valores else None,
                        "p97_5": float(np.percentile(valores, 97.5)) if valores else None,
                        "replicas_validas": len(valores)}
    return saida
