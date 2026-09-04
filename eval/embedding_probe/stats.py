"""Estatisticas de ranking da analise da sonda, em stdlib puro.

Por que reimplementar em vez de usar scipy: o `$PY` do host GPU e um venv enxuto (torch vem do
conda) e nao ha garantia de scipy nele; o resto da sonda ja segue a regra de manter a logica em
stdlib e deixar pandas/numpy so na borda de I/O. Com isso a analise roda no notebook, no Windows e
no CI sem instalar nada, e `tests/test_embedding_probe_stats.py` valida tudo sem ambiente
cientifico.

Todas as funcoes foram conferidas contra os valores que o scipy produziu na analise exploratoria
(ver o teste): Wilcoxon pareado, Mann-Whitney, Spearman e AUROC batem ate a precisao reportada.

Convencoes que espelham o scipy, para os numeros seguirem comparaveis:
  * ``wilcoxon``: descarta pares com diferenca zero (``zero_method='wilcox'``) e usa aproximacao
    normal SEM correcao de continuidade (``correction=False``) -- os defaults do scipy para n>25.
  * ``mann_whitney_u``: aproximacao normal com correcao de empates.
  * empates recebem rank medio em todos os testes.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Num = float


def rankdata(values: Sequence[Num]) -> list[float]:
    """Ranks 1-based com media nos empates (equivalente a ``scipy.stats.rankdata``)."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j + 2) / 2.0  # ranks 1-based de i+1 ate j+1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _tie_term(values: Sequence[Num]) -> float:
    """Somatorio de (t^3 - t) sobre os grupos de empate. Correcao de variancia."""
    counts: dict[Num, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return float(sum(t**3 - t for t in counts.values() if t > 1))


def _two_sided_p(z: float) -> float:
    """p bicaudal de um z-score. Usa erfc para nao perder precisao na cauda (p ~ 1e-37)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def auroc(positives: Sequence[Num], negatives: Sequence[Num]) -> float:
    """Area sob a ROC via estatistica de Mann-Whitney. Empates contam meio ponto."""
    n_pos, n_neg = len(positives), len(negatives)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(list(positives) + list(negatives))
    rank_sum = sum(ranks[:n_pos])
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mann_whitney_u(a: Sequence[Num], b: Sequence[Num]) -> tuple[float, float]:
    """(U da amostra ``a``, p bicaudal). Aproximacao normal com correcao de empates."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    pooled = list(a) + list(b)
    ranks = rankdata(pooled)
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    var = n1 * n2 * (n + 1) / 12.0
    ties = _tie_term(pooled)
    if ties:
        var -= n1 * n2 * ties / (12.0 * n * (n - 1))
    if var <= 0:
        return u1, float("nan")
    return u1, _two_sided_p((u1 - mu) / math.sqrt(var))


def wilcoxon(x: Sequence[Num], y: Sequence[Num]) -> tuple[float, float, int]:
    """Wilcoxon pareado. Retorna ``(W, p bicaudal, n de pares nao-nulos)``.

    ``W`` e o menor entre as somas de ranks positivos e negativos, como no scipy.
    """
    if len(x) != len(y):
        raise ValueError(f"amostras pareadas precisam do mesmo tamanho: {len(x)} != {len(y)}")
    diffs = [float(a) - float(b) for a, b in zip(x, y) if float(a) - float(b) != 0.0]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), 0
    abs_diffs = [abs(d) for d in diffs]
    ranks = rankdata(abs_diffs)
    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - _tie_term(abs_diffs) / 48.0
    if var <= 0:
        return w, float("nan"), n
    return w, _two_sided_p((w_pos - mu) / math.sqrt(var)), n


def spearman(a: Sequence[Num], b: Sequence[Num]) -> tuple[float, float]:
    """(rho de Spearman, p bicaudal pela aproximacao t com n-2 gl)."""
    n = len(a)
    if n != len(b):
        raise ValueError("as duas series precisam do mesmo tamanho")
    if n < 3:
        return float("nan"), float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if va == 0 or vb == 0:
        return float("nan"), float("nan")
    rho = cov / (va * vb)
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return rho, _student_t_two_sided(t, n - 2)


def _student_t_two_sided(t: float, df: int) -> float:
    """p bicaudal da t de Student via a beta incompleta regularizada."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Beta incompleta regularizada I_x(a,b) por fracao continua (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b + lbeta) / a
    if x >= (a + 1) / (a + b + 2):  # converge melhor no complemento
        return 1.0 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        delta = c * d
        f *= delta
        if abs(1.0 - delta) < 1e-12:
            break
    return front * (f - 1.0)


def median(values: Sequence[Num]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def quantile(values: Sequence[Num], q: float) -> float:
    """Quantil por interpolacao linear (mesmo metodo do numpy ``linear``)."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(s[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def bootstrap_auroc_ci(
    positives: Sequence[Num],
    negatives: Sequence[Num],
    *,
    n_boot: int = 2000,
    seed: int = 20260904,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """IC percentil da AUROC por bootstrap estratificado (reamostra P e B separadamente)."""
    rng = random.Random(seed)
    pos, neg = list(positives), list(negatives)
    if not pos or not neg:
        return float("nan"), float("nan")
    draws = [
        auroc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg])
        for _ in range(n_boot)
    ]
    return quantile(draws, alpha / 2), quantile(draws, 1 - alpha / 2)


def variance_decomposition(groups: Sequence[Sequence[Num]]) -> dict[str, float]:
    """ANOVA de 1 fator: quanto da variancia total e ENTRE grupos e quanto e DENTRO.

    Usada para separar "quanto do Delta e explicado pelo contexto genomico (o sitio)" de "quanto e
    explicado pela identidade do alelo (a variacao dentro do sitio)". Compara somas de quadrados --
    nao um range contra um desvio-padrao, que sao escalas diferentes.
    """
    flat = [v for g in groups for v in g]
    n = len(flat)
    if n == 0:
        return {"ss_between": float("nan"), "ss_within": float("nan"),
                "frac_between": float("nan"), "frac_within": float("nan"), "sd_ratio": float("nan")}
    grand = sum(flat) / n
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups if g)
    ss_within = sum((v - sum(g) / len(g)) ** 2 for g in groups if g for v in g)
    total = ss_between + ss_within
    return {
        "ss_between": ss_between,
        "ss_within": ss_within,
        "frac_between": ss_between / total if total else float("nan"),
        "frac_within": ss_within / total if total else float("nan"),
        "sd_ratio": math.sqrt(ss_within / ss_between) if ss_between else float("nan"),
    }
