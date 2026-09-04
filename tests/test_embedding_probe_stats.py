"""Testes das estatisticas da sonda. Stdlib puro: ``python tests/test_embedding_probe_stats.py``.

Alem dos casos sinteticos, ha uma classe de teste que trava os valores que o **scipy** produziu na
analise exploratoria dos dados reais (2026-09-04). Se a reimplementacao em stdlib divergir do
scipy, esses testes quebram -- e a analise publicada depende dessa equivalencia.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedding_probe.stats import (  # noqa: E402
    auroc,
    bootstrap_auroc_ci,
    mann_whitney_u,
    median,
    quantile,
    rankdata,
    spearman,
    variance_decomposition,
    wilcoxon,
)


@contextmanager
def assert_raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


# --------------------------------------------------------------------------------------------
# rankdata
# --------------------------------------------------------------------------------------------


def test_rankdata_simple():
    assert rankdata([10, 20, 30]) == [1.0, 2.0, 3.0]
    assert rankdata([30, 10, 20]) == [3.0, 1.0, 2.0]


def test_rankdata_averages_ties():
    assert rankdata([1, 2, 2, 3]) == [1.0, 2.5, 2.5, 4.0]
    assert rankdata([5, 5, 5]) == [2.0, 2.0, 2.0]


def test_rankdata_empty():
    assert rankdata([]) == []


# --------------------------------------------------------------------------------------------
# auroc
# --------------------------------------------------------------------------------------------


def test_auroc_perfect_and_inverted():
    close(auroc([3, 4, 5], [0, 1, 2]), 1.0)
    close(auroc([0, 1, 2], [3, 4, 5]), 0.0)


def test_auroc_chance_on_identical():
    close(auroc([1, 1, 1], [1, 1, 1]), 0.5)


def test_auroc_ties_count_half():
    # pares (pos,neg): (2,1)=1 · (2,2)=empate 0.5 · (3,1)=1 · (3,2)=1  ->  3.5/4
    close(auroc([2, 3], [1, 2]), 0.875)


def test_auroc_empty_is_nan():
    assert auroc([], [1, 2]) != auroc([], [1, 2])  # NaN != NaN


# --------------------------------------------------------------------------------------------
# testes de hipotese
# --------------------------------------------------------------------------------------------


def test_wilcoxon_detects_consistent_shift():
    x = list(range(1, 21))
    y = [v - 3 for v in x]  # x sempre maior
    w, p, n = wilcoxon(x, y)
    assert n == 20 and p < 1e-3, (w, p, n)


def test_wilcoxon_no_difference_is_not_significant():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    _, p, _ = wilcoxon(x, y)
    assert p > 0.5, p


def test_wilcoxon_drops_zero_pairs():
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 9]  # 3 pares nulos, 1 nao-nulo
    _, _, n = wilcoxon(x, y)
    assert n == 1


def test_wilcoxon_rejects_mismatched_lengths():
    with assert_raises(ValueError):
        wilcoxon([1, 2], [1])


def test_mann_whitney_separated_groups():
    _, p = mann_whitney_u(list(range(20, 40)), list(range(20)))
    assert p < 1e-5, p


def test_mann_whitney_identical_groups():
    _, p = mann_whitney_u([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert p > 0.9, p


def test_spearman_monotonic():
    rho, p = spearman([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 6, 8, 10, 12, 14, 16])
    close(rho, 1.0, 1e-12)
    assert p < 0.001


def test_spearman_inverse_and_flat():
    rho, _ = spearman([1, 2, 3, 4], [4, 3, 2, 1])
    close(rho, -1.0, 1e-12)
    rho2, _ = spearman([1, 2, 3, 4], [7, 7, 7, 7])
    assert rho2 != rho2  # NaN: variancia zero


# --------------------------------------------------------------------------------------------
# equivalencia com o scipy (valores reais medidos em 2026-09-04)
# --------------------------------------------------------------------------------------------


def test_matches_scipy_wilcoxon_on_known_pairs():
    """scipy.stats.wilcoxon nos mesmos dados sinteticos, valores pre-computados."""
    x = [2.1, 3.4, 1.8, 5.0, 4.2, 3.9, 2.7, 6.1, 5.5, 4.8]
    y = [1.9, 3.6, 1.5, 4.1, 4.4, 3.1, 2.9, 5.2, 5.9, 4.0]
    w, p, n = wilcoxon(x, y)
    assert n == 10
    close(w, 13.0, 1e-9)              # scipy: statistic=13.0
    close(p, 0.138128, 1e-6)          # scipy method="approx", correction=False


def test_matches_scipy_mann_whitney():
    a = [12, 15, 9, 20, 18, 14, 22, 17]
    b = [7, 10, 6, 11, 8, 13, 5, 9]
    u, p = mann_whitney_u(a, b)
    close(u, 59.5, 1e-9)              # scipy: statistic=59.5
    close(p, 0.003850, 1e-6)          # scipy method="asymptotic", use_continuity=False


def test_matches_scipy_spearman():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    rho, p = spearman(a, b)
    close(rho, 0.939393939394, 1e-9)  # scipy: statistic=0.939393939
    close(p, 0.00005484, 1e-7)        # scipy: pvalue=5.484e-05


def test_auroc_matches_rank_definition():
    """AUROC tem que ser identica a U/(n1*n2) -- as duas rotas do mesmo numero."""
    pos = [0.9, 0.7, 0.6, 0.8, 0.55]
    neg = [0.5, 0.3, 0.65, 0.2, 0.4, 0.45]
    u, _ = mann_whitney_u(pos, neg)
    close(auroc(pos, neg), u / (len(pos) * len(neg)), 1e-12)


# --------------------------------------------------------------------------------------------
# medianas, quantis, bootstrap, variancia
# --------------------------------------------------------------------------------------------


def test_median_and_quantile():
    close(median([3, 1, 2]), 2.0)
    close(median([4, 1, 3, 2]), 2.5)
    close(quantile([1, 2, 3, 4, 5], 0.5), 3.0)      # numpy: 3.0
    close(quantile([1, 2, 3, 4], 0.25), 1.75)       # numpy linear: 1.75
    close(quantile([10], 0.9), 10.0)


def test_bootstrap_ci_brackets_point_estimate():
    pos = [0.9, 0.85, 0.8, 0.95, 0.7, 0.88, 0.92, 0.78]
    neg = [0.2, 0.3, 0.15, 0.4, 0.25, 0.35, 0.1, 0.28]
    point = auroc(pos, neg)
    lo, hi = bootstrap_auroc_ci(pos, neg, n_boot=400, seed=7)
    assert lo <= point <= hi, (lo, point, hi)
    assert lo >= 0.0 and hi <= 1.0


def test_bootstrap_ci_is_reproducible():
    pos, neg = [3, 4, 5, 6], [1, 2, 3, 4]
    assert bootstrap_auroc_ci(pos, neg, n_boot=200, seed=1) == \
           bootstrap_auroc_ci(pos, neg, n_boot=200, seed=1)


def test_variance_decomposition_all_between():
    """Grupos internamente constantes: toda a variancia e ENTRE grupos."""
    d = variance_decomposition([[1, 1, 1], [5, 5, 5], [9, 9, 9]])
    close(d["frac_between"], 1.0, 1e-12)
    close(d["frac_within"], 0.0, 1e-12)


def test_variance_decomposition_all_within():
    """Grupos com a mesma media: toda a variancia e DENTRO."""
    d = variance_decomposition([[0, 10], [10, 0], [5, 5]])
    close(d["frac_between"], 0.0, 1e-12)
    close(d["frac_within"], 1.0, 1e-12)


def test_variance_decomposition_fractions_sum_to_one():
    d = variance_decomposition([[1, 2, 3], [4, 8], [2, 2, 7, 1]])
    close(d["frac_between"] + d["frac_within"], 1.0, 1e-12)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passaram")
    sys.exit(1 if failed else 0)
