"""Testes do protocolo de avaliacao. Stdlib puro: ``python tests/test_embedding_probe_protocol.py``.

O que estes testes protegem: se o split vazar unidade de bloqueio entre papeis, ou se a agenda de
folds sair do contrato do Mosaic, toda a ablacao de features mede a coisa errada -- e o numero final
pareceria bom justamente por estar contaminado.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedding_probe.protocol import (  # noqa: E402
    DISCRIMINATION_PANELS,
    K_FOLDS,
    Split,
    assert_no_unit_leak,
    cross_fitted_split,
    fold_schedule,
    macro_auroc,
    weighted_cross_fold,
    worst_panel,
)


@contextmanager
def assert_raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b}"


# --------------------------------------------------------------------------------------------
# agenda de folds
# --------------------------------------------------------------------------------------------


def test_schedule_matches_mosaic_contract():
    """teste = i · validation = (i+1) mod 5 · treino = os outros tres."""
    for i in range(K_FOLDS):
        test, val, train = fold_schedule(i)
        assert test == i
        assert val == (i + 1) % K_FOLDS
        assert len(train) == 3
        assert set(train) | {test, val} == set(range(K_FOLDS))
        assert test not in train and val not in train


def test_schedule_rejects_bad_run_id():
    for bad in (-1, 5, 99):
        with assert_raises(ValueError):
            fold_schedule(bad)


def test_every_fold_is_tested_exactly_once_across_runs():
    assert sorted(fold_schedule(i)[0] for i in range(K_FOLDS)) == list(range(K_FOLDS))


# --------------------------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------------------------


def _toy():
    """5 folds x 2 tiers, uma variante de cada combinacao. folds[i] e tiers[i] alinhados."""
    folds, tiers = [], []
    for f in range(K_FOLDS):
        for t in ("gold", "consensus"):
            folds.append(f)
            tiers.append(t)
    return folds, tiers


def test_test_and_validation_are_gold_only():
    folds, tiers = _toy()
    for run in range(K_FOLDS):
        s = cross_fitted_split(folds, tiers, run_id=run)
        assert all(tiers[i] == "gold" for i in s.test), "consensus vazou para o teste"
        assert all(tiers[i] == "gold" for i in s.validation), "consensus vazou para a validation"


def test_train_takes_both_tiers_from_the_three_remaining_folds():
    folds, tiers = _toy()
    s = cross_fitted_split(folds, tiers, run_id=0)
    _, _, train_folds = fold_schedule(0)
    assert {folds[i] for i in s.train} == set(train_folds)
    assert {tiers[i] for i in s.train} == {"gold", "consensus"}
    assert len(s.train) == 6  # 3 folds x 2 tiers


def test_consensus_of_test_and_validation_folds_never_trains():
    """A regra explicita do protocolo."""
    folds, tiers = _toy()
    for run in range(K_FOLDS):
        s = cross_fitted_split(folds, tiers, run_id=run)
        test_fold, val_fold, _ = fold_schedule(run)
        assert not any(folds[i] in (test_fold, val_fold) for i in s.train)


def test_gold_only_training_is_opt_in():
    folds, tiers = _toy()
    s = cross_fitted_split(folds, tiers, run_id=0, train_tiers=("gold",))
    assert {tiers[i] for i in s.train} == {"gold"}
    assert len(s.train) == 3


def test_roles_are_disjoint():
    folds, tiers = _toy()
    for run in range(K_FOLDS):
        s = cross_fitted_split(folds, tiers, run_id=run)
        assert not (set(s.train) & set(s.validation))
        assert not (set(s.train) & set(s.test))
        assert not (set(s.validation) & set(s.test))


def test_none_fold_is_excluded_entirely():
    folds = [0, 1, None, 3]
    tiers = ["gold"] * 4
    for run in range(K_FOLDS):
        s = cross_fitted_split(folds, tiers, run_id=run)
        assert 2 not in s.train + s.validation + s.test


def test_split_rejects_mismatched_lengths():
    with assert_raises(ValueError):
        cross_fitted_split([0, 1], ["gold"], run_id=0)


# --------------------------------------------------------------------------------------------
# vazamento de unidade de bloqueio
# --------------------------------------------------------------------------------------------


def test_unit_leak_passes_when_folds_respect_the_unit():
    """Release bem formado: a unidade determina o fold, entao nunca cruza papeis."""
    folds, tiers, units = [], [], []
    for f in range(K_FOLDS):
        for j in range(3):
            folds.append(f); tiers.append("gold"); units.append(f"ovl:{f}:{j}")
    for run in range(K_FOLDS):
        assert_no_unit_leak(cross_fitted_split(folds, tiers, run_id=run), units)


def test_unit_leak_is_detected():
    """Mesma unidade em dois folds -> o split contamina e a assercao TEM que gritar."""
    folds = [0, 1, 2, 3]
    tiers = ["gold"] * 4
    units = ["ovl:X", "ovl:X", "ovl:B", "ovl:C"]  # a mesma unidade nos folds 0 e 1
    with assert_raises(AssertionError):
        assert_no_unit_leak(cross_fitted_split(folds, tiers, run_id=0), units)


def test_unit_leak_message_names_the_roles():
    units = ["u", "u"]
    split = Split(run_id=0, train=[0], validation=[1], test=[])
    try:
        assert_no_unit_leak(split, units)
    except AssertionError as exc:
        assert "train" in str(exc) and "validation" in str(exc)
    else:
        raise AssertionError("deveria ter detectado o vazamento")


# --------------------------------------------------------------------------------------------
# metricas de selecao e agregacao
# --------------------------------------------------------------------------------------------


def test_macro_is_unweighted_over_the_three_discrimination_panels():
    per = {"missense": 0.6, "splice": 0.9, "noncoding": 0.75, "plof": 0.99, "synonymous": 0.2}
    close(macro_auroc(per), (0.6 + 0.9 + 0.75) / 3)  # guardrails NAO entram


def test_macro_is_none_when_a_discrimination_panel_is_missing():
    assert macro_auroc({"missense": 0.6, "splice": 0.9}) is None
    assert macro_auroc({"missense": 0.6, "splice": 0.9, "noncoding": None}) is None


def test_worst_panel():
    close(worst_panel({"missense": 0.6, "splice": 0.9, "noncoding": 0.75}), 0.6)
    assert worst_panel({"missense": 0.6, "splice": 0.9}) is None


def test_weighted_cross_fold_uses_np_times_nb():
    # fold A: AUROC .6 com 10x10=100 de peso; fold B: .9 com 1x1=1
    got = weighted_cross_fold([(0.6, 10, 10), (0.9, 1, 1)])
    close(got, (0.6 * 100 + 0.9 * 1) / 101)


def test_weighted_cross_fold_skips_single_class_folds():
    got = weighted_cross_fold([(0.8, 5, 5), (float("nan"), 4, 0)])
    close(got, 0.8)
    assert weighted_cross_fold([(float("nan"), 3, 0)]) is None
    assert weighted_cross_fold([]) is None


def test_discrimination_panels_are_the_mosaic_three():
    assert DISCRIMINATION_PANELS == ("missense", "splice", "noncoding")


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
