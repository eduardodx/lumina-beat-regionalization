"""O comparador de avaliacoes calcula as diferencas e detecta quando uma conclusao troca de sinal.

So stdlib:
    PYTHONPATH=. python tests/test_probe_compare_evals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_compare_evals import compare  # noqa: E402


def _cfg(macro, missense):
    return {"macro": macro, "per_panel": {"missense": missense, "splice": 0.99, "noncoding": 0.95}}


def test_deltas_and_ranks():
    old = {"a": _cfg(0.90, 0.80), "b": _cfg(0.85, 0.75), "so_antigo": _cfg(0.5, 0.5)}
    new = {"a": _cfg(0.88, 0.79), "b": _cfg(0.89, 0.81), "so_novo": _cfg(0.5, 0.5)}
    res = compare(old, new, [])
    rows = {r["config"]: r for r in res["rows"]}
    assert set(rows) == {"a", "b"}, rows
    assert abs(rows["a"]["d_macro"] + 0.02) < 1e-12 and abs(rows["b"]["d_missense"] - 0.06) < 1e-12, rows
    assert (rows["a"]["rank_old"], rows["a"]["rank_new"]) == (1, 2), rows["a"]
    assert res["so_no_antigo"] == ["so_antigo"] and res["so_no_novo"] == ["so_novo"], res


def test_contrast_flags_a_conclusion_that_flips_sign():
    old = {"cabecas": _cfg(0.88, 0.75), "hidden": _cfg(0.89, 0.78), "max": _cfg(0.90, 0.77)}
    new = {"cabecas": _cfg(0.90, 0.79), "hidden": _cfg(0.87, 0.76), "max": _cfg(0.91, 0.78)}
    res = compare(old, new, ["cabecas:hidden", "max:hidden", "falta:hidden"])
    by = {c["contrast"]: c["sinal"] for c in res["contrasts"]}
    assert by["cabecas:hidden"] == {"macro": False, "missense": False}, by  # -0.01 -> +0.03 nos dois
    assert by["max:hidden"] == {"macro": True, "missense": False}, by       # macro +0.01 -> +0.04; missense -0.01 -> +0.02
    assert by["falta:hidden"] == {"macro": None, "missense": None}, "config ausente nao pode virar mantido/invertido"


def test_missense_flip_is_reported_even_when_macro_holds():
    # caso real do rerun (mlp/gene_transfer): honestos_mais_cabecas - honestos_mais_v2
    old = {"cabecas": _cfg(0.9372, 0.8513), "v2": _cfg(0.9257, 0.8386)}
    new = {"cabecas": _cfg(0.9379, 0.8540), "v2": _cfg(0.9301, 0.8542)}
    sinal = compare(old, new, ["cabecas:v2"])["contrasts"][0]["sinal"]
    assert sinal == {"macro": True, "missense": False}, sinal


def test_exact_zero_difference_has_no_sign():
    old = {"a": _cfg(0.9, 0.8), "b": _cfg(0.9, 0.8)}
    new = {"a": _cfg(0.91, 0.8), "b": _cfg(0.9, 0.81)}
    sinal = compare(old, new, ["a:b"])["contrasts"][0]["sinal"]
    assert sinal == {"macro": None, "missense": None}, sinal


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
