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
    by = {c["contrast"]: c for c in res["contrasts"]}
    assert by["cabecas:hidden"]["sinal_mantido"] is False, by["cabecas:hidden"]   # -0.01 -> +0.03
    assert by["max:hidden"]["sinal_mantido"] is True, by["max:hidden"]            # +0.01 -> +0.04
    assert by["falta:hidden"]["sinal_mantido"] is None, "config ausente nao pode virar mantido/invertido"


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
