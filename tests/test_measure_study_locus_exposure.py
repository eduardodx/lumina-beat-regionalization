"""Prova que a exposicao de locus e medida por cluster e comparada PAR A PAR entre caso e controle.

A assimetria caso x controle e o unico caminho pelo qual a exposicao compartilhada pode atravessar a interacao:
a exposicao em si e identica para o sistema base e o regionalizado, que treinam no mesmo snapshot.
    PYTHONPATH=. python3 tests/test_measure_study_locus_exposure.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.measure_study_locus_exposure as exp  # noqa: E402

CLINICAL = "br_clinical_evidence"


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _snapshot(rows):
    return pd.DataFrame(rows, columns=["variant_id", "role", "binary_label", "overlap_cluster_id"])


def _members(rows):
    return pd.DataFrame(rows, columns=["variant_id", "study_id", "member_role", "matched_variant_id",
                                       "binary_label", "primary_panel", "overlap_cluster_id"])


def test_conta_so_o_treino_e_separa_p_de_b():
    snapshot = _snapshot([
        ("var:t1", "train", 1, "cl_a"), ("var:t2", "train", 0, "cl_a"), ("var:t3", "train", 1, "cl_b"),
        ("var:v1", "validation", 1, "cl_a"), ("var:s1", "test", 0, "cl_a"),
    ])
    got = exp.train_exposure_by_cluster(snapshot).set_index("overlap_cluster_id")
    assert got.loc["cl_a", "n_treino"] == 2 and got.loc["cl_a", "n_treino_P"] == 1
    assert got.loc["cl_a", "n_treino_B"] == 1 and got.loc["cl_b", "n_treino"] == 1
    assert "cl_c" not in got.index


def test_membro_em_cluster_sem_treino_fica_com_exposicao_zero():
    members = _members([("var:a", CLINICAL, "case", "var:b", 1, "missense", "cl_sem_treino")])
    exposure = exp.train_exposure_by_cluster(_snapshot([("var:t", "train", 1, "cl_outro")]))
    joined = exp.join_exposure(members, exposure)
    assert list(joined["n_treino"]) == [0], joined


def test_comparacao_pareada_detecta_assimetria():
    members = _members([
        ("var:a", CLINICAL, "case", "var:b", 1, "missense", "cl_quente"),
        ("var:b", CLINICAL, "control", "var:a", 1, "missense", "cl_frio"),
        ("var:c", CLINICAL, "case", "var:d", 1, "missense", "cl_quente"),
        ("var:d", CLINICAL, "control", "var:c", 1, "missense", "cl_frio"),
    ])
    snapshot = _snapshot([(f"var:t{i}", "train", 1, "cl_quente") for i in range(10)]
                         + [("var:t10", "train", 0, "cl_frio")])
    joined = exp.join_exposure(members, exp.train_exposure_by_cluster(snapshot))
    got = exp.paired_comparison(joined, CLINICAL)
    assert got["pares"] == 2
    assert got["caso"]["mediana"] == 10 and got["controle"]["mediana"] == 1
    assert got["diferenca_caso_menos_controle"]["fracao_caso_maior"] == 1.0, got


def test_comparacao_pareada_simetrica_da_meio_a_meio():
    members = _members([
        ("var:a", CLINICAL, "case", "var:b", 1, "missense", "cl_1"),
        ("var:b", CLINICAL, "control", "var:a", 1, "missense", "cl_2"),
        ("var:c", CLINICAL, "case", "var:d", 1, "missense", "cl_2"),
        ("var:d", CLINICAL, "control", "var:c", 1, "missense", "cl_1"),
    ])
    snapshot = _snapshot([("var:t1", "train", 1, "cl_1"), ("var:t2", "train", 1, "cl_1"),
                          ("var:t3", "train", 0, "cl_2")])
    got = exp.paired_comparison(exp.join_exposure(members, exp.train_exposure_by_cluster(snapshot)), CLINICAL)
    assert got["diferenca_caso_menos_controle"]["fracao_caso_maior"] == 0.5, got
    assert got["diferenca_caso_menos_controle"]["media"] == 0.0, got


def test_casos_sem_par_ficam_fora_da_comparacao_pareada():
    members = _members([
        ("var:a", CLINICAL, "case", "var:b", 1, "missense", "cl_1"),
        ("var:b", CLINICAL, "control", "var:a", 1, "missense", "cl_1"),
        ("var:z", CLINICAL, "unmatched_case", None, 1, "missense", "cl_1"),
    ])
    snapshot = _snapshot([("var:t1", "train", 1, "cl_1")])
    got = exp.paired_comparison(exp.join_exposure(members, exp.train_exposure_by_cluster(snapshot)), CLINICAL)
    assert got["pares"] == 1, got


def test_describe_marca_a_fracao_sem_exposicao():
    got = exp.describe(pd.Series([0, 0, 5, 10]))
    assert got["zero"] == 2 and got["fracao_zero"] == 0.5 and got["maximo"] == 10, got


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed, skipped = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Skip as exc:
            skipped.append(name)
            print(f"  SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    ran = len(tests) - failed - len(skipped)
    tail = f"  |  {len(skipped)} PULADO(S), sem cobertura: {', '.join(skipped)}" if skipped else ""
    print(f"\n{ran}/{len(tests) - len(skipped)} passaram{tail}")
    if skipped and os.environ.get("REQUIRE_NO_SKIP"):
        print("REQUIRE_NO_SKIP: teste pulado conta como falha")
        sys.exit(1)
    sys.exit(1 if failed else 0)
