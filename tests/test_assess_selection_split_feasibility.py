"""Prova que a viabilidade do conjunto de selecao e medida por celula (painel x classe) e com o custo no treino.

Mais clusters no total nao garante suporte: o recorte tem de atingir o alvo em CADA celula, e o que ele leva sai
do treino de todos os candidatos.
    PYTHONPATH=. python3 tests/test_assess_selection_split_feasibility.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.assess_selection_split_feasibility as sel  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _rows(spec):
    """spec: (cluster, painel, rotulo, n) -> linhas de treino."""
    rows = []
    for i, (cluster, panel, label, n) in enumerate(spec):
        for j in range(n):
            rows.append({"variant_id": f"var:{i}_{j}", "role": "train", "binary_label": label,
                         "primary_panel": panel, "overlap_cluster_id": cluster, "label_tier": "consensus"})
    return pd.DataFrame(rows)


def test_suporte_conta_exemplos_e_clusters_por_celula():
    rows = _rows([("cl_1", "missense", 1, 3), ("cl_2", "missense", 1, 2), ("cl_1", "splice", 0, 1)])
    got = sel.support_by_cell(rows)
    assert got["missense/P"] == {"n": 5, "clusters": 2}, got
    assert got["splice/B"] == {"n": 1, "clusters": 1}, got
    assert got["noncoding/B"] == {"n": 0, "clusters": 0}, got


def test_concentracao_mostra_quando_a_celula_mora_em_poucos_clusters():
    rows = _rows([("cl_1", "splice", 0, 9), ("cl_2", "splice", 0, 1)])
    got = sel.concentration(rows, top=1)
    assert got["splice/B"]["maior_cluster"] == 9
    assert got["splice/B"]["fracao_nos_1_maiores"] == 0.9, got
    assert got["splice/B"]["clusters_para_80pc"] == 1, got


def test_recorte_atinge_o_alvo_em_todas_as_celulas_quando_ha_suporte():
    spec = [(f"cl_{i}", panel, label, 5)
            for i in range(10) for panel in sel.DISCRIMINATION_PANELS for label in (1, 0)]
    report = sel.build_report(_rows(spec), target=10, top=3)
    assert report["recorte_proposto"]["celulas_nao_atingidas"] == [], report["recorte_proposto"]
    assert all(report["recorte_proposto"]["atingiu_o_alvo"].values())
    assert report["recorte_proposto"]["clusters"] <= 10


def test_recorte_declara_a_celula_que_nao_da_para_atingir():
    spec = [("cl_1", "missense", 1, 100), ("cl_1", "missense", 0, 100),
            ("cl_2", "splice", 1, 100), ("cl_2", "splice", 0, 3),
            ("cl_3", "noncoding", 1, 100), ("cl_3", "noncoding", 0, 100)]
    report = sel.build_report(_rows(spec), target=50, top=3)
    assert report["recorte_proposto"]["celulas_nao_atingidas"] == ["splice/B"], report["recorte_proposto"]


def test_custo_no_treino_e_reportado_com_o_recorte():
    spec = [("cl_1", "missense", 1, 50), ("cl_2", "missense", 1, 50), ("cl_2", "missense", 0, 50),
            ("cl_3", "splice", 0, 50), ("cl_3", "splice", 1, 50),
            ("cl_4", "noncoding", 1, 50), ("cl_4", "noncoding", 0, 50)]
    report = sel.build_report(_rows(spec), target=25, top=3)
    reservados = report["recorte_proposto"]["n"]
    assert reservados > 0
    assert report["custo_no_treino"]["n_depois"] == 350 - reservados, report["custo_no_treino"]
    total_clusters = report["teto_no_treino"]["clusters"]
    assert report["custo_no_treino"]["clusters_depois"] == total_clusters - report["recorte_proposto"]["clusters"]


def test_recorte_e_deterministico():
    spec = [(f"cl_{i}", panel, label, 7)
            for i in range(6) for panel in sel.DISCRIMINATION_PANELS for label in (1, 0)]
    rows = _rows(spec)
    primeiro = sel.build_report(rows, target=14, top=3)["clusters_reservados"]
    segundo = sel.build_report(rows.sample(frac=1, random_state=7), target=14, top=3)["clusters_reservados"]
    assert primeiro == segundo, (primeiro, segundo)


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
