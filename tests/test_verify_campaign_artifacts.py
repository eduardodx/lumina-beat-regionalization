"""Prova que o portao de artefatos reprova o que o booleano `final_para_treino` deixaria passar.

`final_para_treino` diz so que UMA lista de clusters foi passada ao G2 -- nao que foi a certa, nem que o resultado
ficou disjunto. Este portao confere contra os artefatos.
    PYTHONPATH=. python3 tests/test_verify_campaign_artifacts.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.verify_campaign_artifacts as ver  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _snapshot(rows):
    return pd.DataFrame(rows, columns=["variant_id", "role", "binary_label", "overlap_cluster_id"])


def _selecao(rows):
    return pd.DataFrame(rows, columns=["variant_id", "overlap_cluster_id", "binary_label", "primary_panel"])


SELECAO = _selecao([("var:s1", "cl_sel", 1, "missense"), ("var:s2", "cl_sel", 0, "splice")])
LIMPO = [("var:t1", "train", 1, "cl_a"), ("var:t2", "train", 0, "cl_b"),
         ("var:v1", "validation", 1, "cl_c"), ("var:x1", "test", 0, "cl_d")]


def test_snapshot_limpo_passa():
    problems, resumo = ver.check_snapshot("ok", _snapshot(LIMPO), selection=SELECAO, members=set())
    assert problems == [], problems
    assert resumo["treino"] == 2 and resumo["treino_P"] == 1 and resumo["validacao"] == 1


def test_variante_do_conjunto_de_selecao_dentro_do_snapshot_reprova():
    rows = LIMPO + [("var:s1", "train", 1, "cl_a")]
    problems, _ = ver.check_snapshot("x", _snapshot(rows), selection=SELECAO, members=set())
    assert any("variantes do conjunto de selecao dentro do snapshot" in p for p in problems), problems


def test_cluster_do_conjunto_de_selecao_no_treino_reprova():
    rows = LIMPO + [("var:outra", "train", 1, "cl_sel")]
    problems, _ = ver.check_snapshot("x", _snapshot(rows), selection=SELECAO, members=set())
    assert any("clusters do conjunto de selecao no treino" in p for p in problems), problems


def test_membro_de_estudo_no_snapshot_reprova():
    problems, _ = ver.check_snapshot("x", _snapshot(LIMPO), selection=SELECAO, members={"var:v1"})
    assert any("membros dos estudos dentro do snapshot" in p for p in problems), problems


def test_validacao_diferente_entre_candidatos_reprova():
    a = ver.role_signature(_snapshot(LIMPO), "validation")
    outro = _snapshot(LIMPO + [("var:v2", "validation", 0, "cl_e")])
    b = ver.role_signature(outro, "validation")
    teste = ver.role_signature(_snapshot(LIMPO), "test")
    problems = ver.check_comparability({
        "um": {"validation": a, "test": teste},
        "dois": {"validation": b, "test": teste},
    })
    assert any("validation difere entre candidatos" in p for p in problems), problems


def test_mesmos_exemplos_em_ordem_diferente_sao_comparaveis():
    embaralhado = _snapshot(list(reversed(LIMPO)))
    assinatura_a = ver.role_signature(_snapshot(LIMPO), "validation")
    assinatura_b = ver.role_signature(embaralhado, "validation")
    assert assinatura_a == assinatura_b
    assert ver.check_comparability({
        "um": {"validation": assinatura_a, "test": ver.role_signature(_snapshot(LIMPO), "test")},
        "dois": {"validation": assinatura_b, "test": ver.role_signature(embaralhado, "test")},
    }) == []


def test_end_to_end_sai_com_codigo_2_quando_ha_vazamento():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        sel_path = raiz / "selecao.parquet"
        SELECAO.to_parquet(sel_path, index=False)
        membros = pd.DataFrame({"variant_id": ["var:membro"]})
        membros_path = raiz / "membros.parquet"
        membros.to_parquet(membros_path, index=False)

        bom = raiz / "bom.parquet"
        _snapshot(LIMPO).to_parquet(bom, index=False)
        ruim = raiz / "ruim.parquet"
        _snapshot(LIMPO + [("var:s1", "train", 1, "cl_a")]).to_parquet(ruim, index=False)

        ok = ver.main(["--selection", str(sel_path), "--brazil-variants", str(membros_path),
                       "--snapshot", f"bom={bom}"])
        assert ok == 0

        falha = ver.main(["--selection", str(sel_path), "--brazil-variants", str(membros_path),
                          "--snapshot", f"bom={bom}", "--snapshot", f"ruim={ruim}"])
        assert falha == 2


def test_hash_declarado_diferente_reprova():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        sel_path = raiz / "selecao.parquet"
        SELECAO.to_parquet(sel_path, index=False)
        membros_path = raiz / "membros.parquet"
        pd.DataFrame({"variant_id": []}).to_parquet(membros_path, index=False)
        snap = raiz / "snap.parquet"
        _snapshot(LIMPO).to_parquet(snap, index=False)
        esperado = raiz / "esperado.json"
        esperado.write_text('{"%s": "%s"}' % (str(snap).replace("\\", "\\\\"), "0" * 64), encoding="utf-8")

        assert ver.main(["--selection", str(sel_path), "--brazil-variants", str(membros_path),
                         "--snapshot", f"snap={snap}", "--esperado", str(esperado)]) == 2


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
