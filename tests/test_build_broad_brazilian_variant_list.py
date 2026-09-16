"""Prova que a regra ampla brasileira ignora o filtro P/B do Mosaic, que e justamente o ponto dela.

O diagnostico de 14-15/09 mediu 1.336 SCVs brasileiras descartadas pelo filtro (959 por origem nao germinativa,
630 por nao contribuirem para o agregado). Para EXCLUIR do treino, elas contam.
    PYTHONPATH=. python3 tests/test_build_broad_brazilian_variant_list.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.build_broad_brazilian_variant_list as ampla  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


INCLUDE = {"500035", "508087"}
ORGS = {"Mendelics": "500035", "Dasa": "508087", "GeneDx": "26957"}


def _org_of(name: str):
    return ORGS.get(name)


def _scv(scv, submitter):
    return NS(scv=scv, submitter=submitter)


def test_marca_scv_brasileira_que_o_filtro_pb_descartaria():
    # origem 'unknown' e sem contribuicao: fora do br_lab_any publicado, dentro da regra ampla.
    vids = {"var:a": ["1"]}
    scvs = {"1": [_scv("SCV1", "Mendelics")]}
    broad = ampla.broad_br_variants(vids, scvs, org_of=_org_of, include_ids=INCLUDE)
    assert broad == {"var:a": ["Mendelics"]}, broad


def test_nao_marca_submissor_nao_reconhecido_nem_internacional():
    vids = {"var:a": ["1"], "var:b": ["2"]}
    scvs = {"1": [_scv("SCV1", "GeneDx")], "2": [_scv("SCV2", "Laboratorio Desconhecido")]}
    assert ampla.broad_br_variants(vids, scvs, org_of=_org_of, include_ids=INCLUDE) == {}


def test_nao_conta_a_mesma_scv_duas_vezes_em_variation_ids_diferentes():
    vids = {"var:a": ["1", "2"]}
    scvs = {"1": [_scv("SCV1", "Dasa")], "2": [_scv("SCV1", "Dasa")]}
    broad = ampla.broad_br_variants(vids, scvs, org_of=_org_of, include_ids=INCLUDE)
    assert broad["var:a"] == ["Dasa"], broad


def test_comparacao_mostra_o_que_a_regra_ampla_acrescenta():
    broad = {"var:a": ["Dasa"], "var:b": ["Mendelics"]}
    published = {"var:a": True, "var:b": False, "var:c": False}
    got = ampla.compare_with_published(broad, published)
    assert got["regra_ampla"] == 2 and got["br_lab_any_publicado"] == 1
    assert got["so_na_regra_ampla"] == 1 and got["so_no_br_lab_any"] == 0, got


def test_lista_que_perde_variante_ja_marcada_no_release_e_reprovada():
    """Ponto 1 da revisao: o gerador tem de FALHAR, nao so relatar, quando a regra ampla nao e superconjunto."""
    problems = ampla.validate_broad_result({"var:a": ["Dasa"]}, {"var:a": True, "var:b": True})
    assert any("ficaram fora da regra ampla" in p for p in problems), problems


def test_lista_vazia_e_reprovada():
    problems = ampla.validate_broad_result({}, {"var:a": False})
    assert any("nenhuma variante marcada" in p for p in problems), problems


def test_lista_superconjunto_do_br_lab_any_passa():
    assert ampla.validate_broad_result({"var:a": ["Dasa"], "var:b": ["Mendelics"]}, {"var:a": True}) == []


def test_contagem_de_controles_usa_so_o_estudo_clinico():
    membership = pd.DataFrame([
        {"variant_id": "var:a", "study_id": "br_clinical_evidence", "member_role": "control"},
        {"variant_id": "var:b", "study_id": "br_clinical_evidence", "member_role": "control"},
        {"variant_id": "var:c", "study_id": "br_clinical_evidence", "member_role": "case"},
        {"variant_id": "var:d", "study_id": "br_population_observed", "member_role": "control"},
    ])
    got = ampla.controls_with_brazilian_scv(membership, {"var:a": ["Dasa"], "var:d": ["Dasa"]})
    assert got["controles_do_estudo_clinico"] == 2, got
    assert got["com_scv_brasileira_regra_ampla"] == 1 and got["fracao"] == 0.5, got


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
