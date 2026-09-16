"""Prova que o G1 so importa o membership brasileiro quando ele cumpre o que o protocolo do Mosaic promete.

Os testes puros e o de ponta a ponta usam um release sintetico (pandas + pyarrow). O ultimo teste usa o release
real, se existir (MOSAIC_RELEASE ou ~/mosaic-v1):
    PYTHONPATH=. python3 tests/test_import_mosaic_brazil_studies.py
Com REQUIRE_NO_SKIP=1, teste pulado conta como falha (use no notebook antes de rodar o G1).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.import_mosaic_brazil_studies as g1  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


CLINICAL = g1.STUDY_CLINICAL
POPULATION = g1.STUDY_POPULATION


def _member(vid, study, role, matched=None, label=1, panel="missense", af_bin="rare", tier="consensus",
            cluster=None, fold=0, br=True, abraom=False):
    return {
        "variant_id": vid, "study_id": study, "member_role": role, "matched_variant_id": matched,
        "stratum": g1.stratum_key(label, panel, af_bin), "label_tier": tier, "binary_label": label,
        "primary_panel": panel, "gnomad_af_bin": af_bin, "overlap_cluster_id": cluster or f"cl_{vid}",
        "core_fold": fold, "br_lab_any": br, "present_abraom": abraom,
    }


def _valid_membership() -> list[dict]:
    return [
        _member("var:a", CLINICAL, "case", matched="var:b"),
        _member("var:b", CLINICAL, "control", matched="var:a", br=False),
        _member("var:c", CLINICAL, "unmatched_case", label=0, panel="splice", af_bin=None),
        _member("var:d", POPULATION, "case", matched="var:e", tier="gold", abraom=True, panel="noncoding"),
        _member("var:e", POPULATION, "control", matched="var:d", tier="gold", br=False, panel="noncoding"),
    ]


def _release_tables(membership: list[dict]) -> dict[str, pd.DataFrame]:
    rows = pd.DataFrame(membership)
    unique = rows.drop_duplicates(subset=["variant_id"])
    # Listas novas em cada tabela: com `to_numpy()` o pandas pode compartilhar o buffer e mudar `partitions`
    # mudaria tambem o `membership`, o que faria o teste de divergencia deixar de testar o que promete.
    examples = pd.DataFrame({
        "variant_id": list(unique["variant_id"]),
        "chrom": ["chr1"] * len(unique),
        "pos_1based": list(range(1000, 1000 + len(unique))),
        "ref": ["A"] * len(unique),
        "alt": ["G"] * len(unique),
        "binary_label": list(unique["binary_label"]),
        "label_tier": list(unique["label_tier"]),
        "sequence_eligible": [True] * len(unique),
    })
    panels = pd.DataFrame({"variant_id": list(unique["variant_id"]),
                           "primary_panel": list(unique["primary_panel"])})
    partitions = pd.DataFrame({
        "variant_id": list(unique["variant_id"]),
        "overlap_cluster_id": list(unique["overlap_cluster_id"]),
        "core_fold": list(unique["core_fold"]),
    })
    return {"membership": rows.copy(), "examples": examples, "panels": panels, "partitions": partitions}


def _write_release(root: Path, tables: dict[str, pd.DataFrame]) -> None:
    (root / "studies" / "brazil").mkdir(parents=True, exist_ok=True)
    tables["membership"].to_parquet(root / "studies/brazil/membership.parquet", index=False)
    tables["examples"].to_parquet(root / "pb_examples.parquet", index=False)
    tables["panels"].to_parquet(root / "pb_panels.parquet", index=False)
    tables["partitions"].to_parquet(root / "pb_partitions.parquet", index=False)


# ------------------------------------------------------------------------------------------------ testes puros


def test_stratum_key_replica_o_mosaic():
    assert g1.stratum_key(1, "missense", "rare") == "1|missense|rare"
    assert g1.stratum_key(0, "splice", None) == "0|splice|missing"
    assert g1.stratum_key(0, "splice", float("nan")) == "0|splice|missing"


def test_membership_valido_passa_em_todas_as_checagens():
    tables = _release_tables(_valid_membership())
    assert g1.validate(tables) == []


def test_par_unidirecional_reprova():
    membership = _valid_membership()
    membership[1]["matched_variant_id"] = "var:zzz"  # controle aponta para outro lugar
    problems = g1.check_pairing(pd.DataFrame(membership))
    assert any("nao bidirecional" in p for p in problems), problems


def test_unmatched_com_par_reprova():
    membership = _valid_membership()
    membership[2]["matched_variant_id"] = "var:b"
    problems = g1.check_pairing(pd.DataFrame(membership))
    assert any("unmatched_case com matched_variant_id" in p for p in problems), problems


def test_controle_reusado_em_dois_casos_reprova():
    membership = _valid_membership()
    membership.append(_member("var:f", CLINICAL, "case", matched="var:b"))
    problems = g1.check_pairing(pd.DataFrame(membership))
    assert any("nao e 1:1" in p for p in problems), problems


def test_controle_apontando_para_variante_inexistente_reprova():
    """Ponto 3 da revisao: os dois sentidos do par tem de ser validados."""
    membership = _valid_membership()
    membership.append(_member("var:extra", CLINICAL, "control", matched="var:fantasma"))
    problems = g1.check_pairing(pd.DataFrame(membership))
    assert any("ausente do estudo" in p for p in problems), problems


def test_membership_vazio_reprova():
    vazio = pd.DataFrame(columns=list(g1.MEMBERSHIP_COLUMNS))
    assert g1.check_not_empty(vazio) == ["membership vazio"]
    tables = _release_tables(_valid_membership())
    tables["membership"] = vazio
    assert g1.validate(tables) == ["membership vazio"]


def test_estudo_sem_controles_reprova():
    membership = [m for m in _valid_membership() if not (m["study_id"] == POPULATION and m["member_role"] == "control")]
    problems = g1.check_not_empty(pd.DataFrame(membership))
    assert any("sem controles" in p for p in problems), problems


def test_variante_com_dois_papeis_no_mesmo_estudo_reprova():
    membership = _valid_membership()
    membership.append(_member("var:a", CLINICAL, "control", matched="var:a"))
    problems = g1.check_roles_disjoint(pd.DataFrame(membership))
    assert any("mais de um papel" in p for p in problems), problems


def test_caso_sem_controle_declarado_como_case_reprova():
    membership = _valid_membership()
    membership[0]["matched_variant_id"] = None
    problems = g1.check_pairing(pd.DataFrame(membership))
    assert any("case sem matched_variant_id" in p for p in problems), problems


def test_variant_id_repetido_na_mesma_visao_reprova():
    membership = _valid_membership()
    membership.append(_member("var:a", CLINICAL, "case", matched="var:b"))
    problems = g1.check_uniqueness(pd.DataFrame(membership))
    assert any("variant_id repetido" in p for p in problems), problems


def test_stratum_incoerente_reprova():
    membership = _valid_membership()
    membership[0]["stratum"] = "1|missense|common"
    problems = g1.check_strata(pd.DataFrame(membership))
    assert any("stratum publicado" in p for p in problems), problems


def test_par_com_estratos_diferentes_reprova():
    membership = _valid_membership()
    membership[1]["primary_panel"] = "splice"
    membership[1]["stratum"] = g1.stratum_key(1, "splice", "rare")
    problems = g1.check_strata(pd.DataFrame(membership))
    assert any("estratos diferentes" in p for p in problems), problems


def test_rotulo_divergente_do_pb_examples_reprova():
    tables = _release_tables(_valid_membership())
    tables["examples"].loc[0, "binary_label"] = 0
    problems = g1.check_against_release(
        tables["membership"], tables["examples"], tables["panels"], tables["partitions"]
    )
    assert any("binary_label diverge de pb_examples" in p for p in problems), problems


def test_cluster_divergente_do_pb_partitions_reprova():
    tables = _release_tables(_valid_membership())
    tables["partitions"].loc[0, "overlap_cluster_id"] = "outro"
    problems = g1.check_against_release(
        tables["membership"], tables["examples"], tables["panels"], tables["partitions"]
    )
    assert any("overlap_cluster_id diverge de pb_partitions" in p for p in problems), problems


def test_membro_fora_do_pb_examples_reprova():
    tables = _release_tables(_valid_membership())
    tables["examples"] = tables["examples"].iloc[1:].copy()
    problems = g1.check_against_release(
        tables["membership"], tables["examples"], tables["panels"], tables["partitions"]
    )
    assert any("ausentes do pb_examples" in p for p in problems), problems


def test_relatorio_conta_os_dois_estudos_separados_e_nunca_soma():
    tables = _release_tables(_valid_membership())
    joined = g1.join_coordinates(tables["membership"], tables["examples"])
    report = g1.build_report(tables["membership"], joined, {"release_root": "x"})
    assert set(report["por_estudo"]) == {CLINICAL, POPULATION}, report["por_estudo"]
    assert report["por_estudo"][CLINICAL]["por_papel"] == {"case": 1, "control": 1, "unmatched_case": 1}
    assert report["por_estudo"][POPULATION]["pares"] == 1
    assert report["por_estudo"][CLINICAL]["casos_sem_par"] == 1
    assert "total" not in report and "geral" not in report, "os dois estudos nao podem ser somados"


# --------------------------------------------------------------------------------------------- ponta a ponta


def _run(membership: list[dict]) -> tuple[int, dict | None, pd.DataFrame | None]:
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "release", Path(tmp) / "out"
        _write_release(root, _release_tables(membership))
        rc = g1.main(["--release-root", str(root), "--out-dir", str(out)])
        report_path = out / "g1_brazil_studies_report.json"
        variants_path = out / "brazil_study_variants.parquet"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        variants = pd.read_parquet(variants_path) if variants_path.exists() else None
        return rc, report, variants


def test_end_to_end_publica_coordenadas_e_relatorio():
    rc, report, variants = _run(_valid_membership())
    assert rc == 0, report
    assert list(variants["chrom"].unique()) == ["chr1"] and variants["pos_1based"].notna().all()
    assert set(g1.MEMBERSHIP_COLUMNS) <= set(variants.columns)
    assert report["membership"]["linhas"] == 5
    assert len(report["membership"]["hash_composicao"]) == 64
    assert len(report["membership"]["hash_conteudo"]) == 64
    assert report["coordenadas"]["sem_coordenada"] == 0
    assert len(report["entradas"]["membership_sha256"]) == 64
    assert len(report["saidas"]["variantes_sha256"]) == 64


def test_hash_de_conteudo_muda_quando_o_rotulo_muda_e_o_de_composicao_nao():
    """Ponto 4 da revisao: hash de composicao identifica o conjunto, nao o conteudo."""
    tables = _release_tables(_valid_membership())
    joined = g1.join_coordinates(tables["membership"], tables["examples"])
    trocado = joined.copy()
    trocado.loc[trocado.index[0], "binary_label"] = 1 - int(joined.iloc[0]["binary_label"])
    composicao = ("variant_id", "study_id", "member_role")
    assert g1.logical_hash(joined, composicao) == g1.logical_hash(trocado, composicao)
    assert g1.logical_hash(joined, g1.CONTENT_COLUMNS) != g1.logical_hash(trocado, g1.CONTENT_COLUMNS)


def test_end_to_end_para_com_codigo_2_e_nao_publica_nada():
    membership = _valid_membership()
    membership[1]["matched_variant_id"] = "var:zzz"
    rc, report, variants = _run(membership)
    assert rc == 2 and report is None and variants is None


def test_release_real_passa_nas_checagens():
    root = Path(os.environ.get("MOSAIC_RELEASE", str(Path.home() / "mosaic-v1")))
    if not (root / "studies/brazil/membership.parquet").exists():
        raise Skip(f"release do Mosaic nao encontrado em {root} (use MOSAIC_RELEASE)")
    tables = g1.load_release(root)
    problems = g1.validate(tables)
    assert problems == [], problems
    membership = tables["membership"]
    assert set(membership["study_id"]) == {CLINICAL, POPULATION}, sorted(set(membership["study_id"]))
    print(f"        release real: {len(membership)} linhas, "
          f"{membership['variant_id'].nunique()} variantes distintas")


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
