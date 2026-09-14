"""Prova que o diagnostico de submissor brasileiro reproduz a agregacao do Mosaic e separa cada motivo, sem tratar
ausencia de marcacao, pais nao resolvido ou correspondencia parcial como evidencia.

Os testes puros so precisam de stdlib. Os de ponta a ponta usam o Mosaic real (pandas + pyarrow + pyyaml e um
clone do lumina-mosaic no commit do release, via MOSAIC_ROOT ou ao lado deste repositorio):
    PYTHONPATH=. python tests/test_diagnose_brazilian_submitter_divergence.py
Com REQUIRE_NO_SKIP=1, teste pulado conta como falha (use no notebook antes de rodar o diagnostico).
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.diagnose_brazilian_submitter_divergence as diag  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


ORGS = {"Dasa": "508087", "Mendelics": "500035"}
INCLUDE = {"508087", "500035"}


def _scv(scv: str, submitter: str, cls: str = "Pathogenic", origin: str = "germline:1", vid: str = "v"):
    return NS(scv=scv, submitter=submitter, classification=cls, origin_counts=origin, contributes_raw=None,
              variation_id=vid, review_status="criteria provided, single submitter", date_last_evaluated="-")


def _whitelist(scv) -> bool:
    """Imita o filtro P/B do Mosaic (classificacao P/B e origem germinativa)."""
    return scv.classification in {"Pathogenic", "Benign"} and "germline" in scv.origin_counts


def _canon(name: str) -> str:
    """Mesma ideia do mosaic.orgs.canonical_keys (';' vira ',', casefold), sem depender do Mosaic."""
    text = " ".join(name.split()).replace(";", ",")
    return ", ".join(p.strip() for p in text.split(",") if p.strip()).casefold()


def test_reproduction_follows_mosaic_aggregation():
    scvs = {
        "1": [_scv("S1", "Dasa")],
        "2": [_scv("S2", "Dasa")], "3": [_scv("S3", "GeneDx", "Benign")],
        "4": [_scv("S4", "Dasa", "Uncertain significance"), _scv("S5", "GeneDx")],
        "5": [_scv("S6", "Mendelics", origin="somatic:1"), _scv("S7", "GeneDx")],
    }
    vids = {"so_br": ["1"], "compartilhada": ["2", "3"], "vus_br_nao_conta": ["4"], "somatica_nao_conta": ["5"],
            "sem_scv": ["9"]}
    got = diag.reproduce_br_flags(vids, scvs, is_whitelisted=_whitelist, org_of=ORGS.get, include_ids=INCLUDE)
    assert got["so_br"] == (True, True, False), got
    assert got["compartilhada"] == (True, False, True), got
    assert got["vus_br_nao_conta"] == (False, False, False), got      # VUS brasileira fica fora do filtro P/B
    assert got["somatica_nao_conta"] == (False, False, False), got    # origem somatica fica fora
    assert got["sem_scv"] == (False, False, False), got


def test_validation_reports_every_mismatch():
    reproduced = {"a": (True, True, False), "b": (False, False, False)}
    agree, diverge = diag.compare_with_published(reproduced, [("a", True, True, False), ("b", True, False, True),
                                                              ("c", False, False, False)])
    assert agree == 1, agree
    assert [d["variant_id"] for d in diverge] == ["b", "c"], diverge   # 'c' nem foi reproduzido: tambem diverge


def test_classification_describes_marks_not_nationality():
    c = diag.classify
    assert c(True, True, False, False, False, False, False) == diag.OUTSIDE_MASTER
    assert c(True, True, False, True, False, False, False) == diag.A1        # v1 sem linha nenhuma
    assert c(True, True, False, True, False, True, True) == diag.A2          # v1 coberta, sem linha brasileira
    assert c(False, False, False, True, True, False, True) == diag.B
    assert c(True, False, True, True, True, False, True) == diag.C1          # v1 sem linha nao BR, Mosaic compart.
    assert c(True, True, False, True, True, True, True) == diag.C2           # v1 com linha nao BR, Mosaic so marcadas
    assert c(True, True, False, True, True, False, True) == diag.BOTH_MARK
    assert c(True, False, True, True, True, True, True) == diag.BOTH_MARK
    # nenhum dos dois marca: com e SEM dado na v1 -- nenhum dos casos e "nao brasileira confirmada"
    assert c(False, False, False, True, False, True, True) == diag.NONE_MARK_COVERED
    assert c(False, False, False, True, False, False, False) == diag.NONE_MARK_UNCOVERED
    assert all("nao_br" not in name.split("_marca")[0] for name in (diag.NONE_MARK_COVERED, diag.NONE_MARK_UNCOVERED))


def test_set_priority_keeps_paired_members_apart_from_unpaired_slice():
    sets = {"t_br_slice": {"k1", "k2"}, "t_br_pareado": {"k1"}, "split_train": {"k3"}}
    assert [diag.assign_set(k, sets) for k in ("k1", "k2", "k3", "k4")] == \
        ["t_br_pareado", "t_br_slice", "split_train", "nenhum"]


def test_submitter_presence_separates_exact_hint_and_indeterminate():
    p = diag.submitter_presence
    joined = "Dasa;Laboratorio de Genetica e Diagnostico Molecular; Hospital Israelita Albert Einstein;Mendelics"
    assert p("lab", "Laboratorio ABC") == diag.PARTIAL                       # o falso positivo apontado na revisao
    assert p("Dasa", "Dasa Laboratories") == diag.PARTIAL
    assert p("Laboratorio de Genetica e Diagnostico Molecular; Hospital Israelita Albert Einstein", joined) == diag.EXACT
    assert p("Dasa", joined) == diag.EXACT and p("Mendelics", joined) == diag.EXACT
    assert p("GeneDx", joined) == diag.ABSENT
    assert p("Alpha;Beta", "Alpha;Beta;Gamma") == diag.SEMICOLON_AMBIGUOUS   # podem ser dois nomes vizinhos
    assert p("Diagnósticos da América", "Diagnosticos da America;Mendelics") == diag.EXACT
    assert p("GeneDx", "Dasa;Mendelics;...(+3)") == diag.TRUNCATED           # truncada: ausencia nao se afirma
    assert p("Mendelics", "Dasa;Mendelics;...(+3)") == diag.EXACT
    for missing in (float("nan"), None, "", "   "):
        assert p("Dasa", missing) == diag.NO_LIST, missing


def test_country_lookup_keeps_unresolved_apart_from_non_brazilian():
    header = "#organization\torganization ID\tinstitution type\tstreet address\tcity\tcountry\n"
    body = ("Dasa\t508087\tlab\t\tSao Paulo\tBrazil\n"
            "Laboratorio Novo\t999\tlab\t\tRecife\tBrazil\n"
            "GeneDx\t26957\tlab\t\tGaithersburg\tUnited States\n"
            "Duplicado\t1\tlab\t\tX\tBrazil\n"
            "Duplicado\t2\tlab\t\tY\tPortugal\n"
            "Sem Pais\t3\tlab\t\tZ\t\n"
            "linha\tquebrada\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "org.txt"
        path.write_text(header + body, encoding="utf-8")
        countries, meta = diag.load_organization_countries(path, _canon)
    assert meta == {"organizacoes": 6, "linhas_ignoradas": 1}, meta
    st = diag.institution_status
    assert st(ORGS.get("Dasa"), INCLUDE, countries.get(_canon("Dasa"))) == diag.LISTED
    assert st(None, INCLUDE, countries.get(_canon("Laboratorio Novo"))) == diag.REGISTRY_BRAZIL_UNLISTED
    assert st(None, INCLUDE, countries.get(_canon("GeneDx"))) == diag.REGISTRY_NOT_BRAZIL
    assert st(None, INCLUDE, countries.get(_canon("Duplicado"))) == diag.COUNTRY_AMBIGUOUS
    assert st(None, INCLUDE, countries.get(_canon("Sem Pais"))) == diag.COUNTRY_UNRESOLVED     # pais vazio
    assert st(None, INCLUDE, countries.get(_canon("Nunca Visto"))) == diag.COUNTRY_UNRESOLVED  # sem registro


def _rows(*specs):
    """(is_br, org_id, passa_filtro, status, presenca). Instituicao da lista => is_br True, como no Mosaic."""
    return [{"is_br_mosaic": b, "org_id_mosaic": o, "passa_filtro_pb": f, "status_instituicao": s,
             "submissor_na_tabela_v1": t} for b, o, f, s, t in specs]


def test_evidence_does_not_promote_hints_or_unknown_countries():
    ev = diag.variant_evidence
    L, BR_OUT, NOT_BR, UNRES = diag.LISTED, diag.REGISTRY_BRAZIL_UNLISTED, diag.REGISTRY_NOT_BRAZIL, diag.COUNTRY_UNRESOLVED
    # A2: so presenca exata conta como presenca
    assert ev(diag.A2, _rows((True, "1", True, L, diag.EXACT))) == "submissor_marcado:presente_exato"
    assert ev(diag.A2, _rows((True, "1", True, L, diag.PARTIAL))) == "submissor_marcado:so_indicio_parcial_ou_ambiguo"
    assert ev(diag.A2, _rows((True, "1", True, L, diag.SEMICOLON_AMBIGUOUS))) == \
        "submissor_marcado:so_indicio_parcial_ou_ambiguo"
    assert ev(diag.A2, _rows((True, "1", True, L, diag.TRUNCATED))) == "submissor_marcado:indeterminado"
    assert ev(diag.A2, _rows((True, "1", True, L, diag.NO_LIST))) == "submissor_marcado:indeterminado"
    assert ev(diag.A2, _rows((True, "1", True, L, diag.ABSENT))) == "submissor_marcado:ausente"
    assert ev(diag.A2, _rows((True, "1", True, L, diag.ABSENT), (True, "2", True, L, diag.EXACT))) == \
        "submissor_marcado:presente_exato"
    # B: flags nao exclusivas; pais nao resolvido nao vira "nao brasileira"
    assert ev(diag.B, _rows((True, "1", False, L, diag.EXACT), (False, None, True, NOT_BR, diag.ABSENT))) == \
        "scv_da_lista_fora_do_filtro_pb"
    assert ev(diag.B, _rows((False, None, True, BR_OUT, diag.ABSENT), (True, "1", False, L, diag.EXACT))) == \
        "scv_da_lista_fora_do_filtro_pb+pais_brazil_no_ncbi_fora_da_lista"
    assert ev(diag.B, _rows((False, None, True, UNRES, diag.ABSENT), (False, None, True, NOT_BR, diag.ABSENT))) == \
        "pais_nao_resolvido_em_alguma_scv"
    assert ev(diag.B, _rows((False, None, True, NOT_BR, diag.ABSENT))) == "so_nao_brasileiras_pelo_registro_ncbi"
    # C1: presenca e status do lado nao marcado
    assert ev(diag.C1, _rows((True, "1", True, L, diag.EXACT), (False, None, True, NOT_BR, diag.ABSENT))) == \
        "submissor_nao_marcado:ausente|status:nao_brasileira_pelo_registro_ncbi"
    assert ev(diag.C1, _rows((True, "1", True, L, diag.EXACT), (False, None, True, UNRES, diag.EXACT))) == \
        "submissor_nao_marcado:presente_exato|status:pais_nao_resolvido"
    # C2: a SCV nao marcada esta fora do filtro; com pais desconhecido, o status diz isso (nao "nao_br")
    assert ev(diag.C2, _rows((True, "1", True, L, diag.EXACT), (False, None, False, UNRES, diag.ABSENT))) == \
        "scv_nao_marcada_fora_do_filtro_pb|status:pais_nao_resolvido"
    assert ev(diag.C2, _rows((True, "1", True, L, diag.EXACT))) == "sem_scv_nao_marcada_no_release"
    assert ev(diag.A1, []) == "v1_sem_linhas_da_variante"


def test_scope_scan_sees_unlisted_brazilian_institution_even_when_pipelines_agree():
    countries = {_canon("Laboratorio Novo"): "Brazil", _canon("GeneDx"): "United States"}
    calls: Counter[str] = Counter()

    def org_of(name):
        calls[name] += 1
        return ORGS.get(name)

    scvs = {
        "1": [_scv("S1", "Laboratorio Novo")],                          # nenhum dos pipelines marca: a varredura ve
        "2": [_scv("S2", "Dasa")],
        "3": [_scv("S3", "Laboratorio Novo", "Uncertain significance")],  # fora do filtro: nao conta
        "4": [_scv("S4", "Laboratorio Novo"), _scv("S5", "GeneDx")],
    }
    vids = {"concordam_sem_marca": ["1"], "lista": ["2"], "vus": ["3"], "outra": ["4"]}
    flagged, per_status, unlisted = diag.scan_unlisted_brazil(
        vids, scvs, is_whitelisted=_whitelist, org_of=org_of, include_ids=INCLUDE, canon=_canon, countries=countries)
    assert flagged == {"concordam_sem_marca": True, "lista": False, "vus": False, "outra": True}, flagged
    assert per_status == {diag.REGISTRY_BRAZIL_UNLISTED: 2, diag.LISTED: 1, diag.REGISTRY_NOT_BRAZIL: 1}, per_status
    assert unlisted == {"Laboratorio Novo": 2}, unlisted
    assert calls["Laboratorio Novo"] == 1, calls   # cache por submissor


def _mosaic_root() -> Path:
    candidates = [os.environ.get("MOSAIC_ROOT"), Path(__file__).resolve().parents[2] / "lumina-mosaic",
                  Path.home() / "testeArq" / "lumina-mosaic"]
    for cand in candidates:
        if cand and (Path(cand) / "src" / "mosaic" / "labels.py").is_file():
            return Path(cand)
    raise Skip("sem clone do lumina-mosaic (defina MOSAIC_ROOT)")


def _require_end_to_end() -> Path:
    try:
        import pandas  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        import yaml  # noqa: F401
    except ImportError as exc:
        raise Skip(f"precisa de pandas + pyarrow + pyyaml (falta {exc.name})") from None
    root = _mosaic_root()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    if not head.startswith(diag.EXPECTED_MOSAIC_COMMIT):
        raise Skip(f"clone do Mosaic em {head.strip()[:7]}; o teste usa a configuracao de {diag.EXPECTED_MOSAIC_COMMIT}")
    return root


def _write_inputs(tmp: Path, mosaic, *, publish_wrong: bool) -> list[str]:
    import pandas as pd

    ss_map = mosaic.spec["submission_summary"]
    keys = ["variation_id", "clinical_significance", "review_status", "origin_counts", "submitter", "scv",
            "date_last_evaluated"]
    cols = [ss_map[k] for k in keys]
    contrib = ss_map.get("contributes_to_aggregate")
    if contrib:
        cols.append(contrib)
    scvs = [  # (VariationID, classificacao, submissor, SCV)
        ("101", "Pathogenic", "Dasa", "SCV1"),                                                        # e1
        ("102", "Pathogenic", "Mendelics", "SCV2"), ("102", "Pathogenic", "GeneDx", "SCV3"),          # e2
        ("103", "Pathogenic", "Laboratorio Novo de Genomica", "SCV4"), ("103", "Pathogenic", "GeneDx", "SCV5"),
        ("104", "Uncertain significance", "Dasa", "SCV6"), ("104", "Benign", "GeneDx", "SCV7"),       # e4
        ("105", "Pathogenic", "Dasa", "SCV8"), ("105", "Pathogenic", "GeneDx", "SCV9"),               # e5
        ("106", "Pathogenic", "GeneDx", "SCV10"),                                                     # e6
    ]
    with gzip.open(tmp / "ss.txt.gz", "wt", encoding="utf-8") as fh:
        fh.write("#" + "\t".join(cols) + "\n")
        for vid, cls, sub, acc in scvs:
            vals = {"variation_id": vid, "clinical_significance": cls,
                    "review_status": "criteria provided, single submitter", "origin_counts": "germline:1",
                    "submitter": sub, "scv": acc, "date_last_evaluated": "Jan 01, 2025"}
            fh.write("\t".join([vals[k] for k in keys] + (["-"] if contrib else [])) + "\n")
    ids = ["e1", "e2", "e3", "e4", "e5", "e6"]
    published = {"e1": (True, True, False), "e2": (True, False, True), "e3": (False, False, False),
                 "e4": (False, False, False), "e5": (True, False, True), "e6": (False, False, False)}
    if publish_wrong:
        published["e3"] = (True, True, False)
    pd.DataFrame({
        "variant_id": ids,
        "clinvar_variation_ids": [["101"], ["102"], ["103"], ["104"], ["105"], ["106"]],
        "chrom": ["chr1"] * 6, "pos_1based": [100, 200, 300, 400, 500, 600],
        "ref": ["A", "C", "G", "T", "A", "G"], "alt": ["G", "T", "A", "C", "C", "T"],
        "label_tier": ["consensus"] * 6, "binary_label": [1, 1, 1, 0, 1, 1],
        "br_lab_any": [published[v][0] for v in ids],
        "br_lab_only": [published[v][1] for v in ids],
        "br_lab_shared": [published[v][2] for v in ids],
    }).to_parquet(tmp / "pb.parquet")
    pd.DataFrame({"variant_id": ["e1", "e6"], "study_id": ["br_clinical_evidence"] * 2,
                  "member_role": ["case", "control"]}).to_parquet(tmp / "mem.parquet")
    pd.DataFrame({  # e2 (1:200:C:T) fica fora do master de proposito
        "variant_key": ["1:100:A:G", "1:300:G:A", "1:400:T:C", "1:500:A:C", "1:600:G:T"],
        "has_brazilian_submitter": [False, True, True, True, False],
        "has_non_brazilian_submitter": [True, True, False, False, True],
        "brazilian_submission_rows": [0, 1, 1, 1, 0],
        "non_brazilian_submission_rows": [1, 1, 0, 0, 1],
        "regional_submission_rows": [1, 2, 1, 1, 1],
        "clinvar_regional_cohort": ["non_brazilian", "mixed", "brazilian", "brazilian", "non_brazilian"],
        "regional_submitters": ["Dasa", "Laboratorio Novo de Genomica;GeneDx", "Dasa", "Dasa", "GeneDx"],
    }).to_parquet(tmp / "master.parquet")
    pd.DataFrame({"variant_key": ["1:400:T:C", "1:500:A:C"]}).to_parquet(tmp / "br.parquet")
    pd.DataFrame({"br_variant_key": ["1:400:T:C"], "nonbr_variant_key": ["1:600:G:T"]}).to_parquet(tmp / "pairs.parquet")
    pd.DataFrame({"variant_key": ["1:100:A:G", "1:300:G:A"], "split_within_gene": ["train", "validation"]}
                 ).to_parquet(tmp / "splits.parquet")
    (tmp / "org.txt").write_text(
        "#organization\torganization ID\tinstitution type\tstreet address\tcity\tcountry\n"
        "Laboratorio Novo de Genomica\t999\tlab\t\tRecife\tBrazil\n"
        "GeneDx\t26957\tlab\t\tGaithersburg\tUnited States\n", encoding="utf-8")
    return ["--submission-summary", str(tmp / "ss.txt.gz"), "--organization-summary", str(tmp / "org.txt"),
            "--pb-examples", str(tmp / "pb.parquet"), "--membership", str(tmp / "mem.parquet"),
            "--master-uri", str(tmp / "master.parquet"), "--br", str(tmp / "br.parquet"),
            "--pairs", str(tmp / "pairs.parquet"), "--splits", str(tmp / "splits.parquet"),
            "--out-dir", str(tmp / "out")]


def _run(publish_wrong: bool):
    root = _require_end_to_end()
    mosaic = diag.load_mosaic(root, release=diag.DEFAULT_RELEASE, expected_commit=diag.EXPECTED_MOSAIC_COMMIT)
    original = diag.verify_source
    diag.verify_source = lambda m, p: {"chave": "teste: arquivo sintetico, sem sha256"}
    try:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            argv = ["--mosaic-root", str(root)] + _write_inputs(tmp, mosaic, publish_wrong=publish_wrong)
            rc = diag.main(argv)
            report = json.loads((tmp / "out" / "diagnostico_submissor_br.json").read_text(encoding="utf-8"))
            evidence = None
            if (tmp / "out" / "evidencia_scv.parquet").is_file():
                import pandas as pd
                evidence = pd.read_parquet(tmp / "out" / "evidencia_scv.parquet")
    finally:
        diag.verify_source = original
    return rc, report, evidence


def test_end_to_end_with_real_mosaic_functions():
    rc, rep, evidence = _run(publish_wrong=False)
    assert rc == 0, rep
    assert rep["reproducao"]["concordam"] == 6 and rep["reproducao"]["divergem"] == 0, rep["reproducao"]
    assert rep["reproducao"]["status"].startswith("fiel"), rep["reproducao"]
    assert rep["categorias"] == {diag.A2: 1, diag.B: 2, diag.C1: 1, diag.NONE_MARK_COVERED: 1,
                                 diag.OUTSIDE_MASTER: 1}, rep["categorias"]
    ev = rep["evidencia_por_categoria"]
    assert ev[diag.A2] == {"submissor_marcado:presente_exato": 1}, ev
    assert ev[diag.B] == {"scv_da_lista_fora_do_filtro_pb": 1, "pais_brazil_no_ncbi_fora_da_lista": 1}, ev
    assert ev[diag.C1] == {"submissor_nao_marcado:ausente|status:nao_brasileira_pelo_registro_ncbi": 1}, ev
    assert rep["categorias_por_conjunto_v1"]["t_br_pareado"] == {diag.B: 1}, rep["categorias_por_conjunto_v1"]
    assert rep["membros_mosaic_por_conjunto_v1"] == {"br_clinical_evidence/case": {"split_train": 1},
                                                     "br_clinical_evidence/control": {"t_nonbr_pareado": 1}}, rep
    scope = rep["escopo_lista_br"]
    assert scope["scvs_do_filtro_por_status"] == {diag.LISTED: 3, diag.REGISTRY_BRAZIL_UNLISTED: 1,
                                                  diag.REGISTRY_NOT_BRAZIL: 5}, scope
    assert scope["submissores_pais_brazil_fora_da_lista"] == [["Laboratorio Novo de Genomica", 1]], scope
    assert scope["variantes_por_categoria"][diag.B] == {"nao": 1, "sim": 1}, scope
    assert set(evidence["categoria"]) == {diag.A2, diag.B, diag.C1}, evidence
    novo = evidence[evidence["submitter"] == "Laboratorio Novo de Genomica"]
    assert list(novo["status_instituicao"]) == [diag.REGISTRY_BRAZIL_UNLISTED], novo


def test_end_to_end_stops_when_reproduction_is_not_faithful():
    rc, rep, evidence = _run(publish_wrong=True)
    assert rc == 2, rep
    assert rep["reproducao"]["divergem"] == 1 and rep["reproducao"]["status"].startswith("NAO_FIEL"), rep["reproducao"]
    assert "categorias" not in rep and evidence is None, "sem reproducao fiel nao pode haver classificacao"


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
