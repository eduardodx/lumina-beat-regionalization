"""Prova que o G2 monta os recortes do core_locus pela agenda oficial e tira os estudos brasileiros dos TRES.

A regra que estes testes protegem: membro de estudo e cluster de membro nao podem sobrar no treino, na validacao
nem no teste reservado -- o `br_population_observed` e gold e cai justamente na validacao e no teste gold.
    PYTHONPATH=. python3 tests/test_build_core_locus_head_snapshot.py
Com REQUIRE_NO_SKIP=1, teste pulado conta como falha (use no notebook antes de rodar o G2).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.build_core_locus_head_snapshot as g2  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _row(vid, fold, label, panel="missense", tier="gold", cluster=None, chrom="chr1", eligible=True, br=False):
    return {
        "variant_id": vid, "chrom": chrom, "pos_1based": 1000 + abs(hash(vid)) % 1000, "ref": "A", "alt": "G",
        "binary_label": label, "label_tier": tier, "sequence_eligible": eligible, "br_lab_any": br,
        "primary_panel": panel, "overlap_cluster_id": cluster or f"cl_{vid}", "core_fold": fold,
    }


def _frame(extra: list[dict] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for fold in (0, 1):  # teste e validacao: so gold conta, com as duas classes em cada painel
        for panel in g2.DISCRIMINATION_PANELS:
            for label in (0, 1):
                rows.append(_row(f"var:g{fold}{panel}{label}", fold, label, panel=panel, tier="gold"))
        rows.append(_row(f"var:c{fold}", fold, 1, tier="consensus"))  # consensus fora de validacao/teste
    for fold in (2, 3, 4):  # treino: gold + consensus
        rows.append(_row(f"var:t{fold}p", fold, 1, tier="consensus"))
        rows.append(_row(f"var:t{fold}b", fold, 0, tier="gold"))
    rows += extra or []
    return pd.DataFrame(rows)


def _splits(frame=None, run_id=0):
    return g2.split_core(frame if frame is not None else _frame(), run_id=run_id)


# ------------------------------------------------------------------------------------------------ testes puros


def test_fold_roles_segue_a_agenda_oficial():
    assert g2.fold_roles(0) == {"test": 0, "validation": 1, "train": (2, 3, 4)}
    assert g2.fold_roles(4) == {"test": 4, "validation": 0, "train": (1, 2, 3)}
    try:
        g2.fold_roles(5)
    except ValueError:
        return
    raise AssertionError("run_id fora do intervalo deveria falhar")


def test_consensus_so_entra_no_treino():
    splits = _splits()
    tiers = {role: set(rows["label_tier"]) for role, rows in splits.items()}
    assert tiers["validation"] == {"gold"} and tiers["test"] == {"gold"}, tiers
    assert tiers["train"] == {"gold", "consensus"}, tiers


def test_sequence_eligible_filtra_os_tres_recortes():
    extra = [_row("var:inelegivel", 1, 1, tier="gold", eligible=False),
             _row("var:inelegivel2", 2, 1, tier="consensus", eligible=False)]
    splits = _splits(_frame(extra))
    todos = set().union(*(set(rows["variant_id"]) for rows in splits.values()))
    assert "var:inelegivel" not in todos and "var:inelegivel2" not in todos, todos


def test_exclusao_de_membros_atinge_treino_validacao_e_teste():
    extra = [_row("var:estudo_t", 0, 1, tier="gold"),      # teste reservado
             _row("var:estudo_v", 1, 0, tier="gold"),      # validacao (caso do br_population_observed)
             _row("var:estudo_tr", 2, 1, tier="consensus")]  # treino
    membros = {"var:estudo_t", "var:estudo_v", "var:estudo_tr"}
    splits, steps = g2.apply_exclusions(
        _splits(_frame(extra)), study_variants=membros, study_clusters=set(), broad_br=None, reserve_chr8=False
    )
    sobrou = set().union(*(set(rows["variant_id"]) for rows in splits.values())) & membros
    assert not sobrou, sobrou
    removidos = steps[0]["removidos"]
    assert removidos["test"]["n"] == 1 and removidos["validation"]["n"] == 1 and removidos["train"]["n"] == 1
    assert g2.check_no_study_leakage(splits, membros, set()) == []


def test_exclusao_por_cluster_remove_vizinho_que_nao_e_membro():
    extra = [_row("var:membro", 2, 1, tier="consensus", cluster="cl_compartilhado"),
             _row("var:vizinho", 1, 1, tier="gold", cluster="cl_compartilhado")]
    splits, steps = g2.apply_exclusions(
        _splits(_frame(extra)), study_variants={"var:membro"}, study_clusters={"cl_compartilhado"},
        broad_br=None, reserve_chr8=False,
    )
    todos = set().union(*(set(rows["variant_id"]) for rows in splits.values()))
    assert "var:vizinho" not in todos, "vizinho de cluster de membro tem de sair"
    assert steps[0]["removidos"]["train"]["n"] == 1, steps[0]
    assert steps[1]["removidos"]["validation"]["n"] == 1, steps[1]


def test_custo_e_medido_por_exclusao_sem_contar_duas_vezes():
    extra = [_row("var:membro", 2, 1, tier="consensus", cluster="cl_x"),
             _row("var:vizinho", 2, 0, tier="gold", cluster="cl_x")]
    _, steps = g2.apply_exclusions(
        _splits(_frame(extra)), study_variants={"var:membro"}, study_clusters={"cl_x"},
        broad_br=None, reserve_chr8=False,
    )
    assert steps[0]["removidos"]["train"] == {"n": 1, "P": 1, "B": 0}, steps[0]
    assert steps[1]["removidos"]["train"] == {"n": 1, "P": 0, "B": 1}, steps[1]


def test_regra_ampla_sem_lista_fica_registrada_como_pendencia():
    _, steps = g2.apply_exclusions(_splits(), study_variants=set(), study_clusters=set(),
                                   broad_br=None, reserve_chr8=False)
    ampla = [s for s in steps if s["exclusao"] == g2.EXCLUSION_BROAD_BR][0]
    assert ampla["status"] == "nao_aplicada", ampla


def test_chr8_reservado_por_padrao_e_desligavel():
    extra = [_row("var:chr8", 2, 1, tier="consensus", chrom="chr8")]
    reservado, _ = g2.apply_exclusions(_splits(_frame(extra)), study_variants=set(), study_clusters=set(),
                                       broad_br=None, reserve_chr8=True)
    mantido, _ = g2.apply_exclusions(_splits(_frame(extra)), study_variants=set(), study_clusters=set(),
                                     broad_br=None, reserve_chr8=False)
    assert "var:chr8" not in set(reservado["train"]["variant_id"])
    assert "var:chr8" in set(mantido["train"]["variant_id"])


def test_validacao_sem_as_duas_classes_reprova():
    frame = _frame()
    frame = frame[~((frame["core_fold"] == 1) & (frame["primary_panel"] == "splice")
                    & (frame["binary_label"] == 0))]
    problems = g2.check_validation_supports_selection(_splits(frame))
    assert any("validacao/splice" in p for p in problems), problems
    assert any("nunca trocando de fold" in p for p in problems), problems


def test_clusters_compartilhados_entre_papeis_reprovam():
    extra = [_row("var:a", 1, 1, tier="gold", cluster="cl_dupla"),
             _row("var:b", 2, 1, tier="consensus", cluster="cl_dupla")]
    problems = g2.check_clusters_disjoint(_splits(_frame(extra)))
    assert any("train e validation" in p for p in problems), problems


def test_hash_do_snapshot_depende_do_papel():
    splits = _splits()
    snapshot = g2.snapshot_frame(splits)
    trocado = snapshot.copy()
    trocado.loc[trocado.index[0], "role"] = "validation" if snapshot.iloc[0]["role"] != "validation" else "test"
    assert g2.snapshot_hash(snapshot) != g2.snapshot_hash(trocado)
    assert len(g2.snapshot_hash(snapshot)) == 64


def test_hash_de_composicao_nao_ve_troca_de_rotulo_mas_o_de_conteudo_ve():
    """Ponto 4 da revisao: os dois hashes existem porque medem coisas diferentes."""
    snapshot = g2.snapshot_frame(_splits())
    trocado = snapshot.copy()
    trocado.loc[trocado.index[0], "binary_label"] = 1 - int(snapshot.iloc[0]["binary_label"])
    assert g2.snapshot_hash(snapshot) == g2.snapshot_hash(trocado), "composicao nao muda: e o esperado"
    assert g2.snapshot_content_hash(snapshot) != g2.snapshot_content_hash(trocado), "conteudo tem de mudar"


# --------------------------------------------------------------------------------------------- ponta a ponta


def _write_release(root: Path, frame: pd.DataFrame, membros: list[str]) -> None:
    (root / "studies" / "brazil").mkdir(parents=True, exist_ok=True)
    frame[list(g2.FRAME_COLUMNS)].copy().to_parquet(root / "pb_examples.parquet", index=False)
    frame[["variant_id", "primary_panel"]].copy().to_parquet(root / "pb_panels.parquet", index=False)
    frame[["variant_id", "overlap_cluster_id", "core_fold"]].copy().to_parquet(
        root / "pb_partitions.parquet", index=False)
    membership = frame[frame["variant_id"].isin(membros)][["variant_id", "overlap_cluster_id"]].copy()
    membership.to_parquet(root / "studies/brazil/membership.parquet", index=False)


def _write_broad(tmp: Path, root: Path, ids: list[str], *, manifesto: bool = True, sha: str | None = None,
                 trocar_conteudo: list[str] | None = None) -> Path:
    """Lista da regra ampla com o manifesto que o G2 exige (sha256 do pb_examples e da propria lista)."""
    list_path = tmp / "broad.txt"
    list_path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    if manifesto:
        g2.broad_manifest_path(list_path).write_text(json.dumps({
            "n_variantes": len(ids),
            "lista_sha256": g2.sha256_file(list_path),
            "pb_examples_sha256": sha if sha is not None else g2.sha256_file(root / "pb_examples.parquet"),
        }), encoding="utf-8")
    if trocar_conteudo is not None:  # mesmo tamanho, ainda superconjunto, conteudo diferente
        list_path.write_text("\n".join(sorted(trocar_conteudo)) + "\n", encoding="utf-8")
    return list_path


def _run(frame: pd.DataFrame, membros: list[str], extra_args: list[str] | None = None, broad=None):
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "release", Path(tmp) / "out"
        _write_release(root, frame, membros)
        args = ["--release-root", str(root), "--out-dir", str(out), *(extra_args or [])]
        if broad is not None:
            args += ["--broad-br-variant-ids", str(broad(Path(tmp), root))]
        rc = g2.main(args)
        report_path = out / "g2_core_snapshot_report.json"
        snapshot_path = out / "core_head_snapshot.parquet"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        snapshot = pd.read_parquet(snapshot_path) if snapshot_path.exists() else None
        return rc, report, snapshot


def test_end_to_end_publica_snapshot_sem_membros_dos_estudos():
    extra = [_row("var:estudo_v", 1, 1, tier="gold", cluster="cl_estudo"),
             _row("var:vizinho", 2, 1, tier="consensus", cluster="cl_estudo")]
    rc, report, snapshot = _run(_frame(extra), ["var:estudo_v"])
    assert rc == 0, report
    assert set(snapshot["role"]) == {"train", "validation", "test"}, set(snapshot["role"])
    assert "var:estudo_v" not in set(snapshot["variant_id"])
    assert "var:vizinho" not in set(snapshot["variant_id"]), "cluster do membro tem de sair junto"
    assert report["pronto_para_congelar"] is False and report["pendencias"], report["pendencias"]
    assert report["antes_das_exclusoes"]["validation"]["n"] > report["depois_das_exclusoes"]["validation"]["n"]
    assert report["papel_de_cada_recorte"]["test"].startswith("avalia so depois"), report["papel_de_cada_recorte"]


def test_end_to_end_com_lista_ampla_validada_fica_pronto_para_congelar():
    frame = _frame([_row("var:brasileira", 2, 1, tier="consensus", br=True)])
    rc, report, snapshot = _run(
        frame, [], broad=lambda tmp, root: _write_broad(tmp, root, ["var:t3p", "var:brasileira"])
    )
    assert rc == 0, report
    assert report["pronto_para_congelar"] is True and report["pendencias"] == []
    assert "var:t3p" not in set(snapshot["variant_id"])
    ampla = [s for s in report["exclusoes"] if s["exclusao"] == g2.EXCLUSION_BROAD_BR][0]
    assert ampla["removidos"]["train"]["n"] == 2, ampla
    assert report["entradas"]["lista_regra_ampla"]["validada"] is True


def test_lista_ampla_que_nao_cobre_o_br_lab_any_do_release_reprova():
    """Ponto 1 da revisao: presenca de arquivo nao pode liberar o congelamento."""
    frame = _frame([_row("var:brasileira", 2, 1, tier="consensus", br=True)])
    rc, report, snapshot = _run(frame, [], broad=lambda tmp, root: _write_broad(tmp, root, ["var:t3p"]))
    assert rc == 2 and report is None and snapshot is None


def test_lista_ampla_vazia_reprova():
    rc, report, _ = _run(_frame(), [], broad=lambda tmp, root: _write_broad(tmp, root, []))
    assert rc == 2 and report is None


def test_lista_ampla_com_conteudo_trocado_reprova_mesmo_mantendo_tamanho():
    """Tamanho e superconjunto nao amarram o conteudo: e o sha256 da lista que fecha."""
    frame = _frame([_row("var:brasileira", 2, 1, tier="consensus", br=True),
                    _row("var:outra", 3, 0, tier="consensus")])
    rc, report, snapshot = _run(frame, [], broad=lambda tmp, root: _write_broad(
        tmp, root, ["var:t3p", "var:brasileira"], trocar_conteudo=["var:outra", "var:brasileira"]))
    assert rc == 2 and snapshot is None, report


def test_lista_ampla_sem_manifesto_ou_de_outro_release_reprova():
    sem = _run(_frame(), [], broad=lambda tmp, root: _write_broad(tmp, root, ["var:t3p"], manifesto=False))
    assert sem[0] == 2, sem[1]
    outro = _run(_frame(), [], broad=lambda tmp, root: _write_broad(tmp, root, ["var:t3p"], sha="0" * 64))
    assert outro[0] == 2, outro[1]


def test_brazil_variants_incompleto_reprova_em_vez_de_encolher_a_exclusao():
    """Ponto 2 da revisao: a saida do G1 e conferencia, nunca substituicao do membership."""
    extra = [_row("var:estudo_a", 2, 1, tier="consensus"), _row("var:estudo_b", 1, 1, tier="gold")]
    frame = _frame(extra)
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "release", Path(tmp) / "out"
        _write_release(root, frame, ["var:estudo_a", "var:estudo_b"])
        incompleto = Path(tmp) / "g1_incompleto.parquet"
        frame[frame["variant_id"] == "var:estudo_a"][["variant_id", "overlap_cluster_id"]].copy().to_parquet(
            incompleto, index=False)
        rc = g2.main(["--release-root", str(root), "--brazil-variants", str(incompleto), "--out-dir", str(out)])
        assert rc == 2, "arquivo incompleto do G1 nao pode virar a fonte das exclusoes"
        assert not (out / "core_head_snapshot.parquet").exists()


def test_end_to_end_para_com_codigo_2_mas_publica_o_diagnostico():
    """Reprovar sem relatorio esconde justamente os numeros que dizem o que corrigir."""
    frame = _frame()
    frame = frame[~((frame["core_fold"] == 1) & (frame["primary_panel"] == "noncoding")
                    & (frame["binary_label"] == 1))]
    rc, report, snapshot = _run(frame, [])
    assert rc == 2 and snapshot is None, "o snapshot nao pode ser publicado"
    assert report is not None and report["status"] == "FALHOU", report
    assert any("validacao/noncoding" in p for p in report["checagens"]), report["checagens"]
    assert report["pronto_para_congelar"] is False
    assert "exclusoes" in report and "antes_das_exclusoes" in report


def test_politica_de_cluster_treino_preserva_a_validacao_e_registra_o_custo_potencial():
    """O que o release real cobrou: com clusters grandes, excluir vizinhos nos tres recortes zera a validacao."""
    extra = [_row("var:membro", 2, 1, tier="consensus", cluster="cl_grande"),
             _row("var:vizinho_val", 1, 0, panel="splice", tier="gold", cluster="cl_grande")]
    frame = _frame(extra)
    membros = {"var:membro"}
    clusters = {"cl_grande"}

    todos, steps_todos = g2.apply_exclusions(_splits(frame), study_variants=membros, study_clusters=clusters,
                                             broad_br=None, reserve_chr8=False, cluster_policy=g2.CLUSTER_ALL)
    treino, steps_treino = g2.apply_exclusions(_splits(frame), study_variants=membros, study_clusters=clusters,
                                               broad_br=None, reserve_chr8=False,
                                               cluster_policy=g2.CLUSTER_TRAIN_ONLY)
    assert "var:vizinho_val" not in set(todos["validation"]["variant_id"])
    assert "var:vizinho_val" in set(treino["validation"]["variant_id"])
    assert g2.check_no_study_leakage(treino, membros, clusters,
                                     cluster_policy=g2.CLUSTER_TRAIN_ONLY) == []
    # O custo da politica estrita fica registrado mesmo quando ela nao e aplicada.
    passo = [s for s in steps_treino if s["exclusao"] == g2.EXCLUSION_STUDY_CLUSTERS][0]
    assert passo["politica"] == g2.CLUSTER_TRAIN_ONLY
    assert passo["custo_potencial_se_todos"]["validation"]["n"] == 1, passo["custo_potencial_se_todos"]
    assert passo["custo_potencial_se_todos"]["validation"]["clusters_atingidos"] == 1
    assert [s for s in steps_todos if s["exclusao"] == g2.EXCLUSION_STUDY_CLUSTERS][0]["removidos"]["validation"][
        "n"] == 1


def test_release_real_bate_com_os_numeros_do_guia():
    root = Path(os.environ.get("MOSAIC_RELEASE", str(Path.home() / "mosaic-v1")))
    if not (root / "pb_examples.parquet").exists():
        raise Skip(f"release do Mosaic nao encontrado em {root} (use MOSAIC_RELEASE)")
    splits = g2.split_core(g2.load_frame(root), run_id=0)
    got = {role: g2.role_counts(rows) for role, rows in splits.items()}
    esperado = {"train": 196096, "validation": 2453, "test": 1758}  # guia de splits, antes das exclusoes
    for role, reference in esperado.items():
        delta = abs(got[role]["n"] - reference) / reference
        assert delta < 0.01, f"{role}: {got[role]['n']} vs {reference} do guia (delta {delta:.3%})"
    print(f"        release real, run 0 antes das exclusoes: {got}")


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
