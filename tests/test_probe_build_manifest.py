"""Testes do censo/selecao/validacao da Fase 0 da sonda de embedding.

Stdlib puro (o script mantem pandas so na borda de carga/escrita), entao roda tambem no Windows:
``python tests/test_probe_build_manifest.py``.
"""

from __future__ import annotations

import random
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_build_manifest import (  # noqa: E402
    ALL_WINDOWS,
    Variant,
    census,
    choose_curated_sites,
    group_by_site,
    pb_sites,
    select,
    site_priority,
    validate_windows,
    window_plan,
)


@contextmanager
def assert_raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


_RNG = random.Random(20260903)
_GENOME = {f"chr{i}": "".join(_RNG.choice("ACGT") for _ in range(120_000)) for i in (1, 2)}


def _fetch(chrom: str, start: int, end: int) -> str:
    if start < 0 or chrom not in _GENOME:
        return ""
    return _GENOME[chrom][start:end]


def _variant(pos: int, alt: str, *, chrom="chr1", label=1, tier="gold", panel="missense") -> Variant:
    ref = _GENOME[chrom][pos - 1]
    return Variant(
        variant_id=f"var:{chrom}:{pos}:{ref}:{alt}", chrom=chrom, pos_1based=pos, ref=ref, alt=alt,
        binary_label=label, label_tier=tier, primary_panel=panel,
        panel_role="discrimination" if panel in ("missense", "splice", "noncoding") else "descriptive",
        region_class=panel, phylop_241way=1.0, mane_gene="GENE", resolved_gene="GENE",
        gnomad_af_bin="rare", overlap_cluster_id="ovl:x", br_lab_any=False,
    )


def _alts_for(pos: int, chrom="chr1") -> list[str]:
    ref = _GENOME[chrom][pos - 1]
    return [b for b in "ACGT" if b != ref]


# --------------------------------------------------------------------------------------------
# Censo
# --------------------------------------------------------------------------------------------


def test_group_by_site_unions_alts_of_same_site():
    """Mesma unidade que mosaic.clusters funde primeiro: (chrom, pos, ref)."""
    a, b = _alts_for(50_000)[:2]
    records = [_variant(50_000, a), _variant(50_000, b), _variant(60_000, _alts_for(60_000)[0])]
    by_site = group_by_site(records)
    assert len(by_site) == 2
    assert len(by_site[("chr1", 50_000, _GENOME["chr1"][49_999])]) == 2


def test_census_counts_multiallelic_and_both_classes():
    a, b, c = _alts_for(50_000)
    records = [
        _variant(50_000, a, label=1), _variant(50_000, b, label=0), _variant(50_000, c, label=1),
        _variant(60_000, _alts_for(60_000)[0], label=1),  # monoalelico
        _variant(70_000, _alts_for(70_000)[0], label=1),
        _variant(70_000, _alts_for(70_000)[1], label=1),  # multialelico, mesma classe
    ]
    stats = census(group_by_site(records), verbose=False)
    assert stats["n_sites"] == 3
    assert stats["n_sites_multiallelic"] == 2
    assert stats["n_sites_3plus"] == 1
    assert stats["gold"]["n_with_both_classes"] == 1
    assert stats["any_tier"]["n_multiallelic"] == 2


def test_census_separates_tiers():
    a, b = _alts_for(50_000)[:2]
    records = [_variant(50_000, a, label=1, tier="gold"), _variant(50_000, b, label=0, tier="consensus")]
    stats = census(group_by_site(records), verbose=False)
    # nao ha DOIS gold no sitio -> gold nao conta como multialelico
    assert stats["gold"]["n_multiallelic"] == 0
    assert stats["gold"]["n_with_both_classes"] == 0
    assert stats["any_tier"]["n_with_both_classes"] == 1


def test_census_flags_panel_heterogeneous_sites():
    a, b = _alts_for(50_000)[:2]
    records = [_variant(50_000, a, panel="missense"), _variant(50_000, b, panel="synonymous")]
    assert census(group_by_site(records), verbose=False)["n_sites_panel_heterogeneous"] == 1


# --------------------------------------------------------------------------------------------
# Prioridade / selecao curated
# --------------------------------------------------------------------------------------------


def test_site_priority_ranks_pb_gold_first():
    a, b = _alts_for(50_000)[:2]
    pb_gold = [_variant(50_000, a, label=1), _variant(50_000, b, label=0)]
    pb_mixed = [_variant(50_000, a, label=1), _variant(50_000, b, label=0, tier="consensus")]
    same_class = [_variant(50_000, a, label=1), _variant(50_000, b, label=1)]
    assert site_priority(pb_gold) < site_priority(pb_mixed) < site_priority(same_class)


def test_site_priority_prefers_heterogeneous_panels_then_more_alts():
    a, b, c = _alts_for(50_000)
    het = [_variant(50_000, a, panel="missense"), _variant(50_000, b, panel="synonymous")]
    homo3 = [_variant(50_000, x, panel="missense") for x in (a, b, c)]
    assert site_priority(het) < site_priority(homo3)  # painel heterogeneo ganha de +1 ALT
    homo2 = [_variant(50_000, x, panel="missense") for x in (a, b)]
    assert site_priority(homo3) < site_priority(homo2)  # entre homogeneos, mais ALTs ganha


def test_choose_curated_sites_respects_budget_and_covers_panels():
    records: list[Variant] = []
    panels = ["missense", "splice", "noncoding", "synonymous", "plof", "other"]
    # 30 sitios homogeneos de missense (alta prioridade por volume) + 1 sitio de cada outro painel
    for i in range(30):
        pos = 10_000 + i * 37
        for alt in _alts_for(pos)[:2]:
            records.append(_variant(pos, alt, panel="missense"))
    for j, panel in enumerate(panels[1:], start=1):
        pos = 40_000 + j * 53
        for alt in _alts_for(pos)[:2]:
            records.append(_variant(pos, alt, panel=panel))

    chosen = choose_curated_sites(group_by_site(records), budget=12)
    assert len(chosen) == 12, "orcamento tem que ser cumprido"
    covered = {r.primary_panel for _, rows in chosen for r in rows}
    assert covered == set(panels), f"o passe de diversidade falhou: {covered}"


def test_choose_curated_sites_caps_at_availability():
    a, b = _alts_for(50_000)[:2]
    records = [_variant(50_000, a), _variant(50_000, b)]
    assert len(choose_curated_sites(group_by_site(records), budget=20)) == 1


def test_choose_curated_sites_is_deterministic():
    records = []
    for i in range(20):
        pos = 10_000 + i * 41
        for alt in _alts_for(pos)[:2]:
            records.append(_variant(pos, alt))
    by_site = group_by_site(records)
    first = [k for k, _ in choose_curated_sites(by_site, budget=8)]
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    second = [k for k, _ in choose_curated_sites(group_by_site(shuffled), budget=8)]
    assert first == second


# --------------------------------------------------------------------------------------------
# Selecao completa
# --------------------------------------------------------------------------------------------


def _args(**kw) -> Namespace:
    return Namespace(seed=1, n_curated_sites=kw.get("n_curated_sites", 4),
                     n_statistical=kw.get("n_statistical", 40),
                     n_consensus_pb_sites=kw.get("n_consensus_pb_sites", 0))


# --------------------------------------------------------------------------------------------
# Braco pareado P-vs-B
# --------------------------------------------------------------------------------------------


def test_pb_sites_keeps_only_requested_tier():
    """Um sitio com gold-P + consensus-B NAO e par gold: a pareacao tem que ser dentro do tier."""
    a, b, c = _alts_for(50_000)
    records = [
        _variant(50_000, a, label=1, tier="gold"),
        _variant(50_000, b, label=0, tier="consensus"),
        _variant(50_000, c, label=1, tier="gold"),
    ]
    assert pb_sites(group_by_site(records), tier="gold") == []
    # com um gold-B no lugar do gold-P extra, vira par gold valido
    records[2] = _variant(50_000, c, label=0, tier="gold")
    sites = pb_sites(group_by_site(records), tier="gold")
    assert len(sites) == 1
    kept = sites[0][1]
    assert {r.label_tier for r in kept} == {"gold"}, "consensus vazou para o par gold"
    assert {int(r.binary_label) for r in kept} == {0, 1}
    assert len(kept) == 2


def test_pb_sites_requires_both_classes():
    a, b = _alts_for(50_000)[:2]
    same = [_variant(50_000, a, label=1), _variant(50_000, b, label=1)]
    assert pb_sites(group_by_site(same), tier="gold") == []


def test_pb_sites_budget_samples_deterministically_and_keeps_order():
    records = []
    for i in range(30):
        pos = 10_000 + i * 47
        alts = _alts_for(pos)
        records += [_variant(pos, alts[0], label=1), _variant(pos, alts[1], label=0)]
    by_site = group_by_site(records)
    assert len(pb_sites(by_site, tier="gold")) == 30
    picked = pb_sites(by_site, tier="gold", budget=7, seed=42)
    assert len(picked) == 7
    assert [k for k, _ in picked] == sorted(k for k, _ in picked), "ordem tem que ser estavel"
    assert picked == pb_sites(by_site, tier="gold", budget=7, seed=42)
    assert picked != pb_sites(by_site, tier="gold", budget=7, seed=43)


def test_select_emits_paired_pb_gold_arm():
    a, b = _alts_for(50_000)[:2]
    records = [_variant(50_000, a, label=1), _variant(50_000, b, label=0)]
    for i in range(10):  # enche o braco statistical
        pos = 70_000 + i * 53
        records.append(_variant(pos, _alts_for(pos)[0], label=i % 2))
    rows, stats = select(records, group_by_site(records), _args(), verbose=False)
    pb_rows = [r for r in rows if r["arm"] == "paired_pb_gold"]
    assert len(pb_rows) == 2
    assert {r["binary_label"] for r in pb_rows} == {0, 1}
    assert len({r["site_key"] for r in pb_rows}) == 1, "o par tem que compartilhar o site_key"
    assert stats["paired_pb"]["paired_pb_gold"] == {"n_sites": 1, "n_alleles": 2}
    assert "paired_pb_consensus" not in stats["paired_pb"], "consensus e opt-in (default 0)"


def test_select_consensus_pb_arm_is_opt_in():
    a, b = _alts_for(50_000)[:2]
    records = [_variant(50_000, a, label=1, tier="consensus"),
               _variant(50_000, b, label=0, tier="consensus"),
               _variant(70_000, _alts_for(70_000)[0], label=1)]
    by_site = group_by_site(records)
    _, off = select(records, by_site, _args(), verbose=False)
    assert "paired_pb_consensus" not in off["paired_pb"]
    rows, on = select(records, by_site, _args(n_consensus_pb_sites=5), verbose=False)
    assert on["paired_pb"]["paired_pb_consensus"]["n_sites"] == 1
    assert {r["label_tier"] for r in rows if r["arm"] == "paired_pb_consensus"} == {"consensus"}


def test_select_reports_available_strata():
    records = [_variant(10_000 + i * 61, _alts_for(10_000 + i * 61)[0],
                        label=i % 2, panel=["missense", "splice"][i % 2]) for i in range(20)]
    _, stats = select(records, group_by_site(records), _args(), verbose=False)
    assert stats["strata_available"], "as contagens por estrato tem que ser reportadas"
    assert all("/" in k for k in stats["strata_available"])


def test_select_emits_both_arms_without_duplicates_within_arm():
    records = []
    for i in range(40):
        pos = 10_000 + i * 61
        panel = ["missense", "splice", "noncoding", "synonymous"][i % 4]
        label = i % 2
        for alt in _alts_for(pos)[: (2 if i < 10 else 1)]:
            records.append(_variant(pos, alt, panel=panel, label=label))

    rows, stats = select(records, group_by_site(records), _args(), verbose=False)
    arms = {r["arm"] for r in rows}
    assert arms == {"curated", "statistical"}
    for arm in arms:
        ids = [r["variant_id"] for r in rows if r["arm"] == arm]
        assert len(ids) == len(set(ids)), f"duplicata dentro do braco {arm}"
    assert stats["n_curated_sites"] == 4
    assert stats["n_statistical"] > 0
    # a mesma variante pode aparecer nos dois bracos -- e intencional (contextos de analise distintos)
    assert set(rows[0]).issuperset(set(Variant._fields) | {"arm", "site_key", "site_rank"})


def test_select_statistical_is_gold_only_and_stratified():
    records = []
    for i in range(60):
        pos = 10_000 + i * 71
        tier = "gold" if i % 2 == 0 else "consensus"
        records.append(_variant(pos, _alts_for(pos)[0], tier=tier,
                                panel=["missense", "splice"][i % 2], label=i % 2))
    rows, stats = select(records, group_by_site(records), _args(n_statistical=20), verbose=False)
    stat_rows = [r for r in rows if r["arm"] == "statistical"]
    assert stat_rows, "braco statistical vazio"
    assert {r["label_tier"] for r in stat_rows} == {"gold"}
    assert stats["n_strata"] >= 1


def test_select_is_reproducible_under_seed():
    records = [_variant(10_000 + i * 83, _alts_for(10_000 + i * 83)[0], label=i % 2) for i in range(50)]
    by_site = group_by_site(records)
    a, _ = select(records, by_site, _args(), verbose=False)
    b, _ = select(records, by_site, _args(), verbose=False)
    assert [r["variant_id"] for r in a] == [r["variant_id"] for r in b]


# --------------------------------------------------------------------------------------------
# Plano e validacao de janelas
# --------------------------------------------------------------------------------------------


def test_window_plan_covers_five_centered_and_three_matched():
    plan = window_plan()
    centered = [bp for bp, focal in plan if focal is None]
    matched = [(bp, focal) for bp, focal in plan if focal is not None]
    assert centered == list(ALL_WINDOWS) == [1024, 2048, 4096, 16384, 32768]
    assert [bp for bp, _ in matched] == [4096, 16384, 32768]
    assert {focal for _, focal in matched} == {2047}, "as casadas tem que dividir o MESMO indice focal"
    assert len(plan) == 8


def test_validate_windows_marks_ok_and_records_failures():
    ok_pos = 60_000   # longe das bordas: todas as 8 janelas cabem
    # pos=3000 cabe nas centradas de 1024/2048/4096 (offsets 511/1023/2047) e nas 3 casadas
    # (focal 2047, so crescem para a direita), mas NAO nas centradas de 16384/32768, que
    # precisariam de 8191/16383 bp upstream. Serve para provar o descarte por janela.
    edge_pos = 3000
    rows = [
        {"variant_id": "ok", "chrom": "chr1", "pos_1based": ok_pos,
         "ref": _GENOME["chr1"][ok_pos - 1], "alt": _alts_for(ok_pos)[0]},
        {"variant_id": "edge", "chrom": "chr1", "pos_1based": edge_pos,
         "ref": _GENOME["chr1"][edge_pos - 1], "alt": _alts_for(edge_pos)[0]},
    ]
    report = validate_windows(rows, _fetch, verbose=False)
    assert report["n_variants"] == 2
    assert report["n_variants_all_ok"] == 1
    assert rows[0]["all_windows_ok"] is True and rows[0]["n_windows_ok"] == 8
    assert rows[1]["all_windows_ok"] is False
    # a variante de borda ainda serve para as janelas pequenas -- nao e descartada por inteiro
    assert 0 < rows[1]["n_windows_ok"] < 8
    assert "1024/centered" in rows[1]["windows_ok"]
    assert any("out_of_bounds" in key for key in report["failures"]), report["failures"]
    assert len(report["plan"]) == 8
    assert report["coverage_per_window"]["1024/centered"] == 2


def test_validate_windows_catches_matched_leaving_validated_envelope():
    """O layout `matched` de 32k vai 30.720 bp downstream -- fora dos +-16.383 do sequence_eligible."""
    pos = 100_000  # a 32k centrada cabe (+-16.383), mas a casada precisa de +30.720
    assert pos - 1 + 30_720 > len(_GENOME["chr1"]) - 1
    rows = [{"variant_id": "v", "chrom": "chr1", "pos_1based": pos,
             "ref": _GENOME["chr1"][pos - 1], "alt": _alts_for(pos)[0]}]
    report = validate_windows(rows, _fetch, verbose=False)
    assert rows[0]["all_windows_ok"] is False
    assert any(k.startswith("32768/matched") for k in report["failures"]), report["failures"]
    # mas as centradas seguem validas -> a variante continua utilizavel
    assert "32768/centered" in rows[0]["windows_ok"]


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
