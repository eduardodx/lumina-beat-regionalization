"""Prova que a auditoria de dados da regionalizacao detecta cada problema E nao dispara em falso.

Precisa de numpy + pandas (+ pyarrow ou fastparquet para o teste ponta-a-ponta):
    PYTHONPATH=. python tests/test_audit_regionalization_data.py

Cada detector tem um caso positivo (o problema existe) e um controle (nao existe). Um detector que
so e testado no positivo pode estar dizendo "confirmado" para qualquer entrada.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    print("SKIP: precisa de numpy + pandas")
    sys.exit(0)

from scripts.audit_regionalization_data import (  # noqa: E402
    audit_abraom_tsv,
    audit_af_gnomad_conditional,
    audit_overlap,
    audit_pair_abraom_concordance,
    build_key,
    composition,
    main,
)


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _require_parquet_engine() -> None:
    for module in ("pyarrow", "fastparquet"):
        try:
            __import__(module)
            return
        except ImportError:
            continue
    raise Skip("sem engine de parquet (pyarrow/fastparquet)")


def _slice(n: int, seed: int, *, conditional: bool) -> pd.DataFrame:
    """Imita o pipeline regional: 1 em cada 10 variantes e nao-SNV, e nao-SNV nunca entra no ABraOM."""
    rng = np.random.default_rng(seed)
    non_snv = np.array([i % 10 == 0 for i in range(n)])
    present = (rng.random(n) < 0.3) & ~non_snv
    af = rng.random(n) * 0.2
    if conditional:
        af_gnomad = np.where(present, af, np.nan)                      # o pipeline regional real
    else:
        af_gnomad = np.where((rng.random(n) < 0.9) & ~non_snv, af, np.nan)  # gnomAD consultado para SNVs
    keys = [f"{1 + i % 20}:{1000 + i}:{'AT:A' if non_snv[i] else 'A:G'}" for i in range(n)]
    return pd.DataFrame({
        "variant_key": keys,
        "GeneSymbol": [f"G{i % 25}" for i in range(n)],
        "label": rng.integers(0, 2, n),
        "variant_type": np.where(non_snv, "Deletion", "single nucleotide variant"),
        "abraom_present": present,
        "af_gnomad": af_gnomad,
    })


def test_detects_af_gnomad_conditional_on_abraom():
    df = _slice(2000, 1, conditional=True)
    snv = ~df["variant_key"].str.contains("AT:A")
    res = audit_af_gnomad_conditional(df[snv], "br/SNV")
    assert res["p_af_preenchido_se_abraom_ausente"] == 0.0, res
    assert res["veredito"].startswith("CONFIRMADO"), res["veredito"]


def test_does_not_flag_when_gnomad_was_looked_up_for_everyone():
    df = _slice(2000, 2, conditional=False)
    snv = ~df["variant_key"].str.contains("AT:A")
    res = audit_af_gnomad_conditional(df[snv], "br/SNV")
    assert res["p_af_preenchido_se_abraom_ausente"] > 0.5, res
    assert res["veredito"].startswith("NAO"), res["veredito"]


def _pool_and_pairs(seed: int, *, match_on_abraom: bool):
    """Pool com estratos MISTOS (q=0.5). Pares que copiam o status do BR = matcher que casou por ABraOM."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(40):
        for lab in (0, 1):
            for j in range(20):
                rows.append({"GeneSymbol": f"G{g}", "label": lab, "variant_type": "single nucleotide variant",
                             "abraom_present": j % 2 == 0})
    pool = pd.DataFrame(rows)
    n = 800
    br_present = rng.random(n) < 0.3
    nb_present = br_present.copy() if match_on_abraom else rng.random(n) < 0.5
    pairs = pd.DataFrame({
        "br_GeneSymbol": [f"G{i % 40}" for i in range(n)],
        "br_label": [i % 2 for i in range(n)],
        "br_variant_type": "single nucleotide variant",
        "br_abraom_present": br_present,
        "nonbr_abraom_present": nb_present,
    })
    return pool, pairs


def test_flags_pairs_matched_on_abraom_presence():
    pool, pairs = _pool_and_pairs(3, match_on_abraom=True)
    res = audit_pair_abraom_concordance(pairs, pool)
    blk = res["estratos_mistos"]
    assert blk["concordancia_observada"] == 1.0, blk
    assert abs(blk["concordancia_esperada_sorteio_no_estrato"] - 0.5) < 1e-9, blk
    assert res["veredito"].startswith("EVIDENCIA FORTE"), res["veredito"]


def test_does_not_flag_pairs_drawn_ignoring_abraom():
    pool, pairs = _pool_and_pairs(4, match_on_abraom=False)
    res = audit_pair_abraom_concordance(pairs, pool)
    assert abs(res["estratos_mistos"]["excesso"]) < 0.05, res["estratos_mistos"]
    assert res["veredito"].startswith("sem evidencia"), res["veredito"]


def test_pure_strata_cannot_be_blamed_on_the_matcher():
    """Estrato onde o pool e 100% ausente do ABraOM: concordancia forcada, nao escolha do matcher."""
    pool = pd.DataFrame({"GeneSymbol": ["G0"] * 50, "label": [1] * 50,
                         "variant_type": ["single nucleotide variant"] * 50, "abraom_present": [False] * 50})
    pairs = pd.DataFrame({"br_GeneSymbol": ["G0"] * 30, "br_label": [1] * 30,
                          "br_variant_type": ["single nucleotide variant"] * 30,
                          "br_abraom_present": [False] * 30, "nonbr_abraom_present": [False] * 30})
    res = audit_pair_abraom_concordance(pairs, pool)
    assert res["estratos_mistos"]["n"] == 0, res
    assert res["veredito"].startswith("sem estratos mistos"), res["veredito"]


def test_composition_counts_benigns_by_snv():
    """A campanha so-SNV depende das BENIGNAS que sobram: a contagem tem de separar tipo e rotulo."""
    df = pd.DataFrame({
        "variant_key": ["1:10:A:G", "1:11:C:T", "1:12:G:A", "1:13:AT:A", "1:14:A:AT"],
        "variant_type": ["single nucleotide variant"] * 3 + ["Deletion", "Insertion"],
        "label": [1, 0, 0, 0, 1],
    })
    res = composition(df)
    assert (res["snv_P"], res["snv_B"], res["nao_snv_P"], res["nao_snv_B"]) == (1, 2, 1, 1), res
    assert res["variant_type_x_label"]["Deletion"] == {"P": 0, "B": 1}, res["variant_type_x_label"]


def test_mosaic_chr_prefix_joins_the_regional_key():
    """Regional usa '17:...', Mosaic usa 'chr17'. Sem normalizar, a sobreposicao sairia 0 em silencio."""
    keys = build_key(pd.Series(["chr17", "chrX", "chrMT"]), pd.Series([43000000, 5, 9]),
                     pd.Series(["a", "C", "G"]), pd.Series(["G", "t", "A"]))
    assert list(keys) == ["17:43000000:A:G", "X:5:C:T", "M:9:G:A"], list(keys)
    overlap = audit_overlap({"gold": set(keys)}, {"t_br": {"17:43000000:A:G", "8:1:A:C"}})
    assert overlap["gold"]["t_br"] == 1, overlap


def test_abraom_tsv_reports_floor_and_index_gap():
    with tempfile.TemporaryDirectory() as d:
        tsv = Path(d) / "abraom.tsv"
        tsv.write_text("chrom\tpos\tref\talt\taf_abraom\n"
                       "17\t100\tA\tG\t0.0004\n"      # rara: no cru, fora do indice
                       "chr17\t200\tC\tT\t0.30\n"     # comum: no cru e no indice
                       "1\t300\tG\tA\t0.0\n", encoding="utf-8")
        br = pd.DataFrame({"variant_key": ["17:100:A:G", "17:200:C:T", "2:400:T:C"],
                           "abraom_present": [False, True, True]})
        res = audit_abraom_tsv(tsv, br, chunksize=2)
    assert res["af_minima_positiva"] == 0.0004, res
    assert res["af_positiva_abaixo_de"]["0.005"] == 1, res
    assert res["t_br_no_tsv_cru"] == 2, res
    assert res["t_br_no_tsv_cru_mas_abraom_present_false"] == 1, res
    assert res["t_br_abraom_present_mas_fora_do_tsv_cru"] == 1, res


def test_main_end_to_end_writes_report():
    _require_parquet_engine()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        br = _slice(300, 5, conditional=True)
        nonbr = _slice(900, 6, conditional=True)
        pool, pairs = _pool_and_pairs(7, match_on_abraom=True)
        pairs["br_variant_key"] = [f"4:{i}:C:T" for i in range(len(pairs))]
        pairs["nonbr_variant_key"] = [f"3:{i}:A:C" for i in range(len(pairs))]
        pairs["nonbr_variant_type"] = "single nucleotide variant"
        pairs["nonbr_label"] = pairs["br_label"]
        nonbr = pd.concat([nonbr, pool.assign(variant_key=[f"5:{i}:C:A" for i in range(len(pool))],
                                              af_gnomad=np.nan)], ignore_index=True)
        br.to_parquet(tmp / "br.parquet")
        nonbr.to_parquet(tmp / "nonbr.parquet")
        pairs.to_parquet(tmp / "pairs.parquet")
        # i=1 do slice BR e SNV: chrom 2, pos 1001 (i=0 e a delecao sintetica)
        pd.DataFrame({"chrom": ["chr2"], "pos_1based": [1001], "ref": ["A"], "alt": ["G"],
                      "label_tier": ["gold"]}).to_parquet(tmp / "pb.parquet")
        out = tmp / "audit.json"
        rc = main(["--br", str(tmp / "br.parquet"), "--nonbr", str(tmp / "nonbr.parquet"),
                   "--pairs", str(tmp / "pairs.parquet"), "--mosaic-examples", str(tmp / "pb.parquet"),
                   "--out", str(out)])
        assert rc == 0
        rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["C_sobreposicao_mosaic"]["gold"]["t_br_slice"] == 1, rep["C_sobreposicao_mosaic"]
    assert rep["B_pareamento_abraom"]["veredito"].startswith("EVIDENCIA FORTE"), rep["B_pareamento_abraom"]
    blocks = {b["slice"]: b for b in rep["A_af_gnomad_condicional"]}
    assert blocks["br_only/SNV"]["veredito"].startswith("CONFIRMADO"), blocks["br_only/SNV"]
    assert blocks["br_only/nao-SNV"]["abraom_present"] == 0, blocks["br_only/nao-SNV"]
    assert rep["D_composicao"]["t_br_pareado"]["n"] == len(pairs), rep["D_composicao"]["t_br_pareado"]


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
    sys.exit(1 if failed else 0)
