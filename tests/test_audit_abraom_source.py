"""Prova que o pool do ABraOM sai limpo e que a normalizacao de cromossomo nao e opcional.

O TSV usa `1` e o snapshot usa `chr1`: juntar sem normalizar daria zero sobreposicao em silencio -- o pior erro
possivel aqui, porque parece "nenhum vazamento".
    PYTHONPATH=. python3 tests/test_audit_abraom_source.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.audit_abraom_source as aud  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _abraom(linhas):
    return pd.DataFrame(linhas, columns=["chrom", "pos", "ref", "alt", "af_abraom"])


def test_chave_normaliza_o_cromossomo_dos_dois_lados():
    assert aud.chave("1", 100, "a", "g") == aud.chave("chr1", 100, "A", "G")
    assert aud.chave("1", 100, "A", "G") != aud.chave("2", 100, "A", "G")


def test_e_snv_recusa_indel_e_base_estranha():
    assert aud.e_snv("A", "G")
    assert not aud.e_snv("AT", "G"), "delecao"
    assert not aud.e_snv("A", "AT"), "insercao"
    assert not aud.e_snv("N", "G"), "base fora de ACGT"
    assert not aud.e_snv("A", "A"), "ref igual a alt"


def test_classificar_aplica_as_regras_na_ordem():
    linhas = _abraom([
        ("1", 100, "A", "G", 0.01),        # ok
        ("1", 200, "AT", "G", 0.02),       # nao snv
        ("X", 300, "A", "G", 0.03),        # fora dos autossomos
        ("2", 400, "A", "G", 0.04),        # membro de estudo
        ("8", 500, "A", "G", 0.05),        # chr8 reservado
    ])
    membros = {aud.chave("chr2", 400, "A", "G")}
    motivos = aud.classificar(linhas, membros=membros, reservar_chr8=True)
    assert list(motivos) == [aud.MOTIVO_OK, aud.MOTIVO_NAO_SNV, aud.MOTIVO_FORA_DOS_AUTOSSOMOS,
                             aud.MOTIVO_MEMBRO_DE_ESTUDO, aud.MOTIVO_CHR8]


def test_membro_de_estudo_e_pego_mesmo_com_prefixo_diferente():
    """O TSV traz `2` e o membership traz `chr2`: sem normalizar, o membro passaria batido."""
    linhas = _abraom([("2", 400, "A", "G", 0.04)])
    motivos = aud.classificar(linhas, membros={aud.chave("chr2", 400, "A", "G")}, reservar_chr8=True)
    assert list(motivos) == [aud.MOTIVO_MEMBRO_DE_ESTUDO]


def test_chr8_so_sai_quando_reservado():
    linhas = _abraom([("8", 500, "A", "G", 0.05)])
    assert list(aud.classificar(linhas, membros=set(), reservar_chr8=True)) == [aud.MOTIVO_CHR8]
    assert list(aud.classificar(linhas, membros=set(), reservar_chr8=False)) == [aud.MOTIVO_OK]


def test_distribuicao_af_cobre_o_espectro_declarado():
    af = pd.Series([0.0005, 0.003, 0.02, 0.3, 0.9])
    bins = aud.distribuicao_af(af)
    assert sum(bins.values()) == 5, bins
    assert len(bins) == len(aud.AF_BINS) - 1


def test_end_to_end_publica_pool_normalizado_e_relatorio():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        tsv = raiz / "abraom.tsv"
        _abraom([
            ("1", 100, "A", "G", 0.01),
            ("1", 200, "AT", "G", 0.02),
            ("8", 500, "A", "G", 0.05),
            ("2", 400, "A", "G", 0.04),
        ]).to_csv(tsv, sep="\t", index=False)

        membros = raiz / "membros.parquet"
        pd.DataFrame({"chrom": ["chr2"], "pos_1based": [400], "ref": ["A"], "alt": ["G"]}).to_parquet(
            membros, index=False)

        snapshot = raiz / "snap.parquet"
        pd.DataFrame({"chrom": ["chr1"], "pos_1based": [100], "ref": ["A"], "alt": ["G"],
                      "role": ["train"]}).to_parquet(snapshot, index=False)

        saida = raiz / "out"
        rc = aud.main(["--abraom", str(tsv), "--brazil-variants", str(membros),
                       "--snapshot", str(snapshot), "--out-dir", str(saida)])
        assert rc == 0
        relatorio = json.loads((saida / "auditoria_abraom.json").read_text(encoding="utf-8"))
        assert relatorio["por_motivo"] == {aud.MOTIVO_OK: 1, aud.MOTIVO_NAO_SNV: 1,
                                           aud.MOTIVO_MEMBRO_DE_ESTUDO: 1, aud.MOTIVO_CHR8: 1}
        pool = pd.read_parquet(saida / "abraom_pool.parquet")
        assert list(pool["chrom"]) == ["chr1"], "o pool sai com o cromossomo normalizado"
        assert relatorio["sobreposicao_com_o_treino"][str(snapshot)] == 1, relatorio["sobreposicao_com_o_treino"]
        assert len(relatorio["saidas"]["pool_sha256"]) == 64


def test_end_to_end_reprova_arquivo_sem_as_colunas():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        tsv = raiz / "errado.tsv"
        pd.DataFrame({"cromossomo": ["1"], "posicao": [100]}).to_csv(tsv, sep="\t", index=False)
        assert aud.main(["--abraom", str(tsv), "--out-dir", str(raiz / "out")]) == 2


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
