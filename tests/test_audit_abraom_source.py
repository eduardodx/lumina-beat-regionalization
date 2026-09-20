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
        ("1", 150, "A", "G", 1.5),         # af invalida
        ("1", 160, "A", "G", 0.0),         # af degenerada
        ("1", 200, "AT", "G", 0.02),       # nao snv
        ("X", 300, "A", "G", 0.03),        # fora dos autossomos
        ("2", 400, "A", "G", 0.04),        # membro de estudo
        ("3", 450, "A", "G", 0.06),        # alelo de avaliacao
        ("8", 500, "A", "G", 0.05),        # chr8 reservado
    ])
    motivos = aud.classificar(
        linhas, membros={aud.chave("chr2", 400, "A", "G")},
        alelos_de_avaliacao={aud.chave("chr3", 450, "A", "G")}, reservar_chr8=True)
    assert list(motivos) == [aud.MOTIVO_OK, aud.MOTIVO_AF_INVALIDA, aud.MOTIVO_AF_DEGENERADA,
                             aud.MOTIVO_NAO_SNV, aud.MOTIVO_FORA_DOS_AUTOSSOMOS,
                             aud.MOTIVO_MEMBRO_DE_ESTUDO, aud.MOTIVO_ALELO_DE_AVALIACAO, aud.MOTIVO_CHR8]


def test_af_invalida_cobre_nan_negativo_e_maior_que_um():
    assert aud.af_invalida(float("nan")) and aud.af_invalida(-0.1) and aud.af_invalida(1.1)
    assert aud.af_invalida("nao_e_numero") and aud.af_invalida(None)
    assert not aud.af_invalida(0.0) and not aud.af_invalida(1.0) and not aud.af_invalida(0.5)


def test_af_degenerada_sai_por_padrao_mas_pode_ser_mantida():
    """AF 0 (nao observada) e AF 1 (fixada) nao sao variacao populacional utilizavel."""
    linhas = _abraom([("1", 100, "A", "G", 0.0), ("1", 200, "A", "G", 1.0), ("1", 300, "A", "G", 0.2)])
    padrao = aud.classificar(linhas, membros=set(), alelos_de_avaliacao=set(), reservar_chr8=False)
    assert list(padrao) == [aud.MOTIVO_AF_DEGENERADA, aud.MOTIVO_AF_DEGENERADA, aud.MOTIVO_OK]
    mantendo = aud.classificar(linhas, membros=set(), alelos_de_avaliacao=set(), reservar_chr8=False,
                               manter_af_degenerada=True)
    assert list(mantendo) == [aud.MOTIVO_OK] * 3


def test_alelo_de_avaliacao_sai_do_pool():
    """Treinar o adapter a reconstruir um alelo que sera pontuado e diferente de so ver o contexto dele."""
    linhas = _abraom([("3", 450, "A", "G", 0.06)])
    motivos = aud.classificar(linhas, membros=set(),
                              alelos_de_avaliacao={aud.chave("chr3", 450, "A", "G")}, reservar_chr8=False)
    assert list(motivos) == [aud.MOTIVO_ALELO_DE_AVALIACAO]


def test_duplicata_com_af_conflitante_e_detectada():
    iguais = _abraom([("1", 100, "A", "G", 0.01), ("1", 100, "A", "G", 0.01)])
    assert len(aud.duplicatas_conflitantes(iguais)) == 0, "duplicata exata nao e conflito"
    conflito = _abraom([("1", 100, "A", "G", 0.01), ("1", 100, "A", "G", 0.99)])
    assert aud.duplicatas_conflitantes(conflito)["chave"].nunique() == 1


def test_membro_de_estudo_e_pego_mesmo_com_prefixo_diferente():
    """O TSV traz `2` e o membership traz `chr2`: sem normalizar, o membro passaria batido."""
    linhas = _abraom([("2", 400, "A", "G", 0.04)])
    motivos = aud.classificar(linhas, membros={aud.chave("chr2", 400, "A", "G")},
                              alelos_de_avaliacao=set(), reservar_chr8=True)
    assert list(motivos) == [aud.MOTIVO_MEMBRO_DE_ESTUDO]


def test_chr8_so_sai_quando_reservado():
    linhas = _abraom([("8", 500, "A", "G", 0.05)])
    comum = dict(membros=set(), alelos_de_avaliacao=set())
    assert list(aud.classificar(linhas, reservar_chr8=True, **comum)) == [aud.MOTIVO_CHR8]
    assert list(aud.classificar(linhas, reservar_chr8=False, **comum)) == [aud.MOTIVO_OK]


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
            ("5", 900, "A", "G", 0.07),   # esta na validacao do core: sera pontuada depois
        ]).to_csv(tsv, sep="\t", index=False)

        membros = raiz / "membros.parquet"
        pd.DataFrame({"chrom": ["chr2"], "pos_1based": [400], "ref": ["A"], "alt": ["G"]}).to_parquet(
            membros, index=False)

        snapshot = raiz / "snap.parquet"
        pd.DataFrame({"chrom": ["chr1", "chr5"], "pos_1based": [100, 900], "ref": ["A", "A"],
                      "alt": ["G", "G"], "role": ["train", "validation"]}).to_parquet(snapshot, index=False)

        selecao = raiz / "selecao.parquet"
        pd.DataFrame({"chrom": ["chr7"], "pos_1based": [1], "ref": ["A"], "alt": ["G"]}).to_parquet(
            selecao, index=False)

        saida = raiz / "out"
        rc = aud.main(["--abraom", str(tsv), "--brazil-variants", str(membros), "--selection", str(selecao),
                       "--snapshot", str(snapshot), "--out-dir", str(saida)])
        assert rc == 0
        relatorio = json.loads((saida / "auditoria_abraom.json").read_text(encoding="utf-8"))
        assert relatorio["por_motivo"] == {aud.MOTIVO_OK: 1, aud.MOTIVO_NAO_SNV: 1,
                                           aud.MOTIVO_MEMBRO_DE_ESTUDO: 1,
                                           aud.MOTIVO_ALELO_DE_AVALIACAO: 1, aud.MOTIVO_CHR8: 1}
        assert relatorio["pronto_para_amostrar"] is True
        assert relatorio["falta_antes_de_treinar"], "pronto para amostrar nao e pronto para treinar"
        assert str(membros) in relatorio["identidades_das_exclusoes"]
        assert str(snapshot) in relatorio["identidades_das_exclusoes"]
        pool = pd.read_parquet(saida / "abraom_pool.parquet")
        assert list(pool["chrom"]) == ["chr1"], "o pool sai com o cromossomo normalizado"
        # A variante de treino da cabeca CONTINUA no pool: sobreposicao com o treino e declarada, nao eliminada.
        assert relatorio["sobreposicao_com_o_treino_da_cabeca"][str(snapshot)] == 1
        assert len(relatorio["saidas"]["pool_sha256"]) == 64


def test_sem_exclusoes_nao_publica_o_pool():
    """Publicar um pool sem tirar estudos e avaliacao seria entregar dado contaminado pronto para treinar."""
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        tsv = raiz / "abraom.tsv"
        _abraom([("1", 100, "A", "G", 0.01)]).to_csv(tsv, sep="	", index=False)
        saida = raiz / "out"
        rc = aud.main(["--abraom", str(tsv), "--out-dir", str(saida)])
        assert rc == 2
        relatorio = json.loads((saida / "auditoria_abraom.json").read_text(encoding="utf-8"))
        assert relatorio["pronto_para_amostrar"] is False and len(relatorio["pendencias"]) == 3
        assert not (saida / "abraom_pool.parquet").exists()


def test_end_to_end_reprova_af_conflitante():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        tsv = raiz / "abraom.tsv"
        _abraom([("1", 100, "A", "G", 0.01), ("1", 100, "A", "G", 0.99)]).to_csv(tsv, sep="	", index=False)
        assert aud.main(["--abraom", str(tsv), "--out-dir", str(raiz / "out")]) == 2


def test_end_to_end_reprova_arquivo_sem_as_colunas():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        tsv = raiz / "errado.tsv"
        pd.DataFrame({"cromossomo": ["1"], "posicao": [100]}).to_csv(tsv, sep="\t", index=False)
        assert aud.main(["--abraom", str(tsv), "--out-dir", str(raiz / "out")]) == 2


def test_faltar_uma_exclusao_ja_impede_publicar():
    """So a selecao, sem os snapshots, deixaria a validacao e o teste da cabeca entrarem no pool -- e vice-versa."""
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        tsv = raiz / "abraom.tsv"
        _abraom([("1", 100, "A", "G", 0.01)]).to_csv(tsv, sep="\t", index=False)
        membros = raiz / "membros.parquet"
        pd.DataFrame({"chrom": ["chr2"], "pos_1based": [400], "ref": ["A"], "alt": ["G"]}).to_parquet(
            membros, index=False)
        selecao = raiz / "selecao.parquet"
        pd.DataFrame({"chrom": ["chr7"], "pos_1based": [1], "ref": ["A"], "alt": ["G"]}).to_parquet(
            selecao, index=False)
        saida = raiz / "out"
        rc = aud.main(["--abraom", str(tsv), "--brazil-variants", str(membros),
                       "--selection", str(selecao), "--out-dir", str(saida)])
        assert rc == 2, "faltou --snapshot: nao pode publicar"
        relatorio = json.loads((saida / "auditoria_abraom.json").read_text(encoding="utf-8"))
        assert any("--snapshot" in p for p in relatorio["pendencias"]), relatorio["pendencias"]
        assert not (saida / "abraom_pool.parquet").exists()


def test_membros_contados_por_estudo_e_papel():
    """A afirmacao e sobre COMPOSICAO: quantos membros de cada estudo o ABraOM tem, nao quantas linhas sairam."""
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        membros = raiz / "membros.parquet"
        pd.DataFrame({
            "chrom": ["chr1", "chr2", "chr3"], "pos_1based": [100, 200, 300],
            "ref": ["A", "A", "A"], "alt": ["G", "G", "G"],
            "study_id": ["br_population_observed", "br_population_observed", "br_clinical_evidence"],
            "member_role": ["case", "control", "case"],
        }).to_parquet(membros, index=False)
        no_arquivo = {aud.chave("chr1", 100, "A", "G"), aud.chave("chr3", 300, "A", "G")}
        contagem = aud.membros_encontrados_por_estudo(membros, no_arquivo)
        assert contagem["br_population_observed"]["case"] == {"membros": 1, "no_abraom": 1}
        assert contagem["br_population_observed"]["control"] == {"membros": 1, "no_abraom": 0}
        assert contagem["br_clinical_evidence"]["case"] == {"membros": 1, "no_abraom": 1}


def test_sem_study_id_a_contagem_por_estudo_e_none():
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "m.parquet"
        pd.DataFrame({"chrom": ["chr1"], "pos_1based": [1], "ref": ["A"], "alt": ["G"]}).to_parquet(
            caminho, index=False)
        assert aud.membros_encontrados_por_estudo(caminho, set()) is None


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
