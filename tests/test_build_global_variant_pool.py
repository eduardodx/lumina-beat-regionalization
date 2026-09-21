"""Prova que o pool global sai da receita declarada, e que o piso de AF e medido mesmo quando nao e aplicado.

O registro do VCF e duck-typed, entao isto roda sem pysam e sem rede -- que e exatamente a parte que costuma
esconder erro de indice de ALT e de filtro.
    PYTHONPATH=. python3 tests/test_build_global_variant_pool.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.build_global_variant_pool as gp  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


class FakeFilter(dict):
    pass


class FakeRec:
    """Registro com a mesma superficie que o script usa do pysam."""

    def __init__(self, chrom, pos, ref, alts, info, filtros=()):
        self.chrom, self.pos, self.ref, self.alts = chrom, pos, ref, alts
        self.info = info
        self.filter = FakeFilter({f: None for f in filtros})


def _fai(caminho: Path, comprimento: int = 1_000_000) -> Path:
    linhas = [f"chr{i}\t{comprimento}\t0\t60\t61" for i in range(1, 23)]
    linhas.append(f"chrX\t{comprimento}\t0\t60\t61")
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def test_fai_le_so_autossomos():
    with tempfile.TemporaryDirectory() as tmp:
        comprimentos = gp.comprimentos_do_fai(_fai(Path(tmp) / "g.fai"))
        assert len(comprimentos) == 22 and "chrX" not in comprimentos


def test_fai_incompleto_falha():
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "g.fai"
        caminho.write_text("chr1\t1000\t0\t60\t61\n", encoding="utf-8")
        try:
            gp.comprimentos_do_fai(caminho)
        except ValueError:
            return
        raise AssertionError("faltando 21 autossomos, tinha de falhar")


def test_alocacao_soma_exatamente_o_pedido():
    alocacao = gp.alocar_regioes({"chr1": 3.0, "chr2": 1.0, "chr3": 1.0}, n_regioes=10)
    assert sum(alocacao.values()) == 10, alocacao
    assert alocacao["chr1"] == 6, alocacao


def test_alocacao_casada_segue_os_pesos_e_nao_o_comprimento():
    """O ponto do modo `casado_ao_abraom`: chr16 pequeno pode receber mais regiao que chr1 grande."""
    alocacao = gp.alocar_regioes({"chr1": 1.0, "chr16": 9.0}, n_regioes=100)
    assert alocacao["chr16"] == 90 and alocacao["chr1"] == 10


def test_regioes_cabem_no_cromossomo_e_sao_deterministicas():
    comprimentos = {"chr1": 1000, "chr2": 1000}
    alocacao = {"chr1": 5, "chr2": 5}
    uma = gp.sortear_regioes(np.random.default_rng(1), comprimentos, alocacao, tamanho_bp=100)
    outra = gp.sortear_regioes(np.random.default_rng(1), comprimentos, alocacao, tamanho_bp=100)
    assert uma == outra
    for chrom, inicio, fim in uma:
        assert 0 <= inicio and fim <= comprimentos[chrom] and fim - inicio == 100


def test_regiao_maior_que_o_cromossomo_falha():
    try:
        gp.sortear_regioes(np.random.default_rng(1), {"chr1": 50}, {"chr1": 1}, tamanho_bp=100)
    except ValueError:
        return
    raise AssertionError("regiao maior que o cromossomo tinha de falhar")


def test_indice_do_alt_e_respeitado_em_multialelico():
    """O erro classico: pegar AF[0] para o segundo ALT. Aqui o segundo ALT tem AF propria."""
    rec = FakeRec("chr1", 100, "A", ("G", "T"), {"AF_joint": [0.1, 0.9], "AC_joint": [10, 90]})
    linhas, _ = gp.linhas_do_registro(rec, af_campo="AF_joint", af_min=None)
    assert [(l["alt"], l[gp.COLUNA_AF]) for l in linhas] == [("G", 0.1), ("T", 0.9)]


def test_registro_sem_pass_e_recusado():
    rec = FakeRec("chr1", 100, "A", ("G",), {"AF_joint": [0.5]}, filtros=("EXOMES_FILTERED",))
    linhas, motivos = gp.linhas_do_registro(rec, af_campo="AF_joint", af_min=None)
    assert linhas == [] and motivos[gp.MOTIVO_FILTRO] == 1


def test_ac0_nao_vira_variante_rara():
    """ADR 0003 do Mosaic: ac0 e ausencia de observacao, nao frequencia baixa."""
    rec = FakeRec("chr1", 100, "A", ("G",), {"AF_joint": [0.0], "AC_joint": [0]})
    linhas, motivos = gp.linhas_do_registro(rec, af_campo="AF_joint", af_min=None)
    assert linhas == [] and motivos[gp.MOTIVO_AC_ZERO] == 1


def test_indel_e_ausencia_de_af_tem_motivos_proprios():
    indel = FakeRec("chr1", 100, "AT", ("A",), {"AF_joint": [0.5], "AC_joint": [5]})
    _, motivos = gp.linhas_do_registro(indel, af_campo="AF_joint", af_min=None)
    assert motivos["nao_snv"] == 1
    sem_af = FakeRec("chr1", 100, "A", ("G",), {"AC_joint": [5]})
    _, motivos = gp.linhas_do_registro(sem_af, af_campo="AF_joint", af_min=None)
    assert motivos[gp.MOTIVO_SEM_AF] == 1


def test_piso_recusa_e_conta():
    rec = FakeRec("chr1", 100, "A", ("G",), {"AF_joint": [1e-6], "AC_joint": [1]})
    linhas, motivos = gp.linhas_do_registro(rec, af_campo="AF_joint", af_min=4.27e-4)
    assert linhas == [] and motivos[gp.MOTIVO_ABAIXO_DO_PISO] == 1
    linhas, _ = gp.linhas_do_registro(rec, af_campo="AF_joint", af_min=None)
    assert len(linhas) == 1, "sem piso declarado, a variante entra"


def _coletar(fetch, regioes, *, seed=1, af_min=None, teto=None, por_bin=10_000):
    return gp.coletar(fetch, regioes, rng=np.random.default_rng(seed), af_campo="AF_joint",
                      af_min=af_min, max_por_bin_por_regiao=teto, por_bin=por_bin)


def test_teto_por_regiao_limita_o_aglomerado():
    def fetch(chrom, inicio, fim):
        return [FakeRec(chrom, inicio + i, "A", ("G",), {"AF_joint": [0.2], "AC_joint": [9]})
                for i in range(10)]
    frame, motivos, _ = _coletar(fetch, [("chr1", 0, 100)], teto=3)
    assert len(frame) == 3 and motivos[gp.MOTIVO_TETO_DA_REGIAO] == 7


def test_teto_sorteia_em_vez_de_pegar_as_primeiras():
    """O DEFEITO do piloto de 20/09: `linhas[:n]` guardava so o comeco de cada regiao de 20 kb."""
    def fetch(chrom, inicio, fim):
        return [FakeRec(chrom, inicio + i, "A", ("G",), {"AF_joint": [0.2], "AC_joint": [9]})
                for i in range(200)]
    posicoes = set()
    for seed in range(6):
        frame, _, _ = _coletar(fetch, [("chr1", 0, 1000)], seed=seed, teto=5)
        posicoes.update(int(p) for p in frame["pos"])
    assert max(posicoes) > 50, f"so posicoes do comeco: {sorted(posicoes)[:10]}..{max(posicoes)}"
    assert len(posicoes) > 10, posicoes


def test_bin_escasso_guarda_tudo_enquanto_o_abundante_e_podado():
    """O ponto do teto POR BIN: no gnomAD 96% cai no bin raro e 0,15% no comum."""
    def fetch(chrom, inicio, fim):
        raras = [FakeRec(chrom, inicio + i, "A", ("G",), {"AF_joint": [1e-4], "AC_joint": [9]})
                 for i in range(100)]
        comuns = [FakeRec(chrom, inicio + 500 + i, "A", ("G",), {"AF_joint": [0.8], "AC_joint": [9]})
                  for i in range(2)]
        return raras + comuns
    frame, _, vistos = _coletar(fetch, [("chr1", 0, 1000)], teto=10)
    por_bin = frame[gp.COLUNA_AF].map(lambda af: af > 0.5).value_counts()
    assert por_bin.get(True, 0) == 2, "o bin escasso tinha de guardar as duas"
    assert por_bin.get(False, 0) == 10, "o bin abundante tinha de ser podado ao teto"
    assert vistos[gp.bin_de_af(1e-4)] == 10 and vistos[gp.bin_de_af(0.8)] == 2


def test_reservatorio_respeita_a_capacidade_do_bin():
    def fetch(chrom, inicio, fim):
        return [FakeRec(chrom, inicio + i, "A", ("G",), {"AF_joint": [0.2], "AC_joint": [9]})
                for i in range(50)]
    frame, _, vistos = _coletar(fetch, [("chr1", 0, 1000), ("chr1", 2000, 3000)], teto=None, por_bin=7)
    assert len(frame) == 7, len(frame)
    assert sum(vistos.values()) == 100, vistos


def test_bin_de_af_usa_o_mesmo_rotulo_que_o_relatorio():
    """As duas tabelas do relatorio tem de casar.

    O pandas escreve o primeiro bin como (-0.001, 0.001] por causa do include_lowest. Montar
    "(0.0, 0.001]" na mao zerava `fracao_amostrada_por_bin` no bin mais raro -- que e exatamente o bin
    que decide se o piso de AF entra.
    """
    from scripts.audit_abraom_source import distribuicao_af

    valores = pd.Series([0.0005, 0.003, 0.9])
    chaves = set(distribuicao_af(valores))
    for af in valores:
        assert gp.bin_de_af(af) in chaves, (af, gp.bin_de_af(af), sorted(chaves))
    assert gp.bin_de_af(0.001) == gp.bin_de_af(0.0005)
    assert gp.bin_de_af(1.0) == gp.bin_de_af(0.9)


def test_fracao_amostrada_casa_com_o_bin_mais_raro():
    """Prova de ponta a ponta do defeito: o bin raro precisa aparecer nas DUAS tabelas."""
    from scripts.audit_abraom_source import distribuicao_af

    raras = pd.Series([1e-4] * 5)
    assert distribuicao_af(raras)[gp.bin_de_af(1e-4)] == 5


def test_duplicata_entre_regioes_sai():
    def fetch(chrom, inicio, fim):
        return [FakeRec(chrom, 500, "A", ("G",), {"AF_joint": [0.2], "AC_joint": [9]})]
    frame, motivos, _ = _coletar(fetch, [("chr1", 0, 1000), ("chr1", 400, 1400)], teto=None)
    assert len(frame) == 1 and motivos["duplicata_entre_regioes"] == 1


def test_custo_do_piso_e_medido_mesmo_sem_aplicar():
    frame = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "pos": [1, 2, 3],
                          gp.COLUNA_AF: [1e-6, 1e-5, 0.3]})
    custo = gp.custo_do_piso(frame, 4.27e-4)
    assert custo["removeria"] == 2 and custo["fracao"] == round(2 / 3, 4)
    assert custo["por_cromossomo"] == {"chr1": 2}


def _entradas(raiz: Path):
    _fai(raiz / "g.fai")
    abraom = raiz / "abraom.parquet"
    pd.DataFrame({"chrom": ["chr1"] * 3 + ["chr16"] * 7, "pos": list(range(10)),
                  "ref": ["A"] * 10, "alt": ["G"] * 10,
                  "af_abraom": [0.000427] + [0.3] * 9}).to_parquet(abraom, index=False)
    membros = raiz / "membros.parquet"
    pd.DataFrame({"chrom": ["chr1"], "pos_1based": [1], "ref": ["A"], "alt": ["G"]}).to_parquet(
        membros, index=False)
    selecao = raiz / "selecao.parquet"
    pd.DataFrame({"chrom": ["chr2"], "pos_1based": [1], "ref": ["A"], "alt": ["G"]}).to_parquet(
        selecao, index=False)
    snap = raiz / "snap.parquet"
    pd.DataFrame({"chrom": ["chr3"], "pos_1based": [1], "ref": ["A"], "alt": ["G"],
                  "role": ["validation"]}).to_parquet(snap, index=False)
    return abraom, membros, selecao, snap


def _fetch_sintetico(chrom, inicio, fim):
    return [FakeRec(chrom, inicio + 1 + i, "A", ("G",), {"AF_joint": [0.01 * (i + 1)], "AC_joint": [50]})
            for i in range(3)]


def test_end_to_end_publica_e_declara_a_receita():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        abraom, membros, selecao, snap = _entradas(raiz)
        original = gp.fetch_por_cromossomo
        gp.fetch_por_cromossomo = lambda regioes, **kw: _fetch_sintetico  # type: ignore[assignment]
        try:
            rc = gp.main(["--fai", str(raiz / "g.fai"), "--abraom-pool", str(abraom),
                          "--geografia", "casado_ao_abraom", "--n-regioes", "20",
                          "--tamanho-regiao-bp", "1000", "--tbi-dir", str(raiz),
                          "--brazil-variants", str(membros),
                          "--selection", str(selecao), "--snapshot", str(snap),
                          "--out-dir", str(raiz / "out")])
        finally:
            gp.fetch_por_cromossomo = original
        assert rc == 0, rc
        relatorio = json.loads((raiz / "out" / "pool_global.json").read_text(encoding="utf-8"))
        assert relatorio["receita"]["geografia"] == "casado_ao_abraom"
        assert relatorio["receita"]["af_min_aplicado"] is None
        assert relatorio["custo_de_casar_o_piso"]["piso"] == 0.000427, "o custo sai mesmo sem aplicar o piso"
        assert relatorio["pronto_para_amostrar"] is True
        assert "chr8" not in relatorio["pool"]["por_cromossomo"], "chr8 reservado nao recebe regiao"
        assert relatorio["geografia_contra_o_abraom"]["maior_diferenca"]["cromossomo"]
        assert (raiz / "out" / "global_pool.parquet").exists()


def test_end_to_end_sem_exclusoes_nao_publica():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        abraom, _, _, _ = _entradas(raiz)
        original = gp.fetch_por_cromossomo
        gp.fetch_por_cromossomo = lambda regioes, **kw: _fetch_sintetico  # type: ignore[assignment]
        try:
            rc = gp.main(["--fai", str(raiz / "g.fai"), "--abraom-pool", str(abraom),
                          "--geografia", "uniforme", "--n-regioes", "5",
                          "--tamanho-regiao-bp", "1000", "--tbi-dir", str(raiz),
                          "--out-dir", str(raiz / "out")])
        finally:
            gp.fetch_por_cromossomo = original
        assert rc == 2
        relatorio = json.loads((raiz / "out" / "pool_global.json").read_text(encoding="utf-8"))
        assert len(relatorio["pendencias"]) == 3
        assert not (raiz / "out" / "global_pool.parquet").exists()


def test_geografia_e_obrigatoria():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        abraom, _, _, _ = _entradas(raiz)
        try:
            gp.main(["--fai", str(raiz / "g.fai"), "--abraom-pool", str(abraom),
                     "--tbi-dir", str(raiz), "--n-regioes", "5", "--out-dir", str(raiz / "out")])
        except SystemExit as exc:
            assert exc.code == 2
            return
        raise AssertionError("--geografia sem default: tem de exigir a declaracao")


def test_indice_ausente_falha_com_o_comando_para_resolver():
    """O piloto de 20/09 morreu em `Could not retrieve index file`: com URL presignada pysam nao acha o .tbi.

    A falha tem de dizer o que fazer, e nao pode ser um traceback de pysam no meio de 2.000 regioes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            gp.abrir_gnomad("chr1", tbi_dir=Path(tmp))
        except SystemExit as exc:
            assert "aws s3 cp" in str(exc) and ".tbi" in str(exc), str(exc)
            return
        except ImportError as exc:
            raise Skip(f"sem boto3/pysam neste ambiente: {exc}") from exc
        raise AssertionError("sem indice local, abrir_gnomad tinha de parar antes de tocar a rede")


def test_caminho_do_tbi_seque_o_nome_do_vcf():
    caminho = gp.caminho_do_tbi(Path("/x"), "chr7")
    assert caminho.name == "gnomad.joint.v4.1.sites.chr7.vcf.bgz.tbi"


def test_regioes_nos_locos_ficam_centradas_nas_variantes_do_abraom():
    """A terceira geografia: a localizacao deixa de distinguir as fontes."""
    abraom = pd.DataFrame({"chrom": ["chr1"] * 4, "pos": [50_000, 60_000, 70_000, 80_000],
                           "ref": ["A"] * 4, "alt": ["G"] * 4, "af_abraom": [0.1] * 4})
    comprimentos = {"chr1": 1_000_000}
    regioes = gp.regioes_nos_locos_do_abraom(np.random.default_rng(1), abraom, comprimentos,
                                             n_regioes=3, tamanho_bp=2_000)
    assert len(regioes) == 3
    for chrom, inicio, fim in regioes:
        centro = inicio + 1_000
        assert chrom == "chr1" and fim - inicio == 2_000
        assert min(abs(centro - (p - 1)) for p in abraom["pos"]) <= 1, (centro, regioes)


def test_regioes_nos_locos_respeitam_a_borda_do_cromossomo():
    abraom = pd.DataFrame({"chrom": ["chr1"], "pos": [5], "ref": ["A"], "alt": ["G"], "af_abraom": [0.1]})
    regioes = gp.regioes_nos_locos_do_abraom(np.random.default_rng(1), abraom, {"chr1": 10_000},
                                             n_regioes=1, tamanho_bp=2_000)
    chrom, inicio, fim = regioes[0]
    assert inicio == 0 and fim == 2_000, regioes


def test_regioes_nos_locos_ignoram_cromossomo_nao_elegivel():
    """chr8 reservado nao entra nem por essa porta."""
    abraom = pd.DataFrame({"chrom": ["chr8", "chr1"], "pos": [100, 200], "ref": ["A", "A"],
                           "alt": ["G", "G"], "af_abraom": [0.1, 0.1]})
    regioes = gp.regioes_nos_locos_do_abraom(np.random.default_rng(1), abraom, {"chr1": 10_000},
                                             n_regioes=2, tamanho_bp=1_000)
    assert {c for c, _, _ in regioes} == {"chr1"}, regioes


def test_geografia_dos_locos_e_uma_opcao_declarada():
    assert gp.GEOGRAFIA_LOCOS in gp.GEOGRAFIAS
    assert len(gp.GEOGRAFIAS) == 3


def test_sobreposicao_de_alelos_e_medida_no_relatorio():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        abraom, membros, selecao, snap = _entradas(raiz)
        original = gp.fetch_por_cromossomo
        gp.fetch_por_cromossomo = lambda regioes, **kw: _fetch_sintetico  # type: ignore[assignment]
        try:
            rc = gp.main(["--fai", str(raiz / "g.fai"), "--abraom-pool", str(abraom),
                          "--geografia", "uniforme", "--n-regioes", "20",
                          "--tamanho-regiao-bp", "1000", "--tbi-dir", str(raiz),
                          "--brazil-variants", str(membros), "--selection", str(selecao),
                          "--snapshot", str(snap), "--out-dir", str(raiz / "out")])
        finally:
            gp.fetch_por_cromossomo = original
        assert rc == 0
        relatorio = json.loads((raiz / "out" / "pool_global.json").read_text(encoding="utf-8"))
        medida = relatorio["sobreposicao_de_alelos_com_o_abraom"]
        assert "alelos_em_ambas_as_metades" in medida and "fracao_do_pool_global" in medida


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
