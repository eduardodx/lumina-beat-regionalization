"""Prova que o localizador do ABraOM decide por sha256, nao por nome parecido.

Varios artefatos da campanha v10 parecem ABraOM e nao sao a fonte do source-lock.
    PYTHONPATH=. python3 tests/test_locate_abraom_source.py
"""

from __future__ import annotations

import gzip
import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.locate_abraom_source as loc  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


LS = """2026-06-01 10:00:00  123456789 benchmarks/mosaic/data/raw/abraom/SABE1171.Abraom.clean.tsv
2026-06-01 10:00:00       4096 benchmarks/mosaic/data/raw/abraom/README.md
2026-06-01 10:00:00  987654321 benchmarks/mosaic/data/raw/clinvar/clinvar_20260606.vcf.gz
"""


def test_parse_ls_monta_uri_completa():
    itens = loc.parse_ls(LS, "s3://ai4bio-lumina/benchmarks/mosaic/data/raw/")
    assert ("s3://ai4bio-lumina/benchmarks/mosaic/data/raw/abraom/SABE1171.Abraom.clean.tsv",
            123456789) in itens
    assert len(itens) == 3


def test_filtra_por_pista_e_extensao():
    assert loc.parece_abraom("s3://b/abraom/SABE1171.Abraom.clean.tsv")
    assert loc.parece_abraom("s3://b/x/sabe1171.txt.gz")
    assert not loc.parece_abraom("s3://b/abraom/README.md"), "extensao errada"
    assert not loc.parece_abraom("s3://b/clinvar/clinvar_20260606.vcf.gz"), "sem pista de abraom"


def test_reconhece_derivados_da_campanha_antiga():
    assert loc.e_derivado_conhecido("s3://b/x/clinvar_regional_abraom_master.parquet")
    assert loc.e_derivado_conhecido("s3://b/lumina-ssm/data/datasets/abraom_frequency_adapter/parte.parquet")
    assert loc.e_derivado_conhecido("s3://b/raw/abraom/SABE1171.Abraom.clean.tsv") is None


def test_confere_bate_pelo_conteudo_e_nao_pelo_nome():
    with tempfile.TemporaryDirectory() as tmp:
        arquivo = Path(tmp) / "qualquer_nome.tsv"
        arquivo.write_bytes(b"chrom\tpos\tref\talt\taf_abraom\n")
        esperado = hashlib.sha256(arquivo.read_bytes()).hexdigest()
        bate, calculados = loc.confere(arquivo, esperado)
        assert bate and calculados["arquivo"] == esperado
        assert loc.confere(arquivo, "0" * 64)[0] is False


def test_gz_confere_tambem_o_conteudo_descomprimido():
    """O hash declarado pode ser do TSV, nao do .gz que o guarda."""
    with tempfile.TemporaryDirectory() as tmp:
        conteudo = b"chrom\tpos\tref\talt\taf_abraom\nchr1\t100\tA\tG\t0.01\n"
        arquivo = Path(tmp) / "abraom.tsv.gz"
        with gzip.open(arquivo, "wb") as handle:
            handle.write(conteudo)
        esperado_descomprimido = hashlib.sha256(conteudo).hexdigest()
        bate, calculados = loc.confere(arquivo, esperado_descomprimido)
        assert bate, calculados
        assert calculados["arquivo"] != esperado_descomprimido, "o hash do .gz nao e o do conteudo"


def test_arvore_monta_uri_por_path_declarado():
    """`<root>/<path>` do sources.yaml: e assim que o G0 confere a arvore inteira de fontes."""
    try:
        import yaml  # noqa: F401
    except ImportError as exc:
        raise Skip(f"pyyaml indisponivel: {exc}") from exc

    import tempfile as _tmp
    with _tmp.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        (raiz / "config").mkdir()
        conteudo_yaml = (
            "sources:\n"
            "  abraom_sabe1171:\n"
            "    path: abraom/SABE1171.Abraom.clean.tsv\n"
            f"    sha256: {'a' * 64}\n"
            "  clinvar_submission_summary_2026-06:\n"
            "    path: clinvar/2026-06/submission_summary_2026-06.txt.gz\n"
            f"    sha256: {'b' * 64}\n"
            "  sem_hash:\n"
            "    path: x/y.tsv\n"
            "    sha256: pending\n"
        )
        (raiz / "config" / "sources.yaml").write_text(conteudo_yaml, encoding="utf-8")
        fontes = loc.todas_as_fontes(raiz)
        assert set(fontes) == {"abraom_sabe1171", "clinvar_submission_summary_2026-06"}, sorted(fontes)
        assert "sem_hash" not in fontes, "fonte sem sha256 fixado nao entra na conferencia"

        original = loc.existe_no_s3
        loc.existe_no_s3 = lambda uri: 123 if "abraom" in uri else None
        try:
            linhas = loc.verificar_arvore(raiz, "s3://balde/lumina/lumina-mosaic/")
        finally:
            loc.existe_no_s3 = original
        por_fonte = {linha["fonte"]: linha for linha in linhas}
        assert por_fonte["abraom_sabe1171"]["uri"] == (
            "s3://balde/lumina/lumina-mosaic/abraom/SABE1171.Abraom.clean.tsv")
        assert por_fonte["abraom_sabe1171"]["existe"] is True
        assert por_fonte["clinvar_submission_summary_2026-06"]["existe"] is False


def test_hash_esperado_vem_do_sources_yaml_do_mosaic():
    """Nunca digitado: se o Mosaic nao declarar, o script para."""
    try:
        import yaml  # noqa: F401
    except ImportError as exc:
        raise Skip(f"pyyaml indisponivel: {exc}") from exc

    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        (raiz / "config").mkdir()
        (raiz / "config" / "sources.yaml").write_text(
            "sources:\n"
            "  abraom_sabe1171:\n"
            "    path: abraom/SABE1171.Abraom.clean.tsv\n"
            "    sha256: " + "a" * 64 + "\n"
            "    schema: [chrom, pos, ref, alt, af_abraom]\n",
            encoding="utf-8")
        spec = loc.expected_source(raiz)
        assert spec["sha256"] == "a" * 64
        assert spec["path"] == "abraom/SABE1171.Abraom.clean.tsv"
        assert spec["schema"] == ["chrom", "pos", "ref", "alt", "af_abraom"]

        (raiz / "config" / "sources.yaml").write_text(
            "sources:\n  abraom_sabe1171:\n    path: x\n    sha256: pending\n", encoding="utf-8")
        try:
            loc.expected_source(raiz)
        except SystemExit:
            return
        raise AssertionError("sha256 'pending' tinha de parar o script")


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
