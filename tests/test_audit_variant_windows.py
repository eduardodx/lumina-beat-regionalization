"""Prova que a auditoria de janelas separa os tres motivos e trata ref_mismatch como erro, nao estatistica.

O `fetch` e injetado, entao nao precisa de FASTA nem de pysam: roda no Windows e no notebook.
    PYTHONPATH=. python3 tests/test_audit_variant_windows.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.audit_variant_windows as aud  # noqa: E402
from eval.embedding_probe.windows import focal_offset  # noqa: E402

JANELA = 64
FOCAL = focal_offset(JANELA)  # L // 2 - 1, convencao do Mosaic


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _genoma(base_focal: str = "A", *, com_n: bool = False, minusculo: bool = False) -> str:
    """Cromossomo sintetico onde a base em FOCAL e conhecida."""
    seq = ["C"] * (JANELA * 4)
    seq[FOCAL] = base_focal
    if com_n:
        seq[FOCAL + 5] = "N"
    if minusculo:
        seq[FOCAL + 7] = "c"
    return "".join(seq)


def _fetch(genoma: str):
    def fetch(chrom: str, start: int, end: int) -> str:
        if chrom != "chr1":
            return ""
        return genoma[start:end]
    return fetch


def _linhas(rows):
    return pd.DataFrame(rows, columns=["variant_id", "chrom", "pos_1based", "ref", "alt",
                                       "role", "primary_panel", "binary_label"])


def test_janela_valida_conta_como_ok():
    rows = _linhas([("var:1", "chr1", FOCAL + 1, "A", "G", "train", "missense", 1)])
    motivos, falhas = aud.audit_windows(rows, _fetch(_genoma("A")), window_bp=JANELA)
    assert motivos["ok"] == 1 and falhas == [], (motivos, falhas)


def test_soft_mask_nao_descarta():
    """Minuscula e normalizada antes da checagem: soft-mask nao pode custar variante."""
    rows = _linhas([("var:1", "chr1", FOCAL + 1, "A", "G", "train", "missense", 1)])
    motivos, _ = aud.audit_windows(rows, _fetch(_genoma("A", minusculo=True)), window_bp=JANELA)
    assert motivos["ok"] == 1, motivos


def test_n_na_janela_descarta_com_motivo_proprio():
    rows = _linhas([("var:1", "chr1", FOCAL + 1, "A", "G", "train", "splice", 0)])
    motivos, falhas = aud.audit_windows(rows, _fetch(_genoma("A", com_n=True)), window_bp=JANELA)
    assert motivos["non_acgt"] == 1 and motivos["ok"] == 0
    assert falhas[0]["motivo"] == "non_acgt" and falhas[0]["primary_panel"] == "splice"


def test_ref_que_nao_bate_com_o_fasta_vira_ref_mismatch():
    rows = _linhas([("var:1", "chr1", FOCAL + 1, "T", "G", "train", "missense", 1)])
    motivos, falhas = aud.audit_windows(rows, _fetch(_genoma("A")), window_bp=JANELA)
    assert motivos["ref_mismatch"] == 1, motivos
    assert "FASTA=A" in falhas[0]["detalhe"] and "REF=T" in falhas[0]["detalhe"], falhas[0]


def test_janela_que_nao_cabe_e_descartada_sem_deslocar():
    rows = _linhas([("var:1", "chr1", 1, "C", "G", "train", "noncoding", 0)])
    motivos, falhas = aud.audit_windows(rows, _fetch(_genoma("A")), window_bp=JANELA)
    assert motivos["out_of_bounds"] == 1, motivos
    assert falhas[0]["motivo"] == "out_of_bounds"


def test_breakdown_separa_por_papel_e_painel():
    falhas = [
        {"motivo": "non_acgt", "role": "train", "primary_panel": "missense"},
        {"motivo": "non_acgt", "role": "train", "primary_panel": "splice"},
        {"motivo": "out_of_bounds", "role": "validation", "primary_panel": "missense"},
    ]
    assert aud.breakdown(falhas, "role") == {"non_acgt": {"train": 2}, "out_of_bounds": {"validation": 1}}
    assert aud.breakdown(falhas, "primary_panel")["non_acgt"] == {"missense": 1, "splice": 1}


def test_end_to_end_sai_com_codigo_2_em_ref_mismatch(monkeypatch=None):
    genoma = _genoma("A")
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        variantes = raiz / "variantes.parquet"
        _linhas([("var:ok", "chr1", FOCAL + 1, "A", "G", "train", "missense", 1),
                 ("var:ruim", "chr1", FOCAL + 1, "T", "G", "train", "missense", 1)]).to_parquet(
            variantes, index=False)
        original = aud.abrir_fasta
        aud.abrir_fasta = lambda caminho: (_fetch(genoma), "fake")  # type: ignore[assignment]
        try:
            rc = aud.main(["--variants", str(variantes), "--fasta", str(raiz / "nao_existe.fa"),
                           "--window-bp", str(JANELA), "--out-dir", str(raiz / "out")])
        finally:
            aud.abrir_fasta = original
        assert rc == 2
        relatorio = json.loads((raiz / "out" / "auditoria_de_janelas.json").read_text(encoding="utf-8"))
        assert relatorio["por_motivo"] == {"ok": 1, "ref_mismatch": 1}, relatorio["por_motivo"]
        assert (raiz / "out" / "variantes_sem_janela.parquet").exists()


def test_end_to_end_ok_reporta_fracao_descartada():
    genoma = _genoma("A", com_n=True)
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        variantes = raiz / "variantes.parquet"
        _linhas([("var:n", "chr1", FOCAL + 1, "A", "G", "train", "missense", 1)]).to_parquet(
            variantes, index=False)
        original = aud.abrir_fasta
        aud.abrir_fasta = lambda caminho: (_fetch(genoma), "fake")  # type: ignore[assignment]
        try:
            rc = aud.main(["--variants", str(variantes), "--fasta", str(raiz / "x.fa"),
                           "--window-bp", str(JANELA), "--out-dir", str(raiz / "out")])
        finally:
            aud.abrir_fasta = original
        assert rc == 0
        relatorio = json.loads((raiz / "out" / "auditoria_de_janelas.json").read_text(encoding="utf-8"))
        assert relatorio["fracao_descartada"] == 1.0 and relatorio["por_motivo"]["non_acgt"] == 1


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
