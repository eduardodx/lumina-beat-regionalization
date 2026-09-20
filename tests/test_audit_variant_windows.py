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


def _linhas_de_plano(rows):
    """Vocabulario do plano do adapter: focal sorteado, inicio derivado e spans."""
    return pd.DataFrame(rows, columns=["variant_id", "chrom", "pos_1based", "ref", "alt",
                                       "focal_index", "window_start", "spans"])


def test_janela_centrada_passa_mas_a_planejada_falha():
    """O DEFEITO relatado: sem usar o focal do plano, o auditor confere outro trecho do cromossomo.

    Montamos um genoma onde a janela centrada e limpa e a planejada cobre um `N`. Auditar a centrada diria `ok`
    para uma janela que o treinador nao conseguiria construir.
    """
    focal_planejado = 10
    pos = FOCAL + 1  # a mesma variante das outras provas: na janela centrada ela cai em FOCAL
    genoma = list("C" * (JANELA * 4))
    genoma[FOCAL] = "A"
    # O N fica fora da janela centrada [0, 64) e dentro da planejada [pos-1-10, pos-1-10+64).
    inicio_planejado = pos - 1 - focal_planejado
    n_em = inicio_planejado + JANELA - 1
    assert n_em >= JANELA, "o N tem de cair fora da janela centrada, senao o teste nao separa os dois casos"
    genoma[n_em] = "N"
    fetch = _fetch("".join(genoma))

    centrada = _linhas([("var:1", "chr1", pos, "A", "G", "train", "missense", 1)])
    motivos, _ = aud.audit_windows(centrada, fetch, window_bp=JANELA)
    assert motivos["ok"] == 1, ("a janela centrada era para passar", motivos)

    planejada = _linhas_de_plano([("var:1", "chr1", pos, "A", "G", focal_planejado, inicio_planejado,
                                   json.dumps([[focal_planejado, focal_planejado + 3, "variante"]]))])
    motivos, falhas = aud.audit_windows(planejada, fetch, window_bp=JANELA)
    assert motivos["non_acgt"] == 1 and motivos["ok"] == 0, ("a planejada tinha de reprovar", motivos)
    assert falhas[0]["focal_index"] == focal_planejado


def test_window_start_que_nao_bate_com_o_focal_e_erro():
    pos = FOCAL + 1
    focal = 10
    linhas = _linhas_de_plano([("var:1", "chr1", pos, "A", "G", focal, pos - 1 - focal + 3,
                                json.dumps([[focal, focal + 2, "variante"]]))])
    motivos, falhas = aud.audit_windows(linhas, _fetch(_genoma("A")), window_bp=JANELA)
    assert motivos[aud.MOTIVO_INICIO_INCOERENTE] == 1, motivos
    assert "pos-1-focal" in falhas[0]["detalhe"]


def test_span_de_referencia_cobrindo_o_focal_e_erro():
    """Se o span que cobre o focal for de referencia, o alvo do MLM deixa de ser a mutacao."""
    pos = FOCAL + 1
    focal = FOCAL
    linhas = _linhas_de_plano([("var:1", "chr1", pos, "A", "G", focal, pos - 1 - focal,
                                json.dumps([[focal - 1, focal + 2, "referencia"]]))])
    motivos, _ = aud.audit_windows(linhas, _fetch(_genoma("A")), window_bp=JANELA)
    assert motivos[aud.MOTIVO_SPAN_FOCAL] == 1, motivos


def test_span_fora_da_janela_e_erro():
    pos = FOCAL + 1
    focal = FOCAL
    linhas = _linhas_de_plano([("var:1", "chr1", pos, "A", "G", focal, pos - 1 - focal,
                                json.dumps([[focal, focal + 2, "variante"], [JANELA - 1, JANELA + 5, "referencia"]]))])
    motivos, _ = aud.audit_windows(linhas, _fetch(_genoma("A")), window_bp=JANELA)
    assert motivos[aud.MOTIVO_SPAN_FORA] == 1, motivos


def test_plano_coerente_passa():
    pos = FOCAL + 1
    focal = FOCAL
    linhas = _linhas_de_plano([("var:1", "chr1", pos, "A", "G", focal, pos - 1 - focal,
                                json.dumps([[focal, focal + 2, "variante"], [2, 5, "referencia"]]))])
    motivos, falhas = aud.audit_windows(linhas, _fetch(_genoma("A")), window_bp=JANELA)
    assert motivos["ok"] == 1 and falhas == [], (motivos, falhas)


def test_layout_declarado_no_relatorio():
    """Sem dizer QUAL janela foi auditada, um `ok` no JSON e ambiguo."""
    sem_plano = _linhas([("var:1", "chr1", FOCAL + 1, "A", "G", "train", "missense", 1)])
    assert aud.descrever_layout(sem_plano, window_bp=JANELA)["layout"] == "centrado"
    com_plano = _linhas_de_plano([("var:1", "chr1", FOCAL + 1, "A", "G", 10, FOCAL - 10, "[]")])
    descricao = aud.descrever_layout(com_plano, window_bp=JANELA)
    assert descricao["layout"] == "declarado pelo plano"
    assert descricao["focal_index"]["min"] == 10 and descricao["window_start_conferido"] is True


def test_end_to_end_reprova_plano_incoerente():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        pos, focal = FOCAL + 1, 10
        _linhas_de_plano([("var:1", "chr1", pos, "A", "G", focal, 0,
                           json.dumps([[focal, focal + 2, "variante"]]))]).to_parquet(
            raiz / "plano.parquet", index=False)
        original = aud.abrir_fasta
        aud.abrir_fasta = lambda caminho: (_fetch(_genoma("A")), "fake")  # type: ignore[assignment]
        try:
            rc = aud.main(["--variants", str(raiz / "plano.parquet"), "--fasta", str(raiz / "x.fa"),
                           "--window-bp", str(JANELA), "--out-dir", str(raiz / "out")])
        finally:
            aud.abrir_fasta = original
        assert rc == 2, rc
        relatorio = json.loads((raiz / "out" / "auditoria_de_janelas.json").read_text(encoding="utf-8"))
        assert relatorio["janela_auditada"]["layout"] == "declarado pelo plano"


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
