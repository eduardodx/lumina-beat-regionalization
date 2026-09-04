"""Testes das janelas da sonda de embedding.

Stdlib puro (sem pytest/torch/pyfaidx) para tambem rodar no Windows local, onde nao ha ambiente
cientifico: ``python tests/test_embedding_probe_windows.py``. O pytest coleta as mesmas funcoes.
"""

from __future__ import annotations

import random
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedding_probe.windows import (  # noqa: E402
    CONSUMER_WINDOWS,
    EXPLORATORY_WINDOWS,
    ProbeWindow,
    WindowError,
    build_window,
    focal_offset,
    matched_focal_index,
    other_bases,
    substituted,
)


@contextmanager
def assert_raises(exc_type, reason: str | None = None):
    try:
        yield
    except exc_type as exc:
        if reason is not None:
            assert getattr(exc, "reason", None) == reason, f"reason={getattr(exc, 'reason', None)!r} != {reason!r}"
        return
    raise AssertionError(f"expected {exc_type.__name__}")


# --------------------------------------------------------------------------------------------
# Genoma sintetico
# --------------------------------------------------------------------------------------------

_RNG = random.Random(20260901)
_GENOME = {"chr1": "".join(_RNG.choice("ACGT") for _ in range(200_000))}


def _fetch(chrom: str, start: int, end: int) -> str:
    seq = _GENOME.get(chrom, "")
    if start < 0:
        return ""
    return seq[start:end]  # curto perto da borda -> build_window detecta pelo comprimento


def _pos_with_ref(base: str, near: int = 100_000) -> int:
    """Menor posicao 1-based >= ``near`` cuja base do genoma sintetico e ``base``."""
    seq = _GENOME["chr1"]
    idx = seq.index(base, near)
    return idx + 1


# --------------------------------------------------------------------------------------------
# Convencao de offset (o contrato com o Mosaic)
# --------------------------------------------------------------------------------------------


def test_focal_offset_matches_mosaic_contract():
    # Valores travados de mosaic.windows.focal_offset (L//2 - 1) / ADR 0005.
    assert focal_offset(32768) == 16383
    assert focal_offset(16384) == 8191
    assert focal_offset(4096) == 2047
    assert focal_offset(2048) == 1023
    assert focal_offset(1024) == 511
    # NAO e L//2 -- a divergencia de 1 base vs eval/clinvar/variant_utils.py e deliberada.
    assert focal_offset(4096) != 4096 // 2


def test_focal_offset_rejects_invalid():
    with assert_raises(ValueError):
        focal_offset(4095)
    with assert_raises(ValueError):
        focal_offset(0)


def test_matched_focal_index_is_smallest_centered_offset():
    assert matched_focal_index(CONSUMER_WINDOWS) == 2047
    assert matched_focal_index(EXPLORATORY_WINDOWS + CONSUMER_WINDOWS) == 511
    with assert_raises(ValueError):
        matched_focal_index(())


# --------------------------------------------------------------------------------------------
# Construcao da janela
# --------------------------------------------------------------------------------------------


def test_centered_window_places_ref_at_focal_and_only_alt_differs():
    pos = _pos_with_ref("A")
    win = build_window(_fetch, chrom="chr1", pos_1based=pos, ref="A", alt="G", window_bp=4096)

    assert isinstance(win, ProbeWindow)
    assert win.layout == "centered"
    assert win.focal_index == 2047
    assert win.window_start == (pos - 1) - 2047
    assert win.window_end == win.window_start + 4096
    assert len(win.ref_seq) == len(win.alt_seq) == 4096
    assert win.ref_seq[2047] == "A"
    assert win.alt_seq[2047] == "G"
    # ref e alt diferem em exatamente uma posicao -- a focal.
    diff = [i for i, (r, a) in enumerate(zip(win.ref_seq, win.alt_seq)) if r != a]
    assert diff == [2047]
    assert win.n_upstream == 2047 and win.n_downstream == 2048


def test_centered_windows_are_nested_substrings_same_anchor():
    """ADR 0005 na NOSSA implementacao: a janela centrada menor e substring da maior."""
    pos = _pos_with_ref("C")
    wins = {
        bp: build_window(_fetch, chrom="chr1", pos_1based=pos, ref="C", alt="T", window_bp=bp)
        for bp in (1024, 2048, 4096, 16384, 32768)
    }
    big = wins[32768]
    for bp, win in wins.items():
        offset = win.window_start - big.window_start
        assert offset >= 0 and offset + bp <= 32768, f"{bp} nao cabe na 32k"
        assert big.ref_seq[offset : offset + bp] == win.ref_seq, f"{bp} nao e substring da 32k"
        # a base focal e a mesma base genomica em todas as janelas
        assert offset + win.focal_index == big.focal_index


def test_matched_layout_holds_focal_index_across_window_sizes():
    pos = _pos_with_ref("G")
    index = matched_focal_index(CONSUMER_WINDOWS)
    wins = {
        bp: build_window(
            _fetch, chrom="chr1", pos_1based=pos, ref="G", alt="A", window_bp=bp, focal_index=index
        )
        for bp in CONSUMER_WINDOWS
    }
    for bp, win in wins.items():
        assert win.layout == "matched"
        assert win.focal_index == index, bp
        assert win.ref_seq[index] == "G"
        # mesmo indice focal => mesma PE senoidal; upstream identico, so o downstream cresce.
        assert win.window_start == (pos - 1) - index
        assert win.n_upstream == index
        assert win.n_downstream == bp - index - 1
    upstreams = {win.ref_seq[:index] for win in wins.values()}
    assert len(upstreams) == 1, "contexto upstream deveria ser identico entre as janelas casadas"


def test_matched_and_centered_differ_for_large_windows():
    """A comparacao que quantifica o artefato de PE: mesma variante, indices focais diferentes."""
    pos = _pos_with_ref("T")
    centered = build_window(_fetch, chrom="chr1", pos_1based=pos, ref="T", alt="A", window_bp=16384)
    matched = build_window(
        _fetch, chrom="chr1", pos_1based=pos, ref="T", alt="A", window_bp=16384, focal_index=2047
    )
    assert centered.focal_index == 8191 and matched.focal_index == 2047
    assert centered.window_start != matched.window_start
    assert centered.ref_seq != matched.ref_seq


# --------------------------------------------------------------------------------------------
# Validacao estrita
# --------------------------------------------------------------------------------------------


def test_ref_mismatch_raises_instead_of_searching_neighbours():
    pos = _pos_with_ref("A")
    # O helper do ClinVar tentaria pos-1/pos+1; aqui tem que falhar alto.
    with assert_raises(WindowError, "ref_mismatch"):
        build_window(_fetch, chrom="chr1", pos_1based=pos, ref="C", alt="G", window_bp=4096)


def test_non_acgt_raises():
    # Base focal valida (A), mas ha N no resto da janela: o Mosaic filtra isso via
    # `sequence_eligible`, e a sonda tem que rejeitar tambem quando a janela sai do envelope
    # validado (caso real do layout `matched`, que se estende alem dos +-16383 conferidos).
    _GENOME["chrN"] = ("N" * 3000) + "A" + ("N" * 3000)
    with assert_raises(WindowError, "non_acgt"):
        build_window(_fetch, chrom="chrN", pos_1based=3001, ref="A", alt="G", window_bp=4096)


def test_out_of_bounds_is_dropped_not_shifted():
    seq = _GENOME["chr1"]

    def alt_for(pos_1based: int) -> tuple[str, str]:
        ref = seq[pos_1based - 1]
        return ref, other_bases(ref)[0]

    ref, alt = alt_for(10)
    with assert_raises(WindowError, "out_of_bounds"):  # start < 0
        build_window(_fetch, chrom="chr1", pos_1based=10, ref=ref, alt=alt, window_bp=4096)
    ref, alt = alt_for(199_000)
    with assert_raises(WindowError, "out_of_bounds"):  # fetch curto no fim do contig
        build_window(_fetch, chrom="chr1", pos_1based=199_000, ref=ref, alt=alt, window_bp=32768)
    with assert_raises(WindowError, "out_of_bounds"):  # contig inexistente -> fetch vazio
        build_window(_fetch, chrom="chrZZ", pos_1based=100_000, ref="A", alt="G", window_bp=4096)


def test_non_snv_raises():
    pos = _pos_with_ref("A")
    for ref, alt in (("AC", "G"), ("A", "AG"), ("A", "A"), ("A", "N")):
        with assert_raises(WindowError, "not_snv"):
            build_window(_fetch, chrom="chr1", pos_1based=pos, ref=ref, alt=alt, window_bp=4096)


def test_focal_index_must_fit_window():
    pos = _pos_with_ref("A")
    with assert_raises(WindowError, "out_of_bounds"):
        build_window(_fetch, chrom="chr1", pos_1based=pos, ref="A", alt="G", window_bp=1024, focal_index=2047)


# --------------------------------------------------------------------------------------------
# Helpers dos controles
# --------------------------------------------------------------------------------------------


def test_substituted_and_other_bases():
    assert substituted("ACGT", 2, "a") == "ACAT"
    with assert_raises(IndexError):
        substituted("ACGT", 4, "A")
    with assert_raises(ValueError):
        substituted("ACGT", 0, "N")
    assert other_bases("A") == ("C", "G", "T")
    assert other_bases("t") == ("A", "C", "G")
    with assert_raises(ValueError):
        other_bases("N")


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
