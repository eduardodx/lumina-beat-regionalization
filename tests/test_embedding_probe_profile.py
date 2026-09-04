"""Testes dos bins do perfil espacial. Stdlib puro: ``python tests/test_embedding_probe_profile.py``."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedding_probe.profile import (  # noqa: E402
    CONV_HORIZON_BP,
    LOCAL_ATTENTION_RADIUS_BP,
    horizon_of,
    magnitude_ranges,
    profile_bins,
)
from eval.embedding_probe.windows import (  # noqa: E402
    CONSUMER_WINDOWS,
    EXPLORATORY_WINDOWS,
    focal_offset,
    matched_focal_index,
)

ALL_WINDOWS = tuple(sorted(EXPLORATORY_WINDOWS + CONSUMER_WINDOWS))


@contextmanager
def assert_raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _all_layouts():
    """As 8 combinacoes (janela, layout) que a Fase B extrai."""
    for bp in ALL_WINDOWS:
        yield bp, focal_offset(bp), "centered"
    matched = matched_focal_index(CONSUMER_WINDOWS)
    for bp in CONSUMER_WINDOWS:
        yield bp, matched, "matched"


# --------------------------------------------------------------------------------------------
# magnitude_ranges
# --------------------------------------------------------------------------------------------


def test_magnitude_ranges_individual_up_to_conv_horizon():
    ranges = magnitude_ranges(100)
    individual = [r for r in ranges if r[0] == r[1]]
    assert [r[0] for r in individual] == list(range(1, CONV_HORIZON_BP + 1))
    # o primeiro bin agregado comeca logo depois do horizonte convolucional
    aggregated = [r for r in ranges if r[0] != r[1]]
    assert aggregated[0][0] == CONV_HORIZON_BP + 1


def test_magnitude_ranges_are_contiguous_and_cover_everything():
    for reach in (1, 7, 25, 26, 100, 511, 512, 513, 2047, 16383, 30720):
        ranges = magnitude_ranges(reach)
        assert ranges[0][0] == 1, reach
        assert ranges[-1][1] == reach, reach
        for (_, hi), (lo_next, _) in zip(ranges, ranges[1:]):
            assert lo_next == hi + 1, f"buraco/sobreposicao em reach={reach}: {hi} -> {lo_next}"


def test_magnitude_ranges_put_landmarks_on_boundaries():
    """25 (conv) e 512 (atencao local) tem que ser FIM de bin, nunca meio."""
    ranges = magnitude_ranges(30_720)
    upper = {hi for _, hi in ranges}
    assert CONV_HORIZON_BP in upper
    assert LOCAL_ATTENTION_RADIUS_BP in upper


def test_magnitude_ranges_empty_when_no_reach():
    assert magnitude_ranges(0) == []
    assert magnitude_ranges(-1) == []


# --------------------------------------------------------------------------------------------
# profile_bins -- particao da janela
# --------------------------------------------------------------------------------------------


def test_profile_bins_partition_every_real_layout():
    """A garantia central: disjuntos, dentro da janela, e cobrindo-a exatamente."""
    for window_bp, focal, layout in _all_layouts():
        bins = profile_bins(focal_index=focal, window_bp=window_bp)
        covered: list[int] = []
        for b in bins:
            assert 0 <= b.start < b.end <= window_bp, f"{layout}/{window_bp} bin {b.label} fora"
            covered.extend(range(b.start, b.end))
        assert len(covered) == len(set(covered)), f"{layout}/{window_bp}: bins se sobrepoem"
        assert sorted(covered) == list(range(window_bp)), f"{layout}/{window_bp}: nao cobre a janela"


def test_profile_bins_focal_is_exactly_one_base():
    for window_bp, focal, _ in _all_layouts():
        bins = profile_bins(focal_index=focal, window_bp=window_bp)
        focal_bins = [b for b in bins if b.direction == "focal"]
        assert len(focal_bins) == 1
        assert focal_bins[0].width == 1
        assert focal_bins[0].start == focal


def test_profile_bins_offsets_match_indices():
    """Cada bin cobre exatamente os indices cujo |offset| cai na faixa declarada."""
    for window_bp, focal, layout in _all_layouts():
        for b in profile_bins(focal_index=focal, window_bp=window_bp):
            for index in range(b.start, b.end):
                offset = index - focal
                if b.direction == "focal":
                    assert offset == 0
                elif b.direction == "up":
                    assert -b.hi <= offset <= -b.lo, f"{layout}/{window_bp} {b.label} idx {index}"
                else:
                    assert b.lo <= offset <= b.hi, f"{layout}/{window_bp} {b.label} idx {index}"


def test_centered_layout_is_nearly_symmetric():
    """Centrado: focal em L//2-1 => upstream tem 1 base a menos que downstream."""
    bins = profile_bins(focal_index=focal_offset(4096), window_bp=4096)
    up = sum(b.width for b in bins if b.direction == "up")
    down = sum(b.width for b in bins if b.direction == "down")
    assert up == 2047 and down == 2048


def test_matched_layout_is_asymmetric_as_designed():
    """Casado: focal fixo em 2047 => a 32k so cresce para a direita. Por isso up/down separados."""
    bins = profile_bins(focal_index=2047, window_bp=32768)
    up = sum(b.width for b in bins if b.direction == "up")
    down = sum(b.width for b in bins if b.direction == "down")
    assert up == 2047
    assert down == 30720
    assert max(b.hi for b in bins if b.direction == "down") == 30720
    assert max(b.hi for b in bins if b.direction == "up") == 2047


def test_profile_bins_rejects_invalid():
    with assert_raises(ValueError):
        profile_bins(focal_index=0, window_bp=0)
    with assert_raises(ValueError):
        profile_bins(focal_index=4096, window_bp=4096)
    with assert_raises(ValueError):
        profile_bins(focal_index=-1, window_bp=4096)


def test_smallest_window_still_reaches_past_conv_horizon():
    """Mesmo a janela de 1024 tem que ir alem dos +-25 bp, senao E3 nao mede nada."""
    bins = profile_bins(focal_index=focal_offset(1024), window_bp=1024)
    assert max(b.hi for b in bins if b.direction != "focal") > CONV_HORIZON_BP


# --------------------------------------------------------------------------------------------
# horizon_of
# --------------------------------------------------------------------------------------------


def test_horizon_of_labels_architectural_regimes():
    assert horizon_of(0) == "focal"
    assert horizon_of(1) == "conv"
    assert horizon_of(CONV_HORIZON_BP) == "conv"
    assert horizon_of(CONV_HORIZON_BP + 1) == "local_attention"
    assert horizon_of(LOCAL_ATTENTION_RADIUS_BP) == "local_attention"
    assert horizon_of(LOCAL_ATTENTION_RADIUS_BP + 1) == "mamba_or_global"
    assert horizon_of(16384) == "mamba_or_global"


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
