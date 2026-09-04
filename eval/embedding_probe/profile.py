"""Bins do perfil de propagacao espacial ||Delta(q)|| em torno da posicao focal.

O perfil e o experimento E3 ("ate que ponto eu consigo pegar essa mutacao"). Ele so significa
alguma coisa contra os horizontes da arquitetura do R03, entao os bins sao escolhidos para que
esses horizontes caiam em FRONTEIRAS de bin, nao no meio:

  * **+-25 bp -- horizonte da via puramente convolucional.** Composicao dos indices de
    ``MultiKernelStem`` (depthwise k=15, p=7) -> 2x ``DownStage`` (conv k=4 s=2 p=1 + refine k=3)
    -> 2x ``UpStage`` (ConvTranspose k=4 s=2 p=1 + refine k=3) mais os skips. Alem de +-25 bp,
    qualquer mudanca passou OBRIGATORIAMENTE pelo mid-stack (Mamba bidirecional / atencao) --
    e e isso que distingue "o modelo integrou a variante" de "existe uma convolucao".
  * **+-512 bp -- alcance da atencao local.** ``local_attention_window_mid_tokens=256`` =>
    ``radius = 256 // 2 = 128`` mid-tokens; cada mid-token e 4 bp (downsample 4x) => 512 bp.
    FIXO: nao muda com o tamanho da janela. Alem disso so restam o estado recorrente do Mamba e
    as ancoras da atencao global (mean-pool de 16 mid-tokens = 64 bp, ``stride=16``).

Offsets 0..25 ficam em bins individuais (resolucao maxima onde a conv atua); dai para fora os
bins dobram. Upstream e downstream sao SEPARADOS: o layout ``matched`` e assimetrico por
construcao (2047 bp upstream, ate 30720 downstream), entao um bin por |offset| misturaria os
dois lados; e mesmo no centrado, saber se a propagacao e simetrica e informativo (o Mamba e
bidirecional, mas nao ha garantia de simetria).

Stdlib puro: os indices de janela sao explicitos e testados em
``tests/test_embedding_probe_profile.py``, para o redutor em torch ser uma soma trivial.
"""

from __future__ import annotations

from typing import NamedTuple

# Horizontes da arquitetura, para anotar o grafico e interpretar o perfil.
CONV_HORIZON_BP = 25
LOCAL_ATTENTION_RADIUS_BP = 512
MEAN_POOL_ANCHOR_BP = 64

# Fronteiras dos bins agregados (limite superior, inclusivo). 25 e 512 aparecem como fronteira
# de proposito -- ver o docstring.
_OCTAVES: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# Ate onde cada offset tem bin proprio.
_INDIVIDUAL_MAX = CONV_HORIZON_BP


class ProfileBin(NamedTuple):
    """Uma faixa de posicoes da janela, em offsets relativos a base focal."""

    label: str
    direction: str  # "focal" | "up" | "down"
    lo: int  # menor |offset| coberto (inclusivo)
    hi: int  # maior |offset| coberto (inclusivo)
    start: int  # indice inicial na janela (inclusivo)
    end: int  # indice final na janela (exclusivo) -- pronto para slice/narrow

    @property
    def width(self) -> int:
        return self.end - self.start


def magnitude_ranges(max_offset: int) -> list[tuple[int, int]]:
    """Faixas de |offset| em 1..max_offset: individuais ate 25, depois oitavas."""
    if max_offset < 1:
        return []
    ranges = [(d, d) for d in range(1, min(_INDIVIDUAL_MAX, max_offset) + 1)]
    lo = _INDIVIDUAL_MAX + 1
    for hi in _OCTAVES:
        if lo > max_offset:
            break
        if hi < lo:
            continue
        ranges.append((lo, min(hi, max_offset)))
        lo = hi + 1
    return ranges


def profile_bins(*, focal_index: int, window_bp: int) -> list[ProfileBin]:
    """Bins que particionam a janela inteira: focal + upstream + downstream.

    Garantias (travadas por teste): os bins sao disjuntos, ficam dentro de ``[0, window_bp)`` e
    a uniao cobre exatamente a janela -- nenhuma base entra duas vezes nem fica de fora.
    """
    if window_bp <= 0:
        raise ValueError(f"window_bp must be positive, got {window_bp}")
    if not 0 <= focal_index < window_bp:
        raise ValueError(f"focal_index {focal_index} outside [0,{window_bp})")

    bins = [ProfileBin("focal", "focal", 0, 0, focal_index, focal_index + 1)]
    for direction, reach in (("up", focal_index), ("down", window_bp - focal_index - 1)):
        for lo, hi in magnitude_ranges(reach):
            if direction == "up":
                start, end = focal_index - hi, focal_index - lo + 1
            else:
                start, end = focal_index + lo, focal_index + hi + 1
            bins.append(ProfileBin(f"{direction}_{lo}_{hi}", direction, lo, hi, start, end))
    return bins


def horizon_of(offset: int) -> str:
    """Rotula um |offset| pela via arquitetural que pode te-lo alcancado. Para agrupar no relato."""
    if offset == 0:
        return "focal"
    if offset <= CONV_HORIZON_BP:
        return "conv"  # explicavel por convolucao pura -- nao prova integracao
    if offset <= LOCAL_ATTENTION_RADIUS_BP:
        return "local_attention"  # conv nao alcanca; atencao local ou Mamba
    return "mamba_or_global"  # so estado recorrente do Mamba ou ancoras mean-pool diluidas
