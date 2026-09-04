"""Janelas ref/alt da sonda de embedding, na convencao do Mosaic.

Constroi as janelas que a sonda passa pelo R03. Diferencas CONSCIENTES vs
``eval/clinvar/variant_utils.py::extract_variant_window`` (que continua servindo o harness do
ClinVar e nao e alterado aqui):

  * **Offset focal = ``L // 2 - 1``** (2047 / 8191 / 16383), espelhando
    ``mosaic.windows.focal_offset`` (lumina-mosaic, ``src/mosaic/windows.py``). O helper do
    ClinVar usa ``L // 2`` -- uma base de diferenca. Adotamos a do Mosaic porque e a convencao
    com que o release calculou ``sequence_eligible``; usar outra invalidaria essa garantia.
  * **Validacao ESTRITA, sem fallback.** O ``_resolve_variant_pos`` do ClinVar procura o REF em
    ``pos-1 +- 1`` para tolerar a ambiguidade de ancora do VCF em indels. O release do Mosaic so
    tem SNV com REF ja conferido contra o GRCh38.p14, entao qualquer divergencia aqui e um erro
    real (FASTA errado, build errado, linha corrompida) e deve falhar alto, nao ser silenciada.
  * **Sem ``boundary_shifted``.** Uma janela que nao cabe e descartada, nao deslocada: deslocar
    tiraria a variante do indice focal declarado e quebraria a comparacao entre janelas.
  * **Layout ``matched``** (novo): permite fixar o indice focal em janelas de tamanhos diferentes.
    Necessario porque a PE do R03 e senoidal -- em janelas centradas o indice focal muda com L
    (2047 / 8191 / 16383) e ``h_pure = Linear(token_emb + pos_emb)`` muda por causa do INDICE, nao
    do contexto. Com o indice casado a PE focal e identica e a unica variavel e o contexto extra.

Modulo em stdlib puro (o ``fetch`` e injetado) para ser testavel sem pyfaidx/torch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Contrato do release Mosaic v1 (mosaic.windows.CONSUMER_WINDOWS, ADR 0005). `sequence_eligible`
# valida a janela centrada de 32768 bp => as janelas centradas de 16384 e 4096 da mesma ancora sao
# substrings validas dela.
CONSUMER_WINDOWS: tuple[int, ...] = (4096, 16384, 32768)

# Fora do contrato do release: pontos extras da curva de contexto/diluicao (o Eduardo levantou
# "de 1024 pra 2048 pode ser diluido"). Sao substrings centradas da janela de 32k validada, logo
# herdam a garantia de ACGT; ficam marcadas como exploratorias no relatorio. Seguras para
# ``model.encode()``, que nao instancia a head de Hi-C (essa exigiria L >= 2000 para ter 1 bin).
EXPLORATORY_WINDOWS: tuple[int, ...] = (1024, 2048)

BASES = frozenset("ACGT")

Fetcher = Callable[[str, int, int], str]
"""``fetch(chrom, start0, end0) -> str`` -- 0-based, half-open, como pyfaidx/pysam."""


class WindowError(ValueError):
    """Janela invalida. ``reason`` e um codigo estavel para agregacao."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}{': ' + detail if detail else ''}")
        self.reason = reason
        self.detail = detail


def focal_offset(window_bp: int) -> int:
    """Indice 0-based da base focal numa janela centrada.

    Espelha ``mosaic.windows.focal_offset``: ``L // 2 - 1``. Reimplementado (em vez de importar
    ``mosaic``) porque ``mosaic/windows.py`` importa ``pysam`` no topo, que nao e dependencia deste
    repo; ``tests/test_embedding_probe_windows.py`` trava os valores (2047 / 8191 / 16383).
    """
    if window_bp <= 0:
        raise ValueError(f"window_bp must be positive, got {window_bp}")
    if window_bp % 2:
        raise ValueError(f"window_bp must be even, got {window_bp}")
    return window_bp // 2 - 1


def matched_focal_index(window_sizes: tuple[int, ...]) -> int:
    """Indice focal comum a todas as janelas de ``window_sizes`` (layout ``matched``).

    E o offset centrado da MENOR janela do conjunto -- o maior indice que ainda cabe em todas.
    Para o contrato (4096, 16384, 32768) da 2047: a menor janela fica centrada e as maiores
    crescem so para a direita, isolando "contexto extra" da mudanca de indice da PE.
    """
    if not window_sizes:
        raise ValueError("window_sizes must not be empty")
    return focal_offset(min(window_sizes))


@dataclass(frozen=True)
class ProbeWindow:
    """Par ref/alt de uma variante, com toda a proveniencia para auditoria."""

    chrom: str
    pos_1based: int
    ref: str
    alt: str
    window_bp: int
    focal_index: int
    layout: str  # "centered" | "matched"
    window_start: int  # 0-based, inclusivo
    ref_seq: str
    alt_seq: str

    @property
    def window_end(self) -> int:
        """0-based, exclusivo."""
        return self.window_start + self.window_bp

    @property
    def n_upstream(self) -> int:
        return self.focal_index

    @property
    def n_downstream(self) -> int:
        return self.window_bp - self.focal_index - 1


def _fetch_validated(fetch: Fetcher, chrom: str, start: int, window_bp: int) -> str:
    if start < 0:
        raise WindowError("out_of_bounds", f"start={start} < 0")
    seq = fetch(chrom, start, start + window_bp).upper()
    if len(seq) != window_bp:
        raise WindowError("out_of_bounds", f"fetched {len(seq)} != {window_bp} bp")
    bad = {base for base in seq if base not in BASES}
    if bad:
        raise WindowError("non_acgt", f"bases {sorted(bad)}")
    return seq


def build_window(
    fetch: Fetcher,
    *,
    chrom: str,
    pos_1based: int,
    ref: str,
    alt: str,
    window_bp: int,
    focal_index: int | None = None,
) -> ProbeWindow:
    """Constroi o par ref/alt de uma SNV. ``focal_index=None`` => janela centrada.

    Levanta ``WindowError`` (nunca silencia): ``out_of_bounds``, ``non_acgt``, ``ref_mismatch``,
    ``not_snv``.
    """
    ref_u, alt_u = ref.upper(), alt.upper()
    if len(ref_u) != 1 or len(alt_u) != 1 or ref_u not in BASES or alt_u not in BASES:
        raise WindowError("not_snv", f"ref={ref!r} alt={alt!r}")
    if ref_u == alt_u:
        raise WindowError("not_snv", "ref == alt")
    if pos_1based < 1:
        raise WindowError("out_of_bounds", f"pos_1based={pos_1based}")

    layout = "centered" if focal_index is None else "matched"
    index = focal_offset(window_bp) if focal_index is None else focal_index
    if not 0 <= index < window_bp:
        raise WindowError("out_of_bounds", f"focal_index={index} outside [0,{window_bp})")

    start = (pos_1based - 1) - index
    seq = _fetch_validated(fetch, chrom, start, window_bp)
    if seq[index] != ref_u:
        raise WindowError("ref_mismatch", f"{chrom}:{pos_1based} FASTA={seq[index]} REF={ref_u}")

    return ProbeWindow(
        chrom=chrom,
        pos_1based=pos_1based,
        ref=ref_u,
        alt=alt_u,
        window_bp=window_bp,
        focal_index=index,
        layout=layout,
        window_start=start,
        ref_seq=seq,
        alt_seq=substituted(seq, index, alt_u),
    )


def substituted(seq: str, index: int, base: str) -> str:
    """``seq`` com a base em ``index`` trocada. Base do controle de vizinhanca e de base aleatoria."""
    if not 0 <= index < len(seq):
        raise IndexError(f"index {index} outside [0,{len(seq)})")
    base_u = base.upper()
    if base_u not in BASES:
        raise ValueError(f"base must be one of ACGT, got {base!r}")
    return seq[:index] + base_u + seq[index + 1 :]


def other_bases(base: str) -> tuple[str, ...]:
    """As tres bases != ``base``, em ordem estavel. Usado no controle de troca aleatoria."""
    base_u = base.upper()
    if base_u not in BASES:
        raise ValueError(f"base must be one of ACGT, got {base!r}")
    return tuple(b for b in "ACGT" if b != base_u)
