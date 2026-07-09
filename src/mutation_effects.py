from __future__ import annotations

from dataclasses import dataclass

from src.constants import (
    AA_CHANGE_MISSENSE,
    AA_CHANGE_NON_CODING,
    AA_CHANGE_NONSENSE,
    AA_CHANGE_SYNONYMOUS,
    AA_NON_CDS,
    AA_STOP,
    CODON_TO_AA,
    MUTATION_EFFECT_MISSENSE,
    MUTATION_EFFECT_STOP,
    MUTATION_EFFECT_SYNONYMOUS,
)

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

_BLOSUM62_MATRIX: tuple[tuple[int, ...], ...] = (
    (4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0),
    (-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3),
    (-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3),
    (-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3),
    (0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1),
    (-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2),
    (-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2),
    (0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3),
    (-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3),
    (-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3),
    (-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1),
    (-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2),
    (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1),
    (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1),
    (-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2),
    (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2),
    (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0),
    (-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3),
    (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1),
    (0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4),
)
_AA_IDX_TO_BLOSUM = {i + 1: i for i in range(20)}


def reverse_complement_dna(seq: str) -> str:
    return "".join(_COMPLEMENT.get(base, "N") for base in reversed(seq.upper()))


def translate_codon(codon: str) -> int:
    return CODON_TO_AA.get(codon.upper(), AA_NON_CDS)


def blosum62_score(ref_aa: int, alt_aa: int) -> float:
    ri = _AA_IDX_TO_BLOSUM.get(ref_aa)
    ai = _AA_IDX_TO_BLOSUM.get(alt_aa)
    if ri is None or ai is None:
        return 0.0
    return float(_BLOSUM62_MATRIX[ri][ai])


def blosum62_substitution_severity(ref_aa: int, alt_aa: int) -> float:
    """Return a [0, 1] severity proxy where lower BLOSUM scores are worse."""
    if ref_aa == alt_aa:
        return 0.0
    score = blosum62_score(ref_aa, alt_aa)
    return max(0.0, min(1.0, (4.0 - score) / 8.0))


def classify_aa_change(ref_aa: int, alt_aa: int) -> int:
    if ref_aa == AA_NON_CDS or alt_aa == AA_NON_CDS:
        return AA_CHANGE_NON_CODING
    if ref_aa == alt_aa:
        return AA_CHANGE_SYNONYMOUS
    if alt_aa == AA_STOP or ref_aa == AA_STOP:
        return AA_CHANGE_NONSENSE
    return AA_CHANGE_MISSENSE


def classify_mutation_effect(ref_aa: int, alt_aa: int) -> int | None:
    if ref_aa == AA_NON_CDS or alt_aa == AA_NON_CDS:
        return None
    if ref_aa == alt_aa:
        return MUTATION_EFFECT_SYNONYMOUS
    if alt_aa == AA_STOP or ref_aa == AA_STOP:
        return MUTATION_EFFECT_STOP
    return MUTATION_EFFECT_MISSENSE


def codon_position_in_cds(
    genomic_pos: int,
    cds_start: int,
    frame: int,
    *,
    cds_end: int | None = None,
    strand: str = "+",
) -> int:
    if strand == "+":
        distance = genomic_pos - cds_start
    elif strand == "-":
        if cds_end is None:
            raise ValueError("cds_end is required when computing minus-strand codon positions.")
        distance = cds_end - 1 - genomic_pos
    else:
        raise ValueError(f"Unsupported strand {strand!r}. Expected '+' or '-'.")
    return (distance + frame) % 3


def extract_codon_plus(ref_seq: str, variant_offset: int, codon_pos: int) -> str | None:
    codon_start = variant_offset - codon_pos
    if codon_start < 0 or codon_start + 3 > len(ref_seq):
        return None
    codon = ref_seq[codon_start : codon_start + 3].upper()
    if "N" in codon:
        return None
    return codon


def extract_codon_minus(ref_seq: str, variant_offset: int, codon_pos: int) -> str | None:
    codon_start = variant_offset - (2 - codon_pos)
    if codon_start < 0 or codon_start + 3 > len(ref_seq):
        return None
    genomic_codon = ref_seq[codon_start : codon_start + 3].upper()
    if "N" in genomic_codon:
        return None
    return reverse_complement_dna(genomic_codon)


def extract_codon(ref_seq: str, variant_offset: int, codon_pos: int, strand: str) -> str | None:
    if strand == "+":
        return extract_codon_plus(ref_seq, variant_offset, codon_pos)
    if strand == "-":
        return extract_codon_minus(ref_seq, variant_offset, codon_pos)
    raise ValueError(f"Unsupported strand {strand!r}. Expected '+' or '-'.")


def mutate_codon(codon: str, codon_pos: int, alt_base: str, strand: str) -> str | None:
    alt = alt_base.upper()
    if len(alt) != 1 or alt not in _COMPLEMENT:
        return None

    codon_index = codon_pos
    if strand == "-":
        alt = _COMPLEMENT.get(alt, "N")

    bases = list(codon.upper())
    if codon_index < 0 or codon_index >= len(bases):
        return None
    bases[codon_index] = alt
    alt_codon = "".join(bases)
    if "N" in alt_codon:
        return None
    return alt_codon


@dataclass(frozen=True)
class SyntheticSnvEffect:
    ref_codon: str
    alt_codon: str
    ref_aa: int
    alt_aa: int
    aa_change_class: int
    mutation_effect_class: int


def resolve_synthetic_snv_effect(
    *,
    ref_seq: str,
    variant_offset: int,
    codon_pos: int,
    alt_base: str,
    strand: str,
) -> SyntheticSnvEffect | None:
    ref_codon = extract_codon(ref_seq, variant_offset, codon_pos, strand)
    if ref_codon is None:
        return None

    alt_codon = mutate_codon(ref_codon, codon_pos, alt_base, strand)
    if alt_codon is None:
        return None

    ref_aa = translate_codon(ref_codon)
    alt_aa = translate_codon(alt_codon)
    mutation_effect_class = classify_mutation_effect(ref_aa, alt_aa)
    if mutation_effect_class is None:
        return None

    return SyntheticSnvEffect(
        ref_codon=ref_codon,
        alt_codon=alt_codon,
        ref_aa=ref_aa,
        alt_aa=alt_aa,
        aa_change_class=classify_aa_change(ref_aa, alt_aa),
        mutation_effect_class=mutation_effect_class,
    )
