"""Utilities for extracting genomic windows around variants."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pyfaidx import Fasta

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantWindow:
    """Result of extracting a genomic window around a variant."""

    ref_seq: str
    alt_seq: str
    variant_offset: int  # 0-based position of variant start within the window
    chrom: str  # with "chr" prefix as used in FASTA
    window_start: int  # 0-based genomic start
    window_end: int  # 0-based genomic end (exclusive)
    n_fraction: float  # fraction of N bases in the ref window
    status: str  # "ok", "boundary_shifted", "ref_mismatch"


def _resolve_variant_pos(
    fasta_chrom,
    pos: int,
    ref: str,
    chrom_len: int,
) -> int | None:
    """Find the 0-based position where ``ref`` matches the FASTA.

    ClinVar's ``Start`` is 1-based.  For SNVs, ``Start - 1`` gives the
    correct 0-based position.  For indels, ``ReferenceAlleleVCF`` includes
    a VCF anchor base that starts one position earlier (``Start - 2``).

    We try the primary offset first, then fall back to +-1 to handle both
    conventions robustly.
    """
    ref_upper = ref.upper()
    ref_len = len(ref_upper)

    primary = pos - 1
    for candidate in (primary, primary - 1, primary + 1):
        if 0 <= candidate <= chrom_len - ref_len:
            fasta_slice = str(fasta_chrom[candidate : candidate + ref_len]).upper()
            if fasta_slice == ref_upper:
                return candidate
    return None


def extract_variant_window(
    fasta: Fasta,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    context_size: int,
) -> VariantWindow | None:
    """Extract ref and alt sequences centered on a variant.

    Parameters
    ----------
    fasta:
        Open pyfaidx.Fasta handle (hg38, expects "chr" prefix).
    chrom:
        Chromosome name *without* "chr" prefix (e.g. "1", "X").
    pos:
        1-based ``Start`` position from ClinVar.
    ref:
        Reference allele string (``ReferenceAlleleVCF``).
    alt:
        Alternate allele string (``AlternateAlleleVCF``).
    context_size:
        Total window length in base pairs.

    Returns
    -------
    VariantWindow or None if the variant cannot be processed.
    """
    fasta_chrom = f"chr{chrom}"
    if fasta_chrom not in fasta:
        log.warning("Chromosome %s not found in FASTA", fasta_chrom)
        return None

    chrom_obj = fasta[fasta_chrom]
    chrom_len = len(chrom_obj)

    pos0 = _resolve_variant_pos(chrom_obj, pos, ref, chrom_len)
    if pos0 is None:
        log.warning(
            "Ref mismatch at %s:%d -- %s not found at any nearby offset",
            fasta_chrom, pos, ref[:40] + ("..." if len(ref) > 40 else ""),
        )
        return None

    half = context_size // 2
    start = pos0 - half
    end = start + context_size
    status = "ok"

    if start < 0:
        end = min(end - start, chrom_len)
        start = 0
        status = "boundary_shifted"
    if end > chrom_len:
        start = max(start - (end - chrom_len), 0)
        end = chrom_len
        status = "boundary_shifted"

    if end - start < context_size:
        log.warning(
            "Window too small for %s:%d (chrom_len=%d, context_size=%d)",
            fasta_chrom, pos, chrom_len, context_size,
        )
        return None

    variant_offset = pos0 - start
    ref_seq = str(chrom_obj[start:end]).upper()

    alt_seq = ref_seq[:variant_offset] + alt.upper() + ref_seq[variant_offset + len(ref) :]

    if len(alt_seq) > context_size:
        alt_seq = alt_seq[:context_size]
    elif len(alt_seq) < context_size:
        deficit = context_size - len(alt_seq)
        pad_start = end
        pad_end = min(end + deficit, chrom_len)
        if pad_end > pad_start:
            pad_bases = str(chrom_obj[pad_start:pad_end]).upper()
            alt_seq += pad_bases
        if len(alt_seq) < context_size:
            alt_seq += "N" * (context_size - len(alt_seq))

    n_fraction = ref_seq.count("N") / len(ref_seq)

    return VariantWindow(
        ref_seq=ref_seq,
        alt_seq=alt_seq,
        variant_offset=variant_offset,
        chrom=fasta_chrom,
        window_start=start,
        window_end=end,
        n_fraction=n_fraction,
        status=status,
    )


def compute_pool_bounds(variant_offset: int, radius: int, seq_len: int) -> tuple[int, int]:
    """Compute clamped [start, end) bounds for local pooling."""
    return max(0, variant_offset - radius), min(seq_len, variant_offset + radius)
