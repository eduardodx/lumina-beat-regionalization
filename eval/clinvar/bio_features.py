"""Biological feature extraction for Regime B ClinVar evaluation.

Extracts model-independent features from genomic annotations so that all
backbone models receive the same biological priors.  This lets the head
leverage coding-region knowledge that compact backbones may not have
learned during pretraining.

Feature vector layout (60 dimensions):
  [0]      phylo100            PhyloP100 conservation at variant site
  [1]      phylo470            PhyloP470 conservation at variant site
  [2]      is_cds              1 if variant is in CDS, else 0
  [3:7]    cds_phase_onehot    one-hot: phase 0 / 1 / 2 / non-CDS
  [7:29]   ref_aa_onehot       one-hot over 22 AA classes (incl non_cds, Stop)
  [29:51]  alt_aa_onehot       one-hot over 22 AA classes
  [51:55]  aa_change_onehot    one-hot: non_coding / synonymous / missense / nonsense
  [55]     blosum62            BLOSUM62 substitution score (0 if non-coding)
  [56]     gc_content          GC fraction in +-32bp window around variant
  [57:60]  variant_type        one-hot: SNV / insertion / deletion
"""

from __future__ import annotations

import bisect
import gzip
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BIO_FEATURE_DIM = 60

NUM_AA = 22
AA_NON_CDS = 0
AA_STOP = 21

# -- Genetic code (codon -> AA index matching src.constants) --

_GENETIC_CODE: dict[str, int] = {
    "TTT": 14, "TTC": 14, "TTA": 11, "TTG": 11,
    "TCT": 16, "TCC": 16, "TCA": 16, "TCG": 16,
    "TAT": 19, "TAC": 19, "TAA": 21, "TAG": 21,
    "TGT": 5, "TGC": 5, "TGA": 21, "TGG": 18,
    "CTT": 11, "CTC": 11, "CTA": 11, "CTG": 11,
    "CCT": 15, "CCC": 15, "CCA": 15, "CCG": 15,
    "CAT": 9, "CAC": 9, "CAA": 6, "CAG": 6,
    "CGT": 2, "CGC": 2, "CGA": 2, "CGG": 2,
    "ATT": 10, "ATC": 10, "ATA": 10, "ATG": 13,
    "ACT": 17, "ACC": 17, "ACA": 17, "ACG": 17,
    "AAT": 3, "AAC": 3, "AAA": 12, "AAG": 12,
    "AGT": 16, "AGC": 16, "AGA": 2, "AGG": 2,
    "GTT": 20, "GTC": 20, "GTA": 20, "GTG": 20,
    "GCT": 1, "GCC": 1, "GCA": 1, "GCG": 1,
    "GAT": 4, "GAC": 4, "GAA": 7, "GAG": 7,
    "GGT": 8, "GGC": 8, "GGA": 8, "GGG": 8,
}

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def _reverse_complement(seq: str) -> str:
    return "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq))


def _translate_codon(codon: str) -> int:
    """Translate a 3-letter DNA codon to an AA index.  Returns AA_NON_CDS on failure."""
    return _GENETIC_CODE.get(codon.upper(), AA_NON_CDS)


# -- BLOSUM62 matrix --

_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"  # maps to AA indices 1..20

_BLOSUM62_MATRIX = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
], dtype=np.float32)

# Map AA index (1..20) to BLOSUM62 row/col index (0..19)
_AA_IDX_TO_BLOSUM = {i + 1: i for i in range(20)}


def blosum62_score(ref_aa: int, alt_aa: int) -> float:
    """Look up BLOSUM62 score for a pair of standard amino acids."""
    return _shared_blosum62_score(ref_aa, alt_aa)


# -- AA change classification --

AA_CHANGE_NON_CODING = 0
AA_CHANGE_SYNONYMOUS = 1
AA_CHANGE_MISSENSE = 2
AA_CHANGE_NONSENSE = 3


def classify_aa_change(ref_aa: int, alt_aa: int) -> int:
    """Classify amino acid change type."""
    if ref_aa == AA_NON_CDS or alt_aa == AA_NON_CDS:
        return AA_CHANGE_NON_CODING
    if ref_aa == alt_aa:
        return AA_CHANGE_SYNONYMOUS
    if alt_aa == AA_STOP or ref_aa == AA_STOP:
        return AA_CHANGE_NONSENSE
    return AA_CHANGE_MISSENSE


# -- CDS interval index --

@dataclass(frozen=True)
class CdsHit:
    """Result of a CDS interval lookup."""

    start: int  # 0-based
    end: int  # exclusive
    strand: str  # "+" or "-"
    frame: int  # GTF frame field (0, 1, 2)


class CdsIndex:
    """Fast point-query index over CDS intervals from a GTF file."""

    def __init__(self, gtf_path: str | Path) -> None:
        raw = self._parse_gtf_cds(Path(gtf_path))
        self._starts: dict[str, list[int]] = {}
        self._intervals: dict[str, list[tuple[int, int, str, int]]] = {}
        for chrom, intervals in raw.items():
            intervals.sort()
            self._starts[chrom] = [iv[0] for iv in intervals]
            self._intervals[chrom] = intervals

    @staticmethod
    def _parse_gtf_cds(gtf_path: Path) -> dict[str, list[tuple[int, int, str, int]]]:
        intervals: dict[str, list[tuple[int, int, str, int]]] = defaultdict(list)
        opener = gzip.open if str(gtf_path).endswith(".gz") else open
        with opener(gtf_path, "rt") as f:  # type: ignore[call-overload]
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 9 or parts[2] != "CDS":
                    continue
                chrom = parts[0]
                start = int(parts[3]) - 1  # GTF is 1-based
                end = int(parts[4])  # GTF end is inclusive, so this is 0-based exclusive
                strand = parts[6]
                frame = int(parts[7]) if parts[7] != "." else 0
                intervals[chrom].append((start, end, strand, frame))
        log.info("Parsed CDS intervals from %s: %d chromosomes", gtf_path, len(intervals))
        return dict(intervals)

    def lookup(self, chrom: str, pos: int) -> CdsHit | None:
        """Find a CDS interval containing ``pos``.  Returns first match."""
        if chrom not in self._starts:
            return None
        starts = self._starts[chrom]
        intervals = self._intervals[chrom]
        idx = bisect.bisect_right(starts, pos)
        for i in range(idx - 1, max(idx - 100, -1), -1):
            start, end, strand, frame = intervals[i]
            if pos - start > 500_000:
                break
            if start <= pos < end:
                return CdsHit(start=start, end=end, strand=strand, frame=frame)
        return None


# -- Codon extraction --

def _codon_position_in_cds(pos: int, cds_start: int, frame: int) -> int:
    """Compute position within codon (0, 1, 2) for a plus-strand CDS."""
    d = pos - cds_start
    return (d + (3 - frame) % 3) % 3


def _extract_codon_plus(
    ref_seq: str,
    variant_offset: int,
    codon_pos: int,
) -> str | None:
    """Extract 3-letter codon from the reference window (plus strand)."""
    codon_start = variant_offset - codon_pos
    if codon_start < 0 or codon_start + 3 > len(ref_seq):
        return None
    codon = ref_seq[codon_start : codon_start + 3]
    if "N" in codon:
        return None
    return codon


def _extract_codon_minus(
    ref_seq: str,
    variant_offset: int,
    codon_pos: int,
) -> str | None:
    """Extract and reverse-complement a codon for minus-strand CDS."""
    codon_start = variant_offset - (2 - codon_pos)
    if codon_start < 0 or codon_start + 3 > len(ref_seq):
        return None
    codon_fwd = ref_seq[codon_start : codon_start + 3]
    if "N" in codon_fwd:
        return None
    return _reverse_complement(codon_fwd)


def _mutate_codon(
    codon: str,
    codon_pos: int,
    alt_base: str,
    strand: str,
) -> str | None:
    """Create the alternate codon after a single-nucleotide change."""
    if len(alt_base) != 1:
        return None
    if strand == "-":
        alt_base = _COMPLEMENT.get(alt_base.upper(), "N")
        codon_pos = 2 - codon_pos
    bases = list(codon)
    bases[codon_pos] = alt_base.upper()
    alt_codon = "".join(bases)
    if "N" in alt_codon:
        return None
    return alt_codon


# -- PhyloP extraction --

class PhyloScorer:
    """Wrapper around pyBigWig for conservation score lookups."""

    def __init__(self, bw_path: str | Path) -> None:
        import pyBigWig

        self._bw = pyBigWig.open(str(bw_path))

    def score(self, chrom: str, pos: int) -> float:
        """Return the conservation score at a single position."""
        try:
            values = self._bw.values(chrom, pos, pos + 1)
        except RuntimeError:
            return 0.0
        if values is None or len(values) == 0 or values[0] is None:
            return 0.0
        v = values[0]
        return 0.0 if np.isnan(v) else float(v)

    def close(self) -> None:
        self._bw.close()


# -- Feature vector construction --

def _gc_content(seq: str, center: int, radius: int = 32) -> float:
    """GC fraction in a local window around a position."""
    start = max(0, center - radius)
    end = min(len(seq), center + radius)
    window = seq[start:end].upper()
    if not window:
        return 0.0
    gc = sum(1 for b in window if b in ("G", "C"))
    return gc / len(window)


def _variant_type(ref: str, alt: str) -> int:
    """0=SNV, 1=insertion, 2=deletion."""
    if len(ref) == 1 and len(alt) == 1:
        return 0
    if len(alt) > len(ref):
        return 1
    return 2


def _one_hot(index: int, size: int) -> list[float]:
    vec = [0.0] * size
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


class BioFeatureExtractor:
    """Extracts the 60-dim biological feature vector for a variant."""

    def __init__(
        self,
        *,
        phylo100_bw_path: str | Path | None = None,
        phylo470_bw_path: str | Path | None = None,
        gtf_path: str | Path | None = None,
    ) -> None:
        self._phylo100 = PhyloScorer(phylo100_bw_path) if phylo100_bw_path else None
        self._phylo470 = PhyloScorer(phylo470_bw_path) if phylo470_bw_path else None
        self._cds_index = CdsIndex(gtf_path) if gtf_path else None

    def close(self) -> None:
        if self._phylo100:
            self._phylo100.close()
        if self._phylo470:
            self._phylo470.close()

    def extract(
        self,
        ref_seq: str,
        variant_offset: int,
        ref_allele: str,
        alt_allele: str,
        chrom: str,
        genomic_pos: int,
    ) -> list[float]:
        """Compute the full 60-dim feature vector for one variant."""
        features: list[float] = []

        # [0:2] PhyloP scores
        p100 = self._phylo100.score(chrom, genomic_pos) if self._phylo100 else 0.0
        p470 = self._phylo470.score(chrom, genomic_pos) if self._phylo470 else 0.0
        features.extend([p100, p470])

        # CDS lookup
        cds_hit = self._cds_index.lookup(chrom, genomic_pos) if self._cds_index else None
        is_cds = cds_hit is not None
        ref_aa = AA_NON_CDS
        alt_aa = AA_NON_CDS

        # [2] is_cds
        features.append(1.0 if is_cds else 0.0)

        # [3:7] cds_phase one-hot
        if is_cds:
            assert cds_hit is not None
            codon_pos = _codon_position_in_cds(genomic_pos, cds_hit.start, cds_hit.frame)
            features.extend(_one_hot(codon_pos, 4))  # 0, 1, 2 active; index 3 = non-CDS

            # Codon extraction (SNVs only)
            is_snv = len(ref_allele) == 1 and len(alt_allele) == 1
            if is_snv:
                if cds_hit.strand == "+":
                    ref_codon = _extract_codon_plus(ref_seq, variant_offset, codon_pos)
                else:
                    ref_codon = _extract_codon_minus(ref_seq, variant_offset, codon_pos)

                if ref_codon is not None:
                    alt_codon = _mutate_codon(ref_codon, codon_pos, alt_allele, cds_hit.strand)
                    ref_aa = _translate_codon(ref_codon)
                    if alt_codon is not None:
                        alt_aa = _translate_codon(alt_codon)
        else:
            features.extend(_one_hot(3, 4))  # non-CDS sentinel

        # [7:29] ref_aa one-hot
        features.extend(_one_hot(ref_aa, NUM_AA))

        # [29:51] alt_aa one-hot
        features.extend(_one_hot(alt_aa, NUM_AA))

        # [51:55] aa_change_type one-hot
        change_type = classify_aa_change(ref_aa, alt_aa)
        features.extend(_one_hot(change_type, 4))

        # [55] blosum62
        features.append(blosum62_score(ref_aa, alt_aa))

        # [56] gc_content
        features.append(_gc_content(ref_seq, variant_offset))

        # [57:60] variant_type one-hot
        features.extend(_one_hot(_variant_type(ref_allele, alt_allele), 3))

        assert len(features) == BIO_FEATURE_DIM, f"Expected {BIO_FEATURE_DIM}, got {len(features)}"
        return features


def zero_bio_features() -> list[float]:
    """Return a zero-valued feature vector (for Regime A or missing annotations)."""
    return [0.0] * BIO_FEATURE_DIM
