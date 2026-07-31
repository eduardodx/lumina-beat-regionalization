from __future__ import annotations

DNA_VOCAB = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "N": 5,
}

SNV_BASES = ("A", "C", "G", "T")
SNV_ALT_TO_INDEX = {base: index for index, base in enumerate(SNV_BASES)}

PAD_ID = 0
MASK_ID = 6
UNK_ID = 7
VOCAB_SIZE = 8

STRUCT_BACKGROUND = 0
STRUCT_SPLICE_CORE = 1
STRUCT_SPLICE_REGION = 2
NUM_STRUCTURE_CLASSES = 3

REGION_INTERGENIC = 0
REGION_INTRON = 1
REGION_NONCODING_EXON = 2
REGION_UTR = 3
REGION_CDS = 4
NUM_REGION_CLASSES = 5

AA_NAMES = [
    "non_cds",
    "Ala",
    "Arg",
    "Asn",
    "Asp",
    "Cys",
    "Gln",
    "Glu",
    "Gly",
    "His",
    "Ile",
    "Leu",
    "Lys",
    "Met",
    "Phe",
    "Pro",
    "Ser",
    "Thr",
    "Trp",
    "Tyr",
    "Val",
    "Stop",
]
NUM_AA_CLASSES = len(AA_NAMES)
AA_NON_CDS = 0
CDS_PHASE_NONE = -1
CODON_IGNORE_INDEX = -100

AA_CHANGE_NON_CODING = 0
AA_CHANGE_SYNONYMOUS = 1
AA_CHANGE_MISSENSE = 2
AA_CHANGE_NONSENSE = 3

MUTATION_EFFECT_SYNONYMOUS = 0
MUTATION_EFFECT_MISSENSE = 1
MUTATION_EFFECT_STOP = 2
NUM_MUTATION_EFFECT_CLASSES = 3
MUTATION_EFFECT_IGNORE_INDEX = -100

COUNTERFACTUAL_EFFECT_SYNONYMOUS = 0
COUNTERFACTUAL_EFFECT_MISSENSE_BENIGN = 1
COUNTERFACTUAL_EFFECT_MISSENSE_DAMAGING = 2
COUNTERFACTUAL_EFFECT_NONSENSE_OR_STOP_GAINED = 3
COUNTERFACTUAL_EFFECT_SPLICE_DONOR_DISRUPT = 4
COUNTERFACTUAL_EFFECT_SPLICE_ACCEPTOR_DISRUPT = 5
COUNTERFACTUAL_EFFECT_SPLICE_REGION_CHANGE = 6
COUNTERFACTUAL_EFFECT_START_CODON_DISRUPT = 7
COUNTERFACTUAL_EFFECT_STOP_CODON_CHANGE = 8
COUNTERFACTUAL_EFFECT_UTR_CHANGE = 9
COUNTERFACTUAL_EFFECT_INTRONIC_DISTAL = 10
COUNTERFACTUAL_EFFECT_INTERGENIC_OR_NO_EFFECT = 11
NUM_COUNTERFACTUAL_EFFECT_CLASSES = 12
COUNTERFACTUAL_EFFECT_IGNORE_INDEX = -100

# RC3 (current head-quality fixes): consolidated 9-class counterfactual scheme applied as a LOAD-TIME REMAP over
# the baked 12-class cache (no re-derivation). Merges the 3 redundant splice classes -> one splice_affecting
# (the dedicated 5-class splice head already scores ~0.91 on the SAME GENCODE donor/acceptor annotation) and
# the benign/damaging missense split -> one missense (the graded ESM-2 missense_severity head is the better,
# threshold-free signal; merging sidesteps the un-recalibrated chr22-estimate ESM-2 damaging threshold). Only
# active when `counterfactual_consolidation_enabled` (default off => old runs byte-identical).
COUNTERFACTUAL9_SYNONYMOUS = 0
COUNTERFACTUAL9_MISSENSE = 1
COUNTERFACTUAL9_NONSENSE_OR_STOP_GAINED = 2
COUNTERFACTUAL9_SPLICE_AFFECTING = 3
COUNTERFACTUAL9_START_CODON_DISRUPT = 4
COUNTERFACTUAL9_STOP_CODON_CHANGE = 5
COUNTERFACTUAL9_UTR_CHANGE = 6
COUNTERFACTUAL9_INTRONIC_DISTAL = 7
COUNTERFACTUAL9_INTERGENIC_OR_NO_EFFECT = 8
NUM_COUNTERFACTUAL9_EFFECT_CLASSES = 9
COUNTERFACTUAL9_CLASS_NAMES = (
    "synonymous", "missense", "nonsense_or_stop_gained", "splice_affecting", "start_codon_disrupt",
    "stop_codon_change", "utr_change", "intronic_distal", "intergenic_or_no_effect",
)
# old 12-class index -> new 9-class index (IGNORE -100 is preserved separately by the remap helper).
_COUNTERFACTUAL_12_TO_9 = (0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8)


def remap_counterfactual_effect_labels(labels):
    """Map baked 12-class counterfactual labels to the consolidated 9-class scheme (RC3).

    ``labels`` is a long tensor with values in ``0..11`` or ``COUNTERFACTUAL_EFFECT_IGNORE_INDEX`` (-100);
    IGNORE is preserved. Any out-of-range value is clamped before lookup, so a corrupt cache cell can never
    index out of bounds.
    """
    import torch

    lut = torch.tensor(_COUNTERFACTUAL_12_TO_9, dtype=labels.dtype, device=labels.device)
    ignore = labels == COUNTERFACTUAL_EFFECT_IGNORE_INDEX
    safe = labels.clamp(min=0, max=len(_COUNTERFACTUAL_12_TO_9) - 1)
    out = lut[safe]
    out[ignore] = COUNTERFACTUAL_EFFECT_IGNORE_INDEX
    return out

ALLELE_EFFECT_NEUTRAL = 0
ALLELE_EFFECT_SYNONYMOUS = 1
ALLELE_EFFECT_MISSENSE_CONSERVATIVE = 2
ALLELE_EFFECT_MISSENSE_NONCONSERVATIVE = 3
ALLELE_EFFECT_STOP_GAINED = 4
ALLELE_EFFECT_SPLICE_CORE = 5
ALLELE_EFFECT_CONSERVED_NONCODING = 6
NUM_ALLELE_EFFECT_CLASSES = 7
ALLELE_EFFECT_IGNORE_INDEX = -100

_AA_INDEX_BY_NAME = {name: index for index, name in enumerate(AA_NAMES)}
AA_STOP = _AA_INDEX_BY_NAME["Stop"]
_STANDARD_GENETIC_CODE = {
    "TTT": "Phe",
    "TTC": "Phe",
    "TTA": "Leu",
    "TTG": "Leu",
    "TCT": "Ser",
    "TCC": "Ser",
    "TCA": "Ser",
    "TCG": "Ser",
    "TAT": "Tyr",
    "TAC": "Tyr",
    "TAA": "Stop",
    "TAG": "Stop",
    "TGT": "Cys",
    "TGC": "Cys",
    "TGA": "Stop",
    "TGG": "Trp",
    "CTT": "Leu",
    "CTC": "Leu",
    "CTA": "Leu",
    "CTG": "Leu",
    "CCT": "Pro",
    "CCC": "Pro",
    "CCA": "Pro",
    "CCG": "Pro",
    "CAT": "His",
    "CAC": "His",
    "CAA": "Gln",
    "CAG": "Gln",
    "CGT": "Arg",
    "CGC": "Arg",
    "CGA": "Arg",
    "CGG": "Arg",
    "ATT": "Ile",
    "ATC": "Ile",
    "ATA": "Ile",
    "ATG": "Met",
    "ACT": "Thr",
    "ACC": "Thr",
    "ACA": "Thr",
    "ACG": "Thr",
    "AAT": "Asn",
    "AAC": "Asn",
    "AAA": "Lys",
    "AAG": "Lys",
    "AGT": "Ser",
    "AGC": "Ser",
    "AGA": "Arg",
    "AGG": "Arg",
    "GTT": "Val",
    "GTC": "Val",
    "GTA": "Val",
    "GTG": "Val",
    "GCT": "Ala",
    "GCC": "Ala",
    "GCA": "Ala",
    "GCG": "Ala",
    "GAT": "Asp",
    "GAC": "Asp",
    "GAA": "Glu",
    "GAG": "Glu",
    "GGT": "Gly",
    "GGC": "Gly",
    "GGA": "Gly",
    "GGG": "Gly",
}
CODON_TO_AA: dict[str, int] = {
    codon: _AA_INDEX_BY_NAME[aa_name]
    for codon, aa_name in _STANDARD_GENETIC_CODE.items()
}

# Complement mapping: A(1)<->T(4), C(2)<->G(3), specials map to themselves.
COMPLEMENT_TABLE: list[int] = [0, 4, 3, 2, 1, 5, 6, 7]

DEFAULT_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
DEFAULT_VAL_CHROMOSOMES = ["chr19", "chr21", "chr22", "chrX"]
