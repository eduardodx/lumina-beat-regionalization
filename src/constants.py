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
MUTATION_EFFECT_CLASS_NAMES = ["synonymous", "missense", "stop"]
NUM_MUTATION_EFFECT_CLASSES = len(MUTATION_EFFECT_CLASS_NAMES)
MUTATION_EFFECT_IGNORE_INDEX = -100

ALLELE_EFFECT_NEUTRAL = 0
ALLELE_EFFECT_SYNONYMOUS = 1
ALLELE_EFFECT_MISSENSE_CONSERVATIVE = 2
ALLELE_EFFECT_MISSENSE_NONCONSERVATIVE = 3
ALLELE_EFFECT_STOP_GAINED = 4
ALLELE_EFFECT_SPLICE_CORE = 5
ALLELE_EFFECT_CONSERVED_NONCODING = 6
ALLELE_EFFECT_OTHER = 7
ALLELE_EFFECT_CLASS_NAMES = [
    "neutral",
    "synonymous",
    "missense_conservative",
    "missense_nonconservative",
    "stop_gained",
    "splice_core",
    "conserved_noncoding",
    "other",
]
NUM_ALLELE_EFFECT_CLASSES = len(ALLELE_EFFECT_CLASS_NAMES)
ALLELE_EFFECT_IGNORE_INDEX = -100

COUNTERFACTUAL_EFFECT_NEUTRAL_OR_UNKNOWN = 0
COUNTERFACTUAL_EFFECT_SYNONYMOUS = 1
COUNTERFACTUAL_EFFECT_MISSENSE_CONSERVATIVE = 2
COUNTERFACTUAL_EFFECT_MISSENSE_NONCONSERVATIVE = 3
COUNTERFACTUAL_EFFECT_STOP_GAINED = 4
COUNTERFACTUAL_EFFECT_STOP_LOST = 5
COUNTERFACTUAL_EFFECT_START_LOST = 6
COUNTERFACTUAL_EFFECT_SPLICE_DONOR_CORE = 7
COUNTERFACTUAL_EFFECT_SPLICE_ACCEPTOR_CORE = 8
COUNTERFACTUAL_EFFECT_SPLICE_REGION = 9
COUNTERFACTUAL_EFFECT_CONSERVED_NONCODING = 10
COUNTERFACTUAL_EFFECT_OTHER_CODING_OR_AMBIGUOUS = 11
COUNTERFACTUAL_EFFECT_CLASS_NAMES = [
    "neutral_or_unknown",
    "synonymous",
    "missense_conservative",
    "missense_nonconservative",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "splice_donor_core",
    "splice_acceptor_core",
    "splice_region",
    "conserved_noncoding",
    "other_coding_or_ambiguous",
]
NUM_COUNTERFACTUAL_EFFECT_CLASSES = len(COUNTERFACTUAL_EFFECT_CLASS_NAMES)
COUNTERFACTUAL_EFFECT_IGNORE_INDEX = -100

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
