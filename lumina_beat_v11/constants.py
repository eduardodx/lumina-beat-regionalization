from __future__ import annotations

DNA_VOCAB = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "N": 5,
}

PAD_ID = 0
MASK_ID = 6
UNK_ID = 7
VOCAB_SIZE = 8

ID_TO_DNA = {
    PAD_ID: "<pad>",
    1: "A",
    2: "C",
    3: "G",
    4: "T",
    5: "N",
    MASK_ID: "<mask>",
    UNK_ID: "<unk>",
}

SNV_BASES = ("A", "C", "G", "T")
SNV_ALT_TO_INDEX = {base: index for index, base in enumerate(SNV_BASES)}

REGION_INTERGENIC = 0
REGION_INTRON = 1
REGION_NONCODING_EXON = 2
REGION_UTR = 3
REGION_CDS = 4
REGION_CLASS_NAMES = [
    "intergenic",
    "intron",
    "noncoding_exon",
    "utr",
    "cds",
]
NUM_REGION_CLASSES = len(REGION_CLASS_NAMES)

SPLICE_NONE = 0
SPLICE_DONOR = 1
SPLICE_ACCEPTOR = 2
SPLICE_DONOR_FLANK = 3
SPLICE_ACCEPTOR_FLANK = 4
SPLICE_CLASS_NAMES = [
    "none",
    "donor",
    "acceptor",
    "donor_flank",
    "acceptor_flank",
]
NUM_SPLICE_CLASSES = len(SPLICE_CLASS_NAMES)

# Conservation scalar targets (order matches conservation_scalar_pred[..., i]).
CONSERVATION_TARGET_NAMES = ["phylo100", "zoo241", "phylo470"]
NUM_CONSERVATION_TARGETS = len(CONSERVATION_TARGET_NAMES)
NUM_CONSERVATION_BINS = 16

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
COUNTERFACTUAL_EFFECT_CLASS_NAMES = [
    "synonymous",
    "missense_benign",
    "missense_damaging",
    "nonsense_or_stop_gained",
    "splice_donor_disrupt",
    "splice_acceptor_disrupt",
    "splice_region_change",
    "start_codon_disrupt",
    "stop_codon_change",
    "utr_change",
    "intronic_distal",
    "intergenic_or_no_effect",
]
NUM_COUNTERFACTUAL_EFFECT_CLASSES = len(COUNTERFACTUAL_EFFECT_CLASS_NAMES)
COUNTERFACTUAL_EFFECT_IGNORE_INDEX = -100

# Complement mapping: A(1)<->T(4), C(2)<->G(3), specials map to themselves.
COMPLEMENT_TABLE: list[int] = [0, 4, 3, 2, 1, 5, 6, 7]

# Cell-type vocabulary for the cell-conditioned Hi-C contact head (index = cell_id; default 0 =
# GM12878). Order matches the training data's CELL_TYPE_TO_ID.
CELL_TYPES: tuple[str, ...] = ("GM12878", "K562", "HepG2", "IMR90")
N_CELL_TYPES: int = len(CELL_TYPES)
