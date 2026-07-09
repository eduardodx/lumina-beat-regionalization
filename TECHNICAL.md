# Beat-v2 Technical Reference

## Motivation

ClinVar fine-tuning on beat-v1 showed that the compact BiMamba3-RC backbone is already strong on splice and intronic signal, but still weak on coding-region consequence discrimination. In particular, missense and nonsense classes remain poorly separated because the backbone never has to internalize reading frames, codon grouping, or amino-acid semantics during pretraining.

Beat-v2 addresses that gap with four additive changes that preserve the O(n) Mamba backbone and keep existing model families untouched.

## Change 1: Amino Acid Prediction Head

Beat-v2 adds a 22-class amino-acid head over token hidden states:

- 20 canonical amino acids
- 1 stop class
- 1 `non_cds` sentinel used as `ignore_index`

The dataset resolves CDS labels from GENCODE canonical/basic transcripts and generates an amino-acid target for every CDS position whose full codon is available inside the sampled window. This forces the backbone to learn frame-consistent codon semantics instead of treating coding sequence as only another high-conservation region.

## Change 2: Codon-Aligned Masking

The masking pipeline still samples span starts the same way as beat-v1, but any sampled CDS position expands to its full codon boundaries. The implementation uses strand-aware CDS phase patterns so both plus-strand `[0, 1, 2]` and minus-strand `[2, 1, 0]` codons are recovered correctly.

This keeps MLM masked-only while making the corruption process more biologically aligned inside coding regions.

## Change 3: Gated Forward/Backward Fusion

Beat-v2 replaces the static linear forward/backward fusion in each BiMamba3-RC block with a position-dependent sigmoid gate:

- forward and reverse-complement streams are projected independently
- a learned gate mixes the two streams at each position
- the residual O(n) Mamba structure is preserved

This gives the model a more expressive strand-fusion mechanism without changing the high-level architecture family.

## Change 4: Codon-Averaged PhyloP

Beat-v2 adds a codon-level regression head trained against codon-averaged PhyloP100 targets. The objective is only supervised where a full codon can be recovered, making it a protein-level evolutionary-constraint proxy that complements the existing per-nucleotide conservation heads.

## Parameter Budget

The target scale remains compact:

- beat-v1: about 8.3M parameters
- beat-v2: about 9.5M parameters

The main delta comes from the gated fusion projections plus the new amino-acid and codon-conservation heads.

## Backward Compatibility

All beat-v2 changes are additive:

- existing dataset consumers still receive the original tensors, plus new codon-aware tensors
- existing models continue to train with their current heads and losses
- beat-v1 remains registered and unchanged
- beat-v2 is introduced as a separate model family with dedicated configs
