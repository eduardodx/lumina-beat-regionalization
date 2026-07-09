# Beat-v1

`beat-v1` is the original compact Lumina family and the historical baseline for the later `beat-*` series.

In the model registry, `beat-v1` maps to the BiMamba3-RC architecture: a compact bidirectional Mamba model with
explicit reverse-complement strand processing and dense biological supervision on hg38.

## What It Established

`beat-v1` is important because it turned the project into a reproducible compact pretraining baseline with:

- single-nucleotide character-level tokenization
- dense PhyloP supervision
- dense splice-structure supervision
- held-out chromosome validation
- explicit reverse-complement-aware sequence processing

It is the family that demonstrated the repo's pretraining core could support real downstream evaluation instead of only
forward-pass experiments.

## Architectural Identity

Historically, the defining pieces of the `beat-v1` stack were:

- BiMamba3-RC blocks with forward and reverse-complement streams
- gated biological supervision over hg38 windows
- compact scale around the original 8M-tier budget
- pure mask replacement MLM with dense auxiliary tasks

The family is still useful as a reference point for later changes because it isolates the pretraining core before the
codon-aware and mutation-effect additions of later versions.

## Why It Was Not Enough

`beat-v1` was strong enough to make the downstream story interesting, but it exposed a limitation that motivated the
next families:

- the backbone did not have an explicit reason to internalize coding consequences, codon structure, or amino-acid
  semantics during pretraining

That made it a strong baseline on splice and intronic signal while still leaving room for improvement on
coding-consequence-sensitive tasks.

## Current Status

`beat-v1` remains useful for:

- reproducing older checkpoints
- historical comparisons
- ablations against later `beat-v2`, `beat-v3`, and `beat-v4` families

It is no longer the main documentation focus, but it remains an important baseline in the Lumina lineage.
