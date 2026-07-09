# Beat-v2

`beat-v2` is the first codon-aware compact Lumina family.

It kept the compact `beat-v1` scale and general O(n) Mamba structure, but added pretraining signals meant to make the
backbone more consequence-aware inside coding regions.

## What It Added

Compared with `beat-v1`, `beat-v2` introduced four important changes:

- an amino-acid prediction head over token hidden states
- codon-aligned masking inside CDS regions
- gated forward/backward strand fusion
- codon-averaged PhyloP supervision

Together, these changes were meant to push the compact backbone beyond "generic genomic context" and toward more
explicit coding semantics.

## Why It Mattered

`beat-v2` was the first clear attempt to improve downstream pathogenicity-relevant signal without abandoning the compact
baseline philosophy.

Its core idea was:

- preserve the compact training recipe
- preserve backward compatibility with the existing stack
- add biologically targeted coding supervision where `beat-v1` appeared weakest

That made it a practical bridge between the original compact baseline and the later representation-focused or
allele-sensitivity-adjacent families.

## Historical Role

`beat-v2` matters historically because it reframed the compact family around coding awareness rather than only generic
conservation and splice structure.

It also became the reference point for later questions such as:

- whether codon-aware objectives improved downstream transfer enough
- whether hidden states exposed those signals directly or only through expressive heads
- whether the next bottleneck was capacity, signal routing, or downstream interface design

## Current Status

`beat-v2` remains useful for:

- reproducing older ClinVar comparisons
- comparing codon-aware compact pretraining against `beat-v1`
- calibrating what later `beat-v3` and `beat-v4` changes actually contributed

It remains in the registry and documentation, but it is not the active family.
