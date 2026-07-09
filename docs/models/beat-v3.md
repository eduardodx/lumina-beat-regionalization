# Beat-v3

`beat-v3` is the scaled representation-focused Lumina family.

Where `beat-v2` tried to improve coding awareness while staying compact, `beat-v3` asked a different question:

- was the next bottleneck model capacity and representation geometry rather than only task design?

## What It Changed

Compared with `beat-v2`, `beat-v3` made three major shifts:

- scaled the backbone from the compact ~9M tier toward the ~30M tier
- replaced auxiliary MLP heads with linear-probe heads
- added PhyloP-weighted MLM to emphasize conserved masked positions

These changes were aimed at making supervised signals more directly readable from the hidden states instead of relying
on expressive pretraining heads to decode them.

## Why It Existed

The motivation for `beat-v3` was not simply "make the model bigger."

The family was designed to test two hypotheses:

- the compact family may have been capacity-limited
- downstream representation quality may improve if the backbone is forced to linearly expose biological features

That makes `beat-v3` the most representation-centric family in the historical line.

## Trade-Off

The intended trade-off was explicit:

- pretraining auxiliary metrics might not look as strong as with more expressive heads
- but the hidden states might transfer better to downstream tasks that consume representations directly

This made `beat-v3` especially relevant to questions raised by ClinVar Regime A style evaluation.

## Current Status

`beat-v3` remains useful for:

- representation-quality ablations against `beat-v2`
- capacity-versus-objective comparisons
- understanding whether later gains should come from larger backbones or better consequence-aware supervision

It remains an important experimental family, but it is not the active documentation anchor.
