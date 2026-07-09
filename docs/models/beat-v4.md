# Beat-v4

`beat-v4` is the compact predecessor to `beat-v5`.

It keeps the compact `beat-v2`-scale backbone shape while adding synthetic CDS SNV consequence supervision through a
mutation-effect head.

## What It Adds

Compared with earlier compact families, `beat-v4` introduces:

- mutation-effect supervision over synthetic CDS SNV outcomes
- a dedicated mutation-effect token head in addition to the prior biological objectives
- continued compatibility with the existing registry-driven training stack

The in-tree base config is:

- `configs/beat_v4/_base.yaml`

and the main long-context training config in-tree is:

- `configs/beat_v4/9m_15ep_32k.yaml`

## Why It Matters

`beat-v4` is not yet the final answer to allele sensitivity. It matters because it pushes the pretraining signal closer
to consequence-aware behavior without pretending that downstream counterfactual validation is already solved.

`beat-v4` mattered because it moved the compact family closer to consequence-aware pretraining. `beat-v5` now replaces
it as the main documentation anchor by fixing the static RC path and adding a shared decoder plus a single-pass
ClinVar interface.

## Relationship To Older Families

Older Lumina families remain in the registry:

- `beat-v1`
- `beat-v2`
- `beat-v3`

They remain useful for:

- reproducibility
- ablations
- compatibility with older checkpoints

`beat-v4` itself also remains useful as the direct ablation against `beat-v5`.
