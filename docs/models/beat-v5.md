# Beat-v5

`beat-v5` is the active documented Lumina family.

It keeps the project compact and reproducible, but replaces `beat-v4`'s static reverse-complement side path with a
true evolving bidirectional backbone, a shared biology decoder, and a single-pass downstream interface that matches
ClinVar fine-tuning more directly.

## Goals

`beat-v5` is designed to be:

- compact enough for repeated ablations and affordable long-context runs
- efficient enough to avoid the default 2x ref/alt encoder cost during Lumina ClinVar fine-tuning
- biology-informed enough to expose short motifs, coding structure, and conservation from shared sequence features
- downstream-aligned enough that pretrained variant information is not isolated inside task-only heads

## Architecture

The active base config is:

- `configs/beat_v5/_base.yaml`

The main long-context training config currently in-tree is:

- `configs/beat_v5/384w_8l_15ep_32k.yaml`

The initial `beat-v5` backbone is width-first:

- `d_model=384`
- `n_layers=8`
- `d_state=64`
- `chunk_size=16`
- `is_mimo=false` by default
- `local_kernel_size=7`
- `decoder_dim=192`

The architecture still supports Mamba3 MIMO, but the shipped `beat-v5` defaults keep it disabled for now because the
current TileLang-backed MIMO kernel path has proven unstable across our active training environments. Re-enabling it
should be treated as an environment/kernel validation task, not a default modeling assumption.

For VRAM control, the shipped `beat-v5` defaults now also enable activation checkpointing across each backbone block
and the shared decoder during training and fine-tuning. That default can be disabled explicitly with
`model_config.activation_checkpointing: false` when a no-checkpoint ablation or debugging pass is needed.

### Input Representation

`beat-v5` keeps the same compact 8-token DNA vocabulary as prior Lumina families. The embedding stage produces one
evolving stream, not separate persistent forward and RC residual streams.

At the input:

1. embed the forward tokens
2. build reverse-complement ids
3. embed and flip them back into forward order
4. average the two aligned views into the initial hidden state

This preserves reverse-complement awareness without carrying a stale RC tensor through the full depth of the network.

### One Beat-v5 Block

Each block updates the current hidden state `h` with three branches:

1. `h_fwd = Mamba3(LayerNorm(h))`
2. `h_rev = flip(Mamba3(flip(LayerNorm(h))))`
3. `h_local = depthwise-conv(LayerNorm(h))`, symmetrized across forward and reverse application

The block then concatenates the three branches, computes a token-wise 3-way softmax gate, projects each branch back to
`d_model`, fuses them, and applies a residual add:

```text
gate = softmax(W_g [h_fwd ; h_rev ; h_local], axis=branch)
fused = gate_f * W_f(h_fwd) + gate_r * W_r(h_rev) + gate_l * W_l(h_local)
output = h + dropout(fused)
```

This gives `beat-v5` a compact mixture of:

- long-range forward sequence modeling
- long-range reverse-orientation sequence modeling
- local motif-sensitive processing

without introducing a second per-layer hidden stream.

### Shared Decoder

`beat-v5` moves most biology tasks onto a shared token decoder built from the final backbone states:

```text
LayerNorm(hidden_states)
-> DWConv(k=3) + DWConv(k=9), each symmetrized across RC
-> pointwise projection 384 -> 192
-> GELU
-> dropout
= decoder_states
```

`decoder_states` are the shared feature space for:

- `phylo100_pred`
- `phylo470_pred`
- `structure_logits`
- `region_logits`
- `aa_logits`
- `codon_phylo_pred`

These heads are intentionally lightweight and mostly linear so the backbone and decoder have to carry more of the
biology directly.

### MLM And Mutation-Effect Heads

MLM stays on the backbone hidden states with a tied output projection, preserving the compact masked-token pretraining
path from earlier families.

The mutation-effect head still emits:

- `mutation_effect_logits` with shape `[B, L, 4, 3]`

but it is no longer a standalone task MLP. Instead, it is allele-conditioned over `decoder_states`, combining per-site
decoder features with learned alternate-allele embeddings before predicting:

- synonymous
- missense
- stop

for each alternate base in `{A, C, G, T}`.

## Single-Pass ClinVar Path

`beat-v5` adds `extract_sequence_features(input_ids)`, which returns:

- `hidden_states`
- `decoder_states`

Those are generic downstream sequence features, not ClinVar-specific modules.

The Lumina ClinVar pipeline now uses them through a downstream task wrapper in one reference-window encoder pass:

1. `site_ref` comes from `hidden_states` at the variant token
2. `local_context` is mean-pooled from `decoder_states` over `+-64 bp`
3. `variant_repr` is built by a ClinVar-side allele-conditioned encoder from:
   - `site_ref`
   - pooled decoder context
   - `ref_allele`
   - `alt_allele`

That keeps the base model generic for future tasks while preserving the single-pass ClinVar interface. Regime A and
Regime B stay unchanged publicly. Metrics, splits, cache behavior, and output artifacts remain the same. The
difference is under the hood: `beat-v5` no longer requires the default paired ref/alt encoder path that older Lumina
families and external baselines still use.

## Why Replace Beat-v4's RC Design

The key `beat-v4` limitation was architectural, not only parametric.

`beat-v4` fused an updated forward stream with a reverse-complement stream that was embedded once and then reused
across all layers. That meant the RC branch did not build hierarchical features in step with the forward branch.

`beat-v5` replaces that with an evolving shared stream processed in both orientations at every layer. This keeps the
model compact while making reverse-complement information part of the actual depth-wise computation rather than a
static side signal.

## Why Start With 384 x 8

The first `beat-v5` config is width-first instead of jumping straight to a much larger model.

That choice is intentional:

- it increases representational room relative to `beat-v4`'s `256 x 8`
- it stays far cheaper than a `768`-wide model in memory and compute
- it preserves the repo's preference for controlled ablations over stacked changes
- it builds on the `beat-v3` lesson that representation quality, not just task count, can be a bottleneck

This makes `384 x 8` a practical first step for testing whether the new backbone and decoder improve transfer without
giving up the compact research posture of Lumina.

## What Stayed The Same

`beat-v5` is deliberately not a full stack reset.

Unchanged pieces include:

- the existing hg38 pretraining data contract
- current multitask loss keys and loss plumbing
- masked-only MLM
- existing ClinVar Regime A/B protocol, metrics, and artifacts
- registry-driven config and training entry points

That keeps the move from `beat-v4` to `beat-v5` interpretable as a model-interface upgrade rather than a benchmark
rewrite.

## Design Influences

The main design influences are:

- [Caduceus](https://pmc.ncbi.nlm.nih.gov/articles/PMC12189541/): RC-aware bidirectional state-space DNA modeling
- [HyenaDNA](https://arxiv.org/abs/2306.15794): strong single-nucleotide long-context DNA modeling without large token vocabularies
- [Enformer](https://www.nature.com/articles/s41592-021-01252-x): the importance of retaining local motif information alongside long-range context
- [SegmentNT](https://www.nature.com/articles/s41592-025-02881-2): shared dense decoders for nucleotide-resolution biology tasks
- [Nucleotide Transformer](https://www.nature.com/articles/s41592-024-02523-z): transfer-oriented genome foundation modeling benchmarks and interfaces
- `beat-v3`: the in-repo lesson that linearized supervision is useful when we want biology to stay visible in the shared representation

## Status

`beat-v5` is now the active documentation anchor. Older families remain in the registry for reproducibility, backward
compatibility, and ablation work:

- `beat-v1`
- `beat-v2`
- `beat-v3`
- `beat-v4`
