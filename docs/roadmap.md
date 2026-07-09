# Roadmap

## Current Status

- Reproducible pretraining, held-out validation, and checkpointing are implemented.
- Dense PhyloP and splice supervision are implemented in the main training path.
- The repository includes a working ClinVar fine-tuning and evaluation pipeline.
- `beat-v5` is the active documented model family.
- The next important gap is not basic infrastructure. It is stronger counterfactual and allele-sensitivity validation.

## Phase 0-3: Foundation Work Already In Place

The following roadmap layers are no longer aspirational. They are established enough to treat as baseline repository
capabilities:

- environment and configuration standardization around `uv`
- import-safe training and sanity entry points
- dense biological supervision over valid positions
- held-out chromosome validation and training metrics
- reproducible checkpointing and SageMaker dispatch support
- compact baseline model families plus active `beat-*` experimentation

These phases still need refinement over time, but they should be described as implemented baseline infrastructure, not
missing work.

## Phase 4: Counterfactual Variant Modeling And Interface Hardening

**Current priority**

The highest-value next step is to turn current downstream evaluation into a stronger counterfactual story.

Focus areas:

- define allele-sensitivity diagnostics that are reusable across runs
- add explicit ref<->alt swap checks
- add same-locus benign/pathogenic comparisons
- compare local token-level delta features against weaker pooled alternatives under the same benchmark setup
- clarify which parts of current ClinVar performance come from backbone signal versus explicit downstream features

The repository already has ref/alt downstream handling in the ClinVar pipeline. What is still missing is a disciplined,
standardized validation protocol that proves the scorer is responding to the edit itself.

## Phase 5: Controlled Ablations

Once the counterfactual interface is better pinned down, the next step is controlled ablation rather than broad
feature stacking.

Priority comparisons:

- dense auxiliary losses versus narrower supervision variants
- fixed task weights versus uncertainty weighting
- reverse-complement loss alternatives
- corruption strategy variants for MLM
- mutation-effect supervision and ClinVar interface choices around the active `beat-v5` family

Each change should remain isolated enough that downstream effects stay interpretable.

## Phase 6: Clinical Benchmark Maturity

ClinVar fine-tuning exists today, but the benchmark story is not complete.

Remaining work:

- temporal evaluation
- clearer consequence-specific reporting
- stronger benchmark preparation policy and provenance
- same-locus and counterfactual controls
- careful baseline comparison language for external models

This is the right place to mature the clinical benchmark narrative without overstating what current results prove.

## Phase 7+: Regional Validation And Translation

These phases remain downstream:

- regional calibration and external validation on Brazilian or Latin American cohorts
- VUS triage and subgroup failure analysis
- laboratory workflow integration and deployment claims

They remain important, but they should not drive present-tense documentation for this repository.
