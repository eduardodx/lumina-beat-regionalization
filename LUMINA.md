# **Lumina**: Genomic AI for Latin America

Authors: Eduardo Souza, Pedro Sanches, and Celso Camilo.

## Proposition

The **Lumina** project investigates how genomic AI can make genetic interpretation faster, more accessible, and better calibrated for the Brazilian and Latin American reality.

That goal remains important, but it needs to be framed with more discipline than a typical broad innovation proposal. The relevant scientific landscape changed quickly. Compact backbones, long-context genomic modeling, and biological priors are no longer enough, by themselves, to constitute a strong research claim. What remains open, and scientifically consequential, is whether a compact genomic model can become genuinely **allele-sensitive** and clinically useful without depending mainly on context priors.

Lumina therefore treats the problem as a staged research program rather than a single all-encompassing deliverable.

## Clinical and Operational Context

Patients referred through the Brazilian public health system often face a slow diagnostic pipeline. Sample batching, sequencing logistics, manual review, and inconclusive interpretation can stretch the interval between collection and report delivery to months.

At the UFG Human Genetics Laboratory, the current workflow is tied to second-generation sequencing infrastructure. That setup offers operational convenience, but it also introduces batching constraints, substantial equipment and reagent costs, and dependence on interpretation workflows that are still labor-intensive. In practice, many difficult cases remain bottlenecked not only by sequencing itself, but by variant interpretation and by the uncertainty surrounding variants that do not fit cleanly into existing evidence frameworks.

This is the practical problem Lumina is trying to address.

## Research Objective

The core research objective is:

**To develop a compact, biologically supervised, allele-sensitive genomic foundation model for clinically relevant noncoding and splice-associated variant interpretation, and to validate it with explicit counterfactual, temporal, and regional calibration protocols.**

This objective is intentionally narrower than "build a better genomic model for everything" and more defensible than claiming, from the outset, a full clinical pipeline plus immediate regional superiority.

## Strategic Reframing

The strongest version of Lumina is not:

- another generic DNA language model
- a claim that small models automatically beat larger ones
- a claim that current models fail only because they are too large
- a claim that Brazilian utility follows automatically from regional motivation

The strongest version of Lumina is:

**a compact, biologically supervised, allele-sensitive genomic modeling program that tests whether current clinical benchmark performance is sometimes inflated by genomic context shortcuts, and that proposes a more counterfactually faithful alternative with explicit Brazilian-aware validation.**

That framing better matches both the scientific opportunity and the current repository scope.

## Why Latin America Still Matters

The regional thesis also benefits from a more careful statement.

It is no longer safe to argue simply that existing genomic foundation models are uniformly Eurocentric or that no diverse human modeling exists. Large foundation-model programs have already incorporated broader population data and post-training supervision. The more defensible and more important claim is this:

**Brazilian and admixed Latin American populations remain under-evaluated, under-calibrated, and underrepresented in current genomic foundation-model benchmarks and downstream clinical validation pipelines.**

This matters because a model can be impressive in aggregate while still being poorly calibrated for the populations and laboratory settings where it is eventually used.

## Paper Propositions

Lumina is best organized as three linked papers.

### Paper 1: Methods

**A compact, allele-sensitive genomic foundation model for noncoding and splice-relevant variant interpretation.**

This paper is the scientific core.

It should focus on:

- dense biological supervision through conservation and splice labels
- compact and reproducible pretraining
- counterfactual reference versus alternative modeling
- local token-level downstream interfaces rather than only frozen pooled embeddings
- rigorous held-out, swap-based, and same-locus evaluation

It should **not** try to carry MinION deployment or broad Brazilian public-health claims in the title.

### Paper 2: Regional Validation

**Brazilian-aware and admixed-population validation, calibration, and failure analysis.**

This paper should focus on:

- temporal validation, where older variants are hidden and newer variants are reserved for testing
- external calibration and subgroup analysis
- Brazilian or Latin American holdout resources where governance permits
- VUS triage, calibration, and failure-mode characterization

This is where regional relevance becomes a measured result rather than a background motivation.

### Paper 3: Translational Pipeline

**An operational workflow that connects sequencing, calling, and model-guided interpretation.**

This paper should focus on:

- third-generation sequencing workflow design
- calling plus interpretation pipeline integration
- cost, turnaround-time, and usability tradeoffs in the UFG context

This is where MinION or related laboratory workflow questions belong.

## Current Repository Scope

The current repository is the research core for the early part of **Paper 1**.

At present, it supports:

- compact bidirectional Mamba-style DNA modeling
- dense PhyloP and splice supervision on hg38
- reproducible sanity and training entry points
- held-out validation and experiment tracking

It does **not yet** provide:

- paired reference and alternative sequence training
- a ClinVar or ABraOM benchmark pipeline
- a production pathogenicity classifier
- a laboratory-facing sequencing workflow

Those are downstream phases of the program, not current implementation claims.

## Near-Term Scientific Priorities

The immediate scientific priority is not another architectural embellishment.

It is to determine whether the model can become sensitive to the actual allelic change.

That means the next decisive work should center on:

- paired reference and alternative windows
- local delta-style token features near the edited position
- ref<->alt swap tests
- same-locus benign versus pathogenic comparisons
- comparisons between pooled global embeddings and token-level variant interfaces

Only after that does it make sense to spend serious effort on more speculative architecture changes or broader translational narratives.

## Regional Validation Logic

Regional validation should be framed as a calibration and external-validity problem, not as an unsupported claim that Latin American-only training automatically solves interpretation.

The most compelling regional study design is likely to include:

- temporal evaluation on older versus newer curated variants
- external holdout or calibration analysis on Brazilian resources
- explicit subgroup and error analysis
- separation between pathogenicity ranking, VUS triage, and final clinical decision-making

This is a more defensible and publishable proposition than claiming immediate end-to-end regional superiority.

## Translational Ambition

The translational ambition remains real.

Portable third-generation sequencing platforms could eventually reduce batching constraints, compress turnaround time, and make laboratory workflows more flexible. In the long run, integrating sequencing, calling, and model-guided interpretation could materially improve access to diagnosis in resource-constrained settings.

But that claim should remain conditional:

- first establish a trustworthy modeling core
- then validate variant interpretation rigorously
- then study calibration in the target regional setting
- only after that, make strong workflow and deployment claims

This sequencing does not weaken the project. It makes the project more credible.

## Expected Impact

If Lumina succeeds, its contribution will be twofold.

Academically, it can help clarify a central question in genomic modeling: whether strong benchmark performance reflects genuine ref->alt reasoning or whether it is partly inflated by context priors, and whether a compact allele-sensitive alternative can close that gap.

Translationally, it can provide a more realistic foundation for future clinical interpretation and workflow integration in Brazil and Latin America, especially if calibration, validation, and deployment are treated as explicit research problems rather than assumptions.
