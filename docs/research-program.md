# Lumina Research Program

Lumina is a compact, biologically supervised genomic modeling program aimed at allele-sensitive variant
interpretation.

The scientific framing is intentionally narrower than "build a better DNA foundation model for everything." The
central question is whether a compact human-genome model can become genuinely sensitive to the causal ref->alt change
instead of relying mainly on genomic context priors.

## Core Objective

Lumina's current research objective is:

**To develop a compact, biologically supervised, allele-sensitive genomic foundation model for clinically relevant
noncoding and splice-associated variant interpretation, and to evaluate it with explicit counterfactual, temporal, and
regional validation protocols.**

## Program Logic

The project is strongest when treated as a staged program:

1. Build a reproducible pretraining core on hg38 with dense biological supervision.
2. Improve allele sensitivity and counterfactual faithfulness.
3. Benchmark clinical utility with controlled downstream evaluation.
4. Validate calibration and failure modes on regional cohorts.
5. Only then make translational workflow claims.

This repository covers the early-to-middle part of that program.

## What The Repository Does Today

The current codebase supports:

- registry-backed bidirectional Mamba-style DNA backbones
- hg38 sampling with dense PhyloP and splice supervision
- held-out chromosome validation and training observability
- SageMaker training and checkpoint packaging workflows
- ClinVar fine-tuning and evaluation through a two-regime downstream pipeline
- active experimentation around `beat-v5`

## What It Does Not Claim Yet

The repository does not yet provide:

- paired ref/alt pretraining as the default training objective
- swap-based or same-locus counterfactual diagnostics as a mature standard pipeline
- temporal clinical benchmark protocols
- Brazilian or Latin American external validation results
- a production pathogenicity classifier or laboratory workflow

Those are explicit downstream goals, not present-tense claims.

## Why Compactness Still Matters

Compactness is not the novelty claim by itself. It matters because it makes controlled ablations, repeated runs, and
reproducible iteration feasible. The scientific claim has to come from biological supervision, counterfactual fidelity,
and evaluation quality.

## Current Research Story

The strongest current framing for Lumina is:

- compact enough for reproducible iteration
- biologically supervised at dense token resolution
- evaluated against real downstream clinical tasks
- moving toward stronger allele-sensitive interfaces rather than generic pooled embeddings

That framing is more defensible than claiming immediate clinical superiority, universal regional benefit, or an
end-to-end production interpretation system.
