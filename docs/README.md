# Documentation

This directory is the primary source of truth for human-facing project documentation.

## Start Here

- [LUMINA](../LUMINA.md): broader project context, motivation, and long-range objectives
- [Research Program](research-program.md): project framing, scientific thesis, and current scope
- [Roadmap](roadmap.md): implementation status and next research phases
- [Local Development](local-dev.md): setup, verification, and platform-specific guidance
- [Confidentiality and AWS Access Term](confidentialidade-e-uso-aws.md): participant confidentiality, AWS usage, and responsibility term
- [SageMaker Ops](ops/sagemaker.md): training and ClinVar launch workflows on SageMaker
- [SageMaker Domain Provisioning](ops/sagemaker-domain.md): idempotent Studio domain bootstrap for the shared AWS account
- [ClinVar Evaluation](eval/clinvar.md): current two-regime ClinVar fine-tuning and evaluation pipeline
- [ClinVar Dataset Curation](eval/clinvar-dataset.md): source parquet, filters, metadata, and notebook inventory
- [Beat-v1](models/beat-v1.md): original compact baseline family
- [Beat-v2](models/beat-v2.md): codon-aware compact family
- [Beat-v3](models/beat-v3.md): scaled representation-focused family
- [Beat-v4](models/beat-v4.md): predecessor compact family with mutation-effect supervision
- [Beat-v5](models/beat-v5.md): active model family with shared bidirectional scan and single-pass ClinVar path

## Supporting References

- [Configs Overview](../configs/README.md): configuration families and which ones are active
- [Scripts Overview](../scripts/README.md): setup and launcher utilities

## Current State

- Pretraining, sanity checks, and held-out validation are implemented.
- The repo includes a real ClinVar fine-tuning and evaluation pipeline.
- `beat-v5` is the active Lumina family for current documentation and planning.
- Counterfactual same-locus diagnostics, swap tests, temporal evaluation, regional validation, and any production
  clinical workflow remain downstream work.
