# M0 Non-Brazilian ClinVar Pathogenicity Baseline

Generated: 2026-06-21

## Purpose

Build the baseline pathogenicity model required by the adapter-fusion blueprint:

`M0 = F0 + A_path + H_mol/H_reg`

This model must not use Brazilian population adapters or explicit ABRAOM features. It is the pathogenicity-only baseline used to test whether later `A_BR` fusion improves Brazilian-relevant calibration without harming global performance.

## Training Dataset

Primary dataset:

`data/datasets/clinvar/regional_abraom/slices/nonbr_only.parquet`

Rows:

| Split | Benign/LB | Pathogenic/LP | Total |
|---|---:|---:|---:|
| train | 3,117 | 9,853 | 12,970 |
| test | 1,192 | 1,355 | 2,547 |
| holdout | 1,412 | 2,083 | 3,495 |

Training uses `train`; validation is created by stratified split from `train`; the built-in final test uses `test`.

## Regional Evaluation Slices

These slices are not training data for M0. They are held for downstream evaluation of the trained M0 checkpoint and later fusion models.

| Slice | Purpose | Rows |
|---|---|---:|
| `br_only.parquet` | Brazilian-submitter-only ClinVar stratum | 4,163 |
| `mixed_br_nonbr.parquet` | Variants with Brazilian and non-Brazilian submitter evidence | 709 |
| `br_any.parquet` | Any Brazilian submitter evidence | 4,872 |
| `abraom_common_benign.parquet` | False-pathogenic-rate pseudo-benign check | 56,309 |
| `abraom_pathogenic_present.parquet` | Do-not-suppress P/LP safety check | 1,596 |

## Recommended Smoke Command

Use a short local or SageMaker smoke first:

```bash
python -m eval.clinvar.run \
  --regime A \
  --model-family lumina \
  --model-version beat-v10 \
  --checkpoint-path artifacts/abraom_regional_eval/checkpoints/base/best_checkpoint.pt \
  --dataset-path data/datasets/clinvar/regional_abraom/slices/nonbr_only.parquet \
  --fasta-path /home/sagemaker-user/lumina/data/genomes/hg38/raw/hg38.fa \
  --context-size 512 \
  --native-feature-heads none \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-epochs 1 \
  --val-fraction 0.02 \
  --precision fp32 \
  --output-dir outputs/clinvar_m0_nonbr_smoke \
  --overwrite
```

## Recommended Full M0 Command

Use SageMaker/GPU for the full run:

```bash
python -m eval.clinvar.run \
  --regime A \
  --model-family lumina \
  --model-version beat-v10 \
  --checkpoint-path artifacts/abraom_regional_eval/checkpoints/base/best_checkpoint.pt \
  --dataset-path data/datasets/clinvar/regional_abraom/slices/nonbr_only.parquet \
  --fasta-path /home/sagemaker-user/lumina/data/genomes/hg38/raw/hg38.fa \
  --context-size 1024 \
  --native-feature-heads none \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --max-epochs 3 \
  --val-fraction 0.1 \
  --loss-type focal \
  --pos-weight auto \
  --precision auto \
  --output-dir outputs/clinvar_m0_nonbr_beatv10 \
  --overwrite
```

## Success Criteria For M0

- Produces `best_model.pt`, `metrics.json`, and `test_predictions.parquet`.
- Establishes non-Brazilian pathogenicity baseline on `nonbr_only` test.
- Later evaluation on BR/ABRAOM slices can quantify false-pathogenic reduction and P/LP sensitivity preservation after regional fusion.

## Current Status

- ABRAOM frequency adapter gate completed.
- `A_BR` passed negative-control comparison against scrambled AF and beat `A_gnomAD` on test NLL/Brier against ABRAOM AF.
- Regional ClinVar slices are already materialized locally.
- Next execution step: smoke `M0` on `nonbr_only.parquet`, then dispatch full M0 on SageMaker if smoke passes.
