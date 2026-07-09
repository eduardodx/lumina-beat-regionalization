# Dynamic Fusion Job Runbook

Generated for the ABRAOM adapter-fusion blueprint completion cycle.

## Launched

- `M4_dynamic_gated`
  - SageMaker job: `clinvar-fuse-clinvar-m4-dynamic-gat-e23659-20260623193644`
  - Instance: `ml.g5.2xlarge`
  - Output prefix: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m4-dynamic-gated-nonbr-beatv10-v1/sagemaker-artifacts/`

- `M5_dynamic_gated`
  - SageMaker job: `clinvar-fuse-clinvar-m5-dynamic-gat-29c3ef-20260623193732`
  - Instance: `ml.g5.4xlarge`
  - Output prefix: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m5-dynamic-gated-bounded-nonbr-beatv10-v1/sagemaker-artifacts/`

## Remaining Training Jobs

Run after the current `ml.g5.2xlarge` / `ml.g5.4xlarge` jobs complete, or choose different instance types if quota allows.

```bash
uv run python scripts/sagemaker_clinvar_fusion.py \
  --experiment clinvar-m2-dynamic-gnomadonly-nonbr-beatv10-v1 \
  --instance-type ml.g5.2xlarge \
  --volume-size-gb 400 \
  --dataset-file nonbr_only.parquet \
  --fusion-mode dynamic_lora \
  --adapter-set gnomad_only \
  --detach \
  -- \
  --max-epochs 5 \
  --batch-size 4 \
  --grad-accum-steps 16 \
  --lr-head 5e-4 \
  --lr-backbone 5e-6 \
  --loss-type focal \
  --wandb-tags clinvar-regional m2-dynamic-gnomadonly
```

```bash
uv run python scripts/sagemaker_clinvar_fusion.py \
  --experiment clinvar-m7-dynamic-scrambled-nonbr-beatv10-v1 \
  --instance-type ml.g5.2xlarge \
  --volume-size-gb 400 \
  --dataset-file nonbr_only.parquet \
  --fusion-mode dynamic_lora \
  --adapter-set scrambled_gnomad \
  --detach \
  -- \
  --max-epochs 5 \
  --batch-size 4 \
  --grad-accum-steps 16 \
  --lr-head 5e-4 \
  --lr-backbone 5e-6 \
  --loss-type focal \
  --wandb-tags clinvar-regional m7-dynamic-scrambled
```

## Regional Evaluation Template

Replace `<experiment>` and `<training-output-prefix>` with the completed training output prefix ending at `/output/`.

```bash
uv run python scripts/sagemaker_clinvar_regional_eval.py \
  --experiment <experiment> \
  --instance-type ml.g5.4xlarge \
  --volume-size-gb 400 \
  --finetuned-s3-prefix <training-output-prefix> \
  --detach
```

## Final Compilation

After all dynamic regional evaluations complete, build one CSV with rows for `M2_gnomad_only`, `M4_dynamic_gated`, `M5_dynamic_gated`, and `M7_dynamic_scrambled`, then run:

```bash
uv run python scripts/compile_adapter_fusion_blueprint_completion_report.py \
  --dynamic-summary-csv artifacts/adapter_fusion_blueprint_completion/dynamic_fusion_regional_summary.csv \
  --overwrite
```
