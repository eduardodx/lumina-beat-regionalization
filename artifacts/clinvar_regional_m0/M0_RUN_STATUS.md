# ClinVar Regional M0 Run Status

## Code Fix

- Fixed `eval/clinvar/dataset.py` so ClinVar batches always include `ref_alleles` and `alt_alleles`.
- Root cause of the failed smoke was `KeyError: 'alt_alleles'` during swap prediction in validation.
- Local checks:
  - `.venv/bin/ruff check eval/clinvar/dataset.py tests/test_eval_clinvar_dataset.py`
  - `.venv/bin/python -m pytest tests/test_eval_clinvar_dataset.py`

## Smoke Gate

- Job: `clinvar-m0-nonbr-smoke-fix1-eb0f6b-20260621181437`
- Status: `Completed`
- Training time: 1991 seconds
- Dataset: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/smoke/m0_nonbr_smoke.parquet`
- Output: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/clinvar-m0-nonbr-smoke-fix1/sagemaker-artifacts/clinvar-m0-nonbr-smoke-fix1-eb0f6b-20260621181437/output/model.tar.gz`
- Local artifacts: `artifacts/clinvar_regional_m0/smoke_fix1_sagemaker/`
- Produced files:
  - `best_model.pt`
  - `metrics.json`
  - `test_predictions.parquet`

Smoke metrics are structurally valid but not scientific because the smoke dataset has only 24 variants.

## Full M0

- Job: `clinvar-m0-nonbr-beatv10-v1-2e6520-20260621191336`
- Status: `Completed`
- Training time: 6290 seconds
- Dataset: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/slices/nonbr_only.parquet`
- Reference: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/hg38/`
- Checkpoint: `s3://ai4bio-lumina/releases/lumina-beat-v10-20260527182934/`
- Output: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/clinvar-m0-nonbr-beatv10-v1/sagemaker-artifacts/clinvar-m0-nonbr-beatv10-v1-2e6520-20260621191336/output/model.tar.gz`
- Local artifacts: `artifacts/clinvar_regional_m0/nonbr_beatv10_v1_sagemaker/`
- Hyperparameters:
  - `context_size=1024`
  - `native_feature_heads=none`
  - `lora_rank=8`
  - `lora_alpha=16`
  - `lora_dropout=0.05`
  - `batch_size=2`
  - `grad_accum_steps=8`
  - `max_epochs=3`
  - `val_fraction=0.1`
  - `loss_type=focal`
  - `pos_weight=auto`
  - `precision=auto`
  - `num_workers=0`

## Full M0 Metrics

- Train/validation/test rows: 11673 / 1297 / 2547
- Best epoch: 3
- Test AUROC: 0.879493
- Test AUPRC: 0.890005
- Test MCC at validation threshold: 0.575774
- Test balanced accuracy at validation threshold: 0.773991
- Test F1 at validation threshold: 0.817311
- Test precision / recall / specificity at validation threshold: 0.739833 / 0.912915 / 0.635067
- Validation optimal threshold: 0.431105
- Default-threshold MCC: 0.594342
- Default-threshold balanced accuracy: 0.795194
- Brier score: 0.149155
- Log loss: 0.465464

Consequence-stratified AUROC:

- SNV: 0.869763 over 2228 variants
- Indel: 0.907132 over 309 variants
- MNV: 1.000000 over 10 variants

Produced files:

- `best_model.pt`
- `metrics.json`
- `test_predictions.parquet`

## Next Gate

Use the trained M0 artifact as the non-Brazilian pathogenicity baseline for regional evaluation slices.
