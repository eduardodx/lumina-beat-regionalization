# ClinVar Regional Fusion Status

Updated: 2026-06-23T00:00:00Z

## Implemented locally

- M6 explicit-frequency head support is implemented and dispatched.
- M4/M5 static adapter-fusion support is implemented locally:
  - `eval/clinvar/fusion_lora.py`
  - `scripts/clinvar_fusion_job.py`
  - `scripts/sagemaker_clinvar_fusion.py`
  - regional evaluator can reload fusion checkpoints.
- Focused validation passed:
  - `ruff check` on modified fusion/ClinVar files.
  - `pytest tests/test_eval_clinvar_fusion_lora.py tests/test_eval_clinvar_heads.py tests/test_eval_clinvar_dataset.py -q`.
  - CLI parsing for M4 and M5 configs.

## Remote execution status

- M6 explicit-frequency control:
  - job: `clinvar-m0-clinvar-m6-explicitfreq-23a613-20260622011931`
  - instance: `ml.g5.2xlarge`
  - status at last check: `Completed`
  - local artifacts: `artifacts/clinvar_regional_m6/explicitfreq_nonbr_beatv10_v1_sagemaker/`
  - internal non-BR test metrics: AUROC `0.912516`, AUPRC `0.915129`, MCC `0.653690`, Brier `0.126116`
- M4 static adapter fusion:
  - failed job: `clinvar-fuse-clinvar-m4-staticfusio-15be05-20260622020009`
  - failed instance: `ml.g5.xlarge`
  - failure reason: insufficient instance memory
  - rerun job: `clinvar-fuse-clinvar-m4-staticfusio-96fd31-20260622112940`
  - rerun instance: `ml.g5.2xlarge`
  - status at last check: `Completed`
  - local artifacts: `artifacts/clinvar_regional_fusion/m4_staticfusion_nonbr_beatv10_v1_sagemaker/`
  - internal non-BR test metrics: AUROC `0.884114`, AUPRC `0.894448`, MCC `0.590222`, Brier `0.149977`
  - output prefix: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m4-staticfusion-nonbr-beatv10-v1-rerun-g52x/sagemaker-artifacts/`
- M5 static adapter fusion + explicit frequency features:
  - job: `clinvar-fuse-clinvar-m5-staticfusio-d2e061-20260622020046`
  - instance: `ml.g5.4xlarge`
  - status at last check: `Completed`
  - local artifacts: `artifacts/clinvar_regional_fusion/m5_staticfusion_explicitfreq_nonbr_beatv10_v1_sagemaker/`
  - internal non-BR test metrics: AUROC `0.918574`, AUPRC `0.921954`, MCC `0.605862`, Brier `0.122873`
  - output prefix: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m5-staticfusion-explicitfreq-nonbr-beatv10-v1-g5-4x/sagemaker-artifacts/`

## Regional evaluation status

- M6 regional eval:
  - job: `clinvar-eval-m6-explicitfreq-nonbr-fc5991-20260622113034`
  - instance: `ml.g5.4xlarge`
  - status at last check: `Completed`
  - local artifacts: `artifacts/clinvar_regional_eval/m6_explicitfreq_nonbr_beatv10_v1_sagemaker/`
  - output prefix: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m6-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/`
- M5 regional eval:
  - job: `clinvar-eval-m5-staticfusion-explic-4bfe8f-20260622113115`
  - instance: `ml.g5.8xlarge`
  - status at last check: `Completed`
  - local artifacts: `artifacts/clinvar_regional_eval/m5_staticfusion_explicitfreq_nonbr_beatv10_v1_sagemaker/`
  - output prefix: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m5-staticfusion-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/`
- M4 regional eval:
  - job: `clinvar-eval-m4-staticfusion-nonbr-a3ff52-20260622190357`
  - instance: `ml.g5.4xlarge`
  - status at last check: `Completed`
  - local artifacts: `artifacts/clinvar_regional_eval/m4_staticfusion_nonbr_beatv10_v1_sagemaker/`
  - output prefix: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m4-staticfusion-nonbr-beatv10-v1/sagemaker-artifacts/`

## Regional findings so far

- M6 and M5 improve Brazilian test AUROC/MCC versus M0:
  - `br_only` MCC: M0 `0.279`, M6 `0.624`, M5 `0.618`
  - `br_any` MCC: M0 `0.263`, M6 `0.620`, M5 `0.617`
- M6 and M5 strongly reduce false positives on `abraom_common_benign`:
  - specificity: M0 `0.803`, M6 `0.998`, M5 `0.990`
- M6 and M5 over-suppress ABRAOM-present pathogenic variants:
  - `abraom_pathogenic_present` recall: M0 `0.417`, M6 `0.018`, M5 `0.135`
  - `abraom_pathogenic_common` recall: M0 `0.367`, M6 `0.000`, M5 `0.067`
- M4 is more conservative:
  - `abraom_common_benign` specificity: M0 `0.803`, M4 `0.894`
  - `abraom_pathogenic_present` recall: M0 `0.417`, M4 `0.288`
  - `global_nonbr_no_abraom` specificity: M0 `0.544`, M4 `0.733`
- Global non-BR specificity drops, especially for M5:
  - `global_nonbr_no_abraom` specificity: M0 `0.544`, M6 `0.354`, M5 `0.190`

## Final comparison artifact

- `artifacts/clinvar_regional_comparison/m0_m4_m5_m6_regional_test_summary.csv`
- `artifacts/clinvar_regional_comparison/REGIONAL_ADAPTER_FUSION_SUMMARY.md`

## Notes

- Parallel M4/M5 on `ml.g5.2xlarge` failed before job creation because the account quota for that exact instance type is 1 and M6 already used it.
- Parallel execution was recovered by using distinct G5 instance sizes where possible:
  - M6 training: `ml.g5.2xlarge`
  - M5 training: `ml.g5.4xlarge`
  - M4 rerun: `ml.g5.2xlarge`
  - M6 eval: `ml.g5.4xlarge`
  - M5 eval: `ml.g5.8xlarge`

## Next execution order

1. Tune regional calibration to reduce over-suppression of ABRAOM-present pathogenic variants.
2. Add sensitivity protection set / founder pathogenic panel before any clinical claim.
3. Re-run M5/M6 with constrained frequency influence and compare against this completed baseline.
