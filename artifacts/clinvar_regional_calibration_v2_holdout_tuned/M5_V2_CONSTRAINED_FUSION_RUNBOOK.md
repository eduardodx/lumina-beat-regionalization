# M5_v2 Constrained Fusion Runbook

Purpose: train the blueprint-aligned follow-up to the post-hoc holdout-tuned calibration.

The new head type is `regime_a_bounded_regional`. It produces a molecular logit and a bounded regional discount, and returns:

`regional_logit = molecular_logit - bounded_discount`

This preserves the non-boosting safety rule from the v2 calibration while making it part of the trainable fusion model.

## Recommended SageMaker Launch

```bash
python scripts/sagemaker_clinvar_fusion.py \
  --experiment clinvar-m5-v2-bounded-regional-nonbr-beatv10-v1 \
  --instance-type ml.g5.4xlarge \
  --volume-size-gb 400 \
  --dataset-file nonbr_only.parquet \
  -- \
  --head-type regime_a_bounded_regional \
  --explicit-feature-columns \
    log10_af_abraom log10_af_gnomad af_delta af_abs_delta af_ratio_log10 \
    af_abraom_missing af_gnomad_missing specificity specificity_missing \
    abraom_present is_snv \
  --max-epochs 5 \
  --batch-size 4 \
  --grad-accum-steps 16 \
  --lr-head 5e-4 \
  --lr-backbone 5e-6 \
  --loss-type focal \
  --wandb-tags clinvar-regional m5-v2 bounded-regional
```

After training completes, run the same regional evaluation flow used for M5/M6 and then run:

```bash
python scripts/calibrate_clinvar_regional_scores.py \
  --bootstrap-iterations 1000
```

## Required Checks

- Compare `M5_v2` against `M0/M4/M5/M6/M5_calibrated/M6_calibrated/M7_scrambled`.
- Confirm ABRAOM-common benign specificity remains at least `0.95`.
- Confirm ABRAOM-present P/LP recall does not collapse.
- Confirm global non-BR MCC and specificity remain near M0.
- Review generated false-benign and false-pathogenic error-analysis CSVs.

## Current Status

Implemented locally:

- `eval.clinvar.heads.RegimeABoundedRegionalHead`
- CLI support via `--head-type regime_a_bounded_regional`
- config validation requiring explicit regional features

Not yet executed:

- full SageMaker training job for `M5_v2`
- regional evaluation of a trained `M5_v2` checkpoint
