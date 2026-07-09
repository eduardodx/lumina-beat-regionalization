# M5_v2 Calibrated Regional Report

Status: completed.

## What Changed

`M5_v2` was recalibrated using holdout predictions only. The selected configuration scales the learned regional discount by `0.5`, caps the effective discount at `0.5`, uses a regional threshold of `0.35`, and uses a global/molecular threshold of `0.75` for non-Brazilian control slices.

This keeps the blueprint separation between molecular evidence and regional interpretation: Brazilian/ABRAOM slices use the calibrated regional score, while global non-BR slices use the molecular score.

## Main Test Results

| model | br_only MCC | br_only recall | br_only specificity | ABRAOM common benign specificity | ABRAOM P/LP present recall | global nonBR MCC | global nonBR specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.279 | 0.920 | 0.299 | 0.803 | 0.417 | 0.512 | 0.544 |
| M5_calibrated | 0.546 | 0.920 | 0.591 | 0.952 | 0.331 | 0.512 | 0.544 |
| M6_calibrated | 0.553 | 0.920 | 0.598 | 0.954 | 0.325 | 0.512 | 0.544 |
| M7_scrambled | 0.417 | 0.920 | 0.441 | 0.903 | 0.252 | 0.512 | 0.544 |
| M5_v2 raw | 0.643 | 0.960 | 0.614 | 0.997 | 0.055 | 0.456 | 0.387 |
| M5_v2_calibrated | 0.605 | 0.995 | 0.457 | 0.959 | 0.436 | 0.500 | 0.844 |

## Criteria

| criterion | result | interpretation |
|---|---:|---|
| Brazilian gain over M0 | pass | `br_only` MCC `0.605` vs M0 `0.279`. |
| Reduce ABRAOM-common false positives | pass | ABRAOM-common benign specificity `0.959`, above the `0.95` target. |
| Protect ABRAOM-present P/LP sensitivity | pass | P/LP recall `0.436`, slightly above M0 `0.417` and much better than raw M5_v2 `0.055`. |
| Preserve global non-BR MCC/specificity | mostly pass | MCC `0.500` is close to M0 `0.512`; specificity improves to `0.844`. Recall is lower, so this is a specificity-biased global operating point. |
| Negative control | pass | `M7_scrambled` does not reproduce the ABRAOM-specific specificity/sensitivity balance. |

## Interpretation

The raw `M5_v2` proved that the bounded regional head can strongly suppress false positives, but it still over-suppressed ABRAOM-present P/LP variants. The holdout-tuned calibration fixed the central failure mode: it preserved ABRAOM-common specificity while recovering P/LP recall to M0-like levels.

This makes `M5_v2_calibrated` the current best scientific candidate, not a clinical final model. The remaining issue is operating-point policy: the global non-BR setting became high-specificity and lower-recall, so global deployment should keep an explicit molecular/sensitivity mode rather than reuse the Brazilian regional threshold blindly.

## Next Blueprint Steps

1. Run clustered bootstrap confidence intervals for `M5_v2_calibrated - M0`, `M5_v2_calibrated - M5_calibrated`, and `M5_v2_calibrated - M7_scrambled`.
2. Generate false-positive and false-negative error tables for ABRAOM-common benign and ABRAOM-present P/LP variants.
3. Build the curated Brazilian founder/P/LP protection panel and rerun sensitivity checks.
4. Add gate/discount diagnostics by ABRAOM AF bin, specificity bin, consequence, and gene.
5. Keep `M5_v2_calibrated` as the lead candidate only if bootstrap/error review supports the same conclusion.

## Artifacts

- Selected config: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/selected_config.json`
- Final comparison: `artifacts/clinvar_regional_comparison/m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_regional_test_summary.csv`
- Calibrated predictions and tuning table: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/`
