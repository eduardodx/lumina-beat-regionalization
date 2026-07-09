# M5_v2 Regional Evaluation Report

Job: `clinvar-eval-clinvar-m5-v2-bounded-6e09fc-20260623165949`

Status: completed on 2026-06-23 17:43:40 UTC.

## Main Result

`M5_v2` improved the Brazilian/regional classification slices, but did not solve the sensitivity-protection problem. At the frozen training threshold (`0.5285`), it strongly reduced false positives in ABRAOM-common benign variants, but still suppressed many ABRAOM-present P/LP variants.

| model | br_only MCC | br_only recall | br_only specificity | ABRAOM common benign specificity | ABRAOM P/LP present recall | global nonBR MCC | global nonBR specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.279 | 0.920 | 0.299 | 0.803 | 0.417 | 0.512 | 0.544 |
| M5_calibrated | 0.546 | 0.920 | 0.591 | 0.952 | 0.331 | 0.512 | 0.544 |
| M6_calibrated | 0.553 | 0.920 | 0.598 | 0.954 | 0.325 | 0.512 | 0.544 |
| M7_scrambled | 0.417 | 0.920 | 0.441 | 0.903 | 0.252 | 0.512 | 0.544 |
| M5_v2 | 0.643 | 0.960 | 0.614 | 0.997 | 0.055 | 0.456 | 0.387 |

## Blueprint Criteria

| criterion | result | interpretation |
|---|---:|---|
| Brazilian gain: `br_only` MCC above M0 | pass | `0.643` vs `0.279`; strongest regional MCC so far. |
| Reduce false positives: ABRAOM-common benign specificity >= 0.95 | pass | `0.997`; excellent false-positive suppression. |
| Protect P/LP sensitivity: ABRAOM-present P/LP recall near M0 | fail | `0.055` vs M0 `0.417`; too many false benign calls. |
| Preserve global nonBR performance | fail | global nonBR MCC `0.456` vs M0 `0.512`; specificity also drops. |
| Beat scrambled control | mixed | beats `M7_scrambled` on `br_only` MCC and specificity, but loses badly on P/LP recall. |

## Component Check

The model emits both `molecular_probability` and regional `probability`. The regional discount is doing real work: ABRAOM-common benign variants have high mean discount (`0.803`) and low mean final probability (`0.157`). However, ABRAOM-present P/LP variants also receive enough discount to fall below threshold: final mean probability is only `0.316`, and recall is `0.055`.

Threshold sensitivity on test suggests the signal is not absent: with final probability threshold near `0.30`, ABRAOM-common benign specificity remains about `0.953` while ABRAOM-present P/LP recall rises to about `0.472`. This is not a valid final claim because that threshold was inspected on test; it should be selected on holdout.

## Recommendation

Do not promote this `M5_v2` checkpoint as the next scientific candidate yet. Use it as evidence that the constrained head can reduce false positives, but the operating point and/or discount cap still need calibration. The next step should be a holdout-tuned `M5_v2_calibrated` evaluation: tune threshold and discount bounds on holdout, then evaluate once on test against `M0/M5/M6/M7`.

## Artifacts

- Combined comparison: `artifacts/clinvar_regional_eval/m5_v2_bounded_regional_eval_beatv10_v1_sagemaker/m0_m4_m5_m6_m5cal_m6cal_m7_m5v2_regional_test_summary.csv`
- Component/threshold analysis: `artifacts/clinvar_regional_eval/m5_v2_bounded_regional_eval_beatv10_v1_sagemaker/m5_v2_component_threshold_analysis.csv`
- Raw regional eval outputs: `artifacts/clinvar_regional_eval/m5_v2_bounded_regional_eval_beatv10_v1_sagemaker/_extracted/`
