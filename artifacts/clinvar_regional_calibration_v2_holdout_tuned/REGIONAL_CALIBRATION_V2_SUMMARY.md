# ClinVar Regional Calibration v2

Generated at UTC: `2026-06-23T12:51:17.126931+00:00`

## Calibration Rule

- `molecular_score`: threshold-normalized M0 score.
- `regional_score`: M0 score after a bounded regional discount.
- Regional evidence can only lower M0; it cannot raise pathogenicity above M0.
- The learned M5/M6 discount is capped by ABRAOM AF and reduced for high ABRAOM specificity.
- `M7_scrambled` uses M5 learned discounts with frequency triples scrambled across evaluated rows.
- Parameters were selected on `holdout` and frozen before `test` evaluation.

## Selected Parameters

```json
{
  "M5_calibrated": {
    "af_log10_temperature": 0.25,
    "af_midpoint": 0.05,
    "max_down_margin": 4.0,
    "scrambled_seed": 1729,
    "specificity_protect_threshold": 0.05,
    "specificity_temperature": 0.01
  },
  "M6_calibrated": {
    "af_log10_temperature": 0.25,
    "af_midpoint": 0.05,
    "max_down_margin": 4.0,
    "scrambled_seed": 1729,
    "specificity_protect_threshold": 0.05,
    "specificity_temperature": 0.01
  },
  "M7_scrambled": {
    "af_log10_temperature": 0.25,
    "af_midpoint": 0.05,
    "max_down_margin": 4.0,
    "scrambled_seed": 1729,
    "specificity_protect_threshold": 0.05,
    "specificity_temperature": 0.01
  }
}
```

## Holdout Tuning Winners

| Model | Score | Max down margin | AF midpoint | AF temp | Specificity protect | BR MCC | Benign specificity | P/LP recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M5_calibrated | 2.400 | 4.000 | 0.050 | 0.250 | 0.050 | 0.501 | 0.950 | 0.353 |
| M6_calibrated | 2.392 | 4.000 | 0.050 | 0.250 | 0.050 | 0.506 | 0.952 | 0.348 |

## Focus Metrics

| Model | Dataset | N | AUROC | AUPRC | MCC | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| M0 | br_only | 504 | 0.745 | 0.896 | 0.279 | 0.920 | 0.299 |
| M0 | global_nonbr_no_abraom | 1989 | 0.849 | 0.919 | 0.512 | 0.918 | 0.544 |
| M0 | abraom_common_benign | 12099 | NA | NA | 0.000 | 0.000 | 0.803 |
| M0 | abraom_pathogenic_present | 163 | NA | 1.000 | 0.000 | 0.417 | 0.000 |
| M4 | br_only | 504 | 0.744 | 0.898 | 0.292 | 0.809 | 0.488 |
| M4 | global_nonbr_no_abraom | 1989 | 0.855 | 0.923 | 0.526 | 0.809 | 0.733 |
| M4 | abraom_common_benign | 12099 | NA | NA | 0.000 | 0.000 | 0.894 |
| M4 | abraom_pathogenic_present | 163 | NA | 1.000 | 0.000 | 0.288 | 0.000 |
| M5 | br_only | 504 | 0.866 | 0.942 | 0.618 | 0.979 | 0.528 |
| M5 | global_nonbr_no_abraom | 1989 | 0.855 | 0.924 | 0.328 | 0.988 | 0.190 |
| M5 | abraom_common_benign | 12099 | NA | NA | 0.000 | 0.000 | 0.990 |
| M5 | abraom_pathogenic_present | 163 | NA | 1.000 | 0.000 | 0.135 | 0.000 |
| M6 | br_only | 504 | 0.866 | 0.944 | 0.624 | 0.960 | 0.591 |
| M6 | global_nonbr_no_abraom | 1989 | 0.843 | 0.917 | 0.435 | 0.966 | 0.354 |
| M6 | abraom_common_benign | 12099 | NA | NA | 0.000 | 0.000 | 0.998 |
| M6 | abraom_pathogenic_present | 163 | NA | 1.000 | 0.000 | 0.018 | 0.000 |
| M5_calibrated | br_only | 504 | 0.843 | 0.933 | 0.546 | 0.920 | 0.591 |
| M5_calibrated | global_nonbr_no_abraom | 1989 | 0.849 | 0.919 | 0.512 | 0.918 | 0.544 |
| M5_calibrated | abraom_common_benign | 12099 | NA | NA | 0.000 | 0.000 | 0.952 |
| M5_calibrated | abraom_pathogenic_present | 163 | NA | 1.000 | 0.000 | 0.331 | 0.000 |
| M6_calibrated | br_only | 504 | 0.846 | 0.933 | 0.553 | 0.920 | 0.598 |
| M6_calibrated | global_nonbr_no_abraom | 1989 | 0.849 | 0.920 | 0.512 | 0.918 | 0.544 |
| M6_calibrated | abraom_common_benign | 12099 | NA | NA | 0.000 | 0.000 | 0.954 |
| M6_calibrated | abraom_pathogenic_present | 163 | NA | 1.000 | 0.000 | 0.325 | 0.000 |
| M7_scrambled | br_only | 504 | 0.811 | 0.922 | 0.417 | 0.920 | 0.441 |
| M7_scrambled | global_nonbr_no_abraom | 1989 | 0.850 | 0.920 | 0.512 | 0.918 | 0.544 |
| M7_scrambled | abraom_common_benign | 12099 | NA | NA | 0.000 | 0.000 | 0.903 |
| M7_scrambled | abraom_pathogenic_present | 163 | NA | 1.000 | 0.000 | 0.252 | 0.000 |

## Criteria

| Model | Criterion | Value | Reference | Passed |
|---|---|---:|---:|---|
| M5_calibrated | br_only_mcc_above_m0 | 0.546 | 0.279 | yes |
| M5_calibrated | abraom_common_benign_specificity_ge_0_95 | 0.952 | 0.950 | yes |
| M5_calibrated | abraom_pathogenic_present_recall_not_collapsed_like_m6 | 0.331 | 0.018 | yes |
| M5_calibrated | global_nonbr_no_abraom_near_m0 | 0.512 | 0.512 | yes |
| M6_calibrated | br_only_mcc_above_m0 | 0.553 | 0.279 | yes |
| M6_calibrated | abraom_common_benign_specificity_ge_0_95 | 0.954 | 0.950 | yes |
| M6_calibrated | abraom_pathogenic_present_recall_not_collapsed_like_m6 | 0.325 | 0.018 | yes |
| M6_calibrated | global_nonbr_no_abraom_near_m0 | 0.512 | 0.512 | yes |
| M7_scrambled | br_only_mcc_above_m0 | 0.417 | 0.279 | yes |
| M7_scrambled | abraom_common_benign_specificity_ge_0_95 | 0.903 | 0.950 | no |
| M7_scrambled | abraom_pathogenic_present_recall_not_collapsed_like_m6 | 0.252 | 0.018 | yes |
| M7_scrambled | global_nonbr_no_abraom_near_m0 | 0.512 | 0.512 | yes |
| M7_scrambled | negative_control_not_equal_or_better_than_real_abraom | 0.417 | 0.553 | yes |

## Calibration Metrics

| Model | Dataset | Brier | ECE | MCE | Slope | Intercept |
|---|---|---:|---:|---:|---:|---:|
| M0 | br_only | 0.164 | 0.054 | 0.156 | 1.138 | 0.210 |
| M0 | regional_benchmark_any | 0.156 | 0.074 | 0.184 | 1.655 | -0.538 |
| M0 | global_nonbr_no_abraom | 0.150 | 0.073 | 0.169 | 1.609 | -0.233 |
| M4 | br_only | 0.180 | 0.137 | 0.246 | 1.246 | 0.670 |
| M4 | regional_benchmark_any | 0.156 | 0.098 | 0.124 | 1.879 | 0.135 |
| M4 | global_nonbr_no_abraom | 0.158 | 0.118 | 0.182 | 1.852 | 0.413 |
| M5 | br_only | 0.113 | 0.072 | 0.289 | 1.589 | -0.445 |
| M5 | regional_benchmark_any | 0.138 | 0.108 | 0.309 | 1.791 | -1.315 |
| M5 | global_nonbr_no_abraom | 0.165 | 0.108 | 0.324 | 1.709 | -1.362 |
| M6 | br_only | 0.117 | 0.109 | 0.227 | 1.580 | 0.251 |
| M6 | regional_benchmark_any | 0.128 | 0.081 | 0.274 | 1.782 | -0.597 |
| M6 | global_nonbr_no_abraom | 0.158 | 0.072 | 0.281 | 1.754 | -0.738 |
| M5_calibrated | br_only | 0.133 | 0.124 | 0.208 | 1.633 | 0.280 |
| M5_calibrated | regional_benchmark_any | 0.136 | 0.092 | 0.166 | 1.811 | -0.375 |
| M5_calibrated | global_nonbr_no_abraom | 0.150 | 0.073 | 0.169 | 1.609 | -0.233 |
| M6_calibrated | br_only | 0.130 | 0.117 | 0.222 | 1.516 | 0.412 |
| M6_calibrated | regional_benchmark_any | 0.132 | 0.081 | 0.155 | 1.723 | -0.292 |
| M6_calibrated | global_nonbr_no_abraom | 0.150 | 0.073 | 0.169 | 1.609 | -0.233 |
| M7_scrambled | br_only | 0.146 | 0.086 | 0.159 | 1.491 | 0.168 |
| M7_scrambled | regional_benchmark_any | 0.143 | 0.082 | 0.175 | 1.759 | -0.440 |
| M7_scrambled | global_nonbr_no_abraom | 0.150 | 0.073 | 0.169 | 1.610 | -0.233 |

## Cluster Bootstrap

| Comparison | Dataset | Metric | Delta | 95% CI |
|---|---|---|---:|---:|
| M5_calibrated - M0 | br_only | mcc | 0.267 | [0.182, 0.357] |
| M5_calibrated - M0 | abraom_common_benign | specificity | 0.150 | [0.139, 0.161] |
| M5_calibrated - M0 | abraom_pathogenic_present | recall | -0.086 | [-0.138, -0.046] |
| M5_calibrated - M0 | global_nonbr_no_abraom | mcc | 0.000 | [0.000, 0.000] |
| M6_calibrated - M0 | br_only | mcc | 0.274 | [0.187, 0.366] |
| M6_calibrated - M0 | abraom_common_benign | specificity | 0.151 | [0.141, 0.163] |
| M6_calibrated - M0 | abraom_pathogenic_present | recall | -0.092 | [-0.143, -0.049] |
| M6_calibrated - M0 | global_nonbr_no_abraom | mcc | 0.000 | [0.000, 0.000] |
| M5_calibrated - M7_scrambled | br_only | mcc | 0.129 | [0.066, 0.191] |
| M5_calibrated - M7_scrambled | abraom_common_benign | specificity | 0.050 | [0.043, 0.056] |

## Error Analysis

- `M5_calibrated_false_benign`: `artifacts/clinvar_regional_calibration_v2_holdout_tuned/error_analysis/M5_calibrated.false_benign_abraom_pathogenic_present.csv`
- `M5_calibrated_false_pathogenic`: `artifacts/clinvar_regional_calibration_v2_holdout_tuned/error_analysis/M5_calibrated.false_pathogenic_abraom_common_benign.csv`
- `M6_calibrated_false_benign`: `artifacts/clinvar_regional_calibration_v2_holdout_tuned/error_analysis/M6_calibrated.false_benign_abraom_pathogenic_present.csv`
- `M6_calibrated_false_pathogenic`: `artifacts/clinvar_regional_calibration_v2_holdout_tuned/error_analysis/M6_calibrated.false_pathogenic_abraom_common_benign.csv`
- `M7_scrambled_false_benign`: `artifacts/clinvar_regional_calibration_v2_holdout_tuned/error_analysis/M7_scrambled.false_benign_abraom_pathogenic_present.csv`
- `M7_scrambled_false_pathogenic`: `artifacts/clinvar_regional_calibration_v2_holdout_tuned/error_analysis/M7_scrambled.false_pathogenic_abraom_common_benign.csv`

## Baseline v1

- CSV: `artifacts/clinvar_regional_comparison/m0_m4_m5_m6_regional_test_summary.csv`
- SHA256: `3ea660be25bfbd41f3acda8440c45b7d30bca96e2b4c3ab1dc102181e776f570`
- Conclusion frozen: ABRAOM improves regional specificity, but v1 frequency weighting is too strong.

## Sensitivity Panel

- `clinvar_plp_abraom_present`: available, rows=1596
- `known_brazilian_founder_variants`: missing, rows=0
- `manual_brazilian_plp_curation`: missing, rows=0

## Release Recommendation

Research-only. The holdout-tuned calibration passes the current slice criteria, but the curated founder/manual P/LP sentinel panel is still missing and the constrained rule is post-hoc rather than a trained dynamic fusion controller.
