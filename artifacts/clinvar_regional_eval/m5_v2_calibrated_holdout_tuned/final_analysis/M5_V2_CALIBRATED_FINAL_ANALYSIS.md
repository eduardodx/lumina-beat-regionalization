# M5_v2 Calibrated Final Analysis

## Decision

`M5_v2_calibrated` remains the lead scientific candidate after holdout-tuned calibration.

## Key Metrics

| Metric | M0 | M5_calibrated | M7_scrambled | M5_v2_calibrated |
|---|---:|---:|---:|---:|
| br_only MCC | 0.279 | 0.546 | 0.417 | 0.605 |
| ABRAOM-common benign specificity | 0.803 | 0.952 | 0.903 | 0.959 |
| ABRAOM-present P/LP recall | 0.417 | 0.331 | 0.252 | 0.436 |
| global nonBR MCC | 0.512 | 0.512 | 0.512 | 0.500 |

## Bootstrap Deltas

| Comparison | Dataset | Metric | Delta | 95% CI |
|---|---|---|---:|---:|
| M5_v2_calibrated - M0 | br_only | mcc | 0.326 | [0.186, 0.471] |
| M5_v2_calibrated - M0 | abraom_common_benign | specificity | 0.156 | [0.143, 0.169] |
| M5_v2_calibrated - M0 | abraom_pathogenic_present | recall | 0.018 | [-0.054, 0.099] |
| M5_v2_calibrated - M0 | global_nonbr_no_abraom | mcc | -0.012 | [-0.063, 0.037] |
| M5_v2_calibrated - M0 | global_nonbr_no_abraom | specificity | 0.300 | [0.263, 0.340] |
| M5_v2_calibrated - M5_calibrated | br_only | mcc | 0.059 | [-0.024, 0.145] |
| M5_v2_calibrated - M5_calibrated | abraom_common_benign | specificity | 0.006 | [0.002, 0.010] |
| M5_v2_calibrated - M5_calibrated | abraom_pathogenic_present | recall | 0.104 | [0.042, 0.178] |
| M5_v2_calibrated - M7_scrambled | br_only | mcc | 0.188 | [0.066, 0.314] |
| M5_v2_calibrated - M7_scrambled | abraom_common_benign | specificity | 0.056 | [0.049, 0.063] |
| M5_v2_calibrated - M7_scrambled | abraom_pathogenic_present | recall | 0.184 | [0.110, 0.263] |
| M5_v2_calibrated - M5_v2 | abraom_pathogenic_present | recall | 0.380 | [0.302, 0.476] |

## Error Review

- False pathogenic ABRAOM-common benign table: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/error_analysis/M5_v2_calibrated.false_pathogenic_abraom_common_benign.csv`
- False benign ABRAOM-present P/LP table: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/error_analysis/M5_v2_calibrated.false_benign_abraom_pathogenic_present.csv`
- P/LP variants rescued from raw M5_v2: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/error_analysis/M5_v2_calibrated.rescued_from_raw_false_benign_abraom_pathogenic_present.csv`

## Diagnostics

- `dataset`: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/diagnostics/m5_v2_calibrated_by_dataset.csv`
- `specificity_bin`: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/diagnostics/m5_v2_calibrated_by_specificity_bin.csv`
- `af_abraom_bin`: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/diagnostics/m5_v2_calibrated_by_af_abraom_bin.csv`
- `discount_bin`: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/diagnostics/m5_v2_calibrated_by_discount_bin.csv`
- `top_error_genes`: `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/diagnostics/m5_v2_calibrated_top_error_genes.csv`

## Sensitivity Panel

- ClinVar P/LP ABRAOM-present panel rows: `1596`
- Test sensitivity rows: `163`
- Test recall: `0.436`
- Curated Brazilian founder/manual P/LP panels are still not available locally.

## Remaining Work

1. Add a curated Brazilian founder/P/LP panel and rerun this analysis.
2. Review the false-benign table before treating the candidate as stable.
3. Decide operating-point policy explicitly: Brazilian regional triage versus global molecular sensitivity mode.
