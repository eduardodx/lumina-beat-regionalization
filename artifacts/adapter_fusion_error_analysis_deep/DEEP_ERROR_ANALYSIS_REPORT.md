# Deep Error Analysis: ABRAOM Adapter Fusion

Generated at UTC: `2026-06-29T12:18:56.956323+00:00`

## Bottom Line

`M5_dynamic_gated` is not ready as the lead model. It is excellent at suppressing ABRAOM-common benign false positives, but the same regional discount mechanism creates a large false-benign safety failure on ABRAOM-present P/LP variants.

## Core Counts

| Model | ABRAOM-common benign FP | ABRAOM-common specificity | ABRAOM P/LP false benign | ABRAOM P/LP recall | br_only MCC | global MCC |
|---|---:|---:|---:|---:|---:|---:|
| `M0` | 2386 | 0.803 | 95 | 0.417 | 0.279 | 0.512 |
| `M4_dynamic_gated` | 1580 | 0.869 | 107 | 0.344 | 0.319 | 0.547 |
| `M5_dynamic_gated` | 26 | 0.998 | 157 | 0.037 | 0.666 | 0.400 |
| `M7_dynamic_scrambled` | 1345 | 0.889 | 112 | 0.313 | 0.301 | 0.539 |
| `M5_v2_calibrated` | 502 | 0.959 | 92 | 0.436 | 0.605 | 0.500 |

## Paired M5 Dynamic vs M5 v2

- `M5_dynamic_gated` rescues `476` ABRAOM-common benign false positives that `M5_v2_calibrated` still calls pathogenic.
- `M5_dynamic_gated` creates `65` dangerous false-benign regressions among ABRAOM-present P/LP variants that `M5_v2_calibrated` keeps positive.
- Top regression genes: `{'ALMS1': 3, 'EYS': 3, 'F13B': 2, 'FKTN': 2, 'JUP': 2, 'MUC5B': 2, 'NLRP3': 2, 'VWF': 2, 'TNFRSF13C': 2, 'NPHP4': 2}`.
- Top rescued benign genes: `{'MUC5B': 12, 'ANKRD26': 6, 'ADGRV1': 6, 'VWF': 6, 'CPLANE1': 5, 'COL4A1': 5, 'COL12A1': 5, 'VPS13D': 5, 'ADAMTS10': 5, 'ALMS1': 4}`.

## Statistical Checks

Cluster bootstrap by `GeneSymbol` is written to `paired_bootstrap_by_gene.csv`. Treat wide CIs in small P/LP slices as uncertainty, not model victory.

## Post-hoc Dynamic Calibration Probe

No simple discount-scale/threshold configuration satisfied all hard constraints: ABRAOM-common specificity >= 0.95, ABRAOM P/LP recall near M0, and global MCC floor >= 0.46.

Best near miss when all constraints cannot be met:

| discount_scale | threshold | constraint_gap | br_only MCC | ABRAOM-common specificity | ABRAOM P/LP recall | global MCC | global specificity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.425 | 0.296 | 0.599 | 0.967 | 0.368 | 0.164 | 0.042 |

The near misses recover P/LP recall and keep ABRAOM-common specificity, but global nonBR performance collapses. This means threshold-only rescue is not enough.

## Failure Categories

| File | Category | n | fraction |
|---|---|---:|---:|
| `M5_dynamic.dangerous_false_benign_regressions_vs_M5_v2.csv` | `unresolved_false_benign` | 44 | 0.677 |
| `M5_dynamic.dangerous_false_benign_regressions_vs_M5_v2.csv` | `common_plp_recessive_or_founder_review` | 11 | 0.169 |
| `M5_dynamic.dangerous_false_benign_regressions_vs_M5_v2.csv` | `threshold_borderline` | 7 | 0.108 |
| `M5_dynamic.dangerous_false_benign_regressions_vs_M5_v2.csv` | `regional_af_over_suppression` | 3 | 0.046 |
| `M5_dynamic_gated.false_benign_abraom_pathogenic_present.csv` | `unresolved_false_benign` | 67 | 0.427 |
| `M5_dynamic_gated.false_benign_abraom_pathogenic_present.csv` | `weak_molecular_signal` | 41 | 0.261 |
| `M5_dynamic_gated.false_benign_abraom_pathogenic_present.csv` | `regional_af_over_suppression` | 23 | 0.146 |
| `M5_dynamic_gated.false_benign_abraom_pathogenic_present.csv` | `common_plp_recessive_or_founder_review` | 19 | 0.121 |
| `M5_dynamic_gated.false_benign_abraom_pathogenic_present.csv` | `threshold_borderline` | 7 | 0.045 |
| `M5_dynamic_gated.false_pathogenic_abraom_common_benign.csv` | `frequency_discount_insufficient_common` | 9 | 0.346 |
| `M5_dynamic_gated.false_pathogenic_abraom_common_benign.csv` | `threshold_borderline` | 9 | 0.346 |
| `M5_dynamic_gated.false_pathogenic_abraom_common_benign.csv` | `molecular_score_overdominant` | 4 | 0.154 |
| `M5_dynamic_gated.false_pathogenic_abraom_common_benign.csv` | `unresolved_false_pathogenic` | 4 | 0.154 |
| `M5_v2_calibrated.false_benign_abraom_pathogenic_present.csv` | `weak_molecular_signal` | 55 | 0.598 |
| `M5_v2_calibrated.false_benign_abraom_pathogenic_present.csv` | `threshold_borderline` | 24 | 0.261 |
| `M5_v2_calibrated.false_benign_abraom_pathogenic_present.csv` | `common_plp_recessive_or_founder_review` | 10 | 0.109 |
| `M5_v2_calibrated.false_benign_abraom_pathogenic_present.csv` | `regional_af_over_suppression` | 3 | 0.033 |
| `M5_v2_calibrated.false_pathogenic_abraom_common_benign.csv` | `frequency_discount_insufficient_common` | 256 | 0.510 |
| `M5_v2_calibrated.false_pathogenic_abraom_common_benign.csv` | `threshold_borderline` | 121 | 0.241 |
| `M5_v2_calibrated.false_pathogenic_abraom_common_benign.csv` | `unresolved_false_pathogenic` | 119 | 0.237 |
| `M5_v2_calibrated.false_pathogenic_abraom_common_benign.csv` | `molecular_score_overdominant` | 6 | 0.012 |

## Recommended Action

1. Keep `M5_v2_calibrated` as the current safest candidate.
2. Use the dangerous regression table as the P/LP sentinel target for the next dynamic calibration.
3. Train or calibrate `M5_dynamic_v2_safety` with an explicit molecular guard and a stricter cap on regional discount.
4. Do not optimize only ABRAOM-common specificity; require P/LP recall and global non-inferiority constraints at selection time.

## Key Artifacts

- `M0_false_benign`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M0.false_benign_abraom_pathogenic_present.csv`
- `M0_false_pathogenic`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M0.false_pathogenic_abraom_common_benign.csv`
- `M2_gnomad_only_false_benign`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M2_gnomad_only.false_benign_abraom_pathogenic_present.csv`
- `M2_gnomad_only_false_pathogenic`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M2_gnomad_only.false_pathogenic_abraom_common_benign.csv`
- `M4_dynamic_gated_false_benign`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M4_dynamic_gated.false_benign_abraom_pathogenic_present.csv`
- `M4_dynamic_gated_false_pathogenic`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M4_dynamic_gated.false_pathogenic_abraom_common_benign.csv`
- `M5_dynamic_gated_false_benign`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M5_dynamic_gated.false_benign_abraom_pathogenic_present.csv`
- `M5_dynamic_gated_false_pathogenic`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M5_dynamic_gated.false_pathogenic_abraom_common_benign.csv`
- `M5_v2_calibrated_false_benign`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M5_v2_calibrated.false_benign_abraom_pathogenic_present.csv`
- `M5_v2_calibrated_false_pathogenic`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M5_v2_calibrated.false_pathogenic_abraom_common_benign.csv`
- `M7_dynamic_scrambled_false_benign`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M7_dynamic_scrambled.false_benign_abraom_pathogenic_present.csv`
- `M7_dynamic_scrambled_false_pathogenic`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M7_dynamic_scrambled.false_pathogenic_abraom_common_benign.csv`
- `m5_dynamic_dangerous_false_benign_regressions_vs_m5_v2`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M5_dynamic.dangerous_false_benign_regressions_vs_M5_v2.csv`
- `m5_dynamic_rescued_false_pathogenic_vs_m5_v2`: `artifacts/adapter_fusion_error_analysis_deep/variant_error_tables/M5_dynamic.rescued_false_pathogenic_vs_M5_v2.csv`
- `error_counts_by_model_dataset.csv`
- `paired_prediction_transitions.csv`
- `paired_bootstrap_by_gene.csv`
- `error_rates_by_group.csv`
- `failure_category_summary.csv`
- `m5_dynamic_discount_scale_grid.csv`
- `m5_dynamic_discount_scale_candidates.csv`
- `m5_dynamic_discount_scale_near_misses.csv`
