# Regional Signal Validation Next Step

Generated at UTC: `2026-06-29T13:28:59.327140+00:00`

## Bottom Line

Decision: `do_not_train_next: prioritize manual critical-error review and external validation; 92 P/LP false benign remain, 60 high-priority.`.

This package audits the locked `M5_v3_safety` outputs. It is not a new training run and it does not use test errors to select a model.

## Locked Config

```json
{
  "discount_scale": 0.5,
  "max_discount": 0.5,
  "molecular_guard_threshold": 0.65,
  "guarded_max_discount": 0.0,
  "guard_score_floor": 0.35,
  "regional_threshold": 0.35,
  "global_threshold": 0.72
}
```

## Critical Error Audit

- P/LP ABRAOM-present false benign variants: `92`
- ABRAOM-common benign false pathogenic variants: `502`
- Derived review/sentinel rows: `757`

| Audit type | Category | Tier | n |
|---|---|---|---:|
| `false_benign_plp` | `threshold_near_miss` | `P1_manual_review` | 40 |
| `false_benign_plp` | `weak_molecular_signal` | `P2_model_diagnostic` | 32 |
| `false_benign_plp` | `weak_molecular_signal` | `P1_manual_review` | 15 |
| `false_benign_plp` | `threshold_near_miss` | `P0_manual_review` | 2 |
| `false_benign_plp` | `weak_molecular_signal` | `P0_manual_review` | 2 |
| `false_benign_plp` | `common_plp_recessive_or_founder_review` | `P1_manual_review` | 1 |
| `false_pathogenic_common_benign` | `threshold_near_miss` | `P3_low_priority` | 273 |
| `false_pathogenic_common_benign` | `regional_discount_insufficient_common` | `P3_low_priority` | 151 |
| `false_pathogenic_common_benign` | `regional_discount_insufficient_common` | `P2_model_diagnostic` | 46 |
| `false_pathogenic_common_benign` | `unresolved_false_pathogenic` | `P3_low_priority` | 12 |
| `false_pathogenic_common_benign` | `regional_discount_insufficient_common` | `P1_manual_review` | 9 |
| `false_pathogenic_common_benign` | `molecular_score_overdominant` | `P1_manual_review` | 6 |
| `false_pathogenic_common_benign` | `abraom_specific_common_benign` | `P2_model_diagnostic` | 2 |
| `false_pathogenic_common_benign` | `unresolved_false_pathogenic` | `P2_model_diagnostic` | 2 |
| `false_pathogenic_common_benign` | `abraom_specific_common_benign` | `P3_low_priority` | 1 |

## Top Error Genes

| Audit type | Gene | n | Mean priority | Median AF ABRAOM |
|---|---|---:|---:|---:|
| `false_benign_plp` | `FKTN` | 12 | 4.252 | 0.0083 |
| `false_benign_plp` | `PCSK9` | 8 | 4.692 | 0.0211 |
| `false_benign_plp` | `F2` | 3 | 5.162 | 0.0171 |
| `false_benign_plp` | `DYSF` | 3 | 5.043 | 0.0162 |
| `false_benign_plp` | `PAX6` | 3 | 4.189 | 0.0078 |
| `false_benign_plp` | `TMC1` | 3 | 4.426 | 0.0154 |
| `false_benign_plp` | `CFH` | 2 | 5.671 | 0.4626 |
| `false_benign_plp` | `HIF1A` | 2 | 5.548 | 0.0515 |
| `false_pathogenic_common_benign` | `MUC5B` | 12 | 3.160 | 0.0115 |
| `false_pathogenic_common_benign` | `ANKRD26` | 6 | 3.215 | 0.0132 |
| `false_pathogenic_common_benign` | `ADGRV1` | 6 | 3.259 | 0.0156 |
| `false_pathogenic_common_benign` | `VWF` | 6 | 3.220 | 0.0152 |
| `false_pathogenic_common_benign` | `MYOM1` | 5 | 3.712 | 0.0312 |
| `false_pathogenic_common_benign` | `PLCG2` | 5 | 3.593 | 0.0218 |
| `false_pathogenic_common_benign` | `COL4A1` | 5 | 3.320 | 0.0158 |
| `false_pathogenic_common_benign` | `VPS13D` | 5 | 3.217 | 0.0107 |

## Strong Negative Controls

| Dataset | Metric | Control | Real | Control mean | P95 | P(control >= real) | Changed discount | Interpretation |
|---|---|---|---:|---:|---:|---:|---:|---|
| `br_only` | `mcc` | `within_gene_af_bin` | 0.605 | 0.603 | 0.605 | 0.6337 | 0.599 | `interpretable` |
| `br_only` | `mcc` | `within_af_bin_variant_type` | 0.605 | 0.600 | 0.611 | 0.3960 | 0.970 | `interpretable` |
| `br_only` | `mcc` | `within_specificity_bin_variant_type` | 0.605 | 0.599 | 0.611 | 0.3267 | 0.963 | `interpretable` |
| `br_only` | `mcc` | `within_chromosome_af_bin` | 0.605 | 0.600 | 0.605 | 0.3663 | 0.875 | `interpretable` |
| `br_only` | `mcc` | `within_chromosome_af_bin_variant_type` | 0.605 | 0.601 | 0.605 | 0.4653 | 0.840 | `interpretable` |
| `br_only` | `mcc` | `within_gene_af_bin_variant_type` | 0.605 | 0.602 | 0.605 | 0.4554 | 0.541 | `interpretable` |
| `abraom_common_benign` | `specificity` | `within_gene_af_bin` | 0.959 | 0.971 | 0.972 | 1.0000 | 0.927 | `interpretable` |
| `abraom_common_benign` | `specificity` | `within_af_bin_variant_type` | 0.959 | 0.977 | 0.978 | 1.0000 | 0.995 | `interpretable` |
| `abraom_common_benign` | `specificity` | `within_specificity_bin_variant_type` | 0.959 | 0.975 | 0.976 | 1.0000 | 0.994 | `interpretable` |
| `abraom_common_benign` | `specificity` | `within_chromosome_af_bin` | 0.959 | 0.976 | 0.977 | 1.0000 | 0.993 | `interpretable` |
| `abraom_common_benign` | `specificity` | `within_chromosome_af_bin_variant_type` | 0.959 | 0.976 | 0.978 | 1.0000 | 0.993 | `interpretable` |
| `abraom_common_benign` | `specificity` | `within_gene_af_bin_variant_type` | 0.959 | 0.971 | 0.972 | 1.0000 | 0.926 | `interpretable` |
| `abraom_pathogenic_present` | `recall` | `within_gene_af_bin` | 0.436 | 0.445 | 0.454 | 1.0000 | 0.294 | `interpretable` |
| `abraom_pathogenic_present` | `recall` | `within_af_bin_variant_type` | 0.436 | 0.441 | 0.460 | 0.7525 | 0.976 | `interpretable` |
| `abraom_pathogenic_present` | `recall` | `within_specificity_bin_variant_type` | 0.436 | 0.434 | 0.454 | 0.5545 | 0.950 | `interpretable` |
| `abraom_pathogenic_present` | `recall` | `within_chromosome_af_bin` | 0.436 | 0.444 | 0.460 | 0.8515 | 0.733 | `interpretable` |
| `abraom_pathogenic_present` | `recall` | `within_chromosome_af_bin_variant_type` | 0.436 | 0.444 | 0.454 | 0.9010 | 0.736 | `interpretable` |
| `abraom_pathogenic_present` | `recall` | `within_gene_af_bin_variant_type` | 0.436 | 0.445 | 0.454 | 1.0000 | 0.300 | `interpretable` |
| `global_nonbr_no_abraom` | `mcc` | `within_gene_af_bin` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.889 | `interpretable` |
| `global_nonbr_no_abraom` | `mcc` | `within_af_bin_variant_type` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.979 | `interpretable` |
| `global_nonbr_no_abraom` | `mcc` | `within_specificity_bin_variant_type` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.979 | `interpretable` |
| `global_nonbr_no_abraom` | `mcc` | `within_chromosome_af_bin` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.968 | `interpretable` |
| `global_nonbr_no_abraom` | `mcc` | `within_chromosome_af_bin_variant_type` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.953 | `interpretable` |
| `global_nonbr_no_abraom` | `mcc` | `within_gene_af_bin_variant_type` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.839 | `interpretable` |

## Interpretation

- The remaining P/LP false benign variants are the immediate scientific and safety bottleneck.
- Strong controls that preserve AF-related strata are the key falsification test. If they match or exceed the real run, ABRAOM specificity is not proven.
- The derived panel is a review queue, not a clinically curated truth set.

## Key Artifacts

- `critical_error_audit/combined_manual_review_queue.csv`
- `critical_error_audit/false_benign_plp_review_queue.csv`
- `critical_error_audit/false_pathogenic_common_benign_review_queue.csv`
- `critical_error_category_summary.csv`
- `critical_error_gene_summary.csv`
- `derived_review_sentinel_panel.csv`
- `strong_negative_control_comparison.csv`
- `strong_negative_control_permutation_diagnostics.csv`
