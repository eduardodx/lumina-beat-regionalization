# M5_v3 Safety Calibration

Generated at UTC: `2026-06-29T12:57:07.650097+00:00`

## Bottom Line

Decision: `conditional_candidate: M5_v3 is safe versus M5_v2 but regional specificity is not fully falsified.`.

`M5_v3_safety` is a locked holdout-selected safety calibration over M5_v2 raw outputs. It adds a molecular guard so strong molecular evidence cannot be fully erased by an ABRAOM frequency discount.

## Selected Config

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

## Test Metrics

| Dataset | M0 | M7 | M5_v2 | M5_v3 | Metric |
|---|---:|---:|---:|---:|---|
| `br_only` | 0.279 | 0.301 | 0.605 | 0.605 | `mcc` |
| `abraom_common_benign` | 0.803 | 0.889 | 0.959 | 0.959 | `specificity` |
| `abraom_pathogenic_present` | 0.417 | 0.313 | 0.436 | 0.436 | `recall` |
| `global_nonbr_no_abraom` | 0.512 | 0.539 | 0.500 | 0.512 | `mcc` |
| `global_nonbr_no_abraom` | 0.544 | 0.699 | 0.844 | 0.788 | `specificity` |

## Sentinel Audit

| Audit | Dataset | n |
|---|---|---:|
| `plp_dangerous_regressions_vs_m5_v2` | `abraom_pathogenic_present` | 0 |
| `plp_rescued_vs_m5_v2` | `abraom_pathogenic_present` | 0 |
| `common_benign_fp_resolved_vs_m5_v2` | `abraom_common_benign` | 0 |
| `common_benign_new_fp_vs_m5_v2` | `abraom_common_benign` | 0 |

## Negative Controls

| Dataset | Metric | Control | Real | Control mean | P95 | Empirical P(control >= real) | Changed discount |
|---|---|---|---:|---:|---:|---:|---:|
| `br_only` | `mcc` | `global` | 0.605 | 0.572 | 0.584 | 0.0196 | 0.981 |
| `br_only` | `mcc` | `within_gene` | 0.605 | 0.588 | 0.596 | 0.0392 | 0.659 |
| `br_only` | `mcc` | `within_af_bin` | 0.605 | 0.600 | 0.611 | 0.3922 | 0.973 |
| `br_only` | `mcc` | `within_chromosome` | 0.605 | 0.575 | 0.590 | 0.0196 | 0.934 |
| `abraom_common_benign` | `specificity` | `global` | 0.959 | 0.977 | 0.977 | 1.0000 | 0.995 |
| `abraom_common_benign` | `specificity` | `within_gene` | 0.959 | 0.971 | 0.972 | 1.0000 | 0.927 |
| `abraom_common_benign` | `specificity` | `within_af_bin` | 0.959 | 0.976 | 0.978 | 1.0000 | 0.995 |
| `abraom_common_benign` | `specificity` | `within_chromosome` | 0.959 | 0.976 | 0.977 | 1.0000 | 0.993 |
| `abraom_pathogenic_present` | `recall` | `global` | 0.436 | 0.429 | 0.445 | 0.4902 | 0.979 |
| `abraom_pathogenic_present` | `recall` | `within_gene` | 0.436 | 0.441 | 0.451 | 0.8824 | 0.440 |
| `abraom_pathogenic_present` | `recall` | `within_af_bin` | 0.436 | 0.443 | 0.460 | 0.8431 | 0.975 |
| `abraom_pathogenic_present` | `recall` | `within_chromosome` | 0.436 | 0.431 | 0.451 | 0.4314 | 0.847 |
| `global_nonbr_no_abraom` | `mcc` | `global` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.980 |
| `global_nonbr_no_abraom` | `mcc` | `within_gene` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.888 |
| `global_nonbr_no_abraom` | `mcc` | `within_af_bin` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.980 |
| `global_nonbr_no_abraom` | `mcc` | `within_chromosome` | 0.512 | 0.512 | 0.512 | 1.0000 | 0.968 |

## Top Holdout Candidates

| Rank | Score | Pass | discount_scale | max_discount | guard_threshold | guarded_cap | threshold | br MCC | benign spec | P/LP recall | global MCC |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.627 | `False` | 0.500 | 0.500 | 0.650 | 0.000 | 0.350 | 0.568 | 0.950 | 0.428 | 0.570 |
| 2 | 3.627 | `False` | 0.500 | 0.500 | 0.650 | 0.100 | 0.350 | 0.568 | 0.950 | 0.428 | 0.570 |
| 3 | 3.627 | `False` | 0.500 | 0.500 | 0.650 | 0.250 | 0.350 | 0.568 | 0.950 | 0.428 | 0.570 |
| 4 | 3.627 | `False` | 0.500 | 0.500 | 0.650 | 0.500 | 0.350 | 0.568 | 0.950 | 0.428 | 0.570 |
| 5 | 3.627 | `False` | 0.500 | 0.500 | 0.750 | 0.000 | 0.350 | 0.568 | 0.950 | 0.428 | 0.570 |

## Interpretation

- If `empirical_p_control_ge_real` is high, the real ABRAOM discount is not clearly better than that scrambled control.
- The test sentinel audit is not used for selection; it only audits the locked holdout-selected configuration.
- A clinically relevant next step still requires external Brazilian P/LP curation; this artifact is scientific validation, not clinical validation.

## Key Artifacts

- `selected_config.json`
- `holdout_tuning_results.csv`
- `m5_v3_safety_regional_test_summary.csv`
- `m5_v3_negative_control_comparison.csv`
- `m5_v3_negative_control_runs.csv`
- `m5_v3_safety_sentinel_audit.csv`
- `sentinel_audit/`
