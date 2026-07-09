# Dynamic Adapter-Fusion Regionalization Report

Generated at UTC: `2026-06-29T11:58:13.610214+00:00`

## Decision

`needs_calibrated_safety_before_claim`

The dynamic matrix is complete for M2/M4/M5/M7, but the lead dynamic M5 is not acceptable as-is because it suppresses ABRAOM-present P/LP recall despite excellent ABRAOM-common benign specificity.

## Key Test Metrics

| Model | br_only MCC | ABRAOM-common benign specificity | ABRAOM-present P/LP recall | global nonBR MCC |
|---|---:|---:|---:|---:|
| `M0` | 0.279 | 0.803 | 0.417 | 0.512 |
| `M2_gnomad_only` | 0.309 | 0.884 | 0.319 | 0.537 |
| `M4_dynamic_gated` | 0.319 | 0.869 | 0.344 | 0.547 |
| `M5_dynamic_gated` | 0.666 | 0.998 | 0.037 | 0.400 |
| `M7_dynamic_scrambled` | 0.301 | 0.889 | 0.313 | 0.539 |
| `M5_v2_calibrated` | 0.605 | 0.959 | 0.436 | 0.500 |

## Interpretation

- `M4_dynamic_gated` gives a modest Brazilian gain over M0 and stays more balanced than static M4.
- `M5_dynamic_gated` gives the strongest `br_only` MCC and near-perfect suppression of ABRAOM-common benign false positives, but its P/LP ABRAOM-present recall collapses.
- `M2_gnomAD_only` is close to `M4_dynamic_gated`, so ABRAOM-specific value is present but modest in this dynamic setup.
- `M7_dynamic_scrambled` does not beat the real ABRAOM dynamic setup on `br_only` MCC, which supports that the ABRAOM signal is not just extra parameters.
- The safer current recommendation remains `M5_v2_calibrated` or a new dynamic calibration using the same bounded/sentinel-protected logic.

## Artifacts

- Summary CSV: `artifacts/adapter_fusion_blueprint_completion/dynamic_fusion_regional_summary.csv`
- Criteria CSV: `artifacts/adapter_fusion_blueprint_completion/dynamic_fusion_success_criteria.csv`
