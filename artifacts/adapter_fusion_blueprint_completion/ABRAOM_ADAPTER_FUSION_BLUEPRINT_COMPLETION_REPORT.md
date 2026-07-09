# ABRAOM Adapter-Fusion Blueprint Completion Report

Generated at UTC: `2026-06-29T11:59:01.574406+00:00`

Decision: `supported`

## Checklist

| Item | Status | Note |
|---|---|---|
| `data_manifest_and_slices` | `complete` | Baseline datasets present: ['abraom_common_benign', 'abraom_pathogenic_common', 'abraom_pathogenic_present', 'br_any', 'br_only', 'global_nonbr_no_abraom', 'nonbr_only', 'regional_benchmark_any'] |
| `abraom_frequency_alignment` | `complete` | artifacts/abraom_frequency_adapter/alignment_comparison/ABRAOM_FREQUENCY_ALIGNMENT_REPORT.md |
| `baseline_ablation_table` | `complete` | Baseline models present: ['M0', 'M4', 'M5', 'M5_calibrated', 'M5_v2', 'M5_v2_calibrated', 'M6', 'M6_calibrated', 'M7_scrambled'] |
| `m5_v2_calibrated_final_analysis` | `complete` | artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis/M5_V2_CALIBRATED_FINAL_ANALYSIS.md |
| `sentinel_panel` | `complete` | artifacts/clinvar_regional_eval/brazilian_plp_sentinel_panel/manifest.json |
| `dynamic_gate_experiments` | `complete` | Dynamic models present: ['M0', 'M2_gnomad_only', 'M4_dynamic_gated', 'M5_dynamic_gated', 'M5_v2_calibrated', 'M7_dynamic_scrambled'] |
| `dynamic_gate_diagnostics` | `complete` | Requires dynamic prediction summaries with alpha/gate columns. |

## Key Metrics

| Metric | Value |
|---|---:|
| `M0_br_only_mcc` | 0.279 |
| `M5_v2_calibrated_br_only_mcc` | 0.605 |
| `M0_abraom_common_benign_specificity` | 0.803 |
| `M5_v2_calibrated_abraom_common_benign_specificity` | 0.959 |
| `M0_abraom_pathogenic_present_recall` | 0.417 |
| `M5_v2_calibrated_abraom_pathogenic_present_recall` | 0.436 |

## Interpretation

All blueprint research gates are represented by artifacts.
