# M7 Scrambled-Control Analysis

Generated at UTC: `2026-06-29T12:40:13.787562+00:00`

## Bottom Line

`M7_dynamic_scrambled` should not be advanced as a candidate model, but it is a useful negative control. It performs close to `M4_dynamic_gated`, which weakens any claim that the dynamic-gated M4 result alone proves specific ABRAOM biology. `M5_v2_calibrated` remains the lead because it beats the scrambled control on the regional safety targets that matter most.

## Core Metrics

| Dataset | Model | MCC | Recall | Specificity | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| `abraom_common_benign` | `M2_gnomad_only` | 0.000 | NA | 0.884 | 1404 | 0 |
| `abraom_common_benign` | `M4_dynamic_gated` | 0.000 | NA | 0.869 | 1580 | 0 |
| `abraom_common_benign` | `M5_v2_calibrated` | 0.000 | NA | 0.959 | 502 | 0 |
| `abraom_common_benign` | `M7_dynamic_scrambled` | 0.000 | NA | 0.889 | 1345 | 0 |
| `abraom_pathogenic_present` | `M2_gnomad_only` | 0.000 | 0.319 | NA | 0 | 111 |
| `abraom_pathogenic_present` | `M4_dynamic_gated` | 0.000 | 0.344 | NA | 0 | 107 |
| `abraom_pathogenic_present` | `M5_v2_calibrated` | 0.000 | 0.436 | NA | 0 | 92 |
| `abraom_pathogenic_present` | `M7_dynamic_scrambled` | 0.000 | 0.313 | NA | 0 | 112 |
| `br_only` | `M2_gnomad_only` | 0.309 | 0.844 | 0.457 | 69 | 59 |
| `br_only` | `M4_dynamic_gated` | 0.319 | 0.875 | 0.417 | 74 | 47 |
| `br_only` | `M5_v2_calibrated` | 0.605 | 0.995 | 0.457 | 69 | 2 |
| `br_only` | `M7_dynamic_scrambled` | 0.301 | 0.838 | 0.457 | 69 | 61 |
| `global_nonbr_no_abraom` | `M2_gnomad_only` | 0.537 | 0.855 | 0.680 | 205 | 196 |
| `global_nonbr_no_abraom` | `M4_dynamic_gated` | 0.547 | 0.875 | 0.660 | 218 | 168 |
| `global_nonbr_no_abraom` | `M5_v2_calibrated` | 0.500 | 0.691 | 0.844 | 100 | 417 |
| `global_nonbr_no_abraom` | `M7_dynamic_scrambled` | 0.539 | 0.843 | 0.699 | 193 | 211 |

## Paired Control Tests

| Dataset | Contrast | Base | Compare | Base-only correct | Compare-only correct | Delta MCC | Delta recall | Delta specificity | McNemar p |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `br_only` | `real_abraom_vs_scrambled` | `M7_dynamic_scrambled` | `M4_dynamic_gated` | 5 | 14 | 0.017 | 0.037 | -0.039 | 0.0636 |
| `br_only` | `m5_v2_vs_scrambled` | `M7_dynamic_scrambled` | `M5_v2_calibrated` | 26 | 85 | 0.304 | 0.156 | 0.000 | 0.0000 |
| `abraom_common_benign` | `real_abraom_vs_scrambled` | `M7_dynamic_scrambled` | `M4_dynamic_gated` | 242 | 7 | 0.000 | NA | -0.019 | 0.0000 |
| `abraom_common_benign` | `m5_v2_vs_scrambled` | `M7_dynamic_scrambled` | `M5_v2_calibrated` | 165 | 1008 | 0.000 | NA | 0.070 | 0.0000 |
| `abraom_pathogenic_present` | `real_abraom_vs_scrambled` | `M7_dynamic_scrambled` | `M4_dynamic_gated` | 0 | 5 | 0.000 | 0.031 | NA | 0.0625 |
| `abraom_pathogenic_present` | `m5_v2_vs_scrambled` | `M7_dynamic_scrambled` | `M5_v2_calibrated` | 8 | 28 | 0.000 | 0.123 | NA | 0.0012 |
| `global_nonbr_no_abraom` | `real_abraom_vs_scrambled` | `M7_dynamic_scrambled` | `M4_dynamic_gated` | 26 | 44 | 0.009 | 0.032 | -0.039 | 0.0414 |
| `global_nonbr_no_abraom` | `m5_v2_vs_scrambled` | `M7_dynamic_scrambled` | `M5_v2_calibrated` | 207 | 94 | -0.038 | -0.153 | 0.145 | 0.0000 |

## Gene-Cluster Bootstrap

| Base | Compare | Dataset | Metric | Delta | 95% CI |
|---|---|---|---|---:|---:|
| `M7_dynamic_scrambled` | `M4_dynamic_gated` | `br_only` | `mcc` | 0.017 | [-0.025, 0.057] |
| `M7_dynamic_scrambled` | `M4_dynamic_gated` | `abraom_common_benign` | `specificity` | -0.019 | [-0.022, -0.017] |
| `M7_dynamic_scrambled` | `M4_dynamic_gated` | `abraom_pathogenic_present` | `recall` | 0.031 | [0.006, 0.059] |
| `M7_dynamic_scrambled` | `M4_dynamic_gated` | `global_nonbr_no_abraom` | `mcc` | 0.009 | [-0.012, 0.031] |
| `M2_gnomad_only` | `M4_dynamic_gated` | `br_only` | `mcc` | 0.010 | [-0.031, 0.047] |
| `M2_gnomad_only` | `M7_dynamic_scrambled` | `br_only` | `mcc` | -0.007 | [-0.033, 0.019] |
| `M7_dynamic_scrambled` | `M5_v2_calibrated` | `br_only` | `mcc` | 0.304 | [0.178, 0.436] |
| `M7_dynamic_scrambled` | `M5_v2_calibrated` | `abraom_common_benign` | `specificity` | 0.070 | [0.061, 0.078] |
| `M7_dynamic_scrambled` | `M5_v2_calibrated` | `abraom_pathogenic_present` | `recall` | 0.123 | [0.057, 0.196] |
| `M7_dynamic_scrambled` | `M5_v2_calibrated` | `global_nonbr_no_abraom` | `mcc` | -0.038 | [-0.071, -0.005] |

## Gate Behavior

On `br_only`, both dynamic gates remain near maximum entropy: M4 median entropy `0.692`, M7 median entropy `0.691`. This suggests weak adapter specialization rather than confident routing to a regional adapter.
M4 and M7 scores on `br_only` are highly similar: correlation `0.997`, prediction agreement `0.962`.

## Interpretations

- `br_only`: Real ABRAOM did not clearly beat scrambled control for this metric.
- `abraom_common_benign`: Scrambled control matched or beat real ABRAOM on the selected metric.
- `abraom_pathogenic_present`: Real ABRAOM beat scrambled control on the selected metric.
- `global_nonbr_no_abraom`: Real ABRAOM did not clearly beat scrambled control for this metric.

## Recommendation

1. Keep `M7_dynamic_scrambled` as a falsification/control model, not as a candidate.
2. Do not use M4-vs-M7 alone to claim strong ABRAOM-specific learning; the difference is too small and often not robust.
3. Use M7 as a required comparator for the next `M5_v3_safety` run.
4. Require the next model to beat M7 on `br_only` MCC, `abraom_common_benign` specificity, and `abraom_pathogenic_present` recall, while staying close to M0/M7 on global nonBR performance.
5. Add stronger negative controls next: scramble within gene, within AF bin, and within chromosome to preserve confounders while breaking variant-frequency identity.

## Key Artifacts

- `m7_control_metrics.csv`
- `m7_pairwise_transitions.csv`
- `m7_gene_cluster_bootstrap.csv`
- `m7_gate_behavior.csv`
- `m7_score_similarity.csv`
- `m7_discordance_group_summary.csv`
- `m7_top_discordant_genes.csv`
- `discordant_variant_tables/`
- Discordant variant table manifest rows: `38`
