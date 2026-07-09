# Public ABRAOM Validation Package

Generated at UTC: `2026-07-01T21:40:59.896541+00:00`

## Bottom Line

This package is a reproducible public-evidence review scaffold, not a completed clinical curation.
Of `75` high-priority variants, `1` have local public ClinVar evidence supporting the current label, `0` have completed manual public curation supporting the current label, `4` have public/manual evidence conflicting with the current label, and `70` need public lookup.

## Evidence Status

| Status | n |
|---|---:|
| `needs_public_lookup` | 70 |
| `public_label_conflict` | 4 |
| `local_public_supports_label` | 1 |

## Evidence Decisions

| Decision | n |
|---|---:|
| `no_local_variation_id_or_significance` | 70 |
| `public_benign_conflicts_with_plp_label` | 4 |
| `supports_plp_sentinel` | 1 |

## High-Priority Breakdown

| Audit | Tier | Status | n |
|---|---|---|---:|
| `false_benign_plp` | `P0_manual_review` | `needs_public_lookup` | 2 |
| `false_benign_plp` | `P0_manual_review` | `public_label_conflict` | 2 |
| `false_benign_plp` | `P1_manual_review` | `local_public_supports_label` | 1 |
| `false_benign_plp` | `P1_manual_review` | `needs_public_lookup` | 53 |
| `false_benign_plp` | `P1_manual_review` | `public_label_conflict` | 2 |
| `false_pathogenic_common_benign` | `P1_manual_review` | `needs_public_lookup` | 15 |

## Panel Metrics

| Panel | Model | n | Recall | Specificity | MCC | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| `high_priority_review_queue` | `M0` | 75 | 0.250 | 0.133 | -0.510 | 13 | 45 |
| `high_priority_review_queue` | `M7_dynamic_scrambled` | 75 | 0.117 | 0.133 | -0.678 | 13 | 53 |
| `high_priority_review_queue` | `M5_v2_calibrated` | 75 | 0.000 | 0.000 | -1.000 | 15 | 60 |
| `high_priority_review_queue` | `M5_v3_safety` | 75 | 0.000 | 0.000 | -1.000 | 15 | 60 |
| `local_public_evidence_available` | `M0` | 5 | 0.000 | 0.000 | 0.000 | 0 | 5 |
| `local_public_evidence_available` | `M7_dynamic_scrambled` | 5 | 0.000 | 0.000 | 0.000 | 0 | 5 |
| `local_public_evidence_available` | `M5_v2_calibrated` | 5 | 0.000 | 0.000 | 0.000 | 0 | 5 |
| `local_public_evidence_available` | `M5_v3_safety` | 5 | 0.000 | 0.000 | 0.000 | 0 | 5 |
| `local_public_supports_label` | `M0` | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 |
| `local_public_supports_label` | `M7_dynamic_scrambled` | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 |
| `local_public_supports_label` | `M5_v2_calibrated` | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 |
| `local_public_supports_label` | `M5_v3_safety` | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 |
| `curated_or_public_supports_label` | `M0` | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 |
| `curated_or_public_supports_label` | `M7_dynamic_scrambled` | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 |
| `curated_or_public_supports_label` | `M5_v2_calibrated` | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 |
| `curated_or_public_supports_label` | `M5_v3_safety` | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 |
| `public_label_conflict` | `M0` | 4 | 0.000 | 0.000 | 0.000 | 0 | 4 |
| `public_label_conflict` | `M7_dynamic_scrambled` | 4 | 0.000 | 0.000 | 0.000 | 0 | 4 |
| `public_label_conflict` | `M5_v2_calibrated` | 4 | 0.000 | 0.000 | 0.000 | 0 | 4 |
| `public_label_conflict` | `M5_v3_safety` | 4 | 0.000 | 0.000 | 0.000 | 0 | 4 |
| `pending_public_lookup` | `M0` | 70 | 0.273 | 0.133 | -0.497 | 13 | 40 |
| `pending_public_lookup` | `M7_dynamic_scrambled` | 70 | 0.127 | 0.133 | -0.672 | 13 | 48 |
| `pending_public_lookup` | `M5_v2_calibrated` | 70 | 0.000 | 0.000 | -1.000 | 15 | 55 |
| `pending_public_lookup` | `M5_v3_safety` | 70 | 0.000 | 0.000 | -1.000 | 15 | 55 |
| `plp_high_priority` | `M0` | 60 | 0.250 | 0.000 | 0.000 | 0 | 45 |
| `plp_high_priority` | `M7_dynamic_scrambled` | 60 | 0.117 | 0.000 | 0.000 | 0 | 53 |
| `plp_high_priority` | `M5_v2_calibrated` | 60 | 0.000 | 0.000 | 0.000 | 0 | 60 |
| `plp_high_priority` | `M5_v3_safety` | 60 | 0.000 | 0.000 | 0.000 | 0 | 60 |
| `common_benign_high_priority` | `M0` | 15 | 0.000 | 0.133 | 0.000 | 13 | 0 |
| `common_benign_high_priority` | `M7_dynamic_scrambled` | 15 | 0.000 | 0.133 | 0.000 | 13 | 0 |
| `common_benign_high_priority` | `M5_v2_calibrated` | 15 | 0.000 | 0.000 | 0.000 | 15 | 0 |
| `common_benign_high_priority` | `M5_v3_safety` | 15 | 0.000 | 0.000 | 0.000 | 15 | 0 |

## Interpretation

- The current blocker is evidence resolution, not architecture.
- Rows marked `public_label_conflict` should be adjudicated before being used as a sentinel truth set.
- Rows marked `needs_public_lookup` require ClinVar/PMID/source entry before any new training or claim of regional specificity.
- This is a stress/error panel, not a prevalence-balanced benchmark; poor metrics here identify failure modes rather than global model quality.
- The ready subset is currently too small to justify a new model decision by itself.

## Key Artifacts

- `public_evidence_review_queue.tsv`
- `public_curated_sentinel_panel.csv`
- `public_validation_metrics_by_panel.csv`
- `PUBLIC_ABRAOM_VALIDATION_DECISION_REPORT.md`
