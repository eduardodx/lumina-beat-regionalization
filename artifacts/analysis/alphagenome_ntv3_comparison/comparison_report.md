# AlphaGenome, Lumina and NTv3 comparison

## Scope

This report compiles existing artifacts into a reproducible comparison packet. It does not rerun model inference or training.

## Track audit

- Track: `ENCSR814RGG`
- NTv3 label: `ATAC_seq (ureter)` / `ATAC-seq`
- AlphaGenome request: `ATAC` / `UBERON:0000056` (ureter, ATAC-seq)
- Overlap class: `exact_assay_tissue_match`
- Rationale: AlphaGenome metadata matches the NTv3 assay and the NTv3 public track label tissue.

## Canonical scores

| Rank | Model | Regime | Adaptation | Pearson | Runs | Notes |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | NTv3 650M (pos) | public NTv3 benchmark model | public benchmark score averaged across runs | 0.798486 | 3 | run_ids=NTVGEN-8576,NTVGEN-8789,NTVGEN-8782 |
| 2 | NTv3 650M (pre) | public NTv3 benchmark model | public benchmark score averaged across runs | 0.750030 | 3 | run_ids=NTVGEN-8578,NTVGEN-8761,NTVGEN-8779 |
| 3 | NTv3 100M (pre) | public NTv3 benchmark model | public benchmark score averaged across runs | 0.714466 | 3 | run_ids=NTVGEN-8577,NTVGEN-8785,NTVGEN-8775 |
| 4 | AlphaGenome | frozen AlphaGenome plus NTv3 readout | ridge readout selected on val (readout_pearson_ntv3_scaled_lambda_0); features=intercept,alpha_raw,sqrt_alpha,log1p_alpha,mean_25,mean_101,mean_501 | 0.698436 | 1 | Head-only calibration evidence; backbone remains frozen. |
| 5 | Lumina beat-v7 context-pyramid | Lumina NTv3 benchmark fine-tuning | benchmark fine-tuned model/head | 0.672015 | 1 | Best honest same-track Lumina score from the existing Lumina-vs-NTv3 delta artifact; delta_vs_ntv3_8m=0.0052545568550821375. |
| 6 | NTv3 8M (pre) | public NTv3 benchmark model | public benchmark score averaged across runs | 0.666760 | 3 | run_ids=NTVGEN-8600,NTVGEN-8778,NTVGEN-8765 |
| 7 | AlphaGenome | native supervised genome-track model | zero-shot native ATAC output, no NTv3 fitting | 0.296621 | 1 | Contextual baseline; not equivalent to NTv3/Lumina training regime. |

## Scientific interpretation

- The primary benchmark-comparable result is Lumina vs NTv3, because both are evaluated inside the NTv3 protocol.
- AlphaGenome zero-shot measures native alignment between its requested ATAC output and the NTv3 target.
- AlphaGenome frozen + readout measures whether its frozen predictions carry calibratable signal for the NTv3 target; it should be presented as a head-only adaptation, not as an identical training regime to Lumina.

## Reproducibility metadata

- Generated at UTC: `2026-06-07T20:24:52.598605+00:00`
- Output directory: `artifacts/analysis/alphagenome_ntv3_comparison`
- Canonical CSV: `artifacts/analysis/alphagenome_ntv3_comparison/canonical_results.csv`
- Track audit CSV: `artifacts/analysis/alphagenome_ntv3_comparison/track_audit.csv`
