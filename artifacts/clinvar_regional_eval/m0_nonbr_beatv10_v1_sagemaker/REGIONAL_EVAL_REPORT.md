# M0 Regional Evaluation Report

## Run

- Job: `clinvar-eval-m0-nonbr-beatv10-v1-3971be-20260621220228`
- Status: `Completed`
- Training/evaluation time: 8580 seconds
- Model evaluated: M0 `beat-v10` fine-tuned on `nonbr_only`
- Evaluation mode: no retraining
- Threshold: `0.4311051070690155` from M0 validation
- Local artifacts: `artifacts/clinvar_regional_eval/m0_nonbr_beatv10_v1_sagemaker/`

## Main Test Metrics

| slice | n | positives | negatives | AUROC | AUPRC | MCC | F1 | recall | specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nonbr_only` | 2547 | 1355 | 1192 | 0.879562 | 0.890059 | 0.578879 | 0.818512 | 0.913653 | 0.637584 |
| `br_only` | 504 | 377 | 127 | 0.744606 | 0.896384 | 0.279113 | 0.853629 | 0.920424 | 0.299213 |
| `br_any` | 564 | 427 | 137 | 0.743466 | 0.900532 | 0.263190 | 0.852747 | 0.908665 | 0.306569 |
| `mixed_br_nonbr` | 60 | 50 | 10 | 0.784000 | 0.953598 | 0.199016 | 0.845361 | 0.820000 | 0.400000 |
| `regional_benchmark_any` | 3111 | 1782 | 1329 | 0.865392 | 0.891975 | 0.552832 | 0.826429 | 0.912458 | 0.603461 |
| `global_nonbr_no_abraom` | 1989 | 1348 | 641 | 0.849439 | 0.919466 | 0.512205 | 0.859924 | 0.917656 | 0.544462 |

## ABRAOM Controls

| slice | n | positives | negatives | AUROC | MCC | recall | specificity | mean probability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `abraom_common` test | 12159 | 60 | 12099 | 0.703711 | 0.029797 | 0.366667 | 0.802794 | n/a |
| `abraom_common_benign` test | 12099 | 0 | 12099 | n/a | 0.000000 | 0.000000 | 0.802794 | 0.285043 |
| `abraom_pathogenic_present` test | 163 | 163 | 0 | n/a | 0.000000 | 0.417178 | 0.000000 | 0.430461 |
| `abraom_pathogenic_common` test | 60 | 60 | 0 | n/a | 0.000000 | 0.366667 | 0.000000 | 0.402844 |

## Facts

- The M0 model preserved high recall on Brazilian slices: `br_only` test recall `0.920424`.
- The M0 model had lower specificity on Brazilian slices: `br_only` test specificity `0.299213` versus `nonbr_only` test specificity `0.637584`.
- Ranking performance dropped from `nonbr_only` test AUROC `0.879562` to `br_only` test AUROC `0.744606`.
- `abraom_common_benign` test had `2386 / 12099` benign variants above the M0 validation threshold.
- `abraom_pathogenic_present` test had `68 / 163` pathogenic variants above the M0 validation threshold.
- `abraom_pathogenic_common` test had `22 / 60` pathogenic variants above the M0 validation threshold.

## Files

- `regional_eval_summary.json`
- `regional_eval_metrics.json`
- `regional_eval_summary.parquet`
- `*.predictions.parquet` for every evaluated slice and split.
