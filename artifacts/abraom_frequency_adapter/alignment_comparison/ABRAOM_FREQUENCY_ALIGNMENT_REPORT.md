# ABRAOM Frequency Alignment Report

Generated: 2026-06-21

## Objective

Validate whether the Brazilian population adapter `A_BR` learned ABRAOM allele-frequency structure before using it in pathogenicity or adapter-fusion experiments.

## Runs

All runs used BEAT-v10 as frozen base, LoRA rank 8, context size 1024, 100k balanced-AF training rows, 5k validation rows, 5k test rows, and the same deterministic seed.

| Run | Target used for training | Metric target |
|---|---|---|
| `A_BR balanced` | `af_abraom` | `af_abraom` |
| `scrambled balanced` | `scrambled_af_abraom` | `af_abraom` |
| `A_gnomAD balanced` | `af_gnomad` | `af_abraom` |

## Test Metrics

| Model | NLL | Brier | MAE | Spearman |
|---|---:|---:|---:|---:|
| `A_BR balanced` | 0.513922 | 0.068556 | 0.204165 | 0.103602 |
| `scrambled balanced` | 0.523908 | 0.071624 | 0.193024 | 0.001261 |
| `A_gnomAD balanced` | 0.534033 | 0.071817 | 0.199224 | 0.117521 |

## Direct Comparisons On Test

| Comparison | Delta NLL | Delta Brier | Delta MAE | Delta Spearman |
|---|---:|---:|---:|---:|
| `A_BR - scrambled` | -0.009986 | -0.003068 | +0.011141 | +0.102341 |
| `A_BR - A_gnomAD` | -0.020111 | -0.003260 | +0.004940 | -0.013919 |

Negative delta is better for NLL, Brier, and MAE. Positive delta is better for Spearman.

## Findings

- `A_BR balanced` outperformed the scrambled-frequency negative control on NLL, Brier, and Spearman.
- The scrambled control had near-zero test Spearman (`0.001261`), consistent with no learned ABRAOM frequency ordering.
- `A_BR balanced` outperformed `A_gnomAD balanced` on test NLL and Brier when evaluated against ABRAOM AF.
- `A_gnomAD balanced` had slightly higher test Spearman than `A_BR balanced`.
- The ABRAOM signal is present, but modest; it is stronger as probability/calibration signal than as rank-order signal.

## Decision

`A_BR` passes the minimum frequency-alignment gate required before ClinVar/fusion work:

- Passed negative control: real ABRAOM target beats scrambled target.
- Passed global-frequency comparator on NLL/Brier: ABRAOM target beats gnomAD-target adapter when evaluated against ABRAOM AF.
- Not fully dominant on rank correlation: gnomAD-target adapter has slightly higher Spearman.

Use `A_BR balanced` as the population adapter candidate for the next stage, but keep `A_gnomAD` and `scrambled` in all downstream ablations.

## Artifacts

- Comparison table: `artifacts/abraom_frequency_adapter/alignment_comparison/abraom_frequency_adapter_comparison.md`
- `A_BR` summary: `artifacts/abraom_frequency_adapter/abraom-balanced-v1-rerun/summary.json`
- `scrambled` summary: `artifacts/abraom_frequency_adapter/scrambled-balanced-v1/summary.json`
- `A_gnomAD` summary: `artifacts/abraom_frequency_adapter/gnomad-balanced-v1/summary.json`
