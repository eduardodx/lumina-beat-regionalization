# ABRAOM Frequency Adapter Run

Generated at UTC: `2026-06-16T22:18:51.096154+00:00`
Best step: `500`

## Final Metrics

| split | rows_scored | model_nll | gnomad_nll | delta_nll | model_brier | gnomad_brier | spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 2000 | 0.439148 | 0.285287 | 0.153862 | 0.058288 | 0.002108 | 0.125078 |
| test | 1999 | 0.451528 | 0.523602 | -0.072074 | 0.059987 | 0.012650 | 0.068021 |

## Outputs

- `output_dir`: `/opt/ml/model`
- `best_adapter`: `/opt/ml/model/best_adapter.pt`
- `final_adapter`: `/opt/ml/model/final_adapter.pt`
- `summary`: `/opt/ml/model/summary.json`
- `report`: `/opt/ml/model/README.md`
