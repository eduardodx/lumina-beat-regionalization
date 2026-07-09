# ABRAOM Frequency Adapter Run

Generated at UTC: `2026-06-16T20:26:06.301261+00:00`
Best step: `1`

## Final Metrics

| split | rows_scored | model_nll | gnomad_nll | delta_nll | model_brier | gnomad_brier | spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 8 | 0.414056 | 0.201143 | 0.212914 | 0.062828 | 0.000127 | 0.428571 |
| test | 8 | 0.285605 | 0.117554 | 0.168052 | 0.038939 | 0.000023 | 0.707528 |

## Outputs

- `output_dir`: `/opt/ml/model`
- `best_adapter`: `/opt/ml/model/best_adapter.pt`
- `final_adapter`: `/opt/ml/model/final_adapter.pt`
- `summary`: `/opt/ml/model/summary.json`
- `report`: `/opt/ml/model/README.md`
