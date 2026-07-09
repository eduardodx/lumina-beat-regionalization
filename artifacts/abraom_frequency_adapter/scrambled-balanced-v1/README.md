# ABRAOM Frequency Adapter Run

Generated at UTC: `2026-06-16T23:39:08.709745+00:00`
Best step: `500`

## Final Metrics

| split | rows_scored | model_nll | gnomad_nll | delta_nll | model_brier | gnomad_brier | spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 4999 | 0.530897 | 0.380156 | 0.150741 | 0.073878 | 0.004449 | -0.012660 |
| test | 5000 | 0.523908 | 0.661654 | -0.137746 | 0.071624 | 0.017479 | 0.001261 |

## Outputs

- `output_dir`: `/opt/ml/model`
- `best_adapter`: `/opt/ml/model/best_adapter.pt`
- `final_adapter`: `/opt/ml/model/final_adapter.pt`
- `summary`: `/opt/ml/model/summary.json`
- `report`: `/opt/ml/model/README.md`
