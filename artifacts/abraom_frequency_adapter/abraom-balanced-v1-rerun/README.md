# ABRAOM Frequency Adapter Run

Generated at UTC: `2026-06-17T03:44:10.443578+00:00`
Best step: `500`

## Final Metrics

| split | rows_scored | model_nll | gnomad_nll | delta_nll | model_brier | gnomad_brier | spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 4999 | 0.518246 | 0.380156 | 0.138090 | 0.070031 | 0.004449 | 0.113630 |
| test | 5000 | 0.513922 | 0.661654 | -0.147732 | 0.068556 | 0.017479 | 0.103602 |

## Outputs

- `output_dir`: `/opt/ml/model`
- `best_adapter`: `/opt/ml/model/best_adapter.pt`
- `final_adapter`: `/opt/ml/model/final_adapter.pt`
- `summary`: `/opt/ml/model/summary.json`
- `report`: `/opt/ml/model/README.md`
