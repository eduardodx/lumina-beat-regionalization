# ABRAOM Frequency Adapter Run

Generated at UTC: `2026-06-21T13:55:38.383786+00:00`
Best step: `1000`

## Final Metrics

| split | rows_scored | model_nll | gnomad_nll | delta_nll | model_brier | gnomad_brier | spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 4999 | 0.523050 | 0.380156 | 0.142894 | 0.070977 | 0.004449 | 0.106748 |
| test | 5000 | 0.534033 | 0.661654 | -0.127622 | 0.071816 | 0.017479 | 0.117521 |

## Outputs

- `output_dir`: `/opt/ml/model`
- `best_adapter`: `/opt/ml/model/best_adapter.pt`
- `final_adapter`: `/opt/ml/model/final_adapter.pt`
- `summary`: `/opt/ml/model/summary.json`
- `report`: `/opt/ml/model/README.md`
