# ABRAOM Frequency Adapter Comparison

| run | split | target_column | rows_scored | best_step | model_nll | gnomad_nll | delta_nll_model_minus_gnomad | model_brier | gnomad_brier | model_spearman | gnomad_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regional_v1 | val | af_abraom | 2000 | 500 | 0.439148 | 0.285287 | 0.153862 | 0.0582882 | 0.00210824 | 0.125078 | 0.928832 |
| regional_v1 | test | af_abraom | 1999 | 500 | 0.451528 | 0.523602 | -0.0720736 | 0.0599872 | 0.0126496 | 0.0680207 | 0.77528 |
