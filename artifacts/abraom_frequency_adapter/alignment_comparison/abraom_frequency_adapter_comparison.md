# ABRAOM Frequency Adapter Comparison

| run | split | target_column | rows_scored | best_step | model_nll | gnomad_nll | delta_nll_model_minus_gnomad | model_brier | gnomad_brier | model_spearman | gnomad_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abraom_balanced | val | af_abraom | 4999 | 500 | 0.518246 | 0.380156 | 0.13809 | 0.0700307 | 0.00444878 | 0.11363 | 0.94093 |
| abraom_balanced | test | af_abraom | 5000 | 500 | 0.513922 | 0.661654 | -0.147732 | 0.0685564 | 0.0174794 | 0.103602 | 0.792585 |
| scrambled_balanced | val | scrambled_af_abraom | 4999 | 500 | 0.530897 | 0.380156 | 0.150741 | 0.0738778 | 0.00444878 | -0.0126601 | 0.94093 |
| scrambled_balanced | test | scrambled_af_abraom | 5000 | 500 | 0.523908 | 0.661654 | -0.137746 | 0.0716244 | 0.0174794 | 0.00126113 | 0.792585 |
| gnomad_balanced | val | af_gnomad | 4999 | 1000 | 0.52305 | 0.380156 | 0.142894 | 0.0709772 | 0.00444878 | 0.106748 | 0.94093 |
| gnomad_balanced | test | af_gnomad | 5000 | 1000 | 0.534033 | 0.661654 | -0.127622 | 0.0718165 | 0.0174794 | 0.117521 | 0.792585 |
| regional_random | val | af_abraom | 2000 | 500 | 0.439148 | 0.285287 | 0.153862 | 0.0582882 | 0.00210824 | 0.125078 | 0.928832 |
| regional_random | test | af_abraom | 1999 | 500 | 0.451528 | 0.523602 | -0.0720736 | 0.0599872 | 0.0126496 | 0.0680207 | 0.77528 |
