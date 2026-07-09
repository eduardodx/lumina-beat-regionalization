# ABRAOM Regionalization Evaluation

base_checkpoint: `/opt/ml/input/data/base/best_checkpoint.pt`
candidate_checkpoint: `/tmp/lumina_candidate_model/best_checkpoint.pt`
pairs_evaluated: 99888

## Overall

| n | mean_base_score | mean_candidate_score | mean_delta_score | median_delta_score |
|---:|---:|---:|---:|---:|
| 99888 | -0.401937 | -0.401492 | 0.000445 | 0.003261 |

## By Specificity Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.001] | 14426 | 0.001947 | 0.002922 |
| (0.001,0.005] | 18653 | 0.005062 | 0.002623 |
| (0.005,0.01] | 16548 | 0.004263 | 0.003024 |
| (0.01,0.05] | 19042 | 0.006951 | 0.002833 |
| (0.05,0.1] | 12876 | 0.008062 | 0.004785 |
| (0.1,0.5] | 12667 | 0.027208 | 0.006414 |
| (0.5,1] | 5295 | -0.139255 | 0.018291 |
| 0 | 381 | 0.020888 | 0.020520 |

## By ABRAOM AF Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.005] | 2 | 0.045865 | 0.845471 |
| (0.005,0.01] | 19278 | 0.001639 | 0.002898 |
| (0.01,0.05] | 26676 | 0.009400 | 0.002646 |
| (0.05,0.1] | 9582 | 0.018757 | 0.005543 |
| (0.1,0.5] | 28504 | 0.007706 | 0.003068 |
| (0.5,1] | 15846 | -0.040224 | 0.006597 |
