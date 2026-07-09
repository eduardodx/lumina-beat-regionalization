# ABRAOM Regionalization Evaluation

base_checkpoint: `/opt/ml/input/data/base/best_checkpoint.pt`
candidate_checkpoint: `/tmp/lumina_candidate_model/best_checkpoint.pt`
pairs_evaluated: 99888

## Overall

| n | mean_base_score | mean_candidate_score | mean_delta_score | median_delta_score |
|---:|---:|---:|---:|---:|
| 99888 | -0.401937 | -0.410955 | -0.009018 | 0.000309 |

## By Specificity Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.001] | 14426 | -0.001438 | 0.003290 |
| (0.001,0.005] | 18653 | 0.001591 | 0.002990 |
| (0.005,0.01] | 16548 | -0.003195 | 0.003415 |
| (0.01,0.05] | 19042 | 0.002134 | 0.003237 |
| (0.05,0.1] | 12876 | -0.014250 | 0.005470 |
| (0.1,0.5] | 12667 | 0.005897 | 0.007397 |
| (0.5,1] | 5295 | -0.148763 | 0.021717 |
| 0 | 381 | -0.002595 | 0.024226 |

## By ABRAOM AF Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.005] | 2 | -0.047361 | 0.371194 |
| (0.005,0.01] | 19278 | -0.007930 | 0.003277 |
| (0.01,0.05] | 26676 | -0.001383 | 0.002998 |
| (0.05,0.1] | 9582 | 0.001034 | 0.006317 |
| (0.1,0.5] | 28504 | 0.000412 | 0.003544 |
| (0.5,1] | 15846 | -0.046231 | 0.007788 |
