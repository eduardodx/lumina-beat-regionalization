# ABRAOM Regionalization Evaluation

base_checkpoint: `/opt/ml/input/data/base/best_checkpoint.pt`
candidate_checkpoint: `/tmp/lumina_candidate_model/best_checkpoint.pt`
pairs_evaluated: 99740

## Overall

| n | mean_base_score | mean_candidate_score | mean_delta_score | median_delta_score |
|---:|---:|---:|---:|---:|
| 99740 | -0.517400 | -0.547595 | -0.030195 | 0.000061 |

## By Specificity Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.001] | 13670 | 0.000189 | 0.003409 |
| (0.001,0.005] | 16202 | -0.005120 | 0.003166 |
| (0.005,0.01] | 14947 | -0.002769 | 0.003828 |
| (0.01,0.05] | 16525 | -0.000219 | 0.003864 |
| (0.05,0.1] | 12813 | -0.006948 | 0.006079 |
| (0.1,0.5] | 12721 | -0.025180 | 0.008114 |
| (0.5,1] | 12525 | -0.199021 | 0.014064 |
| 0 | 337 | 0.046852 | 0.022012 |

## By ABRAOM AF Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.005] | 1 | -1.459698 | 0.000000 |
| (0.005,0.01] | 17876 | -0.003334 | 0.003605 |
| (0.01,0.05] | 23349 | -0.007067 | 0.003258 |
| (0.05,0.1] | 9483 | -0.005683 | 0.007298 |
| (0.1,0.5] | 27477 | -0.010370 | 0.004088 |
| (0.5,1] | 21554 | -0.113519 | 0.008407 |
