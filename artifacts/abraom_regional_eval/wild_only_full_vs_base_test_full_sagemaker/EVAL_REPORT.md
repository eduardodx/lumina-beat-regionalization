# ABRAOM Regionalization Evaluation

base_checkpoint: `/opt/ml/input/data/base/best_checkpoint.pt`
candidate_checkpoint: `/tmp/lumina_candidate_model/best_checkpoint.pt`
pairs_evaluated: 99740

## Overall

| n | mean_base_score | mean_candidate_score | mean_delta_score | median_delta_score |
|---:|---:|---:|---:|---:|
| 99740 | -0.517400 | -0.558030 | -0.040630 | -0.001745 |

## By Specificity Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.001] | 13670 | -0.000394 | 0.002964 |
| (0.001,0.005] | 16202 | -0.002835 | 0.002707 |
| (0.005,0.01] | 14947 | -0.002350 | 0.003364 |
| (0.01,0.05] | 16525 | 0.001640 | 0.003362 |
| (0.05,0.1] | 12813 | 0.000854 | 0.005193 |
| (0.1,0.5] | 12721 | -0.026955 | 0.007128 |
| (0.5,1] | 12525 | -0.293458 | 0.013258 |
| 0 | 337 | 0.042817 | 0.019558 |

## By ABRAOM AF Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.005] | 1 | -0.723878 | 0.000000 |
| (0.005,0.01] | 17876 | -0.000908 | 0.003111 |
| (0.01,0.05] | 23349 | -0.003433 | 0.002835 |
| (0.05,0.1] | 9483 | -0.003171 | 0.006205 |
| (0.1,0.5] | 27477 | -0.009148 | 0.003595 |
| (0.5,1] | 21554 | -0.170450 | 0.007930 |
