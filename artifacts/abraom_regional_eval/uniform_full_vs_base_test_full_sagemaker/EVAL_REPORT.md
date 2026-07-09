# ABRAOM Regionalization Evaluation

base_checkpoint: `/opt/ml/input/data/base/best_checkpoint.pt`
candidate_checkpoint: `/tmp/lumina_candidate_model/best_checkpoint.pt`
pairs_evaluated: 99740

## Overall

| n | mean_base_score | mean_candidate_score | mean_delta_score | median_delta_score |
|---:|---:|---:|---:|---:|
| 99740 | -0.517400 | -0.532343 | -0.014943 | 0.002948 |

## By Specificity Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.001] | 13670 | 0.002695 | 0.002952 |
| (0.001,0.005] | 16202 | -0.000137 | 0.002713 |
| (0.005,0.01] | 14947 | 0.006407 | 0.003341 |
| (0.01,0.05] | 16525 | 0.007090 | 0.003320 |
| (0.05,0.1] | 12813 | 0.018362 | 0.005128 |
| (0.1,0.5] | 12721 | 0.012782 | 0.006871 |
| (0.5,1] | 12525 | -0.171630 | 0.012267 |
| 0 | 337 | 0.040988 | 0.019060 |

## By ABRAOM AF Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.005] | 1 | -0.713939 | 0.000000 |
| (0.005,0.01] | 17876 | 0.009334 | 0.003089 |
| (0.01,0.05] | 23349 | 0.004605 | 0.002822 |
| (0.05,0.1] | 9483 | 0.018334 | 0.006085 |
| (0.1,0.5] | 27477 | 0.007771 | 0.003481 |
| (0.5,1] | 21554 | -0.099818 | 0.007329 |
