# ABRAOM Regionalization Evaluation

base_checkpoint: `/opt/ml/input/data/base/best_checkpoint.pt`
candidate_checkpoint: `/tmp/lumina_candidate_model/best_checkpoint.pt`
pairs_evaluated: 99888

## Overall

| n | mean_base_score | mean_candidate_score | mean_delta_score | median_delta_score |
|---:|---:|---:|---:|---:|
| 99888 | -0.401937 | -0.415388 | -0.013451 | 0.000952 |

## By Specificity Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.001] | 14426 | -0.001027 | 0.002915 |
| (0.001,0.005] | 18653 | 0.003036 | 0.002626 |
| (0.005,0.01] | 16548 | -0.001777 | 0.003062 |
| (0.01,0.05] | 19042 | 0.002470 | 0.002852 |
| (0.05,0.1] | 12876 | -0.006535 | 0.004970 |
| (0.1,0.5] | 12667 | -0.002017 | 0.006744 |
| (0.5,1] | 5295 | -0.245693 | 0.021256 |
| 0 | 381 | 0.019865 | 0.019670 |

## By ABRAOM AF Bin

| group | n | mean_delta_score | sem_delta_score |
|---|---:|---:|---:|
| (0,0.005] | 2 | 0.072554 | 0.706417 |
| (0.005,0.01] | 19278 | -0.005739 | 0.002930 |
| (0.01,0.05] | 26676 | 0.002465 | 0.002654 |
| (0.05,0.1] | 9582 | 0.000062 | 0.005811 |
| (0.1,0.5] | 28504 | -0.003358 | 0.003211 |
| (0.5,1] | 15846 | -0.075967 | 0.007558 |
