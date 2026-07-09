# Comparativo BEAT-v10 ABRAOM Regional Fine-tuning

Data: 2026-06-14

## Objetivo

Comparar três continuações de treino do mesmo checkpoint base BEAT-v10 para entender se o dataset ABRAOM produz sinal regional mensurável. Todos os braços partiram do mesmo modelo base e usaram objetivo MLM, isto é, prever bases mascaradas em sequências de DNA.

## Braços Comparados

| Braço | Conteúdo do treino | Pergunta respondida |
|---|---|---|
| `abraom_weighted` | Sequências com variantes ABRAOM, amostradas com peso por especificidade ABRAOM vs gnomAD | Variantes mais específicas ajudam a regionalizar o modelo? |
| `abraom_uniform` | Sequências com variantes ABRAOM, sem peso por especificidade | O sinal ABRAOM ajuda mesmo sem weighting? |
| `wild_only` | Janelas genômicas sem substituições ABRAOM | O efeito observado vem só de continuar treinando em DNA hg38-like? |

## Configuração Comum

| Item | Valor |
|---|---|
| Modelo inicial | BEAT-v10 base, checkpoint `lumina-beat-v10-20260527182934/best_checkpoint.pt` |
| Objetivo | MLM-only |
| Comprimento de sequência | 4096 |
| Steps | 5000 |
| Instância | `ml.p5.48xlarge`, 8 GPUs |
| Dataset | ABRAOM v2 em `s3://ai4bio-lumina/benchmarks/mosaic/data/processed/gen-abraom-seqs/v2/` |
| Output bucket | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/` |

## Métricas de Treino

| Braço | Status | Train loss | Train MLM acc | Eval loss | Eval MLM acc | Checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `abraom_weighted` | Completed | 0.8156 | 0.632 | 1.0316 | 0.526 | `step_00005000.pt`, `final_checkpoint.pt` |
| `abraom_uniform` | Completed | 1.0994 | 0.489 | 1.1244 | 0.476 | `step_00005000.pt`, `final_checkpoint.pt` |
| `wild_only` | Completed | 1.1199 | 0.477 | 1.1343 | 0.469 | `step_00005000.pt`, `final_checkpoint.pt` |

Observação objetiva: em MLM, o braço `abraom_weighted` teve menor loss e maior acurácia do que `abraom_uniform` e `wild_only`.

## Avaliação Regional REF/ALT

A avaliação regional foi rodada para os três braços contra o BEAT-v10 base nos splits `val` e `test`.

Métrica usada:

```text
score = logP(ALT) - logP(REF)
delta = score_finetuned - score_base
```

Interpretação:

| Delta | Interpretação |
|---:|---|
| `> 0` | O fine-tuning aumentou a preferência pelo alelo ALT ABRAOM |
| `= 0` | Sem mudança |
| `< 0` | O fine-tuning reduziu a preferência pelo alelo ALT ABRAOM |

## Resultado Regional Final

| Braço | Split | N | Mean delta | Median delta | SEM delta | % delta positivo |
|---|---|---:|---:|---:|---:|---:|
| `abraom_weighted` | `val` | 99,888 | -0.009018 | 0.000309 | 0.001990 | 50.05% |
| `abraom_weighted` | `test` | 99,740 | -0.030195 | 0.000061 | 0.002463 | 50.01% |
| `abraom_uniform` | `val` | 99,888 | 0.000445 | 0.003261 | 0.001720 | 50.65% |
| `abraom_uniform` | `test` | 99,740 | -0.014943 | 0.002948 | 0.002127 | 50.53% |
| `wild_only` | `val` | 99,888 | -0.013451 | 0.000952 | 0.001849 | 50.20% |
| `wild_only` | `test` | 99,740 | -0.040630 | -0.001745 | 0.002250 | 49.68% |

Observação objetiva: `abraom_uniform` teve o melhor resultado regional REF/ALT entre os três braços. `abraom_weighted` ficou melhor que `wild_only`, mas pior que `abraom_uniform` nessa métrica. Todos os braços ficaram próximos de 50% de deltas positivos.

## Scores Médios

| Braço | Split | Mean base score | Mean fine-tuned score |
|---|---|---:|---:|
| `abraom_weighted` | `val` | -0.401937 | -0.410955 |
| `abraom_weighted` | `test` | -0.517400 | -0.547595 |
| `abraom_uniform` | `val` | -0.401937 | -0.401492 |
| `abraom_uniform` | `test` | -0.517400 | -0.532343 |
| `wild_only` | `val` | -0.401937 | -0.415388 |
| `wild_only` | `test` | -0.517400 | -0.558030 |

## Maior Especificidade

Resultado no bin `specificity (0.5,1]`, que concentra variantes com maior diferença de frequência ABRAOM vs gnomAD.

| Braço | Split | N | Mean delta | Median delta |
|---|---|---:|---:|---:|
| `abraom_weighted` | `val` | 5,295 | -0.148763 | -0.043565 |
| `abraom_weighted` | `test` | 12,525 | -0.199021 | -0.065227 |
| `abraom_uniform` | `val` | 5,295 | -0.139255 | -0.036709 |
| `abraom_uniform` | `test` | 12,525 | -0.171630 | -0.045100 |
| `wild_only` | `val` | 5,295 | -0.245693 | -0.056889 |
| `wild_only` | `test` | 12,525 | -0.293458 | -0.079711 |

Observação objetiva: todos os braços tiveram delta negativo no bin de maior especificidade. `abraom_uniform` foi o menos negativo; `wild_only` foi o mais negativo.

## Artefatos

### Treinos

| Braço | S3 |
|---|---|
| `abraom_weighted` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-weighted-full-8gpu-ff-v1/` |
| `abraom_uniform` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-uniform-full-8gpu-ff-v1/` |
| `wild_only` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-wild-only-full-8gpu-ff-v1/` |

### Avaliações Já Concluídas

| Avaliação | S3 | Local |
|---|---|---|
| `weighted` vs base, `val` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/evaluations/abraom-regional-eval-weighted-full-vs-base-val-v1/` | `artifacts/abraom_regional_eval/weighted_full_vs_base_val_full_sagemaker/` |
| `weighted` vs base, `test` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/evaluations/abraom-regional-eval-weighted-full-vs-base-test-v1/` | `artifacts/abraom_regional_eval/weighted_full_vs_base_test_full_sagemaker/` |
| `uniform` vs base, `val` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/evaluations/abraom-regional-eval-uniform-full-vs-base-val-v1/` | `artifacts/abraom_regional_eval/uniform_full_vs_base_val_full_sagemaker/` |
| `uniform` vs base, `test` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/evaluations/abraom-regional-eval-uniform-full-vs-base-test-v1/` | `artifacts/abraom_regional_eval/uniform_full_vs_base_test_full_sagemaker/` |
| `wild_only` vs base, `val` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/evaluations/abraom-regional-eval-wild-only-full-vs-base-val-v1/` | `artifacts/abraom_regional_eval/wild_only_full_vs_base_val_full_sagemaker/` |
| `wild_only` vs base, `test` | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/evaluations/abraom-eval-wild-test-v1/` | `artifacts/abraom_regional_eval/wild_only_full_vs_base_test_full_sagemaker/` |

Arquivos por avaliação:

| Arquivo | Conteúdo |
|---|---|
| `eval_summary.json` | Resumo global e por bins |
| `EVAL_REPORT.md` | Relatório textual da avaliação |
| `eval_by_specificity_bin.parquet` | Métricas agregadas por especificidade |
| `eval_by_variant.parquet` | Score base, score fine-tuned e delta por variante |

## Leitura Atual

1. O fine-tuning executou corretamente nos três braços.
2. O braço `abraom_weighted` teve melhor desempenho MLM que `abraom_uniform` e `wild_only`.
3. Na métrica regional REF/ALT, `abraom_uniform` foi o melhor braço entre os três.
4. `abraom_weighted` não superou `abraom_uniform` na métrica regional, apesar de ter melhor MLM.
5. `wild_only` foi o pior braço em `test` e no bin de maior especificidade.
6. Nenhum braço apresentou deslocamento regional forte: as medianas ficaram próximas de zero e a fração de deltas positivos ficou próxima de 50%.
7. O bin de maior especificidade foi negativo para todos os braços, indicando que a preferência por alelos ALT ABRAOM ainda não foi reforçada nessa faixa.

## Conclusão Comparativa

O treino `abraom_weighted` é o melhor se o critério for desempenho MLM no dataset de fine-tuning. Porém, quando o critério é a avaliação regional REF/ALT, o melhor braço observado é `abraom_uniform`. O controle `wild_only` confirma que continuar treinando sem sinal ABRAOM não melhora a métrica regional e tende a piorar os deltas, especialmente no split `test` e no bin de maior especificidade.

Resultado final para decisão:

| Critério | Melhor braço observado |
|---|---|
| MLM train/eval | `abraom_weighted` |
| REF/ALT regional geral | `abraom_uniform` |
| REF/ALT em maior especificidade | `abraom_uniform` |
| Controle negativo | `wild_only` |

Próximo passo técnico: usar `abraom_uniform` como candidato principal para análise regional, mantendo `abraom_weighted` como evidência de que o weighting melhora MLM mas não necessariamente melhora preferência alélica regional.
