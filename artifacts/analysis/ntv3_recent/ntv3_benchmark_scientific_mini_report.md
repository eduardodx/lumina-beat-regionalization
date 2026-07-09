# Mini Report — NTv3 Benchmark Training and Evaluation

## 1. Benchmark Overview

O NTv3 benchmark avalia modelos genômicos em tarefas de sequência-para-função. No recorte usado neste repositório, a tarefa principal é **human functional**: entrada é uma janela de DNA de **32.768 bp**; saída é um tensor denso por posição e por track, em resolução de nucleotídeo, para sinais BigWig funcionais. O notebook oficial de fine-tuning BigWig descreve esse setup com 34 tracks humanas, gradient accumulation, seleção do melhor checkpoint por validation Pearson e recomendação de múltiplas seeds para resultados robustos. Fonte: [notebook oficial NTv3 BigWig](https://huggingface.co/spaces/InstaDeepAI/ntv3/raw/main/notebooks_tutorials/03_fine_tuning_posttrained_model_biwig.ipynb).

O dataset oficial contém FASTA, `splits.bed`, `functional_tracks/*.bigwig`, `genome_annotation/*.bed` e `benchmark_metadata.tsv`, organizado por espécie. Fonte: [NTv3 benchmark dataset](https://huggingface.co/datasets/InstaDeepAI/NTv3_benchmark_dataset). A métrica funcional central é **Pearson correlation coefficient**, reportada por track e agregada como mean Pearson.

Para este repositório, o benchmark é usado como avaliação local controlada de variantes Lumina beat-v5/beat-v6/beat-v7 no slice **human functional**. O NTv3 8M pre aparece como baseline público comparativo, mas a comparação deve ser tratada como parcial se a origem dos números, seeds e protocolo de avaliação não forem idênticos.

## 2. What Is Allowed, Restricted, or Invalid

| Practice | Status | Validity impact | Evidence/source | Notes |
|---|---|---|---|---|
| Usar modelo pré-treinado | Allowed with disclosure | Comparável se checkpoint/dados forem declarados | Notebook usa modelo NTv3 post-trained; paper compara pre/post | Pretraining externo muda interpretação. |
| Treinar nova task head | Allowed with disclosure | Compatível com fine-tuning | Notebook adiciona prediction head BigWig | Head deve ser declarada. |
| Congelar encoder | Allowed with disclosure | Comparação direta fica parcial | Inferência técnica | Não altera splits/labels, mas altera protocolo. |
| Fine-tuning completo | Allowed | Mais próximo do preset local | `eval/ntv3/config.py` | Preset local exige backbone treinável. |
| LoRA/adapters | Allowed with disclosure | Comparável apenas se declarado | Inferência técnica | Não evidenciado nas runs atuais. |
| Usar dados externos | Allowed with disclosure / Invalid if leakage | Pretraining é aceitável; labels de test/val no treino invalidam | Paper compara modelos com diferentes pretraining corpora | Precisa disclosure. |
| Alterar splits | Invalid | Quebra comparação | Dataset oficial fornece `splits.bed` | Risco direto de leakage. |
| Usar test labels no treino | Invalid | Invalida avaliação | Regra metodológica básica; inferência técnica | Val pode selecionar checkpoint. |
| Mudar métrica principal | Not comparable | Quebra benchmark | Notebook usa Pearson | Métricas extras são secundárias. |
| Mudar head/pooling/tokenização | Allowed with disclosure | Comparável se input/output/splits/labels/métrica persistem | Inferência técnica | Arquitetura vira parte do método. |
| Treinar por track/assay separado | Weakly comparable | Não equivale ao modelo multitask agregado | Inferência técnica | Útil como ablação, não claim direto. |

Comparações científicas são enfraquecidas por diferença não declarada de split, budget, seed, arquitetura, dados, checkpoint selection ou protocolo. São inválidas quando há leakage, uso do test set para treinamento/seleção iterativa, alteração oculta de labels/splits ou mudança da métrica principal.

## 3. What This Repository Actually Did

| Item | Evidence in repository | Scientific interpretation | Status |
|---|---|---|---|
| Dataset oficial | `DEFAULT_DATASET_REPO_ID` aponta para `InstaDeepAI/NTv3_benchmark_dataset` em `eval/ntv3/config.py` | Fonte oficial do benchmark | implemented and evidenced |
| Splits oficiais | `splits.bed` carregado em `eval/ntv3/dataset.py` | Evita split customizado | implemented and evidenced |
| Train/val/test separados | Construção explícita em `eval/ntv3/train.py` | Avaliação local rastreável | implemented and evidenced |
| Target transform | Normalização por média + soft clipping em `eval/ntv3/dataset.py` | Reproduz transform usado no pipeline local auditado | implemented and evidenced |
| Loss funcional | Poisson-multinomial em `eval/ntv3/losses.py` | Compatível com profile/scale tracks | implemented and evidenced |
| Gradient accumulation / 19.932 steps | Preset em `eval/ntv3/config.py` | Aproxima protocolo oficial | implemented and evidenced |
| Heads experimentais | Tipos listados em `eval/ntv3/config.py` | Exploração arquitetural local | implemented and evidenced |
| Métricas e artefatos | `metrics_test.json`, `metrics_val.csv`, `dataset_scores.csv` em `artifacts/analysis/ntv3_recent/...` | Resultados auditáveis | implemented and evidenced |
| Melhor resultado local | `beat-v7-20k-context-pyramid-lr125-llrd085-seed0`, mean Pearson `0.531973` | Melhor observação single-seed, não claim robusto | implemented and evidenced |

## 4. Historical Experimental Progression

Critério de inclusão: runs com `run_config.json` e `metrics_test.json`, dataset/split identificável e resultado auditável. Smokes de 2/10 steps foram excluídos como validação operacional. A classificação abaixo é **exploratória**, porque quase todas as runs são seed 0.

| Experiment / run | Hypothesis tested | Configuration | Evidence artifact | Result | Interpretation | Validated premise | Limitation |
|---|---|---|---|---|---|---|---|
| `beat-v5-dec-coupled` | Beat-v5 poderia acoplar ao NTv3 functional | beat-v5, full FT, 19.932 steps | `artifacts/analysis/ntv3_recent/beat-v5-dec-coupled/human/functional/metrics_test.json` | 0.2450 | Baixo desempenho; indicou gargalo forte | Evidência contra essa receita beat-v5 | Legacy; single seed |
| `ntv3-beat-v6-official...` | Beat-v6 melhora sobre v5 | beat-v6, full FT, 19.932 steps | `artifacts/analysis/ntv3_recent/ntv3-beat-v6-official-human-functional-seed0-r2/human/functional/metrics_test.json` | 0.4245 | Efeito grande vs v5 | Forte evidência local de melhora | Single seed |
| `beat-v7-official-hidden-repro` / `r2` | Beat-v7 é melhor backbone local | beat-v7, hidden, full FT, 19.932 steps | `artifacts/analysis/ntv3_recent/ntv3-beat-v7-official-human-functional-seed0-r2/human/functional/metrics_test.json` | 0.4605–0.4615 | Efeito grande vs v6 | Forte evidência local | Ainda single seed |
| `beat-v7-mlp-head-seed0` | MLP melhora head inicial | MLP, sem aux, 19.932 steps | `artifacts/analysis/ntv3_recent/beat-v7-mlp-head-seed0/human/functional/metrics_test.json` | 0.4823 | Melhora observada vs ~0.461 | Tentative | Comparação pode misturar detalhes legacy/head |
| `beat-v7-local-conv-head-k15` | Conv local melhora MLP | local-conv, sem aux | `artifacts/analysis/ntv3_recent/beat-v7-local-conv-head-k15-seed0/human/functional/metrics_test.json` | 0.4723 | Não melhorou nesta seed | No evidence of improvement | Não rejeita hipótese estatisticamente |
| `bio-readout-phylo-structure-mlp` | Aux phylo/structure ajudam | MLP + aux | `artifacts/analysis/ntv3_recent/beat-v7-bio-readout-phylo-structure-mlp-seed0/human/functional/metrics_test.json` | 0.4841 | Ganho pequeno vs MLP | Tentative | Delta pequeno; sem ablação multi-seed |
| `gated-hybrid-phylo-structure` | Gate aprenderia ramo local biologicamente útil | gated-hybrid + aux | `artifacts/analysis/ntv3_recent/beat-v7-gated-hybrid-phylo-structure-seed0/human/functional/gate_analysis/gated_hybrid_gate_report.json` | 0.4853; gate médio 0.022 | Mecanismo de gating não evidenciado | Unsupported mechanism | Uma seed/configuração |
| `recipe-a-bio-readout` | Receita forte melhora treinamento | MLP + aux + EMA/LLRD/bias init, 39.864 steps | `artifacts/analysis/ntv3_recent/beat-v7-ft-recipe-a-bio-readout-seed0/human/functional/metrics_test.json` | 0.5073 | Trajetória combinada melhora métrica | Tentative combined recipe effect | 40k steps; weakly comparable |
| `multiscale-dilated-noresume` | Multi-scale dilated sem gate melhora | multi-scale + aux, 19.932 steps | `artifacts/analysis/ntv3_recent/beat-v7-multiscale-dilated-noresume-seed0/human/functional/metrics_test.json` | 0.4834 | Não melhorou nesta seed | No evidence of improvement | Receita diferente; sem seeds |
| `aggressive-backbone-ema` | Receita agressiva em 20k preserva ganhos | MLP + aux + EMA/LLRD, 19.932 steps | `artifacts/analysis/ntv3_recent/beat-v7-20k-aggressive-backbone-ema-seed0/human/functional/metrics_test.json` | 0.5081 | Ganho observado sem 40k | Tentative | Confunde aux/EMA/LLRD/bias |
| `aggressive-deep-llrd09` | LLRD 0.9 melhora adaptação | MLP + aux + EMA/LLRD 0.9 | `artifacts/analysis/ntv3_recent/beat-v7-20k-aggressive-deep-llrd09-seed0/human/functional/metrics_test.json` | 0.5137 | Pequeno ganho observado | Tentative | Delta pequeno |
| `aggressive-moderate-lr125-llrd085` | LR backbone maior + LLRD 0.85 melhora | MLP + aux + EMA/LLRD | `artifacts/analysis/ntv3_recent/beat-v7-20k-aggressive-moderate-lr125-llrd085-seed0/human/functional/metrics_test.json` | 0.5142 | Melhor MLP recipe observado | Tentative | Possível tuning retrospectivo |
| `global-context-standalone` | Contexto global ajuda além de MLP | global-context + mesma recipe aproximada | `artifacts/analysis/ntv3_recent/beat-v7-20k-global-context-standalone-lr125-llrd085-seed0/human/functional/metrics_test.json` | 0.5227 | Ganho observado vs MLP recipe | Tentative | Delta=0.0084; sem 3 seeds |
| `context-pyramid-lr125-llrd085` | Head piramidal/global aproveita melhor beat-v7 | context-pyramid + aux + recipe | `artifacts/analysis/ntv3_recent/beat-v7-20k-context-pyramid-lr125-llrd085-seed0/human/functional/metrics_test.json` | 0.5320 | Melhor observação local single-seed | Candidate best | Não prova causalidade arquitetural |

A leitura correta é: existe uma trajetória auditável que elevou o mean Pearson local de 0.245 para 0.532. A atribuição causal por componente ainda não está estabelecida. Em especial, `context-pyramid` é o melhor resultado observado, mas não está estatisticamente separado de alternativas próximas sem replicação.

## 5. Validated and Rejected Premises

### Strong local evidence

| Premise | Evidence | Status | Consequence for the research direction |
|---|---|---|---|
| O pipeline local usa dataset/splits oficiais | Código carrega dataset oficial e `splits.bed` | Strong implementation evidence | Avaliação local é auditável. |
| Beat-v7 é melhor que beat-v5/v6 nesta configuração | v5 0.2450, v6 0.4245, v7 ~0.4615 | Strong local effect, still single-seed | Beat-v7 é backbone preferido. |
| A trajetória experimental melhorou o score observado | 0.2450 → 0.5320 | Strong descriptive evidence | Há progresso local real, mas não causalmente decomposto. |

### Partially validated / tentative premises

| Premise | Evidence | Status | Consequence for the research direction |
|---|---|---|---|
| MLP melhora sobre head inicial | 0.4823 vs ~0.461 | Tentative | Manter MLP como controle mínimo. |
| Aux phylo/structure ajudam | 0.4841 vs 0.4823 | Weak/tentative | Precisa ablação com e sem aux. |
| LLRD/EMA/bias init melhoram treinamento | recipes 0.508–0.514 vs MLP 0.482 | Tentative combined effect | Testar receita fixa em 3 seeds. |
| Global/context-pyramid melhora MLP recipe | 0.5227/0.5320 vs 0.5142 | Tentative | Candidato a experimento confirmatório. |
| Comparação com NTv3 8M é defensável | Artifact local reporta 0.5320 vs 0.4982 | Conditional | Depende da origem e protocolo do baseline. |

### Rejected or unsupported premises

| Premise | Evidence | Status | Consequence for the research direction |
|---|---|---|---|
| Gating biológico foi efetivo | Gate médio 0.022 em uma run | Unsupported mechanism | Não narrar gate como mecanismo validado. |
| Local conv simples foi refutado | 0.4723 vs MLP 0.4823 | Not established | Apenas “sem evidência de melhora single-seed”. |
| Multi-scale dilated foi refutado | 0.4834 vs MLP 0.4823 | Not established | Não priorizar, mas não declarar refutação definitiva. |
| Single-seed basta para claim forte | Todas top runs são seed 0 | Unsupported | Precisa replicação. |
| Context-pyramid é causalmente superior | Melhor single-seed, sem ablação cruzada | Unsupported as causal claim | Deve virar hipótese confirmatória. |

## 6. Statistical and Protocol Caveats

A principal limitação é estrutural: quase toda a progressão é **single-seed**. Para deltas pequenos, por exemplo 0.5142 → 0.5227 ou 0.5227 → 0.5320, não há como distinguir ganho real de ruído de seed, ordem de batches, checkpoint selection ou variância por track.

Também existe um **garden of forking paths**: várias configurações foram tentadas e a narrativa tende a destacar a melhor. Isso não invalida a exploração, mas impede tratar o melhor run como confirmação independente.

A Recipe A com 39.864 steps é **weakly comparable**, pois excede o orçamento nominal de 19.932 steps. As runs de 19.932 steps preservam o budget nominal, mas mudanças de EMA, LLRD, bias init e warmup alteram a dinâmica efetiva de treinamento; portanto exigem disclosure.

Por fim, a comparação com NTv3 8M/100M só é forte se os números vierem do mesmo dataset, split, transform, métrica, número de seeds e política de checkpoint. Caso venham de leaderboard/artifact externo, devem ser apresentados como referência pública, não pareamento estatístico direto.

## 7. Supported Claims

### Strongly supported

- O repositório implementa avaliação local NTv3 human functional com artefatos auditáveis.
- O pipeline usa dataset oficial, `splits.bed`, Poisson-multinomial loss e Pearson.
- Beat-v7 apresentou efeito grande sobre beat-v5/v6 nas runs locais.
- A melhor observação local single-seed é `0.531973` com `beat-v7-20k-context-pyramid-lr125-llrd085-seed0`.

### Weakly supported

- Context-pyramid pode ser melhor que MLP/global-context.
- Aux features podem ajudar.
- LLRD/EMA/bias init podem melhorar estabilidade.
- O resultado atual pode ser competitivo com NTv3 8M/100M pre em human functional.

### Not supported

- Claim oficial de liderança no NTv3 benchmark.
- Superioridade robusta sem 3 seeds.
- Causalidade arquitetural do context-pyramid.
- Efetividade biológica do gated-hybrid.
- Generalização multi-espécie.

## 8. Next Steps for Scientific Rigor

| Step | Purpose | Minimum implementation | Outcome |
|---|---|---|---|
| 3 seeds da melhor run | Medir variância | `context-pyramid + same recipe`, 19.932 steps | Robustez do melhor resultado |
| Controle MLP com mesma receita | Isolar efeito da head | `MLP + same recipe`, 3 seeds | Testar se context-pyramid agrega |
| Ablation aux features | Isolar phylo/structure | context-pyramid com e sem aux | Medir contribuição real das aux features |
| Estatística por assay/track | Evitar conclusão só por aggregate | mean/std/CI por overall, assay e track | Identificar ganhos/regressões reais |
| Baseline NTv3 8M comparável | Fortalecer comparação externa | Reavaliar no mesmo pipeline ou declarar fonte | Comparação menos frágil |
| Test final congelado | Reduzir seleção por test | Escolher por val; avaliar test uma vez | Claim mais limpo |

O mínimo defensável é: **3 seeds para MLP+recipe vs context-pyramid+recipe, mesmo budget de 19.932 steps, mesma seleção por validation Pearson, e estatística por track/assay**. Isso transforma a hipótese arquitetural em teste comparativo real.

## 9. Final Scientific Assessment

O uso atual do NTv3 neste repositório é cientificamente válido como **avaliação local controlada e exploratória**. A infraestrutura é auditável, os artefatos existem e a separação entre benchmark local e claim oficial está correta.

A conclusão cientificamente segura é: o repositório construiu uma avaliação local do NTv3 human functional e identificou uma trajetória de configurações que elevou o mean Pearson observado de `0.245` para `0.532` em seed único. A conclusão que ainda **não** é segura é que `context-pyramid`, aux features ou LLRD/EMA sejam causalmente responsáveis por esse ganho.

Antes de qualquer narrativa forte de arquitetura, o próximo passo deve ser replicação estatística e ablação controlada.

## 10. Current Lumina Performance Status

O gráfico abaixo resume o status atual de desempenho de todas as runs Lumina com métrica auditável encontrada nos artefatos locais. Runs completas ou exploratórias longas aparecem coloridas por geração do Lumina; smokes/probes operacionais de poucos steps aparecem em cinza para evitar que sejam interpretados como experimentos científicos completos.

![Lumina NTv3 human functional current run status](lumina_all_runs_status/lumina_ntv3_all_runs_status.png)

Fonte auditável: `lumina_all_runs_status/lumina_ntv3_all_runs_status.csv`. A versão vetorial está em `lumina_all_runs_status/lumina_ntv3_all_runs_status.svg`.
