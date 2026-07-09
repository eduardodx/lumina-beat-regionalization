# Relatorio tecnico: Lumina, NTv3 e AlphaGenome no espaco NTv3

Data: 2026-06-07  
Track avaliada: `ENCSR814RGG` / `ATAC_seq (ureter)`  
Assay: ATAC-seq  
Metrica: Pearson no protocolo NTv3  
Artefato base: `artifacts/analysis/alphagenome_ntv3_comparison/canonical_results.csv`

## 1. Escopo

Este relatorio descreve, de forma factual, as diferencas entre Lumina, NTv3 e AlphaGenome, as aplicacoes documentadas para NTv3, a relacao tecnica do AlphaGenome com essas aplicacoes e os resultados obtidos na track `ENCSR814RGG`.

A comparacao esta separada por regime experimental:

- **Lumina / NTv3:** avaliacao no benchmark NTv3.
- **AlphaGenome zero-shot:** predicao nativa de ATAC-seq, sem treino no NTv3.
- **AlphaGenome frozen + readout:** AlphaGenome congelado, com readout treinado nos splits NTv3.

## 2. Modelos: entrada, saida e treinamento

| Modelo | Entrada usada pelo modelo | Saida | Treinamento relevante | Regime neste relatorio |
| --- | --- | --- | --- | --- |
| **Lumina** | Apenas sequencia de DNA contendo a mutacao | Predicao/representacao derivada da sequencia | Modelo compacto biologicamente supervisionado; no benchmark, fine-tuning NTv3 | Benchmark NTv3 |
| **NTv3** | Sequencia de DNA; no benchmark, janela de 32 kb | Tracks funcionais/anotacoes em resolucao de base | MLM em larga escala + pos-treinamento supervisionado em tracks/anotacoes multi-especie | Benchmark NTv3 |
| **AlphaGenome** | Sequencia de DNA ate 1 Mb + especie + modalidade/ontology solicitada | Tracks funcionais por modalidade, tecido ou cell type; efeitos de variantes por contraste ref/alt | Treinamento supervisionado em tracks funcionais humanos e murinos | Zero-shot e frozen + readout |

Lumina recebe apenas a sequencia com a mutacao. Ele nao recebe ontology term, tecido, modalidade experimental ou identificador de track.

AlphaGenome foi consultado com informacao de modalidade e tecido:

```text
output_type = ATAC
ontology_term = UBERON:0000056
biosample = ureter
```

Essa diferenca de entrada e uma diferenca metodologica entre os regimes avaliados.

## 3. Aplicacoes documentadas do NTv3

A documentacao publica de NTv3 descreve o modelo como um framework unico para quatro classes de uso:

1. **Predicao de tracks funcionais:** predicao de sinais moleculares em resolucao de base a partir da sequencia.
2. **Anotacao genomica:** predicao de elementos estruturais/anotacionais do genoma.
3. **Analise de variantes:** interpretacao de efeitos de variantes por mudancas em predicoes, atribuicoes, masks, saliency ou atencao.
4. **Geracao/design de sequencias:** adaptacao generativa por masked diffusion para desenhar sequencias com restricoes funcionais.

Casos de aplicacao reportados publicamente incluem:

- benchmark multi-especie com 106 tarefas em sete especies;
- predicao funcional e anotacao em humanos, plantas e animais;
- avaliacao cross-species em especies held-out, incluindo cattle e tomato;
- predicao gene-level de expressao e abundancia proteica a partir de sequencia promoter-proximal;
- interpretacao mecanistica de relacoes enhancer-promoter e efeitos de variantes;
- design de enhancers com niveis de atividade e seletividade por promotor, com validacao experimental por STARR-seq.

No benchmark NTv3, a configuracao padronizada citada pela pagina tecnica e de entrada de `32 kb` e saida em resolucao de base.

## 4. Relacao tecnica do AlphaGenome com essas aplicacoes

AlphaGenome tambem e um modelo sequence-to-function, mas sua formulacao publica e centrada em predicao regulatoria supervisionada. O artigo da Nature descreve entrada de `1 Mb` de DNA e predicao de milhares de tracks funcionais humanos e murinos em modalidades como expressao, splicing, acessibilidade cromatinica, histone marks, transcription factor binding e contatos cromatinicos.

Em relacao as classes de uso documentadas para NTv3:

| Classe de uso | NTv3 | AlphaGenome |
| --- | --- | --- |
| Predicao de tracks funcionais | Sim, no benchmark e no pos-treinamento funcional | Sim, como objetivo central sequence-to-function |
| Anotacao genomica | Sim, como parte do framework NTv3 | Parcialmente relacionada por saidas como genes/splicing, mas nao avaliada aqui como benchmark de anotacao NTv3 |
| Analise de variantes | Sim, por mudancas em predicoes e interpretabilidade | Sim, por contraste entre sequencia mutada e nao mutada |
| Geracao/design de sequencias | Reportada para NTv3-generative com enhancers validados por STARR-seq | Nao foi avaliada neste relatorio |
| Multi-especie amplo | 24 especies no pos-treinamento funcional/anotacional reportado | Humano e camundongo no artigo AlphaGenome |

Assim, AlphaGenome cobre parte importante do espaco biologico de NTv3, principalmente predicao funcional e efeito regulatorio de variantes. A avaliacao feita aqui mede somente a correspondencia de AlphaGenome com uma track funcional NTv3 especifica.


## 5. Predicao de patogenicidade e efeito de variantes

A literatura distingue **classificacao clinica de patogenicidade** de **predicao molecular de efeito de variante**.

Classificacao clinica de patogenicidade segue estruturas como ACMG/AMP, que combinam evidencia populacional, segregacao familiar, estudos funcionais, mecanismo biologico, conhecimento gene-doenca e predicoes computacionais. Nessa estrutura, ferramentas computacionais entram como uma categoria de evidencia, nao como classificacao clinica completa por si so.

Predicao molecular de efeito de variante estima como uma alteracao ref/alt modifica uma propriedade biologica mensuravel. Exemplos incluem mudanca de expressao, acessibilidade cromatinica, splicing, ligacao de fator de transcricao, conservacao, estrutura/função proteica ou deleteriousness agregado.

Modelos e familias tecnicas usadas nessa area incluem:

| Classe | Exemplos | Unidade principal | Saida tipica |
| --- | --- | --- | --- |
| Meta-preditores/anotacao | CADD, REVEL | Variante anotada | Score de deleteriousness ou patogenicidade relativa |
| Modelos evolutivos/proteicos | EVE, AlphaMissense | Variante missense/proteina | Efeito funcional ou probabilidade/score de patogenicidade missense |
| Modelos sequence-to-function | Enformer, Borzoi, AlphaGenome, NTv3 | Sequencia DNA ref/alt | Mudanca em tracks funcionais ou anotacoes |
| Modelos clinicos integrativos | frameworks ACMG/AMP, pipelines ClinGen-like | Variante + contexto clinico/genetico | Categoria clinica: benign, likely benign, VUS, likely pathogenic, pathogenic |

### 5.1 NTv3 nesse contexto

NTv3 e apresentado publicamente como modelo fundacional sequence-function multi-especie, nao como classificador clinico direto de patogenicidade no formato ACMG/AMP. O posicionamento tecnico documentado e de um modelo que prediz tracks funcionais, anotacoes e representacoes, e que tambem pode ser usado para analisar efeitos de variantes por mudancas nas predicoes.

No contexto de variantes, a documentacao de NTv3 descreve:

- predicao funcional em resolucao de base;
- anotacao genomica;
- analise de efeitos de variantes por mudancas de sinal, atribuicoes, masks, saliency ou atencao;
- exemplos de interpretacao de relacoes reguladoras, eQTLs patogenicos versus benignos e variantes de splicing;
- adaptacao generativa para design de sequencias, incluindo enhancers avaliados experimentalmente por STARR-seq.

Assim, no eixo de patogenicidade, NTv3 se posiciona como **modelo de efeito funcional e interpretabilidade mecanistica**. A patogenicidade clinica, em sentido estrito, exigiria integracao com evidencia clinica/genetica e calibracao contra labels clinicos ou funcionais apropriados.

### 5.2 AlphaGenome nesse contexto

AlphaGenome e descrito na Nature como modelo unificado de DNA que recebe ate `1 Mb` de sequencia e prediz milhares de tracks funcionais em resolucao ate base-pair para modalidades como expressao, splicing, acessibilidade cromatinica, histone marks, transcription factor binding e contatos cromatinicos.

No eixo de variantes, AlphaGenome esta mais diretamente posicionado como **regulatory variant effect predictor**. O artigo reporta avaliacao em tarefas de efeito de variantes, incluindo eQTLs, QTLs de acessibilidade, DNase sensitivity, transcription factor binding, polyadenylation QTLs e exemplos mecanisticos de variantes clinicamente relevantes.

AlphaGenome nao retorna, no experimento descrito aqui, uma categoria clinica ACMG/AMP. A saida e molecular: mudancas previstas em tracks funcionais entre sequencia de referencia e sequencia alterada. Esse tipo de saida pode descrever mecanismos reguladores, mas nao equivale automaticamente a classificacao clinica de patogenicidade.

### 5.3 Relacao com Lumina

Lumina, conforme definido neste relatorio, recebe apenas a sequencia de DNA contendo a mutacao. No material local avaliado, Lumina esta sendo comparado no benchmark NTv3 em uma tarefa funcional de ATAC-seq, nao em uma tarefa clinica direta de patogenicidade.

A relacao tecnica entre os tres modelos na area de patogenicidade pode ser resumida assim:

| Modelo | Tipo de evidencia produzida para variantes | Categoria clinica direta? | Observacao neste relatorio |
| --- | --- | --- | --- |
| Lumina | Predicao derivada apenas da sequencia mutada | Nao avaliada aqui | Avaliado em track funcional NTv3 |
| NTv3 | Mudancas em tracks/anotacoes e interpretabilidade de variantes | Nao no benchmark usado aqui | Posicionado como modelo funcional/mecanistico |
| AlphaGenome | Mudancas multimodais em tracks funcionais ref/alt | Nao no experimento aqui | Posicionado como regulatory variant effect predictor |



## 6. Lumina pre, NTv3 pre, AlphaGenome e possivel leakage

A comparacao mais proxima entre modelos antes de adaptacao ao benchmark envolve:

```text
Lumina pre-finetuning NTv3
NTv3 pre
AlphaGenome nativo / zero-shot
```

Esses tres regimes nao sao identicos. Eles diferem no tipo de supervisao vista antes da avaliacao.

| Modelo/regime | Entrada no uso avaliado | Supervisao antes do NTv3 Benchmark | Veu tracks funcionais experimentais antes da avaliacao? | Veu anotacoes antes da avaliacao? | Observacao de comparabilidade |
| --- | --- | --- | --- | --- | --- |
| **Lumina pre-finetuning NTv3** | Apenas sequencia de DNA contendo a mutacao | Supervisao biologica local do projeto, como objetivos densos de conservacao/estrutura de splice, alem de sequencia | Nao ha evidencia neste pacote de treino em tracks funcionais do NTv3, como ATAC/RNA/ChIP | Sim, em sentido biologico local se considerados labels derivados de GENCODE/splice; nao como tracks funcionais NTv3 | Mais proximo de um modelo de sequencia biologicamente supervisionado, antes do benchmark |
| **NTv3 pre** | Sequencia de DNA | Pretraining por masked language modeling em larga escala | Nao no checkpoint `pre`, pela descricao publica; tracks funcionais entram no `pos` | Nao no checkpoint `pre`, pela descricao publica; anotacoes entram no `pos` | Comparador conceitual mais proximo de modelo fundacional antes de pos-treinamento funcional |
| **AlphaGenome nativo / zero-shot** | Sequencia de DNA + especie + modalidade/ontology solicitada | Treinamento supervisionado sequence-to-function | Sim; treinamento em milhares de tracks funcionais humanos/murinos | Parcialmente, via modalidades funcionais e splicing; nao e o mesmo regime de anotacao multi-especie do NTv3 `pos` | Nao e simetrico a Lumina pre ou NTv3 pre, pois ja incorpora supervisao funcional direcionada |

Portanto, **AlphaGenome nao e o unico modelo que pode ver dados supervisionados/anotacionais em geral**. A distincao correta e por regime:

- **NTv3 pre:** antes do pos-treinamento funcional/anotacional descrito publicamente.
- **NTv3 pos:** apos pos-treinamento supervisionado em tracks funcionais e anotacoes multi-especie.
- **Lumina pre:** antes do fine-tuning NTv3; com supervisao biologica local, mas sem evidencia neste pacote de exposicao a tracks funcionais NTv3.
- **AlphaGenome nativo:** treinado diretamente em tracks funcionais humanos/murinos; portanto, ja chega a avaliacao com conhecimento supervisionado de modalidades como ATAC, expressao e splicing.

### 6.1 Onde o possivel leakage e mais relevante

Neste relatorio, **leakage** significa sobreposicao entre informacao vista antes da avaliacao e informacao usada no benchmark. A forma mais relevante para esta discussao nao e apenas ver a sequencia genomica, mas ver o **label funcional**, a **modalidade** ou a **track** avaliada.

| Tipo de sobreposicao | Exemplo | Impacto metodologico |
| --- | --- | --- |
| Sequencia | Ver hg38 ou regioes genomicas durante pretraining | Esperado em modelos genomicos; nao implica conhecimento do alvo funcional |
| Modalidade | Treinar previamente em ATAC-seq | Aproxima o modelo da tarefa avaliada |
| Tecido/cell type | Treinar previamente em ureter ou tecido equivalente | Aproxima o modelo do contexto biologico da track |
| Track/accession | Treinar previamente no mesmo accession ou label funcional `ENCSR814RGG` | Forma mais forte de leakage funcional |
| Test-set feedback | Escolher hiperparametros usando resultado do teste | Leakage de benchmark, independente do modelo |

Na track avaliada, a consulta AlphaGenome foi:

```text
NTv3:        ENCSR814RGG / ATAC_seq (ureter)
AlphaGenome: ATAC / UBERON:0000056 / ureter
```

Essa correspondencia estabelece proximidade de modalidade e tecido. O relatorio atual nao contem auditoria de accession-level overlap entre os dados de treinamento do AlphaGenome e `ENCSR814RGG`. Assim, o risco metodologico principal para AlphaGenome e que sua avaliacao zero-shot pode ocorrer em um espaco biologico muito proximo do seu treinamento supervisionado.

### 6.2 Diferenca entre comparar modelos pre e comparar modelos fine-tuned

Ha duas comparacoes distintas:

| Comparacao | Modelos envolvidos | O que mede |
| --- | --- | --- |
| **Antes do fine-tuning NTv3** | Lumina pre, NTv3 pre, AlphaGenome nativo/zero-shot | Informacao ja presente no modelo antes de adaptar ao benchmark |
| **Depois da adaptacao ao NTv3 Benchmark** | Lumina fine-tuned, NTv3 pre fine-tuned, NTv3 pos fine-tuned, AlphaGenome frozen+readout | Desempenho apos usar supervisao/splits do benchmark |

Os resultados numericos principais deste relatorio pertencem ao segundo grupo, exceto AlphaGenome zero-shot:

| Linha no resultado | Regime |
| --- | --- |
| AlphaGenome zero-shot `0.296621` | Antes de ajuste NTv3, mas condicionado por modalidade/tecido |
| AlphaGenome frozen + readout `0.698436` | Apos readout treinado com labels NTv3 |
| Lumina `0.672015` / `0.686848` | Lumina apos fine-tuning/cabeca NTv3 |
| NTv3 `(pre)` | Checkpoint pre avaliado apos fine-tuning downstream do benchmark |
| NTv3 `(pos)` | Checkpoint pos-treinado funcionalmente, depois avaliado no benchmark |

### 6.3 Posicao relativa dos regimes

A comparacao de **Lumina pre vs NTv3 pre vs AlphaGenome nativo** e a camada conceitualmente mais proxima para avaliar informacao presente antes do benchmark. No entanto, neste pacote:

- nao ha score NTv3 direto para Lumina pre sem fine-tuning;
- NTv3 pre aparece nos resultados publicos do benchmark apos adaptacao downstream;
- AlphaGenome zero-shot tem score direto sem ajuste NTv3, mas com entrada condicionada por modalidade/tecido e treinamento supervisionado funcional previo.

Assim, AlphaGenome ocupa uma posicao diferente: ele nao e apenas um modelo de sequencia antes de fine-tuning; e um modelo supervisionado de predicao funcional. NTv3 pos tambem incorpora supervisao funcional/anotacional antes do downstream. Lumina pre, pelos artefatos locais descritos, nao esta documentado como tendo visto tracks funcionais experimentais do benchmark NTv3 antes da avaliacao.


## 7. Protocolo executado na track `ENCSR814RGG`

Track NTv3:

```text
ENCSR814RGG
ATAC_seq (ureter)
```

Auditoria de correspondencia AlphaGenome:

```text
AlphaGenome ATAC / UBERON:0000056 / ureter
classificacao: exact_assay_tissue_match
```

Configuracao da avaliacao:

- modelo AlphaGenome: `all_folds`
- fonte: Hugging Face
- entrada: janelas NTv3 de `32,768` bp
- crop central avaliado: `12,288` bp por janela
- split: teste NTv3
- janelas avaliadas: `10,531`
- posicoes avaliadas: `129,404,928`

Regimes AlphaGenome:

1. **Zero-shot:** saida ATAC nativa comparada ao alvo NTv3.
2. **Frozen + readout:** AlphaGenome congelado; readout ridge treinado com `2,048` janelas de treino e `512` de validacao; teste completo com `10,531` janelas.

## 8. Resultados quantitativos

| Modelo / regime | Pearson | Runs | Observacao |
| --- | ---: | ---: | --- |
| NTv3 650M (pos) | `0.798486` | 3 | Media publica na track |
| NTv3 650M (pre) | `0.750030` | 3 | Media publica na track |
| NTv3 100M (pre) | `0.714466` | 3 | Media publica na track |
| AlphaGenome frozen + readout | `0.698436` | 1 | Backbone congelado; readout treinado no NTv3 |
| Lumina context-pyramid, melhor score local | `0.686848` | 1 | Fonte: `dataset_scores.csv` local |
| Lumina canonico no pacote comparativo | `0.672015` | 1 | Fonte: tabela honesta vs NTv3 8M |
| NTv3 8M (pre) | `0.666760` | 3 | Media publica na track |
| AlphaGenome zero-shot | `0.296621` | 1 | Sem treino no NTv3 |

Deltas observados:

| Comparacao | Delta Pearson |
| --- | ---: |
| AlphaGenome frozen+readout - AlphaGenome zero-shot | `+0.401814` |
| AlphaGenome frozen+readout - Lumina canonico | `+0.026421` |
| AlphaGenome frozen+readout - Lumina melhor score local | `+0.011588` |
| Lumina canonico - NTv3 8M | `+0.005255` |
| Lumina melhor score local - NTv3 8M | `+0.020087` |
| NTv3 650M pos - AlphaGenome frozen+readout | `+0.100050` |

## 9. Interpretacao dos resultados

O resultado AlphaGenome zero-shot (`0.296621`) mostra baixa correspondencia direta entre a saida ATAC nativa e o alvo NTv3, mesmo com correspondencia de tecido e modalidade.

O resultado AlphaGenome frozen + readout (`0.698436`) mostra que as predicoes AlphaGenome contem sinal funcional aproveitavel para a track, mas a transformacao para o espaco NTv3 depende de calibracao supervisionada.

Lumina apresenta resultado acima do NTv3 8M nesta track nas duas fontes locais disponíveis:

```text
Lumina canonico: 0.672015 vs NTv3 8M: 0.666760
Lumina melhor score local: 0.686848 vs NTv3 8M: 0.666760
```

NTv3 100M e NTv3 650M permanecem acima dos resultados Lumina e AlphaGenome frozen + readout nesta track.

## 10. Limites de inferencia

1. A avaliacao e restrita a uma unica track (`ENCSR814RGG`).
2. AlphaGenome foi treinado em tracks funcionais humanos e murinos; a possivel proximidade entre seus dados de treinamento e a track NTv3 nao foi auditada neste relatorio.
3. AlphaGenome frozen + readout nao e zero-shot, pois usa supervisao NTv3 para treinar o readout.
4. Existem duas fontes locais para o score Lumina nesta track: `0.672015` e `0.686848`.
5. NTv3 possui tres runs publicas na tabela usada; Lumina e AlphaGenome aparecem aqui com uma configuracao cada.
6. Esta avaliacao nao mede tarefas de geracao, anotacao genomica ampla ou interpretabilidade mecanistica.

## 11. Referencias

- AlphaGenome Nature: https://www.nature.com/articles/s41586-025-10014-0
- Google DeepMind AlphaGenome: https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/
- NTv3 research page: https://instadeep.com/research/paper/a-foundational-model-for-joint-sequence-function-multi-species-modeling-at-scale-for-long-range-genomic-prediction/
- NTv3 technical blog: https://instadeep.com/2026/02/modelling-the-genome-with-ntv3/

- ACMG/AMP variant interpretation guidelines: https://pmc.ncbi.nlm.nih.gov/articles/PMC4544753/
- ClinGen calibration of computational predictors for PP3/BP4: https://pmc.ncbi.nlm.nih.gov/articles/PMC9748256/
- CADD: https://www.nature.com/articles/ng.2892
- REVEL: https://pubmed.ncbi.nlm.nih.gov/27666373/
- EVE: https://www.nature.com/articles/s41586-021-04043-8
- AlphaMissense: https://pubmed.ncbi.nlm.nih.gov/37733863/
