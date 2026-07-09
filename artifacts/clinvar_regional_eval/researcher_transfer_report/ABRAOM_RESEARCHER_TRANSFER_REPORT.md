---
title: "ABRAOM Regionalization Researcher Transfer Report"
subtitle: "Datasets, artifacts, models, evaluation logic, and handoff instructions"
author: "Lumina ABRAOM/ClinVar regionalization study"
date: "2026-07-06T14:05:27+00:00"
github_branch_url: "https://github.com/eduardodx/lumina-ssm/tree/abraom-regionalization-study"
lang: pt-BR
---

# ABRAOM Regionalization Researcher Transfer Report

**Gerado em UTC:** `2026-07-06T14:05:27+00:00`

**Repositório:** `/home/sagemaker-user/lumina-ssm`

**Branch:** `abraom-regionalization-study`

**GitHub branch URL:** `https://github.com/eduardodx/lumina-ssm/tree/abraom-regionalization-study`

**Commit curto na geração:** `88e3c1c`

**Commit completo na geração:** `88e3c1c008fc53e5da3b30220c2a8beec962fc01`

**Escopo:** este relatório transfere o conhecimento técnico e científico acumulado no estudo de regionalização ABRAOM/ClinVar. Ele foi escrito para um pesquisador que precisa continuar, auditar ou reproduzir o trabalho.

**Uso permitido deste pacote:** validação científica, desenvolvimento de método, análise de erro, preparação de curadoria e desenho de próxima rodada experimental.

**Uso não suportado:** decisão clínica, interpretação clínica final de variante, liberação de modelo para uso diagnóstico, ou alegação definitiva de superioridade regional sem curadoria externa.

## 1. Resumo Executivo

O estudo investigou se informação populacional brasileira do ABRAOM pode melhorar a interpretação de variantes ClinVar em contexto brasileiro sem destruir a sensibilidade para variantes P/LP presentes no ABRAOM.

O resultado mais importante é que **há sinal útil de regionalização**, mas ele ainda não está completamente falsificado contra controles negativos estratificados. O melhor candidato operacional até agora é `M5_v3_safety`: uma calibração de `M5_v2` com separação entre score molecular e score regional, mais uma guarda molecular que impede que frequência ABRAOM apague completamente evidência molecular forte.

Principais números no test set:

| Métrica | M0 baseline | M5_v3_safety | Interpretação |
| --- | --- | --- | --- |
| `br_only` MCC | 0.279 | 0.605 | Ganho brasileiro claro em relação ao baseline geral. |
| `abraom_common_benign` specificity | 0.803 | 0.959 | Redução forte de falso positivo em benignas comuns ABRAOM. |
| `abraom_pathogenic_present` recall | 0.417 | 0.436 | Sensibilidade P/LP recuperada para nível próximo/acima de M0. |
| `global_nonbr_no_abraom` MCC | 0.512 | 0.512 | Não degradação global em MCC no candidato safety. |


Decisão científica atual registrada nos artefatos:

- `M5_v3_safety`: `conditional_candidate: M5_v3 is safe versus M5_v2 but regional specificity is not fully falsified.`
- Próximo passo de validação: `do_not_train_next: prioritize manual critical-error review and external validation; 92 P/LP false benign remain, 60 high-priority.`
- Pacote público ABRAOM: `do_not_train_next_public_evidence_unresolved`

Conclusão prática: **não é hora de treinar um novo modelo como próximo passo imediato**. O gargalo agora é curadoria/evidência pública das variantes críticas.

## 2. Perguntas Científicas que o Estudo Respondeu

1. **ABRAOM melhora algo em relação ao modelo geral?**
   Sim. O ganho aparece principalmente em `br_only` MCC e na especificidade de variantes benignas comuns no ABRAOM.

2. **O ganho vem de abaixar falsos positivos ou falsos negativos?**
   O ganho inicial veio sobretudo de **reduzir falsos positivos** em variantes benignas comuns no ABRAOM. As versões brutas `M5`/`M6` também reduziram falsos positivos, mas criaram problema grave de falsos benignos em P/LP ABRAOM-presentes.

3. **A frequência ABRAOM estava forte demais?**
   Sim. `M5`/`M6` brutos derrubaram demais o recall de P/LP presentes no ABRAOM. Isso motivou `M5_v2_calibrated` e depois `M5_v3_safety`.

4. **O modelo final resolve tudo?**
   Não. `M5_v3_safety` é o melhor candidato científico atual, mas os controles negativos estratificados mostram que a especificidade biológica do sinal ABRAOM ainda não está completamente demonstrada.

5. **Qual é o gargalo agora?**
   Evidência, não arquitetura. Existem 75 variantes críticas em fila pública de alta prioridade; 70 ainda precisam de lookup público, 4 têm conflito de evidência/rótulo, e apenas 1 está pronta como sentinela pública.

## 3. Mapa de Artefatos e Dados

| Item | Caminho | Tamanho local aproximado | Status |
| --- | --- | --- | --- |
| ABRAOM source index | `/home/sagemaker-user/gen-abraom-seqs/data/production_v2` | 41.1 GB | exists |
| ABRAOM frequency adapter dataset | `data/datasets/abraom_frequency_adapter` | 988.8 MB | exists |
| ClinVar x ABRAOM regional dataset | `data/datasets/clinvar/regional_abraom` | 5.8 GB | exists |
| ABRAOM frequency adapter runs | `artifacts/abraom_frequency_adapter` | 48.5 MB | exists |
| ClinVar regional eval/model artifacts | `artifacts/clinvar_regional_eval` | 8.6 GB | exists |
| Adapter fusion blueprint completion | `artifacts/adapter_fusion_blueprint_completion` | 42.2 KB | exists |
| Deep error analysis | `artifacts/adapter_fusion_error_analysis_deep` | 7.1 MB | exists |
| M7 scrambled control analysis | `artifacts/adapter_fusion_m7_control_analysis` | 3.3 MB | exists |
| Final comparison tables | `artifacts/clinvar_regional_comparison` | 27.9 KB | exists |


### O que está versionado no Git

O branch versiona código, configs, documentação, relatórios Markdown/HTML, JSONs, CSVs e TSVs leves de curadoria. O commit atual também inclui tabelas de resultado necessárias para auditoria.

### O que não está versionado no Git

Os seguintes itens são intencionalmente excluídos por `.gitignore` e devem ser transferidos por canal de dados, S3, artefact store ou storage compartilhado:

- `data/datasets/**`: datasets parquet grandes.
- `artifacts/**/*.parquet`: predições, painéis e caches binários.
- `artifacts/**/*.pt`, `*.tar.gz`, `*.safetensors`: checkpoints/modelos.
- `artifacts/**/*.pdf`, `*.png`, `*.svg`: figuras e PDFs derivados.
- caches de janela de sequência e diretórios `_extracted`.

Não reverter isso sem uma decisão explícita de gestão de artefatos.

### Artefatos já apontados no S3 para reuso

Antes de regenerar datasets, treinos ou avaliações, verificar os objetos abaixo. Eles são os prefixes/arquivos S3 já documentados nos relatórios, runbooks ou launchers desta rodada.

| Categoria | Artefato | S3 URI | Como reutilizar | Status | Fonte local |
| --- | --- | --- | --- | --- | --- |
| Dados ABRAOM | ABRAOM v2 processado | `s3://ai4bio-lumina/benchmarks/mosaic/data/processed/gen-abraom-seqs/v2/` | Entrada primária para reconstruir o índice ABRAOM usado no estudo. | documentado | `artifacts/abraom_regional_eval/ABRAOM_FINE_TUNING_COMPARATIVE_REPORT.md` |
| Dados SageMaker | Raiz de dados e saídas do estudo | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/` | Prefixo guarda datasets, treinos, avaliações e artefatos SageMaker do estudo. | documentado | `ABRAOM comparative report e scripts SageMaker` |
| Dados SageMaker | Dataset do adapter de frequência ABRAOM | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/abraom_frequency_adapter/` | Canal `frequency` para treinar/reusar A_BR, A_gnomAD e controle scrambled. | prefixo default do launcher | `scripts/sagemaker_abraom_frequency_adapter.py` |
| Dados SageMaker | Dataset ClinVar x ABRAOM regional | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/` | Raiz esperada para master/slices; usar antes de regenerar parquets grandes. | prefixo default do launcher | `scripts/sagemaker_clinvar_m0.py e scripts/sagemaker_clinvar_fusion.py` |
| Dados SageMaker | Slices regionais ClinVar x ABRAOM | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/slices/` | Canal `dataset` usado por M0, fusion e avaliação regional. | documentado | `artifacts/clinvar_regional_m0/M0_RUN_STATUS.md` |
| Referência | Genoma hg38 | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/hg38/` | Canal `reference` dos jobs SageMaker. | documentado | `artifacts/clinvar_regional_m0/M0_RUN_STATUS.md` |
| Checkpoint base | Lumina BEAT-v10 | `s3://ai4bio-lumina/releases/lumina-beat-v10-20260527182934/` | F0/base checkpoint para M0, adapters e fusion. | documentado | `artifacts/clinvar_regional_m0/M0_RUN_STATUS.md` |
| Modelo ClinVar | M0 baseline non-BR completo | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/clinvar-m0-nonbr-beatv10-v1/sagemaker-artifacts/clinvar-m0-nonbr-beatv10-v1-2e6520-20260621191336/output/model.tar.gz` | Modelo molecular geral usado como baseline e init model da fusion. | documentado | `artifacts/clinvar_regional_m0/M0_RUN_STATUS.md` |
| Adapter de frequência | A_BR ABRAOM balanced | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/abraom-freq-adapter-abraom-balanced-v1-rerun/sagemaker-artifacts/abraom-freq-abraom-balanced-v1-reru-dbb2ad-20260617023522/output/` | Adapter populacional brasileiro usado em M4/M5 fusion. | prefixo default do launcher | `scripts/sagemaker_clinvar_fusion.py` |
| Adapter de frequência | A_gnomAD balanced | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/abraom-freq-adapter-gnomad-balanced-v1/sagemaker-artifacts/abraom-freq-gnomad-balanced-v1-787c1c-20260621124601/output/` | Adapter comparador global usado em M4/M5/M2. | prefixo default do launcher | `scripts/sagemaker_clinvar_fusion.py` |
| Adapter de frequência | A_scrambled balanced | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/abraom-freq-adapter-scrambled-balanced-v1/sagemaker-artifacts/abraom-freq-scrambled-balanced-v1-1b573c-20260616222939/output/` | Controle negativo de adapter embaralhado usado em M7. | prefixo default do launcher | `scripts/sagemaker_clinvar_fusion.py` |
| Treino ClinVar | M6 explicit frequency | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/clinvar-m6-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/` | Controle com features explícitas de frequência, sem adapter populacional. | documentado | `artifacts/clinvar_regional_m6_m4_parallel_status.md` |
| Treino ClinVar | M4 static fusion | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m4-staticfusion-nonbr-beatv10-v1-rerun-g52x/sagemaker-artifacts/` | Fusion estática sem frequência explícita forte. | documentado | `artifacts/clinvar_regional_fusion_status.md` |
| Treino ClinVar | M5 static fusion explicit frequency | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m5-staticfusion-explicitfreq-nonbr-beatv10-v1-g5-4x/sagemaker-artifacts/` | Fusion estática com adapters populacionais e frequência explícita. | documentado | `artifacts/clinvar_regional_fusion_status.md` |
| Treino ClinVar | M4 dynamic gated | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m4-dynamic-gated-nonbr-beatv10-v1/sagemaker-artifacts/` | Fusion dinâmica M4 para completar o blueprint. | documentado | `artifacts/adapter_fusion_blueprint_completion/DYNAMIC_FUSION_JOB_RUNBOOK.md` |
| Treino ClinVar | M5 dynamic gated bounded | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m5-dynamic-gated-bounded-nonbr-beatv10-v1/sagemaker-artifacts/` | Fusion dinâmica M5; forte em especificidade, insegura para P/LP sem calibração. | documentado | `artifacts/adapter_fusion_blueprint_completion/DYNAMIC_FUSION_JOB_RUNBOOK.md` |
| Avaliação regional | M4 regional eval | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m4-staticfusion-nonbr-beatv10-v1/sagemaker-artifacts/` | Saídas SageMaker da avaliação regional M4. | documentado | `artifacts/clinvar_regional_fusion_status.md` |
| Avaliação regional | M5 regional eval | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m5-staticfusion-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/` | Saídas SageMaker da avaliação regional M5. | documentado | `artifacts/clinvar_regional_fusion_status.md` |
| Avaliação regional | M6 regional eval | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m6-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/` | Saídas SageMaker da avaliação regional M6. | documentado | `artifacts/clinvar_regional_fusion_status.md` |
| Fine-tuning ABRAOM inicial | ABRAOM weighted full | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-weighted-full-8gpu-ff-v1/` | Experimento inicial de fine-tuning ABRAOM ponderado. | documentado | `artifacts/abraom_regional_eval/ABRAOM_FINE_TUNING_COMPARATIVE_REPORT.md` |
| Fine-tuning ABRAOM inicial | ABRAOM uniform full | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-uniform-full-8gpu-ff-v1/` | Experimento inicial de fine-tuning ABRAOM uniforme. | documentado | `artifacts/abraom_regional_eval/ABRAOM_FINE_TUNING_COMPARATIVE_REPORT.md` |
| Fine-tuning ABRAOM inicial | ABRAOM wild-only full | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-wild-only-full-8gpu-ff-v1/` | Controle inicial wild-only. | documentado | `artifacts/abraom_regional_eval/ABRAOM_FINE_TUNING_COMPARATIVE_REPORT.md` |


Uso prático:

```bash
aws s3 ls <S3_URI>
aws s3 sync <S3_PREFIX> <local_dir>
aws s3 cp <S3_FILE> <local_file>
```

Notas:

- Para prefixes terminando em `/sagemaker-artifacts/`, procurar subdiretórios de job com `output/model.tar.gz`, `output.tar.gz`, `metrics.json`, `summary.json` ou predições.
- Alguns caminhos são `prefixo default do launcher`: eles vêm do código que lançou/consome os jobs e devem ser validados com `aws s3 ls` antes de assumir disponibilidade.
- Os artefatos leves de decisão continuam no Git; os parquets, checkpoints e outputs SageMaker grandes devem ser recuperados do S3 ou storage compartilhado.
- Se um prefixo S3 existir, **não regenerar** o dataset/job correspondente salvo se a intenção for mudar receita, seed, split ou código.
- Lacuna conhecida: alguns artefatos derivados/calibrados, como M5_v2/M5_v3 e partes das avaliações dinâmicas M2/M7, estão preservados nos artefatos locais/Git, mas o prefixo S3 exato não apareceu nos runbooks locais. Para esses casos, primeiro procurar pelo job no SageMaker/S3; só regenerar se o objeto realmente não existir.

## 4. Origem dos Dados

### 4.1 ABRAOM original

O ABRAOM bruto/indexado usado neste estudo veio de:

`/home/sagemaker-user/gen-abraom-seqs/data/production_v2/`

Arquivos principais:

- `abraom_index.v2.parquet`
- `split_manifest.parquet`
- `release_manifest.json`
- `abraom_index.v2.manifest.json`

O índice ABRAOM v2 contém `17.833.190` variantes de entrada, com colunas como `chrom`, `pos`, `ref`, `alt`, `af_abraom`, `af_gnomad` e `specificity`.

### 4.2 Dataset ABRAOM para adapter de frequência

Script de geração:

`scripts/prepare_abraom_frequency_adapter_dataset.py`

Comando reprodutível padrão:

```bash
uv run python scripts/prepare_abraom_frequency_adapter_dataset.py --overwrite
```

Entradas:

- `/home/sagemaker-user/gen-abraom-seqs/data/production_v2/abraom_index.v2.parquet`
- `/home/sagemaker-user/gen-abraom-seqs/data/production_v2/split_manifest.parquet`

Saídas:

- `data/datasets/abraom_frequency_adapter/abraom_frequency_train.parquet`
- `data/datasets/abraom_frequency_adapter/abraom_frequency_val.parquet`
- `data/datasets/abraom_frequency_adapter/abraom_frequency_test.parquet`
- `data/datasets/abraom_frequency_adapter/summary.json`
- `data/datasets/abraom_frequency_adapter/README.md`

Resumo:

| Campo | Valor |
| --- | --- |
| Variantes ABRAOM de entrada | 17.833.190 |
| Variantes escritas | 17.784.373 |
| Removidas por região não usável | 48.817 |
| Comprimento de contexto recomendado | 4.096 |
| Seed | 42 |
| Tamanho local aproximado | 988.8 MB |


Por split:

| Split | Rows | Mean AF ABRAOM | Mean AF gnomAD | Mean specificity | gnomAD zero |
| --- | --- | --- | --- | --- | --- |
| test | 1.848.888 | 0.1662 | 0.1510 | 0.0307 | 120.208 |
| train | 14.163.875 | 0.1632 | 0.1560 | 0.0235 | 483.452 |
| val | 1.771.610 | 0.1609 | 0.1595 | 0.0177 | 25.569 |


Colunas conceituais importantes:

- `af_abraom`: frequência alélica ABRAOM.
- `af_gnomad`: frequência global comparadora.
- `specificity`: sinal de enriquecimento/diferença regional.
- `logit_af_abraom`, `logit_af_gnomad`, `delta_logit`: alvos transformados para treino/diagnóstico.
- `scrambled_af_abraom`: controle negativo para detectar efeito de arquitetura ou vazamento.
- `af_abraom_bin`, `specificity_bin`: bins usados para balanceamento e controle.
- `block_id`, `split`: herdam a separação espacial do `split_manifest`.

### 4.3 Dataset ClinVar x ABRAOM

Script de geração:

`scripts/prepare_regional_clinvar_dataset.py`

Comando reprodutível padrão:

```bash
uv run python scripts/prepare_regional_clinvar_dataset.py --overwrite
```

Entradas:

- ClinVar base: `/home/sagemaker-user/lumina/data/variants/clinvar/processed`
- ClinVar regional enriquecido: `/home/sagemaker-user/lumina-benchmarks/data/datasets/clinvar/processed/eval_all_enriched/eval_unified.parquet`
- ABRAOM v2: `/home/sagemaker-user/gen-abraom-seqs/data/production_v2/abraom_index.v2.parquet`

Saídas:

- `data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet`
- `data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_train_test.parquet`
- `data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_holdout.parquet`
- `data/datasets/clinvar/regional_abraom/regional_annotation_by_variant.parquet`
- `data/datasets/clinvar/regional_abraom/abraom_matches.parquet`
- `data/datasets/clinvar/regional_abraom/summary.json`

Resumo:

| Campo | Valor |
| --- | --- |
| Linhas totais | 1.089.826 |
| Variant keys únicos | 1.089.826 |
| Labels B/LB | 632.518 |
| Labels P/LP | 457.308 |
| Linhas ClinVar presentes no ABRAOM | 71.591 |
| Matches no benchmark regional | 23.884 |
| Tamanho local aproximado | 5.8 GB |


Por split:

| Split | Rows | Positives | Negatives | SNV | ABRAOM present | Brazilian submitter | Mean AF ABRAOM | Mean specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 190.273 | 53.425 | 136.848 | 166.959 | 14.529 | 785 |  |  |
| test | 190.295 | 49.254 | 141.041 | 168.745 | 15.219 | 564 |  |  |
| train | 709.258 | 354.629 | 354.629 | 580.608 | 41.843 | 3.523 |  |  |


Notas críticas:

- Submitter brasileiro é usado para **estratificação e avaliação**, não como feature de inferência.
- Presença no ABRAOM é evidência populacional, não rótulo clínico.
- Uma variante P/LP presente no ABRAOM pode ser real, founder, recessiva, de penetrância variável, mal classificada ou dependente de contexto de doença. Não use a presença no ABRAOM como regra automática de benignidade.

### 4.4 Slices de avaliação regional

Script:

`scripts/build_regional_clinvar_eval_slices.py`

Comando:

```bash
uv run python scripts/build_regional_clinvar_eval_slices.py --overwrite
```

Parâmetros:

- `common_af_threshold = 0.01`
- `high_specificity_threshold = 0.05`

Slices:

| Slice | Rows | P/LP | B/LB | ABRAOM present | Mean AF ABRAOM | Mean specificity | Uso |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `br_only` | 4.163 | 3.510 | 653 | 381 | 0.2927 | 0.0145 | ClinVar rows with Brazilian submitter evidence and no non-Brazilian submitter evidence in the regional table. |
| `nonbr_only` | 19.012 | 13.291 | 5.721 | 2.804 | 0.2028 | 0.0136 | ClinVar rows with non-Brazilian submitter evidence and no Brazilian submitter evidence in the regional table. |
| `mixed_br_nonbr` | 709 | 632 | 77 | 62 | 0.2634 | 0.0127 | ClinVar rows with both Brazilian and non-Brazilian submitter evidence in the regional table. |
| `br_any` | 4.872 | 4.142 | 730 | 443 | 0.2886 | 0.0142 | ClinVar rows with any Brazilian submitter evidence in the regional table. |
| `regional_benchmark_any` | 23.884 | 17.433 | 6.451 | 3.247 | 0.2145 | 0.0137 | ClinVar rows matched to the enriched regional benchmark table. |
| `abraom_present` | 71.591 | 1.596 | 69.995 | 71.591 | 0.1763 | 0.0123 | ClinVar SNVs present in the filtered ABRAOM v2 index. |
| `abraom_high_specificity` | 1.937 | 38 | 1.899 | 1.937 | 0.3779 | 0.0943 | ClinVar rows present in ABRAOM with specificity >= 0.05. |
| `abraom_common` | 56.944 | 635 | 56.309 | 56.944 | 0.2199 | 0.0147 | ClinVar rows present in ABRAOM with AF_ABRAOM >= 0.01. |
| `abraom_common_benign` | 56.309 | 0 | 56.309 | 56.309 | 0.2213 | 0.0147 | Benign/likely-benign ClinVar rows present in ABRAOM with AF_ABRAOM >= 0.01. |
| `abraom_pathogenic_present` | 1.596 | 1.596 | 0 | 1.596 | 0.0419 | 0.0075 | Pathogenic/likely-pathogenic ClinVar rows present in ABRAOM; this is a do-not-suppress check. |
| `abraom_pathogenic_common` | 635 | 635 | 0 | 635 | 0.0952 | 0.0140 | Pathogenic/likely-pathogenic ClinVar rows present in ABRAOM with AF_ABRAOM >= 0.01; this is a high-risk do-not-suppress check. |
| `global_nonbr_no_abraom` | 16.208 | 13.205 | 3.003 | 0 | NA | NA | Non-Brazilian-only ClinVar rows not present in the filtered ABRAOM v2 index. |


Os slices mais importantes para decisão foram:

- `br_only`: principal métrica brasileira.
- `abraom_common_benign`: mede redução de falso positivo em variantes benignas comuns no ABRAOM.
- `abraom_pathogenic_present`: mede risco de falso benigno em P/LP presentes no ABRAOM.
- `global_nonbr_no_abraom`: mede não degradação global fora da regionalização.

## 5. Arquitetura e Famílias de Modelos

O blueprint original recomendava modularidade:

```text
F0       = modelo base congelado ou parcialmente adaptado
A_BR     = adapter populacional brasileiro treinado com frequência ABRAOM
A_gnomAD = adapter populacional comparador global
A_path   = adapter/head de patogenicidade ClinVar
G        = fusion controller / gate
H_mol    = molecular_effect_score
H_reg    = region_calibrated_pathogenicity_score
```

As versões treinadas/avaliadas neste estudo podem ser lidas assim:

- `M0`: baseline ClinVar non-BR, sem ABRAOM. Serve como modelo molecular geral.
- `M4`: adapter/fusion regional sem frequência explícita forte.
- `M5`: fusion com frequência explícita ABRAOM/gnomAD; bruto.
- `M6`: frequência explícita alternativa; bruto.
- `M5_calibrated`/`M6_calibrated`: versões com desconto de frequência limitado.
- `M7_scrambled`: controle negativo com ABRAOM embaralhado.
- `M2_gnomad_only`: controle populacional usando gnomAD.
- `M5_dynamic_gated`: fusion dinâmica; forte em especificidade, insegura em P/LP.
- `M5_v2_calibrated`: candidato calibrado em holdout.
- `M5_v3_safety`: candidato safety com guarda molecular.

## 6. Resultados Principais

### 6.1 Comparação final M0/M4/M5/M6/M7/M5_v2/M5_v3

| Modelo | Papel | br_only MCC | ABRAOM common benign specificity | ABRAOM P/LP present recall | global nonBR MCC | global nonBR specificity |
| --- | --- | --- | --- | --- | --- | --- |
| `M0` | ClinVar non-BR; baseline molecular sem regionalização | 0.279 | 0.803 | 0.417 | 0.512 | 0.544 |
| `M4` | Static fusion regional sem frequência explícita forte | 0.292 | 0.894 | 0.288 | 0.526 | 0.733 |
| `M5` | Fusion + frequência explícita; bruto | 0.618 | 0.990 | 0.135 | 0.328 | 0.190 |
| `M6` | Frequência explícita alternativa; bruto | 0.624 | 0.998 | 0.018 | 0.435 | 0.354 |
| `M5_calibrated` | M5 com frequência limitada/calibrada v2 inicial | 0.546 | 0.952 | 0.331 | 0.512 | 0.544 |
| `M6_calibrated` | M6 com frequência limitada/calibrada v2 inicial | 0.553 | 0.954 | 0.325 | 0.512 | 0.544 |
| `M7_scrambled` | Controle negativo com ABRAOM embaralhado | 0.417 | 0.903 | 0.252 | 0.512 | 0.544 |
| `M5_v2_calibrated` | Lead calibrado em holdout | 0.605 | 0.959 | 0.436 | 0.500 | 0.844 |
| `M5_v3_safety` | Lead com guarda molecular | 0.605 | 0.959 | 0.436 | 0.512 | 0.788 |


Interpretação:

- `M5` e `M6` brutos mostraram que frequência ABRAOM reduz falsos positivos, mas suprimiram demais P/LP.
- `M5_v2_calibrated` recuperou recall P/LP e manteve especificidade ABRAOM-common >= 0.95.
- `M5_v3_safety` preservou o comportamento de `M5_v2` e restaurou `global_nonbr_no_abraom` MCC para o nível de `M0`.

### 6.2 Fusion dinâmica

| Modelo | br_only MCC | ABRAOM benign specificity | ABRAOM P/LP recall | global nonBR MCC |
| --- | --- | --- | --- | --- |
| `M0` | 0.279 | 0.803 | 0.417 | 0.512 |
| `M2_gnomad_only` | 0.309 | 0.884 | 0.319 | 0.537 |
| `M4_dynamic_gated` | 0.319 | 0.869 | 0.344 | 0.547 |
| `M5_dynamic_gated` | 0.666 | 0.998 | 0.037 | 0.400 |
| `M7_dynamic_scrambled` | 0.301 | 0.889 | 0.313 | 0.539 |
| `M5_v2_calibrated` | 0.605 | 0.959 | 0.436 | 0.500 |


Ponto essencial: `M5_dynamic_gated` foi excelente para reduzir falso positivo em benignas comuns ABRAOM, mas colapsou o recall em P/LP ABRAOM-presentes. Isso mostrou que um gate dinâmico bruto não é suficiente; é necessário limitar o impacto da frequência e proteger evidência molecular forte.

### 6.3 Configuração M5_v3 safety

Configuração selecionada:

```json
{
  "discount_scale": 0.5,
  "max_discount": 0.5,
  "molecular_guard_threshold": 0.65,
  "guarded_max_discount": 0.0,
  "guard_score_floor": 0.35,
  "regional_threshold": 0.35,
  "global_threshold": 0.72
}
```

Interpretação da guarda molecular:

- O modelo mantém dois conceitos separados: `molecular_score` e `regional_score`.
- A frequência regional pode reduzir o score, mas não deve apagar completamente evidência molecular forte.
- A guarda molecular é uma regra de segurança contra falso benigno em variantes P/LP presentes no ABRAOM.

## 7. Controles Negativos e Falsificação

O estudo não deve ser apresentado como "ABRAOM provado biologicamente" sem nuance. Os controles negativos foram desenhados para preservar estruturas como gene, AF bin, cromossomo, tipo de variante e specificity bin enquanto quebram a associação real variante-frequência.

Tabela dos controles estratificados fortes:

| Dataset | Metric | Controle | Real | Média controle | P95 controle | P(controle >= real) | Discount alterado |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `br_only` | `mcc` | `within_gene_af_bin` | 0.605 | 0.603 | 0.605 | 0.6337 | 59.9% |
| `br_only` | `mcc` | `within_af_bin_variant_type` | 0.605 | 0.600 | 0.611 | 0.3960 | 97.0% |
| `br_only` | `mcc` | `within_specificity_bin_variant_type` | 0.605 | 0.599 | 0.611 | 0.3267 | 96.3% |
| `br_only` | `mcc` | `within_chromosome_af_bin` | 0.605 | 0.600 | 0.605 | 0.3663 | 87.5% |
| `br_only` | `mcc` | `within_chromosome_af_bin_variant_type` | 0.605 | 0.601 | 0.605 | 0.4653 | 84.0% |
| `br_only` | `mcc` | `within_gene_af_bin_variant_type` | 0.605 | 0.602 | 0.605 | 0.4554 | 54.1% |
| `abraom_common_benign` | `specificity` | `within_gene_af_bin` | 0.959 | 0.971 | 0.972 | 1.0000 | 92.7% |
| `abraom_common_benign` | `specificity` | `within_af_bin_variant_type` | 0.959 | 0.977 | 0.978 | 1.0000 | 99.5% |
| `abraom_common_benign` | `specificity` | `within_specificity_bin_variant_type` | 0.959 | 0.975 | 0.976 | 1.0000 | 99.4% |
| `abraom_common_benign` | `specificity` | `within_chromosome_af_bin` | 0.959 | 0.976 | 0.977 | 1.0000 | 99.3% |
| `abraom_common_benign` | `specificity` | `within_chromosome_af_bin_variant_type` | 0.959 | 0.976 | 0.978 | 1.0000 | 99.3% |
| `abraom_common_benign` | `specificity` | `within_gene_af_bin_variant_type` | 0.959 | 0.971 | 0.972 | 1.0000 | 92.6% |
| `abraom_pathogenic_present` | `recall` | `within_gene_af_bin` | 0.436 | 0.445 | 0.454 | 1.0000 | 29.4% |
| `abraom_pathogenic_present` | `recall` | `within_af_bin_variant_type` | 0.436 | 0.441 | 0.460 | 0.7525 | 97.6% |
| `abraom_pathogenic_present` | `recall` | `within_specificity_bin_variant_type` | 0.436 | 0.434 | 0.454 | 0.5545 | 95.0% |
| `abraom_pathogenic_present` | `recall` | `within_chromosome_af_bin` | 0.436 | 0.444 | 0.460 | 0.8515 | 73.3% |
| `abraom_pathogenic_present` | `recall` | `within_chromosome_af_bin_variant_type` | 0.436 | 0.444 | 0.454 | 0.9010 | 73.6% |
| `abraom_pathogenic_present` | `recall` | `within_gene_af_bin_variant_type` | 0.436 | 0.445 | 0.454 | 1.0000 | 30.0% |
| `global_nonbr_no_abraom` | `mcc` | `within_gene_af_bin` | 0.512 | 0.512 | 0.512 | 1.0000 | 88.9% |
| `global_nonbr_no_abraom` | `mcc` | `within_af_bin_variant_type` | 0.512 | 0.512 | 0.512 | 1.0000 | 97.9% |
| `global_nonbr_no_abraom` | `mcc` | `within_specificity_bin_variant_type` | 0.512 | 0.512 | 0.512 | 1.0000 | 97.9% |
| `global_nonbr_no_abraom` | `mcc` | `within_chromosome_af_bin` | 0.512 | 0.512 | 0.512 | 1.0000 | 96.8% |
| `global_nonbr_no_abraom` | `mcc` | `within_chromosome_af_bin_variant_type` | 0.512 | 0.512 | 0.512 | 1.0000 | 95.3% |
| `global_nonbr_no_abraom` | `mcc` | `within_gene_af_bin_variant_type` | 0.512 | 0.512 | 0.512 | 1.0000 | 83.9% |


Leitura científica:

- Em `br_only`, o real fica acima de muitos controles, mas alguns controles estratificados chegam muito perto.
- Em `abraom_common_benign`, controles dentro de AF/gene frequentemente igualam ou superam o real, indicando que parte do ganho pode vir de estrutura de frequência/benchmark.
- Em `abraom_pathogenic_present`, o recall real não supera de forma robusta vários controles.
- Portanto, o ganho é útil para calibração regional, mas a especificidade biológica do ABRAOM ainda precisa de curadoria externa e validação independente.

## 8. Análise de Erro e Painel de Curadoria

Resumo registrado:

- P/LP ABRAOM-presentes falso benignos remanescentes: `92`.
- P/LP falso benignos de alta prioridade: `60`.
- ABRAOM-common benignas falso patogênicas remanescentes: `502`.
- Painel derivado review/sentinel: `757` linhas.
- Fila pública de alta prioridade: `75` variantes.

Categorias críticas:

| Audit type | Failure category | Priority tier | n |
| --- | --- | --- | --- |
| `false_benign_plp` | `threshold_near_miss` | `P1_manual_review` | 40 |
| `false_benign_plp` | `weak_molecular_signal` | `P2_model_diagnostic` | 32 |
| `false_benign_plp` | `weak_molecular_signal` | `P1_manual_review` | 15 |
| `false_benign_plp` | `threshold_near_miss` | `P0_manual_review` | 2 |
| `false_benign_plp` | `weak_molecular_signal` | `P0_manual_review` | 2 |
| `false_benign_plp` | `common_plp_recessive_or_founder_review` | `P1_manual_review` | 1 |
| `false_pathogenic_common_benign` | `threshold_near_miss` | `P3_low_priority` | 273 |
| `false_pathogenic_common_benign` | `regional_discount_insufficient_common` | `P3_low_priority` | 151 |
| `false_pathogenic_common_benign` | `regional_discount_insufficient_common` | `P2_model_diagnostic` | 46 |
| `false_pathogenic_common_benign` | `unresolved_false_pathogenic` | `P3_low_priority` | 12 |
| `false_pathogenic_common_benign` | `regional_discount_insufficient_common` | `P1_manual_review` | 9 |
| `false_pathogenic_common_benign` | `molecular_score_overdominant` | `P1_manual_review` | 6 |
| `false_pathogenic_common_benign` | `abraom_specific_common_benign` | `P2_model_diagnostic` | 2 |
| `false_pathogenic_common_benign` | `unresolved_false_pathogenic` | `P2_model_diagnostic` | 2 |
| `false_pathogenic_common_benign` | `abraom_specific_common_benign` | `P3_low_priority` | 1 |


Top genes por prioridade:

| Audit type | Gene | n | Mean priority | Max priority | Median AF ABRAOM | Median molecular probability |
| --- | --- | --- | --- | --- | --- | --- |
| `false_benign_plp` | `FKTN` | 12 | 4.252 | 4.900 | 0.0083 | 0.3350 |
| `false_benign_plp` | `PCSK9` | 8 | 4.692 | 5.430 | 0.0211 | 0.3130 |
| `false_benign_plp` | `F2` | 3 | 5.162 | 5.419 | 0.0171 | 0.3469 |
| `false_benign_plp` | `DYSF` | 3 | 5.043 | 5.228 | 0.0162 | 0.3594 |
| `false_benign_plp` | `PAX6` | 3 | 4.189 | 5.002 | 0.0078 | 0.3016 |
| `false_benign_plp` | `TMC1` | 3 | 4.426 | 4.897 | 0.0154 | 0.3346 |
| `false_benign_plp` | `CFH` | 2 | 5.671 | 6.326 | 0.4626 | 0.3357 |
| `false_benign_plp` | `HIF1A` | 2 | 5.548 | 6.217 | 0.0515 | 0.3733 |
| `false_benign_plp` | `FECH` | 2 | 5.621 | 6.082 | 0.1806 | 0.2894 |
| `false_benign_plp` | `INS` | 2 | 4.423 | 5.044 | 0.3215 | 0.2960 |
| `false_benign_plp` | `FLCN` | 2 | 4.999 | 5.023 | 0.5863 | 0.2496 |
| `false_benign_plp` | `SCN5A` | 2 | 4.928 | 5.012 | 0.0102 | 0.3616 |
| `false_benign_plp` | `JUP` | 2 | 4.945 | 4.993 | 0.0100 | 0.3721 |
| `false_benign_plp` | `UGT1A` | 2 | 4.983 | 4.990 | 0.4505 | 0.2415 |
| `false_benign_plp` | `EYS` | 2 | 4.629 | 4.895 | 0.0124 | 0.3572 |
| `false_benign_plp` | `VWF` | 2 | 4.361 | 4.856 | 0.0079 | 0.3514 |
| `false_benign_plp` | `VLDLR` | 2 | 3.743 | 3.744 | 0.0068 | 0.3034 |
| `false_benign_plp` | `PTF1A` | 1 | 6.106 | 6.106 | 0.5265 | 0.2613 |
| `false_benign_plp` | `PLOD1` | 1 | 5.225 | 5.225 | 0.0081 | 0.3648 |
| `false_benign_plp` | `APOE` | 1 | 5.221 | 5.221 | 0.1298 | 0.3603 |


Status da curadoria pública:

| Status | n |
| --- | --- |
| `needs_public_lookup` | 70 |
| `public_label_conflict` | 4 |
| `local_public_supports_label` | 1 |


Decisões públicas:

| Decision | n |
| --- | --- |
| `no_local_variation_id_or_significance` | 70 |
| `public_benign_conflicts_with_plp_label` | 4 |
| `supports_plp_sentinel` | 1 |


Painel de estresse de alta prioridade:

| Modelo | n | Recall | Specificity | MCC | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| `M0` | 75 | 0.250 | 0.133 | -0.510 | 13 | 45 |
| `M7_dynamic_scrambled` | 75 | 0.117 | 0.133 | -0.678 | 13 | 53 |
| `M5_v2_calibrated` | 75 | 0.000 | 0.000 | -1.000 | 15 | 60 |
| `M5_v3_safety` | 75 | 0.000 | 0.000 | -1.000 | 15 | 60 |


Importante: esse painel é construído a partir de erros de alta prioridade. Ele é um **painel de estresse**, não um benchmark populacional balanceado. Métricas ruins nele identificam falhas; não representam desempenho global.

## 9. Como Reproduzir ou Continuar o Trabalho

### 9.1 Ambiente básico

```bash
cd /home/sagemaker-user/lumina-ssm
git switch abraom-regionalization-study
uv sync
```

Para rodar testes relevantes:

```bash
uv run pytest \
  tests/test_prepare_abraom_frequency_adapter_dataset.py \
  tests/test_prepare_regional_clinvar_dataset.py \
  tests/test_build_regional_clinvar_eval_slices.py \
  tests/test_train_abraom_frequency_adapter.py \
  tests/test_eval_clinvar_fusion_lora.py \
  tests/test_calibrate_m5_v3_safety.py \
  tests/test_validate_regional_signal_next_step.py \
  tests/test_build_public_abraom_validation_package.py
```

### 9.2 Regenerar datasets principais

ABRAOM frequency adapter:

```bash
uv run python scripts/prepare_abraom_frequency_adapter_dataset.py --overwrite
```

ClinVar x ABRAOM:

```bash
uv run python scripts/prepare_regional_clinvar_dataset.py --overwrite
```

Slices regionais:

```bash
uv run python scripts/build_regional_clinvar_eval_slices.py --overwrite
```

### 9.3 Treinar ou avaliar adapter de frequência

Treino local/smoke:

```bash
uv run python scripts/train_abraom_frequency_adapter.py \
  --data-dir data/datasets/abraom_frequency_adapter \
  --output-dir artifacts/abraom_frequency_adapter/smoke-local \
  --max-train-rows 1000 \
  --max-val-rows 500 \
  --max-test-rows 500 \
  --max-steps 10 \
  --overwrite
```

Para produção, usar os scripts SageMaker:

- `scripts/sagemaker_abraom_frequency_adapter.py`
- `scripts/abraom_frequency_adapter_job.py`

### 9.4 Avaliação regional ClinVar

Scripts relevantes:

- `scripts/clinvar_m0_job.py`
- `scripts/clinvar_fusion_job.py`
- `scripts/clinvar_regional_eval_job.py`
- `scripts/sagemaker_clinvar_m0.py`
- `scripts/sagemaker_clinvar_fusion.py`
- `scripts/sagemaker_clinvar_regional_eval.py`

Calibração M5_v2:

```bash
uv run python scripts/calibrate_m5_v2_regional_scores.py
```

Calibração M5_v3 safety:

```bash
uv run python scripts/calibrate_m5_v3_safety.py
```

Validação de sinal e análise de erro:

```bash
uv run python scripts/validate_regional_signal_next_step.py
```

Pacote público de curadoria:

```bash
uv run python scripts/build_public_abraom_validation_package.py
```

### 9.5 Gerar relatórios

Relatório HTML executivo/técnico:

```bash
uv run python scripts/compile_abraom_study_html_report.py
```

Este relatório de transferência:

```bash
uv run python scripts/compile_abraom_researcher_transfer_report.py
```

## 10. Como Usar os Artefatos na Prática

### Para revisar resultados rapidamente

Abrir:

- `artifacts/clinvar_regional_eval/abraom_study_report/abraom_regionalization_study.html`
- `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/M5_V3_SAFETY_REPORT.md`
- `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/REGIONAL_SIGNAL_VALIDATION_NEXT_STEP_REPORT.md`
- `artifacts/clinvar_regional_eval/public_abraom_validation/PUBLIC_ABRAOM_VALIDATION_DECISION_REPORT.md`

### Para usar datasets em análise

Usar:

- `data/datasets/abraom_frequency_adapter/*.parquet` para tarefas de frequência/adapters.
- `data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet` para análise geral ClinVar x ABRAOM.
- `data/datasets/clinvar/regional_abraom/slices/*.parquet` para avaliação regional padronizada.

### Para continuar curadoria

Abrir e preencher:

`artifacts/clinvar_regional_eval/public_abraom_validation/public_evidence_review_queue.tsv`

Campos a preencher:

- `manual_curation_status`
- `manual_public_classification`
- `manual_public_source_url_or_pmid`
- `manual_evidence_note`

Critério de inclusão no painel sentinela:

- Deve haver classificação manual clara.
- Deve haver fonte pública ou PMID.
- Conflitos devem ser marcados como conflito, não forçados para P/LP ou benigno.
- Variantes sem fonte permanecem como `manual_review_incomplete` ou `needs_public_lookup`.

## 11. Recomendações para o Próximo Pesquisador

Prioridade 1: curadoria pública das 75 variantes críticas.

1. Resolver primeiro as 4 variantes `public_label_conflict`.
2. Revisar as 2 P0 `needs_public_lookup`.
3. Revisar as 53 P1 P/LP `needs_public_lookup`.
4. Revisar as 15 benignas comuns P1 `needs_public_lookup`.
5. Rerodar `build_public_abraom_validation_package.py`.

Prioridade 2: transformar a fila em painel sentinela real.

- O painel atual é scaffold/review queue.
- O painel pronto atual tem apenas `1` linha(s), insuficiente para decisão de modelo.
- A próxima decisão de treino deve esperar um painel público/manual suficientemente resolvido.

Prioridade 3: validação externa.

- Ideal: coorte clínica brasileira com variante causal/diagnóstico.
- Alternativa mínima: painel curado de founder/P/LP brasileiros e benignas comuns validadas.
- Sem isso, não usar métricas ABRAOM como prova clínica final.

## 12. Principais Riscos de Interpretação

1. **ABRAOM não é truth set clínico.**
   ABRAOM é frequência populacional, não rótulo de benignidade.

2. **Submitter brasileiro não prova origem populacional brasileira.**
   Submitter country é metadado de submissão; deve ser usado para split/estrato, não como feature clínica.

3. **P/LP presente em ABRAOM não é automaticamente erro.**
   Pode ser founder, recessivo, penetrância reduzida, classificação antiga, ou problema de contexto de doença.

4. **Controle negativo forte importa.**
   Se controle preservando gene/AF/type iguala o real, a alegação ABRAOM-specific fica fraca.

5. **Não otimizar só especificidade.**
   As versões brutas provaram que é fácil reduzir falso positivo destruindo recall P/LP.

## 13. Checklist de Entrega para Outro Pesquisador

Entregar:

- Branch/commit de geração: `abraom-regionalization-study` / `88e3c1c008fc53e5da3b30220c2a8beec962fc01`.
- Este relatório em Markdown/HTML/DOCX.
- Manifesto S3: `artifacts/clinvar_regional_eval/researcher_transfer_report/s3_reuse_manifest.json`.
- `artifacts/clinvar_regional_eval/abraom_study_report/abraom_regionalization_study.html`.
- Diretório `data/datasets/abraom_frequency_adapter/`.
- Diretório `data/datasets/clinvar/regional_abraom/`.
- Diretórios de artefatos leves já versionados.
- Modelos/checkpoints/parquets grandes por storage externo, se forem necessários para reprodução exata.

Não esquecer:

- Confirmar disponibilidade de `/home/sagemaker-user/gen-abraom-seqs/data/production_v2/` ou transferir esse diretório.
- Confirmar disponibilidade dos prefixes listados no manifesto S3 antes de regenerar qualquer dataset ou job SageMaker.
- Confirmar disponibilidade dos dados ClinVar originais de `/home/sagemaker-user/lumina` e `/home/sagemaker-user/lumina-benchmarks` se o pesquisador for regenerar datasets do zero.
- Documentar credenciais/S3/SageMaker separadamente se ele for relançar jobs.

## 14. Apêndice: Arquivos-Chave

Scripts principais:

- `scripts/prepare_abraom_frequency_adapter_dataset.py`
- `scripts/train_abraom_frequency_adapter.py`
- `scripts/prepare_regional_clinvar_dataset.py`
- `scripts/build_regional_clinvar_eval_slices.py`
- `scripts/calibrate_m5_v2_regional_scores.py`
- `scripts/calibrate_m5_v3_safety.py`
- `scripts/validate_regional_signal_next_step.py`
- `scripts/build_public_abraom_validation_package.py`
- `scripts/compile_abraom_study_html_report.py`
- `scripts/compile_abraom_researcher_transfer_report.py`

Documentos e relatórios:

- `artifacts/adapter_fusion_regionalization_blueprint.txt`
- `artifacts/adapter_fusion_blueprint_completion/ABRAOM_ADAPTER_FUSION_BLUEPRINT_COMPLETION_REPORT.md`
- `artifacts/adapter_fusion_error_analysis_deep/DEEP_ERROR_ANALYSIS_REPORT.md`
- `artifacts/adapter_fusion_m7_control_analysis/M7_SCRAMBLED_CONTROL_REPORT.md`
- `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/M5_V2_CALIBRATED_REPORT.md`
- `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/M5_V3_SAFETY_REPORT.md`
- `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/REGIONAL_SIGNAL_VALIDATION_NEXT_STEP_REPORT.md`
- `artifacts/clinvar_regional_eval/public_abraom_validation/PUBLIC_ABRAOM_VALIDATION_DECISION_REPORT.md`

Tabelas de decisão:

- `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_m5v3_summary.csv`
- `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m5_v3_negative_control_comparison.csv`
- `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/strong_negative_control_comparison.csv`
- `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/critical_error_audit/combined_manual_review_queue.csv`
- `artifacts/clinvar_regional_eval/public_abraom_validation/public_evidence_review_queue.tsv`

## 15. Estado Final do Estudo

O blueprint foi explorado de forma ampla: datasets, ABRAOM adapter, gnomAD/scrambled controls, M0 baseline, M4/M5/M6, M7 scrambled, M5_v2 calibrado, M5_v3 safety, controles negativos estratificados e análise profunda de erro.

A estrutura mais promissora é `M5_v3_safety`, porque combina:

- ganho brasileiro (`br_only` MCC),
- redução de falso positivo em benignas comuns ABRAOM,
- proteção parcial/recuperação de P/LP ABRAOM-presentes,
- não degradação global em MCC,
- separação audível entre evidência molecular e regional.

Mas a conclusão final é deliberadamente conservadora: **o próximo avanço depende de curadoria e validação externa, não de mais uma rodada imediata de treino.**
