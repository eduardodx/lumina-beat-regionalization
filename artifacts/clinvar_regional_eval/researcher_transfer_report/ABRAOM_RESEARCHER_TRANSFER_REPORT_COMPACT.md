---
title: "ABRAOM Regionalization Compact Handoff"
subtitle: "Resumo técnico, artefatos S3 e próximos passos"
author: "Lumina ABRAOM/ClinVar regionalization study"
date: "2026-07-06T14:05:27+00:00"
github_branch_url: "https://github.com/eduardodx/lumina-ssm/tree/abraom-regionalization-study"
lang: pt-BR
---

# ABRAOM Regionalization Compact Handoff

**Gerado em UTC:** `2026-07-06T14:05:27+00:00`

**Repositório:** `/home/sagemaker-user/lumina-ssm`

**Branch:** `abraom-regionalization-study`

**GitHub branch URL:** `https://github.com/eduardodx/lumina-ssm/tree/abraom-regionalization-study`

**Commit na geração:** `88e3c1c008fc53e5da3b30220c2a8beec962fc01`

**Versão extensa para auditoria:** `artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md`

## 1. Conclusão Curta

O estudo encontrou **sinal operacional útil nos benchmarks regionais**, principalmente por reduzir falso positivo em variantes benignas comuns no ABRAOM. Isso ainda não prova um sinal biológico específico do ABRAOM. O melhor candidato atual é `M5_v3_safety`, porque combina o sinal regional com uma guarda molecular: a frequência ABRAOM pode reduzir o score de patogenicidade, mas não pode apagar completamente evidência molecular forte.

O resultado ainda **não é prova clínica final**. Os controles negativos estratificados mostram que parte do ganho pode vir de estrutura de frequência, gene ou benchmark. A próxima etapa científica deve ser curadoria pública/externa das variantes críticas, não outro treino imediato.

## 2. Principais Resultados

| Métrica | M0 | M5_v3_safety | Leitura |
| --- | --- | --- | --- |
| `br_only` MCC | 0.279 | 0.605 | ganho brasileiro claro |
| `abraom_common_benign` specificity | 0.803 | 0.959 | menos falso positivo benigno |
| `abraom_pathogenic_present` recall | 0.417 | 0.436 | sensibilidade P/LP recuperada |
| `global_nonbr_no_abraom` MCC | 0.512 | 0.512 | não degradação global em MCC |


Leitura prática:

- `M5` e `M6` brutos reduziram falsos positivos, mas criaram risco forte de falso benigno em P/LP presentes no ABRAOM.
- `M5_v2_calibrated` limitou o impacto da frequência.
- `M5_v3_safety` manteve o ganho regional e adicionou proteção molecular.
- `M7_scrambled` foi usado como controle negativo; ele não deve ser lido como modelo candidato principal.

Alertas no painel crítico de curadoria:

| Panel | Modelo | n | Recall | Specificity | MCC | FP | FN | Leitura |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `high_priority_review_queue` | `M0` | 75 | 0.250 | 0.133 | -0.510 | 13 | 45 | baseline no painel de stress |
| `high_priority_review_queue` | `M5_v3_safety` | 75 | 0.000 | 0.000 | -1.000 | 15 | 60 | alerta: colapso no painel crítico |
| `plp_high_priority` | `M0` | 60 | 0.250 | 0.000 | 0.000 | 0 | 45 | P/LP críticos no baseline |
| `plp_high_priority` | `M5_v3_safety` | 60 | 0.000 | 0.000 | 0.000 | 0 | 60 | alerta: 60 falsos benignos |
| `common_benign_high_priority` | `M0` | 15 | 0.000 | 0.133 | 0.000 | 13 | 0 | benignas comuns críticas no baseline |
| `common_benign_high_priority` | `M5_v3_safety` | 15 | 0.000 | 0.000 | 0.000 | 15 | 0 | alerta: 15 falsos positivos |


Esse painel é um **stress test derivado de erros**, não um benchmark populacional balanceado. Ele mostra onde a hipótese ainda falha e por que a próxima etapa é curadoria, não treino imediato.

## 3. Dados Essenciais

| Item | Valor |
| --- | --- |
| ABRAOM v2 de entrada | 17.833.190 variantes |
| Dataset adapter de frequência | 17.784.373 linhas escritas |
| ClinVar x ABRAOM | 1.089.826 linhas; 1.089.826 variant keys únicos |
| ClinVar presentes no ABRAOM | 71.591 |
| Fila crítica pública | 75 variantes |
| Painel público pronto | 1 variante(s) |


Slices decisivos:

| Slice | Rows | P/LP | B/LB | ABRAOM present | Uso |
| --- | --- | --- | --- | --- | --- |
| `br_only` | 4.163 | 3.510 | 653 | 381 | ganho no subconjunto brasileiro |
| `abraom_common_benign` | 56.309 | 0 | 56.309 | 56.309 | redução de falso positivo |
| `abraom_pathogenic_present` | 1.596 | 1.596 | 0 | 1.596 | proteção contra falso benigno P/LP |
| `global_nonbr_no_abraom` | 16.208 | 13.205 | 3.003 | 0 | não degradação global |


## 4. O que foi Treinado e Comparado

| Modelo | Leitura curta | br_only MCC | Benignas comuns ABRAOM specificity | P/LP ABRAOM recall | Global nonBR MCC |
| --- | --- | --- | --- | --- | --- |
| `M0` | baseline molecular geral, sem ABRAOM | 0.279 | 0.803 | 0.417 | 0.512 |
| `M4` | fusion regional conservadora | 0.292 | 0.894 | 0.288 | 0.526 |
| `M5` | fusion + frequência; reduziu FP, mas suprimiu P/LP | 0.618 | 0.990 | 0.135 | 0.328 |
| `M6` | frequência explícita; forte, mas inseguro para P/LP | 0.624 | 0.998 | 0.018 | 0.435 |
| `M7_scrambled` | controle negativo com ABRAOM embaralhado | 0.417 | 0.903 | 0.252 | 0.512 |
| `M5_v2_calibrated` | M5 com desconto regional limitado | 0.605 | 0.959 | 0.436 | 0.500 |
| `M5_v3_safety` | candidato final com guarda molecular | 0.605 | 0.959 | 0.436 | 0.512 |


Fusion dinâmica:

| Modelo | br_only MCC | ABRAOM benign specificity | ABRAOM P/LP recall | Decisão |
| --- | --- | --- | --- | --- |
| `M5_dynamic_gated` | 0.666 | 0.998 | 0.037 | não selecionado: colapso de P/LP |
| `M7_dynamic_scrambled` | 0.301 | 0.889 | 0.313 | controle negativo; não superou ABRAOM real |


Receita científica:

1. Treinar/usar `M0` como baseline molecular ClinVar non-BR.
2. Treinar adapters de frequência `A_BR`, `A_gnomAD` e `A_scrambled`.
3. Comparar M4/M5/M6 contra M0 nos mesmos slices regionais.
4. Usar M7/scrambled e controles estratificados para falsificação.
5. Calibrar M5 para limitar desconto por frequência.
6. Selecionar `M5_v3_safety` como melhor candidato experimental, com guarda molecular.

## 5. Artefatos S3 para Não Regenerar

Esta seção é crítica. Antes de rodar qualquer script pesado, verificar estes caminhos. Se o objeto existir e a receita não mudou, reutilizar o artefato.

| Categoria | Artefato | S3 URI | Uso | Status |
| --- | --- | --- | --- | --- |
| Dados ABRAOM | ABRAOM v2 processado | `s3://ai4bio-lumina/benchmarks/mosaic/data/processed/gen-abraom-seqs/v2/` | Entrada primária para reconstruir o índice ABRAOM usado no estudo. | documentado |
| Dados SageMaker | Raiz de dados e saídas do estudo | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/` | Prefixo guarda datasets, treinos, avaliações e artefatos SageMaker do estudo. | documentado |
| Dados SageMaker | Dataset do adapter de frequência ABRAOM | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/abraom_frequency_adapter/` | Canal `frequency` para treinar/reusar A_BR, A_gnomAD e controle scrambled. | prefixo default do launcher |
| Dados SageMaker | Dataset ClinVar x ABRAOM regional | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/` | Raiz esperada para master/slices; usar antes de regenerar parquets grandes. | prefixo default do launcher |
| Dados SageMaker | Slices regionais ClinVar x ABRAOM | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/slices/` | Canal `dataset` usado por M0, fusion e avaliação regional. | documentado |
| Referência | Genoma hg38 | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/hg38/` | Canal `reference` dos jobs SageMaker. | documentado |
| Checkpoint base | Lumina BEAT-v10 | `s3://ai4bio-lumina/releases/lumina-beat-v10-20260527182934/` | F0/base checkpoint para M0, adapters e fusion. | documentado |
| Modelo ClinVar | M0 baseline non-BR completo | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/clinvar-m0-nonbr-beatv10-v1/sagemaker-artifacts/clinvar-m0-nonbr-beatv10-v1-2e6520-20260621191336/output/model.tar.gz` | Modelo molecular geral usado como baseline e init model da fusion. | documentado |
| Adapter de frequência | A_BR ABRAOM balanced | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/abraom-freq-adapter-abraom-balanced-v1-rerun/sagemaker-artifacts/abraom-freq-abraom-balanced-v1-reru-dbb2ad-20260617023522/output/` | Adapter populacional brasileiro usado em M4/M5 fusion. | prefixo default do launcher |
| Adapter de frequência | A_gnomAD balanced | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/abraom-freq-adapter-gnomad-balanced-v1/sagemaker-artifacts/abraom-freq-gnomad-balanced-v1-787c1c-20260621124601/output/` | Adapter comparador global usado em M4/M5/M2. | prefixo default do launcher |
| Adapter de frequência | A_scrambled balanced | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/abraom-freq-adapter-scrambled-balanced-v1/sagemaker-artifacts/abraom-freq-scrambled-balanced-v1-1b573c-20260616222939/output/` | Controle negativo de adapter embaralhado usado em M7. | prefixo default do launcher |
| Treino ClinVar | M6 explicit frequency | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/clinvar-m6-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/` | Controle com features explícitas de frequência, sem adapter populacional. | documentado |
| Treino ClinVar | M4 static fusion | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m4-staticfusion-nonbr-beatv10-v1-rerun-g52x/sagemaker-artifacts/` | Fusion estática sem frequência explícita forte. | documentado |
| Treino ClinVar | M5 static fusion explicit frequency | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m5-staticfusion-explicitfreq-nonbr-beatv10-v1-g5-4x/sagemaker-artifacts/` | Fusion estática com adapters populacionais e frequência explícita. | documentado |
| Treino ClinVar | M4 dynamic gated | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m4-dynamic-gated-nonbr-beatv10-v1/sagemaker-artifacts/` | Fusion dinâmica M4 para completar o blueprint. | documentado |
| Treino ClinVar | M5 dynamic gated bounded | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/clinvar-m5-dynamic-gated-bounded-nonbr-beatv10-v1/sagemaker-artifacts/` | Fusion dinâmica M5; forte em especificidade, insegura para P/LP sem calibração. | documentado |
| Avaliação regional | M4 regional eval | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m4-staticfusion-nonbr-beatv10-v1/sagemaker-artifacts/` | Saídas SageMaker da avaliação regional M4. | documentado |
| Avaliação regional | M5 regional eval | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m5-staticfusion-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/` | Saídas SageMaker da avaliação regional M5. | documentado |
| Avaliação regional | M6 regional eval | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/clinvar-regional-eval-m6-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/` | Saídas SageMaker da avaliação regional M6. | documentado |
| Fine-tuning ABRAOM inicial | ABRAOM weighted full | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-weighted-full-8gpu-ff-v1/` | Experimento inicial de fine-tuning ABRAOM ponderado. | documentado |
| Fine-tuning ABRAOM inicial | ABRAOM uniform full | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-uniform-full-8gpu-ff-v1/` | Experimento inicial de fine-tuning ABRAOM uniforme. | documentado |
| Fine-tuning ABRAOM inicial | ABRAOM wild-only full | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-wild-only-full-8gpu-ff-v1/` | Controle inicial wild-only. | documentado |


Comandos mínimos:

```bash
aws s3 ls <S3_URI>
aws s3 sync <S3_PREFIX> <local_dir>
aws s3 cp <S3_FILE> <local_file>
```

Manifesto legível por máquina:

`artifacts/clinvar_regional_eval/researcher_transfer_report/s3_reuse_manifest.json`

Lacuna conhecida: alguns derivados/calibrados, especialmente `M5_v2`, `M5_v3` e partes das avaliações dinâmicas `M2/M7`, estão preservados nos artefatos locais/Git, mas não tinham prefixo S3 exato nos runbooks locais. Para esses casos, procurar primeiro no SageMaker/S3; só regenerar se o objeto realmente não existir.

## 6. Onde Estão os Arquivos Locais Importantes

| Tipo | Caminho |
| --- | --- |
| Relatório compacto | `artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT_COMPACT.md` |
| Relatório extenso | `artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md` |
| Dataset ABRAOM adapter | `data/datasets/abraom_frequency_adapter/` |
| Dataset ClinVar x ABRAOM | `data/datasets/clinvar/regional_abraom/` |
| Tabela final | `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_m5v3_summary.csv` |
| Relatório M5_v3 | `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/M5_V3_SAFETY_REPORT.md` |
| Validação de sinal | `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/REGIONAL_SIGNAL_VALIDATION_NEXT_STEP_REPORT.md` |
| Fila de curadoria | `artifacts/clinvar_regional_eval/public_abraom_validation/public_evidence_review_queue.tsv` |


Cobertura GitHub:

- Estão no GitHub: scripts, relatórios, manifesto S3, CSS/HTML/DOCX/Markdown, CSV/TSV/JSON leves de decisão.
- Não estão no GitHub: `data/datasets/**`, parquets grandes, checkpoints/modelos e artefatos binários grandes; recuperar pelo S3 ou storage compartilhado.
- Para qualquer caminho S3 com status `prefixo default do launcher`, validar existência com `aws s3 ls` antes de assumir disponibilidade.

## 7. Limitações e Próximo Passo

Limitações principais:

- ABRAOM é frequência populacional, não verdade clínica.
- Submitter brasileiro ajuda a estratificar, mas não prova ancestralidade.
- P/LP presente no ABRAOM pode ser founder, recessivo, penetrância variável ou erro de classificação.
- Controles negativos fortes ainda chegam perto do real em alguns cenários.
- O painel público pronto ainda é pequeno demais para decisão clínica.
- No painel de stress high-priority, `M5_v3_safety` ainda falha fortemente; isso deve guiar curadoria antes de qualquer nova alegação de modelo.

Próximo passo imediato:

1. Curar as `75` variantes críticas públicas.
2. Resolver os conflitos e preencher fontes públicas/PMIDs.
3. Rerodar `scripts/build_public_abraom_validation_package.py`.
4. Reavaliar `M5_v3_safety` contra um painel sentinela público maior.
5. Só depois decidir se vale novo treino ou nova arquitetura.

## 8. Comandos de Continuidade

```bash
cd /home/sagemaker-user/lumina-ssm
git switch abraom-regionalization-study
uv sync
uv run python scripts/compile_abraom_researcher_transfer_report.py
```

Decisão atual:

- `M5_v3_safety`: `conditional_candidate: M5_v3 is safe versus M5_v2 but regional specificity is not fully falsified.`
- Validação regional: `do_not_train_next: prioritize manual critical-error review and external validation; 92 P/LP false benign remain, 60 high-priority.`
- Pacote público: `do_not_train_next_public_evidence_unresolved`
