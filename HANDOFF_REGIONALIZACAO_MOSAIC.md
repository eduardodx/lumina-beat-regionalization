# HANDOFF — Regionalização do R03 com o estudo brasileiro do Mosaic

> **Para quem pega num chat novo: este doc é auto-contido.** Leia inteiro antes de tocar em código.
> Datado **2026-09-16**, atualizado em **2026-09-22** (§14). Autor: Gabriel (dev, TCC). Gestor: Eduardo (mantém o Mosaic).
> Branch: **`new_regionalization`**.
> Decisões A–D fechadas pelo Eduardo (§4). **G1 e G2 concluídos; o adapter do G4 treina no R03 real e foi
> medido; o G3 está parcialmente preparado.** O estado atual está na **§14**, que é por onde começar — a §13
> vira histórico de 20/09.

---

## 1. Missão

Refazer a regionalização do Lumina **R03** com duas mudanças pedidas pelo Eduardo: **sem adapter ClinVar** e **sem
fusion**, mantendo os adapters populacionais. A avaliação passa a ser o estudo brasileiro publicado no Mosaic.

Pergunta da campanha (proposta): a adaptação populacional com ABraOM produz ganho diferencial em variantes com
**participação** de instituições brasileiras, comparada com uma adaptação a variação humana global?

---

## 2. Onde ler o quê

| Documento | Papel |
|---|---|
| `docs/proposta_mosaic_regionalizacao_desenvolvimento.md` | **Plano formal atual** (rota, identidades, dados, treino, avaliação, gates, pauta do Eduardo). Commit `848d2dc` |
| `docs/contrato_v2_regionalizacao_r03.md` | Contrato v2 (rascunho). Tem o aviso de que o plano acima substitui §4, §5, §7, §10 e a pendência 9 se o Eduardo aprovar |
| PDF de regionalização do Eduardo (28 páginas, fora do repo) | Protocolo de referência: M0–M4, T_BR/T_nonBR, DiD, chr8, BRCA/TP53, critérios A–I |
| `docs/decisoes_eduardo_fase0.md`, `docs/justificativa_endpoint_auroc.md` | Endpoint AUROC (decisão de 29/07) e a proposta da margem 0,02 (C1), ainda não confirmada |
| `docs/diagnostico_do_adapter_regional.md` | **O que os pilotos do adapter mostram e o que ainda não mostram**: as três perguntas (MLM, M0 × MR, MG × MR), as correções da revisão de 22/09 e o bootstrap. Resumido na §14 |
| `RESULTADOS_REGIONALIZACAO_V11.md` (processo anterior, **referência**) | O que a v11 mediu: sinal sequência→AF real, **sem especificidade regional** (IC cruzando zero). O resíduo que ele propõe foi rodado depois, com AF observada (sinal fraco): `TCC_REGIONALIZACAO_V11.md` e §7.3 do `HANDOFF_CONTINUACAO_V11_POS_FASE4.md` |
| `HANDOFF_EXTRACAO_EMBEDDINGS_R03.md` (branch `embedding-probe-mosaic`) | Pesquisa de extração de embeddings, fechada |
| `HANDOFF_R03_CONTINUACAO.md` | Campanha M0–M4 antiga (com adapter ClinVar e fusion), **superada** |
| `lumina-mosaic/PROTOCOLO.md` + `docs/GUIA_OPERACIONAL_DE_SCORING_DOS_ESPECIALISTAS.md` | Regras do benchmark; a seção "Estudo brasileiro" é a que vale aqui |

---

## 3. Decisões já fixadas (não reabrir)

- **Backbone:** R03 publicado, `best_checkpoint.pt`, **passo 71.000**, pesos sem EMA, em
  `s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt`.
  Não trocar pelo `final_checkpoint.pt` (passo 75.000). sha256 do arquivo já registrado no contrato v1 (`f2983560…`);
  reconferir o arquivo carregado em cada run.
- **Benchmark:** `https://github.com/croma-bioai/lumina-mosaic`. Código no commit `814e7f0`; dados no release
  `clinvar-pb-capability-suite/v1` (`~/mosaic-v1/` no notebook). Código e dados têm identidades separadas.
- **Sistemas (revisto em 15/09 pelo Eduardo):** **M0** = R03 puro (`base`) e **MR** = R03 + **um único adapter
  populacional misto ≈60% global / 40% ABraOM** (`regionalized`). O braço global puro (MG) é ablação adiada, obtida
  com a mesma receita e mistura 100/0. As cabeças H0 e HR são treinadas **separadamente**, com os mesmos dados e o
  mesmo procedimento, depois de congelar cada representação.
- **Treino da cabeça:** folds de treino do `core_locus` do release (`run_id=0`), menos os membros dos dois estudos
  brasileiros e seus `overlap_cluster_id`, menos a regra ampla brasileira, com `sequence_eligible`. Autorizado pelo
  mantenedor do Mosaic em 15/09/2026 — registrar isso no manifesto.
- **Adapter:** MLM com máscaras em span sobre as mutações, mutações em posições aleatórias da janela.
- **Fora:** adapter ClinVar, fusion, AF observada nas entradas da cabeça (AF e presença viram baselines diagnósticas),
  M3/M4 nesta campanha.
- **Cabeças nativas do R03 congeladas** no treino do adapter (usar `freeze_native_feature_heads` de
  `eval/clinvar/lora.py` e conferir os parâmetros que recebem gradiente).

---

## 4. Decisões do Eduardo (15/09) e o que ainda falta

| # | Decisão | Resposta |
|---|---|---|
| A | Estudos do Mosaic (participação) em vez de só-BR | **fechada:** "pode seguir com esses dois splits mesmo" |
| B | Fonte de treino da cabeça clínica | **fechada:** "treina core_locus e avalia neles" → snapshot derivado dos folds de treino do `core_locus` do release, autorizado pelo mantenedor |
| C | Escada M0 → global → ABraOM | **mudou:** "vamos tentar ir direto pra global + abraom direto… algo 60% global 40% Brasil… vamos tentar pular etapas" → **um único adapter misto**; M0 = `base`, MR = `regionalized`; o braço global puro (MG) vira ablação adiada |
| D | Objetivo do adapter populacional | **fechada:** MLM com "máscaras (span) em cima das mutações" e "mutações em posições aleatórias da janela" |
| E | chr8 representacional e BRCA1/BRCA2/TP53 | **em aberto** |

**Ainda falta dele:** o arquivo do ABraOM (`SABE1171.Abraom.clean.tsv`, sha256 `3cd33784…`, `academic_request`, sem
URL — bloqueia G0 e G4), a decisão E, e a confirmação dos parâmetros da §5.1 do plano (janela, variantes por janela,
fração de spans em posições de referência, amostragem por AF, registro da mistura).

**Custo declarado de C:** com um adapter misto, um ganho positivo não é atribuível ao componente brasileiro. A
ablação MG é a mesma receita com mistura 100/0 — uma execução, sem código novo.

**Pontos do Mosaic para ele registrar (não bloqueiam):** a URL do `submission_summary_2026-06` em `config/sources.yaml`
aponta para `archive/2026/`, que não existe (os arquivos estão em `archive/`); `origin_has_germline` só aceita a origem
literal `germline` (`de novo`, `maternal`, `inherited`, `biparental` e `unknown` ficam fora); nenhuma variante gold
recebe marcação brasileira, provavelmente porque, havendo painel, as demais SCVs não contribuem para o agregado.

O texto pronto para encaminhar está na §9 do plano.

---

## 5. Fatos medidos — **não re-derivar**

### 5.1 Auditoria dos dados v1 (`scripts/audit_regionalization_data.py`, commit `92304ab`; JSON em `~/artifacts/redesenho/audit_dados.json`)

- `af_gnomad` do pipeline v1 só existe onde a variante está no índice ABraOM (0 exceções), e o join com o ABraOM só é
  tentado para SNV. Os "91% do T_BR sem gnomAD" eram 3.696 de 4.066 fora do índice ABraOM.
- Nos 710 pares de estratos mistos, concordância de `abraom_present` 0,842 contra 0,716 esperada por sorteio
  (descritivo, não prova de viés).
- Campanha só-SNV custa pouco: dos pares v1, SNV fica com 2.774 P / 529 B (perde 18 benignas).

### 5.2 Diagnóstico de submissor brasileiro (`scripts/diagnose_brazilian_submitter_divergence.py`, commit `3419bea`; saídas em `~/artifacts/redesenho/diagnostico_submissor_br/`)

- **Reprodução fiel:** refizemos o `is_br` do Mosaic com o código dele e bateu em **326.826 de 326.826** exemplos.
  Isso confirma compatibilidade com as flags publicadas, não a nacionalidade real das instituições.
- **A marcação v1 não serve:** a tabela do `lumina-benchmarks` (`eval_unified.parquet`, coluna `cohort`) tinha linha
  para só **9.602 de 190.005** variantes (5%); onde não havia linha, a v1 marcava "não brasileira".
- Categorias (marcações, não nacionalidade): A1 (Mosaic marca, v1 sem linha) **1.580**, com **1.187 no treino v1**;
  A2 (v1 coberta, sem linha brasileira) 133; B (v1 marca, Mosaic não) **1.134**; C1 (v1 "só BR", Mosaic compartilhada)
  **1.026**; C2 6; os dois marcam 206.
- **Por que o Mosaic não conta as SCVs brasileiras de B** (1.336 SCVs): não é classificação (só 6 não são P/B); 959
  têm origem não germinativa (921 `unknown`; Mendelics em 924 das suas 980) e 630 não contribuem para o agregado
  (Dasa 124, INCA 55, Einstein 52, A.C.Camargo 33).
- **T_BR v1 pareado (3.651):** 1.782 estão no Mosaic → 929 compartilhadas, 832 sem marcação e **só 21 só brasileiras**;
  1.869 estão fora (348 não-SNV e 1.521 SNV). Dos 1.688 SNVs do slice fora do Mosaic, **86% tinham agregado de 1
  estrela**. No T_nonBR v1, 46 são marcadas pelo Mosaic.
- **Escopo da lista:** nenhuma SCV P/B de instituição com país Brazil no NCBI fora da lista; mas 138.306 de 932.402
  SCVs P/B (15%) têm país não resolvido. A lista não fica validada por completo.

### 5.3 Tamanhos do estudo brasileiro do Mosaic

- `br_clinical_evidence`: **3.119 casos** (2.808 P / 311 B), sendo 3.116 pareados e 3 sem controle, e **3.116
  controles**. Pareamento 1:1 por rótulo × painel × bin de AF do gnomAD, **sem gene**.
- `br_population_observed`: 1.889 casos (751 pareados, 1.138 sem controle) e 751 controles. Total do membership: 8.875.
- **Só brasileiras com rótulo gold/consensus: 63 (62 P / 1 B).** Compartilhadas: 3.056. Nenhuma variante gold recebe
  marcação brasileira. Por isso o teste só-BR do PDF foi adiado: ele exigiria rótulos de 1 estrela.

### 5.4 Pesquisa de extração (branch `embedding-probe-mosaic`, fechada)

- Candidata: **172 dims de cabeça** (68 W·Δ lineares + 10 MLP + 78 na referência + 16 one-hot de substituição), janela
  de 4 kb, só SNV. Candidata, não vencedora.
- Critério de seleção do MLP unificado com o do ridge e rerodado: ranking preservado (correlação de postos ≥ 0,995);
  sobre a base honesta, a diferença cabeças − v2 (2092 dims) no missense ficou em −0,0004/−0,0002, sem IC.
- A aproximação da leitura antiga avaliada na pesquisa (`infra_atual_pos_norma`, 896 dims) **não tem** o contexto local
  de ±64 bp que a `RegimeAHead` usa; ela fica +0,015 no missense sobre as 172 dims.

### 5.5 Regras do Mosaic que valem para esta campanha (verificadas no código)

- Modo `frozen_pair_evaluation`, sistemas `base` e `regionalized`; nenhum treino, seleção ou calibração dentro do
  estudo. `release_training_allowed`, `release_model_selection_allowed` e `release_threshold_calibration_allowed` são
  **`False`**.
- Dataset de treino = snapshot global **declarado pelo consumidor** (ID, hash e cutoff), não materializado no release.
  O regionalizado parte do mesmo dataset e checkpoint-base e usa obrigatoriamente `abraom_sabe1171`.
- Interação = `delta_br_matched − delta_control`, sem os casos não pareados; deltas na interseção de cobertura.
- Bootstrap por `overlap_cluster_id`, 1.000 réplicas, percentis 2,5–97,5, seed `20260901`. O `comparator_eval` do
  Mosaic monta as coortes brasileiras e reamostra, mas **não calcula a interação**: a regra conjunta é nossa (o plano
  propõe reamostrar clusters em conjunto sobre casos pareados + controles).
- Relatar também o subconjunto `present_abraom` no estudo clínico; declarar sobreposições (SCVs/instituições
  brasileiras, variantes do estudo, outras fontes de regionalização, inclusive o gnomAD do M1).
- `lockbox = False`; `independent_generalization_claim_allowed = False`; margens de sucesso declaradas antes de ver
  scores. Métricas com limiar só com limiar externo congelado.
- O estudo brasileiro **não mudou** entre versões do Mosaic: `brazil_study.py` e as listas de instituições só foram
  tocadas nos commits de 03/09 (o de renomeação mudou um nome de campo); `814e7f0` (07/09) só mexeu em
  `time_promotion`.

---

## 6. Código: o que existe e o que falta

Na branch `new_regionalization`:

| Arquivo | Estado |
|---|---|
| `scripts/audit_regionalization_data.py` + testes | pronto, rodado |
| `scripts/diagnose_brazilian_submitter_divergence.py` + testes | pronto, rodado (10/10 testes com `REQUIRE_NO_SKIP=1`) |
| `scripts/import_mosaic_brazil_studies.py` + testes (G1) | escrito, 15/15 local; **falta rodar no notebook** |
| `scripts/build_core_locus_head_snapshot.py` + testes (G2) | escrito, 14/14 local; **falta rodar no notebook** |
| `scripts/build_broad_brazilian_variant_list.py` + testes | escrito, 5/5 local; gera a lista da regra ampla e conta os controles com SCV brasileira |
| `docs/runbooks/g1_g2_dados_mosaic.md` | runbook dos três passos acima, na ordem |
| `scripts/build_clinvar_splits.py` | v1: monta splits a partir do master regional. Reaproveitável em parte; a marcação BR dele não serve |
| `scripts/prepare_regional_clinvar_dataset.py` | v1: gerou o master regional. Referência histórica |
| `eval/clinvar/train.py` | aplica LoRA e fusion: **falta** um caminho de backbone congelado sem LoRA clínico |
| `eval/clinvar/lora.py` | tem `apply_lora` (exclusion-based) e `freeze_native_feature_heads` |
| `eval/clinvar/heads.py` (`RegimeAHead`) | espera os blocos antigos (`site_ref`, `variant_repr`, contexto ±64 bp) |
| `eval/clinvar/r03_adapter.py` | devolve a leitura antiga (`last_hidden_state` pós-norma) |
| `scripts/train_abraom_frequency_adapter.py` | adapter antigo de frequência (BCE/Huber/MSE). **Não é** o MLM do protocolo |
| `scripts/build_contract_manifest.py` | exige o adapter C e os artefatos v1: precisa de esquema novo |

Falta implementar: importador/validador do membership brasileiro; snapshot externo de treino; extrator portado da
branch `embedding-probe-mosaic` (`eval/embedding_probe/rich.py`) com cache por sistema; cabeça para a extração nova;
gerador de janelas + MLM; consumidor de avaliação do Mosaic (deltas, interação, bootstrap); manifesto do consumidor.

---

## 7. Dados e caminhos (notebook SageMaker)

| O quê | Onde | Identidade |
|---|---|---|
| Release do Mosaic | `~/mosaic-v1/` (`pb_examples.parquet`, `studies/brazil/membership.parquet`) | `membership` com hash lógico `1c1cd65d…` (8.875 linhas); S3 ainda no layout anterior ao ADR 0006 |
| Clone do Mosaic | `~/testeArq/lumina-mosaic` | commit `814e7f0` |
| ClinVar | `~/clinvar/2026-06/submission_summary_2026-06.txt.gz` (sha256 conferido), `organization_summary_2026-09-14.txt` | arquivos mensais ficam em `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/` (sem pasta do ano) |
| Slices e splits v1 | `~/slices_enriched/{br_only,nonbr_only}.enriched.parquet`, `~/artifacts/fase0/t_nonbr_matched_soft.parquet`, `~/artifacts/fase0/clinvar_splits/clinvar_splits_combined.parquet` | **não reutilizar** para a campanha nova |
| Master regional v1 | `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet` | 1.089.826 linhas; tem cobertura e submissores |
| Saídas das análises | `~/artifacts/redesenho/audit_dados.json`, `~/artifacts/redesenho/diagnostico_submissor_br/` | |
| Pesquisa de extração | `~/probe/rich/` (features e evals), `~/hg38/hg38.fa` | |
| gnomAD v4.1 joint | `s3://ai4bio-lumina/data/external/gnomad-joint-v4.1/` | anotador em `lumina-mosaic/src/mosaic/annotations/gnomad.py` |
| ABraOM SABE-WGS-1171 | **a localizar** | sha256 `3cd33784…` nas sources do Mosaic; o TSV não traz AC/AN |

**Não existe `eval_unified.parquet` no notebook** (a tabela v1 de submissores). Por isso a atribuição por submissor é
aproximada, pela coluna `regional_submitters` do master.

---

## 8. Fluxo de trabalho

- **Windows não roda nada pesado.** Claude edita e commita; Gabriel dá push e roda no notebook.
- Nunca `uv sync` no host de GPU. `scripts/env.sh` do `lumina-inference` **não** exporta `$WORK`: incluir
  `export WORK=~/testeArq/lumina-beat-regionalization` nos runbooks.
- O `python3` do notebook (conda) tem pyarrow, pandas e yaml — é o que os scripts novos usam. O `$PY` do
  `lumina-inference` é o de GPU/torch.
- Testes: `REQUIRE_NO_SKIP=1 PYTHONPATH=. python3 tests/<arquivo>.py` — teste pulado conta como falha.
- Ao usar `| tee`, começar com `set -o pipefail` e conferir `$?`, senão o código de erro some.
- `token/token.txt` é untracked e não está no `.gitignore`: **nunca** adicionar; sempre `git add` com caminho explícito.
- A branch `new_regionalization` está **2 commits à frente** do origin em 16/09: `git push origin new_regionalization`.

---

## 9. Próximos passos

1. **Rodar `docs/runbooks/g1_g2_dados_mosaic.md` no notebook** (testes com `REQUIRE_NO_SKIP=1`, depois G1, regra
   ampla e G2) e trazer os relatórios: são os números que fecham o G1 e o G2.
2. **Pedir ao Eduardo o arquivo do ABraOM**, a decisão E e a confirmação dos parâmetros da §5.1 do plano.
3. Escopo dos gates já implementados:
   - **G1 — membership:** importar `studies/brazil/membership.parquet`, juntar com `pb_examples.parquet` para obter
     `chrom/pos_1based/ref/alt`, validar IDs únicos por estudo e papel, ligações bidirecionais, rótulos e estratos, e
     contar P/B por coorte e painel.
   - **G2 — snapshot da cabeça:** `core_locus` `run_id=0` (treino folds 2-4 gold+consensus, validação fold 1 gold,
     teste fold 0 gold), menos os membros dos dois estudos e seus `overlap_cluster_id`, menos a regra ampla
     brasileira, com `sequence_eligible`; medir o custo de cada exclusão e hashear o resultado.
   - **G0 — identidades:** hashes de R03, release e gnomAD (o do ABraOM fica pendente do arquivo).
   - **Contagem de controles com SCV brasileira** pela regra ampla (sai junto com a lista de exclusão), para
     pré-declarar a análise de sensibilidade.
4. Depois: extrator portado e smoke do M0 (G3), gerador + MLM e piloto do adapter misto (G4), escolha da extração na
   validação do core (G5), congelamento e manifesto (G6), avaliação única nos dois estudos (G7).

---

## 10. Armadilhas (já custaram tempo)

- **Treinar no `core_locus` só com as exclusões da §4.2 do plano.** O mantenedor autorizou o protocolo derivado, mas
  `study_membership_used_for_training = False` continua valendo: membros dos dois estudos e seus
  `overlap_cluster_id` ficam fora do treino, senão o estudo perde a validade.
- **Não refazer o pareamento** do Mosaic nem tentar "melhorar" os pares.
- **Não reconstruir o só-BR** nesta rota; e não baixar o `variant_summary` para isso (só se a decisão B pedir).
- **Não usar a marcação BR da v1** para nada além de referência histórica.
- O matcher do Mosaic dá o mesmo resultado para "instituição não reconhecida" e "instituição não brasileira": ao
  reportar, manter separados "não marcado", "país não resolvido" e "não brasileiro pelo registro".
- Diferença sem intervalo de confiança não é empate nem ruído: dizer "próximos, sem incerteza quantificada".
- O diagnóstico para com código 2 se a reprodução do `is_br` não bater 100%; nesse caso, pedir a pasta `interim/` do
  Mosaic ao Eduardo em vez de seguir.

---

## 11. Commits desta frente (branch `new_regionalization`)

| Commit | O que entrou |
|---|---|
| `5c04b2b`, `23d1aad`, `92304ab` | auditoria read-only dos dados v1 |
| `d1ed5b4` | contrato v2 (rascunho) |
| `4e7cd6c`, `e0b2c79`, `f63124a` | contrato v2: rerun do probe, 3ª revisão externa, resultado do diagnóstico |
| `295417d`, `3419bea` | diagnóstico de submissor brasileiro (script, testes e revisão) |
| `848d2dc` | plano com o estudo brasileiro do Mosaic + aviso no contrato |

Na branch `embedding-probe-mosaic`: `1615955` (critério do MLP unificado), `c4d4e97` (sinal por métrica no
comparador), `23fb518` (handoff da extração com os números do rerun).

---

## 12. Achados da revisão de 16/09 (conferidos nos três repositórios)

Detalhes de implementação que faltavam e que decidem como escrever o código dos gates.

1. **O `comparator_eval` do Mosaic não roda nos nossos sistemas.** `point_metrics_for_cohorts`
   (`comparator_eval/pipeline.py:383`) e o bootstrap (`bootstrap.py:109,141,459,492`) iteram
   `OFFICIAL_COMPARATOR_IDS` e leem specs de `config/comparators.yaml`, que é validado contra essa tupla
   (`scores.py:104`). O runner ainda monta os scores a partir das anotações do release (`build_score_table`).
   **Reutilizável como função:** `brazil_views(membership, study_id)` → `full_cohort`/`matched_cases`/`controls`;
   `_with_scores`, que é genérico sobre qualquer coluna `score_*` casada por `variant_id`; `resample_indices`
   (bootstrap por grupo); `metrics_for_panel`, cujo `spec` só é usado em `domain_ok` (um dict sintético serve); e as
   constantes `BOOTSTRAP_SEED=20260901`, `BOOTSTRAP_REPLICATES=1000`, `BOOTSTRAP_PERCENTILES=(2.5, 97.5)`,
   `COHORT_FULL/MATCHED/CONTROLS`, `ALL_PANELS`. O consumidor é nosso; a reutilização é por import, não por CLI.
2. **`variant_id` é um hash opaco**, não coordenada: `var:` + 32 hex de sha256 sobre
   (assembly, sequence, pos, ref, alt) — `src/mosaic/variant_id.py`. O `membership.parquet` só traz `variant_id`,
   então **para pontuar é obrigatório juntar com `pb_examples.parquet`** (`chrom`, `pos_1based`, `ref`, `alt`).
   Isso entra nas interfaces de G1 e G3.
3. **A suíte v1 é só SNV** (`PROTOCOLO.md`: 326.826 SNVs germinativos em GRCh38.p14; `catalog.py` = "Technical SNV
   catalog"). O extrator de 172 dims, que é só-SNV, é compatível com o estudo por construção — a decisão "campanha
   só-SNV" só afeta o snapshot de treino da cabeça, não a avaliação.
4. **Não existe caminho sem LoRA hoje.** `train.py:678` chama `apply_lora` sempre, e `LoRALinear.__init__`
   (`eval/clinvar/lora.py:58`) faz `alpha / rank` → `rank=0` levanta `ZeroDivisionError`; além disso o wrap dos
   `nn.Linear` acontece independentemente do rank. G3 (smoke do M0 com backbone congelado, sem LoRA clínico) exige
   mudança de código: um `use_lora: bool` na config ou um guard `if config.lora_rank > 0`.
5. **O ABraOM tem de vir do Eduardo.** `config/sources.yaml`: `abraom_sabe1171` →
   `abraom/SABE1171.Abraom.clean.tsv`, `obtained: academic_request`, sem URL, sha256
   `3cd3378432909b80053d3a92a1b7d544053a51697623845f6a44f67ba7dbd9d6`, colunas `[chrom, pos, ref, alt, af_abraom]`.
   O índice ABraOM do pipeline v1 não é, por construção, o mesmo objeto: G0 precisa do arquivo com esse hash.
6. **O manifesto do consumidor tem lista fechada.** `BRAZIL_TRAINING_CONTRACT["required_consumer_manifest"]`
   (`brazil_study.py`): `base_checkpoint_id`, `regionalized_checkpoint_id`, `base_training_dataset_id`,
   `base_training_dataset_hash`, `base_training_cutoff`, `abraom_snapshot_hash`, `regionalization_method` — mais
   `study_membership_used_for_training=False` e `study_labels_used_for_training=False`. É o checklist do G6.

**Colunas do `membership.parquet`** (validador do G1, `BRAZIL_MEMBERSHIP_SCHEMA`): `variant_id`, `study_id`,
`member_role`, `matched_variant_id`, `stratum`, `label_tier`, `binary_label`, `primary_panel`, `gnomad_af_bin`,
`overlap_cluster_id`, `core_fold`, `br_lab_any`, `present_abraom`.

**Saídas do R03 relevantes ao MLM** (`lumina-inference/lumina/models/model.py`): `mlm_logits` (com `MASK_ID = 6` em
`lumina/constants.py`), `gnomad_af_pred` e `gnomad_observed_logits` (cabeças populacionais nativas, a congelar),
`hidden_states`/`mid_hidden_states` e `head_outputs`. O `r03_adapter.py` só expõe `last_hidden_state` (448) — o
extrator de 172 dims vive em `eval/embedding_probe/rich.py` na branch `embedding-probe-mosaic`
(`head_readouts`, `MidStackTaps`, `substitution_onehot`, `assert_r03_head_layout`).


---

## 13. Estado em 20/09 (ler primeiro)

### Artefatos prontos, com hash

| Artefato | Conteúdo | Identidade |
|---|---|---|
| `g1_brazil_studies/brazil_study_variants.parquet` | 8.875 membros + coordenadas | validado contra o protocolo |
| `g5_comum/selecao_comum.parquet` | 2.799 variantes, 156 clusters | `sha256 693eb234…` |
| `g2_final_nenhum/core_head_snapshot.parquet` | treino 167.346 (24.097 P) | `9022167f…` |
| `g2_final_janela2048/…` | treino 99.992 (8.421 P) | `af886ad5…` |
| `g2_final_janela4096/…` | treino 86.560 (6.355 P) | `7ffed513…` |
| `g0_fontes/SABE1171.Abraom.clean.tsv` | ABraOM do source-lock | `3cd33784…`, igual ao `sources.yaml` |
| `g4_abraom/abraom_pool.parquet` | 1.224.029 variantes | `40bd0f79…` |
| `g2_verificacao/sha256_declarado.json` | hashes congelados | usado com `verify_campaign_artifacts --esperado` |

O ABraOM está em `s3://croma-bioai-shared-data-us-east-2/lumina/lumina-mosaic/abraom/SABE1171.Abraom.clean.tsv`.
Só 1 das 19 fontes do source-lock mora nessa raiz: é o bucket do arquivo restrito, não um espelho.

### Código novo desta fase

| Script | Papel |
|---|---|
| `verify_campaign_artifacts.py` | portão que o treino chama antes de começar: disjunção, comparabilidade e sha256 |
| `audit_variant_windows.py` | REF contra o FASTA, bordas e `N`; código 2 em qualquer `ref_mismatch` |
| `locate_abraom_source.py` | acha fonte por sha256 do `sources.yaml`; `--root` confere a árvore inteira |
| `audit_abraom_source.py` | publica o pool do adapter com as exclusões e a identidade |
| `build_adapter_window_plan.py` | plano de janelas do adapter (a receita da §5.1) |
| `eval/embedding_probe/{rich,windows}.py` | extrator e janelas portados, fidelidade verificada por texto |
| `eval/clinvar/lora.py` | `rank=0`, `freeze_backbone_in_eval`, `assert_only_head_trains`, `assert_optimizer_covers_trainables`, rsLoRA |

### O que falta, por gate

**G3 (M0):** cache de embeddings por sistema, com a chave ampliada (checkpoint, adapter, extrator, variante, FASTA,
janela, orientação, configuração e ordem das features) e o **smoke real no R03** — só a cabeça treinável, cabeça
muda após um passo, backbone bit a bit idêntico, mesma entrada → mesma representação.

**Provado em 20/09, com escopo estreito:** a **geometria das janelas** do plano confere contra o hg38 — 20.000
janelas **deslocadas** (3.941 índices focais distintos entre 64 e 4.031, `window_start` e spans conferidos), zero
descartes. Isso é REF na posição focal, cabimento e ACGT. **Não** cobre montagem dos alvos mascarados, separação
populacional treino/validação, loss nem treino: tudo isso continua pendente. O auditor declara
`janela_auditada.layout`; se ele disser `centrado` num plano, a auditoria não vale.

**G4, lado dos dados: plano CANDIDATO em 21/09.** ABraOM 1.224.029 (`40bd0f79…`), global 139.495
(`ce749a6d…`) e o **plano `c99e5dae…`**: 50.000 janelas, mistura 0,6 exata, sete bins a ~7.143, 42 substituições
(todas do lado global), zero sem reposição, e auditoria contra o hg38 com **50.000 `ok` e zero descarte**.
Ainda NÃO é congelado: falta rodar a separação por loco sobre ele e fechar a política de piso de AF.

**G4, lado do treino:** o **pool global do gnomAD** (48 VCFs, >500 GB: varredura completa custa horas, amostragem por
`.tbi` custa minutos com viés declarado), o **peso da loss** entre posições de variante e de referência, e o
treinador MLM em si.

### Decisões abertas

1. **Confundimento espacial** entre as fontes da mistura: o pool do ABraOM é concentrado (chr16 aparece mais que
   chr1). Proposta registrada: casar a distribuição por cromossomo do pool global à do ABraOM, para que a única
   diferença seja qual variante é aplicada. **Sem isso, o adapter pode separar as fontes pela região.**
2. **Decisão E** do Eduardo: chr8 reservado (custa 61.737 variantes do pool) e relato BRCA/TP53.
3. **Qual campo de AF** do gnomAD joint conta como "AF global" — ele tem dezenas, e a escolha tem de ser declarada.
   Antes disso: **o que "global" significa** (AF agregada × amostragem por grupos ancestrais) é decisão de desenho.
6. **Assimetria de presença no ABraOM no estudo clínico** (casos 10,4% × controles 2,3%, 4,6×): parte de um ganho
   pode vir de "estar no ABraOM" e não de "participação brasileira". Sensibilidade proposta: repetir a interação
   nos **pares em que caso e controle estão ambos ausentes**, preservando o pareamento do release (tamanho a medir
   pelo `matched_variant_id`, não por subtração). O resultado não conclui sozinho em nenhum dos dois sentidos.
7. **Procedência do `SABE1171.Abraom.clean.tsv`**: 448 variantes/Mb e 5,9× de variação entre cromossomos indicam
   subconjunto filtrado, não callset completo. Perguntar o critério ao Eduardo.
4. **Peso da loss** nas posições de variante × referência.

### Fixado nesta fase (não reabrir)

- Mistura **60/40 global/ABraOM** confirmada pelo Eduardo em 20/09: "focar no ABraOM" era prioridade de trabalho,
  não mudança de desenho.
- **rsLoRA sim, MiCA não** — migrar para o `peft` daria trabalho e exigiria reproduzir a superfície de exclusão
  das cabeças nativas do R03. rsLoRA está implementado nativo, default desligado, e entra no manifesto.
- Política para `N`: soft-mask normalizado, janela com base fora de ACGT descartada. Custo medido: **zero**, porque
  o `sequence_eligible` do release já garantiu ACGT na janela de 32 kb.

---

## 14. Estado em 22/09 (ler primeiro — substitui a §13 como ponto de partida)

O adapter existe, treina no R03 real e foi medido. Duas revisões externas no mesmo dia corrigiram a primeira
leitura e acharam um bug no bootstrap. O raciocínio completo está em `docs/diagnostico_do_adapter_regional.md`.

**Foco: o R03.** A v11 é referência — hipóteses a testar, lições de método, ferramentas —, não evidência sobre o
R03: outro backbone, outro objetivo, cabeça treinável e lote diferente.

### 14.1 Três perguntas, e onde estamos

| Pergunta | Comparação que responde | Estado |
|---|---|---|
| O adapter melhora a reconstrução mascarada? | R03 antes × depois, na validação populacional separada por loco | **avançamos aqui** |
| A adaptação mista melhora a classificação nos estudos brasileiros? | **M0 × MR**, com avaliação clínica congelada | **é a pergunta da campanha**; depende do G3 |
| O componente ABraOM acrescenta algo ao global? | **MG × MR**, mesma receita, orçamento e avaliação | adiada (ablação proposta) |

Melhora de MLM mede reconstrução na distribuição de janelas construída: pode vir de contexto de sequência, de
calibração ou de adaptação à receita. Não identifica, sozinha, aprendizado populacional, nem diz se a
representação ajuda a cabeça clínica.

Os dois estudos brasileiros têm papéis distintos e nunca se somam: `br_clinical_evidence` (participação de
instituição brasileira) é a avaliação principal; `br_population_observed` (presença no ABraOM) dá evidência sobre o
sistema naquele recorte, sobreposto ao ABraOM por construção — não sobre generalização.

Os dados do G4 (pool ABraOM `40bd0f79…`, pool global `ce749a6d…`, plano `c99e5dae…`, separação por loco
`c76d08d4…` / `034eca34…`) são **candidatos identificados por hash**. Piso de AF, geografia, procedência do ABraOM e
peso da loss continuam abertos.

### 14.2 O adapter está provado no modelo real

`scripts/train_population_adapter.py --smoke` → **17/17, `passou: true`**, sobre o R03 de verdade:
`checkpoint_sha256 = f2983560f8f965…` (bate com o contrato), logits `(1, 4096, 4)`, loss em tensores igual à do
núcleo sem torch, **backbone congelado idêntico por hash** depois de um passo, e uma instância nova, construída da
base com o adapter carregado, reproduzindo as predições com diferença **0,00e+00**.

Superfície congelada em `configs/adapter_r03_superficie.json`: **99 módulos**. As camadas 8 e 17 (atenção
esparsa) **participam do cálculo** — a `strided_attn` roda em todo forward, alimentada por camadas adaptadas —, mas
sem LoRA direto (o embrulho em `nn.MultiheadAttention` é inerte) e sem o caminho de âncoras, que só liga com
`edit_mid_mask`.

**Custo medido** com as 44.645 janelas de treino: carga 21,5 s, pico **3,9 GB**, **0,140 s/exemplo** de treino e
**0,088 s/exemplo** de validação.

### 14.3 O que a corrida de 1.000 passos mostrou

| Corrida | Treino disponível | Atualizações | LR | `focal_val` |
|---|---:|---:|---:|---:|
| linha de base (R03 sem delta) | — | 0 | — | 1,7399 |
| `escala` | plano completo | 20 | `1e-4` | 1,7305 |
| piloto 5 | 400 exemplos repetidos | 300 | `1e-4` | melhor 1,7079 (passo 89); final 1,9991 |
| `g4_lr5e6` | plano completo | 1.000 | `5e-6` | **1,6932** |

Todas na mesma validação de 160 janelas. **O que se sustenta:** a receita com `5e-6`, 1.000 atualizações e dados
sem repetição reduziu a perda focal em −0,0467, e o melhor ponto avaliado foi o último — resultado positivo pelo
critério primário declarado. **O que não se sustenta:** atribuir isso à taxa (mudaram taxa, atualizações e
repetição de dados juntos), dizer que "a curva ainda descia" (só a linha final foi vista; a curva está em
`~/artifacts/redesenho/g4_lr5e6/treino_do_adapter.json`) e tratar `5e-6` como a taxa do R03. Ela é
**configuração candidata**.

Diagnósticos da mesma corrida:

| | ABraOM | global |
|---|---:|---:|
| termo de massa | −0,0441 | −0,0412 |
| termo de escolha | −0,0087 | −0,0015 |
| entropia no focal | +0,025 | +0,018 |
| fração média do ALT entre as não-referência | −0,0037 | −0,0027 |
| perda nas posições de referência | +0,0013 | +0,0030 |

O termo de escolha caiu, mas **suavizar uma distribuição confiante demais também o derruba**, sem mudar a ordem
entre as alternativas (frações 0,90 e 0,02 com temperatura 1,5 viram 0,77 e 0,06). A entropia subindo, a fração
caindo e a referência piorando são compatíveis com isso. O runner agora mede a **ordem** do ALT entre as três
não-referência (`alt_em_primeiro_entre_nao_ref`, `posto_do_alt_entre_nao_ref`), que temperatura e massa não mexem.
São **diagnósticos, não portões**: a ordem também muda por alteração genérica, probabilidade melhor sem mudar a
ordem é ganho legítimo, e nada disso descreve a representação que a cabeça clínica vai ler.

Um controle de temperatura (um escalar no R03 congelado, ajustado em dados de desenvolvimento separados) diria
quanto do ganho um ajuste trivial reproduz nessa métrica. Fica como diagnóstico opcional.

### 14.4 Bootstrap: dois defeitos corrigidos antes de qualquer leitura

1. **Pareamento por `variant_id`.** Ele é `chrom:pos:ref:alt`, sem a fonte, e o mesmo alelo pode estar nas duas
   metades da mistura. Com antes e depois **idênticos**, o delta do ABraOM saía +0,25. Agora a chave é
   `fonte|variant_id|focal_index`; janela repetida, loco ausente ou mudança de loco **recusam** o bootstrap; a saída
   informa **janelas e locos por fonte**; e validação sem `locus_id` para **antes** de carregar o modelo.
2. **Detalhe final descartado** quando o último passo caía na cadência de validação. A corrida de 3.000/250 cairia
   nisso e o bootstrap sairia "indisponível". Agora o detalhe do final e o do melhor são guardados, há
   `bootstrap_do_melhor` (o adapter que se usa) e os três conjuntos vão para `detalhe_da_validacao.json`.

Leitura: o IC é **condicional ao modelo escolhido nessa mesma validação** e não inclui variação entre sementes;
cruzar zero não prova ausência de efeito; e `abraom_menos_global` compara amostras distintas — **não substitui o
MG × MR**.

### 14.5 Correções de leitura (não repetir)

- **O resíduo já foi testado na v11**, com AF **observada** (`logit(af_abraom) − logit(af_gnomad)`): +0,026
  [+0,001; +0,050] no teste, IC da validação cruzando zero, seleção de checkpoint degenerada. A versão com a
  previsão nativa é **outra proposta**: a `population_af_head` prevê log-AF, o resíduo carregaria também erro do
  preditor, cobertura e ruído, e seria outra tarefa de treino. **Hipótese para depois, não bloqueio.**
- **256 não é peso de perda**: é `_DEFAULT_MIN_VALID` do normalizador EMA. As perdas de treino não estão no pacote
  de inferência.
- **Camadas 8 e 17 não estão fora do cálculo** (§14.2), e isso não demonstra incompatibilidade entre R03 e MLM.
- **4.096 ≈ 1.024** foi medido na regressão de AF da v11; não garante nada no MLM do R03.
- **chr8:** a avaliação representacional da §11.1 depende da **decisão E**. Se o chr8 for teste final, não se
  consulta repetidamente para escolher receita.
- **MG:** "se der certo a gente volta e tenta explicar" define a **prioridade**, não autoriza execução nem
  orçamento. O MG é a ablação de atribuição **proposta**.

### 14.6 Decisões de 22/09 e próximos passos

**Aprovado pelo Gabriel em 22/09:** a corrida de 3.000 passos segue como **desenvolvimento**; o próximo objetivo é
**viabilizar a comparação clínica M0 × MR**, sem portões novos baseados nos diagnósticos do MLM; ICs exploratórios;
recortes de desenvolvimento explícitos, sem o fold 0; combinação de sementes declarada. Tudo registrado em
`configs/campanha_r03_desenvolvimento.json` e nas §4.2, §5.1 e §5.3 do plano:

| Declaração | Conteúdo |
|---|---|
| recortes | treino = `train` do snapshot; parada e calibração = `validation` (fold 1); comparação de desenvolvimento = seleção comum (`693eb234…`); **fold 0 e estudos brasileiros fora de todo o desenvolvimento** |
| o que o desenvolvimento não mede | a interação regional: os recortes excluem membros e a regra ampla brasileira. M0 × MR ali é classificação geral |
| sementes | adapter 20260921 / 20260922 / 20260923; cabeça 11 / 12 / 13; **pareadas** (MR_i = a_i + h_i; M0_i = h_i) |
| adapter do MR | receita candidata (`5e-6`, 3.000 passos, validação de 800 a cada 250); `adapter_melhor.pt` pelo `focal_alt`; nunca escolhido por métrica clínica ou diagnóstico |
| intervalos | exploratórios: condicionais aos modelos escolhidos, sem variação entre sementes de adapter enquanto só houver a₁ |

**Corrida a₁** (`~/artifacts/redesenho/g4_corrida2`): lançada com `nohup` e saída sem buffer, para sobreviver ao
fechamento do terminal. Log completo em `~/artifacts/redesenho/g4_corrida2.log`; relatório com a curva inteira em
`treino_do_adapter.json`, escrito só no fim; checkpoints a cada 500 passos. Para colar na conversa:
`python3 scripts/resumir_treino_do_adapter.py ~/artifacts/redesenho/g4_corrida2` (funciona também na corrida
`g4_lr5e6`, cuja curva nunca foi vista).

**Resultado da corrida a₁ (23/09, `exit=0`), lido pelas regras declaradas:**

| | ABraOM | global |
|---|---:|---:|
| `focal_alt` (base → final) | 1,7724 → 1,6378 | 1,7066 → 1,5758 |
| delta, IC 95% exploratório por loco | −0,135 [−0,169; −0,102] | −0,131 [−0,162; −0,101] |
| termo de massa / termo de escolha | −0,120 / −0,014 | −0,107 / −0,024 |
| ALT em 1º entre as não-referência | +0,006 [−0,006; +0,019] | +0,004 [−0,002; +0,013] |
| entropia no focal | +0,068 | +0,068 |
| perda nas posições de referência | +0,009 | +0,006 |

- **Critério primário:** 1,7331 → **1,6008** (−0,132) nas 800 janelas; curva monotônica, achatando depois de ~1.750
  passos (quando o cosseno leva a taxa perto de zero); melhor = último. Pela regra, **a₁ = `adapter_melhor.pt`
  `6327a9fa…` está congelado** (registrado em `adapters_congelados` na declaração).
- **Diagnósticos (descrição):** 82–89% do ganho está no termo de massa (global e ABraOM); a **ordem** entre as alternativas não se
  moveu de forma distinguível; entropia subiu e a referência piorou. O que mudou no MLM foi sobretudo a confiança
  nas posições mascaradas.
- **Fontes:** nenhuma diferença distinguível (`focal_ce` −0,004 [−0,051; +0,042]). Não substitui o MG × MR.
- **Linha de base útil:** o R03 já põe o ALT verdadeiro em 1º entre as três não-referência em 44,7% (ABraOM) e
  48,3% (global) das janelas (acaso: 33,3%).

**Próximos passos:**
1. **G3, extração e cache por sistema** (M0 agora; MR quando a₁ terminar), cobrindo o treino de `nenhum` (as três
   políticas são aninhadas), o fold 1 e a seleção comum — e recusando o papel `test` e os estudos por construção.
2. **Cabeça sobre o cache**, com as sementes declaradas, parada e calibração no fold 1.
3. **G5 só com M0**: extração e política pela regra da §4.2.
4. **M0 × MR exploratório** na seleção comum, com o adapter a₁ congelado.
5. **Para o Eduardo:** o MG como ablação proposta, a decisão E e o resíduo como hipótese posterior.

### 14.7 G3: o que já existe e o que falta

**Existe** (commits na `new_regionalization`):

| Peça | Commit |
|---|---|
| caminho sem LoRA clínico (`rank=0`) para o M0 | `85d0f11` |
| congelamento real do backbone e snapshots finais de treino | `bffe670` |
| extrator de 172 dims portado da pesquisa, com teste de equivalência (`eval/embedding_probe/rich.py`) | `2dcf27d` |
| configuração M0/MR imposta pela config e ligada ao treino (`eval/clinvar/config.py`) | `85e8a9c` |
| `windows.py` portado e auditoria de janelas | `44bf674` |
| `variant_encoder` no otimizador e recarga do rsLoRA na avaliação | `101bec9` |

**Caminho escolhido em 23/09: cache, não treino ponta a ponta.** O `eval/clinvar/train.py` refaz o forward do
backbone a cada época. Com o backbone congelado em M0 e em MR, isso dá o mesmo número a um custo de horas por
época; extrair uma vez e treinar as cabeças sobre o cache é exato e barato.

**Escrito em 23/09 (testes sem GPU passando; smoke real pendente):**

| Peça | O que faz |
|---|---|
| `eval/campanha/recortes.py` | lê a declaração e **impõe** os recortes: recusa o papel `test`, variante do fold 0 em qualquer entrada, membro dos estudos, chr8 e variante em dois papéis |
| `eval/campanha/layout.py` | lote de tamanho fixo `[ref_0, alt_0, …]`, completado com cópias; limites do contexto local; dimensões das extrações |
| `eval/campanha/leituras.py` | as duas candidatas do mesmo forward: `cabecas_172` (68 W·Δ + 10 MLP + 78 na referência + 16 de substituição) e `leitura_antiga_1344` (sítio, alt − ref, média em `[f−64, f+64)`) |
| `scripts/extract_campaign_features.py` | carrega M0 ou MR (confere o sha do adapter contra a declaração, a superfície de 99 módulos e o conjunto exato de chaves), extrai em fragmentos retomáveis, recusa retomar com identidade diferente; `--smoke` mede custo, determinismo, independência da posição no lote e, em MR, que o adapter está ativo |

**Smoke no R03 (23/09): M0 e MR passaram.** Tabela de 171.720 variantes; determinismo do mesmo lote = 0 exato;
dependência da vizinhança no lote = 1,9e-6 (M0) e 1,4e-6 (MR) — não zero, como a pesquisa tinha medido; o que
protege a comparação é o protocolo idêntico nos dois sistemas, registrado com a tolerância de 1e-5. Adapter ativo
na representação sem máscara (0,66 nas 172 dims, 0,051 na leitura antiga: mostra que está ativo, não que melhora,
e as duas escalas não se comparam). Custo: M0 ~2,4 h, MR ~3,3 h. As janelas conferidas no smoke são só as 48 da
amostra; a tabela inteira é conferida na extração.

**Revisão de 23/09 do cache, antes da rodada longa (reproduzida e corrigida, `eval/campanha/cache.py`):**

| Defeito | Correção |
|---|---|
| "completo" com variante faltando (`len(total) == len(tabela) − falhas`) | completo = **toda** variante da tabela no cache; falha de janela sai em `falhas.json` e dá saída 2 |
| identidade só com `variant_id + papel` | hash de **conteúdo** (coordenadas, alelos, rótulo, painel, cluster, tier, papel), sha256 dos arquivos que determinam os números e do pacote `lumina` importado, e ambiente (torch, CUDA, GPU) |
| tabela reescrita a cada retomada | gravada uma vez na criação; na retomada, só conferida |
| fragmento gravado direto no nome final | temporário + `fsync` + troca atômica; `.tmp` de queda é apagado na retomada |
| retomada lia só os ids | cada fragmento é validado: legível, formas, finitude, duplicatas, ids e papéis contra a tabela |
| próximo fragmento = contagem de arquivos | maior índice + 1 |
| finitude conferida só no smoke | conferida em todo fragmento, antes de gravar |
| duas execuções no mesmo cache | trava com o PID; trava de processo morto é removida com aviso |

Consequência operacional: **durante a extração, não mudar os arquivos listados em
`ARQUIVOS_QUE_DETERMINAM_AS_FEATURES`** — uma retomada seria recusada. O código novo do G3 vai em arquivos novos.

**Falta:** o ensaio curto do caminho completo (só a validação, ~3 min); a extração do M0 e do MR (~5,7 h, com log
por tentativa e PID); a cabeça sobre o cache com as sementes declaradas; o G5 só com M0; o comparador exploratório
M0 × MR na seleção comum; e, para o G7, o consumidor do Mosaic (interação e bootstrap conjunto por cluster, nos dois
estudos).

### 14.8 O padrão de erro a não repetir

As revisões pegaram, mais de uma vez, **mecanismo afirmado a partir de diagnóstico** ("achatou", "REFUTADO",
"CONFIRMADO", "soube qual alelo"), **comparação confundida** tratada como causal, **v11 usada como evidência sobre
o R03**, e bug de pareamento que teste feliz não pega. Medir antes de concluir; diagnóstico não é portão; e não
trocar a pergunta clínica por uma sequência indefinida de diagnósticos do MLM.
