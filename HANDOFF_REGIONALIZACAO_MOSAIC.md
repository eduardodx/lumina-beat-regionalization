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
- **"99 camadas LoRA" está errado: são 99 módulos adaptados** (projeções distribuídas pela arquitetura). O MR mais
  lento é compatível com o custo do adapter, mas o tempo também depende das condições da execução.
- **Ensemble ≠ média das macros por semente.** O G5 mede a média das macros das três cabeças; o comparador mede a
  macro da média das probabilidades calibradas. As individuais do M0 é que devem reproduzir o G5.
- **Consistência entre sementes de cabeça** mostra consistência nessas execuções; não estima o otimismo da seleção
  e não substitui replicação do adapter.
- **"Mesmo sinal em todos os painéis"** vale para os três da macro; plof (1 benigna) é frágil e synonymous não tem
  AUROC.
- **Comparação exploratória não é critério de aprovação.** Um delta negativo merece registro e investigação, não
  autoriza concluir fracasso; um positivo não demonstraria benefício brasileiro. Uma regra de não degradação só vale
  com métrica, margem e conjunto definidos **antes**.
- **"Os números são confiáveis"** → "a execução e os números apresentados são consistentes": coerência interna não é
  auditoria completa, e a confiança científica depende também do desenho e da amostra.

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
  passos; melhor = último. (Corrigido em 24/09: no passo 1.750 a taxa **não** está perto de zero — pelo cosseno do
  runner é ~1,89e-6, 38% da inicial —, então o achatamento não se explica só pelo scheduler.) Pela regra, **a₁ =
  `adapter_melhor.pt` `6327a9fa…` está congelado** (registrado em `adapters_congelados` na declaração).
- **Diagnósticos (descrição):** 82–89% do ganho está no termo de massa (global e ABraOM); a **ordem** entre as alternativas não se
  moveu de forma distinguível; entropia subiu e a referência piorou. O que mudou no MLM foi sobretudo a confiança
  nas posições mascaradas.
- **Fontes:** não detectamos diferença de melhora entre ABraOM e global (`focal_ce` −0,004 [−0,051; +0,042]) — o
  IC inclui zero, o que **não** é equivalência. Não substitui o MG × MR.
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

**Ensaio (23/09):** o caminho completo passou no GPU (1.575 variantes, completo, `exit=0`) e a retomada também
(segunda execução: `identidade confere`, 0 pendentes).

**Extração M0 (23/09): completa.** 171.720 de 171.720, zero falha de janela, `exit_M0=0`, 0,055 s/variante. O MR foi
disparado em seguida no mesmo job.

**Cabeça, G5 e comparador (23/09, escritos durante a extração do MR, só em arquivos novos):**

| Peça | O que faz |
|---|---|
| `eval/campanha/metricas.py` | AUROC (Mann-Whitney, empate meio ponto, arredondamento 1e-12 relativo) e macro de missense/splice/noncoding portadas da pesquisa; AUPRC; bootstrap pareado por `overlap_cluster_id` (1.000, seed 20260901, exploratório) |
| `eval/campanha/leitura_do_cache.py` | lê um cache completo, confere o hash de conteúdo, alinha a matriz à tabela; `conferir_par` recusa M0 e MR que difiram além do adapter; `linhas_da_politica` recusa política não aninhada |
| `eval/campanha/cabeca.py` | o MLP da pesquisa, portado sem mudar; parada pela macro no fold 1; Platt e limiar de MCC no fold 1; `rodar_sementes` é a ÚNICA função que treina cabeça (G5 e comparador) |
| `eval/campanha/g5.py` | a regra da política (plano §4.2) e a da extração (proposta abaixo) |
| `scripts/g5_escolher_extracao_e_politica.py` | 2 extrações × 3 políticas × 3 sementes só com M0; recusa cache de MR; exige `--confirmo-a-regra-da-extracao`; grava a decisão com o sha da identidade do cache do M0 |
| `scripts/comparar_m0_mr_desenvolvimento.py` | exige a decisão do G5 feita com o MESMO cache do M0, caches pareados e o adapter declarado; H0 e HR com as mesmas sementes; métricas por semente, da média, e IC exploratório |

**Regra da extração do G5 — PROPOSTA em 23/09, registrada antes de qualquer score:** o plano só dizia "critério
declarado antes". Aplica-se a regra da política dentro de cada extração; ganha a extração com a maior macro média
na política escolhida; empate exato fica com `cabecas_172`. Quem roda o G5 confirma com a flag.

**Revisão de 23/09 da cabeça, antes do G5 (reproduzida e corrigida):**

| Defeito | Correção |
|---|---|
| Platt por Newton de passo cheio a partir de a = 1: em `[-10, -8, 8, 10]` dava a = 6,8e9, probabilidades 0/1 e estouro (o ótimo é a ≈ 0,121) | Lin, Lin e Weng (2007): começa em a = 0, perda com os alvos suavizados calculada sem estouro, Newton com busca em linha; sigmoide estável |
| métricas por semente sobre a probabilidade calibrada (a saturação vira empate) | métricas de ordem sobre os LOGITS; a probabilidade calibrada fica para a média entre sementes e para os limiares |
| G5 registrava o caminho dos snapshots, não o conteúdo | G5 grava o sha256 dos três; o comparador recusa snapshot diferente do da decisão |
| cabeças não eram salvas | o comparador salva cada cabeça inteira (pesos, padronização, Platt, limiar, identidades), para pontuar os estudos sem retreinar |

Viés declarado no comparador: o conjunto de seleção foi usado para escolher, só com M0, a extração e a política
(1 de 6, pela regra do G5). A comparação nele **pode favorecer o M0, com viés de tamanho desconhecido** — que não
serve para descontar uma queda do MR (revisão de 23/09).

**Extração MR (23/09): completa.** 171.720 de 171.720, zero falha, `exit_MR=0`, 0,074 s/variante.

**G5 (23/09, só M0, regra da extração confirmada pelo Gabriel ao rodar): `leitura_antiga_1344` + `janela2048`.**
Macro-AUROC (missense/splice/noncoding) no conjunto de seleção, 3 sementes:

| | `janela4096` (86.560) | `janela2048` (99.992) | `nenhum` (167.346) | política pela regra |
|---|---:|---:|---:|---|
| `cabecas_172` | 0,9004 | 0,9024 | 0,9095 | `janela4096` (a 0,0091 da melhor) |
| `leitura_antiga_1344` | 0,9056 | **0,9144** | 0,9193 | `janela2048` (a 0,0049; `janela4096` fica a 0,0137, fora) |

A leitura antiga vence nas três políticas (+0,005 a +0,012), e na escolhida a pior semente dela (0,9133) passa a
melhor da de 172 (0,9044). Isso é consistência **nessas três execuções, no mesmo conjunto de seleção**: não elimina a
incerteza do conjunto nem cobre outras inicializações. A escolha vale entre os candidatos testados, com esta receita
de cabeça — não diz que a leitura antiga é superior em qualquer conjunto — e não contradiz a pesquisa de extração,
onde as 172 dims eram candidata compacta, não vencedora. A macro na faixa da pesquisa é **checagem de
plausibilidade**, não validação da reprodução (recortes e composição são outros); a implementação se valida pelos
testes e pela conferência do processamento. A política troca ~0,005 de macro por mais isolamento de locus, como a
regra prevê. Decisão em `~/artifacts/redesenho/g5/g5_decisao.json`, com o sha da identidade do cache do M0 e dos três
snapshots.

**Comparação exploratória M0 × MR (23/09; leitura antiga + `janela2048`; adapter a₁; cabeças 11/12/13;
`exit_comparacao=0`; saída em `~/artifacts/redesenho/comparacao_dev_a1`):**

| Ensemble no conjunto de seleção (média das probabilidades calibradas das 3 cabeças) | M0 | MR | delta | IC 95% exploratório (1.000, por cluster) |
|---|---:|---:|---:|---|
| **macro (missense/splice/noncoding) — critério declarado** | 0,9178 | 0,9137 | **−0,0041** | [−0,0090; +0,0010] |
| AUROC geral (secundária) | 0,9683 | 0,9658 | −0,0025 | [−0,0051; −0,0006] |
| AUPRC geral (secundária) | 0,9228 | 0,9206 | −0,0022 | [−0,0057; +0,0013] |

| Painel (ensemble) | P | B | M0 | MR | delta |
|---|---:|---:|---:|---:|---:|
| missense | 197 | 345 | 0,8451 | 0,8414 | −0,0037 |
| splice | 158 | 144 | 0,9883 | 0,9878 | −0,0005 |
| noncoding | 100 | 1.014 | 0,9201 | 0,9118 | −0,0083 |
| plof | 212 | **1** | 0,9575 | 0,9575 | empate na precisão mostrada; com 1 benigna, **frágil** |
| synonymous | 0 | 628 | — | — | sem AUROC (só benignas) |

Por semente (macro sobre os logits): h11 −0,0036, h12 −0,0098, h13 −0,0002. As cabeças do M0 reproduzem as do G5
(0,9134 / 0,9166 / 0,9133) — coerência interna entre os dois caminhos, não auditoria completa dos artefatos.
**A macro do ensemble (0,9178) não é a média das macros por semente (0,9144, a medida do G5):** são medidas
diferentes e não precisam coincidir.

**Leitura (corrigida pelas revisões de 23/09):** **pequena piora estimada** na classificação geral, com incerteza que
**inclui ausência de diferença** — o IC da macro contém zero, com a maior parte no lado negativo. Não demonstra
equivalência nem ausência de dano.
- As três cabeças deram delta negativo, mas **compartilham adapter, dados e conjunto**: não são replicações
  independentes do treino populacional.
- A queda é maior em noncoding (−0,0083): registrado, **não** é motivo para mudar a receita nem para escolher outra
  métrica.
- A AUROC geral, com IC inteiro abaixo de zero, é **sinal secundário de piora**, relatado sem substituir o critério.
  Ela mistura pares de painéis diferentes (plof quase só P, synonymous só B); quanto isso pesa **não foi medido**.
- O conjunto de seleção pode favorecer o M0, com viés de tamanho desconhecido, que não desconta a queda. Consistência
  entre sementes **não** estima esse viés: todas foram avaliadas no conjunto que escolheu a configuração.
- **Isto não é a pergunta regional**, que só o G7 mede. Não havia condição de parada para este resultado.

Continua em §14.9 (decisão e caminho até o G7).

### 14.8 O padrão de erro a não repetir

As revisões pegaram, mais de uma vez, **mecanismo afirmado a partir de diagnóstico** ("achatou", "REFUTADO",
"CONFIRMADO", "soube qual alelo"), **comparação confundida** tratada como causal, **v11 usada como evidência sobre
o R03**, e bug de pareamento que teste feliz não pega. Medir antes de concluir; diagnóstico não é portão; e não
trocar a pergunta clínica por uma sequência indefinida de diagnósticos do MLM.

Em 23/09, o mesmo padrão do outro lado: **uma comparação exploratória virando portão informal** ("decidir com o
Eduardo antes de seguir") e **justificativas mais fortes que a evidência** ("dominada", "sugere viés pequeno",
"confiáveis"). O resultado exploratório se registra com estimativa e limites; a receita só muda por regra declarada
antes.

### 14.9 Depois da comparação: decisão e caminho até o G7 (23/09, após duas revisões)

**Decisão: manter a receita e seguir o plano.** A comparação M0 × MR foi declarada exploratória e o delta não era
condição de parada. Migrar para o resíduo mudaria o objetivo; o MG é ablação de atribuição útil, não correção
obrigatória. Configuração mantida: `leitura_antiga_1344`, `janela2048`, mesma receita de cabeça. Confirmar o
orçamento (~10 h de GPU) com o Eduardo é razoável, mas **o resultado sozinho não cria necessidade científica de
autorização nem exige redesenhar a campanha.**

**A pergunta continua aberta:** a adaptação mista ajuda mais os casos brasileiros que os controles, e quais são os
ganhos ou perdas absolutos em cada grupo? Uma interação positiva pode vir de os controles piorarem mais, então o
G7 mostra **a interação e o desempenho absoluto de cada grupo juntos**, como o plano prevê.

**O G7 continua protegido.** Não consultar o G7 para ajustar nada. "Avaliação única" significa não usar o resultado
para adaptar a receita e depois tratar a mesma avaliação como confirmação independente. Reexecutar exatamente os
sistemas congelados para conferir reprodução não invalida o estudo. Mudança motivada pelo resultado vira rodada
**exploratória**, ou pede outra avaliação independente.

**Feito depois das revisões (sem tocar nos 12 arquivos da identidade do cache):**

| Peça | O que faz |
|---|---|
| `--seed-da-validacao` no runner | a subamostra da validação tem semente própria, **obrigatória** com `--limite-validacao`. Antes era `seed + 1`: a₂ seria validada em outro recorte, o que acrescentaria uma fonte de variação e descumpriria o desenho acordado (não "tiraria o sentido" da comparação, como eu tinha dito). O recorte vai para o relatório e para o checkpoint (sha256 das chaves `fonte\|variant_id\|focal_index`) |
| `--recorte-da-validacao-igual-a <pasta>` | **aborta antes de carregar o modelo** se, contra uma corrida anterior (funciona na a₁), diferir o recorte sorteado (do `detalhe_da_validacao.json`) **ou** o sha256 do plano de validação, do plano de treino ou do checkpoint (do `treino_do_adapter.json`). O recorte só identifica as janelas dentro do plano; as máscaras e o resto dos dados estão no plano (terceira revisão de 23/09) |
| resumidor | imprime a semente e o recorte da validação; na a₁, calcula o recorte do detalhe |
| `scripts/conferir_cabecas_salvas.py` | recarrega as 6 cabeças e confere formato, identidades (decisão do G5, snapshot, cache, adapter), Platt e limiar contra o relatório, `a > 0`, receita, **reprodução das probabilidades** (≤ 1e-6) e das métricas do ensemble; grava `conferencia_das_cabecas.json` com o sha256 de cada arquivo (entra no G6) |
| `scripts/conferir_codigo_do_cache.py` | antes de uma extração longa, confere em segundos que código, pacote `lumina` e ambiente são os do cache do M0 |
| comparador | textos corrigidos (ensemble, viés de tamanho desconhecido, cabeças não independentes); passa a imprimir Platt, limiar e onde salvou as cabeças, e marca painel frágil |
| `eval/campanha/cabeca.py` | `montar_rede` (uma definição para treinar e recarregar), `carregar_cabeca_salva`, `pontuar_salva` |

**Composição final, declarada em `g6` na declaração antes da a₂ e da a₃ (e travada por `tests/test_declaracao_g6.py`):**

| Sistema | Componentes | Arquivos (em `~/artifacts/redesenho`) |
|---|---|---|
| M0 | M0 + h11, M0 + h12, M0 + h13 | `comparacao_dev_a1/cabeca_M0_h{11,12,13}.pt` (os comparadores da a₂ e da a₃ têm de reproduzi-las) |
| MR | a₁ + h11, a₂ + h12, a₃ + h13 | `comparacao_dev_a1/cabeca_MR_h11.pt`, `comparacao_dev_a2/cabeca_MR_h12.pt`, `comparacao_dev_a3/cabeca_MR_h13.pt` |

Os três comparadores produzem **nove** cabeças MR. Misturá-las daria outro ensemble; escolher uma depois de olhar
resultado é proibido. Predição do sistema = média das três probabilidades calibradas.

**Limiar do ensemble:** os limiares individuais não definem o da média. Se o G7 reportar MCC, sensibilidade ou
especificidade do ensemble, o limiar é o de MCC na média das probabilidades no fold 1, por sistema, congelado no G6
com proveniência (o Mosaic exige limiar externo congelado para métricas com limiar). Não bloqueia a₂/a₃.

**Antes do G7, separar a decisão científica da implementação:**
- **Margens (ABERTO, do Eduardo):** o protocolo do Mosaic (§13.5 do `PLAN.md`) exige, antes de ver scores, a
  margem mínima de melhoria no coorte BR, a margem máxima de regressão no controle e os painéis em que regressão é
  inaceitável. O "0,02" do plano não diz sobre qual quantidade incide: ganho absoluto, interação e não inferioridade
  geral são perguntas diferentes.
- **Bootstrap (PROPOSTO, §6.3):** reamostrar `overlap_cluster_id` em conjunto, com os mesmos sorteios para M0 e MR; o
  consumidor implementa e testa a regra. A unidade é mudança em relação ao PDF (matched set) e merece alinhamento;
  réplicas e seed são detalhe operacional da equipe.

**Próximos passos, em ordem:**
1. Testes e conferências no notebook com interrupção na primeira falha (`set -euo pipefail`; `| tail` esconde
   reprovação); conferir as 6 cabeças da a₁ antes de qualquer congelamento.
2. Treinar a₂ e a₃ com a mesma receita e orçamento, mudando só `--seed`, com `--seed-da-validacao 20260922` e
   `--recorte-da-validacao-igual-a ~/artifacts/redesenho/g4_corrida2`; a₃ só começa se a₂ terminar com saída 0;
   log com data e hora; pasta de saída nova.
3. Congelar a₂ e a₃ pela regra (menor `focal_alt`, supera a base) em `adapters_congelados`.
4. `conferir_codigo_do_cache.py` e extração de MR_a₂ e MR_a₃ (~3,5 h cada).
5. Comparador por adapter (`comparacao_dev_a2`, `comparacao_dev_a3`); as cabeças do M0 têm de sair idênticas às da
   a₁. Conferir as cabeças de novo.
6. Em paralelo, sem consultar o G7: consumidor dos estudos e manifesto do G6, com testes sintéticos.
7. Congelar (G6: composição, limiar do ensemble, margens, bootstrap) e avaliar (G7).

**Início da a₂/a₃ (23/09):** a trava passou (`recorte 0891cf615c3d7461`, planos e checkpoint iguais aos da a₁).
Conferência das 6 cabeças da a₁ **passou**: reprodução exata (max |Δp| = 0), ensemble idêntico ao relatório, Platt
`a` de 0,56 a 0,86 (nenhuma inverte a ordem), `b` de 3,12 a 3,45 nas seis (o ponto de 50% cai em logits bem
negativos: o Platt compensa uma diferença grande entre treino e fold 1, compatível com proporções de classe
diferentes — não medido), limiares de 0,46 a 0,56, épocas de 190 a 520. A conferência de código caiu no import do
`lumina` sem o shim do tilelang; corrigida (`fc48ab4`).

**a₂ e a₃ (24/09): `exit_a2=0`, `exit_a3=0`.** Mesma receita e orçamento; recorte `0891cf615c3d7461`, planos e
checkpoint idênticos aos da a₁ (a trava confirmou no relatório); zero falha de janela; backbone intacto.

| Semente | `focal_alt` base → melhor | delta | melhor passo | laço |
|---|---|---:|---:|---:|
| a₁ 20260921 | 1,7331 → 1,6008 | −0,1323 | 2999 (final) | ~90 min |
| a₂ 20260922 | 1,7331 → 1,6044 | −0,1287 | 2999 (final) | 77 min |
| a₃ 20260923 | 1,7331 → 1,5978 | −0,1353 | 2999 (final) | 76 min |

A melhora de reconstrução **se repetiu nas três sementes** (amplitude 0,0066): estabilidade do treino, não melhora
clínica nem regional. Diagnósticos, só descrição, iguais aos da a₁: 82–91% do ganho no termo de massa; ordem do ALT
sem mudança distinguível (ICs cruzam zero); termo de escolha cai com IC abaixo de zero — compatível com mudanças nas
probabilidades sem grandes alterações de ordem, **sem identificar um mecanismo único**; entropia +0,065 a +0,071;
perdas de contexto (+0,005) e de referência (+0,006 a +0,007) sobem um pouco; ABraOM − global no `focal_ce`:
−0,0096 [−0,054; +0,030] e −0,0028 [−0,050; +0,042] — **não detectamos diferença** entre as fontes, o que não é
equivalência. O melhor no último passo não pede mais orçamento: as 3.000 atualizações estavam fixadas para as três.

**Congelamento (revisão de 24/09):** a primeira versão de `scripts/congelar_adapter.py` confiava no que a corrida
dizia de si — aprovava `supera_a_base: true` com o melhor pior que a base, janela e dropout do LoRA fora da receita,
e outro recorte de 800 janelas coerente só consigo mesmo; e tratava dois checkpoints sem estado como idênticos.
Agora: melhora **recalculada do histórico** (finita, menor que a base; passo e valor conferidos); receita completa
(inclui janela, dropout, treino inteiro, sem retomada, versão e superfície) conferida no relatório **e** no
checkpoint, que têm de concordar; planos (sha registrado **e** recalculado dos arquivos) e R03 contra a **referência
externa** declarada (`adapter_do_mr.referencia`, sha256 completos); recorte contra o **re-sorteado** do plano
declarado; estados do `adapter.pt` com estado não vazio e chaves declaradas — iguais os estados, os arquivos ainda
têm hashes diferentes, e o script lista os campos que diferem sem atribuir a eles a diferença inteira. As três
corridas (inclusive a a₁) são reconferidas antes do registro.

**Reconferência e registro (24/09): as três passaram.** Recorte re-sorteado do plano declarado =
`0891cf615c3d74613a37f907d6ef3876f4448bb2ed2fe75c412094f3589d3be0`, igual ao detalhe das três corridas; estados do
`adapter.pt` iguais aos do `adapter_melhor.pt` nas três (198 tensores = 99 módulos × 2 matrizes), com os arquivos
diferindo em `criado_em_utc` e `identidades`; conferência de código e ambiente contra o cache do M0 passou.
Registrados em `adapters_congelados`: a₂ `8850e19c…`, a₃ `f2e547e7…`; a entrada da a₁ (`6327a9fa…`) ganhou o recorte
e a conferência, sem mudar o sha. Próximo: extrações MR_a₂ e MR_a₃, comparadores e conferência das cabeças (com o M0
contra o do comparador da a₁).

**Queda noturna (24/09) e ambiente perdido.** A cadeia lançada às 03:48 morreu às 04:49 (~61 min) com 49.152 de
171.720 variantes do MR_a₂ (12 fragmentos), sem nenhuma linha `exit_`: o shell inteiro foi morto de fora. O
container só voltou às 11:28 — compatível com desligamento por ociosidade de ~60 min do espaço do SageMaker (a
conferir na configuração). O reinício **apagou o `mamba_ssm` do `/opt/conda`** (`ambiente.mamba_ssm: 2.3.2.post1 →
None`): o que é instalado fora da home não sobrevive. A checagem de código barrou a retomada, como devia — um
MR_a₂ extraído noutro ambiente não parearia com o M0. Resposta: `scripts/conferir_reproducao_do_cache.py` re-extrai
as primeiras 64 variantes de um cache nos MESMOS lotes da extração original e compara número a número (tolerância
1e-5 do extrator), e a cadeia passou a começar por ela (M0 e todo cache MR já começado) e a receber o interpretador
em `PY` — candidato: o `.venv` do `lumina-inference`, que mora na home e tem o `mamba_ssm` fixado no commit
`0048fbf2` (2.3.2.post1).

**Segunda queda (24/09) e o que ficou fixo.** A retomada com o ambiente reinstalado pelo Gabriel (mamba da ponta do
`main`, `e9594ce1`, 22/07 — a mesma versão declarada `2.3.2.post1` do commit fixado, por isso só a conferência
numérica decide) reproduziu **exatamente a amostra conferida** (as 64 primeiras variantes do fragmento 0, nos lotes
originais) do M0 e do MR_a₂ e retomou o MR_a₂ em 49.152 — é reprodução da amostra, não re-extração do cache
inteiro; com a identidade e a integridade dos fragmentos, basta para não descartar o cache retomado — e parou de
novo. A documentação da AWS explica: no JupyterLab o espaço é ocioso quando **não há sessão ativa de kernel nem de
terminal** (mínimo de 60 min; `nohup` não conta), e o tempo é configurado pelo administrador no domínio ou no perfil.
Peças novas: `scripts/cadeia_mr_a2_a3.sh` (a cadeia versionada, retomável e autoverificada; `PY` = interpretador),
`scripts/acompanhar_cadeia.py` (rodando numa célula de notebook mantém o kernel ocupado enquanto a cadeia roda —
contorno, não garantia) e `scripts/instalar_ambiente_gpu_na_home.sh` (os mesmos passos do Gabriel com `--user` e o
mamba fixado em `e9594ce1`, para o ambiente morar na home e sobreviver a reinícios; rodar com o `/opt/conda` limpo).

**Cadeia completa (24/09, 19:04).** Conferência de ambiente e reprodução do M0 e do MR_a₂ passaram; MR_a₂ e MR_a₃
completos (171.720 de 171.720 cada, zero falha de janela; o MR_a₂ saiu em três sessões — fragmentos 0–11, 12–35 e
36–41 —, cada uma aberta pela conferência de reprodução); comparadores e conferências das cabeças passaram (max |Δp| = 0; ensemble
igual ao relatório); **as cabeças do M0 saíram idênticas às do comparador da a₁ nos dois** — as execuções
conferidas, com essas sementes, dados e ambiente, reproduziram exatamente o M0 (não é garantia para outro ambiente
ou configuração). Platt `a` 0,60–0,90 em todas (nenhuma inverte a ordem), `b` 3,23–3,45; limiares 0,49–0,65
(o menor é o do M0 h12, 0,4883); épocas 100–530. Nas doze cabeças distintas dos três comparadores (o M0 contado
uma vez): `a` 0,5598–0,9005, `b` 3,1242–3,4501, limiares 0,4574–0,6488, épocas 100–530. A tentativa das 14:17
parou antes de extrair por falta do `pyfaidx` depois do reinício (registro acima); a relançada seguinte passou em
todas as conferências.

| Desenvolvimento, conjunto de seleção, ensemble de 3 cabeças por adapter | macro Δ [IC] | AUROC geral Δ [IC] | AUPRC Δ [IC] |
|---|---|---|---|
| a₁ | −0,0041 [−0,0090; +0,0010] | −0,0025 [−0,0051; −0,0006] | −0,0022 [−0,0057; +0,0013] |
| a₂ | −0,0018 [−0,0056; +0,0022] | −0,0009 [−0,0024; +0,0007] | −0,0023 [−0,0053; +0,0008] |
| a₃ | −0,0013 [−0,0075; +0,0045] | −0,0001 [−0,0023; +0,0020] | +0,0017 [−0,0023; +0,0055] |

Painéis (Δ AUROC, a₁/a₂/a₃): missense −0,0037/−0,0031/−0,0017; splice −0,0005/−0,0003/+0,0008; noncoding
−0,0083/−0,0022/−0,0029; plof com 1 benigna, sem leitura. Pares da **composição final** (os que vão ao G7): a₁+h11
−0,0036, a₂+h12 −0,0064, a₃+h13 +0,0025 (média −0,0025, sinais mistos); o ensemble da composição final sai no G6.

Leitura (exploratória, classificação geral, sem participação brasileira): as três macro são pequenas e negativas,
com IC que inclui zero; a a₁ foi a mais negativa, e o IC da AUROC geral abaixo de zero **não se repetiu** em a₂ e a₃.
A variação entre sementes de adapter (0,0028 na macro) é da ordem do próprio delta. As três comparações **não são
independentes**: dividem as cabeças do M0 e o conjunto de seleção (que escolheu a configuração do M0) e variam só o
lado MR. O par com h12 é sempre o mais negativo; o M0 h12 ser a melhor cabeça do M0 nesse conjunto pode contribuir,
mas as cabeças MR também variam e a causa não foi demonstrada. **Leitura (revisão de 24/09): as estimativas indicam
pequenas quedas na macro, sem evidência consistente de melhora; os intervalos incluem zero e não demonstram
equivalência nem ausência de degradação.** A AUPRC positiva da a₃ não é vitória (IC com zero, as outras métricas
não acompanham), e um IC que exclui zero numa semente e inclui noutra não demonstra diferença entre adapters. A
média dos deltas dos pares da composição final (−0,0025) não é a macro do ensemble final, que depende da ordem
depois de combinar as probabilidades: ela sai no G6, descritiva, sem reabrir seleção. Nada disso responde a
pergunta regional (G7).

**Proveniência do MR_a₂ (em andamento, 24/09).** Extraído em sessões separadas por reinícios, cada uma aberta pela
conferência de reprodução contra o M0 e o `fragmento_00000` do próprio MR_a₂: fragmentos 0–11 na sessão original
(noite); 12–35 na sessão das 11:49 (ambiente reinstalado pelo Gabriel; reprodução com **diferença zero**), que caiu
às 13:49 por ociosidade com 147.456 de 171.720. A tentativa das 14:17 parou **antes de extrair**, na conferência de
reprodução, por falta do `pyfaidx` depois do reinício — a proteção funcionando. GPU: NVIDIA L4 (24 GB).

**Desenho do G6/G7: `docs/g6_g7_desenho.md`.** O consumidor aplica as regras de avaliação do Mosaic (PLAN
§13.3–13.5), e o núcleo está escrito e testado com dados sintéticos (`eval/campanha/estudos.py`). Ponto a não perder:
**no G7 o Mosaic manda relatar o coorte inteiro** (AUROC/AUPRC), com painéis como diagnóstico e sem macro — não a
macro do desenvolvimento. As margens (três, exigidas pelo Mosaic §13.5, cada uma com quantidade e regra) e a unidade
da reamostragem são do Eduardo.

**Revisão do desenho (23–24/09), aceita inteira e corrigida:**
- a campanha é **protocolo derivado** (cabeça treinada e calibrada no release): aplicar as regras de avaliação do
  Mosaic não é cumprir o protocolo publicado;
- **`janela2048` não exclui os clusters dos estudos**: tira do treino as variantes a até 2.048 bp de um membro; só os
  clusters da **seleção comum** saem inteiros. O documento prometia isolamento maior que o aplicado;
- o manifesto declara **separados** o pré-treino do R03 (o que não estiver documentado fica "desconhecido", nunca a
  data do ClinVar), o treino da cabeça e os dados do adapter; o sha256 do manifesto vai em arquivo à parte;
- o estudo de precisão é só **cenário** de ordem de grandeza, não a largura dos ICs brasileiros; relevância (Eduardo)
  e precisão (factibilidade) são perguntas separadas, e a margem não se reduz para facilitar;
- a interação **subtrai deltas**, não remove confundimento; o bootstrap conjunto **não preserva os pares**, o por par
  não preserva a dependência entre pares do mesmo cluster — os dois saem juntos;
- o consumidor agora conta os pares com os dois membros cobertos, e as **análises secundárias pré-declaradas** (fora
  do ABraOM, 44 controles com SCV brasileira, exposição empatada, pares completos) estão implementadas; as que faltam
  (Brier, baselines de AF, fold 0) estão declaradas, para escrever ou retirar **antes** do G6.

**Sexta revisão (24/09), aceita inteira.** Além das correções de leitura já aplicadas acima (a conferência de
reprodução é da **amostra conferida**, não do cache inteiro; sai "no máximo, piora pequena"; a AUPRC da a₃ não é
vitória; o M0 idêntico vale para as execuções conferidas; a causa do par com h12 não foi demonstrada; faixas de Platt
e de limiar corrigidas), retiro o que eu tinha dito na conversa: nenhuma causa para a a₁ ter sido a mais negativa
foi medida — atribuí-la "em parte à semente" era explicação sem medida. E fica registrado: **os ICs do bootstrap por
cluster são condicionais aos sistemas treinados** — reamostram variantes com adapters e cabeças fixos e não incluem a
variação de treino.

**Construtor do G6 (24/09): `scripts/construir_g6.py`, regras puras em `eval/campanha/g6.py`.** Não treina nada e
não lê o fold 0 nem os estudos. Confere a composição contra o pareamento; cada componente contra a conferência do
seu comparador (mesmo sha256); o M0 dos comparadores da a₂ e da a₃ idêntico ao da a₁ (reconferido); os quatro
caches (sistema, adapter congelado, o mesmo R03, só o adapter diferindo, mesmas linhas); recarrega as seis cabeças e
confere a reprodução (seleção contra `predicoes_selecao.parquet`; métricas da seleção e do fold 1 contra o
comparador; Platt e limiar da cabeça refeitos no fold 1 — consistência numérica; a identidade das linhas vem de
IDs e hashes, ver a sétima revisão abaixo). Depois: média das três probabilidades por sistema, **limiar do ensemble pela regra do protocolo** e o
**ensemble final no desenvolvimento** (M0 = média de h11/h12/h13; MR = média de a₁+h11, a₂+h12, a₃+h13), descritivo e
sem reabrir a composição. Grava `g6_construcao.json`, `g6_predicoes.parquet` e o **rascunho** do manifesto; só com
`--congelar` e sem bloqueio grava `g6_manifesto.json` + `.sha256`.

- **Regra do limiar = a do protocolo**, não proposta nossa: `config/suite.yaml` do Mosaic (`threshold: metric mcc,
  on validation_gold, tiebreak [specificity, higher_threshold]`) e `calibrate_threshold` (candidatos logo abaixo do
  menor score, cada score distinto e logo acima do maior), conferidos no código em 24/09; o fold 1 é a
  `validation_gold` do run 0.
- **Declaração ampliada (`g6`)**: `regra_do_limiar`, `proveniencia` (pré-treino do R03 com corpus, hash e cutoff
  "desconhecido"; treino da cabeça com o cutoff **dos rótulos**, ClinVar 2026-06; ABraOM `3cd33784…`; dados do
  adapter por prefixo), `exclusoes_aplicadas` (o que foi aplicado, com o que **não** garante),
  `sobreposicoes_declaradas` e `pendencias_antes_do_congelamento`.
- **Bloqueios de hoje (o congelamento é recusado):** margens e unidade do bootstrap (Eduardo); Brier; baselines
  (escrever ou retirar com o motivo); `scripts/avaliar_estudos.py`; ensaio do consumidor na membership real com scores
  **sintéticos**; ABraOM reconferido no arquivo; código do G7 commitado. A sanidade no fold 0 fica para depois do
  congelamento, como declarado, e não bloqueia.
- **Testes:** regras puras (`tests/test_campanha_g6.py`, 20, no Windows); ponta a ponta com G5, três comparadores e
  três conferências sobre caches sintéticos (`tests/test_construir_g6.py`, precisa de torch: roda no notebook).
  Antes de mandar, um ensaio local sem torch (cabeças lineares no lugar do `.pt`, todo o resto real) passou nos seis
  casos: rascunho, `--congelar` com bloqueio recusado, congelamento resolvido com sha conferido, e as adulterações
  (cabeça trocada, caches de adapter trocados, M0 diferente num comparador) reprovando.

**Sétima revisão (24/09), antes de rodar o rascunho: aceita inteira.**
- **Bloqueio conferia o texto do estado, não o conteúdo.** Margens e bootstrap com só `{"estado": "DECLARADO"}`
  passavam nessa parte (reproduzido pela revisão; reproduzido de novo num teste). Agora `g6.problemas_das_margens`
  exige, em cada uma das três margens do Mosaic e na condição 3, estudos, delta, métrica, estatística (estimativa ou
  `p2_5`) e limite **finito** com o sinal certo, e `criterio_proprio` explícito na interação; `problemas_do_bootstrap`
  exige unidade principal e de sensibilidade entre as implementadas, réplicas ≥ 1000, seed e percentis;
  `problemas_das_pendencias` exige `onde` no `FEITO` e `motivo` no `RETIRADO`. A declaração ganhou o modelo com os
  campos nulos, que é o que o Eduardo preenche.
- **Git que falha parecia "sem mudanças".** `estado_do_codigo` chama o git com `check=True`, confere também os
  arquivos fora do git (`ls-files`) e devolve `erro` quando falha; `problemas_do_codigo` bloqueia com `erro`, com
  revisão que não seja 40 hex, com código ausente, fora do git ou modificado.
- **Linhas por IDs e hashes, não por calibração coincidente.** O construtor passa a conferir que os IDs do fold 1
  são exatamente o papel `validation` do snapshot da política (1.575, declarado) e que os da seleção são exatamente
  os de `selecao_comum.parquet` (novo `--selecao`, sha256 com o prefixo `693eb234`); a cadeia cabeça →
  `cache_identidade_sha256` → hash de conteúdo da tabela já era conferida. Reproduzir Platt e limiar fica descrito
  como consistência numérica.
- **Baseline contínua: não retirar.** Minha justificativa ("não está na membership") não se sustentava: o release
  traz `pb_annotations.parquet` e o Mosaic tem o comparador **oficial** `gnomad_rarity` (`config/comparators.yaml`,
  `mosaic-comparators/v1`: `−gnomad_v4_af`, `not_found`/`ac0` com AF 0), calculado dessas colunas sem ler VCF. Papel
  explícito da presença no ABraOM: no clínico, diagnóstico da diferença de composição; no populacional ela **define**
  os grupos (`build_brazil_membership`: casos gold presentes, controles gold ausentes), é constante em cada grupo e
  não se relata como discriminação. O pareamento casa `gnomad_af_bin`, então dentro do par o `gnomad_rarity` só
  difere dentro da faixa. Próximo passo: `scripts/conferir_cobertura_das_baselines.py` (hash lógico do
  `pb_annotations` e do `membership` recalculado com o `logical_contract` do Mosaic contra a referência declarada
  — os hashes lógicos da ADR 0006 (`specs/PLAN-release-migration.md` §7.1: `pb_annotations` `7a6ae9b2…`,
  `membership` `1c1cd65d…`, o do G1), que valem nos dois layouts — e contra o manifesto da cópia (a do notebook
  ainda tem `bundle.manifest.json`, do layout anterior à ADR 0006);
  especificação do `gnomad_rarity`; cobertura por estudo e papel), **só contagens, nenhuma métrica**.
- Ensaio local sem torch refeito com os casos novos (seleção com um ID trocado, fold 1 sem um ID, git que falha com
  todo o resto resolvido): todos recusados pelo motivo certo. Testes puros: 25 do G6 + 4 da declaração + 7 da
  cobertura (1 só roda no notebook, com pyyaml).

**Rascunho do G6 nos artefatos reais (24/09, 20:33; revisão `f1fb3dd`).** Testes no notebook: 25 + 4 + 7 + 24 + 7
(cobertura, inclusive o script inteiro com pyyaml) e o ponta a ponta com torch, todos passando. Os testes no notebook
rodam **como script** (`python3 tests/test_x.py`): `-m unittest tests.x` falha lá porque outro pacote `tests` do
`/opt/conda` sombreia o namespace. Construtor: `exit_g6=0`, todas as conferências passaram — seis componentes com
max |Δp| = 0 e os mesmos sha e Platt de antes; M0 idêntico nos comparadores da a₂ e da a₃ (3/3); fold 1 = 1.575
(P 1.038, B 537) igual ao papel `validation` do snapshot; seleção = 2.799 em 156 clusters igual à `selecao_comum`;
ABraOM reconferido no arquivo (o bloqueio sumiu); código sem mudança fora do commit.

| Limiar do ensemble (fold 1, regra do Mosaic) | limiar | MCC | especificidade | sensibilidade |
|---|---:|---:|---:|---:|
| M0 = média de h11, h12, h13 | 0,474562 | 0,8878 | 0,9125 | 0,9692 |
| MR = média de a₁+h11, a₂+h12, a₃+h13 | 0,552163 | 0,8880 | 0,9181 | 0,9663 |

Os limiares estão **calculados e registrados no rascunho**; só congelam com o manifesto definitivo. O MCC no fold 1
é a quantidade usada no ajuste (o mesmo fold em que as cabeças pararam e foram calibradas), não uma estimativa
independente de desempenho.

| Ensemble FINAL, conjunto de seleção (exploratório; IC por cluster, 1.000 réplicas) | M0 | MR | Δ [IC] |
|---|---:|---:|---|
| macro | 0,9178 | 0,9172 | −0,0006 [−0,0048; +0,0033] |
| AUROC geral | 0,9683 | 0,9679 | −0,0004 [−0,0019; +0,0010] |
| AUPRC | 0,9228 | 0,9210 | −0,0018 [−0,0051; +0,0018] |

Painéis (AUROC M0 → MR): missense 0,8451 → 0,8421 (−0,0030); splice 0,9883 → 0,9880 (−0,0003); noncoding
0,9201 → 0,9216 (**+0,0015**); plof com 1 benigna, sem leitura. No ensemble final **não** houve queda em todos os
painéis — a leitura dos arranjos por adapter não se transporta para ele. Com o limiar do fold 1, na seleção: M0 MCC 0,7319 (sens. 0,9460, esp. 0,8621); MR MCC 0,7403
(0,9310; 0,8766) — a queda em relação ao fold 1 acompanha a outra composição do conjunto (muitas benignas em
noncoding e synonymous) e o fold 1 ter servido à calibração; é descritivo.

**Leitura:** estimativas próximas de zero, levemente negativas nas três métricas, com ICs que incluem zero — não
demonstram equivalência nem ausência de piora. O delta do ensemble final é menor em módulo que o dos ensembles por
adapter (−0,0041/−0,0018/−0,0013) e que a média dos pares (−0,0025); não atribuo causa (as comparações dividem o M0 e
o conjunto de seleção, e os ICs são condicionais aos sistemas treinados). A composição e o limiar não mudam por causa
disto. Continua sem resposta a pergunta regional (G7).

**Cobertura das baselines (24/09): PASSOU.** Hashes lógicos recalculados com o `logical_contract` do Mosaic:
`pb_annotations` `7a6ae9b2…` (n 326.826) e `membership` `1c1cd65d…` (n 8.875), iguais à referência declarada e ao
`bundle.manifest.json` da cópia; Mosaic `814e7f0`, `comparators.yaml` `cd7654bf…`, especificação do `gnomad_rarity`
a declarada. `gnomad_rarity` definido em **100%** dos membros; `present_abraom` sem discordância. Só contagens:

| Grupo | n | gnomAD present / not_found / ac0 | ABraOM presente |
|---|---:|---|---:|
| clínico, case | 3.116 | 2.239 / 772 / 105 | 323 (10,4%) |
| clínico, control | 3.116 | 2.239 / 772 / 105 | 71 (2,3%) |
| clínico, unmatched_case | 3 | 3 / 0 / 0 | 1 |
| populacional, case | 751 | 747 / 3 / 1 | 751 |
| populacional, control | 751 | 747 / 3 / 1 | 0 |
| populacional, unmatched_case | 1.138 | 1.138 / 0 / 0 | 1.138 |

Três fatos de construção, agora medidos: (1) no clínico, casos e controles têm **a mesma** distribuição de status do
gnomAD (o pareamento casa `gnomad_af_bin`, que distingue ausência e `ac0`), e 877 de 3.116 (28,1%) em cada grupo
recebem AF 0 pela regra oficial — empates grandes no topo do `gnomad_rarity`, que o AUROC trata como meio ponto.
**Cobertura do score não é frequência observada**: 772 são `not_found` (sem registro no gnomAD: o 0 é imputação) e
105 são `ac0` (sítio chamado com AC 0, status próprio no Mosaic, nem raro nem ausente pela ADR 0003); (2) a
presença no ABraOM é 323 × 71 (≈ 4,5×, o achado de 20/09), diferença de composição; (3) no populacional a presença é
100% nos casos e 0% nos controles, e a raridade no ABraOM fica constante (0) nos controles **pela regra de imputação
declarada** — a frequência deles está ausente, não medida como zero. A pendência das baselines passa a `A_ESCREVER` (conferência registrada na
declaração).

**Oitava revisão (24/09) e o código do G7.** A revisão aprovou seguir e pediu três precisões, aplicadas acima:
limiares "calculados e registrados" (não "congelados") até o manifesto definitivo; cobertura do score ≠ frequência
observada; zero da raridade no ABraOM nos controles = regra de imputação. E um cuidado: **Brier nunca sobre −AF**
(score de ordenação, não probabilidade).

Escrito, sem tocar nos 12 arquivos da identidade do cache:

| Peça | O que faz |
|---|---|
| `eval/campanha/estudos.py` | `com_brier`: Brier (média de (p − y)²) nos três coortes, só dos **sistemas**, recusando score fora de [0, 1]; células por painel e sem cada painel nos **três** coortes (as margens podem pedi-las); `avaliar_baseline`/`avaliar_score_unico`: baseline com métricas absolutas na própria cobertura, **score constante no coorte sai sem métrica** ("não discrimina por construção"), diferença casos pareados − controles descritiva, **sem Brier** |
| `eval/campanha/baselines.py` | `gnomad_rarity` pelo `comparator_score` do Mosaic; raridade no ABraOM (−`abraom_af`, ausente → 0, regra nossa); ausência no ABraOM só no clínico (no populacional, "não aplicável por construção"); `frequencia_observada` separa medido de imputado |
| `eval/campanha/g7.py` | tabela dos estudos (uma linha por variante, papel `estudo`, recusa atributo divergente); identidade do cache dos estudos = a de desenvolvimento em tudo menos a tabela; scores **sintéticos** do ensaio; `avaliar_margens`: cada regra declarada aplicada mecanicamente, estudo por estudo, com a unidade principal do bootstrap na interação; estudo sem regra sai "nenhuma regra declarada", nunca "atende" por vacuidade |
| `scripts/extrair_estudos.py` | exige o G6 congelado e a mesma declaração; parâmetros de janela, lote e fragmento lidos da identidade de desenvolvimento do mesmo sistema; **antes de extrair**, recusa identidade nova que difira da de desenvolvimento além da tabela; `--so-conferir` sem GPU |
| `scripts/avaliar_estudos.py` | modo real: manifesto congelado, declaração e código do manifesto (sha256 e git), release e entradas conferidos, caches dos estudos conferidos, seis cabeças aplicadas, consumidor com Brier, baselines, margens; modo `--ensaio-sintetico`: o mesmo caminho na membership real com scores sintéticos, sem manifesto nem modelo |
| `scripts/construir_g6.py` | `--entrada regra_ampla=… --entrada exposicao=…`: sha256 no manifesto, **bloqueio** se faltar; registra a pasta dos comparadores |

**Membros no chr8 entram no G7** (regra declarada, `g7.LEITURA_DO_CHR8`): o Mosaic avalia o membership inteiro, a
reserva do chr8 é do treino e das janelas, e o G7 é avaliação única. Fica como pendência **a confirmar com o
Eduardo** junto com a decisão E.

**Testes:** puros no Windows (26 do G6, 18 do G7, 24 do consumidor, 7 da cobertura, 3 do ensaio); com torch, no
notebook, o ponta a ponta do G6 e um novo do **G7 real** (G6 congelado → extração só-conferir → avaliação com as
cabeças congeladas, prob. do sistema = média das três, e três adulterações recusadas). Antes de mandar, o mesmo G7
real rodou aqui com cabeças lineares no lugar do `.pt`: congelou, conferiu, pontuou, a média bateu com as três
cabeças e as adulterações (ambiente trocado, caches de adapter trocados, entrada mudada) reprovaram pelo motivo
certo. As pendências escritas ficam `ESCRITO` até os testes do notebook passarem.

**Nona revisão (24/09), antes do ensaio: aceita inteira; três correções de integridade do G7 real.**
1. **Bootstrap do G7 real = o do manifesto.** O avaliador recebia `--replicas` e `--seed` também no modo real — daria
   para congelar uma configuração e rodar outra (por exemplo, levar as 50 réplicas do ensaio). Agora o manifesto
   registra réplicas, seed e unidade principal da interação (da declaração), o G7 real usa esses valores e **recusa**
   `--replicas`/`--seed` divergentes; as opções livres ficam só no ensaio. A interação passou a trazer os dois ICs em
   `por_unidade` (`cluster_conjunto`, `par`) e, no nível de cima, o da unidade **declarada** como principal — nenhuma
   das duas é chamada de sensibilidade por padrão; sem unidade declarada, o nível de cima só tem a estimativa. A margem
   da interação lê o IC da unidade declarada em `por_unidade`.
2. **Coordenadas amarradas às tabelas oficiais.** A extração lia `--membros` (a saída do G1) sem conferir a identidade:
   um arquivo com os mesmos ids e outra sequência passaria. Agora a tabela é **reconstruída** do `membership` e do
   `pb_examples` do release, os dois com o hash lógico do Mosaic igual à referência congelada (o do `pb_examples`,
   `3c556259…`, n 326.826, entrou na declaração, dos invariantes da ADR 0006); `--membros`, se dado, é conferido campo a
   campo (`g7.diferencas_de_tabela`), **antes** de montar o sistema. E o G7 real confere que o conteúdo da tabela de
   cada cache dos estudos é o da tabela oficial.
3. **Score constante marcado, não apagado.** Com as duas classes, a AUROC de um score constante é 0,5 e a AUPRC é a
   prevalência positiva; retornar `None` descartava essas réplicas do bootstrap e distorcia o IC (sobretudo na ausência
   no ABraOM, binária). Agora a métrica sai definida e o coorte fica marcado `constante`. A decisão explícita de não
   relatar a presença no ABraOM no populacional continua (é outra coisa).

Nenhuma das três pede treino, extração ou calibração de novo. Ensaio local do G7 real refeito: bootstrap do relatório
= 1.000 réplicas, seed 20260901, unidade `cluster_conjunto`, origem "manifesto congelado"; `--replicas 10` recusado;
G1 com um alelo trocado recusado na extração; cache dos estudos coerente consigo mesmo mas com um alelo trocado
recusado no G7 ("não é a tabela oficial"); tabela oficial com os membros do chr8. O chr8 no G7 fica para registrar
na decisão E com o Eduardo (a revisão concordou que é coerente com a reserva do treino).

**Ensaio na membership real (25/09, 01:06; revisão `87e1fe8`): PASSOU.** Log
`~/artifacts/redesenho/g7_ensaio_20260925_010644.log`. Testes no notebook: 26 + 4 + 19 + 25 + 7 + 7 (cobertura, com o
script inteiro e o `comparator_score` real) + 3 (ensaio) e os dois ponta a ponta com torch (G6 e **G7 real**), todos
sem falha nem pulo. As três correções da nona revisão foram exercidas lá: `--replicas 10` recusado ("diverge do
manifesto"), bootstrap do relatório = 1.000/20260901/`cluster_conjunto` "manifesto congelado"; G1 com alelo trocado
recusado na extração e cache com alelo trocado recusado no G7; raridade no ABraOM constante nos controles com AUROC
0,5000 marcada `constante`.

| Leitura dos artefatos reais (scores SINTÉTICOS; nenhuma métrica é resultado) | ensaio | esperado |
|---|---|---|
| hashes lógicos de `membership`, `pb_annotations`, `pb_examples` | conferem | referência declarada (ADR 0006) |
| tabela oficial | 8.875 variantes, 171 no chr8 | 8.875 membros; nenhuma variante nos dois estudos (clínico = consensus, populacional = gold) |
| clínico: casos / pareados / sem par | 3.119 / 3.116 / 3 | idem (G1) |
| clínico: P / B no coorte completo | 2.808 / 311 | `suite.yaml`: 2.808 / 311 |
| populacional: casos / pareados / sem par | 1.889 / 751 / 1.138 | idem (G1) |
| populacional: P / B no coorte completo; nos pareados | 89 / 1.800; 88 / 663 | `suite.yaml`: 89 / 1.800 |
| pares fora do ABraOM (clínico) | 2.744 mantidos, 372 retirados | 323 casos e 71 controles presentes (22 pares com os dois) |
| sem controles com SCV brasileira | 3.072 mantidos, **44** retirados | 44 declarados |
| exposição empatada (raio 4.096) | clínico 1.849 / 1.267; populacional 682 / 69 | — |

**Exposição de locus, medida agora no snapshot final (`g6_exposicao_janela2048_r4096`, dado real, sem score):**
treino 99.992 variantes em 1.576 clusters; no clínico, 77,6% dos casos e 76,0% dos controles sem nenhuma variante de
treino a até 4.096 bp; média 3,11 × 3,31; nos 3.116 pares, 59,3% empatados e, entre os 1.267 diferentes, caso maior em
48,2%, com diferença média −0,2: **essas estatísticas estão próximas do equilíbrio** — o que não elimina
diferenças por classe, por painel nem no efeito sobre M0 e MR. Por cluster (`n_treino`), caso maior em 49,2% dos 2.218
pares diferentes. O registro anterior (0,4832; média +1,55) era **só das benignas** do clínico: outro subconjunto, não
contradiz. Descritivo: exposição igual não implica efeito igual nos dois sistemas.

**Limitações de suporte (contagens; não determinam a largura dos intervalos):** o clínico tem só **311 benignas** (e
2.805 patogênicas nos pareados); o populacional, só **88 patogênicas** nos pareados (e 89 no coorte completo, que é 95%
benigno por causa dos 1.138 casos sem par). No populacional só **39,8% dos casos têm controle**: o coorte completo
descreve todos os casos publicados, a interação descreve o subconjunto pareado, e uma conclusão sobre a interação não
se transfere aos 1.138 casos sem par. Informação para as margens, não motivo para ajustá-las.

Pendências de código e o ensaio passam a `FEITO` com a evidência acima. Bloqueios que restam para o G6 definitivo:
margens e unidade do bootstrap (Eduardo), chr8 na decisão E (171 membros) e, na hora de congelar, `--proveniencia
abraom=…` e `--entrada regra_ampla=… --entrada exposicao=…` (já existe `g6_exposicao_janela2048_r4096`).

**Décima revisão (25/09): aceita; o que dá para resolver sem devolver tudo ao Eduardo.**
- **Precisões de leitura, aplicadas acima:** exposição — "essas estatísticas estão próximas do equilíbrio", não
  "simétrico" (não elimina diferenças por classe, painel ou efeito sobre M0 e MR); as contagens mostram **limitações de
  suporte**, não "a precisão que o G7 vai ter" (não determinam a largura dos ICs); no populacional só 39,8% dos casos
  têm controle, então a interação descreve os 751 pareados e não se transfere aos 1.138 sem par (o consumidor agora
  escreve isso no bloco de pareamento de cada estudo); e o MR sintético do ensaio ter piorado é desta realização, não
  uma garantia matemática do ruído (nenhum teste depende disso).
- **Recomendações operacionais, registradas como `RECOMENDADO` (continuam bloqueando até a confirmação):** chr8 —
  incluir os 171 membros, preservando a membership oficial (§6.6 reserva o chr8 das janelas e do treino da cabeça, o que
  é compatível com avaliá-lo; a reserva é do desenvolvimento e **não** prova que o pré-treino do R03 nunca viu o chr8);
  bootstrap — `cluster_conjunto` principal e `par` como sensibilidade, 1.000 réplicas, seed 20260901 (o cluster é a
  unidade de bloqueio do Mosaic nos outros deltas; nenhum dos dois preserva todas as dependências, por isso os dois
  saem; a escolha **não** se baseia em largura ou sinal de intervalo algum).
- **Margens: decisão científica, não dedutível do código.** O esquema passou a representar o que a revisão distinguiu:
  cada regra é uma lista de condições (`estimativa` ou `p2_5`, `>=` ou `>` estrito), então "estimativa ≥ 0,02 com IC
  excluindo zero" e "limite inferior ≥ 0,02" se escrevem e **dão resultados diferentes** (teste com as duas numa mesma
  célula; e `p2_5 = 0` não "exclui zero"). E cada estudo ganhou um papel, **exigido** ou **descritivo**; o G7 sai com
  `sucesso` pelos exigidos. O 0,02 continua sendo proposta, não requisito.
- Ensaio local do G7 real refeito com o esquema novo: regras com as condições avaliadas, por painel e sem cada painel,
  e a linha `SUCESSO`. O ensaio do avaliador só usa a unidade do bootstrap quando ela está **declarada**; a
  recomendada aparece como tal.

**Folha de decisão para o Eduardo (depois da décima revisão).** Página privada, que o Gabriel compartilha pelo menu
Share: <https://claude.ai/artifact/CYd1cukg8ULPYv1mDPNGL4>. Traz as duas confirmações (chr8 e bootstrap) e a tabela
de sucesso por estudo, e monta a resposta já no formato de `g6.margens`. Não mostra resultado de desenvolvimento nem
score dos estudos, só contagens de rótulo. Dois achados ao prepará-la:
- **Origem do 0,02.** A proposta C1 da fase 0 (`docs/decisoes_eduardo_fase0.md`, nunca confirmada) punha o 0,02 no
  ΔAUROC BR-específico M2 × M1 da escada antiga, uma diferença-em-diferenças (§7.1 do `HANDOFF_CONTEXTO_VALIDADOR.md`).
  O análogo aqui é a **interação**, não a melhoria no coorte BR da condição 1. A C1 pedia também a mesma direção em ≥ 2
  de 3 sementes, que o G7 (ensemble) não avalia: seria código novo, como um guardrail de Brier. Os guardrails do PDF
  (AUROC e AP não regridem mais de 0,02, Brier não piora mais de 0,01, chr8 só descritivo com n_P ou n_B < 20) entram
  na página como referências **não aprovadas**.
- **Limite do esquema.** Cada regra de `g6.margens` vale com os mesmos parâmetros para uma lista de estudos. Se os dois
  estudos forem exigidos com critérios diferentes nas condições 1 a 3, `g6.py` e `g7.py` precisam de uma extensão
  pequena antes do G6. A interação e os painéis já podem valer só para parte dos estudos. A extensão não é feita antes
  da resposta; a página avisa quando a escolha a exigir.
- Pendente na página: contagens por painel e chr8 por estudo, de um bloco no notebook que só lê rótulos (o
  `ensaio_relatorio.json` já tem a composição por painel de cada coorte).

**Suporte por painel nos estudos (25/09, 17:02; revisão `7aaed65`): só rótulos.** Log
`~/artifacts/redesenho/contagens_por_painel_20260925_170144.log`. Antes das contagens, todos os testes passaram no
notebook: 27 + 4 + 20 + 25 + 7 + 7 (cobertura) + 3 (avaliador) e os dois ponta a ponta com torch (G6 e G7 real). Os
números de métrica nessa saída vêm das fixtures sintéticas dos testes, não de dado real.

| P / B | clínico, completo | clínico, pareados = controles | populacional, completo | populacional, pareados = controles |
|---|---|---|---|---|
| missense | 1.244 / 192 | 1.242 / 192 | 61 / 417 | 60 / 309 |
| splice | 408 / 33 | 408 / 33 | 10 / 38 | 10 / 28 |
| noncoding | 95 / 60 | 95 / 60 | 4 / 1.078 | 4 / 153 |
| plof (guarda) | 1.050 / 2 | 1.049 / 2 | 14 / 1 | 14 / 1 |
| synonymous (guarda) | 10 / 24 | 10 / 24 | 0 / 266 | 0 / 172 |
| other | 1 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| clusters | 1.133 | 1.133 e 1.200 (união 1.653) | 84 | 78 e 113 (união 122) |

- **Condição 3:** no clínico ela se aplica com suporte mínimo de até 60, e com os três painéis até 33 (as 33 benignas
  de splice). No populacional, só com mínimo de até 10 (missense e splice). Com o 20 do PDF, o clínico tem os três
  painéis e o populacional só o missense, onde a condição não se aplica.
- **O AUROC do coorte mistura painéis:** no clínico, 37% das patogênicas são plof (1.050, contra 2 benignas no
  painel); no populacional completo, 84% das patogênicas são missense ou plof e 75% das benignas são noncoding ou
  synonymous. Parte da discriminação do coorte vem da composição por painel, não só da ordem dentro de cada painel.
- **Populacional:** 1.889 variantes em 84 clusters; a união do bootstrap da interação tem 122 clusters, e os 751
  casos pareados estão em 78. Entre os painéis de discriminação, só o missense tem mais de 10 patogênicas.
- **chr8:** os 171 membros estão todos no clínico, 78 casos e 93 controles. O populacional não tem membro no chr8.
- A folha do Eduardo foi atualizada com essas contagens (versão 3) e não tem mais pendência de dado.
