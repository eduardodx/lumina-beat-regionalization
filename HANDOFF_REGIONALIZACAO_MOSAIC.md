# HANDOFF — Regionalização do R03 com o estudo brasileiro do Mosaic

> **Para quem pega num chat novo: este doc é auto-contido.** Leia inteiro antes de tocar em código.
> Datado **2026-09-16**. Autor: Gabriel (dev, TCC). Gestor: Eduardo (mantém o Mosaic).
> Branch: **`new_regionalization`**.
> Nada está treinando. **O Eduardo respondeu em 15/09 e fechou as decisões A, B, C e D** (§4); G0, G1 e G2 estão
> desbloqueados.

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

1. **Pedir ao Eduardo o arquivo do ABraOM**, a decisão E e a confirmação dos parâmetros da §5.1 do plano.
2. Desbloqueados agora:
   - **G1 — membership:** importar `studies/brazil/membership.parquet`, juntar com `pb_examples.parquet` para obter
     `chrom/pos_1based/ref/alt`, validar IDs únicos por estudo e papel, ligações bidirecionais, rótulos e estratos, e
     contar P/B por coorte e painel.
   - **G2 — snapshot da cabeça:** `core_locus` `run_id=0` (treino folds 2-4 gold+consensus, validação fold 1 gold,
     teste fold 0 gold), menos os membros dos dois estudos e seus `overlap_cluster_id`, menos a regra ampla
     brasileira, com `sequence_eligible`; medir o custo de cada exclusão e hashear o resultado.
   - **G0 — identidades:** hashes de R03, release e gnomAD (o do ABraOM fica pendente do arquivo).
   - **Contagem de controles com SCV brasileira** pela regra ampla (usa o `submission_summary` já baixado), para
     pré-declarar a análise de sensibilidade.
3. Depois: extrator portado e smoke do M0 (G3), gerador + MLM e piloto do adapter misto (G4), escolha da extração na
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
