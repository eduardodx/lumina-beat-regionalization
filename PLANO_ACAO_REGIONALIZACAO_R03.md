# PLANO DE AÇÃO — Regionalização (campanha M0–M4 do Eduardo) sobre o modelo R03

> **Para quem pega este documento em outro chat:** ele é **auto-contido**. Cole inteiro para
> contextualizar. Cobre (1) o que já foi feito na regionalização v10/v11 e o que descobrimos,
> (2) a proposta NOVA do gestor (Eduardo), (3) o modelo novo **R03** (`lumina-inference`),
> (4) o mapa **reusa / novo / risco / dependência de dados / estimativa** — *verificado no código,
> não suposto* — e (5) o plano em fases. Datado **2026-07-27**. Autor da frente: **Gabriel** (dev).
> Gestor: **Eduardo**. Baseline v10: **Pedro**.
>
> **Estado:** planejamento. **Nada foi implementado desta campanha ainda.** O trabalho v10/v11
> anterior está completo e serve de insumo/priors (ver §1).

---

## 0. TL;DR — o que é esta campanha em 6 frases

1. O Eduardo entregou um protocolo experimental (28 páginas) que **reformula a regionalização** de
   forma muito mais rigorosa que o trabalho v10/v11: pergunta se um adapter de variação
   **brasileira (ABraOM)** ou **admixed-American (gnomAD AMR)** melhora variantes de submissores
   brasileiros **mais que um adapter global**, com arquitetura e orçamento idênticos.
2. É uma **escada de modelos M0→M4**, cujo **baseline correto é o M1 (adapter global)**, não o M0.
   A métrica principal é a **diferença-em-diferenças** `ΔBR-específico = ganho_BR − ganho_nonBR`.
3. **É fundamentalmente diferente do que fizemos** (adapter de *representação* por janelas
   sintéticas, não de *predição de frequência*; baseline global; braço AMR; controle nonBR pareado
   1:1; holdout de cromossomo 8; Brier/calibração; genes BRCA/TP53).
4. Roda sobre um **modelo novo, o R03** (`LUM-20260719-001-R03`, repo `lumina-inference`), da org
   **croma-bioai** — não o `beat-v11 r1` que usamos. Mesma arquitetura (Mamba-3, 52M, d_full=448),
   API de load diferente, bucket S3 diferente.
5. **A proposta é metodologicamente excelente** e acerta as correções que nós descobrimos na marra
   (baseline, DiD, AUROC-separa-ranking). **O maior risco** — que os nossos dados preveem — é
   **falta de poder estatístico no endpoint clínico** (T_BR é pequeno e o pareamento 1:1 encolhe
   mais); por isso adicionamos uma **Fase 0.5 de análise de poder como gate**.
6. Boa parte da **infra de treino/fusion/eval é reusável**; o **grosso do trabalho novo** é o
   gerador de janelas sintéticas populacionais + os dados global/AMR + o port do adapter para o R03.

---

## 1. O que JÁ foi feito (v10/v11) e o que descobrimos — os priors desta campanha

O trabalho anterior (portar o pipeline do Pedro para o Beat-v11 r1) está **completo**. Ele não é a
campanha nova, mas produziu **achados medidos que de-riscam e preveem** o que a campanha do Eduardo
vai encontrar. Não repetir esses testes; usá-los como priors.

**Achados-chave (todos medidos, com controle e poder):**

- **A sequência não carrega sinal de frequência brasileiro-específico.** Adapters de predição de
  AF (A_BR ≈ A_gnomAD ≫ scrambled), fusão crua (Δ real−scrambled = +0.003, CI [−0.024, +0.030],
  n=4163), resíduo (tênue) — **todos nulos, bem-dimensionados**.
- **O ganho da regionalização vinha sendo medido contra a baseline ERRADA.** Reportava-se sobre o
  M0 (que ignora frequência). Contra a régua certa — a própria frequência — o ganho encolhe de
  ~+0.34 para **+0.04 a +0.09**. **Este é exatamente o problema que o desenho do Eduardo conserta
  na raiz** (baseline = M1). *(scripts: `rebaseline_regional_against_frequency.py`,
  `analyze_regional_gain_decomposition.py`, `audit_abraom_presence_confound.py`.)*
- **O `br_only` é, em boa parte, um bit de "está no catálogo ABraOM" — que é um limiar de
  frequência** (piso do índice ≈ AF 0.5%). A regra trivial de um bit dá MCC 0.619 (test) / 0.609
  (holdout), *acima* do M5_v2 publicado (0.574).
- **gnomAD ≥ ABraOM em termos absolutos** (na slice grande, n=71591: AUROC 0.818 vs 0.781). O sinal
  ABraOM-específico existe mas é **borderline**: DiD **+0.027, p unilateral ~0.04, CI encostando em
  zero** — limitado por só ~67 patogênicas brasileiras. *(script: `test_abraom_vs_gnomad_power.py`
  — já implementa o DiD tie-aware + bootstrap BC que a Fase 4 do Eduardo precisa.)*
- **O teto do `br_only` é de DADO/RÓTULO, não de modelo.** Cinco alavancas fechadas com medição:
  re-tunar desconto (saturado), guarda por cabeça nativa (refutada), desacoplar molecular
  (refutada), **treinar no brasileiro k-fold (nulo: AUROC 0.875→0.865)**, cobertura de frequência
  (bloqueada por dados). Founders e benignas comuns **se parecem para todo sinal molecular**.
- **O v11 NÃO é pior que o v10 no brasileiro.** A "regressão" 0.574 vs 0.605 é ruído de n=504: o CI
  do v11 (gene-clustered) cobre o 0.605 nos três splits; AUROC 0.89–0.94.
  *(script: `recompute_v11_br_only_powered.py`.)*
- **A guarda M5_v3 é um beco sem saída:** na tabela do Pedro, v10 M5_v2 e M5_v3 dão o **mesmo**
  `br_only` (0.605). A guarda só moveu o global.

> **Como isso alimenta a campanha nova:** as previsões honestas são (a) o endpoint clínico
> `ΔMCC_BR-específico(M2 vs M1)` tem alto risco de dar **borderline/null**; (b) **M3 (AMR) pode ≥ M2
> (ABraOM)** porque gnomAD já ganha do ABraOM; (c) o sinal com mais chance de aparecer está na
> **avaliação representacional do chr8** (mais poder) e no braço AMR. Levar isso ao planejar.

*Documentos-fonte:* `TCC_REGIONALIZACAO_V11.md`, `RESULTADOS_REGIONALIZACAO_V11_FASES2-4.md`,
`HANDOFF_CONTINUACAO_V11_POS_FASE4.md`. Baseline do Pedro:
`../lumina-ssm/artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md`.

---

## 2. O que o documento do Eduardo propõe (a campanha M0–M4)

**Pergunta principal:** um adapter treinado com variação **ABraOM** ou **gnomAD AMR** melhora a
classificação de variantes de **submissores brasileiros** (T_BR) *mais* do que um adapter de
variação **humana global**, com a mesma arquitetura e o mesmo orçamento de treino?

**A escada (a única variável deliberada é a FONTE populacional do adapter):**

| Modelo | Composição | Isola |
|---|---|---|
| **M0** | backbone + adapter ClinVar + head (sem branch populacional) | baseline clínico histórico |
| **M1** | + adapter **GLOBAL** (gnomAD balanceado entre AFR/AMR/EAS/NFE/SAS, sampler hierárquico) | ganho de *ter a 2ª branch* + variação humana genérica ← **BASELINE CAUSAL** |
| **M2** | + adapter **ABraOM** | ganho brasileiro |
| **M3** | + adapter **gnomAD AMR** | ganho admixed-American |
| **M4** | + adapter **ABraOM + AMR** (50/50, mesmo orçamento) | complementaridade |

> ⚠️ **Choque de nomes:** o **M2 do Eduardo = ABraOM**. O "M2" do Pedro/nosso era **gnomad-only**.
> Nesta campanha, usar SEMPRE a convenção do Eduardo (M0–M4 acima).

**Arquitetura de cada modelo:** `Mj = Hj(Fj(C(B(x)), P_fonte(B(x))))`, onde B=backbone congelado,
C=adapter ClinVar (treinado UMA vez, congelado), P=adapter populacional, F=fusion layer, H=head.
**Treina-se só F e H** (backbone, C e P congelados) — atribuição limpa.

**O adapter populacional (o coração da novidade):** treinado em **janelas de sequência sintéticas**
geradas aplicando variantes da fonte populacional a janelas de referência —
`generate_population_window(reference_window, population_variant_table, frequency_sampling_policy,
seed)` — com manifesto de provenance por janela (para auditar leakage). **NÃO é** um adapter que
prediz AF (o que fizemos); é um adapter que aprende a **distribuição** da população.

**Conjuntos de teste:**
- **T_BR** = `BR-only` (variante com submissor brasileiro e nenhum não-brasileiro). "ClinVar-BR-submitters".
- **T_nonBR** = **pareado 1:1** com T_BR (exato em gene/label/tipo/consequência; aproximado em
  AF global/estrelas/ano/nº submitters). **NÃO parear por presença/AF no ABraOM/AMR** (é o mecanismo).
- **chr8 held-out** de TUDO (adapters, ClinVar, fusion, validação, calibração) → benchmark separado.
- **BRCA1/BRCA2/TP53** (chr17/13/17, independentes do holdout chr8).

**Métrica principal:** `ΔS_BR-específico(Mj) = [S(Mj,T_BR) − S(M1,T_BR)] − [S(Mj,T_nonBR) − S(M1,T_nonBR)]`
para S ∈ {MCC, AP, AUROC, Brier}. **Contraste confirmatório: M2 vs M1 em ΔMCC.** AUROC separa
"melhorou o ranking" de "só mexeu no threshold/calibração"; Brier exige Platt scaling.

**Rigor:** contrato pré-registrado com hashes (checkpoint, ClinVar/ABraOM/gnomAD/hg38, splits);
threshold escolhido no calibration set e **congelado** (mesmo threshold em BR e nonBR); 3 seeds no
confirmatório; **bootstrap pareado por matched-set** (10.000 reps); **cenários A–I** pré-definidos;
critério de sucesso: `ΔMCC_BR-específico(M2) > 0`, CI 95% exclui 0, ≥2/3 seeds mesma direção,
ganho absoluto ≥ 0.05; guardrails (AP/AUROC não regridem >0.02, Brier >0.01, chr8 ≥ M1).

**chr8 representacional (o que o Eduardo chama de indispensável):** Spearman entre o **score
populacional do modelo** e `log10(AF)` em variantes chr8 nunca vistas (ABraOM-chr8, AMR-chr8,
GLOBAL-chr8). Testa se o adapter aprendeu estrutura de frequência, **separado** da tarefa clínica —
e tem **mais poder** que o T_BR_chr8 clínico.

---

## 3. O modelo novo — R03 (`lumina-inference`)

**Fato central:** a campanha roda sobre o **R03**, não o `beat-v11 r1`. O próprio documento do
Eduardo já cita o R03 (o comentário sobre AF menor no chr8). Verificado em `lumina-inference/README.md`
e `TECHNICAL.md`.

| Item | R03 (novo) | beat-v11 r1 (o que usamos) |
|---|---|---|
| Checkpoint | `LUM-20260719-001-R03`, step 71000, 52.124.400 params | `lumina-beat-v11v5-r1-...`, 52.1M |
| S3 | `s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt` | `s3://ai4bio-lumina/releases/...` (org diferente!) |
| Pacote / loader | `from lumina import load_model_from_checkpoint, batch_encode_dna` | `lumina_beat_v11.load_model_from_checkpoint` |
| Arquitetura | Mamba-3 hourglass, d_model=384, d_pure=64, **d_full=448**, downsample ×4 | idêntica |
| `encode()` | existe (`model.py:327`), retorna dict com `last_hidden_state [B,L,448]` + `mid_hidden_state [B,L/4,384]` | idêntico em espírito |
| Superfície LoRA | `in_proj` (mamba), `q/k/v_proj` (attn), `gate`, `stem`, `purity` — **mesma família** | idêntica |
| Interface populacional (p/ chr8) | `gnomad_af_pred` / `gnomad_observed_logits [B,L,4]` — **"weak population prior"** | idêntico (Fase 1 mediu Spearman ~0.13) |
| Scoring zero-shot | **masked-center LLR** documentado (`mlm_logits`, `log p(alt)−log p(ref)`) | não usávamos |
| Kernel Mamba-3 | GPU: `mamba_ssm` via `setup-gpu.sh`; CPU/MPS: drop-in `mamba3` puro | tinha shim tilelang |
| Load | `strict=True`, 630 tensores, 0 missing/unexpected | precisava do package loader |

**Notas de risco do R03 (do próprio TECHNICAL.md §6):** selection chr8 **0.4715** vs main 0.5453 —
o R03 vai **pior no chr8**, com evidência de overfitting aos cromossomos de treino. Isso é
diretamente relevante porque o Eduardo usa o chr8 como holdout representacional: **a interface
populacional do R03 pode ser fraca/enviesada exatamente no chr8**. Precisa ser medido cedo (é um
gate barato — ver Fase 0.5).

**Regra de ouro operacional (do README):** no host GPU, **nunca** rodar `uv sync`/`uv run` (o
`.venv` é torch-free de propósito; torch vem do conda). Usar `bash scripts/setup-gpu.sh` +
`source scripts/env.sh`. Auditar se o backbone R03 viu chr8 no pré-treino (define se o nome é
"adapter-level" ou "foundation-and-adapter chromosome holdout" — o Eduardo exige essa distinção).

---

## 4. Mapa REUSA / NOVO / RISCO / DEPENDÊNCIA — verificado no código

> Verificado lendo o código, não suposto. "Reusa" = existe e serve; "adapta" = existe mas precisa
> port; "novo" = não existe. Estimativa em T-shirt (S/M/L/XL) — tempo real depende do fluxo
> notebook (Gabriel roda os jobs).

| Componente | Estado | Onde / evidência | Risco / dependência |
|---|---|---|---|
| Backbone R03 + loader | **adapta** | `lumina-inference/lumina/` (`encode()` ok) | precisa do checkpoint no S3 croma-bioai |
| Port do adapter (FineTune…R03) | **novo (S)** | espelhar `eval/clinvar/beat_v11_adapter.py` p/ o loader `lumina` | módulos batem; troca de API |
| Superfície LoRA | **reusa+verifica (S)** | `eval/clinvar/lora.py`; nomes `in_proj/q_proj/gate/stem/purity` batem | confirmar nomes exatos no state_dict R03 |
| Adapter ClinVar (C) treinado 1× | **reusa (S)** | `scripts/clinvar_m0_job.py` (é o nosso M0) | receita já validada |
| Camada de fusion (F) | **reusa (M)** | `eval/clinvar/fusion_lora.py` (Static/Dynamic, aceita N adapters populacionais) | hoje espera adapter de frequência; F é agnóstica |
| **Gerador `generate_population_window`** | **novo (L)** | não existe; o `train_abraom_frequency_adapter.py` extrai janela REF/ALT p/ prever AF — mecanismo DIFERENTE | é o coração da novidade |
| Adapter populacional (P) por representação | **novo (L)** | idem acima | depende do gerador |
| Dados **GLOBAL** (gnomAD por população, sampler hierárquico) | **novo + DADO (L)** | nenhum AFR/AMR/EAS/NFE/SAS no repo | **depende de baixar gnomAD estratificado** |
| Dados **AMR** (gnomAD AMR) | **novo + DADO (M)** | idem | **depende de baixar gnomAD AMR** |
| T_BR (BR-only) | **reusa (S)** | `scripts/build_regional_clinvar_eval_slices.py` (br_only/nonbr_only/shared) | ok |
| Tabela revisada de instituições BR | **adapta (M)** | hoje vem da coluna `cohort` do `lumina-benchmarks`; Eduardo exige **revisão manual ≥1×** | trabalho manual + fonte |
| **T_nonBR pareado 1:1** | **novo (M)** | não existe; `matched_regional_clinvar_benchmark` é outra coisa | risco: cobertura de match baixa → encolhe T_BR |
| **Holdout chr8** (de todas as fontes) | **novo (M)** | `grep chr8` no pipeline = vazio | precisa entrar em Fase 0 |
| Eval principal (MCC/AP/AUROC + slices) | **reusa (S)** | `eval/clinvar/metrics.py`, `evaluate_clinvar_finetuned_model.py` | AP/Brier a acrescentar |
| **DiD (ΔBR-específico) + bootstrap pareado** | **reusa (S)** | `scripts/test_abraom_vs_gnomad_power.py` (DiD tie-aware + BC) já implementa | precisa adaptar p/ matched-set bootstrap |
| AUROC tie-aware, bootstrap gene-clustered | **reusa (S)** | `test_abraom_vs_gnomad_power.py`, `recompute_v11_br_only_powered.py` | ok |
| **Calibração Platt + Brier** | **novo (S)** | temos M5_v2/v3 (desconto), não Platt | Platt é simples |
| **Eval representacional chr8 (Spearman score×AF)** | **novo (M)** | interface `gnomad_af_pred` existe mas é fraca | risco: sinal fraco no R03/chr8 |
| **Eval BRCA1/BRCA2/TP53** | **novo (S)** | não existe como slice dedicada | dados já no ClinVar |
| Contrato pré-registrado (hashes/manifests) | **novo (S)** | disciplina, não código | ok |
| Painel sentinel P/LP BR (curadoria) | **reusa parcial** | `scripts/build_brazilian_plp_sentinel_panel.py` existe, TSV curado NÃO (0 linhas) | opcional; gated por curadoria |

**Dependências de dados a resolver ANTES de treinar (bloqueiam a Fase 2):**
1. **Checkpoint R03** acessível do notebook (bucket croma-bioai — confirmar credenciais/permissão).
2. **gnomAD estratificado por população** (AFR/AMR/EAS/NFE/SAS) para o adapter GLOBAL.
3. **gnomAD AMR** para M3/M4.
4. **hg38 fasta** (já temos: `~/hg38/hg38.fa`).
5. **Tabela de instituições brasileiras** revisada.

---

## 5. O plano em FASES

> Ordem = a do Eduardo (§13 do doc dele), com uma **Fase 0.5 nova** (nosso gate de poder) e cada
> fase anotada com reusa/novo/risco/dep/estimativa. Cada fase tem um **gate** — não avançar sem ele.

### Fase 0 — Auditoria, dados e congelamento
**Objetivo:** construir e congelar os conjuntos, sem leakage, com manifests.
- Confirmar acesso ao checkpoint R03 (S3 croma-bioai). *(dep: credenciais)*
- Baixar/versionar gnomAD (estratificado + AMR), ABraOM, ClinVar, hg38 — com hashes. *(dep: dados — L)*
- Construir tabela de instituições brasileiras e **revisar manualmente** ≥1×. *(adapta — M)*
- Normalizar variantes → chave `GRCh38:chrom:pos:ref:alt` (decompor multialélico, left-normalize,
  trim, validar REF vs hg38). *(reusa parcial de `prepare_regional_clinvar_dataset.py`)*
- Particionar **BR-only / nonBR-only / shared**. *(reusa — S)*
- **T_nonBR pareado 1:1** + relatório de cobertura de match. *(novo — M; risco: cobertura baixa)*
- **Excluir chr8** de todas as fontes de treino; separar benchmarks chr8. *(novo — M)*
- Auditar se o **backbone R03 viu chr8** no pré-treino → define o nome do holdout. *(dep: training repo/W&B)*
- Gerar manifests + hashes; verificar **overlap zero** entre splits e com adapters populacionais.
- **Gate:** sem duplicatas, sem aliases não resolvidos, sem test-variant no treino, sem SCV da mesma
  variante em splits diferentes, sem chr8 em fonte de treino.

### Fase 0.5 — Análise de poder (NOSSO gate; barato; antes de qualquer job de treino)
**Objetivo:** decidir se o endpoint clínico é viável, e calibrar critérios de sucesso à realidade.
- Contar `n_BR_matched` por classe após o pareamento 1:1 (e no chr8, e em BRCA/TP53).
- Estimar o **poder** para detectar `ΔMCC_BR-específico ≥ 0.05` dado o n real e a incerteza que
  medimos (CIs de MCC ~±0.05–0.09 nesse n; DiD de frequência ~+0.027 borderline).
- Rodar o **eval representacional chr8 do R03** já (só backbone, sem treino): Spearman
  `gnomad_af_pred × log10(AF)` em ABraOM-chr8/AMR-chr8 — mede se a interface populacional do R03
  presta no chr8 (o TECHNICAL.md §6 alerta que vai mal lá). *(reusa a extração backbone-only)*
- **Gate/decisão:** se o T_BR pareado for pequeno demais para poder, (a) considerar o conjunto
  **BR-associated** (mais amplo, reportado à parte), (b) priorizar o eixo **representacional chr8 +
  AMR** sobre o MCC clínico, (c) congelar critérios de sucesso realistas. *Levar ao Eduardo.*
- **Estimativa: S** (local, sem job). **Alto valor — evita gastar a campanha num endpoint sem poder.**

### Fase 1 — M0 (baseline clínico limpo no R03)
- Port do adapter R03 (`FineTune…R03`, espelhando o do v11). *(novo — S)*
- Treinar o adapter ClinVar (C) 1×, congelar; treinar head M0; **calibrar (Platt)**; congelar threshold; avaliar.
- **Gate:** reproduzir baseline sem leakage. (O MCC ~0.20 histórico é preliminar até isso.)
- **Estimativa: M** (1 port + 1 job de treino + calibração).

### Fase 2 — Adapters populacionais (Global, ABraOM, AMR, Mix)
- Construir o **gerador `generate_population_window`** parametrizado por fonte + manifests de provenance. *(novo — L)*
- Amostrador hierárquico p/ o GLOBAL (grupo→bin AF→tipo→variante). *(novo — M)*
- Treinar **P_global, P_ABraOM, P_AMR, P_mix** com **orçamento idêntico** (mesmos tokens/steps/seeds/janelas). *(4 jobs)*
- **Gate:** comparar manifests dos 4 treinos e confirmar que a ÚNICA diferença é a fonte.
- **Estimativa: XL** (é o grosso do trabalho; depende dos dados da Fase 0). Risco: distribuições
  pareadas entre fontes (SNV/indel, bins de AF, variantes por janela).

### Fase 3 — M1–M4 (fusion + head)
- Para cada P: congelar backbone+C+P, inicializar F/H, treinar no mesmo ClinVar train, selecionar no
  mesmo validation, calibrar no mesmo calibration, congelar threshold, gerar predições. *(reusa F — 4 jobs)*
- **Gate:** receita idêntica entre M1–M4 (só a fonte de P muda).
- **Estimativa: L** (4 jobs de fusion + calibração; infra reusável).

### Fase 4 — Avaliação principal (o teste da hipótese)
- Métricas em T_BR e T_nonBR (MCC/AP/AUROC/Brier); ganhos vs **M1**; **ΔBR-específico**; **bootstrap
  pareado por matched-set (10k)**; 3 seeds no confirmatório. *(reusa DiD/AUROC/bootstrap — S/M)*
- Classificar no **cenário A–I** pré-definido; aplicar critério de sucesso + guardrails.
- **Estimativa: M** (ferramentas prontas; falta o matched-set bootstrap + AP/Brier).

### Fase 5 — chr8 (representacional + clínico)
- Representacional: Spearman score×AF em ABraOM-chr8/AMR-chr8/GLOBAL-chr8 (M1 vs M2 vs M3 vs M4). *(novo — M)*
- Clínico chr8: só se `n_pos ≥ 20 e n_neg ≥ 20`; senão **descritivo**.
- **Estimativa: M.** Provável portador do sinal mais forte (mais poder que o T_BR clínico).

### Fase 6 — BRCA1/BRCA2/TP53
- Pooled `T_BR_cancer` + por gene (só com `n_pos,n_neg ≥ 10`); threshold global congelado (não tunar por gene).
- **Estimativa: S.**

---

## 6. Riscos transversais (priorizados) e mitigação

1. **Poder no endpoint clínico (ALTO).** T_BR pequeno + pareamento 1:1 encolhe. Nossos dados preveem
   borderline. → **Fase 0.5 gate**; priorizar chr8-representacional/AMR; critérios realistas.
2. **Dependência de dados (ALTO).** gnomAD estratificado + AMR não estão no projeto; checkpoint R03
   noutra org S3. → Resolver na Fase 0 **antes** de qualquer treino.
3. **Interface populacional fraca no R03/chr8 (MÉDIO).** `gnomad_af_pred` é "weak prior" e o R03 vai
   pior no chr8 (TECHNICAL.md §6). → medir na Fase 0.5; se fraco, o eval representacional perde força.
4. **Cobertura de pareamento 1:1 (MÉDIO).** Se baixa, T_nonBR representa só uma fração de T_BR. →
   relatório de cobertura obrigatório; comparar distribuição incluídas × excluídas.
5. **O resultado provável é um null honesto (MÉDIO, mas OK).** Se M1≈M2 (cenário A) ou M3≥M2
   (cenário D), é científico e publicável — o desenho aceita isso. Alinhar expectativa com o Eduardo.
6. **Leakage sutil (MÉDIO).** Aliases/SCVs da mesma variante, chr8 no backbone. → gates da Fase 0.

---

## 7. Decisões abertas para o Eduardo (antes de começar)

- **Endpoint primário:** manter MCC clínico como confirmatório, ou promover chr8-representacional/AMR
  dado o risco de poder? (recomendação: definir após a Fase 0.5).
- **Fonte de dados gnomAD** (release, genome/exome, como estratificar por população) — precisa ser fixada.
- **Definição operacional de "global"** (pesos por grupo; AMR entra com 20%?).
- **Curadoria** de P/LP brasileiros (o painel sentinel está com infra pronta e 0 linhas) — vale
  investir para dar poder ao chr8/BRCA, ou fica para fase futura?
- **Orçamento de compute** (nº de jobs: ~1 M0 + 4 adapters + 4 fusion, × 3 seeds no confirmatório).

---

## 8. Referência rápida (caminhos e arquivos)

**Modelo novo (R03):**
- Repo: `../lumina-inference/` (README + TECHNICAL.md). Loader: `from lumina import load_model_from_checkpoint`.
- Checkpoint: `s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt`.
- GPU: `bash scripts/setup-gpu.sh` + `source scripts/env.sh`. **Nunca `uv sync` no host GPU.**

**Infra reusável (repo `lumina-beat-regionalization`):**
- Adapter/LoRA/fusion: `eval/clinvar/{beat_v11_adapter.py, adapters.py, lora.py, fusion_lora.py}`.
- M0/fusion launchers: `scripts/{clinvar_m0_job.py, clinvar_fusion_job.py, sagemaker_clinvar_*.py}`.
- Slices: `scripts/build_regional_clinvar_eval_slices.py`, `scripts/prepare_regional_clinvar_dataset.py`.
- Métricas/eval: `eval/clinvar/metrics.py`, `scripts/evaluate_clinvar_finetuned_model.py` (via `run.py`).
- **Nossas ferramentas de análise (prontas p/ Fase 4):** `scripts/test_abraom_vs_gnomad_power.py`
  (DiD tie-aware + BC bootstrap), `scripts/recompute_v11_br_only_powered.py` (gene-clustered),
  `scripts/rebaseline_regional_against_frequency.py`, `scripts/analyze_regional_gain_decomposition.py`,
  `scripts/audit_abraom_presence_confound.py`.
- Painel sentinel: `scripts/build_brazilian_plp_sentinel_panel.py`.

**Baseline v10 (Pedro):** `../lumina-ssm/artifacts/clinvar_regional_*` +
`.../researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md`.

**Documento do gestor:** `~/Downloads/regionalizacao.pdf` (28 páginas; texto extraído em
`~/Downloads/regionalizacao.extracted.txt`).

---

## 9. Gotchas herdados (do trabalho v11) que continuam valendo

1. **Nunca confie em default/doc — leia o artefato/state_dict.** (d_full=448 não 320; MoE off.)
2. **Poder estatístico não é detalhe** — n=504 enganou nos dois sentidos. Sempre CI + n.
3. **Distinga significância estatística de prática** (magnitude + consistência entre seeds).
4. **`--lora-rank` da fusion = rank do M0/C**, senão as chaves do adapter são filtradas em silêncio.
5. **Eval herda `batch_size` do checkpoint (=2)** se não passar `--batch-size 8` → estoura o tempo.
6. **`list-training-jobs` precisa de `--no-paginate`.**
7. **Split `all` só é out-of-sample nas slices brasileiras** (br_only/br_any); nas outras vaza.
8. **Calibração roda LOCAL** (barata, sobre predictions.parquet) — iterar nela não custa job.
9. **Fluxo de trabalho:** Windows edita/prepara + commita/pusha; o notebook SageMaker dá `git pull` e
   roda (GPU, torch, S3). Nunca rodar AWS/torch do Windows. Entregar runbooks copiáveis.

---

*Fim do plano. Estado em 2026-07-27: planejamento — nada desta campanha implementado. Insumo pronto
(achados v10/v11 como priors), modelo novo mapeado (R03), infra existente inventariada e verificada,
fases e gates definidos. Próximo passo real: Fase 0 (dados + congelamento) e Fase 0.5 (análise de
poder) — as duas gated e baratas, decidem a viabilidade antes de qualquer job de treino.*
