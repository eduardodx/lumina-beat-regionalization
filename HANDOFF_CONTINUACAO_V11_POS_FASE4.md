# HANDOFF — Regionalização Beat-v11, estado PÓS-FASE 4 (continuação em novo chat)

> **Para quem pega este documento:** ele é **auto-contido**. Cole inteiro no novo chat para
> contextualizar. Cobre a missão, os repositórios, o fluxo de trabalho, a infra, a arquitetura,
> **tudo o que já foi feito e com quais resultados**, o código que construímos, os artefatos e
> caminhos, os gotchas que custaram tempo, e o que fazer a seguir. Datado **2026-07-23**.
>
> **Substitui** o `HANDOFF_CONTINUACAO_V11.md` (que era do estado "Fase B", muito atrás).
>
> Autor da frente: **Gabriel** (dev). Gestor: **Eduardo**. Baseline v10: **Pedro**.

---

## 0. TL;DR — onde estamos em 5 frases

1. Portamos o **estudo de regionalização ABRAOM do Pedro** (validado no Beat-v10) para o **Beat-v11**. **O porte está COMPLETO** — todas as fases (0, 1, B, 2, 3, 5, 4) fechadas e validadas.
2. **O resultado principal reproduz o headline do Pedro e o supera num ponto crucial:** a calibração leva o `br_only` MCC de **0.24 → 0.57**, e a **especificidade ABRAOM está FALSIFICADA de forma positiva (p=0.0196)** — coisa que o v10 do Pedro **não conseguiu**.
3. **A tese consolidada:** o valor do ABRAOM **não é um sinal aprendível pela sequência** (adapters/fusion todos nulos, com poder estatístico), **mas a frequência observada usada como calibração carrega sinal regional real e falsificável**.
4. **Depois disso tentamos MELHORAR o `br_only`** com duas alavancas. **Ambas deram negativo bem-dimensionado:** (B) re-tunar o desconto está **saturado**; (A) guarda por cabeças nativas do v11 foi **refutada**.
5. **Estado atual: a frente está tecnicamente encerrada.** O caminho de ganho real agora é **curadoria externa de P/LP brasileiros** (gated por dados) — a mesma conclusão a que o Pedro chegou no v10, agora com evidência medida.

---

## 1. A missão (o "porquê")

**Regionalização** = melhorar a interpretação de patogenicidade de variantes (ClinVar) no **contexto brasileiro/latino-americano**, usando o **ABRAOM** (base genômica brasileira, coorte SABE de São Paulo), que fornece **frequência alélica (AF) por população brasileira**.

**O problema:** em benchmarks de foundation models de DNA, variantes brasileiras são um **ponto cego estrutural**. A hipótese: o gargalo é o **atalho de frequência** — a AF de referência é majoritariamente europeia (gnomAD NFE), o que penaliza variantes comuns em populações não-europeias (falso-positivo). Calibrar a AF por população (ABRAOM) deveria corrigir.

**O perigo (o que torna difícil):** algumas variantes **patogênicas** também são comuns no Brasil (founder/recessivas). Descontar patogenicidade só por frequência **apaga essas P/LP** (colapso de recall). Todo o desafio é equilibrar isso.

**O TCC do Gabriel** tem duas frentes: (a) finetuning ClinVar / benchmark Mosaic (já feito, outra frente) e (b) **esta regionalização**.

---

## 2. Os repositórios

Todos de `github.com/eduardodx/`.

| Repo | Papel |
|---|---|
| **`lumina-beat-regionalization`** | **NOSSO repo de trabalho.** Branch `main`. Contém o pipeline do Pedro (copiado do `lumina-ssm@abraom-regionalization-study`), o pacote vendorizado `lumina_beat_v11/`, e todas as nossas adições. **É aqui que se trabalha.** |
| **`lumina-ssm`** (branch `abraom-regionalization-study`) | A **baseline do Pedro no v10**. Use para `diff`, para ler os artefatos dele (`artifacts/clinvar_regional_*`, `artifacts/abraom_frequency_adapter/*`) e o relatório `ABRAOM_RESEARCHER_TRANSFER_REPORT.md`. **Os números do v10 saem daqui.** |
| **`lumina-beat`** (branch `main`) | Os foundation models Beat. `beat-v11/lumina_beat_v11/` (o pacote que vendorizamos), `beat-v11/TECHNICAL.md` (a referência de arquitetura — excelente), `beat-v11/config/beat_v11_base.json`. |
| `lumina-benchmarks-mosaic-eval` | **Outra frente** (benchmark do gestor). Não mexer. |

---

## 3. ⚠️ O FLUXO DE TRABALHO (crucial — leia antes de qualquer coisa)

- A máquina **Windows local NÃO tem AWS, S3, GPU, torch nem pandas**. Serve só para **editar código e ler os repos**.
- **Todo trabalho pesado roda no SageMaker notebook** (`~/testeArq/lumina-beat-regionalization`), que tem GPU, o env conda "modelo" (torch/mamba_ssm), credenciais AWS e os artefatos.
- **O combinado com o Gabriel:** *o assistente edita/prepara código e comandos no Windows; o Gabriel faz `git push` → `git pull` no notebook e roda lá; o assistente lê os artefatos que o Gabriel traz de volta.*
- **Nunca** tente rodar AWS/SageMaker/torch a partir do Windows. Entregue **runbooks copiáveis**.
- Scripts novos precisam ser **commitados e pushados** para o `main` (o notebook dá `git pull`).

---

## 4. Infra (AWS / SageMaker / S3)

- **Conta AWS:** `085188779747`, região **us-east-2**.
- **Checkpoint alvo (r1):** `s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt`
- **Quota:** só **1 `ml.g5.2xlarge` de treino concorrente**. A `ml.g5.4xlarge` tem pool próprio → use para rodar um 2º job em paralelo.
- **Nome do job:** o launcher imprime `job_name=<base>`, mas o SageMaker **anexa timestamp**. Ache com `list-training-jobs --name-contains <base> --no-paginate`.

### Prefixos S3 principais
```
checkpoint r1     s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/
slices regionais  s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/slices/
hg38              s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/hg38/
artefatos M0      s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/<exp>/sagemaker-artifacts/<JOB>/output/
artefatos fusion  s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/<exp>/sagemaker-artifacts/<JOB>/output/
artefatos eval    s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/<exp>/sagemaker-artifacts/<JOB>/output/
adapters freq     s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/<exp>/sagemaker-artifacts/<JOB>/output/
```

### Baixar um artefato (padrão à prova de erro)
```bash
JOB=<nome completo do job com timestamp>
A=$(aws sagemaker describe-training-job --region us-east-2 --training-job-name "$JOB" \
      --query ModelArtifacts.S3ModelArtifacts --output text)
mkdir -p ~/v11eval/<nome> && aws s3 cp "$A" ~/v11eval/<nome>/m.tar.gz --region us-east-2 \
  && tar -xzf ~/v11eval/<nome>/m.tar.gz -C ~/v11eval/<nome>
```

---

## 5. A arquitetura do Beat-v11 (fatos VERIFICADOS ao vivo)

> **Lição transversal: nesta base, defaults e documentação MENTEM. Sempre leia o artefato / o modelo carregado.**

- **r1:** SISO, **52.1M params**, `config['model']='beat_v11_bioprime'`.
- **`d_full = 448`** (`d_model=384` + `d_pure=64`). **NÃO é 320** (os defaults do dataclass dizem 256/320; o checkpoint sobrescreve).
- **MoE OFF** neste checkpoint (`moe_enabled=False`).
- **`encode()`** retorna `{"last_hidden_state" [B,L,448], "mid_hidden_state" [B,L/4,384]}` — é o readout com **trunk RMSNorm** que **todas as cabeças nativas** consomem. (O v10 retornava `hidden_states`.)
- **Cabeças nativas** (`_token_head_outputs`, model.py:925-960): `mlm`, **`conservation_scalar_pred` [B,L,3]** (phyloP100 / Zoonomia-241 / phyloP470), `conservation_bin`, `conservation_delta`, `splice_class`, `splice_distance`, `region`, `counterfactual_snv`, **`missense_severity_pred` [B,L,4]** (ESM-2 destilado, por base alternativa), `gnomad_af_pred`, `gnomad_observed_logits`. **Confirmado habilitado no r1** (`missense_severity_head_enabled: true`, `num_conservation_targets: 3`).
- **Probe linear frozen no ClinVar: AUROC 0.953** (TECHNICAL.md §11) — a representação já é fortíssima.
- **Loader:** `lumina_beat_v11.load_model_from_checkpoint(path, device=..., strict=True)` (aceita `s3://`). **NÃO use o registry** `build_registered_model` (rejeita as 74 chaves de config).
- **Superfície LoRA @ rank 8:** 105 lineares / 1.248M params (mamba `in/out_proj` + attn `q/k/v/out_proj` + `up_stages.gate` + `stem.purity`; **zero cabeças**).
- **Gotcha tilelang:** o conda do notebook tem `tilelang/tvm` quebrado que aborta o import do `mamba_ssm`. Fix já commitado: shim no topo de `src/models/__init__.py` + em `beat_v11_adapter.py` (`sys.modules["tilelang"]=None` → fallback triton). Nos jobs é no-op (`INSTALL_TILELANG=0`).

**Decisão tomada e validada: NÃO alterar o Beat-v11.** O modelo fica congelado; `encode()` já dá a representação certa; as cabeças nativas são derivadas do mesmo hidden (redundantes com o que LoRA+cabeça extrai). Manter paridade com o Pedro é o correto para isolar o backbone no head-to-head.

---

## 6. A arquitetura da regionalização (a escada de modelos)

Todos treinam sobre o backbone **congelado**, em ClinVar **não-brasileiro** (`nonbr_only`, *leave-Brazilian-out*).

| Modelo | O que é |
|---|---|
| **M0** | Baseline **molecular** de patogenicidade. O piso contra o qual todo ganho regional é medido. |
| **A_BR / A_gnomAD / A_scrambled / A_residual** | **Adapters de frequência** (LoRA que preveem AF a partir da sequência). `A_scrambled` = controle negativo. |
| **M4 (static) / M5 (dynamic)** | **Fusion**: M0 + adapters via *gate* (peso por adapter). Static = peso fixo aprendido; dynamic = gate MLP por-exemplo. |
| **M7s / M7d** | **Controles**: fusion com o adapter **scrambled** no lugar do A_BR. |
| **M5-bounded** | **A "frequência explícita"**: fusion + `--head-type regime_a_bounded_regional` + **11 features de AF explícitas**. A cabeça decompõe em `molecular_logit` e `regional_discount` (com teto na arquitetura). **É o insumo da calibração.** |
| **M5_v2** | Calibração pós-hoc: re-tuna `discount_scale`, `max_discount`, thresholds no **holdout**. **É o nosso LEAD.** |
| **M5_v3** | M5_v2 + **guarda molecular** (protege evidência molecular forte do desconto). **Barrado no v11** (ver §7.6). |

### ⚠️ As duas camadas de calibração (fácil confundir)
1. **Desconto aprendido** — dentro do modelo (`RegimeABoundedRegionalHead`, no treino), a partir da AF explícita.
2. **Calibração de segurança** (M5_v2/v3) — **pós-hoc, sobre os scores** (`calibrate_m5_v*.py`), re-aperta o desconto + adiciona guarda. Roda **local**, sem job.

### A "frequência explícita" é um MODELO, não pós-processamento
A diferença M4 → M5-bounded é **`explicit_feature_columns`**: as 11 features
`log10_af_abraom, log10_af_gnomad, af_delta, af_abs_delta, af_ratio_log10, af_abraom_missing, af_gnomad_missing, specificity, specificity_missing, abraom_present, is_snv`.
**8 delas são engenheiradas em runtime** (`eval/clinvar/dataset.py:99-125`) a partir de 5 colunas-base do parquet — **não estão armazenadas** (isso já assustou uma vez; é alarme falso).

### As 4 slices decisivas (a métrica da frente)
| Slice | Mede | Métrica |
|---|---|---|
| `br_only` | desempenho no subconjunto brasileiro | **MCC** |
| `abraom_common_benign` | falso-positivo em benignas comuns | **specificity** |
| `abraom_pathogenic_present` | **não suprimir** P/LP founder | **recall** |
| `global_nonbr_no_abraom` | não degradar o resto | **MCC** |

**Split de comparação = `test`** (paridade com a tabela §6.1 do Pedro). Verificado batendo os números dele.

---

## 7. O que JÁ FOI FEITO (todas as fases + resultados)

### 7.1 Fase 0 — Integração v11 ✅
`FineTuneBeatV11Adapter` (`eval/clinvar/beat_v11_adapter.py`): carrega o r1 via package loader, extrai `last_hidden_state`, shim tilelang. Família `beat-v11` no `build_finetune_adapter` (`adapters.py:1175`). Validado com smoke tests.

### 7.2 Fase 1 — Adapters de frequência ✅
Spearman (val/test), receita do Pedro (ctx 1024, 1000 steps, balanced-af, sequence-only):

| Adapter | v10 | **v11** |
|---|---|---|
| A_BR | 0.114 / 0.104 | **0.135 / 0.127** |
| A_gnomAD | 0.107 / 0.118 | **0.120 / 0.118** |
| A_scrambled (piso) | −0.013 / 0.001 | **−0.036 / −0.021** |

Bootstrap pareado: `A_BR − A_scrambled` = **+0.15** (CI ≫ 0) → há sinal seq→AF real. `A_BR − A_gnomAD` = **+0.010** (CI cruza 0) → **NÃO é brasileiro-específico**.

### 7.3 Experimento B — resíduo ✅
Alvo `delta_logit = logit(af_abraom) − logit(af_gnomad)`, perda **huber**. Gap `A_residual − A_residual_scrambled` = **+0.026** test [+0.001, +0.050]; val +0.028 (CI cruza por um triz). Trajetória: o gap **cresce para +0.032** no step 1000 (o `best_step` por NLL — degenerada para alvo ilimitado — subestimou). **Veredito: sinal regional fraco mas real.** Decisão: **não reestruturar o fusion**.

### 7.4 Fase 2 — M0 no v11 ✅
Job `clinvar-m0-nonbr-v11-r1-952aa6-20260713011715`. Receita do Pedro (regime A, ctx 1024, focal γ=2.0, LoRA 8/16/0.05, lr 5e-6/5e-4, batch 2×8, 3 épocas, `--native-feature-heads none`).

| Métrica (nonBR test) | v10 | **v11** |
|---|---|---|
| MCC | 0.576 | **0.654** |
| AUROC | 0.879 | **0.927** |

### 7.5 Fase 3 + 5 — Fusion, controles e a falsificação ✅
Jobs: `clinvar-fuse-m4-static-v11-r1-e2a6e0-...`, `m5-dynamic-...8974a8`, `m7-static-scr-...cd2194`, `m7-dynamic-scr-...2afaf7`.

**Global (nonBR test) MCC:** M0 0.654 · M4 0.677 · M5 0.665 · M7s 0.683 · M7d 0.688 → **fusion ≈ M0 e scrambled ≥ real**.
**Pesos aprendidos do gate (static):** abraom **0.504** / gnomad 0.496; scrambled **0.502** / gnomad 0.498 → **o gate não distingue o real do embaralhado**. (Explicação estrutural: a fusão treina em `nonbr_only`, o gate **nunca vê variante brasileira**.)

**Slices (test):**

| Slice | v10 M0 | v11 M0 | v10 M4s | v11 M4s | v11 M5d | v11 M7s | v11 M7d |
|---|---|---|---|---|---|---|---|
| `br_only` MCC | 0.279 | 0.238 | 0.292 | 0.320 | 0.292 | 0.266 | 0.278 |
| `acb` spec | 0.803 | 0.842 | 0.894 | 0.879 | 0.854 | 0.864 | 0.862 |
| `app` recall | 0.417 | 0.460 | 0.288 | 0.350 | 0.436 | 0.387 | 0.393 |
| `global` MCC | 0.512 | **0.606** | 0.526 | **0.631** | 0.614 | 0.641 | 0.647 |

**O teste bem-dimensionado (o fechamento da Fase 3).** O `br_only` no `test` tem só n=504 → sem poder. Mas **as slices brasileiras são disjuntas do treino por construção** (`br_only = has_BR & ~has_nonBR`; treino = `nonbr_only = has_nonBR & ~has_BR`, ver `build_regional_clinvar_eval_slices.py:70/78`), então o split **`all` é 100% out-of-sample** → n=4163 (8×).

| Modelo (`br_only.all`, n=4163) | MCC |
|---|---|
| **v10 M0** | 0.2476 |
| **v11 M0** | **0.3350** |
| v11 M4 / M7s(scr) | 0.3532 / 0.3504 |
| v11 M5 / M7d(scr) | 0.3562 / 0.3445 |

**Falsificação:** `M4 − M7s` = **+0.0028** [−0.024, +0.030]; `M5 − M7d` = +0.0117 [−0.010, +0.033]. Replicado no `br_any` (n=4872). → **o fusion cru NÃO carrega sinal ABRAOM-específico (< ±0.03 MCC)** — negativo *bem-dimensionado*, mais forte que o do Pedro (+0.018, CI cruzando por falta de poder).

**⚠️ Duas correções que o teste com poder produziu** (o n=504 enganou nos DOIS sentidos):
1. O "+0.054 de vantagem do abraom real" no `test` era **ruído** (virou +0.003 com n=4163).
2. "O backbone v11 não ajuda o brasileiro" **estava errado**: com o mesmo conjunto de 4163 variantes, **v10 0.2476 → v11 0.3350 (+0.087)** — quase igual ao ganho global (+0.094). **Mas o gap brasileiro↔global persiste** (−0.264 → −0.271): *um FM melhor levanta a régua inteira, mas não fecha o ponto cego brasileiro*.

### 7.6 Fase 4 — M5-bounded + calibração ✅
**Passo 1 — M5-bounded** (job `clinvar-fuse-m5-bounded-v11-r1-0402dd-20260716220945`): fusion dynamic + `--head-type regime_a_bounded_regional` + as 11 features. nonBR test MCC **0.749** — **curado e aprovado** (não é threshold vazado: LEAKY 0.758 ≈ honesto 0.749; não é overfit: val 0.762; `specificity` vem **mergeada do ABRAOM**, não do rótulo → sem vazamento; o ganho é do `af_gnomad`, critério ACMG legítimo).

**Passo 2 — eval-alvo** (`reval-m5bounded-v11-r1-3a8bb0-...`, 8 slices × test+holdout): o **padrão bounded cru** reproduziu o do Pedro — `br_only` 0.621, `acb` spec 0.994, **`app` recall COLAPSADO 0.080** (Pedro: 0.037), global 0.635 (v11 degrada bem menos que o 0.400 dele).

**Passo 3 — calibração:**
- **M5_v2 (LEAD):** config `discount_scale 1.0, max_discount 1.5, regional_threshold 0.235, global_threshold 0.765`. **Recuperou o recall de 0.08 → 0.405.**
- **M5_v3:** `hold_current_lead` — a guarda resgatou +0.116 de recall (19 variantes) **mas criou 208 FPs** em benignas (spec 0.951→0.934, abaixo do piso 0.95). **Decisão de segurança correta do pipeline.**

**A TABELA DEFINITIVA (test):**

| Slice | v10 M0 | v11 M0 | **v10 M5_v3** (lead Pedro) | **v11 M5_v2** (nosso lead) | v11 M5_v3 |
|---|---|---|---|---|---|
| `br_only` MCC | 0.279 | 0.238 | **0.605** | **0.574** | 0.561 |
| `acb` spec | 0.803 | 0.842 | 0.959 | **0.951** | 0.934 |
| `app` recall | 0.417 | 0.460 | 0.436 | **0.405** | 0.521 |
| `global` MCC | 0.512 | 0.606 | 0.512 | **0.626** | 0.626 |

### 7.7 🎯 A FALSIFICAÇÃO DA CALIBRAÇÃO (o resultado mais importante)
Controles negativos estratificados sobre o **desconto** (embaralha *quem recebe qual desconto*, preservando estrutura), `br_only` MCC, 50 seeds × 4 modos:

| modo | real | controles (média) | **p(controle ≥ real)** |
|---|---|---|---|
| global / within_gene / within_af_bin / within_chromosome | 0.561 | 0.46 – 0.52 | **0.0196 (todos)** |

`p = 1/51` = o mínimo possível → **nenhum dos 200 embaralhamentos alcançou o real**, nem no `within_af_bin` (o mais estrito). **A especificidade ABRAOM está FALSIFICADA de forma positiva no v11** — o ganho depende de *quais variantes específicas* são comuns no ABRAOM, não de frequência genérica. **O v10 do Pedro NÃO conseguiu isso** (lá os controles chegavam perto → "não falsificado"). **A replicação resolveu a pergunta aberta dele.**

---

## 8. As tentativas de MELHORIA (ambas negativas — não repetir)

Depois do porte, o Gabriel pediu para **aumentar o `br_only`, mesmo perdendo global**. Duas alavancas, ambas fechadas com negativo medido:

### 8.1 Alavanca B — re-tunar o desconto: **SATURADO**
Lendo o grid do tuning já existente (`~/v11eval/m5_v2_v11/holdout_tuning_results.csv`), a fronteira `br_only ↔ recall` é **fixa** (o global fica constante em 0.635 o tempo todo):

| piso de recall | br_only MAX | recall |
|---|---|---|
| ≥0.00 | 0.698 | **0.075** ☠️ (colapso) |
| ≥0.30 | 0.612 | 0.310 |
| **≥0.40** | **0.597** | 0.406 ← nosso lead já está aqui |

**O "teto" de 0.698 é uma miragem** — alcançado chamando quase tudo de benigno (perde 92% das patogênicas). O desconto só **desliza** na fronteira; não a levanta. Ganho válido só existe trocando **recall** (não global), o que contraria o objetivo clínico.

### 8.2 Alavanca A ("A-guarda") — guarda por cabeças nativas: **REFUTADA**
**A ideia:** a guarda do M5_v3 protege por `molecular_probability`, que no v11 super-estima benignas comuns. Trocar por **conservação (phyloP) + missense-severity (ESM-2)** — sinais ortogonais à frequência que o v10 não tinha.

**Construímos e testamos (2 tentativas):**
- Tentativa 1 (guarda = molecular **E** conservação): falhou porque **as founders têm `molecular_probability` BAIXO (mediana 0.374)** — só **6 de 187** passam o gate de 0.65 → o `E` bloqueava 97% de quem deveria proteger. (Descoberta lateral: o guard molecular original dispara para 6 founders e **83 benignas** — protege ~14× mais benigna que founder; **é por isso que o v3 do Pedro fez 208 FPs para 19 resgates no v11**.)
- Tentativa 2 (conservação **substitui** o gate molecular, thresholds 1.5–4.0): também selecionou `conservation_guard_threshold = 0.0`.

**Causa-raiz (medida):** os sinais **não separam**. As medianas se sobrepõem — founders phyloP100 **0.357** vs benignas comuns **−0.183**; missense-severity 4.24 vs 3.68 (**enrichment ~1.0–1.8× = inútil**). O enrichment do phyloP tem pico de só **5.4×** (thr 3.0) contra uma **base rate de 61:1** (11497 benignas vs 187 founders) → ainda ~11 benignas guardadas por founder.

> **⚠️ Erro a não repetir:** a proposta nasceu das **médias** (founders 1.41 vs benignas 0.07), que eram puxadas por uma cauda conservada. **As medianas mostram que metade das founders é tão pouco conservada quanto uma benigna comum.**

**O valor do negativo:** nem as cabeças nativas do v11 distinguem uma founder patogênica brasileira de uma benigna comum. **O gargalo não é o modelo** — essas variantes se parecem para todo sinal molecular disponível. Isso **reproduz independentemente a conclusão do Pedro** (`do_not_train_next` → curadoria), agora com evidência medida num backbone melhor.

---

## 9. O código que construímos (o que é cada arquivo)

### Scripts novos (nossos)
| Arquivo | O que faz |
|---|---|
| `scripts/compare_freq_adapters.py` | Bootstrap **pareado** de Spearman entre runs de adapter (Fase 1 / B). |
| `scripts/compare_fusion_falsification.py` | Bootstrap **pareado de MCC/specificity/recall** entre um modelo real e seu controle scrambled, numa slice. MCC/threshold espelham `eval/clinvar/metrics.py` (o ponto reproduz o summary = sanity embutido). **Foi a ferramenta que cravou a Fase 3.** |
| `scripts/assemble_regional_baseline_csv.py` | Monta o **baseline-csv** que a calibração exige (`model,dataset,n,auroc,auprc,mcc,recall,specificity`) a partir dos `regional_eval_summary.json`. O tuning só consulta a linha **M0**. |
| `scripts/extract_native_pathogenicity_features.py` | Extrai **conservação + missense-severity** das cabeças nativas do v11 por variante (standalone, backbone-only; reusa `build_variant_cache`). Grava `{slice}.{split}.native_features.parquet`. **Funciona** (usado na A-guarda). |

### Edições em arquivos do Pedro
| Arquivo | Mudança |
|---|---|
| `eval/clinvar/beat_v11_adapter.py` | O adapter v11 + o método **`extract_native_pathogenicity_features`** (conservação/missense na posição da variante). |
| `eval/clinvar/adapters.py` | Família `beat-v11` no `build_finetune_adapter` + aliases. |
| `eval/clinvar/lora.py` | `_EXCLUDE_PATTERNS` += cabeças v11 + MoE. |
| `scripts/clinvar_m0_job.py`, `scripts/clinvar_fusion_job.py` | Args de job `--model-family`/`--model-version` (default v10) threadados no `_upsert_arg` (antes **hard-forçavam** v10). |
| `scripts/sagemaker_clinvar_m0.py`, `scripts/sagemaker_clinvar_fusion.py` | Passam esses args **antes do `--`** (passthrough depois do `--` é sobrescrito pelo `_upsert_arg`). |
| `scripts/train_abraom_frequency_adapter.py` | `--loss {bce,mse,huber}` + alvos `delta_logit`/`scrambled_delta_logit` (Experimento B). |
| `scripts/calibrate_m5_v3_safety.py` | **A-guarda:** campo `conservation_guard_threshold` no `SafetyConfig`, helper `_guard_mask`, merge das `native_features`, `--native-dir`, grid de conservação. **Back-compat: sem `--native-dir` é o v3 original.** (Refutada, mas o código fica.) |

---

## 10. Artefatos e caminhos no notebook

```
~/testeArq/lumina-beat-regionalization    o repo (git pull aqui)
~/slices/                                  as 13 slices .parquet (baixadas do S3)
~/hg38/hg38.fa                             o fasta (para reconstruir janelas de variante)
~/v11eval/
  ├─ reval_m0b, reval_m4b, reval_m5b, reval_m7sb, reval_m7db     eval regional (test, 5 slices)
  ├─ reval_m0c, reval_m4c, reval_m5c, reval_m7sc, reval_m7dc     eval br_only/br_any split `all`
  ├─ reval_m0d, reval_m7dd                                        M0/M7d em 8 slices × test+holdout
  ├─ reval_m5bounded                                              M5-bounded, 8 slices × test+holdout
  ├─ m5v3_modelroot/                                              layout p/ o v3 (subdirs de nome FIXO)
  │    ├─ m0_nonbr_beatv10_v1_sagemaker/        (recebe o M0 v11)
  │    └─ m7_dynamic_scrambled_nonbr_beatv10_v1_sagemaker/  (recebe o M7d v11)
  ├─ m5_v2_v11/            saída da calibração v2 (selected_config.json + summary + tuning)
  ├─ m5_v3_v11/            saída do v3 original
  ├─ m5_v3_aguarda, m5_v3_aguarda2   as duas tentativas refutadas da A-guarda
  ├─ native_features/      conservação+missense por variante (8 slices × test+holdout)
  └─ baseline_v11_test.csv o baseline-csv da calibração
```

**Jobs principais (nomes completos):**
```
M0          clinvar-m0-nonbr-v11-r1-952aa6-20260713011715
M4 static   clinvar-fuse-m4-static-v11-r1-e2a6e0-20260713193515
M5 dynamic  clinvar-fuse-m5-dynamic-v11-r1-8974a8-20260713165005
M7 static   clinvar-fuse-m7-static-scr-v11-r1-cd2194-20260714044928
M7 dynamic  clinvar-fuse-m7-dynamic-scr-v11-r1-2afaf7-20260714045005
M5-bounded  clinvar-fuse-m5-bounded-v11-r1-0402dd-20260716220945
```

---

## 11. ⚠️ GOTCHAS que custaram tempo (leia — cada um destes já queimou horas)

1. **Nunca confie em default/doc — leia o artefato.** (`d_full` era 448 não 320; MoE off; a receita do Pedro ≠ defaults do trainer.)
2. **`--lora-rank` da fusão TEM que ser 8** (= o rank do M0). O commit `d79fd31` faz o load do `--init` **filtrar silenciosamente** chaves com shape divergente → com rank errado o caminho molecular do M0 **some sem dar erro**.
3. **O eval regional herda o `batch_size` do checkpoint (=2) se você não passar `--batch-size`.** O Pedro passava **8**. Isso + o backbone v11 ~2× mais pesado + a varredura completa **estourou o limite de 8h duas vezes** (`MaxRuntimeExceeded`, resultado parcial sem `summary.json`). **Sempre passe `--batch-size 8` (ou 16) e restrinja `--dataset-files`/`--splits`.**
4. **`list-training-jobs` sem `--no-paginate`** aplica o `--query` **por página** e devolve o nome do job seguido de dezenas de `None`, corrompendo a variável do shell (erro enganoso "nome > 63 caracteres"). Use `--no-paginate`, ou o `job_name=` que o launcher imprime, ou ache o artefato por `aws s3 ls`.
5. **O split `all` só é seguro nas slices BRASILEIRAS** (`br_only`, `br_any` — disjuntas do treino por construção). Em `abraom_common_benign`/`global_nonbr_no_abraom` ele **vazaria** (intersectam o `nonbr_only` do treino).
6. **Poder estatístico não é detalhe.** Com n=504 o `br_only` enganou **nos dois sentidos** (um "sinal" de +0.054 que era ruído; e a conclusão falsa de que o backbone não ajudava o brasileiro). **Sempre pergunte: qual o CI? qual o n?**
7. **Significância estatística ≠ prática.** Na slice de 12k, real×scrambled deu "significativo" nas duas arquiteturas mas **com sinais opostos** (+0.0145 e −0.0079) — efeito minúsculo + direção inconsistente = idiossincrasia, não biologia.
8. **Média × mediana.** A A-guarda foi proposta a partir de médias e refutada pelas medianas. **Olhe a distribuição, não o resumo.**
9. **`_upsert_arg` sobrescreve passthrough.** Args de modelo têm que ir **antes do `--`** nos launchers de M0/fusion.
10. **O `--model-root` do `calibrate_m5_v3_safety.py` espera subdirs de nome FIXO** (`m0_nonbr_beatv10_v1_sagemaker`, `m7_dynamic_scrambled_nonbr_beatv10_v1_sagemaker`, linhas 978-979), mesmo no v11.
11. **`--native-feature-heads` é para beat-v6/v8** — "silently ignored" no v11. Use `none`. Para usar as cabeças nativas do v11 é preciso **código** (foi o que o `extract_native_pathogenicity_features.py` faz).
12. **A calibração roda LOCAL** (lê os `predictions.parquet`), sem job. Iterar nela é barato.
13. **Seleção de checkpoint por NLL é degenerada** para alvos ilimitados (`delta_logit`) — subestimou o Experimento B.

---

## 12. Documentos-chave (leia nesta ordem)

| Documento | O que é |
|---|---|
| **`TCC_REGIONALIZACAO_V11.md`** | **O documento principal** (iteração 3). Didático e completo: conceitos, arquitetura, todas as fases, resultados, decisões, lições, glossário. **Comece por aqui.** |
| **`RESULTADOS_REGIONALIZACAO_V11_FASES2-4.md`** | Resultados das Fases 2-4 + a tabela definitiva + **explicação didática da falsificação** (o que é, por que é boa). |
| `RESULTADOS_REGIONALIZACAO_V11.md` | Resultados da Fase 1 (adapters). |
| `HANDOFF_CONTINUACAO_V11.md` | O handoff antigo (estado "Fase B") — **superado por este**. |
| Baseline do Pedro | `lumina-ssm/artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md` |

---

## 13. O QUE FAZER A SEGUIR

**A frente de regionalização está tecnicamente COMPLETA e as melhorias de modelagem estão esgotadas** (duas alavancas testadas, ambas com negativo bem-dimensionado). As opções, em ordem de valor:

### (a) Curadoria externa de P/LP brasileiros — **o único caminho de ganho real**
É a conclusão a que **duas investigações independentes** (Pedro no v10, nós no v11) chegaram. Os sinais moleculares não distinguem founder de benigna comum; só rótulos melhores resolvem. **Gated por dados** — depende do Eduardo/colaboradores.

### (b) Escrever a dissertação
O material está pronto: o `TCC_REGIONALIZACAO_V11.md` (iteração 3) + o `RESULTADOS_..._FASES2-4.md` cobrem tudo, com a tabela definitiva e a falsificação explicada.

### (c) Se quiserem MAIS `br_only` aceitando perder recall (a troca da Alavanca B)
Zero treino — é só re-selecionar um config do grid já computado:
`recall ≥0.30 → br_only 0.612` · `recall ≥0.20 → br_only 0.629`.
**Não recomendado** (troca a capacidade de pegar founders brasileiras), mas é decisão do Gabriel/Eduardo.

### (d) Itens opcionais, baixa prioridade
- Re-run do Experimento B selecionando checkpoint por **Spearman** (não NLL) — refino, não bloqueio.
- Controles que pulamos do roster do Pedro: `M2` (gnomad-only) e `M6` (freq-explícita alternativa) — não mudam a conclusão.
- ~~Registrar o negativo da A-guarda nos documentos~~ — **FEITO (2026-07-23, iteração 4 do TCC).** Os dois negativos do pós-porte (Alavanca B saturada + A-guarda refutada) estão agora em `TCC_REGIONALIZACAO_V11.md` §7 ("Pós-porte"), §9 (lições 8-10), §10.9, §11 e glossário; e em `RESULTADOS_..._FASES2-4.md` §8.2. **Não estão mais só na memória.**

### Pendência operacional
Confirmar que **tudo está commitado e pushado** no `main` (os scripts novos e as edições da calibração). O notebook trabalha a partir do `git pull`.

---

## 14. Referência rápida

**Checkpoint r1:** `s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt`
**Lead do v11:** `M5_v2` — `br_only` MCC **0.574**, `acb` spec **0.951**, `app` recall **0.405**, `global` MCC **0.626** (test).
**O headline:** calibração leva `br_only` **0.24 → 0.57**; **falsificação positiva p=0.0196** (o v10 não conseguiu); v11 **ganha no global** (0.626 vs 0.512).
**A tese:** *o valor do ABRAOM não é aprendível da sequência — está na frequência observada usada como calibração, e essa calibração é comprovadamente ABRAOM-específica.*

*Fim do handoff. Estado em 2026-07-23: porte COMPLETO (Fases 0-5 + 4), melhorias de modelagem esgotadas com negativos medidos, documentação escrita. Próximo passo real = curadoria externa (gated) ou escrita da dissertação.*
