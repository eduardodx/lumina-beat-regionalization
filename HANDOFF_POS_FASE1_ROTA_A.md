# HANDOFF — Regionalização Beat-v11 no FORK pós-Fase 1 (foco na ROTA A)

> **Quando usar este documento:** você está no **ponto de decisão logo após a Fase 1** (adapters de
> frequência) ter fechado. Há uma bifurcação **Rota A × Rota B** (§8). Este handoff detalha
> especialmente a **Rota A** (completar o porte downstream: M0 → fusion → calibração → eval), para o
> caso do gestor (Eduardo) preferir ir direto por A **por questão de tempo**. Se a escolha for a Rota
> B (teste de resíduo), use o handoff irmão `HANDOFF_CONTINUACAO_V11.md`.
>
> **Auto-contido** — cole inteiro no novo chat. Datado **2026-07-11**. Não há memória compartilhada nem
> histórico de chat; tudo o que importa está aqui.

---

## 0. TL;DR — onde estamos e a decisão

1. Estamos **portando o estudo de regionalização ABRAOM do Pedro** (validado no **Beat-v10**) para o **Beat-v11**, no repo `lumina-beat-regionalization`.
2. **Fase 0 (integração v11) e Fase 1 (adapters de frequência) estão FECHADAS.** Veredito da Fase 1: **há sinal sequência→frequência real no v11, mas NÃO é brasileiro-específico** (A_BR ≈ A_gnomAD ≫ A_scrambled). Reproduz e explica o caveat científico do Pedro.
3. **Estamos no FORK.** Duas rotas (§8): **(A)** completar o porte downstream (M0 + fusion + calibração + eval) = resultado confirmatório rigoroso; **(B)** teste de resíduo (o teste direto de sinal regional). **Este doc detalha a Rota A.**
4. **A Rota A é a escolha "por tempo/segurança":** o ganho headline do Pedro (M5_v3: `br_only` MCC 0.28→0.61) é **calibração de frequência**, que **independe** do null da Fase 1 → tende a reproduzir no v11. Entrega "v11 reproduz a conclusão nuançada do v10", que é publicável.

---

## 1. A missão (o "porquê")

**Regionalização** = melhorar a interpretação de patogenicidade de variantes (ClinVar) no **contexto brasileiro/latino-americano**, usando o **ABRAOM** (base genômica brasileira — coorte SABE de São Paulo). O ABRAOM dá **frequência alélica (AF) por população brasileira**.

Variantes brasileiras são o **ponto cego estrutural** de foundation models de DNA. A hipótese: o gargalo é o **prior/atalho de frequência** (a AF de referência é majoritariamente europeia — gnomAD NFE), que penaliza variantes comuns em populações não-europeias. Calibrar a AF por população (ABRAOM) deveria ajudar.

O Pedro atacou isso no Beat-v10 e o resultado foi **positivo mas cientificamente nuançado** (§3). **Nossa missão:** portar o pipeline pro **Beat-v11** (backbone melhor) e validar se os indícios se sustentam, adaptando o acoplamento à nova arquitetura.

---

## 2. Os repositórios (o que é cada um)

Todos de `github.com/eduardodx/`.

- **`lumina-beat-regionalization`** — **NOSSO repo de trabalho** (branch `main`, já pushado). Montado copiando a árvore git-tracked do `lumina-ssm@abraom-regionalization-study` (pipeline do Pedro) + vendorizando `lumina_beat_v11/` (o modelo). **É aqui que você trabalha; a outra máquina clona daqui.**
- **`lumina-ssm`** (branch `abraom-regionalization-study`) — repo core de modelos do Eduardo; a **baseline do Pedro no v10**. Use pra `diff` (Pedro × nosso), pra ler os artefatos dele (`artifacts/abraom_frequency_adapter/*`, `artifacts/clinvar_regional_eval/*`, `artifacts/clinvar_regional_m0/*`) e os relatórios (`ABRAOM_RESEARCHER_TRANSFER_REPORT.md`).
- **`lumina-beat`** (branch `main`) — repo dos foundation models Beat. Contém `beat-v11/lumina_beat_v11/` (o pacote que vendorizamos) e `beat-v11/PLANO_ACAO_REGIONALIZACAO_V11.md` (plano original). Commit que casa com o r1: `3d15e66`.
- **`lumina-benchmarks-mosaic-eval`** — benchmark do gestor; **outra frente** (contexto, não é esta).

---

## 3. O estudo do Pedro no Beat-v10 (a baseline que estamos portando)

Docs dele em `lumina-ssm@abraom-regionalization-study`: `artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md` (completo).

### 3.1 A escada de modelos (M0→M7) — a arquitetura da Rota A
- **M0** = baseline molecular de patogenicidade (ClinVar **non-BR**, sem ABRAOM). O piso contra o qual todo ganho regional é medido.
- **A_BR / A_gnomAD / A_scrambled** = **adapters de frequência** (LoRA que predizem AF a partir da sequência). **Já treinados no v11 na Fase 1** (§7) → são inputs reutilizáveis do fusion.
- **M4** = static fusion (M0 + adapters via gate, sem frequência explícita forte).
- **M5 / M6** = fusion + frequência explícita (bruto). **Reduziram falso-positivo mas SUPRIMIRAM P/LP** comuns no ABRAOM (founder/recessivo brasileiro) — colapso de recall.
- **M5_v2_calibrated** = desconto de frequência limitado.
- **M5_v3_safety = candidato final:** desconto regional limitado + **guarda molecular** (frequência pode reduzir o score mas NÃO apagar evidência molecular forte). Config do Pedro: `{discount_scale: 0.5, max_discount: 0.5, molecular_guard_threshold: 0.65, guarded_max_discount: 0.0, guard_score_floor: 0.35, regional_threshold: 0.35, global_threshold: 0.72}`.
- **M7_scrambled / M2_gnomad_only** = controles negativos.

### 3.2 O resultado headline e o caveat (CRÍTICO)
- **Headline:** M5_v3_safety: `br_only` MCC **0.279 → 0.605**, specificity de benignas-comuns-ABRAOM **0.803 → 0.959**, sem degradar o global (MCC 0.512). **A calibração de AF regional funciona operacionalmente.**
- **Caveat honesto:** os **controles negativos estratificados** (preservam gene/AF-bin/tipo/specificity/cromossomo, quebram só o link variante↔frequência) **chegam perto do real** → a **especificidade biológica do ABRAOM NÃO está falsificada**. Parte do ganho pode ser estrutura genérica de frequência.
- **Veredicto do Pedro (v10):** `do_not_train_next` — o próximo passo era **curadoria** de 75 variantes críticas + validação externa, NÃO mais treino. Isso está **gated** (precisa de mais dados) — foi por isso que a direção virou "portar pro v11".

### 3.3 As 4 slices decisivas do eval (a métrica da Rota A)
| Slice | Mede | Alvo |
|---|---|---|
| `br_only` | ganho no subconjunto brasileiro | **MCC** ↑ (0.279→0.605 no v10) |
| `abraom_common_benign` | redução de falso-positivo em benignas comuns ABRAOM | **specificity** ↑ (0.803→0.959) |
| `abraom_pathogenic_present` | **do-not-suppress** P/LP founder/recessivo comuns no ABRAOM | **recall** (não colapsar) |
| `global_nonbr_no_abraom` | não-degradação fora da regionalização | **MCC** estável (0.512) |

---

## 4. A arquitetura do Beat-v11 (fatos VERIFICADOS ao vivo — NÃO confie nos defaults)

> **Lição transversal: nesta base, defaults e doc MENTEM. Sempre leia o artefato/modelo carregado.**

- **Checkpoint alvo (r1):** `s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt` (SISO, **52.1M**). `config['model']='beat_v11_bioprime'`, 74 chaves. Casa com `lumina-beat@main` commit `3d15e66`.
- **`d_full = 448`** (`d_model=384` + `d_pure=64`), mid=384. **NÃO 320** — o r1 sobrescreve os defaults do dataclass. Ler `model.cfg`/`model.full_hidden_dim`.
- **MoE OFF no r1** (`moe_enabled=False`, 0 lineares MoE ao vivo).
- **`encode()` renomeou:** v11 = `{"last_hidden_state","mid_hidden_state"}`; v10 = `{"hidden_states","mid_hidden_states"}`.
- **Cabeças nativas:** `population_af_head → gnomad_af_pred[B,L,4]`, `population_observed_head`. O v11 já prediz AF gnomAD nativamente.
- **Loader:** `lumina_beat_v11.load_model_from_checkpoint(checkpoint_path, device=..., strict=True)` (trata S3, config, prefixo). **NÃO use o registry** (`build_registered_model` rejeita as 74 chaves).
- **Superfície LoRA @ rank 8:** 105 lineares / 1.248M params (mamba `in/out_proj` + attn `q/k/v/out_proj` + `up_stages.gate` + `stem.purity`; zero cabeças). Vocab idêntico ao `src.constants`.
- **Gotcha tilelang:** o env conda do notebook tem `tilelang/tvm` quebrado (`tvm_ffi AttributeError`, Py3.12) → **fix já commitado:** shim no topo de `src/models/__init__.py` (`sys.modules["tilelang"]=None` → fallback triton; r1 é SISO). No **job** é no-op (`INSTALL_TILELANG=0`).

---

## 5. O que já foi CONSTRUÍDO (integração v11 — Fase 0 + Fase 1)

Verificado por `diff` contra a baseline do Pedro (mudanças limpas). Tudo em `lumina-beat-regionalization`.

### 5.1 Arquivos NOVOS
| Arquivo | O que é |
|---|---|
| `eval/clinvar/beat_v11_adapter.py` | **`FineTuneBeatV11Adapter`** — carrega o v11 via package loader; extrai `last_hidden_state`; shim tilelang. Implementa o protocolo `FineTuneAdapter`. |
| `scripts/smoke_beat_v11_regionalization.py` | Smoke Fase 0: forward parity (448/384/4/4) + `apply_lora`. |
| `scripts/smoke_beat_v11_freq_train.py` | Smoke de acoplamento de treino (sintético). |
| `scripts/compare_freq_adapters.py` | **Bootstrap pareado** de spearman entre runs (a ferramenta de análise dos adapters). |
| `RESULTADOS_REGIONALIZACAO_V11.md` | Doc de resultados da Fase 1. |
| `lumina_beat_v11/` | Pacote standalone do modelo r1 (vendorizado). |

### 5.2 Edições em arquivos do Pedro (mínimas, back-compat)
| Arquivo | Mudança | Relevante p/ Rota A? |
|---|---|---|
| `eval/clinvar/adapters.py` | Família **`beat-v11`** no `build_finetune_adapter` + aliases. | **SIM — é o que faz o M0/fusion aceitar o v11.** |
| `eval/clinvar/lora.py` | `_EXCLUDE_PATTERNS` += cabeças v11 + MoE (`.router`/`.experts.` congeladas). | SIM (LoRA correto no v11). |
| `src/models/__init__.py` | Shim tilelang. | SIM (import não crasha). |
| `scripts/train_abraom_frequency_adapter.py` | Família beat-v11 (via factory) **+ scaffolding da Rota B** (`--loss`, alvos `delta_logit`/`scrambled_delta_logit`). | O scaffolding B é **back-compat e inócuo pra Rota A** (ignore; só afeta os adapters de frequência, que já rodaram). |
| `scripts/sagemaker_abraom_frequency_adapter.py` | Só comentário. | — |

> **Nota sobre o scaffolding B no trainer:** o repo atual tem as adições `--loss`/`delta_logit` no `train_abraom_frequency_adapter.py`. Elas são da opção B (teste de resíduo) e **não afetam a Rota A** — o M0/fusion nem usam esse script. Pode ignorar.

### 5.3 Validações (todas passaram)
Forward do r1 ✓ · `apply_lora` (105 lineares, nada vazado) ✓ · smoke de treino ✓ · job SageMaker real ✓ · Fase 1 completa (§7).

---

## 6. Infra (AWS / SageMaker / S3)

- **Conta:** `085188779747`, **us-east-2**. Windows/laptop NÃO tem S3/GPU — **tudo roda no SageMaker notebook** (`~/testeArq/lumina-beat-regionalization`; persiste na nuvem, acessível de qualquer máquina). Na outra máquina você **edita** (clone do GitHub) e **sincroniza pro notebook** (git push/pull).
- **S3:** checkpoint r1 = `s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/`; datasets = `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/...`; artefatos dos jobs = `s3://ai4bio-lumina-experiments-v2/lumina-ssm/<pipeline>/<experiment>/sagemaker-artifacts/<JOB>/output/model.tar.gz`.
- **Gotchas (leia):** quota = **1 `ml.g5.2xlarge` de treino concorrente** (use `g5.4xlarge` pra 2º job em paralelo); o launcher imprime `job_name=<base>` mas o SageMaker **anexa timestamp** → ache o job real com `list-training-jobs --name-contains <base>`; `/tmp` do notebook é efêmero → checkpoints em `~/ckpts/`; **idle-shutdown mata run longo e `nohup` NÃO protege** → use jobs SageMaker destacados (`--detach`).
- **Como ler um artefato:** `aws s3 cp <.../model.tar.gz> m.tar.gz && tar -xzf m.tar.gz` → leia o `summary.json`/`metrics.json` (config + métricas). **Sempre leia o artefato, não assuma os defaults.**

---

## 7. RESULTADOS da Fase 1 (FECHADA — a base do fork)

4 adapters de frequência, receita do Pedro (ctx 1024, 1000 steps, balanced-af, sequence-only). Métrica honesta = **Spearman**. Doc: `RESULTADOS_REGIONALIZACAO_V11.md`.

| Adapter | v10 (val/test) | **v11 (val/test)** |
|---|---|---|
| **A_BR** (`af_abraom`) | 0.114 / 0.104 | **0.135 / 0.127** |
| **A_gnomAD** (`af_gnomad`→mede `af_abraom`) | 0.107 / 0.118 | **0.120 / 0.118** |
| **A_scrambled** (controle−) | −0.013 / 0.001 | **−0.036 / −0.021** |

**Bootstrap pareado** (`compare_freq_adapters.py`, B=2000, n≈5000):
- **A_BR − A_scrambled** = +0.172/+0.149, CI **[+0.131,+0.213]/[+0.105,+0.191]**, P=1.000 → **sinal seq→AF real e forte**.
- **A_BR − A_gnomAD** = +0.016/+0.010, CI **[−0.005,+0.036]/[−0.018,+0.039]**, P=0.93/0.73 → **CI cruza 0 = não-regional**.

> **Veredito Fase 1:** `A_BR ≈ A_gnomAD ≫ A_scrambled` — há sinal seq→AF **real** no v11, mas **NÃO brasileiro-específico**. Reproduz + explica upstream o caveat do Pedro. O +20% v11-sobre-v10 é melhora **genérica** (backbone melhor aprende AF melhor). Contexto 4096 não compra nada (AF é local) → **usar ctx 1024 em tudo**.

**Jobs da Fase 1 (todos Completed, artefatos no S3):**
- `abraom-freq-v11-r1-abr-df0483-20260710010459` (A_BR ctx1024)
- `abraom-freq-v11-r1-gnomad-9f35bc-20260710124718` (A_gnomAD)
- `abraom-freq-v11-r1-scrambled-52e9fb-20260710174346` (A_scrambled)
- (+ `abr-ctx4096-5d9db4-...` = ablação de contexto)

**Esses adapters A_BR e A_gnomAD são os INPUTS reutilizáveis do fusion na Rota A.**

---

## 8. O FORK — Rota A × Rota B

Depois do null-regional da Fase 1, a direção deixou de ser óbvia:

- **Rota A — Completar o porte downstream (Fases 2-5):** reproduzir **M0 + fusion + M5_v3_safety + eval** no v11. **Testa se o ganho de CALIBRAÇÃO reproduz** (br_only 0.28→0.61). Como esse ganho é calibração de AF (não depende do adapter ter sinal regional), **tende a reproduzir** → resultado confirmatório rigoroso "v11 reproduz a conclusão nuançada do v10". **Menor risco, resultado garantido, publicável.** É a escolha "por tempo". **← ESTE HANDOFF DETALHA ISTO (§9).**
- **Rota B — Teste de resíduo (`delta_logit`):** o teste DIRETO do componente regional (treinar um adapter em `af_abraom − f(af_gnomad)`). Maior upside (poderia achar sinal que o teste indireto não achou), mas expectativa baixa dado o null. Decide se o fusion precisa de estrutura nova. **Detalhado no handoff irmão `HANDOFF_CONTINUACAO_V11.md`** (código já pronto: `--loss huber` + alvos `delta_logit`/`scrambled_delta_logit`).

**Por que a Rota A não é "reproduzir um null":** o null da Fase 1 é sobre o **adapter de frequência** não ter sinal regional específico. Mas o **M5_v3** ganha via **calibração** (usar a AF ABRAOM pra descontar benignas comuns + guarda molecular pra proteger founder/recessivos) — um mecanismo diferente, que independe do adapter. Reproduzir isso no v11 é uma pergunta **aberta e válida**: o backbone melhor muda o quanto a calibração ajuda? A guarda molecular precisa de re-tuning? Os controles negativos ficam mais ou menos apertados?

---

## 9. ROTA A em detalhe (M0 → fusion → calibração → eval no v11)

Todo o pipeline downstream roda via **`python -m eval.clinvar.run --regime A`** (o caminho de embeddings: two-tower ref/alt → cabeça). **`eval.clinvar.run` despacha por `--model-family` usando `build_finetune_adapter`** — e nós já adicionamos a família `beat-v11` lá (§5.2). **Então o lado do adapter está pronto; o trabalho da Rota A é (i) destravar os launchers e (ii) adaptar o eval.**

### 9.1 O BLOCKER principal — os launchers forçam v10
Diferente do launcher de frequência (que tinha passthrough), os jobs de M0 e fusion **FORÇAM** o modelo via `_upsert_arg` (que sobrescreve até o passthrough):
- `scripts/clinvar_m0_job.py` linhas ~52-53: `_upsert_arg(runtime_args, "--model-family", "lumina")` e `"--model-version", "beat-v10"`.
- `scripts/clinvar_fusion_job.py` linhas ~91-92: idem.

**A EDIÇÃO NECESSÁRIA** (em ambos): trocar `"lumina"→"beat-v11"` e `"beat-v10"→"r1"` — OU, melhor (back-compat), adicionar args de nível-job (`--model-family`/`--model-version` no `parse_args` do job, default v10) e passar esses valores pro `_upsert_arg`. Recomendo a versão parametrizada (não quebra reruns v10). Depois, apontar o **canal `checkpoint`** pro r1 (`--checkpoint-s3-prefix s3://.../lumina-beat-v11v5-r1-202607071631/ckpt/` no launcher `sagemaker_clinvar_m0.py`/`sagemaker_clinvar_fusion.py`).

### 9.2 Fase 2 — M0 (baseline molecular no v11)
- **O que é:** treinar a cabeça de patogenicidade em **ClinVar non-BR** (leave-Brazilian-out — o teste limpo). Dataset: canal `dataset` = `nonbr_only.parquet` (default do `clinvar_m0_job.py`).
- **Chain:** `sagemaker_clinvar_m0.py` (launcher) → `clinvar_m0_job.py` (in-job) → `python -m eval.clinvar.run --regime A ...`.
- **Receita:** extrair do artefato do M0 do Pedro (mesmo método da Fase 1): baixar `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/clinvar-m0-nonbr-beatv10-v1/sagemaker-artifacts/.../model.tar.gz`, ler o `config` do metrics/summary.json (tem `regime, model_family, context_size, batch_size, loss_type, precision, ...`). **NÃO assuma defaults.**
- **Saída:** o checkpoint do M0 (vira o `--init-model` do fusion).

### 9.3 Fase 3 — Fusion M4/M5 (no v11)
- **O que é:** combinar M0 (init) + os **adapters de frequência abraom + gnomad da Fase 1** via gate.
- **Chain:** `sagemaker_clinvar_fusion.py` → `clinvar_fusion_job.py` → `eval.clinvar.run`.
- **Args-chave** (`clinvar_fusion_job.py`): `--fusion-mode {static_lora=M4, dynamic_lora=M5}`; `--fusion-adapter-names abraom gnomad`; `--fusion-adapter-paths <S3 dos adapters da Fase 1>`; `--init-model <M0 da Fase 2>`.
- **REUSÁVEL:** os adapters A_BR (`abraom`) e A_gnomAD (`gnomad`) da Fase 1 são os inputs — já treinados no v11. Os paths S3 deles estão em `s3://.../abraom-frequency-adapter/<experiment>/sagemaker-artifacts/<JOB>/output/`.
- **Mecanismo (em `eval/clinvar/fusion_lora.py`):** `convert_lora_backbone_to_{static,dynamic}_fusion` troca cada `LoRALinear` por um wrapper de fusão (`StaticFusionLoRALinear` = softmax sobre adapters; `DynamicFusionLoRALinear` = gate MLP). É model-agnostic dado o `LoRALinear` — funciona no v11.

### 9.4 Fase 4 — Calibração (M5_v2 / M5_v3_safety)
- **Scripts (model-agnostic, operam em SCORES):** `scripts/calibrate_m5_v2_regional_scores.py` (desconto limitado) e `scripts/calibrate_m5_v3_safety.py` (guarda molecular). Re-fitar os thresholds nos scores do v11 (a distribuição muda → os thresholds do §3.1 quase certamente mudam).
- **Pergunta v11-específica:** as cabeças counterfactual/splice do v11 são diferentes → o `molecular_guard_threshold` (0.65 no v10) provavelmente precisa de re-tuning.

### 9.5 Fase 5 — Eval + falsificação (o entregável)
- **Slices:** as 4 decisivas (§3.3). Materializar com `scripts/build_regional_clinvar_eval_slices.py` (já existe; params `common_af_threshold=0.01`, `high_specificity_threshold=0.05`).
- **Eval regional:** `scripts/evaluate_abraom_regionalization.py` — **⚠️ RE-RODA o modelo** via `src/checkpoints.py:load_lumina_backbone_from_checkpoint` (linha 124, loader **v10**). **Adaptação v11 necessária:** ou trocar por `lumina_beat_v11.load_model_from_checkpoint`, ou adicionar suporte v11 ao `load_lumina_backbone_from_checkpoint`. (É o `evaluate_abraom_regionalization.py` que eu ainda NÃO tinha lido nas fases anteriores — este é o principal ponto de código v11 além dos launchers.) Também há o `clinvar_regional_eval_job.py` + `sagemaker_clinvar_regional_eval.py` (o eval via SageMaker).
- **Falsificação:** `scripts/validate_regional_signal_next_step.py` — os controles negativos estratificados (within gene/AF-bin/tipo/specificity/cromossomo). **É o teste que dá honestidade científica** (no v10 os controles chegaram perto do real).
- **Head-to-head v11 × v10:** o entregável final — comparar as 4 slices v11 vs v10 e ver se o M5_v3 reproduz.

### 9.6 O que a Rota A REUSA vs precisa FAZER
| Reusa as-is | Precisa fazer no v11 |
|---|---|
| Adapters de frequência (Fase 1: A_BR, A_gnomAD) | Editar hardcode dos launchers M0/fusion (§9.1) |
| Datasets (ClinVar×ABRAOM, `nonbr_only`, slices) | Adaptar `evaluate_abraom_regionalization.py` p/ loader v11 (§9.5) |
| `fusion_lora.py` (mecanismo, model-agnostic) | Re-treinar M0 + fusion no v11 (jobs SageMaker) |
| Scripts de calibração (operam em scores) | Re-fitar thresholds M5_v2/M5_v3 nos scores do v11 |
| Framework de controles negativos | Extrair as receitas M0/fusion do Pedro (artefatos) |

### 9.7 Critério de sucesso da Rota A
O M5_v3_safety no v11 reproduz o padrão do v10 nas 4 slices? Especificamente: `br_only` MCC sobe substancialmente (~0.28→~0.6), `abraom_common_benign` specificity sobe (~0.80→~0.95), `abraom_pathogenic_present` recall NÃO colapsa (a guarda molecular funciona), `global_nonbr_no_abraom` MCC estável. **E os controles negativos** dizem se o ganho é ABRAOM-específico ou estrutura de frequência (esperado: não-falsificado, como no v10).

### 9.8 O que eu NÃO verifiquei (leia antes de codar a Rota A)
- A **receita exata** do M0 e do fusion do Pedro (extrair dos artefatos — §9.2).
- O corpo do `eval/clinvar/run.py` (confirmei que despacha por `--model-family`; não li o fluxo completo do regime A / fusion).
- O fluxo completo do `evaluate_abraom_regionalization.py` (confirmei que re-roda o modelo via loader v10; a adaptação v11 é o trabalho).
- Se as cabeças de calibração precisam de dados extras (counterfactual/splice do v11).

---

## 10. Rota B (resumo — se o gestor preferir o teste de resíduo)
Treinar um adapter no resíduo `delta_logit = logit(af_abraom) − logit(af_gnomad)` (o componente ABRAOM além do gnomAD) + o controle `scrambled_delta_logit`, e ver se o gap pareado supera o piso. **Código já pronto** (`--loss huber` + alvos threadados no trainer). Comandos e interpretação completos no **`HANDOFF_CONTINUACAO_V11.md`**. Decide se o fusion precisa de estrutura nova (consumir o resíduo) ou não. Expectativa baixa dado o null da Fase 1, mas é o teste direto.

---

## 11. Plano de ação completo + lições

| Fase | Estado |
|---|---|
| 0 · Integração v11 | ✅ |
| 0.5 · Wiring (frequency) | ✅ · ⬜ launchers M0/fusion (§9.1) |
| 1 · Adapters de frequência | ✅ (null regional) |
| **2 · M0** | ⬜ (§9.2 — editar `clinvar_m0_job.py`) |
| **3 · Fusion M4/M5** | ⬜ (§9.3 — editar `clinvar_fusion_job.py`; reusa adapters da Fase 1) |
| **4 · Calibração M5_v2/v3** | ⬜ (§9.4 — scripts model-agnostic) |
| **5 · Eval + falsificação** | ⬜ (§9.5 — adaptar `evaluate_abraom_regionalization.py` p/ v11) |

**Lições (aplicar sempre):**
1. **NUNCA confie em default/doc — leia o artefato.** (Aconteceu 4×: d_full 448 não 320; MoE off; receita do Pedro ≠ defaults; `--target-column` whitelist.)
2. **Comparações pareadas + bootstrap, não pontos** (`compare_freq_adapters.py`).
3. **Spearman é a métrica honesta** dos adapters; nas slices downstream, MCC/specificity/recall (§3.3).
4. **Ao editar launcher/trainer, cheque o caminho inteiro** (o `_upsert_arg` sobrescreve passthrough; whitelists de argparse; loaders v10-only).
5. **O controle negativo é o que dá sentido ao resultado** — sempre rode o scrambled/estratificado.

---

## 12. Estado do código + transferência
- Repo `lumina-beat-regionalization` no GitHub (`github.com/eduardodx/lumina-beat-regionalization`, `main`). **Na outra máquina: `git clone`.** Tudo o que descrevi (integração v11, Fase 1, os 2 handoffs) está commitado/pushado (confirme com `git status`).
- **NÃO está no git** (recupere do S3): `data/datasets/**` (grandes/gitignored — vêm dos canais S3 nos jobs), checkpoints, `model.tar.gz`. Pro trabalho no notebook você não precisa deles localmente — os jobs montam do S3.
- **Ambiente:** SageMaker notebook (env "modelo": conda torch/mamba_ssm/GPU + pyfaidx; persiste na nuvem). Credenciais AWS já configuradas no notebook.

---

## 13. Referência rápida
- **Checkpoint r1:** `s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt`.
- **Entry do M0/fusion:** `python -m eval.clinvar.run --regime A` (despacha por `--model-family beat-v11`).
- **Launchers a editar (Rota A):** `clinvar_m0_job.py:52-53`, `clinvar_fusion_job.py:91-92` (hardcode `_upsert_arg` → v10); `evaluate_abraom_regionalization.py` (loader v10 → v11).
- **Inputs do fusion (Fase 1, reusar):** adapters `abraom` + `gnomad` em `s3://.../abraom-frequency-adapter/<exp>/sagemaker-artifacts/`.
- **Primeira ação (Rota A):** (1) extrair a receita do M0 do Pedro (artefato S3); (2) parametrizar `clinvar_m0_job.py` p/ beat-v11/r1; (3) lançar M0 no v11; (4) idem fusion; (5) calibrar; (6) eval nas 4 slices + controles.
- **Docs a ler no repo:** `HANDOFF_POS_FASE1_ROTA_A.md` (este) · `HANDOFF_CONTINUACAO_V11.md` (Rota B) · `RESULTADOS_REGIONALIZACAO_V11.md` (Fase 1) · Pedro: `lumina-ssm/.../ABRAOM_RESEARCHER_TRANSFER_REPORT.md`.

*Fim do handoff. Estado em 2026-07-11: Fase 0 + Fase 1 fechadas; no fork A×B. Este doc detalha a Rota A (downstream). O código da integração v11 e da Fase 1 está pronto; a Rota A precisa dos edits de launcher (§9.1) + adaptação do eval (§9.5).*
