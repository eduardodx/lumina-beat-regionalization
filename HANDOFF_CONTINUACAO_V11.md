# HANDOFF — Regionalização Beat-v11 (continuação em outra máquina / novo chat)

> **Para quem pega este documento:** ele é **auto-contido**. Cole inteiro no novo chat para
> contextualizar. Cobre a missão, os 4 repositórios, o estudo do Pedro (baseline v10), a arquitetura
> do Beat-v11, tudo o que construímos, a infra (AWS/SageMaker/S3), os resultados até aqui, o
> experimento em andamento, o plano de ação completo, os gotchas e as lições. Datado **2026-07-11**.
>
> **Não há memória compartilhada nem histórico de chat na outra máquina** — tudo o que importa está aqui.

---

## 0. TL;DR — onde estamos em 3 frases

1. Estamos **portando o estudo de regionalização ABRAOM do Pedro** (validado no **Beat-v10**) para o novo **Beat-v11**, no repo `lumina-beat-regionalization`.
2. **Fase 0 (integração v11) e Fase 1 (adapters de frequência) estão FECHADAS e validadas.** Veredito da Fase 1: **há sinal sequência→frequência real no v11, mas NÃO é brasileiro-específico** (A_BR ≈ A_gnomAD ≫ A_scrambled). Isso reproduz e explica o caveat científico do Pedro.
3. **Agora estamos no experimento B (teste de resíduo)** — o teste direto de sinal regional, que decide se o *fusion* (Fase 3) precisa ser reestruturado ou pode seguir o caminho antigo do Pedro. **O código está pronto e verificado; falta lançar 2 jobs** (ver §8 e §9).

---

## 1. A missão (o "porquê" de tudo)

**Regionalização** = melhorar a interpretação de patogenicidade de variantes (ClinVar) no **contexto brasileiro/latino-americano**, usando o **ABRAOM** (base genômica brasileira — coorte de idosos de São Paulo, estudo SABE). O ABRAOM dá **frequência alélica (AF) por população brasileira**.

**O achado que motiva a frente:** em benchmarks de foundation models de DNA, variantes brasileiras são o **ponto cego estrutural** — modelos vão mal as-shipped, e adaptação leve mal move. A hipótese: o gargalo é o **prior/atalho de frequência** (a AF de referência é majoritariamente europeia — gnomAD NFE), que penaliza variantes comuns em populações não-europeias. Calibrar a AF por população (ABRAOM/AMR) deveria ajudar.

**O Pedro já atacou isso no Beat-v10** e o resultado foi **positivo mas cientificamente nuançado** (ver §3). **Nossa missão:** portar o pipeline dele pro **Beat-v11** (backbone melhor), validar se os indícios se sustentam, e **adaptar o acoplamento à nova arquitetura**.

---

## 2. Os repositórios (o que é cada um)

Há **4 repos** relevantes. Todos são de `github.com/eduardodx/`.

### 2.1 `lumina-beat-regionalization` — **NOSSO repo de trabalho** (onde tudo acontece agora)
- **Remote:** `github.com/eduardodx/lumina-beat-regionalization`. Branch `main`. **Já está commitado/pushado** (a outra máquina clona daqui).
- **Como foi montado:** o gestor criou o repo vazio. Nós copiamos a árvore git-tracked do `lumina-ssm@abraom-regionalization-study` (o pipeline do Pedro, 764 arqs, 2.6 MB) + **vendorizamos o pacote `lumina_beat_v11/`** (o modelo standalone, do `lumina-beat@main`) na raiz.
- **É aqui que você trabalha.** Contém: o pipeline do Pedro (`eval/clinvar/`, `scripts/`, `src/models/`, `data/`, `tests/`), o pacote do modelo v11 (`lumina_beat_v11/`), e **nossas adições** (§5).

### 2.2 `lumina-ssm` — repo core de modelos/treino do Eduardo (a BASELINE do Pedro)
- **Remote:** `github.com/eduardodx/lumina-ssm`. Branch **`abraom-regionalization-study`** = o estudo do Pedro no v10.
- **Papel:** é de onde copiamos o pipeline. Serve como **referência da baseline v10** — os artefatos do Pedro (`artifacts/abraom_frequency_adapter/*/summary.json`, `artifacts/clinvar_regional_eval/...`) e os relatórios dele estão aqui. Use pra `diff` (baseline Pedro × nosso) e pra ler os números do v10.
- Tem `src/models/` com registry (beat_v2…beat_v10, bimamba3) mas **NÃO tem beat_v11** — por isso vendorizamos o pacote no nosso repo.

### 2.3 `lumina-beat` — repo dos foundation models Beat
- **Remote:** `github.com/eduardodx/lumina-beat`. Branch `main`.
- **Papel:** contém **`beat-v11/lumina_beat_v11/`** = o pacote standalone do modelo r1 (foi o que vendorizamos). Também: `beat-v11/PLANO_ACAO_REGIONALIZACAO_V11.md` (o plano de ação original), `beat-v11/model.py` etc. E `beat-v11/mosaic_eval/` = **outra frente** (benchmark Mosaic — não é esta).
- Commit que casa com o checkpoint r1: `main@3d15e66`.

### 2.4 `lumina-benchmarks-mosaic-eval` — benchmark do gestor (CONTEXTO, não é esta frente)
- Benchmark two-lane do Eduardo (7 FMs + especialistas). É **outra frente** (avaliação Mosaic). Relevante só como contexto: o `Beat-minus-AF` dele mostrou que o probe treinado quase não depende de AF no S9 brasileiro — coerente com nossos achados. Não mexa aqui.

---

## 3. O estudo do Pedro no Beat-v10 (a baseline que estamos portando)

O Pedro construiu um pipeline completo de regionalização no v10. **Docs dele** (em `lumina-ssm@abraom-regionalization-study`): `artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md` (completo, 738 linhas) + a versão compacta.

### 3.1 A escada de modelos (M0→M7)
- **M0** = baseline molecular (ClinVar non-BR, sem ABRAOM). A patogenicidade "pura".
- **A_BR / A_gnomAD / A_scrambled** = **adapters de frequência** (LoRA sobre o backbone congelado que predizem AF a partir da sequência). A_BR treina na AF brasileira (ABRAOM), A_gnomAD na global, A_scrambled é o controle negativo (AF embaralhada).
- **M4** = static fusion (M0 + adapters via gate, sem frequência explícita forte).
- **M5 / M6** = fusion + frequência explícita (bruto). **Reduziram falso-positivo mas SUPRIMIRAM P/LP** comuns no ABRAOM (founder/recessivo brasileiro) — colapso de recall.
- **M5_v2_calibrated** = desconto de frequência limitado.
- **M5_v3_safety = candidato final:** desconto regional limitado + **guarda molecular** (a frequência pode reduzir o score mas NÃO apagar evidência molecular forte). Protege os P/LP founder/recessivos.
- **M7_scrambled / M2_gnomad_only** = controles.

### 3.2 O resultado headline e o caveat (CRÍTICO entender)
- **Headline:** M5_v3_safety sobe `br_only` MCC **0.279 → 0.605** e specificity de benignas-comuns-ABRAOM **0.803 → 0.959**, sem degradar o global (MCC 0.512). **A calibração de AF regional funciona operacionalmente.**
- **O caveat honesto (o valor real do report):** os **controles negativos estratificados** (que preservam gene/AF-bin/tipo/specificity/cromossomo e quebram só o link variante↔frequência) **chegam MUITO perto do real** → a **especificidade biológica do ABRAOM NÃO está falsificada**. Parte do ganho pode ser estrutura genérica de frequência, não biologia brasileira.
- **Veredicto do Pedro:** `do_not_train_next` — o próximo passo (no v10) era **curadoria** de 75 variantes críticas + validação externa, NÃO mais treino. Isso está **gated** (precisa de mais dados/curadoria).

### 3.3 Por que portamos pro v11 (a decisão, via Pedro)
Como a curadoria está gated, a direção acordada foi: **pegar os indícios do v10 e validá-los no Beat-v11** (backbone melhor + contexto mais longo), **adaptando o acoplamento à nova arquitetura**. A pergunta científica: *o v11 consegue achar o sinal regional que o v10 não achou de forma limpa?*

### 3.4 O que é reusável vs re-treinável no porte
- **Reusar as-is (model-agnostic):** datasets (ClinVar×ABRAOM, adapter de frequência), slices, eval, calibração, controles negativos, o mecanismo de fusion (`fusion_lora.py`).
- **Re-treinar no v11 (acoplado ao backbone):** M0/A_path, os adapters de frequência (A_BR/A_gnomAD/A_scrambled), o gate do fusion.

---

## 4. A arquitetura do Beat-v11 (fatos VERIFICADOS ao vivo — NÃO confie nos defaults/doc)

> **Lição transversal desta frente (repetida 4×): nesta base, os defaults e a doc MENTEM. Sempre leia o artefato/o modelo carregado.** Exemplos abaixo.

- **Checkpoint alvo (r1):** `s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt` (SISO, **52.1M params**). `config['model']='beat_v11_bioprime'`, 74 chaves de config. Casa com `lumina-beat@main` commit `3d15e66`, pacote `beat-v11/lumina_beat_v11`.
- **`d_full = 448`** (`d_model=384` + `d_pure=64`), mid = 384. **NÃO é 320** — os defaults do dataclass `BeatV11Config` dizem 256/320, mas o checkpoint r1 **sobrescreve**. Sempre ler `model.cfg`/`model.full_hidden_dim`.
- **MoE OFF no r1** (`moe_enabled=False`, **0 lineares MoE ao vivo**). A arquitetura suporta MoE, mas ESTE checkpoint não usa. (Por isso a exclusão de `.router`/`.experts.` no LoRA é no-op/salvaguarda.)
- **`encode()` renomeou as chaves:** v11 retorna `{"last_hidden_state", "mid_hidden_state"}`; o v10 retornava `{"hidden_states", "mid_hidden_states"}`. Isso quebrava o extractor do Pedro (`_extract_lumina_hidden`) — fix no nosso adapter.
- **Cabeças nativas de população:** `population_af_head → gnomad_af_pred[B,L,4]` e `population_observed_head`. Ou seja, **o v11 já prediz AF gnomAD nativamente por posição** — relevante pro experimento B e pro fusion.
- **Loader:** `lumina_beat_v11.load_model_from_checkpoint(checkpoint_path, device=..., strict=True)` — trata download S3, config, strip de prefixo. Aceita `s3://` direto. **Não use o registry `build_registered_model`** — ele rejeita as 74 chaves de config e faz `load_state_dict` strict sem strip.
- **Vocab idêntico ao `src.constants`** (PAD=0, MASK=6, UNK=7, VOCAB_SIZE=8, SNV_BASES=ACGT) → a tokenização do pipeline funciona no v11.
- **Superfície LoRA @ rank 8:** v11 = **105 lineares / 1.248M params** (mamba `in/out_proj` fwd/bwd + attn `q/k/v/out_proj` + `up_stages.*.gate` + `stem.purity`; **zero cabeças**). v10 era 53 / 394k (v11 é mais largo). Mesmos TIPOS de módulo → paridade real de superfície.
- **Gotcha de ambiente (tilelang):** o env conda do notebook SageMaker tem um `tilelang/tvm` quebrado que aborta no import (`tvm_ffi AttributeError`, Py3.12); `mamba_ssm` importa tilelang no import → crasha antes de qualquer shim por-script. **Fix (já commitado):** shim no topo de `src/models/__init__.py` que faz `sys.modules["tilelang"]=None` (força fallback triton; r1 é SISO então tilelang é inútil). No **job** SageMaker isso é no-op (`INSTALL_TILELANG=0`). Alternativa: `pip uninstall tilelang` no notebook.

---

## 5. O que CONSTRUÍMOS (a integração v11) — arquivos e mudanças

Tudo isto está no repo `lumina-beat-regionalization` (verificado por `diff` contra a baseline do Pedro — mudanças limpas, sem edições parasitas).

### 5.1 Arquivos NOVOS (nossos)
| Arquivo | O que é |
|---|---|
| `eval/clinvar/beat_v11_adapter.py` | **`FineTuneBeatV11Adapter`** — carrega o v11 via `lumina_beat_v11.load_model_from_checkpoint` (não pelo registry); extrai `last_hidden_state` (fix da chave v11); shim tilelang. Implementa o protocolo `FineTuneAdapter` (d_model, backbone, tokenize, forward_hidden_states, nuc_window_to_token_bounds, extract_variant_features). |
| `scripts/smoke_beat_v11_regionalization.py` | Smoke de Fase 0: forward parity (448/384/4/4) + `apply_lora` (checa que nenhuma cabeça/MoE é envolvida). |
| `scripts/smoke_beat_v11_freq_train.py` | Smoke de acoplamento de treino (sintético, sem FASTA/dataset): factory→adapter→apply_lora→extract_variant_features→head→backward→step. |
| `scripts/compare_freq_adapters.py` | **Bootstrap PAREADO** de spearman entre runs (lê `{val,test}_predictions.parquet`, inner-join por variant_id, CI da diferença). É a ferramenta de análise dos adapters. |
| `RESULTADOS_REGIONALIZACAO_V11.md` | Doc de resultados da Fase 1 (molde dos `RESULTADOS_*_MOSAIC.md`). |
| `lumina_beat_v11/` | O pacote standalone do modelo v11 (vendorizado; `model.py`, `checkpoint.py`, `tokenizer.py`, `constants.py`, etc.). |

### 5.2 Edições em arquivos do Pedro (mínimas, back-compat)
| Arquivo | Mudança |
|---|---|
| `eval/clinvar/adapters.py` | Família **`beat-v11`** no `build_finetune_adapter` (roteia pro `FineTuneBeatV11Adapter(checkpoint_path, device)`) + aliases no `normalize_finetune_model_family`. |
| `eval/clinvar/lora.py` | `_EXCLUDE_PATTERNS` += cabeças v11 (`population_af_head`, `population_observed_head`, `conservation_scalar_head`, `conservation_bin_head`, `missense_severity_head`, `hic_*`, `cell_film`) + `.router`/`.experts.` (MoE congelada = paridade v10; no-op no r1). |
| `src/models/__init__.py` | Shim tilelang no topo (ver §4). |
| `scripts/train_abraom_frequency_adapter.py` | **(1)** `--loss {bce,mse,huber}` (default bce = back-compat; mse/huber pra alvo de regressão como `delta_logit`). **(2)** Threading de `delta_logit` e `scrambled_delta_logit` como alvos válidos (5 pontos: `FREQUENCY_COLUMNS`, dataclass `FrequencyExample`, construção do exemplo, `_value_for_column`, whitelist dos 2 args `--target-column`/`--metric-target-column`). |
| `scripts/sagemaker_abraom_frequency_adapter.py` | Só ajuste de comentário (o `INSTALL_TILELANG=0` vale pro r1 também). |

### 5.3 Validações (TODAS passaram)
- Forward do r1: `last_hidden_state[...,448]`, `mid[...,384]`, `mlm_logits[...,4]`, `gnomad_af_pred[...,4]` ✓.
- `apply_lora`: 105 lineares, nenhuma cabeça/MoE vazada ✓.
- Smoke de treino sintético: backward chega em LoRA+head, otimizador move a loss ✓.
- Job SageMaker real (ctx 4096, dados reais) completou ✓.
- Fase 1 completa (§7).

---

## 6. Infra (AWS / SageMaker / S3) — como lançar, monitorar, ler

- **Conta AWS:** `085188779747`, região **us-east-2**.
- **Windows/local NÃO tem S3/GPU** — todo treino/fetch é no **SageMaker notebook** (`~/testeArq/lumina-beat-regionalization`). O código é sincronizado Windows↔notebook (via git ou sync). **Na outra máquina, clone o repo do GitHub.**

### 6.1 S3 paths importantes
- Checkpoint r1: `s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt`
- Dataset adapter de frequência: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/abraom_frequency_adapter/`
- Referência hg38: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/hg38/`
- Artefatos dos nossos jobs: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/<experiment>/sagemaker-artifacts/<JOB_NAME>/output/model.tar.gz`

### 6.2 Como lançar um job de adapter de frequência (o launcher tem passthrough)
```bash
python scripts/sagemaker_abraom_frequency_adapter.py \
  --experiment <nome-curto> \
  --checkpoint-s3-prefix s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/ \
  --detach -- \
  --model-family beat-v11 --model-version r1 \
  <ARGS-DO-TRAINER>
```
- Tudo **depois do `--`** vai pro `train_abraom_frequency_adapter.py` (passthrough via `split_cli_args`→`REMAINDER`).
- Os canais `frequency` (dataset ABRAOM) e `reference` (hg38) já têm defaults S3 corretos → **o job não precisa de nada local** (o container tem pyfaidx).
- **`--model-version r1`** é importante (senão o `summary.json` grava `beat-v10` por default — cosmético mas polui o registro).

### 6.3 Como monitorar e ler resultados
```bash
# status (o nome real do job tem timestamp anexado ao base name)
aws sagemaker list-training-jobs --name-contains <experiment> \
  --query 'TrainingJobSummaries[].[TrainingJobName,TrainingJobStatus]' --output table

# motivo de falha
aws sagemaker describe-training-job --training-job-name <JOB> --query FailureReason --output text

# log real (erro Python fica aqui, não no describe)
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <JOB> --since 6h \
  | grep -iE 'train_step|error|traceback|invalid choice' | head

# baixar + ler o summary.json do artefato
mkdir -p ~/v11eval/<name> && cd ~/v11eval
aws s3 cp s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/<experiment>/sagemaker-artifacts/<JOB>/output/model.tar.gz <name>/m.tar.gz
tar -xzf <name>/m.tar.gz -C <name>
python -c "import json,glob; d=json.load(open(glob.glob('<name>/**/summary.json',recursive=True)[0])); print(d['config']); print(d.get('best_step')); [print(s,(d.get(s) or {}).get('overall',{}).get('model_spearman')) for s in ['val_metrics','test_metrics']]"
```

### 6.4 Gotchas de infra (custaram tempo — leia)
- **Quota:** só **1 instância `ml.g5.2xlarge` de treino concorrente**. A `ml.g5.4xlarge` tem pool próprio → use ela pra rodar um 2º job em paralelo.
- **Nome do job:** o launcher imprime `job_name=<base>`, mas o SageMaker **anexa timestamp** → o nome real é `<base>-<YYYYMMDDHHMMSS>`. `describe-training-job` com o base name dá "resource not found". Ache com `list-training-jobs --name-contains <base>`.
- **`/tmp` do notebook é efêmero** (checkpoint sumiu num restart) → guarde checkpoints em `~/ckpts/`.
- **Idle-shutdown mata run longo, e `nohup` NÃO protege** (só contra SIGHUP, não contra a instância desligar) → prefira **jobs SageMaker destacados** (`--detach`). Se rodar direto no notebook, `nohup python -u` (o `-u` porque nohup bufferiza stdout).
- **Receita real do Pedro ≠ defaults do trainer.** Os defaults (10k steps, ctx 4096, dataset inteiro) dariam **~69h** e não seriam comparáveis. A receita do Pedro (verificada nos `summary.json` dele) é: `--max-steps 1000 --context-size 1024 --max-train-rows 100000 --max-val-rows 5000 --max-test-rows 5000 --row-sample-strategy balanced-af` (resto default: rank 8, lr_lora 5e-6, lr_head 5e-4, batch 2, grad_accum 8, `use_gnomad_prior=False` = sequence-only). **~1-2h por job.**

---

## 7. RESULTADOS da Fase 1 (o veredito — FECHADA)

Os 4 adapters de frequência, todos com a receita do Pedro (ctx 1024, 1000 steps, balanced-af, sequence-only). Métrica honesta = **Spearman** (rank; Pearson é trivial aqui porque A_BR treina na escala do af_abraom). Doc completo: `RESULTADOS_REGIONALIZACAO_V11.md`.

### 7.1 Tabela v11 × v10 (Spearman, val / test)
| Adapter | v10 (val/test) | **v11 (val/test)** |
|---|---|---|
| **A_BR** (`af_abraom`) | 0.114 / 0.104 | **0.135 / 0.127** |
| **A_gnomAD** (`af_gnomad`→mede `af_abraom`) | 0.107 / 0.118 | **0.120 / 0.118** |
| **A_scrambled** (controle−) | −0.013 / 0.001 | **−0.036 / −0.021** |
| A_BR @ ctx 4096 (ablação) | — | 0.129 / 0.127 |

### 7.2 Os testes pareados (bootstrap, `compare_freq_adapters.py`, B=2000, n≈5000)
| Comparação | val: Δ [CI] P | test: Δ [CI] P | veredito |
|---|---|---|---|
| **A_BR − A_scrambled** | +0.172 [+0.131,+0.213] **1.000** | +0.149 [+0.105,+0.191] **1.000** | **sinal real** (CI ≫ 0) |
| **A_BR − A_gnomAD** | +0.016 [−0.005,+0.036] 0.93 | +0.010 [−0.018,+0.039] 0.73 | **não-significativo** (CI cruza 0) |

### 7.3 Veredito da Fase 1
> **A_BR ≈ A_gnomAD ≫ A_scrambled** — há sinal sequência→frequência **real** no v11 (P=1.000 acima do piso), mas ele **NÃO é brasileiro-específico** (A_BR e A_gnomAD são estatisticamente indistinguíveis).

Leituras importantes:
1. **Paridade reproduzida e superada:** v11 +~20% sobre v10 em Spearman — mas é melhora **genérica** (backbone melhor aprende AF melhor), não sinal regional.
2. **Contexto 4096 não compra nada** e é de graça (overhead-bound) → AF é sinal **local**; usar ctx 1024 em tudo.
3. **Isto REPRODUZ e EXPLICA o caveat do Pedro:** o sinal regional nunca esteve forte no adapter, desde a origem → por isso os controles negativos dele chegavam perto do real downstream. O ganho do M5_v3 (br_only 0.28→0.61) é **calibração de frequência**, não biologia brasileira aprendida da sequência.
4. **Não é um fracasso** — "há sinal seq→AF real, mas não regional-específico" é uma conclusão completa e honesta que **de-risca** o quanto investir no downstream.

---

## 8. O experimento B em andamento (teste de RESÍDUO — decide a estrutura do fusion)

**Por que B:** o A_BR-vs-A_gnomAD foi um teste **indireto** e de baixa potência (deu null). O **resíduo** é o teste **direto**: treinar um adapter para predizer `delta_logit = logit(af_abraom) − logit(af_gnomad)` — o componente que é ABRAOM *além* do gnomAD. É a **melhor chance** do sinal regional (adapter dedicado ao componente regional).

**A pergunta que B responde (o que o Gabriel pediu):** *precisamos MUDAR a estrutura do fusion, ou continuar no caminho antigo do Pedro?*
- **B mostra sinal (A_residual ≫ scrambled, CI exclui 0)** → há sinal regional que o fusion deve consumir → **RESTRUTURAR o fusion** (input regional dedicado; melhora v11-genuína).
- **B ≈ scrambled (CI cruza 0)** → não há sinal regional em lugar nenhum → **manter o fusion do Pedro** (A_BR + calibração = caminho antigo); conclusão honesta = "regionalização é calibração de AF, não biologia aprendida".

**Expectativa calibrada:** dado o null do A_BR-vs-A_gnomAD, aposto no cenário de baixo (≈ scrambled). Mas B é o teste que **fecha** a pergunta.

### 8.1 O que já foi feito (código pronto e verificado)
- O dataset **já tem** as colunas `delta_logit` e `scrambled_delta_logit` (o prep do Pedro computa: `delta_logit = logit_af_abraom − logit_af_gnomad`; `scrambled_delta_logit = logit(af_abraom_embaralhado_no_split) − logit_af_gnomad`).
- O trainer usava só `binary_cross_entropy_with_logits` (alvo tem que ser [0,1]); `delta_logit` é logit ilimitado → **adicionamos `--loss huber`** e **threadamos `delta_logit`/`scrambled_delta_logit`** como alvos válidos (§5.2). **Verificado via diff — mudanças limpas.**
- **Verificado:** o `build_metrics` NÃO clipa o alvo (só a predição, que já é sigmoid∈[0,1]) → o `model_spearman` do `summary.json` sai **correto** mesmo pro alvo de resíduo (Spearman é rank-based; só Brier/NLL ficam sem sentido — ignorar).

### 8.2 CHECAGEM antes de lançar (risco de KeyError)
O trainer agora lê `delta_logit`/`scrambled_delta_logit` do parquet. Confirme que o dataset deployado tem essas colunas — **rode no SageMaker notebook** (onde o dataset local vive; o laptop clonado do GitHub NÃO tem `data/datasets/`, que é gitignored):
```bash
python -c "import pyarrow.parquet as pq; c=pq.read_schema('data/datasets/abraom_frequency_adapter/abraom_frequency_val.parquet').names; print('TEM:', [x for x in c if x in ('delta_logit','scrambled_delta_logit')])"
```
Se sair `['delta_logit','scrambled_delta_logit']` → pode lançar. Se faltar → o dataset deployado é de um prep antigo; regenerar com `prepare_abraom_frequency_adapter_dataset.py` ou investigar.

### 8.3 Os 2 jobs a lançar
```bash
# A_residual — treina no resíduo regional
python scripts/sagemaker_abraom_frequency_adapter.py \
  --experiment abraom-freq-v11-r1-residual \
  --checkpoint-s3-prefix s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/ \
  --detach -- \
  --model-family beat-v11 --model-version r1 \
  --target-column delta_logit --metric-target-column delta_logit --loss huber \
  --max-steps 1000 --context-size 1024 \
  --max-train-rows 100000 --max-val-rows 5000 --max-test-rows 5000 \
  --row-sample-strategy balanced-af

# A_residual_scrambled — o piso (treina no resíduo embaralhado, avalia no real)
python scripts/sagemaker_abraom_frequency_adapter.py \
  --experiment abraom-freq-v11-r1-residual-scr \
  --checkpoint-s3-prefix s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/ \
  --detach -- \
  --model-family beat-v11 --model-version r1 \
  --target-column scrambled_delta_logit --metric-target-column delta_logit --loss huber \
  --max-steps 1000 --context-size 1024 \
  --max-train-rows 100000 --max-val-rows 5000 --max-test-rows 5000 \
  --row-sample-strategy balanced-af
```
(Quota: 1 `g5.2xlarge` → um de cada vez, ou o 2º em `--instance-type ml.g5.4xlarge`.)

### 8.4 Como analisar (quando os 2 caírem)
Baixe os artefatos (como em §6.3) para `~/v11eval/residual` e `~/v11eval/residual_scr`, e rode:
```bash
python scripts/compare_freq_adapters.py \
  --run A_residual:~/v11eval/residual \
  --run A_residual_scr:~/v11eval/residual_scr \
  --ref A_residual
```

### 8.5 ⚠️ Heads-up de interpretação (não é bug)
O controle `scrambled_delta_logit` **não vai dar piso ~0** como o `scrambled_af_abraom` deu. Motivo: ele embaralha só o abraom mas **mantém o `−logit_gnomad` real** — e o modelo consegue prever o componente gnomad da sequência. Então A_residual_scrambled terá spearman não-nulo vs `delta_logit`. **Isso é esperado.** A leitura certa é o **gap pareado A_residual − A_residual_scrambled** (que é o que o `compare_freq_adapters.py` calcula): CI excluindo 0 → sinal regional; CI cruzando 0 → sem sinal regional.

### 8.6 Histórico (o que já deu errado no B, pra não repetir)
- 1ª tentativa falhou com `argument --target-column: invalid choice: 'delta_logit'` — o `--target-column` tinha whitelist de 3 valores e `delta_logit` nem era carregado. **Já corrigido** (threading dos 5 pontos). Se falhar de novo por isso, é sync: o trainer editado precisa estar no notebook (o launcher empacota o working-tree).

---

## 9. O PLANO DE AÇÃO completo (todas as fases)

| Fase | O que é | Estado |
|---|---|---|
| **0** · Integração v11 | Adapter, factory, apply_lora, forward parity | ✅ **completa** |
| **0.5** · Wiring | Família `beat-v11` no factory + launcher de frequency (tem passthrough) | ✅ (frequency) · ⬜ launchers de M0/fusion |
| **1** · Adapters de frequência | A_BR/A_gnomAD/A_scrambled no v11 + análise | ✅ **fechada** (null regional; §7) |
| **B** · Teste de resíduo | Decide a estrutura do fusion | 🔄 **código pronto; falta lançar 2 jobs** (§8) |
| **2** · M0 = A_path | Baseline molecular (nonBR-only ClinVar) no v11 | ⬜ (exige editar `clinvar_m0_job.py:52`) |
| **3** · Fusion M4/M5 | Static + dynamic-gated no v11 | ⬜ (exige editar `clinvar_fusion_job.py:91`; estrutura decidida pelo B) |
| **4** · Calibração | Re-fitar M5_v2/M5_v3 (guarda molecular) nos scores do v11 | ⬜ (scripts model-agnostic; recomputar thresholds) |
| **5** · Falsificação + head-to-head v11×v10 | Controles negativos estratificados + o entregável científico | ⬜ |

### 9.1 A bifurcação de escopo (decisão de gestão — do Eduardo, não do dev)
Depois da Fase 1 dar null-regional, a direção deixou de ser óbvia:
- **(A) Completar o porte (Fases 2-3):** reproduzir M0+fusion+M5_v3 no v11. Testa se o **ganho de calibração** reproduz (provavelmente sim — não depende de sinal regional no adapter). Resultado confirmatório rigoroso "v11 reproduz a conclusão nuançada do v10".
- **(B, em andamento) Teste de resíduo:** o teste direto do componente regional. **O Gabriel escolheu fazer B primeiro** justamente porque ele decide se o fusion precisa mudar de estrutura (A) ou não.

**Importante:** o null no adapter **NÃO cancela** as Fases 2-3. O ganho do M5_v3 é via **calibração de AF**, que independe do adapter ter aprendido sinal regional. Então reproduzir o downstream no v11 ainda vale.

### 9.2 O que falta tecnicamente pras Fases 2-3
`clinvar_m0_job.py:52` e `clinvar_fusion_job.py:91` usam `_upsert_arg` que **FORÇA** `--model-family lumina --model-version beat-v10` (sem passthrough, ao contrário do launcher de frequency). Esses **precisam de edição real** (parametrizar pra aceitar `beat-v11` + o checkpoint r1) quando chegarmos no M0/fusion. Ainda não lemos o `evaluate_abraom_regionalization.py`.

### 9.3 Refinação v11-nativa (adiada de propósito, mas promissora)
O v11 tem a cabeça `population_af_head` (prediz AF gnomAD nativamente). A formulação natural do resíduo seria treinar o adapter no `af_abraom − f(gnomad_af_pred)` usando essa cabeça — isola o componente regional de forma mais limpa. O experimento B (com `delta_logit` da coluna do dataset) é a versão barata disso. Se B der sinal, a rota nativa é o próximo passo.

---

## 10. Lições e princípios desta frente (aplicar sempre)

1. **NUNCA confie em default/doc nesta base — leia o artefato.** Aconteceu 4×: `d_full` era 448 não 320; "r1 ganhou MoE" mas `moe_enabled=False`; "defaults = receita do Pedro" mas o Pedro sobrescreveu 6 params; e o `--target-column` tinha whitelist que rejeitava `delta_logit`. Sempre: leia o `config` do checkpoint, o `summary.json`, o schema do parquet, o modelo carregado.
2. **Comparações pareadas + bootstrap, não pontos.** Um Δ de +0.02 spearman parece sinal mas cruza 0 no bootstrap pareado. O `compare_freq_adapters.py` é a ferramenta.
3. **Spearman é a métrica honesta** (Pearson favorece trivialmente quem treina na escala do alvo; delta_nll é inutilizável — o baseline gnomad oscila entre splits).
4. **Ao adicionar um alvo de treino, verifique os 5 pontos:** whitelist argparse + `FREQUENCY_COLUMNS` (carga) + dataclass `FrequencyExample` + construção do exemplo + `_value_for_column`. (Eu errei isso uma vez — verifiquei só a loss, não o resto do caminho do alvo.)
5. **O controle negativo é o que dá sentido ao resultado** — sem o scrambled, "spearman 0.13" não significa nada. Sempre rode o controle.

---

## 11. ESTADO DO CÓDIGO + como continuar na outra máquina

- **O repo `lumina-beat-regionalization` está no GitHub** (`github.com/eduardodx/lumina-beat-regionalization`, branch `main`). **Na outra máquina: `git clone` e pronto.**
- ⚠️ **ANTES DE VIAJAR: commite + pushe os 2 arquivos pendentes** (a edição do threading `delta_logit` no `train_abraom_frequency_adapter.py` + este handoff). Rode:
  ```bash
  cd lumina-beat-regionalization
  git add -A && git commit -m "B: delta_logit residual target + continuation handoff" && git push
  ```
- **O que NÃO está no git** (recupere do S3 quando precisar): datasets (`data/datasets/**` são grandes/gitignored — vêm dos canais S3 no job), checkpoints, os `model.tar.gz` dos artefatos. Pro trabalho no notebook você **não precisa deles localmente** — os jobs SageMaker montam tudo do S3.
- **Ambiente:** o trabalho de treino é no **SageMaker notebook** (`~/testeArq/lumina-beat-regionalization`), não no Windows/laptop. Na outra máquina você **edita o código** (clone do GitHub) e **sincroniza pro notebook** (git push → git pull no notebook, ou o sync que já usam). O notebook tem o env "modelo" (conda: torch/mamba_ssm/GPU) + `pyfaidx`.
- **Para lançar jobs:** precisa das credenciais AWS no ambiente (conta `085188779747`, us-east-2). Isso já está configurado no notebook.

---

## 12. Referência rápida (cole no novo chat junto com este doc)

**Checkpoint r1:** `s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt` (SISO, 52M, d_full=448, MoE off).

**Jobs da Fase 1 (todos Completed):**
- `abraom-freq-v11-r1-abr-df0483-20260710010459` (A_BR, ctx1024) — spearman 0.135/0.127
- `abraom-freq-v11-r1-abr-ctx4096-5d9db4-20260710012147` (A_BR ctx4096) — 0.129/0.127
- `abraom-freq-v11-r1-gnomad-9f35bc-20260710124718` (A_gnomAD) — 0.120/0.118
- `abraom-freq-v11-r1-scrambled-52e9fb-20260710174346` (A_scrambled) — −0.036/−0.021

**Receita padrão de um adapter de frequência:** `--max-steps 1000 --context-size 1024 --max-train-rows 100000 --max-val-rows 5000 --max-test-rows 5000 --row-sample-strategy balanced-af` + (`--model-family beat-v11 --model-version r1`).

**Próxima ação concreta:** (1) `git push` do pendente; (2) na outra máquina, clone; (3) sincronizar pro notebook; (4) rodar a checagem do parquet (§8.2); (5) lançar os 2 jobs do experimento B (§8.3); (6) analisar com `compare_freq_adapters.py` (§8.4); (7) o gap A_residual−A_residual_scrambled decide a estrutura do fusion (Fase 3).

**Documentos-chave a ler no repo:**
- `HANDOFF_CONTINUACAO_V11.md` (este)
- `RESULTADOS_REGIONALIZACAO_V11.md` (resultados Fase 1)
- `lumina-beat/beat-v11/PLANO_ACAO_REGIONALIZACAO_V11.md` (plano original + deltas de arquitetura v10→v11)
- Baseline Pedro: `lumina-ssm/artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md`

*Fim do handoff. Estado em 2026-07-11: Fase 0 + Fase 1 fechadas; experimento B (resíduo) com código pronto e verificado, aguardando lançamento dos 2 jobs.*
