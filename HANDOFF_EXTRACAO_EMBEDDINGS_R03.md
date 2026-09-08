# HANDOFF — Refinar a extração de embeddings do Lumina R03

> **Para quem pega num chat novo: este doc é auto-contido.** Leia inteiro antes de tocar em código.
> Datado **2026-09-08**. Autor: Gabriel (dev, TCC). Gestor: Eduardo.
> Branch: **`embedding-probe-mosaic`**. Último commit: `5035304`.
>
> Esta frente **sucede** a sonda de embedding (`HANDOFF_SONDA_EMBEDDING_VARIANTE_MOSAIC.md`), que já
> foi concluída e reportada ao Eduardo. Os achados da sonda são **premissa** do que se faz aqui.

---

## 1. A missão

O Eduardo pediu (08/09, WhatsApp), depois de ler o relatório da sonda:

> *"dado esse relatório que você fez, investiga e propõe uma forma de embeddings para o Lumina, para
> treinarmos modelos com esses embeddings. eu fiz aqui e rodei o Mosaic e gostaria de discutir com
> você, mas não queria já te entregar o que eu fiz para não perder a graça de você aprofundar nisso
> e também para eu ter sua opinião com menos viés."*

O Gabriel decidiu: **não basta propor — vamos testar** o que for testável, para entregar uma
proposta sem suposições. Foco no **R03** (a versão mais recente do beat).

> ⚠️ **Nota de método (do próprio Eduardo):** ele montou a proposta dele em paralelo e preferiu não
> mostrar antes, para não enviesar. Quando comparar, registrar **o que cada um decidiu e por quê**
> ANTES de olhar resultado.

---

## 2. Onde estamos — plano R0→R4

| Fase | O que é | Estado |
|---|---|---|
| **R0** | Harness de avaliação sob o protocolo do Mosaic | ✅ fechado e validado |
| **R1** | Extração rica: uma passada de GPU, tudo de uma vez | ✅ rodado (10.761 gold, 19 min) |
| **R2** | Ablações offline sobre as features | 🔄 **1ª rodada feita; 3 runs pendentes** |
| **R3** | O que exigir extração nova (16k vs 4k, boost nativo) | ⬜ não iniciado |
| **R4** | Configuração final sob o protocolo completo (gold+consensus) | ⬜ não iniciado |

**O próximo passo concreto está no §7.**

---

## 3. Premissas herdadas da sonda (já medidas, não re-medir)

Da frente anterior, sobre o R03 (52M params, Mamba-3 hourglass, `d_full=448 = 384 h_up + 64 h_pure`):

- **Canal trivial**: `h_pure = Linear(token_emb + pos_emb)` é **pointwise**. Num offset focal fixo,
  `h_pure(ref)` assume **4 valores** no dataset inteiro e `h_pure(Δ)`, **12**. É um one-hot
  disfarçado. Responde por **0,065%** da energia da mudança — real, mas numericamente pequeno.
- **82% da variância do Δ focal é o SÍTIO, 18% o alelo** → o embedding cru é um fingerprint de
  locus; a unidade certa é a diferença `Δ = alt − ref`.
- **Horizontes da arquitetura**: **±25 bp** = alcance da via puramente convolucional (composição dos
  índices: stem k=15 → 2× DownStage k=4/s=2 + refine k=3 → 2× UpStage); **±512 bp** = raio da atenção
  local (`local_attention_window_mid_tokens=256` → radius 128 mid-tokens × 4 bp), **fixo**, não cresce
  com L. Âncoras da atenção global = média de 64 bp, `stride=16`.
- **Propagação**: lei de potência sem corte, mensurável a ≥8 kb, **simétrica** (razão up/down 0,97).
- **Efeito de borda**: inflação de 2,4–2,7× nos bins que encostam na borda da janela.
- **Contexto satura**: de 4 kb para 32 kb a magnitude do Δ cai 1,1% e a direção gira 0,7% (com índice
  focal casado). **4 kb é o ponto eficiente.**
- **Modelo é determinístico** (piso de ruído = 0) *desde que* o batch tenha tamanho fixo. Ver §6.
- **`variant_edit_mask` fica `None`** — liga o boost `+0.5·h_stem` E o caminho de âncoras da atenção
  global. Foi banido na *sonda* por validade de medição; para *extração de features* a lógica não
  transfere automaticamente (é um mecanismo desenhado do modelo). Está no R3 como pergunta aberta.

---

## 4. Achados de arquitetura desta frente (lidos no código, verificados em teste)

### 4.1 Sete das dez cabeças são `nn.Linear` sobre o trunk

```
LINEARES (68 dims): mlm(4) · conservation_scalar(3) · conservation_bin(16) · region(5)
                    counterfactual_snv(4×8=32) · population_af(4) · population_observed(4)
MLP      (10 dims): splice_class(5) · splice_distance(1) · missense_severity(4)
```

Como são lineares, vale **exatamente**:

```
cabeça(alt) − cabeça(ref) = W·(h_alt − h_ref) = W·Δ        (o bias cancela)
```

Ou seja, **projetar o Δ two-tower pelas cabeças treinadas é exato e não custa forward adicional**.
O modelo já faz isso: `conservation_delta_pred = F.linear(edit_delta, conservation_scalar_head.weight)`,
e no R03 está **ligado** (`conservation_delta_head_enabled: True`).

Travado em teste (`test_linear_head_identity_is_exact`), com a contraprova de que **não** vale para
as três MLP (`test_mlp_heads_are_not_in_the_linear_span`).

**Ressalva honesta:** as 68 dims lineares **não são informação nova** (estão no span das 448). São
*direções privilegiadas* — o valor é eficiência amostral, não conteúdo. Um probe **linear** não pode
ganhar com elas; só um não-linear revela. É por isso que existe o probe MLP.

### 4.2 As camadas de atenção local NÃO são chamadas como módulos

```python
if kind == "mamba":   x = layer(x)
elif kind == "local":  x = x + attn(norm(x), None)   # o ModuleDict nunca é invocado
elif kind == "sparse": x = layer(x, edit_mid_mask=...)
```

Um `register_forward_hook` no ModuleDict **jamais dispararia** — o sweep de camada teria quatro
buracos invisíveis, sem erro. Solução implementada: **forward PRE-hooks** (no módulo para
mamba/sparse, no `norm` para as locais), que capturam a *entrada* de cada camada = saída da anterior.
Cobre as 27 fronteiras. Validado contra o traço real de um mid-stack falso, bit a bit.

### 4.3 Os register tokens são legíveis de qualquer camada

16 registers × 384 são prependados no início do mid-stack e removidos só no fim (`x = x[:, n_reg:]`).
Dentro de qualquer camada hookada o tensor é `[B, 16 + L/4, 384]`. O `mid_hidden_state` do `encode()`
vem **sem** eles.

### 4.4 Um token do mid vê ~±16 bp

A base focal alcança os tokens do mid **507..515** (offsets −4..+4) numa janela de 4096 com focal
2047. Não é um token só. Por isso o sweep **agrega os tokens −1..+2** em torno de `f//4`
(`MID_SPAN`), que é onde a resposta se concentra (moda 0).

### 4.5 beat v11 é a mesma família

Mesmo `purity` branch, mesmos kernels, mesmo downsample. `d_full=320` (256+64) → **o canal trivial é
20% das dims lá**, contra 14% no R03. `trunk_final_norm_enabled` é **`False` por default no v11** —
conferir no checkpoint se for usar.

---

## 5. RESULTADOS medidos (todos sob o protocolo do Mosaic, `core_locus`, treino gold-only)

### 5.1 Baselines — o que precisamos superar

| bloco | macro | missense | splice | noncoding |
|---|---:|---:|---:|---:|
| **conservação (phyloP, 4 col)** | **0.8768** | 0.8212 | 0.9711 | 0.8380 |
| gnomAD (frequência) | 0.9090 | 0.9270 | 0.8487 | 0.9513 |
| todos os comparadores (40d) | 0.9801 | 0.9633 | 0.9932 | 0.9837 |

> ⚠️ **CIRCULARIDADE.** Os critérios ACMG **BA1/BS1 usam frequência alélica para classificar como
> benigno** — gnomAD é em parte a *definição* do rótulo, não um preditor independente. REVEL/
> AlphaMissense/CADD foram treinados sobre dados adjacentes ao ClinVar (o próprio README do Mosaic
> avisa). **A barra honesta é a conservação (0.8768)**, que vem de alinhamento entre espécies e nunca
> viu ClinVar.
>
> **E o R03 tem cabeça `gnomad_af_pred` treinada** — nossas features podem herdar a circularidade.
> Por isso as configs `cabecas_sem_gnomad` / `proposta_sem_gnomad` (§7).

### 5.2 Sweep de camada — **minha hipótese estava ERRADA**

Monotônico crescente; **a última camada vence**.

```
L00 (entrada) 0.6107  ·  L13 0.6953  ·  L20 0.7220  ·  L25 0.7559  ·  L26 (saída) 0.7570
```

Eu apostei que camadas intermediárias transfeririam melhor (a última alimenta 10 cabeças e seria
especializada demais). **Não é o caso.** Consequência prática: **não precisamos de hooks na extração
final** — o `mid_hidden_state` do `encode()` basta.

### 5.3 Max-pool ≫ média — **o maior efeito isolado, e não estava na proposta**

```
delta_focal + p512 (média)   0.7579
delta_focal + p512 (MAX)     0.8933      ← +0.135
```

Conversa direto com o perfil da sonda: a razão máximo/média crescia com a distância, ou seja o efeito
tem **picos heterogêneos**. A média os dilui, o máximo os captura. (`pooled(..., reduce="max")` é
`window.abs().amax(dim=1)`.)

### 5.4 Cabeças — ganham, mas não pelo motivo que eu previa

```
so_hidden (Δ focal, 384d)        0.7482
so_cabecas_lineares (Δ, 68d)     0.7049
so_cabecas_mlp (Δ, 10d)          0.6950
so_cabecas_todas (156d)          0.8813   ← salto
hidden_mais_cabecas (540d)       0.8771
```

O salto vem do **`heads_ref`** — os valores **absolutos na referência** (conservação, região,
splice), não dos deltas. É *anotação da posição*, não leitura do efeito da variante. E 0.8813 ≈ os
0.8768 da conservação: **o modelo está reproduzindo phyloP**, que é o que a cabeça dele foi treinada
para fazer.

### 5.5 Contexto vs variante — **pergunta em aberto**

```
so_variante (1152d)              0.7712
so_contexto (768d)               0.8872   ← contexto sozinho > variante sozinha
variante_mais_contexto (1920d)   0.9106
```

Pode ser sinal legítimo ("esta posição é importante", que generaliza) ou memorização de gene.
**Só rodamos `core_locus`. Sem `gene_transfer` não dá para separar.** É o run mais importante pendente.

### 5.6 Demais

```
leitura_trunk        0.7482      leitura_mid          0.7414
leitura_registers    0.7883      ← registers (768d) batem trunk e mid, apesar de responderem só 2,1%
rc_so_direta         0.7482      rc_media  0.7576     rc_concatenado 0.7576   ← RC quase não importa
piso_canal_trivial   0.4736      ← ABAIXO do acaso: confirma que é lookup puro
piso_substituicao    0.5316
infra_atual_pos_norma (896d)     0.8983   ← a extração que JÁ existe é forte
proposta_completa    (2092d)     0.9156   ← melhor
proposta_mais_rc     (2476d)     0.9154
```

### 5.7 A leitura honesta do placar

| | macro | vs nossa melhor |
|---|---:|---:|
| **`proposta_completa`** | **0.9156** | — |
| gnomAD sozinho (circular) | 0.9090 | +0.007 |
| **infra atual** | **0.8983** | **+0.017** |
| conservação (barra honesta) | 0.8768 | +0.039 |
| todos os comparadores | 0.9801 | **−0.065** |

**Batemos a conservação, empatamos com a gnomAD, ganhamos pouco da infra existente, e estamos bem
atrás do conjunto de comparadores.** Não reportar "0.9156" sem este contexto.

### 5.8 Checagens da extração (R1)

```
1. mapeamento full-res→mid: pico em −1..+2, MODA 0 (limite geométrico ±4)      OK
2. registers respondem: ||Δ_reg||/||Δ_focal|| = 0.0209                         fracos, não inertes
3. equivariância RC: cos(fwd, rc) = 0.6430                                     REFERENCIAIS DIFERENTES
4. fp16: erro relativo de quantização = 2.3e−4                                 suficiente
```

**A #3 mudou o desenho**: com cos=0,64 as duas fitas não vivem no mesmo referencial — mediá-las é
somar vetores de bases diferentes. Passamos a guardar o **Δ RC cru** além da média, para a ablação
comparar `fwd` / `média` / `concatenação`. (Resultado: quase não importa, §5.6.)

---

## 6. Infraestrutura construída

### Módulos (`eval/embedding_probe/`)
| Arquivo | O que é |
|---|---|
| `windows.py` | Janelas ref/alt na convenção do Mosaic (`offset = L//2 − 1`), validação estrita |
| `profile.py` | Bins do perfil espacial, com os horizontes ±25/±512 em fronteira de bin |
| `extract.py` | `ProbeModel`: hook pré-RMSNorm, decomposição h_up/h_pure, `PROBE_BATCH_SIZE=4` |
| `stats.py` | Estatística de ranking em **stdlib puro** (sem scipy) |
| `protocol.py` | As regras literais do Mosaic + `assert_no_unit_leak` |
| `rich.py` | `MidStackTaps` (pre-hooks), `head_readouts` (W·Δ), `pooled`, RC, one-hot |

### Scripts (`scripts/`)
| Arquivo | Onde roda | O que faz |
|---|---|---|
| `probe_extract_rich.py` | GPU | **R1**: 46 blocos, 16.492 dims/variante, ~19 min para 10.761 gold |
| `probe_feature_eval.py` | CPU | **R0**: probe ridge sob o protocolo; aceita fatias `bloco[a:b]` |
| `probe_baseline_features.py` | CPU | Baselines dos comparadores do Mosaic |
| `probe_analyze.py` | CPU | Análise da sonda anterior (regenera o relatório dela) |
| `probe_build_manifest.py`, `probe_smoke.py`, `probe_run.py`, `probe_analyze_focal.py`, `probe_batch_diagnostic.py` | — | Da frente da sonda; ainda válidos |

### Config
`configs/probe_feature_configs.json` — **63 configurações** em 11 grupos (chaves iniciadas por `_`
são comentários e o harness as ignora).

### Testes: **129** no total
`windows 13 · profile 12 · stats 25 · protocol 19 · build_manifest 21 · rich 17 · feature_eval 9+`
Os de `rich` e `feature_eval` precisam de torch/numpy (rodam no notebook).

### Decisões de desenho que **não devem ser revertidas sem motivo**

1. **Probe ridge, não regressão logística.** Vamos comparar 27 camadas entre si; a variância de um
   otimizador iterativo seria **maior que a diferença a medir**. Ridge tem solução fechada e uma
   decomposição espectral entrega todos os λ.
2. **Pseudo-inversa truncada no ridge.** Direções com autovalor `< max·1e−10` recebem peso **zero**.
   Sem isso, ruído de ponto flutuante no espaço nulo era amplificado por 1/λ e o resultado dependia
   de LAPACK — medimos **0.511 numa máquina e 0.580 noutra** no mesmo dado.
3. **`snap()` antes de ranquear** (1e−12 relativo): sem isso o `rankdata` ordena ruído de 1e−17 e a
   AUROC de um probe degenerado passeia em torno de 0.5 em vez de ser 0.5.
4. **Estatística em stdlib.** O `$PY` do host de GPU é enxuto e não há garantia de scipy. Os testes
   travam equivalência com o scipy até 1e−6.
5. **`PROBE_BATCH_SIZE = 4`, travado em código.** Não há cross-talk entre linhas (invariância de
   conteúdo = 0.0 exato), mas cuBLAS/cuDNN/Mamba escolhem algoritmo por dimensão de batch e o
   resultado muda ~2e−3. Inofensivo porque ref/alt vão no mesmo forward (Δ estável a 0,4%;
   sinal/ruído 426×). **Deixa de ser inofensivo se alguém agrupar vários sítios num batch maior.**
6. **fp32 na extração, TF32 desligado.** cuDNN liga TF32 por default (~10 bits de mantissa).

---

## 7. PENDENTE — o próximo passo, exatamente

Os três runs abaixo já têm tudo pronto no repo. **CPU, ~10 min cada.** As features já estão extraídas
em `~/probe/rich/probe_features.npz` (309 MB, 10.761 gold).

```bash
# 1. core_locus com as configs novas (max-pool isolado, sem-gnomAD, proposta_v2)
export WORK=~/testeArq/lumina-beat-regionalization && cd "$WORK" && git pull && PYTHONPATH="$WORK" "$PY" scripts/probe_feature_eval.py --features ~/probe/rich/probe_features.npz --release-root ~/mosaic-v1 --configs configs/probe_feature_configs.json --out ~/probe/rich/eval_core_v2.json 2>&1 | tee ~/probe/eval_core_v2.log
```

```bash
# 2. gene_transfer -- O MAIS IMPORTANTE
export WORK=~/testeArq/lumina-beat-regionalization && cd "$WORK" && PYTHONPATH="$WORK" "$PY" scripts/probe_feature_eval.py --features ~/probe/rich/probe_features.npz --release-root ~/mosaic-v1 --configs configs/probe_feature_configs.json --track gene_transfer --out ~/probe/rich/eval_gene_transfer.json 2>&1 | tee ~/probe/eval_gene.log
```

```bash
# 3. probe MLP -- responde se as 68 dims lineares ajudam um modelo nao-linear
export WORK=~/testeArq/lumina-beat-regionalization && cd "$WORK" && PYTHONPATH="$WORK" "$PY" scripts/probe_feature_eval.py --features ~/probe/rich/probe_features.npz --release-root ~/mosaic-v1 --configs configs/probe_feature_configs.json --mlp --out ~/probe/rich/eval_core_mlp.json 2>&1 | tee ~/probe/eval_mlp.log
```

### Como ler cada um

**(2) `gene_transfer` é o que decide a especificação.** A **diferença** entre `core_locus` e
`gene_transfer` para a mesma config **é a medida de memorização**. Olhar especialmente:
- `so_contexto`: se despencar em gene_transfer, os 0.8872 eram identidade de gene → **o Bloco C sai**.
- `variante_mais_contexto` vs `so_variante`: se o ganho do contexto sumir, idem.

**(1) `cabecas_sem_gnomad` vs `cabecas_so_gnomad`.** Se o ganho das cabeças vier das 8 dims de
gnomAD, é circularidade herdada e não vale reportar como capacidade do modelo.

**(3) MLP.** Se `so_cabecas_lineares` subir muito com MLP e pouco com ridge, as direções
privilegiadas ajudam por viés indutivo — e o Bloco H fica. Se não, ele encolhe para as 10 dims MLP.

### Depois disso

- **R3**: 4 kb vs 16 kb (4× de custo, só se o pooling largo mostrar ganho) e o **boost nativo**
  (`variant_edit_mask`) — 2× de extração.
- **R4**: configuração vencedora sob o protocolo **completo** (treino gold+consensus, 5 runs de cada
  trilha). Extração de consensus: ~316 mil variantes × 4 forwards ≈ **3,5 h** a 4 kb.
- **Documento para o Eduardo**, e só então comparar com a proposta dele.

---

## 8. Ambiente e fluxo (crucial)

- **Windows local não roda nada pesado.** Claude edita e entrega runbooks; o Gabriel commita, pusha,
  dá `git pull` no notebook SageMaker e roda lá (GPU/torch/S3).
- **`scripts/env.sh` NÃO exporta `$WORK`** — só `PY`, `REPO_ROOT`, `VENV_DIR`. Sempre incluir
  `export WORK=~/testeArq/lumina-beat-regionalization` no runbook. `cd ""` no bash retorna 0 **sem
  mudar de diretório**, então o erro sai confuso.
- Ambiente: `cd ~/testeArq/lumina-inference && source scripts/env.sh` → `$PY`. Rodar com
  `PYTHONPATH="$WORK" "$PY" ...`. **Nunca `uv sync` no host de GPU.**
- GPU: **A10G 23 GB**, suficiente (pico 1,75 GB em 32 k). Não precisa de H100.
- Dados no notebook: release em `~/mosaic-v1/`, features em `~/probe/rich/`, hg38 em `~/hg38/hg38.fa`.
- Checkpoint: `s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt`

### Gotcha do release
O prefixo plano do S3 (`benchmarks/mosaic/v1/`) está **pré-migração do ADR 0006**: traz
`bundle.manifest.json` com `bundle_identity_hash = a2cf6be0…` em vez de `release.manifest.json` com
`475a86c2…`. Os Parquets são byte-idênticos (a migração é um `mv`); só a identidade declarada é a
antiga. Já avisado ao Eduardo? **Não — pendente.**

---

## 9. Como trabalhar nesta frente

O que tem funcionado e deve continuar:

1. **Verificar em vez de supor.** Ler o código do modelo antes de assumir comportamento. Achados que
   vieram só disso: as cabeças lineares, as camadas locais não-hookáveis, o campo receptivo do mid.
2. **Prever antes de medir.** Registrar o número esperado, rodar, comparar. Foi assim que validamos o
   harness (conservação ~0,85 previsto → 0,877 medido).
3. **Revisão adversarial antes de reportar.** Na sonda isso pegou **quatro** interpretações erradas
   que já iam para o Eduardo — nenhuma era bug de código, todas eram leitura larga demais dos números
   certos (cosseno usado para comparar blocos de normas diferentes, tendência lida em ruído,
   intervalos desiguais comparados, AUROC sem IC).
4. **Testes que provam o que importa, não cobertura.** Ex.: o one-hot de locus dando 0.511 bloqueado
   vs 0.728 sem bloqueio prova que o harness não vaza *e* que a feature não é inútil.
5. **Números com a barra certa ao lado.** Nunca "AUROC 0.94" sem dizer que a anotação sozinha dá 0.93.
