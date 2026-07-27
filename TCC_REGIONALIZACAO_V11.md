# Regionalização do Beat-v11 — Documento explicativo do TCC (iteração 3)

> **Propósito deste documento.** Registrar, de forma didática e detalhada, **tudo o que
> fizemos na frente de regionalização** do modelo genômico Beat-v11: a ideia científica,
> a arquitetura, o que é cada componente, as decisões que tomamos, os resultados, e os
> problemas enfrentados. Está escrito para ser **estudado** — parte dos conceitos básicos
> e sobe até os detalhes.
>
> Datado de 2026-07-23 (iteração 1: 2026-07-14; iteração 2: 2026-07-16; iteração 3: 2026-07-19).
> Autor da frente: Gabriel (dev). Gestor: Eduardo. Baseline v10: Pedro.

> **O que mudou na iteração 4** (leia isto se já estudou a iteração 3):
> - **A frente de *melhoria* também fechou — com dois negativos medidos.** Depois que o porte
>   fechou, o objetivo mudou: **aumentar o `br_only`, mesmo pagando em global**. Atacamos por duas
>   alavancas independentes e **as duas deram negativo bem-dimensionado** (§7, "Pós-porte"):
>   (B) re-tunar o desconto está **saturado**; (A) a guarda por **cabeças nativas do v11**
>   (conservação phyloP + missense-severity ESM-2) foi **refutada**.
> - **O achado que isso produz é científico, não operacional:** *nem os sinais moleculares nativos
>   do v11 distinguem uma founder patogênica brasileira de uma benigna comum.* Elas **se parecem**
>   para todo sinal molecular disponível → **o gargalo não é o modelo** (§10.9).
> - **Consequência para a conclusão:** "curadoria externa" deixa de ser uma **herança** da
>   recomendação do Pedro e passa a ser uma **conclusão nossa, com evidência medida** num backbone
>   melhor (§11).
> - **Uma lição metodológica nova:** *média × mediana* — a Alavanca A foi **proposta** a partir de
>   médias e **refutada** pelas medianas (§9, lição 8).

> **O que mudou na iteração 3** (leia isto se já estudou a iteração 2):
> - **A Fase 4 fechou — a frente de regionalização está COMPLETA.** Treinamos o M5
>   frequência-explícita no v11, avaliamos, e rodamos a calibração `M5_v2` → `M5_v3_safety`
>   (§7, Fase 4).
> - **A falsificação da calibração deu POSITIVA** (p=0.0196): o desconto de frequência é
>   comprovadamente **ABRAOM-específico**. **É o resultado que o v10 do Pedro NÃO conseguiu**
>   (lá os controles chegavam perto → *não*-falsificado). É a descoberta central desta iteração
>   (§7 Fase 4, §10.6).
> - **O lead do v11 é o `M5_v2`** — o pipeline barrou o `M5_v3` (a guarda molecular não vale a
>   troca no v11; motivo na §7 Fase 4). No v10 o lead era o `M5_v3`. Uma diferença sutil e
>   informativa entre as versões.
> - **A tabela definitiva v11 × v10** (com a linha calibrada) e a **conclusão compilada para o
>   Pedro** — melhoras *e* pioras — na §10.5–10.8.

> **O que mudou na iteração 2** (leia isto se já estudou a iteração 1):
> - **Fase 5 (eval regional) fechou** — temos as *slices* decisivas no v11 (§7, §10).
> - **O veredito da Fase 3 ficou estatisticamente cravado** com um **teste bem-dimensionado**
>   (§7, Fase 5) — e ele **corrigiu duas conclusões erradas** que um conjunto pequeno produzira.
> - **Descobrimos o mecanismo real da "frequência explícita"** (§5.11): não é pós-hoc, é um
>   **modelo treinado** — o que redefine o que a Fase 4 precisa fazer.
> - **A tese central foi refinada** (§10.4): o backbone melhor levanta brasileiro **e** global
>   quase igualmente, mas **não fecha o gap** entre eles.
> - Novas lições metodológicas sobre **poder estatístico** e **vazamento** (§9).

---

## Sumário

1. [O problema, em linguagem simples](#1-o-problema-em-linguagem-simples)
2. [Conceitos que você precisa ter na cabeça](#2-conceitos-que-você-precisa-ter-na-cabeça)
3. [A hipótese científica central](#3-a-hipótese-científica-central)
4. [A baseline: o estudo do Pedro no Beat-v10](#4-a-baseline-o-estudo-do-pedro-no-beat-v10)
5. [A arquitetura da regionalização — a ideia, em detalhe](#5-a-arquitetura-da-regionalização--a-ideia-em-detalhe)
6. [O que é cada adapter, por que existe, e como interagem](#6-o-que-é-cada-adapter-por-que-existe-e-como-interagem)
7. [O pipeline em fases — o que fizemos, na ordem](#7-o-pipeline-em-fases--o-que-fizemos-na-ordem)
   - *inclui, ao final:* **Pós-porte — as duas tentativas de melhorar o `br_only` (ambas negativas)**
8. [Decisões-chave e o porquê de cada uma](#8-decisões-chave-e-o-porquê-de-cada-uma)
9. [Problemas enfrentados (e o que aprendemos)](#9-problemas-enfrentados-e-o-que-aprendemos)
10. [Resultados consolidados até agora](#10-resultados-consolidados-até-agora)
11. [O que falta](#11-o-que-falta)
12. [Glossário](#12-glossário)

---

## 1. O problema, em linguagem simples

Quando uma pessoa faz um exame genético, encontram-se **variantes** — pontos onde o DNA
dela difere da referência. A pergunta clínica é: **essa variante é patogênica (causa
doença) ou benigna (inofensiva)?** Bancos como o **ClinVar** catalogam variantes já
classificadas por especialistas, e é com eles que se treinam e avaliam modelos que tentam
prever patogenicidade.

O problema que motiva esta frente: os modelos de fundação de DNA (*foundation models*) vão
**mal em variantes brasileiras**. Elas são um **ponto cego estrutural** — o modelo erra mais
nelas, e adaptações leves quase não melhoram. A suspeita é que o gargalo é o **atalho de
frequência** (explicado na §3). O objetivo da regionalização é **corrigir esse viés para o
contexto brasileiro/latino-americano**, usando uma base de frequências brasileiras chamada
**ABRAOM**.

Este trabalho tem duas frentes que compõem o TCC:
- **Finetuning no ClinVar (benchmark Mosaic):** já feito — adaptar o Beat-v11 à tarefa de
  patogenicidade.
- **Regionalização (este documento):** melhorar a interpretação de patogenicidade
  especificamente para variantes brasileiras.

---

## 2. Conceitos que você precisa ter na cabeça

Antes da arquitetura, os blocos básicos. Se algum já é familiar, pule.

**Variante.** Uma diferença pontual no DNA. O caso mais comum é o **SNV** (*single
nucleotide variant*): uma única base trocada (ex.: um `A` que virou `G`). Cada posição do
DNA tem uma base de 4 possíveis: A, C, G, T.

**Patogenicidade.** O rótulo clínico da variante. Simplificando para dois grupos:
**patogênica/provavelmente patogênica (P/LP)** vs **benigna/provavelmente benigna (B/LB)**.
É uma tarefa de **classificação binária**.

**Frequência alélica (AF, *allele frequency*).** Quão comum a variante é numa população.
Vai de 0 (nunca vista) a 1 (todo mundo tem). Uma variante com AF alta (ex.: 20% das pessoas
têm) é, quase por definição, **provavelmente benigna** — se causasse doença grave, a seleção
natural a teria tornado rara. **Frequência alta ⇒ evidência a favor de benigno.** Esse é o
elo entre frequência e patogenicidade.

**gnomAD.** A maior base pública de frequências alélicas. Problema: é **majoritariamente
europeia** (o subgrupo NFE — *Non-Finnish European* — domina). Então a "frequência de
referência" que os modelos usam é, na prática, europeia.

**ABRAOM.** Base de frequências alélicas **brasileiras** (coorte de idosos de São Paulo,
estudo SABE). Dá a AF **na população brasileira** — que pode ser bem diferente da europeia.

**Por que isso gera o ponto cego brasileiro.** Existem variantes **comuns no Brasil mas
raras na Europa** (por *founder effects*, ancestralidade, etc.). Um modelo que usa AF
europeia (gnomAD) vê essas variantes como "raras" e tende a marcá-las como suspeitas
(**falso-positivo**: chama de patogênica algo que é benigno comum no Brasil). Corrigir isso
= usar a AF **brasileira** (ABRAOM) para recalibrar. Mas há um perigo: algumas variantes
**patogênicas** são comuns no Brasil (doenças recessivas/*founder* — ex.: variantes
fundadoras em genes específicos). Se descontarmos patogenicidade só por frequência alta,
**apagamos essas P/LP** (colapso de recall). Todo o desafio é equilibrar isso.

**MCC, AUROC, AUPRC, specificity, recall.** Métricas de classificação.
- **MCC** (*Matthews Correlation Coefficient*): a métrica-resumo mais honesta para classes
  desbalanceadas; vai de −1 a +1 (0 = aleatório). É a nossa métrica principal.
- **AUROC / AUPRC:** medem o quão bem o modelo **ordena** patogênicas acima de benignas,
  independente de um limiar (*threshold*). São "livres de limiar".
- **Specificity:** dentre as benignas, quantas o modelo acertou como benignas (mede
  falso-positivo — chave para as benignas-comuns-ABRAOM).
- **Recall (sensibilidade):** dentre as patogênicas, quantas o modelo pegou (chave para
  **não suprimir** as P/LP founder/recessivas).
- **Spearman:** correlação de **ranking** (usada para avaliar os adapters de frequência —
  ver §6). Mede se o modelo ordena certo, sem depender da escala exata.

**Foundation model (modelo de fundação).** Um modelo grande pré-treinado em MUITO DNA, que
aprendeu uma **representação** rica das sequências (como um "cérebro" que já leu o genoma).
O nosso é o **Beat-v11** (52 milhões de parâmetros). Ele produz, para cada posição do DNA,
um vetor de números (o *hidden state*) que resume o que ele "entende" daquela posição.

**LoRA (*Low-Rank Adaptation*).** Técnica para adaptar um modelo grande **sem re-treiná-lo
inteiro**. Você **congela** o modelo original e adiciona pequenas matrizes treináveis
(de "posto baixo") em pontos específicos. Treina só essas matrizes — muito mais barato,
e não estraga o conhecimento original. Um **adapter**, no nosso vocabulário, é justamente
um conjunto dessas matrizes LoRA treinado para uma tarefa específica.

---

## 3. A hipótese científica central

> **O gargalo das variantes brasileiras é o "atalho de frequência".** Os modelos aprendem a
> usar a AF como pista de patogenicidade — mas a AF de referência é europeia (gnomAD NFE).
> Logo, penalizam variantes comuns em populações não-europeias. **Calibrar a AF por população
> (ABRAOM) deveria corrigir o viés.**

Dessa hipótese nascem duas perguntas que orientam TODO o trabalho, e que é importante
**não confundir**:

1. **A sequência de DNA carrega sinal de frequência *brasileiro-específico*?** Ou seja: só
   olhando o DNA, dá para saber que uma variante é comum *no Brasil especificamente* (e não
   só "comum em geral")? → Isso é o que os **adapters de frequência** testam (§6, Fase 1 e B).

2. **Usar a frequência *observada* (a AF real do ABRAOM) para recalibrar melhora a
   classificação?** → Isso é a **calibração** (§5.4, Fase 4). É um mecanismo diferente: não
   depende da sequência carregar o sinal; usa o número de frequência que já temos medido.

**O grande achado do projeto (spoiler):** a resposta à pergunta 1 é "quase não" (o sinal
brasileiro-específico na sequência é **muito fraco**), mas a resposta à pergunta 2 é "sim"
(a calibração pela AF observada funciona). Ou seja: **o valor do ABRAOM está na frequência
observada (calibração), não num sinal regional aprendível da sequência.** Guarde isso — é o
fio condutor.

---

## 4. A baseline: o estudo do Pedro no Beat-v10

Antes de nós, o **Pedro** montou o pipeline completo de regionalização no modelo anterior,
o **Beat-v10**. O nosso trabalho é **portar** esse pipeline para o Beat-v11 (backbone melhor)
e ver se os achados se sustentam. O estudo dele é a nossa **baseline** — a régua de comparação.

**A "escada" de modelos do Pedro (M0 → M7).** Cada degrau adiciona um ingrediente:

| Modelo | O que é |
|---|---|
| **M0** | Baseline **molecular** de patogenicidade — treinado só em ClinVar **não-brasileiro** (*leave-Brazilian-out*). É a "patogenicidade pura", sem regionalização. É o piso contra o qual todo ganho regional é medido. |
| **A_BR / A_gnomAD / A_scrambled** | Os **adapters de frequência** (§6). Predizem AF a partir da sequência. |
| **M4** | *Static fusion*: combina o M0 com os adapters via um "portão" (gate), **sem** frequência explícita forte. |
| **M5 / M6** | Fusion **+ frequência explícita** (bruto). Reduziram falso-positivo, mas **suprimiram P/LP** comuns no ABRAOM (colapso de recall). |
| **M5_v2_calibrated** | Desconto de frequência **limitado** (calibração). |
| **M5_v3_safety** | **O candidato final:** desconto regional limitado **+ guarda molecular** (a frequência pode reduzir o score, mas NÃO apagar evidência molecular forte). Protege as P/LP founder/recessivas. |
| **M7_scrambled** | **Controle negativo:** fusion com um adapter de frequência **embaralhada** (sem sentido). Serve para falsificar — se o adapter real não bate o embaralhado, não há sinal específico. |

**O resultado headline do Pedro (v10):** o `M5_v3_safety` sobe o MCC no subconjunto
brasileiro (`br_only`) de **0.279 → 0.605** e a specificity de benignas-comuns-ABRAOM de
**0.803 → 0.959**, **sem** degradar o global. **A calibração de AF regional funciona.**

**O caveat honesto (o valor científico real do report dele):** os **controles negativos
estratificados** (que preservam gene, faixa de AF, tipo, cromossomo, e quebram só o elo
variante↔frequência) **chegam perto do real**. Ou seja, a especificidade **biológica** do
ABRAOM **não foi falsificada** — parte do ganho pode ser estrutura genérica de frequência,
não "brasilidade" aprendida. Por isso o próximo passo dele (no v10) era **curadoria** de
variantes, não mais treino — e essa curadoria está *gated* (precisa de mais dados). Foi
**por isso** que a direção virou "portar pro v11 e revalidar".

---

## 5. A arquitetura da regionalização — a ideia, em detalhe

Aqui está o coração do que você pediu para entender. Vou construir a arquitetura camada por
camada.

### 5.1 A base: o backbone congelado (Beat-v11)

Tudo parte do **Beat-v11**, o foundation model. Ele já foi pré-treinado e **fica congelado**
(não mexemos nos pesos dele). Para cada posição do DNA, ele devolve um vetor de **448 números**
(o `last_hidden_state`, de dimensão `d_full = 448`) que resume tudo o que ele entende daquela
posição — conservação evolutiva, contexto, efeito provável de mutações, etc.

Por que congelar? Porque re-treinar 52 milhões de parâmetros é caro, arriscado (o modelo
"esquece" o que sabia) e desnecessário: a representação dele **já é ótima**. Um teste simples
(um *probe* linear sobre o hidden state congelado) já classifica ClinVar com **AUROC 0.953** —
altíssimo. Ou seja, o conhecimento de patogenicidade **já está lá**; só precisamos extraí-lo.

### 5.2 Adaptar sem re-treinar: LoRA e a ideia de "adapter"

Como adaptar um modelo congelado? Com **LoRA**: adicionamos pequenas matrizes treináveis
("adaptadores de posto baixo") em 105 pontos do backbone (nas camadas Mamba e de atenção).
Treinamos **só** essas matrizes (≈1,2 milhão de parâmetros, ~2% do total). O backbone original
não muda.

Um **adapter** = um conjunto dessas matrizes LoRA, treinado para uma tarefa específica. Pense
nele como um "óculos" que você coloca sobre o modelo congelado para fazê-lo enxergar uma
coisa particular (por exemplo, "estime a frequência brasileira desta variante"). Você pode
ter **vários óculos** (vários adapters) e escolher quais usar.

### 5.3 As duas metades do problema: molecular × frequência

A patogenicidade tem **duas fontes de evidência** que a arquitetura trata separadamente:

- **Evidência molecular (o M0):** "essa mudança quebra a proteína / atinge região
  conservada / afeta *splicing*?" — puramente biológica, independente de população. É o que
  o **M0** aprende (treinado em ClinVar não-brasileiro).
- **Evidência de frequência (os adapters):** "essa variante é comum ou rara (e em qual
  população)?" — frequência alta empurra para benigno. É o que os **adapters de frequência**
  capturam.

A **fusão** (*fusion*) combina as duas.

### 5.4 A fusão (fusion) — combinando molecular + frequência

O mecanismo de fusão (no código: `fusion_lora.py`) parte do M0 (o caminho molecular) e
**acrescenta os adapters de frequência**, combinando tudo com um **portão (gate)** que decide
**quanto peso** dar a cada adapter. Existem duas variantes:

- **Fusão estática (M4):** o peso de cada adapter é um número **aprendido, fixo** para todo o
  dataset (um *softmax* sobre "logits de adapter"). Um único peso abraom×gnomad, igual para
  todas as variantes.
- **Fusão dinâmica (M5):** o peso é decidido **por exemplo** (por variante) por uma pequena
  rede (o *gate MLP*), condicionada no hidden state. Pode dar mais peso ao abraom numa
  variante e ao gnomad em outra.

Matematicamente, em cada ponto LoRA a saída fica (simplificado):

```
saída = base_congelada + caminho_M0 + Σ  peso[i] · adapter[i] · escala[i]
                                       i
```

onde `peso[i]` (o "peso de importância" do adapter i) vem do gate, e `escala[i]` é fixa
(vem do treino do adapter — o alpha/rank do LoRA).

> **Ponto que gerou uma decisão importante (§8):** "os pesos de importância entre os adapters"
> = esses `peso[i]`. Eles são **aprendidos** (começam uniformes). A pergunta "precisamos
> mudá-los à mão para o v11?" a gente respondeu **medindo** o que o v11 aprende — não chutando.

### 5.5 A calibração — o mecanismo que de fato regionaliza

Aqui está a parte mais importante e mais sutil. A fusão acima usa os adapters, que **predizem
frequência a partir da sequência**. Mas o achado central (§3) é que esse sinal é fraco. **O
que realmente funciona** é a **calibração**, que é diferente:

A calibração **não usa a predição do adapter** — ela usa a **AF observada** (o número real
medido no ABRAOM). Ela pega o score de patogenicidade e **desconta** um pouco quando a
variante é comum no ABRAOM (empurra para benigno), com dois freios de segurança:

- **desconto limitado** (`discount_scale`, `max_discount`): a frequência não pode zerar o
  score, só reduzi-lo até um teto.
- **guarda molecular** (`molecular_guard_threshold`, `guard_score_floor`): se a evidência
  molecular for **forte** (score molecular acima de um limiar), a frequência **não pode
  descontar** — protege as P/LP founder/recessivas comuns no Brasil.

É a calibração (o `M5_v3_safety`) que produziu o ganho headline do Pedro. **É calibração de
frequência observada, não biologia aprendida da sequência.** Essa distinção é a tese do
trabalho.

### 5.6 Resumo visual da arquitetura

```
                    ┌─────────────────────────────┐
   DNA (ref/alt) ──▶│  Beat-v11 (backbone CONGELADO)  │──▶ hidden state [448 por posição]
                    └─────────────────────────────┘
                                  │
                 ┌────────────────┼─────────────────────────┐
                 ▼                ▼                          ▼
         ┌──────────────┐  ┌──────────────┐        ┌───────────────────┐
         │  M0 (LoRA)   │  │ adapters de   │        │  cabeça de         │
         │  molecular   │  │ frequência    │        │  patogenicidade    │
         │  patogenic.  │  │ A_BR, A_gnomAD│        │  (classificador)   │
         └──────┬───────┘  └──────┬───────┘        └─────────┬─────────┘
                │                 │                          │
                └──── FUSION (gate: peso[i]) ────────────────┘
                                  │
                                  ▼
                          score de patogenicidade
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │  CALIBRAÇÃO (AF observada)     │  ← usa a AF real do ABRAOM,
                  │  desconto limitado +           │     NÃO a predição do adapter
                  │  guarda molecular              │
                  └───────────────────────────────┘
                                  │
                                  ▼
                    score regionalizado (final)
```

### 5.7 Como um adapter (e o M0) é treinado — o mecanismo

Todo o pipeline usa **aprendizado supervisionado**: mostramos ao modelo milhares de exemplos
no formato *(entrada → resposta certa)* e **ajustamos os pesos treináveis** para que ele
produza a resposta certa. Três peças que **não** se devem confundir:

| Peça | O que é | Papel |
|---|---|---|
| **Alvo (*target*)** | a **resposta certa**; uma **coluna do dataset** (um rótulo já medido) | é o **gabarito** |
| **Cabeça (*head*)** | a pequena rede treinável que pega o hidden state e cospe uma predição | quem **chuta** |
| **Perda (*loss*)** | função que mede **o quão errado** o chute está vs o alvo | o **placar do erro** |

**O passo de treino, concretamente** (repetido milhares de vezes — aqui para um adapter de
frequência):

```
1.  Pega uma variante: sequência de DNA (entrada) + af_abraom real (o ALVO).
2.  Passa a sequência pelo backbone CONGELADO + LoRA  →  hidden state (448 nºs/posição).
3.  A CABEÇA pega o hidden state da variante e cospe UM número: o palpite.
4.  A PERDA compara: palpite vs af_abraom real (o alvo)  → "errou por tanto".
5.  Retropropaga o erro e ajusta SÓ o LoRA + a cabeça (o backbone NÃO muda).
6.  Próxima variante. Repete milhares de vezes.
```

O que "congelado" significa em termos práticos: no passo 5, os gradientes **não fluem** para
os pesos do backbone — só para o LoRA e para a cabeça. O conhecimento pré-treinado do Beat-v11
fica intacto; a gente só aprende a **ler** esse conhecimento para uma tarefa nova.

**Vocabulário de treino** (útil citar no TCC):

- **Época (*epoch*):** uma passada completa por todo o dataset de treino.
- **Passo (*step*):** uma atualização dos pesos (após processar um "batch efetivo").
- **Batch:** quantos exemplos processados de uma vez (aqui, 2 — a GPU é pequena).
- **Acumulação de gradiente (*grad_accum*):** processa N batches pequenos e só então atualiza,
  simulando um batch maior. **Batch efetivo = batch × grad_accum** (aqui 2 × 8 = 16). É um
  truque para caber na memória da GPU.
- **Learning rate (lr):** o tamanho do passo de ajuste. Usamos **dois**: um menor para o LoRA
  (`5e-6`) e um maior para a cabeça (`5e-4`). Motivo: a cabeça é nova e precisa aprender do
  zero (passo grande); o LoRA só ajusta de leve a leitura do backbone (passo pequeno, para não
  desestabilizar).
- **Congelar por N passos (*freeze_backbone_steps*):** nos primeiros N passos só a cabeça
  treina (aquece a cabeça antes de deixar o LoRA mexer).
- **Função de perda:** **BCE** (*binary cross-entropy*, para alvos em [0,1] como `af_abraom`
  ou rótulos de classe); **focal** (variante da BCE que dá mais peso aos casos difíceis, usada
  no M0); **huber** (regressão robusta, para alvo **ilimitado** como `delta_logit`, que pode
  ser negativo).
- **LoRA rank / alpha:** `rank` = a "capacidade" do adapter (matrizes de posto `r`; maior =
  mais expressivo, mais parâmetros); `alpha` = a escala com que a saída do adapter entra.
  Usamos rank 8 / alpha 16 (escala efetiva 2.0).
- **Amostragem `balanced-af`:** ao montar os batches, balanceamos as **faixas de frequência**.
  Sem isso, como a **maioria** das variantes é rara (AF baixa), o adapter aprenderia o atalho
  preguiçoso "chute sempre baixo" e teria Spearman enganosamente ok. Balancear força-o a
  distinguir de verdade raras × comuns.

**Por que "sequence-only".** Nos adapters de frequência, **não** damos a AF do gnomAD como
entrada extra (`use_gnomad_prior=False`). Se déssemos, o adapter poderia "colar" (só repetir o
gnomAD) e o teste "a sequência carrega o sinal?" perderia o sentido. Forçando-o a prever
**só a partir da sequência**, medimos exatamente o que a sequência codifica.

**A analogia do placebo (o A_scrambled).** O A_scrambled é treinado **igualzinho** ao A_BR,
mas com o alvo **embaralhado** (frequência da variante errada) — o "princípio ativo" (o elo
real sequência↔frequência) é removido. Ele é o **placebo**: nos diz quanto o maquinário
"acerta" **sem sinal real**. Só declaramos que o A_BR aprendeu algo real porque ele **bate o
placebo com folga** (A_BR − A_scrambled ≫ 0).

### 5.8 As receitas exatas de cada etapa (para reprodutibilidade)

Cada etapa é o **mesmo maquinário** (backbone congelado + LoRA + cabeça), variando **o que
treina**, **os dados**, **o alvo**, **a perda** e a **métrica**. As receitas foram extraídas
dos artefatos do Pedro (não dos defaults — ver §9) e reusadas em paridade no v11.

**(a) M0 — baseline molecular de patogenicidade**
- **Treina:** LoRA (rank 8 / α 16 / dropout 0.05) + **cabeça classificadora**
  (`proj_dim 256 → hidden 128 → 1`, dropout 0.2). Backbone congelado.
- **Dados:** `nonbr_only` (ClinVar não-brasileiro): ~11,7k treino / 1,3k val / 2,5k test.
- **Alvo:** o rótulo de classe (patogênica × benigna).
- **Perda:** *focal* (γ = 2.0), com `pos-weight auto` (compensa desbalanceamento de classes).
- **Hiperparâmetros:** ctx 1024, bf16, lr_backbone `5e-6` / lr_head `5e-4`, freeze 100 passos,
  batch 2 × grad_accum 8, grad_clip 0.5, warmup 0.1, **3 épocas**, extração *two-tower* ref/alt.
- **Métrica:** MCC / AUROC / AUPRC. **Resultado v11:** MCC 0.654 (v10: 0.576).

**(b) Adapters de frequência — A_BR, A_gnomAD, A_scrambled (Fase 1)**
- **Treina:** LoRA (rank 8 / α 16) + **cabeça regressora**
  (`LayerNorm → Linear(d_full·3 → 256) → GELU → Linear(→1)`). Backbone congelado.
- **Dados:** `abraom_frequency_adapter`: 100k treino / 5k val / 5k test, amostragem
  `balanced-af`, **sequence-only**.
- **Alvo:** `af_abraom` (A_BR) · `af_gnomad` (A_gnomAD) · `af_abraom` **embaralhado**
  (A_scrambled).
- **Perda:** BCE (alvos em [0,1]).
- **Hiperparâmetros:** ctx 1024, **1000 passos**, batch 2 × grad_accum 8, lr_lora `5e-6` /
  lr_head `5e-4`, seed 42.
- **Métrica:** Spearman (rank). **Resultado v11:** A_BR 0.127, A_gnomAD 0.118, A_scrambled
  −0.021 (test).

**(c) A_residual e A_residual_scrambled (Experimento B)**
- Igual aos adapters de frequência, **exceto**:
- **Alvo:** `delta_logit = logit(af_abraom) − logit(af_gnomad)` (o componente ABRAOM além do
  gnomAD) · e o controle `scrambled_delta_logit`. Em ambos a **métrica** é medida contra o
  `delta_logit` real.
- **Perda:** *huber* (não BCE!) — porque `delta_logit` é **ilimitado** (pode ser negativo),
  então não cabe em [0,1]. Foi por isso que precisamos threadar um alvo novo no trainer (§9).
- **Resultado:** gap A_residual − A_residual_scrambled = +0.026 (test) — sinal fraco, mas real.

**(d) Fusão estática M4 (Fase 3)**
- **Parte do M0** (via `--init`) e o **congela** (o caminho molecular do M0 vira input fixo).
  Os adapters A_BR e A_gnomAD entram **congelados**. **Treina apenas:** o **peso** de cada
  adapter (`adapter_logits`, um *softmax* fixo por módulo) + a cabeça.
- **Dados / alvo / perda:** iguais ao M0 (`nonbr_only`, rótulo de classe, focal).
- **Hiperparâmetros:** static_lora, rank **8** (obrigatório = rank do M0, ver §9), lr_head
  `1e-4` / lr_backbone `1e-3`, freeze 0, batch 2 × 8, **5 épocas**.
- **Resultado:** peso aprendido abraom 0.504 / gnomad 0.496 (uniforme).

**(e) Fusão dinâmica M5 (Fase 3)**
- Igual à M4, mas o peso vem de um **gate MLP por-exemplo** (`gate_hidden_dim 64`) em vez de um
  peso fixo.
- **Hiperparâmetros:** dynamic_lora, rank 8, lr_head `5e-4` / lr_backbone `5e-6`, freeze 100,
  batch 4 × 16, 5 épocas.

**(f) Controles M7 (static e dynamic scrambled)**
- Idênticos à M4/M5, **trocando o adapter A_BR real por o A_scrambled** (conjunto
  `scrambled + gnomad`). Servem para falsificar a fusão: o abraom real bate o embaralhado?
  (Resposta no v11, como no v10: **não**, no fusion cru — o ganho vem da calibração.)

> **Fio comum:** M0 e adapters são **irmãos** (mesma máquina, tarefas diferentes: M0 classifica,
> adapters regridem frequência). A **fusão** não treina novos adapters — ela **congela** M0 e
> adapters e só aprende **como combiná-los** (o peso) + a cabeça. A **calibração** (Fase 4) não
> treina nada disso: opera **sobre os scores**, usando a AF observada.

### 5.9 A extração *two-tower*: como uma variante vira entrada do modelo

Um detalhe fundamental que ficou implícito até aqui: o backbone lê **sequências** de DNA, mas
uma variante é uma **mudança em uma posição**. Como transformar "posição X: `A→G`" em algo que
o modelo pontue? A resposta é a extração **two-tower** (duas torres) — usada igual no M0 e nos
adapters (é por isso que a cabeça recebe `d_full·3`).

**O procedimento** (função `_extract_paired_variant_features`):

```
1. Monta a janela de REFERÊNCIA: a sequência genômica ao redor da variante (ex.: 1024 bp
   centrados nela), com a base de REFERÊNCIA na posição da variante.
2. Monta a janela ALT: a MESMA janela, mas com a base ALTERNATIVA trocada na posição.
3. Passa o backbone congelado nas DUAS (as "duas torres") → ref_hidden e alt_hidden
   (cada um: 448 números por posição).
4. Na posição da variante, extrai:
      site_ref     = ref_hidden na posição   (como o modelo representa o sítio com a ref)
      site_alt     = alt_hidden na posição   (com a base alternativa)
      variant_repr = site_alt − site_ref     ← a DIFERENÇA: o "efeito" da mutação
5. Calcula local_context = média de ref_hidden numa janela de ±64 bp (o contexto local).
6. Concatena [ site_ref , variant_repr , local_context ] → 3 × 448 = 1344 números.
   Esse vetor é a ENTRADA da cabeça.
```

**Por que esse desenho é inteligente:**
- **`variant_repr = alt − ref`** isola o **efeito** da mutação: subtrai tudo o que é igual entre
  as duas torres e sobra só "o que mudou" na forma como o modelo entende aquele ponto. Uma
  diferença grande sugere efeito funcional (candidato a patogênico).
- **`site_ref`** dá o contexto do sítio de referência.
- **`local_context`** (média ±64 bp) dá a vizinhança — a região é conservada? há sítio de
  *splice* por perto? etc.

**Analogia:** é como mostrar ao modelo as fotos "antes" (ref) e "depois" (alt) do mesmo ponto
e pedir que ele foque na **diferença**. A cabeça então mapeia esses 1344 números → a saída da
tarefa (uma frequência, nos adapters; um score de patogenicidade, no M0).

> Nota: como M0 e adapters usam **a mesma** extração, a comparação v11×v10 isola o backbone, e
> a fusão consegue empilhar o caminho molecular (M0) e os adapters no mesmo formato de features.

### 5.10 A calibração em detalhe — onde nasce o ganho regional (as fórmulas)

Esta é a expansão da §5.5, com o mecanismo exato (do `calibrate_m5_v3_safety.py`). **É a parte
mais importante metodologicamente**, porque é aqui — e não nos adapters — que o ganho regional
de fato acontece.

**Os dois números de entrada** (por variante, já calculados pela fusão/eval):
- **`molecular`** = a probabilidade de patogenicidade **molecular** (0 a 1), derivada do M0.
  É a "biologia pura".
- **`regional_discount`** = quanto a evidência de **frequência** quer **puxar o score para
  baixo** (em unidades de *logit*). Ele nasce (`apply_capped_frequency_discount`) da diferença
  entre a margem molecular e a margem regional-bruta, e é **limitado por um teto que cresce com
  a AF observada do ABRAOM** — ou seja, só se permite desconto grande quando a variante é de
  fato **comum no Brasil**. **Crucial:** usa a **AF observada** (medida), não a predição do
  adapter.

**A config do `M5_v3_safety` (Pedro, v10):** `discount_scale=0.5`, `max_discount=0.5`,
`molecular_guard_threshold=0.65`, `guarded_max_discount=0.0`, `guard_score_floor=0.35`,
`regional_threshold=0.35`, `global_threshold=0.72`.

**A fórmula, passo a passo:**

```
1. raw_discount     = regional_discount × discount_scale      (×0.5)   # aplica só metade
2. capped_discount  = min(raw_discount, max_discount)         (≤0.5)   # teto do desconto
3. guard_mask       = (molecular ≥ molecular_guard_threshold) (≥0.65)  # evidência molecular forte?
4. guarded_discount = guard_mask ? min(capped_discount, guarded_max_discount=0.0)
                                  : capped_discount                    # se forte → desconto = 0
5. regional_score   = sigmoid( logit(molecular) − guarded_discount )   # aplica em espaço logit
6. safety_floor     = guard_mask ? max(regional_score, guard_score_floor=0.35)
                                  : regional_score                     # piso p/ molecular forte

   DECISÃO (qual score + limiar usar):
     • slice GLOBAL (não-BR)  → usa  molecular    , limiar = global_threshold  (0.72)
     • slice REGIONAL (BR)    → usa  safety_floor , limiar = regional_threshold (0.35)
```

**Os dois freios de segurança, em português:**

- **Desconto limitado (passos 1–2):** a frequência pode **reduzir** o score, mas nunca zerá-lo
  — só aplica metade (`discount_scale`) e no máximo `max_discount` (0.5 em logit). Impede que
  "comum no Brasil" apague qualquer evidência.
- **Guarda molecular (passos 3–4 e 6):** se a evidência molecular for **forte**
  (`molecular ≥ 0.65`), então **(a)** o desconto é cortado para **zero** (`guarded_max_discount=0`)
  — a frequência **não toca** — e **(b)** o score não pode cair abaixo de `0.35`
  (`guard_score_floor`). **É isso que protege as P/LP founder/recessivas** comuns no Brasil: uma
  variante patogênica com evidência molecular forte não é rebaixada só por ser frequente.

**Por que funciona (e por que é a tese do trabalho):** o desconto usa a **AF observada** do
ABRAOM — não a predição de nenhum adapter. Por isso a calibração funciona **mesmo** com os
adapters carregando quase nenhum sinal regional (§7): ela não depende do modelo *aprender*
brasilidade da sequência; ela usa o **número de frequência que já medimos**. O limiar regional
mais baixo (0.35) também ajuda: nas variantes brasileiras o modelo era conservador demais
(muitos falso-positivos em benignas comuns), e calibrar + baixar o limiar reduz falso-positivo
enquanto a guarda protege os verdadeiros positivos.

> **O que muda no v11 (Fase 4):** o **mecanismo** é o mesmo, mas os valores (0.65 / 0.35 / 0.72)
> quase certamente precisam ser **re-ajustados** — a distribuição de score do v11 é diferente da
> do v10. Re-fitamos esses thresholds num conjunto de *holdout* (é o trabalho da Fase 4).

### 5.11 De onde vêm `molecular` e `regional_discount` — a "frequência explícita" é um MODELO

A §5.10 assume dois números de entrada (`molecular` e `regional_discount`). **De onde eles
saem?** Parece um detalhe, mas a resposta **redefine o que a Fase 4 tem de fazer** — e só
descobrimos investigando o código (mais um caso da lição da §9).

**Resposta: eles NÃO vêm da fusão crua (M4/M5) que treinamos na Fase 3.** Vêm de uma **variante
de modelo diferente**, que o Pedro chama de **M5 "frequência explícita"**. E a diferença entre
o M4 e o M5-explicit-freq é **uma única configuração**:

| | `explicit_feature_columns` | cabeça usada |
|---|---|---|
| **M4** (o que treinamos) | `[]` (vazio) | `RegimeAHead` — só features de sequência |
| **M5 explicit-freq** | `[af_abraom, af_gnomad, specificity, abraom_present, is_snv, log10_af_abraom, log10_af_gnomad, af_delta, af_abs_delta, af_ratio_log10, ...]` | **`RegimeABoundedRegionalHead`** |

**O que "frequência explícita" significa, então:** dar a **AF real como entrada direta da
cabeça**, junto com as features de sequência. Não é pós-processamento — é o modelo
**recebendo o número da frequência** e aprendendo o que fazer com ele.

**E a cabeça `RegimeABoundedRegionalHead` faz a decomposição** que a calibração consome:

```
molecular_logit   = molecular_head(hidden)                              # a "biologia pura"
regional_discount = max_discount(=4.0) · sigmoid(discount_head(hidden)) # o desconto, JÁ limitado
regional_logit    = molecular_logit − regional_discount                 # o score final
```

Repare: **o teto já nasce na arquitetura** — o `sigmoid` multiplicado por `max_discount=4.0`
garante que o desconto fique em [0, 4]. Ou seja, **o próprio modelo aprende a descontar por
frequência, com um freio embutido**.

**Então existem DUAS camadas de calibração** (é fácil confundi-las):

| Camada | Onde vive | O que faz |
|---|---|---|
| **1. Desconto aprendido** | **dentro do modelo** (`RegimeABoundedRegionalHead`, no treino) | aprende, a partir da AF explícita, **quanto** descontar — já limitado por arquitetura |
| **2. Calibração de segurança** (M5_v2/v3) | **pós-hoc**, sobre os scores (`calibrate_m5_v3_safety.py`) | **re-aperta** o desconto (`discount_scale`, `max_discount`) e adiciona a **guarda molecular** — tunada em *holdout* |

**A consequência prática (e por que isso importa muito):** os nossos M4/M5 da Fase 3 têm
`explicit_feature_columns = []` → cabeça sem decomposição → **não produzem
`molecular_probability` nem `regional_discount`** → **não servem de entrada para a calibração**.
Logo, a **Fase 4 não é "calibrar o que já temos"**: é **treinar um modelo novo** (o M5
explicit-freq no v11), avaliá-lo, e **só então** calibrar. Descobrir isso mudou o plano — e nos
poupou de rodar avaliações "completas" nos modelos errados (§8).

---

## 6. O que é cada adapter, por que existe, e como interagem

Os **adapters de frequência** são todos LoRA (mesma estrutura), treinados sobre o backbone
Beat-v11 **congelado**, cada um com um **alvo** diferente. A tarefa deles: **prever uma
frequência alélica a partir da sequência de DNA**. A "cabeça" (head) de cada um é uma pequena
rede que pega o hidden state e cospe um número (a frequência prevista).

| Adapter | Alvo de treino | Função / por que existe |
|---|---|---|
| **A_BR** | `af_abraom` (AF brasileira) | O adapter "regional". Se ele prevê bem a AF brasileira **a partir da sequência**, então a sequência carrega sinal de "quão comum no Brasil". |
| **A_gnomAD** | `af_gnomad` (AF global/europeia) | O **comparador de "frequência genérica"**. Como `af_gnomad ≈ af_abraom` na maioria dos casos, ele é um baseline forte. **A_BR só tem sinal *regional* se BATER o A_gnomAD.** |
| **A_scrambled** | `af_abraom` **embaralhada** | **Controle negativo.** A frequência é embaralhada (associada à variante errada). Define o **piso**: quanto o maquinário acerta por estrutura, sem sinal real. Sem esse controle, "Spearman 0.13" não significa nada. |
| **A_residual** | `delta_logit = logit(af_abraom) − logit(af_gnomad)` | O teste **direto** do componente regional. O resíduo é o que é ABRAOM **além** do gnomAD. É a melhor chance de achar sinal brasileiro-específico. |

**Como eles se relacionam (a lógica dos controles):**

- `A_BR` vs `A_scrambled` responde: **"existe sinal sequência→frequência de qualquer tipo?"**
  Se A_BR ≫ A_scrambled, sim.
- `A_BR` vs `A_gnomAD` responde: **"esse sinal é *brasileiro-específico*?"** Se A_BR não bate
  A_gnomAD, o sinal é "frequência genérica", não "brasilidade".
- `A_residual` vs `A_residual_scrambled` responde a mesma pergunta de forma **direta**
  (isolando o resíduo regional).

**Por que a métrica é Spearman (e não outra).** Avaliamos os adapters por **Spearman** (rank).
Pearson seria injusto: o A_BR treina na escala do `af_abraom`, então acerta a magnitude
trivialmente — Spearman só premia acertar a **ordenação**, que é o que importa.

**Como os adapters interagem no downstream (fusão).** No M4/M5 (fusão), os adapters A_BR e
A_gnomAD entram como "óculos" adicionais sobre o M0, e o **gate** decide o peso de cada um.
Importante: como a fusão é **treinada em dados não-brasileiros** (o `nonbr_only`), o gate
**nunca vê variantes brasileiras no treino** → ele **não tem como aprender** a dar mais peso
ao abraom para BR. Por isso o ganho regional **não vem do gate**; vem da **calibração** (§5.5).
Isso explica um resultado que veremos na §7 (os pesos aprendidos ficaram uniformes).

---

## 7. O pipeline em fases — o que fizemos, na ordem

### Fase 0 — Integração do Beat-v11
Adaptamos o pipeline do Pedro (feito para o v10) para carregar e usar o Beat-v11. Criamos o
`FineTuneBeatV11Adapter`, que carrega o checkpoint r1, extrai o hidden state certo
(`last_hidden_state`, 448 dims) e aplica LoRA nos 105 pontos corretos (sem tocar nas cabeças
nativas). Validado com *smoke tests*. **Status: ✅ fechada.**

### Fase 1 — Adapters de frequência (A_BR, A_gnomAD, A_scrambled)
Treinamos os três adapters com a **receita exata do Pedro** (contexto 1024, 1000 passos,
`balanced-af`, *sequence-only*). Resultado (Spearman val/test):

| Adapter | v10 (Pedro) | **v11 (nós)** |
|---|---|---|
| A_BR | 0.114 / 0.104 | **0.135 / 0.127** |
| A_gnomAD | 0.107 / 0.118 | **0.120 / 0.118** |
| A_scrambled (piso) | −0.013 / 0.001 | **−0.036 / −0.021** |

Testes pareados (bootstrap): `A_BR − A_scrambled` = **+0.15** (test), CI ≫ 0 → **há sinal
sequência→frequência real**. `A_BR − A_gnomAD` = **+0.010** (test), CI cruza 0 → **NÃO é
brasileiro-específico**.

> **Veredito da Fase 1:** `A_BR ≈ A_gnomAD ≫ A_scrambled`. Há sinal de frequência real, mas
> **genérico** (o v11 aprende AF melhor que o v10, mas não "brasilidade"). Isso **reproduz e
> explica** o caveat do Pedro. **Status: ✅ fechada.**

### Experimento B — Teste de resíduo (A_residual)
Como o A_BR-vs-A_gnomAD é um teste **indireto** e de baixa potência (deu nulo), fizemos o
teste **direto**: treinar A_residual no `delta_logit` (o componente ABRAOM além do gnomAD) e
comparar com seu controle embaralhado.

Resultado (gap pareado A_residual − A_residual_scrambled): **test +0.026, CI [+0.001, +0.050]
(exclui 0); val +0.028, CI cruza 0 por um triz.** A trajetória mostrou que o sinal ainda
subia no passo 1000 (o gap crescia para +0.032).

> **Veredito de B:** um sinal regional **fraco, mas real** (mais limpo que o teste indireto da
> Fase 1). Pequeno demais para justificar mudar a arquitetura, mas não é zero. **Status: ✅
> fechada.** Decisão que gerou: **não reestruturar a fusão** (§8).

### Fase 2 — M0 (baseline molecular no v11)
Treinamos o M0 (patogenicidade molecular, ClinVar não-brasileiro) no v11, com a receita do
Pedro (regime A, foco *focal loss*, LoRA 8/16, 3 épocas).

| Métrica (test) | v10 (Pedro) | **v11 (nós)** |
|---|---|---|
| MCC | 0.576 | **0.654** |
| AUROC | 0.879 | **0.927** |
| AUPRC | 0.890 | **0.940** |

> **Leitura:** o v11 melhora o baseline molecular **com folga** (+0.078 MCC), só pelo backbone
> melhor — como o probe linear (AUROC 0.953) previa. **Status: ✅ fechada.**

### Fase 3 — Fusão (M4 estático, M5 dinâmico) + controles (M7)
Rodamos a fusão real + os controles scrambled. **Detalhe crítico que pegamos nos commits do
Pedro:** o `--lora-rank` da fusão **tem que ser 8** (igual ao M0), senão as chaves LoRA do M0
são **filtradas silenciosamente** e o caminho molecular se perde (commit `d79fd31`).

Métricas globais (nonBR test) + pesos aprendidos:

| Run | modo | adapters | MCC global | peso aprendido |
|---|---|---|---|---|
| M0 (ref) | — | — | 0.654 | — |
| M4 | static | abraom+gnomad | 0.677 | **abraom 0.504 / gnomad 0.496** |
| M7s (controle) | static | scrambled+gnomad | 0.683 | scrambled 0.502 / gnomad 0.498 |
| M5 | dynamic | abraom+gnomad | 0.665 | (gate por-exemplo) |
| M7d (controle) | dynamic | scrambled+gnomad | 0.688 | (gate por-exemplo) |

> **Veredito da Fase 3 (parte 1 — o que o global já mostrava):** (1) o fusion cru **≈ M0**
> globalmente (o caminho dos adapters não adiciona valor molecular); (2) **scrambled ≥ real**
> (o adapter abraom real não bate um embaralhado); (3) os **pesos aprendidos ficaram uniformes**
> (~50/50) e **idênticos para o scrambled** — o gate **não distingue** o abraom real de ruído.
> Isso reproduz o achado do Pedro no v10 (o M7_scrambled dele ficava perto do M4 real).
> **Status: ✅ fechada** — mas o veredito **estatístico definitivo** (com intervalo de confiança
> nas slices brasileiras) só veio na **Fase 5**, com o teste bem-dimensionado. Leia lá: ele
> **corrigiu duas leituras** que fizemos aqui.

> **Por que os pesos ficaram uniformes — a explicação estrutural.** Não é acaso: a fusão é
> **treinada em `nonbr_only`** (dados **não**-brasileiros). O gate **nunca vê uma variante
> brasileira no treino** → ele **não tem como aprender** "para BR, dê mais peso ao abraom".
> Ele fica indiferente porque, nos dados que ele vê, o adapter de frequência brasileira não
> ajuda mesmo. Isso não é um bug — é uma **consequência lógica do desenho leave-Brazilian-out**,
> e é mais um motivo pelo qual o ganho regional **tem** de vir da calibração (§5.5), não do gate.

### Fase 5 — Eval regional + o teste de falsificação (o veredito definitivo)

Aqui avaliamos os checkpoints (M0 + as 4 fusões) nas **slices decisivas** — os subconjuntos que
respondem à pergunta clínica de verdade. O eval roda no v11 **sem mudança de código** (ele lê a
família do modelo do próprio checkpoint e reconstrói a fusão a partir do estado salvo).

**As 4 slices decisivas e o que cada uma mede:**

| Slice | Mede | Métrica |
|---|---|---|
| `br_only` | desempenho no **subconjunto brasileiro** — o alvo da frente | **MCC** |
| `abraom_common_benign` | **falso-positivo** em benignas comuns no Brasil | **specificity** |
| `abraom_pathogenic_present` | **não suprimir** as P/LP do ABRAOM (founder/recessivas) | **recall** |
| `global_nonbr_no_abraom` | **não degradar** o resto (fora da regionalização) | **MCC** |

**Resultados (split `test`, contra o v10 do Pedro):**

| Slice (test) | v10 M0 | **v11 M0** | v10 M4s | **v11 M4s** | **v11 M5d** | **v11 M7s** (scr) | **v11 M7d** (scr) |
|---|---|---|---|---|---|---|---|
| `br_only` MCC (n=504) | 0.279 | 0.238 | 0.292 | 0.320 | 0.292 | 0.266 | 0.278 |
| `abraom_common_benign` spec (n=12k) | 0.803 | 0.842 | 0.894 | 0.879 | 0.854 | 0.864 | 0.862 |
| `abraom_pathogenic_present` recall (n=163) | 0.417 | 0.460 | 0.288 | 0.350 | 0.436 | 0.387 | 0.393 |
| `global_nonbr` MCC (n=1989) | 0.512 | **0.606** | 0.526 | **0.631** | 0.614 | 0.641 | 0.647 |

À primeira vista pareceu que o M4 real batia o scrambled no `br_only` (0.320 vs 0.266 =
**+0.054**) — sugestivo de sinal ABRAOM. **Mas esse número tinha um problema grave.**

#### O teste bem-dimensionado — e por que ele foi decisivo

**O problema: poder estatístico.** O `br_only` no split `test` tem só **504 variantes**. Com um
n desse tamanho, o erro padrão do MCC é grande (~±0.05–0.07) — um Δ de +0.054 pode ser **puro
ruído**. Foi exatamente o que o **bootstrap pareado** mostrou:

| Par | Δ (test, n=504) | 95% CI | veredito |
|---|---|---|---|
| M4 − M7s | +0.0538 | **[−0.021, +0.133]** | CI cruza 0 → inconclusivo |
| M5 − M7d | +0.0138 | [−0.049, +0.081] | CI cruza 0 |

> **O que é "bootstrap pareado" e por que é a ferramenta certa.** Reamostramos as **mesmas**
> variantes (com reposição) milhares de vezes e recalculamos a diferença real−scrambled em cada
> reamostra. Como os dois modelos são avaliados **nas mesmas variantes**, boa parte do ruído se
> cancela — o erro padrão da **diferença** é bem menor que o de cada MCC isolado. O resultado é
> um **intervalo de confiança da diferença**: se ele cruza 0, não podemos afirmar que há efeito.

**A sacada que resolveu.** O `br_only` do `test` é pequeno — **mas o slice inteiro é avaliação
válida**. Por quê? Porque as slices são **disjuntas por construção**:

```
br_only     = tem submissor brasileiro     E NÃO tem submissor não-brasileiro
nonbr_only  = tem submissor não-brasileiro E NÃO tem submissor brasileiro   ← onde o M0 treinou
```

Como o M0/fusão treinaram **só** em `nonbr_only`, e as duas definições são mutuamente
exclusivas, **nenhuma variante do `br_only` esteve no treino**. Logo **o `br_only` inteiro
(split `all`, n=4163) é out-of-sample** — usá-lo não é vazamento, é apenas **usar todos os dados
que já eram legítimos**. Isso multiplica o n por **8×** e encolhe o CI por **√8 ≈ 2.9×**.

> ⚠️ **A armadilha que evitamos.** Isso vale **só para as slices brasileiras** (`br_only`,
> `br_any`). As slices `abraom_common_benign` e `global_nonbr_no_abraom` **intersectam** o
> `nonbr_only` do treino → nelas o split `all` **vazaria**. Nessas, `test` é obrigatório.
> Verificamos as definições no código **antes** de decidir — não assumimos.

**O resultado bem-dimensionado (`br_only`, split `all`, n=4163):**

| Modelo | MCC |
|---|---|
| M0 | 0.3350 |
| **M4** (real, static) | 0.3532 |
| **M7s** (scrambled, static) | 0.3504 |
| **M5** (real, dynamic) | 0.3562 |
| **M7d** (scrambled, dynamic) | 0.3445 |

| Par | Δ (all, n=4163) | 95% CI | P |
|---|---|---|---|
| **M4 − M7s** | **+0.0028** | **[−0.024, +0.030]** | 0.574 |
| **M5 − M7d** | +0.0117 | [−0.010, +0.033] | 0.854 |

**Replicado no `br_any`** (n=4872): +0.0027 e +0.0114 — praticamente idênticos.

> **Veredito (o fechamento da Fase 3):** o **+0.054 era ruído**. Com 8× mais dados ele desaba
> para **+0.003**, e o CI — agora apertado (~0.05 de largura) e **centrado em zero** — permite
> uma afirmação forte: **o fusion cru NÃO carrega sinal ABRAOM-específico; o efeito, se existe,
> é menor que ±0.03 MCC.** Isso é um **negativo bem-dimensionado** — cientificamente mais forte
> que o do Pedro no v10 (+0.018, CI cruzando **por falta de poder**). **Confirmamos e
> fortalecemos** a conclusão dele. **Status: ✅ fechada.**

#### As duas correções que o teste bem-dimensionado produziu

Este é o ponto mais didático da frente inteira: **o conjunto pequeno (n=504) enganou em DUAS
direções opostas**, e só o teste com poder revelou.

**Correção 1 — o "bump do fusion" também era ruído.** No `test`, o M0→M4 no `br_only` parecia
**+0.082** (0.238→0.320). No split `all`: **+0.018** (0.3350→0.3532) — e o **scrambled captura
+0.015 disso** (0.3350→0.3504). Ou seja: o fusion mal move o brasileiro, e o pouco que move é
**estrutura**, não ABRAOM.

**Correção 2 (a mais importante) — o backbone v11 AJUDA o brasileiro.** No `test` (n=504), o
v11 M0 (0.238) parecia **pior** que o v10 (0.279), e chegamos a escrever "backbone melhor não
ajuda o brasileiro". **Estava errado — era ruído.** No split `all`, com **o mesmo conjunto de
4163 variantes nos dois modelos**:

| `br_only` `all` (n=4163) | **v10 M0** | **v11 M0** | Δ |
|---|---|---|---|
| MCC | **0.2476** | **0.3350** | **+0.087** |

O v11 melhora o brasileiro em **+0.087** — praticamente o mesmo ganho que no global (+0.094).
A consequência dessa correção para a tese está na §10.4.

### Fase 4 — M5 frequência-explícita + calibração (fechada)

**Redefinida pela descoberta da §5.11:** não foi "calibrar o que já temos" — foram **três
passos**, porque a calibração precisa de um **modelo que ainda não tínhamos** (o M5 com a AF como
entrada explícita da cabeça).

#### Passo 1 — treinar o M5 frequência-explícita no v11

Fusão dinâmica + `--head-type regime_a_bounded_regional` + as **11 features de frequência**
explícitas (`log10_af_abraom`, `log10_af_gnomad`, `af_delta`, `af_ratio_log10`, `af_*_missing`,
`specificity`, `abraom_present`, `is_snv`, ...). A cabeça `RegimeABoundedRegionalHead` produz a
decomposição `molecular_logit` / `regional_discount` (§5.11), com o teto embutido na arquitetura.

> **Detalhe pinado no código:** as 8 features "engenheiradas" (log10, deltas, `*_missing`) **não
> estão no parquet** — são **derivadas em runtime** das 5 colunas-base (`dataset.py`). O "FALTAM
> 8" que assustou no pré-check era alarme falso.

**Curadoria do resultado (nonBR test MCC = 0.749 — "bom demais"?).** O M0 dava 0.654; o
M5-bounded deu **0.749**. Investigamos *antes* de acreditar (a lição da §9):
- **Não é artefato de threshold** (MCC no threshold "vazado" ≈ o honesto: 0.758 vs 0.749).
- **Não é overfitting** (val 0.762 ≈ test 0.749; matriz de confusão balanceada).
- **Não é vazamento de rótulo:** a feature `specificity` vem **mergeada do ABRAOM** (dado
  populacional), **não** do rótulo ClinVar — estruturalmente não pode vazar o *label*.
- É **ganho legítimo do `af_gnomad`** (frequência global, critério ACMG válido: comum ⇒ benigno)
  × o backbone melhor. Mesmas 11 features do Pedro → sem vazamento novo; o +0.11 sobre ele é o
  backbone.

> **Leitura importante:** esse ganho no nonBR vem da frequência **global** (gnomAD), não da
> brasileira — coerente com a tese (o valor está na AF observada). A parte **brasileiro-
> específica** é o que a falsificação (abaixo) isola.

#### Passo 2 — eval-alvo (o padrão bounded cru)

Avaliamos o M5-bounded em `test`+`holdout` nas 8 slices. O padrão **reproduziu exatamente** o do
Pedro: o modelo aprende a **descontar tudo que é comum** →

| slice (test) | v10 bounded | **v11 bounded (cru)** | leitura |
|---|---|---|---|
| `br_only` MCC | 0.666 | **0.621** | alto (o desconto ajuda o BR) |
| `abraom_common_benign` spec | 0.998 | **0.994** | quase-perfeito (mata falso-positivo) |
| **`abraom_pathogenic_present` recall** | **0.037** ⚠️ | **0.080** ⚠️ | **colapsado** (atropela os P/LP founder) |
| `global_nonbr` MCC | 0.400 | **0.635** | v11 degrada bem menos |

O recall em **0.08** é o **problema-alvo** (não um bug): o desconto cru mata os patogênicos
comuns no Brasil. **É exatamente isso que a calibração conserta.**

#### Passo 3 — calibração M5_v2 (o desconto tunado) e M5_v3 (a guarda)

**M5_v2** re-tuna o desconto no *holdout* (escala 1.0, teto 1.5, thresholds regional 0.235 /
global 0.765). Efeito no `br_only`:

| br_only (test) | MCC | **recall P/LP** | specificity |
|---|---|---|---|
| M5-bounded **cru** | 0.621 | **0.080** ⚠️ | 0.994 |
| **M5_v2 calibrado** | 0.574 | **0.405** ✅ | 0.951 |

A calibração **recuperou o recall P/LP de 0.08 → 0.41**, sacrificando pouco MCC (0.62→0.57).
Exatamente o trade-off que ela existe para fazer. **O `M5_v2` é o nosso lead.**

**M5_v3** adiciona a **guarda molecular** — e aqui o pipeline tomou uma **decisão automática de
segurança**: `hold_current_lead` (o M5_v2 continua o líder). Por quê? A guarda resgatou **+0.12**
de recall P/LP (0.41 → 0.52, resgatando 19 variantes) **mas criou 208 novos falso-positivos** em
benignas comuns (specificity 0.951 → 0.934, abaixo da tolerância). Como o custo em specificity
superou o ganho, o pipeline **manteve o M5_v2**.

> **Por que o v3 ajudou o Pedro (v10) mas não a nós (v11) — um achado da replicação.** A cabeça
> molecular do v11 é **mais forte** → **mais benignas comuns recebem score molecular alto** → a
> guarda (que protege score-molecular-alto do desconto) acaba "desprotegendo" essas benignas
> (os 208 FPs). No v10, com molecular mais fraco, isso doía menos, e o v3 do Pedro avançou. **No
> v11, o M5_v2 já é bom o bastante que a guarda não compensa.** É uma interação sutil entre a
> qualidade do backbone e a guarda — algo que **só a replicação no v11 revelou**.

#### O teste que fecha a frente — a falsificação da calibração

Este é o resultado mais importante da Fase 4 (possivelmente da frente inteira). A pergunta: **o
desconto de frequência da calibração reflete biologia ABRAOM real, ou um desconto embaralhado
faria igual?** (a mesma lógica dos controles scrambled, agora sobre a **calibração**, não o
adapter).

**O teste (controles negativos estratificados):** embaralha-se **quem recebe qual desconto**,
preservando a estrutura (`within_gene`, `within_af_bin`, `within_chromosome`, `global`),
re-mede-se o `br_only` MCC, 50 vezes por modo (200 no total). Se o desconto real rastreia *quais*
variantes são de fato comuns no Brasil, dá-lo às variantes **erradas** deve **piorar**.

**O resultado — o desconto real (0.561) ficou ACIMA de TODOS os 200 embaralhamentos:**

| modo | real | média dos controles | **p(controle ≥ real)** |
|---|---|---|---|
| global | 0.561 | 0.459 | **0.0196** |
| within_gene | 0.561 | 0.507 | **0.0196** |
| within_af_bin | 0.561 | 0.524 | **0.0196** |
| within_chromosome | 0.561 | 0.464 | **0.0196** |

`p = 0.0196 = 1/51` é o **mínimo possível** com 50 seeds → **nenhum** controle alcançou o real,
em **nenhum** modo — inclusive no `within_af_bin`, o mais estrito (preserva a distribuição de
frequência e só quebra o elo variante↔desconto).

> **Veredito — e por que é ÓTIMO para a pesquisa.** No v11, **a especificidade ABRAOM está
> FALSIFICADA de forma limpa e positiva**: o ganho brasileiro **depende de QUAIS variantes
> específicas são comuns no ABRAOM**, não de uma regra genérica de frequência. **Isso é
> exatamente o que o v10 do Pedro NÃO conseguiu** — lá os controles chegavam perto e ele teve que
> ser conservador ("especificidade biológica não falsificada"). **A nossa replicação no v11
> RESOLVEU essa pergunta aberta.** Não só reproduzimos o pipeline: *melhoramos* o resultado
> científico. **Status: ✅ fechada — a frente de regionalização está completa.**

### Pós-porte — as duas tentativas de melhorar o `br_only` (ambas negativas)

Com o porte fechado, o objetivo mudou de natureza. Até aqui a pergunta era *"os achados do Pedro
se sustentam no v11?"* (replicação). A partir daqui virou **otimização**: *"dá para **aumentar** o
`br_only`, mesmo pagando em global?"*

Atacamos por **duas alavancas independentes**. **As duas fecharam com negativo bem-dimensionado.**
Registramos com o mesmo cuidado dos resultados positivos — um negativo **medido** vale muito mais
que uma tentativa não feita, e são exatamente eles que **fundamentam** a conclusão da §11.

#### Alavanca B — re-tunar o desconto: **SATURADO**

A alavanca mais barata primeiro. O `M5_v2` escolheu um ponto do grid de calibração **respeitando um
piso de recall**. E se afrouxarmos esse piso? Não precisa treinar nada: o grid de tuning no
*holdout* **já estava computado** (`holdout_tuning_results.csv`) — bastou relê-lo.

Relendo, a fronteira `br_only ↔ recall` é **fixa**, e o global fica **constante em 0.635** o tempo
inteiro (ou seja: não existe nem o trade-off "perder global para ganhar brasileiro" que o Gabriel
tinha autorizado — o global sequer se move):

| piso de recall imposto | `br_only` MCC máximo alcançável | recall P/LP no ponto |
|---|---|---|
| ≥ 0.00 (sem piso) | **0.698** | **0.075** ☠️ |
| ≥ 0.30 | 0.612 | 0.310 |
| **≥ 0.40** ← o nosso lead já está aqui | **0.597** | 0.406 |

(Entre eles: `recall ≥ 0.20 → br_only 0.629`.)

**Por que o "teto" de 0.698 é uma miragem.** Ele é alcançado **chamando quase tudo de benigno**: o
recall P/LP desaba para **0.075**, isto é, o modelo **perde 92% das variantes patogênicas**.
Clinicamente esse é o pior erro possível — deixar de sinalizar uma variante que causa doença é
muito mais grave do que sinalizar uma benigna a mais. Um `br_only` de 0.698 comprado assim é
**contabilidade, não medicina**.

**O que a saturação significa, mecanicamente.** O desconto apenas **desliza** ao longo da fronteira
recall↔`br_only`; ele **não a levanta**. Isso é o esperado quando se olha o que o desconto *é*: uma
transformação monotônica única do score em função da AF observada. Ele **redistribui** erros entre
as duas classes — não **cria informação nova** para separá-las.

> **Veredito da Alavanca B:** o desconto está **saturado**. O `M5_v2` já ocupa o melhor ponto
> compatível com um recall P/LP defensável, e o único "ganho" disponível é trocar **recall** — a
> capacidade clínica que a frente existe para proteger. **Status: ✅ fechada (negativo).**

#### Alavanca A ("A-guarda") — guarda por cabeças nativas do v11: **REFUTADA**

Se o desconto não pode ser melhorado, a única saída é **levantar a fronteira** — e para isso é
preciso melhorar a **discriminação founder × benigna comum**. Daí nasceu a ideia mais promissora da
frente inteira.

**O raciocínio.** A guarda molecular do `M5_v3` protege do desconto as variantes com
`molecular_probability ≥ 0.65`. No v11 isso falhou (§7, Fase 4): a cabeça molecular mais forte
**super-estima benignas comuns**, então a guarda acaba protegendo **as variantes erradas** (208 FPs
para 19 resgates). **Mas e se a guarda usasse outro sinal?** O Beat-v11 tem **cabeças nativas que o
v10 não tinha**: `conservation_scalar_pred` (phyloP100 / Zoonomia-241 / phyloP470) e
`missense_severity_pred` (destilada do ESM-2). São sinais **específicos de patogenicidade** e — o
ponto crucial — **ortogonais à frequência**. A hipótese: *uma founder P/LP é conservada; uma benigna
comum não é.* Se fosse verdade, a guarda por conservação protegeria as founders (**recall ↑**) sem
desproteger as benignas (**specificity mantida**) → o `br_only` subiria **nos dois eixos**,
**levantando** a fronteira em vez de deslizar nela.

**O que construímos** (o código ficou no repo, mesmo com o resultado negativo):
`FineTuneBeatV11Adapter.extract_native_pathogenicity_features` (lê as duas cabeças nativas na
posição da variante) + `scripts/extract_native_pathogenicity_features.py` (standalone,
**backbone-only** — não exige re-rodar o M5-bounded) + o campo `conservation_guard_threshold` no
`SafetyConfig` da calibração. **A extração funcionou** (187/187 founders e 11497/11497 benignas
comuns mergeadas).

**Tentativa 1 — guarda = molecular *E* conservação. Falhou, e o motivo já é o achado.** As founders
que a guarda existe para proteger têm `molecular_probability` **BAIXO** (mediana **0.374**) — só
**6 de 187** passam o gate de 0.65. O `E` (conjunção) portanto **bloqueava 97% de quem deveria
proteger**.

> **Descoberta lateral que explica retroativamente a Fase 4.** Medindo isso, vimos que a guarda
> molecular original dispara para **6 founders e 83 benignas comuns** — ela protege **~14× mais
> benigna do que founder**. **É exatamente por isso que o `M5_v3` do Pedro produziu 208 FPs para 19
> resgates no v11.** O que era uma *hipótese* na Fase 4 ("a cabeça molecular do v11 super-estima
> benignas comuns") virou um **número medido**.

**Tentativa 2 — conservação *SUBSTITUI* o gate molecular** (thresholds varridos de 1.5 a 4.0,
corrigindo o erro da tentativa 1). O tuning selecionou `conservation_guard_threshold = 0.0` — ou
seja, **o grid preferiu não ter guarda nenhuma**.

**A causa-raiz, medida: os sinais simplesmente não separam.**

| sinal (mediana) | founders P/LP (n=187) | benignas comuns (n=11497) | separação |
|---|---|---|---|
| **phyloP100** (conservação) | **0.357** | **−0.183** | medianas se sobrepõem |
| **missense-severity** (ESM-2) | 4.24 | 3.68 | ~1.0–1.8× = inútil |

E o problema é pior do que as medianas sugerem, por causa da **base rate**: o enrichment do phyloP
tem **pico de apenas 5.4×** (no threshold 3.0) contra uma proporção de **61:1** (11497 benignas
comuns × 187 founders). Mesmo no melhor threshold possível, isso ainda são **~11 benignas guardadas
para cada founder** — a guarda continua protegendo majoritariamente quem não deveria.

> ⚠️ **O erro metodológico que cometemos aqui (vale mais registrado do que escondido).** A proposta
> nasceu das **médias**: founders phyloP **1.41** vs benignas **0.07** — um contraste de 20×,
> animador. **As medianas contam outra história:** 0.357 vs −0.183. A média das founders estava
> sendo puxada por uma **cauda conservada**; **metade das founders é tão pouco conservada quanto uma
> benigna comum**. Olhe a **distribuição**, não o resumo (§9, lição 8).

> **Veredito da Alavanca A: REFUTADA — e o negativo é o resultado mais valioso do pós-porte.**
> *Nem as cabeças nativas do v11 distinguem uma founder patogênica brasileira de uma benigna comum.*
> **O gargalo não é o modelo:** essas variantes **se parecem** para **todo** sinal molecular
> disponível — o classificador molecular treinado, a conservação evolutiva (phyloP) e a severidade
> missense destilada do ESM-2. É um negativo que **reproduz independentemente a conclusão do Pedro**
> (`do_not_train_next` → curadoria), agora com **evidência medida sobre um backbone melhor**.
> **Status: ✅ fechada (negativo).**

#### O que as duas alavancas, juntas, estabelecem

As duas atacaram o mesmo objetivo por caminhos ortogonais — **calibração** (B) e **discriminação**
(A) — e **as duas bateram no mesmo muro**, por motivos diferentes e ambos medidos: o desconto não
tem para onde ir, e não há sinal molecular que separe as classes que precisariam ser separadas.

**É isso que transforma "curadoria externa" de recomendação herdada em conclusão própria.** Não
paramos por falta de ideias ou de compute: paramos porque **medimos que as ideias disponíveis não
funcionam, e por quê**. O próximo ganho real depende de **rótulos melhores** (P/LP brasileiros
curados), não de mais modelagem — e essa afirmação, na iteração 4, tem evidência nossa por trás.

---

## 8. Decisões-chave e o porquê de cada uma

- **Rota B (teste de resíduo) antes da Rota A (porte downstream).** Escolhemos primeiro o
  teste que **decide se a fusão precisa de estrutura nova**. (A Rota A — completar o porte —
  ficou como fallback documentado.)
- **Não reestruturar a fusão.** Dois motivos: (1) o mecanismo já generaliza para N adapters
  (adicionar um adapter regional é *config*, não arquitetura); (2) o caminho do adapter é
  quase inerte no downstream (no v10, o M4 só moveu `br_only` de 0.279→0.292). O ganho vem da
  **calibração**, não do adapter.
- **Paridade com o Pedro na extração.** Usamos a mesma extração *two-tower* do v10, para o
  head-to-head v11×v10 isolar o **backbone** (não introduzir variáveis novas).
- **Medir os pesos, não chutar.** Sobre "ajustar os pesos de importância pro v11": rodamos a
  fusão em paridade e **lemos o que o v11 aprendeu** (ficou uniforme). Impor pesos do v10 às
  cegas seria injustificado e contaminaria a comparação.
- **`--lora-rank 8` em toda a fusão** (não o default 4), para preservar o caminho molecular do
  M0 (ver §7, Fase 3).
- **Refinamentos v11-nativos adiados de propósito.** O Beat-v11 tem cabeças nativas
  (ESM-2 missense-severity, conservação) e maquinário de variante que poderiam melhorar o
  resultado — mas usá-los **agora** confundiria o head-to-head. Ficam para uma fase de ablação
  **pós-paridade**.
- **Não alterar o Beat-v11.** Antes da Fase 3 verificamos, lendo o `model.py`, se valia mexer no
  modelo para melhorar o resultado. **Não vale, e manter como está é o correto:** (1) o
  `encode()` já devolve a representação certa (`last_hidden_state`, 448 dims, com RMSNorm final
  — é o mesmo readout que **todas as cabeças nativas** consomem); (2) essa representação já é
  fortíssima (probe linear no ClinVar = AUROC **0.953**); (3) as cabeças nativas patogenicidade-
  relevantes são **derivadas do mesmo hidden state**, logo largamente redundantes com o que o
  LoRA+cabeça já extrai. O modelo fica **congelado e intacto**.
- **Fechar a Fase 3 com um teste bem-dimensionado, em vez de aceitar o inconclusivo.** O
  bootstrap no `test` deu "CI cruza 0" — tecnicamente um resultado, mas **fraco** ("não deu para
  saber"). Como o `br_only` inteiro sempre foi out-of-sample, dava para ter **8× mais dados** por
  ~5 jobs de minutos. Decidimos gastar isso — e foi o melhor investimento da frente: virou um
  **negativo com intervalo** (`< ±0.03`) **e** corrigiu duas conclusões erradas (§7, §10.4).
- **Manter o eval regional enxuto (e NÃO rodar o "completo") — decisão contra-intuitiva.**
  Surgiu a dúvida: "não seria melhor rodar a avaliação completa agora, mesmo demorando, para não
  refazer depois?" Investigamos o código e a resposta foi **não**: a calibração (Fase 4) consome
  as *predictions* de um **modelo diferente** (o M5 frequência-explícita, §5.11), que ainda nem
  treinamos. Rodar o "completo" nos modelos atuais **não seria reaproveitado**. O escopo enxuto
  **não é desperdício** — é o entregável da falsificação; o "completo" acontece na Fase 4, no
  modelo certo. **Investigar antes de gastar compute.**

---

## 9. Problemas enfrentados (e o que aprendemos)

A **lição transversal** desta frente, repetida várias vezes: **nesta base, os defaults e a
documentação MENTEM — sempre leia o artefato/o modelo carregado.** Exemplos concretos:

- **`d_full` era 448, não 320.** Os defaults do dataclass diziam 256/320, mas o checkpoint r1
  sobrescreve para 384+64=448. Se confiássemos no default, tudo quebraria.
- **MoE desligado.** A arquitetura suporta *Mixture of Experts*, mas este checkpoint tem
  `moe_enabled=False`.
- **A "receita" ≠ defaults do trainer.** O Pedro sobrescreveu ~6 hiperparâmetros; usar os
  defaults daria um treino incomparável (e ~69h em vez de ~2h).
- **Ambiente (tilelang).** O conda do notebook tem um `tilelang/tvm` quebrado que aborta o
  import do `mamba_ssm`. Fix: um *shim* que força o fallback triton.
- **Whitelist do `--target-column`.** A 1ª tentativa do experimento B falhou porque
  `delta_logit` nem era um valor aceito; tivemos que "threadar" o alvo em 5 pontos do trainer.
- **Seleção de checkpoint por NLL.** No experimento B, o melhor passo era escolhido por NLL —
  que é **degenerada** para o alvo `delta_logit`. Isso subestimou levemente o sinal; a leitura
  correta veio da trajetória de Spearman.
- **`_upsert_arg` forçando v10.** Os launchers de M0/fusion **hard-forçavam** `beat-v10`;
  tivemos que parametrizar `--model-family`/`--model-version` (com cuidado: um passthrough
  depois do `--` era sobrescrito de volta).
- **Rank tem que casar (commit `d79fd31`).** Se o rank da fusão ≠ rank do M0, o carregamento
  filtra silenciosamente as chaves do M0 e **perde o caminho molecular sem dar erro**.
- **O eval regional estourou o tempo — duas vezes — por um `--batch-size` que faltou.** Os dois
  primeiros jobs morreram com `MaxRuntimeExceeded` (8h) e entregaram resultado **parcial** (só
  algumas slices, sem o `summary.json`). Causa: o script **herda o `batch_size` do checkpoint**
  (=2, do treino do M0) se você não passar nada — mas **o Pedro passava `--batch-size 8`**.
  Batch 2 vs 8 = **4× mais forward passes**, × o backbone v11 ser ~2× mais pesado = **~8× mais
  lento** que o eval dele, × a varredura completa (12 slices × 3 splits, uma delas com 71k
  variantes). **Fix:** `--batch-size 8` + restringir `--dataset-files`/`--splits` ao que
  interessa. **A lição:** "herdar o default do checkpoint" parecia inofensivo e custou 2 jobs.
- **A paginação do `list-training-jobs` corrompeu uma variável de shell.** `--query
  'TrainingJobSummaries[0].X'` **sem `--no-paginate`** aplica a query **por página** e devolve o
  nome do job seguido de dezenas de `None` — que entraram no `$JOB` e quebraram o `describe`
  seguinte com um erro enganoso ("nome > 63 caracteres"). **Fix:** `--no-paginate`, ou usar o
  `job_name=` que o launcher já imprime.
- **`all` nem sempre é seguro (vazamento).** Usar o split `all` para ganhar poder **só é válido
  se o slice for disjunto do treino**. Vale para `br_only`/`br_any` (verificamos as definições);
  **não** vale para `abraom_common_benign`/`global_nonbr_no_abraom`, que intersectam o
  `nonbr_only` do treino. Checar a definição **antes** de usar mais dados.
- **A "frequência explícita" não era o que parecia.** Assumimos que fosse pós-processamento ou
  um flag da fusão. **Não era:** é um **modelo treinado** (`--explicit-feature-columns` + cabeça
  `RegimeABoundedRegionalHead`, §5.11). Descobrir isso **antes** de rodar o eval "completo" nos
  modelos errados nos poupou horas — a Fase 4 precisa de um modelo **novo**.

**Lições metodológicas que viraram princípio:**
1. Sempre leia o artefato, nunca confie no default/doc.
2. Comparações **pareadas + bootstrap**, não pontos isolados (um Δ de +0.02 pode cruzar 0).
3. Spearman é a métrica honesta dos adapters.
4. **O controle negativo é o que dá sentido ao resultado** — sempre rode o scrambled.
5. **Poder estatístico não é detalhe — é o que separa conclusão de miragem.** Um n pequeno
   engana **nos dois sentidos**: no `br_only` com n=504 nós (a) vimos um "sinal ABRAOM" de
   +0.054 que era ruído (virou +0.003 com n=4163) **e** (b) concluímos que o backbone v11 não
   ajudava o brasileiro, quando na verdade ajudava **+0.087**. Antes de interpretar um Δ,
   pergunte: **qual o CI? qual o n? tenho poder para detectar um efeito desse tamanho?**
6. **Antes de aceitar "não deu significativo", pergunte se dava para ter mais dados.** O
   `br_only` inteiro sempre foi out-of-sample — o `test` (n=504) era uma restrição
   **desnecessária**. Trocar "underpowered, não dá pra saber" por **"o efeito é < ±0.03"** foi
   uma mudança qualitativa no valor científico do resultado, e custou 5 jobs de minutos.
7. **Distinga significância estatística de significância prática.** Na slice de 12k, o
   real×scrambled deu "significativo" nas duas arquiteturas — **mas com sinais opostos**
   (+0.0145 e −0.0079). Efeito minúsculo + direção inconsistente = **idiossincrasia**, não
   biologia. Com n grande, quase tudo fica "significativo"; olhe **magnitude e consistência**.
8. **Média × mediana: olhe a distribuição, não o resumo.** A "A-guarda" (§7, Pós-porte) foi
   **proposta** com base nas **médias** de conservação (founders 1.41 × benignas 0.07 — 20× de
   contraste) e **refutada** pelas **medianas** (0.357 × −0.183 — sobrepostas). A média estava
   sendo puxada por uma **cauda**: metade das founders é tão pouco conservada quanto uma benigna
   comum. Uma média pode descrever um subgrupo que existe **sem** descrever a população — e uma
   guarda opera sobre a **população inteira**, não sobre a cauda.
9. **Num problema desbalanceado, enrichment sozinho não decide nada — compare com a base rate.**
   O phyloP separava founders de benignas com enrichment de **5.4×**, o que soa ótimo. Mas a
   proporção de fundo é de **61:1** (11497 benignas × 187 founders): 5.4× contra 61:1 ainda deixa
   **~11 benignas protegidas por founder**. **Sempre pergunte: o meu sinal é mais forte que o
   desbalanceamento que ele precisa vencer?**
10. **Um negativo bem-medido é entregável, não fracasso.** As duas alavancas de melhoria fecharam
   negativas — e é o **par** delas (nenhuma calibração restante × nenhum sinal discriminante) que
   converte "curadoria externa" de palpite herdado em **conclusão sustentada** (§11). Documentar
   *por que* algo não funcionou impede que a próxima pessoa gaste o mesmo compute.

---

## 10. Resultados consolidados até agora

### 10.1 Nível molecular / global (métricas de treino, nonBR test)

| Modelo | v10 (Pedro) MCC | **v11 MCC** | AUROC v11 |
|---|---|---|---|
| M0 | 0.576 | **0.654** | 0.927 |
| M4 static | 0.590 | **0.677** | 0.922 |
| M5 dynamic | 0.639 | **0.665** | 0.929 |
| M7 static scrambled | — | 0.683 | 0.924 |
| M7 dynamic scrambled | — | 0.688 | 0.928 |

**Pesos aprendidos dos adapters (fusão estática):** abraom **0.504** / gnomad **0.496**
(uniforme); scrambled **0.502** / gnomad **0.498** (idêntico) → o gate não distingue o real do
embaralhado (a explicação estrutural está na §7, Fase 3).

### 10.2 As slices regionais (Fase 5) — o head-to-head v11 × v10

**Split `test`** (paridade com a tabela do Pedro):

| Slice (test) | v10 M0 | **v11 M0** | v10 M4s | **v11 M4s** | **v11 M5d** | v11 M7s (scr) | v11 M7d (scr) |
|---|---|---|---|---|---|---|---|
| `br_only` MCC | 0.279 | 0.238 | 0.292 | 0.320 | 0.292 | 0.266 | 0.278 |
| `abraom_common_benign` spec | 0.803 | 0.842 | 0.894 | 0.879 | 0.854 | 0.864 | 0.862 |
| `abraom_pathogenic_present` recall | 0.417 | 0.460 | 0.288 | 0.350 | 0.436 | 0.387 | 0.393 |
| `global_nonbr` MCC | 0.512 | **0.606** | 0.526 | **0.631** | 0.614 | 0.641 | 0.647 |

**Split `all` — o número confiável do brasileiro** (n=4163, 8× mais dados; ver §7, Fase 5):

| `br_only` `all` | **v10 M0** | **v11 M0** | v11 M4 | v11 M7s (scr) | v11 M5 | v11 M7d (scr) |
|---|---|---|---|---|---|---|
| MCC | 0.2476 | **0.3350** | 0.3532 | 0.3504 | 0.3562 | 0.3445 |

### 10.3 A falsificação (real × scrambled) — bem-dimensionada

| Par | Δ (`br_only` all, n=4163) | 95% CI | veredito |
|---|---|---|---|
| M4 − M7s | **+0.0028** | [−0.024, +0.030] | **≈ zero** |
| M5 − M7d | +0.0117 | [−0.010, +0.033] | CI cruza 0 |

Replicado no `br_any` (n=4872). **Conclusão:** o fusion cru **não carrega sinal
ABRAOM-específico** — o efeito, se existe, é **< ±0.03 MCC**. É um **negativo bem-dimensionado**,
mais forte que o do Pedro no v10 (+0.018, CI cruzando por falta de poder).

### 10.4 A tese refinada — o backbone levanta tudo, mas o gap PERSISTE

Este é o achado conceitual central da iteração 2. Comparando o **mesmo conjunto de variantes**
nos dois modelos:

| | **v10** | **v11** | Δ (o ganho do backbone) |
|---|---|---|---|
| `br_only` (all, n=4163) MCC | 0.248 | 0.335 | **+0.087** |
| `global_nonbr` (test) MCC | 0.512 | 0.606 | **+0.094** |
| **gap brasileiro ↔ global** | **−0.264** | **−0.271** | **≈ inalterado** |

**Como ler isso.** O Beat-v11 é um modelo genuinamente melhor: ele levanta o brasileiro (+0.087)
**e** o global (+0.094) — de forma **quase uniforme**. Mas o **abismo entre os dois permanece
praticamente idêntico** (−0.264 no v10 → −0.271 no v11).

> **A tese, na versão correta e bem-dimensionada:** *um foundation model melhor levanta a régua
> inteira, mas **não fecha** o ponto cego brasileiro.* O gap não é falta de representação — é
> falta de **calibração de frequência**. Por isso nenhuma quantidade de "backbone melhor"
> resolve, e por isso a calibração (Fase 4) é a peça que importa.

> ⚠️ **Nota de honestidade científica.** A iteração 1 afirmava, com base no split `test`
> (n=504), que o backbone v11 **não** ajudava o brasileiro (v11 0.238 < v10 0.279). **Isso
> estava errado — era ruído amostral.** A versão acima, medida em n=4163, é a correta. Fica
> registrado porque **o erro em si é didático**: mostra como um conjunto pequeno produz
> conclusões confiantes e falsas (§9).

### 10.5 Fase 4 — a calibração (a tabela definitiva v11 × v10)

| slice (test) | v10 M0 | v11 M0 | **v10 M5_v3** (lead Pedro) | **v11 M5_v2** (nosso lead) | v11 M5_v3 |
|---|---|---|---|---|---|
| `br_only` MCC | 0.279 | 0.238 | **0.605** | **0.574** | 0.561 |
| `abraom_common_benign` spec | 0.803 | 0.842 | 0.959 | **0.951** | 0.934 |
| `abraom_pathogenic_present` recall | 0.417 | 0.460 | 0.436 | **0.405** | 0.521 |
| `global_nonbr` MCC | 0.512 | 0.606 | 0.512 | **0.626** | 0.626 |

A calibração leva o `br_only` de **0.24 (M0) → 0.57 (M5_v2)** — espelhando o 0.28→0.61 do Pedro
— preservando recall e specificity. O lead é o **M5_v2** (o `M5_v3` foi barrado; §7 Fase 4).

### 10.6 A falsificação da calibração — o resultado que o v11 conseguiu e o v10 não

Controles negativos estratificados sobre o **desconto** (real × embaralhado), `br_only` MCC:

| modo (`global` / `within_gene` / `within_af_bin` / `within_chromosome`) | real | controle (média) | p(controle ≥ real) |
|---|---|---|---|
| todos os 4 modos | 0.561 | 0.46–0.52 | **0.0196** (o mínimo com 50 seeds) |

**O desconto real bate os 200 embaralhamentos** (nenhum alcança o real, nem no `within_af_bin`,
o mais estrito). No v11, a **especificidade ABRAOM está falsificada** (p<0.05) — o desconto é
ABRAOM-**específico**, não frequência genérica. **O v10 do Pedro não fechou isso** (controles
chegavam perto). A replicação **resolveu** a pergunta aberta.

### 10.7 Conclusão — v11 × v10: o que melhorou, o que piorou (para o Pedro)

**O que a replicação do pipeline de regionalização no v11 nos permitiu estabelecer:**

**✅ Melhoras do v11 sobre o v10:**
1. **Baseline molecular (M0):** MCC global **0.576 → 0.654** (+0.078); no brasileiro
   (`br_only` all) **0.248 → 0.335** (+0.087). O backbone é genuinamente melhor.
2. **Não-degradação global do modelo calibrado:** o lead do v11 (M5_v2) tem `global_nonbr` MCC
   **0.626** vs **0.512** do lead do Pedro (M5_v3) — **+0.114**. **O v11 regionaliza sem sacrificar
   o resto** — a maior melhora *prática*.
3. **A falsificação — o salto científico:** o v11 **falsifica** a especificidade ABRAOM
   (p=0.0196, desconto real bate todos os controles); o v10 **não conseguiu** (controles perto).
   Saímos de "consistente com o caveat" para "**caveat resolvido**".
4. **O negativo bem-dimensionado da Fase 3:** o adapter cru não carrega sinal ABRAOM, agora com
   intervalo (**< ±0.03 MCC**) — mais forte que o +0.018/CI-cruzando do v10.

**➖ Onde o v11 ficou um pouco abaixo do v10 (honestidade):**
1. **`br_only` MCC do lead:** v11 M5_v2 **0.574** vs v10 M5_v3 **0.605** (−0.031). Ligeiramente
   abaixo no headline regional.
2. **Recall P/LP e specificity do lead:** 0.405 vs 0.436 (−0.031); 0.951 vs 0.959 (−0.008).
   Diferenças marginais.
3. **A guarda molecular (M5_v3) não reproduz como lead:** no v10 era o candidato final; no v11 o
   pipeline a barrou (`hold_current_lead`), porque o backbone mais forte a faz criar
   falso-positivos em benignas (§7 Fase 4). **Não é regressão de qualidade — é uma diferença de
   mecanismo** que a replicação revelou.

**🔬 O que a replicação nos permitiu DESCOBRIR (além de reproduzir):**
- **O gap brasileiro↔global persiste apesar de um backbone melhor** (§10.4): +0.09 nos dois, gap
  ~inalterado → a lacuna é de **calibração**, não de representação.
- **O valor do ABRAOM não é aprendível da sequência, mas a AF observada É ABRAOM-específica** —
  as duas metades da tese, agora com **falsificação positiva** fechando o caveat do v10.
- **A guarda molecular é dependente do backbone** — ajuda modelos molecularmente fracos (v10),
  atrapalha os fortes (v11). Insight novo sobre a interação backbone × calibração.

### 10.8 A história em uma frase

O v11 elevou a qualidade molecular **e** a brasileira (~+0.09 cada) e **regionaliza sem degradar
o global** (melhor que o v10 nisso), mas **não fecha o gap** brasileiro↔global — porque o valor
regional do ABRAOM **não é sinal aprendível pela sequência** (adapter/gate falsificados como
nulos), e sim **calibração pela AF observada** — que no v11 é **comprovadamente ABRAOM-específica**
(falsificação p=0.0196), o que o v10 não conseguiu demonstrar. As quatro frentes convergem:
**Fase 1** (adapter genérico), **B** (resíduo tênue), **Fase 3+5** (fusão nula, < ±0.03),
**Fase 4** (calibração real *e* falsificada). **A frente de replicação está completa** — o que
tentamos *depois* dela, para melhorar o resultado, está na §10.9.

### 10.9 Pós-porte — as duas alavancas de melhoria (negativos medidos)

Fechado o porte, o objetivo virou **aumentar o `br_only` mesmo pagando em global**. Duas alavancas
independentes, ambas fechadas com negativo bem-dimensionado (o detalhe está na §7, "Pós-porte"):

| Alavanca | O que era | Resultado | Por quê (medido) |
|---|---|---|---|
| **B** — re-tunar o desconto | reler o grid de calibração afrouxando o piso de recall | **SATURADO** | a fronteira `br_only ↔ recall` é fixa e o global nem se move (0.635 constante); o "teto" de `br_only` **0.698** custa recall **0.075** (perde 92% das patogênicas) |
| **A** — "A-guarda": guarda por cabeças nativas | trocar a guarda molecular por **conservação phyloP + missense-severity ESM-2** (sinais do v11 ortogonais à frequência) | **REFUTADA** | as medianas se sobrepõem (phyloP100 **0.357** × **−0.183**; missense 4.24 × 3.68); enrichment de pico **5.4×** contra base rate **61:1** → ainda ~11 benignas guardadas por founder |

**O achado científico que os dois negativos produzem juntos:**

> **Nenhum sinal molecular disponível — nem o classificador molecular treinado, nem a conservação
> evolutiva, nem a severidade missense destilada do ESM-2 — distingue uma founder patogênica
> brasileira de uma benigna comum no Brasil.** Para todos eles, essas variantes **se parecem**.

Isso tem uma consequência forte e específica: **o gargalo não é o modelo.** Um backbone melhor já
foi testado (v10 → v11: +0.087 no brasileiro, §10.4) e não fechou o gap; mais calibração está
saturada; e os sinais moleculares ortogonais não separam as classes. **O que falta é rótulo, não
representação** — e é isso que torna a §11 uma conclusão nossa, não uma herança.

**Um subproduto que fecha uma pergunta da Fase 4.** Medindo a guarda, descobrimos que a guarda
molecular original dispara para **6 founders e 83 benignas comuns** (~14× mais benigna que founder).
Isso **explica quantitativamente** por que o `M5_v3` do Pedro produziu 208 FPs para 19 resgates no
v11 (§7, Fase 4) — a hipótese registrada lá ("a cabeça molecular do v11 super-estima benignas
comuns") passou a ter número.

---

## 11. O que falta

**A frente de regionalização está COMPLETA — em dois sentidos.** A **replicação** fechou (Fases 0-5:
o headline calibrado e a falsificação positiva estão prontos) **e as melhorias de modelagem estão
esgotadas** (as duas alavancas testadas, ambas com negativo medido — §7 Pós-porte, §10.9). Não
paramos por falta de tempo ou de ideias: paramos porque **medimos que as ideias disponíveis não
funcionam, e por quê**.

**O único caminho de ganho real que resta:**

- **Curadoria externa de variantes P/LP brasileiras.** É o gargalo, e agora com **duas
  investigações independentes** apontando para ele: o Pedro no v10 (`do_not_train_next`) e nós no
  v11, com evidência medida (§10.9). Segue ***gated* por dados** — depende do Eduardo e de
  colaboradores, não de compute. A falsificação nos dá validação **científica** (o desconto é
  ABRAOM-específico); a curadoria é o que faltaria para validação **clínica**. A distinção é
  importante e deve ser mantida na escrita.

**Itens opcionais, de baixa prioridade (nenhum muda a conclusão):**

- **Re-run do Experimento B** selecionando o checkpoint por **Spearman** em vez de NLL (degenerada
  para o alvo ilimitado `delta_logit`) — refino de um número já reportado, não bloqueio.
- **Controles que pulamos do roster do Pedro:** `M2` (gnomad-only) e `M6` (frequência-explícita
  alternativa). Fecham o roster por completude; a conclusão não depende deles.

**O que saiu desta lista na iteração 4 (importante para quem leu a versão anterior):** a iteração 3
listava aqui *"refinamentos v11-nativos (cabeças de conservação/missense-severity) como ablações
futuras"*. **Isso foi feito — e refutado** (§7, Alavanca A). Deixou de ser trabalho pendente e
virou **resultado**: as cabeças nativas não separam founder de benigna comum. Não vale re-tentar
pelo mesmo caminho.

**Entregável científico:** este documento + o `RESULTADOS_REGIONALIZACAO_V11_FASES2-4.md` são a
base pronta para a escrita da dissertação — agora incluindo os negativos do pós-porte, que são o
que sustenta a recomendação final.

---

## 12. Glossário

- **Variante / SNV:** diferença pontual no DNA; SNV = troca de uma base (A/C/G/T).
- **Patogenicidade (P/LP × B/LB):** o rótulo clínico; nossa tarefa de classificação binária.
- **AF (frequência alélica):** quão comum a variante é numa população (0 a 1). Alta ⇒ tende a
  benigna.
- **gnomAD:** base de AF global, majoritariamente europeia (NFE).
- **ABRAOM:** base de AF brasileira (coorte SABE).
- **ClinVar:** base de variantes classificadas por especialistas (o "gabarito").
- **Foundation model / backbone:** modelo grande pré-treinado (Beat-v11) que representa o DNA.
- **Hidden state:** o vetor (448 números) que o backbone produz por posição.
- **LoRA:** técnica de adaptar um modelo congelado com pequenas matrizes treináveis.
- **Adapter:** um conjunto de matrizes LoRA treinado para uma tarefa (ex.: prever uma AF).
- **M0:** baseline de patogenicidade molecular (ClinVar não-brasileiro).
- **Fusion (M4/M5):** combina M0 + adapters de frequência via um gate (peso por adapter).
- **Gate:** o "portão" que pondera os adapters; estático (peso fixo) ou dinâmico (por exemplo).
- **Calibração (M5_v2/v3):** ajuste do score usando a **AF observada** (desconto + guarda
  molecular). É o que de fato regionaliza.
- **Controle scrambled:** adapter/fusão com frequência embaralhada; define o piso e falsifica.
- **MCC / AUROC / AUPRC / specificity / recall / Spearman:** métricas (ver §2).
- **Slices:** subconjuntos de avaliação (`br_only`, `abraom_common_benign`,
  `abraom_pathogenic_present`, `global_nonbr_no_abraom`).
- **Regime A:** o caminho de embeddings *two-tower* (ref/alt → cabeça) usado no pipeline.
- **r1:** o checkpoint alvo do Beat-v11 (`lumina-beat-v11v5-r1-...`, 52M params, d_full 448).
- **Split (`test` / `holdout` / `all`):** partições de avaliação. `test` e `holdout` são
  subconjuntos (por gene); `all` é o slice inteiro. Para as slices **brasileiras**, `all` é
  legítimo (nada delas esteve no treino) e dá **muito mais poder** (§7, Fase 5).
- **Vazamento (*leakage*):** avaliar o modelo em dados que ele viu no treino — infla a métrica
  e invalida a conclusão. Foi o que checamos antes de usar o split `all`.
- **Bootstrap pareado:** reamostrar **as mesmas** variantes milhares de vezes e recalcular a
  **diferença** entre dois modelos em cada reamostra → dá o **intervalo de confiança da
  diferença**. Como o ruído comum aos dois se cancela, é bem mais sensível que comparar dois
  números isolados. É a ferramenta central de decisão desta frente.
- **Intervalo de confiança (CI 95%):** faixa que contém o valor real com 95% de confiança. **Se
  cruza 0**, não podemos afirmar que existe efeito. **CI estreito e centrado em 0** = negativo
  forte ("o efeito é menor que X"); **CI largo cruzando 0** = apenas "faltou poder".
- **Poder estatístico:** a capacidade de **detectar** um efeito que existe. Depende do n. Com n
  pequeno, efeitos reais passam despercebidos **e** ruídos parecem sinais (§9, lição 5).
- **Frequência explícita (o M5):** dar a **AF real como entrada da cabeça**
  (`explicit_feature_columns`). É o que diferencia o M5 do M4 — e é um **modelo treinado**,
  não pós-processamento (§5.11).
- **`RegimeABoundedRegionalHead`:** a cabeça do M5-freq-explícita. Decompõe em
  `molecular_logit` (biologia) e `regional_discount` (desconto por frequência, **limitado por
  arquitetura**), com `regional_logit = molecular − desconto`. É ela que produz os dois números
  que a calibração consome.
- **M5-bounded (cru):** o M5 frequência-explícita **antes** da calibração pós-hoc. Desconta
  agressivo demais → `br_only` alto mas recall P/LP colapsado (0.08). É o **insumo** da
  calibração, não o resultado.
- **M5_v2 / M5_v3:** as duas calibrações pós-hoc sobre os scores do M5-bounded. **M5_v2** = re-tuna
  o desconto (escala/teto/thresholds) no *holdout*. **M5_v3** = M5_v2 **+ guarda molecular**. No
  v11, o lead é o **M5_v2** (o v3 não compensou — ver `hold_current_lead`).
- **Guarda molecular:** o freio que **impede a frequência de descontar** quando a evidência
  molecular é forte (`molecular ≥ limiar`) — protege os P/LP founder/recessivos. É o ingrediente
  do M5_v3.
- **Controles negativos estratificados (falsificação):** embaralhar **quem recebe qual desconto**
  preservando a estrutura (gene / faixa de AF / cromossomo) e re-medir. Se o real bate os
  embaralhados, o efeito é **específico** (não estrutura genérica). No v11: real bate todos os 200
  controles (p=0.0196) → especificidade ABRAOM **falsificada** (o v10 não conseguia).
- **`hold_current_lead`:** a decisão automática do pipeline de calibração quando o candidato novo
  (M5_v3) **não** supera o anterior (M5_v2) dentro das restrições de segurança. Foi o veredito no
  v11 — e é um resultado válido, não uma falha.
- **Founder (variante fundadora / efeito fundador):** variante que se tornou **comum** numa
  população porque estava presente num grupo ancestral pequeno que se expandiu — não porque seja
  inofensiva. É a razão de existirem P/LP **frequentes** no Brasil, e é a classe que a guarda
  molecular existe para proteger do desconto por frequência.
- **Cabeças nativas (do Beat-v11):** as saídas auxiliares que o próprio backbone já traz treinadas
  (conservação, missense-severity, splice, região, etc.), derivadas do mesmo hidden state. O v10
  **não as tinha** — foram a base da tentativa "A-guarda".
- **phyloP / Zoonomia-241:** medidas de **conservação evolutiva** — quão preservada uma posição do
  DNA está entre espécies. Posição muito conservada ⇒ mudança ali tende a ser deletéria. No v11
  vêm da cabeça `conservation_scalar_pred` (phyloP100 / Zoonomia-241 / phyloP470).
- **Missense-severity (ESM-2 destilado):** predição de **quão danosa** é a troca de aminoácido,
  destilada de um modelo de proteína (ESM-2). No v11 é a cabeça `missense_severity_pred`.
- **A-guarda:** a tentativa (pós-porte) de trocar a guarda molecular do `M5_v3` por uma guarda
  baseada em **conservação + missense-severity** — sinais ortogonais à frequência. **Refutada**: os
  sinais não separam founder de benigna comum (§7 Pós-porte, §10.9).
- **Fronteira `br_only` ↔ recall (saturação):** o trade-off fixo do desconto de frequência — subir o
  MCC brasileiro só é possível **descendo** o recall P/LP. O desconto **desliza** nessa fronteira,
  mas não a **levanta**; levantá-la exigiria discriminação nova (que a A-guarda tentou e não achou).
- **Base rate:** a proporção "de fundo" entre as classes (aqui, **61 benignas comuns para cada
  founder**). Um sinal só é útil se o **enrichment** que ele produz superar a base rate — 5.4× contra
  61:1 ainda deixa ~11 falsos por acerto (§9, lição 9).
- **Enrichment:** quantas vezes um filtro concentra a classe de interesse em relação ao acaso. Alto
  enrichment **não** implica filtro útil: depende da base rate contra a qual ele opera.

---

*Fim da iteração 4 (2026-07-23). Estado: **frente de regionalização COMPLETA nos dois sentidos** —
a **replicação** fechada (Fases 0, 1, B, 2, 3, 5 e **4**) **e as melhorias de modelagem esgotadas**
com dois negativos medidos (§7 Pós-porte, §10.9). Lead do v11 = **M5_v2** (`br_only` 0.24 → 0.57 via
calibração, sem degradar o global). A especificidade ABRAOM está **falsificada de forma positiva**
(p=0.0196) — superando o caveat que o v10 do Pedro deixou aberto. **O próximo ganho real depende de
curadoria externa de P/LP brasileiros** (gated por dados), **não de mais modelagem** — conclusão
agora sustentada por evidência nossa, não só herdada. Documento-irmão com os números completos:
`RESULTADOS_REGIONALIZACAO_V11_FASES2-4.md`.*
