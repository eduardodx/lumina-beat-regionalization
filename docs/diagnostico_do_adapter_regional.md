# Por que o adapter regional está com desempenho fraco — diagnóstico

Escrito em 2026-09-22, depois dos pilotos 1–5 e da medição de escala. Cruza quatro fontes: o **documento de
regionalização do Eduardo**, os **resultados da v11** (`RESULTADOS_REGIONALIZACAO_V11.md`), a **arquitetura do
R03** (`lumina/models/`) e o **nosso desenho atual** (`eval/adapter/`, `scripts/train_population_adapter.py`).

O que medimos até aqui: com o conjunto de treino inteiro, a perda focal melhora **−0,0093**, e a decomposição
mostra que a melhora vem **inteira do termo de massa** (−0,011) enquanto o **termo de escolha piora** (+0,002).
Ou seja, o adapter aprendeu a desconfiar da base de referência e **não** aprendeu qual variante está ali.

Abaixo, sete achados, do mais acionável ao mais estrutural. Três deles são problemas nossos; quatro são
limitações de desenho que precisam de decisão.

---

## 1. A taxa de aprendizado é 20× a da receita que funcionou [ACIONÁVEL, testável numa corrida]

A v11 treinou adapters de frequência sobre o mesmo tipo de backbone congelado, com **a mesma superfície LoRA**
(o doc diz "105 lineares LoRA-ados: mamba `in/out_proj` + attn `q/k/v/out_proj` + `up_stages.gate` +
`stem.purity`; zero cabeças" — exatamente os 105 que encontramos antes de remover os 6 inertes) e o mesmo rank 8
/ alpha 16.

| | v11 (`train_abraom_frequency_adapter.py`) | nós |
|---|---|---|
| LR dos parâmetros LoRA | **5 × 10⁻⁶** | **1 × 10⁻⁴** |
| LR da cabeça | 5 × 10⁻⁴ | não há cabeça |
| lote efetivo | 2 × 8 = 16 | 8 |
| passos | 1.000 | 300 (piloto) |

**Estamos empurrando o delta do backbone 20 vezes mais forte do que a receita que produziu sinal
sequência→frequência real.** E o que observamos — ganho todo no termo de massa, sobreajuste em 89 passos — é
exatamente o que uma taxa alta demais produz: o otimizador chega ao mínimo mais barato (um viés global contra a
base de referência) antes de aprender qualquer coisa sutil.

**Custo de testar:** uma corrida (~60 min). É a primeira coisa a fazer.

---

## 2. O experimento que a iteração anterior identificou como DECISIVO não está sendo feito [DESENHO]

`RESULTADOS_REGIONALIZACAO_V11.md`, seção "próximos passos", item 1:

> Treinar o adapter no **resíduo** `af_abraom − f(gnomad_af_pred)`, usando a cabeça de população nativa
> (`population_af_head`, que já prediz AF gnomAD por posição). Isola **exatamente** o componente regional — mais
> limpo que o A_BR-vs-A_gnomAD indireto. (…) é o experimento que **decide** a pergunta regional.

Nós trocamos isso por MLM, que **não isola nada**. E a razão de o resíduo ser decisivo está medida na própria
v11: `A_BR − A_gnomAD` = +0,016 / +0,010 com **IC cruzando zero**. Treinar na AF brasileira **não** bate treinar
na AF global, quando as duas são avaliadas na AF brasileira. O componente regional, se existe, é o que sobra
depois de descontar o global — e é isso que o resíduo isola.

O R03 tem a peça necessária pronta: `population_af_head = Linear(d_full, 4)`, saída `[B, L, 4]`, uma AF prevista
por base alternativa e por posição. Foi supervisionada no pré-treino com peso **256** (`losses.py`).

---

## 3. O desenho atual não pode responder à pergunta principal do documento [DESENHO]

O documento do Eduardo define a pergunta assim:

> Um adapter treinado com variação populacional brasileira (…) melhora a classificação (…) **mais do que um
> adapter de variação humana global com a mesma arquitetura e o mesmo orçamento**?

E o contraste confirmatório é **M2 × M1** — ABraOM contra global. Não M0 × MR.

Nossa campanha tem **um único adapter misto 60/40** contra nenhum adapter. Mesmo um resultado perfeito responde
"adaptação populacional ajuda?" e **não** "adaptação *brasileira* ajuda mais que a global?". Separar H0 (efeito
genérico) de H1 (efeito regional) exige o braço global puro — o **MG**, mistura 100/0 — que foi adiado.

Isso foi decisão do Eduardo ("vamos tentar pular etapas") e o custo foi declarado na época. Mas agora que o sinal
está fraco, vale reencarar: **sem MG não existe o comparador que torna o resultado interpretável.** E o MG é a
mesma receita com outra mistura — uma execução, sem código novo.

---

## 4. Falta a avaliação representacional, que é o que o protocolo manda medir [LACUNA]

Seção 11.1 do documento, marcada por ele como **"o componente adicional que considero indispensável"**:

> Score: usar o score escalar produzido pela **interface populacional** do Lumina ou do adapter. (…) Métrica
> representacional mínima: `Spearman(score, log10(AF + ε))`.

Nosso critério primário é **entropia cruzada de reconstrução mascarada**. É outra coisa. A avaliação
representacional pergunta "o adapter ordena variantes por frequência melhor do que antes, e mais na fonte em que
foi treinado?" — que é a pergunta da regionalização.

E o documento antecipa exatamente o nosso risco, no **Cenário F**:

> M2/M4 melhoram ClinVar mas não melhoram ABraOM-chr8 → o ganho pode vir de regularização, calibração ou
> compatibilidade com a distribuição do ClinVar, e **não de regionalização representacional demonstrada**.

Sem essa métrica não temos como distinguir os dois, e ela é barata: o chr8 está reservado, o pool do ABraOM tem
61.737 variantes lá, e o score sai de uma cabeça que já existe.

---

## 5. O objetivo MLM tem uma solução degenerada, e nossos números mostram o modelo a tomando [NOSSO]

**Toda janela de treino tem a base focal não-referência, por construção.** Então existe uma política que reduz a
perda focal sem aprender nada sobre populações: baixar globalmente a probabilidade da base de referência em
posições mascaradas.

Os spans de referência existem como guarda, e têm 3,3× mais peso agregado que o focal (1.055 × 0,5 contra
160 × 1,0). Mesmo assim o modelo fez a troca — porque o ganho **por posição** no focal é muito maior: ali se pede
justamente aquilo que o modelo pré-treinado em referência considera improvável.

A decomposição confirma: **termo de massa −0,011, termo de escolha +0,002**. A guarda está funcionando como
**detector**, não como impedimento.

---

## 6. Contexto longo não ajuda sinal populacional, e nós pagamos por ele [EFICIÊNCIA]

Achado 5 da v11:

> ctx 4096 ≈ ctx 1024 (test 0,1273 vs 0,1274), a custo ~zero. (…) é **ausência de sinal de longo alcance** para
> frequência — biologicamente plausível: AF é local (constraint / conservação / mutabilidade).

Nossas janelas são de 4.096 bp e custam **0,140 s por exemplo**. Se o sinal populacional é local, janelas de
1.024 dariam aproximadamente **4× mais passos pelo mesmo tempo** — o que importa quando o suspeito número 1 é
treino insuficiente ou mal dimensionado.

Ressalva: o objetivo da v11 era regressão de AF e o nosso é MLM; a transferência do achado não é automática. Mas
é barato testar.

---

## 7. O caminho de variante da arquitetura fica completamente fora do treino [ESTRUTURAL]

O R03 tem uma via dedicada a "aqui foi aplicada uma variante": o `variant_edit_mask`, o `conservation_delta` (só
existe com edição declarada) e, nas camadas **8 e 17**, as `anchor_query_attn` / `anchor_key_attn`, cujo papel é
**difundir o efeito de uma edição ao longo da sequência**.

Sem `edit_mid_mask`, `backbone.py` pula esse caminho e só toca os parâmetros com magnitude zero para o DDP ver um
conjunto constante. Foi o que medimos: gradiente exatamente zero ali.

Some-se a isso que o LoRA nessas camadas é **inerte** (elas usam `nn.MultiheadAttention`, que lê `out_proj.weight`
em vez de chamar o módulo) e o quadro fecha: **as camadas 8 e 17 estão inteiramente fora do nosso laço de
treino**, pelos dois motivos ao mesmo tempo.

Isso **não** é um erro a corrigir sem pensar: em MLM não se pode dizer ao modelo onde está a variante — seria
vazamento, já que é exatamente o que se pergunta. Mas registra uma incompatibilidade real entre o objetivo
escolhido e a arquitetura: o R03 tem maquinário de efeito-de-variante que o MLM não aciona, e o adapter não pode
ajustá-lo.

---

## O que eu faria, em ordem

**Primeiro, o barato e decisivo sobre a dinâmica:** uma corrida com `--lr 5e-6` (a receita da v11) em vez de
`1e-4`. Se a solução degenerada sumir e o termo de escolha começar a se mover, o problema era otimização, não
desenho. Uma hora de GPU.

**Segundo, a métrica que falta:** Spearman entre o score populacional e a AF observada, por fonte, nas variantes
de chr8 que nenhum treino viu. É o que o protocolo manda medir, distingue regionalização de calibração, e usa
uma cabeça que já existe no modelo.

**Terceiro, para o Eduardo, duas perguntas de desenho:**

1. **O resíduo.** A iteração anterior apontou `af_abraom − f(gnomad_af_pred)` como o experimento que decide a
   pergunta regional, e o MLM não o substitui. Vale rodar os dois, ou trocar?
2. **O comparador.** Sem o braço global puro (MG, mistura 100/0), M0 × MR não separa "adaptação populacional
   ajuda" de "adaptação brasileira ajuda mais". É uma execução com a receita que já existe.

**O que NÃO mudaria agora:** os pesos da loss. O piloto não isolou problema neles, e mexer junto com a taxa de
aprendizado impediria saber qual mudança fez efeito.
