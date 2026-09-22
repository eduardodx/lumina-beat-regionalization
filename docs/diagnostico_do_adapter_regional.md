# Adapter regional no R03: o que os pilotos mostram e o que ainda não mostram

Escrito em 2026-09-22, depois dos pilotos 1–5 e da corrida de 1.000 passos. **Revisto no mesmo dia** depois de
duas revisões externas, que acharam um bug no bootstrap e afirmações fortes demais. Cruza quatro fontes: o
**documento de regionalização do Eduardo**, os **resultados da v11**, a **arquitetura do R03**
(`lumina/models/`) e o **nosso desenho atual** (`eval/adapter/`, `scripts/train_population_adapter.py`).

**A v11 é referência, não evidência sobre o R03.** Outro backbone, outro objetivo (regressão de AF), cabeça
treinável e lote diferente. O que vem de lá serve como hipótese a testar ou lição de método.

---

## Três perguntas diferentes

| Pergunta | Comparação que responde | Onde estamos |
|---|---|---|
| O adapter melhora a reconstrução mascarada? | R03 antes × depois, na validação populacional separada por loco | **avançamos aqui** |
| A adaptação mista melhora a classificação nos estudos brasileiros? | **M0 × MR**, com avaliação clínica congelada | depende do G3 |
| O componente ABraOM acrescenta algo ao global? | **MG × MR**, mesma receita, orçamento e avaliação | adiada |

Este documento trata da primeira. Melhora de MLM mede reconstrução na distribuição de janelas que construímos: pode
vir de contexto de sequência, de probabilidades mais bem calibradas ou de adaptação à própria receita. **Não
identifica, sozinha, aprendizado populacional**, nem diz se a representação ajuda a cabeça clínica.

---

## O que foi corrigido nesta revisão

| Afirmação anterior | Correção |
|---|---|
| "a taxa 20× maior causava o mínimo barato — CONFIRMADO" | a comparação mudou taxa, número de atualizações e repetição de dados ao mesmo tempo; **a causa não foi isolada** |
| "sem sobreajuste; a curva ainda descia" | **não houve degradação relatada e o melhor ponto avaliado foi o último**; a curva não foi vista (está em `treino_do_adapter.json` no notebook) |
| a troca de sinal do termo de escolha mostra que o adapter "soube qual alelo" | é **compatível com suavização** e o termo **não** é invariante a temperatura; a ordem do ALT entre as alternativas passou a ser medida |
| "o experimento decisivo da v11 (resíduo) não está sendo feito" | a v11 **fez** um resíduo, com AF **observada**: sinal fraco. A versão com a cabeça nativa é outra proposta, nunca rodada, e "isola exatamente" era forte demais |
| a cabeça populacional foi "supervisionada com peso 256" | **256 é `_DEFAULT_MIN_VALID`** (contagem mínima para o normalizador EMA), não peso. O pacote de inferência não traz as perdas de treino |
| "as camadas 8 e 17 estão inteiramente fora do laço" | a `strided_attn` **roda sempre**, alimentada por camadas adaptadas; faltam só LoRA direto nessas projeções e o caminho de âncoras |
| "contexto longo não ajuda; 1.024 dariam 4× mais passos" | evidência de **regressão de AF** na v11, sem garantia de aceleração; não transfere automaticamente ao MLM do R03 |
| o chr8 como diagnóstico do adapter | depende da **decisão E**; se o chr8 for teste final, não pode ser consultado repetidamente para escolher a receita |
| `bootstrap_do_delta` pronto para ler | **tinha um bug de pareamento**, e o detalhe final era descartado quando o último passo caía na cadência (corrigidos) |

---

## 1. A receita da v11 melhorou o MLM; a causa não foi isolada

| Corrida | Treino disponível | Atualizações | LR | `focal_val` |
|---|---:|---:|---:|---:|
| linha de base (R03 sem delta) | — | 0 | — | 1,7399 |
| `escala` | plano completo | 20 | `1e-4` | 1,7305 |
| piloto 5 | **400 exemplos** (mediana de 6 visitas) | 300 | `1e-4` | melhor 1,7079 no passo 89; final 1,9991 |
| `g4_lr5e6` | plano completo | **1.000** | **`5e-6`** | **1,6932** (melhor = último avaliado) |

Todas na mesma validação de 160 janelas (64 ABraOM + 96 global).

**O que se sustenta:** a receita com `5e-6`, 1.000 atualizações e dados sem repetição reduziu a perda focal em
**−0,0467** sem degradação relatada. Pelo critério primário declarado (`focal_alt`), é resultado positivo de
reconstrução mascarada nessa validação.

**O que não se sustenta:** atribuir a diferença à taxa. Contra o `escala`, mudaram taxa (20×) e atualizações (50×);
contra o piloto 5, mudaram também os dados (400 exemplos repetidos contra o plano inteiro). E a v11 justifica
**testar** `5e-6`, não fixá-la: lá eram outro backbone, regressão de AF, cabeça treinável e lote efetivo 16.
`5e-6` fica como **configuração candidata**.

### Os diagnósticos da mesma corrida

| | ABraOM | global |
|---|---:|---:|
| termo de massa | −0,0441 | −0,0412 |
| termo de escolha | −0,0087 | −0,0015 |
| entropia no focal | **+0,025** | **+0,018** |
| fração média do ALT entre as não-referência | **−0,0037** | **−0,0027** |
| `p_alt` médio | −0,0001 | +0,0005 |
| perda nas posições de referência | **+0,0013** | **+0,0030** |

Com `1e-4 × 20` o termo de escolha **subia** (+0,0017 / +0,0009); aqui ele **cai**. Mas a saída ficou mais plana, a
fração média caiu e a referência piorou — e suavizar uma distribuição confiante demais **baixa o termo de escolha
sem mudar a ordem entre as alternativas**. Exemplo conferido: frações 0,90 e 0,02 com temperatura 1,5 viram 0,77
e 0,06; a fração média cai de 0,46 para 0,42, o `−log` médio de 2,01 para 1,52, e a alternativa preferida não
muda em nenhuma das duas.

Então o termo de escolha não separa "discriminou melhor" de "suavizou". O runner agora mede também a **ordem** do
ALT entre as três não-referência (`alt_em_primeiro_entre_nao_ref`, `posto_do_alt_entre_nao_ref`), que temperatura
e massa tirada da referência não mexem. São **diagnósticos, não portões**:

- a ordem também muda por alteração genérica ou prejudicial, então mudança de ordem não prova aprendizado
  populacional;
- probabilidade melhor sem mudar a ordem é ganho legítimo de uma regra de pontuação própria;
- e nada disso diz o que aconteceu com a representação que a cabeça clínica vai ler.

Um **controle de temperatura** (um escalar ajustado no R03 congelado, com dados de desenvolvimento separados da
validação) diria quanto do −0,047 um ajuste trivial reproduz **nessa métrica**. Não provaria que o adapter mudou
só por calibração. Fica como diagnóstico opcional.

Também: o "5,9× mais no ABraOM" vinha de 64 × 96 posições, sem IC. Mesmo com IC, a diferença entre as fontes
compara amostras distintas e **não substitui o MG × MR**; e um IC cruzando zero não prova que o efeito era ruído.

---

## 2. O resíduo: hipótese para depois, não bloqueio

`RESULTADOS_REGIONALIZACAO_V11.md` (10/07) propôs treinar no resíduo `af_abraom − f(gnomad_af_pred)`. **Dois dias
depois a v11 rodou uma versão dele** (Experimento B, `TCC_REGIONALIZACAO_V11.md` e §7.3 do
`HANDOFF_CONTINUACAO_V11_POS_FASE4.md`), com AF **observada** dos dois lados:

- alvo `logit(af_abraom) − logit(af_gnomad)`, perda huber;
- ganho contra o controle embaralhado **+0,026 [+0,001; +0,050]** no teste; na validação o IC cruza zero;
- a seleção de checkpoint por NLL era degenerada para esse alvo ilimitado;
- leitura da época: sinal regional **fraco**.

A versão com a **previsão nativa** do modelo nunca foi rodada, e não é o mesmo experimento. Três cuidados antes de
propô-la:

- a `population_af_head` prevê **log-AF** ("observed + log-AF", `model.py:171`): subtrair a saída dela de AF bruta
  exige definir transformação e calibração;
- o resíduo não isola "exatamente" o componente regional: carrega também erro do preditor global, cobertura,
  filtragem e ruído de frequência;
- é **outra tarefa de treino**. Trocar o MLM por ela agora adiaria a pergunta da campanha.

---

## 3. M0 × MR responde uma pergunta; MG × MR, outra

O documento do Eduardo define o contraste confirmatório como M2 × M1 (ABraOM contra global, mesmo orçamento). A
campanha, por decisão dele em 15/09, é M0 × MR com um adapter misto. Isso **não torna o desenho errado; limita a
conclusão**: um ganho sustenta "a adaptação mista ajuda", não "o ABraOM ajuda além do global".

"Se der certo a gente volta e tenta explicar" define a **prioridade**. Não autoriza a execução nem o orçamento do
MG, que continua sendo a **ablação de atribuição proposta**.

---

## 4. Avaliação representacional: depende da decisão E

A §11.1 do documento chama de indispensável `Spearman(score populacional, log10 AF)` no chr8, e o **Cenário F**
(ganho em ClinVar sem ganho representacional = calibração, não regionalização) continua sendo o aviso certo.

Mas como adaptar isso a esta campanha está na **decisão E, ainda aberta**. Se o chr8 continuar reservado como teste
final, **não pode ser consultado repetidamente para escolher a receita**. Os diagnósticos de desenvolvimento usam a
validação do próprio adapter, separada por loco.

---

## 5. O objetivo tem um atalho, e os números mostram parte dele

Toda janela tem o focal não-referência por construção, e na entrada todas as posições mascaradas são iguais
(`MASK`). Reduzir a confiança em todas elas derruba a perda focal a custo pequeno nas posições de referência.

Com `1e-4` o ganho veio todo do termo de massa. Com `5e-6` as posições de referência pioraram (+0,0023) e a
entropia subiu: **parte** do ganho é compatível com esse atalho. Os spans de referência funcionam como detector,
não como impedimento. É propriedade do objetivo a monitorar, não falha demonstrada.

---

## 6. Contexto de 4.096 bp: hipótese de custo, sem prioridade

Na v11, contexto 4.096 ≈ 1.024 em Spearman **e em custo** (carga dominada por overhead) — na regressão de AF. Isso
não mostra que contexto longo é inútil no MLM do R03, nem garante aceleração com janelas menores. Só vale testar se
o custo virar gargalo.

---

## 7. Camadas 8 e 17: participam do cálculo, sem adaptação direta

Nas duas camadas de atenção esparsa, a `strided_attn` **executa em todo forward**, antes da condição que pula as
âncoras (`backbone.py`, `SparseGlobalAttention.forward`), e recebe entradas de camadas adaptadas. O que falta:

- LoRA direto nas projeções dela (o embrulho em `nn.MultiheadAttention` é inerte e foi removido da superfície);
- o caminho de âncoras (`anchor_query_attn` / `anchor_key_attn`), que só liga com `edit_mid_mask`.

Isso **não demonstra incompatibilidade entre o R03 e o MLM**. Declarar a variante na entrada mudaria a tarefa (é o
que o modelo deveria inferir) e exigiria definição cuidadosa.

---

## 8. O bootstrap: dois defeitos corrigidos antes de qualquer leitura

1. **Pareamento.** `variant_id` é `chrom:pos:ref:alt`, sem a fonte, e o mesmo alelo pode estar nas duas metades da
   mistura com janelas diferentes. Parear só por ele fazia um registro sobrescrever o outro: com antes e depois
   **idênticos**, o delta do ABraOM saía +0,25. Agora a chave é `fonte|variant_id|focal_index`; janela repetida,
   loco ausente ou janela que muda de loco **recusam** o bootstrap; e a saída informa **janelas e locos por
   fonte**. Loco ausente não cai mais para o `variant_id`: a validação sem `locus_id` para antes de carregar o
   modelo.
2. **Detalhe final descartado.** O detalhe por janela era removido em toda validação da cadência, inclusive na
   última. Com `--passos` múltiplo de `--validar-a-cada` (a corrida de 3.000/250), o bootstrap do final sairia
   "indisponível". Agora o detalhe do final e o do melhor são guardados, há bootstrap também para o **melhor** (o
   adapter que se usa), e os três conjuntos vão para `detalhe_da_validacao.json`.

O IC continua **condicional ao modelo escolhido nessa mesma validação** e não inclui variação entre sementes.

---

## O que fazer, em ordem

1. **Feito:** bootstrap corrigido, ordem do ALT medida, textos de leitura corrigidos.
2. **A corrida de 3.000 passos com `5e-6`**, como desenvolvimento do adapter candidato. Critério primário
   inalterado; termo de escolha, ordem e contraste entre fontes como diagnósticos, sem perseguir significância.
3. **Completar o G3 para medir M0 × MR.** Não é preciso resolver o mecanismo do MLM antes de medir se o adapter
   ajuda na tarefa clínica. Antes de comparar sistemas, **declarar a regra que congela o adapter**, para a
   comparação clínica não virar seleção de adapter.
4. **Para o Eduardo:** o MG como ablação de atribuição proposta; a decisão E (chr8); o resíduo como hipótese
   posterior, distinguindo AF observada de previsão nativa.

**O que não mudaria agora:** os pesos da loss e o critério primário.
