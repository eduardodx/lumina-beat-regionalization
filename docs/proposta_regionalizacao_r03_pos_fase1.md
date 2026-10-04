# Proposta de regionalização do R03 após a Fase 1

Data: 04/10/2026. Rascunho para discussão com Eduardo; não autoriza uma corrida de adaptação.
Base: resultados reais registrados em [fase1_r03_congelado.md](fase1_r03_congelado.md), diagnóstico e inventário
da revisão `6a55704`, registrados em `8ef3bc5`. Nenhum experimento novo foi executado para escrever esta proposta.

**Proposta:** preservar a combinação do R03 com frequência global como referência, corrigir primeiro como a
informação regional é interpretada e, em seguida, testar se um objetivo populacional explícito consegue ensinar
essa informação à representação. Não repetir o MLM misto da campanha anterior como experimento principal.

Já temos evidência suficiente para discutir essa proposta. Ainda não temos evidência de que ela produzirá
regionalização efetiva, nem certificamos ausência de degradação nos demais estudos do Mosaic.

## 0. Números-chave da Fase 1

Teste das cinco execuções do core de 4 kb e recortes brasileiros, com IC de 95% por bootstrap (1.000 réplicas).
Delta = braço novo − braço base.

| Leitura | Valor |
|---|---|
| Núcleo, macro AUROC (avaliador oficial) | F 0,910 · E 0,913 · E+F 0,9765 · E+F+BR 0,9772 |
| Núcleo, F → E+F (macro AUPRC, contraste oficial) | +0,111 [0,066; 0,159] |
| Missense, AUROC | E 0,828 · E+F 0,948 · REVEL 0,948 · AlphaMissense 0,941 |
| Proxy clínico, E → E+F (Δ AUROC, coorte completo) | +0,064 [0,050; 0,078] |
| Populacional/benefício, E → E+F (Δ AUROC) | +0,045 [0,025; 0,084] |
| Proxy clínico, E+F → E+F+BR | +0,0015 [0,0006; 0,0025]; controles +0,0013; interação +0,0003 [−0,0010; 0,0016] |
| Populacional/benefício, E+F → E+F+BR | +0,0027 [0,0006; 0,0061]; interação no populacional −0,0005 [−0,0043; 0,0039] |
| P-BR (1.050), E+F → E+F+BR | 21 perdidas, 11 ganhas; limite superior 0,052, acima da margem de 0,01 |
| Campanha anterior (Mosaic antigo): adapter rsLoRA com MLM misto, proxy clínico | −0,0013 [−0,0043; +0,0018] |

Os comparadores oficiais são scores congelados; as cabeças da Fase 1 foram treinadas nestes rótulos, então a
linha do missense não é uma comparação de igual para igual.

O coorte completo do `br_population_observed` e a coorte de benefício são as **mesmas 2.057 variantes** (gold
presentes no ABraOM, 98 P). Os deltas pontuais coincidem; só o IC muda, porque a unidade de bootstrap é outra.
Na prática, são dois conjuntos brasileiros de teste: o proxy clínico e esse conjunto populacional, além das P-BR
e das críticas. Não contar as duas leituras como evidências independentes.

## 1. O que aprendemos e o que continua sendo hipótese

E é a leitura de 1.344 dimensões do R03 congelado; F é o bloco oficial de frequência global do gnomAD; BR é o
bloco ABraOM. As cabeças desta fase são regressões logísticas ajustadas nas cinco execuções do core de 4 kb.

| Evidência observada | Consequência para a proposta | Limite da conclusão |
|---|---|---|
| E+F melhora substancialmente a classificação frente a E e F separados | Usar E+F como referência forte | Combinar duas entradas não regionaliza o embedding |
| Acrescentar BR dá ganhos pequenos; a interação dos proxies fica perto de zero e incerta | Não vender o BR atual como solução regional demonstrada | Não exclui uma forma melhor de usar ou aprender informação regional |
| E+F+BR perde 21 P-BR e ganha 11 no ponto de operação declarado; 16 das perdas persistem ao ajustar a especificidade de validação | Não basta trocar o limiar para resolver todas as perdas | O diagnóstico não identifica qual coluna BR causou a reordenação |
| Em E → E+F, o saldo muda ao comparar na especificidade de validação de E: 66 perdidas e 71 ganhas | Separar desempenho de ranking de política de decisão | O saldo positivo não protege cada caso perdido nem prova não inferioridade |
| HbS e HbC recebem chamadas negativas inclusive em E | Investigar a leitura/cabeça além da frequência | Não prova que toda informação necessária esteja ausente do R03 |

O resultado com especificidade ajustada é posterior ao teste. O código escolhe o novo limiar usando benignas
da **validação de cada execução**, para obter especificidade pelo menos igual à da referência. Empates podem
impedir igualdade exata. Isso não garante a mesma especificidade nos dados de teste ou no Brasil, nem substitui
a regra original de retenção de P-BR.

O BR atual já distingue `present`, `ac0`, `no_call`, `not_found`, FILTER PASS e AN/2.342, além de log-AF.
Portanto, o problema não é simplesmente ter ignorado ausência ou denominador. Uma cabeça linear aditiva pode
não representar bem a interação entre contagem, frequência, qualidade e contexto; essa é uma hipótese a testar.
Também podem contribuir correlações entre features, regularização, diferenças entre tiers e composição dos dados.

Entre as 21 perdas de E+F → E+F+BR, 12 têm AF regional abaixo de 0,1%, não todas as 21. Das 109 perdas de
F → F+BR, 102 estão nessa faixa. Com AN válido limitado a 2.342, uma AF positiva abaixo de 0,1% é compatível
com AC de 1 ou 2, mas a análise seguinte deve conferir o **AC registrado**, a consistência AC/AN/AF e a qualidade
por variante. Uma razão regional/global alta com poucas cópias não demonstra sozinha ruído ou enriquecimento real.

HbS e HbC não aparecem no arquivo de exceções BA1 do Mosaic. Essa lista participa da construção da verdade
regional e das regras, não do cálculo do probe linear desta fase. Sua ausência não explica por si as chamadas
negativas. A lista crítica separada as inclui e deve continuar sendo auditada, sem inseri-las como exceções
aprendidas à mão para fazer o benchmark passar.

Os subsets também não representam uma única pergunta: `br_clinical_evidence` identifica participação de
instituições brasileiras, não ancestralidade do portador. Só 19,7% de seus casos têm presença no ABraOM nesta
entrega. `br_population_observed` é definido justamente por essa presença. Um alvo populacional regional pode
ajudar o segundo ou as células de viés e não melhorar o primeiro. Se o foco for o clínico, decompor os erros
por gene, consequência, painel e qualidade de evidência é complementar; não presumir que todo erro brasileiro
seja falta de frequência regional. Corrigir contagens pequenas também não resolve automaticamente HbS/HbC,
que já falham no braço sem frequências.

**Escala dos efeitos.** Nos dois conjuntos brasileiros, a maior melhora medida veio da combinação com a
frequência global (E → E+F: +0,064 e +0,045 de AUROC), não da informação regional (E+F → E+F+BR: +0,0015 e
+0,0027). Isso não é teto para outra forma de usar ou aprender o ABraOM, mas é a escala observada até aqui. Antes
de investir em adaptação, vale decidir com o Eduardo o que conta como sucesso:
- efeitos de AUROC dessa ordem, que são detectáveis aqui (o IC do proxy clínico tem cerca de ±0,001);
- ou segurança e viés, onde a informação regional é estruturalmente relevante.

Melhorias globais que também levantam os subsets brasileiros fazem parte do mesmo objetivo. São exemplos a
integração com a frequência e o missense, onde E sozinho fica em 0,828.

## 2. Primeiro experimento: interpretar melhor BR sem nova extração

**Pergunta:** o benefício pequeno e as perdas decorrem, em parte, da forma como o classificador usa o ABraOM?

Reutilizar E, F, os cinco folds e as purgas atuais. Fazer uma ablação limitada, escolhida antes de ler seus novos
resultados, em vez de procurar indefinidamente uma combinação favorável.

| Braço | Entradas/objetivo | Papel |
|---|---|---|
| Referência | E+F, receita já medida | Preservar o comparador |
| BR original | E+F+BR, receita já medida | Reproduzir o efeito observado |
| BR com suporte amostral | E+F + bloco regional com contagem e incerteza | Testar a hipótese sobre a interpretação regional |

Bloco proposto, com poucas features e tratamento explícito de valores sem medição:

- AC e AN registrados, faixas de AC (incluindo 1 e 2), AF e medida de incerteza de AC/AN;
- estados de observação e qualidade, incluindo a classe CEGH quando disponível;
- evidência de frequência regional alta sustentada pelo limite inferior de confiança, mantendo a incerteza
  separada da frequência pontual;
- contraste regional/global e sua interação com suporte e qualidade. A comparação com `popmax` pode ser uma
  ablação, mas não substitui silenciosamente o `AF_joint` usado para definir as células do Mosaic.

O Mosaic já calcula o limite inferior unilateral de AC/AN em `regional_truth.af_lower_bound` e exige PASS,
`vSR` e suporte de AN em suas regras pertinentes. Reusar essas funções evita outra definição incompatível.
**Limite inferior sozinho não descreve toda a incerteza** e não torna automaticamente segura uma decisão.
Não transformar presença, uma cópia ou uma regra genérica de 1% em benignidade. Se forem usadas features BA1/BS1,
elas devem seguir as regras e exclusões oficiais, com relato da circularidade, não regras inventadas para este teste.

Primeiro ler coeficientes padronizados e remover grupos de features BR de forma controlada. A correção de
contagem pode retirar um sinal útil de variantes realmente enriquecidas; não presumir que aproximar tudo da
frequência global seja a solução.

Se for necessário um MLP ou uma cabeça com interações, repetir E+F e BR original com a mesma arquitetura,
regularização e orçamento. Melhorar o braço regional com uma cabeça mais expressiva, deixando a base linear,
confundiria ganho regional com capacidade. A transformação e a padronização são ajustadas só no treino.

Escolher a receita na validação; conservar o limiar MCC original como referência e declarar uma análise adicional
de sensibilidade/especificidade. Uma restrição de sensibilidade nas P presentes no ABraOM só pode selecionar o
limiar com exemplos de validação e suporte informado; poucos casos não certificam segurança. Esta política de
limiar é um controle de decisão, não uma mudança do embedding.

Os testes desta fase já foram vistos. Uma nova rodada continua sendo **desenvolvimento**, mesmo que suas escolhas
sejam feitas apenas na validação. O sucesso desta ablação orientaria a arquitetura, mas não provaria que um
adapter de sequência consegue aprender o mesmo benefício.

## 3. Proposta de adaptação aprendida, específica ao R03

**Hipótese:** uma correção regional condicionada à informação global, com suporte amostral explícito, é um alvo
mais alinhado à pergunta do que reconstruir bases mascaradas de variantes amostradas por frequência.

O R03 de referência tem 52.124.400 parâmetros. Sua saída por base tem 448 dimensões. A leitura E concatena:
vetor REF no sítio, diferença ALT−REF no sítio e média local de REF, cada um com 448 dimensões. É a mesma leitura
já extraída para 326.818 variantes elegíveis. A receita clínica e a extração devem ser iguais entre os sistemas.

### 3.1 Teste de recuperabilidade antes de atualizar o tronco

Treinar uma cabeça populacional pequena sobre E congelado, em uma amostra do ABraOM dos blocos **expostos**
de 1 Mb. Avaliar em blocos **não expostos**, respeitando o buffer. A tabela regional é alvo de treino, não
entrada da representação que será avaliada como aprendida. Não usar como entradas membership brasileiro,
`bias_cell`, rótulo de avaliação ou lista de críticas.

Uma parametrização candidata é uma correção no logit da frequência:

```text
q_regional = sigmoid(logit(q_global) + g(E))
alvo observado = AC_regional em AN_regional alelos chamados
```

`q_global` usa frequência global observada ou uma referência global calibrada no treino; zeros e ausência têm
tratamentos distintos e declarados. A função `g` estima a correção, não o rótulo patogênico.
Uma perda baseada em AC/AN, com incerteza e qualidade consideradas, é candidata; compará-la com log-AF direto
sob o mesmo orçamento pode revelar se aprender o resíduo ajuda. Um modelo binomial é uma aproximação de trabalho,
não prova de independência dos alelos: dependência e sobredispersão precisam ser consideradas. Não deixar o
denominador maior de uma fonte dominar automaticamente a loss conjunta.

A cabeça nativa `population_af_head` é linear sobre as 448 dimensões REF. Assim, suas quatro saídas no sítio
podem ser recuperadas em CPU a partir do primeiro bloco de E e dos pesos do checkpoint, conferindo layout,
norma e ordem A/C/G/T. Isso pode ajudar a diagnosticar o prior existente sem repetir o forward inteiro.
**Não tratar `gnomad_af_pred` como AF calibrada:** a arquitetura é confirmada, mas a transformação do alvo
e sua calibração precisam ser verificadas antes de usar esse valor em um resíduo.

As saídas da `population_af_head` e da `population_observed_head` no sítio **já estão no cache**, no bloco
`cabecas_172`: o valor na REF e o delta W·Δ, para as 326.818 elegíveis. O diagnóstico do prior nativo (quanto
ele acompanha a AF do gnomAD e a presença no ABraOM) pode começar em CPU, sem o checkpoint.

O teste de recuperabilidade em si **não é só CPU**. Ele precisa de E para uma amostra do ABraOM em blocos expostos
e não expostos, o que pede nova extração em GPU e o extrato do ABraOM da entrega
(`data/annotations/abraom/sabe1171-wgs/`, ainda não baixado). As células de viés não servem como amostra de treino:
só contêm benignas da camada B (1–5%), uma fatia estreita da distribuição de frequências.

Demografia, efeito fundador e processo de amostragem não são determinados só pela janela REF/ALT. O objetivo
é testar a parcela generalizável, não prometer reconstruir a tabela ABraOM em loci novos. Falha de uma cabeça
congelada orienta o orçamento; não é demonstração de impossibilidade de adaptar a sequência.

### 3.2 Alterar a representação com controle de atribuição

Se houver sinal útil, testar rsLoRA no R03, com pesos base e cabeças nativas congelados, e uma cabeça auxiliar
regional própria. Reusar carregamento, superfície efetiva de 99 módulos, testes de gradiente e checkpoints
da campanha anterior; conferir esse contrato contra o checkpoint antes de treinar. Os módulos de atenção
esparsa não recebem LoRA diretamente nessa superfície, mas continuam participando do forward.

Objetivos candidatos:

- alvo populacional regional condicional descrito acima;
- preservação de saídas/representação global em exemplos de treino permitidos, para controlar esquecimento.

Uma cabeça auxiliar pode absorver a tarefa sem tornar o embedding clinicamente melhor. Por isso comparar:

1. R03 congelado + cabeça clínica reajustada;
2. R2c: continuação global + mesma cabeça clínica;
3. R2: adaptação regional + mesma cabeça clínica.

R2 e R2c recebem a mesma capacidade, orçamento de passos/tokens, protocolo de amostragem e separação geográfica
pertinente. O conteúdo/alvo regional é a diferença declarada. Os folds/purgas clínicos continuam valendo;
rótulos usados em algum objetivo auxiliar só podem vir do treino permitido daquela execução.

Uma versão menor, posterior à extração, `E' = E + U·GELU(V·E)`, pode testar uma transformação aprendida usando
o cache. Nesse caso o tronco R03 permanece congelado: é um adapter sobre a leitura, e deve ser identificado assim.
O controle recebe o mesmo módulo e orçamento. É uma alternativa de baixo custo, não uma atualização interna
do R03 nem prova de regionalização por ter mudado os vetores.

O critério é melhorar a utilidade clínica e a generalização em locais não usados no ajuste. Distância entre
embeddings, UMAP ou melhora da tarefa auxiliar, isoladamente, não demonstram regionalização efetiva.

## 4. Como decidir se funcionou

Antes da adaptação, fixar os contrastes e margens relevantes. Não selecionar uma margem depois de ver um resultado.

- **Utilidade regional:** ganho nos subsets/proxies brasileiros, por painel, com o efeito nos controles ao lado;
  interação regional não se confunde com melhora geral. Benefício oficial R1−RW-3 é outra avaliação e ainda
  requer a integração correspondente.
- **Mecanismo:** falsos positivos na célula primária contra a comparável, com pesos e ajuste oficiais. Para R2,
  comparar com R2c nos blocos não expostos. Não pontuar essa verdade construída de AF regional com consulta
  direta à mesma AF e chamar o resultado de generalização aprendida.
- **Segurança:** perdas brutas e ganhos de P-BR separadamente, gold e consensus, além das 13 críticas. Saldo
  positivo não apaga perdas; uma crítica negativa já na base não pode desaparecer do relato. Duas críticas
  estão fora do release e precisam de pontuação externa para uma avaliação completa.
- **Custo global:** core por painel; incluir `gene_transfer` e `time` antes de afirmar “sem degradar os outros
  splits”. A Fase 1 atual não cobre esses dois estudos.
- **Incerteza:** reamostragem na unidade exigida pelo contraste, não variantes vizinhas independentes. O esquema
  exposto/não exposto do R2 precisa ser aplicado pelo consumidor; o CLI geral não garante esse contraste sozinho.

Todo esse corpus já orienta escolhas, conforme a nova frente do Mosaic. Resultados nele são desenvolvimento;
confirmação exige um sistema congelado e dados posteriores que não tenham servido às escolhas.

## 5. O que fazer agora

**Enviar esta proposta para discussão, sem esperar outra extração.** O primeiro experimento sugerido é a ablação
CPU de BR com contagem/qualidade/incerteza, sem recalcular os embeddings. Não estimar “minutos” sem medir os
ajustes: só sabemos que dispensa uma nova extração de GPU.

Se a discussão priorizar redução do viés regional, executar a Fase 1b `main` como diagnóstico específico:
124.194 variantes elegíveis, 6.088 já cobertas, 118.106 novas (~2,1 h de extração na taxa observada, além de
preparação e avaliação). Todas as definições exigem 589.552 novas (~10,4 h). Não são pré-requisito para redigir
a proposta ou para toda hipótese de adaptação.

O inventário usa as funções do avaliador e confere as execuções pedidas, mas apenas conta IDs nos caches.
Não valida automaticamente identidade numérica, coordenadas, janelas ou prontidão para pontuar externas.
Antes de extrair: validar essas entradas, preservar `trained_run` e os pesos oficiais e preparar a rota
`--candidate` com o extrato CADD. As variantes sem execução não devem receber uma cabeça arbitrária.

**Mensagem sugerida para Eduardo:**

> A Fase 1 mostrou que o embedding do R03 e a frequência global se complementam bastante: nos recortes
> brasileiros, essa combinação deu +0,045 a +0,064 de AUROC. Acrescentar ABraOM cru dá ganhos pequenos (até
> +0,003, iguais em casos e controles) e também derruba chamadas de patogênicas, parte delas mesmo após ajustar
> o limiar.
> Isso aponta para testar uma interpretação regional com contagem, qualidade e incerteza antes de investir em
> outro treino longo. Propomos E+F como referência, uma ablação curta do bloco BR e, depois, um adapter com alvo
> regional condicional ao global, comparado a uma continuação global de mesmo orçamento nos blocos não expostos.
> A ideia é ensinar informação útil à representação e medir benefício sem esconder perdas de patogênicas.
> Ainda não demonstramos regionalização nem preservação de desempenho em todos os splits. A célula de viés
> `main` custa cerca de duas horas de extração e pode complementar o diagnóstico, mas já temos base para debater
> a arquitetura agora.

## Fontes de implementação

- [Resultados e declaração da Fase 1](fase1_r03_congelado.md).
- [Diagnóstico: FPR e especificidade por execução](../scripts/fase1_diagnostico.py).
- [Inventário: células e pedidos oficiais](../scripts/fase1b_inventario.py).
- [Features BR e treino dos probes](../scripts/fase1_bracos.py).
- [Layout da leitura E](../eval/campanha/leituras.py).
- [Arquitetura do R03](../../lumina-inference/lumina/models/model.py).
- [Contagens, qualidade, confiança e blocos do R2](../../lumina-mosaic/src/mosaic/regional_truth.py).
- [Regras de consulta regional](../../lumina-mosaic/src/mosaic/real_world.py).
- [Críticas](../../lumina-mosaic/config/critical-variants-br.yaml) e
  [exceções BA1](../../lumina-mosaic/config/ba1-exceptions.yaml).
