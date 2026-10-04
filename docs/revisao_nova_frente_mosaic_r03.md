# Revisão independente da nova frente Mosaic/R03

Data: 03/10/2026. Escopo: revisar o mapa do commit `24af4fd`, confrontá-lo com a implementação do Mosaic e do
R03 e recomendar o próximo experimento. Nenhum modelo foi treinado nesta revisão.

Fontes locais: Mosaic `95adc39` (entrega com código `f2e9a9f`), inference `b9b042f` e este consumidor `24af4fd`.
As contagens abaixo são as publicadas pelo Mosaic, não uma nova contagem dos Parquets completos. A revisão foi
de código, configurações, histórico e documentação; não foi uma revalidação dos 300 GiB publicados ou uma
execução completa do avaliador.

## 1. Parecer

**Podemos avançar no desenvolvimento.** A orientação do Eduardo de partir do benchmark para formular hipóteses
é compatível com o Mosaic atual, que declara o corpus como desenvolvimento, sem lockbox. O passo seguinte deve
medir qual informação acrescenta valor ao R03; não repetir automaticamente o MLM misto anterior.

O mapa do outro chat identifica corretamente a nova verdade regional, os proxies preservados, as vistas e o
cross-fitting. Entretanto, não se sustentam as afirmações de ganho por construção, proteção de todas as críticas
pelo avaliador atual ou impossibilidade de reaproveitar qualquer extração antiga. Essas passagens foram
corrigidas no [mapa do Mosaic](analise_mosaic_v1_2026_09_30.md).

Explorar o corpus deliberadamente para escolher um método pode encontrar uma solução útil. Depois dessas
escolhas, uma melhora nesse mesmo corpus continua sendo desenvolvimento. O argumento de que o ClinVar é
circular não torna independente uma avaliação que orientou o método. O próprio
[ADR 0011](../../lumina-mosaic/docs/adr/0011-development-and-confirmation.md) reconhece isso: refazer folds não
apaga exposição anterior; confirmação exige dados posteriores não usados nas escolhas, com o sistema congelado.

## 2. Quais perguntas brasileiras realmente existem

| Avaliação | Como a coorte nasce | O que responde | O que não responde |
| --- | --- | --- | --- |
| `br_clinical_evidence` | ClinVar consensus com SCV P/B de instituição brasileira, contra consensus sem essa participação | Se o desempenho muda no subconjunto com participação institucional brasileira | Ancestralidade brasileira do portador ou causalidade de um efeito populacional |
| `br_population_observed` | ClinVar gold presente no ABraOM, contra gold não presente | Desempenho em alelos observados nessa coorte e contraste pareado | Representatividade de todo o Brasil ou prova de adaptação do embedding |
| Viés regional | Variantes benignas por frequência/qualidade do ABraOM; célula primária contra comparável | Excesso ajustado de falsos positivos nas variantes relativamente mais frequentes no Brasil | AUROC entre classes: aqui só há benignas |
| Benefício regional | Gold de teste presente no ABraOM, cross-fitted | Cobertura de decisões corretas de candidato + ABraOM contra RW-3 | A interação caso–controle do proxy populacional |
| Segurança | P/LP presentes no ABraOM, gold e consensus, e lista crítica separada | Se a informação regional faz perder chamadas patogênicas | Ausência de degradação geral do modelo em todos os outros estudos |

O clínico tem 3.119 casos, dos quais 3.116 pareados. O populacional passou a 2.057 casos, com 621 pares. Os pares
igualam **rótulo × painel × bin de frequência global**, sem reposição. Isso não iguala gene, frequência exata,
conservação ou contexto. Casos sem par não entram na interação pareada.

`present_abraom` exige AN > 0 e AC > 0 e aceita qualquer FILTER. Isso não é a qualidade exigida para verdade
benigna: a verdade usa PASS, classe `vSR`, suporte de AN, exclusões de críticas/P/LP/conflitos e regras BA1.
Ausência no arquivo observado não fornece AN e não é uma medição de frequência zero.

Fontes: [pareamento](../../lumina-mosaic/src/mosaic/brazil_study.py),
[anotação ABraOM](../../lumina-mosaic/src/mosaic/annotations/abraom.py),
[coorte de benefício](../../lumina-mosaic/src/mosaic/comparator_eval/primary_cohorts.py).

### A nova célula não significa “ultrarrara no mundo”

A célula primária tem 10.829 variantes benignas da camada B, com AF no ABraOM entre 1% e 5% e pelo menos cinco
vezes a AF global. Por exemplo, 4% no ABraOM e 0,8% no gnomAD satisfazem a regra. É uma diferença **relativa**,
não necessariamente raridade absoluta. A célula comparável tem frequência global dentro de um fator dois da
regional. O contraste principal de falsos positivos é ajustado por painel e quartil de phyloP241.

O ABraOM completo fornece muito mais loci, AC/AN e qualidade que o recorte anterior. Ainda representa uma
coorte específica, SABE-1171, não uma amostra uniforme de todas as populações brasileiras.

Fontes: [verdade regional](../../lumina-mosaic/src/mosaic/regional_truth.py),
[protocolo dos estudos](../../lumina-mosaic/config/study-protocol.yaml).

## 3. Como usar os novos splits corretamente

O novo caminho para o R03 é a vista de **4.096 bp**. Ela tem folds, elegibilidade e purgas próprios; não é o
snapshot da campanha anterior com nomes novos.

Para cada uma das cinco execuções, o fold de teste gira, o próximo é validação e os três restantes treinam.
Treino admite gold + consensus; validação seleciona no gold; o resultado principal do núcleo usa gold de teste.
Também é necessário fornecer predições de consensus de teste para a segurança regional.

Há duas unidades distintas:

- `core_unit_id`: segmento usado para distribuir folds na vista de 4 kb;
- `overlap_cluster_id`: componente transitivo mantido como unidade de inferência.

Segmentar os componentes não basta: `core_purged_runs` retira por execução as variantes de treino/validação
cujo contexto toca teste ou validação mantida. O consumidor deve aplicar essa lista, não apenas selecionar
`core_fold`. O núcleo pode ter genes presentes dos dois lados; generalização para genes novos é o estudo
`gene_transfer`, com outra divisão.

Para variantes externas, o Mosaic fornece a execução que pode pontuá-las: a que mantém fora do treino todas as
unidades vizinhas tocadas. Variantes sem vizinhos recebem uma execução por hash; variantes que tocam folds
incompatíveis ficam inelegíveis para sistemas treinados. Usar uma cabeça arbitrária ou a média dos cinco modelos
em todas as variantes quebraria esse cross-fitting.

O R2 tem uma segunda separação: treina informação do ABraOM só em blocos expostos de 1 Mb; avalia a generalização
em blocos não expostos, com buffer. Essa separação não substitui os folds da cabeça clínica. Se o objetivo
auxiliar usar rótulos clínicos, também precisa respeitar o treino clínico permitido para cada execução.

Fontes: [vistas e purgas](../../lumina-mosaic/src/mosaic/views.py),
[anexação de variantes externas](../../lumina-mosaic/src/mosaic/clusters.py),
[guia de submissão](../../lumina-mosaic/docs/GUIA_DE_SUBMISSAO.md).

## 4. O que corrigir na interpretação do outro chat

### Frequência regional não garante interação positiva

Os controles do proxy populacional não têm uma observação positiva no ABraOM, mas podem ter indicadores de
ausência, valores imputados e scores alterados por pesos compartilhados. Uma cabeça reajustada muda o cálculo
em ambos os grupos. Um adapter muda a representação dos controles também.

Mesmo que uma regra alterasse só casos, isso não asseguraria melhora: torná-los mais benignos pode recuperar B
e prejudicar P. A AUROC depende da ordenação entre ambas as classes. A direção da interação precisa ser medida.

### O benefício não tem os controles daquele proxy

O benefício oficial usa todo o gold de teste presente no ABraOM e compara candidato dentro das regras com
ABraOM (R1) contra as regras automatizadas também com ABraOM (RW-3). Ambos recebem a informação populacional.
O ganho precisa acrescentar algo ao comparador; não basta mostrar que adicionar ABraOM ao R03 muda scores.

Nas camadas benignas definidas por frequência, uma regra que lê a própria frequência usada para definir a
verdade é parcialmente circular. Por isso esse resultado não prova que o tronco aprendeu regionalização.
Isso também não garante acerto em toda a camada B: ela começa em 1%, enquanto BA1 padrão exige limite inferior
de AF acima de 5%, e as regras por gene têm suas próprias condições.

### Diferenças nos especialistas não diagnosticam ancestralidade

O REVEL tem estimativa menor no proxy clínico brasileiro em missense. Isso justifica investigar genes, efeitos
de sequência, frequências e composição de evidência. Não estabelece sozinho um mecanismo populacional, nem
prova que esse é o único subconjunto com espaço para ganho. Não medimos ainda o viés do R03 nesse release.

### O R03 está confirmado; o objetivo original da cabeça não está inteiramente verificado

O inference confirma R03 com cerca de 52 milhões de parâmetros, `last_hidden_state` com 448 dimensões e
`mid_hidden_state` com 384 dimensões a 4 bp/token. As cabeças populacionais têm quatro saídas por posição:
`gnomad_af_pred` e `gnomad_observed_logits`.

O plano do Mosaic atribui a elas um alvo global gnomAD. O inference documenta log-AF, mas omite pipeline de
dados e loss. Não inferir automaticamente a base do log, censura, máscaras de observação ou calibração da AF
a partir do nome do tensor. Confirmar esses detalhes antes de usar a saída como frequência numérica ou
calcular um resíduo que dependa dessa escala.

`lumina-research` ajudaria nessa confirmação, mas não bloqueia uma cabeça auxiliar nova ou rsLoRA neste
consumidor: já dispomos do carregamento e do treino de adapter exercitados no R03 real.

Fontes: [contrato técnico do R03](../../lumina-inference/TECHNICAL.md),
[encode e cabeças](../../lumina-inference/lumina/models/model.py).

## 5. Lacuna concreta no avaliador de segurança

Em [evaluate_safety.py](../../lumina-mosaic/scripts/evaluate_safety.py), o código primeiro restringe a coorte
a P/LP com `present_abraom=True` e somente depois cruza a lista crítica. Assim, uma crítica ausente do ABraOM
fica fora da lista avaliada. A própria
[configuração crítica](../../lumina-mosaic/config/critical-variants-br.yaml) declara TP53 p.R337H ausente e o
plano informa que ela não está no release.

Fiz uma reprodução isolada usando as funções reais `main` e `test_calls`, com leituras de Parquet substituídas
por DataFrames sintéticos: 580 P-BR preservadas e uma crítica ausente do ABraOM. O resultado foi
`critical: []`, `critical_lost: 0` e `safety_declared: true`. Isso demonstra a omissão, não um resultado sobre
o R03. Para x=0, o quantil usado na fixture foi calculado pela fórmula exata do Clopper–Pearson.

Há também falta de exigência de cobertura completa antes da contagem: chamadas R0 ausentes não contam como
“reconhecidas”, embora o denominador continue sendo a coorte. Isso precisa ficar explícito para não confundir
falta de predição com ausência de perda.

Antes de uma afirmação de segurança, o consumidor deve conferir as críticas **pela lista completa**, inclusive
as externas ao release, e publicar cobertura, chamada da base e chamada regional de cada uma. Falta de cobertura
deve impedir a afirmação sobre aquela variante. É preciso separar a exigência absoluta de “continua patogênica”
da perda relativa R0→R1: uma variante que a base já não reconhece também exige relato.

Outra distinção de escopo: a segurança automatizada compara o candidato dentro das regras **sem/com ABraOM**.
Se esse candidato já for um R2 adaptado, o contraste não recupera automaticamente perdas que a adaptação causou
em relação ao R03 original. A comparação base→R2 nas patogênicas precisa existir também no consumidor.

Não alterei o repositório do Mosaic nem sua identidade congelada. A lacuna deve ser corrigida em versão declarada
do avaliador ou coberta explicitamente pelo consumidor. Ela não bloqueia treino exploratório de candidatos.

## 6. O que reaproveitar da campanha anterior

Reaproveitar a infraestrutura validada: carregamento estrito do R03, construção REF/ALT, auditoria contra FASTA,
rsLoRA efetivo, checagem de gradientes, preservação dos pesos base, cache com proveniência e leitura de cabeças.
Não precisamos passar meses demonstrando outra vez essas mesmas propriedades sem mudança de implementação.

A leitura antiga que venceu o G5 tem 1.344 dimensões: embedding REF no focal, diferença ALT−REF no focal e média
local REF com raio 64. A leitura compacta de 172 dimensões é composta de projeções nas cabeças e informações da
substituição; ela foi candidata, não vencedora. São baselines disponíveis, não a definição comprovada da “fase A”
do repositório `lumina-embeddings` mencionado no histórico do Mosaic.

Valores já extraídos podem ser reaproveitados por variante após verificar identidade de sequência, checkpoint,
extrator, orientação, janela, layout e condições numéricas. O novo consumidor precisa emitir um manifesto próprio
que cite a origem desse reaproveitamento, sem simplesmente trocar o release no manifesto antigo. Extrair apenas
o que faltar pode reduzir muito o custo inicial. Conferir uma amostra com o forward atual e uma tolerância
declarada: a composição do lote também produz pequenas diferenças numéricas, já medidas na campanha anterior.

Não transportar automaticamente: folds antigos, seleção comum, política de exclusão escolhida no G5, cabeças,
Platt e limiares. Foram ajustados para outro contrato. Os números antigos continuam registrados como resultados
da campanha encerrada; não demonstram o comportamento nos universos novos.

Fontes: [leitura implementada](../eval/campanha/leituras.py),
[layout](../eval/campanha/layout.py),
[migração semântica](../../lumina-mosaic/docs/adr/0013-semantic-release-naming.md).

## 7. Experimento recomendado: separar informação de representação

### Primeiro: medir o valor da informação com o R03 congelado

Usar o novo core de 4 kb e o mesmo protocolo de cabeça para todos os braços. Começar com um classificador simples
sobre a representação já disponível; manter capacidade, sementes, linhas de avaliação e calibração comparáveis.

| Braço | Entradas | Pergunta |
| --- | --- | --- |
| F | Bloco de frequência global oficial | Quanto já se explica sem sequência? |
| E | Representação do R03 congelado | O que o R03 oferece por si? |
| E+F | Mesma representação + mesmo bloco global | O R03 acrescenta ao prior global? |
| E+F+BR | Mesmas entradas + observação/frequência/qualidade regional | Existe ganho adicional com a informação brasileira? |

O bloco global oficial tem AF joint, popmax, AMR, AFR, NFE e indicadores de observação. Para o braço BR,
preservar estados distintos de presente, AC zero, sem chamada e não encontrado; incluir suporte/qualidade
quando disponíveis. Uma imputação para o classificador não deve ser descrita como uma medição de AF zero.

**Distinção de implementação:** o braço oficial `system_plus_frequency` do Mosaic combina o **score** do sistema
com frequências numa regressão logística. E+F acima combina **embeddings** com frequências; é outro sistema do
consumidor, que deve ser identificado separadamente. Pode-se calcular ambos sem confundir seus nomes.

Somar BR é uma intervenção de informação, não uma mudança no embedding. Ela é o limite prático inicial para
descobrir se vale aprender essa informação. Todos os sistemas podem ser comparados nos dois proxies e nos
endpoints regionais, mantendo o custo no núcleo e resultados por painel. A comparação R1/RW-3 continua separada
da comparação de classificadores puros.

Fontes: [features oficiais](../../lumina-mosaic/src/mosaic/comparator_eval/frequency_arms.py),
[RW-4](../../lumina-mosaic/scripts/build_rw4.py).

### Segundo: verificar se a informação regional é aprendível do embedding

Se BR acrescentar algo, testar uma cabeça populacional pequena sobre o R03 congelado, treinada só nos blocos
expostos. Candidatos a alvo são log-AF regional ou uma diferença regional–global, com denominadores, qualidade e
tratamento de zeros declarados. Não usar a cabeça nativa como frequência calibrada antes de confirmar sua escala.

Avaliar nos blocos não expostos. Esse teste barato responde se há sinal que a representação atual consegue
recuperar. Se somente a consulta direta à tabela ajuda, não concluir que um adapter de sequência conseguiria
reproduzir o ganho: demografia e amostragem não estão determinadas unicamente por uma janela REF/ALT.

As classes de painel podem orientar features ou cabeças especializadas, porque são calculadas a partir da
variante/anotação. Devem estar disponíveis igualmente à base e ao regionalizado. Não confundir painel com rótulo
P/B, membership brasileiro ou `bias_cell`, que usa a própria comparação de frequências avaliada.

### Terceiro: testar mudança aprendida do espaço com objetivo explícito

Se o probe populacional mostrar sinal útil, comparar R2 com R2c de mesmo orçamento. Uma candidata coerente é
rsLoRA ou um pequeno módulo residual sobre o R03, com objetivo auxiliar que estime a informação regional,
mantendo um objetivo/controle global para medir esquecimento. Não substituir a frequência regional por apenas
amostragem de variantes: o MLM anterior recebeu bases como alvos, não a diferença de frequências populacionais.

O CLI `evaluate_submission.py --studies regional` não implementa sozinho o contraste R2−R2c nos blocos não
expostos. Esse recorte, a comparação pareada e a proveniência do treino precisam ser produzidos pelo consumidor;
rodar o CLI geral e chamar o sistema de R2 não verifica essas condições.

Comparar a mesma extração e receita de classificador antes/depois, com heads reajustadas sob o mesmo protocolo;
como diagnóstico adicional, aplicar a cabeça base congelada à representação nova. O primeiro mede utilidade
da representação para um consumidor reajustado; o segundo mede compatibilidade com o consumidor existente.

Medir mudança de embedding é útil, mas magnitude, UMAP ou separação visual não provam regionalização. Uma
rotação invertível pode mudar muito o vetor sem acrescentar informação. O que importa é se um consumidor
controlado recupera melhor informação útil em loci não usados no ajuste e se o efeito melhora decisões sem
perder patogênicas ou desempenho global.

## 8. Como decidir o próximo passo sem criar novos portões

1. Validar os artefatos publicados necessários e construir as cinco divisões de 4 kb a partir dos papéis e purgas
   do novo release. Não reconstruir o Mosaic nem baixar todos os intermediários para começar.
2. Auditar o reaproveitamento das features M0 existentes e extrair o complemento. Montar E/F/E+F/E+F+BR.
3. Medir onde ocorre ganho ou falha: missense, splice, noncoding, verdade regional e segurança. Não exigir um
   resultado do diagnóstico de viés como condição nova para experimentar uma ideia autorizada pelo Eduardo.
4. Usar os resultados para escolher uma única proposta aprendida, testar em escala pequena e comparar com R2c.
5. Para afirmar “sem degradar outros splits”, declarar quais estudos e qual margem prática se aplicam. O protocolo
   atual deixou `global_cost` como descritivo: ele não aprova automaticamente não inferioridade. IC que inclui
   zero também não prova ausência de degradação.

Não é necessário solicitar decisões já respondidas por código e protocolos, como definição dos painéis, origem
dos proxies ou regra dos folds. O que o dado não resolve sozinho é qual perda de desempenho o projeto considera
aceitável, ou qual alegação final pretende sustentar. Podemos propor esses critérios explicitamente como
decisões de desenvolvimento, sem atribuí-los ao Eduardo e sem atrasar os baselines.

**Resumo para o Eduardo:** vamos separar “usar informação brasileira” de “aprender informação brasileira”. Primeiro
mediremos o R03 congelado com e sem o bloco global e ABraOM. Depois testaremos se uma cabeça simples recupera o
sinal regional em blocos não expostos. Só então escolheremos um ajuste do espaço com alvo populacional explícito,
comparado a continuação global. Os proxies brasileiros ficam como leituras complementares; viés, benefício,
segurança e custo global são perguntas distintas. Todo resultado no corpus usado para essas escolhas permanece
desenvolvimento.

## 9. Conferência desta revisão (04/10)

Aceita. Os pontos verificáveis foram conferidos no código:

- **Lacuna de segurança: confirmada.** Em `scripts/evaluate_safety.py`, a coorte `pbr` é filtrada para P/LP com
  `present_abraom` antes de cruzar a lista crítica. Pela própria `config/critical-variants-br.yaml`, 6 das 13
  críticas são ausentes do ABraOM (TP53 R337H, PPOX R168H, as duas POLH, TTR V50M e GBA1 G416S) e nunca entram na
  conferência. Além disso, `critical_lost` só conta perda em relação ao R0: uma crítica que o R0 já não chama de P/LP
  não é sinalizada. E chamada ausente vira "não reconhecida", sem gerar perda.
- **Regra do ABraOM no R1/RW-3: confirmada.** Só chamadas PASS e `vSR` (`real_world.py:341`), limite inferior
  comparado com `>` (`real_world.py:178`). O texto de `config/real-world.yaml` diz `>= 0.05`; é uma divergência
  menor do próprio Mosaic.
- **O CLI não faz R2 − R2c: confirmado.** `evaluate_submission.py` não lê `r2_role`; só `regional_truth.py` o
  define.
- **Leitura de 1.344 dimensões: confirmada** (`eval/campanha/leituras.py`): referência no sítio, ALT − REF no sítio
  e média da referência em `[f − 64, f + 64)`, no tronco pós-norma de 448.
- **Geometria da janela compatível.** O nosso extrator usa o offset `L // 2 − 1` (2047 em 4.096 bp), a convenção do
  Mosaic (`eval/embedding_probe/windows.py`). O reaproveitamento continua exigindo conferir checkpoint, FASTA,
  versão do extrator e tamanho do lote, que muda a numérica em cerca de 2e-3.

Duas adições ao desenho do §7:

1. **E+F+BR é experimento de valor da informação, não sistema regional oficial.** As células do viés são definidas
   pela razão entre a AF do ABraOM e a do gnomAD; um classificador que lê essas duas frequências reduz FP ali por
   construção. No R1, ele contaria o ABraOM duas vezes. O viés continua sendo medido nos sistemas sem ABraOM na
   entrada (E, E+F) e, depois, no R2 nos blocos não expostos.
2. **Os membros dos proxies brasileiros estão no núcleo.** Cada um cai em algum fold de treino do `core_locus` de 4
   kb. Para pontuá-los fora da amostra, usa-se a execução cujo fold de teste os contém, com a mesma regra em todos
   os braços. Isso se afasta do "par congelado" do track `brazil`, porque são cinco modelos e não um, e continua
   sendo desenvolvimento.

O plano executável da primeira etapa está em [fase1_r03_congelado.md](fase1_r03_congelado.md).
