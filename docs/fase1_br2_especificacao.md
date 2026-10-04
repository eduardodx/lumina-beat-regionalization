# Fase 1, BR v2: especificação, escrita antes de rodar

Data: 04/10/2026. É desenvolvimento exploratório, posterior ao teste da Fase 1: os testes já foram vistos. Uma única
versão do bloco, sem busca de receitas. Não autoriza nenhuma corrida de GPU. Contexto em
[fase1_r03_congelado.md](fase1_r03_congelado.md) e em [proposta_regionalizacao_r03_pos_fase1.md](proposta_regionalizacao_r03_pos_fase1.md).

## Pergunta

O ganho pequeno e as perdas de P-BR do bloco ABraOM original (BR) vêm, em parte, da forma como o classificador lê o
ABraOM? Em particular: tratar uma ou duas cópias numa amostra de 1.171 pessoas como se fosse frequência medida.

## Comparação controlada

| Braço | Entradas da cabeça | Origem |
|---|---|---|
| E+F | embedding do R03 + frequência global | predições do passo 3 (referência) |
| E+F+BR | E+F + bloco ABraOM original | predições do passo 3 |
| E+F+BR2 | E+F + bloco ABraOM v2 (abaixo) | treinado agora |

- **Mesmas condições:** release, embeddings (os três caches do passo 2), cinco execuções, purgas, linhas e
  padronização no treino. A receita também é a mesma: regressão logística L2, grade de C, escolha pela macro AUROC da
  validation e regra revisada de convergência.
- **Sem retreino dos bracos de referencia:** E+F e E+F+BR não são retreinados. Usam-se as predições do passo 3, feitas
  pelo mesmo código de cabeça; o script confere que as linhas são as mesmas.
- **Mesma família de cabeça:** se um MLP for testado depois, E+F e E+F+BR recebem a mesma arquitetura.

## Insumos e conferências

Do release, em `variant-annotations.parquet`:
- ABraOM: `abraom_ac`, `abraom_an`, `abraom_af`, `abraom_filter` e `abraom_status`;
- gnomAD: `gnomad_v4_af` (AF_joint) e `gnomad_status`.

**Falta no release:** a classe CEGH e a contagem de homozigotos. Elas estão só no extrato do ABraOM da entrega
(`data/annotations/abraom/sabe1171-wgs/`, não baixado). Ficam fora desta versão; não proponho download agora.

**Conferências que interrompem a corrida:**
- `present`: AC ≥ 1, AN ≥ 1 e AC ≤ AN ≤ 2.342;
- `ac0`: AC = 0, AN ≥ 1 e AF = 0;
- `no_call`: AN nulo ou zero;
- `not_found`: AC, AN e AF nulos;
- qualquer outro estado, exceto nulo.

**Relatado sem interromper:**
- a diferença entre a AF publicada e AC/AN;
- a distribuição de AC entre as presentes;
- os valores de FILTER.

## Features do BR v2 e hipótese de cada uma

A frequência pontual passa a ser AC/AN, das contagens registradas. A AF publicada só entra na conferência.

| Grupo | Features | Hipótese |
|---|---|---|
| Estados | `not_found`, `no_call`, `ac0` | Ausência e falha de chamada são estados, não AF zero |
| Contagem | `ac1`, `ac2`, `ac3a9`, `ac10mais`, que partem `present` | Uma ou duas cópias numa amostra pequena não deveriam pesar como frequência; com faixas, a cabeça dá peso próprio a cada nível de suporte |
| Frequência pontual | log10(AC/AN + 1e-6); −6 fora de `present` | Conserva a informação do BR original: um enriquecimento regional verdadeiro não é apagado |
| Frequência sustentada | log10(LI + 1e-6), com LI = limite inferior unilateral de 95% de Clopper–Pearson de AC/AN (`regional_truth.af_lower_bound` do Mosaic) | Separa frequência bem sustentada de estimativa pontual com poucas cópias |
| Qualidade | FILTER PASS; AN/2.342; AN ≥ 80% de 2.342 | Sítio mal chamado não deveria contar como frequência |
| Contraste regional/global | excesso sustentado = max(0, log10(LI + 1e-6) − log10(AF_joint + 1e-6)), só com `present`, PASS e AN suficiente | Enriquecimento sobre o global só quando sustentado e de boa qualidade |

Na última linha, AF_joint é a mesma do bloco F (zero para `not_found` e `ac0`), a referência das células do Mosaic;
`popmax` não a substitui.

## Leituras

As do passo 4, nos pares E+F → E+F+BR (reprodução), E+F → E+F+BR2 e E+F+BR → E+F+BR2:

- **Núcleo:** contraste oficial (macro AUPRC e MCC) e AUROC por painel.
- **Recortes brasileiros:**
  - proxies: coorte completo, pareados, controles e interação;
  - benefício, que tem as mesmas 2.057 variantes do coorte completo populacional.
- **P-BR, no limiar MCC original:**
  - perdas e ganhos separados;
  - quantas das 21 perdas do BR original o BR2 recupera;
  - quais perdas são novas.
- **Especificidade equivalente:** análise posterior, com a especificidade obtida em cada execução.
- **Críticas:** as 13.
- **Coeficientes:** os padronizados das features BR e BR2, por execução, só como leitura.
- **Avaliador oficial:** roda no braço novo e confere os limiares.

**Diagnóstico auxiliar, que não bloqueia:** as saídas populacionais nativas do R03 no bloco `cabecas_172`.
- **Conferência do layout nos dados:**
  - a base REF do one-hot de substituição tem de bater com a variante;
  - o argmax da cabeça MLM na REF é relatado.
- **Comparação:** com a AF do gnomAD e com a presença no gnomAD e no ABraOM.
- **Limite:** a saída nativa não é tratada como AF calibrada.

## Leitura declarada antes de rodar

Orientação de desenvolvimento, não critério de avanço demonstrado:

- **Apoio à hipótese:** E+F → E+F+BR2 perde no máximo metade das P-BR perdidas por E+F → E+F+BR (até 10 de 21),
  no limiar MCC e na especificidade equivalente, e:
  - o Δ AUROC nos recortes brasileiros não fica abaixo do BR original;
  - as recuperadas se concentram nas faixas de AC 1 e 2.
- **Contra a hipótese:** perdas na mesma ordem (14 ou mais) ou nas mesmas variantes.
- **Entre esses valores:** inconclusivo.
- **Em qualquer caso:**
  - perdas e ganhos são relatados separados, e saldo positivo não é segurança;
  - benefício brasileiro exige interação acima dos controles.

## Consequência para a arquitetura

- **Com apoio:** mudar a interpretação da informação regional, nas features e no alvo de um adapter com suporte
  amostral.
- **Contra:** a forma da feature não explica o problema. Investigar a cabeça ou a representação, ou um alvo regional
  explícito.

Nada avança para GPU sem nova decisão.
