# Fase 1, BR v2: resultado e leitura

Data: 04/10/2026. Corrida das 23:00 às 23:43, na revisão `6a32dc7`. É desenvolvimento exploratório, posterior ao
teste da Fase 1. A [especificação](fase1_br2_especificacao.md) não foi alterada depois da corrida; o runbook
registrou o hash dela no log. Nada aqui autoriza corrida de GPU.

Pastas no notebook:
- treino: `~/artifacts/mosaic_v1/br2_treino_20261004_230006_665`;
- avaliador oficial: `~/artifacts/mosaic_v1/avaliacao_br2_20261004_230006_665`;
- leituras: `~/artifacts/mosaic_v1/leituras_br2_20261004_230006_665/br2.md`.

## Conferências

- **Execução:** testes, seleção do passo 3 (165 ajustes), treino, contrato do Mosaic (337.398 linhas), avaliador e
  leituras terminaram com saída 0.
- **Limiares:** os de E+F e E+F+BR são iguais aos do passo 4; o do BR2 bate com o avaliador.
- **ABraOM:** os estados somam as 326.818 elegíveis.
  - Estados: present 134.055, not_found 192.748, ac0 12, no_call 3.
  - AC entre as presentes: 1 → 12.964; 2 → 6.482; 3–9 → 21.583; ≥ 10 → 93.026.
  - AF publicada × AC/AN: maior diferença relativa de 0,0011; nenhuma acima de 1%.
- **Treino:** C escolhido 0,1; 10; 10; 1; 1. Macro de validation 0,9767; 0,9716; 0,9764; 0,9809; 0,9795. Todos os
  C da grade convergiram.

## Resultados

Δ = novo − base, com IC de 95% por bootstrap (1.000 réplicas).

| leitura | E+F → E+F+BR | E+F → E+F+BR2 | E+F+BR → E+F+BR2 |
|---|---|---|---|
| núcleo, Δ macro AUPRC | +0,0009 [−0,0005; 0,0023] | +0,0010 [−0,0003; 0,0027] | +0,0001 [−0,0004; 0,0008] |
| noncoding, Δ AUROC | +0,0014 [0,0002; 0,0030] | +0,0018 [0,0002; 0,0037] | +0,0004 [−0,0000; 0,0008] |
| proxy clínico, coorte completo | +0,0015 [0,0006; 0,0025] | +0,0015 [0,0005; 0,0026] | +0,0000 [−0,0004; 0,0004] |
| proxy clínico, interação | +0,0003 [−0,0010; 0,0016] | −0,0001 [−0,0016; 0,0014] | −0,0004 [−0,0008; 0,0001] |
| proxy populacional, coorte completo | +0,0027 [0,0007; 0,0043] | +0,0038 [0,0011; 0,0060] | +0,0011 [−0,0003; 0,0025] |
| proxy populacional, interação | −0,0005 [−0,0043; 0,0039] | +0,0009 [−0,0026; 0,0050] | +0,0014 [−0,0008; 0,0032] |
| benefício, Δ AUROC | +0,0027 [0,0006; 0,0061] | +0,0038 [0,0010; 0,0085] | +0,0011 [−0,0002; 0,0027] |
| P-BR perdidas / ganhas | 21 / 11 | 12 / 11 | 7 / 16 |
| P-BR, limite superior de 95% | 0,0517 | 0,0333 | 0,0225 |

- **Painéis:** Δ AUROC de missense, splice e coorte em torno de zero (até ±0,0002) nos três pares.
- **Avaliador oficial no E+F+BR2:** AUROC macro 0,9772; missense 0,9481; splice 0,9945; noncoding 0,9891.
- **Benefício no ponto de operação:**
  - as 98 P do recorte são as P-BR gold;
  - E+F, E+F+BR e E+F+BR2 chamam 69, 67 e 66 delas;
  - os falso-positivos são 20, 17 e 19 entre as 1.959 benignas.
- **Críticas:** chamadas iguais nos três braços. HbS e HbC seguem negativas. O falso-positivo de validation que a
  chamada exigiria é:
  - 0,574 e 0,394 em E+F;
  - 0,678 e 0,516 em E+F+BR;
  - 0,706 e 0,471 em E+F+BR2.

### Perdas e recuperação de P-BR

- **No limiar MCC original:**
  - **Recuperadas:** das 21 perdas do BR, o BR2 recupera 12. Por AC: 1 com 1 cópia, 8 com 2 e 3 com 3–9.
  - **Mantidas:** 9. Por AC: 1 com 1 cópia, 2 com 2, 4 com 3–9 e 2 com ≥ 10.
  - **Novas:** 3, com 1, 2 e 3–9 cópias.
  - **Ganhos do BR:** o BR2 desfaz 4 dos 11.
- **Na especificidade equivalente** (posterior, com a especificidade de validation obtida igual ao alvo):
  - o BR perde 16 e o BR2 perde 10;
  - o BR2 recupera 7, 6 delas com 1–2 cópias; mantém 9 e tem 1 nova;
  - a sensibilidade em P-BR é 0,754 em E+F, 0,750 em E+F+BR e 0,756 em E+F+BR2.
- **As 12 perdas do BR2:**
  - 9 são de ordenação e 10 são missense;
  - 4 gold e 8 consensus. No gold, o BR2 perde 4 e ganha 1; o BR perdia 3 e ganhava 1. A recuperação está no
    consensus;
  - 7 têm AF no ABraOM ≥ 5× a do gnomAD.

### Coeficientes e cabeças nativas (só leitura)

- **Coeficientes padronizados, média das execuções.** As features são colineares, então sinais isolados não se
  interpretam.
  - No BR, o peso está em `br_log_af` (−2,95).
  - No BR2, o peso se espalha pelas faixas e estados: `ac10mais` −0,97, `ac1` +0,24, `ac2` +0,11, `log_af` −0,38,
    `log_af_inferior` +0,50 e `not_found` +0,79.
- **Cabeças populacionais nativas do R03:**
  - layout conferido: o MLM na REF aponta a base REF em 0,886 das variantes;
  - `population_af_head` × log10 da AF do gnomAD: Spearman 0,21, nas 277.967 achadas;
  - `population_observed_head`: AUROC 0,66 para achada no gnomAD e 0,49 para presente no ABraOM;
  - não são AF calibrada.

## Leitura declarada antes de rodar

| critério de apoio | resultado | cumpre |
|---|---|---|
| perdas no limiar MCC ≤ 10 de 21 | 12 | não |
| perdas na especificidade equivalente ≤ metade das do BR (8 de 16) | 10 | não |
| Δ AUROC nos recortes brasileiros não abaixo do BR | igual no clínico; acima no populacional e no benefício | sim |
| recuperadas concentradas em AC 1–2 | 9 de 12 no MCC; 6 de 7 na especificidade equivalente | sim |

O critério contra (14 ou mais perdas, ou perdas nas mesmas variantes) também não se cumpre: são 12 perdas, e 12 das
21 do BR foram recuperadas. **Pela regra declarada, o resultado é inconclusivo.**

## Leitura posterior

Esta seção não foi declarada antes de rodar. Serve de orientação de desenvolvimento.

- **A hipótese explica parte das perdas.** As recuperadas são quase todas de uma ou duas cópias, 8 delas com AC = 2.
- **Não explica o restante.**
  - Das 9 mantidas, 6 têm 3 ou mais cópias; das 12 perdas do BR2, 7 têm AF no ABraOM ≥ 5× a do gnomAD.
  - E+F chamava todas elas. O bloco regional as rebaixa com uma frequência brasileira bem sustentada.
  - Isso não se corrige pela forma da feature. A cabeça linear converte frequência regional em benignidade do mesmo
    jeito para todo gene e para qualquer força da evidência funcional.
- **A forma de ler não explica o ganho pequeno.**
  - O BR2 não aumenta o ganho do BR no núcleo.
  - Nenhum dos dois blocos tem interação brasileira acima dos controles.
  - O ganho mensurável continua concentrado no noncoding.
- **Segurança.**
  - Com margem 0,01 e n = 580 grupos de gene, a regra tolera no máximo uma perda: o limite superior é 0,0082 com 1
    perda e 0,0108 com 2.
  - Nenhum reajuste testado chega perto.
  - Não sabemos quantas perdas um reajuste sem informação nenhuma já causaria.

## Consequência para a arquitetura

1. **Interpretação da informação regional:** manter o princípio do BR2 como referência: estados, suporte amostral e
   qualidade.
   - É o mesmo princípio da verdade regional do Mosaic (`mosaic.regional_truth`): limite inferior de
     Clopper–Pearson, PASS, AN ≥ 80%, classe CEGH `vSR`, teto BA1 do CSpec por gene e listas de exceção.
   - Usado como feature, esse princípio tem fonte comum com as células da Fase 1b, que precisam dessa ressalva.
2. **Cabeça:** é onde está o resíduo. O próximo teste em CPU, especificado antes de rodar, deixaria a penalidade
   regional depender da evidência funcional.
   - Um caminho é a interação com o escore de E fora da amostra, pela partição interna do S+F.
   - Se o release tiver algum atributo de gene, o contexto do gene entra também.
   - E+F, E+F+BR e E+F+BR2 recebem a mesma família de cabeça.
3. **Representação:** não é o gargalo destas perdas, porque E+F chamava as 9 mantidas. HbS e HbC já são negativas em
   E. É um problema da leitura funcional desse mecanismo somado ao da frequência, não algo específico da
   regionalização.
4. **Alvo regional explícito no adapter:** sem indicação por ora.
   - Nenhuma versão do bloco mostra interação brasileira.
   - A campanha anterior de adapter deu Δ AUROC −0,0013 no clínico.
   - As cabeças nativas acompanham pouco a frequência do gnomAD e nada a presença no ABraOM.
   - Argumento, não medido aqui: a diferença de frequência entre populações vem em boa parte da história
     demográfica (deriva, efeito fundador), que a sequência local não prevê. Um adapter treinado para prever
     frequência regional tenderia a reproduzir o mesmo rebaixamento dentro da representação, sem a transparência
     de uma feature.

## Próximos passos possíveis

Todos em CPU; nenhum foi rodado.

- **Controle nulo:** E+F mais o bloco BR2 embaralhado entre variantes, com 3 sementes. Mede quantas perdas e ganhos
  de P-BR um reajuste sem informação causa. É necessário para ler 12 contra 21 e para saber se a regra de segurança
  é atingível por qualquer cabeça reajustada.
- **Caso a caso:** as 9 mantidas, as 3 novas e as 12 recuperadas. Para cada uma: gene, consequência, AC/AN, gnomAD,
  tier e escores de E, E+F e E+F+BR2.
- **Cabeça com interação:** a do item 2, com especificação escrita antes de rodar.

Decisões que ficam com o Eduardo:
- a regra de segurança, que tolera no máximo uma perda, frente ao piso de reajuste;
- a Fase 1b `main`, cerca de 2,1 h de GPU.
  - É onde o benefício regional é medido: nas células de viés, com variantes comuns no Brasil e raras no gnomAD.
  - Na Fase 1, esse lado aparece pouco: são 20 falso-positivos no recorte de benefício.
