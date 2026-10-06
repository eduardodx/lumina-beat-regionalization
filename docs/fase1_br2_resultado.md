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
| perdas na especificidade equivalente (ver a ambiguidade abaixo) | 10 | depende da leitura |
| Δ AUROC nos recortes brasileiros não abaixo do BR | igual no clínico; acima no populacional e no benefício | sim |
| recuperadas concentradas em AC 1–2 | 9 de 12 no MCC; 6 de 7 na especificidade equivalente | sim |

- **Ambiguidade da especificação.** Fica registrada aqui, sem reescrever a especificação. O texto diz: "no máximo
  metade das P-BR perdidas por E+F → E+F+BR (até 10 de 21), no limiar MCC e na especificidade equivalente".
  - Para a especificidade equivalente, cabem duas leituras: o número entre parênteses (até 10) ou metade das perdas
    do BR naquela leitura (8 de 16).
  - Com 10 perdas, o BR2 cumpre a primeira e não a segunda.
  - A primeira versão deste relatório aplicou só a segunda, sem dizer que havia escolha.
- **O veredito não depende dessa escolha:** no limiar MCC são 12 perdas, acima de 10.

O critério contra (14 ou mais perdas, ou perdas nas mesmas variantes) também não se cumpre: são 12 perdas, e 12 das
21 do BR foram recuperadas. **Pela regra declarada, o resultado é inconclusivo.**

## Leitura posterior

Esta seção não foi declarada antes de rodar. Serve de orientação de desenvolvimento. Foi revisada em 06/10; as
correções estão na última seção.

- **Compatível com a hipótese em parte das perdas, sem isolar a causa.**
  - 9 das 12 recuperadas têm uma ou duas cópias no ABraOM, 8 delas com AC = 2.
  - O BR2 muda várias features ao mesmo tempo, e o classificador foi reajustado. Não se sabe qual mudança produziu a
    recuperação.
  - Ainda há perdas nessas faixas: 3 das 9 mantidas e 2 das 3 novas têm uma ou duas cópias.
- **As 9 mantidas não estão explicadas.**
  - **Cópias não bastam.** 6 delas têm 3 ou mais cópias, mas isso não é frequência bem sustentada. Com AN = 2.342, o
    limite inferior unilateral de 95% fica assim:

    | cópias | estimativa pontual | limite inferior de 95% |
    |---|---|---|
    | 3 | 0,00128 | 0,00035 |
    | 10 | 0,00427 | 0,00232 |

  - **A razão sobre o gnomAD também não basta.** A faixa "≥ 5× o gnomAD" é quase automática para variantes raras. A
    menor AF não nula do ABraOM é 1/2.342 ≈ 0,00043. Uma variante com AF no gnomAD abaixo de 0,000085, vista uma
    vez no ABraOM, já cai nela.
    - Essa faixa não indica enriquecimento brasileiro real.
    - O mesmo vale para a leitura do passo 4: 15 das 21 perdas do BR estavam nela.
  - **O que não se pode afirmar:** que sejam patogênicas mais frequentes no Brasil, ou fundadoras. É preciso ver caso
    a caso: AC/AN, qualidade, incerteza, gene e revisão do ClinVar.
  - **O que E+F mostra:** E+F chamar essas variantes mostra que a configuração anterior as classificava. Não
    identifica por que o novo ajuste as perdeu.
- **Esta forma de ler o ABraOM não aumentou o ganho.**
  - O BR2 não aumenta o ganho do BR no núcleo.
  - Nenhum dos dois blocos tem interação brasileira acima dos controles.
  - O ganho mensurável continua no noncoding.
- **A recuperação é no consensus.** No gold, o BR2 reconhece 66 das 98, contra 69 em E+F e 67 com o BR. Isso impede
  chamar o resultado de melhora de segurança.
- **Segurança.** Com margem 0,01 e n = 580 grupos de gene, a regra tolera no máximo uma perda: o limite superior é
  0,0082 com 1 perda e 0,0108 com 2. A margem não muda.

## Consequência para a arquitetura

Leitura de desenvolvimento, não demonstração.

1. **Interpretação da informação regional.** O BR2 passa a ser o bloco ABraOM de referência nos próximos testes:
   perde menos P-BR que o BR, sem piorar os outros recortes.
   - O princípio dele (estados, suporte amostral, qualidade) é o da verdade regional do Mosaic
     (`mosaic.regional_truth`): limite inferior de Clopper–Pearson, PASS, AN ≥ 80%, classe CEGH `vSR`, teto BA1 do
     CSpec por gene e listas de exceção.
   - Usado como feature, esse princípio tem fonte comum com as células da Fase 1b, que precisam dessa ressalva.
2. **Cabeça: hipótese a testar.** A cabeça linear aplica a frequência regional do mesmo jeito para qualquer força da
   evidência funcional. O teste em CPU é uma cabeça com interação entre o sinal funcional e a informação regional
   confiável. O braço sem ABraOM recebe a mesma família de cabeça.
3. **Representação: não demonstrado nem descartado.** As duas saídas populacionais nativas acompanham pouco a
   frequência do gnomAD e não distinguem presença no ABraOM. Isso não prova ausência de informação no embedding
   inteiro. HbS e HbC já são negativas em E.
4. **Adapter: nenhum adapter longo agora.**
   - Nenhuma versão do bloco mostra interação brasileira.
   - A campanha anterior de adapter deu Δ AUROC −0,0013 no clínico.
   - Argumento, não medido aqui: a diferença de frequência entre populações vem em boa parte da história
     demográfica (deriva, efeito fundador), que a sequência local não prevê.

## Próximos passos

Todos em CPU.

1. **Caso a caso.** As 9 mantidas, as 3 novas e as 12 recuperadas, com atenção às gold. Script:
   `scripts/fase1_br2_casos.py`, que lê os passos 3 e 4 e o BR v2; runbook em
   [br2_casos_mosaic_v1.sh](runbooks/br2_casos_mosaic_v1.sh). Ao lado de cada variante põe:
   - o escore e a chamada de E sozinho, para ver se o sinal funcional aponta para patogenicidade nas perdas;
   - o limite inferior do ABraOM, ao lado da AF e da FAF95 do gnomAD.
2. **Cabeça com interação.** Especificada depois do caso a caso e antes de rodar.
   - Pergunta: a frequência brasileira pode ajudar a reconhecer benignas sem rebaixar automaticamente uma variante
     cujo sinal funcional aponta para patogenicidade?
   - O braço sem ABraOM recebe a mesma família de cabeça.
   - Receita escolhida na validation, purgas mantidas, perdas e ganhos relatados separados.
3. **Controle embaralhado: no máximo análise auxiliar.**
   - Não estima um piso universal.
   - Não é pré-requisito para comparar as predições atuais.
   - Não justifica mudar a margem.

Ficam com o Eduardo:
- **A Fase 1b `main`,** cerca de 2,1 h de GPU.
  - Mede o mecanismo específico de falso-positivos regionais: variantes comuns no Brasil e raras no gnomAD.
  - Na Fase 1 esse lado aparece pouco: são 20 falso-positivos no recorte de benefício.
- **Qualquer adapter.**

Nenhum dos dois bloqueia o trabalho em CPU.

## Revisão de 06/10

Uma revisão externa dos commits `6a32dc7` e `11149a2` corrigiu a primeira versão da leitura posterior. Os números e
o veredito não mudaram.

Correções aceitas:
- **Causa das perdas recuperadas.** Antes: "a hipótese explica parte das perdas". Agora: compatível com a hipótese,
  sem isolar a causa.
- **As 9 mantidas.** Antes: "frequência brasileira bem sustentada" e "patogênicas realmente mais frequentes". Agora:
  não estabelecido.
- **Feature e representação.** Antes: "nenhuma forma de feature corrige" e "a representação não é o gargalo". Agora:
  não demonstrado.
- **Controle embaralhado.** Antes: "necessário" e "mostra o piso de perdas". Agora: no máximo análise auxiliar, sem
  piso universal e sem relação com a margem.
- **Leitura declarada.** A ambiguidade na especificidade equivalente agora está explícita.

Acréscimo desta revisão: a faixa "≥ 5× o gnomAD" é quase automática para variantes raras e não indica
enriquecimento.
