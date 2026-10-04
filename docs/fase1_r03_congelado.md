# Fase 1: valor da informação com o R03 congelado

Objetivo: antes de qualquer adaptação aprendida, medir o que cada fonte de informação acrescenta ao R03 nos splits
do Mosaic novo (`mosaic-v1-2026-09-30`). É o primeiro experimento de
[revisao_nova_frente_mosaic_r03.md](revisao_nova_frente_mosaic_r03.md) §7. Tudo aqui é desenvolvimento: os
cortes orientam a escolha do método e não confirmam nada.

## 1. Braços

Mesma receita de cabeça, mesmas cinco execuções, mesmas linhas em todos.

| Braço | Entradas | Pergunta |
|---|---|---|
| F | bloco de frequência global oficial (`frequency_arms.frequency_features`) | quanto se explica sem sequência |
| F+BR | F + bloco ABraOM | o que o ABraOM acrescenta à frequência global (refaz o fato 3 do plano com o dump completo) |
| E | leitura `leitura_antiga_1344` do R03 congelado | o que a representação oferece |
| E+F | E + F | se o R03 acrescenta ao prior global |
| E+F+BR | E + F + bloco ABraOM | se há ganho adicional com a informação brasileira |
| S+F | score de E + F, regressão logística | variante treinada de `system_plus_frequency`, com score de treino por cross-fitting interno |

**Bloco ABraOM (BR).** É lido de `variant-annotations.parquet`:
- `log10(abraom_af + 1e-6)`;
- indicadores dos estados `present`, `ac0`, `no_call` e `not_found`;
- indicador de FILTER PASS;
- AN / 2.342.

Um estado sem medição não é tratado como AF zero: o valor imputado só existe para o classificador, e os
indicadores dizem o que aconteceu. A classe CEGH e os homozigotos estão no extrato do ABraOM e entram depois, se a
Fase 1 mostrar sinal.

**Cabeça.** Regressão logística com L2 sobre features padronizadas no treino de cada execução. C vem da grade {1e-3,
1e-2, 1e-1, 1, 10}, escolhido pela macro AUROC de missense, splice e noncoding na validation. É um probe linear,
para controlar a receita da cabeça. Acrescentar colunas também aumenta o número de parâmetros: usar a mesma
regressão não iguala capacidade exatamente. Um MLP fica como sensibilidade posterior.

No S+F, S é o score do braço E, somado ao bloco F. Como E é treinado, os scores usados para ajustar o segundo
classificador precisam de uma estratégia declarada, sem usar o teste externo: por exemplo, predições fora da
amostra geradas dentro do treino da execução. Não confundir score de treino de E com score fora da amostra.

## 2. Divisões

A vista de 4 kb do release novo: `views/4kb/partitions.parquet`. Na execução `i`:
- **Treino:** folds ≠ {`i`, `i+1`}, gold + consensus, sem as linhas com `i` em `core_purged_runs`.
- **Validation:** fold `i+1`, gold, sem as purgas.
- **Teste:** fold `i`, **todos os tiers**. O núcleo oficial usa só gold; consensus é necessário para os proxies e
  para as P-BR.

Sanidade: o run 0 tem de dar 194.666 de treino, 2.106 de validation e 2.110 de teste gold elegível
(`GUIA_OPERACIONAL_DE_SPLITS.md` §10.1).

Essa contagem de treino é anterior ao filtro de sequência. Para comparar os braços com as mesmas linhas, o
treino efetivo precisa usar a interseção elegível com features disponíveis, inclusive nos braços só de frequência.

Cada variante do release fora do treino é pontuada pela execução cujo fold de teste a contém. Vale para os membros
dos proxies brasileiros, para a coorte de benefício e para as P-BR.

**Exposição dos proxies.** Nesta análise exploratória, um membro pode treinar cabeças de outras execuções, embora
não treine a execução que o pontua como teste. Portanto, não se deve afirmar que nenhum membro dos proxies entra
no treino, nem chamar esse desenho de avaliação oficial do par brasileiro congelado. O estudo `regional` e seus
folds de núcleo são distintos desse contrato. Uma avaliação oficial de `brazil` exigirá respeitar seu contrato
próprio; as leituras cross-fitted desta fase ficam identificadas como desenvolvimento exploratório.

## 3. Avaliação

| Leitura | Coorte | Medidas |
|---|---|---|
| Núcleo | teste gold de 4 kb, cinco execuções | avaliador oficial (`scripts/evaluate_candidate.py`) por braço; deltas pareados entre braços com bootstrap por `overlap_cluster_id` da vista |
| Proxies | `studies/brazilian-proxies/membership.parquet` | deltas entre braços no coorte completo, nos pareados e nos controles, e a interação; bootstrap por `overlap_cluster_id` do membership |
| Benefício (descritivo) | teste gold de 4 kb presente no ABraOM (2.057, 98 P) | AUROC/AUPRC por braço e deltas; não é o endpoint oficial com RW-4 |
| P-BR | P/LP presentes no ABraOM, gold e consensus, no teste de 4 kb | sensibilidade no limiar da validation (regra do MCC) e perdas entre braços |
| Críticas | as 13 de `config/critical-variants-br.yaml` | caso a caso: no release ou não, presença no ABraOM, score e chamada de cada braço |

Tudo é lido por painel. Pares de braços: F → F+BR, E → E+F, E+F → E+F+BR. Bootstrap com 1.000 réplicas e seed
20260901.

**Fora da Fase 1:**
- os endpoints oficiais do estudo `regional` (viés nas células, RW-4, R0/R1);
- `gene_transfer` e `time`.

O viés exige extrair as cerca de 624 mil variantes externas pedidas; fica para a Fase 1b, com os braços E e E+F. O
E+F+BR não entra nas células, porque é circular ali.

## 4. Passos

1. **Inventário (sem GPU).**
   - Checkout dedicado do Mosaic no commit da entrega (`f2e9a9f`).
   - Download de `artifacts/`, `config/`, do manifest de distribuição e de `outputs/candidate-manifest/`.
   - `validate-suite` e conferência dos checksums dos pedidos.
   - `scripts/inventario_mosaic_v1.py`: contagens por papel e execução, proxies, coorte de benefício, P-BR, críticas,
     pedidos, e o quanto os caches antigos do M0 cobrem.
   - Executar [inventariar_mosaic_v1.sh](runbooks/inventariar_mosaic_v1.sh): ele usa o mesmo Python nos testes,
     validação e inventário e interrompe se qualquer checagem falhar. A contagem de IDs em comum não autoriza
     reaproveitamento dos embeddings; as 13 críticas listadas ainda não tiveram suas predições verificadas.
   **Resultado (04/10, `~/artifacts/mosaic_v1/inventario_20261004_033946/`).** `validate-suite` ok e pedidos
   conferidos com o release e o protocolo.
   - **Downloads:** release com 124 MB e pedidos com 84 MB.
   - **Execuções:** batem com o guia (run 0: 194.666 / 2.106 / 2.110), com 1.295 a 1.863 purgadas por execução.
   - **Proxy clínico:** 3.116 pares e 3 sem par (todos P). Presença no ABraOM de 19,7% nos casos e 9,6% nos
     controles, uma razão de 2,0×; com o recorte antigo era 4,6×.
   - **Proxy populacional:** 621 pares (97 P / 524 B) e 1.436 sem par (1 P / 1.435 B: 1.001 noncoding, 240
     missense, 166 synonymous, 26 splice, 2 other). O coorte completo é dominado por benignas sem par.
   - **Benefício e P-BR:** batem com o plano (2.057 gold, 98 P, 232 unidades; 1.050 P-BR, 580 grupos de gene).
   - **Críticas:** 11 no release, todas elegíveis em 4 kb; TP53 R337H e GBA1 N370S estão fora dele. Das 11, só 6
     estão presentes no ABraOM (as três HBB, MYO15A, CYP1B1 e CFTR) e entram na conferência do avaliador do Mosaic.
     PPOX, as duas POLH, TTR V50M e GBA1 G416S estão no release, mas ficam de fora dela. HbS é consensus.
   - **Caches antigos:** os do M0 (`g3_cache` 171.720 e `g7_cache` 8.875, sem sobreposição) cobrem 180.595
     variantes elegíveis, com extrator `campanha_r03_extracao_v2`, R03 `f2983560…`, janela 4.096, offset 2.047 e
     lote de 8 pares. O complemento do núcleo tem 146.223 variantes. Os pedidos regionais pedem 611.233 além dos
     caches.
2. **Extração (GPU).** `scripts/extrair_mosaic_v1.py`, pelo
   [runbook](runbooks/extrair_mosaic_v1_gpu.sh), em duas etapas:
   1. **Conferência.** Reextrai uma amostra determinística de 512 variantes dos dois caches antigos e compara os
      vetores com a tolerância do extrator (1e-5).
   2. **Complemento.** Só se a conferência passar: as 146.223 variantes, cerca de 2 h na taxa medida do M0.

   As duas exigem a identidade da referência em tudo menos a tabela; qualquer diferença (código, ambiente, FASTA,
   lote) recusa a extração. Se a identidade divergir ou a conferência reprovar, o reaproveitamento deixa de valer
   e a saída é extrair as 326.818 variantes de novo, com identidade própria.
   A revisão de 04/10 também confere **cada cache antigo**, incluindo o dos estudos: manifesto completo, hashes
   da tabela, dimensões, finitude, duplicatas e cobertura dos fragmentos. As coordenadas e alelos de cada ID em
   comum têm de concordar com o release novo; rótulos, painéis e folds continuam vindo do release novo. A amostra
   é comparada separadamente contra cada cache, e NaN, formas incompatíveis ou vetores ausentes reprovam.
   `fontes_da_extracao.json` registra os hashes dos arquivos reaproveitados e das entradas do release. Os logs
   externos e internos recebem nomes únicos; `nohup`/`setsid` protegem contra fechar o terminal, não contra
   desligar a instância. Essas mudanças não alteram o caminho numérico do extrator da campanha.

   **Resultado (04/10, 15:59; revisão `946ae22`).** Log `~/artifacts/mosaic_v1/passo2_saida_20261004_155759_667.out`.
   - **Caches antigos:** os dois passaram na validação completa (identidade, manifesto, tabela, fragmentos,
     coordenadas).
   - **Conferência:** as 512 variantes reextraídas (0,0636 s/variante) são **numericamente idênticas na amostra**.
     As 256 de cada cache dão diferença máxima 0,0 nas duas leituras, `cabecas_172` e `leitura_antiga_1344`.
     Registro em `~/artifacts/mosaic_v1/cache_conferencia_20261004_155759_824/conferencia.json`.
     A conferência não reextraiu os 180.595 vetores antigos; a autorização de reaproveitamento combina essa
     amostra com as checagens de identidade, coordenadas e integridade do cache inteiro.
   - **Complemento:** a corrida interrompida (revisão `3be6982`) já tinha gravado as 146.223 variantes em 36
     fragmentos. A retomada conferiu identidade, tabela e fragmentos, encontrou 0 pendentes e fechou com
     `"completo": true` em `~/artifacts/mosaic_v1/cache_M0_complemento/`, com `fontes_da_extracao.json`.

   As 326.818 variantes elegíveis em 4 kb têm agora a leitura do R03 congelado, em três caches com a mesma
   identidade numérica: `~/artifacts/redesenho/g3_cache/M0` (171.720), `~/artifacts/redesenho/g7_cache/M0`
   (8.875) e `~/artifacts/mosaic_v1/cache_M0_complemento` (146.223). Os papéis gravados nas tabelas desses caches
   são da campanha antiga e não valem para a Fase 1. Rótulo, painel, fold e purgas vêm sempre do release novo.
3. **Braços (CPU).** As cinco execuções dos seis braços, em `predictions.parquet` e `system.yaml` no formato do
   Mosaic, com a exposição declarada. `scripts/fase1_bracos.py`, pelo [runbook](runbooks/bracos_mosaic_v1.sh), no
   `.venv` do Mosaic (sklearn e o pacote `mosaic`); de 1 a 2 h de CPU.
   - **Conferências antes de treinar:**
     - a identidade da entrega (release, protocolo e protocolo de estudos);
     - as contagens do run 0 contra o guia;
     - o bloco F igual ao `frequency_features` oficial, valor a valor, e o log AF e a presença do ABraOM iguais às
       colunas oficiais;
     - os três caches pela validação do passo 2, sem variante repetida entre eles e com toda elegível coberta.
   - **Cabeça:** `LogisticRegression` L2 do sklearn (newton-cholesky, tol 1e-6), com intercepto sem penalidade. C
     percorre a grade em ordem crescente, com warm start; empate na macro da validation fica com o menor C. O score é
     o logit.
     Um C cujo ajuste não convergiu não concorre; se não houver C convergido com macro finita, a corrida para.
     Qualquer modelo interno de E sem convergência também interrompe S+F. A grade e os motivos ficam registrados.
   - **S+F:** o `build_frequency_arms.py` oficial não serve para sistema treinado, porque toma um score por variante
     sem respeitar a execução.
     - Aqui, S nas linhas de treino vem de cross-fitting pelos três folds de treino da execução, com o C escolhido
       para E e purga interna: sai do treino interno toda linha a menos de 4.096 bp de alguma do fold interno de
       teste.
     - Na validation e no teste, S é o score do modelo E da execução.
     - Os modelos internos treinam com cerca de dois terços do treino, então a escala de S no treino pode diferir da
       validation. O `selecao.json` registra média e desvio de S nos três papéis.
   - **Linhas:** as 8 variantes não elegíveis ficam sem score em todos os braços. O avaliador do Mosaic as mantém na
     validation (cobertura menor que 1 ali), mas calibra o limiar só nas linhas pontuadas.
   - **Exposição:** os braços com R03 declaram `training_cutoff: null` e a exposição do tronco como desconhecida. F e
     F+BR declaram os rótulos do ClinVar 2026-06. O bloco `benchmark` leva os três hashes da entrega.
   - **Saídas:** uma pasta nova, gravada como `.tmp` e renomeada só depois das checagens de contrato do próprio
     Mosaic (`load_system`, `check_benchmark_reference`, `load_predictions`, `check_fold_roles`). Ela contém:
     - um diretório por braço (`fase1-f`, `fase1-f-br`, `fase1-e`, `fase1-e-f`, `fase1-e-f-br`, `fase1-s-f`);
     - `modelos/`, para pontuar fora do release na Fase 1b;
     - `selecao.json`, `linhas.json`, `fontes.json` e `diagnosticos/s_fora_da_amostra.parquet`.
   - O passo 3 não lê métricas de teste. Imprime só a macro da validation, que é a mesma usada para escolher C e,
     portanto, otimista.
4. **Leituras (CPU).** Avaliador oficial do núcleo e o consumidor próprio para os deltas, os proxies, o benefício, as
   P-BR e as críticas. As P-BR usam o limiar de cada execução que o avaliador grava em
   `validation-thresholds.parquet`.

## 5. Como ler o resultado

- **E+F+BR ganha de E+F nos proxies ou no recorte de benefício, sem perder P-BR:** há ganho incremental nesta
  receita e nestas amostras. A Fase 2 testa
  se ela é aprendível do embedding (cabeça populacional nos blocos expostos, avaliada nos não expostos).
- **Não ganha:** não detectamos ganho com esta cabeça, features e amostra. Isso orienta o orçamento, mas não
  refuta toda proposta de adaptação regional; capacidade, regularização, cobertura e precisão também importam.
- **F+BR ganha de F, mas E+F+BR não ganha de E+F:** resultado compatível com redundância na presença de E, mas não
  prova que o R03 represente a informação do ABraOM. Confirmar por probe populacional e diagnósticos de precisão.
- **Ganho com perda de P-BR:** não é regionalização segura. As perdas são lidas por painel e caso a caso nas
  críticas.

Esses pontos são orientações de desenvolvimento, não critérios de parada cientificamente demonstrados. Um
probe congelado que falha não demonstra que adaptar o tronco seja inútil: ele pode não conter a informação de
forma recuperável por essa cabeça. Antes de medir, declarar os contrastes, o efeito relevante e as leituras de
segurança; não escolher critérios de avanço depois de olhar os resultados. A análise de custo global precisa
de estudos e margens próprios. `gene_transfer` e `time` estão fora desta fase, portanto ela não certifica
“sem degradar outros splits”.
