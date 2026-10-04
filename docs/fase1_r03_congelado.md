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
| S+F | score de E + F, regressão logística | o braço `system_plus_frequency` oficial do Mosaic, identificado à parte |

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
para que a capacidade da cabeça não se confunda com a informação de cada braço. Um MLP fica como sensibilidade
posterior.

## 2. Divisões

A vista de 4 kb do release novo: `views/4kb/partitions.parquet`. Na execução `i`:
- **Treino:** folds ≠ {`i`, `i+1`}, gold + consensus, sem as linhas com `i` em `core_purged_runs`.
- **Validation:** fold `i+1`, gold, sem as purgas.
- **Teste:** fold `i`, **todos os tiers**. O núcleo oficial usa só gold; consensus é necessário para os proxies e
  para as P-BR.

Sanidade: o run 0 tem de dar 194.666 de treino, 2.106 de validation e 2.110 de teste gold elegível
(`GUIA_OPERACIONAL_DE_SPLITS.md` §10.1).

Cada variante do release fora do treino é pontuada pela execução cujo fold de teste a contém. Vale para os membros
dos proxies brasileiros, para a coorte de benefício e para as P-BR.

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
2. **Extração (GPU).** A leitura do R03 congelado para o que faltar das variantes do release elegíveis em 4 kb, com
   identidade própria e o mesmo extrator, mais uma conferência numérica contra os caches antigos numa amostra em
   comum. A decisão entre reaproveitar ou extrair de novo sai do inventário: reaproveitar exige o mesmo lote; extrair
   tudo de novo dá uma identidade só.
3. **Braços (CPU).** As cinco execuções dos seis braços, em `predictions.parquet` e `system.yaml` no formato do
   Mosaic, com a exposição declarada.
4. **Leituras (CPU).** Avaliador oficial do núcleo e o consumidor próprio para os deltas, os proxies, o benefício, as
   P-BR e as críticas.

## 5. Como ler o resultado

- **E+F+BR ganha de E+F nos proxies e no benefício, sem perder P-BR:** há informação brasileira útil. A Fase 2 testa
  se ela é aprendível do embedding (cabeça populacional nos blocos expostos, avaliada nos não expostos).
- **Não ganha:** a hipótese de que falta informação regional ao R03 perde força antes de gastar uma corrida longa.
- **F+BR ganha de F, mas E+F+BR não ganha de E+F:** o R03 já carrega o que o ABraOM traria.
- **Ganho com perda de P-BR:** não é regionalização segura. As perdas são lidas por painel e caso a caso nas
  críticas.
