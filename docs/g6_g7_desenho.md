# G6 e G7 — congelamento e avaliação única nos estudos brasileiros

Estado: **23/09**, escrito enquanto a₂ e a₃ treinam. Nenhum score dos estudos foi calculado. Este documento separa
o que o protocolo do Mosaic **fixa**, o que já foi **declarado** por nós, o que está **proposto** (confirmar no G6)
e o que está **aberto** (decisão do Eduardo, antes do G7).

Fontes: `lumina-mosaic` commit `814e7f0` (`specs/PLAN.md` §13.3–13.5, `src/mosaic/protocol.py:brazil_protocol_section`,
`src/mosaic/comparator_eval/`); plano da campanha (`docs/proposta_mosaic_regionalizacao_desenvolvimento.md` §6–8);
declaração `configs/campanha_r03_desenvolvimento.json` (seção `g6`).

## 1. Ordem

1. **a₂, a₃** treinados, congelados pela regra (menor `focal_alt`) → caches de desenvolvimento MR_a₂, MR_a₃ →
   comparadores → conferência das cabeças.
2. **G6**: manifesto congelado (seção 5). Depois disso, nada é ajustado.
3. **Extração dos estudos** para M0, MR_a₁, MR_a₂, MR_a₃ (~9 mil variantes; ~10 min por sistema), pelo mesmo
   caminho numérico do desenvolvimento (seção 6). Só com o G6 congelado.
4. **Pontuação** com as cabeças congeladas e **consumidor** (seção 2). Uma vez.

Reexecutar exatamente os sistemas congelados para conferir reprodução não invalida o estudo. Mudar qualquer coisa
por causa do resultado transforma a rodada seguinte em exploratória.

## 2. O que o protocolo do Mosaic fixa (implementado em `eval/campanha/estudos.py`)

| Regra | Como está no consumidor |
|---|---|
| Estudos separados, nunca unidos | `avaliar_estudo` roda um estudo por vez |
| Coorte completo = `case` + `unmatched_case` → Δ_BR_full | `visoes()`, como o `brazil_views` do Mosaic |
| Casos pareados = `case` com controle bidirecional → Δ_BR_matched; controles → Δ_control | o pareamento é **reconferido**; quebrado, o consumidor recusa |
| Interação = Δ_BR_matched − Δ_control, sem `unmatched_case` | `interacao()`; cada delta na interseção de cobertura do próprio grupo |
| AUROC e AUPRC só com as duas classes | AUPRC = precisão média em degraus (a `average_precision_score` do Mosaic; teste confere a definição) |
| n_P, n_B e cobertura sempre; deltas na **interseção** de cobertura; sem imputação | `comparar()`: cobertura de cada sistema no coorte e métricas na interseção |
| Relatório do coorte inteiro obrigatório; painéis como diagnóstico; **sem macro brasileira**; plof e synonymous como guarda; sem piso 50/50 | coorte inteiro + por painel no coorte completo (AUROC/AUPRC só em missense/splice/noncoding) |
| Métricas com limiar só com limiar externo congelado, com proveniência | sem limiar declarado, são omitidas |
| Bootstrap pareado por `overlap_cluster_id`, 1.000 réplicas, seed 20260901, percentis 2,5/97,5 | os **mesmos sorteios** para os dois sistemas (teste: transformação monótona dá delta 0 em toda réplica) |
| No estudo clínico, relatar o subconjunto `present_abraom` | coorte completo com `present_abraom = true` |
| Taxa de pareamento e composição antes e depois | `pareamento` e `composicao` por coorte |

**Atenção: a métrica principal do G7 não é a do desenvolvimento.** No desenvolvimento, o critério declarado foi a
macro de missense/splice/noncoding. No G7 o protocolo do Mosaic manda relatar o **coorte inteiro** (AUROC e AUPRC)
e trata os painéis como diagnóstico, sem macro obrigatória. Na comparação exploratória, essa métrica de coorte
inteiro (a "AUROC geral") foi a que teve IC todo abaixo de zero (−0,0025 [−0,0051; −0,0006]). Isso não é portão: o
desenvolvimento não mede a pergunta regional, e a interação desconta o que for comum a casos e controles. Mas é o
dado de desenvolvimento mais próximo da escala do G7, e pesa na escolha das margens (seção 4).

## 3. Declarado por nós (seção `g6` da declaração, 23/09)

- **Composição final:** M0 = h11, h12, h13; MR = a₁+h11, a₂+h12, a₃+h13. As nove cabeças MR dos três comparadores
  não se misturam; nenhuma é escolhida depois de ver resultado (`tests/test_declaracao_g6.py`).
- **Predição do sistema:** média das três probabilidades calibradas.
- **Limiar do ensemble:** MCC na média das probabilidades no fold 1, por sistema, congelado no G6. Os limiares
  individuais das cabeças não servem.
- **Um gerador por análise**, com a seed declarada: o resultado de uma análise não depende de quais outras rodam.

## 4. Proposto (confirmar no G6) e aberto (Eduardo)

**Proposto — implementado e testado, a confirmar:**

| Item | Proposta | Por quê |
|---|---|---|
| Reamostragem da interação | clusters sorteados **em conjunto** sobre casos pareados + controles, mesmos sorteios para M0 e MR | o Mosaic reamostra cada coorte separadamente e não define a da interação; um cluster com caso e controle entra inteiro |
| Sensibilidade da interação | reamostragem por **par** | a unidade do PDF (matched set); ignora a dependência entre pares do mesmo cluster |
| Sensibilidade do ABraOM | interação só nos pares com caso **e** controle fora do ABraOM, sem desfazer pares | casos 4,6× mais presentes no ABraOM que os controles (achado de 20/09); não é teste decisivo |
| Regra do limiar do ensemble | a do `calibrate_threshold` do Mosaic: maior MCC; empate → maior especificidade → maior limiar | alinhar com o protocolo; a regra das cabeças individuais (primeiro máximo) era nossa |

**Aberto — decisão científica do Eduardo, antes de qualquer score dos estudos (Mosaic §13.5):**

1. **Margem mínima de melhoria no coorte BR.** Qual delta (Δ_BR_full ou Δ_BR_matched), qual métrica (AUROC,
   AUPRC ou as duas), e qual regra (estimativa ≥ margem, ou limite inferior do IC ≥ margem).
2. **Margem máxima de regressão no controle** (Δ_control), com a mesma especificação.
3. **Painéis em que regressão é inaceitável.**
4. **Se a interação tem critério próprio** ou é só relatada com os absolutos.
5. **"Benefício não explicado por um único painel"** (condição 3 do Mosaic): o consumidor já mede o delta do coorte
   **sem** cada painel (`sem_painel:*`); falta a regra.
6. **Unidade da reamostragem** (cluster em conjunto × par, como o PDF previa). Réplicas e seed são detalhe da equipe.

**Para decidir as margens sem olhar o estudo (plano §6.5):** estimar a precisão esperada dos deltas com dados de
**desenvolvimento**, reamostrando as predições de M0 e MR do conjunto de seleção com a composição de cada coorte
brasileiro (rótulo × painel; o clínico é 90% P, o populacional 95% B). Isso dá a largura provável dos ICs e mostra
que margens são detectáveis, sem consultar nenhum candidato no estudo.

## 5. O manifesto do G6

Um arquivo `g6_manifesto.json` com o próprio sha256, que o consumidor exige. Conteúdo:

| Grupo | Conteúdo |
|---|---|
| Os 7 campos do `required_consumer_manifest` | `base_checkpoint_id` (R03 `best_checkpoint.pt`, passo 71.000, `f2983560…`); `regionalized_checkpoint_id` (R03 + a₁/a₂/a₃ por sha256); `base_training_dataset_id` (papel `train` de `g2_final_janela2048/core_head_snapshot.parquet`, derivado do `core_locus` v1 com as exclusões da §4.2); `base_training_dataset_hash` (sha256 do snapshot, o registrado na decisão do G5); `base_training_cutoff` (ClinVar 2026-06, o do release); `abraom_snapshot_hash` (`3cd33784…`); `regionalization_method` (rsLoRA r=8, α=16, 99 módulos, MLM em janelas ~60% gnomAD v4.1 joint / 40% ABraOM, 3.000 passos, 5e-6, checkpoint pelo `focal_alt`; planos `c76d08d4…`/`034eca34…`) |
| Sistemas | cada componente: arquivo e sha256 da cabeça (da conferência), adapter e sha256, cache de desenvolvimento e sha256 da identidade |
| Conferências | as três conferências de cabeças passaram; as cabeças do M0 idênticas nos três comparadores; os três adapters com a mesma receita, o mesmo recorte de validação e os mesmos planos |
| Limiares | do ensemble, por sistema, com a regra e a proveniência (fold 1, n_P, n_B) |
| Margens e regra de decisão | as da seção 4, já decididas; **sem elas o manifesto não congela** |
| Bootstrap | unidade, réplicas, seed, regra da interação e sensibilidades |
| Código | sha256 do consumidor (`eval/campanha/estudos.py`, `metricas.py`, `cabeca.py`) e do script de pontuação |
| Sobreposições declaradas | gnomAD como outra fonte de regionalização; alelos dos estudos fora do pool do adapter; membros, clusters e regra ampla brasileira fora do treino da cabeça; `br_population_observed` sobreposto ao ABraOM por construção |

## 6. Extração dos estudos

As variantes dos estudos passam pelo **mesmo** caminho numérico do desenvolvimento: as funções do extrator
(`montar_sistema`, `identidade`, `rodar_extracao`), sem mudar nenhum dos 12 arquivos da identidade. O consumidor
confere que a identidade de cada cache dos estudos é igual à do cache de desenvolvimento do mesmo sistema em tudo
menos a tabela. Tabela: uma linha por variante (quem está nos dois estudos é extraído uma vez; os rótulos e o
cluster vêm do release e são os mesmos). O extrator recusa membros dos estudos por construção: a extração dos estudos
é um script próprio, que exige o manifesto do G6.

## 7. O que o resultado poderá afirmar (plano §7)

- Um ganho sustenta transferência diferencial no recorte de **participação** brasileira do Mosaic, com um adapter
  populacional **misto** — não que o componente brasileiro é a causa (isso pede o MG).
- Não sustenta resultado em só-BR, em pacientes ou na população brasileira; o estudo populacional não é evidência
  independente (sobreposto ao ABraOM).
- A interação vem sempre com os absolutos: um positivo pode vir de o controle piorar mais.

## 8. Estado

| Peça | Estado |
|---|---|
| núcleo do consumidor (`eval/campanha/estudos.py`) | **escrito e testado com dados sintéticos** (`tests/test_campanha_estudos.py`, 19 testes); ~5 min por estudo com 1.000 réplicas |
| script de extração dos estudos | a escrever (depende só do extrator) |
| construtor do manifesto do G6 | a escrever (precisa de a₂/a₃ para o teste real; testes sintéticos antes) |
| script de pontuação e relatório do G7 | a escrever |
| análises adicionais pré-declaradas do plano §6.4 (controles com SCV brasileira, baselines de AF, exposição de locus, sanidade no fold 0 depois do congelamento) | a decidir se entram antes do G6 ou ficam declaradas como posteriores |
