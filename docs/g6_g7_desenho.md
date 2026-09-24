# G6 e G7 — congelamento e avaliação única nos estudos brasileiros

Estado: **24/09** (escrito em 23/09, enquanto a₂ e a₃ treinavam; construtor do G6 em 24/09). Nenhum score dos
estudos foi calculado.
Separa o que as regras de avaliação do Mosaic **fixam**, o que já foi **declarado** por nós, o que está
**proposto** (confirmar no G6) e o que está **aberto** (decisão do Eduardo, antes do G7).

**Esta campanha é um protocolo derivado.** O consumidor aplica as regras de **avaliação** do Mosaic, mas a cabeça é
treinada e calibrada no `core_locus` do release, com exclusões próprias (autorizado pelo mantenedor em 15/09) — o
Mosaic publicado não prevê treino, seleção nem calibração no release. Descrever como "protocolo derivado", nunca como
cumprimento integral do protocolo publicado.

Fontes: `lumina-mosaic` commit `814e7f0` (`specs/PLAN.md` §13.3–13.5, `src/mosaic/protocol.py:brazil_protocol_section`,
`src/mosaic/comparator_eval/`); plano da campanha (`docs/proposta_mosaic_regionalizacao_desenvolvimento.md` §4.2,
§6–8); declaração `configs/campanha_r03_desenvolvimento.json` (seção `g6`).

## 1. Ordem

1. **a₂, a₃** treinados e congelados pela regra (menor `focal_alt`) → caches de desenvolvimento MR_a₂, MR_a₃ →
   comparadores → conferência das cabeças.
2. **G6**: manifesto congelado (seção 5). Depois disso, nada é ajustado.
3. **Extração dos estudos** para M0, MR_a₁, MR_a₂, MR_a₃ (~9 mil variantes; ~10 min por sistema), pelo mesmo
   caminho numérico do desenvolvimento (seção 6). Só com o G6 congelado.
4. **Pontuação** com as cabeças congeladas e **consumidor** (seção 2). Uma vez.

Reexecutar exatamente os sistemas congelados para conferir reprodução não invalida o estudo. Mudar qualquer coisa
por causa do resultado transforma a rodada seguinte em exploratória.

## 2. Regras de avaliação do Mosaic, como o consumidor as aplica (`eval/campanha/estudos.py`)

| Regra | Como está no consumidor |
|---|---|
| Estudos separados, nunca unidos | `avaliar_estudo` roda um estudo por vez |
| Coorte completo = `case` + `unmatched_case` → Δ_BR_full | `visoes()`, como o `brazil_views` do Mosaic |
| Casos pareados = `case` com controle bidirecional → Δ_BR_matched; controles → Δ_control | o pareamento é **reconferido**; quebrado, o consumidor recusa |
| Interação = Δ_BR_matched − Δ_control, sem `unmatched_case` | `interacao()`; cada delta na interseção de cobertura do próprio grupo, e a contagem de pares com os dois membros cobertos pelos dois sistemas |
| AUROC e AUPRC só com as duas classes | AUPRC = precisão média em degraus (a `average_precision_score` do Mosaic; um teste confere a definição) |
| n_P, n_B e cobertura sempre; deltas na **interseção** de cobertura; sem imputação | `comparar()`: cobertura de cada sistema no coorte e métricas na interseção |
| Relatório do coorte inteiro obrigatório; painéis como diagnóstico; **sem macro brasileira**; plof e synonymous como guarda; sem piso 50/50 | coorte inteiro + por painel no coorte completo (AUROC/AUPRC só em missense/splice/noncoding) + o coorte sem cada painel |
| Métricas com limiar só com limiar externo congelado, com proveniência | sem limiar declarado, são omitidas |
| Limiar: maior MCC na `validation_gold`; empate → maior especificidade → maior limiar (`config/suite.yaml`, `calibrate_threshold`) | `g6.limiar_do_mosaic` na média das probabilidades do ensemble, no fold 1 (a `validation_gold` do run 0), por sistema; congelado no manifesto |
| Bootstrap pareado por `overlap_cluster_id`, 1.000 réplicas, seed 20260901, percentis 2,5/97,5 | os **mesmos sorteios** para os dois sistemas (teste: transformação monótona dá delta 0 em toda réplica). Os ICs são **condicionais aos sistemas congelados**: reamostram variantes, não o treino dos adapters e das cabeças |
| No estudo clínico, relatar o subconjunto `present_abraom` | coorte completo com `present_abraom = true` |
| Taxa de pareamento e composição antes e depois | `pareamento` e `composicao` por coorte |

**A métrica principal do G7 não é a do desenvolvimento.** No desenvolvimento, o critério declarado foi a macro de
missense/splice/noncoding. No G7 o Mosaic manda relatar o **coorte inteiro** (AUROC e AUPRC) e trata os painéis como
diagnóstico, sem macro obrigatória. Na comparação exploratória, a métrica de coorte inteiro (a "AUROC geral") teve
IC todo abaixo de zero com a₁ (−0,0025 [−0,0051; −0,0006]), mas **não com a₂ nem com a₃** (−0,0009 [−0,0024;
+0,0007] e −0,0001 [−0,0023; +0,0020]; 24/09). Não é portão — o desenvolvimento não mede a pergunta regional —, e
o conjunto das três sementes, não a primeira sozinha, é o que entra na discussão das margens (seção 4).

**A interação subtrai os deltas observados.** Não remove, por si, confundimento nem diferenças de composição entre
casos e controles (por exemplo, a presença no ABraOM, 4,6× maior nos casos clínicos). Por isso ela sai sempre com os
valores absolutos de cada grupo e com as sensibilidades da seção 4.

## 3. Declarado por nós (seção `g6` da declaração, 23/09)

- **Composição final:** M0 = h11, h12, h13; MR = a₁+h11, a₂+h12, a₃+h13. As nove cabeças MR dos três comparadores
  não se misturam; nenhuma é escolhida depois de ver resultado (`tests/test_declaracao_g6.py`).
- **Predição do sistema:** média das três probabilidades calibradas.
- **Limiar do ensemble:** MCC na média das probabilidades no fold 1, por sistema, congelado no G6. Os limiares
  individuais das cabeças não servem.
- **Um gerador por análise**, com a seed declarada: o resultado de uma análise não depende de quais outras rodam.

## 4. Proposto (confirmar no G6) e aberto (Eduardo)

**Proposto — implementado e testado, a confirmar:**

| Item | Proposta | Pressuposto e limite |
|---|---|---|
| IC da interação, principal | clusters sorteados **em conjunto** sobre casos pareados + controles | preserva a dependência genômica (cluster inteiro); **não** preserva os pares — caso e controle em clusters diferentes saem em sorteios independentes |
| IC da interação, sensibilidade | sorteio por **par** | preserva o pareamento; **não** preserva a dependência entre pares do mesmo cluster |

A regra do limiar do ensemble saiu desta tabela em 24/09: não é proposta nossa. O `config/suite.yaml` do Mosaic a
fixa (`threshold: metric mcc, on validation_gold, tiebreak [specificity, higher_threshold]`), conferida no código
(`calibrate_threshold`, candidatos logo abaixo do menor score, cada score distinto e logo acima do maior); a regra
das cabeças individuais (primeiro máximo) era nossa e não entra no G7.

Os dois ICs da interação saem juntos, como métodos de pressupostos diferentes. Uma unidade que preservasse as duas
coisas seria o componente conexo do grafo cluster–par; a viabilidade depende do tamanho desses componentes, que se
mede só com o membership, sem score — fica como opção para a decisão da unidade, não implementada.

**Análises secundárias pré-declaradas** (plano §6.4 e achado de 20/09). Nenhuma muda o resultado oficial, nenhuma é
escolhida depois dos resultados, todas mantêm pares inteiros; entrada ausente sai como `nao_calculada`, com o motivo:

| Análise | Definição | Entrada (hash no G6) | Estado |
|---|---|---|---|
| Pares fora do ABraOM | interação só nos pares com caso **e** controle fora do ABraOM (clínico) | membership | implementada |
| Controles com SCV brasileira | interação sem os pares cujo controle tem SCV de instituição da lista (44 de 3.116 no clínico), sem refazer o pareamento | `g2_regra_ampla/broad_brazilian_variant_ids.txt` | implementada |
| Exposição de locus | interação só nos pares com exposição empatada (diferença 0) de variantes de treino na janela, no snapshot final (`janela2048`) com raio 4.096 | `exposicao_por_membro.parquet` desse snapshot e raio (gerar se não existir) | implementada; raio e tolerância PROPOSTOS |
| Pares completos na cobertura | só se a cobertura desfizer pares: interação nos pares com os dois membros cobertos | scores | implementada |
| Métricas com limiar e Brier | com os limiares e calibradores congelados | manifesto | métricas com limiar implementadas; Brier a escrever |
| Baselines diagnósticas | fora dos sistemas, nos mesmos coortes e pares: (1) `gnomad_rarity`, o comparador **oficial** do Mosaic (`−gnomad_v4_af`, `not_found`/`ac0` com AF 0); (2) raridade no ABraOM (`−abraom_af`, ausente = 0; regra **nossa**, análoga); (3) presença no ABraOM com **papel explícito** — no clínico, diagnóstico da diferença de composição; no populacional ela **define** os grupos, é constante dentro de cada um e não se relata como discriminação | `pb_annotations.parquet` do release, por `variant_id` (hash lógico contra os invariantes da ADR 0006 declarados em `g6.proveniencia.release_do_mosaic` e contra o manifesto da cópia) | cobertura e proveniência a conferir (`scripts/conferir_cobertura_das_baselines.py`, só contagens); depois escrever. Não retirar pela falta na membership: a informação está no release |
| Sanidade no fold 0 | AUROC/AUPRC de M0 e MR no teste do `core_locus`, só depois do G6 | extração do fold 0 depois do congelamento | a escrever |

**Por que a baseline contínua fica (revisão de 24/09).** Eu tinha proposto retirar a de AF contínua porque ela "não
está na membership". Não se sustenta: o release traz `pb_annotations.parquet`, uma linha por `variant_id` da suíte,
com `gnomad_v4_af`, `gnomad_status`, `abraom_af` e `abraom_status`, e o `gnomad_rarity` é um dos comparadores
oficiais do Mosaic, calculado dessas colunas sem nova leitura de VCF. Duas propriedades de construção limitam a
leitura, e vão declaradas: o pareamento casa `gnomad_af_bin`, então dentro do par o `gnomad_rarity` só difere dentro
da faixa; e no estudo populacional a presença no ABraOM define os grupos (casos presentes, controles ausentes), o que
torna a presença constante em cada grupo e a raridade no ABraOM constante (0) nos controles. Nenhuma métrica de
baseline nos estudos antes do G7.

**Aberto — decisão científica do Eduardo, antes de qualquer score dos estudos (Mosaic §13.5):**

1. **Margem mínima de melhoria no coorte BR.** Qual delta (Δ_BR_full ou Δ_BR_matched), qual métrica (AUROC, AUPRC ou
   as duas) e qual regra (estimativa ≥ margem, ou limite inferior do IC ≥ margem).
2. **Margem máxima de regressão no controle** (Δ_control), com a mesma especificação.
3. **Painéis em que regressão é inaceitável.**
4. **Se a interação tem critério próprio** ou é só relatada com os absolutos.
5. **"Benefício não explicado por um único painel"** (condição 3 do Mosaic): o consumidor já mede o delta do coorte
   sem cada painel (`sem_painel:*`); falta a regra.
6. **Unidade da reamostragem** principal. Réplicas e seed são detalhe da equipe.

Métrica, margem e regra se fixam juntas e antes; não se escolhe depois a combinação mais favorável.

**Como declarar (24/09).** A seção `g6.margens` da declaração traz o modelo com os campos nulos: cada regra é
`estatística >= limite` sobre um delta MR − M0, com `estudos`, `delta` (Δ_BR_full ou Δ_BR_matched na melhoria;
Δ_control no controle), `metrica` (AUROC ou AUPRC), `estatistica` (estimativa ou `p2_5`, o limite inferior do IC) e
`limite` finito (≥ 0 na melhoria, ≤ 0 na regressão); painéis entre missense, splice e noncoding (lista vazia só com
motivo); a condição 3 com `suporte_minimo_por_painel`; a interação com `criterio_proprio` explícito. O bootstrap
traz `unidade_principal` e `unidade_de_sensibilidade` (entre `cluster_conjunto` e `par`). O construtor recusa
congelar com qualquer campo nulo ou fora do domínio.

**Duas perguntas separadas para as margens.** (a) Qual melhora seria **cientificamente relevante**? É a decisão do
Eduardo, e não se reduz a margem para facilitar um resultado positivo. (b) Com os dados disponíveis, que melhora se
consegue **estimar com precisão**? Para (b), uma análise **limitada de cenários** com dados de desenvolvimento
(plano §6.5): reamostrar as predições de M0 e MR do conjunto de seleção com a proporção de rótulos e painéis de cada
coorte brasileiro. Ela **não** dá a largura dos ICs brasileiros: a incerteza real depende também do número e da
concentração dos clusters, da dependência entre casos e controles, da distribuição dos scores dentro dos painéis e
da correlação entre M0 e MR no estudo; e células escassas no desenvolvimento (plof/B tem 1 variante) não ganham
diversidade por repetição. Serve como ordem de grandeza, e não deve virar outra frente longa antes de avançar.

## 5. O manifesto do G6

Arquivo `g6_manifesto.json` e, **à parte**, `g6_manifesto.json.sha256` com o sha256 dos bytes do manifesto (um
arquivo não contém o próprio hash). O consumidor recusa um manifesto cujo sha256 não confira
(`g6.ler_manifesto_congelado`).

**Construtor: `scripts/construir_g6.py`** (regras puras em `eval/campanha/g6.py`). Não treina nada nem lê o fold 0
ou os estudos. Confere, antes de qualquer número: a composição contra o pareamento de sementes; cada componente
contra a conferência do seu comparador (mesmo sha256); o M0 dos comparadores da a₂ e da a₃ idêntico ao da a₁
(reconferido); os quatro caches (sistema, adapter congelado, o mesmo R03, M0 × MR só diferindo pelo adapter,
mesmas linhas). A **identidade das linhas vem de IDs e hashes**: cada cabeça aponta
(`cache_identidade_sha256`) para a identidade do seu cache, que fixa o hash de conteúdo da tabela, papel incluído
(reconferido ao carregar); os IDs do fold 1 são exatamente o papel `validation` do snapshot da política e os da
seleção exatamente os de `selecao_comum.parquet` (sha256 com o prefixo declarado), nas contagens declaradas.
Recarrega as seis cabeças e confere a reprodução: probabilidades da seleção contra `predicoes_selecao.parquet`,
métricas da seleção e do fold 1 contra o relatório do comparador, Platt e limiar da cabeça refeitos no fold 1 — isso
é **consistência numérica**, não o que sustenta a identidade das linhas (revisão de 24/09). Então: média das três
probabilidades por sistema, limiar do ensemble pela regra do Mosaic, e o **ensemble final no desenvolvimento**,
descritivo, com IC por cluster condicional aos sistemas treinados — o resultado não muda composição nem limiar.
Saídas: `g6_construcao.json`, `g6_predicoes.parquet`, `g6_manifesto_rascunho.json`; com `--congelar` e sem
bloqueio, `g6_manifesto.json` e o sha256. **Bloqueios** (o congelamento é recusado com qualquer um), que
conferem **conteúdo**, não o texto do estado: cada margem com estudos, delta, métrica, estatística (estimativa ou
limite inferior do IC) e limite finito com o sinal certo, a condição 3 do Mosaic com o suporte mínimo por painel e a
interação com `criterio_proprio` explícito (`g6.problemas_das_margens`); bootstrap com unidade principal e de
sensibilidade entre as implementadas, réplicas, seed e percentis (`problemas_do_bootstrap`); pendência `FEITO` com
`onde` e `RETIRADO` com `motivo`; revisão do git conferida e nenhum arquivo de código ausente, fora do git ou com
mudança fora do commit — **um git que falha bloqueia**, não vale como "sem mudanças" (`problemas_do_codigo`);
`abraom_snapshot_hash` reconferido no arquivo. Testes: regras puras
(`tests/test_campanha_g6.py`, sem torch) e ponta a ponta sobre caches sintéticos com G5, três comparadores e três
conferências (`tests/test_construir_g6.py`, com torch).

**Os campos do `required_consumer_manifest`, adaptados ao nosso desenho.** O Mosaic supõe um sistema base treinado
num snapshot global; aqui há três fontes de treino, que o manifesto declara separadas:

| Campo do Mosaic | No nosso desenho |
|---|---|
| `base_checkpoint_id` | R03 `best_checkpoint.pt` (LUM-20260719-001, run R03, passo 71.000, sha256 `f2983560…`) + as três cabeças do M0 (sha256 da conferência). O "sistema base" é R03 congelado + cabeça |
| `regionalized_checkpoint_id` | o mesmo R03 + adapters a₁/a₂/a₃ (sha256) + as três cabeças MR (sha256) |
| `base_training_dataset_id`, `_hash`, `_cutoff` | declarados em **duas partes**: (a) **pré-treino do R03** — a proveniência disponível; corpus e cutoff que não estiverem documentados ficam **"desconhecido"**, nunca preenchidos com a data do ClinVar; (b) **treino supervisionado da cabeça** — papel `train` de `g2_final_janela2048/core_head_snapshot.parquet` (sha256 da decisão do G5 e hash de conteúdo do G2), rótulos do release v1, cujo cutoff é o do ClinVar do release (2026-06) — cutoff **dos rótulos da cabeça**, não do pré-treino |
| `abraom_snapshot_hash` | `3cd33784…` (SABE1171, conferido em 20/09) |
| `regionalization_method` | rsLoRA r=8, α=16, 99 módulos, MLM em janelas ~60% gnomAD v4.1 joint / 40% ABraOM, 3.000 passos, 5e-6, checkpoint pelo `focal_alt`; **dados do adapter à parte**: pool ABraOM `40bd0f79…`, pool global `ce749a6d…`, plano `c99e5dae…`, separação por loco `c76d08d4…`/`034eca34…` |

**Exclusões, como de fato aplicadas** (política `janela2048`; o manifesto não pode prometer isolamento maior):

| O quê | Onde | Garantia |
|---|---|---|
| Membros dos dois estudos (casos, casos sem par, controles) | treino, validação e teste da cabeça | nenhum membro em papel algum |
| Variantes com SCV de instituição brasileira (regra ampla, 5.995) | treino, validação e teste | nenhuma |
| Não `sequence_eligible`; chr8 | os três papéis | nenhuma |
| Clusters da **seleção comum** (156) | treino, cluster **inteiro** | nenhum locus da seleção no treino |
| Variantes de treino a até **2.048 bp** de um membro | treino | nenhuma variante de treino **dentro** da janela de leitura de um membro. **Não** garante: ausência de sobreposição de janelas (resídua em ~25% dos membros) nem remoção do **cluster** do membro — a exclusão por cluster dos estudos foi abandonada por inviável (tirava 84,5% dos patogênicos do treino) |
| Alelos dos membros | pool e janelas do adapter | nenhum alelo dos estudos no treino populacional |

Exposição residual medida no snapshot final (raio 4.096, benignas do clínico): 0,4832 de "caso maior" entre os
pares diferentes, média +1,55, mediana absoluta 0 — descritiva; efeito sobre os modelos desconhecido.

**Resto do manifesto:** cada componente (arquivo e sha256 da cabeça, adapter, cache de desenvolvimento e sha256 da
identidade); as três conferências de cabeças; M0 idêntico nos três comparadores; os três adapters com a mesma
receita, o mesmo recorte de validação e os mesmos planos; os limiares do ensemble com regra e proveniência; margens
e regra de decisão (**sem elas o manifesto não congela**); bootstrap (unidade, réplicas, seed, regra da interação e
sensibilidade); as análises secundárias e as entradas delas, com hash; o sha256 do código do consumidor e do script
de pontuação; e as sobreposições declaradas (gnomAD como outra fonte de regionalização; `br_population_observed`
sobreposto ao ABraOM por construção).

## 6. Extração dos estudos

As variantes dos estudos passam pelo **mesmo** caminho numérico do desenvolvimento: as funções do extrator
(`montar_sistema`, `identidade`, `rodar_extracao`), sem mudar nenhum dos 12 arquivos da identidade. O consumidor
confere que a identidade de cada cache dos estudos é igual à do cache de desenvolvimento do mesmo sistema em tudo
menos a tabela. Tabela: uma linha por variante (quem está nos dois estudos é extraído uma vez; rótulo e cluster vêm
do release e são os mesmos). O extrator de desenvolvimento recusa membros dos estudos por construção: a extração dos
estudos é um script próprio, que exige o manifesto do G6.

## 7. O que o resultado poderá afirmar (plano §7)

- Um ganho sustenta transferência diferencial no recorte de **participação** brasileira do Mosaic, com um adapter
  populacional **misto**, num protocolo derivado — não que o componente brasileiro é a causa (isso pede o MG).
- Não sustenta resultado em só-BR, em pacientes ou na população brasileira; o estudo populacional não é evidência
  independente (sobreposto ao ABraOM).
- A interação vem sempre com os absolutos: um positivo pode vir de o controle piorar mais.

## 8. Estado

| Peça | Estado |
|---|---|
| núcleo do consumidor (`eval/campanha/estudos.py`) | escrito e testado com dados **sintéticos** (`tests/test_campanha_estudos.py`); ~5 a 8 min por estudo com 1.000 réplicas. Ainda não validado ponta a ponta nos artefatos da campanha |
| script de extração dos estudos | a escrever |
| construtor do G6 (`scripts/construir_g6.py`, `eval/campanha/g6.py`) | escrito em 24/09 e testado com dados sintéticos (regras puras no Windows; ponta a ponta com torch no notebook); rascunho a rodar nos artefatos; congelamento bloqueado até as margens, a unidade do bootstrap e as pendências |
| script de pontuação e relatório do G7 | a escrever |
| Brier, sanidade no fold 0 | a escrever (a sanidade no fold 0 fica para depois do congelamento) |
| baselines diagnósticas | conferência de cobertura e proveniência escrita (`scripts/conferir_cobertura_das_baselines.py`), a rodar; depois escrever no consumidor |
