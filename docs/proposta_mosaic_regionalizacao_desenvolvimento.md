# Plano: regionalização do R03 com o estudo brasileiro do Mosaic

Data: 2026-09-16 (atualizado com as respostas do Eduardo de 15/09) · Branch: `new_regionalization` ·
Status: **decisões A, B, C e D fechadas pelo Eduardo; E em aberto; parâmetros de execução a declarar.**

Relação com `contrato_v2_regionalizacao_r03.md`: este plano substitui no contrato a fonte de treino (§4, §7), a
decisão do conjunto de teste (§5), os detalhes de avaliação (§10) e a pendência 9 (§15). O restante do contrato
continua valendo e é citado aqui (objetivo do adapter, gerador de janelas, extração, gates).

Legenda: **[FIXADO]** decidido pelo Gabriel, pelo Eduardo ou exigido pelo protocolo do Mosaic · **[PROPOSTO]**
recomendação nossa · **[ABERTO]** decisão necessária antes de implementar.

---

## 0. Resumo

- **[FIXADO]** Backbone: R03 publicado (`best_checkpoint.pt`, passo 71.000). Benchmark: `croma-bioai/lumina-mosaic`.
- **[FIXADO — Eduardo, 15/09]** Avaliação nos dois estudos brasileiros do Mosaic, que respondem perguntas
  **diferentes** e nunca se somam: `br_clinical_evidence` mede **participação de instituições brasileiras** e é a
  avaliação principal; `br_population_observed` mede **presença no ABraOM**. O teste só-BR fica para depois.
- **[FIXADO]** Esta é uma **campanha derivada do Mosaic**, segundo a orientação do Eduardo: desenvolvimento no
  `core_locus` depois das exclusões e avaliação nos estudos brasileiros congelados. O código publicado continua
  declarando `release_training_allowed = False`; encaixar no formato `base` × `regionalized` não é o mesmo que
  cumprir integralmente o protocolo publicado, e a diferença é declarada, não contornada.
- **[FIXADO — Eduardo, 15/09]** **Um único adapter populacional misto**, treinado em global + ABraOM na proporção
  aproximada de 60% global / 40% Brasil. As etapas separadas (adapter global puro e adapter ABraOM puro) são
  puladas: ficam como ablação de atribuição, se o resultado principal for positivo.
- **[FIXADO — Eduardo, 15/09]** A cabeça clínica treina nos splits de **treino do `core_locus`** do release e é
  avaliada nos dois estudos brasileiros. Isso fecha a decisão B: é um protocolo derivado do release, autorizado por
  escrito pelo mantenedor do Mosaic em 15/09/2026.
- **[FIXADO — Eduardo, 15/09]** Objetivo do adapter: **MLM com máscaras em span sobre as mutações**, com as
  mutações em **posições aleatórias da janela**.
- **[ABERTO]** Decisão E (chr8 representacional e relato BRCA1/BRCA2/TP53) e os parâmetros de execução da seção 5.1.

---

## 1. Identidades

| Item | Identidade | Status |
|---|---|---|
| Backbone | `LUM-20260719-001-R03`, `best_checkpoint.pt`, passo 71.000, pesos sem EMA, em `s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt` (README e `config/lumina_r03_base.json` do `lumina-inference`). Não trocar pelo `final_checkpoint.pt` (passo 75.000). | **[FIXADO]**. sha256 do arquivo já registrado no contrato v1 (`f2983560…`); reconferir o arquivo carregado em cada run |
| Mosaic, código | `https://github.com/croma-bioai/lumina-mosaic`, commit `814e7f0a17ac45c9bd4a63958aafb3cffaddfe22`; ponta de `main` conferida em 14/09; clone do notebook no mesmo commit | **[FIXADO]**; reconferir a ponta antes de congelar |
| Mosaic, dados | release `clinvar-pb-capability-suite/v1` em `~/mosaic-v1/`. Já conferido: `membership.parquet` com o hash lógico de referência (`1c1cd65d…`, 8.875 linhas). O S3 ainda usa o layout anterior ao ADR 0006 (`bundle.manifest.json` em vez de `release.manifest.json`) | **[FIXADO]** como fonte; registrar o hash real de cada arquivo usado |
| ABraOM | snapshot `abraom_sabe1171`: `s3://croma-bioai-shared-data-us-east-2/lumina/lumina-mosaic/abraom/SABE1171.Abraom.clean.tsv`, 33,6 MB, colunas `[chrom, pos, ref, alt, af_abraom]` | **[FIXADO e CONFERIDO em 20/09]**: sha256 `3cd3378432909b80053d3a92a1b7d544053a51697623845f6a44f67ba7dbd9d6`, igual ao do `sources.yaml`. O índice ABraOM da v1 **não** é este objeto |
| gnomAD (parte global da mistura) | v4.1 joint com AF por grupo (`s3://ai4bio-lumina/data/external/gnomad-joint-v4.1/`) | **[PROPOSTO]**; fixar versão e grupos |
| Genoma | GRCh38, `hg38.fa` (sha256 `056974f6…`) | **[FIXADO]** |
| Snapshot de treino da cabeça | derivado do `core_locus` do release v1; ID, hash e cutoff a registrar (seção 4.2) | **[FIXADO]** como fonte; identidade a gerar |

Código e dados têm identidades separadas: ter o commit certo não prova que o release foi gerado por ele.

**Onde as fontes moram (medido em 20/09).** Das 19 fontes do source-lock com sha256 fixado, só o ABraOM está em
`s3://croma-bioai-shared-data-us-east-2/lumina/lumina-mosaic/` — é o bucket do arquivo de acesso restrito, não um
espelho do source-lock. As outras 18 (ClinVar, dbNSFP, conservação, SpliceAI, VEP, MANE, HGNC) continuam vindo das
suas origens públicas. `scripts/locate_abraom_source.py --root` refaz essa conferência quando for preciso.

---

## 2. Pergunta, sistemas e contrastes

**Pergunta desta campanha [FIXADO]:** a adaptação populacional com dados brasileiros somados aos globais produz
ganho diferencial em variantes com participação de instituições brasileiras, nos casos e controles congelados do
Mosaic?

| Sistema | Papel no Mosaic | Representação | Classificador clínico |
|---|---|---|---|
| **M0** | `base` | R03 original, congelado | H0 |
| **MR** | `regionalized` | mesmo R03 + **adapter misto ≈60% global / 40% ABraOM**, congelado depois de treinado | HR |
| MG (adiado) | — | mesmo R03 + adapter global puro (mesma receita, mistura 100/0) | HG |

**[FIXADO]**

- O checkpoint-base (ancestral comum) é o R03, não o sistema M0 com H0 já treinada.
- H0 e HR são treinadas separadamente, com os mesmos dados, a mesma arquitetura, o mesmo procedimento e o mesmo
  orçamento, depois de congelar a representação de cada sistema. Isso mede o benefício da adaptação no **sistema
  completo, depois de treinar sua cabeça**. Manter H0 fixa e trocar só a representação seria outro experimento.
- Pesos do R03 e cabeças nativas congelados. Sem adapter ClinVar, sem fusion e sem AF observada nas entradas da
  cabeça: AF e presença ficam como baselines diagnósticas separadas.
- As cabeças nativas do R03 já carregam informação populacional do pré-treino: M0 não é "sem informação
  populacional".

**Contrastes [FIXADO], registrados antes de qualquer resultado:**

| Contraste | Papel |
|---|---|
| MR − M0 | pergunta principal, no formato `base` × `regionalized` do Mosaic |
| MR − MG | ablação de atribuição, **adiada**: separa o efeito do componente brasileiro do efeito de adaptação genérica |

**Custo declarado da decisão de pular etapas.** Com um adapter misto, um ganho positivo sustenta "a adaptação
populacional mista ajuda nas variantes com participação brasileira"; **não** sustenta "o ABraOM ajuda além do
global". A ablação MG não exige código novo: a proporção da mistura é um parâmetro, e o braço global puro é a mesma
receita com 100/0. Custa uma execução de adapter mais uma de cabeça, quando o resultado principal justificar.

---

## 3. O que o estudo brasileiro do Mosaic exige

Verificado em `PROTOCOLO.md` (Estudo brasileiro), `src/mosaic/protocol.py:brazil_protocol_section`,
`src/mosaic/brazil_study.py`, `config/suite.yaml`, `docs/GUIA_OPERACIONAL_DE_SCORING_DOS_ESPECIALISTAS.md` (§5.3 e §6)
e `src/mosaic/comparator_eval/`.

| Regra | Consequência para o plano |
|---|---|
| Modo `frozen_pair_evaluation`; sistemas `base` e `regionalized`; nenhum treino, seleção ou calibração dentro do estudo | sistemas e limiares congelados antes de pontuar o estudo |
| `release_training_allowed`, `release_model_selection_allowed` e `release_threshold_calibration_allowed` = `False` | treinar a cabeça no `core_locus` é protocolo derivado, **autorizado pelo mantenedor em 15/09** e registrado no manifesto |
| Dataset de treino = snapshot global do consumidor, declarado por ID, hash e cutoff, **não materializado no release** | o nosso é derivado do release: declarar origem, exclusões e hash próprio (4.2) |
| O regionalizado parte do mesmo dataset e do mesmo checkpoint-base; fonte obrigatória `abraom_sabe1171` | R03 comum; HR treinada no mesmo snapshot de H0; a mistura inclui o ABraOM exigido e **declara o gnomAD** como outra fonte de regionalização |
| Dois estudos, nunca unidos: `br_clinical_evidence` (consensus com participação brasileira) e `br_population_observed` (gold presente no ABraOM) | o clínico é a avaliação principal; o populacional dá evidência sobre o comportamento do sistema naquele recorte, mas não sobre generalização para a população brasileira — e está sobreposto ao ABraOM por construção, o que reforça a regra de alelos da 4.3 |
| Pares 1:1 sem reposição em rótulo × painel × bin de AF do gnomAD, sem gene; papéis `case`, `unmatched_case` e `control` | importar os pares, nunca refazer |
| Deltas no coorte completo, nos casos pareados e nos controles; interação `delta_br_matched − delta_control`, sem os casos não pareados; deltas na interseção de cobertura | seção 6 |
| AUROC e AUPRC; métricas com limiar só com limiar externo congelado; macro brasileira não exigida | seção 6 |
| No estudo clínico, relatar também o subconjunto `present_abraom` | obrigatório |
| Declarar sobreposições: SCVs ou instituições brasileiras, variantes do estudo, outras fontes de regionalização | inclui o gnomAD da mistura |
| Bootstrap por `overlap_cluster_id`, 1.000 réplicas, percentis 2,5 e 97,5, seed `20260901` | o `comparator_eval` reamostra cada coorte separadamente e **não calcula a interação**: a regra conjunta é do consumidor (6.3) |
| `lockbox = False`; `independent_generalization_claim_allowed = False`; margens de sucesso declaradas antes de ver scores | campanha de desenvolvimento |

O `comparator_eval` também é fechado nos sete comparadores oficiais (`OFFICIAL_COMPARATOR_IDS`): dele reaproveitamos
funções (`brazil_views`, `_with_scores`, `resample_indices`, `metrics_for_panel` com spec sintética) e constantes, não
a CLI.

---

## 4. Dados

### 4.1 Estudo brasileiro [FIXADO]

Importar `studies/brazil/membership.parquet` sem refazer o pareamento e validar: IDs únicos por estudo e papel,
ligações bidirecionais entre caso e controle, rótulos e estratos coerentes com `pb_examples` e `pb_panels`.

Contagens do release (8.875 linhas):

- clínico: 3.119 casos (3.116 pareados e 3 sem controle; 2.808 P / 311 B no coorte completo) e 3.116 controles;
- populacional: 1.889 casos (751 pareados e 1.138 sem controle) e 751 controles.

O `variant_id` é um hash opaco: para pontuar é obrigatório juntar com `pb_examples.parquet` e obter `chrom`,
`pos_1based`, `ref` e `alt`. O tamanho efetivo sai da recontagem de P/B por coorte e painel **depois da cobertura**
de cada sistema, não destas contagens.

### 4.2 Snapshot de treino da cabeça [FIXADO — Eduardo, 15/09: "treina core_locus e avalia neles"]

Fonte: partição `core_locus` do release v1, na agenda oficial de cinco execuções
(`docs/GUIA_OPERACIONAL_DE_SPLITS.md` §2 e §3). **[PROPOSTO]** usar `run_id = 0` e declará-lo:

| Papel | Recorte | n de referência | P | B |
|---|---|---:|---:|---:|
| treino | `core_fold` 2, 3, 4 · gold + consensus | 196.096 | 33.902 | 162.194 |
| validação | `core_fold` 1 · só gold | 2.453 | 1.182 | 1.271 |
| teste do core | `core_fold` 0 · só gold | 1.758 | 1.199 | 559 |

Os números são de referência do guia e são o **ponto de partida**, não o tamanho do treino: recontar depois das
exclusões.

**Papel de cada recorte [FIXADO]:**

| Recorte | Uso permitido |
|---|---|
| folds 2, 3 e 4 | treinar a cabeça |
| fold 1, gold | early stopping e calibração (Platt e limiar); diagnóstico complementar |
| fold 0, gold | avaliar **depois** de congeladas todas as escolhas |
| estudos brasileiros | avaliar os sistemas já congelados |

O fold 0 não participa de nenhuma escolha: se orientar qualquer mudança, deixa de ser teste reservado e vira
desenvolvimento.

Regras obrigatórias sobre esse recorte:

1. **Excluir todos os membros dos dois estudos brasileiros** (casos, casos sem controle e controles) e **todas as
   variantes dos seus `overlap_cluster_id`**, nos **três** papéis — treino, validação/calibração e teste reservado.
   A exclusão por cluster é a mesma unidade atômica que o `core_locus` usa, então não quebra a agenda de folds. Vale
   em especial para o `br_population_observed`, que é gold e portanto cai justamente na validação e no teste gold.
   Sem isso, `study_membership_used_for_training = False` e `study_labels_used_for_training = False` ficam violados
   e o estudo perde a validade.
2. **Excluir qualquer variante com SCV de instituição da lista brasileira do Mosaic**, de qualquer classificação,
   origem ou contribuição (regra ampla; ausência de informação não vira "não brasileira").
3. Aplicar `sequence_eligible = True` nos três papéis (o modelo exige janela de sequência; os 118 inelegíveis do
   release são consensus).
4. chr8 fora, se continuar reservado (decisão E).

#### Vizinhos de cluster: medido em 16/09, política declarada antes de treinar

Executado no release real (`run_id=0`), o custo da exclusão dos **vizinhos de cluster** dos membros é:

| Recorte | Antes | Membros | Vizinhos de cluster | Sobra se excluir vizinhos |
|---|---:|---:|---:|---:|
| treino | 196.010 (33.897 P) | −5.164 (3.412 P) | **−153.663 (28.656 P)** | 35.044, só **1.704 P** |
| validação | 2.453 (1.182 P) | −741 (37 P) | −1.710 (37 clusters de 38) | ~1 variante |
| teste do core | 1.758 (1.199 P) | −400 (25 P) | −1.358 (29 clusters de 31) | ~0 |

Atribuição correta das perdas no treino: a exclusão de vizinhos sozinha leva **28.656 dos 33.897 patogênicos
iniciais (84,5%)**, ou 94% dos que restam depois da exclusão dos membros; os ~95% acumulados até 1.704 P somam
todas as exclusões. Os clusters do Mosaic são componentes conectados em até 32 kb, e os membros cobrem os genes
clinicamente sequenciados — que é onde vivem os patogênicos do ClinVar.

Excluir vizinhos **nos três recortes** é inviável: zera a validação. Excluir **só no treino** é uma proteção real
(a cabeça não treina em clusters presentes no estudo) e treinar e avaliar em loci disjuntos é justamente o desenho
de transferência do `core_locus` — o problema é o custo e a mudança de distribuição, não incoerência. 1.704
positivos não provam, por si, que o treino seria inútil; provam que a composição precisa ser estudada antes.

**Proposta [a declarar antes de treinar, ainda não aprovada]:** rodar `nenhum` como **diagnóstico** — excluindo
sempre membros, regra ampla e chr8 — e **medir** a exposição de locus
(`scripts/measure_study_locus_exposure.py`) em duas unidades: o `overlap_cluster_id` e a **janela real** de
±2.048 bp, porque componente conectado encadeia variantes distantes e compartilhar cluster não é compartilhar a
janela de 4.096 bp que o modelo lê. A medida principal é a comparação par a par entre caso e controle,
estratificada por rótulo e painel, publicando maior, menor e empate.

**Medido em 17/09, com a política `nenhum` (treino 183.779, 29.192 P):** no agregado a exposição é simétrica entre
casos e controles (mediana 6 nos dois, 49,2% de "caso maior" entre os 2.510 pares diferentes). **Estratificado, não
é:** nas 2.805 patogênicas fica em 48,0%, mas nas **311 benignas** o caso tem mais exposição em 141 pares contra 93
(60,3% dos diferentes), com média +44 variantes na janela. Por painel tudo fica perto de 0,5. No estudo
populacional a diferença é maior (58,4%; mediana 27 contra 7), o que é esperado de "gold presente no ABraOM".

Estes números são **descritivos**. Um teste de sinal trataria os pares como independentes, e eles não são: pares
diferentes compartilham clusters, regiões e as mesmas variantes de treino vizinhas. O que fica registrado é que a
assimetria existe, que ela está na classe escassa do estudo clínico, e que **o efeito dela sobre o desempenho é
desconhecido**.

Por isso o G2 ganhou `--window-exclusion-bp`, com a regra declarada explicitamente:

| Raio | O que garante | O que não garante |
|---|---|---|
| 2.048 (L/2) | nenhuma variante de treino **dentro** da janela de leitura de um membro | janelas de treino ainda podem sobrepor a do membro (dois centros a 3.000 bp compartilham ~1.096 bp) |
| 4.096 (L) | **nenhuma sobreposição de sequência** entre a janela de um membro e a de uma variante de treino | nada sobre validação, calibração ou o treino populacional do adapter |

Medir exposição zero com o mesmo raio da exclusão é **verificação de implementação**, não validação independente.
Essa verificação foi feita em 17/09: no snapshot de 4.096 medido a 4.096, todos os 3.116 casos e controles ficam em
zero (`fracao_empate = 1`, `pares_diferentes = 0`).

#### Custo medido das três políticas candidatas (17/09, `run_id=0`)

| Candidato | Treino | P | % do P | Assimetria nas benignas | Sobreposição de janela |
|---|---:|---:|---:|---:|---|
| `nenhum` | 183.779 | 29.192 | 100% | 0,603 (média +44, medido a 2.048) | existe |
| janela 2.048 | 103.884 | 9.374 | 32% | **0,500** (média +1,7, medido a 4.096) | resídua em 25% dos membros |
| janela 4.096 | 89.359 | 7.022 | 24% | sem pares diferentes | **zero por construção** |
| cluster (só treino) | 35.044 | 1.704 | 6% | — | não medida; inviável por outros motivos |

Relativo a `nenhum`, a janela de 2.048 remove **43,5% das linhas e 67,9% dos patogênicos** (os denominadores têm
de ser o `nenhum`, não os totais anteriores às demais exclusões).

Com **régua única** (exposição medida a 4.096 nos dois), nas 311 benignas do estudo clínico:

| Benignas, medido a 4.096 | `nenhum` | janela 2.048 |
|---|---:|---:|
| caso maior entre os pares diferentes | 0,627 | **0,500** |
| média da diferença | +62,4 | +1,7 |
| mediana absoluta | 24 | 1 |
| empate | 24,1% | 49,8% |

Nas patogênicas, 0,481 → 0,491. Isso é **redução grande de magnitude e equilíbrio na direção das diferenças nessa
medida**; não é igualdade de distribuições nem ausência de efeito sobre os modelos.

A verificação de zero foi conferida em **todos os membros**, não só nos pares clínicos: os seis grupos
(caso, controle e caso sem par, nos dois estudos, incluindo os 1.138 não pareados) ficam em `fracao_zero = 1,0`.

Validação (1.575) e teste (1.250) são **idênticos** nos três, porque a exclusão por janela só atinge o treino.
Os três são **aninhados**: treino(4.096) ⊂ treino(2.048) ⊂ treino(`nenhum`) — uma passagem de extração cobre os três
**por sistema**: M0 usa um cache, MR precisa do seu, porque o adapter muda os embeddings.

#### Artefatos materializados [FIXADO em 17/09]

| Artefato | Conteúdo | Identidade |
|---|---|---|
| `g5_comum/selecao_comum.parquet` | 2.799 variantes em 156 clusters, do candidato 4.096, **com coordenadas** | `sha256 693eb234…` |
| `g2_final_nenhum/` | treino 167.346 (24.097 P), 1.614 clusters | `hash_conteudo 89517c95…` |
| `g2_final_janela2048/` | treino 99.992 (8.421 P), 1.576 clusters | `f3f7416f…` |
| `g2_final_janela4096/` | treino 86.560 (6.355 P), 1.542 clusters | `4805b4fe…` |

Verificado nos três: **zero** variantes e **zero** clusters em comum com o conjunto de seleção, `final_para_treino`
verdadeiro, e validação (1.575) e teste (1.250) idênticos. A seleção foi regerada em 17/09 para incluir as
coordenadas (sem elas o conjunto pontuado não produz janela); a lista de clusters saiu **idêntica**, então os três
snapshots finais e os seus hashes não mudaram — só o da seleção. Os hashes declarados ficam em
`g2_verificacao/sha256_declarado.json` e o portão `verify_campaign_artifacts.py` os confere com `--esperado`. Os tamanhos batem com o custo previsto antes da
materialização.

Exposição recalculada **depois** da reserva (raio 4.096, benignas do estudo clínico): `nenhum` 0,6205 com média
+58,2; `janela2048` 0,4832 com média +1,55 e mediana absoluta 0; `janela4096` exatamente zero, com empate em 1,0
nos dois estratos. Tirar dados não aumenta exposição individual, mas mexe no equilíbrio caso × controle — por isso
o número que vale é este, medido no artefato que será usado.

#### Como a política será escolhida [PROPOSTO — declarar antes de treinar]

Não por argumento: por medição que **não toca o estudo brasileiro**. Papéis, declarados antes de treinar:

| Conjunto | Papel |
|---|---|
| treino de cada candidato | ajusta os pesos da cabeça |
| **seleção comum reservada** | compara extrações **e políticas**, com critério declarado antes |
| fold 1 gold | calibração (Platt e limiar) e diagnóstico complementar; **early stopping acontece aqui**, declarado |
| fold 0 gold | avaliação só depois do congelamento |
| estudos brasileiros | comparação final, congelada |

A comparação entre as três políticas usa o **conjunto de seleção comum**, não o fold 1 — foi justamente a fragilidade
do fold 1 (10 benignas de splice em 2 clusters) que motivou o conjunto reservado. Regra proposta: **escolher a
política mais isolada cuja macro-AUROC no conjunto de seleção fique dentro de 0,01 da melhor**; empate resolve a
favor do mais isolado. Os 0,01 são **regra operacional declarada, não prova de equivalência estatística**. A escolha
é feita **só com M0** — não depende do ABraOM — e a mesma política é aplicada a MR. A comparação usa a **média de
3 seeds** por candidato, com as mesmas sementes em todos, e as extrações candidatas são as duas da §5.2; a margem de
0,01 e a regra de desempate ficam fixadas aqui, antes de qualquer score.

#### Conjunto de seleção comum [DECLARADO em 17/09, antes de qualquer treino]

Duas partes, porque definir só por cluster faria cada política ser avaliada num conjunto de variantes diferente:

- **o que se pontua:** as variantes daqueles clusters **no candidato mais restritivo** (4.096) — idênticas nos três;
- **o que se exclui do treino:** os **clusters inteiros**, em cada candidato, para que nenhum treino contenha um
  locus que aparece na seleção.

Parâmetros: alvo **100 por célula** com **mínimo de 20 clusters por célula** → **156 clusters**, 2.799 variantes
pontuadas. Custo do mesmo conjunto, recontado em cada candidato (os mesmos clusters carregam mais variantes onde há
menos exclusão):

| Candidato | Reservado | Treino depois | P | `noncoding/P` |
|---|---:|---:|---:|---:|
| `nenhum` | 16.433 | 167.346 | 24.097 | 419 |
| janela 2.048 | 3.892 | 99.992 | 8.421 | 124 |
| janela 4.096 | 2.799 | 86.560 | 6.355 | 71 |

O conjunto é majoritariamente consensus, e essa diferença de tier em relação ao fold 1 (todo gold) fica registrada.
`noncoding/P` no candidato de 4.096 fica em 71 — suporte muito limitado. Consequência declarada: a comparação mede
o efeito **combinado** da política de isolamento e da reserva comum, que é o que de fato seria usado; se o candidato
mais isolado perder por falta de dados de treino, esse é o resultado da regra, não um defeito dela.

**O que essa medida não resolve:** snapshot compartilhado **não** faz o risco desaparecer no contraste M0 × MR —
as representações são diferentes e podem aproveitar os mesmos loci de formas diferentes, então exposição igual não
implica efeito igual. A assimetria caso × controle é um mecanismo, não o único. Aceitar exposição numa campanha de
desenvolvimento é diferente de demonstrar que ela é inofensiva, e a distinção fica declarada nas afirmações (§7).

**`pronto_para_congelar` do G2 não decide isto:** ele diz que as entradas são válidas e que as checagens da
política escolhida passaram, não que a política de isolamento por locus foi aprovada.

Medir e registrar o custo de cada exclusão. O G2 verifica, depois de aplicá-las: nenhuma variante dos estudos em
qualquer um dos três recortes (e nenhum cluster, onde a política os excluir); tamanhos e P/B por painel em cada
papel; e presença das duas classes em missense, splice e noncoding na validação. Se a validação não sustentar a seleção planejada, isso se resolve
**antes** de treinar — e nunca trocando de fold depois de ver resultado de modelo.

Identidade do snapshot para o manifesto: ID próprio, hash lógico das linhas, cutoff = o do release (ClinVar
2026-06), origem = "derivado do `core_locus` do release v1, segundo a orientação do mantenedor em 15/09/2026", e a
lista de exclusões aplicadas.

### 4.3 Dados populacionais [PROPOSTO]

- ABraOM: o snapshot do source-lock, conferido por sha256 em 20/09. gnomAD v4.1 por grupo para a parte global da
  mistura; o `af_gnomad` antigo, condicional ao índice ABraOM, não serve.

**O lado global é de outra ordem de grandeza [medido em 20/09].** O gnomAD v4.1 joint em
`s3://ai4bio-lumina/data/external/gnomad-joint-v4.1/` são 48 objetos, um VCF por cromossomo com o seu `.tbi`: só o
chr1 tem **72 GB**, e o conjunto passa de meio terabyte. Ler tudo como se fez com o TSV de 33 MB do ABraOM não é
opção. Duas estratégias, com custos diferentes:

| Estratégia | Custo | O que ela introduz |
|---|---|---|
| varredura completa com subamostragem por bin | uma passada por cromossomo (horas de I/O) | nenhum viés adicional |
| amostragem por região via índice `.tbi` | minutos, lê ~1% | o viés da escolha das regiões, que precisa ser declarado |

**Escolhida a amostragem por região**, por custo. O `copy_local: false` do `sources.yaml` proíbe **copiar** o VCF
naquele fluxo e o `specs/GNOMAD_S3_READ.md` explica por que o join não faz scan (817 GiB; ausência de alelo é
`not_found`, não se prova lendo tudo). Nada disso proíbe uma varredura remota para montar pool — é padrão de
acesso **novo**, que o contrato do lookup não cobre e que entra declarado no manifesto. As regiões são sorteadas
por semente a partir dos comprimentos dos cromossomos e **nunca a partir dos loci do benchmark**.

**Medido no piloto de 20/09 (40 regiões, 800 kb lidos):** o gnomAD tem ~**200 variantes por kb** — duas ordens
de grandeza acima do pool do ABraOM (448/Mb). O espectro é dominado pelo raro: **96,3% no bin ≤ 0,001** e **0,15%
acima de 0,5**. Cortar o lado global no piso do ABraOM (4,27 × 10⁻⁴) removeria **95,1%** do que foi amostrado, todo
do bin mais raro.

Isso tem uma consequência de desenho que o número sozinho esconde: **o mesmo bin não significa a mesma coisa nas
duas fontes.** Uma AF entre 4,3 × 10⁻⁴ e 10⁻³ é 1 a 2 cópias em 2.342 alelos no ABraOM, e algumas centenas de
cópias em ~1,6 milhão no gnomAD. Estratificar pela mesma grade iguala o rótulo, não a quantidade observada.

Por causa da desproporção entre bins, a coleta usa **dois limites declarados**: teto por bin **dentro de cada
região** (contra desequilíbrio de ligação — variantes vizinhas não são observações independentes) e **reservatório
por bin** ao longo de todas as regiões (senão encher o bin comum exigiria guardar milhões de linhas do bin raro).
O relatório publica `vistos_por_bin` e `fracao_amostrada_por_bin`: a fração amostrada de cada bin **é** o viés da
receita, declarado em número.

**Segundo piloto (200 regiões, 4 Mb lidos, `--por-bin 3000`) — a receita está dimensionada.** Cinco dos sete bins
saturaram a capacidade; os dois que não são `(0,05; 0,1]` com 1.922 e `(0,5; 1,0]` com 2.010. Uma **explicação
plausível** é o espectro de frequências em U, com o excesso perto de 1,0 vindo de `AF_joint` ser do alelo ALT —
em muitos sítios o minoritário é o alelo da referência. Mas **estes números não demonstram isso**: os bins têm
larguras diferentes e os rendimentos passaram por filtro de `FILTER`, teto por região e reservatório. O que se
pode usar com segurança é operacional: o bin `(0,05; 0,1]` é o escasso e rende **~9,6 por região**, e é ele que
dimensiona a rodada.

Com casamento por cromossomo, a maior diferença de geografia contra o ABraOM caiu para **−0,0123** (chr16); com
40 regiões era −0,0356. O resíduo é arredondamento da alocação, e diminui com mais regiões.

As exclusões dispararam em dado real, em volume pequeno mas não nulo: 1 membro de estudo, 2 alelos de avaliação,
4 com AF nas extremidades — a maquinaria está ligada, não só compilando.

**O custo do piso, e o que ele NÃO mede.** Sobre o pool estratificado o corte é 15,65%, e cai inteiro dentro de
um único bin, onde remove 98,6%. Mas isso mede **filtrar o pool depois de pronto** — e essa não é a operação que
está em discussão. Ao **reextrair** com `--af-min`, o piso entra dentro de `linhas_do_registro`, ou seja, **antes
do teto por região e do reservatório**: as variantes abaixo dele nem disputam vaga, e o bin se enche das
elegíveis. Quantas elegíveis existem, só outro piloto responde. O campo `custo_de_casar_o_piso` carrega esse
aviso para quem o ler fora de contexto.

**Pool global definitivo [MATERIALIZADO em 21/09].** 2.500 regiões de 20 kb, 50 Mb lidos (~1,6% do genoma),
**21m51s**. Resultado: **139.495 variantes**, `sha256 ce749a6d…`, com os sete bins praticamente cheios
(20.000 / 19.964 / 19.929 / 19.945 / 19.861 / 19.925 / 19.871). Receita: `AF_joint`, sem piso,
`--geografia casado_ao_abraom`, teto de 20 por bin por região, reservatório de 20.000 por bin, semente 20260920,
chr8 reservado.

A geografia convergiu: maior diferença contra o ABraOM de **−0,0073** (chr15), depois de −0,0356 com 40 regiões
e −0,0123 com 200. O resíduo é arredondamento da alocação por cromossomo.

As exclusões voltaram a disparar em dado real: **10 membros de estudo, 11 alelos de avaliação, 31 nas extremidades
de AF**. Volume pequeno — 52 em 139.547 — mas o que importa é que nenhum deles chegaria ao adapter.

E o piso confirmou-se na escala definitiva: removeria **19.702 de 20.000 do bin mais raro (98,5%)**, e nada dos
outros seis. Custo de reamostrar com piso: 22 minutos. **Barato o bastante para materializar as duas versões**
quando a decisão for tomada, em vez de escolher no escuro.

**Mistura 60/40 materializada [21/09] e a politica para janela invalida, FECHADA.** O plano com as duas metades
saiu exato: 50.000 janelas, 30.000 globais e 20.000 do ABraOM, `fracao_global_efetiva_nas_linhas = 0,6`, sete bins
de AF a ~7.143 cada. A auditoria contra o hg38 encontrou **42 `non_acgt` em 50.000 (0,084%)** — como esperado, o
lado global vem de regiões sorteadas às cegas e algumas caem em trecho com `N`; o lado do ABraOM não produz isso
porque suas variantes vêm de um callset.

**Política decidida: repor, não descartar.** Descartar encolheria o plano e deslocaria a mistura e a
estratificação — que são exatamente o que a receita declara. Com `--fasta`, o gerador confere cada janela e repõe
a inválida por outra **da mesma fonte e do mesmo bin de AF**, em até `--max-rodadas-de-reparo` rodadas, e publica
`substituicoes_por_fonte`. O que não tiver reposição no estrato sai declarado em `janelas_sem_reposicao` e
bloqueia `pronto_para_campanha`.

**O viés que isso introduz, declarado:** variantes cuja janela de 4.096 bp contém base fora de ACGT ficam
sistematicamente de fora do treino do adapter. É inevitável — o modelo não lê essa janela de qualquer forma — mas
é viés, não neutralidade, e por isso está escrito no manifesto em vez de apenas acontecer.

**Resultado [21/09]: plano final `sha256 c99e5dae…`** — 50.000 janelas, 30.000 globais e 20.000 do ABraOM,
`fracao_global_efetiva_nas_linhas = 0,6`, sete bins a ~7.143, `janelas_sem_reposicao = 0`, e a reauditoria contra
o hg38 voltou **50.000 `ok`, zero descarte**. As substituições foram **42, todas do lado global e nenhuma do
ABraOM**. A explicação provável é que variante de callset já mora em região chamável, mas **isto vale para esta
amostra e não é regra**: uma posição pode ter chamada válida e ainda ter `N` no contexto de 4.096 bp ao redor.

**Separação populacional do adapter [IMPLEMENTADA em 21/09]:** `scripts/split_adapter_plan_by_locus.py`. A
unidade de separação é o **loco**, não a janela — duas janelas de 4.096 bp a 500 bp de distância compartilham 87%
da sequência, então separar ao acaso mediria memorização do mesmo trecho. Janelas cujas janelas se sobrepõem são
encadeadas por ligação simples e o loco inteiro vai para um lado só; o encadeamento é necessário, não conservador
demais (se A cobre B e B cobre C, mandar A e C para lados opostos obrigaria B a sobrepor um deles).

A escolha dos locos é gulosa por célula `fonte × bin de AF`, para que a mistura 60/40 e o espectro sobrevivam nos
dois recortes. E **a disjunção é verificada, não assumida**: uma varredura por cromossomo falha com código 2 se
qualquer janela de validação tocar qualquer janela de treino. Sem ela, um erro de encadeamento passaria como
"separado" e contaminaria a única medida honesta do adapter.

O que isso **não** dá: independência estatística. Locos distintos ainda podem ser parecidos — parálogos,
repetições, famílias de genes. O que está garantido é ausência de **sobreposição de sequência**, que é o
vazamento grosseiro.

**Estado correto: existe um plano CANDIDATO 60/40 com sequências válidas.** Chamar o lado dos dados de "fechado"
foi cedo demais: a separação por loco é parte dos dados e ainda não rodou sobre o plano real. Falta, além dela,
fechar a política de AF. O que resta depois disso é o **peso da loss** (única decisão da §5.1 sem valor) e o
**treinador MLM**.

**Expectativa corrigida sobre os locos.** Eu havia dito que quase todo loco teria uma janela só, tirando 2,7 Gb ÷
50.000 ≈ 54 kb de espaçamento médio. Isso está errado: as 30.000 janelas globais saíram de **2.500 regiões de
20 kb**, ou ~12 janelas de 4.096 bp por região — elas se sobrepõem muito entre si. O lado do ABraOM, sim, vem de
um pool espalhado pelo genoma. Então as duas metades entram na separação com estruturas de loco **diferentes**:
poucos locos grandes do lado global, muitos pequenos do lado brasileiro. Isso não invalida a separação, mas muda
o que a validação global mede — generalização entre regiões, não entre janelas —, e tem de ser lido no relatório
(`locos.maior`, `locos.com_uma_janela`), não presumido.

**Separação por loco rodada no plano real [21/09].** `disjuncao_verificada: true`, **zero violações**,
mistura 0,6003 no treino e 0,5978 na validação, bins de AF equilibrados nos dois lados, `pendencias: []`.
Treino 44.645 janelas em 7.681 locos; validação 5.355 em 910 locos. Saídas `c76d08d4…` (treino) e `034eca34…`
(validação).

**A concentração é maior do que eu previa, e nos dois lados.** Medido: **8.591 locos para 50.000 janelas**,
mediana de **4** janelas por loco, maior com **208**, e só 1.884 singletons. Do lado global isso era esperado — as
30.000 janelas saem de 2.500 regiões de 20 kb, ~12 janelas de 4.096 bp por região. Do lado do ABraOM **não era**:
20.000 variantes de um pool de 1,2 milhão deveriam dar ~136 kb de espaçamento médio e quase um loco por janela,
o que somado ao global daria ~22.500 locos. Faltam ~14 mil. A decomposição por fonte **mediu** o que eu tinha inferido:

| fonte | janelas | locos | mediana | maior | singletons |
|---|---:|---:|---:|---:|---:|
| ABraOM | 20.000 | **4.778** | 3 | **175** | 1.366 |
| global | 30.000 | 3.889 | 6 | 33 | 529 |

**Referência declarada, por simulação:** 20.000 janelas de 4.096 bp com posições **uniformes sobre os
autossomos**, encadeadas pela mesma regra, dão **19.401 locos** (20 réplicas, 19.356–19.442). Antes eu havia
citado "4,2×" a partir de 20.000 ÷ 4.778, que é a **média de janelas por loco** — chamar aquilo de razão contra
um esperado era erro.

**Mas a comparação ainda não está fechada.** Os 4.778 são locos **conjuntos** que contêm ABraOM, e a simulação é
só de ABraOM: são universos diferentes. Para fechar, a comparação tem de usar `locos_da_fonte_sozinha`, com o
mesmo conjunto de cromossomos e a mesma regra de intervalo. O fator ~4× fica como **indicação**, não como medida
— e a leitura que se sustenta sem ele continua sendo a qualitativa: a amostra do ABraOM tem concentração
espacial relevante.

Duas ressalvas de pé: a referência uniforme é escolha declarada, não a distribuição verdadeira de variantes num
genoma (que já é agrupada), e a estratificação por AF pode alterar a concentração da amostra. Além disso,
`locos_que_a_contem` usa os locos **conjuntos** — uma janela global pode emendar dois grupos do ABraOM e inflar a
concentração aparente dele. O relatório passou a publicar `locos_da_fonte_sozinha`, que refaz o encadeamento só
com as janelas da fonte.

A conclusão que os números sustentam: **a amostra do ABraOM tem concentração espacial relevante**. Eles não
identificam o filtro que a produziu — a procedência continua sendo pergunta para o Eduardo.

**[MEDIDO, e é o achado principal de 21/09] As duas metades da mistura quase não se encontram no genoma.** Só
**76 de 8.591 locos** contêm as duas fontes. Isso **não é anomalia**: o lado global cobre ~70 Mb (2,6% do genoma),
então o acaso preveria ~123 locos compartilhados, e 76 é a mesma ordem de grandeza. É o comportamento esperado de
duas amostras esparsas e independentes.

O que isso mostra é que **casar por cromossomo iguala a distribuição grossa e não toca na escala fina** — as
duas metades continuam em lugares diferentes. O que isso **não** mostra: que o modelo consiga identificar a fonte
pela sequência, nem que um eventual ganho venha da geografia. São hipóteses a controlar numa ablação, não
achados. E o contraste MR × MG, onde isso morderia mais forte, já está adiado por decisão do Eduardo — o
principal segue sendo **M0 × MR**, com adapter único.

**Terceira opção implementada como CANDIDATA EXPERIMENTAL, não como controle:**
`--geografia casado_aos_locos_do_abraom` centra cada região numa posição do próprio pool do ABraOM. Ela aproxima
os locais amostrados, mas **não cria pareamento entre as janelas finais** das duas fontes e portanto **não
garante** que a localização deixe de distingui-las: o lado do ABraOM continua amostrado à parte, o global
continua passando por teto e reservatório, sortear **variante** (não loco) favorece região com mais variantes do
ABraOM, e regiões centradas em variantes vizinhas se sobrepõem. Adotá-la exigiria medir a cobertura genômica
única e a sobreposição entre fontes no resultado final.

**O custo também é real:** o lado global herdaria o viés de cobertura do arquivo do ABraOM, e "global" passaria a
significar "variação agregada **nos locos que o ABraOM cobre**". **A receita atual não é substituída** — a
alternativa fica identificada, e a decisão é do Eduardo.

**Defeito corrigido junto:** a deduplicação acontecia **depois** do reservatório, então um alelo encontrado em
regiões sobrepostas entrava várias vezes e tinha mais chance de ser guardado. Pouco sob a geografia atual, grave
sob a ancorada no ABraOM, onde as regiões se sobrepõem muito. Agora é por alelo, antes do reservatório.

**A correção não muda retroativamente o pool já produzido.** O `ce749a6d…` e tudo que dele deriva — plano
`c99e5dae…`, treino `c76d08d4…`, validação `034eca34…` — vêm da versão anterior do coletor e **ficam assim**,
identificados como tal, para o smoke do adapter. Uma reextração futura é outra versão e recebe hashes novos; não
se mistura as duas nem se refaz artefato por causa de um defeito cujo efeito, nesta geografia, era pequeno.

O relatório também passou a medir `sobreposicao_de_alelos_com_o_abraom`: quase toda variante do ABraOM também
está no gnomAD, então o mesmo alelo pode cair nas duas metades. Não é vazamento — a fonte diz de qual
distribuição a variante foi sorteada, e o adapter nunca vê a AF —, mas duplica janela de treino e tem de ser
decisão declarada.

**Como a validação do adapter tem de ser lida.** A validação tem **5.355 janelas agrupadas em 910 locos**, e
qualquer incerteza tem de respeitar esse agrupamento na análise. Nem "o tamanho amostral efetivo é 910" nem "910
é o teto do tamanho amostral efetivo" são afirmações corretas — a segunda também não é regra geral. **910 é o
número de agrupamentos.** O que se relata é isso: janelas, locos, e o tratamento da dependência.

**[ABERTO] A unidade da média da loss de validação** — por posição mascarada, por janela ou por loco — são
perguntas diferentes, e a receita do piloto tem de declarar qual. Entra na pauta do Eduardo com o peso da loss.

**Nucleo do MLM do adapter [IMPLEMENTADO em 21/09]: `eval/adapter/mlm.py`.** Stdlib puro, sem torch — a regra
fica provavel no Windows e a parte que toca o modelo fica fina.

**O que o adapter e, escrito no codigo para nao derivar:** adapter POPULACIONAL treinado com MLM sobre janelas em
que variantes amostradas por frequencia foram aplicadas. **Ele nao preve AF.** A frequencia entra na AMOSTRAGEM,
nunca como entrada nem como alvo. Trocar por regressao de frequencia seria outro experimento.

**As tres categorias de posicao mascarada** — e a distincao que se perde quando alguem fala em "loss do span da
variante":

| categoria | o que e | alvo |
|---|---|---|
| `focal_alt` | a UNICA posicao onde o ALT foi aplicado | o alelo alternativo |
| `contexto_da_variante` | as demais posicoes DENTRO do span que cobre o focal | bases de **referencia** |
| `referencia` | posicoes dos spans que nao tocam o focal | bases de referencia |

O span tem de 3 a 10 bp, entao ele quase sempre cobre bases de referencia alem do focal: **"loss do span da
variante" nao e "loss do alelo variante"**, e juntar as duas diluiria justamente a medida da campanha. O teste
`test_alvo_do_focal_e_o_alt_e_nao_a_referencia` trava isso.

**[FECHADO] A armadilha da varredura de pesos.** Comparar configuracoes pela propria loss ponderada de cada uma
nao compara nada — cada uma otimiza um objetivo diferente, e venceria a de pesos mais frouxos. O criterio e
COMUM e independente dos pesos: entropia cruzada media **nao ponderada**, publicada por categoria, com
`CRITERIO_PRIMARIO = focal_alt` declarado antes de rodar. Ha teste que exibe a armadilha: duas configuracoes
sobre as mesmas perdas dao losses de treino diferentes e criterio identico.

**Receita inicial declarada** (ponto de partida, nao resultado de busca): peso 1,0 no focal e 0,5 nas outras
duas, pesando **por posicao** e nao por categoria — com peso igual, uma janela com 1 focal e 12 de referencia nao
pode valer metade focal e metade referencia.

**O vocabulario do R03** (A=1, C=2, G=3, T=4, N=5, MASK=6) e redeclarado em vez de importado, porque
`lumina/__init__.py` puxa torch no topo. O teste carrega `lumina/constants.py` por caminho e trava a igualdade:
se o vocabulario do modelo mudar, o teste quebra em vez de o treinador tokenizar errado em silencio.

**Nucleo em torch do treino [IMPLEMENTADO em 21/09]: `eval/adapter/treino.py`.** O que foi conferido no
codigo antes de escrever, e que corrige duas coisas que eu havia afirmado:

- **`build_finetune_adapter` despacha por FAMILIA**, e `"lumina"` e `"lumina-r03"` sao ramos diferentes. Receber
  um checkpoint nao seleciona o R03: a familia tem de ser pedida.
- **`native_feature_heads` so e passado ao ramo `"lumina"`.** O `FineTuneR03Adapter(checkpoint_path, device)` nao
  o recebe, entao `["none"]` nao remove nem congela nada ali. Eu havia dito que congelava.
- `FineTuneR03Adapter` carrega com `strict=True` e `forward_hidden_states` **nao** tem `no_grad` (o `no_grad` esta
  so em `extract_native_pathogenicity_features`). O caminho do MLM e `encode` → `mlm_head`.
- **`mlm_head` esta em `_EXCLUDE_PATTERNS`**: o `apply_lora` nao a embrulha. Ela fica congelada e mesmo assim no
  grafo — e por ela que o erro chega ao adapter. Nada de `no_grad()` no caminho de treino.
- **`lora_b` nasce em zeros.** Logo o gradiente de `lora_a` e zero por construcao no primeiro passo: exigir
  gradiente nao nulo em TODO tensor do adapter reprovaria um treino correto. O que se exige e que o adapter
  receba sinal e **mude**, enquanto o congelado fica identico (verificado por hash dos nao treinaveis).
- `eval()` e decisao **separada** de congelar peso: congelar zera gradiente, nao desliga dropout. O modo vai
  declarado no manifesto.

**Checkpoint com contrato proprio, `lumina_population_adapter_mlm_v1`.** O antigo (`abraom_frequency_adapter_v1`)
salvava **todo** parametro com `requires_grad`, o que no modelo dele incluia a cabeca de regressao de AF e
LayerNorm — nao era "so os deltas", como eu disse. E o carregador dele usava `strict=False` e so recusava
algumas chaves **inesperadas**: chave **ausente** passava em silencio e deixava parte do adapter na
inicializacao. O novo exige o **conjunto exato** de chaves nos dois sentidos, confere formas, guarda a
identidade do checkpoint base e a receita do rsLoRA, e recusa salvar se algo fora do adapter estiver treinavel.
Ha teste para chave removida e para chave sobrando.

**A loss e reescrita em tensores** porque `mlm.perda_ponderada` converte para `float` e romperia o gradiente. O
teste que liga os dois mundos confere igualdade numerica entre as duas formulas — se divergirem, quebra.

**Smoke do adapter no R03 real [22/09] — passou em quase tudo, e o que reprovou era defeito de verdade.**
Confirmado no modelo: `checkpoint_sha256 = f2983560f8f965…`, que **bate com o contrato**; `lumina` importado do
pacote deste repo; 597 parametros congelados; logits `(1, 4096, 4)`; loss em tensores igual a do nucleo sem
torch; focal 4 / contexto 14 / referencia 23 posicoes; backbone congelado **identico por hash** apos o passo; e a
instancia **nova**, construida da base com o adapter carregado, reproduzindo as predicoes com diferenca
**0,00e+00**.

**O que reprovou: 4 tensores com `grad is None`** — `strided_attn.out_proj` das camadas 8 e 17. A causa e do
adapter, nao do smoke: **`nn.MultiheadAttention` nao chama o proprio `out_proj` como modulo**, ela repassa
`out_proj.weight` para `F.multi_head_attention_forward`. Como o `LoRALinear` expoe `.weight` como propriedade do
`base`, o embrulho fica **inerte**: o delta nunca e aplicado, os parametros ficam fora do grafo, e ainda assim
entram no otimizador, onde o `weight_decay` os move.

Eram **6 dos 105** modulos adaptados: `strided_attn` e os dois `anchor_*` das camadas 8 e 17, as de atencao
esparsa. Os `anchor_*` eram inertes **duas vezes** — sem `variant_edit_mask`, o `backbone.py` toca os parametros
deles com magnitude zero so para o DDP ver um conjunto constante de treinaveis, entao receberiam gradiente
exatamente zero mesmo se o embrulho funcionasse.

`apply_lora` passa a pular esses `Linear` e a declara-los em `modulos_inertes_ignorados`. **Numericamente nao
muda nada** — eles nunca participaram do forward —, mas tira 6 tensores mortos do otimizador e corrige o
manifesto, onde "105 modulos adaptados" superestimava a superficie real. A superficie efetiva do adapter no R03
e de **99 modulos**, e ela foi **congelada** em `configs/adapter_r03_superficie.json`.

**Smoke aprovado na segunda rodada [22/09]: 17/17 checagens, `passou: true`.** 99 modulos, 198 tensores do
adapter, `sem_gradiente` vazio. O quadro de gradientes saiu como a teoria previa: **99 `lora_b` com gradiente nao
nulo e 99 `lora_a` com gradiente zero** -- `lora_b` nasce em zeros, entao no primeiro passo `lora_a` nao recebe
sinal por construcao. Backbone congelado identico por hash; instancia nova com o adapter reproduz as predicoes
com diferenca 0,00.

**A superficie declarada tem uma consequencia, e ela e mais estreita do que eu disse.** As camadas **8 e 17**
ficam sem adapter: sao as de atencao esparsa (`strided_attn` + `anchor_*`), sobre `nn.MultiheadAttention`, cujo
`out_proj` nao e chamado como modulo. O que se pode afirmar e apenas isto:

> As projecoes das duas atencoes globais esparsas **nao recebem LoRA diretamente** nesta receita.

Eu havia escrito que "o adapter nao alcanca o caminho de longo alcance do R03", e isso **nao se sustenta**. Essas
atencoes continuam funcionando, e suas ENTRADAS vem de camadas adaptadas -- entao suas saidas mudam mesmo com os
pesos congelados. Os blocos Mamba bidirecionais tambem propagam informacao ao longo da sequencia. A limitacao e
de **atualizacao direta de determinados parametros**, nao de alcance.

E a justificativa da correcao tambem estava errada: **`grad is None` nao e gradiente zero para o AdamW**.
Parametro sem gradiente e **pulado** no passo, entao o `weight_decay` nao o movia. O motivo de remover os
wrappers e outro e mais simples: aqueles deltas estavam fora do calculo util. Eram **6 modulos, ou 12 tensores
LoRA** (210 → 198), nao 6 tensores.

**Nao ler nada do `por_fonte` desta rodada.** Ele saiu com 2 posicoes focais por fonte -- ABraOM 1,296 contra
global 1,067 -- e com esse n a diferenca nao distingue nada. O campo existe para o piloto, nao para o smoke.

O diagnostico tambem ficou mais fino: `sem_gradiente` (nem entrou no grafo) passou a ser separado de
`com_gradiente_zero` (entrou e nao recebeu sinal, como `lora_a` no primeiro passo). Estavam juntos e sao
perguntas diferentes.

**Laco de treino do adapter [IMPLEMENTADO em 22/09]: `train_population_adapter.py --treinar`.** O ponto sutil e a
**acumulacao de gradiente**: microlotes tem quantidades DIFERENTES de posicoes mascaradas, porque o numero de
spans e o comprimento deles variam por janela. Dividir cada microlote pelo numero de microlotes daria peso igual
a um com 3 posicoes e a outro com 30. A implementacao soma `w_i * CE_i` em cada microlote e divide os gradientes
pelo **peso total** antes do passo; ha teste que compara o gradiente acumulado com o de um lote unico, tensor a
tensor.

O resto do contrato: validacao em `eval()` e sem gradiente, **agregada por soma e contagem** (nunca media de
medias) e tambem por fonte; scheduler contado por **atualizacoes do otimizador**; interrupcao em loss ou
gradiente nao finito, com o motivo no relatorio; retomada explicita com o aviso de que o estado do AdamW **nao**
e restaurado; e reconferencia, no fim, de que o backbone congelado continua identico por hash.

**O hash do checkpoint passou a ser sempre calculado.** Reexecutar o smoke sem a flag sobrescrevia o relatorio
anterior por um que dizia "nao calculado", e a ligacao entre resultado, checkpoint e versao do codigo se perdia.
sha256 de 600 MB custa segundos -- eu havia suposto minutos.

**Piloto curto rodado [22/09] — a maquina anda, e o piloto achou um defeito meu.** 20 passos, 20 atualizacoes,
`backbone_congelado_intacto: true`, `motivo_de_parada: passos concluidos`. Mas o `por_fonte` da validacao veio
**so com `abraom`**: `--limite-treino`/`--limite-validacao` usavam `plano.head(n)`, e o gerador concatena o
ABraOM antes do global — entao o inicio do arquivo e de uma fonte so. **O piloto treinou e validou 100% em
ABraOM**, sem a mistura 60/40, que e o centro do desenho. E o mesmo defeito que a revisao pegara no smoke, e que
eu corrigi la sem levar a correcao para o laco.

Corrigido: a subamostra passa a respeitar a PROPORCAO de cada fonte no plano, o relatorio publica
`mistura_do_treino` e `mistura_da_validacao`, e carregar de um plano misto uma subamostra de fonte unica
**interrompe**. O piloto tem de ser refeito.

**Uma leitura que ja vale, e que enquadra o que o adapter tem de fazer.** A referencia de uma previsao uniforme
sobre 4 bases e `ln 4 = 1,3863`. Medido no inicio:

| categoria | entropia cruzada | contra `ln 4` |
|---|---:|---|
| `focal_alt` | 1,696 | **acima** — o modelo da ao ALT menos de 25% |
| `contexto_da_variante` | 1,052 | abaixo |
| `referencia` | 1,123 | abaixo |

**O que `ln 4` significa, com precisao:** acima dela, a **media geometrica da probabilidade atribuida ao alvo**
ficou abaixo de 25%; abaixo dela, acima de 25%. **Nao e acuracia**, e ficar acima **nao demonstra** que o modelo
"nunca viu variacao populacional" -- eu havia escrito isso e nao se sustenta. O padrao e compativel com um modelo
pre-treinado em genoma de referencia ao qual se pede a base nao referencia, e serve como **ancora de escala**
para o delta, nao como diagnostico do pre-treino. Entrou no relatorio para nao depender de quem lembre de
calcular `ln 4`.

**Nao ler a queda de `focal_val`** (1,7047 → 1,6962 em 20 passos): sao 0,008 sobre 64 posicoes focais, com a loss
de treino oscilando entre 1,32 e 2,78. Nao distingue aprendizado de ruido, e o piloto e tecnico.

**Piloto 2 [22/09], com a mistura certa: 60 passos, 40/60 exato na validacao.** 64 posicoes focais do ABraOM e
96 do global, `backbone_congelado_intacto: true`, parada limpa. A dinamica do treino esta provada.

`focal_val` caiu nos seis pontos de validacao: 1,7353 → 1,7313 → 1,7276 → 1,7238 → 1,7233 → 1,7230. **Nao sao
seis confirmacoes independentes**: e o mesmo conjunto avaliado por modelos sucessivos de uma unica trajetoria, e
a desaceleracao acompanha o decaimento do cosseno. A magnitude e 0,012 sobre 1,72 (0,7%), sem intervalo de
confianca, e os 160 exemplos nao garantem precisao: qualquer incerteza teria de **respeitar os locos**, porque as
janelas vem agrupadas. Nao se conclui nada dai.

**Faltava o numero que torna isso legivel, e ele foi acrescentado: a validacao ANTES do primeiro passo.** Como
`lora_b` nasce em zeros, o adapter comeca como um **no-op exato** — a linha de base e, literalmente, o R03 puro
sobre a mesma amostra. Sem ela, "1,7230 no fim" nao tem contra o que ser comparado: a primeira validacao do
piloto vinha depois de 10 passos. O relatorio passa a publicar `linha_de_base` e `delta_da_validacao` por
categoria e por fonte, medidos na **mesma** amostra (sem ruido de amostragem entre os dois lados).

**O numero a vigiar, e que ainda nao se le:** `focal_alt` do ABraOM (1,781) contra o do global (1,685). Isso
**nao** demonstra dificuldade especificamente brasileira: sao exemplos diferentes, com contextos, variantes e
distribuicoes diferentes -- e as grades de AF ja sao reconhecidamente incomparaveis entre as fontes. Alem disso
esses valores sao do modelo **depois** do piloto, nao da linha de base, que so agora passou a ser medida. Fica
como quantidade a acompanhar, com o contraste correto sendo entre a linha de base e o fim **na mesma fonte**.

E mesmo que o ABraOM melhore mais no delta, isso sera **diferenca descritiva de reconstrucao nestas amostras**,
nao atribuicao causal ao componente brasileiro. A atribuicao exigiria a ablacao MG, que o Eduardo adiou.

**Bug meu, corrigido na revisao [22/09]: o commit `aaa9d40` quebrava o smoke.** O bloco que calcula o delta foi
inserido com `str.replace` sobre uma ancora (`relatorio = {\n "proveniencia": ...`) que existia em **duas**
funcoes, e o Python substitui todas as ocorrencias: `rodar_smoke` passou a referenciar `historico` e
`linha_de_base`, que la nao existem. Estouraria `NameError` **depois** de toda a execucao em GPU -- o pior lugar
possivel. `tests/test_population_runner.py` fecha a classe do erro por `symtable`: reprova qualquer global
referenciado em `rodar_smoke` que nao exista no modulo, sem precisar de GPU.

Junto vieram quatro correcoes: `--seed` passou a semear tambem o **torch**, que e quem sorteia a inicializacao do
`lora_a` (sem isso o adapter comecava diferente a cada execucao, e o delta nao era reprodutivel); na **retomada**
a linha de base nao e o R03 puro, entao ela passou a ser rotulada (`r03_sem_delta` × `adapter_retomado`); o delta
sumia quando o ultimo passo nao caia na cadencia de validacao, e agora ha avaliacao final; e entraram o sha256 do
plano de validacao e a recusa de loss de validacao nao finita.

**Piloto 3 [22/09]: o delta contra a linha de base, e a ambiguidade que ele expoe.** 60 atualizacoes,
backbone intacto, `linha_de_base.sistema = r03_sem_delta`, as duas fontes presentes. Delta da validacao (negativo
= melhorou), medido na MESMA amostra antes e depois:

| categoria | delta | ABraOM | global |
|---|---:|---:|---:|
| `focal_alt` | **−0,0213** | **−0,0282** | −0,0168 |
| `contexto_da_variante` | +0,0007 | +0,0014 | +0,0002 |
| `referencia` | +0,0031 | +0,0018 | +0,0040 |

Linha de base por fonte: ABraOM 1,8025 → 1,7743; global 1,6981 → 1,6813.

**O padrao e limpo e ambiguo ao mesmo tempo.** O focal melhorou e as posicoes de REFERENCIA pioraram. Na moeda da
loss de treino (delta × posicoes × peso) o ganho focal e +3,41 e o custo nas outras e −1,96: liquido +1,46, entao
o treino de fato desceu. Mas o modelo **nao ve diferenca, na entrada, entre a posicao focal e uma de contexto** —
todas sao `MASK`. Entao "tirar massa da base de REFERENCIA em toda posicao mascarada" produz exatamente esse
padrao: derruba a perda focal (onde o alvo nunca e a referencia) e sobe a das posicoes de referencia. **Sem
aprender nada sobre qual alelo a populacao carrega.**

**E por isso que "o ABraOM melhorou mais" nao se le ainda.** Alem de ABraOM e global serem amostras diferentes, o
ABraOM **comecou pior** (1,8025 contra 1,6981): ha mais folga para melhorar. Em termos relativos sao −1,56% e
−0,99%, uma diferenca bem menor que a absoluta sugere.

**[IMPLEMENTADO] O diagnostico que desempata: `P(ALT) / (1 − P(REF))` no focal.** Tirar massa da referencia sem
saber nada sobre o alelo redistribui entre as **tres** bases restantes e deixa essa fracao parada em ~1/3.
Aprender qual alelo a populacao carrega a faz **subir**. O `Exemplo` passou a carregar a base de referencia do
focal, e a validacao publica `diagnostico_do_focal` com `p_ref`, `p_alt` e a fracao, por fonte — com o delta
entre a linha de base e o fim. Ha teste que exibe os tres regimes: massa na referencia, massa tirada sem saber o
alelo (fracao em 1/3), e alelo aprendido (fracao > 0,9).

**Sem esse numero, o piloto nao distingue adaptacao populacional de um atalho.** Com ele, a pergunta vira
verificavel sem esperar a avaliacao clinica.

**[ABERTO] Confundimento espacial entre as duas fontes.** O pool do ABraOM é concentrado onde o ABraOM tem dado —
o chr16 aparece mais que o chr1, que é cinco vezes maior. Se o lado global for amostrado uniformemente pelo genoma,
as duas metades da mistura passam a diferir **também pela localização**, e o adapter pode separar "global" de
"brasileiro" pela região em vez de pela estatística populacional.

A ordem correta das decisões é esta, e ela importa:

1. **Primeiro, o que "global" significa.** Frequência agregada do gnomAD joint, ou amostragem por grupos
   ancestrais? São desenhos diferentes, com campos diferentes no VCF, e a resposta muda o que se amostra.

   **[PROPOSTO] Usar `INFO/AF_joint`** — a agregada — também na amostragem do treino. O fundamento é coerência: é
   o campo que alimenta `gnomad_v4_af` e, por ele, o `gnomad_af_bin` que pareia caso e controle nos dois estudos
   brasileiros (`src/mosaic/annotations/derived.py`, `specs/GNOMAD_S3_READ.md`). Com outro campo, a campanha teria
   duas noções de "AF global": uma no pareamento da avaliação, outra no treino do adapter.

   **Mas isto é proposta nossa, não decisão herdada.** O Mosaic fixou a unidade de comparação da AVALIAÇÃO; ele
   não determina por qual estatística o adapter deve amostrar. São perguntas distintas — qual AF compara casos e
   controles, qual AF decide o que o adapter vê, e quanto cada ancestralidade contribui. A terceira continua
   totalmente em aberto. Do mesmo modo, o `PLAN.md` §13 proíbe usar `AF_amr` como **proxy brasileiro** no release;
   isso não implica que o treino deva usar só a agregada.
2. **Depois, a receita de amostragem.** Balancear cromossomos **pode reduzir diferenças espaciais grosseiras entre
   as fontes; não demonstra equivalência dos contextos** — dentro do mesmo cromossomo ainda diferem genes, regiões
   codificantes, cobertura e os filtros de descoberta de cada projeto. É mitigação declarada, não controle.

Por isso o balanceamento por cromossomo fica como **proposta**, não requisito, até o gnomAD ser inspecionado.

**Medido em 20/09: a concentração é do arquivo, não da amostragem.** No pool, a densidade por megabase varia
**5,9×** entre o chr16 (2,75× a média) e o chr14 (0,46×); o chr16 tem 2,9× a densidade do chr1, que é quase três
vezes maior. A estratificação por AF é geograficamente neutra — a maior diferença entre a fração de um cromossomo
no pool e no plano amostrado é de 0,0029.

E há um sinal mais forte sobre a procedência: o pool tem **448 variantes por megabase**, ou uma a cada ~2,2 kb.
Um callset WGS de 1.171 indivíduos produz densidade uma a duas ordens de grandeza maior. Isso é **incompatível com
um callset completo** e indica filtragem ou subconjunto — mas **não identifica qual filtro**: densidade e nome de
arquivo não são procedência. O que a métrica autoriza a dizer é que a pergunta precisa ser feita, não que a
resposta já se conhece. `SABE1171.Abraom.clean.tsv` é `academic_request`: **perguntar ao Eduardo a procedência e o
critério** antes de fechar a receita do lado global, porque o filtro decide o que "a metade brasileira" representa.

**Pool de amostragem do ABraOM [FIXADO em 20/09; contagem final pendente].** O arquivo traz o cromossomo como `1`
enquanto o snapshot e o FASTA usam `chr1`: juntar sem normalizar daria **zero sobreposição em silêncio**, que é o
pior erro possível aqui porque se parece com "nenhum vazamento". `scripts/audit_abraom_source.py` normaliza os dois
lados e descarta, em ordem declarada: AF inválida (NaN, negativa, > 1), AF nas extremidades, não-SNV, fora
de chr1–chr22, **membro dos estudos**, **alelo do conjunto de seleção ou da validação e teste do core**, e chr8 se
reservado.

Medido em 20/09: **1.812 com `AF = 0` e 514 com `AF = 1`** (total 2.326, que é o motivo agregado).

**As duas extremidades de AF não são a mesma coisa, e nenhuma delas é "AF inválida".** `AF = 0` significa que o ALT
não foi observado na amostra — não há variação populacional ali para o adapter aprender. `AF = 1` é uma
**frequência perfeitamente válida**: o ALT está fixado na amostra. Excluir as duas é **escolha de desenho** desta
campanha (o adapter deve ver variação, não alelos monomórficos), reversível com `--manter-af-degenerada`. O
relatório conta as duas separadamente e registra a justificativa de cada uma.

A distinção que importa: **ver o contexto genômico** de uma variante que será pontuada é diferente de **treinar o
adapter a reconstruir o alelo dela**. Por isso os alelos que serão pontuados saem do pool; a sobreposição com o
treino da cabeça é medida e declarada, não eliminada — ela é esperada, porque as duas coisas vêm do mesmo genoma.

O script só publica o pool com as exclusões em mãos, registra o sha256 de cada arquivo usado para excluir, e para
com código 2 se o ABraOM não bater o source-lock ou se houver o mesmo alelo com AF conflitante.

**Medido em 20/09** (de 1.365.230 linhas): pool com **1.224.029** variantes, `sha256 40bd0f79…`. Saíram 74.532 fora
dos autossomos, **61.737 no chr8** (o custo da decisão E do lado do adapter), 2.326 nas extremidades de AF, 2.282
membros dos estudos e 324 alelos do conjunto de seleção ou da validação e teste. Zero não-SNV e zero duplicatas.
A sobreposição declarada com o treino da cabeça é de 16.649 variantes no candidato `nenhum` e 6.733 no `janela4096`.

As contagens por motivo são **sequenciais**: tirar as extremidades de AF primeiro reduz quantas linhas sobram para
ser classificadas como chr8 ou membro de estudo. Comparar dois relatórios motivo a motivo exige lembrar disso.

**Os 2.282 membros removidos não demonstram, sozinhos, qual estudo estava presente.** Subtrair totais não é
composição: o `br_population_observed` é definido como "gold presente no ABraOM", então a afirmação que interessa é
a **interseção por estudo e papel**, publicada em `membros_encontrados_por_estudo`. Medida em 20/09:

| estudo | papel | membros | no ABraOM |
|---|---|---:|---:|
| `br_population_observed` | case + unmatched_case | 1.889 | **1.889 (100%)** |
| `br_population_observed` | control | 751 | **0 (0%)** |
| `br_clinical_evidence` | case (+1 unmatched) | 3.119 | 324 (10,4%) |
| `br_clinical_evidence` | control | 3.116 | 71 (2,3%) |

O estudo populacional sai **exato nos dois extremos**: todo caso presente, nenhum controle. Isso é a definição dele,
e serve de conferência forte da cadeia inteira — normalização de cromossomo, construção da chave, importação do
membership e leitura do ABraOM. Qualquer defeito em qualquer um desses passos apareceria como ruído aqui.

A soma das interseções é 2.284, e o motivo `membro_de_estudo` conta 2.282: **2 membros foram pegos por uma regra
anterior** (AF nas extremidades, não-SNV ou fora dos autossomos), porque as regras são sequenciais. Os dois números
respondem perguntas diferentes e ambos estão certos.

**[ABERTO — achado de 20/09] Os casos do estudo clínico estão 4,6× mais presentes no ABraOM que seus controles**
(10,4% × 2,3%). O pareamento do Mosaic é por `rótulo | painel | bin de AF do gnomAD`, e **não** inclui presença no
ABraOM: a assimetria sobrevive ao pareamento. Isso importa porque a interação mede (ganho nos casos) − (ganho nos
controles), e o MR é justamente o braço adaptado ao ABraOM. Parte de um ganho positivo pode ser atribuível a
**estar no ABraOM**, não a **ter participação brasileira** — construtos correlacionados, não iguais. Os alelos dos
membros saem do pool do adapter, então não é memorização do alelo; é o contexto deles estar super-representado no
treino.

O protocolo do Mosaic já pede relatar o subconjunto `present_abraom` do estudo clínico — agora sabe-se o tamanho:
**324 casos e 71 controles**.

**Proposta de sensibilidade [PROPOSTO], corrigida:** restringir aos **pares em que caso E controle estão ambos
ausentes** do ABraOM, **preservando o pareamento materializado pelo Mosaic**. Comparar "todos os casos ausentes"
contra "todos os controles ausentes" seria errado: descasa os pares que o release construiu e ainda mistura os
`unmatched_case`. O tamanho desse subconjunto tem de ser medido pelo `matched_variant_id`, não estimado por
subtração — 3.116 − 324 − 71 não é o número de pares completos. Relatar tamanho, classes e painéis do que sobra.

E o resultado não conclui sozinho, nos dois sentidos. Se o efeito **sumir** ali, isso não prova que era "sobre o
banco": o subconjunto tem menos poder e composição diferente. Se **persistir**, não elimina todos os efeitos de
contexto — só o mais direto. É análise de sensibilidade declarada, não teste decisivo.

O espectro de AF observado é **compatível com passos de 1/2342** — o mínimo do pool, 0,000427, corresponde a um
alelo em 1.171 genomas diploides. Isso **não demonstra denominador constante**: o arquivo não traz AC/AN, os
valores estão arredondados, e o AN pode variar por loco conforme as chamadas disponíveis, como acontece no próprio
gnomAD. O fato robusto é outro e basta para a receita: **54% das variantes caem no bin mais raro** (≤ 0,001, que
comporta singletons e doubletons, não só singletons). Amostrar uniformemente encheria o adapter de variantes
raríssimas; é por isso que a amostragem é estratificada por bin.
- Janelas: **nenhum alelo de variante dos dois estudos** e nada do chr8 enquanto ele estiver reservado. Contexto de
  referência pode aparecer; alelo do estudo, nunca. Isso vale com mais força agora: `br_population_observed` é, por
  definição, gold presente no ABraOM, que passa a ser fonte de treino do adapter.
- Membership serve só para exclusão e auditoria, nunca para escolher alelos ou alvos de treino.
- Registrar a sobreposição que não puder ser eliminada, inclusive a do pré-treino do R03 (fontes e chr8).

---

## 5. Treino

### 5.1 Adapter populacional misto [FIXADO no desenho — Eduardo, 15/09; parâmetros a declarar]

Desenho fixado: **MLM com máscaras em span sobre as mutações**, com as mutações em **posições aleatórias da janela**
(não centralizadas). Um único adapter, treinado na mistura ≈60% global / 40% ABraOM.

Parâmetros a declarar antes de treinar, todos no manifesto do gerador. São **propostas nossas**, não especificação
do Eduardo:

| Parâmetro | Proposta para o piloto |
|---|---|
| Comprimento da janela | 4.096 bp, o mesmo comprimento de contexto da extração, para o adapter treinar no regime em que será usado. Não é compartilhamento de cache: janela sintética mascarada e embedding de avaliação são artefatos diferentes |
| Variantes por janela | **uma variante focal por janela**, o que torna equivalentes as três leituras da mistura e permite auditá-la |
| Posição da variante | **deslocar o início da janela** em torno da variante, preservando a coordenada genômica e o contexto real. A mutação **não** é transportada para outra posição da sequência |
| Mistura | **60% das janelas** de fonte global / 40% ABraOM. É parâmetro: 100/0 dá o braço global puro (MG). Declarar sempre a unidade (janelas, aqui) |
| Amostragem por frequência | estratificada por bin de AF, para o adapter ver o espectro populacional e não só variantes raras |
| Alvo | a sequência sintética recebe o **alelo alternativo** na posição da variante; o modelo reconstrói as bases mascaradas do span, isto é, o alelo aplicado mais o contexto de referência coberto |
| Span de máscara | spans curtos cobrindo a posição variante, **mais** uma fração de spans em posições só de referência. Motivo: sem eles, mascarar passa a coincidir com "aqui há variante" — é um risco a investigar no piloto, não um efeito demonstrado |
| Loss | definir o **peso** de posições variantes e de referência (não basta relatar separado) e reportar as duas curvas (contrato v2 §2) |
| Validação do adapter | populacional, separada por **loci/janelas** do treino, para não medir memorização. Nunca o estudo brasileiro |
| Congelamento | só o LoRA do adapter recebe gradiente; cabeças nativas congeladas (`freeze_native_feature_heads`) |
| Orçamento e seeds | idênticos em qualquer braço comparado; registrar seed por execução |

Ordem: smoke sintético, depois piloto pequeno com uma seed, depois a execução da campanha.

### 5.2 Extração [PROPOSTO]

Candidatas: as 172 dimensões de cabeça da pesquisa e a leitura antiga completa (`site_ref`, `variant_repr` e o
contexto local de ±64 bp; a aproximação de 896 dimensões avaliada na pesquisa não tinha esse contexto). A suíte do
Mosaic é só SNV, então a extração só-SNV é compatível por construção. Escolher **uma** no **conjunto de seleção
comum** (§4.2) — não no fold 1, cuja fragilidade motivou o conjunto —, com critério declarado antes, e usar a mesma
em M0 e MR. Nunca escolher no estudo brasileiro.

O extrator foi portado em 17/09 de `embedding-probe-mosaic` (`23fb518`) para `eval/embedding_probe/rich.py`, sem
alteração de comportamento, com o código executável verificado igual ao da origem por comparação de texto.

O cache é identificado por R03, **adapter**, versão do extrator e chaves das variantes — e, pela revisão de 17/09,
também por **FASTA, comprimento da janela, orientação/RC, configuração e ordem das features**: duas extrações com
a mesma variante e FASTA diferente não são o mesmo objeto.

**Política para `N` e soft-mask [FIXADO em 17/09; contagem pendente].** O `windows.py` portado (mesmo commit)
resolve as duas coisas de formas diferentes: **soft-mask é normalizado** (a janela vai a maiúsculo antes de
qualquer checagem, então minúscula não custa variante) e **janela com base fora de ACGT é descartada**, com o
motivo estável `non_acgt` — nada de substituir ou mascarar o `N`. Janela que não cabe no cromossomo também é
descartada (`out_of_bounds`), nunca deslocada: deslocar tiraria a variante do índice focal declarado.

O módulo segue a convenção do Mosaic: **offset focal `L // 2 - 1`**, que é a mesma com que o release calculou
`sequence_eligible` — usar `L // 2`, como o helper antigo do ClinVar, invalidaria essa garantia. E a validação é
estrita, sem o fallback de ±1 base que o helper antigo usa para indels: o release só tem SNV com REF já conferido
contra o GRCh38.p14, então divergência ali é FASTA ou build errado.

`scripts/audit_variant_windows.py` mede quantas variantes cada motivo tira, por papel, painel e classe, e **sai com
código 2 em qualquer `ref_mismatch`** — isso é erro, não estatística.

**Medido em 17/09, janela de 4.096 bp:** treino de `nenhum` 170.171, treino de `janela4096` 89.385 e os 8.875
membros dos dois estudos — **todos ok**. Zero `ref_mismatch` (o FASTA é o build certo e a coluna REF é consistente
em toda a campanha), zero `non_acgt` e zero `out_of_bounds`. A política custa zero variantes aqui, e o motivo é
estrutural: o `sequence_eligible` do release já validou a janela centrada de 32.768 bp como ACGT, e a de 4.096 é
substring dela — a garantia é herdada. A checagem continua no caminho porque é ela que pegaria FASTA trocado.

### 5.3 Cabeças e calibração [PROPOSTO]

**O que conta como "cabeça" [FIXADO em 17/09, a declarar no manifesto]:** além da `head`, o
`ClinVarVariantEncoder` (`variant_encoder`) tem pesos próprios — `Linear` e `Embedding` — que **nascem aleatórios e
não vêm do checkpoint do R03**. Congelá-lo seria usar uma projeção aleatória, então ele é parte do classificador,
treinado igual em M0 e MR. Está declarado em `CLASSIFIER_PREFIXES` e é o que `assert_only_head_trains` permite;
tudo fora disso não pode receber gradiente.

Mesma arquitetura e procedimento em H0 e HR; padronização ajustada só no treino; early stopping na validação do
`core_locus` (fold 1); Platt e limiar de MCC ajustados na mesma validação, congelados por sistema e iguais para
casos e controles. A escolha de extração e de política acontece no conjunto de seleção comum, não aqui. A campanha usa repetições de adapter e cabeça (PDF §10: pelo menos três), com predição final pela média
das probabilidades calibradas e métricas também por seed.

---

## 6. Avaliação, pré-registrada antes de pontuar o estudo

### 6.1 Saída no formato do Mosaic [FIXADO pelo protocolo]

Para o par M0 (`base`) × MR (`regionalized`), em cada estudo separadamente:

- AUROC e AUPRC no coorte completo, nos casos pareados e nos controles, com `n_P`, `n_B`, prevalência e cobertura;
- deltas na interseção de cobertura e a interação `delta_br_matched − delta_control`;
- resultados por painel no coorte completo;
- no estudo clínico, também o subconjunto `present_abraom`.

Reaproveitar do `comparator_eval` as visões (`brazil_views`) e a reamostragem por grupo, para garantir as mesmas
coortes e a mesma unidade.

### 6.2 Extensão do consumidor [PROPOSTO]

Se a ablação MG for executada, as mesmas quantidades para MR × MG, com os mesmos pares e a interseção de cobertura do
contraste. Reportar sempre os ganhos absolutos: uma interação positiva pode vir de piora no controle.

### 6.3 Incerteza [PROPOSTO]

O Mosaic reamostra `overlap_cluster_id` com 1.000 réplicas e seed `20260901`, mas cada coorte separadamente, e não
define a reamostragem da interação. Proposta: reamostrar os clusters **em conjunto** sobre a união de casos pareados e
controles, com os mesmos sorteios para os dois sistemas do contraste, e relatar a reamostragem por par como
sensibilidade. Fixar a regra, as réplicas e a seed antes dos resultados. O PDF previa 10.000 réplicas por matched set:
registrar a escolha.

### 6.4 Análises adicionais pré-declaradas [PROPOSTO]

- **Controles com SCV brasileira:** contar os controles do estudo clínico com alguma SCV de instituição da lista
  (regra ampla) e repetir a interação sem os pares desses controles, sem refazer o pareamento. O resultado oficial não
  muda, e a direção de um eventual efeito não é assumida.
- **Métricas com limiar** (MCC, sensibilidade, especificidade) e **Brier:** só com os limiares e calibradores
  congelados no desenvolvimento, com proveniência.
- **Baselines diagnósticas:** presença no ABraOM e AF (gnomAD, ABraOM), pontuadas nos mesmos pares, fora dos sistemas.
- **Exposição de locus:** se o snapshot final ainda tiver exposição não nula, repetir a interação restrita aos
  pares com exposição equilibrada (empate ou diferença pequena). **Não existe "interação por rótulo":** AUROC e
  AUPRC exigem as duas classes, então a interação fica sempre em conjuntos com P e B. Por rótulo relatamos
  exposição, distribuição dos scores, especificidade nas benignas e sensibilidade nas patogênicas, com os limiares
  congelados.
- **Sanidade no `core_locus`:** AUROC/AUPRC de M0 e MR no teste do core (fold 0, gold), pontuado **só depois** de
  congeladas todas as escolhas, para mostrar que a cabeça funciona e que o adapter não degradou o desempenho geral.

### 6.5 Margem e precisão [ABERTO]

A margem de 0,02 é proposta, não requisito do Mosaic nem expectativa de resultado. Antes de fixá-la, estudar precisão
e poder em cenários plausíveis: pares disponíveis depois da cobertura, estrutura de clusters, correlação entre os
scores dos sistemas e tamanho de efeito assumido. Usar dados de desenvolvimento ou simulação, nunca resultados dos
candidatos no estudo.

### 6.6 Partes do PDF fora do Mosaic [ABERTO — decisão E]

- **chr8 representacional** (PDF §11.1): continua possível como validação populacional do adapter, se o chr8 ficar
  fora das janelas e do treino da cabeça. Manter a reserva custa pouco agora e é irreversível depois.
- **BRCA1/BRCA2/TP53** (PDF §12): os controles do Mosaic não são pareados por gene, então não há interação por gene;
  no máximo, relato descritivo dos casos.

---

## 7. O que poderá ser afirmado

- Um ganho positivo sustenta transferência diferencial no recorte de **participação** brasileira do Mosaic, com um
  adapter populacional **misto**.
- Não poderá ser afirmado, sem a ablação MG: que o componente brasileiro é a causa do ganho.
- Não poderá ser afirmado: resultado em só-BR, em pacientes brasileiros ou na população brasileira nacional;
  generalização independente; o estudo populacional como evidência independente.
- A interação não elimina diferenças entre laboratórios ou mecanismos de rotulação.
- Se a campanha aceitar loci compartilhados entre o treino da cabeça e os estudos (§4.2), isso é declarado como
  **campanha de desenvolvimento com exposição de locus medida**, não como evidência de que a exposição é
  inofensiva. O relato traz as duas unidades (cluster e janela) e a comparação caso × controle.
- Tier consensus não prova ausência de exposição. Conferir casos, controles e clusters contra o histórico: a pesquisa
  de extração usou gold, e o treino da cabeça agora usa os folds de treino do `core_locus`.

---

## 8. Ordem de execução e gates

| Gate | Entrega | Critério para avançar |
|---|---|---|
| G0 | Identidades da seção 1 | hashes reais de R03, release, ABraOM e gnomAD registrados; arquivo do ABraOM em mãos |
| G1 | Importação e validação do membership, com coordenadas de `pb_examples` | checagens da 4.1 sem erro; contagens por coorte e painel |
| G2 | Snapshot de treino: `core_locus` `run_id=0` menos as exclusões da 4.2 | sobreposição zero com os estudos, seus clusters, a regra ampla brasileira e o chr8; custo de cada exclusão medido; hash do snapshot |
| G3 | Extrator portado e smoke de M0 sem LoRA clínico | só a cabeça recebe gradiente (hoje `apply_lora` é incondicional e `rank=0` divide por zero: precisa de guard); cache com identidade |
| G4 | Gerador de janelas + MLM: smoke sintético e piloto do adapter misto | só o LoRA recebe gradiente; aprendizado nas posições variantes; manifesto de janelas; nenhum alelo dos estudos |
| G5 | Escolha da extração e da política de isolamento no **conjunto de seleção comum** | critério declarado antes; média de 3 seeds; mesma extração nos dois sistemas |
| G6 | Sistemas congelados, manifesto do consumidor, margens e regra de bootstrap declaradas | os sete campos de `required_consumer_manifest` preenchidos; nada ajustado depois de ver o estudo |
| G7 | Avaliação única nos dois estudos | saídas da seção 6 |

G0, G1 e G2 não dependem de mais nenhuma decisão. Duas regras separadas, para a ordem não ficar circular:

- **antes de treinar:** receita, dados e critério de seleção aprovados e registrados (G0 a G2, mais o smoke do G4);
- **antes de pontuar os estudos brasileiros:** sistemas, calibradores, limiares, margens e regra de bootstrap
  congelados (G6). Depois disso, nada é ajustado.

**Não são pré-requisitos nesta rota:** contar só-BR de 1 estrela, explicar cada inconsistência histórica da v1, mudar
a política do Mosaic e refazer os pares. O `variant_summary` não é mais necessário: o snapshot da cabeça vem do
`core_locus` do release.

---

## 9. O que ainda falta do Eduardo

1. **Arquivo do ABraOM** (`SABE1171.Abraom.clean.tsv`, sha256 `3cd33784…`): é `academic_request`, não tem URL, e o
   índice da v1 não é o mesmo objeto. Bloqueia o G0 e o G4.
2. **Decisão E:** manter o chr8 reservado e fora das janelas (custa pouco agora, é irreversível depois) e se o relato
   BRCA1/BRCA2/TP53 entra como descritivo.
3. **Confirmar os parâmetros da 5.1** que ele não especificou: janela de 4.096 bp, variantes por janela, fração de
   spans em posições de referência, amostragem estratificada por AF e como registrar a mistura 60/40 no manifesto.
4. **Ciência do custo de pular etapas:** sem o braço global puro, o ganho não é atribuível ao componente brasileiro.
   A ablação é a mesma receita com mistura 100/0, uma execução, quando o resultado principal justificar.

### Confirmação operacional curta para encaminhar

"Vamos seguir com M0 versus adapter misto e treinamento da cabeça no `core_locus`. Retiraremos os estudos
brasileiros e seus clusters também da validação, e o fold de teste não vai selecionar nada — só avalia depois de
congelado. No piloto, proponho uma variante focal por janela, 60% das janelas globais e 40% ABraOM, variando a
posição da variante pelo deslocamento da janela. Manteremos o chr8 reservado por enquanto. O resultado inicial
avalia a mistura; a contribuição específica do ABraOM fica para a ablação. Falta o arquivo do ABraOM
(`SABE1171.Abraom.clean.tsv`) para rodar o piloto."

Pontos do Mosaic para registrar, sem bloquear a campanha: a URL do `submission_summary_2026-06` no `sources.yaml`
aponta para uma pasta que não existe; `origin_has_germline` só aceita a origem literal `germline` (`de novo`,
`maternal`, `inherited` e `unknown` ficam fora); nenhuma variante gold recebe marcação brasileira, provavelmente porque,
com painel de especialistas, as demais SCVs não contribuem para o agregado.

---

## 10. Decisões do Eduardo em 15/09 (registro)

| # | Pergunta | Resposta | Onde entrou |
|---|---|---|---|
| A | Usar `br_clinical_evidence` e `br_population_observed`, passando de só-BR para participação | "Pode seguir com esses dois splits mesmo" | 0, 4.1 |
| B | Fonte de treino da cabeça clínica | "Treina core_locus e avalia neles" | 4.2 |
| C | Escada M0 → global → ABraOM | "Vamos tentar ir direto pra global + abraom direto. Algo 60% global 40% Brasil? Vamos tentar pular etapas. Se der certo a gente volta e tenta explicar" | 2, 5.1 |
| D | Como fazer o MLM | "Fazer as máscaras (span) em cima das mutações"; "talvez seja até melhor ter as mutações em posições aleatórias da janela" | 5.1 |
| E | chr8 e BRCA/TP53 | não respondido | 6.6, 9 |
