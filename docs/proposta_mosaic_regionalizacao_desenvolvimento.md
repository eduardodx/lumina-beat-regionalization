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
| ABraOM | snapshot `abraom_sabe1171` do source-lock do Mosaic: `abraom/SABE1171.Abraom.clean.tsv`, `obtained: academic_request` (sem URL), sha256 `3cd33784…`, colunas `[chrom, pos, ref, alt, af_abraom]`. Obrigatório para o sistema regionalizado | **[FIXADO]**; **pedir o arquivo ao Eduardo** — o índice ABraOM da v1 não é o mesmo objeto |
| gnomAD (parte global da mistura) | v4.1 joint com AF por grupo (`s3://ai4bio-lumina/data/external/gnomad-joint-v4.1/`) | **[PROPOSTO]**; fixar versão e grupos |
| Genoma | GRCh38, `hg38.fa` (sha256 `056974f6…`) | **[FIXADO]** |
| Snapshot de treino da cabeça | derivado do `core_locus` do release v1; ID, hash e cutoff a registrar (seção 4.2) | **[FIXADO]** como fonte; identidade a gerar |

Código e dados têm identidades separadas: ter o commit certo não prova que o release foi gerado por ele.

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
| Dois estudos, nunca unidos: `br_clinical_evidence` (consensus com participação brasileira) e `br_population_observed` (gold presente no ABraOM) | o clínico é a avaliação principal; o populacional é descritivo, sobreposto ao ABraOM por construção — e agora também sobreposto ao **treino do adapter**, o que reforça a regra de alelos da 4.3 |
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
| fold 1, gold | escolher extração e hiperparâmetros, early stopping, ajustar Platt e limiar |
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

- ABraOM: o snapshot do source-lock. gnomAD v4.1 por grupo para a parte global da mistura; o `af_gnomad` antigo,
  condicional ao índice ABraOM, não serve.
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
Mosaic é só SNV, então a extração só-SNV é compatível por construção. Escolher **uma** na validação do `core_locus`
(fold 1), com critério declarado antes, e usar a mesma em M0 e MR. Nunca escolher no estudo brasileiro.

O extrator está na branch `embedding-probe-mosaic` (`eval/embedding_probe/rich.py`) e precisa ser portado com
proveniência. O cache é identificado por R03, adapter, versão do extrator e chaves das variantes.

### 5.3 Cabeças e calibração [PROPOSTO]

Mesma arquitetura e procedimento em H0 e HR; padronização ajustada só no treino; early stopping na validação do
`core_locus`; Platt e limiar de MCC ajustados na mesma validação, congelados por sistema e iguais para casos e
controles. A campanha usa repetições de adapter e cabeça (PDF §10: pelo menos três), com predição final pela média
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
| G5 | Escolha da extração na validação do `core_locus` | critério declarado antes; mesma extração nos dois sistemas |
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
