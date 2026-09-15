# Plano: regionalização do R03 com o estudo brasileiro do Mosaic

Data: 2026-09-15 · Branch: `new_regionalization` · Status: **proposta para decisão do Eduardo; nada congelado.**

Relação com `contrato_v2_regionalizacao_r03.md`: se aprovado, este plano substitui no contrato a fonte de treino
(§4, §7), a decisão do conjunto de teste (§5), os detalhes de avaliação (§10) e a pendência 9 (§15). O restante do
contrato continua valendo e é citado aqui (objetivo do adapter, gerador de janelas, extração, gates).

Legenda: **[FIXADO]** decidido pelo Gabriel ou exigido pelo protocolo do Mosaic · **[PROPOSTO]** recomendação que
aguarda o Eduardo · **[ABERTO]** decisão necessária antes de implementar.

---

## 0. Resumo

- **[FIXADO]** Backbone: R03 publicado (`best_checkpoint.pt`, passo 71.000). Benchmark: `croma-bioai/lumina-mosaic`.
- **[FIXADO]** Três sistemas completos sobre o mesmo R03: M0 (sem adapter), M1 (adapter global) e M2 (adapter
  ABraOM), cada um com sua cabeça clínica treinada pelo mesmo procedimento. Sem adapter ClinVar e sem fusion.
- **[PROPOSTO — decisão A]** Avaliar no estudo brasileiro do Mosaic, que mede **participação** de instituições
  brasileiras. O teste só-BR do PDF fica adiado: com rótulos gold/consensus, só 63 variantes são só brasileiras
  (1 benigna).
- **[ABERTO — decisão B]** O estudo só avalia sistemas congelados: a cabeça clínica precisa de um snapshot de
  treino declarado, fora do release. É a principal pendência prática.
- **[PROPOSTO — decisão C]** Contraste regional principal M2 × M1 (o controle do PDF), com M2 × M0 no formato
  base × regionalizado do Mosaic.

---

## 1. Identidades

| Item | Identidade | Status |
|---|---|---|
| Backbone | `LUM-20260719-001-R03`, `best_checkpoint.pt`, passo 71.000, pesos sem EMA, em `s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt` (README e `config/lumina_r03_base.json` do `lumina-inference`). Não trocar pelo `final_checkpoint.pt` (passo 75.000). | **[FIXADO]**. sha256 do arquivo já registrado no contrato v1 (`f2983560…`); reconferir o arquivo carregado em cada run |
| Mosaic, código | `https://github.com/croma-bioai/lumina-mosaic`, commit `814e7f0a17ac45c9bd4a63958aafb3cffaddfe22`; ponta de `main` conferida em 14/09; clone do notebook no mesmo commit | **[FIXADO]**; reconferir a ponta antes de congelar |
| Mosaic, dados | release `clinvar-pb-capability-suite/v1` em `~/mosaic-v1/`. Já conferido: `membership.parquet` com o hash lógico de referência (`1c1cd65d…`, 8.875 linhas). O S3 ainda usa o layout anterior ao ADR 0006 (`bundle.manifest.json` em vez de `release.manifest.json`) | **[FIXADO]** como fonte; registrar o hash real de cada arquivo usado |
| ABraOM | snapshot `abraom_sabe1171` do source-lock do Mosaic (TSV SABE-WGS-1171, sha256 `3cd33784…`), obrigatório para o regionalizado | **[FIXADO]**; localizar o arquivo |
| gnomAD (M1) | v4.1 joint com AF por grupo (`s3://ai4bio-lumina/data/external/gnomad-joint-v4.1/`) | **[PROPOSTO]**; fixar versão e grupos |
| Genoma | GRCh38, `hg38.fa` (sha256 `056974f6…`) | **[FIXADO]** |
| Snapshot de treino da cabeça | ID, hash e cutoff, exigidos pelo manifesto do Mosaic | **[ABERTO — decisão B]** |

Código e dados têm identidades separadas: ter o commit certo não prova que o release foi gerado por ele.

---

## 2. Pergunta, sistemas e contrastes

**Pergunta desta campanha [PROPOSTO — decisão A]:** a adaptação populacional com ABraOM produz ganho diferencial em
variantes com participação de instituições brasileiras, comparada com uma adaptação a variação humana global, nos
casos e controles congelados do Mosaic?

| Sistema | Representação | Classificador clínico |
|---|---|---|
| M0 | R03 original, congelado | H0 |
| M1 | mesmo R03 + adapter populacional global, congelado depois de treinado | H1 |
| M2 | mesmo R03 + adapter populacional ABraOM, congelado depois de treinado | H2 |

**[FIXADO]**

- O checkpoint-base (ancestral comum) é o R03, não o sistema M0 com H0 já treinada.
- Os adapters de M1 e M2 são ramos paralelos do R03: M2 não parte de M1.
- H0, H1 e H2 são treinadas separadamente, com os mesmos dados, a mesma arquitetura, o mesmo procedimento e o mesmo
  orçamento, depois de congelar a representação de cada sistema. Isso mede o benefício da adaptação no **sistema
  completo, depois de treinar sua cabeça**. Manter H0 fixa e trocar só a representação seria outro experimento.
- Pesos do R03 e cabeças nativas congelados. Sem adapter ClinVar, sem fusion e sem AF observada nas entradas da
  cabeça: AF e presença ficam como baselines diagnósticas separadas. M3 e M4 ficam para depois.
- As cabeças nativas do R03 já carregam informação populacional do pré-treino: M0 não é "sem informação
  populacional".

**Contrastes [PROPOSTO — decisão C], registrados antes de qualquer resultado:**

| Contraste | Papel |
|---|---|
| M2 − M1 | pergunta regional principal: efeito do ABraOM além da adaptação global (controle do PDF). É extensão do consumidor: o Mosaic compara só base × regionalizado |
| M2 − M0 | efeito total da regionalização, no formato base × regionalizado do Mosaic |
| M1 − M0 | diagnóstico do ganho genérico de adaptação |

O ancestral comum não basta para cumprir o protocolo do Mosaic: faltam os dados da cabeça, as exclusões, o
congelamento e o manifesto (seções 4, 5 e 8). Como registrar os sistemas e as cabeças no manifesto é ponto para o
Eduardo.

---

## 3. O que o estudo brasileiro do Mosaic exige

Verificado em `PROTOCOLO.md` (Estudo brasileiro), `src/mosaic/protocol.py:brazil_protocol_section`,
`src/mosaic/brazil_study.py`, `config/suite.yaml`, `docs/GUIA_OPERACIONAL_DE_SCORING_DOS_ESPECIALISTAS.md` (§5.3 e §6)
e `src/mosaic/comparator_eval/`.

| Regra | Consequência para o plano |
|---|---|
| Modo `frozen_pair_evaluation`; sistemas `base` e `regionalized`; nenhum treino, seleção ou calibração dentro do estudo | sistemas e limiares congelados antes de pontuar o estudo |
| `release_training_allowed`, `release_model_selection_allowed` e `release_threshold_calibration_allowed` = `False` | não treinar nem calibrar com o release sem autorização (decisão B) |
| Dataset de treino = snapshot global do consumidor, declarado por ID, hash e cutoff, **não materializado no release** | construir e declarar o snapshot da cabeça |
| O regionalizado parte do mesmo dataset e do mesmo checkpoint-base; fonte obrigatória `abraom_sabe1171` | R03 comum; H2 treinada no mesmo snapshot de H0 |
| Dois estudos, nunca unidos: `br_clinical_evidence` (consensus com participação brasileira) e `br_population_observed` (gold presente no ABraOM) | o clínico é a avaliação principal; o populacional é descritivo, sobreposto ao ABraOM por construção |
| Pares 1:1 sem reposição em rótulo × painel × bin de AF do gnomAD, sem gene; papéis `case`, `unmatched_case` e `control` | importar os pares, nunca refazer |
| Deltas no coorte completo, nos casos pareados e nos controles; interação `delta_br_matched − delta_control`, sem os casos não pareados; deltas na interseção de cobertura | seção 6 |
| AUROC e AUPRC; métricas com limiar só com limiar externo congelado; macro brasileira não exigida | seção 6 |
| No estudo clínico, relatar também o subconjunto `present_abraom` | obrigatório |
| Declarar sobreposições: SCVs ou instituições brasileiras, variantes do estudo, outras fontes de regionalização | inclui o gnomAD do M1 |
| Bootstrap por `overlap_cluster_id`, 1.000 réplicas, percentis 2,5 e 97,5, seed `20260901` | o `comparator_eval` reamostra cada coorte separadamente e **não calcula a interação**: a regra conjunta é do consumidor (6.3) |
| `lockbox = False`; `independent_generalization_claim_allowed = False`; margens de sucesso declaradas antes de ver scores | campanha de desenvolvimento |

---

## 4. Dados

### 4.1 Estudo brasileiro [FIXADO]

Importar `studies/brazil/membership.parquet` sem refazer o pareamento e validar: IDs únicos por estudo e papel,
ligações bidirecionais entre caso e controle, rótulos e estratos coerentes com `pb_examples` e `pb_panels`.

Contagens do release (8.875 linhas):

- clínico: 3.119 casos (3.116 pareados e 3 sem controle; 2.808 P / 311 B no coorte completo) e 3.116 controles;
- populacional: 1.889 casos (751 pareados e 1.138 sem controle) e 751 controles.

O tamanho efetivo sai da recontagem de P/B por coorte e painel **depois da cobertura** de cada sistema, não destas
contagens.

### 4.2 Snapshot de treino da cabeça [ABERTO — decisão B]

Exclusões fixas, qualquer que seja a fonte:

1. todos os membros dos dois estudos (casos, casos sem controle e controles) e as variantes dos seus
   `overlap_cluster_id`;
2. qualquer variante com SCV de instituição da lista brasileira do Mosaic, de qualquer classificação, origem ou
   contribuição; ausência de informação não vira "não brasileira";
3. chr8, se continuar reservado (decisão E).

Medir o custo de cada exclusão. Rótulos propostos: P/B da política `clinvar-germline-pb/v1`, tiers gold + consensus,
a mesma qualidade de rótulo do estudo. Divisão treino/validação/calibração por variante canônica, com manifestos e
hashes.

| Opção | Fonte | A favor | Contra |
|---|---|---|---|
| S1 | ClinVar 2026-06 cru, construído por nós com o código de política do Mosaic | não usa o artefato do release | na prática, o mesmo conteúdo de S2; exige `variant_summary_2026-06` (439 MB) e um catálogo próprio, porque o `variation_allele` fixado pelo Mosaic não está mais disponível no NCBI (declarar a diferença) |
| S2 | `pb_examples` do release, menos as exclusões | rótulos idênticos aos do estudo; identidade por hash; pouco trabalho | contraria `release_training_allowed = False`: só como **protocolo derivado de desenvolvimento**, autorizado pelo Eduardo |
| S3 | ClinVar mais antigo (por exemplo, 2021-12, o T0 do Mosaic, com sha256 fixado) | separação temporal entre os rótulos de treino e os do estudo | menos exemplos; também exige construir rótulos |

S1 e S2 têm quase o mesmo conteúdo; a diferença é formal. Por isso a escolha é do Eduardo, que mantém o Mosaic.

### 4.3 Dados populacionais [PROPOSTO]

- ABraOM: o snapshot do source-lock. gnomAD v4.1 por grupo para o sampler global; o `af_gnomad` antigo, condicional
  ao índice ABraOM, não serve.
- Janelas: nenhum alelo de variante dos estudos e nada do chr8. **[ABERTO]** política para janelas que cobrem
  posições dos estudos (contexto de referência sim, alelo nunca).
- Membership serve só para exclusão e auditoria, nunca para escolher alelos ou alvos de treino.
- Registrar a sobreposição que não puder ser eliminada, inclusive a do pré-treino do R03 (fontes e chr8).

---

## 5. Treino

### 5.1 Adapters populacionais [PROPOSTO — decisão D]

Piloto com MLM nas janelas sintéticas, com a mesma loss e o mesmo orçamento nos dois braços e as cabeças nativas
congeladas. A especificação obrigatória está no contrato v2 §2: construção das janelas, posições mascaradas, loss
relatada separadamente em posições variantes e de referência, validação populacional separada, e não chamar as
janelas de haplótipos. AF supervisionada é uma alternativa distinta, não um substituto.

Primeiro um smoke sintético, depois um piloto pequeno em dados permitidos, com uma seed.

### 5.2 Extração [PROPOSTO]

Candidatas: as 172 dimensões de cabeça da pesquisa e a leitura antiga completa (`site_ref`, `variant_repr` e o
contexto local de ±64 bp; a aproximação de 896 dimensões avaliada na pesquisa não tinha esse contexto). Escolher
**uma** no desenvolvimento, na validação do snapshot e depois do piloto, com critério declarado antes, e usar a mesma
em M0, M1 e M2. Nunca escolher no estudo brasileiro.

O extrator está na branch `embedding-probe-mosaic` e precisa ser portado com proveniência. O cache é identificado
por R03, adapter, versão do extrator e chaves das variantes.

### 5.3 Cabeças e calibração [PROPOSTO]

Mesma arquitetura e procedimento em H0, H1 e H2; padronização ajustada só no treino; early stopping na validação;
Platt e limiar de MCC no calibration do snapshot, congelados por sistema e iguais para casos e controles. A campanha
usa repetições de adapter e cabeça (PDF §10: pelo menos três), com predição final pela média das probabilidades
calibradas e métricas também por seed.

---

## 6. Avaliação, pré-registrada antes de pontuar o estudo

### 6.1 Saída no formato do Mosaic [FIXADO pelo protocolo]

Para o par M0 (base) × M2 (regionalizado), em cada estudo separadamente:

- AUROC e AUPRC no coorte completo, nos casos pareados e nos controles, com `n_P`, `n_B`, prevalência e cobertura;
- deltas na interseção de cobertura e a interação `delta_br_matched − delta_control`;
- resultados por painel no coorte completo;
- no estudo clínico, também o subconjunto `present_abraom`.

Reaproveitar do `comparator_eval` do Mosaic as visões (`brazil_views`) e a reamostragem por grupo, para garantir as
mesmas coortes e a mesma unidade.

### 6.2 Extensão do consumidor [PROPOSTO]

As mesmas quantidades para M2 × M1 (pergunta principal) e M1 × M0, com os mesmos pares e a interseção de cobertura de
cada contraste. Reportar sempre os ganhos absolutos: uma interação positiva pode vir de piora no controle.

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

### 6.5 Margem e precisão [ABERTO]

A margem de 0,02 é proposta, não requisito do Mosaic nem expectativa de resultado. Antes de fixá-la, estudar precisão
e poder em cenários plausíveis: pares disponíveis depois da cobertura, estrutura de clusters, correlação entre os
scores dos sistemas e tamanho de efeito assumido. Usar dados de desenvolvimento ou simulação, nunca resultados dos
candidatos no estudo.

### 6.6 Partes do PDF fora do Mosaic [ABERTO — decisão E]

- **chr8 representacional** (PDF §11.1): continua possível como validação populacional dos adapters, se o chr8 ficar
  fora das janelas e do treino da cabeça.
- **BRCA1/BRCA2/TP53** (PDF §12): os controles do Mosaic não são pareados por gene, então não há interação por gene;
  no máximo, relato descritivo dos casos.

---

## 7. O que poderá ser afirmado

- Um ganho positivo sustenta transferência diferencial no recorte de **participação** brasileira do Mosaic.
- Não poderá ser afirmado: resultado em só-BR, em pacientes brasileiros ou na população brasileira nacional;
  generalização independente; o estudo populacional como evidência independente.
- A interação não elimina diferenças entre laboratórios ou mecanismos de rotulação.
- Tier consensus não prova ausência de exposição. Conferir casos, controles e clusters contra o histórico: a pesquisa
  de extração usou gold, e os folds de `core_locus` e `gene_transfer` do Mosaic treinam com consensus.

---

## 8. Ordem de execução e gates

| Gate | Entrega | Critério para avançar |
|---|---|---|
| G0 | Identidades da seção 1 | hashes reais de R03, release, ABraOM e gnomAD registrados |
| G1 | Importação e validação do membership | checagens da 4.1 sem erro; contagens por coorte e painel |
| G2 | Snapshot de treino da cabeça, depois da decisão B | sobreposição zero com estudos, clusters, regra ampla brasileira e chr8; custo das exclusões medido |
| G3 | Extrator portado e smoke de M0 sem LoRA clínico | só a cabeça recebe gradiente; cache com identidade |
| G4 | Gerador e MLM: smoke sintético e piloto M1/M2 | só o LoRA recebe gradiente (conferir `freeze_native_feature_heads` de `eval/clinvar/lora.py`); aprendizado nas posições variantes; manifestos de janelas comparáveis entre braços |
| G5 | Escolha da extração no desenvolvimento | critério declarado antes; mesma extração para todos os braços |
| G6 | Sistemas congelados, manifesto do consumidor, margens e regra de bootstrap declaradas | nada ajustado depois de ver o estudo |
| G7 | Avaliação única no estudo | saídas da seção 6 |

Antes das decisões dá para preparar G0, G1, interfaces, fixtures e inspeção de fontes. Não iniciar treino completo
nem pontuar o estudo.

**Não são pré-requisitos nesta rota:** contar só-BR de 1 estrela, explicar cada inconsistência histórica da v1, mudar
a política do Mosaic e refazer os pares. O `variant_summary` só volta a ser necessário se a decisão B escolher S1 ou
S3, e por esse motivo.

---

## 9. Pauta para o Eduardo

1. **Pergunta (A):** aceitar a participação brasileira do Mosaic nesta campanha e adiar só-BR?
2. **Treino da cabeça (B):** snapshot externo (S1 ou S3) ou protocolo derivado do release (S2)? Com quais rótulos e
   qual cutoff?
3. **Comparação (C):** confirmar adapters paralelos sobre o mesmo R03, cabeças treinadas separadamente, M2 × M1 como
   contraste regional e M2 × M0 no formato do Mosaic; e como registrar sistemas e cabeças no manifesto.
4. **Objetivo populacional (D):** confirmar o piloto MLM global/ABraOM.
5. **Escopo (E):** manter ou adiar a avaliação representacional no chr8 e o relato BRCA1/BRCA2/TP53.

Pontos do Mosaic para registrar, sem bloquear a campanha: a URL do `submission_summary_2026-06` no `sources.yaml`
aponta para uma pasta que não existe; `origin_has_germline` só aceita a origem literal `germline` (`de novo`,
`maternal`, `inherited` e `unknown` ficam fora); nenhuma variante gold recebe marcação brasileira, provavelmente porque,
com painel de especialistas, as demais SCVs não contribuem para o agregado.

### Texto para encaminhar

"Queremos continuar a regionalização do R03 sem adapter ClinVar e sem fusion, usando o estudo brasileiro já publicado
no Mosaic. Fixamos o R03 publicado (`best_checkpoint`, passo 71.000) como backbone comum e o
`croma-bioai/lumina-mosaic` como benchmark. A auditoria mostrou por que os splits antigos não servem, e só-BR com
rótulos de qualidade praticamente não existe, então propomos medir participação brasileira nesta etapa. Os sistemas
seriam M0, M1 global e M2 ABraOM, com cabeças treinadas separadamente pelo mesmo procedimento, M2 × M1 como contraste
regional e M2 × M0 no formato base × regionalizado. Como o estudo só avalia sistemas congelados, precisamos da sua
decisão sobre a fonte de treino da cabeça: um snapshot externo declarado ou um protocolo derivado do release. Você
também concorda com o piloto MLM e com adiar só-BR? Nada do estudo será usado para escolher a extração, ajustar
modelos ou calibrar."
