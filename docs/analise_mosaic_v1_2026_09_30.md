# Mosaic v1 (`mosaic-v1-2026-09-30`): como funciona e como os splits brasileiros são montados

Análise do repositório `lumina-mosaic` (commit `95adc39`, 01/10/2026; código da entrega `f2e9a9f`) e do
`lumina-inference` (R03), feita para preparar uma nova frente de regionalização do R03. O caminho proposto pelo
Eduardo é o inverso da campanha anterior: partir de como o Mosaic monta os splits brasileiros e então propor uma
regionalização que melhore o R03 neles sem piorar os outros. Este documento só mapeia; a proposta vem depois.

Convenções: **fato** = lido no código, na configuração ou em número publicado pelo Mosaic, com a fonte;
**inferência** = leitura minha a partir desses fatos, marcada como tal. Números do Mosaic são da identidade atual
(`d9312580…`), salvo indicação.

## 0. Resumo em dez linhas

1. O Mosaic virou um benchmark independente: entrega dados, partições, protocolos e um avaliador de referência;
   treino, embeddings e adaptação ficam com o consumidor (ADR 0012).
2. Há agora **duas famílias brasileiras distintas**: o track `brazil` (os dois proxies pareados que usamos no G7,
   rebaixado a secundário) e o estudo **`regional`**, novo, construído sobre o dump completo do ABraOM.
3. O `brazil` continua com `br_clinical_evidence` (consensus com laboratório brasileiro) e
   `br_population_observed` (gold presente no ABraOM), pareados 1:1 por classe × painel × bin de AF do gnomAD.
4. O `br_population_observed` **mudou** com o dump completo do ABraOM: 2.057 casos (98 P), 621 pares (eram 1.889 e
   751 no release do G7).
5. O `regional` cria verdade benigna por frequência no ABraOM (camadas A ≥ 5% e B 1–5%), uma verdade clínica (B/LB e
   P/LP observadas no ABraOM) e variantes críticas brasileiras; mede **viés**, **benefício** e **segurança**.
6. O viés compara falsos positivos em benignas comuns no Brasil e raras no gnomAD (célula primária, 10.829) contra
   benignas com frequência parecida nas duas fontes (célula comparável).
7. Um sistema treinado participa do `regional` pelo perfil de 4.096 bp: cada variante é pontuada pelo modelo do
   `core_locus` de 4 kb da execução que a cobre (cross-fitting), com o limiar da validation daquela execução.
8. Regionalização aprendida (R2) só vale com treino nos blocos de 1 Mb **expostos** e comparação contra uma
   continuação global de mesmo orçamento (R2c) nos blocos **não expostos**.
9. O R03 tem uma cabeça populacional (`population_af_head`, `population_observed_head`) cujo alvo é a AF global do
   gnomAD (`AF_joint`). O Mosaic formula exatamente isso como hipótese de mecanismo do viés regional (H-R).
10. Os splits recompensam informação que só existe nos casos brasileiros (presença e frequência no ABraOM). Isso
    define o que "roubar" significa e onde está o risco: segurança das patogênicas brasileiras (margem de 1 ponto).

## 1. O Mosaic novo em uma página

**Fronteira.** O builder não emite métricas (`builder_emits_metrics=false`); o avaliador de referência
(`src/mosaic/comparator_eval/`) avalia especialistas e predições externas e escreve em `outputs/`; o consumidor
produz `system.yaml` e `predictions.parquet` (ADRs 0008, 0012). Não há lockbox: núcleo e cortes de VUS até 2026-06
são evidência de desenvolvimento; confirmação só em rótulos posteriores a 2026-06, com tudo congelado antes (ADR
0011).

**Universo.** ClinVar 2026-06, GRCh38.p14, SNVs autossômicos canônicos com REF conferido: 326.826 exemplos, 10.761
gold (5.908 P / 4.853 B), 316.065 consensus. Painéis mutuamente exclusivos: `missense`, `splice`, `noncoding`
(discriminação), `plof` (guardrail de sensibilidade), `synonymous` (guardrail de especificidade), `other`
(descritivo) (`README.md`).

**Tracks e estudos.**

| Parte | Pergunta | Desenho |
|---|---|---|
| `core_locus` | Generaliza para contexto genômico não visto? | 5 folds bloqueados por cluster de overlap da vista |
| `gene_transfer` | Transfere para genes não vistos? | 5 folds bloqueados por grupo de gene |
| `time` | Funciona em alelos novos e promoções de VUS? | treino/validation em 2021-12, teste em 2026-06 |
| `brazil` | Base × regionalizado em dois proxies brasileiros | par congelado, sem treino no estudo |
| `vus` | Classifica VUS resolvidas depois? | cortes 2021-12, 2023-12, 2024-12, 2025-12 |
| genoma inteiro | Discrimina em qualquer classe de região? | contra roteamento fixo de especialistas |
| `regional` | Reduz erros em variantes observadas no Brasil, preservando P? | verdade do ABraOM; viés, benefício, segurança |
| funcional | Acompanha splicing, regulação, causalidade? | SpliceVarDB, MPRA, TraitGym, caQTL |

**Vistas por janela (ADR 0009).** O release tem vistas W ∈ {4.096, 16.384, 32.768} bp, cada uma com
`sequence_eligible`, clusters (`|Δpos| < W`), folds e contagens próprios; rótulos, painéis e anotações são
compartilhados. Um sistema de contexto C é avaliado nas vistas com W ≥ C. Unidades com gold: 397 (4 kb), 181 (16 kb),
142 (32 kb). A vista de 4 kb corta clusters em segmentos de até 32 kb (`core_unit_id`) e purga, por execução,
validation a menos de W do teste e treino a menos de W do teste ou da validation (`core_purged_runs`); a unidade de
inferência continua o cluster transitivo (`overlap_cluster_id`). BRCA1 e BRCA2 são 33% do gold e um cluster cada
(`specs/PLAN-vus-genoma-regional.md` §9.1).

**Protocolo do consumidor.** Execução `i`: teste = fold `i` (gold), validation = fold `(i+1) mod 5` (gold), treino =
os outros três (gold + consensus, sem o consensus dos folds de teste e validation, sem as purgas). Hiperparâmetros
pela macro AUROC de missense, splice e noncoding na validation; um limiar por sistema e execução pelo MCC máximo no
gold completo da validation (desempate: especificidade, maior limiar). Bootstrap de 1.000 réplicas, seed `20260901`,
na unidade de bloqueio (`PROTOCOLO.md`).

**Arquivos (nomes novos, ADR 0013).** Na raiz `artifacts/mosaic-v1-2026-09-30/`:

| Arquivo | Conteúdo | Nome antigo |
|---|---|---|
| `clinical-variants.parquet` | rótulo, tier, coordenadas, `br_lab_any`, status T0 | `pb_examples.parquet` |
| `views/{4kb,16kb,32kb}/partitions.parquet` | elegibilidade, folds, unidades, purgas da vista | `views/w4096/` etc. |
| `reference-partitions-32kb.parquet` | partições de referência de 32 kb (temporal, VUS) | `pb_partitions.parquet` |
| `evaluation-panels.parquet` | painel primário | `pb_panels.parquet` |
| `variant-annotations.parquet` | gnomAD, ABraOM, conservação, especialistas | `pb_annotations.parquet` |
| `studies/brazilian-proxies/membership.parquet` | casos, sem par e controles dos dois proxies | `studies/brazil/membership.parquet` |
| `studies/temporal/variants.parquet` | universo temporal | `studies/time/examples.parquet` |
| `evaluation-counts.parquet` | contagens e cobertura | `suite_counts.parquet` |
| `protocol.json`, manifests | contrato, identidade | — |

Fora do release, mas na entrega: `data/annotations/abraom/sabe1171-wgs/` (extrato do dump),
`data/annotations/population-observations/`, `data/annotations/bias-cells/`, a verdade regional e as anexações às
vistas, e o manifesto de pedidos `outputs/candidate-manifest/` (`variants.parquet`, `requests.parquet`). Entrega
em `s3://croma-bioai-lumina-releases-us-east-2/benchmarks/mosaic/releases/mosaic-v1-2026-09-30/` (cerca de 300
GiB, 16.075 arquivos; `docs/PUBLICACAO.md`), identidade `d93125804e7c…`, protocolo `67470b1443c4…`.

## 2. As duas famílias brasileiras

| | Track `brazil` (proxies) | Estudo `regional` |
|---|---|---|
| Papel | secundário pareado | primário da regionalização |
| Verdade | rótulos P/B do ClinVar | frequência no ABraOM (camadas A/B), clínica observada no ABraOM, críticas |
| Universo | membros do release | dump completo do ABraOM (60,7 milhões de SNVs) + núcleo de 4 kb |
| Sistemas | par base × regionalizado congelados | candidato, RW-2/RW-3, R0/R1, R2/R2c |
| Vista | referência de 32 kb, congelada | perfil fixo de 4 kb, cross-fitting |
| Treino no estudo | proibido | proibido; R2 só nos blocos expostos |
| Avaliação | manual (o CLI não aceita `study: brazil`) | `scripts/evaluate_submission.py --studies regional` |

### 2.1 Track `brazil`: como os dois proxies são montados

Código: `src/mosaic/brazil_study.py` (`build_brazil_membership`, `match_controls`).

- **Só `sequence_eligible`.** A população de partida são exemplos elegíveis do release.
- **`br_clinical_evidence`.** Casos: `label_tier == consensus` e `br_lab_any`. Pool de controles: consensus sem
  `br_lab_any`. `br_lab_any` = pelo menos um SCV ativo germinativo P/B de instituição do freeze brasileiro
  (`config/clinvar-organizations-br.yaml`: `organization_summary` de 2026-08-22, país Brazil/Brasil, revisão manual).
  3.119 casos (2.808 P / 311 B).
- **`br_population_observed`.** Casos: `label_tier == gold` e `present_abraom`. Pool: gold sem `present_abraom`.
  `present_abraom` = AC > 0 no dump agregado do WGS-1171, com **qualquer FILTER** (`annotations/abraom.py:
  classify_abraom`). 2.057 casos (98 P / 1.959 B).
- **Pareamento.** 1:1, sem reposição, exato em `binary_label × primary_panel × gnomad_af_bin`; casos e opções de
  controle ordenados por `hash(seed, variant_id)`. Caso sem controle no estrato vira `unmatched_case`.
  `gnomad_af_bin` ∈ {`absent`, `ac0`, `rare` (< 1e-4), `intermediate` (1e-4 a 1%), `common` (≥ 1%)} (`CONTEXT.md`).
- **Leitura.** Coorte completo = `case` + `unmatched_case`; pareado = `case`; controle = `control`; interação =
  Δ_BR_matched − Δ_control, sem `unmatched_case`. Deltas na interseção de cobertura, bootstrap por
  `overlap_cluster_id` da referência de 32 kb, sem limiar brasileiro (MCC só com limiar externo declarado antes).
- **Contrato do par.** Base e regionalizado partem do mesmo snapshot global congelado **declarado pelo
  consumidor** (ID, hash, cutoff); o regionalizado usa `abraom_sabe1171`; membership, controles e rótulos não entram
  em treino, seleção ou calibração (`BRAZIL_TRAINING_CONTRACT`).

**O que mudou desde o G7 (fato).** O G7 usou o release do commit `814e7f0`, em que o ABraOM era o recorte de 928
genes, só com AF. O release atual usa o dump completo (AC, AN, homozigotos, filtros). Por isso `present_abraom`
mudou e o `br_population_observed` passou de 1.889 casos e 751 pares para 2.057 casos (98 P) e 621 pares
(`specs/PLAN-vus-genoma-regional.md` §10.1). O `br_clinical_evidence` não depende do ABraOM e mantém 3.119 casos.

**Números dos especialistas (fato; `docs/resultados/leaderboard-didatico.md` §6, só pareados).**

| Proxy | Painel | Referência | AUROC BR | AUROC controle | P/B BR pareados |
|---|---|---|---|---|---|
| Instituição BR | missense | REVEL | 0,936 [0,909; 0,958] | 0,973 [0,960; 0,984] | 1.242/192 |
| Instituição BR | splice | SpliceAI | 0,991 | 0,998 | 408/33 |
| Instituição BR | noncoding | gnomAD rarity | 0,839 | 0,867 | 95/60 |
| Observado no ABraOM | missense | REVEL | 0,931 [0,880; 0,967] | 0,919 [0,872; 0,954] | 68/247 |
| Observado no ABraOM | splice | SpliceAI | 0,973 | 0,948 | 10/20 |
| Observado no ABraOM | noncoding | gnomAD rarity | 0,756 | 0,670 | 4/116 |

**Inferências sobre a estrutura.**

- No populacional, o controle é gold **ausente** do ABraOM no mesmo bin do gnomAD. Variante comum no gnomAD quase
  sempre aparece em 1.171 genomas, então os casos do bin `common` tendem a ficar sem controle. Isso explica por que
  o coorte completo é dominado por benignas comuns sem par. No release do G7, os 1.138 sem par eram 1 P e 1.137 B,
  a maioria noncoding; agora são 1.436, contagem por painel ainda não medida.
- Dentro de cada par, o bin do gnomAD é igual; o que difere entre caso e controle é justamente a observação no
  ABraOM. Qualquer informação derivada do ABraOM existe **só nos casos**. Uma feature de ABraOM pode mudar o caso e
  nunca o controle, o que tende a produzir interação positiva por construção.
- No clínico, os especialistas já ordenam pior os casos brasileiros que os controles em missense (0,936 contra
  0,973). É o único gap visível nos proxies, e o único em que um ganho regional teria margem para aparecer sem
  depender só do ABraOM. No G7 os casos do clínico estavam 4,6 vezes mais presentes no ABraOM que os controles (com o
  recorte antigo; a razão com o dump completo não foi medida).

### 2.2 Estudo `regional`: o universo e a verdade

Código: `src/mosaic/regional.py`, `src/mosaic/regional_truth.py`, `src/mosaic/bias_cells.py`,
`src/mosaic/population_observations.py`, `scripts/build_regional_universe.py`, `scripts/build_regional_truth.py`,
`scripts/build_bias_cells.py`, `scripts/attach_external_views.py`.

**Universo regional.** O dump completo do ABraOM SABE-WGS-1171 (`SABE1171.Abraom.clean.tar.gz`, hg38): 60,7
milhões de SNVs autossômicos (52,3 milhões PASS), com AC, AN, homozigotos, FILTER e a classe CEGH-Filter; REF
conferido (0 divergências), junto ao extrato do gnomAD v4.1 joint (85,9% presentes). O dump só lista sítios com
variante: ausência não tem denominador (`observed_only`).

**Camadas de verdade benigna** (`regional_truth.truth_for_chrom`; status atribuído nesta ordem):

1. `qc_fail` (FILTER ≠ PASS ou AN < 80% do máximo do cromossomo) ou `qc_low_confidence` (CEGH-Filter ≠ `vSR`;
   filtro do Mosaic, mais estrito que a alta confiança do ABraOM);
2. `excluded_critical` (lista crítica brasileira);
3. `excluded_label` (P/LP ou conflitante no ClinVar);
4. `excluded_list` (exceções de BA1, variantes nomeadas no CSpec, listas de VCEP);
5. `no_specification` (gene com BA1 ambíguo que alcança o piso, como PAH na camada B);
6. `below_ba1` (gene com BA1 do CSpec que o limite inferior unilateral de 95% da AF não ultrapassa);
7. `benign`.

| Camada | Definição | Benignas finais | Uso |
|---|---|---:|---|
| A | AF no ABraOM ≥ 5% | 6.004.793 | verdade benigna para o erro base |
| B | 1% ≤ AF < 5% | 4.073.308 | verdade benigna para testar o viés |
| C | AF < 1% | — | só descritiva |
| Clínica | B/LB do ClinVar observadas no ABraOM | gold 1.959, consensus 131.050 | benefício |
| P-BR | P/LP do ClinVar observadas no ABraOM | gold 98, consensus 952 | segurança |
| Críticas | 13 SNVs P/LP da literatura (HbS, TP53 p.R337H e outras) | — | tolerância zero |

Limitação declarada pelo próprio plano: as camadas são proxies de frequência numa coorte de idosos de São Paulo, não
aplicação clínica de BA1/BS1; a HbS mostra o risco (patogênica e comum no Brasil). A limpeza usa rótulos do ClinVar,
então concordância das camadas com o ClinVar não é evidência.

**Células do viés** (`regional_truth.bias_cell`, só na camada B benigna):

| Célula | Regra | Variantes |
|---|---|---:|
| `primary` | AF no ABraOM ≥ 5 × AF_joint do gnomAD (`ac0` = 0) | 10.829 (2.111 blocos de 1 Mb; 5.034 em blocos não expostos) |
| `comparable` | AF_joint dentro de um fator 2 da AF do ABraOM | 3.854.004 (pontuada numa amostra de 100.000 + censo codificante) |
| `gnomad_absent` | ausente do gnomAD | 1.885 (estrato descritivo) |
| `other` | o resto | — |

5× e 2× são convenções do Mosaic fixadas só com contagens. Raridade absoluta não serve: `AF_joint` < 1e-3 deixa 202
variantes, e popmax < 1e-3 deixa 18, porque AMR e AFR do gnomAD já cobrem quase tudo o que é comum no Brasil. Em
contraste, 2,17 milhões de benignas da camada B são pelo menos 10 vezes mais comuns no Brasil que no NFE.
Sensibilidades pré-registradas: razões 3× e 10×, fatores 1,5 e 3, qualidade `vSR`/`SR`/`WK`, e controle positivo
com AFR e AMR do gnomAD no papel de "região" (`config/study-protocol.yaml`, `regional`).

**Partição do R2.** Blocos de 1 Mb, metade exposta por `sha256(seed, "r2", chrom, bloco) mod 2`; bloco não exposto a
menos de 32 kb de um exposto vira `buffer`. Na camada B benigna: 1.989.780 expostas, 2.017.139 não expostas, 66.389
no buffer (`regional_truth.r2_roles`).

### 2.3 Estudo `regional`: intervenções, endpoints e margens

**Braços** (`specs/PLAN-vus-genoma-regional.md` §7.4):

| Braço | O que é | Afirmação possível |
|---|---|---|
| R0 | sistema global | referência |
| RW-3 | ACMG computacional automatizado (RW-2) + ABraOM nos critérios populacionais | comparador regional de mundo real |
| R1 | candidato + ABraOM como evidência separada, parâmetros da literatura, sem rótulo brasileiro | comportamento do produto nas variantes do ABraOM |
| R2 | regionalização aprendida (ex.: ABraOM como alvo populacional adicional, ou PEFT) | generalização, só em blocos não expostos |
| R2c | continuação global com o mesmo orçamento | controle obrigatório do R2 |

Na prática do avaliador, para um sistema treinado: **R0 = RW-4**, isto é, o RW-2 com o candidato como provedor de
PP3/BP4, calibrado pela adaptação do método de Pejaver na validation de cada execução, até força "forte"
(`scripts/build_rw4.py`). **R1 = RW-4 com as regras do ABraOM** (`--abraom`): com a AF do ABraOM das chamadas
`vSR`, BA1 se o limite inferior unilateral de 95% ≥ 5% ou acima do BA1 do gene; BS1 acima do BS1 do gene; sem PM2
quando observada no ABraOM (`config/real-world.yaml`). Nas camadas A/B, R1 e RW-3 aplicam a mesma regra que define a verdade e acertam por
construção; lá só informam os sistemas sem ABraOM (viés) e o R2 em blocos não expostos.

**Endpoints primários** (`config/study-protocol.yaml`, `primaries`; efeitos mínimos aprovados em 2026-09-29):

| Claim | Contraste | Coorte | Efeito mínimo / margem |
|---|---|---|---|
| Viés | taxa de FP na célula primária − comparável, ajustada por painel × quartil do phyloP241 | camada B benigna, 4 kb | 1 ponto |
| Benefício | R1 − RW-3 em cobertura de decisões corretas, com segurança | teste do núcleo de 4 kb presente no ABraOM: 2.057 gold, 98 P-BR, 232 unidades | 4 pontos |
| Segurança | perda bruta de P-BR que o R0 reconhecia como P/LP e o R1 perde | P-BR de teste do núcleo de 4 kb, gold e consensus | ≤ 1 ponto, Clopper–Pearson com n = grupos de gene (580 em 4 kb); críticas com tolerância zero |
| Generalização do R2 | redução de FP nas camadas A/B em blocos não expostos contra R2c | camadas A/B | 25% relativo na célula primária (provisório) |
| Custo global | regionalizado − base no `core_locus` (macro AUROC, sensibilidade) | gold do núcleo | descritivo, sem margem |

A taxa de FP do viés usa o limiar de MCC da validation do núcleo de 4 kb de cada execução, sobre todas as variantes
que o sistema pontua (`scripts/evaluate_regional_bias.py`). Como as células só têm benignas, toda chamada positiva é
FP. A simulação de segurança mostra o custo da margem: só um candidato que praticamente não perde P-BR demonstra
segurança (com perda verdadeira de 0,25%, demonstra em 26–47% das vezes).

**Referências medidas (fato; desenvolvimento).**

- Viés, definição principal, diferença ajustada em pontos: phyloP +0,86 [0,08; 1,69]; roteamento +0,14; CADD −0,35;
  REVEL −0,35; AlphaMissense −1,94; SpliceAI −3,04; frequência sozinha +0,04; raridade gnomAD +0,03; RW-2 0,00. O
  controle positivo detecta viés onde se espera (roteamento +2,21 no AFR; AlphaMissense +7,18 e REVEL +4,25 no AMR).
- Benefício: RW-3 tem 89,8% de decisões corretas [85,1; 92,9], sensibilidade de 15,3% nas P-BR e FP zero; um
  substituto no formato R1 (CADD de genoma calibrado por execução) ganha +2,0 pontos [0,2; 3,9].
- Frequência domina os rótulos: regressão logística só com frequência do gnomAD tem AUROC gold 0,853 (missense 0,922,
  splice 0,850, noncoding 0,953). Somar AF e presença no ABraOM dá Δ +0,0001 [−0,0002; +0,0002]; no subconjunto
  presente no ABraOM, −0,0008. **Esses dois números vieram do recorte antigo do ABraOM** e o plano avisa que mudam
  com o dump completo (§1, fatos 1 e 3).

### 2.4 Como um sistema treinado participa do `regional`

- **Perfil fixo de 4 kb.** `regional` e `functional` aceitam só contexto ≤ 4.096 bp. O R03 lê até 32 kb, mas aqui
  entra com janela de 4.096 (a mesma da campanha anterior).
- **Cross-fitting** (`clusters.anchor_external`, ADR 0009). Cada variante externa registra as unidades do núcleo a
  menos de W, sem criar arestas. É pontuada pela execução cujo fold de teste contém todas as unidades tocadas; sem
  unidade tocada, pela execução `hash(seed, variant_id) mod 5`; tocando mais de um fold, fica `trained_ineligible`
  (0,41% em 4 kb).
- **O que devolver.** Os modelos das cinco execuções do `core_locus` de 4 kb, com validation e teste de cada
  execução (o teste inclui consensus, porque a segurança usa P-BR de consensus), mais as linhas `regional` de
  `requests.parquet` pontuadas pela execução pedida. O manifesto completo pede 1.051.974 variantes (725.148 fora do
  release), das quais 623.884 do viés regional (`docs/GUIA_DE_SUBMISSAO.md` §1.4 e §2.3; plano §10.5).

## 3. O R03 visto pelo que importa para regionalizar

Fonte: `lumina-inference` (`TECHNICAL.md`, `lumina/models/model.py`, `config/lumina_r03_base.json`).

- **Modelo.** `LUM-20260719-001-R03`, passo 71.000, cerca de 52 milhões de parâmetros. Ampulheta Mamba-3
  bidirecional (SISO), janela até 32.768 bp, `d_full = 448` (`h_up` 384 + `h_pure` 64), meio da rede a 4 bp/token
  (`mid_hidden_state`, 384).
- **Cabeças por posição** sobre o tronco `last_hidden_state` [B, L, 448]: MLM (A/C/G/T), conservação (escalar e bins),
  splice (classe e distância), região, efeito contrafactual por alelo (`[B, L, 4, 8]`), severidade missense
  (destilada do ESM-2), **população** (`gnomad_af_pred` e `gnomad_observed_logits`, `[B, L, 4]`, uma saída por
  alelo alternativo), regulatória ENCODE (grade de 8 bp) e Hi-C.
- **A cabeça populacional** é `nn.Linear(448, 4)` para log-AF e outra para "observado". O plano do Mosaic registra,
  citando o repositório de treino, que o alvo é `log(AF)` do gnomAD v4.1 joint, isto é, **`AF_joint`, a frequência
  global** (§1, fato 4). Não conferi o código de treino (`lumina-research`), que não está nesta máquina.
- **Pontuação zero-shot.** LLR do MLM com a posição mascarada: `log p(alt) − log p(ref)`.
- **O que já sabemos (campanha anterior, release antigo).**
  - O canal `h_pure` é trivial (64 das 448 dimensões).
  - A melhor leitura congelada foram 172 dimensões tiradas das cabeças, candidata e não vencedora.
  - O G5 escolheu `leitura_antiga_1344`.
  - O adapter rsLoRA misto (60% gnomAD, 40% ABraOM) não moveu o clínico: Δ AUROC −0,0013 [−0,0043; +0,0018], com o
    M0 em 0,9286 contra 0,9020 do `gnomad_rarity`.
- **O que o Mosaic diz sobre o Lumina (histórico, removido em `5562720`).** O guia apagado
  `docs/HANDOFF-F2-LUMINA.md` pedia dois sistemas ao consumidor `lumina-embeddings`:
  - `lumina-r03`: um probe sobre o embedding, na "configuração da fase A", com contexto de 4.096 bp;
  - `lumina-r03-freq`: o mesmo probe com o bloco fixo de frequência do Mosaic.

  A versão anterior do plano estimava a extração em cerca de 240 mil variantes por hora por GPU ("fase B",
  `batch_size` 32). Ela sequenciava a F2 (R03 nos universos novos, mais a saída da cabeça populacional e o RW-4) e a
  F3 (R1). R2/R2c entravam **só se a F1–F2 mostrassem erros concentrados na célula primária e o R1 deixasse
  resíduo**. A configuração da fase A está no `lumina-embeddings`, que não está clonado aqui.

## 4. O que os splits recompensam (leitura para a proposta, sem propô-la ainda)

Esta seção é inferência a partir das regras acima; cada item vira uma medição antes de virar desenho.

1. **No proxy populacional e no benefício, informação do ABraOM é assimétrica por construção.** Os casos são
   definidos pela presença no ABraOM e os controles pela ausência, no mesmo bin do gnomAD. Uma feature de ABraOM pode
   melhorar o caso e não tem como mudar o controle. "Roubar" aqui é barato e circular: frequência do ABraOM como
   entrada.
2. **O número que limitaria o ganho veio do recorte antigo.** Somar ABraOM a um modelo de frequência do gnomAD não
   ajudava. O R03 não é um modelo de frequência, e o ganho que o ABraOM traria **para ele** nunca foi medido.
3. **No proxy clínico há um gap real nos especialistas** (REVEL 0,936 contra 0,973 em missense). Entender por que os
   casos com laboratório brasileiro são mais difíceis (genes, faixa de frequência, presença no ABraOM, tipo de
   evidência) é o ponto em que um ganho regional seria menos trivial.
4. **O viés só pode ser corrigido se existir no R03.** Nos especialistas ele quase não aparece, e os sistemas de
   frequência ficam em zero, porque chamam B o que é comum. O primeiro passo é medir a taxa de FP do R03 nas duas
   células. É também o diagnóstico de mecanismo do plano: a saída da cabeça populacional do R03 contra `AF_joint`,
   popmax e ABraOM.
5. **O benefício contra o RW-3 tem de vir da discriminação do modelo.** O RW-3 já aplica as regras do ABraOM, e o R1
   também. A diferença vem do PP3/BP4 do candidato nas variantes presentes no ABraOM (2.057 gold, só 98 P).
6. **A segurança é o freio de qualquer ideia que empurre variantes comuns no Brasil para benigno.** A margem é de 1
   ponto com 580 grupos de gene, praticamente sem perda permitida, e a tolerância nas críticas é zero (HbS é
   patogênica e comum).
7. **Regionalização aprendida tem regra de validade.**
   - Treino só com o ABraOM dos blocos expostos.
   - Avaliação nos não expostos.
   - Controle R2c com o mesmo orçamento.

   Uma primeira rodada que "rouba" pode ignorar isso para descobrir o que funciona, mas o resultado não serve de
   claim até ser refeito com a partição.
8. **As notas do Eduardo cabem no desenho do Mosaic.**
   - "Comparar classificador sem e com regionalização" é o par R0/R2 ou base/regionalizado, com o mesmo probe nas
     mesmas execuções do núcleo de 4 kb.
   - "Torcer o espaço" pede diagnósticos do embedding além da métrica. Exemplos: quanto um probe linear recupera da
     AF do ABraOM antes e depois; se as células primária e comparável ficam separáveis; e o quanto o embedding mudou.
   - "Fazer uma feature a partir dos painéis" é ler tudo por missense, splice e noncoding, que é como o Mosaic já
     relata.

## 5. O que falta para começar a medir

- **Dados no notebook.** O release novo (vista de 4 kb, `clinical-variants`, `variant-annotations`,
  `evaluation-panels`, `studies/brazilian-proxies`), o manifesto de pedidos, as células do viés, a verdade regional
  e o extrato do ABraOM. Os caminhos estão em `docs/GUIA_DE_SUBMISSAO.md` §1.1. Nada da campanha anterior serve como
  está: a identidade do release mudou.
- **`lumina-embeddings`.** Para reproduzir o probe de referência do Lumina ("fase A"), seria preciso ter acesso a
  esse repositório. Sem ele, usamos a nossa extração, declarando a diferença.
- **Código de treino do R03 (`lumina-research`).** Necessário para confirmar o alvo da cabeça populacional e para
  qualquer R2 que treine cabeças ou o tronco.
