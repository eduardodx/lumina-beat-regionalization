# Contrato experimental v2 — Regionalização populacional no R03 (RASCUNHO)

> **Status:** rascunho para discussão com o Eduardo. **Nada aqui está congelado.**
> Data: 2026-09-14 · Branch: `new_regionalization` · Autor: Gabriel (com Claude) · Revisor externo: consultado.
>
> **Atualização 2026-09-15:** a rota em decisão está em `docs/proposta_mosaic_regionalizacao_desenvolvimento.md`.
> Se o Eduardo aprovar, ela substitui aqui a fonte de treino (§4, §7), a decisão do conjunto de teste (§5), os
> detalhes de avaliação (§10) e a pendência 9 (§15). Já fixado pelo Gabriel: o R03 publicado
> (`best_checkpoint.pt`, passo 71.000) como checkpoint-base comum e o `croma-bioai/lumina-mosaic` como benchmark.
>
> **Legenda de cada item:**
> **[FIXADO]** decidido (por instrução do Eduardo/Gabriel ou por exigência do protocolo) ·
> **[PROPOSTO]** recomendação com evidência, aguarda aprovação ·
> **[ABERTO]** decisão necessária antes de implementar.
>
> Este contrato substitui o v1 (Fase 1 / M0, gerado por `scripts/build_contract_manifest.py`) nos pontos
> listados na §13. O protocolo de referência continua sendo o documento de regionalização do Eduardo
> (28 páginas); as seções citadas como "PDF §x" apontam para ele.

---

## 0. Pergunta e escopo

**[FIXADO] Pergunta desta campanha.** Sem adaptação clínica do backbone, um adapter populacional treinado
com ABraOM produz ganho **seletivo** na classificação de variantes associadas a submissões brasileiras,
maior do que um adapter global comparável?

Endpoint: **ΔAUROC BR-específico de M2 vs M1** (PDF §8, endpoint AUROC decidido pelo Eduardo em 29/07).

**[FIXADO] O que sai desta campanha.** O adapter ClinVar (LoRA clínico, "C") e a fusion ("F"). O motivo é
custo e maturidade — o adapter ClinVar é o componente mais caro de treinar, e a extração de embeddings e o
Mosaic ainda estão evoluindo —, **não** uma hipótese de que eles sejam desnecessários.

**[FIXADO] Escada mínima.** M0, M1 e M2. M3 (AMR) e M4 (mistura) entram depois que o pipeline estiver
funcionando, **independentemente** do resultado de M2.

**O que esta campanha não responde:** se o ganho populacional permanece quando um adapter ClinVar for
adicionado. Isso é outra campanha (§12).

---

## 1. Arquitetura

| Modelo | Backbone | Adapter populacional | Extração | Cabeça |
|---|---|---|---|---|
| M0 | R03 congelado | — | E (versão única) | H |
| M1 | R03 congelado | P_global | E | H |
| M2 | R03 congelado | P_ABraOM | E | H |
| M3 | R03 congelado | P_AMR | E | H |
| M4 | R03 congelado | P_mix (**um** adapter treinado na mistura de fontes, mesmo orçamento; sem fusion) | E | H |

- **[FIXADO] Duas etapas.** (1) Treinar o adapter populacional com os pesos originais do R03 congelados;
  (2) congelar o adapter, extrair as features e treinar **só a cabeça** com ClinVar.
- **[FIXADO] Cabeças nativas do R03 congeladas** durante o treino do adapter. O `apply_lora` já não as envolve
  com LoRA, mas os pesos delas continuam treináveis se não forem desligados explicitamente
  (`requires_grad=False`). Se uma cabeça nativa treinasse, o adapter poderia absorver a população nela sem
  mudar o trunk que a extração lê — e a própria extração passaria a diferir entre braços.
- **[PROPOSTO] Superfície do LoRA:** a mesma já verificada no R03 (exclusion-based, 105 módulos: Mamba
  `in_proj`/`out_proj` e atenção local). Declarar que o LoRA na atenção sparse/anchor é **inerte** (medido na
  Fase 0.5), o que limita onde o adapter consegue agir.
- **[ABERTO]** rank, alpha e dropout do LoRA populacional; número de passos e de tokens.

---

## 2. Objetivo de treino do adapter populacional — **[ABERTO] decisão central**

O PDF define como gerar os dados do adapter (janelas sintéticas, §4.3) mas **não define a loss**. "Gerar
sequências com variantes" não é uma função de perda. A escolha muda o que o adapter aprende:

| Opção | O que o adapter aprende | Risco |
|---|---|---|
| **A. MLM** nas janelas sintéticas (o R03 expõe `mlm_logits`; o vocabulário tem `MASK`) | a distribuição de sequência da população, sem rótulo | herda frequência se o gerador amostrar variantes por AF |
| **B. Prever AF / observed** com as cabeças populacionais nativas (`population_prior`) | frequência, diretamente | injeta na representação o sinal que as baselines diagnósticas deveriam isolar (circularidade ACMG BA1/BS1) |
| **C. Combinação** | as duas coisas | atribuição mais difícil |

**[PROPOSTO] Opção A como principal**, por ser a mais fiel a "aprender a distribuição da população sem
rótulos clínicos"; B como variante exploratória declarada.

**Especificação obrigatória, qualquer que seja a opção:**

1. Como as sequências são construídas e como variantes compatíveis são aplicadas (só SNV; sem duas variantes
   no mesmo sítio; REF conferido na coordenada declarada).
2. Quais posições são mascaradas e onde a loss é calculada.
3. **Reportar separadamente** a loss nas posições com variante aplicada e nas posições de referência. Uma
   melhora média sobre milhares de bases de referência pode esconder ausência de aprendizado nas posições
   variantes.
4. Cabeça MLM (e demais cabeças nativas) congelada.
5. Validação populacional **separada** do treino (blocos genômicos ou cromossomos fora do treino e fora do
   chr8 de teste), com um sinal de aprendizado mensurável definido antes.
6. **Não chamar as janelas de haplótipos:** os alelos são amostrados de forma independente, sem LD.
7. A receita de mascaramento do pré-treino do R03 não está no repositório de inferência: buscá-la no
   repositório de treino (`experiments/LUM-20260719-001/README.md`) ou declarar a escolha.

---

## 3. Gerador de janelas sintéticas (PDF §4.3)

- **[FIXADO]** `generate_population_window(reference_window, population_variant_table,
  frequency_sampling_policy, seed)`, com manifesto de proveniência por janela (cromossomo, início, fim, fonte,
  variantes aplicadas com posição/ref/alt/AF, seed).
- **[FIXADO]** chr8 fora de todas as janelas; alelos de variantes de teste **nunca** aplicados.
- **[FIXADO]** Orçamento idêntico entre braços: tokens, passos, seeds, distribuição de variantes por janela,
  distribuição de bins de AF. Gate: comparar os manifestos dos braços (§12).
- **[PROPOSTO]** Só SNV na campanha principal.
- **[ABERTO]** Número de variantes por janela e política de amostragem (proporcional à AF? uniforme por bin?).
- **[ABERTO]** Bins de AF que o ABraOM consegue resolver. O TSV do SABE-WGS-1171 **não traz AC nem AN**, só a
  AF; `1/(2×1.171) ≈ 4,27×10⁻⁴` só vale com todos os indivíduos chamados. AN por sítio, cobertura e filtros
  precisam da fonte original.
- **[ABERTO]** Se o contexto de referência em volta de posições de teste pode entrar em janelas (sem aplicar o
  alelo de teste).
- **[ABERTO] Definição do "global"** (PDF §3, M1): sampler hierárquico grupo → bin de AF → variante, com pesos
  iguais para AFR/AMR/EAS/NFE/SAS? O gnomAD v4.1 joint tem AF por população.

---

## 4. Fontes de dados

| Fonte | Versão / identidade | Onde | Status |
|---|---|---|---|
| Backbone | `LUM-20260719-001-R03`, `best_checkpoint.pt` (file sha256 `f2983560…`; model/resolved config `76d74157…`) | `s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/` | [FIXADO] |
| Genoma | GRCh38, `hg38.fa` (sha256 `056974f6…`) | notebook `~/hg38/hg38.fa` | [FIXADO]; confirmar coordenadas contra o GRCh38.p14 do Mosaic nos SNVs usados |
| ClinVar | 2026-06, via Mosaic `clinvar-pb-capability-suite/v1` (código `814e7f0`; `artifact_contracts_hash` `4a2077e9…`; `membership.parquet` `1c1cd65d…`) | `s3://croma-bioai-lumina-releases-us-east-2/benchmarks/mosaic/v1/` | [PROPOSTO] fonte única. O master regional antigo (lumina-ssm) vira referência histórica |
| gnomAD | v4.1 joint (genome+exome): AF global e por população, status `present` / `ac0` / `not_found` | `s3://ai4bio-lumina/data/external/gnomad-joint-v4.1/`; anotador `lumina-mosaic/src/mosaic/annotations/gnomad.py` | [PROPOSTO] |
| ABraOM | SABE-WGS-1171, sha256 `3cd33784…` (sources do Mosaic) | **localizar** (não está no notebook) | [PROPOSTO]; o índice v2 (gen-abraom-seqs) fica fora |
| Instituições brasileiras | lista revisada do Mosaic (`config/clinvar-organizations-br.yaml` + `config/submitter-name-to-org.yaml`) | repo Mosaic | [PROPOSTO], pendente do diagnóstico da §6 |

**Por que não reaproveitar as frequências antigas:** a auditoria (`scripts/audit_regionalization_data.py`,
commit `92304ab`) confirmou que o `af_gnomad` do pipeline regional só existe onde a variante está no índice
ABraOM (0 exceções), e que o join com o ABraOM só é tentado para SNV. Os "91% do br_only ausentes do gnomAD"
eram 3.696 de 4.066 variantes **fora do índice ABraOM**.

---

## 5. Conjuntos de teste — **[ABERTO] decisão (0)**

**Opção A — T_BR v2 no desenho do PDF.** BR-only; pareamento 1:1 exato em gene, rótulo e tipo, com AF global
(real), estrelas e número de submitters na distância e consequência como preferência. Reconstruído do ClinVar
2026-06 inteiro, com a lista revisada de instituições e gnomAD real.

**Opção B — track `brazil` do Mosaic.** Dois estudos só de avaliação, com pares já materializados (estrato
exato rótulo | painel | bin de AF do gnomAD real). O contrato do Mosaic **proíbe** usar membros, controles e
rótulos em treino, seleção ou calibração.

**Opção C — as duas, com papéis declarados.**

| | A: T_BR v2 (PDF) | B: `br_clinical_evidence` | B: `br_population_observed` |
|---|---|---|---|
| Construto | só submissor brasileiro | participação de instituição brasileira (a maioria compartilhada, segundo o Mosaic) | presença no ABraOM |
| Tiers | todos os rótulos P/B; **na prática, 1 estrela** (§5.1) | consensus | gold |
| Tamanho | só brasileira com rótulo gold/consensus: **63 (62 P / 1 B)**; com 1 estrela: a medir (o v1 pareado SNV tinha 2.774 P / 529 B, com a marcação v1) | 3.119 casos: 2.808 P / 311 B | 1.889 casos: 89 P / 1.800 B; só 751 pareados |
| Pareamento | gene, tipo, consequência + AF | rótulo, painel, bin gnomAD | rótulo, painel, bin gnomAD |
| Independência da extração | no v1, 280 BR + 226 nonBR pareados estavam no gold usado na seleção | **intocado** (consensus) | **inteiro dentro** da seleção (gold); sobreposto ao ABraOM por construção |
| Formato | M0–M2 com M1 como baseline causal | base × regionalizado: declarar como M1 e M2 entram no manifesto do consumidor | idem |
| Trabalho | reconstrução completa | pronto; excluir membros do treino | pronto; excluir membros do treino |

### 5.1 O que o diagnóstico de 14/09 mudou

A proposta anterior (T_BR v2 só brasileira como principal, track `brazil` complementar) dependia de haver
variantes só brasileiras em número suficiente. O diagnóstico (§6) mediu que isso não acontece com rótulos de
qualidade:

- **Com os rótulos do Mosaic (gold + consensus) e o filtro de SCV dele, só 63 variantes são só brasileiras: 62 P
  e 1 B**, todas consensus. As que têm participação brasileira somam 3.119 (2.808 P / 311 B), das quais 3.056 são
  compartilhadas. Nenhuma variante gold tem marcação brasileira.
- **É estrutural.** Consensus exige pelo menos dois submissores sem conflito, e o agregado de uma variante gold vem
  do painel de especialistas. Uma variante só brasileira costuma ter um único submissor (1 estrela), tier que o
  Mosaic exclui por qualidade de rótulo.
- **O T_BR v1 confirma.** Dos 1.782 membros BR pareados que estão no Mosaic, 929 são compartilhados, 832 não têm
  marcação brasileira pelo filtro e só 21 são só brasileiros. Dos 1.688 SNVs do slice T_BR v1 fora do Mosaic,
  1.446 (86%) tinham `max_review_status_rank_aggregate` = 1 no release da v1 (173 com 2, 32 com 3, 37 sem valor).
- **Hipótese, não medida:** a variante só brasileira de submissor único é onde um laboratório brasileiro mais
  provavelmente usou frequência local (ABraOM) para chamar benigno. É onde o sinal regional esperado é maior e
  também onde a circularidade é maior.

Revisar a proposta agora é legítimo: nenhum modelo rodou, e tamanho e qualidade de rótulo não são resultado.

**[ABERTO] Decisão (0), reformulada: construto e qualidade de rótulo juntos.**

- **A′ — só brasileira com rótulos de 1 estrela** (construto do PDF). Exige declarar a política de rótulo (P/B de
  submissor único), parear também por estrelas (PDF §6.3), prever um controle de circularidade (por exemplo, a
  DiD estratificada por presença no ABraOM) e medir o tamanho no ClinVar 2026-06 antes de decidir.
- **B — participação (`br_clinical_evidence`).** Pronto, rótulo consensus, pares materializados, 311 benignas.
  Mede outro construto: a maioria das variantes é compartilhada com laboratórios internacionais, o que pode diluir
  um efeito regional.
- **C — as duas**, com a principal escolhida antes de qualquer modelo, por tamanho e qualidade de rótulo.

**[PROPOSTO] Próximo passo:** medir A′ (pendência 9). Se A′ não tiver o poder mínimo, a principal é B e A′ fica
exploratória. Se tiver, a escolha entre A′ (construto do PDF, rótulo mais fraco, circularidade maior) e B (rótulo
melhor, sinal possivelmente diluído) é do Eduardo. Condições que valem para qualquer escolha:

1. **Tamanho e poder:** recalcular o poder do ΔAUROC com os tamanhos reais (método da
   `docs/justificativa_endpoint_auroc.md`). Depois de declarada a principal, não se troca o teste por outro que
   "funcione melhor".
2. **Independência:** a extração foi escolhida no gold do Mosaic, que no v1 continha 280 BR + 226 nonBR
   pareados. Reportar o contraste principal também **sem** as variantes de teste que estavam nesse gold
   (análise de sensibilidade pré-declarada).
3. `br_population_observed` continua exploratório (sobreposto ao ABraOM por construção).
4. Construir os candidatos na etapa de dados e excluir a **união** deles dos splits (§7): a decisão só atribui
   papéis e não obriga a refazer os splits.

Benchmarks complementares mantidos: chr8 e BRCA1/BRCA2/TP53 (PDF §11–12; descritivos abaixo dos mínimos).

---

## 6. Definição de submissor brasileiro — **[ABERTO]**, com diagnóstico feito em 14/09

A auditoria (bloco [F]) mostrou que as duas definições em uso discordam:

- 1.623 casos do `br_clinical_evidence` estão nos nossos splits "não-BR" v1 (1.250 no treino);
- 46 estão no nosso T_nonBR v1;
- 20 controles não brasileiros do Mosaic estão no nosso T_BR v1;
- só 1.022 dos 3.116 casos clínicos pareados do Mosaic estão no nosso T_BR v1 (parte da diferença é esperada:
  BR-only exclui variantes compartilhadas por desenho).

| Definição | Origem |
|---|---|
| Mosaic `br_lab_any` | SCV do ClinVar 2026-06 cujo submissor casa com a lista **revisada** de instituições; só SCVs que passam no filtro de classificação P/B |
| Pipeline regional `has_brazilian_submitter` | coluna `cohort` do `eval_unified.parquet` (lumina-benchmarks), **sem revisão**, release desconhecido |

**Como cada lado marca (conferido no código em 14/09):**

- **Mosaic** (`labels.py`, `orgs.py`, `examples.py`): uma SCV é brasileira se o submissor casa, por chave
  canônica e sem ambiguidade, com uma instituição `include: true` da lista (60 instituições, congelada em
  2026-08-22 a partir do `organization_summary`) ou com um alias revisado. Só contam SCVs que contribuem para o
  agregado, têm origem germinativa e classificação P/B. A agregação usa todas as `VariationID` da variante:
  `br_lab_any` = alguma SCV brasileira; `br_lab_only` = todas; `br_lab_shared` = alguma, mas não todas.
- **Limite do Mosaic:** o matcher só indexa instituições incluídas. Submissor não reconhecido e instituição
  comprovadamente não brasileira recebem o mesmo `org_id` nulo, e o build só falha para não reconhecidos com
  ≥ 50 SCVs cujo nome contenha um termo-gatilho ou compartilhe palavras com as instituições incluídas.
- **Pipeline regional v1:** `has_brazilian_submitter` = alguma linha do `eval_unified.parquet` com
  `cohort == brazilian`, sem filtro de classe no agrupamento (não verificado se a tabela já vem filtrada). Nos
  splits v1, variante **sem linha** na tabela regional virou `False` (`fillna(False)`): "sem dado" contou como
  não brasileira.

**Leitura do [F]:** o pool de treino v1 já excluía toda variante com `has_brazilian_submitter`, inclusive as
"mixed". Os 1.250 casos do Mosaic no treino v1 **não** se explicam pela diferença any × only: pela nossa
definição, nenhuma submissão brasileira foi vista neles.

**Diagnóstico executado em 14/09** (`scripts/diagnose_brazilian_submitter_divergence.py`, commit `3419bea`;
saídas em `~/artifacts/redesenho/diagnostico_submissor_br/` no notebook). O `is_br` do Mosaic foi refeito com as
funções e a configuração do próprio Mosaic (`814e7f0`) e o `submission_summary_2026-06` conferido por sha256, e
**bateu com as flags publicadas em 326.826 de 326.826 exemplos**. Isso confirma compatibilidade com as flags, não a
correção geográfica das instituições. As categorias descrevem **marcações**, não nacionalidade.

| Categoria | Variantes | Leitura |
|---|---:|---|
| A1: Mosaic marca; a v1 não tem linha da variante | 1.580 (1.187 no treino v1) | cobertura: só 9.602 dos 190.005 exemplos do Mosaic presentes no master v1 (5%) tinham linha na tabela regional |
| A2: Mosaic marca; a v1 tem linhas, nenhuma brasileira | 133 | a submissão brasileira (Dasa em 101) não estava na tabela v1; as listas v1 mais comuns só têm laboratórios internacionais |
| B: a v1 marca; o Mosaic não | 1.134 | todas com SCV de instituição da lista fora do filtro do Mosaic (abaixo) |
| C1: a v1 sem linha não brasileira; Mosaic compartilhada | 1.026 | as submissões internacionais (Labcorp/Invitae, GeneDx, Ambry) não estavam na tabela v1 |
| C2: a v1 com linha não brasileira; Mosaic só marcadas | 6 | — |
| Os dois marcam, com a mesma divisão | 206 | — |

- **Onde a v1 tinha dado, ela concordou:** marcou brasileira 1.238 das 1.371 variantes cobertas que o Mosaic marca.
  Não apareceu erro de mapeamento de instituição. "Não brasileira" na v1 era, quase sempre, "sem dado".
- **Por que o Mosaic não conta as SCVs brasileiras de B** (1.336 SCVs): não é classificação (só 6 não são P/B).
  959 têm origem não germinativa (921 `unknown`; Mendelics em 924 das suas 980) e 630 não contribuem para o agregado
  (entre elas, Dasa 124, INCA 55, Einstein 52 e A.C.Camargo 33); 253 têm as duas coisas.
- **Escopo da lista do Mosaic:** nenhuma SCV P/B de instituição com país Brazil no NCBI fora da lista, em todos os
  exemplos; mas 138.306 das 932.402 SCVs P/B (15%) têm país não resolvido. Nas variantes divergentes, os 96
  submissores com país não resolvido não contêm nenhum termo-gatilho do Mosaic (os maiores são OMIM, ARUP,
  GeneReviews, SickKids e ENIGMA). A lista não fica validada de forma completa.
- **Pontos da política do Mosaic a confirmar com o Eduardo:**
  (a) `origin_has_germline` só aceita o termo literal `germline`: `de novo`, `biparental`, `maternal`, `inherited` e
  `unknown` ficam fora (conferido no código; não encontrei documentação dessa escolha);
  (b) nenhuma variante gold recebe marcação brasileira; explicação provável: quando há painel de especialistas, o
  ClinVar marca as demais SCVs como não contribuindo para o agregado, e o filtro exige contribuição;
  (c) a URL do `submission_summary_2026-06` em `config/sources.yaml` aponta para `archive/2026/`, que não existe
  (os bytes batem com o arquivo em `archive/`).

**[PROPOSTO] Definição para a v2, com dois usos:**

1. **Exclusão de treino, validação e calibração (vazamento): ampla.** Qualquer SCV de instituição da lista, de
   qualquer classificação, origem ou contribuição, tira a variante do treino. O custo é perder variantes do treino;
   o erro contrário é vazamento. Cobre também o que a v1 contava.
2. **Pertencimento ao teste: segue o construto da §5.** Para participação (B), vale o filtro do Mosaic, já
   materializado nos pares. Para só brasileira com 1 estrela (A′), declarar o filtro antes; proposta: P/B e não
   exclusivamente somática (a origem `unknown` conta, como no PDF §5.5), com sensibilidade usando o filtro do Mosaic.

A pasta `interim/` do Mosaic deixou de ser necessária para o diagnóstico: a reprodução bateu.

**[FIXADO] Qualquer que seja a definição:** submissões brasileiras ficam fora de treino, validação e calibração
(PDF §4.2).

---

## 7. Splits de treino, validação e calibração — [PROPOSTO]

- Fonte: ClinVar 2026-06, rótulos P/LP × B/LB, variante canônica GRCh38.
- Status brasileiro calculado para **toda** variante a partir do `submission_summary` completo, com a definição
  resolvida na §6. Ausência de dado não conta como não brasileira (no v1, contava).
- Excluir: variantes com qualquer SCV de instituição brasileira da lista, de qualquer classificação, origem ou
  contribuição (§6, uso 1); chr8; **todos** os membros dos candidatos a teste (T_BR v2, T_nonBR v2 e o track
  `brazil` do Mosaic, com controles e variantes do mesmo `overlap_cluster_id`).
- Só SNV na campanha principal. A auditoria [D] mostrou que o custo é pequeno: nos pares v1, SNV fica com
  2.774 P / 529 B de 3.104 P / 547 B (perde 18 benignas); os pares de indel tinham só 18 benignas.
- 80/10/10 por variante canônica ([ABERTO]: ou por gene), manifestos sha256 e gate de sobreposição zero,
  reaproveitando `scripts/build_clinvar_splits.py`.
- Referência histórica v1: 839.310 / 104.734 / 104.633 (sha256 `02aeb8ba…` / `d7f7bd73…` / `f6a172e5…`).
  **Não reutilizar:** usam a marcação v1 de submissor brasileiro; o treino v1 contém 1.252 variantes marcadas
  pelo Mosaic (1.187 delas sem nenhuma linha na tabela regional).

---

## 8. Extração de embeddings — [PROPOSTO]

- **Candidata:** 172 dimensões lidas das cabeças do R03 — `heads_lin` (68, W·Δ pós-norma, exato e sem forward
  extra), `heads_mlp` (10, Δ das três cabeças MLP), `heads_ref` (78, valores na referência) e `subst` (16,
  one-hot de base e substituição) —, janela de 4 kb, convenção focal do Mosaic (`L//2 − 1`), validação
  **estrita** de REF (sem o fallback ±1 do harness antigo).
- **Não é vencedora estabelecida.** Na pesquisa: (a) o probe MLP escolhia a época pela AUROC conjunta da
  validação e o ridge pela macro dos painéis. Corrigido (`1615955`) e rerodado em 14/09: a macro mudou 0,002
  em média e a ordem das configurações se manteve (correlação de postos ≥ 0,995), calculado das tabelas
  impressas (só a célula mlp/gene foi conferida nos JSONs). Somadas à base honesta, as cabeças continuam
  acrescentando no missense (+0,033 core / +0,037 gene); nessa comparação, a diferença para o embedding de 2092
  dims no missense foi de +0,006/+0,013 para −0,0004/−0,0002: valores muito próximos, **sem incerteza
  quantificada** e sem demonstrar equivalência (isoladas, as 2092 dims ficam à frente no missense: +0,027 core /
  +0,042 gene; e nada foi medido depois da adaptação populacional). As 172 dims perdem menos em noncoding que o
  v2 nessas avaliações (−0,003/−0,025 contra −0,018/−0,053), ainda com perda e sem causa demonstrada, e são
  12× menores; (b) a premissa "as cabeças estão no span, então um probe linear não ganha" não se
  aplicava à comparação feita (`delta_focal` é h_up **pré-norma**; as cabeças leem o trunk **pós-norma**);
  (c) as nove diferenças positivas vêm de avaliações correlacionadas, sem controle de multiplicidade.
- **Seleção** em dados de desenvolvimento (train/validation), **nunca** nos testes, com critério declarado
  antes. Comparar com a extração antiga (two-tower pós-norma + média de ±64 bp): no MLP da pesquisa, a aproximação
  dela sem LoRA (`infra_atual_pos_norma`, 896 dims) fica +0,0015 acima das 172 dims na macro e +0,015 no
  missense, sem incerteza quantificada. Rodar um piloto com um adapter
  para checar se a leitura sobrevive à adaptação, e escolher com um critério que não favoreça o M0.
- **Congelar UMA extração** para todos os braços (extração diferente por braço confundiria fonte com
  representação).
- **[PROPOSTO] Busca encerrada no benchmark da pesquisa:** nenhuma configuração nova de extração no gold do
  Mosaic. A única comparação que resta é a candidata compacta × a leitura antiga completa, no desenvolvimento,
  depois do piloto (§12, etapa 5).
- **Identidade do cache:** checkpoint + adapter + versão do extrator (janela, focal, RC, validação de REF) +
  chaves das variantes. Mudar o Mosaic (rótulos, splits) permite reaproveitar embeddings; mudar backbone,
  adapter ou janela exige extrair de novo.

---

## 9. Cabeça, treino e calibração

- **[ABERTO]** Arquitetura da cabeça (proposta: MLP pequena sobre features padronizadas), escolhida no
  desenvolvimento e **idêntica** em todos os braços.
- **[FIXADO]** Mesmos exemplos, mesma ordem por seed, mesmo orçamento; early stopping no validation.
- **[FIXADO]** Platt scaling no calibration; limiar de MCC escolhido no calibration e congelado; o **mesmo**
  limiar em BR e nonBR (PDF §9).
- **[FIXADO]** Três seeds na etapa confirmatória, cobrindo adapter e cabeça; predição final = média das
  probabilidades calibradas; métricas também por seed (PDF §10).

---

## 10. Avaliação pré-definida (feita uma vez, depois de congelar tudo)

- **[FIXADO] Contraste principal:** M2 vs M1 em ΔAUROC BR-específico.
- **[FIXADO] Reportar também os ganhos absolutos em BR e em nonBR.** Uma interação positiva pode vir de o
  modelo ter piorado no controle.
- **Secundárias:** AP, MCC, Brier, com TP/TN/FP/FN, sensibilidade e especificidade.
- **Intervalos:** bootstrap pareado (por matched set; por `overlap_cluster_id` no Mosaic), 10.000 réplicas;
  análise de sensibilidade à dependência por gene.
- **[PROPOSTO] Critério de sucesso** (escala AUROC), proposto por nós em `docs/decisoes_eduardo_fase0.md` C1 e
  ainda sem confirmação do Eduardo. A redação anterior ("ganho ≥ 0,02") não dizia **qual** ganho; declarar
  separadamente:
  1. **Interação:** ΔAUROC BR-específico (M2 vs M1) ≥ 0,02, IC 95% excluindo zero, mesma direção em ≥ 2 de 3
     seeds. O 0,02 foi pensado nesta escala: o prior de ~0,027 é uma diferença-em-diferenças.
  2. **Ganho no conjunto brasileiro:** G_BR = AUROC(M2, T_BR) − AUROC(M1, T_BR) > 0, reportado com IC. Sem ele,
     uma interação positiva pode vir só de piora no controle.
  3. **Guardrails, em T_BR e em T_nonBR:** AP e AUROC não regridem mais de 0,02; Brier não piora mais de 0,01;
     nenhuma classe colapsa.
- **[FIXADO] Baselines diagnósticas separadas:** regra de presença no ABraOM e AF explícita (gnomAD, ABraOM).
  Elas medem quanto uma regra populacional simples já explica — e é por elas que se enxerga a circularidade.
- **[FIXADO]** Enquanto extração e Mosaic mudarem, resultados contam como **desenvolvimento**. Cada comparação
  formal congela uma versão.
- Interpretação pelos cenários A–I do PDF §15.

---

## 11. chr8

- **Representacional:** score populacional (probabilidade do alelo alternativo na MLM, se a opção A for
  escolhida, ou a cabeça populacional) × log10(AF), Spearman, em variantes chr8 nunca vistas (ABraOM-chr8,
  AMR-chr8, GLOBAL-chr8).
- **[ABERTO]** Se o pré-treino do R03 viu o chr8. Isso define se o nome é "adapter-level" ou
  "foundation-and-adapter chromosome holdout" (PDF §11). O `TECHNICAL.md` do R03 só lista o chr8 como família
  de métrica de seleção.
- **Interação a declarar:** o R03 já tem cabeças populacionais treinadas. Esse conhecimento prévio é o mesmo em
  todos os braços, mas pode favorecer algumas fontes, reduzir ganhos adicionais ou interagir com os adapters —
  afeta a interpretação e, potencialmente, o resultado.

---

## 12. Ordem de execução e gates

**[PROPOSTO] Ordem** (revisão externa de 14/09):

| Etapa | Trabalho | Critério para avançar |
|---|---|---|
| 1. Divergência brasileira | diagnóstico da §6 | entender quais diferenças são de definição, cobertura, versão ou erro |
| 2. Dados v2 | gnomAD independente do ABraOM, fonte ABraOM verificada, REF estrito, BR-only / shared / nonBR, candidatos a teste e exclusões (§5, §7) | contagens, cobertura e sobreposições verificadas |
| 3. Desenho mínimo | teste principal, objetivo populacional, gerador, sampler global, orçamento, critério de sucesso | contrato executável, sem decisões implícitas |
| 4. Piloto pequeno de M1/M2 | gradientes, aprendizado populacional (§2) e compatibilidade da extração | evidência de que os adapters aprendem a tarefa definida |
| 5. Extração congelada e campanha | candidata compacta × leitura antiga completa (§8); depois M0, M1, M2 | mesma receita e protocolo entre braços |

Não dependem do Eduardo: a etapa 1, localizar e inspecionar o ABraOM, verificar mapeamentos e preparar código.
Teste principal, loss e critério de sucesso vão a ele como propostas concretas.

**Gates antes do treino completo:**

1. Só os parâmetros previstos recebem gradiente (log de parâmetros treináveis; o smoke do M0 antigo pegou
   7,5M de parâmetros do backbone treináveis sem querer).
2. O adapter muda a representação de forma mensurável e aprende a loss definida (piloto, §2).
3. Nenhuma variante de teste nem chr8 nas janelas (manifestos de proveniência).
4. Orçamento e distribuições comparáveis entre braços.
5. Sobreposição zero entre splits e testes, incluindo membros do Mosaic se o track for usado.

**Quando o adapter ClinVar voltar:** a modularidade facilita o encaixe técnico, mas não garante compatibilidade
científica. LoRAs modificam o backbone, e somar o ClinVar pode alterar a representação populacional. Será
preciso definir a regra de combinação (sequencial, merge, fusion) e repetir a avaliação. O C antigo
(`edf99295…`) foi treinado com a extração e os dados v1 e **não é plugável** neste desenho.

---

## 13. O que muda em relação ao contrato v1

| Item | v1 | v2 | Por quê |
|---|---|---|---|
| Adapter ClinVar | LoRA C (`edf99295…`) | removido | custo e maturidade |
| Fusion | mistura de LoRAs dentro do backbone | removida | idem |
| Extração | two-tower pós-norma + média ±64 bp → `RegimeAHead` | candidata de 172 dims, validada no desenvolvimento | pesquisa de extração |
| Frequência gnomAD | condicional ao índice ABraOM | gnomAD v4.1 real com status | auditoria [A] |
| Pareamento T_nonBR | usava a AF condicional | reconstruir ou usar o Mosaic | auditoria [B] (excesso descritivo de +0,126 nos estratos mistos) |
| Submissor brasileiro | coluna `cohort` sem revisão, com linha para só 5% das variantes | lista revisada do Mosaic, por SCV; exclusão ampla do treino | diagnóstico de 14/09 (§6) |
| Domínio | SNV + indel + MNV | só SNV na campanha principal | pesquisa + auditoria [D] |
| Validação de REF | fallback ±1 | estrita | revisão |
| Fonte ClinVar | master regional (release desconhecido) | ClinVar 2026-06 via Mosaic | proposta |
| Teste principal | T_BR v1 (BR-only pela coluna `cohort`) | a decidir: participação (Mosaic, consensus) ou só brasileira com rótulos de 1 estrela | §5.1: com rótulo gold/consensus, só brasileira tem 1 benigna |

---

## 14. O que **não** será afirmado

- Melhora em pacientes brasileiros, na população brasileira nacional ou na prática clínica brasileira.
- Validação da arquitetura futura com adapter ClinVar.
- `br_population_observed` como generalização independente (a sobreposição com o ABraOM é real por construção).

---

## 15. Pendências

| # | Pendência | Quem | Bloqueia |
|---|---|---|---|
| 1 | ~~Diagnóstico da divergência de submissor brasileiro~~ feito em 14/09 (§6) | Claude, Gabriel | — |
| 2 | Decisão (0): construto do teste e qualidade de rótulo (§5.1) | Eduardo | §5, §7 |
| 3 | Loss do adapter populacional e gerador de janelas | Eduardo | §2, §3 |
| 4 | Definição do "global", bins de AF e AN do ABraOM | Eduardo, Gabriel | §3 |
| 5 | Critério de sucesso: interação e ganho em T_BR declarados separadamente | Eduardo | §10 |
| 6 | O pré-treino do R03 viu o chr8? | Eduardo (repositório de treino) | §11 |
| 7 | Localizar o TSV do SABE-WGS-1171 | Gabriel, Eduardo | §3, §4 |
| 8 | ~~Rerodar os probes MLP com o critério macro e comparar~~ feito em 14/09 (resultado no §8) | Gabriel (notebook) | — |
| 9 | Medir o tamanho de só brasileira com rótulos de 1 estrela no ClinVar 2026-06 (precisa do `variant_summary_2026-06`, 439 MB) | Claude (script), Gabriel (notebook) | §5 |
| 10 | Filtro de SCV para pertencimento ao teste (§6, uso 2) | Eduardo | §5, §6 |
| 11 | Confirmar com o Eduardo os pontos da política do Mosaic (origens germinativas, contribuição em gold, URL no `sources.yaml`) | Gabriel, Eduardo | §6 |
