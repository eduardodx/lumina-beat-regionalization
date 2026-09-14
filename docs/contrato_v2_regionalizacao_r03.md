# Contrato experimental v2 — Regionalização populacional no R03 (RASCUNHO)

> **Status:** rascunho para discussão com o Eduardo. **Nada aqui está congelado.**
> Data: 2026-09-14 · Branch: `new_regionalization` · Autor: Gabriel (com Claude) · Revisor externo: consultado.
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
| Construto | só submissor brasileiro | participação de instituição brasileira | presença no ABraOM |
| Tiers | todos os rótulos P/B | consensus | gold |
| Tamanho | reconstruir (v1 pareado SNV: 2.774 P / 529 B) | 3.119 casos: 2.808 P / 311 B | 1.889 casos: 89 P / 1.800 B; só 751 pareados |
| Pareamento | gene, tipo, consequência + AF | rótulo, painel, bin gnomAD | rótulo, painel, bin gnomAD |
| Independência da extração | no v1, 280 BR + 226 nonBR pareados estavam no gold usado na seleção | **intocado** (consensus) | **inteiro dentro** da seleção (gold); sobreposto ao ABraOM por construção |
| Formato | M0–M2 com M1 como baseline causal | base × regionalizado: declarar como M1 e M2 entram no manifesto do consumidor | idem |
| Trabalho | reconstrução completa | pronto; excluir membros do treino | pronto; excluir membros do treino |

**[PROPOSTO] Opção C:** `br_clinical_evidence` como teste **confirmatório**; `br_population_observed` como
**exploratório**; T_BR v2 BR-only como contraste secundário, se o Eduardo quiser o construto do PDF. Checar o
poder com 311 benignas antes de congelar.

Benchmarks complementares mantidos: chr8 e BRCA1/BRCA2/TP53 (PDF §11–12; descritivos abaixo dos mínimos).

---

## 6. Definição de submissor brasileiro — **[ABERTO] até o diagnóstico**

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

**Causas candidatas, ainda não separadas:** diferença de release, mapeamento de instituições, ou quais classes
de SCV contam (uma VUS submetida por laboratório brasileiro não conta no Mosaic).

**Diagnóstico:** a pasta `interim/` do passo `build-labels` do Mosaic (`assertions.parquet` com `submitter`,
`org_id` e `is_br` por SCV; `org_match.json`; `org_unresolved.json`). Não está publicada no S3. **Plano B:**
reproduzir o `is_br` com o código do Mosaic a partir do `submission_summary_2026-06`, só para as variantes
divergentes.

**[FIXADO] Qualquer que seja a definição:** submissões brasileiras ficam fora de treino, validação e calibração
(PDF §4.2).

---

## 7. Splits de treino, validação e calibração — [PROPOSTO]

- Fonte: ClinVar 2026-06, rótulos P/LP × B/LB, variante canônica GRCh38.
- Excluir: variantes com submissão brasileira (§6); chr8; **todos** os membros de teste; se o track do Mosaic
  for usado, também os controles e as variantes do mesmo `overlap_cluster_id`.
- Só SNV na campanha principal. A auditoria [D] mostrou que o custo é pequeno: nos pares v1, SNV fica com
  2.774 P / 529 B de 3.104 P / 547 B (perde 18 benignas); os pares de indel tinham só 18 benignas.
- 80/10/10 por variante canônica ([ABERTO]: ou por gene), manifestos sha256 e gate de sobreposição zero,
  reaproveitando `scripts/build_clinvar_splits.py`.
- Referência histórica v1: 839.310 / 104.734 / 104.633 (sha256 `02aeb8ba…` / `d7f7bd73…` / `f6a172e5…`).
  **Não reutilizar:** contêm membros do track brazil e usam a definição antiga de submissor brasileiro.

---

## 8. Extração de embeddings — [PROPOSTO]

- **Candidata:** 172 dimensões lidas das cabeças do R03 — `heads_lin` (68, W·Δ pós-norma, exato e sem forward
  extra), `heads_mlp` (10, Δ das três cabeças MLP), `heads_ref` (78, valores na referência) e `subst` (16,
  one-hot de base e substituição) —, janela de 4 kb, convenção focal do Mosaic (`L//2 − 1`), validação
  **estrita** de REF (sem o fallback ±1 do harness antigo).
- **Não é vencedora estabelecida.** Na pesquisa: (a) o probe MLP escolhia a época pela AUROC conjunta da
  validação e o ridge pela macro dos painéis — critérios diferentes, correção na branch
  `embedding-probe-mosaic`; (b) a premissa "as cabeças estão no span, então um probe linear não ganha" não se
  aplicava à comparação feita (`delta_focal` é h_up **pré-norma**; as cabeças leem o trunk **pós-norma**);
  (c) as nove diferenças positivas vêm de avaliações correlacionadas, sem controle de multiplicidade.
- **Seleção** em dados de desenvolvimento (train/validation), **nunca** nos testes, com critério declarado
  antes. Comparar com a extração antiga (two-tower pós-norma + média de ±64 bp). Rodar um piloto com um adapter
  para checar se a leitura sobrevive à adaptação, e escolher com um critério que não favoreça o M0.
- **Congelar UMA extração** para todos os braços (extração diferente por braço confundiria fonte com
  representação).
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
- **[PROPOSTO] Critério de sucesso** (escala AUROC, `docs/decisoes_eduardo_fase0.md` C1): ΔAUROC BR-específico
  > 0, IC 95% excluindo zero, ganho ≥ 0,02, mesma direção em ≥ 2 de 3 seeds. Guardrails: AP e AUROC não
  regridem mais de 0,02; Brier não piora mais de 0,01; nenhuma classe colapsa.
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

## 12. Gates antes do treino completo

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
| Submissor brasileiro | coluna `cohort` sem revisão | lista revisada (pendente de diagnóstico) | auditoria [F] |
| Domínio | SNV + indel + MNV | só SNV na campanha principal | pesquisa + auditoria [D] |
| Validação de REF | fallback ±1 | estrita | revisão |
| Fonte ClinVar | master regional (release desconhecido) | ClinVar 2026-06 via Mosaic | proposta |

---

## 14. O que **não** será afirmado

- Melhora em pacientes brasileiros, na população brasileira nacional ou na prática clínica brasileira.
- Validação da arquitetura futura com adapter ClinVar.
- `br_population_observed` como generalização independente (a sobreposição com o ABraOM é real por construção).

---

## 15. Pendências

| # | Pendência | Quem | Bloqueia |
|---|---|---|---|
| 1 | Diagnóstico da divergência de submissor brasileiro (pasta `interim/` do Mosaic) | Eduardo (arquivos), Claude (script) | §5, §6, §7 |
| 2 | Decisão (0): conjunto de teste | Eduardo | §5, §7 |
| 3 | Loss do adapter populacional e gerador de janelas | Eduardo | §2, §3 |
| 4 | Definição do "global", bins de AF e AN do ABraOM | Eduardo, Gabriel | §3 |
| 5 | Critério de sucesso em AUROC | Eduardo | §10 |
| 6 | O pré-treino do R03 viu o chr8? | Eduardo (repositório de treino) | §11 |
| 7 | Localizar o TSV do SABE-WGS-1171 | Gabriel, Eduardo | §3, §4 |
| 8 | Rerodar os probes MLP com o critério macro e comparar | Gabriel (notebook) | §8 |
