# Resultados — Regionalização do Beat-v11, Fase 1 (adapters de frequência ABRAOM)

**Modelo:** Beat-v11 BioPrime r1 (`lumina-beat-v11v5-r1-202607071631`, SISO, 52M, d_full=448) · **Base a bater:** os adapters de frequência do Pedro no **Beat-v10** · **Tarefa:** predizer frequência alélica a partir da sequência (adapter LoRA sobre o backbone congelado) · **Métrica primária honesta:** **Spearman** de `pred` vs `af_abraom` no test (rank, robusto à escala). · **Janela:** 1024 bp (paridade) + ablação 4096.

> Este documento fecha a **Fase 1** do porte da regionalização ABRAOM do v10 para o v11: os três adapters de frequência (A_BR, A_gnomAD, A_scrambled) treinados sobre o backbone v11 com a **receita exata do Pedro**, e a pergunta que eles respondem — **existe sinal regional (brasileiro-específico) de sequência, ou só sinal de frequência genérico?**

---

## 1. Principais conclusões (TL;DR)

1. **O pipeline do v10 reproduz no v11** — a Fase 0 (integração) foi validada e os três adapters treinaram com a receita do Pedro (ctx 1024, 1000 steps, 100k rows balanced-af, `use_gnomad_prior=False` = sequence-only).
2. **Há sinal de sequência→frequência REAL no v11.** A_BR e A_gnomAD batem o controle embaralhado (A_scrambled) com folga esmagadora: `A_BR − A_scrambled` = **+0.17 / +0.15** (val/test), **CI 95% exclui 0, P=1.000**. O adapter aprende mesmo a mapear sequência→AF.
3. **Mas NÃO há sinal regional-específico.** A_BR (treina `af_abraom`) **não** supera A_gnomAD (treina `af_gnomad`, mede `af_abraom`): `A_BR − A_gnomAD` = **+0.016 / +0.010**, **CI cruza 0** (P=0.93 / 0.74). O modelo aprende "frequência genérica", não "brasilidade".
4. **O v11 é melhor que o v10 — mas genericamente.** A_BR sobe Spearman **0.104→0.127** (test; +~20%). É um backbone melhor aprendendo AF melhor, não um sinal regional emergindo.
5. **Contexto longo não ajuda a predizer AF** — e é de graça. ctx 4096 ≈ ctx 1024 (test 0.1273 vs 0.1274), a custo ~zero (workload overhead-bound). O maquinário de longo alcance do v11 não carrega sinal de frequência (biologicamente plausível: AF é local — constraint/conservação/mutabilidade).
6. **Isto reproduz e explica o caveat do Pedro.** O sinal regional nunca esteve forte no adapter, desde a origem — o que explica upstream por que os controles negativos estratificados dele chegavam tão perto do real lá no downstream (o ganho do M5_v3 é **calibração de frequência**, não biologia brasileira aprendida da sequência).

---

## 2. Metodologia (resumo)

- **Quatro adapters de frequência**, todos LoRA (rank 8, α 16) sobre o **backbone v11 congelado** (105 lineares LoRA-ados: mamba `in/out_proj` + attn `q/k/v/out_proj` + `up_stages.gate` + `stem.purity`; **zero cabeças**, MoE congelada — no r1 a MoE está OFF de qualquer forma). Head: `LayerNorm→Linear(d_full·3→256)→GELU→Linear(→1)`.
- **Receita do Pedro (verificada nos `summary.json` dele, não nos defaults do trainer):** ctx 1024, `max_steps=1000`, 100k train / 5k val / 5k test rows, `row_sample_strategy=balanced-af`, batch 2, grad-accum 8, lr_lora 5e-6 / lr_head 5e-4, seed 42, **`use_gnomad_prior=False`** (sequence-only — o adapter NÃO recebe a AF do gnomAD como feature).
- **Os quatro alvos:**
  - **A_BR** — `--target-column af_abraom` (o adapter regional).
  - **A_gnomAD** — `--target-column af_gnomad --metric-target-column af_abraom` (treina na AF global, avaliado na brasileira). **O comparador científico:** como `af_gnomad ≈ af_abraom`, ele é um baseline forte de "frequência genérica". A_BR só tem sinal regional se **superar** este.
  - **A_scrambled** — `--target-column scrambled_af_abraom --metric-target-column af_abraom` (controle negativo: AF embaralhada). Define o **piso** — quanto o maquinário sozinho acerta sem sinal real.
  - **A_BR@4096** — idem A_BR, `context_size=4096` (ablação de contexto).
- **Comparação pareada:** todos os adapters são avaliados nas **mesmas rows** (splits fixos) → o Spearman de cada um é comparado por **bootstrap pareado** (B=2000; inner-join por `variant_id`), que dá o CI da *diferença* — muito mais apertado que a SE de cada Spearman isolado (~0.014 @ n=5000). Script: `scripts/compare_freq_adapters.py`.

---

## 3. Resultados

### 3.1 Os quatro adapters — v11 × v10 (Spearman, val / test)

| Adapter | v10 (val / test) | **v11 (val / test)** | papel |
|---|---|---|---|
| **A_BR** (`af_abraom`) | 0.114 / 0.104 | **0.135 / 0.127** | adapter regional |
| **A_gnomAD** (`af_gnomad`→`af_abraom`) | 0.107 / 0.118 | **0.120 / 0.118** | comparador "frequência genérica" |
| **A_scrambled** (controle −) | −0.013 / 0.001 | **−0.036 / −0.021** | piso |
| A_BR @ ctx 4096 | — | 0.129 / 0.127 | ablação de contexto |

*(Pearson acompanha mas não é citado como diferencial: o A_BR ganha em Pearson já no v10 — treina na escala do `af_abraom`, então acerta a magnitude. Não é sinal de rank. Métrica honesta = Spearman.)*

### 3.2 Os testes pareados (bootstrap, B=2000, n≈5000)

| Comparação | val: Δ [95% CI] P | test: Δ [95% CI] P | veredito |
|---|---|---|---|
| **A_BR − A_scrambled** | +0.172 [+0.131, +0.213] **1.000** | +0.149 [+0.105, +0.191] **1.000** | **sinal real** (CI ≫ 0) |
| **A_BR − A_gnomAD** | +0.016 [−0.005, +0.036] 0.93 | +0.010 [−0.018, +0.039] 0.74 | **não-significativo** (CI cruza 0) |

**Leitura:** `A_BR ≈ A_gnomAD ≫ A_scrambled`. O primeiro par (vs scrambled) estabelece que **existe sinal de sequência→AF**; o segundo (vs gnomAD) estabelece que esse sinal **não é regional** — treinar na AF brasileira não bate treinar na AF global, dentro do ruído.

### 3.3 A ablação de contexto (por que ctx 1024 não handicapa o v11)

A_BR@4096 = 0.1273 (test) vs A_BR@1024 = 0.1274 — **idêntico**. E os dois jobs rodaram a ~3.5 s/step (o de 4096 até levemente mais rápido): o workload é **overhead-bound** (batch 2, d=384 → tensores pequenos), então 4× de contexto custou ~zero. Conclusão: não é um trade-off de compute — é **ausência de sinal de longo alcance** para frequência. Isso valida usar ctx 1024 em tudo (paridade + comparabilidade), sem penalizar o v11.

---

## 4. Interpretação: o fio que amarra com o estudo do Pedro

O achado central do Pedro no v10 (report de transferência ABRAOM) foi **nuançado**: o M5_v3_safety melhora o brasileiro (`br_only` MCC 0.279→0.605), mas os controles negativos estratificados chegam perto do real → a **especificidade biológica do ABRAOM não foi falsificada**.

A Fase 1 do v11 **dá o mecanismo upstream desse caveat:** o sinal regional-específico **nunca esteve forte no adapter de frequência**, nem no v10 nem no v11 (com backbone melhor e teste pareado limpo). Logo o ganho downstream do M5_v3 é **calibração de frequência** (usar a AF ABRAOM para descontar benignas comuns) — não biologia brasileira aprendida da sequência. Os dois resultados são **consistentes**, e o v11 confirma a leitura conservadora do Pedro em vez de derrubá-la.

**Isto é um resultado completo e honesto, não um fracasso:** "há sinal seq→AF real no v11, mas não é brasileiro-específico" é uma conclusão publicável, e ela **de-risca** a decisão sobre quanto investir no downstream.

---

## 5. Cuidados de leitura

1. **A melhora v11-sobre-v10 (+20%) é real mas modesta** (~1 SE isolada; sustenta a leitura a consistência val+test). É melhora de AF **genérica**, não regional.
2. **O v10 saturou (best_step 500), o v11 ainda subia (best_step 1000 = último eval).** Só 2 pontos de eval (a cada 500) → o budget de 1000 steps, calibrado no v10, pode estar subdimensionado pro v11. Um treino mais longo (2–3k steps, ~1–2 h) poderia firmar OU matar o Δ regional — de baixo retorno esperado dado o null, mas barato.
3. **n=5000 por split.** O bootstrap pareado é o teste certo (rows idênticas), mas a Fase 1 inteira é de baixa potência para diferenças ~0.01.
4. **Sequence-only por design** (`use_gnomad_prior=False`) — para isolar o que a sequência carrega. Um adapter com o prior gnomAD explícito é outra pergunta.

---

## 6. Próximos passos

1. **Refinação v11-nativa (o teste direto do que faltou):** treinar o adapter no **resíduo** `af_abraom − f(gnomad_af_pred)`, usando a cabeça de população nativa do v11 (`population_af_head`, que já prediz AF gnomAD por posição). Isola **exatamente** o componente regional — mais limpo que o A_BR-vs-A_gnomAD indireto. Expectativa calibrada pra baixo pelo null de hoje, mas é o experimento que **decide** a pergunta regional.
2. **Fases 2-3 (M0 + fusion + M5_v3) no v11:** o null no adapter **não** cancela isto — o ganho do M5_v3 é via calibração de AF, que independe do adapter ter sinal regional. Reproduzir testa se o ganho de calibração reproduz no backbone novo. ⚠️ Exige editar `clinvar_m0_job.py:52` e `clinvar_fusion_job.py:91` (usam `_upsert_arg` que força `lumina`/`beat-v10`, sem passthrough).
3. **Bifurcação de escopo (decisão de gestão):** **(A)** completar o porte (resultado confirmatório rigoroso) vs **(B)** pivotar pro ângulo nativo/resíduo (maior risco, único caminho pra sinal que o adapter não acha).

---

*Estado em 2026-07-10: Fase 0 (integração v11) e Fase 1 (adapters de frequência) completas e validadas. Veredito da Fase 1: sinal seq→AF real, não regional-específico — reproduz e explica o caveat do Pedro. Artefatos em `s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/`.*
