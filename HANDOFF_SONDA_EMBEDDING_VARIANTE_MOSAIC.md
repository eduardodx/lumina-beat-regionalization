# HANDOFF — Sonda de embedding: o modelo (Lumina R03) percebe variantes? (frente Mosaic)

> **Para quem pega num chat novo:** auto-contido. É uma frente NOVA, mas usa o **mesmo modelo Lumina R03**
> e a mesma infra da campanha de regionalização (`HANDOFF_R03_CONTINUACAO.md`) — reusa MUITA coisa, ver §3.
> Datado **2026-08-25**. Autor: **Gabriel** (dev, TCC). Gestor: **Eduardo**. Este doc é o meu levantamento
> técnico; some com o `.md` que o Gabriel escreveu (com o pedido do Eduardo) — os dois se complementam.

---

## 1. A missão (o que o Eduardo pediu)

Testar se o **modelo Lumina consegue identificar variantes** e se esse sinal é **perceptível** no
embedding. É **parte** de um trabalho maior (o Gabriel tem mais a fazer nesta frente).

**Método sugerido pelo Eduardo (sonda de embedding two-tower):**
1. Pegar uma variante do **novo repositório do Mosaic** (que traz variantes do ClinVar): `chrom, pos, ref, alt`.
2. **Espelhar no hg38:** construir a janela de referência do hg38 centrada em `pos` (a base é o `ref`).
3. Passar essa janela **de referência (sem a variante)** pelo modelo → pegar o **embedding na posição `pos`**.
4. Construir a **mesma janela com a variante** (troca a base por `alt`) → embedding na **mesma posição**.
5. **Comparar os dois embeddings.** Se mudou, o modelo "percebeu" a variante. (Eduardo sugeriu **soma**,
   mas disse explicitamente "do jeito que achar melhor" → ver a métrica recomendada no §5.)
6. **Caso ideal:** achar **duas variantes na mesma posição** (`ref`, `alt1`, `alt2`) e comparar os **3
   embeddings** — os 3 sendo diferentes mostra que o modelo distingue *qual* alelo, não só "tem/não tem".

> **Tradução técnica:** é exatamente o **two-tower ref/alt**. `embedding_ref[pos]` vs `embedding_alt[pos]`,
> na mesma janela, mudando só a base focal. A magnitude da diferença = quanto o modelo "sente" a variante.

---

## 2. O modelo — R03 e o `encode()` (a ferramenta do teste)

- **Modelo:** Lumina **R03** (`LUM-20260719-001-R03`, Mamba-3 hourglass, 52M, **d_full=448**), repo
  `lumina-inference`. Loader: `from lumina import load_model_from_checkpoint, batch_encode_dna`.
- **`model.encode(input_ids, *, variant_edit_mask=None)`** (`lumina/models/model.py:327`) retorna
  `{"last_hidden_state": [B, L, 448], "mid_hidden_state": [B, L/4, 384]}`.
  - **`last_hidden_state` é FULL-RES: 1 posição por base.** Então `emb[:, pos_na_janela, :]` é o vetor de
    448-dim da posição da variante — é EXATAMENTE o que o teste precisa.
  - **`variant_edit_mask`** (opcional) ativa um "variant-residual boost" nativo na posição
    (`model.py:365` — `h_up + gamma*h_stem` onde o mask é True). **NÃO use no two-tower do Eduardo** (o
    sinal viria "de graça" do boost, não da representação aprendida). Deixe `None` e mude a base na
    sequência (é o que o `r03_adapter` já faz — ver §3).
- **Outras cabeças que "veem" a variante** (`_token_head_outputs`, `model.py:378`), úteis como sondas
  ALTERNATIVAS/complementares ao embedding:
  - `mlm_logits [B,L,vocab]` → **masked-center LLR**: `log p(alt) − log p(ref)` na posição = score
    zero-shot de efeito (documentado no TECHNICAL.md do R03).
  - `counterfactual_effect_logits [B,L,4,8]` → cabeça que prediz o **efeito de cada SNV alternativo** por
    posição (o modelo já tem uma noção nativa de "efeito de variante").
  - `missense_severity_pred [B,L,4]`, `conservation_scalar_pred`, `gnomad_af_pred [B,L,4]` (interface
    populacional — ver o prior no §7).

**Checkpoint (S3, croma-bioai — outra org; acesso confirmado do notebook):**
`s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt`
(627 MB). *Confirmar com o Eduardo se a sonda é sobre o R03 ou um checkpoint específico do novo Mosaic.*

---

## 3. Infra REUSÁVEL (o teste já está ~80% pronto)

Verificado no código. Reusar em vez de reescrever:

| Peça | Onde | O que faz |
|---|---|---|
| **`_extract_paired_variant_features`** | `eval/clinvar/adapters.py` | O **coração**: dado ref/alt, retorna `site_ref` (emb da posição), **`variant_repr = alt_emb − ref_emb`** (a diferença que o Eduardo quer!), `local_context` (mean ±64 bp). |
| **`extract_two_tower_features.py`** | `scripts/` | Roda o acima **em lote** sobre um dataset de variantes; salva `.npz` [N, 3·448] + índice parquet. Já faz ref/alt do hg38, ONE backbone pass, sem treino. |
| **`FineTuneR03Adapter`** | `eval/clinvar/r03_adapter.py` | Adapter R03 já portado: `tokenize` (batch_encode_dna), `forward_hidden_states` (= `encode()['last_hidden_state']`), `extract_variant_features` (two-tower). Backbone congelado. |
| **`build_variant_cache`** | `eval/clinvar/dataset.py` | Constrói as **janelas ref/alt do hg38** a partir de `Chromosome/Start/ReferenceAlleleVCF/AlternateAlleleVCF` (left-normaliza, valida REF vs hg38, extrai a janela `context_size`). |
| **`prepare_matched_eval_slices.py`** | `scripts/` | Materializa variantes do master ClinVar (join por `variant_key` → colunas `Chromosome/Start/Ref/Alt/label`). Bom pra montar o dataset de teste. |
| **`extract_native_pathogenicity_features`** | `eval/clinvar/adapters.py` / `r03_adapter.py` | Lê as cabeças nativas (conservation, missense) no sítio — molde pra ler `mlm_logits`/`counterfactual` se quiser as sondas alternativas. |

> **Ou seja:** `variant_repr = alt_emb − ref_emb` **já é computado** pela infra. O trabalho novo é
> **(a)** medir/estatística da magnitude dessa diferença (não só extrair), **(b)** o caso de 2 alelos na
> mesma posição, e **(c)** ligar ao **novo Mosaic** como fonte de variantes.

---

## 4. Dados e caminhos

- **hg38:** `~/hg38/hg38.fa` (no notebook). Referência do "espelhar".
- **ClinVar (dentro do Mosaic):** `s3://ai4bio-lumina/benchmarks/mosaic/data/raw/clinvar/clinvar_20260606.vcf.gz`
  (+ XMLs VCV/RCV na mesma pasta). Mosaic processado: `s3://ai4bio-lumina/benchmarks/mosaic/data/processed/`.
- **Master ClinVar×ABraOM (já labeled, sem VUS):**
  `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet`
  (1.089.826 variantes canônicas; colunas `variant_key, Chromosome, Start, ReferenceAlleleVCF, AlternateAlleleVCF, label, GeneSymbol, consequence_bucket`). Fonte prática pra tirar variantes já normalizadas.
- **Repositório do Mosaic (a fonte das variantes) — NOVA VERSÃO:** **`github.com/croma-bioai/lumina-mosaic`**,
  na org **croma-bioai** (a mesma do checkpoint R03). É **privado** (404 sem auth); clonar no **notebook**
  (que tem acesso GitHub): `git clone git@github.com:croma-bioai/lumina-mosaic.git` em `~/testeArq/`. É uma
  versão nova/separada do benchmark Mosaic do Eduardo (o antigo era `lumina-benchmarks-mosaic-eval`, frente
  de ClinVar/FMs — ver `HANDOFF_CONTINUACAO_V11.md`). **Único ponto a confirmar ao abrir o repo:** ONDE
  estão as variantes ClinVar e em que FORMATO (VCF? parquet? já com `chrom/pos/ref/alt`?) — isso amarra a
  fonte de variantes ao two-tower. (Não consegui inspecionar do Windows: sem `gh`/auth e o repo é privado.)

---

## 5. Desenho técnico proposto (como implementar com rigor)

**Núcleo (o que o Eduardo pediu), por variante:**
1. Janela `ref` do hg38 centrada em `pos` (base = REF, validada). Janela `alt` = idem com a base focal = ALT.
2. `emb_ref = encode(ref)['last_hidden_state'][0, pos_janela]` ; `emb_alt = encode(alt)['last_hidden_state'][0, pos_janela]`. (Ambos 448-dim. Reusar `_extract_paired_variant_features`.)
3. **Métrica de diferença (recomendo mais que "soma"):**
   - **Distância cosseno** `1 − cos(emb_ref, emb_alt)` (escala-invariante, robusta) **e** **L2** `‖emb_alt − emb_ref‖₂`.
   - "Soma" do Eduardo ≈ `Σ(emb_alt − emb_ref)` — inclua também, mas ela cancela sinais opostos; cosine/L2 são melhores. A infra já dá o vetor `variant_repr`, então as três saem do mesmo dado.
4. **Sinal perceptível?** Comparar a diferença ref→alt contra **controles** (sem isso, "≠0" não diz nada,
   porque o modelo tem ruído numérico):
   - **Controle nulo:** `emb_ref` vs `emb_ref` (mesma janela 2×) → deve dar ~0 (sanity de determinismo).
   - **Controle de escala:** diferença ref→alt vs a diferença entre **duas posições vizinhas** da própria
     janela (quão grande é uma diferença "natural" de posição) — a variante deve destacar-se na posição focal.
   - **Controle de permutação:** trocar a base focal por uma base **aleatória** N vezes → distribuição de
     magnitude; a variante real deve estar dentro/acima dela (mede "o modelo reage a QUALQUER troca?").
5. **Caso ideal (2 alelos, mesma posição):** `ref, alt1, alt2` → 3 embeddings → as **3 distâncias
   par-a-par** (ref–alt1, ref–alt2, **alt1–alt2**) > 0. A `alt1–alt2 > 0` é a prova forte: o modelo
   distingue *qual* alelo, não só "mudou algo".

**Estatística sobre N variantes (é isso que responde "perceptível"):**
- Distribuição das distâncias ref→alt em N variantes ClinVar vs a distribuição dos controles. Perceptível
  = separação clara (teste pareado / effect size), não uma variante isolada.
- **Bônus com alto valor (liga "percebe variante" a "percebe patogenicidade"):** as **patogênicas** têm
  diferença de embedding **maior** que as **benignas**? Se sim, o embedding carrega sinal clínico. (O
  ClinVar/master tem `label` — dá pra estratificar de graça.) AUROC(distância → patogenicidade) como resumo.
- Onde ler o embedding: o Eduardo pediu a **posição focal**; considere também `local_context` (±64 bp) e o
  `mid_hidden_state` — o efeito da variante pode se espalhar pelo contexto.

**Sondas alternativas (se o embedding-diff for fraco — ver prior §7):** `mlm_logits` LLR
`log p(alt)−log p(ref)`, `counterfactual_effect_logits`, `gnomad_af_pred`. Todas do mesmo `encode()` +
`_token_head_outputs`.

---

## 6. Fluxo de trabalho (crucial) e ambiente

- **Windows local NÃO roda nada pesado** (sem torch/GPU/AWS) — só edita código e escreve runbooks. O Gabriel
  **commita, dá `git pull` no notebook SageMaker e roda** (GPU/torch/S3). Nunca rodar torch/AWS do Windows.
- **Ambiente R03 no notebook:** `cd ~/testeArq/lumina-inference && source scripts/env.sh` → exporta **`$PY`**
  (`.venv/bin/python`, torch do conda + mamba) e `$REPO_ROOT`. `$WORK` = `~/testeArq/lumina-beat-regionalization`.
  Rodar scripts do repo de trabalho com `PYTHONPATH="$WORK" "$PY" ...`. **Nunca `uv sync` no host GPU.**
- **Sonda é LEVE:** é inferência (um forward do backbone por janela), sem treino. Cabe numa GPU pequena; não
  precisa das 8×H100 do M0. Determinística → reprodutível.

---

## 7. Priors e gotchas (do trabalho R03 — poupam tempo)

1. **verify-over-doc:** ler o state_dict/código, não confiar em default/doc (foi assim que se achou
   d_full=448, o apply_lora exclusion-based, etc.).
2. **Prior da Fase 0.5 (relevante!):** a **interface populacional nativa** do R03 (`gnomad_af_pred`) é um
   **"weak prior" fraco** — ρ≈+0.14 entre score e log10(AF). Isso NÃO é o mesmo que o embedding-diff (que
   mede sensibilidade à base, não frequência), MAS avisa que **sinais nativos do R03 podem ser modestos**;
   dimensionar expectativa e usar bom controle/estatística. O embedding two-tower tende a ter sinal mais
   forte que a cabeça de AF (é a representação inteira, não uma cabeça específica).
3. **`variant_edit_mask` = boost nativo** — não usar no two-tower (contamina o teste). Base trocada na
   sequência é o caminho limpo.
4. **d_full=448** = 384 (d_model) + 64 (d_pure). O `last_hidden_state` é a concatenação `[h_up | h_pure]`.
5. **Janela e centralização:** a variante tem que ficar no centro da janela (contexto simétrico). O
   `build_variant_cache`/`_extract_paired_variant_features` já cuidam disso; confira o `context_size` (o
   ClinVar usou 4096; o two-tower usa 1024 por default — escolher e fixar).
6. **Normalização de variante:** left-normalize + validar REF vs hg38 antes (o `build_variant_cache` faz).
   Indels precisam de cuidado (a "posição" é ambígua); começar por **SNVs** (posição limpa) e tratar indel
   depois.

---

## 8. Primeiros passos sugeridos

1. **Clonar o Mosaic no notebook** (`git clone git@github.com:croma-bioai/lumina-mosaic.git ~/testeArq/`)
   e **inspecionar**: onde estão as variantes ClinVar e o formato (VCF/parquet, colunas) — §4.
2. **Smoke:** 1 variante SNV do ClinVar → `_extract_paired_variant_features` → imprimir `‖variant_repr‖`,
   cosine(ref,alt). Validar que ≠0 e que ref-vs-ref ≈ 0. (Reusa tudo do §3; roda em segundos.)
3. **Achar 2 alelos na mesma posição** no ClinVar (é comum: mesma pos, alt diferentes) → os 3 embeddings.
4. **Escalar:** N variantes (SNV) via `extract_two_tower_features.py` → distribuição das distâncias +
   controles + estratificar por `label` (patogênica vs benigna) → AUROC.
5. Só então indels / sondas alternativas / o resto do trabalho da frente.

**O que NÃO fazer:** não usar `variant_edit_mask` (boost) no two-tower; não rodar torch no Windows; não
reescrever a extração ref/alt (reusar `_extract_paired_variant_features`); não misturar com a frente
seq-lab (branch `seq-lab-pipeline`) nem com a campanha M0–M4 (essa sonda é diagnóstica, não a escada).

---

## 9. Referência rápida (caminhos)

- **Modelo R03:** repo `~/testeArq/lumina-inference` · `from lumina import load_model_from_checkpoint, batch_encode_dna` · `model.encode()` em `lumina/models/model.py:327`.
- **Checkpoint:** `s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt`.
- **Infra two-tower:** `eval/clinvar/adapters.py::_extract_paired_variant_features` · `scripts/extract_two_tower_features.py` · `eval/clinvar/r03_adapter.py`.
- **Mosaic (repo novo, fonte primária das variantes):** `github.com/croma-bioai/lumina-mosaic` (privado; clonar no notebook em `~/testeArq/`).
- **Variantes (fontes S3 já conhecidas):** ClinVar `s3://ai4bio-lumina/benchmarks/mosaic/data/raw/clinvar/clinvar_20260606.vcf.gz` · master `s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet` · hg38 `~/hg38/hg38.fa`.
- **Ambiente:** `cd ~/testeArq/lumina-inference && source scripts/env.sh` (exporta `$PY`); `$WORK=~/testeArq/lumina-beat-regionalization`; rodar com `PYTHONPATH="$WORK" "$PY" ...`.

*Fim. Estado: frente recém-aberta, nada implementado. O teste do Eduardo é uma sonda de embedding two-tower
que reusa ~80% da infra do R03 (`_extract_paired_variant_features` já dá `alt_emb − ref_emb`). Próximo passo
real: confirmar o repo Mosaic novo + smoke de 1 variante.*
