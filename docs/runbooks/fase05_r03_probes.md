# Runbook — Fase 0.5: sondas baratas do R03 (rodar no JupyterLab do notebook)

Duas sondas read-only, **sem treino**, que decidem viabilidade antes de gastar jobs:

- **(b) sonda representacional chr8** — `scripts/r03_chr8_population_probe.py`: a interface
  populacional do R03 (`gnomad_af_pred`) ainda ordena frequência no chr8, ou colapsou?
  (TECHNICAL.md §6 avisa que o R03 vai mal no chr8; é o nosso *fallback de poder*, então
  precisamos saber se está furado pelo mesmo buraco — a "ameaça dupla".)
- **cheque da superfície LoRA** — `scripts/r03_lora_surface_check.py`: confirma sobre o
  checkpoint REAL o que o `apply_lora` envolve (Mamba `in_proj`/`out_proj`, atenção) e que
  **nenhuma head** vaza.

> Rode num **terminal do JupyterLab** (não em célula `!`, porque `source env.sh` precisa persistir).
> **Nunca `uv sync`/`uv run` no host GPU** — torch vem do conda; o `.venv` é torch-free de propósito.
> `uv pip install <pkg>` (aditivo) é ok — é assim que o próprio `setup-gpu.sh` monta o env.

---

## 0. Puxar o código (após eu commitar/você pushar do Windows)

```bash
cd ~/lumina-beat-regionalization && git pull
```

## 1. Ativar o runtime do R03

```bash
cd ~/lumina-inference && bash scripts/setup-gpu.sh && source scripts/env.sh
```

`env.sh` exporta `$PY` (= `~/lumina-inference/.venv/bin/python`, o interpretador com torch-conda +
mamba_ssm) e imprime `env.sh OK: torch @ ... (conda)`. É esse `$PY` que roda tudo abaixo.

O cheque LoRA precisa só de torch+lumina. A sonda chr8 precisa também de `pyfaidx` (e pandas/numpy).
Confirme e instale o que faltar **no `.venv`** (aditivo, não é `uv sync`):

```bash
"$PY" -c "import torch,pandas,numpy,pyfaidx,lumina; print('deps ok', torch.cuda.is_available())" || uv pip install --python "$PY" pyfaidx pandas numpy
```

---

## 2. Checkpoint R03 — descoberta e acesso no S3 (vira o cartão)

```bash
aws s3 ls s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/
```

Espera ver `best_checkpoint.pt`. `AccessDenied`/`NoSuchBucket` = permissão/credencial → me avisa
(é o bloqueio nº1). Se listar, guarde o URI e faça o smoke de load (confirma 630 tensores + shapes):

```bash
export R03_CKPT="s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt"
```

```bash
LUMINA_CHECKPOINT="$R03_CKPT" "$PY" ~/lumina-inference/examples/smoke_inference.py
```

Espera `last_hidden_state=(1, L, 448)`, `mid_hidden_state=(1, L/4, 384)`, `gnomad_af_pred=(1, L, 4)`.

---

## 3. Cheque da superfície LoRA (device=cuda p/ pegar os nomes reais do mamba_ssm)

Rodar da raiz do repo de trabalho, com ele no PYTHONPATH (pro `import eval.clinvar`):

```bash
cd ~/lumina-beat-regionalization && PYTHONPATH="$PWD" "$PY" scripts/r03_lora_surface_check.py --checkpoint "$R03_CKPT" --device cuda --out ~/artifacts/fase05/r03_lora_surface.json
```

**Ler o JSON:** `head_leaks` deve ser `[]`; `buckets.mamba_in_proj > 0` (o Mamba recebe LoRA — o
grosso dos params); `mha_wrapped_submodules` deve ter só `out_proj` das atenções sparse/anchor (com
o aviso de que esse LoRA provavelmente fica inerte). É a superfície treinável real do R03.

---

## 4. Sonda (b) — representacional chr8

**4a. Smoke (rápido, confirma o pipeline end-to-end)** — usa a slice br_only que já temos.
Descubra o caminho e rode:

```bash
ls ~/slices
```

```bash
cd ~/lumina-beat-regionalization && PYTHONPATH="$PWD" "$PY" scripts/r03_chr8_population_probe.py --checkpoint "$R03_CKPT" --variants ~/slices/br_only.parquet --af-col af_gnomad --fasta ~/hg38/hg38.fa --eval-chrom chr8 --indomain-chrom chr1 --context-size 4096 --batch-size 8 --max-variants 4000 --out ~/artifacts/fase05/r03_chr8_probe_slices.json
```

**4b. Medição com poder** — o índice ABraOM v2 (~17.8M variantes, tem `af_gnomad` + chrom/pos/ref/alt;
o script sintetiza o `variant_key` e lê só as colunas necessárias). É o set representacional que o
Eduardo prioriza. Aponte `--variants` pro parquet do índice:

```bash
cd ~/lumina-beat-regionalization && PYTHONPATH="$PWD" "$PY" scripts/r03_chr8_population_probe.py --checkpoint "$R03_CKPT" --variants <PARQUET_INDICE_ABRAOM> --af-col af_gnomad --fasta ~/hg38/hg38.fa --eval-chrom chr8 --indomain-chrom chr1 --context-size 4096 --batch-size 8 --max-variants 20000 --out ~/artifacts/fase05/r03_chr8_probe_abraom.json
```

**Ler o JSON:** compare `results[].gnomad_af_pred.spearman` (+ `ci95`) entre `set: eval` (chr8) e
`set: indomain` (chr1). Piso de referência: a Fase 1 do v11 mediu Spearman ~0.13 no adapter de AF.
- rho_chr8 se sustenta perto do in-domain → **fallback representacional vivo**.
- rho_chr8 colapsa pra ~0 com gap grande → **interface populacional degradada no chr8** (fallback
  furado) → levar ao Eduardo antes da Fase 2.

---

## O que me trazer de volta

Os JSONs de `~/artifacts/fase05/` (`r03_lora_surface.json`, `r03_chr8_probe_*.json`) + o resultado
do `aws s3 ls`. Com isso fecho a Fase 0.5, monto a estimativa de poder (sonda a1) e o pedido de
decisão ao Eduardo.
