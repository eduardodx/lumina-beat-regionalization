# Lumina

**A compact, biologically supervised genomic foundation model research core aimed at allele-sensitive variant interpretation.**

## Context

The field moved quickly.

Bi-directional DNA sequence modeling, long-context pretraining, and biologically informed supervision are no longer empty territory. That matters for Lumina because it changes what can count as a serious research contribution. A compact BiMamba backbone, dense biological priors, and efficient long-context modeling are still worthwhile design choices, but they are no longer sufficient as the main novelty claim on their own.

Lumina therefore adopts a narrower and more defensible research position:

- dense biological supervision remains a strong and underused training principle
- compact models still matter for reproducibility, iteration speed, and realistic deployment
- the central scientific question is no longer "can a small model exist?"
- the central scientific question is whether a compact genomic model can become genuinely **allele-sensitive** rather than relying mostly on genomic context priors

## Research Objective

The research objective for Lumina is:

**To develop a compact, biologically supervised, allele-sensitive genomic foundation model for clinically relevant noncoding and splice-associated variant interpretation, and to evaluate it with explicit counterfactual and regional calibration protocols.**

This is deliberately narrower than "build a better DNA language model" and more realistic than trying to prove, in one step, that a compact model beats every larger system at every downstream task.

## Project Thesis

Lumina is guided by four claims.

1. **Dense biological supervision is non-negotiable.**
   MLM is useful, but conservation and splice structure should supervise the model densely across all valid genomic positions.

2. **Compactness is a means, not the main novelty.**
   Smaller models make controlled ablation, repeated runs, and accessible deployment easier, but compactness alone is not the scientific story.

3. **Allele sensitivity is the critical missing property.**
   A clinically useful genomic model should respond to the actual ref->alt change, not just exploit surrounding context or locus priors.

4. **Clinical and regional claims must be earned.**
   Pathogenicity, calibration, and Latin American relevance require explicit benchmarks, held-out evaluation, and careful comparison against strong baselines.

## Paper Propositions

Lumina is best understood as a three-part research program rather than a single oversized paper.

### Paper 1: Methods

**A compact, allele-sensitive genomic foundation model for noncoding and splice-relevant variant interpretation.**

Core ingredients:

- compact bidirectional Mamba-style backbone
- dense PhyloP and splice supervision
- counterfactual ref/alt modeling
- token-level clinical interfaces rather than only frozen global pooling
- rigorous held-out and counterfactual evaluation

### Paper 2: Regional Validation

**Brazilian-aware and admixed-population validation, calibration, and failure analysis.**

Core ingredients:

- external holdout and calibration analysis on Brazilian or Latin American data resources
- temporal validation splits for variant interpretation
- subgroup calibration and error analysis
- VUS triage and regional performance characterization

### Paper 3: Translational Pipeline

**An end-to-end laboratory workflow that connects sequencing, calling, and model-based interpretation.**

Core ingredients:

- MinION or related third-generation sequencing workflow integration
- variant calling and interpretation pipeline design
- turnaround-time, cost, and usability evaluation in the UFG setting

This repository is primarily the research core for **Paper 1**.

## What This Repository Contains

This repository focuses on reproducible pretraining of a bidirectional Mamba-style DNA model on the human reference genome with multi-task biological supervision.

Current components include:

- a **BiMamba DNA backbone** for nucleotide-level modeling
- a dataset pipeline over **hg38**
- auxiliary targets from **PhyloP** conservation tracks
- token-level **splice structure** labels derived from GENCODE
- reverse-complement augmentation and consistency training
- held-out chromosome validation
- local metric history, checkpoint selection, and optional `wandb` tracking
- sanity-check and training entry points for reproducible local experimentation

Important scope note:

- this repository does **not** yet implement paired ref/alt training
- it does **not** yet include a ClinVar or ABraOM benchmark pipeline
- it does **not** yet provide a production clinical classifier or MinION workflow

Those are downstream program stages, not current codebase claims.

## Reproducible Setup

Phase 0 standardizes this repository around a single `uv` workflow.

- Supported Python policy: `>=3.11,<3.13`
- Recommended local default: Python `3.11`
- Default install path: `uv`, not raw `pip`
- Notebook tooling is optional
- Experiment tracking is optional through the `tracking` extra

Create the environment and install the supported development toolchain:

```bash
uv venv --python 3.11
uv sync --extra dev
```

If you also want baseline experiment tracking support:

```bash
uv sync --extra dev --extra tracking
```

If you also want notebook tooling:

```bash
uv sync --extra dev --extra notebook
```

Expected local data layout:

- `data/hg38/hg38.fa`
- `data/phylo/hg38.phyloP100way.bw`
- `data/phylo/hg38.phyloP470way.bw`
- `data/gencode/gencode.v38.annotation.gtf.gz`

### Data Downloads

Official download links for the files expected by this repository:

- GENCODE v38 annotation GTF:
  [gencode.v38.annotation.gtf.gz](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz)
- hg38 reference FASTA:
  [hg38.fa.gz](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/latest/hg38.fa.gz)
- hg38 PhyloP 100-way bigWig:
  [hg38.phyloP100way.bw](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw)
- hg38 PhyloP 470-way bigWig:
  [hg38.phyloP470way.bw](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP470way/hg38.phyloP470way.bw)

After downloading:

- decompress `hg38.fa.gz` into `data/hg38/hg38.fa`
- place the GENCODE file at `data/gencode/gencode.v38.annotation.gtf.gz`
- place the PhyloP files at `data/phylo/hg38.phyloP100way.bw` and `data/phylo/hg38.phyloP470way.bw`

Canonical sanity-check command:

```bash
uv run python -m src.sanity \
  --fasta-path data/hg38/hg38.fa \
  --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
  --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
  --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
  --model bimamba \
  --seq-len 1024 \
  --batch-size 2 \
  --num-workers 0
```

Config-driven runs now select an architecture family with `model` and put architecture overrides under nested `model_config`:

```yaml
model: bimamba
model_config:
  d_model: 256
  n_layers: 8
  d_state: 64
  d_conv: 4
  expand: 2
  dropout: 0.1
```

Canonical baseline command:

```bash
uv run python -m src.train \
  --model bimamba \
  --fasta-path data/hg38/hg38.fa \
  --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
  --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
  --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
  --seq-len 4096 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 1000 \
  --output-dir outputs/lumina_8m_baseline \
  --eval-every 100 \
  --save-every 100 \
  --log-every 10
```

Tracked baseline command:

```bash
uv run python -m src.train \
  --model bimamba \
  --fasta-path data/hg38/hg38.fa \
  --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
  --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
  --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
  --seq-len 4096 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 1000 \
  --output-dir outputs/lumina_8m_baseline \
  --eval-every 100 \
  --save-every 100 \
  --log-every 10 \
  --wandb-enabled \
  --wandb-project lumina
```

Short debug-train command:

```bash
uv run python -m src.train \
  --model bimamba \
  --fasta-path data/hg38/hg38.fa \
  --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
  --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
  --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
  --seq-len 1024 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 50 \
  --output-dir outputs/lumina_debug \
  --eval-every 10 \
  --save-every 0 \
  --log-every 5
```

SageMaker training (B200):

```bash
# Requires: uv sync --extra sagemaker (boto3, botocore[crt], sagemaker, python-dotenv)
# Requires: SAGEMAKER_ROLE and WANDB_API_KEY set in .env or environment

# Launch beat-v1 on 8xB200 (spot, detached)
bash scripts/dispatch_beat_v1_b200.sh

# Or call sagemaker_train.py directly for custom experiments
python scripts/sagemaker_train.py \
  --experiment my-experiment \
  --config configs/beat_v1/8m_15ep_32k_b200.yaml \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot --detach
```

Note: SageMaker source packaging uses `git archive HEAD`. Commit any new configs or launcher scripts before dispatching a job, or the container will not receive them.

## ClinVar Embedding Evaluation

The `eval/clinvar/` pipeline extracts ref/alt embeddings from DNA foundation models for ClinVar variants. It produces six mean-pooled embedding vectors per variant (full-window and local ±64bp/±256bp for both ref and alt sequences) and writes them alongside the original ClinVar metadata to a parquet file.

Install the eval dependencies:

```bash
uv sync --extra eval
```

### Nucleotide Transformers v3

NTv3 uses a character-level tokenizer over genomic bases, so the local `±64bp` / `±256bp`
embedding pools align directly to nucleotide positions.

```bash
# 8M (small, runs on CPU/MPS)
python -m eval.clinvar.extract_embeddings \
  --model-family ntv3 --model-version 8M_pre \
  --fasta-path data/hg38/hg38.fa \
  --context-size 4096 --batch-size 32

# 100M
python -m eval.clinvar.extract_embeddings \
  --model-family ntv3 --model-version 100M_pre \
  --fasta-path data/hg38/hg38.fa \
  --context-size 4096 --batch-size 16

# 650M
python -m eval.clinvar.extract_embeddings \
  --model-family ntv3 --model-version 650M_pre \
  --fasta-path data/hg38/hg38.fa \
  --context-size 4096 --batch-size 8
```

### Caduceus (GPU only, requires Triton)

```bash
# caduceus-ph
python -m eval.clinvar.extract_embeddings \
  --model-family caduceus --model-version caduceus-ph \
  --fasta-path data/hg38/hg38.fa \
  --context-size 4096 --batch-size 32

# caduceus-ps
python -m eval.clinvar.extract_embeddings \
  --model-family caduceus --model-version caduceus-ps \
  --fasta-path data/hg38/hg38.fa \
  --context-size 4096 --batch-size 32
```

### DNABERT-2 (GPU only, requires Triton)

```bash
python -m eval.clinvar.extract_embeddings \
  --model-family dnabert2 --model-version 117M \
  --fasta-path data/hg38/hg38.fa \
  --context-size 4096 --batch-size 16
```

### Lumina

The output directory is derived from the checkpoint's parent folder name (e.g. `outputs/lumina_8m_baseline/best_model.pt` produces `eval/clinvar/embeddings/lumina/lumina_8m_baseline/`).

```bash
python -m eval.clinvar.extract_embeddings \
  --model-family lumina --model-version bimamba \
  --fasta-path data/hg38/hg38.fa \
  --checkpoint-path outputs/lumina_8m_baseline/best_model.pt \
  --context-size 4096 --batch-size 64
```

### ClinVar Fine-Tuning on SageMaker

Use the dedicated ClinVar SageMaker dispatcher for single-run LoRA fine-tuning:

```bash
CHECKPOINT_DIR=s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking \
HEAD_HIDDEN_DIM=256 \
HEAD_DROPOUT=0.1 \
LORA_R=32 \
WANDB_ENABLED=true \
bash scripts/dispatch_clinvar_finetune_b200.sh
```

The SageMaker job writes intermediates to local scratch and uploads only:

- `metrics.json`
- `test_predictions.parquet`
- `best_model.pt`

Published outputs land under `s3://<bucket>/lumina-ssm/eval/clinvar/finetune/<experiment>/`.
Lumina runs can pass either a container-local checkpoint directory under `/opt/ml/` or an S3 checkpoint prefix under `s3://<bucket>/lumina-ssm/data/checkpoints/...`; the launcher stages remote bundles into `/opt/ml/input/data/training/checkpoints/<name>/` and resolves `best_checkpoint.pt` there.
Inside the container, the job launches `torch.distributed.run` so all visible GPUs participate in one distributed fine-tuning run.
Before launching `torchrun`, the container self-stages `datasets/clinvar/`, `hg38/`, `gencode/`, and `phylo/` from `s3://<bucket>/lumina-ssm/data/` into `SM_DATA`, and mirrors the Hugging Face hub cache from `s3://<bucket>/huggingface/hub/` into `/tmp/huggingface/hub`.
The launcher also forces offline Hugging Face resolution inside the container, so gated backbones such as NTv3 and DNABERT-2 load from the mirrored cache instead of requiring a runtime token.

Concrete local Lumina example:

```bash
env PYTHONPATH=$PWD torchrun \
  --nproc-per-node "$(nvidia-smi -L | wc -l)" \
  -m eval.clinvar.run \
  --regime A \
  --model-family lumina \
  --model-version beat-v1 \
  --checkpoint-path outputs/lumina_8m_beat_v1_15ep_32k_consmasking/best_checkpoint.pt \
  --dataset-path data/datasets/clinvar/processed/clinvar_dataset.parquet \
  --fasta-path data/hg38/hg38.fa \
  --context-size 4096 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --max-epochs 12 \
  --lr-backbone 5e-5 \
  --lr-head 3e-4 \
  --hidden-dim 256 \
  --head-dropout 0.1 \
  --pos-weight 1.0 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --precision bf16 \
  --allow-tf32 \
  --output-dir eval/clinvar/metrics/lumina/beat-v1/finetune_4k_lumina_push \
  --overwrite
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--context-size` | 4096 | Genomic window size in bp centered on the variant |
| `--batch-size` | 32 | Per-device batch size |
| `--precision` | `bf16` | Eval precision policy (`auto`, `bf16`, or `fp32`) |
| `--allow-tf32` | on | Allow TF32 matmuls on supported CUDA devices |
| `--max-variants` | all | Limit to first N variants (for debugging) |
| `--output-dir` | auto | Override output path (default: `eval/clinvar/embeddings/<family>/<version>/`) |
| `--resume` | off | Resume from a partial run |

Output parquets follow the naming convention `{family}_{version}_{context_size}_embeddings.parquet`, for example:

- `eval/clinvar/embeddings/nucleotide-transformers-v3/8M_pre/nucleotide-transformers-v3_8M_pre_4096_embeddings.parquet`
- `eval/clinvar/embeddings/lumina/lumina_8m_baseline/lumina_lumina_8m_baseline_4096_embeddings.parquet`

Multi-GPU is auto-detected when CUDA devices > 1.

## ClinVar MLP Evaluation

The embedding extractor can now feed a second-stage multiscale allele-aware MLP evaluator. The evaluator reads a single embedding parquet, splits rows by `split_within_gene`, trains on the three ref/alt scales `[global, ±256bp, ±64bp]`, and writes a metrics JSON per model.

Evaluate one embedding run:

```bash
python -m eval.clinvar.evaluate_mlp \
  --embeddings-path eval/clinvar/embeddings/nucleotide-transformers-v3/8M_pre/embeddings.parquet
```

Concrete example for the current `ntv3 / 8M_pre` run:

```bash
python -m eval.clinvar.evaluate_mlp \
  --embeddings-path eval/clinvar/embeddings/nucleotide-transformers-v3/8M_pre/embeddings.parquet \
  --model-family nucleotide-transformers-v3 \
  --model-version 8M_pre
```

Evaluate every discovered family/version under `eval/clinvar/embeddings/`:

```bash
python -m eval.clinvar.run_all_mlp
```

Outputs:

- per model: `eval/clinvar/metrics/<family>/<version>/multiscale_allele_mlp.json`
- aggregate summary: `eval/clinvar/metrics/clinvar_multiscale_mlp_summary.json`

The per-model JSON reports the main benchmark metrics `MCC`, `F1`, `optimal_F1`, plus `AUPRC`, `AUROC`, `balanced_accuracy`, `specificity`, `brier_score`, `log_loss`, and threshold-specific test blocks.

## Current Training Direction

The current recommended direction for this repository is:

- keep **masked language modeling** as a masked-token objective
- compute **auxiliary biological losses at every valid position**, not only masked positions
- establish a strong and reproducible small baseline first
- insert **counterfactual variant modeling** before claiming clinical usefulness
- treat frozen global embeddings as a side interface, not the main clinical interface
- add harder ideas such as curriculum scaling, contrastive RC loss, RoPE, and multi-scale hierarchy only after the baseline and variant interface are trustworthy
- evaluate success on held-out biological metrics, counterfactual tests, and downstream clinical benchmarks, not just training loss

The current baseline is still valuable even before paired ref/alt modeling, because it establishes:

- a reproducible compact pretraining core
- a strong dense-supervision baseline over hg38
- held-out MLM, conservation, and splice metrics
- an implementation surface that can support later allele-sensitive experiments

## Why This Matters

Many genomic models are impressive at scale while remaining difficult to reproduce, difficult to probe mechanistically, and difficult to calibrate for under-evaluated populations.

Lumina is motivated by a different sequence of priorities:

- first make the core pretraining pipeline scientifically sound
- then make the model sensitive to actual allelic change
- then benchmark clinical usefulness carefully
- then test regional calibration explicitly
- only after that, build translational workflow claims on top

If this project succeeds, it will not be because Lumina was simply smaller than other genomic models.

It will be because Lumina became a compact model that used biological supervision intelligently, responded to the actual ref->alt change, and earned downstream claims with rigorous evaluation.
