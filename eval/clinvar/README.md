# ClinVar Pathogenicity Fine-Tuning

Fine-tune and evaluate DNA foundation models on ClinVar variant pathogenicity classification using a two-regime framework designed for fair cross-model comparison.

For architectural details and design rationale, see [DESIGN.md](DESIGN.md).

## Prerequisites

### Data files

| File | Path | Required by |
|------|------|-------------|
| hg38 reference FASTA | `data/hg38/hg38.fa` | Both regimes |
| ClinVar dataset | `data/datasets/clinvar/processed/clinvar_dataset.parquet` | Both regimes |
| PhyloP100 BigWig | `data/phylo/hg38.phyloP100way.bw` | Regime B |
| PhyloP470 BigWig | `data/phylo/hg38.phyloP470way.bw` | Regime B |
| GENCODE GTF | `data/gencode/gencode.v38.annotation.gtf.gz` | Regime B |

The ClinVar parquet must contain columns: `Chromosome`, `Start`, `ReferenceAlleleVCF`, `AlternateAlleleVCF`, `label`, `split_within_gene`.

### Environment

```bash
# From the repository root
uv sync --extra dev
```

For tracked runs:

```bash
uv sync --extra dev --extra tracking
```

## Quick Start

### Step 1: Choose a Regime

- **Regime A** — Representation quality.  Uses only backbone embeddings.  Tests which model learned better genomic representations.
- **Regime B** — Practical utility.  Adds explicit biological features (PhyloP, CDS/codon, BLOSUM62).  Tests best achievable performance per backbone.

### Step 2: Run Fine-Tuning + Evaluation

Fine-tuning and evaluation happen in a single command.  The pipeline:

1. Builds a variant window cache from FASTA (first run only, then reused)
2. For Regime B, extracts biological features and caches them alongside
3. Trains with LoRA + classification head using cosine LR + focal loss
4. Evaluates on the held-out test split after each epoch
5. Saves the best checkpoint (by MCC) and final metrics

ClinVar fine-tuning now defaults to `--precision bf16` with `--allow-tf32` enabled on CUDA.  Use
`--precision auto` or `--precision fp32` on unsupported devices, and pass `--no-tf32` if you need
to disable TF32 matmuls explicitly.

```bash
uv run python -m eval.clinvar.run \
    --regime A \
    --model-family lumina --model-version beat-v2 \
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/lumina_beat_v2_A
```

### Step 3: Read Results

Results are written to `--output-dir`:

```
outputs/clinvar/lumina_beat_v2_A/
  metrics.json              # All metrics + hyperparameters
  test_predictions.parquet  # Per-variant probabilities
  best_model.pt             # LoRA + head checkpoint
```

Primary metric is **MCC** (Matthews correlation coefficient), stored under the `"value"` key in `metrics.json`.

---

## Running Each Model

### Lumina

Lumina requires `--checkpoint-path` pointing to a pretrained checkpoint.

```bash
# Regime A
uv run python -m eval.clinvar.run --regime A \
    --model-family lumina --model-version beat-v2 \
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/lumina_beat_v2_A

# Regime B
uv run python -m eval.clinvar.run --regime B \
    --model-family lumina --model-version beat-v2 \
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
    --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
    --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
    --output-dir outputs/clinvar/lumina_beat_v2_B
```

### Nucleotide Transformer v3

Available versions: `8M_pre`, `100M_pre`, `650M_pre`.  Weights are downloaded from HuggingFace on first use.

```bash
# Regime A
python -m eval.clinvar.run --regime A \
    --model-family ntv3 --model-version 8M_pre \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/ntv3_8m_A

# Regime B
python -m eval.clinvar.run --regime B \
    --model-family ntv3 --model-version 8M_pre \
    --fasta-path data/hg38/hg38.fa \
    --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
    --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
    --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
    --output-dir outputs/clinvar/ntv3_8m_B
```

### Caduceus

Available versions: `caduceus-ph`, `caduceus-ps`.

```bash
# Regime A
python -m eval.clinvar.run --regime A \
    --model-family caduceus --model-version caduceus-ph \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/caduceus_ph_A

# Regime B
python -m eval.clinvar.run --regime B \
    --model-family caduceus --model-version caduceus-ph \
    --fasta-path data/hg38/hg38.fa \
    --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
    --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
    --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
    --output-dir outputs/clinvar/caduceus_ph_B
```

### DNABERT-2

Available versions: `117M`.

```bash
# Regime A
python -m eval.clinvar.run --regime A \
    --model-family dnabert2 --model-version 117M \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/dnabert2_117m_A

# Regime B
python -m eval.clinvar.run --regime B \
    --model-family dnabert2 --model-version 117M \
    --fasta-path data/hg38/hg38.fa \
    --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
    --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
    --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
    --output-dir outputs/clinvar/dnabert2_117m_B
```

---

## Multi-GPU Training

DDP is auto-detected from `torchrun` environment variables.  No extra flags needed.

```bash
uv run torchrun --nproc_per_node 8 -m eval.clinvar.run \
    --regime A \
    --model-family lumina --model-version beat-v2 \
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --batch-size 4 \
    --output-dir outputs/clinvar/lumina_beat_v2_A
```

`--batch-size` is per-rank.  With 8 GPUs the effective batch is `4 * 8 = 32`.

---

## Full Comparison Run

Run all four models in both regimes to produce a complete comparison table.

```bash
#!/bin/bash
set -euo pipefail

FASTA=data/hg38/hg38.fa
PHYLO100=data/phylo/hg38.phyloP100way.bw
PHYLO470=data/phylo/hg38.phyloP470way.bw
GTF=data/gencode/gencode.v38.annotation.gtf.gz
LUMINA_CKPT=outputs/beat_v2/best_checkpoint.pt

MODELS=(
    "lumina beat-v2 --checkpoint-path $LUMINA_CKPT"
    "ntv3 8M_pre"
    "caduceus caduceus-ph"
    "dnabert2 117M"
)

for regime in A B; do
    for model_args in "${MODELS[@]}"; do
        read -r family version extra <<< "$model_args"

        cmd="uv run python -m eval.clinvar.run --regime $regime"
        cmd="$cmd --model-family $family --model-version $version"
        cmd="$cmd --fasta-path $FASTA"
        cmd="$cmd --output-dir outputs/clinvar/${family}_${version}_${regime}"

        if [ -n "${extra:-}" ]; then
            cmd="$cmd $extra"
        fi

        if [ "$regime" = "B" ]; then
            cmd="$cmd --phylo100-bw-path $PHYLO100"
            cmd="$cmd --phylo470-bw-path $PHYLO470"
            cmd="$cmd --gtf-path $GTF"
        fi

        echo "=== $family/$version regime $regime ==="
        eval $cmd
    done
done
```

---

## Interpreting Results

`metrics.json` records `requested_precision`, `resolved_precision`, and `allow_tf32` at the top level
so you can confirm whether the run actually executed in `bf16` or fell back to `fp32`.

After running all models, compare `metrics.json` across output directories:

```bash
# Quick comparison table
for d in outputs/clinvar/*/; do
    if [ -f "$d/metrics.json" ]; then
        python -c "
import json, pathlib
m = json.loads(pathlib.Path('$d/metrics.json').read_text())
print(f\"{m.get('model_family','?'):>10} {m.get('model_version','?'):>12} regime {m.get('regime','?')}  MCC={m.get('mcc',0):.4f}  AUROC={m.get('auroc',0):.4f}  AUPRC={m.get('auprc',0):.4f}\")
"
    fi
done
```

### What to look for

| Metric | Meaning |
|--------|---------|
| **MCC** | Primary metric.  Robust to class imbalance.  Range [-1, 1]. |
| **AUROC** | Discrimination ability across all thresholds. |
| **AUPRC** | Precision-recall balance (informative when positives are rare). |
| **F1** | Harmonic mean of precision and recall at optimised threshold. |
| **Brier score** | Calibration quality (lower is better). |

### Comparing Regime A vs Regime B

- **Regime A gap between models** = difference in backbone representation quality
- **Regime B gap between models** = difference in practical system performance
- **Regime B - Regime A for same model** = how much the biological features add on top of the backbone

If Regime A scores are similar across models but Regime B diverges, the explicit features are doing the heavy lifting and the backbone choice matters less.

---

## Hyperparameter Reference

All values below are the defaults used for the fair comparison.  Override via CLI flags.

| Parameter | Flag | Default | Notes |
|-----------|------|---------|-------|
| LoRA rank | `--lora-rank` | 4 | Same for all models |
| LoRA alpha | `--lora-alpha` | 8.0 | |
| LoRA dropout | `--lora-dropout` | 0.1 | |
| Backbone LR | `--lr-backbone` | 5e-6 | For LoRA + LayerNorm params |
| Head LR | `--lr-head` | 5e-4 | |
| Backbone weight decay | `--wd-backbone` | 0.01 | |
| Head weight decay | `--wd-head` | 1e-4 | |
| Batch size | `--batch-size` | 4 | Per device |
| Gradient accumulation | `--grad-accum-steps` | 16 | Effective batch = batch_size * accum * n_gpus |
| Max epochs | `--max-epochs` | 5 | Best model saved by MCC |
| Warmup fraction | `--warmup-fraction` | 0.10 | Of total training steps |
| Gradient clip | `--grad-clip` | 0.5 | Max gradient norm |
| Backbone freeze steps | `--freeze-backbone-steps` | 100 | Head-only warmup |
| Loss | `--loss-type` | focal | `focal` or `bce` |
| Focal gamma | `--focal-gamma` | 2.0 | Only used with focal loss |
| Pos weight | `--pos-weight` | 1.0 | `auto` for class-imbalance scaling |
| Context size | `--context-size` | 4096 | Window around each variant |
| Projection dim | `--proj-dim` | 256 | Common dim after d_model projection |
| Hidden dim | `--hidden-dim` | 128 | MLP hidden layer width |
| Head dropout | `--head-dropout` | 0.2 | |

---

## W&B Tracking

```bash
python -m eval.clinvar.run --regime A \
    --model-family lumina --model-version beat-v2 \
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/lumina_beat_v2_A \
    --wandb-enabled \
    --wandb-project lumina-clinvar \
    --wandb-tags regime-A lumina beat-v2
```

Logged metrics: `train/loss`, `train/lr_backbone`, `train/lr_head`, `eval/auroc`, `eval/auprc`, `eval/mcc`, `eval/f1`, `final/*`.

---

## SageMaker

The SageMaker dispatch scripts need updating to use the new `eval.clinvar.run` entry point.  The container entrypoint should call:

```bash
torchrun --nproc_per_node $SM_NUM_GPUS -m eval.clinvar.run \
    --regime $REGIME \
    --model-family $MODEL_FAMILY \
    --model-version $MODEL_VERSION \
    --checkpoint-path $CHECKPOINT_PATH \
    --fasta-path /opt/ml/input/data/training/hg38/hg38.fa \
    --phylo100-bw-path /opt/ml/input/data/training/phylo/hg38.phyloP100way.bw \
    --phylo470-bw-path /opt/ml/input/data/training/phylo/hg38.phyloP470way.bw \
    --gtf-path /opt/ml/input/data/training/gencode/gencode.v38.annotation.gtf.gz \
    --dataset-path /opt/ml/input/data/training/datasets/clinvar/processed/clinvar_dataset.parquet \
    --output-dir /opt/ml/output/data \
    --wandb-enabled
```

---

## Caching

Variant window extraction from FASTA is expensive on first run.  The pipeline caches extracted windows (and biological features for Regime B) to a parquet file alongside the dataset.

- Cache files are named by a deterministic hash of `(dataset_path, context_size, regime)`
- Regime A and B have separate caches (Regime B includes bio features)
- To force a rebuild, delete the cache parquet or pass `--overwrite`
- Cache location can be customised with `--cache-dir`

---

## Troubleshooting

**CUDA out of memory**: Reduce `--batch-size` or `--context-size`.  Gradient accumulation preserves the effective batch size.

**Slow first run**: The variant window cache is being built.  Subsequent runs with the same dataset/context reuse the cache.

**Regime B validation error**: Ensure `--phylo100-bw-path`, `--phylo470-bw-path`, and `--gtf-path` are all provided.

**DNABERT-2 Triton warning**: Expected on systems without Triton.  The adapter falls back to PyTorch attention automatically.

**Model version not found**: Check available versions in `adapters.py` — `NTV3_REPOS`, `CADUCEUS_REPOS`, `DNABERT2_REPOS`.  Lumina versions correspond to model registry keys.
