# ClinVar Fine-Tuning: Two-Regime Evaluation Design

## Motivation

ClinVar fine-tuning on beat-v1 showed that the compact BiMamba3-RC backbone is strong on splice and intronic signal but weak on coding-region consequence discrimination.  Beat-v2 addresses this gap with codon-aware pretraining objectives, but the downstream evaluation framework itself needs to be redesigned to:

1. **Fairly compare** Lumina against external baselines (DNABERT-2, NTv3, Caduceus)
2. **Diagnose** whether improvements come from better representations or from architectural tricks in the classification head
3. **Stabilise training** — the previous head (HybridMultiscalePerturbationHead) produced chaotic loss curves with no convergence trend

This rewrite replaces the previous eval/clinvar pipeline entirely with a two-regime framework.

---

## Two Evaluation Regimes

### Regime A — Representation Quality

**Purpose:** Which backbone learned better genomic representations?

- **Embedding-only head**: `[proj(site_ref), proj(site_delta), proj(local_context)]` → MLP → logit
- **d_model normalisation**: All models project to a common `proj_dim=256` before the shared MLP, equalising information bandwidth regardless of backbone hidden dimension
- **No biological features**: The head sees only what the backbone encodes
- **Identical LoRA config**: rank=4, alpha=8, all linear layers, same for all models

This isolates backbone signal.  If a model scores well here, its embeddings genuinely carry allele-sensitivity information.

### Regime B — Practical Utility

**Purpose:** Best ClinVar performance you can get from each backbone?

- **Two-stream head**: Same embedding stream as Regime A + a biological feature encoder
- **Explicit biological features** (60 dims): PhyloP conservation, CDS/codon annotations, AA change type, BLOSUM62, GC content, variant type
- **Same LoRA and training config** as Regime A

This tests the full system: backbone + biological priors.  The explicit features inject coding-region knowledge that compact backbones may lack.

### Why Both Regimes Matter

| Outcome | Interpretation |
|---------|---------------|
| Lumina wins in A + B | Strongest story: better representations AND better practical utility |
| Lumina wins in B only | Backbone isn't contributing much beyond the explicit features |
| Lumina wins in A only | Good backbone, but the bio-feature integration needs work |
| External model wins both | Lumina's pretraining needs more work before clinical claims |

---

## Architecture

### Shared Components (Identical Across Models)

| Component | Configuration |
|-----------|--------------|
| LoRA | rank=4, alpha=8.0, dropout=0.1, all linear layers |
| Adapter mode | lora_plus_norm (LoRA + LayerNorm fine-tuning) |
| d_model projection | Linear(d_model → 256) — normalises feature bandwidth |
| Optimizer | AdamW, backbone_lr=5e-6, head_lr=5e-4 |
| Schedule | Cosine annealing, 10% warmup, min_lr_ratio=0.01 |
| Backbone freeze | First 100 steps (head warms up before LoRA adapts) |
| Gradient clipping | max_norm=0.5 |
| Loss | Focal loss (gamma=2.0) — down-weights easy examples |
| Batch size | 4 per device, 16 gradient accumulation steps |
| Epochs | 30 max |

### Regime A Head (RegimeAHead)

```
Input: site_ref [B, D], site_delta [B, D], local_context [B, D]
  ↓
Linear(D → 256) projection (shared weights across all three)
  ↓
Concat [B, 768]
  ↓
Linear(768 → 128) → LayerNorm → GELU → Dropout(0.2)
  ↓
Linear(128 → 128) → LayerNorm → GELU → Dropout(0.2)
  ↓
Linear(128 → 1)  →  logit
```

### Regime B Head (RegimeBHead)

```
Stream 1 (embeddings):
  [site_ref, site_delta, local_context] → proj(D→256) → concat [768]
  → Linear(768→128) → LN → GELU → Drop(0.2)  →  [128]

Stream 2 (biological features):
  bio_features [60]
  → Linear(60→64) → LN → GELU → Drop(0.1)  →  [64]

Fusion:
  concat [192]
  → Linear(192→128) → LN → GELU → Drop(0.2)
  → Linear(128→1)  →  logit
```

### Embedding Features

For each variant, three embedding vectors are extracted from the backbone:

1. **site_ref** — Reference-allele embedding at the variant token position
2. **site_delta** — `alt_embedding[variant_pos] - ref_embedding[variant_pos]`
3. **local_context** — Mean-pooled reference embeddings in a ±64bp window around the variant

### Biological Features (Regime B only, 60 dimensions)

| Feature | Dims | Source |
|---------|------|--------|
| PhyloP100 at variant | 1 | BigWig |
| PhyloP470 at variant | 1 | BigWig |
| is_CDS | 1 | GTF |
| CDS phase one-hot | 4 | GTF (codon position 0/1/2 + non-CDS) |
| Reference AA one-hot | 22 | GTF + FASTA + genetic code |
| Alternate AA one-hot | 22 | GTF + FASTA + genetic code |
| AA change type one-hot | 4 | Derived (non-coding / syn / missense / nonsense) |
| BLOSUM62 score | 1 | Standard substitution matrix |
| GC content (±32bp) | 1 | Sequence |
| Variant type one-hot | 3 | Ref/Alt lengths (SNV / ins / del) |

These features are computed once during cache building and are identical for all models.

---

## Training Stability Improvements

The previous system exhibited severe training instability (loss oscillating 0.1–1.75 with no convergence).  This design addresses the root causes:

1. **Lower LR** — backbone 5e-6 (was 1e-5), head 5e-4 (was 1e-3)
2. **Longer warmup** — 10% (was 5%)
3. **Backbone freeze period** — 100 steps of head-only training to establish a stable gradient signal before LoRA kicks in
4. **Tighter gradient clipping** — 0.5 (was 1.0)
5. **Focal loss** — Down-weights easy examples, concentrates on ambiguous coding variants
6. **Simpler head** — Far fewer parameters, more stable gradients
7. **Lower LoRA rank** — 4 (was 8), reducing overfitting surface

---

## Supported Models

| Family | Versions | d_model | Tokenisation |
|--------|----------|---------|-------------|
| Lumina | beat-v1, beat-v2 | 256 | Character-level |
| NTv3 | 8M_pre, 100M_pre, 650M_pre | varies | Character-level |
| Caduceus | caduceus-ph, caduceus-ps | 256 | Character-level |
| DNABERT-2 | 117M | 768 | BPE (~3nt/token) |

All models go through the same adapter interface and receive the same LoRA rank, head architecture, and training hyperparameters.

---

## File Map

| File | Purpose |
|------|---------|
| `config.py` | Unified FineTuneConfig dataclass |
| `adapters.py` | Gradient-enabled adapters for all 4 model families |
| `lora.py` | LoRA implementation with uniform rank |
| `bio_features.py` | Biological feature extraction (PhyloP, CDS, AA, BLOSUM62) |
| `heads.py` | RegimeAHead and RegimeBHead |
| `model.py` | End-to-end model (backbone → features → head) |
| `dataset.py` | Data loading, variant window caching, bio feature caching |
| `losses.py` | BCE and focal loss |
| `metrics.py` | Classification metrics (AUROC, AUPRC, MCC, F1, etc.) |
| `train.py` | Training loop with DDP, freeze schedule, checkpointing |
| `run.py` | CLI entry point |
| `variant_utils.py` | Genomic window extraction from FASTA |

---

## Usage

### Single GPU

```bash
# Regime A — Lumina
python -m eval.clinvar.run --regime A \
    --model-family lumina --model-version beat-v2 \
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/lumina_beat_v2_A

# Regime B — Lumina
python -m eval.clinvar.run --regime B \
    --model-family lumina --model-version beat-v2 \
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
    --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
    --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
    --output-dir outputs/clinvar/lumina_beat_v2_B

# Regime A — NTv3
python -m eval.clinvar.run --regime A \
    --model-family ntv3 --model-version 8M_pre \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/ntv3_8m_A

# Regime A — Caduceus
python -m eval.clinvar.run --regime A \
    --model-family caduceus --model-version caduceus-ph \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/caduceus_ph_A

# Regime A — DNABERT-2
python -m eval.clinvar.run --regime A \
    --model-family dnabert2 --model-version 117M \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/dnabert2_117m_A
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node 8 -m eval.clinvar.run \
    --regime A --model-family lumina --model-version beat-v2 \
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --output-dir outputs/clinvar/lumina_beat_v2_A
```

### Full Comparison Script

```bash
#!/bin/bash
# Run all models in both regimes for complete comparison
FASTA=data/hg38/hg38.fa
PHYLO100=data/phylo/hg38.phyloP100way.bw
PHYLO470=data/phylo/hg38.phyloP470way.bw
GTF=data/gencode/gencode.v38.annotation.gtf.gz
CKPT=outputs/beat_v2/best_checkpoint.pt

# Regime A (representation quality)
for model_args in \
    "lumina beat-v2 --checkpoint-path $CKPT" \
    "ntv3 8M_pre" \
    "caduceus caduceus-ph" \
    "dnabert2 117M"; do
  set -- $model_args
  family=$1; version=$2; shift 2
  python -m eval.clinvar.run --regime A \
    --model-family $family --model-version $version "$@" \
    --fasta-path $FASTA \
    --output-dir outputs/clinvar/${family}_${version}_A
done

# Regime B (practical utility)
for model_args in \
    "lumina beat-v2 --checkpoint-path $CKPT" \
    "ntv3 8M_pre" \
    "caduceus caduceus-ph" \
    "dnabert2 117M"; do
  set -- $model_args
  family=$1; version=$2; shift 2
  python -m eval.clinvar.run --regime B \
    --model-family $family --model-version $version "$@" \
    --fasta-path $FASTA \
    --phylo100-bw-path $PHYLO100 \
    --phylo470-bw-path $PHYLO470 \
    --gtf-path $GTF \
    --output-dir outputs/clinvar/${family}_${version}_B
done
```

---

## Output Artifacts

Each run produces in `--output-dir`:

| File | Content |
|------|---------|
| `metrics.json` | All evaluation metrics, hyperparameters, architecture info |
| `test_predictions.parquet` | Per-variant probabilities and labels |
| `best_model.pt` | Full model checkpoint (LoRA + head weights) |

The `metrics.json` includes:
- Primary metric: MCC (under `"value"` key)
- AUROC, AUPRC, F1, balanced accuracy, precision, recall, specificity
- Brier score and log-loss (calibration)
- Metrics at both the optimised threshold and the default 0.5 threshold
- Full hyperparameter and architecture metadata for reproducibility

---

## Fairness Guarantees

1. **Same LoRA rank** (4) applied to all linear layers in all models
2. **Same d_model projection** (→ 256) normalises feature bandwidth
3. **Same head architecture** and parameter count (minus projection layer)
4. **Same optimizer, schedule, loss, and training budget**
5. **Same data** — identical ClinVar splits and variant windows
6. **Same biological features** in Regime B — model-independent
7. **Reported trainable parameter counts** for transparency
