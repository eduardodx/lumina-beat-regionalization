# BiMamba3-RC Technical Reference

Technical decisions and design choices for the BiMamba3-RC model family, intended as the working reference for Paper 1.

---

## 1. Reference Genome and Annotations

### 1.1 Genome Assembly

- **Assembly**: GRCh38 (hg38), human reference genome
- **Source**: UCSC Genome Browser (`hg38.fa`)
- **Training scope**: Human-only, single-species. No multi-species pretraining.

### 1.2 Conservation Tracks

Two PhyloP tracks provide per-base evolutionary conservation scores:

- **PhyloP100way**: Conservation across 100 vertebrate species, with heavy primate weight. Primate-resolution conservation signal.
- **PhyloP470way**: Conservation across 470 species, deeper evolutionary time. Captures ancient functional constraint.

Both are loaded as BigWig files and read per-position for each sampled window.

**Normalization**: Values are clipped to [-10, +10] and divided by 10, yielding a normalized range of [-1, +1]. NaN, +inf, and -inf are replaced with 0.0 before clipping.

### 1.3 Gene Annotations

- **Source**: GENCODE v38 (`gencode.v38.annotation.gtf.gz`)
- **Feature extraction**: Exon records only, grouped by transcript
- **Transcript biotypes retained**: `protein_coding`, `lncRNA`, `processed_transcript`, `retained_intron`, `nonsense_mediated_decay`
- **Coordinate system**: BED-like 0-based half-open `[start, end)`

### 1.4 Splice Site Labels

Splice sites are derived from exon boundaries within multi-exon transcripts:

- For each consecutive exon pair, donor (3' end of upstream exon) and acceptor (5' start of downstream exon) positions are extracted
- **Core zone**: ±2 bp around each splice junction (label = `SPLICE_CORE`)
- **Region zone**: ±10 bp around each splice junction (label = `SPLICE_REGION`)
- **Background**: All other positions (label = `BACKGROUND`)
- Priority rule: core > region > background when zones overlap

Intervals are pre-merged per chromosome into sorted `IntervalIndex` structures with binary-search lookup for O(log n) window queries.

### 1.5 Coordinate Integrity

Before any training begins, a validation step confirms:
- Every training chromosome exists in FASTA, PhyloP100, and PhyloP470 sources
- Chromosome lengths are identical across all three sources
- GTF exon coordinates fall within validated chromosome bounds

This prevents silent coordinate mismatches between genome, conservation, and annotation data.

---

## 2. Tokenization and Vocabulary

- **Vocabulary size**: 8 tokens
- **Encoding**: Single-nucleotide, character-level. No k-mer or BPE tokenization.

| Token | ID | Role |
|---|---|---|
| PAD | 0 | Padding (shorter sequences or batch alignment) |
| A | 1 | Adenine |
| C | 2 | Cytosine |
| G | 3 | Guanine |
| T | 4 | Thymine |
| N | 5 | Ambiguous base |
| MASK | 6 | Masked position for MLM |
| UNK | 7 | Unknown |

- **Complement table**: `[0, 4, 3, 2, 1, 5, 6, 7]` — A(1)<->T(4), C(2)<->G(3), specials map to themselves
- Reverse complement is computed as: complement each token via lookup table, then reverse the sequence

---

## 3. Data Sampling

### 3.1 Chromosome Sampling

Chromosomes are sampled with probability proportional to their length in bases, so longer chromosomes contribute proportionally more training windows. Sampling uses a precomputed CDF with binary search.

**Default chromosome pool**: chr1–chr22 + chrX (23 chromosomes).

### 3.2 Train/Validation Split

Chromosome-level held-out split:
- **Validation chromosomes**: chr19, chr21, chr22, chrX
- **Training chromosomes**: All remaining (chr1–chr18, chr20)

No window-level or position-level splitting — entire chromosomes are held out, preventing any data leakage from positional overlap.

### 3.3 Window Sampling

Within a selected chromosome:
- A random start position is drawn uniformly from `[0, chrom_length - seq_len]`
- The window is `[start, start + seq_len)` — a contiguous, non-overlapping genomic segment

### 3.4 N-Content Filtering

Windows with high ambiguous-base content (centromeric, telomeric, and assembly-gap regions) are rejected:

- **Threshold**: `max_n_fraction = 0.25` (default)
- **Mechanism**: Up to 20 sampling attempts per window. Each attempt tracks the best (lowest N-fraction) candidate seen.
- If a window with N-fraction ≤ 0.25 is found within 20 attempts, it is used.
- If no acceptable window is found, the best candidate is used anyway (fallback).
- The `n_filter_fallback_used` flag is tracked per sample and aggregated across batches for observability.

This avoids both wasting compute on uninformative centromeric windows and creating systematic sampling gaps.

### 3.5 Batch Composition Logging

Each batch records and logs:
- `n_fraction`: mean ambiguous-base fraction across samples
- `splice_positive_fraction`: mean fraction of positions with any splice label (core or region)
- `splice_core_fraction`: mean fraction of positions with core splice label
- `exon_fraction`: mean fraction of positions overlapping exon intervals
- `n_filter_fallback_fraction`: fraction of samples that used the N-rejection fallback

---

## 4. Masking Strategy

### 4.1 Span Masking

MLM uses span-based masking rather than independent per-position masking:

- **Mask probability**: 15% of non-N, non-PAD positions
- **Span length**: Geometrically distributed with configurable mean (default `mean_span_len = 3`)
- **Span start**: Sampled uniformly (or conservation-weighted, see below)
- **Unmaskable positions**: N bases and PAD tokens are never masked

Span masking is harder than independent masking because the model cannot rely on immediately adjacent unmasked tokens for local k-mer context recovery.

### 4.2 Mask Application

All masked positions receive the `MASK` token (ID = 6). No BERT-style corruption mix (80% MASK / 10% random / 10% keep) — pure mask replacement only.

MLM labels are set to the original token ID at masked positions and `PAD_ID` elsewhere, so the cross-entropy loss is computed only at masked positions.

### 4.3 Conservation-Weighted Masking (Optional)

When `conservation_mix > 0.0`, span start positions are sampled from a mixed distribution that biases toward high-conservation positions:

- PhyloP100 values are clamped to `[0, +inf)` — only conserved positions receive elevated weight
- Unmaskable positions (N, PAD) are zeroed out
- Final sampling distribution: `conservation_mix * normalized_phylo + (1 - conservation_mix) * uniform`
- Default: `conservation_mix = 0.0` (pure uniform, backwards-compatible)
- Recommended starting value: `conservation_mix = 0.5`

**Rationale**: Standard uniform masking means most masked positions fall in non-conserved regions where nucleotide identity is biologically low-stakes. Conservation-weighted masking forces the model to predict nucleotide identity at high-PhyloP positions — exactly where pathogenic variants cluster — improving token-level allele sensitivity at functionally constrained positions.

---

## 5. Model Architecture

### 5.1 Architecture Family: BiMamba3-RC

Bidirectional Mamba3-based SSM with explicit reverse-complement strand processing.

### 5.2 Token Embedding

- `nn.Embedding(vocab_size=8, d_model=256, padding_idx=0)`
- Shared between forward and RC strands (same embedding weights)
- MLM head weight-ties to this embedding (transpose of embedding matrix)

### 5.3 Bidirectional Block Design

Each of the 8 blocks contains:

1. **Separate normalization**: `LayerNorm(d_model)` for forward strand, separate `LayerNorm(d_model)` for RC strand
2. **Forward mixer**: Mamba3 SSM processing the forward strand left-to-right
3. **RC mixer**: Mamba3 SSM processing the reverse-complement strand left-to-right, output then flipped back to forward coordinates
4. **Fusion**: Concatenate `[h_fwd, flip(h_rc)]` (512D) → `Linear(512, 256)` (no bias)
5. **Residual connection** + **Dropout**

This differs from the baseline BiMamba which processes a single sequence in both directions (flip the same input). BiMamba3-RC explicitly constructs the reverse-complement sequence and processes it natively, allowing the model to learn strand-specific features.

### 5.4 Mamba3 SSM Configuration

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 256 | Hidden dimension |
| `d_state` | 128 | SSM state dimension (2x baseline BiMamba's 64) |
| `expand` | 2 | Inner dimension expansion factor |
| `headdim` | 64 | Per-head dimension for attention-like mechanics |
| `ngroups` | 1 | Number of groups for grouped operations |
| `chunk_size` | 16 | Sequence chunk size for efficient computation |

### 5.5 Rotary Position Embeddings (RoPE)

- `rope_fraction = 1.0` — full RoPE applied to all 256 hidden dimensions
- Provides relative positional information within the SSM
- This is a departure from BiMamba3's default of 0.5 (partial RoPE)

### 5.6 MIMO (Multi-Input Multi-Output)

- `is_mimo = True`, `mimo_rank = 4`
- Parameter-efficient low-rank projection within each Mamba3 block
- Reduces parameter count while maintaining representational capacity
- Chunk size is adjusted when MIMO is active with bf16: `max(1, 64 // mimo_rank) = 16`

### 5.7 Output Projection Normalization

- `is_outproj_norm = True`
- Normalizes the output projection within each Mamba3 block
- Improves training stability, especially with MIMO active

### 5.8 Output Heads

All heads operate on the final hidden states `[B, L, 256]` after the last block's LayerNorm + Dropout:

| Head | Architecture | Output |
|---|---|---|
| **MLM** | `Linear(256, 8)` weight-tied to embedding | Per-position token logits |
| **PhyloP100** | `Linear(256, 256) → GELU → Dropout → Linear(256, 1)` | Per-position conservation regression |
| **PhyloP470** | `Linear(256, 256) → GELU → Dropout → Linear(256, 1)` | Per-position conservation regression |
| **Structure** | `Linear(256, 256) → GELU → Dropout → Linear(256, 3)` | Per-position splice classification |
| **Global Projection** | `Linear(256, 256) → GELU → Linear(256, 256)` | Mean-pooled sequence embedding (optional) |

### 5.9 Model Scale

- **Parameter tier**: ~8M trainable parameters
- **Design intent**: Compact enough for rapid iteration, controlled ablation, and accessible deployment, while large enough to learn meaningful biological representations

---

## 6. Training Objectives

### 6.1 Masked Language Modeling (MLM)

- Cross-entropy loss at masked positions only
- Labels are original token IDs; non-masked positions are ignored via `ignore_index=PAD_ID`
- **Metric**: MLM accuracy (argmax prediction vs. true token at masked positions)

### 6.2 PhyloP Conservation Regression

Two parallel regression heads predict per-position normalized PhyloP scores:

- **Loss**: Smooth L1 (Huber) loss
- **Supervision scope**: All valid positions (non-N, non-PAD) — dense supervision, not limited to masked positions
- **Metric**: Pearson correlation between predicted and true PhyloP values, accumulated across batches

### 6.3 Splice Structure Classification

3-class per-position classification (background / splice_core / splice_region):

- **Loss**: Cross-entropy with inverse-frequency class weighting
- **Class weights**: Computed per-batch as `total_positions / (num_classes_present * class_count)`, capped at 8.0
- **Supervision scope**: All valid positions (dense)
- **Metrics**: Per-class precision, recall, F1; aggregate splice-positive P/R/F1

### 6.4 Reverse Complement Consistency (Optional)

When `w_rc > 0.0`, a separate forward pass processes the RC-masked input:

- **Loss**: `1.0 - cosine_similarity(z_fwd, z_rc)` on mean-pooled sequence embeddings
- **Alternative**: MSE loss (configurable via `rc_loss_type`)
- Encourages strand-invariant global representations

**Note**: In the BiMamba3-RC architecture, RC handling is already structural (dual-strand processing in every block). The RC loss is an additional optional regularizer, not the primary mechanism for RC awareness.

### 6.5 Loss Weighting

Two modes:

**Fixed weighting** (when `use_uncertainty_weighting = False`):
- `total = w_mlm * L_mlm + w_phylo100 * L_phylo100 + w_phylo470 * L_phylo470 + w_structure * L_structure + w_rc * L_rc`

**Uncertainty weighting** (when `use_uncertainty_weighting = True`):
- Kendall et al. (2018) learned homoscedastic uncertainty
- Per-task learnable log-sigma parameters: `log_σ_mlm`, `log_σ_phylo100`, `log_σ_phylo470`, `log_σ_structure`
- `L_task_weighted = L_task * exp(-2 * log_σ) / 2 + log_σ`
- Uncertainty parameters are optimized jointly with model parameters (separate param group, no weight decay)
- Learned sigma values are logged for observability

---

## 7. Optimization

### 7.1 Optimizer

- **Algorithm**: AdamW
- **Learning rate**: 8e-4
- **Weight decay**: 0.1
- **Betas**: (0.9, 0.95)
- **Gradient clipping**: Max norm 1.0

### 7.2 Learning Rate Schedule

- **Scheduler**: Cosine annealing with linear warmup
- **Warmup**: 384 steps (linear ramp from 0 to `lr`)
- **Cosine decay**: From `lr` to `min_lr` over remaining steps
- **Minimum LR**: 8e-5 (10% of peak)
- Uncertainty log-sigma parameters are excluded from LR scheduling (fixed LR)

### 7.3 Batch Configuration

- **Batch size**: 8 per device
- **Sequence length**: 4,096 tokens
- **Tokens per step per device**: 32,768
- **Effective tokens per step** (multi-GPU): `batch_size * seq_len * world_size`

### 7.4 Training Duration

- **1 epoch** ≈ 3,840 steps (over the ~1 Gb effective hg38 training subset)
- **Standard run**: 5 epochs = 19,200 steps
- **Evaluation**: Every 1,024 steps
- **Checkpoint save**: Every 3,840 steps (once per epoch)

---

## 8. Evaluation

### 8.1 Validation Protocol

- Chromosome-level held-out split (chr19, chr21, chr22, chrX)
- 20 validation steps per evaluation round
- Same forward pass and loss computation as training (without gradient)
- All scalar stats averaged; all metric accumulators summed then derived

### 8.2 Biological Metrics

| Metric | Type | Scope |
|---|---|---|
| MLM accuracy | Classification accuracy | Masked positions only |
| PhyloP100 Pearson r | Correlation | All valid positions |
| PhyloP470 Pearson r | Correlation | All valid positions |
| Splice positive P/R/F1 | Binary (positive vs. background) | All valid positions |
| Splice core P/R/F1 | Per-class | All valid positions |
| Splice region P/R/F1 | Per-class | All valid positions |
| Splice background P/R/F1 | Per-class | All valid positions |

Pearson correlation is computed via streaming Welford-style accumulation (running sums of x, y, xx, yy, xy) to avoid materializing all predictions.

Splice metrics use a full confusion matrix accumulated across evaluation steps.

### 8.3 Best Checkpoint Selection

- **Selection metric**: `splice_positive_f1`
- **Tie-breaking**: Lower validation loss wins
- Best checkpoint is saved separately at `best_checkpoint.pt`

### 8.4 Observability

Training logs include:
- Per-task loss components (mlm, phylo100, phylo470, structure, rc)
- Gradient norm
- Step time and tokens/sec throughput
- Batch composition stats (N fraction, splice fraction, exon fraction)
- Chromosome frequency distribution
- Learned uncertainty sigma values (when uncertainty weighting is active)
- All biological metrics at each eval point
- Full metrics history in JSONL format

---

## 9. Infrastructure

### 9.1 Precision

- **Default**: `auto` — resolves to bf16 on CUDA, float32 on MPS/CPU
- **TF32**: Enabled by default on CUDA (`allow_tf32 = True`)
- Mixed precision via `torch.autocast` context

### 9.2 Distributed Training

- Auto-detected from `torchrun` environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`)
- `DistributedDataParallel` wrapping with per-rank device assignment
- Seed offset by rank for independent sampling across workers
- All-reduce for loss stats (averaged), count stats (summed), and metric accumulators
- Barriers for checkpoint I/O synchronization

### 9.3 Data Loading

- `IterableDataset` with infinite sampling (no epoch boundary)
- Worker-safe: file handles are lazily opened per worker; `__getstate__` clears handles for pickling
- Per-worker RNG seeded from PyTorch worker seed
- Configurable `num_workers` and `prefetch_factor` for I/O pipelining
- `persistent_workers = True` when workers > 0

### 9.4 Checkpointing

- Model state dict, optimizer state dict, step counter, config, and metadata
- Metadata includes: best checkpoint state, tokens seen, latest validation results, wandb run info
- Rolling checkpoint cleanup: keeps last N checkpoints (default 3)
- Resume support: restores model, optimizer, step, best state, and token count

### 9.5 Experiment Tracking

- Optional wandb integration (`--wandb-enabled`)
- Run name includes timestamp suffix for uniqueness
- All scalar stats and biological metrics logged per step/eval
- Config serialized as wandb config for experiment comparison

### 9.6 Config System

- YAML-based configs with `_inherit` chain resolution (up to depth 8)
- Deep merge for nested `model_config` overrides
- CLI flags override any config value
- Config validation against registered model specs
- Hyphen-to-underscore normalization for keys

---

## 10. Design Decisions and Rationale

### 10.1 Single-Nucleotide Tokenization

We use character-level tokenization (vocab size 8) rather than k-mer or BPE. This preserves single-nucleotide resolution for:
- Exact variant position identification (critical for ref/alt modeling)
- Per-position conservation regression
- Per-position splice classification
- No tokenization boundary artifacts at variant sites

### 10.2 Dense Auxiliary Supervision

PhyloP regression and splice classification are computed over all valid positions, not just masked positions. This provides the model with dense biological signal at every training step, rather than the sparse 15% coverage of MLM alone.

### 10.3 Structural RC vs. Loss-Based RC

BiMamba3-RC processes both strands structurally at every block (dual embedding, dual mixer, fusion). This is stronger than a loss-based RC consistency penalty on global embeddings because:
- Every layer explicitly sees both strand orientations
- The fusion learned per-block can capture strand-specific features
- RC awareness is built into the representation, not just regularized at the output

### 10.4 Uncertainty Weighting

Rather than manually tuning fixed loss weights across four tasks with different scales and dynamics, Kendall et al. (2018) uncertainty weighting learns per-task precision automatically. This avoids the combinatorial search over weight configurations and adapts as task difficulties change during training.

### 10.5 N-Rejection Sampling

Centromeric and telomeric regions contain long runs of Ns that provide no biological signal. Rather than hard-filtering these regions upfront (which would require maintaining a separate valid-region index), we use rejection sampling with a fallback. This is simpler, preserves uniform positional coverage within valid regions, and logs fallback frequency for monitoring.

### 10.6 Chromosome-Level Validation Split

Position-level or window-level splits risk data leakage from overlapping genomic context. Holding out entire chromosomes eliminates this. The validation set (chr19, chr21, chr22, chrX) covers chromosomes with diverse gene density and repetitive element content.

### 10.7 Conservation-Weighted Masking

Standard MLM is nucleotide-identity-agnostic in its masking distribution. Conservation-weighted masking preferentially challenges the model at positions under purifying selection — exactly the positions where pathogenic variants cluster. This is designed to improve allele sensitivity at functionally constrained sites without requiring a different pretraining objective.
