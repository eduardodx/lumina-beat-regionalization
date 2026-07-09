# Scripts

This directory contains setup helpers and SageMaker launch utilities.

## Recommended Entry Points

- `setup-macos.sh`: recommended local environment setup on Apple Silicon
- `setup-gpu.sh`: environment bootstrap inside Linux/CUDA containers
- `sagemaker_train.py`: canonical config-driven pretraining launcher
- `sagemaker_clinvar_finetune.py`: canonical ClinVar SageMaker launcher
- `sagemaker_ntv3_benchmark.py`: canonical NTv3 benchmark SageMaker launcher
- `dispatch_clinvar_finetune_b200.sh`: convenience wrapper for ClinVar fine-tuning
- `download_encode.sh`: download ENCODE / Roadmap BigWigs from a manifest into `data/encode/`

## Legacy Or Convenience Launchers

- `dispatch_beat_v1_b200.sh`: older convenience wrapper for a specific beat-v1 training config
- `dispatch_ntv3_beat_v5_smoke.sh`: diagnostic NTv3 smoke A, single-process and on-demand by default
- `dispatch_ntv3_beat_v5_smoke_b.sh`: diagnostic NTv3 smoke B, DDP with workers disabled
- `dispatch_ntv3_beat_v5_smoke_c.sh`: diagnostic NTv3 smoke C, DDP with the stable worker configuration
- `dispatch_ntv3_beat_v5_full.sh`: convenience wrapper for the stable NTv3 full run on the successful beat-v5 checkpoint
- `dispatch_ntv3_beat_v5_coupled.sh`: experimental beat-v5-coupled NTv3 run using decoder features and discriminative LRs
- `dispatch_ntv3_beat_v6_full.sh`: convenience wrapper for the official-like NTv3 full run on the pure beat-v6 base checkpoint
- `dispatch_ntv3_beat_v7_full.sh`: primary NTv3 full run on the pure beat-v7 base checkpoint
- `dispatch_ntv3_beat_v7_mlp_head.sh`: beat-v7 ablation that changes only the NTv3 functional head to an MLP
- `dispatch_ntv3_beat_v7_local_conv_head.sh`: beat-v7 ablation that changes only the NTv3 functional head to an MLP plus local depthwise context
- `dispatch_ntv3_beat_v7_bio_readout_phylo_structure.sh`: beat-v7 ablation that keeps the MLP head and adds frozen phylo/structure predictions from Lumina as auxiliary readout features
- `dispatch_ntv3_beat_v7_gated_hybrid_bio_readout.sh`: beat-v7 ablation that keeps phylo/structure readout and adds a gated local residual branch per track
- `dispatch_ntv3_beat_v7_context_pyramid.sh`: beat-v7 diagnostic that combines the best global-context recipe with ungated multi-scale dilated context inside the NTv3 head

When writing new operational documentation, prefer the Python launchers over one-off shell wrappers.

## NTv3 Beat-v7 Protocol

Use `scripts/dispatch_ntv3_beat_v7_full.sh` as the main NTv3 path. The wrapper
passes the current best-supported recipe explicitly instead of relying on the
hard preset: `feature_source=hidden`, full backbone fine-tuning, batch global
32, `fp32`, modified square decay, warmup 598, and no discriminative LR groups.
The previous beat-v5 decoder-coupled experiment was stable but underperformed,
so do not port that recipe to beat-v7 without an explicit ablation.

Use `scripts/dispatch_ntv3_beat_v7_mlp_head.sh` for the first controlled
beat-v7 ablation after reproducing the linear-head baseline. It preserves the
same data, optimizer schedule, batch schedule, checkpoint, and feature source,
and sets `FUNCTIONAL_HEAD_TYPE=mlp`.

Use `scripts/dispatch_ntv3_beat_v7_local_conv_head.sh` after the MLP-head run.
It keeps the same recipe and sets `FUNCTIONAL_HEAD_TYPE=local-conv` with
`FUNCTIONAL_HEAD_KERNEL_SIZE=15`, testing whether explicit local signal shape
improves the benchmark without changing the backbone or NTv3 data protocol.

Use `scripts/dispatch_ntv3_beat_v7_bio_readout_phylo_structure.sh` after
comparing MLP and local-conv. It intentionally returns to the stronger MLP head
and sets `FUNCTIONAL_HEAD_AUX_FEATURES=phylo-structure`, testing whether
Lumina's own frozen phylo/structure token predictions help the benchmark head
condition on biological modalities already learned during pretraining.

Use `scripts/dispatch_ntv3_beat_v7_gated_hybrid_bio_readout.sh` after the
bio-readout run. It sets `FUNCTIONAL_HEAD_TYPE=gated-hybrid` and preserves
`FUNCTIONAL_HEAD_AUX_FEATURES=phylo-structure`, testing whether a conservative
track-gated local residual branch can improve broad regulatory assays without
repeating the local-conv regression on PRO-cap.

Use `scripts/dispatch_ntv3_beat_v7_context_pyramid.sh` to test the next
readout-only beat-v7 strategy after global-context. It preserves the current
best 20k recipe and NTv3-compliant data/loss/evaluation path, sets
`FUNCTIONAL_HEAD_TYPE=context-pyramid`, keeps `phylo-structure` auxiliary
readouts, and adds ungated multi-scale dilated branches plus the pooled Lumina
window embedding. This specifically probes whether RNA/ATAC/Histone gaps are
from missing spatial context in the readout rather than a benchmark-pipeline
change.

## Beat-v7 SageMaker Note

`scripts/sagemaker_train.py` auto-enables a `flash-attn` install for configs whose resolved
`model` is `beat-v7`. The container bootstrap uses `pip install flash-attn --no-build-isolation`
after torch is active, and you can override the package or build settings via environment
variables such as `FLASH_ATTN_SPEC`, `FLASH_ATTENTION_FORCE_BUILD`, `MAX_JOBS`, and `NVCC_THREADS`.

## Beat-v7 Smoke Checks

Local import, lint, type-check, and unit coverage all work on macOS. If the Apple Silicon
environment exposes a `Mamba3`-capable package, the end-to-end sanity check is:

```bash
uv run --no-sources-package mamba-ssm python -m src.sanity \
  --model beat-v7 \
  --fasta-path data/hg38/hg38.fa \
  --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
  --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
  --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
  --seq-len 4096 \
  --batch-size 2 \
  --num-workers 0
```

To include ENCODE track regression targets in the same sanity pass, first populate
`data/encode/track_manifest.tsv` from `data/encode/track_manifest.example.tsv`, run
`bash scripts/download_encode.sh`, then add `--with-encode-tracks`.
