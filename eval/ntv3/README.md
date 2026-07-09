# NTv3 Benchmark

This package fine-tunes and evaluates Lumina checkpoints on the
NTv3 benchmark using the official NTv3 dataset layout and metrics:

- `functional`: per-track Pearson correlation on `functional_tracks/*.bigwig`
- `annotation`: per-element MCC on `genome_annotation/*.bed`

The current stable default is the strongest completed `beat-v7` NTv3 run:

- model version: `beat-v7`
- checkpoint source: SageMaker model artifact
  `s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beat-v7-12m-15ep-32k/lumina-ssm-beat-v7-12m-15ep-32k-20260423192051/output/model.tar.gz`

## Local layout

- Checkpoint bundle:
  `data/checkpoints/ntv3/lumina-ssm-beat-v7-12m-15ep-32k-20260423192051/`
- Dataset root:
  `data/datasets/ntv3/`
- Outputs:
  `outputs/ntv3/<species>/<task>/`

## Canonical commands

Stage the Lumina checkpoint locally. The default source is a SageMaker
`model.tar.gz`; `stage-checkpoint` extracts `best_checkpoint.pt` and the
associated metadata into the local checkpoint directory:

```bash
uv run python -m eval.ntv3.run stage-checkpoint
```

Download the NTv3 dataset locally:

```bash
uv run python -m eval.ntv3.run stage-dataset
```

The runner loads `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` from the repo-local
`.env` automatically when present.

Run a single species/task pair:

```bash
uv run python -m eval.ntv3.run evaluate-species \
  --species human \
  --task-type functional \
  --train-backbone \
  --wandb-enabled \
  --wandb-project lumina-ntv3 \
  --wandb-entity ai4bio-lumina
```

For the stable `beat-v7` path, the CLI defaults point at the strongest
completed NTv3 model/checkpoint:

- `--model-version beat-v7`
- `--precision auto`
- `--num-workers 6`
- `--prefetch-factor 4`

The SageMaker launcher no longer forces the official NTv3 preset. Add
`--official-human-functional` explicitly only when you intentionally want the
official reproduction schedule.

Experimental beat-v5 coupling runs should not use `--official-human-functional`.
They keep the NTv3 data and evaluation protocol fixed, but change the adaptation
recipe with `--feature-source decoder`, discriminative optimizer LRs, and
`--head-only-warmup-steps`.

Do not carry the beat-v5 decoder-coupled recipe over to beat-v7 by default.
Beat-v7 token heads consume the final hidden states directly, and the completed
beat-v7 official-like run outperformed the coupled beat-v5 run. Use
`--feature-source hidden`, a single LR, and the official-like schedule as the
mainline; vary one scientific variable at a time after that baseline.

The first architecture-aware beat-v7 ablation keeps that recipe fixed and
changes only the fresh NTv3 functional head:

- baseline: `--functional-head-type linear`
- ablation: `--functional-head-type mlp`
- next ablation: `--functional-head-type local-conv`

The MLP and local-conv heads are still trained only on official NTv3 train data
and preserve the same validation/test protocol, targets, crop, and
single-nucleotide output resolution. The local-conv variant adds an explicit
depthwise 1D context window on top of the MLP trunk while keeping the beat-v7
backbone and hidden feature source fixed.

The next beat-v7 ablation keeps the stronger MLP readout and adds a
`bio-readout` auxiliary path:

- `--functional-head-type mlp`
- `--functional-head-aux-features phylo-structure`

This path concatenates hidden states with small projections of Lumina's own
predicted `phylo100`, `phylo470`, and structure logits. It does not consume
NTv3 validation/test labels, extra benchmark targets, or altered splits. The
pretraining heads used to emit those auxiliary predictions are preserved frozen
and in eval mode, so they act as fixed biological feature extractors while the
fresh NTv3 head and backbone adapt to the official functional-track objective.

The follow-up architecture-aware ablation keeps the same auxiliary `phylo-structure`
readout but changes the fresh NTv3 head to `--functional-head-type gated-hybrid`.
This head keeps an MLP base path and adds a depthwise local residual branch behind
a small learned gate per track. The goal is to let broad assays use local context
without forcing the same smoothing onto point-like tracks such as PRO-cap.

For the official-like human functional reproduction preset, the runner enforces
the public NTv3 notebook defaults:

- full fine-tuning (`--train-backbone`)
- `mini_batch_size=4`
- gradient accumulation `8`
- `19932` training steps
- `initial LR=1e-5`, `peak LR=5e-5`
- warmup `598`
- validation every `500` steps
- `seed=0`
- `num_workers=16`

Run the full benchmark and aggregate the leaderboard CSV:

```bash
uv run python -m eval.ntv3.run evaluate-all
```

For Weights & Biases tracking, install the optional dependency first:

```bash
uv sync --extra tracking
```

The NTv3 runner logs head-training metrics such as `train/loss`,
`train/learning_rate`, validation summary metrics, and final test metrics per
species/task. The multi-GPU launcher forwards the same W&B flags:

```bash
uv run python -m eval.ntv3.launch_multi_gpu \
  --output-root outputs/ntv3-seed42-multigpu \
  --run-id lumina_beat_v7_ntv3_seed42 \
  --checkpoint-dir data/checkpoints/ntv3/lumina-ssm-beat-v7-12m-15ep-32k-20260423192051 \
  --dataset-root data/datasets/ntv3 \
  --wandb-enabled \
  --wandb-project lumina-ntv3 \
  --wandb-entity ai4bio-lumina
```

The final leaderboard-compatible CSV is written to:

```text
outputs/ntv3/ntv3_benchmark_results.csv
```

## SageMaker

Launch the official-like `human functional` run on SageMaker:

```bash
uv run --extra sagemaker python scripts/sagemaker_ntv3_benchmark.py \
  --experiment ntv3-human-functional-official-seed0 \
  --bucket ai4bio-lumina-experiments-v2 \
  --detach
```

The launcher stages the Lumina checkpoint bundle from S3, downloads the gated
NTv3 dataset inside the container using `HF_TOKEN`, runs
`eval.ntv3.run evaluate-species --train-backbone`,
and uploads the benchmark artifacts back to S3 under:

```text
s3://<bucket>/lumina-ssm/eval/ntv3-benchmark/<experiment>/
```
