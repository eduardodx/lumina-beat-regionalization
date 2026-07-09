# Local Development

## Environment Policy

- Python support: `>=3.11,<3.13`
- Recommended interpreter: Python `3.11`
- User-facing Python commands should be run with `uv run`
- SageMaker Linux/CUDA remains the canonical environment for long training runs, distributed runs, and benchmark
  reporting

## Setup

### macOS (Apple Silicon)

The repository can be developed locally on macOS, and in the current checked environment:

- `mamba_ssm` imports successfully in `.venv`
- a lightweight local model forward pass succeeds

That is enough to treat macOS as a viable local development environment for import checks, lightweight forwards,
linting, type-checking, and non-distributed experiments. It is **not** the canonical environment for large training,
performance-sensitive evaluation, or published benchmark results.

Recommended setup:

```bash
bash scripts/setup-macos.sh dev
```

If you need ClinVar evaluation dependencies as well:

```bash
bash scripts/setup-macos.sh "dev,eval"
```

Why this path is recommended:

- `scripts/setup-macos.sh` uses `uv pip install -e ...`
- this avoids the universal-resolution issues that can cause `uv sync` on macOS to try building Linux-only packages

### Linux / CUDA

Recommended setup:

```bash
uv venv --python 3.11
uv sync --extra dev
```

Add optional extras as needed:

```bash
uv sync --extra dev --extra tracking
uv sync --extra dev --extra eval
uv sync --extra dev --extra sagemaker
```

## Verification Commands

Lint:

```bash
uv run ruff check .
```

Type check:

```bash
uv run pyrefly check --summarize-errors
```

Tests:

```bash
uv run pytest
```

Import-safety check:

```bash
uv run python -c "import src.dataset, src.model, src.train, src.sanity"
```

Lightweight sanity command:

```bash
uv run python -m src.sanity \
  --fasta-path data/hg38/hg38.fa \
  --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
  --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
  --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
  --model beat-v5 \
  --seq-len 1024 \
  --batch-size 2 \
  --num-workers 0
```

## Data Layout

Default local paths expected by training and sanity entry points:

- `data/hg38/hg38.fa`
- `data/phylo/hg38.phyloP100way.bw`
- `data/phylo/hg38.phyloP470way.bw`
- `data/gencode/gencode.v38.annotation.gtf.gz`
- `data/datasets/clinvar/processed/clinvar_dataset.parquet` for ClinVar work

If those assets are missing, keep verification import-safe and note any data-dependent checks that were not run.
