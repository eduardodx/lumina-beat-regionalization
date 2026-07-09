# ClinVar Evaluation

The repository includes a real ClinVar fine-tuning and evaluation pipeline under `eval/clinvar/`.

For the dataset curation trail, schema notes, and notebook inventory, see
[ClinVar Dataset Curation](clinvar-dataset.md).

It is currently organized around two regimes:

- **Regime A**: representation quality using backbone-derived features only
- **Regime B**: practical utility using backbone features plus explicit biological features

## Prerequisites

Install the required extras:

```bash
uv sync --extra dev --extra eval
```

Add tracking only if needed:

```bash
uv sync --extra dev --extra eval --extra tracking
```

Required files:

- `data/hg38/hg38.fa`
- `data/datasets/clinvar/processed/clinvar_dataset.parquet`
- `data/phylo/hg38.phyloP100way.bw` for Regime B
- `data/phylo/hg38.phyloP470way.bw` for Regime B
- `data/gencode/gencode.v38.annotation.gtf.gz` for Regime B

## Lumina Models

Supported Lumina model versions come from the model registry. At the time of this cleanup, those are:

- `beat-v1`
- `beat-v2`
- `beat-v3`
- `beat-v4`
- `beat-v5`

`beat-v5` is the active documented family. Older families remain available for reproducibility and ablations.

For Lumina specifically:

- `beat-v5` uses a single reference-window encoder pass by default during fine-tuning
- `site_ref` comes from backbone hidden states at the edited token
- `local_context` comes from pooled shared decoder states over `+-64 bp`
- `variant_repr` is built from pretrained allele-conditioned sequence features plus `ref_allele` and `alt_allele`
- older Lumina families and external baselines keep the paired ref/alt two-pass path

## Quick Start

Regime A:

```bash
uv run python -m eval.clinvar.run \
  --regime A \
  --model-family lumina \
  --model-version <model-version> \
  --checkpoint-path <path-to-best_checkpoint.pt> \
  --dataset-path data/datasets/clinvar/processed/clinvar_dataset.parquet \
  --fasta-path data/hg38/hg38.fa \
  --output-dir outputs/clinvar/<run-name>
```

Regime B:

```bash
uv run python -m eval.clinvar.run \
  --regime B \
  --model-family lumina \
  --model-version <model-version> \
  --checkpoint-path <path-to-best_checkpoint.pt> \
  --dataset-path data/datasets/clinvar/processed/clinvar_dataset.parquet \
  --fasta-path data/hg38/hg38.fa \
  --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \
  --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \
  --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \
  --output-dir outputs/clinvar/<run-name>
```

## Current Defaults

- precision default: `auto`
- TF32 default: enabled when supported and not explicitly disabled
- primary selection metric: MCC
- outputs:
  - `metrics.json`
  - `test_predictions.parquet`
  - `best_model.pt`

## External Baselines

The current pipeline also supports:

- `ntv3`
- `caduceus`
- `dnabert2`

Use the same `eval.clinvar.run` entry point with the corresponding `--model-family` and `--model-version`.

## Caching

The ClinVar pipeline caches extracted variant windows, and for Regime B it caches biological features alongside them.

- cache files are keyed by dataset path, context size, and regime
- cached windows are reused across runs
- cached records preserve `ref_seq`, `alt_seq`, `variant_offset`, `label`, `original_index`,
  `split_within_gene`, `consequence_bucket`, and `gene_symbol`
- Regime B adds a cached `bio_features` column
- use `--overwrite` or remove the cache file to force a rebuild

## Scope Notes

The ClinVar pipeline is implemented, but it should not yet be documented as a finished counterfactual validation story.

Still missing:

- same-locus benign/pathogenic control suites
- explicit ref<->alt swap diagnostics
- temporal evaluation
- regional validation and calibration

Those belong in the roadmap as active downstream work, not as established present-tense claims.

## SageMaker

For SageMaker dispatch, use the dedicated launcher documented in [SageMaker Ops](../ops/sagemaker.md).
