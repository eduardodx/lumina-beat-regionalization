# Regional ClinVar ABRAOM Dataset

This local preparation step joins three existing artifacts without regenerating
them:

- legacy ClinVar pathogenicity splits from `/home/sagemaker-user/lumina`
- ClinVar Brazilian/non-Brazilian submitter metadata from `/home/sagemaker-user/lumina-benchmarks`
- ABRAOM v2 allele frequencies from `/home/sagemaker-user/gen-abraom-seqs`

Run from the `lumina-ssm` repository:

```bash
python scripts/prepare_regional_clinvar_dataset.py --overwrite
```

Default output directory:

```text
data/datasets/clinvar/regional_abraom/
```

Main outputs:

- `clinvar_regional_abraom_master.parquet`: all converted rows, including holdout.
- `clinvar_regional_abraom_train_test.parquet`: train/test rows in the schema expected by `eval.clinvar`.
- `clinvar_regional_abraom_holdout.parquet`: held-out rows preserved for later analysis.
- `regional_annotation_by_variant.parquet`: ClinVar regional submitter aggregation by variant.
- `abraom_matches.parquet`: ClinVar variants found in ABRAOM.
- `summary.json` and `README.md`: local QC summary.

The preparation keeps ABRAOM absence as missing frequency values plus
`abraom_present=false`; it does not treat absence from the filtered ABRAOM index
as population frequency zero.

## Next Regionalization Artifacts

The adapter-fusion blueprint requires a regional adapter target that is more
direct than continued MLM. The local preparation script below turns the existing
ABRAOM v2 index into variant-level train/val/test frequency targets:

```bash
python scripts/prepare_abraom_frequency_adapter_dataset.py --overwrite
```

Default output directory:

```text
data/datasets/abraom_frequency_adapter/
```

Main outputs:

- `abraom_frequency_train.parquet`
- `abraom_frequency_val.parquet`
- `abraom_frequency_test.parquet`
- `summary.json` and `README.md`

The split is inherited from
`/home/sagemaker-user/gen-abraom-seqs/data/production_v2/split_manifest.parquet`.
Rows include `af_abraom`, `af_gnomad`, `specificity`, logit targets, AF and
specificity bins, raw sampling weights, genomic block metadata, and a scrambled
ABRAOM frequency target for negative-control training.

Clinical regional evaluation slices can be materialized from the local
ClinVar x ABRAOM master table:

```bash
python scripts/build_regional_clinvar_eval_slices.py --overwrite
```

Default output directory:

```text
data/datasets/clinvar/regional_abraom/slices/
```

These slices cover Brazilian-only, non-Brazilian-only, mixed submitter evidence,
ABRAOM-present, high-specificity ABRAOM, common ABRAOM benign, and
pathogenic-present do-not-suppress checks.

## Train The ABRAOM Frequency Adapter

The first adapter proposed by the regionalization blueprint is `A_BR`: a
regional frequency adapter trained against ABRAOM allele frequencies. This is a
soft-label frequency objective, not a pathogenicity objective.

Smoke test locally:

```bash
python scripts/train_abraom_frequency_adapter.py \
  --max-train-rows 64 \
  --max-val-rows 32 \
  --max-test-rows 32 \
  --max-steps 2 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --context-size 512 \
  --precision fp32 \
  --output-dir outputs/abraom_frequency_adapter_smoke \
  --overwrite
```

Full training should run on SageMaker/GPU capacity, using the same script with
the full train/val/test parquets and `--context-size 4096`.

Main validation criteria:

- model NLL/Brier on held-out ABRAOM must improve over the gnomAD frequency
  baseline.
- Spearman correlation against held-out `af_abraom` should be positive and
  materially above the scrambled-target control.
- calibration by `af_abraom_bin` and `specificity_bin` should not show a narrow
  gain isolated to one easy frequency range.
- a negative-control run with `--target-column scrambled_af_abraom
  --metric-target-column af_abraom` should not reproduce the real-target
  improvement.

The script intentionally does not feed `specificity`, `delta_af`, or
`delta_logit` as input features because those columns are derived from the
ABRAOM target and would leak the answer. `--use-gnomad-prior` is available as an
explicit global-frequency prior, but it should be reported separately from the
sequence-only adapter.
