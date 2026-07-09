# ClinVar Regional M6/M4 Parallel Status

## M6 Explicit-Frequency Control

- Objective: test whether explicit ABRAOM/gnomAD features alone explain regional gains.
- Model: `F0 + A_path + explicit frequency features`, no population adapter.
- Code added:
  - `RegimeAPlusFeaturesHead` in `eval/clinvar/heads.py`
  - `explicit_feature_columns` in `FineTuneConfig` and CLI
  - explicit feature cache support in `eval/clinvar/dataset.py`
  - regional evaluator support for fine-tuned checkpoints that require explicit features
- Local validation:
  - `.venv/bin/ruff check ...`
  - `.venv/bin/python -m pytest tests/test_eval_clinvar_heads.py tests/test_eval_clinvar_dataset.py -q`
- SageMaker job:
  - `clinvar-m0-clinvar-m6-explicitfreq-23a613-20260622011931`
  - status at dispatch validation: `InProgress / Training`
  - output prefix: `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/clinvar-m6-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/`
- Feature vector:
  - `af_abraom`
  - `af_gnomad`
  - `specificity`
  - `abraom_present`
  - `is_snv`
  - `af_abraom_missing`
  - `af_gnomad_missing`
  - `log10_af_abraom`
  - `log10_af_gnomad`
  - `af_delta`
  - `af_abs_delta`
  - `af_ratio_log10`

## M4/M5 Adapter Fusion Prep

- Existing frequency adapters are LoRA checkpoints over the same BEAT-v10 backbone:
  - `artifacts/abraom_frequency_adapter/abraom-balanced-v1-rerun/best_adapter.pt`
  - `artifacts/abraom_frequency_adapter/gnomad-balanced-v1/best_adapter.pt`
  - `artifacts/abraom_frequency_adapter/scrambled-balanced-v1/best_adapter.pt`
- Each checkpoint stores `trainable_state_dict` keys under `backbone.*.lora_a` and `backbone.*.lora_b`.
- Directly loading multiple adapters into the current `LoRALinear` is not enough: the current wrapper supports only one LoRA update per layer.
- Required implementation for M4/M5:
  - multi-adapter LoRA wrapper or representation-level multi-branch fusion;
  - load frozen `A_path` from M0;
  - load frozen `A_BR` and `A_gnomAD`;
  - train only fusion gate and regional head;
  - M5 reuses the explicit feature vector implemented for M6.

## Next Gate

When M6 completes:

1. Download `model.tar.gz`.
2. Run regional evaluation with the existing evaluator.
3. Compare `M6` against `M0` on:
   - `nonbr_only`
   - `br_only`
   - `br_any`
   - `abraom_common_benign`
   - `abraom_pathogenic_present`
   - `abraom_pathogenic_common`
4. Use the result to decide whether M4/M5 must beat an already-strong explicit-frequency baseline.
