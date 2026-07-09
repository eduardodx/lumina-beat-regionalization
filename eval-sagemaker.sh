# Regime A

# Results: s3://ai4bio-lumina-experiments-v2/lumina-ssm/eval/clinvar/finetune/clinvar-ntv3-8M_pre-regime-a-32k/
# SageMaker artifacts: s3://ai4bio-lumina-experiments-v2/lumina-ssm/sagemaker-artifacts/clinvar-finetune/clinvar-ntv3-8M_pre-regime-a-32k/
## results = s3://ai4bio-lumina-experiments-v2/lumina-ssm/eval/clinvar/finetune/clinvar-lumina-beat-v2-regime-a-32k/

UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
  --experiment clinvar-lumina-beat-v2-regime-a-32k \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach \
  --checkpoint-dir s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/checkpoints/lumina-ssm-beat-v2-9m-15ep-32k-b200-20260401155503 \
  -- \
  --regime A \
  --model-family lumina \
  --model-version beat-v2 \
  --context-size 32768 \
  --batch-size 5 \
  --grad-accum-steps 6 \
  --wandb-enabled \
  --wandb-project lumina-ssm \
  --wandb-entity ai4bio-lumina

UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
  --experiment clinvar-ntv3-8M_pre-regime-a-32k \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach \
  -- \
  --regime A \
  --model-family ntv3 \
  --model-version 8M_pre \
  --context-size 32768 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --wandb-enabled \
  --wandb-project lumina-ssm \
  --wandb-entity ai4bio-lumina

UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
  --experiment clinvar-caduceus-ph-regime-a-32k \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach \
  -- \
  --regime A \
  --model-family caduceus \
  --model-version caduceus-ph \
  --context-size 32768 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --wandb-enabled \
  --wandb-project lumina-ssm \
  --wandb-entity ai4bio-lumina

UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
  --experiment clinvar-caduceus-ps-regime-a-32k \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach \
  -- \
  --regime A \
  --model-family caduceus \
  --model-version caduceus-ps \
  --context-size 32768 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --wandb-enabled \
  --wandb-project lumina-ssm \
  --wandb-entity ai4bio-lumina

# UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
#   --experiment clinvar-dnabert-2-regime-a-4k \
#   --bucket ai4bio-lumina-experiments-v2 \
#   --instance-type ml.p6-b200.48xlarge \
#   --spot \
#   --detach \
#   -- \
#   --regime A \
#   --model-family dnabert2 \
#   --model-version 117M \
#   --context-size 4096 \
#   --batch-size 8 \
#   --grad-accum-steps 4 \
#   --wandb-enabled \
#   --wandb-project lumina-ssm \
#   --wandb-entity ai4bio-lumina

# Regime B

## results = s3://ai4bio-lumina-experiments-v2/lumina-ssm/eval/clinvar/finetune/clinvar-lumina-beat-v2-regime-a-32k/

UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
  --experiment clinvar-lumina-beat-v2-regime-b-32k \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach \
  --checkpoint-dir s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/checkpoints/lumina-ssm-beat-v2-9m-15ep-32k-b200-20260401155503 \
  -- \
  --regime B \
  --model-family lumina \
  --model-version beat-v2 \
  --context-size 32768 \
  --batch-size 5 \
  --grad-accum-steps 6 \
  --wandb-enabled \
  --wandb-project lumina-ssm \
  --wandb-entity ai4bio-lumina

UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
  --experiment clinvar-ntv3-8M_pre-regime-b-32k \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach \
  -- \
  --regime B \
  --model-family ntv3 \
  --model-version 8M_pre \
  --context-size 32768 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --wandb-enabled \
  --wandb-project lumina-ssm \
  --wandb-entity ai4bio-lumina

UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
  --experiment clinvar-caduceus-ph-regime-b-32k \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach \
  -- \
  --regime B \
  --model-family caduceus \
  --model-version caduceus-ph \
  --context-size 32768 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --wandb-enabled \
  --wandb-project lumina-ssm \
  --wandb-entity ai4bio-lumina

UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
  --experiment clinvar-caduceus-ps-regime-b-32k \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach \
  -- \
  --regime B \
  --model-family caduceus \
  --model-version caduceus-ps \
  --context-size 32768 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --wandb-enabled \
  --wandb-project lumina-ssm \
  --wandb-entity ai4bio-lumina

# UV_NO_SOURCES_PACKAGE=mamba-ssm uv run python scripts/sagemaker_clinvar_finetune.py \
#   --experiment clinvar-dnabert-2-regime-b-4k \
#   --bucket ai4bio-lumina-experiments-v2 \
#   --instance-type ml.p6-b200.48xlarge \
#   --spot \
#   --detach \
#   -- \
#   --regime B \
#   --model-family dnabert2 \
#   --model-version 117M \
#   --context-size 4096 \
#   --batch-size 8 \
#   --grad-accum-steps 4 \
#   --wandb-enabled \
#   --wandb-project lumina-ssm \
#   --wandb-entity ai4bio-lumina