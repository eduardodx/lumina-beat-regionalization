#!/usr/bin/env bash
# Dispatch ClinVar fine-tuning on SageMaker via the dedicated ClinVar launcher.
#
# Expected for Lumina runs:
#   CHECKPOINT_DIR=s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking \
#   HEAD_HIDDEN_DIM=256 HEAD_DROPOUT=0.1 \
#   WANDB_ENABLED=true \
#   bash scripts/dispatch_clinvar_finetune_b200.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUCKET="${BUCKET:-ai4bio-lumina-experiments-v2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.p6-b200.48xlarge}"
DEFAULT_IMAGE_URI="085188779747.dkr.ecr.us-east-2.amazonaws.com/lumina/hf-training-fa2-b200:pt280-tr4562-cu129-py312-fa283-sm100"
TRAINING_IMAGE_URI="${TRAINING_IMAGE_URI:-${DEFAULT_IMAGE_URI}}"

EXPERIMENT="${EXPERIMENT:-clinvar-finetune-lumina-beat-v1-b200}"
MODEL_FAMILY="${MODEL_FAMILY:-lumina}"
MODEL_VERSION="${MODEL_VERSION:-beat-v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

if [[ -n "${ADAPTER_MODE+x}" ]]; then
    echo "ERROR: ADAPTER_MODE is no longer supported by the current ClinVar runner." >&2
    echo "       Remove ADAPTER_MODE from the environment and use the built-in LoRA path." >&2
    exit 1
fi

CONTEXT_SIZE="${CONTEXT_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-128}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.3}"
HEAD_TYPE="${HEAD_TYPE:-}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
LR_HEAD="${LR_HEAD:-1e-3}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
PAIRWISE_RANK_WEIGHT="${PAIRWISE_RANK_WEIGHT:-}"
PAIRWISE_RANK_MARGIN="${PAIRWISE_RANK_MARGIN:-0.5}"
SWAP_CONSISTENCY_WEIGHT="${SWAP_CONSISTENCY_WEIGHT:-}"
SWAP_CONSISTENCY_MARGIN="${SWAP_CONSISTENCY_MARGIN:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PRECISION="${PRECISION:-bf16}"
ALLOW_TF32="${ALLOW_TF32:-true}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-72}"
OVERWRITE="${OVERWRITE:-false}"
DATA_S3_PREFIX="${DATA_S3_PREFIX:-s3://${BUCKET}/lumina-ssm/data/}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lumina-ssm}"
WANDB_ENTITY="${WANDB_ENTITY:-ai4bio-lumina}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

if [[ -z "${HEAD_TYPE}" && "${MODEL_FAMILY}" == "lumina" && "${MODEL_VERSION}" == "beat-v7" ]]; then
    HEAD_TYPE="regime_a_v7"
fi
if [[ -z "${HEAD_TYPE}" && "${MODEL_FAMILY}" == "lumina" && "${MODEL_VERSION}" == "beat-v8" ]]; then
    HEAD_TYPE="regime_a_v8"
fi

if [[ "${MODEL_FAMILY}" == "lumina" && -z "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: CHECKPOINT_DIR is required when MODEL_FAMILY=lumina." >&2
    echo "       It must point to a container-local directory under /opt/ml/ or an S3 prefix" >&2
    echo "       under s3://<bucket>/lumina-ssm/data/checkpoints/." >&2
    exit 1
fi

CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/sagemaker_clinvar_finetune.py"
    --experiment "${EXPERIMENT}"
    --bucket "${BUCKET}"
    --instance-type "${INSTANCE_TYPE}"
    --training-image-uri "${TRAINING_IMAGE_URI}"
    --max-run-hours "${MAX_RUN_HOURS}"
    --data-s3-prefix "${DATA_S3_PREFIX}"
    --spot
    --detach
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --
    --model-family "${MODEL_FAMILY}"
    --model-version "${MODEL_VERSION}"
    --context-size "${CONTEXT_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --max-epochs "${MAX_EPOCHS}"
    --hidden-dim "${HEAD_HIDDEN_DIM}"
    --head-dropout "${HEAD_DROPOUT}"
    --lr-backbone "${LR_BACKBONE}"
    --lr-head "${LR_HEAD}"
    --lora-rank "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --pos-weight "${POS_WEIGHT}"
    --pairwise-rank-margin "${PAIRWISE_RANK_MARGIN}"
    --swap-consistency-margin "${SWAP_CONSISTENCY_MARGIN}"
    --num-workers "${NUM_WORKERS}"
    --precision "${PRECISION}"
)

if [[ -n "${HEAD_TYPE}" ]]; then
    CMD+=(--head-type "${HEAD_TYPE}")
fi

if [[ -n "${PAIRWISE_RANK_WEIGHT}" ]]; then
    CMD+=(--pairwise-rank-weight "${PAIRWISE_RANK_WEIGHT}")
fi

if [[ -n "${SWAP_CONSISTENCY_WEIGHT}" ]]; then
    CMD+=(--swap-consistency-weight "${SWAP_CONSISTENCY_WEIGHT}")
fi

case "${ALLOW_TF32}" in
    true|TRUE|1|yes|YES)
        CMD+=(--allow-tf32)
        ;;
    false|FALSE|0|no|NO)
        CMD+=(--no-tf32)
        ;;
    *)
        echo "ERROR: ALLOW_TF32 must be true or false, got ${ALLOW_TF32}" >&2
        exit 1
        ;;
esac

if [[ "${OVERWRITE}" == "true" ]]; then
    CMD+=(--overwrite)
fi

if [[ "${WANDB_ENABLED}" == "true" ]]; then
    CMD+=(--wandb-enabled)
    CMD+=(--wandb-project "${WANDB_PROJECT}")
    CMD+=(--wandb-entity "${WANDB_ENTITY}")
    if [[ -n "${WANDB_RUN_NAME}" ]]; then
        CMD+=(--wandb-run-name "${WANDB_RUN_NAME}")
    fi
    if [[ -n "${WANDB_TAGS}" ]]; then
        CMD+=(--wandb-tags "${WANDB_TAGS}")
    fi
fi

"${CMD[@]}"
