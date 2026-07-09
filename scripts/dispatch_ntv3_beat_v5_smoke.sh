#!/usr/bin/env bash
# Dispatch the first diagnostic NTv3 smoke run backed by the successful beat-v5 checkpoint.
#
# Required environment:
#   SAGEMAKER_ROLE=<role-arn>
#   HF_TOKEN=<huggingface-token>
#
# Optional overrides:
#   EXPERIMENT=ntv3-beat-v5-smoke-a-seed42 \
#   BUCKET=ai4bio-lumina-experiments-v2 \
#   INSTANCE_TYPE=ml.p5en.48xlarge \
#   TRAINING_IMAGE_URI=<custom-image> \
#   bash scripts/dispatch_ntv3_beat_v5_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

if [[ -z "${SAGEMAKER_ROLE:-}" ]]; then
    echo "ERROR: SAGEMAKER_ROLE must be set before dispatching the NTv3 benchmark job." >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
    echo "ERROR: HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set to stage the gated NTv3 dataset." >&2
    exit 1
fi

BUCKET="${BUCKET:-ai4bio-lumina-experiments-v2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.p5en.48xlarge}"
TRAINING_IMAGE_URI="${TRAINING_IMAGE_URI:-}"
EXPERIMENT="${EXPERIMENT:-ntv3-beat-v5-smoke-a-seed42}"
CHECKPOINT_S3_PREFIX="${CHECKPOINT_S3_PREFIX:-s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beat-v5-384w-8l-15ep-32k/lumina-ssm-beat-v5-384w-8l-15ep-32k-20260414113157/output/model.tar.gz}"
DATASET_REPO_ID="${DATASET_REPO_ID:-InstaDeepAI/NTv3_benchmark_dataset}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-12}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-12}"
SPECIES="${SPECIES:-human}"
TASK_TYPE="${TASK_TYPE:-functional}"
PRECISION="${PRECISION:-auto}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_RUNTIME_BATCH_SIZE_PER_RANK="${MAX_RUNTIME_BATCH_SIZE_PER_RANK:-1}"
NUM_STEPS_TRAINING="${NUM_STEPS_TRAINING:-20}"
VALIDATE_EVERY_N_STEPS="${VALIDATE_EVERY_N_STEPS:-10}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-10}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export LUMINA_CUDA_DEBUG_SYNC="${LUMINA_CUDA_DEBUG_SYNC:-1}"

CMD=(
    uv run --extra sagemaker python "${REPO_ROOT}/scripts/sagemaker_ntv3_benchmark.py"
    --experiment "${EXPERIMENT}"
    --bucket "${BUCKET}"
    --instance-type "${INSTANCE_TYPE}"
    --spot
    --checkpoint-s3-prefix "${CHECKPOINT_S3_PREFIX}"
    --dataset-repo-id "${DATASET_REPO_ID}"
    --max-run-hours "${MAX_RUN_HOURS}"
    --max-wait-hours "${MAX_WAIT_HOURS}"
    --nproc-per-node "${NPROC_PER_NODE}"
    --detach
)

if [[ -n "${TRAINING_IMAGE_URI}" ]]; then
    CMD+=(--training-image-uri "${TRAINING_IMAGE_URI}")
fi

CMD+=(
    --
    --model-version beat-v5
    --species "${SPECIES}"
    --task-type "${TASK_TYPE}"
    --precision "${PRECISION}"
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --max-runtime-batch-size-per-rank "${MAX_RUNTIME_BATCH_SIZE_PER_RANK}"
    --num-steps-training "${NUM_STEPS_TRAINING}"
    --validate-every-n-steps "${VALIDATE_EVERY_N_STEPS}"
    --save-every-n-steps "${SAVE_EVERY_N_STEPS}"
    --learning-rate "${LEARNING_RATE}"
)

"${CMD[@]}"
