#!/usr/bin/env bash
# Dispatch a Lumina-coupled NTv3 human/functional run backed by the beat-v5 checkpoint.
#
# This is intentionally not the official-like reproduction preset. It keeps the
# NTv3 data/evaluation protocol intact while adapting the fine-tuning mechanics
# to beat-v5's dense decoder representation.
#
# Required environment:
#   SAGEMAKER_ROLE=<role-arn>
#   HF_TOKEN=<huggingface-token>
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
EXPERIMENT="${EXPERIMENT:-beat-v5-dec-coupled}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
CHECKPOINT_S3_PREFIX="${CHECKPOINT_S3_PREFIX:-s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beat-v5-384w-8l-15ep-32k/lumina-ssm-beat-v5-384w-8l-15ep-32k-20260414113157/output/model.tar.gz}"
DATASET_REPO_ID="${DATASET_REPO_ID:-InstaDeepAI/NTv3_benchmark_dataset}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-72}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-144}"
SPECIES="${SPECIES:-human}"
TASK_TYPE="${TASK_TYPE:-functional}"
PRECISION="${PRECISION:-fp32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_RUNTIME_BATCH_SIZE_PER_RANK="${MAX_RUNTIME_BATCH_SIZE_PER_RANK:-2}"
NUM_STEPS_TRAINING="${NUM_STEPS_TRAINING:-19932}"
VALIDATE_EVERY_N_STEPS="${VALIDATE_EVERY_N_STEPS:-500}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-4000}"
INITIAL_LEARNING_RATE="${INITIAL_LEARNING_RATE:-1e-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-4}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-5e-6}"
DECODER_LEARNING_RATE="${DECODER_LEARNING_RATE:-1e-5}"
HEAD_ONLY_WARMUP_STEPS="${HEAD_ONLY_WARMUP_STEPS:-1000}"
NUM_STEPS_WARMUP="${NUM_STEPS_WARMUP:-1500}"
SCHEDULER_NAME="${SCHEDULER_NAME:-modified_square_decay}"
FINAL_LEARNING_RATE_MULTIPLIER="${FINAL_LEARNING_RATE_MULTIPLIER:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SEED="${SEED:-0}"
MODEL_NAME="${MODEL_NAME:-Lumina beat-v5 decoder-coupled}"

CMD=(
    uv run --extra sagemaker python "${REPO_ROOT}/scripts/sagemaker_ntv3_benchmark.py"
    --experiment "${EXPERIMENT}"
    --job-name-prefix "${JOB_NAME_PREFIX}"
    --bucket "${BUCKET}"
    --instance-type "${INSTANCE_TYPE}"
    --checkpoint-s3-prefix "${CHECKPOINT_S3_PREFIX}"
    --dataset-repo-id "${DATASET_REPO_ID}"
    --max-run-hours "${MAX_RUN_HOURS}"
    --max-wait-hours "${MAX_WAIT_HOURS}"
    --nproc-per-node "${NPROC_PER_NODE}"
    --spot
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
    --train-backbone
    --feature-source decoder
    --precision "${PRECISION}"
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --max-runtime-batch-size-per-rank "${MAX_RUNTIME_BATCH_SIZE_PER_RANK}"
    --num-steps-training "${NUM_STEPS_TRAINING}"
    --validate-every-n-steps "${VALIDATE_EVERY_N_STEPS}"
    --save-every-n-steps "${SAVE_EVERY_N_STEPS}"
    --initial-learning-rate "${INITIAL_LEARNING_RATE}"
    --learning-rate "${LEARNING_RATE}"
    --head-learning-rate "${HEAD_LEARNING_RATE}"
    --backbone-learning-rate "${BACKBONE_LEARNING_RATE}"
    --decoder-learning-rate "${DECODER_LEARNING_RATE}"
    --head-only-warmup-steps "${HEAD_ONLY_WARMUP_STEPS}"
    --num-steps-warmup "${NUM_STEPS_WARMUP}"
    --scheduler-name "${SCHEDULER_NAME}"
    --final-learning-rate-multiplier "${FINAL_LEARNING_RATE_MULTIPLIER}"
    --weight-decay "${WEIGHT_DECAY}"
    --no-weight-decay-norm-bias
    --seed "${SEED}"
    --model-name "${MODEL_NAME}"
)

"${CMD[@]}"
