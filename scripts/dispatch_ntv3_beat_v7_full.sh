#!/usr/bin/env bash
# Dispatch the primary NTv3 human/functional run backed by the pure beat-v7
# base checkpoint.
#
# This is intentionally official-like: beat-v7 token heads consume hidden
# states directly, so this launcher keeps feature_source=hidden and avoids the
# beat-v5 decoder-coupled recipe that underperformed.
#
# Required environment:
#   SAGEMAKER_ROLE=<role-arn>
#   HF_TOKEN=<huggingface-token>
#
# Optional overrides:
#   EXPERIMENT=beat-v7-official-hidden-seed0 \
#   BUCKET=ai4bio-lumina-experiments-v2 \
#   INSTANCE_TYPE=ml.p5en.48xlarge \
#   TRAINING_IMAGE_URI=<custom-image> \
#   FUNCTIONAL_HEAD_TYPE=mlp \
#   FUNCTIONAL_HEAD_AUX_FEATURES=phylo-structure \
#   bash scripts/dispatch_ntv3_beat_v7_full.sh
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
EXPERIMENT="${EXPERIMENT:-beat-v7-official-hidden-seed0}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
CHECKPOINT_S3_PREFIX="${CHECKPOINT_S3_PREFIX:-s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beat-v7-12m-15ep-32k/lumina-ssm-beat-v7-12m-15ep-32k-20260423192051/output/model.tar.gz}"
DATASET_REPO_ID="${DATASET_REPO_ID:-InstaDeepAI/NTv3_benchmark_dataset}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-72}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-144}"
SPECIES="${SPECIES:-human}"
TASK_TYPE="${TASK_TYPE:-functional}"
FEATURE_SOURCE="${FEATURE_SOURCE:-hidden}"
FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-linear}"
FUNCTIONAL_HEAD_HIDDEN_DIM="${FUNCTIONAL_HEAD_HIDDEN_DIM:-}"
FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.05}"
FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"
FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-16}"
PRECISION="${PRECISION:-fp32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_RUNTIME_BATCH_SIZE_PER_RANK="${MAX_RUNTIME_BATCH_SIZE_PER_RANK:-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NUM_STEPS_TRAINING="${NUM_STEPS_TRAINING:-19932}"
VALIDATE_EVERY_N_STEPS="${VALIDATE_EVERY_N_STEPS:-500}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-4000}"
INITIAL_LEARNING_RATE="${INITIAL_LEARNING_RATE:-1e-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
NUM_STEPS_WARMUP="${NUM_STEPS_WARMUP:-598}"
SCHEDULER_NAME="${SCHEDULER_NAME:-modified_square_decay}"
FINAL_LEARNING_RATE_MULTIPLIER="${FINAL_LEARNING_RATE_MULTIPLIER:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-}"
DECODER_LEARNING_RATE="${DECODER_LEARNING_RATE:-}"
BACKBONE_LAYERWISE_LR_DECAY="${BACKBONE_LAYERWISE_LR_DECAY:-}"
HEAD_ONLY_WARMUP_STEPS="${HEAD_ONLY_WARMUP_STEPS:-0}"
EMA_DECAY="${EMA_DECAY:-}"
FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT="${FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT:-none}"
FUNCTIONAL_RC_CONSISTENCY_WEIGHT="${FUNCTIONAL_RC_CONSISTENCY_WEIGHT:-0.0}"
SEED="${SEED:-0}"
MODEL_NAME="${MODEL_NAME:-Lumina beat-v7 official-hidden}"
MODEL_VERSION="${MODEL_VERSION:-beat-v7}"
AUTO_RESUME="${AUTO_RESUME:-1}"

if [[ "${FEATURE_SOURCE}" != "hidden" ]]; then
    echo "ERROR: ${MODEL_VERSION} NTv3 primary protocol requires FEATURE_SOURCE=hidden." >&2
    echo "       This launcher has no decoder_states path; use a separate ablation script for non-hidden features." >&2
    exit 1
fi

if [[ "${FUNCTIONAL_HEAD_TYPE}" != "linear" && "${FUNCTIONAL_HEAD_TYPE}" != "mlp" && "${FUNCTIONAL_HEAD_TYPE}" != "local-conv" && "${FUNCTIONAL_HEAD_TYPE}" != "gated-hybrid" && "${FUNCTIONAL_HEAD_TYPE}" != "multi-scale-dilated" && "${FUNCTIONAL_HEAD_TYPE}" != "global-context" && "${FUNCTIONAL_HEAD_TYPE}" != "context-pyramid" && "${FUNCTIONAL_HEAD_TYPE}" != "v10-representation-pyramid" && "${FUNCTIONAL_HEAD_TYPE}" != "v10-bio-aux-pyramid" && "${FUNCTIONAL_HEAD_TYPE}" != "v10-assay-gated-bio-pyramid" && "${FUNCTIONAL_HEAD_TYPE}" != "v10-bioprogram-stack" && "${FUNCTIONAL_HEAD_TYPE}" != "v10-profile-count-bioaux" && "${FUNCTIONAL_HEAD_TYPE}" != "v10-profile-count-bioaux-rc-gated-residual" && "${FUNCTIONAL_HEAD_TYPE}" != "v10-assay-rescue-hybrid" && "${FUNCTIONAL_HEAD_TYPE}" != "v10-biocov-residual" ]]; then
    echo "ERROR: FUNCTIONAL_HEAD_TYPE must be 'linear', 'mlp', 'local-conv', 'gated-hybrid', 'multi-scale-dilated', 'global-context', 'context-pyramid', or one of the v10 experimental heads." >&2
    exit 1
fi

if [[ "${FUNCTIONAL_HEAD_AUX_FEATURES}" != "none" && "${FUNCTIONAL_HEAD_AUX_FEATURES}" != "phylo" && "${FUNCTIONAL_HEAD_AUX_FEATURES}" != "structure" && "${FUNCTIONAL_HEAD_AUX_FEATURES}" != "phylo-structure" ]]; then
    echo "ERROR: FUNCTIONAL_HEAD_AUX_FEATURES must be 'none', 'phylo', 'structure', or 'phylo-structure'." >&2
    exit 1
fi

if [[ "${FUNCTIONAL_HEAD_AUX_FEATURES}" != "none" && "${FUNCTIONAL_HEAD_TYPE}" != "mlp" && "${FUNCTIONAL_HEAD_TYPE}" != "gated-hybrid" && "${FUNCTIONAL_HEAD_TYPE}" != "multi-scale-dilated" && "${FUNCTIONAL_HEAD_TYPE}" != "global-context" && "${FUNCTIONAL_HEAD_TYPE}" != "context-pyramid" ]]; then
    echo "ERROR: auxiliary functional head features currently require FUNCTIONAL_HEAD_TYPE=mlp, gated-hybrid, multi-scale-dilated, global-context, or context-pyramid." >&2
    exit 1
fi

if (( FUNCTIONAL_HEAD_KERNEL_SIZE <= 0 || FUNCTIONAL_HEAD_KERNEL_SIZE % 2 == 0 )); then
    echo "ERROR: FUNCTIONAL_HEAD_KERNEL_SIZE must be a positive odd integer." >&2
    exit 1
fi

if (( FUNCTIONAL_HEAD_AUX_PROJECTION_DIM <= 0 )); then
    echo "ERROR: FUNCTIONAL_HEAD_AUX_PROJECTION_DIM must be positive." >&2
    exit 1
fi

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

case "${AUTO_RESUME}" in
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
        export LUMINA_NTV3_NO_AUTO_RESUME=1
        CMD+=(--no-auto-resume)
        ;;
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        ;;
    *)
        echo "ERROR: AUTO_RESUME must be a boolean-like value, got '${AUTO_RESUME}'." >&2
        exit 1
        ;;
esac

CMD+=(
    --
    --model-version "${MODEL_VERSION}"
    --species "${SPECIES}"
    --task-type "${TASK_TYPE}"
    --train-backbone
    --feature-source "${FEATURE_SOURCE}"
    --functional-head-type "${FUNCTIONAL_HEAD_TYPE}"
    --functional-head-dropout "${FUNCTIONAL_HEAD_DROPOUT}"
    --functional-head-kernel-size "${FUNCTIONAL_HEAD_KERNEL_SIZE}"
    --functional-head-aux-features "${FUNCTIONAL_HEAD_AUX_FEATURES}"
    --functional-head-aux-projection-dim "${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM}"
    --functional-head-output-bias-init "${FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT}"
    --functional-rc-consistency-weight "${FUNCTIONAL_RC_CONSISTENCY_WEIGHT}"
    --precision "${PRECISION}"
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --max-runtime-batch-size-per-rank "${MAX_RUNTIME_BATCH_SIZE_PER_RANK}"
    --grad-clip "${GRAD_CLIP}"
    --num-steps-training "${NUM_STEPS_TRAINING}"
    --validate-every-n-steps "${VALIDATE_EVERY_N_STEPS}"
    --save-every-n-steps "${SAVE_EVERY_N_STEPS}"
    --initial-learning-rate "${INITIAL_LEARNING_RATE}"
    --learning-rate "${LEARNING_RATE}"
    --num-steps-warmup "${NUM_STEPS_WARMUP}"
    --scheduler-name "${SCHEDULER_NAME}"
    --final-learning-rate-multiplier "${FINAL_LEARNING_RATE_MULTIPLIER}"
    --weight-decay "${WEIGHT_DECAY}"
    --head-only-warmup-steps "${HEAD_ONLY_WARMUP_STEPS}"
    --seed "${SEED}"
    --model-name "${MODEL_NAME}"
)

if [[ -n "${FUNCTIONAL_HEAD_HIDDEN_DIM}" ]]; then
    CMD+=(--functional-head-hidden-dim "${FUNCTIONAL_HEAD_HIDDEN_DIM}")
fi

if [[ -n "${HEAD_LEARNING_RATE}" ]]; then
    CMD+=(--head-learning-rate "${HEAD_LEARNING_RATE}")
fi
if [[ -n "${BACKBONE_LEARNING_RATE}" ]]; then
    CMD+=(--backbone-learning-rate "${BACKBONE_LEARNING_RATE}")
fi
if [[ -n "${DECODER_LEARNING_RATE}" ]]; then
    CMD+=(--decoder-learning-rate "${DECODER_LEARNING_RATE}")
fi
if [[ -n "${BACKBONE_LAYERWISE_LR_DECAY}" ]]; then
    CMD+=(--backbone-layerwise-lr-decay "${BACKBONE_LAYERWISE_LR_DECAY}")
fi
if [[ -n "${EMA_DECAY}" ]]; then
    CMD+=(--ema-decay "${EMA_DECAY}")
fi

"${CMD[@]}"
