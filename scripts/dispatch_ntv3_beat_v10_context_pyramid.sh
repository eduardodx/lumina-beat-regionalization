#!/usr/bin/env bash
# Dispatch NTv3 human/functional fine-tuning from the Lumina beat-v10 release checkpoint.
#
# This transfers the current winning NTv3 shape where it is compatible with
# beat-v10: full-backbone fine-tuning, context-pyramid readout, cosine schedule,
# head warmup, layerwise LR decay, and EMA. The v7 phylo/structure auxiliary
# readouts are intentionally disabled because beat-v10 exposes substitution
# heads instead of the scalar phylo100/phylo470/structure heads consumed by the
# current NTv3 auxiliary feature path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-20k-context-pyramid-noaux-lr125-llrd085-seed0}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 20k context-pyramid no-aux LR1.25e-4 LLRD0.85}"
export MODEL_VERSION="${MODEL_VERSION:-beat-v10}"
export CHECKPOINT_S3_PREFIX="${CHECKPOINT_S3_PREFIX:-s3://ai4bio-lumina/releases/lumina-beat-v10-20260527182934/}"
export INSTANCE_TYPE="${INSTANCE_TYPE:-ml.p5en.48xlarge}"

export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-context-pyramid}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-16}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.02}"
export FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"
export FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT="${FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT:-scaled-track-mean}"

export PRECISION="${PRECISION:-fp32}"
export SCHEDULER_NAME="${SCHEDULER_NAME:-cosine}"
export NUM_STEPS_TRAINING="${NUM_STEPS_TRAINING:-19932}"
export NUM_STEPS_WARMUP="${NUM_STEPS_WARMUP:-800}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-1000}"

export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-3e-4}"
export BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1.25e-4}"
export BACKBONE_LAYERWISE_LR_DECAY="${BACKBONE_LAYERWISE_LR_DECAY:-0.85}"
export HEAD_ONLY_WARMUP_STEPS="${HEAD_ONLY_WARMUP_STEPS:-250}"
export EMA_DECAY="${EMA_DECAY:-0.999}"

# New experiment IDs should not silently resume stale incompatible checkpoints.
export AUTO_RESUME="${AUTO_RESUME:-0}"

if [[ "${FUNCTIONAL_HEAD_AUX_FEATURES}" != "none" ]]; then
    echo "ERROR: beat-v10 NTv3 aux readouts are not wired yet. Use FUNCTIONAL_HEAD_AUX_FEATURES=none." >&2
    echo "       The current aux path expects v7 scalar phylo100/phylo470/structure heads." >&2
    exit 1
fi

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v7_full.sh"
