#!/usr/bin/env bash
# Dispatch the beat-v10 SOTA head with a longer non-zero LR tail.
#
# Hypothesis: the profile/count BioAux RC residual head was still improving at
# step 18k, but the cosine schedule had decayed to zero. This run keeps the same
# architecture and tests whether a longer modified-square decay tail improves
# NTv3 human/functional performance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-24k-v10-profile-count-bioaux-rc005-gated-residual-longtail-lr150-llrd090-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 24k Profile/Count BioAux RC0.05 GatedResidual LongTail LR1.5e-4 LLRD0.90}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-v10-profile-count-bioaux-rc-gated-residual}"
export FUNCTIONAL_HEAD_HIDDEN_DIM="${FUNCTIONAL_HEAD_HIDDEN_DIM:-128}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-128}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.03}"
export FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"
export FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT="${FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT:-scaled-track-mean}"
export FUNCTIONAL_RC_CONSISTENCY_WEIGHT="${FUNCTIONAL_RC_CONSISTENCY_WEIGHT:-0.05}"

export NUM_STEPS_TRAINING="${NUM_STEPS_TRAINING:-24000}"
export NUM_STEPS_WARMUP="${NUM_STEPS_WARMUP:-1000}"
export SCHEDULER_NAME="${SCHEDULER_NAME:-modified_square_decay}"
export FINAL_LEARNING_RATE_MULTIPLIER="${FINAL_LEARNING_RATE_MULTIPLIER:-0.25}"

export INITIAL_LEARNING_RATE="${INITIAL_LEARNING_RATE:-1e-5}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-3e-4}"
export BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1.5e-4}"
export BACKBONE_LAYERWISE_LR_DECAY="${BACKBONE_LAYERWISE_LR_DECAY:-0.90}"
export HEAD_ONLY_WARMUP_STEPS="${HEAD_ONLY_WARMUP_STEPS:-250}"
export EMA_DECAY="${EMA_DECAY:-0.999}"
export GRAD_CLIP="${GRAD_CLIP:-1.0}"

export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-1000}"
export AUTO_RESUME="${AUTO_RESUME:-0}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
