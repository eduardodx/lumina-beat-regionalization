#!/usr/bin/env bash
# Beat-v7 NTv3 20k ablation: increase backbone adaptation moderately.
#
# Hypothesis: the aggressive 20k recipe improved because beat-v7 was
# under-adapting within the NTv3 20k budget. This run raises the backbone LR
# and makes LLRD less conservative while keeping the validated bio-readout
# head, target transform, loss, splits, and evaluation unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v7-20k-aggressive-moderate-lr125-llrd085-seed0}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v7 20k aggressive moderate LR1.25e-4 LLRD0.85}"

export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-mlp}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-phylo-structure}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-16}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.02}"
export FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT="${FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT:-scaled-track-mean}"

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
export AUTO_RESUME="${AUTO_RESUME:-1}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v7_full.sh"
