#!/usr/bin/env bash
# Beat-v7 NTv3 20k ablation: adapt deeper backbone layers more strongly.
#
# Hypothesis: the best 20k recipe may still be too conservative in lower and
# intermediate beat-v7 layers. This keeps the successful backbone LR and raises
# LLRD so more of the backbone participates, without changing data, loss,
# target transform, splits, or evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v7-20k-aggressive-deep-llrd09-seed0}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v7 20k aggressive deep LLRD0.90}"

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
export BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1e-4}"
export BACKBONE_LAYERWISE_LR_DECAY="${BACKBONE_LAYERWISE_LR_DECAY:-0.9}"
export HEAD_ONLY_WARMUP_STEPS="${HEAD_ONLY_WARMUP_STEPS:-250}"
export EMA_DECAY="${EMA_DECAY:-0.999}"
export AUTO_RESUME="${AUTO_RESUME:-1}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v7_full.sh"
