#!/usr/bin/env bash
# Beat-v7 NTv3 head diagnostic: aux-aware parallel multi-scale dilated conv readout, no gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v7-multiscale-dilated-bio-readout-seed0}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v7 multi-scale dilated bio-readout}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-multi-scale-dilated}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-phylo-structure}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-16}"
export FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.05}"
export FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT="${FUNCTIONAL_HEAD_OUTPUT_BIAS_INIT:-scaled-track-mean}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v7_full.sh"
