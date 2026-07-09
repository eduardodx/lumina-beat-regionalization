#!/usr/bin/env bash
# Dispatch beat-v10 NTv3 with a low-rank biological covariance residual head.
#
# Hypothesis: the 34 NTv3 tracks share latent biological covariance that can be
# captured from beat-v10's grouped pretraining readouts without forcing the full
# prediction through the heavier BioProgram stack bottleneck.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-20k-v10-biocov-residual-lr125-llrd085-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 20k BioCov Residual LR1.25e-4 LLRD0.85}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-v10-biocov-residual}"
export FUNCTIONAL_HEAD_HIDDEN_DIM="${FUNCTIONAL_HEAD_HIDDEN_DIM:-16}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-256}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.03}"
export FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
