#!/usr/bin/env bash
# Dispatch beat-v10 NTv3 with a profile/count head over v10 biological features.
#
# Hypothesis: dense functional tracks are better modeled as two coupled factors:
# a local base-resolution profile and a window-level per-track signal amplitude.
# This tests whether beat-v10 sequence embeddings and frozen biological readouts
# help separate assay shape from assay mass.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-20k-v10-profile-count-bioaux-lr125-llrd085-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 20k Profile/Count BioAux LR1.25e-4 LLRD0.85}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-v10-profile-count-bioaux}"
export FUNCTIONAL_HEAD_HIDDEN_DIM="${FUNCTIONAL_HEAD_HIDDEN_DIM:-128}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-128}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.03}"
export FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
