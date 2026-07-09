#!/usr/bin/env bash
# Dispatch the first architecture-aware NTv3 beat-v7 ablation:
# same reproduced recipe, replacing only the functional linear head with an MLP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v7-mlp-head-seed0}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v7 MLP-head}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-mlp}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.05}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v7_full.sh"
