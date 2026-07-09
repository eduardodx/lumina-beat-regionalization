#!/usr/bin/env bash
# Dispatch the next beat-v7 NTv3 ablation:
# same winning recipe, replacing only the functional MLP head with a local-conv head.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v7-local-conv-head-k15-seed0}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v7 local-conv-head k15}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-local-conv}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.05}"
export FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v7_full.sh"
