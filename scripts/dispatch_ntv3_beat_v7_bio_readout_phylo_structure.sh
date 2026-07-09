#!/usr/bin/env bash
# Beat-v7 NTv3 ablation that keeps the successful MLP head recipe and adds
# auxiliary readout features from Lumina's own frozen phylo/structure heads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v7-bio-readout-phylo-structure-mlp-seed0}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ntv3}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v7 bio-readout phylo-structure MLP}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-mlp}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-phylo-structure}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-16}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.05}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v7_full.sh"
