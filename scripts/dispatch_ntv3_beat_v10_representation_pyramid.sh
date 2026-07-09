#!/usr/bin/env bash
# Dispatch beat-v10 NTv3 with a v10-native representation pyramid head.
#
# Hypothesis: exposing beat-v10's full-resolution h_pure channel and mid-stack
# regional state to the NTv3 readout improves tracks that depend on local motif
# fidelity plus broader locus context, without relying on frozen biological heads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-20k-v10-representation-pyramid-lr125-llrd085-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 20k v10-representation-pyramid LR1.25e-4 LLRD0.85}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-v10-representation-pyramid}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-128}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
