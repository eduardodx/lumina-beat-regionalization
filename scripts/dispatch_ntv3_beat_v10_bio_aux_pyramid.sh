#!/usr/bin/env bash
# Dispatch beat-v10 NTv3 with frozen beat-v10 biological auxiliary readouts.
#
# Hypothesis: pretraining heads for substitution, splice, region, gnomAD,
# counterfactual SNV effect, and regulatory tracks provide biologically
# meaningful coordinates that make NTv3 dense functional fine-tuning more
# sample-efficient and assay-aware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-20k-v10-bio-aux-pyramid-lr125-llrd085-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 20k v10-bio-aux-pyramid LR1.25e-4 LLRD0.85}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-v10-bio-aux-pyramid}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-128}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
