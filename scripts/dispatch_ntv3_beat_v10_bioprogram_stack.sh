#!/usr/bin/env bash
# Dispatch beat-v10 NTv3 with the stacked BioProgram proof-of-concept head.
#
# This intentionally uses every beat-v10 interface relevant to dense functional
# prediction: h_up, h_pure, mid_hidden_states, sequence_embedding_head, MLM,
# conservation/substitution, splice, region, gnomAD, counterfactual SNV, and
# regulatory heads. The biological heads are frozen, but gradients pass through
# them to shape the beat-v10 hidden space during full fine-tuning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-20k-v10-bioprogram-stack-allfeatures-lr125-llrd085-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 20k BioProgram Stack all-features LR1.25e-4 LLRD0.85}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-v10-bioprogram-stack}"
export FUNCTIONAL_HEAD_HIDDEN_DIM="${FUNCTIONAL_HEAD_HIDDEN_DIM:-32}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-256}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.03}"
export FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
