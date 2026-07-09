#!/usr/bin/env bash
# Dispatch the final beat-v10 NTv3 candidate: profile/count BioAux with RC
# consistency and a small gated residual for assay-specific corrections.
#
# This combines only mechanisms that showed positive evidence: profile/count
# decomposition, frozen beat-v10 BioAux readouts, and low-weight reverse-
# complement consistency. The residual starts nearly closed and is intended to
# recover ATAC/histone/eCLIP tracks without overriding the PRO-cap/RNA gains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-18k-v10-profile-count-bioaux-rc005-gated-residual-lr125-llrd085-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 18k Profile/Count BioAux RC0.05 GatedResidual LR1.25e-4 LLRD0.85}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-v10-profile-count-bioaux-rc-gated-residual}"
export FUNCTIONAL_HEAD_HIDDEN_DIM="${FUNCTIONAL_HEAD_HIDDEN_DIM:-128}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-128}"
export FUNCTIONAL_HEAD_DROPOUT="${FUNCTIONAL_HEAD_DROPOUT:-0.03}"
export FUNCTIONAL_HEAD_KERNEL_SIZE="${FUNCTIONAL_HEAD_KERNEL_SIZE:-15}"
export FUNCTIONAL_RC_CONSISTENCY_WEIGHT="${FUNCTIONAL_RC_CONSISTENCY_WEIGHT:-0.05}"

# Prior v10 runs peaked around 16k-17.5k steps and then drifted slightly down.
# Use a shorter budget while retaining validation/best-checkpoint selection.
export NUM_STEPS_TRAINING="${NUM_STEPS_TRAINING:-18000}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-1000}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
