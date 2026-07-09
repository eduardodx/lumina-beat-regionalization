#!/usr/bin/env bash
# Dispatch beat-v10 NTv3 with a per-track gate over local, contextual, and
# biological branches.
#
# Hypothesis: NTv3 tracks differ in causal scale. Promoter/proximal assays may
# prefer local motif and transcription-start signals, while chromatin/RNA tracks
# may need broader context or frozen biological coordinates. A learned per-track
# mixture tests whether one shared readout is hiding assay-specific preferences.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-20k-v10-assay-gated-bio-pyramid-lr125-llrd085-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 20k v10-assay-gated-bio-pyramid LR1.25e-4 LLRD0.85}"
export FUNCTIONAL_HEAD_TYPE="${FUNCTIONAL_HEAD_TYPE:-v10-assay-gated-bio-pyramid}"
export FUNCTIONAL_HEAD_AUX_FEATURES="${FUNCTIONAL_HEAD_AUX_FEATURES:-none}"
export FUNCTIONAL_HEAD_AUX_PROJECTION_DIM="${FUNCTIONAL_HEAD_AUX_PROJECTION_DIM:-128}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
