#!/usr/bin/env bash
# Dispatch a short NTv3 smoke run for Lumina beat-v10 before the full recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-beat-v10-smoke-context-pyramid-noaux-seed0}"
export MODEL_NAME="${MODEL_NAME:-Lumina beat-v10 smoke context-pyramid no-aux}"
export MAX_RUN_HOURS="${MAX_RUN_HOURS:-12}"
export MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-12}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export MAX_RUNTIME_BATCH_SIZE_PER_RANK="${MAX_RUNTIME_BATCH_SIZE_PER_RANK:-1}"
export NUM_STEPS_TRAINING="${NUM_STEPS_TRAINING:-20}"
export VALIDATE_EVERY_N_STEPS="${VALIDATE_EVERY_N_STEPS:-10}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-3e-4}"
export BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1.25e-4}"
export HEAD_ONLY_WARMUP_STEPS="${HEAD_ONLY_WARMUP_STEPS:-0}"
export AUTO_RESUME="${AUTO_RESUME:-0}"
export LUMINA_CUDA_DEBUG_SYNC="${LUMINA_CUDA_DEBUG_SYNC:-1}"

exec bash "${SCRIPT_DIR}/dispatch_ntv3_beat_v10_context_pyramid.sh"
