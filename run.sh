#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh single  [extra flags...]   # single-GPU (default)
#   ./run.sh multi   [extra flags...]   # multi-GPU via torchrun (auto-detects GPU count)
#
# Override the config with CONFIG_PATH:
#   CONFIG_PATH=configs/bimamba/8m.yaml ./run.sh multi
#
# Extra flags are forwarded to src.train and override config values:
#   ./run.sh multi --max-steps 5000 --output-dir outputs/my_run

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CONFIG_PATH="${CONFIG_PATH:-configs/beat_v2/9m_15ep_for_real_32k_b200_v2.yaml.yaml}"

MODE="${1:-single}"
shift || true  # consume mode arg; remaining args are forwarded

case "$MODE" in
  single)
    cmd=(
      uv run python -m src.train
      --config "$CONFIG_PATH"
    )
    ;;
  multi)
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [[ "$NUM_GPUS" -lt 2 ]]; then
      echo "ERROR: multi mode requires >=2 GPUs, found $NUM_GPUS" >&2
      exit 1
    fi

    cmd=(
      uv run torchrun
      --nproc_per_node "$NUM_GPUS"
      -m src.train
      --config "$CONFIG_PATH"
    )
    ;;
  *)
    echo "Usage: $0 {single|multi} [extra flags...]" >&2
    exit 1
    ;;
esac

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running (%s, %s GPU(s)):' "$MODE" "${NUM_GPUS:-1}"
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
