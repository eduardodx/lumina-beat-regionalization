#!/usr/bin/env bash
set -euo pipefail

uv run python -m eval.ntv3.run stage-checkpoint
uv run python -m eval.ntv3.run stage-dataset
uv run python -m eval.ntv3.run evaluate-all "$@"
