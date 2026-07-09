#!/usr/bin/env bash
# Dispatch beat-v1 8M 32k training on 8xB200 via SageMaker.
#
# Override defaults with environment variables:
#   BUCKET=my-bucket ./scripts/dispatch_beat_v1_b200.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUCKET="${BUCKET:-ai4bio-lumina-experiments-v2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.p6-b200.48xlarge}"
DEFAULT_IMAGE_URI="085188779747.dkr.ecr.us-east-2.amazonaws.com/lumina/hf-training-fa2-b200:pt280-tr4562-cu129-py312-fa283-sm100"
TRAINING_IMAGE_URI="${TRAINING_IMAGE_URI:-${DEFAULT_IMAGE_URI}}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/sagemaker_train.py" \
  --experiment beat-v1-8m-15ep-32k-b200 \
  --config configs/beat_v1/8m_15ep_32k_b200.yaml \
  --bucket "${BUCKET}" \
  --instance-type "${INSTANCE_TYPE}" \
  --training-image-uri "${TRAINING_IMAGE_URI}" \
  --max-run-hours 24 \
  --spot \
  --detach
