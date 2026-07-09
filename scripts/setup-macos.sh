#!/usr/bin/env bash
# Setup script for macOS local development (Apple Silicon).
# Installs all dependencies including mamba-ssm-macos for ARM64 Macs.
set -euo pipefail

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script is intended for macOS only." >&2
    echo "       For Linux GPU environments, use setup-gpu.sh instead." >&2
    exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    echo "WARNING: This machine is ${ARCH}. mamba-ssm-macos requires Apple Silicon (arm64)." >&2
    echo "         The install will proceed, but mamba-ssm-macos may not be available." >&2
fi

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "macOS $(sw_vers -productVersion) (${ARCH})"
echo ""

# ---------------------------------------------------------------------------
# Build environment
# ---------------------------------------------------------------------------
EXTRAS="${1:-dev}"

echo "Creating venv and installing (extras: ${EXTRAS})..."
uv venv --python 3.11
# Use `uv pip install` instead of `uv sync` to resolve only for the current
# platform.  `uv sync` performs universal resolution and attempts to build
# Linux-only packages (mamba-ssm from source) even on macOS.
uv pip install -e ".[${EXTRAS}]"

echo ""
echo "Environment ready. Activate with:  source .venv/bin/activate"
