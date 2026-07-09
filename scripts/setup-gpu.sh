#!/usr/bin/env bash
# Setup script for Linux GPU environments (e.g. SageMaker).
# Builds mamba-ssm from source against the installed CUDA toolkit.
set -euo pipefail

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: This script is intended for Linux GPU machines only." >&2
    exit 1
fi

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Ensure the CUDA toolkit is installed and on PATH." >&2
    echo "       On SageMaker, try: export PATH=/usr/local/cuda/bin:\$PATH" >&2
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "CUDA toolkit:"
nvcc --version
echo ""

if command -v nvidia-smi &>/dev/null; then
    echo "GPU driver:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo ""
fi

# ---------------------------------------------------------------------------
# Build environment
# ---------------------------------------------------------------------------
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

EXTRAS="${1:-dev,tracking}"
HOST_PYTHON="${HOST_PYTHON:-python3}"
USE_SYSTEM_TORCH="${USE_SYSTEM_TORCH:-auto}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
CAUSAL_CONV1D_SPEC="${CAUSAL_CONV1D_SPEC:-causal-conv1d>=1.2.0}"
MAMBA_SSM_SPEC="${MAMBA_SSM_SPEC:-mamba-ssm @ git+https://github.com/state-spaces/mamba.git}"
TILELANG_SPEC="${TILELANG_SPEC:-tilelang}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn}"
INSTALL_TILELANG="${INSTALL_TILELANG:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
APACHE_TVM_FFI_SPEC="${APACHE_TVM_FFI_SPEC:-}"

is_truthy() {
    case "${1,,}" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

install_flash_attn_if_requested() {
    if ! is_truthy "${INSTALL_FLASH_ATTN}"; then
        return 0
    fi

    echo "Installing flash-attn: ${FLASH_ATTN_SPEC}"
    uv pip install psutil ninja
    uv pip install --no-build-isolation --no-deps "${FLASH_ATTN_SPEC}"
}

resolve_system_torch_mode() {
    local requested="$1"
    case "${requested}" in
        true|false)
            printf '%s\n' "${requested}"
            return 0
            ;;
        auto)
            ;;
        *)
            echo "ERROR: USE_SYSTEM_TORCH must be one of: auto, true, false" >&2
            exit 1
            ;;
    esac

    if ! command -v "${HOST_PYTHON}" &>/dev/null; then
        printf 'false\n'
        return 0
    fi

    if "${HOST_PYTHON}" - <<'PY' &>/dev/null
import torch
PY
    then
        printf 'true\n'
    else
        printf 'false\n'
    fi
}

maybe_export_torch_cuda_arch_list() {
    if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (pre-set)"
        return 0
    fi
    if ! command -v "${HOST_PYTHON}" &>/dev/null; then
        return 0
    fi

    local detected_arch=""
    detected_arch="$("${HOST_PYTHON}" - <<'PY' 2>/dev/null || true
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}.{minor}")
except Exception:
    pass
PY
)"
    if [[ -n "${detected_arch}" ]]; then
        export TORCH_CUDA_ARCH_LIST="${detected_arch}"
        echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (detected)"
    fi
}

SYSTEM_TORCH_MODE="$(resolve_system_torch_mode "${USE_SYSTEM_TORCH}")"
echo "System torch mode: ${SYSTEM_TORCH_MODE}"
echo "Package specs:"
echo "  causal-conv1d: ${CAUSAL_CONV1D_SPEC}"
echo "  mamba-ssm: ${MAMBA_SSM_SPEC}"
echo "  tilelang: ${TILELANG_SPEC} (install=${INSTALL_TILELANG})"
echo "  apache-tvm-ffi: ${APACHE_TVM_FFI_SPEC:-<resolver default>} (install=${INSTALL_TILELANG})"
maybe_export_torch_cuda_arch_list

echo "Creating venv and syncing (extras: ${EXTRAS})..."
if [[ "${SYSTEM_TORCH_MODE}" == "true" ]]; then
    # Use the host Python so the venv can see system site-packages (incl. torch).
    # A version mismatch (e.g. host=3.12, venv=3.11) makes site-packages invisible.
    HOST_PY_VERSION="$("${HOST_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "Using host/container torch via system site-packages (Python ${HOST_PY_VERSION})."
    uv venv --python "${HOST_PY_VERSION}" --system-site-packages
    uv pip install setuptools wheel packaging

    # ── Why we do NOT use `uv pip install -e ".[extras]"` here ──────────
    # pyproject.toml pins torch to the pytorch-cu126 index via [tool.uv.sources].
    # That build lacks CUDA kernels for newer GPU architectures (e.g. sm_100 /
    # Blackwell).  The DLC's *system* torch is compiled for the correct arch, so
    # we must keep it and never shadow it with the cu126 wheel.
    #
    # Crucially, `uv pip` does NOT consider system-site-packages when resolving
    # dependencies — it would install torch into the venv even though the system
    # already has it.  This creates an ABI mismatch between the CUDA extensions
    # (compiled against one torch) and the runtime torch.
    #
    # Strategy:
    #   1. Build CUDA extensions with --no-deps so only system torch is visible.
    #   2. Install the editable package with --no-deps.
    #   3. Install remaining deps (from pyproject.toml + mamba-ssm transitive).
    #   4. Remove any torch that leaked into the venv via transitive deps.

    # Step 1 — CUDA extensions against the system torch.
    # --no-deps prevents uv from installing torch (or anything) into the venv.
    # --no-build-isolation means the build sees system torch via system-site-packages.
    uv pip install --no-build-isolation --no-deps \
        "${CAUSAL_CONV1D_SPEC}" \
        "${MAMBA_SSM_SPEC}"

    # Step 2 — editable project (no dependency resolution)
    uv pip install --no-deps -e "."

    # Step 3 — remaining deps, excluding packages already provided by the
    #          system or built above (torch, mamba-ssm, causal-conv1d).
    #          Also includes mamba-ssm's own transitive deps that were skipped
    #          by --no-deps in Step 1. TileLang is required for the upstream
    #          Mamba-3 MIMO kernels used by beat-v1 / bimamba3_rc.
    "${HOST_PYTHON}" - "${EXTRAS}" <<'PYSCRIPT' > /tmp/_remaining_deps.txt
import os, pathlib, sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib          # Python 3.10 fallback

data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())

SKIP = {"torch", "mamba-ssm", "mamba_ssm", "causal-conv1d", "causal_conv1d", "mamba-ssm-macos"}

def should_skip(spec: str) -> bool:
    name = spec.split(";")[0].split(">")[0].split("<")[0].split("=")[0].split("[")[0].split("@")[0].strip()
    return name.lower().replace("-", "_") in {s.replace("-", "_") for s in SKIP}

deps = [d for d in data["project"]["dependencies"] if not should_skip(d)]

extras = data["project"].get("optional-dependencies", {})
for name in (sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else []):
    deps.extend(extras.get(name, []))

# mamba-ssm's own transitive deps (installed --no-deps above).
deps.extend(["einops", "triton", "ninja"])

install_tilelang = os.environ.get("INSTALL_TILELANG", "1").strip().lower() in {"1", "true", "yes", "on"}
if install_tilelang:
    deps.append(os.environ.get("TILELANG_SPEC", "tilelang"))

apache_tvm_ffi_spec = os.environ.get("APACHE_TVM_FFI_SPEC", "").strip()
if install_tilelang and apache_tvm_ffi_spec:
    deps.append(apache_tvm_ffi_spec)

for d in deps:
    print(d)
PYSCRIPT
    echo "Installing remaining deps:"
    cat /tmp/_remaining_deps.txt
    uv pip install -r /tmp/_remaining_deps.txt

    # Step 4 — safety net: some transitive deps (e.g. transformers) may have
    #          pulled torch into the venv, shadowing the system torch.
    #          Remove it so the DLC's torch (correct ABI + GPU arch) stays active.
    if uv pip show torch 2>/dev/null | grep -q "Location.*\.venv"; then
        echo "Removing venv torch (transitive dep leaked); system torch will take over."
        uv pip uninstall torch
        # Also remove nvidia CUDA packages that leaked as transitive deps of tilelang /
        # transformers. These venv copies shadow the DLC system CUDA libraries and cause
        # version-mismatch ImportErrors (e.g. libcusparse 12.9 needs libnvJitLink 12.9
        # but finds the venv's 12.8 copy first).
        _leaked_nvidia=$(uv pip list 2>/dev/null | awk 'NR>2 && /^nvidia-/{print $1}')
        if [[ -n "${_leaked_nvidia}" ]]; then
            echo "Removing leaked nvidia CUDA packages from venv: $(echo ${_leaked_nvidia} | tr '\n' ' ')"
            # shellcheck disable=SC2086
            uv pip uninstall ${_leaked_nvidia} 2>/dev/null || true
        fi
        unset _leaked_nvidia
    fi

    # Verify the system torch is the one that's active
    .venv/bin/python -c "
import torch, os
loc = os.path.dirname(torch.__file__)
print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  path={loc}')
print(f'  arch_list={torch.cuda.get_arch_list() if hasattr(torch.cuda, \"get_arch_list\") else \"N/A\"}')
assert '.venv' not in loc, f'ERROR: venv torch is active ({loc}), expected system torch'
print('  ✓ system torch confirmed')
" || echo "WARNING: could not verify system torch"
else
    echo "Host/container torch not available; installing torch from ${TORCH_INDEX_URL}."
    uv venv --python 3.11
    # Phase 1: install mamba-ssm's undeclared build deps
    uv pip install setuptools wheel packaging
    uv pip install torch --index-url "${TORCH_INDEX_URL}"
    # Phase 2: full sync — mamba-ssm builds from source with torch available
    uv sync --extra ${EXTRAS//,/ --extra }
    if is_truthy "${INSTALL_TILELANG}"; then
        uv pip install "${TILELANG_SPEC}"
        if [[ -n "${APACHE_TVM_FFI_SPEC}" ]]; then
            uv pip install "${APACHE_TVM_FFI_SPEC}"
        fi
    fi
fi

install_flash_attn_if_requested

echo ""
echo "Environment ready. Activate with:  source .venv/bin/activate"
