#!/usr/bin/env bash
# Reinstala o ambiente de GPU do R03 na HOME (~/.local), para sobreviver ao reinicio do espaco do SageMaker Studio.
#
# Tudo fora da home e recriado a partir da imagem a cada reinicio: em 24/09 o mamba_ssm instalado no /opt/conda sumiu
# duas vezes. Estes sao os MESMOS passos que reproduziram os caches naquele dia (diferenca exatamente zero no M0 e no
# MR_a2), com duas mudancas:
#   --user  -> ~/.local/lib/python3.12/site-packages, que fica na home e vem antes do /opt/conda no sys.path;
#   mamba FIXADO no commit instalado naquele dia (e9594ce1, 22/07/2026; versao 2.3.2.post1), e nao a ponta do main,
#   que pode andar -- e cuja versao declarada nao muda entre commits.
# O toolkit CUDA do conda so serve para COMPILAR: ele some no proximo reinicio e nao faz falta para rodar.
#
# RODAR LOGO DEPOIS DE UM REINICIO (/opt/conda limpo): so assim as dependencias que faltam na imagem tambem vao para a
# home. O script recusa se ja houver mamba_ssm fora da home. Depois, a cadeia confere o ambiente sozinha.
#
# USO (notebook, terminal):  cd ~/testeArq/lumina-beat-regionalization && bash scripts/instalar_ambiente_gpu_na_home.sh
set -euo pipefail
MAMBA_COMMIT="${MAMBA_COMMIT:-e9594ce1c732d97440f0332fdc43170a2294dbfa}"
cd "$(dirname "$0")/.."

USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())')"
FORA="$(python3 -c 'import importlib.util as u; e = u.find_spec("mamba_ssm"); print(e.origin if e else "")' 2>/dev/null || true)"
if [[ -n "${FORA}" && "${FORA}" != "${USER_SITE}"* ]]; then
    echo "FALHOU: ja existe mamba_ssm fora da home (${FORA}). Rode logo depois de um reinicio do espaco, com o"
    echo "        /opt/conda limpo; senao as dependencias ficariam no /opt/conda e sumiriam no proximo reinicio."
    exit 2
fi

echo "== toolkit CUDA para compilar (efemero, fica no /opt/conda)"
conda install -c "nvidia/label/cuda-12.9.0" cuda-cudart-dev cuda-nvcc -y
export CUDA_HOME=/opt/conda/targets/x86_64-linux
export PATH=/opt/conda/nvvm/bin:$PATH

echo "== pacotes na home (${USER_SITE})"
python3 -m pip install --user packaging setuptools wheel
CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="8.9" \
    python3 -m pip install --user --no-build-isolation causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="8.9" \
    python3 -m pip install --user --no-build-isolation "git+https://github.com/state-spaces/mamba.git@${MAMBA_COMMIT}"
python3 -m pip install --user -e .

echo "== conferencia"
python3 - <<'PY'
import importlib.metadata as md
import json
import site

import causal_conv1d
import mamba_ssm

direto = json.loads(md.distribution("mamba_ssm").read_text("direct_url.json") or "{}")
print("mamba_ssm", mamba_ssm.__version__, "commit", direto.get("vcs_info", {}).get("commit_id"), "em", mamba_ssm.__file__)
print("causal_conv1d", causal_conv1d.__version__, "em", causal_conv1d.__file__)
for modulo in (mamba_ssm, causal_conv1d):
    assert modulo.__file__.startswith(site.getusersitepackages()), f"{modulo.__name__} nao veio da home"
print("OK: mamba_ssm e causal_conv1d instalados na home")
PY
