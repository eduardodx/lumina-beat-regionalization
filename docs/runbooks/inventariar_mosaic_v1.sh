#!/usr/bin/env bash
# Fase 0: somente download, validacao e inventario. Nao extrai embeddings nem treina.
# Executar no Linux/SageMaker a partir do checkout de lumina-beat-regionalization.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
COMMIT=f2e9a9f2ee14084a7eb626df8ee2a28d8869f57e
MV=${MOSAIC_ENTREGA_DIR:-"$HOME/mosaic-v1-2026-09-30"}
MOSAIC_REPO=${MOSAIC_REPO_DIR:-"$HOME/testeArq/lumina-mosaic"}
A=${MOSAIC_INVENTARIO_DIR:-"$HOME/artifacts/mosaic_v1"}
H=$(date +%Y%m%d_%H%M%S)_$$
mkdir -p "$A"
LOG="$A/passo1_inventario_$H.log"
OUT="$A/inventario_$H"
MS3=s3://croma-bioai-lumina-releases-us-east-2/benchmarks/mosaic/releases/mosaic-v1-2026-09-30

passo() {
  local nome=$1; shift
  local s=0
  "$@" 2>&1 | tee -a "$LOG" || s=$?
  echo "exit_$nome=$s" | tee -a "$LOG"
  return "$s"
}
trap 's=$?; if (( s != 0 )); then echo "FALHOU: fase 0 interrompida (exit=$s); log $LOG" >&2; fi' EXIT

echo "revisao $(git rev-parse --short HEAD) | inicio $(date '+%F %T') | log $LOG" | tee -a "$LOG"
test -f "$REPO/scripts/inventario_mosaic_v1.py"
test -f "$REPO/tests/test_inventario_mosaic_v1.py"
command -v aws >/dev/null
df -h "$A" | tee -a "$LOG"

if [ ! -e "$MV" ]; then
  passo fetch git -C "$MOSAIC_REPO" fetch origin
  passo worktree git -C "$MOSAIC_REPO" worktree add --detach "$MV" "$COMMIT"
fi
if [ "$(git -C "$MV" rev-parse HEAD)" != "$COMMIT" ]; then
  echo "FALHOU: checkout do Mosaic fora do commit da entrega" | tee -a "$LOG"
  exit 1
fi

# O inventario e os testes precisam do MESMO Python que valida o Mosaic.
if command -v uv >/dev/null; then
  (cd "$MV" && passo uv_sync uv sync --frozen)
  PY=(uv run --project "$MV" --frozen python)
else
  PY=(python3)
  echo "sem uv: verificando as dependencias do Python do sistema" | tee -a "$LOG"
fi
passo ambiente env PYTHONPATH="$MV/src" "${PY[@]}" -c \
  'import sys; assert sys.version_info >= (3, 12), "Python >= 3.12 necessario"; import numpy, pandas, pyarrow, yaml; import mosaic.cli; print("Python:", sys.executable, "| dependencias conferidas")'
passo testes env PYTHONPATH="$REPO" "${PY[@]}" "$REPO/tests/test_inventario_mosaic_v1.py"

for p in artifacts/mosaic-v1-2026-09-30 config outputs/candidate-manifest; do
  passo "tamanho_${p//\//_}" aws s3 ls --recursive --summarize "$MS3/$p/" --region us-east-2
  passo "sync_${p//\//_}" aws s3 sync "$MS3/$p/" "$MV/$p/" \
    --region us-east-2 --exclude '.DS_Store' --only-show-errors
done
passo manifest aws s3 cp "$MS3/delivery.manifest.json" "$MV/outputs/delivery.manifest.json" \
  --region us-east-2 --only-show-errors
passo checksum_manifest bash -o pipefail -c \
  'printf "%s  %s\n" "$1" "$2" | sha256sum -c -' _ \
  1970f424f758e39275f43fe24a7dbd8294c41300e0e9917b45bd6bdc4877d7d8 \
  "$MV/outputs/delivery.manifest.json"

(cd "$MV" && passo validacao env PYTHONPATH="$MV/src" "${PY[@]}" \
  -m mosaic.cli validate-suite artifacts/mosaic-v1-2026-09-30)
(cd "$MV" && passo pedidos env PYTHONPATH="$MV/src" "${PY[@]}" - <<'PY'
import json
from pathlib import Path

import yaml
from mosaic.hashing import sha256_file
from mosaic.identity import verify_release_identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


identity = verify_release_identity(Path("artifacts/mosaic-v1-2026-09-30"))
require(identity["release_identity_hash"] == "d93125804e7cbc06d1187fd51580eebb758c81871ec381c29fc0a345644e0dc1",
        "identidade do release divergente")
require(identity["protocol_hash"] == "67470b1443c4bf1bb8abee2760508bf93a1c97a50f9e99b17056d4326ae43885",
        "hash do protocolo divergente")
protocol_path = Path("config/study-protocol.yaml")
require(yaml.safe_load(protocol_path.read_text())["release_identity"] == identity["release_identity_hash"],
        "protocolo de estudos de outro release")
m = json.loads(Path("outputs/candidate-manifest/manifest.json").read_text())
for key in ["release_identity_hash", "protocol_hash"]:
    require(m[key] == identity[key], f"pedidos: {key} divergente")
require(m["study_protocol_sha256"] == sha256_file(protocol_path), "protocolo dos pedidos divergente")
for name in ["variants", "requests"]:
    require(m[f"{name}_sha256"] == sha256_file(Path(f"outputs/candidate-manifest/{name}.parquet")),
            f"checksum dos pedidos {name} divergente")
print("PEDIDOS CONFEREM com o release e o protocolo")
PY
)
passo inventario env PYTHONPATH="$REPO" "${PY[@]}" "$REPO/scripts/inventario_mosaic_v1.py" \
  --entrega "$MV" \
  --cache-antigo "$HOME/artifacts/redesenho/g3_cache/M0" \
  --cache-antigo "$HOME/artifacts/redesenho/g7_cache/M0" \
  --out-dir "$OUT"
echo "VALIDACAO OK | inventario $OUT/inventario.json" | tee -a "$LOG"
echo "FIM $(date '+%F %T') | log $LOG" | tee -a "$LOG"
