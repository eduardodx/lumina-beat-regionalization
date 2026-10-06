#!/usr/bin/env bash
# Fase 1, BR v2: caso a caso das P-BR perdidas e recuperadas (so CPU; descritivo, posterior ao teste). Pedido da
# revisao de 06/10 (docs/fase1_br2_resultado.md, proximos passos). Nao altera nenhum artefato; grava numa pasta nova.
#   1. teste do BR v2 (inclui o caso a caso no cenario sintetico);
#   2. scripts/fase1_br2_casos.py sobre os passos 3 e 4 e o BR v2.
# Leva poucos minutos; pode rodar em primeiro plano:
#   B="$HOME/artifacts/mosaic_v1/bracos_20261004_171558_1832"
#   L="$HOME/artifacts/mosaic_v1/leituras_20261004_184608_3203"
#   T="$HOME/artifacts/mosaic_v1/br2_treino_20261004_230006_665"
#   L2="$HOME/artifacts/mosaic_v1/leituras_br2_20261004_230006_665"
#   bash docs/runbooks/br2_casos_mosaic_v1.sh "$B" "$L" "$T" "$L2"
# O resultado fica em ~/artifacts/mosaic_v1/casos_br2_<H>/casos.md (e casos.csv, casos.json).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$PWD
MV=${MOSAIC_ENTREGA_DIR:-"$HOME/mosaic-v1-2026-09-30"}
A="$HOME/artifacts/mosaic_v1"
USO="uso: bash docs/runbooks/br2_casos_mosaic_v1.sh <passo 3> <passo 4> <treino do BR v2> <leituras do BR v2>"
BRACOS=${1:?"$USO"}
LEITURAS=${2:?"$USO"}
TREINO=${3:?"$USO"}
LEITURAS_BR2=${4:?"$USO"}
H=$(date +%Y%m%d_%H%M%S)_$$
LOG="$A/casos_br2_$H.log"
OUT="$A/casos_br2_$H"
PY=(uv run --project "$MV" --frozen python)

passo() {
  local nome=$1
  shift
  local s=0
  PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" "$@" 2>&1 | tee -a "$LOG" || s=$?
  echo "exit_${nome}=$s" | tee -a "$LOG"
  return "$s"
}
trap 's=$?; if (( s != 0 )); then echo "FALHOU: caso a caso interrompido (exit=$s); log $LOG" >&2; fi' EXIT

echo "revisao $(git rev-parse --short HEAD) | inicio $(date '+%F %T') | log $LOG" | tee -a "$LOG"
command -v uv >/dev/null
test -f "$BRACOS/selecao.json"
test -f "$LEITURAS/leituras.json"
test -f "$TREINO/fase1-e-f-br2/predictions.parquet"
test -f "$LEITURAS_BR2/br2.json"
passo test_fase1_br2 "${PY[@]}" tests/test_fase1_br2.py
passo casos "${PY[@]}" scripts/fase1_br2_casos.py --entrega "$MV" --bracos "$BRACOS" --leituras "$LEITURAS" \
  --br2 "$TREINO" --leituras-br2 "$LEITURAS_BR2" --out-dir "$OUT"
echo "FIM $(date '+%F %T') | caso a caso $OUT/casos.md" | tee -a "$LOG"
