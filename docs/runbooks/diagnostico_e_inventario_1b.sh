#!/usr/bin/env bash
# Depois da Fase 1 (so CPU; nada aqui extrai embedding, treina ou pontua):
#   1. diagnostico posterior das perdas e ganhos de P-BR e das criticas (scripts/fase1_diagnostico.py);
#   2. preparacao da Fase 1b: tamanho das entradas no S3, download de data/annotations/bias-cells/ na entrega e
#      inventario das celulas de vies com as funcoes do proprio avaliador (scripts/fase1b_inventario.py).
# O CADD (exigido pelo avaliador de vies) so tem o tamanho listado; o download fica para quando a Fase 1b rodar.
# Uso, desacoplado do terminal (minutos, mais o download):
#   B="$HOME/artifacts/mosaic_v1/bracos_20261004_171558_1832"
#   L="$HOME/artifacts/mosaic_v1/leituras_20261004_184608_3203"
#   SAIDA="$HOME/artifacts/mosaic_v1/pos_fase1_saida_$(date +%Y%m%d_%H%M%S)_$$.out"
#   nohup setsid bash docs/runbooks/diagnostico_e_inventario_1b.sh "$B" "$L" > "$SAIDA" 2>&1 < /dev/null &
#   echo "pid=$! saida=$SAIDA"
#   tail -f "$SAIDA"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$PWD
MV=${MOSAIC_ENTREGA_DIR:-"$HOME/mosaic-v1-2026-09-30"}
A="$HOME/artifacts/mosaic_v1"
R="$HOME/artifacts/redesenho"
BRACOS=${1:?"uso: bash docs/runbooks/diagnostico_e_inventario_1b.sh <pasta do passo 3> <pasta do passo 4>"}
LEITURAS=${2:?"uso: bash docs/runbooks/diagnostico_e_inventario_1b.sh <pasta do passo 3> <pasta do passo 4>"}
H=$(date +%Y%m%d_%H%M%S)_$$
LOG="$A/pos_fase1_$H.log"
MS3=s3://croma-bioai-lumina-releases-us-east-2/benchmarks/mosaic/releases/mosaic-v1-2026-09-30
PY=(uv run --project "$MV" --frozen python)

passo() {
  local nome=$1
  shift
  local s=0
  PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" "$@" 2>&1 | tee -a "$LOG" || s=$?
  echo "exit_${nome}=$s" | tee -a "$LOG"
  return "$s"
}
trap 's=$?; if (( s != 0 )); then echo "FALHOU: pos-Fase 1 interrompido (exit=$s); log $LOG" >&2; fi' EXIT

echo "revisao $(git rev-parse --short HEAD) | inicio $(date '+%F %T') | log $LOG" | tee -a "$LOG"
command -v uv >/dev/null
command -v aws >/dev/null
test -f "$BRACOS/selecao.json"
test -f "$LEITURAS/leituras.json"
df -h "$A" "$MV" | tee -a "$LOG"

for t in test_fase1_diagnostico test_fase1b_inventario; do
  passo "$t" "${PY[@]}" "tests/$t.py"
done
passo diagnostico "${PY[@]}" scripts/fase1_diagnostico.py --entrega "$MV" --bracos "$BRACOS" --leituras "$LEITURAS" \
  --out-dir "$A/diagnostico_$H"

for p in data/annotations/bias-cells data/annotations/cadd/v1.7-genome; do
  echo "tamanho de $p:" | tee -a "$LOG"
  aws s3 ls --recursive --summarize "$MS3/$p/" --region us-east-2 | tail -n 2 | tee -a "$LOG"
done
passo sync_bias_cells aws s3 sync "$MS3/data/annotations/bias-cells/" "$MV/data/annotations/bias-cells/" \
  --region us-east-2 --exclude '.DS_Store' --only-show-errors
passo inventario_1b "${PY[@]}" scripts/fase1b_inventario.py --entrega "$MV" \
  --cache "$R/g3_cache/M0" --cache "$R/g7_cache/M0" --cache "$A/cache_M0_complemento" --out-dir "$A/inventario_1b_$H"
echo "FIM $(date '+%F %T') | diagnostico $A/diagnostico_$H/diagnostico.md | inventario $A/inventario_1b_$H/inventario_1b.json" \
  | tee -a "$LOG"
