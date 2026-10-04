#!/usr/bin/env bash
# Fase 1, passo 3: os seis bracos (F, F+BR, E, E+F, E+F+BR, S+F) nas cinco execucoes da vista de 4 kb.
# So CPU. Roda no .venv do Mosaic (uv): ele tem o sklearn e o pacote `mosaic` que confere a entrega, o bloco de
# frequencia oficial e o contrato das predicoes. Nao usa o python3 do conda.
# Leva da ordem de 1 a 2 h (regressoes de ~195 mil x ~1.350 colunas). O notebook nao tem tmux; rodar desacoplado:
#   SAIDA="$HOME/artifacts/mosaic_v1/passo3_saida_$(date +%Y%m%d_%H%M%S)_$$.out"
#   nohup setsid bash docs/runbooks/bracos_mosaic_v1.sh > "$SAIDA" 2>&1 < /dev/null &
#   echo "pid=$! saida=$SAIDA"
#   tail -f "$SAIDA"
# Isso protege contra o fechamento do terminal; nao contra desligar/reiniciar a instancia.
# Pico de memoria esperado: ~6-7 GB (leitura E em float32, 1,8 GB; treino em float64, 2,1 GB; produto do Newton).
# A saida vai para uma pasta nova; se cair no meio, rodar de novo (nao ha retomada: nada e gravado como valido
# antes do fim, e a pasta .tmp da tentativa anterior fica so para inspecao).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$PWD
MV=${MOSAIC_ENTREGA_DIR:-"$HOME/mosaic-v1-2026-09-30"}
A="$HOME/artifacts/mosaic_v1"
R="$HOME/artifacts/redesenho"
H=$(date +%Y%m%d_%H%M%S)_$$
LOG="$A/passo3_bracos_$H.log"
OUT="$A/bracos_$H"
mkdir -p "$A"
PY=(uv run --project "$MV" --frozen python)

passo() {
  local nome=$1
  shift
  local s=0
  PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" "$@" 2>&1 | tee -a "$LOG" || s=$?
  echo "exit_${nome}=$s" | tee -a "$LOG"
  return "$s"
}
trap 's=$?; if (( s != 0 )); then echo "FALHOU: passo 3 interrompido (exit=$s); log $LOG" >&2; fi' EXIT

echo "revisao $(git rev-parse --short HEAD) | inicio $(date '+%F %T') | log $LOG" | tee -a "$LOG"
command -v uv >/dev/null
test -d "$MV/artifacts/mosaic-v1-2026-09-30"
for c in "$R/g3_cache/M0" "$R/g7_cache/M0" "$A/cache_M0_complemento"; do
  test -f "$c/manifesto.json"
done
{ echo "nproc $(nproc)"; free -g; df -h "$A"; } | tee -a "$LOG"

passo ambiente "${PY[@]}" -c 'import sys, numpy, pandas, sklearn, yaml, mosaic.cli; print("Python:", sys.executable, sys.version.split()[0], "| numpy", numpy.__version__, "| pandas", pandas.__version__, "| sklearn", sklearn.__version__)'
for t in test_fase1_bracos test_inventario_mosaic_v1 test_extrair_mosaic_v1; do
  passo "$t" "${PY[@]}" "tests/$t.py"
done
passo bracos "${PY[@]}" scripts/fase1_bracos.py --entrega "$MV" \
  --cache "$R/g3_cache/M0" --cache "$R/g7_cache/M0" --cache "$A/cache_M0_complemento" --out-dir "$OUT"
echo "FIM $(date '+%F %T') | bracos $OUT | selecao $OUT/selecao.json" | tee -a "$LOG"
