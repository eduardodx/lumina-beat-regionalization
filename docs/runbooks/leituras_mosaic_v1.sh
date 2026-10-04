#!/usr/bin/env bash
# Fase 1, passo 4: leituras dos seis bracos do passo 3. So CPU, no .venv do Mosaic (uv).
#   1. o avaliador oficial (scripts/evaluate_candidate.py) em cada braco, na worktree da entrega;
#   2. as leituras do consumidor (scripts/fase1_leituras.py): contrastes oficiais, proxies, beneficio, P-BR e as
#      13 criticas. Elas conferem os limiares contra o avaliador e recusam um passo 3 que nao valha com a regra
#      revisada de convergencia.
# Leva da ordem de 1 a 2 h. O notebook nao tem tmux; rodar desacoplado, passando a pasta do passo 3:
#   B="$HOME/artifacts/mosaic_v1/bracos_20261004_171558_1832"
#   SAIDA="$HOME/artifacts/mosaic_v1/passo4_saida_$(date +%Y%m%d_%H%M%S)_$$.out"
#   nohup setsid bash docs/runbooks/leituras_mosaic_v1.sh "$B" > "$SAIDA" 2>&1 < /dev/null &
#   echo "pid=$! saida=$SAIDA"
#   tail -f "$SAIDA"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$PWD
MV=${MOSAIC_ENTREGA_DIR:-"$HOME/mosaic-v1-2026-09-30"}
A="$HOME/artifacts/mosaic_v1"
BRACOS=${1:?"uso: bash docs/runbooks/leituras_mosaic_v1.sh <pasta do passo 3 (bracos_...)>"}
H=$(date +%Y%m%d_%H%M%S)_$$
LOG="$A/passo4_leituras_$H.log"
AVAL="$A/avaliacao_$H"
OUT="$A/leituras_$H"
PY=(uv run --project "$MV" --frozen python)

passo() {
  local nome=$1
  shift
  local s=0
  PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" "$@" 2>&1 | tee -a "$LOG" || s=$?
  echo "exit_${nome}=$s" | tee -a "$LOG"
  return "$s"
}
trap 's=$?; if (( s != 0 )); then echo "FALHOU: passo 4 interrompido (exit=$s); log $LOG" >&2; fi' EXIT

echo "revisao $(git rev-parse --short HEAD) | inicio $(date '+%F %T') | log $LOG | passo 3 $BRACOS" | tee -a "$LOG"
command -v uv >/dev/null
test -d "$MV/artifacts/mosaic-v1-2026-09-30"
test -f "$BRACOS/selecao.json"
{ echo "nproc $(nproc)"; free -g; df -h "$A"; } | tee -a "$LOG"

passo ambiente "${PY[@]}" -c 'import sys, numpy, pandas, sklearn, yaml, scipy, mosaic.cli; print("Python:", sys.executable, sys.version.split()[0], "| pandas", pandas.__version__, "| scipy", scipy.__version__)'
for t in test_fase1_leituras test_fase1_bracos; do
  passo "$t" "${PY[@]}" "tests/$t.py"
done
# Falha em segundos, antes do avaliador, se o passo 3 nao valer com a regra revisada de convergencia (ef4b29d).
passo selecao "${PY[@]}" -c 'import json, sys; from scripts.fase1_leituras import conferir_selecao; r = conferir_selecao(json.load(open(sys.argv[1], encoding="utf-8"))); print("selecao do passo 3 vale com a regra revisada:", {k: r[k] for k in ("ajustes", "nao_convergidos_fora_da_escolha")}, "| avisos:", r["avisos"][:5])' "$BRACOS/selecao.json"
for id in fase1-f fase1-f-br fase1-e fase1-e-f fase1-e-f-br fase1-s-f; do
  (cd "$MV" && passo "avaliador_$id" "${PY[@]}" scripts/evaluate_candidate.py \
    --predictions "$BRACOS/$id/predictions.parquet" --system "$BRACOS/$id/system.yaml" \
    --release artifacts/mosaic-v1-2026-09-30 --protocol config/study-protocol.yaml --output-root "$AVAL")
done
passo leituras "${PY[@]}" scripts/fase1_leituras.py --entrega "$MV" --bracos "$BRACOS" --avaliacao "$AVAL" \
  --out-dir "$OUT"
echo "FIM $(date '+%F %T') | avaliacao $AVAL | leituras $OUT | resumo $OUT/resumo.md" | tee -a "$LOG"
