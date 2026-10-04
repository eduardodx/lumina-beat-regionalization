#!/usr/bin/env bash
# Fase 1, BR v2 (so CPU; desenvolvimento exploratorio posterior ao teste). Especificacao, escrita antes de rodar:
# docs/fase1_br2_especificacao.md. Nao altera os artefatos dos passos 3 e 4; tudo vai para pastas novas.
#   1. testes;
#   2. conferencia rapida da selecao do passo 3 (regra revisada de convergencia);
#   3. treino do braco E+F+BR2 (scripts/fase1_br2_treinar.py), com as mesmas linhas e receita do passo 3;
#   4. avaliador oficial (scripts/evaluate_candidate.py) no braco novo;
#   5. leituras dos tres bracos (scripts/fase1_br2_ler.py), com o diagnostico das perdas, a recuperacao, os
#      coeficientes e, como auxiliar, as cabecas populacionais nativas do R03 em cabecas_172.
# Leva da ordem de 1 a 2 h. Rodar desacoplado, passando as pastas dos passos 3 e 4:
#   B="$HOME/artifacts/mosaic_v1/bracos_20261004_171558_1832"
#   L="$HOME/artifacts/mosaic_v1/leituras_20261004_184608_3203"
#   SAIDA="$HOME/artifacts/mosaic_v1/br2_saida_$(date +%Y%m%d_%H%M%S)_$$.out"
#   nohup setsid bash docs/runbooks/br2_mosaic_v1.sh "$B" "$L" > "$SAIDA" 2>&1 < /dev/null &
#   echo "pid=$! saida=$SAIDA"
#   tail -f "$SAIDA"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$PWD
MV=${MOSAIC_ENTREGA_DIR:-"$HOME/mosaic-v1-2026-09-30"}
A="$HOME/artifacts/mosaic_v1"
R="$HOME/artifacts/redesenho"
BRACOS=${1:?"uso: bash docs/runbooks/br2_mosaic_v1.sh <pasta do passo 3> <pasta do passo 4>"}
LEITURAS=${2:?"uso: bash docs/runbooks/br2_mosaic_v1.sh <pasta do passo 3> <pasta do passo 4>"}
H=$(date +%Y%m%d_%H%M%S)_$$
LOG="$A/br2_$H.log"
TREINO="$A/br2_treino_$H"
AVAL="$A/avaliacao_br2_$H"
OUT="$A/leituras_br2_$H"
CACHES=(--cache "$R/g3_cache/M0" --cache "$R/g7_cache/M0" --cache "$A/cache_M0_complemento")
PY=(uv run --project "$MV" --frozen python)

passo() {
  local nome=$1
  shift
  local s=0
  PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" "$@" 2>&1 | tee -a "$LOG" || s=$?
  echo "exit_${nome}=$s" | tee -a "$LOG"
  return "$s"
}
trap 's=$?; if (( s != 0 )); then echo "FALHOU: BR v2 interrompido (exit=$s); log $LOG" >&2; fi' EXIT

echo "revisao $(git rev-parse --short HEAD) | inicio $(date '+%F %T') | log $LOG" | tee -a "$LOG"
command -v uv >/dev/null
test -f "$BRACOS/selecao.json"
test -f "$LEITURAS/leituras.json"
test -f docs/fase1_br2_especificacao.md
echo "especificacao $(sha256sum docs/fase1_br2_especificacao.md)" | tee -a "$LOG"
{ echo "nproc $(nproc)"; free -g; df -h "$A"; } | tee -a "$LOG"

for t in test_fase1_br2 test_fase1_bracos test_fase1_leituras test_fase1_diagnostico; do
  passo "$t" "${PY[@]}" "tests/$t.py"
done
passo selecao "${PY[@]}" -c 'import json, sys; from scripts.fase1_leituras import conferir_selecao; print(conferir_selecao(json.load(open(sys.argv[1], encoding="utf-8")))["ajustes"], "ajustes do passo 3 valem")' "$BRACOS/selecao.json"
passo treino "${PY[@]}" scripts/fase1_br2_treinar.py --entrega "$MV" --bracos "$BRACOS" "${CACHES[@]}" \
  --out-dir "$TREINO"
(cd "$MV" && passo avaliador "${PY[@]}" scripts/evaluate_candidate.py \
  --predictions "$TREINO/fase1-e-f-br2/predictions.parquet" --system "$TREINO/fase1-e-f-br2/system.yaml" \
  --release artifacts/mosaic-v1-2026-09-30 --protocol config/study-protocol.yaml --output-root "$AVAL")
passo leituras "${PY[@]}" scripts/fase1_br2_ler.py --entrega "$MV" --bracos "$BRACOS" --br2 "$TREINO" \
  --leituras "$LEITURAS" --avaliacao-br2 "$AVAL" "${CACHES[@]}" --out-dir "$OUT"
echo "FIM $(date '+%F %T') | treino $TREINO | avaliacao $AVAL | leituras $OUT/br2.md" | tee -a "$LOG"
