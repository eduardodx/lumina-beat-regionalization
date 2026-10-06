#!/usr/bin/env bash
# Fase 1, cabeca com interacao (so CPU; desenvolvimento exploratorio posterior ao teste). Especificacao, congelada
# antes de rodar: docs/fase1_cabeca_interacao_especificacao.md. Nao altera os artefatos anteriores; tudo vai para
# pastas novas.
#   1. testes;
#   2. treino de H(E+F) e H(E+F+BR2) (scripts/fase1_interacao_treinar.py), que antes confere o S reaproveitado do
#      passo 3 (ids, execucao, s_resumo, purgas e C internos) e para se ele nao bater;
#   3. avaliador oficial (scripts/evaluate_candidate.py) nos dois bracos novos;
#   4. leituras dos quatro bracos (scripts/fase1_interacao_ler.py), com a reproducao do BR v2 e o veredito da leitura
#      declarada.
# Leva da ordem de 1 a 1,5 h. Rodar desacoplado, passando as pastas dos passos 3 e 4 e do BR v2:
#   B="$HOME/artifacts/mosaic_v1/bracos_20261004_171558_1832"
#   L="$HOME/artifacts/mosaic_v1/leituras_20261004_184608_3203"
#   T="$HOME/artifacts/mosaic_v1/br2_treino_20261004_230006_665"
#   L2="$HOME/artifacts/mosaic_v1/leituras_br2_20261004_230006_665"
#   SAIDA="$HOME/artifacts/mosaic_v1/interacao_saida_$(date +%Y%m%d_%H%M%S)_$$.out"
#   nohup setsid bash docs/runbooks/interacao_mosaic_v1.sh "$B" "$L" "$T" "$L2" > "$SAIDA" 2>&1 < /dev/null &
#   echo "pid=$! saida=$SAIDA"
#   tail -f "$SAIDA"
# A saida comeca com os testes sinteticos (tabelas pequenas, 20 replicas, /tmp); o resultado real fica em
# ~/artifacts/mosaic_v1/interacao_leituras_<H>/interacao.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$PWD
MV=${MOSAIC_ENTREGA_DIR:-"$HOME/mosaic-v1-2026-09-30"}
A="$HOME/artifacts/mosaic_v1"
R="$HOME/artifacts/redesenho"
USO="uso: bash docs/runbooks/interacao_mosaic_v1.sh <passo 3> <passo 4> <treino do BR v2> <leituras do BR v2>"
BRACOS=${1:?"$USO"}
LEITURAS=${2:?"$USO"}
TREINO_BR2=${3:?"$USO"}
LEITURAS_BR2=${4:?"$USO"}
H=$(date +%Y%m%d_%H%M%S)_$$
LOG="$A/interacao_$H.log"
TREINO="$A/interacao_treino_$H"
AVAL="$A/interacao_avaliacao_$H"
OUT="$A/interacao_leituras_$H"
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
trap 's=$?; if (( s != 0 )); then echo "FALHOU: cabeca com interacao interrompida (exit=$s); log $LOG" >&2; fi' EXIT

echo "revisao $(git rev-parse --short HEAD) | inicio $(date '+%F %T') | log $LOG" | tee -a "$LOG"
command -v uv >/dev/null
test -f "$BRACOS/selecao.json"
test -f "$BRACOS/diagnosticos/s_fora_da_amostra.parquet"
test -f "$LEITURAS/leituras.json"
test -f "$TREINO_BR2/fase1-e-f-br2/predictions.parquet"
test -f "$LEITURAS_BR2/br2.json"
test -f docs/fase1_cabeca_interacao_especificacao.md
echo "especificacao $(sha256sum docs/fase1_cabeca_interacao_especificacao.md)" | tee -a "$LOG"
{ echo "nproc $(nproc)"; free -g; df -h "$A"; } | tee -a "$LOG"

for t in test_fase1_interacao test_fase1_br2; do
  passo "$t" "${PY[@]}" "tests/$t.py"
done
passo treino "${PY[@]}" scripts/fase1_interacao_treinar.py --entrega "$MV" --bracos "$BRACOS" "${CACHES[@]}" \
  --out-dir "$TREINO"
for id in fase1-h-e-f fase1-h-e-f-br2; do
  (cd "$MV" && passo "avaliador_$id" "${PY[@]}" scripts/evaluate_candidate.py \
    --predictions "$TREINO/$id/predictions.parquet" --system "$TREINO/$id/system.yaml" \
    --release artifacts/mosaic-v1-2026-09-30 --protocol config/study-protocol.yaml --output-root "$AVAL")
done
passo leituras "${PY[@]}" scripts/fase1_interacao_ler.py --entrega "$MV" --bracos "$BRACOS" --leituras "$LEITURAS" \
  --br2 "$TREINO_BR2" --leituras-br2 "$LEITURAS_BR2" --h "$TREINO" --avaliacao-h "$AVAL" --out-dir "$OUT"
echo "FIM $(date '+%F %T') | treino $TREINO | avaliacao $AVAL | leituras $OUT/interacao.md" | tee -a "$LOG"
