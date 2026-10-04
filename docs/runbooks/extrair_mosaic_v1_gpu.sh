#!/usr/bin/env bash
# Fase 1, passo 2: leitura do M0 (R03 congelado) no release novo. O notebook nao tem tmux: rodar desacoplado do
# terminal, para sobreviver se a aba fechar, e acompanhar pelo arquivo de saida:
#   nohup setsid bash docs/runbooks/extrair_mosaic_v1_gpu.sh > ~/artifacts/mosaic_v1/passo2_saida.out 2>&1 < /dev/null &
#   tail -f ~/artifacts/mosaic_v1/passo2_saida.out
# Primeiro confere os caches antigos numa amostra (minutos). So se passar, extrai o complemento (~2 h).
# Usa o python3 do conda (torch e Mamba-3), como a extracao da campanha; nao usa o .venv do Mosaic.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$PWD
MV=${MOSAIC_ENTREGA_DIR:-"$HOME/mosaic-v1-2026-09-30"}
A="$HOME/artifacts/mosaic_v1"
R="$HOME/artifacts/redesenho"
H=$(date +%Y%m%d_%H%M%S)
LOG="$A/passo2_extracao_$H.log"
CK="$HOME/artifacts/r03/best_checkpoint.pt"
FA="$HOME/hg38/hg38.fa"
CONF="$A/cache_conferencia_$H"
# Pasta fixa: se a extracao cair, rodar de novo retoma dos fragmentos ja gravados (mesma identidade e tabela).
COMP="$A/cache_M0_complemento"
mkdir -p "$A"

passo() {
  local nome=$1
  shift
  local s=0
  PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" "$@" 2>&1 | tee -a "$LOG" || s=$?
  echo "exit_${nome}=$s" | tee -a "$LOG"
  return "$s"
}
trap 's=$?; if (( s != 0 )); then echo "FALHOU: passo 2 interrompido (exit=$s); log $LOG" >&2; fi' EXIT

echo "revisao $(git rev-parse --short HEAD) | inicio $(date '+%F %T') | log $LOG" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$LOG"
df -h "$A" | tee -a "$LOG"
test -d "$MV/artifacts/mosaic-v1-2026-09-30"

COMUM=(--entrega "$MV" --referencia "$R/g3_cache/M0" --cache-antigo "$R/g3_cache/M0" --cache-antigo "$R/g7_cache/M0"
       --checkpoint "$CK" --fasta "$FA")
for t in test_extrair_mosaic_v1 test_inventario_mosaic_v1; do
  passo "$t" python3 "tests/$t.py"
done
passo conferencia python3 scripts/extrair_mosaic_v1.py --modo conferencia "${COMUM[@]}" --out-dir "$CONF"
passo complemento python3 scripts/extrair_mosaic_v1.py --modo complemento "${COMUM[@]}" --out-dir "$COMP"
echo "FIM $(date '+%F %T') | conferencia $CONF/conferencia.json | complemento $COMP/manifesto.json" | tee -a "$LOG"
