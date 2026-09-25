#!/usr/bin/env bash
# Executar no SageMaker dentro de tmux/screen: bash docs/runbooks/extrair_g7_gpu.sh
# Apenas extracao: nao executa a avaliacao G7 nem modifica o manifesto congelado.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

A="$HOME/artifacts/redesenho"
M="$HOME/testeArq/lumina-mosaic"
H=$(date +%Y%m%d_%H%M%S)
LOG="$A/g7_extracao_$H.log"
G6="$A/g6_definitivo_20260925_193006"
CK="$HOME/artifacts/r03/best_checkpoint.pt"
FA="$HOME/hg38/hg38.fa"

git merge-base --is-ancestor 5325f4a HEAD || {
  echo "FALHOU: revisao sem o commit 5325f4a"
  exit 1
}

passo() {
  local nome=$1
  shift
  local s=0
  PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" "$@" 2>&1 | tee -a "$LOG" || s=$?
  echo "exit_${nome}=$s" | tee -a "$LOG"
  return "$s"
}

echo "revisao $(git rev-parse --short HEAD) | manifesto $(cat "$G6/g6_manifesto.json.sha256") | inicio $(date '+%F %T')" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$LOG"
declare -A NOME=([M0]=M0 [20260921]=MR_a1 [20260922]=MR_a2 [20260923]=MR_a3)
for S in M0 20260921 20260922 20260923; do
  if [ "$S" = M0 ]; then
    SIS=(--sistema M0)
  else
    SIS=(--sistema MR --semente-do-adapter "$S")
  fi
  echo "== $S inicio $(date '+%F %T')" | tee -a "$LOG"
  passo extrair_$S python3 scripts/extrair_estudos.py --manifesto "$G6" "${SIS[@]}" \
    --release-root "$HOME/mosaic-v1" --mosaic-root "$M" \
    --membros "$A/g1_brazil_studies/brazil_study_variants.parquet" \
    --checkpoint "$CK" --fasta "$FA" --out-dir "$A/g7_cache/${NOME[$S]}"
done
ls -la "$A/g7_cache" | tee -a "$LOG"
echo "FIM $(date '+%F %T') | log $LOG" | tee -a "$LOG"
