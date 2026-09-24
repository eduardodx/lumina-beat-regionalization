#!/usr/bin/env bash
# Cadeia do MR_a2 e do MR_a3: extracao (GPU), comparador de desenvolvimento e conferencia das cabecas (CPU).
#
# RETOMAVEL: rodar de novo continua de onde parou. O extrator retoma pelos fragmentos gravados (e confere a
# identidade do cache); comparador com conferencia aprovada e pulado; pasta de comparador sem conferencia aprovada
# para a cadeia e pede inspecao, sem sobrescrever cabecas.
#
# AUTOVERIFICADA: antes de extrair, confere que codigo e ambiente sao os do cache do M0 e que o ambiente REPRODUZ,
# numero a numero, o M0 e todo cache MR ja comecado (scripts/conferir_reproducao_do_cache.py). Um reinicio do espaco
# do SageMaker apaga o que esta fora da home; se o ambiente mudou, a cadeia para aqui sem extrair nada.
#
# Para na primeira falha; cada passo imprime `exit_<passo>=<codigo>`; o fim imprime `CADEIA COMPLETA`.
#
# USO (notebook), com o interpretador em PY:
#   cd ~/testeArq/lumina-beat-regionalization && LOG=~/artifacts/redesenho/mr_a2_a3_$(date +%Y%m%d_%H%M%S).log && \
#     { PY="$(command -v python3)" nohup bash scripts/cadeia_mr_a2_a3.sh > "$LOG" 2>&1 & } && echo "pid=$! log=$LOG"
PY="${PY:-python3}"
A="$HOME/artifacts/redesenho"
BASE="--checkpoint $HOME/artifacts/r03/best_checkpoint.pt --fasta $HOME/hg38/hg38.fa"
COMUM="$BASE --snapshot $A/g2_final_nenhum/core_head_snapshot.parquet --selecao $A/g5_comum/selecao_comum.parquet --estudos $A/g1_brazil_studies/brazil_study_variants.parquet"
SNAP="--snapshot nenhum=$A/g2_final_nenhum/core_head_snapshot.parquet --snapshot janela2048=$A/g2_final_janela2048/core_head_snapshot.parquet --snapshot janela4096=$A/g2_final_janela4096/core_head_snapshot.parquet"
cd "$(dirname "$0")/.."

passo() { NOME="$1"; shift; PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" "$@"; S=$?; echo "exit_$NOME=$S"; if [ "$S" -ne 0 ]; then exit "$S"; fi; }

echo "interpretador: $PY | revisao: $(git rev-parse --short HEAD 2>/dev/null) | inicio: $(date '+%Y-%m-%d %H:%M:%S')"
passo ambiente "$PY" scripts/conferir_codigo_do_cache.py --cache "$A/g3_cache/M0"
passo reproducao_M0 "$PY" scripts/conferir_reproducao_do_cache.py --cache "$A/g3_cache/M0" $BASE
for i in 2 3; do
  if [ -f "$A/g3_cache/MR_a$i/fragmento_00000.npz" ]; then
    passo reproducao_a$i "$PY" scripts/conferir_reproducao_do_cache.py --cache "$A/g3_cache/MR_a$i" $BASE \
      --adapter "$A/g4_a$i/adapter_melhor.pt" --semente-do-adapter $((20260920 + i))
  fi
done
for i in 2 3; do
  passo extracao_a$i "$PY" scripts/extract_campaign_features.py --sistema MR --adapter "$A/g4_a$i/adapter_melhor.pt" \
    --semente-do-adapter $((20260920 + i)) $COMUM --out-dir "$A/g3_cache/MR_a$i"
done
for i in 2 3; do
  C="$A/comparacao_dev_a$i"
  if grep -qs '"passou": true' "$C/conferencia_das_cabecas.json"; then echo "comparacao_a$i ja conferida: pulando"; continue; fi
  if [ -e "$C" ]; then echo "FALHOU: $C existe sem conferencia aprovada; inspecionar antes de refazer"; exit 2; fi
  passo comparacao_a$i "$PY" scripts/comparar_m0_mr_desenvolvimento.py --cache-m0 "$A/g3_cache/M0" \
    --cache-mr "$A/g3_cache/MR_a$i" --decisao-g5 "$A/g5/g5_decisao.json" $SNAP --out-dir "$C"
  passo conferencia_a$i "$PY" scripts/conferir_cabecas_salvas.py --comparacao "$C" --cache-m0 "$A/g3_cache/M0" \
    --cache-mr "$A/g3_cache/MR_a$i" --decisao-g5 "$A/g5/g5_decisao.json" --m0-de-referencia "$A/comparacao_dev_a1"
done
echo "CADEIA COMPLETA $(date '+%Y-%m-%d %H:%M:%S')"
