#!/usr/bin/env bash
# Generate ClinVar embeddings
set -euo pipefail

# Nucleotide Transformers v3

python -m eval.clinvar.extract_embeddings \
    --model-family ntv3 \
    --model-version 8M_pre \
    --fasta-path data/hg38/hg38.fa \
    --context-size 4096 \
    --batch-size 896

# Caduceus

python -m eval.clinvar.extract_embeddings \
    --model-family caduceus \
    --model-version caduceus-ph \
    --fasta-path data/hg38/hg38.fa \
    --context-size 4096 \
    --batch-size 256

python -m eval.clinvar.extract_embeddings \
    --model-family caduceus \
    --model-version caduceus-ps \
    --fasta-path data/hg38/hg38.fa \
    --context-size 4096 \
    --batch-size 128

# DNABERT-2

python -m eval.clinvar.extract_embeddings \
    --model-family dnabert2 \
    --model-version 117M \
    --fasta-path data/hg38/hg38.fa \
    --context-size 2048 \
    --batch-size 32

# # Lumina

python -m eval.clinvar.extract_embeddings \
    --model-family lumina \
    --model-version bimamba3-rc \
    --checkpoint-path outputs/lumina_8m_bimamba3_rc_5ep_consmasking/best_checkpoint.pt \
    --fasta-path data/hg38/hg38.fa \
    --context-size 4096 \
    --batch-size 192


