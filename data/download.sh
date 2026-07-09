#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

HG38_DIR="${SCRIPT_DIR}/hg38"
GENCODE_DIR="${SCRIPT_DIR}/gencode"
PHYLO_DIR="${SCRIPT_DIR}/phylo"

HG38_FASTA_GZ="${HG38_DIR}/hg38.fa.gz"
HG38_FASTA="${HG38_DIR}/hg38.fa"
HG38_FASTA_FAI="${HG38_FASTA}.fai"
GENCODE_GTF_GZ="${GENCODE_DIR}/gencode.v38.annotation.gtf.gz"
PHYLO100_BW="${PHYLO_DIR}/hg38.phyloP100way.bw"
PHYLO470_BW="${PHYLO_DIR}/hg38.phyloP470way.bw"

: "${GENCODE_URL:=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz}"
: "${HG38_URL:=https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/latest/hg38.fa.gz}"
: "${PHYLO100_URL:=https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw}"
: "${PHYLO470_URL:=https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP470way/hg38.phyloP470way.bw}"

FORCE=0
REMOVE_FASTA_ARCHIVE=0

usage() {
  cat <<'EOF'
Usage: data/download.sh [--force] [--remove-fasta-archive]

Downloads and prepares the reference files expected by this repository:
  - data/hg38/hg38.fa
  - data/hg38/hg38.fa.fai
  - data/phylo/hg38.phyloP100way.bw
  - data/phylo/hg38.phyloP470way.bw
  - data/gencode/gencode.v38.annotation.gtf.gz

Options:
  --force                 Re-download files and rebuild prepared outputs.
  --remove-fasta-archive  Delete data/hg38/hg38.fa.gz after hg38.fa is prepared.
  -h, --help              Show this help message.

Environment overrides:
  GENCODE_URL
  HG38_URL
  PHYLO100_URL
  PHYLO470_URL
EOF
}

log() {
  printf '[download] %s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

download_with_curl() {
  local url="$1"
  local part_path="$2"

  if [[ -f "$part_path" ]]; then
    if ! curl -L --fail --retry 5 --retry-delay 5 --continue-at - --output "$part_path" "$url"; then
      log "Resume failed for $(basename "$part_path"); restarting from scratch"
      rm -f "$part_path"
      curl -L --fail --retry 5 --retry-delay 5 --output "$part_path" "$url"
    fi
    return
  fi

  curl -L --fail --retry 5 --retry-delay 5 --output "$part_path" "$url"
}

download_with_wget() {
  local url="$1"
  local part_path="$2"
  wget --tries=5 --continue --output-document "$part_path" "$url"
}

download_file() {
  local url="$1"
  local dest_path="$2"
  local part_path="${dest_path}.part"

  mkdir -p "$(dirname "$dest_path")"

  if [[ -s "$dest_path" && "$FORCE" -eq 0 ]]; then
    log "Using existing $(basename "$dest_path")"
    return
  fi

  if [[ "$FORCE" -eq 1 ]]; then
    rm -f "$part_path"
  fi

  log "Downloading $(basename "$dest_path")"
  if have_command curl; then
    download_with_curl "$url" "$part_path"
  elif have_command wget; then
    download_with_wget "$url" "$part_path"
  else
    die "Neither curl nor wget is installed"
  fi

  mv -f "$part_path" "$dest_path"
}

decompress_hg38() {
  if [[ -s "$HG38_FASTA" && "$FORCE" -eq 0 ]]; then
    log "Using existing $(basename "$HG38_FASTA")"
    return
  fi

  log "Decompressing $(basename "$HG38_FASTA_GZ")"
  python3 - "$HG38_FASTA_GZ" "$HG38_FASTA" <<'PY'
import gzip
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
tmp = f"{dst}.tmp"

with gzip.open(src, "rb") as handle_in, open(tmp, "wb") as handle_out:
    shutil.copyfileobj(handle_in, handle_out, length=1024 * 1024)

os.replace(tmp, dst)
PY
}

build_fasta_index() {
  if [[ -s "$HG38_FASTA_FAI" && "$FORCE" -eq 0 ]]; then
    log "Using existing $(basename "$HG38_FASTA_FAI")"
    return
  fi

  log "Building FASTA index $(basename "$HG38_FASTA_FAI")"
  # Write a standard .fai index without depending on samtools.
  python3 - "$HG38_FASTA" "$HG38_FASTA_FAI" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

fasta_path = Path(sys.argv[1])
fai_path = Path(sys.argv[2])
tmp_path = Path(f"{fai_path}.tmp")

name: str | None = None
length = 0
offset = 0
line_bases = 0
line_width = 0
records: list[tuple[str, int, int, int, int]] = []


def flush_record() -> None:
    global name, length, offset, line_bases, line_width
    if name is None:
        return
    if line_bases == 0:
        raise SystemExit(f"Sequence {name!r} has no bases")
    records.append((name, length, offset, line_bases, line_width))


with fasta_path.open("rb") as handle:
    while True:
        line_start = handle.tell()
        line = handle.readline()
        if not line:
            break

        if line.startswith(b">"):
            flush_record()
            header = line[1:].strip().decode("utf-8")
            if not header:
                raise SystemExit("Encountered an empty FASTA header")

            name = header.split()[0]
            length = 0
            offset = handle.tell()
            line_bases = 0
            line_width = 0
            continue

        if name is None:
            raise SystemExit(f"Found sequence data before any FASTA header at byte {line_start}")

        stripped = line.rstrip(b"\r\n")
        if not stripped:
            continue

        if line_bases == 0:
            line_bases = len(stripped)
            line_width = len(line)

        length += len(stripped)

flush_record()

with tmp_path.open("w", encoding="utf-8", newline="") as handle:
    for record_name, record_length, record_offset, record_line_bases, record_line_width in records:
        handle.write(
            f"{record_name}\t{record_length}\t{record_offset}\t{record_line_bases}\t{record_line_width}\n"
        )

os.replace(tmp_path, fai_path)
PY
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      ;;
    --remove-fasta-archive)
      REMOVE_FASTA_ARCHIVE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

have_command python3 || die "python3 is required"

mkdir -p "$HG38_DIR" "$GENCODE_DIR" "$PHYLO_DIR"

download_file "$GENCODE_URL" "$GENCODE_GTF_GZ"
download_file "$HG38_URL" "$HG38_FASTA_GZ"
download_file "$PHYLO100_URL" "$PHYLO100_BW"
download_file "$PHYLO470_URL" "$PHYLO470_BW"

decompress_hg38
build_fasta_index

if [[ "$REMOVE_FASTA_ARCHIVE" -eq 1 ]]; then
  log "Removing $(basename "$HG38_FASTA_GZ")"
  rm -f "$HG38_FASTA_GZ"
fi

log "Data is ready:"
printf '  %s\n' \
  "data/hg38/hg38.fa" \
  "data/hg38/hg38.fa.fai" \
  "data/phylo/hg38.phyloP100way.bw" \
  "data/phylo/hg38.phyloP470way.bw" \
  "data/gencode/gencode.v38.annotation.gtf.gz"
