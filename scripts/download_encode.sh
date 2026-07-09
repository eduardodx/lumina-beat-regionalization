#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST_PATH="${1:-${ENCODE_TRACK_MANIFEST:-${REPO_ROOT}/data/encode/track_manifest.tsv}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/encode}"
DOWNLOAD_LOG="${OUTPUT_ROOT}/downloaded_tracks.tsv"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "Missing ENCODE track manifest: ${MANIFEST_PATH}" >&2
    echo "Create a tab-separated file with columns:" >&2
    echo "  assay<TAB>cell_line<TAB>track_name<TAB>url<TAB>transform<TAB>normalize" >&2
    echo "Example:" >&2
    echo "  DNase-seq<TAB>K562<TAB>dnase_fcoc<TAB>https://www.encodeproject.org/files/ENCFF.../@@download/ENCFF....bigWig<TAB>asinh<TAB>per_chromosome_zscore" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
printf "name\tbw_path\ttransform\tnormalize\n" > "${DOWNLOAD_LOG}"

while IFS=$'\t' read -r assay cell_line track_name url transform normalize; do
    if [[ -z "${assay}" || "${assay}" == \#* ]]; then
        continue
    fi
    if [[ -z "${cell_line}" || -z "${track_name}" || -z "${url}" ]]; then
        echo "Skipping malformed manifest row: assay='${assay}' cell_line='${cell_line}' track='${track_name}'" >&2
        continue
    fi

    safe_assay="${assay// /_}"
    safe_cell_line="${cell_line// /_}"
    destination_dir="${OUTPUT_ROOT}/${safe_assay}/${safe_cell_line}"
    destination_path="${destination_dir}/${track_name}.bw"

    mkdir -p "${destination_dir}"
    echo "Downloading ${track_name} -> ${destination_path}"
    curl --fail --location --retry 3 --retry-delay 2 "${url}" -o "${destination_path}.part"
    mv "${destination_path}.part" "${destination_path}"

    printf "%s\t%s\t%s\t%s\n" \
        "${track_name}" \
        "${destination_path#${REPO_ROOT}/}" \
        "${transform:-asinh}" \
        "${normalize:-per_chromosome_zscore}" \
        >> "${DOWNLOAD_LOG}"
done < "${MANIFEST_PATH}"

echo
echo "Downloaded ENCODE tracks under ${OUTPUT_ROOT}"
echo "Track summary written to ${DOWNLOAD_LOG}"
