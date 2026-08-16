#!/usr/bin/env bash
# Writes <output_dir>/<base>.raw: uint8 RGB frames, shape (N, 874, 1164, 3), no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

ARCHIVE_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  python -m "submissions.${SUB_NAME}.inflate" "$ARCHIVE_DIR" "${OUTPUT_DIR}/${BASE}.raw"
done < "$FILE_LIST"
