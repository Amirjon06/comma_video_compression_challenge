#!/usr/bin/env bash
# Builds archive.zip from a trained checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

CHECKPOINT="${ROOT}/.hnerv_cache/hnerv.pt"
WEIGHT_BITS="6"
EMBED_BITS="6"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --weight-bits) WEIGHT_BITS="$2"; shift 2 ;;
    --embed-bits) EMBED_BITS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
python -m submissions.hnerv_metric.pack \
  --checkpoint "$CHECKPOINT" \
  --out "${HERE}/archive.zip" \
  --weight-bits "$WEIGHT_BITS" \
  --embed-bits "$EMBED_BITS"
