#!/bin/bash
# One-time setup: Python environment, pinned models, LiveCodeBench, prepared data, Astra's materials.
set -euo pipefail
source "$(dirname "$0")/env.sh"
uv sync --frozen
mkdir -p "$WORK/models"
[ -e "$QWEN3/config.json" ] || "$ROOT/.venv/bin/hf" download "$QWEN3_ID" --revision "$QWEN3_REV" --local-dir "$QWEN3"
[ -e "$QWEN35/config.json" ] || "$ROOT/.venv/bin/hf" download "$QWEN35_ID" --revision "$QWEN35_REV" --local-dir "$QWEN35"
if [ ! -d "$LCB_REPO/.git" ]; then git clone "$LCB_GIT" "$LCB_REPO"; fi
git -C "$LCB_REPO" checkout --quiet "$LCB_COMMIT"
$PY scripts/data/prepare_datasets.py --out "$DATA"
$PY scripts/data/fetch_astra.py --out "$ASTRA"
echo "setup done"
