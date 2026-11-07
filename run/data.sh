#!/bin/bash
# Rebuild the Qwen3-1.7B dev and test prompt splits from the loop labels and check them against the shipped copies.
# The labels (data/qwen3/loop_labels.jsonl) come from 4 greedy draws per prompt; run/qwen3_reference_draws.sh makes
# new draws and scripts/data/loop_labels.py turns them into labels (greedy draws depend on hardware and batching, so
# new labels can differ slightly from ours, and so can the splits).
set -euo pipefail
source "$(dirname "$0")/env.sh"
LABELS=${LABELS:-data/qwen3/loop_labels.jsonl}
OUT=$WORK/splits
$PY scripts/steer/loop_dev_split.py --labels "$LABELS" --data "$DATA" --out "$OUT/dev"
$PY scripts/steer/loop_test_split.py --labels "$LABELS" --data "$DATA" --dev "$OUT/dev" --out "$OUT/test12"
diff -r "$OUT/dev" "$SPLITS/dev" && diff -r "$OUT/test12" "$SPLITS/test12" && echo "splits identical to data/qwen3/splits"
