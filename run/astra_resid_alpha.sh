#!/bin/bash
# Astra regime, residual site (needs run/astra_exp1.sh and run/astra_controls.sh first): s = alpha v with the N = 32
# direction, applied at run time to the block output of layers 15 and 19, for alpha in 0.2, 0.5, 1, 1.5.
# Dev (validation_1) picks alpha by the same rule as everywhere else: highest accuracy, then lower loop rate.
# The chosen alpha then runs on the 1,152 test problems, compared with the write-site and residual-site edits.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
SLOTS=${ASTRA_SLOTS:-3}
ALPHAS=${ALPHAS:-0.2 0.5 1 1.5}
W=$WORK/astra_regime; S=$W/s_run; G=$W/gen; K=$W/controls; mkdir -p "$K"
AR="$PY scripts/steer/astra_regime.py"
cd "$ROOT"
for al in $ALPHAS; do
  [ -e "$K/resid_sv_n32_a$al.pt" ] || $AR resid-spec --astra-src "$ASTRA" --v "$S/v_n32.pt" --alpha $al --dest "$K/resid_sv_n32_a$al.pt"
done
job() { echo "mkdir -p $G/$2/$1 && [ -e $G/$2/$1/summary.json ] || CUDA_VISIBLE_DEVICES=\$GPU $AR gen --astra-src $ASTRA --model $QWEN35 --split $2 --out $G/$2/$1 --gpu-mem \$MEM ${ASTRA_LIMIT:+--limit $ASTRA_LIMIT} --resid $K/$1.pt > $G/$2/$1/run.log 2>&1"; }

# dev
for al in $ALPHAS; do job resid_sv_n32_a$al validation_1; done > "$K/jobs_alpha_dev"
pool "$K/jobs_alpha_dev"
CHOSEN=$($PY - "$G" $ALPHAS <<'EOF'
import json, sys
from pathlib import Path
g, alphas = Path(sys.argv[1]), sys.argv[2:]
rows = []
for al in alphas:
    r = [json.loads(l) for l in open(g / "validation_1" / f"resid_sv_n32_a{al}" / "results.jsonl")]
    acc, loop = sum(x["correct"] for x in r) / len(r), sum(x["loop"] for x in r) / len(r)
    rows.append((-acc, loop, f"resid_sv_n32_a{al}"))
    print(f"resid_sv_n32_a{al}: dev accuracy {acc:.3f} loop {loop:.3f}", file=sys.stderr)
print(min(rows)[2])
EOF
)
echo "chosen on dev: $CHOSEN" | tee "$W/resid_alpha_selection.txt"

# test
for sp in validation_4 validation_3 final_1 final_2 final_3; do job $CHOSEN $sp; done > "$K/jobs_alpha_test"
pool "$K/jobs_alpha_test"
$PY scripts/steer/astra_regime_summary.py test --gen "$G" --arms $CHOSEN,resid_sv_n32_a4,resid_ul_n32_b0.1,sv_n32_a4,ul_n32_b0.1 \
  --refs control,sv_n32_a4,ul_n32_b0.1 --out "$W/test_summary_resid_alpha.json"
echo "residual alpha sweep done: $W"
