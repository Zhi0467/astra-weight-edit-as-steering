#!/bin/bash
# The clas arm with corrective NTP data on Qwen3-1.7B (greedy, blocks 15-18); run after run/qwen3_exp1.sh, reusing its v
# and stopping-rule traces. NTP is trained on each pair's productive continuation placed after k = 2 .. (the pair's
# number of copies) copies of its doubt sentence (loop_s_learn.py --ntp-doubt; arm clasd). KL stays on the plain
# productive text. Everything else as for clas in experiment 1:
#  1. learned s for N = 4, 32 and beta 0, 0.1 (Adam lr 3e-3, 24 updates from s = 0).
#  2. Stopping update of each s (same rule and traces as experiment 1).
#  3. Dev rollouts of each s at its stopping update; per N, the highest pooled dev accuracy (loop rate breaks ties).
#  4. Test rollouts of the chosen arms on the 1,000 test prompts.
# Outputs under $WORK/qwen3: s_run/s/clasd_*, runs/{dev,test}/.../clasd_*, exp1/{devscore,selection}_clasd.json.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
Q=$WORK/qwen3; S=$Q/s_run; E=$Q/exp1; D=$Q/runs/dev; mkdir -p "$S/logs" "$E"
L="$PY scripts/steer/loop_s_learn.py"; A="--out $S --astra-dir $ASTRA --model $QWEN3"
for n in 4 32; do [ -e "$S/v_n$n.pt" ] || { echo "missing $S/v_n$n.pt: run run/qwen3_exp1.sh first"; exit 1; }; done
ARMS=(); for n in 4 32; do for b in 0 0.1; do ARMS+=(clasd_n${n}_b$b); done; done

# 1. learned s
: > "$E/jobs_train_clasd"
for arm in "${ARMS[@]}"; do n=${arm#*_n}; n=${n%%_*}; b=${arm##*_b}
  [ -e "$S/s/$arm.pt" ] || echo "CUDA_VISIBLE_DEVICES=\$GPU $L train $A --n $n --obj clas --ntp-doubt --beta $b > $S/logs/train_$arm.log 2>&1" >> "$E/jobs_train_clasd"; done
SLOTS=1 pool "$E/jobs_train_clasd"

# 2. stopping updates (traces as in run/qwen3_exp1.sh)
if [ "${STOP_TRACES:-own}" = ours ]; then
  TR=("$ROOT/data/qwen3/stop_traces/math_level5_dev_control_traces.jsonl.gz" "$ROOT/data/qwen3/stop_traces/livecodebench_dev_control_traces.jsonl.gz")
else
  TR=("$(ls $D/math_level5/control/*.jsonl.gz)" "$(ls $D/livecodebench/control/*.jsonl.gz)")
fi
[ -e "$E/devscore_clasd.json" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY scripts/steer/loop_s_devloss.py score --out "$S" --model "$QWEN3" \
  --bundle "${TR[0]}" --bundle "${TR[1]}" --arms "$(IFS=,; echo "${ARMS[*]}")" --json "$E/devscore_clasd.json" > "$E/devscore_clasd.log" 2>&1
STOP=$($PY -c "
import json, sys; sys.path.insert(0, 'scripts/steer')
from loop_s_devloss import choose
r = json.load(open('$E/devscore_clasd.json'))['arms']
print(' '.join(f'{k}@{choose(v[\"history\"], float(\"inf\"))}' for k, v in r.items()))")
echo "stopping updates: $STOP"

# 3. candidates, dev rollouts, selection
C=(); CANDS=()
for x in $STOP; do arm=${x%@*}; st=${x#*@}; n=${arm#*_n}; n=${n%%_*}; b=${arm##*_b}
  C+=(--cond clasd:$n:$b@$st); CANDS+=(${arm}_s$st); done
need=(); for i in "${!CANDS[@]}"; do [ -e "$S/checkpoints/${CANDS[$i]}/config.json" ] || need+=("${C[$((2 * i))]}" "${C[$((2 * i + 1))]}"); done
[ ${#need[@]} = 0 ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $L export $A "${need[@]}" > "$S/logs/export_clasd.log" 2>&1
: > "$E/jobs_dev_clasd"
for arm in "${CANDS[@]}"; do for ds in math_level5 livecodebench; do
  echo "bash run/gen_qwen3.sh $S/checkpoints/$arm $arm $ds dev" >> "$E/jobs_dev_clasd"; done; done
pool "$E/jobs_dev_clasd"
T=$E/devtable; mkdir -p "$T"; ln -sfn "$D" "$T/gen_dev"; ln -sfn "$D" "$T/graded_dev"
$PY scripts/steer/exp1_rollout_table.py "$T" > "$E/rollouts_dev_clasd.json"
$PY scripts/steer/exp1_select.py --only clasd --rollouts "$E/rollouts_dev_clasd.json" --scores "$E/devscore_clasd.json" \
  --json "$E/selection_clasd.json" | tee "$E/selection_clasd.txt"
CHOSEN=$($PY -c "import json; print(' '.join(a for a in json.load(open('$E/selection_clasd.json'))['test_arms'] if a != 'control'))")

# 4. test rollouts
: > "$E/jobs_test_clasd"
for src in train_source heldout_source; do for ds in math_level5 livecodebench; do for arm in $CHOSEN; do
  echo "bash run/gen_qwen3.sh $S/checkpoints/$arm $arm $ds test $src" >> "$E/jobs_test_clasd"; done; done; done
pool "$E/jobs_test_clasd"
R=$Q/runs/test
REFS=$($PY -c "import json; print(','.join(a for a in json.load(open('$E/exp1_selection.json'))['test_arms'] if a.startswith('sv_')))")
$PY scripts/steer/exp1_test_summary.py --gen "$R" --graded "$R" --split "$SPLITS/test12" \
  --arms "control,$REFS,$(echo "$CHOSEN" | tr ' ' ',')" --json "$E/test_summary_clasd.json"
echo "clasd (Qwen3) done: $CHOSEN"

