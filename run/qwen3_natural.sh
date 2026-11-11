#!/bin/bash
# Section 4 comparison on Qwen3-1.7B (greedy, blocks 15-18): steering data from the model's own traces instead of the
# short synthetic pairs, everything else as in run/qwen3_exp1.sh (run that first; this reuses its unedited dev
# rollouts, stopping-rule traces, short-pair dev rollouts and test rollouts).
#  1. Traces: 32 looping + 32 finishing unedited dev rollouts (36-token windows), leaving out the prompts the stopping
#     rule scores. v from them; learned ul s for beta 0 and 0.1 (Adam lr 3e-3, 24 full-batch updates from s = 0).
#  2. Stopping update of each s (same rule and traces as experiment 1).
#  3. Dev rollouts of s = alpha v (alpha 1-4) and each learned s at its stopping update.
#  4. Selection on the reduced dev (dev minus the 64 trace prompts), for the natural arms and, on the same prompts, the
#     short pairs' N = 32 s = alpha v and ul arms; test rollouts of the chosen arms (a re-chosen short arm included).
#  5. Paired summary against the unedited model and between the two data sources.
# Outputs under $WORK/qwen3/natural: s_run/ (traces, v, s, checkpoints), runs/ (dev and test rollouts), summaries.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
Q=$WORK/qwen3; E=$Q/exp1; D=$Q/runs/dev; N=$Q/natural; S=$N/s_run; NR=$N/runs; mkdir -p "$S/logs"
[ -e "$E/exp1_selection.json" ] || { echo "run run/qwen3_exp1.sh first"; exit 1; }
B=(--bundle "$(ls $D/math_level5/control/*.jsonl.gz)" --bundle "$(ls $D/livecodebench/control/*.jsonl.gz)")
if [ "${STOP_TRACES:-own}" = ours ]; then  # as in run/qwen3_exp1.sh
  TR=("$ROOT/data/qwen3/stop_traces/math_level5_dev_control_traces.jsonl.gz" "$ROOT/data/qwen3/stop_traces/livecodebench_dev_control_traces.jsonl.gz")
else
  TR=("$(ls $D/math_level5/control/*.jsonl.gz)" "$(ls $D/livecodebench/control/*.jsonl.gz)")
fi
ND="$PY scripts/steer/natural_direction.py"

# 1. traces, v, learned s (training backpropagates through traces of up to 40,960 tokens: about 14 GB per GPU)
[ -e "$S/natural_traces.json" ] || $ND select "${B[@]}" --stop-traces "${TR[0]}" --stop-traces "${TR[1]}" --out "$S" > "$S/logs/select.log" 2>&1
[ -e "$S/v_n32.pt" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $ND direction "${B[@]}" --out "$S" --model "$QWEN3" \
  --ref-v "$Q/s_run/v_n32.pt" > "$S/logs/direction.log" 2>&1
: > "$N/jobs_train"
for b in 0 0.1; do [ -e "$S/s/ul_n32_b$b.pt" ] || echo "CUDA_VISIBLE_DEVICES=\$GPU $ND train ${B[*]} --out $S --model $QWEN3 --beta $b > $S/logs/train_b$b.log 2>&1" >> "$N/jobs_train"; done
SLOTS=1 pool "$N/jobs_train"

# 2. stopping updates
[ -e "$N/devscore.json" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY scripts/steer/loop_s_devloss.py score --out "$S" --model "$QWEN3" \
  --bundle "${TR[0]}" --bundle "${TR[1]}" --arms ul_n32_b0,ul_n32_b0.1 --json "$N/devscore.json" > "$N/devscore.log" 2>&1
STOP=$($PY -c "
import json, sys; sys.path.insert(0, 'scripts/steer')
from loop_s_devloss import choose
r = json.load(open('$N/devscore.json'))['arms']
print(' '.join(f'{k}@{choose(v[\"history\"], float(\"inf\"))}' for k, v in r.items()))")
echo "stopping updates: $STOP"

# 3. candidate checkpoints and dev rollouts (own RUNS: the arm names repeat experiment 1's)
C=(); CANDS=()
for al in 1 2 3 4; do C+=(--cond sv:32:$al); CANDS+=(sv_n32_a$al); done
for x in $STOP; do arm=${x%@*}; st=${x#*@}; b=${arm##*_b}; C+=(--cond ul:32:$b@$st); CANDS+=(${arm}_s$st); done
need=(); for i in "${!CANDS[@]}"; do [ -e "$S/checkpoints/${CANDS[$i]}/config.json" ] || need+=("${C[$((2 * i))]}" "${C[$((2 * i + 1))]}"); done
[ ${#need[@]} = 0 ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY scripts/steer/loop_s_learn.py export --out "$S" --model "$QWEN3" "${need[@]}" > "$S/logs/export.log" 2>&1
: > "$N/jobs_dev"
for arm in "${CANDS[@]}"; do for ds in math_level5 livecodebench; do
  echo "RUNS=$NR bash run/gen_qwen3.sh $S/checkpoints/$arm $arm $ds dev" >> "$N/jobs_dev"; done; done
pool "$N/jobs_dev"

# 4. selection on the reduced dev, both data sources; test rollouts
SEL="$PY scripts/steer/natural_dev_select.py --drop $S/natural_traces_summary.json"
$SEL --runs "$NR/dev" --arms "$(IFS=,; echo "${CANDS[*]}")" --json "$N/dev_selection_natural.json" | tee "$N/dev_selection_natural.txt"
SHORT=$($PY -c "
import json; s = json.load(open('$E/exp1_selection.json'))
print(','.join(x['arm'] for f in ('sv_n32', 'ul_n32') for x in s['families'][f]['candidates']))")
$SEL --runs "$D" --arms "$SHORT" --json "$N/dev_selection_short.json" | tee "$N/dev_selection_short.txt"
pick() { $PY -c "import json; print(json.load(open('$1'))['chosen']['$2'])"; }
NSV=$(pick "$N/dev_selection_natural.json" sv); NUL=$(pick "$N/dev_selection_natural.json" ul)
SSV=$(pick "$N/dev_selection_short.json" sv); SUL=$(pick "$N/dev_selection_short.json" ul)
: > "$N/jobs_test"
for src in train_source heldout_source; do for ds in math_level5 livecodebench; do
  for arm in $NSV $NUL; do echo "RUNS=$NR bash run/gen_qwen3.sh $S/checkpoints/$arm $arm $ds test $src" >> "$N/jobs_test"; done
  for arm in $SSV $SUL; do echo "bash run/gen_qwen3.sh $Q/s_run/checkpoints/$arm $arm $ds test $src" >> "$N/jobs_test"; done  # no-op if done
done; done
pool "$N/jobs_test"

# 5. summary
R=$Q/runs/test
$PY scripts/steer/natural_test_summary.py --split "$SPLITS/test12" --control "$R" "$R" --short "$R" "$R" "$SSV" "$SUL" \
  --natural "$NR/test" "$NR/test" "$NSV" "$NUL" --json "$N/test_summary.json" | tee "$N/test_summary.txt"
echo "natural-trace comparison done: $N"
