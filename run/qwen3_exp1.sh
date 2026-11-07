#!/bin/bash
# Experiment 1 on Qwen3-1.7B (greedy, blocks 15-18): learned sensing vector s vs s = alpha v, v fixed.
#  1. v from the first N short pairs (N = 4: Astra's pairs; N = 32: all), and the 12 learned s
#     (objective clas | ul | contrast x beta 0 | 0.1 x N 4 | 32; Adam lr 3e-3, 24 full-batch updates from s = 0).
#  2. Unedited dev rollouts (MATH + LiveCodeBench dev, 128 prompts each).
#  3. Stopping update of each s: teacher-forced on those rollouts, the lowest loop score among updates whose </think>
#     top-1 rate is no worse than the unedited model's.
#  4. Dev rollouts of every candidate: s = alpha v (alpha 1-4) and each learned s at its stopping update.
#  5. Selection on dev: per family and N, the highest pooled dev accuracy (loop rate breaks ties); N by the best arm.
#  6. Test rollouts of the chosen arms, the unedited model and the unedited model with thinking off (1,000 prompts).
#  7. Paired summary (prompt bootstrap) and figures.
# Outputs under $WORK/qwen3: s_run/ (v, s, checkpoints), runs/, exp1/ (scores, tables, selection, summary).
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
Q=$WORK/qwen3; S=$Q/s_run; E=$Q/exp1; mkdir -p "$S/logs" "$E"
L="$PY scripts/steer/loop_s_learn.py"; A="--out $S --astra-dir $ASTRA --model $QWEN3"
ARMS12=(); for n in 4 32; do for obj in clas ul contrast; do for b in 0 0.1; do ARMS12+=(${obj}_n${n}_b$b); done; done; done

# 1. v and s
for n in 4 32; do [ -e "$S/v_n$n.pt" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $L direction $A --n $n > "$S/logs/direction_n$n.log" 2>&1; done
: > "$E/jobs_train"
for arm in "${ARMS12[@]}"; do
  obj=${arm%%_n*}; n=${arm#*_n}; n=${n%%_*}; b=${arm##*_b}
  [ -e "$S/s/$arm.pt" ] || echo "CUDA_VISIBLE_DEVICES=\$GPU $L train $A --n $n --obj $obj --beta $b > $S/logs/train_$arm.log 2>&1" >> "$E/jobs_train"
done
SLOTS=1 pool "$E/jobs_train"

# 2. unedited dev rollouts
printf '%s\n' "bash run/gen_qwen3.sh $QWEN3 control math_level5 dev" "bash run/gen_qwen3.sh $QWEN3 control livecodebench dev" > "$E/jobs_dev0"
pool "$E/jobs_dev0"

# 3. stopping updates
D=$Q/runs/dev
[ -e "$E/devscore.json" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY scripts/steer/loop_s_devloss.py score --out "$S" --model "$QWEN3" \
  --bundle "$(ls $D/math_level5/control/*.jsonl.gz)" --bundle "$(ls $D/livecodebench/control/*.jsonl.gz)" \
  --arms "$(IFS=,; echo "${ARMS12[*]}")" --json "$E/devscore.json" > "$E/devscore.log" 2>&1
STOP=$($PY -c "
import json, sys; sys.path.insert(0, 'scripts/steer')
from loop_s_devloss import choose
r = json.load(open('$E/devscore.json'))['arms']
print(' '.join(f'{k}@{choose(v[\"history\"], float(\"inf\"))}' for k, v in r.items()))")
echo "stopping updates: $STOP"

# 4. candidate checkpoints and dev rollouts
C=(); CANDS=()
for n in 4 32; do for al in 1 2 3 4; do C+=(--cond sv:$n:$al); CANDS+=(sv_n${n}_a$al); done; done
for x in $STOP; do arm=${x%@*}; st=${x#*@}; obj=${arm%%_n*}; n=${arm#*_n}; n=${n%%_*}; b=${arm##*_b}
  C+=(--cond $obj:$n:$b@$st); CANDS+=(${arm}_s$st); done
need=(); for i in "${!CANDS[@]}"; do [ -e "$S/checkpoints/${CANDS[$i]}/config.json" ] || need+=("${C[$((2 * i))]}" "${C[$((2 * i + 1))]}"); done
[ ${#need[@]} = 0 ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $L export $A "${need[@]}" > "$S/logs/export.log" 2>&1
: > "$E/jobs_dev"
for arm in "${CANDS[@]}"; do for ds in math_level5 livecodebench; do
  echo "bash run/gen_qwen3.sh $S/checkpoints/$arm $arm $ds dev" >> "$E/jobs_dev"; done; done
pool "$E/jobs_dev"

# 5. selection (exp1_rollout_table.py reads <root>/gen_dev/math_level5 and <root>/graded_dev/livecodebench)
T=$E/devtable; mkdir -p "$T"; ln -sfn "$D" "$T/gen_dev"; ln -sfn "$D" "$T/graded_dev"
$PY scripts/steer/exp1_rollout_table.py "$T" > "$E/rollouts_dev.json"
$PY scripts/steer/exp1_select.py --rollouts "$E/rollouts_dev.json" --scores "$E/devscore.json" --json "$E/exp1_selection.json" | tee "$E/selection.txt"

# 6. test rollouts
TEST=$(json_list "$E/exp1_selection.json" test_arms)
: > "$E/jobs_test"
for src in train_source heldout_source; do for ds in math_level5 livecodebench; do
  for arm in $TEST; do m=$S/checkpoints/$arm; [ $arm = control ] && m=$QWEN3
    echo "bash run/gen_qwen3.sh $m $arm $ds test $src" >> "$E/jobs_test"; done
  echo "THINKING=off bash run/gen_qwen3.sh $QWEN3 control_nothink $ds test $src" >> "$E/jobs_test"
done; done
pool "$E/jobs_test"

# 7. summary (exp1_test_summary.py reads <gen>/<src>/math_level5/<arm> and <graded>/<src>/livecodebench/<arm>)
R=$Q/runs/test
$PY scripts/steer/exp1_test_summary.py --gen "$R" --graded "$R" --split "$SPLITS/test12" \
  --arms "control_nothink,$(echo $TEST | tr ' ' ',')" --json "$E/exp1_test_summary.json" | tee "$E/exp1_test_summary.txt"
echo "experiment 1 (Qwen3) done: $E"
