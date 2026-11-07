#!/bin/bash
# Experiment 2 on Qwen3-1.7B (greedy, blocks 15-18, N = 32, s = alpha v): how the steering vector v is built.
# Needs run/qwen3_exp1.sh first (it supplies v_n32, the unedited and s = alpha v dev/test runs).
#  1. v by construction from the 32 short pairs: mean difference on the MLP output (md_mlp), logistic regression
#     (logreg), recursive feature machine (rfm), first principal component (pc1); plus the random-direction control.
#  2. Dev rollouts at alpha 1-4 per construction; per construction the alpha with the highest pooled dev accuracy.
#  3. Test rollouts of the chosen arms and the random control at alpha 2; paired summary against the residual mean
#     difference (experiment 1's sv_n32_a2) and the unedited model.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
Q=$WORK/qwen3; S=$Q/s_run; E=$Q/exp2; mkdir -p "$E"
V="$PY scripts/steer/loop_v_constructions.py"
[ -e "$S/v_exp2_n32.pt" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $V build --out "$S" --astra-dir "$ASTRA" --model "$QWEN3" --n 32 > "$S/logs/v_exp2.log" 2>&1

ARMS=(); C=()
for c in md_mlp logreg rfm pc1; do for al in 1 2 3 4; do
  [ -e "$S/checkpoints/vc_${c}_n32_a$al/config.json" ] || C+=(--cond $c:$al); ARMS+=(vc_${c}_n32_a$al); done; done
[ -e "$S/checkpoints/vc_random_n32_a2/config.json" ] || C+=(--cond random:2)
[ ${#C[@]} = 0 ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $V export --out "$S" --model "$QWEN3" --n 32 "${C[@]}" > "$S/logs/export_exp2.log" 2>&1
: > "$E/jobs_dev"
for arm in "${ARMS[@]}"; do for ds in math_level5 livecodebench; do
  echo "bash run/gen_qwen3.sh $S/checkpoints/$arm $arm $ds dev" >> "$E/jobs_dev"; done; done
pool "$E/jobs_dev"
T=$E/devtable; mkdir -p "$T"; ln -sfn "$Q/runs/dev" "$T/gen_dev"; ln -sfn "$Q/runs/dev" "$T/graded_dev"
$PY scripts/steer/exp1_rollout_table.py "$T" > "$E/rollouts_dev.json"
$PY scripts/steer/exp2_select.py --rollouts "$E/rollouts_dev.json" --json "$E/exp2_selection.json"

TEST=$(json_list "$E/exp2_selection.json" test_arms)
: > "$E/jobs_test"
for src in train_source heldout_source; do for ds in math_level5 livecodebench; do for arm in $TEST; do
  echo "bash run/gen_qwen3.sh $S/checkpoints/$arm $arm $ds test $src" >> "$E/jobs_test"; done; done; done
pool "$E/jobs_test"
R=$Q/runs/test
$PY scripts/steer/exp1_test_summary.py --gen "$R" --graded "$R" --split "$SPLITS/test12" \
  --arms "control,sv_n32_a2,$(echo $TEST | tr ' ' ',')" --json "$E/exp2_test_summary.json" | tee "$E/exp2_test_summary.txt"
echo "experiment 2 done: $E"
