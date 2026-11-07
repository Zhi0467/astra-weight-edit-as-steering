#!/bin/bash
# Experiment 1 in Astra's regime: Qwen3.5-0.8B, thinking on, Astra's sampler (T 0.6, top-p 0.95, top-k 20, presence
# penalty 1.5, 32,768 new tokens), layers 15 and 19, Astra's problems and grading.
#  1. Astra's submitted edit rebuilt bit-exactly (astra_top64_a6); v (N = 4, 32) and the 12 learned s.
#  2. Unedited dev rollouts (validation_1, 128 problems); stopping update of each s: teacher-forced on those rollouts,
#     the lowest loop score among updates whose </think> top-1 rate is no worse than the unedited model's.
#  3. Dev rollouts of every candidate (s = alpha v, alpha 1-4; learned s at its stopping update; Astra's edit).
#  4. Selection on dev per family and N (pooled dev accuracy, loop rate breaks ties).
#  5. Test rollouts (validation_3, validation_4, final_1-3: 1,152 problems) of the chosen arms, Astra's edit, the
#     unedited model with thinking on and off; paired summaries.
# Outputs under $WORK/astra_regime.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
SLOTS=${ASTRA_SLOTS:-3}
W=$WORK/astra_regime; S=$W/s_run; G=$W/gen; mkdir -p "$S/logs" "$G"
L="$PY scripts/steer/loop_s_learn.py"; A="--out $S --astra-dir $ASTRA --model $QWEN35 --blocks 15,19"
TESTSPLITS="validation_4 validation_3 final_1 final_2 final_3"
gen() {  # arm model split [extra args]: one job line
  echo "mkdir -p $G/$3/$1 && [ -e $G/$3/$1/summary.json ] || CUDA_VISIBLE_DEVICES=\$GPU $PY scripts/steer/astra_regime.py gen --astra-src $ASTRA --model $2 --split $3 --out $G/$3/$1 --gpu-mem \$MEM ${4:-} > $G/$3/$1/run.log 2>&1"
}
ARMS12=(); for n in 4 32; do for obj in clas ul contrast; do for b in 0 0.1; do ARMS12+=(${obj}_n${n}_b$b); done; done; done

# 1. reference edit, v, s
[ -e "$W/checkpoints/astra_top64_a6/config.json" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY scripts/steer/astra_regime.py astra-ckpt \
  --astra-src "$ASTRA" --base "$QWEN35" --dest "$W/checkpoints/astra_top64_a6" > "$S/logs/astra_ckpt.log" 2>&1
for n in 4 32; do [ -e "$S/v_n$n.pt" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $L direction $A --n $n > "$S/logs/direction_n$n.log" 2>&1; done
: > "$W/jobs_train"
for arm in "${ARMS12[@]}"; do obj=${arm%%_n*}; n=${arm#*_n}; n=${n%%_*}; b=${arm##*_b}
  [ -e "$S/s/$arm.pt" ] || echo "CUDA_VISIBLE_DEVICES=\$GPU $L train $A --n $n --obj $obj --beta $b > $S/logs/train_$arm.log 2>&1" >> "$W/jobs_train"; done
SLOTS=1 pool "$W/jobs_train"

# 2. unedited dev rollouts and stopping updates
gen control "$QWEN35" validation_1 > "$W/jobs_dev0"; pool "$W/jobs_dev0"
mkdir -p "$W/devloss"; : > "$W/jobs_devloss"
for arm in "${ARMS12[@]}"; do [ -e "$W/devloss/$arm.json" ] || echo "CUDA_VISIBLE_DEVICES=\$GPU $PY scripts/steer/astra_regime_devloss.py --s-run $S --gen-dir $G/validation_1/control --astra-src $ASTRA --model $QWEN35 --blocks 15,19 --arm $arm --json $W/devloss/$arm.json > $W/devloss/$arm.log 2>&1" >> "$W/jobs_devloss"; done
SLOTS=1 pool "$W/jobs_devloss"
STOP=$($PY scripts/steer/astra_stop_rule.py "$W"/devloss/*.json | awk '{print $1 "@" $2}')
echo "stopping updates: $STOP" | tr '\n' ' '; echo

# 3. candidates (an s kept at the last update, 24, is named without the _s suffix) and dev rollouts
CANDS=(); C=()
for n in 4 32; do for al in 1 2 3 4; do CANDS+=(sv_n${n}_a$al); C+=(--cond sv:$n:$al); done; done
for x in $STOP; do arm=${x%@*}; st=${x#*@}; obj=${arm%%_n*}; n=${arm#*_n}; n=${n%%_*}; b=${arm##*_b}
  name=$arm; [ "$st" = 24 ] || name=${arm}_s$st
  CANDS+=($name); [ -e "$S/checkpoints/$name/config.json" ] && continue
  CUDA_VISIBLE_DEVICES=$FIRST_GPU $L export $A --cond $obj:$n:$b@$st >> "$S/logs/export.log" 2>&1
  [ "$name" = "${arm}_s$st" ] || mv "$S/checkpoints/${arm}_s$st" "$S/checkpoints/$name"
done
need=(); for i in $(seq 0 7); do [ -e "$S/checkpoints/${CANDS[$i]}/config.json" ] || need+=("${C[$((2 * i))]}" "${C[$((2 * i + 1))]}"); done
[ ${#need[@]} = 0 ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $L export $A "${need[@]}" >> "$S/logs/export.log" 2>&1
{ gen astra_top64_a6 "$W/checkpoints/astra_top64_a6" validation_1
  for arm in "${CANDS[@]}"; do gen $arm "$S/checkpoints/$arm" validation_1; done; } > "$W/jobs_dev"
pool "$W/jobs_dev"

# 4. selection
$PY scripts/steer/astra_regime_summary.py dev --gen "$G" --out "$W/dev_selection.json" | tee "$W/dev_selection.txt"
CHOSEN=$($PY -c "import json; print(' '.join(json.load(open('$W/dev_selection.json'))['chosen_per_family_and_n'].values()))")

# 5. test rollouts and summaries
: > "$W/jobs_test"
for sp in $TESTSPLITS; do
  gen control "$QWEN35" $sp; gen control_nothink "$QWEN35" $sp "--thinking off"
  gen astra_top64_a6 "$W/checkpoints/astra_top64_a6" $sp
  for arm in $CHOSEN; do gen $arm "$S/checkpoints/$arm" $sp; done
done >> "$W/jobs_test"
pool "$W/jobs_test"
SV4=$($PY -c "import json; print(json.load(open('$W/dev_selection.json'))['chosen_per_family_and_n']['sv_n4'])")
SV32=$($PY -c "import json; print(json.load(open('$W/dev_selection.json'))['chosen_per_family_and_n']['sv_n32'])")
$PY scripts/steer/astra_regime_summary.py test --gen "$G" --arms "$(echo $CHOSEN astra_top64_a6 | tr ' ' ',')" \
  --refs "control,$SV4,$SV32" --out "$W/test_summary.json" | tee "$W/test_summary.txt"
$PY scripts/steer/astra_regime_summary.py test --gen "$G" --arms control_nothink --refs control --out "$W/test_summary_nothink.json"
echo "experiment 1 (Astra regime) done: $W"
