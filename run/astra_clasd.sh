#!/bin/bash
# The clas arm with corrective NTP data, in Astra's regime (run after run/astra_exp1.sh; reuses its v, unedited dev
# rollouts and test rollouts). NTP is trained on each pair's productive continuation placed after k = 2 .. (the pair's
# number of copies) copies of its doubt sentence, i.e. where the model would keep repeating and the target breaks out
# (loop_s_learn.py --ntp-doubt; arm clasd). KL stays on the plain productive text. Everything else as for clas:
#  1. learned s for N = 4, 32 and beta 0, 0.1 (Adam lr 3e-3, 24 updates from s = 0).
#  2. Stopping update of each s (same rule and unedited dev rollouts as experiment 1).
#  3. Dev rollouts of each s at its stopping update (validation_1); per N, the highest dev accuracy (loop rate breaks ties).
#  4. Test rollouts of the chosen arms on the 1,152 test problems.
# Outputs under $WORK/astra_regime (s_run/s/clasd_*, gen/<split>/clasd_*, dev_selection_clasd.json).
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
W=$WORK/astra_regime; S=$W/s_run; G=$W/gen; mkdir -p "$S/logs" "$G"
L="$PY scripts/steer/loop_s_learn.py"; A="--out $S --astra-dir $ASTRA --model $QWEN35 --blocks 15,19"
TESTSPLITS="validation_4 validation_3 final_1 final_2 final_3"
gen() {  # arm model split: one job line (as in run/astra_exp1.sh)
  echo "mkdir -p $G/$3/$1 && [ -e $G/$3/$1/summary.json ] || CUDA_VISIBLE_DEVICES=\$GPU $PY scripts/steer/astra_regime.py gen --astra-src $ASTRA --model $2 --split $3 --sampler astra --out $G/$3/$1 --gpu-mem \$MEM ${ASTRA_LIMIT:+--limit $ASTRA_LIMIT} > $G/$3/$1/run.log 2>&1"
}
for n in 4 32; do [ -e "$S/v_n$n.pt" ] || { echo "missing $S/v_n$n.pt: run run/astra_exp1.sh first"; exit 1; }; done
[ -e "$G/validation_1/control/summary.json" ] || { echo "missing unedited dev rollouts: run run/astra_exp1.sh first"; exit 1; }
ARMS=(); for n in 4 32; do for b in 0 0.1; do ARMS+=(clasd_n${n}_b$b); done; done

# 1. learned s
: > "$W/jobs_train_clasd"
for arm in "${ARMS[@]}"; do n=${arm#*_n}; n=${n%%_*}; b=${arm##*_b}
  [ -e "$S/s/$arm.pt" ] || echo "CUDA_VISIBLE_DEVICES=\$GPU $L train $A --n $n --obj clas --ntp-doubt --beta $b > $S/logs/train_$arm.log 2>&1" >> "$W/jobs_train_clasd"; done
SLOTS=1 pool "$W/jobs_train_clasd"

# 2. stopping updates
mkdir -p "$W/devloss"; : > "$W/jobs_devloss_clasd"
for arm in "${ARMS[@]}"; do [ -e "$W/devloss/$arm.json" ] || echo "CUDA_VISIBLE_DEVICES=\$GPU $PY scripts/steer/astra_regime_devloss.py --s-run $S --gen-dir $G/validation_1/control --astra-src $ASTRA --model $QWEN35 --blocks 15,19 --arm $arm --json $W/devloss/$arm.json > $W/devloss/$arm.log 2>&1" >> "$W/jobs_devloss_clasd"; done
SLOTS=1 pool "$W/jobs_devloss_clasd"
STOP=$($PY scripts/steer/astra_stop_rule.py $(for a in "${ARMS[@]}"; do echo "$W/devloss/$a.json"; done) | awk '{print $1 "@" $2}')
echo "stopping updates: $STOP" | tr '\n' ' '; echo

# 3. candidates, dev rollouts, selection
CANDS=()
for x in $STOP; do arm=${x%@*}; st=${x#*@}; n=${arm#*_n}; n=${n%%_*}; b=${arm##*_b}
  name=$arm; [ "$st" = 24 ] || name=${arm}_s$st
  CANDS+=($name); [ -e "$S/checkpoints/$name/config.json" ] && continue
  CUDA_VISIBLE_DEVICES=$FIRST_GPU $L export $A --cond clasd:$n:$b@$st >> "$S/logs/export_clasd.log" 2>&1
  [ "$name" = "${arm}_s$st" ] || mv "$S/checkpoints/${arm}_s$st" "$S/checkpoints/$name"
done
for arm in "${CANDS[@]}"; do gen $arm "$S/checkpoints/$arm" validation_1; done > "$W/jobs_dev_clasd"
pool "$W/jobs_dev_clasd"
DEVX=$(ls "$G/validation_1" | { grep -v -e '^control$' -e '^clasd_' || true; } | tr '\n' ',')
$PY scripts/steer/astra_regime_summary.py dev --gen "$G" --exclude "$DEVX" --out "$W/dev_selection_clasd.json" | tee "$W/dev_selection_clasd.txt"
CHOSEN=$($PY -c "import json; c = json.load(open('$W/dev_selection_clasd.json'))['chosen_per_family_and_n']; print(c['clasd_n4'], c['clasd_n32'])")

# 4. test rollouts
for sp in $TESTSPLITS; do for arm in $CHOSEN; do gen $arm "$S/checkpoints/$arm" $sp; done; done > "$W/jobs_test_clasd"
pool "$W/jobs_test_clasd"
# Summarize selected methods using the same test-summary path as the other methods.
$PY scripts/steer/astra_regime_summary.py test --gen "$G" --arms "$(echo "$CHOSEN" | tr ' ' ',')" --refs control,sv_n4_a4,sv_n32_a4 --out "$W/test_summary_clasd.json"
echo "clasd (Astra regime) done: $CHOSEN"
