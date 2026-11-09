#!/bin/bash
# Controls in Astra's regime (needs run/astra_exp1.sh first): on the 1,152 test problems,
#  - the same edits at the residual site: s = 4v and the learned unlikelihood s (N = 32, beta 0.1), applied at run
#    time to the block output of layers 15 and 19 instead of the MLP output;
#  - random-direction edits: a full-column edit norm-matched to s = 4v (N = 32), and an edit on Astra's 64 columns
#    norm-matched to Astra's edit.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
SLOTS=${ASTRA_SLOTS:-3}
W=$WORK/astra_regime; S=$W/s_run; G=$W/gen; K=$W/controls; mkdir -p "$K"
AR="$PY scripts/steer/astra_regime.py"
cd "$ROOT"
[ -e "$K/resid_sv_n32_a4.pt" ] || $AR resid-spec --astra-src "$ASTRA" --v "$S/v_n32.pt" --alpha 4 --dest "$K/resid_sv_n32_a4.pt"
[ -e "$K/resid_ul_n32_b0.1.pt" ] || $AR resid-spec --astra-src "$ASTRA" --v "$S/v_n32.pt" --s-file "$S/s/ul_n32_b0.1.pt" --dest "$K/resid_ul_n32_b0.1.pt"
[ -e "$K/random_full_matched_sv_n32_a4/config.json" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $AR random-ckpt --astra-src "$ASTRA" --base "$QWEN35" \
  --v "$S/v_n32.pt" --kind full --match "$S/checkpoints/sv_n32_a4" --dest "$K/random_full_matched_sv_n32_a4"
[ -e "$K/random_top64_matched_astra/config.json" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $AR random-ckpt --astra-src "$ASTRA" --base "$QWEN35" \
  --v "$S/v_n32.pt" --kind top64 --match "$W/checkpoints/astra_top64_a6" --dest "$K/random_top64_matched_astra"
job() { echo "mkdir -p $G/$3/$1 && [ -e $G/$3/$1/summary.json ] || CUDA_VISIBLE_DEVICES=\$GPU $AR gen --astra-src $ASTRA --model $2 --split $3 --out $G/$3/$1 --gpu-mem \$MEM ${ASTRA_LIMIT:+--limit $ASTRA_LIMIT} ${4:-} > $G/$3/$1/run.log 2>&1"; }
for sp in validation_4 validation_3 final_1 final_2 final_3; do
  job resid_sv_n32_a4 "$QWEN35" $sp "--resid $K/resid_sv_n32_a4.pt"
  job resid_ul_n32_b0.1 "$QWEN35" $sp "--resid $K/resid_ul_n32_b0.1.pt"
  job random_full_matched_sv_n32_a4 "$K/random_full_matched_sv_n32_a4" $sp
  job random_top64_matched_astra "$K/random_top64_matched_astra" $sp
done > "$K/jobs"
pool "$K/jobs"
SUM="$PY scripts/steer/astra_regime_summary.py test --gen $G"
$SUM --arms resid_sv_n32_a4,resid_ul_n32_b0.1,sv_n32_a4,ul_n32_b0.1 --refs control,sv_n32_a4,ul_n32_b0.1 --out "$W/test_summary_resid.json"
$SUM --arms random_full_matched_sv_n32_a4,random_top64_matched_astra,sv_n32_a4,astra_top64_a6 --refs control,sv_n32_a4,astra_top64_a6 --out "$W/test_summary_random.json"
echo "controls done: $W"
