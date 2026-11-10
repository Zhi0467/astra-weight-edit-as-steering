#!/bin/bash
# Controls in Astra's regime (needs run/astra_exp1.sh first): on the 1,152 test problems,
#  - the same edits at the residual site: the dev-chosen s = alpha v and learned unlikelihood s at N = 32 (ours:
#    s = 4v and ul beta 0.1), applied at run time to the block output of layers 15 and 19 instead of the MLP output;
#  - random-direction edits: a full-column edit norm-matched to that s = alpha v, and an edit on Astra's 64 columns
#    norm-matched to Astra's edit.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
SLOTS=${ASTRA_SLOTS:-3}
W=$WORK/astra_regime; S=$W/s_run; G=$W/gen; K=$W/controls; mkdir -p "$K"
AR="$PY scripts/steer/astra_regime.py"
cd "$ROOT"
chosen() { $PY -c "import json; print(json.load(open('$W/dev_selection.json'))['chosen_per_family_and_n']['$1'])"; }
SV=$(chosen sv_n32); UL=$(chosen ul_n32)   # e.g. sv_n32_a4, ul_n32_b0.1
[[ $UL != *_s* ]] || { echo "$UL uses a non-default stopping update; resid-spec takes the kept update only" >&2; exit 1; }
[ -e "$K/resid_$SV.pt" ] || $AR resid-spec --astra-src "$ASTRA" --v "$S/v_n32.pt" --alpha ${SV##*_a} --dest "$K/resid_$SV.pt"
[ -e "$K/resid_$UL.pt" ] || $AR resid-spec --astra-src "$ASTRA" --v "$S/v_n32.pt" --s-file "$S/s/${UL%_s*}.pt" --dest "$K/resid_$UL.pt"
[ -e "$K/random_full_matched_$SV/config.json" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $AR random-ckpt --astra-src "$ASTRA" --base "$QWEN35" \
  --v "$S/v_n32.pt" --kind full --match "$S/checkpoints/$SV" --dest "$K/random_full_matched_$SV"
[ -e "$K/random_top64_matched_astra/config.json" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU $AR random-ckpt --astra-src "$ASTRA" --base "$QWEN35" \
  --v "$S/v_n32.pt" --kind top64 --match "$W/checkpoints/astra_top64_a6" --dest "$K/random_top64_matched_astra"
job() { echo "mkdir -p $G/$3/$1 && [ -e $G/$3/$1/summary.json ] || CUDA_VISIBLE_DEVICES=\$GPU $AR gen --astra-src $ASTRA --model $2 --split $3 --out $G/$3/$1 --gpu-mem \$MEM ${ASTRA_LIMIT:+--limit $ASTRA_LIMIT} ${4:-} > $G/$3/$1/run.log 2>&1"; }
for sp in validation_4 validation_3 final_1 final_2 final_3; do
  job resid_$SV "$QWEN35" $sp "--resid $K/resid_$SV.pt"
  job resid_$UL "$QWEN35" $sp "--resid $K/resid_$UL.pt"
  job random_full_matched_$SV "$K/random_full_matched_$SV" $sp
  job random_top64_matched_astra "$K/random_top64_matched_astra" $sp
done > "$K/jobs"
pool "$K/jobs"
SUM="$PY scripts/steer/astra_regime_summary.py test --gen $G"
$SUM --arms resid_$SV,resid_$UL,$SV,$UL --refs control,$SV,$UL --out "$W/test_summary_resid.json"
$SUM --arms random_full_matched_$SV,random_top64_matched_astra,$SV,astra_top64_a6 --refs control,$SV,astra_top64_a6 --out "$W/test_summary_random.json"
echo "controls done: $W"
