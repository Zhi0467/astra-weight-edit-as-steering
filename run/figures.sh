#!/bin/bash
# Blog figures from the test summaries.
#   bash run/figures.sh                 from your rerun's summaries under $WORK
#   FROM=results bash run/figures.sh    from our shipped summaries (gives exactly blog/figures)
# Output: $WORK/figures
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ "${FROM:-work}" = results ]; then
  QS=$ROOT/results/qwen3/exp1_test_summary.json; A=$ROOT/results/astra_regime; SPECS=$A/resid_specs
else
  QS=$WORK/qwen3/exp1/exp1_test_summary.json; A=$WORK/astra_regime; SPECS=$A/controls
fi
OUT=$WORK/figures; mkdir -p "$OUT"
cd scripts/report   # the builders import steering_delta_grid from here
$PY build_exp1_learned_sensing_heatmaps.py --qwen3 "$QS" --astra "$A/test_summary.json" \
  --astra-nothink "$A/test_summary_nothink.json" --astra-random "$A/test_summary_random.json" --out "$OUT"
$PY build_blog_fig2_site.py --summary "$A/test_summary_resid.json" --specs "$SPECS" --out "$OUT/fig2_write_vs_residual.html"
echo "figures in $OUT"
