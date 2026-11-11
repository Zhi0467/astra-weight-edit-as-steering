#!/bin/bash
# Blog figures from the test summaries, all of them or the ones named.
#   bash run/figures.sh                 every figure, from your rerun's summaries under $WORK
#   bash run/figures.sh 1a 1b           only Figures 1a and 1b (needs only run/astra_exp1.sh)
#   FROM=results bash run/figures.sh    from our shipped summaries (gives exactly blog/figures)
# Figures: 1a, 1b  Astra regime, GSM8K / MATH            (run/astra_exp1.sh; random-direction rows if run/astra_controls.sh ran)
#          2a, 2b  Qwen3-1.7B, MATH / LiveCodeBench       (run/qwen3_exp1.sh)
#          site    write-site vs residual-site figure     (run/astra_controls.sh)
#          directions    short synthetic pairs vs natural traces as steering data (run/qwen3_natural.sh)
# Output: $WORK/figures
set -euo pipefail
source "$(dirname "$0")/env.sh"
FIGS=${*:-1a 1b 2a 2b site directions}
if [ "${FROM:-work}" = results ]; then
  QS=$ROOT/results/qwen3/exp1_test_summary.json; A=$ROOT/results/astra_regime; SPECS=$A/resid_specs
  NS=$ROOT/results/qwen3/natural_vs_short/test_summary.json
else
  QS=$WORK/qwen3/exp1/exp1_test_summary.json; A=$WORK/astra_regime; SPECS=$A/controls
  NS=$WORK/qwen3/natural/test_summary.json
fi
OUT=$WORK/figures; mkdir -p "$OUT"
cd scripts/report   # the builders import steering_delta_grid from here
GRIDS=
for f in $FIGS; do
  case $f in
    1a|1b|2a|2b) GRIDS+="$f," ;;
    site|directions) ;;
    *) echo "unknown figure: $f" >&2; exit 1 ;;
  esac
done
if [ -n "$GRIDS" ]; then
  opt=()
  [ -e "$A/test_summary_nothink.json" ] && opt+=(--astra-nothink "$A/test_summary_nothink.json")
  [ -e "$A/test_summary_random.json" ] && opt+=(--astra-random "$A/test_summary_random.json")
  $PY build_exp1_learned_sensing_heatmaps.py --only "${GRIDS%,}" --qwen3 "$QS" --astra "$A/test_summary.json" "${opt[@]}" --out "$OUT"
fi
if [[ " $FIGS " == *" site "* ]]; then
  $PY build_blog_fig2_site.py --summary "$A/test_summary_resid.json" --specs "$SPECS" --out "$OUT/fig2_write_vs_residual.html"
fi
if [[ " $FIGS " == *" directions "* ]]; then
  $PY build_natural_vs_short.py --summary "$NS" --out "$OUT/qwen3_natural_vs_short.svg"
fi
echo "figures in $OUT"
