#!/bin/bash
# Current blog figures. Examples:
#   bash run/figures.sh --released
#   bash run/figures.sh --released 2a 3b
# --run prepares single-run summaries from WORK; --released uses bundled results.
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE=${1:-}
case "$MODE" in
  --released|--run) shift ;;
  *) echo "Usage: bash run/figures.sh --released|--run [figure numbers]" >&2; exit 2 ;;
esac
if [ "$MODE" = --run ]; then FIGS=${*:-2a 2b 2c 4 3a 3b}; else FIGS=${*:-1 2a 2b 2c 4 5 3a 3b}; fi
OUT=${OUT:-$WORK/figures/${MODE#--}}
if [ "$MODE" = --run ]; then
  for f in $FIGS; do
    case "$f" in
      1|5) echo "Figure $f is a publication SVG, not a fresh-run plot; select 2a 2b 2c 4 3a 5b." >&2; exit 2 ;;
      2a|2b|2c|4|3a|3b) ;;
      *) echo "Unknown figure: $f" >&2; exit 2 ;;
    esac
  done
  case " $FIGS " in
    *" 2a "*|*" 2b "*|*" 2c "*) "$PY" scripts/report/prepare_run_figures.py --work "$WORK" --setting astra ;;
  esac
  case " $FIGS " in
    *" 3a "*|*" 3b "*) "$PY" scripts/report/prepare_run_figures.py --work "$WORK" --setting qwen3 ;;
  esac
  A=$WORK/astra_regime/test_summary_with_clasd.json
  Q=$WORK/qwen3/exp1/exp1_test_summary_with_clasd.json
  N=$WORK/astra_regime/test_summary_nothink.json
  NATURAL_SUMMARY=$WORK/qwen3/natural/test_summary.json
else
  A=${ASTRA_SUMMARY:-$ROOT/results/astra_regime/test_summary_with_clasd.json}
  Q=${QWEN3_SUMMARY:-$ROOT/results/qwen3/exp1_test_summary_with_clasd.json}
  N=${ASTRA_NOTHINK:-$ROOT/results/astra_regime/test_summary_nothink.json}
fi
mkdir -p "$OUT"
echo "Figure source: ${MODE#--}; output: $OUT"
GRIDS=
for f in $FIGS; do
  case "$f" in
    2a|2b|3a|3b) GRIDS+="$f," ;;
    1) cp "$ROOT/blog/figures/mlp_equivalence.svg" "$OUT/" ;;
    2c)
      "$PY" scripts/report/build_blog_pooled_summary.py --summary "$A" --out "$OUT"
      "$PY" scripts/report/build_blog_pooled_summary.py --summary "$A" --out "$OUT" --include-positive ;;
    4) "$PY" scripts/report/build_natural_vs_short.py --summary "${NATURAL_SUMMARY:-$ROOT/results/qwen3/natural_vs_short/test_summary.json}" --out "$OUT/qwen3_natural_vs_short.svg" ;;
    5) cp "$ROOT/blog/figures/write_vs_residual.svg" "$OUT/" ;;
    *) echo "Unknown figure: $f. Use 1, 2a, 2b, 2c, 4, 5, 3a or 5b." >&2; exit 1 ;;
  esac
done
if [ -n "$GRIDS" ]; then
  "$PY" scripts/report/build_blog_corrective_grids.py --only "${GRIDS%,}" --astra "$A" --qwen3 "$Q" --astra-nothink "$N" --out "$OUT"
fi
echo "Figures in $OUT"
