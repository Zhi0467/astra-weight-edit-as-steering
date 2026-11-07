#!/bin/bash
# Smoke test on one GPU (about 20 minutes): every step of the pipelines on a few prompts. Output: $WORK/smoke.
set -euo pipefail
source "$(dirname "$0")/env.sh"
S=$WORK/smoke; mkdir -p "$S"; export CUDA_VISIBLE_DEVICES=$FIRST_GPU
L="$PY scripts/steer/loop_s_learn.py"
echo "== Qwen3-1.7B: v (N = 4), learned s (ul, beta 0.1), checkpoints"
$L direction --out "$S/qwen3" --astra-dir "$ASTRA" --model "$QWEN3" --n 4 > "$S/direction.log" 2>&1
$L train --out "$S/qwen3" --astra-dir "$ASTRA" --model "$QWEN3" --n 4 --obj ul --beta 0.1 > "$S/train.log" 2>&1
$L export --out "$S/qwen3" --astra-dir "$ASTRA" --model "$QWEN3" --cond sv:4:2 --cond ul:4:0.1@14 > "$S/export.log" 2>&1
echo "== Qwen3-1.7B: generation and grading on 4 dev prompts per dataset"
for ds in math_level5 livecodebench; do
  RUNS=$S/runs MAX_SAMPLES=4 bash run/gen_qwen3.sh "$S/qwen3/checkpoints/sv_n4_a2" sv_n4_a2 $ds dev
done
echo "== Qwen3.5-0.8B: Astra's edit rebuilt bit-exactly, generation on 4 problems"
$PY scripts/steer/astra_regime.py astra-ckpt --astra-src "$ASTRA" --base "$QWEN35" --dest "$S/astra_top64_a6" > "$S/astra_ckpt.log" 2>&1
$PY scripts/steer/astra_regime.py gen --astra-src "$ASTRA" --model "$S/astra_top64_a6" --split validation_1 --limit 4 \
  --out "$S/astra_gen" --gpu-mem 0.5 > "$S/astra_gen.log" 2>&1
python3 - "$S" <<'PY'
import gzip, json, sys
from pathlib import Path
S = Path(sys.argv[1])
for ds in ("math_level5", "livecodebench"):
    f = next((S / "runs/dev" / ds / "sv_n4_a2").glob("*.jsonl.gz"))
    rows = [json.loads(l) for l in gzip.open(f, "rt")]
    ok = sum(bool((r["rollouts"][0].get("grading") or {}).get("correct") or (r["rollouts"][0].get("grading") or {}).get("passed")) for r in rows)
    print(f"{ds}: {len(rows)} prompts, {ok} correct")
s = json.loads((S / "astra_gen/summary.json").read_text())
print("astra edit, 4 problems:", {k: s[k] for k in s if k in ("n", "accuracy", "loop")})
PY
echo "smoke test passed"
