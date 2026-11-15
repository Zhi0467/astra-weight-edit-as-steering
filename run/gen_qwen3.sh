#!/bin/bash
# One Qwen3-1.7B generation run (thinking on, greedy, 1 draw, 40,960-token context), graded.
#   gen_qwen3.sh MODEL_DIR ARM DATASET SPLIT [SRC]
# MODEL_DIR  the unedited model ($QWEN3) or an edited checkpoint; ARM names the run.
# DATASET    math_level5 | livecodebench
# SPLIT      dev  (64 looping + 64 non-looping train_source prompts per dataset)
#            test (500 prompts per dataset; run once per SRC = train_source | heldout_source, since the LiveCodeBench
#                  releases differ: v1-v5 for train_source, v6 for heldout_source)
# Output: $RUNS/<split>[/<src>]/<dataset>/<arm>/ with the rollout bundle (.json + .jsonl.gz) and DONE; LiveCodeBench
# runs are then graded on CPU into the same folder (GRADED). Env: GPU (default $FIRST_GPU), QWEN3_MEM (share of GPU memory,
# default pool share, or 0.85 when run directly), THINKING (on | off; off = the no-thinking reference row), RUNS (default $WORK/qwen3/runs), MAX_SAMPLES
# (only the first N prompts; for smoke tests).
set -uo pipefail
source "$(dirname "$0")/env.sh"
model=${1:?model dir}; arm=${2:?arm}; ds=${3:?dataset}; split=${4:?split}
RUNS=${RUNS:-$WORK/qwen3/runs}; THINKING=${THINKING:-on}
case $split in
  dev)  src=train_source; rel=v1_v5; dir=$RUNS/dev/$ds/$arm
        msub=$SPLITS/dev/math_level5_dev_subset.jsonl; lex=$SPLITS/dev/livecodebench_dev_exclude.jsonl;;
  test) src=${5:?SRC}; dir=$RUNS/test/$src/$ds/$arm
        case $src in train_source) rel=v1_v5;; heldout_source) rel=v6;; *) echo "bad SRC $src"; exit 2;; esac
        msub=$SPLITS/test12/math_level5_${src}_test_subset.jsonl; lex=$SPLITS/test12/livecodebench_${src}_test_exclude.jsonl;;
  *) echo "bad SPLIT $split"; exit 2;;
esac
[ -e "$model/config.json" ] || { echo "missing $model"; exit 2; }
out=$dir/${ds}__${src}__thinking_${THINKING}__$arm.json
mkdir -p "$dir"
if [ ! -e "$dir/DONE" ]; then
  if [ "$ds" = math_level5 ]; then
    task=(--task-kind math_freeform --dataset "$msub" --question-field problem --answer-field solution
      --prompt-format chat_template --metadata-fields level,type,_contract_split,_contract_dataset_key,_contract_source_role)
  else
    task=(--task-kind livecodebench_codegen --dataset livecodebench/code_generation_lite --livecodebench-repo "$LCB_REPO"
      --release-version $rel --lm-style-override HFChatTemplate --defer-cpu-finalize --exclude-prompt-jsonl "$lex")
  fi
  echo "$(date -Is) start $ds $arm gpu ${GPU:-$FIRST_GPU}" >> "$dir/run.log"
  CUDA_VISIBLE_DEVICES=${GPU:-$FIRST_GPU} VLLM_GPU_MEMORY_UTILIZATION=${QWEN3_MEM:-${MEM:-0.85}} $PY scripts/rollout/collect_model_stats.py \
    "${task[@]}" --split $src --model-id "$model" --temperature 0.0 --top-p 0.95 --top-k 20 --presence-penalty 0 \
    --num-generations 1 --seed 0 --max-tokens 81920 --max-model-len 40960 --max-num-seqs 64 --no-enforce-eager \
    --enable-prefix-caching --attention-backend FLASH_ATTN --loop-n 30 --loop-k 20 --thinking-mode "$THINKING" \
    --tp 1 --dp 1 ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES} --resume --out "$out" >> "$dir/run.log" 2>&1 || { echo "generation failed: $dir"; exit 1; }
  touch "$dir/DONE"
fi
if [ "$ds" = livecodebench ] && [ ! -e "$dir/GRADED" ]; then
  CUDA_VISIBLE_DEVICES='' LCB_NUM_PROCESS_EVALUATE=${LCB_PROCS:-16} $PY scripts/rollout/collect_model_stats.py \
    --task-kind livecodebench_codegen --out "$out" --livecodebench-repo "$LCB_REPO" --release-version $rel \
    --finalize-only > "$dir/grade.log" 2>&1 || { echo "grading failed: $dir"; exit 1; }
  touch "$dir/GRADED"
fi
