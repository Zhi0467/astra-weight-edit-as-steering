#!/bin/bash
# Optional: regenerate the reference runs behind data/qwen3/loop_labels.jsonl (4 greedy draws per prompt of every MATH
# level-5 and LiveCodeBench prompt, both source splits; about 19,000 completions of up to 40,960 tokens), then the labels.
# Greedy decoding in vLLM depends on batching and hardware, so new labels can differ from ours for some prompts.
#   bash run/qwen3_reference_draws.sh && LABELS=$WORK/qwen3/reference/loop_labels.jsonl bash run/data.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/lib.sh"
R=$WORK/qwen3/reference; mkdir -p "$R"; : > "$R/jobs"
for src in train_source heldout_source; do
  case $src in train_source) rel=v1_v5;; heldout_source) rel=v6;; esac
  for ds in math_level5 livecodebench; do
    d=$R/$ds/$src; mkdir -p "$d"
    if [ $ds = math_level5 ]; then
      task="--task-kind math_freeform --dataset $DATA/math_level5/$src.jsonl --question-field problem --answer-field solution --prompt-format chat_template --metadata-fields level,type,_contract_split,_contract_dataset_key,_contract_source_role"
    else
      task="--task-kind livecodebench_codegen --dataset livecodebench/code_generation_lite --livecodebench-repo $LCB_REPO --release-version $rel --lm-style-override HFChatTemplate"
    fi
    echo "[ -e $d/DONE ] || { CUDA_VISIBLE_DEVICES=\$GPU VLLM_GPU_MEMORY_UTILIZATION=0.85 $PY scripts/rollout/collect_model_stats.py $task --split $src --model-id $QWEN3 --temperature 0.0 --top-p 0.95 --top-k 20 --num-generations 4 --seed 0 --max-tokens 81920 --max-model-len 40960 --max-num-seqs 64 --no-enforce-eager --enable-prefix-caching --attention-backend FLASH_ATTN --loop-n 30 --loop-k 20 --thinking-mode on --tp 1 --dp 1 --resume --out $d/${ds}__${src}__thinking_on.json > $d/run.log 2>&1 && touch $d/DONE; }" >> "$R/jobs"
  done
done
SLOTS=1 pool "$R/jobs"
$PY scripts/data/loop_labels.py --bundles "$R" --out "$R/loop_labels.jsonl"
