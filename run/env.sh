# Settings shared by every run script. Override any of them in the environment, e.g. GPUS=0,1 WORK=/scratch/ss bash run/...
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${WORK:-$ROOT/work}          # all downloads and outputs
GPUS=${GPUS:-0}                   # comma-separated GPU ids the scripts may use
SLOTS=${SLOTS:-2}                 # concurrent generation jobs per GPU
PY=${PY:-$ROOT/.venv/bin/python}  # created by run/setup.sh (uv sync)

# Pinned inputs (the revisions our runs used)
QWEN3_ID=Qwen/Qwen3-1.7B;       QWEN3_REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
QWEN35_ID=Qwen/Qwen3.5-0.8B;    QWEN35_REV=2fc06364715b967f1860aea9cf38778875588b17
LCB_GIT=https://github.com/LiveCodeBench/LiveCodeBench; LCB_COMMIT=28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
export LCB_DATASET_REVISION=0fe84c3912ea0c4d4a78037083943e8f0c4dd505   # livecodebench/code_generation_lite

QWEN3=$WORK/models/Qwen3-1.7B
QWEN35=$WORK/models/Qwen3.5-0.8B
LCB_REPO=$WORK/LiveCodeBench
DATA=$WORK/data                   # prepared MATH level-5 pools
ASTRA=$WORK/astra                 # Astra's materials (fetch_astra.py)
SPLITS=$ROOT/data/qwen3/splits    # dev and test prompt splits (shipped; rebuilt by run/data.sh)

export HF_HOME=${HF_HOME:-$WORK/hf} PYTHONPATH=$ROOT/src TOKENIZERS_PARALLELISM=false
export PATH=$ROOT/.venv/bin:$PATH   # vLLM builds Qwen3.5 kernels with the locked ninja
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}   # LiveCodeBench loading requires it
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_BATCH_INVARIANT=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SUPPRESS_SEM_UNLINK_ERRORS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
FIRST_GPU=${GPUS%%,*}
cd "$ROOT"
