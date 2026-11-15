# Context-dependent activation steering

Code, data and results for the [blog post](https://zhi0467.github.io/posts/2026/post6/).
See [supplementary methods](blog/supplementary-methods.md) for the experimental
protocol, data construction and additional comparisons.

## Setup

Run commands from the repository root. Training and generation require Linux
and NVIDIA GPUs (the experiments used H100 80GB GPUs). Install
[uv](https://docs.astral.sh/uv/) and git, then:

```bash
bash run/setup.sh
GPUS=0 bash run/smoke.sh
```

Setup installs the locked Python 3.12 environment and downloads the pinned
models and datasets. Settings and input revisions live in
[`run/env.sh`](run/env.sh), [`run/setup.sh`](run/setup.sh) and the data scripts;
dependencies are pinned in `uv.lock`.

Set `GPUS=0,1` to choose devices, `SLOTS=1` for one generation process per GPU,
and `WORK=/path/to/work` to relocate downloads and outputs (default: `work/`).

## Reproduce the experiments

Run the scripts in each row from left to right after setup. Shared prerequisites
only need to run once.

| Figure | Experiment | Commands, in order |
|---|---|---|
| **2a–c** | Qwen3.5-0.8B: GSM8K, MATH and pooled results, including Corrective NTP | `bash run/astra_exp1.sh` → `bash run/astra_clasd.sh` |
| **3a–b** | Qwen3-1.7B: MATH level 5 and LiveCodeBench, including Corrective NTP | `STOP_TRACES=ours bash run/qwen3_exp1.sh` → `STOP_TRACES=ours bash run/qwen3_clasd.sh` |
| **4** | Qwen3-1.7B: short synthetic pairs versus natural traces | `STOP_TRACES=ours bash run/qwen3_exp1.sh` → `bash run/qwen3_natural.sh` |
| **5** | MLP write-site versus residual-stream steering, including recalibrated strength | `bash run/astra_exp1.sh` → `bash run/astra_controls.sh` → `bash run/astra_resid_alpha.sh` |

`STOP_TRACES=ours` uses the shipped Qwen3 stopping-rule traces; omit it to use
your own dev rollouts. By default, new results are written to
`work/astra_regime/` and `work/qwen3/` inside the repository. Setting `WORK`
changes that output root. Released results are in `results/astra_regime/`
and `results/qwen3/`.

After the experiment commands, plot your own outputs with:

```bash
bash run/figures.sh --run 2a 2b 2c       # Astra run, including Corrective NTP
bash run/figures.sh --run 4             # short versus natural traces
bash run/figures.sh --run 3a 3b         # Qwen3 run, including Corrective NTP
```

This prepares summaries from your selected arms and test rollouts under `work/`,
then writes figures to `work/figures/run/`. It does not read the released
scores. These are single-run comparisons; released comparisons average two
reference runs. Figures 1 and 5 remain publication SVG assets.

## Use released results (no GPU runs)

After installing dependencies with `uv sync --frozen`:

```bash
bash run/figures.sh --released          # bundled results → work/figures/released
bash run/figures.sh --released 2a 3b                # selected figures
.venv/bin/python scripts/report/clasd_two_run_summary.py --base results/astra_regime/test_summary.json --out /tmp/astra-summary.json --check
.venv/bin/python scripts/report/qwen3_two_run_summary.py --out /tmp/qwen3-summary.json --check
.venv/bin/python -m unittest discover -s tests
```

Figures 1 and 5 use the publication SVG assets. Other figures use the released
summaries. Set `OUT` to change the output folder; `ASTRA_SUMMARY`,
`QWEN3_SUMMARY`, `ASTRA_NOTHINK` and `NATURAL_SUMMARY` select replacement inputs.

The combined summaries average reference methods over two runs per prompt;
Corrective NTP contributes one run. Bundled scored records reproduce the paired
bootstrap summaries without model generation. They contain outcomes and token
counts, not model checkpoints or completion text.

## Credits and license

Astra's [materials](https://github.com/RikiyaT/lm_understand-logs) supply the
original four pairs and baseline (data: CC BY 4.0; code: MIT). The other 28
synthetic pairs were written by Claude for this project. Repository code: [MIT](LICENSE).
