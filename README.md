# Sensed steering: code and data for "Astra's weight-space edit of Qwen is a special case of sensed steering"

This repository holds the code, data construction and small datasets behind the blog post in [`blog/main.md`](blog/main.md).
The post shows that Astra's loop-reducing weight edit of Qwen is activation steering written into the weights:
a rank-one edit `x ← x − v (sᵀx)` on an MLP's output, with a steering vector `v` and a sensing vector `s`.
We then ask whether learning `s` or building `v` differently does better.

Everything needed to rerun the experiments is here: models and data at pinned versions, the exact generation and
grading code, the dev/test splits, the selection rules, and the summary and figure scripts. Rollouts are not included.
[`results/`](results/) holds our summaries for comparison. We ran the pipeline twice; the post's numbers average each problem's outcome over the two runs (each run's dev-chosen learned sensors; fixed arms chosen on both runs' dev results) and then apply the usual paired bootstraps. `results/*/runs/1` and `runs/2` hold the two runs' own summaries.

## What reproduces what

| Post | Script | Our numbers |
|---|---|---|
| §2–3, Figures 1a/1b: Astra regime (Qwen3.5-0.8B, Astra's sampler, 1,152 problems); §7 sampler diagnostic; endnote cosine | `run/astra_exp1.sh` | `results/astra_regime/` |
| §4 write-site vs residual-site table; the random-direction rows of Figures 1a/1b | `run/astra_controls.sh` | `results/astra_regime/test_summary_{resid,random}.json` |
| §4 recalibrated residual-site s = αv (α chosen on dev from 0.2, 0.5, 1, 1.5) | `run/astra_resid_alpha.sh` | `results/astra_regime/{resid_alpha_dev,test_summary_resid_alpha}.json` |
| §3–5 Qwen3-1.7B stopping updates; §5, Figures 2a/2b: Qwen3-1.7B, greedy, 1,000 prompts | `run/qwen3_exp1.sh` | `results/qwen3/exp1_*` |
| Figures, all or one at a time (`run/figures.sh 1a 2b site`) | `run/figures.sh` | `blog/figures/` |
| Every number quoted in the post | `scripts/report/blog_numbers.py` | prints them from `results/` or from a rerun |

```bash
python3 scripts/report/blog_numbers.py --qwen3 results/qwen3 --astra results/astra_regime          # ours
python3 scripts/report/blog_numbers.py --qwen3 work/qwen3/exp1 --astra work/astra_regime          # your rerun
```

The two Qwen3-1.7B comparisons in §4 ("Where the sensor reads matters" and "Short synthetic pairs were especially
effective") come from an earlier round. That round ran on an older software stack and built its natural-trace
directions from earlier saved rollouts, so this repository does not reproduce it.
`bash run/figures.sh directions` redraws the natural-vs-synthetic plot from its reported point estimates.


## Setup

Requirements: Linux, NVIDIA GPUs (we used H100 80GB), [uv](https://docs.astral.sh/uv/), git, and about 60 GB of
disk for models, the environment and outputs.

```bash
bash run/setup.sh      # uv environment, pinned models, LiveCodeBench at a pinned commit, MATH pools, Astra's files
GPUS=0 bash run/smoke.sh                  # about 20 minutes on one GPU
FROM=results bash run/figures.sh          # our figures from our summaries, into work/figures (or name some: 1a 1b 2a 2b site)
```

`run/env.sh` holds every setting: `WORK` (all outputs, default `./work`), `GPUS` (e.g. `0,1,2,3`), `SLOTS`
(concurrent jobs per GPU), and the pinned revisions:

| Input | Version |
|---|---|
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |
| MATH (SuperSecureHuman/competition_math_hf_dataset) | `fbc86cb47a39a9465b9235622014b33f8ca9bb7f` |
| livecodebench/code_generation_lite | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` |
| LiveCodeBench runner (github.com/LiveCodeBench/LiveCodeBench) | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` |
| Astra's materials (github.com/RikiyaT/lm_understand-logs, `exp8_math`) | `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` |

## Running the experiments

Each script runs end to end and skips finished steps, so it can be restarted.

```bash
GPUS=0,1,2,3,4,5,6,7 bash run/astra_exp1.sh     # then run/astra_controls.sh, run/astra_resid_alpha.sh
GPUS=0,1,2,3,4,5,6,7 bash run/qwen3_exp1.sh
bash run/figures.sh
```

Every experiment uses four disjoint roles:

- **Pairs** build `v` and train `s`: `scripts/steer/loop_short_pairs.json`. N = 4 uses Astra's four pairs; N = 32
  adds 28 written in the same format.
- **Dev** prompts pick every hyperparameter: α, β, the training update and the pair count.
- **Test** prompts are run once, on the chosen arms.
- **Teacher-forced scores** pick the training update: the lowest loop score among updates that keep the unedited
  model's rate of ranking `</think>` first where it closes its thinking.

| Script | Steps | Generation runs |
|---|---|---|
| `astra_exp1.sh` | Astra's edit; `v` (N = 4, 32); 12 learned `s` (clas / ul / contrast × β 0, 0.1 × N); unedited dev rollouts; stopping update per `s`; dev rollouts of 21 candidates; selection; test rollouts on 5 problem sets | 22 dev + 55 test |
| `astra_controls.sh` | residual-site versions of s = 4v and learned ul; norm-matched random-direction edits | 20 test |
| `astra_resid_alpha.sh` | s = αv at the residual site, α ∈ {0.2, 0.5, 1, 1.5} on dev; the chosen α on test | 4 dev + 5 test |
| `qwen3_exp1.sh` | as above for Qwen3-1.7B at blocks 15–18 (MATH level 5 and LiveCodeBench, dev 128 + 128 prompts) | 42 dev + 40 test |

A Qwen3-1.7B test run is one arm on one dataset and one source split, 250–300 prompts at up to 40,960 tokens.
LiveCodeBench runs are graded on CPU right after generation.

## Data

| Path | What it is |
|---|---|
| `scripts/steer/loop_short_pairs.json` | 32 short productive/repetitive text pairs. The first 4 are Astra's (CC BY 4.0); the other 28 were written by Claude for this project. |
| `data/qwen3/loop_labels.jsonl` | Per prompt, whether each of 4 greedy Qwen3-1.7B draws loops (4,683 prompts). The splits are stratified by it. |
| `data/qwen3/splits/dev/` | 64 looping + 64 non-looping train-split prompts per dataset. |
| `data/qwen3/splits/test12/` | 500 prompts per dataset: every looping prompt outside dev, then a seeded sample of non-looping prompts. |
| `data/qwen3/stop_traces/` | The 48 unedited-model dev traces (12 finished, 12 looping per dataset) our stopping rule scored. `STOP_TRACES=ours bash run/qwen3_exp1.sh` uses them; by default the rule draws from your own dev rollouts, with the same seed. |
| `data/qwen3/math_level5_train_validation_ids.json`, `SHA256SUMS` | Lets `prepare_datasets.py` write our MATH files byte-for-byte. |

A **loop** is a 30-token sequence that occurs at least 20 times in a completion.

`loop_labels.jsonl` comes from our reference runs. `run/qwen3_reference_draws.sh` regenerates such runs (about
19,000 long completions), and `scripts/data/loop_labels.py` turns them into labels. New labels can differ for a few
prompts, for the reason given above.

## Credits

- The edit we analyse is Astra's, from "LLMs can understand and write neural network weights"
  ([logs and code](https://github.com/RikiyaT/lm_understand-logs)). We use their four pairs, problem sets, grading and
  submitted edit (CC BY 4.0 for data, MIT for code).
- The weight-orthogonalization ⇔ directional-ablation equivalence is from Arditi et al. (2024), "Refusal in language
  models is mediated by a single direction".
- The learned sensing vector follows CLAS; our "clas" arm is its productive-text-only, no-bias variant at the MLP write.
- Benchmarks: MATH (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021), LiveCodeBench (Jain et al., 2024).

## License

MIT 
