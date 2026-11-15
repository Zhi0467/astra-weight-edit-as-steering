#!/usr/bin/env python3
"""Qwen3-1.7B test summary with the corrective positive-only arm (clasd, one run) added to the post's two-run numbers.

Per prompt, each arm's metrics are averaged over our two runs (each run's dev-chosen arm of a family; s = alpha v,
control and thinking-off are the same arms in both runs), then exp1_test_summary.summarize computes levels and paired
differences with its stratified bootstrap. clasd enters with its single run. --check compares every shipped arm with
results/qwen3/exp1_test_summary.json.

  qwen3_two_run_summary.py --repo REPO --out OUT.json [--check] [--no-clasd]
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
p.add_argument("--scored", help="Per-prompt scored JSONL; defaults to results/qwen3/scored.jsonl.gz")
p.add_argument("--out", required=True)
p.add_argument("--check", action="store_true")
p.add_argument("--no-clasd", action="store_true")
a = p.parse_args()
R = Path(a.repo)
sys.path.insert(0, str(R / "scripts/steer"))
import exp1_test_summary as T  # noqa: E402

T.METRICS = T.METRICS + ("tokens",)  # completion length, for the figure's token column


# Scored records retain the original prompt identities and metrics from each rollout.
# Two baseline runs are averaged per prompt; Corrective NTP contributes one run.
records = {}
scored = Path(a.scored) if a.scored else R / "results/qwen3/scored.jsonl.gz"
open_scored = gzip.open if scored.suffix == ".gz" else open
for line in open_scored(scored, "rt"):
    row = json.loads(line)
    k = (row["source"], row["dataset"], row["prompt_id"])
    d = records.setdefault((row["run"], row["arm"]), {})
    if k in d:
        raise ValueError(f"duplicate prompt: {row['run']}/{row['arm']}/{k}")
    d[k] = row["metrics"]


def read_scored(gen, graded, arm, label, subset=False):
    d = records[(gen, arm)]
    if set(d) != set(label):
        raise ValueError(f"prompt mismatch: {gen}/{arm}")
    return d


T.read_arm = read_scored
runs = [(run, run, json.load(open(R / f"results/qwen3/runs/{n}/exp1_selection.json"))["test_arms"])
        for n, run in ((1, "run1"), (2, "run2"))]
label = T.read_labels(R / "data/qwen3/splits/test12", R / "data/qwen3/loop_labels.jsonl")


def avg(parts):
    return {k: {m: float(np.mean([d[k][m] for d in parts])) for m in T.METRICS} for k in parts[0]}


def family(arms, fam, n):
    return [x for x in arms if x.startswith(f"{fam}_n{n}_")][0]


data = {}
data["control_nothink"] = avg([T.read_arm(g, r, "control_nothink", label, subset=True) for g, r, _ in runs])
data["control"] = avg([T.read_arm(g, r, "control", label) for g, r, _ in runs])
for n in (4, 32):
    sv = family(runs[0][2], "sv", n)
    assert sv == family(runs[1][2], "sv", n)
    data[sv] = avg([T.read_arm(g, r, sv, label) for g, r, _ in runs])
    for fam in ("clas", "ul", "contrast"):
        data[f"{fam}_n{n}"] = avg([T.read_arm(g, r, family(arms, fam, n), label) for g, r, arms in runs])
    if not a.no_clasd:
        sel = json.load(open(R / "results/qwen3/selection_clasd.json"))["test_arms"]
        data[f"clasd_n{n}"] = T.read_arm("corrective", "corrective", family(sel, "clasd", n), label)
# report order as in the shipped summary, clasd after clas
order = ["control_nothink", "control"]
for n in (4, 32):
    order += [k for k in data if k.startswith(f"sv_n{n}_")] + [f"{f}_n{n}" for f in ("clas", "clasd", "ul", "contrast") if f"{f}_n{n}" in data]
data = {k: data[k] for k in order}
out = T.summarize(data, label)
def o_m(part, arm, g):
    return out[part][arm][g]


if a.check:
    ship = json.load(open(R / "results/qwen3/exp1_test_summary.json"))
    worst = 0.0
    for part in ("levels", "vs_control", "vs_sv"):
        for arm, v in ship[part].items():
            for g, ms in v.items():
                for m, s in ms.items():
                    if s is None or m not in o_m(part, arm, g):
                        continue
                    o = out[part][arm][g][m]
                    worst = max(worst, abs(o["mean"] - s["mean"]), abs(o["lo"] - s["lo"]), abs(o["hi"] - s["hi"]))
    if worst != 0:
        raise AssertionError(f"shipped summary differs by {worst}")
    print("All original means and intervals match the shipped summary exactly.")
Path(a.out).write_text(json.dumps(out, indent=1))
T.report(out)
