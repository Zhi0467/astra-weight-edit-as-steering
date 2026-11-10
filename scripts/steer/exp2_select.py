#!/usr/bin/env python3
"""Experiment-2 dev selection (Qwen3-1.7B greedy, N = 32): the alpha of each steering-vector construction, s = alpha v.

Rule: per construction, the highest pooled dev accuracy over MATH + LiveCodeBench (128 + 128 prompts), loop rate as
tie-break, over alpha in 1-4. The residual mean difference (md_resid) is experiment 1's s = alpha v arm (sv_n32_a*);
the random direction is a test-only control at alpha 2.
Input: dev tables from exp1_rollout_table.py ({"arm@update": {dataset: {n, acc, loop, ...}}}), merged.

  exp2_select.py --rollouts rollouts_dev_exp2.json --rollouts rollouts_dev_exp1.json --json exp2_selection.json
"""

import argparse
import json
from pathlib import Path

DS = ("math_level5", "livecodebench")
CONSTRUCTIONS = ("md_mlp", "logreg", "rfm", "pc1")


def pooled(r):
    n = sum(r[d]["n"] for d in DS)
    return {"acc": sum(r[d]["acc"] * r[d]["n"] for d in DS) / n, "loop": sum(r[d]["loop"] * r[d]["n"] for d in DS) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", action="append", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    roll = {}
    for p in a.rollouts:
        roll.update(json.loads(Path(p).read_text()))
    out = {"rule": "per construction: highest pooled dev accuracy over MATH + LiveCodeBench (128 + 128 prompts), loop "
                   "rate as tie-break", "selected": {}}
    for c in CONSTRUCTIONS:
        dev = {str(al): pooled(roll[f"vc_{c}_n32_a{al}@24"]) for al in (1, 2, 3, 4)}
        al = max(dev, key=lambda k: (dev[k]["acc"], -dev[k]["loop"]))
        out["selected"][c] = {"alpha": int(al), "arm": f"vc_{c}_n32_a{al}", "dev_pooled": dev}
    dev = {str(al): pooled(roll[f"sv_n32_a{al}@24"]) for al in (1, 2, 3, 4)}
    al = max(dev, key=lambda k: (dev[k]["acc"], -dev[k]["loop"]))  # experiment 1's rule for s = alpha v at N = 32
    out["md_resid"] = {"arm": f"sv_n32_a{al}", "note": "experiment 1's dev-chosen s = alpha v at N = 32; its test run "
                       "comes from run/qwen3_exp1.sh", "dev_pooled": dev}
    out["control"] = {"dev_pooled": pooled(roll["control@0"])}
    out["random"] = {"arm": "vc_random_n32_a2", "note": "test-only control, alpha 2 fixed in the experiment-2 design"}
    out["test_arms"] = [out["selected"][c]["arm"] for c in CONSTRUCTIONS] + ["vc_random_n32_a2"]
    Path(a.json).write_text(json.dumps(out, indent=1))
    for c in CONSTRUCTIONS:
        print(c, out["selected"][c]["arm"])


if __name__ == "__main__":
    main()
