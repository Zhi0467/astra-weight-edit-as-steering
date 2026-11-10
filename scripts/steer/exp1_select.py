#!/usr/bin/env python3
"""Experiment-1 dev selection (Qwen3-1.7B greedy), from finished dev rollouts and the teacher-forced stopping updates.

Rules:
  - stopping update of each learned s: lowest teacher-forced loop score among updates whose </think> closure check is no
    worse than the unedited model's (loop_s_devloss.py; read here from the score JSONs via choose(..., tol=inf)).
  - alpha (s = alpha v) and beta (learned s at its stopping update): highest pooled dev accuracy over MATH + LiveCodeBench
    (64 looping + 64 non-looping prompts each), loop rate as tie-break; per family and N.
  - N: 32 if the best N = 32 arm beats the best N = 4 arm on pooled dev accuracy, else 4.
  - families are not chosen; each family's chosen arm at each N runs on test with the unedited control.
Input: --rollouts JSON from rollout_table.py ({"arm@update": {dataset: {n, acc, loop, ...}}}), --scores devscore JSONs.

  exp1_select.py --rollouts rollouts_dev.json --scores 'devscore_part*.json' --json selection.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_s_devloss import choose  # noqa: E402

DS = ("math_level5", "livecodebench")


def pooled(r):
    n = sum(r[d]["n"] for d in DS)
    return sum(r[d]["acc"] * r[d]["n"] for d in DS) / n, sum(r[d]["loop"] * r[d]["n"] for d in DS) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    roll = json.loads(Path(a.rollouts).read_text())
    arms = {}
    for p in sorted(glob.glob(a.scores)):
        arms.update(json.loads(Path(p).read_text())["arms"])
    stop = {name: choose(r["history"], float("inf")) for name, r in arms.items()}

    cands, missing = {}, []
    for n in (4, 32):
        cands[("sv", n)] = [(f"sv_n{n}_a{al}", f"sv_n{n}_a{al}@24") for al in (1, 2, 3, 4)]
        for obj in ("clas", "ul", "contrast"):
            cands[(obj, n)] = [(f"{obj}_n{n}_b{b}_s{stop[f'{obj}_n{n}_b{b}']}", f"{obj}_n{n}_b{b}@{stop[f'{obj}_n{n}_b{b}']}")
                               for b in ("0", "0.1")]
    out = {"stop_update": stop, "families": {}, "control": None}
    c = roll.get("control@0")
    out["control"] = dict(zip(("acc", "loop"), pooled(c))) if c and all(d in c for d in DS) else None
    for (fam, n), lst in cands.items():
        rows = []
        for run, key in lst:
            r = roll.get(key)
            if not r or not all(d in r for d in DS):
                missing.append(run); continue
            acc, loop = pooled(r)
            rows.append({"arm": run, "acc": acc, "loop": loop, "math": r["math_level5"]["acc"], "lcb": r["livecodebench"]["acc"]})
        best = max(rows, key=lambda x: (x["acc"], -x["loop"])) if len(rows) == len(lst) else None
        out["families"][f"{fam}_n{n}"] = {"candidates": rows, "chosen": best["arm"] if best else None}
    b = {n: max((f for k, f in out["families"].items() if k.endswith(f"_n{n}") and f["chosen"]),
                key=lambda f: max(x["acc"] for x in f["candidates"]), default=None) for n in (4, 32)}
    if all(b.values()):
        acc = {n: max(x["acc"] for x in b[n]["candidates"]) for n in (4, 32)}
        out["N"] = 32 if acc[32] > acc[4] else 4
        out["best_acc_by_N"] = acc
    out["missing"] = missing
    out["test_arms"] = ["control"] + [f["chosen"] for f in out["families"].values() if f["chosen"]]
    Path(a.json).write_text(json.dumps(out, indent=1))
    print("stopping updates:", stop)
    if out["control"]:
        print(f"control: pooled acc {out['control']['acc']:.3f} loop {out['control']['loop']:.3f}")
    for k, f in out["families"].items():
        print(f"{k:12s} chosen {f['chosen']}")
        for x in f["candidates"]:
            print(f"    {x['arm']:22s} pooled {x['acc']:.3f} (MATH {x['math']:.3f} LCB {x['lcb']:.3f}) loop {x['loop']:.3f}")
    print("N:", out.get("N"), out.get("best_acc_by_N"), "| missing:", missing)


if __name__ == "__main__":
    main()
