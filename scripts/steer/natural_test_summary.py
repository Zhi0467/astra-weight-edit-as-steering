#!/usr/bin/env python3
"""Test summary of the natural-trace comparison (Qwen3-1.7B greedy, the 1,000 Section 5 test prompts): the steering
data's two sources, each with s = alpha v and a learned unlikelihood s chosen on the reduced dev set
(natural_dev_select.py), against the unedited model and against each other.

Several runs of the pipeline can be passed (--control, --short repeated): per prompt, every metric is averaged over the
runs, as in our results (the short pairs' arms come from two runs; each run's learned s is its own dev choice). Same
paired stratified bootstrap as exp1_test_summary.py (cells source x dataset x loop label, 10,000 resamples).

  natural_test_summary.py --split data/qwen3/splits/test12 \
      --control GEN GRADED [--control ...] --short GEN GRADED SV_ARM UL_ARM [--short ...] \
      --natural GEN GRADED SV_ARM UL_ARM --json OUT.json
GEN/<src>/math_level5/<arm>/ and GRADED/<src>/livecodebench/<arm>/ as in exp1_test_summary.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp1_test_summary as QS  # noqa: E402

GROUPS = ("all", "loop", "nonloop", "math", "lcb")
METRICS = ("correct", "loop", "closed")
PAIRS = (("natural_sv", "control"), ("natural_ul", "control"), ("short_sv", "control"), ("short_ul", "control"),
         ("natural_sv", "short_sv"), ("natural_ul", "short_ul"))


def avg(runs):
    return {k: {m: float(np.mean([r[k][m] for r in runs])) for m in METRICS} for k in runs[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--labels", default=str(QS.LABELS))
    ap.add_argument("--control", nargs=2, action="append", required=True, metavar=("GEN", "GRADED"))
    ap.add_argument("--short", nargs=4, action="append", required=True, metavar=("GEN", "GRADED", "SV_ARM", "UL_ARM"))
    ap.add_argument("--natural", nargs=4, required=True, metavar=("GEN", "GRADED", "SV_ARM", "UL_ARM"))
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    label = QS.read_labels(a.split, a.labels)
    read = lambda gen, graded, arm: QS.read_arm(gen, graded, arm, label)  # noqa: E731
    data = {
        "control": avg([read(g, r, "control") for g, r in a.control]),
        "short_sv": avg([read(g, r, sv) for g, r, sv, _ in a.short]),
        "short_ul": avg([read(g, r, ul) for g, r, _, ul in a.short]),
        "natural_sv": read(a.natural[0], a.natural[1], a.natural[2]),
        "natural_ul": read(a.natural[0], a.natural[1], a.natural[3]),
    }
    keys = sorted(label)
    rng = np.random.default_rng(20261106)
    cells = {}
    for i, k in enumerate(keys):
        cells.setdefault((k[0], k[1], label[k]), []).append(i)
    boot = {c: rng.integers(0, len(ix), size=(QS.B, len(ix))) for c, ix in cells.items()}

    def stat(x, grp):
        sel = [c for c in cells if QS.GROUPS[grp](*c)]
        ix = np.concatenate([np.array(cells[c]) for c in sel])
        tot = np.zeros(QS.B)
        for c in sel:
            tot += x[np.array(cells[c])][boot[c]].sum(1)
        bs = tot / len(ix)
        return {"mean": float(x[ix].mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                "n": int(len(ix))}

    arr = lambda arm, m: np.array([data[arm][k][m] for k in keys])  # noqa: E731
    out = {"arms": {"short": [{"sv": sv, "ul": ul} for _, _, sv, ul in a.short],
                    "natural": {"sv": a.natural[2], "ul": a.natural[3]}, "control_runs": len(a.control)},
           "n_prompts": len(keys), "levels": {}, "diffs": {}}
    for arm in data:
        out["levels"][arm] = {g: {m: stat(arr(arm, m), g) for m in METRICS} for g in GROUPS}
    for x, y in PAIRS:
        out["diffs"][f"{x} - {y}"] = {g: {m: stat(arr(x, m) - arr(y, m), g) for m in METRICS} for g in GROUPS}
    Path(a.json).write_text(json.dumps(out, indent=1))

    f = lambda s: f"{100 * s['mean']:+5.1f} [{100 * s['lo']:+5.1f}, {100 * s['hi']:+5.1f}]"  # noqa: E731
    print("arms:", json.dumps(out["arms"]))
    for g in GROUPS:
        print(f"\n== {g} (n {out['levels']['control'][g]['correct']['n']}) ==  accuracy / loop / closed thinking, %")
        for arm, L in out["levels"].items():
            print(f"  {arm:12s} " + " / ".join(f"{100 * L[g][m]['mean']:5.1f}" for m in METRICS))
        for p, D in out["diffs"].items():
            print(f"  {p:24s} acc {f(D[g]['correct'])}  loop {f(D[g]['loop'])}")


if __name__ == "__main__":
    main()
