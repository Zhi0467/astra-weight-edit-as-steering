#!/usr/bin/env python3
"""Dev selection for the natural-trace comparison: the same rule as exp1_select.py (per family, the highest pooled
MATH + LiveCodeBench dev accuracy, loop rate breaking ties), on the reduced dev set, i.e. dev minus the prompts whose
traces built the natural v and s (natural_traces_summary.json, drop_from_dev). Used for both data sources, so the short
pairs' arms are re-chosen on the same prompts.

  natural_dev_select.py --drop natural_traces_summary.json --runs RUNS/dev --arms sv_n32_a1,...,ul_n32_b0_s10,... --json OUT
RUNS/dev/<dataset>/<arm>/ hold finished dev rollouts (gen_qwen3.sh; LiveCodeBench graded). Families: sv (s = alpha v) and
the learned objective of each arm (ul, ...).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

DS = ("math_level5", "livecodebench")


def per_prompt(d):
    f = [x for x in glob.glob(f"{d}/*.jsonl.gz") if "preexisting" not in x]
    if not f or not Path(d, "DONE" if "math_level5" in d else "GRADED").exists():
        raise SystemExit(f"missing finished rollouts in {d}")
    out = {}
    for line in gzip.open(f[0], "rt"):
        r = json.loads(line)
        x = r["rollouts"][0]
        g = x.get("grading", {})
        out[r["sample_id"]] = (bool(g.get("correct", 0) or g.get("passed", False)), bool(x["loop_flag"]))
    return out


def stats(rows, drop):
    tot = {"n": 0, "acc": 0, "loop": 0}
    by = {}
    for ds in DS:
        k = [s for s in rows[ds] if s not in drop.get(ds, ())]
        a = sum(rows[ds][s][0] for s in k)
        lp = sum(rows[ds][s][1] for s in k)
        by[ds] = {"n": len(k), "acc": a / len(k), "loop": lp / len(k)}
        tot["n"] += len(k); tot["acc"] += a; tot["loop"] += lp
    return {"n": tot["n"], "acc": tot["acc"] / tot["n"], "loop": tot["loop"] / tot["n"], **by}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", required=True, help="natural_traces_summary.json")
    ap.add_argument("--runs", required=True, help="dev rollout root: <runs>/<dataset>/<arm>/")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    drop = {k: set(v) for k, v in json.loads(Path(a.drop).read_text())["drop_from_dev"].items()}
    arms = {}
    for arm in a.arms.split(","):
        rows = {ds: per_prompt(f"{a.runs}/{ds}/{arm}") for ds in DS}
        arms[arm] = {"full": stats(rows, {}), "reduced": stats(rows, drop)}
    fams = {}
    for arm in arms:
        fams.setdefault(arm.split("_n")[0], []).append(arm)
    chosen = {f: max(v, key=lambda x: (arms[x]["reduced"]["acc"], -arms[x]["reduced"]["loop"])) for f, v in fams.items()}
    Path(a.json).write_text(json.dumps({"arms": arms, "chosen": chosen}, indent=1))
    for arm, s in arms.items():
        f, r = s["full"], s["reduced"]
        print(f"{arm:20s} reduced dev (n {r['n']}): acc {100 * r['acc']:.1f} loop {100 * r['loop']:.1f} "
              f"| full dev (n {f['n']}): acc {100 * f['acc']:.1f} loop {100 * f['loop']:.1f}")
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
