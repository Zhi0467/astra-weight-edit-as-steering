#!/usr/bin/env python3
"""Summaries for the Astra-regime sensing experiment (outputs of astra_regime.py gen under <gen>/<split>/<arm>/).

gate   Unedited control and Astra's edit on validation_4: our accuracy, cap hits (finish = length), loop flag, tokens,
       beside Astra's per-problem record (final_confirmation.csv), with per-problem agreement.
dev    Every arm on validation_1 (minus --exclude, e.g. learned arms superseded by a later stopping update). Per family
       (sv = s = alpha v; clas; ul; contrast) and per N (4, 32): the setting with the highest pooled accuracy, ties broken
       by the lower loop rate, then by name. N = 32 is chosen only if its best arm (over all
       families) beats N = 4's best arm on accuracy.
test   Given arms on the confirmation sets (validation_3, validation_4, final_1-3; 1,152 problems). For each arm against
       each reference arm: paired differences (arm - reference) in accuracy, cap-hit rate, loop rate and completion
       tokens, with 95% problem-bootstrap intervals (10,000 resamples of problems, seeded), pooled, by dataset (GSM8K /
       MATH) and by set.
diag   Dev-only sampler diagnostic on validation_1 (greedy; no presence penalty) against Astra's sampler, per arm.

  astra_regime_summary.py gate --gen G --astra-src A
  astra_regime_summary.py dev --gen G [--exclude a,b] --out dev_selection.json
  astra_regime_summary.py test --gen G --arms a,b,c --refs control,sv_n4_a2 --out test_summary.json
  astra_regime_summary.py diag --gen G --arms control,sv_n32_a4 --samplers greedy,nopp --out diag.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONF = ("validation_3", "validation_4", "final_1", "final_2", "final_3")
METRICS = (("accuracy", "correct"), ("cap_hit", "cut"), ("loop", "loop"), ("think_closed", "think_closed"), ("tokens", "tokens"))


def load(gen, split, arm):
    p = Path(gen) / split / arm / "results.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else None


def stats(rows):
    n = len(rows)
    return {"n": n, **{m: sum(float(r[k]) for r in rows) / n for m, k in METRICS}}


def family(arm):
    return arm.split("_n")[0]


def n_of(arm):
    return int(arm.split("_n")[1].split("_")[0])


def cmd_gate(a):
    A = {r["id"]: r for r in csv.DictReader(open(Path(a.astra_src) / "final_confirmation.csv"))}
    out = {}
    for arm, col in (("control", "original"), ("astra_top64_a6", "edited")):
        R = load(a.gen, "validation_4", arm)
        ids = list(R)
        ours = stats(list(R.values()))
        theirs = {"accuracy": sum(int(A[i][col]) for i in ids) / len(ids), "cap_hit": sum(int(A[i][col + "_cut"]) for i in ids) / len(ids),
                  "tokens": sum(int(A[i][col + "_tokens"]) for i in ids) / len(ids)}
        agree = sum(int(R[i]["correct"]) == int(A[i][col]) for i in ids) / len(ids)
        out[arm] = {"ours": ours, "astra": theirs, "per_problem_correct_agreement": agree,
                    "by_source": {s: stats([R[i] for i in ids if R[i]["source"] == s]) for s in ("gsm8k", "math")}}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def cmd_dev(a):
    excl = set(filter(None, a.exclude.split(",")))
    arms = sorted(p.name for p in (Path(a.gen) / "validation_1").iterdir() if (p / "summary.json").exists() and p.name not in excl)
    S = {arm: stats(list(load(a.gen, "validation_1", arm).values())) for arm in arms}
    key = lambda arm: (-S[arm]["accuracy"], S[arm]["loop"], arm)
    chosen = {}
    for n in (4, 32):
        for fam in ("sv", "clas", "ul", "contrast", "clasd"):
            c = sorted([x for x in arms if x not in ("control", "astra_top64_a6") and family(x) == fam and n_of(x) == n], key=key)
            if c:
                chosen[f"{fam}_n{n}"] = c[0]
    best = {n: min([v for k, v in chosen.items() if k.endswith(f"_n{n}")], key=key) for n in (4, 32)}
    n_choice = 32 if S[best[32]]["accuracy"] > S[best[4]]["accuracy"] else 4
    res = {"split": "validation_1", "per_arm": S, "chosen_per_family_and_n": chosen, "best_per_n": best, "chosen_n": n_choice,
           "excluded": sorted(excl),
           "rule": "highest pooled accuracy, then lower loop rate; N=32 only if its best arm beats N=4's best on accuracy"}
    for arm in sorted(arms, key=key):
        s = S[arm]
        print(f"{arm:22s} acc {s['accuracy']:.3f} cap {s['cap_hit']:.3f} loop {s['loop']:.3f} tok {s['tokens']:.0f}")
    print(json.dumps({k: res[k] for k in ("chosen_per_family_and_n", "best_per_n", "chosen_n")}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def boot(diffs, B=10000, seed=0):
    import numpy as np
    d = np.asarray(diffs, dtype=float)
    ms = np.sort(d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1))
    return [float(ms[int(0.025 * B)]), float(ms[int(0.975 * B) - 1])]


def cmd_diag(a):
    """Dev-only sampler diagnostic: each arm on validation_1 under Astra's sampler and under each alternative sampler
    (gen/validation_1__<sampler>), with paired differences (alternative - Astra's sampler) per arm."""
    out = {}
    for arm in a.arms.split(","):
        base = load(a.gen, "validation_1", arm)
        out[arm] = {"astra": stats(list(base.values()))}
        for smp in a.samplers.split(","):
            R = load(a.gen, f"validation_1__{smp}", arm)
            out[arm][smp] = stats(list(R.values()))
            out[arm][f"{smp} - astra"] = {m: {"diff": sum(float(R[i][k]) - float(base[i][k]) for i in base) / len(base),
                                             "ci95": boot([float(R[i][k]) - float(base[i][k]) for i in base], B=a.boot)} for m, k in METRICS}
        for smp in ["astra", *a.samplers.split(",")]:
            q = out[arm][smp]
            print(f"{arm:18s} {smp:7s} acc {q['accuracy']:.3f} cap {q['cap_hit']:.3f} loop {q['loop']:.3f} closed {q['think_closed']:.3f} tok {q['tokens']:.0f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def cmd_test(a):
    arms, refs = a.arms.split(","), a.refs.split(",")
    data = {arm: {} for arm in set(arms) | set(refs)}
    setof = {}
    for split in CONF:
        for arm in data:
            R = load(a.gen, split, arm)
            if R is None:
                raise SystemExit(f"missing {split}/{arm}")
            data[arm].update(R)
            for i in R:
                setof[i] = split
    ids = sorted(data[arms[0]])
    src = {i: data[arms[0]][i]["source"] for i in ids}
    groups = {"pooled": ids, **{s: [i for i in ids if src[i] == s] for s in ("gsm8k", "math")},
              **{s: [i for i in ids if setof[i] == s] for s in CONF}}
    out = {"n": len(ids), "per_arm": {arm: {g: stats([data[arm][i] for i in G]) for g, G in groups.items()} for arm in data}, "paired": {}}
    for arm in arms:
        for ref in refs:
            if arm == ref:
                continue
            d = {}
            for g, G in groups.items():
                d[g] = {}
                for m, k in METRICS:
                    diffs = [float(data[arm][i][k]) - float(data[ref][i][k]) for i in G]
                    d[g][m] = {"diff": sum(diffs) / len(diffs), "ci95": boot(diffs, B=a.boot)}
            out["paired"][f"{arm} - {ref}"] = d
    for arm in data:
        q = out["per_arm"][arm]["pooled"]
        print(f"{arm:22s} acc {q['accuracy']:.3f} cap {q['cap_hit']:.3f} loop {q['loop']:.3f} closed {q['think_closed']:.3f} tok {q['tokens']:.0f}")
    for k, d in out["paired"].items():
        p = d["pooled"]
        print(f"{k:40s} acc {p['accuracy']['diff']:+.3f} [{p['accuracy']['ci95'][0]:+.3f},{p['accuracy']['ci95'][1]:+.3f}]"
              f"  cap {p['cap_hit']['diff']:+.3f}  loop {p['loop']['diff']:+.3f}  closed {p['think_closed']['diff']:+.3f}  tok {p['tokens']['diff']:+.0f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "dev", "test", "diag"])
    ap.add_argument("--samplers", default="greedy,nopp")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--astra-src", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--refs", default="control")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"gate": cmd_gate, "dev": cmd_dev, "test": cmd_test, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
