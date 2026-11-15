#!/usr/bin/env python3
"""Add the clasd arms (one run) to the post's two-run Astra-regime test summary.

Per problem, each reference arm (control, s = 4v at N = 4 and 32) is averaged over our two runs, as in
results/astra_regime/test_summary.json; clasd's outcome is its single run. Paired differences and 95% problem-bootstrap
intervals use astra_regime_summary.boot (10,000 resamples, seed 0). --check recomputes 'ul_n32 - control' from the two
runs' dev-chosen arms and compares it with the shipped summary.

  clasd_two_run_summary.py --base results/astra_regime/test_summary.json --out test_summary_with_clasd.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/steer"))
from astra_regime_summary import CONF, METRICS, boot, load  # noqa: E402

R1 = ROOT / "results/astra_regime/scored/run1"
R1X = ROOT / "results/astra_regime/scored/run1_extra"
R2 = ROOT / "results/astra_regime/scored/run2"
CL = ROOT / "results/astra_regime/scored/corrective"
REFS = {"control": [(R1, "control"), (R2, "control")],
        "sv_n4_a4": [(R1X, "sv_n4_a4"), (R2, "sv_n4_a4")],
        "sv_n32_a4": [(R1, "sv_n32_a4"), (R2, "sv_n32_a4")]}
NEW = {"clasd_n4": [(CL, "clasd_n4_b0.1")], "clasd_n32": [(CL, "clasd_n32_b0.1")]}
CHECK = {"ul_n32": [(R1, "ul_n32_b0.1"), (R2, "ul_n32_b0")]}


def per_problem(runs):
    out, setof = {}, {}
    for split in CONF:
        rs = [load(g, split, a) for g, a in runs]
        assert all(r is not None for r in rs), (split, runs)
        for i in rs[0]:
            out[i] = {k: sum(float(r[i][k]) for r in rs) / len(rs) for _, k in METRICS}
            out[i]["source"] = rs[0][i]["source"]
            setof[i] = split
    return out, setof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    s = json.loads(Path(a.base).read_text())
    arms = {**REFS, **NEW, **(CHECK if a.check else {})}
    data = {}
    for name, runs in arms.items():
        data[name], setof = per_problem(runs)
    ids = sorted(data["control"])
    groups = {"pooled": ids, **{g: [i for i in ids if data["control"][i]["source"] == g] for g in ("gsm8k", "math")},
              **{sp: [i for i in ids if setof[i] == sp] for sp in CONF}}

    def stats(arm, G):
        return {"n": len(G), **{m: sum(data[arm][i][k] for i in G) / len(G) for m, k in METRICS}}

    def paired(x, y):
        d = {}
        for g, G in groups.items():
            d[g] = {}
            for m, k in METRICS:
                diffs = [data[x][i][k] - data[y][i][k] for i in G]
                d[g][m] = {"diff": sum(diffs) / len(diffs), "ci95": boot(diffs)}
        return d

    for name in list(NEW) + (list(CHECK) if a.check else []):
        stat = {g: stats(name, G) for g, G in groups.items()}
        pr = {f"{name} - {ref}": paired(name, ref) for ref in REFS}
        if name in CHECK:
            for g in ("pooled", "gsm8k", "math"):
                for m in ("accuracy", "loop", "tokens"):
                    ours, shipped = pr[f"{name} - control"][g][m], s["paired"][f"{name} - control"][g][m]
                    print(f"check {name} {g} {m}: ours {ours['diff']:+.4f} {ours['ci95']}  shipped {shipped['diff']:+.4f} {shipped['ci95']}")
            continue
        s["per_arm"][name] = stat
        s["paired"].update(pr)
    for ref in REFS:  # the references must be exactly the shipped ones
        for m in ("accuracy", "loop"):
            assert abs(stats(ref, ids)[m] - s["per_arm"][ref]["pooled"][m]) < 1e-9, (ref, m)
    Path(a.out).write_text(json.dumps(s, indent=1))
    for name in NEW:
        p = s["per_arm"][name]["pooled"]
        print(f"{name}: acc {p['accuracy']:.3f} loop {p['loop']:.3f} cap {p['cap_hit']:.3f} tok {p['tokens']:.0f}")


if __name__ == "__main__":
    main()
