#!/usr/bin/env python3
"""Experiment-1 (learned sensing s vs s = alpha v) test results as delta heatmaps, in the steering-report grid style.

Four figures, one per regime x dataset (the pairs that train s and build v are math-style text, so MATH is near the
training distribution and LiveCodeBench / GSM8K show transfer differently):
  Qwen3-1.7B greedy: MATH level 5, LiveCodeBench.   Qwen3.5-0.8B Astra sampler: GSM8K, MATH.
Rows are the dev-selected arms grouped by pair count N, plus a reference section (the unedited model with thinking off
when its runs exist; in the Astra regime also Astra's edit (s=4v, N=4)). Column groups: paired differences against the
unedited model (thinking on), and against s = alpha v at the same N. Cell text is the difference in points; '*' marks a
95% bootstrap interval that excludes zero (the interval is in each cell's hover title). Green = better (higher accuracy
/ thinking closed, lower loop / cap rate), red = worse.

  build_exp1_learned_sensing_heatmaps.py --qwen3 exp1_test_summary.json --astra test_summary_final.json --out DIR
The no-thinking row is read from the same files (arm control_nothink) when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from steering_delta_grid import DeltaGridSection, DeltaMetricGroup, DeltaMetricSpec, write_delta_grid_svg

LABEL = {"sv": "s = αv", "clas": "clas (NTP)", "ul": "ul (unlikelihood)", "contrast": "contrast (NTP+UL)"}
REF_LABEL = {"control_nothink": "unedited, thinking off", "sv_n4_a4": "Astra’s edit",
             
             "random_full_matched": "random dir. (‖ΔW‖ = s=αv N32)"}
NOTHINK = "control_nothink"


def is_random(arm):
    return arm.startswith("random_")


def arm_label(arm):
    if arm in REF_LABEL:
        return REF_LABEL[arm]
    if arm.startswith("random_full_matched_"):  # matched to the dev-chosen s = alpha v at N = 32
        return REF_LABEL["random_full_matched"]
    fam = arm.split("_n")[0]
    rest = arm.split("_n")[1].split("_", 1)[1] if "_" in arm.split("_n")[1] else ""
    knobs = rest.replace("a", "α=", 1) if fam == "sv" else rest.replace("b", "β=", 1).replace("_s", ", upd ")
    return f"{LABEL.get(fam, fam)}  ({knobs})" if knobs else LABEL.get(fam, fam)


def cell(stat):
    """stat: {'mean','lo','hi'} as fractions -> (value, text, interval text)."""
    if stat is None:
        return None, "—", ""
    sig = stat["lo"] > 0 or stat["hi"] < 0
    return stat["mean"], f"{100 * stat['mean']:+.1f}{'*' if sig else ''}", \
        f"{100 * stat['mean']:+.1f} [{100 * stat['lo']:+.1f}, {100 * stat['hi']:+.1f}]"


def n_of(arm):
    return int(arm.split("_n")[1].split("_")[0])


def make_row(arm, cols, get):
    r = {"arm": arm, "_title": {}, "_text": {}}
    for c in cols:
        r[c[0]], r["_text"][c[0]], r["_title"][c[0]] = cell(get(arm, *c[1:]))
    return r


def qwen3_figure(d, ds, out):
    g0 = {"math": "math", "lcb": "lcb"}[ds]
    cols = [("c_acc", "vs_control", g0, "correct"), ("c_acc_loop", "vs_control", f"{g0}_loop", "correct"),
            ("c_acc_non", "vs_control", f"{g0}_nonloop", "correct"), ("c_loop", "vs_control", g0, "loop"),
            ("c_closed", "vs_control", g0, "closed"),
            ("s_acc", "vs_sv", g0, "correct"), ("s_acc_loop", "vs_sv", f"{g0}_loop", "correct"),
            ("s_acc_non", "vs_sv", f"{g0}_nonloop", "correct"), ("s_loop", "vs_sv", g0, "loop")]

    def get(arm, cmp_, g, m):
        if arm == NOTHINK and m == "closed":
            return None  # no thinking block to close
        return d[cmp_].get(arm, {}).get(g, {}).get(m)

    rows = {4: [], 32: []}
    for arm in d["levels"]:
        if arm not in ("control", NOTHINK):
            rows[n_of(arm)].append(make_row(arm, cols, get))
    sections = []
    if NOTHINK in d["levels"]:
        sections.append(DeltaGridSection("Reference", [make_row(NOTHINK, cols, get)]))
    sections += [DeltaGridSection("N = 4 training pairs", rows[4]),
                 DeltaGridSection("N = 32 training pairs (chosen on dev)", rows[32])]
    L = d["levels"]["control"]
    pct = lambda g, m: 100 * L[g][m]["mean"]
    n, nl = L[g0]["correct"]["n"], L[f"{g0}_loop"]["correct"]["n"]
    name = {"math": "MATH level 5", "lcb": "LiveCodeBench"}[ds]
    metrics = [DeltaMetricSpec("c_acc", "acc · all", "positive", 0.08), DeltaMetricSpec("c_acc_loop", "acc · looping", "positive", 0.08),
               DeltaMetricSpec("c_acc_non", "acc · non-loop", "positive", 0.08), DeltaMetricSpec("c_loop", "loop rate", "negative", 0.40),
               DeltaMetricSpec("c_closed", "closed think", "positive", 0.20),
               DeltaMetricSpec("s_acc", "acc · all", "positive", 0.08), DeltaMetricSpec("s_acc_loop", "acc · looping", "positive", 0.08),
               DeltaMetricSpec("s_acc_non", "acc · non-loop", "positive", 0.08), DeltaMetricSpec("s_loop", "loop rate", "negative", 0.20)]
    draw(out, f"Qwen3-1.7B greedy, {name}: learned s vs s = αv (test, {n} prompts)",
         ["Cells: paired difference in points; * = 95% prompt-bootstrap interval excludes 0 (hover for the interval).",
          "Green = better (higher accuracy or closure, fewer loops), red = worse. Accuracy colour saturates at ±8 pp.",
          f"unedited (thinking on): acc {pct(g0, 'correct'):.1f}% (looping {pct(g0 + '_loop', 'correct'):.1f}%, "
          f"non-looping {pct(g0 + '_nonloop', 'correct'):.1f}%), loop {pct(g0, 'loop'):.1f}%, closed {pct(g0, 'closed'):.1f}%"],
         sections, metrics, [DeltaMetricGroup("Δ vs unedited model (thinking on)", 0, 5), DeltaMetricGroup("Δ vs s = αv at the same N", 5, 4)],
         f"{nl} looping / {n - nl} non-looping prompts (loop-enriched by design), one greedy draw each. "
         "s and v are trained / built from math-style hand-written pairs.")


def astra_figure(s, ds, out):
    P = s["paired"]
    sv = {n_of(a): a for a in s["per_arm"] if a.startswith("sv_n")}  # the dev-chosen s = alpha v per N
    RANDOM = tuple(sorted((a for a in s["per_arm"] if is_random(a) and a != "random_top64_matched_astra"), reverse=True))
    cols = [("c_acc", "control", "accuracy"), ("c_loop", "control", "loop"), ("c_closed", "control", "think_closed"),
            ("c_cap", "control", "cap_hit"), ("s_acc", "sv", "accuracy"), ("s_loop", "sv", "loop")]

    def get(arm, ref, m):
        if ref == "sv" and (arm.startswith("sv") or arm in REF_LABEL or is_random(arm)):
            return None
        if arm == NOTHINK and m == "think_closed":
            return None
        x = P.get(f"{arm} - {'control' if ref == 'control' else sv[n_of(arm)]}", {}).get(ds, {}).get(m)
        return None if x is None else {"mean": x["diff"], "lo": x["ci95"][0], "hi": x["ci95"][1]}

    order = ["sv", "clas", "ul", "contrast"]
    rows = {4: [], 32: []}
    arms = [a for a in s["per_arm"] if a not in ("control", NOTHINK, "astra_top64_a6", "sv_n4_a4", "random_top64_matched_astra", "clas_n4_b0") + RANDOM]  # clas_n4_b0 = off-rule update 24
    for arm in sorted(arms, key=lambda a: order.index(a.split("_n")[0])):
        rows[n_of(arm)].append(make_row(arm, cols, get))
    ref = [make_row(a, cols, get) for a in (NOTHINK, "sv_n4_a4") + RANDOM if a in s["per_arm"]]
    c = s["per_arm"]["control"][ds]
    name = {"gsm8k": "GSM8K", "math": "MATH"}[ds]
    metrics = [DeltaMetricSpec("c_acc", "acc", "positive", 0.50), DeltaMetricSpec("c_loop", "loop rate", "negative", 0.80),
               DeltaMetricSpec("c_closed", "closed think", "positive", 0.80), DeltaMetricSpec("c_cap", "cap hit", "negative", 0.80),
               DeltaMetricSpec("s_acc", "acc", "positive", 0.10), DeltaMetricSpec("s_loop", "loop rate", "negative", 0.30)]
    draw(out, f"Qwen3.5-0.8B, Astra sampler, {name}: learned s vs s = αv (test, {c['n']} problems)",
         ["Cells: paired difference in points; * = 95% problem-bootstrap interval excludes 0 (hover for the interval).",
          "Green = better, red = worse. Left block saturates at ±50 pp (the edit itself is large); right block at ±10 pp.",
          f"unedited (thinking on): acc {100 * c['accuracy']:.1f}%, loop {100 * c['loop']:.1f}%, "
          f"closed {100 * c['think_closed']:.1f}%, cap hit {100 * c['cap_hit']:.1f}%"],
         [DeltaGridSection("Reference", ref), DeltaGridSection("N = 4 training pairs", rows[4]),
          DeltaGridSection("N = 32 training pairs", rows[32])],
         metrics, [DeltaMetricGroup("Δ vs unedited model (thinking on)", 0, 4), DeltaMetricGroup("Δ vs s = αv, same N", 4, 2)],
         "T 0.6, presence 1.5, 32,768-token cap, one seeded draw per problem. ul/contrast keep update 24 (closure rule passes every update).",
         min_width=960)


def draw(out, title, subtitles, sections, metrics, groups, footer, min_width=1240):
    write_delta_grid_svg(
        out_path=out, title=title, subtitles=subtitles, sections=sections, metrics=metrics, metric_groups=groups,
        row_label=lambda r: arm_label(r["arm"]), cell_text=lambda r, m: r["_text"][m.field],
        cell_title=lambda r, m, t: f"{arm_label(r['arm'])} · {m.label}: {r['_title'][m.field] or 'not applicable'} (points, 95% CI)",
        footer=footer, left=290, top=150, cell_w=96, row_h=34, min_width=min_width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3", default="", help="Qwen3-1.7B test summary (Figures 2a, 2b)")
    ap.add_argument("--astra", default="", help="Astra-regime test summary (Figures 1a, 1b)")
    ap.add_argument("--only", default="1a,1b,2a,2b", help="figures to build, comma-separated: 1a 1b 2a 2b")
    ap.add_argument("--astra-nothink", default="", help="test summary of control_nothink vs control (Astra regime)")
    ap.add_argument("--astra-random", default="", help="test summary of the random-direction controls vs control "
                    "(astra_regime_summary.py test --arms random_... --refs control); added as reference rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    only = set(a.only.split(","))
    paths = []
    if only & {"2a", "2b"}:
        q = json.loads(Path(a.qwen3).read_text())
        for fig, ds in (("2a", "math"), ("2b", "lcb")):
            if fig in only:
                paths.append(out / f"exp1_qwen3_{ds}_delta_grid.svg")
                qwen3_figure(q, ds, paths[-1])
    if only & {"1a", "1b"}:
        s = json.loads(Path(a.astra).read_text())
        if a.astra_nothink:  # astra_regime_summary.py test --arms control_nothink --refs control
            t = json.loads(Path(a.astra_nothink).read_text())
            s["per_arm"][NOTHINK] = t["per_arm"][NOTHINK]
            s["paired"][f"{NOTHINK} - control"] = t["paired"][f"{NOTHINK} - control"]
        if a.astra_random:
            t = json.loads(Path(a.astra_random).read_text())
            for r in (x for x in t["per_arm"] if is_random(x)):
                s["per_arm"][r] = t["per_arm"][r]
                s["paired"][f"{r} - control"] = t["paired"][f"{r} - control"]
        for fig, ds in (("1a", "gsm8k"), ("1b", "math")):
            if fig in only:
                paths.append(out / f"exp1_astra_{ds}_delta_grid.svg")
                astra_figure(s, ds, paths[-1])
    print(*paths, sep="\n")


if __name__ == "__main__":
    main()
