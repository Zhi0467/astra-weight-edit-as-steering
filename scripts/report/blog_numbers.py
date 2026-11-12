#!/usr/bin/env python3
"""Print every number the blog post quotes, computed from the summaries, so a rerun can be checked line by line.

  blog_numbers.py --qwen3 results/qwen3 --astra results/astra_regime            ours (the post: two runs averaged)
  blog_numbers.py --qwen3 results/qwen3/runs/1 --astra results/astra_regime/runs/1   one of our runs
  blog_numbers.py --qwen3 $WORK/qwen3/exp1 --astra $WORK/astra_regime           your run

The "capped but not loop-flagged" count needs per-problem outputs, so it is printed only when <astra>/gen exists
(a rerun), or from capped_ul_n32.json in results/. The §4 short-pairs-vs-natural-traces numbers are read from
<qwen3>/natural_vs_short/ (ours) or <qwen3>/../natural/ (a rerun of run/qwen3_natural.sh), if present.
"""

import argparse
import json
from pathlib import Path

SPLITS = ("validation_4", "validation_3", "final_1", "final_2", "final_3")


def pct(x):
    return f"{100 * x:.1f}%"


def pp(x):
    return f"{100 * x:+.1f}"


def ci(d):
    return f"{pp(d['diff'])} [{pp(d['ci95'][0])}, {pp(d['ci95'][1])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3", required=True, help="folder with exp1_test_summary.json and exp1_selection.json")
    ap.add_argument("--astra", required=True, help="folder with test_summary*.json, dev and diagnostic JSONs "
                    "(test_summary_resid.json from run/astra_controls.sh is optional)")
    a = ap.parse_args()
    Q, A = Path(a.qwen3), Path(a.astra)
    T = json.loads((A / "test_summary.json").read_text())
    NT = json.loads((A / "test_summary_nothink.json").read_text())
    sel = json.loads((A / "dev_selection.json").read_text())["chosen_per_family_and_n"]
    arm = lambda f, n: sel[f"{f}_n{n}"]
    m = lambda d, x, g="pooled": d["per_arm"][x][g]

    print("§2 baselines (Astra regime, 1,152 problems): accuracy / loop / cap hit")
    for name, d, x in (("unedited, thinking on", T, "control"), ("unedited, thinking off", NT, "control_nothink"),
                       ("Astra's edit", T, "sv_n4_a4")):
        r = m(d, x)
        print(f"  {name:24s} {pct(r['accuracy'])} / {pct(r['loop'])} / {pct(r['cap_hit'])}")

    print("§3 N = 32: accuracy, loop")
    for f in ("sv", "ul", "contrast", "clas"):
        r = m(T, arm(f, 32))
        print(f"  {f:9s} {arm(f, 32):18s} acc {pct(r['accuracy'])}  loop {pct(r['loop'])}  cap {pct(r['cap_hit'])}")
    print("§3 N = 4: accuracy, loop")
    for f in ("sv", "ul", "contrast", "clas"):
        r = m(T, arm(f, 4))
        print(f"  {f:9s} {arm(f, 4):18s} acc {pct(r['accuracy'])}  loop {pct(r['loop'])}")
    print("  N = 4 -> 32 accuracy gain:", ", ".join(
        f"{f} {pp(m(T, arm(f, 32))['accuracy'] - m(T, arm(f, 4))['accuracy'])}" for f in ("sv", "ul", "contrast")))
    for n in (4, 32):
        for f in ("ul", "contrast"):
            key = arm(f, n) + " - " + arm("sv", n)
            print(f"  N = {n} {f} - sv accuracy: {ci(T['paired'][key]['pooled']['accuracy'])}")
    print("§3 N = 32 change vs s = alpha v, GSM8K | MATH (pp)")
    for key, name in (("accuracy", "accuracy"), ("loop", "loop flag"), ("cap_hit", "cap hits"), ("think_closed", "thinking closure")):
        cells = []
        for g in ("gsm8k", "math"):
            cells.append(" / ".join(pp(T["paired"][arm(f, 32) + " - " + arm("sv", 32)][g][key]["diff"]) for f in ("ul", "contrast")))
        print(f"  {name:17s} UL / contrast: {cells[0]} | {cells[1]}")
    gen = A / "gen"
    if (A / "capped_ul_n32.json").exists():  # our two runs averaged
        c = json.loads((A / "capped_ul_n32.json").read_text())
        print(f"  capped ul N = 32 responses, all runs: {c['capped']} of {c['responses']}, of which not loop-flagged: "
              f"{c['capped_not_loop']} ({pct(c['capped_not_loop'] / c['capped'])})")
    elif gen.exists():
        rows = [json.loads(l) for s in SPLITS for l in open(gen / s / arm("ul", 32) / "results.jsonl")]
        capped = [r for r in rows if r["finish"] == "length"]
        n_free = sum(not r["loop"] for r in capped)
        print(f"  capped {arm('ul', 32)} responses: {len(capped)}, of which not loop-flagged: {n_free}")

    R = A / "test_summary_resid.json"  # run/astra_controls.sh
    if R.exists():
        R = json.loads(R.read_text())
        sv, ul = arm("sv", 32), arm("ul", 32)
        print("§4 MLP write vs residual stream, N = 32: accuracy / loop / cap hit")
        for x in (sv, "resid_" + sv, ul, "resid_" + ul):
            r = m(R, x)
            print(f"  {x:20s} {pct(r['accuracy'])} / {pct(r['loop'])} / {pct(r['cap_hit'])}")
        d = R["paired"][f"resid_{ul} - {ul}"]["pooled"]
        print(f"  residual - write, learned ul: accuracy {ci(d['accuracy'])}, loop {pp(d['loop']['diff'])}")
    RA, RD = A / "test_summary_resid_alpha.json", A / "resid_alpha_dev.json"  # run/astra_resid_alpha.sh
    if RA.exists() and RD.exists():
        RA, RD = json.loads(RA.read_text()), json.loads(RD.read_text())
        x = RD["chosen"]
        print("§4 residual s = alpha v, dev accuracy:", ", ".join(f"{k.split('_a')[-1]}: {pct(v['accuracy'])}" for k, v in RD["per_arm"].items()),
              "-> chosen", x)
        r = m(RA, x)
        print(f"  {x} on test: {pct(r['accuracy'])} / {pct(r['loop'])} / {pct(r['cap_hit'])}")
        for ref in (arm("sv", 32), arm("ul", 32)):
            d = RA["paired"][f"{x} - {ref}"]["pooled"]
            print(f"  {x} - {ref} (write): accuracy {ci(d['accuracy'])}, loop {ci(d['loop'])}")

    D = json.loads((A / "diag_samplers_validation_1.json").read_text())
    print("§7 dev sampler diagnostic (validation_1): accuracy under Astra's sampler -> greedy -> no presence penalty")
    for x, r in D.items():
        if x == "astra_top64_a6":
            continue  # Later neuron-selected version is reported only in the endnote.
        print(f"  {x:18s} {pct(r['astra']['accuracy'])} -> {pct(r['greedy']['accuracy'])} -> {pct(r['nopp']['accuracy'])}")
    r = m(T, "astra_top64_a6")
    print(f"Endnote: later neuron-selected edit: {pct(r['accuracy'])} / {pct(r['loop'])} / {pct(r['cap_hit'])}")
    for f in [A / "astra_check_n4.json"] if (A / "astra_check_n4.json").exists() else sorted(A.glob("runs/*/astra_check_n4.json")):
        C = json.loads(f.read_text())
        print(f"Endnote: |cos(our four-pair v, Astra's u)| per layer ({f.parent.name}):",
              ", ".join(f"{l}: {c['abs_cos_v_u']:.6f}" for l, c in C.items()))

    S = json.loads((Q / "exp1_test_summary.json").read_text())
    QS = json.loads((Q / "exp1_selection.json").read_text())
    print("§3/§4 Qwen3-1.7B stopping updates:", QS["stop_update"])
    print(f"§5 Qwen3-1.7B (greedy, {S['n_prompts']} prompts): learned - s = alpha v, accuracy pp [95% CI], pooled | MATH | LCB")
    for x, v in S["vs_sv"].items():
        c = v["all"]["correct"]
        print(f"  {x:22s} {pp(c['mean'])} [{pp(c['lo'])}, {pp(c['hi'])}] | {pp(v['math']['correct']['mean'])} | {pp(v['lcb']['correct']['mean'])}")
    c = S["vs_control"]["control_nothink"]["all"]["correct"]
    print(f"  thinking off - on, accuracy: {pp(c['mean'])} [{pp(c['lo'])}, {pp(c['hi'])}]")

    N = next((d for d in (Q / "natural_vs_short", Q.parent / "natural") if (d / "test_summary.json").exists()), None)
    if N:
        NS = json.loads((N / "test_summary.json").read_text())
        print(f"§4 steering data, short pairs vs natural traces ({NS['n_prompts']} prompts; arms {json.dumps(NS['arms'])}):")
        for k, v in NS["diffs"].items():
            c, lp = v["all"]["correct"], v["all"]["loop"]
            print(f"  {k:24s} accuracy {pp(c['mean'])} [{pp(c['lo'])}, {pp(c['hi'])}]  loop {pp(lp['mean'])} [{pp(lp['lo'])}, {pp(lp['hi'])}]")
        for f in sorted(N.glob("dev_selection_*.json")):
            d = json.loads(f.read_text())
            print(f"  reduced-dev loop rate, {f.stem[len('dev_selection_'):]}:",
                  ", ".join(f"{x}: {pct(r['reduced']['loop'])}" for x, r in d["arms"].items()), "| chosen", d["chosen"])


if __name__ == "__main__":
    main()
