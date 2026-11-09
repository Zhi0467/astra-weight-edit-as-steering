#!/usr/bin/env python3
"""Print every number the blog post quotes, computed from the summaries, so a rerun can be checked line by line.

  blog_numbers.py --qwen3 results/qwen3 --astra results/astra_regime            our numbers (shipped)
  blog_numbers.py --qwen3 $WORK/qwen3/exp1 --astra $WORK/astra_regime           your rerun

The "capped but not loop-flagged" count needs per-problem outputs, so it is printed only when <astra>/gen exists
(a rerun). The two §4 tables come from an earlier round that this repository does not reproduce.
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
                       ("Astra's top-64 edit", T, "astra_top64_a6")):
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
    if gen.exists():
        rows = [json.loads(l) for s in SPLITS for l in open(gen / s / arm("ul", 32) / "results.jsonl")]
        capped = [r for r in rows if r["finish"] == "length"]
        n_free = sum(not r["loop"] for r in capped)
        print(f"  capped {arm('ul', 32)} responses: {len(capped)}, of which not loop-flagged: {n_free}")

    R = A / "test_summary_resid.json"  # run/astra_controls.sh
    if R.exists():
        R = json.loads(R.read_text())
        print("§4 MLP write vs residual stream, N = 32: accuracy / loop / cap hit")
        for x in ("sv_n32_a4", "resid_sv_n32_a4", "ul_n32_b0.1", "resid_ul_n32_b0.1"):
            r = m(R, x)
            print(f"  {x:20s} {pct(r['accuracy'])} / {pct(r['loop'])} / {pct(r['cap_hit'])}")
        d = R["paired"]["resid_ul_n32_b0.1 - ul_n32_b0.1"]["pooled"]
        print(f"  residual - write, learned ul: accuracy {ci(d['accuracy'])}, loop {pp(d['loop']['diff'])}")
    RA, RD = A / "test_summary_resid_alpha.json", A / "resid_alpha_dev.json"  # run/astra_resid_alpha.sh
    if RA.exists() and RD.exists():
        RA, RD = json.loads(RA.read_text()), json.loads(RD.read_text())
        x = RD["chosen"]
        print("§4 residual s = alpha v, dev accuracy:", ", ".join(f"{k.split('_a')[-1]}: {pct(v['accuracy'])}" for k, v in RD["per_arm"].items()),
              "-> chosen", x)
        r = m(RA, x)
        print(f"  {x} on test: {pct(r['accuracy'])} / {pct(r['loop'])} / {pct(r['cap_hit'])}")
        for ref in ("sv_n32_a4", "ul_n32_b0.1"):
            d = RA["paired"][f"{x} - {ref}"]["pooled"]
            print(f"  {x} - {ref} (write): accuracy {ci(d['accuracy'])}, loop {ci(d['loop'])}")

    D = json.loads((A / "diag_samplers_validation_1.json").read_text())
    print("§7 dev sampler diagnostic (validation_1): accuracy under Astra's sampler -> greedy -> no presence penalty")
    for x, r in D.items():
        print(f"  {x:18s} {pct(r['astra']['accuracy'])} -> {pct(r['greedy']['accuracy'])} -> {pct(r['nopp']['accuracy'])}")
    C = json.loads((A / "astra_check_n4.json").read_text())
    print("Endnote: |cos(our four-pair v, Astra's u)| per layer:", ", ".join(f"{l}: {c['abs_cos_v_u']:.6f}" for l, c in C.items()))

    S = json.loads((Q / "exp1_test_summary.json").read_text())
    QS = json.loads((Q / "exp1_selection.json").read_text())
    print("§3/§4 Qwen3-1.7B stopping updates:", QS["stop_update"])
    print(f"§5 Qwen3-1.7B (greedy, {S['n_prompts']} prompts): learned - s = alpha v, accuracy pp [95% CI], pooled | MATH | LCB")
    for x, v in S["vs_sv"].items():
        c = v["all"]["correct"]
        print(f"  {x:22s} {pp(c['mean'])} [{pp(c['lo'])}, {pp(c['hi'])}] | {pp(v['math']['correct']['mean'])} | {pp(v['lcb']['correct']['mean'])}")
    c = S["vs_control"]["control_nothink"]["all"]["correct"]
    print(f"  thinking off - on, accuracy: {pp(c['mean'])} [{pp(c['lo'])}, {pp(c['hi'])}]")


if __name__ == "__main__":
    main()
