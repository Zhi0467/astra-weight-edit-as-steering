#!/usr/bin/env python3
"""Experiment-1 test summary (Qwen3-1.7B greedy): paired per-prompt differences of the dev-selected arms against the
unedited control and against s = alpha v at the same pair count, on the experiment-1/2 test split.

Test split (loop_test_split.py): per dataset 500 prompts = every prompt that looped in any of 4 saved greedy draws, plus
a seeded sample of non-looping prompts; train_source (official train) and heldout_source (official test) parts. The set
is loop-enriched by design, so results are reported per loop stratum; "all" is the plain mean over the 1,000 test
prompts (not a natural-distribution estimate).

Metrics per prompt (one greedy rollout): correct (MATH grading.correct, LiveCodeBench grading.passed), loop (loop_flag),
closed (</think> in the completion), cap (max_length_hit). Intervals: 95% percentile bootstrap over prompts, resampled
within source x dataset x loop-stratum cells (10,000 resamples, seed 20261106).

  exp1_test_summary.py --gen GEN_TEST_DIR --graded GRADED_TEST_DIR --split TEST_PROMPTS_DIR --json OUT.json
GEN_TEST_DIR/<src>/math_level5/<arm>/*.jsonl.gz, GRADED_TEST_DIR/<src>/livecodebench/<arm>/*.jsonl.gz.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

SRCS = ("train_source", "heldout_source")
DSS = ("math_level5", "livecodebench")
LABELS = Path(__file__).resolve().parents[2] / "data" / "qwen3" / "loop_labels.jsonl"
METRICS = ("correct", "loop", "closed", "cap")
B = 10_000
GROUPS = {  # name -> predicate on (src, ds, looping)
    "all": lambda s, d, l: True,
    "loop": lambda s, d, l: l,
    "nonloop": lambda s, d, l: not l,
    "math": lambda s, d, l: d == "math_level5",
    "lcb": lambda s, d, l: d == "livecodebench",
    "math_loop": lambda s, d, l: d == "math_level5" and l,
    "math_nonloop": lambda s, d, l: d == "math_level5" and not l,
    "lcb_loop": lambda s, d, l: d == "livecodebench" and l,
    "lcb_nonloop": lambda s, d, l: d == "livecodebench" and not l,
    "official_test": lambda s, d, l: s == "heldout_source",
    "official_test_loop": lambda s, d, l: s == "heldout_source" and l,
    "official_test_nonloop": lambda s, d, l: s == "heldout_source" and not l,
}


def bundle(d, flag):
    if not Path(f"{d}/{flag}").exists():  # unfinished or ungraded run: treat as missing
        return None
    fs = [f for f in glob.glob(f"{d}/*.jsonl.gz") if "preexisting" not in f]
    return fs[0] if len(fs) == 1 else None


def key(r):
    """prompt identity: LiveCodeBench question_id (test runs are re-indexed); MATH sample_id (question_id is None, and
    the MATH test subsets keep the original sample ids)."""
    return r["question_id"] if r.get("question_id") is not None else r["sample_id"]


def load(f):
    out = {}
    for line in gzip.open(f, "rt"):
        r = json.loads(line)
        x = r["rollouts"][0]
        g = x.get("grading") or {}
        out[key(r)] = (r["sample_id"], {
            "correct": float(bool(g.get("correct", 0) or g.get("passed", False))),
            "loop": float(bool(x["loop_flag"])),
            "closed": float("</think>" in x["completion_text"]),
            "cap": float(bool(x["max_length_hit"]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--graded", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", default="control,sv_n4_a1,clas_n4_b0.1_s8,ul_n4_b0.1_s14,contrast_n4_b0.1_s12,"
                                      "sv_n32_a2,clas_n32_b0_s14,ul_n32_b0_s14,contrast_n32_b0.1_s12")
    ap.add_argument("--json", required=True)
    ap.add_argument("--labels", default=str(LABELS), help="per-prompt loop labels (scripts/data/loop_labels.py)")
    ap.add_argument("--subset-arms", default="control_nothink",
                    help="arms whose runs hold more prompts than the test selection; extra prompts are dropped. "
                         "control_nothink: the LiveCodeBench exclusion list matches rendered thinking-on prompt text, so "
                         "its thinking-off LCB runs generated every prompt of the source split")
    a = ap.parse_args()
    arms = a.arms.split(",")
    subset = set(a.subset_arms.split(",")) if a.subset_arms else set()

    # loop label per prompt, keyed by question_id (LiveCodeBench test runs are re-indexed after the exclusion step)
    label = {}
    for ds in DSS:
        sel = json.loads((Path(a.split) / f"{ds}_test_selection.json").read_text())
        for src in SRCS:
            qid = {}
            for l in open(a.labels):
                r = json.loads(l)
                if r["dataset"] == ds and r["split"] == src:
                    qid[r["sample_id"]] = key(r)
            for k, lab in (("loop_prompt_ids", True), ("nonloop_prompt_ids", False)):
                for i in sel[src][k]:
                    label[(src, ds, qid[i])] = lab

    data, missing = {}, []  # arm -> {(src, ds, qid): metrics}
    for arm in arms:
        data[arm] = {}
        for src in SRCS:
            for ds in DSS:
                root = a.gen if ds == "math_level5" else a.graded
                f = bundle(f"{root}/{src}/{ds}/{arm}", "DONE" if ds == "math_level5" else "GRADED")
                if f is None:
                    missing.append(f"{src}/{ds}/{arm}")
                    continue
                for q, (_, m) in load(f).items():
                    if (src, ds, q) not in label and arm in subset:
                        continue
                    if (src, ds, q) not in label:
                        raise SystemExit(f"prompt {src}/{ds}/{q} of {arm} not in the test selection")
                    data[arm][(src, ds, q)] = m
                n = sum(1 for k in data[arm] if k[:2] == (src, ds))
                want = sum(1 for k in label if k[:2] == (src, ds))
                if n != want:
                    raise SystemExit(f"{src}/{ds}/{arm}: {n} prompts, selection has {want}")
    keys = sorted(label)
    rng = np.random.default_rng(20261106)
    cells = {}
    for i, k in enumerate(keys):
        cells.setdefault((k[0], k[1], label[k]), []).append(i)
    boot_idx = {c: rng.integers(0, len(ix), size=(B, len(ix))) for c, ix in cells.items()}

    def arr(arm, metric):
        return np.array([data[arm][k][metric] if k in data[arm] else np.nan for k in keys])

    def stat(x, grp):
        """mean of x over the group and its stratified bootstrap 95% interval (None if any value is missing)."""
        sel = [c for c in cells if GROUPS[grp](*c)]
        ix = np.concatenate([np.array(cells[c]) for c in sel])
        if np.isnan(x[ix]).any():
            return None
        tot = np.zeros(B)
        for c in sel:
            tot += x[np.array(cells[c])][boot_idx[c]].sum(1)
        bs = tot / len(ix)
        return {"mean": float(x[ix].mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                "n": int(len(ix))}

    out = {"n_prompts": len(keys), "missing": missing, "levels": {}, "vs_control": {}, "vs_sv": {}}
    for arm in arms:
        out["levels"][arm] = {g: {m: stat(arr(arm, m), g) for m in METRICS} for g in GROUPS}
    sv = {n: next((x for x in arms if x.startswith(f"sv_n{n}_")), None) for n in (4, 32)}
    for arm in arms:
        if arm == "control":
            continue
        out["vs_control"][arm] = {g: {m: stat(arr(arm, m) - arr("control", m), g) for m in METRICS} for g in GROUPS}
        if arm.startswith("control_"):  # reference rows (control_nothink: unedited, thinking off) have no pair count
            continue
        n = int(arm.split("_n")[1].split("_")[0])
        if not arm.startswith("sv_") and sv[n]:
            out["vs_sv"][arm] = {g: {m: stat(arr(arm, m) - arr(sv[n], m), g) for m in METRICS} for g in GROUPS}
    Path(a.json).write_text(json.dumps(out, indent=1))

    def fmt(s, pct=True):
        if s is None:
            return "   (missing)    "
        f = 100 if pct else 1
        return f"{s['mean'] * f:+6.1f} [{s['lo'] * f:+5.1f},{s['hi'] * f:+5.1f}]"

    print(f"{len(keys)} test prompts; missing runs: {missing}")
    for g in ("all", "loop", "nonloop", "official_test"):
        print(f"\n== {g} ==   accuracy / loop rate / closed thinking (levels %, then differences in points, 95% CI)")
        for arm in arms:
            L = out["levels"][arm][g]
            lv = " / ".join("  -  " if L[m] is None else f"{100 * L[m]['mean']:5.1f}" for m in ("correct", "loop", "closed"))
            line = f"{arm:22s} {lv}"
            if arm in out["vs_control"]:
                line += "  | vs control acc " + fmt(out["vs_control"][arm][g]["correct"]) + " loop " + fmt(out["vs_control"][arm][g]["loop"])
            if arm in out["vs_sv"]:
                line += "  | vs sv acc " + fmt(out["vs_sv"][arm][g]["correct"])
            print(line)


if __name__ == "__main__":
    main()
