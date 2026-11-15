#!/usr/bin/env python3
"""Test prompts for the Qwen3-1.7B experiment: 500 prompts per dataset.

Pool: every prompt of both splits, labelled from 4 greedy Qwen3-1.7B draws per prompt, thinking on
(data/qwen3/loop_labels.jsonl),
train_source (official train: MATH train+val level 5, LiveCodeBench v1-v5) and heldout_source (official test: MATH test
level 5, LiveCodeBench v6), minus the dev prompts of loop_dev_split.py. A prompt "loops" if any saved draw is
loop-flagged. All looping prompts go in, then a seeded uniform sample of non-looping prompts fills the set to 500.
Each prompt keeps its split, so results can also be reported on the official-test part alone.

  loop_test_split.py --labels data/qwen3/loop_labels.jsonl --data DATA_DIR --dev DEV_DIR --out OUT_DIR
Writes per dataset and split: {ds}_{split}_test_subset.jsonl (MATH dataset rows) or {ds}_{split}_test_exclude.jsonl
(LiveCodeBench sample ids of every other prompt in that split, for --exclude-prompt-jsonl), plus
{ds}_test_selection.json and test_split_summary.json. Generation runs once per split (LCB releases differ).
"""

import argparse
import json
import random
from pathlib import Path

from loop_dev_split import load_labels

SIZE = 500
SPLITS = ("train_source", "heldout_source")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = a.data
    dev, out = Path(a.dev), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {}
    for ds in ("math_level5", "livecodebench"):
        dsel = json.loads((dev / f"{ds}_dev_selection.json").read_text())
        dev_ids = {("train_source", i) for i in dsel["loop_prompt_ids"] + dsel["nonloop_prompt_ids"]}
        rows = {sp: load_labels(a.labels, ds, sp) for sp in SPLITS}
        pool = [(sp, r) for sp in SPLITS for r in rows[sp] if (sp, r["sample_id"]) not in dev_ids]
        loop = [(sp, r["sample_id"]) for sp, r in pool if any(x["loop_flag"] for x in r["rollouts"])]
        non = [(sp, r["sample_id"]) for sp, r in pool if not any(x["loop_flag"] for x in r["rollouts"])]
        rng = random.Random(f"20261106-{ds}-test")
        pick = rng.sample(non, SIZE - len(loop))
        keep = set(loop) | set(pick)
        assert not keep & dev_ids and len(keep) == SIZE
        for sp in SPLITS:
            ids = {i for s, i in keep if s == sp}
            if ds == "math_level5":
                lines = Path(f"{D}/{ds}/{sp}.jsonl").read_text().splitlines()
                assert len(lines) == len(rows[sp])
                assert [r["sample_id"] for r in rows[sp]] == list(range(len(rows[sp])))  # sample_id = dataset line index
                (out / f"{ds}_{sp}_test_subset.jsonl").write_text("".join(lines[i] + "\n" for i in sorted(ids)))
            else:
                with open(out / f"{ds}_{sp}_test_exclude.jsonl", "w") as f:
                    for r in rows[sp]:
                        if r["sample_id"] not in ids:
                            f.write(json.dumps({"sample_id": r["sample_id"]}) + "\n")
        sel = {sp: {"loop_prompt_ids": sorted(i for s, i in loop if s == sp),
                    "nonloop_prompt_ids": sorted(i for s, i in pick if s == sp)} for sp in SPLITS}
        (out / f"{ds}_test_selection.json").write_text(json.dumps(sel))
        summ[ds] = {sp: {"pool": sum(1 for s, _ in pool if s == sp), "loop": len(sel[sp]["loop_prompt_ids"]),
                         "nonloop": len(sel[sp]["nonloop_prompt_ids"])} for sp in SPLITS}
        summ[ds]["total"] = SIZE
    (out / "test_split_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
