#!/usr/bin/env python3
"""Dev prompts for choosing alpha / beta (experiments 1 and 2): 64 looping + 64 non-looping prompts per dataset.

Source: per-prompt loop labels of Qwen3-1.7B from 4 greedy draws per prompt, thinking on (data/qwen3/loop_labels.jsonl,
written by scripts/data/loop_labels.py). train_source is
built from the official training splits (MATH train, LiveCodeBench v1-v5); heldout_source is the official test splits
(MATH test level 5, LiveCodeBench v6) and is kept for the final test. A prompt "loops" if any draw is loop-flagged.
Seeded sample of 64 of each kind.

  loop_dev_split.py --labels data/qwen3/loop_labels.jsonl --data DATA_DIR --out OUT_DIR
DATA_DIR is the output of scripts/data/prepare_datasets.py. Writes OUT_DIR/math_level5_dev_subset.jsonl (dataset rows
of the chosen prompts), livecodebench_dev_exclude.jsonl (sample ids of every other train_source prompt, for
collect_model_stats.py --exclude-prompt-jsonl) and {ds}_dev_selection.json.
"""

import argparse
import json
import random
from pathlib import Path

K = 64


def load_labels(path, ds, sp):
    """Label rows of one dataset and split, in the reference runs' order: {"sample_id", "rollouts": [{"loop_flag"}]}."""
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r["dataset"] == ds and r["split"] == sp:
            rows.append({"sample_id": r["sample_id"], "rollouts": [{"loop_flag": f} for f in r["loop_draws"]]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = a.data
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {}
    for ds in ("math_level5", "livecodebench"):
        rows = load_labels(a.labels, ds, "train_source")
        loop = [r for r in rows if any(x["loop_flag"] for x in r["rollouts"])]
        non = [r for r in rows if not any(x["loop_flag"] for x in r["rollouts"])]
        rng = random.Random(f"20261106-{ds}-dev")
        pl, pn = rng.sample(loop, min(K, len(loop))), rng.sample(non, K)
        keep = {r["sample_id"] for r in pl + pn}
        if ds == "math_level5":
            lines = Path(f"{D}/{ds}/train_source.jsonl").read_text().splitlines()
            assert len(lines) == len(rows)
            assert [r["sample_id"] for r in rows] == list(range(len(rows)))  # sample_id is the dataset line index
            (out / f"{ds}_dev_subset.jsonl").write_text("".join(lines[i] + "\n" for i in sorted(keep)))
        else:
            with open(out / f"{ds}_dev_exclude.jsonl", "w") as f:
                for r in rows:
                    if r["sample_id"] not in keep:
                        f.write(json.dumps({"sample_id": r["sample_id"]}) + "\n")
        (out / f"{ds}_dev_selection.json").write_text(json.dumps({"loop_prompt_ids": sorted(r["sample_id"] for r in pl),
                                                                   "nonloop_prompt_ids": sorted(r["sample_id"] for r in pn)}))
        summ[ds] = {"source_prompts": len(rows), "looping": len(loop), "picked_loop": len(pl), "picked_nonloop": len(pn)}
    (out / "dev_split_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
