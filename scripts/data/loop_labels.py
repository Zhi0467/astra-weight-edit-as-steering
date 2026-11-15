#!/usr/bin/env python3
"""Per-prompt loop labels of Qwen3-1.7B from the reference greedy runs (4 greedy draws per prompt, thinking on).

The dev and test splits of the Qwen3-1.7B experiments are stratified by these labels: a prompt "loops" if any of its
4 draws is loop-flagged (some 30-token sequence occurs at least 20 times in the completion). The labels shipped in
data/qwen3/loop_labels.jsonl were written by this script from our own reference runs; run/qwen3_reference_draws.sh
regenerates such runs, and this script turns them into a labels file.

  loop_labels.py --bundles DIR --out data/qwen3/loop_labels.jsonl
DIR holds <dataset>/<split>/<dataset>__<split>__thinking_on.jsonl.gz (collect_model_stats.py output). Rows keep the
bundle order, which the seeded split scripts depend on.
"""

import argparse
import gzip
import json
from pathlib import Path

DATASETS = ("math_level5", "livecodebench")
SPLITS = ("train_source", "heldout_source")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for ds in DATASETS:
            for sp in SPLITS:
                p = Path(a.bundles) / ds / sp / f"{ds}__{sp}__thinking_on.jsonl.gz"
                for line in gzip.open(p, "rt"):
                    r = json.loads(line)
                    f.write(json.dumps({"dataset": ds, "split": sp, "sample_id": r["sample_id"],
                                        "question_id": r.get("question_id"),
                                        "loop_draws": [bool(x["loop_flag"]) for x in r["rollouts"]]}) + "\n")


if __name__ == "__main__":
    main()
