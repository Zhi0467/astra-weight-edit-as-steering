#!/usr/bin/env python3
"""Prompt pools of the Qwen3-1.7B experiments: MATH level 5 rows, one JSONL per source split.

  train_source    MATH train, Level 5 (2,304 rows)    -> DATA_DIR/math_level5/train_source.jsonl
  heldout_source  MATH test,  Level 5 (1,324 rows)    -> DATA_DIR/math_level5/heldout_source.jsonl

Source: SuperSecureHuman/competition_math_hf_dataset at a pinned revision, rows kept in dataset order. A row's line
index is its sample_id everywhere else (splits, loop labels, summaries). `_contract_split` marks a fixed 10% of
train_source as "validation" (data/qwen3/math_level5_train_validation_ids.json); it is metadata only and is not used
by any experiment. The output is byte-identical to the files our runs read (checked with --check).

LiveCodeBench needs no preparation: the collector downloads livecodebench/code_generation_lite release files itself
(train_source = releases v1-v5, 880 problems; heldout_source = v6, 175), sorted by question_id.

  prepare_datasets.py --out DATA_DIR [--check SHA256SUMS]
"""

import argparse
import hashlib
import json
from pathlib import Path

REPO = "SuperSecureHuman/competition_math_hf_dataset"
REVISION = "fbc86cb47a39a9465b9235622014b33f8ca9bb7f"
HERE = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=str(HERE / "data" / "qwen3" / "SHA256SUMS"))
    a = ap.parse_args()
    import datasets

    val = set(json.loads((HERE / "data" / "qwen3" / "math_level5_train_validation_ids.json").read_text()))
    out = Path(a.out) / "math_level5"
    out.mkdir(parents=True, exist_ok=True)
    for split, role in (("train", "train_source"), ("test", "heldout_source")):
        ds = datasets.load_dataset(REPO, split=split, revision=REVISION)
        rows = [r for r in ds if r["level"] == "Level 5"]
        with open(out / f"{role}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                cs = "test" if role == "heldout_source" else ("validation" if i in val else "train")
                rec = {"_contract_dataset_key": "math_level5", "_contract_source_index": i, "_contract_source_role": role,
                       "_contract_split": cs, "_source_sample_id": i, "level": r["level"], "problem": r["problem"],
                       "solution": r["solution"], "type": r["type"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(role, len(rows))
    if a.check and Path(a.check).exists():
        for line in Path(a.check).read_text().splitlines():
            if not line.strip().startswith(tuple("0123456789abcdef")):
                continue
            digest, name = line.split()
            got = hashlib.sha256((Path(a.out) / name).read_bytes()).hexdigest()
            print("ok" if got == digest else "MISMATCH", name)
            assert got == digest, name


if __name__ == "__main__":
    main()
