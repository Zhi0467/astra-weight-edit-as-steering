#!/usr/bin/env python3
"""Download the materials of Astra's loop edit (github.com/RikiyaT/lm_understand-logs, exp8_math) at a pinned commit.

Files, with their local names (the layout astra_regime.py, loop_s_learn.py and loop_v_constructions.py read):
  practice/{gsm8k_train,math_train}.jsonl  practice problems with Astra's ids and extracted gold answers
  splits.json, fresh_splits.json           problem sets (validation_1 ... final_3)
  edit/{edit.npz,manifest.json}            the submitted top-64, alpha-6 edit (layers 15 and 19 down_proj)
  final_confirmation.csv                   Astra's per-problem results on the 1,152 confirmation problems
  fwd4_request.json                        the four short productive/repetitive pairs (first 4 of our 32 pairs)
  transfer_probe_request.json              six more pairs (a teacher-forced check only)
  mathqa.py, evaluate.py, profiles.py, eval_prompt_format.txt   their grading and prompt code, for reference
Every file is checked against the SHA-256 of the copy our runs used.

  fetch_astra.py --out ASTRA_DIR
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
URL = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{commit}/{path}"
FILES = [  # (local name, path in the upstream repo, sha256)
    ("practice/gsm8k_train.jsonl", "exp8_math/practice/gsm8k_train.jsonl", "06d0dd560a908c2c2883bfb7d27ab55a87b213ff0d625f5cd734836aeac9a3e5"),
    ("practice/math_train.jsonl", "exp8_math/practice/math_train.jsonl", "abe7982b4d8c98394fe185fb152568a2c784a4157a75fe42b9c01857e71f0c6e"),
    ("practice/stats.json", "exp8_math/practice/stats.json", "34a3be5db6095e0968bdc5555faa1c4149898e4ea8d6c4c6578975703a208dd6"),
    ("edit/edit.npz", "exp8_math/runs/p50c/edit/edit.npz", "e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7"),
    ("edit/manifest.json", "exp8_math/runs/p50c/edit/manifest.json", "83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225"),
    ("splits.json", "exp8_math/runs/p50c/work_notes/splits.json", "b88db2a521c2497ebce42559c264be7311d184693a2055559931f57255190ac1"),
    ("fresh_splits.json", "exp8_math/runs/p50c/work_notes/fresh_splits.json", "b08197c17a5001caffa10957dddf253921f7b2bae312114dfa312828c5517274"),
    ("fwd4_request.json", "exp8_math/runs/p50c/requests/fwd_4.json.claimed", "79ca5d87116d364b2ae915191d7edee7d95ca5e5758ec01b7dab4aa95fd87874"),
    ("transfer_probe_request.json", "exp8_math/runs/p50c/work_notes/transfer_probe_request.json", "d940f18c3f2c2e530bb1f3b0b0e1a7391e675c802a68ddd1943b8d8c57847413"),
    ("final_confirmation.csv", "exp8_math/runs/p50c/work_notes/final_confirmation.csv", "f3f4a1933ff7bcc3a9cc1e0ccf3dc7ce4e9c70ec4ef87052e6e20bc2cd96030f"),
    ("eval_prompt_format.txt", "exp8_math/runs/p50c/work_notes/eval_prompt_format.txt", "6aff9ff9c1f67cfa071fa85db87d3020acece233e90fcc3d80218587cb9066a4"),
    ("mathqa.py", "exp8_math/code/mathqa.py", "f0d576a1adc625dede7bff749b3bf8c9a5a8f6bae43d51e2d0ee36d694e89843"),
    ("evaluate.py", "exp8_math/code/evaluate.py", "c93f477c02b3ae0d6b37abe3afad8bb7353e273322ff963eec7ff2270012035d"),
    ("profiles.py", "exp8_math/code/profiles.py", "ba4be365e44c3e165b9c3982af9dbcd8ec94f7fd0430e5fd9de1d57f249a4ae4"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    for name, path, digest in FILES:
        dest = out / name
        if not (dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(URL.format(commit=COMMIT, path=path)) as r:
                dest.write_bytes(r.read())
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert got == digest, f"{name}: sha256 {got} != {digest}"
        print("ok", name)
    (out / "SOURCE_COMMIT").write_text(COMMIT + "\ngithub.com/RikiyaT/lm_understand-logs exp8_math\n")


if __name__ == "__main__":
    main()
