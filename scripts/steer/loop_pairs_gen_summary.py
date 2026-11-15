#!/usr/bin/env python3
"""Summarize the in-loop direction generation runs (write and residual arms) against the unedited control.

  loop_pairs_gen_summary.py <out_dir>    # reads <out_dir>/gen/<dataset>/<arm>/*.jsonl.gz, writes gen_summary.json
Per arm and dataset, on the prompts both the arm and control finished: loop rate, accuracy (when graded), mean
completion tokens, and the paired difference from control with a 95% bootstrap interval over prompts (10,000
resamples, seed 0). LiveCodeBench runs graded on CPU are read from <out_dir>/graded/<dataset>/<arm> when present. Also the loop rate on the prompts that looped in the saved greedy bundle (test_prompts selection).
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path


def rows(path: Path) -> dict[int, dict]:
    out = {}
    for f in path.glob("*.jsonl.gz"):
        with gzip.open(f, "rt") as fh:
            lines = []
            try:  # a run still writing has an unterminated gzip stream; keep its complete rows
                for line in fh:
                    lines.append(line)
            except EOFError:
                lines = lines[:-1]
            for line in lines:
                r = json.loads(line)
                ro = r["rollouts"][0]
                g = ro.get("grading") or {}
                ok = g.get("correct", g.get("passed"))  # MATH: correct; LiveCodeBench: passed (after CPU grading)
                out[r["sample_id"]] = {"loop": int(ro["loop_flag"]), "tokens": ro["completion_token_count"],
                                       "correct": None if ok is None else int(ok)}
    return out


def boot(d: list[float], n: int = 10000) -> list[float]:
    rng = random.Random(0)
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [m[int(0.025 * n)], m[int(0.975 * n)]]


def main() -> None:
    base = Path(sys.argv[1])
    summary = {}
    for ds_dir in sorted((base / "gen").iterdir()):
        ds = ds_dir.name
        sel = json.loads((base / "test_prompts" / f"{ds}_selection.json").read_text())
        looped = set(sel["loop_prompt_ids"])
        pick = lambda arm: (base / "graded" / ds / arm) if (base / "graded" / ds / arm / "GRADED").exists() else ds_dir / arm
        ctrl = rows(pick("control"))
        summary[ds] = {}
        for arm_dir in sorted(ds_dir.iterdir()):
            arm = arm_dir.name
            r = rows(pick(arm))
            ids = sorted(set(r) & set(ctrl))
            if not ids:
                continue
            s = {"done": (arm_dir / "DONE").exists(), "n": len(ids)}
            for key in ("loop", "correct", "tokens"):
                if any(r[i][key] is None or ctrl[i][key] is None for i in ids):
                    continue
                s[key] = sum(r[i][key] for i in ids) / len(ids)
                if arm != "control":
                    d = [r[i][key] - ctrl[i][key] for i in ids]
                    s[f"d_{key}"] = sum(d) / len(d)
                    s[f"d_{key}_ci95"] = boot(d)
            lp = [i for i in ids if i in looped]
            s["loop_on_saved_loopers"] = sum(r[i]["loop"] for i in lp) / len(lp) if lp else None
            s["n_saved_loopers"] = len(lp)
            summary[ds][arm] = s
    (base / "gen_summary.json").write_text(json.dumps(summary, indent=1))
    for ds, arms in summary.items():
        print(f"\n{ds}")
        for arm, s in arms.items():
            f = lambda k: f"{100 * s[k]:.1f}" if k in s else "-"
            ci = lambda k: f"{100 * s[k]:+.1f} [{100 * s[k + '_ci95'][0]:+.1f},{100 * s[k + '_ci95'][1]:+.1f}]" if k in s else ""
            print(f"  {arm:28s} n={s['n']:3d}{'' if s['done'] else '*'} loop {f('loop'):>5} {ci('d_loop'):24s} acc {f('correct'):>5} {ci('d_correct'):24s} tok {s.get('tokens', 0) / 1000:.1f}k  saved-loopers {100 * (s['loop_on_saved_loopers'] or 0):.1f}")


if __name__ == "__main__":
    main()
