#!/usr/bin/env python3
"""Prepare figure summaries from one fresh run, never from released results."""
import argparse
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/steer"))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))


def astra(work):
    import astra_regime_summary as t
    w = work / "astra_regime"
    selected = read(w / "dev_selection.json")["chosen_per_family_and_n"]
    selected.update(read(w / "dev_selection_clasd.json")["chosen_per_family_and_n"])
    aliases = {"control": "control", "control_nothink": "control_nothink"}
    for n in (4, 32):
        sv = selected[f"sv_n{n}"]
        aliases[sv] = sv
        for fam in ("ul", "clas", "clasd"):
            aliases[f"{fam}_n{n}"] = selected[f"{fam}_n{n}"]
    # Check exact prompt alignment before calling the existing paired estimator.
    for split in t.CONF:
        rows = {name: t.load(w / "gen", split, arm) for name, arm in aliases.items()}
        if any(v is None for v in rows.values()):
            raise ValueError(f"missing Astra test outputs in {split}")
        keys = set(rows["control"])
        if any(set(v) != keys for v in rows.values()):
            raise ValueError(f"Astra prompt mismatch in {split}")
    dest = w / "test_summary_with_clasd.json"
    refs = ["control"] + [selected[f"sv_n{n}"] for n in (4, 32)]
    t.cmd_test(SimpleNamespace(gen=w / "gen", arms=",".join(aliases.values()),
                              refs=",".join(refs), boot=10000, out=str(dest)))
    raw = read(dest)
    reverse = {arm: name for name, arm in aliases.items()}
    raw["per_arm"] = {reverse[k]: v for k, v in raw["per_arm"].items()}
    raw["paired"] = {" - ".join(reverse[k] for k in key.split(" - ")): v
                     for key, v in raw["paired"].items()}
    write(dest, raw)
    write(w / "test_summary_nothink.json", {
        "per_arm": {"control_nothink": raw["per_arm"]["control_nothink"]},
        "paired": {"control_nothink - control": raw["paired"]["control_nothink - control"]}})
    return dest


def qwen(work):
    import exp1_test_summary as t
    w = work / "qwen3"
    arms = read(w / "exp1/exp1_selection.json")["test_arms"]
    arms += read(w / "exp1/selection_clasd.json")["test_arms"]
    arms = list(dict.fromkeys(["control_nothink", "control"] + arms))
    labels = t.read_labels(ROOT / "data/qwen3/splits/test12")
    gen = w / "runs/test"
    data = {}
    for arm in arms:
        records = t.read_arm(gen, gen, arm, labels, subset=arm == "control_nothink")
        for src in t.SRCS:
            for ds in t.DSS:
                bundle = t.bundle(gen / src / ds / arm, "DONE" if ds == "math_level5" else "GRADED")
                with gzip.open(bundle, "rt") as f:
                    for line in f:
                        row = json.loads(line)
                        key = (src, ds, t.key(row))
                        if key not in records:
                            continue
                        rollout = row["rollouts"][0]
                        count = rollout.get("completion_token_count")
                        if count is None:
                            count = len(rollout["completion_token_ids"])
                        records[key]["tokens"] = count
        name = arm if arm.startswith(("control", "sv_")) else "_".join(arm.split("_")[:2])
        if name in data:
            raise ValueError(f"multiple selected arms for {name}")
        data[name] = records
    t.METRICS = ("correct", "loop", "closed", "cap", "tokens")
    dest = w / "exp1/exp1_test_summary_with_clasd.json"
    write(dest, t.summarize(data, labels))
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--setting", choices=["astra", "qwen3"], required=True)
    a = p.parse_args()
    print((astra if a.setting == "astra" else qwen)(a.work.resolve()))


if __name__ == "__main__":
    main()
