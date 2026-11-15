#!/usr/bin/env python3
"""Blog Section 4 figure: steering data from short synthetic pairs vs the model's own (natural) traces, Qwen3-1.7B
greedy, MLP-output edits at blocks 15-18, the Section 5 test prompts (500 MATH level 5, 500 LiveCodeBench).

Rows: each data source with s = alpha v and with learned unlikelihood s, every setting chosen on the same reduced dev
set. Columns: change in loop rate and in accuracy vs the unedited model, in percentage points, with 95% stratified-bootstrap
intervals (results/qwen3/natural_vs_short/test_summary.json, from scripts/steer/natural_test_summary.py).

  build_natural_vs_short.py --summary results/qwen3/natural_vs_short/test_summary.json --out qwen3_natural_vs_short.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401

ASTRA, TRAINED = "#508dd3", "#26a996"  # method colors; shape encodes data source
ROWS = []  # (label, summary key, colour, marker), filled from the summary by rows()


def rows(d):
    """s = alpha v with each source's dev-chosen alpha, then the learned s."""
    def sv(arm):
        al = arm.split("_a")[-1]
        return "s = v" if al == "1" else f"s = {al}v"
    short_sv = {r["sv"] for r in d["arms"]["short"]}
    assert len(short_sv) == 1, short_sv  # s = alpha v is the same edit in every run
    return [(f"Short pairs, {sv(short_sv.pop())}", "short_sv - control", ASTRA, "o"),
            (f"Natural traces, {sv(d['arms']['natural']['sv'])}", "natural_sv - control", ASTRA, "D"),
            ("Short pairs, learned s", "short_ul - control", TRAINED, "o"),
            ("Natural traces, learned s", "natural_ul - control", TRAINED, "D")]

Y = (3.4, 2.6, 1.0, 0.2)  # two families, a gap between them
BENCH = (("math", "MATH level 5 (500 prompts)"), ("lcb", "LiveCodeBench (500 prompts)"))
COLS = (("loop", "Δ loop rate (percentage points) ↓", (-42, 6)), ("correct", "Δ accuracy (percentage points) ↑", (-9, 9)))


def panel(ax, d, grp, metric, xlim, first_col):
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.4, 4.0)
    ax.grid(axis="x", color="#e6e4de", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#7a786f", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    for (label, key, col, mk), y in zip(ROWS, Y):
        s = d["diffs"][key][grp][metric]
        m, lo, hi = 100 * s["mean"], 100 * s["lo"], 100 * s["hi"]
        ax.plot([lo, hi], [y, y], color=col, linewidth=1.5, solid_capstyle="round", zorder=2)
        ax.scatter([m], [y], color=col, marker=mk, s=70, edgecolors=col, linewidths=1.2, zorder=3)
        ax.annotate(f"{m:+.1f}".replace("-", "\u2212"), (hi, y), xytext=(7, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=11.5, color="#1f1e1c")
    ax.set_yticks(Y, [r[0] for r in ROWS] if first_col else [])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.0f}".replace("-", "\u2212") if x else "0"))
    ax.tick_params(axis="both", length=0, pad=8)
    ax.tick_params(axis="y", labelsize=12, colors="#1f1e1c")
    ax.tick_params(axis="x", labelsize=11, colors="#55534d")
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    d = json.loads(a.summary.read_text())
    ROWS[:] = rows(d)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 12,
                         "text.color": "#1f1e1c", "svg.fonttype": "none", "svg.hashsalt": "qwen3-natural-vs-short"}):
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.7), gridspec_kw={"width_ratios": (1.5, 1)})
        fig.subplots_adjust(left=0.215, right=0.965, top=0.775, bottom=0.07, wspace=0.12, hspace=0.45)
        fig.text(0.02, 0.965, "Steering data: short synthetic pairs vs the model's own traces", fontsize=16,
                 fontweight="bold", ha="left")
        fig.text(0.02, 0.928, "Change vs the unedited model in percentage points, with 95% intervals. Qwen3-1.7B, greedy, "
                 "blocks 15–18; 32 + 32 examples per source.", fontsize=11.5, color="#55534d", ha="left")
        for r, (grp, btitle) in enumerate(BENCH):
            for c, (metric, ctitle, xlim) in enumerate(COLS):
                ax = axes[r][c]
                panel(ax, d, grp, metric, xlim, c == 0)
                ax.set_title(ctitle, loc="left", fontsize=12, color="#55534d", pad=8)
            fig.text(0.02, axes[r][0].get_position().y1 + 0.052, btitle, fontsize=13.5, fontweight="bold", ha="left")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.out, format="svg", facecolor="white",
                    metadata={"Date": None, "Title": "Short synthetic pairs vs natural traces as steering data"})
        plt.close(fig)
    print(a.out)


if __name__ == "__main__":
    main()
