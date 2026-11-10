#!/usr/bin/env python3
"""Plot the earlier Qwen3 direction comparison reported in Section 4 of the blog.

These are the reported point estimates, not a recomputation: the earlier round's
rollouts are not shipped in this repository. Values are changes in percentage points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = ("Natural, first loop positions\nα = 2", "Natural, last loop positions\nα = 4",
          "Short synthetic\nα = 2")
# Each pair is (loop-rate change, accuracy change), in LABELS order.
BENCHMARKS = (
    ("MATH level 5 · 304 prompts", ((-3.6, 1.6), (-5.9, 0.0), (-14.5, -1.3))),
    ("LiveCodeBench · 420 prompts", ((-12.1, 1.9), (-5.2, -1.2), (-29.0, -1.7))),
)
METRICS = (("Loop rate (↓ better)", "#237a69", "o", 0.13),
           ("Accuracy (↑ better)", "#7156a5", "D", -0.13))


def draw_panel(ax, title, points):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=16)
    ax.set_xlim(-36, 7)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([-30, -20, -10, 0, 5], ["−30", "−20", "−10", "0", "+5"])
    ax.set_yticks([2, 1, 0], LABELS)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#6b7280", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    for metric, (label, color, marker, offset) in enumerate(METRICS):
        values = [point[metric] for point in points]
        positions = [2 - row + offset for row in range(len(points))]
        ax.scatter(values, positions, color=color, marker=marker, s=64, label=label, zorder=3)
        for value, y in zip(values, positions):
            ax.annotate(f"{value:+.1f}", (value, y), xytext=(-8 if metric == 0 else 8, 0),
                        textcoords="offset points", ha="right" if metric == 0 else "left",
                        va="center", fontsize=12, color=color)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.tick_params(axis="y", labelsize=12)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")


def write_plot(out):
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 12, "text.color": "#1f2328", "xtick.color": "#59636e",
                         "ytick.color": "#374151", "svg.fonttype": "none",
                         "svg.hashsalt": "qwen3-direction-comparison"}):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.255, right=0.975, top=0.63, bottom=0.18, wspace=0.17)
        fig.text(0.5, 0.95, "Data used to build the steering direction", fontsize=17, fontweight="bold", ha="center")
        for ax, (title, points) in zip(axes, BENCHMARKS):
            draw_panel(ax, title, points)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.615, 0.84),
                   frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.6)
        fig.text(0.615, 0.055, "Change vs unedited model (percentage points)", ha="center",
                 fontsize=12, color="#374151")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="svg", facecolor="white",
                    metadata={"Date": None, "Title": "Qwen3 steering-direction comparison",
                              "Description": "Reported point estimates from the earlier Section 4 comparison."})
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="output SVG")
    args = ap.parse_args()
    write_plot(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
