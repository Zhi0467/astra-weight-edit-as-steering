#!/usr/bin/env python3
"""Blog Fig 2: the same sensed edit at the write site (MLP output, a weight edit) vs the residual site (block output,
run time), Astra regime (Qwen3.5-0.8B, Astra's sampler, 1,152 confirmation problems).

Two small panels (accuracy, loop rate), one group per edit (s = 4v; learned ul s), one bar per site, the unedited model
as a dashed reference line. Each group is captioned with its effective gain beta = s . v and the paired residual - write
difference with its 95% problem-bootstrap interval. Self-contained HTML (inline SVG, light/dark, hover titles, table).

  build_blog_fig2_site.py --summary test_summary_resid.json --specs DIR_WITH_resid_*.pt --out fig2_write_vs_residual.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

GROUPS = [  # (label, write arm, residual arm, spec file)
    ("s = 4v", "sv_n32_a4", "resid_sv_n32_a4", "resid_sv_n32_a4.pt"),
    ("learned s (unlikelihood)", "ul_n32_b0.1", "resid_ul_n32_b0.1", "resid_ul_n32_b0.1.pt"),
]
PANELS = [("accuracy", "Accuracy", "higher is better"), ("loop", "Loop rate", "lower is better")]


def beta(spec):
    import torch
    d = torch.load(spec, map_location="cpu")
    return [float(d["s"][l].double() @ d["v"][l].double()) for l in d["blocks"]]


def panel(s, metric, title, note, betas, x0):
    W, H, top, base = 330, 300, 46, 250
    sc = lambda f: base - f * (base - top)  # 0..1 -> y
    out = [f'<g transform="translate({x0 + 12},0)">',
           f'<text x="0" y="16" class="t1">{title}</text><text x="0" y="33" class="t3">{note} · 1,152 problems</text>']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="34" x2="{W}" y1="{sc(t):.1f}" y2="{sc(t):.1f}" class="grid"/>'
                   f'<text x="28" y="{sc(t) + 4:.1f}" class="t3" text-anchor="end">{int(t * 100)}%</text>')
    ctrl = s["per_arm"]["control"]["pooled"][metric]
    gw = (W - 44) / len(GROUPS)
    for gi, (lab, wa, ra, _) in enumerate(GROUPS):
        gx = 44 + gi * gw
        for bi, (arm, cls, site) in enumerate(((wa, "s1", "write site"), (ra, "s2", "residual site"))):
            v = s["per_arm"][arm]["pooled"][metric]
            bx, bw = gx + 18 + bi * 52, 46
            y = sc(v)
            r = min(4, base - y)
            path = (f"M{bx},{base} V{y + r:.1f} Q{bx},{y:.1f} {bx + r},{y:.1f} H{bx + bw - r} "
                    f"Q{bx + bw},{y:.1f} {bx + bw},{y + r:.1f} V{base} Z")
            out.append(f'<path d="{path}" class="{cls}"><title>{html.escape(lab)}, {site}: '
                       f'{100 * v:.1f}%</title></path>'
                       f'<text x="{bx + bw / 2}" y="{y - 6:.1f}" class="t2" text-anchor="middle">{100 * v:.1f}</text>')
        d = s["paired"][f"{ra} - {wa}"]["pooled"][metric]
        b = betas[gi]
        out.append(f'<text x="{gx + 70}" y="{base + 18}" class="t2" text-anchor="middle">{html.escape(lab)}</text>'
                   f'<text x="{gx + 70}" y="{base + 34}" class="t3" text-anchor="middle">β = s·v: {b[0]:.2g} / {b[1]:.2g}</text>'
                   f'<text x="{gx + 70}" y="{base + 50}" class="t3" text-anchor="middle">Δ {100 * d["diff"]:+.1f} '
                   f'[{100 * d["ci95"][0]:+.1f}, {100 * d["ci95"][1]:+.1f}]</text>')
    out.append(f'<line x1="34" x2="{W}" y1="{sc(ctrl):.1f}" y2="{sc(ctrl):.1f}" class="ref"><title>unedited model: {100 * ctrl:.1f}%</title></line>')
    out.append(f'<line x1="34" x2="{W}" y1="{base}" y2="{base}" class="axis"/></g>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    betas = [beta(Path(a.specs) / g[3]) for g in GROUPS]
    svg = (f'<svg viewBox="0 0 740 315" role="img" aria-label="Accuracy and loop rate, write vs residual site">'
           f'{panel(s, "accuracy", "Accuracy", "higher is better", betas, 0)}'
           f'{panel(s, "loop", "Loop rate", "lower is better", betas, 380)}</svg>')
    rows = "".join(
        f"<tr><td>{html.escape(lab)}</td><td>{site}</td>"
        + "".join(f"<td>{100 * s['per_arm'][arm]['pooled'][m]:.1f}</td>" for m in ("accuracy", "loop", "cap_hit", "think_closed"))
        + "</tr>"
        for lab, wa, ra, _ in GROUPS for arm, site in ((wa, "write (MLP output, weight edit)"), (ra, "residual (block output, run time)")))
    c = s["per_arm"]["control"]["pooled"]
    rows += ("<tr><td>unedited</td><td>—</td>" + "".join(f"<td>{100 * c[m]:.1f}</td>" for m in ("accuracy", "loop", "cap_hit", "think_closed")) + "</tr>")
    css = """
.viz-root{--surface-1:#fcfcfb;--text-primary:#1f1e1c;--text-secondary:#55534d;--text-muted:#7a786f;--grid:#e6e4de;--series-1:#2a78d6;--series-2:#eb6834;
 font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:var(--surface-1);color:var(--text-primary);max-width:760px;margin:1.5em auto;padding:0 1em}
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a988e;--grid:#2e2d2a;--series-1:#3987e5;--series-2:#d95926}}
body{margin:0;background:var(--surface-1)}
.t1{font-size:15px;font-weight:600;fill:var(--text-primary)}.t2{font-size:12.5px;fill:var(--text-primary)}.t3{font-size:11.5px;fill:var(--text-muted)}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--text-muted);stroke-width:1}.ref{stroke:var(--text-secondary);stroke-width:1.5;stroke-dasharray:4 3}
.s1{fill:var(--series-1)}.s2{fill:var(--series-2)}path:hover{opacity:.85}
.legend{display:flex;gap:18px;margin:.3em 0 .6em;color:var(--text-secondary);font-size:13px}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
h3{margin:.2em 0 .1em;font-size:17px}p{color:var(--text-secondary);margin:.3em 0}table{border-collapse:collapse;font-size:13px;margin-top:.5em}
td,th{border-bottom:1px solid var(--grid);padding:.3em .7em;text-align:right}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}
details{margin-top:.8em;color:var(--text-secondary)}"""
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fig 2: write site vs residual site</title><style>{css}</style></head><body><div class="viz-root">
<h3>Same edit, two sites: the write site is stable, the residual site depends on β = s·v</h3>
<p>Qwen3.5-0.8B, Astra's sampler, layers 15 and 19. Only the site changes: the MLP output (a weight edit) or the full block output (at run time).</p>
<div class="legend"><span><span class="sw s1" style="background:var(--series-1)"></span>write site (MLP output)</span><span><span class="sw" style="background:var(--series-2)"></span>residual site (block output)</span><span>- - unedited model (accuracy {100 * c['accuracy']:.1f}%, loop {100 * c['loop']:.1f}%)</span></div>
{svg}
<p>With β = 4 each residual edit flips and triples the sensed coordinate, and layer 19 re-senses layer 15's correction; the learned s has β ≈ 0.3, a contracting map. Δ = residual − write in points, 95% paired problem bootstrap; β at layers 15 / 19; one seeded draw per problem.</p>
<details><summary>Table</summary><table><tr><th>edit</th><th>site</th><th>acc %</th><th>loop %</th><th>cap hit %</th><th>closed think %</th></tr>{rows}</table></details>
</div></body></html>"""
    Path(a.out).write_text(page)
    print(a.out)


if __name__ == "__main__":
    main()
