"""Build Astra and Qwen3 comparison grids with one shared SVG renderer."""
import argparse
import html
import json
from pathlib import Path

NAME = {"control_nothink": "Original model, thinking off", "ul": "Unlikelihood",
        "clas": "Positive-only NTP", "clasd": "Corrective NTP"}
GOOD, BAD, NEUTRAL = (92, 180, 165), (197, 114, 123), (237, 238, 240)


def esc(x):
    return html.escape(str(x), quote=True)


def mix(t, end):
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(NEUTRAL, end))


def label(arm):
    if arm.startswith("sv_"):
        al = arm.split("_a")[-1]
        return "Astra’s steering (s = v)" if al == "1" else f"Astra’s steering (s = {al}v)"
    return NAME.get(arm) or NAME[arm.split("_n")[0]]


def sections(levels, include_positive):
    families = ("ul", "clas", "clasd") if include_positive else ("ul", "clasd")
    return [("Reference", ["control_nothink"])] + [
        (f"{n} pairs", [a for a in levels if a.startswith(f"sv_n{n}_")] +
         [f"{f}_n{n}" for f in families]) for n in (4, 32)]


def astra_heatmap(s, ds, out, include_positive=False):
    P = s["paired"]
    sv = {n: next(a for a in s["per_arm"] if a.startswith(f"sv_n{n}_")) for n in (4, 32)}
    # (header, reference, metric, good direction, colour cap)
    cols = [("Accuracy ↑", "control", "accuracy", +1, 0.50), ("Loop rate ↓", "control", "loop", -1, 0.80),
            ("Thinking closed ↑", "control", "think_closed", +1, 0.80), ("Cap hits ↓", "control", "cap_hit", -1, 0.80),
            ("Tokens", "control", "tokens", -1, 15000),
            ("Accuracy ↑", "sv", "accuracy", +1, 0.10), ("Loop rate ↓", "sv", "loop", -1, 0.30), ("Tokens", "sv", "tokens", -1, 3000)]
    groups = [("Δ vs original model (thinking on)", 0, 5), ("Δ vs s = 4v, same pair count" if all(a.endswith("_a4") for a in sv.values()) else "Δ vs Astra’s steering, same pair count", 5, 3)]
    def get(arm, ref, m):
        if arm == "control_nothink" and m == "think_closed":
            return None
        if ref == "sv":
            if arm.startswith(("sv_", "random_", "control")):
                return None
            ref = sv[int(arm.split("_n")[1].split("_")[0])]
        x = P.get(f"{arm} - {ref}", {}).get(ds, {}).get(m)
        return (x["diff"], *x["ci95"]) if x else None

    render(out, cols, groups, sections(s["per_arm"], include_positive), get,
           left=270, cw=115, astra=True)


def heatmap(d, ds, out, include_positive=False):
    g = {"math": "math", "lcb": "lcb"}[ds]
    # (header, comparison, group, metric, good direction, colour cap)
    cols = [("Accuracy (all) ↑", "vs_control", g, "correct", +1, 0.08), ("Accuracy (looping) ↑", "vs_control", f"{g}_loop", "correct", +1, 0.08),
            ("Accuracy (non-looping) ↑", "vs_control", f"{g}_nonloop", "correct", +1, 0.08), ("Loop rate ↓", "vs_control", g, "loop", -1, 0.40),
            ("Thinking closed ↑", "vs_control", g, "closed", +1, 0.20), ("Tokens", "vs_control", g, "tokens", -1, 8000)]
    groups = [("Δ vs original model (thinking on)", 0, 6)]
    def get(arm, cmp_, grp, m):
        if arm == "control_nothink" and m == "closed":
            return None
        x = d[cmp_].get(arm, {}).get(grp, {}).get(m)
        return (x["mean"], x["lo"], x["hi"]) if x else None

    render(out, cols, groups, sections(d["levels"], include_positive), get,
           left=240, cw=150)


def render(out, cols, groups, sections, get, *, left, cw, astra=False):
    ch, rowh, top = 30, 34, 60
    def row_label(arm):
        if astra and arm.startswith("sv_"):
            return f"Astra’s steering (s = {arm.split('_a')[-1]}v)"
        return label(arm)
    header_style = 'class="label"' if astra else 'class="small" style="font-size:10px"'
    width = left + cw * len(cols) + 40
    b = ['<rect width="100%" height="100%" fill="#ffffff"/>']
    for t, st, sp in groups:
        x = left + st * cw
        b.append(f'<rect x="{x}" y="12" width="{sp * cw}" height="20" fill="#f3f4f6" rx="4"/>')
        b.append(f'<text x="{x + sp * cw / 2}" y="26" class="label" font-weight="700" text-anchor="middle">{esc(t)}</text>')
    for j, c in enumerate(cols):
        b.append(f'<text x="{left + j * cw + cw / 2}" y="{top - 18}" {header_style} font-weight="700" text-anchor="middle">{esc(c[0])}</text>')
    y = top - 2
    for title, arms in sections:
        b.append(f'<rect x="28" y="{y}" width="{width - 56}" height="24" fill="#eef2f7" rx="4"/>')
        b.append(f'<text x="42" y="{y + 17}" class="label" font-weight="700">{esc(title)}</text>')
        y += 30
        for arm in arms:
            b.append(f'<text x="{left - 14}" y="{y + 19}" class="small" text-anchor="end">{esc(row_label(arm))}</text>')
            for j, (hdr, *keys, m, good, cap) in enumerate(cols):
                x = left + j * cw + 2
                s = get(arm, *keys, m)
                if s is None:
                    fill, txt, tip = mix(0, GOOD), "—", "not applicable"
                else:
                    v, lo, hi = s
                    t = abs(v) / cap
                    fill = mix(t, GOOD if v * good > 0 else BAD) if abs(v) > 1e-12 else mix(0, GOOD)
                    f = (lambda z: f"{z / 1000:+.1f}k" if abs(z) >= 1000 else f"{z:+.0f}") if m == "tokens" else (lambda z: f"{100 * z:+.1f}")
                    star = "*" if lo > 0 or hi < 0 else ""
                    txt, tip = f(v) + star, f"{f(v)} [{f(lo)}, {f(hi)}]"
                b.append(f'<rect x="{x}" y="{y}" width="{cw - 4}" height="{ch}" rx="6" ry="6" fill="{fill}" stroke="white" stroke-width="1">'
                         f'<title>{esc(row_label(arm))} ({esc(hdr)}): {esc(tip)}</title></rect>')
                b.append(f'<text x="{x + (cw - 4) / 2}" y="{y + 19}" font-size="13" font-weight="650" text-anchor="middle" '
                         f'style="fill:#000000">{esc(txt).replace("-", "−")}</text>')
            y += rowh
    if astra:
        sep = left + 5 * cw
        b.append(f'<line x1="{sep}" y1="8" x2="{sep}" y2="{y}" stroke="#6b7280" stroke-width="1.25"/>')
    style = """<style>
      text { font-family: "DejaVu Sans", sans-serif; fill: #111827; }
      .label { font-size: 12px; fill: #374151; }
      .small { font-size: 12px; fill: #4b5563; }
    </style>"""
    out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{y + 16}" viewBox="0 0 {width} {y + 16}" role="img">\n'
                   f'{style}\n' + "\n".join(b) + "\n</svg>\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Build blog comparison grids with Corrective NTP.")
    ap.add_argument("--out", type=Path, default=root / "blog/figures")
    ap.add_argument("--qwen3", type=Path, default=root / "results/qwen3/exp1_test_summary_with_clasd.json")
    ap.add_argument("--astra", type=Path, default=root / "results/astra_regime/test_summary_with_clasd.json")
    ap.add_argument("--astra-nothink", type=Path, default=root / "results/astra_regime/test_summary_nothink.json")
    ap.add_argument("--only", default="2a,2b,3a,3b")
    a = ap.parse_args()
    selected = set(a.only.split(","))
    if selected - {"2a", "2b", "3a", "3b"}:
        ap.error("valid grids: 2a,2b,3a,3b")
    a.out.mkdir(parents=True, exist_ok=True)
    d = json.loads(a.qwen3.read_text()) if selected & {"3a", "3b"} else None
    for ds in ("math", "lcb"):
        if {"math":"3a", "lcb":"3b"}[ds] not in selected: continue
        name = f"exp1_qwen3_{ds}_delta_grid"
        heatmap(d, ds, a.out / f"{name}.svg")
        heatmap(d, ds, a.out / f"{name}_with_positive_ntp.svg", include_positive=True)
    if selected & {"2a", "2b"}:
        s = json.loads(a.astra.read_text())
        t = json.loads(a.astra_nothink.read_text())
        s["per_arm"]["control_nothink"] = t["per_arm"]["control_nothink"]
        s["paired"]["control_nothink - control"] = t["paired"]["control_nothink - control"]
    for ds in ("gsm8k", "math"):
        if {"gsm8k":"2a", "math":"2b"}[ds] not in selected: continue
        astra_heatmap(s, ds, a.out / f"exp1_astra_{ds}_delta_grid.svg")
        astra_heatmap(s, ds, a.out / f"exp1_astra_{ds}_delta_grid_with_positive_ntp.svg", include_positive=True)

if __name__ == "__main__":
    main()
