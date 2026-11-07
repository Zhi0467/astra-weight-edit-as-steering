"""Shared SVG delta-grid renderer for steering reports."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DeltaMetricSpec:
    field: str
    label: str
    good_direction: str
    cap: float
    kind: str = "rate"


@dataclass(frozen=True)
class DeltaMetricGroup:
    label: str
    start: int
    span: int


@dataclass(frozen=True)
class DeltaGridSection:
    title: str
    rows: list[dict[str, Any]]
    subtitle: str = ""
    metric_subtitles: dict[str, str] | None = None


def write_delta_grid_svg(
    *,
    out_path: Path,
    title: str,
    subtitles: list[str],
    sections: list[DeltaGridSection],
    metrics: list[DeltaMetricSpec],
    row_label: Callable[[dict[str, Any]], str],
    metric_groups: list[DeltaMetricGroup] | None = None,
    cell_title: Callable[[dict[str, Any], DeltaMetricSpec, str], str] | None = None,
    cell_text: Callable[[dict[str, Any], DeltaMetricSpec], str] | None = None,
    cell_subtext: Callable[[dict[str, Any], DeltaMetricSpec], str] | None = None,
    footer: str = "",
    left: int = 210,
    top: int = 120,
    cell_w: int = 104,
    row_h: int = 32,
    section_h: int = 28,
    min_width: int = 0,
) -> None:
    data_row_count = sum(len(section.rows) for section in sections)
    has_section_context = any(
        section.subtitle or section.metric_subtitles for section in sections
    )
    effective_section_h = max(section_h, 42) if has_section_context else section_h
    width = max(left + cell_w * len(metrics) + 34, min_width)  # min_width: room for long titles / footers
    height = top + len(sections) * effective_section_h + data_row_count * row_h + 60
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="34" class="title">{_esc(title)}</text>',
    ]
    for idx, subtitle in enumerate(subtitles):
        body.append(f'<text x="32" y="{56 + idx * 20}" class="subtitle">{_esc(subtitle)}</text>')
    separator_positions: list[float] = []
    if metric_groups:
        for group in metric_groups:
            x = left + group.start * cell_w
            w = group.span * cell_w
            body.append(
                f'<rect x="{x}" y="{top - 48}" width="{w}" height="20" '
                'fill="#f3f4f6" rx="4"/>'
            )
            body.append(
                f'<text x="{x + w / 2:.1f}" y="{top - 34}" class="label" '
                f'font-weight="700" text-anchor="middle">{_esc(group.label)}</text>'
            )
        for group in metric_groups[:-1]:
            separator_positions.append(left + (group.start + group.span) * cell_w)
    for col_idx, metric in enumerate(metrics):
        x = left + col_idx * cell_w + cell_w / 2
        body.append(
            f'<text x="{x:.1f}" y="{top - 18}" class="label" '
            f'font-weight="700" text-anchor="middle">{_esc(metric.label)}</text>'
        )

    y = top
    for section in sections:
        body.append(
            f'<rect x="28" y="{y - 2}" width="{width - 56}" height="{effective_section_h - 4}" '
            'fill="#eef2f7" rx="4"/>'
        )
        if section.subtitle:
            body.append(
                f'<text x="42" y="{y + 14}" class="label" font-weight="700">{_esc(section.title)}</text>'
            )
            body.append(
                f'<text x="42" y="{y + 31}" class="small">{_esc(section.subtitle)}</text>'
            )
        else:
            body.append(
                f'<text x="42" y="{y + 17}" class="label" font-weight="700">{_esc(section.title)}</text>'
            )
        metric_subtitles = section.metric_subtitles or {}
        for col_idx, metric in enumerate(metrics):
            value = metric_subtitles.get(metric.field)
            if not value:
                continue
            x = left + col_idx * cell_w + cell_w / 2
            body.append(
                f'<text x="{x:.1f}" y="{y + 31}" class="small" '
                f'text-anchor="middle">{_esc(value)}</text>'
            )
        y += effective_section_h
        for row in section.rows:
            label = row_label(row)
            body.append(
                f'<text x="{left - 14}" y="{y + row_h / 2 + 4:.1f}" '
                f'class="small" text-anchor="end">{_esc(label)}</text>'
            )
            for col_idx, metric in enumerate(metrics):
                value = _float(row.get(metric.field))
                fill = _delta_color(value, good_direction=metric.good_direction, cap=metric.cap)
                text = cell_text(row, metric) if cell_text is not None else _format_value(value, metric.kind)
                subtext = cell_subtext(row, metric) if cell_subtext is not None else ""
                x = left + col_idx * cell_w
                title_text = (
                    cell_title(row, metric, text)
                    if cell_title is not None
                    else f"{section.title} {label} {metric.label}: {text}"
                )
                body.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{row_h}" '
                    f'class="cell" fill="{fill}"><title>{_esc(title_text)}</title></rect>'
                )
                text_y = y + row_h / 2 + (-2 if subtext else 4)
                body.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{text_y:.1f}" '
                    'font-size="11" font-weight="650" text-anchor="middle" '
                    f'style="fill: {_text_color(fill)}">{_esc(text)}</text>'
                )
                if subtext:
                    body.append(
                        f'<text x="{x + cell_w / 2:.1f}" y="{y + row_h / 2 + 11:.1f}" '
                        'font-size="9" text-anchor="middle" '
                        f'style="fill: {_text_color(fill)}">{_esc(subtext)}</text>'
                    )
            y += row_h

    for x in separator_positions:
        body.append(
            f'<line x1="{x:.1f}" y1="{top - 52}" x2="{x:.1f}" y2="{y}" '
            'class="group-separator"/>'
        )

    if footer:
        body.append(f'<text x="{left}" y="{height - 24}" class="small">{_esc(footer)}</text>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_svg(width, height, body), encoding="utf-8")


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _format_value(value: float | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "tokens":
        if abs(value) >= 1000.0:
            return f"{value / 1000.0:+.1f}k"
        return f"{value:+,.0f}"
    if kind == "pct":
        return f"{100.0 * value:.1f}%"
    if kind == "rate":
        return f"{100.0 * value:+.1f} pp"
    return f"{value:+.3g}"


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    red = round(a[0] + (b[0] - a[0]) * t)
    green = round(a[1] + (b[1] - a[1]) * t)
    blue = round(a[2] + (b[2] - a[2]) * t)
    return f"rgb({red},{green},{blue})"


def _delta_color(value: float | None, *, good_direction: str, cap: float) -> str:
    neutral = (246, 247, 249)
    good_color = (47, 125, 91)
    bad_color = (191, 75, 68)
    if value is None:
        return "rgb(246,247,249)"
    if abs(value) < 1e-12:
        return "rgb(246,247,249)"
    is_good = value > 0 if good_direction == "positive" else value < 0
    target = good_color if is_good else bad_color
    return _blend(neutral, target, min(abs(value) / cap, 1.0))


def _text_color(fill: str) -> str:
    if not fill.startswith("rgb("):
        return "#111827"
    parts = [int(part) for part in fill[4:-1].split(",")]
    luminance = 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2]
    return "#ffffff" if luminance < 135 else "#111827"


def _svg(width: int, height: int, body: list[str]) -> str:
    style = """
    <style>
      text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }
      .title { font-size: 22px; font-weight: 700; }
      .subtitle { font-size: 13px; fill: #4b5563; }
      .label { font-size: 12px; fill: #374151; }
      .small { font-size: 10px; fill: #4b5563; }
      .cell { stroke: #ffffff; stroke-width: 2; rx: 4; ry: 4; }
      .group-separator { stroke: #6b7280; stroke-width: 1.25; stroke-dasharray: 4 4; }
    </style>
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"{style}\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )
