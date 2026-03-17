"""
taintflow.visualization.leakage_heatmap -- Leakage heatmap rendering.

Generates feature × stage heatmaps showing how much information (in bits)
leaks at each point in the ML pipeline.  Supports SVG-based rendering,
interactive HTML with hover tooltips, and terminal-based ASCII heatmaps
using Unicode block characters.

All rendering uses the standard library—no matplotlib or plotly.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from taintflow.core.types import (
    FeatureLeakage,
    LeakageReport,
    Severity,
    StageLeakage,
)

# ===================================================================
#  Utility helpers
# ===================================================================


def _escape(text: str) -> str:
    return html.escape(str(text), quote=True)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ===================================================================
#  ColorMapper – map float values to RGB
# ===================================================================


class ColorMapper:
    """Map a floating-point value in ``[vmin, vmax]`` to an RGB colour.

    Two built-in colour scales are provided:

    * ``"sequential"`` – white → red  (default, good for 0-to-max).
    * ``"diverging"``  – blue → white → red  (for signed or centred data).

    Custom scales can be supplied as a list of ``(position, (r, g, b))``
    stops where *position* ∈ [0, 1].

    Parameters
    ----------
    vmin : float
        Value mapped to the low end of the scale.
    vmax : float
        Value mapped to the high end of the scale.
    scale : str
        ``"sequential"`` or ``"diverging"``.
    stops : list, optional
        Custom colour stops.
    """

    _SEQUENTIAL_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
        (0.0, (255, 255, 255)),
        (0.25, (255, 235, 210)),
        (0.5, (252, 174, 100)),
        (0.75, (227, 74, 51)),
        (1.0, (179, 0, 0)),
    ]
    _DIVERGING_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
        (0.0, (33, 102, 172)),
        (0.25, (103, 169, 207)),
        (0.5, (255, 255, 255)),
        (0.75, (239, 138, 98)),
        (1.0, (178, 24, 43)),
    ]

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 1.0,
        scale: str = "sequential",
        stops: Optional[List[Tuple[float, Tuple[int, int, int]]]] = None,
    ) -> None:
        self.vmin = vmin
        self.vmax = vmax if vmax > vmin else vmin + 1.0
        if stops is not None:
            self.stops = sorted(stops, key=lambda s: s[0])
        elif scale == "diverging":
            self.stops = list(self._DIVERGING_STOPS)
        else:
            self.stops = list(self._SEQUENTIAL_STOPS)

    def to_rgb(self, value: float) -> Tuple[int, int, int]:
        """Map *value* to an ``(r, g, b)`` tuple.

        Values outside ``[vmin, vmax]`` are clamped.
        """
        t = _clamp((value - self.vmin) / (self.vmax - self.vmin))
        # Find bounding stops
        for i in range(len(self.stops) - 1):
            t0, c0 = self.stops[i]
            t1, c1 = self.stops[i + 1]
            if t0 <= t <= t1:
                if t1 == t0:
                    frac = 0.0
                else:
                    frac = (t - t0) / (t1 - t0)
                r = int(c0[0] + frac * (c1[0] - c0[0]))
                g = int(c0[1] + frac * (c1[1] - c0[1]))
                b = int(c0[2] + frac * (c1[2] - c0[2]))
                return (_clamp_int(r), _clamp_int(g), _clamp_int(b))
        return self.stops[-1][1]

    def to_hex(self, value: float) -> str:
        """Map *value* to a hex colour string ``#rrggbb``."""
        r, g, b = self.to_rgb(value)
        return f"#{r:02x}{g:02x}{b:02x}"

    def text_colour(self, value: float) -> str:
        """Return black or white text colour for readability on the cell."""
        r, g, b = self.to_rgb(value)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if luminance > 128 else "#ffffff"


def _clamp_int(v: int) -> int:
    return max(0, min(255, v))


# ===================================================================
#  LeakageHeatmap – SVG heatmap
# ===================================================================


class LeakageHeatmap:
    """Feature × stage leakage heatmap with SVG rendering.

    Rows represent output features, columns represent pipeline stages,
    and cell colours encode the leakage amount in bits.

    Parameters
    ----------
    report : LeakageReport
        The audit report to visualise.
    scale : str
        Colour scale: ``"sequential"`` or ``"diverging"``.
    cell_width : float
        Width of each heatmap cell in SVG units.
    cell_height : float
        Height of each heatmap cell.
    """

    def __init__(
        self,
        report: LeakageReport,
        *,
        scale: str = "sequential",
        cell_width: float = 70,
        cell_height: float = 28,
    ) -> None:
        self.report = report
        self.scale = scale
        self.cell_width = cell_width
        self.cell_height = cell_height

        self._matrix, self._features, self._stages = self._build_matrix()
        vmax = max((max(row) for row in self._matrix), default=1.0) if self._matrix else 1.0
        vmax = max(vmax, 1e-9)
        self._mapper = ColorMapper(vmin=0.0, vmax=vmax, scale=scale)

    # -- matrix construction -------------------------------------------------

    def _build_matrix(
        self,
    ) -> Tuple[List[List[float]], List[str], List[str]]:
        """Extract a features × stages matrix from the report.

        Returns
        -------
        matrix : list[list[float]]
            ``matrix[feature_idx][stage_idx]`` = leakage in bits.
        features : list[str]
            Ordered feature (column) names.
        stages : list[str]
            Ordered stage names.
        """
        stage_names: List[str] = []
        feature_set: Dict[str, int] = {}

        for sl in self.report.stage_leakages:
            stage_names.append(sl.stage_name or sl.stage_id)
            for fl in sl.feature_leakages:
                if fl.column_name not in feature_set:
                    feature_set[fl.column_name] = len(feature_set)

        features = sorted(feature_set.keys())
        feat_idx = {f: i for i, f in enumerate(features)}
        n_feat = len(features)
        n_stage = len(stage_names)

        matrix: List[List[float]] = [[0.0] * n_stage for _ in range(n_feat)]
        for si, sl in enumerate(self.report.stage_leakages):
            for fl in sl.feature_leakages:
                fi = feat_idx.get(fl.column_name)
                if fi is not None:
                    matrix[fi][si] = fl.bit_bound

        return matrix, features, stage_names

    @property
    def features(self) -> List[str]:
        """Ordered list of feature names (rows)."""
        return list(self._features)

    @property
    def stages(self) -> List[str]:
        """Ordered list of stage names (columns)."""
        return list(self._stages)

    @property
    def matrix(self) -> List[List[float]]:
        """The raw leakage matrix ``[feature_idx][stage_idx]``."""
        return [list(row) for row in self._matrix]

    # -- SVG rendering -------------------------------------------------------

    def to_svg(self) -> str:
        """Render the heatmap as a complete SVG string.

        Returns
        -------
        str
            SVG document.
        """
        if not self._matrix or not self._features or not self._stages:
            return self._empty_svg()

        label_margin_left = max(len(f) for f in self._features) * 7 + 10
        label_margin_top = max(len(s) for s in self._stages) * 6 + 10
        n_feat = len(self._features)
        n_stage = len(self._stages)

        grid_w = n_stage * self.cell_width
        grid_h = n_feat * self.cell_height
        total_w = label_margin_left + grid_w + 80  # extra space for legend
        total_h = label_margin_top + grid_h + 20

        parts: List[str] = []
        parts.append(
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{total_w}" height="{total_h}" '
            f'viewBox="0 0 {total_w} {total_h}">'
        )
        parts.append(f'<rect width="{total_w}" height="{total_h}" fill="white"/>')

        # Column headers (stage names, rotated)
        for si, stage in enumerate(self._stages):
            x = label_margin_left + si * self.cell_width + self.cell_width / 2
            y = label_margin_top - 6
            parts.append(
                f'<text x="{x}" y="{y}" text-anchor="end" '
                f'font-family="sans-serif" font-size="9" fill="#333" '
                f'transform="rotate(-45 {x} {y})">{_escape(stage)}</text>'
            )

        # Row headers (feature names)
        for fi, feat in enumerate(self._features):
            x = label_margin_left - 6
            y = label_margin_top + fi * self.cell_height + self.cell_height / 2 + 3
            parts.append(
                f'<text x="{x}" y="{y}" text-anchor="end" '
                f'font-family="sans-serif" font-size="9" fill="#333">'
                f'{_escape(feat)}</text>'
            )

        # Cells
        for fi in range(n_feat):
            for si in range(n_stage):
                val = self._matrix[fi][si]
                fill = self._mapper.to_hex(val)
                tc = self._mapper.text_colour(val)
                cx = label_margin_left + si * self.cell_width
                cy = label_margin_top + fi * self.cell_height

                parts.append(
                    f'<rect x="{cx}" y="{cy}" '
                    f'width="{self.cell_width}" height="{self.cell_height}" '
                    f'fill="{fill}" stroke="#ddd" stroke-width="0.5">'
                    f'<title>{_escape(self._features[fi])} @ '
                    f'{_escape(self._stages[si])}: {val:.3f} bits</title></rect>'
                )
                # Value text
                if val > 0:
                    tx = cx + self.cell_width / 2
                    ty = cy + self.cell_height / 2 + 3
                    parts.append(
                        f'<text x="{tx}" y="{ty}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="8" '
                        f'fill="{tc}">{val:.1f}</text>'
                    )

        # Colour scale legend
        self._render_legend(parts, label_margin_left + grid_w + 15, label_margin_top, grid_h)

        parts.append("</svg>")
        return "\n".join(parts)

    def _render_legend(
        self, parts: List[str], x: float, y: float, height: float
    ) -> None:
        """Add a vertical colour-scale legend."""
        bar_w = 15
        n_steps = 20
        step_h = height / n_steps
        for i in range(n_steps):
            t = i / max(n_steps - 1, 1)
            val = self._mapper.vmin + t * (self._mapper.vmax - self._mapper.vmin)
            fill = self._mapper.to_hex(val)
            sy = y + (n_steps - 1 - i) * step_h
            parts.append(
                f'<rect x="{x}" y="{sy}" width="{bar_w}" height="{step_h + 1}" '
                f'fill="{fill}" stroke="none"/>'
            )
        # Min / max labels
        parts.append(
            f'<text x="{x + bar_w + 4}" y="{y + height}" '
            f'font-family="sans-serif" font-size="8" fill="#333">'
            f'{self._mapper.vmin:.1f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w + 4}" y="{y + 8}" '
            f'font-family="sans-serif" font-size="8" fill="#333">'
            f'{self._mapper.vmax:.1f}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{y - 6}" '
            f'font-family="sans-serif" font-size="9" fill="#333">bits</text>'
        )

    def _empty_svg(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60">'
            '<text x="10" y="30" font-family="sans-serif" font-size="12" '
            'fill="#999">(no data)</text></svg>'
        )

    def save_svg(self, path: str | Path) -> None:
        """Write the SVG heatmap to a file."""
        Path(path).write_text(self.to_svg(), encoding="utf-8")


# ===================================================================
#  HTMLHeatmap – interactive HTML with hover tooltips
# ===================================================================


class HTMLHeatmap:
    """Interactive HTML heatmap with hover tooltips.

    Wraps a :class:`LeakageHeatmap` and emits a self-contained HTML page
    with CSS-based tooltips.  No JavaScript dependencies.

    Parameters
    ----------
    heatmap : LeakageHeatmap
        The underlying heatmap data.
    title : str
        Page title.
    """

    def __init__(self, heatmap: LeakageHeatmap, title: str = "Leakage Heatmap") -> None:
        self.heatmap = heatmap
        self.title = title

    def to_html(self) -> str:
        """Render the heatmap as a self-contained HTML string.

        Returns
        -------
        str
            Complete HTML document.
        """
        matrix = self.heatmap.matrix
        features = self.heatmap.features
        stages = self.heatmap.stages
        mapper = self.heatmap._mapper

        rows: List[str] = []
        rows.append("<!DOCTYPE html>")
        rows.append(f"<html><head><meta charset='utf-8'><title>{_escape(self.title)}</title>")
        rows.append("<style>")
        rows.append(self._css())
        rows.append("</style></head><body>")
        rows.append(f"<h2>{_escape(self.title)}</h2>")
        rows.append('<div class="heatmap-wrap"><table class="heatmap">')

        # Header row
        rows.append("<tr><th></th>")
        for stage in stages:
            rows.append(f'<th class="col-header">{_escape(stage)}</th>')
        rows.append("</tr>")

        # Data rows
        for fi, feat in enumerate(features):
            rows.append(f'<tr><td class="row-header">{_escape(feat)}</td>')
            for si, stage in enumerate(stages):
                val = matrix[fi][si] if fi < len(matrix) and si < len(matrix[fi]) else 0.0
                bg = mapper.to_hex(val)
                tc = mapper.text_colour(val)
                sev = Severity.from_bits(val)
                tooltip = f"{feat} @ {stage}: {val:.3f} bits ({sev.name})"
                cell_text = f"{val:.1f}" if val > 0 else ""
                rows.append(
                    f'<td class="cell" style="background:{bg};color:{tc}" '
                    f'title="{_escape(tooltip)}">{cell_text}</td>'
                )
            rows.append("</tr>")

        rows.append("</table></div>")
        rows.append("</body></html>")
        return "\n".join(rows)

    @staticmethod
    def _css() -> str:
        return (
            "body { font-family: sans-serif; margin: 20px; }\n"
            ".heatmap-wrap { overflow-x: auto; }\n"
            ".heatmap { border-collapse: collapse; }\n"
            ".heatmap th, .heatmap td { padding: 4px 8px; font-size: 11px; }\n"
            ".col-header { writing-mode: vertical-lr; transform: rotate(180deg); "
            "text-align: left; max-width: 30px; }\n"
            ".row-header { text-align: right; font-weight: normal; white-space: nowrap; }\n"
            ".cell { text-align: center; min-width: 50px; cursor: default; "
            "border: 1px solid #eee; }\n"
            ".cell:hover { outline: 2px solid #333; z-index: 1; }\n"
        )

    def save_html(self, path: str | Path) -> None:
        """Write the HTML heatmap to a file."""
        Path(path).write_text(self.to_html(), encoding="utf-8")


# ===================================================================
#  ASCIIHeatmap – terminal rendering with Unicode blocks
# ===================================================================

# Increasing-density block characters
_BLOCKS = " ░▒▓█"


class ASCIIHeatmap:
    """Terminal-based heatmap using Unicode block characters.

    Parameters
    ----------
    heatmap : LeakageHeatmap
        The underlying heatmap data.
    max_label_width : int
        Maximum characters for feature labels.
    """

    def __init__(self, heatmap: LeakageHeatmap, max_label_width: int = 16) -> None:
        self.heatmap = heatmap
        self.max_label_width = max_label_width

    def render(self) -> str:
        """Render the heatmap as an ASCII string.

        Returns
        -------
        str
            Multi-line Unicode block heatmap.
        """
        matrix = self.heatmap.matrix
        features = self.heatmap.features
        stages = self.heatmap.stages

        if not matrix or not features or not stages:
            return "(no data)"

        vmax = max(max(row) for row in matrix) if matrix else 1.0
        vmax = max(vmax, 1e-9)

        lw = self.max_label_width
        lines: List[str] = []

        # Header
        header = " " * (lw + 2)
        for s in stages:
            header += s[:3].center(4)
        lines.append(header)
        lines.append(" " * (lw + 2) + "─" * (len(stages) * 4))

        # Rows
        for fi, feat in enumerate(features):
            label = feat[:lw].rjust(lw)
            row_str = f"{label} │"
            for si in range(len(stages)):
                val = matrix[fi][si]
                t = _clamp(val / vmax)
                idx = int(t * (len(_BLOCKS) - 1))
                row_str += f" {_BLOCKS[idx]}  "
            lines.append(row_str)

        # Legend
        lines.append("")
        lines.append(f"Scale: {_BLOCKS[0]}=0  {_BLOCKS[-1]}={vmax:.1f} bits")
        sev_line = "Severity: "
        for s in (Severity.NEGLIGIBLE, Severity.WARNING, Severity.CRITICAL):
            lo, hi = _severity_range(s)
            sev_line += f"  {s.name}({lo:.0f}-{hi:.0f}b)"
        lines.append(sev_line)
        return "\n".join(lines)


def _severity_range(sev: Severity) -> Tuple[float, float]:
    """Return (lo, hi) bit bounds for a severity level."""
    if sev == Severity.NEGLIGIBLE:
        return (0.0, 1.0)
    if sev == Severity.WARNING:
        return (1.0, 8.0)
    return (8.0, math.inf)


# ===================================================================
#  AggregationViews – summary bar charts
# ===================================================================


class AggregationViews:
    """Generate aggregated summary views from a leakage heatmap.

    Provides per-feature totals, per-stage totals, and cumulative
    leakage curves as SVG bar charts.

    Parameters
    ----------
    heatmap : LeakageHeatmap
        The underlying heatmap data.
    bar_height : float
        Height of each bar in horizontal bar charts.
    max_bar_width : float
        Maximum bar width in SVG units.
    """

    def __init__(
        self,
        heatmap: LeakageHeatmap,
        bar_height: float = 20,
        max_bar_width: float = 300,
    ) -> None:
        self.heatmap = heatmap
        self.bar_height = bar_height
        self.max_bar_width = max_bar_width

    # -- per-feature total leakage bar chart ---------------------------------

    def per_feature_svg(self) -> str:
        """SVG horizontal bar chart of total leakage per feature.

        Returns
        -------
        str
            SVG document.
        """
        matrix = self.heatmap.matrix
        features = self.heatmap.features
        if not matrix or not features:
            return _empty_agg_svg()

        totals = [sum(row) for row in matrix]
        return self._horizontal_bars(features, totals, "Per-Feature Total Leakage (bits)")

    # -- per-stage total leakage bar chart -----------------------------------

    def per_stage_svg(self) -> str:
        """SVG horizontal bar chart of total leakage per stage.

        Returns
        -------
        str
            SVG document.
        """
        matrix = self.heatmap.matrix
        stages = self.heatmap.stages
        if not matrix or not stages:
            return _empty_agg_svg()

        totals: List[float] = []
        for si in range(len(stages)):
            totals.append(sum(matrix[fi][si] for fi in range(len(matrix))))
        return self._horizontal_bars(stages, totals, "Per-Stage Total Leakage (bits)")

    # -- cumulative leakage through pipeline stages --------------------------

    def cumulative_svg(self) -> str:
        """SVG line chart of cumulative leakage through pipeline stages.

        Returns
        -------
        str
            SVG document.
        """
        matrix = self.heatmap.matrix
        stages = self.heatmap.stages
        if not matrix or not stages:
            return _empty_agg_svg()

        stage_totals: List[float] = []
        for si in range(len(stages)):
            stage_totals.append(sum(matrix[fi][si] for fi in range(len(matrix))))

        cum: List[float] = []
        running = 0.0
        for st in stage_totals:
            running += st
            cum.append(running)

        return self._cumulative_line(stages, cum, "Cumulative Leakage Through Pipeline")

    # -- internal renderers --------------------------------------------------

    def _horizontal_bars(
        self, labels: List[str], values: List[float], title: str
    ) -> str:
        max_val = max(values) if values else 1.0
        max_val = max(max_val, 1e-9)
        label_w = max(len(l) for l in labels) * 7 + 10
        n = len(labels)
        chart_h = n * (self.bar_height + 4) + 40
        total_w = label_w + self.max_bar_width + 60

        parts: List[str] = []
        parts.append(
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{total_w}" height="{chart_h}" '
            f'viewBox="0 0 {total_w} {chart_h}">'
        )
        parts.append(f'<rect width="{total_w}" height="{chart_h}" fill="white"/>')
        # Title
        parts.append(
            f'<text x="{total_w / 2}" y="16" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#333" '
            f'font-weight="bold">{_escape(title)}</text>'
        )

        y_off = 30
        for i, (label, val) in enumerate(zip(labels, values)):
            y = y_off + i * (self.bar_height + 4)
            bar_w = (val / max_val) * self.max_bar_width
            sev = Severity.from_bits(val)
            fill = {"negligible": "#28a745", "warning": "#ffc107", "critical": "#dc3545"}.get(
                sev.value, "#888"
            )
            # Label
            parts.append(
                f'<text x="{label_w - 4}" y="{y + self.bar_height / 2 + 3}" '
                f'text-anchor="end" font-family="sans-serif" font-size="9" '
                f'fill="#333">{_escape(label)}</text>'
            )
            # Bar
            parts.append(
                f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" '
                f'height="{self.bar_height}" fill="{fill}" rx="2" ry="2">'
                f'<title>{_escape(label)}: {val:.3f} bits</title></rect>'
            )
            # Value text
            parts.append(
                f'<text x="{label_w + bar_w + 4}" y="{y + self.bar_height / 2 + 3}" '
                f'font-family="sans-serif" font-size="8" fill="#555">'
                f'{val:.2f}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)

    def _cumulative_line(
        self, labels: List[str], values: List[float], title: str
    ) -> str:
        if not values:
            return _empty_agg_svg()
        max_val = max(values) if values else 1.0
        max_val = max(max_val, 1e-9)
        margin_l, margin_t, margin_r, margin_b = 60, 30, 20, 60
        plot_w = max(len(values) * 60, 200)
        plot_h = 180
        total_w = margin_l + plot_w + margin_r
        total_h = margin_t + plot_h + margin_b

        parts: List[str] = []
        parts.append(
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{total_w}" height="{total_h}" '
            f'viewBox="0 0 {total_w} {total_h}">'
        )
        parts.append(f'<rect width="{total_w}" height="{total_h}" fill="white"/>')
        parts.append(
            f'<text x="{total_w / 2}" y="18" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#333" '
            f'font-weight="bold">{_escape(title)}</text>'
        )

        n = len(values)
        # Axes
        ax_x = margin_l
        ax_y = margin_t + plot_h
        parts.append(
            f'<line x1="{ax_x}" y1="{margin_t}" x2="{ax_x}" y2="{ax_y}" '
            f'stroke="#333" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{ax_x}" y1="{ax_y}" x2="{ax_x + plot_w}" y2="{ax_y}" '
            f'stroke="#333" stroke-width="1"/>'
        )

        # Gridlines and y-axis labels
        n_grid = 4
        for gi in range(n_grid + 1):
            gy_val = max_val * gi / n_grid
            gy = margin_t + plot_h - (gi / n_grid) * plot_h
            parts.append(
                f'<line x1="{ax_x}" y1="{gy}" x2="{ax_x + plot_w}" y2="{gy}" '
                f'stroke="#eee" stroke-width="0.5"/>'
            )
            parts.append(
                f'<text x="{ax_x - 4}" y="{gy + 3}" text-anchor="end" '
                f'font-family="sans-serif" font-size="8" fill="#666">'
                f'{gy_val:.1f}</text>'
            )

        # Data points and path
        points: List[str] = []
        for i, val in enumerate(values):
            px = ax_x + (i / max(n - 1, 1)) * plot_w
            py = margin_t + plot_h - (val / max_val) * plot_h
            points.append(f"{px:.1f},{py:.1f}")
            # Dot
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#dc3545">'
                f'<title>{_escape(labels[i])}: {val:.2f} bits</title></circle>'
            )

        # Polyline
        if points:
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="#dc3545" stroke-width="1.5"/>'
            )

        # X-axis labels
        for i, lab in enumerate(labels):
            px = ax_x + (i / max(n - 1, 1)) * plot_w
            parts.append(
                f'<text x="{px}" y="{ax_y + 14}" text-anchor="end" '
                f'font-family="sans-serif" font-size="8" fill="#333" '
                f'transform="rotate(-40 {px} {ax_y + 14})">'
                f'{_escape(lab[:12])}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)


def _empty_agg_svg() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60">'
        '<text x="10" y="30" font-family="sans-serif" font-size="12" '
        'fill="#999">(no data)</text></svg>'
    )
