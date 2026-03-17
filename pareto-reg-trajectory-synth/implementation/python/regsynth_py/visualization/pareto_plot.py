"""Pareto frontier visualization using pure SVG generation.

Produces self-contained SVG strings for 2-D Pareto plots, 3-D projections,
parallel-coordinate charts, and radar charts.
"""

from __future__ import annotations

import math
from typing import Optional

from regsynth_py.visualization.svg_utils import (
    PALETTE,
    SVGElement,
    color_interpolate,
    render_svg,
    scale_linear,
    svg_circle,
    svg_document,
    svg_group,
    svg_line,
    svg_path,
    svg_polygon,
    svg_polyline,
    svg_rect,
    svg_text,
    svg_title,
)


class ParetoPlotter:
    """Generate SVG visualizations of Pareto frontiers and multi-objective data."""

    def __init__(self, width: int = 800, height: int = 600, margin: int = 60) -> None:
        self.width = width
        self.height = height
        self.margin = margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_2d(
        self,
        points: list[tuple[float, float]],
        labels: Optional[list[str]] = None,
        pareto_front: Optional[list[int]] = None,
        title: str = "Pareto Frontier",
        x_label: str = "Cost",
        y_label: str = "Coverage",
    ) -> str:
        if not points:
            return render_svg(svg_document(self.width, self.height))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_pad = max((x_max - x_min) * 0.05, 1e-9)
        y_pad = max((y_max - y_min) * 0.05, 1e-9)
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        svg = svg_document(self.width, self.height)
        svg.add_child(svg_rect(0, 0, self.width, self.height, "#ffffff"))

        plot_group = svg_group()
        svg.add_child(plot_group)

        m = self.margin
        pw = self.width - 2 * m
        ph = self.height - 2 * m

        x_ticks = self._nice_ticks(x_min, x_max)
        y_ticks = self._nice_ticks(y_min, y_max)
        self._draw_grid(plot_group, x_ticks, y_ticks, x_min, x_max, y_min, y_max, m, pw, ph)
        self._draw_axes(plot_group, x_ticks, y_ticks, x_min, x_max, y_min, y_max, m, pw, ph)

        pareto_set: set[int] = set(pareto_front) if pareto_front else set()

        # Pareto frontier line
        if pareto_front and len(pareto_front) >= 2:
            front_pts = sorted(pareto_front, key=lambda i: points[i][0])
            line_pts = [
                (
                    m + scale_linear(points[i][0], x_min, x_max, 0, pw),
                    m + ph - scale_linear(points[i][1], y_min, y_max, 0, ph),
                )
                for i in front_pts
            ]
            plot_group.add_child(svg_polyline(line_pts, PALETTE[1], stroke_width=2))

        # Points
        for idx, (px, py) in enumerate(points):
            sx = m + scale_linear(px, x_min, x_max, 0, pw)
            sy = m + ph - scale_linear(py, y_min, y_max, 0, ph)
            is_pareto = idx in pareto_set
            color = PALETTE[1] if is_pareto else PALETTE[0]
            r = 6 if is_pareto else 4
            circ = svg_circle(sx, sy, r, color, stroke="#fff")
            tip_text = f"({self._format_tick(px)}, {self._format_tick(py)})"
            if labels and idx < len(labels):
                tip_text = f"{labels[idx]}: {tip_text}"
            circ.add_child(svg_title(tip_text))
            plot_group.add_child(circ)

        # Title
        svg.add_child(svg_text(
            self.width / 2, 25, title, font_size=16,
            fill="#333", anchor="middle", font_weight="bold",
        ))
        # Axis labels
        svg.add_child(svg_text(
            self.width / 2, self.height - 10, x_label,
            font_size=13, fill="#555", anchor="middle",
        ))
        svg.add_child(svg_text(
            15, self.height / 2, y_label,
            font_size=13, fill="#555", anchor="middle", rotate=-90,
        ))

        # Legend
        legend_items = [("All points", PALETTE[0]), ("Pareto-optimal", PALETTE[1])]
        self._draw_legend(svg, legend_items, self.width - m - 130, m + 10)

        return render_svg(svg)

    def plot_3d_projection(
        self,
        points: list[tuple[float, float, float]],
        labels: Optional[list[str]] = None,
        title: str = "3D Pareto",
    ) -> str:
        if not points:
            return render_svg(svg_document(self.width, self.height))

        svg = svg_document(self.width, self.height)
        svg.add_child(svg_rect(0, 0, self.width, self.height, "#ffffff"))

        cx, cy = self.width / 2, self.height / 2 + 30
        scale = min(self.width, self.height) * 0.28

        # Isometric projection angles
        angle_x = math.radians(210)
        angle_y = math.radians(330)

        def project(x: float, y: float, z: float) -> tuple[float, float]:
            px = cx + scale * (x * math.cos(angle_x) + y * math.cos(angle_y))
            py = cy + scale * (x * math.sin(angle_x) + y * math.sin(angle_y) - z)
            return px, py

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        def norm(v: float, lo: float, hi: float) -> float:
            return (v - lo) / (hi - lo) if hi > lo else 0.5

        # Draw three axes
        origin = project(0, 0, 0)
        ax_x = project(1, 0, 0)
        ax_y = project(0, 1, 0)
        ax_z = project(0, 0, 1)
        for ax, lbl in [(ax_x, "X"), (ax_y, "Y"), (ax_z, "Z")]:
            svg.add_child(svg_line(origin[0], origin[1], ax[0], ax[1], "#999", stroke_width=1, dash="4,2"))
            svg.add_child(svg_text(ax[0] + 5, ax[1], lbl, font_size=12, fill="#666"))

        # Plot points, colored by Z
        for idx, (px, py, pz) in enumerate(points):
            nx = norm(px, x_min, x_max)
            ny = norm(py, y_min, y_max)
            nz = norm(pz, z_min, z_max)
            sx, sy = project(nx, ny, nz)
            color = color_interpolate("#4e79a7", "#e15759", nz)
            circ = svg_circle(sx, sy, 5, color, stroke="#fff", opacity=0.85)
            tip = f"({self._format_tick(px)}, {self._format_tick(py)}, {self._format_tick(pz)})"
            if labels and idx < len(labels):
                tip = f"{labels[idx]}: {tip}"
            circ.add_child(svg_title(tip))
            svg.add_child(circ)

        svg.add_child(svg_text(
            self.width / 2, 25, title, font_size=16,
            fill="#333", anchor="middle", font_weight="bold",
        ))
        return render_svg(svg)

    def plot_parallel_coordinates(
        self,
        points: list[list[float]],
        dimension_names: list[str],
        highlight: Optional[list[int]] = None,
        title: str = "Parallel Coordinates",
    ) -> str:
        if not points or not dimension_names:
            return render_svg(svg_document(self.width, self.height))

        ndims = len(dimension_names)
        svg = svg_document(self.width, self.height)
        svg.add_child(svg_rect(0, 0, self.width, self.height, "#ffffff"))

        m = self.margin
        pw = self.width - 2 * m
        ph = self.height - 2 * m - 20
        top = m + 30

        dim_mins = [min(p[d] for p in points) for d in range(ndims)]
        dim_maxs = [max(p[d] for p in points) for d in range(ndims)]

        axis_xs = [m + pw * i / max(ndims - 1, 1) for i in range(ndims)]

        # Draw axes
        for i, ax_x in enumerate(axis_xs):
            svg.add_child(svg_line(ax_x, top, ax_x, top + ph, "#ccc"))
            svg.add_child(svg_text(ax_x, top - 8, dimension_names[i], font_size=11, fill="#555", anchor="middle"))
            svg.add_child(svg_text(ax_x, top + ph + 15, self._format_tick(dim_mins[i]), font_size=9, fill="#999", anchor="middle"))
            svg.add_child(svg_text(ax_x, top - 20, self._format_tick(dim_maxs[i]), font_size=9, fill="#999", anchor="middle"))

        highlight_set = set(highlight) if highlight else set()

        # Draw lines (non-highlighted first)
        for idx, pt in enumerate(points):
            if idx in highlight_set:
                continue
            coords = []
            for d in range(ndims):
                y = top + ph - scale_linear(pt[d], dim_mins[d], dim_maxs[d], 0, ph)
                coords.append((axis_xs[d], y))
            svg.add_child(svg_polyline(coords, PALETTE[0], stroke_width=1))

        # Draw highlighted lines on top
        for idx in sorted(highlight_set):
            if idx >= len(points):
                continue
            pt = points[idx]
            coords = []
            for d in range(ndims):
                y = top + ph - scale_linear(pt[d], dim_mins[d], dim_maxs[d], 0, ph)
                coords.append((axis_xs[d], y))
            svg.add_child(svg_polyline(coords, PALETTE[1], stroke_width=2.5))

        svg.add_child(svg_text(
            self.width / 2, 20, title, font_size=16,
            fill="#333", anchor="middle", font_weight="bold",
        ))
        return render_svg(svg)

    def plot_radar(
        self,
        values: dict[str, list[float]],
        labels: list[str],
        title: str = "Radar Chart",
    ) -> str:
        if not values or not labels:
            return render_svg(svg_document(self.width, self.height))

        n = len(labels)
        svg = svg_document(self.width, self.height)
        svg.add_child(svg_rect(0, 0, self.width, self.height, "#ffffff"))

        cx = self.width / 2
        cy = self.height / 2 + 15
        radius = min(self.width, self.height) * 0.32

        angle_step = 2 * math.pi / n

        def polar(i: int, r: float) -> tuple[float, float]:
            a = -math.pi / 2 + i * angle_step
            return cx + r * math.cos(a), cy + r * math.sin(a)

        # Background concentric polygons
        for level in range(1, 6):
            r = radius * level / 5
            pts = [polar(i, r) for i in range(n)]
            svg.add_child(svg_polygon(pts, "none", stroke="#ddd"))

        # Axis lines and labels
        for i in range(n):
            ex, ey = polar(i, radius)
            svg.add_child(svg_line(cx, cy, ex, ey, "#ccc"))
            lx, ly = polar(i, radius + 18)
            svg.add_child(svg_text(lx, ly + 4, labels[i], font_size=10, fill="#555", anchor="middle"))

        # Data traces
        for trace_idx, (name, vals) in enumerate(values.items()):
            color = PALETTE[trace_idx % len(PALETTE)]
            pts = []
            for i in range(n):
                v = vals[i] if i < len(vals) else 0
                v = max(0.0, min(1.0, v))
                pts.append(polar(i, radius * v))
            svg.add_child(svg_polygon(pts, color + "33", stroke=color))
            for px, py in pts:
                svg.add_child(svg_circle(px, py, 3, color))

        # Title
        svg.add_child(svg_text(
            self.width / 2, 25, title, font_size=16,
            fill="#333", anchor="middle", font_weight="bold",
        ))

        # Legend
        legend_items = [(name, PALETTE[i % len(PALETTE)]) for i, name in enumerate(values)]
        self._draw_legend(svg, legend_items, self.width - self.margin - 120, self.margin + 5)

        return render_svg(svg)

    # ------------------------------------------------------------------
    # Pareto computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pareto_front(
        points: list[tuple], minimize: Optional[list[bool]] = None,
    ) -> list[int]:
        """Return indices of Pareto-optimal points.

        *minimize* is a list of booleans per dimension (True = minimize that
        objective, False = maximize).  Defaults to minimizing all dimensions.
        """
        n = len(points)
        if n == 0:
            return []
        ndim = len(points[0])
        if minimize is None:
            minimize = [True] * ndim

        signs = [1 if m else -1 for m in minimize]
        adjusted = [tuple(signs[d] * points[i][d] for d in range(ndim)) for i in range(n)]

        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if all(adjusted[j][d] <= adjusted[i][d] for d in range(ndim)) and any(
                    adjusted[j][d] < adjusted[i][d] for d in range(ndim)
                ):
                    dominated[i] = True
                    break
        return [i for i in range(n) if not dominated[i]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_axes(
        self, svg: SVGElement,
        x_ticks: list[float], y_ticks: list[float],
        x_min: float, x_max: float, y_min: float, y_max: float,
        m: int, pw: float, ph: float,
    ) -> None:
        # X axis
        svg.add_child(svg_line(m, m + ph, m + pw, m + ph, "#333"))
        for v in x_ticks:
            sx = m + scale_linear(v, x_min, x_max, 0, pw)
            svg.add_child(svg_line(sx, m + ph, sx, m + ph + 5, "#333"))
            svg.add_child(svg_text(sx, m + ph + 18, self._format_tick(v), font_size=10, fill="#666", anchor="middle"))
        # Y axis
        svg.add_child(svg_line(m, m, m, m + ph, "#333"))
        for v in y_ticks:
            sy = m + ph - scale_linear(v, y_min, y_max, 0, ph)
            svg.add_child(svg_line(m - 5, sy, m, sy, "#333"))
            svg.add_child(svg_text(m - 8, sy + 4, self._format_tick(v), font_size=10, fill="#666", anchor="end"))

    def _draw_grid(
        self, svg: SVGElement,
        x_ticks: list[float], y_ticks: list[float],
        x_min: float, x_max: float, y_min: float, y_max: float,
        m: int, pw: float, ph: float,
    ) -> None:
        for v in x_ticks:
            sx = m + scale_linear(v, x_min, x_max, 0, pw)
            svg.add_child(svg_line(sx, m, sx, m + ph, "#eee"))
        for v in y_ticks:
            sy = m + ph - scale_linear(v, y_min, y_max, 0, ph)
            svg.add_child(svg_line(m, sy, m + pw, sy, "#eee"))

    @staticmethod
    def _draw_legend(
        svg: SVGElement, items: list[tuple[str, str]], x: float, y: float,
    ) -> None:
        bg = svg_rect(x - 5, y - 5, 140, len(items) * 22 + 10, "#fff", stroke="#ddd", rx=4, opacity=0.9)
        svg.add_child(bg)
        for i, (label, color) in enumerate(items):
            iy = y + i * 22
            svg.add_child(svg_circle(x + 8, iy + 8, 5, color))
            svg.add_child(svg_text(x + 20, iy + 12, label, font_size=11, fill="#555"))

    @staticmethod
    def _format_tick(value: float) -> str:
        if abs(value) >= 1e6:
            return f"{value / 1e6:.1f}M"
        if abs(value) >= 1e3:
            return f"{value / 1e3:.1f}K"
        if value == int(value):
            return str(int(value))
        if abs(value) < 0.01:
            return f"{value:.2e}"
        return f"{value:.2f}"

    @staticmethod
    def _nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
        rng = hi - lo
        if rng <= 0:
            return [lo]
        rough = rng / target
        exponent = math.floor(math.log10(rough))
        fraction = rough / (10 ** exponent)
        if fraction <= 1.5:
            nice = 1
        elif fraction <= 3:
            nice = 2
        elif fraction <= 7:
            nice = 5
        else:
            nice = 10
        step = nice * (10 ** exponent)
        start = math.ceil(lo / step) * step
        ticks: list[float] = []
        v = start
        while v <= hi + step * 0.001:
            ticks.append(round(v, 10))
            v += step
        return ticks

    def save(self, svg_content: str, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(svg_content)
