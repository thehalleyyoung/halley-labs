"""Conflict relationship graph visualization using pure SVG.

Generates force-directed graphs, conflict heat-maps, and chord diagrams
for regulatory-framework conflict analysis.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from regsynth_py.visualization.svg_utils import (
    PALETTE,
    RISK_COLORS,
    SVGElement,
    color_interpolate,
    render_svg,
    scale_linear,
    svg_circle,
    svg_document,
    svg_group,
    svg_line,
    svg_path,
    svg_rect,
    svg_text,
    svg_title,
)

_CONFLICT_TYPE_COLORS: dict[str, str] = {
    "direct": "#d62728",
    "indirect": "#ff7f0e",
    "potential": "#ffbb33",
    "resolved": "#2ca02c",
    "partial": "#9467bd",
}


class ConflictGraphPlotter:
    """Visualize regulatory-framework conflicts as graphs, matrices, and chords."""

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width = width
        self.height = height

    # ------------------------------------------------------------------
    # Force-directed conflict graph
    # ------------------------------------------------------------------

    def plot_conflict_graph(
        self,
        nodes: list[dict],
        edges: list[dict],
        title: str = "Conflict Graph",
    ) -> str:
        """Render a node-link conflict graph.

        *nodes*: ``{id, label, type, size}``
        *edges*: ``{source, target, weight, conflict_type, label}``
        """
        if not nodes:
            return render_svg(svg_document(self.width, self.height))

        positions = self._spring_layout(nodes, edges)

        svg = svg_document(self.width, self.height)
        svg.add_child(svg_rect(0, 0, self.width, self.height, "#ffffff"))

        # Determine node degrees for sizing
        degree: dict[str, int] = {n["id"]: 0 for n in nodes}
        for e in edges:
            degree[e["source"]] = degree.get(e["source"], 0) + 1
            degree[e["target"]] = degree.get(e["target"], 0) + 1
        max_deg = max(degree.values()) if degree else 1

        # Draw edges first (below nodes)
        for e in edges:
            if e["source"] not in positions or e["target"] not in positions:
                continue
            x1, y1 = positions[e["source"]]
            x2, y2 = positions[e["target"]]
            w = max(1, float(e.get("weight", 1)))
            color = _CONFLICT_TYPE_COLORS.get(e.get("conflict_type", ""), "#999")
            sw = min(1 + w, 6)
            ln = svg_line(x1, y1, x2, y2, color, stroke_width=sw)
            tip = e.get("label", f"{e['source']}↔{e['target']}")
            ln.add_child(svg_title(tip))
            svg.add_child(ln)

            # Edge label at midpoint
            if e.get("label"):
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                svg.add_child(svg_text(mx, my - 4, e["label"], font_size=8, fill="#888", anchor="middle"))

        # Draw nodes
        node_map = {n["id"]: n for n in nodes}
        type_colors: dict[str, str] = {}
        for n in nodes:
            ntype = n.get("type", "default")
            if ntype not in type_colors:
                type_colors[ntype] = PALETTE[len(type_colors) % len(PALETTE)]

        for nid, (px, py) in positions.items():
            nd = node_map.get(nid, {})
            base_r = float(nd.get("size", 10))
            deg_scale = 1 + 0.7 * (degree.get(nid, 0) / max(max_deg, 1))
            r = base_r * deg_scale
            color = type_colors.get(nd.get("type", "default"), PALETTE[0])
            circ = svg_circle(px, py, r, color, stroke="#fff")
            circ.add_child(svg_title(nd.get("label", nid)))
            svg.add_child(circ)
            svg.add_child(svg_text(px, py + r + 14, nd.get("label", nid), font_size=10, fill="#444", anchor="middle"))

        # Legend for conflict types used
        used_types = {e.get("conflict_type", "") for e in edges if e.get("conflict_type")}
        legend_items = [(ct, _CONFLICT_TYPE_COLORS.get(ct, "#999")) for ct in sorted(used_types)]
        if legend_items:
            self._draw_legend(svg, legend_items, self.width - 140, 40)

        svg.add_child(svg_text(self.width / 2, 25, title, font_size=16, fill="#333", anchor="middle", font_weight="bold"))
        return render_svg(svg)

    # ------------------------------------------------------------------
    # Conflict matrix heatmap
    # ------------------------------------------------------------------

    def plot_conflict_matrix(
        self,
        matrix: dict[tuple[str, str], float],
        labels: list[str],
        title: str = "Conflict Matrix",
    ) -> str:
        n = len(labels)
        if n == 0:
            return render_svg(svg_document(self.width, self.height))

        max_val = max(matrix.values()) if matrix else 1
        min_val = min(matrix.values()) if matrix else 0

        label_offset = 120
        cell = min((self.width - label_offset - 60) / max(n, 1), (self.height - label_offset - 80) / max(n, 1))
        cell = max(cell, 18)

        total_w = label_offset + n * cell + 80
        total_h = label_offset + n * cell + 60

        svg = svg_document(int(total_w), int(total_h))
        svg.add_child(svg_rect(0, 0, total_w, total_h, "#ffffff"))

        for row_i, r_label in enumerate(labels):
            svg.add_child(svg_text(label_offset - 8, label_offset + row_i * cell + cell / 2 + 4, r_label, font_size=10, fill="#333", anchor="end"))
            for col_i, c_label in enumerate(labels):
                x = label_offset + col_i * cell
                y = label_offset + row_i * cell
                val = matrix.get((r_label, c_label), 0)
                t = (val - min_val) / max(max_val - min_val, 1e-9)
                color = color_interpolate("#f7fbff", "#d62728", t)
                rect = svg_rect(x, y, cell - 1, cell - 1, color, rx=2)
                rect.add_child(svg_title(f"{r_label} ↔ {c_label}: {val}"))
                svg.add_child(rect)
                if cell > 24:
                    svg.add_child(svg_text(x + cell / 2, y + cell / 2 + 3, str(int(val)) if val == int(val) else f"{val:.1f}", font_size=8, fill="#333" if t < 0.5 else "#fff", anchor="middle"))

        # Column labels (rotated)
        for col_i, c_label in enumerate(labels):
            svg.add_child(svg_text(label_offset + col_i * cell + cell / 2, label_offset - 8, c_label, font_size=10, fill="#333", anchor="end", rotate=-45))

        # Color scale legend
        legend_x = label_offset + n * cell + 15
        legend_h = min(n * cell, 200)
        for li in range(20):
            t = li / 19
            lc = color_interpolate("#f7fbff", "#d62728", t)
            ly = label_offset + (1 - t) * legend_h
            svg.add_child(svg_rect(legend_x, ly, 14, legend_h / 19 + 1, lc))
        svg.add_child(svg_text(legend_x + 20, label_offset + 10, f"{max_val:.0f}", font_size=9, fill="#666"))
        svg.add_child(svg_text(legend_x + 20, label_offset + legend_h, f"{min_val:.0f}", font_size=9, fill="#666"))

        svg.add_child(svg_text(total_w / 2, 25, title, font_size=16, fill="#333", anchor="middle", font_weight="bold"))
        return render_svg(svg)

    # ------------------------------------------------------------------
    # Chord diagram
    # ------------------------------------------------------------------

    def plot_conflict_chord(
        self,
        frameworks: list[str],
        conflicts: list[dict],
        title: str = "Conflict Chord",
    ) -> str:
        """Circular chord diagram of framework conflicts.

        *conflicts*: ``{source, target, count, severity}``
        """
        n = len(frameworks)
        if n == 0:
            return render_svg(svg_document(self.width, self.height))

        svg = svg_document(self.width, self.height)
        svg.add_child(svg_rect(0, 0, self.width, self.height, "#ffffff"))

        cx, cy = self.width / 2, self.height / 2 + 10
        outer_r = min(self.width, self.height) * 0.36
        inner_r = outer_r * 0.88

        fw_idx = {f: i for i, f in enumerate(frameworks)}
        angle_step = 2 * math.pi / n
        gap = 0.04  # radians gap between arcs

        # Draw outer arcs and labels
        for i, fw in enumerate(frameworks):
            a_start = i * angle_step + gap / 2
            a_end = (i + 1) * angle_step - gap / 2
            color = PALETTE[i % len(PALETTE)]

            # Outer arc as path
            x1o = cx + outer_r * math.cos(a_start)
            y1o = cy + outer_r * math.sin(a_start)
            x2o = cx + outer_r * math.cos(a_end)
            y2o = cy + outer_r * math.sin(a_end)
            x1i = cx + inner_r * math.cos(a_end)
            y1i = cy + inner_r * math.sin(a_end)
            x2i = cx + inner_r * math.cos(a_start)
            y2i = cy + inner_r * math.sin(a_start)
            large = 1 if (a_end - a_start) > math.pi else 0
            d = (
                f"M{x1o:.1f},{y1o:.1f} "
                f"A{outer_r:.1f},{outer_r:.1f} 0 {large},1 {x2o:.1f},{y2o:.1f} "
                f"L{x1i:.1f},{y1i:.1f} "
                f"A{inner_r:.1f},{inner_r:.1f} 0 {large},0 {x2i:.1f},{y2i:.1f} Z"
            )
            arc = svg_path(d, fill=color, stroke="none")
            arc.add_child(svg_title(fw))
            svg.add_child(arc)

            # Label
            mid_a = (a_start + a_end) / 2
            lx = cx + (outer_r + 16) * math.cos(mid_a)
            ly = cy + (outer_r + 16) * math.sin(mid_a)
            anchor = "start" if math.cos(mid_a) >= 0 else "end"
            svg.add_child(svg_text(lx, ly + 4, fw, font_size=10, fill="#444", anchor=anchor))

        # Draw chords
        max_count = max((c.get("count", 1) for c in conflicts), default=1)
        for c in conflicts:
            si = fw_idx.get(c["source"])
            ti = fw_idx.get(c["target"])
            if si is None or ti is None or si == ti:
                continue
            a_s = (si + 0.5) * angle_step
            a_t = (ti + 0.5) * angle_step
            sx = cx + inner_r * math.cos(a_s)
            sy = cy + inner_r * math.sin(a_s)
            tx = cx + inner_r * math.cos(a_t)
            ty = cy + inner_r * math.sin(a_t)
            count = c.get("count", 1)
            sw = max(1, 6 * count / max(max_count, 1))
            color = self._severity_color(c.get("severity", "medium"))
            d = f"M{sx:.1f},{sy:.1f} Q{cx:.1f},{cy:.1f} {tx:.1f},{ty:.1f}"
            chord = svg_path(d, fill="none", stroke=color, stroke_width=sw)
            chord.set_attr("opacity", "0.55")
            chord.add_child(svg_title(f"{c['source']} ↔ {c['target']}: {count}"))
            svg.add_child(chord)

        svg.add_child(svg_text(self.width / 2, 25, title, font_size=16, fill="#333", anchor="middle", font_weight="bold"))
        return render_svg(svg)

    # ------------------------------------------------------------------
    # Spring layout
    # ------------------------------------------------------------------

    def _spring_layout(
        self,
        nodes: list[dict],
        edges: list[dict],
        iterations: int = 50,
    ) -> dict[str, tuple[float, float]]:
        rng = random.Random(42)
        m = 80
        w = self.width - 2 * m
        h = self.height - 2 * m

        pos: dict[str, list[float]] = {}
        for n in nodes:
            pos[n["id"]] = [m + rng.random() * w, m + 40 + rng.random() * h]

        ids = list(pos.keys())
        k_repulse = 8000.0
        k_attract = 0.005
        damping = 0.9
        vel: dict[str, list[float]] = {nid: [0.0, 0.0] for nid in ids}

        for _ in range(iterations):
            forces: dict[str, list[float]] = {nid: [0.0, 0.0] for nid in ids}

            # Repulsion
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    dx = pos[ids[i]][0] - pos[ids[j]][0]
                    dy = pos[ids[i]][1] - pos[ids[j]][1]
                    dist = math.sqrt(dx * dx + dy * dy) + 0.01
                    f = k_repulse / (dist * dist)
                    fx, fy = f * dx / dist, f * dy / dist
                    forces[ids[i]][0] += fx
                    forces[ids[i]][1] += fy
                    forces[ids[j]][0] -= fx
                    forces[ids[j]][1] -= fy

            # Attraction along edges
            for e in edges:
                s, t = e["source"], e["target"]
                if s not in pos or t not in pos:
                    continue
                dx = pos[s][0] - pos[t][0]
                dy = pos[s][1] - pos[t][1]
                dist = math.sqrt(dx * dx + dy * dy) + 0.01
                f = k_attract * dist
                fx, fy = f * dx / dist, f * dy / dist
                forces[s][0] -= fx
                forces[s][1] -= fy
                forces[t][0] += fx
                forces[t][1] += fy

            # Center gravity
            for nid in ids:
                cx_off = (m + w / 2) - pos[nid][0]
                cy_off = (m + 40 + h / 2) - pos[nid][1]
                forces[nid][0] += cx_off * 0.001
                forces[nid][1] += cy_off * 0.001

            for nid in ids:
                vel[nid][0] = (vel[nid][0] + forces[nid][0]) * damping
                vel[nid][1] = (vel[nid][1] + forces[nid][1]) * damping
                pos[nid][0] = max(m, min(m + w, pos[nid][0] + vel[nid][0]))
                pos[nid][1] = max(m + 40, min(m + 40 + h, pos[nid][1] + vel[nid][1]))

        return {nid: (p[0], p[1]) for nid, p in pos.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _severity_color(severity: str) -> str:
        mapping = {
            "critical": RISK_COLORS["critical"],
            "high": RISK_COLORS["high"],
            "medium": RISK_COLORS["medium"],
            "low": RISK_COLORS["low"],
        }
        return mapping.get(severity, "#999")

    @staticmethod
    def _draw_legend(
        svg: SVGElement, items: list[tuple[str, str]], x: float, y: float,
    ) -> None:
        bg = svg_rect(x - 5, y - 5, 130, len(items) * 20 + 10, "#fff", stroke="#ddd", rx=4, opacity=0.92)
        svg.add_child(bg)
        for i, (label, color) in enumerate(items):
            iy = y + i * 20
            svg.add_child(svg_rect(x, iy, 12, 12, color, rx=2))
            svg.add_child(svg_text(x + 18, iy + 10, label, font_size=10, fill="#555"))

    def save(self, svg_content: str, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(svg_content)
