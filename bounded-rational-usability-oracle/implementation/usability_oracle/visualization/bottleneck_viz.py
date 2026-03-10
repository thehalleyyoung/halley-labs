"""
usability_oracle.visualization.bottleneck_viz — Bottleneck heatmaps.

Renders bottleneck detection results as text heatmaps, severity
rankings, and bottleneck-interaction matrices.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class BottleneckVisualizer:
    """Visualize bottleneck detection results."""

    def __init__(self, width: int = 60) -> None:
        self._width = width

    # ------------------------------------------------------------------
    # Bottleneck ranking
    # ------------------------------------------------------------------

    def render_ranking(
        self,
        bottlenecks: list[dict[str, Any]],
        title: str = "Bottleneck Ranking",
    ) -> str:
        """Render a ranked list of detected bottlenecks.

        Each bottleneck dict: type, severity, location, description, cost.
        """
        if not bottlenecks:
            return f"{title}: No bottlenecks detected ✓"

        sorted_bns = sorted(bottlenecks, key=lambda b: -b.get("severity", 0))
        max_type = max(len(b.get("type", "?")) for b in sorted_bns)
        max_loc = max(len(str(b.get("location", "?"))) for b in sorted_bns)

        lines = [f"{title} ({len(sorted_bns)} found):", ""]
        lines.append(
            f"  # {'Type':<{max_type}}  {'Location':<{max_loc}}  {'Severity':>8}  {'Cost':>8}  Description"
        )
        lines.append(
            f"  ─ {'─' * max_type}  {'─' * max_loc}  {'─' * 8}  {'─' * 8}  {'─' * 20}"
        )

        for i, b in enumerate(sorted_bns, 1):
            btype = b.get("type", "unknown")
            location = str(b.get("location", "?"))
            severity = b.get("severity", 0)
            cost = b.get("cost", 0)
            desc = b.get("description", "")

            sev_bar = "█" * int(severity * 10) + "░" * (10 - int(severity * 10))
            icon = "🔴" if severity > 0.7 else ("🟡" if severity > 0.4 else "🟢")

            lines.append(
                f"  {i} {btype:<{max_type}}  {location:<{max_loc}}  {icon}{sev_bar}  {cost:>8.3f}  {desc[:40]}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def render_heatmap(
        self,
        matrix: np.ndarray,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        title: str = "Bottleneck Heatmap",
    ) -> str:
        """Render a matrix as a text heatmap."""
        M = np.asarray(matrix, dtype=float)
        if M.ndim != 2 or M.size == 0:
            return f"{title}: (no data)"

        rows, cols = M.shape
        r_labels = row_labels or [f"r{i}" for i in range(rows)]
        c_labels = col_labels or [f"c{j}" for j in range(cols)]

        m_min, m_max = float(M.min()), float(M.max())
        m_range = m_max - m_min if m_max > m_min else 1.0

        chars = " ░▒▓█"
        max_rl = max(len(l) for l in r_labels)
        col_w = max(3, max(len(l) for l in c_labels))

        # Header
        lines = [f"{title}:", ""]
        header = " " * (max_rl + 2) + " ".join(f"{l:>{col_w}}" for l in c_labels)
        lines.append(header)

        for i in range(rows):
            row_chars = []
            for j in range(cols):
                level = int(((M[i, j] - m_min) / m_range) * (len(chars) - 1))
                level = max(0, min(len(chars) - 1, level))
                cell = chars[level] * col_w
                row_chars.append(cell)
            lines.append(f"  {r_labels[i]:>{max_rl}} │{' '.join(row_chars)}│")

        lines.append(f"\n  Legend: ' '={m_min:.2f}  ░  ▒  ▓  █={m_max:.2f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Bottleneck interaction matrix
    # ------------------------------------------------------------------

    def render_interaction_matrix(
        self,
        bottleneck_types: list[str],
        interactions: np.ndarray,
        title: str = "Bottleneck Interactions",
    ) -> str:
        """Render interaction strength between bottleneck types."""
        n = len(bottleneck_types)
        I = np.asarray(interactions, dtype=float)
        if I.shape != (n, n):
            return f"{title}: shape mismatch"

        max_name = max(len(t) for t in bottleneck_types) if bottleneck_types else 0
        col_w = max(4, max_name)

        lines = [f"{title}:", ""]
        header = " " * (max_name + 2) + " ".join(f"{t[:col_w]:>{col_w}}" for t in bottleneck_types)
        lines.append(header)

        for i in range(n):
            cells = []
            for j in range(n):
                val = I[i, j]
                if i == j:
                    cells.append(f"{'─':>{col_w}}")
                elif val > 0.5:
                    cells.append(f"{'●':>{col_w}}")
                elif val > 0.1:
                    cells.append(f"{'◐':>{col_w}}")
                else:
                    cells.append(f"{'○':>{col_w}}")
            lines.append(f"  {bottleneck_types[i]:>{max_name}} │{' '.join(cells)}│")

        lines.append(f"\n  ● strong (>0.5)  ◐ moderate (>0.1)  ○ weak")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cost-per-bottleneck
    # ------------------------------------------------------------------

    def render_cost_attribution(
        self,
        attributions: dict[str, float],
        title: str = "Cost Attribution by Bottleneck",
    ) -> str:
        """Render how much each bottleneck type contributes to total cost."""
        if not attributions:
            return f"{title}: (no data)"

        total = sum(attributions.values())
        max_name = max(len(k) for k in attributions)
        max_val = max(attributions.values()) if attributions else 1

        lines = [f"{title} (total cost: {total:.3f}):", ""]

        for name, cost in sorted(attributions.items(), key=lambda x: -x[1]):
            bar_len = int(cost / max_val * 30) if max_val > 0 else 0
            pct = cost / total * 100 if total > 0 else 0
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"  {name:<{max_name}} │{bar}│ {cost:.3f} ({pct:.1f}%)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Bottleneck timeline
    # ------------------------------------------------------------------

    def render_timeline(
        self,
        events: list[dict[str, Any]],
        title: str = "Bottleneck Timeline",
    ) -> str:
        """Render bottlenecks along a task timeline.

        Each event dict: time, type, severity, description.
        """
        if not events:
            return f"{title}: (no events)"

        sorted_events = sorted(events, key=lambda e: e.get("time", 0))
        t_min = sorted_events[0].get("time", 0)
        t_max = sorted_events[-1].get("time", 0)
        t_range = t_max - t_min if t_max > t_min else 1

        lines = [f"{title}:", ""]

        # Timeline bar
        timeline = [" "] * self._width
        for e in sorted_events:
            t = e.get("time", 0)
            pos = int(((t - t_min) / t_range) * (self._width - 1))
            pos = max(0, min(self._width - 1, pos))
            sev = e.get("severity", 0)
            timeline[pos] = "█" if sev > 0.7 else ("▓" if sev > 0.4 else "▒")

        lines.append(f"  {''.join(timeline)}")
        lines.append(f"  {t_min:.1f}s{' ' * max(0, self._width - 12)}{t_max:.1f}s")
        lines.append("")

        # Event list
        for e in sorted_events:
            t = e.get("time", 0)
            btype = e.get("type", "?")
            sev = e.get("severity", 0)
            desc = e.get("description", "")
            icon = "🔴" if sev > 0.7 else ("🟡" if sev > 0.4 else "🟢")
            lines.append(f"  {icon} t={t:.2f}s  {btype}: {desc}")

        return "\n".join(lines)
