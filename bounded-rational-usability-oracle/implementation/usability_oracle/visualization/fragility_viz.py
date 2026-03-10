"""
usability_oracle.visualization.fragility_viz — Fragility and cliff visualization.

Renders fragility analysis results including cliff detection, parameter
sensitivity landscapes, and regression risk assessments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


class FragilityVisualizer:
    """Visualize fragility analysis and cliff detection results."""

    def __init__(self, width: int = 60, height: int = 15) -> None:
        self._width = width
        self._height = height

    # ------------------------------------------------------------------
    # Cliff map
    # ------------------------------------------------------------------

    def render_cliff_map(
        self,
        parameter_values: np.ndarray,
        metric_values: np.ndarray,
        cliff_locations: list[int] | None = None,
        title: str = "Fragility Cliff Map",
    ) -> str:
        """Render a 1-D cliff map showing where metric drops sharply."""
        p = np.asarray(parameter_values, dtype=float)
        m = np.asarray(metric_values, dtype=float)
        n = len(p)
        if n == 0:
            return f"{title}: (no data)"

        # Normalise to display range
        m_min, m_max = float(m.min()), float(m.max())
        m_range = m_max - m_min if m_max > m_min else 1.0

        # Create grid
        grid = [[" " for _ in range(self._width)] for _ in range(self._height)]

        # Map data to grid
        for i in range(n):
            col = int((i / max(n - 1, 1)) * (self._width - 1))
            row = self._height - 1 - int(((m[i] - m_min) / m_range) * (self._height - 1))
            row = max(0, min(self._height - 1, row))
            col = max(0, min(self._width - 1, col))
            grid[row][col] = "█"

        # Mark cliff locations
        if cliff_locations:
            for idx in cliff_locations:
                if 0 <= idx < n:
                    col = int((idx / max(n - 1, 1)) * (self._width - 1))
                    for row in range(self._height):
                        if grid[row][col] == " ":
                            grid[row][col] = "│"

        lines = [f"{title}:", ""]
        for row_idx, row in enumerate(grid):
            if row_idx == 0:
                label = f"{m_max:.2f}"
            elif row_idx == self._height - 1:
                label = f"{m_min:.2f}"
            else:
                label = "      "
            lines.append(f"  {label:>6} │{''.join(row)}│")

        lines.append(f"         └{'─' * self._width}┘")
        lines.append(f"         {p.min():.2f}{' ' * max(0, self._width - 12)}{p.max():.2f}")

        if cliff_locations:
            lines.append(f"\n  Cliffs detected at indices: {cliff_locations}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fragility summary
    # ------------------------------------------------------------------

    def render_fragility_summary(
        self,
        components: list[dict[str, Any]],
        title: str = "Fragility Summary",
    ) -> str:
        """Render a summary of component fragilities.

        Each component dict: name, fragility_score (0-1), cliff_risk,
        sensitivity, description.
        """
        if not components:
            return f"{title}: (no components)"

        lines = [f"{title}:", ""]
        max_name = max(len(c.get("name", "?")) for c in components)

        lines.append(
            f"  {'Component':<{max_name}}  {'Fragility':>10}  {'Risk':>6}  {'Sensitivity':>12}"
        )
        lines.append(
            f"  {'─' * max_name}  {'─' * 10}  {'─' * 6}  {'─' * 12}"
        )

        for c in sorted(components, key=lambda x: -x.get("fragility_score", 0)):
            name = c.get("name", "?")
            score = c.get("fragility_score", 0)
            risk = c.get("cliff_risk", "low")
            sens = c.get("sensitivity", 0)

            # Visual score bar
            bar_len = int(score * 20)
            bar = "▓" * bar_len + "░" * (20 - bar_len)

            risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🔴"}.get(risk, "⚪")

            lines.append(
                f"  {name:<{max_name}}  {bar} {score:.2f}  {risk_icon}  {sens:>11.4f}"
            )

            desc = c.get("description", "")
            if desc:
                lines.append(f"  {' ' * max_name}  └ {desc}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Sensitivity landscape (2D heatmap in text)
    # ------------------------------------------------------------------

    def render_sensitivity_landscape(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        x_label: str = "Parameter 1",
        y_label: str = "Parameter 2",
        title: str = "Sensitivity Landscape",
    ) -> str:
        """Render a 2D sensitivity landscape as a text heatmap."""
        Z = np.asarray(z_values, dtype=float)
        if Z.ndim != 2 or Z.shape[0] == 0:
            return f"{title}: (no data)"

        rows, cols = Z.shape
        # Subsample if too large
        max_rows = min(rows, self._height)
        max_cols = min(cols, self._width)
        step_r = max(1, rows // max_rows)
        step_c = max(1, cols // max_cols)
        Z_sub = Z[::step_r, ::step_c]

        z_min, z_max = float(Z_sub.min()), float(Z_sub.max())
        z_range = z_max - z_min if z_max > z_min else 1.0

        # Heatmap characters (from low to high)
        chars = " ░▒▓█"

        lines = [f"{title}:", f"  {y_label} ↕  /  {x_label} ↔", ""]

        for i in range(Z_sub.shape[0]):
            row_chars = []
            for j in range(Z_sub.shape[1]):
                level = int(((Z_sub[i, j] - z_min) / z_range) * (len(chars) - 1))
                level = max(0, min(len(chars) - 1, level))
                row_chars.append(chars[level])
            y_val = y_values[i * step_r] if i * step_r < len(y_values) else 0
            lines.append(f"  {y_val:>6.2f} │{''.join(row_chars)}│")

        lines.append(f"         └{'─' * Z_sub.shape[1]}┘")

        x_arr = np.asarray(x_values, dtype=float)
        if len(x_arr) > 0:
            lines.append(f"         {x_arr.min():.2f}{' ' * max(0, Z_sub.shape[1] - 12)}{x_arr.max():.2f}")

        lines.append(f"\n  Legend: ' '=low  ░  ▒  ▓  █=high  (range: {z_min:.3f} to {z_max:.3f})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Regression risk assessment
    # ------------------------------------------------------------------

    def render_risk_assessment(
        self,
        risks: list[dict[str, Any]],
        title: str = "Regression Risk Assessment",
    ) -> str:
        """Render a risk assessment table.

        Each risk dict: component, probability, impact, risk_score, mitigation.
        """
        if not risks:
            return f"{title}: (no risks identified)"

        lines = [f"{title}:", ""]
        max_name = max(len(r.get("component", "?")) for r in risks)

        lines.append(
            f"  {'Component':<{max_name}}  {'Prob':>6}  {'Impact':>7}  {'Risk':>6}  Mitigation"
        )
        lines.append(
            f"  {'─' * max_name}  {'─' * 6}  {'─' * 7}  {'─' * 6}  {'─' * 20}"
        )

        for r in sorted(risks, key=lambda x: -x.get("risk_score", 0)):
            name = r.get("component", "?")
            prob = r.get("probability", 0)
            impact = r.get("impact", 0)
            score = r.get("risk_score", 0)
            mitigation = r.get("mitigation", "none")

            risk_level = "🔴" if score > 0.7 else ("🟡" if score > 0.3 else "🟢")

            lines.append(
                f"  {name:<{max_name}}  {prob:>5.2f}  {impact:>6.2f}  {risk_level}{score:.2f}  {mitigation}"
            )

        return "\n".join(lines)
