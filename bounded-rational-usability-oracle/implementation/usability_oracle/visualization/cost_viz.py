"""
usability_oracle.visualization.cost_viz — Cost landscape visualization.

Renders cost distributions, cognitive load breakdowns, Fitts' law
analysis displays, and cost-comparison charts in text format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class CostVizConfig:
    """Configuration for cost visualization."""
    bar_width: int = 40
    precision: int = 3
    show_percentages: bool = True
    histogram_bins: int = 20
    histogram_height: int = 10


class CostVisualizer:
    """Visualize cost distributions and breakdowns."""

    def __init__(self, config: CostVizConfig | None = None) -> None:
        self._config = config or CostVizConfig()

    # ------------------------------------------------------------------
    # Cost breakdown chart
    # ------------------------------------------------------------------

    def render_breakdown(
        self,
        costs: dict[str, float],
        title: str = "Cost Breakdown",
    ) -> str:
        """Render a horizontal bar chart of cost components."""
        if not costs:
            return f"{title}: (no data)"

        total = sum(abs(v) for v in costs.values())
        max_val = max(abs(v) for v in costs.values()) if costs else 1
        max_name = max(len(k) for k in costs)

        lines = [f"{title} (total: {total:.{self._config.precision}f}):", ""]
        for name, value in sorted(costs.items(), key=lambda x: -abs(x[1])):
            bar_len = int(abs(value) / max_val * self._config.bar_width) if max_val > 0 else 0
            bar = "█" * bar_len
            pct = f" ({abs(value) / total * 100:.1f}%)" if self._config.show_percentages and total > 0 else ""
            lines.append(f"  {name:>{max_name}} │{bar:<{self._config.bar_width}}│ {value:.{self._config.precision}f}{pct}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cost histogram
    # ------------------------------------------------------------------

    def render_histogram(
        self,
        values: Sequence[float],
        title: str = "Cost Distribution",
        n_bins: int | None = None,
    ) -> str:
        """Render a text-based histogram."""
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return f"{title}: (no data)"

        bins = n_bins or self._config.histogram_bins
        counts, edges = np.histogram(arr, bins=bins)
        max_count = int(counts.max())
        height = self._config.histogram_height

        lines = [f"{title} (n={len(arr)}, μ={arr.mean():.3f}, σ={arr.std():.3f}):", ""]

        # Vertical histogram
        for row in range(height, 0, -1):
            threshold = max_count * row / height
            bar_chars = []
            for c in counts:
                bar_chars.append("█" if c >= threshold else " ")
            count_label = f"{int(threshold):>4}" if row == height or row == 1 else "    "
            lines.append(f"  {count_label} │{''.join(bar_chars)}│")

        # X-axis
        lines.append(f"       └{'─' * bins}┘")
        lines.append(f"       {edges[0]:.2f}{' ' * max(0, bins - 12)}{edges[-1]:.2f}")

        # Summary stats
        lines.append(f"\n  Min: {arr.min():.{self._config.precision}f}  "
                      f"Median: {np.median(arr):.{self._config.precision}f}  "
                      f"Max: {arr.max():.{self._config.precision}f}")
        lines.append(f"  P25: {np.percentile(arr, 25):.{self._config.precision}f}  "
                      f"P75: {np.percentile(arr, 75):.{self._config.precision}f}  "
                      f"P95: {np.percentile(arr, 95):.{self._config.precision}f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cost comparison (before/after)
    # ------------------------------------------------------------------

    def render_comparison(
        self,
        costs_a: dict[str, float],
        costs_b: dict[str, float],
        label_a: str = "Before",
        label_b: str = "After",
    ) -> str:
        """Render side-by-side cost comparison."""
        all_keys = sorted(set(costs_a.keys()) | set(costs_b.keys()))
        max_name = max(len(k) for k in all_keys) if all_keys else 0

        lines = [
            f"Cost Comparison: {label_a} vs {label_b}",
            "",
            f"  {'Component':<{max_name}}  {label_a:>10}  {label_b:>10}  {'Change':>10}  {'%':>8}",
            f"  {'─' * max_name}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 8}",
        ]

        total_a, total_b = 0.0, 0.0
        for key in all_keys:
            va = costs_a.get(key, 0.0)
            vb = costs_b.get(key, 0.0)
            total_a += va
            total_b += vb
            diff = vb - va
            pct = (diff / va * 100) if abs(va) > 1e-10 else 0.0
            sign = "+" if diff > 0 else ""
            marker = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "=")
            lines.append(
                f"  {key:<{max_name}}  {va:>10.3f}  {vb:>10.3f}  {sign}{diff:>9.3f}  {pct:>7.1f}% {marker}"
            )

        diff_total = total_b - total_a
        pct_total = (diff_total / total_a * 100) if abs(total_a) > 1e-10 else 0.0
        lines.append(f"  {'─' * max_name}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 8}")
        sign = "+" if diff_total > 0 else ""
        lines.append(
            f"  {'TOTAL':<{max_name}}  {total_a:>10.3f}  {total_b:>10.3f}  {sign}{diff_total:>9.3f}  {pct_total:>7.1f}%"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fitts' law analysis
    # ------------------------------------------------------------------

    def render_fitts_analysis(
        self,
        targets: list[dict[str, Any]],
        title: str = "Fitts' Law Analysis",
    ) -> str:
        """Render Fitts' law analysis for UI targets.

        Each target dict should have keys: name, distance, width, id_value, time.
        """
        if not targets:
            return f"{title}: (no targets)"

        lines = [f"{title}:", ""]
        max_name = max(len(t.get("name", "?")) for t in targets)
        lines.append(
            f"  {'Target':<{max_name}}  {'Distance':>10}  {'Width':>8}  {'ID':>6}  {'Time':>8}"
        )
        lines.append(
            f"  {'─' * max_name}  {'─' * 10}  {'─' * 8}  {'─' * 6}  {'─' * 8}"
        )

        for t in sorted(targets, key=lambda x: -x.get("id_value", 0)):
            name = t.get("name", "?")
            dist = t.get("distance", 0)
            width = t.get("width", 0)
            id_val = t.get("id_value", 0)
            time_val = t.get("time", 0)
            difficulty = "⚠" if id_val > 4.0 else ("◑" if id_val > 3.0 else "○")
            lines.append(
                f"  {name:<{max_name}}  {dist:>10.1f}  {width:>8.1f}  {id_val:>5.2f}  {time_val:>7.3f}s {difficulty}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Hick-Hyman choice analysis
    # ------------------------------------------------------------------

    def render_hick_analysis(
        self,
        choices: list[dict[str, Any]],
        title: str = "Hick-Hyman Analysis",
    ) -> str:
        """Render Hick-Hyman law analysis for choice points.

        Each choice dict: name, n_options, probabilities, decision_time.
        """
        if not choices:
            return f"{title}: (no choices)"

        lines = [f"{title}:", ""]
        max_name = max(len(c.get("name", "?")) for c in choices)
        lines.append(
            f"  {'Choice Point':<{max_name}}  {'Options':>8}  {'Entropy':>8}  {'Time':>8}"
        )
        lines.append(
            f"  {'─' * max_name}  {'─' * 8}  {'─' * 8}  {'─' * 8}"
        )

        for c in sorted(choices, key=lambda x: -x.get("decision_time", 0)):
            name = c.get("name", "?")
            n_opts = c.get("n_options", 0)
            probs = c.get("probabilities", [])
            entropy = 0.0
            if probs:
                for p in probs:
                    if p > 0:
                        entropy -= p * math.log2(p)
            else:
                entropy = math.log2(max(n_opts, 1))
            dt = c.get("decision_time", 0)
            complexity = "⚠" if n_opts > 7 else ("◑" if n_opts > 4 else "○")
            lines.append(
                f"  {name:<{max_name}}  {n_opts:>8}  {entropy:>8.3f}  {dt:>7.3f}s {complexity}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Working memory load
    # ------------------------------------------------------------------

    def render_memory_load(
        self,
        chunks: list[dict[str, Any]],
        capacity: int = 4,
    ) -> str:
        """Render working memory load analysis.

        Each chunk dict: name, load (number of chunks), required.
        """
        lines = ["Working Memory Load:", f"  Capacity: {capacity} chunks", ""]

        total_load = sum(c.get("load", 1) for c in chunks)
        overloaded = total_load > capacity

        for c in chunks:
            name = c.get("name", "?")
            load = c.get("load", 1)
            bar = "■" * load + "□" * max(0, capacity - load)
            status = "⚠" if load > capacity else ""
            lines.append(f"  {name:<20} {bar} ({load}/{capacity}) {status}")

        lines.append(f"\n  Total load: {total_load}/{capacity} " +
                      ("⚠ OVERLOADED" if overloaded else "✓ Within capacity"))

        return "\n".join(lines)
