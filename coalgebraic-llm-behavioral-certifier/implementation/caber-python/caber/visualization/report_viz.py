"""
CABER Report Visualization — Text-based audit report rendering for terminal output.

Part of CABER (Coalgebraic Behavioral Auditing of Foundation Models).
Renders structured audit reports as formatted text with bar charts,
comparison tables, learning curves, and summary statistics.

Uses only stdlib (dataclasses, math, datetime). Supports Unicode box-drawing
and optional ANSI color output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data models (defined locally to avoid circular imports)
# ---------------------------------------------------------------------------

@dataclass
class PropertyResult:
    """Result of checking a single behavioral property."""

    name: str
    satisfied: bool
    satisfaction_degree: float  # 0.0 – 1.0
    num_queries: int


@dataclass
class ModelResult:
    """Aggregated audit results for one model."""

    model_name: str
    properties: List[PropertyResult]
    total_queries: int
    wall_time_seconds: float
    num_states: int
    num_transitions: int


@dataclass
class RegressionItem:
    """Comparison of a single property between two audit runs."""

    property_name: str
    old_score: float
    new_score: float

    @property
    def delta(self) -> float:
        """Difference new − old."""
        return self.new_score - self.old_score

    @property
    def is_regression(self) -> bool:
        """True when the new score dropped by more than 0.05."""
        return self.new_score < self.old_score - 0.05


@dataclass
class LearningPoint:
    """A single observation along the L*-style learning curve."""

    iteration: int
    num_states: int
    fidelity: float
    queries_used: int


@dataclass
class AuditReport:
    """Top-level container for a complete CABER audit report."""

    title: str
    timestamp: str
    models: List[ModelResult]
    regressions: Optional[List[RegressionItem]] = None
    learning_curve: Optional[List[LearningPoint]] = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format *seconds* into a human-readable duration string.

    Returns ``"1h 23m 45s"`` for long durations or ``"45.2s"`` for short ones.
    """
    if seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_percentage(value: float) -> str:
    """Format a 0–1 float as a percentage string, e.g. ``'78.5%'``."""
    return f"{value * 100:.1f}%"


def report_from_dict(data: dict) -> AuditReport:
    """Parse a plain dictionary into an :class:`AuditReport`.

    Expected keys mirror the dataclass fields.  Nested lists of dicts are
    converted to the appropriate dataclass instances.
    """
    models: List[ModelResult] = []
    for m in data.get("models", []):
        props = [PropertyResult(**p) for p in m.get("properties", [])]
        models.append(ModelResult(
            model_name=m["model_name"],
            properties=props,
            total_queries=m.get("total_queries", 0),
            wall_time_seconds=m.get("wall_time_seconds", 0.0),
            num_states=m.get("num_states", 0),
            num_transitions=m.get("num_transitions", 0),
        ))

    regressions: Optional[List[RegressionItem]] = None
    if "regressions" in data and data["regressions"] is not None:
        regressions = [
            RegressionItem(
                property_name=r["property_name"],
                old_score=r["old_score"],
                new_score=r["new_score"],
            )
            for r in data["regressions"]
        ]

    learning_curve: Optional[List[LearningPoint]] = None
    if "learning_curve" in data and data["learning_curve"] is not None:
        learning_curve = [LearningPoint(**lp) for lp in data["learning_curve"]]

    return AuditReport(
        title=data.get("title", "CABER Report"),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        models=models,
        regressions=regressions,
        learning_curve=learning_curve,
        summary=data.get("summary", ""),
    )


# ---------------------------------------------------------------------------
# ReportVisualizer
# ---------------------------------------------------------------------------

class ReportVisualizer:
    """Render :class:`AuditReport` objects as formatted terminal text.

    Parameters
    ----------
    max_width : int
        Maximum line width for the output.
    use_unicode : bool
        When *True* use Unicode box-drawing characters; otherwise fall back
        to plain ASCII.
    use_color : bool
        When *True* wrap certain tokens in ANSI escape codes for coloured
        terminal output.
    """

    # ANSI colour codes
    _ANSI = {
        "green": "\033[32m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }

    def __init__(
        self,
        max_width: int = 100,
        use_unicode: bool = True,
        use_color: bool = False,
    ) -> None:
        self.max_width = max_width
        self.use_unicode = use_unicode
        self.use_color = use_color

    # -- public entry points ------------------------------------------------

    def render_full_report(self, report: AuditReport) -> str:
        """Render *report* as a complete multi-section text document.

        Sections:
        1. Header with title and timestamp.
        2. Per-model property results with bar charts.
        3. Regression comparison (when regressions are provided).
        4. Learning curve (when learning points are provided).
        5. Summary statistics for every model.
        """
        parts: List[str] = []

        # 1. Header
        parts.append(self.render_header(report.title))
        parts.append(f"  Timestamp: {report.timestamp}")
        parts.append(f"  Models evaluated: {len(report.models)}")
        parts.append("")

        # 2. Per-model results
        for model in report.models:
            parts.append(self.render_section_header(f"Model: {model.model_name}"))
            if model.properties:
                parts.append(self.render_property_bars(model.properties))
            else:
                parts.append("  (no properties evaluated)")
            parts.append("")

        # 3. Model comparison table (if >1 model)
        if len(report.models) > 1:
            parts.append(self.render_section_header("Model Comparison"))
            parts.append(self.render_model_comparison_table(report.models))
            parts.append("")

        # 4. Regression report
        if report.regressions:
            parts.append(self.render_section_header("Regression Analysis"))
            parts.append(self.render_regression_report(report.regressions))
            parts.append("")

        # 5. Learning curve
        if report.learning_curve:
            parts.append(self.render_section_header("Learning Curve"))
            parts.append(self.render_learning_curve(report.learning_curve))
            parts.append("")
            parts.append(self.render_section_header("State Growth"))
            parts.append(self.render_state_growth_curve(report.learning_curve))
            parts.append("")

        # 6. Summary stats per model
        parts.append(self.render_section_header("Summary Statistics"))
        for model in report.models:
            parts.append(self.render_summary_stats(model))
            parts.append("")

        # 7. Report summary text
        if report.summary:
            parts.append(self.render_section_header("Overall Summary"))
            parts.append(f"  {report.summary}")
            parts.append("")

        # Closing rule
        parts.append(self._hrule())
        return "\n".join(parts)

    # -- bar charts ---------------------------------------------------------

    def render_property_bars(self, results: List[PropertyResult]) -> str:
        """Render a horizontal bar chart of property satisfaction degrees.

        Properties are sorted by satisfaction degree in descending order.
        Each row looks like::

            refusal_persistence  |████████████░░░░| 0.78  ✓
        """
        if not results:
            return "  (no property results)"

        sorted_props = sorted(results, key=lambda p: p.satisfaction_degree, reverse=True)

        name_width = max(len(p.name) for p in sorted_props)
        bar_width = min(30, self.max_width - name_width - 20)
        bar_width = max(10, bar_width)

        lines: List[str] = []
        threshold = 0.5

        # Threshold indicator
        thresh_offset = int(threshold * bar_width)
        thresh_line = " " * (name_width + 3) + " " * thresh_offset + "▼ threshold" if self.use_unicode else ""
        if thresh_line:
            lines.append(thresh_line)

        for prop in sorted_props:
            label = prop.name.ljust(name_width)
            bar = self._bar(prop.satisfaction_degree, 1.0, bar_width)
            score = f"{prop.satisfaction_degree:.2f}"
            if self.use_unicode:
                status = self._color("✓", "green") if prop.satisfied else self._color("✗", "red")
            else:
                status = self._color("[PASS]", "green") if prop.satisfied else self._color("[FAIL]", "red")
            lines.append(f"  {label}  |{bar}| {score}  {status}")

        return "\n".join(lines)

    # -- comparison table ---------------------------------------------------

    def render_model_comparison_table(self, models: List[ModelResult]) -> str:
        """Render a table comparing property scores across models.

        Columns: Property | Model‑A Score | Model‑B Score | … | Delta | Status
        Includes a summary row with per-model averages.
        """
        if not models:
            return "  (no models to compare)"

        # Collect the union of property names (preserving first-seen order)
        prop_names: List[str] = []
        seen: set = set()
        for m in models:
            for p in m.properties:
                if p.name not in seen:
                    prop_names.append(p.name)
                    seen.add(p.name)

        if not prop_names:
            return "  (no properties to compare)"

        # Build score lookup  model_name -> prop_name -> score
        scores: dict[str, dict[str, float]] = {}
        for m in models:
            scores[m.model_name] = {p.name: p.satisfaction_degree for p in m.properties}

        headers = ["Property"] + [m.model_name for m in models]
        if len(models) == 2:
            headers += ["Delta", "Status"]

        rows: List[List[str]] = []
        for pname in prop_names:
            row: List[str] = [pname]
            vals: List[float] = []
            for m in models:
                v = scores[m.model_name].get(pname, float("nan"))
                vals.append(v)
                row.append(format_percentage(v) if not math.isnan(v) else "N/A")
            if len(models) == 2:
                if math.isnan(vals[0]) or math.isnan(vals[1]):
                    row.append("N/A")
                    row.append("")
                else:
                    d = vals[1] - vals[0]
                    delta_str = f"{d:+.1%}"
                    if d < -0.05:
                        delta_str = self._color(delta_str, "red")
                        status = self._color("REGRESSION", "red")
                    elif d > 0.05:
                        delta_str = self._color(delta_str, "green")
                        status = self._color("IMPROVED", "green")
                    else:
                        status = "STABLE"
                    row.append(delta_str)
                    row.append(status)
            rows.append(row)

        # Summary row
        summary_row: List[str] = [self._color("Average", "bold")]
        for m in models:
            model_scores = [
                scores[m.model_name].get(pn, float("nan"))
                for pn in prop_names
            ]
            valid = [v for v in model_scores if not math.isnan(v)]
            avg = sum(valid) / len(valid) if valid else 0.0
            summary_row.append(format_percentage(avg))
        if len(models) == 2:
            a_scores = [scores[models[0].model_name].get(pn, float("nan")) for pn in prop_names]
            b_scores = [scores[models[1].model_name].get(pn, float("nan")) for pn in prop_names]
            valid_a = [v for v in a_scores if not math.isnan(v)]
            valid_b = [v for v in b_scores if not math.isnan(v)]
            avg_a = sum(valid_a) / len(valid_a) if valid_a else 0.0
            avg_b = sum(valid_b) / len(valid_b) if valid_b else 0.0
            summary_row.append(f"{avg_b - avg_a:+.1%}")
            summary_row.append("")
        rows.append(summary_row)

        return self._format_table(headers, rows)

    # -- regression report --------------------------------------------------

    def render_regression_report(self, regressions: List[RegressionItem]) -> str:
        """Render a focused regression report.

        Each property shows *old → new* scores, delta, and a directional
        arrow (↑ improvement, ↓ regression, → stable).
        """
        if not regressions:
            return "  No regressions to report."

        lines: List[str] = []
        name_width = max(len(r.property_name) for r in regressions)

        num_regressions = 0
        num_improvements = 0

        for item in regressions:
            label = item.property_name.ljust(name_width)
            old_s = format_percentage(item.old_score)
            new_s = format_percentage(item.new_score)
            delta = item.delta
            delta_str = f"{delta:+.1%}"

            if self.use_unicode:
                if delta < -0.05:
                    arrow = self._color("↓", "red")
                    delta_str = self._color(delta_str, "red")
                    flag = self._color(" [REGRESSION]", "red")
                    num_regressions += 1
                elif delta > 0.05:
                    arrow = self._color("↑", "green")
                    delta_str = self._color(delta_str, "green")
                    flag = self._color(" [IMPROVED]", "green")
                    num_improvements += 1
                else:
                    arrow = "→"
                    flag = ""
            else:
                if delta < -0.05:
                    arrow = self._color("-", "red")
                    delta_str = self._color(delta_str, "red")
                    flag = self._color(" [REGRESSION]", "red")
                    num_regressions += 1
                elif delta > 0.05:
                    arrow = self._color("+", "green")
                    delta_str = self._color(delta_str, "green")
                    flag = self._color(" [IMPROVED]", "green")
                    num_improvements += 1
                else:
                    arrow = "="
                    flag = ""

            lines.append(
                f"  {label}  {old_s} {arrow} {new_s}  ({delta_str}){flag}"
            )

        lines.append("")
        summary = (
            f"  Summary: {num_regressions} regression(s) detected, "
            f"{num_improvements} improvement(s)"
        )
        if num_regressions > 0:
            summary = self._color(summary, "yellow")
        lines.append(summary)

        return "\n".join(lines)

    # -- learning curve plot ------------------------------------------------

    def render_learning_curve(
        self, points: List[LearningPoint], height: int = 15
    ) -> str:
        """Render a text-based line plot of fidelity over iterations.

        Y-axis spans 0.0–1.0; X-axis shows iteration numbers.  A convergence
        point is reported if the fidelity stabilises within 0.01 for ≥3
        consecutive iterations.
        """
        if not points:
            return "  (no learning data)"

        sorted_pts = sorted(points, key=lambda p: p.iteration)
        plot_char = "●" if self.use_unicode else "*"
        width = min(self.max_width - 12, len(sorted_pts))
        width = max(10, width)

        # Resample to fit width
        sampled = self._resample(sorted_pts, width)

        lines: List[str] = []
        lines.append("  Fidelity vs Iteration")
        lines.append("")

        # Build grid  (row 0 = top = fidelity 1.0)
        grid: List[List[str]] = [[" " for _ in range(width)] for _ in range(height)]

        for col, pt in enumerate(sampled):
            row = height - 1 - int(pt.fidelity * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][col] = plot_char

        # Render rows with Y-axis labels
        for r in range(height):
            fidelity_val = 1.0 - r / (height - 1)
            if r % max(1, height // 5) == 0 or r == height - 1:
                y_label = f"{fidelity_val:.1f}"
            else:
                y_label = "   "
            y_label = y_label.rjust(4)
            row_str = "".join(grid[r])

            # Grid lines at labelled rows
            if y_label.strip():
                bg_row = row_str.replace(" ", "·" if self.use_unicode else ".")
                # Keep plot chars intact
                merged = []
                for orig, bg in zip(row_str, bg_row):
                    merged.append(orig if orig != " " else bg)
                row_str = "".join(merged)

            lines.append(f"  {y_label} |{row_str}|")

        # X-axis
        x_axis = "  " + " " * 4 + " +" + "─" * width + "+" if self.use_unicode else "  " + " " * 4 + " +" + "-" * width + "+"
        lines.append(x_axis)

        # X-axis labels
        first_iter = sampled[0].iteration
        last_iter = sampled[-1].iteration
        x_labels = f"  {'':>4}  {first_iter:<{width // 2}}{last_iter:>{width - width // 2}}"
        lines.append(x_labels)
        lines.append(f"  {'':>4}  {'Iteration':^{width}}")

        # Convergence detection
        convergence_iter: Optional[int] = None
        if len(sorted_pts) >= 3:
            for i in range(len(sorted_pts) - 2):
                window = sorted_pts[i : i + 3]
                max_f = max(p.fidelity for p in window)
                min_f = min(p.fidelity for p in window)
                if max_f - min_f <= 0.01:
                    convergence_iter = window[0].iteration
                    break
        if convergence_iter is not None:
            lines.append("")
            conv_fidelity = next(
                (p.fidelity for p in sorted_pts if p.iteration == convergence_iter),
                0.0,
            )
            lines.append(
                f"  Convergence detected at iteration {convergence_iter} "
                f"(fidelity {conv_fidelity:.3f})"
            )

        return "\n".join(lines)

    # -- state growth curve -------------------------------------------------

    def render_state_growth_curve(
        self, points: List[LearningPoint], height: int = 12
    ) -> str:
        """Render a text-based staircase plot of states vs queries used.

        Shows how the hypothesis automaton grows during learning.
        """
        if not points:
            return "  (no learning data)"

        sorted_pts = sorted(points, key=lambda p: p.queries_used)
        plot_char = "■" if self.use_unicode else "#"
        width = min(self.max_width - 12, len(sorted_pts))
        width = max(10, width)

        sampled = self._resample(sorted_pts, width)

        max_states = max(p.num_states for p in sampled)
        if max_states == 0:
            max_states = 1

        lines: List[str] = []
        lines.append("  States vs Queries")
        lines.append("")

        grid: List[List[str]] = [[" " for _ in range(width)] for _ in range(height)]

        for col, pt in enumerate(sampled):
            row = height - 1 - int((pt.num_states / max_states) * (height - 1))
            row = max(0, min(height - 1, row))
            # Staircase: fill from row down to bottom
            for r in range(row, height):
                grid[r][col] = plot_char

        for r in range(height):
            state_val = max_states * (1.0 - r / (height - 1))
            if r % max(1, height // 4) == 0 or r == height - 1:
                y_label = f"{state_val:>4.0f}"
            else:
                y_label = "    "
            row_str = "".join(grid[r])
            lines.append(f"  {y_label} |{row_str}|")

        hrchar = "─" if self.use_unicode else "-"
        x_axis = "  " + " " * 4 + " +" + hrchar * width + "+"
        lines.append(x_axis)

        first_q = sampled[0].queries_used
        last_q = sampled[-1].queries_used
        x_labels = f"  {'':>4}  {first_q:<{width // 2}}{last_q:>{width - width // 2}}"
        lines.append(x_labels)
        lines.append(f"  {'':>4}  {'Queries':^{width}}")

        return "\n".join(lines)

    # -- summary stats ------------------------------------------------------

    def render_summary_stats(self, model: ModelResult) -> str:
        """Render formatted summary statistics for a single model."""
        props = model.properties
        total = len(props)
        passed = sum(1 for p in props if p.satisfied)
        failed = total - passed
        pass_rate = passed / total if total > 0 else 0.0
        avg_satisfaction = (
            sum(p.satisfaction_degree for p in props) / total if total > 0 else 0.0
        )

        lines: List[str] = []
        lines.append(f"  {self._color(model.model_name, 'bold')}")
        hrchar = "─" if self.use_unicode else "-"
        lines.append(f"  {hrchar * (len(model.model_name) + 4)}")
        lines.append(f"  Properties evaluated : {total}")

        pass_str = self._color(str(passed), "green")
        fail_str = self._color(str(failed), "red") if failed else str(failed)
        lines.append(f"  Passed               : {pass_str}")
        lines.append(f"  Failed               : {fail_str}")

        rate_color = "green" if pass_rate >= 0.8 else ("yellow" if pass_rate >= 0.5 else "red")
        lines.append(f"  Pass rate            : {self._color(format_percentage(pass_rate), rate_color)}")
        lines.append(f"  Avg satisfaction     : {format_percentage(avg_satisfaction)}")
        lines.append(f"  Total queries        : {model.total_queries:,}")
        lines.append(f"  States / transitions : {model.num_states} / {model.num_transitions}")
        lines.append(f"  Wall time            : {format_duration(model.wall_time_seconds)}")

        return "\n".join(lines)

    # -- headers & decorators -----------------------------------------------

    def render_header(self, title: str, width: int | None = None) -> str:
        """Render a prominent header with box-drawing decoration.

        Unicode mode::

            ══════════════════════
            ║  CABER REPORT      ║
            ══════════════════════

        ASCII mode::

            ======================
            |  CABER REPORT      |
            ======================
        """
        w = width or self.max_width
        inner = w - 4  # space inside border characters

        if self.use_unicode:
            top = "═" * w
            side = "║"
        else:
            top = "=" * w
            side = "|"

        padded_title = f"  {title}"
        if len(padded_title) > inner:
            padded_title = padded_title[:inner]
        padded_title = padded_title.ljust(inner)

        lines = [
            top,
            f"{side}{padded_title}{side}",
            top,
        ]
        return "\n".join(lines)

    def render_section_header(self, title: str) -> str:
        """Render a smaller section header with a horizontal rule."""
        hrchar = "─" if self.use_unicode else "-"
        rule = hrchar * self.max_width
        return f"{rule}\n  {self._color(title, 'bold')}\n{rule}"

    def render_mini_bar(self, value: float, width: int = 20) -> str:
        """Return a compact inline bar suitable for embedding in tables."""
        return self._bar(value, 1.0, width)

    # -- private helpers ----------------------------------------------------

    def _format_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: List[int] | None = None,
    ) -> str:
        """Render a generic bordered table.

        If *col_widths* is not given, columns are auto-sized to fit the
        widest cell (including headers).
        """
        num_cols = len(headers)
        # Normalise row lengths
        norm_rows = [r + [""] * (num_cols - len(r)) for r in rows]

        if col_widths is None:
            col_widths = [len(h) for h in headers]
            for row in norm_rows:
                for i, cell in enumerate(row):
                    visible_len = self._visible_len(cell)
                    if visible_len > col_widths[i]:
                        col_widths[i] = visible_len

        # Characters
        if self.use_unicode:
            h, v, tl, tr, bl, br = "─", "│", "┌", "┐", "└", "┘", 
            ml, mr, cross, tj, bj = "├", "┤", "┼", "┬", "┴"
        else:
            h, v, tl, tr, bl, br = "-", "|", "+", "+", "+", "+"
            ml, mr, cross, tj, bj = "+", "+", "+", "+", "+"

        def separator(left: str, mid: str, right: str) -> str:
            parts = [h * (w + 2) for w in col_widths]
            return left + mid.join(parts) + right

        def format_row(cells: List[str]) -> str:
            parts = []
            for i, cell in enumerate(cells):
                pad = col_widths[i] - self._visible_len(cell)
                parts.append(f" {cell}{' ' * pad} ")
            return v + v.join(parts) + v

        lines: List[str] = []
        lines.append("  " + separator(tl, tj, tr))
        lines.append("  " + format_row(headers))
        lines.append("  " + separator(ml, cross, mr))
        for row in norm_rows:
            lines.append("  " + format_row(row))
        lines.append("  " + separator(bl, bj, br))

        return "\n".join(lines)

    def _color(self, text: str, color: str) -> str:
        """Wrap *text* in ANSI escape codes if colour output is enabled."""
        if not self.use_color or color not in self._ANSI:
            return text
        return f"{self._ANSI[color]}{text}{self._ANSI['reset']}"

    def _bar(
        self,
        value: float,
        max_val: float,
        width: int,
        filled: str = "█",
        empty: str = "░",
    ) -> str:
        """Generate a horizontal bar string of *width* characters."""
        if not self.use_unicode:
            filled = "#"
            empty = "."
        ratio = max(0.0, min(1.0, value / max_val if max_val else 0.0))
        filled_count = int(ratio * width)
        return filled * filled_count + empty * (width - filled_count)

    def _hrule(self) -> str:
        """Return a full-width horizontal rule."""
        char = "═" if self.use_unicode else "="
        return char * self.max_width

    @staticmethod
    def _visible_len(text: str) -> int:
        """Return the visible length of *text*, ignoring ANSI escapes."""
        import re
        return len(re.sub(r"\033\[[0-9;]*m", "", text))

    @staticmethod
    def _resample(
        points: List[LearningPoint], target_width: int
    ) -> List[LearningPoint]:
        """Down-sample or up-sample *points* to *target_width* entries."""
        if len(points) <= target_width:
            return list(points)
        step = len(points) / target_width
        sampled: List[LearningPoint] = []
        for i in range(target_width):
            idx = int(i * step)
            idx = min(idx, len(points) - 1)
            sampled.append(points[idx])
        return sampled


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _pass_count = 0
    _fail_count = 0

    def _check(condition: bool, label: str) -> None:
        global _pass_count, _fail_count
        if condition:
            _pass_count += 1
            print(f"  PASS  {label}")
        else:
            _fail_count += 1
            print(f"  FAIL  {label}")

    # -- helpers for sample data -------------------------------------------

    def _sample_properties() -> List[PropertyResult]:
        return [
            PropertyResult("refusal_persistence", True, 0.92, 120),
            PropertyResult("safety_alignment", True, 0.85, 200),
            PropertyResult("instruction_following", False, 0.45, 150),
            PropertyResult("output_consistency", True, 0.78, 180),
            PropertyResult("toxicity_avoidance", True, 0.99, 100),
        ]

    def _sample_model(name: str = "gpt-4-test") -> ModelResult:
        return ModelResult(
            model_name=name,
            properties=_sample_properties(),
            total_queries=750,
            wall_time_seconds=3845.6,
            num_states=24,
            num_transitions=67,
        )

    def _sample_model_b() -> ModelResult:
        return ModelResult(
            model_name="claude-3-test",
            properties=[
                PropertyResult("refusal_persistence", True, 0.88, 130),
                PropertyResult("safety_alignment", True, 0.91, 210),
                PropertyResult("instruction_following", True, 0.72, 160),
                PropertyResult("output_consistency", False, 0.40, 190),
                PropertyResult("toxicity_avoidance", True, 0.97, 110),
            ],
            total_queries=800,
            wall_time_seconds=4200.0,
            num_states=30,
            num_transitions=82,
        )

    def _sample_regressions() -> List[RegressionItem]:
        return [
            RegressionItem("refusal_persistence", 0.92, 0.88),
            RegressionItem("safety_alignment", 0.85, 0.91),
            RegressionItem("instruction_following", 0.45, 0.72),
            RegressionItem("output_consistency", 0.78, 0.40),
            RegressionItem("toxicity_avoidance", 0.99, 0.97),
        ]

    def _sample_learning_curve() -> List[LearningPoint]:
        pts = []
        for i in range(20):
            fidelity = min(1.0, 0.3 + 0.04 * i - 0.002 * max(0, i - 12))
            states = min(24, 2 + i * 2 if i < 12 else 24)
            pts.append(LearningPoint(
                iteration=i + 1,
                num_states=states,
                fidelity=round(fidelity, 3),
                queries_used=50 * (i + 1),
            ))
        return pts

    # -- Test 1: Property bar chart ----------------------------------------

    print("\n=== Test 1: Property bar chart ===")
    viz = ReportVisualizer(max_width=80)
    bars = viz.render_property_bars(_sample_properties())
    _check("refusal_persistence" in bars, "contains property name")
    _check("0.92" in bars, "contains score")
    _check("✓" in bars, "contains pass symbol")
    _check("✗" in bars, "contains fail symbol")
    _check(bars.count("\n") >= 4, "has multiple lines")
    print(bars)

    # -- Test 2: Model comparison table ------------------------------------

    print("\n=== Test 2: Model comparison table ===")
    table = viz.render_model_comparison_table([_sample_model(), _sample_model_b()])
    _check("gpt-4-test" in table, "contains model A name")
    _check("claude-3-test" in table, "contains model B name")
    _check("refusal_persistence" in table, "contains property name")
    _check("Average" in table, "contains summary row")
    print(table)

    # -- Test 3: Regression report -----------------------------------------

    print("\n=== Test 3: Regression report ===")
    reg_report = viz.render_regression_report(_sample_regressions())
    _check("output_consistency" in reg_report, "contains regressed property")
    _check("REGRESSION" in reg_report, "flags regression")
    _check("IMPROVED" in reg_report, "flags improvement")
    _check("Summary" in reg_report, "contains summary")
    print(reg_report)

    # -- Test 4: Learning curve plot ---------------------------------------

    print("\n=== Test 4: Learning curve plot ===")
    lc = viz.render_learning_curve(_sample_learning_curve())
    _check("Fidelity" in lc, "has Y-axis title")
    _check("Iteration" in lc, "has X-axis title")
    _check("●" in lc, "has plot character")
    _check(lc.count("\n") >= 15, "has sufficient height")
    print(lc)

    # -- Test 5: Summary stats ---------------------------------------------

    print("\n=== Test 5: Summary stats ===")
    stats = viz.render_summary_stats(_sample_model())
    _check("gpt-4-test" in stats, "contains model name")
    _check("750" in stats, "contains query count")
    _check("80.0%" in stats, "contains pass rate")
    _check("1h" in stats, "contains wall time")
    print(stats)

    # -- Test 6: Full report -----------------------------------------------

    print("\n=== Test 6: Full report ===")
    report = AuditReport(
        title="CABER Behavioral Audit v2.1",
        timestamp="2025-01-15T14:30:00Z",
        models=[_sample_model(), _sample_model_b()],
        regressions=_sample_regressions(),
        learning_curve=_sample_learning_curve(),
        summary="Two models evaluated. One regression detected in output_consistency.",
    )
    full = viz.render_full_report(report)
    _check("CABER Behavioral Audit" in full, "contains report title")
    _check("Model Comparison" in full, "contains comparison section")
    _check("Regression Analysis" in full, "contains regression section")
    _check("Learning Curve" in full, "contains learning curve section")
    _check("Summary Statistics" in full, "contains summary section")
    _check(len(full) > 500, "report is substantial")
    print(full)

    # -- Test 7: ASCII mode ------------------------------------------------

    print("\n=== Test 7: ASCII mode ===")
    ascii_viz = ReportVisualizer(max_width=80, use_unicode=False)
    ascii_bars = ascii_viz.render_property_bars(_sample_properties())
    _check("[PASS]" in ascii_bars, "ASCII pass marker")
    _check("[FAIL]" in ascii_bars, "ASCII fail marker")
    _check("#" in ascii_bars, "ASCII bar fill character")
    _check("█" not in ascii_bars, "no unicode bar chars")

    ascii_header = ascii_viz.render_header("Test Header")
    _check("=" in ascii_header, "ASCII header uses =")
    _check("|" in ascii_header, "ASCII header uses |")
    _check("═" not in ascii_header, "no unicode box chars in header")
    print(ascii_bars)
    print(ascii_header)

    ascii_lc = ascii_viz.render_learning_curve(_sample_learning_curve())
    _check("*" in ascii_lc, "ASCII plot character")
    _check("●" not in ascii_lc, "no unicode plot char")
    print(ascii_lc)

    # -- Test 8: Color mode ------------------------------------------------

    print("\n=== Test 8: Color mode ===")
    color_viz = ReportVisualizer(max_width=80, use_color=True)
    color_bars = color_viz.render_property_bars(_sample_properties())
    _check("\033[" in color_bars, "contains ANSI escape code")
    _check("\033[32m" in color_bars, "contains green code")

    no_color_viz = ReportVisualizer(max_width=80, use_color=False)
    plain_bars = no_color_viz.render_property_bars(_sample_properties())
    _check("\033[" not in plain_bars, "no ANSI codes when color disabled")
    print(color_bars)

    # -- Test 9: Edge cases ------------------------------------------------

    print("\n=== Test 9: Edge cases ===")

    # Empty properties
    empty_bars = viz.render_property_bars([])
    _check("no property" in empty_bars.lower(), "empty properties handled")

    # Empty models
    empty_table = viz.render_model_comparison_table([])
    _check("no models" in empty_table.lower(), "empty models handled")

    # Empty regressions
    empty_reg = viz.render_regression_report([])
    _check("no regressions" in empty_reg.lower(), "empty regressions handled")

    # Empty learning curve
    empty_lc = viz.render_learning_curve([])
    _check("no learning" in empty_lc.lower(), "empty learning curve handled")

    # Single model comparison
    single_table = viz.render_model_comparison_table([_sample_model()])
    _check("gpt-4-test" in single_table, "single model table works")

    # Model with no properties
    empty_model = ModelResult("empty-model", [], 0, 0.0, 0, 0)
    empty_stats = viz.render_summary_stats(empty_model)
    _check("empty-model" in empty_stats, "empty model stats works")
    _check("0.0%" in empty_stats, "zero pass rate")

    # report_from_dict
    data = {
        "title": "Dict Report",
        "timestamp": "2025-01-01T00:00:00Z",
        "models": [
            {
                "model_name": "test-model",
                "properties": [
                    {"name": "prop1", "satisfied": True,
                     "satisfaction_degree": 0.9, "num_queries": 50}
                ],
                "total_queries": 50,
                "wall_time_seconds": 10.0,
                "num_states": 5,
                "num_transitions": 8,
            }
        ],
        "regressions": [
            {"property_name": "prop1", "old_score": 0.8, "new_score": 0.9}
        ],
        "learning_curve": [
            {"iteration": 1, "num_states": 2, "fidelity": 0.5, "queries_used": 10}
        ],
        "summary": "Test summary",
    }
    parsed = report_from_dict(data)
    _check(parsed.title == "Dict Report", "report_from_dict title")
    _check(len(parsed.models) == 1, "report_from_dict models")
    _check(parsed.models[0].model_name == "test-model", "report_from_dict model name")
    _check(len(parsed.models[0].properties) == 1, "report_from_dict properties")
    _check(parsed.regressions is not None and len(parsed.regressions) == 1,
           "report_from_dict regressions")
    _check(parsed.learning_curve is not None and len(parsed.learning_curve) == 1,
           "report_from_dict learning_curve")

    # format_duration edge cases
    _check(format_duration(0.5) == "0.5s", "format_duration sub-second")
    _check(format_duration(65) == "1m 05s", "format_duration minutes")
    _check(format_duration(3661) == "1h 01m 01s", "format_duration hours")
    _check(format_duration(-5) == "0.0s", "format_duration negative")

    # format_percentage
    _check(format_percentage(0.785) == "78.5%", "format_percentage")
    _check(format_percentage(1.0) == "100.0%", "format_percentage 100")
    _check(format_percentage(0.0) == "0.0%", "format_percentage 0")

    # RegressionItem properties
    ri = RegressionItem("test", 0.9, 0.8)
    _check(abs(ri.delta - (-0.1)) < 1e-9, "RegressionItem delta")
    _check(ri.is_regression is True, "RegressionItem is_regression True")

    ri_ok = RegressionItem("test", 0.9, 0.87)
    _check(ri_ok.is_regression is False, "RegressionItem is_regression False (within tolerance)")

    # State growth curve
    sg = viz.render_state_growth_curve(_sample_learning_curve())
    _check("States" in sg, "state growth has title")
    _check("Queries" in sg, "state growth has x-axis label")
    _check("■" in sg, "state growth has plot char")

    # Mini bar
    mb = viz.render_mini_bar(0.75, width=20)
    _check(len(mb) == 20, "mini bar correct width")
    _check("█" in mb, "mini bar filled chars")
    _check("░" in mb, "mini bar empty chars")

    # Header widths
    h40 = viz.render_header("Short", width=40)
    _check(len(h40.split("\n")[0]) == 40, "header respects custom width")

    # -- Final summary -----------------------------------------------------

    print(f"\n{'=' * 60}")
    print(f"  Results: {_pass_count} passed, {_fail_count} failed")
    print(f"{'=' * 60}")
    if _fail_count > 0:
        sys.exit(1)
    else:
        print("  All tests passed.")
