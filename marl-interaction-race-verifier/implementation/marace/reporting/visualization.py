"""Visualization utilities for the MARACE verification pipeline.

Provides both matplotlib-based graphical plots and ASCII-based terminal
renderings for state spaces, schedule coverage, race regions, interaction
groups, convergence diagnostics, and metric dashboards.

matplotlib is an optional dependency; all classes degrade gracefully to
ASCII-only output when it is unavailable.
"""

from __future__ import annotations

import math
import io
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure as _mpl_figure  # noqa: F401

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# ASCII utility helpers
# ---------------------------------------------------------------------------

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


class ASCIIVisualizer:
    """Static helper methods for common ASCII visualization primitives."""

    @staticmethod
    def horizontal_bar(value: float, max_val: float, width: int = 40) -> str:
        """Render a horizontal bar ``[████░░░░░░]`` style.

        Args:
            value: Current value.
            max_val: Maximum value (determines full-bar width).
            width: Character width of the bar body.

        Returns:
            A single-line string representing the bar.
        """
        if max_val <= 0:
            filled = 0
        else:
            filled = int(round(min(value / max_val, 1.0) * width))
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    @staticmethod
    def sparkline(values: list[float]) -> str:
        """Render a Unicode sparkline for a sequence of numeric values.

        Args:
            values: Numeric data points.

        Returns:
            A compact single-line sparkline string.
        """
        if not values:
            return ""
        lo = min(values)
        hi = max(values)
        span = hi - lo if hi != lo else 1.0
        return "".join(
            _SPARK_CHARS[min(int((v - lo) / span * (len(_SPARK_CHARS) - 1)), len(_SPARK_CHARS) - 1)]
            for v in values
        )

    @staticmethod
    def table(
        headers: list[str],
        rows: list[list[Any]],
        alignment: list[str] | None = None,
    ) -> str:
        """Render a bordered ASCII table.

        Args:
            headers: Column header labels.
            rows: List of row data (each row a list matching *headers* length).
            alignment: Per-column alignment chars (``'l'``, ``'r'``, ``'c'``).
                       Defaults to left-align for all columns.

        Returns:
            Multi-line table string.
        """
        if alignment is None:
            alignment = ["l"] * len(headers)
        str_rows = [[str(c) for c in row] for row in rows]
        col_widths = [
            max(len(h), *(len(r[i]) for r in str_rows) if str_rows else 0)
            for i, h in enumerate(headers)
        ]
        # Ensure minimum width equals header length when no rows
        col_widths = [max(w, len(h)) for w, h in zip(col_widths, headers)]

        def _fmt(cells: list[str], aligns: list[str]) -> str:
            parts: list[str] = []
            for cell, w, a in zip(cells, col_widths, aligns):
                if a == "r":
                    parts.append(cell.rjust(w))
                elif a == "c":
                    parts.append(cell.center(w))
                else:
                    parts.append(cell.ljust(w))
            return "│ " + " │ ".join(parts) + " │"

        sep = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
        top = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
        bot = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

        lines = [top, _fmt(headers, alignment), sep]
        for row in str_rows:
            lines.append(_fmt(row, alignment))
        lines.append(bot)
        return "\n".join(lines)

    @staticmethod
    def box(title: str, lines: list[str], width: int = 60) -> str:
        """Render a titled box around *lines*.

        Args:
            title: Box title (centered in the top border).
            lines: Content lines.
            width: Outer width in characters.

        Returns:
            Multi-line boxed string.
        """
        inner = width - 4  # "│ " + content + " │"
        title_seg = f" {title} "
        pad = width - 2 - len(title_seg)
        top = "┌" + "─" * (pad // 2) + title_seg + "─" * (pad - pad // 2) + "┐"
        bot = "└" + "─" * (width - 2) + "┘"
        body = []
        for line in lines:
            truncated = line[:inner]
            body.append("│ " + truncated.ljust(inner) + " │")
        return "\n".join([top, *body, bot])


# ---------------------------------------------------------------------------
# StateSpaceVisualizer
# ---------------------------------------------------------------------------


class StateSpaceVisualizer:
    """2-D / 3-D scatter-style visualization of abstract states.

    Safe states are rendered as ``'.'`` and race-inducing states as ``'X'``
    in ASCII mode.
    """

    def __init__(self, width: int = 80, height: int = 24) -> None:
        self.width = width
        self.height = height

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _project(
        states: list[dict[str, Any]],
        dims: tuple[str, str],
    ) -> list[tuple[float, float]]:
        """Project each state dict onto two named dimensions.

        Args:
            states: List of state dictionaries.
            dims: Two keys to extract as (x, y).

        Returns:
            List of (x, y) pairs.
        """
        dx, dy = dims
        return [(float(s.get(dx, 0.0)), float(s.get(dy, 0.0))) for s in states]

    # -- matplotlib ---------------------------------------------------------

    def plot_2d(
        self,
        states: list[dict[str, Any]],
        dimensions: tuple[str, str],
        races: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Produce a matplotlib scatter figure of states.

        Args:
            states: Abstract state dictionaries.
            dimensions: Pair of dimension keys to project onto axes.
            races: Optional list of race descriptors.  Each must contain at
                least a ``"state"`` key whose value is a state dict.

        Returns:
            A :class:`matplotlib.figure.Figure` if matplotlib is available,
            otherwise ``None``.
        """
        if not _HAS_MPL:
            return None

        pts = self._project(states, dimensions)
        race_set: set[int] = set()
        if races:
            race_states = [r["state"] for r in races if "state" in r]
            race_pts = self._project(race_states, dimensions)
            race_set = {hash(p) for p in race_pts}

        fig, ax = plt.subplots(figsize=(8, 6))
        safe_x, safe_y, race_x, race_y = [], [], [], []
        for x, y in pts:
            if hash((x, y)) in race_set:
                race_x.append(x)
                race_y.append(y)
            else:
                safe_x.append(x)
                safe_y.append(y)

        ax.scatter(safe_x, safe_y, c="steelblue", marker="o", label="safe", alpha=0.6)
        ax.scatter(race_x, race_y, c="crimson", marker="x", s=80, label="race", alpha=0.9)
        ax.set_xlabel(dimensions[0])
        ax.set_ylabel(dimensions[1])
        ax.set_title("State-Space Projection")
        ax.legend()
        return fig

    # -- ASCII --------------------------------------------------------------

    def to_ascii(
        self,
        states: list[dict[str, Any]],
        dimensions: tuple[str, str],
        races: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render an ASCII scatter plot of the state space.

        Args:
            states: State dictionaries.
            dimensions: Pair of dimension keys.
            races: Optional race descriptors (each with a ``"state"`` key).

        Returns:
            Multi-line ASCII scatter plot.
        """
        pts = self._project(states, dimensions)
        race_coords: set[tuple[float, float]] = set()
        if races:
            race_coords = set(
                self._project(
                    [r["state"] for r in races if "state" in r], dimensions
                )
            )

        if not pts:
            return "(no states)"

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = x_max - x_min if x_max != x_min else 1.0
        y_span = y_max - y_min if y_max != y_min else 1.0

        plot_w = self.width - 6  # leave room for y-axis labels
        plot_h = self.height - 3  # leave room for x-axis + title

        grid: list[list[str]] = [[" "] * plot_w for _ in range(plot_h)]

        for x, y in pts:
            col = int((x - x_min) / x_span * (plot_w - 1))
            row = plot_h - 1 - int((y - y_min) / y_span * (plot_h - 1))
            marker = "X" if (x, y) in race_coords else "."
            grid[row][col] = marker

        buf = io.StringIO()
        buf.write(f"  {dimensions[0]} vs {dimensions[1]}\n")
        for r, row in enumerate(grid):
            if r == 0:
                label = f"{y_max:5.1f}"
            elif r == plot_h - 1:
                label = f"{y_min:5.1f}"
            else:
                label = "     "
            buf.write(f"{label}|{''.join(row)}\n")
        buf.write("     +" + "─" * plot_w + "\n")
        buf.write(f"      {x_min:<.1f}{' ' * (plot_w - 10)}{x_max:>.1f}\n")
        return buf.getvalue().rstrip("\n")


# ---------------------------------------------------------------------------
# ScheduleSpaceVisualizer
# ---------------------------------------------------------------------------


class ScheduleSpaceVisualizer:
    """Visualize schedule exploration coverage."""

    def plot(self, explored_schedules: list[Any], total_bound: int) -> Any:
        """Render a matplotlib bar/pie of schedule coverage.

        Args:
            explored_schedules: Schedules that have been explored.
            total_bound: Upper bound on total number of schedules.

        Returns:
            A matplotlib figure or ``None``.
        """
        if not _HAS_MPL:
            return None

        explored = len(explored_schedules)
        remaining = max(total_bound - explored, 0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Explored", "Remaining"], [explored, remaining],
               color=["steelblue", "lightgray"])
        ax.set_ylabel("Schedules")
        ax.set_title("Schedule Space Coverage")
        return fig

    def to_ascii(self, explored: list[Any], total: int) -> str:
        """Render a progress bar with coverage statistics.

        Args:
            explored: List of explored schedules.
            total: Total schedule bound.

        Returns:
            Multi-line ASCII progress summary.
        """
        n = len(explored)
        pct = (n / total * 100) if total > 0 else 0.0
        bar = ASCIIVisualizer.horizontal_bar(n, total, width=40)
        lines = [
            "Schedule Coverage",
            f"  {bar} {pct:5.1f}%",
            f"  explored: {n}  /  bound: {total}",
        ]
        return "\n".join(lines)

    def coverage_heatmap(
        self, schedules: list[Any], agent_ids: list[str]
    ) -> str:
        """Render a per-agent coverage heatmap in ASCII.

        Each schedule is expected to be a sequence of agent-id strings.
        The heatmap counts how often each agent appears at each position
        index across all schedules.

        Args:
            schedules: List of schedules (each a sequence of agent ids).
            agent_ids: Ordered list of agent identifiers.

        Returns:
            Multi-line heatmap string.
        """
        if not schedules:
            return "(no schedules)"
        max_len = max(len(s) for s in schedules)
        counts: dict[str, list[int]] = {a: [0] * max_len for a in agent_ids}
        for sched in schedules:
            for pos, aid in enumerate(sched):
                if aid in counts:
                    counts[aid][pos] += 1

        heat_chars = " ░▒▓█"
        peak = max(max(v) for v in counts.values()) if counts else 1
        peak = max(peak, 1)

        buf = io.StringIO()
        header_positions = "".join(f"{i % 10}" for i in range(max_len))
        label_w = max(len(a) for a in agent_ids)
        buf.write(f"{'':>{label_w}} pos: {header_positions}\n")
        for aid in agent_ids:
            row = ""
            for c in counts[aid]:
                idx = int(c / peak * (len(heat_chars) - 1))
                row += heat_chars[idx]
            buf.write(f"{aid:>{label_w}}    : {row}\n")
        return buf.getvalue().rstrip("\n")


# ---------------------------------------------------------------------------
# RaceRegionVisualizer
# ---------------------------------------------------------------------------


class RaceRegionVisualizer:
    """Highlight race regions in state space."""

    def plot(
        self,
        safe_regions: list[dict[str, Any]],
        race_regions: list[dict[str, Any]],
    ) -> Any:
        """Render race vs safe regions on a matplotlib figure.

        Each region dict is expected to have ``"x_min"``, ``"x_max"``,
        ``"y_min"``, ``"y_max"`` keys.

        Returns:
            A matplotlib figure or ``None``.
        """
        if not _HAS_MPL:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        for region in safe_regions:
            rect = mpatches.Rectangle(
                (region["x_min"], region["y_min"]),
                region["x_max"] - region["x_min"],
                region["y_max"] - region["y_min"],
                linewidth=1, edgecolor="steelblue", facecolor="lightblue", alpha=0.4,
            )
            ax.add_patch(rect)
        for region in race_regions:
            rect = mpatches.Rectangle(
                (region["x_min"], region["y_min"]),
                region["x_max"] - region["x_min"],
                region["y_max"] - region["y_min"],
                linewidth=1, edgecolor="crimson", facecolor="salmon", alpha=0.5,
            )
            ax.add_patch(rect)
        ax.autoscale_view()
        ax.set_title("Race Region Map")
        ax.legend(
            handles=[
                mpatches.Patch(color="lightblue", label="Safe"),
                mpatches.Patch(color="salmon", label="Race"),
            ]
        )
        return fig

    def to_ascii(
        self,
        safe: list[dict[str, Any]],
        race: list[dict[str, Any]],
        width: int = 60,
        height: int = 20,
    ) -> str:
        """Render an ASCII map of race vs safe regions.

        Args:
            safe: Safe region dicts with ``x_min/x_max/y_min/y_max``.
            race: Race region dicts with the same keys.
            width: Character width.
            height: Character height.

        Returns:
            Multi-line ASCII region map.
        """
        all_regions = safe + race
        if not all_regions:
            return "(no regions)"
        gx_min = min(r["x_min"] for r in all_regions)
        gx_max = max(r["x_max"] for r in all_regions)
        gy_min = min(r["y_min"] for r in all_regions)
        gy_max = max(r["y_max"] for r in all_regions)
        x_span = gx_max - gx_min if gx_max != gx_min else 1.0
        y_span = gy_max - gy_min if gy_max != gy_min else 1.0

        grid: list[list[str]] = [[" "] * width for _ in range(height)]

        def _fill(regions: list[dict[str, Any]], char: str) -> None:
            for reg in regions:
                c0 = int((reg["x_min"] - gx_min) / x_span * (width - 1))
                c1 = int((reg["x_max"] - gx_min) / x_span * (width - 1))
                r0 = height - 1 - int((reg["y_max"] - gy_min) / y_span * (height - 1))
                r1 = height - 1 - int((reg["y_min"] - gy_min) / y_span * (height - 1))
                for r in range(max(r0, 0), min(r1 + 1, height)):
                    for c in range(max(c0, 0), min(c1 + 1, width)):
                        grid[r][c] = char

        _fill(safe, "░")
        _fill(race, "█")

        buf = io.StringIO()
        buf.write("Race Region Map  (░=safe  █=race)\n")
        for row in grid:
            buf.write("".join(row) + "\n")
        return buf.getvalue().rstrip("\n")

    def overlay_on_trajectory(
        self,
        trajectory: list[dict[str, Any]],
        race_regions: list[dict[str, Any]],
    ) -> str:
        """Annotate a trajectory with race-region membership.

        Each trajectory entry is expected to have ``"x"`` and ``"y"`` keys.

        Args:
            trajectory: Ordered list of state dicts with ``"x"``/``"y"``.
            race_regions: Race region dicts with bounding-box keys.

        Returns:
            Multi-line annotated trajectory listing.
        """

        def _in_region(pt: dict[str, Any], reg: dict[str, Any]) -> bool:
            return (
                reg["x_min"] <= pt["x"] <= reg["x_max"]
                and reg["y_min"] <= pt["y"] <= reg["y_max"]
            )

        lines: list[str] = ["Trajectory with race overlay:"]
        for i, pt in enumerate(trajectory):
            in_race = any(_in_region(pt, r) for r in race_regions)
            marker = "⚠ RACE" if in_race else "  safe"
            lines.append(f"  step {i:3d}: ({pt['x']:7.2f}, {pt['y']:7.2f})  {marker}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# InteractionGroupVisualizer
# ---------------------------------------------------------------------------


class InteractionGroupVisualizer:
    """Visualize agent interaction groups as a graph / matrix."""

    def plot(
        self,
        groups: list[list[str]],
        interactions: list[tuple[str, str]],
    ) -> Any:
        """Render interaction groups on a matplotlib figure.

        Args:
            groups: Clusters of agent ids.
            interactions: Pairwise agent interactions.

        Returns:
            A matplotlib figure or ``None``.
        """
        if not _HAS_MPL:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        all_agents: list[str] = []
        group_map: dict[str, int] = {}
        for gidx, group in enumerate(groups):
            for agent in group:
                all_agents.append(agent)
                group_map[agent] = gidx

        n = len(all_agents)
        angle_step = 2 * math.pi / max(n, 1)
        pos: dict[str, tuple[float, float]] = {}
        for i, agent in enumerate(all_agents):
            pos[agent] = (math.cos(i * angle_step), math.sin(i * angle_step))

        colors = ["steelblue", "orange", "green", "purple", "brown", "pink"]
        for agent in all_agents:
            x, y = pos[agent]
            c = colors[group_map[agent] % len(colors)]
            ax.scatter(x, y, c=c, s=200, zorder=3)
            ax.annotate(agent, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        for a, b in interactions:
            if a in pos and b in pos:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        "k-", alpha=0.3)

        ax.set_title("Interaction Groups")
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    def to_ascii(
        self,
        groups: list[list[str]],
        interactions: list[tuple[str, str]],
    ) -> str:
        """Render an adjacency-matrix display of agent interactions.

        Groups are indicated by labelled rows/columns; interactions by
        ``'●'`` marks.

        Args:
            groups: Agent clusters.
            interactions: Pairwise interactions.

        Returns:
            Multi-line adjacency matrix string.
        """
        all_agents: list[str] = [a for g in groups for a in g]
        if not all_agents:
            return "(no agents)"

        idx = {a: i for i, a in enumerate(all_agents)}
        n = len(all_agents)
        matrix = [[" "] * n for _ in range(n)]
        for a, b in interactions:
            if a in idx and b in idx:
                matrix[idx[a]][idx[b]] = "●"
                matrix[idx[b]][idx[a]] = "●"

        label_w = max(len(a) for a in all_agents)
        buf = io.StringIO()

        # Group legend
        for gi, group in enumerate(groups):
            buf.write(f"  Group {gi}: {', '.join(group)}\n")
        buf.write("\n")

        # Column headers (single-char abbreviations)
        col_hdr = " " * (label_w + 2)
        for a in all_agents:
            col_hdr += a[0]
        buf.write(col_hdr + "\n")
        buf.write(" " * (label_w + 1) + "┌" + "─" * n + "┐\n")

        for i, agent in enumerate(all_agents):
            row_str = "".join(matrix[i])
            buf.write(f"{agent:>{label_w}} │{row_str}│\n")

        buf.write(" " * (label_w + 1) + "└" + "─" * n + "┘\n")
        return buf.getvalue().rstrip("\n")


# ---------------------------------------------------------------------------
# ConvergenceVisualizer
# ---------------------------------------------------------------------------


class ConvergenceVisualizer:
    """Plot fixpoint convergence (abstract-state diameter vs iteration)."""

    def plot(self, iteration_data: list[dict[str, Any]]) -> Any:
        """Render a convergence line plot.

        Each dict in *iteration_data* should contain ``"iteration"`` (int)
        and ``"diameter"`` (float) keys.

        Returns:
            A matplotlib figure or ``None``.
        """
        if not _HAS_MPL:
            return None

        iters = [d["iteration"] for d in iteration_data]
        diams = [d["diameter"] for d in iteration_data]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iters, diams, "o-", color="steelblue")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Abstract-state diameter")
        ax.set_title("Fixpoint Convergence")
        ax.grid(True, alpha=0.3)
        return fig

    def to_ascii(
        self,
        iteration_data: list[dict[str, Any]],
        width: int = 60,
        height: int = 15,
    ) -> str:
        """Render an ASCII line chart of convergence.

        Args:
            iteration_data: Dicts with ``"iteration"`` and ``"diameter"``.
            width: Plot width in characters.
            height: Plot height in characters.

        Returns:
            Multi-line ASCII chart.
        """
        if not iteration_data:
            return "(no data)"

        diams = [d["diameter"] for d in iteration_data]
        lo, hi = min(diams), max(diams)
        span = hi - lo if hi != lo else 1.0

        plot_w = width - 8  # y-axis label space
        n = len(diams)
        grid: list[list[str]] = [[" "] * plot_w for _ in range(height)]

        for i, d in enumerate(diams):
            col = int(i / max(n - 1, 1) * (plot_w - 1))
            row = height - 1 - int((d - lo) / span * (height - 1))
            row = max(0, min(height - 1, row))
            col = max(0, min(plot_w - 1, col))
            grid[row][col] = "●"

        # Connect adjacent points with simple vertical fills
        for i in range(len(diams) - 1):
            c1 = int(i / max(n - 1, 1) * (plot_w - 1))
            c2 = int((i + 1) / max(n - 1, 1) * (plot_w - 1))
            r1 = height - 1 - int((diams[i] - lo) / span * (height - 1))
            r2 = height - 1 - int((diams[i + 1] - lo) / span * (height - 1))
            r1 = max(0, min(height - 1, r1))
            r2 = max(0, min(height - 1, r2))
            if c2 - c1 > 1:
                for c in range(c1 + 1, c2):
                    frac = (c - c1) / (c2 - c1)
                    r = int(r1 + frac * (r2 - r1))
                    r = max(0, min(height - 1, r))
                    if grid[r][c] == " ":
                        grid[r][c] = "·"

        buf = io.StringIO()
        buf.write("Fixpoint Convergence\n")
        for r, row in enumerate(grid):
            if r == 0:
                label = f"{hi:6.2f}"
            elif r == height - 1:
                label = f"{lo:6.2f}"
            else:
                label = "      "
            buf.write(f"{label} |{''.join(row)}\n")
        buf.write("       +" + "─" * plot_w + "\n")
        buf.write(f"        iter 0{' ' * (plot_w - 10)}iter {n - 1}\n")
        return buf.getvalue().rstrip("\n")


# ---------------------------------------------------------------------------
# MetricsDashboard
# ---------------------------------------------------------------------------


class MetricsDashboard:
    """Summary dashboard combining key verification metrics."""

    def __init__(self) -> None:
        pass

    def render(self, metrics: dict[str, Any]) -> str:
        """Render a full ASCII dashboard.

        *metrics* is a flexible dict; recognised top-level keys include:

        - ``"states_explored"`` / ``"states_total"`` — state coverage.
        - ``"schedules_explored"`` / ``"schedules_bound"`` — schedule coverage.
        - ``"races_found"`` — number of races detected.
        - ``"convergence_iterations"`` — iterations to fixpoint.
        - ``"agents"`` — number of agents.
        - ``"time_elapsed_s"`` — wall-clock seconds.
        - ``"extra"`` — dict of additional key/value pairs.

        Returns:
            Multi-line dashboard string.
        """
        sections: list[str] = []

        # -- Coverage --
        coverage_lines: list[str] = []
        se = metrics.get("states_explored", 0)
        st = metrics.get("states_total", 0)
        if st > 0:
            coverage_lines.append(
                self._render_bar("States", se, st, 40)
            )
        sce = metrics.get("schedules_explored", 0)
        scb = metrics.get("schedules_bound", 0)
        if scb > 0:
            coverage_lines.append(
                self._render_bar("Schedules", sce, scb, 40)
            )
        if coverage_lines:
            sections.append(self._render_box("Coverage", "\n".join(coverage_lines), 58))

        # -- Results --
        result_lines: list[str] = []
        rf = metrics.get("races_found", 0)
        result_lines.append(f"Races found      : {rf}")
        ci = metrics.get("convergence_iterations")
        if ci is not None:
            result_lines.append(f"Convergence iters: {ci}")
        na = metrics.get("agents")
        if na is not None:
            result_lines.append(f"Agents           : {na}")
        te = metrics.get("time_elapsed_s")
        if te is not None:
            result_lines.append(f"Time elapsed     : {te:.2f}s")
        sections.append(self._render_box("Results", "\n".join(result_lines), 58))

        # -- Extra --
        extra = metrics.get("extra", {})
        if extra:
            extra_lines = [f"{k}: {v}" for k, v in extra.items()]
            sections.append(self._render_box("Details", "\n".join(extra_lines), 58))

        return "\n".join(sections)

    @staticmethod
    def _render_box(title: str, content: str, width: int) -> str:
        """Render a titled box around *content*."""
        return ASCIIVisualizer.box(title, content.split("\n"), width)

    @staticmethod
    def _render_bar(label: str, value: float, max_value: float, width: int) -> str:
        """Render a labelled progress bar with percentage."""
        bar = ASCIIVisualizer.horizontal_bar(value, max_value, width)
        pct = value / max_value * 100 if max_value > 0 else 0.0
        return f"{label:>12s} {bar} {pct:5.1f}%"
