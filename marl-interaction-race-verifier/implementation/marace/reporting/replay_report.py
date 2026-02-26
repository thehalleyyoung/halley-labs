"""Replay report generation and visualization for adversarial schedule replays.

Provides dataclasses and renderers for inspecting, comparing, and visualizing
the adversarial schedules discovered by the MARACE race-condition verifier.
"""

from __future__ import annotations

import textwrap
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ReplayReport:
    """Detailed description of an adversarial replay.

    Captures the full schedule of agent actions, the resulting state trajectory,
    and metadata about the safety violation that was triggered.

    Attributes:
        replay_id: Unique identifier for this replay.
        race_id: Identifier of the race condition this replay demonstrates.
        schedule: Ordered list of ``(agent_id, action)`` pairs executed.
        states: Snapshot of the environment state after each scheduled step.
        critical_step: Index into *schedule* where the safety violation becomes
            unavoidable.
        total_steps: Total number of steps in the replay (``len(schedule)``).
        agents_involved: Agent identifiers that participate in the race.
        safety_violation: Human-readable description of the violated property.
        probability_estimate: Estimated probability of this schedule under the
            agents' joint policy (0.0–1.0).
        timestamp: When the replay was generated.
    """

    replay_id: str
    race_id: str
    schedule: List[Tuple[str, str]]
    states: List[Dict[str, object]]
    critical_step: int
    total_steps: int
    agents_involved: List[str]
    safety_violation: str
    probability_estimate: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def step_agent(self, step: int) -> str:
        """Return the agent id that acts at *step*."""
        return self.schedule[step][0]

    def step_action(self, step: int) -> str:
        """Return the action taken at *step*."""
        return self.schedule[step][1]

    def state_at(self, step: int) -> Dict[str, object]:
        """Return the state snapshot recorded after *step*."""
        return self.states[step]


class TimelineRenderer:
    """Render a :class:`ReplayReport` as a human-readable ASCII timeline.

    Two layout modes are supported: *horizontal* (steps go left-to-right) and
    *vertical* (steps go top-to-bottom).  The default :meth:`render` delegates
    to :meth:`render_vertical`.
    """

    CRITICAL_MARKER = ">>>"
    STEP_MARKER = "   "

    def render(self, report: ReplayReport) -> str:
        """Return a full ASCII timeline (vertical layout by default)."""
        return self.render_vertical(report)

    # ---- horizontal layout ------------------------------------------------

    def render_horizontal(self, report: ReplayReport) -> str:
        """Render the schedule as a horizontal ASCII timeline.

        Each agent gets its own row.  Steps are columns separated by ``|``.
        The critical step column is bracketed with ``>>>``.
        """
        agents = sorted(report.agents_involved)
        col_width = 14

        # Header row: step indices
        header_cells = [f"{'Step':>{col_width}}"]
        for i in range(report.total_steps):
            label = f"[{i}]" if i != report.critical_step else f">>>[{i}]<<<"
            header_cells.append(f"{label:^{col_width}}")
        lines: List[str] = ["|".join(header_cells)]
        lines.append("-" * len(lines[0]))

        # One row per agent
        for agent in agents:
            row_cells = [f"{agent:>{col_width}}"]
            for i in range(report.total_steps):
                aid, action = report.schedule[i]
                cell = action if aid == agent else ""
                row_cells.append(f"{cell:^{col_width}}")
            lines.append("|".join(row_cells))

        return "\n".join(lines)

    # ---- vertical layout --------------------------------------------------

    def render_vertical(self, report: ReplayReport) -> str:
        """Render the schedule as a vertical ASCII timeline.

        Each line represents one step.  The agent and action are shown, and the
        critical step is prominently marked.
        """
        agents = sorted(report.agents_involved)
        agent_col_w = max((len(a) for a in agents), default=5) + 2
        action_col_w = max(
            (len(act) for _, act in report.schedule), default=8
        ) + 2

        buf = io.StringIO()
        buf.write(f"Replay {report.replay_id}  (race {report.race_id})\n")
        buf.write(f"Violation: {report.safety_violation}\n")
        buf.write(
            f"P(schedule) ≈ {report.probability_estimate:.4e}\n"
        )
        buf.write("=" * (12 + agent_col_w + action_col_w) + "\n")

        for i in range(report.total_steps):
            agent_id, action = report.schedule[i]
            marker = self.CRITICAL_MARKER if i == report.critical_step else self.STEP_MARKER
            step_label = f"{marker} Step {i:>3d}"
            buf.write(
                f"{step_label} | {agent_id:<{agent_col_w}} | {action:<{action_col_w}}\n"
            )
            if i == report.critical_step:
                rule = "-" * (12 + agent_col_w + action_col_w)
                buf.write(f"{rule}\n")
                buf.write(
                    f"{'':>12} *** CRITICAL MOMENT *** {report.safety_violation}\n"
                )
                buf.write(f"{rule}\n")

        buf.write("=" * (12 + agent_col_w + action_col_w) + "\n")
        return buf.getvalue()


class StateTrajectoryPlotter:
    """Extract and visualise state-dimension trajectories from a replay.

    Since MARACE avoids heavy dependencies, plotting is done as plain-text
    ASCII art or CSV export rather than bitmap images.
    """

    def plot(
        self,
        report: ReplayReport,
        dimensions: List[str],
    ) -> Dict[str, Dict[str, List[object]]]:
        """Build per-agent plot data for the requested state *dimensions*.

        Returns a dict keyed by agent id.  Each value is a dict with keys
        ``"x"`` (step indices), ``"y"`` (mapping dimension → values), and
        ``"labels"`` (action labels at each step).

        Missing dimension values are recorded as ``None``.
        """
        agents = sorted(report.agents_involved)
        result: Dict[str, Dict[str, object]] = {}

        for agent in agents:
            xs: List[int] = []
            ys: Dict[str, List[object]] = {d: [] for d in dimensions}
            labels: List[str] = []

            for i in range(report.total_steps):
                aid, action = report.schedule[i]
                if aid != agent:
                    continue
                xs.append(i)
                labels.append(action)
                state = report.states[i] if i < len(report.states) else {}
                for d in dimensions:
                    ys[d].append(state.get(d))

            result[agent] = {"x": xs, "y": ys, "labels": labels}

        return result

    # ---- ASCII rendering ---------------------------------------------------

    def to_ascii(
        self,
        plot_data: Dict[str, Dict[str, List[object]]],
        *,
        width: int = 60,
        height: int = 16,
    ) -> str:
        """Render *plot_data* as an ASCII scatter / line chart.

        Only the first numeric dimension found per agent is plotted.  The
        x-axis is the step index and the y-axis is the dimension value.
        """
        all_x: List[float] = []
        all_y: List[float] = []
        series: List[Tuple[str, List[float], List[float]]] = []

        for agent, data in plot_data.items():
            xs = [float(v) for v in data["x"]]
            dim_name = next(iter(data["y"]), None)
            if dim_name is None:
                continue
            raw_ys = data["y"][dim_name]
            ys = [float(v) if v is not None else 0.0 for v in raw_ys]
            if not xs:
                continue
            series.append((agent, xs, ys))
            all_x.extend(xs)
            all_y.extend(ys)

        if not all_x or not all_y:
            return "(no numeric data to plot)\n"

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = x_max - x_min or 1.0
        y_range = y_max - y_min or 1.0

        # Build a character grid
        grid = [[" "] * width for _ in range(height)]
        markers = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        legend_lines: List[str] = []
        for idx, (agent, xs, ys) in enumerate(series):
            marker = markers[idx % len(markers)]
            legend_lines.append(f"  {marker} = {agent}")
            for xi, yi in zip(xs, ys):
                col = int((xi - x_min) / x_range * (width - 1))
                row = height - 1 - int((yi - y_min) / y_range * (height - 1))
                col = max(0, min(width - 1, col))
                row = max(0, min(height - 1, row))
                grid[row][col] = marker

        buf = io.StringIO()
        y_label_w = max(len(f"{y_max:.2f}"), len(f"{y_min:.2f}"))
        for r in range(height):
            if r == 0:
                label = f"{y_max:.2f}"
            elif r == height - 1:
                label = f"{y_min:.2f}"
            else:
                label = ""
            buf.write(f"{label:>{y_label_w}} |{''.join(grid[r])}\n")
        buf.write(f"{' ' * y_label_w} +{'-' * width}\n")
        buf.write(
            f"{' ' * y_label_w}  {x_min:.0f}"
            f"{' ' * (width - len(f'{x_min:.0f}') - len(f'{x_max:.0f}'))}"
            f"{x_max:.0f}\n"
        )
        buf.write("Legend:\n")
        buf.write("\n".join(legend_lines) + "\n")
        return buf.getvalue()

    # ---- CSV export --------------------------------------------------------

    def to_csv(
        self,
        plot_data: Dict[str, Dict[str, List[object]]],
    ) -> str:
        """Serialise *plot_data* to CSV text (one row per agent per step)."""
        buf = io.StringIO()

        # Collect all dimension names across agents
        all_dims: List[str] = []
        for data in plot_data.values():
            for d in data["y"]:
                if d not in all_dims:
                    all_dims.append(d)

        header = ["agent", "step", "action"] + all_dims
        buf.write(",".join(header) + "\n")

        for agent, data in sorted(plot_data.items()):
            xs = data["x"]
            labels = data["labels"]
            for j, step in enumerate(xs):
                action = labels[j] if j < len(labels) else ""
                dim_vals = [
                    str(data["y"].get(d, [None] * len(xs))[j])
                    if j < len(data["y"].get(d, []))
                    else ""
                    for d in all_dims
                ]
                row = [agent, str(step), action] + dim_vals
                buf.write(",".join(row) + "\n")

        return buf.getvalue()


class CriticalMomentHighlighter:
    """Identify and format the critical moment in an adversarial replay.

    The *critical step* recorded in the report is the single point of no
    return.  This class widens that to a contextual window and attempts to
    reconstruct a causal chain leading to the violation.
    """

    def identify_critical_window(
        self,
        report: ReplayReport,
        window_size: int = 3,
    ) -> Tuple[int, int]:
        """Return ``(start, end)`` indices of the critical window.

        The window is centred on :pyattr:`ReplayReport.critical_step` and
        clamped to valid step bounds.
        """
        half = window_size // 2
        start = max(0, report.critical_step - half)
        end = min(report.total_steps - 1, report.critical_step + half)
        return start, end

    def highlight(self, report: ReplayReport) -> str:
        """Return formatted text showing the critical moment with context.

        Steps inside the critical window are indented and annotated.  The
        critical step itself is marked with an arrow.
        """
        start, end = self.identify_critical_window(report)
        buf = io.StringIO()

        buf.write(f"Critical Moment Analysis  (replay {report.replay_id})\n")
        buf.write(f"Violation: {report.safety_violation}\n")
        buf.write(f"Critical step: {report.critical_step}\n")
        buf.write(f"Window: steps {start}–{end}\n")
        buf.write("-" * 60 + "\n")

        for i in range(report.total_steps):
            agent_id, action = report.schedule[i]
            in_window = start <= i <= end
            is_critical = i == report.critical_step

            prefix = "  "
            if is_critical:
                prefix = "→ "
            elif in_window:
                prefix = "· "

            line = f"{prefix}[{i:>3d}] {agent_id}: {action}"

            if is_critical:
                line += "  ← CRITICAL"
            buf.write(line + "\n")

            if is_critical:
                buf.write(
                    textwrap.indent(
                        f"Safety property violated: {report.safety_violation}\n",
                        "        ",
                    )
                )
                if i < len(report.states):
                    state_summary = ", ".join(
                        f"{k}={v}" for k, v in report.states[i].items()
                    )
                    buf.write(
                        textwrap.indent(
                            f"State: {{{state_summary}}}\n", "        "
                        )
                    )

        buf.write("-" * 60 + "\n")
        return buf.getvalue()

    def extract_causal_chain(self, report: ReplayReport) -> List[str]:
        """Reconstruct a simplified causal chain ending at the violation.

        Each entry is a human-readable sentence describing the step and its
        consequence.  The chain runs from step 0 to the critical step
        (inclusive).
        """
        chain: List[str] = []
        for i in range(report.critical_step + 1):
            agent_id, action = report.schedule[i]
            state_desc = ""
            if i < len(report.states):
                keys = list(report.states[i].keys())
                if keys:
                    state_desc = f" → state change in {', '.join(keys)}"

            if i < report.critical_step:
                chain.append(
                    f"Step {i}: Agent '{agent_id}' performs '{action}'{state_desc}"
                )
            else:
                chain.append(
                    f"Step {i} [CRITICAL]: Agent '{agent_id}' performs '{action}'"
                    f" triggering violation: {report.safety_violation}"
                )
        return chain


class CounterfactualComparison:
    """Compare a violating schedule against a safe alternative side-by-side.

    Useful for understanding *why* a particular interleaving is dangerous
    while a closely-related one is safe.
    """

    def compare(
        self,
        unsafe_replay: ReplayReport,
        safe_replay: ReplayReport,
    ) -> str:
        """Return a side-by-side textual diff of two replays.

        Steps before the divergence point are shown once; after that both
        columns are printed so the reader can see where the schedules part.
        """
        div = self._find_divergence_point(unsafe_replay, safe_replay)
        return self._format_side_by_side(
            unsafe_replay.schedule,
            safe_replay.schedule,
            div,
        )

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _find_divergence_point(
        unsafe: ReplayReport,
        safe: ReplayReport,
    ) -> int:
        """Return the first step index where the two schedules differ."""
        limit = min(unsafe.total_steps, safe.total_steps)
        for i in range(limit):
            if unsafe.schedule[i] != safe.schedule[i]:
                return i
        return limit

    @staticmethod
    def _format_side_by_side(
        unsafe_steps: Sequence[Tuple[str, str]],
        safe_steps: Sequence[Tuple[str, str]],
        divergence_point: int,
    ) -> str:
        """Format two schedules in two columns with a divergence marker."""
        col_w = 30
        buf = io.StringIO()

        buf.write(f"{'UNSAFE':^{col_w}} | {'SAFE':^{col_w}}\n")
        buf.write(f"{'-' * col_w}-+-{'-' * col_w}\n")

        max_len = max(len(unsafe_steps), len(safe_steps))
        for i in range(max_len):
            if i < len(unsafe_steps):
                ua, uact = unsafe_steps[i]
                left = f"[{i}] {ua}: {uact}"
            else:
                left = ""

            if i < len(safe_steps):
                sa, sact = safe_steps[i]
                right = f"[{i}] {sa}: {sact}"
            else:
                right = ""

            marker = " "
            if i == divergence_point:
                marker = "*"

            buf.write(f"{left:<{col_w}} |{marker}{right:<{col_w}}\n")

            if i == divergence_point:
                note = "  ^^^ schedules diverge here ^^^"
                buf.write(f"{note:^{col_w * 2 + 3}}\n")

        buf.write(f"{'-' * col_w}-+-{'-' * col_w}\n")
        return buf.getvalue()

    def generate_what_if(self, replay: ReplayReport) -> List[str]:
        """Generate natural-language what-if questions for the replay.

        Each question suggests a single schedule alteration at or near the
        critical step and asks whether the violation would still occur.
        """
        crit = replay.critical_step
        agent_id, action = replay.schedule[crit]
        questions: List[str] = []

        questions.append(
            f"What if agent '{agent_id}' did NOT perform '{action}' at step {crit}?"
        )

        if crit > 0:
            prev_agent, prev_action = replay.schedule[crit - 1]
            questions.append(
                f"What if step {crit - 1} ('{prev_agent}': '{prev_action}') "
                f"and step {crit} were swapped?"
            )

        other_agents = [a for a in replay.agents_involved if a != agent_id]
        if other_agents:
            alt = other_agents[0]
            questions.append(
                f"What if agent '{alt}' acted at step {crit} instead of "
                f"agent '{agent_id}'?"
            )

        if crit + 1 < replay.total_steps:
            next_agent, next_action = replay.schedule[crit + 1]
            questions.append(
                f"What if agent '{agent_id}' waited and '{next_agent}' "
                f"performed '{next_action}' first?"
            )

        questions.append(
            f"What if a lock prevented concurrent access at step {crit}?"
        )

        return questions
