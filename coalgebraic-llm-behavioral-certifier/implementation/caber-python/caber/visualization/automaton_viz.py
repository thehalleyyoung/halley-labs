"""
CABER — Coalgebraic Behavioral Auditing of Foundation Models.

Text-based automaton visualization for terminal output.

Provides renderers for state diagrams, transition tables, distribution
bar-charts, transition matrices, compact path notation, and side-by-side
automaton comparison — all without external dependencies beyond stdlib.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data models (defined locally to avoid circular imports from evaluation)
# ---------------------------------------------------------------------------

@dataclass
class VizState:
    """A single state in an automaton visualisation."""

    state_id: str
    label: str
    is_initial: bool = False
    is_accepting: bool = False
    x: float = 0.0
    y: float = 0.0


@dataclass
class VizTransition:
    """A labelled, probabilistic transition between two states."""

    source: str
    target: str
    label: str
    probability: float = 1.0


@dataclass
class VizAutomaton:
    """Complete automaton ready for visualisation."""

    states: List[VizState] = field(default_factory=list)
    transitions: List[VizTransition] = field(default_factory=list)
    title: str = ""


@dataclass
class TableCell:
    """A single cell inside a rendered table."""

    content: str
    width: int = 12
    align: str = "left"


@dataclass
class BarSpec:
    """Specification for one horizontal bar in a bar-chart."""

    label: str
    value: float
    max_value: float = 1.0
    bar_char: str = "█"


# ---------------------------------------------------------------------------
# Core visualiser
# ---------------------------------------------------------------------------

class AutomatonVisualizer:
    """Text-based automaton visualiser for terminal / log output."""

    # Unicode box-drawing pieces
    _UNI = {
        "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
        "h": "─", "v": "│", "tj": "┬", "bj": "┴",
        "lj": "├", "rj": "┤", "cross": "┼",
        "arrow_r": "→", "arrow_l": "←",
        "arrow_d": "↓", "arrow_u": "↑",
        "double_tl": "╔", "double_tr": "╗",
        "double_bl": "╚", "double_br": "╝",
        "double_h": "═", "double_v": "║",
        "self_loop": "↺",
    }

    # ASCII fallback
    _ASCII = {
        "tl": "+", "tr": "+", "bl": "+", "br": "+",
        "h": "-", "v": "|", "tj": "+", "bj": "+",
        "lj": "+", "rj": "+", "cross": "+",
        "arrow_r": "->", "arrow_l": "<-",
        "arrow_d": "v", "arrow_u": "^",
        "double_tl": "+", "double_tr": "+",
        "double_bl": "+", "double_br": "+",
        "double_h": "=", "double_v": "‖",
        "self_loop": "@",
    }

    def __init__(self, max_width: int = 120, use_unicode: bool = True) -> None:
        """Initialise the visualiser.

        Args:
            max_width: Maximum character width of rendered output.
            use_unicode: Use Unicode box-drawing characters when *True*;
                fall back to plain ASCII when *False*.
        """
        self.max_width = max_width
        self.use_unicode = use_unicode
        self._ch = self._UNI if use_unicode else self._ASCII

    # ------------------------------------------------------------------
    # Public renderers
    # ------------------------------------------------------------------

    def render_state_diagram(self, automaton: VizAutomaton) -> str:
        """Render a text-based state-transition diagram.

        States are drawn as boxes arranged in a grid.  Transitions appear
        as labelled arrows.  Initial states get an incoming ``→`` marker,
        accepting states get a double border (Unicode) or ``*`` (ASCII).
        Self-loops are annotated beneath the owning state.

        Args:
            automaton: The automaton to render.

        Returns:
            Multi-line string containing the complete diagram.
        """
        if not automaton.states:
            return "(empty automaton)"

        laid_out = self._layout_states(automaton.states, automaton.transitions)

        # Build lookup structures
        state_map: Dict[str, VizState] = {s.state_id: s for s in laid_out}
        trans_by_pair: Dict[Tuple[str, str], List[VizTransition]] = defaultdict(list)
        self_loops: Dict[str, List[VizTransition]] = defaultdict(list)
        for t in automaton.transitions:
            if t.source == t.target:
                self_loops[t.source].append(t)
            else:
                trans_by_pair[(t.source, t.target)].append(t)

        # Determine grid bounds
        col_set = sorted({int(s.x) for s in laid_out})
        row_set = sorted({int(s.y) for s in laid_out})
        col_idx = {c: i for i, c in enumerate(col_set)}
        row_idx = {r: i for i, r in enumerate(row_set)}

        box_width = 12
        box_height = 3
        h_spacing = box_width + 8
        v_spacing = box_height + 4

        canvas_w = max(len(col_set) * h_spacing + box_width, 40)
        canvas_h = max(len(row_set) * v_spacing + box_height + 4, 10)
        canvas: List[List[str]] = [[" "] * canvas_w for _ in range(canvas_h)]

        def _put(r: int, c: int, text: str) -> None:
            for i, ch in enumerate(text):
                if 0 <= r < canvas_h and 0 <= c + i < canvas_w:
                    canvas[r][c + i] = ch

        # Draw state boxes
        box_positions: Dict[str, Tuple[int, int]] = {}
        for st in laid_out:
            ci = col_idx[int(st.x)]
            ri = row_idx[int(st.y)]
            left = ci * h_spacing + 4
            top = ri * v_spacing + 1
            box_positions[st.state_id] = (top, left)
            box_lines = self._draw_box(st.label, st.is_initial, st.is_accepting)
            for dl, line in enumerate(box_lines):
                _put(top + dl, left, line)

            # Initial-state marker
            if st.is_initial:
                arr = self._ch["arrow_r"]
                _put(top + 1, left - len(arr) - 1, arr + " ")

            # Self-loop annotation
            if st.state_id in self_loops:
                loop_labels = ", ".join(
                    f"{sl.label}/{sl.probability:.2f}" for sl in self_loops[st.state_id]
                )
                note = f"{self._ch['self_loop']} {loop_labels}"
                _put(top + box_height, left, note)

        # Draw transitions between distinct states
        for (src, tgt), t_list in trans_by_pair.items():
            if src not in box_positions or tgt not in box_positions:
                continue
            sr, sc = box_positions[src]
            tr, tc = box_positions[tgt]
            combined = ", ".join(
                f"{t.label}/{t.probability:.2f}" for t in t_list
            )
            mid_r = (sr + tr) // 2
            mid_c = (sc + tc) // 2
            arrow = self._draw_arrow("right" if tc >= sc else "left", combined)
            _put(mid_r + 1, mid_c, arrow)

        # Compose
        lines: List[str] = []
        if automaton.title:
            lines.append(automaton.title)
            lines.append(self._ch["h"] * min(len(automaton.title), self.max_width))
        for row in canvas:
            rendered = "".join(row).rstrip()
            if rendered:
                lines.append(rendered)
        # Trim trailing blank lines
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def render_transition_table(self, automaton: VizAutomaton) -> str:
        """Render a formatted transition table.

        Columns: Source | Symbol | Target | Probability.
        Rows are sorted by source state then symbol.  High-probability
        transitions (≥ 0.8) are highlighted with a trailing ``*``.

        Args:
            automaton: The automaton whose transitions to tabulate.

        Returns:
            Formatted table string.
        """
        headers = ["Source", "Symbol", "Target", "Probability"]
        sorted_trans = sorted(
            automaton.transitions, key=lambda t: (t.source, t.label)
        )
        rows: List[List[str]] = []
        for t in sorted_trans:
            prob_str = f"{t.probability:.4f}"
            if t.probability >= 0.8:
                prob_str += " *"
            rows.append([t.source, t.label, t.target, prob_str])
        title = "Transition Table"
        if automaton.title:
            title = f"Transition Table — {automaton.title}"
        return f"{title}\n{self._format_table(headers, rows)}"

    def render_state_info_table(self, automaton: VizAutomaton) -> str:
        """Render a table of per-state information.

        Columns: State | Type | In-Degree | Out-Degree | Self-Loop |
        Dominant Behavior.

        Args:
            automaton: The automaton to analyse.

        Returns:
            Formatted table string.
        """
        in_deg: Dict[str, int] = defaultdict(int)
        out_deg: Dict[str, int] = defaultdict(int)
        self_loop_labels: Dict[str, List[str]] = defaultdict(list)
        out_trans: Dict[str, List[VizTransition]] = defaultdict(list)

        for t in automaton.transitions:
            out_deg[t.source] += 1
            in_deg[t.target] += 1
            out_trans[t.source].append(t)
            if t.source == t.target:
                self_loop_labels[t.source].append(t.label)

        headers = ["State", "Type", "In-Deg", "Out-Deg", "Self-Loop", "Dominant Behavior"]
        rows: List[List[str]] = []
        for st in sorted(automaton.states, key=lambda s: s.state_id):
            stype_parts: List[str] = []
            if st.is_initial:
                stype_parts.append("init")
            if st.is_accepting:
                stype_parts.append("accept")
            if not stype_parts:
                stype_parts.append("normal")
            stype = ", ".join(stype_parts)

            loop = ", ".join(self_loop_labels.get(st.state_id, [])) or "-"

            # Dominant behaviour = highest-probability outgoing transition
            ot = out_trans.get(st.state_id, [])
            if ot:
                best = max(ot, key=lambda t: t.probability)
                dominant = f"{best.label}{self._ch['arrow_r']}{best.target} ({best.probability:.2f})"
            else:
                dominant = "-"

            rows.append([
                st.state_id,
                stype,
                str(in_deg.get(st.state_id, 0)),
                str(out_deg.get(st.state_id, 0)),
                loop,
                dominant,
            ])

        title = "State Information"
        if automaton.title:
            title = f"State Information — {automaton.title}"
        return f"{title}\n{self._format_table(headers, rows)}"

    def render_state_distribution(self, state_visit_counts: Dict[str, int]) -> str:
        """Render a horizontal bar-chart of state visit frequencies.

        Each bar:  ``q0 |████████████     | 42 (35%)``

        Bars are sorted by count in descending order and auto-scaled to
        fit *max_width*.

        Args:
            state_visit_counts: Mapping from state id to visit count.

        Returns:
            Formatted bar-chart string.
        """
        if not state_visit_counts:
            return "(no visit data)"

        total = sum(state_visit_counts.values())
        max_count = max(state_visit_counts.values())
        label_width = max(len(s) for s in state_visit_counts) + 1
        count_width = len(str(max_count)) + 8  # " 42 (35%)"
        bar_budget = self.max_width - label_width - count_width - 6  # borders
        bar_budget = max(bar_budget, 10)

        lines: List[str] = ["State Visit Distribution", ""]
        for sid, cnt in sorted(
            state_visit_counts.items(), key=lambda kv: -kv[1]
        ):
            pct = (cnt / total * 100) if total else 0
            bar = self._horizontal_bar(cnt, max_count, bar_budget)
            line = f"{sid:<{label_width}} |{bar}| {cnt} ({pct:.1f}%)"
            lines.append(line)

        lines.append("")
        lines.append(f"Total visits: {total}")
        return "\n".join(lines)

    def render_transition_matrix(self, automaton: VizAutomaton) -> str:
        """Render an N×N transition-probability matrix.

        States form row and column headers.  Zero entries are shown as
        ``"."``.  Values ≥ 0.5 are highlighted with surrounding brackets.

        Args:
            automaton: The automaton to render.

        Returns:
            Formatted matrix string.
        """
        state_ids = sorted(s.state_id for s in automaton.states)
        n = len(state_ids)
        if n == 0:
            return "(no states)"

        idx = {sid: i for i, sid in enumerate(state_ids)}
        matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        for t in automaton.transitions:
            si = idx.get(t.source)
            ti = idx.get(t.target)
            if si is not None and ti is not None:
                matrix[si][ti] += t.probability

        cell_w = max(max((len(sid) for sid in state_ids), default=4), 7)
        header_pad = cell_w + 2

        lines: List[str] = []
        title = "Transition Matrix"
        if automaton.title:
            title = f"Transition Matrix — {automaton.title}"
        lines.append(title)
        lines.append("")

        # Header row
        hdr = " " * header_pad + "  ".join(sid.center(cell_w) for sid in state_ids)
        lines.append(hdr)
        sep = " " * header_pad + (self._ch["h"] * (cell_w + 2)) * n
        lines.append(sep[:self.max_width])

        for i, sid in enumerate(state_ids):
            cells: List[str] = []
            for j in range(n):
                v = matrix[i][j]
                if v == 0.0:
                    cells.append(".".center(cell_w))
                elif v >= 0.5:
                    cells.append(f"[{v:.3f}]".center(cell_w))
                else:
                    cells.append(f"{v:.4f}".center(cell_w))
            row_str = f"{sid:>{cell_w}}  " + "  ".join(cells)
            lines.append(row_str)

        return "\n".join(lines)

    def render_compact(self, automaton: VizAutomaton) -> str:
        """Render a compact path representation via BFS from initial states.

        Example output::

            q0 --a/0.80--> q1 --b/0.50--> q2

        Args:
            automaton: The automaton to render.

        Returns:
            Compact multi-line path string suitable for log output.
        """
        adj: Dict[str, List[VizTransition]] = defaultdict(list)
        for t in automaton.transitions:
            if t.source != t.target:
                adj[t.source].append(t)

        initial = [s.state_id for s in automaton.states if s.is_initial]
        if not initial:
            initial = [automaton.states[0].state_id] if automaton.states else []

        max_depth = 6
        paths: List[str] = []
        visited_paths: set[str] = set()

        for start in initial:
            queue: deque[Tuple[str, List[str], int]] = deque()
            queue.append((start, [start], 0))
            while queue:
                current, path_parts, depth = queue.popleft()
                if depth >= max_depth or current not in adj:
                    path_str = " ".join(path_parts)
                    if path_str not in visited_paths:
                        visited_paths.add(path_str)
                        paths.append(path_str)
                    continue
                expanded = False
                for t in sorted(adj[current], key=lambda tr: -tr.probability):
                    if t.target in " ".join(path_parts):
                        continue
                    arrow = f"--{t.label}/{t.probability:.2f}-->"
                    new_parts = path_parts + [arrow, t.target]
                    queue.append((t.target, new_parts, depth + 1))
                    expanded = True
                if not expanded:
                    path_str = " ".join(path_parts)
                    if path_str not in visited_paths:
                        visited_paths.add(path_str)
                        paths.append(path_str)

        title = "Compact Paths"
        if automaton.title:
            title = f"Compact Paths — {automaton.title}"
        lines = [title, ""]
        for p in paths:
            lines.append(f"  {p}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _layout_states(
        self,
        states: List[VizState],
        transitions: List[VizTransition],
    ) -> List[VizState]:
        """Auto-layout states using a simple force-directed algorithm.

        1. Place states on a circle.
        2. Apply repulsive forces between all pairs and attractive forces
           along transitions for 50 iterations.
        3. Quantise positions to integer grid coordinates.

        Already-positioned states (both *x* and *y* non-zero) are kept
        as-is.

        Args:
            states: List of states (may be mutated via copy).
            transitions: List of transitions driving attraction.

        Returns:
            New list of ``VizState`` with grid-quantised positions.
        """
        n = len(states)
        if n == 0:
            return []

        # Check if positions are already set
        already_set = all(s.x != 0.0 or s.y != 0.0 for s in states)
        copies = [
            VizState(
                state_id=s.state_id,
                label=s.label,
                is_initial=s.is_initial,
                is_accepting=s.is_accepting,
                x=s.x,
                y=s.y,
            )
            for s in states
        ]

        if already_set:
            return copies

        # Initialise on a circle
        radius = max(2.0, n * 0.8)
        for i, st in enumerate(copies):
            angle = 2.0 * math.pi * i / n
            st.x = radius * math.cos(angle)
            st.y = radius * math.sin(angle)

        id_to_idx = {st.state_id: i for i, st in enumerate(copies)}

        # Force-directed iterations
        repulsion_k = 2.0
        attraction_k = 0.1
        damping = 0.9
        dt = 0.3

        vx = [0.0] * n
        vy = [0.0] * n

        for _iteration in range(50):
            fx = [0.0] * n
            fy = [0.0] * n

            # Repulsive force between all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    dx = copies[i].x - copies[j].x
                    dy = copies[i].y - copies[j].y
                    dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                    force = repulsion_k / (dist * dist)
                    nx_ = dx / dist
                    ny_ = dy / dist
                    fx[i] += force * nx_
                    fy[i] += force * ny_
                    fx[j] -= force * nx_
                    fy[j] -= force * ny_

            # Attractive force along transitions
            for t in transitions:
                si = id_to_idx.get(t.source)
                ti = id_to_idx.get(t.target)
                if si is None or ti is None or si == ti:
                    continue
                dx = copies[ti].x - copies[si].x
                dy = copies[ti].y - copies[si].y
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                force = attraction_k * dist
                nx_ = dx / dist
                ny_ = dy / dist
                fx[si] += force * nx_
                fy[si] += force * ny_
                fx[ti] -= force * nx_
                fy[ti] -= force * ny_

            # Update positions
            for i in range(n):
                vx[i] = (vx[i] + fx[i] * dt) * damping
                vy[i] = (vy[i] + fy[i] * dt) * damping
                copies[i].x += vx[i] * dt
                copies[i].y += vy[i] * dt

        # Quantise to non-negative grid
        min_x = min(s.x for s in copies)
        min_y = min(s.y for s in copies)
        for s in copies:
            s.x = round(s.x - min_x)
            s.y = round(s.y - min_y)

        # Resolve collisions: nudge states that share a grid cell
        occupied: Dict[Tuple[float, float], List[VizState]] = defaultdict(list)
        for s in copies:
            occupied[(s.x, s.y)].append(s)
        for _key, group in occupied.items():
            for offset, st in enumerate(group):
                st.x += offset

        return copies

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _draw_box(
        self,
        label: str,
        is_initial: bool = False,
        is_accepting: bool = False,
    ) -> List[str]:
        """Return lines that render a state box.

        Accepting states use a double border (Unicode) or ``*`` corners
        (ASCII).

        Args:
            label: Text to display inside the box.
            is_initial: Whether this is an initial state.
            is_accepting: Whether this is an accepting state.

        Returns:
            List of strings, one per line of the box.
        """
        inner_w = max(len(label) + 2, 8)
        ch = self._ch

        if is_accepting:
            tl, tr = ch["double_tl"], ch["double_tr"]
            bl, br = ch["double_bl"], ch["double_br"]
            h, v = ch["double_h"], ch["double_v"]
        else:
            tl, tr = ch["tl"], ch["tr"]
            bl, br = ch["bl"], ch["br"]
            h, v = ch["h"], ch["v"]

        top_line = tl + h * inner_w + tr
        mid_line = v + label.center(inner_w) + v
        bot_line = bl + h * inner_w + br
        return [top_line, mid_line, bot_line]

    def _draw_arrow(self, direction: str, label: str) -> str:
        """Return a text arrow with an attached label.

        Args:
            direction: ``"right"`` or ``"left"``.
            label: Text placed above or beside the arrow.

        Returns:
            Arrow string.
        """
        ch = self._ch
        if direction == "right":
            shaft = ch["h"] * 2
            return f"{shaft} {label} {shaft}{ch['arrow_r']}"
        elif direction == "left":
            shaft = ch["h"] * 2
            return f"{ch['arrow_l']}{shaft} {label} {shaft}"
        elif direction == "down":
            return f"{ch['arrow_d']} {label}"
        else:
            return f"{ch['arrow_u']} {label}"

    # ------------------------------------------------------------------
    # Table / bar helpers
    # ------------------------------------------------------------------

    def _format_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[int]] = None,
    ) -> str:
        """Render a bordered table.

        If *col_widths* is ``None``, widths are auto-computed from the
        header and cell contents.

        Args:
            headers: Column header strings.
            rows: List of row data (each row is a list of cell strings).
            col_widths: Optional explicit column widths.

        Returns:
            Formatted table string including borders.
        """
        ch = self._ch
        ncols = len(headers)

        if col_widths is None:
            col_widths = [len(h) + 2 for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    if i < ncols:
                        col_widths[i] = max(col_widths[i], len(cell) + 2)

        def _sep(left: str, mid: str, right: str, fill: str) -> str:
            return left + mid.join(fill * w for w in col_widths) + right

        def _row(cells: List[str]) -> str:
            parts: List[str] = []
            for i, cell in enumerate(cells):
                w = col_widths[i] if i < len(col_widths) else 12
                parts.append(f" {cell:<{w - 2}} ")
            return ch["v"] + ch["v"].join(parts) + ch["v"]

        top = _sep(ch["tl"], ch["tj"], ch["tr"], ch["h"])
        hdr_sep = _sep(ch["lj"], ch["cross"], ch["rj"], ch["h"])
        bottom = _sep(ch["bl"], ch["bj"], ch["br"], ch["h"])

        lines = [top, _row(headers), hdr_sep]
        for r in rows:
            # Pad row to ncols
            padded = list(r) + [""] * (ncols - len(r))
            lines.append(_row(padded[:ncols]))
        lines.append(bottom)
        return "\n".join(lines)

    def _horizontal_bar(
        self,
        value: float,
        max_value: float,
        width: int = 40,
    ) -> str:
        """Return a horizontal bar string.

        Args:
            value: Current value.
            max_value: The reference maximum value (100 %).
            width: Total character width of the bar area.

        Returns:
            Bar string of exactly *width* characters.
        """
        if max_value <= 0:
            return " " * width
        ratio = min(value / max_value, 1.0)
        filled = int(ratio * width)
        bar_char = "█" if self.use_unicode else "#"
        return bar_char * filled + " " * (width - filled)


# ---------------------------------------------------------------------------
# Standalone comparison renderer
# ---------------------------------------------------------------------------

def render_comparison(automaton_a: VizAutomaton, automaton_b: VizAutomaton) -> str:
    """Render a side-by-side structural comparison of two automata.

    Reports differences in state count, transition count, alphabet size,
    initial / accepting state counts, and average out-degree.

    Args:
        automaton_a: First automaton.
        automaton_b: Second automaton.

    Returns:
        Multi-line comparison string.
    """
    def _alphabet(aut: VizAutomaton) -> set[str]:
        return {t.label for t in aut.transitions}

    def _avg_out_degree(aut: VizAutomaton) -> float:
        if not aut.states:
            return 0.0
        out: Dict[str, int] = defaultdict(int)
        for t in aut.transitions:
            out[t.source] += 1
        return sum(out.values()) / len(aut.states)

    def _initial_count(aut: VizAutomaton) -> int:
        return sum(1 for s in aut.states if s.is_initial)

    def _accepting_count(aut: VizAutomaton) -> int:
        return sum(1 for s in aut.states if s.is_accepting)

    name_a = automaton_a.title or "Automaton A"
    name_b = automaton_b.title or "Automaton B"

    metrics: List[Tuple[str, str, str]] = [
        ("States", str(len(automaton_a.states)), str(len(automaton_b.states))),
        ("Transitions", str(len(automaton_a.transitions)), str(len(automaton_b.transitions))),
        ("Alphabet size", str(len(_alphabet(automaton_a))), str(len(_alphabet(automaton_b)))),
        ("Initial states", str(_initial_count(automaton_a)), str(_initial_count(automaton_b))),
        ("Accepting states", str(_accepting_count(automaton_a)), str(_accepting_count(automaton_b))),
        ("Avg out-degree", f"{_avg_out_degree(automaton_a):.2f}", f"{_avg_out_degree(automaton_b):.2f}"),
    ]

    label_w = max(len(m[0]) for m in metrics) + 2
    val_w = 14

    lines: List[str] = []
    lines.append("Automaton Comparison")
    lines.append("=" * 60)
    header = f"{'Metric':<{label_w}} {name_a:>{val_w}} {name_b:>{val_w}}  Delta"
    lines.append(header)
    lines.append("-" * len(header))

    for label, va, vb in metrics:
        # Compute delta for numeric values
        try:
            delta_val = float(vb) - float(va)
            delta = f"{delta_val:+.2f}"
        except ValueError:
            delta = "-"
        marker = " ◄" if va != vb else ""
        lines.append(f"{label:<{label_w}} {va:>{val_w}} {vb:>{val_w}}  {delta}{marker}")

    # Alphabet diff
    alpha_a = _alphabet(automaton_a)
    alpha_b = _alphabet(automaton_b)
    only_a = alpha_a - alpha_b
    only_b = alpha_b - alpha_a
    if only_a or only_b:
        lines.append("")
        lines.append("Symbol differences:")
        if only_a:
            lines.append(f"  Only in {name_a}: {', '.join(sorted(only_a))}")
        if only_b:
            lines.append(f"  Only in {name_b}: {', '.join(sorted(only_b))}")

    # State-id overlap
    ids_a = {s.state_id for s in automaton_a.states}
    ids_b = {s.state_id for s in automaton_b.states}
    common = ids_a & ids_b
    only_ids_a = ids_a - ids_b
    only_ids_b = ids_b - ids_a
    lines.append("")
    lines.append(f"Common states:        {len(common)}")
    if only_ids_a:
        lines.append(f"Only in {name_a}:  {', '.join(sorted(only_ids_a))}")
    if only_ids_b:
        lines.append(f"Only in {name_b}:  {', '.join(sorted(only_ids_b))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def automaton_from_dict(data: dict) -> VizAutomaton:
    """Build a ``VizAutomaton`` from a plain dictionary.

    Expected schema::

        {
            "title": "...",
            "states": [
                {"id": "q0", "label": "q0", "initial": true, "accepting": false},
                ...
            ],
            "transitions": [
                {"source": "q0", "target": "q1", "label": "a", "probability": 0.8},
                ...
            ]
        }

    Missing optional fields fall back to dataclass defaults.

    Args:
        data: Dictionary representation.

    Returns:
        Populated ``VizAutomaton``.
    """
    states: List[VizState] = []
    for sd in data.get("states", []):
        states.append(VizState(
            state_id=sd.get("id", sd.get("state_id", "")),
            label=sd.get("label", sd.get("id", sd.get("state_id", ""))),
            is_initial=sd.get("initial", sd.get("is_initial", False)),
            is_accepting=sd.get("accepting", sd.get("is_accepting", False)),
            x=float(sd.get("x", 0.0)),
            y=float(sd.get("y", 0.0)),
        ))

    transitions: List[VizTransition] = []
    for td in data.get("transitions", []):
        transitions.append(VizTransition(
            source=td.get("source", ""),
            target=td.get("target", ""),
            label=td.get("label", ""),
            probability=float(td.get("probability", 1.0)),
        ))

    return VizAutomaton(
        states=states,
        transitions=transitions,
        title=data.get("title", ""),
    )


def automaton_to_dot(automaton: VizAutomaton) -> str:
    """Export an automaton to Graphviz DOT format.

    Produces a directed graph with labelled transitions.  Accepting
    states use a ``doublecircle`` shape; initial states get an invisible
    entry node with an arrow.

    Args:
        automaton: The automaton to export.

    Returns:
        DOT-language string.
    """
    lines: List[str] = []
    safe_title = (automaton.title or "automaton").replace('"', '\\"')
    lines.append(f'digraph "{safe_title}" {{')
    lines.append("    rankdir=LR;")
    lines.append("    node [shape=circle, fontname=monospace];")
    lines.append("")

    # Invisible entry nodes for initial states
    init_idx = 0
    for st in automaton.states:
        if st.is_initial:
            entry = f"__init{init_idx}__"
            lines.append(f'    {entry} [shape=point, width=0];')
            lines.append(f'    {entry} -> "{st.state_id}";')
            init_idx += 1

    # State nodes
    for st in automaton.states:
        attrs: List[str] = []
        if st.is_accepting:
            attrs.append("shape=doublecircle")
        label_esc = st.label.replace('"', '\\"')
        attrs.append(f'label="{label_esc}"')
        attr_str = ", ".join(attrs)
        lines.append(f'    "{st.state_id}" [{attr_str}];')

    lines.append("")

    # Transitions
    for t in automaton.transitions:
        edge_label = f"{t.label}"
        if t.probability < 1.0:
            edge_label += f" / {t.probability:.2f}"
        edge_label = edge_label.replace('"', '\\"')
        lines.append(f'    "{t.source}" -> "{t.target}" [label="{edge_label}"];')

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _make_sample_automaton() -> VizAutomaton:
        """Build a small sample automaton for testing."""
        return VizAutomaton(
            title="Sample DFA",
            states=[
                VizState("q0", "q0", is_initial=True),
                VizState("q1", "q1"),
                VizState("q2", "q2", is_accepting=True),
                VizState("q3", "q3"),
            ],
            transitions=[
                VizTransition("q0", "q1", "a", 0.8),
                VizTransition("q0", "q2", "b", 0.2),
                VizTransition("q1", "q2", "a", 0.5),
                VizTransition("q1", "q1", "b", 0.5),
                VizTransition("q2", "q0", "a", 0.3),
                VizTransition("q2", "q3", "b", 0.7),
                VizTransition("q3", "q3", "a", 1.0),
            ],
        )

    def _make_second_automaton() -> VizAutomaton:
        """Build a second automaton for comparison tests."""
        return VizAutomaton(
            title="Variant DFA",
            states=[
                VizState("q0", "q0", is_initial=True),
                VizState("q1", "q1", is_accepting=True),
                VizState("q4", "q4"),
            ],
            transitions=[
                VizTransition("q0", "q1", "a", 0.9),
                VizTransition("q0", "q4", "c", 0.1),
                VizTransition("q1", "q0", "b", 0.6),
                VizTransition("q4", "q4", "c", 1.0),
            ],
        )

    _counts = {"passed": 0, "failed": 0}

    def _check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            _counts["passed"] += 1
            print(f"  [PASS] {name}")
        else:
            _counts["failed"] += 1
            msg = f"  [FAIL] {name}"
            if detail:
                msg += f" — {detail}"
            print(msg)

    # ---- Test 1: State diagram rendering ----
    print("Test 1: State diagram rendering")
    viz = AutomatonVisualizer()
    aut = _make_sample_automaton()
    diagram = viz.render_state_diagram(aut)
    _check("diagram is non-empty", len(diagram) > 0)
    _check("title appears", "Sample DFA" in diagram)
    _check("state q0 appears", "q0" in diagram)
    _check("state q2 appears", "q2" in diagram)
    _check("contains box chars", "│" in diagram or "|" in diagram)
    print(f"\n--- Diagram preview (first 20 lines) ---")
    for line in diagram.split("\n")[:20]:
        print(f"  {line}")
    print()

    # ---- Test 2: Transition table rendering ----
    print("Test 2: Transition table rendering")
    table = viz.render_transition_table(aut)
    _check("table is non-empty", len(table) > 0)
    _check("header Source present", "Source" in table)
    _check("header Probability present", "Probability" in table)
    _check("high-prob marked", "*" in table)
    _check("transition q0->q1 present", "q0" in table and "q1" in table)
    print(f"\n--- Transition Table ---")
    print(table)
    print()

    # ---- Test 3: State info table ----
    print("Test 3: State info table rendering")
    info = viz.render_state_info_table(aut)
    _check("info table non-empty", len(info) > 0)
    _check("In-Deg header", "In-Deg" in info)
    _check("Out-Deg header", "Out-Deg" in info)
    _check("Dominant Behavior header", "Dominant" in info)
    print(f"\n--- State Info ---")
    print(info)
    print()

    # ---- Test 4: State distribution ----
    print("Test 4: State distribution rendering")
    visits = {"q0": 42, "q1": 30, "q2": 18, "q3": 10}
    dist = viz.render_state_distribution(visits)
    _check("distribution non-empty", len(dist) > 0)
    _check("bar char present", "█" in dist)
    _check("q0 count present", "42" in dist)
    _check("total present", "Total visits: 100" in dist)
    _check("percentage present", "%" in dist)
    print(f"\n--- Distribution ---")
    print(dist)
    print()

    # ---- Test 5: Transition matrix ----
    print("Test 5: Transition matrix rendering")
    matrix = viz.render_transition_matrix(aut)
    _check("matrix non-empty", len(matrix) > 0)
    _check("dot for zero", "." in matrix)
    _check("bracket for high-prob", "[" in matrix)
    _check("state headers", "q0" in matrix and "q3" in matrix)
    print(f"\n--- Transition Matrix ---")
    print(matrix)
    print()

    # ---- Test 6: Compact rendering ----
    print("Test 6: Compact rendering")
    compact = viz.render_compact(aut)
    _check("compact non-empty", len(compact) > 0)
    _check("arrow notation", "--" in compact)
    _check("probability in path", "0.80" in compact or "0.8" in compact)
    print(f"\n--- Compact ---")
    print(compact)
    print()

    # ---- Test 7: Comparison ----
    print("Test 7: Comparison rendering")
    aut2 = _make_second_automaton()
    comp = render_comparison(aut, aut2)
    _check("comparison non-empty", len(comp) > 0)
    _check("delta column", "Delta" in comp)
    _check("state count difference", "4" in comp and "3" in comp)
    _check("symbol differences", "Symbol differences" in comp or "Only in" in comp)
    print(f"\n--- Comparison ---")
    print(comp)
    print()

    # ---- Test 8: DOT export ----
    print("Test 8: DOT export")
    dot = automaton_to_dot(aut)
    _check("dot non-empty", len(dot) > 0)
    _check("digraph keyword", "digraph" in dot)
    _check("rankdir", "rankdir=LR" in dot)
    _check("doublecircle for accepting", "doublecircle" in dot)
    _check("edge present", '-> "q1"' in dot or "-> q1" in dot)
    _check("initial point node", "shape=point" in dot)
    print(f"\n--- DOT (first 15 lines) ---")
    for line in dot.split("\n")[:15]:
        print(f"  {line}")
    print()

    # ---- Test 9: ASCII mode ----
    print("Test 9: ASCII mode")
    viz_ascii = AutomatonVisualizer(use_unicode=False)
    diagram_ascii = viz_ascii.render_state_diagram(aut)
    _check("ascii diagram non-empty", len(diagram_ascii) > 0)
    _check("no unicode box chars", "┌" not in diagram_ascii)
    _check("ascii box chars", "+" in diagram_ascii or "-" in diagram_ascii)
    table_ascii = viz_ascii.render_transition_table(aut)
    _check("ascii table non-empty", len(table_ascii) > 0)
    dist_ascii = viz_ascii.render_state_distribution(visits)
    _check("ascii bar uses #", "#" in dist_ascii)
    print(f"\n--- ASCII diagram (first 15 lines) ---")
    for line in diagram_ascii.split("\n")[:15]:
        print(f"  {line}")
    print()

    # ---- Test 10: automaton_from_dict ----
    print("Test 10: automaton_from_dict")
    raw = {
        "title": "From dict",
        "states": [
            {"id": "s0", "label": "start", "initial": True},
            {"id": "s1", "label": "end", "accepting": True},
        ],
        "transitions": [
            {"source": "s0", "target": "s1", "label": "x", "probability": 0.75},
        ],
    }
    parsed = automaton_from_dict(raw)
    _check("title parsed", parsed.title == "From dict")
    _check("states parsed", len(parsed.states) == 2)
    _check("transitions parsed", len(parsed.transitions) == 1)
    _check("initial flag", parsed.states[0].is_initial is True)
    _check("accepting flag", parsed.states[1].is_accepting is True)
    _check("probability", parsed.transitions[0].probability == 0.75)
    print()

    # ---- Test 11: Edge cases ----
    print("Test 11: Edge cases")
    empty_aut = VizAutomaton()
    _check("empty diagram", viz.render_state_diagram(empty_aut) == "(empty automaton)")
    _check("empty matrix", "(no states)" in viz.render_transition_matrix(empty_aut))
    _check("empty distribution", "(no visit data)" in viz.render_state_distribution({}))

    single = VizAutomaton(
        states=[VizState("only", "only", is_initial=True, is_accepting=True)],
        transitions=[],
    )
    single_diag = viz.render_state_diagram(single)
    _check("single state renders", "only" in single_diag)
    print()

    # ---- Summary ----
    passed = _counts["passed"]
    failed = _counts["failed"]
    total = passed + failed
    print(f"{'=' * 50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests FAILED.")
