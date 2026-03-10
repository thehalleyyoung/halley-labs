"""
usability_oracle.mdp.visualization — MDP graph visualization.

Exports :class:`MDPVisualizer` which can render an MDP as:
- GraphViz DOT source (for ``dot``, ``neato``, etc.)
- A ``networkx.DiGraph`` with styled node/edge attributes
- Highlighted versions showing a policy or a sampled trajectory

Also provides a simple state-value heatmap in ASCII / ANSI form.

References
----------
- Graphviz: https://graphviz.org/
- NetworkX: https://networkx.org/
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from usability_oracle.mdp.models import MDP


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def _value_to_hex(value: float, vmin: float, vmax: float) -> str:
    """Map a scalar *value* in [vmin, vmax] to a red→yellow→green hex colour."""
    if vmax <= vmin:
        return "#aaaaaa"
    t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    # Red (high cost) → Green (low cost)
    r = int(255 * (1.0 - t))
    g = int(255 * t)
    b = 60
    return f"#{r:02x}{g:02x}{b:02x}"


def _ansi_color(value: float, vmin: float, vmax: float) -> str:
    """Return an ANSI escape for a value-based colour."""
    if vmax <= vmin:
        return "\033[37m"
    t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    if t < 0.33:
        return "\033[31m"  # red (high cost)
    elif t < 0.66:
        return "\033[33m"  # yellow
    else:
        return "\033[32m"  # green (low cost)


# ---------------------------------------------------------------------------
# MDPVisualizer
# ---------------------------------------------------------------------------


class MDPVisualizer:
    """Visualise MDP structure, policies, and trajectories.

    All methods are stateless; the visualizer is essentially a namespace
    for rendering functions.
    """

    # ── DOT output --------------------------------------------------------

    @staticmethod
    def to_dot(
        mdp: MDP,
        max_states: int = 200,
        show_costs: bool = True,
        values: Optional[dict[str, float]] = None,
    ) -> str:
        """Render the MDP as a GraphViz DOT string.

        Parameters
        ----------
        mdp : MDP
        max_states : int
            Limit the number of states rendered (for readability).
        show_costs : bool
            Whether to annotate edges with transition costs.
        values : dict, optional
            If provided, colour nodes by their value.

        Returns
        -------
        str
            DOT source.
        """
        lines: list[str] = ["digraph MDP {"]
        lines.append("  rankdir=LR;")
        lines.append('  node [shape=ellipse, style=filled, fillcolor="#e8e8e8"];')
        lines.append('  edge [fontsize=9];')
        lines.append("")

        state_ids = list(mdp.states.keys())[:max_states]
        state_set = set(state_ids)

        vmin = vmax = 0.0
        if values:
            vs = [values.get(s, 0.0) for s in state_ids]
            vmin, vmax = min(vs), max(vs)

        for sid in state_ids:
            state = mdp.states[sid]
            label = state.label or sid
            attrs: list[str] = [f'label="{_dot_escape(label)}"']

            if state.is_goal:
                attrs.append("shape=doublecircle")
                attrs.append('fillcolor="#90ee90"')
            elif state.is_terminal:
                attrs.append("shape=box")
                attrs.append('fillcolor="#ffcccc"')
            elif values:
                color = _value_to_hex(values.get(sid, 0.0), vmin, vmax)
                attrs.append(f'fillcolor="{color}"')

            if sid == mdp.initial_state:
                attrs.append('penwidth="3"')

            lines.append(f'  "{_dot_escape(sid)}" [{", ".join(attrs)}];')

        lines.append("")

        # Edges
        seen_edges: set[tuple[str, str, str]] = set()
        for t in mdp.transitions:
            if t.source not in state_set or t.target not in state_set:
                continue
            edge_key = (t.source, t.target, t.action)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            parts: list[str] = [t.action.split(":")[-1]]
            if show_costs and t.cost > 0:
                parts.append(f"c={t.cost:.2f}")
            if t.probability < 1.0:
                parts.append(f"p={t.probability:.2f}")
            elabel = ", ".join(parts)
            lines.append(
                f'  "{_dot_escape(t.source)}" -> "{_dot_escape(t.target)}" '
                f'[label="{_dot_escape(elabel)}"];'
            )

        lines.append("}")
        return "\n".join(lines)

    # ── NetworkX graph ----------------------------------------------------

    @staticmethod
    def to_networkx(mdp: MDP) -> Any:
        """Create a ``networkx.DiGraph`` from the MDP.

        Nodes have attributes: ``label``, ``is_goal``, ``is_terminal``,
        ``features``.
        Edges have attributes: ``action``, ``probability``, ``cost``.
        """
        return mdp.to_networkx()

    @staticmethod
    def highlight_policy(
        mdp: MDP,
        policy: dict[str, str],
        graph: Optional[Any] = None,
    ) -> Any:
        """Highlight edges corresponding to the policy in *graph*.

        Sets ``in_policy=True`` and ``color="blue"`` on policy edges,
        ``in_policy=False`` on all others.

        Parameters
        ----------
        mdp : MDP
        policy : dict[str, str]
        graph : nx.DiGraph, optional
            If None, a new graph is created from *mdp*.

        Returns
        -------
        nx.DiGraph
        """
        import networkx as nx  # type: ignore[import-untyped]

        G = graph if graph is not None else mdp.to_networkx()

        # Reset all edges
        for u, v, data in G.edges(data=True):
            data["in_policy"] = False
            data["color"] = "#cccccc"
            data["penwidth"] = 1.0

        for sid, aid in policy.items():
            outcomes = mdp.get_transitions(sid, aid)
            for target, prob, cost in outcomes:
                if G.has_edge(sid, target):
                    edata = G[sid][target]
                    edata["in_policy"] = True
                    edata["color"] = "#0000ff"
                    edata["penwidth"] = 2.5

        # Highlight policy states
        for nid in G.nodes:
            G.nodes[nid]["in_policy"] = nid in policy
            if nid in policy:
                G.nodes[nid]["color"] = "#cce5ff"

        return G

    @staticmethod
    def highlight_trajectory(
        mdp: MDP,
        trajectory: Any,
        graph: Optional[Any] = None,
    ) -> Any:
        """Highlight edges traversed by a trajectory.

        Parameters
        ----------
        mdp : MDP
        trajectory : Trajectory (or any object with ``.steps``)
        graph : nx.DiGraph, optional

        Returns
        -------
        nx.DiGraph
        """
        import networkx as nx  # type: ignore[import-untyped]

        G = graph if graph is not None else mdp.to_networkx()

        # Reset
        for u, v, data in G.edges(data=True):
            data["in_trajectory"] = False

        for nid in G.nodes:
            G.nodes[nid]["in_trajectory"] = False
            G.nodes[nid]["visit_count"] = 0

        steps = getattr(trajectory, "steps", [])
        for step in steps:
            sid = getattr(step, "state_id", None)
            nxt = getattr(step, "next_state_id", None)
            if sid and nxt and G.has_edge(sid, nxt):
                G[sid][nxt]["in_trajectory"] = True
                G[sid][nxt]["color"] = "#ff6600"
                G[sid][nxt]["penwidth"] = 3.0
            if sid and sid in G.nodes:
                G.nodes[sid]["in_trajectory"] = True
                G.nodes[sid]["visit_count"] = G.nodes[sid].get("visit_count", 0) + 1
            if nxt and nxt in G.nodes:
                G.nodes[nxt]["in_trajectory"] = True

        return G

    # ── Value heatmap (ASCII) --------------------------------------------

    @staticmethod
    def state_value_heatmap(
        values: dict[str, float],
        max_rows: int = 40,
        label_width: int = 40,
    ) -> str:
        """Render an ASCII heatmap of state values.

        Returns a multi-line string with ANSI colour codes (for terminal
        rendering) showing each state's value as a coloured bar.

        Parameters
        ----------
        values : dict[str, float]
        max_rows : int
        label_width : int

        Returns
        -------
        str
        """
        if not values:
            return "(no values to display)"

        sorted_items = sorted(values.items(), key=lambda x: x[1])[:max_rows]
        vmin = sorted_items[0][1]
        vmax = sorted_items[-1][1] if len(sorted_items) > 1 else vmin + 1.0

        bar_width = 30
        lines: list[str] = []
        reset = "\033[0m"

        for sid, val in sorted_items:
            label = sid[:label_width].ljust(label_width)
            colour = _ansi_color(val, vmin, vmax)
            frac = (val - vmin) / max(vmax - vmin, 1e-9)
            n_bars = int(frac * bar_width)
            bar = colour + "█" * n_bars + reset + "░" * (bar_width - n_bars)
            lines.append(f"  {label} │ {bar} {val:+.3f}")

        header = " " * label_width + f"   {'low':>{bar_width // 2}}{'high':>{bar_width // 2}}"
        return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot_escape(s: str) -> str:
    """Escape special characters for DOT labels."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
