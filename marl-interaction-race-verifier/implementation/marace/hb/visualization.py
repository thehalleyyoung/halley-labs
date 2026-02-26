"""
HB graph visualization for the MARACE system.

Generates space-time diagrams, interaction group diagrams, causal-chain
highlights, and exports to DOT, JSON, and ASCII art.
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx  # type: ignore

from marace.hb.interaction_groups import InteractionGroup


# ======================================================================
# Color palette
# ======================================================================

_DEFAULT_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def _color_for_index(idx: int) -> str:
    return _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)]


# ======================================================================
# HBVisualizer
# ======================================================================

class HBVisualizer:
    """Generate visualizations of HB graphs.

    Delegates to specialized diagram classes or exports to standard
    formats (DOT, JSON, ASCII).

    Args:
        hb_graph: The :class:`~marace.hb.hb_graph.HBGraph` to visualize.
    """

    def __init__(self, hb_graph: Any) -> None:
        self._hb = hb_graph

    def to_dot(self, *, rankdir: str = "TB") -> str:
        """Export the HB graph to Graphviz DOT format.

        Args:
            rankdir: Graph direction (``TB``, ``LR``, etc.).

        Returns:
            DOT-language string.
        """
        lines = [f'digraph "{self._hb.name}" {{', f"  rankdir={rankdir};"]
        g = self._hb.graph

        for node, data in g.nodes(data=True):
            label_parts = [str(node)]
            if "agent_id" in data:
                label_parts.append(f"agent={data['agent_id']}")
            if "timestep" in data:
                label_parts.append(f"t={data['timestep']}")
            label = "\\n".join(label_parts)
            agent = data.get("agent_id", "")
            color = _color_for_index(hash(agent) % len(_DEFAULT_PALETTE))
            lines.append(
                f'  "{node}" [label="{label}", style=filled, '
                f'fillcolor="{color}"];'
            )

        for u, v, data in g.edges(data=True):
            style = "solid"
            source_type = data.get("source", "explicit")
            if source_type == "physics_mediated":
                style = "dashed"
            elif source_type == "environment_mediated":
                style = "dotted"
            lines.append(f'  "{u}" -> "{v}" [style={style}];')

        lines.append("}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export to JSON for web visualization.

        Returns a JSON string with ``nodes`` and ``edges`` arrays
        suitable for D3.js or similar libraries.
        """
        g = self._hb.graph
        nodes = []
        for node, data in g.nodes(data=True):
            entry: Dict[str, Any] = {"id": node}
            entry.update(data)
            nodes.append(entry)

        edges = []
        for u, v, data in g.edges(data=True):
            entry = {"source": u, "target": v}
            entry.update(data)
            edges.append(entry)

        return json.dumps({"nodes": nodes, "edges": edges}, indent=2, default=str)

    def to_ascii(self, max_width: int = 80) -> str:
        """Render a simple ASCII art representation.

        Shows agent timelines vertically with HB edges between them.

        Args:
            max_width: Maximum character width.

        Returns:
            Multi-line ASCII string.
        """
        diagram = SpaceTimeDiagram(self._hb)
        return diagram.render_ascii(max_width=max_width)


# ======================================================================
# SpaceTimeDiagram
# ======================================================================

class SpaceTimeDiagram:
    """Agent timelines with HB edges between them.

    Arranges agents as vertical lanes and events as rows ordered by
    timestep, with directed arrows showing HB edges that cross lanes.

    Args:
        hb_graph: The HBGraph to diagram.
    """

    def __init__(self, hb_graph: Any) -> None:
        self._hb = hb_graph

    def render_ascii(self, max_width: int = 80) -> str:
        """Render the space-time diagram as ASCII art.

        Args:
            max_width: Maximum line width.

        Returns:
            Multi-line string.
        """
        g = self._hb.graph
        if g.number_of_nodes() == 0:
            return "(empty HB graph)"

        # Collect agents and sort
        agents: Set[str] = set()
        for _, data in g.nodes(data=True):
            if "agent_id" in data:
                agents.add(data["agent_id"])
        if not agents:
            agents = {"?"}
        agent_list = sorted(agents)

        # Assign column positions
        n_agents = len(agent_list)
        col_width = max(6, min(max_width // max(n_agents, 1), 20))
        agent_col = {a: i * col_width for i, a in enumerate(agent_list)}

        # Group events by timestep
        events_by_step: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
        for node, data in g.nodes(data=True):
            t = data.get("timestep", 0)
            a = data.get("agent_id", "?")
            events_by_step[t].append((a, node))

        timesteps = sorted(events_by_step.keys())
        lines: List[str] = []

        # Header
        header_parts = []
        for a in agent_list:
            header_parts.append(a.center(col_width))
        lines.append("".join(header_parts))
        lines.append("-" * (n_agents * col_width))

        # Collect cross-agent edges for annotation
        cross_edges: Dict[int, List[Tuple[str, str, str, str]]] = defaultdict(list)
        for u, v in g.edges():
            u_data = g.nodes[u]
            v_data = g.nodes[v]
            u_agent = u_data.get("agent_id", "?")
            v_agent = v_data.get("agent_id", "?")
            if u_agent != v_agent:
                t = u_data.get("timestep", 0)
                cross_edges[t].append((u_agent, v_agent, u, v))

        # Rows
        for t in timesteps:
            row = [" "] * (n_agents * col_width)
            for agent, eid in events_by_step[t]:
                if agent in agent_col:
                    col = agent_col[agent]
                    marker = f"[{eid[:col_width - 4]}]" if len(eid) > col_width - 4 else f"[{eid}]"
                    for ci, ch in enumerate(marker):
                        pos = col + ci
                        if pos < len(row):
                            row[pos] = ch

            line = "".join(row).rstrip()
            lines.append(f"t={t:<3} {line}")

            # Show cross-agent edges
            for src_a, dst_a, src_e, dst_e in cross_edges.get(t, []):
                if src_a in agent_col and dst_a in agent_col:
                    arrow_line = [" "] * (n_agents * col_width)
                    c1 = agent_col[src_a] + col_width // 2
                    c2 = agent_col[dst_a] + col_width // 2
                    lo, hi = min(c1, c2), max(c1, c2)
                    for ci in range(lo, min(hi + 1, len(arrow_line))):
                        arrow_line[ci] = "-"
                    if c1 < c2:
                        if hi < len(arrow_line):
                            arrow_line[hi] = ">"
                    else:
                        arrow_line[lo] = "<"
                    lines.append("     " + "".join(arrow_line).rstrip())

        return "\n".join(lines)

    def to_dot(self, *, rankdir: str = "LR") -> str:
        """Render as DOT with subgraph clusters per agent."""
        g = self._hb.graph
        agents: Dict[str, List[str]] = defaultdict(list)
        for node, data in g.nodes(data=True):
            a = data.get("agent_id", "unknown")
            agents[a].append(node)

        lines = [f'digraph spacetime {{', f"  rankdir={rankdir};"]
        for idx, (agent, nodes) in enumerate(sorted(agents.items())):
            color = _color_for_index(idx)
            lines.append(f'  subgraph cluster_{agent} {{')
            lines.append(f'    label="{agent}";')
            lines.append(f'    color="{color}";')
            for n in sorted(nodes):
                t = g.nodes[n].get("timestep", "?")
                lines.append(f'    "{n}" [label="{n}\\nt={t}"];')
            lines.append("  }")

        for u, v in g.edges():
            lines.append(f'  "{u}" -> "{v}";')
        lines.append("}")
        return "\n".join(lines)


# ======================================================================
# InteractionGroupDiagram
# ======================================================================

class InteractionGroupDiagram:
    """Colored visualization of interaction groups.

    Each group is assigned a color; agents in the same group share the
    same color.  Useful for quickly seeing the interaction structure.

    Args:
        hb_graph: The HBGraph to visualize.
        groups: The interaction groups to color.
    """

    def __init__(
        self,
        hb_graph: Any,
        groups: List[InteractionGroup],
    ) -> None:
        self._hb = hb_graph
        self._groups = groups

    def to_dot(self) -> str:
        """Render as DOT with group coloring."""
        agent_to_group: Dict[str, int] = {}
        for idx, g in enumerate(self._groups):
            for a in g.agent_ids:
                agent_to_group[a] = idx

        g = self._hb.graph
        lines = ['digraph interaction_groups {', "  rankdir=TB;"]

        for idx, grp in enumerate(self._groups):
            color = _color_for_index(idx)
            lines.append(f"  subgraph cluster_group_{idx} {{")
            lines.append(f'    label="Group {idx} (strength={grp.interaction_strength:.2f})";')
            lines.append(f'    style=filled; color="{color}80";')
            for node, data in g.nodes(data=True):
                if data.get("agent_id") in grp.agent_ids:
                    lines.append(
                        f'    "{node}" [label="{node}", '
                        f'fillcolor="{color}", style=filled];'
                    )
            lines.append("  }")

        for u, v in g.edges():
            lines.append(f'  "{u}" -> "{v}";')
        lines.append("}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export group diagram data as JSON."""
        return json.dumps({
            "groups": [g.to_dict() for g in self._groups],
            "graph": json.loads(HBVisualizer(self._hb).to_json()),
        }, indent=2, default=str)

    def summary_ascii(self) -> str:
        """ASCII summary table of groups."""
        lines = ["Interaction Groups", "=" * 50]
        for idx, grp in enumerate(self._groups):
            agents = ", ".join(sorted(grp.agent_ids))
            lines.append(
                f"  Group {idx}: [{agents}] "
                f"strength={grp.interaction_strength:.3f} "
                f"events={grp.num_events}"
            )
        return "\n".join(lines)


# ======================================================================
# CausalChainHighlighter
# ======================================================================

class CausalChainHighlighter:
    """Highlight specific causal chains in the HB graph.

    Given a set of event IDs forming a causal chain, produces a
    visualization where the chain is visually prominent (bold/colored)
    and the rest of the graph is dimmed.

    Args:
        hb_graph: The HBGraph to visualize.
    """

    def __init__(self, hb_graph: Any) -> None:
        self._hb = hb_graph

    def highlight_chain(
        self, chain: List[str], *, color: str = "#e15759",
    ) -> str:
        """Render the graph as DOT with *chain* highlighted.

        Args:
            chain: Ordered list of event IDs forming the causal chain.
            color: Color for highlighted nodes/edges.

        Returns:
            DOT string.
        """
        chain_set = set(chain)
        chain_edges = set()
        for i in range(len(chain) - 1):
            chain_edges.add((chain[i], chain[i + 1]))

        g = self._hb.graph
        lines = ['digraph causal_chain {', "  rankdir=TB;"]

        for node, data in g.nodes(data=True):
            if node in chain_set:
                lines.append(
                    f'  "{node}" [label="{node}", style=filled, '
                    f'fillcolor="{color}", penwidth=3];'
                )
            else:
                lines.append(
                    f'  "{node}" [label="{node}", style=filled, '
                    f'fillcolor="#cccccc", fontcolor="#888888"];'
                )

        for u, v in g.edges():
            if (u, v) in chain_edges:
                lines.append(
                    f'  "{u}" -> "{v}" [color="{color}", penwidth=3];'
                )
            else:
                lines.append(f'  "{u}" -> "{v}" [color="#cccccc"];')

        lines.append("}")
        return "\n".join(lines)

    def highlight_between(
        self, source: str, target: str, *, color: str = "#e15759",
    ) -> str:
        """Highlight all shortest paths between *source* and *target*.

        Args:
            source: Start event.
            target: End event.
            color: Highlight color.

        Returns:
            DOT string with paths highlighted.
        """
        g = self._hb.graph
        try:
            paths = list(nx.all_shortest_paths(g, source, target))
        except nx.NetworkXNoPath:
            paths = []

        highlight_nodes: Set[str] = set()
        highlight_edges: Set[Tuple[str, str]] = set()
        for path in paths:
            highlight_nodes.update(path)
            for i in range(len(path) - 1):
                highlight_edges.add((path[i], path[i + 1]))

        lines = ['digraph causal_paths {', "  rankdir=TB;"]
        for node, data in g.nodes(data=True):
            if node in highlight_nodes:
                lines.append(
                    f'  "{node}" [style=filled, fillcolor="{color}", penwidth=3];'
                )
            else:
                lines.append(f'  "{node}" [style=filled, fillcolor="#cccccc"];')
        for u, v in g.edges():
            if (u, v) in highlight_edges:
                lines.append(f'  "{u}" -> "{v}" [color="{color}", penwidth=3];')
            else:
                lines.append(f'  "{u}" -> "{v}" [color="#cccccc"];')
        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self, chain: List[str]) -> str:
        """Simple ASCII representation of a causal chain."""
        if not chain:
            return "(empty chain)"
        return " -> ".join(chain)
