"""
Pipeline graph visualization.

Exports pipeline graphs to DOT format for Graphviz rendering and
provides ASCII-art rendering for terminal output.  Supports highlighting
affected nodes, repair actions, and Fragment F membership.
"""

from __future__ import annotations

import io
import textwrap
from typing import Any, Sequence

from arc.graph.pipeline import PipelineGraph, PipelineNode, PipelineEdge
from arc.types.base import EdgeType
from arc.types.operators import SQLOperator


# =====================================================================
# DOT format export
# =====================================================================

# Operator -> colour mapping for DOT nodes
_OPERATOR_COLORS: dict[SQLOperator, str] = {
    SQLOperator.SOURCE: "#4CAF50",     # green
    SQLOperator.SINK: "#F44336",       # red
    SQLOperator.FILTER: "#2196F3",     # blue
    SQLOperator.SELECT: "#2196F3",
    SQLOperator.PROJECT: "#2196F3",
    SQLOperator.JOIN: "#FF9800",       # orange
    SQLOperator.GROUP_BY: "#9C27B0",   # purple
    SQLOperator.UNION: "#795548",      # brown
    SQLOperator.WINDOW: "#607D8B",     # grey-blue
    SQLOperator.CTE: "#009688",        # teal
    SQLOperator.TRANSFORM: "#3F51B5",  # indigo
    SQLOperator.EXTERNAL_CALL: "#E91E63",  # pink
    SQLOperator.UDF: "#E91E63",
    SQLOperator.OPAQUE: "#9E9E9E",     # grey
    SQLOperator.PANDAS_OP: "#00BCD4",  # cyan
    SQLOperator.PYSPARK_OP: "#00BCD4",
    SQLOperator.DBT_MODEL: "#8BC34A",  # light green
}

_EDGE_TYPE_STYLES: dict[EdgeType, str] = {
    EdgeType.DATA_FLOW: "solid",
    EdgeType.SCHEMA_DEPENDENCY: "dashed",
    EdgeType.QUALITY_DEPENDENCY: "dotted",
    EdgeType.CONTROL_FLOW: "bold",
    EdgeType.TEMPORAL: "tapered",
}


def to_dot(
    graph: PipelineGraph,
    highlight_nodes: Sequence[str] = (),
    highlight_color: str = "#FFD700",
    repair_nodes: Sequence[str] = (),
    repair_color: str = "#FF6B6B",
    show_schemas: bool = False,
    show_costs: bool = False,
    show_fragment_f: bool = True,
    title: str = "",
    rankdir: str = "TB",
) -> str:
    """Export the pipeline graph to Graphviz DOT format.

    Parameters
    ----------
    highlight_nodes:
        Nodes to highlight (e.g., affected by perturbation).
    repair_nodes:
        Nodes that are part of a repair plan.
    show_schemas:
        Include column names in node labels.
    show_costs:
        Include cost estimates in node labels.
    show_fragment_f:
        Mark Fragment F membership.
    """
    highlight_set = set(highlight_nodes)
    repair_set = set(repair_nodes)

    lines: list[str] = []
    lines.append("digraph pipeline {")
    lines.append(f'  rankdir={rankdir};')
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')

    graph_title = title or graph.name
    if graph_title:
        lines.append(f'  label="{graph_title}";')
        lines.append('  labelloc=t;')
        lines.append('  fontsize=14;')

    lines.append("")

    # Nodes
    for nid, node in graph.nodes.items():
        label_parts = [nid]
        label_parts.append(f"[{node.operator.value}]")

        if show_schemas and node.output_schema.columns:
            cols = ", ".join(node.output_schema.column_list[:6])
            if len(node.output_schema.columns) > 6:
                cols += f", +{len(node.output_schema.columns) - 6}"
            label_parts.append(f"({cols})")

        if show_costs:
            cost = node.cost_estimate.total_weighted_cost
            if cost > 0:
                label_parts.append(f"cost={cost:.2f}")

        if show_fragment_f:
            frag = "F" if node.in_fragment_f else "~F"
            label_parts.append(f"[{frag}]")

        label = "\\n".join(label_parts)

        # Determine fill colour
        if nid in repair_set:
            fill = repair_color
        elif nid in highlight_set:
            fill = highlight_color
        else:
            fill = _OPERATOR_COLORS.get(node.operator, "#FFFFFF")

        # Font colour for contrast
        font_color = "white" if nid in repair_set else "black"

        attrs = f'label="{label}", fillcolor="{fill}", fontcolor="{font_color}"'
        lines.append(f'  "{nid}" [{attrs}];')

    lines.append("")

    # Edges
    for (src, tgt), edge in graph.edges.items():
        style = _EDGE_TYPE_STYLES.get(edge.edge_type, "solid")
        edge_attrs = f'style={style}'
        if edge.label:
            edge_attrs += f', label="{edge.label}"'
        if edge.edge_type != EdgeType.DATA_FLOW:
            edge_attrs += f', color="gray"'
        # Highlight edges on the affected path
        if src in highlight_set and tgt in highlight_set:
            edge_attrs += ', color="red", penwidth=2'
        lines.append(f'  "{src}" -> "{tgt}" [{edge_attrs}];')

    lines.append("}")
    return "\n".join(lines)


def save_dot(
    graph: PipelineGraph,
    path: str,
    **kwargs: Any,
) -> None:
    """Write DOT file to disk."""
    dot = to_dot(graph, **kwargs)
    with open(path, "w") as f:
        f.write(dot)


# =====================================================================
# ASCII art rendering
# =====================================================================

_OP_SYMBOLS: dict[SQLOperator, str] = {
    SQLOperator.SOURCE: "◉",
    SQLOperator.SINK: "◎",
    SQLOperator.FILTER: "▽",
    SQLOperator.SELECT: "□",
    SQLOperator.PROJECT: "□",
    SQLOperator.JOIN: "⋈",
    SQLOperator.GROUP_BY: "Σ",
    SQLOperator.UNION: "∪",
    SQLOperator.WINDOW: "⊞",
    SQLOperator.CTE: "↻",
    SQLOperator.TRANSFORM: "⊳",
    SQLOperator.EXTERNAL_CALL: "⚡",
    SQLOperator.UDF: "λ",
    SQLOperator.OPAQUE: "?",
    SQLOperator.PANDAS_OP: "🐼",
    SQLOperator.PYSPARK_OP: "⚡",
    SQLOperator.DBT_MODEL: "dbt",
}


def to_ascii(
    graph: PipelineGraph,
    highlight_nodes: Sequence[str] = (),
    repair_nodes: Sequence[str] = (),
    max_width: int = 80,
    show_fragment_f: bool = True,
) -> str:
    """Render the pipeline graph as ASCII art for terminal output.

    Uses a top-to-bottom layer-based layout for DAGs.
    """
    if graph.node_count == 0:
        return "(empty pipeline)"

    highlight_set = set(highlight_nodes)
    repair_set = set(repair_nodes)

    # Compute layers via topological ordering
    if graph.is_dag():
        layers = _compute_layers(graph)
    else:
        layers = [graph.node_ids]

    buf = io.StringIO()

    # Title
    title = graph.name or "Pipeline"
    buf.write(f"{'=' * max_width}\n")
    buf.write(f" {title}\n")
    buf.write(f"{'=' * max_width}\n\n")

    for layer_idx, layer in enumerate(layers):
        # Render nodes in this layer
        node_strs: list[str] = []
        for nid in layer:
            node = graph.get_node(nid)
            sym = _OP_SYMBOLS.get(node.operator, "○")

            frag = ""
            if show_fragment_f:
                frag = " [F]" if node.in_fragment_f else " [~F]"

            marker = ""
            if nid in repair_set:
                marker = " *REPAIR*"
            elif nid in highlight_set:
                marker = " *AFFECTED*"

            node_str = f"[{sym} {nid}{frag}{marker}]"
            node_strs.append(node_str)

        # Join node strings for this layer
        line = "  ".join(node_strs)
        # Center the layer
        padding = max(0, (max_width - len(line)) // 2)
        buf.write(" " * padding + line + "\n")

        # Draw edges to the next layer
        if layer_idx < len(layers) - 1:
            next_layer = layers[layer_idx + 1]
            edge_lines = _draw_edges_ascii(graph, layer, next_layer, max_width)
            for el in edge_lines:
                buf.write(el + "\n")

    buf.write(f"\n{'=' * max_width}\n")

    # Legend
    stats_parts: list[str] = [
        f"Nodes: {graph.node_count}",
        f"Edges: {graph.edge_count}",
        f"Depth: {graph.depth() if graph.is_dag() else '?'}",
    ]
    if highlight_set:
        stats_parts.append(f"Affected: {len(highlight_set)}")
    if repair_set:
        stats_parts.append(f"Repair: {len(repair_set)}")
    buf.write(" | ".join(stats_parts) + "\n")

    return buf.getvalue()


def _compute_layers(graph: PipelineGraph) -> list[list[str]]:
    """Compute layer assignment for a DAG."""
    topo = graph.topological_sort()
    level: dict[str, int] = {}
    for nid in topo:
        preds = graph.predecessors(nid)
        if not preds:
            level[nid] = 0
        else:
            level[nid] = max(level[p] for p in preds) + 1

    max_level = max(level.values()) if level else 0
    layers: list[list[str]] = [[] for _ in range(max_level + 1)]
    for nid in topo:
        layers[level[nid]].append(nid)
    return layers


def _draw_edges_ascii(
    graph: PipelineGraph,
    source_layer: list[str],
    target_layer: list[str],
    max_width: int,
) -> list[str]:
    """Draw ASCII edges between two adjacent layers."""
    lines: list[str] = []

    # Simple: draw vertical pipes for each connection
    connections: list[tuple[int, int]] = []
    for i, src in enumerate(source_layer):
        for j, tgt in enumerate(target_layer):
            if graph.has_edge(src, tgt):
                connections.append((i, j))

    if not connections:
        return [""]

    # Simple arrow drawing
    n_src = len(source_layer)
    n_tgt = len(target_layer)

    # Draw connecting lines
    edge_strs: list[str] = []
    for si, ti in connections:
        edge_strs.append(f"  {source_layer[si]} -> {target_layer[ti]}")

    # Use simple vertical pipes
    line = " " * ((max_width - 1) // 2) + "|"
    lines.append(line)
    line = " " * ((max_width - 1) // 2) + "v"
    lines.append(line)

    return lines


# =====================================================================
# Summary table
# =====================================================================

def node_summary_table(
    graph: PipelineGraph,
    highlight_nodes: Sequence[str] = (),
) -> str:
    """Render a tabular summary of all nodes."""
    highlight_set = set(highlight_nodes)

    lines: list[str] = []
    header = f"{'Node ID':<25} {'Operator':<15} {'Cols':<6} {'F?':<4} {'Cost':<10} {'Status'}"
    lines.append(header)
    lines.append("-" * len(header))

    order = graph.topological_sort() if graph.is_dag() else sorted(graph.node_ids)
    for nid in order:
        node = graph.get_node(nid)
        cols = len(node.output_schema.columns)
        frag = "F" if node.in_fragment_f else "~F"
        cost = f"{node.cost_estimate.total_weighted_cost:.2f}"
        status = ""
        if nid in highlight_set:
            status = "AFFECTED"

        lines.append(f"{nid:<25} {node.operator.value:<15} {cols:<6} {frag:<4} {cost:<10} {status}")

    return "\n".join(lines)


# =====================================================================
# Mermaid export (for documentation)
# =====================================================================

def to_mermaid(
    graph: PipelineGraph,
    direction: str = "TD",
) -> str:
    """Export the pipeline graph to Mermaid diagram format."""
    lines: list[str] = [f"graph {direction}"]

    for nid, node in graph.nodes.items():
        label = f"{nid}[{node.operator.value}]"
        if node.operator == SQLOperator.SOURCE:
            lines.append(f"  {nid}(({nid}))")
        elif node.operator == SQLOperator.SINK:
            lines.append(f"  {nid}[/{nid}/]")
        else:
            lines.append(f"  {nid}[{nid}]")

    for (src, tgt), edge in graph.edges.items():
        if edge.label:
            lines.append(f"  {src} -->|{edge.label}| {tgt}")
        else:
            lines.append(f"  {src} --> {tgt}")

    return "\n".join(lines)


# =====================================================================
# Detailed node info panel
# =====================================================================

def node_detail(graph: PipelineGraph, node_id: str) -> str:
    """Render a detailed text description of a single node."""
    node = graph.get_node(node_id)
    lines: list[str] = []
    lines.append(f"╔{'═' * 58}╗")
    lines.append(f"║ Node: {node_id:<51}║")
    lines.append(f"╠{'═' * 58}╣")
    lines.append(f"║ Operator: {node.operator.value:<47}║")

    frag = "Yes" if node.in_fragment_f else "No"
    lines.append(f"║ Fragment F: {frag:<46}║")

    if node.query_text:
        q = node.query_text[:45]
        lines.append(f"║ Query: {q:<51}║")

    # Input schema
    if node.input_schema.columns:
        lines.append(f"║ Input Schema ({len(node.input_schema.columns)} cols):{' ' * 34}║")
        for col in node.input_schema.columns[:8]:
            c_str = f"  {col.name}: {col.sql_type}"
            lines.append(f"║   {c_str:<55}║")
        if len(node.input_schema.columns) > 8:
            lines.append(f"║   ... +{len(node.input_schema.columns) - 8} more{' ' * 42}║")

    # Output schema
    if node.output_schema.columns:
        lines.append(f"║ Output Schema ({len(node.output_schema.columns)} cols):{' ' * 33}║")
        for col in node.output_schema.columns[:8]:
            c_str = f"  {col.name}: {col.sql_type}"
            lines.append(f"║   {c_str:<55}║")
        if len(node.output_schema.columns) > 8:
            lines.append(f"║   ... +{len(node.output_schema.columns) - 8} more{' ' * 42}║")

    # Quality constraints
    if node.quality_constraints:
        lines.append(f"║ Quality Constraints ({len(node.quality_constraints)}):{' ' * 30}║")
        for qc in node.quality_constraints[:5]:
            q_str = f"  [{qc.severity.value}] {qc.constraint_id}: {qc.predicate}"
            lines.append(f"║   {q_str[:55]:<55}║")

    # Cost
    cost = node.cost_estimate
    cost_str = f"compute={cost.compute_seconds:.1f}s, rows={cost.row_estimate}"
    lines.append(f"║ Cost: {cost_str:<52}║")

    # Availability
    avail = node.availability_contract
    avail_str = f"SLA={avail.sla_percentage:.1f}%"
    lines.append(f"║ Availability: {avail_str:<43}║")

    # Connections
    preds = graph.predecessors(node_id)
    succs = graph.successors(node_id)
    lines.append(f"║ Predecessors: {', '.join(preds) if preds else '(none)':<43}║")
    lines.append(f"║ Successors: {', '.join(succs) if succs else '(none)':<45}║")

    # Metadata
    meta = node.metadata
    if meta.owner:
        lines.append(f"║ Owner: {meta.owner:<51}║")
    if meta.tags:
        lines.append(f"║ Tags: {', '.join(meta.tags):<52}║")
    if meta.dialect:
        lines.append(f"║ Dialect: {meta.dialect:<49}║")

    lines.append(f"╚{'═' * 58}╝")
    return "\n".join(lines)


# =====================================================================
# Edge detail
# =====================================================================

def edge_summary_table(graph: PipelineGraph) -> str:
    """Render a tabular summary of all edges."""
    lines: list[str] = []
    header = f"{'Source':<20} {'Target':<20} {'Type':<18} {'Mapping'}"
    lines.append(header)
    lines.append("-" * len(header))

    for (src, tgt), edge in graph.edges.items():
        mapping = ""
        if edge.column_mapping:
            mapping = ", ".join(f"{k}->{v}" for k, v in list(edge.column_mapping.items())[:3])
            if len(edge.column_mapping) > 3:
                mapping += f", +{len(edge.column_mapping) - 3}"
        lines.append(f"{src:<20} {tgt:<20} {edge.edge_type.value:<18} {mapping}")

    return "\n".join(lines)


# =====================================================================
# Impact visualization
# =====================================================================

def render_impact(
    graph: PipelineGraph,
    origin_node: str,
    affected_nodes: list[str],
    format: str = "ascii",
) -> str:
    """Render an impact analysis result as a visual diagram."""
    if format == "dot":
        return to_dot(
            graph,
            highlight_nodes=affected_nodes,
            highlight_color="#FFCDD2",
            repair_nodes=[origin_node],
            repair_color="#D32F2F",
            title=f"Impact from {origin_node}",
        )
    return to_ascii(
        graph,
        highlight_nodes=affected_nodes,
        repair_nodes=[origin_node],
    )


# =====================================================================
# Cost heat map (text-based)
# =====================================================================

def cost_heatmap(graph: PipelineGraph, width: int = 60) -> str:
    """Render a text-based cost heat map of pipeline nodes."""
    if graph.node_count == 0:
        return "(empty)"

    costs = graph.cost_distribution()
    max_cost = max(costs.values()) if costs else 1.0
    max_cost = max(max_cost, 0.001)  # avoid division by zero

    lines: list[str] = []
    lines.append("Cost Heat Map")
    lines.append("=" * width)

    bar_width = width - 35
    order = graph.topological_sort() if graph.is_dag() else sorted(graph.node_ids)

    for nid in order:
        cost = costs.get(nid, 0.0)
        ratio = cost / max_cost
        filled = int(ratio * bar_width)
        empty = bar_width - filled

        # Use different chars for intensity
        if ratio > 0.75:
            char = "█"
        elif ratio > 0.5:
            char = "▓"
        elif ratio > 0.25:
            char = "▒"
        else:
            char = "░"

        bar = char * filled + "·" * empty
        lines.append(f"  {nid:<20} [{bar}] {cost:.2f}")

    lines.append("=" * width)
    return "\n".join(lines)


# =====================================================================
# Execution wave visualization
# =====================================================================

def render_waves(
    graph: "PipelineGraph",
    waves: list | None = None,
    width: int = 80,
) -> str:
    """Render execution waves as a text timeline.

    Each wave is a set of nodes that can execute in parallel.
    """
    from arc.graph.analysis import compute_execution_waves

    if waves is None:
        waves = compute_execution_waves(graph)

    lines: list[str] = []
    lines.append("Execution Waves")
    lines.append("=" * width)

    for i, wave in enumerate(waves):
        node_ids = sorted(wave.node_ids)
        parallel = len(node_ids)
        est_time = wave.estimated_time if hasattr(wave, 'estimated_time') else 0.0

        lines.append(f"\n  Wave {i} (parallelism={parallel}, est={est_time:.2f}s)")
        lines.append("  " + "-" * (width - 4))

        for nid in node_ids:
            node = graph.get_node(nid)
            op = node.operator.value if node.operator else "—"
            cost_str = ""
            if node.cost_estimate:
                cost_str = f" [{node.cost_estimate.compute_seconds:.1f}s]"
            lines.append(f"    ├── {nid} ({op}){cost_str}")

    lines.append("\n" + "=" * width)
    total_waves = len(waves)
    total_nodes = sum(len(w.node_ids) for w in waves)
    max_par = max((len(w.node_ids) for w in waves), default=0)
    lines.append(f"  Waves: {total_waves}, Nodes: {total_nodes}, Max parallelism: {max_par}")
    return "\n".join(lines)


# =====================================================================
# Schema map visualization
# =====================================================================

def schema_map(graph: "PipelineGraph", width: int = 80) -> str:
    """Render a text map showing schema transformations across the pipeline.

    Each node shows its output schema columns, highlighting additions
    and removals compared to its immediate upstream.
    """
    lines: list[str] = []
    lines.append("Schema Map")
    lines.append("=" * width)

    order = graph.topological_sort() if graph.is_dag() else sorted(graph.node_ids)

    for nid in order:
        node = graph.get_node(nid)
        op = node.operator.value if node.operator else "—"
        lines.append(f"\n  [{nid}] ({op})")

        if not node.output_schema:
            lines.append("    (no schema)")
            continue

        out_cols = {c.name for c in node.output_schema.columns}

        # Gather upstream columns
        upstream_cols: set[str] = set()
        predecessors = list(graph.predecessors(nid))
        for pred_id in predecessors:
            pred = graph.get_node(pred_id)
            if pred.output_schema:
                upstream_cols |= {c.name for c in pred.output_schema.columns}

        added = out_cols - upstream_cols if upstream_cols else set()
        removed = upstream_cols - out_cols if upstream_cols else set()
        kept = out_cols & upstream_cols if upstream_cols else out_cols

        for col in node.output_schema.columns:
            marker = " "
            style = ""
            if col.name in added:
                marker = "+"
                style = " (new)"
            elif col.name in removed:
                marker = "-"
                style = " (dropped)"

            type_str = str(col.sql_type)
            nullable = "?" if col.nullable else "!"
            lines.append(f"    {marker} {col.name}: {type_str}{nullable}{style}")

        if removed:
            for col_name in sorted(removed):
                lines.append(f"    - {col_name} (dropped from upstream)")

    lines.append("\n" + "=" * width)
    return "\n".join(lines)


# =====================================================================
# Fragment F region visualization
# =====================================================================

def render_fragment_f(graph: "PipelineGraph", width: int = 80) -> str:
    """Render a visualization highlighting which nodes belong to Fragment F.

    Nodes in Fragment F are shown in green; nodes outside are in red.
    """
    from arc.graph.analysis import FragmentClassifier

    lines: list[str] = []
    lines.append("Fragment F Classification")
    lines.append("=" * width)

    classifier = FragmentClassifier()
    result = classifier.classify(graph)
    order = graph.topological_sort() if graph.is_dag() else sorted(graph.node_ids)

    in_f: list[str] = list(result.fragment_f_nodes)
    not_f: list[str] = list(result.non_fragment_f_nodes)

    for nid in order:
        node = graph.get_node(nid)
        op = node.operator.value if node.operator else "—"

        if nid in result.fragment_f_nodes:
            marker = "[F]"
        else:
            marker = "[X]"

        violation = result.violations.get(nid, "")
        lines.append(f"  {marker} {nid:<25} ({op})")
        if violation:
            lines.append(f"      reason: {violation}")

    lines.append("\n" + "-" * width)
    total = len(order)
    ratio = len(in_f) / total if total else 0
    lines.append(f"  In Fragment F: {len(in_f)}/{total} ({ratio:.0%})")
    lines.append(f"  Outside:       {len(not_f)}/{total}")
    lines.append("=" * width)
    return "\n".join(lines)
