"""
taintflow.dag.visualization -- DAG visualization in Graphviz, Mermaid, ASCII, and HTML.

Provides multiple output formats for rendering PI-DAGs with colour coding
by node type, severity, and leakage level.  Edge width encodes capacity
or provenance fraction.  Nodes are clustered into pipeline stages.
"""

from __future__ import annotations

import html as html_mod
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from taintflow.core.types import (
    EdgeKind,
    NodeKind,
    OpType,
    Origin,
    Severity,
)
from taintflow.dag.node import (
    PipelineNode,
    DataSourceNode,
    PartitionNode,
    TransformNode,
    FitNode,
    PredictNode,
    PandasOpNode,
    AggregationNode,
    FeatureEngineeringNode,
    SelectionNode,
    CustomNode,
    SinkNode,
)
from taintflow.dag.edge import PipelineEdge
from taintflow.dag.pidag import PIDAG, PipelineStage


# ===================================================================
#  Colour palettes
# ===================================================================

_NODE_COLORS: dict[NodeKind, str] = {
    NodeKind.DATA_SOURCE: "#4CAF50",
    NodeKind.SPLIT: "#FF9800",
    NodeKind.TRANSFORM: "#2196F3",
    NodeKind.ESTIMATOR_FIT: "#9C27B0",
    NodeKind.ESTIMATOR_PREDICT: "#673AB7",
    NodeKind.ESTIMATOR_TRANSFORM: "#7B1FA2",
    NodeKind.MERGE: "#FF5722",
    NodeKind.FEATURE_ENGINEERING: "#00BCD4",
    NodeKind.EVALUATION: "#FFC107",
    NodeKind.SINK: "#607D8B",
    NodeKind.UNKNOWN: "#9E9E9E",
}

_SEVERITY_COLORS: dict[Severity, str] = {
    Severity.NEGLIGIBLE: "#81C784",
    Severity.WARNING: "#FFB74D",
    Severity.CRITICAL: "#E57373",
}

_EDGE_KIND_STYLES: dict[EdgeKind, Tuple[str, str]] = {
    EdgeKind.DATA_FLOW: ("solid", "#333333"),
    EdgeKind.FIT_DEPENDENCY: ("dashed", "#9C27B0"),
    EdgeKind.PARAMETER_FLOW: ("dotted", "#2196F3"),
    EdgeKind.CONTROL_FLOW: ("dotted", "#FF9800"),
    EdgeKind.LABEL_FLOW: ("dashed", "#F44336"),
    EdgeKind.INDEX_FLOW: ("dotted", "#607D8B"),
    EdgeKind.AUXILIARY: ("dotted", "#9E9E9E"),
}

_NODE_SHAPES: dict[NodeKind, str] = {
    NodeKind.DATA_SOURCE: "cylinder",
    NodeKind.SPLIT: "diamond",
    NodeKind.TRANSFORM: "box",
    NodeKind.ESTIMATOR_FIT: "hexagon",
    NodeKind.ESTIMATOR_PREDICT: "hexagon",
    NodeKind.ESTIMATOR_TRANSFORM: "hexagon",
    NodeKind.MERGE: "invtriangle",
    NodeKind.FEATURE_ENGINEERING: "parallelogram",
    NodeKind.EVALUATION: "octagon",
    NodeKind.SINK: "house",
    NodeKind.UNKNOWN: "ellipse",
}


# ===================================================================
#  Graphviz DOT output
# ===================================================================


def to_dot(
    dag: PIDAG,
    *,
    highlight_leakage: bool = True,
    show_columns: bool = False,
    show_capacity: bool = True,
    cluster_by_stage: bool = True,
    max_label_length: int = 40,
    title: str = "",
) -> str:
    """Render the DAG as a Graphviz DOT string.

    Parameters
    ----------
    dag : PIDAG
        The DAG to render.
    highlight_leakage : bool
        Highlight nodes/edges involved in leakage paths.
    show_columns : bool
        Show column names on edges.
    show_capacity : bool
        Show capacity values on edges.
    cluster_by_stage : bool
        Group nodes into subgraph clusters by pipeline stage.
    max_label_length : int
        Maximum characters in node labels.
    title : str
        Optional graph title.
    """
    leakage_nodes: set[str] = set()
    leakage_edges: set[Tuple[str, str]] = set()
    if highlight_leakage:
        for path in dag.find_leakage_paths():
            for nid in path:
                leakage_nodes.add(nid)
            for i in range(len(path) - 1):
                leakage_edges.add((path[i], path[i + 1]))

    lines: list[str] = []
    lines.append("digraph PIDAG {")
    lines.append('  rankdir=TB;')
    lines.append('  fontname="Helvetica";')
    lines.append('  node [fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')

    if title:
        lines.append(f'  label="{_dot_escape(title)}";')
        lines.append("  labelloc=t;")
        lines.append("  fontsize=14;")

    if cluster_by_stage:
        stages = dag.get_pipeline_stages()
        staged_nodes: set[str] = set()
        for idx, stage in enumerate(stages):
            lines.append(f"  subgraph cluster_{idx} {{")
            lines.append(f'    label="{_dot_escape(stage.stage_type)}";')
            lines.append(f'    style=dashed; color="#BBBBBB";')
            for nid in stage.node_ids:
                node = dag.get_node(nid)
                lines.append(f"    {_dot_node(node, leakage_nodes, max_label_length)}")
                staged_nodes.add(nid)
            lines.append("  }")

        for nid, node in dag.nodes.items():
            if nid not in staged_nodes:
                lines.append(f"  {_dot_node(node, leakage_nodes, max_label_length)}")
    else:
        for nid, node in dag.nodes.items():
            lines.append(f"  {_dot_node(node, leakage_nodes, max_label_length)}")

    for edge in dag.edges:
        lines.append(
            f"  {_dot_edge(edge, leakage_edges, show_columns, show_capacity)}"
        )

    if highlight_leakage and leakage_nodes:
        lines.append("")
        lines.append("  // Legend")
        lines.append("  subgraph cluster_legend {")
        lines.append('    label="Legend"; style=filled; fillcolor="#F5F5F5";')
        lines.append('    leg_normal [label="Normal" shape=box style=filled fillcolor="#2196F3" fontcolor=white];')
        lines.append('    leg_leakage [label="Leakage" shape=box style=filled fillcolor="#E57373" fontcolor=white];')
        lines.append('    leg_normal -> leg_leakage [style=invis];')
        lines.append("  }")

    lines.append("}")
    return "\n".join(lines)


def _dot_node(
    node: PipelineNode,
    leakage_nodes: set[str],
    max_label_length: int,
) -> str:
    """Generate DOT for a single node."""
    kind = node.node_kind
    shape = _NODE_SHAPES.get(kind, "ellipse")
    color = _NODE_COLORS.get(kind, "#9E9E9E")

    if node.node_id in leakage_nodes:
        color = "#E57373"
        penwidth = "3.0"
    else:
        penwidth = "1.0"

    label = _node_label(node, max_label_length)
    safe_id = _dot_id(node.node_id)

    attrs = [
        f'label="{_dot_escape(label)}"',
        f'shape={shape}',
        f'style=filled',
        f'fillcolor="{color}"',
        f'fontcolor="white"',
        f'penwidth={penwidth}',
    ]
    return f'{safe_id} [{", ".join(attrs)}];'


def _dot_edge(
    edge: PipelineEdge,
    leakage_edges: set[Tuple[str, str]],
    show_columns: bool,
    show_capacity: bool,
) -> str:
    """Generate DOT for a single edge."""
    style, color = _EDGE_KIND_STYLES.get(edge.edge_kind, ("solid", "#333333"))

    if edge.pair in leakage_edges:
        color = "#E57373"
        penwidth = "3.0"
    else:
        penwidth = str(max(1.0, min(5.0, edge.capacity / 100.0)))

    label_parts: list[str] = []
    if show_columns and edge.columns:
        cols_str = ", ".join(sorted(edge.columns)[:5])
        if len(edge.columns) > 5:
            cols_str += f" (+{len(edge.columns) - 5})"
        label_parts.append(cols_str)
    if show_capacity and edge.capacity > 0:
        label_parts.append(f"{edge.capacity:.1f}b")
    if edge.provenance_fraction > 0.01:
        label_parts.append(f"ρ={edge.provenance_fraction:.2f}")

    label = "\\n".join(label_parts)
    src = _dot_id(edge.source_id)
    tgt = _dot_id(edge.target_id)

    attrs = [
        f'style={style}',
        f'color="{color}"',
        f'penwidth={penwidth}',
    ]
    if label:
        attrs.append(f'label="{_dot_escape(label)}"')
    return f'{src} -> {tgt} [{", ".join(attrs)}];'


def _dot_escape(s: str) -> str:
    """Escape a string for DOT labels."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _dot_id(node_id: str) -> str:
    """Convert a node ID to a valid DOT identifier."""
    safe = node_id.replace("-", "_").replace(".", "_").replace("/", "_")
    safe = safe.replace(" ", "_").replace(":", "_")
    if safe and safe[0].isdigit():
        safe = "n_" + safe
    return f'"{safe}"'


def _node_label(node: PipelineNode, max_len: int) -> str:
    """Build a human-readable label for a node."""
    parts: list[str] = [node.op_type.name]
    if isinstance(node, DataSourceNode) and node.file_path:
        fname = node.file_path.rsplit("/", 1)[-1]
        parts.append(fname)
    elif isinstance(node, FitNode) and node.estimator_class:
        parts.append(node.estimator_class)
    elif isinstance(node, TransformNode) and node.estimator_class:
        parts.append(node.estimator_class)
    elif isinstance(node, PredictNode) and node.estimator_class:
        parts.append(node.estimator_class)
    elif isinstance(node, AggregationNode) and node.groupby_columns:
        parts.append(f"by({','.join(node.groupby_columns[:3])})")
    elif isinstance(node, CustomNode) and node.function_name:
        parts.append(node.function_name)

    label = " | ".join(parts)
    if len(label) > max_len:
        label = label[: max_len - 3] + "..."
    return label


# ===================================================================
#  Mermaid output
# ===================================================================


def to_mermaid(
    dag: PIDAG,
    *,
    highlight_leakage: bool = True,
    show_capacity: bool = False,
    direction: str = "TD",
) -> str:
    """Render the DAG as a Mermaid diagram string.

    Parameters
    ----------
    dag : PIDAG
    highlight_leakage : bool
    show_capacity : bool
    direction : str
        Mermaid direction: "TD" (top-down), "LR" (left-right).
    """
    leakage_nodes: set[str] = set()
    if highlight_leakage:
        for path in dag.find_leakage_paths():
            for nid in path:
                leakage_nodes.add(nid)

    lines: list[str] = [f"graph {direction}"]

    stages = dag.get_pipeline_stages()
    staged_nodes: set[str] = set()
    for idx, stage in enumerate(stages):
        lines.append(f"    subgraph {_mermaid_id(stage.stage_id)}[\"{stage.stage_type}\"]")
        for nid in stage.node_ids:
            node = dag.get_node(nid)
            lines.append(f"        {_mermaid_node(node)}")
            staged_nodes.add(nid)
        lines.append("    end")

    for nid, node in dag.nodes.items():
        if nid not in staged_nodes:
            lines.append(f"    {_mermaid_node(node)}")

    for edge in dag.edges:
        lines.append(f"    {_mermaid_edge(edge, show_capacity)}")

    if leakage_nodes:
        lines.append("")
        for nid in leakage_nodes:
            mid = _mermaid_id(nid)
            lines.append(f"    style {mid} fill:#E57373,stroke:#C62828,color:white")

    kind_colors = {
        NodeKind.DATA_SOURCE: "#4CAF50",
        NodeKind.ESTIMATOR_FIT: "#9C27B0",
        NodeKind.ESTIMATOR_PREDICT: "#673AB7",
        NodeKind.SPLIT: "#FF9800",
        NodeKind.SINK: "#607D8B",
    }
    for nid, node in dag.nodes.items():
        if nid not in leakage_nodes:
            kind = node.node_kind
            if kind in kind_colors:
                mid = _mermaid_id(nid)
                lines.append(f"    style {mid} fill:{kind_colors[kind]},color:white")

    return "\n".join(lines)


def _mermaid_node(node: PipelineNode) -> str:
    """Generate Mermaid syntax for a node."""
    mid = _mermaid_id(node.node_id)
    label = _node_label(node, 30)
    kind = node.node_kind

    if kind == NodeKind.DATA_SOURCE:
        return f'{mid}[("{label}")]'
    if kind == NodeKind.SPLIT:
        return f"{mid}{{{label}}}"
    if kind in {NodeKind.ESTIMATOR_FIT, NodeKind.ESTIMATOR_PREDICT, NodeKind.ESTIMATOR_TRANSFORM}:
        return f"{mid}[/{label}/]"
    if kind == NodeKind.SINK:
        return f'{mid}(["{label}"])'
    return f'{mid}["{label}"]'


def _mermaid_edge(edge: PipelineEdge, show_capacity: bool) -> str:
    """Generate Mermaid syntax for an edge."""
    src = _mermaid_id(edge.source_id)
    tgt = _mermaid_id(edge.target_id)
    label = ""
    if show_capacity and edge.capacity > 0:
        label = f"|{edge.capacity:.1f}b|"
    if edge.edge_kind == EdgeKind.DATA_FLOW:
        return f"{src} -->{label} {tgt}"
    if edge.edge_kind == EdgeKind.FIT_DEPENDENCY:
        return f"{src} -.->|fit| {tgt}"
    return f"{src} -.-> {tgt}"


def _mermaid_id(node_id: str) -> str:
    """Convert a node ID to a valid Mermaid identifier."""
    safe = node_id.replace("-", "_").replace(".", "_").replace("/", "_")
    safe = safe.replace(" ", "_").replace(":", "_")
    return safe


# ===================================================================
#  ASCII art output
# ===================================================================


def to_ascii(
    dag: PIDAG,
    *,
    max_width: int = 120,
    show_edges: bool = True,
) -> str:
    """Render the DAG as an ASCII art diagram.

    Produces a top-down layered layout suitable for terminal display.
    """
    if not dag.n_nodes:
        return "(empty DAG)"

    levels = dag.node_levels()
    by_level: dict[int, list[str]] = defaultdict(list)
    for nid, lv in levels.items():
        by_level[lv].append(nid)

    max_level = max(by_level.keys()) if by_level else 0
    lines: list[str] = []

    header = f"PI-DAG ({dag.n_nodes} nodes, {dag.n_edges} edges)"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    node_positions: dict[str, int] = {}

    for lv in range(max_level + 1):
        level_nodes = by_level.get(lv, [])
        if not level_nodes:
            continue

        node_strs: list[str] = []
        for nid in level_nodes:
            node = dag.get_node(nid)
            box = _ascii_box(node, max_width // max(len(level_nodes), 1))
            node_strs.append(box)

        row = "  ".join(node_strs)
        if len(row) > max_width:
            for ns in node_strs:
                lines.append(ns)
        else:
            lines.append(row)

        col = 0
        for i, nid in enumerate(level_nodes):
            box_width = len(node_strs[i]) if i < len(node_strs) else 10
            node_positions[nid] = col + box_width // 2
            col += box_width + 2

        if show_edges and lv < max_level:
            next_level = by_level.get(lv + 1, [])
            edge_lines = _ascii_edges(level_nodes, next_level, dag, node_positions, max_width)
            lines.extend(edge_lines)

    lines.append("")

    leakage_paths = dag.find_leakage_paths()
    if leakage_paths:
        lines.append(f"⚠ {len(leakage_paths)} leakage path(s) detected!")
        for i, path in enumerate(leakage_paths[:5]):
            path_str = " -> ".join(path)
            lines.append(f"  Path {i + 1}: {path_str}")

    return "\n".join(lines)


def _ascii_box(node: PipelineNode, max_width: int) -> str:
    """Create an ASCII box for a node."""
    label = _node_label(node, max(max_width - 4, 10))
    box_width = len(label) + 4
    top = "+" + "-" * (box_width - 2) + "+"
    mid = "| " + label + " |"
    bot = "+" + "-" * (box_width - 2) + "+"
    kind_tag = node.node_kind.value[:3].upper()
    return f"{top}\n|{kind_tag}| {label} |\n{bot}"


def _ascii_edges(
    current_level: list[str],
    next_level: list[str],
    dag: PIDAG,
    positions: dict[str, int],
    max_width: int,
) -> list[str]:
    """Draw ASCII edges between two levels."""
    lines: list[str] = []
    connections: list[Tuple[str, str]] = []

    for src_id in current_level:
        for succ_id in dag.successors(src_id):
            if succ_id in next_level:
                connections.append((src_id, succ_id))

    if not connections:
        lines.append("  |")
        lines.append("  v")
        return lines

    arrow_line = [" "] * max_width
    for src_id, tgt_id in connections:
        src_pos = positions.get(src_id, 0)
        tgt_pos = positions.get(tgt_id, 0)
        mid_pos = (src_pos + tgt_pos) // 2
        if 0 <= mid_pos < max_width:
            arrow_line[mid_pos] = "|"

    lines.append("".join(arrow_line).rstrip())

    arrow_line2 = [" "] * max_width
    for src_id, tgt_id in connections:
        src_pos = positions.get(src_id, 0)
        tgt_pos = positions.get(tgt_id, 0)
        mid_pos = (src_pos + tgt_pos) // 2
        if 0 <= mid_pos < max_width:
            arrow_line2[mid_pos] = "v"

    lines.append("".join(arrow_line2).rstrip())
    return lines


# ===================================================================
#  Interactive HTML
# ===================================================================


def to_html(
    dag: PIDAG,
    *,
    title: str = "PI-DAG Visualization",
    highlight_leakage: bool = True,
    width: int = 1200,
    height: int = 800,
) -> str:
    """Generate a standalone HTML page with an interactive DAG visualization.

    Uses inline SVG generated from the DOT representation with embedded
    JavaScript for hover tooltips and click-to-select.
    """
    dot_src = to_dot(dag, highlight_leakage=highlight_leakage, show_capacity=True)
    mermaid_src = to_mermaid(dag, highlight_leakage=highlight_leakage)

    summary = dag.summary()
    leakage_count = summary.get("n_leakage_paths", 0)

    node_info_json_parts: list[str] = []
    for nid, node in dag.nodes.items():
        info = {
            "id": nid,
            "op_type": node.op_type.name,
            "kind": node.node_kind.value,
            "in_cols": len(node.input_schema),
            "out_cols": len(node.output_schema),
            "capacity": node.capacity_bound(),
            "test_fraction": node.max_test_fraction(),
        }
        parts = ", ".join(f'"{k}": {_js_value(v)}' for k, v in info.items())
        node_info_json_parts.append(f'    "{html_mod.escape(nid)}": {{{parts}}}')
    node_info_json = "{\n" + ",\n".join(node_info_json_parts) + "\n  }"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{html_mod.escape(title)}</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
    .header {{ background: #1976D2; color: white; padding: 16px 24px; border-radius: 8px; margin-bottom: 16px; }}
    .header h1 {{ margin: 0; font-size: 20px; }}
    .stats {{ display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }}
    .stat {{ background: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    .stat .value {{ font-size: 24px; font-weight: bold; color: #1976D2; }}
    .stat .label {{ font-size: 12px; color: #666; }}
    .leakage {{ background: #FFEBEE; border-left: 4px solid #F44336; }}
    .leakage .value {{ color: #F44336; }}
    .container {{ display: flex; gap: 16px; }}
    .diagram {{ flex: 1; background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); overflow: auto; }}
    .sidebar {{ width: 300px; background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    .node-detail {{ border: 1px solid #e0e0e0; border-radius: 4px; padding: 8px; margin-top: 8px; }}
    .node-detail h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .node-detail p {{ margin: 4px 0; font-size: 12px; color: #555; }}
    pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 11px; }}
    .mermaid {{ background: white; padding: 20px; }}
    .tab-bar {{ display: flex; gap: 4px; margin-bottom: 8px; }}
    .tab {{ padding: 8px 16px; cursor: pointer; border-radius: 4px 4px 0 0; background: #e0e0e0; border: none; }}
    .tab.active {{ background: white; font-weight: bold; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{html_mod.escape(title)}</h1>
  </div>
  <div class="stats">
    <div class="stat">
      <div class="value">{summary['n_nodes']}</div>
      <div class="label">Nodes</div>
    </div>
    <div class="stat">
      <div class="value">{summary['n_edges']}</div>
      <div class="label">Edges</div>
    </div>
    <div class="stat">
      <div class="value">{summary.get('depth', 0)}</div>
      <div class="label">Depth</div>
    </div>
    <div class="stat {'leakage' if leakage_count > 0 else ''}">
      <div class="value">{leakage_count}</div>
      <div class="label">Leakage Paths</div>
    </div>
    <div class="stat">
      <div class="value">{summary.get('total_edge_capacity', 0):.1f}</div>
      <div class="label">Total Capacity (bits)</div>
    </div>
  </div>
  <div class="container">
    <div class="diagram">
      <div class="tab-bar">
        <button class="tab active" onclick="showTab('dot')">Graphviz DOT</button>
        <button class="tab" onclick="showTab('mermaid')">Mermaid</button>
      </div>
      <div id="dot-view">
        <pre>{html_mod.escape(dot_src)}</pre>
      </div>
      <div id="mermaid-view" style="display:none;">
        <pre class="mermaid">{html_mod.escape(mermaid_src)}</pre>
      </div>
    </div>
    <div class="sidebar">
      <h2>Node Inspector</h2>
      <p>Select a node from the list below:</p>
      <select id="node-select" onchange="showNodeDetail()" style="width:100%; padding:8px;">
        <option value="">-- Select node --</option>
        {"".join(f'<option value="{html_mod.escape(nid)}">{html_mod.escape(nid)} ({node.op_type.name})</option>' for nid, node in dag.nodes.items())}
      </select>
      <div id="node-detail"></div>
    </div>
  </div>
  <script>
    var nodeInfo = {node_info_json};

    function showTab(name) {{
      document.getElementById('dot-view').style.display = name === 'dot' ? 'block' : 'none';
      document.getElementById('mermaid-view').style.display = name === 'mermaid' ? 'block' : 'none';
      document.querySelectorAll('.tab').forEach(function(t) {{ t.classList.remove('active'); }});
      event.target.classList.add('active');
    }}

    function showNodeDetail() {{
      var sel = document.getElementById('node-select').value;
      var el = document.getElementById('node-detail');
      if (!sel || !nodeInfo[sel]) {{ el.innerHTML = ''; return; }}
      var n = nodeInfo[sel];
      el.innerHTML = '<div class="node-detail">'
        + '<h3>' + n.id + '</h3>'
        + '<p><b>Op:</b> ' + n.op_type + '</p>'
        + '<p><b>Kind:</b> ' + n.kind + '</p>'
        + '<p><b>Input cols:</b> ' + n.in_cols + '</p>'
        + '<p><b>Output cols:</b> ' + n.out_cols + '</p>'
        + '<p><b>Capacity:</b> ' + n.capacity.toFixed(2) + ' bits</p>'
        + '<p><b>Test fraction:</b> ' + n.test_fraction.toFixed(4) + '</p>'
        + '</div>';
    }}
  </script>
</body>
</html>"""


def _js_value(v: Any) -> str:
    """Convert a Python value to a JavaScript literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return f'"{html_mod.escape(v)}"'
    return f'"{html_mod.escape(str(v))}"'


# ===================================================================
#  DAGRenderer: configurable rendering engine
# ===================================================================


@dataclass
class RenderConfig:
    """Configuration for DAG rendering."""

    format: str = "dot"
    highlight_leakage: bool = True
    show_columns: bool = False
    show_capacity: bool = True
    cluster_by_stage: bool = True
    max_label_length: int = 40
    title: str = ""
    direction: str = "TD"
    max_width: int = 120
    width: int = 1200
    height: int = 800
    color_scheme: str = "default"


class DAGRenderer:
    """Configurable rendering engine for PI-DAGs.

    Supports multiple output formats through a uniform interface.
    """

    def __init__(self, config: RenderConfig | None = None) -> None:
        self._config = config or RenderConfig()

    @property
    def config(self) -> RenderConfig:
        return self._config

    def render(self, dag: PIDAG, format: str | None = None) -> str:
        """Render the DAG in the specified format.

        Parameters
        ----------
        dag : PIDAG
        format : str | None
            Output format: ``"dot"``, ``"mermaid"``, ``"ascii"``, ``"html"``.
            Defaults to the renderer's configured format.
        """
        fmt = format or self._config.format

        if fmt == "dot":
            return to_dot(
                dag,
                highlight_leakage=self._config.highlight_leakage,
                show_columns=self._config.show_columns,
                show_capacity=self._config.show_capacity,
                cluster_by_stage=self._config.cluster_by_stage,
                max_label_length=self._config.max_label_length,
                title=self._config.title,
            )

        if fmt == "mermaid":
            return to_mermaid(
                dag,
                highlight_leakage=self._config.highlight_leakage,
                show_capacity=self._config.show_capacity,
                direction=self._config.direction,
            )

        if fmt == "ascii":
            return to_ascii(
                dag,
                max_width=self._config.max_width,
            )

        if fmt == "html":
            return to_html(
                dag,
                title=self._config.title or "PI-DAG",
                highlight_leakage=self._config.highlight_leakage,
                width=self._config.width,
                height=self._config.height,
            )

        raise ValueError(f"Unsupported format: {fmt!r}. Use 'dot', 'mermaid', 'ascii', or 'html'.")

    def render_to_file(self, dag: PIDAG, path: str, format: str | None = None) -> None:
        """Render the DAG and write the result to a file."""
        content = self.render(dag, format=format)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def supported_formats(self) -> list[str]:
        """List of supported output formats."""
        return ["dot", "mermaid", "ascii", "html"]
