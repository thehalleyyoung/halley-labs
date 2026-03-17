"""
taintflow.visualization.dag_viz -- DAG visualization for the Pipeline Information DAG.

Generates SVG, DOT (Graphviz), and ASCII art representations of a
:class:`~taintflow.dag.pi_dag.PipelineDAG`.  Nodes are colour-coded by
leakage severity, sized proportionally to their leakage contribution,
and laid out using a simplified Sugiyama layered-graph algorithm.

All rendering uses the standard library only—no external dependencies.
"""

from __future__ import annotations

import html
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from taintflow.core.types import EdgeKind, OpType, Severity
from taintflow.dag.edges import DAGEdge
from taintflow.dag.nodes import DAGNode
from taintflow.dag.pi_dag import PipelineDAG

# ===================================================================
#  Colour palettes
# ===================================================================

_SEVERITY_FILL: Dict[str, str] = {
    "negligible": "#d4edda",
    "warning": "#fff3cd",
    "critical": "#f8d7da",
}
_SEVERITY_STROKE: Dict[str, str] = {
    "negligible": "#28a745",
    "warning": "#ffc107",
    "critical": "#dc3545",
}
_EDGE_COLOURS: Dict[str, str] = {
    EdgeKind.DATA_FLOW.value: "#333333",
    EdgeKind.FIT_DEPENDENCY.value: "#0066cc",
    EdgeKind.PARAMETER_FLOW.value: "#9933cc",
    EdgeKind.CONTROL_FLOW.value: "#999999",
    EdgeKind.LABEL_FLOW.value: "#cc6600",
    EdgeKind.INDEX_FLOW.value: "#669999",
    EdgeKind.AUXILIARY.value: "#bbbbbb",
}
_OP_ICONS: Dict[str, str] = {
    "data_source": "📂",
    "split": "✂",
    "merge": "🔗",
    "transform": "⚙",
    "fit": "🎓",
    "predict": "🔮",
    "feature_engineering": "🧪",
    "evaluation": "📊",
    "sink": "💾",
}


def _escape(text: str) -> str:
    """XML-escape a string for safe embedding in SVG."""
    return html.escape(str(text), quote=True)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


# ===================================================================
#  SVGElement – lightweight builder for SVG markup
# ===================================================================


class SVGElement:
    """Builder for individual SVG elements.

    Supports nesting, arbitrary attributes, and text content.

    Example::

        rect = SVGElement("rect", x="10", y="10", width="80", height="40")
        print(rect.render())
    """

    def __init__(self, tag: str, text: str = "", **attrs: Any) -> None:
        self.tag: str = tag
        self.text: str = text
        self.attrs: Dict[str, str] = {k.rstrip("_").replace("_", "-"): str(v) for k, v in attrs.items()}
        self.children: List[SVGElement] = []

    def add(self, child: "SVGElement") -> "SVGElement":
        """Append a child element and return *self* for chaining."""
        self.children.append(child)
        return self

    def render(self, indent: int = 0) -> str:
        """Render this element (and its children) as an SVG string."""
        pad = "  " * indent
        attr_str = "".join(f' {k}="{_escape(v)}"' for k, v in self.attrs.items())
        if not self.children and not self.text:
            return f"{pad}<{self.tag}{attr_str}/>"
        parts = [f"{pad}<{self.tag}{attr_str}>"]
        if self.text:
            parts.append(f"{pad}  {_escape(self.text)}")
        for child in self.children:
            parts.append(child.render(indent + 1))
        parts.append(f"{pad}</{self.tag}>")
        return "\n".join(parts)


# ===================================================================
#  SVGCanvas – coordinate system wrapper
# ===================================================================


class SVGCanvas:
    """Canvas that accumulates SVG elements and emits a complete SVG document.

    Parameters
    ----------
    width : float
        Canvas width in user units.
    height : float
        Canvas height in user units.
    padding : float
        Padding around the content area.
    """

    def __init__(self, width: float = 800, height: float = 600, padding: float = 20) -> None:
        self.width: float = width
        self.height: float = height
        self.padding: float = padding
        self._defs: List[SVGElement] = []
        self._body: List[SVGElement] = []

    def add_def(self, element: SVGElement) -> None:
        """Add an element to the ``<defs>`` section."""
        self._defs.append(element)

    def add(self, element: SVGElement) -> None:
        """Add an element to the canvas body."""
        self._body.append(element)

    def render(self) -> str:
        """Render the complete SVG document as a string."""
        root = SVGElement(
            "svg",
            xmlns="http://www.w3.org/2000/svg",
            width=str(self.width),
            height=str(self.height),
            viewBox=f"0 0 {self.width} {self.height}",
        )
        if self._defs:
            defs = SVGElement("defs")
            for d in self._defs:
                defs.add(d)
            root.add(defs)
        # Background
        root.add(SVGElement("rect", width=str(self.width), height=str(self.height), fill="white"))
        group = SVGElement("g", transform=f"translate({self.padding},{self.padding})")
        for elem in self._body:
            group.add(elem)
        root.add(group)
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + root.render()


# ===================================================================
#  Layout helpers
# ===================================================================


@dataclass
class NodeLayout:
    """Computed position and dimensions for a single node."""

    node_id: str
    x: float = 0.0
    y: float = 0.0
    width: float = 120.0
    height: float = 50.0
    layer: int = 0

    @property
    def cx(self) -> float:
        """Horizontal centre."""
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        """Vertical centre."""
        return self.y + self.height / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.width


# ===================================================================
#  SugiyamaLayout – simplified layered graph layout
# ===================================================================


class SugiyamaLayout:
    """Simplified Sugiyama layered-graph layout.

    Steps:
    1. Assign layers via topological order (longest-path layering).
    2. Minimise edge crossings using the barycenter heuristic.
    3. Position nodes within layers to minimise edge length.

    Parameters
    ----------
    node_width : float
        Default node width.
    node_height : float
        Default node height.
    h_gap : float
        Horizontal gap between nodes in the same layer.
    v_gap : float
        Vertical gap between layers.
    """

    def __init__(
        self,
        node_width: float = 140,
        node_height: float = 54,
        h_gap: float = 40,
        v_gap: float = 80,
    ) -> None:
        self.node_width = node_width
        self.node_height = node_height
        self.h_gap = h_gap
        self.v_gap = v_gap

    # -- public API ----------------------------------------------------------

    def layout(
        self,
        dag: PipelineDAG,
        *,
        node_sizes: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, NodeLayout]:
        """Compute layout positions for every node in *dag*.

        Parameters
        ----------
        dag : PipelineDAG
            The pipeline graph.
        node_sizes : dict, optional
            ``{node_id: (width, height)}`` overrides.

        Returns
        -------
        dict[str, NodeLayout]
        """
        if dag.node_count == 0:
            return {}

        topo = dag.topological_sort()
        layers = self._assign_layers(dag, topo)
        layers = self._minimize_crossings(dag, layers)
        return self._position_nodes(layers, dag, node_sizes)

    # -- step 1: layer assignment (longest-path) -----------------------------

    def _assign_layers(
        self, dag: PipelineDAG, topo: List[str]
    ) -> Dict[int, List[str]]:
        """Assign each node to a layer using longest-path layering."""
        depth: Dict[str, int] = {}
        for nid in topo:
            preds = dag.predecessors(nid)
            if not preds:
                depth[nid] = 0
            else:
                depth[nid] = max(depth.get(p, 0) for p in preds) + 1

        layers: Dict[int, List[str]] = defaultdict(list)
        for nid, d in depth.items():
            layers[d].append(nid)
        return dict(layers)

    # -- step 2: crossing minimisation (barycenter heuristic) ----------------

    def _minimize_crossings(
        self, dag: PipelineDAG, layers: Dict[int, List[str]]
    ) -> Dict[int, List[str]]:
        """Reduce edge crossings using the barycenter heuristic.

        Two sweeps (down then up) are applied.
        """
        layer_indices = sorted(layers.keys())
        if len(layer_indices) < 2:
            return layers

        # Build position index
        pos: Dict[str, int] = {}
        for layer_nodes in layers.values():
            for idx, nid in enumerate(layer_nodes):
                pos[nid] = idx

        # Down sweep
        for i in range(1, len(layer_indices)):
            li = layer_indices[i]
            bary: Dict[str, float] = {}
            for nid in layers[li]:
                preds = dag.predecessors(nid)
                pred_positions = [pos[p] for p in preds if p in pos]
                if pred_positions:
                    bary[nid] = sum(pred_positions) / len(pred_positions)
                else:
                    bary[nid] = float(pos.get(nid, 0))
            layers[li] = sorted(layers[li], key=lambda n: bary.get(n, 0.0))
            for idx, nid in enumerate(layers[li]):
                pos[nid] = idx

        # Up sweep
        for i in range(len(layer_indices) - 2, -1, -1):
            li = layer_indices[i]
            bary = {}
            for nid in layers[li]:
                succs = dag.successors(nid)
                succ_positions = [pos[s] for s in succs if s in pos]
                if succ_positions:
                    bary[nid] = sum(succ_positions) / len(succ_positions)
                else:
                    bary[nid] = float(pos.get(nid, 0))
            layers[li] = sorted(layers[li], key=lambda n: bary.get(n, 0.0))
            for idx, nid in enumerate(layers[li]):
                pos[nid] = idx

        return layers

    # -- step 3: position assignment -----------------------------------------

    def _position_nodes(
        self,
        layers: Dict[int, List[str]],
        dag: PipelineDAG,
        node_sizes: Optional[Dict[str, Tuple[float, float]]],
    ) -> Dict[str, NodeLayout]:
        """Assign (x, y) coordinates to each node within its layer."""
        result: Dict[str, NodeLayout] = {}
        max_layer_width = 0.0

        for li in sorted(layers.keys()):
            layer_width = 0.0
            for nid in layers[li]:
                w, h = self.node_width, self.node_height
                if node_sizes and nid in node_sizes:
                    w, h = node_sizes[nid]
                layer_width += w + self.h_gap
            layer_width -= self.h_gap
            max_layer_width = max(max_layer_width, layer_width)

        for li in sorted(layers.keys()):
            y = li * (self.node_height + self.v_gap)
            layer_widths: List[float] = []
            for nid in layers[li]:
                w, h = self.node_width, self.node_height
                if node_sizes and nid in node_sizes:
                    w, h = node_sizes[nid]
                layer_widths.append(w)
            total_w = sum(layer_widths) + self.h_gap * max(0, len(layers[li]) - 1)
            x_offset = (max_layer_width - total_w) / 2

            x = x_offset
            for idx, nid in enumerate(layers[li]):
                w = layer_widths[idx]
                h = self.node_height
                if node_sizes and nid in node_sizes:
                    _, h = node_sizes[nid]
                result[nid] = NodeLayout(
                    node_id=nid, x=x, y=y, width=w, height=h, layer=li,
                )
                x += w + self.h_gap

        return result


# ===================================================================
#  LayoutEngine – façade
# ===================================================================


class LayoutEngine:
    """Façade for automatic node positioning.

    Delegates to :class:`SugiyamaLayout` by default.

    Parameters
    ----------
    algorithm : str
        Layout algorithm name (currently only ``"sugiyama"``).
    """

    def __init__(self, algorithm: str = "sugiyama", **kwargs: Any) -> None:
        self.algorithm = algorithm
        if algorithm == "sugiyama":
            self._impl = SugiyamaLayout(**kwargs)
        else:
            raise ValueError(f"Unknown layout algorithm: {algorithm!r}")

    def compute(
        self,
        dag: PipelineDAG,
        *,
        node_sizes: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, NodeLayout]:
        """Compute positions for all nodes in *dag*."""
        return self._impl.layout(dag, node_sizes=node_sizes)


# ===================================================================
#  NodeRenderer
# ===================================================================


class NodeRenderer:
    """Render DAG nodes as SVG elements.

    Nodes are drawn as rounded rectangles with an operation label, an
    optional icon, and colour coding by severity.

    Parameters
    ----------
    show_icons : bool
        Prepend an emoji icon to the node label.
    font_size : float
        Font size in points for the label text.
    """

    def __init__(self, show_icons: bool = True, font_size: float = 11) -> None:
        self.show_icons = show_icons
        self.font_size = font_size

    def render(
        self,
        node: DAGNode,
        layout: NodeLayout,
        severity: Severity = Severity.NEGLIGIBLE,
        leakage_bits: float = 0.0,
    ) -> SVGElement:
        """Render a single node as an SVG group element.

        Parameters
        ----------
        node : DAGNode
            The pipeline node.
        layout : NodeLayout
            Pre-computed position and dimensions.
        severity : Severity
            Leakage severity for colour coding.
        leakage_bits : float
            Leakage amount (used in tooltip).

        Returns
        -------
        SVGElement
        """
        sev_key = severity.value
        fill = _SEVERITY_FILL.get(sev_key, "#ffffff")
        stroke = _SEVERITY_STROKE.get(sev_key, "#888888")
        stroke_width = "2" if severity != Severity.NEGLIGIBLE else "1.5"

        group = SVGElement("g", class_="dag-node", id=f"node-{_escape(node.node_id)}")

        # Rounded rectangle
        group.add(SVGElement(
            "rect",
            x=str(layout.x), y=str(layout.y),
            width=str(layout.width), height=str(layout.height),
            rx="6", ry="6",
            fill=fill, stroke=stroke, stroke_width=stroke_width,
        ))

        # Label
        label = node.label or node.op_type.value
        if self.show_icons:
            icon = _OP_ICONS.get(node.op_type.value, "")
            if not icon:
                for key, ico in _OP_ICONS.items():
                    if key in node.op_type.value:
                        icon = ico
                        break
            if icon:
                label = f"{icon} {label}"

        # Truncate label to fit
        max_chars = int(layout.width / (self.font_size * 0.6))
        if len(label) > max_chars:
            label = label[: max_chars - 1] + "…"

        group.add(SVGElement(
            "text",
            text=label,
            x=str(layout.cx), y=str(layout.cy + self.font_size / 3),
            text_anchor="middle",
            font_family="sans-serif",
            font_size=str(self.font_size),
            fill="#333333",
        ))

        # Tooltip
        tooltip_lines = [
            f"Node: {node.node_id}",
            f"Op: {node.op_type.value}",
            f"Severity: {severity.name}",
        ]
        if leakage_bits > 0:
            tooltip_lines.append(f"Leakage: {leakage_bits:.2f} bits")
        if node.shape:
            tooltip_lines.append(f"Shape: {node.shape.n_rows}×{node.shape.n_cols}")
        group.add(SVGElement("title", text="\n".join(tooltip_lines)))

        return group


# ===================================================================
#  EdgeRenderer
# ===================================================================


class EdgeRenderer:
    """Render DAG edges as SVG paths with arrowheads and optional labels.

    Parameters
    ----------
    font_size : float
        Font size for edge labels.
    show_labels : bool
        Whether to render edge labels.
    """

    _ARROW_MARKER_ID = "arrowhead"

    def __init__(self, font_size: float = 9, show_labels: bool = True) -> None:
        self.font_size = font_size
        self.show_labels = show_labels

    def arrow_marker_def(self) -> SVGElement:
        """Return an SVG ``<marker>`` definition for arrowheads."""
        marker = SVGElement(
            "marker",
            id=self._ARROW_MARKER_ID,
            viewBox="0 0 10 10",
            refX="10", refY="5",
            markerWidth="8", markerHeight="8",
            orient="auto-start-reverse",
        )
        marker.add(SVGElement("path", d="M 0 0 L 10 5 L 0 10 z", fill="#333"))
        return marker

    def render(
        self,
        edge: DAGEdge,
        src_layout: NodeLayout,
        tgt_layout: NodeLayout,
        leakage_bits: float = 0.0,
        severity: Severity = Severity.NEGLIGIBLE,
    ) -> SVGElement:
        """Render a single edge as an SVG group.

        Parameters
        ----------
        edge : DAGEdge
            The DAG edge.
        src_layout, tgt_layout : NodeLayout
            Layouts for source and target nodes.
        leakage_bits : float
            Leakage through this edge (for annotation).
        severity : Severity
            Severity for colour coding.

        Returns
        -------
        SVGElement
        """
        colour = _EDGE_COLOURS.get(edge.edge_kind.value, "#333333")
        if severity == Severity.CRITICAL:
            colour = _SEVERITY_STROKE["critical"]
        elif severity == Severity.WARNING:
            colour = _SEVERITY_STROKE["warning"]

        stroke_width = "1.5"
        if severity == Severity.CRITICAL:
            stroke_width = "2.5"
        elif severity == Severity.WARNING:
            stroke_width = "2"

        # Compute path from bottom-centre of source to top-centre of target
        x1, y1 = src_layout.cx, src_layout.bottom
        x2, y2 = tgt_layout.cx, tgt_layout.top

        group = SVGElement("g", class_="dag-edge")

        # Quadratic bezier for a smooth curve
        mid_y = (y1 + y2) / 2
        d = f"M {x1:.1f} {y1:.1f} C {x1:.1f} {mid_y:.1f}, {x2:.1f} {mid_y:.1f}, {x2:.1f} {y2:.1f}"

        dash = ""
        if edge.edge_kind == EdgeKind.FIT_DEPENDENCY:
            dash = "5,3"
        elif edge.edge_kind == EdgeKind.PARAMETER_FLOW:
            dash = "2,2"

        path_attrs: Dict[str, str] = {
            "d": d,
            "fill": "none",
            "stroke": colour,
            "stroke_width": stroke_width,
            "marker_end": f"url(#{self._ARROW_MARKER_ID})",
        }
        if dash:
            path_attrs["stroke_dasharray"] = dash

        group.add(SVGElement("path", **path_attrs))

        # Edge label
        if self.show_labels and leakage_bits > 0:
            lx = (x1 + x2) / 2 + 5
            ly = mid_y - 4
            group.add(SVGElement(
                "text",
                text=f"{leakage_bits:.1f}b",
                x=str(lx), y=str(ly),
                font_family="sans-serif",
                font_size=str(self.font_size),
                fill=colour,
            ))

        # Tooltip
        tips = [f"Edge: {edge.source_id[:8]} → {edge.target_id[:8]}"]
        tips.append(f"Kind: {edge.edge_kind.value}")
        if edge.weight > 0:
            tips.append(f"Capacity: {edge.weight:.2f} bits")
        group.add(SVGElement("title", text="\n".join(tips)))

        return group


# ===================================================================
#  DAGVisualizer – main entry point
# ===================================================================


class DAGVisualizer:
    """Generate visual representations of a :class:`PipelineDAG`.

    Supports SVG output, DOT format (for Graphviz), and ASCII art
    fallback for terminal rendering.

    Parameters
    ----------
    dag : PipelineDAG
        The pipeline graph to visualise.
    node_leakage : dict, optional
        ``{node_id: leakage_bits}`` mapping for colour coding.
    edge_leakage : dict, optional
        ``{(source_id, target_id): leakage_bits}`` mapping.
    layout_engine : LayoutEngine, optional
        Custom layout engine.  Defaults to Sugiyama.
    """

    def __init__(
        self,
        dag: PipelineDAG,
        node_leakage: Optional[Dict[str, float]] = None,
        edge_leakage: Optional[Dict[Tuple[str, str], float]] = None,
        layout_engine: Optional[LayoutEngine] = None,
    ) -> None:
        self.dag = dag
        self.node_leakage: Dict[str, float] = node_leakage or {}
        self.edge_leakage: Dict[Tuple[str, str], float] = edge_leakage or {}
        self._layout_engine = layout_engine or LayoutEngine()
        self._node_renderer = NodeRenderer()
        self._edge_renderer = EdgeRenderer()

    # -- severity helpers ----------------------------------------------------

    def _severity_for_node(self, node_id: str) -> Severity:
        bits = self.node_leakage.get(node_id, 0.0)
        return Severity.from_bits(bits)

    def _severity_for_edge(self, src: str, tgt: str) -> Severity:
        bits = self.edge_leakage.get((src, tgt), 0.0)
        return Severity.from_bits(bits)

    # -- node sizing ---------------------------------------------------------

    def _compute_node_sizes(self) -> Dict[str, Tuple[float, float]]:
        """Size nodes proportionally to leakage contribution."""
        sizes: Dict[str, Tuple[float, float]] = {}
        if not self.node_leakage:
            return sizes
        max_leak = max(self.node_leakage.values()) if self.node_leakage else 1.0
        max_leak = max(max_leak, 1e-9)
        for nid, bits in self.node_leakage.items():
            scale = 1.0 + 0.5 * _clamp(bits / max_leak)
            sizes[nid] = (140 * scale, 54 * scale)
        return sizes

    # -- SVG -----------------------------------------------------------------

    def to_svg(self, *, width: Optional[float] = None, height: Optional[float] = None) -> str:
        """Render the DAG as a complete SVG string.

        Parameters
        ----------
        width, height : float, optional
            Override canvas dimensions.  Auto-calculated if omitted.

        Returns
        -------
        str
            The SVG document.
        """
        node_sizes = self._compute_node_sizes()
        layouts = self._layout_engine.compute(self.dag, node_sizes=node_sizes)

        if not layouts:
            canvas = SVGCanvas(width=width or 200, height=height or 100)
            canvas.add(SVGElement("text", text="(empty DAG)", x="50", y="50", fill="#999"))
            return canvas.render()

        # Compute canvas bounds
        max_x = max(nl.right for nl in layouts.values()) + 40
        max_y = max(nl.bottom for nl in layouts.values()) + 40
        canvas = SVGCanvas(
            width=width or max_x + 40,
            height=height or max_y + 40,
        )

        # Arrow marker definition
        canvas.add_def(self._edge_renderer.arrow_marker_def())

        # Render edges first (so nodes draw on top)
        all_edges = self.dag.edges
        for edge in all_edges:
            src_l = layouts.get(edge.source_id)
            tgt_l = layouts.get(edge.target_id)
            if src_l and tgt_l:
                bits = self.edge_leakage.get((edge.source_id, edge.target_id), edge.weight)
                sev = Severity.from_bits(bits)
                canvas.add(self._edge_renderer.render(edge, src_l, tgt_l, bits, sev))

        # Render nodes
        for nid, nl in layouts.items():
            node = self.dag.get_node(nid)
            bits = self.node_leakage.get(nid, 0.0)
            sev = self._severity_for_node(nid)
            canvas.add(self._node_renderer.render(node, nl, sev, bits))

        # Legend
        self._add_legend(canvas, max_x)

        return canvas.render()

    def save_svg(self, path: str | Path) -> None:
        """Write the SVG to a file."""
        Path(path).write_text(self.to_svg(), encoding="utf-8")

    # -- legend --------------------------------------------------------------

    def _add_legend(self, canvas: SVGCanvas, x_offset: float) -> None:
        """Add a severity legend to the canvas."""
        legend_x = 10.0
        legend_y = canvas.height - 70
        items = [
            ("Negligible", "#d4edda", "#28a745"),
            ("Warning", "#fff3cd", "#ffc107"),
            ("Critical", "#f8d7da", "#dc3545"),
        ]
        for i, (label, fill, stroke) in enumerate(items):
            y = legend_y + i * 20
            canvas.add(SVGElement(
                "rect", x=str(legend_x), y=str(y),
                width="14", height="14", rx="2", ry="2",
                fill=fill, stroke=stroke, stroke_width="1.5",
            ))
            canvas.add(SVGElement(
                "text", text=label,
                x=str(legend_x + 20), y=str(y + 11),
                font_family="sans-serif", font_size="10", fill="#333",
            ))

    # -- DOT export ----------------------------------------------------------

    def to_dot(self) -> str:
        """Export the DAG in Graphviz DOT format.

        Returns
        -------
        str
            DOT language string.
        """
        lines: List[str] = ['digraph PipelineDAG {']
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10];')
        lines.append('  edge [fontname="sans-serif", fontsize=8];')
        lines.append("")

        for nid in self.dag.node_ids:
            node = self.dag.get_node(nid)
            bits = self.node_leakage.get(nid, 0.0)
            sev = self._severity_for_node(nid)
            fill = _SEVERITY_FILL.get(sev.value, "#ffffff")
            stroke = _SEVERITY_STROKE.get(sev.value, "#888888")
            label = node.label or node.op_type.value
            tooltip = f"{nid}: {bits:.2f} bits"
            lines.append(
                f'  "{nid}" [label="{_escape(label)}", '
                f'fillcolor="{fill}", color="{stroke}", '
                f'tooltip="{_escape(tooltip)}"];'
            )

        lines.append("")

        for edge in self.dag.edges:
            bits = self.edge_leakage.get((edge.source_id, edge.target_id), edge.weight)
            sev = Severity.from_bits(bits)
            colour = _EDGE_COLOURS.get(edge.edge_kind.value, "#333")
            if sev == Severity.CRITICAL:
                colour = _SEVERITY_STROKE["critical"]
            elif sev == Severity.WARNING:
                colour = _SEVERITY_STROKE["warning"]
            style = "solid"
            if edge.edge_kind == EdgeKind.FIT_DEPENDENCY:
                style = "dashed"
            elif edge.edge_kind == EdgeKind.PARAMETER_FLOW:
                style = "dotted"
            label = f"{bits:.1f}b" if bits > 0 else ""
            lines.append(
                f'  "{edge.source_id}" -> "{edge.target_id}" '
                f'[label="{label}", color="{colour}", style="{style}"];'
            )

        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, path: str | Path) -> None:
        """Write the DOT representation to a file."""
        Path(path).write_text(self.to_dot(), encoding="utf-8")

    # -- ASCII art -----------------------------------------------------------

    def to_ascii(self, *, max_width: int = 100) -> str:
        """Render a simplified ASCII art representation.

        Suitable for terminal output when SVG rendering is unavailable.

        Parameters
        ----------
        max_width : int
            Maximum character width.

        Returns
        -------
        str
            Multi-line ASCII art string.
        """
        if self.dag.node_count == 0:
            return "(empty DAG)"

        topo = self.dag.topological_sort()
        # Assign layers
        depth: Dict[str, int] = {}
        for nid in topo:
            preds = self.dag.predecessors(nid)
            if not preds:
                depth[nid] = 0
            else:
                depth[nid] = max(depth.get(p, 0) for p in preds) + 1

        layers: Dict[int, List[str]] = defaultdict(list)
        for nid, d in depth.items():
            layers[d].append(nid)

        lines: List[str] = []
        sev_chars = {"negligible": " ", "warning": "!", "critical": "X"}

        for li in sorted(layers.keys()):
            row_boxes: List[str] = []
            for nid in layers[li]:
                node = self.dag.get_node(nid)
                bits = self.node_leakage.get(nid, 0.0)
                sev = self._severity_for_node(nid)
                marker = sev_chars.get(sev.value, "?")
                label = node.label or node.op_type.value
                # Truncate
                label = label[:16]
                box_content = f"[{marker}{label:^16s}{marker}]"
                row_boxes.append(box_content)

            row_line = "  ".join(row_boxes)
            if len(row_line) > max_width:
                row_line = row_line[: max_width - 3] + "..."
            lines.append(row_line)

            # Draw connectors to next layer
            if li < max(layers.keys()):
                next_layer = layers.get(li + 1, [])
                if next_layer:
                    # Simple vertical pipe connectors
                    connector_parts: List[str] = []
                    for nid in layers[li]:
                        succs = set(self.dag.successors(nid))
                        has_child = bool(succs & set(next_layer))
                        box_w = 20  # matches box width
                        if has_child:
                            connector_parts.append(" " * 9 + "|" + " " * 10)
                        else:
                            connector_parts.append(" " * 20)
                    conn_line = "".join(connector_parts)
                    if conn_line.strip():
                        lines.append(conn_line[: max_width])

        # Footer with summary
        total_bits = sum(self.node_leakage.values())
        n_critical = sum(
            1 for nid in self.dag.node_ids
            if self._severity_for_node(nid) == Severity.CRITICAL
        )
        n_warning = sum(
            1 for nid in self.dag.node_ids
            if self._severity_for_node(nid) == Severity.WARNING
        )
        lines.append("")
        lines.append(
            f"Legend: [ ] negligible  [!] warning  [X] critical  "
            f"| Total: {total_bits:.1f} bits, "
            f"{n_critical} critical, {n_warning} warning"
        )
        return "\n".join(lines)
