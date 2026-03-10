"""
DAG visualization helpers for reports.

Provides functions to annotate DAG visualizations with fragility scores
and highlight witness edit sets.  Outputs Graphviz DOT format for maximum
compatibility, plus SVG bar-chart and convergence-plot helpers.
"""

from __future__ import annotations

import html
import math
from typing import Any, Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    FragilityChannel,
    FragilityScore,
    StructuralEdit,
)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _score_to_hex(score: float) -> str:
    """Map a fragility score in [0,1] to a hex colour (green→yellow→red)."""
    score = max(0.0, min(1.0, score))
    if score < 0.5:
        r = int(2 * score * 255)
        g = 255
    else:
        r = 255
        g = int(2 * (1 - score) * 255)
    return f"#{r:02x}{g:02x}00"


def _severity_color(score: float) -> str:
    """Return a named colour string for a severity level."""
    if score >= 0.7:
        return "red"
    elif score >= 0.4:
        return "orange"
    elif score >= 0.1:
        return "gold"
    return "forestgreen"


# ---------------------------------------------------------------------------
# DOT generation
# ---------------------------------------------------------------------------


def annotate_dag_dot(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
    fragility_scores: Sequence[FragilityScore] | None = None,
    witness_edits: Sequence[StructuralEdit] | None = None,
    treatment: int | None = None,
    outcome: int | None = None,
    title: str = "CausalCert DAG",
) -> str:
    """Generate a Graphviz DOT string with fragility annotations.

    Edges are coloured on a red-green scale by fragility, and witness
    edits are highlighted with dashed/bold styling.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Node names.
    fragility_scores : Sequence[FragilityScore] | None
        Per-edge fragility scores.
    witness_edits : Sequence[StructuralEdit] | None
        Witness edit set to highlight.
    treatment, outcome : int | None
        Treatment and outcome nodes (drawn differently).
    title : str
        Graph title.

    Returns
    -------
    str
        Annotated DOT string.
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    n = adj_arr.shape[0]
    names = node_names or [str(i) for i in range(n)]

    # Build lookup for fragility scores
    edge_scores: dict[tuple[int, int], float] = {}
    if fragility_scores:
        for fs in fragility_scores:
            edge_scores[fs.edge] = fs.total_score

    # Build lookup for witness edits
    witness_set: set[tuple[int, int]] = set()
    witness_adds: set[tuple[int, int]] = set()
    witness_deletes: set[tuple[int, int]] = set()
    witness_reverses: set[tuple[int, int]] = set()
    if witness_edits:
        for edit in witness_edits:
            edge = (edit.source, edit.target)
            witness_set.add(edge)
            if edit.edit_type == EditType.ADD:
                witness_adds.add(edge)
            elif edit.edit_type == EditType.DELETE:
                witness_deletes.add(edge)
            elif edit.edit_type == EditType.REVERSE:
                witness_reverses.add(edge)

    lines: list[str] = []
    lines.append(f'digraph "{title}" {{')
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=ellipse, style=filled, fillcolor=white, '
                 'fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')
    lines.append("")

    # Nodes
    for i in range(n):
        name = _dot_escape(names[i])
        attrs: list[str] = [f'label="{name}"']

        if treatment is not None and i == treatment:
            attrs.append('fillcolor="#cce5ff"')
            attrs.append('penwidth=2')
        elif outcome is not None and i == outcome:
            attrs.append('fillcolor="#d4edda"')
            attrs.append('penwidth=2')

        lines.append(f'  {i} [{", ".join(attrs)}];')

    lines.append("")

    # Existing edges
    for i in range(n):
        for j in range(n):
            if not adj_arr[i, j]:
                continue

            edge = (i, j)
            attrs: list[str] = []

            # Fragility colour
            if edge in edge_scores:
                score = edge_scores[edge]
                colour = _score_to_hex(score)
                attrs.append(f'color="{colour}"')
                attrs.append(f'label="{score:.2f}"')
                if score >= 0.7:
                    attrs.append("penwidth=3")
                elif score >= 0.4:
                    attrs.append("penwidth=2")

            # Witness styling
            if edge in witness_deletes:
                attrs.append('style=dashed')
                attrs.append('color="red"')
                attrs.append('label="DELETE"')
            elif edge in witness_reverses:
                attrs.append('style=bold')
                attrs.append('color="blue"')
                attrs.append('label="REVERSE"')

            attr_str = f' [{", ".join(attrs)}]' if attrs else ""
            lines.append(f"  {i} -> {j}{attr_str};")

    # Witness additions (edges not in original graph)
    for edge in witness_adds:
        i, j = edge
        lines.append(
            f'  {i} -> {j} [style=dashed, color="green", '
            f'label="ADD", penwidth=2];'
        )

    lines.append("}")
    return "\n".join(lines)


def _dot_escape(text: str) -> str:
    """Escape a string for use in DOT labels."""
    return text.replace('"', '\\"').replace("\\", "\\\\")


# ---------------------------------------------------------------------------
# Fragility bar chart (SVG)
# ---------------------------------------------------------------------------


def fragility_bar_chart_svg(
    scores: Sequence[FragilityScore],
    node_names: list[str] | None = None,
    max_edges: int = 15,
    width: int = 600,
    bar_height: int = 28,
    title: str = "Edge Fragility Scores",
) -> str:
    """Generate an SVG bar chart of fragility scores.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges (should be pre-sorted descending).
    node_names : list[str] | None
        Node names.
    max_edges : int
        Maximum bars to show.
    width : int
        SVG width.
    bar_height : int
        Height per bar.
    title : str
        Chart title.

    Returns
    -------
    str
        SVG string.
    """
    shown = list(scores)[:max_edges]
    if not shown:
        return '<svg width="200" height="30"><text x="5" y="20">No data</text></svg>'

    label_w = 160
    bar_w = width - label_w - 80
    title_h = 30
    total_h = len(shown) * bar_height + title_h + 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{total_h}">',
        '<style>',
        '  text.title { font: bold 14px sans-serif; fill: #333; }',
        '  text.label { font: 12px monospace; fill: #333; }',
        '  text.value { font: bold 11px monospace; fill: #333; }',
        '</style>',
        f'<text x="{width // 2}" y="18" class="title" text-anchor="middle">'
        f'{html.escape(title)}</text>',
    ]

    for i, fs in enumerate(shown):
        src, tgt = fs.edge
        if node_names:
            src_n = node_names[src] if src < len(node_names) else str(src)
            tgt_n = node_names[tgt] if tgt < len(node_names) else str(tgt)
            label = f"{src_n}→{tgt_n}"
        else:
            label = f"{src}→{tgt}"

        y = i * bar_height + title_h + 5
        w = max(2, int(fs.total_score * bar_w))
        colour = _score_to_hex(fs.total_score)

        parts.append(
            f'<text x="5" y="{y + 18}" class="label">'
            f'{html.escape(label[:20])}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{w}" '
            f'height="{bar_height - 6}" fill="{colour}" rx="3" '
            f'opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{label_w + w + 5}" y="{y + 17}" class="value">'
            f'{fs.total_score:.3f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Channel decomposition chart
# ---------------------------------------------------------------------------


def channel_decomposition_svg(
    scores: Sequence[FragilityScore],
    node_names: list[str] | None = None,
    max_edges: int = 10,
    width: int = 700,
    bar_height: int = 30,
) -> str:
    """Generate an SVG stacked bar chart showing channel decomposition.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges.
    node_names : list[str] | None
        Node names.
    max_edges : int
        Maximum bars.
    width : int
        SVG width.
    bar_height : int
        Height per bar.

    Returns
    -------
    str
        SVG string.
    """
    shown = list(scores)[:max_edges]
    if not shown:
        return '<svg width="200" height="30"><text x="5" y="20">No data</text></svg>'

    label_w = 160
    bar_w = width - label_w - 100
    legend_h = 30
    total_h = len(shown) * bar_height + legend_h + 30

    channel_colors = {
        FragilityChannel.D_SEPARATION: "#4e79a7",
        FragilityChannel.IDENTIFICATION: "#f28e2b",
        FragilityChannel.ESTIMATION: "#e15759",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{total_h}">',
        '<style>text { font: 11px monospace; fill: #333; }</style>',
    ]

    # Legend
    lx = label_w
    for ch, color in channel_colors.items():
        parts.append(
            f'<rect x="{lx}" y="5" width="12" height="12" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{lx + 16}" y="16">{ch.value}</text>'
        )
        lx += 130

    for i, fs in enumerate(shown):
        src, tgt = fs.edge
        if node_names:
            src_n = node_names[src] if src < len(node_names) else str(src)
            tgt_n = node_names[tgt] if tgt < len(node_names) else str(tgt)
            label = f"{src_n}→{tgt_n}"
        else:
            label = f"{src}→{tgt}"

        y = i * bar_height + legend_h + 10
        parts.append(
            f'<text x="5" y="{y + 18}">{html.escape(label[:20])}</text>'
        )

        x_pos = label_w
        for ch, color in channel_colors.items():
            val = fs.channel_scores.get(ch, 0.0)
            w = max(0, int(val * bar_w))
            if w > 0:
                parts.append(
                    f'<rect x="{x_pos}" y="{y}" width="{w}" '
                    f'height="{bar_height - 6}" fill="{color}" opacity="0.8"/>'
                )
            x_pos += w

        parts.append(
            f'<text x="{x_pos + 5}" y="{y + 17}">'
            f'{fs.total_score:.3f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Convergence plot (text-based)
# ---------------------------------------------------------------------------


def radius_convergence_text(
    bounds_history: Sequence[tuple[int, int, float]],
) -> str:
    """Generate a text-based convergence plot for the radius solver.

    Parameters
    ----------
    bounds_history : Sequence[tuple[int, int, float]]
        List of (lower_bound, upper_bound, time_s) snapshots.

    Returns
    -------
    str
        Text representation.
    """
    if not bounds_history:
        return "No convergence data available."

    lines: list[str] = [
        "Radius Convergence",
        "=" * 50,
        f"{'Step':>5}  {'LB':>4}  {'UB':>4}  {'Gap':>8}  {'Time':>8}",
        "-" * 50,
    ]

    for i, (lb, ub, t) in enumerate(bounds_history, 1):
        gap = (ub - lb) / max(ub, 1) * 100
        lines.append(f"{i:>5}  {lb:>4}  {ub:>4}  {gap:>7.1f}%  {t:>7.2f}s")

    lb_final, ub_final, t_final = bounds_history[-1]
    if lb_final == ub_final:
        lines.append(f"\nConverged: radius = {lb_final} (exact)")
    else:
        lines.append(f"\nFinal: [{lb_final}, {ub_final}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison across methods
# ---------------------------------------------------------------------------


def method_comparison_table(
    results: dict[str, Any],
    node_names: list[str] | None = None,
    top_k: int = 5,
) -> str:
    """Generate a text comparison table across aggregation methods.

    Parameters
    ----------
    results : dict[str, list[FragilityScore]]
        Mapping from method name to scored edges.
    node_names : list[str] | None
        Node names.
    top_k : int
        Number of top edges per method.

    Returns
    -------
    str
    """
    methods = sorted(results.keys())
    if not methods:
        return "No comparison data."

    col_w = 22
    header = f"{'Rank':>4}"
    for m in methods:
        header += f"  {m[:col_w]:<{col_w}}"
    sep = "-" * len(header)

    lines = ["Method Comparison (Top Edges)", sep, header, sep]

    for rank in range(top_k):
        row = f"{rank + 1:>4}"
        for m in methods:
            scored = results.get(m, [])
            if rank < len(scored):
                fs = scored[rank]
                src, tgt = fs.edge
                if node_names:
                    sn = node_names[src] if src < len(node_names) else str(src)
                    tn = node_names[tgt] if tgt < len(node_names) else str(tgt)
                    label = f"{sn}→{tn}"
                else:
                    label = f"{src}→{tgt}"
                cell = f"{label[:14]} {fs.total_score:.3f}"
            else:
                cell = "-"
            row += f"  {cell:<{col_w}}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)
