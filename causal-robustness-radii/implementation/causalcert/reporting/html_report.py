"""
HTML report generation using Jinja2 templates.

Produces a self-contained HTML document with the audit results,
fragility heatmap, and DAG visualisation placeholder.  Falls back to
simple string formatting if Jinja2 is not installed.
"""

from __future__ import annotations

import datetime
import html
import logging
from pathlib import Path
from typing import Any

from causalcert.types import (
    AuditReport,
    FragilityChannel,
    FragilityScore,
    StructuralEdit,
)

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_html_report(
    report: AuditReport,
    output_path: str | Path | None = None,
    node_names: list[str] | None = None,
    template_name: str = "audit_report.html.j2",
) -> str:
    """Render an audit report to HTML.

    Parameters
    ----------
    report : AuditReport
        Audit report.
    output_path : str | Path | None
        If given, write the HTML to this file.
    node_names : list[str] | None
        Node names for display.
    template_name : str
        Jinja2 template filename.

    Returns
    -------
    str
        HTML string.
    """
    context = _build_template_context(report, node_names)

    try:
        html_str = _render_with_jinja(template_name, context)
    except (ImportError, FileNotFoundError):
        logger.info("Jinja2 or template unavailable, using fallback renderer")
        html_str = _render_fallback(context)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_str, encoding="utf-8")

    return html_str


def _load_template(name: str = "audit_report.html.j2") -> str:
    """Load a Jinja2 template from the templates directory.

    Parameters
    ----------
    name : str
        Template file name.

    Returns
    -------
    str
        Template source.
    """
    template_path = _TEMPLATE_DIR / name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text()


# ---------------------------------------------------------------------------
# Template context builder
# ---------------------------------------------------------------------------


def _build_template_context(
    report: AuditReport,
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the Jinja2 template context from an AuditReport."""
    tn = node_names[report.treatment] if node_names and report.treatment < len(node_names) else str(report.treatment)
    on = node_names[report.outcome] if node_names and report.outcome < len(node_names) else str(report.outcome)

    # Build fragility ranking data
    fragility_ranking = []
    for fs in report.fragility_ranking:
        src, tgt = fs.edge
        if node_names:
            src_name = node_names[src] if src < len(node_names) else str(src)
            tgt_name = node_names[tgt] if tgt < len(node_names) else str(tgt)
            edge_label = f"{src_name} → {tgt_name}"
        else:
            edge_label = f"{src} → {tgt}"

        fragility_ranking.append({
            "edge": edge_label,
            "total_score": fs.total_score,
            "channel_scores": {
                ch.value: v for ch, v in fs.channel_scores.items()
            },
        })

    # Witness edits
    witness_edits = []
    for edit in report.radius.witness_edits:
        src_name = node_names[edit.source] if node_names and edit.source < len(node_names) else str(edit.source)
        tgt_name = node_names[edit.target] if node_names and edit.target < len(node_names) else str(edit.target)
        witness_edits.append({
            "edit_type": edit.edit_type.value,
            "source": src_name,
            "target": tgt_name,
        })

    return {
        "treatment_name": tn,
        "outcome_name": on,
        "n_nodes": report.n_nodes,
        "n_edges": report.n_edges,
        "radius": report.radius,
        "witness_edits": witness_edits,
        "fragility_ranking": fragility_ranking,
        "baseline_estimate": report.baseline_estimate,
        "perturbed_estimates": report.perturbed_estimates,
        "version": report.metadata.get("version", "0.1.0"),
        "seed": report.metadata.get("seed", "N/A"),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "node_names": node_names,
    }


# ---------------------------------------------------------------------------
# Jinja2 renderer
# ---------------------------------------------------------------------------


def _render_with_jinja(template_name: str, context: dict[str, Any]) -> str:
    """Render using Jinja2."""
    try:
        import jinja2
    except ImportError:
        raise ImportError("Jinja2 is required for HTML reports")

    template_src = _load_template(template_name)
    env = jinja2.Environment(
        autoescape=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.from_string(template_src)
    return template.render(**context)


# ---------------------------------------------------------------------------
# Fallback renderer (no Jinja2 dependency)
# ---------------------------------------------------------------------------


def _render_fallback(context: dict[str, Any]) -> str:
    """Render HTML without Jinja2 using string formatting."""
    parts: list[str] = []

    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('  <meta charset="UTF-8">')
    parts.append('  <meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append("  <title>CausalCert Audit Report</title>")
    parts.append("  <style>")
    parts.append(_get_css())
    parts.append("  </style>")
    parts.append("</head>")
    parts.append("<body>")

    # Header
    parts.append("  <h1>🔒 CausalCert Structural Robustness Audit</h1>")

    # Summary
    parts.append("  <h2>Summary</h2>")
    tn = html.escape(str(context["treatment_name"]))
    on = html.escape(str(context["outcome_name"]))
    parts.append(f"  <p>Treatment: <strong>{tn}</strong> → "
                 f"Outcome: <strong>{on}</strong></p>")
    parts.append(f"  <p>DAG: {context['n_nodes']} nodes, "
                 f"{context['n_edges']} edges</p>")

    # Radius
    parts.append("  <h2>Robustness Radius</h2>")
    radius = context["radius"]
    lb = radius.lower_bound
    ub = radius.upper_bound
    certified = radius.certified
    gap = radius.gap
    cert_badge = ('<span class="badge badge-certified">CERTIFIED</span>'
                  if certified
                  else f'<span class="badge badge-gap">Gap: {gap * 100:.1f}%</span>')
    parts.append(f"  <p>Lower bound: <strong>{lb}</strong> | "
                 f"Upper bound: <strong>{ub}</strong> {cert_badge}</p>")
    parts.append(f"  <p>Solver: {radius.solver_strategy.value} | "
                 f"Time: {radius.solver_time_s:.2f}s</p>")

    # Witness edits
    witness = context.get("witness_edits", [])
    if witness:
        parts.append("  <h2>Witness Edit Set</h2>")
        parts.append("  <table>")
        parts.append("    <tr><th>Edit</th><th>Source</th><th>Target</th></tr>")
        for w in witness:
            parts.append(f"    <tr><td>{html.escape(w['edit_type'])}</td>"
                         f"<td>{html.escape(str(w['source']))}</td>"
                         f"<td>{html.escape(str(w['target']))}</td></tr>")
        parts.append("  </table>")

    # Fragility ranking
    ranking = context.get("fragility_ranking", [])
    if ranking:
        parts.append(f"  <h2>Fragility Ranking (Top {len(ranking)})</h2>")
        parts.append("  <table>")
        parts.append("    <tr><th>Rank</th><th>Edge</th><th>Score</th>"
                     "<th>d-Sep</th><th>ID</th><th>Est</th></tr>")
        for i, fs in enumerate(ranking, 1):
            score = fs["total_score"]
            cls = "fragile" if score > 0.7 else ("robust" if score < 0.3 else "")
            ch = fs.get("channel_scores", {})
            dsep = ch.get("d_separation", 0.0)
            id_s = ch.get("identification", 0.0)
            est_s = ch.get("estimation", 0.0)
            parts.append(
                f'    <tr><td>{i}</td>'
                f'<td>{html.escape(fs["edge"])}</td>'
                f'<td class="{cls}">{score:.3f}</td>'
                f'<td>{dsep:.3f}</td>'
                f'<td>{id_s:.3f}</td>'
                f'<td>{est_s:.3f}</td></tr>'
            )
        parts.append("  </table>")

    # Estimation
    baseline = context.get("baseline_estimate")
    if baseline is not None:
        parts.append("  <h2>Baseline Estimation</h2>")
        parts.append(f"  <p>ATE: <strong>{baseline.ate:.4f}</strong> "
                     f"(SE: {baseline.se:.4f})</p>")
        parts.append(f"  <p>95% CI: [{baseline.ci_lower:.4f}, "
                     f"{baseline.ci_upper:.4f}]</p>")
        parts.append(f"  <p>Method: {baseline.method} | "
                     f"N: {baseline.n_obs}</p>")

    perturbed = context.get("perturbed_estimates", [])
    if perturbed:
        parts.append("  <h2>Perturbed Estimates</h2>")
        parts.append("  <table>")
        parts.append("    <tr><th>#</th><th>ATE</th><th>SE</th>"
                     "<th>CI Lower</th><th>CI Upper</th><th>Method</th></tr>")
        for i, pe in enumerate(perturbed, 1):
            parts.append(
                f"    <tr><td>{i}</td><td>{pe.ate:.4f}</td>"
                f"<td>{pe.se:.4f}</td><td>{pe.ci_lower:.4f}</td>"
                f"<td>{pe.ci_upper:.4f}</td><td>{pe.method}</td></tr>"
            )
        parts.append("  </table>")

    # Metadata
    parts.append("  <h2>Metadata</h2>")
    parts.append(f'  <p class="meta">Generated by CausalCert '
                 f'v{context["version"]} | Seed: {context["seed"]} | '
                 f'{context["timestamp"]}</p>')

    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def _get_css() -> str:
    """Return CSS for the HTML report."""
    return """
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 2em;
        color: #1a1a2e;
        line-height: 1.6;
    }
    h1 { color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 0.3em; }
    h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 0.3em; margin-top: 1.5em; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.95em; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
    th { background-color: #0f3460; color: white; }
    tr:nth-child(even) { background-color: #f8f9fa; }
    tr:hover { background-color: #e9ecef; }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .badge-certified { background: #28a745; color: white; }
    .badge-gap { background: #ffc107; color: #333; }
    .fragile { color: #dc3545; font-weight: bold; }
    .robust { color: #28a745; font-weight: bold; }
    .meta { color: #666; font-size: 0.9em; margin-top: 2em; }
    p { margin: 0.5em 0; }
    strong { color: #16213e; }
    @media (max-width: 600px) {
        body { padding: 1em; }
        table { font-size: 0.85em; }
        th, td { padding: 4px 6px; }
    }
    """


# ---------------------------------------------------------------------------
# Heatmap SVG helper
# ---------------------------------------------------------------------------


def fragility_heatmap_svg(
    scores: list[FragilityScore],
    node_names: list[str] | None = None,
    max_edges: int = 15,
    width: int = 600,
    bar_height: int = 25,
) -> str:
    """Generate an SVG fragility score heatmap (horizontal bars).

    Parameters
    ----------
    scores : list[FragilityScore]
        Scored edges (should be pre-sorted).
    node_names : list[str] | None
        Node names.
    max_edges : int
        Maximum number of edges to show.
    width : int
        SVG width in pixels.
    bar_height : int
        Height of each bar.

    Returns
    -------
    str
        SVG string.
    """
    shown = scores[:max_edges]
    if not shown:
        return '<svg width="100" height="30"><text x="5" y="20">No data</text></svg>'

    label_width = 180
    bar_width = width - label_width - 20
    total_height = len(shown) * bar_height + 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{total_height}" '
        f'viewBox="0 0 {width} {total_height}">',
        '<style>',
        '  text { font-family: monospace; font-size: 12px; }',
        '  .label { fill: #333; }',
        '  .value { fill: white; font-weight: bold; font-size: 11px; }',
        '</style>',
    ]

    for i, fs in enumerate(shown):
        src, tgt = fs.edge
        if node_names:
            src_name = node_names[src] if src < len(node_names) else str(src)
            tgt_name = node_names[tgt] if tgt < len(node_names) else str(tgt)
            label = f"{src_name}→{tgt_name}"
        else:
            label = f"{src}→{tgt}"

        y = i * bar_height + 20
        w = max(1, int(fs.total_score * bar_width))

        # Color: green (low) -> yellow -> red (high)
        r = min(255, int(fs.total_score * 2 * 255))
        g = min(255, int((1 - fs.total_score) * 2 * 255))
        color = f"rgb({r},{g},0)"

        label_escaped = html.escape(label[:22])
        parts.append(
            f'  <text x="5" y="{y + 16}" class="label">{label_escaped}</text>'
        )
        parts.append(
            f'  <rect x="{label_width}" y="{y}" width="{w}" '
            f'height="{bar_height - 4}" fill="{color}" rx="2"/>'
        )
        parts.append(
            f'  <text x="{label_width + w + 5}" y="{y + 16}" '
            f'class="label">{fs.total_score:.3f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)
