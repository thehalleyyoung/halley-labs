"""
taintflow.report.html_report – Self-contained HTML report generation.

Produces a single HTML file with embedded CSS and inline SVG that
visualises the results of a TaintFlow leakage audit.  No external
dependencies (no Jinja2, no weasyprint).
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, IO, List, Optional, TextIO, Union

from taintflow.core.types import (
    Severity,
    FeatureLeakage,
    StageLeakage,
    LeakageReport,
)
from taintflow.core.config import TaintFlowConfig, SeverityThresholds


# ===================================================================
#  Constants
# ===================================================================

_TOOL_VERSION = "0.1.0"

_SEVERITY_COLORS: Dict[str, str] = {
    "negligible": "#22c55e",
    "warning": "#eab308",
    "critical": "#ef4444",
}

_SEVERITY_BG: Dict[str, str] = {
    "negligible": "#f0fdf4",
    "warning": "#fefce8",
    "critical": "#fef2f2",
}

_SEVERITY_THRESHOLDS_DOC = (
    "negligible &lt; 0.1 bits · warning 0.1 – 1.0 bits · critical &gt; 1.0 bits"
)


# ===================================================================
#  HTMLTemplate – lightweight template engine
# ===================================================================


class HTMLTemplate:
    """Minimal template engine backed by ``str.format_map``.

    Placeholders use ``{key}`` syntax.  Literal braces are doubled: ``{{``.
    """

    def __init__(self, source: str) -> None:
        self._source = source

    def render(self, context: Dict[str, Any]) -> str:
        """Render the template with *context* values."""
        safe: Dict[str, Any] = {k: v for k, v in context.items()}
        return self._source.format_map(_DefaultDict(safe))


class _DefaultDict(dict):
    """Return the key wrapped in braces when missing so partial renders work."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


# ===================================================================
#  CSS styles (embedded)
# ===================================================================

_CSS = """\
:root {
    --green: #22c55e; --green-bg: #f0fdf4;
    --yellow: #eab308; --yellow-bg: #fefce8;
    --red: #ef4444; --red-bg: #fef2f2;
    --gray-50: #f9fafb; --gray-100: #f3f4f6; --gray-200: #e5e7eb;
    --gray-300: #d1d5db; --gray-600: #4b5563; --gray-800: #1f2937;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
            Helvetica, Arial, sans-serif;
}
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: var(--font); color: var(--gray-800); line-height: 1.6;
    max-width: 1100px; margin: 0 auto; padding: 2rem 1rem;
    background: #fff;
}
h1, h2, h3 { margin-top: 2rem; }
h1 { font-size: 1.75rem; border-bottom: 2px solid var(--gray-200); padding-bottom: .5rem; }
h2 { font-size: 1.35rem; color: var(--gray-600); }
h3 { font-size: 1.1rem; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 9999px;
    font-size: .8rem; font-weight: 600; text-transform: uppercase;
}
.badge-negligible { background: var(--green-bg); color: #166534; }
.badge-warning    { background: var(--yellow-bg); color: #854d0e; }
.badge-critical   { background: var(--red-bg); color: #991b1b; }
.summary-card {
    background: var(--gray-50); border: 1px solid var(--gray-200);
    border-radius: 8px; padding: 1.25rem; margin: 1rem 0;
}
.summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
}
.metric { text-align: center; }
.metric-value { font-size: 1.5rem; font-weight: 700; }
.metric-label { font-size: .85rem; color: var(--gray-600); }
table {
    width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: .9rem;
}
th, td {
    text-align: left; padding: .6rem .75rem;
    border-bottom: 1px solid var(--gray-200);
}
th {
    background: var(--gray-100); font-weight: 600; cursor: pointer;
    user-select: none; position: sticky; top: 0;
}
th:hover { background: var(--gray-200); }
tr:hover td { background: var(--gray-50); }
.heatmap-cell {
    display: inline-block; width: 28px; height: 28px; border-radius: 4px;
    border: 1px solid var(--gray-200);
}
.toc { background: var(--gray-50); padding: 1rem 1.5rem; border-radius: 8px; }
.toc ul { list-style: none; padding-left: 1rem; }
.toc li { margin: .35rem 0; }
.remediation {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    padding: .75rem 1rem; margin: .5rem 0; border-radius: 0 6px 6px 0;
}
.config-block {
    background: var(--gray-50); padding: 1rem; border-radius: 6px;
    font-family: monospace; font-size: .85rem; overflow-x: auto;
    white-space: pre-wrap; word-break: break-word;
}
.dag-container { overflow-x: auto; margin: 1rem 0; }
footer {
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--gray-200);
    font-size: .8rem; color: var(--gray-600); text-align: center;
}
"""

# ===================================================================
#  Inline JavaScript for sortable tables
# ===================================================================

_JS_SORT = """\
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('th[data-sort]').forEach(function(th) {
        th.addEventListener('click', function() {
            var table = th.closest('table');
            var tbody = table.querySelector('tbody');
            var rows = Array.from(tbody.querySelectorAll('tr'));
            var idx = Array.from(th.parentNode.children).indexOf(th);
            var type = th.dataset.sort;
            var asc = th.dataset.dir !== 'asc';
            th.dataset.dir = asc ? 'asc' : 'desc';
            rows.sort(function(a, b) {
                var va = a.children[idx].textContent.trim();
                var vb = b.children[idx].textContent.trim();
                if (type === 'num') { va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }
                if (va < vb) return asc ? -1 : 1;
                if (va > vb) return asc ? 1 : -1;
                return 0;
            });
            rows.forEach(function(r) { tbody.appendChild(r); });
        });
    });
});
"""


# ===================================================================
#  SVG DAG rendering helpers
# ===================================================================

def _escape(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text), quote=True)


def _severity_css_class(severity: Severity) -> str:
    return f"badge-{severity.value}"


def _severity_color(severity: Severity) -> str:
    return _SEVERITY_COLORS.get(severity.value, "#6b7280")


def _severity_bg(severity: Severity) -> str:
    return _SEVERITY_BG.get(severity.value, "#f9fafb")


def _render_dag_svg(stages: List[StageLeakage], width: int = 900) -> str:
    """Render a simple box-and-arrow SVG of the pipeline stages.

    Each stage is drawn as a rectangle colour-coded by severity.  Arrows
    connect consecutive stages to convey the pipeline flow.
    """
    if not stages:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">' \
               '<text x="10" y="25" font-size="14" fill="#6b7280">No stages</text></svg>'

    box_w = 160
    box_h = 56
    h_gap = 40
    v_gap = 80
    padding = 20
    cols = max(1, (width - 2 * padding) // (box_w + h_gap))
    n = len(stages)
    rows = (n + cols - 1) // cols
    svg_w = min(n, cols) * (box_w + h_gap) - h_gap + 2 * padding
    svg_h = rows * (box_h + v_gap) - v_gap + 2 * padding

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">'
    ]

    positions: List[tuple] = []
    for i, stage in enumerate(stages):
        col = i % cols
        row = i // cols
        x = padding + col * (box_w + h_gap)
        y = padding + row * (box_h + v_gap)
        positions.append((x, y))

        fill = _severity_bg(stage.severity)
        stroke = _severity_color(stage.severity)
        name = _escape(stage.stage_name[:20])
        bits = f"{stage.max_bit_bound:.2f}b"

        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x + box_w // 2}" y="{y + 22}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#1f2937">{name}</text>'
        )
        parts.append(
            f'<text x="{x + box_w // 2}" y="{y + 40}" text-anchor="middle" '
            f'font-size="11" fill="{stroke}">{bits} – {_escape(stage.severity.value)}</text>'
        )

    # Draw arrows between consecutive stages
    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        same_row = y1 == y2
        if same_row:
            ax, ay = x1 + box_w, y1 + box_h // 2
            bx, by = x2, y2 + box_h // 2
        else:
            ax, ay = x1 + box_w // 2, y1 + box_h
            bx, by = x2 + box_w // 2, y2
        parts.append(
            f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" '
            f'stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )

    # Arrow marker definition
    parts.insert(1, (
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 Z" fill="#9ca3af"/>'
        '</marker></defs>'
    ))
    parts.append('</svg>')
    return '\n'.join(parts)


# ===================================================================
#  Heatmap rendering
# ===================================================================

def _render_heatmap_html(report: LeakageReport) -> str:
    """Render a colour-coded heatmap of per-feature leakage across stages."""
    all_features: Dict[str, Dict[str, FeatureLeakage]] = {}
    stage_ids: List[str] = []

    for sl in report.stage_leakages:
        stage_ids.append(sl.stage_id)
        for fl in sl.feature_leakages:
            all_features.setdefault(fl.column_name, {})[sl.stage_id] = fl

    if not all_features:
        return '<p>No feature leakage data available for heatmap.</p>'

    rows: List[str] = []
    rows.append('<div style="overflow-x:auto;"><table>')
    header = '<tr><th>Feature</th>'
    for sid in stage_ids:
        header += f'<th>{_escape(sid[:15])}</th>'
    header += '</tr>'
    rows.append(f'<thead>{header}</thead><tbody>')

    for feat_name in sorted(all_features):
        row = f'<tr><td><code>{_escape(feat_name)}</code></td>'
        for sid in stage_ids:
            fl = all_features[feat_name].get(sid)
            if fl is None:
                row += '<td><span class="heatmap-cell" style="background:#fff;" ' \
                       'title="N/A"></span></td>'
            else:
                color = _severity_color(fl.severity)
                bg = _severity_bg(fl.severity)
                title = f"{fl.bit_bound:.3f} bits ({fl.severity.value})"
                row += (
                    f'<td><span class="heatmap-cell" '
                    f'style="background:{bg};border-color:{color};" '
                    f'title="{_escape(title)}"></span> '
                    f'{fl.bit_bound:.2f}</td>'
                )
            row += ''
        row += '</tr>'
        rows.append(row)

    rows.append('</tbody></table></div>')
    return '\n'.join(rows)


# ===================================================================
#  HTMLReportGenerator
# ===================================================================


@dataclass
class HTMLReportGenerator:
    """Generate self-contained HTML audit reports.

    Parameters
    ----------
    config : TaintFlowConfig, optional
        Audit configuration; used for threshold display and config snapshot.
    title : str
        Page ``<title>`` override.
    include_toc : bool
        Whether to generate a table of contents.
    include_dag : bool
        Whether to render the pipeline DAG as inline SVG.
    include_heatmap : bool
        Whether to render the leakage heatmap.
    include_config : bool
        Whether to include a configuration details section.
    custom_css : str
        Extra CSS injected after the default stylesheet.
    """

    config: Optional[TaintFlowConfig] = None
    title: str = "TaintFlow Leakage Report"
    include_toc: bool = True
    include_dag: bool = True
    include_heatmap: bool = True
    include_config: bool = True
    custom_css: str = ""

    # -- thresholds (bits) ---------------------------------------------------
    _negligible_max: float = field(init=False, default=0.1)
    _warning_max: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        if self.config is not None:
            self._negligible_max = self.config.severity.negligible_max
            self._warning_max = self.config.severity.warning_max

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def generate(self, report: LeakageReport) -> str:
        """Return the full HTML report as a string."""
        sections: List[str] = []
        sections.append(self._render_header(report))
        if self.include_toc:
            sections.append(self._render_toc(report))
        sections.append(self._render_executive_summary(report))
        sections.append(self._render_feature_table(report))
        sections.append(self._render_stage_breakdown(report))
        if self.include_dag:
            sections.append(self._render_dag_section(report))
        if self.include_heatmap:
            sections.append(self._render_heatmap_section(report))
        sections.append(self._render_remediations(report))
        if self.include_config:
            sections.append(self._render_config_section(report))
        sections.append(self._render_footer())

        body = '\n'.join(sections)
        return self._wrap_page(body)

    def generate_to_file(self, report: LeakageReport, path: str) -> None:
        """Write the HTML report to *path*."""
        content = self.generate(report)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def generate_to_stream(self, report: LeakageReport, stream: IO[str]) -> None:
        """Write the HTML report to an open text stream."""
        stream.write(self.generate(report))

    # -----------------------------------------------------------------
    #  Page wrapper
    # -----------------------------------------------------------------

    def _wrap_page(self, body_html: str) -> str:
        css = _CSS
        if self.custom_css:
            css += "\n" + self.custom_css
        return (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n<head>\n'
            '<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{_escape(self.title)}</title>\n'
            f'<style>\n{css}\n</style>\n'
            f'<script>\n{_JS_SORT}\n</script>\n'
            '</head>\n<body>\n'
            f'{body_html}\n'
            '</body>\n</html>\n'
        )

    # -----------------------------------------------------------------
    #  Section renderers
    # -----------------------------------------------------------------

    def _render_header(self, report: LeakageReport) -> str:
        sev = report.overall_severity
        badge = f'<span class="badge {_severity_css_class(sev)}">{_escape(sev.value)}</span>'
        return (
            f'<h1 id="top">&#x1F50D; {_escape(self.title)} {badge}</h1>\n'
            f'<p>Pipeline: <strong>{_escape(report.pipeline_name)}</strong> &middot; '
            f'Generated: {_escape(report.timestamp)}</p>'
        )

    def _render_toc(self, report: LeakageReport) -> str:
        items = [
            ("summary", "Executive Summary"),
            ("features", "Per-Feature Leakage"),
            ("stages", "Per-Stage Breakdown"),
        ]
        if self.include_dag:
            items.append(("dag", "Pipeline DAG"))
        if self.include_heatmap:
            items.append(("heatmap", "Leakage Heatmap"))
        items.append(("remediations", "Remediation Suggestions"))
        if self.include_config:
            items.append(("config", "Configuration"))
        links = ''.join(f'<li><a href="#{aid}">{label}</a></li>\n' for aid, label in items)
        return f'<nav class="toc"><strong>Contents</strong>\n<ul>\n{links}</ul></nav>'

    def _render_executive_summary(self, report: LeakageReport) -> str:
        sev = report.overall_severity
        badge = f'<span class="badge {_severity_css_class(sev)}">{_escape(sev.value)}</span>'

        # Top bottleneck stages (up to 3)
        bottlenecks = report.stages_by_severity()[:3]
        bn_items = ""
        for sl in bottlenecks:
            bn_items += (
                f'<li><strong>{_escape(sl.stage_name)}</strong>: '
                f'{sl.max_bit_bound:.2f} bits '
                f'<span class="badge {_severity_css_class(sl.severity)}">'
                f'{_escape(sl.severity.value)}</span></li>\n'
            )
        bn_section = ""
        if bn_items:
            bn_section = f'<h3>Top Bottleneck Stages</h3>\n<ol>\n{bn_items}</ol>'

        duration_str = ""
        if report.analysis_duration_ms > 0:
            secs = report.analysis_duration_ms / 1000.0
            duration_str = f'<div class="metric"><div class="metric-value">{secs:.1f}s</div>' \
                           '<div class="metric-label">Analysis time</div></div>'

        return (
            f'<h2 id="summary">Executive Summary</h2>\n'
            f'<div class="summary-card">\n'
            f'<div class="summary-grid">\n'
            f'  <div class="metric">'
            f'<div class="metric-value">{report.total_bit_bound:.2f}</div>'
            f'<div class="metric-label">Total leakage (bits)</div></div>\n'
            f'  <div class="metric">'
            f'<div class="metric-value">{badge}</div>'
            f'<div class="metric-label">Overall severity</div></div>\n'
            f'  <div class="metric">'
            f'<div class="metric-value">{report.n_leaking_features}/{report.n_features}</div>'
            f'<div class="metric-label">Leaking features</div></div>\n'
            f'  <div class="metric">'
            f'<div class="metric-value">{report.n_stages}</div>'
            f'<div class="metric-label">Pipeline stages</div></div>\n'
            f'{duration_str}'
            f'</div>\n'
            f'{bn_section}\n'
            f'<p style="font-size:.8rem;color:#6b7280;">'
            f'Thresholds: {_SEVERITY_THRESHOLDS_DOC}</p>\n'
            f'</div>'
        )

    def _render_feature_table(self, report: LeakageReport) -> str:
        rows: List[str] = []
        all_features: List[tuple] = []
        for sl in report.stage_leakages:
            for fl in sl.feature_leakages:
                all_features.append((sl.stage_name, fl))

        all_features.sort(key=lambda x: (-x[1].bit_bound, x[1].column_name))

        for stage_name, fl in all_features:
            sev = fl.severity
            badge = f'<span class="badge {_severity_css_class(sev)}">{_escape(sev.value)}</span>'
            origins = ", ".join(sorted(o.value for o in fl.origins)) if fl.origins else "—"
            rows.append(
                f'<tr>'
                f'<td><code>{_escape(fl.column_name)}</code></td>'
                f'<td>{fl.bit_bound:.4f}</td>'
                f'<td>{badge}</td>'
                f'<td>{_escape(stage_name)}</td>'
                f'<td>{_escape(origins)}</td>'
                f'<td>{fl.confidence:.0%}</td>'
                f'</tr>\n'
            )

        tbody = ''.join(rows) if rows else '<tr><td colspan="6">No features analysed.</td></tr>'
        return (
            f'<h2 id="features">Per-Feature Leakage</h2>\n'
            f'<table>\n<thead><tr>'
            f'<th data-sort="str">Feature</th>'
            f'<th data-sort="num">Bits</th>'
            f'<th data-sort="str">Severity</th>'
            f'<th data-sort="str">Stage</th>'
            f'<th data-sort="str">Origins</th>'
            f'<th data-sort="num">Confidence</th>'
            f'</tr></thead>\n<tbody>\n{tbody}</tbody>\n</table>'
        )

    def _render_stage_breakdown(self, report: LeakageReport) -> str:
        parts: List[str] = [f'<h2 id="stages">Per-Stage Breakdown</h2>']
        stages = report.stages_by_severity()
        if not stages:
            parts.append('<p>No stage information available.</p>')
            return '\n'.join(parts)

        parts.append(
            '<table>\n<thead><tr>'
            '<th data-sort="str">Stage</th>'
            '<th data-sort="str">Type</th>'
            '<th data-sort="num">Max bits</th>'
            '<th data-sort="num">Mean bits</th>'
            '<th data-sort="num">Leaking features</th>'
            '<th data-sort="str">Severity</th>'
            '</tr></thead>\n<tbody>'
        )
        for sl in stages:
            sev_badge = (
                f'<span class="badge {_severity_css_class(sl.severity)}">'
                f'{_escape(sl.severity.value)}</span>'
            )
            parts.append(
                f'<tr>'
                f'<td><strong>{_escape(sl.stage_name)}</strong><br>'
                f'<small style="color:#6b7280">{_escape(sl.stage_id)}</small></td>'
                f'<td>{_escape(sl.op_type.value)}</td>'
                f'<td>{sl.max_bit_bound:.4f}</td>'
                f'<td>{sl.mean_bit_bound:.4f}</td>'
                f'<td>{sl.n_leaking_features}</td>'
                f'<td>{sev_badge}</td>'
                f'</tr>'
            )
        parts.append('</tbody>\n</table>')
        return '\n'.join(parts)

    def _render_dag_section(self, report: LeakageReport) -> str:
        svg = _render_dag_svg(report.stage_leakages)
        return (
            f'<h2 id="dag">Pipeline DAG</h2>\n'
            f'<div class="dag-container">\n{svg}\n</div>'
        )

    def _render_heatmap_section(self, report: LeakageReport) -> str:
        heatmap = _render_heatmap_html(report)
        return (
            f'<h2 id="heatmap">Leakage Heatmap</h2>\n'
            f'<p>Colour intensity indicates leakage severity per feature per stage.</p>\n'
            f'{heatmap}'
        )

    def _render_remediations(self, report: LeakageReport) -> str:
        parts: List[str] = [f'<h2 id="remediations">Remediation Suggestions</h2>']
        seen: Dict[str, bool] = {}
        has_any = False

        for sl in report.stages_by_severity():
            for fl in sl.feature_leakages:
                if fl.remediation and fl.remediation not in seen:
                    seen[fl.remediation] = True
                    has_any = True
                    sev_badge = (
                        f'<span class="badge {_severity_css_class(fl.severity)}">'
                        f'{_escape(fl.severity.value)}</span>'
                    )
                    parts.append(
                        f'<div class="remediation">'
                        f'<strong>{_escape(fl.column_name)}</strong> '
                        f'in <em>{_escape(sl.stage_name)}</em> {sev_badge}<br>'
                        f'{_escape(fl.remediation)}'
                        f'</div>'
                    )

        if not has_any:
            parts.append(
                '<p>No specific remediation suggestions. '
                'The pipeline appears clean or remediations were not computed.</p>'
            )
        return '\n'.join(parts)

    def _render_config_section(self, report: LeakageReport) -> str:
        parts: List[str] = [f'<h2 id="config">Configuration Details</h2>']
        snapshot = report.config_snapshot
        if self.config is not None:
            snapshot = self.config.to_dict() if hasattr(self.config, "to_dict") else snapshot

        if not snapshot:
            parts.append('<p>No configuration snapshot available.</p>')
        else:
            import json
            formatted = json.dumps(snapshot, indent=2, default=str, sort_keys=True)
            parts.append(f'<pre class="config-block">{_escape(formatted)}</pre>')

        return '\n'.join(parts)

    def _render_footer(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f'<footer>'
            f'Generated by <strong>TaintFlow</strong> v{_TOOL_VERSION} on {now}. '
            f'<a href="#top">Back to top ↑</a>'
            f'</footer>'
        )


# ===================================================================
#  Convenience function
# ===================================================================


def generate_html_report(
    report: LeakageReport,
    config: Optional[TaintFlowConfig] = None,
    path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """One-shot helper: build an HTML report and optionally write to *path*.

    Parameters
    ----------
    report : LeakageReport
        The audit result to render.
    config : TaintFlowConfig, optional
        Audit configuration for threshold display.
    path : str, optional
        If given, write the report to this file path.
    **kwargs
        Forwarded to :class:`HTMLReportGenerator`.

    Returns
    -------
    str
        The full HTML document.
    """
    gen = HTMLReportGenerator(config=config, **kwargs)
    content = gen.generate(report)
    if path is not None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    return content
