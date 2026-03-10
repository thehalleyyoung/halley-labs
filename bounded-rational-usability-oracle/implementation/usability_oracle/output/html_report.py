"""
usability_oracle.output.html_report — Self-contained HTML report generator.

Produces a single HTML file with embedded CSS and JavaScript that
visualises the usability-oracle pipeline result.  No external
dependencies are required to view the report.
"""

from __future__ import annotations

import html as html_mod
import math
from datetime import datetime, timezone
from typing import Any

from usability_oracle.core.enums import Severity
from usability_oracle.output.models import (
    BottleneckDescription,
    CostComparison,
    PipelineResult,
    StageTimingInfo,
)


def _esc(text: str) -> str:
    return html_mod.escape(str(text))


def _pct(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return "N/A"
    return f"{value:+.1f}%"


def _secs(value: float) -> str:
    if value < 1.0:
        return f"{value * 1000:.1f} ms"
    return f"{value:.3f} s"


_SEVERITY_COLORS: dict[Severity, str] = {
    Severity.CRITICAL: "#dc3545",
    Severity.HIGH: "#fd7e14",
    Severity.MEDIUM: "#ffc107",
    Severity.LOW: "#17a2b8",
    Severity.INFO: "#6c757d",
}

_VERDICT_COLORS: dict[str, str] = {
    "regression": "#dc3545",
    "improvement": "#28a745",
    "neutral": "#6c757d",
    "inconclusive": "#ffc107",
}


class HTMLReportGenerator:
    """Generate a self-contained HTML report from a :class:`PipelineResult`."""

    def __init__(self, title: str = "Usability Oracle Report") -> None:
        self._title = title

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, result: PipelineResult) -> str:
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f"<title>{_esc(self._title)}</title>",
            "<style>",
            self._css(),
            "</style>",
            "</head>",
            "<body>",
            self._header(),
            '<div class="container">',
            self._summary_section(result),
            self._cost_comparison_section(result.comparison),
            self._bottleneck_section(result.bottlenecks),
            self._recommendations_section(result.recommendations),
            self._timeline_section(result.timing),
            self._annotations_section(result.annotated_elements),
            "</div>",
            "<script>",
            self._javascript(),
            "</script>",
            "</body>",
            "</html>",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header(self) -> str:
        return (
            '<header class="header">'
            f'<h1>{_esc(self._title)}</h1>'
            "<p>Bounded-Rational Usability Oracle &mdash; automated usability regression analysis</p>"
            "</header>"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _summary_section(self, result: PipelineResult) -> str:
        verdict = result.verdict.value
        color = _VERDICT_COLORS.get(verdict, "#6c757d")
        ts = datetime.fromtimestamp(result.timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        n_bottlenecks = len(result.bottlenecks)
        n_critical = len(result.critical_bottlenecks)
        total = _secs(result.total_time)

        return (
            '<section class="section">'
            "<h2>Summary</h2>"
            '<div class="summary-grid">'
            f'<div class="summary-card" style="border-left:4px solid {color}">'
            f'<span class="label">Verdict</span>'
            f'<span class="value" style="color:{color}">{_esc(verdict.upper())}</span>'
            "</div>"
            '<div class="summary-card">'
            '<span class="label">Bottlenecks</span>'
            f'<span class="value">{n_bottlenecks} ({n_critical} critical)</span>'
            "</div>"
            '<div class="summary-card">'
            '<span class="label">Total Time</span>'
            f'<span class="value">{total}</span>'
            "</div>"
            '<div class="summary-card">'
            '<span class="label">Timestamp</span>'
            f'<span class="value">{ts}</span>'
            "</div>"
            "</div>"
            "</section>"
        )

    # ------------------------------------------------------------------
    # Cost comparison
    # ------------------------------------------------------------------

    def _cost_comparison_section(self, comparison: CostComparison | None) -> str:
        if comparison is None:
            return ""
        rows = ""
        cost_a_d = comparison.cost_a.to_dict()
        cost_b_d = comparison.cost_b.to_dict()
        delta_d = comparison.delta.to_dict()
        for channel in cost_a_d:
            va = cost_a_d[channel]
            vb = cost_b_d.get(channel, 0)
            vd = delta_d.get(channel, 0)
            cls = "delta-pos" if vd > 0 else "delta-neg" if vd < 0 else ""
            rows += (
                f"<tr><td>{_esc(str(channel).title())}</td>"
                f"<td>{va:.4f}</td><td>{vb:.4f}</td>"
                f'<td class="{cls}">{vd:+.4f}</td></tr>'
            )
        # totals
        ta = comparison.cost_a.total_weighted_cost
        tb = comparison.cost_b.total_weighted_cost
        td = comparison.delta.total_weighted_cost
        cls = "delta-pos" if td > 0 else "delta-neg" if td < 0 else ""
        rows += (
            f'<tr class="total-row"><td><strong>Total (weighted)</strong></td>'
            f"<td><strong>{ta:.4f}</strong></td><td><strong>{tb:.4f}</strong></td>"
            f'<td class="{cls}"><strong>{td:+.4f}</strong> ({_pct(comparison.percentage_change)})</td></tr>'
        )
        return (
            '<section class="section collapsible" data-title="Cost Comparison">'
            '<h2 class="section-toggle">Cost Comparison <span class="toggle-icon">▼</span></h2>'
            '<div class="section-body">'
            '<table class="cost-table">'
            "<thead><tr><th>Channel</th><th>Before</th><th>After</th><th>Delta</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
            "</div></section>"
        )

    # ------------------------------------------------------------------
    # Bottlenecks
    # ------------------------------------------------------------------

    def _bottleneck_section(self, bottlenecks: list[BottleneckDescription]) -> str:
        if not bottlenecks:
            return (
                '<section class="section"><h2>Bottlenecks</h2>'
                "<p>No usability bottlenecks detected.</p></section>"
            )
        cards = ""
        for b in bottlenecks:
            color = _SEVERITY_COLORS.get(b.severity, "#6c757d")
            elems = ", ".join(b.affected_elements[:6]) if b.affected_elements else "—"
            rec = f'<p class="rec">{_esc(b.recommendation)}</p>' if b.recommendation else ""
            cards += (
                f'<div class="bottleneck-card" style="border-left:4px solid {color}">'
                f'<div class="bn-header">'
                f'<span class="bn-type">{_esc(b.bottleneck_type.value)}</span>'
                f'<span class="bn-severity" style="background:{color}">{_esc(b.severity.value)}</span>'
                "</div>"
                f'<p class="bn-desc">{_esc(b.description)}</p>'
                f'<p class="bn-impact">Cost impact: {b.cost_impact:+.4f}</p>'
                f'<p class="bn-elements">Affected: {_esc(elems)}</p>'
                f"{rec}"
                "</div>"
            )
        return (
            '<section class="section collapsible" data-title="Bottlenecks">'
            '<h2 class="section-toggle">Bottlenecks <span class="toggle-icon">▼</span></h2>'
            f'<div class="section-body bottleneck-grid">{cards}</div>'
            "</section>"
        )

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _recommendations_section(self, recommendations: list[str]) -> str:
        if not recommendations:
            return ""
        items = "".join(f"<li>{_esc(r)}</li>" for r in recommendations)
        return (
            '<section class="section collapsible" data-title="Recommendations">'
            '<h2 class="section-toggle">Recommendations <span class="toggle-icon">▼</span></h2>'
            f'<div class="section-body"><ul class="rec-list">{items}</ul></div>'
            "</section>"
        )

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _timeline_section(self, timing: list[StageTimingInfo]) -> str:
        if not timing:
            return ""
        total = sum(t.elapsed_seconds for t in timing) or 1e-9
        bars = ""
        for t in timing:
            pct = (t.elapsed_seconds / total) * 100
            bars += (
                f'<div class="timeline-bar" style="width:{max(pct, 1):.1f}%" '
                f'title="{_esc(t.stage.value)}: {_secs(t.elapsed_seconds)}">'
                f'<span class="tl-label">{_esc(t.stage.value)}</span>'
                "</div>"
            )
        rows = "".join(
            f"<tr><td>{_esc(t.stage.value)}</td><td>{_secs(t.elapsed_seconds)}</td>"
            f"<td>{(t.elapsed_seconds / total) * 100:.1f}%</td></tr>"
            for t in timing
        )
        return (
            '<section class="section collapsible" data-title="Pipeline Timing">'
            '<h2 class="section-toggle">Pipeline Timing <span class="toggle-icon">▼</span></h2>'
            '<div class="section-body">'
            f'<div class="timeline-track">{bars}</div>'
            '<table class="timing-table"><thead>'
            "<tr><th>Stage</th><th>Elapsed</th><th>% of Total</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>"
            "</div></section>"
        )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    @staticmethod
    def _annotations_section(elements: list[Any]) -> str:
        if not elements:
            return ""
        rows = ""
        for ae in elements:
            color = _SEVERITY_COLORS.get(ae.severity, "#6c757d")
            rows += (
                f"<tr><td>{_esc(ae.element_id)}</td>"
                f'<td style="color:{color}">{_esc(ae.severity.value)}</td>'
                f"<td>{_esc(ae.annotation)}</td>"
                f"<td>{_esc(ae.location or '—')}</td></tr>"
            )
        return (
            '<section class="section collapsible" data-title="Annotated Elements">'
            '<h2 class="section-toggle">Annotated Elements <span class="toggle-icon">▼</span></h2>'
            '<div class="section-body">'
            '<table class="ann-table"><thead>'
            "<tr><th>Element</th><th>Severity</th><th>Annotation</th><th>Location</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>"
            "</div></section>"
        )

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    @staticmethod
    def _css() -> str:
        return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f4f6f9; color: #212529; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #1a1a2e, #16213e); color: #fff;
                  padding: 2rem; text-align: center; }
        .header h1 { margin-bottom: 0.25rem; }
        .container { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
        .section { background: #fff; border-radius: 8px; padding: 1.5rem;
                   margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
        .section-toggle { cursor: pointer; user-select: none; }
        .toggle-icon { font-size: 0.8em; transition: transform 0.2s; display: inline-block; }
        .section.collapsed .toggle-icon { transform: rotate(-90deg); }
        .section.collapsed .section-body { display: none; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 1rem; margin-top: 1rem; }
        .summary-card { padding: 1rem; border-radius: 6px; background: #f8f9fa; }
        .summary-card .label { display: block; font-size: 0.85rem; color: #6c757d; }
        .summary-card .value { display: block; font-size: 1.3rem; font-weight: 600; margin-top: 0.25rem; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #dee2e6; }
        th { background: #f8f9fa; font-weight: 600; }
        .total-row td { border-top: 2px solid #212529; }
        .delta-pos { color: #dc3545; }
        .delta-neg { color: #28a745; }
        .bottleneck-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                           gap: 1rem; }
        .bottleneck-card { background: #f8f9fa; border-radius: 6px; padding: 1rem; }
        .bn-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
        .bn-type { font-weight: 600; }
        .bn-severity { color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
        .bn-desc { margin-bottom: 0.4rem; }
        .bn-impact { font-size: 0.9rem; color: #495057; }
        .bn-elements { font-size: 0.85rem; color: #6c757d; }
        .rec { margin-top: 0.5rem; padding: 0.5rem; background: #e8f5e9; border-radius: 4px; font-size: 0.9rem; }
        .rec-list { padding-left: 1.5rem; }
        .rec-list li { margin-bottom: 0.4rem; }
        .timeline-track { display: flex; height: 32px; border-radius: 6px; overflow: hidden;
                          margin-top: 1rem; background: #e9ecef; }
        .timeline-bar { display: flex; align-items: center; justify-content: center;
                        background: #4361ee; color: #fff; font-size: 0.7rem; min-width: 24px;
                        border-right: 1px solid rgba(255,255,255,.3); }
        .timeline-bar:nth-child(2n) { background: #3a0ca3; }
        .tl-label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 0 4px; }
        """

    # ------------------------------------------------------------------
    # JavaScript
    # ------------------------------------------------------------------

    @staticmethod
    def _javascript() -> str:
        return """
        document.addEventListener('DOMContentLoaded', function() {
            // Collapsible sections
            document.querySelectorAll('.section-toggle').forEach(function(el) {
                el.addEventListener('click', function() {
                    var section = el.closest('.section');
                    if (section) {
                        section.classList.toggle('collapsed');
                    }
                });
            });

            // Add colour to timeline bars based on percentage
            var bars = document.querySelectorAll('.timeline-bar');
            var palette = ['#4361ee','#3a0ca3','#7209b7','#f72585','#4cc9f0','#4895ef','#560bad'];
            bars.forEach(function(bar, i) {
                bar.style.background = palette[i % palette.length];
            });
        });
        """
