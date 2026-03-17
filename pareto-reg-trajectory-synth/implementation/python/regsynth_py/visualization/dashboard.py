"""HTML dashboard combining all RegSynth visualizations.

Generates a self-contained HTML page with embedded CSS, minimal JS, and inline
SVG charts — no external dependencies required.
"""

from __future__ import annotations

from typing import Any, Optional

from regsynth_py.visualization.conflict_graph import ConflictGraphPlotter
from regsynth_py.visualization.pareto_plot import ParetoPlotter
from regsynth_py.visualization.timeline_plot import TimelinePlotter


class DashboardGenerator:
    """Build a complete HTML compliance dashboard."""

    def __init__(self, title: str = "RegSynth Compliance Dashboard") -> None:
        self.title = title

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> str:
        """Return a full HTML page string.

        *data* may contain keys:
        ``summary``, ``pareto``, ``timeline``, ``conflicts``, ``coverage``,
        ``recommendations``.
        """
        sections = [
            self._generate_summary_section(data.get("summary", {})),
            self._generate_pareto_section(data.get("pareto", {})),
            self._generate_timeline_section(data.get("timeline", {})),
            self._generate_conflict_section(data.get("conflicts", {})),
            self._generate_coverage_section(data.get("coverage", {})),
            self._generate_recommendation_section(data.get("recommendations", [])),
        ]
        body = "\n".join(sections)
        return self._page_template(body)

    def generate_from_analysis(
        self,
        jurisdiction_db: Any = None,
        conflicts: Any = None,
        pareto_data: Any = None,
        coverage: Any = None,
    ) -> str:
        """High-level helper that converts raw analysis objects into dashboard HTML."""
        data: dict[str, Any] = {}

        # --- Summary ---
        summary: dict[str, Any] = {}
        if jurisdiction_db is not None:
            obligations = getattr(jurisdiction_db, "obligations", None)
            summary["total_obligations"] = len(obligations) if obligations else 0
            frameworks = getattr(jurisdiction_db, "frameworks", None)
            summary["frameworks"] = len(frameworks) if frameworks else 0
        if conflicts is not None:
            if isinstance(conflicts, list):
                summary["conflicts"] = len(conflicts)
            elif isinstance(conflicts, dict):
                summary["conflicts"] = len(conflicts.get("edges", conflicts.get("items", [])))
        if coverage is not None:
            if isinstance(coverage, dict):
                summary["coverage_pct"] = coverage.get("overall", coverage.get("coverage_pct", 0))
            elif isinstance(coverage, (int, float)):
                summary["coverage_pct"] = coverage
        data["summary"] = summary

        # --- Pareto ---
        if pareto_data is not None:
            pdata: dict[str, Any] = {}
            if isinstance(pareto_data, dict):
                pdata = pareto_data
            elif isinstance(pareto_data, list):
                pdata = {"points": [(p[0], p[1]) if isinstance(p, (list, tuple)) else (0, 0) for p in pareto_data]}
            data["pareto"] = pdata

        # --- Timeline ---
        if jurisdiction_db is not None:
            deadlines = getattr(jurisdiction_db, "deadlines", None)
            if deadlines:
                data["timeline"] = {"milestones": deadlines if isinstance(deadlines, list) else []}

        # --- Conflicts ---
        if conflicts is not None:
            cdata: dict[str, Any] = {}
            if isinstance(conflicts, dict):
                cdata = conflicts
            elif isinstance(conflicts, list):
                cdata = {"edges": conflicts}
            data["conflicts"] = cdata

        # --- Coverage ---
        if coverage is not None:
            data["coverage"] = coverage if isinstance(coverage, dict) else {"overall": coverage}

        return self.generate(data)

    # ------------------------------------------------------------------
    # Section generators
    # ------------------------------------------------------------------

    def _generate_header(self, title: str) -> str:
        return (
            '<header class="dashboard-header">'
            f'  <h1>{_esc(title)}</h1>'
            '  <nav class="tabs" id="nav-tabs">'
            '    <button class="tab active" data-tab="all">All</button>'
            '    <button class="tab" data-tab="summary">Summary</button>'
            '    <button class="tab" data-tab="pareto">Pareto</button>'
            '    <button class="tab" data-tab="timeline">Timeline</button>'
            '    <button class="tab" data-tab="conflicts">Conflicts</button>'
            '    <button class="tab" data-tab="coverage">Coverage</button>'
            '  </nav>'
            '</header>'
        )

    def _generate_summary_section(self, data: dict) -> str:
        total = data.get("total_obligations", 0)
        frameworks = data.get("frameworks", 0)
        conflicts = data.get("conflicts", 0)
        cov = data.get("coverage_pct", 0)
        trend_cov = "up" if cov > 75 else ("flat" if cov > 50 else "down")
        trend_conf = "down" if conflicts < 5 else ("flat" if conflicts < 15 else "up")
        cards = (
            self._metric_card("Total Obligations", str(total), color="#4e79a7")
            + self._metric_card("Frameworks", str(frameworks), color="#59a14f")
            + self._metric_card("Conflicts", str(conflicts), trend=trend_conf, color="#e15759")
            + self._metric_card("Coverage", f"{cov:.0f}%", trend=trend_cov, color="#76b7b2")
        )
        return self._section_card(
            "Executive Summary",
            f'<div class="metric-grid">{cards}</div>',
            tab="summary",
        )

    def _generate_pareto_section(self, data: dict) -> str:
        points = data.get("points", [])
        labels = data.get("labels")
        pareto_front = data.get("pareto_front")
        svg_str = ""
        if points:
            pp = ParetoPlotter(width=760, height=440, margin=55)
            if pareto_front is None:
                pareto_front = pp.compute_pareto_front(points)
            svg_str = pp.plot_2d(
                points, labels=labels, pareto_front=pareto_front,
                title="Pareto Frontier: Cost vs Coverage",
                x_label=data.get("x_label", "Cost"),
                y_label=data.get("y_label", "Coverage"),
            )
            # Strip XML declaration for inline embedding
            svg_str = _strip_xml_decl(svg_str)
        analysis = data.get("analysis", "")
        return self._section_card(
            "Pareto Analysis",
            f'<div class="chart-container">{svg_str}</div>'
            f'<p class="analysis-text">{_esc(str(analysis))}</p>',
            tab="pareto",
        )

    def _generate_timeline_section(self, data: dict) -> str:
        milestones = data.get("milestones", [])
        tasks = data.get("tasks", [])
        svg_str = ""
        if tasks:
            tp = TimelinePlotter(width=920, height=max(350, 80 + len(tasks) * 34), margin=55)
            svg_str = _strip_xml_decl(tp.plot_gantt(tasks, title="Compliance Roadmap"))
        elif milestones:
            tp = TimelinePlotter(width=920, height=400, margin=55)
            svg_str = _strip_xml_decl(tp.plot_milestone_timeline(milestones, title="Regulatory Milestones"))
        return self._section_card("Timeline", f'<div class="chart-container">{svg_str}</div>', tab="timeline")

    def _generate_conflict_section(self, data: dict) -> str:
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        matrix = data.get("matrix")
        matrix_labels = data.get("matrix_labels", [])
        parts: list[str] = []
        if nodes and edges:
            cg = ConflictGraphPlotter(width=760, height=500)
            parts.append(f'<div class="chart-container">{_strip_xml_decl(cg.plot_conflict_graph(nodes, edges))}</div>')
        if matrix and matrix_labels:
            cg2 = ConflictGraphPlotter(width=760, height=600)
            parts.append(f'<div class="chart-container">{_strip_xml_decl(cg2.plot_conflict_matrix(matrix, matrix_labels))}</div>')
        return self._section_card("Conflict Analysis", "\n".join(parts), tab="conflicts")

    def _generate_coverage_section(self, data: dict) -> str:
        if not data:
            return self._section_card("Coverage Analysis", "<p>No coverage data available.</p>", tab="coverage")
        overall = data.get("overall", data.get("coverage_pct", 0))
        by_framework = data.get("by_framework", {})

        bar_html_parts: list[str] = []
        for fw, pct in sorted(by_framework.items(), key=lambda x: -x[1]):
            color = "#4e79a7" if pct >= 75 else ("#ffbb33" if pct >= 50 else "#e15759")
            bar_html_parts.append(
                f'<div class="bar-row">'
                f'  <span class="bar-label">{_esc(fw)}</span>'
                f'  <div class="bar-track">'
                f'    <div class="bar-fill" style="width:{min(pct, 100):.0f}%;background:{color}"></div>'
                f'  </div>'
                f'  <span class="bar-value">{pct:.0f}%</span>'
                f'</div>'
            )

        overall_color = "#2ca02c" if overall >= 75 else ("#ffbb33" if overall >= 50 else "#e15759")
        content = (
            f'<div class="overall-coverage">'
            f'  <span class="big-number" style="color:{overall_color}">{overall:.0f}%</span>'
            f'  <span class="big-label">Overall Coverage</span>'
            f'</div>'
            + "\n".join(bar_html_parts)
        )
        return self._section_card("Coverage Analysis", content, tab="coverage")

    def _generate_recommendation_section(self, data) -> str:
        recs = data if isinstance(data, list) else []
        if not recs:
            return ""
        items = "".join(
            f'<li class="rec-item"><strong>{_esc(str(r.get("title", r if isinstance(r, str) else "")))}</strong>'
            f' <span class="rec-detail">{_esc(str(r.get("detail", "")))}</span></li>'
            if isinstance(r, dict)
            else f'<li class="rec-item">{_esc(str(r))}</li>'
            for r in recs
        )
        return self._section_card(
            "Recommendations",
            f'<ol class="rec-list">{items}</ol>',
            tab="all",
        )

    def _generate_footer(self) -> str:
        return (
            '<footer class="dashboard-footer">'
            '  <p>Generated by <strong>RegSynth</strong> &mdash; Pareto-Optimal Regulatory Trajectory Synthesis</p>'
            '</footer>'
        )

    # ------------------------------------------------------------------
    # CSS / JS
    # ------------------------------------------------------------------

    def _generate_css(self) -> str:
        return """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#f0f2f5;color:#1a1a2e;line-height:1.6}
.dashboard-header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#fff;padding:24px 32px 0;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.dashboard-header h1{font-size:1.5rem;font-weight:600;margin-bottom:12px}
.tabs{display:flex;gap:4px;overflow-x:auto}
.tab{background:transparent;color:rgba(255,255,255,.7);border:none;padding:8px 18px;cursor:pointer;font-size:.85rem;border-radius:6px 6px 0 0;transition:background .2s,color .2s}
.tab:hover{color:#fff;background:rgba(255,255,255,.08)}
.tab.active{color:#fff;background:rgba(255,255,255,.15);font-weight:600}
.main{max-width:1200px;margin:0 auto;padding:24px 16px 48px}
.section-card{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:24px;overflow:hidden;transition:box-shadow .2s}
.section-card:hover{box-shadow:0 4px 14px rgba(0,0,0,.1)}
.section-title{font-size:1.1rem;font-weight:600;padding:18px 24px;border-bottom:1px solid #eee;color:#1a1a2e}
.section-body{padding:20px 24px}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}
.metric-card{background:#fafbfc;border-radius:10px;padding:20px;text-align:center;border:1px solid #eee;transition:transform .15s}
.metric-card:hover{transform:translateY(-2px)}
.metric-value{font-size:2rem;font-weight:700;line-height:1.1}
.metric-label{font-size:.8rem;color:#666;margin-top:6px;text-transform:uppercase;letter-spacing:.5px}
.metric-trend{font-size:.75rem;margin-top:4px}
.trend-up{color:#2ca02c}
.trend-down{color:#e15759}
.trend-flat{color:#999}
.chart-container{overflow-x:auto;text-align:center;padding:8px 0}
.chart-container svg{max-width:100%;height:auto}
.analysis-text{color:#555;margin-top:12px;font-size:.92rem}
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.bar-label{width:110px;font-size:.85rem;text-align:right;flex-shrink:0;color:#444}
.bar-track{flex:1;background:#eee;border-radius:6px;height:18px;overflow:hidden}
.bar-fill{height:100%;border-radius:6px;transition:width .4s}
.bar-value{width:45px;font-size:.85rem;font-weight:600;color:#333}
.overall-coverage{text-align:center;padding:16px 0 24px}
.big-number{display:block;font-size:3.2rem;font-weight:800;line-height:1}
.big-label{display:block;font-size:.85rem;color:#888;margin-top:4px;text-transform:uppercase;letter-spacing:1px}
.rec-list{padding-left:20px}
.rec-item{margin-bottom:10px;font-size:.92rem}
.rec-item strong{color:#1a1a2e}
.rec-detail{display:block;color:#666;font-size:.84rem}
.dashboard-footer{text-align:center;padding:24px;color:#999;font-size:.8rem}
@media(max-width:640px){.metric-grid{grid-template-columns:1fr 1fr}.dashboard-header{padding:16px}.main{padding:12px 8px}}
"""

    def _generate_js(self) -> str:
        return """
(function(){
  var tabs=document.querySelectorAll('.tab');
  var sections=document.querySelectorAll('.section-card');
  tabs.forEach(function(btn){
    btn.addEventListener('click',function(){
      tabs.forEach(function(b){b.classList.remove('active')});
      btn.classList.add('active');
      var t=btn.getAttribute('data-tab');
      sections.forEach(function(s){
        if(t==='all'){s.style.display='';}
        else{s.style.display=s.getAttribute('data-tab')===t?'':'none';}
      });
    });
  });
})();
"""

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _metric_card(
        self, label: str, value: str,
        trend: Optional[str] = None, color: Optional[str] = None,
    ) -> str:
        style = f' style="color:{color}"' if color else ""
        trend_html = ""
        if trend == "up":
            trend_html = '<div class="metric-trend trend-up">&#9650; improving</div>'
        elif trend == "down":
            trend_html = '<div class="metric-trend trend-down">&#9660; needs attention</div>'
        elif trend == "flat":
            trend_html = '<div class="metric-trend trend-flat">&#9654; stable</div>'
        return (
            f'<div class="metric-card">'
            f'  <div class="metric-value"{style}>{_esc(value)}</div>'
            f'  <div class="metric-label">{_esc(label)}</div>'
            f'  {trend_html}'
            f'</div>'
        )

    def _section_card(self, title: str, content: str, tab: str = "all") -> str:
        return (
            f'<section class="section-card" data-tab="{tab}">'
            f'  <div class="section-title">{_esc(title)}</div>'
            f'  <div class="section-body">{content}</div>'
            f'</section>'
        )

    def _page_template(self, body: str) -> str:
        css = self._generate_css()
        js = self._generate_js()
        header = self._generate_header(self.title)
        footer = self._generate_footer()
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '  <meta charset="UTF-8">\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"  <title>{_esc(self.title)}</title>\n"
            f"  <style>{css}</style>\n"
            "</head>\n<body>\n"
            f"{header}\n"
            f'<main class="main">\n{body}\n</main>\n'
            f"{footer}\n"
            f"<script>{js}</script>\n"
            "</body>\n</html>"
        )

    def save(self, html_content: str, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(html_content)


# ------------------------------------------------------------------
# Module-private helpers
# ------------------------------------------------------------------

def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _strip_xml_decl(svg: str) -> str:
    if svg.startswith("<?xml"):
        idx = svg.find("?>")
        if idx != -1:
            return svg[idx + 2:].lstrip("\n")
    return svg
