"""
Compliance analysis report generator.

Produces HTML, plain-text, and JSON reports summarising regulatory
compliance status across one or more frameworks.
"""

import json
import os
from datetime import datetime

from .templates import Templates


class ComplianceReportGenerator:
    """Generate compliance analysis reports in multiple formats."""

    def __init__(self, jurisdiction_db=None):
        self._tpl = Templates()
        self._jurisdiction_db = jurisdiction_db

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, analysis_data: dict, format_type: str = "html") -> str:
        """Dispatch to the appropriate format generator."""
        dispatch = {
            "html": self.generate_html,
            "text": self.generate_text,
            "json": self.generate_json,
        }
        handler = dispatch.get(format_type)
        if handler is None:
            raise ValueError(f"Unsupported format: {format_type}")
        return handler(analysis_data)

    # ------------------------------------------------------------------ #
    # HTML report
    # ------------------------------------------------------------------ #

    def generate_html(self, data: dict) -> str:
        frameworks = data.get("frameworks", [])
        obligations = data.get("obligations", [])
        strategies = data.get("strategies", [])
        coverage = data.get("coverage", {})
        gaps = data.get("gaps", [])
        recommendations = data.get("recommendations", [])

        score = self._compute_compliance_score(data)
        coverage_by_fw = self._compute_coverage_by_framework(data)
        critical_gaps = self._identify_critical_gaps(data)
        ordered_recs = self._prioritize_recommendations(data)

        sections = []
        sections.append(self._executive_summary(data))
        sections.append(self._framework_coverage_table(frameworks, obligations))
        sections.append(self._obligation_status_table(obligations))
        sections.append(self._gap_analysis(gaps))
        risks = data.get("risks", [])
        sections.append(self._risk_summary(risks))
        sections.append(self._recommendations(ordered_recs))

        # Appendix – detailed obligation list
        sections.append(self._obligation_appendix(obligations))

        content = "\n".join(sections)
        return self._tpl.render_html(
            self._tpl.HTML_REPORT_TEMPLATE,
            title="Compliance Analysis Report",
            content=content,
        )

    # ------------------------------------------------------------------ #
    # Text report
    # ------------------------------------------------------------------ #

    def generate_text(self, data: dict) -> str:
        frameworks = data.get("frameworks", [])
        obligations = data.get("obligations", [])
        gaps = data.get("gaps", [])
        recommendations = data.get("recommendations", [])
        score = self._compute_compliance_score(data)
        ordered_recs = self._prioritize_recommendations(data)

        lines = [
            "COMPLIANCE ANALYSIS REPORT",
            "=" * 72,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            f"Overall compliance score: {score:.1f}/100",
            f"Frameworks analysed: {len(frameworks)}",
            f"Total obligations: {len(obligations)}",
            f"Gaps identified: {len(gaps)}",
            "",
        ]

        # Framework coverage
        lines.append("FRAMEWORK COVERAGE")
        lines.append("-" * 40)
        cov = self._compute_coverage_by_framework(data)
        for fw_name, pct in cov.items():
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            lines.append(f"  {fw_name:<30} [{bar}] {pct:.1f}%")
        lines.append("")

        # Obligations
        lines.append("OBLIGATION STATUS")
        lines.append("-" * 40)
        for ob in obligations:
            status = ob.get("status", "unknown").upper()
            name = ob.get("name", ob.get("id", "?"))
            lines.append(f"  [{status:<15}] {name}")
        lines.append("")

        # Gaps
        lines.append("GAP ANALYSIS")
        lines.append("-" * 40)
        for i, gap in enumerate(gaps, 1):
            severity = gap.get("severity", "medium")
            desc = gap.get("description", str(gap))
            lines.append(f"  {i}. [{severity.upper()}] {desc}")
        lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(ordered_recs, 1):
            priority = rec.get("priority", "medium")
            desc = rec.get("description", str(rec))
            lines.append(f"  {i}. [{priority.upper()}] {desc}")
        lines.append("")

        lines.append("=" * 72)
        lines.append("End of Report")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # JSON report
    # ------------------------------------------------------------------ #

    def generate_json(self, data: dict) -> str:
        score = self._compute_compliance_score(data)
        coverage_by_fw = self._compute_coverage_by_framework(data)
        critical_gaps = self._identify_critical_gaps(data)
        ordered_recs = self._prioritize_recommendations(data)

        report = {
            "report_type": "compliance",
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "data": {
                "score": round(score, 2),
                "frameworks": data.get("frameworks", []),
                "obligations": data.get("obligations", []),
                "gaps": data.get("gaps", []),
                "recommendations": [
                    {
                        "description": r.get("description", str(r)),
                        "priority": r.get("priority", "medium"),
                    }
                    for r in ordered_recs
                ],
                "coverage_by_framework": coverage_by_fw,
                "critical_gaps": critical_gaps,
            },
        }
        schema = self._tpl.get_json_schema("compliance")
        return self._tpl.render_json(report["data"], schema)

    # ------------------------------------------------------------------ #
    # Section builders (HTML)
    # ------------------------------------------------------------------ #

    def _executive_summary(self, data: dict) -> str:
        score = self._compute_compliance_score(data)
        n_fw = len(data.get("frameworks", []))
        n_ob = len(data.get("obligations", []))
        n_gaps = len(data.get("gaps", []))
        compliant = sum(
            1
            for o in data.get("obligations", [])
            if o.get("status", "").lower() == "compliant"
        )

        color = "#27ae60" if score >= 70 else ("#f39c12" if score >= 40 else "#c0392b")

        return (
            '<div class="section">'
            "<h2>Executive Summary</h2>"
            '<div class="metric-grid">'
            f'<div class="metric-card"><div class="metric-value" style="color:{color}">'
            f"{score:.0f}%</div>"
            '<div class="metric-label">Compliance Score</div></div>'
            f'<div class="metric-card"><div class="metric-value">{n_fw}</div>'
            '<div class="metric-label">Frameworks</div></div>'
            f'<div class="metric-card"><div class="metric-value">{compliant}/{n_ob}</div>'
            '<div class="metric-label">Obligations Met</div></div>'
            f'<div class="metric-card"><div class="metric-value">{n_gaps}</div>'
            '<div class="metric-label">Gaps Identified</div></div>'
            "</div></div>"
        )

    def _framework_coverage_table(self, frameworks: list, obligations: list) -> str:
        cov = self._compute_coverage_by_framework(
            {"frameworks": frameworks, "obligations": obligations}
        )
        rows = []
        for fw_name, pct in cov.items():
            bar_cls = (
                "bar-fill-low"
                if pct >= 70
                else ("bar-fill-medium" if pct >= 40 else "bar-fill-high")
            )
            rows.append(
                "<tr>"
                f"<td>{self._tpl.escape_html(fw_name)}</td>"
                f'<td><div class="bar-track"><div class="bar-fill {bar_cls}" '
                f'style="width:{pct:.0f}%">{pct:.0f}%</div></div></td>'
                "</tr>"
            )
        return (
            "<h2>Framework Coverage</h2>"
            '<table><thead><tr><th>Framework</th><th>Coverage</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _obligation_status_table(self, obligations: list) -> str:
        rows = []
        for ob in obligations:
            name = self._tpl.escape_html(ob.get("name", ob.get("id", "—")))
            framework = self._tpl.escape_html(ob.get("framework", "—"))
            status = ob.get("status", "unknown")
            badge = self._tpl.format_status_badge(status)
            rows.append(
                f"<tr><td>{name}</td><td>{framework}</td><td>{badge}</td></tr>"
            )
        return (
            "<h2>Obligation Status</h2>"
            "<table><thead><tr><th>Obligation</th><th>Framework</th>"
            f"<th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )

    def _gap_analysis(self, gaps: list) -> str:
        if not gaps:
            return '<div class="section"><h2>Gap Analysis</h2><p>No gaps identified.</p></div>'
        items = []
        for gap in gaps:
            sev = gap.get("severity", "medium")
            desc = self._tpl.escape_html(gap.get("description", str(gap)))
            badge = self._tpl.format_risk_badge(sev)
            items.append(f"<tr><td>{badge}</td><td>{desc}</td></tr>")
        return (
            "<h2>Gap Analysis</h2>"
            "<table><thead><tr><th>Severity</th><th>Description</th></tr></thead>"
            f"<tbody>{''.join(items)}</tbody></table>"
        )

    def _risk_summary(self, risks: list) -> str:
        if not risks:
            return '<div class="section"><h2>Risk Assessment</h2><p>No risks recorded.</p></div>'
        items = []
        for risk in risks:
            level = risk.get("level", "medium")
            desc = self._tpl.escape_html(risk.get("description", str(risk)))
            badge = self._tpl.format_risk_badge(level)
            impact = self._tpl.escape_html(risk.get("impact", "—"))
            items.append(f"<tr><td>{badge}</td><td>{desc}</td><td>{impact}</td></tr>")
        return (
            "<h2>Risk Assessment</h2>"
            "<table><thead><tr><th>Level</th><th>Risk</th><th>Impact</th></tr></thead>"
            f"<tbody>{''.join(items)}</tbody></table>"
        )

    def _recommendations(self, recs: list) -> str:
        if not recs:
            return '<div class="section"><h2>Recommendations</h2><p>None.</p></div>'
        items = []
        for rec in recs:
            priority = rec.get("priority", "medium").lower()
            desc = self._tpl.escape_html(rec.get("description", str(rec)))
            items.append(
                f'<li class="priority-{priority}"><strong>[{priority.upper()}]</strong> '
                f"{desc}</li>"
            )
        return (
            "<h2>Recommendations</h2>"
            f'<ul class="rec-list">{"".join(items)}</ul>'
        )

    def _obligation_appendix(self, obligations: list) -> str:
        if not obligations:
            return ""
        rows = []
        for ob in obligations:
            oid = self._tpl.escape_html(ob.get("id", "—"))
            name = self._tpl.escape_html(ob.get("name", "—"))
            fw = self._tpl.escape_html(ob.get("framework", "—"))
            article = self._tpl.escape_html(ob.get("article", "—"))
            status = ob.get("status", "unknown")
            badge = self._tpl.format_status_badge(status)
            rows.append(
                f"<tr><td>{oid}</td><td>{name}</td><td>{fw}</td>"
                f"<td>{article}</td><td>{badge}</td></tr>"
            )
        return (
            "<h2>Appendix — Full Obligation List</h2>"
            "<table><thead><tr><th>ID</th><th>Obligation</th><th>Framework</th>"
            "<th>Article</th><th>Status</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #

    def _compute_compliance_score(self, data: dict) -> float:
        """Return a 0-100 compliance score."""
        obligations = data.get("obligations", [])
        if not obligations:
            return 0.0
        status_weights = {"compliant": 1.0, "partial": 0.5, "non-compliant": 0.0}
        total = sum(
            status_weights.get(o.get("status", "non-compliant").lower(), 0.0)
            for o in obligations
        )
        return (total / len(obligations)) * 100.0

    def _compute_coverage_by_framework(self, data: dict) -> dict:
        """Return ``{framework_name: coverage_pct}``."""
        obligations = data.get("obligations", [])
        frameworks = data.get("frameworks", [])
        fw_names = [
            f.get("name", f.get("id", str(f))) if isinstance(f, dict) else str(f)
            for f in frameworks
        ]
        result = {}
        for fw in fw_names:
            fw_obs = [o for o in obligations if o.get("framework") == fw]
            if not fw_obs:
                result[fw] = 0.0
                continue
            compliant = sum(
                1 for o in fw_obs if o.get("status", "").lower() in ("compliant", "partial")
            )
            result[fw] = (compliant / len(fw_obs)) * 100.0
        return result

    def _identify_critical_gaps(self, data: dict) -> list:
        """Return gaps with severity ``critical`` or ``high``."""
        return [
            g
            for g in data.get("gaps", [])
            if g.get("severity", "").lower() in ("critical", "high")
        ]

    def _prioritize_recommendations(self, data: dict) -> list:
        """Return recommendations sorted by priority (high → low)."""
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recs = list(data.get("recommendations", []))
        recs.sort(key=lambda r: priority_order.get(r.get("priority", "medium").lower(), 1))
        return recs

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, content: str, filepath: str) -> None:
        """Write *content* to *filepath*, creating parent directories."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
