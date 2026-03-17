"""
Conflict analysis report generator.

Produces HTML, plain-text, and JSON reports that summarise regulatory
conflicts detected between frameworks, including MUS explanations and
resolution options.
"""

import json
import os
from collections import Counter
from datetime import datetime

from .templates import Templates


class ConflictReportGenerator:
    """Generate conflict analysis reports in multiple formats."""

    def __init__(self):
        self._tpl = Templates()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        conflicts: list,
        frameworks: list,
        format_type: str = "html",
    ) -> str:
        dispatch = {
            "html": self.generate_html,
            "text": self.generate_text,
            "json": self.generate_json,
        }
        handler = dispatch.get(format_type)
        if handler is None:
            raise ValueError(f"Unsupported format: {format_type}")
        return handler(conflicts, frameworks)

    # ------------------------------------------------------------------ #
    # HTML report
    # ------------------------------------------------------------------ #

    def generate_html(self, conflicts: list, frameworks: list) -> str:
        sections = [
            self._conflict_summary(conflicts),
            self._severity_chart_html(conflicts),
        ]
        for conflict in conflicts:
            sections.append(self._conflict_detail(conflict))
            if conflict.get("severity", "").lower() in ("critical", "high"):
                sections.append(self._mus_explanation(conflict))
            sections.append(self._resolution_options(conflict))

        sections.append(self._cross_reference_table(conflicts, frameworks))
        sections.append(self._impact_analysis(conflicts))

        content = "\n".join(sections)
        return self._tpl.render_html(
            self._tpl.HTML_REPORT_TEMPLATE,
            title="Conflict Analysis Report",
            content=content,
        )

    # ------------------------------------------------------------------ #
    # Text report
    # ------------------------------------------------------------------ #

    def generate_text(self, conflicts: list, frameworks: list) -> str:
        severity_dist = Counter(
            c.get("severity", "medium").lower() for c in conflicts
        )
        score = self._compute_conflict_score(conflicts)

        lines = [
            "CONFLICT ANALYSIS REPORT",
            "=" * 72,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total conflicts: {len(conflicts)}",
            f"Conflict score:  {score:.1f}/100",
            f"Critical: {severity_dist.get('critical', 0)}  "
            f"High: {severity_dist.get('high', 0)}  "
            f"Medium: {severity_dist.get('medium', 0)}  "
            f"Low: {severity_dist.get('low', 0)}",
            "",
        ]

        for i, conflict in enumerate(conflicts, 1):
            sev = conflict.get("severity", "medium").upper()
            desc = conflict.get("description", str(conflict))
            fw_a = conflict.get("framework_a", "?")
            fw_b = conflict.get("framework_b", "?")
            art_a = conflict.get("article_a", "—")
            art_b = conflict.get("article_b", "—")
            lines.append(f"CONFLICT #{i} [{sev}]")
            lines.append("-" * 40)
            lines.append(f"  {fw_a} Art.{art_a}  <->  {fw_b} Art.{art_b}")
            lines.append(f"  {desc}")
            mus = conflict.get("mus", [])
            if mus:
                lines.append(f"  MUS: {', '.join(str(m) for m in mus)}")
            resolutions = conflict.get("resolutions", [])
            if resolutions:
                lines.append("  Resolutions:")
                for r in resolutions:
                    lines.append(f"    - {r.get('description', str(r))}")
            lines.append("")

        # Cross-reference
        lines.append("CROSS-REFERENCE TABLE")
        lines.append("-" * 40)
        fw_names = [
            f.get("name", str(f)) if isinstance(f, dict) else str(f)
            for f in frameworks
        ]
        header = f"{'':20}" + "".join(f"{n[:12]:>14}" for n in fw_names)
        lines.append(header)
        for fw in fw_names:
            row = f"{fw[:20]:20}"
            for other in fw_names:
                count = sum(
                    1
                    for c in conflicts
                    if {c.get("framework_a"), c.get("framework_b")} == {fw, other}
                    or (
                        fw == other
                        and c.get("framework_a") == fw
                        and c.get("framework_b") == fw
                    )
                )
                row += f"{count:>14}"
            lines.append(row)
        lines.append("")

        lines.append("=" * 72)
        lines.append("End of Report")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # JSON report
    # ------------------------------------------------------------------ #

    def generate_json(self, conflicts: list, frameworks: list) -> str:
        severity_dist = Counter(
            c.get("severity", "medium").lower() for c in conflicts
        )
        score = self._compute_conflict_score(conflicts)

        report_data = {
            "conflicts": conflicts,
            "summary": {
                "total": len(conflicts),
                "score": round(score, 2),
                "severity_distribution": dict(severity_dist),
            },
            "resolutions": [
                {
                    "conflict_id": c.get("id", f"conflict-{i}"),
                    "options": c.get("resolutions", []),
                }
                for i, c in enumerate(conflicts, 1)
            ],
            "frameworks": [
                f.get("name", str(f)) if isinstance(f, dict) else str(f)
                for f in frameworks
            ],
        }
        schema = self._tpl.get_json_schema("conflict")
        return self._tpl.render_json(report_data, schema)

    # ------------------------------------------------------------------ #
    # Section builders (HTML)
    # ------------------------------------------------------------------ #

    def _conflict_summary(self, conflicts: list) -> str:
        severity_dist = Counter(
            c.get("severity", "medium").lower() for c in conflicts
        )
        score = self._compute_conflict_score(conflicts)
        color = "#27ae60" if score < 30 else ("#f39c12" if score < 60 else "#c0392b")

        return (
            '<div class="section"><h2>Conflict Summary</h2>'
            '<div class="metric-grid">'
            f'<div class="metric-card"><div class="metric-value">{len(conflicts)}'
            '</div><div class="metric-label">Total Conflicts</div></div>'
            f'<div class="metric-card"><div class="metric-value" style="color:{color}">'
            f'{score:.0f}</div><div class="metric-label">Conflict Score</div></div>'
            f'<div class="metric-card"><div class="metric-value">'
            f'{severity_dist.get("critical", 0)}</div>'
            '<div class="metric-label">Critical</div></div>'
            f'<div class="metric-card"><div class="metric-value">'
            f'{severity_dist.get("high", 0)}</div>'
            '<div class="metric-label">High</div></div>'
            "</div></div>"
        )

    def _severity_chart_html(self, conflicts: list) -> str:
        severity_dist = Counter(
            c.get("severity", "medium").lower() for c in conflicts
        )
        total = max(len(conflicts), 1)
        bar_map = {
            "critical": "bar-fill-high",
            "high": "bar-fill-high",
            "medium": "bar-fill-medium",
            "low": "bar-fill-low",
        }
        rows = []
        for level in ("critical", "high", "medium", "low"):
            count = severity_dist.get(level, 0)
            pct = (count / total) * 100
            css = bar_map.get(level, "bar-fill-default")
            rows.append(
                f'<div class="bar-row">'
                f'<span class="bar-label">{level.capitalize()} ({count})</span>'
                f'<div class="bar-track"><div class="bar-fill {css}" '
                f'style="width:{pct:.0f}%">{count}</div></div></div>'
            )
        return (
            '<div class="section"><h2>Severity Distribution</h2>'
            f'<div class="bar-chart">{"".join(rows)}</div></div>'
        )

    def _conflict_detail(self, conflict: dict) -> str:
        sev = conflict.get("severity", "medium")
        badge = self._tpl.format_risk_badge(sev)
        fw_a = self._tpl.escape_html(conflict.get("framework_a", "?"))
        fw_b = self._tpl.escape_html(conflict.get("framework_b", "?"))
        art_a = self._tpl.escape_html(str(conflict.get("article_a", "—")))
        art_b = self._tpl.escape_html(str(conflict.get("article_b", "—")))
        desc = self._tpl.escape_html(conflict.get("description", ""))
        cid = self._tpl.escape_html(conflict.get("id", "—"))

        return (
            '<div class="section">'
            f"<h3>Conflict: {cid} {badge}</h3>"
            f"<p><strong>{fw_a}</strong> Art.&nbsp;{art_a} &harr; "
            f"<strong>{fw_b}</strong> Art.&nbsp;{art_b}</p>"
            f"<p>{desc}</p></div>"
        )

    def _mus_explanation(self, conflict: dict) -> str:
        """Explain the Minimal Unsatisfiable Subset for this conflict.

        The MUS is the smallest set of constraints that cannot all be
        satisfied simultaneously. Removing any single element from the MUS
        makes the remaining set satisfiable.
        """
        mus = conflict.get("mus", [])
        if not mus:
            return ""
        items = "".join(
            f"<li>{self._tpl.escape_html(str(m))}</li>" for m in mus
        )
        cid = self._tpl.escape_html(conflict.get("id", "—"))
        return (
            '<div class="section" style="border-left:4px solid #c0392b;'
            'padding-left:1.25rem">'
            f"<h3>MUS Analysis — {cid}</h3>"
            "<p>The following constraints form a <em>minimal unsatisfiable "
            "subset</em> (MUS).  No proper subset of these constraints is "
            "contradictory; however, taken together they are irreconcilable. "
            "Resolving the conflict requires relaxing or removing at least "
            "one element.</p>"
            f"<ol>{items}</ol></div>"
        )

    def _resolution_options(self, conflict: dict) -> str:
        resolutions = conflict.get("resolutions", [])
        if not resolutions:
            return ""
        items = []
        for res in resolutions:
            desc = self._tpl.escape_html(res.get("description", str(res)))
            impact = self._tpl.escape_html(res.get("impact", "—"))
            feasibility = self._tpl.escape_html(res.get("feasibility", "—"))
            items.append(
                f"<tr><td>{desc}</td><td>{impact}</td><td>{feasibility}</td></tr>"
            )
        return (
            "<h3>Resolution Options</h3>"
            "<table><thead><tr><th>Option</th><th>Impact</th>"
            f"<th>Feasibility</th></tr></thead><tbody>{''.join(items)}"
            "</tbody></table>"
        )

    def _cross_reference_table(self, conflicts: list, frameworks: list) -> str:
        fw_names = [
            f.get("name", str(f)) if isinstance(f, dict) else str(f)
            for f in frameworks
        ]
        # Build a conflict-count matrix
        matrix: dict[str, dict[str, int]] = {
            a: {b: 0 for b in fw_names} for a in fw_names
        }
        for c in conflicts:
            a = c.get("framework_a", "")
            b = c.get("framework_b", "")
            if a in matrix and b in matrix[a]:
                matrix[a][b] += 1
            if b in matrix and a in matrix[b]:
                matrix[b][a] += 1

        header = "".join(
            f"<th>{self._tpl.escape_html(n)}</th>" for n in fw_names
        )
        rows = []
        for fw in fw_names:
            cells = "".join(
                f"<td>{matrix[fw][other]}</td>" for other in fw_names
            )
            rows.append(
                f"<tr><td><strong>{self._tpl.escape_html(fw)}</strong></td>"
                f"{cells}</tr>"
            )
        return (
            "<h2>Cross-Reference Matrix</h2>"
            f"<table><thead><tr><th></th>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _impact_analysis(self, conflicts: list) -> str:
        if not conflicts:
            return '<div class="section"><h2>Impact Analysis</h2><p>No conflicts to analyse.</p></div>'

        high_impact = [
            c for c in conflicts if c.get("severity", "").lower() in ("critical", "high")
        ]
        affected_frameworks = set()
        for c in conflicts:
            affected_frameworks.add(c.get("framework_a", "?"))
            affected_frameworks.add(c.get("framework_b", "?"))

        score = self._compute_conflict_score(conflicts)
        assessment = (
            "The regulatory landscape presents <strong>significant tensions</strong>."
            if score >= 60
            else (
                "There are <strong>moderate tensions</strong> between frameworks."
                if score >= 30
                else "Conflicts are <strong>manageable</strong> with targeted remediation."
            )
        )

        return (
            '<div class="section"><h2>Impact Analysis</h2>'
            f"<p>{assessment}</p>"
            f"<p><strong>{len(high_impact)}</strong> conflict(s) are rated critical "
            f"or high severity, affecting <strong>{len(affected_frameworks)}</strong> "
            "framework(s).</p>"
            "<p>Immediate attention should focus on critical conflicts, as "
            "they represent irreconcilable requirements that may expose the "
            "organisation to regulatory enforcement risk.</p></div>"
        )

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #

    def _compute_conflict_score(self, conflicts: list) -> float:
        """Return 0-100 score where higher means more severe conflict landscape."""
        if not conflicts:
            return 0.0
        weights = {"critical": 10, "high": 6, "medium": 3, "low": 1}
        total_weight = sum(
            weights.get(c.get("severity", "medium").lower(), 3)
            for c in conflicts
        )
        max_weight = len(conflicts) * 10
        return min((total_weight / max_weight) * 100, 100.0)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, content: str, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
