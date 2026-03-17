"""
Compliance roadmap report generator.

Produces HTML, plain-text, and JSON reports describing a phased
compliance roadmap with timelines, budgets, resources, and milestones.
"""

import json
import os
from datetime import datetime

from .templates import Templates


class RoadmapReportGenerator:
    """Generate compliance roadmap reports in multiple formats."""

    def __init__(self):
        self._tpl = Templates()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, roadmap_data: dict, format_type: str = "html") -> str:
        dispatch = {
            "html": self.generate_html,
            "text": self.generate_text,
            "json": self.generate_json,
        }
        handler = dispatch.get(format_type)
        if handler is None:
            raise ValueError(f"Unsupported format: {format_type}")
        return handler(roadmap_data)

    # ------------------------------------------------------------------ #
    # HTML report
    # ------------------------------------------------------------------ #

    def generate_html(self, data: dict) -> str:
        phases = data.get("phases", [])
        milestones = data.get("milestones", [])
        resources = data.get("resources", [])
        budget = data.get("budget", {})

        total_months = self._estimate_total_duration(phases)
        total_cost = self._estimate_total_cost(phases)
        critical = self._critical_path(phases)

        sections = [
            self._overview_metrics(phases, total_months, total_cost),
            self._phase_overview(phases),
        ]
        for phase in phases:
            sections.append(self._phase_detail(phase))

        sections.append(self._resource_table(resources))
        sections.append(self._budget_breakdown(budget))
        sections.append(self._milestone_list(milestones))
        sections.append(self._dependency_diagram_html(phases))

        content = "\n".join(sections)
        return self._tpl.render_html(
            self._tpl.HTML_REPORT_TEMPLATE,
            title="Compliance Roadmap",
            content=content,
        )

    # ------------------------------------------------------------------ #
    # Text report
    # ------------------------------------------------------------------ #

    def generate_text(self, data: dict) -> str:
        phases = data.get("phases", [])
        milestones = data.get("milestones", [])
        resources = data.get("resources", [])
        budget = data.get("budget", {})

        total_months = self._estimate_total_duration(phases)
        total_cost = self._estimate_total_cost(phases)
        critical = self._critical_path(phases)

        lines = [
            "COMPLIANCE ROADMAP",
            "=" * 72,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "OVERVIEW",
            "-" * 40,
            f"Total phases:   {len(phases)}",
            f"Total duration: {total_months} month(s)",
            f"Total cost:     {self._tpl.format_currency(total_cost)}",
            f"Critical path:  {' -> '.join(critical) if critical else '—'}",
            "",
        ]

        # Phases
        for i, phase in enumerate(phases, 1):
            name = phase.get("name", f"Phase {i}")
            duration = phase.get("duration_months", "?")
            cost = phase.get("cost", 0)
            lines.append(f"PHASE {i}: {name}")
            lines.append("-" * 40)
            lines.append(f"  Duration: {duration} month(s)")
            lines.append(f"  Cost:     {self._tpl.format_currency(cost)}")
            for action in phase.get("actions", []):
                lines.append(f"  - {action}")
            risks = phase.get("risks", [])
            if risks:
                lines.append("  Risks:")
                for r in risks:
                    lines.append(f"    ! {r}")
            deps = phase.get("dependencies", [])
            if deps:
                lines.append(f"  Depends on: {', '.join(deps)}")
            deliverables = phase.get("deliverables", [])
            if deliverables:
                lines.append("  Deliverables:")
                for d in deliverables:
                    lines.append(f"    * {d}")
            lines.append("")

        # Resources
        if resources:
            lines.append("RESOURCES")
            lines.append("-" * 40)
            for res in resources:
                name = res.get("name", "?")
                role = res.get("role", "—")
                alloc = res.get("allocation", "—")
                lines.append(f"  {name:<25} {role:<20} {alloc}")
            lines.append("")

        # Budget
        if budget:
            lines.append("BUDGET BREAKDOWN")
            lines.append("-" * 40)
            for cat, amount in budget.items():
                lines.append(
                    f"  {cat:<30} {self._tpl.format_currency(float(amount))}"
                )
            lines.append("")

        # Milestones
        if milestones:
            lines.append("KEY MILESTONES")
            lines.append("-" * 40)
            for ms in milestones:
                date = ms.get("date", "TBD")
                desc = ms.get("description", str(ms))
                lines.append(f"  [{date}] {desc}")
            lines.append("")

        # Dependency diagram
        lines.append("DEPENDENCY DIAGRAM")
        lines.append("-" * 40)
        lines.append(self._dependency_diagram(phases))
        lines.append("")

        lines.append("=" * 72)
        lines.append("End of Report")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # JSON report
    # ------------------------------------------------------------------ #

    def generate_json(self, data: dict) -> str:
        phases = data.get("phases", [])
        total_months = self._estimate_total_duration(phases)
        total_cost = self._estimate_total_cost(phases)
        critical = self._critical_path(phases)

        report_data = {
            "phases": phases,
            "timeline": {
                "total_months": total_months,
                "critical_path": critical,
            },
            "budget": data.get("budget", {}),
            "milestones": data.get("milestones", []),
            "resources": data.get("resources", []),
            "total_cost": round(total_cost, 2),
        }
        schema = self._tpl.get_json_schema("roadmap")
        return self._tpl.render_json(report_data, schema)

    # ------------------------------------------------------------------ #
    # Section builders (HTML)
    # ------------------------------------------------------------------ #

    def _overview_metrics(
        self, phases: list, total_months: int, total_cost: float
    ) -> str:
        critical = self._critical_path(phases)
        return (
            '<div class="section"><h2>Roadmap Overview</h2>'
            '<div class="metric-grid">'
            f'<div class="metric-card"><div class="metric-value">{len(phases)}'
            '</div><div class="metric-label">Phases</div></div>'
            f'<div class="metric-card"><div class="metric-value">{total_months}'
            '</div><div class="metric-label">Months</div></div>'
            f'<div class="metric-card"><div class="metric-value">'
            f'{self._tpl.format_currency(total_cost)}</div>'
            '<div class="metric-label">Total Cost</div></div>'
            f'<div class="metric-card"><div class="metric-value">{len(critical)}'
            '</div><div class="metric-label">Critical-path Phases</div></div>'
            "</div></div>"
        )

    def _phase_overview(self, phases: list) -> str:
        rows = []
        for i, phase in enumerate(phases, 1):
            name = self._tpl.escape_html(phase.get("name", f"Phase {i}"))
            dur = phase.get("duration_months", "?")
            cost = self._tpl.format_currency(phase.get("cost", 0))
            deps = ", ".join(phase.get("dependencies", [])) or "—"
            rows.append(
                f"<tr><td>{i}</td><td>{name}</td><td>{dur} mo</td>"
                f"<td>{cost}</td><td>{self._tpl.escape_html(deps)}</td></tr>"
            )
        return (
            "<h2>Phase Overview</h2>"
            "<table><thead><tr><th>#</th><th>Phase</th><th>Duration</th>"
            "<th>Cost</th><th>Dependencies</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _phase_detail(self, phase: dict) -> str:
        name = self._tpl.escape_html(phase.get("name", "Unnamed Phase"))
        dur = phase.get("duration_months", "?")
        cost = self._tpl.format_currency(phase.get("cost", 0))

        actions = "".join(
            f"<li>{self._tpl.escape_html(a)}</li>"
            for a in phase.get("actions", [])
        )
        deliverables = "".join(
            f"<li>{self._tpl.escape_html(d)}</li>"
            for d in phase.get("deliverables", [])
        )
        resources = "".join(
            f"<li>{self._tpl.escape_html(r)}</li>"
            for r in phase.get("resources", [])
        )
        risks = phase.get("risks", [])
        risk_html = ""
        if risks:
            risk_items = "".join(
                f"<li>{self._tpl.format_risk_badge('medium')} "
                f"{self._tpl.escape_html(r)}</li>"
                for r in risks
            )
            risk_html = f"<h4>Risk Factors</h4><ul>{risk_items}</ul>"

        return (
            f'<div class="phase-card"><h3>{name}</h3>'
            f"<p>Duration: <strong>{dur} month(s)</strong> &mdash; "
            f"Budget: <strong>{cost}</strong></p>"
            f"<h4>Actions</h4><ul>{actions or '<li>None specified</li>'}</ul>"
            f"<h4>Deliverables</h4><ul>{deliverables or '<li>None specified</li>'}</ul>"
            f"<h4>Resources</h4><ul>{resources or '<li>None specified</li>'}</ul>"
            f"{risk_html}</div>"
        )

    def _resource_table(self, resources: list) -> str:
        if not resources:
            return ""
        rows = []
        for res in resources:
            name = self._tpl.escape_html(res.get("name", "?"))
            role = self._tpl.escape_html(res.get("role", "—"))
            alloc = self._tpl.escape_html(str(res.get("allocation", "—")))
            rows.append(f"<tr><td>{name}</td><td>{role}</td><td>{alloc}</td></tr>")
        return (
            "<h2>Resource Requirements</h2>"
            "<table><thead><tr><th>Resource</th><th>Role</th>"
            f"<th>Allocation</th></tr></thead><tbody>{''.join(rows)}"
            "</tbody></table>"
        )

    def _budget_breakdown(self, budget: dict) -> str:
        if not budget:
            return ""
        rows = []
        total = 0.0
        for category, amount in budget.items():
            amt = float(amount)
            total += amt
            rows.append(
                f"<tr><td>{self._tpl.escape_html(category)}</td>"
                f"<td>{self._tpl.format_currency(amt)}</td></tr>"
            )
        rows.append(
            f"<tr><td><strong>Total</strong></td>"
            f"<td><strong>{self._tpl.format_currency(total)}</strong></td></tr>"
        )
        return (
            "<h2>Budget Breakdown</h2>"
            "<table><thead><tr><th>Category</th><th>Amount</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _milestone_list(self, milestones: list) -> str:
        if not milestones:
            return ""
        items = []
        for ms in milestones:
            date = self._tpl.escape_html(ms.get("date", "TBD"))
            desc = self._tpl.escape_html(ms.get("description", str(ms)))
            items.append(f"<tr><td><strong>{date}</strong></td><td>{desc}</td></tr>")
        return (
            "<h2>Key Milestones</h2>"
            "<table><thead><tr><th>Date</th><th>Milestone</th></tr></thead>"
            f"<tbody>{''.join(items)}</tbody></table>"
        )

    def _dependency_diagram_html(self, phases: list) -> str:
        """Render the ASCII dependency diagram inside a <pre> block."""
        diagram = self._dependency_diagram(phases)
        return (
            '<div class="section"><h2>Dependency Diagram</h2>'
            f"<pre>{self._tpl.escape_html(diagram)}</pre></div>"
        )

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #

    def _dependency_diagram(self, phases: list) -> str:
        """Build an ASCII-art dependency diagram.

        Example output::

            [Phase 1] ──> [Phase 2] ──> [Phase 4]
                     \\──> [Phase 3] ──/
        """
        if not phases:
            return "(no phases)"

        name_map = {}
        for i, p in enumerate(phases):
            name_map[p.get("name", f"Phase {i + 1}")] = p

        lines = []
        for phase in phases:
            name = phase.get("name", "?")
            deps = phase.get("dependencies", [])
            if not deps:
                lines.append(f"[{name}]  (no dependencies)")
            else:
                for dep in deps:
                    lines.append(f"[{dep}] --> [{name}]")
        return "\n".join(lines) if lines else "(no dependencies)"

    def _critical_path(self, phases: list) -> list:
        """Compute the critical path (longest chain of dependencies).

        Uses a simple topological-sort approach where the critical path is
        the sequence of phases whose cumulative duration is maximal.
        """
        if not phases:
            return []

        name_to_phase = {}
        for i, p in enumerate(phases):
            name = p.get("name", f"Phase {i + 1}")
            name_to_phase[name] = p

        # Memoised longest-path calculation
        memo: dict[str, tuple[int, list[str]]] = {}

        def longest(name: str) -> tuple[int, list[str]]:
            if name in memo:
                return memo[name]
            phase = name_to_phase.get(name)
            if phase is None:
                memo[name] = (0, [])
                return memo[name]
            dur = phase.get("duration_months", 0)
            deps = phase.get("dependencies", [])
            if not deps:
                memo[name] = (dur, [name])
                return memo[name]
            best_len = 0
            best_path: list[str] = []
            for dep in deps:
                dep_len, dep_path = longest(dep)
                if dep_len > best_len:
                    best_len = dep_len
                    best_path = dep_path
            memo[name] = (best_len + dur, best_path + [name])
            return memo[name]

        best_overall: list[str] = []
        best_dur = 0
        for name in name_to_phase:
            dur, path = longest(name)
            if dur > best_dur:
                best_dur = dur
                best_overall = path
        return best_overall

    def _estimate_total_duration(self, phases: list) -> int:
        """Total months along the critical path."""
        critical = self._critical_path(phases)
        name_to_phase = {}
        for i, p in enumerate(phases):
            name_to_phase[p.get("name", f"Phase {i + 1}")] = p
        return sum(
            name_to_phase.get(n, {}).get("duration_months", 0) for n in critical
        )

    def _estimate_total_cost(self, phases: list) -> float:
        """Sum cost across all phases."""
        return sum(float(p.get("cost", 0)) for p in phases)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, content: str, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
