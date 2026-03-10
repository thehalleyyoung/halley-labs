"""
usability_oracle.wcag.reporter — WCAG conformance report generation.

Generates conformance reports in JSON, Markdown, HTML, and WCAG-EM formats.
Includes summary statistics and remediation priority rankings using
cognitive cost weighting.
"""

from __future__ import annotations

import json
import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    SuccessCriterion,
    WCAGGuideline,
    WCAGPrinciple,
    WCAGResult,
    WCAGViolation,
)


# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PrincipleStats:
    """Summary statistics for one WCAG principle."""

    principle: WCAGPrinciple
    violation_count: int = 0
    critical_count: int = 0
    serious_count: int = 0
    moderate_count: int = 0
    minor_count: int = 0

    @property
    def total_weighted(self) -> float:
        """Weighted violation score (critical=4, serious=3, moderate=2, minor=1)."""
        return (
            self.critical_count * 4.0
            + self.serious_count * 3.0
            + self.moderate_count * 2.0
            + self.minor_count * 1.0
        )


@dataclass
class LevelStats:
    """Summary statistics for one conformance level."""

    level: ConformanceLevel
    violation_count: int = 0
    affected_criteria: int = 0


@dataclass
class ReportSummary:
    """Aggregate summary of a WCAG evaluation."""

    total_violations: int
    total_criteria_tested: int
    total_criteria_passed: int
    conformance_ratio: float
    is_conformant: bool
    target_level: ConformanceLevel
    by_principle: Dict[str, PrincipleStats]
    by_level: Dict[str, LevelStats]
    by_impact: Dict[str, int]
    top_violated_criteria: List[Tuple[str, str, int]]  # (sc_id, name, count)


def compute_summary(result: WCAGResult) -> ReportSummary:
    """Compute summary statistics from a WCAGResult."""
    # By principle
    by_principle: Dict[str, PrincipleStats] = {}
    for p in WCAGPrinciple:
        by_principle[p.value] = PrincipleStats(principle=p)

    for v in result.violations:
        pkey = v.criterion.principle.value
        ps = by_principle[pkey]
        ps.violation_count += 1
        if v.impact == ImpactLevel.CRITICAL:
            ps.critical_count += 1
        elif v.impact == ImpactLevel.SERIOUS:
            ps.serious_count += 1
        elif v.impact == ImpactLevel.MODERATE:
            ps.moderate_count += 1
        else:
            ps.minor_count += 1

    # By level
    by_level: Dict[str, LevelStats] = {}
    for lvl in ConformanceLevel:
        by_level[lvl.value] = LevelStats(level=lvl)
    criteria_violated_at_level: Dict[str, set[str]] = defaultdict(set)

    for v in result.violations:
        lkey = v.conformance_level.value
        by_level[lkey].violation_count += 1
        criteria_violated_at_level[lkey].add(v.sc_id)

    for lkey, crit_set in criteria_violated_at_level.items():
        by_level[lkey].affected_criteria = len(crit_set)

    # By impact
    by_impact: Dict[str, int] = {imp.value: 0 for imp in ImpactLevel}
    for v in result.violations:
        by_impact[v.impact.value] += 1

    # Top violated criteria
    criteria_counts: Dict[str, Tuple[str, int]] = {}
    for v in result.violations:
        if v.sc_id in criteria_counts:
            name, count = criteria_counts[v.sc_id]
            criteria_counts[v.sc_id] = (name, count + 1)
        else:
            criteria_counts[v.sc_id] = (v.criterion.name, 1)

    top_violated = sorted(
        [(sc_id, name, count) for sc_id, (name, count) in criteria_counts.items()],
        key=lambda x: x[2],
        reverse=True,
    )

    return ReportSummary(
        total_violations=result.violation_count,
        total_criteria_tested=result.criteria_tested,
        total_criteria_passed=result.criteria_passed,
        conformance_ratio=result.conformance_ratio,
        is_conformant=result.is_conformant,
        target_level=result.target_level,
        by_principle=by_principle,
        by_level=by_level,
        by_impact=by_impact,
        top_violated_criteria=top_violated[:10],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Remediation priority ranking
# ═══════════════════════════════════════════════════════════════════════════

# Cognitive cost multipliers per impact level (bits of additional cognitive load)
_COGNITIVE_COST_WEIGHTS: Dict[ImpactLevel, float] = {
    ImpactLevel.CRITICAL: 8.0,   # ~3 bits of surprise × high uncertainty
    ImpactLevel.SERIOUS: 4.0,
    ImpactLevel.MODERATE: 2.0,
    ImpactLevel.MINOR: 0.5,
}

# Priority boost for level A criteria (legal/fundamental)
_LEVEL_PRIORITY_BOOST: Dict[ConformanceLevel, float] = {
    ConformanceLevel.A: 3.0,
    ConformanceLevel.AA: 1.5,
    ConformanceLevel.AAA: 1.0,
}


@dataclass(frozen=True, slots=True)
class RemediationItem:
    """A prioritised remediation recommendation."""

    sc_id: str
    criterion_name: str
    violation_count: int
    max_impact: ImpactLevel
    priority_score: float
    cognitive_cost_delta: float
    affected_node_ids: Tuple[str, ...]
    remediation_hint: str


def rank_remediations(result: WCAGResult) -> List[RemediationItem]:
    """Rank violations by remediation priority using cognitive cost.

    Priority is computed as:
        score = sum_violations(cognitive_cost_weight × level_boost)

    Higher scores indicate more impactful violations that should be
    fixed first.
    """
    # Group violations by criterion
    by_criterion: Dict[str, List[WCAGViolation]] = defaultdict(list)
    for v in result.violations:
        by_criterion[v.sc_id].append(v)

    items: List[RemediationItem] = []
    for sc_id, violations in by_criterion.items():
        max_impact = max(violations, key=lambda v: v.impact.numeric).impact
        level = violations[0].conformance_level

        cognitive_cost = sum(
            _COGNITIVE_COST_WEIGHTS.get(v.impact, 1.0) for v in violations
        )
        level_boost = _LEVEL_PRIORITY_BOOST.get(level, 1.0)
        priority = cognitive_cost * level_boost

        # Use the most common remediation suggestion
        remediations = [v.remediation for v in violations if v.remediation]
        hint = remediations[0] if remediations else "Review and fix this criterion."

        items.append(RemediationItem(
            sc_id=sc_id,
            criterion_name=violations[0].criterion.name,
            violation_count=len(violations),
            max_impact=max_impact,
            priority_score=priority,
            cognitive_cost_delta=cognitive_cost,
            affected_node_ids=tuple(v.node_id for v in violations),
            remediation_hint=hint,
        ))

    items.sort(key=lambda x: x.priority_score, reverse=True)
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Report formatters
# ═══════════════════════════════════════════════════════════════════════════

class WCAGConformanceReporter:
    """Format WCAG conformance results for various output channels.

    Implements the :class:`~usability_oracle.wcag.protocols.WCAGReporter`
    protocol.

    Supported formats: ``"json"``, ``"markdown"``, ``"html"``, ``"wcag-em"``.
    """

    def format_result(
        self,
        result: WCAGResult,
        *,
        format: str = "json",
    ) -> str:
        """Render a WCAGResult to a string.

        Parameters
        ----------
        result : WCAGResult
        format : str
            One of ``"json"``, ``"markdown"``, ``"html"``, ``"wcag-em"``.
        """
        formatters = {
            "json": self._format_json,
            "markdown": self._format_markdown,
            "html": self._format_html,
            "wcag-em": self._format_wcag_em,
        }
        fmt = format.lower().strip()
        if fmt not in formatters:
            raise ValueError(f"Unknown format {format!r}; expected one of {list(formatters.keys())}")
        return formatters[fmt](result)

    def summary(self, result: WCAGResult) -> str:
        """One-line summary suitable for CI/CD log output."""
        status = "PASS" if result.is_conformant else "FAIL"
        return (
            f"WCAG {result.target_level.value} [{status}] — "
            f"{result.criteria_passed}/{result.criteria_tested} criteria passed, "
            f"{result.violation_count} violations"
        )

    # -- JSON ---------------------------------------------------------------

    def _format_json(self, result: WCAGResult) -> str:
        summary = compute_summary(result)
        remediations = rank_remediations(result)

        doc = {
            "wcag_version": "2.2",
            "target_level": result.target_level.value,
            "conformant": result.is_conformant,
            "page_url": result.page_url,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_violations": summary.total_violations,
                "criteria_tested": summary.total_criteria_tested,
                "criteria_passed": summary.total_criteria_passed,
                "conformance_ratio": round(summary.conformance_ratio, 4),
                "by_impact": summary.by_impact,
                "by_principle": {
                    k: {
                        "violations": v.violation_count,
                        "weighted_score": round(v.total_weighted, 2),
                    }
                    for k, v in summary.by_principle.items()
                },
                "by_level": {
                    k: {
                        "violations": v.violation_count,
                        "affected_criteria": v.affected_criteria,
                    }
                    for k, v in summary.by_level.items()
                },
            },
            "violations": [v.to_dict() for v in result.violations],
            "remediations": [
                {
                    "sc_id": r.sc_id,
                    "criterion_name": r.criterion_name,
                    "violation_count": r.violation_count,
                    "max_impact": r.max_impact.value,
                    "priority_score": round(r.priority_score, 2),
                    "cognitive_cost_delta": round(r.cognitive_cost_delta, 2),
                    "remediation": r.remediation_hint,
                }
                for r in remediations
            ],
        }
        return json.dumps(doc, indent=2, ensure_ascii=False)

    # -- Markdown -----------------------------------------------------------

    def _format_markdown(self, result: WCAGResult) -> str:
        summary = compute_summary(result)
        remediations = rank_remediations(result)
        lines: List[str] = []

        status = "✅ PASS" if result.is_conformant else "❌ FAIL"
        lines.append(f"# WCAG 2.2 Conformance Report — Level {result.target_level.value}")
        lines.append("")
        lines.append(f"**Status:** {status}")
        lines.append(f"**Page:** {result.page_url or '(not specified)'}")
        lines.append(f"**Criteria tested:** {summary.total_criteria_tested}")
        lines.append(f"**Criteria passed:** {summary.total_criteria_passed} "
                      f"({summary.conformance_ratio:.1%})")
        lines.append(f"**Violations:** {summary.total_violations}")
        lines.append("")

        # Impact breakdown
        lines.append("## Violations by Impact")
        lines.append("")
        lines.append("| Impact | Count |")
        lines.append("|--------|-------|")
        for imp in reversed(list(ImpactLevel)):
            count = summary.by_impact.get(imp.value, 0)
            lines.append(f"| {imp.value.capitalize()} | {count} |")
        lines.append("")

        # Principle breakdown
        lines.append("## Violations by Principle")
        lines.append("")
        lines.append("| Principle | Violations | Weighted Score |")
        lines.append("|-----------|-----------|----------------|")
        for p in WCAGPrinciple:
            ps = summary.by_principle[p.value]
            lines.append(f"| {p.value.capitalize()} | {ps.violation_count} | {ps.total_weighted:.1f} |")
        lines.append("")

        # Remediation priorities
        if remediations:
            lines.append("## Remediation Priorities")
            lines.append("")
            for i, r in enumerate(remediations[:10], 1):
                lines.append(f"### {i}. SC {r.sc_id} — {r.criterion_name}")
                lines.append(f"- **Violations:** {r.violation_count}")
                lines.append(f"- **Max impact:** {r.max_impact.value}")
                lines.append(f"- **Priority score:** {r.priority_score:.1f}")
                lines.append(f"- **Cognitive cost Δ:** {r.cognitive_cost_delta:.1f} bits")
                lines.append(f"- **Recommendation:** {r.remediation_hint}")
                lines.append("")

        return "\n".join(lines)

    # -- HTML ---------------------------------------------------------------

    def _format_html(self, result: WCAGResult) -> str:
        summary = compute_summary(result)
        status = "PASS" if result.is_conformant else "FAIL"
        status_class = "pass" if result.is_conformant else "fail"

        violations_html = ""
        for v in result.violations:
            violations_html += (
                f"<tr>"
                f"<td>{v.sc_id}</td>"
                f"<td>{v.criterion.name}</td>"
                f"<td>{v.impact.value}</td>"
                f"<td>{v.node_id}</td>"
                f"<td>{_escape_html(v.message)}</td>"
                f"<td>{_escape_html(v.remediation)}</td>"
                f"</tr>\n"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WCAG 2.2 Conformance Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem; }}
  .pass {{ color: #22863a; }} .fail {{ color: #cb2431; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
  th {{ background: #f6f8fa; }}
</style>
</head>
<body>
<h1>WCAG 2.2 Level {result.target_level.value} — <span class="{status_class}">{status}</span></h1>
<p><strong>Page:</strong> {result.page_url or 'N/A'}</p>
<p><strong>Criteria tested:</strong> {summary.total_criteria_tested},
   <strong>Passed:</strong> {summary.total_criteria_passed}
   ({summary.conformance_ratio:.1%}),
   <strong>Violations:</strong> {summary.total_violations}</p>

<h2>Violations</h2>
<table>
<thead><tr><th>SC</th><th>Name</th><th>Impact</th><th>Node</th><th>Message</th><th>Remediation</th></tr></thead>
<tbody>
{violations_html}
</tbody>
</table>
</body>
</html>"""

    # -- WCAG-EM (Website Accessibility Conformance Evaluation) -------------

    def _format_wcag_em(self, result: WCAGResult) -> str:
        """Generate a WCAG-EM compliant JSON-LD report.

        Follows the W3C WCAG-EM Report Tool data format.
        """
        summary = compute_summary(result)

        # Build assertion outcomes per criterion
        assertions: List[Dict[str, Any]] = []
        violations_by_sc: Dict[str, List[WCAGViolation]] = defaultdict(list)
        for v in result.violations:
            violations_by_sc[v.sc_id].append(v)

        from usability_oracle.wcag.parser import WCAGXMLParser
        parser = WCAGXMLParser()
        all_criteria = parser.load_criteria()

        for sc in all_criteria:
            if sc.level > result.target_level:
                continue

            sc_violations = violations_by_sc.get(sc.sc_id, [])
            if sc_violations:
                outcome = "earl:failed"
            else:
                # Only mark as passed if we have a checker for it
                outcome = "earl:passed" if sc.sc_id not in violations_by_sc else "earl:cantTell"

            assertion: Dict[str, Any] = {
                "@type": "Assertion",
                "test": {
                    "@type": "TestRequirement",
                    "title": f"SC {sc.sc_id} {sc.name}",
                    "isPartOf": f"https://www.w3.org/TR/WCAG22/",
                },
                "result": {
                    "@type": "TestResult",
                    "outcome": outcome,
                },
                "mode": "earl:automatic",
            }

            if sc_violations:
                assertion["result"]["description"] = sc_violations[0].message
                assertion["result"]["pointer"] = [
                    {"@type": "Pointer", "expression": v.node_id}
                    for v in sc_violations[:5]  # limit pointers
                ]

            assertions.append(assertion)

        wcag_em_doc: Dict[str, Any] = {
            "@context": "https://www.w3.org/TR/WCAG-EM/",
            "@type": "Evaluation",
            "title": f"WCAG 2.2 Level {result.target_level.value} Evaluation",
            "commissioner": "usability_oracle",
            "date": datetime.datetime.utcnow().isoformat() + "Z",
            "scope": {
                "@type": "WebSite",
                "uri": result.page_url or "unknown",
                "conformanceTarget": f"wcag22:{result.target_level.value}",
            },
            "auditResults": assertions,
            "summary": {
                "conformant": result.is_conformant,
                "criteriaEvaluated": summary.total_criteria_tested,
                "criteriaPassed": summary.total_criteria_passed,
                "conformanceRatio": round(summary.conformance_ratio, 4),
            },
        }

        return json.dumps(wcag_em_doc, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _escape_html(s: str) -> str:
    """Minimal HTML escaping."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


__all__ = [
    "LevelStats",
    "PrincipleStats",
    "RemediationItem",
    "ReportSummary",
    "WCAGConformanceReporter",
    "compute_summary",
    "rank_remediations",
]
