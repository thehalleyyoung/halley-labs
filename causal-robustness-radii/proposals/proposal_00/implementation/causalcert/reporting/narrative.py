"""
Plain-language narrative generation for audit reports.

Produces human-readable summaries of the robustness analysis, suitable
for inclusion in applied research papers and policy documents.

The narrative is template-based: each section generates a paragraph
describing one aspect of the analysis, and the sections are assembled
into a coherent multi-paragraph summary.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from causalcert.types import (
    AuditReport,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_narrative(
    report: AuditReport,
    node_names: list[str] | None = None,
    include_recommendations: bool = True,
) -> str:
    """Generate a plain-language narrative summary.

    Parameters
    ----------
    report : AuditReport
        Audit report.
    node_names : list[str] | None
        Node names for human-readable output.
    include_recommendations : bool
        If True, include actionable recommendations.

    Returns
    -------
    str
        Multi-paragraph narrative.
    """
    names = _resolve_names(report, node_names)

    paragraphs: list[str] = []
    paragraphs.append(_describe_overview(report, names))
    paragraphs.append(_describe_radius(report, names))
    paragraphs.append(_describe_fragility(report, names))

    if report.baseline_estimate is not None:
        paragraphs.append(_describe_estimates(report, names))

    paragraphs.append(_describe_limitations(report, names))

    if include_recommendations:
        paragraphs.append(_describe_recommendations(report, names))

    return "\n\n".join(p for p in paragraphs if p)


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


def _resolve_names(
    report: AuditReport,
    node_names: list[str] | None,
) -> list[str]:
    """Resolve node names, falling back to string IDs."""
    if node_names is not None:
        return list(node_names)
    max_id = max(report.treatment, report.outcome, report.n_nodes - 1)
    return [f"X{i}" for i in range(max_id + 1)]


def _tn(report: AuditReport, names: list[str]) -> str:
    """Treatment name."""
    return names[report.treatment] if report.treatment < len(names) else str(report.treatment)


def _on(report: AuditReport, names: list[str]) -> str:
    """Outcome name."""
    return names[report.outcome] if report.outcome < len(names) else str(report.outcome)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def _describe_overview(report: AuditReport, names: list[str]) -> str:
    """Generate the overview paragraph."""
    tn = _tn(report, names)
    on = _on(report, names)
    return (
        f"STRUCTURAL ROBUSTNESS AUDIT\n"
        f"{'=' * 40}\n\n"
        f"This report summarises the structural robustness of the causal "
        f"conclusion that {tn} has a causal effect on {on}. The analysis "
        f"was conducted on a directed acyclic graph (DAG) with "
        f"{report.n_nodes} variables and {report.n_edges} directed edges. "
        f"We systematically assess how sensitive this conclusion is to "
        f"possible misspecifications in the assumed causal structure."
    )


def _describe_radius(report: AuditReport, names: list[str]) -> str:
    """Generate the radius description paragraph."""
    tn = _tn(report, names)
    on = _on(report, names)
    r = report.radius

    if r.certified:
        cert_phrase = (
            f"The robustness radius is exactly {r.lower_bound}. This means "
            f"that at least {r.lower_bound} edge modification(s) to the DAG "
            f"are required before the causal conclusion could potentially "
            f"be overturned, and we have certified that this bound is tight."
        )
    else:
        cert_phrase = (
            f"The robustness radius lies between {r.lower_bound} (lower "
            f"bound) and {r.upper_bound} (upper bound), with an optimality "
            f"gap of {r.gap:.1%}. This means at least {r.lower_bound} edge "
            f"modification(s) are needed to potentially overturn the "
            f"conclusion."
        )

    interpretation = _interpret_radius(r.lower_bound, report.n_edges)

    witness_phrase = ""
    if r.witness_edits:
        n_edits = len(r.witness_edits)
        witness_phrase = (
            f" We identified a specific set of {n_edits} edit(s) that would "
            f"change the conclusion."
        )

    return (
        f"ROBUSTNESS RADIUS\n"
        f"{'-' * 40}\n\n"
        f"The causal conclusion that {tn} affects {on} is robust to at "
        f"least {r.lower_bound} edge modification(s) of the DAG. "
        f"{cert_phrase} {interpretation}{witness_phrase}"
    )


def _interpret_radius(radius: int, n_edges: int) -> str:
    """Generate an interpretation of the radius value."""
    if n_edges == 0:
        return ""
    ratio = radius / max(n_edges, 1)
    if radius == 0:
        return (
            "A radius of 0 indicates that the conclusion does not hold "
            "even under the given DAG. This suggests the assumed causal "
            "structure may already be inconsistent with the data."
        )
    elif ratio > 0.3:
        return (
            f"This represents {ratio:.0%} of the total edges in the graph, "
            f"suggesting high structural robustness."
        )
    elif ratio > 0.1:
        return (
            f"This represents {ratio:.0%} of the total edges, indicating "
            f"moderate robustness."
        )
    else:
        return (
            f"This represents only {ratio:.0%} of the total edges, "
            f"suggesting the conclusion is sensitive to structural "
            f"assumptions."
        )


def _describe_fragility(report: AuditReport, names: list[str]) -> str:
    """Generate the fragility description paragraph."""
    ranking = report.fragility_ranking
    if not ranking:
        return (
            "FRAGILITY ANALYSIS\n"
            f"{'-' * 40}\n\n"
            "No fragility analysis was performed."
        )

    n_scored = len(ranking)
    n_critical = sum(1 for s in ranking if s.total_score >= 0.7)
    n_important = sum(1 for s in ranking if 0.4 <= s.total_score < 0.7)
    n_cosmetic = sum(1 for s in ranking if s.total_score < 0.1)

    parts: list[str] = [
        f"FRAGILITY ANALYSIS\n{'-' * 40}\n\n"
        f"We scored {n_scored} potential edge modifications for fragility. "
    ]

    if n_critical > 0:
        parts.append(
            f"We identified {n_critical} critical edge(s) whose "
            f"modification could substantially alter the conclusion. "
        )

    if n_important > 0:
        parts.append(
            f"Additionally, {n_important} edge(s) were classified as "
            f"important (moderate impact on the conclusion). "
        )

    if n_cosmetic > 0:
        parts.append(
            f"{n_cosmetic} edge(s) were classified as cosmetic "
            f"(negligible impact). "
        )

    # Describe top edges
    top = ranking[:3]
    if top:
        parts.append("\nThe most fragile edges are:\n")
        for i, fs in enumerate(top, 1):
            src, tgt = fs.edge
            src_name = names[src] if src < len(names) else str(src)
            tgt_name = names[tgt] if tgt < len(names) else str(tgt)
            parts.append(
                f"  {i}. {src_name} → {tgt_name} "
                f"(score: {fs.total_score:.3f})\n"
            )

    # Channel interpretation
    if top:
        best = top[0]
        dsep = best.channel_scores.get(FragilityChannel.D_SEPARATION, 0.0)
        ident = best.channel_scores.get(FragilityChannel.IDENTIFICATION, 0.0)
        est = best.channel_scores.get(FragilityChannel.ESTIMATION, 0.0)

        dominant = max(
            [(dsep, "d-separation relations"), (ident, "causal identifiability"),
             (est, "the numerical effect estimate")],
            key=lambda x: x[0],
        )
        if dominant[0] > 0.01:
            src, tgt = best.edge
            src_name = names[src] if src < len(names) else str(src)
            tgt_name = names[tgt] if tgt < len(names) else str(tgt)
            parts.append(
                f"\nYour conclusion depends most critically on the edge "
                f"{src_name} → {tgt_name}, primarily through its impact "
                f"on {dominant[1]}."
            )

    return "".join(parts)


def _describe_estimates(report: AuditReport, names: list[str]) -> str:
    """Generate the estimation comparison paragraph."""
    tn = _tn(report, names)
    on = _on(report, names)
    be = report.baseline_estimate
    if be is None:
        return ""

    parts: list[str] = [
        f"EFFECT ESTIMATES\n{'-' * 40}\n\n"
        f"Under the original DAG, the estimated average treatment effect "
        f"of {tn} on {on} is {be.ate:.4f} (SE = {be.se:.4f}, "
        f"95%% CI: [{be.ci_lower:.4f}, {be.ci_upper:.4f}]). "
    ]

    if report.perturbed_estimates:
        ates = [pe.ate for pe in report.perturbed_estimates]
        min_ate = min(ates)
        max_ate = max(ates)
        mean_ate = sum(ates) / len(ates)

        parts.append(
            f"Under the {len(report.perturbed_estimates)} perturbed DAGs "
            f"considered, the ATE ranges from {min_ate:.4f} to "
            f"{max_ate:.4f} (mean: {mean_ate:.4f}). "
        )

        if be.ate != 0:
            max_change = max(abs(pe.ate - be.ate) for pe in report.perturbed_estimates)
            rel_change = max_change / abs(be.ate) * 100
            parts.append(
                f"The largest relative change is {rel_change:.1f}%% from "
                f"the baseline estimate."
            )

        # Sign change check
        sign_changes = sum(
            1 for pe in report.perturbed_estimates
            if (pe.ate > 0) != (be.ate > 0)
        )
        if sign_changes > 0:
            parts.append(
                f" Notably, {sign_changes} perturbed DAG(s) reverse the "
                f"sign of the effect estimate, suggesting the direction "
                f"of the effect is sensitive to structural assumptions."
            )

    return "".join(parts)


def _describe_limitations(report: AuditReport, names: list[str]) -> str:
    """Generate the limitations paragraph."""
    parts: list[str] = [
        f"LIMITATIONS AND CAVEATS\n{'-' * 40}\n\n"
        f"This analysis should be interpreted with the following caveats:\n\n"
    ]

    parts.append(
        "1. FAITHFULNESS ASSUMPTION: The analysis assumes that the true "
        "data-generating process is faithful to the DAG, i.e., that all "
        "and only the conditional independences implied by d-separation "
        "hold in the population. Violations of faithfulness (e.g., due to "
        "exact parameter cancellations) could affect the results.\n\n"
    )

    parts.append(
        "2. IDENTIFICATION STRATEGY: We consider only the back-door "
        "criterion for causal identification. Alternative strategies "
        "(front-door, instrumental variables) are not assessed.\n\n"
    )

    if report.radius.gap > 0:
        parts.append(
            f"3. OPTIMALITY GAP: The robustness radius has a gap of "
            f"{report.radius.gap:.1%}. The true minimum edit distance "
            f"may lie between {report.radius.lower_bound} and "
            f"{report.radius.upper_bound}.\n\n"
        )

    parts.append(
        "4. STATISTICAL POWER: The power of conditional independence "
        "tests depends on sample size, the number of conditioning "
        "variables, and the true effect sizes. Small samples may lead "
        "to inconclusive or misleading results."
    )

    return "".join(parts)


def _describe_recommendations(report: AuditReport, names: list[str]) -> str:
    """Generate actionable recommendations."""
    tn = _tn(report, names)
    on = _on(report, names)

    parts: list[str] = [
        f"RECOMMENDATIONS\n{'-' * 40}\n\n"
    ]

    r = report.radius
    if r.lower_bound == 0:
        parts.append(
            "• URGENT: The conclusion does not appear robust. Consider "
            "re-evaluating the assumed DAG structure or gathering "
            "additional domain knowledge.\n"
        )
    elif r.lower_bound <= 2:
        parts.append(
            "• CAUTION: The robustness radius is small. Verify the most "
            "fragile edges with domain experts before relying on this "
            "conclusion for policy decisions.\n"
        )
    else:
        parts.append(
            "• The conclusion appears structurally robust. Nonetheless, "
            "consider sensitivity analyses on the most fragile edges.\n"
        )

    ranking = report.fragility_ranking
    if ranking:
        n_critical = sum(1 for s in ranking if s.total_score >= 0.7)
        if n_critical > 0:
            parts.append(
                f"• {n_critical} critical edge(s) identified. Consider "
                f"conducting targeted experiments or collecting additional "
                f"data to confirm these structural assumptions.\n"
            )

        top = ranking[:3]
        if top:
            for fs in top:
                src, tgt = fs.edge
                src_name = names[src] if src < len(names) else str(src)
                tgt_name = names[tgt] if tgt < len(names) else str(tgt)
                parts.append(
                    f"• Edge {src_name} → {tgt_name}: Consider whether "
                    f"domain knowledge supports this edge. An intervention "
                    f"or natural experiment targeting this relationship "
                    f"would strengthen the analysis.\n"
                )

    return "".join(parts)
