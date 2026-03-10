"""
Structural audit report generation.

Orchestrates the creation of a full audit report from pipeline results,
including fragility rankings, robustness radius, and effect estimates.
Supports multiple output formats (JSON, HTML, LaTeX) and assembles
per-section content from the other reporting sub-modules.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

from causalcert.types import (
    AuditReport,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
    StructuralEdit,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

_FORMAT_GENERATORS: dict[str, Any] = {}


def _ensure_format_registry() -> None:
    """Lazily populate the format registry."""
    if _FORMAT_GENERATORS:
        return
    _FORMAT_GENERATORS["json"] = _generate_json
    _FORMAT_GENERATORS["html"] = _generate_html
    _FORMAT_GENERATORS["latex"] = _generate_latex
    _FORMAT_GENERATORS["narrative"] = _generate_narrative_file


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_audit_report(
    report: AuditReport,
    output_dir: str | Path,
    formats: list[str] | None = None,
    node_names: list[str] | None = None,
) -> dict[str, Path]:
    """Generate audit reports in the requested formats.

    Parameters
    ----------
    report : AuditReport
        The audit report to serialise.
    output_dir : str | Path
        Directory for output files.
    formats : list[str] | None
        Formats to generate: ``"json"``, ``"html"``, ``"latex"``,
        ``"narrative"``.  Defaults to ``["json"]``.
    node_names : list[str] | None
        Human-readable node names.

    Returns
    -------
    dict[str, Path]
        Mapping from format name to output file path.
    """
    _ensure_format_registry()

    if formats is None:
        formats = ["json"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for fmt in formats:
        generator = _FORMAT_GENERATORS.get(fmt)
        if generator is None:
            logger.warning("Unknown format %r, skipping", fmt)
            continue
        try:
            path = generator(report, output_dir, node_names)
            results[fmt] = path
            logger.info("Generated %s report: %s", fmt, path)
        except Exception as exc:
            logger.error("Failed to generate %s report: %s", fmt, exc)

    return results


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def build_header_section(
    report: AuditReport,
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the header section of the audit report.

    Returns
    -------
    dict
        Keys: treatment, outcome, n_nodes, n_edges, timestamp, version.
    """
    tn = node_names[report.treatment] if node_names and report.treatment < len(node_names) else str(report.treatment)
    on = node_names[report.outcome] if node_names and report.outcome < len(node_names) else str(report.outcome)
    return {
        "treatment": report.treatment,
        "treatment_name": tn,
        "outcome": report.outcome,
        "outcome_name": on,
        "n_nodes": report.n_nodes,
        "n_edges": report.n_edges,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": report.metadata.get("version", "0.1.0"),
    }


def build_fragility_section(
    report: AuditReport,
    node_names: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build the fragility section of the audit report.

    Returns
    -------
    dict
        Keys: ranking, n_scored, n_critical, n_cosmetic, load_bearing.
    """
    ranking = report.fragility_ranking[:top_k]

    n_critical = sum(1 for s in report.fragility_ranking if s.total_score >= 0.7)
    n_cosmetic = sum(1 for s in report.fragility_ranking if s.total_score < 0.1)
    load_bearing = [s for s in report.fragility_ranking if s.total_score >= 0.7]

    ranking_data = []
    for i, fs in enumerate(ranking):
        src, tgt = fs.edge
        entry: dict[str, Any] = {
            "rank": i + 1,
            "source": src,
            "target": tgt,
            "total_score": fs.total_score,
            "channel_scores": {ch.value: v for ch, v in fs.channel_scores.items()},
        }
        if node_names:
            entry["source_name"] = node_names[src] if src < len(node_names) else str(src)
            entry["target_name"] = node_names[tgt] if tgt < len(node_names) else str(tgt)
        ranking_data.append(entry)

    return {
        "ranking": ranking_data,
        "n_scored": len(report.fragility_ranking),
        "n_critical": n_critical,
        "n_cosmetic": n_cosmetic,
        "load_bearing": [
            {"edge": s.edge, "score": s.total_score} for s in load_bearing
        ],
    }


def build_radius_section(
    report: AuditReport,
) -> dict[str, Any]:
    """Build the robustness radius section.

    Returns
    -------
    dict
    """
    r = report.radius
    witness_data = []
    for edit in r.witness_edits:
        witness_data.append({
            "edit_type": edit.edit_type.value,
            "source": edit.source,
            "target": edit.target,
        })

    return {
        "lower_bound": r.lower_bound,
        "upper_bound": r.upper_bound,
        "certified": r.certified,
        "gap": r.gap,
        "solver_strategy": r.solver_strategy.value,
        "solver_time_s": r.solver_time_s,
        "witness_edits": witness_data,
    }


def build_estimation_section(
    report: AuditReport,
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the estimation section.

    Returns
    -------
    dict
    """
    baseline = None
    if report.baseline_estimate is not None:
        be = report.baseline_estimate
        baseline = {
            "ate": be.ate,
            "se": be.se,
            "ci_lower": be.ci_lower,
            "ci_upper": be.ci_upper,
            "adjustment_set": sorted(be.adjustment_set),
            "method": be.method,
            "n_obs": be.n_obs,
        }

    perturbed = []
    for pe in report.perturbed_estimates:
        perturbed.append({
            "ate": pe.ate,
            "se": pe.se,
            "ci_lower": pe.ci_lower,
            "ci_upper": pe.ci_upper,
            "adjustment_set": sorted(pe.adjustment_set),
            "method": pe.method,
            "n_obs": pe.n_obs,
        })

    return {
        "baseline": baseline,
        "perturbed_estimates": perturbed,
        "n_perturbed": len(perturbed),
    }


def build_limitations_section(
    report: AuditReport,
) -> dict[str, Any]:
    """Build the honest limitations section.

    Returns
    -------
    dict
    """
    caveats: list[str] = [
        "Robustness radius assumes the faithfulness assumption holds. "
        "Violations may reduce the effective radius.",
        "The back-door criterion is sufficient but not necessary for "
        "identification. Front-door or instrumental variable strategies "
        "are not considered.",
        "Estimation-channel fragility depends on the chosen estimator "
        "and may differ under alternative methods.",
    ]

    if report.radius.gap > 0:
        caveats.append(
            f"The robustness radius has an optimality gap of "
            f"{report.radius.gap:.1%}. The true minimum edit distance "
            f"may be between {report.radius.lower_bound} and "
            f"{report.radius.upper_bound}."
        )

    if not report.baseline_estimate:
        caveats.append(
            "No baseline estimation was performed. Estimation-channel "
            "fragility scores are unavailable."
        )

    power_note = (
        "Statistical power for conditional independence tests depends "
        "on sample size and the number of conditioning variables. "
        "Results should be interpreted with the study's sample size in mind."
    )

    return {
        "caveats": caveats,
        "power_note": power_note,
        "faithfulness_warning": True,
    }


# ---------------------------------------------------------------------------
# Format-specific generators
# ---------------------------------------------------------------------------


def _generate_json(
    report: AuditReport,
    output_dir: Path,
    node_names: list[str] | None = None,
) -> Path:
    """Generate JSON report."""
    from causalcert.reporting.json_report import to_json_report

    content = to_json_report(report, node_names=node_names)
    path = output_dir / "audit_report.json"
    path.write_text(content, encoding="utf-8")
    return path


def _generate_html(
    report: AuditReport,
    output_dir: Path,
    node_names: list[str] | None = None,
) -> Path:
    """Generate HTML report."""
    from causalcert.reporting.html_report import to_html_report

    path = output_dir / "audit_report.html"
    to_html_report(report, output_path=path, node_names=node_names)
    return path


def _generate_latex(
    report: AuditReport,
    output_dir: Path,
    node_names: list[str] | None = None,
) -> Path:
    """Generate LaTeX tables."""
    from causalcert.reporting.latex_report import to_latex_tables

    tables = to_latex_tables(report, node_names=node_names)
    path = output_dir / "audit_tables.tex"
    content_parts = []
    for name, table_str in tables.items():
        content_parts.append(f"% --- {name} ---\n{table_str}\n")
    path.write_text("\n".join(content_parts), encoding="utf-8")
    return path


def _generate_narrative_file(
    report: AuditReport,
    output_dir: Path,
    node_names: list[str] | None = None,
) -> Path:
    """Generate narrative summary."""
    from causalcert.reporting.narrative import generate_narrative

    narrative = generate_narrative(report, node_names=node_names)
    path = output_dir / "audit_narrative.txt"
    path.write_text(narrative, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Report summary (quick text output)
# ---------------------------------------------------------------------------


def format_audit_summary(
    report: AuditReport,
    node_names: list[str] | None = None,
) -> str:
    """Generate a compact text summary of the audit report.

    Parameters
    ----------
    report : AuditReport
        Audit report.
    node_names : list[str] | None
        Node names.

    Returns
    -------
    str
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  CausalCert Structural Robustness Audit")
    lines.append("=" * 60)

    tn = node_names[report.treatment] if node_names and report.treatment < len(node_names) else str(report.treatment)
    on = node_names[report.outcome] if node_names and report.outcome < len(node_names) else str(report.outcome)
    lines.append(f"  Query: {tn} → {on}")
    lines.append(f"  DAG:   {report.n_nodes} nodes, {report.n_edges} edges")
    lines.append("")

    r = report.radius
    cert = "CERTIFIED" if r.certified else f"gap={r.gap:.1%}"
    lines.append(f"  Robustness Radius: [{r.lower_bound}, {r.upper_bound}] ({cert})")
    if r.witness_edits:
        lines.append(f"  Witness: {len(r.witness_edits)} edit(s)")
    lines.append("")

    if report.fragility_ranking:
        n_crit = sum(1 for s in report.fragility_ranking if s.total_score >= 0.7)
        lines.append(f"  Fragility: {len(report.fragility_ranking)} edges scored, "
                      f"{n_crit} critical")
        top = report.fragility_ranking[:3]
        for i, fs in enumerate(top, 1):
            src, tgt = fs.edge
            if node_names:
                sn = node_names[src] if src < len(node_names) else str(src)
                tn2 = node_names[tgt] if tgt < len(node_names) else str(tgt)
            else:
                sn, tn2 = str(src), str(tgt)
            lines.append(f"    {i}. {sn}→{tn2}  score={fs.total_score:.3f}")

    if report.baseline_estimate:
        be = report.baseline_estimate
        lines.append(f"\n  Baseline ATE: {be.ate:.4f} (SE={be.se:.4f})")
        lines.append(f"  95% CI: [{be.ci_lower:.4f}, {be.ci_upper:.4f}]")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
