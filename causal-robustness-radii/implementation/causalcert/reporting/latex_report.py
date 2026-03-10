"""
LaTeX table generation for academic papers.

Produces publication-ready LaTeX tables for the fragility ranking,
robustness radius summary, and effect estimates.  All tables use the
``booktabs`` package for clean horizontal rules.
"""

from __future__ import annotations

import math
from typing import Sequence

from causalcert.types import (
    AuditReport,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_latex_tables(
    report: AuditReport,
    node_names: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, str]:
    """Generate LaTeX tables from an audit report.

    Parameters
    ----------
    report : AuditReport
        Audit report.
    node_names : list[str] | None
        Node names for labelling.
    top_k : int
        Number of fragility ranking rows.

    Returns
    -------
    dict[str, str]
        Mapping from table name to LaTeX string.
        Keys: ``"fragility"``, ``"radius"``, ``"estimates"``,
        ``"nearby_dags"``.
    """
    tables: dict[str, str] = {}

    tables["fragility"] = fragility_table(
        report.fragility_ranking, node_names=node_names, top_k=top_k
    )
    tables["radius"] = radius_summary_table(report.radius)
    tables["estimates"] = estimates_table(report, node_names=node_names)

    if report.radius.witness_edits:
        tables["nearby_dags"] = witness_table(
            report.radius.witness_edits, node_names=node_names
        )

    return tables


# ---------------------------------------------------------------------------
# Fragility table
# ---------------------------------------------------------------------------


def fragility_table(
    scores: Sequence[FragilityScore],
    node_names: list[str] | None = None,
    top_k: int = 10,
    caption: str = "Per-edge fragility scores (top edges by composite score).",
    label: str = "tab:fragility",
) -> str:
    """Generate a LaTeX table of the top-k fragile edges.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Fragility scores (sorted).
    node_names : list[str] | None
        Node names.
    top_k : int
        Number of rows.
    caption : str
        Table caption.
    label : str
        LaTeX label.

    Returns
    -------
    str
        LaTeX tabular environment string.
    """
    shown = list(scores)[:top_k]
    lines: list[str] = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{_escape_latex(caption)}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"  \begin{tabular}{clcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Rank & Edge & $F_{\text{total}}$ & "
                 r"$F_{\text{d-sep}}$ & $F_{\text{id}}$ & $F_{\text{est}}$ \\")
    lines.append(r"    \midrule")

    for i, fs in enumerate(shown, 1):
        src, tgt = fs.edge
        if node_names:
            src_name = _escape_latex(
                node_names[src] if src < len(node_names) else str(src)
            )
            tgt_name = _escape_latex(
                node_names[tgt] if tgt < len(node_names) else str(tgt)
            )
            edge_str = f"{src_name} $\\to$ {tgt_name}"
        else:
            edge_str = f"{src} $\\to$ {tgt}"

        f_dsep = fs.channel_scores.get(FragilityChannel.D_SEPARATION, 0.0)
        f_id = fs.channel_scores.get(FragilityChannel.IDENTIFICATION, 0.0)
        f_est = fs.channel_scores.get(FragilityChannel.ESTIMATION, 0.0)

        # Bold critical scores
        total_fmt = _format_score(fs.total_score, bold_threshold=0.7)
        dsep_fmt = _format_score(f_dsep)
        id_fmt = _format_score(f_id)
        est_fmt = _format_score(f_est)

        lines.append(
            f"    {i} & {edge_str} & {total_fmt} & "
            f"{dsep_fmt} & {id_fmt} & {est_fmt} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Radius summary table
# ---------------------------------------------------------------------------


def radius_summary_table(
    radius: RobustnessRadius,
    caption: str = "Robustness radius summary.",
    label: str = "tab:radius",
) -> str:
    """Generate a LaTeX table summarising the robustness radius.

    Parameters
    ----------
    radius : RobustnessRadius
    caption : str
        Table caption.
    label : str
        LaTeX label.

    Returns
    -------
    str
        LaTeX string.
    """
    lines: list[str] = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{_escape_latex(caption)}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \toprule")
    lines.append(r"    Property & Value \\")
    lines.append(r"    \midrule")

    lines.append(f"    Lower bound & {radius.lower_bound} \\\\")
    lines.append(f"    Upper bound & {radius.upper_bound} \\\\")

    cert_str = "Yes" if radius.certified else "No"
    lines.append(f"    Certified & {cert_str} \\\\")

    if radius.gap > 0:
        lines.append(f"    Optimality gap & {radius.gap:.1%} \\\\")

    lines.append(
        f"    Solver & {_escape_latex(radius.solver_strategy.value)} \\\\"
    )
    lines.append(f"    Solver time & {radius.solver_time_s:.2f}\\,s \\\\")
    lines.append(
        f"    Witness size & {len(radius.witness_edits)} edit(s) \\\\"
    )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Estimates table
# ---------------------------------------------------------------------------


def estimates_table(
    report: AuditReport,
    node_names: list[str] | None = None,
    caption: str = "Causal effect estimates under original and perturbed DAGs.",
    label: str = "tab:estimates",
) -> str:
    """Generate a LaTeX table of treatment effect estimates.

    Parameters
    ----------
    report : AuditReport
    node_names : list[str] | None
    caption, label : str
        Table metadata.

    Returns
    -------
    str
    """
    lines: list[str] = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{_escape_latex(caption)}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    DAG & ATE & SE & 95\% CI & Method \\")
    lines.append(r"    \midrule")

    if report.baseline_estimate is not None:
        be = report.baseline_estimate
        ci_str = f"[{be.ci_lower:.3f}, {be.ci_upper:.3f}]"
        lines.append(
            f"    Original & {be.ate:.4f} & {be.se:.4f} & "
            f"{ci_str} & {_escape_latex(be.method)} \\\\"
        )

    for i, pe in enumerate(report.perturbed_estimates, 1):
        ci_str = f"[{pe.ci_lower:.3f}, {pe.ci_upper:.3f}]"
        lines.append(
            f"    Perturbed {i} & {pe.ate:.4f} & {pe.se:.4f} & "
            f"{ci_str} & {_escape_latex(pe.method)} \\\\"
        )

    if not report.baseline_estimate and not report.perturbed_estimates:
        lines.append(r"    \multicolumn{5}{c}{\emph{No estimates available}} \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Witness edits table
# ---------------------------------------------------------------------------


def witness_table(
    witness_edits: Sequence[object],
    node_names: list[str] | None = None,
    caption: str = "Witness edit set (closest violating perturbation).",
    label: str = "tab:witness",
) -> str:
    """Generate a LaTeX table of witness edits.

    Parameters
    ----------
    witness_edits : Sequence[StructuralEdit]
    node_names : list[str] | None
    caption, label : str

    Returns
    -------
    str
    """
    lines: list[str] = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{_escape_latex(caption)}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"  \begin{tabular}{cll}")
    lines.append(r"    \toprule")
    lines.append(r"    \# & Edit Type & Edge \\")
    lines.append(r"    \midrule")

    for i, edit in enumerate(witness_edits, 1):
        etype = _escape_latex(edit.edit_type.value)  # type: ignore[union-attr]
        src = edit.source  # type: ignore[union-attr]
        tgt = edit.target  # type: ignore[union-attr]
        if node_names:
            src_name = _escape_latex(
                node_names[src] if src < len(node_names) else str(src)
            )
            tgt_name = _escape_latex(
                node_names[tgt] if tgt < len(node_names) else str(tgt)
            )
            edge_str = f"{src_name} $\\to$ {tgt_name}"
        else:
            edge_str = f"{src} $\\to$ {tgt}"

        lines.append(f"    {i} & {etype} & {edge_str} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    result = text
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    return result


def _format_score(
    score: float,
    decimals: int = 3,
    bold_threshold: float | None = None,
) -> str:
    """Format a score value for LaTeX.

    Parameters
    ----------
    score : float
    decimals : int
    bold_threshold : float | None
        If given, bold scores above this threshold.
    """
    fmt = f"{score:.{decimals}f}"
    if bold_threshold is not None and score >= bold_threshold:
        return f"\\textbf{{{fmt}}}"
    return fmt
