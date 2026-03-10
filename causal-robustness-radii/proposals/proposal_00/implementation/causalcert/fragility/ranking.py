"""
Edge ranking and classification for fragility analysis.

Provides convenience functions for sorting, filtering, classifying, and
formatting fragility scores.  Supports:

* Simple ranking by total score
* Classification into severity tiers (critical / important / moderate / cosmetic)
* Load-bearing vs cosmetic edge identification
* Comparison across different conclusion predicates
* Bootstrap-based stability analysis
* Rich text formatting for reports
"""

from __future__ import annotations

import enum
import math
from typing import Any, Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    EdgeTuple,
    FragilityChannel,
    FragilityScore,
    NodeId,
    StructuralEdit,
)


# ---------------------------------------------------------------------------
# Severity tiers
# ---------------------------------------------------------------------------


class EdgeSeverity(enum.Enum):
    """Edge fragility severity classification."""

    CRITICAL = "critical"
    """Score ≥ 0.7 — removing/changing this edge very likely alters the conclusion."""

    IMPORTANT = "important"
    """Score in [0.4, 0.7) — significant impact on the conclusion."""

    MODERATE = "moderate"
    """Score in [0.1, 0.4) — moderate impact."""

    COSMETIC = "cosmetic"
    """Score < 0.1 — negligible impact."""


# Default thresholds
SEVERITY_THRESHOLDS = {
    EdgeSeverity.CRITICAL: 0.7,
    EdgeSeverity.IMPORTANT: 0.4,
    EdgeSeverity.MODERATE: 0.1,
    EdgeSeverity.COSMETIC: 0.0,
}


def classify_edge(
    score: float,
    thresholds: dict[EdgeSeverity, float] | None = None,
) -> EdgeSeverity:
    """Classify an edge by its fragility score.

    Parameters
    ----------
    score : float
        Total fragility score in [0, 1].
    thresholds : dict | None
        Custom severity thresholds.

    Returns
    -------
    EdgeSeverity
    """
    t = thresholds or SEVERITY_THRESHOLDS
    if score >= t.get(EdgeSeverity.CRITICAL, 0.7):
        return EdgeSeverity.CRITICAL
    if score >= t.get(EdgeSeverity.IMPORTANT, 0.4):
        return EdgeSeverity.IMPORTANT
    if score >= t.get(EdgeSeverity.MODERATE, 0.1):
        return EdgeSeverity.MODERATE
    return EdgeSeverity.COSMETIC


# ---------------------------------------------------------------------------
# Ranking functions
# ---------------------------------------------------------------------------


def rank_edges(
    scores: Sequence[FragilityScore],
    descending: bool = True,
) -> list[FragilityScore]:
    """Sort fragility scores by total score.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Unsorted scores.
    descending : bool
        If ``True`` (default), most fragile first.

    Returns
    -------
    list[FragilityScore]
    """
    return sorted(scores, key=lambda s: s.total_score, reverse=descending)


def top_k_fragile(
    scores: Sequence[FragilityScore],
    k: int = 10,
) -> list[FragilityScore]:
    """Return the *k* most fragile edges.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges.
    k : int
        Number of top entries to return.

    Returns
    -------
    list[FragilityScore]
    """
    ranked = rank_edges(scores)
    return ranked[:k]


def bottom_k_robust(
    scores: Sequence[FragilityScore],
    k: int = 10,
) -> list[FragilityScore]:
    """Return the *k* most robust (least fragile) edges.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges.
    k : int
        Number of bottom entries.

    Returns
    -------
    list[FragilityScore]
    """
    ranked = rank_edges(scores, descending=False)
    return ranked[:k]


def load_bearing_edges(
    scores: Sequence[FragilityScore],
    threshold: float = 0.7,
) -> list[FragilityScore]:
    """Identify load-bearing edges (score above threshold).

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges.
    threshold : float
        Minimum score to be considered load-bearing.

    Returns
    -------
    list[FragilityScore]
        Load-bearing edges sorted by decreasing score.
    """
    return rank_edges([s for s in scores if s.total_score >= threshold])


def cosmetic_edges(
    scores: Sequence[FragilityScore],
    threshold: float = 0.1,
) -> list[FragilityScore]:
    """Identify cosmetic edges (score below threshold).

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges.
    threshold : float
        Maximum score to be considered cosmetic.

    Returns
    -------
    list[FragilityScore]
    """
    return rank_edges([s for s in scores if s.total_score < threshold], descending=False)


def classify_all_edges(
    scores: Sequence[FragilityScore],
    thresholds: dict[EdgeSeverity, float] | None = None,
) -> dict[EdgeSeverity, list[FragilityScore]]:
    """Classify all edges into severity tiers.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges.
    thresholds : dict | None
        Custom thresholds.

    Returns
    -------
    dict[EdgeSeverity, list[FragilityScore]]
        Edges grouped by severity.
    """
    classified: dict[EdgeSeverity, list[FragilityScore]] = {
        sev: [] for sev in EdgeSeverity
    }
    for fs in scores:
        sev = classify_edge(fs.total_score, thresholds)
        classified[sev].append(fs)

    # Sort within each tier
    for sev in classified:
        classified[sev] = rank_edges(classified[sev])

    return classified


def severity_counts(
    scores: Sequence[FragilityScore],
    thresholds: dict[EdgeSeverity, float] | None = None,
) -> dict[str, int]:
    """Count edges in each severity tier.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges.
    thresholds : dict | None
        Custom thresholds.

    Returns
    -------
    dict[str, int]
    """
    classified = classify_all_edges(scores, thresholds)
    return {sev.value: len(edges) for sev, edges in classified.items()}


# ---------------------------------------------------------------------------
# Cross-predicate comparison
# ---------------------------------------------------------------------------


def compare_rankings(
    ranking_a: Sequence[FragilityScore],
    ranking_b: Sequence[FragilityScore],
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare two fragility rankings (e.g. for different predicates).

    Parameters
    ----------
    ranking_a, ranking_b : Sequence[FragilityScore]
        Two rankings to compare.
    top_k : int
        Number of top edges to compare.

    Returns
    -------
    dict
        'kendall_tau': rank correlation on common edges
        'top_k_overlap': Jaccard similarity of top-k sets
        'rank_changes': edges with largest rank change
    """
    sorted_a = rank_edges(ranking_a)
    sorted_b = rank_edges(ranking_b)

    edges_a = [s.edge for s in sorted_a]
    edges_b = [s.edge for s in sorted_b]

    # Top-k overlap (Jaccard)
    top_a = set(edges_a[:top_k])
    top_b = set(edges_b[:top_k])
    union = top_a | top_b
    jaccard = len(top_a & top_b) / len(union) if union else 1.0

    # Rank correlation on common edges
    common = set(edges_a) & set(edges_b)
    rank_a = {e: i for i, e in enumerate(edges_a)}
    rank_b = {e: i for i, e in enumerate(edges_b)}

    if len(common) >= 2:
        ranks_a = [rank_a[e] for e in common]
        ranks_b = [rank_b[e] for e in common]
        # Kendall tau (simplified: rank correlation via Spearman as proxy)
        n = len(ranks_a)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff_a = ranks_a[i] - ranks_a[j]
                diff_b = ranks_b[i] - ranks_b[j]
                if diff_a * diff_b > 0:
                    concordant += 1
                elif diff_a * diff_b < 0:
                    discordant += 1
        pairs = concordant + discordant
        tau = (concordant - discordant) / pairs if pairs > 0 else 1.0
    else:
        tau = float("nan")

    # Largest rank changes
    rank_changes: list[dict[str, Any]] = []
    for e in common:
        delta = abs(rank_a[e] - rank_b[e])
        if delta > 0:
            rank_changes.append({
                "edge": e,
                "rank_a": rank_a[e],
                "rank_b": rank_b[e],
                "delta": delta,
            })
    rank_changes.sort(key=lambda x: x["delta"], reverse=True)

    return {
        "kendall_tau": tau,
        "top_k_overlap": jaccard,
        "rank_changes": rank_changes[:top_k],
        "n_common": len(common),
    }


# ---------------------------------------------------------------------------
# Bootstrap stability analysis
# ---------------------------------------------------------------------------


def bootstrap_fragility_stability(
    score_fn: Any,
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    data: Any = None,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Assess stability of fragility rankings via bootstrap resampling.

    Resamples the data (if provided) n_bootstrap times, re-scores, and
    measures rank stability.

    Parameters
    ----------
    score_fn : callable
        Scoring function: ``(adj, treatment, outcome, data) -> list[FragilityScore]``.
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome.
    data : DataFrame | None
        Observational data.
    n_bootstrap : int
        Number of bootstrap iterations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        'mean_scores': per-edge mean score across bootstraps
        'std_scores': per-edge std of scores
        'rank_std': per-edge std of ranks
        'stable_top_5': edges in top-5 in >80% of bootstraps
    """
    import pandas as pd

    rng = np.random.default_rng(seed)

    # Get baseline
    baseline = score_fn(adj, treatment, outcome, data)
    all_edges = [s.edge for s in baseline]

    if not all_edges:
        return {
            "mean_scores": {},
            "std_scores": {},
            "rank_std": {},
            "stable_top_5": [],
        }

    score_matrix: dict[EdgeTuple, list[float]] = {e: [] for e in all_edges}
    rank_matrix: dict[EdgeTuple, list[int]] = {e: [] for e in all_edges}
    top5_counts: dict[EdgeTuple, int] = {e: 0 for e in all_edges}

    for b in range(n_bootstrap):
        if data is not None and isinstance(data, pd.DataFrame):
            # Resample data
            n_obs = len(data)
            idx = rng.choice(n_obs, size=n_obs, replace=True)
            boot_data = data.iloc[idx].reset_index(drop=True)
        else:
            boot_data = data

        try:
            boot_scores = score_fn(adj, treatment, outcome, boot_data)
        except Exception:
            continue

        boot_edge_scores = {s.edge: s.total_score for s in boot_scores}
        boot_ranked = rank_edges(boot_scores)
        boot_ranks = {s.edge: i for i, s in enumerate(boot_ranked)}

        for edge in all_edges:
            score_matrix[edge].append(boot_edge_scores.get(edge, 0.0))
            rank_matrix[edge].append(boot_ranks.get(edge, len(all_edges)))

        top5_set = {s.edge for s in boot_ranked[:5]}
        for e in top5_set:
            if e in top5_counts:
                top5_counts[e] += 1

    mean_scores = {e: float(np.mean(v)) if v else 0.0 for e, v in score_matrix.items()}
    std_scores = {e: float(np.std(v)) if v else 0.0 for e, v in score_matrix.items()}
    rank_std = {e: float(np.std(v)) if v else 0.0 for e, v in rank_matrix.items()}

    threshold = max(1, int(0.8 * n_bootstrap))
    stable_top_5 = sorted(
        [e for e, c in top5_counts.items() if c >= threshold],
        key=lambda e: mean_scores.get(e, 0.0),
        reverse=True,
    )

    return {
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "rank_std": rank_std,
        "stable_top_5": stable_top_5,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_ranking_table(
    scores: Sequence[FragilityScore],
    node_names: list[str] | None = None,
    max_rows: int = 20,
) -> str:
    """Format fragility scores as a human-readable table.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges (pre-ranked).
    node_names : list[str] | None
        Optional node names for display.
    max_rows : int
        Maximum number of rows to show.

    Returns
    -------
    str
        Formatted table string.
    """
    ranked = rank_edges(scores)[:max_rows]

    # Column headers
    header = f"{'Rank':>4}  {'Edge':<20}  {'Total':>7}  {'d-Sep':>7}  {'ID':>7}  {'Est':>7}  {'Severity':<10}"
    separator = "-" * len(header)
    lines = [header, separator]

    for i, fs in enumerate(ranked, 1):
        src, tgt = fs.edge
        if node_names:
            src_name = node_names[src] if src < len(node_names) else str(src)
            tgt_name = node_names[tgt] if tgt < len(node_names) else str(tgt)
            edge_str = f"{src_name} -> {tgt_name}"
        else:
            edge_str = f"{src} -> {tgt}"

        f_dsep = fs.channel_scores.get(FragilityChannel.D_SEPARATION, 0.0)
        f_id = fs.channel_scores.get(FragilityChannel.IDENTIFICATION, 0.0)
        f_est = fs.channel_scores.get(FragilityChannel.ESTIMATION, 0.0)
        sev = classify_edge(fs.total_score)

        line = (
            f"{i:>4}  {edge_str:<20}  {fs.total_score:>7.3f}  "
            f"{f_dsep:>7.3f}  {f_id:>7.3f}  {f_est:>7.3f}  {sev.value:<10}"
        )
        lines.append(line)

    if len(scores) > max_rows:
        lines.append(f"... ({len(scores) - max_rows} more edges)")

    return "\n".join(lines)


def format_severity_summary(
    scores: Sequence[FragilityScore],
) -> str:
    """Format a summary of edge severity distribution.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Scored edges.

    Returns
    -------
    str
        Summary string.
    """
    counts = severity_counts(scores)
    total = sum(counts.values())
    lines = ["Edge Severity Distribution:", "-" * 35]
    for sev_name, count in counts.items():
        pct = 100 * count / total if total > 0 else 0.0
        bar = "█" * int(pct / 5)
        lines.append(f"  {sev_name:<10}  {count:>4}  ({pct:5.1f}%)  {bar}")
    lines.append(f"  {'Total':<10}  {total:>4}")
    return "\n".join(lines)


def edge_report(
    fs: FragilityScore,
    node_names: list[str] | None = None,
) -> str:
    """Generate a detailed single-edge report.

    Parameters
    ----------
    fs : FragilityScore
        Scored edge.
    node_names : list[str] | None
        Node names.

    Returns
    -------
    str
        Multi-line report.
    """
    src, tgt = fs.edge
    if node_names:
        src_name = node_names[src] if src < len(node_names) else str(src)
        tgt_name = node_names[tgt] if tgt < len(node_names) else str(tgt)
    else:
        src_name, tgt_name = str(src), str(tgt)

    sev = classify_edge(fs.total_score)
    lines = [
        f"Edge: {src_name} → {tgt_name}",
        f"  Total Score:    {fs.total_score:.4f}",
        f"  Severity:       {sev.value}",
    ]
    for ch, val in sorted(fs.channel_scores.items(), key=lambda x: x[0].value):
        lines.append(f"  {ch.value:<16} {val:.4f}")

    if fs.witness_ci is not None:
        ci = fs.witness_ci
        lines.append(f"  Witness CI:     ({ci.x}, {ci.y} | {set(ci.conditioning_set)}) "
                      f"p={ci.p_value:.4f}")

    return "\n".join(lines)
