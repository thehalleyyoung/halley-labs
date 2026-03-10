"""
Score aggregation strategies for combining fragility channels.

Supports multiple aggregation methods: weighted average, max,
product-of-complements, hierarchical, and confidence-weighted.

Each method maps a dictionary of per-channel scores (each in [0, 1]) to a
single aggregate score in [0, 1].  The choice of aggregation can
significantly affect the final ranking and should be reported as part of the
sensitivity analysis.
"""

from __future__ import annotations

import enum
import math
from typing import Any, Sequence

import numpy as np

from causalcert.types import FragilityChannel, FragilityScore


# ---------------------------------------------------------------------------
# Aggregation method enum
# ---------------------------------------------------------------------------


class AggregationMethod(enum.Enum):
    """Available channel-score aggregation methods."""

    WEIGHTED_AVERAGE = "weighted_average"
    """Weighted mean of channel scores."""

    MAX = "max"
    """Maximum of channel scores (conservative / worst-case)."""

    PRODUCT_COMPLEMENT = "product_complement"
    """1 − Π(1 − s_i): probability-style union."""

    HIERARCHICAL = "hierarchical"
    """Check F_id first (binary), then F_dsep, then F_est."""

    CONFIDENCE_WEIGHTED = "confidence_weighted"
    """Weight F_est by estimation confidence."""

    GEOMETRIC_MEAN = "geometric_mean"
    """Geometric mean of (1 + s_i), minus 1."""

    L2_NORM = "l2_norm"
    """L2 norm of channel scores, normalised to [0,1]."""


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[FragilityChannel, float] = {
    FragilityChannel.D_SEPARATION: 0.4,
    FragilityChannel.IDENTIFICATION: 0.4,
    FragilityChannel.ESTIMATION: 0.2,
}


# ---------------------------------------------------------------------------
# Core aggregation function
# ---------------------------------------------------------------------------


def aggregate_scores(
    channel_scores: dict[FragilityChannel, float],
    method: AggregationMethod = AggregationMethod.MAX,
    weights: dict[FragilityChannel, float] | None = None,
    confidence: float | None = None,
) -> float:
    """Aggregate per-channel scores into a single total fragility score.

    Parameters
    ----------
    channel_scores : dict[FragilityChannel, float]
        Per-channel scores in [0, 1].
    method : AggregationMethod
        Aggregation strategy.
    weights : dict[FragilityChannel, float] | None
        Channel weights (used for ``WEIGHTED_AVERAGE`` and
        ``CONFIDENCE_WEIGHTED``).
    confidence : float | None
        Estimation confidence in [0, 1] (used by ``CONFIDENCE_WEIGHTED``).

    Returns
    -------
    float
        Aggregated score in [0, 1].
    """
    if not channel_scores:
        return 0.0

    scores = list(channel_scores.values())

    if method == AggregationMethod.MAX:
        return _aggregate_max(channel_scores)

    elif method == AggregationMethod.WEIGHTED_AVERAGE:
        return _aggregate_weighted(channel_scores, weights)

    elif method == AggregationMethod.PRODUCT_COMPLEMENT:
        return _aggregate_product_complement(channel_scores)

    elif method == AggregationMethod.HIERARCHICAL:
        return _aggregate_hierarchical(channel_scores)

    elif method == AggregationMethod.CONFIDENCE_WEIGHTED:
        return _aggregate_confidence_weighted(
            channel_scores, weights, confidence
        )

    elif method == AggregationMethod.GEOMETRIC_MEAN:
        return _aggregate_geometric_mean(channel_scores)

    elif method == AggregationMethod.L2_NORM:
        return _aggregate_l2_norm(channel_scores)

    else:
        return max(scores)


# ---------------------------------------------------------------------------
# Individual aggregation strategies
# ---------------------------------------------------------------------------


def _aggregate_max(
    channel_scores: dict[FragilityChannel, float],
) -> float:
    """Maximum (worst-case) aggregation.

    Conservative: the overall fragility is determined by the most
    vulnerable channel.
    """
    if not channel_scores:
        return 0.0
    return max(channel_scores.values())


def _aggregate_weighted(
    channel_scores: dict[FragilityChannel, float],
    weights: dict[FragilityChannel, float] | None = None,
) -> float:
    """Weighted average aggregation.

    Parameters
    ----------
    channel_scores : dict[FragilityChannel, float]
        Per-channel scores.
    weights : dict[FragilityChannel, float] | None
        Channel weights.  Defaults to :data:`DEFAULT_WEIGHTS`.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0
    for ch, score in channel_scores.items():
        weight = w.get(ch, 1.0)
        weighted_sum += weight * score
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return _clamp(weighted_sum / total_weight)


def _aggregate_product_complement(
    channel_scores: dict[FragilityChannel, float],
) -> float:
    """Product-of-complements (probability union) aggregation.

    Computes ``1 - prod(1 - s_i)`` which models the score as the
    probability that *at least one* channel is affected, assuming
    independence.
    """
    if not channel_scores:
        return 0.0
    product = 1.0
    for score in channel_scores.values():
        product *= (1.0 - score)
    return _clamp(1.0 - product)


def _aggregate_hierarchical(
    channel_scores: dict[FragilityChannel, float],
) -> float:
    """Hierarchical aggregation.

    Priority order: identification (binary), then d-separation
    (continuous), then estimation.
    - If F_id ≈ 1, return 1.0 immediately
    - Otherwise return max(F_id, F_dsep, F_est)
    """
    f_id = channel_scores.get(FragilityChannel.IDENTIFICATION, 0.0)
    if f_id >= 0.99:
        return 1.0
    f_dsep = channel_scores.get(FragilityChannel.D_SEPARATION, 0.0)
    f_est = channel_scores.get(FragilityChannel.ESTIMATION, 0.0)
    return _clamp(max(f_id, f_dsep, f_est))


def _aggregate_confidence_weighted(
    channel_scores: dict[FragilityChannel, float],
    weights: dict[FragilityChannel, float] | None = None,
    confidence: float | None = None,
) -> float:
    """Confidence-weighted aggregation.

    Like weighted average, but scales the estimation channel by the
    estimation confidence.  A poorly-estimated effect contributes less
    to the aggregate score.

    Parameters
    ----------
    channel_scores : dict
        Per-channel scores.
    weights : dict | None
        Base weights.
    confidence : float | None
        Estimation confidence in [0, 1].  None defaults to 1.0.
    """
    w = dict(weights if weights is not None else DEFAULT_WEIGHTS)
    conf = confidence if confidence is not None else 1.0
    conf = max(0.0, min(1.0, conf))

    # Downweight estimation channel by confidence
    if FragilityChannel.ESTIMATION in w:
        w[FragilityChannel.ESTIMATION] *= conf

    return _aggregate_weighted(channel_scores, w)


def _aggregate_geometric_mean(
    channel_scores: dict[FragilityChannel, float],
) -> float:
    """Geometric mean aggregation.

    Computes (Π(1 + s_i))^(1/n) - 1.  Rewards consistent fragility
    across all channels.
    """
    if not channel_scores:
        return 0.0
    n = len(channel_scores)
    product = 1.0
    for score in channel_scores.values():
        product *= (1.0 + score)
    return _clamp(product ** (1.0 / n) - 1.0)


def _aggregate_l2_norm(
    channel_scores: dict[FragilityChannel, float],
) -> float:
    """L2 norm aggregation normalised by sqrt(n).

    Euclidean distance from zero in channel-score space, divided by
    sqrt(n) so that the result lies in [0, 1].
    """
    if not channel_scores:
        return 0.0
    n = len(channel_scores)
    ss = sum(s ** 2 for s in channel_scores.values())
    return _clamp(math.sqrt(ss / n))


# ---------------------------------------------------------------------------
# Batch aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_fragility_scores(
    scores: Sequence[FragilityScore],
    method: AggregationMethod = AggregationMethod.MAX,
    weights: dict[FragilityChannel, float] | None = None,
) -> list[FragilityScore]:
    """Re-aggregate a list of FragilityScore objects under a new method.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Existing scored edges.
    method : AggregationMethod
        New aggregation method.
    weights : dict[FragilityChannel, float] | None
        New weights.

    Returns
    -------
    list[FragilityScore]
        New scores with updated total_score.
    """
    results: list[FragilityScore] = []
    for fs in scores:
        new_total = aggregate_scores(fs.channel_scores, method, weights)
        results.append(FragilityScore(
            edge=fs.edge,
            total_score=new_total,
            channel_scores=fs.channel_scores,
            witness_ci=fs.witness_ci,
        ))
    return results


def sensitivity_analysis(
    scores: Sequence[FragilityScore],
    methods: Sequence[AggregationMethod] | None = None,
) -> dict[str, list[FragilityScore]]:
    """Run sensitivity analysis across multiple aggregation methods.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Base scored edges.
    methods : Sequence[AggregationMethod] | None
        Methods to test.  Defaults to all available methods.

    Returns
    -------
    dict[str, list[FragilityScore]]
        Mapping from method name to re-aggregated scores.
    """
    if methods is None:
        methods = list(AggregationMethod)

    result: dict[str, list[FragilityScore]] = {}
    for method in methods:
        re_agg = aggregate_fragility_scores(scores, method)
        re_agg.sort(key=lambda s: s.total_score, reverse=True)
        result[method.value] = re_agg
    return result


def rank_stability(
    sensitivity_results: dict[str, list[FragilityScore]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Measure ranking stability across aggregation methods.

    Parameters
    ----------
    sensitivity_results : dict[str, list[FragilityScore]]
        Output of :func:`sensitivity_analysis`.
    top_k : int
        Number of top edges to compare.

    Returns
    -------
    dict
        'overlap_matrix': pairwise Jaccard overlap of top-k edges
        'stable_edges': edges in top-k for ALL methods
        'unstable_edges': edges that move significantly across methods
    """
    methods = list(sensitivity_results.keys())
    n_methods = len(methods)

    # Top-k edge sets per method
    top_sets: dict[str, set[tuple[int, int]]] = {}
    for method, scores in sensitivity_results.items():
        top_sets[method] = {s.edge for s in scores[:top_k]}

    # Pairwise Jaccard overlap
    overlap = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            si = top_sets[methods[i]]
            sj = top_sets[methods[j]]
            union = si | sj
            if union:
                overlap[i, j] = len(si & sj) / len(union)
            else:
                overlap[i, j] = 1.0

    # Stable edges: in top-k for all methods
    if top_sets:
        stable = set.intersection(*top_sets.values())
    else:
        stable = set()

    # Unstable: in top-k for some but not all
    all_top_edges: set[tuple[int, int]] = set()
    for s in top_sets.values():
        all_top_edges |= s
    unstable = all_top_edges - stable

    return {
        "overlap_matrix": overlap.tolist(),
        "methods": methods,
        "stable_edges": sorted(stable),
        "unstable_edges": sorted(unstable),
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, value))
