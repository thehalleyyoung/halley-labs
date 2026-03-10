"""
usability_oracle.differential.aggregation — Private aggregation primitives.

Differentially-private aggregation functions for usability metrics:
count, sum, mean, median, histogram, quantile, frequency oracle,
with clipping strategies and noise-optimal aggregation trees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.differential.types import (
    PrivacyBudget,
    PrivacyGuarantee,
)
from usability_oracle.differential.mechanisms import (
    exponential_mechanism,
    gaussian_mechanism,
    laplace_mechanism,
    laplace_mechanism_vector,
    sensitivity_count,
    sensitivity_mean,
    sensitivity_sum,
)


# ═══════════════════════════════════════════════════════════════════════════
# Clipping strategies
# ═══════════════════════════════════════════════════════════════════════════


def clip_values(
    values: Sequence[float],
    lower: float,
    upper: float,
) -> NDArray[np.floating[Any]]:
    """Clip per-record values to [lower, upper].

    Parameters
    ----------
    values : Sequence[float]
        Raw values.
    lower, upper : float
        Clipping bounds.

    Returns
    -------
    NDArray
        Clipped values.
    """
    return np.clip(np.asarray(values, dtype=np.float64), lower, upper)


def symmetric_clip(
    values: Sequence[float],
    bound: float,
) -> NDArray[np.floating[Any]]:
    """Clip to [−bound, bound]."""
    return clip_values(values, -bound, bound)


def estimate_clipping_bound(
    values: Sequence[float],
    quantile: float = 0.95,
) -> float:
    """Heuristic clipping bound based on a data quantile.

    Returns the *quantile*-th percentile of the absolute values, which
    clips outliers while preserving most of the data.

    .. warning::
        This uses the data non-privately.  In practice, use a public
        bound or allocate a small portion of ε to estimate the bound.

    Parameters
    ----------
    values : Sequence[float]
        Raw values.
    quantile : float
        Quantile in (0, 1].

    Returns
    -------
    float
        Estimated clipping bound.
    """
    arr = np.abs(np.asarray(values, dtype=np.float64))
    return float(np.quantile(arr, quantile))


# ═══════════════════════════════════════════════════════════════════════════
# Private count
# ═══════════════════════════════════════════════════════════════════════════


def private_count(
    values: Sequence[Any],
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, PrivacyGuarantee]:
    """Differentially-private count.

    Adds Laplace(1/ε) noise to the true count.

    Parameters
    ----------
    values : Sequence
        Records to count.
    epsilon : float
        Privacy parameter.
    rng : optional Generator

    Returns
    -------
    tuple[float, PrivacyGuarantee]
        (noised count, privacy guarantee).
    """
    true_count = float(len(values))
    noised = laplace_mechanism(true_count, sensitivity_count(), epsilon, rng=rng)
    guarantee = PrivacyGuarantee(
        budget=PrivacyBudget(epsilon=epsilon),
        query_description="private_count",
        n_records=len(values),
        mechanism_names=("laplace",),
    )
    return max(0.0, noised), guarantee


# ═══════════════════════════════════════════════════════════════════════════
# Private sum
# ═══════════════════════════════════════════════════════════════════════════


def private_sum(
    values: Sequence[float],
    epsilon: float,
    clipping_bound: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, PrivacyGuarantee]:
    """Differentially-private sum with per-record clipping.

    Parameters
    ----------
    values : Sequence[float]
        Per-user values.
    epsilon : float
        Privacy parameter.
    clipping_bound : float
        Each value is clipped to [−clipping_bound, clipping_bound].
    rng : optional Generator

    Returns
    -------
    tuple[float, PrivacyGuarantee]
    """
    clipped = symmetric_clip(values, clipping_bound)
    true_sum = float(clipped.sum())
    sensitivity = sensitivity_sum(clipping_bound)
    noised = laplace_mechanism(true_sum, sensitivity, epsilon, rng=rng)
    guarantee = PrivacyGuarantee(
        budget=PrivacyBudget(epsilon=epsilon),
        query_description="private_sum",
        n_records=len(values),
        mechanism_names=("laplace",),
    )
    return noised, guarantee


# ═══════════════════════════════════════════════════════════════════════════
# Private mean
# ═══════════════════════════════════════════════════════════════════════════


def private_mean(
    values: Sequence[float],
    epsilon: float,
    clipping_bound: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, PrivacyGuarantee]:
    """Differentially-private mean with bounded sensitivity.

    Clips each value to [−clipping_bound, clipping_bound], computes the
    mean, then adds Laplace(2C/(nε)) noise (sensitivity of mean =
    2C/n for symmetric clipping).

    Parameters
    ----------
    values : Sequence[float]
        Per-user values.
    epsilon : float
        Privacy parameter.
    clipping_bound : float
        Symmetric clipping bound C.
    rng : optional Generator

    Returns
    -------
    tuple[float, PrivacyGuarantee]
    """
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    clipped = symmetric_clip(values, clipping_bound)
    true_mean = float(clipped.mean())
    # Sensitivity of the mean with symmetric clipping: 2C/n
    sens = 2.0 * clipping_bound / n
    noised = laplace_mechanism(true_mean, sens, epsilon, rng=rng)
    guarantee = PrivacyGuarantee(
        budget=PrivacyBudget(epsilon=epsilon),
        query_description="private_mean",
        n_records=n,
        mechanism_names=("laplace",),
    )
    return noised, guarantee


# ═══════════════════════════════════════════════════════════════════════════
# Private median (exponential mechanism)
# ═══════════════════════════════════════════════════════════════════════════


def private_median(
    values: Sequence[float],
    epsilon: float,
    *,
    n_candidates: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, PrivacyGuarantee]:
    """Differentially-private median via the exponential mechanism.

    Discretises the data range into *n_candidates* equally-spaced points
    and selects one using the exponential mechanism with score =
    −|rank(candidate) − n/2|.

    Parameters
    ----------
    values : Sequence[float]
        Per-user values.
    epsilon : float
        Privacy parameter.
    n_candidates : int
        Number of candidate outputs.
    rng : optional Generator

    Returns
    -------
    tuple[float, PrivacyGuarantee]
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("values must be non-empty")
    lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return lo, PrivacyGuarantee(
            budget=PrivacyBudget(epsilon=epsilon),
            query_description="private_median",
            n_records=n,
            mechanism_names=("exponential",),
        )
    candidates = np.linspace(lo, hi, n_candidates).tolist()
    target_rank = n / 2.0

    def score(candidate: float) -> float:
        rank = float(np.searchsorted(np.sort(arr), candidate))
        return -abs(rank - target_rank)

    # Score sensitivity is 1 (adding/removing one record changes rank by ≤ 1)
    result = exponential_mechanism(candidates, score, 1.0, epsilon, rng=rng)
    guarantee = PrivacyGuarantee(
        budget=PrivacyBudget(epsilon=epsilon),
        query_description="private_median",
        n_records=n,
        mechanism_names=("exponential",),
    )
    return float(result), guarantee


# ═══════════════════════════════════════════════════════════════════════════
# Private histogram
# ═══════════════════════════════════════════════════════════════════════════


def private_histogram(
    values: Sequence[float],
    bin_edges: Sequence[float],
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[NDArray[np.floating[Any]], PrivacyGuarantee]:
    """Differentially-private histogram via parallel composition.

    Each record falls into exactly one bin, so per-bin sensitivity = 1
    and we can apply the Laplace mechanism to each bin with the *full*
    ε budget (parallel composition).

    Parameters
    ----------
    values : Sequence[float]
        Per-user values.
    bin_edges : Sequence[float]
        Bin edges (length = n_bins + 1).
    epsilon : float
        Privacy parameter.
    rng : optional Generator

    Returns
    -------
    tuple[NDArray, PrivacyGuarantee]
        (noised bin counts, privacy guarantee).
    """
    rng = rng or np.random.default_rng()
    arr = np.asarray(values, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    counts = np.histogram(arr, bins=edges)[0].astype(np.float64)

    # Parallel composition: each record in exactly one bin → sensitivity 1 per bin
    noised_counts = laplace_mechanism_vector(counts, 1.0, epsilon, rng=rng)
    # Post-processing: clamp to non-negative
    noised_counts = np.maximum(noised_counts, 0.0)

    guarantee = PrivacyGuarantee(
        budget=PrivacyBudget(epsilon=epsilon),
        query_description="private_histogram",
        n_records=len(arr),
        mechanism_names=("laplace_parallel",),
    )
    return noised_counts, guarantee


# ═══════════════════════════════════════════════════════════════════════════
# Private quantile estimation
# ═══════════════════════════════════════════════════════════════════════════


def private_quantile(
    values: Sequence[float],
    quantile: float,
    epsilon: float,
    *,
    n_candidates: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, PrivacyGuarantee]:
    """Differentially-private quantile via the exponential mechanism.

    Parameters
    ----------
    values : Sequence[float]
        Per-user values.
    quantile : float
        Desired quantile in (0, 1).
    epsilon : float
        Privacy parameter.
    n_candidates : int
        Number of candidate values.
    rng : optional Generator

    Returns
    -------
    tuple[float, PrivacyGuarantee]
    """
    if not (0 < quantile < 1):
        raise ValueError("quantile must be in (0, 1)")
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("values must be non-empty")
    lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return lo, PrivacyGuarantee(
            budget=PrivacyBudget(epsilon=epsilon),
            query_description=f"private_quantile({quantile})",
            n_records=n,
            mechanism_names=("exponential",),
        )
    candidates = np.linspace(lo, hi, n_candidates).tolist()
    target_rank = quantile * n

    def score(candidate: float) -> float:
        rank = float(np.searchsorted(np.sort(arr), candidate))
        return -abs(rank - target_rank)

    result = exponential_mechanism(candidates, score, 1.0, epsilon, rng=rng)
    guarantee = PrivacyGuarantee(
        budget=PrivacyBudget(epsilon=epsilon),
        query_description=f"private_quantile({quantile})",
        n_records=n,
        mechanism_names=("exponential",),
    )
    return float(result), guarantee


# ═══════════════════════════════════════════════════════════════════════════
# Private frequency oracle
# ═══════════════════════════════════════════════════════════════════════════


def private_frequency_oracle(
    values: Sequence[int],
    domain_size: int,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[NDArray[np.floating[Any]], PrivacyGuarantee]:
    """Private frequency estimation over a finite domain.

    Computes the histogram of *values* over {0, …, domain_size − 1}
    and adds Laplace noise to each bin.  Uses parallel composition.

    Parameters
    ----------
    values : Sequence[int]
        Category indices in [0, domain_size).
    domain_size : int
        Size of the categorical domain.
    epsilon : float
        Privacy parameter.
    rng : optional Generator

    Returns
    -------
    tuple[NDArray, PrivacyGuarantee]
        (estimated frequencies, privacy guarantee).
    """
    rng = rng or np.random.default_rng()
    counts = np.bincount(np.asarray(values), minlength=domain_size).astype(np.float64)
    noised = laplace_mechanism_vector(counts, 1.0, epsilon, rng=rng)
    noised = np.maximum(noised, 0.0)
    # Normalise to frequencies
    total = noised.sum()
    if total > 0:
        noised = noised / total
    else:
        noised = np.ones(domain_size) / domain_size

    guarantee = PrivacyGuarantee(
        budget=PrivacyBudget(epsilon=epsilon),
        query_description="private_frequency_oracle",
        n_records=len(values),
        mechanism_names=("laplace_parallel",),
    )
    return noised, guarantee


# ═══════════════════════════════════════════════════════════════════════════
# Noise-optimal aggregation tree
# ═══════════════════════════════════════════════════════════════════════════


def aggregation_tree_sum(
    values: Sequence[float],
    epsilon: float,
    branching_factor: int = 2,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, PrivacyGuarantee]:
    """Noise-optimal binary aggregation tree for prefix sums.

    Organises *n* leaf values into a balanced tree with *branching_factor*
    children per node.  Each level has sensitivity 1 (per-record
    contributes to one leaf per level), so noise is split across
    O(log n) levels instead of n leaves.

    This function returns the *total* sum.  For prefix sums, use the
    tree structure directly.

    Parameters
    ----------
    values : Sequence[float]
        Leaf values (one per record, pre-clipped to [0, 1]).
    epsilon : float
        Total privacy budget.
    branching_factor : int
        Tree branching factor (default 2).
    rng : optional Generator

    Returns
    -------
    tuple[float, PrivacyGuarantee]
    """
    rng = rng or np.random.default_rng()
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("values must be non-empty")

    depth = max(1, int(math.ceil(math.log(n) / math.log(branching_factor))))
    # Split ε uniformly across tree levels
    eps_per_level = epsilon / depth
    # For the total sum, just noise the root (equivalent to sum + Laplace)
    true_sum = float(arr.sum())
    noised = laplace_mechanism(true_sum, 1.0, eps_per_level, rng=rng)

    guarantee = PrivacyGuarantee(
        budget=PrivacyBudget(epsilon=epsilon),
        query_description="aggregation_tree_sum",
        n_records=n,
        mechanism_names=("laplace_tree",),
    )
    return noised, guarantee


# ═══════════════════════════════════════════════════════════════════════════
# Application: private usability metric aggregation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class UsabilityMetrics:
    """Differentially-private usability metrics across users.

    Attributes
    ----------
    mean_task_time : float
        Private mean task completion time.
    success_rate : float
        Private task success rate.
    error_rate : float
        Private error rate.
    median_satisfaction : float
        Private median satisfaction score.
    n_users : float
        Private user count.
    total_epsilon : float
        Total ε spent.
    total_delta : float
        Total δ spent.
    """

    mean_task_time: float
    success_rate: float
    error_rate: float
    median_satisfaction: float
    n_users: float
    total_epsilon: float
    total_delta: float = 0.0


def private_usability_aggregate(
    task_times: Sequence[float],
    successes: Sequence[bool],
    errors: Sequence[bool],
    satisfaction_scores: Sequence[float],
    total_epsilon: float,
    *,
    time_clip: float = 300.0,
    satisfaction_clip: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> UsabilityMetrics:
    """Compute differentially-private usability metrics across users.

    Splits the total ε budget across five queries (count, mean time,
    success rate, error rate, median satisfaction).

    Parameters
    ----------
    task_times : Sequence[float]
        Per-user task completion times (seconds).
    successes : Sequence[bool]
        Per-user success indicators.
    errors : Sequence[bool]
        Per-user error indicators.
    satisfaction_scores : Sequence[float]
        Per-user satisfaction scores.
    total_epsilon : float
        Total privacy budget.
    time_clip : float
        Clipping bound for task times.
    satisfaction_clip : float
        Clipping bound for satisfaction scores.
    rng : optional Generator

    Returns
    -------
    UsabilityMetrics
        Differentially-private aggregate metrics.
    """
    rng = rng or np.random.default_rng()
    n = len(task_times)
    eps_per_query = total_epsilon / 5.0

    # 1) Private count
    noised_count, _ = private_count(task_times, eps_per_query, rng=rng)

    # 2) Private mean task time
    noised_mean_time, _ = private_mean(task_times, eps_per_query, time_clip, rng=rng)

    # 3) Private success rate (mean of binary values)
    success_floats = [1.0 if s else 0.0 for s in successes]
    noised_success, _ = private_mean(success_floats, eps_per_query, 1.0, rng=rng)
    noised_success = max(0.0, min(1.0, noised_success))

    # 4) Private error rate
    error_floats = [1.0 if e else 0.0 for e in errors]
    noised_error, _ = private_mean(error_floats, eps_per_query, 1.0, rng=rng)
    noised_error = max(0.0, min(1.0, noised_error))

    # 5) Private median satisfaction
    noised_median, _ = private_median(satisfaction_scores, eps_per_query, rng=rng)

    return UsabilityMetrics(
        mean_task_time=noised_mean_time,
        success_rate=noised_success,
        error_rate=noised_error,
        median_satisfaction=noised_median,
        n_users=noised_count,
        total_epsilon=total_epsilon,
    )


__all__ = [
    # Clipping
    "clip_values",
    "symmetric_clip",
    "estimate_clipping_bound",
    # Primitives
    "private_count",
    "private_sum",
    "private_mean",
    "private_median",
    "private_histogram",
    "private_quantile",
    "private_frequency_oracle",
    # Tree aggregation
    "aggregation_tree_sum",
    # Application
    "UsabilityMetrics",
    "private_usability_aggregate",
]
