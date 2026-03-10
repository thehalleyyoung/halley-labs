"""
usability_oracle.differential.mechanisms — Differential privacy mechanisms.

Concrete noise mechanisms for protecting usability data:
Laplace, Gaussian, Exponential, Randomized Response, Geometric,
Report-Noisy-Max, Sparse Vector, and Above Threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from usability_oracle.differential.types import (
    MechanismType,
    NoiseConfig,
    PrivacyBudget,
)


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity helpers
# ═══════════════════════════════════════════════════════════════════════════


def sensitivity_count() -> float:
    """Global sensitivity of a counting query (adding/removing one record)."""
    return 1.0


def sensitivity_sum(clipping_bound: float) -> float:
    """Global sensitivity of a sum query with per-record clipping.

    Parameters
    ----------
    clipping_bound : float
        Maximum absolute contribution of a single record.
    """
    return abs(clipping_bound)


def sensitivity_mean(clipping_bound: float, n: int) -> float:
    """Global sensitivity of a mean query.

    Parameters
    ----------
    clipping_bound : float
        Maximum absolute value of a single record.
    n : int
        Number of records.  Must be > 0.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return clipping_bound / n


def sensitivity_median(clipping_bound: float) -> float:
    """Global sensitivity of the median (used with exponential mechanism).

    For the exponential mechanism on sorted data the score sensitivity is 1,
    but the *output* sensitivity equals the data range when computed directly.
    We return 1.0 here because the exponential mechanism scores individual
    candidates and the score function (rank-based) has sensitivity 1.
    """
    return 1.0


def l2_sensitivity(l1_sensitivity: float, dimensionality: int = 1) -> float:
    """Convert L1 sensitivity to L2 sensitivity (worst-case bound).

    Parameters
    ----------
    l1_sensitivity : float
        L1 global sensitivity.
    dimensionality : int
        Output dimensionality.  For a scalar query, use 1.
    """
    return l1_sensitivity / math.sqrt(dimensionality) if dimensionality > 1 else l1_sensitivity


# ═══════════════════════════════════════════════════════════════════════════
# Laplace mechanism
# ═══════════════════════════════════════════════════════════════════════════


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Compute Laplace noise scale b = Δf / ε.

    Parameters
    ----------
    sensitivity : float
        Global L1 sensitivity Δf.
    epsilon : float
        Privacy parameter ε > 0.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    return sensitivity / epsilon


def laplace_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Apply the Laplace mechanism to a numeric value.

    Satisfies (ε, 0)-differential privacy.

    Parameters
    ----------
    value : float
        True query answer.
    sensitivity : float
        Global L1 sensitivity.
    epsilon : float
        Privacy parameter.
    rng : optional Generator
        Numpy random generator.

    Returns
    -------
    float
        Noised answer.
    """
    rng = rng or np.random.default_rng()
    scale = laplace_scale(sensitivity, epsilon)
    return float(value + rng.laplace(0.0, scale))


def laplace_mechanism_vector(
    values: NDArray[np.floating[Any]],
    sensitivity: float,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating[Any]]:
    """Apply the Laplace mechanism independently to each element."""
    rng = rng or np.random.default_rng()
    scale = laplace_scale(sensitivity, epsilon)
    noise = rng.laplace(0.0, scale, size=values.shape)
    return values + noise


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian mechanism
# ═══════════════════════════════════════════════════════════════════════════


def gaussian_scale(
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> float:
    """Compute Gaussian noise σ for (ε, δ)-DP via the analytic Gaussian mechanism.

    σ = Δ₂f · √(2 ln(1.25/δ)) / ε

    Parameters
    ----------
    sensitivity : float
        Global L2 sensitivity Δ₂f.
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ ∈ (0, 1).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def gaussian_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    delta: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Apply the Gaussian mechanism to a numeric value.

    Satisfies (ε, δ)-differential privacy.

    Parameters
    ----------
    value : float
        True query answer.
    sensitivity : float
        Global L2 sensitivity.
    epsilon : float
        Privacy parameter.
    delta : float
        Failure probability.
    rng : optional Generator
        Numpy random generator.

    Returns
    -------
    float
        Noised answer.
    """
    rng = rng or np.random.default_rng()
    sigma = gaussian_scale(sensitivity, epsilon, delta)
    return float(value + rng.normal(0.0, sigma))


def gaussian_mechanism_vector(
    values: NDArray[np.floating[Any]],
    sensitivity: float,
    epsilon: float,
    delta: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating[Any]]:
    """Apply the Gaussian mechanism independently to each element."""
    rng = rng or np.random.default_rng()
    sigma = gaussian_scale(sensitivity, epsilon, delta)
    noise = rng.normal(0.0, sigma, size=values.shape)
    return values + noise


# ═══════════════════════════════════════════════════════════════════════════
# Exponential mechanism
# ═══════════════════════════════════════════════════════════════════════════


def exponential_mechanism(
    candidates: Sequence[Any],
    score_fn: Callable[[Any], float],
    sensitivity: float,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """Select an element from *candidates* proportional to exp(ε · score / 2Δ).

    Satisfies (ε, 0)-differential privacy.

    Parameters
    ----------
    candidates : Sequence
        Set of possible outputs.
    score_fn : callable
        Quality score function  u(x, r) → ℝ  for each candidate *r*.
    sensitivity : float
        Global sensitivity of *score_fn* (Δu).
    epsilon : float
        Privacy parameter ε > 0.
    rng : optional Generator
        Numpy random generator.

    Returns
    -------
    Any
        Selected candidate.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    rng = rng or np.random.default_rng()

    scores = np.array([score_fn(c) for c in candidates], dtype=np.float64)
    # Numerical stability: subtract max before exponentiating
    log_weights = (epsilon / (2.0 * sensitivity)) * scores
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    probabilities = weights / weights.sum()

    idx = int(rng.choice(len(candidates), p=probabilities))
    return candidates[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Randomized response
# ═══════════════════════════════════════════════════════════════════════════


def randomized_response(
    true_value: bool,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> bool:
    """Randomized response for a single binary value.

    Reports the true value with probability  p = e^ε / (1 + e^ε),
    and flips it otherwise.  Satisfies (ε, 0)-DP.

    Parameters
    ----------
    true_value : bool
        The user's true answer.
    epsilon : float
        Privacy parameter ε > 0.
    rng : optional Generator
        Numpy random generator.

    Returns
    -------
    bool
        Perturbed answer.
    """
    rng = rng or np.random.default_rng()
    p_truth = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    if rng.random() < p_truth:
        return true_value
    return not true_value


def randomized_response_categorical(
    true_value: int,
    n_categories: int,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Generalised randomized response for categorical data.

    Reports the true category with probability  e^ε / (e^ε + d − 1),
    and a uniformly random other category otherwise.  Satisfies (ε, 0)-DP.

    Parameters
    ----------
    true_value : int
        Index of the true category in [0, n_categories).
    n_categories : int
        Total number of categories d ≥ 2.
    epsilon : float
        Privacy parameter.
    rng : optional Generator
        Numpy random generator.

    Returns
    -------
    int
        Perturbed category index.
    """
    if n_categories < 2:
        raise ValueError("n_categories must be >= 2")
    if not (0 <= true_value < n_categories):
        raise ValueError(f"true_value must be in [0, {n_categories})")
    rng = rng or np.random.default_rng()

    p_truth = math.exp(epsilon) / (math.exp(epsilon) + n_categories - 1)
    if rng.random() < p_truth:
        return true_value
    # Pick a uniformly random *other* category
    other = int(rng.integers(0, n_categories - 1))
    if other >= true_value:
        other += 1
    return other


# ═══════════════════════════════════════════════════════════════════════════
# Report Noisy Max / Argmax
# ═══════════════════════════════════════════════════════════════════════════


def report_noisy_max(
    scores: Sequence[float],
    sensitivity: float,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Report-Noisy-Max: return the index of the highest noised score.

    Adds independent Laplace(Δ/ε) noise to each score and returns the
    index of the maximum.  Satisfies (ε, 0)-DP.

    Parameters
    ----------
    scores : Sequence[float]
        True quality scores for each candidate.
    sensitivity : float
        Global sensitivity of the score function.
    epsilon : float
        Privacy parameter.
    rng : optional Generator
        Numpy random generator.

    Returns
    -------
    int
        Index of the noisily highest-scoring candidate.
    """
    rng = rng or np.random.default_rng()
    arr = np.asarray(scores, dtype=np.float64)
    scale = laplace_scale(sensitivity, epsilon)
    noised = arr + rng.laplace(0.0, scale, size=arr.shape)
    return int(np.argmax(noised))


def report_noisy_argmax(
    scores: Sequence[float],
    sensitivity: float,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, float]:
    """Like :func:`report_noisy_max` but also returns the noised maximum value.

    Returns
    -------
    tuple[int, float]
        (index, noised_score) of the maximum.
    """
    rng = rng or np.random.default_rng()
    arr = np.asarray(scores, dtype=np.float64)
    scale = laplace_scale(sensitivity, epsilon)
    noised = arr + rng.laplace(0.0, scale, size=arr.shape)
    idx = int(np.argmax(noised))
    return idx, float(noised[idx])


# ═══════════════════════════════════════════════════════════════════════════
# Sparse Vector Technique (SVT)
# ═══════════════════════════════════════════════════════════════════════════


def sparse_vector_technique(
    queries: Sequence[float],
    threshold: float,
    sensitivity: float,
    epsilon: float,
    max_above: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> List[Optional[bool]]:
    """Sparse Vector Technique (SVT).

    Answers a (potentially long) stream of numeric queries against a
    threshold.  Only reveals *which* queries are above threshold (up to
    *max_above* of them), consuming O(1) budget per non-above answer.

    Satisfies (ε, 0)-DP over all queries.

    Parameters
    ----------
    queries : Sequence[float]
        True query answers.
    threshold : float
        Public threshold T.
    sensitivity : float
        Global sensitivity of each query.
    epsilon : float
        Total privacy budget.
    max_above : int
        Stop after this many ``True`` answers.
    rng : optional Generator
        Numpy random generator.

    Returns
    -------
    list[Optional[bool]]
        For each query: ``True`` (above), ``False`` (below), or ``None``
        (not evaluated because the budget was exhausted).
    """
    if max_above <= 0:
        raise ValueError("max_above must be > 0")
    rng = rng or np.random.default_rng()

    # Split budget: half for threshold noise, half for query noise
    eps_t = epsilon / 2.0
    eps_q = epsilon / (2.0 * max_above)

    noisy_threshold = threshold + rng.laplace(0.0, sensitivity / eps_t)

    results: List[Optional[bool]] = []
    count_above = 0

    for q in queries:
        if count_above >= max_above:
            results.append(None)
            continue
        noisy_q = q + rng.laplace(0.0, sensitivity / eps_q)
        if noisy_q >= noisy_threshold:
            results.append(True)
            count_above += 1
        else:
            results.append(False)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Above Threshold
# ═══════════════════════════════════════════════════════════════════════════


def above_threshold(
    queries: Sequence[float],
    threshold: float,
    sensitivity: float,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Above Threshold: return the index of the first query above a noisy threshold.

    A specialisation of SVT with ``max_above=1``.  Satisfies (ε, 0)-DP.

    Parameters
    ----------
    queries : Sequence[float]
        True query answers.
    threshold : float
        Public threshold.
    sensitivity : float
        Sensitivity of each query.
    epsilon : float
        Privacy budget.
    rng : optional Generator

    Returns
    -------
    int
        Index of the first query above the noisy threshold, or ``-1`` if
        none exceeds it.
    """
    results = sparse_vector_technique(
        queries, threshold, sensitivity, epsilon, max_above=1, rng=rng,
    )
    for i, r in enumerate(results):
        if r is True:
            return i
    return -1


# ═══════════════════════════════════════════════════════════════════════════
# Geometric mechanism
# ═══════════════════════════════════════════════════════════════════════════


def geometric_mechanism(
    value: int,
    sensitivity: int,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Two-sided geometric mechanism for integer-valued queries.

    Adds noise drawn from a two-sided geometric distribution with
    parameter  p = 1 − e^{−ε/Δ}.  Satisfies (ε, 0)-DP.

    Parameters
    ----------
    value : int
        True integer query answer.
    sensitivity : int
        Integer L1 sensitivity.
    epsilon : float
        Privacy parameter.
    rng : optional Generator

    Returns
    -------
    int
        Noised integer answer.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    rng = rng or np.random.default_rng()
    p = 1.0 - math.exp(-epsilon / sensitivity)
    # Two-sided geometric = difference of two geometric RVs
    pos = int(rng.geometric(p)) - 1
    neg = int(rng.geometric(p)) - 1
    return value + pos - neg


# ═══════════════════════════════════════════════════════════════════════════
# Truncated mechanism variants
# ═══════════════════════════════════════════════════════════════════════════


def truncated_laplace(
    value: float,
    sensitivity: float,
    epsilon: float,
    lower: float,
    upper: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Laplace mechanism with output truncated (clamped) to [lower, upper].

    Post-processing does not degrade the privacy guarantee.

    Parameters
    ----------
    value : float
        True query answer.
    sensitivity : float
        Global L1 sensitivity.
    epsilon : float
        Privacy parameter.
    lower, upper : float
        Output bounds.
    rng : optional Generator

    Returns
    -------
    float
        Clamped noised answer.
    """
    noised = laplace_mechanism(value, sensitivity, epsilon, rng=rng)
    return float(np.clip(noised, lower, upper))


def truncated_gaussian(
    value: float,
    sensitivity: float,
    epsilon: float,
    delta: float,
    lower: float,
    upper: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Gaussian mechanism with output truncated to [lower, upper].

    Parameters
    ----------
    value : float
        True query answer.
    sensitivity : float
        Global L2 sensitivity.
    epsilon : float
        Privacy parameter.
    delta : float
        Failure probability.
    lower, upper : float
        Output bounds.
    rng : optional Generator

    Returns
    -------
    float
        Clamped noised answer.
    """
    noised = gaussian_mechanism(value, sensitivity, epsilon, delta, rng=rng)
    return float(np.clip(noised, lower, upper))


# ═══════════════════════════════════════════════════════════════════════════
# Composition helpers
# ═══════════════════════════════════════════════════════════════════════════


def compose_epsilons_basic(epsilons: Sequence[float]) -> float:
    """Basic sequential composition: ε_total = Σ εᵢ."""
    return float(sum(epsilons))


def compose_epsilons_advanced(
    epsilons: Sequence[float],
    delta_target: float,
) -> float:
    """Advanced (strong) composition theorem.

    For *k* mechanisms each satisfying (ε, 0)-DP, the composed mechanism
    satisfies (ε', δ')-DP where

        ε' = ε √(2k ln(1/δ')) + k ε (e^ε − 1)

    We assume homogeneous ε here and use *delta_target* as δ'.

    Parameters
    ----------
    epsilons : Sequence[float]
        Per-mechanism ε values (assumed identical for the theorem).
    delta_target : float
        Target failure probability δ' > 0.

    Returns
    -------
    float
        Composed ε'.
    """
    if delta_target <= 0 or delta_target >= 1:
        raise ValueError("delta_target must be in (0, 1)")
    k = len(epsilons)
    if k == 0:
        return 0.0
    eps = max(epsilons)  # worst-case per-step ε
    term1 = eps * math.sqrt(2.0 * k * math.log(1.0 / delta_target))
    term2 = k * eps * (math.exp(eps) - 1.0)
    return term1 + term2


def noise_config_laplace(
    sensitivity: float,
    epsilon: float,
    *,
    clipping_bound: Optional[float] = None,
) -> NoiseConfig:
    """Build a :class:`NoiseConfig` for the Laplace mechanism."""
    return NoiseConfig(
        mechanism=MechanismType.LAPLACE,
        scale=laplace_scale(sensitivity, epsilon),
        sensitivity=sensitivity,
        budget=PrivacyBudget(epsilon=epsilon, delta=0.0),
        clipping_bound=clipping_bound,
    )


def noise_config_gaussian(
    sensitivity: float,
    epsilon: float,
    delta: float,
    *,
    clipping_bound: Optional[float] = None,
) -> NoiseConfig:
    """Build a :class:`NoiseConfig` for the Gaussian mechanism."""
    return NoiseConfig(
        mechanism=MechanismType.GAUSSIAN,
        scale=gaussian_scale(sensitivity, epsilon, delta),
        sensitivity=sensitivity,
        budget=PrivacyBudget(epsilon=epsilon, delta=delta),
        clipping_bound=clipping_bound,
    )


__all__ = [
    # Sensitivity helpers
    "sensitivity_count",
    "sensitivity_sum",
    "sensitivity_mean",
    "sensitivity_median",
    "l2_sensitivity",
    # Laplace
    "laplace_scale",
    "laplace_mechanism",
    "laplace_mechanism_vector",
    # Gaussian
    "gaussian_scale",
    "gaussian_mechanism",
    "gaussian_mechanism_vector",
    # Exponential
    "exponential_mechanism",
    # Randomized response
    "randomized_response",
    "randomized_response_categorical",
    # Report noisy max
    "report_noisy_max",
    "report_noisy_argmax",
    # SVT
    "sparse_vector_technique",
    "above_threshold",
    # Geometric
    "geometric_mechanism",
    # Truncated
    "truncated_laplace",
    "truncated_gaussian",
    # Composition helpers
    "compose_epsilons_basic",
    "compose_epsilons_advanced",
    "noise_config_laplace",
    "noise_config_gaussian",
]
