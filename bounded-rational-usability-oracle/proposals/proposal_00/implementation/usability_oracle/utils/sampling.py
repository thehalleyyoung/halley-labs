"""
usability_oracle.utils.sampling — Statistical sampling utilities.

Provides distribution sampling, Latin hypercube sampling, stratified
sampling, importance sampling, and bootstrap estimation.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Hashable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Basic sampling
# ---------------------------------------------------------------------------

def sample_from_distribution(
    probs: dict[str, float],
    n: int,
    seed: int | None = None,
) -> list[str]:
    """Draw *n* samples from a categorical distribution.

    Parameters:
        probs: Mapping from outcome label to probability.
        n: Number of samples.
        seed: RNG seed for reproducibility.

    Returns:
        List of sampled labels.
    """
    rng = np.random.RandomState(seed)
    labels = list(probs.keys())
    p = np.array([probs[k] for k in labels], dtype=float)
    total = p.sum()
    if total <= 0:
        p = np.ones(len(labels)) / len(labels)
    else:
        p = p / total
    indices = rng.choice(len(labels), size=n, p=p)
    return [labels[i] for i in indices]


# ---------------------------------------------------------------------------
# Latin hypercube sampling
# ---------------------------------------------------------------------------

def latin_hypercube(
    n_samples: int,
    n_dims: int,
    bounds: list[tuple[float, float]] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Latin hypercube sample.

    Divides each dimension into *n_samples* equal strata, then places
    exactly one sample in each stratum with a random offset.

    Parameters:
        n_samples: Number of sample points.
        n_dims: Dimensionality.
        bounds: Optional list of (lo, hi) bounds per dimension.
            Defaults to [0, 1] for each dimension.
        seed: RNG seed.

    Returns:
        Array of shape ``(n_samples, n_dims)`` with samples in [lo, hi].
    """
    rng = np.random.RandomState(seed)
    if bounds is None:
        bounds = [(0.0, 1.0)] * n_dims
    if len(bounds) != n_dims:
        raise ValueError("bounds must have length n_dims")

    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        lo, hi = bounds[d]
        # Create stratified random positions in [0, 1]
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            stratum = perm[i]
            u = rng.uniform(0, 1)
            unit = (stratum + u) / n_samples
            samples[i, d] = lo + unit * (hi - lo)
    return samples


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    population: list[Any],
    strata_fn: Callable[[Any], Hashable],
    n_per_stratum: int,
    seed: int | None = None,
) -> list[Any]:
    """Sample *n_per_stratum* items from each stratum of *population*.

    Parameters:
        population: Full population list.
        strata_fn: Function mapping an item to its stratum key.
        n_per_stratum: Samples to draw from each stratum.
        seed: RNG seed.

    Returns:
        Combined list of sampled items (order: stratum by stratum).
    """
    rng = np.random.RandomState(seed)
    strata: dict[Hashable, list[Any]] = {}
    for item in population:
        key = strata_fn(item)
        strata.setdefault(key, []).append(item)

    result: list[Any] = []
    for key in sorted(strata.keys(), key=str):
        members = strata[key]
        k = min(n_per_stratum, len(members))
        indices = rng.choice(len(members), size=k, replace=False)
        for idx in indices:
            result.append(members[idx])
    return result


# ---------------------------------------------------------------------------
# Importance sampling
# ---------------------------------------------------------------------------

def importance_sampling(
    target_fn: Callable[[np.ndarray], float],
    proposal_fn: Callable[[np.ndarray], float],
    n_samples: int,
    dim: int = 1,
    proposal_sampler: Callable[[int], np.ndarray] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an expectation under *target_fn* using importance sampling.

    Parameters:
        target_fn: Un-normalised target density π(x).
        proposal_fn: Proposal density q(x).
        n_samples: Number of samples.
        dim: Dimensionality for default Gaussian proposal sampler.
        proposal_sampler: Callable returning samples from q.  Defaults to
            standard-normal sampling.
        seed: RNG seed.

    Returns:
        (samples, normalised_weights) where weights sum to 1.
    """
    rng = np.random.RandomState(seed)
    if proposal_sampler is None:
        samples = rng.randn(n_samples, dim)
    else:
        samples = proposal_sampler(n_samples)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

    log_weights = np.zeros(n_samples)
    for i in range(n_samples):
        x = samples[i]
        p_val = target_fn(x)
        q_val = proposal_fn(x)
        if q_val > 0 and p_val > 0:
            log_weights[i] = math.log(p_val) - math.log(q_val)
        elif p_val <= 0:
            log_weights[i] = -1e30
        else:
            log_weights[i] = 1e30

    # Normalise weights via log-sum-exp
    max_lw = np.max(log_weights)
    weights = np.exp(log_weights - max_lw)
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights = np.full(n_samples, 1.0 / n_samples)

    return samples, weights


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Non-parametric bootstrap estimation of a statistic.

    Parameters:
        data: 1-D array of observations.
        statistic: Function computing a scalar from a data array.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: RNG seed.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    data = np.asarray(data, dtype=float).ravel()
    rng = np.random.RandomState(seed)
    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)

    point = statistic(data)
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = data[rng.choice(n, size=n, replace=True)]
        boot_stats[i] = statistic(resample)

    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (float(point), ci_lo, ci_hi)
