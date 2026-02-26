"""Distributional diagnostics for MPS-compressed probability vectors."""

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, ones_mps
from tn_check.tensor.operations import (
    mps_marginalize,
    mps_to_dense,
    mps_inner_product,
)


def marginal_distribution(mps: MPS, site: int) -> NDArray:
    """Extract the marginal probability distribution at a single site.

    Traces out all other sites by summing over their physical indices.

    Args:
        mps: Input MPS representing a probability distribution.
        site: Site index to extract the marginal for.

    Returns:
        1-D array of shape (d_site,) with the marginal probabilities.
    """
    if site < 0 or site >= mps.num_sites:
        raise ValueError(f"site {site} out of range [0, {mps.num_sites})")

    marginal_mps = mps_marginalize(mps, [site])
    return mps_to_dense(marginal_mps)


def compute_moments(mps: MPS, site: int, max_order: int = 4) -> dict:
    """Compute statistical moments of the marginal distribution at a site.

    Args:
        mps: Input MPS representing a probability distribution.
        site: Site index.
        max_order: Maximum moment order (up to 4 for skewness/kurtosis).

    Returns:
        Dictionary with keys 'mean', 'variance', and optionally
        'skewness', 'kurtosis' (excess kurtosis).
    """
    p = marginal_distribution(mps, site)
    d = len(p)
    x = np.arange(d, dtype=np.float64)

    total = np.sum(p)
    if total < 1e-15:
        return {"mean": 0.0, "variance": 0.0, "skewness": 0.0, "kurtosis": 0.0}

    # Normalize for moment computation
    p_norm = p / total

    mean = float(np.dot(x, p_norm))
    centered = x - mean
    variance = float(np.dot(centered ** 2, p_norm))

    result = {"mean": mean, "variance": variance}

    if max_order >= 3:
        if variance > 1e-15:
            std = np.sqrt(variance)
            skewness = float(np.dot(centered ** 3, p_norm)) / (std ** 3)
        else:
            skewness = 0.0
        result["skewness"] = skewness

    if max_order >= 4:
        if variance > 1e-15:
            kurtosis = float(np.dot(centered ** 4, p_norm)) / (variance ** 2) - 3.0
        else:
            kurtosis = 0.0
        result["kurtosis"] = kurtosis

    return result


def kl_divergence_marginal(mps_a: MPS, mps_b: MPS, site: int) -> float:
    """Compute KL divergence between marginal distributions at a site.

    KL(p || q) = sum_i p_i * log(p_i / q_i)

    Args:
        mps_a: First MPS (distribution p).
        mps_b: Second MPS (distribution q).
        site: Site index for marginal extraction.

    Returns:
        KL divergence (non-negative float). Returns inf if q has zeros
        where p is nonzero.
    """
    p = marginal_distribution(mps_a, site)
    q = marginal_distribution(mps_b, site)

    if len(p) != len(q):
        raise ValueError(
            f"Marginal dimensions differ: {len(p)} vs {len(q)}"
        )

    kl = 0.0
    for pi, qi in zip(p, q):
        if pi < 1e-15:
            continue
        if qi < 1e-15:
            return float("inf")
        kl += pi * np.log(pi / qi)

    return max(0.0, float(kl))


def total_negative_mass(mps: MPS) -> float:
    """Compute the total negative mass in the MPS distribution.

    For small systems converts to dense; for large systems uses sampling.

    Args:
        mps: Input MPS.

    Returns:
        Sum of absolute values of all negative entries.
    """
    total_size = mps.full_size
    if total_size <= 1_000_000:
        v = mps_to_dense(mps)
        return float(np.sum(np.abs(v[v < 0])))
    else:
        # Sampling-based estimation for large systems
        rng = np.random.default_rng(42)
        n_samples = 10000
        neg_total = 0.0
        from tn_check.tensor.operations import mps_probability_at_index

        for _ in range(n_samples):
            idx = tuple(int(rng.integers(0, d)) for d in mps.physical_dims)
            val = mps_probability_at_index(mps, idx)
            if val < 0:
                neg_total += abs(val)

        return neg_total / n_samples * total_size


def validate_probability_vector(mps: MPS, tolerance: float = 1e-6) -> dict:
    """Validate that an MPS represents a valid probability distribution.

    Checks:
      - non_negative: whether all entries are >= -tolerance
      - normalized: whether total probability is within tolerance of 1.0
      - marginal_consistent: whether marginals sum to total probability

    Args:
        mps: Input MPS.
        tolerance: Tolerance for checks.

    Returns:
        Dictionary with boolean results for each check and numeric details.
    """
    neg_mass = total_negative_mass(mps)

    # Compute total probability via dense conversion for small systems,
    # or via marginal sum for larger ones
    if mps.full_size <= 1_000_000:
        v = mps_to_dense(mps)
        total_prob = float(np.sum(v))
    else:
        # Use first marginal sum as proxy
        marg0 = marginal_distribution(mps, 0)
        total_prob = float(np.sum(marg0))

    is_normalized = abs(total_prob - 1.0) <= tolerance
    is_nonneg = neg_mass <= tolerance

    # Marginal consistency: each marginal should sum to total_prob
    marginal_consistent = True
    marginal_sums = []
    for site in range(mps.num_sites):
        marg = marginal_distribution(mps, site)
        marg_sum = float(np.sum(marg))
        marginal_sums.append(marg_sum)
        if abs(marg_sum - total_prob) > tolerance:
            marginal_consistent = False

    return {
        "non_negative": is_nonneg,
        "normalized": is_normalized,
        "marginal_consistent": marginal_consistent,
        "total_probability": total_prob,
        "negative_mass": neg_mass,
        "marginal_sums": marginal_sums,
    }
