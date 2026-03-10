"""
usability_oracle.information_theory.bounds — Information-theoretic bounds.

Implements fundamental inequalities (Fano, Pinsker, data processing) and
cognitive performance bounds derived from channel capacity constraints.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from usability_oracle.information_theory.entropy import (
    _as_prob,
    binary_entropy,
    shannon_entropy,
)
from usability_oracle.information_theory.mutual_information import (
    kl_divergence,
    mutual_information,
    total_variation_distance,
)


_LOG2 = math.log(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Fano's inequality
# ═══════════════════════════════════════════════════════════════════════════

def fano_inequality(
    error_probability: float,
    alphabet_size: int,
) -> float:
    """Fano's inequality: lower bound on conditional entropy from error rate.

    H(X|Y) ≤ h(P_e) + P_e log(|X| - 1)

    Parameters
    ----------
    error_probability : float
        Probability of error P_e ∈ [0, 1].
    alphabet_size : int
        Size of the alphabet |X| ≥ 2.

    Returns
    -------
    float
        Upper bound on H(X|Y) in bits.
    """
    if alphabet_size < 2:
        return 0.0
    pe = max(0.0, min(1.0, error_probability))
    if pe == 0.0:
        return 0.0
    return binary_entropy(pe) + pe * math.log2(alphabet_size - 1)


def fano_min_error(
    conditional_entropy_bits: float,
    alphabet_size: int,
) -> float:
    """Inverse Fano: minimum error probability given H(X|Y).

    Finds the smallest P_e such that Fano's inequality is satisfied.

    Parameters
    ----------
    conditional_entropy_bits : float
        H(X|Y) in bits.
    alphabet_size : int
        Alphabet size |X|.

    Returns
    -------
    float
        Lower bound on error probability.
    """
    if conditional_entropy_bits <= 0:
        return 0.0
    if alphabet_size < 2:
        return 0.0

    # Binary search for P_e
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        bound = fano_inequality(mid, alphabet_size)
        if bound < conditional_entropy_bits:
            lo = mid
        else:
            hi = mid
    return lo


# ═══════════════════════════════════════════════════════════════════════════
# Data processing inequality
# ═══════════════════════════════════════════════════════════════════════════

def data_processing_inequality_check(
    joint_xy: Union[Sequence[Sequence[float]], NDArray],
    joint_xz: Union[Sequence[Sequence[float]], NDArray],
) -> dict:
    """Check data processing inequality: I(X;Z) ≤ I(X;Y) for Markov chain X→Y→Z.

    Parameters
    ----------
    joint_xy : 2-D array-like
        Joint distribution p(x, y).
    joint_xz : 2-D array-like
        Joint distribution p(x, z).

    Returns
    -------
    dict
        Keys: "i_xy", "i_xz", "satisfied", "gap"
    """
    i_xy = mutual_information(joint_xy)
    i_xz = mutual_information(joint_xz)
    return {
        "i_xy": i_xy,
        "i_xz": i_xz,
        "satisfied": i_xz <= i_xy + 1e-10,
        "gap": i_xy - i_xz,
    }


def information_bottleneck_bound(
    i_xy: float,
    i_xz: float,
) -> bool:
    """Check if the information bottleneck bound I(X;Z) ≤ I(X;Y) holds.

    Parameters
    ----------
    i_xy : float
        I(X;Y) in bits.
    i_xz : float
        I(X;Z) in bits.

    Returns
    -------
    bool
        True if DPI is satisfied.
    """
    return i_xz <= i_xy + 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# Pinsker's inequality
# ═══════════════════════════════════════════════════════════════════════════

def pinsker_inequality(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
) -> dict:
    """Pinsker's inequality: TV(p,q) ≤ √(½ D_KL(p‖q)).

    Provides a bound on total variation distance from KL divergence.

    Parameters
    ----------
    p, q : array-like
        Probability distributions.

    Returns
    -------
    dict
        Keys: "tv_distance", "kl_divergence_nats", "pinsker_bound", "tight"
    """
    tv = total_variation_distance(p, q)
    kl_nats = kl_divergence(p, q, base=math.e)
    pinsker_bound = math.sqrt(0.5 * kl_nats) if np.isfinite(kl_nats) else float("inf")
    return {
        "tv_distance": tv,
        "kl_divergence_nats": kl_nats,
        "pinsker_bound": pinsker_bound,
        "tight": tv <= pinsker_bound + 1e-10,
    }


def reverse_pinsker(
    tv_distance: float,
) -> float:
    """Reverse Pinsker: lower bound on D_KL from TV distance.

    D_KL(p‖q) ≥ 2 TV(p,q)²

    Parameters
    ----------
    tv_distance : float
        Total variation distance.

    Returns
    -------
    float
        Lower bound on KL divergence in nats.
    """
    return 2.0 * tv_distance ** 2


# ═══════════════════════════════════════════════════════════════════════════
# Mrs. Gerber's lemma
# ═══════════════════════════════════════════════════════════════════════════

def mrs_gerber_lemma(
    h_x: float,
    crossover_p: float,
) -> float:
    """Mrs. Gerber's lemma: lower bound on H(Y) for BSC.

    If X is binary with H(X) = h and Y = X ⊕ N where N ~ Bernoulli(p),
    then H(Y) ≥ h(h⁻¹(h) * (1-2p) + p)

    where h is binary entropy and h⁻¹ is its inverse on [0, 0.5].

    Parameters
    ----------
    h_x : float
        Entropy H(X) in bits (0 ≤ h ≤ 1).
    crossover_p : float
        BSC crossover probability p.

    Returns
    -------
    float
        Lower bound on H(Y) in bits.
    """
    if h_x <= 0:
        return binary_entropy(crossover_p)
    if h_x >= 1.0:
        return 1.0

    # Invert binary entropy: find q such that h(q) = h_x, q ∈ [0, 0.5]
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = (lo + hi) / 2
        if binary_entropy(mid) < h_x:
            lo = mid
        else:
            hi = mid
    q = (lo + hi) / 2

    # Compute the BSC convolution: q * p = q(1-p) + (1-q)p
    conv = q * (1 - crossover_p) + (1 - q) * crossover_p
    return binary_entropy(conv)


# ═══════════════════════════════════════════════════════════════════════════
# Channel coding bounds
# ═══════════════════════════════════════════════════════════════════════════

def channel_coding_converse(
    capacity_bits: float,
    rate_bits: float,
    block_length: int,
) -> float:
    """Strong converse bound on error probability.

    For rate R > C, the error probability → 1 exponentially fast.
    Uses the sphere-packing exponent approximation.

    Parameters
    ----------
    capacity_bits : float
        Channel capacity C in bits.
    rate_bits : float
        Coding rate R in bits.
    block_length : int
        Block length n.

    Returns
    -------
    float
        Lower bound on error probability.
    """
    if rate_bits <= capacity_bits:
        return 0.0  # achievable
    excess = rate_bits - capacity_bits
    # Approximate: P_e ≥ 1 - exp(-n·E_sp(R))
    exponent = excess * _LOG2  # simplified bound
    return max(0.0, 1.0 - math.exp(-block_length * exponent))


def channel_coding_achievability(
    capacity_bits: float,
    rate_bits: float,
    block_length: int,
) -> float:
    """Random coding bound on achievable error probability.

    For rate R < C, the error probability ≤ exp(-n·E_r(R)).

    Parameters
    ----------
    capacity_bits : float
        Channel capacity C.
    rate_bits : float
        Coding rate R.
    block_length : int
        Block length n.

    Returns
    -------
    float
        Upper bound on error probability.
    """
    if rate_bits >= capacity_bits:
        return 1.0
    slack = capacity_bits - rate_bits
    exponent = slack * _LOG2  # simplified random coding exponent
    return min(1.0, math.exp(-block_length * exponent))


# ═══════════════════════════════════════════════════════════════════════════
# Source coding bounds
# ═══════════════════════════════════════════════════════════════════════════

def source_coding_bound(
    entropy_bits: float,
    rate_bits: float,
    block_length: int,
) -> float:
    """Probability of source coding failure.

    If R < H(X), lossless compression is impossible and failure probability → 1.
    If R > H(X), failure probability → 0 exponentially.

    Parameters
    ----------
    entropy_bits : float
        Source entropy H(X) in bits.
    rate_bits : float
        Coding rate R in bits.
    block_length : int
        Block length.

    Returns
    -------
    float
        Bound on coding failure probability.
    """
    if rate_bits >= entropy_bits:
        slack = rate_bits - entropy_bits
        return min(1.0, math.exp(-block_length * slack * _LOG2))
    else:
        deficit = entropy_bits - rate_bits
        return max(0.0, 1.0 - math.exp(-block_length * deficit * _LOG2))


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive performance bounds
# ═══════════════════════════════════════════════════════════════════════════

def max_throughput(
    capacity_bits_per_second: float,
    bits_per_item: float,
) -> float:
    """Maximum items per second given channel capacity.

    Parameters
    ----------
    capacity_bits_per_second : float
        Cognitive channel capacity (bits/s).
    bits_per_item : float
        Information content per item (bits).

    Returns
    -------
    float
        Maximum items per second.
    """
    if bits_per_item <= 0:
        return float("inf")
    return capacity_bits_per_second / bits_per_item


def min_error_rate(
    capacity_bits: float,
    entropy_bits: float,
    alphabet_size: int,
) -> float:
    """Minimum error rate given information constraint.

    Uses Fano's inequality to bound the minimum error rate when the
    user's channel capacity limits the mutual information I(X;Y) ≤ C.

    P_e ≥ (H(X) - C - 1) / log(|X| - 1)

    Parameters
    ----------
    capacity_bits : float
        User's channel capacity in bits.
    entropy_bits : float
        Task entropy H(X) in bits.
    alphabet_size : int
        Number of alternatives.

    Returns
    -------
    float
        Lower bound on error rate.
    """
    if capacity_bits >= entropy_bits:
        return 0.0
    if alphabet_size < 2:
        return 0.0
    # H(X|Y) ≥ H(X) - I(X;Y) ≥ H(X) - C
    h_given_y = entropy_bits - capacity_bits
    return fano_min_error(h_given_y, alphabet_size)


def optimal_tradeoff_curve(
    capacity_bits: float,
    entropy_bits: float,
    n_alternatives: int,
    n_points: int = 50,
) -> list[Tuple[float, float]]:
    """Compute the optimal speed-accuracy tradeoff curve.

    Given a fixed channel capacity, traces out achievable (time, error_rate)
    pairs.  Longer observation times allow more information to be gathered.

    Parameters
    ----------
    capacity_bits : float
        Bits per second of cognitive capacity.
    entropy_bits : float
        Total stimulus entropy in bits.
    n_alternatives : int
        Number of response alternatives.
    n_points : int
        Number of points on the curve.

    Returns
    -------
    list[tuple[float, float]]
        (time_seconds, error_rate) pairs.
    """
    if capacity_bits <= 0:
        return [(float("inf"), 0.0)]

    min_time = entropy_bits / capacity_bits
    max_time = min_time * 3.0  # some margin

    results: list[Tuple[float, float]] = []
    for t in np.linspace(0.01, max_time, n_points):
        bits_available = capacity_bits * t
        if bits_available >= entropy_bits:
            error = 0.0
        else:
            h_given_y = entropy_bits - bits_available
            error = fano_min_error(h_given_y, n_alternatives)
        results.append((float(t), error))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Asymptotic equipartition property (AEP)
# ═══════════════════════════════════════════════════════════════════════════

def aep_typical_set_size(
    entropy_bits: float,
    block_length: int,
    epsilon: float = 0.1,
) -> Tuple[float, float]:
    """Size of the ε-typical set from AEP.

    |A_ε^(n)| ≈ 2^{nH(X)} with bounds:
    (1-ε)·2^{n(H-ε)} ≤ |A_ε^(n)| ≤ 2^{n(H+ε)}

    Parameters
    ----------
    entropy_bits : float
        Source entropy H(X) in bits.
    block_length : int
        Block length n.
    epsilon : float
        AEP tolerance ε.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) on typical set size (as log₂).
    """
    lower = block_length * (entropy_bits - epsilon)
    upper = block_length * (entropy_bits + epsilon)
    return (max(lower, 0.0), upper)


def aep_probability_bound(
    entropy_bits: float,
    block_length: int,
    epsilon: float = 0.1,
) -> float:
    """Probability that a sequence is ε-typical (from AEP).

    P(A_ε^(n)) ≥ 1 - Var[log p(X)] / (n ε²)

    For IID sources with finite variance, this approaches 1.

    Parameters
    ----------
    entropy_bits : float
        Source entropy.
    block_length : int
        Block length.
    epsilon : float
        Tolerance.

    Returns
    -------
    float
        Lower bound on probability of being typical.
    """
    # For IID sources, we use Chebyshev's bound
    # A rough upper bound on variance: H(X) log₂(|X|) (heuristic)
    var_bound = entropy_bits * max(entropy_bits, 1.0)
    prob = 1.0 - var_bound / (block_length * epsilon ** 2)
    return max(0.0, min(1.0, prob))


def aep_compression_rate(
    distribution: Union[Sequence[float], NDArray],
    block_length: int,
    epsilon: float = 0.1,
) -> dict:
    """AEP-based compression analysis.

    Parameters
    ----------
    distribution : array-like
        Source distribution.
    block_length : int
        Block length for compression.
    epsilon : float
        AEP tolerance.

    Returns
    -------
    dict
        Keys: "entropy_rate", "bits_per_block", "typical_set_log_size",
              "compression_ratio"
    """
    p = _as_prob(distribution)
    h = shannon_entropy(p, base=2.0)
    n = len(p)

    typical_lower, typical_upper = aep_typical_set_size(h, block_length, epsilon)
    total_sequences_log = block_length * math.log2(n) if n > 1 else 0.0

    return {
        "entropy_rate": h,
        "bits_per_block": block_length * h,
        "typical_set_log_size": (typical_lower + typical_upper) / 2.0,
        "total_sequences_log": total_sequences_log,
        "compression_ratio": h / math.log2(n) if n > 1 else 1.0,
    }


__all__ = [
    "aep_compression_rate",
    "aep_probability_bound",
    "aep_typical_set_size",
    "channel_coding_achievability",
    "channel_coding_converse",
    "data_processing_inequality_check",
    "fano_inequality",
    "fano_min_error",
    "information_bottleneck_bound",
    "max_throughput",
    "min_error_rate",
    "mrs_gerber_lemma",
    "optimal_tradeoff_curve",
    "pinsker_inequality",
    "reverse_pinsker",
    "source_coding_bound",
]
