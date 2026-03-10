"""
usability_oracle.information_theory.rate_distortion — Rate-distortion theory.

Implements the rate-distortion function R(D) via the Blahut-Arimoto algorithm,
parametric source models, and operational rate-distortion interpretations for
bounded-rational cognitive agents.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from usability_oracle.information_theory.entropy import (
    _as_prob,
    _safe_log,
    shannon_entropy,
)
from usability_oracle.information_theory.mutual_information import kl_divergence
from usability_oracle.information_theory.types import RateDistortionPoint


_LOG2 = math.log(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Distortion measures
# ═══════════════════════════════════════════════════════════════════════════

def hamming_distortion(n: int) -> NDArray:
    """Hamming distortion matrix: d(x, x̂) = 0 if x==x̂, 1 otherwise.

    Parameters
    ----------
    n : int
        Alphabet size.

    Returns
    -------
    NDArray
        n×n distortion matrix.
    """
    return 1.0 - np.eye(n)


def squared_error_distortion(values: Union[Sequence[float], NDArray]) -> NDArray:
    """Squared-error distortion matrix d(x, x̂) = (x - x̂)².

    Parameters
    ----------
    values : array-like
        Alphabet of reconstruction values.

    Returns
    -------
    NDArray
        n×n distortion matrix.
    """
    v = np.asarray(values, dtype=np.float64)
    return (v[:, None] - v[None, :]) ** 2


def absolute_error_distortion(values: Union[Sequence[float], NDArray]) -> NDArray:
    """Absolute-error distortion matrix d(x, x̂) = |x - x̂|.

    Parameters
    ----------
    values : array-like
        Alphabet of reconstruction values.

    Returns
    -------
    NDArray
        n×n distortion matrix.
    """
    v = np.asarray(values, dtype=np.float64)
    return np.abs(v[:, None] - v[None, :])


# ═══════════════════════════════════════════════════════════════════════════
# Blahut-Arimoto for rate-distortion
# ═══════════════════════════════════════════════════════════════════════════

def rate_distortion_point(
    source_dist: Union[Sequence[float], NDArray],
    distortion_matrix: Union[Sequence[Sequence[float]], NDArray],
    beta: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
) -> RateDistortionPoint:
    """Compute a single point on the rate-distortion curve R(D) via Blahut-Arimoto.

    Minimizes F = R + β·D, producing the optimal test channel q(x̂|x).

    The algorithm alternates:
      1. q(x̂|x) ∝ q(x̂) exp(-β d(x,x̂))           [update encoder]
      2. q(x̂) = Σ_x p(x) q(x̂|x)                   [update marginal]

    Parameters
    ----------
    source_dist : array-like
        Source distribution p(x), length |X|.
    distortion_matrix : 2-D array-like
        Distortion d(x, x̂), shape (|X|, |X̂|).
    beta : float
        Inverse temperature / slope parameter (β > 0).
    tolerance : float
        Convergence tolerance on rate.
    max_iterations : int
        Maximum iterations.

    Returns
    -------
    RateDistortionPoint
        Point on the R(D) curve at the given β.
    """
    p = _as_prob(source_dist)
    d = np.asarray(distortion_matrix, dtype=np.float64)
    n_x = len(p)
    n_xhat = d.shape[1]

    if beta <= 0:
        # β=0: no constraint, infinite rate possible → return max distortion point
        return RateDistortionPoint(
            distortion=0.0,
            rate_bits=shannon_entropy(p, base=2.0),
            beta=0.0,
        )

    # Initialize reproduction distribution q(x̂) as uniform
    q_xhat = np.full(n_xhat, 1.0 / n_xhat)

    prev_rate = float("inf")
    q_xhat_given_x = np.zeros((n_x, n_xhat))

    for it in range(max_iterations):
        # Step 1: Update encoder q(x̂|x) ∝ q(x̂) exp(-β d(x,x̂))
        log_q_xhat = np.log(np.maximum(q_xhat, 1e-300))
        for x in range(n_x):
            log_unnorm = log_q_xhat - beta * d[x]
            log_unnorm -= log_unnorm.max()  # numerical stability
            unnorm = np.exp(log_unnorm)
            total = unnorm.sum()
            if total > 0:
                q_xhat_given_x[x] = unnorm / total
            else:
                q_xhat_given_x[x] = 1.0 / n_xhat

        # Step 2: Update reproduction marginal q(x̂) = Σ_x p(x) q(x̂|x)
        q_xhat = p @ q_xhat_given_x
        q_xhat = np.maximum(q_xhat, 1e-300)
        q_xhat /= q_xhat.sum()

        # Compute rate I(X; X̂) = Σ_x p(x) D_KL(q(x̂|x) ‖ q(x̂))
        rate_nats = 0.0
        for x in range(n_x):
            if p[x] > 0:
                rate_nats += p[x] * _kl_nats(q_xhat_given_x[x], q_xhat)
        rate_bits = rate_nats / _LOG2

        if abs(rate_bits - prev_rate) < tolerance:
            break
        prev_rate = rate_bits

    # Compute expected distortion
    distortion = 0.0
    for x in range(n_x):
        distortion += p[x] * np.dot(q_xhat_given_x[x], d[x])

    return RateDistortionPoint(
        distortion=float(distortion),
        rate_bits=max(rate_bits, 0.0),
        beta=beta,
        optimal_encoding=tuple(
            tuple(float(v) for v in row) for row in q_xhat_given_x
        ),
    )


def _kl_nats(p: NDArray, q: NDArray) -> float:
    """KL divergence in nats (internal helper)."""
    mask = p > 0
    if np.any(mask & (q == 0)):
        return float("inf")
    vals = np.zeros_like(p)
    vals[mask] = p[mask] * (np.log(p[mask]) - np.log(q[mask]))
    return max(float(vals.sum()), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Rate-distortion curve
# ═══════════════════════════════════════════════════════════════════════════

def rate_distortion_curve(
    source_dist: Union[Sequence[float], NDArray],
    distortion_matrix: Union[Sequence[Sequence[float]], NDArray],
    beta_range: Tuple[float, float] = (0.01, 100.0),
    n_points: int = 50,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    log_scale: bool = True,
) -> list[RateDistortionPoint]:
    """Compute the full R(D) curve over a range of β values.

    Parameters
    ----------
    source_dist : array-like
        Source distribution p(x).
    distortion_matrix : 2-D array-like
        Distortion matrix d(x, x̂).
    beta_range : tuple
        (β_min, β_max) range.
    n_points : int
        Number of points to compute.
    tolerance : float
        Convergence tolerance per point.
    max_iterations : int
        Maximum iterations per point.
    log_scale : bool
        If True, space β values logarithmically.

    Returns
    -------
    list[RateDistortionPoint]
        Points on the R(D) curve, ordered by increasing β.
    """
    b_min, b_max = beta_range
    if log_scale:
        betas = np.logspace(np.log10(b_min), np.log10(b_max), n_points)
    else:
        betas = np.linspace(b_min, b_max, n_points)

    return [
        rate_distortion_point(
            source_dist, distortion_matrix, float(b),
            tolerance=tolerance, max_iterations=max_iterations,
        )
        for b in betas
    ]


def distortion_rate_function(
    source_dist: Union[Sequence[float], NDArray],
    distortion_matrix: Union[Sequence[Sequence[float]], NDArray],
    target_rate_bits: float,
    *,
    tolerance: float = 1e-8,
    max_bisection: int = 100,
) -> RateDistortionPoint:
    """Compute D(R): minimum distortion achievable at rate R.

    Uses bisection on β to find the operating point with the target rate.

    Parameters
    ----------
    source_dist : array-like
        Source distribution p(x).
    distortion_matrix : 2-D array-like
        Distortion matrix.
    target_rate_bits : float
        Target rate in bits.
    tolerance : float
        Tolerance on the rate match.
    max_bisection : int
        Maximum bisection iterations.

    Returns
    -------
    RateDistortionPoint
        The R-D point closest to the target rate.
    """
    beta_lo, beta_hi = 1e-4, 1e4
    best = rate_distortion_point(source_dist, distortion_matrix, beta_lo)

    for _ in range(max_bisection):
        beta_mid = math.sqrt(beta_lo * beta_hi)  # geometric mean
        pt = rate_distortion_point(source_dist, distortion_matrix, beta_mid)
        if abs(pt.rate_bits - target_rate_bits) < tolerance:
            return pt
        if pt.rate_bits > target_rate_bits:
            beta_hi = beta_mid
        else:
            beta_lo = beta_mid
        best = pt

    return best


# ═══════════════════════════════════════════════════════════════════════════
# Parametric source models
# ═══════════════════════════════════════════════════════════════════════════

def rd_discrete_memoryless(
    source_dist: Union[Sequence[float], NDArray],
    beta: float,
) -> RateDistortionPoint:
    """R(D) for discrete memoryless source with Hamming distortion.

    Parameters
    ----------
    source_dist : array-like
        Source distribution.
    beta : float
        Inverse temperature.

    Returns
    -------
    RateDistortionPoint
        Point on the R(D) curve.
    """
    p = _as_prob(source_dist)
    d = hamming_distortion(len(p))
    return rate_distortion_point(p, d, beta)


def rd_gaussian(
    variance: float,
    distortion: float,
) -> float:
    """Closed-form R(D) for Gaussian source with squared-error distortion.

    R(D) = max(0, 0.5 log₂(σ²/D))

    Parameters
    ----------
    variance : float
        Source variance σ².
    distortion : float
        Target distortion D.

    Returns
    -------
    float
        Rate in bits.
    """
    if variance <= 0:
        raise ValueError("Variance must be positive")
    if distortion <= 0:
        return float("inf")
    if distortion >= variance:
        return 0.0
    return 0.5 * math.log2(variance / distortion)


def rd_gaussian_distortion(
    variance: float,
    rate_bits: float,
) -> float:
    """Closed-form D(R) for Gaussian source: D(R) = σ² 2^{-2R}.

    Parameters
    ----------
    variance : float
        Source variance σ².
    rate_bits : float
        Rate in bits.

    Returns
    -------
    float
        Minimum distortion at the given rate.
    """
    if variance <= 0:
        raise ValueError("Variance must be positive")
    if rate_bits < 0:
        raise ValueError("Rate must be non-negative")
    return variance * (2.0 ** (-2.0 * rate_bits))


def rd_markov_source(
    transition_matrix: Union[Sequence[Sequence[float]], NDArray],
    distortion_matrix: Union[Sequence[Sequence[float]], NDArray],
    beta: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> RateDistortionPoint:
    """Rate-distortion for a first-order Markov source.

    Approximates R(D) by computing the rate-distortion function for each
    conditional distribution p(x_t | x_{t-1}) weighted by the stationary
    distribution, then taking the mixture.

    Parameters
    ----------
    transition_matrix : 2-D array-like
        Markov chain transition matrix P(x_t | x_{t-1}).
    distortion_matrix : 2-D array-like
        Distortion d(x, x̂).
    beta : float
        Inverse temperature.
    tolerance : float
        Convergence tolerance.
    max_iterations : int
        Maximum iterations.

    Returns
    -------
    RateDistortionPoint
        Approximate R-D point.
    """
    P = _as_prob(transition_matrix)
    n = P.shape[0]

    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    pi /= pi.sum()

    # Weighted average of conditional R-D
    total_rate = 0.0
    total_dist = 0.0
    for i in range(n):
        if pi[i] > 1e-15:
            pt = rate_distortion_point(
                P[i], distortion_matrix, beta,
                tolerance=tolerance, max_iterations=max_iterations,
            )
            total_rate += pi[i] * pt.rate_bits
            total_dist += pi[i] * pt.distortion

    return RateDistortionPoint(
        distortion=total_dist,
        rate_bits=total_rate,
        beta=beta,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive rate-distortion interpretation
# ═══════════════════════════════════════════════════════════════════════════

def cognitive_detail_level(
    source_dist: Union[Sequence[float], NDArray],
    distortion_matrix: Union[Sequence[Sequence[float]], NDArray],
    capacity_bits: float,
) -> RateDistortionPoint:
    """How much detail can a user perceive at cognitive capacity β?

    Maps cognitive channel capacity to an operating point on the R(D) curve.
    At rate = capacity, returns the minimum achievable distortion.

    Parameters
    ----------
    source_dist : array-like
        Distribution over UI states/elements.
    distortion_matrix : 2-D array-like
        Distortion between states.
    capacity_bits : float
        User's cognitive channel capacity in bits.

    Returns
    -------
    RateDistortionPoint
        Operating point showing achievable distortion at capacity.
    """
    return distortion_rate_function(
        source_dist, distortion_matrix, capacity_bits,
    )


def optimal_abstraction_level(
    n_states: int,
    capacity_bits: float,
) -> int:
    """Optimal number of abstract categories given cognitive capacity.

    A user with capacity C bits can distinguish at most 2^C categories.
    The optimal abstraction groups n_states into min(n, 2^C) categories.

    Parameters
    ----------
    n_states : int
        Number of concrete states.
    capacity_bits : float
        Cognitive capacity in bits.

    Returns
    -------
    int
        Optimal number of abstract categories.
    """
    if capacity_bits <= 0:
        return 1
    max_categories = int(2.0 ** capacity_bits)
    return min(n_states, max(max_categories, 1))


def rd_bisimulation_interpretation(
    source_dist: Union[Sequence[float], NDArray],
    distortion_matrix: Union[Sequence[Sequence[float]], NDArray],
    beta: float,
) -> dict:
    """Interpret R-D solution as a soft bisimulation quotient.

    The optimal test channel q(x̂|x) from rate-distortion theory defines a
    soft clustering of source states into abstract states.  This is the
    information-theoretic analogue of bisimulation quotients in MDP theory.

    Parameters
    ----------
    source_dist : array-like
        Source distribution.
    distortion_matrix : 2-D array-like
        Distortion matrix.
    beta : float
        Inverse temperature.

    Returns
    -------
    dict
        Dictionary with:
          - "rd_point": the R-D operating point
          - "soft_partition": the test channel q(x̂|x)
          - "hard_partition": deterministic assignment argmax_x̂ q(x̂|x)
          - "n_effective_clusters": effective number of clusters
          - "compression_ratio": how much the state space is compressed
    """
    pt = rate_distortion_point(source_dist, distortion_matrix, beta)
    p = _as_prob(source_dist)
    n = len(p)

    if not pt.optimal_encoding:
        return {
            "rd_point": pt,
            "soft_partition": None,
            "hard_partition": list(range(n)),
            "n_effective_clusters": n,
            "compression_ratio": 1.0,
        }

    Q = np.array(pt.optimal_encoding)
    hard_assign = np.argmax(Q, axis=1).tolist()

    # Effective number of clusters from the marginal q(x̂)
    q_xhat = p @ Q
    q_xhat = q_xhat[q_xhat > 0]
    n_eff = float(np.exp(shannon_entropy(q_xhat, base=math.e)))

    return {
        "rd_point": pt,
        "soft_partition": Q.tolist(),
        "hard_partition": hard_assign,
        "n_effective_clusters": n_eff,
        "compression_ratio": n / max(n_eff, 1.0),
    }


__all__ = [
    "absolute_error_distortion",
    "cognitive_detail_level",
    "distortion_rate_function",
    "hamming_distortion",
    "optimal_abstraction_level",
    "rate_distortion_curve",
    "rate_distortion_point",
    "rd_bisimulation_interpretation",
    "rd_discrete_memoryless",
    "rd_gaussian",
    "rd_gaussian_distortion",
    "rd_markov_source",
    "squared_error_distortion",
]
