"""
usability_oracle.information_theory.channel_capacity — Channel capacity algorithms.

Implements the Blahut-Arimoto algorithm for computing the capacity of discrete
memoryless channels, with extensions for cost constraints, warm-starting,
specific channel types, and cognitive channel models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from usability_oracle.information_theory.entropy import (
    _as_prob,
    _safe_log,
    binary_entropy,
    shannon_entropy,
)
from usability_oracle.information_theory.mutual_information import (
    kl_divergence,
    mutual_information,
)
from usability_oracle.information_theory.types import Channel, ChannelCapacity


_LOG2 = math.log(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Blahut-Arimoto algorithm
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BlahutArimotoState:
    """Internal state for warm-starting Blahut-Arimoto."""
    input_dist: NDArray
    iteration: int = 0
    lower_bound: float = 0.0
    upper_bound: float = float("inf")


def blahut_arimoto(
    transition_matrix: Union[Sequence[Sequence[float]], NDArray],
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    initial_distribution: Optional[Union[Sequence[float], NDArray]] = None,
) -> ChannelCapacity:
    """Blahut-Arimoto algorithm for discrete channel capacity.

    Computes C = max_{p(x)} I(X;Y) for a discrete memoryless channel
    specified by its transition matrix p(y|x).

    Parameters
    ----------
    transition_matrix : 2-D array-like
        Channel transition matrix p(y|x), shape (|X|, |Y|).
        Each row must sum to 1.
    tolerance : float
        Convergence tolerance on the capacity bounds gap.
    max_iterations : int
        Maximum number of iterations.
    initial_distribution : array-like or None
        Starting input distribution p(x).  If None, uses uniform.

    Returns
    -------
    ChannelCapacity
        Channel capacity with optimal input distribution.
    """
    W = _as_prob(transition_matrix)
    n_x, n_y = W.shape

    # Initialize input distribution
    if initial_distribution is not None:
        r = _as_prob(initial_distribution).copy()
    else:
        r = np.full(n_x, 1.0 / n_x)

    converged = False
    lower = 0.0
    upper = float("inf")

    for it in range(1, max_iterations + 1):
        # Step 1: Compute output distribution q(y) = Σ_x r(x) W(y|x)
        q = r @ W  # shape (n_y,)
        q = np.maximum(q, 1e-300)

        # Step 2: Compute c(x) = exp(Σ_y W(y|x) log(W(y|x)/q(y)))
        # = Π_y (W(y|x)/q(y))^{W(y|x)}
        log_ratio = np.zeros((n_x, n_y))
        mask = W > 0
        log_ratio[mask] = np.log(W[mask]) - np.log(np.broadcast_to(q, (n_x, n_y))[mask])
        c_x = np.exp(np.sum(W * log_ratio, axis=1))  # shape (n_x,)

        # Bounds
        I_lower = math.log(np.dot(r, c_x)) / _LOG2
        I_upper = math.log(np.max(c_x)) / _LOG2
        lower = max(lower, I_lower)
        upper = I_upper

        if upper - lower < tolerance:
            converged = True
            # Update r one last time
            r = r * c_x
            r /= r.sum()
            break

        # Step 3: Update input distribution
        r = r * c_x
        total = r.sum()
        if total > 0:
            r /= total
        else:
            r = np.full(n_x, 1.0 / n_x)

    capacity = (lower + upper) / 2.0
    return ChannelCapacity(
        capacity_bits=max(capacity, 0.0),
        optimal_input_distribution=tuple(float(x) for x in r),
        iterations=it if 'it' in dir() else 0,
        converged=converged,
        tolerance=tolerance,
        lower_bound=lower,
        upper_bound=upper,
    )


def blahut_arimoto_cost_constrained(
    transition_matrix: Union[Sequence[Sequence[float]], NDArray],
    cost_vector: Union[Sequence[float], NDArray],
    max_cost: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    lambda_init: float = 0.0,
    lambda_step: float = 0.01,
) -> ChannelCapacity:
    """Blahut-Arimoto with per-symbol cost constraint.

    Computes C(P) = max_{p(x): E[c(x)]≤P} I(X;Y).

    Uses a Lagrangian approach: optimize I(X;Y) - λ(E[c(x)] - P)
    and bisect on λ until the cost constraint is met.

    Parameters
    ----------
    transition_matrix : 2-D array-like
        Channel transition matrix p(y|x).
    cost_vector : array-like
        Cost c(x) for each input symbol.
    max_cost : float
        Maximum average cost P.
    tolerance : float
        Convergence tolerance.
    max_iterations : int
        Maximum iterations per BA run.
    lambda_init : float
        Initial Lagrange multiplier.
    lambda_step : float
        Step size for λ search.

    Returns
    -------
    ChannelCapacity
        Cost-constrained channel capacity.
    """
    W = _as_prob(transition_matrix)
    c = np.asarray(cost_vector, dtype=np.float64)
    n_x = W.shape[0]

    def _run_with_lambda(lam: float) -> Tuple[NDArray, float, float]:
        """Run BA with cost penalty λ, return (dist, capacity, avg_cost)."""
        r = np.full(n_x, 1.0 / n_x)
        for _ in range(max_iterations):
            q = np.maximum(r @ W, 1e-300)
            log_ratio = np.zeros_like(W)
            mask = W > 0
            log_ratio[mask] = np.log(W[mask]) - np.log(
                np.broadcast_to(q, W.shape)[mask]
            )
            c_x = np.exp(np.sum(W * log_ratio, axis=1) - lam * c)
            r_new = r * c_x
            total = r_new.sum()
            if total > 0:
                r_new /= total
            else:
                r_new = np.full(n_x, 1.0 / n_x)
            if np.max(np.abs(r_new - r)) < tolerance:
                r = r_new
                break
            r = r_new
        avg_cost = float(np.dot(r, c))
        # Compute MI
        joint = r[:, None] * W
        mi = mutual_information(joint)
        return r, mi, avg_cost

    # Bisection on lambda
    lam_lo = 0.0
    lam_hi = 10.0

    # Check if unconstrained already satisfies
    r0, cap0, cost0 = _run_with_lambda(0.0)
    if cost0 <= max_cost + tolerance:
        return ChannelCapacity(
            capacity_bits=cap0,
            optimal_input_distribution=tuple(float(x) for x in r0),
            converged=True,
            tolerance=tolerance,
        )

    # Find upper bound for lambda
    while True:
        _, _, cost_hi = _run_with_lambda(lam_hi)
        if cost_hi <= max_cost:
            break
        lam_hi *= 2.0
        if lam_hi > 1e10:
            break

    # Bisect
    for _ in range(100):
        lam_mid = (lam_lo + lam_hi) / 2.0
        r_mid, cap_mid, cost_mid = _run_with_lambda(lam_mid)
        if abs(cost_mid - max_cost) < tolerance:
            return ChannelCapacity(
                capacity_bits=cap_mid,
                optimal_input_distribution=tuple(float(x) for x in r_mid),
                converged=True,
                tolerance=tolerance,
            )
        if cost_mid > max_cost:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    r_final, cap_final, _ = _run_with_lambda((lam_lo + lam_hi) / 2.0)
    return ChannelCapacity(
        capacity_bits=cap_final,
        optimal_input_distribution=tuple(float(x) for x in r_final),
        converged=False,
        tolerance=tolerance,
    )


def channel_mutual_information(
    transition_matrix: Union[Sequence[Sequence[float]], NDArray],
    input_distribution: Union[Sequence[float], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Compute I(X;Y) for a channel with given input distribution.

    Parameters
    ----------
    transition_matrix : 2-D array-like
        Channel p(y|x).
    input_distribution : array-like
        Input distribution p(x).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Mutual information in the specified unit.
    """
    W = _as_prob(transition_matrix)
    px = _as_prob(input_distribution)
    joint = px[:, None] * W
    return mutual_information(joint, base=base)


def compute_capacity(channel: Channel, **kwargs) -> ChannelCapacity:
    """Compute capacity for a Channel type object."""
    W = channel.to_numpy()
    return blahut_arimoto(W, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Specific channel types
# ═══════════════════════════════════════════════════════════════════════════

def binary_symmetric_channel(p: float) -> NDArray:
    """Transition matrix for BSC with crossover probability p.

    Parameters
    ----------
    p : float
        Crossover probability (0 ≤ p ≤ 1).

    Returns
    -------
    NDArray
        2×2 transition matrix.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("Crossover probability must be in [0, 1]")
    return np.array([[1.0 - p, p], [p, 1.0 - p]])


def bsc_capacity(p: float) -> float:
    """Capacity of the binary symmetric channel: C = 1 - h(p) bits."""
    return max(1.0 - binary_entropy(p), 0.0)


def binary_erasure_channel(epsilon: float) -> NDArray:
    """Transition matrix for BEC with erasure probability ε.

    Input alphabet: {0, 1}, Output alphabet: {0, e, 1}.

    Parameters
    ----------
    epsilon : float
        Erasure probability (0 ≤ ε ≤ 1).

    Returns
    -------
    NDArray
        2×3 transition matrix.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Erasure probability must be in [0, 1]")
    return np.array([
        [1.0 - epsilon, epsilon, 0.0],
        [0.0, epsilon, 1.0 - epsilon],
    ])


def bec_capacity(epsilon: float) -> float:
    """Capacity of the binary erasure channel: C = 1 - ε bits."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Erasure probability must be in [0, 1]")
    return 1.0 - epsilon


def z_channel(p: float) -> NDArray:
    """Transition matrix for Z-channel.

    0 → 0 with probability 1
    1 → 0 with probability p, 1 → 1 with probability 1-p.

    Parameters
    ----------
    p : float
        Crossover probability for input 1.

    Returns
    -------
    NDArray
        2×2 transition matrix.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("Crossover probability must be in [0, 1]")
    return np.array([[1.0, 0.0], [p, 1.0 - p]])


def z_channel_capacity(p: float) -> float:
    """Capacity of the Z-channel, computed via Blahut-Arimoto."""
    W = z_channel(p)
    result = blahut_arimoto(W, tolerance=1e-14)
    return result.capacity_bits


def gaussian_channel_capacity(
    power: float,
    noise_variance: float,
) -> float:
    """Capacity of the AWGN channel: C = 0.5 log₂(1 + P/N) bits.

    Parameters
    ----------
    power : float
        Signal power P.
    noise_variance : float
        Noise variance N.

    Returns
    -------
    float
        Channel capacity in bits per channel use.
    """
    if power < 0 or noise_variance <= 0:
        raise ValueError("Power must be ≥ 0 and noise variance > 0")
    return 0.5 * math.log2(1.0 + power / noise_variance)


def additive_noise_channel_capacity(
    noise_entropy_bits: float,
    power_constraint: float,
) -> float:
    """Capacity of additive noise channel with given noise entropy.

    C = max h(Y) - h(N) where h(Y) is maximized under power constraint.
    For continuous channels with power constraint, the worst-case noise
    is Gaussian, giving C = 0.5 log₂(2πe P) - noise_entropy.

    Parameters
    ----------
    noise_entropy_bits : float
        Differential entropy of the noise in bits.
    power_constraint : float
        Power constraint on the input.

    Returns
    -------
    float
        Upper bound on capacity in bits.
    """
    if power_constraint <= 0:
        return 0.0
    max_output_entropy = 0.5 * math.log2(2.0 * math.pi * math.e * power_constraint)
    return max(max_output_entropy - noise_entropy_bits, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive channel capacity models
# ═══════════════════════════════════════════════════════════════════════════

def fitts_channel_capacity(
    width: float,
    amplitude: float,
    movement_time: float,
) -> float:
    """Estimate information channel capacity from Fitts' law parameters.

    Fitts' law: MT = a + b · log₂(2A/W)
    Index of difficulty: ID = log₂(2A/W) bits
    Throughput (capacity): TP = ID / MT bits/s

    Parameters
    ----------
    width : float
        Target width W.
    amplitude : float
        Movement amplitude A.
    movement_time : float
        Observed movement time (seconds).

    Returns
    -------
    float
        Throughput in bits/second.
    """
    if width <= 0 or amplitude <= 0 or movement_time <= 0:
        raise ValueError("All parameters must be positive")
    id_bits = math.log2(2.0 * amplitude / width)
    return max(id_bits / movement_time, 0.0)


def hick_hyman_rate(
    n_alternatives: int,
    reaction_time: float,
    *,
    equal_probability: bool = True,
    probabilities: Optional[Sequence[float]] = None,
) -> float:
    """Estimate information processing rate from Hick-Hyman law.

    Hick's law: RT = a + b · H(X)
    where H(X) is the stimulus entropy.
    Information rate = H(X) / RT

    Parameters
    ----------
    n_alternatives : int
        Number of stimulus-response alternatives.
    reaction_time : float
        Observed reaction time (seconds).
    equal_probability : bool
        If True, assume equal probabilities (H = log₂ n).
    probabilities : sequence of float or None
        Stimulus probabilities if not equal.

    Returns
    -------
    float
        Information processing rate in bits/second.
    """
    if reaction_time <= 0:
        raise ValueError("Reaction time must be positive")
    if equal_probability:
        h = math.log2(n_alternatives) if n_alternatives > 0 else 0.0
    else:
        if probabilities is None:
            raise ValueError("Must provide probabilities if not equal")
        h = shannon_entropy(list(probabilities), base=2.0)
    return h / reaction_time


def visual_search_capacity(
    set_size: int,
    search_time: float,
    *,
    target_present: bool = True,
) -> float:
    """Estimate visual search information processing rate.

    Models visual search as a channel where the observer must locate
    a target among distractors.

    Parameters
    ----------
    set_size : int
        Number of items in the display.
    search_time : float
        Time to find/reject the target (seconds).
    target_present : bool
        Whether the target was present.

    Returns
    -------
    float
        Estimated bits/second of visual processing.
    """
    if set_size <= 0 or search_time <= 0:
        raise ValueError("Set size and search time must be positive")
    # Information content: log₂(set_size) for location + 1 bit for present/absent
    info_bits = math.log2(set_size) + 1.0
    return info_bits / search_time


def human_information_rate(
    task_type: str = "choice_reaction",
    *,
    n_alternatives: int = 4,
    error_rate: float = 0.0,
) -> float:
    """Estimate human information processing rate for common tasks.

    Uses established HCI literature values as defaults.

    Parameters
    ----------
    task_type : str
        One of: "choice_reaction", "reading", "typing", "pointing"
    n_alternatives : int
        Number of alternatives (for choice reaction).
    error_rate : float
        Error rate for correction (reduces effective rate).

    Returns
    -------
    float
        Estimated bits/second.
    """
    # Literature-based estimates (Card, Moran & Newell, 1983)
    base_rates = {
        "choice_reaction": 6.0,   # ~6 bits/s for choice RT
        "reading": 45.0,          # ~45 bits/s for skilled reading
        "typing": 10.0,           # ~10 bits/s for skilled typing
        "pointing": 10.5,         # ~10.5 bits/s (Fitts' law throughput)
    }
    if task_type not in base_rates:
        raise ValueError(
            f"Unknown task type: {task_type}. "
            f"Choose from {list(base_rates.keys())}"
        )
    rate = base_rates[task_type]
    # Adjust for errors: effective rate ≈ rate × (1 - H(error))
    if 0 < error_rate < 1:
        rate *= (1.0 - binary_entropy(error_rate))
    return rate


__all__ = [
    "BlahutArimotoState",
    "additive_noise_channel_capacity",
    "bec_capacity",
    "binary_erasure_channel",
    "binary_symmetric_channel",
    "blahut_arimoto",
    "blahut_arimoto_cost_constrained",
    "bsc_capacity",
    "channel_mutual_information",
    "compute_capacity",
    "fitts_channel_capacity",
    "gaussian_channel_capacity",
    "hick_hyman_rate",
    "human_information_rate",
    "visual_search_capacity",
    "z_channel",
    "z_channel_capacity",
]
