"""
usability_oracle.differential.local_dp — Local differential privacy.

Protocols where each user perturbs their own data before sending it to
the aggregator: randomized response, RAPPOR, optimal local randomizers,
frequency estimation, heavy-hitter detection, and usability event logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Counter, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.differential.types import PrivacyBudget


# ═══════════════════════════════════════════════════════════════════════════
# Randomized response for surveys
# ═══════════════════════════════════════════════════════════════════════════


def rr_survey_binary(
    true_answer: bool,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> bool:
    """Local randomized response for a binary survey question.

    Reports truth with probability  p = e^ε / (1 + e^ε).
    Satisfies ε-LDP.

    Parameters
    ----------
    true_answer : bool
        The user's true answer.
    epsilon : float
        Local privacy parameter.
    rng : optional Generator

    Returns
    -------
    bool
        Perturbed answer.
    """
    rng = rng or np.random.default_rng()
    p = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    return true_answer if rng.random() < p else (not true_answer)


def rr_estimate_proportion(
    reports: Sequence[bool],
    epsilon: float,
) -> float:
    """Estimate the true proportion of ``True`` from randomized-response reports.

    Applies the unbiasing formula:
        p̂ = (observed_fraction · (e^ε + 1) − 1) / (e^ε − 1)

    Parameters
    ----------
    reports : Sequence[bool]
        Perturbed reports from rr_survey_binary.
    epsilon : float
        The ε used during perturbation.

    Returns
    -------
    float
        Estimated true proportion in [0, 1].
    """
    n = len(reports)
    if n == 0:
        return 0.0
    observed = sum(1 for r in reports if r) / n
    e_eps = math.exp(epsilon)
    estimate = (observed * (e_eps + 1.0) - 1.0) / (e_eps - 1.0)
    return max(0.0, min(1.0, estimate))


def rr_survey_categorical(
    true_value: int,
    n_categories: int,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Generalised randomized response for categorical data.

    Reports the true category with probability  e^ε / (e^ε + d − 1).
    Satisfies ε-LDP.

    Parameters
    ----------
    true_value : int
        Index in [0, n_categories).
    n_categories : int
        Domain size d ≥ 2.
    epsilon : float
        Local privacy parameter.
    rng : optional Generator

    Returns
    -------
    int
        Perturbed category.
    """
    if n_categories < 2:
        raise ValueError("n_categories must be >= 2")
    rng = rng or np.random.default_rng()
    p = math.exp(epsilon) / (math.exp(epsilon) + n_categories - 1)
    if rng.random() < p:
        return true_value
    other = int(rng.integers(0, n_categories - 1))
    if other >= true_value:
        other += 1
    return other


# ═══════════════════════════════════════════════════════════════════════════
# RAPPOR protocol
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RAPPORConfig:
    """Configuration for the RAPPOR protocol.

    Attributes
    ----------
    n_hash : int
        Number of hash functions (Bloom filter size *h*).
    n_bits : int
        Bloom filter width (number of bits *k*).
    f : float
        Permanent randomized response probability.
    p : float
        Instantaneous randomized response probability for reporting 1.
    q : float
        Instantaneous randomized response probability for reporting 0.
    """

    n_hash: int = 2
    n_bits: int = 128
    f: float = 0.5
    p: float = 0.75
    q: float = 0.25

    @property
    def epsilon_permanent(self) -> float:
        """Privacy parameter of the permanent RR step."""
        if self.f <= 0 or self.f >= 1:
            return float("inf")
        return math.log((1.0 - self.f / 2.0) / (self.f / 2.0))

    @property
    def epsilon_instantaneous(self) -> float:
        """Privacy parameter of the instantaneous RR step."""
        if self.q <= 0 or self.p >= 1:
            return float("inf")
        return math.log(self.p * (1.0 - self.q) / (self.q * (1.0 - self.p)))


def rappor_encode(
    value: str,
    config: RAPPORConfig,
    *,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.uint8]:
    """RAPPOR encoding: Bloom filter → permanent RR → instantaneous RR.

    Parameters
    ----------
    value : str
        The user's true value (e.g. a string identifier).
    config : RAPPORConfig
        Protocol parameters.
    rng : optional Generator

    Returns
    -------
    NDArray[np.uint8]
        Bit vector of length ``config.n_bits`` (the reported value).
    """
    rng = rng or np.random.default_rng()

    # Step 1: Bloom filter encoding
    bloom = np.zeros(config.n_bits, dtype=np.uint8)
    for h in range(config.n_hash):
        # Deterministic hash using value + hash index
        hash_val = hash((value, h)) % config.n_bits
        bloom[hash_val] = 1

    # Step 2: Permanent randomized response (memoised)
    permanent = np.copy(bloom)
    for i in range(config.n_bits):
        r = rng.random()
        if r < config.f / 2.0:
            permanent[i] = 1
        elif r < config.f:
            permanent[i] = 0
        # else: keep the Bloom filter bit

    # Step 3: Instantaneous randomized response
    reported = np.zeros(config.n_bits, dtype=np.uint8)
    for i in range(config.n_bits):
        if permanent[i] == 1:
            reported[i] = 1 if rng.random() < config.p else 0
        else:
            reported[i] = 1 if rng.random() < config.q else 0

    return reported


def rappor_aggregate(
    reports: Sequence[NDArray[np.uint8]],
    config: RAPPORConfig,
) -> NDArray[np.floating[Any]]:
    """Aggregate RAPPOR reports to estimate per-bit frequencies.

    Applies the unbiasing formula to each bit position.

    Parameters
    ----------
    reports : Sequence[NDArray]
        Collection of RAPPOR-encoded bit vectors.
    config : RAPPORConfig

    Returns
    -------
    NDArray
        Estimated true per-bit frequency (proportion of users with bit=1
        in their Bloom filter).
    """
    n = len(reports)
    if n == 0:
        return np.zeros(config.n_bits)
    stacked = np.stack(reports).astype(np.float64)
    observed_freq = stacked.mean(axis=0)

    # Unbiasing
    # Expected observed frequency for bit b:
    #   E[report_b] = (1-f/2) * p * true_b + (f/2) * p + ...
    # Simplified: E[b_obs] = q + (p - q) * ( (1 - f/2) true_b + f/2 )
    # Solve for true_b:
    f, p, q = config.f, config.p, config.q
    denom = (p - q) * (1.0 - f / 2.0)
    if abs(denom) < 1e-12:
        return observed_freq  # degenerate config
    estimated = (observed_freq - q - (p - q) * f / 2.0) / denom
    return np.clip(estimated, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Optimal local randomizer
# ═══════════════════════════════════════════════════════════════════════════


def optimal_local_randomizer(
    true_value: float,
    lower: float,
    upper: float,
    epsilon: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Optimal unary encoding local randomizer for a bounded numeric value.

    For ε ≤ ln(2), uses symmetric Laplace-like noise.
    For ε > ln(2), uses the Duchi–Jordan–Wainwright optimal mechanism.

    Satisfies ε-LDP.

    Parameters
    ----------
    true_value : float
        The user's true value in [lower, upper].
    lower, upper : float
        Public bounds on the value.
    epsilon : float
        Local privacy parameter.
    rng : optional Generator

    Returns
    -------
    float
        Perturbed value.
    """
    rng = rng or np.random.default_rng()
    # Normalise to [−1, 1]
    if upper <= lower:
        return true_value
    t = 2.0 * (true_value - lower) / (upper - lower) - 1.0
    t = max(-1.0, min(1.0, t))

    e_eps = math.exp(epsilon)
    if epsilon <= math.log(2.0):
        # Staircase mechanism
        p = (e_eps - 1.0) / (2.0 * e_eps + 2.0)
        prob_pos = 0.5 + p * t
        out = 1.0 if rng.random() < prob_pos else -1.0
        out = out * (e_eps + 1.0) / (e_eps - 1.0)
    else:
        # Piecewise mechanism (Duchi et al.)
        C = (e_eps / 2.0 + 1.0) / (e_eps - 1.0)
        prob_pos = (e_eps - 1.0) / (2.0 * e_eps) * t + 0.5
        if rng.random() < prob_pos:
            out = float(rng.uniform(0, C))
        else:
            out = float(rng.uniform(-C, 0))

    # Denormalise
    return (out + 1.0) / 2.0 * (upper - lower) + lower


def optimal_local_mean_estimate(
    reports: Sequence[float],
    lower: float,
    upper: float,
    epsilon: float,
) -> float:
    """Estimate the true mean from optimal local randomizer reports.

    Parameters
    ----------
    reports : Sequence[float]
        Perturbed reports from optimal_local_randomizer.
    lower, upper : float
        Public bounds.
    epsilon : float
        The ε used.

    Returns
    -------
    float
        Estimated mean.
    """
    if len(reports) == 0:
        return (lower + upper) / 2.0
    return float(np.mean(reports))


# ═══════════════════════════════════════════════════════════════════════════
# Frequency estimation from LDP reports
# ═══════════════════════════════════════════════════════════════════════════


def ldp_frequency_estimate(
    reports: Sequence[int],
    domain_size: int,
    epsilon: float,
) -> NDArray[np.floating[Any]]:
    """Estimate category frequencies from LDP categorical reports.

    Assumes reports were generated using generalised randomized response.

    Parameters
    ----------
    reports : Sequence[int]
        Perturbed category indices.
    domain_size : int
        Number of categories.
    epsilon : float
        The ε used for perturbation.

    Returns
    -------
    NDArray
        Estimated frequency for each category (sums to ≈ 1).
    """
    n = len(reports)
    if n == 0:
        return np.ones(domain_size) / domain_size

    counts = np.bincount(np.asarray(reports), minlength=domain_size).astype(np.float64)
    observed = counts / n

    e_eps = math.exp(epsilon)
    d = domain_size
    # Unbias: p_true = (observed * (e^ε + d - 1) - 1) / (e^ε - 1)
    estimated = (observed * (e_eps + d - 1.0) - 1.0) / (e_eps - 1.0)
    # Project onto the probability simplex
    estimated = np.maximum(estimated, 0.0)
    total = estimated.sum()
    if total > 0:
        estimated = estimated / total
    else:
        estimated = np.ones(domain_size) / domain_size
    return estimated


# ═══════════════════════════════════════════════════════════════════════════
# Heavy hitter detection under LDP
# ═══════════════════════════════════════════════════════════════════════════


def ldp_heavy_hitters(
    reports: Sequence[int],
    domain_size: int,
    epsilon: float,
    threshold: float = 0.05,
) -> List[Tuple[int, float]]:
    """Detect heavy hitters (frequent items) under LDP.

    Estimates frequencies from LDP reports and returns categories
    whose estimated frequency exceeds *threshold*.

    Parameters
    ----------
    reports : Sequence[int]
        LDP-perturbed category reports.
    domain_size : int
        Number of categories.
    epsilon : float
        The ε used.
    threshold : float
        Minimum frequency to qualify as a heavy hitter.

    Returns
    -------
    list[tuple[int, float]]
        List of (category_index, estimated_frequency) for heavy hitters,
        sorted by frequency descending.
    """
    freqs = ldp_frequency_estimate(reports, domain_size, epsilon)
    heavy = [(int(i), float(freqs[i])) for i in range(domain_size) if freqs[i] >= threshold]
    heavy.sort(key=lambda x: x[1], reverse=True)
    return heavy


# ═══════════════════════════════════════════════════════════════════════════
# Local DP for usability event logging
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LDPEventConfig:
    """Configuration for LDP usability event logging.

    Attributes
    ----------
    event_types : list[str]
        Enumeration of event type names.
    epsilon : float
        Per-event local privacy parameter.
    time_granularity_s : float
        Temporal granularity for time quantisation (seconds).
    time_range_s : tuple[float, float]
        Valid time range (start, end) in seconds.
    """

    event_types: List[str] = field(default_factory=list)
    epsilon: float = 1.0
    time_granularity_s: float = 60.0
    time_range_s: Tuple[float, float] = (0.0, 3600.0)

    @property
    def n_event_types(self) -> int:
        return len(self.event_types)

    @property
    def n_time_slots(self) -> int:
        lo, hi = self.time_range_s
        return max(1, int((hi - lo) / self.time_granularity_s))


def ldp_encode_event(
    event_type: str,
    event_time_s: float,
    config: LDPEventConfig,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int]:
    """Encode a usability event under local DP.

    Perturbs both the event type (categorical RR) and the time slot
    (categorical RR), splitting ε equally between them.

    Parameters
    ----------
    event_type : str
        True event type name.
    event_time_s : float
        True event timestamp in seconds.
    config : LDPEventConfig
        LDP event configuration.
    rng : optional Generator

    Returns
    -------
    tuple[int, int]
        (perturbed_event_type_index, perturbed_time_slot).
    """
    rng = rng or np.random.default_rng()
    eps_half = config.epsilon / 2.0

    # Event type perturbation
    try:
        type_idx = config.event_types.index(event_type)
    except ValueError:
        type_idx = 0
    perturbed_type = rr_survey_categorical(
        type_idx, config.n_event_types, eps_half, rng=rng,
    )

    # Time slot perturbation
    lo, hi = config.time_range_s
    slot = int((event_time_s - lo) / config.time_granularity_s)
    slot = max(0, min(slot, config.n_time_slots - 1))
    perturbed_slot = rr_survey_categorical(
        slot, config.n_time_slots, eps_half, rng=rng,
    )

    return perturbed_type, perturbed_slot


def ldp_aggregate_events(
    reports: Sequence[Tuple[int, int]],
    config: LDPEventConfig,
) -> Dict[str, NDArray[np.floating[Any]]]:
    """Aggregate LDP-encoded usability events.

    Returns estimated frequency distributions over event types and time
    slots.

    Parameters
    ----------
    reports : Sequence[tuple[int, int]]
        Collection of (perturbed_event_type, perturbed_time_slot).
    config : LDPEventConfig

    Returns
    -------
    dict with keys:
        "event_type_freq" : NDArray — estimated event type frequencies.
        "time_slot_freq"  : NDArray — estimated time slot frequencies.
    """
    eps_half = config.epsilon / 2.0
    type_reports = [r[0] for r in reports]
    slot_reports = [r[1] for r in reports]

    type_freq = ldp_frequency_estimate(type_reports, config.n_event_types, eps_half)
    slot_freq = ldp_frequency_estimate(slot_reports, config.n_time_slots, eps_half)

    return {
        "event_type_freq": type_freq,
        "time_slot_freq": slot_freq,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Utility-privacy tradeoff analysis
# ═══════════════════════════════════════════════════════════════════════════


def rr_variance(epsilon: float, n: int) -> float:
    """Variance of the proportion estimator from binary randomised response.

    Var(p̂) = (e^ε + 1)² / (n · (e^ε − 1)²)

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    n : int
        Number of reports.

    Returns
    -------
    float
        Estimator variance.
    """
    e_eps = math.exp(epsilon)
    return (e_eps + 1.0) ** 2 / (n * (e_eps - 1.0) ** 2)


def ldp_utility_tradeoff(
    epsilon_range: Sequence[float],
    n: int,
    domain_size: int = 2,
) -> List[Tuple[float, float]]:
    """Compute utility-privacy tradeoff curve for LDP frequency estimation.

    Returns (ε, MSE) pairs for different ε values.

    Parameters
    ----------
    epsilon_range : Sequence[float]
        Range of ε values to evaluate.
    n : int
        Number of reports.
    domain_size : int
        Number of categories.

    Returns
    -------
    list[tuple[float, float]]
        (epsilon, expected_MSE) pairs.
    """
    results = []
    for eps in epsilon_range:
        e_eps = math.exp(eps)
        d = domain_size
        # MSE per category for categorical RR
        variance_per_cat = (e_eps + d - 1.0) ** 2 / (n * (e_eps - 1.0) ** 2) if e_eps > 1 else float("inf")
        # Average MSE across categories
        mse = variance_per_cat / d
        results.append((float(eps), float(mse)))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Minimax optimal mechanisms
# ═══════════════════════════════════════════════════════════════════════════


def minimax_mse_bound(epsilon: float, n: int) -> float:
    """Minimax lower bound on MSE for mean estimation under ε-LDP.

    The minimax rate for estimating a mean in [−1, 1] under ε-LDP is
    Θ(1 / (n · min(ε², 1))).

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    n : int
        Number of users.

    Returns
    -------
    float
        Minimax MSE lower bound.
    """
    if n <= 0:
        return float("inf")
    eps_eff = min(epsilon ** 2, 1.0)
    return 1.0 / (n * eps_eff)


def minimax_sample_complexity(
    epsilon: float,
    target_mse: float,
    domain_size: int = 2,
) -> int:
    """Minimum sample size to achieve target MSE under ε-LDP.

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    target_mse : float
        Desired MSE.
    domain_size : int
        Domain size.

    Returns
    -------
    int
        Minimum number of users.
    """
    if target_mse <= 0:
        return 0
    e_eps = math.exp(epsilon)
    d = domain_size
    # From categorical RR variance
    variance_per_sample = (e_eps + d - 1.0) ** 2 / ((e_eps - 1.0) ** 2) if e_eps > 1 else float("inf")
    if variance_per_sample == float("inf"):
        return 0
    n = variance_per_sample / (target_mse * d)
    return max(1, int(math.ceil(n)))


__all__ = [
    # Survey RR
    "rr_survey_binary",
    "rr_estimate_proportion",
    "rr_survey_categorical",
    # RAPPOR
    "RAPPORConfig",
    "rappor_encode",
    "rappor_aggregate",
    # Optimal local randomizer
    "optimal_local_randomizer",
    "optimal_local_mean_estimate",
    # Frequency estimation
    "ldp_frequency_estimate",
    "ldp_heavy_hitters",
    # Usability event logging
    "LDPEventConfig",
    "ldp_encode_event",
    "ldp_aggregate_events",
    # Tradeoff analysis
    "rr_variance",
    "ldp_utility_tradeoff",
    # Minimax
    "minimax_mse_bound",
    "minimax_sample_complexity",
]
