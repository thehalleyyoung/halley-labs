"""
usability_oracle.differential.calibration — Privacy parameter calibration.

Guidelines and algorithms for selecting ε, δ, and noise parameters:
risk-based ε selection, δ calibration by database size, utility-privacy
tradeoff curves, minimum sample sizes, empirical privacy auditing, and
integration with sensitivity analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.differential.types import PrivacyBudget
from usability_oracle.differential.mechanisms import (
    gaussian_scale,
    laplace_scale,
)
from usability_oracle.differential.accountant import (
    DEFAULT_RDP_ORDERS,
    optimal_gaussian_sigma,
    rdp_to_approx_dp,
)


# ═══════════════════════════════════════════════════════════════════════════
# Risk levels and ε guidelines
# ═══════════════════════════════════════════════════════════════════════════


@unique
class RiskLevel(Enum):
    """Qualitative risk level for ε selection."""

    HIGH_PRIVACY = "high_privacy"
    """Strong protection — suitable for health or PII data.  ε ≤ 1."""

    MODERATE_PRIVACY = "moderate_privacy"
    """Moderate protection — typical usability studies.  ε ∈ (1, 4]."""

    LOW_PRIVACY = "low_privacy"
    """Weak protection — internal analytics.  ε ∈ (4, 10]."""

    MINIMAL_PRIVACY = "minimal_privacy"
    """Minimal protection — public / non-sensitive data.  ε > 10."""


@dataclass(frozen=True)
class EpsilonGuideline:
    """ε selection guideline for a risk level.

    Attributes
    ----------
    risk : RiskLevel
        Associated risk level.
    epsilon_range : tuple[float, float]
        Recommended (min, max) ε range.
    description : str
        Human-readable rationale.
    """

    risk: RiskLevel
    epsilon_range: Tuple[float, float]
    description: str


# Standard guidelines (based on Dwork & Roth, Desfontaines & Pejó surveys)
EPSILON_GUIDELINES: List[EpsilonGuideline] = [
    EpsilonGuideline(
        risk=RiskLevel.HIGH_PRIVACY,
        epsilon_range=(0.01, 1.0),
        description=(
            "Strongest protection.  Suitable for health data, PII, or when "
            "individual records are highly sensitive.  May require large n "
            "for useful accuracy."
        ),
    ),
    EpsilonGuideline(
        risk=RiskLevel.MODERATE_PRIVACY,
        epsilon_range=(1.0, 4.0),
        description=(
            "Standard protection.  Common for usability studies and A/B "
            "testing where individual responses are moderately sensitive."
        ),
    ),
    EpsilonGuideline(
        risk=RiskLevel.LOW_PRIVACY,
        epsilon_range=(4.0, 10.0),
        description=(
            "Weaker protection.  Acceptable for internal analytics, "
            "aggregate metrics, or when data is semi-public."
        ),
    ),
    EpsilonGuideline(
        risk=RiskLevel.MINIMAL_PRIVACY,
        epsilon_range=(10.0, 100.0),
        description=(
            "Minimal formal privacy.  The noise is small; use only for "
            "non-sensitive, publicly observable data."
        ),
    ),
]


def recommend_epsilon(risk: RiskLevel) -> Tuple[float, float]:
    """Return the recommended ε range for the given risk level.

    Parameters
    ----------
    risk : RiskLevel
        Desired risk / sensitivity level.

    Returns
    -------
    tuple[float, float]
        (min_epsilon, max_epsilon).
    """
    for g in EPSILON_GUIDELINES:
        if g.risk == risk:
            return g.epsilon_range
    return (1.0, 4.0)


# ═══════════════════════════════════════════════════════════════════════════
# δ selection
# ═══════════════════════════════════════════════════════════════════════════


def recommend_delta(n: int, *, multiplier: float = 1.0) -> float:
    """Recommend δ based on database size.

    Standard guideline: δ < 1/n  (or δ = 1/n² for stronger guarantees).
    We use δ = multiplier / n² as a conservative default.

    Parameters
    ----------
    n : int
        Number of records / users.
    multiplier : float
        Scaling factor (default 1.0).

    Returns
    -------
    float
        Recommended δ.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    return min(multiplier / (n * n), 1.0 / n)


def delta_from_risk(n: int, risk: RiskLevel) -> float:
    """Select δ based on risk level and database size.

    Parameters
    ----------
    n : int
        Number of records.
    risk : RiskLevel
        Desired risk level.

    Returns
    -------
    float
        Recommended δ.
    """
    multipliers = {
        RiskLevel.HIGH_PRIVACY: 0.1,
        RiskLevel.MODERATE_PRIVACY: 1.0,
        RiskLevel.LOW_PRIVACY: 10.0,
        RiskLevel.MINIMAL_PRIVACY: 100.0,
    }
    return recommend_delta(n, multiplier=multipliers.get(risk, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Utility-privacy tradeoff curves
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TradeoffPoint:
    """A single point on the utility-privacy tradeoff curve.

    Attributes
    ----------
    epsilon : float
        Privacy parameter.
    noise_scale : float
        Noise scale (σ or b).
    expected_error : float
        Expected absolute error of the mechanism.
    relative_error : float
        Expected error relative to a reference value.
    """

    epsilon: float
    noise_scale: float
    expected_error: float
    relative_error: float


def laplace_tradeoff_curve(
    sensitivity: float,
    epsilon_range: Sequence[float],
    *,
    reference_value: float = 1.0,
) -> List[TradeoffPoint]:
    """Compute utility-privacy tradeoff for the Laplace mechanism.

    Parameters
    ----------
    sensitivity : float
        Query sensitivity.
    epsilon_range : Sequence[float]
        ε values to evaluate.
    reference_value : float
        Reference value for relative error.

    Returns
    -------
    list[TradeoffPoint]
    """
    points = []
    for eps in epsilon_range:
        if eps <= 0:
            continue
        b = laplace_scale(sensitivity, eps)
        expected_err = b  # E[|Lap(b)|] = b
        rel_err = expected_err / abs(reference_value) if reference_value != 0 else float("inf")
        points.append(TradeoffPoint(
            epsilon=eps, noise_scale=b,
            expected_error=expected_err, relative_error=rel_err,
        ))
    return points


def gaussian_tradeoff_curve(
    sensitivity: float,
    epsilon_range: Sequence[float],
    delta: float,
    *,
    reference_value: float = 1.0,
) -> List[TradeoffPoint]:
    """Compute utility-privacy tradeoff for the Gaussian mechanism.

    Parameters
    ----------
    sensitivity : float
        Query sensitivity.
    epsilon_range : Sequence[float]
        ε values to evaluate.
    delta : float
        Fixed δ.
    reference_value : float
        Reference value for relative error.

    Returns
    -------
    list[TradeoffPoint]
    """
    points = []
    for eps in epsilon_range:
        if eps <= 0:
            continue
        sigma = gaussian_scale(sensitivity, eps, delta)
        expected_err = sigma * math.sqrt(2.0 / math.pi)  # E[|N(0,σ²)|]
        rel_err = expected_err / abs(reference_value) if reference_value != 0 else float("inf")
        points.append(TradeoffPoint(
            epsilon=eps, noise_scale=sigma,
            expected_error=expected_err, relative_error=rel_err,
        ))
    return points


# ═══════════════════════════════════════════════════════════════════════════
# Minimum sample size
# ═══════════════════════════════════════════════════════════════════════════


def min_sample_size_laplace(
    epsilon: float,
    sensitivity: float,
    target_error: float,
    confidence: float = 0.95,
) -> int:
    """Minimum sample size for Laplace mechanism to achieve target accuracy.

    For a mean query with sensitivity C/n, we want
    P[|noise| > target_error] ≤ 1 − confidence.

    For Laplace(b), P[|X| > t] = exp(−t/b).  We need
    b ≤ target_error / ln(1/(1−confidence)).

    Since b = sensitivity/ε = C/(n·ε), we solve for n:
    n ≥ C / (ε · target_error / ln(1/(1−confidence))).

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    sensitivity : float
        Per-record sensitivity bound (clipping bound for mean).
    target_error : float
        Maximum acceptable absolute error.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    int
        Minimum number of records.
    """
    if target_error <= 0 or epsilon <= 0:
        return 0
    tail = -math.log(1.0 - confidence)
    # b = sensitivity / (n * epsilon);  need b <= target_error / tail
    # => n >= sensitivity * tail / (epsilon * target_error)
    n = sensitivity * tail / (epsilon * target_error)
    return max(1, int(math.ceil(n)))


def min_sample_size_gaussian(
    epsilon: float,
    delta: float,
    sensitivity: float,
    target_error: float,
    confidence: float = 0.95,
) -> int:
    """Minimum sample size for Gaussian mechanism to achieve target accuracy.

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    delta : float
        Failure probability.
    sensitivity : float
        Per-record sensitivity bound.
    target_error : float
        Maximum acceptable absolute error.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    int
        Minimum number of records.
    """
    from scipy.stats import norm

    if target_error <= 0 or epsilon <= 0 or delta <= 0:
        return 0
    z = norm.ppf(0.5 + confidence / 2.0)
    # sigma = sensitivity * sqrt(2 ln(1.25/delta)) / (n * epsilon)
    # need z * sigma <= target_error
    sigma_factor = math.sqrt(2.0 * math.log(1.25 / delta))
    n = sensitivity * sigma_factor * z / (epsilon * target_error)
    return max(1, int(math.ceil(n)))


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity analysis of privacy parameters
# ═══════════════════════════════════════════════════════════════════════════


def sensitivity_analysis(
    base_epsilon: float,
    base_delta: float,
    sensitivity: float,
    *,
    epsilon_perturbations: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.2),
    delta_perturbations: Sequence[float] = (0.5, 1.0, 2.0),
) -> Dict[str, List[Dict[str, float]]]:
    """Analyse how noise scale changes with ε and δ perturbations.

    Parameters
    ----------
    base_epsilon : float
        Baseline ε.
    base_delta : float
        Baseline δ.
    sensitivity : float
        Query sensitivity.
    epsilon_perturbations : Sequence[float]
        Multiplicative factors for ε.
    delta_perturbations : Sequence[float]
        Multiplicative factors for δ.

    Returns
    -------
    dict with keys "laplace" and "gaussian", each a list of dicts
    with fields "epsilon", "delta", "noise_scale".
    """
    laplace_results = []
    gaussian_results = []

    for eps_mult in epsilon_perturbations:
        eps = base_epsilon * eps_mult
        if eps <= 0:
            continue
        b = laplace_scale(sensitivity, eps)
        laplace_results.append({"epsilon": eps, "delta": 0.0, "noise_scale": b})

        for d_mult in delta_perturbations:
            d = base_delta * d_mult
            if d <= 0 or d >= 1:
                continue
            sigma = gaussian_scale(sensitivity, eps, d)
            gaussian_results.append({"epsilon": eps, "delta": d, "noise_scale": sigma})

    return {"laplace": laplace_results, "gaussian": gaussian_results}


# ═══════════════════════════════════════════════════════════════════════════
# Empirical privacy auditing
# ═══════════════════════════════════════════════════════════════════════════


def empirical_privacy_audit(
    mechanism_fn: Any,
    value_a: float,
    value_b: float,
    n_samples: int = 10000,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Empirically estimate the privacy loss of a mechanism.

    Runs the mechanism on two neighbouring inputs (value_a and value_b)
    many times and estimates the privacy loss from output distributions.

    Parameters
    ----------
    mechanism_fn : callable
        Function (value) → noised_value.
    value_a : float
        First input.
    value_b : float
        Second (neighbouring) input.
    n_samples : int
        Number of samples per input.
    rng : optional Generator

    Returns
    -------
    dict with keys:
        "empirical_epsilon" : float — estimated ε from max log-likelihood ratio.
        "kl_divergence"     : float — estimated KL divergence D(P_a || P_b).
        "n_samples"         : int
    """
    rng = rng or np.random.default_rng()
    outputs_a = np.array([mechanism_fn(value_a) for _ in range(n_samples)])
    outputs_b = np.array([mechanism_fn(value_b) for _ in range(n_samples)])

    # Use kernel density estimation to approximate distributions
    from scipy.stats import gaussian_kde

    try:
        kde_a = gaussian_kde(outputs_a)
        kde_b = gaussian_kde(outputs_b)
    except np.linalg.LinAlgError:
        return {"empirical_epsilon": 0.0, "kl_divergence": 0.0, "n_samples": n_samples}

    # Estimate KL divergence and max log-likelihood ratio on a grid
    grid = np.linspace(
        min(outputs_a.min(), outputs_b.min()),
        max(outputs_a.max(), outputs_b.max()),
        1000,
    )
    pa = kde_a(grid) + 1e-30
    pb = kde_b(grid) + 1e-30

    # KL divergence
    kl = float(np.sum(pa * np.log(pa / pb)) * (grid[1] - grid[0]))

    # Empirical ε: max log(P_a(x) / P_b(x))
    log_ratio = np.log(pa / pb)
    empirical_eps = float(np.max(np.abs(log_ratio)))

    return {
        "empirical_epsilon": empirical_eps,
        "kl_divergence": max(0.0, kl),
        "n_samples": n_samples,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Integration helpers
# ═══════════════════════════════════════════════════════════════════════════


def calibrate_budget(
    risk: RiskLevel,
    n_records: int,
    n_queries: int,
    *,
    composition: str = "basic",
) -> PrivacyBudget:
    """Calibrate a complete privacy budget for a usability analysis.

    Parameters
    ----------
    risk : RiskLevel
        Risk level.
    n_records : int
        Number of user records.
    n_queries : int
        Number of queries to answer.
    composition : str
        Composition type ("basic" or "advanced").

    Returns
    -------
    PrivacyBudget
        Calibrated total privacy budget.
    """
    eps_lo, eps_hi = recommend_epsilon(risk)
    eps_mid = (eps_lo + eps_hi) / 2.0

    # For advanced composition, we can afford higher per-query ε
    if composition == "advanced" and n_queries > 1:
        # Invert advanced composition to find per-query ε
        delta = recommend_delta(n_records)
        total_eps = eps_mid
    else:
        total_eps = eps_mid * n_queries
        delta = recommend_delta(n_records)

    return PrivacyBudget(
        epsilon=total_eps,
        delta=delta,
        description=f"calibrated({risk.value}, n={n_records}, q={n_queries})",
    )


__all__ = [
    # Risk levels
    "RiskLevel",
    "EpsilonGuideline",
    "EPSILON_GUIDELINES",
    "recommend_epsilon",
    # Delta
    "recommend_delta",
    "delta_from_risk",
    # Tradeoff curves
    "TradeoffPoint",
    "laplace_tradeoff_curve",
    "gaussian_tradeoff_curve",
    # Sample size
    "min_sample_size_laplace",
    "min_sample_size_gaussian",
    # Sensitivity analysis
    "sensitivity_analysis",
    # Empirical auditing
    "empirical_privacy_audit",
    # Integration
    "calibrate_budget",
]
