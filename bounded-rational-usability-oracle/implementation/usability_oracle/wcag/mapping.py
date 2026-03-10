"""
usability_oracle.wcag.mapping — WCAG-to-cognitive-cost mapping.

Maps WCAG violations to cognitive cost deltas using an information-theoretic
model.  Each accessibility barrier is modelled as additional uncertainty
(entropy) injected into the user's interaction, quantified in bits of
cognitive load.

The mapping integrates with the cost algebra
(:mod:`usability_oracle.algebra`) so WCAG barriers can be composed with
other usability cost factors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    WCAGPrinciple,
    WCAGResult,
    WCAGViolation,
)


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive cost model constants
# ═══════════════════════════════════════════════════════════════════════════

# Base cognitive load (bits) added per violation impact level.
# Derived from Hick-Hyman law: log2(n+1) where n is the number of
# additional alternatives the user must consider due to the barrier.
_IMPACT_BASE_BITS: Dict[ImpactLevel, float] = {
    ImpactLevel.MINOR: 0.5,      # ~1 extra alternative
    ImpactLevel.MODERATE: 1.5,   # ~2–3 extra alternatives
    ImpactLevel.SERIOUS: 3.0,    # ~7 extra alternatives
    ImpactLevel.CRITICAL: 5.0,   # ~31 extra alternatives (near-blocking)
}

# Principle-specific scaling factors.
# Perceivable barriers increase sensory processing cost.
# Operable barriers increase motor/navigation cost.
# Understandable barriers increase comprehension cost.
# Robust barriers increase uncertainty about state.
_PRINCIPLE_SCALE: Dict[WCAGPrinciple, float] = {
    WCAGPrinciple.PERCEIVABLE: 1.2,
    WCAGPrinciple.OPERABLE: 1.0,
    WCAGPrinciple.UNDERSTANDABLE: 1.3,
    WCAGPrinciple.ROBUST: 0.8,
}

# Level A violations are more fundamental; boost their cost.
_LEVEL_MULTIPLIER: Dict[ConformanceLevel, float] = {
    ConformanceLevel.A: 1.5,
    ConformanceLevel.AA: 1.0,
    ConformanceLevel.AAA: 0.7,
}

# Specific criterion overrides (bits) for well-studied barriers.
_CRITERION_OVERRIDES: Dict[str, float] = {
    "1.1.1": 4.0,   # Missing alt text: screen-reader user has zero information
    "1.4.3": 2.0,   # Low contrast: degraded signal-to-noise
    "2.1.1": 5.0,   # Keyboard inaccessible: complete task blockage for keyboard users
    "2.4.1": 2.5,   # No skip nav: linear scan cost ≈ log2(N) for N repeated blocks
    "3.1.1": 1.5,   # Missing lang: pronunciation uncertainty
    "4.1.2": 3.5,   # Missing name/role: AT user must guess the control type
}

# Variance scaling: higher impact → more variable cost experience
_IMPACT_VARIANCE_SCALE: Dict[ImpactLevel, float] = {
    ImpactLevel.MINOR: 0.1,
    ImpactLevel.MODERATE: 0.5,
    ImpactLevel.SERIOUS: 1.5,
    ImpactLevel.CRITICAL: 4.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Cost delta computation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CognitiveCostDelta:
    """The additional cognitive cost imposed by WCAG violations.

    Attributes
    ----------
    mu_delta : float
        Mean additional cost in bits.
    sigma_sq_delta : float
        Variance of the additional cost.
    kappa_delta : float
        Skewness delta (accessibility barriers tend to be right-skewed).
    lambda_delta : float
        Tail-risk delta (probability of catastrophic failure).
    violation_count : int
        Number of violations contributing to this delta.
    breakdown : dict
        Per-criterion cost breakdown.
    """

    mu_delta: float
    sigma_sq_delta: float
    kappa_delta: float
    lambda_delta: float
    violation_count: int
    breakdown: Mapping[str, float] = field(default_factory=dict)


def compute_violation_cost(violation: WCAGViolation) -> float:
    """Compute the cognitive cost (bits) of a single WCAG violation.

    The cost model is:

    .. math::
        C = \\max(C_{\\text{override}},\\; C_{\\text{base}} \\times s_p \\times m_l)

    where:
    - ``C_override`` is a criterion-specific override (if available)
    - ``C_base`` is the impact-level base cost
    - ``s_p`` is the principle scaling factor
    - ``m_l`` is the conformance level multiplier
    """
    sc_id = violation.sc_id
    impact = violation.impact
    principle = violation.criterion.principle
    level = violation.criterion.level

    # Check for criterion-specific override
    if sc_id in _CRITERION_OVERRIDES:
        override = _CRITERION_OVERRIDES[sc_id]
        # Still modulate by level
        return override * _LEVEL_MULTIPLIER.get(level, 1.0)

    base = _IMPACT_BASE_BITS.get(impact, 1.0)
    principle_scale = _PRINCIPLE_SCALE.get(principle, 1.0)
    level_mult = _LEVEL_MULTIPLIER.get(level, 1.0)

    return base * principle_scale * level_mult


def compute_violation_variance(violation: WCAGViolation) -> float:
    """Compute the variance of cognitive cost for a single violation."""
    base_var = _IMPACT_VARIANCE_SCALE.get(violation.impact, 0.5)
    level_mult = _LEVEL_MULTIPLIER.get(violation.criterion.level, 1.0)
    return base_var * level_mult


def compute_cost_delta(result: WCAGResult) -> CognitiveCostDelta:
    """Compute the aggregate cognitive cost delta from a WCAG evaluation.

    Aggregates individual violation costs assuming approximate independence
    between violations (conservative estimate).

    Parameters
    ----------
    result : WCAGResult
        Evaluation result with violations.

    Returns
    -------
    CognitiveCostDelta
        The additional cognitive cost imposed by all violations.
    """
    if not result.violations:
        return CognitiveCostDelta(
            mu_delta=0.0,
            sigma_sq_delta=0.0,
            kappa_delta=0.0,
            lambda_delta=0.0,
            violation_count=0,
        )

    costs: List[float] = []
    variances: List[float] = []
    breakdown: Dict[str, float] = defaultdict(float)

    for v in result.violations:
        cost = compute_violation_cost(v)
        var = compute_violation_variance(v)
        costs.append(cost)
        variances.append(var)
        breakdown[v.sc_id] += cost

    mu_delta = sum(costs)
    sigma_sq_delta = sum(variances)

    # Skewness: accessibility barriers typically create right-skewed cost
    # distributions (long tail of error-recovery scenarios).
    # Use the empirical skewness coefficient scaled by impact.
    costs_arr = np.array(costs)
    if len(costs_arr) > 2 and np.std(costs_arr) > 0:
        skew_raw = float(np.mean(((costs_arr - np.mean(costs_arr)) / np.std(costs_arr)) ** 3))
        kappa_delta = max(0.0, skew_raw)  # accessibility barriers are right-skewed
    else:
        kappa_delta = 0.5 * len(costs)  # default mild right skew

    # Tail risk: probability of encountering a blocking barrier.
    # Modelled as 1 - prod(1 - p_i) where p_i is the per-violation
    # blocking probability.
    blocking_probs = [
        _blocking_probability(v) for v in result.violations
    ]
    lambda_delta = 1.0 - math.prod(1.0 - p for p in blocking_probs)

    return CognitiveCostDelta(
        mu_delta=mu_delta,
        sigma_sq_delta=sigma_sq_delta,
        kappa_delta=kappa_delta,
        lambda_delta=min(1.0, lambda_delta),
        violation_count=len(result.violations),
        breakdown=dict(breakdown),
    )


def _blocking_probability(v: WCAGViolation) -> float:
    """Estimate the probability that a violation completely blocks the user.

    Critical violations have high blocking probability; minor ones are
    nearly zero.
    """
    return {
        ImpactLevel.CRITICAL: 0.4,
        ImpactLevel.SERIOUS: 0.15,
        ImpactLevel.MODERATE: 0.03,
        ImpactLevel.MINOR: 0.005,
    }.get(v.impact, 0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Integration with cost algebra
# ═══════════════════════════════════════════════════════════════════════════

def to_cost_element(delta: CognitiveCostDelta) -> Any:
    """Convert a CognitiveCostDelta to a CostElement for algebraic composition.

    Returns
    -------
    CostElement
        A cost algebra element representing the WCAG barrier cost.

    Raises
    ------
    ImportError
        If the algebra module is not available.
    """
    from usability_oracle.algebra.models import CostElement
    return CostElement(
        mu=delta.mu_delta,
        sigma_sq=delta.sigma_sq_delta,
        kappa=delta.kappa_delta,
        lambda_=delta.lambda_delta,
    )


def wcag_cost_summary(result: WCAGResult) -> Dict[str, Any]:
    """Generate a summary mapping of WCAG evaluation to cognitive costs.

    Returns a dictionary suitable for JSON serialisation or logging.
    """
    delta = compute_cost_delta(result)

    return {
        "total_cognitive_cost_bits": round(delta.mu_delta, 3),
        "cost_variance": round(delta.sigma_sq_delta, 3),
        "skewness": round(delta.kappa_delta, 3),
        "blocking_risk": round(delta.lambda_delta, 4),
        "violation_count": delta.violation_count,
        "per_criterion": {
            sc_id: round(cost, 3)
            for sc_id, cost in sorted(delta.breakdown.items(), key=lambda x: x[1], reverse=True)
        },
        "interpretation": _interpret_cost(delta.mu_delta),
    }


def _interpret_cost(mu: float) -> str:
    """Human-readable interpretation of cognitive cost."""
    if mu < 1.0:
        return "Minimal additional cognitive load"
    if mu < 5.0:
        return "Moderate additional cognitive load; some users may struggle"
    if mu < 15.0:
        return "Significant cognitive load; many users will have difficulty"
    return "Severe cognitive load; likely blocking for assistive technology users"


__all__ = [
    "CognitiveCostDelta",
    "compute_cost_delta",
    "compute_violation_cost",
    "compute_violation_variance",
    "rank_remediations_by_cost",
    "to_cost_element",
    "wcag_cost_summary",
]


def rank_remediations_by_cost(result: WCAGResult) -> List[Tuple[str, float]]:
    """Rank criteria by total cognitive cost (descending).

    Returns
    -------
    List[Tuple[str, float]]
        (sc_id, total_cost) pairs sorted by cost descending.
    """
    by_criterion: Dict[str, float] = defaultdict(float)
    for v in result.violations:
        by_criterion[v.sc_id] += compute_violation_cost(v)

    return sorted(by_criterion.items(), key=lambda x: x[1], reverse=True)
