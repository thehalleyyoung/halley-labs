"""
usability_oracle.statistics.power_analysis — Statistical power analysis.

Computes statistical power, required sample sizes, and minimum detectable
effects for usability regression tests.

Functions:
    - compute_power: power given effect size, n, alpha, test type
    - required_sample_size: sample size for target power
    - minimum_detectable_effect: smallest detectable effect at given n
    - power_curve: power as function of effect size and n
    - sequential_power: group sequential design with alpha spending
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq

from usability_oracle.statistics.types import (
    PowerAnalysisResult,
    TestType,
)


# ---------------------------------------------------------------------------
# Core power computations
# ---------------------------------------------------------------------------

def _t_test_power(
    effect_size: float, n: int, alpha: float, two_sided: bool = True,
) -> float:
    """Power of a one-sample or paired t-test.

    Power = P(reject H₀ | d, n, α)

    Uses the non-central t distribution:
        ncp = d · √n
        df  = n − 1

    For a two-sided test:
        power = P(|T| > t_crit | ncp, df)
    """
    if n < 2:
        return 0.0
    df = n - 1
    ncp = effect_size * math.sqrt(n)
    if two_sided:
        t_crit = float(sp_stats.t.ppf(1 - alpha / 2, df))
        power = (
            1.0
            - float(sp_stats.nct.cdf(t_crit, df, ncp))
            + float(sp_stats.nct.cdf(-t_crit, df, ncp))
        )
    else:
        t_crit = float(sp_stats.t.ppf(1 - alpha, df))
        power = 1.0 - float(sp_stats.nct.cdf(t_crit, df, ncp))
    return max(0.0, min(1.0, power))


def _independent_t_power(
    effect_size: float, n1: int, n2: int, alpha: float, two_sided: bool = True,
) -> float:
    """Power of a two-sample independent t-test.

    ncp = d · √(n₁·n₂ / (n₁ + n₂))
    df  = n₁ + n₂ − 2
    """
    if n1 < 2 or n2 < 2:
        return 0.0
    df = n1 + n2 - 2
    ncp = effect_size * math.sqrt(n1 * n2 / (n1 + n2))
    if two_sided:
        t_crit = float(sp_stats.t.ppf(1 - alpha / 2, df))
        power = (
            1.0
            - float(sp_stats.nct.cdf(t_crit, df, ncp))
            + float(sp_stats.nct.cdf(-t_crit, df, ncp))
        )
    else:
        t_crit = float(sp_stats.t.ppf(1 - alpha, df))
        power = 1.0 - float(sp_stats.nct.cdf(t_crit, df, ncp))
    return max(0.0, min(1.0, power))


def _mann_whitney_power(
    effect_size: float, n: int, alpha: float, two_sided: bool = True,
) -> float:
    """Approximate power for Mann-Whitney / Wilcoxon test.

    Uses the asymptotic relative efficiency (ARE) of 3/π ≈ 0.955 for
    normal distributions relative to the t-test.
    """
    are = math.pi / 3.0  # ≈ 1.047; reciprocal gives efficiency
    adjusted_d = effect_size * math.sqrt(3.0 / math.pi)
    return _t_test_power(adjusted_d, n, alpha, two_sided)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    test_type: TestType = TestType.WELCH_T,
    two_sided: bool = True,
) -> float:
    """Compute statistical power for given parameters.

    Parameters:
        effect_size: Cohen's d (standardised effect).
        n: Sample size per group (paired) or per arm (independent).
        alpha: Significance level.
        test_type: Type of hypothesis test.
        two_sided: Whether the test is two-sided.

    Returns:
        Power (probability of correctly rejecting H₀).
    """
    if effect_size == 0.0:
        return alpha if two_sided else alpha  # size under H₀
    if test_type in (TestType.WELCH_T, TestType.BOOTSTRAP, TestType.BAYESIAN):
        return _t_test_power(effect_size, n, alpha, two_sided)
    if test_type == TestType.MANN_WHITNEY_U:
        return _mann_whitney_power(effect_size, n, alpha, two_sided)
    if test_type == TestType.PERMUTATION:
        return _t_test_power(effect_size, n, alpha, two_sided)
    # Default fallback
    return _t_test_power(effect_size, n, alpha, two_sided)


def required_sample_size(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
    test_type: TestType = TestType.WELCH_T,
    two_sided: bool = True,
) -> PowerAnalysisResult:
    """Compute minimum sample size for desired power.

    Searches for the smallest n such that power(n) ≥ target power.

    Parameters:
        effect_size: Minimum effect size to detect (Cohen's d).
        power: Target statistical power (1 − β).
        alpha: Significance level.
        test_type: Assumed test type.
        two_sided: Whether the test is two-sided.

    Returns:
        PowerAnalysisResult with required sample size.
    """
    if effect_size == 0.0:
        raise ValueError("Cannot compute sample size for zero effect size.")

    for n in range(2, 1_000_001):
        pwr = compute_power(effect_size, n, alpha, test_type, two_sided)
        if pwr >= power:
            return PowerAnalysisResult(
                target_effect_size=effect_size,
                alpha=alpha,
                power=power,
                required_sample_size=n,
                actual_power=pwr,
                test_type=test_type,
            )
    return PowerAnalysisResult(
        target_effect_size=effect_size, alpha=alpha, power=power,
        required_sample_size=1_000_000, actual_power=pwr,
        test_type=test_type,
    )


def minimum_detectable_effect(
    n: int,
    power: float = 0.80,
    alpha: float = 0.05,
    test_type: TestType = TestType.WELCH_T,
    two_sided: bool = True,
) -> float:
    """Compute smallest detectable effect size at given n and power.

    Uses Brent's method to find d such that power(d, n) = target.

    Parameters:
        n: Sample size per group.
        power: Target power.
        alpha: Significance level.
        test_type: Test type.
        two_sided: Whether the test is two-sided.

    Returns:
        Minimum detectable Cohen's d.
    """
    if n < 2:
        return float("inf")

    def objective(d: float) -> float:
        return compute_power(d, n, alpha, test_type, two_sided) - power

    # Check if the power at d=10 is sufficient
    if objective(10.0) < 0:
        return float("inf")

    try:
        result = brentq(objective, 1e-6, 10.0, xtol=1e-4)
        return float(result)
    except ValueError:
        return float("inf")


def power_curve(
    effect_sizes: Sequence[float],
    n_range: Sequence[int],
    alpha: float = 0.05,
    test_type: TestType = TestType.WELCH_T,
    two_sided: bool = True,
) -> Dict[float, List[Tuple[int, float]]]:
    """Compute power as a function of effect size and sample size.

    Parameters:
        effect_sizes: Effect sizes to evaluate.
        n_range: Sample sizes to evaluate.
        alpha: Significance level.
        test_type: Test type.
        two_sided: Whether the test is two-sided.

    Returns:
        Dictionary mapping each effect size to a list of (n, power) tuples.
    """
    result: Dict[float, List[Tuple[int, float]]] = {}
    for d in effect_sizes:
        curve: List[Tuple[int, float]] = []
        for n in n_range:
            pwr = compute_power(d, n, alpha, test_type, two_sided)
            curve.append((n, pwr))
        result[d] = curve
    return result


def sequential_power(
    alpha_spending: Sequence[float],
    n_max: int,
    effect_size: float,
    test_type: TestType = TestType.WELCH_T,
) -> List[Tuple[int, float, float]]:
    """Group sequential design with alpha-spending function.

    Computes cumulative power and alpha spent at each interim analysis.

    Parameters:
        alpha_spending: Alpha to spend at each look (must sum ≤ overall α).
        n_max: Maximum total sample size.
        effect_size: Target effect size.
        test_type: Test type.

    Returns:
        List of (n_at_look, alpha_spent, cumulative_power) tuples.
    """
    k = len(alpha_spending)
    if k == 0:
        return []
    total_alpha = sum(alpha_spending)
    results: List[Tuple[int, float, float]] = []
    n_per_look = max(2, n_max // k)
    cum_power = 0.0

    for i, a_i in enumerate(alpha_spending):
        n_i = min((i + 1) * n_per_look, n_max)
        # Power at this look with its alpha level
        pwr_i = compute_power(effect_size, n_i, a_i, test_type, two_sided=True)
        # Cumulative power ≈ 1 - Π(1 - power_i) (simplified independence approx)
        cum_power = 1.0 - (1.0 - cum_power) * (1.0 - pwr_i)
        results.append((n_i, a_i, cum_power))

    return results
