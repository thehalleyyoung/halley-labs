"""
Tests for sensitivity analysis methods.

Covers E-value computation, omitted variable bias, Rosenbaum bounds,
and comparison against known results from the literature.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# E-value computation
# ---------------------------------------------------------------------------


def e_value_rr(rr: float) -> float:
    """Compute E-value for a risk ratio point estimate.

    The E-value is the minimum strength of association that an unmeasured
    confounder would need to have with both treatment and outcome to explain
    away the observed effect.  (VanderWeele & Ding, 2017)

    E-value = RR + sqrt(RR * (RR - 1))  for RR >= 1
    """
    if rr < 1:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1))


def e_value_hr(hr: float) -> float:
    """Approximate E-value for a hazard ratio using RR approximation."""
    return e_value_rr(hr)


def e_value_or(or_val: float, prevalence: float = 0.1) -> float:
    """Compute E-value for an odds ratio.

    When the outcome is rare (prevalence < 15%), OR ≈ RR, so we can
    use the RR formula.  Otherwise, convert OR to RR first.
    """
    if prevalence < 0.15:
        return e_value_rr(or_val)
    # Convert OR to RR: RR = OR / (1 - p0 + p0 * OR) where p0 is baseline risk
    rr = or_val / (1 - prevalence + prevalence * or_val)
    return e_value_rr(rr)


def e_value_ci(rr: float, ci_bound: float) -> float:
    """E-value for the confidence interval limit closest to the null."""
    if ci_bound < 1:
        return 1.0  # CI includes null
    return e_value_rr(ci_bound)


def e_value_smd(smd: float) -> float:
    """E-value for a standardized mean difference.

    Convert SMD to approximate RR: RR ≈ exp(0.91 * SMD)
    """
    rr = math.exp(0.91 * abs(smd))
    return e_value_rr(rr)


# ---------------------------------------------------------------------------
# Omitted variable bias (Cinelli & Hazlett, 2020)
# ---------------------------------------------------------------------------


def omitted_variable_bias(
    r2_yd_x: float,
    r2_dz_x: float,
    se: float,
    dof: int,
) -> float:
    """Compute the bias due to an omitted confounder Z.

    Uses the partial R² sensitivity framework of Cinelli & Hazlett (2020).

    Parameters
    ----------
    r2_yd_x : float
        Partial R² of Y on Z after controlling for X (treatment) and
        other observed covariates.
    r2_dz_x : float
        Partial R² of D (treatment) on Z after controlling for covariates.
    se : float
        Standard error of the treatment coefficient.
    dof : int
        Degrees of freedom.

    Returns
    -------
    float
        Estimated bias in the treatment effect.
    """
    # Bias ≈ SE * sqrt(dof) * sqrt(r2_yd_x * r2_dz_x / (1 - r2_dz_x))
    numerator = math.sqrt(r2_yd_x * r2_dz_x)
    denominator = math.sqrt(1 - r2_dz_x)
    return se * math.sqrt(dof) * numerator / denominator


def robustness_value(
    r2_yd_x_max: float,
    r2_dz_x_max: float,
    treatment_effect: float,
    se: float,
    dof: int,
    alpha: float = 0.05,
) -> float:
    """Compute the robustness value (RV).

    The RV is the minimum confounder strength such that the treatment
    effect would be driven to zero (or non-significance).
    """
    q = treatment_effect / se
    # Simple approximation: RV ≈ q^2 / (q^2 + dof)
    return q * q / (q * q + dof)


def adjusted_estimate(
    original_estimate: float,
    r2_yd_x: float,
    r2_dz_x: float,
    se: float,
    dof: int,
) -> float:
    """Compute the treatment effect adjusted for confounding."""
    bias = omitted_variable_bias(r2_yd_x, r2_dz_x, se, dof)
    return original_estimate - bias


# ---------------------------------------------------------------------------
# Rosenbaum bounds
# ---------------------------------------------------------------------------


def rosenbaum_gamma_bound(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma: float = 1.0,
) -> tuple[float, float]:
    """Compute Rosenbaum sensitivity bounds for matched pairs.

    Parameters
    ----------
    treated_outcomes : np.ndarray
        Outcomes for treated units.
    control_outcomes : np.ndarray
        Outcomes for matched control units.
    gamma : float
        Sensitivity parameter (Γ >= 1). Γ=1 corresponds to no hidden bias.

    Returns
    -------
    tuple[float, float]
        (p_value_lower, p_value_upper) bounds on the p-value under
        the assumption that treatment assignment probabilities can differ
        by at most a factor of Γ.
    """
    diffs = treated_outcomes - control_outcomes
    n = len(diffs)

    if n == 0:
        return 1.0, 1.0

    # Signed rank test
    abs_diffs = np.abs(diffs)
    ranks = sp_stats.rankdata(abs_diffs)
    signs = np.sign(diffs)
    T_obs = np.sum(ranks * (signs > 0))

    # Under Γ, the probability of positive sign varies
    p_upper = gamma / (1 + gamma)
    p_lower = 1 / (1 + gamma)

    # Expected T under extreme biases
    E_upper = np.sum(ranks * p_upper)
    E_lower = np.sum(ranks * p_lower)
    Var = np.sum(ranks ** 2 * p_upper * (1 - p_upper))

    if Var < 1e-12:
        return 0.5, 0.5

    z_upper = (T_obs - E_upper) / math.sqrt(Var)
    z_lower = (T_obs - E_lower) / math.sqrt(Var)

    p_val_upper = float(1 - sp_stats.norm.cdf(z_lower))
    p_val_lower = float(1 - sp_stats.norm.cdf(z_upper))

    return max(0, p_val_lower), min(1, p_val_upper)


def rosenbaum_critical_gamma(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    alpha: float = 0.05,
    gamma_max: float = 10.0,
    tol: float = 0.01,
) -> float:
    """Find the critical Γ at which the treatment effect becomes non-significant.

    Uses bisection to find the smallest Γ such that the upper bound
    p-value exceeds α.

    Parameters
    ----------
    treated_outcomes, control_outcomes : np.ndarray
        Matched pair outcomes.
    alpha : float
        Significance level.
    gamma_max : float
        Maximum Γ to search.
    tol : float
        Tolerance for bisection.

    Returns
    -------
    float
        Critical Γ value.
    """
    lo, hi = 1.0, gamma_max

    # Check if effect survives at gamma=1
    _, p_upper = rosenbaum_gamma_bound(treated_outcomes, control_outcomes, 1.0)
    if p_upper > alpha:
        return 1.0

    # Check if effect survives at gamma_max
    _, p_upper = rosenbaum_gamma_bound(treated_outcomes, control_outcomes, gamma_max)
    if p_upper <= alpha:
        return gamma_max

    while hi - lo > tol:
        mid = (lo + hi) / 2
        _, p_upper = rosenbaum_gamma_bound(treated_outcomes, control_outcomes, mid)
        if p_upper <= alpha:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Sensitivity analysis for ATE
# ---------------------------------------------------------------------------


def ate_sensitivity_contour(
    observed_ate: float,
    se: float,
    r2_yz_grid: np.ndarray,
    r2_tz_grid: np.ndarray,
    dof: int,
) -> np.ndarray:
    """Compute bias-adjusted ATE on a grid of confounder strengths.

    Returns a matrix of adjusted ATE values.
    """
    result = np.zeros((len(r2_yz_grid), len(r2_tz_grid)))
    for i, r2_yz in enumerate(r2_yz_grid):
        for j, r2_tz in enumerate(r2_tz_grid):
            bias = omitted_variable_bias(r2_yz, r2_tz, se, dof)
            result[i, j] = observed_ate - bias
    return result


# ===================================================================
# Tests
# ===================================================================


class TestEValue:
    def test_rr_equal_one(self):
        """E-value of RR=1 should be 1 (null effect)."""
        assert e_value_rr(1.0) == pytest.approx(1.0, abs=1e-6)

    def test_rr_two(self):
        """E-value of RR=2: known result = 2 + sqrt(2*1) ≈ 3.41."""
        ev = e_value_rr(2.0)
        expected = 2.0 + math.sqrt(2.0)
        assert ev == pytest.approx(expected, abs=1e-4)

    def test_rr_three(self):
        ev = e_value_rr(3.0)
        expected = 3.0 + math.sqrt(3.0 * 2.0)
        assert ev == pytest.approx(expected, abs=1e-4)

    def test_rr_inverted(self):
        """RR < 1 should be inverted to 1/RR."""
        ev_low = e_value_rr(0.5)
        ev_high = e_value_rr(2.0)
        assert ev_low == pytest.approx(ev_high, abs=1e-6)

    def test_increasing_in_rr(self):
        """E-value should increase with RR."""
        ev1 = e_value_rr(1.5)
        ev2 = e_value_rr(2.0)
        ev3 = e_value_rr(3.0)
        assert ev1 < ev2 < ev3

    def test_smd_positive(self):
        ev = e_value_smd(0.5)
        assert ev > 1.0

    def test_smd_zero(self):
        ev = e_value_smd(0.0)
        assert ev == pytest.approx(1.0, abs=0.05)

    def test_or_rare_outcome(self):
        """With rare outcome, OR ≈ RR."""
        ev_or = e_value_or(2.0, prevalence=0.05)
        ev_rr = e_value_rr(2.0)
        assert ev_or == pytest.approx(ev_rr, abs=1e-4)

    def test_or_common_outcome(self):
        """With common outcome, OR should give different E-value."""
        ev_common = e_value_or(2.0, prevalence=0.5)
        ev_rare = e_value_or(2.0, prevalence=0.05)
        assert ev_common != pytest.approx(ev_rare, abs=0.1)

    def test_ci_below_null(self):
        """If CI bound < 1, E-value should be 1."""
        ev = e_value_ci(2.0, 0.8)
        assert ev == 1.0

    def test_ci_above_null(self):
        ev = e_value_ci(2.0, 1.5)
        assert ev > 1.0


class TestOmittedVariableBias:
    def test_zero_confounding(self):
        """Zero partial R² should give zero bias."""
        bias = omitted_variable_bias(0.0, 0.0, se=0.1, dof=100)
        assert bias == pytest.approx(0.0, abs=1e-10)

    def test_positive_bias(self):
        bias = omitted_variable_bias(0.1, 0.1, se=0.1, dof=100)
        assert bias > 0

    def test_increasing_in_r2(self):
        b1 = omitted_variable_bias(0.05, 0.05, se=0.1, dof=100)
        b2 = omitted_variable_bias(0.1, 0.1, se=0.1, dof=100)
        b3 = omitted_variable_bias(0.2, 0.2, se=0.1, dof=100)
        assert b1 < b2 < b3

    def test_adjusted_estimate(self):
        original = 2.0
        adj = adjusted_estimate(original, 0.1, 0.1, se=0.1, dof=100)
        assert adj < original

    def test_robustness_value_positive(self):
        rv = robustness_value(0.3, 0.3, 2.0, 0.5, 100)
        assert 0 < rv < 1

    def test_large_effect_high_rv(self):
        """Large treatment effect should have high robustness value."""
        rv_small = robustness_value(0.3, 0.3, 1.0, 0.5, 100)
        rv_large = robustness_value(0.3, 0.3, 5.0, 0.5, 100)
        assert rv_large > rv_small


class TestRosenbaumBounds:
    def test_no_bias_significant(self):
        """With Γ=1 (no bias) and clear effect, should be significant."""
        rng = np.random.default_rng(42)
        treated = 5.0 + rng.standard_normal(50)
        control = 0.0 + rng.standard_normal(50)
        p_lo, p_hi = rosenbaum_gamma_bound(treated, control, gamma=1.0)
        assert p_hi < 0.05

    def test_high_gamma_not_significant(self):
        """With high Γ, p_upper should increase (less certain)."""
        rng = np.random.default_rng(42)
        treated = 1.0 + rng.standard_normal(30)
        control = 0.0 + rng.standard_normal(30)
        _, p_hi_1 = rosenbaum_gamma_bound(treated, control, gamma=1.0)
        _, p_hi_5 = rosenbaum_gamma_bound(treated, control, gamma=5.0)
        # p_upper at high gamma should be >= p_upper at low gamma
        assert p_hi_5 >= p_hi_1 - 0.01

    def test_gamma_one_baseline(self):
        """Γ=1 should give standard signed-rank test bounds."""
        rng = np.random.default_rng(42)
        treated = 2.0 + rng.standard_normal(20)
        control = rng.standard_normal(20)
        p_lo, p_hi = rosenbaum_gamma_bound(treated, control, gamma=1.0)
        assert p_lo <= p_hi

    def test_increasing_gamma(self):
        """p_upper should increase with Γ."""
        rng = np.random.default_rng(42)
        treated = 2.0 + rng.standard_normal(30)
        control = rng.standard_normal(30)
        _, p1 = rosenbaum_gamma_bound(treated, control, gamma=1.0)
        _, p2 = rosenbaum_gamma_bound(treated, control, gamma=2.0)
        _, p3 = rosenbaum_gamma_bound(treated, control, gamma=3.0)
        assert p1 <= p2 + 0.01  # allow small tolerance
        assert p2 <= p3 + 0.01

    def test_empty_arrays(self):
        p_lo, p_hi = rosenbaum_gamma_bound(np.array([]), np.array([]))
        assert p_lo == 1.0

    def test_critical_gamma(self):
        rng = np.random.default_rng(42)
        treated = 3.0 + rng.standard_normal(40)
        control = rng.standard_normal(40)
        critical = rosenbaum_critical_gamma(treated, control, alpha=0.05)
        assert critical >= 1.0

    def test_critical_gamma_strong_effect(self):
        """Strong effect should have higher critical Γ."""
        rng = np.random.default_rng(42)
        treated_weak = 0.5 + rng.standard_normal(40)
        treated_strong = 3.0 + rng.standard_normal(40)
        control = rng.standard_normal(40)
        crit_weak = rosenbaum_critical_gamma(treated_weak, control)
        crit_strong = rosenbaum_critical_gamma(treated_strong, control)
        assert crit_strong >= crit_weak


class TestATESensitivityContour:
    def test_shape(self):
        grid_y = np.linspace(0, 0.3, 5)
        grid_t = np.linspace(0, 0.3, 5)
        contour = ate_sensitivity_contour(2.0, 0.5, grid_y, grid_t, dof=100)
        assert contour.shape == (5, 5)

    def test_zero_confounding_unchanged(self):
        grid_y = np.array([0.0])
        grid_t = np.array([0.0])
        contour = ate_sensitivity_contour(2.0, 0.5, grid_y, grid_t, dof=100)
        assert contour[0, 0] == pytest.approx(2.0, abs=1e-6)

    def test_decreasing_with_confounding(self):
        grid_y = np.array([0.0, 0.1, 0.2])
        grid_t = np.array([0.1])
        contour = ate_sensitivity_contour(2.0, 0.5, grid_y, grid_t, dof=100)
        assert contour[0, 0] >= contour[1, 0] >= contour[2, 0]


class TestLiteratureValues:
    """Compare against known values from published examples."""

    def test_vanderweele_evalue_example(self):
        """VanderWeele & Ding (2017) Table 1: RR=3.9 → E-value ≈ 7.26."""
        ev = e_value_rr(3.9)
        expected = 3.9 + math.sqrt(3.9 * 2.9)
        assert ev == pytest.approx(expected, abs=0.01)

    def test_evalue_monotonicity(self):
        """E-value should be monotonically increasing in |effect|."""
        rrs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        evalues = [e_value_rr(rr) for rr in rrs]
        for i in range(len(evalues) - 1):
            assert evalues[i] <= evalues[i + 1]

    def test_evalue_symmetry(self):
        """E-value for RR and 1/RR should be the same."""
        for rr in [1.5, 2.0, 3.0, 5.0]:
            assert e_value_rr(rr) == pytest.approx(e_value_rr(1 / rr), abs=1e-10)

    def test_rosenbaum_known_example(self):
        """With very strong effect, critical Γ should be large."""
        rng = np.random.default_rng(0)
        n = 100
        treated = 10 + rng.standard_normal(n)
        control = rng.standard_normal(n)
        crit = rosenbaum_critical_gamma(treated, control, alpha=0.05, gamma_max=20)
        assert crit > 3.0  # strong effect → robust to bias
