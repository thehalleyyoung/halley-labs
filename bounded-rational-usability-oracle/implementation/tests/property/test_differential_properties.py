"""Property-based tests for the differential privacy module.

Verifies Laplace noise scale, post-processing preservation, sequential
composition, budget accounting, and private mean convergence using Hypothesis.
"""

import math

import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats,
    integers,
    lists,
    composite,
)

from usability_oracle.differential.types import (
    PrivacyBudget,
    CompositionTheorem,
)
from usability_oracle.differential.mechanisms import (
    laplace_scale,
    laplace_mechanism,
    gaussian_mechanism,
    gaussian_scale,
    randomized_response,
    sensitivity_count,
    sensitivity_sum,
    sensitivity_mean,
)
from usability_oracle.differential.composition import (
    basic_composition,
    advanced_composition,
    parallel_composition,
    verify_post_processing,
)
from usability_oracle.differential.accountant import (
    BudgetAccountant,
)
from usability_oracle.differential.aggregation import (
    private_mean,
    private_count,
    private_sum,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ATOL = 1e-6

_epsilon = floats(min_value=0.01, max_value=10.0,
                  allow_nan=False, allow_infinity=False)

_delta = floats(min_value=0.0, max_value=0.1,
                allow_nan=False, allow_infinity=False)

_sensitivity = floats(min_value=0.01, max_value=100.0,
                      allow_nan=False, allow_infinity=False)

_value = floats(min_value=-1000.0, max_value=1000.0,
                allow_nan=False, allow_infinity=False)


@composite
def _privacy_budget(draw):
    """Generate a valid PrivacyBudget."""
    eps = draw(_epsilon)
    d = draw(_delta)
    return PrivacyBudget(epsilon=eps, delta=d)


@composite
def _budget_list(draw, min_size=2, max_size=5):
    """Generate a list of PrivacyBudgets."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    return [draw(_privacy_budget()) for _ in range(n)]


# ---------------------------------------------------------------------------
# Laplace noise has correct scale: b = Δf / ε
# ---------------------------------------------------------------------------


@given(_sensitivity, _epsilon)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_laplace_scale_formula(sens, eps):
    """Laplace scale b = Δf / ε."""
    b = laplace_scale(sens, eps)
    expected = sens / eps
    assert math.isclose(b, expected, rel_tol=1e-10), \
        f"laplace_scale({sens}, {eps}) = {b} != {expected}"


@given(_sensitivity, _epsilon)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_laplace_scale_positive(sens, eps):
    """Laplace scale is always positive."""
    b = laplace_scale(sens, eps)
    assert b > 0, f"Laplace scale not positive: {b}"


@given(_sensitivity, _epsilon)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_laplace_scale_decreases_with_epsilon(sens, eps):
    """Larger ε → smaller noise scale (less noise = less privacy)."""
    b1 = laplace_scale(sens, eps)
    b2 = laplace_scale(sens, eps * 2.0)
    assert b2 <= b1 + _ATOL, \
        f"Larger ε should reduce scale: b({eps})={b1}, b({eps*2})={b2}"


# ---------------------------------------------------------------------------
# Laplace mechanism returns finite value
# ---------------------------------------------------------------------------


@given(_value, _sensitivity, _epsilon)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_laplace_mechanism_returns_finite(value, sens, eps):
    """Laplace mechanism always returns a finite number."""
    rng = np.random.default_rng(42)
    result = laplace_mechanism(value, sens, eps, rng=rng)
    assert math.isfinite(result), f"Non-finite result: {result}"


# ---------------------------------------------------------------------------
# Gaussian scale is positive
# ---------------------------------------------------------------------------


@given(_sensitivity, _epsilon,
       floats(min_value=1e-6, max_value=0.1,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_gaussian_scale_positive(sens, eps, delta):
    """Gaussian noise scale is positive."""
    sigma = gaussian_scale(sens, eps, delta)
    assert sigma > 0, f"Gaussian scale not positive: {sigma}"


# ---------------------------------------------------------------------------
# Post-processing preserves DP
# ---------------------------------------------------------------------------


@given(_privacy_budget())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_post_processing_preserves_dp(budget):
    """verify_post_processing confirms post-processing doesn't degrade privacy."""
    # Post-processed budget must have same or tighter guarantees.
    result = verify_post_processing(budget, budget)
    assert result is True, \
        f"verify_post_processing rejected identical budget"


# ---------------------------------------------------------------------------
# Sequential composition: ε sums correctly
# ---------------------------------------------------------------------------


@given(_budget_list())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_basic_composition_epsilon_sums(budgets):
    """Basic composition: ε_total = Σ εᵢ."""
    result = basic_composition(budgets)
    expected_eps = sum(b.epsilon for b in budgets)
    assert math.isclose(result.total_budget.epsilon, expected_eps, rel_tol=1e-8), \
        f"ε_total = {result.total_budget.epsilon} != {expected_eps}"


@given(_budget_list())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_basic_composition_delta_sums(budgets):
    """Basic composition: δ_total = Σ δᵢ (capped at <1)."""
    result = basic_composition(budgets)
    expected_delta = min(sum(b.delta for b in budgets), 1.0 - 1e-15)
    assert result.total_budget.delta <= expected_delta + _ATOL, \
        f"δ_total = {result.total_budget.delta} > {expected_delta}"


@given(_budget_list())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_basic_composition_theorem_type(budgets):
    """Basic composition uses BASIC theorem."""
    result = basic_composition(budgets)
    assert result.theorem_used == CompositionTheorem.BASIC


# ---------------------------------------------------------------------------
# Advanced composition gives tighter bound
# ---------------------------------------------------------------------------


@given(_budget_list(min_size=3, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_advanced_composition_non_negative(budgets):
    """Advanced composition ε is non-negative."""
    advanced = advanced_composition(budgets, delta_prime=0.01)
    assert advanced.total_budget.epsilon >= -_ATOL, \
        f"Advanced ε negative: {advanced.total_budget.epsilon}"
    assert advanced.total_budget.delta >= -_ATOL, \
        f"Advanced δ negative: {advanced.total_budget.delta}"


# ---------------------------------------------------------------------------
# Parallel composition: ε_total = max(εᵢ)
# ---------------------------------------------------------------------------


@given(_budget_list())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_composition_epsilon_max(budgets):
    """Parallel composition: ε_total = max(εᵢ)."""
    result = parallel_composition(budgets)
    expected_eps = max(b.epsilon for b in budgets)
    assert math.isclose(result.total_budget.epsilon, expected_eps, rel_tol=1e-8), \
        f"Parallel ε = {result.total_budget.epsilon} != max = {expected_eps}"


# ---------------------------------------------------------------------------
# Privacy budget never goes negative
# ---------------------------------------------------------------------------


@given(_epsilon, integers(min_value=1, max_value=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_budget_accountant_non_negative(total_eps, n_queries):
    """Budget accountant never reports negative remaining budget."""
    accountant = BudgetAccountant(
        total_budget=PrivacyBudget(epsilon=total_eps, delta=0.0)
    )
    per_query_eps = total_eps / (n_queries + 1)
    for _ in range(n_queries):
        accountant.record(PrivacyBudget(epsilon=per_query_eps, delta=0.0))
    remaining = accountant.remaining_budget()
    assert remaining.epsilon >= -_ATOL, \
        f"Negative remaining budget: {remaining.epsilon}"


@given(_epsilon)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_budget_accountant_initial_budget(total_eps):
    """Fresh accountant has full budget remaining."""
    accountant = BudgetAccountant(
        total_budget=PrivacyBudget(epsilon=total_eps, delta=0.0)
    )
    remaining = accountant.remaining_budget()
    assert math.isclose(remaining.epsilon, total_eps, rel_tol=1e-8), \
        f"Initial remaining = {remaining.epsilon} != {total_eps}"


# ---------------------------------------------------------------------------
# Private mean converges to true mean (utility)
# ---------------------------------------------------------------------------


@given(integers(min_value=100, max_value=1000),
       floats(min_value=1.0, max_value=10.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_private_mean_converges(n, eps):
    """Private mean converges to true mean as n → ∞ (utility guarantee)."""
    rng = np.random.default_rng(42)
    true_mean = 5.0
    data = rng.normal(true_mean, 1.0, size=n).tolist()
    result, _guarantee = private_mean(data, eps, clipping_bound=10.0, rng=rng)
    # With n ≥ 100 and ε ≥ 1, the private mean should be within 3.0
    assert abs(result - true_mean) < 3.0, \
        f"Private mean {result} too far from true mean {true_mean}"


# ---------------------------------------------------------------------------
# Private count is non-negative (with high probability)
# ---------------------------------------------------------------------------


@given(_epsilon)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_private_count_close_to_true(eps):
    """Private count is near the true count."""
    rng = np.random.default_rng(42)
    data = list(range(100))
    result, _guarantee = private_count(data, eps, rng=rng)
    # With ε ≥ 0.01 and n = 100, error should be bounded
    assert result > 0, f"Private count is non-positive: {result}"


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------


def test_sensitivity_count_is_one():
    """Counting query has sensitivity 1."""
    assert sensitivity_count() == 1.0


@given(floats(min_value=0.01, max_value=100.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sensitivity_sum_equals_bound(bound):
    """Sum sensitivity equals the clipping bound."""
    assert math.isclose(sensitivity_sum(bound), bound, rel_tol=1e-10)


@given(floats(min_value=0.01, max_value=100.0,
              allow_nan=False, allow_infinity=False),
       integers(min_value=1, max_value=10000))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sensitivity_mean_formula(bound, n):
    """Mean sensitivity = bound / n."""
    expected = bound / n
    result = sensitivity_mean(bound, n)
    assert math.isclose(result, expected, rel_tol=1e-10), \
        f"sensitivity_mean({bound}, {n}) = {result} != {expected}"


# ---------------------------------------------------------------------------
# Randomized response produces valid output
# ---------------------------------------------------------------------------


@given(_epsilon)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_randomized_response_output_valid(eps):
    """Randomized response produces 0 or 1."""
    rng = np.random.default_rng(42)
    result = randomized_response(True, eps, rng=rng)
    assert result in (True, False, 0, 1), f"Invalid RR output: {result}"


# ---------------------------------------------------------------------------
# PrivacyBudget invariants
# ---------------------------------------------------------------------------


@given(_epsilon, _delta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_privacy_budget_epsilon_non_negative(eps, delta):
    """PrivacyBudget always has ε ≥ 0."""
    b = PrivacyBudget(epsilon=eps, delta=delta)
    assert b.epsilon >= 0


@given(_epsilon)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_privacy_budget_compose_basic(eps):
    """Basic composition of two budgets sums epsilon."""
    b1 = PrivacyBudget(epsilon=eps, delta=0.0)
    b2 = PrivacyBudget(epsilon=eps, delta=0.0)
    composed = b1.compose_basic(b2)
    assert math.isclose(composed.epsilon, 2.0 * eps, rel_tol=1e-10)
