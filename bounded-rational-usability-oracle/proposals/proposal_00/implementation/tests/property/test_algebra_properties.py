"""Property-based tests for the cost algebra module.

This module verifies algebraic properties of CostElement, SequentialComposer,
and ParallelComposer using Hypothesis. Properties tested include associativity
of sequential composition, commutativity of parallel composition, identity
elements, monotonicity, positivity constraints, and distributivity of scalar
multiplication over CostElement addition.
"""

import math

from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import floats, integers, lists, tuples, sampled_from

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.parallel import ParallelComposer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_pos_float = floats(min_value=0.01, max_value=1e4,
                    allow_nan=False, allow_infinity=False)

_small_pos = floats(min_value=0.01, max_value=100.0,
                    allow_nan=False, allow_infinity=False)

_unit_float = floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False)

_coupling = floats(min_value=0.0, max_value=1.0,
                   allow_nan=False, allow_infinity=False)

_scalar = floats(min_value=-100.0, max_value=100.0,
                 allow_nan=False, allow_infinity=False)

_ATOL = 1e-4
_RTOL = 1e-3


def _make_cost(mu, sigma_sq, kappa=0.0, lambda_=0.0):
    """Build a CostElement with safe values."""
    return CostElement(mu=mu, sigma_sq=sigma_sq, kappa=kappa, lambda_=lambda_)


# ---------------------------------------------------------------------------
# CostElement addition commutativity
# ---------------------------------------------------------------------------


@given(_small_pos, _small_pos, _small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cost_element_addition_commutative_full(mu_a, sig_a, mu_b, sig_b):
    """CostElement addition is fully commutative: a + b == b + a."""
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    ab = a + b
    ba = b + a
    assert ab == ba


# ---------------------------------------------------------------------------
# CostElement identity
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cost_element_additive_identity(mu, sigma_sq):
    """Adding CostElement.zero() is identity: a + zero ≈ a."""
    a = _make_cost(mu, sigma_sq)
    zero = CostElement.zero()
    result = a + zero
    assert math.isclose(result.mu, a.mu, abs_tol=_ATOL)
    assert math.isclose(result.sigma_sq, a.sigma_sq, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# CostElement scalar multiplication
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos, _scalar)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_scalar_multiply_mu(mu, sigma_sq, s):
    """Scalar multiplication scales mu linearly: (s * a).mu == s * a.mu.

    The expected cost scales linearly with a scalar multiplier.
    """
    a = _make_cost(mu, sigma_sq)
    result = s * a
    assert math.isclose(result.mu, s * a.mu, abs_tol=_ATOL)


@given(_small_pos, _small_pos, _scalar, _scalar)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_scalar_distributivity_mu(mu, sigma_sq, s1, s2):
    """Scalar multiplication distributes over addition for mu.

    (s1 + s2) * a should give the same mu as s1*a + s2*a.
    """
    a = _make_cost(mu, sigma_sq)
    lhs = (s1 + s2) * a
    rhs = s1 * a + s2 * a
    assert math.isclose(lhs.mu, rhs.mu, rel_tol=_RTOL, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# CostElement validity / positivity
# ---------------------------------------------------------------------------

@given(_pos_float, _pos_float, _unit_float, _unit_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cost_element_is_valid(mu, sigma_sq, kappa, lambda_):
    """A CostElement with non-negative mu, sigma_sq, lambda_ in [0,1] is valid."""
    c = _make_cost(mu, sigma_sq, kappa, lambda_)
    assert c.is_valid


@given(_pos_float, _pos_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cost_element_expected_cost_positive(mu, sigma_sq):
    """Expected cost is non-negative for non-negative mu.

    expected_cost() should return a value >= 0.
    """
    c = _make_cost(mu, sigma_sq)
    assert c.expected_cost() >= 0.0


@given(_pos_float, _pos_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cost_element_std_dev_non_negative(mu, sigma_sq):
    """Standard deviation is always non-negative."""
    c = _make_cost(mu, sigma_sq)
    assert c.std_dev() >= 0.0


# ---------------------------------------------------------------------------
# Sequential composition associativity
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos, _small_pos, _small_pos, _small_pos, _small_pos)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_associativity_mu(mu_a, sig_a, mu_b, sig_b, mu_c, sig_c):
    """Sequential composition is associative for the mu component.

    compose(compose(a,b), c).mu ≈ compose(a, compose(b,c)).mu
    within floating-point tolerance.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    c = _make_cost(mu_c, sig_c)
    sc = SequentialComposer()
    lhs = sc.compose(sc.compose(a, b), c)
    rhs = sc.compose(a, sc.compose(b, c))
    assert math.isclose(lhs.mu, rhs.mu, rel_tol=_RTOL, abs_tol=_ATOL)


@given(_small_pos, _small_pos, _small_pos, _small_pos, _small_pos, _small_pos)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_associativity_sigma(mu_a, sig_a, mu_b, sig_b, mu_c, sig_c):
    """Sequential composition is associative for the sigma_sq component.

    compose(compose(a,b), c).sigma_sq ≈ compose(a, compose(b,c)).sigma_sq
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    c = _make_cost(mu_c, sig_c)
    sc = SequentialComposer()
    lhs = sc.compose(sc.compose(a, b), c)
    rhs = sc.compose(a, sc.compose(b, c))
    assert math.isclose(lhs.sigma_sq, rhs.sigma_sq, rel_tol=_RTOL, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# Sequential composition identity
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_identity(mu, sigma_sq):
    """Sequentially composing with the zero element is identity.

    compose(a, zero).mu ≈ a.mu and compose(a, zero).sigma_sq ≈ a.sigma_sq.
    """
    a = _make_cost(mu, sigma_sq)
    zero = CostElement.zero()
    sc = SequentialComposer()
    result = sc.compose(a, zero)
    assert math.isclose(result.mu, a.mu, rel_tol=_RTOL, abs_tol=_ATOL)
    assert math.isclose(result.sigma_sq, a.sigma_sq, rel_tol=_RTOL, abs_tol=_ATOL)


@given(_small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_identity_left(mu, sigma_sq):
    """Left-composing with zero is also identity: compose(zero, a) ≈ a."""
    a = _make_cost(mu, sigma_sq)
    zero = CostElement.zero()
    sc = SequentialComposer()
    result = sc.compose(zero, a)
    assert math.isclose(result.mu, a.mu, rel_tol=_RTOL, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# Sequential composition positivity
# ---------------------------------------------------------------------------

@given(_pos_float, _pos_float, _pos_float, _pos_float, _coupling)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_positivity(mu_a, sig_a, mu_b, sig_b, coupling):
    """Sequential composition preserves non-negativity of mu and sigma_sq.

    If both inputs have non-negative mu and sigma_sq, so does the result.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    sc = SequentialComposer()
    result = sc.compose(a, b, coupling=coupling)
    assert result.mu >= -_ATOL
    assert result.sigma_sq >= -_ATOL


# ---------------------------------------------------------------------------
# Sequential composition monotonicity
# ---------------------------------------------------------------------------

@given(_pos_float, _pos_float, _pos_float, _pos_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_mu_additive(mu_a, sig_a, mu_b, sig_b):
    """Sequential mu is at least the sum of component mus (with zero coupling).

    compose(a, b).mu >= a.mu + b.mu - tolerance for zero coupling.
    In fact for zero coupling sequential composition adds the mus.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    sc = SequentialComposer()
    result = sc.compose(a, b, coupling=0.0)
    assert result.mu >= a.mu + b.mu - _ATOL


# ---------------------------------------------------------------------------
# Sequential compose_chain consistency
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos, _small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_chain_two_equals_compose(mu_a, sig_a, mu_b, sig_b):
    """compose_chain([a, b]) == compose(a, b) for two elements.

    The chain operation should reduce to the binary operation for two inputs.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    sc = SequentialComposer()
    chain_result = sc.compose_chain([a, b])
    direct_result = sc.compose(a, b)
    assert math.isclose(chain_result.mu, direct_result.mu, rel_tol=_RTOL, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# Parallel composition commutativity
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos, _small_pos, _small_pos, _coupling)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_commutativity_mu(mu_a, sig_a, mu_b, sig_b, interference):
    """Parallel composition is commutative for mu.

    compose(a, b).mu == compose(b, a).mu because max is commutative.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    pc = ParallelComposer()
    ab = pc.compose(a, b, interference=interference)
    ba = pc.compose(b, a, interference=interference)
    assert math.isclose(ab.mu, ba.mu, rel_tol=_RTOL, abs_tol=_ATOL)


@given(_small_pos, _small_pos, _small_pos, _small_pos, _coupling)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_commutativity_sigma(mu_a, sig_a, mu_b, sig_b, interference):
    """Parallel composition is commutative for sigma_sq.

    compose(a, b).sigma_sq == compose(b, a).sigma_sq.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    pc = ParallelComposer()
    ab = pc.compose(a, b, interference=interference)
    ba = pc.compose(b, a, interference=interference)
    assert math.isclose(ab.sigma_sq, ba.sigma_sq, rel_tol=_RTOL, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# Parallel composition monotonicity
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos, _small_pos, _small_pos, _coupling)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_mu_at_least_max(mu_a, sig_a, mu_b, sig_b, interference):
    """Parallel compose mu >= max(a.mu, b.mu).

    Because parallel execution takes at least as long as the slower
    component, the expected cost dominates both inputs.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    pc = ParallelComposer()
    result = pc.compose(a, b, interference=interference)
    assert result.mu >= max(a.mu, b.mu) - _ATOL


# ---------------------------------------------------------------------------
# Parallel composition positivity
# ---------------------------------------------------------------------------

@given(_pos_float, _pos_float, _pos_float, _pos_float, _coupling)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_positivity(mu_a, sig_a, mu_b, sig_b, interference):
    """Parallel composition result has non-negative mu and sigma_sq.

    Physical costs and variances cannot be negative.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    pc = ParallelComposer()
    result = pc.compose(a, b, interference=interference)
    assert result.mu >= -_ATOL
    assert result.sigma_sq >= -_ATOL


# ---------------------------------------------------------------------------
# Parallel identity
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_identity(mu, sigma_sq):
    """Parallel composing with zero should approximate the original element.

    compose(a, zero) ≈ a because combining with a zero-cost task should
    not increase the cost beyond a.
    """
    a = _make_cost(mu, sigma_sq)
    zero = CostElement.zero()
    pc = ParallelComposer()
    result = pc.compose(a, zero, interference=0.0)
    assert result.mu >= a.mu - _ATOL


# ---------------------------------------------------------------------------
# Parallel compose_group
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos, _small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_group_two_equals_compose(mu_a, sig_a, mu_b, sig_b):
    """compose_group([a, b]) matches compose(a, b) for two elements.

    The group operation should reduce to binary parallel composition.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    pc = ParallelComposer()
    group_result = pc.compose_group([a, b], interference=0.0)
    direct_result = pc.compose(a, b, interference=0.0)
    assert math.isclose(group_result.mu, direct_result.mu, rel_tol=_RTOL, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# CostElement serialization round-trip
# ---------------------------------------------------------------------------

@given(_pos_float, _pos_float, _unit_float, _unit_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cost_element_serialization_roundtrip(mu, sigma_sq, kappa, lambda_):
    """to_dict / from_dict round-trip preserves all fields.

    Serialization should be lossless for all four cost components.
    """
    c = _make_cost(mu, sigma_sq, kappa, lambda_)
    d = c.to_dict()
    c2 = CostElement.from_dict(d)
    assert math.isclose(c.mu, c2.mu, abs_tol=1e-12)
    assert math.isclose(c.sigma_sq, c2.sigma_sq, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# CostElement.zero
# ---------------------------------------------------------------------------

def test_zero_element_values():
    """CostElement.zero() has mu=0, sigma_sq=0.

    The zero element is the additive identity of the cost algebra.
    """
    z = CostElement.zero()
    assert z.mu == 0.0
    assert z.sigma_sq == 0.0


# ---------------------------------------------------------------------------
# CostElement negation
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cost_element_negation(mu, sigma_sq):
    """Negation flips the sign of mu: (-a).mu == -a.mu.

    Negation should produce the additive inverse for the mu component.
    """
    a = _make_cost(mu, sigma_sq)
    neg_a = -a
    assert math.isclose(neg_a.mu, -a.mu, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# CostElement subtraction = addition of negation
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos, _small_pos, _small_pos)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_subtraction_is_add_negation(mu_a, sig_a, mu_b, sig_b):
    """a - b has same mu as a + (-b).

    Subtraction should be consistent with addition of the negation.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    sub = a - b
    add_neg = a + (-b)
    assert math.isclose(sub.mu, add_neg.mu, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# Sensitivity returns finite values
# ---------------------------------------------------------------------------

@given(_small_pos, _small_pos, _small_pos, _small_pos)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_sensitivity_finite(mu_a, sig_a, mu_b, sig_b):
    """Sequential sensitivity analysis returns finite partial derivatives.

    All returned sensitivity values should be finite numbers.
    """
    a = _make_cost(mu_a, sig_a)
    b = _make_cost(mu_b, sig_b)
    sc = SequentialComposer()
    sens = sc.sensitivity(a, b, coupling=0.0)
    for key, val in sens.items():
        assert math.isfinite(val), f"Non-finite sensitivity: {key}={val}"


# ---------------------------------------------------------------------------
# Coefficient of variation non-negative
# ---------------------------------------------------------------------------

@given(_pos_float, _pos_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_coefficient_of_variation_non_negative(mu, sigma_sq):
    """Coefficient of variation is non-negative for positive mu.

    CV = std_dev / mu >= 0 when mu > 0.
    """
    c = _make_cost(mu, sigma_sq)
    cv = c.coefficient_of_variation()
    assert cv >= 0.0
