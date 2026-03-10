"""Property-based tests for the variational free-energy module.

Verifies mathematical invariants of KL divergence, Jensen-Shannon divergence,
softmax policy, free energy, capacity estimation, and the Blahut-Arimoto
algorithm using the Hypothesis library.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats,
    integers,
    lists,
    composite,
)

from usability_oracle.variational import (
    compute_kl_divergence,
    symmetric_kl,
    compute_softmax_policy,
    compute_free_energy,
)
from usability_oracle.variational.kl_divergence import (
    compute_kl_gaussian,
    renyi_divergence,
    compute_mutual_information,
)
from usability_oracle.variational.capacity import (
    estimate_fitts_capacity,
    estimate_hick_capacity,
    estimate_memory_capacity,
    blahut_arimoto,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_pos_float = floats(min_value=0.01, max_value=100.0,
                    allow_nan=False, allow_infinity=False)

_beta = floats(min_value=0.1, max_value=50.0,
               allow_nan=False, allow_infinity=False)

_large_beta = floats(min_value=50.0, max_value=500.0,
                     allow_nan=False, allow_infinity=False)


@composite
def _probability_distribution(draw, min_size=2, max_size=10):
    """Generate a valid probability distribution that sums to 1."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    raw = draw(lists(
        floats(min_value=0.01, max_value=10.0,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    total = sum(raw)
    assume(total > 0)
    return np.array([x / total for x in raw])


@composite
def _distribution_pair(draw, min_size=2, max_size=10):
    """Generate two valid probability distributions of the same length."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    raw_p = draw(lists(
        floats(min_value=0.01, max_value=10.0,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    raw_q = draw(lists(
        floats(min_value=0.01, max_value=10.0,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    total_p = sum(raw_p)
    total_q = sum(raw_q)
    assume(total_p > 0 and total_q > 0)
    p = np.array([x / total_p for x in raw_p])
    q = np.array([x / total_q for x in raw_q])
    return p, q


@composite
def _q_values(draw, min_size=2, max_size=10):
    """Generate Q-values (state-action values) for softmax policy."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    vals = draw(lists(
        floats(min_value=-10.0, max_value=10.0,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    return np.array(vals)


@composite
def _channel_matrix(draw, n_in=3, n_out=3):
    """Generate a valid channel matrix (rows sum to 1)."""
    rows = []
    for _ in range(n_in):
        raw = draw(lists(
            floats(min_value=0.01, max_value=10.0,
                   allow_nan=False, allow_infinity=False),
            min_size=n_out, max_size=n_out,
        ))
        total = sum(raw)
        assume(total > 0)
        rows.append([x / total for x in raw])
    return np.array(rows)


_ATOL = 1e-6

# ---------------------------------------------------------------------------
# KL divergence is non-negative
# ---------------------------------------------------------------------------


@given(_distribution_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_kl_divergence_non_negative(pair):
    """KL(p || q) >= 0 for any two valid distributions (Gibbs' inequality)."""
    p, q = pair
    kl = compute_kl_divergence(p, q)
    assert kl >= -_ATOL, f"KL divergence is negative: {kl}"


@given(_probability_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_kl_divergence_zero_for_identical(p):
    """KL(p || p) = 0 for any distribution p."""
    kl = compute_kl_divergence(p, p)
    assert abs(kl) < _ATOL, f"KL(p||p) should be 0, got {kl}"


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence is bounded by log(2)
# ---------------------------------------------------------------------------


@given(_distribution_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_jensen_shannon_bounded(pair):
    """JS(p, q) is bounded in [0, log(2)] ≈ [0, 0.693]."""
    p, q = pair
    js = symmetric_kl(p, q)
    assert js >= -_ATOL, f"JS divergence is negative: {js}"
    assert js <= math.log(2) + _ATOL, f"JS divergence exceeds log(2): {js}"


@given(_probability_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_jensen_shannon_zero_for_identical(p):
    """JS(p, p) = 0."""
    js = symmetric_kl(p, p)
    assert abs(js) < _ATOL, f"JS(p, p) should be 0, got {js}"


@given(_distribution_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_jensen_shannon_symmetric(pair):
    """JS(p, q) = JS(q, p) — symmetric by definition."""
    p, q = pair
    js_pq = symmetric_kl(p, q)
    js_qp = symmetric_kl(q, p)
    assert math.isclose(js_pq, js_qp, abs_tol=_ATOL), \
        f"JS not symmetric: {js_pq} vs {js_qp}"


# ---------------------------------------------------------------------------
# Softmax policy sums to 1
# ---------------------------------------------------------------------------


@given(_q_values(), _beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_softmax_policy_sums_to_one(q_vals, beta):
    """Softmax policy π(a) = exp(β·Q(a)) / Z must sum to 1."""
    policy = compute_softmax_policy(q_vals, beta)
    assert math.isclose(policy.sum(), 1.0, abs_tol=1e-6), \
        f"Softmax policy sums to {policy.sum()}, expected 1.0"


@given(_q_values(), _beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_softmax_policy_non_negative(q_vals, beta):
    """All softmax probabilities must be non-negative."""
    policy = compute_softmax_policy(q_vals, beta)
    assert np.all(policy >= -1e-12), f"Negative probabilities: {policy}"


@given(_q_values(), _beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_softmax_policy_highest_q_gets_highest_prob(q_vals, beta):
    """Action with highest Q-value gets highest probability."""
    assume(len(np.unique(q_vals)) == len(q_vals))
    # Require a minimum gap so numerical precision doesn't break argmax
    sorted_q = np.sort(q_vals)
    assume(sorted_q[-1] - sorted_q[-2] > 1e-6)
    policy = compute_softmax_policy(q_vals, beta)
    assert np.argmax(policy) == np.argmax(q_vals)


# ---------------------------------------------------------------------------
# Softmax becomes deterministic as β → ∞
# ---------------------------------------------------------------------------


@given(_q_values())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_softmax_deterministic_high_beta(q_vals):
    """As β → ∞, softmax concentrates on the argmax action."""
    assume(len(np.unique(q_vals)) == len(q_vals))  # all unique
    sorted_q = np.sort(q_vals)
    assume(sorted_q[-1] - sorted_q[-2] > 0.5)  # need clear gap for β=100
    policy = compute_softmax_policy(q_vals, beta=100.0)
    max_idx = np.argmax(q_vals)
    assert policy[max_idx] > 0.99, \
        f"At β=100, max action should have >0.99 prob, got {policy[max_idx]}"


@given(_q_values())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_softmax_uniform_low_beta(q_vals):
    """As β → 0, softmax approaches uniform distribution."""
    policy = compute_softmax_policy(q_vals, beta=0.001)
    n = len(q_vals)
    expected = 1.0 / n
    for p in policy:
        assert math.isclose(p, expected, abs_tol=0.05), \
            f"At β≈0, policy should be ≈uniform ({expected}), got {p}"


# ---------------------------------------------------------------------------
# Free energy is finite for finite rewards and β > 0
# ---------------------------------------------------------------------------


@given(_distribution_pair(), _beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_free_energy_is_finite(pair, beta):
    """Free energy F = E[C] - (1/β)H[π] is finite for finite inputs."""
    policy, prior = pair
    n = len(policy)
    rewards = np.random.default_rng(42).uniform(-5, 5, size=n)
    fe = compute_free_energy(policy, rewards, prior, beta)
    assert math.isfinite(fe), f"Free energy is not finite: {fe}"


@given(_probability_distribution(), _beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_free_energy_uniform_prior_relationship(policy, beta):
    """Free energy with uniform prior includes entropy term."""
    n = len(policy)
    rewards = np.zeros(n)
    prior = np.ones(n) / n
    fe = compute_free_energy(policy, rewards, prior, beta)
    assert math.isfinite(fe)


# ---------------------------------------------------------------------------
# KL divergence for Gaussians
# ---------------------------------------------------------------------------


@given(_pos_float, _pos_float, _pos_float, _pos_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_kl_gaussian_non_negative(mu1, sigma1, mu2, sigma2):
    """KL divergence between Gaussians is non-negative."""
    kl = compute_kl_gaussian(mu1, sigma1, mu2, sigma2)
    assert kl >= -_ATOL, f"Gaussian KL is negative: {kl}"


@given(_pos_float, _pos_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_kl_gaussian_zero_for_identical(mu, sigma):
    """KL(N(μ,σ) || N(μ,σ)) = 0."""
    kl = compute_kl_gaussian(mu, sigma, mu, sigma)
    assert abs(kl) < _ATOL, f"KL for identical Gaussians should be 0, got {kl}"


# ---------------------------------------------------------------------------
# Rényi divergence
# ---------------------------------------------------------------------------


@given(_distribution_pair(),
       floats(min_value=0.1, max_value=10.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_renyi_divergence_non_negative(pair, alpha):
    """Rényi divergence D_α(p || q) >= 0 for any α > 0, α ≠ 1."""
    p, q = pair
    assume(abs(alpha - 1.0) > 0.01)
    rd = renyi_divergence(p, q, alpha)
    assert rd >= -_ATOL, f"Rényi divergence is negative: {rd}"


# ---------------------------------------------------------------------------
# Capacity estimation
# ---------------------------------------------------------------------------


@given(floats(min_value=1.0, max_value=500.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.5, max_value=100.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_capacity_non_negative(distance, width):
    """Fitts' capacity (throughput) is non-negative."""
    cap = estimate_fitts_capacity(distance, width)
    assert cap >= -_ATOL, f"Fitts capacity is negative: {cap}"


@given(integers(min_value=2, max_value=20))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_hick_capacity_non_negative(n_alternatives):
    """Hick's law capacity is non-negative."""
    cap = estimate_hick_capacity(n_alternatives)
    assert cap >= -_ATOL, f"Hick capacity is negative: {cap}"


@given(integers(min_value=1, max_value=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_memory_capacity_non_negative(n_chunks):
    """Memory capacity is non-negative."""
    cap = estimate_memory_capacity(n_chunks)
    assert cap >= -_ATOL, f"Memory capacity is negative: {cap}"


@given(integers(min_value=2, max_value=10), integers(min_value=3, max_value=15))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_hick_capacity_increases_with_alternatives(n_small, n_extra):
    """More alternatives should yield higher Hick capacity (more bits needed)."""
    n_large = n_small + n_extra
    cap_small = estimate_hick_capacity(n_small)
    cap_large = estimate_hick_capacity(n_large)
    assert cap_large >= cap_small - _ATOL, \
        f"Hick capacity should increase: {cap_small} vs {cap_large}"


# ---------------------------------------------------------------------------
# Blahut-Arimoto channel capacity
# ---------------------------------------------------------------------------


@given(_channel_matrix())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_blahut_arimoto_capacity_non_negative(channel):
    """Channel capacity from Blahut-Arimoto is non-negative."""
    cap, input_dist = blahut_arimoto(channel, tolerance=1e-6, max_iter=200)
    assert cap >= -_ATOL, f"Blahut-Arimoto capacity is negative: {cap}"


@given(_channel_matrix())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_blahut_arimoto_input_dist_valid(channel):
    """Optimal input distribution from Blahut-Arimoto is a valid distribution."""
    cap, input_dist = blahut_arimoto(channel, tolerance=1e-6, max_iter=200)
    assert math.isclose(np.sum(input_dist), 1.0, abs_tol=1e-4), \
        f"Input distribution sums to {np.sum(input_dist)}"
    assert np.all(input_dist >= -1e-10), "Input distribution has negative values"


# ---------------------------------------------------------------------------
# Mutual information non-negativity
# ---------------------------------------------------------------------------


def test_mutual_information_non_negative_uniform():
    """Mutual information I(X;Y) >= 0 for a uniform joint distribution."""
    # Create a simple joint distribution
    joint = np.array([[0.25, 0.0], [0.0, 0.25], [0.25, 0.0], [0.0, 0.25]])
    joint = joint / joint.sum()  # normalise
    mi = compute_mutual_information(joint)
    assert mi >= -_ATOL, f"Mutual information is negative: {mi}"


def test_mutual_information_zero_independent():
    """I(X;Y) ≈ 0 when X and Y are independent (product distribution)."""
    px = np.array([0.5, 0.5])
    py = np.array([0.5, 0.5])
    joint = np.outer(px, py)
    mi = compute_mutual_information(joint)
    assert abs(mi) < 1e-6, f"MI for independent should be ≈0, got {mi}"


# ---------------------------------------------------------------------------
# Softmax monotonicity in β
# ---------------------------------------------------------------------------


@given(_q_values())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_softmax_entropy_decreases_with_beta(q_vals):
    """Entropy of softmax policy decreases as β increases."""
    assume(len(np.unique(q_vals)) == len(q_vals))
    sorted_q = np.sort(q_vals)
    assume(sorted_q[-1] - sorted_q[-2] > 1e-6)
    beta_low = 0.5
    beta_high = 10.0
    policy_low = compute_softmax_policy(q_vals, beta_low)
    policy_high = compute_softmax_policy(q_vals, beta_high)
    # Compute Shannon entropy
    h_low = -np.sum(policy_low * np.log(policy_low + 1e-30))
    h_high = -np.sum(policy_high * np.log(policy_high + 1e-30))
    assert h_low >= h_high - _ATOL, \
        f"Lower β should have higher entropy: H(β={beta_low})={h_low}, H(β={beta_high})={h_high}"
