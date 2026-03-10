"""Property-based tests for the information theory module.

Verifies fundamental information-theoretic inequalities using Hypothesis:
non-negativity of entropy, maximality of uniform distribution, symmetry of
mutual information, Gibbs' inequality (KL divergence ≥ 0), the data
processing inequality, the chain rule of entropy, and channel capacity bounds.
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

from usability_oracle.information_theory.entropy import (
    shannon_entropy,
    joint_entropy,
    conditional_entropy,
    conditional_entropy_yx,
    binary_entropy,
    cross_entropy,
)
from usability_oracle.information_theory.mutual_information import (
    kl_divergence,
    mutual_information,
    mutual_information_full,
    jensen_shannon_divergence,
    jensen_shannon_distance,
    total_variation_distance,
    hellinger_distance,
    total_correlation,
)
from usability_oracle.information_theory.channel_capacity import (
    blahut_arimoto,
    bsc_capacity,
    bec_capacity,
    binary_symmetric_channel,
    binary_erasure_channel,
    gaussian_channel_capacity,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ATOL = 1e-6
_RTOL = 1e-4

_prob = floats(min_value=0.001, max_value=0.999,
               allow_nan=False, allow_infinity=False)

_small_int = integers(min_value=2, max_value=20)

_crossover = floats(min_value=0.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False)


@composite
def _probability_distribution(draw, min_size=2, max_size=10):
    """Generate a valid probability distribution that sums to 1."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    raw = [draw(floats(min_value=0.01, max_value=10.0,
                       allow_nan=False, allow_infinity=False))
           for _ in range(n)]
    total = sum(raw)
    return [r / total for r in raw]


@composite
def _joint_distribution(draw, min_rows=2, max_rows=5,
                        min_cols=2, max_cols=5):
    """Generate a valid 2-D joint distribution."""
    nr = draw(integers(min_value=min_rows, max_value=max_rows))
    nc = draw(integers(min_value=min_cols, max_value=max_cols))
    raw = np.array([
        [draw(floats(min_value=0.01, max_value=10.0,
                     allow_nan=False, allow_infinity=False))
         for _ in range(nc)]
        for _ in range(nr)
    ])
    raw /= raw.sum()
    return raw


@composite
def _transition_matrix(draw, min_in=2, max_in=5,
                       min_out=2, max_out=5):
    """Generate a valid row-stochastic transition matrix."""
    nr = draw(integers(min_value=min_in, max_value=max_in))
    nc = draw(integers(min_value=min_out, max_value=max_out))
    mat = np.zeros((nr, nc))
    for i in range(nr):
        raw = [draw(floats(min_value=0.01, max_value=10.0,
                           allow_nan=False, allow_infinity=False))
               for _ in range(nc)]
        total = sum(raw)
        for j in range(nc):
            mat[i, j] = raw[j] / total
    return mat


# ---------------------------------------------------------------------------
# Entropy non-negativity
# ---------------------------------------------------------------------------


@given(_probability_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_entropy_non_negative(p):
    """Shannon entropy H(X) ≥ 0 for any valid distribution."""
    h = shannon_entropy(p)
    assert h >= -_ATOL, f"Entropy is negative: {h}"


# ---------------------------------------------------------------------------
# Entropy maximized by uniform distribution
# ---------------------------------------------------------------------------


@given(_small_int)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_entropy_maximized_by_uniform(n):
    """H(X) ≤ log₂(n) for any distribution over n outcomes."""
    uniform = [1.0 / n] * n
    h_max = shannon_entropy(uniform)
    assert math.isclose(h_max, math.log2(n), rel_tol=_RTOL), \
        f"Uniform entropy {h_max} != log2({n}) = {math.log2(n)}"


@given(_probability_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_entropy_at_most_log_n(p):
    """H(X) ≤ log₂(|X|) for any distribution p."""
    n = len(p)
    h = shannon_entropy(p)
    assert h <= math.log2(n) + _ATOL, \
        f"H(X) = {h} > log2({n}) = {math.log2(n)}"


# ---------------------------------------------------------------------------
# Binary entropy properties
# ---------------------------------------------------------------------------


@given(_crossover)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_binary_entropy_in_unit(p):
    """Binary entropy h(p) ∈ [0, 1] for p ∈ [0, 1]."""
    h = binary_entropy(p)
    assert -_ATOL <= h <= 1.0 + _ATOL, f"h({p}) = {h} out of range"


@given(_crossover)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_binary_entropy_symmetric(p):
    """h(p) = h(1-p): binary entropy is symmetric around 0.5."""
    h1 = binary_entropy(p)
    h2 = binary_entropy(1.0 - p)
    assert math.isclose(h1, h2, abs_tol=_ATOL), \
        f"h({p}) = {h1} != h({1-p}) = {h2}"


# ---------------------------------------------------------------------------
# Mutual information is symmetric: I(X;Y) = I(Y;X)
# ---------------------------------------------------------------------------


@given(_joint_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_mutual_information_symmetric(pxy):
    """I(X;Y) = I(Y;X): mutual information is symmetric."""
    mi_xy = mutual_information(pxy)
    mi_yx = mutual_information(pxy.T)
    assert math.isclose(mi_xy, mi_yx, abs_tol=_ATOL), \
        f"I(X;Y) = {mi_xy} != I(Y;X) = {mi_yx}"


# ---------------------------------------------------------------------------
# Mutual information is non-negative
# ---------------------------------------------------------------------------


@given(_joint_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_mutual_information_non_negative(pxy):
    """I(X;Y) ≥ 0."""
    mi = mutual_information(pxy)
    assert mi >= -_ATOL, f"MI is negative: {mi}"


# ---------------------------------------------------------------------------
# Mutual information ≤ min(H(X), H(Y))
# ---------------------------------------------------------------------------


@given(_joint_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_mutual_information_bounded(pxy):
    """I(X;Y) ≤ min(H(X), H(Y))."""
    result = mutual_information_full(pxy)
    mi = result.mutual_info_bits
    hx = result.entropy_x_bits
    hy = result.entropy_y_bits
    assert mi <= min(hx, hy) + _ATOL, \
        f"I(X;Y) = {mi} > min(H(X)={hx}, H(Y)={hy})"


# ---------------------------------------------------------------------------
# KL divergence is non-negative (Gibbs' inequality)
# ---------------------------------------------------------------------------


@composite
def _two_distributions_same_size(draw, min_size=2, max_size=10):
    """Generate two distributions of the same size."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    p_raw = [draw(floats(min_value=0.01, max_value=10.0,
                         allow_nan=False, allow_infinity=False))
             for _ in range(n)]
    q_raw = [draw(floats(min_value=0.01, max_value=10.0,
                         allow_nan=False, allow_infinity=False))
             for _ in range(n)]
    p_total = sum(p_raw)
    q_total = sum(q_raw)
    return [r / p_total for r in p_raw], [r / q_total for r in q_raw]


@given(_two_distributions_same_size())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_kl_divergence_non_negative(pq):
    """D_KL(p || q) ≥ 0 (Gibbs' inequality)."""
    p, q = pq
    d = kl_divergence(p, q)
    assert d >= -_ATOL, f"KL divergence is negative: {d}"


@given(_probability_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_kl_divergence_zero_for_identical(p):
    """D_KL(p || p) = 0."""
    d = kl_divergence(p, p)
    assert math.isclose(d, 0.0, abs_tol=_ATOL), \
        f"KL(p || p) = {d} != 0"


# ---------------------------------------------------------------------------
# Cross entropy H(p, q) ≥ H(p)
# ---------------------------------------------------------------------------


@given(_two_distributions_same_size())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cross_entropy_geq_entropy(pq):
    """H(p, q) ≥ H(p) — cross entropy is at least self-entropy."""
    p, q = pq
    h_p = shannon_entropy(p)
    h_pq = cross_entropy(p, q)
    assert h_pq >= h_p - _ATOL, \
        f"H(p, q) = {h_pq} < H(p) = {h_p}"


# ---------------------------------------------------------------------------
# Chain rule of entropy: H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
# ---------------------------------------------------------------------------


@given(_joint_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_chain_rule_of_entropy(pxy):
    """H(X,Y) = H(X) + H(Y|X)."""
    hxy = joint_entropy(pxy)
    px = pxy.sum(axis=1)
    hx = shannon_entropy(px)
    hy_given_x = conditional_entropy_yx(pxy)
    lhs = hxy
    rhs = hx + hy_given_x
    assert math.isclose(lhs, rhs, abs_tol=1e-4), \
        f"H(X,Y) = {lhs} != H(X) + H(Y|X) = {rhs}"


@given(_joint_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_chain_rule_other_direction(pxy):
    """H(X,Y) = H(Y) + H(X|Y)."""
    hxy = joint_entropy(pxy)
    py = pxy.sum(axis=0)
    hy = shannon_entropy(py)
    hx_given_y = conditional_entropy(pxy)
    lhs = hxy
    rhs = hy + hx_given_y
    assert math.isclose(lhs, rhs, abs_tol=1e-4), \
        f"H(X,Y) = {lhs} != H(Y) + H(X|Y) = {rhs}"


# ---------------------------------------------------------------------------
# Conditional entropy ≤ marginal entropy: H(X|Y) ≤ H(X)
# ---------------------------------------------------------------------------


@given(_joint_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_conditioning_reduces_entropy(pxy):
    """H(X|Y) ≤ H(X): conditioning cannot increase entropy."""
    px = pxy.sum(axis=1)
    hx = shannon_entropy(px)
    hx_given_y = conditional_entropy(pxy)
    assert hx_given_y <= hx + _ATOL, \
        f"H(X|Y) = {hx_given_y} > H(X) = {hx}"


# ---------------------------------------------------------------------------
# Data processing inequality via channel capacity
# ---------------------------------------------------------------------------


@given(_transition_matrix(min_in=2, max_in=4, min_out=2, max_out=4),
       _transition_matrix(min_in=2, max_in=4, min_out=2, max_out=4))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_data_processing_inequality_capacity(W1, W2):
    """Cascading two channels cannot increase capacity: C(W1·W2) ≤ C(W1).

    This is a consequence of the data processing inequality.
    """
    assume(W1.shape[1] == W2.shape[0])
    W_cascade = W1 @ W2
    # Normalize rows
    row_sums = W_cascade.sum(axis=1, keepdims=True)
    assume(np.all(row_sums > 0))
    W_cascade /= row_sums

    cap1 = blahut_arimoto(W1, max_iterations=200).capacity_bits
    cap_cascade = blahut_arimoto(W_cascade, max_iterations=200).capacity_bits
    assert cap_cascade <= cap1 + 0.01, \
        f"Cascade capacity {cap_cascade} > W1 capacity {cap1}"


# ---------------------------------------------------------------------------
# Channel capacity ≥ 0 and ≤ log(|output alphabet|)
# ---------------------------------------------------------------------------


@given(_transition_matrix())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_channel_capacity_non_negative(W):
    """Channel capacity C ≥ 0."""
    result = blahut_arimoto(W, max_iterations=200)
    assert result.capacity_bits >= -_ATOL, \
        f"Capacity is negative: {result.capacity_bits}"


@given(_transition_matrix())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_channel_capacity_upper_bound(W):
    """C ≤ log₂(|Y|) where |Y| is the output alphabet size."""
    n_y = W.shape[1]
    result = blahut_arimoto(W, max_iterations=200)
    assert result.capacity_bits <= math.log2(n_y) + 0.01, \
        f"Capacity {result.capacity_bits} > log2({n_y}) = {math.log2(n_y)}"


# ---------------------------------------------------------------------------
# BSC capacity: C = 1 - h(p)
# ---------------------------------------------------------------------------


@given(_crossover)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_bsc_capacity_non_negative(p):
    """BSC capacity 1 - h(p) ≥ 0."""
    c = bsc_capacity(p)
    assert c >= -_ATOL, f"BSC capacity negative: {c}"


@given(_crossover)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_bsc_capacity_at_most_one(p):
    """BSC capacity ≤ 1 bit."""
    c = bsc_capacity(p)
    assert c <= 1.0 + _ATOL, f"BSC capacity > 1: {c}"


def test_bsc_capacity_perfect_channel():
    """BSC(0) has capacity 1 bit."""
    c = bsc_capacity(0.0)
    assert math.isclose(c, 1.0, abs_tol=_ATOL)


def test_bsc_capacity_useless_channel():
    """BSC(0.5) has capacity 0 bits."""
    c = bsc_capacity(0.5)
    assert math.isclose(c, 0.0, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# BEC capacity: C = 1 - ε
# ---------------------------------------------------------------------------


@given(_crossover)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_bec_capacity_formula(eps):
    """BEC capacity equals 1 - ε."""
    c = bec_capacity(eps)
    assert math.isclose(c, 1.0 - eps, abs_tol=_ATOL), \
        f"BEC({eps}) capacity = {c} != {1.0 - eps}"


# ---------------------------------------------------------------------------
# Gaussian channel capacity
# ---------------------------------------------------------------------------


@given(floats(min_value=0.01, max_value=100.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.01, max_value=100.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_gaussian_capacity_non_negative(power, noise):
    """AWGN channel capacity ≥ 0."""
    c = gaussian_channel_capacity(power, noise)
    assert c >= -_ATOL, f"Gaussian capacity negative: {c}"


@given(floats(min_value=0.01, max_value=100.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.01, max_value=100.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_gaussian_capacity_increases_with_power(power, noise):
    """Higher power → higher Gaussian channel capacity."""
    c1 = gaussian_channel_capacity(power, noise)
    c2 = gaussian_channel_capacity(power * 2.0, noise)
    assert c2 >= c1 - _ATOL, \
        f"Doubling power decreased capacity: {c2} < {c1}"


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------


@given(_two_distributions_same_size())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_jsd_non_negative(pq):
    """JSD(p || q) ≥ 0."""
    p, q = pq
    jsd = jensen_shannon_divergence(p, q)
    assert jsd >= -_ATOL, f"JSD is negative: {jsd}"


@given(_two_distributions_same_size())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_jsd_symmetric(pq):
    """JSD(p || q) = JSD(q || p)."""
    p, q = pq
    jsd_pq = jensen_shannon_divergence(p, q)
    jsd_qp = jensen_shannon_divergence(q, p)
    assert math.isclose(jsd_pq, jsd_qp, abs_tol=_ATOL), \
        f"JSD not symmetric: {jsd_pq} vs {jsd_qp}"


@given(_probability_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_jsd_zero_for_identical(p):
    """JSD(p || p) = 0."""
    jsd = jensen_shannon_divergence(p, p)
    assert math.isclose(jsd, 0.0, abs_tol=_ATOL), \
        f"JSD(p || p) = {jsd} != 0"


# ---------------------------------------------------------------------------
# Total correlation non-negative
# ---------------------------------------------------------------------------


@given(_joint_distribution())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_total_correlation_non_negative(pxy):
    """Total correlation C(X₁,...,Xₙ) ≥ 0."""
    tc = total_correlation(pxy)
    assert tc >= -_ATOL, f"Total correlation negative: {tc}"
