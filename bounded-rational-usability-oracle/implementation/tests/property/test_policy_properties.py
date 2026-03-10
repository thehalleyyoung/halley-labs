"""Property-based tests for softmax policy construction.

This module verifies probabilistic and information-theoretic properties of
the SoftmaxPolicy using Hypothesis. Properties tested include probability
normalization, range constraints, entropy-monotonicity with respect to beta,
convergence to greedy/uniform limits, and KL divergence non-negativity.
"""

import math

import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats, integers, lists, tuples, sampled_from,
)

from usability_oracle.policy.models import Policy, QValues
from usability_oracle.policy.softmax import SoftmaxPolicy


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_q_val = floats(min_value=-50.0, max_value=50.0,
                allow_nan=False, allow_infinity=False)

_pos_beta = floats(min_value=0.01, max_value=20.0,
                   allow_nan=False, allow_infinity=False)

_n_actions = integers(min_value=2, max_value=8)

_n_states = integers(min_value=1, max_value=5)


def _make_q_values(n_states, n_actions, q_vals_flat):
    """Build a QValues object from a flat list of Q-values.

    Distributes q_vals_flat across states and actions in a round-robin
    fashion and pads with zeros if necessary.
    """
    states = [f"s{i}" for i in range(n_states)]
    actions = [f"a{j}" for j in range(n_actions)]
    values = {}
    idx = 0
    for s in states:
        values[s] = {}
        for a in actions:
            if idx < len(q_vals_flat):
                values[s][a] = q_vals_flat[idx]
                idx += 1
            else:
                values[s][a] = 0.0
    return QValues(values=values)


def _make_simple_q(n_actions, q_list):
    """Build a single-state QValues from a list of action values."""
    actions = [f"a{j}" for j in range(n_actions)]
    vals = {actions[i]: q_list[i] for i in range(min(len(q_list), n_actions))}
    return QValues(values={"s0": vals})


# ---------------------------------------------------------------------------
# Probabilities sum to 1
# ---------------------------------------------------------------------------


@given(_n_actions, lists(_q_val, min_size=2, max_size=8), _pos_beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_probabilities_sum_to_one(n_act, q_list, beta):
    """Softmax probabilities sum to 1 for every state.

    The Boltzmann distribution must be properly normalized so that
    Σ_a π(a|s) = 1 for every state s.
    """
    n_act = min(n_act, len(q_list))
    assume(n_act >= 2)
    qv = _make_simple_q(n_act, q_list[:n_act])
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    probs = policy.action_probs("s0")
    total = sum(probs.values())
    assert math.isclose(total, 1.0, abs_tol=1e-6), f"Sum = {total}"


@given(_n_states, _n_actions,
       lists(_q_val, min_size=4, max_size=40), _pos_beta)
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_all_states_sum_to_one(n_s, n_a, q_flat, beta):
    """Every state's action probabilities sum to 1.

    For a multi-state MDP, each state independently has a normalized
    softmax distribution.
    """
    assume(n_s >= 1 and n_a >= 2)
    qv = _make_q_values(n_s, n_a, q_flat)
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    for s in qv.values:
        probs = policy.action_probs(s)
        total = sum(probs.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# All probabilities in [0, 1]
# ---------------------------------------------------------------------------

@given(_n_actions, lists(_q_val, min_size=2, max_size=8), _pos_beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_probabilities_in_unit_range(n_act, q_list, beta):
    """Every action probability is in [0, 1].

    No probability may be negative or exceed 1.
    """
    n_act = min(n_act, len(q_list))
    assume(n_act >= 2)
    qv = _make_simple_q(n_act, q_list[:n_act])
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    probs = policy.action_probs("s0")
    for a, p in probs.items():
        assert -1e-9 <= p <= 1.0 + 1e-9, f"P({a}) = {p} out of range"


# ---------------------------------------------------------------------------
# Higher beta → lower entropy
# ---------------------------------------------------------------------------

@given(lists(_q_val, min_size=2, max_size=6))
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_higher_beta_lower_entropy(q_list):
    """Increasing beta (rationality) decreases entropy.

    A more rational agent concentrates probability on the best action,
    reducing the Shannon entropy of the policy.
    """
    assume(len(q_list) >= 2)
    assume(max(q_list) - min(q_list) > 0.1)  # non-degenerate Q-values
    qv = _make_simple_q(len(q_list), q_list)
    beta_low = 0.1
    beta_high = 10.0
    p_low = SoftmaxPolicy.from_q_values(qv, beta_low)
    p_high = SoftmaxPolicy.from_q_values(qv, beta_high)
    assert p_high.entropy("s0") <= p_low.entropy("s0") + 1e-6


@given(lists(_q_val, min_size=3, max_size=6))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_entropy_monotone_decreasing(q_list):
    """Entropy is monotonically non-increasing as beta increases.

    For a sequence of increasing betas, entropy should not increase.
    """
    assume(len(q_list) >= 3)
    assume(max(q_list) - min(q_list) > 0.1)
    qv = _make_simple_q(len(q_list), q_list)
    betas = [0.1, 1.0, 5.0, 20.0]
    entropies = []
    for b in betas:
        pol = SoftmaxPolicy.from_q_values(qv, b)
        entropies.append(pol.entropy("s0"))
    for i in range(len(entropies) - 1):
        assert entropies[i + 1] <= entropies[i] + 1e-6


# ---------------------------------------------------------------------------
# Beta → ∞: greedy convergence
# ---------------------------------------------------------------------------

@given(lists(_q_val, min_size=2, max_size=6))
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_large_beta_approaches_greedy(q_list):
    """With very large beta, the policy becomes nearly greedy.

    The action with the lowest Q-value (best for cost minimisation)
    should receive probability close to 1.
    """
    assume(len(q_list) >= 2)
    vals = q_list[:len(q_list)]
    unique = list(set(vals))
    assume(len(unique) >= 2)

    qv = _make_simple_q(len(vals), vals)
    policy = SoftmaxPolicy.from_q_values(qv, beta=100.0)
    probs = policy.action_probs("s0")

    # Find the action with the minimum Q-value
    min_q = min(vals)
    min_actions = [f"a{i}" for i, v in enumerate(vals) if v == min_q]
    total_min_prob = sum(probs.get(a, 0.0) for a in min_actions)
    assert total_min_prob > 0.9, f"Greedy prob = {total_min_prob}"


@given(lists(_q_val, min_size=2, max_size=6))
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_large_beta_is_deterministic(q_list):
    """With very large beta, the policy should be nearly deterministic.

    is_deterministic should return True with a reasonable threshold.
    """
    assume(len(q_list) >= 2)
    unique = list(set(q_list))
    assume(len(unique) >= 2)
    qv = _make_simple_q(len(q_list), q_list)
    policy = SoftmaxPolicy.from_q_values(qv, beta=200.0)
    assert policy.is_deterministic("s0", threshold=0.95)


# ---------------------------------------------------------------------------
# Beta → 0: uniform convergence
# ---------------------------------------------------------------------------

@given(_n_actions)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_small_beta_approaches_uniform(n_act):
    """With very small beta, the policy approaches uniform.

    When beta → 0 the agent ignores Q-values and explores uniformly.
    Each action should get probability ≈ 1/n_actions.
    """
    assume(n_act >= 2)
    q_list = [float(i) for i in range(n_act)]
    qv = _make_simple_q(n_act, q_list)
    policy = SoftmaxPolicy.from_q_values(qv, beta=0.001)
    probs = policy.action_probs("s0")
    expected = 1.0 / n_act
    for a, p in probs.items():
        assert math.isclose(p, expected, abs_tol=0.05), \
            f"P({a}) = {p}, expected ≈ {expected}"


@given(_n_actions)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_small_beta_high_entropy(n_act):
    """With very small beta, entropy approaches log(n_actions).

    The uniform distribution maximises Shannon entropy.
    """
    assume(n_act >= 2)
    q_list = [float(i) for i in range(n_act)]
    qv = _make_simple_q(n_act, q_list)
    policy = SoftmaxPolicy.from_q_values(qv, beta=0.001)
    h = policy.entropy("s0")
    max_h = math.log(n_act)
    assert math.isclose(h, max_h, abs_tol=0.1)


# ---------------------------------------------------------------------------
# Entropy is non-negative
# ---------------------------------------------------------------------------

@given(_n_actions, lists(_q_val, min_size=2, max_size=8), _pos_beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_entropy_non_negative(n_act, q_list, beta):
    """Shannon entropy of a softmax policy is always non-negative.

    H(π) = -Σ π(a|s) log π(a|s) >= 0 by definition.
    """
    n_act = min(n_act, len(q_list))
    assume(n_act >= 2)
    qv = _make_simple_q(n_act, q_list[:n_act])
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    assert policy.entropy("s0") >= -1e-9


# ---------------------------------------------------------------------------
# Entropy bounded above by log(n)
# ---------------------------------------------------------------------------

@given(_n_actions, lists(_q_val, min_size=2, max_size=8), _pos_beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_entropy_bounded_above(n_act, q_list, beta):
    """Entropy is at most log(n_actions).

    The uniform distribution has maximum entropy log(n).
    """
    n_act = min(n_act, len(q_list))
    assume(n_act >= 2)
    qv = _make_simple_q(n_act, q_list[:n_act])
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    assert policy.entropy("s0") <= math.log(n_act) + 1e-6


# ---------------------------------------------------------------------------
# KL divergence >= 0
# ---------------------------------------------------------------------------

@given(lists(_q_val, min_size=2, max_size=6), _pos_beta, _pos_beta)
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_kl_divergence_non_negative(q_list, beta1, beta2):
    """KL divergence between any two softmax policies is non-negative.

    D_KL(p || q) >= 0 by Gibbs' inequality.
    """
    assume(len(q_list) >= 2)
    qv = _make_simple_q(len(q_list), q_list)
    p = SoftmaxPolicy.from_q_values(qv, beta1)
    q = SoftmaxPolicy.from_q_values(qv, beta2)
    kl = SoftmaxPolicy.kl_divergence(p, q, "s0")
    assert kl >= -1e-9, f"KL = {kl}"


# ---------------------------------------------------------------------------
# KL divergence == 0 for identical distributions
# ---------------------------------------------------------------------------

@given(lists(_q_val, min_size=2, max_size=6), _pos_beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_kl_divergence_self_is_zero(q_list, beta):
    """KL divergence of a policy with itself is zero: D_KL(p || p) = 0.

    A distribution is zero-divergent from itself.
    """
    assume(len(q_list) >= 2)
    qv = _make_simple_q(len(q_list), q_list)
    p = SoftmaxPolicy.from_q_values(qv, beta)
    kl = SoftmaxPolicy.kl_divergence(p, p, "s0")
    assert math.isclose(kl, 0.0, abs_tol=1e-6), f"KL(self) = {kl}"


# ---------------------------------------------------------------------------
# kl_from_uniform >= 0
# ---------------------------------------------------------------------------

@given(lists(_q_val, min_size=2, max_size=6), _pos_beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_kl_from_uniform_non_negative(q_list, beta):
    """KL divergence from uniform is non-negative.

    D_KL(π || uniform) >= 0 since the uniform distribution has max entropy.
    """
    assume(len(q_list) >= 2)
    qv = _make_simple_q(len(q_list), q_list)
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    assert policy.kl_from_uniform("s0") >= -1e-9


# ---------------------------------------------------------------------------
# QValues → Policy round-trip consistency
# ---------------------------------------------------------------------------

@given(lists(_q_val, min_size=2, max_size=6), _pos_beta)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_qvalues_to_policy_consistent(q_list, beta):
    """QValues.to_policy(beta) produces same result as SoftmaxPolicy.from_q_values.

    Both conversion paths should agree on the action probabilities.
    """
    assume(len(q_list) >= 2)
    qv = _make_simple_q(len(q_list), q_list)
    p1 = qv.to_policy(beta)
    p2 = SoftmaxPolicy.from_q_values(qv, beta)
    probs1 = p1.action_probs("s0")
    probs2 = p2.action_probs("s0")
    for a in probs1:
        assert math.isclose(probs1[a], probs2[a], abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Policy.n_states consistency
# ---------------------------------------------------------------------------

@given(_n_states, _n_actions,
       lists(_q_val, min_size=4, max_size=40), _pos_beta)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_policy_n_states(n_s, n_a, q_flat, beta):
    """Policy.n_states() matches the number of states in QValues.

    The policy should have exactly as many states as the Q-value table.
    """
    assume(n_s >= 1 and n_a >= 2)
    qv = _make_q_values(n_s, n_a, q_flat)
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    assert policy.n_states() == n_s


# ---------------------------------------------------------------------------
# Beta sweep produces correct number of policies
# ---------------------------------------------------------------------------

@given(lists(_q_val, min_size=2, max_size=6))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_beta_sweep_length(q_list):
    """beta_sweep returns one policy per beta value.

    The number of policies returned must equal the number of betas given.
    """
    assume(len(q_list) >= 2)
    qv = _make_simple_q(len(q_list), q_list)
    betas = [0.1, 1.0, 5.0, 10.0]
    policies = SoftmaxPolicy.beta_sweep(qv, betas)
    assert len(policies) == len(betas)


# ---------------------------------------------------------------------------
# sample_action returns valid action
# ---------------------------------------------------------------------------

@given(lists(_q_val, min_size=2, max_size=6), _pos_beta)
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_sample_action_valid(q_list, beta):
    """sample_action returns an action from the action set.

    The sampled action must be one of the actions defined for that state.
    """
    assume(len(q_list) >= 2)
    qv = _make_simple_q(len(q_list), q_list)
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    rng = np.random.default_rng(42)
    action = policy.sample_action("s0", rng=rng)
    assert action in policy.action_probs("s0")


# ---------------------------------------------------------------------------
# mean_entropy bounded
# ---------------------------------------------------------------------------

@given(_n_states, _n_actions,
       lists(_q_val, min_size=4, max_size=40), _pos_beta)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_mean_entropy_bounded(n_s, n_a, q_flat, beta):
    """Mean entropy is between 0 and log(n_actions).

    The average entropy across all states must be within the entropy bounds.
    """
    assume(n_s >= 1 and n_a >= 2)
    qv = _make_q_values(n_s, n_a, q_flat)
    policy = SoftmaxPolicy.from_q_values(qv, beta)
    me = policy.mean_entropy()
    assert me >= -1e-9
    assert me <= math.log(n_a) + 1e-6
