"""
usability_oracle.information_theory.free_energy — Free energy computations.

Implements variational free energy F = E[R] − (1/β) D_KL(π ‖ p₀) and related
quantities for bounded-rational decision making.  Includes minimization
algorithms, Bethe approximation, temperature annealing, and landscape analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as sp_optimize

from usability_oracle.information_theory.entropy import (
    _as_prob,
    _safe_log,
    shannon_entropy,
)
from usability_oracle.information_theory.mutual_information import kl_divergence


_LOG2 = math.log(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Core free energy
# ═══════════════════════════════════════════════════════════════════════════

def variational_free_energy(
    policy: Union[Sequence[float], NDArray],
    prior: Union[Sequence[float], NDArray],
    rewards: Union[Sequence[float], NDArray],
    beta: float,
    *,
    base: float = math.e,
) -> float:
    """Variational free energy F(π) = E_π[R] − (1/β) D_KL(π ‖ p₀).

    In the bounded-rational framework, the agent maximizes F, or equivalently
    minimizes −F = (1/β) D_KL(π ‖ p₀) − E_π[R].

    Parameters
    ----------
    policy : array-like
        Agent policy distribution π over actions.
    prior : array-like
        Prior (default) distribution p₀ over actions.
    rewards : array-like
        Reward R(a) for each action.
    beta : float
        Rationality parameter (inverse temperature).  β → 0: random,
        β → ∞: fully rational.
    base : float
        Logarithm base for KL divergence.

    Returns
    -------
    float
        Free energy value.
    """
    pi = _as_prob(policy)
    p0 = _as_prob(prior)
    r = np.asarray(rewards, dtype=np.float64)

    expected_reward = float(np.dot(pi, r))
    kl = kl_divergence(pi, p0, base=base)

    if beta <= 0:
        return expected_reward
    return expected_reward - (1.0 / beta) * kl


def free_energy_decomposition(
    policy: Union[Sequence[float], NDArray],
    prior: Union[Sequence[float], NDArray],
    rewards: Union[Sequence[float], NDArray],
    beta: float,
) -> dict:
    """Decompose free energy into accuracy and complexity terms.

    F = Accuracy - Complexity
    Accuracy = E_π[R]  (expected reward under policy)
    Complexity = (1/β) D_KL(π ‖ p₀)  (information cost)

    Parameters
    ----------
    policy, prior, rewards, beta : see variational_free_energy

    Returns
    -------
    dict
        Keys: "free_energy", "accuracy", "complexity", "beta"
    """
    pi = _as_prob(policy)
    p0 = _as_prob(prior)
    r = np.asarray(rewards, dtype=np.float64)

    accuracy = float(np.dot(pi, r))
    kl = kl_divergence(pi, p0, base=math.e)
    complexity = (1.0 / beta) * kl if beta > 0 else 0.0

    return {
        "free_energy": accuracy - complexity,
        "accuracy": accuracy,
        "complexity": complexity,
        "beta": beta,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Optimal (bounded-rational) policy
# ═══════════════════════════════════════════════════════════════════════════

def optimal_policy(
    prior: Union[Sequence[float], NDArray],
    rewards: Union[Sequence[float], NDArray],
    beta: float,
) -> NDArray:
    """Compute the optimal bounded-rational policy.

    π*(a) ∝ p₀(a) exp(β R(a))

    This is the unique minimizer of −F = (1/β) D_KL(π ‖ p₀) − E_π[R].

    Parameters
    ----------
    prior : array-like
        Prior distribution p₀ over actions.
    rewards : array-like
        Reward for each action.
    beta : float
        Inverse temperature.

    Returns
    -------
    NDArray
        Optimal policy distribution π*.
    """
    p0 = _as_prob(prior)
    r = np.asarray(rewards, dtype=np.float64)

    if beta <= 0:
        return p0.copy()

    log_unnorm = np.log(np.maximum(p0, 1e-300)) + beta * r
    log_unnorm -= log_unnorm.max()  # numerical stability
    unnorm = np.exp(log_unnorm)
    total = unnorm.sum()
    if total > 0:
        return unnorm / total
    return p0.copy()


def optimal_free_energy(
    prior: Union[Sequence[float], NDArray],
    rewards: Union[Sequence[float], NDArray],
    beta: float,
) -> float:
    """Free energy achieved by the optimal policy.

    F* = (1/β) log Σ_a p₀(a) exp(β R(a))

    Parameters
    ----------
    prior : array-like
        Prior distribution.
    rewards : array-like
        Rewards.
    beta : float
        Inverse temperature.

    Returns
    -------
    float
        Optimal free energy.
    """
    p0 = _as_prob(prior)
    r = np.asarray(rewards, dtype=np.float64)

    if beta <= 0:
        return float(np.dot(p0, r))

    # log-sum-exp trick
    br = beta * r
    log_p0 = _safe_log(p0)
    log_terms = log_p0 + br
    max_term = log_terms[p0 > 0].max() if np.any(p0 > 0) else 0.0
    log_Z = max_term + math.log(float(np.sum(np.where(p0 > 0, np.exp(log_terms - max_term), 0.0))))
    return log_Z / beta


# ═══════════════════════════════════════════════════════════════════════════
# Free energy minimization (iterative)
# ═══════════════════════════════════════════════════════════════════════════

def minimize_free_energy(
    prior: Union[Sequence[float], NDArray],
    reward_fn: Callable[[NDArray], float],
    beta: float,
    *,
    n_actions: Optional[int] = None,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> Tuple[NDArray, float]:
    """Minimize negative free energy via iterative softmax updates.

    For general (possibly state-dependent) reward functions, this performs
    policy iteration in the information-bounded setting.

    Parameters
    ----------
    prior : array-like
        Prior distribution p₀.
    reward_fn : callable
        Maps policy → expected reward.  For simple case, use np.dot(π, r).
    beta : float
        Inverse temperature.
    n_actions : int or None
        Number of actions (inferred from prior if None).
    tolerance : float
        Convergence tolerance.
    max_iterations : int
        Maximum iterations.

    Returns
    -------
    tuple of (NDArray, float)
        (optimal_policy, free_energy)
    """
    p0 = _as_prob(prior)
    n = n_actions or len(p0)
    pi = p0.copy()

    best_fe = -float("inf")
    for _ in range(max_iterations):
        reward = reward_fn(pi)
        fe = reward - (1.0 / beta) * kl_divergence(pi, p0, base=math.e) if beta > 0 else reward
        if abs(fe - best_fe) < tolerance:
            return pi, fe
        best_fe = fe

        # Estimate per-action rewards via finite differences
        r_est = np.zeros(n)
        eps = 1e-6
        for a in range(n):
            pi_plus = pi.copy()
            pi_plus[a] = min(pi_plus[a] + eps, 1.0)
            pi_plus /= pi_plus.sum()
            r_est[a] = reward_fn(pi_plus)

        # Softmax update
        pi = optimal_policy(p0, r_est, beta)

    return pi, best_fe


# ═══════════════════════════════════════════════════════════════════════════
# Expected free energy (active inference)
# ═══════════════════════════════════════════════════════════════════════════

def expected_free_energy(
    likelihood: Union[Sequence[Sequence[float]], NDArray],
    prior_states: Union[Sequence[float], NDArray],
    preferred_obs: Union[Sequence[float], NDArray],
) -> NDArray:
    """Expected free energy G(π) for active inference.

    G = ambiguity + risk
      = E_q[H(o|s)] - E_q[log p̃(o)]

    where q is the posterior predictive distribution under policy π.

    Parameters
    ----------
    likelihood : 2-D array-like
        Likelihood matrix p(o|s), shape (n_states, n_obs).
    prior_states : array-like
        Prior over states q(s).
    preferred_obs : array-like
        Preferred observation distribution p̃(o) (log preferences).

    Returns
    -------
    NDArray
        Expected free energy contributions per state.
    """
    A = _as_prob(likelihood)
    qs = _as_prob(prior_states)
    pref = np.asarray(preferred_obs, dtype=np.float64)

    n_states, n_obs = A.shape

    # Ambiguity: E_q[H(o|s)] = Σ_s q(s) H(A(·|s))
    ambiguity = 0.0
    for s in range(n_states):
        ambiguity += qs[s] * shannon_entropy(A[s], base=math.e)

    # Risk: -E_q[log p̃(o)] = -Σ_o q(o) log p̃(o)
    qo = qs @ A  # predictive distribution q(o) = Σ_s q(s) p(o|s)
    log_pref = _safe_log(pref)
    risk = -float(np.dot(qo, log_pref))

    # Return as array per state for downstream use
    G = np.zeros(n_states)
    for s in range(n_states):
        state_ambiguity = shannon_entropy(A[s], base=math.e)
        state_risk = -float(np.dot(A[s], log_pref))
        G[s] = state_ambiguity + state_risk

    return G


# ═══════════════════════════════════════════════════════════════════════════
# Bethe free energy approximation
# ═══════════════════════════════════════════════════════════════════════════

def bethe_free_energy(
    node_beliefs: Sequence[NDArray],
    edge_beliefs: Sequence[NDArray],
    node_potentials: Sequence[NDArray],
    edge_potentials: Sequence[NDArray],
    node_degrees: Sequence[int],
) -> float:
    """Bethe free energy approximation for a factor graph.

    F_Bethe = Σ_a E_b_a[log(b_a/ψ_a)] - Σ_i (d_i - 1) E_b_i[log b_i]

    where b_a are factor beliefs, b_i are variable beliefs, ψ_a are
    factor potentials, and d_i is the degree of variable i.

    Parameters
    ----------
    node_beliefs : sequence of NDArray
        Beliefs b_i for each variable node.
    edge_beliefs : sequence of NDArray
        Beliefs b_a for each factor.
    node_potentials : sequence of NDArray
        Potentials for variable nodes.
    edge_potentials : sequence of NDArray
        Potentials for factor nodes.
    node_degrees : sequence of int
        Degree of each variable node in the factor graph.

    Returns
    -------
    float
        Bethe free energy.
    """
    # Energy term: Σ_a E_{b_a}[-log ψ_a]
    energy = 0.0
    for b_a, psi_a in zip(edge_beliefs, edge_potentials):
        b_flat = b_a.ravel()
        psi_flat = np.maximum(psi_a.ravel(), 1e-300)
        energy -= float(np.dot(b_flat, np.log(psi_flat)))

    # Factor entropy: -Σ_a E_{b_a}[log b_a]
    factor_entropy = 0.0
    for b_a in edge_beliefs:
        b_flat = b_a.ravel()
        mask = b_flat > 0
        factor_entropy -= float(np.dot(b_flat[mask], np.log(b_flat[mask])))

    # Variable entropy: Σ_i (d_i - 1) E_{b_i}[log b_i]
    var_entropy_correction = 0.0
    for b_i, d_i in zip(node_beliefs, node_degrees):
        b_flat = b_i.ravel()
        mask = b_flat > 0
        h_i = -float(np.dot(b_flat[mask], np.log(b_flat[mask])))
        var_entropy_correction += (d_i - 1) * h_i

    return energy - factor_entropy + var_entropy_correction


# ═══════════════════════════════════════════════════════════════════════════
# Temperature annealing schedules
# ═══════════════════════════════════════════════════════════════════════════

def linear_annealing(
    beta_start: float,
    beta_end: float,
    n_steps: int,
) -> NDArray:
    """Linear β annealing schedule.

    Parameters
    ----------
    beta_start : float
        Initial β value.
    beta_end : float
        Final β value.
    n_steps : int
        Number of annealing steps.

    Returns
    -------
    NDArray
        Array of β values.
    """
    return np.linspace(beta_start, beta_end, n_steps)


def exponential_annealing(
    beta_start: float,
    beta_end: float,
    n_steps: int,
) -> NDArray:
    """Exponential β annealing schedule.

    β(t) = β_start × (β_end/β_start)^{t/(n-1)}

    Parameters
    ----------
    beta_start, beta_end : float
        Start and end β values (both must be > 0).
    n_steps : int
        Number of steps.

    Returns
    -------
    NDArray
        Array of β values.
    """
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError("β values must be positive for exponential schedule")
    return np.logspace(
        math.log10(beta_start), math.log10(beta_end), n_steps,
    )


def cosine_annealing(
    beta_start: float,
    beta_end: float,
    n_steps: int,
) -> NDArray:
    """Cosine β annealing schedule.

    β(t) = β_end + 0.5(β_start - β_end)(1 + cos(πt/(n-1)))

    Parameters
    ----------
    beta_start, beta_end : float
        Start and end β values.
    n_steps : int
        Number of steps.

    Returns
    -------
    NDArray
        Array of β values.
    """
    t = np.arange(n_steps, dtype=np.float64)
    if n_steps > 1:
        t /= (n_steps - 1)
    return beta_end + 0.5 * (beta_start - beta_end) * (1.0 + np.cos(math.pi * t))


# ═══════════════════════════════════════════════════════════════════════════
# Free energy landscape analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LandscapePoint:
    """A point on the free energy landscape."""
    beta: float
    policy: NDArray
    free_energy: float
    is_local_minimum: bool = False
    is_saddle: bool = False


def free_energy_landscape(
    prior: Union[Sequence[float], NDArray],
    rewards: Union[Sequence[float], NDArray],
    beta_values: Union[Sequence[float], NDArray],
) -> list[LandscapePoint]:
    """Analyze the free energy landscape across β values.

    Computes the optimal policy and free energy at each β,
    identifying phase transitions and critical points.

    Parameters
    ----------
    prior : array-like
        Prior distribution.
    rewards : array-like
        Rewards.
    beta_values : array-like
        β values to evaluate.

    Returns
    -------
    list[LandscapePoint]
        Landscape analysis points.
    """
    p0 = _as_prob(prior)
    r = np.asarray(rewards, dtype=np.float64)
    betas = np.asarray(beta_values, dtype=np.float64)

    points: list[LandscapePoint] = []
    for b in betas:
        pi = optimal_policy(p0, r, float(b))
        fe = optimal_free_energy(p0, r, float(b))
        points.append(LandscapePoint(beta=float(b), policy=pi, free_energy=fe))

    # Detect phase transitions: discontinuities in dF/dβ
    if len(points) > 2:
        fes = np.array([p.free_energy for p in points])
        dfe = np.diff(fes)
        d2fe = np.diff(dfe)
        for i in range(1, len(points) - 1):
            if i - 1 < len(d2fe) and abs(d2fe[i - 1]) > np.std(d2fe) * 2:
                points[i].is_saddle = True

    return points


def find_phase_transitions(
    prior: Union[Sequence[float], NDArray],
    rewards: Union[Sequence[float], NDArray],
    beta_range: Tuple[float, float] = (0.01, 100.0),
    n_points: int = 200,
) -> list[float]:
    """Find phase transition points in β.

    Phase transitions occur where the optimal policy changes
    qualitatively (discontinuity in the policy or its derivative).

    Parameters
    ----------
    prior : array-like
        Prior distribution.
    rewards : array-like
        Rewards.
    beta_range : tuple
        (β_min, β_max) search range.
    n_points : int
        Resolution for the search.

    Returns
    -------
    list[float]
        β values at which phase transitions occur.
    """
    p0 = _as_prob(prior)
    r = np.asarray(rewards, dtype=np.float64)
    betas = np.logspace(
        math.log10(beta_range[0]), math.log10(beta_range[1]), n_points,
    )

    policies = np.array([optimal_policy(p0, r, float(b)) for b in betas])
    transitions: list[float] = []

    # Detect transitions via large jumps in policy
    for i in range(1, len(betas)):
        policy_diff = np.max(np.abs(policies[i] - policies[i - 1]))
        beta_diff = betas[i] - betas[i - 1]
        # Normalized rate of change
        if beta_diff > 0:
            rate = policy_diff / beta_diff
            if i >= 2:
                prev_rate = np.max(np.abs(policies[i - 1] - policies[i - 2])) / (
                    betas[i - 1] - betas[i - 2]
                ) if betas[i - 1] - betas[i - 2] > 0 else 0
                if rate > 5 * max(prev_rate, 1e-10):
                    transitions.append(float(betas[i]))

    return transitions


def metastable_states(
    prior: Union[Sequence[float], NDArray],
    rewards: Union[Sequence[float], NDArray],
    beta: float,
    *,
    n_random_starts: int = 50,
    tolerance: float = 1e-8,
) -> list[Tuple[NDArray, float]]:
    """Find metastable states (local minima) of the free energy.

    Uses random initialization to discover multiple local minima.

    Parameters
    ----------
    prior : array-like
        Prior distribution.
    rewards : array-like
        Rewards.
    beta : float
        Inverse temperature.
    n_random_starts : int
        Number of random starting policies.
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    list[tuple[NDArray, float]]
        List of (policy, free_energy) for each local minimum found.
    """
    p0 = _as_prob(prior)
    r = np.asarray(rewards, dtype=np.float64)
    n = len(p0)

    # The optimal policy is always the global maximum for this convex problem
    # but for non-linear reward functions, there can be multiple local optima
    found: list[Tuple[NDArray, float]] = []
    seen_fes: list[float] = []

    for _ in range(n_random_starts):
        # Random Dirichlet initialization
        pi = np.random.dirichlet(np.ones(n))
        # Iterate softmax
        for _ in range(200):
            fe = variational_free_energy(pi, p0, r, beta)
            pi_new = optimal_policy(p0, r, beta)
            if np.max(np.abs(pi_new - pi)) < tolerance:
                pi = pi_new
                break
            pi = pi_new
        fe = variational_free_energy(pi, p0, r, beta)

        # Check if this is a new local minimum
        is_new = True
        for seen_fe in seen_fes:
            if abs(fe - seen_fe) < tolerance * 10:
                is_new = False
                break
        if is_new:
            found.append((pi.copy(), fe))
            seen_fes.append(fe)

    found.sort(key=lambda x: -x[1])  # sort by descending free energy
    return found


# ═══════════════════════════════════════════════════════════════════════════
# Bounded-rational policy computation helpers
# ═══════════════════════════════════════════════════════════════════════════

def bounded_rational_value(
    prior: Union[Sequence[float], NDArray],
    q_values: Union[Sequence[float], NDArray],
    beta: float,
) -> float:
    """Soft value function V = (1/β) log Σ_a p₀(a) exp(β Q(a)).

    This is the free energy of the optimal bounded-rational policy
    given Q-values.

    Parameters
    ----------
    prior : array-like
        Prior action distribution.
    q_values : array-like
        Q-values for each action.
    beta : float
        Inverse temperature.

    Returns
    -------
    float
        Soft value.
    """
    return optimal_free_energy(prior, q_values, beta)


def information_cost(
    policy: Union[Sequence[float], NDArray],
    prior: Union[Sequence[float], NDArray],
    beta: float,
) -> float:
    """Information processing cost (1/β) D_KL(π ‖ p₀).

    Parameters
    ----------
    policy : array-like
        Current policy.
    prior : array-like
        Prior policy.
    beta : float
        Inverse temperature.

    Returns
    -------
    float
        Information cost.
    """
    if beta <= 0:
        return 0.0
    return (1.0 / beta) * kl_divergence(policy, prior, base=math.e)


__all__ = [
    "LandscapePoint",
    "bethe_free_energy",
    "bounded_rational_value",
    "cosine_annealing",
    "expected_free_energy",
    "exponential_annealing",
    "find_phase_transitions",
    "free_energy_decomposition",
    "free_energy_landscape",
    "information_cost",
    "linear_annealing",
    "metastable_states",
    "minimize_free_energy",
    "optimal_free_energy",
    "optimal_policy",
    "variational_free_energy",
]
