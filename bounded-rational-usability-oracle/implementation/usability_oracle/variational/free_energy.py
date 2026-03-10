"""
usability_oracle.variational.free_energy — Core free-energy computation.

Implements the bounded-rational objective:

.. math::

    F(\\pi) = \\mathbb{E}_\\pi[R] - \\frac{1}{\\beta}\\,
              D_{\\mathrm{KL}}(\\pi \\| p_0)

Provides:

* :class:`FreeEnergyComputer` — main computation engine
* :func:`compute_free_energy` — standalone objective evaluation
* :func:`compute_softmax_policy` — bounded-rational policy from Q-values
* :func:`compute_value_iteration` — soft Bellman iteration with KL penalty
* :func:`compute_policy_gradient` — gradient of F w.r.t. policy parameters
* :func:`compute_optimal_beta` — capacity parameter estimation
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from usability_oracle.variational.types import (
    ConvergenceInfo,
    ConvergenceStatus,
    FreeEnergyResult,
    KLDivergenceResult,
    VariationalConfig,
)
from usability_oracle.variational.kl_divergence import (
    compute_kl_divergence,
    compute_policy_kl,
)

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ═══════════════════════════════════════════════════════════════════════════
# Standalone functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_free_energy(
    policy: np.ndarray,
    reward: np.ndarray,
    prior: np.ndarray,
    beta: float,
) -> float:
    r"""Compute the variational free energy for a discrete policy.

    .. math::

        F(\pi) = \sum_a \pi(a)\,R(a)
                 - \frac{1}{\beta}\,D_{\mathrm{KL}}(\pi \| p_0)

    For the cost formulation (minimise cost rather than maximise reward),
    pass negative costs as *reward*.

    Parameters
    ----------
    policy : np.ndarray
        Policy distribution π(a), shape ``(n_actions,)``.
    reward : np.ndarray
        Expected reward per action, shape ``(n_actions,)``.
    prior : np.ndarray
        Prior/reference distribution p₀(a), shape ``(n_actions,)``.
    beta : float
        Rationality parameter (inverse temperature).  β → ∞ yields the
        optimal policy; β → 0⁺ yields the prior.

    Returns
    -------
    float
        Scalar free energy value.
    """
    policy = np.asarray(policy, dtype=np.float64).ravel()
    reward = np.asarray(reward, dtype=np.float64).ravel()
    prior = np.asarray(prior, dtype=np.float64).ravel()

    if not (policy.shape == reward.shape == prior.shape):
        raise ValueError("policy, reward, prior must have the same shape")

    expected_reward = float(np.dot(policy, reward))
    kl = compute_kl_divergence(policy, prior, validate=True)

    if beta <= 0:
        # β=0 → pure entropy maximisation; KL term dominates
        return -kl if np.isfinite(kl) else float("-inf")

    return expected_reward - (1.0 / beta) * kl


def compute_softmax_policy(
    q_values: np.ndarray,
    beta: float,
    prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Derive the bounded-rational softmax policy from Q-values.

    .. math::

        \pi^*(a) = \frac{p_0(a)\,\exp(\beta\,Q(a))}
                        {\sum_{a'} p_0(a')\,\exp(\beta\,Q(a'))}

    Parameters
    ----------
    q_values : np.ndarray
        State–action values Q(a), shape ``(n_actions,)`` or
        ``(n_states, n_actions)``.
    beta : float
        Rationality parameter.
    prior : np.ndarray, optional
        Prior distribution p₀(a).  Uniform if ``None``.

    Returns
    -------
    np.ndarray
        Policy probabilities, same shape as *q_values*.
    """
    q = np.asarray(q_values, dtype=np.float64)
    if q.ndim == 1:
        return _softmax_1d(q, beta, prior)

    if q.ndim == 2:
        n_states, n_actions = q.shape
        result = np.empty_like(q)
        for s in range(n_states):
            p_s = prior[s] if (prior is not None and prior.ndim == 2) else prior
            result[s] = _softmax_1d(q[s], beta, p_s)
        return result

    raise ValueError(f"q_values must be 1-D or 2-D, got ndim={q.ndim}")


def _softmax_1d(
    q: np.ndarray,
    beta: float,
    prior: Optional[np.ndarray],
) -> np.ndarray:
    """Softmax policy for a single state."""
    log_pi = beta * q
    if prior is not None:
        prior = np.asarray(prior, dtype=np.float64).ravel()
        log_pi = log_pi + np.log(np.maximum(prior, _EPS))

    # Numerically stable softmax via logsumexp
    log_pi -= logsumexp(log_pi)
    return np.exp(log_pi)


def compute_value_iteration(
    transition: np.ndarray,
    reward: np.ndarray,
    beta: float,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    prior_policy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    r"""Soft value iteration (Bellman with KL penalty).

    Iterates the soft Bellman equation:

    .. math::

        V(s) = \frac{1}{\beta}\,\ln\!\sum_a p_0(a|s)\,
               \exp\!\bigl(\beta\,[R(s,a) + \gamma\,\sum_{s'} T(s'|s,a)\,V(s')]\bigr)

    Parameters
    ----------
    transition : np.ndarray
        Transition tensor T(s'|s,a), shape ``(n_states, n_actions, n_states)``.
    reward : np.ndarray
        Reward matrix R(s,a), shape ``(n_states, n_actions)``.
    beta : float
        Rationality parameter.
    gamma : float
        Discount factor in [0, 1).
    tolerance : float
        Convergence criterion on ||V_new − V_old||_∞.
    max_iterations : int
        Maximum number of iterations.
    prior_policy : np.ndarray, optional
        Prior policy p₀(a|s), shape ``(n_states, n_actions)``.  Uniform if
        ``None``.

    Returns
    -------
    V : np.ndarray
        Optimal soft value function, shape ``(n_states,)``.
    policy : np.ndarray
        Optimal bounded-rational policy π*(a|s), shape ``(n_states, n_actions)``.
    iterations : int
        Number of iterations performed.
    """
    n_states, n_actions, _ = transition.shape
    if reward.shape != (n_states, n_actions):
        raise ValueError("reward shape must be (n_states, n_actions)")

    V = np.zeros(n_states, dtype=np.float64)

    if prior_policy is None:
        log_prior = np.full((n_states, n_actions), -np.log(n_actions))
    else:
        prior_policy = np.asarray(prior_policy, dtype=np.float64)
        log_prior = np.log(np.maximum(prior_policy, _EPS))

    for it in range(1, max_iterations + 1):
        # Q(s,a) = R(s,a) + gamma * sum_s' T(s'|s,a) * V(s')
        Q = reward + gamma * np.einsum("san,n->sa", transition, V)

        # Soft Bellman: V(s) = (1/beta) * logsumexp(beta*Q(s,:) + log p0(s,:))
        log_terms = beta * Q + log_prior
        V_new = np.array([
            logsumexp(log_terms[s]) / beta for s in range(n_states)
        ])

        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < tolerance:
            logger.debug("Soft value iteration converged in %d iterations", it)
            break

    # Extract policy
    Q_final = reward + gamma * np.einsum("san,n->sa", transition, V)
    policy = compute_softmax_policy(Q_final, beta, prior_policy)

    return V, policy, it


def compute_policy_gradient(
    policy: np.ndarray,
    reward: np.ndarray,
    prior: np.ndarray,
    beta: float,
) -> np.ndarray:
    r"""Gradient of the free energy w.r.t. policy parameters.

    For the softmax parameterisation π(a) ∝ exp(θ_a), the gradient of

    .. math::

        F(\theta) = \sum_a \pi_\theta(a)\,R(a)
                    - \frac{1}{\beta}\,D_{\mathrm{KL}}(\pi_\theta \| p_0)

    with respect to θ is:

    .. math::

        \nabla_\theta F = \pi \odot \bigl(R - \bar{R}
            - \frac{1}{\beta}(\ln\pi - \ln p_0 - \overline{\ln\pi - \ln p_0})\bigr)

    where :math:`\bar{\cdot}` denotes the expectation under π.

    Parameters
    ----------
    policy : np.ndarray
        Current policy π(a), shape ``(n_actions,)``.
    reward : np.ndarray
        Reward per action, shape ``(n_actions,)``.
    prior : np.ndarray
        Prior p₀(a), shape ``(n_actions,)``.
    beta : float
        Rationality parameter.

    Returns
    -------
    np.ndarray
        Gradient vector, shape ``(n_actions,)``.
    """
    policy = np.asarray(policy, dtype=np.float64).ravel()
    reward = np.asarray(reward, dtype=np.float64).ravel()
    prior = np.asarray(prior, dtype=np.float64).ravel()

    log_ratio = np.log(np.maximum(policy, _EPS)) - np.log(np.maximum(prior, _EPS))
    mean_reward = np.dot(policy, reward)
    mean_log_ratio = np.dot(policy, log_ratio)

    if beta > 0:
        advantage = reward - mean_reward - (1.0 / beta) * (log_ratio - mean_log_ratio)
    else:
        advantage = -(log_ratio - mean_log_ratio)

    return policy * advantage


def compute_optimal_beta(
    reward_distribution: np.ndarray,
    target_mutual_info: float,
    prior: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    r"""Estimate the rationality parameter β that achieves a target channel capacity.

    Finds β such that the mutual information under the softmax policy equals
    *target_mutual_info*:

    .. math::

        I_\beta = D_{\mathrm{KL}}(\pi_\beta \| p_0)
                = \text{target\_mutual\_info}

    Uses bisection search.

    Parameters
    ----------
    reward_distribution : np.ndarray
        Reward vector R(a), shape ``(n_actions,)``.
    target_mutual_info : float
        Target mutual information in nats.
    prior : np.ndarray, optional
        Prior distribution.  Uniform if ``None``.
    tolerance : float
        Convergence tolerance on |I(β) − target|.
    max_iterations : int
        Maximum bisection iterations.

    Returns
    -------
    float
        Estimated β value.
    """
    reward_distribution = np.asarray(reward_distribution, dtype=np.float64).ravel()
    n = reward_distribution.shape[0]

    if prior is None:
        prior = np.ones(n) / n

    if target_mutual_info <= 0:
        return 0.0

    max_mi = np.log(n)
    if target_mutual_info >= max_mi:
        logger.warning(
            "Target MI (%.4f) exceeds max entropy (%.4f); returning large β",
            target_mutual_info, max_mi,
        )
        return 1e6

    # Bisection on beta
    beta_lo, beta_hi = 0.0, 1.0

    # Find upper bound: double beta_hi until MI exceeds target
    for _ in range(50):
        pi = compute_softmax_policy(reward_distribution, beta_hi, prior)
        mi = compute_kl_divergence(pi, prior, validate=True)
        if mi >= target_mutual_info:
            break
        beta_hi *= 2.0
    else:
        return beta_hi

    # Bisection
    for _ in range(max_iterations):
        beta_mid = 0.5 * (beta_lo + beta_hi)
        pi = compute_softmax_policy(reward_distribution, beta_mid, prior)
        mi = compute_kl_divergence(pi, prior, validate=True)

        if abs(mi - target_mutual_info) < tolerance:
            return beta_mid

        if mi < target_mutual_info:
            beta_lo = beta_mid
        else:
            beta_hi = beta_mid

    return 0.5 * (beta_lo + beta_hi)


# ═══════════════════════════════════════════════════════════════════════════
# FreeEnergyComputer class
# ═══════════════════════════════════════════════════════════════════════════

class FreeEnergyComputer:
    r"""Engine for computing variational free energy and bounded-rational policies.

    Implements the objective:

    .. math::

        F(\pi) = \mathbb{E}_\pi[C] - \frac{1}{\beta}\,
                 D_{\mathrm{KL}}(\pi \| \pi_0)

    where C is the cost (negative reward), β is the rationality parameter,
    and π₀ is the reference (prior) policy.

    Supports multi-subsystem β parameters for different cognitive channels
    (e.g., β_motor for Fitts' law, β_choice for Hick's law).

    Parameters
    ----------
    config : VariationalConfig
        Solver configuration.
    subsystem_betas : dict[str, float], optional
        Per-subsystem rationality parameters.  If provided, the overall β
        in *config* is overridden for each named subsystem.
    """

    def __init__(
        self,
        config: VariationalConfig,
        subsystem_betas: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config = config
        self.subsystem_betas = subsystem_betas or {}
        self._rng = np.random.default_rng(config.seed)

    def compute(
        self,
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
    ) -> FreeEnergyResult:
        """Compute the optimal bounded-rational policy via Blahut–Arimoto iteration.

        Parameters
        ----------
        cost_matrix : dict[str, dict[str, float]]
            Mapping  state → {action → immediate cost}.
        reference_policy : dict[str, dict[str, float]]
            Prior policy  π₀(a|s).

        Returns
        -------
        FreeEnergyResult
        """
        t0 = time.monotonic()
        beta = self.config.beta
        states = sorted(cost_matrix.keys())

        if not states:
            return self._empty_result(t0)

        # Build arrays per state
        all_actions: Dict[str, List[str]] = {}
        for s in states:
            all_actions[s] = sorted(cost_matrix[s].keys())

        # Initialise policy: start from reference
        policy: Dict[str, Dict[str, float]] = {}
        for s in states:
            actions = all_actions[s]
            n_a = len(actions)
            if n_a == 0:
                policy[s] = {}
                continue
            ref_s = reference_policy.get(s, {})
            probs = np.array([ref_s.get(a, 1.0 / n_a) for a in actions])
            probs = np.maximum(probs, _EPS)
            probs /= probs.sum()
            policy[s] = {a: float(probs[i]) for i, a in enumerate(actions)}

        obj_trace: List[float] = []
        grad_trace: List[float] = []

        for it in range(1, self.config.max_iterations + 1):
            new_policy: Dict[str, Dict[str, float]] = {}

            for s in states:
                actions = all_actions[s]
                n_a = len(actions)
                if n_a == 0:
                    new_policy[s] = {}
                    continue

                costs = np.array([cost_matrix[s][a] for a in actions])
                ref_probs = np.array([
                    reference_policy.get(s, {}).get(a, 1.0 / n_a)
                    for a in actions
                ])
                ref_probs = np.maximum(ref_probs, _EPS)
                ref_probs /= ref_probs.sum()

                # Blahut–Arimoto update: π(a|s) ∝ p₀(a|s) exp(-β C(s,a))
                log_pi = np.log(ref_probs) - beta * costs
                log_pi -= logsumexp(log_pi)
                probs = np.exp(log_pi)
                new_policy[s] = {a: float(probs[i]) for i, a in enumerate(actions)}

            # Compute objective
            obj = self._evaluate_objective(new_policy, cost_matrix, reference_policy, beta)
            obj_trace.append(obj)

            # Gradient norm (approximate via policy change)
            g_norm = self._policy_distance(policy, new_policy, states, all_actions)
            grad_trace.append(g_norm)

            policy = new_policy

            if len(obj_trace) >= 2:
                prev = obj_trace[-2]
                if abs(prev) > 0:
                    rel_change = abs(obj - prev) / abs(prev)
                else:
                    rel_change = abs(obj - prev)
                if rel_change < self.config.tolerance:
                    elapsed = time.monotonic() - t0
                    return self._build_result(
                        policy, cost_matrix, reference_policy, beta,
                        obj_trace, grad_trace,
                        ConvergenceStatus.CONVERGED, rel_change, elapsed,
                    )

            if not np.isfinite(obj):
                elapsed = time.monotonic() - t0
                return self._build_result(
                    policy, cost_matrix, reference_policy, beta,
                    obj_trace, grad_trace,
                    ConvergenceStatus.DIVERGED, float("inf"), elapsed,
                )

        # Max iterations reached
        elapsed = time.monotonic() - t0
        rel_change = 0.0
        if len(obj_trace) >= 2 and abs(obj_trace[-2]) > 0:
            rel_change = abs(obj_trace[-1] - obj_trace[-2]) / abs(obj_trace[-2])
        return self._build_result(
            policy, cost_matrix, reference_policy, beta,
            obj_trace, grad_trace,
            ConvergenceStatus.MAX_ITERATIONS, rel_change, elapsed,
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _evaluate_objective(
        self,
        policy: Dict[str, Dict[str, float]],
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        beta: float,
    ) -> float:
        """Evaluate F(π) = E_π[C] − (1/β) D_KL(π ‖ π₀)."""
        states = sorted(policy.keys())
        n_states = len(states)
        if n_states == 0:
            return 0.0

        expected_cost = 0.0
        total_kl = 0.0

        for s in states:
            actions = sorted(policy[s].keys())
            if not actions:
                continue
            n_a = len(actions)
            p = np.array([policy[s][a] for a in actions])
            c = np.array([cost_matrix[s][a] for a in actions])
            ref = np.array([
                reference_policy.get(s, {}).get(a, 1.0 / n_a)
                for a in actions
            ])
            ref = np.maximum(ref, _EPS)
            ref /= ref.sum()

            expected_cost += np.dot(p, c) / n_states
            total_kl += compute_kl_divergence(p, ref, validate=False) / n_states

        if beta > 0:
            return expected_cost - (1.0 / beta) * total_kl
        return -total_kl

    def _policy_distance(
        self,
        old: Dict[str, Dict[str, float]],
        new: Dict[str, Dict[str, float]],
        states: List[str],
        all_actions: Dict[str, List[str]],
    ) -> float:
        """L2 distance between two policies (approximate gradient norm)."""
        total = 0.0
        for s in states:
            for a in all_actions[s]:
                diff = new[s].get(a, 0.0) - old[s].get(a, 0.0)
                total += diff * diff
        return math.sqrt(total)

    def _build_result(
        self,
        policy: Dict[str, Dict[str, float]],
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        beta: float,
        obj_trace: List[float],
        grad_trace: List[float],
        status: ConvergenceStatus,
        rel_change: float,
        elapsed: float,
    ) -> FreeEnergyResult:
        kl_result = compute_policy_kl(policy, reference_policy)

        # Expected cost
        states = sorted(policy.keys())
        exp_cost = 0.0
        n_s = len(states)
        for s in states:
            actions = sorted(policy[s].keys())
            if actions:
                p = np.array([policy[s][a] for a in actions])
                c = np.array([cost_matrix[s][a] for a in actions])
                exp_cost += np.dot(p, c) / n_s

        # Entropy
        entropy = 0.0
        for s in states:
            actions = sorted(policy[s].keys())
            if actions:
                p = np.array([policy[s][a] for a in actions])
                mask = p > 0
                entropy -= np.sum(np.where(mask, p * np.log(np.maximum(p, _EPS)), 0.0)) / n_s

        fe = exp_cost - (1.0 / beta) * kl_result.total_kl if beta > 0 else -kl_result.total_kl

        convergence = ConvergenceInfo(
            status=status,
            iterations_used=len(obj_trace),
            objective_trace=tuple(obj_trace),
            gradient_norm_trace=tuple(grad_trace),
            relative_change=rel_change,
            wall_clock_seconds=elapsed,
        )

        return FreeEnergyResult(
            free_energy=fe,
            expected_cost=exp_cost,
            entropy=entropy,
            kl_divergence=kl_result,
            policy=policy,
            convergence=convergence,
            config=self.config,
        )

    def _empty_result(self, t0: float) -> FreeEnergyResult:
        elapsed = time.monotonic() - t0
        return FreeEnergyResult(
            free_energy=0.0,
            expected_cost=0.0,
            entropy=0.0,
            kl_divergence=KLDivergenceResult(
                total_kl=0.0, per_state_kl={}, max_state_kl=0.0,
                max_state_id="", is_finite=True,
            ),
            policy={},
            convergence=ConvergenceInfo(
                status=ConvergenceStatus.CONVERGED,
                iterations_used=0,
                objective_trace=(),
                gradient_norm_trace=(),
                relative_change=0.0,
                wall_clock_seconds=elapsed,
            ),
            config=self.config,
        )
