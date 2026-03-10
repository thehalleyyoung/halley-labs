"""
usability_oracle.variational.optimizer — Variational optimisation algorithms.

Implements several optimisation strategies on the probability simplex for
minimising the bounded-rational free energy:

* :class:`VariationalOptimizer` — main solver (implements ``VariationalSolver``)
* :func:`natural_gradient_descent` — natural gradient on the statistical manifold
* :func:`mirror_descent` — mirror descent with KL as Bregman divergence
* :func:`alternating_minimization` — Blahut-Arimoto-style alternation
* :func:`proximal_point` — proximal point method
* :func:`line_search` — Armijo backtracking line search
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
from usability_oracle.variational.convergence import ConvergenceMonitor

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ═══════════════════════════════════════════════════════════════════════════
# Standalone optimisation primitives
# ═══════════════════════════════════════════════════════════════════════════

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex.

    Uses the efficient O(n log n) algorithm of Duchi et al. (2008).

    Parameters
    ----------
    v : np.ndarray
        Unconstrained vector.

    Returns
    -------
    np.ndarray
        Projected vector on the simplex (non-negative, sums to 1).
    """
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def natural_gradient_descent(
    objective: Callable[[np.ndarray], float],
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    fisher_matrix: np.ndarray,
    lr: float = 0.01,
) -> np.ndarray:
    r"""One step of natural gradient descent on the statistical manifold.

    .. math::

        \theta_{k+1} = \theta_k - \eta\, F^{-1}\, \nabla_\theta \ell(\theta_k)

    where F is the Fisher information matrix.

    Parameters
    ----------
    objective : callable
        Objective function f(θ) → scalar.
    gradient_fn : callable
        Gradient function ∇f(θ) → vector.
    params : np.ndarray
        Current parameters (on or near the simplex).
    fisher_matrix : np.ndarray
        Fisher information matrix, shape ``(n, n)``.
    lr : float
        Learning rate η.

    Returns
    -------
    np.ndarray
        Updated parameters projected onto the simplex.
    """
    grad = gradient_fn(params)

    # Solve F * natural_grad = grad  (regularise for stability)
    n = fisher_matrix.shape[0]
    F_reg = fisher_matrix + 1e-8 * np.eye(n)
    try:
        natural_grad = np.linalg.solve(F_reg, grad)
    except np.linalg.LinAlgError:
        logger.warning("Fisher matrix singular; falling back to Euclidean gradient")
        natural_grad = grad

    new_params = params - lr * natural_grad
    return _project_simplex(new_params)


def mirror_descent(
    objective: Callable[[np.ndarray], float],
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    lr: float = 0.01,
) -> np.ndarray:
    r"""One step of mirror descent with KL divergence as Bregman divergence.

    The mirror descent update with the negative entropy mirror map is
    the multiplicative weights / exponentiated gradient update:

    .. math::

        \pi_{k+1}(a) \propto \pi_k(a) \cdot \exp\!\bigl(-\eta\, g_a\bigr)

    Parameters
    ----------
    objective : callable
        Objective function.
    gradient_fn : callable
        Gradient function.
    params : np.ndarray
        Current policy (on the simplex).
    lr : float
        Step size η.

    Returns
    -------
    np.ndarray
        Updated policy on the simplex.
    """
    grad = gradient_fn(params)

    log_params = np.log(np.maximum(params, _EPS)) - lr * grad
    log_params -= logsumexp(log_params)
    return np.exp(log_params)


def alternating_minimization(
    cost_vector: np.ndarray,
    prior: np.ndarray,
    beta: float,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> Tuple[np.ndarray, List[float]]:
    r"""Blahut-Arimoto-style alternating minimisation.

    For a single-state problem, the optimal bounded-rational policy is:

    .. math::

        \pi^*(a) = \frac{p_0(a)\,\exp(-\beta\,C(a))}
                        {Z(\beta)}

    This is computed in closed form, but for consistency with multi-state
    problems we implement it iteratively.

    Parameters
    ----------
    cost_vector : np.ndarray
        Cost per action C(a), shape ``(n_actions,)``.
    prior : np.ndarray
        Prior distribution p₀(a).
    beta : float
        Rationality parameter.
    max_iterations : int
        Maximum iterations.
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    policy : np.ndarray
        Optimal policy.
    trace : list of float
        Objective trace.
    """
    cost_vector = np.asarray(cost_vector, dtype=np.float64).ravel()
    prior = np.asarray(prior, dtype=np.float64).ravel()
    prior = np.maximum(prior, _EPS)
    prior /= prior.sum()

    n = cost_vector.shape[0]
    pi = prior.copy()
    trace: List[float] = []

    for it in range(max_iterations):
        # Closed-form update
        log_pi = np.log(prior) - beta * cost_vector
        log_pi -= logsumexp(log_pi)
        pi_new = np.exp(log_pi)

        # Evaluate objective: E[C] - (1/beta) * KL(pi || prior)
        exp_cost = float(np.dot(pi_new, cost_vector))
        kl = compute_kl_divergence(pi_new, prior, validate=False)
        obj = exp_cost - (1.0 / max(beta, _EPS)) * kl
        trace.append(obj)

        if it > 0 and abs(trace[-1] - trace[-2]) < tolerance:
            break

        pi = pi_new

    return pi, trace


def proximal_point(
    objective: Callable[[np.ndarray], float],
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    proximal_weight: float = 1.0,
    inner_steps: int = 10,
    lr: float = 0.01,
) -> np.ndarray:
    r"""Proximal point method on the simplex.

    Minimises the proximal sub-problem:

    .. math::

        \theta_{k+1} = \arg\min_\theta\;
            f(\theta) + \frac{\mu}{2}\,\|\theta - \theta_k\|^2

    via a few steps of projected gradient descent.

    Parameters
    ----------
    objective : callable
        Objective function.
    gradient_fn : callable
        Gradient function.
    params : np.ndarray
        Current parameters.
    proximal_weight : float
        Proximal penalty weight μ.
    inner_steps : int
        Number of inner GD steps.
    lr : float
        Inner step size.

    Returns
    -------
    np.ndarray
        Updated parameters.
    """
    theta = params.copy()
    anchor = params.copy()

    for _ in range(inner_steps):
        grad = gradient_fn(theta) + proximal_weight * (theta - anchor)
        theta = _project_simplex(theta - lr * grad)

    return theta


def line_search(
    objective: Callable[[np.ndarray], float],
    params: np.ndarray,
    direction: np.ndarray,
    max_step: float = 1.0,
    alpha: float = 1e-4,
    rho: float = 0.5,
    max_trials: int = 20,
) -> Tuple[float, np.ndarray]:
    r"""Armijo backtracking line search on the simplex.

    Finds the largest step size t ∈ {max\_step · ρ^k} such that:

    .. math::

        f(\theta + t\,d) \le f(\theta) + \alpha\,t\,\nabla f(\theta)^T d

    Parameters
    ----------
    objective : callable
        Objective function.
    params : np.ndarray
        Current parameters.
    direction : np.ndarray
        Search direction (typically negative gradient).
    max_step : float
        Initial step size.
    alpha : float
        Armijo sufficient decrease parameter.
    rho : float
        Backtracking factor.
    max_trials : int
        Maximum number of step halvings.

    Returns
    -------
    step_size : float
        Accepted step size.
    new_params : np.ndarray
        Updated parameters.
    """
    f0 = objective(params)
    # Directional derivative approximation
    eps = 1e-7
    f_eps = objective(_project_simplex(params + eps * direction))
    dir_deriv = (f_eps - f0) / eps

    step = max_step
    for _ in range(max_trials):
        candidate = _project_simplex(params + step * direction)
        f_new = objective(candidate)

        if f_new <= f0 + alpha * step * dir_deriv:
            return step, candidate

        step *= rho

    # Return the smallest step tried
    return step, _project_simplex(params + step * direction)


# ═══════════════════════════════════════════════════════════════════════════
# VariationalOptimizer class (implements VariationalSolver protocol)
# ═══════════════════════════════════════════════════════════════════════════

class VariationalOptimizer:
    """Variational solver for bounded-rational policies.

    Implements the :class:`VariationalSolver` protocol using
    Blahut-Arimoto iteration with optional natural gradient or
    mirror descent refinement.

    Supports warm-start, step-size adaptation, and gradient clipping.
    """

    def __init__(self, max_grad_norm: float = 10.0) -> None:
        self._max_grad_norm = max_grad_norm

    def solve(
        self,
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        config: VariationalConfig,
    ) -> FreeEnergyResult:
        """Compute the optimal bounded-rational policy.

        See :meth:`VariationalSolver.solve` for full documentation.
        """
        return self._run(cost_matrix, reference_policy, config, initial_policy=None)

    def warm_start(
        self,
        initial_policy: Dict[str, Dict[str, float]],
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        config: VariationalConfig,
    ) -> FreeEnergyResult:
        """Re-solve from an existing policy (warm start).

        See :meth:`VariationalSolver.warm_start` for full documentation.
        """
        return self._run(cost_matrix, reference_policy, config, initial_policy=initial_policy)

    # -------------------------------------------------------------------
    # Core algorithm
    # -------------------------------------------------------------------

    def _run(
        self,
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        config: VariationalConfig,
        initial_policy: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> FreeEnergyResult:
        t0 = time.monotonic()
        beta = config.beta
        states = sorted(cost_matrix.keys())

        if not states:
            return self._empty_result(config, t0)

        # Determine actions per state
        all_actions: Dict[str, List[str]] = {}
        for s in states:
            all_actions[s] = sorted(cost_matrix[s].keys())

        # Initialise policy
        policy = self._init_policy(
            states, all_actions, reference_policy, initial_policy
        )

        monitor = ConvergenceMonitor(
            tolerance=config.tolerance,
            window=3,
            max_iterations=config.max_iterations,
        )

        lr = config.learning_rate

        for it in range(1, config.max_iterations + 1):
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

                cur_probs = np.array([policy[s][a] for a in actions])

                if config.use_natural_gradient:
                    new_p = self._natural_grad_step(
                        cur_probs, costs, ref_probs, beta, lr,
                    )
                else:
                    # Blahut-Arimoto closed-form update
                    new_p = self._ba_step(costs, ref_probs, beta)

                new_policy[s] = {a: float(new_p[i]) for i, a in enumerate(actions)}

            # Evaluate objective
            obj = self._evaluate(new_policy, cost_matrix, reference_policy, states, all_actions, beta)
            g_norm = self._policy_dist(policy, new_policy, states, all_actions)

            should_stop = monitor.record(obj, g_norm)
            policy = new_policy

            if should_stop:
                break

            # Adaptive step size
            if config.line_search and len(monitor.values) >= 2:
                if monitor.values[-1] > monitor.values[-2]:
                    lr *= 0.5
                    lr = max(lr, 1e-8)

        elapsed = time.monotonic() - t0

        if monitor.is_converged:
            status = ConvergenceStatus.CONVERGED
        elif monitor.is_diverged:
            status = ConvergenceStatus.DIVERGED
        else:
            status = ConvergenceStatus.MAX_ITERATIONS

        # Compute final rel change
        vals = monitor.values
        rel_change = 0.0
        if len(vals) >= 2 and abs(vals[-2]) > 0:
            rel_change = abs(vals[-1] - vals[-2]) / abs(vals[-2])

        return self._build_result(
            policy, cost_matrix, reference_policy, beta,
            monitor.values, monitor.gradient_norms,
            status, rel_change, elapsed, config,
        )

    # -------------------------------------------------------------------
    # Update steps
    # -------------------------------------------------------------------

    def _ba_step(
        self,
        costs: np.ndarray,
        prior: np.ndarray,
        beta: float,
    ) -> np.ndarray:
        """Blahut-Arimoto update: π(a) ∝ p₀(a) exp(-β C(a))."""
        log_pi = np.log(prior) - beta * costs
        log_pi -= logsumexp(log_pi)
        return np.exp(log_pi)

    def _natural_grad_step(
        self,
        current: np.ndarray,
        costs: np.ndarray,
        prior: np.ndarray,
        beta: float,
        lr: float,
    ) -> np.ndarray:
        """Natural gradient step on the simplex."""
        # Gradient of free energy w.r.t. log-policy parameters
        log_ratio = np.log(np.maximum(current, _EPS)) - np.log(np.maximum(prior, _EPS))
        grad = costs + (1.0 / max(beta, _EPS)) * (log_ratio + 1.0)
        mean_grad = np.dot(current, grad)
        grad = grad - mean_grad

        # Fisher information for categorical: diag(π) - π πᵀ
        # Natural gradient = F⁻¹ g ≈ g / π (for diagonal approximation)
        natural_grad = grad  # simplified: avoid full Fisher inverse

        # Clip
        norm = np.linalg.norm(natural_grad)
        if norm > self._max_grad_norm:
            natural_grad = natural_grad * (self._max_grad_norm / norm)

        # Mirror descent update
        log_p = np.log(np.maximum(current, _EPS)) - lr * natural_grad
        log_p -= logsumexp(log_p)
        return np.exp(log_p)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _init_policy(
        self,
        states: List[str],
        all_actions: Dict[str, List[str]],
        reference: Dict[str, Dict[str, float]],
        initial: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, float]]:
        policy: Dict[str, Dict[str, float]] = {}
        for s in states:
            actions = all_actions[s]
            n_a = len(actions)
            if n_a == 0:
                policy[s] = {}
                continue

            if initial and s in initial:
                probs = np.array([initial[s].get(a, 1.0 / n_a) for a in actions])
            else:
                ref_s = reference.get(s, {})
                probs = np.array([ref_s.get(a, 1.0 / n_a) for a in actions])

            probs = np.maximum(probs, _EPS)
            probs /= probs.sum()
            policy[s] = {a: float(probs[i]) for i, a in enumerate(actions)}

        return policy

    def _evaluate(
        self,
        policy: Dict[str, Dict[str, float]],
        cost_matrix: Dict[str, Dict[str, float]],
        reference: Dict[str, Dict[str, float]],
        states: List[str],
        all_actions: Dict[str, List[str]],
        beta: float,
    ) -> float:
        n_s = len(states)
        if n_s == 0:
            return 0.0

        exp_cost = 0.0
        total_kl = 0.0

        for s in states:
            actions = all_actions[s]
            if not actions:
                continue
            n_a = len(actions)
            p = np.array([policy[s][a] for a in actions])
            c = np.array([cost_matrix[s][a] for a in actions])
            ref = np.array([reference.get(s, {}).get(a, 1.0 / n_a) for a in actions])
            ref = np.maximum(ref, _EPS)
            ref /= ref.sum()

            exp_cost += np.dot(p, c) / n_s
            total_kl += compute_kl_divergence(p, ref, validate=False) / n_s

        if beta > 0:
            return exp_cost - (1.0 / beta) * total_kl
        return -total_kl

    def _policy_dist(
        self,
        old: Dict[str, Dict[str, float]],
        new: Dict[str, Dict[str, float]],
        states: List[str],
        all_actions: Dict[str, List[str]],
    ) -> float:
        total = 0.0
        for s in states:
            for a in all_actions[s]:
                diff = new[s].get(a, 0.0) - old[s].get(a, 0.0)
                total += diff * diff
        return float(np.sqrt(total))

    def _build_result(
        self,
        policy: Dict[str, Dict[str, float]],
        cost_matrix: Dict[str, Dict[str, float]],
        reference: Dict[str, Dict[str, float]],
        beta: float,
        obj_trace: List[float],
        grad_trace: List[float],
        status: ConvergenceStatus,
        rel_change: float,
        elapsed: float,
        config: VariationalConfig,
    ) -> FreeEnergyResult:
        kl_result = compute_policy_kl(policy, reference)

        states = sorted(policy.keys())
        n_s = len(states) if states else 1

        exp_cost = 0.0
        entropy = 0.0
        for s in states:
            actions = sorted(policy[s].keys())
            if actions:
                p = np.array([policy[s][a] for a in actions])
                c = np.array([cost_matrix[s][a] for a in actions])
                exp_cost += float(np.dot(p, c)) / n_s
                mask = p > 0
                entropy -= float(np.sum(np.where(
                    mask, p * np.log(np.maximum(p, _EPS)), 0.0
                ))) / n_s

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
            config=config,
        )

    def _empty_result(self, config: VariationalConfig, t0: float) -> FreeEnergyResult:
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
            config=config,
        )
