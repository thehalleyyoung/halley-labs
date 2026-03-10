"""
usability_oracle.policy.gradient — Policy gradient methods for bounded rationality.

Implements policy gradient algorithms that directly optimise the bounded-
rational free-energy objective:

    F(π_θ) = E_π[cost] + (1/β) · D_KL(π_θ ‖ p₀)

The gradient of this objective is:

    ∇_θ F(π_θ) = E_π[∇_θ log π_θ(a|s) · (Q^π(s,a) + (1/β)(log π_θ(a|s) − log p₀(a|s)))]

Methods
-------
- **REINFORCE with baseline** — Monte-Carlo policy gradient with a
  learned value-function baseline for variance reduction.
- **Natural policy gradient** — uses the Fisher information matrix to
  obtain updates in the natural gradient direction, yielding faster
  convergence in the policy manifold.
- **Softmax policy parameterisation** — θ maps to log-linear action
  preferences; the gradient has a clean closed form.
- **Variance reduction** — control variates, advantage normalisation.
- **Compatible function approximation** — ensures the critic is compatible
  with the natural gradient.
- **Application** — learning a cognitive policy from observed interaction
  traces.

References
----------
- Williams, R. J. (1992). Simple statistical gradient-following algorithms
  for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.
- Kakade, S. M. (2001). A natural policy gradient. *NeurIPS*.
- Sutton, R. S. et al. (1999). Policy gradient methods for reinforcement
  learning with function approximation. *NeurIPS*.
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
- Peters, J. & Schaal, S. (2008). Natural actor-critic. *Neurocomputing*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.policy.models import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory data structure for policy gradient
# ---------------------------------------------------------------------------

@dataclass
class GradientTrajectory:
    """A trajectory of (state, action, cost) tuples for policy gradient.

    Attributes
    ----------
    states : list[str]
    actions : list[str]
    costs : list[float]
        Per-step costs (lower is better).
    """

    states: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.states)


# ---------------------------------------------------------------------------
# Softmax policy parameterisation
# ---------------------------------------------------------------------------

class SoftmaxPolicyParam:
    """Softmax (log-linear) policy parameterised by θ.

    π_θ(a|s) = exp(θ[s,a]) / Σ_{a'} exp(θ[s,a'])

    where θ[s,a] = features(s,a)^T · w  (or a tabular lookup).

    Parameters
    ----------
    states : list[str]
    actions_per_state : dict[str, list[str]]
    beta : float
        Rationality parameter for the information cost term.
    prior : Policy or None
        Prior policy p₀.
    """

    def __init__(
        self,
        states: list[str],
        actions_per_state: dict[str, list[str]],
        beta: float = 1.0,
        prior: Optional[Policy] = None,
    ) -> None:
        self.states = states
        self.actions_per_state = actions_per_state
        self.beta = beta
        self.prior = prior

        # Tabular parameters θ[s][a]
        self.theta: dict[str, dict[str, float]] = {}
        for s in states:
            self.theta[s] = {a: 0.0 for a in actions_per_state.get(s, [])}

    def action_probs(self, state: str) -> dict[str, float]:
        """Compute π_θ(·|s) via softmax over θ[s,·].

        Returns
        -------
        dict[str, float]
        """
        theta_s = self.theta.get(state, {})
        if not theta_s:
            return {}

        actions = list(theta_s.keys())
        logits = np.array([theta_s[a] for a in actions], dtype=np.float64)

        # Numerically stable softmax
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        return {actions[i]: float(probs[i]) for i in range(len(actions))}

    def log_prob(self, state: str, action: str) -> float:
        """Log probability log π_θ(a|s)."""
        probs = self.action_probs(state)
        p = probs.get(action, 1e-10)
        return math.log(max(p, 1e-10))

    def log_prob_gradient(self, state: str, action: str) -> dict[str, dict[str, float]]:
        """Gradient of log π_θ(a|s) w.r.t. θ.

        For softmax: ∇_θ[s,a'] log π(a|s) = 1(a'=a) − π(a'|s)

        Only non-zero entries for the given state.

        Returns
        -------
        dict[str, dict[str, float]]
            Sparse gradient: ``{state: {action: value}}``.
        """
        probs = self.action_probs(state)
        grad: dict[str, float] = {}
        for a in probs:
            grad[a] = (1.0 if a == action else 0.0) - probs[a]
        return {state: grad}

    def to_policy(self) -> Policy:
        """Convert current parameters to a :class:`Policy` object."""
        state_action_probs: dict[str, dict[str, float]] = {}
        for s in self.states:
            probs = self.action_probs(s)
            if probs:
                state_action_probs[s] = probs
        return Policy(state_action_probs=state_action_probs, beta=self.beta)

    def information_cost(self, state: str, action: str) -> float:
        """Per-action information cost: log π_θ(a|s) − log p₀(a|s).

        Returns
        -------
        float
        """
        log_pi = self.log_prob(state, action)
        if self.prior is None:
            n = len(self.theta.get(state, {}))
            log_p0 = -math.log(max(n, 1))
        else:
            p0 = self.prior.action_probability(state, action)
            log_p0 = math.log(max(p0, 1e-10))
        return log_pi - log_p0


# ---------------------------------------------------------------------------
# REINFORCE with baseline
# ---------------------------------------------------------------------------

class REINFORCE:
    """REINFORCE policy gradient with a learned baseline.

    Optimises the bounded-rational objective:

        F(π_θ) = E_π[cost] + (1/β) D_KL(π_θ ‖ p₀)

    using the gradient:

        ∇_θ F ≈ (1/N) Σ_n Σ_t ∇_θ log π_θ(a_t|s_t) · Â_t

    where Â_t = G_t − b(s_t) + (1/β)(log π_θ(a_t|s_t) − log p₀(a_t|s_t))
    and b(s) is a learned state-dependent baseline.

    Parameters
    ----------
    policy_param : SoftmaxPolicyParam
    learning_rate : float
    baseline_lr : float
        Learning rate for the baseline (value function).
    discount : float
    normalise_advantages : bool
        If True, normalise advantages to zero mean / unit variance.
    """

    def __init__(
        self,
        policy_param: SoftmaxPolicyParam,
        learning_rate: float = 0.01,
        baseline_lr: float = 0.01,
        discount: float = 0.99,
        normalise_advantages: bool = True,
    ) -> None:
        self.policy = policy_param
        self.lr = learning_rate
        self.baseline_lr = baseline_lr
        self.discount = discount
        self.normalise_advantages = normalise_advantages

        # Tabular baseline b(s) ≈ V(s)
        self.baseline: dict[str, float] = {
            s: 0.0 for s in policy_param.states
        }

    def update(self, trajectories: list[GradientTrajectory]) -> dict[str, float]:
        """Perform one policy gradient update from a batch of trajectories.

        Parameters
        ----------
        trajectories : list[GradientTrajectory]

        Returns
        -------
        dict[str, float]
            Training metrics: ``mean_cost``, ``mean_kl``, ``grad_norm``.
        """
        all_advantages: list[float] = []
        all_grads: list[dict[str, dict[str, float]]] = []

        total_cost = 0.0
        total_kl = 0.0
        n_steps = 0

        for traj in trajectories:
            returns = self._compute_returns(traj.costs)

            for t in range(traj.length):
                s, a = traj.states[t], traj.actions[t]
                g_t = returns[t]

                # Bounded-rational advantage
                info_cost = self.policy.information_cost(s, a)
                advantage = g_t - self.baseline.get(s, 0.0) + info_cost / max(self.policy.beta, 1e-10)

                all_advantages.append(advantage)
                all_grads.append(self.policy.log_prob_gradient(s, a))

                # Update baseline toward return
                b = self.baseline.get(s, 0.0)
                self.baseline[s] = b + self.baseline_lr * (g_t - b)

                total_cost += traj.costs[t]
                total_kl += info_cost
                n_steps += 1

        # Normalise advantages
        if self.normalise_advantages and len(all_advantages) > 1:
            adv_arr = np.array(all_advantages, dtype=np.float64)
            mean_adv = float(np.mean(adv_arr))
            std_adv = float(np.std(adv_arr, ddof=1))
            if std_adv > 1e-8:
                all_advantages = [(a - mean_adv) / std_adv for a in all_advantages]

        # Accumulate gradient
        grad_accum: dict[str, dict[str, float]] = {}
        for advantage, grad in zip(all_advantages, all_grads):
            for s, action_grads in grad.items():
                if s not in grad_accum:
                    grad_accum[s] = {}
                for a, g in action_grads.items():
                    grad_accum[s][a] = grad_accum[s].get(a, 0.0) + advantage * g

        # Apply gradient update (gradient descent on free energy)
        grad_norm = 0.0
        n = max(len(trajectories), 1)
        for s in grad_accum:
            for a in grad_accum[s]:
                g = grad_accum[s][a] / n
                grad_norm += g * g
                self.policy.theta[s][a] -= self.lr * g

        return {
            "mean_cost": total_cost / max(n_steps, 1),
            "mean_kl": total_kl / max(n_steps, 1),
            "grad_norm": math.sqrt(grad_norm),
        }

    def _compute_returns(self, costs: list[float]) -> list[float]:
        """Discounted returns G_t = Σ_{k=0}^{T-t} γ^k c_{t+k}."""
        n = len(costs)
        returns = [0.0] * n
        g = 0.0
        for t in range(n - 1, -1, -1):
            g = costs[t] + self.discount * g
            returns[t] = g
        return returns

    def get_policy(self) -> Policy:
        return self.policy.to_policy()


# ---------------------------------------------------------------------------
# Natural Policy Gradient
# ---------------------------------------------------------------------------

class NaturalPolicyGradient:
    """Natural policy gradient for the bounded-rational objective.

    Uses the Fisher information matrix F_θ to transform the vanilla
    gradient into the natural gradient:

        θ ← θ − α · F_θ^{-1} ∇_θ F(π_θ)

    The Fisher is estimated from the same trajectory batch.  For the
    softmax parameterisation, F_θ(s) = diag(π) − π·π^T evaluated per state.

    Parameters
    ----------
    policy_param : SoftmaxPolicyParam
    learning_rate : float
    discount : float
    fisher_damping : float
        Tikhonov damping for Fisher inversion: (F + λI)^{-1}.
    """

    def __init__(
        self,
        policy_param: SoftmaxPolicyParam,
        learning_rate: float = 0.01,
        discount: float = 0.99,
        fisher_damping: float = 1e-3,
    ) -> None:
        self.policy = policy_param
        self.lr = learning_rate
        self.discount = discount
        self.damping = fisher_damping

        self.baseline: dict[str, float] = {s: 0.0 for s in policy_param.states}

    def update(self, trajectories: list[GradientTrajectory]) -> dict[str, float]:
        """Natural policy gradient update.

        Parameters
        ----------
        trajectories : list[GradientTrajectory]

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Collect per-state gradient and Fisher contributions
        state_grads: dict[str, np.ndarray] = {}
        state_fishers: dict[str, np.ndarray] = {}
        state_actions_list: dict[str, list[str]] = {}

        total_cost = 0.0
        n_steps = 0

        for traj in trajectories:
            returns = self._compute_returns(traj.costs)
            for t in range(traj.length):
                s, a = traj.states[t], traj.actions[t]
                g_t = returns[t]
                info_cost = self.policy.information_cost(s, a)
                advantage = g_t - self.baseline.get(s, 0.0) + info_cost / max(self.policy.beta, 1e-10)

                self.baseline[s] = self.baseline.get(s, 0.0) + 0.01 * (g_t - self.baseline.get(s, 0.0))

                probs = self.policy.action_probs(s)
                actions = list(probs.keys())
                if s not in state_actions_list:
                    state_actions_list[s] = actions
                    n_a = len(actions)
                    state_grads[s] = np.zeros(n_a, dtype=np.float64)
                    state_fishers[s] = np.zeros((n_a, n_a), dtype=np.float64)

                action_idx = actions.index(a) if a in actions else 0
                pi = np.array([probs.get(ac, 0.0) for ac in actions], dtype=np.float64)

                # ∇ log π = e_a - π
                score = -pi.copy()
                score[action_idx] += 1.0

                state_grads[s] += advantage * score

                # Fisher: E[score · score^T] ≈ sample outer product
                state_fishers[s] += np.outer(score, score)

                total_cost += traj.costs[t]
                n_steps += 1

        # Apply natural gradient per state
        grad_norm = 0.0
        n = max(len(trajectories), 1)
        for s in state_grads:
            actions = state_actions_list[s]
            g = state_grads[s] / n
            F = state_fishers[s] / n

            # Damped Fisher inverse
            F_damped = F + self.damping * np.eye(len(actions))
            try:
                nat_grad = np.linalg.solve(F_damped, g)
            except np.linalg.LinAlgError:
                nat_grad = g

            grad_norm += float(np.dot(nat_grad, nat_grad))

            for i, a in enumerate(actions):
                self.policy.theta[s][a] -= self.lr * float(nat_grad[i])

        return {
            "mean_cost": total_cost / max(n_steps, 1),
            "grad_norm": math.sqrt(grad_norm),
        }

    def _compute_returns(self, costs: list[float]) -> list[float]:
        n = len(costs)
        returns = [0.0] * n
        g = 0.0
        for t in range(n - 1, -1, -1):
            g = costs[t] + self.discount * g
            returns[t] = g
        return returns

    def get_policy(self) -> Policy:
        return self.policy.to_policy()


# ---------------------------------------------------------------------------
# Compatible Function Approximation
# ---------------------------------------------------------------------------

class CompatibleCritic:
    """Compatible value function approximation for natural policy gradients.

    The compatible critic w satisfies:

        f_w(s, a) = ∇_θ log π_θ(a|s)^T · w

    ensuring that the policy gradient computed with this critic is exact
    (no approximation error bias).  The weights w are learned by minimising
    the TD error projected onto the score function space.

    Parameters
    ----------
    policy_param : SoftmaxPolicyParam
    learning_rate : float
    """

    def __init__(
        self,
        policy_param: SoftmaxPolicyParam,
        learning_rate: float = 0.01,
    ) -> None:
        self.policy = policy_param
        self.lr = learning_rate

        # Critic weights per state: w[s] is a vector over actions
        self.w: dict[str, np.ndarray] = {}
        for s in policy_param.states:
            n_a = len(policy_param.actions_per_state.get(s, []))
            self.w[s] = np.zeros(n_a, dtype=np.float64)

    def predict(self, state: str, action: str) -> float:
        """Predict the advantage f_w(s, a) = score(s,a)^T · w."""
        probs = self.policy.action_probs(state)
        actions = list(probs.keys())
        if action not in actions:
            return 0.0
        idx = actions.index(action)
        pi = np.array([probs.get(a, 0.0) for a in actions], dtype=np.float64)

        score = -pi.copy()
        score[idx] += 1.0

        w = self.w.get(state, np.zeros(len(actions)))
        return float(np.dot(score, w))

    def update(self, state: str, action: str, td_error: float) -> None:
        """Update critic weights toward the TD error.

        w ← w + α · δ · score(s, a)
        """
        probs = self.policy.action_probs(state)
        actions = list(probs.keys())
        if action not in actions:
            return
        idx = actions.index(action)
        pi = np.array([probs.get(a, 0.0) for a in actions], dtype=np.float64)

        score = -pi.copy()
        score[idx] += 1.0

        w = self.w.get(state, np.zeros(len(actions)))
        self.w[state] = w + self.lr * td_error * score


# ---------------------------------------------------------------------------
# Learning cognitive policy from interaction traces
# ---------------------------------------------------------------------------

def learn_cognitive_policy(
    traces: list[GradientTrajectory],
    states: list[str],
    actions_per_state: dict[str, list[str]],
    beta: float = 1.0,
    prior: Optional[Policy] = None,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    use_natural_gradient: bool = False,
) -> Policy:
    """Learn a bounded-rational cognitive policy from observed interaction traces.

    Fits a softmax policy to the observed trajectories by maximising the
    bounded-rational log-likelihood:

        L(θ) = Σ_n Σ_t [log π_θ(a_t|s_t) − (1/β) D_KL(π_θ(·|s_t) ‖ p₀(·|s_t))]

    Parameters
    ----------
    traces : list[GradientTrajectory]
        Observed interaction trajectories.
    states : list[str]
    actions_per_state : dict[str, list[str]]
    beta : float
    prior : Policy, optional
    n_epochs : int
    learning_rate : float
    use_natural_gradient : bool

    Returns
    -------
    Policy
    """
    param = SoftmaxPolicyParam(states, actions_per_state, beta, prior)

    if use_natural_gradient:
        optimiser: Any = NaturalPolicyGradient(
            param, learning_rate=learning_rate
        )
    else:
        optimiser = REINFORCE(
            param, learning_rate=learning_rate
        )

    for epoch in range(n_epochs):
        metrics = optimiser.update(traces)
        if epoch % 20 == 0:
            logger.debug(
                "Epoch %d: cost=%.4f, grad_norm=%.4f",
                epoch,
                metrics.get("mean_cost", 0.0),
                metrics.get("grad_norm", 0.0),
            )

    return optimiser.get_policy()
