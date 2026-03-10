"""
usability_oracle.policy.models — Core data structures for policies and Q-values.

Defines :class:`Policy`, :class:`QValues`, and :class:`PolicyResult` used
throughout the bounded-rational policy computation pipeline.

A :class:`Policy` stores the full conditional distribution P(a|s) for every
state, supporting both deterministic and stochastic policies.

A bounded-rational policy is computed via free-energy minimisation:

    π_β(a|s) = p₀(a|s) · exp(β · Q(s,a)) / Z(s)

where β is the rationality parameter, Q is the action-value function,
p₀ is the prior policy, and Z(s) = Σ_a p₀(a|s) exp(β Q(s,a)).

References
----------
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning*, 2nd ed.
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """A stochastic policy mapping states to action distributions.

    ``state_action_probs[s][a]`` = P(a | s).

    For each state s, the action probabilities must sum to 1 (within
    numerical tolerance).

    Parameters
    ----------
    state_action_probs : dict[str, dict[str, float]]
        Mapping ``state_id -> {action_id: probability}``.
    beta : float
        Rationality parameter used to compute this policy.
    values : dict[str, float]
        State-value function V(s) (optional, populated after solving).
    q_values : dict[str, dict[str, float]]
        Action-value function Q(s, a) (optional).
    metadata : dict
        Arbitrary extra data.
    """

    state_action_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    beta: float = 1.0
    values: dict[str, float] = field(default_factory=dict)
    q_values: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Backward-compatible aliases ---------------------------------------

    def action_probs(self, state: str) -> dict[str, float]:
        """Return action probability distribution at *state*."""
        return self.state_action_probs.get(state, {})

    def prob(self, state: str, action: str) -> float:
        """Return π(action | state)."""
        return self.state_action_probs.get(state, {}).get(action, 0.0)

    # ── Core queries ------------------------------------------------------

    def action_probability(self, state: str, action: str) -> float:
        """Return P(action | state), defaulting to 0 if unknown."""
        return self.state_action_probs.get(state, {}).get(action, 0.0)

    def sample_action(
        self, state: str, rng: Optional[np.random.Generator] = None
    ) -> str:
        """Sample an action from P(·|state).

        Parameters
        ----------
        state : str
        rng : np.random.Generator, optional

        Returns
        -------
        str
            Sampled action ID.

        Raises
        ------
        ValueError
            If no actions are available for *state*.
        """
        dist = self.state_action_probs.get(state)
        if not dist:
            raise ValueError(f"No actions available for state {state!r}")

        if rng is not None:
            actions = list(dist.keys())
            probs = np.array([dist[a] for a in actions], dtype=np.float64)
            total = probs.sum()
            if total <= 0:
                return str(rng.choice(actions))
            probs /= total
            idx = rng.choice(len(actions), p=probs)
            return actions[idx]

        # Fallback to stdlib random
        actions = list(dist.keys())
        weights = [dist[a] for a in actions]
        return random.choices(actions, weights=weights, k=1)[0]

    def entropy(self, state: str) -> float:
        """Shannon entropy H(π(·|s)) = −Σ_a π(a|s) log π(a|s).

        Returns 0 for deterministic policies.

        Parameters
        ----------
        state : str

        Returns
        -------
        float
            Entropy in nats.
        """
        dist = self.state_action_probs.get(state, {})
        if not dist:
            return 0.0

        h = 0.0
        for p in dist.values():
            if p > 0:
                h -= p * math.log(p)
        return h

    def max_entropy(self, state: str) -> float:
        """Return log(|A(s)|) — the maximum entropy for this state."""
        n = len(self.action_probs(state))
        return math.log(n) if n > 0 else 0.0

    def expected_value(
        self, state: str, q_vals: dict[str, float]
    ) -> float:
        """Expected Q-value under the policy: E_{a~π(·|s)}[Q(s,a)].

        Parameters
        ----------
        state : str
        q_vals : dict[str, float]
            Mapping ``action_id -> Q(s, a)`` for the given state.

        Returns
        -------
        float
        """
        dist = self.state_action_probs.get(state, {})
        ev = 0.0
        for a, p in dist.items():
            ev += p * q_vals.get(a, 0.0)
        return ev

    def is_deterministic(self, state: str, threshold: float = 0.99) -> bool:
        """True if one action has probability ≥ threshold in *state*."""
        dist = self.state_action_probs.get(state, {})
        if not dist:
            return True
        return any(p >= threshold for p in dist.values())

    def kl_from_uniform(self, state: str) -> float:
        """D_KL(π(·|s) || U) where U is uniform over available actions."""
        probs = self.action_probs(state)
        n = len(probs)
        if n == 0:
            return 0.0
        uniform = 1.0 / n
        kl = 0.0
        for p in probs.values():
            if p > 0:
                kl += p * math.log(p / uniform)
        return kl

    def greedy_action(self, state: str) -> Optional[str]:
        """Return the most-probable action at *state*."""
        probs = self.action_probs(state)
        if not probs:
            return None
        return max(probs, key=probs.get)  # type: ignore[arg-type]

    def n_states(self) -> int:
        return len(self.state_action_probs)

    def mean_entropy(self) -> float:
        """Mean entropy across all states."""
        if not self.state_action_probs:
            return 0.0
        return sum(self.entropy(s) for s in self.state_action_probs) / len(
            self.state_action_probs
        )

    def __repr__(self) -> str:
        return f"Policy(states={self.n_states()}, β={self.beta:.2f}, mean_H={self.mean_entropy():.3f})"


# ---------------------------------------------------------------------------
# QValues
# ---------------------------------------------------------------------------

@dataclass
class QValues:
    """State-action value function Q(s, a).

    ``values[s][a]`` = Q(s, a) representing the expected cost-to-go
    (negative reward) from state s after taking action a.

    Parameters
    ----------
    values : dict[str, dict[str, float]]
    """

    values: dict[str, dict[str, float]] = field(default_factory=dict)

    def best_action(self, state: str) -> Optional[str]:
        """Return the action minimising Q(s, a) (lowest cost).

        Parameters
        ----------
        state : str

        Returns
        -------
        str or None
        """
        q_s = self.values.get(state, {})
        if not q_s:
            return None
        return min(q_s, key=q_s.get)  # type: ignore[arg-type]

    def value(self, state: str, action: str) -> float:
        """Return Q(state, action), defaulting to 0."""
        return self.values.get(state, {}).get(action, 0.0)

    def advantage(self, state: str, action: str) -> float:
        """Advantage A(s, a) = Q(s, a) − min_a' Q(s, a').

        Parameters
        ----------
        state : str
        action : str

        Returns
        -------
        float
        """
        q_s = self.values.get(state, {})
        if not q_s:
            return 0.0
        v_s = min(q_s.values())
        return self.value(state, action) - v_s

    def to_policy(self, beta: float) -> Policy:
        """Convert Q-values to a softmax (Boltzmann) policy.

        π_β(a|s) = exp(−β Q(s,a)) / Σ_{a'} exp(−β Q(s,a'))

        Note: we negate Q because Q represents costs (lower is better),
        and softmax maximises.

        Parameters
        ----------
        beta : float
            Rationality parameter (inverse temperature). β → ∞ gives
            the deterministic optimal policy; β → 0 gives uniform random.

        Returns
        -------
        Policy
        """
        state_action_probs: dict[str, dict[str, float]] = {}

        for state, q_s in self.values.items():
            if not q_s:
                continue
            actions = list(q_s.keys())
            q_arr = np.array([-beta * q_s[a] for a in actions], dtype=np.float64)

            # Numerically stable softmax via log-sum-exp
            q_max = np.max(q_arr)
            exp_q = np.exp(q_arr - q_max)
            probs = exp_q / exp_q.sum()

            state_action_probs[state] = {
                actions[i]: float(probs[i]) for i in range(len(actions))
            }

        return Policy(state_action_probs=state_action_probs, beta=beta)

    def state_value(self, state: str) -> float:
        """V(s) = min_a Q(s, a) — optimal value under deterministic policy."""
        q_s = self.values.get(state, {})
        return min(q_s.values()) if q_s else 0.0

    def n_states(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"QValues(states={self.n_states()})"


# ---------------------------------------------------------------------------
# PolicyResult
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    """Result of a policy computation algorithm.

    Bundles the computed policy, Q-values, state values, free energy,
    and convergence diagnostics.

    Attributes
    ----------
    policy : Policy
    q_values : QValues
    state_values : dict[str, float]
        V(s) for each state.
    free_energy : float
        Total free energy F(π) = E_π[cost] + (1/β) D_KL(π ‖ p₀).
    convergence_info : dict
        Algorithm-specific convergence diagnostics (iterations, residual, etc.).
    """

    policy: Policy
    q_values: QValues
    state_values: dict[str, float] = field(default_factory=dict)
    free_energy: float = 0.0
    convergence_info: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PolicyResult(F={self.free_energy:.4f}, "
            f"states={len(self.state_values)}, "
            f"converged={self.convergence_info.get('converged', '?')})"
        )
