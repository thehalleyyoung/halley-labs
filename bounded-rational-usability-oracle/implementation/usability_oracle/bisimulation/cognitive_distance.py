"""
usability_oracle.bisimulation.cognitive_distance — Cognitive distance metric.

Implements the bounded-rational cognitive distance:

    d_cog(s₁, s₂) = sup_{β' ≤ β}  d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))

where d_TV(p, q) = ½ Σᵢ |pᵢ - qᵢ| is the total-variation distance, and
π_{β'}(a|s) ∝ exp(−β' · c(s,a)) is the bounded-rational (Boltzmann) policy
at rationality level β'.

The distance d_cog captures *behavioural* distinguishability: if a bounded-
rational user with capacity up to β cannot tell two states apart (their
optimal action distributions are identical), the states are cognitively
equivalent.

The supremum is computed via grid search over [0, β_max] followed by
local refinement with scipy.optimize.minimize_scalar.

References
----------
- Ortega & Braun (2013). Thermodynamics as a theory of decision-making
  with information-processing costs. *Proc. R. Soc. A*.
- Givan, Dean & Greig (2003). Equivalence notions and model minimization
  in Markov decision processes. *Artificial Intelligence*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from usability_oracle.bisimulation.models import CognitiveDistanceMatrix
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft value iteration (needed to compute π_β)
# ---------------------------------------------------------------------------

def _soft_value_iteration(
    mdp: MDP,
    beta: float,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, float]:
    """Compute soft (free-energy) state values via value iteration.

    V(s) = (1/β) log Σ_a exp(β · Q(s, a))
    Q(s, a) = -E[cost(s,a)] + γ Σ_{s'} T(s'|s,a) V(s')

    For β → 0 this approaches uniform random; for β → ∞ it approaches
    the hard-max optimal value function.

    Parameters
    ----------
    mdp : MDP
    beta : float
        Rationality parameter (> 0).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on max |V_new - V_old|.

    Returns
    -------
    dict[str, float]
        State-value function V(s).
    """
    if beta <= 0:
        return {sid: 0.0 for sid in mdp.states}

    values = {sid: 0.0 for sid in mdp.states}
    gamma = mdp.discount

    for _ in range(max_iter):
        new_values: dict[str, float] = {}
        max_delta = 0.0

        for sid, state in mdp.states.items():
            if state.is_terminal or state.is_goal:
                new_values[sid] = 0.0
                continue

            actions = mdp.get_actions(sid)
            if not actions:
                new_values[sid] = 0.0
                continue

            q_values: list[float] = []
            for aid in actions:
                transitions = mdp.get_transitions(sid, aid)
                expected_cost = 0.0
                expected_future = 0.0
                for target, prob, cost in transitions:
                    expected_cost += prob * cost
                    expected_future += prob * values.get(target, 0.0)
                q = -expected_cost + gamma * expected_future
                q_values.append(q)

            # Log-sum-exp for numerical stability
            q_arr = np.array(q_values, dtype=np.float64)
            max_q = float(np.max(q_arr))
            scaled = beta * (q_arr - max_q)
            # Clamp to avoid overflow in exp
            scaled = np.clip(scaled, -500.0, 500.0)
            logsumexp = max_q + float(np.log(np.sum(np.exp(scaled)))) / beta
            new_values[sid] = logsumexp

            max_delta = max(max_delta, abs(new_values[sid] - values[sid]))

        values = new_values
        if max_delta < tol:
            break

    return values


# ---------------------------------------------------------------------------
# CognitiveDistanceComputer
# ---------------------------------------------------------------------------

@dataclass
class CognitiveDistanceComputer:
    """Compute the cognitive distance metric d_cog between MDP states.

    The metric measures behavioural distinguishability under bounded
    rationality, taking the supremum over all rationality levels up to β:

        d_cog(s₁, s₂) = sup_{β' ≤ β}  d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))

    Parameters
    ----------
    n_grid : int
        Number of grid points for the initial β-sweep (default 30).
    refine : bool
        Whether to perform local refinement with scipy after grid search.
    cache_values : bool
        Whether to cache soft-value-iteration results across β values.
    """

    n_grid: int = 30
    refine: bool = True
    cache_values: bool = True

    _value_cache: dict[float, dict[str, float]] = field(
        default_factory=dict, repr=False
    )

    # ── Public API --------------------------------------------------------

    def compute_distance(self, s1: str, s2: str, mdp: MDP, beta: float) -> float:
        """Compute d_cog(s1, s2) for a single pair of states.

        Parameters
        ----------
        s1, s2 : str
            State identifiers.
        mdp : MDP
            The underlying Markov Decision Process.
        beta : float
            Maximum rationality parameter.

        Returns
        -------
        float
            Cognitive distance in [0, 1].
        """
        return self._supremum_over_beta(mdp, s1, s2, beta)

    def compute_distance_matrix(
        self, mdp: MDP, beta: float
    ) -> CognitiveDistanceMatrix:
        """Compute the full pairwise cognitive-distance matrix.

        Parameters
        ----------
        mdp : MDP
        beta : float

        Returns
        -------
        CognitiveDistanceMatrix
            Symmetric matrix of pairwise d_cog values.
        """
        state_ids = sorted(mdp.states.keys())
        n = len(state_ids)
        distances = np.zeros((n, n), dtype=np.float64)

        # Pre-compute values for all β grid points
        self._value_cache.clear()
        beta_grid = np.linspace(0.01, beta, self.n_grid)
        for b in beta_grid:
            self._value_cache[float(b)] = _soft_value_iteration(mdp, float(b))

        logger.info(
            "Computing %d×%d distance matrix (β_max=%.2f, grid=%d)",
            n, n, beta, self.n_grid,
        )

        for i in range(n):
            for j in range(i + 1, n):
                d = self._supremum_over_beta(mdp, state_ids[i], state_ids[j], beta)
                distances[i, j] = d
                distances[j, i] = d

        self._value_cache.clear()
        return CognitiveDistanceMatrix(distances=distances, state_ids=state_ids)

    # ── Core computations -------------------------------------------------

    def _total_variation_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute the total-variation distance between two distributions.

            d_TV(p, q) = ½ Σᵢ |pᵢ − qᵢ|

        Parameters
        ----------
        p, q : np.ndarray
            Probability vectors (must sum to 1, same length).

        Returns
        -------
        float
            d_TV ∈ [0, 1].
        """
        return 0.5 * float(np.sum(np.abs(p - q)))

    def _policy_at_state(
        self,
        mdp: MDP,
        state: str,
        beta: float,
        values: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute the bounded-rational policy distribution π_β(·|s).

        π_β(a|s) = exp(β · Q(s, a)) / Σ_{a'} exp(β · Q(s, a'))

        Parameters
        ----------
        mdp : MDP
        state : str
        beta : float
        values : dict or None
            Pre-computed state values; computed if ``None``.

        Returns
        -------
        np.ndarray
            Action probability vector (ordered by ``mdp.get_actions(state)``).
        """
        if values is None:
            values = self._get_values(mdp, beta)

        q = self._q_values_at_state(mdp, state, values)
        if len(q) == 0:
            return np.array([])

        # Softmax with numerical stability
        scaled = beta * q
        scaled -= np.max(scaled)
        exp_q = np.exp(scaled)
        total = np.sum(exp_q)
        if total <= 0:
            return np.ones(len(q)) / len(q)
        return exp_q / total

    def _q_values_at_state(
        self,
        mdp: MDP,
        state: str,
        values: dict[str, float],
    ) -> np.ndarray:
        """Compute Q(s, a) for all available actions at *state*.

        Q(s, a) = −E[cost(s, a)] + γ Σ_{s'} T(s'|s, a) · V(s')

        Parameters
        ----------
        mdp : MDP
        state : str
        values : dict
            State-value function V.

        Returns
        -------
        np.ndarray
            Q-values for each available action.
        """
        actions = mdp.get_actions(state)
        if not actions:
            return np.array([])

        gamma = mdp.discount
        q_values = np.zeros(len(actions), dtype=np.float64)

        for idx, aid in enumerate(actions):
            transitions = mdp.get_transitions(state, aid)
            expected_cost = 0.0
            expected_future = 0.0
            for target, prob, cost in transitions:
                expected_cost += prob * cost
                expected_future += prob * values.get(target, 0.0)
            q_values[idx] = -expected_cost + gamma * expected_future

        return q_values

    def _supremum_over_beta(
        self,
        mdp: MDP,
        s1: str,
        s2: str,
        beta_max: float,
    ) -> float:
        """Compute sup_{β' ∈ [0, β_max]} d_TV(π_{β'}(·|s1), π_{β'}(·|s2)).

        First performs a coarse grid search over *n_grid* points, then
        optionally refines with ``scipy.optimize.minimize_scalar`` around
        the grid maximum.

        Parameters
        ----------
        mdp : MDP
        s1, s2 : str
        beta_max : float

        Returns
        -------
        float
            The supremal TV distance ∈ [0, 1].
        """
        # Both states must have the same action set for comparison
        actions_1 = mdp.get_actions(s1)
        actions_2 = mdp.get_actions(s2)
        if set(actions_1) != set(actions_2):
            # Different action availability → maximally distinguishable
            return 1.0

        if not actions_1:
            return 0.0

        # Canonical action ordering for both states
        action_order = sorted(set(actions_1))

        def tv_at_beta(b: float) -> float:
            """Negative TV distance at β = b (for minimisation)."""
            if b <= 1e-10:
                # β → 0: uniform policy regardless of state
                return 0.0
            values = self._get_values(mdp, b)
            p1 = self._policy_at_state_ordered(mdp, s1, b, values, action_order)
            p2 = self._policy_at_state_ordered(mdp, s2, b, values, action_order)
            return self._total_variation_distance(p1, p2)

        # Phase 1: coarse grid search
        beta_grid = np.linspace(0.01, beta_max, self.n_grid)
        tv_values = np.array([tv_at_beta(float(b)) for b in beta_grid])
        best_idx = int(np.argmax(tv_values))
        best_tv = float(tv_values[best_idx])

        # Phase 2: local refinement via scipy
        if self.refine and self.n_grid > 2:
            lo = float(beta_grid[max(0, best_idx - 1)])
            hi = float(beta_grid[min(len(beta_grid) - 1, best_idx + 1)])
            try:
                result = minimize_scalar(
                    lambda b: -tv_at_beta(b),
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-4, "maxiter": 50},
                )
                refined_tv = -result.fun
                if refined_tv > best_tv:
                    best_tv = refined_tv
            except Exception:
                # Fall back to grid result
                pass

        return min(best_tv, 1.0)

    # ── Helpers -----------------------------------------------------------

    def _policy_at_state_ordered(
        self,
        mdp: MDP,
        state: str,
        beta: float,
        values: dict[str, float],
        action_order: list[str],
    ) -> np.ndarray:
        """Compute π_β(·|s) with actions in a fixed canonical order.

        This ensures that the probability vectors for different states are
        aligned for TV-distance computation.
        """
        q = np.zeros(len(action_order), dtype=np.float64)
        gamma = mdp.discount
        for idx, aid in enumerate(action_order):
            transitions = mdp.get_transitions(state, aid)
            expected_cost = 0.0
            expected_future = 0.0
            for target, prob, cost in transitions:
                expected_cost += prob * cost
                expected_future += prob * values.get(target, 0.0)
            q[idx] = -expected_cost + gamma * expected_future

        scaled = beta * q
        scaled -= np.max(scaled)
        exp_q = np.exp(scaled)
        total = np.sum(exp_q)
        if total <= 0:
            return np.ones(len(action_order)) / len(action_order)
        return exp_q / total

    def _get_values(self, mdp: MDP, beta: float) -> dict[str, float]:
        """Return soft values at *beta*, using cache if available."""
        # Round β for cache key
        key = round(beta, 6)
        if self.cache_values and key in self._value_cache:
            return self._value_cache[key]
        values = _soft_value_iteration(mdp, beta)
        if self.cache_values:
            self._value_cache[key] = values
        return values

    def clear_cache(self) -> None:
        """Clear the value-function cache."""
        self._value_cache.clear()
