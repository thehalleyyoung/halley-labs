"""
usability_oracle.mdp.solver — Classical MDP solvers.

Implements three standard algorithms for computing optimal value functions
and deterministic policies in finite MDPs:

1. **Value Iteration** — iterative Bellman updates until convergence.
2. **Policy Iteration** — alternates policy evaluation and improvement.
3. **Linear Programming** — LP formulation solved via ``scipy.optimize.linprog``.

All solvers operate on the :class:`MDP` data structure and return
``(values, policy)`` pairs where *values* maps state IDs to V*(s) and
*policy* maps state IDs to the optimal action ID.

References
----------
- Puterman, M. L. (1994). *Markov Decision Processes*. Wiley.
- Bertsekas, D. P. (2012). *Dynamic Programming and Optimal Control*, 4th ed.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Value Iteration
# ---------------------------------------------------------------------------


class ValueIterationSolver:
    """Solve an MDP via value iteration (Bellman updates).

    Convergence is guaranteed for γ < 1 and bounded costs.  The contraction
    factor is γ, so the number of iterations to ε-optimality is
    O(log(1/ε) / log(1/γ)).

    Parameters
    ----------
    mdp : MDP
    discount : float, optional
        Override for ``mdp.discount``.
    epsilon : float
        Convergence threshold on max|V_{k+1}(s) − V_k(s)|.
    max_iter : int
        Safety cap on the number of iterations.
    """

    def solve(
        self,
        mdp: MDP,
        discount: Optional[float] = None,
        epsilon: float = 1e-6,
        max_iter: int = 10_000,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Run value iteration and return ``(values, policy)``.

        The Bellman optimality equation for costs (minimisation):

            V*(s) = min_a  Σ_{s'} T(s'|s,a) [c(s,a,s') + γ V*(s')]

        Returns
        -------
        values : dict[str, float]
            Optimal value function V*(s).
        policy : dict[str, str]
            Deterministic optimal policy mapping state → action.
        """
        gamma = discount if discount is not None else mdp.discount
        state_ids = list(mdp.states.keys())

        # Initialise values: 0 for goal/terminal states, large for others
        values: dict[str, float] = {}
        for sid in state_ids:
            if mdp.states[sid].is_goal or mdp.states[sid].is_terminal:
                values[sid] = 0.0
            else:
                values[sid] = 0.0

        for iteration in range(max_iter):
            new_values = self._bellman_update(mdp, values, gamma)
            if self._check_convergence(values, new_values, epsilon):
                logger.info(
                    "Value iteration converged in %d iterations (ε=%.2e)",
                    iteration + 1,
                    epsilon,
                )
                values = new_values
                break
            values = new_values
        else:
            logger.warning(
                "Value iteration did not converge within %d iterations", max_iter
            )

        policy = self._extract_policy(mdp, values, gamma)
        return values, policy

    def _bellman_update(
        self, mdp: MDP, values: dict[str, float], discount: float
    ) -> dict[str, float]:
        """One synchronous Bellman backup for all states.

        V_{k+1}(s) = min_a  Σ_{s'} T(s'|s,a) [c(s,a,s') + γ V_k(s')]
        """
        new_values: dict[str, float] = {}
        for sid in mdp.states:
            state = mdp.states[sid]
            if state.is_terminal or state.is_goal:
                new_values[sid] = 0.0
                continue

            available = mdp.get_actions(sid)
            if not available:
                new_values[sid] = values.get(sid, 0.0)
                continue

            best_val = math.inf
            for aid in available:
                outcomes = mdp.get_transitions(sid, aid)
                q_sa = 0.0
                for target, prob, cost in outcomes:
                    q_sa += prob * (cost + discount * values.get(target, 0.0))
                best_val = min(best_val, q_sa)

            new_values[sid] = best_val
        return new_values

    def _extract_policy(
        self, mdp: MDP, values: dict[str, float], discount: float
    ) -> dict[str, str]:
        """Extract greedy policy from converged value function.

        π*(s) = argmin_a  Σ_{s'} T(s'|s,a) [c(s,a,s') + γ V*(s')]
        """
        policy: dict[str, str] = {}
        for sid in mdp.states:
            state = mdp.states[sid]
            if state.is_terminal or state.is_goal:
                continue

            available = mdp.get_actions(sid)
            if not available:
                continue

            best_action = available[0]
            best_val = math.inf
            for aid in available:
                outcomes = mdp.get_transitions(sid, aid)
                q_sa = 0.0
                for target, prob, cost in outcomes:
                    q_sa += prob * (cost + discount * values.get(target, 0.0))
                if q_sa < best_val:
                    best_val = q_sa
                    best_action = aid

            policy[sid] = best_action
        return policy

    def _check_convergence(
        self,
        old_values: dict[str, float],
        new_values: dict[str, float],
        epsilon: float,
    ) -> bool:
        """Check if max absolute difference between value vectors < ε."""
        if not old_values:
            return False
        max_diff = 0.0
        for sid in new_values:
            diff = abs(new_values[sid] - old_values.get(sid, 0.0))
            max_diff = max(max_diff, diff)
        return max_diff < epsilon


# ---------------------------------------------------------------------------
# Policy Iteration
# ---------------------------------------------------------------------------


class PolicyIterationSolver:
    """Solve an MDP via policy iteration (Howard, 1960).

    Alternates exact policy evaluation (solving a linear system) with
    policy improvement (greedy one-step lookahead).  Converges in at
    most |A|^|S| iterations but typically much fewer.

    References
    ----------
    - Howard, R. A. (1960). *Dynamic Programming and Markov Processes*.
    """

    def solve(
        self,
        mdp: MDP,
        discount: Optional[float] = None,
        epsilon: float = 1e-8,
        max_iter: int = 1_000,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Run policy iteration and return ``(values, policy)``."""
        gamma = discount if discount is not None else mdp.discount
        state_ids = list(mdp.states.keys())
        n = len(state_ids)
        sid_to_idx = {sid: i for i, sid in enumerate(state_ids)}

        # Initialise with an arbitrary policy (first available action)
        policy: dict[str, str] = {}
        for sid in state_ids:
            actions = mdp.get_actions(sid)
            if actions:
                policy[sid] = actions[0]

        values: dict[str, float] = {sid: 0.0 for sid in state_ids}

        for iteration in range(max_iter):
            # ── Policy evaluation ────────────────────────────────────
            values = self._policy_evaluation(mdp, policy, gamma, epsilon)

            # ── Policy improvement ───────────────────────────────────
            new_policy = self._policy_improvement(mdp, values, gamma)

            # Check for convergence (policy unchanged)
            if new_policy == policy:
                logger.info("Policy iteration converged in %d iterations", iteration + 1)
                break
            policy = new_policy
        else:
            logger.warning("Policy iteration did not converge within %d iterations", max_iter)

        return values, policy

    def _policy_evaluation(
        self,
        mdp: MDP,
        policy: dict[str, str],
        discount: float,
        epsilon: float = 1e-8,
        max_eval_iter: int = 5_000,
    ) -> dict[str, float]:
        """Evaluate a fixed policy by iterating until convergence.

        For each non-terminal state s with policy action a = π(s):
            V^π(s) = Σ_{s'} T(s'|s,a) [c(s,a,s') + γ V^π(s')]

        Uses iterative evaluation (Gauss-Seidel style) for numerical
        stability with large state spaces.
        """
        values: dict[str, float] = {sid: 0.0 for sid in mdp.states}

        for _ in range(max_eval_iter):
            max_delta = 0.0
            for sid in mdp.states:
                state = mdp.states[sid]
                if state.is_terminal or state.is_goal:
                    continue
                aid = policy.get(sid)
                if aid is None:
                    continue

                outcomes = mdp.get_transitions(sid, aid)
                new_val = 0.0
                for target, prob, cost in outcomes:
                    new_val += prob * (cost + discount * values.get(target, 0.0))

                delta = abs(new_val - values[sid])
                max_delta = max(max_delta, delta)
                values[sid] = new_val

            if max_delta < epsilon:
                break

        return values

    def _policy_improvement(
        self, mdp: MDP, values: dict[str, float], discount: float
    ) -> dict[str, str]:
        """One step of greedy policy improvement.

        π_{k+1}(s) = argmin_a  Σ_{s'} T(s'|s,a) [c(s,a,s') + γ V^{π_k}(s')]
        """
        policy: dict[str, str] = {}
        for sid in mdp.states:
            state = mdp.states[sid]
            if state.is_terminal or state.is_goal:
                continue
            available = mdp.get_actions(sid)
            if not available:
                continue

            best_action = available[0]
            best_val = math.inf
            for aid in available:
                outcomes = mdp.get_transitions(sid, aid)
                q_sa = 0.0
                for target, prob, cost in outcomes:
                    q_sa += prob * (cost + discount * values.get(target, 0.0))
                if q_sa < best_val:
                    best_val = q_sa
                    best_action = aid
            policy[sid] = best_action

        return policy


# ---------------------------------------------------------------------------
# Linear Program Solver
# ---------------------------------------------------------------------------


class LinearProgramSolver:
    """Solve an MDP via linear programming.

    The primal LP formulation for cost minimisation:

        maximise  Σ_s V(s)
        subject to  V(s) ≤ Σ_{s'} T(s'|s,a) [c(s,a,s') + γ V(s')]
                    for all s ∈ S, a ∈ A(s)

    which is equivalent to the dual of the standard occupancy-measure LP.

    Uses ``scipy.optimize.linprog`` with the HiGHS solver.

    References
    ----------
    - de Farias, D. P. & Van Roy, B. (2003). The linear programming approach
      to approximate dynamic programming. *Operations Research*.
    """

    def solve(
        self,
        mdp: MDP,
        discount: Optional[float] = None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Solve the MDP LP and return ``(values, policy)``.

        Returns
        -------
        values : dict[str, float]
        policy : dict[str, str]
        """
        from scipy.optimize import linprog, LinearConstraint  # type: ignore[import-untyped]

        gamma = discount if discount is not None else mdp.discount
        state_ids = [
            sid for sid in mdp.states
            if not mdp.states[sid].is_terminal and not mdp.states[sid].is_goal
        ]
        terminal_ids = [
            sid for sid in mdp.states
            if mdp.states[sid].is_terminal or mdp.states[sid].is_goal
        ]
        n = len(state_ids)

        if n == 0:
            return {sid: 0.0 for sid in mdp.states}, {}

        sid_to_idx = {sid: i for i, sid in enumerate(state_ids)}

        # Objective: maximise Σ V(s) ≡ minimise -Σ V(s)
        c = -np.ones(n)

        # Constraints:  V(s) - γ Σ_{s'} T(s'|s,a) V(s') ≤ Σ_{s'} T(s'|s,a) c(s,a,s')
        # i.e.  A_ub @ V ≤ b_ub
        A_rows: list[np.ndarray] = []
        b_vals: list[float] = []

        for sid in state_ids:
            i = sid_to_idx[sid]
            available = mdp.get_actions(sid)
            for aid in available:
                outcomes = mdp.get_transitions(sid, aid)
                row = np.zeros(n)
                row[i] = 1.0
                rhs = 0.0
                for target, prob, cost in outcomes:
                    rhs += prob * cost
                    j = sid_to_idx.get(target)
                    if j is not None:
                        row[j] -= gamma * prob
                    # Terminal targets contribute 0 to V
                A_rows.append(row)
                b_vals.append(rhs)

        if not A_rows:
            values = {sid: 0.0 for sid in mdp.states}
            return values, {}

        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)

        # Solve LP
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=(None, None),
            method="highs",
            options={"maxiter": 100_000, "disp": False},
        )

        if not result.success:
            logger.warning("LP solver did not find optimal solution: %s", result.message)
            # Fallback to value iteration
            vi = ValueIterationSolver()
            return vi.solve(mdp, discount=gamma)

        # Extract values
        values: dict[str, float] = {}
        for sid in terminal_ids:
            values[sid] = 0.0
        for sid in state_ids:
            values[sid] = float(result.x[sid_to_idx[sid]])

        # Extract greedy policy
        policy: dict[str, str] = {}
        for sid in state_ids:
            available = mdp.get_actions(sid)
            if not available:
                continue
            best_action = available[0]
            best_val = math.inf
            for aid in available:
                outcomes = mdp.get_transitions(sid, aid)
                q_sa = 0.0
                for target, prob, cost in outcomes:
                    q_sa += prob * (cost + gamma * values.get(target, 0.0))
                if q_sa < best_val:
                    best_val = q_sa
                    best_action = aid
            policy[sid] = best_action

        return values, policy
