"""
usability_oracle.policy.value_iteration — Soft (entropy-regularised) value iteration.

Implements the soft Bellman equation for computing bounded-rational
policies under the free-energy formulation:

    V(s) = (1/β) · log Σ_a p₀(a|s) · exp(β · [R(s,a) + γ Σ_{s'} T(s'|s,a) V(s')])

In the cost formulation (negate rewards):

    V(s) = −(1/β) · log Σ_a p₀(a|s) · exp(−β · [c(s,a) + γ Σ_{s'} T(s'|s,a) V(s')])

This reduces to standard value iteration as β → ∞ and to the prior-
averaged cost as β → 0.

The optimal bounded-rational policy is:

    π*(a|s) = p₀(a|s) · exp(−β · Q*(s,a)) / Z(s)

where Q*(s,a) = c(s,a) + γ Σ_{s'} T(s'|s,a) V*(s').

References
----------
- Todorov, E. (2007). Linearly-solvable Markov decision problems. *NIPS*.
- Todorov, E. (2009). Efficient computation of optimal actions. *PNAS*,
  106(28), 11478–11483.
- Levine, S. (2018). Reinforcement learning and control as probabilistic
  inference: Tutorial and review. *arXiv:1805.00909*.
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from usability_oracle.mdp.models import MDP
from usability_oracle.policy.models import Policy, PolicyResult, QValues

logger = logging.getLogger(__name__)


class SoftValueIteration:
    """Entropy-regularised value iteration for bounded-rational policies.

    Computes V*, Q*, and the optimal softmax policy for a given rationality
    parameter β.  Convergence is guaranteed for γ < 1 and bounded costs.

    Usage
    -----
    >>> solver = SoftValueIteration()
    >>> result = solver.solve(mdp, beta=2.0, prior=uniform_policy)
    >>> result.policy  # bounded-rational softmax policy
    """

    def solve(
        self,
        mdp: MDP,
        beta: float,
        prior: Policy,
        discount: Optional[float] = None,
        epsilon: float = 1e-6,
        max_iter: int = 5_000,
    ) -> PolicyResult:
        """Run soft value iteration.

        Parameters
        ----------
        mdp : MDP
        beta : float
            Rationality parameter (inverse temperature).
        prior : Policy
            Prior policy p₀(a|s).
        discount : float, optional
            Override ``mdp.discount``.
        epsilon : float
            Convergence threshold.
        max_iter : int

        Returns
        -------
        PolicyResult
            Contains policy, Q-values, state values, free energy, and
            convergence diagnostics.
        """
        gamma = discount if discount is not None else mdp.discount
        state_ids = list(mdp.states.keys())

        # Initialise soft values
        values: dict[str, float] = {sid: 0.0 for sid in state_ids}

        converged = False
        final_residual = float("inf")

        for iteration in range(max_iter):
            new_values = self._soft_bellman_update(mdp, values, beta, prior, gamma)

            # Check convergence
            residual = max(
                abs(new_values[s] - values[s]) for s in state_ids
            ) if state_ids else 0.0

            if residual < epsilon:
                converged = True
                final_residual = residual
                values = new_values
                logger.info(
                    "Soft VI converged in %d iterations (residual=%.2e, β=%.2f)",
                    iteration + 1,
                    residual,
                    beta,
                )
                break

            values = new_values
            final_residual = residual

        if not converged:
            logger.warning(
                "Soft VI did not converge in %d iterations (residual=%.2e)",
                max_iter,
                final_residual,
            )

        # Extract Q-values and policy
        q_values = self._extract_q_values(mdp, values, gamma)
        policy = self._extract_policy(q_values, beta, prior)

        # Compute free energy
        from usability_oracle.policy.free_energy import FreeEnergyComputer

        fe_computer = FreeEnergyComputer()
        free_energy = fe_computer.compute(policy, mdp, beta, prior)

        return PolicyResult(
            policy=policy,
            q_values=q_values,
            state_values=values,
            free_energy=free_energy,
            convergence_info={
                "converged": converged,
                "iterations": iteration + 1 if converged else max_iter,
                "residual": final_residual,
                "beta": beta,
                "discount": gamma,
            },
        )

    def _soft_bellman_update(
        self,
        mdp: MDP,
        values: dict[str, float],
        beta: float,
        prior: Policy,
        discount: float,
    ) -> dict[str, float]:
        """One soft Bellman backup for all states.

        Soft Bellman equation (cost formulation):

            V(s) = −(1/β) log Σ_a p₀(a|s) exp(−β [c(s,a) + γ Σ_{s'} T(s'|s,a) V(s')])

        When β → ∞ this recovers the standard Bellman optimality equation.
        When β → 0 this gives V(s) = E_{a~p₀}[c(s,a) + γ E[V(s')]].

        Parameters
        ----------
        mdp : MDP
        values : dict[str, float]
        beta : float
        prior : Policy
        discount : float

        Returns
        -------
        dict[str, float]
            Updated soft values.
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

            # Get prior probabilities
            prior_dist = prior.state_action_probs.get(sid, {})

            # Compute Q(s,a) for each available action
            q_vals: list[float] = []
            p0_vals: list[float] = []

            for aid in available:
                outcomes = mdp.get_transitions(sid, aid)
                q_sa = 0.0
                for target, prob, cost in outcomes:
                    q_sa += prob * (cost + discount * values.get(target, 0.0))
                q_vals.append(q_sa)

                p0_a = prior_dist.get(aid, 1.0 / len(available))
                p0_vals.append(max(p0_a, 1e-300))

            q_arr = np.array(q_vals, dtype=np.float64)
            p0_arr = np.array(p0_vals, dtype=np.float64)

            # Normalise prior
            p0_sum = p0_arr.sum()
            if p0_sum > 0:
                p0_arr /= p0_sum

            if beta <= 1e-10:
                # β → 0: prior-weighted average
                new_values[sid] = float(np.dot(p0_arr, q_arr))
                continue

            # Soft Bellman: V(s) = -(1/β) log Σ_a p₀(a|s) exp(-β Q(s,a))
            # Use log-sum-exp trick for numerical stability
            neg_beta_q = -beta * q_arr
            log_p0 = np.log(np.maximum(p0_arr, 1e-300))
            log_terms = log_p0 + neg_beta_q

            # log-sum-exp
            max_log = float(np.max(log_terms))
            log_sum = max_log + float(np.log(np.sum(np.exp(log_terms - max_log))))

            new_values[sid] = -log_sum / beta

        return new_values

    def _extract_q_values(
        self,
        mdp: MDP,
        values: dict[str, float],
        discount: float,
    ) -> QValues:
        """Extract Q(s,a) from converged soft value function.

        Q(s,a) = Σ_{s'} T(s'|s,a) [c(s,a,s') + γ V(s')]

        Parameters
        ----------
        mdp : MDP
        values : dict[str, float]
        discount : float

        Returns
        -------
        QValues
        """
        q_dict: dict[str, dict[str, float]] = {}

        for sid in mdp.states:
            state = mdp.states[sid]
            if state.is_terminal or state.is_goal:
                continue

            available = mdp.get_actions(sid)
            if not available:
                continue

            q_s: dict[str, float] = {}
            for aid in available:
                outcomes = mdp.get_transitions(sid, aid)
                q_sa = 0.0
                for target, prob, cost in outcomes:
                    q_sa += prob * (cost + discount * values.get(target, 0.0))
                q_s[aid] = q_sa

            q_dict[sid] = q_s

        return QValues(values=q_dict)

    def _extract_policy(
        self,
        q_values: QValues,
        beta: float,
        prior: Policy,
    ) -> Policy:
        """Extract the softmax policy from Q-values.

        π*(a|s) = p₀(a|s) · exp(−β Q(s,a)) / Z(s)

        Parameters
        ----------
        q_values : QValues
        beta : float
        prior : Policy

        Returns
        -------
        Policy
        """
        from usability_oracle.policy.softmax import SoftmaxPolicy

        return SoftmaxPolicy.from_q_values(q_values, beta, prior)
