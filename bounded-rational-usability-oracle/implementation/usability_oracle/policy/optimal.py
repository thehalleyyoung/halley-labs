"""
usability_oracle.policy.optimal — Optimal and bounded-rational policy computation.

Provides :class:`OptimalPolicyComputer` which wraps the MDP solvers and
soft value iteration to compute:

1. **Deterministic optimal policy** — minimises expected cost (β → ∞).
2. **Bounded-rational policy** — minimises free energy F(π) for a given β.
3. **Pareto front** — the cost vs. information trade-off as β varies.
4. **ε-greedy exploration** — for Monte Carlo evaluation.

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
- Genewein, T. et al. (2015). Bounded rationality, abstraction, and
  hierarchical decision-making. *arXiv:1506.04373*.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import numpy as np

from usability_oracle.mdp.models import MDP
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.policy.models import Policy, PolicyResult, QValues
from usability_oracle.policy.softmax import SoftmaxPolicy
from usability_oracle.policy.value_iteration import SoftValueIteration

logger = logging.getLogger(__name__)


class OptimalPolicyComputer:
    """Compute optimal and bounded-rational policies for an MDP.

    Provides a unified interface for computing policies under different
    rationality assumptions:

    - ``compute()`` — standard deterministic optimal (rational) policy
    - ``compute_bounded_rational()`` — softmax policy for a given β
    - ``pareto_front()`` — sweep over β values to characterise the trade-off

    Parameters
    ----------
    discount : float
        Default discount factor.
    epsilon : float
        Convergence threshold for iterative solvers.
    max_iter : int
        Maximum solver iterations.
    """

    def __init__(
        self,
        discount: float = 0.99,
        epsilon: float = 1e-6,
        max_iter: int = 5000,
    ) -> None:
        self.discount = discount
        self.epsilon = epsilon
        self.max_iter = max_iter

    # ── Deterministic optimal policy --------------------------------------

    def compute(
        self,
        mdp: MDP,
        objective: str = "cost",
    ) -> Policy:
        """Compute the deterministic optimal policy.

        Solves the MDP via value iteration and returns the greedy
        (deterministic) policy that minimises expected cost.

        Parameters
        ----------
        mdp : MDP
        objective : str
            ``"cost"`` (minimise) or ``"reward"`` (maximise).

        Returns
        -------
        Policy
            A deterministic policy (each state maps to one action with
            probability 1).
        """
        solver = ValueIterationSolver()
        values, det_policy = solver.solve(
            mdp,
            discount=self.discount,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
        )

        # Convert deterministic dict to Policy object
        return self._greedy_policy_from_dict(det_policy)

    # ── Bounded-rational policy -------------------------------------------

    def compute_bounded_rational(
        self,
        mdp: MDP,
        beta: float,
        prior: Optional[Policy] = None,
    ) -> PolicyResult:
        """Compute the bounded-rational policy for a given β.

        Minimises the free energy:
            F(π) = E_π[cost] + (1/β) D_KL(π ‖ p₀)

        Parameters
        ----------
        mdp : MDP
        beta : float
            Rationality parameter. β → 0 gives the prior; β → ∞ gives
            the deterministic optimum.
        prior : Policy, optional
            Prior policy p₀. Defaults to uniform.

        Returns
        -------
        PolicyResult
        """
        if prior is None:
            prior = self._uniform_prior(mdp)

        solver = SoftValueIteration()
        return solver.solve(
            mdp=mdp,
            beta=beta,
            prior=prior,
            discount=self.discount,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
        )

    # ── Pareto front ------------------------------------------------------

    def pareto_front(
        self,
        mdp: MDP,
        betas: list[float],
        prior: Optional[Policy] = None,
    ) -> list[tuple[float, float, Policy]]:
        """Compute the Pareto front of cost vs. information.

        For each β, computes the optimal bounded-rational policy and
        evaluates (expected_cost, information_cost, policy).

        The Pareto front characterises the fundamental trade-off:
        moving along the front, one can only reduce information cost
        at the expense of higher expected cost, and vice versa.

        Parameters
        ----------
        mdp : MDP
        betas : list[float]
            Rationality parameters to evaluate (should be sorted).
        prior : Policy, optional

        Returns
        -------
        list[tuple[float, float, Policy]]
            List of ``(expected_cost, information_cost, policy)`` tuples,
            one per β value.
        """
        if prior is None:
            prior = self._uniform_prior(mdp)

        from usability_oracle.policy.free_energy import FreeEnergyComputer

        fe_computer = FreeEnergyComputer()
        front: list[tuple[float, float, Policy]] = []

        for beta in sorted(betas):
            if beta <= 1e-10:
                # β → 0: prior policy
                ec = fe_computer._expected_cost_under_policy(prior, mdp)
                front.append((ec, 0.0, prior))
                continue

            result = self.compute_bounded_rational(mdp, beta, prior)
            decomp = fe_computer.decompose(result.policy, mdp, beta, prior)
            front.append((
                decomp.expected_cost,
                decomp.information_cost,
                result.policy,
            ))

        return front

    # ── Helper policies ---------------------------------------------------

    @staticmethod
    def _greedy_policy(q_values: QValues) -> Policy:
        """Convert Q-values to a deterministic greedy policy.

        π(a|s) = 1 if a = argmin_{a'} Q(s, a'), else 0.

        Parameters
        ----------
        q_values : QValues

        Returns
        -------
        Policy
        """
        state_action_probs: dict[str, dict[str, float]] = {}
        for state, q_s in q_values.values.items():
            if not q_s:
                continue
            best = min(q_s, key=q_s.get)  # type: ignore[arg-type]
            state_action_probs[state] = {a: (1.0 if a == best else 0.0) for a in q_s}
        return Policy(state_action_probs=state_action_probs)

    @staticmethod
    def _epsilon_greedy(q_values: QValues, epsilon: float = 0.1) -> Policy:
        """Create an ε-greedy policy from Q-values.

        With probability ε, choose uniformly at random; otherwise choose
        the greedy action.

        π(a|s) = (1 − ε) + ε/|A(s)|  if a = argmin Q(s,a)
                 ε / |A(s)|            otherwise

        Parameters
        ----------
        q_values : QValues
        epsilon : float
            Exploration probability.

        Returns
        -------
        Policy
        """
        state_action_probs: dict[str, dict[str, float]] = {}
        for state, q_s in q_values.values.items():
            if not q_s:
                continue
            n = len(q_s)
            best = min(q_s, key=q_s.get)  # type: ignore[arg-type]

            dist: dict[str, float] = {}
            for a in q_s:
                if a == best:
                    dist[a] = (1.0 - epsilon) + epsilon / n
                else:
                    dist[a] = epsilon / n

            state_action_probs[state] = dist

        return Policy(state_action_probs=state_action_probs)

    @staticmethod
    def _greedy_policy_from_dict(det_policy: dict[str, str]) -> Policy:
        """Convert a deterministic policy dict to a :class:`Policy`."""
        state_action_probs: dict[str, dict[str, float]] = {}
        for sid, aid in det_policy.items():
            state_action_probs[sid] = {aid: 1.0}
        return Policy(state_action_probs=state_action_probs)

    @staticmethod
    def _uniform_prior(mdp: MDP) -> Policy:
        """Construct a uniform prior policy."""
        state_action_probs: dict[str, dict[str, float]] = {}
        for sid in mdp.states:
            actions = mdp.get_actions(sid)
            if actions:
                p = 1.0 / len(actions)
                state_action_probs[sid] = {a: p for a in actions}
        return Policy(state_action_probs=state_action_probs)
