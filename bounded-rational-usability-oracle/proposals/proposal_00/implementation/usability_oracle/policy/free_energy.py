"""
usability_oracle.policy.free_energy — Free-energy computation and decomposition.

Implements the free-energy objective for bounded-rational decision-making:

    F(π) = E_π[cost] + (1/β) · D_KL(π ‖ p₀)

This is the variational objective whose minimiser is the softmax policy
π*(a|s) ∝ p₀(a|s) exp(−β Q*(s,a)).  The free energy decomposes into:

  - **Expected cost** E_π[c]: the performance component.
  - **Information cost** (1/β) D_KL(π ‖ p₀): the cognitive effort to deviate
    from default behaviour.

The trade-off is governed by the rationality parameter β:
  - β → 0 :  F → (1/β) D_KL → ∞ unless π = p₀  (habitual behaviour)
  - β → ∞ :  F → E_π[c]  (fully rational, ignore information cost)

The **rate-distortion curve** traces (information cost, expected cost)
as β varies, characterising the fundamental trade-off between cognitive
effort and task performance.

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*,
  469(2153), 20120683.
- Todorov, E. (2009). Efficient computation of optimal actions. *PNAS*.
- Still, S. & Precup, D. (2012). An information-theoretic approach to
  curiosity-driven reinforcement learning. *Theory in Biosciences*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.mdp.models import MDP
from usability_oracle.policy.models import Policy, QValues, PolicyResult
from usability_oracle.policy.softmax import SoftmaxPolicy


# ---------------------------------------------------------------------------
# Free-energy decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreeEnergyDecomposition:
    """Decomposition of the free energy into its components.

    Attributes
    ----------
    total_free_energy : float
        F(π) = expected_cost + information_cost.
    expected_cost : float
        E_π[c] — average cost under the policy.
    information_cost : float
        (1/β) D_KL(π ‖ p₀) — cognitive effort component.
    per_state_free_energy : dict[str, float]
        F_s for each state: local free energy contribution.
    per_state_expected_cost : dict[str, float]
    per_state_information_cost : dict[str, float]
    beta : float
        Rationality parameter used.
    """

    total_free_energy: float
    expected_cost: float
    information_cost: float
    per_state_free_energy: dict[str, float] = field(default_factory=dict)
    per_state_expected_cost: dict[str, float] = field(default_factory=dict)
    per_state_information_cost: dict[str, float] = field(default_factory=dict)
    beta: float = 1.0

    def __repr__(self) -> str:
        return (
            f"FreeEnergyDecomposition(F={self.total_free_energy:.4f}, "
            f"E[c]={self.expected_cost:.4f}, "
            f"I={self.information_cost:.4f}, β={self.beta:.2f})"
        )


# ---------------------------------------------------------------------------
# FreeEnergyComputer
# ---------------------------------------------------------------------------

class FreeEnergyComputer:
    """Compute and decompose the free energy of a bounded-rational policy.

    The free energy is defined as:

        F(π) = E_π[c(s, a)] + (1/β) · D_KL(π ‖ p₀)

    where the expectation is over the stationary state distribution
    induced by π in the MDP.

    For tractability we compute the per-state free energy and average
    over the reachable states (assuming a uniform weighting or using the
    stationary distribution when available).
    """

    def compute(
        self,
        policy: Policy,
        mdp: MDP,
        beta: float,
        prior: Policy,
    ) -> float:
        """Compute the total free energy F(π).

        F(π) = E_π[cost] + (1/β) · D_KL(π ‖ p₀)

        Parameters
        ----------
        policy : Policy
        mdp : MDP
        beta : float
        prior : Policy

        Returns
        -------
        float
        """
        expected_cost = self._expected_cost_under_policy(policy, mdp)
        info_cost = self._information_cost(policy, prior, beta)
        return expected_cost + info_cost

    def decompose(
        self,
        policy: Policy,
        mdp: MDP,
        beta: float,
        prior: Policy,
    ) -> FreeEnergyDecomposition:
        """Decompose F(π) into expected cost and information cost per state.

        Parameters
        ----------
        policy : Policy
        mdp : MDP
        beta : float
        prior : Policy

        Returns
        -------
        FreeEnergyDecomposition
        """
        per_state = self._per_state_decomposition(policy, mdp, beta, prior)

        per_state_fe: dict[str, float] = {}
        per_state_ec: dict[str, float] = {}
        per_state_ic: dict[str, float] = {}

        total_ec = 0.0
        total_ic = 0.0
        n_states = 0

        for sid, (ec, ic) in per_state.items():
            per_state_ec[sid] = ec
            per_state_ic[sid] = ic
            per_state_fe[sid] = ec + ic
            total_ec += ec
            total_ic += ic
            n_states += 1

        # Average over states
        if n_states > 0:
            total_ec /= n_states
            total_ic /= n_states

        return FreeEnergyDecomposition(
            total_free_energy=total_ec + total_ic,
            expected_cost=total_ec,
            information_cost=total_ic,
            per_state_free_energy=per_state_fe,
            per_state_expected_cost=per_state_ec,
            per_state_information_cost=per_state_ic,
            beta=beta,
        )

    def optimal_policy(
        self,
        mdp: MDP,
        beta: float,
        prior: Policy,
        discount: float = 0.99,
        epsilon: float = 1e-6,
        max_iter: int = 5000,
    ) -> Policy:
        """Compute the policy that minimises the free energy.

        The optimal bounded-rational policy is:
            π*(a|s) ∝ p₀(a|s) · exp(−β · Q*(s,a))

        where Q* is obtained via soft value iteration.

        Parameters
        ----------
        mdp : MDP
        beta : float
        prior : Policy
        discount : float
        epsilon : float
        max_iter : int

        Returns
        -------
        Policy
        """
        from usability_oracle.policy.value_iteration import SoftValueIteration

        solver = SoftValueIteration()
        result = solver.solve(
            mdp=mdp,
            beta=beta,
            prior=prior,
            discount=discount,
            epsilon=epsilon,
            max_iter=max_iter,
        )
        return result.policy

    # ── Internal computations ---------------------------------------------

    def _expected_cost_under_policy(
        self, policy: Policy, mdp: MDP
    ) -> float:
        """E_π[c(s,a)] averaged over reachable states.

        For each state s, the expected one-step cost is:
            E_s[c] = Σ_a π(a|s) · Σ_{s'} T(s'|s,a) · c(s,a,s')

        We average these over all states with policy entries.
        """
        total_cost = 0.0
        n_states = 0

        for sid in policy.state_action_probs:
            if sid not in mdp.states:
                continue
            state = mdp.states[sid]
            if state.is_terminal or state.is_goal:
                continue

            state_cost = 0.0
            dist = policy.state_action_probs[sid]
            for aid, pi_a in dist.items():
                if pi_a <= 0:
                    continue
                outcomes = mdp.get_transitions(sid, aid)
                for target, prob, cost in outcomes:
                    state_cost += pi_a * prob * cost

            total_cost += state_cost
            n_states += 1

        return total_cost / max(n_states, 1)

    def _information_cost(
        self,
        policy: Policy,
        prior: Policy,
        beta: float,
    ) -> float:
        """(1/β) · D_KL(π ‖ p₀) averaged over states.

        Parameters
        ----------
        policy : Policy
        prior : Policy
        beta : float

        Returns
        -------
        float
        """
        if beta <= 0:
            return 0.0

        total_kl = 0.0
        n_states = 0

        for sid in policy.state_action_probs:
            kl = SoftmaxPolicy.kl_divergence(policy, prior, sid)
            total_kl += kl
            n_states += 1

        avg_kl = total_kl / max(n_states, 1)
        return avg_kl / beta

    def _per_state_decomposition(
        self,
        policy: Policy,
        mdp: MDP,
        beta: float,
        prior: Policy,
    ) -> dict[str, tuple[float, float]]:
        """Per-state (expected_cost, information_cost) decomposition.

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping ``state_id -> (E_s[c], (1/β) D_KL_s)``.
        """
        result: dict[str, tuple[float, float]] = {}

        for sid in policy.state_action_probs:
            if sid not in mdp.states:
                continue
            state = mdp.states[sid]
            if state.is_terminal or state.is_goal:
                result[sid] = (0.0, 0.0)
                continue

            # Expected cost for this state
            state_cost = 0.0
            dist = policy.state_action_probs[sid]
            for aid, pi_a in dist.items():
                if pi_a <= 0:
                    continue
                outcomes = mdp.get_transitions(sid, aid)
                for target, prob, cost in outcomes:
                    state_cost += pi_a * prob * cost

            # Information cost for this state
            kl = SoftmaxPolicy.kl_divergence(policy, prior, sid)
            info_cost = kl / beta if beta > 0 else 0.0

            result[sid] = (state_cost, info_cost)

        return result

    # ── Rate-distortion curve ---------------------------------------------

    def rate_distortion_curve(
        self,
        mdp: MDP,
        betas: list[float],
        prior: Optional[Policy] = None,
        discount: float = 0.99,
    ) -> list[tuple[float, float]]:
        """Trace the rate-distortion curve as β varies.

        For each β, compute the optimal bounded-rational policy and
        evaluate its (information_cost, expected_cost) pair.

        The curve characterises the fundamental trade-off:
        - Low β (left): low information cost, high expected cost
        - High β (right): high information cost, low expected cost

        Parameters
        ----------
        mdp : MDP
        betas : list[float]
            Rationality parameters to evaluate.
        prior : Policy, optional
        discount : float

        Returns
        -------
        list[tuple[float, float]]
            List of (information_cost, expected_cost) pairs.
        """
        from usability_oracle.policy.value_iteration import SoftValueIteration

        if prior is None:
            prior = self._uniform_prior(mdp)

        solver = SoftValueIteration()
        curve: list[tuple[float, float]] = []

        for beta in sorted(betas):
            if beta <= 0:
                curve.append((0.0, self._expected_cost_under_policy(prior, mdp)))
                continue

            result = solver.solve(mdp=mdp, beta=beta, prior=prior, discount=discount)
            decomp = self.decompose(result.policy, mdp, beta, prior)
            curve.append((decomp.information_cost, decomp.expected_cost))

        return curve

    @staticmethod
    def _uniform_prior(mdp: MDP) -> Policy:
        """Construct a uniform prior policy over all actions in each state."""
        state_action_probs: dict[str, dict[str, float]] = {}
        for sid in mdp.states:
            actions = mdp.get_actions(sid)
            if actions:
                p = 1.0 / len(actions)
                state_action_probs[sid] = {a: p for a in actions}
        return Policy(state_action_probs=state_action_probs)
