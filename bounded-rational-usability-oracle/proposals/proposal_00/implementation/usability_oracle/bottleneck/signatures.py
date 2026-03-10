"""
usability_oracle.bottleneck.signatures — Information-theoretic signature computation.

For each UI state, computes the information-theoretic signature that
characterises its cognitive demands:

  - **Action entropy** H(π(·|s)) — how uncertain is the user's action
    selection?
  - **Transition entropy** H(S'|S=s, A=a) — how unpredictable are the
    outcomes?
  - **Mutual information** I(A; S' | S=s) — how much does the action choice
    determine the outcome?
  - **Channel utilization** ρ = I / C — what fraction of cognitive capacity
    is consumed?
  - **KL from uniform** D_KL(π(·|s) ‖ U) — how much does the policy
    differ from random?

The signature pattern classifies bottleneck types:

  ┌───────────────────────────┬────────┬────────┬────────┬────────┐
  │ Bottleneck                │ H(π)   │ I(A;S')│ ρ      │ D_KL   │
  ├───────────────────────────┼────────┼────────┼────────┼────────┤
  │ Perceptual overload       │ high   │ low    │ > 1    │ low    │
  │ Choice paralysis          │ high   │ high   │ medium │ low    │
  │ Motor difficulty          │ low    │ high   │ medium │ high   │
  │ Memory decay              │ medium │ low    │ medium │ medium │
  │ Cross-channel interference│ medium │ medium │ > 1    │ medium │
  └───────────────────────────┴────────┴────────┴────────┴────────┘

References
----------
- Cover, T. & Thomas, J. (2006). *Elements of Information Theory*.
- Ortega, P. & Braun, D. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bottleneck.models import BottleneckSignature
from usability_oracle.core.enums import BottleneckType
from usability_oracle.mdp.models import MDP
from usability_oracle.policy.models import Policy


# ---------------------------------------------------------------------------
# Default channel capacities (nats/s)
# ---------------------------------------------------------------------------

_DEFAULT_CAPACITY_NATS = 7.0 * math.log(2)  # ~4.85 nats/s (≈7 bits/s)


# ---------------------------------------------------------------------------
# SignatureComputer
# ---------------------------------------------------------------------------

@dataclass
class SignatureComputer:
    """Compute information-theoretic signatures for MDP states.

    Parameters
    ----------
    default_capacity : float
        Default channel capacity in nats/s (used when channel-specific
        capacity is unavailable).  Default ≈ 4.85 nats/s (7 bits/s).
    """

    default_capacity: float = _DEFAULT_CAPACITY_NATS

    # ── Public API --------------------------------------------------------

    def compute(
        self,
        mdp: MDP,
        policy: Policy,
        state: str,
    ) -> BottleneckSignature:
        """Compute the full information-theoretic signature for *state*.

        Parameters
        ----------
        mdp : MDP
        policy : Policy
        state : str

        Returns
        -------
        BottleneckSignature
        """
        entropy = self._action_entropy(policy, state)
        mi = self._mutual_information(policy, mdp, state)
        capacity = self._estimate_channel_capacity(mdp, state)
        utilization = self._channel_utilization(mi, capacity)

        return BottleneckSignature(
            entropy=entropy,
            mutual_info=mi,
            channel_capacity=capacity,
            utilization=utilization,
        )

    def classify_signature(self, sig: BottleneckSignature) -> BottleneckType:
        """Rule-based classification of a signature into a bottleneck type.

        Uses the pattern table from the module docstring to match the
        signature to the most likely bottleneck type.

        Parameters
        ----------
        sig : BottleneckSignature

        Returns
        -------
        BottleneckType
            The best-matching bottleneck type.
        """
        # Threshold definitions (in nats)
        H_LOW = 0.5
        H_HIGH = 2.0
        I_LOW = 0.3
        I_HIGH = 1.5
        RHO_HIGH = 0.8

        h = sig.entropy
        mi = sig.mutual_info
        rho = sig.utilization

        # Compute match scores for each type
        scores: dict[BottleneckType, float] = {}

        # Perceptual overload: high H, low I, high ρ
        scores[BottleneckType.PERCEPTUAL_OVERLOAD] = (
            _sigmoid(h, H_HIGH, 1.0)
            * _sigmoid(-mi, -I_LOW, 1.0)
            * _sigmoid(rho, RHO_HIGH, 2.0)
        )

        # Choice paralysis: high H, high I, moderate ρ, low D_KL
        scores[BottleneckType.CHOICE_PARALYSIS] = (
            _sigmoid(h, H_HIGH, 1.0)
            * _sigmoid(mi, I_HIGH, 1.0)
            * (1.0 - _sigmoid(rho, RHO_HIGH, 2.0)) * 0.5
            + _sigmoid(h, H_HIGH, 1.0) * 0.5
        )

        # Motor difficulty: low H, high I, moderate ρ
        scores[BottleneckType.MOTOR_DIFFICULTY] = (
            _sigmoid(-h, -H_LOW, 1.0)
            * _sigmoid(mi, I_HIGH, 1.0)
        )

        # Memory decay: medium H, low I, medium ρ
        scores[BottleneckType.MEMORY_DECAY] = (
            _bell(h, (H_LOW + H_HIGH) / 2, 1.0)
            * _sigmoid(-mi, -I_LOW, 1.0)
            * _bell(rho, 0.5, 1.0)
        )

        # Cross-channel interference: medium H, medium I, high ρ
        scores[BottleneckType.CROSS_CHANNEL_INTERFERENCE] = (
            _bell(h, (H_LOW + H_HIGH) / 2, 1.0)
            * _bell(mi, (I_LOW + I_HIGH) / 2, 1.0)
            * _sigmoid(rho, RHO_HIGH, 2.0)
        )

        # Return the type with the highest score
        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best_type

    # ── Entropy measures --------------------------------------------------

    def _action_entropy(self, policy: Policy, state: str) -> float:
        """Compute the policy entropy H(π(·|s)).

        H(π(·|s)) = -Σ_a π(a|s) ln π(a|s)

        Parameters
        ----------
        policy : Policy
        state : str

        Returns
        -------
        float
            Entropy in nats.
        """
        probs = policy.action_probs(state)
        if not probs:
            return 0.0
        h = 0.0
        for p in probs.values():
            if p > 0:
                h -= p * math.log(p)
        return h

    def _transition_entropy(
        self,
        mdp: MDP,
        state: str,
        action: str,
    ) -> float:
        """Compute the transition entropy H(S' | S=s, A=a).

        H(S'|S=s, A=a) = -Σ_{s'} T(s'|s,a) ln T(s'|s,a)

        Parameters
        ----------
        mdp : MDP
        state : str
        action : str

        Returns
        -------
        float
            Entropy in nats.
        """
        transitions = mdp.get_transitions(state, action)
        if not transitions:
            return 0.0
        h = 0.0
        for _, prob, _ in transitions:
            if prob > 0:
                h -= prob * math.log(prob)
        return h

    def _mutual_information(
        self,
        policy: Policy,
        mdp: MDP,
        state: str,
    ) -> float:
        """Compute I(A; S' | S = s): mutual information between action and next state.

        I(A; S' | S=s) = H(S' | S=s) - H(S' | S=s, A)
                       = H(S' | S=s) - Σ_a π(a|s) H(S'|S=s, A=a)

        where H(S'|S=s) = -Σ_{s'} P(s'|s) ln P(s'|s)
        and   P(s'|s) = Σ_a π(a|s) T(s'|s,a)

        Parameters
        ----------
        policy : Policy
        mdp : MDP
        state : str

        Returns
        -------
        float
            Mutual information in nats (≥ 0).
        """
        action_probs = policy.action_probs(state)
        if not action_probs:
            return 0.0

        # Compute marginal next-state distribution P(s'|s)
        marginal: dict[str, float] = {}
        for aid, pi_a in action_probs.items():
            for target, t_prob, _ in mdp.get_transitions(state, aid):
                marginal[target] = marginal.get(target, 0.0) + pi_a * t_prob

        # H(S'|S=s)
        h_marginal = 0.0
        for p in marginal.values():
            if p > 0:
                h_marginal -= p * math.log(p)

        # E_a[H(S'|S=s, A=a)]
        expected_conditional = 0.0
        for aid, pi_a in action_probs.items():
            h_cond = self._transition_entropy(mdp, state, aid)
            expected_conditional += pi_a * h_cond

        mi = h_marginal - expected_conditional
        return max(0.0, mi)  # Ensure non-negative (rounding errors)

    def _channel_utilization(
        self,
        info_rate: float,
        capacity: float,
    ) -> float:
        """Compute channel utilization ρ = info_rate / capacity.

        Parameters
        ----------
        info_rate : float
            Information rate (nats/s or nats/step).
        capacity : float
            Channel capacity.

        Returns
        -------
        float
            Utilization ∈ [0, ∞).
        """
        if capacity <= 0:
            return float("inf") if info_rate > 0 else 0.0
        return info_rate / capacity

    def _kl_from_uniform(self, policy: Policy, state: str) -> float:
        """Compute D_KL(π(·|s) ‖ U) where U is uniform.

        D_KL(π ‖ U) = Σ_a π(a|s) ln(π(a|s) / (1/|A|))
                     = Σ_a π(a|s) ln π(a|s) + ln|A|
                     = ln|A| - H(π)

        Parameters
        ----------
        policy : Policy
        state : str

        Returns
        -------
        float
            KL divergence in nats (≥ 0).
        """
        probs = policy.action_probs(state)
        n = len(probs)
        if n <= 1:
            return 0.0
        h = self._action_entropy(policy, state)
        return math.log(n) - h

    def _estimate_channel_capacity(self, mdp: MDP, state: str) -> float:
        """Estimate the effective channel capacity at *state*.

        Uses state features (e.g., ``n_elements``, ``visual_complexity``) to
        estimate the dominant channel's capacity.

        Parameters
        ----------
        mdp : MDP
        state : str

        Returns
        -------
        float
            Estimated capacity in nats/s.
        """
        state_obj = mdp.states.get(state)
        if state_obj is None:
            return self.default_capacity

        features = state_obj.features
        n_elements = features.get("n_elements", 5.0)
        complexity = features.get("visual_complexity", 0.5)

        # Capacity degrades with visual complexity
        degradation = 1.0 / (1.0 + 0.1 * complexity)
        return self.default_capacity * degradation


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sigmoid(x: float, threshold: float, steepness: float) -> float:
    """Logistic sigmoid centred at *threshold*."""
    z = steepness * (x - threshold)
    z = max(-20.0, min(20.0, z))  # Clamp for numerical stability
    return 1.0 / (1.0 + math.exp(-z))


def _bell(x: float, centre: float, width: float) -> float:
    """Gaussian bell curve centred at *centre* with given *width*."""
    return math.exp(-0.5 * ((x - centre) / max(width, 0.01)) ** 2)
