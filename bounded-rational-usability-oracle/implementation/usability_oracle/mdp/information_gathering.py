"""
usability_oracle.mdp.information_gathering — Active information gathering.

Provides utilities for computing the value of information, optimal
stopping (when to act vs. gather more info), and expected information
gain — all in the context of POMDP-based usability modelling.

Models the user's decision to *explore the UI* (gather information about
hidden elements, scroll to reveal content) vs. *act* (click, type).

Key concepts:
- **Value of Information (VoI)**: expected improvement from observing
  before acting.
- **Optimal stopping**: when further observation is not worth its cost.
- **Active inference**: free-energy minimisation for exploration/exploitation.

References
----------
- Howard, R. A. (1966). Information value theory. *IEEE Trans. SSC*.
- Friston, K. et al. (2017). Active inference: a process theory. *Neural
  Computation*.
- Schwartenbeck, P. et al. (2019). Computational mechanisms of curiosity
  and goal-directed exploration. *eLife*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.mdp.pomdp import BeliefState, POMDP
from usability_oracle.mdp.belief import (
    BeliefUpdater,
    belief_entropy,
    belief_kl_divergence,
)
from usability_oracle.mdp.pomdp_solver import POMDPPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value of Information
# ---------------------------------------------------------------------------


class ValueOfInformation:
    """Compute the value of information for POMDP actions.

    VoI measures how much better the agent could do by first observing
    and then acting, compared to acting immediately.

    VoI(b, a_info) = V*(E_o[b' | b, a_info]) − V*(b)

    where a_info is an information-gathering action.

    Parameters
    ----------
    pomdp : POMDP
    policy : POMDPPolicy
        Current policy (provides V(b)).
    updater : BeliefUpdater, optional
    """

    def __init__(
        self,
        pomdp: POMDP,
        policy: POMDPPolicy,
        updater: Optional[BeliefUpdater] = None,
    ) -> None:
        self.pomdp = pomdp
        self.policy = policy
        self.updater = updater or BeliefUpdater(pomdp)

    def voi_action(self, belief: BeliefState, action_id: str) -> float:
        """Compute the value of information for a specific action.

        VoI(b, a) = Σ_o P(o|b,a) · V(b') − V(b)

        where b' = update(b, a, o).

        Parameters
        ----------
        belief : BeliefState
        action_id : str

        Returns
        -------
        float
            VoI ≥ 0 (information can never hurt in expectation).
        """
        current_value = self.policy.value(belief)
        expected_value_after = 0.0

        for oid in self.pomdp.observation_ids:
            p_obs = self.updater.observation_likelihood(belief, action_id, oid)
            if p_obs < 1e-10:
                continue
            b_prime = self.updater.update(belief, action_id, oid)
            expected_value_after += p_obs * self.policy.value(b_prime)

        return max(0.0, expected_value_after - current_value)

    def voi_all_actions(
        self, belief: BeliefState
    ) -> dict[str, float]:
        """Compute VoI for all actions.

        Returns
        -------
        dict[str, float]
            Action → VoI.
        """
        return {
            aid: self.voi_action(belief, aid)
            for aid in self.pomdp.action_ids
        }

    def best_information_action(self, belief: BeliefState) -> tuple[str, float]:
        """Find the action with highest VoI.

        Returns
        -------
        tuple[str, float]
            (action_id, voi_value)
        """
        vois = self.voi_all_actions(belief)
        if not vois:
            return "", 0.0
        best_aid = max(vois, key=vois.get)  # type: ignore[arg-type]
        return best_aid, vois[best_aid]

    def perfect_voi(self, belief: BeliefState) -> float:
        """Value of *perfect* information: VoPI(b) = E_s[V(δ_s)] − V(b).

        The expected gain from knowing the true state exactly.

        Parameters
        ----------
        belief : BeliefState

        Returns
        -------
        float
        """
        current_value = self.policy.value(belief)
        expected_known = 0.0

        for sid, prob in belief.distribution.items():
            if prob <= 0:
                continue
            point_belief = BeliefState.point(sid)
            expected_known += prob * self.policy.value(point_belief)

        return max(0.0, expected_known - current_value)


# ---------------------------------------------------------------------------
# Expected information gain
# ---------------------------------------------------------------------------


class InformationGain:
    """Compute expected information gain (entropy reduction) per action.

    IG(b, a) = H(b) − E_o[H(b' | b, a, o)]

    This measures how much uncertainty is reduced by taking action *a*
    and receiving observation *o*.

    Parameters
    ----------
    pomdp : POMDP
    updater : BeliefUpdater, optional
    """

    def __init__(
        self, pomdp: POMDP, updater: Optional[BeliefUpdater] = None
    ) -> None:
        self.pomdp = pomdp
        self.updater = updater or BeliefUpdater(pomdp)

    def expected_gain(self, belief: BeliefState, action_id: str) -> float:
        """Expected information gain for action *a* from belief *b*.

        IG(b, a) = H(b) − Σ_o P(o|b,a) H(update(b, a, o))

        Parameters
        ----------
        belief : BeliefState
        action_id : str

        Returns
        -------
        float
            Non-negative information gain in nats.
        """
        current_h = belief_entropy(belief)
        expected_posterior_h = 0.0

        for oid in self.pomdp.observation_ids:
            p_obs = self.updater.observation_likelihood(belief, action_id, oid)
            if p_obs < 1e-10:
                continue
            b_prime = self.updater.update(belief, action_id, oid)
            expected_posterior_h += p_obs * belief_entropy(b_prime)

        return max(0.0, current_h - expected_posterior_h)

    def expected_gain_all(
        self, belief: BeliefState
    ) -> dict[str, float]:
        """Compute IG for all actions.

        Returns
        -------
        dict[str, float]
            Action → expected information gain.
        """
        return {
            aid: self.expected_gain(belief, aid)
            for aid in self.pomdp.action_ids
        }

    def most_informative_action(
        self, belief: BeliefState
    ) -> tuple[str, float]:
        """Find the action that maximises expected information gain.

        Returns
        -------
        tuple[str, float]
            (action_id, gain)
        """
        gains = self.expected_gain_all(belief)
        if not gains:
            return "", 0.0
        best = max(gains, key=gains.get)  # type: ignore[arg-type]
        return best, gains[best]

    def conditional_gain(
        self, belief: BeliefState, action_id: str, obs_id: str
    ) -> float:
        """Information gain for a specific (action, observation) pair.

        IG(b, a, o) = H(b) − H(update(b, a, o))
        """
        current_h = belief_entropy(belief)
        b_prime = self.updater.update(belief, action_id, obs_id)
        return max(0.0, current_h - belief_entropy(b_prime))


# ---------------------------------------------------------------------------
# Optimal stopping
# ---------------------------------------------------------------------------


class OptimalStopping:
    """Optimal stopping: when to act vs. gather more information.

    Computes the *stopping index* for each action—a combination of
    VoI and action cost that determines whether to explore or exploit.

    The agent should *stop exploring* when:
        VoI(b, best_info_action) < cost(best_info_action)

    Parameters
    ----------
    pomdp : POMDP
    policy : POMDPPolicy
    exploration_cost : float
        Cost of taking an information-gathering action.
    """

    def __init__(
        self,
        pomdp: POMDP,
        policy: POMDPPolicy,
        exploration_cost: float = 1.0,
    ) -> None:
        self.pomdp = pomdp
        self.policy = policy
        self.exploration_cost = exploration_cost
        self.voi_computer = ValueOfInformation(pomdp, policy)
        self.ig_computer = InformationGain(pomdp)

    def should_explore(self, belief: BeliefState) -> bool:
        """Determine whether the agent should continue exploring.

        Returns True if the expected value of further information
        exceeds its cost.

        Parameters
        ----------
        belief : BeliefState

        Returns
        -------
        bool
        """
        _, best_voi = self.voi_computer.best_information_action(belief)
        return best_voi > self.exploration_cost

    def stopping_index(self, belief: BeliefState) -> float:
        """Compute a stopping index (higher → more reason to stop).

        SI(b) = V(b) / (V(b) + VoI(b))

        SI near 1.0 → acting now is almost as good as exploring first.
        SI near 0.0 → large information gain available.

        Parameters
        ----------
        belief : BeliefState

        Returns
        -------
        float
            Stopping index in [0, 1].
        """
        current_val = self.policy.value(belief)
        _, best_voi = self.voi_computer.best_information_action(belief)

        denom = abs(current_val) + best_voi
        if denom < 1e-10:
            return 1.0
        return abs(current_val) / denom

    def explore_or_act(
        self, belief: BeliefState
    ) -> tuple[str, str]:
        """Recommend an action and whether it is exploratory or exploitative.

        Returns
        -------
        tuple[str, str]
            (action_id, "explore" or "exploit")
        """
        if self.should_explore(belief):
            best_info_action, _ = self.voi_computer.best_information_action(belief)
            return best_info_action, "explore"
        else:
            return self.policy.action(belief), "exploit"


# ---------------------------------------------------------------------------
# Myopic vs. non-myopic strategies
# ---------------------------------------------------------------------------


class InformationStrategy:
    """Myopic and non-myopic information gathering strategies.

    - **Myopic**: greedy one-step information gain maximisation.
    - **Non-myopic**: multi-step lookahead using the belief-space tree.

    Parameters
    ----------
    pomdp : POMDP
    policy : POMDPPolicy
    horizon : int
        Lookahead horizon for non-myopic strategy.
    """

    def __init__(
        self,
        pomdp: POMDP,
        policy: POMDPPolicy,
        horizon: int = 3,
    ) -> None:
        self.pomdp = pomdp
        self.policy = policy
        self.horizon = horizon
        self.updater = BeliefUpdater(pomdp)
        self.ig_computer = InformationGain(pomdp, self.updater)

    def myopic_action(self, belief: BeliefState) -> tuple[str, float]:
        """Select action by greedy (one-step) information gain.

        Returns
        -------
        tuple[str, float]
            (action_id, gain)
        """
        return self.ig_computer.most_informative_action(belief)

    def non_myopic_action(
        self, belief: BeliefState, depth: Optional[int] = None
    ) -> tuple[str, float]:
        """Select action by multi-step expected information gain.

        Uses a recursive belief-tree search to compute cumulative
        information gain over multiple steps.

        Parameters
        ----------
        belief : BeliefState
        depth : int, optional
            Override for lookahead depth.

        Returns
        -------
        tuple[str, float]
            (best_action_id, cumulative_gain)
        """
        h = depth if depth is not None else self.horizon
        best_action = ""
        best_gain = -math.inf

        for aid in self.pomdp.action_ids:
            gain = self._recursive_gain(belief, aid, h)
            if gain > best_gain:
                best_gain = gain
                best_action = aid

        return best_action, best_gain

    def _recursive_gain(
        self, belief: BeliefState, action_id: str, depth: int
    ) -> float:
        """Recursively compute expected information gain."""
        ig = self.ig_computer.expected_gain(belief, action_id)

        if depth <= 1:
            return ig

        # Expected future gain
        future_gain = 0.0
        for oid in self.pomdp.observation_ids:
            p_obs = self.updater.observation_likelihood(belief, action_id, oid)
            if p_obs < 1e-10:
                continue
            b_prime = self.updater.update(belief, action_id, oid)

            # At the next step, pick the best action
            best_future = 0.0
            for next_aid in self.pomdp.action_ids:
                g = self._recursive_gain(b_prime, next_aid, depth - 1)
                best_future = max(best_future, g)

            future_gain += p_obs * best_future

        discount = self.pomdp.mdp.discount
        return ig + discount * future_gain


# ---------------------------------------------------------------------------
# Entropy reduction as intrinsic reward
# ---------------------------------------------------------------------------


class EntropyReductionReward:
    """Intrinsic reward based on entropy reduction.

    Adds an exploration bonus:

        R_intrinsic(b, a) = w · [H(b) − E_o H(b')]

    to encourage information gathering when uncertainty is high.

    Parameters
    ----------
    pomdp : POMDP
    weight : float
        Weight for the entropy reduction bonus.
    """

    def __init__(
        self, pomdp: POMDP, weight: float = 1.0
    ) -> None:
        self.pomdp = pomdp
        self.weight = weight
        self.updater = BeliefUpdater(pomdp)

    def intrinsic_reward(
        self, belief: BeliefState, action_id: str
    ) -> float:
        """Compute entropy-reduction intrinsic reward.

        Parameters
        ----------
        belief : BeliefState
        action_id : str

        Returns
        -------
        float
        """
        current_h = belief_entropy(belief)
        expected_h = 0.0

        for oid in self.pomdp.observation_ids:
            p_obs = self.updater.observation_likelihood(belief, action_id, oid)
            if p_obs < 1e-10:
                continue
            b_prime = self.updater.update(belief, action_id, oid)
            expected_h += p_obs * belief_entropy(b_prime)

        return self.weight * max(0.0, current_h - expected_h)

    def augmented_reward(
        self, belief: BeliefState, action_id: str
    ) -> float:
        """Task reward + intrinsic entropy reduction reward.

        R_total(b, a) = R_task(b, a) + R_intrinsic(b, a)
        """
        task_reward = self.pomdp.expected_reward(belief, action_id)
        return task_reward + self.intrinsic_reward(belief, action_id)


# ---------------------------------------------------------------------------
# Free-energy active inference
# ---------------------------------------------------------------------------


class ActiveInferenceAgent:
    """Active inference agent for UI exploration.

    Combines expected free energy (EFE) minimisation:

        G(a) = E_o[D_KL(q(s'|o) ‖ p(s'))] − E_o[H(o|s')]

    The first term is epistemic value (information gain) and the second
    is pragmatic value (goal achievement).

    References
    ----------
    - Friston, K. et al. (2017). Active inference: a process theory.
    - Da Costa, L. et al. (2020). Active inference on discrete state spaces.
    """

    def __init__(
        self,
        pomdp: POMDP,
        prior_preferences: Optional[dict[str, float]] = None,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 1.0,
    ) -> None:
        self.pomdp = pomdp
        self.updater = BeliefUpdater(pomdp)
        self.prior_preferences = prior_preferences or {}
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight

    def expected_free_energy(
        self, belief: BeliefState, action_id: str
    ) -> float:
        """Compute expected free energy G(a) for an action.

        G(a) = −ε · IG(b, a) − π · R(b, a)

        where ε is epistemic weight and π is pragmatic weight.

        Lower G → preferred action.

        Parameters
        ----------
        belief : BeliefState
        action_id : str

        Returns
        -------
        float
            Expected free energy (lower is better).
        """
        # Epistemic value: expected information gain
        current_h = belief_entropy(belief)
        expected_h = 0.0
        for oid in self.pomdp.observation_ids:
            p_obs = self.updater.observation_likelihood(belief, action_id, oid)
            if p_obs < 1e-10:
                continue
            b_prime = self.updater.update(belief, action_id, oid)
            expected_h += p_obs * belief_entropy(b_prime)
        epistemic = current_h - expected_h

        # Pragmatic value: expected reward under prior preferences
        pragmatic = 0.0
        for oid in self.pomdp.observation_ids:
            p_obs = self.updater.observation_likelihood(belief, action_id, oid)
            if p_obs < 1e-10:
                continue
            pref = self.prior_preferences.get(oid, 0.0)
            pragmatic += p_obs * pref

        # Combine (negate because lower G is better)
        return -(self.epistemic_weight * epistemic + self.pragmatic_weight * pragmatic)

    def action_probabilities(
        self, belief: BeliefState, precision: float = 1.0
    ) -> dict[str, float]:
        """Softmax policy over expected free energy.

        π(a) ∝ exp(−precision · G(a))

        Parameters
        ----------
        belief : BeliefState
        precision : float
            Inverse temperature.

        Returns
        -------
        dict[str, float]
        """
        G = {}
        for aid in self.pomdp.action_ids:
            G[aid] = self.expected_free_energy(belief, aid)

        actions = list(G.keys())
        g_values = np.array([G[a] for a in actions], dtype=np.float64)

        # σ(−G) = softmax over negative G (prefer low G)
        neg_g = -precision * g_values
        neg_g -= neg_g.max()
        exp_g = np.exp(neg_g)
        probs = exp_g / exp_g.sum()

        return {a: float(p) for a, p in zip(actions, probs)}

    def select_action(
        self,
        belief: BeliefState,
        precision: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """Sample an action from the active inference policy.

        Parameters
        ----------
        belief : BeliefState
        precision : float
        rng : np.random.Generator, optional

        Returns
        -------
        str
        """
        rng = rng or np.random.default_rng()
        probs = self.action_probabilities(belief, precision)
        if not probs:
            return ""
        actions = list(probs.keys())
        p = np.array([probs[a] for a in actions], dtype=np.float64)
        idx = int(rng.choice(len(actions), p=p))
        return actions[idx]
