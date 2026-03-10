"""
usability_oracle.bisimulation.simulation_relation — Simulation relations.

Implements simulation preorders and distances for MDPs.  A simulation
relation R ⊆ S × S captures an asymmetric notion: s₁ R s₂ means "s₂
can simulate s₁", i.e. any behaviour observable from s₁ can be matched
by s₂.

Key capabilities:
  - Simulation preorder via greatest fixed point
  - Ready simulation (matching available action sets)
  - Failure simulation (matching refusal sets)
  - Probabilistic simulation
  - Game-based characterisation
  - Simulation distance
  - Maximal simulation relation

References
----------
- Park, D. (1981). Concurrency and automata on infinite sequences. *TCS*.
- Baier, C. et al. (2000). Simulation for continuous-time Markov chains.
  *CONCUR*.
- de Alfaro, L. et al. (2004). Game relations and metrics. *LICS*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bisimulation.models import (
    CognitiveDistanceMatrix,
    Partition,
)
from usability_oracle.bisimulation.probabilistic import kantorovich_distance
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation relation (greatest fixed point)
# ---------------------------------------------------------------------------

@dataclass
class SimulationRelation:
    """Compute the maximal simulation relation over an MDP.

    The simulation preorder is the greatest fixed point of:

        s₁ ≤ s₂  iff  for every action a available at s₁, a is also
        available at s₂ and for every target block B under the current
        relation, Σ_{s'∈B} T(s'|s₁,a) ≤ Σ_{s'∈B} T(s'|s₂,a)  (lifted).

    For deterministic MDPs this simplifies to: every transition from s₁
    can be matched by a transition from s₂ to a simulating state.

    Parameters
    ----------
    max_iterations : int
        Maximum number of refinement rounds.
    check_costs : bool
        If True, also require cost(s₁, a) ≥ cost(s₂, a) (s₂ is at
        least as cheap).
    """

    max_iterations: int = 500
    check_costs: bool = False

    def compute(self, mdp: MDP) -> np.ndarray:
        """Compute the maximal simulation relation.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        np.ndarray
            Boolean matrix R where R[i,j]=True means s_j simulates s_i.
            Shape (n, n).
        """
        state_ids = sorted(mdp.states.keys())
        n = len(state_ids)
        state_idx = {sid: i for i, sid in enumerate(state_ids)}

        # Initialise: R = full relation (all True) minus obvious failures
        R = np.ones((n, n), dtype=bool)

        # Remove pairs where action sets are incompatible
        for i, s1 in enumerate(state_ids):
            actions_1 = set(mdp.get_actions(s1))
            for j, s2 in enumerate(state_ids):
                actions_2 = set(mdp.get_actions(s2))
                if not actions_1.issubset(actions_2):
                    R[i, j] = False

        # Remove pairs with different goal/terminal status
        for i, s1 in enumerate(state_ids):
            st1 = mdp.states[s1]
            for j, s2 in enumerate(state_ids):
                st2 = mdp.states[s2]
                if st1.is_goal != st2.is_goal:
                    R[i, j] = False
                if st1.is_terminal and not st2.is_terminal:
                    R[i, j] = False

        # Fixed-point iteration
        for iteration in range(self.max_iterations):
            changed = False

            for i, s1 in enumerate(state_ids):
                for j, s2 in enumerate(state_ids):
                    if not R[i, j] or i == j:
                        continue

                    if not self._check_simulation(
                        s1, s2, mdp, R, state_ids, state_idx,
                    ):
                        R[i, j] = False
                        changed = True

            if not changed:
                logger.info(
                    "Simulation relation converged after %d iterations",
                    iteration + 1,
                )
                break

        return R

    def _check_simulation(
        self,
        s1: str,
        s2: str,
        mdp: MDP,
        R: np.ndarray,
        state_ids: list[str],
        state_idx: dict[str, int],
    ) -> bool:
        """Check if s2 still simulates s1 under the current relation R."""
        for aid in mdp.get_actions(s1):
            if aid not in mdp.get_actions(s2):
                return False

            if self.check_costs:
                c1 = sum(p * c for _, p, c in mdp.get_transitions(s1, aid))
                c2 = sum(p * c for _, p, c in mdp.get_transitions(s2, aid))
                if c2 > c1 + 1e-10:
                    return False

            # For each successor of s1, there must be a simulating
            # successor of s2
            for target1, prob1, _ in mdp.get_transitions(s1, aid):
                if prob1 < 1e-12:
                    continue
                i1 = state_idx.get(target1)
                if i1 is None:
                    continue

                # Check that some successor of s2 under aid simulates target1
                matched = False
                for target2, prob2, _ in mdp.get_transitions(s2, aid):
                    if prob2 < 1e-12:
                        continue
                    i2 = state_idx.get(target2)
                    if i2 is not None and R[i1, i2]:
                        matched = True
                        break

                if not matched:
                    return False

        return True

    def is_simulated_by(
        self,
        s1: str,
        s2: str,
        mdp: MDP,
    ) -> bool:
        """Check whether s2 simulates s1.

        Parameters
        ----------
        s1, s2 : str
            State identifiers.
        mdp : MDP

        Returns
        -------
        bool
            True if s2 simulates s1.
        """
        R = self.compute(mdp)
        state_ids = sorted(mdp.states.keys())
        state_idx = {sid: i for i, sid in enumerate(state_ids)}
        i = state_idx.get(s1)
        j = state_idx.get(s2)
        if i is None or j is None:
            return False
        return bool(R[i, j])


# ---------------------------------------------------------------------------
# Ready simulation
# ---------------------------------------------------------------------------

@dataclass
class ReadySimulation:
    """Ready simulation: simulation that also matches the *ready set*.

    s₁ ≤_ready s₂ iff s₁ ≤ s₂ and actions(s₁) = actions(s₂).

    This is strictly finer than plain simulation and coarser than
    bisimulation.

    Parameters
    ----------
    max_iterations : int
        Maximum refinement rounds.
    """

    max_iterations: int = 500

    def compute(self, mdp: MDP) -> np.ndarray:
        """Compute the maximal ready simulation relation.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        np.ndarray
            Boolean matrix R[i,j] = True means s_j ready-simulates s_i.
        """
        sim = SimulationRelation(max_iterations=self.max_iterations)
        R = sim.compute(mdp)

        state_ids = sorted(mdp.states.keys())
        n = len(state_ids)

        # Additionally require identical action sets
        for i, s1 in enumerate(state_ids):
            actions_1 = set(mdp.get_actions(s1))
            for j, s2 in enumerate(state_ids):
                if not R[i, j]:
                    continue
                actions_2 = set(mdp.get_actions(s2))
                if actions_1 != actions_2:
                    R[i, j] = False

        return R


# ---------------------------------------------------------------------------
# Failure simulation
# ---------------------------------------------------------------------------

@dataclass
class FailureSimulation:
    """Failure simulation: additionally matches refusal sets.

    s₁ ≤_fail s₂ iff s₁ ≤ s₂ and for every set of actions X that s₁
    can refuse (none of X is available), s₂ can also refuse X.

    For finite-action MDPs this is equivalent to: actions(s₂) ⊆ actions(s₁),
    combined with the simulation condition.

    Parameters
    ----------
    max_iterations : int
        Maximum refinement rounds.
    """

    max_iterations: int = 500

    def compute(self, mdp: MDP) -> np.ndarray:
        """Compute the maximal failure simulation relation.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        np.ndarray
            Boolean matrix.
        """
        sim = SimulationRelation(max_iterations=self.max_iterations)
        R = sim.compute(mdp)

        state_ids = sorted(mdp.states.keys())

        # Failure condition: actions(s₂) ⊆ actions(s₁)
        for i, s1 in enumerate(state_ids):
            actions_1 = set(mdp.get_actions(s1))
            for j, s2 in enumerate(state_ids):
                if not R[i, j]:
                    continue
                actions_2 = set(mdp.get_actions(s2))
                if not actions_2.issubset(actions_1):
                    R[i, j] = False

        return R


# ---------------------------------------------------------------------------
# Probabilistic simulation
# ---------------------------------------------------------------------------

@dataclass
class ProbabilisticSimulation:
    """Probabilistic simulation via weight functions.

    s₁ ≤_prob s₂ iff for every action a and every equivalence class C
    (under the current relation), the probability mass from s₁ reaching
    C is ≤ the mass from s₂ reaching states that simulate C members.

    Parameters
    ----------
    max_iterations : int
        Maximum fixed-point iterations.
    tolerance : float
        Numerical tolerance.
    """

    max_iterations: int = 500
    tolerance: float = 1e-10

    def compute(self, mdp: MDP) -> np.ndarray:
        """Compute the maximal probabilistic simulation relation.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        np.ndarray
            Boolean matrix R where R[i,j] means s_j prob-simulates s_i.
        """
        state_ids = sorted(mdp.states.keys())
        n = len(state_ids)
        state_idx = {sid: i for i, sid in enumerate(state_ids)}

        R = np.ones((n, n), dtype=bool)

        # Initialise with action-set compatibility
        for i, s1 in enumerate(state_ids):
            a1 = set(mdp.get_actions(s1))
            for j, s2 in enumerate(state_ids):
                a2 = set(mdp.get_actions(s2))
                if not a1.issubset(a2):
                    R[i, j] = False

        for iteration in range(self.max_iterations):
            changed = False

            for i, s1 in enumerate(state_ids):
                for j, s2 in enumerate(state_ids):
                    if not R[i, j] or i == j:
                        continue

                    if not self._check_prob_simulation(
                        s1, s2, mdp, R, state_ids, state_idx,
                    ):
                        R[i, j] = False
                        changed = True

            if not changed:
                logger.info(
                    "Probabilistic simulation converged after %d iterations",
                    iteration + 1,
                )
                break

        return R

    def _check_prob_simulation(
        self,
        s1: str,
        s2: str,
        mdp: MDP,
        R: np.ndarray,
        state_ids: list[str],
        state_idx: dict[str, int],
    ) -> bool:
        """Check probabilistic simulation condition."""
        n = len(state_ids)

        for aid in mdp.get_actions(s1):
            if aid not in mdp.get_actions(s2):
                return False

            # Build distributions
            p = np.zeros(n, dtype=np.float64)
            q = np.zeros(n, dtype=np.float64)

            for target, prob, _ in mdp.get_transitions(s1, aid):
                idx = state_idx.get(target)
                if idx is not None:
                    p[idx] += prob
            for target, prob, _ in mdp.get_transitions(s2, aid):
                idx = state_idx.get(target)
                if idx is not None:
                    q[idx] += prob

            # Check: for each state s' reachable from s1, the total
            # probability of reaching states that simulate s' from s2
            # must be at least p[s']
            for k in range(n):
                if p[k] < self.tolerance:
                    continue
                # Total probability of reaching a state that simulates k
                simulating_prob = sum(
                    q[m] for m in range(n) if R[k, m]
                )
                if simulating_prob < p[k] - self.tolerance:
                    return False

        return True


# ---------------------------------------------------------------------------
# Simulation distance
# ---------------------------------------------------------------------------

@dataclass
class SimulationDistance:
    """Quantitative simulation distance.

    Computes an asymmetric distance d_sim(s₁, s₂) that quantifies how
    far s₂ is from being able to simulate s₁.  d_sim(s₁, s₂) = 0 iff
    s₂ simulates s₁.

    Parameters
    ----------
    discount : float
        Discount factor for the distance computation.
    max_iterations : int
        Maximum fixed-point iterations.
    tolerance : float
        Convergence tolerance.
    """

    discount: float = 0.99
    max_iterations: int = 200
    tolerance: float = 1e-6

    def compute(self, mdp: MDP) -> CognitiveDistanceMatrix:
        """Compute the simulation distance matrix.

        Note: the resulting matrix is asymmetric. Entry [i,j] represents
        the cost for s_j to simulate s_i.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        CognitiveDistanceMatrix
            Asymmetric distance matrix (d[i,j] ≠ d[j,i] in general).
        """
        state_ids = sorted(mdp.states.keys())
        n = len(state_ids)
        state_idx = {sid: i for i, sid in enumerate(state_ids)}
        gamma = self.discount

        # Initialise: d(s, s) = 0, d(s, t) = 1
        d = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(d, 0.0)

        for iteration in range(self.max_iterations):
            d_new = np.zeros_like(d)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    s1, s2 = state_ids[i], state_ids[j]
                    d_new[i, j] = self._update(
                        s1, s2, mdp, d, state_idx, gamma,
                    )

            delta = float(np.max(np.abs(d_new - d)))
            d = d_new

            if delta < self.tolerance:
                logger.info(
                    "Simulation distance converged after %d iterations "
                    "(Δ=%.2e)", iteration + 1, delta,
                )
                break

        return CognitiveDistanceMatrix(distances=d, state_ids=state_ids)

    def _update(
        self,
        s1: str,
        s2: str,
        mdp: MDP,
        d: np.ndarray,
        state_idx: dict[str, int],
        gamma: float,
    ) -> float:
        """One Bellman-style update for the simulation distance."""
        actions_1 = mdp.get_actions(s1)
        actions_2 = mdp.get_actions(s2)

        if not actions_1:
            return 0.0

        max_dist = 0.0
        n = d.shape[0]

        for aid in actions_1:
            if aid not in actions_2:
                max_dist = max(max_dist, 1.0)
                continue

            # Cost mismatch
            c1 = sum(p * c for _, p, c in mdp.get_transitions(s1, aid))
            c2 = sum(p * c for _, p, c in mdp.get_transitions(s2, aid))
            cost_penalty = max(0.0, c2 - c1)

            # Transition distribution mismatch (directional Kantorovich)
            p_dist = np.zeros(n, dtype=np.float64)
            q_dist = np.zeros(n, dtype=np.float64)
            for target, prob, _ in mdp.get_transitions(s1, aid):
                idx = state_idx.get(target)
                if idx is not None:
                    p_dist[idx] += prob
            for target, prob, _ in mdp.get_transitions(s2, aid):
                idx = state_idx.get(target)
                if idx is not None:
                    q_dist[idx] += prob

            w = kantorovich_distance(p_dist, q_dist, d)
            max_dist = max(max_dist, cost_penalty + gamma * w)

        return min(max_dist, 1.0)


# ---------------------------------------------------------------------------
# Game-based characterisation
# ---------------------------------------------------------------------------

@dataclass
class SimulationGame:
    """Game-based characterisation of simulation.

    Models simulation checking as a two-player game between Spoiler
    (trying to show non-simulation) and Duplicator (trying to maintain
    the simulation).

    A simulation exists iff Duplicator has a winning strategy.

    Parameters
    ----------
    max_rounds : int
        Maximum number of game rounds.
    """

    max_rounds: int = 100

    def play(
        self, s1: str, s2: str, mdp: MDP,
    ) -> tuple[bool, list[dict]]:
        """Play the simulation game starting from (s1, s2).

        Duplicator wins iff s2 simulates s1.

        Parameters
        ----------
        s1 : str
            Spoiler's initial state.
        s2 : str
            Duplicator's initial state.
        mdp : MDP

        Returns
        -------
        tuple[bool, list[dict]]
            (duplicator_wins, game_trace) where game_trace records each
            round.
        """
        trace: list[dict] = []
        current_s1, current_s2 = s1, s2

        for round_num in range(self.max_rounds):
            actions_1 = mdp.get_actions(current_s1)
            actions_2 = set(mdp.get_actions(current_s2))

            if not actions_1:
                # s1 is stuck → Duplicator wins trivially
                trace.append({
                    "round": round_num,
                    "s1": current_s1,
                    "s2": current_s2,
                    "result": "duplicator_wins_stuck",
                })
                return True, trace

            # Spoiler picks the action hardest for Duplicator to match
            spoiler_action = None
            best_spoiler_advantage = -1.0

            for aid in actions_1:
                if aid not in actions_2:
                    trace.append({
                        "round": round_num,
                        "s1": current_s1,
                        "s2": current_s2,
                        "spoiler_action": aid,
                        "result": "spoiler_wins_no_action",
                    })
                    return False, trace

                # Evaluate difficulty for Duplicator
                targets_1 = mdp.get_transitions(current_s1, aid)
                targets_2 = mdp.get_transitions(current_s2, aid)
                advantage = len(targets_1) - len(targets_2)
                if advantage > best_spoiler_advantage:
                    best_spoiler_advantage = advantage
                    spoiler_action = aid

            if spoiler_action is None:
                spoiler_action = actions_1[0]

            # Spoiler moves s1
            trans_1 = mdp.get_transitions(current_s1, spoiler_action)
            if not trans_1:
                trace.append({
                    "round": round_num,
                    "s1": current_s1,
                    "s2": current_s2,
                    "action": spoiler_action,
                    "result": "duplicator_wins_no_transition",
                })
                return True, trace

            # Pick highest-probability successor for s1
            trans_1.sort(key=lambda t: -t[1])
            next_s1 = trans_1[0][0]

            # Duplicator must match with a successor of s2
            trans_2 = mdp.get_transitions(current_s2, spoiler_action)
            if not trans_2:
                trace.append({
                    "round": round_num,
                    "s1": current_s1,
                    "s2": current_s2,
                    "action": spoiler_action,
                    "result": "spoiler_wins_no_match",
                })
                return False, trace

            # Duplicator picks the best matching successor
            next_s2 = trans_2[0][0]  # heuristic: highest probability

            trace.append({
                "round": round_num,
                "s1": current_s1,
                "s2": current_s2,
                "action": spoiler_action,
                "next_s1": next_s1,
                "next_s2": next_s2,
            })

            current_s1 = next_s1
            current_s2 = next_s2

        # Reached max rounds without Spoiler winning → Duplicator wins
        return True, trace
