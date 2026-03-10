"""
usability_oracle.bisimulation.quotient — Quotient MDP construction.

Given a partition Π of the state space, the quotient MDP M/Π has:
  - Abstract states: one per block B ∈ Π.
  - Abstract transitions:
        T_abs(B_i, a, B_j) = Σ_{s ∈ B_i} w(s) · Σ_{s' ∈ B_j} T(s'|s, a)
    where w(s) = 1/|B_i| is the uniform weight within each block.
  - Abstract costs:
        c_abs(B_i, a) = Σ_{s ∈ B_i} w(s) · E_{T(·|s,a)}[c(s, a, s')]

The quotient preserves the optimal policy structure within the abstraction
error bound.

References
----------
- Li, Walsh & Littman (2006). Towards a unified theory of state
  abstraction for MDPs. *ISAIM*.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bisimulation.models import Partition
from usability_oracle.mdp.models import MDP, Action, State, Transition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QuotientMDPBuilder
# ---------------------------------------------------------------------------

@dataclass
class QuotientMDPBuilder:
    """Build a quotient (abstract) MDP from an MDP and a partition.

    The quotient MDP has one state per block and preserves the transition
    structure averaged over block members.

    Parameters
    ----------
    verify : bool
        If True, compute the abstraction error after building.
    state_prefix : str
        Prefix for abstract state ids (default ``"B"``).
    """

    verify: bool = True
    state_prefix: str = "B"

    # ── Public API --------------------------------------------------------

    def build(self, mdp: MDP, partition: Partition) -> MDP:
        """Construct the quotient MDP.

        Parameters
        ----------
        mdp : MDP
            The concrete (original) MDP.
        partition : Partition
            A valid partition of the MDP state space.

        Returns
        -------
        MDP
            The quotient MDP with |Π| states.

        Raises
        ------
        ValueError
            If the partition is invalid or inconsistent with the MDP.
        """
        if not partition.is_valid():
            raise ValueError("Partition is not valid")

        abstract_states = self._abstract_states(mdp, partition)
        abstract_actions = self._abstract_actions(mdp, partition)
        abstract_transitions = self._abstract_transitions(mdp, partition)

        # Determine initial and goal states
        initial_block = partition.state_to_block.get(mdp.initial_state, 0)
        initial_state = f"{self.state_prefix}{initial_block}"

        goal_states: set[str] = set()
        for gs in mdp.goal_states:
            block_idx = partition.state_to_block.get(gs)
            if block_idx is not None:
                goal_states.add(f"{self.state_prefix}{block_idx}")

        quotient = MDP(
            states=abstract_states,
            actions=abstract_actions,
            transitions=abstract_transitions,
            initial_state=initial_state,
            goal_states=goal_states,
            discount=mdp.discount,
        )

        logger.info(
            "Built quotient MDP: %d states, %d actions, %d transitions "
            "(compression %.1f%%)",
            quotient.n_states, quotient.n_actions, quotient.n_transitions,
            (1.0 - quotient.n_states / max(mdp.n_states, 1)) * 100,
        )

        return quotient

    # ── Abstract states ---------------------------------------------------

    def _abstract_states(
        self,
        mdp: MDP,
        partition: Partition,
    ) -> dict[str, State]:
        """Create one abstract state per block.

        Features are averaged over block members.

        Parameters
        ----------
        mdp : MDP
        partition : Partition

        Returns
        -------
        dict[str, State]
        """
        abstract_states: dict[str, State] = {}

        for idx, block in enumerate(partition.blocks):
            abs_id = f"{self.state_prefix}{idx}"
            features = self._compute_block_features(mdp, block)

            # A block is terminal/goal if any member is
            is_terminal = any(
                mdp.states[s].is_terminal for s in block if s in mdp.states
            )
            is_goal = any(
                mdp.states[s].is_goal for s in block if s in mdp.states
            )

            # Label from first member
            first_member = sorted(block)[0] if block else ""
            member_state = mdp.states.get(first_member)
            label = f"Block {idx}"
            if member_state:
                label = f"Block {idx} ({member_state.label or first_member})"

            abstract_states[abs_id] = State(
                state_id=abs_id,
                features=features,
                label=label,
                is_terminal=is_terminal,
                is_goal=is_goal,
                metadata={
                    "block_index": idx,
                    "block_size": len(block),
                    "members": sorted(block),
                },
            )

        return abstract_states

    # ── Abstract actions --------------------------------------------------

    def _abstract_actions(
        self,
        mdp: MDP,
        partition: Partition,
    ) -> dict[str, Action]:
        """Collect all actions reachable from any block.

        In the quotient MDP, we keep the original action set but restrict
        availability to blocks whose members can take the action.

        Parameters
        ----------
        mdp : MDP
        partition : Partition

        Returns
        -------
        dict[str, Action]
        """
        used_actions: set[str] = set()
        for block in partition.blocks:
            for state in block:
                for aid in mdp.get_actions(state):
                    used_actions.add(aid)

        abstract_actions: dict[str, Action] = {}
        for aid in used_actions:
            original = mdp.actions.get(aid)
            if original is not None:
                abstract_actions[aid] = original
            else:
                abstract_actions[aid] = Action(action_id=aid, action_type="unknown")

        return abstract_actions

    # ── Abstract transitions ----------------------------------------------

    def _abstract_transitions(
        self,
        mdp: MDP,
        partition: Partition,
    ) -> list[Transition]:
        """Compute abstract transition probabilities.

        For each source block B_i and action a, the abstract transition is:

            T_abs(B_i, a, B_j) = (1/|B_i|) Σ_{s ∈ B_i} Σ_{s' ∈ B_j} T(s'|s, a)

        The cost is similarly averaged:

            c_abs(B_i, a, B_j) = (1/n_ij) Σ weighted costs

        where n_ij counts contributing transitions.

        Parameters
        ----------
        mdp : MDP
        partition : Partition

        Returns
        -------
        list[Transition]
        """
        transitions: list[Transition] = []

        for bi, block in enumerate(partition.blocks):
            block_size = len(block)
            if block_size == 0:
                continue

            # Accumulate probabilities and costs per (action, target_block)
            accum: dict[tuple[str, int], tuple[float, float, int]] = defaultdict(
                lambda: (0.0, 0.0, 0)
            )

            for state in block:
                for aid in mdp.get_actions(state):
                    for target, prob, cost in mdp.get_transitions(state, aid):
                        bj = partition.state_to_block.get(target)
                        if bj is None:
                            continue
                        key = (aid, bj)
                        old_prob, old_cost, old_count = accum[key]
                        accum[key] = (
                            old_prob + prob / block_size,
                            old_cost + cost * prob / block_size,
                            old_count + 1,
                        )

            # Create transitions
            abs_source = f"{self.state_prefix}{bi}"
            for (aid, bj), (total_prob, total_cost, count) in accum.items():
                if total_prob < 1e-12:
                    continue
                abs_target = f"{self.state_prefix}{bj}"
                avg_cost = total_cost / total_prob if total_prob > 0 else 0.0
                transitions.append(
                    Transition(
                        source=abs_source,
                        action=aid,
                        target=abs_target,
                        probability=total_prob,
                        cost=avg_cost,
                    )
                )

        # Normalise probabilities per (source, action)
        transitions = self._normalise_transitions(transitions)
        return transitions

    def _normalise_transitions(
        self,
        transitions: list[Transition],
    ) -> list[Transition]:
        """Renormalise transition probabilities per (source, action) pair."""
        totals: dict[tuple[str, str], float] = defaultdict(float)
        for t in transitions:
            totals[(t.source, t.action)] += t.probability

        normalised: list[Transition] = []
        for t in transitions:
            total = totals[(t.source, t.action)]
            if total <= 0:
                continue
            normalised.append(
                Transition(
                    source=t.source,
                    action=t.action,
                    target=t.target,
                    probability=t.probability / total,
                    cost=t.cost,
                )
            )
        return normalised

    # ── Verification ------------------------------------------------------

    def verify_quotient(
        self,
        original: MDP,
        quotient: MDP,
        partition: Partition,
    ) -> float:
        """Verify the quotient MDP and compute the abstraction error.

        The abstraction error is estimated as the maximum difference in
        expected cost between the concrete and abstract transition models:

            error = max_{B_i, a} Σ_{B_j} |T_abs(B_i,a,B_j) - T̂(B_i,a,B_j)|

        where T̂ is recomputed from the partition.

        Parameters
        ----------
        original : MDP
        quotient : MDP
        partition : Partition

        Returns
        -------
        float
            Estimated abstraction error.
        """
        max_error = 0.0

        for bi, block in enumerate(partition.blocks):
            if not block:
                continue
            abs_source = f"{self.state_prefix}{bi}"

            # Get all actions for this abstract state
            abs_actions = quotient.get_actions(abs_source)

            for aid in abs_actions:
                # Quotient transitions
                abs_trans = quotient.get_transitions(abs_source, aid)
                abs_probs: dict[str, float] = {}
                for target, prob, _ in abs_trans:
                    abs_probs[target] = abs_probs.get(target, 0.0) + prob

                # Recompute from scratch
                recomputed: dict[str, float] = defaultdict(float)
                block_size = len(block)
                for state in block:
                    for target, prob, _ in original.get_transitions(state, aid):
                        bj = partition.state_to_block.get(target)
                        if bj is not None:
                            abs_target = f"{self.state_prefix}{bj}"
                            recomputed[abs_target] += prob / block_size

                # Compute error
                all_targets = set(abs_probs.keys()) | set(recomputed.keys())
                error = sum(
                    abs(abs_probs.get(t, 0.0) - recomputed.get(t, 0.0))
                    for t in all_targets
                )
                max_error = max(max_error, error)

        return max_error

    # ── Feature aggregation -----------------------------------------------

    def _compute_block_features(
        self,
        mdp: MDP,
        block: frozenset[str],
    ) -> dict[str, float]:
        """Compute averaged features for a block of states.

        For each numeric feature present in any member state, compute the
        mean, min, max, and standard deviation.

        Parameters
        ----------
        mdp : MDP
        block : frozenset[str]

        Returns
        -------
        dict[str, float]
            Feature dictionary with keys like ``"feat_mean"``, ``"feat_std"``.
        """
        if not block:
            return {}

        # Collect all feature values
        feature_values: dict[str, list[float]] = defaultdict(list)
        for state_id in block:
            state = mdp.states.get(state_id)
            if state is None:
                continue
            for key, value in state.features.items():
                feature_values[key].append(value)

        result: dict[str, float] = {"block_size": float(len(block))}
        for key, values in feature_values.items():
            arr = np.array(values)
            result[f"{key}_mean"] = float(np.mean(arr))
            result[f"{key}_std"] = float(np.std(arr))
            result[f"{key}_min"] = float(np.min(arr))
            result[f"{key}_max"] = float(np.max(arr))

        return result
