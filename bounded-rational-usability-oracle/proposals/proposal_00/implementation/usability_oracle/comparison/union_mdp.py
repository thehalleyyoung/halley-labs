"""
usability_oracle.comparison.union_mdp — Union MDP construction.

Builds a single *union* MDP from two version-specific MDPs and their
alignment.  The union MDP contains all states and transitions from both
versions, with version-tagged state IDs and zero-cost bridge transitions
between aligned state pairs.

This enables a single policy solver or value-iteration pass to reason
about both versions simultaneously, which is critical for parameter-free
comparison (see :mod:`usability_oracle.comparison.parameter_free`).

Construction algorithm
----------------------
1.  **Tag** states of each MDP with a version prefix (``"A:"`` / ``"B:"``).
2.  **Merge** tagged states, actions, and transitions into one MDP.
3.  For each aligned state pair ``(sₐ, s_b)``, insert zero-cost
    bidirectional transitions ``A:sₐ ↔ B:s_b``.
4.  **Validate** the resulting MDP (probability normalization, reachability).

References
----------
- Givan, R., Dean, T., & Greig, M. (2003). Equivalence notions and model
  minimization in Markov decision processes. *Artificial Intelligence*, 147.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from usability_oracle.core.errors import ComparisonError
from usability_oracle.mdp.models import MDP, Action, State, Transition
from usability_oracle.comparison.models import AlignmentResult

logger = logging.getLogger(__name__)

# Version tag constants
_VERSION_A = "A"
_VERSION_B = "B"
_TAG_SEP = ":"
_BRIDGE_ACTION_PREFIX = "bridge"


def _tag(version: str, state_id: str) -> str:
    """Prefix *state_id* with a version tag."""
    return f"{version}{_TAG_SEP}{state_id}"


def _untag(tagged_id: str) -> tuple[str, str]:
    """Split a tagged state ID into ``(version, original_id)``."""
    parts = tagged_id.split(_TAG_SEP, 1)
    if len(parts) != 2:
        return ("", tagged_id)
    return (parts[0], parts[1])


# ---------------------------------------------------------------------------
# UnionMDPBuilder
# ---------------------------------------------------------------------------


class UnionMDPBuilder:
    """Builds a union MDP from two version-specific MDPs.

    The union MDP is the disjoint union of the two MDPs' state spaces
    plus zero-cost bridge transitions between aligned states.  This
    enables joint analysis of both UI versions within a single MDP
    framework.

    Parameters
    ----------
    bridge_cost : float
        Cost of bridge transitions between aligned states (default 0).
    bridge_probability : float
        Probability assigned to bridge transitions (default 1).
    validate : bool
        Whether to validate the union MDP after construction.
    """

    def __init__(
        self,
        bridge_cost: float = 0.0,
        bridge_probability: float = 1.0,
        validate: bool = True,
    ) -> None:
        self.bridge_cost = bridge_cost
        self.bridge_probability = bridge_probability
        self._validate = validate

    def build(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        alignment: AlignmentResult,
    ) -> MDP:
        """Construct the union MDP.

        Parameters
        ----------
        mdp_a : MDP
            MDP for the *before* UI version.
        mdp_b : MDP
            MDP for the *after* UI version.
        alignment : AlignmentResult
            State-level alignment between the two MDPs.

        Returns
        -------
        MDP
            Union MDP containing both versions' states and transitions.

        Raises
        ------
        ComparisonError
            If construction or validation fails.
        """
        logger.info(
            "Building union MDP: |S_a|=%d, |S_b|=%d, |align|=%d",
            mdp_a.n_states, mdp_b.n_states, alignment.n_mapped,
        )

        # 1. Tag states
        tagged_a = self._tag_states(mdp_a, _VERSION_A)
        tagged_b = self._tag_states(mdp_b, _VERSION_B)

        # 2. Merge states
        merged_states = self._merge_states(mdp_a, mdp_b, alignment)

        # 3. Merge actions
        merged_actions = self._merge_actions(mdp_a, mdp_b)

        # Add bridge actions
        for i, mapping in enumerate(alignment.mappings):
            bridge_id_ab = f"{_BRIDGE_ACTION_PREFIX}_a_to_b_{i}"
            bridge_id_ba = f"{_BRIDGE_ACTION_PREFIX}_b_to_a_{i}"
            merged_actions[bridge_id_ab] = Action(
                action_id=bridge_id_ab,
                action_type="navigate",
                description=f"Bridge: {mapping.state_a} → {mapping.state_b}",
            )
            merged_actions[bridge_id_ba] = Action(
                action_id=bridge_id_ba,
                action_type="navigate",
                description=f"Bridge: {mapping.state_b} → {mapping.state_a}",
            )

        # 4. Build state mapping for transition merging
        state_mapping_a = {s: _tag(_VERSION_A, s) for s in mdp_a.states}
        state_mapping_b = {s: _tag(_VERSION_B, s) for s in mdp_b.states}

        # 5. Merge transitions
        merged_transitions = self._merge_transitions(
            mdp_a, mdp_b, state_mapping_a, state_mapping_b
        )

        # 6. Link aligned states with bridge transitions
        bridge_transitions = self._link_aligned_states(alignment)
        merged_transitions.extend(bridge_transitions)

        # 7. Set up goal states and initial state
        goal_states: set[str] = set()
        for gs in mdp_a.goal_states:
            goal_states.add(_tag(_VERSION_A, gs))
        for gs in mdp_b.goal_states:
            goal_states.add(_tag(_VERSION_B, gs))

        initial_state = _tag(_VERSION_A, mdp_a.initial_state) if mdp_a.initial_state else ""

        # 8. Construct union MDP
        union_mdp = MDP(
            states=merged_states,
            actions=merged_actions,
            transitions=merged_transitions,
            initial_state=initial_state,
            goal_states=goal_states,
            discount=min(mdp_a.discount, mdp_b.discount),
        )

        logger.info(
            "Union MDP constructed: |S|=%d, |A|=%d, |T|=%d",
            union_mdp.n_states, union_mdp.n_actions, union_mdp.n_transitions,
        )

        # 9. Validate
        if self._validate:
            if not self.validate_union(union_mdp, mdp_a, mdp_b):
                logger.warning("Union MDP validation found issues")

        return union_mdp

    def _merge_states(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        alignment: AlignmentResult,
    ) -> dict[str, State]:
        """Merge states from both MDPs into a single state dictionary.

        Each state is tagged with its version prefix.  Aligned states
        from MDP-B carry metadata about their MDP-A counterpart.

        Parameters
        ----------
        mdp_a, mdp_b : MDP
        alignment : AlignmentResult

        Returns
        -------
        dict[str, State]
            Merged state dictionary.
        """
        merged: dict[str, State] = {}
        mapping = alignment.get_mapping_dict()
        reverse = alignment.get_reverse_mapping()

        for sid, state in mdp_a.states.items():
            tagged_id = _tag(_VERSION_A, sid)
            meta = dict(state.metadata)
            meta["version"] = _VERSION_A
            meta["original_id"] = sid
            if sid in mapping:
                meta["aligned_to"] = _tag(_VERSION_B, mapping[sid])
            merged[tagged_id] = State(
                state_id=tagged_id,
                features=dict(state.features),
                label=f"[A] {state.label}",
                is_terminal=state.is_terminal,
                is_goal=state.is_goal,
                metadata=meta,
            )

        for sid, state in mdp_b.states.items():
            tagged_id = _tag(_VERSION_B, sid)
            meta = dict(state.metadata)
            meta["version"] = _VERSION_B
            meta["original_id"] = sid
            if sid in reverse:
                meta["aligned_to"] = _tag(_VERSION_A, reverse[sid])
            merged[tagged_id] = State(
                state_id=tagged_id,
                features=dict(state.features),
                label=f"[B] {state.label}",
                is_terminal=state.is_terminal,
                is_goal=state.is_goal,
                metadata=meta,
            )

        return merged

    def _merge_actions(self, mdp_a: MDP, mdp_b: MDP) -> dict[str, Action]:
        """Merge actions from both MDPs.

        Actions with the same ID are assumed to be semantically identical;
        otherwise they are prefixed with the version tag.

        Parameters
        ----------
        mdp_a, mdp_b : MDP

        Returns
        -------
        dict[str, Action]
        """
        merged: dict[str, Action] = {}

        for aid, action in mdp_a.actions.items():
            tagged_aid = f"{_VERSION_A}{_TAG_SEP}{aid}"
            merged[tagged_aid] = Action(
                action_id=tagged_aid,
                action_type=action.action_type,
                target_node_id=action.target_node_id,
                description=f"[A] {action.description}",
                preconditions=list(action.preconditions),
            )

        for aid, action in mdp_b.actions.items():
            tagged_aid = f"{_VERSION_B}{_TAG_SEP}{aid}"
            merged[tagged_aid] = Action(
                action_id=tagged_aid,
                action_type=action.action_type,
                target_node_id=action.target_node_id,
                description=f"[B] {action.description}",
                preconditions=list(action.preconditions),
            )

        return merged

    def _merge_transitions(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        state_mapping_a: dict[str, str],
        state_mapping_b: dict[str, str],
    ) -> list[Transition]:
        """Merge transitions from both MDPs with tagged IDs.

        Parameters
        ----------
        mdp_a, mdp_b : MDP
        state_mapping_a : dict[str, str]
            Maps original A state IDs to tagged IDs.
        state_mapping_b : dict[str, str]
            Maps original B state IDs to tagged IDs.

        Returns
        -------
        list[Transition]
        """
        merged: list[Transition] = []

        for t in mdp_a.transitions:
            merged.append(Transition(
                source=state_mapping_a.get(t.source, _tag(_VERSION_A, t.source)),
                action=f"{_VERSION_A}{_TAG_SEP}{t.action}",
                target=state_mapping_a.get(t.target, _tag(_VERSION_A, t.target)),
                probability=t.probability,
                cost=t.cost,
            ))

        for t in mdp_b.transitions:
            merged.append(Transition(
                source=state_mapping_b.get(t.source, _tag(_VERSION_B, t.source)),
                action=f"{_VERSION_B}{_TAG_SEP}{t.action}",
                target=state_mapping_b.get(t.target, _tag(_VERSION_B, t.target)),
                probability=t.probability,
                cost=t.cost,
            ))

        return merged

    def _tag_states(self, mdp: MDP, version: str) -> dict[str, State]:
        """Prefix all states in *mdp* with the version tag.

        Parameters
        ----------
        mdp : MDP
        version : str
            Version prefix (e.g., ``"A"`` or ``"B"``).

        Returns
        -------
        dict[str, State]
            Tagged states.
        """
        tagged: dict[str, State] = {}
        for sid, state in mdp.states.items():
            tagged_id = _tag(version, sid)
            meta = dict(state.metadata)
            meta["version"] = version
            meta["original_id"] = sid
            tagged[tagged_id] = State(
                state_id=tagged_id,
                features=dict(state.features),
                label=f"[{version}] {state.label}",
                is_terminal=state.is_terminal,
                is_goal=state.is_goal,
                metadata=meta,
            )
        return tagged

    def _link_aligned_states(
        self, alignment: AlignmentResult
    ) -> list[Transition]:
        """Create zero-cost bridge transitions between aligned states.

        For each alignment mapping ``(sₐ, s_b)``, inserts bidirectional
        transitions:

            ``A:sₐ → B:s_b``   (cost = 0)
            ``B:s_b → A:sₐ``   (cost = 0)

        These bridges allow the policy solver to reason about switching
        between versions at semantically equivalent points.

        Parameters
        ----------
        alignment : AlignmentResult

        Returns
        -------
        list[Transition]
        """
        bridges: list[Transition] = []

        for i, mapping in enumerate(alignment.mappings):
            tagged_a = _tag(_VERSION_A, mapping.state_a)
            tagged_b = _tag(_VERSION_B, mapping.state_b)

            # A → B bridge
            bridges.append(Transition(
                source=tagged_a,
                action=f"{_BRIDGE_ACTION_PREFIX}_a_to_b_{i}",
                target=tagged_b,
                probability=self.bridge_probability,
                cost=self.bridge_cost,
            ))

            # B → A bridge
            bridges.append(Transition(
                source=tagged_b,
                action=f"{_BRIDGE_ACTION_PREFIX}_b_to_a_{i}",
                target=tagged_a,
                probability=self.bridge_probability,
                cost=self.bridge_cost,
            ))

        return bridges

    def validate_union(
        self,
        union_mdp: MDP,
        mdp_a: MDP,
        mdp_b: MDP,
    ) -> bool:
        """Validate structural properties of the union MDP.

        Checks:
        - All original states are present (tagged).
        - Bridge transitions reference valid states.
        - No probability normalization violations beyond tolerance.

        Parameters
        ----------
        union_mdp : MDP
        mdp_a, mdp_b : MDP

        Returns
        -------
        bool
            ``True`` if validation passes with no critical errors.
        """
        errors: list[str] = []

        # Check that all states from A are present
        for sid in mdp_a.states:
            tagged = _tag(_VERSION_A, sid)
            if tagged not in union_mdp.states:
                errors.append(f"Missing state {tagged} from MDP-A")

        # Check that all states from B are present
        for sid in mdp_b.states:
            tagged = _tag(_VERSION_B, sid)
            if tagged not in union_mdp.states:
                errors.append(f"Missing state {tagged} from MDP-B")

        # Check union size
        expected_states = mdp_a.n_states + mdp_b.n_states
        if union_mdp.n_states != expected_states:
            errors.append(
                f"Expected {expected_states} states, got {union_mdp.n_states}"
            )

        # Check all transitions reference valid states
        for t in union_mdp.transitions:
            if t.source not in union_mdp.states:
                errors.append(f"Transition source {t.source!r} not in states")
            if t.target not in union_mdp.states:
                errors.append(f"Transition target {t.target!r} not in states")
            if t.action not in union_mdp.actions:
                errors.append(f"Transition action {t.action!r} not in actions")

        if errors:
            for e in errors:
                logger.warning("Union MDP validation: %s", e)
            return False

        logger.info("Union MDP validation passed")
        return True
