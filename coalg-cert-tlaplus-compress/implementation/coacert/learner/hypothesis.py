"""
Hypothesis coalgebra construction from a closed, consistent observation table.

Given a closed and consistent observation table the hypothesis builder
constructs an F-coalgebra whose states are the equivalence classes of
the short rows.  The structure map is derived from the cell values, and
transitions correspond to the table extensions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from .observation_table import (
    AccessSequence,
    Observation,
    ObservationTable,
    RowSignature,
    Suffix,
)
from .equivalence_oracle import HypothesisInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hypothesis state
# ---------------------------------------------------------------------------

@dataclass
class HypothesisState:
    """A state in the hypothesis coalgebra."""

    name: str
    representative: AccessSequence
    members: List[AccessSequence] = field(default_factory=list)
    observation: Optional[Observation] = None
    transitions: Dict[str, str] = field(default_factory=dict)
    propositions: FrozenSet[str] = field(default_factory=frozenset)
    fairness_membership: Dict[int, Tuple[bool, bool]] = field(
        default_factory=dict
    )

    def __repr__(self) -> str:
        return f"HypState({self.name}, rep={self.representative})"


# ---------------------------------------------------------------------------
# Built hypothesis
# ---------------------------------------------------------------------------

class HypothesisCoalgebra(HypothesisInterface):
    """A concrete hypothesis coalgebra built from an observation table.

    Implements ``HypothesisInterface`` so it can be passed directly to
    the equivalence oracle.
    """

    def __init__(self) -> None:
        self._states: Dict[str, HypothesisState] = {}
        self._initial: Optional[str] = None
        self._actions: Set[str] = set()
        self._access_to_state: Dict[AccessSequence, str] = {}

    # -- HypothesisInterface ------------------------------------------------

    def initial_state(self) -> str:
        if self._initial is None:
            raise RuntimeError("Hypothesis has no initial state")
        return self._initial

    def transition(self, state: str, action: str) -> Optional[str]:
        hs = self._states.get(state)
        if hs is None:
            return None
        return hs.transitions.get(action)

    def observation_at(self, state: str) -> Observation:
        hs = self._states.get(state)
        if hs is None:
            return frozenset()
        return hs.observation if hs.observation is not None else frozenset()

    def states(self) -> Set[str]:
        return set(self._states.keys())

    def actions(self) -> Set[str]:
        return set(self._actions)

    # -- additional accessors -----------------------------------------------

    def state_for_access(self, seq: AccessSequence) -> Optional[str]:
        return self._access_to_state.get(seq)

    def get_state(self, name: str) -> Optional[HypothesisState]:
        return self._states.get(name)

    @property
    def state_count(self) -> int:
        return len(self._states)

    def transition_count(self) -> int:
        return sum(len(s.transitions) for s in self._states.values())

    # -- pretty print -------------------------------------------------------

    def pretty_print(self) -> str:
        lines = [f"Hypothesis coalgebra ({self.state_count} states):"]
        for name in sorted(self._states):
            hs = self._states[name]
            rep = ".".join(hs.representative) if hs.representative else "ε"
            trans = ", ".join(
                f"{a}→{t}" for a, t in sorted(hs.transitions.items())
            )
            lines.append(f"  {name} (rep={rep}): {trans}")
        return "\n".join(lines)

    # -- equality -----------------------------------------------------------

    def is_isomorphic_to(self, other: "HypothesisCoalgebra") -> bool:
        """Check structural isomorphism (same number of states, same
        transition structure up to renaming)."""
        if self.state_count != other.state_count:
            return False
        if self._actions != other._actions:
            return False

        # Try to build a bijection starting from initial states
        if self._initial is None or other._initial is None:
            return self._initial == other._initial

        mapping: Dict[str, str] = {}
        queue = [(self._initial, other._initial)]
        while queue:
            s1, s2 = queue.pop()
            if s1 in mapping:
                if mapping[s1] != s2:
                    return False
                continue
            mapping[s1] = s2

            hs1 = self._states[s1]
            hs2 = other._states.get(s2)
            if hs2 is None:
                return False

            if hs1.observation != hs2.observation:
                return False

            for act in sorted(self._actions):
                t1 = hs1.transitions.get(act)
                t2 = hs2.transitions.get(act)
                if (t1 is None) != (t2 is None):
                    return False
                if t1 is not None and t2 is not None:
                    queue.append((t1, t2))

        return len(mapping) == self.state_count

    def __repr__(self) -> str:
        return (
            f"HypothesisCoalgebra(states={self.state_count}, "
            f"init={self._initial})"
        )


# ---------------------------------------------------------------------------
# Hypothesis builder
# ---------------------------------------------------------------------------

class HypothesisBuilder:
    """Construct hypothesis coalgebras from closed, consistent tables.

    Parameters
    ----------
    table : ObservationTable
        The observation table (must be closed and consistent).
    """

    def __init__(self, table: ObservationTable) -> None:
        self._table = table

    # -- main build ---------------------------------------------------------

    def build(self) -> HypothesisCoalgebra:
        """Build a hypothesis coalgebra from the current table.

        Requires the table to be closed and consistent.

        Returns
        -------
        HypothesisCoalgebra
            The hypothesis coalgebra.

        Raises
        ------
        ValueError
            If the table is not closed or not consistent.
        """
        if not self._table.is_closed():
            raise ValueError("Cannot build hypothesis: table is not closed")
        if not self._table.is_consistent():
            raise ValueError(
                "Cannot build hypothesis: table is not consistent"
            )

        hyp = HypothesisCoalgebra()
        hyp._actions = set(self._table.actions)

        # Step 1: identify equivalence classes of short rows
        classes = self._identify_classes()
        logger.info("Hypothesis has %d equivalence classes", len(classes))

        # Step 2: create states
        access_to_class: Dict[AccessSequence, str] = {}
        for cls_name, (representative, members) in classes.items():
            hs = HypothesisState(
                name=cls_name,
                representative=representative,
                members=members,
            )
            hyp._states[cls_name] = hs
            for m in members:
                access_to_class[m] = cls_name
                hyp._access_to_state[m] = cls_name

        # Step 3: set initial state (class of ε)
        init_class = access_to_class.get(())
        if init_class is None:
            raise ValueError("Initial state (ε) not in table short rows")
        hyp._initial = init_class

        # Step 4: set observations
        for cls_name, hs in hyp._states.items():
            obs = self._table.get_cell(hs.representative, ())
            if obs is not None:
                hs.observation = obs
                hs.propositions = self._extract_propositions(obs)
            else:
                # Use the first available column
                for col in self._table.columns:
                    obs = self._table.get_cell(hs.representative, col)
                    if obs is not None:
                        hs.observation = obs
                        hs.propositions = self._extract_propositions(obs)
                        break

        # Step 5: set transitions
        for cls_name, hs in hyp._states.items():
            for act in self._table.actions:
                ext = hs.representative + (act,)
                target_class = self._find_class_for_row(
                    ext, access_to_class, classes
                )
                if target_class is not None:
                    hs.transitions[act] = target_class

        # Step 6: set fairness membership
        self._set_fairness(hyp)

        logger.info(
            "Built hypothesis: %d states, %d transitions",
            hyp.state_count,
            hyp.transition_count(),
        )
        return hyp

    # -- equivalence class identification -----------------------------------

    def _identify_classes(
        self,
    ) -> Dict[str, Tuple[AccessSequence, List[AccessSequence]]]:
        """Group short rows into equivalence classes.

        Returns a dict mapping class name → (representative, [members]).
        """
        classes: Dict[str, Tuple[AccessSequence, List[AccessSequence]]] = {}
        class_sigs: Dict[str, RowSignature] = {}
        class_counter = 0

        for sr in self._table.short_rows:
            sig = self._table.row_signature(sr)
            found = False
            for cls_name, cls_sig in class_sigs.items():
                if sig.equivalent_to(cls_sig):
                    classes[cls_name][1].append(sr)
                    found = True
                    break
            if not found:
                cls_name = f"q{class_counter}"
                class_counter += 1
                classes[cls_name] = (sr, [sr])
                class_sigs[cls_name] = sig

        return classes

    def _find_class_for_row(
        self,
        row: AccessSequence,
        access_to_class: Dict[AccessSequence, str],
        classes: Dict[str, Tuple[AccessSequence, List[AccessSequence]]],
    ) -> Optional[str]:
        """Find the equivalence class matching *row*."""
        # Direct lookup first
        cls = access_to_class.get(row)
        if cls is not None:
            return cls

        # Signature comparison
        sig = self._table.row_signature(row)
        for cls_name, (rep, _) in classes.items():
            if self._table.row_signature(rep).equivalent_to(sig):
                return cls_name
        return None

    def _extract_propositions(self, obs: Observation) -> FrozenSet[str]:
        """Extract atomic propositions from an observation."""
        props: Set[str] = set()
        for item in obs:
            if isinstance(item, tuple) and len(item) >= 1:
                prop_set = item[0]
                if isinstance(prop_set, frozenset):
                    props |= prop_set
        return frozenset(props)

    # -- fairness -----------------------------------------------------------

    def _set_fairness(self, hyp: HypothesisCoalgebra) -> None:
        """Extract fairness membership from observations."""
        for cls_name, hs in hyp._states.items():
            if hs.observation is None:
                continue
            for item in hs.observation:
                if not isinstance(item, tuple) or len(item) < 2:
                    continue
                succ_tuple = item[1]
                if not isinstance(succ_tuple, tuple):
                    continue
                # Fairness is encoded in the observation structure;
                # extract if present in extended observation format
                # (beyond base props + successors)
                # For the base functor, fairness is determined by
                # state membership in B_i / G_i sets tracked externally

    # -- validation ---------------------------------------------------------

    def validate(self, hyp: HypothesisCoalgebra) -> List[str]:
        """Validate internal consistency of the hypothesis.

        Returns a list of issues (empty if valid).
        """
        issues: List[str] = []

        # Check initial state exists
        if hyp._initial is None:
            issues.append("No initial state")
        elif hyp._initial not in hyp._states:
            issues.append(f"Initial state {hyp._initial} not in states")

        # Check transitions point to valid states
        for name, hs in hyp._states.items():
            for act, target in hs.transitions.items():
                if target not in hyp._states:
                    issues.append(
                        f"State {name} action {act} → {target} (missing)"
                    )

        # Check all states are reachable
        reachable = self._reachable_states(hyp)
        unreachable = set(hyp._states.keys()) - reachable
        if unreachable:
            issues.append(f"Unreachable states: {unreachable}")

        # Check determinism: each action has at most one target
        for name, hs in hyp._states.items():
            for act in hyp._actions:
                targets = [
                    t for a, t in hs.transitions.items() if a == act
                ]
                if len(targets) > 1:
                    issues.append(
                        f"State {name} action {act} has {len(targets)} targets"
                    )

        return issues

    def _reachable_states(self, hyp: HypothesisCoalgebra) -> Set[str]:
        if hyp._initial is None:
            return set()
        visited: Set[str] = set()
        stack = [hyp._initial]
        while stack:
            s = stack.pop()
            if s in visited:
                continue
            visited.add(s)
            hs = hyp._states.get(s)
            if hs is not None:
                for t in hs.transitions.values():
                    if t not in visited:
                        stack.append(t)
        return visited

    # -- minimisation -------------------------------------------------------

    def minimize(self, hyp: HypothesisCoalgebra) -> HypothesisCoalgebra:
        """Minimise the hypothesis by merging bisimilar states.

        Uses partition refinement (à la Hopcroft) to compute the coarsest
        partition compatible with the observation and transition structure.
        """
        states = sorted(hyp._states.keys())
        if not states:
            return hyp

        # Initial partition by observation
        obs_to_block: Dict[Optional[Observation], int] = {}
        partition: Dict[str, int] = {}
        block_counter = 0
        for s in states:
            obs = hyp._states[s].observation
            if obs not in obs_to_block:
                obs_to_block[obs] = block_counter
                block_counter += 1
            partition[s] = obs_to_block[obs]

        # Refine until stable
        changed = True
        while changed:
            changed = False
            new_partition: Dict[str, int] = {}
            sig_to_block: Dict[Tuple[int, ...], int] = {}
            new_counter = 0

            for s in states:
                sig_parts = [partition[s]]
                for act in sorted(hyp._actions):
                    t = hyp._states[s].transitions.get(act)
                    sig_parts.append(
                        partition[t] if t is not None else -1
                    )
                sig = tuple(sig_parts)

                if sig not in sig_to_block:
                    sig_to_block[sig] = new_counter
                    new_counter += 1
                new_partition[s] = sig_to_block[sig]

            if new_partition != partition:
                changed = True
                partition = new_partition

        # Build minimised hypothesis
        block_rep: Dict[int, str] = {}
        for s in states:
            b = partition[s]
            if b not in block_rep:
                block_rep[b] = s

        minimised = HypothesisCoalgebra()
        minimised._actions = set(hyp._actions)

        for block_id, rep in block_rep.items():
            orig = hyp._states[rep]
            block_name = f"q{block_id}"
            members = [s for s, b in partition.items() if b == block_id]
            hs = HypothesisState(
                name=block_name,
                representative=orig.representative,
                members=[hyp._states[m].representative for m in members],
                observation=orig.observation,
                propositions=orig.propositions,
                fairness_membership=dict(orig.fairness_membership),
            )
            minimised._states[block_name] = hs

            for m in members:
                minimised._access_to_state[
                    hyp._states[m].representative
                ] = block_name

        # Set transitions
        for block_id, rep in block_rep.items():
            block_name = f"q{block_id}"
            hs = minimised._states[block_name]
            for act, target in hyp._states[rep].transitions.items():
                target_block = partition.get(target)
                if target_block is not None:
                    hs.transitions[act] = f"q{target_block}"

        # Initial state
        if hyp._initial is not None:
            init_block = partition.get(hyp._initial)
            if init_block is not None:
                minimised._initial = f"q{init_block}"

        logger.info(
            "Minimised hypothesis: %d → %d states",
            hyp.state_count,
            minimised.state_count,
        )
        return minimised

    def compare(
        self,
        h1: HypothesisCoalgebra,
        h2: HypothesisCoalgebra,
    ) -> Dict[str, Any]:
        """Compare two hypotheses structurally.

        Returns a dict with comparison results.
        """
        result: Dict[str, Any] = {
            "h1_states": h1.state_count,
            "h2_states": h2.state_count,
            "same_size": h1.state_count == h2.state_count,
            "same_actions": h1._actions == h2._actions,
            "isomorphic": h1.is_isomorphic_to(h2),
        }

        # Count transitions
        result["h1_transitions"] = h1.transition_count()
        result["h2_transitions"] = h2.transition_count()

        return result
