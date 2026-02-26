"""
Stuttering closure monad T for CoaCert-TLA.

The stutter monad T maps a transition system to its stutter-closure.
Two paths are stutter-equivalent if they differ only in the number of
consecutive repetitions of states with identical labeling.

T is an endofunctor on the category of coalgebras equipped with:
  - unit   η_X : X → T(X)
  - mult   μ_X : T(T(X)) → T(X)
satisfying monad laws.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stutter paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StutterPath:
    """A finite or infinite-prefix path in a transition system.

    Each element is a state name. Consecutive duplicates represent stuttering.
    """

    states: Tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.states)

    @property
    def is_empty(self) -> bool:
        return len(self.states) == 0

    def first(self) -> str:
        return self.states[0]

    def last(self) -> str:
        return self.states[-1]

    def stutter_free_core(self) -> "StutterPath":
        """Remove consecutive duplicates to get the stutter-free core."""
        if self.is_empty:
            return self
        core: List[str] = [self.states[0]]
        for s in self.states[1:]:
            if s != core[-1]:
                core.append(s)
        return StutterPath(states=tuple(core))

    def stutter_count(self) -> int:
        """Count the number of stutter steps (consecutive duplicates)."""
        count = 0
        for i in range(1, len(self.states)):
            if self.states[i] == self.states[i - 1]:
                count += 1
        return count

    def is_stutter_free(self) -> bool:
        return self.stutter_count() == 0

    def blocks(self) -> List[Tuple[str, int]]:
        """Decompose into maximal blocks of identical states.

        Returns list of (state, repetition_count).
        """
        if self.is_empty:
            return []
        result: List[Tuple[str, int]] = []
        current = self.states[0]
        count = 1
        for s in self.states[1:]:
            if s == current:
                count += 1
            else:
                result.append((current, count))
                current = s
                count = 1
        result.append((current, count))
        return result

    def extend_at(self, position: int, repetitions: int) -> "StutterPath":
        """Insert additional stuttering at a given position."""
        if position < 0 or position >= len(self.states):
            raise IndexError(f"Position {position} out of range")
        state = self.states[position]
        new_states = (
            self.states[:position]
            + (state,) * repetitions
            + self.states[position:]
        )
        return StutterPath(states=new_states)

    def contract_at(self, position: int) -> "StutterPath":
        """Remove one stutter step at a position (if it is a stutter)."""
        if position <= 0 or position >= len(self.states):
            raise IndexError(f"Position {position} out of range")
        if self.states[position] != self.states[position - 1]:
            return self  # not a stutter step
        new_states = self.states[:position] + self.states[position + 1:]
        return StutterPath(states=new_states)

    def __add__(self, other: "StutterPath") -> "StutterPath":
        return StutterPath(states=self.states + other.states)


def are_stutter_equivalent(p1: StutterPath, p2: StutterPath) -> bool:
    """Two paths are stutter-equivalent iff their stutter-free cores are equal."""
    return p1.stutter_free_core() == p2.stutter_free_core()


def all_stutter_extensions(
    path: StutterPath, max_stutter: int = 2
) -> List[StutterPath]:
    """Generate all stutter extensions of a path up to max_stutter
    additional repetitions per block.

    This is exponential in path length; use only for small paths.
    """
    blocks = path.blocks()
    if not blocks:
        return [path]

    results: List[StutterPath] = []
    _generate_extensions(blocks, 0, [], max_stutter, results)
    return results


def _generate_extensions(
    blocks: List[Tuple[str, int]],
    index: int,
    current: List[str],
    max_extra: int,
    results: List[StutterPath],
) -> None:
    if index == len(blocks):
        results.append(StutterPath(states=tuple(current)))
        return

    state, base_count = blocks[index]
    for extra in range(max_extra + 1):
        extended = current + [state] * (base_count + extra)
        _generate_extensions(blocks, index + 1, extended, max_extra, results)


# ---------------------------------------------------------------------------
# Stutter equivalence classes
# ---------------------------------------------------------------------------

@dataclass
class StutterEquivalenceClass:
    """An equivalence class of states under stutter trace equivalence.

    Two states are stutter-trace equivalent if they have the same
    stutter-free traces.
    """

    representative: str
    members: FrozenSet[str]
    label: FrozenSet[str]  # shared atomic propositions

    def contains(self, state: str) -> bool:
        return state in self.members

    def size(self) -> int:
        return len(self.members)


# ---------------------------------------------------------------------------
# Stutter-closed transition relation
# ---------------------------------------------------------------------------

@dataclass
class StutterClosedTransition:
    """A transition in the stutter-closed system T(X).

    In T(X), state s can reach state t via action a if there is a path
    from s to t in the original system where all intermediate states
    have the same labeling as s (a stutter path).
    """

    source: str
    target: str
    action: str
    stutter_depth: int = 0  # number of intermediate stutter steps

    def is_direct(self) -> bool:
        return self.stutter_depth == 0


# ---------------------------------------------------------------------------
# The stutter monad T
# ---------------------------------------------------------------------------

class StutterMonad:
    """The stuttering closure monad T.

    Given a labeled transition system (states, labels, transitions),
    T computes the stutter closure where:
    - States are equivalence classes under stutter-trace equivalence
    - Transitions capture reachability through stutter paths
    """

    def __init__(self):
        self._label_fn: Dict[str, FrozenSet[str]] = {}
        self._transitions: Dict[str, Dict[str, Set[str]]] = {}
        self._states: Set[str] = set()
        self._actions: Set[str] = set()
        self._stutter_classes: Optional[List[StutterEquivalenceClass]] = None
        self._stutter_closed_transitions: Optional[
            Dict[str, Dict[str, Set[str]]]
        ] = None

    def load_system(
        self,
        states: Set[str],
        labels: Dict[str, FrozenSet[str]],
        transitions: Dict[str, Dict[str, Set[str]]],
        actions: Set[str],
    ) -> None:
        """Load a transition system into the monad."""
        self._states = set(states)
        self._label_fn = dict(labels)
        self._transitions = {
            s: {a: set(ts) for a, ts in acts.items()}
            for s, acts in transitions.items()
        }
        self._actions = set(actions)
        self._stutter_classes = None
        self._stutter_closed_transitions = None
        logger.info("Loaded system with %d states, %d actions", len(states), len(actions))

    def load_from_coalgebra(self, coalgebra: Any) -> None:
        """Load from an FCoalgebra object."""
        states = set(coalgebra.states)
        labels: Dict[str, FrozenSet[str]] = {}
        transitions: Dict[str, Dict[str, Set[str]]] = {}

        for s in states:
            cs = coalgebra.get_state(s)
            if cs is not None:
                labels[s] = cs.propositions
                transitions[s] = {
                    act: set(targets)
                    for act, targets in cs.successors.items()
                }

        self.load_system(states, labels, transitions, coalgebra.actions)

    # -- η: unit of the monad -----------------------------------------------

    def unit(self, state: str) -> str:
        """η_X: X → T(X). Embeds a state into the stutter-closed system.

        The unit maps each state to its equivalence class representative.
        In the simplest case, this is the identity on representatives.
        """
        if self._stutter_classes is None:
            self._compute_stutter_classes()

        for cls in self._stutter_classes:
            if state in cls.members:
                return cls.representative
        return state  # singleton class

    def unit_map(self) -> Dict[str, str]:
        """Return the full unit map η as a dictionary."""
        if self._stutter_classes is None:
            self._compute_stutter_classes()

        result: Dict[str, str] = {}
        for cls in self._stutter_classes:
            for s in cls.members:
                result[s] = cls.representative
        return result

    # -- μ: multiplication of the monad ------------------------------------

    def multiply(
        self,
        nested_classes: Dict[str, str],
        outer_classes: Dict[str, str],
    ) -> Dict[str, str]:
        """μ_X: T(T(X)) → T(X). Flatten nested stutter closure.

        Given:
          - nested_classes: inner stutter closure mapping
          - outer_classes: outer stutter closure mapping
        Returns: the composed flattening.
        """
        result: Dict[str, str] = {}
        for state in self._states:
            inner = nested_classes.get(state, state)
            outer = outer_classes.get(inner, inner)
            result[state] = outer
        return result

    # -- monad law verification ---------------------------------------------

    def verify_left_unit_law(self) -> Tuple[bool, List[str]]:
        """Verify μ ∘ Tη = id.

        For every state s, μ(η(η(s))) = η(s).
        """
        violations: List[str] = []
        eta = self.unit_map()

        for s in self._states:
            eta_s = eta.get(s, s)
            eta_eta_s = eta.get(eta_s, eta_s)
            mu_result = self.multiply(eta, eta)
            if mu_result.get(s, s) != eta_s:
                violations.append(
                    f"Left unit violation at {s}: "
                    f"μ(Tη({s})) = {mu_result.get(s)} ≠ η({s}) = {eta_s}"
                )

        return len(violations) == 0, violations

    def verify_right_unit_law(self) -> Tuple[bool, List[str]]:
        """Verify μ ∘ ηT = id.

        For every state s, μ(η(s), id) should give η(s).
        """
        violations: List[str] = []
        eta = self.unit_map()

        identity = {s: s for s in self._states}
        mu_result = self.multiply(identity, eta)

        for s in self._states:
            expected = eta.get(s, s)
            actual = mu_result.get(s, s)
            if actual != expected:
                violations.append(
                    f"Right unit violation at {s}: "
                    f"μ(ηT({s})) = {actual} ≠ {expected}"
                )

        return len(violations) == 0, violations

    def verify_associativity(self) -> Tuple[bool, List[str]]:
        """Verify μ ∘ Tμ = μ ∘ μT.

        Associativity of the monad multiplication.
        """
        violations: List[str] = []
        eta = self.unit_map()

        left = self.multiply(eta, self.multiply(eta, eta))
        right = self.multiply(self.multiply(eta, eta), eta)

        for s in self._states:
            l_val = left.get(s, s)
            r_val = right.get(s, s)
            if l_val != r_val:
                violations.append(
                    f"Associativity violation at {s}: "
                    f"μ∘Tμ({s}) = {l_val} ≠ μ∘μT({s}) = {r_val}"
                )

        return len(violations) == 0, violations

    def verify_all_laws(self) -> Tuple[bool, Dict[str, List[str]]]:
        """Verify all three monad laws."""
        results: Dict[str, List[str]] = {}
        all_ok = True

        ok, viols = self.verify_left_unit_law()
        results["left_unit"] = viols
        all_ok = all_ok and ok

        ok, viols = self.verify_right_unit_law()
        results["right_unit"] = viols
        all_ok = all_ok and ok

        ok, viols = self.verify_associativity()
        results["associativity"] = viols
        all_ok = all_ok and ok

        return all_ok, results

    # -- stutter equivalence computation ------------------------------------

    def compute_stutter_equivalence_classes(
        self,
    ) -> List[StutterEquivalenceClass]:
        """Compute stutter equivalence classes.

        Two states s, t are stutter-equivalent if:
        1. They have the same labeling L(s) = L(t), AND
        2. There exist stutter paths from s reaching the same
           label-distinct successors as from t, and vice versa.
        """
        if self._stutter_classes is not None:
            return list(self._stutter_classes)
        self._compute_stutter_classes()
        return list(self._stutter_classes)

    def _compute_stutter_classes(self) -> None:
        """Iterative partition refinement for stutter equivalence."""
        label_groups: Dict[FrozenSet[str], Set[str]] = defaultdict(set)
        for s in self._states:
            lab = self._label_fn.get(s, frozenset())
            label_groups[lab].add(s)

        partition: List[Set[str]] = list(label_groups.values())

        state_to_block: Dict[str, int] = {}
        for i, block in enumerate(partition):
            for s in block:
                state_to_block[s] = i

        changed = True
        iteration = 0
        while changed:
            changed = False
            iteration += 1
            new_partition: List[Set[str]] = []
            new_state_to_block: Dict[str, int] = {}

            for block in partition:
                sub_blocks: Dict[Any, Set[str]] = defaultdict(set)
                for s in block:
                    sig = self._stutter_signature(s, state_to_block)
                    sub_blocks[sig].add(s)

                for sub_block in sub_blocks.values():
                    idx = len(new_partition)
                    new_partition.append(sub_block)
                    for s in sub_block:
                        new_state_to_block[s] = idx

            if len(new_partition) != len(partition):
                changed = True

            partition = new_partition
            state_to_block = new_state_to_block

        self._stutter_classes = []
        for block in partition:
            representative = min(block)
            label = self._label_fn.get(representative, frozenset())
            self._stutter_classes.append(
                StutterEquivalenceClass(
                    representative=representative,
                    members=frozenset(block),
                    label=label,
                )
            )

        logger.info(
            "Computed %d stutter equivalence classes in %d iterations",
            len(self._stutter_classes),
            iteration,
        )

    def _stutter_signature(
        self, state: str, state_to_block: Dict[str, int]
    ) -> Tuple:
        """Compute a signature for partition refinement.

        The signature captures which blocks are reachable via
        stutter-insensitive transitions (transitions to a different label).
        """
        my_label = self._label_fn.get(state, frozenset())
        reachable_blocks: Dict[str, FrozenSet[int]] = {}

        for act in self._actions:
            blocks_for_act: Set[int] = set()
            direct_succs = self._transitions.get(state, {}).get(act, set())

            for t in direct_succs:
                t_label = self._label_fn.get(t, frozenset())
                if t_label != my_label:
                    blocks_for_act.add(state_to_block.get(t, -1))
                else:
                    vis = self._stutter_reachable_blocks(
                        t, my_label, state_to_block, act
                    )
                    blocks_for_act |= vis

            reachable_blocks[act] = frozenset(blocks_for_act)

        return tuple(sorted(
            (act, tuple(sorted(blocks)))
            for act, blocks in reachable_blocks.items()
        ))

    def _stutter_reachable_blocks(
        self,
        start: str,
        stutter_label: FrozenSet[str],
        state_to_block: Dict[str, int],
        action: str,
    ) -> Set[int]:
        """BFS to find blocks reachable through states with the same label."""
        visited: Set[str] = set()
        queue: Deque[str] = deque([start])
        result: Set[int] = set()

        while queue:
            s = queue.popleft()
            if s in visited:
                continue
            visited.add(s)

            s_label = self._label_fn.get(s, frozenset())
            if s_label != stutter_label:
                result.add(state_to_block.get(s, -1))
                continue

            result.add(state_to_block.get(s, -1))
            for t in self._transitions.get(s, {}).get(action, set()):
                if t not in visited:
                    queue.append(t)

        return result

    # -- stutter-closed transition relation ---------------------------------

    def compute_stutter_closed_transitions(
        self,
    ) -> Dict[str, Dict[str, Set[str]]]:
        """Compute the stutter-closed transition relation.

        In T(X), state s can reach t if there is a path from s to t
        in the original system where all intermediate states have
        the same label as s.
        """
        if self._stutter_closed_transitions is not None:
            return self._stutter_closed_transitions

        closed: Dict[str, Dict[str, Set[str]]] = {}

        for s in self._states:
            s_label = self._label_fn.get(s, frozenset())
            closed[s] = {}

            for act in self._actions:
                reachable = self._stutter_close_from(s, s_label, act)
                if reachable:
                    closed[s][act] = reachable

        self._stutter_closed_transitions = closed
        return closed

    def _stutter_close_from(
        self, start: str, start_label: FrozenSet[str], action: str
    ) -> Set[str]:
        """Find all states reachable from start via action, allowing
        intermediate stuttering (states with the same label).
        """
        same_label_states = {
            s for s in self._states
            if self._label_fn.get(s, frozenset()) == start_label
        }

        stutter_reachable: Set[str] = set()
        queue: Deque[str] = deque([start])
        visited: Set[str] = set()

        while queue:
            s = queue.popleft()
            if s in visited:
                continue
            visited.add(s)
            stutter_reachable.add(s)

            for t in self._transitions.get(s, {}).get(action, set()):
                if t in same_label_states and t not in visited:
                    queue.append(t)

        non_stutter_targets: Set[str] = set()
        for s in stutter_reachable:
            for t in self._transitions.get(s, {}).get(action, set()):
                if t not in same_label_states:
                    non_stutter_targets.add(t)

        direct = self._transitions.get(start, {}).get(action, set())
        return direct | non_stutter_targets

    # -- stutter trace equivalence ------------------------------------------

    def are_stutter_trace_equivalent(
        self, s1: str, s2: str, depth: int = -1
    ) -> bool:
        """Check if two states are stutter-trace equivalent.

        If depth < 0, use the full fixed-point computation.
        Otherwise, check up to the given depth.
        """
        if self._stutter_classes is None:
            self._compute_stutter_classes()

        if depth < 0:
            for cls in self._stutter_classes:
                if s1 in cls.members and s2 in cls.members:
                    return True
                if s1 in cls.members or s2 in cls.members:
                    return False
            return s1 == s2

        return self._bounded_stutter_equiv(s1, s2, depth)

    def _bounded_stutter_equiv(self, s1: str, s2: str, depth: int) -> bool:
        """Bounded-depth stutter equivalence check."""
        if depth == 0:
            return self._label_fn.get(s1, frozenset()) == self._label_fn.get(
                s2, frozenset()
            )

        l1 = self._label_fn.get(s1, frozenset())
        l2 = self._label_fn.get(s2, frozenset())
        if l1 != l2:
            return False

        for act in self._actions:
            succs1 = self._transitions.get(s1, {}).get(act, set())
            succs2 = self._transitions.get(s2, {}).get(act, set())

            for t1 in succs1:
                t1_label = self._label_fn.get(t1, frozenset())
                if t1_label == l1:
                    if not self._bounded_stutter_equiv(s1, t1, depth - 1):
                        continue
                found_match = False
                for t2 in succs2:
                    if self._bounded_stutter_equiv(t1, t2, depth - 1):
                        found_match = True
                        break
                if not found_match:
                    return False

            for t2 in succs2:
                t2_label = self._label_fn.get(t2, frozenset())
                if t2_label == l2:
                    if not self._bounded_stutter_equiv(s2, t2, depth - 1):
                        continue
                found_match = False
                for t1 in succs1:
                    if self._bounded_stutter_equiv(t2, t1, depth - 1):
                        found_match = True
                        break
                if not found_match:
                    return False

        return True

    # -- stutter extension/contraction of concrete paths --------------------

    def stutter_extend(
        self, path: StutterPath, max_extra: int = 2
    ) -> List[StutterPath]:
        """Generate all stutter extensions of a path."""
        return all_stutter_extensions(path, max_extra)

    def stutter_contract(self, path: StutterPath) -> StutterPath:
        """Compute the stutter-free core of a path."""
        return path.stutter_free_core()

    def check_path_valid(self, path: StutterPath) -> bool:
        """Check if a path is a valid path in the transition system."""
        for i in range(len(path.states) - 1):
            s = path.states[i]
            t = path.states[i + 1]
            if s == t:
                continue  # stutter step always valid
            reachable = False
            for act_succs in self._transitions.get(s, {}).values():
                if t in act_succs:
                    reachable = True
                    break
            if not reachable:
                return False
        return True

    def get_stutter_class(self, state: str) -> Optional[StutterEquivalenceClass]:
        """Get the stutter equivalence class of a state."""
        if self._stutter_classes is None:
            self._compute_stutter_classes()
        for cls in self._stutter_classes:
            if state in cls.members:
                return cls
        return None
