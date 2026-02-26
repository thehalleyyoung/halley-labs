"""
Behavioral equivalence for CoaCert-TLA.

Two states are behaviorally equivalent if they cannot be distinguished
by any observation (test). For the functor
  F(X) = P(AP) × P(X)^Act × Fair(X),
behavioral equivalence is computed via iterative partition refinement.

This module implements:
  - Standard behavioral equivalence (AP + successors)
  - Stuttering behavioral equivalence (incorporating stutter monad T)
  - Fair behavioral equivalence (incorporating fairness constraints)
  - Full F-behavioral equivalence combining all components
  - Paige-Tarjan style partition refinement
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
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
# Equivalence classes
# ---------------------------------------------------------------------------

@dataclass
class EquivalenceClass:
    """A single equivalence class (block) of behaviorally equivalent states."""

    representative: str
    members: FrozenSet[str]
    propositions: FrozenSet[str] = frozenset()
    iteration_stabilized: int = 0

    def contains(self, state: str) -> bool:
        return state in self.members

    def size(self) -> int:
        return len(self.members)

    def is_singleton(self) -> bool:
        return len(self.members) == 1

    def __repr__(self) -> str:
        return (
            f"EquivalenceClass(rep={self.representative}, "
            f"size={self.size()}, props={sorted(self.propositions)})"
        )


# ---------------------------------------------------------------------------
# Partition refinement (Paige-Tarjan style)
# ---------------------------------------------------------------------------

class PartitionRefinement:
    """Paige-Tarjan partition refinement algorithm.

    Given an initial partition P and a transition relation,
    iteratively refine P until a fixed point is reached.
    Each refinement step splits blocks based on successor block
    membership.
    """

    def __init__(self):
        self._partition: List[Set[str]] = []
        self._state_to_block: Dict[str, int] = {}
        self._iteration: int = 0

    @property
    def partition(self) -> List[FrozenSet[str]]:
        return [frozenset(b) for b in self._partition]

    @property
    def num_blocks(self) -> int:
        return len(self._partition)

    def initialize(self, initial_partition: List[Set[str]]) -> None:
        """Set the initial partition."""
        self._partition = [set(b) for b in initial_partition]
        self._state_to_block = {}
        for i, block in enumerate(self._partition):
            for s in block:
                self._state_to_block[s] = i
        self._iteration = 0

    def refine_step(
        self,
        splitter: FrozenSet[str],
        predecessor_fn: Callable[[str], Set[str]],
    ) -> bool:
        """One step of Paige-Tarjan refinement using a splitter set.

        For each block B, split it into:
          B ∩ pre(splitter) and B \\ pre(splitter)
        where pre(splitter) = {s : ∃t ∈ splitter. s →  t}.

        Returns True if any block was split.
        """
        self._iteration += 1
        predecessors_of_splitter: Set[str] = set()
        for t in splitter:
            predecessors_of_splitter |= predecessor_fn(t)

        new_partition: List[Set[str]] = []
        changed = False

        for block in self._partition:
            intersect = block & predecessors_of_splitter
            diff = block - predecessors_of_splitter

            if intersect and diff:
                new_partition.append(intersect)
                new_partition.append(diff)
                changed = True
            else:
                new_partition.append(block)

        if changed:
            self._partition = new_partition
            self._state_to_block = {}
            for i, block in enumerate(self._partition):
                for s in block:
                    self._state_to_block[s] = i

        return changed

    def refine_by_signature(
        self,
        signature_fn: Callable[[str, Dict[str, int]], Any],
    ) -> bool:
        """Refine by a general signature function.

        Split each block into sub-blocks where states have the same signature.
        """
        self._iteration += 1
        new_partition: List[Set[str]] = []
        changed = False

        for block in self._partition:
            sub_blocks: Dict[Any, Set[str]] = defaultdict(set)
            for s in block:
                sig = signature_fn(s, self._state_to_block)
                sub_blocks[sig].add(s)

            if len(sub_blocks) > 1:
                changed = True

            new_partition.extend(sub_blocks.values())

        if changed:
            self._partition = new_partition
            self._state_to_block = {}
            for i, block in enumerate(self._partition):
                for s in block:
                    self._state_to_block[s] = i

        return changed

    def get_block_of(self, state: str) -> int:
        return self._state_to_block.get(state, -1)

    def are_equivalent(self, s1: str, s2: str) -> bool:
        return (
            self._state_to_block.get(s1, -1) == self._state_to_block.get(s2, -2)
        )

    def to_equivalence_classes(
        self,
        label_fn: Optional[Callable[[str], FrozenSet[str]]] = None,
    ) -> List[EquivalenceClass]:
        """Convert current partition to EquivalenceClass objects."""
        classes = []
        for block in self._partition:
            rep = min(block)
            props = label_fn(rep) if label_fn else frozenset()
            classes.append(
                EquivalenceClass(
                    representative=rep,
                    members=frozenset(block),
                    propositions=props,
                    iteration_stabilized=self._iteration,
                )
            )
        return classes


# ---------------------------------------------------------------------------
# Behavioral equivalence
# ---------------------------------------------------------------------------

class BehavioralEquivalence:
    """Compute behavioral equivalence for F-coalgebras.

    Uses iterative partition refinement starting from the AP-based
    partition and refining by successor structure and fairness.
    """

    def __init__(self, coalgebra: Any):
        self.coalgebra = coalgebra
        self._refiner = PartitionRefinement()
        self._equivalence_classes: Optional[List[EquivalenceClass]] = None
        self._iteration_count: int = 0

    # -- standard behavioral equivalence ------------------------------------

    def compute(self) -> List[EquivalenceClass]:
        """Compute standard behavioral equivalence (AP + successors).

        This is the coarsest partition where states in the same block
        have the same AP labels and, for each action, their successor
        sets map to the same set of blocks.
        """
        initial = self._ap_partition()
        self._refiner.initialize(initial)

        changed = True
        iteration = 0
        while changed:
            iteration += 1
            changed = self._refiner.refine_by_signature(
                self._successor_signature
            )
            logger.debug(
                "Iteration %d: %d blocks", iteration, self._refiner.num_blocks
            )

        self._iteration_count = iteration
        self._equivalence_classes = self._refiner.to_equivalence_classes(
            label_fn=lambda s: self.coalgebra.apply_functor(s).propositions
        )

        logger.info(
            "Behavioral equivalence: %d classes after %d iterations",
            len(self._equivalence_classes),
            iteration,
        )
        return self._equivalence_classes

    def _ap_partition(self) -> List[Set[str]]:
        """Initial partition based on atomic proposition labels."""
        groups: Dict[FrozenSet[str], Set[str]] = defaultdict(set)
        for s in self.coalgebra.states:
            fv = self.coalgebra.apply_functor(s)
            groups[fv.propositions].add(s)
        return list(groups.values())

    def _successor_signature(
        self, state: str, state_to_block: Dict[str, int]
    ) -> Tuple:
        """Compute the successor signature: for each action, the sorted
        tuple of blocks of successors.
        """
        fv = self.coalgebra.apply_functor(state)
        sig_parts: List[Tuple[str, Tuple[int, ...]]] = []
        for act in sorted(self.coalgebra.actions):
            succs = fv.successor_set(act)
            block_ids = tuple(sorted(state_to_block.get(t, -1) for t in succs))
            sig_parts.append((act, block_ids))
        return tuple(sig_parts)

    # -- fair behavioral equivalence ----------------------------------------

    def compute_fair(self) -> List[EquivalenceClass]:
        """Compute fair behavioral equivalence (AP + successors + fairness).

        Refines standard behavioral equivalence by also requiring that
        states in the same block agree on fairness membership.
        """
        initial = self._ap_fairness_partition()
        self._refiner.initialize(initial)

        changed = True
        iteration = 0
        while changed:
            iteration += 1
            changed = self._refiner.refine_by_signature(
                self._fair_successor_signature
            )

        self._iteration_count = iteration
        self._equivalence_classes = self._refiner.to_equivalence_classes(
            label_fn=lambda s: self.coalgebra.apply_functor(s).propositions
        )

        logger.info(
            "Fair behavioral equivalence: %d classes after %d iterations",
            len(self._equivalence_classes),
            iteration,
        )
        return self._equivalence_classes

    def _ap_fairness_partition(self) -> List[Set[str]]:
        """Initial partition based on AP labels AND fairness membership."""
        groups: Dict[Tuple, Set[str]] = defaultdict(set)
        for s in self.coalgebra.states:
            fv = self.coalgebra.apply_functor(s)
            fair_key = tuple(sorted(fv.fairness_membership.items()))
            key = (fv.propositions, fair_key)
            groups[key].add(s)
        return list(groups.values())

    def _fair_successor_signature(
        self, state: str, state_to_block: Dict[str, int]
    ) -> Tuple:
        """Signature including both successor blocks and fairness info."""
        fv = self.coalgebra.apply_functor(state)
        sig_parts: List[Any] = []

        for act in sorted(self.coalgebra.actions):
            succs = fv.successor_set(act)
            block_ids = tuple(sorted(state_to_block.get(t, -1) for t in succs))
            sig_parts.append((act, block_ids))

        fair_part = tuple(sorted(fv.fairness_membership.items()))
        sig_parts.append(fair_part)

        return tuple(sig_parts)

    # -- stuttering behavioral equivalence ----------------------------------

    def compute_stuttering(
        self, stutter_monad: Any
    ) -> List[EquivalenceClass]:
        """Compute stuttering behavioral equivalence.

        States are equivalent if they are indistinguishable under
        stutter-insensitive observation.
        """
        stutter_monad.load_from_coalgebra(self.coalgebra)
        stutter_classes = stutter_monad.compute_stutter_equivalence_classes()

        initial: List[Set[str]] = [
            set(cls.members) for cls in stutter_classes
        ]
        self._refiner.initialize(initial)

        closed = stutter_monad.compute_stutter_closed_transitions()

        changed = True
        iteration = 0
        while changed:
            iteration += 1
            changed = self._refiner.refine_by_signature(
                lambda s, stb: self._stutter_signature(s, stb, closed)
            )

        self._iteration_count = iteration
        self._equivalence_classes = self._refiner.to_equivalence_classes(
            label_fn=lambda s: self.coalgebra.apply_functor(s).propositions
        )

        logger.info(
            "Stuttering behavioral equivalence: %d classes after %d iterations",
            len(self._equivalence_classes),
            iteration,
        )
        return self._equivalence_classes

    def _stutter_signature(
        self,
        state: str,
        state_to_block: Dict[str, int],
        closed_transitions: Dict[str, Dict[str, Set[str]]],
    ) -> Tuple:
        """Signature using stutter-closed transitions."""
        sig_parts: List[Tuple[str, Tuple[int, ...]]] = []
        trans = closed_transitions.get(state, {})
        for act in sorted(self.coalgebra.actions):
            succs = trans.get(act, set())
            block_ids = tuple(sorted(state_to_block.get(t, -1) for t in succs))
            sig_parts.append((act, block_ids))
        return tuple(sig_parts)

    # -- full F-behavioral equivalence --------------------------------------

    def compute_full(
        self,
        stutter_monad: Optional[Any] = None,
    ) -> List[EquivalenceClass]:
        """Compute the full F-behavioral equivalence combining:
          - AP labels
          - Successor structure
          - Fairness membership
          - Stutter closure (if stutter_monad provided)
        """
        if stutter_monad is not None:
            stutter_monad.load_from_coalgebra(self.coalgebra)
            stutter_classes = stutter_monad.compute_stutter_equivalence_classes()
            closed = stutter_monad.compute_stutter_closed_transitions()

            stutter_initial: List[Set[str]] = [
                set(cls.members) for cls in stutter_classes
            ]
            fair_initial = self._ap_fairness_partition()

            merged = self._intersect_partitions(stutter_initial, fair_initial)
        else:
            merged = self._ap_fairness_partition()
            closed = None

        self._refiner.initialize(merged)

        changed = True
        iteration = 0
        while changed:
            iteration += 1
            if closed is not None:
                changed = self._refiner.refine_by_signature(
                    lambda s, stb: self._full_signature(s, stb, closed)
                )
            else:
                changed = self._refiner.refine_by_signature(
                    self._fair_successor_signature
                )

        self._iteration_count = iteration
        self._equivalence_classes = self._refiner.to_equivalence_classes(
            label_fn=lambda s: self.coalgebra.apply_functor(s).propositions
        )

        logger.info(
            "Full F-behavioral equivalence: %d classes after %d iterations",
            len(self._equivalence_classes),
            iteration,
        )
        return self._equivalence_classes

    def _full_signature(
        self,
        state: str,
        state_to_block: Dict[str, int],
        closed_transitions: Dict[str, Dict[str, Set[str]]],
    ) -> Tuple:
        """Full signature combining stutter-closed successors and fairness."""
        fv = self.coalgebra.apply_functor(state)
        sig_parts: List[Any] = []

        trans = closed_transitions.get(state, {})
        for act in sorted(self.coalgebra.actions):
            succs = trans.get(act, set())
            block_ids = tuple(sorted(state_to_block.get(t, -1) for t in succs))
            sig_parts.append((act, block_ids))

        fair_part = tuple(sorted(fv.fairness_membership.items()))
        sig_parts.append(fair_part)

        return tuple(sig_parts)

    def _intersect_partitions(
        self,
        p1: List[Set[str]],
        p2: List[Set[str]],
    ) -> List[Set[str]]:
        """Compute the intersection (meet) of two partitions.

        The result is the finest partition coarser than both.
        """
        p2_map: Dict[str, int] = {}
        for i, block in enumerate(p2):
            for s in block:
                p2_map[s] = i

        result: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        for i, block in enumerate(p1):
            for s in block:
                j = p2_map.get(s, -1)
                result[(i, j)].add(s)

        return list(result.values())

    # -- query methods ------------------------------------------------------

    def are_equivalent(self, s1: str, s2: str) -> bool:
        """Check if two states are behaviorally equivalent."""
        if self._equivalence_classes is None:
            self.compute()
        for cls in self._equivalence_classes:
            if s1 in cls.members and s2 in cls.members:
                return True
            if s1 in cls.members or s2 in cls.members:
                return False
        return s1 == s2

    def get_class_of(self, state: str) -> Optional[EquivalenceClass]:
        """Get the equivalence class of a state."""
        if self._equivalence_classes is None:
            self.compute()
        for cls in self._equivalence_classes:
            if state in cls.members:
                return cls
        return None

    def quotient_partition(self) -> List[FrozenSet[str]]:
        """Get the partition suitable for quotient construction."""
        if self._equivalence_classes is None:
            self.compute()
        return [cls.members for cls in self._equivalence_classes]

    def equivalence_class_count(self) -> int:
        if self._equivalence_classes is None:
            self.compute()
        return len(self._equivalence_classes)

    def iteration_count(self) -> int:
        return self._iteration_count

    def enumerate_classes(self) -> List[EquivalenceClass]:
        """Return all equivalence classes."""
        if self._equivalence_classes is None:
            self.compute()
        return list(self._equivalence_classes)

    def distinguishing_action(
        self, s1: str, s2: str
    ) -> Optional[Tuple[str, str]]:
        """Find an action that distinguishes two non-equivalent states.

        Returns (action, explanation) or None if they are equivalent.
        """
        fv1 = self.coalgebra.apply_functor(s1)
        fv2 = self.coalgebra.apply_functor(s2)

        if fv1.propositions != fv2.propositions:
            return (
                "AP",
                f"Different labels: {sorted(fv1.propositions)} vs {sorted(fv2.propositions)}",
            )

        for act in sorted(self.coalgebra.actions):
            succs1 = fv1.successor_set(act)
            succs2 = fv2.successor_set(act)
            blocks1 = frozenset(self._refiner.get_block_of(t) for t in succs1)
            blocks2 = frozenset(self._refiner.get_block_of(t) for t in succs2)
            if blocks1 != blocks2:
                return (
                    act,
                    f"Under action '{act}': blocks {sorted(blocks1)} vs {sorted(blocks2)}",
                )

        for idx in set(fv1.fairness_membership) | set(fv2.fairness_membership):
            m1 = fv1.fairness_membership.get(idx, (False, False))
            m2 = fv2.fairness_membership.get(idx, (False, False))
            if m1 != m2:
                return (
                    f"fair_{idx}",
                    f"Fairness pair {idx}: {m1} vs {m2}",
                )

        return None
