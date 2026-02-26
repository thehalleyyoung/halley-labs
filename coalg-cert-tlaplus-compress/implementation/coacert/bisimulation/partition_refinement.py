"""
Paige–Tarjan partition refinement for bisimulation computation.

Implements the classical Paige–Tarjan algorithm (O(m log n) time) for
computing the coarsest partition that is stable with respect to a given
transition relation.  Extends the core algorithm with:

  - AP-refinement: split by atomic proposition labeling
  - Action-refinement: for each action a, split by {s : succ(s,a) ∩ B ≠ ∅}
  - Fairness-refinement: split by acceptance pair membership
  - Incremental refinement support
  - Refinement history tracking
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from .relation import BisimulationRelation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """A block (equivalence class) in the current partition."""

    id: int
    members: Set[str]
    is_compound: bool = False  # has been used as splitter and produced >1 sub-block

    @property
    def size(self) -> int:
        return len(self.members)

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Block):
            return NotImplemented
        return self.id == other.id


@dataclass
class RefinementStep:
    """Record of a single refinement step."""

    round_number: int
    splitter_block_id: int
    action: Optional[str]
    old_block_count: int
    new_block_count: int
    splits_performed: int
    elapsed_ms: float


@dataclass
class RefinementResult:
    """Final result of the full refinement computation."""

    partition: BisimulationRelation
    num_rounds: int
    history: List[RefinementStep]
    total_elapsed_ms: float
    initial_blocks: int
    final_blocks: int
    states_processed: int
    transitions_processed: int


# ---------------------------------------------------------------------------
# Transition graph adapter
# ---------------------------------------------------------------------------

class _TransitionIndex:
    """Pre-computed forward/backward successor index."""

    def __init__(
        self,
        states: Set[str],
        actions: Set[str],
        transitions: Dict[str, Dict[str, Set[str]]],
    ) -> None:
        self.states = states
        self.actions = actions
        # forward: state -> action -> {targets}
        self.fwd: Dict[str, Dict[str, Set[str]]] = {}
        # backward: state -> action -> {sources}
        self.bwd: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._edge_count = 0
        for src, amap in transitions.items():
            self.fwd[src] = {}
            for act, targets in amap.items():
                self.fwd[src][act] = set(targets)
                self._edge_count += len(targets)
                for tgt in targets:
                    self.bwd[tgt][act].add(src)

    @property
    def edge_count(self) -> int:
        return self._edge_count

    def successors(self, state: str, action: str) -> Set[str]:
        return self.fwd.get(state, {}).get(action, set())

    def predecessors(self, state: str, action: str) -> Set[str]:
        return self.bwd.get(state, {}).get(action, set())

    def all_successors(self, state: str) -> Set[str]:
        result: Set[str] = set()
        for targets in self.fwd.get(state, {}).values():
            result |= targets
        return result

    def all_predecessors(self, state: str) -> Set[str]:
        result: Set[str] = set()
        for sources in self.bwd.get(state, {}).values():
            result |= sources
        return result


# ---------------------------------------------------------------------------
# PartitionRefinement
# ---------------------------------------------------------------------------

class PartitionRefinement:
    """Paige–Tarjan partition refinement engine.

    Usage::

        pr = PartitionRefinement(states, actions, transitions, labels)
        result = pr.refine()
        partition = result.partition
    """

    def __init__(
        self,
        states: Set[str],
        actions: Set[str],
        transitions: Dict[str, Dict[str, Set[str]]],
        labels: Dict[str, Set[str]],
        fairness_pairs: Optional[List[Tuple[Set[str], Set[str]]]] = None,
    ) -> None:
        self._states = set(states)
        self._actions = set(actions)
        self._labels = dict(labels)
        self._fairness_pairs: List[Tuple[Set[str], Set[str]]] = (
            list(fairness_pairs) if fairness_pairs else []
        )
        self._index = _TransitionIndex(self._states, self._actions, transitions)

        # block management
        self._next_block_id = 0
        self._blocks: Dict[int, Block] = {}
        self._state_block: Dict[str, int] = {}

        # work-list of splitter candidates
        self._worklist: Deque[Tuple[int, Optional[str]]] = deque()

        # history
        self._history: List[RefinementStep] = []
        self._round = 0

    # -- block management ---------------------------------------------------

    def _new_block(self, members: Set[str]) -> Block:
        bid = self._next_block_id
        self._next_block_id += 1
        blk = Block(id=bid, members=set(members))
        self._blocks[bid] = blk
        for s in members:
            self._state_block[s] = bid
        return blk

    def _remove_block(self, bid: int) -> None:
        if bid in self._blocks:
            del self._blocks[bid]

    def _block_of(self, state: str) -> Block:
        return self._blocks[self._state_block[state]]

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    # -- initial partition --------------------------------------------------

    def _build_initial_partition(self) -> None:
        """Partition states by AP labeling, then by fairness membership."""
        groups: Dict[Any, Set[str]] = defaultdict(set)
        for s in self._states:
            ap_key = frozenset(self._labels.get(s, set()))
            fair_key_parts = []
            for i, (b_set, g_set) in enumerate(self._fairness_pairs):
                fair_key_parts.append((i, s in b_set, s in g_set))
            key = (ap_key, tuple(fair_key_parts))
            groups[key].add(s)

        for members in groups.values():
            blk = self._new_block(members)
            # each initial block is a potential splitter for every action
            for act in self._actions:
                self._worklist.append((blk.id, act))

        logger.debug(
            "Initial partition: %d blocks from %d states",
            self.num_blocks,
            len(self._states),
        )

    # -- core splitting -----------------------------------------------------

    def _split_block_by_splitter(
        self,
        block: Block,
        splitter_states: Set[str],
        action: str,
    ) -> Optional[Tuple[Block, Block]]:
        """Split *block* into states that can/cannot reach *splitter_states* via *action*.

        Returns ``(block_in, block_out)`` if a split occurred, else ``None``.
        """
        has_succ_in_splitter: Set[str] = set()
        no_succ_in_splitter: Set[str] = set()

        for s in block.members:
            succs = self._index.successors(s, action)
            if succs & splitter_states:
                has_succ_in_splitter.add(s)
            else:
                no_succ_in_splitter.add(s)

        if not has_succ_in_splitter or not no_succ_in_splitter:
            return None  # no split

        return self._perform_split(block, has_succ_in_splitter, no_succ_in_splitter)

    def _perform_split(
        self,
        block: Block,
        part_a: Set[str],
        part_b: Set[str],
    ) -> Tuple[Block, Block]:
        """Replace *block* with two new blocks."""
        self._remove_block(block.id)

        blk_a = self._new_block(part_a)
        blk_b = self._new_block(part_b)

        return blk_a, blk_b

    # -- Paige–Tarjan refinement step ---------------------------------------

    def _refine_step(self) -> int:
        """Pop one splitter from the work-list and refine.

        Returns number of splits performed.
        """
        if not self._worklist:
            return 0

        splitter_id, action = self._worklist.popleft()
        if splitter_id not in self._blocks:
            return 0  # block was already split away

        splitter = self._blocks[splitter_id]
        if action is None:
            return 0

        splits = 0
        current_block_ids = list(self._blocks.keys())

        for bid in current_block_ids:
            if bid not in self._blocks:
                continue
            blk = self._blocks[bid]
            if blk.size <= 1:
                continue

            result = self._split_block_by_splitter(blk, splitter.members, action)
            if result is not None:
                blk_a, blk_b = result
                splits += 1
                # For nondeterministic bisimulation, both sub-blocks
                # must be added as splitters (Hopcroft's smaller-only
                # optimization is only valid for deterministic systems).
                for act in self._actions:
                    self._worklist.append((blk_a.id, act))
                    self._worklist.append((blk_b.id, act))

        return splits

    # -- AP-refinement ------------------------------------------------------

    def refine_by_ap(self) -> int:
        """Split all blocks by atomic proposition labeling.

        Returns number of new blocks created.
        """
        old_count = self.num_blocks
        block_ids = list(self._blocks.keys())

        for bid in block_ids:
            if bid not in self._blocks:
                continue
            blk = self._blocks[bid]
            if blk.size <= 1:
                continue
            groups: Dict[FrozenSet[str], Set[str]] = defaultdict(set)
            for s in blk.members:
                groups[frozenset(self._labels.get(s, set()))].add(s)
            if len(groups) <= 1:
                continue
            # split
            self._remove_block(bid)
            for members in groups.values():
                new_blk = self._new_block(members)
                for act in self._actions:
                    self._worklist.append((new_blk.id, act))

        return self.num_blocks - old_count

    # -- fairness refinement ------------------------------------------------

    def refine_by_fairness(self) -> int:
        """Split blocks by acceptance pair membership.

        Returns number of new blocks created.
        """
        old_count = self.num_blocks

        for pair_idx, (b_set, g_set) in enumerate(self._fairness_pairs):
            block_ids = list(self._blocks.keys())
            for bid in block_ids:
                if bid not in self._blocks:
                    continue
                blk = self._blocks[bid]
                if blk.size <= 1:
                    continue
                groups: Dict[Tuple[bool, bool], Set[str]] = defaultdict(set)
                for s in blk.members:
                    groups[(s in b_set, s in g_set)].add(s)
                if len(groups) <= 1:
                    continue
                self._remove_block(bid)
                for members in groups.values():
                    new_blk = self._new_block(members)
                    for act in self._actions:
                        self._worklist.append((new_blk.id, act))

        return self.num_blocks - old_count

    # -- action refinement --------------------------------------------------

    def refine_by_action(self, action: str) -> int:
        """Refine all blocks with respect to a specific action.

        For each block B, split other blocks into states that have/lack a
        successor in B under *action*.  Returns number of splits.
        """
        old_count = self.num_blocks
        splitter_ids = list(self._blocks.keys())

        for sid in splitter_ids:
            if sid not in self._blocks:
                continue
            splitter = self._blocks[sid]
            target_ids = list(self._blocks.keys())
            for bid in target_ids:
                if bid not in self._blocks:
                    continue
                blk = self._blocks[bid]
                if blk.size <= 1:
                    continue
                self._split_block_by_splitter(blk, splitter.members, action)

        return self.num_blocks - old_count

    # -- full refinement ----------------------------------------------------

    def refine(self, max_rounds: int = 100_000) -> RefinementResult:
        """Run the full Paige–Tarjan refinement to a fixed point.

        Returns a :class:`RefinementResult` containing the final partition,
        round count, and history.
        """
        t0 = time.monotonic()
        self._build_initial_partition()
        initial_blocks = self.num_blocks

        while self._worklist and self._round < max_rounds:
            self._round += 1
            t_step = time.monotonic()
            old_count = self.num_blocks

            splitter_id, action = self._worklist[0]  # peek
            splits = self._refine_step()

            step = RefinementStep(
                round_number=self._round,
                splitter_block_id=splitter_id,
                action=action,
                old_block_count=old_count,
                new_block_count=self.num_blocks,
                splits_performed=splits,
                elapsed_ms=(time.monotonic() - t_step) * 1000,
            )
            self._history.append(step)

            if splits > 0:
                logger.debug(
                    "Round %d: %d splits, %d -> %d blocks (%.1f ms)",
                    self._round,
                    splits,
                    old_count,
                    self.num_blocks,
                    step.elapsed_ms,
                )

        partition = self._build_partition()
        total_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "Refinement complete: %d rounds, %d -> %d blocks in %.1f ms",
            self._round,
            initial_blocks,
            self.num_blocks,
            total_ms,
        )

        return RefinementResult(
            partition=partition,
            num_rounds=self._round,
            history=self._history,
            total_elapsed_ms=total_ms,
            initial_blocks=initial_blocks,
            final_blocks=self.num_blocks,
            states_processed=len(self._states),
            transitions_processed=self._index.edge_count,
        )

    # -- build final partition ----------------------------------------------

    def _build_partition(self) -> BisimulationRelation:
        """Convert the current block structure to a :class:`BisimulationRelation`."""
        rel = BisimulationRelation()
        for blk in self._blocks.values():
            mlist = sorted(blk.members)
            for s in mlist:
                rel.make_set(s)
            for s in mlist[1:]:
                rel.union(mlist[0], s)
        return rel

    # -- incremental refinement ---------------------------------------------

    def add_states(
        self,
        new_states: Set[str],
        new_transitions: Dict[str, Dict[str, Set[str]]],
        new_labels: Dict[str, Set[str]],
    ) -> None:
        """Add new states and transitions for incremental refinement.

        New states are placed in blocks based on their AP labels and then
        the worklist is repopulated for refinement to continue.
        """
        # extend the index
        for src, amap in new_transitions.items():
            if src not in self._index.fwd:
                self._index.fwd[src] = {}
            for act, targets in amap.items():
                existing = self._index.fwd[src].get(act, set())
                new_targets = targets - existing
                self._index.fwd[src][act] = existing | targets
                self._index._edge_count += len(new_targets)
                for tgt in new_targets:
                    self._index.bwd[tgt][act].add(src)

        self._states |= new_states
        self._labels.update(new_labels)
        self._index.states = self._states

        # place new states in blocks based on labeling
        groups: Dict[Any, Set[str]] = defaultdict(set)
        for s in new_states:
            ap_key = frozenset(new_labels.get(s, set()))
            fair_parts = []
            for i, (b_set, g_set) in enumerate(self._fairness_pairs):
                fair_parts.append((i, s in b_set, s in g_set))
            key = (ap_key, tuple(fair_parts))
            groups[key].add(s)

        for key, members in groups.items():
            # try to find an existing block with matching key
            placed = False
            for blk in list(self._blocks.values()):
                sample = next(iter(blk.members))
                sample_ap = frozenset(self._labels.get(sample, set()))
                sample_fair = []
                for i, (b_set, g_set) in enumerate(self._fairness_pairs):
                    sample_fair.append((i, sample in b_set, sample in g_set))
                sample_key = (sample_ap, tuple(sample_fair))
                if sample_key == key:
                    blk.members |= members
                    for s in members:
                        self._state_block[s] = blk.id
                    for act in self._actions:
                        self._worklist.append((blk.id, act))
                    placed = True
                    break
            if not placed:
                new_blk = self._new_block(members)
                for act in self._actions:
                    self._worklist.append((new_blk.id, act))

    # -- complexity report --------------------------------------------------

    def complexity_report(self) -> Dict[str, Any]:
        """Return complexity metrics for the current state."""
        n = len(self._states)
        m = self._index.edge_count
        k = self.num_blocks
        return {
            "n_states": n,
            "m_transitions": m,
            "k_blocks": k,
            "theoretical_bound": f"O({m} * log({n})) = O({m * max(1, n.bit_length())})",
            "rounds_used": self._round,
            "worklist_remaining": len(self._worklist),
        }

    # -- snapshot -----------------------------------------------------------

    def snapshot(self) -> BisimulationRelation:
        """Return the current partition as a :class:`BisimulationRelation`
        without completing refinement."""
        return self._build_partition()

    # -- convenience --------------------------------------------------------

    @classmethod
    def from_transition_graph(
        cls,
        graph: Any,
        fairness_pairs: Optional[List[Tuple[Set[str], Set[str]]]] = None,
    ) -> "PartitionRefinement":
        """Build a ``PartitionRefinement`` from a
        :class:`~coacert.functor.coalgebra.TransitionGraph`.
        """
        return cls(
            states=graph.states,
            actions=graph.actions,
            transitions=graph.transitions,
            labels=graph.labels,
            fairness_pairs=fairness_pairs,
        )

    @classmethod
    def from_coalgebra(
        cls,
        coalgebra: Any,
    ) -> "PartitionRefinement":
        """Build from an :class:`~coacert.functor.coalgebra.FCoalgebra`."""
        transitions: Dict[str, Dict[str, Set[str]]] = {}
        labels: Dict[str, Set[str]] = {}
        for s_name in coalgebra.states:
            st = coalgebra.get_state(s_name)
            if st is None:
                continue
            labels[s_name] = set(st.propositions)
            transitions[s_name] = {
                act: set(targets) for act, targets in st.successors.items()
            }
        fairness_pairs = [
            (set(fc.b_states), set(fc.g_states))
            for fc in coalgebra.fairness_constraints
        ]
        return cls(
            states=set(coalgebra.states),
            actions=set(coalgebra.actions),
            transitions=transitions,
            labels=labels,
            fairness_pairs=fairness_pairs,
        )
