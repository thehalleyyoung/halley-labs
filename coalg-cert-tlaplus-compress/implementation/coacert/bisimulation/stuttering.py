"""
Stuttering bisimulation via the Groote–Vaandrager algorithm.

Two states *s* and *t* are stuttering bisimilar if for every transition
s → s':
  - Either s' is in the same equivalence class as s (stutter step), or
  - There exists a path t = t₀ → t₁ → ··· → tₖ where t₀, …, tₖ₋₁ are
    equivalent to t and tₖ is equivalent to s'.

This module provides:
  - Iterative Groote–Vaandrager refinement
  - Divergence-sensitive stuttering bisimulation
  - Branching bisimulation computation
  - Stuttering bisimulation verification
  - Counterexample generation
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Deque,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from .relation import BisimulationRelation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StutterCounterexample:
    """Witness that two states are not stuttering bisimilar."""

    state_a: str
    state_b: str
    transition_action: Optional[str]
    transition_target: str
    reason: str
    # path attempted from state_b that fails to match
    attempted_path: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        path_str = " -> ".join(self.attempted_path) if self.attempted_path else "(none)"
        return (
            f"States {self.state_a} and {self.state_b} are not stuttering bisimilar: "
            f"{self.reason}. Transition {self.state_a} -> {self.transition_target} "
            f"cannot be matched. Attempted path from {self.state_b}: {path_str}"
        )


@dataclass
class StutteringResult:
    """Result of a stuttering bisimulation computation."""

    partition: BisimulationRelation
    num_rounds: int
    total_elapsed_ms: float
    initial_blocks: int
    final_blocks: int
    divergence_sensitive: bool
    divergent_states: FrozenSet[str]


# ---------------------------------------------------------------------------
# StutteringBisimulation
# ---------------------------------------------------------------------------

class StutteringBisimulation:
    """Groote–Vaandrager algorithm for stuttering bisimulation.

    Computes the coarsest stuttering bisimulation relation over a given
    labeled transition system.
    """

    def __init__(
        self,
        states: Set[str],
        actions: Set[str],
        transitions: Dict[str, Dict[str, Set[str]]],
        labels: Dict[str, Set[str]],
        divergence_sensitive: bool = False,
    ) -> None:
        self._states = set(states)
        self._actions = set(actions)
        self._transitions = transitions
        self._labels = dict(labels)
        self._divergence_sensitive = divergence_sensitive

        # pre-compute all-action successor/predecessor maps
        self._fwd: Dict[str, Set[str]] = defaultdict(set)
        self._bwd: Dict[str, Set[str]] = defaultdict(set)
        self._fwd_by_act: Dict[str, Dict[str, Set[str]]] = {}
        for src, amap in transitions.items():
            self._fwd_by_act[src] = {}
            for act, targets in amap.items():
                self._fwd[src] |= targets
                self._fwd_by_act[src][act] = set(targets)
                for tgt in targets:
                    self._bwd[tgt].add(src)

        # partition state
        self._partition: Optional[BisimulationRelation] = None
        self._divergent_states: Set[str] = set()

    # -- initial partition --------------------------------------------------

    def _initial_partition(self) -> BisimulationRelation:
        """Group states by atomic proposition labeling."""
        return BisimulationRelation.from_labeling(
            self._states,
            lambda s: frozenset(self._labels.get(s, set())),
        )

    # -- Groote–Vaandrager refinement step ----------------------------------

    def _gv_refine_step(
        self, partition: BisimulationRelation
    ) -> Tuple[BisimulationRelation, bool]:
        """One iteration of the Groote–Vaandrager algorithm.

        For every pair (s, s') where s → s' and s' is not in the same block
        as s, we check that every state t in the block of s can match the
        transition via a stutter path.  If not, we split.

        Returns ``(new_partition, changed)``.
        """
        changed = False
        new_partition = partition.copy()

        for block_members in partition.classes():
            if len(block_members) <= 1:
                continue

            # collect outgoing non-stutter transitions per block member
            # group: target_block_rep -> set of states that can reach it
            reachable_map: Dict[str, Set[str]] = defaultdict(set)

            for s in block_members:
                for tgt in self._fwd.get(s, set()):
                    tgt_rep = partition.find(tgt)
                    src_rep = partition.find(s)
                    if tgt_rep != src_rep:
                        # non-stutter transition: s -> tgt crosses blocks
                        reachable_map[tgt_rep].add(s)

            # for each target block, check if all members of the current
            # block can reach it via a stutter path
            for tgt_rep, can_reach_directly in reachable_map.items():
                # compute the set of states in this block that can reach
                # tgt_rep via stutter paths (intra-block transitions)
                can_reach = self._stutter_reachable(
                    block_members, tgt_rep, partition
                )

                cannot_reach = block_members - can_reach
                if cannot_reach and can_reach:
                    # split: states that can stutter-reach tgt_rep vs those that cannot
                    keep = set(can_reach)
                    remove = set(cannot_reach)

                    # rebuild partition with this split
                    new_partition = self._apply_split(
                        new_partition, block_members, keep, remove
                    )
                    changed = True
                    break  # re-iterate from scratch after a split

            if changed:
                break

        return new_partition, changed

    def _stutter_reachable(
        self,
        block_members: FrozenSet[str],
        target_rep: str,
        partition: BisimulationRelation,
    ) -> Set[str]:
        """Find states in *block_members* that can reach a state in the
        block of *target_rep* via a stutter path.

        A stutter path is a sequence s = s₀ → s₁ → ··· → sₖ where
        s₀, …, sₖ₋₁ are all in the same block and sₖ is in the target block.
        """
        # start from states that have a direct non-stutter transition to target
        seed: Set[str] = set()
        for s in block_members:
            for tgt in self._fwd.get(s, set()):
                if partition.find(tgt) == target_rep:
                    seed.add(s)
                    break

        # backward BFS within block_members: if a state s in the block
        # has a transition to some state already in `can_reach` that is
        # also in the same block, then s can stutter-reach the target
        can_reach = set(seed)
        worklist: Deque[str] = deque(seed)

        while worklist:
            s = worklist.popleft()
            for pred in self._bwd.get(s, set()):
                if pred in block_members and pred not in can_reach:
                    if partition.find(pred) == partition.find(s):
                        can_reach.add(pred)
                        worklist.append(pred)

        return can_reach

    def _apply_split(
        self,
        partition: BisimulationRelation,
        original_block: FrozenSet[str],
        keep: Set[str],
        remove: Set[str],
    ) -> BisimulationRelation:
        """Rebuild the partition with one block split into two."""
        blocks = []
        for cls_members in partition.classes():
            if cls_members == original_block:
                blocks.append(keep)
                blocks.append(remove)
            else:
                blocks.append(set(cls_members))
        return BisimulationRelation.from_blocks(blocks)

    # -- divergence detection -----------------------------------------------

    def _detect_divergent_states(
        self, partition: BisimulationRelation
    ) -> Set[str]:
        """Find states that can diverge (infinite stuttering within a block).

        A state s is divergent if there is an infinite path from s staying
        entirely within the same block (i.e. a cycle of intra-block transitions).
        """
        divergent: Set[str] = set()

        for block_members in partition.classes():
            if len(block_members) <= 1:
                # single state: check self-loop
                s = next(iter(block_members))
                if s in self._fwd.get(s, set()):
                    divergent.add(s)
                continue

            # build intra-block subgraph and find SCCs
            intra_fwd: Dict[str, Set[str]] = defaultdict(set)
            for s in block_members:
                for tgt in self._fwd.get(s, set()):
                    if tgt in block_members:
                        intra_fwd[s].add(tgt)

            sccs = self._tarjan_scc(block_members, intra_fwd)
            for scc in sccs:
                if len(scc) > 1:
                    divergent |= scc
                elif len(scc) == 1:
                    s = next(iter(scc))
                    if s in intra_fwd.get(s, set()):
                        divergent.add(s)

        return divergent

    def _tarjan_scc(
        self, nodes: FrozenSet[str], fwd: Dict[str, Set[str]]
    ) -> List[Set[str]]:
        """Tarjan's SCC algorithm on a subgraph."""
        index_counter = [0]
        stack: List[str] = []
        on_stack: Set[str] = set()
        indices: Dict[str, int] = {}
        lowlinks: Dict[str, int] = {}
        sccs: List[Set[str]] = []

        def strongconnect(v: str) -> None:
            indices[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in fwd.get(v, set()):
                if w not in nodes:
                    continue
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                scc: Set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in nodes:
            if v not in indices:
                strongconnect(v)

        return sccs

    def _refine_by_divergence(
        self,
        partition: BisimulationRelation,
        divergent: Set[str],
    ) -> Tuple[BisimulationRelation, bool]:
        """Split blocks by divergence: divergent vs non-divergent states."""
        changed = False
        blocks: List[Set[str]] = []

        for cls_members in partition.classes():
            div_in = cls_members & divergent
            div_out = cls_members - divergent
            if div_in and div_out:
                blocks.append(set(div_in))
                blocks.append(set(div_out))
                changed = True
            else:
                blocks.append(set(cls_members))

        if changed:
            return BisimulationRelation.from_blocks(blocks), True
        return partition, False

    # -- main entry point ---------------------------------------------------

    def compute(self, max_rounds: int = 100_000) -> StutteringResult:
        """Compute the coarsest stuttering bisimulation.

        Returns a :class:`StutteringResult` with the final partition.
        """
        t0 = time.monotonic()
        partition = self._initial_partition()
        initial_blocks = partition.num_classes()

        round_num = 0
        while round_num < max_rounds:
            round_num += 1
            partition, changed = self._gv_refine_step(partition)
            if not changed:
                break

        # divergence-sensitive variant
        if self._divergence_sensitive:
            self._divergent_states = self._detect_divergent_states(partition)
            if self._divergent_states:
                partition, div_changed = self._refine_by_divergence(
                    partition, self._divergent_states
                )
                if div_changed:
                    # re-run GV refinement after divergence split
                    extra_rounds = 0
                    while extra_rounds < max_rounds:
                        extra_rounds += 1
                        round_num += 1
                        partition, changed = self._gv_refine_step(partition)
                        if not changed:
                            break

        self._partition = partition
        elapsed = (time.monotonic() - t0) * 1000

        logger.info(
            "Stuttering bisimulation: %d rounds, %d -> %d blocks (%.1f ms)",
            round_num,
            initial_blocks,
            partition.num_classes(),
            elapsed,
        )

        return StutteringResult(
            partition=partition,
            num_rounds=round_num,
            total_elapsed_ms=elapsed,
            initial_blocks=initial_blocks,
            final_blocks=partition.num_classes(),
            divergence_sensitive=self._divergence_sensitive,
            divergent_states=frozenset(self._divergent_states),
        )

    # -- branching bisimulation ---------------------------------------------

    def compute_branching(self, max_rounds: int = 100_000) -> StutteringResult:
        """Compute branching bisimulation.

        Branching bisimulation is like stuttering bisimulation but
        additionally requires that the intermediate states on a stutter
        path are equivalent to *both* the source and target (not just
        the source).
        """
        t0 = time.monotonic()
        partition = self._initial_partition()
        initial_blocks = partition.num_classes()

        round_num = 0
        while round_num < max_rounds:
            round_num += 1
            partition, changed = self._branching_refine_step(partition)
            if not changed:
                break

        self._partition = partition
        elapsed = (time.monotonic() - t0) * 1000

        return StutteringResult(
            partition=partition,
            num_rounds=round_num,
            total_elapsed_ms=elapsed,
            initial_blocks=initial_blocks,
            final_blocks=partition.num_classes(),
            divergence_sensitive=False,
            divergent_states=frozenset(),
        )

    def _branching_refine_step(
        self, partition: BisimulationRelation
    ) -> Tuple[BisimulationRelation, bool]:
        """One refinement step for branching bisimulation.

        The difference from stuttering: intermediate states on the matching
        path from t must be equivalent to *s* (not just t).
        """
        changed = False
        new_partition = partition.copy()

        for block_members in partition.classes():
            if len(block_members) <= 1:
                continue

            for s in block_members:
                for tgt in self._fwd.get(s, set()):
                    tgt_rep = partition.find(tgt)
                    src_rep = partition.find(s)
                    if tgt_rep == src_rep:
                        continue  # stutter step

                    # check if all other members of the block can match
                    can_match: Set[str] = set()
                    cannot_match: Set[str] = set()

                    for t in block_members:
                        if self._can_branching_match(
                            t, src_rep, tgt_rep, partition, block_members
                        ):
                            can_match.add(t)
                        else:
                            cannot_match.add(t)

                    if cannot_match and can_match:
                        new_partition = self._apply_split(
                            new_partition, block_members, can_match, cannot_match
                        )
                        changed = True
                        break

                if changed:
                    break
            if changed:
                break

        return new_partition, changed

    def _can_branching_match(
        self,
        t: str,
        src_rep: str,
        tgt_rep: str,
        partition: BisimulationRelation,
        block_members: FrozenSet[str],
    ) -> bool:
        """Check if state *t* can match a transition to *tgt_rep* via a
        branching stutter path.

        All intermediate states must be in the source block (equiv to src).
        """
        # BFS from t, staying within the source block
        visited: Set[str] = set()
        worklist: Deque[str] = deque([t])

        while worklist:
            current = worklist.popleft()
            if current in visited:
                continue
            visited.add(current)

            for succ in self._fwd.get(current, set()):
                succ_rep = partition.find(succ)
                if succ_rep == tgt_rep:
                    return True  # found matching transition
                if succ_rep == src_rep and succ in block_members:
                    worklist.append(succ)  # continue stutter path

        return False

    # -- verification -------------------------------------------------------

    def verify(
        self,
        partition: BisimulationRelation,
        stuttering: bool = True,
    ) -> Tuple[bool, List[StutterCounterexample]]:
        """Verify that a partition is a (stuttering) bisimulation.

        Returns ``(is_valid, counterexamples)``.
        """
        counterexamples: List[StutterCounterexample] = []

        for block_members in partition.classes():
            if len(block_members) <= 1:
                continue

            # check AP consistency
            ap_sets = {frozenset(self._labels.get(s, set())) for s in block_members}
            if len(ap_sets) > 1:
                mlist = sorted(block_members)
                counterexamples.append(StutterCounterexample(
                    state_a=mlist[0],
                    state_b=mlist[1],
                    transition_action=None,
                    transition_target=mlist[0],
                    reason="Different atomic propositions in same block",
                ))
                continue

            # for each state s and each non-stutter transition s -> s',
            # check that every other state t in the block can match it
            for s in block_members:
                for tgt in self._fwd.get(s, set()):
                    tgt_rep = partition.find(tgt)
                    src_rep = partition.find(s)
                    if tgt_rep == src_rep:
                        continue  # stutter step

                    for t in block_members:
                        if t == s:
                            continue
                        if stuttering:
                            can_match = bool(
                                self._stutter_reachable(
                                    block_members, tgt_rep, partition
                                ) & {t}
                            )
                        else:
                            # strong bisimulation: t must have direct transition
                            can_match = any(
                                partition.find(tgt2) == tgt_rep
                                for tgt2 in self._fwd.get(t, set())
                            )

                        if not can_match:
                            counterexamples.append(StutterCounterexample(
                                state_a=s,
                                state_b=t,
                                transition_action=self._find_action(s, tgt),
                                transition_target=tgt,
                                reason=(
                                    f"State {t} cannot match transition "
                                    f"{s} -> {tgt}"
                                ),
                            ))

        return len(counterexamples) == 0, counterexamples

    def _find_action(self, src: str, tgt: str) -> Optional[str]:
        """Find the action label for a transition src -> tgt."""
        for act, targets in self._fwd_by_act.get(src, {}).items():
            if tgt in targets:
                return act
        return None

    # -- counterexample generation ------------------------------------------

    def generate_counterexample(
        self,
        state_a: str,
        state_b: str,
    ) -> Optional[StutterCounterexample]:
        """Generate a counterexample showing that two states are not
        stuttering bisimilar, or ``None`` if they are."""
        if self._partition is None:
            self.compute()
        assert self._partition is not None

        if self._partition.equivalent(state_a, state_b):
            return None

        # find a distinguishing transition
        for tgt in self._fwd.get(state_a, set()):
            tgt_rep = self._partition.find(tgt)
            src_rep = self._partition.find(state_a)
            if tgt_rep == src_rep:
                continue

            b_block = self._partition.class_of(state_b)
            can_reach = self._stutter_reachable(
                b_block, tgt_rep, self._partition
            )
            if state_b not in can_reach:
                # try to build an attempted path for diagnostics
                path = self._best_effort_path(state_b, tgt_rep, b_block, self._partition)
                return StutterCounterexample(
                    state_a=state_a,
                    state_b=state_b,
                    transition_action=self._find_action(state_a, tgt),
                    transition_target=tgt,
                    reason=f"No stutter path from {state_b} reaching block of {tgt}",
                    attempted_path=path,
                )

        # try from state_b's side
        for tgt in self._fwd.get(state_b, set()):
            tgt_rep = self._partition.find(tgt)
            src_rep = self._partition.find(state_b)
            if tgt_rep == src_rep:
                continue

            a_block = self._partition.class_of(state_a)
            can_reach = self._stutter_reachable(
                a_block, tgt_rep, self._partition
            )
            if state_a not in can_reach:
                path = self._best_effort_path(state_a, tgt_rep, a_block, self._partition)
                return StutterCounterexample(
                    state_a=state_b,
                    state_b=state_a,
                    transition_action=self._find_action(state_b, tgt),
                    transition_target=tgt,
                    reason=f"No stutter path from {state_a} reaching block of {tgt}",
                    attempted_path=path,
                )

        return StutterCounterexample(
            state_a=state_a,
            state_b=state_b,
            transition_action=None,
            transition_target="(unknown)",
            reason="States are in different blocks but no explicit witness found",
        )

    def _best_effort_path(
        self,
        start: str,
        target_rep: str,
        block_members: FrozenSet[str],
        partition: BisimulationRelation,
    ) -> List[str]:
        """BFS to find the longest stutter path toward *target_rep*."""
        visited: Set[str] = set()
        parent: Dict[str, Optional[str]] = {start: None}
        best_endpoint = start
        best_depth = 0
        worklist: Deque[Tuple[str, int]] = deque([(start, 0)])

        while worklist:
            current, depth = worklist.popleft()
            if current in visited:
                continue
            visited.add(current)

            if depth > best_depth:
                best_depth = depth
                best_endpoint = current

            for succ in self._fwd.get(current, set()):
                if partition.find(succ) == target_rep:
                    parent[succ] = current
                    # reconstruct path
                    path = [succ]
                    node: Optional[str] = current
                    while node is not None:
                        path.append(node)
                        node = parent.get(node)
                    path.reverse()
                    return path
                if succ in block_members and succ not in visited:
                    parent[succ] = current
                    worklist.append((succ, depth + 1))

        # no full path found; return partial
        path = [best_endpoint]
        node = parent.get(best_endpoint)
        while node is not None:
            path.append(node)
            node = parent.get(node)
        path.reverse()
        return path

    # -- comparison ---------------------------------------------------------

    def compare_with_strong(
        self,
    ) -> Tuple[BisimulationRelation, BisimulationRelation, List[Tuple[str, str]]]:
        """Compare stuttering bisimulation with strong bisimulation.

        Returns ``(stuttering_partition, strong_partition, extra_merges)``
        where *extra_merges* are pairs merged by stuttering but not strong.
        """
        from .partition_refinement import PartitionRefinement

        stut_result = self.compute()
        stut_part = stut_result.partition

        pr = PartitionRefinement(
            states=self._states,
            actions=self._actions,
            transitions=self._transitions,
            labels=self._labels,
        )
        strong_result = pr.refine()
        strong_part = strong_result.partition

        extra = stut_part.difference_witnesses(strong_part)
        return stut_part, strong_part, extra

    # -- convenience constructors -------------------------------------------

    @classmethod
    def from_transition_graph(
        cls,
        graph: Any,
        divergence_sensitive: bool = False,
    ) -> "StutteringBisimulation":
        return cls(
            states=graph.states,
            actions=graph.actions,
            transitions=graph.transitions,
            labels=graph.labels,
            divergence_sensitive=divergence_sensitive,
        )

    @classmethod
    def from_coalgebra(
        cls,
        coalgebra: Any,
        divergence_sensitive: bool = False,
    ) -> "StutteringBisimulation":
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
        return cls(
            states=set(coalgebra.states),
            actions=set(coalgebra.actions),
            transitions=transitions,
            labels=labels,
            divergence_sensitive=divergence_sensitive,
        )
