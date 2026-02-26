"""
Fairness-respecting equivalence.

Extends stuttering bisimulation with Streett/Rabin fairness constraints.
Two states *s* and *t* are fair-equivalent if they are stuttering bisimilar
AND, for every acceptance pair (Bᵢ, Gᵢ):
  - If a fair path from s visits Bᵢ infinitely often and Gᵢ infinitely often,
    then there exists a corresponding fair path from t.

Provides:
  - Fair equivalence relation computation (refinement over stuttering bisimulation)
  - Integration with T-Fair coherence checking
  - Fairness preservation verification in quotient systems
  - Fair cycle detection in quotient
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Deque,
    Dict,
    FrozenSet,
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
class FairCycleInfo:
    """A fair cycle in the quotient (an accepting SCC)."""

    states: FrozenSet[str]
    satisfied_pairs: List[int]  # indices of acceptance pairs satisfied


@dataclass
class FairEquivalenceResult:
    """Result of a fairness-respecting equivalence computation."""

    partition: BisimulationRelation
    stuttering_partition: BisimulationRelation
    num_fairness_splits: int
    total_rounds: int
    fair_classes: int
    stuttering_classes: int


@dataclass
class FairnessVerificationResult:
    """Result of verifying fairness preservation in a quotient."""

    is_preserved: bool
    pair_results: List[Tuple[int, bool, str]]  # (index, ok, message)
    fair_cycles_original: int
    fair_cycles_quotient: int


# ---------------------------------------------------------------------------
# FairnessEquivalence
# ---------------------------------------------------------------------------

class FairnessEquivalence:
    """Fairness-respecting equivalence refining stuttering bisimulation.

    Start with the coarsest stuttering bisimulation, then split blocks
    where states disagree on fairness-reachability.
    """

    def __init__(
        self,
        states: Set[str],
        actions: Set[str],
        transitions: Dict[str, Dict[str, Set[str]]],
        labels: Dict[str, Set[str]],
        fairness_pairs: List[Tuple[Set[str], Set[str]]],
    ) -> None:
        self._states = set(states)
        self._actions = set(actions)
        self._transitions = transitions
        self._labels = dict(labels)
        self._fairness_pairs = list(fairness_pairs)

        # precompute successor/predecessor
        self._fwd: Dict[str, Set[str]] = defaultdict(set)
        self._bwd: Dict[str, Set[str]] = defaultdict(set)
        for src, amap in transitions.items():
            for _act, targets in amap.items():
                self._fwd[src] |= targets
                for tgt in targets:
                    self._bwd[tgt].add(src)

    # -- core: refine stuttering partition by fairness ----------------------

    def compute(
        self,
        stuttering_partition: Optional[BisimulationRelation] = None,
        max_rounds: int = 10_000,
    ) -> FairEquivalenceResult:
        """Compute the fairness-respecting equivalence.

        If *stuttering_partition* is not provided, it is computed internally
        via :class:`StutteringBisimulation`.
        """
        if stuttering_partition is None:
            from .stuttering import StutteringBisimulation

            sb = StutteringBisimulation(
                states=self._states,
                actions=self._actions,
                transitions=self._transitions,
                labels=self._labels,
            )
            stut_result = sb.compute()
            stuttering_partition = stut_result.partition

        stuttering_classes = stuttering_partition.num_classes()
        partition = stuttering_partition.copy()

        total_splits = 0
        round_num = 0

        while round_num < max_rounds:
            round_num += 1
            changed = False

            # for each acceptance pair, refine by fair reachability
            for pair_idx, (b_set, g_set) in enumerate(self._fairness_pairs):
                splits = self._refine_by_fair_pair(
                    partition, pair_idx, b_set, g_set
                )
                if splits > 0:
                    total_splits += splits
                    changed = True

            if not changed:
                break

        logger.info(
            "Fair equivalence: %d rounds, %d fairness splits, "
            "%d -> %d blocks",
            round_num,
            total_splits,
            stuttering_classes,
            partition.num_classes(),
        )

        return FairEquivalenceResult(
            partition=partition,
            stuttering_partition=stuttering_partition,
            num_fairness_splits=total_splits,
            total_rounds=round_num,
            fair_classes=partition.num_classes(),
            stuttering_classes=stuttering_classes,
        )

    def _refine_by_fair_pair(
        self,
        partition: BisimulationRelation,
        pair_idx: int,
        b_set: Set[str],
        g_set: Set[str],
    ) -> int:
        """Refine by a single acceptance pair (Bᵢ, Gᵢ).

        Split blocks where some states can reach a fair cycle visiting
        Bᵢ and Gᵢ infinitely, and some cannot.

        Returns the number of splits performed.
        """
        # find all states that lie on a fair SCC for this pair
        fair_sccs = self._fair_sccs_for_pair(b_set, g_set)
        on_fair_scc: Set[str] = set()
        for scc in fair_sccs:
            on_fair_scc |= scc

        # compute backward reachability from fair SCCs
        can_reach_fair = self._backward_reach(on_fair_scc)

        # refine: split each block into states that can/cannot reach fair cycle
        splits = 0
        blocks = list(partition.classes())
        new_blocks: List[Set[str]] = []
        need_rebuild = False

        for block_members in blocks:
            yes = block_members & can_reach_fair
            no = block_members - can_reach_fair
            if yes and no:
                new_blocks.append(set(yes))
                new_blocks.append(set(no))
                splits += 1
                need_rebuild = True
            else:
                new_blocks.append(set(block_members))

        if need_rebuild:
            rebuilt = BisimulationRelation.from_blocks(new_blocks)
            # update partition in place
            partition._parent = rebuilt._parent
            partition._rank = rebuilt._rank
            partition._size = rebuilt._size
            partition._invalidate()

        return splits

    def _fair_sccs_for_pair(
        self, b_set: Set[str], g_set: Set[str]
    ) -> List[Set[str]]:
        """Find SCCs that satisfy the Streett condition for one pair.

        An SCC satisfies the pair (B, G) if it either does not intersect B
        or it intersects G (i.e. visiting B infinitely often implies
        visiting G infinitely often).
        """
        sccs = self._tarjan_scc(self._states, self._fwd)
        fair: List[Set[str]] = []
        for scc in sccs:
            if len(scc) <= 1:
                s = next(iter(scc))
                if s not in self._fwd.get(s, set()):
                    continue  # trivial SCC, no self-loop
            intersects_b = bool(scc & b_set)
            intersects_g = bool(scc & g_set)
            if not intersects_b or intersects_g:
                fair.append(scc)
        return fair

    def _backward_reach(self, targets: Set[str]) -> Set[str]:
        """Backward reachability from *targets*."""
        reached = set(targets)
        worklist: Deque[str] = deque(targets)
        while worklist:
            s = worklist.popleft()
            for pred in self._bwd.get(s, set()):
                if pred not in reached:
                    reached.add(pred)
                    worklist.append(pred)
        return reached

    def _tarjan_scc(
        self, nodes: Set[str], fwd: Dict[str, Set[str]]
    ) -> List[Set[str]]:
        """Tarjan's SCC algorithm."""
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

    # -- T-Fair coherence integration ---------------------------------------

    def check_tfair_coherence(
        self,
        partition: BisimulationRelation,
    ) -> List[Tuple[int, str]]:
        """Check T-Fair coherence: for every acceptance pair, every quotient
        block that intersects Bᵢ must also intersect Gᵢ within the same
        fair SCC.

        Returns a list of ``(pair_index, violation_message)`` for failures.
        """
        violations: List[Tuple[int, str]] = []

        for pair_idx, (b_set, g_set) in enumerate(self._fairness_pairs):
            for cls_members in partition.classes():
                cls_in_b = cls_members & b_set
                cls_in_g = cls_members & g_set
                if cls_in_b and not cls_in_g:
                    rep = partition.find(next(iter(cls_members)))
                    violations.append((
                        pair_idx,
                        f"Block {rep} intersects B_{pair_idx} "
                        f"but not G_{pair_idx}",
                    ))

        return violations

    # -- fairness verification in quotient ----------------------------------

    def verify_fairness_in_quotient(
        self,
        partition: BisimulationRelation,
    ) -> FairnessVerificationResult:
        """Verify that fairness constraints are preserved by the quotient.

        For every acceptance pair, fair cycles in the original system must
        have corresponding fair cycles in the quotient.
        """
        quotient_map = {}
        for s in self._states:
            quotient_map[s] = partition.find(s)

        # build quotient transition relation
        q_fwd: Dict[str, Set[str]] = defaultdict(set)
        for src, amap in self._transitions.items():
            q_src = quotient_map.get(src, src)
            for _act, targets in amap.items():
                for tgt in targets:
                    q_tgt = quotient_map.get(tgt, tgt)
                    if q_src != q_tgt:
                        q_fwd[q_src].add(q_tgt)

        pair_results: List[Tuple[int, bool, str]] = []
        original_fair_cycles = 0
        quotient_fair_cycles = 0

        for pair_idx, (b_set, g_set) in enumerate(self._fairness_pairs):
            orig_sccs = self._fair_sccs_for_pair(b_set, g_set)
            original_fair_cycles += len(orig_sccs)

            # quotient b/g sets
            q_b = {quotient_map.get(s, s) for s in b_set}
            q_g = {quotient_map.get(s, s) for s in g_set}

            q_states = set(partition.representatives())
            q_sccs = self._tarjan_scc(q_states, q_fwd)
            q_fair = []
            for scc in q_sccs:
                if len(scc) <= 1:
                    s = next(iter(scc))
                    if s not in q_fwd.get(s, set()):
                        continue
                intersects_b = bool(scc & q_b)
                intersects_g = bool(scc & q_g)
                if not intersects_b or intersects_g:
                    q_fair.append(scc)
            quotient_fair_cycles += len(q_fair)

            # check that every original fair SCC maps to a quotient fair SCC
            ok = True
            for orig_scc in orig_sccs:
                mapped_reps = {quotient_map.get(s, s) for s in orig_scc}
                found_covering = any(
                    mapped_reps & q_scc for q_scc in q_fair
                )
                if not found_covering:
                    ok = False
                    break

            msg = "preserved" if ok else "NOT preserved"
            pair_results.append((pair_idx, ok, msg))

        all_ok = all(ok for _, ok, _ in pair_results)
        return FairnessVerificationResult(
            is_preserved=all_ok,
            pair_results=pair_results,
            fair_cycles_original=original_fair_cycles,
            fair_cycles_quotient=quotient_fair_cycles,
        )

    # -- fair cycle detection in quotient -----------------------------------

    def detect_fair_cycles_in_quotient(
        self,
        partition: BisimulationRelation,
    ) -> List[FairCycleInfo]:
        """Detect all fair cycles (accepting SCCs) in the quotient system."""
        quotient_map = {s: partition.find(s) for s in self._states}

        q_fwd: Dict[str, Set[str]] = defaultdict(set)
        for src, amap in self._transitions.items():
            q_src = quotient_map.get(src, src)
            for _act, targets in amap.items():
                for tgt in targets:
                    q_tgt = quotient_map.get(tgt, tgt)
                    if q_src != q_tgt:
                        q_fwd[q_src].add(q_tgt)

        q_states = set(partition.representatives())
        q_sccs = self._tarjan_scc(q_states, q_fwd)
        fair_cycles: List[FairCycleInfo] = []

        for scc in q_sccs:
            if len(scc) <= 1:
                s = next(iter(scc))
                if s not in q_fwd.get(s, set()):
                    continue  # trivial

            # check each acceptance pair
            satisfied: List[int] = []
            for pair_idx, (b_set, g_set) in enumerate(self._fairness_pairs):
                q_b = {quotient_map.get(s, s) for s in b_set}
                q_g = {quotient_map.get(s, s) for s in g_set}
                intersects_b = bool(scc & q_b)
                intersects_g = bool(scc & q_g)
                if not intersects_b or intersects_g:
                    satisfied.append(pair_idx)

            # an SCC is accepting (Streett) if it satisfies ALL pairs
            if len(satisfied) == len(self._fairness_pairs):
                fair_cycles.append(FairCycleInfo(
                    states=frozenset(scc),
                    satisfied_pairs=satisfied,
                ))

        return fair_cycles

    # -- convenience constructors -------------------------------------------

    @classmethod
    def from_transition_graph(
        cls,
        graph: Any,
        fairness_pairs: List[Tuple[Set[str], Set[str]]],
    ) -> "FairnessEquivalence":
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
    ) -> "FairnessEquivalence":
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
