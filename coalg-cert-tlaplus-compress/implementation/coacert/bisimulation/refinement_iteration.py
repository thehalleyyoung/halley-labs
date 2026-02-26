"""
Refinement iteration engine.

Orchestrates the full bisimulation computation pipeline:
  1. Start with the coarsest partition (by AP labeling).
  2. Refine by action successors (Paige–Tarjan).
  3. Incorporate stuttering (Groote–Vaandrager).
  4. Incorporate fairness (fair-equivalence refinement).
  5. Iterate until fixed point.

Supports different strategies (eager, lazy), convergence detection,
and comparison with a learned bisimulation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
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
# Strategy
# ---------------------------------------------------------------------------

class RefinementStrategy(Enum):
    """Strategy for ordering refinement phases."""

    EAGER = "eager"    # run all phases in each round
    LAZY = "lazy"      # run phases only when previous one stabilises
    AP_THEN_TRANSITION = "ap_then_transition"  # AP first, then transition
    STUTTERING_FIRST = "stuttering_first"      # stuttering before strong


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PhaseRecord:
    """Record for a single phase execution."""

    phase: str
    blocks_before: int
    blocks_after: int
    elapsed_ms: float
    splits: int

    @property
    def changed(self) -> bool:
        return self.blocks_before != self.blocks_after


@dataclass
class RoundRecord:
    """Record of a full round (one iteration over all phases)."""

    round_number: int
    phases: List[PhaseRecord]
    total_blocks_before: int
    total_blocks_after: int
    elapsed_ms: float

    @property
    def changed(self) -> bool:
        return self.total_blocks_before != self.total_blocks_after


@dataclass
class EngineResult:
    """Result of the full refinement engine computation."""

    partition: BisimulationRelation
    rounds: List[RoundRecord]
    total_rounds: int
    total_elapsed_ms: float
    initial_blocks: int
    final_blocks: int
    strategy: str
    converged: bool
    stuttering_used: bool
    fairness_used: bool

    def summary(self) -> str:
        return (
            f"RefinementEngine: {self.total_rounds} rounds, "
            f"{self.initial_blocks} -> {self.final_blocks} blocks, "
            f"converged={self.converged}, "
            f"strategy={self.strategy}, "
            f"{self.total_elapsed_ms:.1f} ms"
        )


@dataclass
class ComparisonResult:
    """Comparison between two partitions (e.g. learned vs computed)."""

    partition_a_blocks: int
    partition_b_blocks: int
    a_finer_than_b: bool
    b_finer_than_a: bool
    are_equal: bool
    extra_merges_in_a: List[Tuple[str, str]]
    extra_merges_in_b: List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# RefinementEngine
# ---------------------------------------------------------------------------

class RefinementEngine:
    """Orchestrate the full bisimulation computation pipeline.

    Combines AP-refinement, action-refinement, stuttering, and fairness
    into a single convergence loop.
    """

    def __init__(
        self,
        states: Set[str],
        actions: Set[str],
        transitions: Dict[str, Dict[str, Set[str]]],
        labels: Dict[str, Set[str]],
        fairness_pairs: Optional[List[Tuple[Set[str], Set[str]]]] = None,
        strategy: RefinementStrategy = RefinementStrategy.EAGER,
        use_stuttering: bool = True,
        use_fairness: bool = True,
        divergence_sensitive: bool = False,
    ) -> None:
        self._states = set(states)
        self._actions = set(actions)
        self._transitions = transitions
        self._labels = dict(labels)
        self._fairness_pairs: List[Tuple[Set[str], Set[str]]] = (
            list(fairness_pairs) if fairness_pairs else []
        )
        self._strategy = strategy
        self._use_stuttering = use_stuttering
        self._use_fairness = use_fairness and bool(self._fairness_pairs)
        self._divergence_sensitive = divergence_sensitive

        self._rounds: List[RoundRecord] = []

    # -- phase executors ----------------------------------------------------

    def _phase_ap(self, partition: BisimulationRelation) -> PhaseRecord:
        """Refine partition by atomic proposition labeling."""
        t0 = time.monotonic()
        before = partition.num_classes()
        splits = partition.refine_by(
            lambda s: frozenset(self._labels.get(s, set()))
        )
        after = partition.num_classes()
        elapsed = (time.monotonic() - t0) * 1000
        return PhaseRecord(
            phase="ap",
            blocks_before=before,
            blocks_after=after,
            elapsed_ms=elapsed,
            splits=splits,
        )

    def _phase_action(self, partition: BisimulationRelation) -> PhaseRecord:
        """Refine partition by action successor patterns.

        For each action and each block B, split other blocks into states
        that have/lack a successor in B under that action.
        """
        t0 = time.monotonic()
        before = partition.num_classes()
        total_splits = 0

        changed = True
        while changed:
            changed = False
            for action in self._actions:
                disc_key = self._action_discriminator(partition, action)
                s = partition.refine_by(disc_key)
                if s > 0:
                    total_splits += s
                    changed = True

        after = partition.num_classes()
        elapsed = (time.monotonic() - t0) * 1000
        return PhaseRecord(
            phase="action",
            blocks_before=before,
            blocks_after=after,
            elapsed_ms=elapsed,
            splits=total_splits,
        )

    def _action_discriminator(
        self, partition: BisimulationRelation, action: str
    ) -> Any:
        """Return a discriminator function: state -> set of target block reps."""

        def disc(s: str) -> FrozenSet[str]:
            targets = self._transitions.get(s, {}).get(action, set())
            return frozenset(partition.find(t) for t in targets)

        return disc

    def _phase_stuttering(
        self, partition: BisimulationRelation
    ) -> PhaseRecord:
        """Refine using stuttering bisimulation (Groote–Vaandrager)."""
        from .stuttering import StutteringBisimulation

        t0 = time.monotonic()
        before = partition.num_classes()

        sb = StutteringBisimulation(
            states=self._states,
            actions=self._actions,
            transitions=self._transitions,
            labels=self._labels,
            divergence_sensitive=self._divergence_sensitive,
        )
        stut_result = sb.compute()

        # intersect with current partition (keep the finer of the two)
        refined = partition.intersect(stut_result.partition)
        partition._parent = refined._parent
        partition._rank = refined._rank
        partition._size = refined._size
        partition._invalidate()

        after = partition.num_classes()
        elapsed = (time.monotonic() - t0) * 1000
        return PhaseRecord(
            phase="stuttering",
            blocks_before=before,
            blocks_after=after,
            elapsed_ms=elapsed,
            splits=after - before,
        )

    def _phase_fairness(self, partition: BisimulationRelation) -> PhaseRecord:
        """Refine by fairness constraints."""
        from .fairness_equiv import FairnessEquivalence

        t0 = time.monotonic()
        before = partition.num_classes()

        fe = FairnessEquivalence(
            states=self._states,
            actions=self._actions,
            transitions=self._transitions,
            labels=self._labels,
            fairness_pairs=self._fairness_pairs,
        )
        fair_result = fe.compute(stuttering_partition=partition)

        partition._parent = fair_result.partition._parent
        partition._rank = fair_result.partition._rank
        partition._size = fair_result.partition._size
        partition._invalidate()

        after = partition.num_classes()
        elapsed = (time.monotonic() - t0) * 1000
        return PhaseRecord(
            phase="fairness",
            blocks_before=before,
            blocks_after=after,
            elapsed_ms=elapsed,
            splits=fair_result.num_fairness_splits,
        )

    # -- phase ordering by strategy -----------------------------------------

    def _get_phases(self) -> List[str]:
        """Return the phase names for the current strategy."""
        if self._strategy == RefinementStrategy.STUTTERING_FIRST:
            phases = ["ap"]
            if self._use_stuttering:
                phases.append("stuttering")
            phases.append("action")
            if self._use_fairness:
                phases.append("fairness")
            return phases

        if self._strategy == RefinementStrategy.AP_THEN_TRANSITION:
            phases = ["ap", "action"]
            if self._use_stuttering:
                phases.append("stuttering")
            if self._use_fairness:
                phases.append("fairness")
            return phases

        # EAGER / LAZY default ordering
        phases = ["ap", "action"]
        if self._use_stuttering:
            phases.append("stuttering")
        if self._use_fairness:
            phases.append("fairness")
        return phases

    def _run_phase(
        self, phase: str, partition: BisimulationRelation
    ) -> PhaseRecord:
        dispatch = {
            "ap": self._phase_ap,
            "action": self._phase_action,
            "stuttering": self._phase_stuttering,
            "fairness": self._phase_fairness,
        }
        return dispatch[phase](partition)

    # -- main loop ----------------------------------------------------------

    def run(self, max_rounds: int = 1000) -> EngineResult:
        """Run the refinement engine to convergence.

        Returns an :class:`EngineResult` with the final partition.
        """
        t0 = time.monotonic()

        # start from coarsest partition (all states in one class)
        partition = BisimulationRelation.coarsest(self._states)
        initial_blocks = partition.num_classes()

        phases = self._get_phases()
        converged = False

        for round_num in range(1, max_rounds + 1):
            t_round = time.monotonic()
            blocks_before = partition.num_classes()
            phase_records: List[PhaseRecord] = []

            if self._strategy == RefinementStrategy.LAZY:
                # lazy: run phases sequentially, stop at first one that changes
                for phase in phases:
                    record = self._run_phase(phase, partition)
                    phase_records.append(record)
                    if record.changed:
                        break
            else:
                # eager / others: run all phases in each round
                for phase in phases:
                    record = self._run_phase(phase, partition)
                    phase_records.append(record)

            blocks_after = partition.num_classes()
            round_elapsed = (time.monotonic() - t_round) * 1000

            rr = RoundRecord(
                round_number=round_num,
                phases=phase_records,
                total_blocks_before=blocks_before,
                total_blocks_after=blocks_after,
                elapsed_ms=round_elapsed,
            )
            self._rounds.append(rr)

            logger.debug(
                "Round %d: %d -> %d blocks (%.1f ms)",
                round_num,
                blocks_before,
                blocks_after,
                round_elapsed,
            )

            if not rr.changed:
                converged = True
                break

        total_elapsed = (time.monotonic() - t0) * 1000

        result = EngineResult(
            partition=partition,
            rounds=self._rounds,
            total_rounds=len(self._rounds),
            total_elapsed_ms=total_elapsed,
            initial_blocks=initial_blocks,
            final_blocks=partition.num_classes(),
            strategy=self._strategy.value,
            converged=converged,
            stuttering_used=self._use_stuttering,
            fairness_used=self._use_fairness,
        )

        logger.info(result.summary())
        return result

    # -- comparison with learned bisimulation --------------------------------

    @staticmethod
    def compare(
        partition_a: BisimulationRelation,
        partition_b: BisimulationRelation,
    ) -> ComparisonResult:
        """Compare two partitions (e.g. learned vs computed)."""
        a_finer = partition_a.is_finer_than(partition_b)
        b_finer = partition_b.is_finer_than(partition_a)
        are_equal = a_finer and b_finer

        extra_a = partition_a.difference_witnesses(partition_b)
        extra_b = partition_b.difference_witnesses(partition_a)

        return ComparisonResult(
            partition_a_blocks=partition_a.num_classes(),
            partition_b_blocks=partition_b.num_classes(),
            a_finer_than_b=a_finer,
            b_finer_than_a=b_finer,
            are_equal=are_equal,
            extra_merges_in_a=extra_a,
            extra_merges_in_b=extra_b,
        )

    # -- incremental: on-the-fly refinement with learner --------------------

    def refine_with_hint(
        self,
        current: BisimulationRelation,
        learned_partition: BisimulationRelation,
    ) -> BisimulationRelation:
        """Refine *current* using a learned partition as a hint.

        The result is the intersection (meet) of the current partition
        and the learned one, followed by one round of action refinement.
        """
        refined = current.intersect(learned_partition)

        # one round of action refinement
        changed = True
        while changed:
            changed = False
            for action in self._actions:
                disc = self._action_discriminator(refined, action)
                s = refined.refine_by(disc)
                if s > 0:
                    changed = True

        return refined

    # -- convergence report -------------------------------------------------

    def convergence_report(self) -> Dict[str, Any]:
        """Generate a convergence report over all rounds."""
        if not self._rounds:
            return {"status": "not_run"}

        block_counts = [r.total_blocks_before for r in self._rounds]
        block_counts.append(self._rounds[-1].total_blocks_after)

        phase_times: Dict[str, float] = {}
        for rr in self._rounds:
            for pr in rr.phases:
                phase_times.setdefault(pr.phase, 0.0)
                phase_times[pr.phase] += pr.elapsed_ms

        return {
            "total_rounds": len(self._rounds),
            "converged": self._rounds[-1].total_blocks_before == self._rounds[-1].total_blocks_after,
            "block_progression": block_counts,
            "phase_time_ms": phase_times,
            "total_time_ms": sum(r.elapsed_ms for r in self._rounds),
        }

    # -- convenience constructors -------------------------------------------

    @classmethod
    def from_transition_graph(
        cls,
        graph: Any,
        fairness_pairs: Optional[List[Tuple[Set[str], Set[str]]]] = None,
        **kwargs: Any,
    ) -> "RefinementEngine":
        return cls(
            states=graph.states,
            actions=graph.actions,
            transitions=graph.transitions,
            labels=graph.labels,
            fairness_pairs=fairness_pairs,
            **kwargs,
        )

    @classmethod
    def from_coalgebra(
        cls,
        coalgebra: Any,
        **kwargs: Any,
    ) -> "RefinementEngine":
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
            **kwargs,
        )
