"""
Quotient system construction from a bisimulation relation.

Given a bisimulation relation (partition) over a transition system or
F-coalgebra, construct the quotient system where:
  - Each equivalence class becomes a single state.
  - Transitions are projected through the quotient map.
  - Atomic-proposition labeling is inherited from the representative.
  - Fairness constraints are lifted to the quotient.
  - Stutter self-loops are removed.
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
class QuotientStats:
    """Statistics about a quotient construction."""

    original_states: int
    quotient_states: int
    original_transitions: int
    quotient_transitions: int
    stutter_loops_removed: int
    state_compression_ratio: float
    transition_compression_ratio: float


@dataclass
class QuotientVerificationResult:
    """Result of verifying quotient correctness."""

    is_correct: bool
    ap_preserved: bool
    transitions_preserved: bool
    no_spurious_transitions: bool
    fairness_preserved: bool
    violations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# QuotientBuilder
# ---------------------------------------------------------------------------

class QuotientBuilder:
    """Construct a quotient transition system from a bisimulation relation.

    Usage::

        builder = QuotientBuilder(states, actions, transitions, labels, partition)
        q_states, q_transitions, q_labels = builder.build()
    """

    def __init__(
        self,
        states: Set[str],
        actions: Set[str],
        transitions: Dict[str, Dict[str, Set[str]]],
        labels: Dict[str, Set[str]],
        partition: BisimulationRelation,
        fairness_pairs: Optional[List[Tuple[Set[str], Set[str]]]] = None,
        representative_selector: Optional[Callable[[FrozenSet[str]], str]] = None,
    ) -> None:
        self._states = set(states)
        self._actions = set(actions)
        self._transitions = transitions
        self._labels = dict(labels)
        self._partition = partition
        self._fairness_pairs: List[Tuple[Set[str], Set[str]]] = (
            list(fairness_pairs) if fairness_pairs else []
        )
        self._rep_selector = representative_selector or self._default_rep_selector

        # computed fields
        self._quotient_map: Dict[str, str] = {}  # original -> representative
        self._representatives: Set[str] = set()
        self._q_transitions: Dict[str, Dict[str, Set[str]]] = {}
        self._q_labels: Dict[str, Set[str]] = {}
        self._q_initial: Set[str] = set()
        self._q_fairness: List[Tuple[Set[str], Set[str]]] = []
        self._stutter_loops_removed = 0
        self._original_edge_count = 0
        self._quotient_edge_count = 0
        self._built = False

    # -- representative selection -------------------------------------------

    @staticmethod
    def _default_rep_selector(members: FrozenSet[str]) -> str:
        """Select the lexicographically smallest state as representative."""
        return min(members)

    def _select_representatives(self) -> None:
        """Choose one representative per equivalence class."""
        for cls_members in self._partition.classes():
            rep = self._rep_selector(cls_members)
            self._representatives.add(rep)
            for s in cls_members:
                self._quotient_map[s] = rep

    # -- quotient transitions -----------------------------------------------

    def _build_transitions(self, remove_stutter: bool = True) -> None:
        """Project all transitions through the quotient map."""
        raw: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for src, amap in self._transitions.items():
            q_src = self._quotient_map.get(src, src)
            for act, targets in amap.items():
                self._original_edge_count += len(targets)
                for tgt in targets:
                    q_tgt = self._quotient_map.get(tgt, tgt)
                    if remove_stutter and q_src == q_tgt:
                        self._stutter_loops_removed += 1
                        continue
                    raw[q_src][act].add(q_tgt)

        self._q_transitions = {
            src: dict(amap) for src, amap in raw.items()
        }
        for amap in self._q_transitions.values():
            for targets in amap.values():
                self._quotient_edge_count += len(targets)

    # -- quotient labeling --------------------------------------------------

    def _build_labels(self) -> None:
        """Inherit AP labeling from the representative of each class."""
        for rep in self._representatives:
            self._q_labels[rep] = set(self._labels.get(rep, set()))

    # -- quotient fairness --------------------------------------------------

    def _build_fairness(self) -> None:
        """Lift fairness acceptance pairs to the quotient."""
        for b_set, g_set in self._fairness_pairs:
            q_b: Set[str] = set()
            q_g: Set[str] = set()
            for s in b_set:
                rep = self._quotient_map.get(s, s)
                q_b.add(rep)
            for s in g_set:
                rep = self._quotient_map.get(s, s)
                q_g.add(rep)
            self._q_fairness.append((q_b, q_g))

    # -- main build ---------------------------------------------------------

    def build(
        self,
        remove_stutter: bool = True,
        initial_states: Optional[Set[str]] = None,
    ) -> Tuple[Set[str], Dict[str, Dict[str, Set[str]]], Dict[str, Set[str]]]:
        """Build the full quotient system.

        Returns ``(quotient_states, quotient_transitions, quotient_labels)``.
        """
        self._select_representatives()
        self._build_transitions(remove_stutter=remove_stutter)
        self._build_labels()
        self._build_fairness()

        if initial_states is not None:
            self._q_initial = {
                self._quotient_map.get(s, s) for s in initial_states
            }

        self._built = True

        logger.info(
            "Quotient built: %d -> %d states, %d -> %d transitions, "
            "%d stutter loops removed",
            len(self._states),
            len(self._representatives),
            self._original_edge_count,
            self._quotient_edge_count,
            self._stutter_loops_removed,
        )

        return (
            set(self._representatives),
            dict(self._q_transitions),
            dict(self._q_labels),
        )

    # -- accessors ----------------------------------------------------------

    @property
    def quotient_map(self) -> Dict[str, str]:
        """Mapping from original state to its quotient representative."""
        return dict(self._quotient_map)

    @property
    def quotient_states(self) -> Set[str]:
        return set(self._representatives)

    @property
    def quotient_transitions(self) -> Dict[str, Dict[str, Set[str]]]:
        return dict(self._q_transitions)

    @property
    def quotient_labels(self) -> Dict[str, Set[str]]:
        return dict(self._q_labels)

    @property
    def quotient_initial_states(self) -> Set[str]:
        return set(self._q_initial)

    @property
    def quotient_fairness(self) -> List[Tuple[Set[str], Set[str]]]:
        return list(self._q_fairness)

    # -- statistics ---------------------------------------------------------

    def stats(self) -> QuotientStats:
        if not self._built:
            raise RuntimeError("Call build() first")
        n_orig = len(self._states)
        n_quot = len(self._representatives)
        return QuotientStats(
            original_states=n_orig,
            quotient_states=n_quot,
            original_transitions=self._original_edge_count,
            quotient_transitions=self._quotient_edge_count,
            stutter_loops_removed=self._stutter_loops_removed,
            state_compression_ratio=(
                1.0 - n_quot / n_orig if n_orig > 0 else 0.0
            ),
            transition_compression_ratio=(
                1.0 - self._quotient_edge_count / self._original_edge_count
                if self._original_edge_count > 0
                else 0.0
            ),
        )

    # -- verification -------------------------------------------------------

    def verify(self) -> QuotientVerificationResult:
        """Verify correctness of the quotient construction.

        Checks:
          1. AP labeling preserved within each class.
          2. Every original transition has a quotient counterpart.
          3. No spurious transitions introduced.
          4. Fairness constraints preserved.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        violations: List[str] = []
        ap_ok = True
        trans_ok = True
        no_spurious = True
        fair_ok = True

        # 1. AP preservation
        for cls_members in self._partition.classes():
            rep = self._rep_selector(cls_members)
            rep_ap = frozenset(self._labels.get(rep, set()))
            for s in cls_members:
                s_ap = frozenset(self._labels.get(s, set()))
                if s_ap != rep_ap:
                    ap_ok = False
                    violations.append(
                        f"AP mismatch in class of {rep}: "
                        f"{s} has {s_ap} vs representative {rep_ap}"
                    )

        # 2. Transition preservation
        for src, amap in self._transitions.items():
            q_src = self._quotient_map.get(src, src)
            for act, targets in amap.items():
                for tgt in targets:
                    q_tgt = self._quotient_map.get(tgt, tgt)
                    if q_src == q_tgt:
                        continue  # stutter (allowed to be removed)
                    if q_tgt not in self._q_transitions.get(q_src, {}).get(act, set()):
                        trans_ok = False
                        violations.append(
                            f"Transition {src}-[{act}]->{tgt} "
                            f"(maps to {q_src}-[{act}]->{q_tgt}) "
                            f"missing in quotient"
                        )

        # 3. Spurious transitions
        # Build set of expected quotient transitions from original
        expected: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for src, amap in self._transitions.items():
            q_src = self._quotient_map.get(src, src)
            for act, targets in amap.items():
                for tgt in targets:
                    q_tgt = self._quotient_map.get(tgt, tgt)
                    if q_src != q_tgt:
                        expected[q_src][act].add(q_tgt)

        for q_src, amap in self._q_transitions.items():
            for act, targets in amap.items():
                for q_tgt in targets:
                    if q_tgt not in expected.get(q_src, {}).get(act, set()):
                        no_spurious = False
                        violations.append(
                            f"Spurious quotient transition "
                            f"{q_src}-[{act}]->{q_tgt}"
                        )

        # 4. Fairness preservation
        for i, (orig_b, orig_g) in enumerate(self._fairness_pairs):
            q_b_expected = {self._quotient_map.get(s, s) for s in orig_b}
            q_g_expected = {self._quotient_map.get(s, s) for s in orig_g}
            if i < len(self._q_fairness):
                q_b_actual, q_g_actual = self._q_fairness[i]
                if set(q_b_actual) != q_b_expected:
                    fair_ok = False
                    violations.append(
                        f"Fairness pair {i}: B set mismatch in quotient"
                    )
                if set(q_g_actual) != q_g_expected:
                    fair_ok = False
                    violations.append(
                        f"Fairness pair {i}: G set mismatch in quotient"
                    )
            else:
                fair_ok = False
                violations.append(f"Fairness pair {i} missing in quotient")

        is_correct = ap_ok and trans_ok and no_spurious and fair_ok
        return QuotientVerificationResult(
            is_correct=is_correct,
            ap_preserved=ap_ok,
            transitions_preserved=trans_ok,
            no_spurious_transitions=no_spurious,
            fairness_preserved=fair_ok,
            violations=violations,
        )

    # -- TransitionGraph output ---------------------------------------------

    def to_transition_graph(self) -> Any:
        """Convert the quotient to a ``TransitionGraph`` for property checking.

        Requires ``coacert.functor.coalgebra.TransitionGraph`` to be
        importable (avoids hard circular import by lazy import).
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        try:
            from coacert.functor.coalgebra import TransitionGraph
        except ImportError:
            # fall back to a dict representation
            return {
                "states": set(self._representatives),
                "initial_states": set(self._q_initial),
                "actions": set(self._actions),
                "transitions": dict(self._q_transitions),
                "labels": dict(self._q_labels),
            }

        g = TransitionGraph()
        g.states = set(self._representatives)
        g.initial_states = set(self._q_initial)
        g.actions = set(self._actions)
        g.transitions = {
            src: {act: set(tgts) for act, tgts in amap.items()}
            for src, amap in self._q_transitions.items()
        }
        g.labels = {s: set(ap) for s, ap in self._q_labels.items()}
        return g

    # -- FCoalgebra output --------------------------------------------------

    def to_coalgebra(self, name: str = "quotient") -> Any:
        """Convert the quotient to an ``FCoalgebra``.

        Requires ``coacert.functor.coalgebra.FCoalgebra``.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        try:
            from coacert.functor.coalgebra import FCoalgebra
        except ImportError:
            raise ImportError(
                "Cannot create FCoalgebra: coacert.functor.coalgebra not available"
            )

        coalg = FCoalgebra(
            name=name,
            atomic_propositions=set().union(*self._q_labels.values()) if self._q_labels else set(),
            actions=set(self._actions),
        )

        for rep in self._representatives:
            is_init = rep in self._q_initial
            coalg.add_state(
                rep,
                propositions=self._q_labels.get(rep, set()),
                is_initial=is_init,
            )

        for src, amap in self._q_transitions.items():
            for act, targets in amap.items():
                for tgt in targets:
                    coalg.add_transition(src, act, tgt)

        for i, (q_b, q_g) in enumerate(self._q_fairness):
            coalg.add_fairness_constraint(frozenset(q_b), frozenset(q_g))

        return coalg

    # -- convenience constructors -------------------------------------------

    @classmethod
    def from_transition_graph(
        cls,
        graph: Any,
        partition: BisimulationRelation,
        fairness_pairs: Optional[List[Tuple[Set[str], Set[str]]]] = None,
        **kwargs: Any,
    ) -> "QuotientBuilder":
        return cls(
            states=graph.states,
            actions=graph.actions,
            transitions=graph.transitions,
            labels=graph.labels,
            partition=partition,
            fairness_pairs=fairness_pairs,
            **kwargs,
        )

    @classmethod
    def from_coalgebra(
        cls,
        coalgebra: Any,
        partition: BisimulationRelation,
        **kwargs: Any,
    ) -> "QuotientBuilder":
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
            partition=partition,
            fairness_pairs=fairness_pairs,
            **kwargs,
        )
