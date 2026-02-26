"""
Fairness constraint tracking for TLA+ model checking.

Implements weak fairness (WF) and strong fairness (SF) analysis over
transition graphs, including fair SCC computation and fair cycle detection.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from .graph import TransitionGraph, StateNode, TransitionEdge


# ---------------------------------------------------------------------------
# Fairness constraint types
# ---------------------------------------------------------------------------

class FairnessKind(Enum):
    WEAK = auto()    # WF: continuously enabled ⇒ eventually taken
    STRONG = auto()  # SF: repeatedly enabled  ⇒ eventually taken


@dataclass(frozen=True)
class FairnessConstraint:
    """
    A single fairness constraint for an action.

    For WF_vars(Action): in any suffix where Action is continuously enabled,
    Action must eventually be taken.

    For SF_vars(Action): in any suffix where Action is infinitely often
    enabled, Action must be infinitely often taken.
    """

    kind: FairnessKind
    action_label: str
    name: Optional[str] = None

    def __repr__(self) -> str:
        prefix = "WF" if self.kind == FairnessKind.WEAK else "SF"
        label = self.name or self.action_label
        return f"{prefix}({label})"


@dataclass(frozen=True)
class AcceptancePair:
    """
    Streett acceptance pair (B_i, G_i) derived from a fairness constraint.

    B_i = states where the action is enabled (or edges labelled with it).
    G_i = states/edges where the action is taken.

    A fair path must satisfy: visit B_i infinitely often ⇒ visit G_i infinitely often.
    """

    constraint: FairnessConstraint
    enabled_states: FrozenSet[str]    # B_i
    taken_states: FrozenSet[str]      # G_i (states from which the action is taken)

    @property
    def name(self) -> str:
        return repr(self.constraint)

    def is_satisfied_by_scc(self, scc: Set[str]) -> bool:
        """
        An SCC satisfies this pair if:
        - For WF: if all states in SCC are in B_i, then some transition
                  in SCC corresponds to G_i.
        - For SF: if SCC ∩ B_i ≠ ∅, then SCC ∩ G_i ≠ ∅.
        """
        if self.constraint.kind == FairnessKind.WEAK:
            if scc <= self.enabled_states:
                return bool(scc & self.taken_states)
            return True
        else:
            if scc & self.enabled_states:
                return bool(scc & self.taken_states)
            return True


# ---------------------------------------------------------------------------
# Lasso representation
# ---------------------------------------------------------------------------

@dataclass
class Lasso:
    """
    Represents an ultimately periodic path: prefix · loop^ω.

    The prefix is a finite sequence of state hashes from an initial state
    to the loop entry.  The loop is a finite sequence forming the cycle.
    """

    prefix: List[str]
    loop: List[str]

    @property
    def loop_entry(self) -> str:
        return self.loop[0] if self.loop else self.prefix[-1]

    @property
    def total_length(self) -> int:
        return len(self.prefix) + len(self.loop)

    def all_states(self) -> Set[str]:
        return set(self.prefix) | set(self.loop)

    def loop_states(self) -> Set[str]:
        return set(self.loop)

    def is_valid(self, graph: TransitionGraph) -> bool:
        """Check that all consecutive pairs are actual transitions."""
        path = self.prefix + self.loop
        for i in range(len(path) - 1):
            if not graph.get_edges(path[i], path[i + 1]):
                return False
        if self.loop:
            if not graph.get_edges(self.loop[-1], self.loop[0]):
                return False
        return True


# ---------------------------------------------------------------------------
# Fairness tracker
# ---------------------------------------------------------------------------

class FairnessTracker:
    """
    Tracks fairness constraints and determines whether paths / SCCs
    satisfy them.

    Parameters
    ----------
    graph : TransitionGraph
        The transition graph to analyze.
    constraints : list of FairnessConstraint
        The fairness requirements from the specification.
    enabled_predicate : callable, optional
        Given (action_label, state_hash) returns True iff the action is
        enabled in that state.  If not provided, an action is assumed
        enabled if at least one outgoing transition with that label exists.
    """

    def __init__(
        self,
        graph: TransitionGraph,
        constraints: List[FairnessConstraint],
        enabled_predicate: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self._graph = graph
        self._constraints = list(constraints)
        self._enabled_pred = enabled_predicate or self._default_enabled
        self._acceptance_pairs: Optional[List[AcceptancePair]] = None

    @property
    def constraints(self) -> List[FairnessConstraint]:
        return list(self._constraints)

    def add_constraint(self, constraint: FairnessConstraint) -> None:
        self._constraints.append(constraint)
        self._acceptance_pairs = None

    # -- Enabled predicate -------------------------------------------------

    def _default_enabled(self, action: str, state_hash: str) -> bool:
        """Action is enabled if there is an outgoing edge with that label."""
        for _, edge in self._graph.get_successors(state_hash):
            if edge.action_label == action:
                return True
        return False

    def is_action_enabled(self, action: str, state_hash: str) -> bool:
        return self._enabled_pred(action, state_hash)

    def is_action_taken(self, action: str, state_hash: str) -> bool:
        """Action is taken from this state if there is an actual transition."""
        for _, edge in self._graph.get_successors(state_hash):
            if edge.action_label == action and not edge.is_stuttering:
                return True
        return False

    # -- Acceptance pairs --------------------------------------------------

    def compute_acceptance_pairs(self) -> List[AcceptancePair]:
        if self._acceptance_pairs is not None:
            return self._acceptance_pairs

        pairs: List[AcceptancePair] = []
        all_hashes = self._graph.all_state_hashes()

        for constraint in self._constraints:
            action = constraint.action_label
            enabled: Set[str] = set()
            taken: Set[str] = set()

            for h in all_hashes:
                if self.is_action_enabled(action, h):
                    enabled.add(h)
                if self.is_action_taken(action, h):
                    taken.add(h)

            pairs.append(
                AcceptancePair(
                    constraint=constraint,
                    enabled_states=frozenset(enabled),
                    taken_states=frozenset(taken),
                )
            )

        self._acceptance_pairs = pairs
        return pairs

    # -- Fair path checking ------------------------------------------------

    def is_fair_lasso(self, lasso: Lasso) -> bool:
        """Check whether a lasso-shaped path satisfies all fairness constraints."""
        if not lasso.loop:
            return True

        loop_states = lasso.loop_states()
        pairs = self.compute_acceptance_pairs()

        for pair in pairs:
            if pair.constraint.kind == FairnessKind.WEAK:
                continuously_enabled = all(
                    h in pair.enabled_states for h in loop_states
                )
                if continuously_enabled:
                    eventually_taken = any(
                        h in pair.taken_states for h in loop_states
                    )
                    if not eventually_taken:
                        return False
            else:
                repeatedly_enabled = any(
                    h in pair.enabled_states for h in loop_states
                )
                if repeatedly_enabled:
                    eventually_taken = any(
                        h in pair.taken_states for h in loop_states
                    )
                    if not eventually_taken:
                        return False
        return True

    # -- Fair SCC computation ----------------------------------------------

    def fair_sccs(self) -> List[Set[str]]:
        """
        Compute all non-trivial SCCs that satisfy every fairness constraint.

        An SCC is fair if for every acceptance pair (B_i, G_i):
          - WF: if all SCC states ∈ B_i, then SCC ∩ G_i ≠ ∅
          - SF: if SCC ∩ B_i ≠ ∅, then SCC ∩ G_i ≠ ∅
        """
        nontrivial = self._graph.nontrivial_sccs()
        pairs = self.compute_acceptance_pairs()
        result: List[Set[str]] = []

        for scc in nontrivial:
            if all(pair.is_satisfied_by_scc(scc) for pair in pairs):
                result.append(scc)
        return result

    def unfair_sccs(self) -> List[Set[str]]:
        """SCCs that violate at least one fairness constraint."""
        nontrivial = self._graph.nontrivial_sccs()
        pairs = self.compute_acceptance_pairs()
        result: List[Set[str]] = []
        for scc in nontrivial:
            if not all(pair.is_satisfied_by_scc(scc) for pair in pairs):
                result.append(scc)
        return result

    def violated_constraints_in_scc(
        self, scc: Set[str]
    ) -> List[FairnessConstraint]:
        """Return constraints violated by this SCC."""
        pairs = self.compute_acceptance_pairs()
        violated: List[FairnessConstraint] = []
        for pair in pairs:
            if not pair.is_satisfied_by_scc(scc):
                violated.append(pair.constraint)
        return violated

    # -- Fair cycle detection ----------------------------------------------

    def find_fair_cycle(self) -> Optional[Lasso]:
        """
        Find a reachable fair cycle (lasso) if one exists.

        Strategy: for each fair SCC, find a cycle within it and a prefix
        from an initial state to that cycle.
        """
        fair = self.fair_sccs()
        if not fair:
            return None

        for scc in fair:
            lasso = self._extract_lasso_from_scc(scc)
            if lasso is not None:
                return lasso
        return None

    def find_all_fair_cycles(self, limit: int = 100) -> List[Lasso]:
        """Find up to *limit* distinct fair lassos."""
        fair = self.fair_sccs()
        lassos: List[Lasso] = []
        for scc in fair:
            lasso = self._extract_lasso_from_scc(scc)
            if lasso is not None:
                lassos.append(lasso)
                if len(lassos) >= limit:
                    break
        return lassos

    def _extract_lasso_from_scc(self, scc: Set[str]) -> Optional[Lasso]:
        """Build a lasso: initial → scc entry → cycle within scc."""
        if not scc:
            return None

        entry = self._find_scc_entry(scc)
        if entry is None:
            return None

        prefix = self._find_prefix_to(entry)
        if prefix is None:
            prefix = [entry]

        loop = self._find_cycle_in_scc(scc, entry)
        if loop is None:
            return None

        return Lasso(prefix=prefix, loop=loop)

    def _find_scc_entry(self, scc: Set[str]) -> Optional[str]:
        """Find a state in the SCC reachable from an initial state."""
        initials = self._graph.initial_states
        if not initials:
            return next(iter(scc)) if scc else None

        for init_h in initials:
            reachable = self._graph.bfs_reachable(init_h)
            for h in scc:
                if h in reachable:
                    return h
        return next(iter(scc))

    def _find_prefix_to(self, target: str) -> Optional[List[str]]:
        """Find shortest path from any initial state to target."""
        for init_h in self._graph.initial_states:
            path = self._graph.shortest_path(init_h, target)
            if path is not None:
                return path
        return None

    def _find_cycle_in_scc(
        self, scc: Set[str], start: str
    ) -> Optional[List[str]]:
        """Find a cycle within the SCC starting and ending at *start*."""
        if start not in scc:
            return None

        visited: Set[str] = set()
        parent: Dict[str, Optional[str]] = {start: None}
        queue: deque[str] = deque()

        for succ_h in self._graph.get_successor_hashes(start):
            if succ_h in scc:
                if succ_h == start:
                    return [start]
                parent[succ_h] = start
                queue.append(succ_h)
                visited.add(succ_h)

        while queue:
            current = queue.popleft()
            for succ_h in self._graph.get_successor_hashes(current):
                if succ_h not in scc:
                    continue
                if succ_h == start:
                    path = [start]
                    node = current
                    while node is not None and node != start:
                        path.append(node)
                        node = parent.get(node)
                    path.reverse()
                    return path
                if succ_h not in visited:
                    visited.add(succ_h)
                    parent[succ_h] = current
                    queue.append(succ_h)
        return None

    # -- Integration helpers -----------------------------------------------

    def classify_actions(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        For each fairness-constrained action, classify states into
        'enabled', 'taken', and 'enabled_not_taken'.
        """
        result: Dict[str, Dict[str, Set[str]]] = {}
        for pair in self.compute_acceptance_pairs():
            action = pair.constraint.action_label
            enabled_not_taken = pair.enabled_states - pair.taken_states
            result[action] = {
                "enabled": set(pair.enabled_states),
                "taken": set(pair.taken_states),
                "enabled_not_taken": set(enabled_not_taken),
            }
        return result

    def summary(self) -> Dict[str, Any]:
        pairs = self.compute_acceptance_pairs()
        fair = self.fair_sccs()
        unfair = self.unfair_sccs()
        return {
            "num_constraints": len(self._constraints),
            "weak_fairness": sum(
                1 for c in self._constraints if c.kind == FairnessKind.WEAK
            ),
            "strong_fairness": sum(
                1 for c in self._constraints if c.kind == FairnessKind.STRONG
            ),
            "num_fair_sccs": len(fair),
            "num_unfair_sccs": len(unfair),
            "acceptance_pairs": [
                {
                    "action": p.constraint.action_label,
                    "kind": p.constraint.kind.name,
                    "enabled_count": len(p.enabled_states),
                    "taken_count": len(p.taken_states),
                }
                for p in pairs
            ],
        }
