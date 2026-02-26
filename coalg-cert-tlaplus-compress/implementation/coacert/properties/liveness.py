"""
Liveness property verification under fairness for CoaCert-TLA.

Implements SCC-based liveness checking with Streett acceptance,
fair-cycle detection, and counterexample generation (lasso-shaped
traces).  Supports weak fairness (WF), strong fairness (SF),
□◇φ (infinitely often), ◇□φ (eventually always), and φ ⤳ ψ (leads-to).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from .ctl_star import CTLLabeler, KripkeAdapter
from .temporal_logic import (
    And,
    Atomic,
    ExistsPath,
    FalseFormula,
    Finally,
    ForallPath,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    TemporalFormula,
    TrueFormula,
    Until,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Fairness specifications
# ============================================================================

class FairnessKind(Enum):
    """Type of fairness constraint."""
    WEAK = auto()    # WF_v(A): continuously enabled → eventually taken
    STRONG = auto()  # SF_v(A): repeatedly enabled → eventually taken


@dataclass(frozen=True)
class FairnessSpec:
    """A single fairness constraint.

    enabled_states: states where the action is enabled
    taken_states:   states where the action has just been taken
                    (equivalently, states reached by taking the action)
    """
    kind: FairnessKind
    name: str
    enabled_states: FrozenSet[str]
    taken_states: FrozenSet[str]

    def is_satisfied_by_scc(self, scc: FrozenSet[str]) -> bool:
        """Check if an SCC satisfies this fairness constraint.

        WF: if the SCC intersects enabled_states permanently (i.e., the
            SCC is a subset of enabled_states), it must intersect
            taken_states.
        SF: if the SCC intersects enabled_states at all, it must
            intersect taken_states.
        """
        if self.kind == FairnessKind.WEAK:
            permanently_enabled = scc <= self.enabled_states
            if permanently_enabled:
                return bool(scc & self.taken_states)
            return True
        else:
            # Strong fairness: if repeatedly enabled, must be taken
            if scc & self.enabled_states:
                return bool(scc & self.taken_states)
            return True


# ============================================================================
# Liveness property types
# ============================================================================

class LivenessKind(Enum):
    """Classification of liveness properties."""
    INFINITELY_OFTEN = auto()   # □◇φ
    EVENTUALLY_ALWAYS = auto()  # ◇□φ
    LEADS_TO = auto()           # φ ⤳ ψ  (= □(φ → ◇ψ))
    RESPONSE = auto()           # □(φ → ◇ψ) under fairness
    PERSISTENCE = auto()        # ◇□φ


@dataclass(frozen=True)
class LivenessProperty:
    """A liveness property to be verified."""
    name: str
    kind: LivenessKind
    phi: TemporalFormula
    psi: Optional[TemporalFormula] = None  # used for leads-to / response
    description: str = ""


# ============================================================================
# Liveness check results
# ============================================================================

@dataclass
class LivenessCheckResult:
    """Result of a liveness property check."""
    property_name: str
    holds: bool
    method: str = "scc"
    counterexample_stem: Optional[List[str]] = None
    counterexample_cycle: Optional[List[str]] = None
    accepting_sccs_count: int = 0
    total_sccs_count: int = 0

    @property
    def counterexample(self) -> Optional[Tuple[List[str], List[str]]]:
        if self.counterexample_stem and self.counterexample_cycle:
            return (self.counterexample_stem, self.counterexample_cycle)
        return None

    def summary(self) -> str:
        status = "HOLDS" if self.holds else "VIOLATED"
        msg = f"[{status}] {self.property_name} (method={self.method}, "
        msg += f"accepting SCCs: {self.accepting_sccs_count}/{self.total_sccs_count})"
        if not self.holds and self.counterexample_stem:
            stem_str = " → ".join(self.counterexample_stem)
            cycle_str = " → ".join(self.counterexample_cycle or [])
            msg += f"\n  lasso: ({stem_str}) ({cycle_str})ω"
        return msg


# ============================================================================
# Liveness checker
# ============================================================================

class LivenessChecker:
    """Liveness property verifier using SCC-based analysis.

    Integrates with the coalgebraic quotient system and respects
    fairness constraints (weak and strong).
    """

    def __init__(
        self,
        coalgebra: object,
        fairness: Optional[List[FairnessSpec]] = None,
    ) -> None:
        self._coalg = coalgebra
        self._kripke = KripkeAdapter.from_coalgebra(coalgebra)
        self._labeler = CTLLabeler(self._kripke)
        self._fairness = fairness or []
        self._sccs: Optional[List[FrozenSet[str]]] = None

    @property
    def kripke(self) -> KripkeAdapter:
        return self._kripke

    def _compute_sccs(self) -> List[FrozenSet[str]]:
        """Compute and cache all SCCs."""
        if self._sccs is None:
            self._sccs = self._tarjan_scc(
                self._kripke.states,
                {s: self._kripke.successors(s) for s in self._kripke.states},
            )
        return self._sccs

    # -- Top-level checking -------------------------------------------------

    def check(self, prop: LivenessProperty) -> LivenessCheckResult:
        """Check a liveness property under the configured fairness constraints."""
        logger.info("Checking liveness property: %s", prop.name)

        if prop.kind == LivenessKind.INFINITELY_OFTEN:
            return self._check_infinitely_often(prop)
        elif prop.kind == LivenessKind.EVENTUALLY_ALWAYS:
            return self._check_eventually_always(prop)
        elif prop.kind in (LivenessKind.LEADS_TO, LivenessKind.RESPONSE):
            return self._check_leads_to(prop)
        elif prop.kind == LivenessKind.PERSISTENCE:
            return self._check_eventually_always(prop)
        else:
            logger.error("Unknown liveness kind: %s", prop.kind)
            return LivenessCheckResult(
                property_name=prop.name,
                holds=False,
                method="unknown",
            )

    # -- □◇φ (infinitely often) --------------------------------------------

    def _check_infinitely_often(self, prop: LivenessProperty) -> LivenessCheckResult:
        """Check □◇φ: on every fair path, φ holds infinitely often.

        Negation: ◇□¬φ on some fair path.
        Find a reachable fair SCC that does not intersect φ-states.
        If no such SCC exists, the property holds.
        """
        phi_states = self._labeler.label(prop.phi)
        sccs = self._compute_sccs()
        fair_sccs = self._filter_fair_sccs(sccs)
        non_trivial = self._non_trivial_sccs(fair_sccs)

        # A fair SCC that avoids φ-states is a counterexample to □◇φ
        violating_sccs = [
            scc for scc in non_trivial
            if not (scc & phi_states)
        ]

        if not violating_sccs:
            return LivenessCheckResult(
                property_name=prop.name,
                holds=True,
                method="scc_inf_often",
                accepting_sccs_count=len(non_trivial),
                total_sccs_count=len(sccs),
            )

        # Find reachable violating SCC
        stem, cycle = self._find_lasso_to_scc(violating_sccs)

        return LivenessCheckResult(
            property_name=prop.name,
            holds=False,
            method="scc_inf_often",
            counterexample_stem=stem,
            counterexample_cycle=cycle,
            accepting_sccs_count=len(non_trivial),
            total_sccs_count=len(sccs),
        )

    # -- ◇□φ (eventually always) -------------------------------------------

    def _check_eventually_always(self, prop: LivenessProperty) -> LivenessCheckResult:
        """Check ◇□φ: there exists a fair path on which φ holds from some point.

        Verified as: there exists a reachable fair SCC contained entirely
        within φ-states.
        """
        phi_states = self._labeler.label(prop.phi)
        sccs = self._compute_sccs()
        fair_sccs = self._filter_fair_sccs(sccs)
        non_trivial = self._non_trivial_sccs(fair_sccs)

        # Find a fair SCC entirely within phi_states
        good_sccs = [scc for scc in non_trivial if scc <= phi_states]

        if good_sccs:
            # Check reachability from initial states
            for scc in good_sccs:
                for init in self._kripke.initial_states:
                    path = self._bfs_to_set(init, scc)
                    if path is not None:
                        cycle = self._find_cycle_in_scc(scc)
                        return LivenessCheckResult(
                            property_name=prop.name,
                            holds=True,
                            method="scc_eventually_always",
                            counterexample_stem=path,
                            counterexample_cycle=cycle,
                            accepting_sccs_count=len(good_sccs),
                            total_sccs_count=len(sccs),
                        )

        return LivenessCheckResult(
            property_name=prop.name,
            holds=False,
            method="scc_eventually_always",
            accepting_sccs_count=len(good_sccs) if good_sccs else 0,
            total_sccs_count=len(sccs),
        )

    # -- φ ⤳ ψ (leads-to / response) ---------------------------------------

    def _check_leads_to(self, prop: LivenessProperty) -> LivenessCheckResult:
        """Check φ ⤳ ψ ≡ □(φ → ◇ψ): whenever φ holds, ψ eventually holds.

        Negation: ◇(φ ∧ □¬ψ) on some fair path.
        Find a reachable state satisfying φ from which there is a fair
        cycle avoiding ψ-states entirely.
        """
        assert prop.psi is not None, "leads-to requires psi"
        phi_states = self._labeler.label(prop.phi)
        psi_states = self._labeler.label(prop.psi)
        not_psi = self._kripke.states - psi_states
        sccs = self._compute_sccs()

        # SCCs within ¬ψ states that are fair
        neg_psi_sccs = [scc for scc in sccs if scc <= not_psi]
        fair_neg_psi = self._filter_fair_sccs(neg_psi_sccs)
        non_trivial = self._non_trivial_sccs(fair_neg_psi)

        # Check if any fair ¬ψ SCC is reachable from a φ-state
        for scc in non_trivial:
            scc_reach = self._backward_reachable_restricted(scc, not_psi)
            reachable_phi = scc_reach & phi_states
            if not reachable_phi:
                continue

            # Check if any such φ-state is reachable from initial states
            for init in self._kripke.initial_states:
                for target in reachable_phi:
                    path = self._bfs_path(init, target)
                    if path is not None:
                        # Build lasso: stem to φ-state, then to SCC, then cycle
                        bridge = self._bfs_restricted_path(target, scc, not_psi)
                        if bridge is None:
                            bridge = [target]
                        cycle = self._find_cycle_in_scc(scc)
                        stem = path + bridge[1:]
                        return LivenessCheckResult(
                            property_name=prop.name,
                            holds=False,
                            method="scc_leads_to",
                            counterexample_stem=stem,
                            counterexample_cycle=cycle,
                            accepting_sccs_count=len(non_trivial),
                            total_sccs_count=len(sccs),
                        )

        return LivenessCheckResult(
            property_name=prop.name,
            holds=True,
            method="scc_leads_to",
            accepting_sccs_count=0,
            total_sccs_count=len(sccs),
        )

    # -- Fair SCC filtering -------------------------------------------------

    def _filter_fair_sccs(self, sccs: List[FrozenSet[str]]) -> List[FrozenSet[str]]:
        """Keep only SCCs satisfying all fairness constraints."""
        if not self._fairness:
            return sccs
        result: List[FrozenSet[str]] = []
        for scc in sccs:
            if all(f.is_satisfied_by_scc(scc) for f in self._fairness):
                result.append(scc)
        return result

    def _non_trivial_sccs(self, sccs: List[FrozenSet[str]]) -> List[FrozenSet[str]]:
        """Filter to non-trivial SCCs (size > 1 or has self-loop)."""
        result: List[FrozenSet[str]] = []
        for scc in sccs:
            if len(scc) > 1:
                result.append(scc)
            elif len(scc) == 1:
                s = next(iter(scc))
                if s in self._kripke.successors(s):
                    result.append(scc)
        return result

    # -- Streett acceptance -------------------------------------------------

    def check_streett_acceptance(
        self,
        acceptance_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
    ) -> LivenessCheckResult:
        """Generalized Streett acceptance check.

        An SCC satisfies Streett acceptance if for every pair (B_i, G_i):
          if the SCC intersects B_i, it also intersects G_i.

        This corresponds to the condition that a run visiting B_i
        infinitely often also visits G_i infinitely often.
        """
        sccs = self._compute_sccs()
        non_trivial = self._non_trivial_sccs(sccs)

        accepting: List[FrozenSet[str]] = []
        for scc in non_trivial:
            is_acc = True
            for b_states, g_states in acceptance_pairs:
                if (scc & b_states) and not (scc & g_states):
                    is_acc = False
                    break
            if is_acc:
                accepting.append(scc)

        # Check if all initial states can reach an accepting SCC
        all_reachable = True
        violating_init: Optional[str] = None
        for init in self._kripke.initial_states:
            can_reach = False
            for scc in accepting:
                if self._bfs_to_set(init, scc) is not None:
                    can_reach = True
                    break
            if not can_reach:
                all_reachable = False
                violating_init = init
                break

        if all_reachable:
            return LivenessCheckResult(
                property_name="Streett acceptance",
                holds=True,
                method="streett",
                accepting_sccs_count=len(accepting),
                total_sccs_count=len(sccs),
            )

        # Generate counterexample from violating initial state
        stem: Optional[List[str]] = None
        cycle: Optional[List[str]] = None
        if violating_init:
            # Find any reachable non-accepting SCC
            for scc in non_trivial:
                if scc not in accepting:
                    path = self._bfs_to_set(violating_init, scc)
                    if path:
                        stem = path
                        cycle = self._find_cycle_in_scc(scc)
                        break

        return LivenessCheckResult(
            property_name="Streett acceptance",
            holds=False,
            method="streett",
            counterexample_stem=stem,
            counterexample_cycle=cycle,
            accepting_sccs_count=len(accepting),
            total_sccs_count=len(sccs),
        )

    # -- Fairness-aware cycle detection -------------------------------------

    def find_fair_cycles(self) -> List[Tuple[List[str], FrozenSet[str]]]:
        """Find all fair cycles: cycles satisfying all fairness constraints.

        Returns list of (cycle_states, scc) pairs.
        """
        sccs = self._compute_sccs()
        fair = self._filter_fair_sccs(sccs)
        non_trivial = self._non_trivial_sccs(fair)
        result: List[Tuple[List[str], FrozenSet[str]]] = []
        for scc in non_trivial:
            cycle = self._find_cycle_in_scc(scc)
            if cycle:
                result.append((cycle, scc))
        return result

    def find_unfair_cycles(self) -> List[Tuple[FrozenSet[str], List[FairnessSpec]]]:
        """Find non-trivial SCCs that violate at least one fairness constraint.

        Returns (scc, violated_specs) pairs.
        """
        sccs = self._compute_sccs()
        non_trivial = self._non_trivial_sccs(sccs)
        result: List[Tuple[FrozenSet[str], List[FairnessSpec]]] = []
        for scc in non_trivial:
            violated = [f for f in self._fairness if not f.is_satisfied_by_scc(scc)]
            if violated:
                result.append((scc, violated))
        return result

    # -- Quotient integration -----------------------------------------------

    def check_on_quotient(
        self,
        prop: LivenessProperty,
        quotient_coalg: object,
        quotient_fairness: Optional[List[FairnessSpec]] = None,
    ) -> Tuple[LivenessCheckResult, LivenessCheckResult]:
        """Check liveness on both original and quotient systems.

        The quotient must preserve fairness (T-Fair coherence).
        """
        orig = self.check(prop)
        q_checker = LivenessChecker(quotient_coalg, quotient_fairness)
        q_result = q_checker.check(prop)
        return orig, q_result

    # -- Batch checking -----------------------------------------------------

    def check_all(self, properties: List[LivenessProperty]) -> List[LivenessCheckResult]:
        """Check multiple liveness properties."""
        return [self.check(p) for p in properties]

    # -- Utility methods ----------------------------------------------------

    def _find_lasso_to_scc(
        self,
        target_sccs: List[FrozenSet[str]],
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Find a lasso (stem, cycle) reaching one of the target SCCs."""
        for init in sorted(self._kripke.initial_states):
            for scc in target_sccs:
                path = self._bfs_to_set(init, scc)
                if path is not None:
                    cycle = self._find_cycle_in_scc(scc)
                    return path, cycle
        return None, None

    def _bfs_to_set(
        self,
        source: str,
        targets: FrozenSet[str],
    ) -> Optional[List[str]]:
        """BFS shortest path from *source* to any state in *targets*."""
        if source in targets:
            return [source]
        visited: Set[str] = {source}
        parent: Dict[str, str] = {}
        queue: deque[str] = deque([source])
        while queue:
            s = queue.popleft()
            for t in self._kripke.successors(s):
                if t in visited:
                    continue
                visited.add(t)
                parent[t] = s
                if t in targets:
                    path = [t]
                    cur = t
                    while cur in parent:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                queue.append(t)
        return None

    def _bfs_path(self, source: str, target: str) -> Optional[List[str]]:
        """BFS shortest path between two states."""
        return self._bfs_to_set(source, frozenset({target}))

    def _bfs_restricted_path(
        self,
        source: str,
        targets: FrozenSet[str],
        allowed: FrozenSet[str],
    ) -> Optional[List[str]]:
        """BFS path staying within *allowed* states."""
        if source in targets:
            return [source]
        if source not in allowed:
            return None
        visited: Set[str] = {source}
        parent: Dict[str, str] = {}
        queue: deque[str] = deque([source])
        while queue:
            s = queue.popleft()
            for t in self._kripke.successors(s):
                if t not in allowed or t in visited:
                    continue
                visited.add(t)
                parent[t] = s
                if t in targets:
                    path = [t]
                    cur = t
                    while cur in parent:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                queue.append(t)
        return None

    def _find_cycle_in_scc(self, scc: FrozenSet[str]) -> Optional[List[str]]:
        """Find a cycle within an SCC using DFS."""
        if not scc:
            return None
        start = min(scc)
        # DFS from start, find back-edge to start
        visited: Set[str] = set()
        parent: Dict[str, str] = {}

        stack: List[Tuple[str, bool]] = [(start, False)]
        while stack:
            s, processed = stack.pop()
            if processed:
                continue
            if s in visited:
                if s == start and parent:
                    # Found cycle back to start
                    cycle = [s]
                    cur = parent.get(cycle[-1])
                    while cur is not None and cur != start:
                        cycle.append(cur)
                        cur = parent.get(cur)
                    cycle.append(start)
                    cycle.reverse()
                    return cycle
                continue
            visited.add(s)
            for t in self._kripke.successors(s):
                if t in scc:
                    if t == start and s != start:
                        parent[t] = s
                        cycle = [start]
                        cur: Optional[str] = s
                        while cur is not None and cur != start:
                            cycle.append(cur)
                            cur = parent.get(cur)
                        cycle.append(start)
                        cycle.reverse()
                        return cycle
                    if t not in visited:
                        parent[t] = s
                        stack.append((t, False))

        # Fallback for self-loop
        if start in self._kripke.successors(start):
            return [start, start]
        return None

    def _backward_reachable_restricted(
        self,
        targets: FrozenSet[str],
        allowed: FrozenSet[str],
    ) -> FrozenSet[str]:
        """Backward reachability from *targets* restricted to *allowed* states."""
        rev = self._kripke.reverse_graph()
        visited: Set[str] = set(targets)
        queue: deque[str] = deque(targets)
        while queue:
            s = queue.popleft()
            for pred in rev.get(s, set()):
                if pred in allowed and pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
        return frozenset(visited)

    @staticmethod
    def _tarjan_scc(
        nodes: FrozenSet[str],
        adj: Dict[str, Set[str]],
    ) -> List[FrozenSet[str]]:
        """Tarjan's SCC algorithm."""
        index_counter = [0]
        stack: List[str] = []
        on_stack: Set[str] = set()
        index_map: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        sccs: List[FrozenSet[str]] = []

        def strongconnect(v: str) -> None:
            index_map[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in adj.get(v, set()):
                if w not in index_map:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index_map[w])

            if lowlink[v] == index_map[v]:
                component: Set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.add(w)
                    if w == v:
                        break
                sccs.append(frozenset(component))

        for v in sorted(nodes):
            if v not in index_map:
                strongconnect(v)

        return sccs


# ============================================================================
# Convenience constructors for liveness properties
# ============================================================================

def make_infinitely_often(name: str, proposition: str) -> LivenessProperty:
    """□◇ proposition: proposition holds infinitely often."""
    return LivenessProperty(
        name=name,
        kind=LivenessKind.INFINITELY_OFTEN,
        phi=Atomic(proposition),
        description=f"□◇{proposition}",
    )


def make_eventually_always(name: str, proposition: str) -> LivenessProperty:
    """◇□ proposition: proposition holds from some point onward."""
    return LivenessProperty(
        name=name,
        kind=LivenessKind.EVENTUALLY_ALWAYS,
        phi=Atomic(proposition),
        description=f"◇□{proposition}",
    )


def make_leads_to(name: str, trigger: str, response: str) -> LivenessProperty:
    """trigger ⤳ response: every trigger eventually leads to response."""
    return LivenessProperty(
        name=name,
        kind=LivenessKind.LEADS_TO,
        phi=Atomic(trigger),
        psi=Atomic(response),
        description=f"{trigger} ⤳ {response}",
    )


def make_weak_fairness(
    name: str,
    enabled_states: FrozenSet[str],
    taken_states: FrozenSet[str],
) -> FairnessSpec:
    """Create a weak fairness constraint WF_v(A)."""
    return FairnessSpec(
        kind=FairnessKind.WEAK,
        name=name,
        enabled_states=enabled_states,
        taken_states=taken_states,
    )


def make_strong_fairness(
    name: str,
    enabled_states: FrozenSet[str],
    taken_states: FrozenSet[str],
) -> FairnessSpec:
    """Create a strong fairness constraint SF_v(A)."""
    return FairnessSpec(
        kind=FairnessKind.STRONG,
        name=name,
        enabled_states=enabled_states,
        taken_states=taken_states,
    )
