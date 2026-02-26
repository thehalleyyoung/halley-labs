"""
Safety property verification for CoaCert-TLA.

Provides BFS-based invariant checking on the coalgebraic Kripke structure,
inductive invariant verification, k-induction, and differential checking
(same safety result on original vs quotient).
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

from .ctl_star import KripkeAdapter
from .temporal_logic import (
    Atomic,
    And,
    FalseFormula,
    Implies,
    Not,
    Or,
    TemporalFormula,
    TrueFormula,
    is_stuttering_invariant,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Safety property types
# ============================================================================

class SafetyKind(Enum):
    """Classification of safety properties."""
    STATE_INVARIANT = auto()    # predicate over states
    ACTION_CONSTRAINT = auto()  # predicate over transitions
    TYPE_INVARIANT = auto()     # type-level constraint on state variables


@dataclass(frozen=True)
class SafetyProperty:
    """A safety property to be verified.

    A safety property asserts that something bad never happens,
    i.e., every reachable state satisfies the predicate.
    """
    name: str
    kind: SafetyKind
    predicate: Callable[[str, FrozenSet[str]], bool]
    description: str = ""

    def evaluate(self, state: str, labels: FrozenSet[str]) -> bool:
        """Evaluate the predicate at a given state."""
        return self.predicate(state, labels)


@dataclass(frozen=True)
class ActionSafetyProperty:
    """Safety property constraining transitions.

    Asserts that every transition (s, a, t) satisfies the constraint.
    """
    name: str
    constraint: Callable[[str, str, str, FrozenSet[str], FrozenSet[str]], bool]
    description: str = ""

    def evaluate(
        self,
        src: str,
        action: str,
        dst: str,
        src_labels: FrozenSet[str],
        dst_labels: FrozenSet[str],
    ) -> bool:
        return self.constraint(src, action, dst, src_labels, dst_labels)


# ============================================================================
# Safety check results
# ============================================================================

@dataclass
class SafetyCheckResult:
    """Result of a safety property check."""
    property_name: str
    holds: bool
    violating_state: Optional[str] = None
    counterexample_trace: Optional[List[str]] = None
    states_checked: int = 0
    depth_reached: int = 0
    method: str = "bfs"

    def summary(self) -> str:
        status = "SAFE" if self.holds else "UNSAFE"
        msg = f"[{status}] {self.property_name} ({self.states_checked} states checked, depth {self.depth_reached})"
        if not self.holds and self.counterexample_trace:
            trace = " → ".join(self.counterexample_trace)
            msg += f"\n  counterexample: {trace}"
        return msg


@dataclass
class InductiveCheckResult:
    """Result of an inductive invariant check."""
    property_name: str
    is_inductive: bool
    base_holds: bool
    step_holds: bool
    step_counterexample: Optional[Tuple[str, str]] = None  # (state, successor)
    k: int = 1

    def summary(self) -> str:
        status = "INDUCTIVE" if self.is_inductive else "NOT INDUCTIVE"
        msg = f"[{status}] {self.property_name} (k={self.k})"
        if not self.base_holds:
            msg += "\n  base case failed"
        if not self.step_holds and self.step_counterexample:
            s, t = self.step_counterexample
            msg += f"\n  inductive step failed: {s} → {t}"
        return msg


@dataclass
class StrengtheningCandidate:
    """A suggested strengthened invariant."""
    original: str
    strengthened_predicate: Callable[[str, FrozenSet[str]], bool]
    description: str
    is_inductive: bool = False


# ============================================================================
# Safety checker
# ============================================================================

class SafetyChecker:
    """Safety property verifier operating on coalgebraic Kripke structures.

    Supports BFS-based reachability analysis, inductive invariant checking,
    k-induction, and differential checking against quotient systems.
    """

    def __init__(self, coalgebra: object) -> None:
        self._coalg = coalgebra
        self._kripke = KripkeAdapter.from_coalgebra(coalgebra)

    @property
    def kripke(self) -> KripkeAdapter:
        return self._kripke

    # -- BFS invariant checking ---------------------------------------------

    def check_invariant(
        self,
        prop: SafetyProperty,
        *,
        max_depth: Optional[int] = None,
    ) -> SafetyCheckResult:
        """Check that *prop* holds in every reachable state via BFS.

        If the property is violated, returns the shortest path from an
        initial state to the first violating state.
        """
        logger.info("Checking safety property: %s", prop.name)

        visited: Set[str] = set()
        parent: Dict[str, Optional[str]] = {}
        depth_map: Dict[str, int] = {}
        queue: deque[str] = deque()

        for init in self._kripke.initial_states:
            visited.add(init)
            parent[init] = None
            depth_map[init] = 0
            queue.append(init)

            labels = self._kripke.labels.get(init, frozenset())
            if not prop.evaluate(init, labels):
                return SafetyCheckResult(
                    property_name=prop.name,
                    holds=False,
                    violating_state=init,
                    counterexample_trace=[init],
                    states_checked=1,
                    depth_reached=0,
                )

        max_depth_reached = 0

        while queue:
            state = queue.popleft()
            d = depth_map[state]
            max_depth_reached = max(max_depth_reached, d)

            if max_depth is not None and d >= max_depth:
                continue

            for succ in self._kripke.successors(state):
                if succ in visited:
                    continue
                visited.add(succ)
                parent[succ] = state
                depth_map[succ] = d + 1
                queue.append(succ)

                labels = self._kripke.labels.get(succ, frozenset())
                if not prop.evaluate(succ, labels):
                    trace = self._reconstruct_path(succ, parent)
                    return SafetyCheckResult(
                        property_name=prop.name,
                        holds=False,
                        violating_state=succ,
                        counterexample_trace=trace,
                        states_checked=len(visited),
                        depth_reached=depth_map[succ],
                    )

        return SafetyCheckResult(
            property_name=prop.name,
            holds=True,
            states_checked=len(visited),
            depth_reached=max_depth_reached,
        )

    def check_action_constraint(
        self,
        prop: ActionSafetyProperty,
    ) -> SafetyCheckResult:
        """Check that every reachable transition satisfies *prop*."""
        logger.info("Checking action constraint: %s", prop.name)

        visited: Set[str] = set()
        parent: Dict[str, Optional[str]] = {}
        queue: deque[str] = deque()

        for init in self._kripke.initial_states:
            visited.add(init)
            parent[init] = None
            queue.append(init)

        states_checked = 0

        while queue:
            state = queue.popleft()
            states_checked += 1
            src_labels = self._kripke.labels.get(state, frozenset())

            for act, act_map in self._kripke.transitions.items():
                for dst in act_map.get(state, set()):
                    dst_labels = self._kripke.labels.get(dst, frozenset())
                    if not prop.evaluate(state, act, dst, src_labels, dst_labels):
                        trace = self._reconstruct_path(state, parent) + [dst]
                        return SafetyCheckResult(
                            property_name=prop.name,
                            holds=False,
                            violating_state=state,
                            counterexample_trace=trace,
                            states_checked=states_checked,
                            depth_reached=len(trace) - 1,
                            method="action_bfs",
                        )
                    if dst not in visited:
                        visited.add(dst)
                        parent[dst] = state
                        queue.append(dst)

        return SafetyCheckResult(
            property_name=prop.name,
            holds=True,
            states_checked=states_checked,
            depth_reached=0,
            method="action_bfs",
        )

    def check_ap_invariant(self, proposition: str) -> SafetyCheckResult:
        """Check AG(proposition): the atomic proposition holds everywhere."""
        prop = SafetyProperty(
            name=f"AG({proposition})",
            kind=SafetyKind.STATE_INVARIANT,
            predicate=lambda s, lbl: proposition in lbl,
            description=f"Invariant: {proposition} holds in all reachable states",
        )
        return self.check_invariant(prop)

    def check_predicate_invariant(
        self,
        name: str,
        predicate: Callable[[str, FrozenSet[str]], bool],
    ) -> SafetyCheckResult:
        """Check that *predicate* holds in all reachable states."""
        prop = SafetyProperty(
            name=name,
            kind=SafetyKind.STATE_INVARIANT,
            predicate=predicate,
        )
        return self.check_invariant(prop)

    # -- Inductive invariant checking ---------------------------------------

    def check_inductive(
        self,
        prop: SafetyProperty,
    ) -> InductiveCheckResult:
        """Check if *prop* is an inductive invariant.

        An inductive invariant satisfies:
          1. Init ⊆ Inv  (base case)
          2. For all (s, t) ∈ R: Inv(s) → Inv(t)  (inductive step)
        """
        logger.info("Checking inductive invariant: %s", prop.name)

        # Base case: all initial states satisfy Inv
        base_holds = True
        for init in self._kripke.initial_states:
            labels = self._kripke.labels.get(init, frozenset())
            if not prop.evaluate(init, labels):
                base_holds = False
                break

        # Inductive step: for every state satisfying Inv, all successors
        # must also satisfy Inv
        step_holds = True
        step_cex: Optional[Tuple[str, str]] = None

        for state in self._kripke.states:
            labels = self._kripke.labels.get(state, frozenset())
            if not prop.evaluate(state, labels):
                continue  # only need implication from Inv(s)

            for succ in self._kripke.successors(state):
                succ_labels = self._kripke.labels.get(succ, frozenset())
                if not prop.evaluate(succ, succ_labels):
                    step_holds = False
                    step_cex = (state, succ)
                    break
            if not step_holds:
                break

        return InductiveCheckResult(
            property_name=prop.name,
            is_inductive=base_holds and step_holds,
            base_holds=base_holds,
            step_holds=step_holds,
            step_counterexample=step_cex,
        )

    # -- k-induction --------------------------------------------------------

    def check_k_induction(
        self,
        prop: SafetyProperty,
        max_k: int = 10,
    ) -> InductiveCheckResult:
        """k-induction: strengthen induction by unrolling k steps.

        Base case: all states reachable in ≤ k steps satisfy Inv.
        Step: if Inv holds for k consecutive states on any path, it holds
              for the (k+1)-th.

        Returns the result for the smallest k that succeeds, or the
        last failure if no k ≤ max_k works.
        """
        logger.info("k-induction for: %s (max_k=%d)", prop.name, max_k)

        for k in range(1, max_k + 1):
            result = self._k_induction_step(prop, k)
            if result.is_inductive:
                return result

        # Return last failure
        return self._k_induction_step(prop, max_k)

    def _k_induction_step(
        self,
        prop: SafetyProperty,
        k: int,
    ) -> InductiveCheckResult:
        """Perform k-induction at depth k."""
        # Base case: BFS up to depth k
        base_result = self.check_invariant(prop, max_depth=k)
        if not base_result.holds:
            return InductiveCheckResult(
                property_name=prop.name,
                is_inductive=False,
                base_holds=False,
                step_holds=False,
                k=k,
            )

        # Inductive step: check k-step paths
        step_holds = True
        step_cex: Optional[Tuple[str, str]] = None

        for state in self._kripke.states:
            labels = self._kripke.labels.get(state, frozenset())
            if not prop.evaluate(state, labels):
                continue

            # Check if k consecutive steps from state all satisfy inv
            # implies the (k+1)-th does too
            if not self._k_step_induction(state, prop, k):
                step_holds = False
                # Find the failing successor
                for succ in self._kripke.successors(state):
                    succ_labels = self._kripke.labels.get(succ, frozenset())
                    if not prop.evaluate(succ, succ_labels):
                        step_cex = (state, succ)
                        break
                if step_cex is None:
                    step_cex = (state, state)
                break

        return InductiveCheckResult(
            property_name=prop.name,
            is_inductive=base_result.holds and step_holds,
            base_holds=base_result.holds,
            step_holds=step_holds,
            step_counterexample=step_cex,
            k=k,
        )

    def _k_step_induction(
        self,
        state: str,
        prop: SafetyProperty,
        k: int,
    ) -> bool:
        """Check whether the k-step induction hypothesis holds from *state*.

        Enumerate all paths of length k from state.  If all states on every
        such path satisfy the invariant, then every successor of the last
        state must also satisfy it (by the induction hypothesis).
        """
        # Generate paths of length k starting from state
        paths: List[List[str]] = [[state]]

        for _ in range(k):
            new_paths: List[List[str]] = []
            for path in paths:
                last = path[-1]
                succs = self._kripke.successors(last)
                if not succs:
                    new_paths.append(path)
                    continue
                for s in succs:
                    new_paths.append(path + [s])
            paths = new_paths

        # Check that the invariant holds for all states on all paths
        for path in paths:
            all_hold = True
            for s in path:
                labels = self._kripke.labels.get(s, frozenset())
                if not prop.evaluate(s, labels):
                    all_hold = False
                    break

            if all_hold and path:
                last = path[-1]
                for succ in self._kripke.successors(last):
                    succ_labels = self._kripke.labels.get(succ, frozenset())
                    if not prop.evaluate(succ, succ_labels):
                        return False
        return True

    # -- Differential checking: original vs quotient -----------------------

    def check_differential(
        self,
        prop: SafetyProperty,
        quotient_coalg: object,
        projection: Mapping[str, str],
    ) -> Tuple[SafetyCheckResult, SafetyCheckResult, bool]:
        """Check *prop* on both original and quotient, compare results.

        Returns (original_result, quotient_result, agree) where *agree*
        is True iff both agree on whether the property holds.
        """
        orig_result = self.check_invariant(prop)

        q_checker = SafetyChecker(quotient_coalg)
        q_result = q_checker.check_invariant(prop)

        agree = orig_result.holds == q_result.holds

        if not agree:
            logger.warning(
                "DISCREPANCY in safety check '%s': original=%s, quotient=%s",
                prop.name,
                "SAFE" if orig_result.holds else "UNSAFE",
                "SAFE" if q_result.holds else "UNSAFE",
            )

        return orig_result, q_result, agree

    # -- Property strengthening suggestions --------------------------------

    def suggest_strengthening(
        self,
        prop: SafetyProperty,
    ) -> List[StrengtheningCandidate]:
        """Suggest stronger invariants that might be inductive.

        Heuristics:
          1. Conjoin with reachability information.
          2. Add state-predicate constraints from backward analysis.
        """
        candidates: List[StrengtheningCandidate] = []

        # Heuristic 1: reachable states predicate
        reachable = self._bfs_reachable()

        def reachable_and_inv(state: str, labels: FrozenSet[str]) -> bool:
            return state in reachable and prop.evaluate(state, labels)

        cand1 = StrengtheningCandidate(
            original=prop.name,
            strengthened_predicate=reachable_and_inv,
            description=f"Reachable ∧ {prop.name}",
        )
        # Check if this strengthening is inductive
        strong_prop = SafetyProperty(
            name=cand1.description,
            kind=SafetyKind.STATE_INVARIANT,
            predicate=reachable_and_inv,
        )
        ind_result = self.check_inductive(strong_prop)
        cand1 = StrengtheningCandidate(
            original=prop.name,
            strengthened_predicate=reachable_and_inv,
            description=cand1.description,
            is_inductive=ind_result.is_inductive,
        )
        candidates.append(cand1)

        # Heuristic 2: conjunction with backward invariant from safe states
        safe_states = frozenset(
            s for s in self._kripke.states
            if prop.evaluate(s, self._kripke.labels.get(s, frozenset()))
        )
        backward_safe = self._backward_reachable(safe_states)

        def backward_strengthened(state: str, labels: FrozenSet[str]) -> bool:
            return state in backward_safe and prop.evaluate(state, labels)

        cand2_desc = f"BackwardSafe ∧ {prop.name}"
        strong_prop2 = SafetyProperty(
            name=cand2_desc,
            kind=SafetyKind.STATE_INVARIANT,
            predicate=backward_strengthened,
        )
        ind_result2 = self.check_inductive(strong_prop2)
        candidates.append(StrengtheningCandidate(
            original=prop.name,
            strengthened_predicate=backward_strengthened,
            description=cand2_desc,
            is_inductive=ind_result2.is_inductive,
        ))

        # Heuristic 3: remove states not co-reachable from initial
        deadlock = frozenset(
            s for s in self._kripke.states
            if not self._kripke.successors(s)
        )
        non_deadlock = self._kripke.states - deadlock

        def no_deadlock_inv(state: str, labels: FrozenSet[str]) -> bool:
            return state in non_deadlock and prop.evaluate(state, labels)

        cand3_desc = f"NonDeadlock ∧ {prop.name}"
        strong_prop3 = SafetyProperty(
            name=cand3_desc,
            kind=SafetyKind.STATE_INVARIANT,
            predicate=no_deadlock_inv,
        )
        ind_result3 = self.check_inductive(strong_prop3)
        candidates.append(StrengtheningCandidate(
            original=prop.name,
            strengthened_predicate=no_deadlock_inv,
            description=cand3_desc,
            is_inductive=ind_result3.is_inductive,
        ))

        return candidates

    # -- Batch checking -----------------------------------------------------

    def check_all(
        self,
        properties: List[SafetyProperty],
    ) -> List[SafetyCheckResult]:
        """Check multiple safety properties, stopping at first violation."""
        results: List[SafetyCheckResult] = []
        for prop in properties:
            result = self.check_invariant(prop)
            results.append(result)
        return results

    def check_all_inductive(
        self,
        properties: List[SafetyProperty],
    ) -> List[InductiveCheckResult]:
        """Check multiple properties for inductiveness."""
        return [self.check_inductive(p) for p in properties]

    # -- Utility helpers ----------------------------------------------------

    def _reconstruct_path(
        self,
        target: str,
        parent: Dict[str, Optional[str]],
    ) -> List[str]:
        """Reconstruct the BFS path from initial state to *target*."""
        path: List[str] = []
        cur: Optional[str] = target
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path

    def _bfs_reachable(self) -> FrozenSet[str]:
        """Return all BFS-reachable states from initial states."""
        visited: Set[str] = set()
        queue: deque[str] = deque()
        for init in self._kripke.initial_states:
            visited.add(init)
            queue.append(init)
        while queue:
            s = queue.popleft()
            for t in self._kripke.successors(s):
                if t not in visited:
                    visited.add(t)
                    queue.append(t)
        return frozenset(visited)

    def _backward_reachable(self, targets: FrozenSet[str]) -> FrozenSet[str]:
        """Return all states from which *targets* is backward-reachable."""
        rev = self._kripke.reverse_graph()
        visited: Set[str] = set(targets)
        queue: deque[str] = deque(targets)
        while queue:
            s = queue.popleft()
            for pred in rev.get(s, set()):
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
        return frozenset(visited)


# ============================================================================
# Convenience constructors for common safety properties
# ============================================================================

def make_ap_invariant(proposition: str) -> SafetyProperty:
    """Create a safety property checking that AP *proposition* always holds."""
    return SafetyProperty(
        name=f"AG({proposition})",
        kind=SafetyKind.STATE_INVARIANT,
        predicate=lambda s, lbl, p=proposition: p in lbl,
        description=f"Invariant: {proposition} holds in all reachable states",
    )


def make_exclusion_invariant(prop_a: str, prop_b: str) -> SafetyProperty:
    """Mutual exclusion: prop_a and prop_b never both hold."""
    return SafetyProperty(
        name=f"AG(¬({prop_a} ∧ {prop_b}))",
        kind=SafetyKind.STATE_INVARIANT,
        predicate=lambda s, lbl, a=prop_a, b=prop_b: not (a in lbl and b in lbl),
        description=f"Mutual exclusion: {prop_a} and {prop_b}",
    )


def make_type_invariant(
    name: str,
    valid_labels: FrozenSet[str],
) -> SafetyProperty:
    """Type invariant: state labels must be a subset of valid_labels."""
    return SafetyProperty(
        name=name,
        kind=SafetyKind.TYPE_INVARIANT,
        predicate=lambda s, lbl, vl=valid_labels: lbl <= vl,
        description=f"Type invariant: labels ⊆ {valid_labels}",
    )


def make_action_constraint(
    name: str,
    allowed_actions: FrozenSet[str],
) -> ActionSafetyProperty:
    """Action constraint: only allowed actions may be taken."""
    return ActionSafetyProperty(
        name=name,
        constraint=lambda s, a, d, sl, dl, aa=allowed_actions: a in aa,
        description=f"Only actions {allowed_actions} permitted",
    )
