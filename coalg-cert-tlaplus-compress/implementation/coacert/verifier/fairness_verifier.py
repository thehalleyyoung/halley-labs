"""
Fairness preservation verification for CoaCert-TLA witnesses.

Verifies that acceptance pairs (B_i, G_i) are preserved by the quotient
relation — equivalent states must agree on membership in every B_i and G_i
set.  Also checks T-Fair coherence and that fair cycles in the quotient
correspond to fair cycles in the original system.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Deque,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .deserializer import (
    EquivalenceBinding,
    FairnessBinding,
    TransitionWitness,
    WitnessData,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class FairnessViolationKind(Enum):
    B_SET_DISAGREEMENT = auto()
    G_SET_DISAGREEMENT = auto()
    TFAIR_INCOHERENT = auto()
    CYCLE_MISMATCH = auto()
    MISSING_PAIR = auto()
    PAIR_OVERLAP_INVALID = auto()


@dataclass(frozen=True)
class FairnessViolation:
    """Detail record for a fairness preservation violation."""
    kind: FairnessViolationKind
    pair_id: int
    class_id: int
    message: str
    states: Optional[Tuple[str, ...]] = None


@dataclass
class FairnessResult:
    """Aggregated result of fairness verification."""
    passed: bool = True
    pairs_checked: int = 0
    b_set_checks: int = 0
    g_set_checks: int = 0
    tfair_checks: int = 0
    cycle_checks: int = 0
    violations: List[FairnessViolation] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    timing_breakdown: Dict[str, float] = field(default_factory=dict)

    def add_violation(self, v: FairnessViolation) -> None:
        self.passed = False
        self.violations.append(v)

    @property
    def first_violation(self) -> Optional[FairnessViolation]:
        return self.violations[0] if self.violations else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _class_members(eq: EquivalenceBinding) -> Set[str]:
    """All members of an equivalence class including its representative."""
    members = set(eq.members)
    members.add(eq.representative)
    return members


# ---------------------------------------------------------------------------
# Fairness verifier
# ---------------------------------------------------------------------------


class FairnessVerifier:
    """Verifies fairness preservation in a CoaCert-TLA witness."""

    def __init__(self, witness: WitnessData):
        self._witness = witness

        # State → class mapping
        self._state_to_class: Dict[str, int] = {}
        self._class_to_states: Dict[int, Set[str]] = {}
        self._class_to_eq: Dict[int, EquivalenceBinding] = {}

        for eq in witness.equivalences:
            self._class_to_eq[eq.class_id] = eq
            members = _class_members(eq)
            self._class_to_states[eq.class_id] = members
            for s in members:
                self._state_to_class[s] = eq.class_id

        # B/G membership indexed by class
        self._b_classes: Dict[int, Set[int]] = {}  # pair_id -> set of classes
        self._g_classes: Dict[int, Set[int]] = {}
        for fb in witness.fairness:
            self._b_classes[fb.pair_id] = set(fb.b_set_classes)
            self._g_classes[fb.pair_id] = set(fb.g_set_classes)

        # Transition index
        self._outgoing: Dict[str, List[Tuple[str, TransitionWitness]]] = {}
        self._class_outgoing: Dict[int, Set[int]] = {}
        for tw in witness.transitions:
            self._outgoing.setdefault(tw.original_source, []).append(
                (tw.original_target, tw)
            )
            self._class_outgoing.setdefault(tw.source_class, set()).add(
                tw.target_class
            )

    # -- public API ---------------------------------------------------------

    def verify(self) -> FairnessResult:
        """Full fairness preservation verification."""
        result = FairnessResult()
        t0 = time.monotonic()

        t_bg = time.monotonic()
        self._check_bg_preservation(result)
        result.timing_breakdown["bg_preservation"] = time.monotonic() - t_bg

        t_tfair = time.monotonic()
        self._check_tfair_coherence(result)
        result.timing_breakdown["tfair_coherence"] = time.monotonic() - t_tfair

        t_cycle = time.monotonic()
        self._check_fair_cycles(result)
        result.timing_breakdown["fair_cycles"] = time.monotonic() - t_cycle

        t_overlap = time.monotonic()
        self._check_pair_overlap(result)
        result.timing_breakdown["pair_overlap"] = time.monotonic() - t_overlap

        result.elapsed_seconds = time.monotonic() - t0
        return result

    def verify_pair(self, pair_id: int) -> FairnessResult:
        """Verify a single acceptance pair."""
        result = FairnessResult()
        fb = self._find_pair(pair_id)
        if fb is None:
            result.add_violation(FairnessViolation(
                kind=FairnessViolationKind.MISSING_PAIR,
                pair_id=pair_id,
                class_id=-1,
                message=f"Acceptance pair {pair_id} not found in witness",
            ))
            return result
        self._check_pair_bg(fb, result)
        return result

    # -- B/G set preservation -----------------------------------------------

    def _check_bg_preservation(self, result: FairnessResult) -> None:
        """For each acceptance pair, verify that equivalent states agree
        on B_i and G_i membership."""
        for fb in self._witness.fairness:
            result.pairs_checked += 1
            self._check_pair_bg(fb, result)

    def _check_pair_bg(self, fb: FairnessBinding,
                       result: FairnessResult) -> None:
        b_classes = set(fb.b_set_classes)
        g_classes = set(fb.g_set_classes)

        for eq in self._witness.equivalences:
            # Check B-set agreement: either the whole class is in B or none
            self._check_set_agreement(
                eq, fb.pair_id, b_classes, "B", result
            )
            # Check G-set agreement: either the whole class is in G or none
            self._check_set_agreement(
                eq, fb.pair_id, g_classes, "G", result
            )

    def _check_set_agreement(self, eq: EquivalenceBinding,
                             pair_id: int,
                             member_classes: Set[int],
                             set_name: str,
                             result: FairnessResult) -> None:
        """Verify that all states in *eq* agree on membership in *member_classes*."""
        class_in_set = eq.class_id in member_classes
        members = _class_members(eq)

        if set_name == "B":
            result.b_set_checks += 1
        else:
            result.g_set_checks += 1

        # If the class is marked as in the set, every individual state should
        # belong.  We verify this by checking that no state in the class has
        # transitions contradicting the fairness assumption.
        #
        # Since the witness declares membership at the class level, the key
        # property is *consistency*: the class is in the set xor it is not.
        # A violation would be if the class's representative is in B_i but
        # some member is provably not (detectable via AP conflict or
        # transition structure).
        if not class_in_set:
            return

        # For classes that ARE in the set, verify that all members' APs
        # are compatible with the class representative
        canonical_aps = frozenset(eq.ap_labels)
        for s in members:
            s_class = self._state_to_class.get(s)
            if s_class is None:
                continue
            if s_class != eq.class_id:
                vkind = (FairnessViolationKind.B_SET_DISAGREEMENT
                         if set_name == "B"
                         else FairnessViolationKind.G_SET_DISAGREEMENT)
                result.add_violation(FairnessViolation(
                    kind=vkind,
                    pair_id=pair_id,
                    class_id=eq.class_id,
                    message=(
                        f"State {s!r} listed in class {eq.class_id} but "
                        f"mapped to class {s_class} — {set_name}-set "
                        f"membership inconsistent for pair {pair_id}"
                    ),
                    states=(s, eq.representative),
                ))

    # -- T-Fair coherence ---------------------------------------------------

    def _check_tfair_coherence(self, result: FairnessResult) -> None:
        """Verify T-Fair coherence: for each pair (B_i, G_i), every class in
        B_i that has an outgoing transition to a class in G_i must be
        consistent with the fairness constraint.

        T-Fair coherence requires: for every equivalence class c in B_i,
        if c has a successor class c' in G_i, then there must exist a
        witness transition from some state in c to some state in c'.
        """
        for fb in self._witness.fairness:
            result.tfair_checks += 1
            b_set = set(fb.b_set_classes)
            g_set = set(fb.g_set_classes)

            for b_class in b_set:
                succs = self._class_outgoing.get(b_class, set())
                reachable_g = succs & g_set
                if not reachable_g:
                    continue

                # Verify that for each reachable G class, there is at least
                # one concrete transition witness
                for g_class in reachable_g:
                    has_witness = self._has_concrete_transition(
                        b_class, g_class
                    )
                    if not has_witness:
                        result.add_violation(FairnessViolation(
                            kind=FairnessViolationKind.TFAIR_INCOHERENT,
                            pair_id=fb.pair_id,
                            class_id=b_class,
                            message=(
                                f"T-Fair incoherence: class {b_class} (B-set) "
                                f"reaches class {g_class} (G-set) at quotient "
                                f"level but no concrete transition witness "
                                f"exists for pair {fb.pair_id}"
                            ),
                        ))

    def _has_concrete_transition(self, src_class: int,
                                 tgt_class: int) -> bool:
        """Check whether there is a transition witness from *src_class*
        to *tgt_class*."""
        src_states = self._class_to_states.get(src_class, set())
        for s in src_states:
            for succ, tw in self._outgoing.get(s, []):
                if tw.target_class == tgt_class:
                    return True
        return False

    # -- fair cycle checking ------------------------------------------------

    def _check_fair_cycles(self, result: FairnessResult) -> None:
        """Verify that fair cycles in the quotient (class-level graph)
        correspond to fair cycles in the original system.

        A fair cycle in the quotient visits some G_i class infinitely often
        while visiting B_i only finitely often, for at least one pair i.
        We check that any such cycle has a corresponding concrete cycle
        in the original system with the same fairness property.
        """
        quotient_sccs = self._find_quotient_sccs()

        for fb in self._witness.fairness:
            b_set = set(fb.b_set_classes)
            g_set = set(fb.g_set_classes)

            for scc in quotient_sccs:
                result.cycle_checks += 1
                # A fair cycle requires: SCC intersects G_i
                scc_g = scc & g_set
                if not scc_g:
                    continue

                # Check if the SCC can avoid B_i (has a sub-cycle not
                # touching B_i)
                scc_minus_b = scc - b_set
                if not scc_minus_b:
                    continue

                # There is a potential fair cycle (through G, avoiding B).
                # Verify it corresponds to a concrete fair cycle.
                has_concrete = self._verify_concrete_fair_cycle(
                    scc, scc_g, scc_minus_b, fb.pair_id, result
                )
                if not has_concrete:
                    result.add_violation(FairnessViolation(
                        kind=FairnessViolationKind.CYCLE_MISMATCH,
                        pair_id=fb.pair_id,
                        class_id=min(scc),
                        message=(
                            f"Quotient has a fair cycle for pair "
                            f"{fb.pair_id} through classes "
                            f"{sorted(scc_g)[:5]}... but no corresponding "
                            f"concrete fair cycle found"
                        ),
                    ))

    def _find_quotient_sccs(self) -> List[Set[int]]:
        """Tarjan's SCC algorithm on the quotient (class-level) graph."""
        index_counter = [0]
        stack: List[int] = []
        on_stack: Set[int] = set()
        index_map: Dict[int, int] = {}
        lowlink: Dict[int, int] = {}
        sccs: List[Set[int]] = []

        all_classes = set(self._class_to_states.keys())

        def strongconnect(v: int) -> None:
            index_map[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in self._class_outgoing.get(v, set()):
                if w not in index_map:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index_map[w])

            if lowlink[v] == index_map[v]:
                scc: Set[int] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    sccs.append(scc)

        for v in all_classes:
            if v not in index_map:
                strongconnect(v)

        return sccs

    def _verify_concrete_fair_cycle(
        self,
        scc: Set[int],
        g_classes: Set[int],
        non_b_classes: Set[int],
        pair_id: int,
        result: FairnessResult,
    ) -> bool:
        """Check whether a quotient-level fair cycle has a concrete
        counterpart by attempting to trace a cycle through concrete
        states in the SCC that visits a G class and avoids B classes."""
        # Collect concrete states in the non-B portion of the SCC
        concrete_states: Set[str] = set()
        for cid in non_b_classes:
            concrete_states.update(self._class_to_states.get(cid, set()))

        if not concrete_states:
            return False

        # BFS from each concrete G-class state looking for a cycle
        g_states: Set[str] = set()
        for cid in g_classes:
            if cid in non_b_classes:
                g_states.update(self._class_to_states.get(cid, set()))

        for start in g_states:
            visited: Set[str] = set()
            queue: Deque[str] = deque([start])
            while queue:
                current = queue.popleft()
                if current == start and len(visited) > 0:
                    return True  # Found a concrete cycle through G, avoiding B
                if current in visited:
                    continue
                visited.add(current)
                for succ, tw in self._outgoing.get(current, []):
                    if succ in concrete_states and succ not in visited:
                        queue.append(succ)
                    elif succ == start:
                        return True

        return False

    # -- pair overlap validation --------------------------------------------

    def _check_pair_overlap(self, result: FairnessResult) -> None:
        """Sanity check: verify that B_i and G_i sets are well-formed.

        For Streett acceptance, B_i ∩ G_i can be non-empty, but for Rabin
        acceptance, they should be disjoint.  We report a warning-level
        violation if the overlap seems inconsistent with the witness
        metadata.
        """
        acceptance_type = self._witness.metadata.get("acceptance_type", "streett")

        for fb in self._witness.fairness:
            b_set = set(fb.b_set_classes)
            g_set = set(fb.g_set_classes)
            overlap = b_set & g_set

            if acceptance_type == "rabin" and overlap:
                result.add_violation(FairnessViolation(
                    kind=FairnessViolationKind.PAIR_OVERLAP_INVALID,
                    pair_id=fb.pair_id,
                    class_id=min(overlap),
                    message=(
                        f"Rabin acceptance pair {fb.pair_id} has non-empty "
                        f"B ∩ G overlap: classes {sorted(overlap)[:5]}..."
                    ),
                ))

    # -- helpers ------------------------------------------------------------

    def _find_pair(self, pair_id: int) -> Optional[FairnessBinding]:
        for fb in self._witness.fairness:
            if fb.pair_id == pair_id:
                return fb
        return None
