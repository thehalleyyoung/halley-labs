"""
Bisimulation closure validation for CoaCert-TLA witnesses.

Verifies that the equivalence relation encoded in a witness is indeed a
bisimulation (or stuttering bisimulation) over the induced transition system.
Checks forward closure, backward closure, AP preservation, and stutter-step
bounds.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
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
    TransitionWitness,
    WitnessData,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

MAX_STUTTER_DEPTH = 1000  # safety bound for stutter-step chains


class ViolationKind(Enum):
    FORWARD_CLOSURE = auto()
    BACKWARD_CLOSURE = auto()
    AP_PRESERVATION = auto()
    STUTTER_BOUND_EXCEEDED = auto()
    STUTTER_CHAIN_OPEN = auto()
    MISSING_WITNESS = auto()
    INVALID_PATH = auto()


@dataclass(frozen=True)
class ClosureViolation:
    """Concrete counterexample for a closure property violation."""
    kind: ViolationKind
    state_s: str
    state_t: str
    class_s: int
    class_t: int
    message: str
    transition_label: Optional[str] = None
    expected_path: Optional[Tuple[str, ...]] = None
    actual_path: Optional[Tuple[str, ...]] = None


@dataclass
class ClosureResult:
    """Aggregated result of closure validation."""
    passed: bool = True
    forward_checked: int = 0
    backward_checked: int = 0
    ap_checked: int = 0
    stutter_chains_checked: int = 0
    violations: List[ClosureViolation] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    mode: str = "full"

    def add_violation(self, v: ClosureViolation) -> None:
        self.passed = False
        self.violations.append(v)

    @property
    def first_violation(self) -> Optional[ClosureViolation]:
        return self.violations[0] if self.violations else None


# ---------------------------------------------------------------------------
# Internal indices
# ---------------------------------------------------------------------------


@dataclass
class _TransitionIndex:
    """Bidirectional index of transitions by state and equivalence class."""

    # state -> list of (target_state, transition_witness)
    outgoing: Dict[str, List[Tuple[str, TransitionWitness]]] = field(
        default_factory=dict
    )
    # state -> list of (source_state, transition_witness)
    incoming: Dict[str, List[Tuple[str, TransitionWitness]]] = field(
        default_factory=dict
    )
    # (source_class, target_class) -> list of transition witnesses
    class_transitions: Dict[Tuple[int, int], List[TransitionWitness]] = field(
        default_factory=dict
    )

    def add(self, tw: TransitionWitness) -> None:
        self.outgoing.setdefault(tw.original_source, []).append(
            (tw.original_target, tw)
        )
        self.incoming.setdefault(tw.original_target, []).append(
            (tw.original_source, tw)
        )
        key = (tw.source_class, tw.target_class)
        self.class_transitions.setdefault(key, []).append(tw)


# ---------------------------------------------------------------------------
# Closure validator
# ---------------------------------------------------------------------------


class ClosureValidator:
    """Validates bisimulation closure properties of a witness."""

    def __init__(self, witness: WitnessData, *,
                 stutter_bound: int = MAX_STUTTER_DEPTH):
        self._witness = witness
        self._stutter_bound = stutter_bound

        # Build state-to-class and class-to-states mappings
        self._state_to_class: Dict[str, int] = {}
        self._class_to_states: Dict[int, Set[str]] = {}
        self._class_to_aps: Dict[int, FrozenSet[str]] = {}
        self._state_aps: Dict[str, FrozenSet[str]] = {}

        for eq in witness.equivalences:
            aps = frozenset(eq.ap_labels)
            self._class_to_aps[eq.class_id] = aps
            members = set(eq.members)
            members.add(eq.representative)
            self._class_to_states[eq.class_id] = members
            for s in members:
                self._state_to_class[s] = eq.class_id
                self._state_aps[s] = aps

        # Build transition index
        self._tx_idx = _TransitionIndex()
        for tw in witness.transitions:
            self._tx_idx.add(tw)

        # Equivalence pairs: all (s, t) where s ~ t (same class)
        self._is_stuttering = witness.header.has_stuttering

    # -- public API ---------------------------------------------------------

    def validate_full(self) -> ClosureResult:
        """Exhaustive closure validation."""
        result = ClosureResult(mode="full")
        t0 = time.monotonic()

        t_ap = time.monotonic()
        self._check_ap_preservation(result)
        result.timing_breakdown["ap_preservation"] = time.monotonic() - t_ap

        t_fwd = time.monotonic()
        self._check_forward_closure(result)
        result.timing_breakdown["forward_closure"] = time.monotonic() - t_fwd

        t_bwd = time.monotonic()
        self._check_backward_closure(result)
        result.timing_breakdown["backward_closure"] = time.monotonic() - t_bwd

        if self._is_stuttering:
            t_st = time.monotonic()
            self._check_stutter_bounds(result)
            result.timing_breakdown["stutter_bounds"] = time.monotonic() - t_st

        result.elapsed_seconds = time.monotonic() - t0
        return result

    def validate_statistical(self, sample_fraction: float = 0.1,
                             seed: int = 42) -> ClosureResult:
        """Random sampling closure check."""
        result = ClosureResult(mode="statistical")
        rng = random.Random(seed)
        t0 = time.monotonic()

        # Sample equivalence classes for AP check
        classes = list(self._class_to_states.keys())
        sample_size = max(1, int(len(classes) * sample_fraction))
        sampled_classes = rng.sample(classes, min(sample_size, len(classes)))
        self._check_ap_preservation_subset(result, sampled_classes)

        # Sample transitions for forward/backward
        transitions = list(self._witness.transitions)
        tx_sample = max(1, int(len(transitions) * sample_fraction))
        sampled_tx = rng.sample(transitions, min(tx_sample, len(transitions)))
        self._check_forward_closure_subset(result, sampled_tx)
        self._check_backward_closure_subset(result, sampled_tx)

        if self._is_stuttering:
            stutter_tx = [tw for tw in sampled_tx if tw.is_stutter]
            self._check_stutter_bounds_subset(result, stutter_tx)

        result.elapsed_seconds = time.monotonic() - t0
        return result

    # -- AP preservation ----------------------------------------------------

    def _check_ap_preservation(self, result: ClosureResult) -> None:
        """All states in the same equivalence class must have identical AP."""
        for eq in self._witness.equivalences:
            canonical_aps = frozenset(eq.ap_labels)
            all_members = set(eq.members)
            all_members.add(eq.representative)
            for s in all_members:
                result.ap_checked += 1
                state_aps = self._state_aps.get(s)
                if state_aps is None:
                    continue
                if state_aps != canonical_aps:
                    result.add_violation(ClosureViolation(
                        kind=ViolationKind.AP_PRESERVATION,
                        state_s=s,
                        state_t=eq.representative,
                        class_s=eq.class_id,
                        class_t=eq.class_id,
                        message=(
                            f"State {s!r} has APs {set(state_aps)} but class "
                            f"{eq.class_id} has APs {set(canonical_aps)}"
                        ),
                    ))

    def _check_ap_preservation_subset(self, result: ClosureResult,
                                      class_ids: Sequence[int]) -> None:
        for cid in class_ids:
            states = self._class_to_states.get(cid, set())
            canonical_aps = self._class_to_aps.get(cid, frozenset())
            for s in states:
                result.ap_checked += 1
                s_aps = self._state_aps.get(s, frozenset())
                if s_aps != canonical_aps:
                    result.add_violation(ClosureViolation(
                        kind=ViolationKind.AP_PRESERVATION,
                        state_s=s,
                        state_t="",
                        class_s=cid,
                        class_t=cid,
                        message=(
                            f"AP mismatch for {s!r} in class {cid}: "
                            f"{set(s_aps)} vs {set(canonical_aps)}"
                        ),
                    ))

    # -- forward closure ----------------------------------------------------

    def _check_forward_closure(self, result: ClosureResult) -> None:
        """For every s ~ t and s -> s', verify there exists t ->...-> t'
        with s' ~ t'."""
        for eq in self._witness.equivalences:
            all_members = set(eq.members)
            all_members.add(eq.representative)
            pairs = self._generate_pairs(all_members)
            for s, t in pairs:
                self._check_forward_pair(s, t, eq.class_id, result)

    def _check_forward_pair(self, s: str, t: str, class_id: int,
                            result: ClosureResult) -> None:
        s_outgoing = self._tx_idx.outgoing.get(s, [])
        for s_prime, tw_s in s_outgoing:
            result.forward_checked += 1
            s_prime_class = self._state_to_class.get(s_prime)
            if s_prime_class is None:
                result.add_violation(ClosureViolation(
                    kind=ViolationKind.MISSING_WITNESS,
                    state_s=s,
                    state_t=t,
                    class_s=class_id,
                    class_t=class_id,
                    message=f"Successor {s_prime!r} of {s!r} not in any class",
                ))
                continue

            # Find a matching transition from t to some t' in s_prime's class
            found = self._find_matching_transition(
                t, s_prime_class, result
            )
            if not found:
                result.add_violation(ClosureViolation(
                    kind=ViolationKind.FORWARD_CLOSURE,
                    state_s=s,
                    state_t=t,
                    class_s=class_id,
                    class_t=class_id,
                    message=(
                        f"Forward closure violation: {s!r} -> {s_prime!r} "
                        f"(class {s_prime_class}) but no matching transition "
                        f"from {t!r} to class {s_prime_class}"
                    ),
                    transition_label=s_prime,
                ))

    def _check_forward_closure_subset(self, result: ClosureResult,
                                      transitions: Sequence[TransitionWitness]
                                      ) -> None:
        for tw in transitions:
            src_class = tw.source_class
            src_states = self._class_to_states.get(src_class, set())
            for t in src_states:
                if t == tw.original_source:
                    continue
                result.forward_checked += 1
                tgt_class = tw.target_class
                found = self._find_matching_transition(t, tgt_class, result)
                if not found:
                    result.add_violation(ClosureViolation(
                        kind=ViolationKind.FORWARD_CLOSURE,
                        state_s=tw.original_source,
                        state_t=t,
                        class_s=src_class,
                        class_t=src_class,
                        message=(
                            f"Forward closure violation (sampled): "
                            f"{tw.original_source!r} -> {tw.original_target!r} "
                            f"but {t!r} has no match to class {tgt_class}"
                        ),
                    ))

    # -- backward closure ---------------------------------------------------

    def _check_backward_closure(self, result: ClosureResult) -> None:
        """For every s ~ t and t -> t', verify there exists s ->...-> s'
        with s' ~ t'."""
        for eq in self._witness.equivalences:
            all_members = set(eq.members)
            all_members.add(eq.representative)
            pairs = self._generate_pairs(all_members)
            for s, t in pairs:
                self._check_backward_pair(s, t, eq.class_id, result)

    def _check_backward_pair(self, s: str, t: str, class_id: int,
                             result: ClosureResult) -> None:
        t_outgoing = self._tx_idx.outgoing.get(t, [])
        for t_prime, tw_t in t_outgoing:
            result.backward_checked += 1
            t_prime_class = self._state_to_class.get(t_prime)
            if t_prime_class is None:
                result.add_violation(ClosureViolation(
                    kind=ViolationKind.MISSING_WITNESS,
                    state_s=s,
                    state_t=t,
                    class_s=class_id,
                    class_t=class_id,
                    message=f"Successor {t_prime!r} of {t!r} not in any class",
                ))
                continue

            found = self._find_matching_transition(
                s, t_prime_class, result
            )
            if not found:
                result.add_violation(ClosureViolation(
                    kind=ViolationKind.BACKWARD_CLOSURE,
                    state_s=s,
                    state_t=t,
                    class_s=class_id,
                    class_t=class_id,
                    message=(
                        f"Backward closure violation: {t!r} -> {t_prime!r} "
                        f"(class {t_prime_class}) but no matching transition "
                        f"from {s!r} to class {t_prime_class}"
                    ),
                    transition_label=t_prime,
                ))

    def _check_backward_closure_subset(self, result: ClosureResult,
                                       transitions: Sequence[TransitionWitness]
                                       ) -> None:
        for tw in transitions:
            src_class = tw.source_class
            src_states = self._class_to_states.get(src_class, set())
            for s in src_states:
                if s == tw.original_source:
                    continue
                result.backward_checked += 1
                tgt_class = tw.target_class
                found = self._find_matching_transition(s, tgt_class, result)
                if not found:
                    result.add_violation(ClosureViolation(
                        kind=ViolationKind.BACKWARD_CLOSURE,
                        state_s=s,
                        state_t=tw.original_source,
                        class_s=src_class,
                        class_t=src_class,
                        message=(
                            f"Backward closure violation (sampled): "
                            f"{tw.original_source!r} -> {tw.original_target!r} "
                            f"but {s!r} has no match to class {tgt_class}"
                        ),
                    ))

    # -- stutter bounds -----------------------------------------------------

    def _check_stutter_bounds(self, result: ClosureResult) -> None:
        """Verify stutter-step chains are properly bounded."""
        for tw in self._witness.transitions:
            if not tw.is_stutter:
                continue
            result.stutter_chains_checked += 1
            self._validate_stutter_chain(tw, result)

    def _check_stutter_bounds_subset(self, result: ClosureResult,
                                     transitions: Sequence[TransitionWitness]
                                     ) -> None:
        for tw in transitions:
            result.stutter_chains_checked += 1
            self._validate_stutter_chain(tw, result)

    def _validate_stutter_chain(self, tw: TransitionWitness,
                                result: ClosureResult) -> None:
        """Validate that a stutter chain is bounded and well-formed."""
        if tw.stutter_depth > self._stutter_bound:
            result.add_violation(ClosureViolation(
                kind=ViolationKind.STUTTER_BOUND_EXCEEDED,
                state_s=tw.original_source,
                state_t=tw.original_target,
                class_s=tw.source_class,
                class_t=tw.target_class,
                message=(
                    f"Stutter depth {tw.stutter_depth} exceeds bound "
                    f"{self._stutter_bound}"
                ),
            ))
            return

        # Verify stutter chain: all intermediate states must remain in the
        # same equivalence class as the source.
        path = tw.matching_path
        if not path:
            return

        src_class = tw.source_class
        for i, intermediate in enumerate(path[:-1]):
            inter_class = self._state_to_class.get(intermediate)
            if inter_class is None:
                result.add_violation(ClosureViolation(
                    kind=ViolationKind.STUTTER_CHAIN_OPEN,
                    state_s=tw.original_source,
                    state_t=intermediate,
                    class_s=src_class,
                    class_t=-1,
                    message=(
                        f"Stutter chain intermediate {intermediate!r} at "
                        f"position {i} not in any equivalence class"
                    ),
                    actual_path=tw.matching_path,
                ))
                return
            if inter_class != src_class:
                result.add_violation(ClosureViolation(
                    kind=ViolationKind.STUTTER_CHAIN_OPEN,
                    state_s=tw.original_source,
                    state_t=intermediate,
                    class_s=src_class,
                    class_t=inter_class,
                    message=(
                        f"Stutter chain breaks: intermediate {intermediate!r} "
                        f"(class {inter_class}) differs from source class "
                        f"{src_class}"
                    ),
                    actual_path=tw.matching_path,
                ))
                return

        # Final state of stutter chain must be in the target class
        final = path[-1]
        final_class = self._state_to_class.get(final)
        if final_class is not None and final_class != tw.target_class:
            result.add_violation(ClosureViolation(
                kind=ViolationKind.INVALID_PATH,
                state_s=tw.original_source,
                state_t=final,
                class_s=tw.source_class,
                class_t=tw.target_class,
                message=(
                    f"Stutter chain final state {final!r} is in class "
                    f"{final_class}, expected class {tw.target_class}"
                ),
                actual_path=tw.matching_path,
            ))

    # -- matching helpers ---------------------------------------------------

    def _find_matching_transition(self, state: str, target_class: int,
                                  result: ClosureResult) -> bool:
        """Check if *state* has any outgoing transition reaching *target_class*.

        For stuttering bisimulation, we allow multi-step paths where
        intermediate states remain in the same class as *state*.
        """
        outgoing = self._tx_idx.outgoing.get(state, [])
        for succ, tw in outgoing:
            succ_class = self._state_to_class.get(succ)
            if succ_class == target_class:
                return True

        if self._is_stuttering:
            # BFS allowing stutter steps (intermediate in same class)
            src_class = self._state_to_class.get(state)
            if src_class is None:
                return False
            visited: Set[str] = {state}
            frontier: List[str] = [state]
            depth = 0
            while frontier and depth < self._stutter_bound:
                next_frontier: List[str] = []
                for s in frontier:
                    for succ, tw in self._tx_idx.outgoing.get(s, []):
                        if succ in visited:
                            continue
                        succ_class = self._state_to_class.get(succ)
                        if succ_class == target_class:
                            return True
                        if succ_class == src_class:
                            visited.add(succ)
                            next_frontier.append(succ)
                frontier = next_frontier
                depth += 1

        return False

    @staticmethod
    def _generate_pairs(members: Set[str]) -> List[Tuple[str, str]]:
        """Generate all ordered pairs from the member set.

        For large classes, this is O(n²); callers may want to sample.
        """
        sorted_members = sorted(members)
        pairs: List[Tuple[str, str]] = []
        for i, s in enumerate(sorted_members):
            for t in sorted_members[i + 1:]:
                pairs.append((s, t))
                pairs.append((t, s))
        return pairs
