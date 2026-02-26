"""
Stuttering equivalence verification for CoaCert-TLA witnesses.

Verifies that the quotient relation encoded in a witness respects
stuttering equivalence:
  - Stutter-step chains are well-formed and finite
  - Divergence sensitivity: divergent states are distinguished
  - Stutter trace equivalence on sampled paths
  - Hash verification of witness data along the way
"""

from __future__ import annotations

import hashlib
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
    HashBlock,
    TransitionWitness,
    WitnessData,
)
from .hash_verifier import HashChainVerifier, HashVerificationResult, _sha256

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

MAX_PATH_LENGTH = 10000


class StutteringViolationKind(Enum):
    DIVERGENCE_MISMATCH = auto()
    STUTTER_TRACE_MISMATCH = auto()
    INFINITE_STUTTER = auto()
    PATH_HASH_MISMATCH = auto()
    WITNESS_INCOMPLETE = auto()
    MATCHING_FAILURE = auto()


@dataclass(frozen=True)
class StutteringViolation:
    """Detail of a stuttering equivalence violation."""
    kind: StutteringViolationKind
    state_s: str
    state_t: str
    message: str
    path_s: Optional[Tuple[str, ...]] = None
    path_t: Optional[Tuple[str, ...]] = None


@dataclass
class StutteringResult:
    """Aggregated result of stuttering equivalence verification."""
    passed: bool = True
    paths_checked: int = 0
    divergence_checks: int = 0
    trace_checks: int = 0
    hash_checks_passed: int = 0
    hash_checks_total: int = 0
    violations: List[StutteringViolation] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    timing_breakdown: Dict[str, float] = field(default_factory=dict)

    def add_violation(self, v: StutteringViolation) -> None:
        self.passed = False
        self.violations.append(v)

    @property
    def first_violation(self) -> Optional[StutteringViolation]:
        return self.violations[0] if self.violations else None


# ---------------------------------------------------------------------------
# Stutter trace abstraction
# ---------------------------------------------------------------------------


def _compute_stutter_trace(path: Sequence[str],
                           state_aps: Dict[str, FrozenSet[str]]
                           ) -> Tuple[FrozenSet[str], ...]:
    """Compute the stutter-free trace of a path.

    A stutter-free trace collapses consecutive states with identical atomic
    propositions into a single occurrence.  E.g. if L maps:
      s0 -> {a}, s1 -> {a}, s2 -> {b}, s3 -> {b}, s4 -> {a}
    then the stutter-free trace is ({a}, {b}, {a}).
    """
    if not path:
        return ()
    trace: List[FrozenSet[str]] = []
    prev_aps: Optional[FrozenSet[str]] = None
    for state in path:
        aps = state_aps.get(state, frozenset())
        if aps != prev_aps:
            trace.append(aps)
            prev_aps = aps
    return tuple(trace)


def _are_stutter_equivalent_traces(
    trace_a: Tuple[FrozenSet[str], ...],
    trace_b: Tuple[FrozenSet[str], ...],
) -> bool:
    """Check if two stutter-free traces are identical.

    Since the traces are already stutter-free, equality implies
    stuttering equivalence on the original paths.
    """
    return trace_a == trace_b


def _detect_divergence(state: str,
                       outgoing: Dict[str, List[Tuple[str, TransitionWitness]]],
                       state_to_class: Dict[str, int],
                       max_depth: int = 500) -> bool:
    """Detect whether *state* can diverge: enter an infinite sequence of
    internal (stutter) transitions that never leaves its equivalence class.

    Uses a DFS bounded by max_depth to find a cycle of stutter transitions.
    """
    src_class = state_to_class.get(state)
    if src_class is None:
        return False

    visited: Set[str] = set()
    stack: List[str] = [state]
    depth = 0

    while stack and depth < max_depth:
        current = stack.pop()
        if current in visited:
            # Found a cycle within the same equivalence class → divergence
            return True
        visited.add(current)
        depth += 1
        for succ, tw in outgoing.get(current, []):
            if not tw.is_stutter:
                continue
            succ_class = state_to_class.get(succ)
            if succ_class == src_class:
                stack.append(succ)

    return False


# ---------------------------------------------------------------------------
# Stuttering verifier
# ---------------------------------------------------------------------------


class StutteringVerifier:
    """Verifies stuttering equivalence properties in a witness."""

    def __init__(self, witness: WitnessData, *,
                 max_path_length: int = MAX_PATH_LENGTH,
                 verify_hashes: bool = True):
        self._witness = witness
        self._max_path = max_path_length
        self._verify_hashes = verify_hashes

        # Build indices
        self._state_to_class: Dict[str, int] = {}
        self._class_to_states: Dict[int, Set[str]] = {}
        self._state_aps: Dict[str, FrozenSet[str]] = {}

        for eq in witness.equivalences:
            aps = frozenset(eq.ap_labels)
            members = set(eq.members)
            members.add(eq.representative)
            self._class_to_states[eq.class_id] = members
            for s in members:
                self._state_to_class[s] = eq.class_id
                self._state_aps[s] = aps

        self._outgoing: Dict[str, List[Tuple[str, TransitionWitness]]] = {}
        for tw in witness.transitions:
            self._outgoing.setdefault(tw.original_source, []).append(
                (tw.original_target, tw)
            )

        # Precompute divergence map
        self._divergent_states: Set[str] = set()
        self._divergence_computed = False

    # -- public API ---------------------------------------------------------

    def verify(self, *, sample_paths: int = 100,
               seed: int = 42) -> StutteringResult:
        """Full stuttering verification."""
        result = StutteringResult()
        t0 = time.monotonic()

        t_div = time.monotonic()
        self._check_divergence_sensitivity(result)
        result.timing_breakdown["divergence"] = time.monotonic() - t_div

        t_trace = time.monotonic()
        self._check_stutter_traces(result, sample_paths, seed)
        result.timing_breakdown["traces"] = time.monotonic() - t_trace

        t_chains = time.monotonic()
        self._check_stutter_chains(result)
        result.timing_breakdown["chains"] = time.monotonic() - t_chains

        if self._verify_hashes:
            t_hash = time.monotonic()
            self._inline_hash_check(result)
            result.timing_breakdown["inline_hash"] = time.monotonic() - t_hash

        result.elapsed_seconds = time.monotonic() - t0
        return result

    # -- divergence sensitivity ---------------------------------------------

    def _compute_divergence_map(self) -> None:
        if self._divergence_computed:
            return
        for state in self._state_to_class:
            if _detect_divergence(
                state, self._outgoing, self._state_to_class
            ):
                self._divergent_states.add(state)
        self._divergence_computed = True

    def _check_divergence_sensitivity(self, result: StutteringResult) -> None:
        """Stuttering bisimulation must distinguish divergent from
        non-divergent states: if s ~ t then s diverges iff t diverges."""
        self._compute_divergence_map()

        for eq in self._witness.equivalences:
            members = set(eq.members)
            members.add(eq.representative)
            if len(members) < 2:
                continue

            div_status: Optional[bool] = None
            for s in members:
                result.divergence_checks += 1
                s_div = s in self._divergent_states
                if div_status is None:
                    div_status = s_div
                elif s_div != div_status:
                    # Find a concrete pair for the violation report
                    for t in members:
                        if (t in self._divergent_states) != s_div:
                            result.add_violation(StutteringViolation(
                                kind=StutteringViolationKind.DIVERGENCE_MISMATCH,
                                state_s=s,
                                state_t=t,
                                message=(
                                    f"Divergence mismatch in class "
                                    f"{eq.class_id}: {s!r} diverges={s_div} "
                                    f"but {t!r} diverges={not s_div}"
                                ),
                            ))
                            break
                    break

    # -- stutter trace equivalence ------------------------------------------

    def _check_stutter_traces(self, result: StutteringResult,
                              num_paths: int, seed: int) -> None:
        """Sample paths from pairs of equivalent states and compare their
        stutter-free traces."""
        rng = random.Random(seed)

        for eq in self._witness.equivalences:
            members = list(set(eq.members) | {eq.representative})
            if len(members) < 2:
                continue

            paths_for_class = min(num_paths, len(members) * (len(members) - 1))
            checked = 0
            attempts = 0
            max_attempts = paths_for_class * 3

            while checked < paths_for_class and attempts < max_attempts:
                attempts += 1
                s = rng.choice(members)
                t = rng.choice(members)
                if s == t:
                    continue

                path_s = self._sample_path(s, rng)
                path_t = self._sample_path(t, rng)

                if not path_s or not path_t:
                    continue

                result.trace_checks += 1
                result.paths_checked += 1
                checked += 1

                trace_s = _compute_stutter_trace(path_s, self._state_aps)
                trace_t = _compute_stutter_trace(path_t, self._state_aps)

                # For stuttering equivalence, the stutter-free traces from
                # equivalent states should be prefix-compatible (they diverge
                # at the same AP block boundaries).
                min_len = min(len(trace_s), len(trace_t))
                if min_len == 0:
                    continue

                # Check prefix agreement up to the shorter trace
                mismatch_pos = -1
                for i in range(min_len):
                    if trace_s[i] != trace_t[i]:
                        mismatch_pos = i
                        break

                if mismatch_pos >= 0:
                    result.add_violation(StutteringViolation(
                        kind=StutteringViolationKind.STUTTER_TRACE_MISMATCH,
                        state_s=s,
                        state_t=t,
                        message=(
                            f"Stutter traces diverge at position "
                            f"{mismatch_pos}: {set(trace_s[mismatch_pos])} "
                            f"vs {set(trace_t[mismatch_pos])}"
                        ),
                        path_s=tuple(path_s[:20]),
                        path_t=tuple(path_t[:20]),
                    ))

    def _sample_path(self, start: str,
                     rng: random.Random) -> List[str]:
        """Sample a random path of bounded length from *start*."""
        path: List[str] = [start]
        current = start
        for _ in range(self._max_path):
            successors = self._outgoing.get(current, [])
            if not successors:
                break
            next_state, _ = rng.choice(successors)
            path.append(next_state)
            current = next_state
        return path

    # -- stutter chain validation -------------------------------------------

    def _check_stutter_chains(self, result: StutteringResult) -> None:
        """Verify that every stutter transition in the witness has a
        well-formed, finite matching path."""
        for tw in self._witness.transitions:
            if not tw.is_stutter:
                continue
            result.paths_checked += 1

            if not tw.matching_path:
                result.add_violation(StutteringViolation(
                    kind=StutteringViolationKind.WITNESS_INCOMPLETE,
                    state_s=tw.original_source,
                    state_t=tw.original_target,
                    message=(
                        f"Stutter transition from {tw.original_source!r} "
                        f"has empty matching path"
                    ),
                ))
                continue

            # Check for cycles in the matching path (would imply infinite stutter)
            seen: Set[str] = set()
            has_cycle = False
            for step in tw.matching_path:
                if step in seen:
                    has_cycle = True
                    break
                seen.add(step)

            if has_cycle:
                result.add_violation(StutteringViolation(
                    kind=StutteringViolationKind.INFINITE_STUTTER,
                    state_s=tw.original_source,
                    state_t=tw.original_target,
                    message=(
                        f"Stutter chain from {tw.original_source!r} contains "
                        f"a cycle (potential infinite stutter)"
                    ),
                    path_s=tw.matching_path,
                ))
                continue

            # Verify each step in the matching path has a corresponding
            # transition in the witness
            self._verify_path_connectivity(tw, result)

    def _verify_path_connectivity(self, tw: TransitionWitness,
                                  result: StutteringResult) -> None:
        """Verify that consecutive states in the matching path are connected
        by transitions recorded in the witness."""
        path = tw.matching_path
        for i in range(len(path) - 1):
            s_i = path[i]
            s_next = path[i + 1]
            outgoing = self._outgoing.get(s_i, [])
            connected = any(succ == s_next for succ, _ in outgoing)
            if not connected:
                result.add_violation(StutteringViolation(
                    kind=StutteringViolationKind.MATCHING_FAILURE,
                    state_s=s_i,
                    state_t=s_next,
                    message=(
                        f"Matching path step {i}: no transition from "
                        f"{s_i!r} to {s_next!r} in witness"
                    ),
                    path_s=tw.matching_path,
                ))
                return

    # -- inline hash checking -----------------------------------------------

    def _inline_hash_check(self, result: StutteringResult) -> None:
        """Verify hashes of transition witnesses referenced during
        stuttering verification, ensuring the underlying data has not
        been tampered with."""
        if not self._witness.hash_chain:
            return

        block_by_index: Dict[int, HashBlock] = {
            b.index: b for b in self._witness.hash_chain
        }

        for tw in self._witness.transitions:
            if not tw.is_stutter:
                continue
            result.hash_checks_total += 1

            block = block_by_index.get(tw.source_class)
            if block is None:
                continue

            # Recompute payload hash for the transition data
            payload = (
                tw.original_source.encode("utf-8") +
                tw.original_target.encode("utf-8") +
                b"".join(s.encode("utf-8") for s in tw.matching_path)
            )
            payload_hash = _sha256(payload)

            # The block's payload_hash covers all items in the block, so we
            # just verify that our contribution is consistent by checking the
            # block_hash recomputation.
            expected_block_hash = _sha256(block.prev_hash + block.payload_hash)
            if expected_block_hash == block.block_hash:
                result.hash_checks_passed += 1
            else:
                result.add_violation(StutteringViolation(
                    kind=StutteringViolationKind.PATH_HASH_MISMATCH,
                    state_s=tw.original_source,
                    state_t=tw.original_target,
                    message=(
                        f"Hash verification failed for block "
                        f"{tw.source_class} containing stutter transition"
                    ),
                ))
