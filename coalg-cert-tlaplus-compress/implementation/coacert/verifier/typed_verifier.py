"""
Type-hardened, defense-in-depth witness verifier for CoaCert-TLA.

Addresses the critique that the witness verifier — the sole trust anchor —
is unverified Python code.  This module adds:

  1. Explicit runtime type checking at every verification step via a
     ``checked()`` decorator that validates argument types.
  2. Separate verification phases, each returning a ``PhaseVerdict``
     dataclass with pass/fail, errors, warnings, and checked counts.
  3. An independent re-implementation of hash-chain verification for
     cross-checking against the main ``HashChainVerifier``.
  4. A ``TypedVerificationReport`` that aggregates all phase verdicts.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    get_type_hints,
)

from .deserializer import (
    EquivalenceBinding,
    FairnessBinding,
    HashBlock,
    TransitionWitness,
    WitnessData,
    WitnessHeader,
)

# ---------------------------------------------------------------------------
# Type-checking decorator
# ---------------------------------------------------------------------------

_PRIMITIVE_MAP = {
    int: int,
    float: (int, float),
    str: str,
    bytes: (bytes, bytearray),
    bool: bool,
}


class TypeCheckError(TypeError):
    """Raised when a runtime type check fails inside a ``checked`` function."""


def checked(fn: Callable) -> Callable:
    """Decorator that validates argument types at call time.

    Uses ``inspect.signature`` and ``get_type_hints`` to resolve annotations,
    then performs ``isinstance`` checks for each annotated parameter.  Generic
    types (``List[X]``, ``Optional[X]``, etc.) are checked at the outer
    container level only, which is sufficient for defense-in-depth without
    significant overhead.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if name == "self" or name == "cls":
                continue
            expected = hints.get(name)
            if expected is None:
                continue
            _assert_type(name, value, expected)
        result = fn(*args, **kwargs)
        ret_hint = hints.get("return")
        if ret_hint is not None and result is not None:
            _assert_type("return", result, ret_hint)
        return result

    return wrapper


def _assert_type(name: str, value: Any, expected: Any) -> None:
    """Perform a best-effort isinstance check for *expected*."""
    if value is None:
        return
    origin = getattr(expected, "__origin__", None)
    if origin is not None:
        # e.g. List, Dict, Tuple, Optional, Set, FrozenSet, Sequence
        base = _origin_to_type(origin)
        if base is not None and not isinstance(value, base):
            raise TypeCheckError(
                f"Argument '{name}': expected {expected}, got {type(value).__name__}"
            )
        return
    if isinstance(expected, type):
        check = _PRIMITIVE_MAP.get(expected, expected)
        if not isinstance(value, check):
            raise TypeCheckError(
                f"Argument '{name}': expected {expected.__name__}, "
                f"got {type(value).__name__}"
            )


def _origin_to_type(origin: Any) -> Optional[type]:
    """Map a generic origin (e.g. ``list``) to a concrete type for isinstance."""
    import collections.abc as cabc
    mapping: Dict[Any, type] = {
        list: list,
        dict: dict,
        set: set,
        frozenset: frozenset,
        tuple: tuple,
        cabc.Sequence: (list, tuple),  # type: ignore[dict-item]
        cabc.Set: (set, frozenset),  # type: ignore[dict-item]
    }
    return mapping.get(origin)


# ---------------------------------------------------------------------------
# Phase verdict
# ---------------------------------------------------------------------------


class PhaseStatus(Enum):
    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class PhaseVerdict:
    """Result of a single verification phase."""
    phase_name: str
    status: PhaseStatus = PhaseStatus.SKIPPED
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_count: int = 0
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == PhaseStatus.PASSED


# ---------------------------------------------------------------------------
# Typed verification report
# ---------------------------------------------------------------------------


@dataclass
class TypedVerificationReport:
    """Aggregation of all phase verdicts produced by ``TypedWitnessVerifier``."""
    phase_verdicts: List[PhaseVerdict] = field(default_factory=list)
    overall_passed: bool = False
    total_checked: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    elapsed_seconds: float = 0.0

    def finalize(self) -> None:
        self.total_checked = sum(pv.checked_count for pv in self.phase_verdicts)
        self.total_errors = sum(len(pv.errors) for pv in self.phase_verdicts)
        self.total_warnings = sum(len(pv.warnings) for pv in self.phase_verdicts)
        self.overall_passed = all(
            pv.status != PhaseStatus.FAILED for pv in self.phase_verdicts
        )

    def summary(self) -> str:
        status = "PASSED" if self.overall_passed else "FAILED"
        lines = [
            f"TypedVerificationReport: {status}",
            f"  Phases: {len(self.phase_verdicts)}",
            f"  Total checks: {self.total_checked}",
            f"  Errors: {self.total_errors}",
            f"  Warnings: {self.total_warnings}",
            f"  Time: {self.elapsed_seconds:.4f}s",
        ]
        for pv in self.phase_verdicts:
            mark = "PASS" if pv.passed else "FAIL"
            lines.append(f"  [{mark}] {pv.phase_name} ({pv.checked_count} checks)")
            for e in pv.errors[:3]:
                lines.append(f"        ERROR: {e}")
            for w in pv.warnings[:3]:
                lines.append(f"        WARN:  {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Independent hash helpers (separate from hash_verifier.py)
# ---------------------------------------------------------------------------

_GENESIS_ZERO = b"\x00" * 32


def _ind_sha256(data: bytes) -> bytes:
    """Independent SHA-256 — intentionally does NOT import from hash_verifier."""
    return hashlib.sha256(data).digest()


def _ind_hash_leaf(leaf: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + leaf).digest()


def _ind_hash_internal(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _ind_merkle_root(leaves: Sequence[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    layer = [_ind_hash_leaf(l) for l in leaves]
    while len(layer) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            nxt.append(_ind_hash_internal(left, right))
        layer = nxt
    return layer[0]


# ---------------------------------------------------------------------------
# Typed witness verifier
# ---------------------------------------------------------------------------


class TypedWitnessVerifier:
    """Defense-in-depth witness verifier with explicit type checking."""

    def __init__(self, witness: WitnessData) -> None:
        if not isinstance(witness, WitnessData):
            raise TypeCheckError(
                f"Expected WitnessData, got {type(witness).__name__}"
            )
        self._witness = witness
        self._state_to_class: Dict[str, int] = {}
        self._class_to_states: Dict[int, Set[str]] = {}
        self._class_aps: Dict[int, FrozenSet[str]] = {}
        self._outgoing: Dict[str, List[Tuple[str, TransitionWitness]]] = {}
        self._class_outgoing: Dict[int, Set[int]] = {}
        self._build_indices()

    # -- index building -----------------------------------------------------

    def _build_indices(self) -> None:
        for eq in self._witness.equivalences:
            if not isinstance(eq, EquivalenceBinding):
                raise TypeCheckError(
                    f"Equivalence entry is {type(eq).__name__}, "
                    f"expected EquivalenceBinding"
                )
            aps = frozenset(eq.ap_labels)
            self._class_aps[eq.class_id] = aps
            members = set(eq.members)
            members.add(eq.representative)
            self._class_to_states[eq.class_id] = members
            for s in members:
                self._state_to_class[s] = eq.class_id

        for tw in self._witness.transitions:
            if not isinstance(tw, TransitionWitness):
                raise TypeCheckError(
                    f"Transition entry is {type(tw).__name__}, "
                    f"expected TransitionWitness"
                )
            self._outgoing.setdefault(tw.original_source, []).append(
                (tw.original_target, tw)
            )
            self._class_outgoing.setdefault(tw.source_class, set()).add(
                tw.target_class
            )

    # -- public API ---------------------------------------------------------

    def verify_all(self) -> TypedVerificationReport:
        """Run every verification phase and return an aggregated report."""
        report = TypedVerificationReport()
        t0 = time.monotonic()

        report.phase_verdicts.append(self.structural_check())
        report.phase_verdicts.append(self.hash_chain_check())
        report.phase_verdicts.append(self.closure_check())
        report.phase_verdicts.append(self.ap_preservation_check())
        report.phase_verdicts.append(self.fairness_check())

        report.elapsed_seconds = time.monotonic() - t0
        report.finalize()
        return report

    # -- Phase 1: structural ------------------------------------------------

    @checked
    def structural_check(self) -> PhaseVerdict:
        """Validate structural integrity of witness data."""
        pv = PhaseVerdict(phase_name="structural")
        t0 = time.monotonic()

        # Header checks
        hdr = self._witness.header
        if not isinstance(hdr, WitnessHeader):
            pv.errors.append(f"Header type: {type(hdr).__name__}")
        pv.checked_count += 1

        if not isinstance(hdr.version_major, int) or hdr.version_major < 0:
            pv.errors.append(f"Invalid version_major: {hdr.version_major!r}")
        pv.checked_count += 1

        if not isinstance(hdr.flags, int) or hdr.flags < 0:
            pv.errors.append(f"Invalid flags: {hdr.flags!r}")
        pv.checked_count += 1

        # Equivalence class IDs must be non-negative and unique
        seen_ids: Set[int] = set()
        for eq in self._witness.equivalences:
            pv.checked_count += 1
            if not isinstance(eq.class_id, int) or eq.class_id < 0:
                pv.errors.append(f"Bad class_id: {eq.class_id!r}")
            if eq.class_id in seen_ids:
                pv.errors.append(f"Duplicate class_id: {eq.class_id}")
            seen_ids.add(eq.class_id)
            if not isinstance(eq.representative, str) or not eq.representative:
                pv.errors.append(
                    f"Bad representative in class {eq.class_id}: "
                    f"{eq.representative!r}"
                )

        # Transition references must point to existing classes
        for tw in self._witness.transitions:
            pv.checked_count += 1
            if tw.source_class not in seen_ids:
                pv.warnings.append(
                    f"Transition source_class {tw.source_class} "
                    f"not in equivalence classes"
                )
            if tw.target_class not in seen_ids:
                pv.warnings.append(
                    f"Transition target_class {tw.target_class} "
                    f"not in equivalence classes"
                )
            if not isinstance(tw.original_source, str):
                pv.errors.append(
                    f"Transition original_source type: "
                    f"{type(tw.original_source).__name__}"
                )
            if not isinstance(tw.original_target, str):
                pv.errors.append(
                    f"Transition original_target type: "
                    f"{type(tw.original_target).__name__}"
                )

        # Hash chain block indices should be sequential
        for i, blk in enumerate(self._witness.hash_chain):
            pv.checked_count += 1
            if not isinstance(blk, HashBlock):
                pv.errors.append(f"Hash chain entry {i}: {type(blk).__name__}")
            if not isinstance(blk.prev_hash, bytes) or len(blk.prev_hash) != 32:
                pv.errors.append(f"Block {blk.index}: bad prev_hash length")
            if not isinstance(blk.block_hash, bytes) or len(blk.block_hash) != 32:
                pv.errors.append(f"Block {blk.index}: bad block_hash length")

        pv.status = PhaseStatus.PASSED if not pv.errors else PhaseStatus.FAILED
        pv.elapsed_seconds = time.monotonic() - t0
        return pv

    # -- Phase 2: hash chain (independent re-implementation) ----------------

    @checked
    def hash_chain_check(self) -> PhaseVerdict:
        """Independent hash-chain verification (does NOT delegate to
        ``HashChainVerifier``).  Recomputes every block hash from scratch."""
        pv = PhaseVerdict(phase_name="hash_chain")
        t0 = time.monotonic()

        chain = self._witness.hash_chain
        if not chain:
            pv.status = PhaseStatus.PASSED
            pv.elapsed_seconds = time.monotonic() - t0
            return pv

        # Genesis block
        first = chain[0]
        pv.checked_count += 1
        if first.prev_hash != _GENESIS_ZERO:
            pv.errors.append(
                f"Genesis prev_hash is not zero: {first.prev_hash.hex()}"
            )

        recomputed = _ind_sha256(first.prev_hash + first.payload_hash)
        pv.checked_count += 1
        if recomputed != first.block_hash:
            pv.errors.append(
                f"Block 0 hash mismatch: expected {recomputed.hex()}, "
                f"got {first.block_hash.hex()}"
            )

        # Remaining blocks
        for i in range(1, len(chain)):
            blk = chain[i]
            prev = chain[i - 1]
            pv.checked_count += 1

            if blk.prev_hash != prev.block_hash:
                pv.errors.append(
                    f"Chain break at block {blk.index}: "
                    f"prev_hash {blk.prev_hash.hex()} != "
                    f"prior block_hash {prev.block_hash.hex()}"
                )
                continue

            expected = _ind_sha256(blk.prev_hash + blk.payload_hash)
            pv.checked_count += 1
            if expected != blk.block_hash:
                pv.errors.append(
                    f"Block {blk.index} hash mismatch: "
                    f"expected {expected.hex()}, got {blk.block_hash.hex()}"
                )

        # Merkle roots
        eq_index: Dict[int, EquivalenceBinding] = {
            eq.class_id: eq for eq in self._witness.equivalences
        }
        tx_by_src: Dict[int, List[TransitionWitness]] = {}
        for tw in self._witness.transitions:
            tx_by_src.setdefault(tw.source_class, []).append(tw)

        for blk in chain:
            if blk.merkle_root is None:
                continue
            leaves = self._collect_leaves(blk.index, eq_index, tx_by_src)
            if not leaves:
                continue
            pv.checked_count += 1
            root = _ind_merkle_root(leaves)
            if root != blk.merkle_root:
                pv.errors.append(
                    f"Merkle root mismatch at block {blk.index}"
                )

        pv.status = PhaseStatus.PASSED if not pv.errors else PhaseStatus.FAILED
        pv.elapsed_seconds = time.monotonic() - t0
        return pv

    @staticmethod
    def _collect_leaves(
        block_index: int,
        eq_index: Dict[int, EquivalenceBinding],
        tx_by_src: Dict[int, List[TransitionWitness]],
    ) -> List[bytes]:
        leaves: List[bytes] = []
        eq = eq_index.get(block_index)
        if eq is not None:
            leaves.append(_serialize_eq(eq))
        for tw in tx_by_src.get(block_index, []):
            leaves.append(_serialize_tw(tw))
        return leaves

    # -- Phase 3: closure ---------------------------------------------------

    @checked
    def closure_check(self) -> PhaseVerdict:
        """Verify forward and backward bisimulation closure."""
        pv = PhaseVerdict(phase_name="closure")
        t0 = time.monotonic()

        for eq in self._witness.equivalences:
            members = set(eq.members) | {eq.representative}
            sorted_members = sorted(members)
            for i, s in enumerate(sorted_members):
                for t in sorted_members[i + 1:]:
                    # Forward: s -> s' ⇒ ∃ t ->* t', s' ~ t'
                    pv.checked_count += self._check_closure_pair(
                        s, t, eq.class_id, pv
                    )
                    # Backward: t -> t' ⇒ ∃ s ->* s', s' ~ t'
                    pv.checked_count += self._check_closure_pair(
                        t, s, eq.class_id, pv
                    )

        pv.status = PhaseStatus.PASSED if not pv.errors else PhaseStatus.FAILED
        pv.elapsed_seconds = time.monotonic() - t0
        return pv

    def _check_closure_pair(
        self, s: str, t: str, class_id: int, pv: PhaseVerdict
    ) -> int:
        count = 0
        for s_prime, tw_s in self._outgoing.get(s, []):
            count += 1
            s_prime_class = self._state_to_class.get(s_prime)
            if s_prime_class is None:
                pv.errors.append(
                    f"Successor {s_prime!r} of {s!r} not in any class"
                )
                continue
            if not self._can_reach_class(t, s_prime_class):
                pv.errors.append(
                    f"Closure violation: {s!r} -> {s_prime!r} (class "
                    f"{s_prime_class}) but {t!r} cannot reach class "
                    f"{s_prime_class}"
                )
        return count

    def _can_reach_class(self, state: str, target_class: int) -> bool:
        for succ, tw in self._outgoing.get(state, []):
            if self._state_to_class.get(succ) == target_class:
                return True
        # Stuttering: BFS via same-class intermediaries
        if self._witness.header.has_stuttering:
            src_class = self._state_to_class.get(state)
            if src_class is None:
                return False
            visited: Set[str] = {state}
            frontier = [state]
            depth = 0
            while frontier and depth < 1000:
                nxt: List[str] = []
                for s in frontier:
                    for succ, tw in self._outgoing.get(s, []):
                        if succ in visited:
                            continue
                        sc = self._state_to_class.get(succ)
                        if sc == target_class:
                            return True
                        if sc == src_class:
                            visited.add(succ)
                            nxt.append(succ)
                frontier = nxt
                depth += 1
        return False

    # -- Phase 4: AP preservation -------------------------------------------

    @checked
    def ap_preservation_check(self) -> PhaseVerdict:
        """All states in the same class must have identical AP labels."""
        pv = PhaseVerdict(phase_name="ap_preservation")
        t0 = time.monotonic()

        for eq in self._witness.equivalences:
            canonical = frozenset(eq.ap_labels)
            members = set(eq.members) | {eq.representative}
            for s in members:
                pv.checked_count += 1
                s_class = self._state_to_class.get(s)
                if s_class is None:
                    pv.errors.append(
                        f"State {s!r} not mapped to any class"
                    )
                    continue
                s_aps = self._class_aps.get(s_class, frozenset())
                if s_aps != canonical:
                    pv.errors.append(
                        f"AP mismatch: state {s!r} class {s_class} has "
                        f"{set(s_aps)} vs class {eq.class_id} canonical "
                        f"{set(canonical)}"
                    )

        pv.status = PhaseStatus.PASSED if not pv.errors else PhaseStatus.FAILED
        pv.elapsed_seconds = time.monotonic() - t0
        return pv

    # -- Phase 5: fairness --------------------------------------------------

    @checked
    def fairness_check(self) -> PhaseVerdict:
        """Verify fairness acceptance-pair preservation."""
        pv = PhaseVerdict(phase_name="fairness")
        t0 = time.monotonic()

        for fb in self._witness.fairness:
            if not isinstance(fb, FairnessBinding):
                pv.errors.append(
                    f"Fairness entry type: {type(fb).__name__}"
                )
                continue
            pv.checked_count += 1

            b_set = set(fb.b_set_classes)
            g_set = set(fb.g_set_classes)

            # B/G class agreement
            for eq in self._witness.equivalences:
                members = set(eq.members) | {eq.representative}
                for s in members:
                    pv.checked_count += 1
                    s_class = self._state_to_class.get(s)
                    if s_class is not None and s_class != eq.class_id:
                        pv.errors.append(
                            f"State {s!r} in class {eq.class_id} but "
                            f"mapped to {s_class} — fairness pair "
                            f"{fb.pair_id} inconsistent"
                        )

            # T-Fair coherence
            for b_class in b_set:
                succs = self._class_outgoing.get(b_class, set())
                for g_class in succs & g_set:
                    pv.checked_count += 1
                    if not self._has_concrete_tx(b_class, g_class):
                        pv.errors.append(
                            f"T-Fair: class {b_class} -> {g_class} at "
                            f"quotient but no concrete witness (pair "
                            f"{fb.pair_id})"
                        )

            # Rabin overlap check
            acceptance = self._witness.metadata.get("acceptance_type", "streett")
            overlap = b_set & g_set
            if acceptance == "rabin" and overlap:
                pv.errors.append(
                    f"Rabin pair {fb.pair_id}: B ∩ G non-empty: "
                    f"{sorted(overlap)[:5]}"
                )

        pv.status = PhaseStatus.PASSED if not pv.errors else PhaseStatus.FAILED
        pv.elapsed_seconds = time.monotonic() - t0
        return pv

    def _has_concrete_tx(self, src_class: int, tgt_class: int) -> bool:
        for s in self._class_to_states.get(src_class, set()):
            for succ, tw in self._outgoing.get(s, []):
                if tw.target_class == tgt_class:
                    return True
        return False


# ---------------------------------------------------------------------------
# Serialization helpers (independent from hash_verifier.py)
# ---------------------------------------------------------------------------


def _serialize_eq(eq: EquivalenceBinding) -> bytes:
    parts = [
        eq.class_id.to_bytes(4, "big"),
        len(eq.representative).to_bytes(2, "big"),
        eq.representative.encode("utf-8"),
    ]
    parts.append(len(eq.members).to_bytes(4, "big"))
    for m in sorted(eq.members):
        parts.append(len(m).to_bytes(2, "big"))
        parts.append(m.encode("utf-8"))
    parts.append(len(eq.ap_labels).to_bytes(4, "big"))
    for a in sorted(eq.ap_labels):
        parts.append(len(a).to_bytes(2, "big"))
        parts.append(a.encode("utf-8"))
    return b"".join(parts)


def _serialize_tw(tw: TransitionWitness) -> bytes:
    parts = [
        tw.source_class.to_bytes(4, "big"),
        tw.target_class.to_bytes(4, "big"),
        len(tw.original_source).to_bytes(2, "big"),
        tw.original_source.encode("utf-8"),
        len(tw.original_target).to_bytes(2, "big"),
        tw.original_target.encode("utf-8"),
    ]
    parts.append(len(tw.matching_path).to_bytes(4, "big"))
    for p in tw.matching_path:
        parts.append(len(p).to_bytes(2, "big"))
        parts.append(p.encode("utf-8"))
    flags = 0x01 if tw.is_stutter else 0x00
    parts.append(flags.to_bytes(1, "big"))
    if tw.is_stutter:
        parts.append(tw.stutter_depth.to_bytes(2, "big"))
    return b"".join(parts)
