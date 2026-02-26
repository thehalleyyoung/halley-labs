"""
Independent cross-validation checker for CoaCert-TLA witnesses.

Provides a completely independent verification implementation that uses
different algorithms and a separate hash computation path.  The main
purpose is cross-checking: if the main verifier and this independent
checker disagree on a witness, a ``VerificationDiscrepancy`` is raised,
signalling a potential bug in one of the two implementations.

Key design choices:
  - No code shared with ``hash_verifier.py`` or ``closure_validator.py``.
  - Hash computations use a fresh ``hashlib`` call chain.
  - Partition validity is checked using union-find instead of direct
    set lookups.
  - The ``CrossValidationReport`` compares verdicts field by field.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
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
    FairnessBinding,
    HashBlock,
    TransitionWitness,
    WitnessData,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VerificationDiscrepancy(Exception):
    """Raised when the independent checker disagrees with another verifier."""

    def __init__(self, phase: str, message: str,
                 main_result: Any = None, independent_result: Any = None):
        self.phase = phase
        self.main_result = main_result
        self.independent_result = independent_result
        super().__init__(f"[{phase}] {message}")


# ---------------------------------------------------------------------------
# Union-Find for partition validity
# ---------------------------------------------------------------------------


class _UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def make_set(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def same_set(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    @property
    def elements(self) -> Set[str]:
        return set(self._parent.keys())


# ---------------------------------------------------------------------------
# Independent result types
# ---------------------------------------------------------------------------


@dataclass
class IndependentPhaseResult:
    """Result of one phase in the independent checker."""
    phase_name: str
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    checked_count: int = 0
    elapsed_seconds: float = 0.0

    def add_error(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)


@dataclass
class CrossValidationReport:
    """Comparison of main verifier results vs independent checker results."""
    main_phases: Dict[str, bool] = field(default_factory=dict)
    independent_phases: Dict[str, bool] = field(default_factory=dict)
    agreements: List[str] = field(default_factory=list)
    discrepancies: List[str] = field(default_factory=list)
    overall_agreement: bool = True
    elapsed_seconds: float = 0.0

    def add_agreement(self, phase: str) -> None:
        self.agreements.append(phase)

    def add_discrepancy(self, phase: str, detail: str) -> None:
        self.discrepancies.append(f"{phase}: {detail}")
        self.overall_agreement = False

    def summary(self) -> str:
        status = "AGREE" if self.overall_agreement else "DISAGREE"
        lines = [
            f"CrossValidationReport: {status}",
            f"  Agreements: {len(self.agreements)}",
            f"  Discrepancies: {len(self.discrepancies)}",
        ]
        for d in self.discrepancies:
            lines.append(f"    DISCREPANCY: {d}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Independent checker
# ---------------------------------------------------------------------------


class IndependentChecker:
    """Re-verifies a witness using algorithms independent of the main
    verifier, enabling cross-validation."""

    def __init__(self, witness: WitnessData) -> None:
        self._witness = witness
        self._uf = _UnionFind()
        self._state_to_class: Dict[str, int] = {}
        self._class_to_states: Dict[int, Set[str]] = {}
        self._class_aps: Dict[int, FrozenSet[str]] = {}
        self._outgoing: Dict[str, List[Tuple[str, TransitionWitness]]] = {}
        self._build()

    def _build(self) -> None:
        for eq in self._witness.equivalences:
            members = set(eq.members) | {eq.representative}
            self._class_to_states[eq.class_id] = members
            self._class_aps[eq.class_id] = frozenset(eq.ap_labels)
            for s in members:
                self._uf.make_set(s)
                self._state_to_class[s] = eq.class_id
            sorted_m = sorted(members)
            for i in range(1, len(sorted_m)):
                self._uf.union(sorted_m[0], sorted_m[i])

        for tw in self._witness.transitions:
            self._outgoing.setdefault(tw.original_source, []).append(
                (tw.original_target, tw)
            )

    # -- public API ---------------------------------------------------------

    def check_all(self) -> List[IndependentPhaseResult]:
        """Run all independent checks and return phase results."""
        results = [
            self.check_hash_chain(),
            self.check_partition_validity(),
            self.check_closure(),
            self.check_ap_preservation(),
            self.check_fairness(),
        ]
        return results

    def cross_validate(
        self, main_results: Dict[str, bool]
    ) -> CrossValidationReport:
        """Compare independent results against main verifier results.

        *main_results* maps phase name → passed boolean from the main
        verifier.  Raises ``VerificationDiscrepancy`` on the first
        disagreement (fail-fast).
        """
        report = CrossValidationReport()
        t0 = time.monotonic()

        ind_results = self.check_all()
        ind_map = {r.phase_name: r.passed for r in ind_results}

        report.main_phases = dict(main_results)
        report.independent_phases = dict(ind_map)

        for phase, ind_passed in ind_map.items():
            main_passed = main_results.get(phase)
            if main_passed is None:
                continue
            if main_passed == ind_passed:
                report.add_agreement(phase)
            else:
                detail = (
                    f"main={main_passed}, independent={ind_passed}"
                )
                report.add_discrepancy(phase, detail)
                raise VerificationDiscrepancy(
                    phase=phase,
                    message=f"Verifier disagreement: {detail}",
                    main_result=main_passed,
                    independent_result=ind_passed,
                )

        report.elapsed_seconds = time.monotonic() - t0
        return report

    # -- Phase: hash chain --------------------------------------------------

    def check_hash_chain(self) -> IndependentPhaseResult:
        """Independent hash chain verification using separate hash calls."""
        res = IndependentPhaseResult(phase_name="hash_chain")
        t0 = time.monotonic()
        chain = self._witness.hash_chain

        if not chain:
            res.elapsed_seconds = time.monotonic() - t0
            return res

        genesis = chain[0]
        res.checked_count += 1
        if genesis.prev_hash != b"\x00" * 32:
            res.add_error(
                f"Genesis prev_hash non-zero: {genesis.prev_hash.hex()}"
            )

        res.checked_count += 1
        h = hashlib.sha256(genesis.prev_hash + genesis.payload_hash).digest()
        if h != genesis.block_hash:
            res.add_error(f"Block 0 hash mismatch")

        for i in range(1, len(chain)):
            blk, prev = chain[i], chain[i - 1]
            res.checked_count += 1
            if blk.prev_hash != prev.block_hash:
                res.add_error(f"Chain break at block {blk.index}")
                continue
            res.checked_count += 1
            expected = hashlib.sha256(
                blk.prev_hash + blk.payload_hash
            ).digest()
            if expected != blk.block_hash:
                res.add_error(f"Block {blk.index} hash mismatch")

        # Independent Merkle verification
        eq_idx = {eq.class_id: eq for eq in self._witness.equivalences}
        tx_src: Dict[int, List[TransitionWitness]] = {}
        for tw in self._witness.transitions:
            tx_src.setdefault(tw.source_class, []).append(tw)

        for blk in chain:
            if blk.merkle_root is None:
                continue
            leaves = self._ind_collect_leaves(blk.index, eq_idx, tx_src)
            if not leaves:
                continue
            res.checked_count += 1
            root = self._ind_merkle_root(leaves)
            if root != blk.merkle_root:
                res.add_error(f"Merkle root mismatch at block {blk.index}")

        res.elapsed_seconds = time.monotonic() - t0
        return res

    # -- Phase: partition validity ------------------------------------------

    def check_partition_validity(self) -> IndependentPhaseResult:
        """Verify partition structure using union-find cross-check.

        Ensures:
          - No state belongs to two different classes.
          - Union-find components agree with declared classes.
        """
        res = IndependentPhaseResult(phase_name="partition_validity")
        t0 = time.monotonic()

        # Check for multi-class membership
        state_classes: Dict[str, List[int]] = {}
        for eq in self._witness.equivalences:
            members = set(eq.members) | {eq.representative}
            for s in members:
                state_classes.setdefault(s, []).append(eq.class_id)

        for s, classes in state_classes.items():
            res.checked_count += 1
            if len(classes) > 1:
                res.add_error(
                    f"State {s!r} appears in classes: {classes}"
                )

        # Union-find consistency: states in the same declared class should
        # have the same UF representative.
        for eq in self._witness.equivalences:
            members = sorted(set(eq.members) | {eq.representative})
            if len(members) < 2:
                res.checked_count += 1
                continue
            rep_root = self._uf.find(members[0])
            for s in members[1:]:
                res.checked_count += 1
                if self._uf.find(s) != rep_root:
                    res.add_error(
                        f"UF inconsistency: {s!r} and {members[0]!r} "
                        f"should be in same set (class {eq.class_id})"
                    )

        # States in different classes must NOT share a UF root
        class_roots: Dict[int, str] = {}
        for eq in self._witness.equivalences:
            members = set(eq.members) | {eq.representative}
            if not members:
                continue
            root = self._uf.find(next(iter(members)))
            res.checked_count += 1
            for other_cid, other_root in class_roots.items():
                if root == other_root and other_cid != eq.class_id:
                    res.add_error(
                        f"Classes {eq.class_id} and {other_cid} share "
                        f"UF root {root!r}"
                    )
            class_roots[eq.class_id] = root

        res.elapsed_seconds = time.monotonic() - t0
        return res

    # -- Phase: closure (independent) ---------------------------------------

    def check_closure(self) -> IndependentPhaseResult:
        """Independent closure check using adjacency-set intersection."""
        res = IndependentPhaseResult(phase_name="closure")
        t0 = time.monotonic()

        for eq in self._witness.equivalences:
            members = sorted(set(eq.members) | {eq.representative})
            for i, s in enumerate(members):
                for t in members[i + 1:]:
                    res.checked_count += self._ind_closure_pair(s, t, eq.class_id, res)
                    res.checked_count += self._ind_closure_pair(t, s, eq.class_id, res)

        res.elapsed_seconds = time.monotonic() - t0
        return res

    def _ind_closure_pair(
        self, s: str, t: str, class_id: int,
        res: IndependentPhaseResult,
    ) -> int:
        count = 0
        s_succs = self._outgoing.get(s, [])
        for s_prime, tw_s in s_succs:
            count += 1
            s_prime_class = self._state_to_class.get(s_prime)
            if s_prime_class is None:
                res.add_error(
                    f"Successor {s_prime!r} of {s!r} has no class"
                )
                continue
            # Check t can reach s_prime_class
            t_succ_classes = {
                self._state_to_class.get(succ)
                for succ, _ in self._outgoing.get(t, [])
            }
            if s_prime_class not in t_succ_classes:
                # For stuttering, try BFS
                if self._witness.header.has_stuttering:
                    if self._stutter_reach(t, s_prime_class):
                        continue
                res.add_error(
                    f"Closure: {s!r}->{s_prime!r} (class {s_prime_class}), "
                    f"but {t!r} cannot reach class {s_prime_class}"
                )
        return count

    def _stutter_reach(self, state: str, target_class: int) -> bool:
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

    # -- Phase: AP preservation ---------------------------------------------

    def check_ap_preservation(self) -> IndependentPhaseResult:
        """Independent AP preservation check."""
        res = IndependentPhaseResult(phase_name="ap_preservation")
        t0 = time.monotonic()

        for eq in self._witness.equivalences:
            canonical = frozenset(eq.ap_labels)
            members = set(eq.members) | {eq.representative}
            for s in members:
                res.checked_count += 1
                s_class = self._state_to_class.get(s)
                if s_class is None:
                    res.add_error(f"State {s!r} not in any class")
                    continue
                s_aps = self._class_aps.get(s_class, frozenset())
                if s_aps != canonical:
                    res.add_error(
                        f"AP mismatch: {s!r} in class {s_class} has "
                        f"{set(s_aps)} vs canonical {set(canonical)}"
                    )

        res.elapsed_seconds = time.monotonic() - t0
        return res

    # -- Phase: fairness ----------------------------------------------------

    def check_fairness(self) -> IndependentPhaseResult:
        """Independent fairness preservation check."""
        res = IndependentPhaseResult(phase_name="fairness")
        t0 = time.monotonic()

        for fb in self._witness.fairness:
            res.checked_count += 1
            b_set = set(fb.b_set_classes)
            g_set = set(fb.g_set_classes)

            # Class membership consistency
            for eq in self._witness.equivalences:
                members = set(eq.members) | {eq.representative}
                for s in members:
                    res.checked_count += 1
                    s_class = self._state_to_class.get(s)
                    if s_class is not None and s_class != eq.class_id:
                        res.add_error(
                            f"State {s!r} mapped to class {s_class} "
                            f"but declared in class {eq.class_id} "
                            f"(pair {fb.pair_id})"
                        )

            # Rabin overlap
            acceptance = self._witness.metadata.get(
                "acceptance_type", "streett"
            )
            if acceptance == "rabin" and (b_set & g_set):
                res.add_error(
                    f"Rabin pair {fb.pair_id}: B ∩ G overlap"
                )

        res.elapsed_seconds = time.monotonic() - t0
        return res

    # -- Independent hash helpers -------------------------------------------

    @staticmethod
    def _ind_hash_leaf(leaf: bytes) -> bytes:
        return hashlib.sha256(b"\x00" + leaf).digest()

    @staticmethod
    def _ind_hash_internal(left: bytes, right: bytes) -> bytes:
        return hashlib.sha256(b"\x01" + left + right).digest()

    @classmethod
    def _ind_merkle_root(cls, leaves: Sequence[bytes]) -> bytes:
        if not leaves:
            return hashlib.sha256(b"").digest()
        layer = [cls._ind_hash_leaf(l) for l in leaves]
        while len(layer) > 1:
            nxt: List[bytes] = []
            for i in range(0, len(layer), 2):
                left = layer[i]
                right = layer[i + 1] if i + 1 < len(layer) else left
                nxt.append(cls._ind_hash_internal(left, right))
            layer = nxt
        return layer[0]

    @staticmethod
    def _ind_collect_leaves(
        block_index: int,
        eq_idx: Dict[int, EquivalenceBinding],
        tx_src: Dict[int, List[TransitionWitness]],
    ) -> List[bytes]:
        leaves: List[bytes] = []
        eq = eq_idx.get(block_index)
        if eq is not None:
            leaves.append(_ind_serialize_eq(eq))
        for tw in tx_src.get(block_index, []):
            leaves.append(_ind_serialize_tw(tw))
        return leaves


# ---------------------------------------------------------------------------
# Independent serialization (separate from hash_verifier.py)
# ---------------------------------------------------------------------------


def _ind_serialize_eq(eq: EquivalenceBinding) -> bytes:
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


def _ind_serialize_tw(tw: TransitionWitness) -> bytes:
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
