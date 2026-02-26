"""
Hash chain and Merkle proof verification for CoaCert-TLA witnesses.

Verifies the integrity of the hash chain from genesis block to tip,
recomputes block hashes, and validates Merkle proofs for equivalence
classes and transition witnesses.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .deserializer import (
    EquivalenceBinding,
    HashBlock,
    TransitionWitness,
    WitnessData,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

HASH_ALGO = "sha256"
GENESIS_PREV_HASH = b"\x00" * 32


class FailureKind(Enum):
    CHAIN_BREAK = auto()
    HASH_MISMATCH = auto()
    MERKLE_ROOT_MISMATCH = auto()
    MERKLE_PROOF_INVALID = auto()
    GENESIS_INVALID = auto()
    MISSING_BLOCK = auto()
    DUPLICATE_INDEX = auto()


@dataclass(frozen=True)
class HashFailure:
    """Detail record for a single hash verification failure."""
    kind: FailureKind
    block_index: int
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class HashVerificationResult:
    """Aggregated result of hash chain verification."""
    passed: bool = True
    blocks_checked: int = 0
    blocks_total: int = 0
    failures: List[HashFailure] = field(default_factory=list)
    merkle_proofs_checked: int = 0
    merkle_proofs_passed: int = 0
    elapsed_seconds: float = 0.0
    timing_breakdown: Dict[str, float] = field(default_factory=dict)

    def add_failure(self, f: HashFailure) -> None:
        self.passed = False
        self.failures.append(f)

    @property
    def first_failure(self) -> Optional[HashFailure]:
        return self.failures[0] if self.failures else None


# ---------------------------------------------------------------------------
# Merkle tree utilities
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _hash_leaf(leaf: bytes) -> bytes:
    return _sha256(b"\x00" + leaf)


def _hash_internal(left: bytes, right: bytes) -> bytes:
    return _sha256(b"\x01" + left + right)


def compute_merkle_root(leaves: Sequence[bytes]) -> bytes:
    """Compute the Merkle root for a list of leaf data items."""
    if not leaves:
        return _sha256(b"")
    layer: List[bytes] = [_hash_leaf(leaf) for leaf in leaves]
    while len(layer) > 1:
        next_layer: List[bytes] = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            next_layer.append(_hash_internal(left, right))
        layer = next_layer
    return layer[0]


def verify_merkle_proof(leaf_data: bytes, proof: Sequence[bytes],
                        root: bytes, leaf_index: int) -> bool:
    """Verify a Merkle inclusion proof for *leaf_data* against *root*."""
    current = _hash_leaf(leaf_data)
    idx = leaf_index
    for sibling in proof:
        if idx % 2 == 0:
            current = _hash_internal(current, sibling)
        else:
            current = _hash_internal(sibling, current)
        idx //= 2
    return current == root


def _serialize_equivalence(eq: EquivalenceBinding) -> bytes:
    """Canonical byte serialization of an equivalence binding for hashing."""
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


def _serialize_transition(tw: TransitionWitness) -> bytes:
    """Canonical byte serialization of a transition witness for hashing."""
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


# ---------------------------------------------------------------------------
# Hash chain verifier
# ---------------------------------------------------------------------------


class HashChainVerifier:
    """Verifies hash chain integrity and Merkle proofs in a witness."""

    def __init__(self, witness: WitnessData, *,
                 max_workers: int = 4):
        self._witness = witness
        self._max_workers = max_workers
        self._eq_index: Dict[int, EquivalenceBinding] = {
            eq.class_id: eq for eq in witness.equivalences
        }
        self._tx_by_source: Dict[int, List[TransitionWitness]] = {}
        for tw in witness.transitions:
            self._tx_by_source.setdefault(tw.source_class, []).append(tw)

    # -- public API ---------------------------------------------------------

    def verify_full(self) -> HashVerificationResult:
        """Full verification of the hash chain and all Merkle proofs."""
        result = HashVerificationResult(blocks_total=len(self._witness.hash_chain))
        t0 = time.monotonic()

        t_chain = time.monotonic()
        self._verify_chain(result)
        result.timing_breakdown["chain"] = time.monotonic() - t_chain

        t_merkle = time.monotonic()
        self._verify_all_merkle(result)
        result.timing_breakdown["merkle"] = time.monotonic() - t_merkle

        result.elapsed_seconds = time.monotonic() - t0
        return result

    def verify_partial(self, sample_fraction: float = 0.1,
                       seed: int = 42) -> HashVerificationResult:
        """Spot-check a random subset of blocks and proofs."""
        import random
        rng = random.Random(seed)
        chain = self._witness.hash_chain
        result = HashVerificationResult(blocks_total=len(chain))
        t0 = time.monotonic()

        sample_size = max(1, int(len(chain) * sample_fraction))
        sample_indices: Set[int] = set()
        if chain:
            sample_indices.add(0)
            sample_indices.add(len(chain) - 1)
        while len(sample_indices) < min(sample_size, len(chain)):
            sample_indices.add(rng.randint(0, len(chain) - 1))

        sorted_indices = sorted(sample_indices)
        for i in sorted_indices:
            block = chain[i]
            prev_block = chain[i - 1] if i > 0 else None
            self._verify_single_block(block, prev_block, result)
            self._verify_block_merkle(block, result)

        result.elapsed_seconds = time.monotonic() - t0
        return result

    def verify_parallel(self) -> HashVerificationResult:
        """Parallel verification, splitting blocks across threads."""
        chain = self._witness.hash_chain
        result = HashVerificationResult(blocks_total=len(chain))
        t0 = time.monotonic()

        # Chain continuity must be sequential
        self._verify_chain(result)

        # Merkle proofs can be parallelized
        if self._max_workers > 1 and len(chain) > 100:
            self._verify_merkle_parallel(result)
        else:
            self._verify_all_merkle(result)

        result.elapsed_seconds = time.monotonic() - t0
        return result

    # -- chain verification -------------------------------------------------

    def _verify_chain(self, result: HashVerificationResult) -> None:
        chain = self._witness.hash_chain
        if not chain:
            return

        seen_indices: Set[int] = set()
        for i, block in enumerate(chain):
            if block.index in seen_indices:
                result.add_failure(HashFailure(
                    kind=FailureKind.DUPLICATE_INDEX,
                    block_index=block.index,
                    message=f"Duplicate block index {block.index}",
                ))
                continue
            seen_indices.add(block.index)

            prev = chain[i - 1] if i > 0 else None
            self._verify_single_block(block, prev, result)

    def _verify_single_block(self, block: HashBlock,
                             prev: Optional[HashBlock],
                             result: HashVerificationResult) -> None:
        result.blocks_checked += 1

        # Genesis block must reference zero hash
        if prev is None:
            if block.prev_hash != GENESIS_PREV_HASH:
                result.add_failure(HashFailure(
                    kind=FailureKind.GENESIS_INVALID,
                    block_index=block.index,
                    message="Genesis block prev_hash is not the zero hash",
                    expected=GENESIS_PREV_HASH.hex(),
                    actual=block.prev_hash.hex(),
                ))
                return
        else:
            # Chain continuity: prev_hash must equal prior block_hash
            if block.prev_hash != prev.block_hash:
                result.add_failure(HashFailure(
                    kind=FailureKind.CHAIN_BREAK,
                    block_index=block.index,
                    message=(
                        f"Chain break: block {block.index} prev_hash does not "
                        f"match block {prev.index} hash"
                    ),
                    expected=prev.block_hash.hex(),
                    actual=block.prev_hash.hex(),
                ))
                return

        # Recompute block hash: H(prev_hash || payload_hash)
        recomputed = _sha256(block.prev_hash + block.payload_hash)
        if recomputed != block.block_hash:
            result.add_failure(HashFailure(
                kind=FailureKind.HASH_MISMATCH,
                block_index=block.index,
                message=f"Block {block.index} hash mismatch",
                expected=recomputed.hex(),
                actual=block.block_hash.hex(),
            ))

    # -- Merkle verification ------------------------------------------------

    def _verify_all_merkle(self, result: HashVerificationResult) -> None:
        for block in self._witness.hash_chain:
            self._verify_block_merkle(block, result)

    def _verify_block_merkle(self, block: HashBlock,
                             result: HashVerificationResult) -> None:
        if block.merkle_root is None:
            return

        # Recompute the expected Merkle root from witness data
        leaves = self._collect_leaves_for_block(block.index)
        if leaves:
            expected_root = compute_merkle_root(leaves)
            result.merkle_proofs_checked += 1
            if expected_root == block.merkle_root:
                result.merkle_proofs_passed += 1
            else:
                result.add_failure(HashFailure(
                    kind=FailureKind.MERKLE_ROOT_MISMATCH,
                    block_index=block.index,
                    message=f"Merkle root mismatch at block {block.index}",
                    expected=expected_root.hex(),
                    actual=block.merkle_root.hex(),
                ))

        # Verify individual inclusion proofs
        if block.merkle_proof is not None and leaves:
            for leaf_idx, leaf_data in enumerate(leaves):
                if leaf_idx < len(block.merkle_proof):
                    proof_segment = self._extract_proof_path(
                        block.merkle_proof, leaf_idx, len(leaves)
                    )
                    result.merkle_proofs_checked += 1
                    if verify_merkle_proof(leaf_data, proof_segment,
                                           block.merkle_root, leaf_idx):
                        result.merkle_proofs_passed += 1
                    else:
                        result.add_failure(HashFailure(
                            kind=FailureKind.MERKLE_PROOF_INVALID,
                            block_index=block.index,
                            message=(
                                f"Merkle proof invalid for leaf {leaf_idx} "
                                f"in block {block.index}"
                            ),
                        ))

    def _collect_leaves_for_block(self, block_index: int) -> List[bytes]:
        """Collect serialized leaves that belong to a given block index.

        Mapping strategy: block *i* contains equivalence class *i* (if it
        exists) and all transition witnesses whose source_class == i.
        """
        leaves: List[bytes] = []
        eq = self._eq_index.get(block_index)
        if eq is not None:
            leaves.append(_serialize_equivalence(eq))
        for tw in self._tx_by_source.get(block_index, []):
            leaves.append(_serialize_transition(tw))
        return leaves

    def _extract_proof_path(self, full_proof: Tuple[bytes, ...],
                            leaf_index: int,
                            num_leaves: int) -> List[bytes]:
        """Extract the proof path for a single leaf from a flattened proof.

        The proof is stored as a concatenated list of sibling hashes for each
        leaf in order.  Each leaf needs ceil(log2(num_leaves)) siblings.
        """
        if num_leaves <= 1:
            return []
        depth = (num_leaves - 1).bit_length()
        start = leaf_index * depth
        end = start + depth
        if end > len(full_proof):
            return list(full_proof[start:])
        return list(full_proof[start:end])

    # -- parallel Merkle verification ---------------------------------------

    def _verify_merkle_parallel(self, result: HashVerificationResult) -> None:
        chain = self._witness.hash_chain
        chunk_size = max(1, len(chain) // self._max_workers)
        chunks: List[List[HashBlock]] = []
        for i in range(0, len(chain), chunk_size):
            chunks.append(chain[i:i + chunk_size])

        partial_results: List[HashVerificationResult] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._verify_chunk_merkle, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                partial_results.append(future.result())

        for pr in partial_results:
            result.merkle_proofs_checked += pr.merkle_proofs_checked
            result.merkle_proofs_passed += pr.merkle_proofs_passed
            for f in pr.failures:
                result.add_failure(f)

    def _verify_chunk_merkle(self,
                             blocks: List[HashBlock]) -> HashVerificationResult:
        chunk_result = HashVerificationResult()
        for block in blocks:
            self._verify_block_merkle(block, chunk_result)
        return chunk_result
