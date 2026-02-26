"""
Compact witness representation for large state spaces.

Provides space-optimized encodings including delta encoding, variable-
length integers, Bloom filters for membership testing, and truncated
hashes, with full decompression back to standard witnesses.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .equivalence_binding import EquivalenceBinding
from .hash_chain import HashChain
from .merkle_tree import EMPTY_HASH, HASH_LEN, sha256
from .transition_witness import WitnessSet

# ---------------------------------------------------------------------------
# Variable-length integer encoding (LEB128 unsigned)
# ---------------------------------------------------------------------------


def encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as an unsigned LEB128 varint."""
    if value < 0:
        raise ValueError("varint must be non-negative")
    parts: list[int] = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            parts.append(byte | 0x80)
        else:
            parts.append(byte)
            break
    return bytes(parts)


def decode_varint(buf: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode an unsigned LEB128 varint.  Returns ``(value, new_offset)``."""
    result = 0
    shift = 0
    pos = offset
    while True:
        byte = buf[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, pos


def encode_signed_varint(value: int) -> bytes:
    """ZigZag + LEB128 encoding for signed integers."""
    zigzag = (value << 1) ^ (value >> 63)
    return encode_varint(zigzag & 0xFFFFFFFFFFFFFFFF)


def decode_signed_varint(buf: bytes, offset: int = 0) -> Tuple[int, int]:
    zigzag, pos = decode_varint(buf, offset)
    value = (zigzag >> 1) ^ -(zigzag & 1)
    return value, pos


# ---------------------------------------------------------------------------
# Delta encoding
# ---------------------------------------------------------------------------


def delta_encode(values: Sequence[int]) -> List[int]:
    """Delta-encode a sorted sequence of integers."""
    if not values:
        return []
    deltas = [values[0]]
    for i in range(1, len(values)):
        deltas.append(values[i] - values[i - 1])
    return deltas


def delta_decode(deltas: Sequence[int]) -> List[int]:
    """Reconstruct a sorted integer list from its delta encoding."""
    if not deltas:
        return []
    values = [deltas[0]]
    for i in range(1, len(deltas)):
        values.append(values[-1] + deltas[i])
    return values


def encode_delta_bytes(values: Sequence[int]) -> bytes:
    """Delta-encode and serialize to varints."""
    deltas = delta_encode(sorted(values))
    parts = [encode_varint(len(deltas))]
    for d in deltas:
        parts.append(encode_signed_varint(d))
    return b"".join(parts)


def decode_delta_bytes(buf: bytes, offset: int = 0) -> Tuple[List[int], int]:
    count, pos = decode_varint(buf, offset)
    deltas: list[int] = []
    for _ in range(count):
        d, pos = decode_signed_varint(buf, pos)
        deltas.append(d)
    return delta_decode(deltas), pos


# ---------------------------------------------------------------------------
# Bloom filter
# ---------------------------------------------------------------------------


class BloomFilter:
    """
    A probabilistic set membership structure.

    Uses *k* independent hash functions simulated by double-hashing over
    SHA-256 to achieve a target false-positive rate.
    """

    def __init__(
        self,
        expected_items: int = 1000,
        fp_rate: float = 0.01,
    ) -> None:
        if expected_items < 1:
            expected_items = 1
        if fp_rate <= 0 or fp_rate >= 1:
            raise ValueError("fp_rate must be in (0, 1)")
        self._n = expected_items
        self._fp = fp_rate
        self._m = self._optimal_bits(expected_items, fp_rate)
        self._k = self._optimal_hashes(self._m, expected_items)
        self._bits = bytearray((self._m + 7) // 8)
        self._count = 0

    @staticmethod
    def _optimal_bits(n: int, p: float) -> int:
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return max(8, int(math.ceil(m)))

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        k = (m / max(n, 1)) * math.log(2)
        return max(1, int(round(k)))

    def _hash_indices(self, item: bytes) -> List[int]:
        h = hashlib.sha256(item).digest()
        h1 = int.from_bytes(h[:8], "big")
        h2 = int.from_bytes(h[8:16], "big")
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, item: bytes) -> None:
        for idx in self._hash_indices(item):
            self._bits[idx // 8] |= 1 << (idx % 8)
        self._count += 1

    def __contains__(self, item: bytes) -> bool:
        for idx in self._hash_indices(item):
            if not (self._bits[idx // 8] & (1 << (idx % 8))):
                return False
        return True

    def maybe_contains(self, item: bytes) -> bool:
        return item in self

    @property
    def count(self) -> int:
        return self._count

    @property
    def bit_count(self) -> int:
        return self._m

    @property
    def hash_count(self) -> int:
        return self._k

    @property
    def size_bytes(self) -> int:
        return len(self._bits)

    @property
    def estimated_fp_rate(self) -> float:
        """Current estimated false-positive rate given items inserted."""
        if self._count == 0:
            return 0.0
        exponent = -self._k * self._count / self._m
        return (1 - math.exp(exponent)) ** self._k

    def to_bytes(self) -> bytes:
        header = struct.pack(">IIIf", self._m, self._k, self._count, self._fp)
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["BloomFilter", int]:
        m, k, count, fp = struct.unpack_from(">IIIf", buf, offset)
        pos = offset + 16
        bits_len = (m + 7) // 8
        bf = cls.__new__(cls)
        bf._m = m
        bf._k = k
        bf._n = count
        bf._fp = fp
        bf._count = count
        bf._bits = bytearray(buf[pos : pos + bits_len])
        return bf, pos + bits_len

    def __repr__(self) -> str:
        return (
            f"BloomFilter(items={self._count}, bits={self._m}, "
            f"hashes={self._k}, fp≈{self.estimated_fp_rate:.6f})"
        )

    def __len__(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Truncated hashes
# ---------------------------------------------------------------------------


def truncate_hash(h: bytes, length: int = 16) -> bytes:
    """Return the first *length* bytes of a hash digest."""
    if length < 4 or length > HASH_LEN:
        raise ValueError(f"Truncation length must be in [4, {HASH_LEN}]")
    return h[:length]


def truncated_hash_collision_probability(n: int, length: int = 16) -> float:
    """
    Estimated probability of at least one collision among *n* items
    with *length*-byte truncated hashes (birthday approximation).
    """
    bits = length * 8
    if n <= 1:
        return 0.0
    exponent = -(n * (n - 1)) / (2 * (2 ** bits))
    return 1 - math.exp(exponent)


# ---------------------------------------------------------------------------
# CompactWitness
# ---------------------------------------------------------------------------


class CompactWitness:
    """
    Space-optimized witness for large state spaces.

    Combines delta encoding, varint serialization, Bloom filters, and
    optional hash truncation to minimize certificate size.
    """

    def __init__(
        self,
        equivalence: EquivalenceBinding,
        witnesses: WitnessSet,
        chain: HashChain,
        hash_truncation: int = HASH_LEN,
    ) -> None:
        self.equivalence = equivalence
        self.witnesses = witnesses
        self.chain = chain
        self.hash_truncation = hash_truncation
        self._bloom: Optional[BloomFilter] = None

    # -- Bloom filter -------------------------------------------------------

    def build_bloom_filter(self, fp_rate: float = 0.01) -> BloomFilter:
        """Build a Bloom filter over all state hashes for quick membership tests."""
        total = self.equivalence.total_states
        bf = BloomFilter(expected_items=max(total, 1), fp_rate=fp_rate)
        for cid in self.equivalence.class_ids:
            binding = self.equivalence.get_class(cid)
            for mh in binding.sorted_member_hashes:
                bf.add(mh)
        self._bloom = bf
        return bf

    @property
    def bloom_filter(self) -> Optional[BloomFilter]:
        return self._bloom

    # -- delta-encoded class sizes ------------------------------------------

    def _encode_class_sizes(self) -> bytes:
        """Delta-encode the sorted list of class member counts."""
        sizes = sorted(
            self.equivalence.get_class(cid).member_count
            for cid in self.equivalence.class_ids
        )
        return encode_delta_bytes(sizes)

    def _decode_class_sizes(self, buf: bytes, offset: int = 0) -> Tuple[List[int], int]:
        return decode_delta_bytes(buf, offset)

    # -- compact hash representation ----------------------------------------

    def _compact_hash(self, h: bytes) -> bytes:
        return truncate_hash(h, self.hash_truncation)

    def _encode_compact_binding(self) -> bytes:
        """
        Compact equivalence binding:
        - varint class_count
        - for each class:  varint(class_id_len) + class_id + truncated_rep_hash + varint(member_count)
        """
        parts: list[bytes] = []
        class_ids = self.equivalence.class_ids
        parts.append(encode_varint(len(class_ids)))
        for cid in class_ids:
            binding = self.equivalence.get_class(cid)
            cid_b = cid.encode("utf-8")
            parts.append(encode_varint(len(cid_b)))
            parts.append(cid_b)
            parts.append(self._compact_hash(binding.representative_hash))
            parts.append(encode_varint(binding.member_count))
        return b"".join(parts)

    def _encode_compact_transitions(self) -> bytes:
        """
        Compact transition witnesses:
        - varint count
        - for each: varint(src_len) + src + varint(tgt_len) + tgt
                   + varint(action_len) + action + truncated_digest
        """
        parts: list[bytes] = []
        transitions = self.witnesses.transitions
        parts.append(encode_varint(len(transitions)))
        for tw in transitions:
            for s in (tw.source_class, tw.target_class):
                sb = s.encode("utf-8")
                parts.append(encode_varint(len(sb)))
                parts.append(sb)
            action_b = tw.action if isinstance(tw.action, bytes) else repr(tw.action).encode("utf-8")
            parts.append(encode_varint(len(action_b)))
            parts.append(action_b)
            parts.append(self._compact_hash(tw.digest))
        return b"".join(parts)

    def _encode_compact_chain_summary(self) -> bytes:
        """
        Instead of the full chain, store only block hashes.
        - varint block_count
        - for each block: truncated block hash
        """
        parts: list[bytes] = []
        blocks = self.chain.blocks
        parts.append(encode_varint(len(blocks)))
        for block in blocks:
            parts.append(self._compact_hash(block.hash))
        return b"".join(parts)

    # -- full compact serialization -----------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize the compact witness."""
        parts: list[bytes] = []

        # Header: truncation length
        parts.append(struct.pack(">B", self.hash_truncation))

        # Section 1: compact binding
        binding_data = self._encode_compact_binding()
        parts.append(encode_varint(len(binding_data)))
        parts.append(binding_data)

        # Section 2: compact transitions
        trans_data = self._encode_compact_transitions()
        parts.append(encode_varint(len(trans_data)))
        parts.append(trans_data)

        # Section 3: class sizes (delta-encoded)
        sizes_data = self._encode_class_sizes()
        parts.append(encode_varint(len(sizes_data)))
        parts.append(sizes_data)

        # Section 4: chain summary
        chain_data = self._encode_compact_chain_summary()
        parts.append(encode_varint(len(chain_data)))
        parts.append(chain_data)

        # Section 5: Bloom filter (if built)
        if self._bloom is not None:
            bloom_data = self._bloom.to_bytes()
            parts.append(encode_varint(len(bloom_data)))
            parts.append(bloom_data)
        else:
            parts.append(encode_varint(0))

        return b"".join(parts)

    @classmethod
    def from_bytes(
        cls,
        buf: bytes,
        equivalence: EquivalenceBinding,
        witnesses: WitnessSet,
        chain: HashChain,
        offset: int = 0,
    ) -> Tuple["CompactWitness", int]:
        """Deserialize a compact witness (requires full components for reconstruction)."""
        pos = offset
        trunc_len = buf[pos]
        pos += 1

        # Skip binding section
        binding_len, pos = decode_varint(buf, pos)
        pos += binding_len

        # Skip transitions section
        trans_len, pos = decode_varint(buf, pos)
        pos += trans_len

        # Skip sizes section
        sizes_len, pos = decode_varint(buf, pos)
        pos += sizes_len

        # Skip chain summary
        chain_len, pos = decode_varint(buf, pos)
        pos += chain_len

        # Bloom filter
        bloom_len, pos = decode_varint(buf, pos)
        bloom: Optional[BloomFilter] = None
        if bloom_len > 0:
            bloom, _ = BloomFilter.from_bytes(buf, pos)
            pos += bloom_len

        cw = cls(
            equivalence=equivalence,
            witnesses=witnesses,
            chain=chain,
            hash_truncation=trunc_len,
        )
        cw._bloom = bloom
        return cw, pos

    # -- decompression to full witness --------------------------------------

    def to_full_witness(self) -> Tuple[EquivalenceBinding, WitnessSet, HashChain]:
        """Return the full (non-compact) witness components."""
        return self.equivalence, self.witnesses, self.chain

    # -- size analysis ------------------------------------------------------

    @property
    def compact_size(self) -> int:
        return len(self.to_bytes())

    @property
    def full_size(self) -> int:
        eq = len(self.equivalence.to_bytes())
        tw = len(self.witnesses.to_bytes())
        ch = len(self.chain.to_bytes())
        return eq + tw + ch

    @property
    def compression_ratio(self) -> float:
        full = self.full_size
        if full == 0:
            return 1.0
        return self.compact_size / full

    @property
    def space_savings(self) -> float:
        return 1.0 - self.compression_ratio

    def size_analysis(self) -> Dict[str, Any]:
        """Detailed space analysis with recommendations."""
        compact = self.compact_size
        full = self.full_size
        bloom_size = self._bloom.size_bytes if self._bloom else 0

        analysis: Dict[str, Any] = {
            "compact_size_bytes": compact,
            "full_size_bytes": full,
            "compression_ratio": self.compression_ratio,
            "space_savings_pct": self.space_savings * 100,
            "bloom_filter_bytes": bloom_size,
            "hash_truncation_bytes": self.hash_truncation,
            "collision_probability": truncated_hash_collision_probability(
                self.equivalence.total_states,
                self.hash_truncation,
            ),
        }

        # Recommendations
        recs: list[str] = []
        if self.hash_truncation == HASH_LEN and self.equivalence.total_states < 2**32:
            recs.append(
                f"Consider truncating hashes to 16 bytes "
                f"(collision prob ≈ {truncated_hash_collision_probability(self.equivalence.total_states, 16):.2e})"
            )
        if self._bloom is None and self.equivalence.total_states > 100:
            recs.append("Build a Bloom filter for fast membership queries")
        if self.compression_ratio > 0.8:
            recs.append("Compact encoding provides minimal savings; consider gzip on the full witness")

        analysis["recommendations"] = recs
        return analysis

    def __repr__(self) -> str:
        return (
            f"CompactWitness(compact={self.compact_size}B, "
            f"full={self.full_size}B, "
            f"ratio={self.compression_ratio:.2%})"
        )
