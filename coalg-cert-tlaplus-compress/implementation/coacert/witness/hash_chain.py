"""
Hash chain integrity for bisimulation witness certificates.

Implements a blockchain-style hash chain where each block references
the previous block's hash, providing tamper-evident sequencing of
equivalence bindings, transition witnesses, and fairness data.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .merkle_tree import EMPTY_HASH, HASH_LEN, MerkleTree, sha256

# ---------------------------------------------------------------------------
# Block types
# ---------------------------------------------------------------------------

GENESIS_PREV_HASH = b"\x00" * HASH_LEN


class BlockType(IntEnum):
    GENESIS = 0x00
    EQUIVALENCE = 0x01
    TRANSITION = 0x02
    FAIRNESS = 0x03


@dataclass
class _BlockBase:
    """Common fields shared by all block types."""

    index: int
    prev_hash: bytes
    timestamp: float
    payload: bytes
    block_type: BlockType

    _hash: Optional[bytes] = field(default=None, repr=False, compare=False)

    @property
    def hash(self) -> bytes:
        if self._hash is None:
            self._hash = self._compute_hash()
        return self._hash

    def _compute_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(struct.pack(">I", self.index))
        h.update(self.prev_hash)
        h.update(struct.pack(">d", self.timestamp))
        h.update(struct.pack(">B", int(self.block_type)))
        h.update(struct.pack(">I", len(self.payload)))
        h.update(self.payload)
        return h.digest()

    def header_bytes(self) -> bytes:
        return (
            struct.pack(">I", self.index)
            + self.prev_hash
            + struct.pack(">dBI", self.timestamp, int(self.block_type), len(self.payload))
        )

    def to_bytes(self) -> bytes:
        return self.header_bytes() + self.payload + self.hash

    @classmethod
    def _parse_header(cls, buf: bytes, offset: int) -> Tuple[int, bytes, float, int, int, int]:
        pos = offset
        (index,) = struct.unpack_from(">I", buf, pos)
        pos += 4
        prev_hash = buf[pos : pos + HASH_LEN]
        pos += HASH_LEN
        timestamp, btype, payload_len = struct.unpack_from(">dBI", buf, pos)
        pos += 13  # 8 + 1 + 4
        return index, prev_hash, timestamp, btype, payload_len, pos

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "block_type": self.block_type.name,
            "prev_hash": self.prev_hash.hex(),
            "hash": self.hash.hex(),
            "timestamp": self.timestamp,
            "payload_size": len(self.payload),
        }


# ---------------------------------------------------------------------------
# Concrete block types
# ---------------------------------------------------------------------------


class EquivalenceBlock(_BlockBase):
    """Block containing equivalence binding data."""

    def __init__(
        self,
        index: int,
        prev_hash: bytes,
        payload: bytes,
        timestamp: Optional[float] = None,
    ) -> None:
        super().__init__(
            index=index,
            prev_hash=prev_hash,
            timestamp=timestamp if timestamp is not None else time.time(),
            payload=payload,
            block_type=BlockType.EQUIVALENCE,
        )

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["EquivalenceBlock", int]:
        index, prev_hash, ts, btype, plen, pos = cls._parse_header(buf, offset)
        assert btype == BlockType.EQUIVALENCE
        payload = buf[pos : pos + plen]
        pos += plen
        stored_hash = buf[pos : pos + HASH_LEN]
        pos += HASH_LEN
        block = cls(index=index, prev_hash=prev_hash, payload=payload, timestamp=ts)
        block._hash = stored_hash
        return block, pos


class TransitionBlock(_BlockBase):
    """Block containing transition witness data."""

    def __init__(
        self,
        index: int,
        prev_hash: bytes,
        payload: bytes,
        timestamp: Optional[float] = None,
    ) -> None:
        super().__init__(
            index=index,
            prev_hash=prev_hash,
            timestamp=timestamp if timestamp is not None else time.time(),
            payload=payload,
            block_type=BlockType.TRANSITION,
        )

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["TransitionBlock", int]:
        index, prev_hash, ts, btype, plen, pos = cls._parse_header(buf, offset)
        assert btype == BlockType.TRANSITION
        payload = buf[pos : pos + plen]
        pos += plen
        stored_hash = buf[pos : pos + HASH_LEN]
        pos += HASH_LEN
        block = cls(index=index, prev_hash=prev_hash, payload=payload, timestamp=ts)
        block._hash = stored_hash
        return block, pos


class FairnessBlock(_BlockBase):
    """Block containing fairness witness data."""

    def __init__(
        self,
        index: int,
        prev_hash: bytes,
        payload: bytes,
        timestamp: Optional[float] = None,
    ) -> None:
        super().__init__(
            index=index,
            prev_hash=prev_hash,
            timestamp=timestamp if timestamp is not None else time.time(),
            payload=payload,
            block_type=BlockType.FAIRNESS,
        )

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["FairnessBlock", int]:
        index, prev_hash, ts, btype, plen, pos = cls._parse_header(buf, offset)
        assert btype == BlockType.FAIRNESS
        payload = buf[pos : pos + plen]
        pos += plen
        stored_hash = buf[pos : pos + HASH_LEN]
        pos += HASH_LEN
        block = cls(index=index, prev_hash=prev_hash, payload=payload, timestamp=ts)
        block._hash = stored_hash
        return block, pos


_BLOCK_PARSERS = {
    BlockType.EQUIVALENCE: EquivalenceBlock.from_bytes,
    BlockType.TRANSITION: TransitionBlock.from_bytes,
    BlockType.FAIRNESS: FairnessBlock.from_bytes,
}


def _block_from_bytes(buf: bytes, offset: int) -> Tuple[_BlockBase, int]:
    """Dispatch block deserialization by peeking at the block-type byte."""
    # block type is at offset + 4 (index) + 32 (prev_hash) + 8 (timestamp) = +44
    btype_byte = buf[offset + 44]
    btype = BlockType(btype_byte)
    parser = _BLOCK_PARSERS.get(btype)
    if parser is None:
        raise ValueError(f"Unknown block type {btype_byte:#x}")
    return parser(buf, offset)


# ---------------------------------------------------------------------------
# HashChain
# ---------------------------------------------------------------------------


class HashChain:
    """
    A tamper-evident hash chain over bisimulation witness blocks.

    Each block stores a payload and references the hash of the previous
    block, forming a chain from a genesis block to the tip.
    """

    def __init__(self) -> None:
        self._blocks: List[_BlockBase] = []
        self._tree: Optional[MerkleTree] = None
        self._dirty: bool = True

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        equivalence_payloads: Sequence[bytes],
        transition_payloads: Sequence[bytes],
        fairness_payloads: Sequence[bytes],
    ) -> "HashChain":
        """
        Build a chain from pre-serialized payloads.

        Order: equivalence blocks → transition blocks → fairness blocks.
        """
        chain = cls()
        prev = GENESIS_PREV_HASH

        for payload in equivalence_payloads:
            block = EquivalenceBlock(
                index=len(chain._blocks),
                prev_hash=prev,
                payload=payload,
            )
            chain._blocks.append(block)
            prev = block.hash

        for payload in transition_payloads:
            block = TransitionBlock(
                index=len(chain._blocks),
                prev_hash=prev,
                payload=payload,
            )
            chain._blocks.append(block)
            prev = block.hash

        for payload in fairness_payloads:
            block = FairnessBlock(
                index=len(chain._blocks),
                prev_hash=prev,
                payload=payload,
            )
            chain._blocks.append(block)
            prev = block.hash

        chain._dirty = True
        return chain

    def append_block(self, block_type: BlockType, payload: bytes) -> _BlockBase:
        """Append a single block to the chain."""
        prev = self._blocks[-1].hash if self._blocks else GENESIS_PREV_HASH
        constructors = {
            BlockType.EQUIVALENCE: EquivalenceBlock,
            BlockType.TRANSITION: TransitionBlock,
            BlockType.FAIRNESS: FairnessBlock,
        }
        ctor = constructors[block_type]
        block = ctor(index=len(self._blocks), prev_hash=prev, payload=payload)
        self._blocks.append(block)
        self._dirty = True
        return block

    # -- accessors ----------------------------------------------------------

    @property
    def blocks(self) -> List[_BlockBase]:
        return list(self._blocks)

    @property
    def length(self) -> int:
        return len(self._blocks)

    @property
    def tip(self) -> Optional[_BlockBase]:
        return self._blocks[-1] if self._blocks else None

    @property
    def tip_hash(self) -> bytes:
        return self._blocks[-1].hash if self._blocks else GENESIS_PREV_HASH

    @property
    def genesis(self) -> Optional[_BlockBase]:
        return self._blocks[0] if self._blocks else None

    def block_at(self, index: int) -> _BlockBase:
        return self._blocks[index]

    # -- verification -------------------------------------------------------

    def verify(self) -> Tuple[bool, List[str]]:
        """
        Verify the entire chain from genesis to tip.

        Checks:
        1. Each block's stored hash matches its recomputed hash.
        2. Each block's ``prev_hash`` matches the previous block's hash.
        3. Block indices are sequential.
        """
        errors: List[str] = []
        if not self._blocks:
            return True, []

        for i, block in enumerate(self._blocks):
            if block.index != i:
                errors.append(
                    f"Block {i}: expected index {i}, got {block.index}"
                )

            recomputed = block._compute_hash()
            if block.hash != recomputed:
                errors.append(
                    f"Block {i}: hash mismatch "
                    f"(stored {block.hash.hex()[:16]}…, "
                    f"recomputed {recomputed.hex()[:16]}…)"
                )

            expected_prev = (
                self._blocks[i - 1].hash if i > 0 else GENESIS_PREV_HASH
            )
            if block.prev_hash != expected_prev:
                errors.append(
                    f"Block {i}: prev_hash mismatch "
                    f"(stored {block.prev_hash.hex()[:16]}…, "
                    f"expected {expected_prev.hex()[:16]}…)"
                )

        return (len(errors) == 0, errors)

    def verify_range(self, start: int, end: int) -> Tuple[bool, List[str]]:
        """Verify a sub-range of the chain (spot-checking)."""
        errors: List[str] = []
        if start < 0 or end > len(self._blocks) or start >= end:
            errors.append(f"Invalid range [{start}, {end})")
            return False, errors

        for i in range(start, end):
            block = self._blocks[i]
            recomputed = block._compute_hash()
            if block.hash != recomputed:
                errors.append(f"Block {i}: hash mismatch")
            if i > 0:
                if block.prev_hash != self._blocks[i - 1].hash:
                    errors.append(f"Block {i}: prev_hash mismatch")
            elif i == 0:
                if block.prev_hash != GENESIS_PREV_HASH:
                    errors.append(f"Block 0: prev_hash is not genesis hash")

        return (len(errors) == 0, errors)

    def detect_tamper(self) -> Optional[int]:
        """
        Return the index of the first tampered block, or ``None`` if
        the chain is intact.
        """
        for i, block in enumerate(self._blocks):
            if block.hash != block._compute_hash():
                return i
            expected_prev = (
                self._blocks[i - 1].hash if i > 0 else GENESIS_PREV_HASH
            )
            if block.prev_hash != expected_prev:
                return i
        return None

    # -- Merkle root over chain ---------------------------------------------

    def _rebuild_tree(self) -> None:
        if not self._dirty:
            return
        hashes = [b.hash for b in self._blocks]
        self._tree = MerkleTree(hashes) if hashes else MerkleTree()
        self._dirty = False

    @property
    def merkle_root(self) -> bytes:
        self._rebuild_tree()
        assert self._tree is not None
        return self._tree.root

    # -- statistics ---------------------------------------------------------

    @property
    def total_payload_size(self) -> int:
        return sum(len(b.payload) for b in self._blocks)

    @property
    def total_size(self) -> int:
        """Total serialized size in bytes."""
        # header per block: 4 + 32 + 8 + 1 + 4 = 49, plus payload + hash(32)
        return sum(49 + len(b.payload) + HASH_LEN for b in self._blocks) + 4

    def block_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for b in self._blocks:
            name = b.block_type.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    # -- serialization ------------------------------------------------------

    def to_bytes(self) -> bytes:
        parts: list[bytes] = [struct.pack(">I", self.length)]
        for block in self._blocks:
            parts.append(block.to_bytes())
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> "HashChain":
        chain = cls()
        pos = offset
        (count,) = struct.unpack_from(">I", buf, pos)
        pos += 4
        for _ in range(count):
            block, pos = _block_from_bytes(buf, pos)
            chain._blocks.append(block)
        chain._dirty = True
        return chain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "tip_hash": self.tip_hash.hex(),
            "merkle_root": self.merkle_root.hex(),
            "total_payload_size": self.total_payload_size,
            "block_type_counts": self.block_type_counts(),
            "blocks": [b.to_dict() for b in self._blocks],
        }

    def pretty(self) -> str:
        lines = [
            f"HashChain(length={self.length}, "
            f"tip={self.tip_hash.hex()[:16]}…)"
        ]
        for b in self._blocks:
            lines.append(
                f"  [{b.index}] {b.block_type.name:12s} "
                f"hash={b.hash.hex()[:16]}… "
                f"prev={b.prev_hash.hex()[:16]}… "
                f"payload={len(b.payload)}B"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HashChain(length={self.length}, "
            f"tip={self.tip_hash.hex()[:16]}…)"
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> _BlockBase:
        return self._blocks[index]
