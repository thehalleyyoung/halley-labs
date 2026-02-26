"""
Equivalence class binding for bisimulation witnesses.

Binds each equivalence class to its canonical representative state,
records all member states, and produces a Merkle-tree commitment
over the entire binding.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .merkle_tree import (
    EMPTY_HASH,
    HASH_LEN,
    MerkleProof,
    MerkleTree,
    hash_leaf,
    sha256,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_to_bytes(state: Any) -> bytes:
    if isinstance(state, bytes):
        return state
    if isinstance(state, str):
        return state.encode("utf-8")
    if isinstance(state, int):
        return state.to_bytes(8, "big", signed=True)
    return repr(state).encode("utf-8")


def _hash_state(state: Any) -> bytes:
    return sha256(_state_to_bytes(state))


# ---------------------------------------------------------------------------
# ClassBinding
# ---------------------------------------------------------------------------


@dataclass
class ClassBinding:
    """
    Binding for a single equivalence class.

    Attributes
    ----------
    class_id : str
        Unique identifier for this class.
    representative : Any
        The canonical representative state.
    members : frozenset
        All states belonging to this class (including the representative).
    """

    class_id: str
    representative: Any
    members: FrozenSet[Any]

    # -- derived hashes (lazily cached) -------------------------------------

    _rep_hash: Optional[bytes] = field(default=None, repr=False, compare=False)
    _members_hash: Optional[bytes] = field(default=None, repr=False, compare=False)
    _binding_hash: Optional[bytes] = field(default=None, repr=False, compare=False)

    @property
    def representative_hash(self) -> bytes:
        if self._rep_hash is None:
            self._rep_hash = _hash_state(self.representative)
        return self._rep_hash

    @property
    def sorted_member_hashes(self) -> List[bytes]:
        hashes = [_hash_state(m) for m in self.members]
        hashes.sort()
        return hashes

    @property
    def members_hash(self) -> bytes:
        """SHA-256 over the sorted concatenation of all member hashes."""
        if self._members_hash is None:
            h = hashlib.sha256()
            for mh in self.sorted_member_hashes:
                h.update(mh)
            self._members_hash = h.digest()
        return self._members_hash

    @property
    def binding_hash(self) -> bytes:
        """Overall hash for this class: H(class_id ‖ rep_hash ‖ members_hash)."""
        if self._binding_hash is None:
            h = hashlib.sha256()
            h.update(self.class_id.encode("utf-8"))
            h.update(self.representative_hash)
            h.update(self.members_hash)
            self._binding_hash = h.digest()
        return self._binding_hash

    @property
    def member_count(self) -> int:
        return len(self.members)

    # -- compact representation ---------------------------------------------

    def compact_bytes(self) -> bytes:
        """class_id (len-prefixed) + representative hash + member count."""
        cid = self.class_id.encode("utf-8")
        return (
            struct.pack(">H", len(cid))
            + cid
            + self.representative_hash
            + struct.pack(">I", self.member_count)
        )

    # -- full representation ------------------------------------------------

    def full_bytes(self) -> bytes:
        """class_id + representative + every member as state bytes."""
        parts: list[bytes] = []
        cid = self.class_id.encode("utf-8")
        parts.append(struct.pack(">H", len(cid)))
        parts.append(cid)
        rep_b = _state_to_bytes(self.representative)
        parts.append(struct.pack(">I", len(rep_b)))
        parts.append(rep_b)
        parts.append(struct.pack(">I", self.member_count))
        for mh in self.sorted_member_hashes:
            parts.append(mh)
        return b"".join(parts)

    @classmethod
    def from_full_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["ClassBinding", int]:
        pos = offset
        (cid_len,) = struct.unpack_from(">H", buf, pos)
        pos += 2
        class_id = buf[pos : pos + cid_len].decode("utf-8")
        pos += cid_len
        (rep_len,) = struct.unpack_from(">I", buf, pos)
        pos += 4
        representative = buf[pos : pos + rep_len]
        pos += rep_len
        (count,) = struct.unpack_from(">I", buf, pos)
        pos += 4
        member_hashes: list[bytes] = []
        for _ in range(count):
            member_hashes.append(buf[pos : pos + HASH_LEN])
            pos += HASH_LEN
        binding = cls.__new__(cls)
        binding.class_id = class_id
        binding.representative = representative
        binding.members = frozenset()  # hashes only in this path
        binding._rep_hash = None
        binding._members_hash = None
        binding._binding_hash = None
        return binding, pos

    def __repr__(self) -> str:
        return (
            f"ClassBinding(id={self.class_id!r}, rep={self.representative!r}, "
            f"members={self.member_count})"
        )


# ---------------------------------------------------------------------------
# EquivalenceBinding
# ---------------------------------------------------------------------------


class EquivalenceBinding:
    """
    Top-level binding of all equivalence classes.

    Builds a class-level Merkle tree whose leaves are the individual
    class binding hashes, enabling compact proofs of class membership.
    """

    def __init__(self) -> None:
        self._classes: Dict[str, ClassBinding] = {}
        self._state_to_class: Dict[Any, str] = {}
        self._tree: Optional[MerkleTree] = None
        self._dirty: bool = True

    # -- construction -------------------------------------------------------

    def add_class(
        self,
        class_id: str,
        representative: Any,
        members: Sequence[Any],
    ) -> ClassBinding:
        """Register an equivalence class."""
        member_set = frozenset(members)
        if representative not in member_set:
            member_set = member_set | {representative}
        binding = ClassBinding(
            class_id=class_id,
            representative=representative,
            members=member_set,
        )
        self._classes[class_id] = binding
        for m in member_set:
            self._state_to_class[m] = class_id
        self._dirty = True
        return binding

    def remove_class(self, class_id: str) -> None:
        """Remove an equivalence class."""
        if class_id not in self._classes:
            raise KeyError(f"Unknown class: {class_id}")
        binding = self._classes.pop(class_id)
        for m in binding.members:
            if self._state_to_class.get(m) == class_id:
                del self._state_to_class[m]
        self._dirty = True

    def _rebuild_tree(self) -> None:
        if not self._dirty:
            return
        binding_hashes = [
            self._classes[cid].binding_hash
            for cid in sorted(self._classes)
        ]
        self._tree = MerkleTree(binding_hashes)
        self._dirty = False

    # -- queries ------------------------------------------------------------

    def class_for_state(self, state: Any) -> Optional[str]:
        return self._state_to_class.get(state)

    def get_class(self, class_id: str) -> ClassBinding:
        return self._classes[class_id]

    @property
    def class_ids(self) -> List[str]:
        return sorted(self._classes)

    @property
    def class_count(self) -> int:
        return len(self._classes)

    @property
    def total_states(self) -> int:
        return sum(b.member_count for b in self._classes.values())

    # -- Merkle root --------------------------------------------------------

    @property
    def root(self) -> bytes:
        self._rebuild_tree()
        assert self._tree is not None
        return self._tree.root

    def proof_for_class(self, class_id: str) -> MerkleProof:
        """Return an inclusion proof for a specific class."""
        self._rebuild_tree()
        assert self._tree is not None
        binding = self._classes[class_id]
        return self._tree.proof(binding.binding_hash)

    # -- verification -------------------------------------------------------

    def verify_partition(self) -> Tuple[bool, List[str]]:
        """
        Verify that every state belongs to exactly one class.

        Returns ``(ok, errors)`` where *errors* lists any violations.
        """
        errors: List[str] = []
        seen: Dict[Any, str] = {}
        for cid in sorted(self._classes):
            binding = self._classes[cid]
            if binding.representative not in binding.members:
                errors.append(
                    f"Class {cid}: representative {binding.representative!r} "
                    f"not in members"
                )
            for m in binding.members:
                if m in seen:
                    errors.append(
                        f"State {m!r} appears in both class {seen[m]!r} "
                        f"and class {cid!r}"
                    )
                else:
                    seen[m] = cid
        return (len(errors) == 0, errors)

    def verify_binding_hashes(self) -> bool:
        """Recompute every class binding hash and compare."""
        for cid in sorted(self._classes):
            binding = self._classes[cid]
            expected = binding.binding_hash
            # force recompute
            binding._binding_hash = None
            binding._rep_hash = None
            binding._members_hash = None
            if binding.binding_hash != expected:
                return False
        return True

    # -- serialization ------------------------------------------------------

    def to_bytes(self, compact: bool = False) -> bytes:
        """
        Serialize the entire binding.

        If *compact* is True, store only class ID + representative hash +
        member count.  Otherwise store all member hashes.
        """
        self._rebuild_tree()
        parts: list[bytes] = []
        parts.append(struct.pack(">I", self.class_count))
        for cid in sorted(self._classes):
            binding = self._classes[cid]
            if compact:
                data = binding.compact_bytes()
            else:
                data = binding.full_bytes()
            parts.append(struct.pack(">I", len(data)))
            parts.append(data)
        assert self._tree is not None
        parts.append(self._tree.root)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> "EquivalenceBinding":
        """Deserialize from full-representation bytes."""
        eb = cls()
        pos = offset
        (count,) = struct.unpack_from(">I", buf, pos)
        pos += 4
        for _ in range(count):
            (data_len,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            binding, _ = ClassBinding.from_full_bytes(buf, pos)
            eb._classes[binding.class_id] = binding
            pos += data_len
        # skip stored root hash
        pos += HASH_LEN
        eb._dirty = True
        return eb

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation."""
        self._rebuild_tree()
        return {
            "class_count": self.class_count,
            "total_states": self.total_states,
            "root": self.root.hex(),
            "classes": {
                cid: {
                    "representative": repr(b.representative),
                    "member_count": b.member_count,
                    "binding_hash": b.binding_hash.hex(),
                }
                for cid, b in sorted(self._classes.items())
            },
        }

    def __repr__(self) -> str:
        return (
            f"EquivalenceBinding(classes={self.class_count}, "
            f"states={self.total_states}, "
            f"root={self.root.hex()[:16]}…)"
        )

    def __contains__(self, state: Any) -> bool:
        return state in self._state_to_class

    def __len__(self) -> int:
        return self.class_count
