"""
Merkle tree construction and proof generation for bisimulation witnesses.

Builds SHA-256-based Merkle trees over equivalence classes, supports
inclusion proof generation/verification, serialization, and incremental
updates.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HASH_LEN = 32  # SHA-256 digest length in bytes
EMPTY_HASH = b"\x00" * HASH_LEN

# Binary format tags
_TAG_LEAF = 0x01
_TAG_INTERNAL = 0x02
_TAG_SPARSE_DEFAULT = 0x03

# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def sha256(data: bytes) -> bytes:
    """Return the SHA-256 digest of *data*."""
    return hashlib.sha256(data).digest()


def hash_pair(left: bytes, right: bytes) -> bytes:
    """Deterministic hash of two child hashes (sorted to ensure canonical order)."""
    if left > right:
        left, right = right, left
    return sha256(left + right)


def hash_leaf(item: bytes) -> bytes:
    """Domain-separated leaf hash to prevent second-preimage attacks."""
    return sha256(b"\x00" + item)


def hash_internal(left: bytes, right: bytes) -> bytes:
    """Domain-separated internal node hash."""
    if left > right:
        left, right = right, left
    return sha256(b"\x01" + left + right)


def _item_to_bytes(item: Any) -> bytes:
    """Coerce an item to a canonical byte representation."""
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode("utf-8")
    if isinstance(item, int):
        return item.to_bytes(8, "big", signed=True)
    return repr(item).encode("utf-8")


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


@dataclass
class LeafNode:
    """A leaf in the Merkle tree, wrapping a single item."""

    item: bytes
    digest: bytes = field(init=False)
    index: int = 0

    def __post_init__(self) -> None:
        self.digest = hash_leaf(self.item)

    @property
    def is_leaf(self) -> bool:
        return True

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        return f"{prefix}Leaf[{self.index}] {self.digest.hex()[:16]}…"

    def to_bytes(self) -> bytes:
        item_len = len(self.item)
        return struct.pack(">BI", _TAG_LEAF, item_len) + self.item

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["LeafNode", int]:
        tag = buf[offset]
        assert tag == _TAG_LEAF, f"Expected leaf tag, got {tag:#x}"
        item_len = struct.unpack_from(">I", buf, offset + 1)[0]
        start = offset + 5
        item = buf[start : start + item_len]
        node = cls(item=item)
        return node, start + item_len


@dataclass
class InternalNode:
    """An internal node in the Merkle tree."""

    left: Any  # LeafNode | InternalNode
    right: Any  # LeafNode | InternalNode | None (for odd-count padding)
    digest: bytes = field(init=False)

    def __post_init__(self) -> None:
        right_digest = self.right.digest if self.right is not None else EMPTY_HASH
        self.digest = hash_internal(self.left.digest, right_digest)

    @property
    def is_leaf(self) -> bool:
        return False

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}Internal {self.digest.hex()[:16]}…"]
        lines.append(self.left.pretty(indent + 1))
        if self.right is not None:
            lines.append(self.right.pretty(indent + 1))
        else:
            lines.append(f"{'  ' * (indent + 1)}(empty)")
        return "\n".join(lines)

    def to_bytes(self) -> bytes:
        left_data = self.left.to_bytes()
        if self.right is not None:
            right_data = self.right.to_bytes()
            has_right = 1
        else:
            right_data = b""
            has_right = 0
        header = struct.pack(">BII", _TAG_INTERNAL, len(left_data), has_right)
        return header + left_data + right_data

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["InternalNode", int]:
        tag = buf[offset]
        assert tag == _TAG_INTERNAL, f"Expected internal tag, got {tag:#x}"
        left_len, has_right = struct.unpack_from(">II", buf, offset + 1)
        left_start = offset + 9
        left_child, left_end = _node_from_bytes(buf, left_start)
        if has_right:
            right_child, right_end = _node_from_bytes(buf, left_end)
        else:
            right_child = None
            right_end = left_end
        node = object.__new__(cls)
        node.left = left_child
        node.right = right_child
        right_digest = right_child.digest if right_child is not None else EMPTY_HASH
        node.digest = hash_internal(left_child.digest, right_digest)
        return node, right_end


def _node_from_bytes(buf: bytes, offset: int) -> Tuple[Any, int]:
    """Dispatch deserialization based on the tag byte."""
    tag = buf[offset]
    if tag == _TAG_LEAF:
        return LeafNode.from_bytes(buf, offset)
    if tag == _TAG_INTERNAL:
        return InternalNode.from_bytes(buf, offset)
    raise ValueError(f"Unknown node tag {tag:#x} at offset {offset}")


# ---------------------------------------------------------------------------
# Merkle proof
# ---------------------------------------------------------------------------


@dataclass
class MerkleProof:
    """
    An inclusion proof for a single leaf.

    The *siblings* list goes from the leaf level upward to the root.
    Each entry is ``(sibling_hash, direction)`` where *direction* is
    ``'L'`` when the sibling is to the left or ``'R'`` when to the right.
    """

    leaf_hash: bytes
    siblings: List[Tuple[bytes, str]]
    root_hash: bytes

    def verify(self) -> bool:
        """Recompute the root from leaf hash and sibling path."""
        current = self.leaf_hash
        for sibling, direction in self.siblings:
            if direction == "L":
                current = hash_internal(sibling, current)
            else:
                current = hash_internal(current, sibling)
        return current == self.root_hash

    def to_bytes(self) -> bytes:
        parts: list[bytes] = []
        parts.append(self.leaf_hash)
        parts.append(struct.pack(">H", len(self.siblings)))
        for sib_hash, direction in self.siblings:
            dir_byte = 0 if direction == "L" else 1
            parts.append(struct.pack(">B", dir_byte))
            parts.append(sib_hash)
        parts.append(self.root_hash)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["MerkleProof", int]:
        leaf_hash = buf[offset : offset + HASH_LEN]
        pos = offset + HASH_LEN
        (sib_count,) = struct.unpack_from(">H", buf, pos)
        pos += 2
        siblings: List[Tuple[bytes, str]] = []
        for _ in range(sib_count):
            dir_byte = buf[pos]
            pos += 1
            sib_hash = buf[pos : pos + HASH_LEN]
            pos += HASH_LEN
            siblings.append((sib_hash, "L" if dir_byte == 0 else "R"))
        root_hash = buf[pos : pos + HASH_LEN]
        pos += HASH_LEN
        return cls(leaf_hash=leaf_hash, siblings=siblings, root_hash=root_hash), pos

    def __repr__(self) -> str:
        return (
            f"MerkleProof(leaf={self.leaf_hash.hex()[:12]}…, "
            f"depth={len(self.siblings)}, "
            f"root={self.root_hash.hex()[:12]}…)"
        )


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------


class MerkleTree:
    """
    A standard binary Merkle tree built over an ordered sequence of items.

    Items are converted to bytes, hashed as leaves, and paired upward
    until a single root remains.
    """

    def __init__(self, items: Sequence[Any] | None = None) -> None:
        self._leaves: List[LeafNode] = []
        self._root: Optional[InternalNode | LeafNode] = None
        self._item_index: Dict[bytes, int] = {}
        if items:
            self.build(items)

    # -- construction -------------------------------------------------------

    def build(self, items: Sequence[Any]) -> None:
        """Build the tree from *items* (deterministically sorted)."""
        raw = [_item_to_bytes(it) for it in items]
        raw.sort()
        self._leaves = []
        self._item_index = {}
        for idx, item_bytes in enumerate(raw):
            leaf = LeafNode(item=item_bytes, index=idx)
            # __post_init__ already ran; just fix the index
            object.__setattr__(leaf, "index", idx)
            self._leaves.append(leaf)
            self._item_index[item_bytes] = idx
        self._root = self._build_layer(list(self._leaves))

    def _build_layer(self, nodes: list) -> Any:
        """Recursively pair nodes upward until a single root remains."""
        if len(nodes) == 0:
            leaf = LeafNode(item=b"")
            return leaf
        if len(nodes) == 1:
            return nodes[0]
        next_layer: list = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else None
            next_layer.append(InternalNode(left=left, right=right))
        return self._build_layer(next_layer)

    # -- root ---------------------------------------------------------------

    @property
    def root(self) -> bytes:
        """Return the Merkle root hash."""
        if self._root is None:
            return EMPTY_HASH
        return self._root.digest

    @property
    def root_node(self) -> Any:
        return self._root

    # -- proof generation ---------------------------------------------------

    def proof(self, item: Any) -> MerkleProof:
        """Generate an inclusion proof for *item*."""
        item_bytes = _item_to_bytes(item)
        if item_bytes not in self._item_index:
            raise KeyError(f"Item not in tree: {item!r}")
        idx = self._item_index[item_bytes]
        leaf = self._leaves[idx]
        siblings = self._collect_siblings(idx)
        return MerkleProof(
            leaf_hash=leaf.digest,
            siblings=siblings,
            root_hash=self.root,
        )

    def _collect_siblings(self, leaf_idx: int) -> List[Tuple[bytes, str]]:
        """Walk from *leaf_idx* to the root, collecting sibling hashes."""
        siblings: List[Tuple[bytes, str]] = []
        layer: list = list(self._leaves)

        idx = leaf_idx
        while len(layer) > 1:
            next_layer: list = []
            for i in range(0, len(layer), 2):
                left = layer[i]
                right = layer[i + 1] if i + 1 < len(layer) else None
                parent = InternalNode(left=left, right=right)
                next_layer.append(parent)

                if i == idx or (i + 1 == idx and i + 1 < len(layer)):
                    if i == idx:
                        sib_digest = right.digest if right is not None else EMPTY_HASH
                        siblings.append((sib_digest, "R"))
                    else:
                        siblings.append((left.digest, "L"))
                    idx = i // 2

            layer = next_layer

        return siblings

    def batch_proofs(self, items: Sequence[Any]) -> List[MerkleProof]:
        """Generate inclusion proofs for multiple items."""
        return [self.proof(it) for it in items]

    # -- verification -------------------------------------------------------

    @staticmethod
    def verify_proof(proof: MerkleProof) -> bool:
        """Verify an inclusion proof against the stored root."""
        return proof.verify()

    # -- incremental update -------------------------------------------------

    def add_leaf(self, item: Any) -> None:
        """Add a leaf and rebuild the tree."""
        all_items = [leaf.item for leaf in self._leaves]
        all_items.append(_item_to_bytes(item))
        self.build(all_items)

    def remove_leaf(self, item: Any) -> None:
        """Remove a leaf and rebuild the tree."""
        item_bytes = _item_to_bytes(item)
        all_items = [leaf.item for leaf in self._leaves if leaf.item != item_bytes]
        if len(all_items) == len(self._leaves):
            raise KeyError(f"Item not in tree: {item!r}")
        self.build(all_items)

    def update_leaf(self, old_item: Any, new_item: Any) -> None:
        """Replace *old_item* with *new_item* and rebuild."""
        old_bytes = _item_to_bytes(old_item)
        new_bytes = _item_to_bytes(new_item)
        all_items = [
            new_bytes if leaf.item == old_bytes else leaf.item
            for leaf in self._leaves
        ]
        self.build(all_items)

    # -- serialization ------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize the entire tree to a compact binary format."""
        if self._root is None:
            return struct.pack(">I", 0)
        tree_data = self._root.to_bytes()
        return struct.pack(">I", len(tree_data)) + tree_data

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> "MerkleTree":
        """Deserialize a tree from bytes produced by ``to_bytes``."""
        (tree_len,) = struct.unpack_from(">I", buf, offset)
        tree = cls.__new__(cls)
        tree._leaves = []
        tree._item_index = {}
        if tree_len == 0:
            tree._root = None
            return tree
        tree._root, _ = _node_from_bytes(buf, offset + 4)
        tree._leaves = _collect_leaves(tree._root)
        for idx, leaf in enumerate(tree._leaves):
            object.__setattr__(leaf, "index", idx)
            tree._item_index[leaf.item] = idx
        return tree

    # -- pretty-print -------------------------------------------------------

    def pretty(self) -> str:
        """Human-readable tree representation."""
        if self._root is None:
            return "(empty tree)"
        return self._root.pretty()

    # -- statistics ---------------------------------------------------------

    @property
    def depth(self) -> int:
        """Maximum depth of the tree (root = 0)."""
        return self._compute_depth(self._root)

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    @property
    def node_count(self) -> int:
        return self._count_nodes(self._root)

    @property
    def proof_size(self) -> int:
        """Size of a single inclusion proof in bytes (worst case)."""
        return HASH_LEN + 2 + self.depth * (1 + HASH_LEN) + HASH_LEN

    def _compute_depth(self, node: Any) -> int:
        if node is None:
            return 0
        if isinstance(node, LeafNode):
            return 0
        left_d = self._compute_depth(node.left)
        right_d = self._compute_depth(node.right) if node.right else 0
        return 1 + max(left_d, right_d)

    def _count_nodes(self, node: Any) -> int:
        if node is None:
            return 0
        if isinstance(node, LeafNode):
            return 1
        right_c = self._count_nodes(node.right) if node.right else 0
        return 1 + self._count_nodes(node.left) + right_c

    def __len__(self) -> int:
        return self.leaf_count

    def __contains__(self, item: Any) -> bool:
        return _item_to_bytes(item) in self._item_index

    def __repr__(self) -> str:
        return (
            f"MerkleTree(leaves={self.leaf_count}, "
            f"depth={self.depth}, "
            f"root={self.root.hex()[:16]}…)"
        )


def _collect_leaves(node: Any) -> List[LeafNode]:
    """In-order traversal to collect all leaves."""
    if node is None:
        return []
    if isinstance(node, LeafNode):
        return [node]
    result = _collect_leaves(node.left)
    if node.right is not None:
        result.extend(_collect_leaves(node.right))
    return result


# ---------------------------------------------------------------------------
# Sparse Merkle tree
# ---------------------------------------------------------------------------


class SparseMerkleTree:
    """
    A sparse Merkle tree of fixed depth.

    Only populated leaves are stored; all other leaves default to
    ``EMPTY_HASH``.  Useful when the key space is enormous but only a
    small fraction of keys are occupied (e.g., state identifiers).
    """

    def __init__(self, depth: int = 256) -> None:
        if depth < 1 or depth > 256:
            raise ValueError("Depth must be between 1 and 256")
        self.depth = depth
        self._default_hashes = self._precompute_defaults()
        self._store: Dict[int, bytes] = {}
        self._root: bytes = self._default_hashes[self.depth]

    # -- default hashes for empty sub-trees ---------------------------------

    def _precompute_defaults(self) -> List[bytes]:
        """defaults[0] = leaf default, defaults[d] = root of empty tree depth d."""
        defaults = [EMPTY_HASH] * (self.depth + 1)
        for i in range(1, self.depth + 1):
            defaults[i] = hash_internal(defaults[i - 1], defaults[i - 1])
        return defaults

    # -- public API ---------------------------------------------------------

    @property
    def root(self) -> bytes:
        return self._root

    def get(self, key: int) -> bytes:
        """Return the leaf value at *key*, or ``EMPTY_HASH``."""
        return self._store.get(key, EMPTY_HASH)

    def set(self, key: int, value: bytes) -> None:
        """Set the leaf at *key* to *value* and recompute the root."""
        if key < 0 or key >= (1 << self.depth):
            raise ValueError(f"Key {key} out of range for depth {self.depth}")
        leaf_digest = hash_leaf(value)
        self._store[key] = leaf_digest
        self._root = self._recompute_root()

    def delete(self, key: int) -> None:
        """Remove the leaf at *key* and recompute the root."""
        if key in self._store:
            del self._store[key]
            self._root = self._recompute_root()

    def proof(self, key: int) -> MerkleProof:
        """Generate an inclusion (or non-inclusion) proof for *key*."""
        siblings: List[Tuple[bytes, str]] = []
        path_bits = self._key_to_path(key)

        current_keys = dict(self._store)
        for level in range(self.depth):
            bit = path_bits[level]
            left_keys: Dict[int, bytes] = {}
            right_keys: Dict[int, bytes] = {}
            for k, v in current_keys.items():
                k_bits = self._key_to_path(k)
                if k_bits[level] == 0:
                    left_keys[k] = v
                else:
                    right_keys[k] = v

            if bit == 0:
                sibling_hash = self._subtree_hash(right_keys, level + 1)
                siblings.append((sibling_hash, "R"))
                current_keys = left_keys
            else:
                sibling_hash = self._subtree_hash(left_keys, level + 1)
                siblings.append((sibling_hash, "L"))
                current_keys = right_keys

        leaf_hash = self._store.get(key, EMPTY_HASH)
        # Siblings were collected top-down; MerkleProof.verify walks bottom-up.
        siblings.reverse()
        return MerkleProof(leaf_hash=leaf_hash, siblings=siblings, root_hash=self._root)

    def verify_proof(self, proof: MerkleProof) -> bool:
        return proof.verify()

    # -- internals ----------------------------------------------------------

    def _key_to_path(self, key: int) -> List[int]:
        """Return the bit-path for *key* from root to leaf."""
        bits: List[int] = []
        for i in range(self.depth - 1, -1, -1):
            bits.append((key >> i) & 1)
        return bits

    def _subtree_hash(self, keys: Dict[int, bytes], level: int) -> bytes:
        """Compute the hash of a subtree rooted at *level*."""
        remaining_depth = self.depth - level
        if not keys:
            return self._default_hashes[remaining_depth]
        if remaining_depth == 0:
            assert len(keys) == 1
            return next(iter(keys.values()))
        left_keys: Dict[int, bytes] = {}
        right_keys: Dict[int, bytes] = {}
        for k, v in keys.items():
            k_bits = self._key_to_path(k)
            if k_bits[level] == 0:
                left_keys[k] = v
            else:
                right_keys[k] = v
        left_h = self._subtree_hash(left_keys, level + 1)
        right_h = self._subtree_hash(right_keys, level + 1)
        return hash_internal(left_h, right_h)

    def _recompute_root(self) -> bytes:
        """Full root recomputation from all stored leaves."""
        return self._subtree_hash(dict(self._store), 0)

    # -- statistics ---------------------------------------------------------

    @property
    def populated_count(self) -> int:
        return len(self._store)

    @property
    def total_leaves(self) -> int:
        return 1 << self.depth

    def pretty(self) -> str:
        lines = [
            f"SparseMerkleTree(depth={self.depth}, "
            f"populated={self.populated_count}, "
            f"root={self._root.hex()[:16]}…)",
        ]
        for key in sorted(self._store):
            lines.append(f"  [{key}] {self._store[key].hex()[:16]}…")
        return "\n".join(lines)

    def to_bytes(self) -> bytes:
        """Serialize: depth + number of entries + (key, value) pairs."""
        parts: list[bytes] = [struct.pack(">HI", self.depth, len(self._store))]
        for key in sorted(self._store):
            val = self._store[key]
            parts.append(struct.pack(">I", key))
            parts.append(val)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> "SparseMerkleTree":
        depth, count = struct.unpack_from(">HI", buf, offset)
        tree = cls(depth=depth)
        pos = offset + 6
        for _ in range(count):
            (key,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            val = buf[pos : pos + HASH_LEN]
            pos += HASH_LEN
            tree._store[key] = val
        tree._root = tree._recompute_root()
        return tree

    def __repr__(self) -> str:
        return (
            f"SparseMerkleTree(depth={self.depth}, "
            f"populated={self.populated_count}, "
            f"root={self._root.hex()[:16]}…)"
        )

    def __contains__(self, key: int) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return self.populated_count
