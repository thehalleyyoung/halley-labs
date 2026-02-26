"""
Transition witness encoding for bisimulation certificates.

For every transition in the quotient system, stores a concrete witness
consisting of source/target states in the respective equivalence classes,
the action label, and cryptographic hashes for integrity.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .merkle_tree import EMPTY_HASH, HASH_LEN, MerkleProof, MerkleTree, sha256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_bytes(val: Any) -> bytes:
    if isinstance(val, bytes):
        return val
    if isinstance(val, str):
        return val.encode("utf-8")
    if isinstance(val, int):
        return val.to_bytes(8, "big", signed=True)
    return repr(val).encode("utf-8")


def _hash_witness_triple(source: Any, action: Any, target: Any) -> bytes:
    """Hash a (source, action, target) witness triple."""
    h = hashlib.sha256()
    h.update(_to_bytes(source))
    h.update(b"|")
    h.update(_to_bytes(action))
    h.update(b"|")
    h.update(_to_bytes(target))
    return h.digest()


# ---------------------------------------------------------------------------
# TransitionWitness
# ---------------------------------------------------------------------------


@dataclass
class TransitionWitness:
    """
    A concrete witness for a single quotient-level transition.

    For a quotient transition  class_i --action--> class_j  the witness
    contains concrete states  s ∈ class_i  and  t ∈ class_j  such that
    s --action--> t  holds in the original system.
    """

    source_class: str
    target_class: str
    action: Any
    concrete_source: Any
    concrete_target: Any

    _digest: Optional[bytes] = field(default=None, repr=False, compare=False)

    @property
    def digest(self) -> bytes:
        if self._digest is None:
            self._digest = self._compute_hash()
        return self._digest

    def _compute_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.source_class.encode("utf-8"))
        h.update(self.target_class.encode("utf-8"))
        h.update(_to_bytes(self.action))
        h.update(_to_bytes(self.concrete_source))
        h.update(_to_bytes(self.concrete_target))
        return h.digest()

    def verify_classes(
        self,
        state_to_class: Dict[Any, str],
    ) -> Tuple[bool, List[str]]:
        """Check that the concrete states actually belong to the claimed classes."""
        errors: List[str] = []
        src_cls = state_to_class.get(self.concrete_source)
        if src_cls != self.source_class:
            errors.append(
                f"Source {self.concrete_source!r} in class {src_cls!r}, "
                f"expected {self.source_class!r}"
            )
        tgt_cls = state_to_class.get(self.concrete_target)
        if tgt_cls != self.target_class:
            errors.append(
                f"Target {self.concrete_target!r} in class {tgt_cls!r}, "
                f"expected {self.target_class!r}"
            )
        return (len(errors) == 0, errors)

    # -- serialization ------------------------------------------------------

    def to_bytes(self) -> bytes:
        parts: list[bytes] = []
        for field_val in (
            self.source_class,
            self.target_class,
            _to_bytes(self.action),
            _to_bytes(self.concrete_source),
            _to_bytes(self.concrete_target),
        ):
            raw = field_val if isinstance(field_val, bytes) else field_val.encode("utf-8") if isinstance(field_val, str) else _to_bytes(field_val)
            parts.append(struct.pack(">I", len(raw)))
            parts.append(raw)
        parts.append(self.digest)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["TransitionWitness", int]:
        pos = offset
        fields: list[bytes] = []
        for _ in range(5):
            (flen,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            fields.append(buf[pos : pos + flen])
            pos += flen
        digest = buf[pos : pos + HASH_LEN]
        pos += HASH_LEN
        tw = cls(
            source_class=fields[0].decode("utf-8"),
            target_class=fields[1].decode("utf-8"),
            action=fields[2],
            concrete_source=fields[3],
            concrete_target=fields[4],
        )
        tw._digest = digest
        return tw, pos

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_class": self.source_class,
            "target_class": self.target_class,
            "action": repr(self.action),
            "concrete_source": repr(self.concrete_source),
            "concrete_target": repr(self.concrete_target),
            "digest": self.digest.hex(),
        }

    def __repr__(self) -> str:
        return (
            f"TransitionWitness({self.source_class} "
            f"--{self.action!r}--> "
            f"{self.target_class})"
        )


# ---------------------------------------------------------------------------
# StutterWitness
# ---------------------------------------------------------------------------


@dataclass
class StutterWitness:
    """
    Witness for a stuttering transition.

    A stutter step is one where the equivalence class does not change;
    the witness stores a path through the concrete system that remains
    within the same class.
    """

    equiv_class: str
    path: List[Any]  # sequence of concrete states within the class
    actions: List[Any]  # action labels along the path

    _digest: Optional[bytes] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.path) - 1 and len(self.path) > 0:
            if len(self.actions) != max(0, len(self.path) - 1):
                raise ValueError(
                    f"Path length {len(self.path)} requires "
                    f"{len(self.path) - 1} actions, got {len(self.actions)}"
                )

    @property
    def digest(self) -> bytes:
        if self._digest is None:
            self._digest = self._compute_hash()
        return self._digest

    def _compute_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.equiv_class.encode("utf-8"))
        h.update(struct.pack(">I", len(self.path)))
        for s in self.path:
            h.update(_to_bytes(s))
        for a in self.actions:
            h.update(_to_bytes(a))
        return h.digest()

    @property
    def path_length(self) -> int:
        return len(self.path)

    def verify_within_class(
        self,
        state_to_class: Dict[Any, str],
    ) -> Tuple[bool, List[str]]:
        """Check that every state in the path belongs to the claimed class."""
        errors: List[str] = []
        for i, s in enumerate(self.path):
            cls = state_to_class.get(s)
            if cls != self.equiv_class:
                errors.append(
                    f"Path[{i}] state {s!r} in class {cls!r}, "
                    f"expected {self.equiv_class!r}"
                )
        return (len(errors) == 0, errors)

    def to_bytes(self) -> bytes:
        parts: list[bytes] = []
        cls_b = self.equiv_class.encode("utf-8")
        parts.append(struct.pack(">H", len(cls_b)))
        parts.append(cls_b)
        parts.append(struct.pack(">I", len(self.path)))
        for s in self.path:
            sb = _to_bytes(s)
            parts.append(struct.pack(">I", len(sb)))
            parts.append(sb)
        parts.append(struct.pack(">I", len(self.actions)))
        for a in self.actions:
            ab = _to_bytes(a)
            parts.append(struct.pack(">I", len(ab)))
            parts.append(ab)
        parts.append(self.digest)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["StutterWitness", int]:
        pos = offset
        (cls_len,) = struct.unpack_from(">H", buf, pos)
        pos += 2
        equiv_class = buf[pos : pos + cls_len].decode("utf-8")
        pos += cls_len
        (path_len,) = struct.unpack_from(">I", buf, pos)
        pos += 4
        path: list[bytes] = []
        for _ in range(path_len):
            (slen,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            path.append(buf[pos : pos + slen])
            pos += slen
        (act_len,) = struct.unpack_from(">I", buf, pos)
        pos += 4
        actions: list[bytes] = []
        for _ in range(act_len):
            (alen,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            actions.append(buf[pos : pos + alen])
            pos += alen
        digest = buf[pos : pos + HASH_LEN]
        pos += HASH_LEN
        sw = cls(equiv_class=equiv_class, path=path, actions=actions)
        sw._digest = digest
        return sw, pos

    def __repr__(self) -> str:
        return (
            f"StutterWitness(class={self.equiv_class!r}, "
            f"path_len={self.path_length})"
        )


# ---------------------------------------------------------------------------
# FairnessWitness
# ---------------------------------------------------------------------------


@dataclass
class FairnessWitness:
    """
    Witness for fairness (acceptance pair) preservation.

    For each acceptance pair (L, U) in the specification, stores concrete
    states demonstrating that the quotient respects the acceptance condition.
    """

    pair_id: str
    accepting_class: str
    accepting_state: Any  # concrete state in the accepting set
    rejecting_class: str
    rejecting_state: Any  # concrete state in the rejecting set
    cycle_witness: List[Any]  # states forming a cycle through the pair

    _digest: Optional[bytes] = field(default=None, repr=False, compare=False)

    @property
    def digest(self) -> bytes:
        if self._digest is None:
            self._digest = self._compute_hash()
        return self._digest

    def _compute_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.pair_id.encode("utf-8"))
        h.update(self.accepting_class.encode("utf-8"))
        h.update(_to_bytes(self.accepting_state))
        h.update(self.rejecting_class.encode("utf-8"))
        h.update(_to_bytes(self.rejecting_state))
        h.update(struct.pack(">I", len(self.cycle_witness)))
        for s in self.cycle_witness:
            h.update(_to_bytes(s))
        return h.digest()

    def to_bytes(self) -> bytes:
        parts: list[bytes] = []
        for s in (self.pair_id, self.accepting_class, self.rejecting_class):
            sb = s.encode("utf-8")
            parts.append(struct.pack(">H", len(sb)))
            parts.append(sb)
        for val in (self.accepting_state, self.rejecting_state):
            vb = _to_bytes(val)
            parts.append(struct.pack(">I", len(vb)))
            parts.append(vb)
        parts.append(struct.pack(">I", len(self.cycle_witness)))
        for s in self.cycle_witness:
            sb = _to_bytes(s)
            parts.append(struct.pack(">I", len(sb)))
            parts.append(sb)
        parts.append(self.digest)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["FairnessWitness", int]:
        pos = offset
        strings: list[str] = []
        for _ in range(3):
            (slen,) = struct.unpack_from(">H", buf, pos)
            pos += 2
            strings.append(buf[pos : pos + slen].decode("utf-8"))
            pos += slen
        vals: list[bytes] = []
        for _ in range(2):
            (vlen,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            vals.append(buf[pos : pos + vlen])
            pos += vlen
        (cyc_len,) = struct.unpack_from(">I", buf, pos)
        pos += 4
        cycle: list[bytes] = []
        for _ in range(cyc_len):
            (clen,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            cycle.append(buf[pos : pos + clen])
            pos += clen
        digest = buf[pos : pos + HASH_LEN]
        pos += HASH_LEN
        fw = cls(
            pair_id=strings[0],
            accepting_class=strings[1],
            rejecting_class=strings[2],
            accepting_state=vals[0],
            rejecting_state=vals[1],
            cycle_witness=cycle,
        )
        fw._digest = digest
        return fw, pos

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "accepting_class": self.accepting_class,
            "accepting_state": repr(self.accepting_state),
            "rejecting_class": self.rejecting_class,
            "rejecting_state": repr(self.rejecting_state),
            "cycle_length": len(self.cycle_witness),
            "digest": self.digest.hex(),
        }

    def __repr__(self) -> str:
        return (
            f"FairnessWitness(pair={self.pair_id!r}, "
            f"accept={self.accepting_class!r}, "
            f"reject={self.rejecting_class!r})"
        )


# ---------------------------------------------------------------------------
# WitnessSet
# ---------------------------------------------------------------------------


class WitnessSet:
    """
    The complete witness set for a quotient system.

    Contains all transition witnesses, stutter witnesses, and fairness
    witnesses, together with a Merkle tree over their hashes.
    """

    def __init__(self) -> None:
        self._transitions: List[TransitionWitness] = []
        self._stutters: List[StutterWitness] = []
        self._fairness: List[FairnessWitness] = []
        self._tree: Optional[MerkleTree] = None
        self._dirty: bool = True

    # -- construction -------------------------------------------------------

    def add_transition(self, tw: TransitionWitness) -> None:
        self._transitions.append(tw)
        self._dirty = True

    def add_stutter(self, sw: StutterWitness) -> None:
        self._stutters.append(sw)
        self._dirty = True

    def add_fairness(self, fw: FairnessWitness) -> None:
        self._fairness.append(fw)
        self._dirty = True

    def _rebuild_tree(self) -> None:
        if not self._dirty:
            return
        all_hashes: list[bytes] = []
        all_hashes.extend(tw.digest for tw in self._transitions)
        all_hashes.extend(sw.digest for sw in self._stutters)
        all_hashes.extend(fw.digest for fw in self._fairness)
        self._tree = MerkleTree(all_hashes) if all_hashes else MerkleTree()
        self._dirty = False

    # -- accessors ----------------------------------------------------------

    @property
    def transitions(self) -> List[TransitionWitness]:
        return list(self._transitions)

    @property
    def stutters(self) -> List[StutterWitness]:
        return list(self._stutters)

    @property
    def fairness_witnesses(self) -> List[FairnessWitness]:
        return list(self._fairness)

    @property
    def root(self) -> bytes:
        self._rebuild_tree()
        assert self._tree is not None
        return self._tree.root

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    @property
    def stutter_count(self) -> int:
        return len(self._stutters)

    @property
    def fairness_count(self) -> int:
        return len(self._fairness)

    @property
    def total_count(self) -> int:
        return self.transition_count + self.stutter_count + self.fairness_count

    # -- completeness check -------------------------------------------------

    def check_completeness(
        self,
        expected_transitions: Set[Tuple[str, Any, str]],
    ) -> Tuple[bool, List[str]]:
        """
        Check that every expected quotient transition has a witness.

        *expected_transitions* is a set of ``(source_class, action, target_class)``
        tuples representing all transitions in the quotient system.
        """
        witnessed: Set[Tuple[str, Any, str]] = set()
        for tw in self._transitions:
            key = (tw.source_class, tw.action, tw.target_class)
            witnessed.add(key)
        missing = expected_transitions - witnessed
        errors: List[str] = []
        for src, act, tgt in sorted(missing):
            errors.append(f"Missing witness: {src} --{act!r}--> {tgt}")
        return (len(errors) == 0, errors)

    # -- consistency check --------------------------------------------------

    def check_consistency(
        self,
        state_to_class: Dict[Any, str],
    ) -> Tuple[bool, List[str]]:
        """
        Check that all witnesses are consistent with the equivalence binding.

        Each concrete source/target must belong to the claimed class.
        """
        errors: List[str] = []
        for tw in self._transitions:
            ok, errs = tw.verify_classes(state_to_class)
            errors.extend(errs)
        for sw in self._stutters:
            ok, errs = sw.verify_within_class(state_to_class)
            errors.extend(errs)
        return (len(errors) == 0, errors)

    # -- proof generation ---------------------------------------------------

    def proof_for_transition(self, index: int) -> MerkleProof:
        self._rebuild_tree()
        assert self._tree is not None
        tw = self._transitions[index]
        return self._tree.proof(tw.digest)

    def proof_for_stutter(self, index: int) -> MerkleProof:
        self._rebuild_tree()
        assert self._tree is not None
        sw = self._stutters[index]
        return self._tree.proof(sw.digest)

    def proof_for_fairness(self, index: int) -> MerkleProof:
        self._rebuild_tree()
        assert self._tree is not None
        fw = self._fairness[index]
        return self._tree.proof(fw.digest)

    # -- serialization ------------------------------------------------------

    def to_bytes(self) -> bytes:
        parts: list[bytes] = []
        parts.append(struct.pack(">III",
                                 self.transition_count,
                                 self.stutter_count,
                                 self.fairness_count))
        for tw in self._transitions:
            data = tw.to_bytes()
            parts.append(struct.pack(">I", len(data)))
            parts.append(data)
        for sw in self._stutters:
            data = sw.to_bytes()
            parts.append(struct.pack(">I", len(data)))
            parts.append(data)
        for fw in self._fairness:
            data = fw.to_bytes()
            parts.append(struct.pack(">I", len(data)))
            parts.append(data)
        parts.append(self.root)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> "WitnessSet":
        ws = cls()
        pos = offset
        t_count, s_count, f_count = struct.unpack_from(">III", buf, pos)
        pos += 12
        for _ in range(t_count):
            (dlen,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            tw, _ = TransitionWitness.from_bytes(buf, pos)
            ws._transitions.append(tw)
            pos += dlen
        for _ in range(s_count):
            (dlen,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            sw, _ = StutterWitness.from_bytes(buf, pos)
            ws._stutters.append(sw)
            pos += dlen
        for _ in range(f_count):
            (dlen,) = struct.unpack_from(">I", buf, pos)
            pos += 4
            fw, _ = FairnessWitness.from_bytes(buf, pos)
            ws._fairness.append(fw)
            pos += dlen
        # skip root
        pos += HASH_LEN
        ws._dirty = True
        return ws

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_count": self.transition_count,
            "stutter_count": self.stutter_count,
            "fairness_count": self.fairness_count,
            "root": self.root.hex(),
            "transitions": [tw.to_dict() for tw in self._transitions],
            "fairness": [fw.to_dict() for fw in self._fairness],
        }

    def __repr__(self) -> str:
        return (
            f"WitnessSet(transitions={self.transition_count}, "
            f"stutters={self.stutter_count}, "
            f"fairness={self.fairness_count}, "
            f"root={self.root.hex()[:16]}…)"
        )

    def __len__(self) -> int:
        return self.total_count
