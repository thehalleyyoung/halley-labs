"""
TLA-lite state representation.

A *state* maps variable names to TLAValues.  The module provides:

* ``TLAState`` – single state (immutable after construction).
* ``StateSignature`` – compact fingerprint for a state.
* ``StateSpace`` – collection of discovered states with efficient lookup.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import struct
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from .values import TLAValue, TLAValueError, values_to_json_string, value_from_json_string


# ---------------------------------------------------------------------------
# Zobrist table – lazily initialised per-variable random bit strings
# ---------------------------------------------------------------------------
class _ZobristTable:
    """Per-variable random 64-bit masks for Zobrist-style hashing.

    The table is seeded deterministically from a fixed seed so that state
    fingerprints are reproducible across runs (useful for regression tests).
    """

    _SEED = 0xCAFE_BEEF_DEAD_C0DE

    def __init__(self) -> None:
        self._rng = random.Random(self._SEED)
        self._table: Dict[str, int] = {}

    def mask_for(self, var_name: str) -> int:
        if var_name not in self._table:
            self._table[var_name] = self._rng.getrandbits(64)
        return self._table[var_name]

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed if seed is not None else self._SEED)
        self._table.clear()


_zobrist = _ZobristTable()


# ---------------------------------------------------------------------------
# StateSignature – compact 64-bit fingerprint
# ---------------------------------------------------------------------------
class StateSignature:
    """64-bit Zobrist hash fingerprint for a state.

    Two states with the same variable bindings will have the same signature
    (no false negatives).  Collisions are theoretically possible but
    extremely unlikely for practical state-space sizes.
    """

    __slots__ = ("_hash",)

    def __init__(self, h: int = 0) -> None:
        self._hash = h & 0xFFFF_FFFF_FFFF_FFFF

    @classmethod
    def from_bindings(cls, bindings: Dict[str, TLAValue]) -> "StateSignature":
        h = 0
        for var_name in sorted(bindings):
            val_hash = hash(bindings[var_name]) & 0xFFFF_FFFF_FFFF_FFFF
            mask = _zobrist.mask_for(var_name)
            h ^= _mix64(val_hash ^ mask)
        return cls(h)

    @property
    def value(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StateSignature) and self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"StateSignature(0x{self._hash:016x})"

    def hex(self) -> str:
        return f"0x{self._hash:016x}"


def _mix64(x: int) -> int:
    """Bijective 64-bit mixing function (splitmix-style)."""
    x = ((x >> 30) ^ x) * 0xBF58476D1CE4E5B9
    x &= 0xFFFF_FFFF_FFFF_FFFF
    x = ((x >> 27) ^ x) * 0x94D049BB133111EB
    x &= 0xFFFF_FFFF_FFFF_FFFF
    x = (x >> 31) ^ x
    return x & 0xFFFF_FFFF_FFFF_FFFF


# ---------------------------------------------------------------------------
# TLAState
# ---------------------------------------------------------------------------
class TLAState:
    """Immutable mapping from variable names to TLAValues.

    States are the nodes of the state graph explored during model checking.
    """

    __slots__ = ("_bindings", "_sig", "_canon_hash")

    def __init__(self, bindings: Dict[str, TLAValue] | None = None) -> None:
        self._bindings: Dict[str, TLAValue] = dict(bindings) if bindings else {}
        self._sig: Optional[StateSignature] = None
        self._canon_hash: Optional[int] = None

    # --- accessors --------------------------------------------------------

    def get(self, var: str) -> TLAValue:
        if var not in self._bindings:
            raise TLAValueError(f"Variable '{var}' not in state. Have: {sorted(self._bindings)}")
        return self._bindings[var]

    def __getitem__(self, var: str) -> TLAValue:
        return self.get(var)

    def has_var(self, var: str) -> bool:
        return var in self._bindings

    @property
    def variables(self) -> FrozenSet[str]:
        return frozenset(self._bindings.keys())

    @property
    def bindings(self) -> Dict[str, TLAValue]:
        return dict(self._bindings)

    def items(self) -> Iterable[Tuple[str, TLAValue]]:
        return self._bindings.items()

    def __len__(self) -> int:
        return len(self._bindings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._bindings)

    def __contains__(self, var: str) -> bool:
        return var in self._bindings

    # --- derived state constructors ---------------------------------------

    def with_update(self, var: str, val: TLAValue) -> "TLAState":
        new_bindings = dict(self._bindings)
        new_bindings[var] = val
        return TLAState(new_bindings)

    def with_updates(self, updates: Dict[str, TLAValue]) -> "TLAState":
        new_bindings = dict(self._bindings)
        new_bindings.update(updates)
        return TLAState(new_bindings)

    def project(self, vars: Iterable[str]) -> "TLAState":
        """Return a new state containing only the listed variables."""
        return TLAState({v: self._bindings[v] for v in vars if v in self._bindings})

    # --- canonical form / hashing -----------------------------------------

    def canonical_key(self) -> str:
        """Deterministic string key for dictionary lookup and comparison."""
        parts: List[str] = []
        for var in sorted(self._bindings):
            parts.append(f"{var}={values_to_json_string(self._bindings[var])}")
        return "|".join(parts)

    def fingerprint(self) -> StateSignature:
        if self._sig is None:
            self._sig = StateSignature.from_bindings(self._bindings)
        return self._sig

    def content_hash(self) -> str:
        """SHA-256 of the canonical serialization (for archival/logging)."""
        raw = self.canonical_key().encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    # --- comparison -------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TLAState):
            return NotImplemented
        if self._bindings.keys() != other._bindings.keys():
            return False
        return all(self._bindings[k] == other._bindings[k] for k in self._bindings)

    def __hash__(self) -> int:
        if self._canon_hash is None:
            self._canon_hash = hash(self.canonical_key())
        return self._canon_hash

    def __lt__(self, other: "TLAState") -> bool:
        return self.canonical_key() < other.canonical_key()

    # --- serialization ----------------------------------------------------

    def to_json(self) -> dict:
        return {
            "_type": "State",
            "bindings": {
                var: val.to_json() for var, val in sorted(self._bindings.items())
            },
        }

    @classmethod
    def from_json(cls, obj: dict) -> "TLAState":
        if obj.get("_type") != "State":
            raise TLAValueError(f"Expected State JSON, got {obj.get('_type')}")
        bindings = {
            var: TLAValue.from_json(val_obj)
            for var, val_obj in obj["bindings"].items()
        }
        return cls(bindings)

    def to_json_string(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True)

    @classmethod
    def from_json_string(cls, s: str) -> "TLAState":
        return cls.from_json(json.loads(s))

    # --- pretty printing --------------------------------------------------

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [f"{pad}/\\"]
        for var in sorted(self._bindings):
            lines.append(f"{pad}  {var} = {self._bindings[var].pretty(indent + 1)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v.pretty()}" for k, v in sorted(self._bindings.items()))
        return f"TLAState({items})"


# ---------------------------------------------------------------------------
# StateSpace – tracks all discovered states
# ---------------------------------------------------------------------------
class StateSpace:
    """Efficient container for the set of discovered states.

    Uses both a canonical-key dictionary (for exact dedup) and a Zobrist
    fingerprint set (for fast probable membership).  Supports graph edges
    (transitions) between states.
    """

    def __init__(self) -> None:
        self._by_key: Dict[str, TLAState] = {}
        self._by_fp: Dict[StateSignature, List[TLAState]] = {}
        self._transitions: Dict[str, Set[str]] = {}
        self._initial_keys: Set[str] = set()

    # --- insertion --------------------------------------------------------

    def add(self, state: TLAState, *, is_initial: bool = False) -> bool:
        """Add a state. Returns True if the state was new."""
        key = state.canonical_key()
        if key in self._by_key:
            return False
        self._by_key[key] = state
        fp = state.fingerprint()
        self._by_fp.setdefault(fp, []).append(state)
        if is_initial:
            self._initial_keys.add(key)
        return True

    def add_transition(self, src: TLAState, dst: TLAState) -> None:
        sk = src.canonical_key()
        dk = dst.canonical_key()
        self._transitions.setdefault(sk, set()).add(dk)

    # --- lookup -----------------------------------------------------------

    def __contains__(self, state: TLAState) -> bool:
        return state.canonical_key() in self._by_key

    def get(self, state: TLAState) -> Optional[TLAState]:
        return self._by_key.get(state.canonical_key())

    def probably_contains(self, state: TLAState) -> bool:
        """Fast probabilistic check using Zobrist fingerprint."""
        fp = state.fingerprint()
        return fp in self._by_fp

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[TLAState]:
        return iter(self._by_key.values())

    # --- graph queries ----------------------------------------------------

    @property
    def num_transitions(self) -> int:
        return sum(len(dsts) for dsts in self._transitions.values())

    def successors(self, state: TLAState) -> List[TLAState]:
        key = state.canonical_key()
        dst_keys = self._transitions.get(key, set())
        return [self._by_key[dk] for dk in dst_keys if dk in self._by_key]

    def initial_states(self) -> List[TLAState]:
        return [self._by_key[k] for k in sorted(self._initial_keys)]

    def deadlock_states(self) -> List[TLAState]:
        """States with no outgoing transitions."""
        all_keys = set(self._by_key.keys())
        has_successors = set(self._transitions.keys())
        deadlocked = all_keys - has_successors
        return [self._by_key[k] for k in sorted(deadlocked)]

    # --- statistics -------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "num_states": len(self._by_key),
            "num_initial": len(self._initial_keys),
            "num_transitions": self.num_transitions,
            "num_deadlocks": len(self.deadlock_states()),
            "num_fingerprint_buckets": len(self._by_fp),
        }

    # --- serialization ----------------------------------------------------

    def to_json(self) -> dict:
        states = {k: s.to_json() for k, s in sorted(self._by_key.items())}
        transitions = {k: sorted(v) for k, v in sorted(self._transitions.items())}
        return {
            "_type": "StateSpace",
            "states": states,
            "transitions": transitions,
            "initial": sorted(self._initial_keys),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "StateSpace":
        space = cls()
        state_map: Dict[str, TLAState] = {}
        for key, sobj in obj["states"].items():
            st = TLAState.from_json(sobj)
            state_map[key] = st
            space.add(st, is_initial=(key in obj.get("initial", [])))
        for src_key, dst_keys in obj.get("transitions", {}).items():
            src = state_map[src_key]
            for dk in dst_keys:
                dst = state_map[dk]
                space.add_transition(src, dst)
        return space

    def clear(self) -> None:
        self._by_key.clear()
        self._by_fp.clear()
        self._transitions.clear()
        self._initial_keys.clear()

    def __repr__(self) -> str:
        return (
            f"StateSpace(states={len(self._by_key)}, "
            f"transitions={self.num_transitions})"
        )
