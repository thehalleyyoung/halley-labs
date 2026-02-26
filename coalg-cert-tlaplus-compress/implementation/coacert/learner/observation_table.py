"""
Observation table data structure for L*-style coalgebraic learning.

The table is indexed by *access sequences* (rows) and *distinguishing
suffixes* (columns).  Each cell stores the F-behaviour observed when the
access sequence is followed by the suffix on the concrete transition
system.  Nondeterministic systems produce set-valued cells.

Short rows S contain representative access sequences.  Long rows SA
contain one-step extensions of short rows by each available action.
A table is *closed* when every long row has an equivalent short row,
and *consistent* when equivalent short rows have equivalent extensions.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import textwrap
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# A sequence of actions (the empty tuple represents the initial state).
AccessSequence = Tuple[str, ...]

# A distinguishing suffix (also a tuple of actions).
Suffix = Tuple[str, ...]

# A single observation: set of proposition-sets and successor-action info.
# In the nondeterministic case every reachable configuration contributes.
Observation = FrozenSet[Tuple[FrozenSet[str], Tuple[Tuple[str, FrozenSet[str]], ...]]]


def _make_observation(
    propositions: FrozenSet[str],
    successor_map: Dict[str, FrozenSet[str]],
) -> Observation:
    """Create an atomic (single-configuration) observation."""
    succ_tuple = tuple(sorted(
        (act, targets) for act, targets in successor_map.items()
    ))
    return frozenset({(propositions, succ_tuple)})


def _merge_observations(a: Observation, b: Observation) -> Observation:
    """Merge two observations (union for nondeterminism)."""
    return a | b


# ---------------------------------------------------------------------------
# Row value wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowSignature:
    """The signature of a row: the tuple of cell values across all columns."""

    values: Tuple[Optional[Observation], ...]

    def equivalent_to(self, other: "RowSignature") -> bool:
        if len(self.values) != len(other.values):
            return False
        for v1, v2 in zip(self.values, other.values):
            if v1 is None or v2 is None:
                continue  # unknown cells are not distinguishing
            if v1 != v2:
                return False
        return True

    def digest(self) -> str:
        h = hashlib.sha256()
        for v in self.values:
            h.update(repr(v).encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Compressed cell store
# ---------------------------------------------------------------------------

class _CellStore:
    """Memory-efficient cell storage with deduplication.

    Observations can be large (sets of frozensets).  This store keeps a
    single canonical copy of every distinct observation and maps cells to
    indices into that pool.
    """

    def __init__(self) -> None:
        self._pool: List[Observation] = []
        self._pool_index: Dict[int, int] = {}  # id(obs) is useless for frozen; use hash
        self._obs_to_idx: Dict[Observation, int] = {}
        self._cells: Dict[Tuple[AccessSequence, Suffix], int] = {}
        self._none_cells: Set[Tuple[AccessSequence, Suffix]] = set()

    # -- public interface ---------------------------------------------------

    def get(
        self, seq: AccessSequence, suffix: Suffix
    ) -> Optional[Observation]:
        key = (seq, suffix)
        if key in self._none_cells:
            return None
        idx = self._cells.get(key)
        if idx is None:
            return None
        return self._pool[idx]

    def set(
        self, seq: AccessSequence, suffix: Suffix, obs: Optional[Observation]
    ) -> None:
        key = (seq, suffix)
        if obs is None:
            self._none_cells.add(key)
            self._cells.pop(key, None)
            return
        self._none_cells.discard(key)
        idx = self._intern(obs)
        self._cells[key] = idx

    def delete_row(self, seq: AccessSequence) -> None:
        keys_to_remove = [k for k in self._cells if k[0] == seq]
        for k in keys_to_remove:
            del self._cells[k]
        none_to_remove = [k for k in self._none_cells if k[0] == seq]
        for k in none_to_remove:
            self._none_cells.discard(k)

    def delete_column(self, suffix: Suffix) -> None:
        keys_to_remove = [k for k in self._cells if k[1] == suffix]
        for k in keys_to_remove:
            del self._cells[k]
        none_to_remove = [k for k in self._none_cells if k[1] == suffix]
        for k in none_to_remove:
            self._none_cells.discard(k)

    def filled_count(self) -> int:
        return len(self._cells)

    def total_tracked(self) -> int:
        return len(self._cells) + len(self._none_cells)

    def pool_size(self) -> int:
        return len(self._pool)

    # -- internals ----------------------------------------------------------

    def _intern(self, obs: Observation) -> int:
        idx = self._obs_to_idx.get(obs)
        if idx is not None:
            return idx
        idx = len(self._pool)
        self._pool.append(obs)
        self._obs_to_idx[obs] = idx
        return idx

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for checkpointing."""
        pool_serial = []
        for obs in self._pool:
            obs_serial = []
            for props, succ_tuple in obs:
                obs_serial.append({
                    "props": sorted(props),
                    "succs": [(a, sorted(ts)) for a, ts in succ_tuple],
                })
            pool_serial.append(obs_serial)
        cells_serial = {
            repr(k): v for k, v in self._cells.items()
        }
        return {"pool": pool_serial, "cells": cells_serial}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_CellStore":
        store = cls()
        for obs_serial in data.get("pool", []):
            items: Set[Tuple[FrozenSet[str], Tuple[Tuple[str, FrozenSet[str]], ...]]] = set()
            for entry in obs_serial:
                props = frozenset(entry["props"])
                succs = tuple((a, frozenset(ts)) for a, ts in entry["succs"])
                items.add((props, succs))
            store._pool.append(frozenset(items))
        store._obs_to_idx = {obs: i for i, obs in enumerate(store._pool)}
        return store


# ---------------------------------------------------------------------------
# Main observation table
# ---------------------------------------------------------------------------

class ObservationTable:
    """L*-style observation table for coalgebraic learning.

    Parameters
    ----------
    actions : set of str
        The action alphabet Σ (Act).
    initial_suffix : Suffix, optional
        The initial column; defaults to the empty suffix ``()``.
    """

    def __init__(
        self,
        actions: Set[str],
        initial_suffix: Optional[Suffix] = None,
    ) -> None:
        self._actions: List[str] = sorted(actions)
        self._action_set: Set[str] = set(actions)

        # Ordered collections to preserve insertion order.
        self._short_rows: OrderedDict[AccessSequence, None] = OrderedDict()
        self._long_rows: OrderedDict[AccessSequence, None] = OrderedDict()
        self._columns: List[Suffix] = []
        self._column_set: Set[Suffix] = set()

        self._cells = _CellStore()

        # Statistics
        self._query_count: int = 0
        self._update_count: int = 0

        # Initialise with ε row and default column.
        self.add_short_row(())
        col = initial_suffix if initial_suffix is not None else ()
        self.add_column(col)

    # -- row management -----------------------------------------------------

    @property
    def short_rows(self) -> List[AccessSequence]:
        return list(self._short_rows.keys())

    @property
    def long_rows(self) -> List[AccessSequence]:
        return list(self._long_rows.keys())

    @property
    def all_rows(self) -> List[AccessSequence]:
        return self.short_rows + self.long_rows

    @property
    def columns(self) -> List[Suffix]:
        return list(self._columns)

    @property
    def actions(self) -> List[str]:
        return list(self._actions)

    def has_row(self, seq: AccessSequence) -> bool:
        return seq in self._short_rows or seq in self._long_rows

    def is_short(self, seq: AccessSequence) -> bool:
        return seq in self._short_rows

    def is_long(self, seq: AccessSequence) -> bool:
        return seq in self._long_rows

    def add_short_row(self, seq: AccessSequence) -> bool:
        """Add *seq* as a short (representative) row.

        Returns True if the row was newly added.
        """
        if seq in self._short_rows:
            return False
        # Promote from long rows if present.
        if seq in self._long_rows:
            del self._long_rows[seq]
        self._short_rows[seq] = None
        # Ensure one-step extensions exist as long rows.
        for act in self._actions:
            ext = seq + (act,)
            if ext not in self._short_rows and ext not in self._long_rows:
                self._long_rows[ext] = None
        logger.debug("Added short row %s", seq)
        return True

    def add_long_row(self, seq: AccessSequence) -> bool:
        """Add *seq* as a long (extension) row."""
        if seq in self._short_rows or seq in self._long_rows:
            return False
        self._long_rows[seq] = None
        logger.debug("Added long row %s", seq)
        return True

    def promote_to_short(self, seq: AccessSequence) -> bool:
        """Promote a long row to a short row, creating its extensions."""
        if seq in self._short_rows:
            return False
        if seq in self._long_rows:
            del self._long_rows[seq]
        return self.add_short_row(seq)

    def remove_row(self, seq: AccessSequence) -> bool:
        if seq in self._short_rows:
            del self._short_rows[seq]
            self._cells.delete_row(seq)
            return True
        if seq in self._long_rows:
            del self._long_rows[seq]
            self._cells.delete_row(seq)
            return True
        return False

    # -- column management --------------------------------------------------

    def add_column(self, suffix: Suffix) -> bool:
        """Add a distinguishing suffix (column)."""
        if suffix in self._column_set:
            return False
        self._columns.append(suffix)
        self._column_set.add(suffix)
        logger.debug("Added column %s", suffix)
        return True

    def has_column(self, suffix: Suffix) -> bool:
        return suffix in self._column_set

    def remove_column(self, suffix: Suffix) -> bool:
        if suffix not in self._column_set:
            return False
        self._columns.remove(suffix)
        self._column_set.discard(suffix)
        self._cells.delete_column(suffix)
        return True

    # -- cell access --------------------------------------------------------

    def get_cell(
        self, seq: AccessSequence, suffix: Suffix
    ) -> Optional[Observation]:
        self._query_count += 1
        return self._cells.get(seq, suffix)

    def set_cell(
        self, seq: AccessSequence, suffix: Suffix, obs: Optional[Observation]
    ) -> None:
        self._update_count += 1
        self._cells.set(seq, suffix, obs)

    def update_cell(
        self,
        seq: AccessSequence,
        suffix: Suffix,
        obs: Observation,
        merge: bool = True,
    ) -> None:
        """Update a cell, optionally merging with existing observations."""
        existing = self._cells.get(seq, suffix)
        if existing is not None and merge:
            obs = _merge_observations(existing, obs)
        self.set_cell(seq, suffix, obs)

    def get_row_values(
        self, seq: AccessSequence
    ) -> List[Optional[Observation]]:
        return [self._cells.get(seq, col) for col in self._columns]

    # -- row signatures and equivalence -------------------------------------

    def row_signature(self, seq: AccessSequence) -> RowSignature:
        vals = tuple(self._cells.get(seq, col) for col in self._columns)
        return RowSignature(values=vals)

    def rows_equivalent(self, s1: AccessSequence, s2: AccessSequence) -> bool:
        return self.row_signature(s1).equivalent_to(self.row_signature(s2))

    def find_equivalent_short_row(
        self, seq: AccessSequence
    ) -> Optional[AccessSequence]:
        """Find a short row equivalent to *seq*, or None."""
        sig = self.row_signature(seq)
        for sr in self._short_rows:
            if self.row_signature(sr).equivalent_to(sig):
                return sr
        return None

    def distinct_short_signatures(self) -> Dict[str, List[AccessSequence]]:
        """Group short rows by signature digest."""
        groups: Dict[str, List[AccessSequence]] = {}
        for sr in self._short_rows:
            digest = self.row_signature(sr).digest()
            groups.setdefault(digest, []).append(sr)
        return groups

    def equivalence_classes(self) -> List[List[AccessSequence]]:
        """Partition all rows into equivalence classes."""
        classes: List[List[AccessSequence]] = []
        assigned: Set[AccessSequence] = set()
        for row in self.all_rows:
            if row in assigned:
                continue
            cls_members = [row]
            assigned.add(row)
            sig = self.row_signature(row)
            for other in self.all_rows:
                if other in assigned:
                    continue
                if self.row_signature(other).equivalent_to(sig):
                    cls_members.append(other)
                    assigned.add(other)
            classes.append(cls_members)
        return classes

    # -- closedness and consistency -----------------------------------------

    def find_unclosed_row(self) -> Optional[AccessSequence]:
        """Return a long row with no equivalent short row, or None."""
        for lr in self._long_rows:
            if self.find_equivalent_short_row(lr) is None:
                return lr
        return None

    def is_closed(self) -> bool:
        return self.find_unclosed_row() is None

    def find_inconsistency(
        self,
    ) -> Optional[Tuple[AccessSequence, AccessSequence, str, Suffix]]:
        """Find an inconsistency: two equivalent short rows whose
        extensions under some action differ on some suffix.

        Returns ``(s1, s2, action, suffix)`` or None.
        """
        short_list = self.short_rows
        for i, s1 in enumerate(short_list):
            sig1 = self.row_signature(s1)
            for s2 in short_list[i + 1:]:
                if not sig1.equivalent_to(self.row_signature(s2)):
                    continue
                # s1 ≡ s2 — check extensions
                for act in self._actions:
                    ext1 = s1 + (act,)
                    ext2 = s2 + (act,)
                    sig_e1 = self.row_signature(ext1)
                    sig_e2 = self.row_signature(ext2)
                    if not sig_e1.equivalent_to(sig_e2):
                        # Find the distinguishing column
                        for col_idx, col in enumerate(self._columns):
                            v1 = sig_e1.values[col_idx] if col_idx < len(sig_e1.values) else None
                            v2 = sig_e2.values[col_idx] if col_idx < len(sig_e2.values) else None
                            if v1 is not None and v2 is not None and v1 != v2:
                                return (s1, s2, act, col)
        return None

    def is_consistent(self) -> bool:
        return self.find_inconsistency() is None

    # -- unfilled cells -----------------------------------------------------

    def unfilled_cells(self) -> List[Tuple[AccessSequence, Suffix]]:
        """Return all (row, column) pairs that have no observation."""
        result: List[Tuple[AccessSequence, Suffix]] = []
        for row in self.all_rows:
            for col in self._columns:
                if self._cells.get(row, col) is None:
                    result.append((row, col))
        return result

    def fill_ratio(self) -> float:
        total = len(self.all_rows) * len(self._columns)
        if total == 0:
            return 1.0
        filled = sum(
            1 for row in self.all_rows
            for col in self._columns
            if self._cells.get(row, col) is not None
        )
        return filled / total

    # -- statistics ---------------------------------------------------------

    @dataclass
    class Stats:
        short_row_count: int
        long_row_count: int
        column_count: int
        filled_cells: int
        total_cells: int
        fill_ratio: float
        distinct_signatures: int
        pool_size: int
        query_count: int
        update_count: int

    def stats(self) -> "ObservationTable.Stats":
        total = len(self.all_rows) * len(self._columns)
        filled = self._cells.filled_count()
        dist = len(self.distinct_short_signatures())
        return ObservationTable.Stats(
            short_row_count=len(self._short_rows),
            long_row_count=len(self._long_rows),
            column_count=len(self._columns),
            filled_cells=filled,
            total_cells=total,
            fill_ratio=self.fill_ratio(),
            distinct_signatures=dist,
            pool_size=self._cells.pool_size(),
            query_count=self._query_count,
            update_count=self._update_count,
        )

    # -- pretty printing ----------------------------------------------------

    def pretty_print(
        self,
        max_col_width: int = 24,
        max_rows: int = 60,
        file: Any = None,
    ) -> str:
        """Return a human-readable rendering of the table."""
        buf = io.StringIO()

        def _fmt_seq(seq: AccessSequence) -> str:
            if not seq:
                return "ε"
            return ".".join(seq)

        def _fmt_obs(obs: Optional[Observation]) -> str:
            if obs is None:
                return "?"
            parts = []
            for props, succ_tuple in sorted(obs):
                p_str = "{" + ",".join(sorted(props)) + "}"
                s_parts = []
                for act, targets in succ_tuple:
                    s_parts.append(f"{act}→{{{','.join(sorted(targets))}}}")
                s_str = "[" + "; ".join(s_parts) + "]"
                parts.append(f"{p_str}{s_str}")
            full = " | ".join(parts)
            if len(full) > max_col_width:
                return full[: max_col_width - 1] + "…"
            return full

        # Header
        col_headers = [_fmt_seq(c) for c in self._columns]
        col_widths = [max(len(h), 6) for h in col_headers]
        row_label_width = 20

        header_line = " " * row_label_width + " | ".join(
            h.ljust(w) for h, w in zip(col_headers, col_widths)
        )
        buf.write(header_line + "\n")
        buf.write("-" * len(header_line) + "\n")

        # Short rows
        displayed = 0
        for seq in self._short_rows:
            if displayed >= max_rows:
                buf.write(f"  ... ({len(self._short_rows) - displayed} more short rows)\n")
                break
            label = _fmt_seq(seq).ljust(row_label_width)[:row_label_width]
            cells = []
            for col, w in zip(self._columns, col_widths):
                cells.append(_fmt_obs(self._cells.get(seq, col)).ljust(w)[:w])
            buf.write(label + " | ".join(cells) + "\n")
            displayed += 1

        buf.write("-" * len(header_line) + "\n")

        # Long rows
        displayed = 0
        for seq in self._long_rows:
            if displayed >= max_rows:
                buf.write(f"  ... ({len(self._long_rows) - displayed} more long rows)\n")
                break
            label = _fmt_seq(seq).ljust(row_label_width)[:row_label_width]
            cells = []
            for col, w in zip(self._columns, col_widths):
                cells.append(_fmt_obs(self._cells.get(seq, col)).ljust(w)[:w])
            buf.write(label + " | ".join(cells) + "\n")
            displayed += 1

        result = buf.getvalue()
        if file is not None:
            file.write(result)
        return result

    # -- serialization for checkpointing ------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full table state to a JSON-friendly dict."""
        return {
            "actions": self._actions,
            "short_rows": [list(s) for s in self._short_rows],
            "long_rows": [list(s) for s in self._long_rows],
            "columns": [list(c) for c in self._columns],
            "cells": self._cells.to_dict(),
            "query_count": self._query_count,
            "update_count": self._update_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservationTable":
        actions = set(data["actions"])
        tbl = cls.__new__(cls)
        tbl._actions = sorted(actions)
        tbl._action_set = actions
        tbl._short_rows = OrderedDict()
        tbl._long_rows = OrderedDict()
        tbl._columns = []
        tbl._column_set = set()
        tbl._cells = _CellStore.from_dict(data.get("cells", {}))
        tbl._query_count = data.get("query_count", 0)
        tbl._update_count = data.get("update_count", 0)

        for s in data.get("short_rows", []):
            tbl._short_rows[tuple(s)] = None
        for s in data.get("long_rows", []):
            tbl._long_rows[tuple(s)] = None
        for c in data.get("columns", []):
            suffix = tuple(c)
            tbl._columns.append(suffix)
            tbl._column_set.add(suffix)
        return tbl

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, s: str) -> "ObservationTable":
        return cls.from_dict(json.loads(s))

    # -- compression --------------------------------------------------------

    def compress(self) -> "ObservationTable":
        """Return a compressed copy that removes redundant columns.

        A column is redundant if removing it does not change the
        equivalence-class structure of the short rows.
        """
        essential: List[Suffix] = []
        base_classes = self._partition_short_rows(self._columns)
        for col in self._columns:
            candidate = [c for c in essential if c != col]
            if not candidate and len(self._columns) > 1:
                candidate = [c for c in self._columns if c != col]
            trial_classes = self._partition_short_rows(candidate + [col])
            if trial_classes != base_classes:
                essential.append(col)
            elif col not in essential:
                # Check if already covered
                trial_without = self._partition_short_rows(
                    [c for c in essential]
                ) if essential else {0: list(self._short_rows.keys())}
                if trial_without != base_classes:
                    essential.append(col)

        if not essential:
            essential = list(self._columns[:1])

        compressed = ObservationTable(self._action_set)
        compressed._columns = essential
        compressed._column_set = set(essential)
        compressed._short_rows = OrderedDict(self._short_rows)
        compressed._long_rows = OrderedDict(self._long_rows)

        for row in compressed.all_rows:
            for col in essential:
                obs = self._cells.get(row, col)
                if obs is not None:
                    compressed._cells.set(row, col, obs)
        return compressed

    def _partition_short_rows(
        self, columns: List[Suffix]
    ) -> Dict[int, List[AccessSequence]]:
        """Partition short rows by their values on the given columns."""
        sig_map: Dict[Tuple[Optional[Observation], ...], int] = {}
        partition: Dict[int, List[AccessSequence]] = {}
        class_id = 0
        for sr in self._short_rows:
            sig = tuple(self._cells.get(sr, c) for c in columns)
            if sig not in sig_map:
                sig_map[sig] = class_id
                partition[class_id] = []
                class_id += 1
            partition[sig_map[sig]].append(sr)
        return partition

    # -- copy ---------------------------------------------------------------

    def copy(self) -> "ObservationTable":
        """Deep copy the table."""
        return ObservationTable.from_dict(self.to_dict())

    # -- iteration helpers --------------------------------------------------

    def iter_short_extensions(
        self, seq: AccessSequence
    ) -> Iterator[Tuple[str, AccessSequence]]:
        """Yield ``(action, extension)`` for each one-step extension."""
        for act in self._actions:
            yield act, seq + (act,)

    def missing_extensions(self) -> List[AccessSequence]:
        """Return short-row extensions not present as long rows."""
        missing: List[AccessSequence] = []
        for sr in self._short_rows:
            for act in self._actions:
                ext = sr + (act,)
                if ext not in self._short_rows and ext not in self._long_rows:
                    missing.append(ext)
        return missing

    def ensure_extensions(self) -> int:
        """Add any missing extensions as long rows. Return count added."""
        added = 0
        for ext in self.missing_extensions():
            self.add_long_row(ext)
            added += 1
        return added

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ObservationTable(short={s.short_row_count}, "
            f"long={s.long_row_count}, cols={s.column_count}, "
            f"fill={s.fill_ratio:.1%})"
        )
