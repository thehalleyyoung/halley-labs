"""
taintflow.instrument.context -- Thread-local instrumentation context management.

Provides the ``InstrumentationContext`` that threads through every
instrumented operation, plus helper registries for column lineage,
row provenance, and live DataFrame tracking.

Classes
-------
InstrumentationContext
    Thread-local context for tracking the current instrumentation state.
ColumnTracker
    Tracks column names (and their lineage) through DataFrame operations.
ProvenanceStore
    Stores roaring-bitmap provenance annotations for live DataFrames.
DataFrameRegistry
    Weak-reference registry of live DataFrames keyed by ``id(df)``.
"""

from __future__ import annotations

import inspect
import threading
import time
import uuid
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from taintflow.core.types import (
    ColumnSchema,
    OpType,
    Origin,
    ProvenanceInfo,
    ShapeMetadata,
)
from taintflow.core.errors import InstrumentationError
from taintflow.core.logging_utils import get_logger
from taintflow.dag.node import SourceLocation
from taintflow.dag.builder import TraceEvent
from taintflow.utils.bitmap import ProvenanceBitmap

logger = get_logger("taintflow.instrument.context")


# ===================================================================
#  Column lineage record
# ===================================================================

@dataclass
class ColumnLineageRecord:
    """One step in the lineage of a column."""

    operation: str
    source_column: str
    source_df_id: int
    timestamp: float = field(default_factory=time.monotonic)


# ===================================================================
#  ColumnTracker
# ===================================================================


class ColumnTracker:
    """Track column names (and their lineage) through DataFrame operations.

    Every live DataFrame ``id`` maps to the *set* of column names it
    currently holds.  When columns are renamed, merged, or dropped the
    tracker records the lineage so that downstream analysis can
    attribute leakage back to the original data source.
    """

    def __init__(self) -> None:
        self._columns: dict[int, set[str]] = {}
        self._lineage: dict[int, dict[str, list[ColumnLineageRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._lock = threading.Lock()

    # -- creation / registration -------------------------------------------

    def track_creation(self, df_id: int, columns: Iterable[str]) -> None:
        """Register a newly created DataFrame and its columns."""
        col_set = set(columns)
        with self._lock:
            self._columns[df_id] = col_set
            for col in col_set:
                self._lineage[df_id][col].append(
                    ColumnLineageRecord(
                        operation="create",
                        source_column=col,
                        source_df_id=df_id,
                    )
                )

    # -- rename ------------------------------------------------------------

    def track_rename(
        self, df_id: int, old: str, new: str, *, operation: str = "rename"
    ) -> None:
        """Record that *old* was renamed to *new* inside *df_id*."""
        with self._lock:
            cols = self._columns.get(df_id)
            if cols is None:
                return
            cols.discard(old)
            cols.add(new)
            prior = list(self._lineage[df_id].get(old, []))
            self._lineage[df_id][new] = prior + [
                ColumnLineageRecord(
                    operation=operation,
                    source_column=old,
                    source_df_id=df_id,
                )
            ]
            self._lineage[df_id].pop(old, None)

    def track_bulk_rename(
        self, df_id: int, mapping: Mapping[str, str], *, operation: str = "rename"
    ) -> None:
        """Rename multiple columns at once according to *mapping*."""
        for old_name, new_name in mapping.items():
            self.track_rename(df_id, old_name, new_name, operation=operation)

    # -- drop --------------------------------------------------------------

    def track_drop(self, df_id: int, columns: Iterable[str]) -> None:
        """Record that *columns* were dropped from *df_id*."""
        drop_set = set(columns)
        with self._lock:
            cols = self._columns.get(df_id)
            if cols is None:
                return
            cols -= drop_set
            for col in drop_set:
                self._lineage[df_id].pop(col, None)

    # -- add / assign ------------------------------------------------------

    def track_add(
        self,
        df_id: int,
        column: str,
        *,
        source_df_id: Optional[int] = None,
        source_column: Optional[str] = None,
        operation: str = "assign",
    ) -> None:
        """Record that *column* was added (or overwritten) in *df_id*."""
        with self._lock:
            cols = self._columns.setdefault(df_id, set())
            cols.add(column)
            rec = ColumnLineageRecord(
                operation=operation,
                source_column=source_column or column,
                source_df_id=source_df_id or df_id,
            )
            self._lineage[df_id][column].append(rec)

    # -- merge / join / concat ---------------------------------------------

    def track_merge(
        self,
        left_id: int,
        right_id: int,
        result_id: int,
        how: str = "inner",
    ) -> None:
        """Record a merge of *left_id* and *right_id* → *result_id*."""
        with self._lock:
            left_cols = self._columns.get(left_id, set())
            right_cols = self._columns.get(right_id, set())
            merged = left_cols | right_cols
            self._columns[result_id] = set(merged)

            for col in left_cols:
                self._lineage[result_id][col].append(
                    ColumnLineageRecord(
                        operation=f"merge_{how}",
                        source_column=col,
                        source_df_id=left_id,
                    )
                )
            for col in right_cols:
                self._lineage[result_id][col].append(
                    ColumnLineageRecord(
                        operation=f"merge_{how}",
                        source_column=col,
                        source_df_id=right_id,
                    )
                )

    def track_concat(
        self,
        source_ids: Sequence[int],
        result_id: int,
        axis: int = 0,
    ) -> None:
        """Record a concat of several DataFrames into *result_id*."""
        with self._lock:
            if axis == 0:
                all_cols: set[str] = set()
                for sid in source_ids:
                    all_cols |= self._columns.get(sid, set())
                self._columns[result_id] = all_cols
                for sid in source_ids:
                    for col in self._columns.get(sid, set()):
                        self._lineage[result_id][col].append(
                            ColumnLineageRecord(
                                operation="concat_row",
                                source_column=col,
                                source_df_id=sid,
                            )
                        )
            else:
                all_cols = set()
                for sid in source_ids:
                    for col in self._columns.get(sid, set()):
                        all_cols.add(col)
                        self._lineage[result_id][col].append(
                            ColumnLineageRecord(
                                operation="concat_col",
                                source_column=col,
                                source_df_id=sid,
                            )
                        )
                self._columns[result_id] = all_cols

    # -- copy / slice ------------------------------------------------------

    def track_copy(self, source_id: int, target_id: int) -> None:
        """Record that *target_id* is a copy of *source_id*."""
        with self._lock:
            src_cols = self._columns.get(source_id, set())
            self._columns[target_id] = set(src_cols)
            for col in src_cols:
                prior = list(self._lineage[source_id].get(col, []))
                self._lineage[target_id][col] = prior + [
                    ColumnLineageRecord(
                        operation="copy",
                        source_column=col,
                        source_df_id=source_id,
                    )
                ]

    def track_subset(
        self, source_id: int, target_id: int, columns: Iterable[str]
    ) -> None:
        """Record that *target_id* is a column-subset of *source_id*."""
        col_set = set(columns)
        with self._lock:
            self._columns[target_id] = set(col_set)
            for col in col_set:
                prior = list(self._lineage[source_id].get(col, []))
                self._lineage[target_id][col] = prior + [
                    ColumnLineageRecord(
                        operation="subset",
                        source_column=col,
                        source_df_id=source_id,
                    )
                ]

    # -- queries -----------------------------------------------------------

    def get_columns(self, df_id: int) -> set[str]:
        """Return the current set of columns for *df_id*."""
        with self._lock:
            return set(self._columns.get(df_id, set()))

    def get_lineage(
        self, df_id: int, column: str
    ) -> list[tuple[str, str]]:
        """Return lineage for a column as ``[(operation, source_column), ...]``."""
        with self._lock:
            records = self._lineage.get(df_id, {}).get(column, [])
            return [(r.operation, r.source_column) for r in records]

    def get_full_lineage(self, df_id: int, column: str) -> list[ColumnLineageRecord]:
        """Return the full lineage records for a column."""
        with self._lock:
            return list(self._lineage.get(df_id, {}).get(column, []))

    def has_df(self, df_id: int) -> bool:
        """Check whether *df_id* is tracked."""
        return df_id in self._columns

    def remove(self, df_id: int) -> None:
        """Remove tracking for *df_id*."""
        with self._lock:
            self._columns.pop(df_id, None)
            self._lineage.pop(df_id, None)

    def clear(self) -> None:
        """Drop all tracking data."""
        with self._lock:
            self._columns.clear()
            self._lineage.clear()


# ===================================================================
#  ProvenanceStore
# ===================================================================


class ProvenanceStore:
    """Store and propagate provenance bitmaps for DataFrames.

    Each live DataFrame is associated with a :class:`ProvenanceBitmap`
    whose set bits indicate which *global* row indices originated from the
    test split.  All propagation helpers produce a new bitmap for the
    result DataFrame while preserving the originals.
    """

    def __init__(self) -> None:
        self._store: dict[int, ProvenanceBitmap] = {}
        self._total_rows: dict[int, int] = {}
        self._lock = threading.Lock()

    # -- register / deregister ---------------------------------------------

    def register_dataframe(
        self, df_id: int, provenance: ProvenanceBitmap, total_rows: int = 0
    ) -> None:
        """Associate *df_id* with a provenance bitmap."""
        with self._lock:
            self._store[df_id] = provenance
            if total_rows > 0:
                self._total_rows[df_id] = total_rows
            else:
                self._total_rows[df_id] = provenance.cardinality()

    def get_provenance(self, df_id: int) -> Optional[ProvenanceBitmap]:
        """Return the provenance bitmap for *df_id*, or ``None``."""
        with self._lock:
            return self._store.get(df_id)

    def has(self, df_id: int) -> bool:
        return df_id in self._store

    def remove(self, df_id: int) -> None:
        with self._lock:
            self._store.pop(df_id, None)
            self._total_rows.pop(df_id, None)

    # -- propagation helpers -----------------------------------------------

    def propagate_filter(
        self, source_id: int, target_id: int, mask: Sequence[bool]
    ) -> None:
        """Propagate provenance through a boolean filter (row selection).

        Only the rows where *mask* is ``True`` are retained.
        """
        with self._lock:
            source_prov = self._store.get(source_id)
            if source_prov is None:
                return
            new_bm = ProvenanceBitmap()
            target_idx = 0
            for global_idx, keep in enumerate(mask):
                if keep:
                    if source_prov.contains(global_idx):
                        new_bm.add(target_idx)
                    target_idx += 1
            self._store[target_id] = new_bm
            self._total_rows[target_id] = target_idx

    def propagate_index_select(
        self, source_id: int, target_id: int, indices: Sequence[int]
    ) -> None:
        """Propagate provenance through an integer-index selection."""
        with self._lock:
            source_prov = self._store.get(source_id)
            if source_prov is None:
                return
            new_bm = ProvenanceBitmap()
            for new_idx, old_idx in enumerate(indices):
                if source_prov.contains(old_idx):
                    new_bm.add(new_idx)
            self._store[target_id] = new_bm
            self._total_rows[target_id] = len(indices)

    def propagate_merge(
        self,
        left_id: int,
        right_id: int,
        result_id: int,
        join_type: str = "inner",
        left_indices: Optional[Sequence[int]] = None,
        right_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Propagate provenance through a merge / join.

        The result row is considered *test-tainted* if **either** contributing
        row (left or right) was a test row.  When explicit index mappings are
        not provided we conservatively mark all result rows whose *left* or
        *right* source carried test provenance.
        """
        with self._lock:
            left_prov = self._store.get(left_id)
            right_prov = self._store.get(right_id)
            new_bm = ProvenanceBitmap()

            if left_indices is not None and right_indices is not None:
                n_result = len(left_indices)
                for i in range(n_result):
                    li = left_indices[i]
                    ri = right_indices[i]
                    left_test = left_prov is not None and li >= 0 and left_prov.contains(li)
                    right_test = right_prov is not None and ri >= 0 and right_prov.contains(ri)
                    if left_test or right_test:
                        new_bm.add(i)
                self._store[result_id] = new_bm
                self._total_rows[result_id] = n_result
            else:
                left_card = self._total_rows.get(left_id, 0)
                right_card = self._total_rows.get(right_id, 0)
                est_rows = max(left_card, right_card)
                left_frac = self.test_fraction(left_id)
                right_frac = self.test_fraction(right_id)
                combined_frac = 1.0 - (1.0 - left_frac) * (1.0 - right_frac)
                test_count = int(round(combined_frac * est_rows))
                for i in range(test_count):
                    new_bm.add(i)
                self._store[result_id] = new_bm
                self._total_rows[result_id] = est_rows

    def propagate_concat(
        self, source_ids: Sequence[int], target_id: int, axis: int = 0
    ) -> None:
        """Propagate provenance through a concatenation."""
        with self._lock:
            new_bm = ProvenanceBitmap()
            if axis == 0:
                offset = 0
                total = 0
                for sid in source_ids:
                    prov = self._store.get(sid)
                    n = self._total_rows.get(sid, 0)
                    if prov is not None:
                        for row in range(n):
                            if prov.contains(row):
                                new_bm.add(offset + row)
                    offset += n
                    total += n
                self._store[target_id] = new_bm
                self._total_rows[target_id] = total
            else:
                first = source_ids[0] if source_ids else None
                if first is not None:
                    prov = self._store.get(first)
                    n = self._total_rows.get(first, 0)
                    if prov is not None:
                        for row in range(n):
                            if prov.contains(row):
                                new_bm.add(row)
                    self._store[target_id] = new_bm
                    self._total_rows[target_id] = n

    def propagate_sort(self, source_id: int, target_id: int) -> None:
        """Sort does not change provenance — copy directly."""
        with self._lock:
            prov = self._store.get(source_id)
            if prov is not None:
                self._store[target_id] = prov
                self._total_rows[target_id] = self._total_rows.get(source_id, 0)

    def propagate_copy(self, source_id: int, target_id: int) -> None:
        """Copy provenance from one df_id to another."""
        self.propagate_sort(source_id, target_id)

    # -- queries -----------------------------------------------------------

    def test_fraction(self, df_id: int) -> float:
        """Return ρ — the fraction of test rows in *df_id*.

        Returns 0.0 if *df_id* is not tracked or has no rows.
        """
        with self._lock:
            prov = self._store.get(df_id)
            total = self._total_rows.get(df_id, 0)
            if prov is None or total <= 0:
                return 0.0
            test_count = prov.cardinality()
            return min(test_count / total, 1.0)

    def total_rows(self, df_id: int) -> int:
        with self._lock:
            return self._total_rows.get(df_id, 0)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._total_rows.clear()


# ===================================================================
#  DataFrameRegistry
# ===================================================================


class DataFrameRegistry:
    """Weak-reference registry of live DataFrames keyed by ``id(df)``.

    We store weak references so that DataFrames can be garbage-collected
    normally; the registry automatically cleans up stale entries.
    """

    def __init__(self) -> None:
        self._refs: dict[int, weakref.ref] = {}
        self._shapes: dict[int, tuple[int, int]] = {}
        self._dtypes: dict[int, dict[str, str]] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _cleanup(self, df_id: int) -> None:
        """Called when a weakref is collected."""
        with self._lock:
            self._refs.pop(df_id, None)
            self._shapes.pop(df_id, None)
            self._dtypes.pop(df_id, None)
            self._metadata.pop(df_id, None)

    def register(
        self,
        df: Any,
        *,
        shape: Optional[tuple[int, int]] = None,
        dtypes: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Register a DataFrame and return its ``id``."""
        df_id = id(df)
        with self._lock:
            try:
                self._refs[df_id] = weakref.ref(df, lambda _ref: self._cleanup(df_id))
            except TypeError:
                self._refs[df_id] = None  # type: ignore[assignment]

            if shape is not None:
                self._shapes[df_id] = shape
            else:
                try:
                    self._shapes[df_id] = (df.shape[0], df.shape[1])
                except (AttributeError, IndexError, TypeError):
                    self._shapes[df_id] = (0, 0)

            if dtypes is not None:
                self._dtypes[df_id] = dtypes
            else:
                try:
                    self._dtypes[df_id] = {str(c): str(t) for c, t in df.dtypes.items()}
                except (AttributeError, TypeError):
                    self._dtypes[df_id] = {}

            self._metadata[df_id] = metadata or {}
        return df_id

    def deregister(self, df_id: int) -> None:
        """Remove *df_id* from the registry."""
        with self._lock:
            self._refs.pop(df_id, None)
            self._shapes.pop(df_id, None)
            self._dtypes.pop(df_id, None)
            self._metadata.pop(df_id, None)

    def get_shape(self, df_id: int) -> tuple[int, int]:
        """Return ``(n_rows, n_cols)`` for *df_id*."""
        with self._lock:
            return self._shapes.get(df_id, (0, 0))

    def get_dtypes(self, df_id: int) -> dict[str, str]:
        """Return ``{col_name: dtype_str}`` for *df_id*."""
        with self._lock:
            return dict(self._dtypes.get(df_id, {}))

    def lookup_by_id(self, df_id: int) -> Optional[Any]:
        """Dereference the weakref for *df_id*, or ``None`` if collected."""
        with self._lock:
            ref = self._refs.get(df_id)
            if ref is None:
                return None
            if callable(ref):
                return ref()
            return None

    def contains(self, df_id: int) -> bool:
        return df_id in self._refs

    def all_ids(self) -> list[int]:
        with self._lock:
            return list(self._refs.keys())

    def clear(self) -> None:
        with self._lock:
            self._refs.clear()
            self._shapes.clear()
            self._dtypes.clear()
            self._metadata.clear()


# ===================================================================
#  InstrumentationContext
# ===================================================================

_context_local = threading.local()


class InstrumentationContext:
    """Thread-local context for tracking the current instrumentation state.

    An ``InstrumentationContext`` is created once per audited pipeline run.
    It is passed to the tracer, monkey-patcher, and every hook so that
    they can record trace events, track column lineage, and propagate
    row provenance through a shared, thread-safe state object.

    Usage::

        ctx = InstrumentationContext()
        with ctx:
            # all instrumented calls see this context via get_current_context()
            ...
    """

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        source_file: str = "",
        max_depth: int = 50,
        max_events: int = 500_000,
    ) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self.source_file: str = source_file
        self.active: bool = False
        self.depth: int = 0
        self._max_depth: int = max_depth
        self._max_events: int = max_events

        self.current_operation: Optional[str] = None
        self.current_module: str = ""

        self.trace_events: list[TraceEvent] = []
        self.operation_stack: list[str] = []
        self._op_stack_detail: list[dict[str, Any]] = []

        self.column_tracker: ColumnTracker = ColumnTracker()
        self.provenance_store: ProvenanceStore = ProvenanceStore()
        self.df_registry: DataFrameRegistry = DataFrameRegistry()

        self._start_time: float = 0.0
        self._event_count: int = 0
        self._dropped_events: int = 0
        self._lock = threading.Lock()

    # -- operation stack ---------------------------------------------------

    def push_operation(
        self,
        operation: str,
        *,
        module: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Push an operation onto the call stack."""
        with self._lock:
            self.operation_stack.append(operation)
            self._op_stack_detail.append({
                "operation": operation,
                "module": module,
                "depth": self.depth,
                "timestamp": time.monotonic(),
                "metadata": metadata or {},
            })
            self.depth = len(self.operation_stack)
            self.current_operation = operation
            self.current_module = module

    def pop_operation(self) -> Optional[str]:
        """Pop the most recent operation off the call stack."""
        with self._lock:
            if not self.operation_stack:
                return None
            op = self.operation_stack.pop()
            self._op_stack_detail.pop()
            self.depth = len(self.operation_stack)
            if self.operation_stack:
                self.current_operation = self.operation_stack[-1]
                self.current_module = self._op_stack_detail[-1].get("module", "")
            else:
                self.current_operation = None
                self.current_module = ""
            return op

    def current_stack(self) -> list[str]:
        """Return a snapshot of the current operation stack."""
        with self._lock:
            return list(self.operation_stack)

    # -- event recording ---------------------------------------------------

    def record_event(self, event: TraceEvent) -> bool:
        """Append a :class:`TraceEvent`.

        Returns ``True`` if the event was recorded, ``False`` if the
        maximum event count has been reached.
        """
        with self._lock:
            if self._event_count >= self._max_events:
                self._dropped_events += 1
                return False
            self.trace_events.append(event)
            self._event_count += 1
            return True

    def record_event_from_parts(
        self,
        *,
        event_type: str = "call",
        function: str = "",
        module: str = "",
        class_name: str = "",
        file: str = "",
        line: int = 0,
        args_schema: Optional[list[dict[str, Any]]] = None,
        return_schema: Optional[list[dict[str, Any]]] = None,
        args_shape: Optional[tuple[int, ...]] = None,
        return_shape: Optional[tuple[int, ...]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Build a :class:`TraceEvent` from keyword arguments and record it."""
        event = TraceEvent(
            timestamp=time.monotonic(),
            event_type=event_type,
            function=function,
            module=module,
            class_name=class_name,
            file=file,
            line=line,
            args_schema=args_schema or [],
            return_schema=return_schema or [],
            args_shape=args_shape,
            return_shape=return_shape,
            metadata=metadata or {},
        )
        return self.record_event(event)

    # -- source location ---------------------------------------------------

    def get_source_location(self, stack_offset: int = 2) -> SourceLocation:
        """Inspect the Python call stack and return a :class:`SourceLocation`.

        *stack_offset* controls how many frames to skip (default 2 skips
        this method and its caller).
        """
        try:
            frame = inspect.stack()[stack_offset]
            return SourceLocation(
                file=frame.filename,
                line=frame.lineno,
                function_name=frame.function,
                class_name="",
            )
        except (IndexError, AttributeError):
            return SourceLocation()

    # -- context manager protocol ------------------------------------------

    def __enter__(self) -> "InstrumentationContext":
        self.active = True
        self._start_time = time.monotonic()
        _context_local.current = self
        logger.debug(f"InstrumentationContext {self.session_id} activated")
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        self.active = False
        elapsed = time.monotonic() - self._start_time
        _context_local.current = None
        logger.debug(
            f"InstrumentationContext {self.session_id} deactivated after "
            f"{elapsed:.3f}s — {self._event_count} events "
            f"({self._dropped_events} dropped)"
        )
        return False

    # -- helper queries ----------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds since the context was activated."""
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def at_max_depth(self) -> bool:
        return self.depth >= self._max_depth

    def summary(self) -> dict[str, Any]:
        """Return a diagnostic summary of the current context state."""
        return {
            "session_id": self.session_id,
            "active": self.active,
            "depth": self.depth,
            "events_recorded": self._event_count,
            "events_dropped": self._dropped_events,
            "operation_stack": list(self.operation_stack),
            "elapsed_s": round(self.elapsed, 3),
        }

    def reset(self) -> None:
        """Clear all state — useful between test runs."""
        with self._lock:
            self.active = False
            self.depth = 0
            self.current_operation = None
            self.current_module = ""
            self.trace_events.clear()
            self.operation_stack.clear()
            self._op_stack_detail.clear()
            self._event_count = 0
            self._dropped_events = 0
            self._start_time = 0.0
            self.column_tracker.clear()
            self.provenance_store.clear()
            self.df_registry.clear()


# ===================================================================
#  Module-level helpers
# ===================================================================


def get_current_context() -> Optional[InstrumentationContext]:
    """Return the thread-local :class:`InstrumentationContext`, or ``None``."""
    return getattr(_context_local, "current", None)


def require_context() -> InstrumentationContext:
    """Return the current context or raise :class:`InstrumentationError`."""
    ctx = get_current_context()
    if ctx is None:
        raise InstrumentationError(
            "No active InstrumentationContext — "
            "wrap your pipeline in `with InstrumentationContext(): ...`"
        )
    return ctx
