"""
taintflow.instrument.tracer – sys.settrace-based instrumentation for
dynamic DAG extraction.

Captures call graphs, function-level events, and lightweight local-variable
snapshots while an ML pipeline executes under instrumentation.  The trace
is buffered, thread-safe, and exportable to JSON for downstream DAG
construction.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from taintflow.core.types import OpType, Origin, ProvenanceInfo, ShapeMetadata
from taintflow.dag.nodes import DAGNode, NodeFactory, SourceLocation


# ===================================================================
#  Constants
# ===================================================================

_DEFAULT_BUFFER_LIMIT: int = 50_000
_SAFE_REPR_LIMIT: int = 256
_OVERHEAD_SAMPLE_INTERVAL: int = 500

# Standard library / CPython paths that should never be traced.
_STDLIB_PREFIXES: Tuple[str, ...] = (
    "<frozen",
    "<string>",
    "importlib",
    "abc",
    "posixpath",
    "genericpath",
    "os.path",
    "_bootstrap",
    "sre_",
    "codecs",
    "encodings",
    "io",
    "zipimport",
)


# ===================================================================
#  Enumerations
# ===================================================================

class TraceEventType(Enum):
    """Types of events captured by the tracer."""

    CALL = auto()
    RETURN = auto()
    EXCEPTION = auto()
    C_CALL = auto()
    C_RETURN = auto()
    C_EXCEPTION = auto()
    LINE = auto()

    @classmethod
    def from_settrace(cls, event: str) -> "TraceEventType":
        """Map a ``sys.settrace`` event string to the enum."""
        mapping = {
            "call": cls.CALL,
            "return": cls.RETURN,
            "exception": cls.EXCEPTION,
            "c_call": cls.C_CALL,
            "c_return": cls.C_RETURN,
            "c_exception": cls.C_EXCEPTION,
            "line": cls.LINE,
        }
        return mapping.get(event, cls.CALL)


# ===================================================================
#  Dataclasses
# ===================================================================

@dataclass
class TraceEvent:
    """A single captured trace event.

    Attributes:
        timestamp: Monotonic timestamp (seconds since tracer start).
        event_type: Kind of trace event.
        filename: Source file where the event occurred.
        lineno: Line number in the source file.
        function_name: Name of the function / method.
        class_name: Enclosing class, if determinable.
        locals_snapshot: Lightweight snapshot of selected local variables.
        return_value_repr: ``repr`` of the return value (for RETURN events).
        call_depth: Nesting depth of the call stack.
        thread_name: Name of the thread that produced the event.
        event_id: Unique event identifier.
    """

    timestamp: float
    event_type: TraceEventType
    filename: str
    lineno: int
    function_name: str
    class_name: str = ""
    locals_snapshot: Dict[str, str] = field(default_factory=dict)
    return_value_repr: str = ""
    call_depth: int = 0
    thread_name: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.name,
            "filename": self.filename,
            "lineno": self.lineno,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "locals_snapshot": self.locals_snapshot,
            "return_value_repr": self.return_value_repr,
            "call_depth": self.call_depth,
            "thread_name": self.thread_name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceEvent":
        """Deserialize from a dictionary."""
        return cls(
            timestamp=float(d["timestamp"]),
            event_type=TraceEventType[d["event_type"]],
            filename=d["filename"],
            lineno=int(d["lineno"]),
            function_name=d["function_name"],
            class_name=d.get("class_name", ""),
            locals_snapshot=d.get("locals_snapshot", {}),
            return_value_repr=d.get("return_value_repr", ""),
            call_depth=int(d.get("call_depth", 0)),
            thread_name=d.get("thread_name", ""),
            event_id=d.get("event_id", uuid.uuid4().hex[:12]),
        )


@dataclass
class PerformanceMetrics:
    """Overhead measurements for the tracing session.

    Attributes:
        total_events: Total number of trace events captured.
        dropped_events: Number of events dropped due to buffer limits.
        trace_start: Monotonic time when tracing started.
        trace_end: Monotonic time when tracing stopped.
        overhead_samples: List of per-event overhead measurements (ns).
        peak_buffer_size: Maximum number of events buffered at once.
    """

    total_events: int = 0
    dropped_events: int = 0
    trace_start: float = 0.0
    trace_end: float = 0.0
    overhead_samples: List[float] = field(default_factory=list)
    peak_buffer_size: int = 0

    @property
    def wall_time_s(self) -> float:
        """Total wall-clock time of the tracing session."""
        return self.trace_end - self.trace_start if self.trace_end > self.trace_start else 0.0

    @property
    def mean_overhead_ns(self) -> float:
        """Mean per-event overhead in nanoseconds."""
        if not self.overhead_samples:
            return 0.0
        return sum(self.overhead_samples) / len(self.overhead_samples)

    @property
    def max_overhead_ns(self) -> float:
        """Maximum per-event overhead in nanoseconds."""
        return max(self.overhead_samples) if self.overhead_samples else 0.0

    @property
    def events_per_second(self) -> float:
        """Throughput: events captured per second."""
        if self.wall_time_s <= 0:
            return 0.0
        return self.total_events / self.wall_time_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "dropped_events": self.dropped_events,
            "wall_time_s": round(self.wall_time_s, 6),
            "mean_overhead_ns": round(self.mean_overhead_ns, 2),
            "max_overhead_ns": round(self.max_overhead_ns, 2),
            "events_per_second": round(self.events_per_second, 1),
            "peak_buffer_size": self.peak_buffer_size,
        }


# ===================================================================
#  FilterConfig
# ===================================================================

@dataclass
class FilterConfig:
    """Controls which modules and functions the tracer captures.

    Attributes:
        include_prefixes: Module-path prefixes to trace (e.g. ``sklearn.``,
            ``pandas.``).  If non-empty only these are traced.
        exclude_prefixes: Module-path prefixes to unconditionally skip.
        include_functions: Specific function names to always trace.
        exclude_functions: Specific function names to always skip.
        trace_line_events: Whether to emit LINE events (expensive).
        trace_c_calls: Whether to trace C-extension calls.
        max_locals: Maximum number of local variables to snapshot per event.
        max_repr_len: Maximum ``repr`` length for local variable values.
    """

    include_prefixes: Tuple[str, ...] = (
        "sklearn.",
        "pandas.",
        "numpy.",
        "scipy.",
        "xgboost.",
        "lightgbm.",
        "catboost.",
    )
    exclude_prefixes: Tuple[str, ...] = _STDLIB_PREFIXES + (
        "taintflow.instrument.",
        "threading.",
        "logging.",
        "unittest.",
        "pytest.",
        "_pytest.",
        "pkg_resources.",
        "setuptools.",
    )
    include_functions: FrozenSet[str] = frozenset({
        "fit", "transform", "fit_transform", "predict", "predict_proba",
        "score", "decision_function", "inverse_transform", "fit_predict",
        "merge", "concat", "groupby", "apply", "agg", "aggregate",
        "read_csv", "read_parquet", "read_json", "read_excel",
        "train_test_split", "cross_val_score", "cross_validate",
        "get_dummies", "fillna", "dropna", "pivot", "melt",
    })
    exclude_functions: FrozenSet[str] = frozenset({
        "__repr__", "__str__", "__hash__", "__eq__", "__ne__",
        "__lt__", "__le__", "__gt__", "__ge__", "__len__",
        "__contains__", "__iter__", "__next__", "__bool__",
        "__getattr__", "__setattr__", "__delattr__",
    })
    trace_line_events: bool = False
    trace_c_calls: bool = False
    max_locals: int = 20
    max_repr_len: int = _SAFE_REPR_LIMIT

    def should_trace_file(self, filename: str) -> bool:
        """Return True if *filename* passes the include/exclude filters."""
        if not filename:
            return False
        for prefix in self.exclude_prefixes:
            if prefix in filename:
                return False
        if self.include_prefixes:
            return any(prefix in filename for prefix in self.include_prefixes)
        return True

    def should_trace_function(self, function_name: str) -> bool:
        """Return True if *function_name* passes the include/exclude filters."""
        if function_name in self.exclude_functions:
            return False
        if self.include_functions:
            return function_name in self.include_functions
        return True

    def accepts(self, filename: str, function_name: str) -> bool:
        """Combined filter: True when both file and function pass."""
        return self.should_trace_file(filename) and self.should_trace_function(function_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "include_prefixes": list(self.include_prefixes),
            "exclude_prefixes": list(self.exclude_prefixes),
            "include_functions": sorted(self.include_functions),
            "exclude_functions": sorted(self.exclude_functions),
            "trace_line_events": self.trace_line_events,
            "trace_c_calls": self.trace_c_calls,
            "max_locals": self.max_locals,
            "max_repr_len": self.max_repr_len,
        }


# ===================================================================
#  FrameInspector
# ===================================================================

class FrameInspector:
    """Safely extracts information from CPython stack frames.

    All access is wrapped in try/except to handle frames that have been
    partially torn down or belong to C extensions.
    """

    def __init__(self, max_locals: int = 20, max_repr_len: int = _SAFE_REPR_LIMIT) -> None:
        self._max_locals = max_locals
        self._max_repr_len = max_repr_len

    # ------------------------------------------------------------------
    #  Basic frame attributes
    # ------------------------------------------------------------------

    @staticmethod
    def get_filename(frame: Any) -> str:
        """Return the filename from *frame*, or ``"<unknown>"``."""
        try:
            return frame.f_code.co_filename or "<unknown>"
        except (AttributeError, ValueError):
            return "<unknown>"

    @staticmethod
    def get_lineno(frame: Any) -> int:
        """Return the current line number from *frame*."""
        try:
            return frame.f_lineno
        except (AttributeError, ValueError):
            return 0

    @staticmethod
    def get_function_name(frame: Any) -> str:
        """Return the function name from *frame*."""
        try:
            return frame.f_code.co_name or "<unknown>"
        except (AttributeError, ValueError):
            return "<unknown>"

    @staticmethod
    def get_class_name(frame: Any) -> str:
        """Try to determine the enclosing class name from *frame*'s locals.

        Looks for ``self`` or ``cls`` in locals and extracts the class name
        from the object.
        """
        try:
            local_vars = frame.f_locals
            if "self" in local_vars:
                obj = local_vars["self"]
                return type(obj).__name__
            if "cls" in local_vars:
                cls_obj = local_vars["cls"]
                if isinstance(cls_obj, type):
                    return cls_obj.__name__
        except (AttributeError, ValueError, TypeError):
            pass
        return ""

    def snapshot_locals(self, frame: Any) -> Dict[str, str]:
        """Capture a safe repr-snapshot of the frame's local variables.

        Only the first ``max_locals`` variables are captured and each
        repr is truncated to ``max_repr_len`` characters.
        """
        result: Dict[str, str] = {}
        try:
            items = list(frame.f_locals.items())
        except (AttributeError, ValueError, TypeError):
            return result

        for key, value in items[: self._max_locals]:
            result[key] = self._safe_repr(value)
        return result

    def get_call_depth(self, frame: Any) -> int:
        """Count the stack depth by following ``f_back`` links."""
        depth = 0
        current = frame
        try:
            while current is not None:
                current = current.f_back
                depth += 1
                if depth > 500:
                    break
        except (AttributeError, ValueError):
            pass
        return depth

    def extract_return_repr(self, arg: Any) -> str:
        """Safely produce a repr for a return value."""
        return self._safe_repr(arg)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _safe_repr(self, obj: Any) -> str:
        """Return a truncated repr that never raises."""
        try:
            r = repr(obj)
        except Exception:
            r = f"<{type(obj).__name__} (repr failed)>"
        if len(r) > self._max_repr_len:
            return r[: self._max_repr_len - 3] + "..."
        return r


# ===================================================================
#  TraceRecorder
# ===================================================================

class TraceRecorder:
    """Accumulates :class:`TraceEvent` instances into a structured log.

    Thread-safe: events from multiple threads are interleaved and can be
    separated later by ``thread_name``.  A configurable buffer limit
    prevents unbounded memory growth; once exceeded, the oldest events
    are silently dropped.
    """

    def __init__(self, buffer_limit: int = _DEFAULT_BUFFER_LIMIT) -> None:
        self._buffer_limit = buffer_limit
        self._events: List[TraceEvent] = []
        self._lock = threading.Lock()
        self._dropped: int = 0
        self._peak_size: int = 0

    # ------------------------------------------------------------------
    #  Recording
    # ------------------------------------------------------------------

    def record(self, event: TraceEvent) -> None:
        """Append *event* to the buffer, dropping the oldest if full."""
        with self._lock:
            if len(self._events) >= self._buffer_limit:
                self._events.pop(0)
                self._dropped += 1
            self._events.append(event)
            if len(self._events) > self._peak_size:
                self._peak_size = len(self._events)

    def record_batch(self, events: Sequence[TraceEvent]) -> None:
        """Append multiple events atomically."""
        with self._lock:
            for ev in events:
                if len(self._events) >= self._buffer_limit:
                    self._events.pop(0)
                    self._dropped += 1
                self._events.append(ev)
            if len(self._events) > self._peak_size:
                self._peak_size = len(self._events)

    # ------------------------------------------------------------------
    #  Querying
    # ------------------------------------------------------------------

    @property
    def events(self) -> List[TraceEvent]:
        """Return a snapshot of all buffered events."""
        with self._lock:
            return list(self._events)

    @property
    def dropped_count(self) -> int:
        return self._dropped

    @property
    def peak_size(self) -> int:
        return self._peak_size

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def filter_by_type(self, event_type: TraceEventType) -> List[TraceEvent]:
        """Return events of a specific type."""
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]

    def filter_by_function(self, function_name: str) -> List[TraceEvent]:
        """Return events for a specific function name."""
        with self._lock:
            return [e for e in self._events if e.function_name == function_name]

    def filter_by_file(self, filename_substring: str) -> List[TraceEvent]:
        """Return events whose filename contains *filename_substring*."""
        with self._lock:
            return [e for e in self._events if filename_substring in e.filename]

    def filter_by_thread(self, thread_name: str) -> List[TraceEvent]:
        """Return events from a specific thread."""
        with self._lock:
            return [e for e in self._events if e.thread_name == thread_name]

    def calls_only(self) -> List[TraceEvent]:
        """Return only CALL events."""
        return self.filter_by_type(TraceEventType.CALL)

    def returns_only(self) -> List[TraceEvent]:
        """Return only RETURN events."""
        return self.filter_by_type(TraceEventType.RETURN)

    # ------------------------------------------------------------------
    #  Export
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Export all events as a JSON string."""
        with self._lock:
            data = {
                "n_events": len(self._events),
                "dropped": self._dropped,
                "peak_buffer_size": self._peak_size,
                "events": [e.to_dict() for e in self._events],
            }
        return json.dumps(data, indent=indent, default=str)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Export all events as a list of dictionaries."""
        with self._lock:
            return [e.to_dict() for e in self._events]

    def clear(self) -> None:
        """Discard all buffered events."""
        with self._lock:
            self._events.clear()


# ===================================================================
#  CallGraphBuilder
# ===================================================================

@dataclass(frozen=True)
class CallEdge:
    """An edge in the call graph: caller -> callee."""

    caller_file: str
    caller_function: str
    caller_class: str
    callee_file: str
    callee_function: str
    callee_class: str
    count: int = 1

    @property
    def caller_label(self) -> str:
        prefix = f"{self.caller_class}." if self.caller_class else ""
        return f"{prefix}{self.caller_function}"

    @property
    def callee_label(self) -> str:
        prefix = f"{self.callee_class}." if self.callee_class else ""
        return f"{prefix}{self.callee_function}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caller": self.caller_label,
            "callee": self.callee_label,
            "caller_file": self.caller_file,
            "callee_file": self.callee_file,
            "count": self.count,
        }


class CallGraphBuilder:
    """Builds an aggregated call graph from a stream of trace events.

    The graph is stored as a dictionary mapping (caller, callee) tuples
    to call counts.  Each node is identified by ``(filename, class,
    function_name)``.
    """

    def __init__(self) -> None:
        self._edges: Dict[Tuple[str, str, str, str, str, str], int] = {}
        self._nodes: Set[Tuple[str, str, str]] = set()
        self._call_stack: List[TraceEvent] = []

    def process_event(self, event: TraceEvent) -> None:
        """Update the call graph with a single trace event."""
        node = (event.filename, event.class_name, event.function_name)
        self._nodes.add(node)

        if event.event_type == TraceEventType.CALL:
            if self._call_stack:
                caller = self._call_stack[-1]
                edge_key = (
                    caller.filename,
                    caller.class_name,
                    caller.function_name,
                    event.filename,
                    event.class_name,
                    event.function_name,
                )
                self._edges[edge_key] = self._edges.get(edge_key, 0) + 1
            self._call_stack.append(event)
        elif event.event_type == TraceEventType.RETURN:
            if self._call_stack:
                self._call_stack.pop()

    def process_events(self, events: Sequence[TraceEvent]) -> None:
        """Process a batch of events."""
        for event in events:
            self.process_event(event)

    @property
    def nodes(self) -> Set[Tuple[str, str, str]]:
        """Return the set of (filename, class, function) nodes."""
        return set(self._nodes)

    @property
    def edges(self) -> List[CallEdge]:
        """Return all edges as :class:`CallEdge` instances."""
        result: List[CallEdge] = []
        for (cf, cc, cfn, ef, ec, efn), count in self._edges.items():
            result.append(CallEdge(
                caller_file=cf,
                caller_class=cc,
                caller_function=cfn,
                callee_file=ef,
                callee_class=ec,
                callee_function=efn,
                count=count,
            ))
        return result

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    def callees_of(self, function_name: str) -> List[CallEdge]:
        """Return all edges where *function_name* is the caller."""
        return [e for e in self.edges if e.caller_function == function_name]

    def callers_of(self, function_name: str) -> List[CallEdge]:
        """Return all edges where *function_name* is the callee."""
        return [e for e in self.edges if e.callee_function == function_name]

    def to_dict(self) -> Dict[str, Any]:
        """Export the call graph as a JSON-compatible dictionary."""
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "nodes": [
                {"file": f, "class": c, "function": fn}
                for f, c, fn in sorted(self._nodes)
            ],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export the call graph as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._edges.clear()
        self._nodes.clear()
        self._call_stack.clear()


# ===================================================================
#  PipelineTracer
# ===================================================================

class PipelineTracer:
    """``sys.settrace``-based tracer for ML pipeline execution.

    Usage::

        with PipelineTracer(filter_config=my_config) as tracer:
            run_my_pipeline()
        events = tracer.recorder.events
        graph  = tracer.call_graph

    Thread-safety is achieved via :class:`threading.local`: each thread
    gets its own call-depth counter and stack, while all threads share
    the :class:`TraceRecorder`.
    """

    def __init__(
        self,
        filter_config: Optional[FilterConfig] = None,
        buffer_limit: int = _DEFAULT_BUFFER_LIMIT,
        record_locals: bool = True,
    ) -> None:
        self._filter = filter_config or FilterConfig()
        self._recorder = TraceRecorder(buffer_limit=buffer_limit)
        self._call_graph = CallGraphBuilder()
        self._inspector = FrameInspector(
            max_locals=self._filter.max_locals,
            max_repr_len=self._filter.max_repr_len,
        )
        self._record_locals = record_locals
        self._metrics = PerformanceMetrics()
        self._active = False
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._old_trace: Optional[Callable[..., Any]] = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def recorder(self) -> TraceRecorder:
        """The :class:`TraceRecorder` accumulating events."""
        return self._recorder

    @property
    def call_graph(self) -> CallGraphBuilder:
        """The :class:`CallGraphBuilder` aggregating call edges."""
        return self._call_graph

    @property
    def metrics(self) -> PerformanceMetrics:
        """Performance metrics for the tracing session."""
        return self._metrics

    @property
    def is_active(self) -> bool:
        """Whether the tracer is currently installed."""
        return self._active

    @property
    def filter_config(self) -> FilterConfig:
        """The active filter configuration."""
        return self._filter

    # ------------------------------------------------------------------
    #  Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "PipelineTracer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    #  Start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Install the trace function via ``sys.settrace``."""
        if self._active:
            return
        self._old_trace = sys.gettrace()
        self._start_time = time.monotonic()
        self._metrics.trace_start = self._start_time
        self._active = True
        sys.settrace(self._trace_dispatch)
        threading.settrace(self._trace_dispatch)

    def stop(self) -> None:
        """Remove the trace function and finalise metrics."""
        if not self._active:
            return
        sys.settrace(self._old_trace)
        threading.settrace(self._old_trace or (lambda *_: None))  # type: ignore[arg-type]
        self._active = False
        self._metrics.trace_end = time.monotonic()
        self._metrics.total_events = len(self._recorder)
        self._metrics.dropped_events = self._recorder.dropped_count
        self._metrics.peak_buffer_size = self._recorder.peak_size

        # Build call graph from recorded events
        self._call_graph.process_events(self._recorder.events)

    # ------------------------------------------------------------------
    #  Trace dispatch (hot path – kept minimal)
    # ------------------------------------------------------------------

    def _trace_dispatch(self, frame: Any, event: str, arg: Any) -> Optional[Callable[..., Any]]:
        """The function installed via ``sys.settrace``.

        This is called for every Python event in every traced thread, so
        it must be as fast as possible.  Filtering and recording are
        delegated to helpers.
        """
        if not self._active:
            return None

        t0 = time.monotonic_ns() if (self._metrics.total_events % _OVERHEAD_SAMPLE_INTERVAL == 0) else 0

        filename = self._inspector.get_filename(frame)
        function_name = self._inspector.get_function_name(frame)

        if not self._filter.trace_line_events and event == "line":
            return self._trace_dispatch

        if not self._filter.trace_c_calls and event in ("c_call", "c_return", "c_exception"):
            return self._trace_dispatch

        if not self._filter.accepts(filename, function_name):
            return self._trace_dispatch

        self._record_event(frame, event, arg, filename, function_name)

        if t0:
            elapsed_ns = float(time.monotonic_ns() - t0)
            self._metrics.overhead_samples.append(elapsed_ns)

        return self._trace_dispatch

    # ------------------------------------------------------------------
    #  Event recording
    # ------------------------------------------------------------------

    def _record_event(
        self,
        frame: Any,
        event: str,
        arg: Any,
        filename: str,
        function_name: str,
    ) -> None:
        """Create a :class:`TraceEvent` and hand it to the recorder."""
        event_type = TraceEventType.from_settrace(event)
        class_name = self._inspector.get_class_name(frame)
        lineno = self._inspector.get_lineno(frame)
        timestamp = time.monotonic() - self._start_time

        locals_snapshot: Dict[str, str] = {}
        if self._record_locals and event_type == TraceEventType.CALL:
            locals_snapshot = self._inspector.snapshot_locals(frame)

        return_repr = ""
        if event_type == TraceEventType.RETURN and arg is not None:
            return_repr = self._inspector.extract_return_repr(arg)

        call_depth = getattr(self._tls, "depth", 0)
        if event_type == TraceEventType.CALL:
            self._tls.depth = call_depth + 1
        elif event_type == TraceEventType.RETURN:
            self._tls.depth = max(0, call_depth - 1)

        trace_event = TraceEvent(
            timestamp=timestamp,
            event_type=event_type,
            filename=filename,
            lineno=lineno,
            function_name=function_name,
            class_name=class_name,
            locals_snapshot=locals_snapshot,
            return_value_repr=return_repr,
            call_depth=call_depth,
            thread_name=threading.current_thread().name,
        )

        self._recorder.record(trace_event)

    # ------------------------------------------------------------------
    #  Export helpers
    # ------------------------------------------------------------------

    def export_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Export trace events and call graph to JSON.

        If *path* is given the JSON is also written to that file.

        Returns the JSON string.
        """
        data = {
            "trace": json.loads(self._recorder.to_json(indent=None)),
            "call_graph": self._call_graph.to_dict(),
            "metrics": self._metrics.to_dict(),
            "filter_config": self._filter.to_dict(),
        }
        blob = json.dumps(data, indent=indent, default=str)
        if path is not None:
            Path(path).write_text(blob, encoding="utf-8")
        return blob

    def export_events(self) -> List[Dict[str, Any]]:
        """Return all events as a list of dictionaries."""
        return self._recorder.to_dicts()

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary of the tracing session."""
        return {
            "active": self._active,
            "total_events": len(self._recorder),
            "dropped_events": self._recorder.dropped_count,
            "call_graph_nodes": self._call_graph.n_nodes,
            "call_graph_edges": self._call_graph.n_edges,
            "wall_time_s": round(self._metrics.wall_time_s, 4),
            "mean_overhead_ns": round(self._metrics.mean_overhead_ns, 1),
        }

    def dag_nodes(self) -> List[DAGNode]:
        """Convert captured CALL events into preliminary :class:`DAGNode`
        instances via :class:`NodeFactory`.

        Only events whose function name maps to a known :class:`OpType`
        are converted.
        """
        op_lookup: Dict[str, OpType] = {op.value: op for op in OpType}
        nodes: List[DAGNode] = []
        for event in self._recorder.calls_only():
            op_type = op_lookup.get(event.function_name)
            if op_type is None:
                continue
            loc = SourceLocation(
                file_path=event.filename,
                line_number=event.lineno,
                function_name=event.function_name,
                class_name=event.class_name,
            )
            node = NodeFactory.create(op_type=op_type, source_location=loc)
            nodes.append(node)
        return nodes

    def reset(self) -> None:
        """Clear all recorded data and reset metrics."""
        self._recorder.clear()
        self._call_graph.reset()
        self._metrics = PerformanceMetrics()


# ===================================================================
#  Convenience context-manager function
# ===================================================================

@contextmanager
def trace_pipeline(
    filter_config: Optional[FilterConfig] = None,
    buffer_limit: int = _DEFAULT_BUFFER_LIMIT,
    record_locals: bool = True,
) -> Generator[PipelineTracer, None, None]:
    """Context manager that traces a pipeline execution block.

    Example::

        with trace_pipeline() as tracer:
            pipeline.fit(X_train, y_train)
        print(tracer.summary())
    """
    tracer = PipelineTracer(
        filter_config=filter_config,
        buffer_limit=buffer_limit,
        record_locals=record_locals,
    )
    tracer.start()
    try:
        yield tracer
    finally:
        tracer.stop()
