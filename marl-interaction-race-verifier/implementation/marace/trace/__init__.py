"""MARACE trace module — HB-stamped execution traces for multi-agent systems.

Public API
----------
Event types (``events``)::

    EventType, Event, ActionEvent, ObservationEvent,
    CommunicationEvent, EnvironmentEvent, SyncEvent

Vector-clock helpers::

    VectorClock, vc_zero, vc_increment, vc_merge,
    vc_leq, vc_strictly_less, vc_concurrent

Trace containers (``trace``)::

    ExecutionTrace, MultiAgentTrace, TraceSegment,
    TraceStatistics, TraceValidator, TraceValidationResult

Serialization (``serialization``)::

    TraceSerialization, StreamingTraceWriter, StreamingTraceReader,
    traces_equivalent, trace_diff_summary

Replay (``replay``)::

    TraceReplayer, SchedulePermutation, ReplayValidator, ReplayResult

Construction (``construction``)::

    TraceConstructor, AsyncTraceBuilder, TraceRecorder, TraceMerger,
    infer_causal_chains, compute_influence_matrix
"""

# -- events -----------------------------------------------------------------
from .events import (
    EventType,
    Event,
    ActionEvent,
    ObservationEvent,
    CommunicationEvent,
    EnvironmentEvent,
    SyncEvent,
    VectorClock,
    vc_zero,
    vc_increment,
    vc_merge,
    vc_leq,
    vc_strictly_less,
    vc_concurrent,
    event_from_dict,
    make_action_event,
    make_observation_event,
    make_communication_event,
    make_environment_event,
    make_sync_event,
    validate_event,
    EventValidationError,
)

# -- trace ------------------------------------------------------------------
from .trace import (
    ExecutionTrace,
    MultiAgentTrace,
    TraceSegment,
    TraceStatistics,
    TraceValidator,
    TraceValidationResult,
)

# -- serialization ----------------------------------------------------------
from .serialization import (
    TraceSerialization,
    StreamingTraceWriter,
    StreamingTraceReader,
    traces_equivalent,
    trace_diff_summary,
)

# -- replay -----------------------------------------------------------------
from .replay import (
    TraceReplayer,
    SchedulePermutation,
    ReplayValidator,
    ReplayResult,
    ReplayStepResult,
    AgentState,
)

# -- construction -----------------------------------------------------------
from .construction import (
    TraceConstructor,
    AsyncTraceBuilder,
    TraceRecorder,
    TraceMerger,
    infer_causal_chains,
    compute_influence_matrix,
)

__all__ = [
    # events
    "EventType",
    "Event",
    "ActionEvent",
    "ObservationEvent",
    "CommunicationEvent",
    "EnvironmentEvent",
    "SyncEvent",
    "VectorClock",
    "vc_zero",
    "vc_increment",
    "vc_merge",
    "vc_leq",
    "vc_strictly_less",
    "vc_concurrent",
    "event_from_dict",
    "make_action_event",
    "make_observation_event",
    "make_communication_event",
    "make_environment_event",
    "make_sync_event",
    "validate_event",
    "EventValidationError",
    # trace
    "ExecutionTrace",
    "MultiAgentTrace",
    "TraceSegment",
    "TraceStatistics",
    "TraceValidator",
    "TraceValidationResult",
    # serialization
    "TraceSerialization",
    "StreamingTraceWriter",
    "StreamingTraceReader",
    "traces_equivalent",
    "trace_diff_summary",
    # replay
    "TraceReplayer",
    "SchedulePermutation",
    "ReplayValidator",
    "ReplayResult",
    "ReplayStepResult",
    "AgentState",
    # construction
    "TraceConstructor",
    "AsyncTraceBuilder",
    "TraceRecorder",
    "TraceMerger",
    "infer_causal_chains",
    "compute_influence_matrix",
]
