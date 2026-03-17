"""
taintflow.instrument -- Instrumentation layer for ML pipeline leakage auditing.

This package intercepts pandas, scikit-learn, and numpy operations at runtime
to build a provenance-aware execution trace.  The trace is later consumed by
:mod:`taintflow.dag.builder` to construct the Pipeline Information DAG (PI-DAG).

Submodules
----------
context
    Thread-local state management (call stacks, column tracking, provenance).
tracer
    ``sys.settrace``-based execution tracer for capturing function calls.
pandas_hooks
    Per-operation hooks for every instrumented pandas method.
provenance
    Row-level provenance tracking using compressed bitmaps.

Quick Start
-----------
>>> from taintflow.instrument import InstrumentationContext, PipelineTracer
>>> ctx = InstrumentationContext()
>>> tracer = PipelineTracer()
>>> # Execute pipeline under instrumentation
>>> events = tracer.get_events()
"""

from __future__ import annotations

from taintflow.instrument.context import (
    InstrumentationContext,
    ColumnTracker,
    ProvenanceStore,
    DataFrameRegistry,
)
from taintflow.instrument.tracer import (
    PipelineTracer,
    FilterConfig,
    TraceEvent,
    TraceRecorder,
)
from taintflow.instrument.pandas_hooks import (
    PandasInterceptor as PandasHooks,
)
from taintflow.instrument.provenance import (
    ProvenanceTracker,
)

__all__ = [
    # context
    "InstrumentationContext",
    "ColumnTracker",
    "ProvenanceStore",
    "DataFrameRegistry",
    # tracer
    "PipelineTracer",
    "FilterConfig",
    "TraceEvent",
    "TraceRecorder",
    # pandas_hooks
    "PandasHooks",
    # provenance
    "ProvenanceTracker",
]
