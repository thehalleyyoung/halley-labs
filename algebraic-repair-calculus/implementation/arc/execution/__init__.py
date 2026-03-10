"""
``arc.execution`` — DuckDB-based repair plan execution engine.

Provides:

* :class:`ExecutionEngine` – execute repair plans against DuckDB.
* :class:`CheckpointManager` – atomic checkpointing and rollback.
* :class:`RepairValidator` – correctness validation.
* :class:`ExecutionScheduler` – dependency-aware scheduling.
"""

from arc.execution.engine import ExecutionEngine
from arc.execution.checkpoint import CheckpointManager
from arc.execution.validation import RepairValidator
from arc.execution.scheduler import ExecutionScheduler
from arc.execution.incremental import (
    DataDelta as IncrementalDataDelta,
    DataDeltaRow,
    DeltaType,
    IncrementalExecutor,
    IncrementalResult,
    create_data_delta,
    execute_incremental,
)
from arc.execution.materialization import (
    AccessPattern,
    MaterializationManager,
    MaterializedView,
    RefreshMode,
    RefreshResult,
    StorageEstimate,
    ViewState,
    materialize_node,
    refresh_view,
)

__all__ = [
    "ExecutionEngine",
    "CheckpointManager",
    "RepairValidator",
    "ExecutionScheduler",
    # Incremental
    "IncrementalDataDelta",
    "DataDeltaRow",
    "DeltaType",
    "IncrementalExecutor",
    "IncrementalResult",
    "create_data_delta",
    "execute_incremental",
    # Materialization
    "AccessPattern",
    "MaterializationManager",
    "MaterializedView",
    "RefreshMode",
    "RefreshResult",
    "StorageEstimate",
    "ViewState",
    "materialize_node",
    "refresh_view",
]
