"""
taintflow.core – foundational modules for the TaintFlow analysis engine.

This package exposes the abstract domain (lattice), the core type system,
configuration handling, structured logging, and the exception hierarchy.
"""

from __future__ import annotations

from taintflow.core.types import (
    AnalysisConfig, AnalysisPhase, ChannelParams, ColumnSchema, EdgeKind,
    FeatureLeakage, LeakageReport, NodeKind, OpType, Origin,
    PipelineMetadata, ProvenanceInfo, Severity, ShapeMetadata,
    StageLeakage, TaintLabel,
)
from taintflow.core.lattice import (
    ColumnTaintMap, DataFrameAbstractState, NarrowOperator,
    PartitionTaintLattice, TaintElement, WidenOperator,
)
from taintflow.core.config import SeverityThresholds, TaintFlowConfig
from taintflow.core.errors import (
    AnalysisError, AttributionError, CapacityComputationError, ConfigError,
    CycleDetectedError, DAGConstructionError, FixpointDivergenceError,
    FlowComputationError, InstrumentationError, MissingNodeError,
    MonkeyPatchError, NumericalInstabilityError, ReportError,
    SerializationError, TaintFlowError, TraceError, TransferFunctionError,
    ValidationError,
)
from taintflow.core.logging_utils import LogEvent, TaintFlowLogger, log_performance, track_memory

__all__: list[str] = [
    "Origin", "Severity", "OpType", "AnalysisPhase", "NodeKind", "EdgeKind",
    "ColumnSchema", "ShapeMetadata", "ProvenanceInfo", "TaintLabel",
    "FeatureLeakage", "StageLeakage", "LeakageReport",
    "PipelineMetadata", "AnalysisConfig", "ChannelParams",
    "TaintElement", "PartitionTaintLattice", "ColumnTaintMap",
    "DataFrameAbstractState", "WidenOperator", "NarrowOperator",
    "TaintFlowConfig", "SeverityThresholds",
    "TaintFlowError", "ConfigError", "ValidationError",
    "InstrumentationError", "TraceError", "MonkeyPatchError",
    "DAGConstructionError", "CycleDetectedError", "MissingNodeError",
    "CapacityComputationError", "NumericalInstabilityError",
    "AnalysisError", "FixpointDivergenceError", "TransferFunctionError",
    "AttributionError", "FlowComputationError",
    "ReportError", "SerializationError",
    "TaintFlowLogger", "LogEvent", "log_performance", "track_memory",
]
