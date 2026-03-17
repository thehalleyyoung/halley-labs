"""
TaintFlow – ML Pipeline Leakage Auditor.

Detect train/test information leakage in machine-learning pipelines via
abstract interpretation over a partition-taint lattice.  TaintFlow traces
every DataFrame column and transformation through your pipeline graph and
computes an upper bound (in bits) on how much test-set information could
have leaked into the training path.

Public API
----------
The main entry points are:

* :class:`TaintFlowConfig`       – configuration for an audit run.
* :class:`PartitionTaintLattice` – the abstract domain.
* :class:`TaintElement`          – an element in the lattice.
* :class:`ColumnTaintMap`        – per-column taint state.
* :class:`DataFrameAbstractState`– full abstract state for a DataFrame.
* :func:`audit_pipeline`         – convenience function (to be wired up).
"""

from __future__ import annotations

__version__: str = "0.1.0"
__author__: str = "TaintFlow Authors"
__license__: str = "MIT"

# Core types ------------------------------------------------------------------
from taintflow.core.types import (
    AnalysisConfig,
    AnalysisPhase,
    ChannelParams,
    ColumnSchema,
    EdgeKind,
    FeatureLeakage,
    LeakageReport,
    NodeKind,
    OpType,
    Origin,
    PipelineMetadata,
    ProvenanceInfo,
    Severity,
    ShapeMetadata,
    StageLeakage,
    TaintLabel,
)

# Lattice constructs ----------------------------------------------------------
from taintflow.core.lattice import (
    ColumnTaintMap,
    DataFrameAbstractState,
    NarrowOperator,
    PartitionTaintLattice,
    TaintElement,
    WidenOperator,
)

# Configuration ---------------------------------------------------------------
from taintflow.core.config import (
    SeverityThresholds,
    TaintFlowConfig,
)

# Errors ----------------------------------------------------------------------
from taintflow.core.errors import (
    TaintFlowError,
)

# Logging utilities -----------------------------------------------------------
from taintflow.core.logging_utils import (
    TaintFlowLogger,
)

__all__: list[str] = [
    "__version__",
    "Origin", "Severity", "OpType", "AnalysisPhase", "NodeKind", "EdgeKind",
    "ColumnSchema", "ShapeMetadata", "ProvenanceInfo", "TaintLabel",
    "FeatureLeakage", "StageLeakage", "LeakageReport",
    "PipelineMetadata", "AnalysisConfig", "ChannelParams",
    "TaintElement", "PartitionTaintLattice", "ColumnTaintMap",
    "DataFrameAbstractState", "WidenOperator", "NarrowOperator",
    "TaintFlowConfig", "SeverityThresholds",
    "TaintFlowError",
    "TaintFlowLogger",
]
