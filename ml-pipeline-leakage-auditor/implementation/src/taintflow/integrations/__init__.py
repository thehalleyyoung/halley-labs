"""
taintflow.integrations – Scikit-learn and pandas integration for leakage auditing.

This package provides wrappers and interceptors that instrument third-party
ML libraries so that every operation is tracked in the Pipeline Information
DAG (PI-DAG).  The three sub-modules cover:

* **sklearn_interceptor**: Audited wrappers for scikit-learn estimators and
  pipelines.  Drop-in replacements that record shapes, provenance, and
  leakage-relevant metadata.
* **pandas_interceptor**: Audited wrappers for pandas DataFrames and Series
  with column-level provenance tracking through every operation.
* **pipeline_wrapper**: High-level utilities for wrapping entire ML scripts
  or notebooks and collecting audit results into a single session.
"""

from __future__ import annotations

from taintflow.integrations.sklearn_interceptor import (
    AuditedEstimator,
    AuditedMinMaxScaler,
    AuditedOneHotEncoder,
    AuditedOrdinalEncoder,
    AuditedPCA,
    AuditedPipeline,
    AuditedRobustScaler,
    AuditedSelectKBest,
    AuditedSimpleImputer,
    AuditedKNNImputer,
    AuditedStandardScaler,
    AuditedTargetEncoder,
    AuditedTransformerMixin,
    AuditedTruncatedSVD,
    AuditedVarianceThreshold,
    EstimatorClassifier,
    PipelineAuditor,
    SklearnVersionDetector,
)
from taintflow.integrations.pandas_interceptor import (
    AuditedDataFrame,
    AuditedSeries,
    ColumnLineage,
    DataFrameAuditor,
    OperationLog,
    ProvenanceAnnotation,
)
from taintflow.integrations.pipeline_wrapper import (
    AuditSession,
    CleanupManager,
    DAGBuilder,
    InstrumentationConfig,
    NotebookExecutor,
    PipelineValidator,
    PipelineWrapper,
    ScriptExecutor,
)
from taintflow.integrations.torch_interceptor import (
    AuditedDataLoader,
    AuditedDataset,
    DatasetAuditLog,
    TorchLeakageFinding,
    TorchProvenanceRecord,
    detect_normalization_leakage,
    detect_shared_dataset_leakage,
)
from taintflow.integrations.formats import (
    FormatDetector,
    IngestionRecord,
    get_partition,
    get_provenance,
    load_csv,
    load_parquet,
    save_parquet,
)

__all__: list[str] = [
    # sklearn
    "AuditedPipeline",
    "AuditedEstimator",
    "AuditedTransformerMixin",
    "PipelineAuditor",
    "AuditedStandardScaler",
    "AuditedMinMaxScaler",
    "AuditedRobustScaler",
    "AuditedPCA",
    "AuditedTruncatedSVD",
    "AuditedSimpleImputer",
    "AuditedKNNImputer",
    "AuditedOneHotEncoder",
    "AuditedOrdinalEncoder",
    "AuditedTargetEncoder",
    "AuditedSelectKBest",
    "AuditedVarianceThreshold",
    "EstimatorClassifier",
    "SklearnVersionDetector",
    # pandas
    "AuditedDataFrame",
    "AuditedSeries",
    "DataFrameAuditor",
    "ProvenanceAnnotation",
    "OperationLog",
    "ColumnLineage",
    # pipeline wrapper
    "PipelineWrapper",
    "ScriptExecutor",
    "NotebookExecutor",
    "DAGBuilder",
    "PipelineValidator",
    "AuditSession",
    "InstrumentationConfig",
    "CleanupManager",
    # PyTorch
    "AuditedDataset",
    "AuditedDataLoader",
    "DatasetAuditLog",
    "TorchProvenanceRecord",
    "TorchLeakageFinding",
    "detect_normalization_leakage",
    "detect_shared_dataset_leakage",
    # File formats
    "FormatDetector",
    "IngestionRecord",
    "load_csv",
    "load_parquet",
    "save_parquet",
    "get_provenance",
    "get_partition",
]
