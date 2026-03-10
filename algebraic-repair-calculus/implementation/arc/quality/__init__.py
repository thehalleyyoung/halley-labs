"""
``arc.quality`` — Quality monitoring for the Algebraic Repair Calculus.

Provides:

* :class:`QualityMonitor` – batch quality checks (nulls, ranges, uniqueness, …).
* :class:`DistributionAnalyzer` – statistical shift detection (KS, PSI, χ², JSD).
* :class:`ConstraintEngine` – constraint definition, evaluation, and inference.
* :class:`DataProfiler` – comprehensive data profiling.
"""

from arc.quality.monitor import QualityMonitor
from arc.quality.distribution import DistributionAnalyzer
from arc.quality.constraints import ConstraintEngine
from arc.quality.profiler import DataProfiler
from arc.quality.drift import (
    Alert,
    AlertAction,
    AlertConfig,
    ColumnDrift,
    ColumnProfile as DriftColumnProfile,
    DriftDetector,
    DriftResult,
    DriftSeverity,
    DriftTimeSeries,
    DriftType,
    SchemaDrift,
    detect_drift,
    detect_schema_drift,
    is_data_stale,
)

__all__ = [
    "QualityMonitor",
    "DistributionAnalyzer",
    "ConstraintEngine",
    "DataProfiler",
    # Drift
    "Alert",
    "AlertAction",
    "AlertConfig",
    "ColumnDrift",
    "DriftColumnProfile",
    "DriftDetector",
    "DriftResult",
    "DriftSeverity",
    "DriftTimeSeries",
    "DriftType",
    "SchemaDrift",
    "detect_drift",
    "detect_schema_drift",
    "is_data_stale",
]
