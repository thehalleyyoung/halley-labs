"""CPA Benchmarks — Generators, metrics, and runners.

Provides synthetic data generators, evaluation metrics, and benchmark
execution infrastructure for validating the Causal-Plasticity Atlas.

Modules
-------
generators
    Synthetic benchmark generators (FSVP, CSVM, TPS).
metrics
    Evaluation metrics for classification, tipping points, certificates.
runners
    Benchmark execution and result aggregation.
"""

from benchmarks.generators import (
    FSVPGenerator,
    CSVMGenerator,
    TPSGenerator,
    SemiSyntheticGenerator,
)
from benchmarks.metrics import (
    ClassificationMetrics,
    TippingPointMetrics,
    CertificateMetrics,
    ArchiveMetrics,
)
from benchmarks.runners import (
    BenchmarkRunner,
    ResultAggregator,
)

__all__ = [
    "FSVPGenerator",
    "CSVMGenerator",
    "TPSGenerator",
    "SemiSyntheticGenerator",
    "ClassificationMetrics",
    "TippingPointMetrics",
    "CertificateMetrics",
    "ArchiveMetrics",
    "BenchmarkRunner",
    "ResultAggregator",
]
