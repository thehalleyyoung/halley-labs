"""Analysis tools for CausalQD archive inspection and inference.

Provides sensitivity analysis, causal inference from archive diversity,
algorithm comparison, and diagnostic tools.
"""

from causal_qd.analysis.sensitivity import (
    DataSensitivityAnalyzer,
    ScoreSensitivityAnalyzer,
    ParameterSensitivityAnalyzer,
)
from causal_qd.analysis.causal_inference import (
    ArchiveCausalInference,
    InterventionEstimator,
    CausalQueryEngine,
)
from causal_qd.analysis.comparison import (
    AlgorithmComparator,
    BenchmarkSuite,
)
from causal_qd.analysis.convergence_analysis import (
    ConvergenceAnalyzer,
    ConvergenceSnapshot,
)
from causal_qd.analysis.supermartingale import SupermartingaleTracker
from causal_qd.analysis.ergodicity import ErgodicityChecker
from causal_qd.analysis.diagnostics import (
    ArchiveDiagnostics,
    OperatorDiagnostics,
    ScoreDiagnostics,
)

__all__ = [
    "DataSensitivityAnalyzer",
    "ScoreSensitivityAnalyzer",
    "ParameterSensitivityAnalyzer",
    "ArchiveCausalInference",
    "InterventionEstimator",
    "CausalQueryEngine",
    "AlgorithmComparator",
    "BenchmarkSuite",
    "ConvergenceAnalyzer",
    "ConvergenceSnapshot",
    "SupermartingaleTracker",
    "ErgodicityChecker",
    "ArchiveDiagnostics",
    "OperatorDiagnostics",
    "ScoreDiagnostics",
]
