"""
usability_oracle.analysis — Advanced analysis capabilities.

Provides sensitivity analysis, convergence diagnostics, numerical stability
analysis, computational complexity estimation, information-theoretic
analysis, and statistical testing frameworks.
"""

from __future__ import annotations

from usability_oracle.analysis.sensitivity import (
    SensitivityAnalyzer,
    SobolIndices,
    MorrisResult,
)
from usability_oracle.analysis.convergence import (
    ConvergenceAnalyzer,
    ConvergenceResult,
)
from usability_oracle.analysis.stability import (
    StabilityAnalyzer,
    StabilityResult,
)
from usability_oracle.analysis.complexity import (
    ComplexityAnalyzer,
    ComplexityEstimate,
)
from usability_oracle.analysis.information import (
    InformationAnalyzer,
    ChannelAnalysis,
)
from usability_oracle.analysis.statistical import (
    StatisticalTester,
    TestResult,
)

__all__ = [
    "SensitivityAnalyzer",
    "SobolIndices",
    "MorrisResult",
    "ConvergenceAnalyzer",
    "ConvergenceResult",
    "StabilityAnalyzer",
    "StabilityResult",
    "ComplexityAnalyzer",
    "ComplexityEstimate",
    "InformationAnalyzer",
    "ChannelAnalysis",
    "StatisticalTester",
    "TestResult",
]
