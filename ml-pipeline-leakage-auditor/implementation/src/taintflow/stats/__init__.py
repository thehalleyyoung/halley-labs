"""
TaintFlow statistical testing utilities.

Provides independence tests, distribution utilities, and hypothesis testing
for auditing information leakage in ML pipelines.  All implementations use
only the Python standard library (no NumPy / SciPy).
"""

from __future__ import annotations

from taintflow.stats.independence import (
    ChiSquaredTest,
    ConditionalIndependenceTest,
    DistanceCorrelationTest,
    HilbertSchmidtIndependenceTest,
    IndependenceTestResult,
    IndependenceTestSuite,
    MultipleTestingCorrection,
    MutualInformationTest,
)
from taintflow.stats.distribution import (
    ContinuousDistributionApprox,
    DiscreteDistribution,
    EmpiricalDistribution,
    HistogramEstimator,
    JSDivergence,
    KLDivergence,
    MomentEstimator,
    QuantileEstimator,
    TwoSampleTest,
)
from taintflow.stats.hypothesis import (
    ConfidenceInterval,
    EffectSize,
    HypothesisTest,
    LeakageSignificanceTest,
    MetaAnalysis,
    OneSampleTest,
    PairedTest,
    PowerAnalysis,
    SequentialTest,
    TestResult,
    TwoSampleHypothesisTest,
)

__all__: list[str] = [
    # independence
    "ChiSquaredTest",
    "ConditionalIndependenceTest",
    "DistanceCorrelationTest",
    "HilbertSchmidtIndependenceTest",
    "IndependenceTestResult",
    "IndependenceTestSuite",
    "MultipleTestingCorrection",
    "MutualInformationTest",
    # distribution
    "ContinuousDistributionApprox",
    "DiscreteDistribution",
    "EmpiricalDistribution",
    "HistogramEstimator",
    "JSDivergence",
    "KLDivergence",
    "MomentEstimator",
    "QuantileEstimator",
    "TwoSampleTest",
    # hypothesis
    "ConfidenceInterval",
    "EffectSize",
    "HypothesisTest",
    "LeakageSignificanceTest",
    "MetaAnalysis",
    "OneSampleTest",
    "PairedTest",
    "PowerAnalysis",
    "SequentialTest",
    "TestResult",
    "TwoSampleHypothesisTest",
]
