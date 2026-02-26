"""
Invariance testing module for Causal-Shielded Adaptive Trading.

Provides sequential, anytime-valid invariance tests for causal graph edges
across market regimes using e-value methodology.
"""

from .e_values import (
    EValueConstructor,
    ProductEValue,
    MixtureEValue,
    WealthProcess,
    GROWMartingale,
    ConfidenceSequence,
)
from .scit import (
    SCITAlgorithm,
    EdgeClassification,
    EBHProcedure,
)
from .anytime_inference import (
    AnytimeInference,
    MixtureMartingale,
    SubGaussianEProcess,
    SubExponentialEProcess,
    SequentialTest,
)
from .power_analysis import (
    PowerAnalyzer,
    HSICEffectSize,
    PowerCurve,
    SampleSizeCalculator,
)

__all__ = [
    "EValueConstructor",
    "ProductEValue",
    "MixtureEValue",
    "WealthProcess",
    "GROWMartingale",
    "ConfidenceSequence",
    "SCITAlgorithm",
    "EdgeClassification",
    "EBHProcedure",
    "AnytimeInference",
    "MixtureMartingale",
    "SubGaussianEProcess",
    "SubExponentialEProcess",
    "SequentialTest",
    "PowerAnalyzer",
    "HSICEffectSize",
    "PowerCurve",
    "SampleSizeCalculator",
]
