"""
Data Analysis module for neural network phase diagram analysis.

Implements dataset-dependent kernel analysis, random feature approximations,
and task complexity measurements for understanding how data properties
interact with network architecture in determining phase behavior.
"""

from .dataset_kernel import (
    DataDependentNTK,
    KernelTargetAlignment,
    GeneralizationBound,
    EffectiveDimension,
    SpectralBiasAnalyzer,
)
from .feature_maps import (
    RandomFeatureApproximation,
    FeatureDimensionEstimator,
    FeatureQualityMetric,
    FeatureAlignmentAnalyzer,
    RandomFeatureRegression,
)
from .task_complexity import (
    TargetSmoothnessEstimator,
    RKHSNormComputer,
    CurriculumLearningAnalyzer,
    TaskArchitectureCompatibility,
)

__all__ = [
    "DataDependentNTK",
    "KernelTargetAlignment",
    "GeneralizationBound",
    "EffectiveDimension",
    "SpectralBiasAnalyzer",
    "RandomFeatureApproximation",
    "FeatureDimensionEstimator",
    "FeatureQualityMetric",
    "FeatureAlignmentAnalyzer",
    "RandomFeatureRegression",
    "TargetSmoothnessEstimator",
    "RKHSNormComputer",
    "CurriculumLearningAnalyzer",
    "TaskArchitectureCompatibility",
]
