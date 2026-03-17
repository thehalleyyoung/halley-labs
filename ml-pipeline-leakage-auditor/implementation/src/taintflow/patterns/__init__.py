"""
taintflow.patterns – Library of known ML pipeline leakage patterns.

This package defines detectors for common categories of data leakage in
machine-learning pipelines:

* **Temporal leakage** – future data used to train on past observations.
* **Target leakage** – target variable information flowing into features.
* **Feature leakage** – feature selection / importance on unsplit data.
* **Preprocessing leakage** – scaling, imputation, encoding before split.

Each module exposes a top-level detector class together with specialised
sub-detectors and a ``*Pattern`` dataclass describing matched patterns.
"""

from __future__ import annotations

from taintflow.patterns.temporal import (
    LookAheadDetector,
    RollingWindowAuditor,
    SeasonalLeakageDetector,
    TemporalLeakageDetector,
    TemporalLeakagePattern,
    TemporalSplitAnalyzer,
    TimeSeriesValidator,
)
from taintflow.patterns.target_leakage import (
    DirectTargetDetector,
    ProxyFeatureDetector,
    TargetDerivedFeatureDetector,
    TargetEncodingAuditor,
    TargetLeakageDetector,
    TargetLeakagePattern,
    TargetLeakageSeverityEstimator,
)
from taintflow.patterns.feature_leakage import (
    CorrelationFilterAuditor,
    DimensionalityReductionAuditor,
    FeatureImportanceLeakDetector,
    FeatureLeakageDetector,
    FeatureLeakagePattern,
    SelectionBeforeSplitDetector,
)
from taintflow.patterns.preprocessing import (
    EncodingLeakageDetector,
    ImputationLeakageDetector,
    LeakagePatternLibrary,
    NormalizationLeakageDetector,
    OutlierLeakageDetector,
    PatternMatcher,
    PreprocessingLeakageDetector,
    PreprocessingLeakagePattern,
    ScalingLeakageDetector,
)

__all__: list[str] = [
    # temporal
    "TemporalLeakageDetector",
    "TemporalLeakagePattern",
    "TimeSeriesValidator",
    "LookAheadDetector",
    "RollingWindowAuditor",
    "SeasonalLeakageDetector",
    "TemporalSplitAnalyzer",
    # target
    "TargetLeakageDetector",
    "TargetLeakagePattern",
    "DirectTargetDetector",
    "TargetEncodingAuditor",
    "ProxyFeatureDetector",
    "TargetDerivedFeatureDetector",
    "TargetLeakageSeverityEstimator",
    # feature
    "FeatureLeakageDetector",
    "FeatureLeakagePattern",
    "SelectionBeforeSplitDetector",
    "CorrelationFilterAuditor",
    "FeatureImportanceLeakDetector",
    "DimensionalityReductionAuditor",
    # preprocessing
    "PreprocessingLeakageDetector",
    "PreprocessingLeakagePattern",
    "ScalingLeakageDetector",
    "ImputationLeakageDetector",
    "EncodingLeakageDetector",
    "OutlierLeakageDetector",
    "NormalizationLeakageDetector",
    "PatternMatcher",
    "LeakagePatternLibrary",
]
