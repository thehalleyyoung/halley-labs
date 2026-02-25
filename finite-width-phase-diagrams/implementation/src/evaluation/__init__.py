"""Evaluation harness for finite-width phase diagram predictions.

Provides ground-truth training, evaluation metrics, ablation studies,
and retrodiction validation against known theoretical results.
"""

from .ground_truth import (
    TrainingConfig,
    TrainingMeasurement,
    TrainingRun,
    GroundTruthResult,
    GroundTruthHarness,
)
from .metrics import (
    BoundaryMetrics,
    CalibrationMetrics,
    MetricsComputer,
    MetricsResult,
    RegimeMetrics,
)
from .ablation import (
    AblationComparison,
    AblationConfig,
    AblationResult,
    AblationRunner,
)
from .retrodiction import (
    KnownResult,
    RetrodictionResult,
    RetrodictionValidator,
)
