"""
usability_oracle.evaluation — Evaluation framework for the usability oracle.

Provides ordinal validation, regression-detection metrics, ablation studies,
baseline comparators, and evaluation reporting.
"""

from __future__ import annotations

from usability_oracle.evaluation.ordinal import OrdinalResult, OrdinalValidator
from usability_oracle.evaluation.regression import RegressionDetectionMetrics
from usability_oracle.evaluation.ablation import AblationResult, AblationStudy
from usability_oracle.evaluation.baselines import BaselineComparator
from usability_oracle.evaluation.reporting import EvaluationReporter

__all__ = [
    "OrdinalResult",
    "OrdinalValidator",
    "RegressionDetectionMetrics",
    "AblationResult",
    "AblationStudy",
    "BaselineComparator",
    "EvaluationReporter",
]
