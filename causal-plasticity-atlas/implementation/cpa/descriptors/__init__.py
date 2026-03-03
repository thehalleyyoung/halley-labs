"""
Causal-Plasticity Atlas — Plasticity Descriptor Module.

Implements Algorithm 2 (ALG2): Unified 4D Plasticity Descriptor Computation.
Provides structural, parametric, emergence, and context-sensitivity descriptors
for characterizing mechanism plasticity across heterogeneous causal contexts.

Main classes:
    PlasticityComputer      — Single-variable 4D descriptor computation.
    BatchPlasticityComputer  — Batch computation across all variables.
    StabilitySelector        — Subsample-based structural stability.
    ParametricBootstrap      — Parametric bootstrap CIs.
    PermutationCalibrator    — Null-distribution threshold calibration.
    PlasticityClassifier     — Threshold-based mechanism classification.
    ClassificationValidator  — Cross-validation of boundaries.
    ClassificationReport     — Human-readable reports.
"""

from __future__ import annotations

from cpa.descriptors.plasticity import (
    BatchPlasticityComputer,
    PlasticityComputer,
    PlasticityDescriptor,
    PlasticityConfig,
)
from cpa.descriptors.confidence import (
    ParametricBootstrap,
    PermutationCalibrator,
    StabilitySelector,
)
from cpa.descriptors.classification import (
    ClassificationReport,
    ClassificationValidator,
    PlasticityClassifier,
)

__all__ = [
    "PlasticityComputer",
    "BatchPlasticityComputer",
    "PlasticityDescriptor",
    "PlasticityConfig",
    "StabilitySelector",
    "ParametricBootstrap",
    "PermutationCalibrator",
    "PlasticityClassifier",
    "ClassificationValidator",
    "ClassificationReport",
]
