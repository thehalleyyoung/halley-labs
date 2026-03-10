"""
usability_oracle.resources — Wickens' multiple-resource theory modelling.

Implements the four-dimensional cognitive resource model (processing stage,
modality, visual channel, processing code) for computing interference
between concurrent UI operations.

::

    from usability_oracle.resources import Resource, DemandVector, InterferenceMatrix
"""

from __future__ import annotations

from usability_oracle.resources.types import (
    DemandVector,
    InterferenceMatrix,
    PerceptualModality,
    ProcessingCode,
    ProcessingStage,
    Resource,
    ResourceAllocation,
    ResourceConflict,
    ResourceDemand,
    VisualChannel,
)

from usability_oracle.resources.protocols import (
    DemandEstimator,
    InterferenceComputer,
    ResourceAllocator,
)

from usability_oracle.resources.wickens_model import WickensModel
from usability_oracle.resources.conflict_matrix import ResourceConflictMatrix
from usability_oracle.resources.allocator import (
    ResourceAllocator as ResourceAllocatorImpl,
)
from usability_oracle.resources.demand_estimator import (
    DemandEstimator as DemandEstimatorImpl,
)

__all__ = [
    # types
    "DemandVector",
    "InterferenceMatrix",
    "PerceptualModality",
    "ProcessingCode",
    "ProcessingStage",
    "Resource",
    "ResourceAllocation",
    "ResourceConflict",
    "ResourceDemand",
    "VisualChannel",
    # protocols
    "DemandEstimator",
    "InterferenceComputer",
    "ResourceAllocator",
    # implementations
    "WickensModel",
    "ResourceConflictMatrix",
    "ResourceAllocatorImpl",
    "DemandEstimatorImpl",
]
