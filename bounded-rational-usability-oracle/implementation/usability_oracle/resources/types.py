"""
usability_oracle.resources.types — Data types for Wickens' multiple-resource theory.

Provides immutable value types for modelling the four-dimensional resource
space from Wickens' multiple resource theory (MRT, 2002, 2008):

    1. **Processing stages**: perception vs. cognition vs. response
    2. **Perceptual modalities**: visual vs. auditory
    3. **Visual channels**: focal vs. ambient
    4. **Processing codes**: spatial vs. verbal

Tasks that share the same resource dimension(s) produce *structural
interference*; tasks on orthogonal dimensions can proceed in parallel
with minimal interference.

Reference: Wickens, C. D. (2002). Multiple resources and performance
prediction. *Theoretical Issues in Ergonomics Science*, 3(2), 159–177.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class ProcessingStage(Enum):
    """Wickens MRT processing stage dimension."""

    PERCEPTION = "perception"
    """Early perceptual encoding."""

    COGNITION = "cognition"
    """Central cognitive processing (working memory, decision)."""

    RESPONSE = "response"
    """Motor response selection and execution."""


@unique
class PerceptualModality(Enum):
    """Wickens MRT perceptual modality dimension."""

    VISUAL = "visual"
    AUDITORY = "auditory"


@unique
class VisualChannel(Enum):
    """Wickens MRT visual processing channel."""

    FOCAL = "focal"
    """Foveal / high-acuity processing."""

    AMBIENT = "ambient"
    """Peripheral / spatial-orientation processing."""


@unique
class ProcessingCode(Enum):
    """Wickens MRT processing code dimension."""

    SPATIAL = "spatial"
    VERBAL = "verbal"


# ═══════════════════════════════════════════════════════════════════════════
# Resource
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Resource:
    """A single cognitive resource in Wickens' four-dimensional space.

    A resource is identified by its position along each MRT dimension.
    Not all dimension combinations are meaningful; for example,
    ``visual_channel`` is only relevant when ``modality`` is ``VISUAL``.

    Attributes:
        stage: Processing stage.
        modality: Perceptual modality (``None`` for cognition/response).
        visual_channel: Visual channel (``None`` for auditory or non-
            perceptual stages).
        code: Processing code (spatial vs. verbal).
        label: Optional human-readable label for this resource.
    """

    stage: ProcessingStage
    modality: Optional[PerceptualModality] = None
    visual_channel: Optional[VisualChannel] = None
    code: ProcessingCode = ProcessingCode.SPATIAL
    label: str = ""

    @property
    def dimension_tuple(self) -> Tuple[str, Optional[str], Optional[str], str]:
        """Canonical tuple representation for hashing/comparison."""
        return (
            self.stage.value,
            self.modality.value if self.modality else None,
            self.visual_channel.value if self.visual_channel else None,
            self.code.value,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "modality": self.modality.value if self.modality else None,
            "visual_channel": (
                self.visual_channel.value if self.visual_channel else None
            ),
            "code": self.code.value,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Resource:
        mod = d.get("modality")
        vc = d.get("visual_channel")
        return cls(
            stage=ProcessingStage(d["stage"]),
            modality=PerceptualModality(mod) if mod else None,
            visual_channel=VisualChannel(vc) if vc else None,
            code=ProcessingCode(d.get("code", "spatial")),
            label=str(d.get("label", "")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ResourceDemand
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ResourceDemand:
    """Demand that a single UI operation places on a cognitive resource.

    Demand is measured on a normalised [0, 1] scale where 0 is no demand
    and 1 is full saturation of the resource.

    Attributes:
        resource: The cognitive resource being demanded.
        demand_level: Demand intensity in [0, 1].
        operation_id: Identifier of the UI operation generating this demand.
        description: Optional human-readable description.
    """

    resource: Resource
    demand_level: float
    operation_id: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.demand_level <= 1.0):
            raise ValueError(
                f"demand_level must be in [0, 1], got {self.demand_level}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource.to_dict(),
            "demand_level": self.demand_level,
            "operation_id": self.operation_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResourceDemand:
        return cls(
            resource=Resource.from_dict(d["resource"]),
            demand_level=float(d["demand_level"]),
            operation_id=str(d.get("operation_id", "")),
            description=str(d.get("description", "")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# DemandVector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class DemandVector:
    """Aggregated demand profile across all resources for an operation.

    The demand vector is the central representation for computing
    resource conflicts and interference costs.

    Attributes:
        demands: Individual resource demands.
        operation_id: Identifier of the composite operation.
        total_demand: Sum of all demand levels (may exceed 1.0 when
            multiple resources are engaged simultaneously).
    """

    demands: Tuple[ResourceDemand, ...]
    operation_id: str = ""
    total_demand: float = 0.0

    def __post_init__(self) -> None:
        if self.total_demand == 0.0 and self.demands:
            object.__setattr__(
                self, "total_demand",
                sum(d.demand_level for d in self.demands),
            )

    @property
    def num_resources(self) -> int:
        """Number of distinct resources with non-zero demand."""
        return sum(1 for d in self.demands if d.demand_level > 0.0)

    @property
    def peak_demand(self) -> float:
        """Maximum demand across all resources."""
        return max((d.demand_level for d in self.demands), default=0.0)

    def demand_for(self, stage: ProcessingStage) -> float:
        """Total demand on a specific processing stage."""
        return sum(
            d.demand_level for d in self.demands
            if d.resource.stage == stage
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "demands": [d.to_dict() for d in self.demands],
            "operation_id": self.operation_id,
            "total_demand": self.total_demand,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DemandVector:
        return cls(
            demands=tuple(ResourceDemand.from_dict(x) for x in d["demands"]),
            operation_id=str(d.get("operation_id", "")),
            total_demand=float(d.get("total_demand", 0.0)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ResourceConflict
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ResourceConflict:
    """A detected conflict between two operations on a shared resource.

    Conflicts arise when two concurrent operations demand the same
    resource, causing structural interference (increased error rate
    or completion time).

    Attributes:
        resource: The shared cognitive resource.
        operation_a: First competing operation identifier.
        operation_b: Second competing operation identifier.
        demand_a: Demand from operation A on this resource.
        demand_b: Demand from operation B on this resource.
        conflict_severity: Severity in [0, 1] — product of demands
            weighted by the Wickens conflict matrix.
        description: Human-readable explanation.
    """

    resource: Resource
    operation_a: str
    operation_b: str
    demand_a: float
    demand_b: float
    conflict_severity: float
    description: str = ""

    @property
    def combined_demand(self) -> float:
        """Sum of demands from both operations."""
        return self.demand_a + self.demand_b

    @property
    def is_overload(self) -> bool:
        """Whether the combined demand exceeds 1.0 (resource overload)."""
        return self.combined_demand > 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource.to_dict(),
            "operation_a": self.operation_a,
            "operation_b": self.operation_b,
            "demand_a": self.demand_a,
            "demand_b": self.demand_b,
            "conflict_severity": self.conflict_severity,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResourceConflict:
        return cls(
            resource=Resource.from_dict(d["resource"]),
            operation_a=str(d["operation_a"]),
            operation_b=str(d["operation_b"]),
            demand_a=float(d["demand_a"]),
            demand_b=float(d["demand_b"]),
            conflict_severity=float(d["conflict_severity"]),
            description=str(d.get("description", "")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# InterferenceMatrix
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class InterferenceMatrix:
    """Pairwise interference costs between concurrent operations.

    The matrix M[i, j] gives the interference cost when operation i
    and operation j are performed concurrently.  Diagonal entries are
    zero (no self-interference).

    Attributes:
        operation_ids: Ordered list of operation identifiers (axis labels).
        matrix: Flattened upper-triangular interference values stored
            row-major.  Length = n*(n−1)/2 for n operations.
        conflicts: Detailed conflict records for pairs with non-zero
            interference.
    """

    operation_ids: Tuple[str, ...]
    matrix: Tuple[float, ...]
    conflicts: Tuple[ResourceConflict, ...]

    @property
    def num_operations(self) -> int:
        """Number of operations in the matrix."""
        return len(self.operation_ids)

    def get_interference(self, op_a: str, op_b: str) -> float:
        """Look up the interference between two operations.

        Parameters:
            op_a: First operation identifier.
            op_b: Second operation identifier.

        Returns:
            Interference cost (0.0 if no conflict).
        """
        if op_a == op_b:
            return 0.0
        ids = list(self.operation_ids)
        try:
            i, j = sorted([ids.index(op_a), ids.index(op_b)])
        except ValueError:
            return 0.0
        n = len(ids)
        idx = i * n - i * (i + 1) // 2 + (j - i - 1)
        return self.matrix[idx] if 0 <= idx < len(self.matrix) else 0.0

    @property
    def max_interference(self) -> float:
        """Maximum pairwise interference in the matrix."""
        return max(self.matrix, default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_ids": list(self.operation_ids),
            "matrix": list(self.matrix),
            "conflicts": [c.to_dict() for c in self.conflicts],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> InterferenceMatrix:
        return cls(
            operation_ids=tuple(d["operation_ids"]),
            matrix=tuple(d["matrix"]),
            conflicts=tuple(
                ResourceConflict.from_dict(c) for c in d["conflicts"]
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ResourceAllocation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    """Optimal allocation of cognitive resources to concurrent operations.

    The allocation specifies how much of each resource to devote to each
    operation to minimise total expected cost (or maximise throughput).

    Attributes:
        allocations: Mapping  operation_id → {resource_label → fraction}.
            Each fraction is in [0, 1] and fractions for a given resource
            must sum to ≤ 1.0 across operations.
        total_cost: Total expected cognitive cost under this allocation.
        interference_cost: Portion of total cost attributable to
            resource conflicts.
        spare_capacity: Mapping  resource_label → remaining unused
            capacity in [0, 1].
    """

    allocations: Dict[str, Dict[str, float]]
    total_cost: float
    interference_cost: float
    spare_capacity: Dict[str, float]

    @property
    def utilisation(self) -> float:
        """Overall resource utilisation (1 − mean spare capacity)."""
        if not self.spare_capacity:
            return 0.0
        return 1.0 - sum(self.spare_capacity.values()) / len(self.spare_capacity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocations": self.allocations,
            "total_cost": self.total_cost,
            "interference_cost": self.interference_cost,
            "spare_capacity": self.spare_capacity,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResourceAllocation:
        return cls(
            allocations=d["allocations"],
            total_cost=float(d["total_cost"]),
            interference_cost=float(d["interference_cost"]),
            spare_capacity={k: float(v) for k, v in d["spare_capacity"].items()},
        )
