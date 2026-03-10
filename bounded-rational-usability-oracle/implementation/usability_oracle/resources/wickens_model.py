"""
usability_oracle.resources.wickens_model — Wickens Multiple Resource Theory.

Implements the four-dimensional cognitive resource model:
    1. Stage: perceptual / cognitive / response
    2. Modality: visual / auditory
    3. Code: spatial / verbal
    4. Channel: focal / ambient (visual only)

Provides interference computation based on published weights
(Wickens 2002, 2008).

Reference:
    Wickens, C. D. (2002). Multiple resources and performance prediction.
    Theoretical Issues in Ergonomics Science, 3(2), 159–177.

    Wickens, C. D. (2008). Multiple resources and mental workload.
    Human Factors, 50(3), 449–455.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

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


# ═══════════════════════════════════════════════════════════════════════════
# Published interference weights (Wickens 2002, 2008)
# ═══════════════════════════════════════════════════════════════════════════

# Same-dimension conflict multipliers.
# When two tasks share a dimension level, the interference is multiplied
# by the weight for that dimension.  When they differ on a dimension,
# the weight is 0 (no structural interference from that dimension).

# Stage dimension: tasks sharing the same stage interfere.
# Perception-perception is moderate; cognition-cognition is highest.
STAGE_WEIGHTS: Dict[ProcessingStage, float] = {
    ProcessingStage.PERCEPTION: 0.7,
    ProcessingStage.COGNITION: 1.0,
    ProcessingStage.RESPONSE: 0.6,
}

# Modality dimension: visual-visual is highest conflict.
MODALITY_WEIGHTS: Dict[Tuple[PerceptualModality, PerceptualModality], float] = {
    (PerceptualModality.VISUAL, PerceptualModality.VISUAL): 1.0,
    (PerceptualModality.AUDITORY, PerceptualModality.AUDITORY): 0.8,
    (PerceptualModality.VISUAL, PerceptualModality.AUDITORY): 0.2,
    (PerceptualModality.AUDITORY, PerceptualModality.VISUAL): 0.2,
}

# Channel dimension (visual only): focal-focal is high.
CHANNEL_WEIGHTS: Dict[Tuple[Optional[VisualChannel], Optional[VisualChannel]], float] = {
    (VisualChannel.FOCAL, VisualChannel.FOCAL): 1.0,
    (VisualChannel.AMBIENT, VisualChannel.AMBIENT): 0.7,
    (VisualChannel.FOCAL, VisualChannel.AMBIENT): 0.3,
    (VisualChannel.AMBIENT, VisualChannel.FOCAL): 0.3,
}

# Code dimension: spatial-spatial or verbal-verbal.
CODE_WEIGHTS: Dict[Tuple[ProcessingCode, ProcessingCode], float] = {
    (ProcessingCode.SPATIAL, ProcessingCode.SPATIAL): 1.0,
    (ProcessingCode.VERBAL, ProcessingCode.VERBAL): 0.9,
    (ProcessingCode.SPATIAL, ProcessingCode.VERBAL): 0.2,
    (ProcessingCode.VERBAL, ProcessingCode.SPATIAL): 0.2,
}

# Default capacity for each resource type (normalised 0-1).
DEFAULT_CAPACITY = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# WickensModel
# ═══════════════════════════════════════════════════════════════════════════

class WickensModel:
    """Wickens Multiple Resource Theory implementation.

    Computes demand vectors, interference, time-sharing efficiency,
    and dual-task costs for concurrent UI operations.
    """

    def __init__(
        self,
        stage_weights: Optional[Dict[ProcessingStage, float]] = None,
        modality_weights: Optional[Dict[Tuple[PerceptualModality, PerceptualModality], float]] = None,
        code_weights: Optional[Dict[Tuple[ProcessingCode, ProcessingCode], float]] = None,
        channel_weights: Optional[Dict[Tuple[Optional[VisualChannel], Optional[VisualChannel]], float]] = None,
    ) -> None:
        self._stage_w = stage_weights or STAGE_WEIGHTS
        self._modality_w = modality_weights or MODALITY_WEIGHTS
        self._code_w = code_weights or CODE_WEIGHTS
        self._channel_w = channel_weights or CHANNEL_WEIGHTS

    # -------------------------------------------------------------------
    # Demand vector computation
    # -------------------------------------------------------------------

    def compute_demand_vector(
        self,
        task_component: Dict[str, object],
    ) -> DemandVector:
        """Compute the resource demand vector for a task component.

        Parameters:
            task_component: Dict with keys:
                - "operation_id": str
                - "stage": str (perception/cognition/response)
                - "modality": str (visual/auditory) [optional]
                - "code": str (spatial/verbal) [optional]
                - "channel": str (focal/ambient) [optional for visual]
                - "demand_level": float in [0,1]

        Returns:
            DemandVector with a single ResourceDemand.
        """
        op_id = str(task_component.get("operation_id", ""))
        stage = ProcessingStage(str(task_component.get("stage", "cognition")))
        demand_level = float(task_component.get("demand_level", 0.5))  # type: ignore[arg-type]
        demand_level = max(0.0, min(1.0, demand_level))

        modality = None
        mod_str = task_component.get("modality")
        if mod_str is not None:
            modality = PerceptualModality(str(mod_str))

        code = ProcessingCode(str(task_component.get("code", "spatial")))

        channel = None
        ch_str = task_component.get("channel")
        if ch_str is not None:
            channel = VisualChannel(str(ch_str))

        resource = Resource(
            stage=stage,
            modality=modality,
            visual_channel=channel,
            code=code,
            label=f"{stage.value}_{modality.value if modality else 'none'}_{code.value}",
        )
        rd = ResourceDemand(
            resource=resource,
            demand_level=demand_level,
            operation_id=op_id,
        )
        return DemandVector(demands=(rd,), operation_id=op_id)

    # -------------------------------------------------------------------
    # Interference between two resource demands
    # -------------------------------------------------------------------

    def _resource_interference_weight(
        self, ra: Resource, rb: Resource,
    ) -> float:
        """Compute the Wickens interference weight between two resources.

        The interference is the product of per-dimension weights:
            w = w_stage · w_modality · w_code · w_channel

        Dimensions that are None on either side contribute a weight of 1.0
        (no evidence of reduced conflict).
        """
        # Stage must match for there to be structural interference
        if ra.stage != rb.stage:
            return 0.0
        w_stage = self._stage_w.get(ra.stage, 0.5)

        # Modality
        if ra.modality is not None and rb.modality is not None:
            w_mod = self._modality_w.get((ra.modality, rb.modality), 0.2)
        else:
            w_mod = 0.5  # unknown modality

        # Code
        w_code = self._code_w.get((ra.code, rb.code), 0.2)

        # Channel (only relevant for visual)
        if ra.visual_channel is not None and rb.visual_channel is not None:
            w_chan = self._channel_w.get(
                (ra.visual_channel, rb.visual_channel), 0.3,
            )
        else:
            w_chan = 1.0  # no channel info → no channel-based reduction

        return w_stage * w_mod * w_code * w_chan

    def compute_interference(
        self, demand_a: ResourceDemand, demand_b: ResourceDemand,
    ) -> float:
        """Compute interference cost between two resource demands.

        interference = demand_a · demand_b · w(resource_a, resource_b)

        Parameters:
            demand_a: First resource demand.
            demand_b: Second resource demand.

        Returns:
            Interference cost in [0, 1].
        """
        w = self._resource_interference_weight(
            demand_a.resource, demand_b.resource,
        )
        return demand_a.demand_level * demand_b.demand_level * w

    # -------------------------------------------------------------------
    # Pairwise conflicts between demand vectors
    # -------------------------------------------------------------------

    def compute_pairwise(
        self, dv_a: DemandVector, dv_b: DemandVector,
    ) -> List[ResourceConflict]:
        """Compute all resource conflicts between two demand vectors.

        Returns:
            List of ResourceConflict, sorted by severity descending.
        """
        conflicts: List[ResourceConflict] = []
        for da in dv_a.demands:
            for db in dv_b.demands:
                severity = self.compute_interference(da, db)
                if severity > 1e-9:
                    conflicts.append(ResourceConflict(
                        resource=da.resource,
                        operation_a=dv_a.operation_id,
                        operation_b=dv_b.operation_id,
                        demand_a=da.demand_level,
                        demand_b=db.demand_level,
                        conflict_severity=severity,
                        description=(
                            f"{da.resource.stage.value} conflict: "
                            f"{da.resource.code.value} demand "
                            f"{da.demand_level:.2f} × {db.demand_level:.2f}"
                        ),
                    ))
        conflicts.sort(key=lambda c: c.conflict_severity, reverse=True)
        return conflicts

    # -------------------------------------------------------------------
    # Time-sharing efficiency
    # -------------------------------------------------------------------

    def compute_time_sharing_efficiency(
        self, demands: Sequence[DemandVector],
    ) -> float:
        """Compute overall time-sharing efficiency for concurrent tasks.

        Efficiency η = 1 / (1 + total_interference)

        η = 1.0 means no interference (perfect time sharing).
        η → 0.0 means severe interference.
        """
        total = self.total_interference(demands)
        return 1.0 / (1.0 + total)

    # -------------------------------------------------------------------
    # Dual-task cost
    # -------------------------------------------------------------------

    def predict_dual_task_cost(
        self,
        task_a: DemandVector,
        task_b: DemandVector,
    ) -> float:
        """Predict the cost of performing two tasks concurrently.

        cost = base_cost_a + base_cost_b + interference_cost

        where base cost is the total demand and interference cost is
        the sum of pairwise resource conflicts.
        """
        base_a = task_a.total_demand
        base_b = task_b.total_demand
        conflicts = self.compute_pairwise(task_a, task_b)
        interference = sum(c.conflict_severity for c in conflicts)
        return base_a + base_b + interference

    # -------------------------------------------------------------------
    # Difficulty index
    # -------------------------------------------------------------------

    def difficulty_index(self, demand: DemandVector) -> float:
        """Compute scalar difficulty from a demand vector.

        DI = √(Σ dᵢ²)  (L2 norm of demands)

        A single scalar capturing the overall resource load.
        """
        if not demand.demands:
            return 0.0
        arr = np.array([d.demand_level for d in demand.demands])
        return float(np.linalg.norm(arr))

    # -------------------------------------------------------------------
    # InterferenceComputer protocol methods
    # -------------------------------------------------------------------

    def compute_matrix(
        self, demand_vectors: Sequence[DemandVector],
    ) -> InterferenceMatrix:
        """Compute full pairwise interference matrix."""
        n = len(demand_vectors)
        op_ids = tuple(dv.operation_id for dv in demand_vectors)
        upper_tri: List[float] = []
        all_conflicts: List[ResourceConflict] = []

        for i in range(n):
            for j in range(i + 1, n):
                conflicts = self.compute_pairwise(
                    demand_vectors[i], demand_vectors[j],
                )
                cost = sum(c.conflict_severity for c in conflicts)
                upper_tri.append(cost)
                all_conflicts.extend(conflicts)

        return InterferenceMatrix(
            operation_ids=op_ids,
            matrix=tuple(upper_tri),
            conflicts=tuple(all_conflicts),
        )

    def total_interference(
        self, demand_vectors: Sequence[DemandVector],
    ) -> float:
        """Total interference cost across all pairs."""
        mat = self.compute_matrix(demand_vectors)
        return sum(mat.matrix)
