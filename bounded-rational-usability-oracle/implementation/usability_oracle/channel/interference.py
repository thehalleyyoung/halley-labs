"""
usability_oracle.channel.interference — Channel interference modelling.

Implements Wickens' four-dimensional resource model for computing
interference between concurrent information-processing channels:

1. **Processing stage**: perceptual / cognitive / response
2. **Perceptual modality**: visual / auditory
3. **Visual channel**: focal / ambient
4. **Processing code**: spatial / verbal

Cross-resource interference is computed from the product of per-dimension
conflict weights, calibrated from published MRT data (Wickens 2002, 2008).

References
----------
- Wickens, C. D. (2002). Multiple resources and performance prediction.
  Theoretical Issues in Ergonomics Science, 3(2), 159–177.
- Wickens, C. D. (2008). Multiple resources and mental workload.
  Human Factors, 50(3), 449–455.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from usability_oracle.channel.types import (
    ChannelAllocation,
    InterferenceMatrix,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)


# ═══════════════════════════════════════════════════════════════════════════
# Wickens dimension weights — calibrated from published data
# ═══════════════════════════════════════════════════════════════════════════

# Same-stage conflict weights.
_STAGE_CONFLICT: Dict[Tuple[WickensResource, WickensResource], float] = {
    (WickensResource.PERCEPTUAL, WickensResource.PERCEPTUAL): 0.70,
    (WickensResource.COGNITIVE,  WickensResource.COGNITIVE):  1.00,
    (WickensResource.RESPONSE,   WickensResource.RESPONSE):   0.60,
}

# Same-modality conflict weights.
_MODALITY_CONFLICT: Dict[Tuple[WickensResource, WickensResource], float] = {
    (WickensResource.VISUAL,   WickensResource.VISUAL):   1.00,
    (WickensResource.AUDITORY, WickensResource.AUDITORY):  0.80,
    (WickensResource.TACTILE,  WickensResource.TACTILE):   0.70,
    (WickensResource.VISUAL,   WickensResource.AUDITORY):  0.20,
    (WickensResource.AUDITORY, WickensResource.VISUAL):    0.20,
    (WickensResource.VISUAL,   WickensResource.TACTILE):   0.15,
    (WickensResource.TACTILE,  WickensResource.VISUAL):    0.15,
    (WickensResource.AUDITORY, WickensResource.TACTILE):   0.15,
    (WickensResource.TACTILE,  WickensResource.AUDITORY):  0.15,
}

# Visual channel conflict weights (only meaningful for visual modality).
_VISUAL_CHANNEL_CONFLICT: Dict[Tuple[WickensResource, WickensResource], float] = {
    (WickensResource.FOCAL,   WickensResource.FOCAL):   1.00,
    (WickensResource.AMBIENT, WickensResource.AMBIENT):  0.70,
    (WickensResource.FOCAL,   WickensResource.AMBIENT):  0.30,
    (WickensResource.AMBIENT, WickensResource.FOCAL):    0.30,
}

# Processing-code conflict weights.
_CODE_CONFLICT: Dict[Tuple[WickensResource, WickensResource], float] = {
    (WickensResource.SPATIAL, WickensResource.SPATIAL): 1.00,
    (WickensResource.VERBAL,  WickensResource.VERBAL):  0.90,
    (WickensResource.SPATIAL, WickensResource.VERBAL):  0.20,
    (WickensResource.VERBAL,  WickensResource.SPATIAL): 0.20,
}

# Motor effector conflict weights.
_EFFECTOR_CONFLICT: Dict[Tuple[WickensResource, WickensResource], float] = {
    (WickensResource.MANUAL, WickensResource.MANUAL): 1.00,
    (WickensResource.VOCAL,  WickensResource.VOCAL):  0.90,
    (WickensResource.PEDAL,  WickensResource.PEDAL):  0.80,
    (WickensResource.MANUAL, WickensResource.VOCAL):  0.15,
    (WickensResource.VOCAL,  WickensResource.MANUAL): 0.15,
    (WickensResource.MANUAL, WickensResource.PEDAL):  0.10,
    (WickensResource.PEDAL,  WickensResource.MANUAL): 0.10,
    (WickensResource.VOCAL,  WickensResource.PEDAL):  0.10,
    (WickensResource.PEDAL,  WickensResource.VOCAL):  0.10,
}

# Map each WickensResource to which dimension-conflict table it belongs.
_DIMENSION_TABLE: Dict[str, Dict[Tuple[WickensResource, WickensResource], float]] = {
    "stage":          _STAGE_CONFLICT,
    "modality":       _MODALITY_CONFLICT,
    "visual_channel": _VISUAL_CHANNEL_CONFLICT,
    "code":           _CODE_CONFLICT,
    "effector":       _EFFECTOR_CONFLICT,
}


# ═══════════════════════════════════════════════════════════════════════════
# Resource profile — multi-dimensional task description
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Multi-dimensional resource profile for a task.

    Describes which Wickens dimensions a task occupies and the demand
    level on each dimension.

    Attributes
    ----------
    stage : WickensResource
        Processing stage (PERCEPTUAL, COGNITIVE, RESPONSE).
    modality : WickensResource or None
        Perceptual modality (VISUAL, AUDITORY, TACTILE).
    visual_channel : WickensResource or None
        Visual sub-channel (FOCAL, AMBIENT). Only when modality = VISUAL.
    code : WickensResource or None
        Processing code (SPATIAL, VERBAL).
    effector : WickensResource or None
        Motor effector (MANUAL, VOCAL, PEDAL).
    demand : float
        Demand intensity in [0, 1].
    """

    stage: WickensResource = WickensResource.COGNITIVE
    modality: Optional[WickensResource] = None
    visual_channel: Optional[WickensResource] = None
    code: Optional[WickensResource] = None
    effector: Optional[WickensResource] = None
    demand: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise interference computation
# ═══════════════════════════════════════════════════════════════════════════

def dimension_conflict(
    ra: WickensResource,
    rb: WickensResource,
) -> float:
    """Compute conflict weight between two resources on the same dimension.

    If the two resources belong to different dimensions the conflict is
    zero (orthogonal resources).  If they are on the same dimension,
    return the calibrated weight.

    Parameters
    ----------
    ra, rb : WickensResource
        Two resource identifiers.

    Returns
    -------
    float
        Conflict weight in [0, 1].
    """
    dim_a = ra.dimension
    dim_b = rb.dimension
    if dim_a != dim_b:
        return 0.0
    table = _DIMENSION_TABLE.get(dim_a, {})
    return table.get((ra, rb), 0.0)


def profile_interference(
    pa: ResourceProfile,
    pb: ResourceProfile,
) -> float:
    """Compute the Wickens interference between two resource profiles.

    The interference is the product of per-dimension conflict weights,
    scaled by the demand levels of both tasks:

        I(a, b) = d_a · d_b · Π_dim w_dim(a, b)

    If a dimension is None for either task, it contributes weight 1.0
    (no evidence of reduced conflict on that dimension).

    Parameters
    ----------
    pa, pb : ResourceProfile
        Multi-dimensional resource profiles for two concurrent tasks.

    Returns
    -------
    float
        Interference score in [0, 1].
    """
    w = 1.0

    # Stage dimension.
    w *= _STAGE_CONFLICT.get((pa.stage, pb.stage), 0.0)
    if w < 1e-12:
        return 0.0

    # Modality dimension.
    if pa.modality is not None and pb.modality is not None:
        w *= _MODALITY_CONFLICT.get((pa.modality, pb.modality), 0.2)
    # else: unknown modality → no reduction

    # Visual channel.
    if pa.visual_channel is not None and pb.visual_channel is not None:
        w *= _VISUAL_CHANNEL_CONFLICT.get(
            (pa.visual_channel, pb.visual_channel), 0.3,
        )

    # Code.
    if pa.code is not None and pb.code is not None:
        w *= _CODE_CONFLICT.get((pa.code, pb.code), 0.2)

    # Effector.
    if pa.effector is not None and pb.effector is not None:
        w *= _EFFECTOR_CONFLICT.get((pa.effector, pb.effector), 0.1)

    return pa.demand * pb.demand * w


# ═══════════════════════════════════════════════════════════════════════════
# WickensInterferenceModel — protocol implementation
# ═══════════════════════════════════════════════════════════════════════════

class WickensInterferenceModel:
    """Compute interference between MRT channel allocations.

    Implements the ``InterferenceModel`` protocol from
    ``usability_oracle.channel.protocols``.
    """

    def __init__(
        self,
        stage_weights: Optional[Dict[Tuple[WickensResource, WickensResource], float]] = None,
        modality_weights: Optional[Dict[Tuple[WickensResource, WickensResource], float]] = None,
        code_weights: Optional[Dict[Tuple[WickensResource, WickensResource], float]] = None,
        channel_weights: Optional[Dict[Tuple[WickensResource, WickensResource], float]] = None,
        effector_weights: Optional[Dict[Tuple[WickensResource, WickensResource], float]] = None,
    ) -> None:
        if stage_weights is not None:
            _STAGE_CONFLICT.update(stage_weights)
        if modality_weights is not None:
            _MODALITY_CONFLICT.update(modality_weights)
        if code_weights is not None:
            _CODE_CONFLICT.update(code_weights)
        if channel_weights is not None:
            _VISUAL_CHANNEL_CONFLICT.update(channel_weights)
        if effector_weights is not None:
            _EFFECTOR_CONFLICT.update(effector_weights)

    # ---- InterferenceModel protocol ------------------------------------

    def compute_interference(
        self,
        allocation_a: ChannelAllocation,
        allocation_b: ChannelAllocation,
    ) -> float:
        """Compute aggregate interference between two allocations.

        For each pair of active resources (one from each allocation),
        the structural interference is the dimension-conflict weight
        scaled by both demands.  The aggregate is the sum over all
        cross-pairs, clamped to [0, 1].

        Parameters
        ----------
        allocation_a, allocation_b : ChannelAllocation

        Returns
        -------
        float
            Aggregate interference in [0, 1].
        """
        total = 0.0
        for ra, da in allocation_a.demands.items():
            for rb, db in allocation_b.demands.items():
                w = dimension_conflict(ra, rb)
                total += da * db * w
        return min(total, 1.0)

    def build_interference_matrix(
        self,
        resources: Sequence[WickensResource],
    ) -> InterferenceMatrix:
        """Construct the full pairwise interference matrix.

        Parameters
        ----------
        resources : Sequence[WickensResource]
            Channels to include.

        Returns
        -------
        InterferenceMatrix
            Symmetric matrix with κ(i, j) = dimension_conflict(r_i, r_j).
        """
        n = len(resources)
        labels = tuple(r.value for r in resources)
        coeffs: List[Tuple[float, ...]] = []
        for i in range(n):
            row: List[float] = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                else:
                    row.append(dimension_conflict(resources[i], resources[j]))
            coeffs.append(tuple(row))

        return InterferenceMatrix(
            resource_labels=labels,
            coefficients=tuple(coeffs),
        )

    def effective_capacity(
        self,
        channel: ResourceChannel,
        concurrent_load: Mapping[WickensResource, float],
        interference: InterferenceMatrix,
    ) -> float:
        """Compute effective capacity under concurrent interference.

        Effective capacity is the base capacity reduced by the weighted
        interference from all other concurrently loaded channels:

            C_eff(i) = C_base(i) · (1 − Σ_j κ(i,j) · load(j))

        Clamped so effective capacity ≥ 5 % of base.

        Parameters
        ----------
        channel : ResourceChannel
        concurrent_load : Mapping[WickensResource, float]
        interference : InterferenceMatrix

        Returns
        -------
        float
            Effective capacity in bits/s.
        """
        base = channel.capacity_bits_per_s
        label = channel.resource.value
        if label not in interference.resource_labels:
            return base

        idx = interference.resource_labels.index(label)
        penalty = 0.0
        for other_res, load in concurrent_load.items():
            other_label = other_res.value
            if other_label not in interference.resource_labels:
                continue
            j = interference.resource_labels.index(other_label)
            penalty += interference.coefficients[idx][j] * load

        return base * max(1.0 - penalty, 0.05)

    # ---- Temporal overlap interference ---------------------------------

    def temporal_overlap_interference(
        self,
        allocation_a: ChannelAllocation,
        allocation_b: ChannelAllocation,
        overlap_fraction: float,
    ) -> float:
        """Compute interference modulated by temporal overlap.

        If two tasks overlap 100 % in time, interference is full.
        If overlap is 0 %, interference is zero (serial execution).

        Parameters
        ----------
        allocation_a, allocation_b : ChannelAllocation
        overlap_fraction : float
            Fraction of time the two tasks are concurrent, in [0, 1].

        Returns
        -------
        float
            Temporally weighted interference in [0, 1].
        """
        overlap_fraction = max(0.0, min(1.0, overlap_fraction))
        structural = self.compute_interference(allocation_a, allocation_b)
        return structural * overlap_fraction

    # ---- Time-sharing efficiency ---------------------------------------

    def time_sharing_efficiency(
        self,
        allocations: Sequence[ChannelAllocation],
    ) -> float:
        """Estimate overall time-sharing efficiency for concurrent tasks.

        η = 1 / (1 + Σ_{i<j} I(i,j))

        where I(i,j) is the pairwise interference.

        Returns
        -------
        float
            Efficiency in (0, 1].  1.0 = perfect independence.
        """
        total_intf = 0.0
        n = len(allocations)
        for i in range(n):
            for j in range(i + 1, n):
                total_intf += self.compute_interference(
                    allocations[i], allocations[j],
                )
        return 1.0 / (1.0 + total_intf)

    # ---- Structural interference from shared pathways ------------------

    def structural_interference(
        self,
        resources_a: Set[WickensResource],
        resources_b: Set[WickensResource],
    ) -> float:
        """Compute structural interference from shared neural pathways.

        Two tasks that use the same set of MRT resources (same stage,
        modality, code, channel) have maximum structural interference.
        The metric is the Jaccard similarity of the resource sets weighted
        by dimension-conflict.

        Parameters
        ----------
        resources_a, resources_b : Set[WickensResource]
            Resources used by each task.

        Returns
        -------
        float
            Structural interference in [0, 1].
        """
        if not resources_a or not resources_b:
            return 0.0
        shared = resources_a & resources_b
        union = resources_a | resources_b
        if not union:
            return 0.0
        # Weighted by same-dimension conflict.
        w_shared = sum(dimension_conflict(r, r) for r in shared)
        w_union = sum(
            max(dimension_conflict(r, r), 0.01) for r in union
        )
        return w_shared / w_union if w_union > 0 else 0.0

    # ---- Interference cost for cost-algebra integration ----------------

    def interference_cost(
        self,
        allocation_a: ChannelAllocation,
        allocation_b: ChannelAllocation,
    ) -> float:
        """Compute interference cost suitable for cost-algebra integration.

        This returns the interference as a scaling factor η ∈ [0, 1] that
        can be passed directly to the parallel composition operator ⊗ in
        the cost algebra:

            cost(a ⊗ b) = max(μ_a, μ_b) + η · min(μ_a, μ_b)

        Parameters
        ----------
        allocation_a, allocation_b : ChannelAllocation

        Returns
        -------
        float
            Interference factor η ∈ [0, 1].
        """
        return self.compute_interference(allocation_a, allocation_b)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: build interference matrix for all standard resources
# ═══════════════════════════════════════════════════════════════════════════

def build_standard_interference_matrix() -> InterferenceMatrix:
    """Build the full interference matrix for all WickensResource values.

    Returns
    -------
    InterferenceMatrix
        13 × 13 matrix covering all enumerated resources.
    """
    model = WickensInterferenceModel()
    all_resources = list(WickensResource)
    return model.build_interference_matrix(all_resources)


__all__ = [
    "ResourceProfile",
    "WickensInterferenceModel",
    "build_standard_interference_matrix",
    "dimension_conflict",
    "profile_interference",
]
