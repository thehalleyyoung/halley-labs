"""
usability_oracle.channel.types — Multiple Resource Theory channel capacity types.

Models Wickens' Multiple Resource Theory (MRT) where human information
processing is constrained by parallel channels (visual, auditory, cognitive,
manual, vocal) with inter-channel interference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from usability_oracle.core.types import Interval


# ═══════════════════════════════════════════════════════════════════════════
# WickensResource — Wickens MRT resource dimensions
# ═══════════════════════════════════════════════════════════════════════════

@unique
class WickensResource(Enum):
    """Resource dimensions from Wickens' Multiple Resource Theory.

    The four dimensions are: processing stage, perceptual modality,
    visual channel (focal vs. ambient), and response code (spatial vs. verbal).
    """

    # Perceptual modality
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"

    # Processing stages
    PERCEPTUAL = "perceptual"
    COGNITIVE = "cognitive"
    RESPONSE = "response"

    # Visual channels
    FOCAL = "focal"
    AMBIENT = "ambient"

    # Response codes
    SPATIAL = "spatial"
    VERBAL = "verbal"

    # Motor effectors
    MANUAL = "manual"
    VOCAL = "vocal"
    PEDAL = "pedal"

    @property
    def dimension(self) -> str:
        """Which MRT dimension this resource belongs to."""
        return _RESOURCE_DIMENSION[self]

    def __str__(self) -> str:
        return self.value


_RESOURCE_DIMENSION: Dict[WickensResource, str] = {
    WickensResource.VISUAL: "modality",
    WickensResource.AUDITORY: "modality",
    WickensResource.TACTILE: "modality",
    WickensResource.PERCEPTUAL: "stage",
    WickensResource.COGNITIVE: "stage",
    WickensResource.RESPONSE: "stage",
    WickensResource.FOCAL: "visual_channel",
    WickensResource.AMBIENT: "visual_channel",
    WickensResource.SPATIAL: "code",
    WickensResource.VERBAL: "code",
    WickensResource.MANUAL: "effector",
    WickensResource.VOCAL: "effector",
    WickensResource.PEDAL: "effector",
}


# ═══════════════════════════════════════════════════════════════════════════
# ResourceChannel — a single resource channel with capacity
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ResourceChannel:
    """A single resource channel in the MRT framework.

    Attributes
    ----------
    resource : WickensResource
        Which Wickens resource this channel represents.
    capacity_bits_per_s : float
        Channel capacity in bits per second.
    current_load : float
        Current utilisation fraction [0, 1].
    capacity_interval : Optional[Interval]
        Uncertainty interval for individual differences in capacity.
    label : str
        Human-readable label.
    """

    resource: WickensResource
    capacity_bits_per_s: float
    current_load: float = 0.0
    capacity_interval: Optional[Interval] = None
    label: str = ""

    @property
    def available_capacity(self) -> float:
        """Remaining capacity in bits/s."""
        return self.capacity_bits_per_s * (1.0 - self.current_load)

    @property
    def is_saturated(self) -> bool:
        """True if the channel is at or above capacity."""
        return self.current_load >= 1.0

    @property
    def utilisation_pct(self) -> float:
        """Current load as a percentage."""
        return self.current_load * 100.0


# ═══════════════════════════════════════════════════════════════════════════
# InterferenceMatrix — pairwise interference between resource channels
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class InterferenceMatrix:
    """Pairwise interference coefficients between resource channels.

    The interference coefficient κ(i, j) ∈ [0, 1] indicates how much
    concurrent demand on channel j degrades throughput on channel i.
    κ = 0 ⟹ fully independent; κ = 1 ⟹ fully conflicting.

    Attributes
    ----------
    resource_labels : tuple[str, ...]
        Ordered labels for each channel dimension.
    coefficients : tuple[tuple[float, ...], ...]
        Symmetric matrix of interference coefficients.
        Shape: (n_channels, n_channels).
    """

    resource_labels: Tuple[str, ...]
    coefficients: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.resource_labels)
        if len(self.coefficients) != n:
            raise ValueError(
                f"Expected {n} rows, got {len(self.coefficients)}"
            )
        for i, row in enumerate(self.coefficients):
            if len(row) != n:
                raise ValueError(
                    f"Row {i} has {len(row)} columns, expected {n}"
                )

    @property
    def n_channels(self) -> int:
        return len(self.resource_labels)

    def interference(self, i: int, j: int) -> float:
        """Get interference coefficient between channels i and j."""
        return self.coefficients[i][j]

    def max_interference_for(self, channel_idx: int) -> float:
        """Maximum interference any other channel exerts on channel_idx."""
        row = self.coefficients[channel_idx]
        return max(row[j] for j in range(self.n_channels) if j != channel_idx)


# ═══════════════════════════════════════════════════════════════════════════
# ChannelAllocation — how task demand is distributed across channels
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ChannelAllocation:
    """Allocation of a task's information-processing demand across channels.

    Attributes
    ----------
    task_id : str
        Task whose demand is being allocated.
    demands : Mapping[WickensResource, float]
        Demand in bits/s on each resource channel.
    total_demand_bits_per_s : float
        Sum of all channel demands.
    bottleneck_resource : Optional[WickensResource]
        The channel closest to saturation, if any.
    effective_throughput_bits_per_s : float
        Throughput after interference, in bits/s.
    """

    task_id: str
    demands: Mapping[WickensResource, float] = field(default_factory=dict)
    total_demand_bits_per_s: float = 0.0
    bottleneck_resource: Optional[WickensResource] = None
    effective_throughput_bits_per_s: float = 0.0

    @property
    def active_channels(self) -> int:
        """Number of channels with non-zero demand."""
        return sum(1 for d in self.demands.values() if d > 0)


# ═══════════════════════════════════════════════════════════════════════════
# ResourcePool — complete set of channels for an agent
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ResourcePool:
    """Complete set of resource channels for a human operator.

    Represents the full MRT resource budget, potentially parametrised
    by individual differences (age, expertise, disability).

    Attributes
    ----------
    channels : tuple[ResourceChannel, ...]
        All available resource channels.
    interference : Optional[InterferenceMatrix]
        Pairwise interference model (None if channels are assumed independent).
    population_percentile : float
        Individual-difference percentile [0, 100] this pool represents.
    label : str
        Description (e.g. "average adult", "novice user", "screen-reader user").
    """

    channels: Tuple[ResourceChannel, ...] = ()
    interference: Optional[InterferenceMatrix] = None
    population_percentile: float = 50.0
    label: str = ""

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def total_capacity_bits_per_s(self) -> float:
        """Sum of all channel capacities (theoretical maximum, ignoring interference)."""
        return sum(ch.capacity_bits_per_s for ch in self.channels)

    @property
    def saturated_channels(self) -> Tuple[ResourceChannel, ...]:
        """Channels that are at or above capacity."""
        return tuple(ch for ch in self.channels if ch.is_saturated)

    def channel_by_resource(self, resource: WickensResource) -> Optional[ResourceChannel]:
        """Look up a channel by its Wickens resource type."""
        for ch in self.channels:
            if ch.resource == resource:
                return ch
        return None


__all__ = [
    "ChannelAllocation",
    "InterferenceMatrix",
    "ResourceChannel",
    "ResourcePool",
    "WickensResource",
]
