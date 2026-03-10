"""
usability_oracle.channel.capacity — Channel capacity estimation.

Estimates per-channel information-processing capacity based on established
human-factors models:

* **Visual**: display complexity (number of elements, grouping, eccentricity)
* **Auditory**: signal-to-noise ratio, frequency bandwidth
* **Motor**: Fitts' law throughput as channel capacity
* **Cognitive**: working-memory capacity (Cowan's 4 ± 1 chunks)

Capacity varies with fatigue, expertise, and individual differences.

References
----------
- Fitts, P. M. (1954). The information capacity of the human motor system.
- Cowan, N. (2001). The magical number 4 in short-term memory.
- Hick, W. E. (1952). On the rate of gain of information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.channel.types import (
    ChannelAllocation,
    InterferenceMatrix,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)
from usability_oracle.core.types import Interval


# ═══════════════════════════════════════════════════════════════════════════
# Published baseline capacities (bits/s) — median adult
# ═══════════════════════════════════════════════════════════════════════════

# Visual capacity from Card, Moran & Newell (1983) and related literature.
_BASELINE_CAPACITY: Dict[WickensResource, float] = {
    WickensResource.VISUAL:     39.0,   # ~39 bits/s visual search
    WickensResource.AUDITORY:   32.0,   # ~32 bits/s speech recognition
    WickensResource.TACTILE:    12.0,   # ~12 bits/s tactile
    WickensResource.PERCEPTUAL: 45.0,   # perceptual stage aggregate
    WickensResource.COGNITIVE:  16.0,   # ~16 bits/s Hick-Hyman
    WickensResource.RESPONSE:   25.0,   # response selection
    WickensResource.FOCAL:      30.0,   # foveal processing
    WickensResource.AMBIENT:    15.0,   # peripheral/ambient
    WickensResource.SPATIAL:    20.0,   # spatial working memory
    WickensResource.VERBAL:     18.0,   # verbal/phonological loop
    WickensResource.MANUAL:     10.5,   # Fitts' law throughput
    WickensResource.VOCAL:       8.0,   # vocal response
    WickensResource.PEDAL:       5.0,   # foot controls
}

# Standard deviation of capacity across population (bits/s).
_CAPACITY_SD: Dict[WickensResource, float] = {
    r: c * 0.20 for r, c in _BASELINE_CAPACITY.items()
}

# Expertise multipliers relative to median.
_EXPERTISE_MULTIPLIER: Dict[str, float] = {
    "novice":       0.70,
    "intermediate": 1.00,
    "expert":       1.30,
}


# ═══════════════════════════════════════════════════════════════════════════
# Visual capacity model
# ═══════════════════════════════════════════════════════════════════════════

def visual_capacity(
    n_elements: int = 10,
    grouping_factor: float = 1.0,
    eccentricity_deg: float = 0.0,
    display_clutter: float = 0.0,
) -> float:
    """Estimate visual channel capacity in bits/s.

    Parameters
    ----------
    n_elements : int
        Number of distinct visual elements in the display.
    grouping_factor : float
        Gestalt grouping benefit in (0, 1]. 1.0 = no grouping benefit,
        lower values indicate strong grouping (fewer effective elements).
    eccentricity_deg : float
        Average eccentricity from fovea in degrees. Capacity degrades
        ~50 % per 10° eccentricity (Anstis, 1974).
    display_clutter : float
        Clutter score in [0, 1]. Reduces capacity via crowding.

    Returns
    -------
    float
        Estimated visual capacity in bits/s.
    """
    base = _BASELINE_CAPACITY[WickensResource.VISUAL]
    # Effective element count after grouping.
    n_eff = max(1.0, n_elements * grouping_factor)
    # log2 scaling — capacity needed grows with log of alternatives.
    info_load = math.log2(n_eff) if n_eff > 1 else 0.0
    # Normalise to a fraction of baseline (10 elements ≈ baseline).
    load_fraction = info_load / math.log2(10.0)
    capacity = base / max(load_fraction, 0.3)
    # Eccentricity decay.
    ecc_decay = math.exp(-0.07 * eccentricity_deg)
    capacity *= ecc_decay
    # Clutter penalty.
    capacity *= (1.0 - 0.5 * max(0.0, min(1.0, display_clutter)))
    return max(capacity, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Auditory capacity model
# ═══════════════════════════════════════════════════════════════════════════

def auditory_capacity(
    snr_db: float = 20.0,
    bandwidth_hz: float = 4000.0,
    n_streams: int = 1,
) -> float:
    """Estimate auditory channel capacity in bits/s.

    Uses a simplified Shannon capacity model adapted for the human
    auditory system, with corrections for multi-stream (cocktail party)
    degradation.

    Parameters
    ----------
    snr_db : float
        Signal-to-noise ratio in dB.
    bandwidth_hz : float
        Effective bandwidth in Hz (speech ≈ 300–3400 Hz).
    n_streams : int
        Number of concurrent auditory streams (cocktail-party effect).

    Returns
    -------
    float
        Estimated auditory capacity in bits/s.
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    # Shannon capacity for the channel.
    shannon_cap = bandwidth_hz * math.log2(1.0 + snr_linear)
    # Human bottleneck: cap at ~50 bits/s for speech, scale down.
    human_cap = min(shannon_cap, 50.0)
    # Multi-stream degradation: capacity drops ~40 % per additional stream.
    if n_streams > 1:
        human_cap *= 0.6 ** (n_streams - 1)
    return max(human_cap, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Motor capacity — Fitts' law throughput
# ═══════════════════════════════════════════════════════════════════════════

def motor_capacity_fitts(
    target_width_px: float = 40.0,
    target_distance_px: float = 200.0,
    movement_time_s: Optional[float] = None,
) -> float:
    """Estimate motor channel capacity via Fitts' law throughput.

    Throughput TP = ID / MT where ID = log₂(D/W + 1) (Shannon formulation).

    Parameters
    ----------
    target_width_px : float
        Target width in pixels.
    target_distance_px : float
        Movement distance in pixels.
    movement_time_s : float or None
        Observed movement time. If None, use empirical average
        (Soukoreff & MacKenzie 2004: ~1050 ms for ID ≈ 4 bits).

    Returns
    -------
    float
        Throughput in bits/s.
    """
    if target_width_px <= 0:
        target_width_px = 1.0
    index_of_difficulty = math.log2(target_distance_px / target_width_px + 1.0)
    if movement_time_s is not None and movement_time_s > 0:
        return index_of_difficulty / movement_time_s
    # Empirical average throughput ~4.9 bits/s (ISO 9241-9 mouse).
    return 4.9


def cognitive_capacity_wm(
    n_chunks: int = 4,
    chunk_bits: float = 3.5,
    retention_s: float = 18.0,
) -> float:
    """Estimate cognitive channel capacity from working-memory model.

    Uses Cowan's (2001) model: capacity ≈ 4 ± 1 chunks, each encoding
    ~3–4 bits, retained for ~18 s without rehearsal.

    Parameters
    ----------
    n_chunks : int
        Working memory chunk capacity (Cowan's K).
    chunk_bits : float
        Information per chunk in bits.
    retention_s : float
        Retention duration in seconds.

    Returns
    -------
    float
        Cognitive throughput in bits/s.
    """
    total_bits = n_chunks * chunk_bits
    return total_bits / retention_s


# ═══════════════════════════════════════════════════════════════════════════
# Capacity degradation and variation
# ═══════════════════════════════════════════════════════════════════════════

def fatigue_degradation(
    base_capacity: float,
    time_on_task_min: float,
    half_life_min: float = 45.0,
) -> float:
    """Apply exponential fatigue degradation to capacity.

    Capacity decays exponentially with time-on-task.  Empirical half-life
    is approximately 45 minutes for sustained vigilance (Warm et al., 2008).

    Parameters
    ----------
    base_capacity : float
        Capacity at start of task (bits/s).
    time_on_task_min : float
        Minutes of sustained performance.
    half_life_min : float
        Time for capacity to halve. Default 45 min.

    Returns
    -------
    float
        Degraded capacity in bits/s.
    """
    decay = math.exp(-math.log(2.0) * time_on_task_min / half_life_min)
    return base_capacity * max(decay, 0.25)  # floor at 25 % of base


def age_adjustment(
    base_capacity: float,
    age_years: float,
    resource: WickensResource,
) -> float:
    """Adjust capacity for age-related changes.

    Uses published decline curves:
    - Cognitive peaks ~25, declines ~0.5 %/yr after 30 (Salthouse, 2004).
    - Visual acuity declines faster after 50.
    - Motor throughput declines ~0.3 %/yr after 40.

    Parameters
    ----------
    base_capacity : float
        Baseline capacity (age 25–30).
    age_years : float
        User's age in years.
    resource : WickensResource
        Resource type (determines decline curve).

    Returns
    -------
    float
        Age-adjusted capacity.
    """
    if age_years <= 30.0:
        return base_capacity

    excess = age_years - 30.0
    dim = resource.dimension

    if dim == "modality" and resource == WickensResource.VISUAL:
        # Visual acuity: 0.6 % / yr, accelerating after 50.
        rate = 0.006 if age_years < 50 else 0.012
    elif dim in ("stage",) and resource == WickensResource.COGNITIVE:
        rate = 0.005  # 0.5 % / yr cognitive
    elif dim == "effector":
        rate = 0.003  # 0.3 % / yr motor
    else:
        rate = 0.004  # generic

    return base_capacity * max(1.0 - rate * excess, 0.30)


def expertise_adjustment(
    base_capacity: float,
    expertise_level: str,
) -> float:
    """Adjust capacity for expertise / skill level.

    Parameters
    ----------
    base_capacity : float
        Baseline capacity.
    expertise_level : str
        One of "novice", "intermediate", "expert".

    Returns
    -------
    float
        Adjusted capacity.
    """
    mult = _EXPERTISE_MULTIPLIER.get(expertise_level, 1.0)
    return base_capacity * mult


# ═══════════════════════════════════════════════════════════════════════════
# ChannelCapacityEstimator — full estimator (implements CapacityEstimator)
# ═══════════════════════════════════════════════════════════════════════════

class ChannelCapacityEstimator:
    """Estimate per-channel capacities for a user profile.

    Implements the ``CapacityEstimator`` protocol from
    ``usability_oracle.channel.protocols``.
    """

    def __init__(
        self,
        baseline_overrides: Optional[Dict[WickensResource, float]] = None,
        fatigue_half_life_min: float = 45.0,
    ) -> None:
        self._baselines = dict(_BASELINE_CAPACITY)
        if baseline_overrides:
            self._baselines.update(baseline_overrides)
        self._fatigue_hl = fatigue_half_life_min

    # ---- single channel ------------------------------------------------

    def estimate_channel_capacity(
        self,
        resource: WickensResource,
        *,
        population_percentile: float = 50.0,
        age_years: Optional[float] = None,
        expertise_level: Optional[str] = None,
        time_on_task_min: float = 0.0,
    ) -> ResourceChannel:
        """Estimate capacity for one channel.

        Parameters
        ----------
        resource : WickensResource
            Target resource.
        population_percentile : float
            Percentile in [0, 100].
        age_years : float or None
            Age for decline adjustment.
        expertise_level : str or None
            Skill level.
        time_on_task_min : float
            Time on task for fatigue.

        Returns
        -------
        ResourceChannel
        """
        base = self._baselines[resource]
        sd = _CAPACITY_SD[resource]

        # Individual differences via percentile (inverse normal CDF).
        z = _ppf(population_percentile / 100.0)
        cap = base + z * sd

        # Adjustments.
        if age_years is not None:
            cap = age_adjustment(cap, age_years, resource)
        if expertise_level is not None:
            cap = expertise_adjustment(cap, expertise_level)
        if time_on_task_min > 0:
            cap = fatigue_degradation(cap, time_on_task_min, self._fatigue_hl)

        cap = max(cap, 0.5)

        # Confidence interval ± 1 SD.
        lo = max(cap - sd, 0.5)
        hi = cap + sd
        interval = Interval(low=lo, high=hi)

        return ResourceChannel(
            resource=resource,
            capacity_bits_per_s=cap,
            current_load=0.0,
            capacity_interval=interval,
            label=f"{resource.value} (p{population_percentile:.0f})",
        )

    # ---- full pool -----------------------------------------------------

    def estimate_pool(
        self,
        *,
        population_percentile: float = 50.0,
        age_years: Optional[float] = None,
        expertise_level: Optional[str] = None,
        time_on_task_min: float = 0.0,
    ) -> ResourcePool:
        """Estimate a complete resource pool.

        Parameters
        ----------
        population_percentile : float
            Population percentile.
        age_years : float or None
            User age.
        expertise_level : str or None
            Skill level.
        time_on_task_min : float
            Fatigue time.

        Returns
        -------
        ResourcePool
        """
        channels: List[ResourceChannel] = []
        for resource in WickensResource:
            ch = self.estimate_channel_capacity(
                resource,
                population_percentile=population_percentile,
                age_years=age_years,
                expertise_level=expertise_level,
                time_on_task_min=time_on_task_min,
            )
            channels.append(ch)

        label_parts: List[str] = []
        if expertise_level:
            label_parts.append(expertise_level)
        if age_years is not None:
            label_parts.append(f"age={age_years:.0f}")
        label_parts.append(f"p{population_percentile:.0f}")
        label = " ".join(label_parts)

        return ResourcePool(
            channels=tuple(channels),
            interference=None,
            population_percentile=population_percentile,
            label=label,
        )

    # ---- combined multi-channel capacity --------------------------------

    def combined_capacity(
        self,
        resources: Sequence[WickensResource],
        interference_matrix: Optional[InterferenceMatrix] = None,
        **kwargs,
    ) -> float:
        """Compute effective combined capacity across multiple channels.

        Without interference, combined capacity is the sum of individual
        capacities.  With interference, each channel's effective capacity
        is reduced by the weighted average interference from other active
        channels.

        Parameters
        ----------
        resources : Sequence[WickensResource]
            Active resources.
        interference_matrix : InterferenceMatrix or None
            Interference model.  If None, channels are independent.
        **kwargs
            Passed to ``estimate_channel_capacity``.

        Returns
        -------
        float
            Combined effective capacity in bits/s.
        """
        channels = [
            self.estimate_channel_capacity(r, **kwargs) for r in resources
        ]
        caps = np.array([ch.capacity_bits_per_s for ch in channels])

        if interference_matrix is None or len(resources) <= 1:
            return float(np.sum(caps))

        # Build interference coefficient array.
        n = len(resources)
        labels = [r.value for r in resources]
        intf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    li = labels[i]
                    lj = labels[j]
                    if (li in interference_matrix.resource_labels
                            and lj in interference_matrix.resource_labels):
                        ii = interference_matrix.resource_labels.index(li)
                        jj = interference_matrix.resource_labels.index(lj)
                        intf[i, j] = interference_matrix.coefficients[ii][jj]

        # Effective capacity = cap_i * (1 - mean interference from others).
        mean_intf = np.sum(intf, axis=1) / max(n - 1, 1)
        effective = caps * (1.0 - np.clip(mean_intf, 0.0, 0.95))
        return float(np.sum(effective))

    # ---- bottleneck identification --------------------------------------

    def identify_bottleneck(
        self,
        demands: Mapping[WickensResource, float],
        pool: Optional[ResourcePool] = None,
        **kwargs,
    ) -> Optional[WickensResource]:
        """Identify the channel closest to or over capacity.

        Parameters
        ----------
        demands : Mapping[WickensResource, float]
            Demand in bits/s per resource.
        pool : ResourcePool or None
            If None, estimate a default pool.

        Returns
        -------
        WickensResource or None
            The bottleneck resource, or None if no channel is loaded.
        """
        if pool is None:
            pool = self.estimate_pool(**kwargs)

        worst_ratio = -1.0
        worst_resource: Optional[WickensResource] = None

        for resource, demand in demands.items():
            ch = pool.channel_by_resource(resource)
            if ch is None:
                continue
            if ch.capacity_bits_per_s <= 0:
                return resource
            ratio = demand / ch.capacity_bits_per_s
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_resource = resource

        return worst_resource


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ppf(p: float) -> float:
    """Approximate inverse standard-normal CDF (probit).

    Uses Abramowitz & Stegun rational approximation (26.2.23).
    Accurate to ~4.5 × 10⁻⁴ for 0.0027 < p < 0.9973.
    """
    p = max(1e-6, min(1.0 - 1e-6, p))
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _rational_approx(t: float) -> float:
    """Rational approximation helper for _ppf."""
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


__all__ = [
    "ChannelCapacityEstimator",
    "auditory_capacity",
    "age_adjustment",
    "cognitive_capacity_wm",
    "expertise_adjustment",
    "fatigue_degradation",
    "motor_capacity_fitts",
    "visual_capacity",
]
