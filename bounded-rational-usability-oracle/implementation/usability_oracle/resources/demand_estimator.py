"""
usability_oracle.resources.demand_estimator — Demand estimation from UI elements.

Maps UI operations and elements to cognitive resource demand vectors
using Wickens' MRT dimensions.  Estimates perceptual, cognitive, motor,
and memory demands based on element properties and task context.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from usability_oracle.resources.types import (
    DemandVector,
    PerceptualModality,
    ProcessingCode,
    ProcessingStage,
    Resource,
    ResourceDemand,
    VisualChannel,
)


# ---------------------------------------------------------------------------
# AccessibilityRole → typical demand profiles
# ---------------------------------------------------------------------------

# Maps ARIA/accessibility roles to default demand estimates
# (stage, modality, code, channel, demand_level)
_ROLE_DEFAULTS: Dict[str, List[Dict[str, Any]]] = {
    "button": [
        {"stage": "perception", "modality": "visual", "code": "spatial",
         "channel": "focal", "demand": 0.3},
        {"stage": "cognition", "modality": None, "code": "verbal",
         "channel": None, "demand": 0.2},
        {"stage": "response", "modality": None, "code": "spatial",
         "channel": None, "demand": 0.3},
    ],
    "textbox": [
        {"stage": "perception", "modality": "visual", "code": "verbal",
         "channel": "focal", "demand": 0.4},
        {"stage": "cognition", "modality": None, "code": "verbal",
         "channel": None, "demand": 0.5},
        {"stage": "response", "modality": None, "code": "verbal",
         "channel": None, "demand": 0.4},
    ],
    "link": [
        {"stage": "perception", "modality": "visual", "code": "verbal",
         "channel": "focal", "demand": 0.3},
        {"stage": "cognition", "modality": None, "code": "verbal",
         "channel": None, "demand": 0.3},
        {"stage": "response", "modality": None, "code": "spatial",
         "channel": None, "demand": 0.2},
    ],
    "image": [
        {"stage": "perception", "modality": "visual", "code": "spatial",
         "channel": "focal", "demand": 0.4},
        {"stage": "cognition", "modality": None, "code": "spatial",
         "channel": None, "demand": 0.3},
    ],
    "slider": [
        {"stage": "perception", "modality": "visual", "code": "spatial",
         "channel": "focal", "demand": 0.5},
        {"stage": "cognition", "modality": None, "code": "spatial",
         "channel": None, "demand": 0.4},
        {"stage": "response", "modality": None, "code": "spatial",
         "channel": None, "demand": 0.5},
    ],
    "checkbox": [
        {"stage": "perception", "modality": "visual", "code": "spatial",
         "channel": "focal", "demand": 0.2},
        {"stage": "cognition", "modality": None, "code": "verbal",
         "channel": None, "demand": 0.3},
        {"stage": "response", "modality": None, "code": "spatial",
         "channel": None, "demand": 0.2},
    ],
    "dropdown": [
        {"stage": "perception", "modality": "visual", "code": "verbal",
         "channel": "focal", "demand": 0.4},
        {"stage": "cognition", "modality": None, "code": "verbal",
         "channel": None, "demand": 0.5},
        {"stage": "response", "modality": None, "code": "spatial",
         "channel": None, "demand": 0.3},
    ],
    "alert": [
        {"stage": "perception", "modality": "visual", "code": "verbal",
         "channel": "ambient", "demand": 0.5},
        {"stage": "cognition", "modality": None, "code": "verbal",
         "channel": None, "demand": 0.4},
    ],
    "audio": [
        {"stage": "perception", "modality": "auditory", "code": "verbal",
         "channel": None, "demand": 0.5},
        {"stage": "cognition", "modality": None, "code": "verbal",
         "channel": None, "demand": 0.3},
    ],
}

# Default for unknown roles
_DEFAULT_PROFILE: List[Dict[str, Any]] = [
    {"stage": "perception", "modality": "visual", "code": "spatial",
     "channel": "focal", "demand": 0.3},
    {"stage": "cognition", "modality": None, "code": "verbal",
     "channel": None, "demand": 0.3},
    {"stage": "response", "modality": None, "code": "spatial",
     "channel": None, "demand": 0.2},
]


class DemandEstimator:
    """Estimate cognitive resource demands from UI element properties.

    Implements the DemandEstimator protocol. Maps UI operations to
    demand vectors in Wickens' four-dimensional resource space.
    """

    # -------------------------------------------------------------------
    # Protocol: estimate
    # -------------------------------------------------------------------

    def estimate(
        self,
        operation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DemandVector:
        """Estimate the demand vector for a single UI operation.

        Parameters:
            operation: Dict with keys:
                - "type": str (e.g. "click", "read", "type", "scan")
                - "role": str (ARIA role, e.g. "button", "textbox")
                - "operation_id": str
                - "target_size": float (pixels, for motor demand)
                - "target_distance": float (pixels, for Fitts' law)
                - "text_length": int (characters, for reading)
                - "n_options": int (for decision complexity)
            context: Optional dict with:
                - "n_concurrent_tasks": int
                - "expertise_level": float in [0,1]
                - "n_items_in_memory": int

        Returns:
            DemandVector for the operation.
        """
        op_type = str(operation.get("type", "click"))
        role = str(operation.get("role", "button"))
        op_id = str(operation.get("operation_id", f"{op_type}_{role}"))
        ctx = context or {}

        demands: List[ResourceDemand] = []

        # Perceptual demand
        perc = self.estimate_perceptual_demand(operation)
        demands.extend(perc)

        # Cognitive demand
        cog = self.estimate_cognitive_demand(operation, ctx)
        demands.extend(cog)

        # Motor demand (if response stage is involved)
        if op_type in ("click", "type", "drag", "scroll", "swipe"):
            motor = self.estimate_motor_demand(operation)
            demands.extend(motor)

        # Memory demand from context
        n_items = ctx.get("n_items_in_memory", 0)
        if n_items and isinstance(n_items, (int, float)) and n_items > 0:
            mem = self.estimate_memory_demand(ctx, int(n_items))
            demands.extend(mem)

        # Expertise scaling
        expertise = float(ctx.get("expertise_level", 0.5))
        scaling = 1.0 - 0.3 * expertise  # experts need up to 30% less resource
        scaled = []
        for rd in demands:
            new_level = max(0.0, min(1.0, rd.demand_level * scaling))
            scaled.append(ResourceDemand(
                resource=rd.resource,
                demand_level=new_level,
                operation_id=op_id,
                description=rd.description,
            ))

        return DemandVector(demands=tuple(scaled), operation_id=op_id)

    # -------------------------------------------------------------------
    # Protocol: estimate_batch
    # -------------------------------------------------------------------

    def estimate_batch(
        self,
        operations: Sequence[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Sequence[DemandVector]:
        """Estimate demand vectors for multiple operations."""
        return [self.estimate(op, context) for op in operations]

    # -------------------------------------------------------------------
    # Perceptual demand
    # -------------------------------------------------------------------

    def estimate_perceptual_demand(
        self,
        element: Dict[str, Any],
    ) -> List[ResourceDemand]:
        """Estimate visual/auditory demand from element properties.

        Visual demand scales with:
            - Small target size (Fitts' law index)
            - Text length (reading time)
            - Visual clutter (ambient channel)

        Auditory demand for audio/alert elements.
        """
        role = str(element.get("role", "button"))
        profile = _ROLE_DEFAULTS.get(role, _DEFAULT_PROFILE)

        demands: List[ResourceDemand] = []
        for entry in profile:
            if entry["stage"] != "perception":
                continue
            mod = (
                PerceptualModality(entry["modality"])
                if entry["modality"] else None
            )
            code = ProcessingCode(entry.get("code", "spatial"))
            channel = (
                VisualChannel(entry["channel"])
                if entry.get("channel") else None
            )
            base_demand = float(entry["demand"])

            # Adjust for target size (smaller → harder to perceive)
            target_size = element.get("target_size")
            if target_size is not None and isinstance(target_size, (int, float)):
                size = max(1.0, float(target_size))
                # Penalty for small targets (below 44px recommended)
                if size < 44.0:
                    base_demand = min(1.0, base_demand + 0.2 * (1.0 - size / 44.0))

            # Adjust for text length (longer text → more reading demand)
            text_len = element.get("text_length")
            if text_len is not None and isinstance(text_len, (int, float)):
                chars = max(0, int(text_len))
                # Reading demand increases logarithmically
                if chars > 0:
                    base_demand = min(1.0, base_demand + 0.05 * math.log2(1 + chars / 10))

            resource = Resource(
                stage=ProcessingStage.PERCEPTION,
                modality=mod,
                visual_channel=channel,
                code=code,
                label=f"perception_{mod.value if mod else 'none'}_{code.value}",
            )
            demands.append(ResourceDemand(
                resource=resource,
                demand_level=min(1.0, max(0.0, base_demand)),
                description=f"Perceptual demand for {role}",
            ))
        return demands

    # -------------------------------------------------------------------
    # Cognitive demand
    # -------------------------------------------------------------------

    def estimate_cognitive_demand(
        self,
        element: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[ResourceDemand]:
        """Estimate cognitive demand from decision complexity.

        Uses Hick's law for choice complexity:
            T = a + b · log₂(n + 1)

        where n is the number of options/alternatives.
        """
        role = str(element.get("role", "button"))
        profile = _ROLE_DEFAULTS.get(role, _DEFAULT_PROFILE)

        demands: List[ResourceDemand] = []
        for entry in profile:
            if entry["stage"] != "cognition":
                continue
            code = ProcessingCode(entry.get("code", "verbal"))
            base_demand = float(entry["demand"])

            # Hick's law adjustment for number of options
            n_options = element.get("n_options")
            if n_options is not None and isinstance(n_options, (int, float)):
                n = max(1, int(n_options))
                # Hick's law: demand ∝ log₂(n + 1)
                hick_factor = math.log2(n + 1) / math.log2(3)  # normalised so 2 options → 1.0
                base_demand = min(1.0, base_demand * hick_factor)

            # Concurrent task penalty
            n_concurrent = int(context.get("n_concurrent_tasks", 1))
            if n_concurrent > 1:
                base_demand = min(1.0, base_demand + 0.1 * (n_concurrent - 1))

            resource = Resource(
                stage=ProcessingStage.COGNITION,
                code=code,
                label=f"cognition_{code.value}",
            )
            demands.append(ResourceDemand(
                resource=resource,
                demand_level=min(1.0, max(0.0, base_demand)),
                description=f"Cognitive demand for {role}",
            ))
        return demands

    # -------------------------------------------------------------------
    # Motor demand
    # -------------------------------------------------------------------

    def estimate_motor_demand(
        self,
        element: Dict[str, Any],
    ) -> List[ResourceDemand]:
        """Estimate motor demand from Fitts' ID and action type.

        Fitts' law index of difficulty:
            ID = log₂(2D / W)

        where D = distance to target, W = target width.
        Motor demand normalised to [0, 1] via ID / ID_max.
        """
        role = str(element.get("role", "button"))
        op_type = str(element.get("type", "click"))

        # Base motor demand from role profile
        profile = _ROLE_DEFAULTS.get(role, _DEFAULT_PROFILE)
        base_demand = 0.3
        for entry in profile:
            if entry["stage"] == "response":
                base_demand = float(entry["demand"])
                break

        # Fitts' law adjustment
        distance = element.get("target_distance")
        width = element.get("target_size")
        if (distance is not None and width is not None
                and isinstance(distance, (int, float))
                and isinstance(width, (int, float))):
            d = max(1.0, float(distance))
            w = max(1.0, float(width))
            fitts_id = math.log2(2 * d / w)
            # Normalise: typical ID range is 1-7 bits
            base_demand = min(1.0, max(0.0, fitts_id / 7.0))

        # Typing is more demanding than clicking
        if op_type == "type":
            base_demand = min(1.0, base_demand + 0.2)

        resource = Resource(
            stage=ProcessingStage.RESPONSE,
            code=ProcessingCode.SPATIAL,
            label="response_spatial",
        )
        return [ResourceDemand(
            resource=resource,
            demand_level=min(1.0, max(0.0, base_demand)),
            description=f"Motor demand for {op_type} on {role}",
        )]

    # -------------------------------------------------------------------
    # Memory demand
    # -------------------------------------------------------------------

    def estimate_memory_demand(
        self,
        context: Dict[str, Any],
        n_items: int,
    ) -> List[ResourceDemand]:
        """Estimate working-memory demand from context switching.

        Based on Miller's law (7 ± 2) and Cowan's limit (4 ± 1):
            WM demand ≈ n_items / capacity

        where capacity defaults to 4 (Cowan's limit).
        """
        capacity = 4.0  # Cowan's limit
        demand = min(1.0, n_items / capacity)

        resource = Resource(
            stage=ProcessingStage.COGNITION,
            code=ProcessingCode.VERBAL,
            label="cognition_verbal_wm",
        )
        return [ResourceDemand(
            resource=resource,
            demand_level=demand,
            description=f"Working memory demand for {n_items} items",
        )]

    # -------------------------------------------------------------------
    # Aggregate demands
    # -------------------------------------------------------------------

    def aggregate_demands(
        self,
        element_demands: Sequence[DemandVector],
    ) -> DemandVector:
        """Compose demands across task steps.

        Aggregates by taking the maximum demand per unique resource
        (assumes sequential steps with potential overlap).

        Parameters:
            element_demands: Demand vectors from individual elements.

        Returns:
            Aggregated DemandVector.
        """
        if not element_demands:
            return DemandVector(demands=(), operation_id="aggregate")

        # Group demands by resource dimension tuple, take max
        best: Dict[tuple, ResourceDemand] = {}
        for dv in element_demands:
            for rd in dv.demands:
                key = rd.resource.dimension_tuple
                existing = best.get(key)
                if existing is None or rd.demand_level > existing.demand_level:
                    best[key] = rd

        agg_demands = tuple(best.values())
        return DemandVector(
            demands=agg_demands,
            operation_id="aggregate",
        )
