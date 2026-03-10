"""
usability_oracle.bottleneck.interference — Cross-channel interference detector.

Detects states where concurrent cognitive demands create interference
across perceptual/motor channels, based on Wickens' Multiple Resource
Theory (MRT).

MRT posits that human information processing draws from multiple
independent resource pools, characterised by:
  - **Stage**: perception vs. cognition vs. response
  - **Code**: spatial vs. verbal
  - **Modality**: visual vs. auditory (input), manual vs. vocal (output)

When two concurrent tasks demand the same resource pool, interference
occurs and performance degrades.  The interference level is:

    I = Σ_{channels} max(0, demand(channel) - capacity(channel))

The resource conflict between two actions using the same channel is:

    conflict(c) = demand_1(c) · demand_2(c)  if both > 0, else 0

References
----------
- Wickens, C. D. (2002). Multiple resources and performance prediction.
  *Theoretical Issues in Ergonomics Science* 3(2), 159–177.
- Wickens, C. D. (2008). Multiple resources and mental workload. *HF* 50(3).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import MDP


# ---------------------------------------------------------------------------
# Channel capacity model (bits/second)
# ---------------------------------------------------------------------------

DEFAULT_CHANNEL_CAPACITIES: dict[str, float] = {
    "visual": 40.0,
    "auditory": 30.0,
    "haptic": 5.0,
    "motor_hand": 10.0,
    "motor_finger": 8.0,
    "motor_eye": 15.0,
    "motor_voice": 12.0,
    "cognitive_spatial": 7.0,
    "cognitive_verbal": 7.0,
}

# Interference matrix: pairs of channels that interfere strongly
INTERFERENCE_PAIRS: set[frozenset[str]] = {
    frozenset({"visual", "motor_eye"}),           # eye movement conflicts
    frozenset({"cognitive_spatial", "motor_hand"}),  # spatial processing + manual response
    frozenset({"cognitive_verbal", "motor_voice"}),  # verbal + vocal
    frozenset({"visual", "cognitive_spatial"}),      # visual + spatial cognition
    frozenset({"auditory", "cognitive_verbal"}),     # auditory + verbal cognition
}


# ---------------------------------------------------------------------------
# CrossChannelInterferenceDetector
# ---------------------------------------------------------------------------

@dataclass
class CrossChannelInterferenceDetector:
    """Detect cross-channel interference at UI states.

    Based on Wickens' Multiple Resource Theory, flags states where
    concurrent action demands create resource conflicts across
    cognitive/motor channels.

    Parameters
    ----------
    INTERFERENCE_THRESHOLD : float
        Normalised interference level above which a bottleneck is flagged.
        Default 0.5.
    TEMPORAL_OVERLAP_THRESHOLD : float
        Fraction of temporal overlap required for interference.
        Default 0.3.
    channel_capacities : dict[str, float]
        Channel capacity model (bits/second per channel).
    """

    INTERFERENCE_THRESHOLD: float = 0.5
    TEMPORAL_OVERLAP_THRESHOLD: float = 0.3
    channel_capacities: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CHANNEL_CAPACITIES)
    )

    # ── Public API --------------------------------------------------------

    def detect(
        self,
        state: str,
        actions: list[str],
        mdp: MDP,
    ) -> Optional[BottleneckResult]:
        """Detect cross-channel interference at *state*.

        Parameters
        ----------
        state : str
            State identifier.
        actions : list[str]
            Available action ids at this state.
        mdp : MDP

        Returns
        -------
        BottleneckResult or None
        """
        if len(actions) < 2:
            return None

        # Gather channel demands for each action
        action_demands: list[dict[str, float]] = []
        for aid in actions:
            demands = self._action_channel_demands(aid, mdp)
            action_demands.append(demands)

        # Compute aggregate channel demands
        aggregate_demands = self._channel_demands(action_demands)

        # Compute interference level
        interference = self._interference_level(aggregate_demands)

        # Compute resource conflicts
        conflict = self._resource_conflict_total(action_demands)

        # Compute temporal overlap (heuristic: concurrent UI elements)
        temporal_overlap = self._temporal_overlap(action_demands)

        # Build evidence
        evidence: dict[str, float] = {
            "interference_level": interference,
            "resource_conflict": conflict,
            "temporal_overlap": temporal_overlap,
            "n_channels_used": float(len(aggregate_demands)),
            "n_actions": float(len(actions)),
        }
        for ch, demand in aggregate_demands.items():
            evidence[f"demand_{ch}"] = demand

        # Detection logic
        is_interfering = (
            interference > self.INTERFERENCE_THRESHOLD
            or conflict > self.INTERFERENCE_THRESHOLD
        )

        if not is_interfering:
            return None

        # Confidence
        max_metric = max(interference, conflict)
        confidence = min(1.0, max_metric / (self.INTERFERENCE_THRESHOLD * 2.0) + 0.3)

        # Find the most-conflicting channels
        overloaded_channels = [
            ch for ch, d in aggregate_demands.items()
            if d > self.channel_capacities.get(ch, float("inf")) * 0.7
        ]

        severity = self._severity_from_interference(interference, conflict)

        return BottleneckResult(
            bottleneck_type=BottleneckType.CROSS_CHANNEL_INTERFERENCE,
            severity=severity,
            confidence=confidence,
            affected_states=[state],
            affected_actions=actions,
            cognitive_law=CognitiveLaw.PERCEPTUAL_PROCESSING,
            channel=overloaded_channels[0] if overloaded_channels else "cognitive_spatial",
            evidence=evidence,
            description=(
                f"Cross-channel interference at state {state!r}: "
                f"interference={interference:.2f}, conflict={conflict:.2f}, "
                f"overloaded channels: {overloaded_channels}"
            ),
            recommendation=(
                "Reduce concurrent demands on the same cognitive channel by "
                "serialising interactions or redistributing across modalities."
            ),
            repair_hints=[
                "Separate visual and spatial tasks temporally",
                "Use auditory feedback to offload visual channel",
                "Serialise concurrent interaction requirements",
                "Reduce the number of simultaneous input modes required",
                "Provide single-modality alternatives for complex tasks",
            ],
        )

    # ── Channel demand computation ----------------------------------------

    def _action_channel_demands(
        self,
        action_id: str,
        mdp: MDP,
    ) -> dict[str, float]:
        """Compute per-channel demand for a single action.

        Uses the action's ``channels`` attribute if available, otherwise
        infers from ``action_type``.

        Parameters
        ----------
        action_id : str
        mdp : MDP

        Returns
        -------
        dict[str, float]
            Maps channel name → demand (bits/second).
        """
        action = mdp.actions.get(action_id)
        demands: dict[str, float] = {}

        if action is None:
            return {"cognitive_spatial": 3.0}

        # Use explicit channel annotations if present
        channels = getattr(action, "channels", None) or []
        if channels:
            for ch in channels:
                demands[ch] = demands.get(ch, 0.0) + 5.0
            return demands

        # Infer from action type
        action_type = getattr(action, "action_type", "click")

        channel_map: dict[str, dict[str, float]] = {
            "click": {"visual": 5.0, "motor_hand": 5.0, "cognitive_spatial": 3.0},
            "type": {"visual": 3.0, "motor_finger": 7.0, "cognitive_verbal": 5.0},
            "read": {"visual": 8.0, "cognitive_verbal": 6.0},
            "scroll": {"visual": 4.0, "motor_hand": 3.0},
            "navigate": {"visual": 5.0, "cognitive_spatial": 5.0},
            "select": {"visual": 5.0, "motor_hand": 4.0, "cognitive_spatial": 3.0},
            "tab": {"motor_finger": 2.0, "cognitive_spatial": 2.0},
            "back": {"motor_hand": 2.0, "cognitive_spatial": 3.0},
        }

        return channel_map.get(action_type, {"cognitive_spatial": 3.0})

    def _channel_demands(
        self,
        action_demands: list[dict[str, float]],
    ) -> dict[str, float]:
        """Compute aggregate per-channel demand across all actions.

        Parameters
        ----------
        action_demands : list[dict[str, float]]

        Returns
        -------
        dict[str, float]
            Total demand per channel.
        """
        aggregate: dict[str, float] = defaultdict(float)
        for demands in action_demands:
            for ch, demand in demands.items():
                aggregate[ch] += demand
        return dict(aggregate)

    def _interference_level(self, demands: dict[str, float]) -> float:
        """Compute the normalised interference level based on MRT.

        I = Σ_{channels} max(0, demand(c) - capacity(c)) / capacity(c)

        Normalised by the number of active channels.

        Parameters
        ----------
        demands : dict[str, float]

        Returns
        -------
        float
            Interference ∈ [0, ∞); 0 = no interference.
        """
        if not demands:
            return 0.0

        total_interference = 0.0
        n_active = 0

        for channel, demand in demands.items():
            capacity = self.channel_capacities.get(channel, 10.0)
            if demand > 0:
                n_active += 1
                excess = max(0.0, demand - capacity)
                total_interference += excess / capacity

        if n_active == 0:
            return 0.0
        return total_interference / n_active

    def _resource_conflict(self, channels: list[str]) -> float:
        """Compute the resource conflict score for a set of channels.

        Based on MRT interference matrix: known-conflicting channel pairs
        contribute 1.0 each; others contribute 0.5 if they share a resource
        dimension (stage, code, modality).

        Parameters
        ----------
        channels : list[str]
            Active channel names.

        Returns
        -------
        float
            Conflict score ∈ [0, 1].
        """
        if len(channels) < 2:
            return 0.0

        n_pairs = 0
        total_conflict = 0.0

        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                pair = frozenset({channels[i], channels[j]})
                n_pairs += 1
                if pair in INTERFERENCE_PAIRS:
                    total_conflict += 1.0
                elif self._share_resource_dimension(channels[i], channels[j]):
                    total_conflict += 0.5

        if n_pairs == 0:
            return 0.0
        return min(1.0, total_conflict / n_pairs)

    def _resource_conflict_total(
        self,
        action_demands: list[dict[str, float]],
    ) -> float:
        """Compute total resource conflict across all actions.

        conflict = Σ_{i<j} Σ_c demand_i(c) · demand_j(c) / capacity(c)²

        Normalised by number of action pairs.

        Parameters
        ----------
        action_demands : list[dict[str, float]]

        Returns
        -------
        float
        """
        n = len(action_demands)
        if n < 2:
            return 0.0

        total = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                n_pairs += 1
                for ch in set(action_demands[i].keys()) & set(action_demands[j].keys()):
                    cap = self.channel_capacities.get(ch, 10.0)
                    total += (action_demands[i][ch] * action_demands[j][ch]) / (cap ** 2)

        if n_pairs == 0:
            return 0.0
        return total / n_pairs

    def _temporal_overlap(self, action_demands: list[dict[str, float]]) -> float:
        """Estimate temporal overlap between actions.

        Heuristic: the more actions share the same channels, the higher
        the temporal overlap (they compete for the same resources).

        Parameters
        ----------
        action_demands : list[dict[str, float]]

        Returns
        -------
        float
            Overlap ∈ [0, 1].
        """
        if len(action_demands) < 2:
            return 0.0

        all_channels: set[str] = set()
        for d in action_demands:
            all_channels.update(d.keys())

        if not all_channels:
            return 0.0

        shared_count = 0
        total_count = 0
        for ch in all_channels:
            using = sum(1 for d in action_demands if ch in d)
            total_count += 1
            if using > 1:
                shared_count += 1

        return shared_count / total_count

    def _share_resource_dimension(self, ch1: str, ch2: str) -> bool:
        """Check if two channels share a resource dimension (MRT)."""
        # Simple prefix-based heuristic
        prefixes = ["motor_", "cognitive_", "visual", "auditory"]
        for prefix in prefixes:
            if ch1.startswith(prefix) and ch2.startswith(prefix):
                return True
        # Eye-related channels share a visual/oculomotor resource
        eye_channels = {"visual", "motor_eye"}
        if ch1 in eye_channels and ch2 in eye_channels:
            return True
        return False

    def _severity_from_interference(
        self,
        interference: float,
        conflict: float,
    ) -> Severity:
        """Map interference metrics to severity."""
        score = interference + conflict
        if score > 2.0:
            return Severity.CRITICAL
        elif score > 1.2:
            return Severity.HIGH
        elif score > 0.7:
            return Severity.MEDIUM
        elif score > 0.3:
            return Severity.LOW
        return Severity.INFO
