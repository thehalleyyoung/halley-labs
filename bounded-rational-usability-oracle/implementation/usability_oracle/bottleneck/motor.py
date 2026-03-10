"""
usability_oracle.bottleneck.motor — Motor difficulty detector.

Detects actions that are difficult to execute due to small target size,
large distance, or high precision requirements, using Fitts' law:

    T_movement = a + b · ID

where the Index of Difficulty is:

    ID = log₂(2D / W)

with D = distance to target centre and W = target width along the movement
axis (the "effective target width").

For 2-D targets, the bivariate Fitts' model uses:

    ID = log₂(√((D/W)² + (D/H)²) + 1)

where W and H are target width and height.

Targets are flagged when:
  - ID exceeds a difficulty threshold (default 4.5 bits ≈ very difficult)
  - Target is "too small" (< 24px in either dimension)
  - Target is "too distant" (D > 500px from expected cursor position)

The precision requirement is inversely proportional to target area.

References
----------
- Fitts, P. M. (1954). The information capacity of the human motor system
  in controlling the amplitude of movement. *JEPG* 47(6), 381–391.
- MacKenzie, I. S. (1992). Fitts' law as a research and design tool in
  human-computer interaction. *HCI* 7, 91–139.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import MDP


# ---------------------------------------------------------------------------
# MotorDifficultyDetector
# ---------------------------------------------------------------------------

@dataclass
class MotorDifficultyDetector:
    """Detect motor-difficulty bottlenecks for UI actions.

    Flags actions where the Fitts' law Index of Difficulty exceeds a
    threshold, or where target sizing / positioning violates ergonomic
    guidelines.

    Parameters
    ----------
    DIFFICULTY_THRESHOLD : float
        Index of Difficulty (bits) above which an action is flagged.
        Default 4.5 bits corresponds to a very difficult acquisition.
    FITTS_A : float
        Fitts' law intercept (seconds).
    FITTS_B : float
        Fitts' law slope (seconds per bit).
    MIN_TARGET_PX : float
        Minimum recommended target size (pixels).
    MAX_DISTANCE_PX : float
        Maximum reasonable cursor travel distance (pixels).
    """

    DIFFICULTY_THRESHOLD: float = 4.5
    FITTS_A: float = 0.05
    FITTS_B: float = 0.15
    MIN_TARGET_PX: float = 24.0
    MAX_DISTANCE_PX: float = 500.0

    # ── Public API --------------------------------------------------------

    def detect(
        self,
        state: str,
        action: str,
        mdp: MDP,
        features: dict[str, float],
    ) -> Optional[BottleneckResult]:
        """Detect motor difficulty for action *action* at *state*.

        Parameters
        ----------
        state : str
        action : str
        mdp : MDP
        features : dict[str, float]
            State features, expected to include target geometry:
            - ``target_width``, ``target_height``: target dimensions (px).
            - ``target_distance``: distance to target centre (px).
            - ``cursor_x``, ``cursor_y``: current cursor position (optional).

        Returns
        -------
        BottleneckResult or None
        """
        # Extract target geometry from features or action metadata
        action_obj = mdp.actions.get(action)
        target_width = features.get("target_width", 44.0)
        target_height = features.get("target_height", 44.0)
        target_distance = features.get("target_distance", 100.0)

        # Override from action features if available
        if action_obj is not None:
            # Action objects may store target geometry in features
            action_meta = getattr(action_obj, "metadata", {}) or {}
            target_width = action_meta.get("target_width", target_width)
            target_height = action_meta.get("target_height", target_height)
            target_distance = action_meta.get("target_distance", target_distance)

        # Compute Fitts' law metrics
        fitts_id = self._index_of_difficulty(target_distance, target_width)
        movement_time = self._fitts_difficulty(target_distance, target_width)
        small_target = self._is_small_target(target_width, target_height)
        distant_target = self._is_distant_target(target_distance)
        precision = self._precision_requirement(target_width, target_height)

        # Build evidence
        evidence: dict[str, float] = {
            "fitts_id": fitts_id,
            "movement_time_seconds": movement_time,
            "target_width": target_width,
            "target_height": target_height,
            "target_distance": target_distance,
            "precision_requirement": precision,
            "is_small_target": float(small_target),
            "is_distant_target": float(distant_target),
        }

        # Detection logic
        is_difficult = False
        confidence = 0.0

        if fitts_id > self.DIFFICULTY_THRESHOLD:
            is_difficult = True
            excess = fitts_id - self.DIFFICULTY_THRESHOLD
            confidence = max(confidence, min(1.0, 0.5 + excess * 0.15))

        if small_target:
            is_difficult = True
            size_ratio = min(target_width, target_height) / self.MIN_TARGET_PX
            confidence = max(confidence, min(1.0, 1.0 - size_ratio + 0.3))

        if distant_target:
            is_difficult = True
            dist_ratio = target_distance / self.MAX_DISTANCE_PX
            confidence = max(confidence, min(1.0, dist_ratio * 0.6))

        if not is_difficult:
            return None

        severity = self._severity_from_id(fitts_id, small_target, distant_target)

        return BottleneckResult(
            bottleneck_type=BottleneckType.MOTOR_DIFFICULTY,
            severity=severity,
            confidence=confidence,
            affected_states=[state],
            affected_actions=[action],
            cognitive_law=CognitiveLaw.FITTS,
            channel="motor_hand",
            evidence=evidence,
            description=(
                f"Motor difficulty for action {action!r} at state {state!r}: "
                f"ID={fitts_id:.1f} bits, T={movement_time:.2f}s"
                f"{', small target' if small_target else ''}"
                f"{', distant target' if distant_target else ''}"
            ),
            recommendation=(
                "Increase target size, reduce distance to target, "
                "or add keyboard shortcuts as alternatives."
            ),
            repair_hints=[
                "Increase target size to at least 44×44 pixels",
                "Reduce distance between sequential interaction targets",
                "Add keyboard shortcuts for frequent actions",
                "Place primary actions in easy-to-reach screen regions",
                "Use snap-to or magnetic targets for precision tasks",
            ],
        )

    # ── Fitts' law computations -------------------------------------------

    def _fitts_difficulty(self, distance: float, width: float) -> float:
        """Compute Fitts' law movement time.

        T = a + b · log₂(2D / W)

        Parameters
        ----------
        distance : float
            Distance to target centre (pixels).
        width : float
            Effective target width (pixels).

        Returns
        -------
        float
            Movement time in seconds.
        """
        idx = self._index_of_difficulty(distance, width)
        return self.FITTS_A + self.FITTS_B * idx

    def _index_of_difficulty(self, distance: float, width: float) -> float:
        """Compute the Fitts' Index of Difficulty (Shannon formulation).

        ID = log₂(D / W + 1)

        This is the Shannon formulation which avoids negative ID values.

        Parameters
        ----------
        distance : float
            Distance to target (pixels).
        width : float
            Target width along movement axis (pixels).

        Returns
        -------
        float
            Index of difficulty in bits.
        """
        if width <= 0:
            return float("inf")
        # Shannon formulation: ID = log2(D/W + 1)
        return math.log2(distance / width + 1.0)

    def _is_small_target(self, width: float, height: float) -> bool:
        """Check if the target is below minimum size guidelines.

        Parameters
        ----------
        width, height : float
            Target dimensions in pixels.

        Returns
        -------
        bool
        """
        return width < self.MIN_TARGET_PX or height < self.MIN_TARGET_PX

    def _is_distant_target(self, distance: float) -> bool:
        """Check if the target exceeds maximum reasonable distance.

        Parameters
        ----------
        distance : float
            Distance in pixels.

        Returns
        -------
        bool
        """
        return distance > self.MAX_DISTANCE_PX

    def _precision_requirement(self, target_width: float, target_height: float) -> float:
        """Compute the precision requirement for a target.

        precision = 1 / (W · H)

        normalised so that a 44×44 px target has precision 1.0.

        Parameters
        ----------
        target_width, target_height : float
            Target dimensions in pixels.

        Returns
        -------
        float
            Precision requirement (higher = more precise pointing needed).
        """
        reference_area = 44.0 * 44.0  # WCAG recommended size
        target_area = max(target_width * target_height, 1.0)
        return reference_area / target_area

    # ── Helpers -----------------------------------------------------------

    def _severity_from_id(
        self,
        fitts_id: float,
        small_target: bool,
        distant_target: bool,
    ) -> Severity:
        """Map Fitts' ID and target characteristics to severity."""
        score = fitts_id / self.DIFFICULTY_THRESHOLD
        if small_target:
            score += 0.5
        if distant_target:
            score += 0.3

        if score > 2.5:
            return Severity.CRITICAL
        elif score > 1.8:
            return Severity.HIGH
        elif score > 1.2:
            return Severity.MEDIUM
        elif score > 0.8:
            return Severity.LOW
        return Severity.INFO
