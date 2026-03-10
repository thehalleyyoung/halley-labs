"""Cognitive modeling primitives for the usability oracle.

This module defines the core data structures used throughout the cognitive
modeling subsystem, including enumerations for cognitive laws and channels,
dataclasses for operations, cost elements, geometric primitives, and
perceptual scenes.

References
----------
Card, S. K., Moran, T. P., & Newell, A. (1983).
    *The Psychology of Human-Computer Interaction*. Lawrence Erlbaum.
Wickens, C. D. (2008). Multiple resources and mental workload.
    *Human Factors*, 50(3), 449-455.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CognitiveLaw(Enum):
    """Enumeration of well-known cognitive/motor laws used in prediction.

    Each member maps to a specific empirical regularity that the oracle
    can invoke when estimating interaction time or error rate.
    """

    FITTS = "fitts"
    HICK_HYMAN = "hick_hyman"
    VISUAL_SEARCH = "visual_search"
    WORKING_MEMORY = "working_memory"
    KLM_KEYSTROKE = "klm_keystroke"
    KLM_POINTING = "klm_pointing"
    READING = "reading"
    SACCADE = "saccade"


class MotorChannel(Enum):
    """Motor output channels through which a user can act.

    Based on the Multiple Resource Theory (Wickens, 2008).
    """

    HAND = "hand"
    EYE = "eye"
    VOICE = "voice"
    FOOT = "foot"


class PerceptualChannel(Enum):
    """Perceptual input channels through which a user receives information.

    Based on the Multiple Resource Theory (Wickens, 2008).
    """

    VISUAL = "visual"
    AUDITORY = "auditory"
    HAPTIC = "haptic"


# ---------------------------------------------------------------------------
# Dataclasses — operations and cost
# ---------------------------------------------------------------------------


@dataclass
class CognitiveOperation:
    """A single cognitive or motor operation in a task decomposition.

    Attributes
    ----------
    law : CognitiveLaw
        The empirical law governing this operation's time prediction.
    channel : MotorChannel | PerceptualChannel
        The resource channel occupied by this operation.
    parameters : dict
        Law-specific parameters (e.g. ``{"distance": 200, "width": 40}``
        for Fitts' law).
    description : str
        Human-readable description of the operation.
    """

    law: CognitiveLaw
    channel: Union[MotorChannel, PerceptualChannel]
    parameters: dict
    description: str

    def uses_motor(self) -> bool:
        """Return True if this operation occupies a motor channel."""
        return isinstance(self.channel, MotorChannel)

    def uses_perceptual(self) -> bool:
        """Return True if this operation occupies a perceptual channel."""
        return isinstance(self.channel, PerceptualChannel)


@dataclass
class CostElement:
    """Predicted time cost for a single interaction element.

    Stores both the mean predicted time and its variance so that
    downstream analyses can propagate uncertainty.

    Attributes
    ----------
    mean_time : float
        Expected completion time in seconds.
    variance : float
        Variance of the completion time estimate (seconds²).
    channel : str
        Name of the channel this cost is charged to.
    law : str
        Name of the cognitive law used to derive this cost.
    """

    mean_time: float
    variance: float
    channel: str
    law: str

    def total_cost(self) -> float:
        """Return the mean time as the scalar cost metric.

        For a single element the total cost is simply the mean predicted
        time.  When aggregating multiple elements, callers should sum
        ``total_cost()`` across elements on the critical path.

        Returns
        -------
        float
            Mean predicted time in seconds.
        """
        return self.mean_time

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Return a symmetric confidence interval around the mean.

        Parameters
        ----------
        z : float, optional
            Number of standard deviations for the interval (default 1.96
            corresponds to ≈95 % coverage under normality).

        Returns
        -------
        tuple[float, float]
            ``(lower_bound, upper_bound)`` in seconds.

        Raises
        ------
        ValueError
            If *variance* is negative.
        """
        if self.variance < 0:
            raise ValueError("Variance must be non-negative.")
        sd = math.sqrt(self.variance)
        return (self.mean_time - z * sd, self.mean_time + z * sd)


@dataclass
class CognitiveContext:
    """Contextual state that modulates cognitive performance predictions.

    Captures the history of preceding operations, current working-memory
    load, elapsed session time, and an explicit fatigue factor so that
    prediction models can account for temporal context effects.

    Attributes
    ----------
    prior_operations : list[CognitiveOperation]
        Operations already performed in the current task.
    working_memory_load : int
        Number of chunks currently held in working memory (0–7+).
    elapsed_time : float
        Wall-clock seconds since the task began.
    fatigue_factor : float
        Normalised fatigue level in [0, 1] where 0 means fresh and 1
        means maximally fatigued.
    """

    prior_operations: List[CognitiveOperation] = field(default_factory=list)
    working_memory_load: int = 0
    elapsed_time: float = 0.0
    fatigue_factor: float = 0.0

    def fatigue_multiplier(self) -> float:
        """Compute a multiplicative slowdown factor due to fatigue.

        The mapping uses a logistic-style curve so that low fatigue has
        negligible impact while high fatigue can roughly double reaction
        times:

        .. math::

            m = 1 + \\frac{f^2}{1 - 0.5\\,f}

        where *f* is ``fatigue_factor`` clamped to [0, 0.99].

        Returns
        -------
        float
            Multiplier ≥ 1.0 to apply to base time predictions.
        """
        f = max(0.0, min(self.fatigue_factor, 0.99))
        return 1.0 + (f * f) / (1.0 - 0.5 * f)

    def effective_wm_capacity(self, base_capacity: int = 7) -> int:
        """Return effective working-memory capacity given fatigue.

        Fatigue reduces the usable number of WM slots.  The result is
        clamped to at least 1.

        Parameters
        ----------
        base_capacity : int, optional
            Miller's number or a calibrated capacity (default 7).

        Returns
        -------
        int
            Effective capacity (≥ 1).
        """
        reduction = int(self.fatigue_factor * 3)
        return max(1, base_capacity - reduction)


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------


@dataclass
class Point2D:
    """A point in 2-D screen coordinates (pixels).

    Attributes
    ----------
    x : float
        Horizontal coordinate.
    y : float
        Vertical coordinate.
    """

    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        """Euclidean distance to *other*.

        Parameters
        ----------
        other : Point2D
            Target point.

        Returns
        -------
        float
            Distance in pixels.
        """
        return math.hypot(self.x - other.x, self.y - other.y)

    def angle_to(self, other: Point2D) -> float:
        """Angle from this point to *other* in radians, measured from
        the positive x-axis counter-clockwise.

        Parameters
        ----------
        other : Point2D
            Target point.

        Returns
        -------
        float
            Angle in radians in (-π, π].
        """
        return math.atan2(other.y - self.y, other.x - self.x)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for a UI element.

    Attributes
    ----------
    x : float
        Left edge in pixels.
    y : float
        Top edge in pixels.
    width : float
        Width in pixels (> 0).
    height : float
        Height in pixels (> 0).
    """

    x: float
    y: float
    width: float
    height: float

    def center(self) -> Point2D:
        """Return the centre point of the box.

        Returns
        -------
        Point2D
        """
        return Point2D(self.x + self.width / 2.0, self.y + self.height / 2.0)

    def area(self) -> float:
        """Return the area in square pixels.

        Returns
        -------
        float
        """
        return self.width * self.height

    def distance_to(self, other: BoundingBox) -> float:
        """Centre-to-centre Euclidean distance to *other* bounding box.

        This is the standard distance metric used by Fitts' law
        implementations when predicting pointing time between two UI
        elements.

        Parameters
        ----------
        other : BoundingBox
            The target bounding box.

        Returns
        -------
        float
            Distance in pixels.
        """
        c1 = self.center()
        c2 = other.center()
        return c1.distance_to(c2)

    def min_extent(self) -> float:
        """Return the smaller of width and height.

        Useful as the *W* parameter in Fitts' law when the approach
        angle is unknown and the constraint dimension is the minimum.

        Returns
        -------
        float
        """
        return min(self.width, self.height)


# ---------------------------------------------------------------------------
# Composite action / scene structures
# ---------------------------------------------------------------------------


@dataclass
class MotorAction:
    """A directed motor action between two screen regions.

    Attributes
    ----------
    source_bbox : BoundingBox
        Starting region (e.g. current cursor location).
    target_bbox : BoundingBox
        Target region the user must acquire.
    action_type : str
        Semantic label for the action (e.g. ``"click"``, ``"drag"``).
    """

    source_bbox: BoundingBox
    target_bbox: BoundingBox
    action_type: str

    def movement_distance(self) -> float:
        """Centre-to-centre distance of the action in pixels."""
        return self.source_bbox.distance_to(self.target_bbox)

    def target_width(self) -> float:
        """Effective target width (minimum extent of target box)."""
        return self.target_bbox.min_extent()


@dataclass
class PerceptualScene:
    """A snapshot of the visual scene presented to the user.

    Holds parallel lists describing each visible element's bounding box,
    text label, and bottom-up saliency score.

    Attributes
    ----------
    elements : list[BoundingBox]
        Bounding boxes for every visible UI element.
    labels : list[str]
        Corresponding text labels (may be empty strings).
    saliency : list[float]
        Normalised saliency scores in [0, 1] for each element.
    """

    elements: List[BoundingBox] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    saliency: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.elements)
        if len(self.labels) != n or len(self.saliency) != n:
            raise ValueError(
                "elements, labels, and saliency must all have the same length "
                f"(got {n}, {len(self.labels)}, {len(self.saliency)})."
            )

    @property
    def size(self) -> int:
        """Number of elements in the scene."""
        return len(self.elements)

    def mean_saliency(self) -> float:
        """Return the arithmetic mean of saliency scores.

        Returns
        -------
        float
            Mean saliency, or 0.0 if the scene is empty.
        """
        if not self.saliency:
            return 0.0
        return float(np.mean(self.saliency))

    def most_salient_index(self) -> Optional[int]:
        """Return the index of the element with the highest saliency.

        Returns
        -------
        int or None
            Index into ``elements``, or ``None`` if the scene is empty.
        """
        if not self.saliency:
            return None
        return int(np.argmax(self.saliency))
