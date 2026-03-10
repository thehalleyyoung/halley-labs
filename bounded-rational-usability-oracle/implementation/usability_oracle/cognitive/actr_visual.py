"""ACT-R visual module for usability modelling.

Implements the visual attention component of ACT-R (Anderson, 2007),
incorporating the EMMA model of eye movements (Salvucci, 2001) and
feature-based guided visual search (Wolfe, 2007).  Provides encoding
time prediction, saccade planning, icon persistence, and integration
with accessibility tree spatial layouts.

References
----------
Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical
    Universe?* Oxford University Press.
Salvucci, D. D. (2001). An integrated model of eye movements and visual
    encoding. *Cognitive Systems Research*, 1(4), 201-220.
Wolfe, J. M. (2007). Guided Search 4.0. In W. Gray (Ed.), *Integrated
    Models of Cognitive Systems* (pp. 99-119). Oxford University Press.
Rayner, K. (1998). Eye movements in reading and information processing:
    20 years of research. *Psychological Bulletin*, 124(3), 372-422.
Irwin, D. E. (1991). Information integration across saccadic eye
    movements. *Cognitive Psychology*, 23(3), 420-456.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.cognitive.models import BoundingBox, Point2D


# ---------------------------------------------------------------------------
# Visual object representation
# ---------------------------------------------------------------------------


@dataclass
class VisualObject:
    """A visual object in the attention module's visicon.

    Attributes
    ----------
    name : str
        Unique identifier.
    kind : str
        Object type (e.g. ``"button"``, ``"text"``, ``"icon"``).
    bbox : BoundingBox
        Screen bounding box.
    features : dict[str, Any]
        Feature map (e.g. ``{"color": "red", "shape": "circle"}``).
    frequency : float
        Feature familiarity/frequency in [0, 1].  Higher means faster
        encoding (Salvucci, 2001).
    onset_time : float
        Time the object appeared in the display (seconds).
    """

    name: str
    kind: str
    bbox: BoundingBox
    features: Dict[str, Any] = field(default_factory=dict)
    frequency: float = 0.5
    onset_time: float = 0.0

    @property
    def center(self) -> Point2D:
        return self.bbox.center()


# ---------------------------------------------------------------------------
# EMMA model parameters
# ---------------------------------------------------------------------------


@dataclass
class EMMAParams:
    """Parameters for the EMMA eye movement model (Salvucci, 2001).

    Attributes
    ----------
    encoding_factor : float
        Scaling constant *K* for encoding time (default 0.006).
    encoding_exponent : float
        Frequency exponent *f* in encoding equation (default 0.5).
    saccade_preparation : float
        Time to programme a saccade (seconds, default 0.135).
    execution_base : float
        Base saccade execution time (seconds, default 0.020).
    execution_rate : float
        Per-degree execution time (seconds/degree, default 0.002).
    pixels_per_degree : float
        Conversion factor (default 38.0 at 60 cm viewing distance).
    icon_decay_rate : float
        Exponential decay rate for iconic memory (1/s, default 4.0).
        Iconic trace half-life ≈ 170 ms (Irwin, 1991).
    """

    encoding_factor: float = 0.006
    encoding_exponent: float = 0.5
    saccade_preparation: float = 0.135
    execution_base: float = 0.020
    execution_rate: float = 0.002
    pixels_per_degree: float = 38.0
    icon_decay_rate: float = 4.0


# ---------------------------------------------------------------------------
# ACT-R visual module
# ---------------------------------------------------------------------------


class ACTRVisualModule:
    """ACT-R visual attention module with EMMA eye movements.

    The module maintains a *visicon* (set of visible objects), a current
    fixation point, and a visual buffer that holds the currently attended
    object.  Encoding times depend on eccentricity and feature frequency.

    Parameters
    ----------
    params : EMMAParams, optional
        EMMA model parameters.  Defaults are used if not supplied.
    """

    def __init__(self, params: Optional[EMMAParams] = None) -> None:
        self.params = params or EMMAParams()

        self._visicon: Dict[str, VisualObject] = {}
        self._fixation = Point2D(0.0, 0.0)
        self._visual_buffer: Optional[VisualObject] = None
        self._icon_buffer: Dict[str, Tuple[VisualObject, float]] = {}

    # ------------------------------------------------------------------ #
    # Visicon management
    # ------------------------------------------------------------------ #

    def add_object(self, obj: VisualObject) -> None:
        """Add or update an object in the visicon."""
        self._visicon[obj.name] = obj

    def remove_object(self, name: str) -> None:
        """Remove an object, moving it to the icon buffer with a timestamp."""
        obj = self._visicon.pop(name, None)
        if obj is not None:
            self._icon_buffer[name] = (obj, 0.0)  # decay starts now

    def clear_visicon(self) -> None:
        """Remove all objects from the visicon."""
        self._visicon.clear()
        self._icon_buffer.clear()

    @property
    def visicon(self) -> List[VisualObject]:
        """All objects currently in the visicon."""
        return list(self._visicon.values())

    @property
    def fixation(self) -> Point2D:
        """Current fixation location."""
        return self._fixation

    @property
    def visual_buffer(self) -> Optional[VisualObject]:
        """Currently attended visual object."""
        return self._visual_buffer

    def set_fixation(self, point: Point2D) -> None:
        """Set the fixation location directly."""
        self._fixation = point

    # ------------------------------------------------------------------ #
    # Eccentricity computation
    # ------------------------------------------------------------------ #

    def eccentricity_pixels(self, target: Point2D) -> float:
        """Pixel distance from current fixation to *target*."""
        return self._fixation.distance_to(target)

    def eccentricity_degrees(self, target: Point2D) -> float:
        """Eccentricity in degrees of visual angle."""
        px = self.eccentricity_pixels(target)
        return px / self.params.pixels_per_degree

    # ------------------------------------------------------------------ #
    # Encoding time (EMMA)
    # ------------------------------------------------------------------ #

    def encoding_time(self, obj: VisualObject) -> float:
        """Predict visual encoding time using the EMMA model.

        .. math::

            T_{\\text{enc}} = K \\cdot e^{k \\cdot \\text{ecc}}
            \\cdot e^{-f \\cdot \\text{freq}}

        where *ecc* is the retinal eccentricity in degrees, *freq* is
        the feature frequency, *K* is the encoding factor, *k* is an
        eccentricity scaling constant (≈ 0.4), and *f* is the frequency
        exponent (Salvucci, 2001).

        Parameters
        ----------
        obj : VisualObject
            Target object to encode.

        Returns
        -------
        float
            Encoding time in seconds.
        """
        ecc = self.eccentricity_degrees(obj.center)
        freq = max(obj.frequency, 0.01)
        k_ecc = 0.4  # eccentricity scaling

        t_enc = (
            self.params.encoding_factor
            * math.exp(k_ecc * ecc)
            * math.exp(-self.params.encoding_exponent * freq)
        )
        return max(t_enc, 0.001)

    def encoding_time_batch(
        self,
        objects: Sequence[VisualObject],
    ) -> NDArray[np.floating]:
        """Compute encoding times for multiple objects (vectorised).

        Returns
        -------
        numpy.ndarray
            Array of encoding times in seconds.
        """
        if not objects:
            return np.array([], dtype=np.float64)

        centers = np.array(
            [[o.center.x, o.center.y] for o in objects], dtype=np.float64
        )
        fix = np.array([self._fixation.x, self._fixation.y], dtype=np.float64)
        ecc_px = np.linalg.norm(centers - fix, axis=1)
        ecc_deg = ecc_px / self.params.pixels_per_degree

        freqs = np.array(
            [max(o.frequency, 0.01) for o in objects], dtype=np.float64
        )

        k_ecc = 0.4
        t_enc = (
            self.params.encoding_factor
            * np.exp(k_ecc * ecc_deg)
            * np.exp(-self.params.encoding_exponent * freqs)
        )
        return np.maximum(t_enc, 0.001)

    # ------------------------------------------------------------------ #
    # Saccade planning and execution
    # ------------------------------------------------------------------ #

    def saccade_time(self, target: Point2D) -> float:
        """Total time for a saccade to *target* (preparation + execution).

        .. math::

            T_{\\text{sac}} = T_{\\text{prep}} + T_{\\text{exec,base}}
            + T_{\\text{exec,rate}} \\cdot \\text{ecc}_{\\deg}

        Parameters
        ----------
        target : Point2D
            Saccade destination.

        Returns
        -------
        float
            Total saccade time in seconds.
        """
        ecc = self.eccentricity_degrees(target)
        execution = (
            self.params.execution_base
            + self.params.execution_rate * ecc
        )
        return self.params.saccade_preparation + execution

    def saccade_time_batch(
        self,
        targets: Sequence[Point2D],
    ) -> NDArray[np.floating]:
        """Saccade times to multiple targets (vectorised)."""
        if not targets:
            return np.array([], dtype=np.float64)

        pts = np.array([[t.x, t.y] for t in targets], dtype=np.float64)
        fix = np.array([self._fixation.x, self._fixation.y], dtype=np.float64)
        ecc_px = np.linalg.norm(pts - fix, axis=1)
        ecc_deg = ecc_px / self.params.pixels_per_degree

        execution = self.params.execution_base + self.params.execution_rate * ecc_deg
        return self.params.saccade_preparation + execution

    def plan_saccade(self, target: Point2D) -> Tuple[float, Point2D]:
        """Plan and execute a saccade, updating fixation.

        Adds Gaussian landing-position noise proportional to the saccade
        amplitude (Rayner, 1998).

        Parameters
        ----------
        target : Point2D
            Intended saccade target.

        Returns
        -------
        tuple[float, Point2D]
            (saccade_time, actual_landing_position).
        """
        t = self.saccade_time(target)
        ecc_deg = self.eccentricity_degrees(target)

        # Landing noise: σ ≈ 0.1 * amplitude in degrees → pixels
        noise_std = 0.1 * ecc_deg * self.params.pixels_per_degree
        rng = np.random.default_rng()
        dx = float(rng.normal(0.0, max(noise_std, 0.1)))
        dy = float(rng.normal(0.0, max(noise_std, 0.1)))

        landed = Point2D(target.x + dx, target.y + dy)
        self._fixation = landed
        return t, landed

    # ------------------------------------------------------------------ #
    # Attend (encode + optional saccade)
    # ------------------------------------------------------------------ #

    def attend(self, obj: VisualObject) -> float:
        """Attend to a visual object: saccade + encode.

        If the object is farther than ~2° from fixation, a saccade is
        programmed first; otherwise only encoding is performed.

        Parameters
        ----------
        obj : VisualObject
            Object to attend to.

        Returns
        -------
        float
            Total attention time in seconds.
        """
        total = 0.0
        ecc = self.eccentricity_degrees(obj.center)

        # Saccade if eccentricity exceeds 2 degrees
        if ecc > 2.0:
            sac_time, _ = self.plan_saccade(obj.center)
            total += sac_time

        total += self.encoding_time(obj)
        self._visual_buffer = obj
        return total

    # ------------------------------------------------------------------ #
    # Feature-based visual search (guided search model)
    # ------------------------------------------------------------------ #

    def guided_search(
        self,
        target_features: Dict[str, Any],
        objects: Optional[Sequence[VisualObject]] = None,
    ) -> Tuple[Optional[VisualObject], float, int]:
        """Perform feature-based guided visual search.

        Objects sharing features with the target are prioritised.  The
        search proceeds serially through the priority-ranked list until
        the target is found or the list is exhausted (Wolfe, 2007).

        Parameters
        ----------
        target_features : dict
            Feature values the target must possess.
        objects : sequence of VisualObject, optional
            Objects to search.  Defaults to the current visicon.

        Returns
        -------
        tuple[VisualObject | None, float, int]
            (found_object, total_time, n_fixations).
        """
        objs = list(objects) if objects is not None else self.visicon
        if not objs:
            return None, 0.0, 0

        # Compute guidance score (number of matching features)
        scores = np.array(
            [
                sum(
                    1 for k, v in target_features.items()
                    if o.features.get(k) == v
                )
                for o in objs
            ],
            dtype=np.float64,
        )
        # Break ties by eccentricity (closer objects first)
        ecc = np.array(
            [self.eccentricity_pixels(o.center) for o in objs],
            dtype=np.float64,
        )
        # Rank: higher score first, then lower eccentricity
        order = np.lexsort((ecc, -scores))

        total_time = 0.0
        n_fixations = 0

        for idx in order:
            obj = objs[int(idx)]
            total_time += self.attend(obj)
            n_fixations += 1

            # Check if this is the target
            match = all(
                obj.features.get(k) == v
                for k, v in target_features.items()
            )
            if match:
                return obj, total_time, n_fixations

        return None, total_time, n_fixations

    # ------------------------------------------------------------------ #
    # Icon persistence and decay
    # ------------------------------------------------------------------ #

    def icon_strength(self, elapsed: float) -> float:
        """Iconic memory trace strength after *elapsed* seconds.

        .. math::

            S = e^{-\\lambda \\cdot t}

        Parameters
        ----------
        elapsed : float
            Time since the object disappeared.

        Returns
        -------
        float
            Trace strength in [0, 1].
        """
        return math.exp(-self.params.icon_decay_rate * max(elapsed, 0.0))

    def available_icons(self, current_time: float) -> List[VisualObject]:
        """Return iconic traces that are still above threshold (0.1).

        Parameters
        ----------
        current_time : float
            Current simulation time.

        Returns
        -------
        list[VisualObject]
            Objects whose iconic trace is still strong enough to be
            accessible.
        """
        threshold = 0.1
        result: List[VisualObject] = []
        for name, (obj, removal_time) in list(self._icon_buffer.items()):
            elapsed = current_time - removal_time
            if self.icon_strength(elapsed) >= threshold:
                result.append(obj)
            else:
                del self._icon_buffer[name]
        return result

    # ------------------------------------------------------------------ #
    # Integration with accessibility tree layout
    # ------------------------------------------------------------------ #

    def from_accessibility_tree(
        self,
        nodes: Sequence[Dict[str, Any]],
        current_time: float = 0.0,
    ) -> List[VisualObject]:
        """Build visicon from accessibility tree nodes.

        Each node dict should have at minimum ``"name"``, ``"role"``,
        and ``"bounds"`` (a dict with x, y, width, height).

        Parameters
        ----------
        nodes : sequence of dict
            Accessibility tree nodes.
        current_time : float
            Current simulation time.

        Returns
        -------
        list[VisualObject]
            Created visual objects (also added to visicon).
        """
        created: List[VisualObject] = []
        for node in nodes:
            bounds = node.get("bounds", {})
            bbox = BoundingBox(
                x=float(bounds.get("x", 0)),
                y=float(bounds.get("y", 0)),
                width=max(float(bounds.get("width", 10)), 1.0),
                height=max(float(bounds.get("height", 10)), 1.0),
            )

            features: Dict[str, Any] = {}
            if "role" in node:
                features["role"] = node["role"]
            if "label" in node:
                features["label"] = node["label"]
            if "color" in node:
                features["color"] = node["color"]

            obj = VisualObject(
                name=str(node.get("name", f"node_{len(created)}")),
                kind=str(node.get("role", "generic")),
                bbox=bbox,
                features=features,
                frequency=float(node.get("frequency", 0.5)),
                onset_time=current_time,
            )
            self.add_object(obj)
            created.append(obj)

        return created

    # ------------------------------------------------------------------ #
    # Saccade planning with eccentricity-dependent encoding
    # ------------------------------------------------------------------ #

    def optimal_saccade_sequence(
        self,
        targets: Sequence[VisualObject],
    ) -> Tuple[List[VisualObject], float]:
        """Plan a saccade sequence visiting all targets.

        Uses a nearest-neighbour heuristic to minimise total eye
        movement time.

        Parameters
        ----------
        targets : sequence of VisualObject
            Objects to visit.

        Returns
        -------
        tuple[list[VisualObject], float]
            (ordered visit sequence, total time).
        """
        if not targets:
            return [], 0.0

        remaining = list(targets)
        sequence: List[VisualObject] = []
        total_time = 0.0

        while remaining:
            # Find nearest unvisited target
            ecc = [
                self.eccentricity_pixels(o.center) for o in remaining
            ]
            nearest_idx = int(np.argmin(ecc))
            obj = remaining.pop(nearest_idx)

            total_time += self.attend(obj)
            sequence.append(obj)

        return sequence, total_time
