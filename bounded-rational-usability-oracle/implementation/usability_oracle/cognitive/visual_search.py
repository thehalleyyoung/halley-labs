"""Visual search models based on feature integration theory and guided search.

Implements computational models of visual search performance grounded in
the Feature Integration Theory (Treisman & Gelade, 1980) and Guided Search
model (Wolfe, 1994, 2007). These models predict search times, fixation
counts, and set-size effects for feature, conjunction, and guided search
tasks in user interfaces.

References
----------
Treisman, A., & Gelade, G. (1980). A feature-integration theory of
    attention. Cognitive Psychology, 12(1), 97-136.
Wolfe, J. M. (1994). Guided Search 2.0: A revised model of visual search.
    Psychonomic Bulletin & Review, 1(2), 202-238.
Wolfe, J. M. (2007). Guided Search 4.0: Current progress with a model of
    visual search. In W. Gray (Ed.), Integrated Models of Cognitive Systems
    (pp. 99-119). Oxford University Press.
Duncan, J., & Humphreys, G. W. (1989). Visual search and stimulus
    similarity. Psychological Review, 96(3), 433-458.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from usability_oracle.cognitive.models import BoundingBox, PerceptualScene, Point2D


class VisualSearchModel:
    """Computational model of human visual search performance.

    Models three search regimes:
    - Parallel (pop-out): Target differs from distractors by a single
      basic feature. RT is approximately independent of set size.
    - Serial (conjunction): Target shares features with distractors.
      RT increases linearly with set size (Treisman & Gelade, 1980).
    - Guided: Top-down guidance reduces the effective set size by
      prioritising likely target locations (Wolfe, 2007).

    All times are in seconds unless otherwise noted.
    """

    # --- Class-level constants (empirical parameters) -----------------------

    EFFICIENT_SLOPE: float = 0.005
    """Slope for efficient (feature/pop-out) search: ~5 ms/item
    (Treisman & Gelade, 1980)."""

    INEFFICIENT_SLOPE: float = 0.025
    """Slope for inefficient (conjunction) search: ~25 ms/item
    (Treisman & Gelade, 1980)."""

    BASE_RT: float = 0.400
    """Base reaction time (400 ms) encompassing perceptual encoding,
    response selection and motor execution."""

    # --- Core prediction methods --------------------------------------------

    @staticmethod
    def predict_serial(
        n_items: int,
        target_present: bool = True,
        slope: float = INEFFICIENT_SLOPE,
    ) -> float:
        """Predict reaction time for serial self-terminating search.

        In serial search the observer inspects items one at a time.
        When the target is present, on average half the items are checked
        before the target is found.  When absent, all items must be
        checked (Treisman & Gelade, 1980).

        Parameters
        ----------
        n_items : int
            Number of items in the display (set size).
        target_present : bool
            Whether the target is present in the display.
        slope : float
            Time per item (seconds).  Defaults to INEFFICIENT_SLOPE.

        Returns
        -------
        float
            Predicted reaction time in seconds.
        """
        n_items = max(1, int(n_items))
        if target_present:
            return VisualSearchModel.BASE_RT + slope * n_items / 2.0
        return VisualSearchModel.BASE_RT + slope * n_items

    @staticmethod
    def predict_parallel(
        n_items: int,
        base_rt: float = BASE_RT,
    ) -> float:
        """Predict reaction time for parallel (pop-out / feature) search.

        In pop-out search the target is detected pre-attentively and RT
        is nearly independent of the number of distractors.  A small
        logarithmic component accounts for statistical-decision noise
        that grows weakly with set size (Wolfe, 2007).

        Parameters
        ----------
        n_items : int
            Number of items in the display.
        base_rt : float
            Baseline reaction time in seconds.

        Returns
        -------
        float
            Predicted reaction time in seconds.
        """
        n_items = max(1, int(n_items))
        # Small logarithmic set-size effect observed empirically:
        # approximately 2 ms per doubling of set size.
        log_factor = 0.002
        return base_rt + log_factor * math.log2(max(1, n_items))

    @staticmethod
    def predict_guided(
        n_items: int,
        guidance_factor: float,
        slope: float = INEFFICIENT_SLOPE,
    ) -> float:
        """Predict RT under guided search (Wolfe, 2007).

        Top-down guidance allows the observer to restrict attention to a
        subset of the display.  The *guidance_factor* (0–1) captures the
        proportion of distractors that can be rejected pre-attentively,
        reducing the effective set size before serial inspection begins.

        Parameters
        ----------
        n_items : int
            Total number of items in the display.
        guidance_factor : float
            Proportion of distractors filtered (0 = no guidance, 1 = perfect).
        slope : float
            Per-item inspection time (seconds).

        Returns
        -------
        float
            Predicted reaction time in seconds.
        """
        n_items = max(1, int(n_items))
        guidance_factor = float(np.clip(guidance_factor, 0.0, 1.0))
        effective_n = n_items * (1.0 - guidance_factor)
        effective_n = max(1.0, effective_n)
        # Serial self-terminating on the effective set (target present)
        return VisualSearchModel.BASE_RT + slope * effective_n / 2.0

    # --- Saliency and set-size helpers --------------------------------------

    @staticmethod
    def saliency_from_structure(
        element_bbox: BoundingBox,
        siblings: list[BoundingBox],
    ) -> float:
        """Compute structural saliency of an element relative to siblings.

        Saliency is a weighted combination of three factors normalised
        to [0, 1]:
          1. **Relative size** – ratio of the element's area to the
             median sibling area.
          2. **Isolation** – mean distance to siblings normalised by the
             scene diagonal.
          3. **Eccentricity** – proximity to the centre of the collective
             bounding box (central items are more salient in Western
             reading cultures).

        Inspired by the feature-contrast component of Wolfe's Guided
        Search 4.0 (Wolfe, 2007) and Duncan & Humphreys (1989).

        Parameters
        ----------
        element_bbox : BoundingBox
            The bounding box of the target element.
        siblings : list[BoundingBox]
            Bounding boxes of all other elements in the display.

        Returns
        -------
        float
            Saliency score in [0, 1].
        """
        if not siblings:
            return 1.0

        # --- 1. Relative size -----------------------------------------------
        element_area = element_bbox.width * element_bbox.height
        sibling_areas = np.array(
            [s.width * s.height for s in siblings], dtype=np.float64
        )
        median_area = float(np.median(sibling_areas)) if len(sibling_areas) > 0 else 1.0
        median_area = max(median_area, 1e-9)
        size_ratio = element_area / median_area
        # Map through a sigmoid so extreme ratios saturate at 0/1
        size_saliency = 1.0 / (1.0 + math.exp(-2.0 * (size_ratio - 1.0)))

        # --- 2. Isolation ---------------------------------------------------
        element_center = element_bbox.center()
        distances = np.array(
            [element_center.distance_to(s.center()) for s in siblings],
            dtype=np.float64,
        )
        all_boxes = [element_bbox] + list(siblings)
        xs = [b.x for b in all_boxes] + [b.x + b.width for b in all_boxes]
        ys = [b.y for b in all_boxes] + [b.y + b.height for b in all_boxes]
        scene_diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        scene_diagonal = max(scene_diagonal, 1e-9)
        mean_dist = float(np.mean(distances))
        isolation_saliency = min(1.0, mean_dist / (scene_diagonal * 0.5))

        # --- 3. Eccentricity from scene centre ------------------------------
        scene_cx = (min(xs) + max(xs)) / 2.0
        scene_cy = (min(ys) + max(ys)) / 2.0
        scene_center = Point2D(scene_cx, scene_cy)
        eccentricity = element_center.distance_to(scene_center)
        max_ecc = scene_diagonal / 2.0
        # Central items are more salient → invert the ratio
        eccentricity_saliency = 1.0 - min(1.0, eccentricity / max(max_ecc, 1e-9))

        # Weighted combination
        w_size, w_iso, w_ecc = 0.40, 0.35, 0.25
        raw = w_size * size_saliency + w_iso * isolation_saliency + w_ecc * eccentricity_saliency
        return float(np.clip(raw, 0.0, 1.0))

    @staticmethod
    def effective_set_size(
        elements: list[BoundingBox],
        labels: list[str],
        target_label: str,
    ) -> float:
        """Estimate the effective set size for visual search.

        Items that share the same label as the target are counted as
        distractors that must be serially inspected.  Items with a
        different label can be filtered via feature guidance and
        contribute only partially to the search load (Duncan &
        Humphreys, 1989).

        Parameters
        ----------
        elements : list[BoundingBox]
            Bounding boxes for all display elements.
        labels : list[str]
            Category label for each element (same length as *elements*).
        target_label : str
            Label of the search target.

        Returns
        -------
        float
            Effective number of items that must be inspected.
        """
        if not elements:
            return 1.0
        same_label_count = 0
        diff_label_count = 0
        for label in labels:
            if label == target_label:
                same_label_count += 1
            else:
                diff_label_count += 1
        # Distractors sharing the target label are fully inspected.
        # Others contribute a reduced load (guidance discounts ~70%).
        guidance_discount = 0.3
        effective = same_label_count + guidance_discount * diff_label_count
        return max(1.0, effective)

    # --- Eccentricity and fixation models -----------------------------------

    @staticmethod
    def eccentricity_cost(
        fixation: Point2D,
        target: Point2D,
        acuity_falloff: float = 0.05,
    ) -> float:
        """Compute the cost factor due to retinal eccentricity.

        Visual acuity degrades linearly with distance from the fovea.
        The cost factor inflates processing time for peripheral targets.

        Parameters
        ----------
        fixation : Point2D
            Current fixation location (pixels).
        target : Point2D
            Target location (pixels).
        acuity_falloff : float
            Proportional increase in processing time per pixel of
            eccentricity.  A value of 0.05 means 5 % increase per
            degree-equivalent pixel.

        Returns
        -------
        float
            Multiplicative cost factor (>= 1.0).
        """
        eccentricity_px = fixation.distance_to(target)
        # Approximate conversion: 1 degree ≈ 38 px at typical viewing distance
        pixels_per_degree = 38.0
        eccentricity_deg = eccentricity_px / pixels_per_degree
        return 1.0 + acuity_falloff * eccentricity_deg

    @staticmethod
    def search_time_distribution(
        n_items: int,
        slope: float,
        target_present: bool = True,
    ) -> tuple[float, float]:
        """Return (mean, variance) of the search time distribution.

        For a serial self-terminating search the position of the target
        among inspected items is uniformly distributed, yielding
        analytical expressions for the first two moments (Wolfe, 2007).

        Parameters
        ----------
        n_items : int
            Display set size.
        slope : float
            Per-item inspection time (seconds).
        target_present : bool
            Whether the target is in the display.

        Returns
        -------
        tuple[float, float]
            (mean_rt, variance_rt) in seconds and seconds² respectively.
        """
        n_items = max(1, int(n_items))
        base = VisualSearchModel.BASE_RT
        if target_present:
            mean = base + slope * n_items / 2.0
            # Uniform over {1, …, n}: Var = slope² * n(n+2)/12
            variance = (slope ** 2) * n_items * (n_items + 2) / 12.0
        else:
            mean = base + slope * n_items
            # No variance from search position; only residual motor noise
            variance = (slope ** 2) * n_items / 12.0
        return (mean, variance)

    @staticmethod
    def predict_with_eccentricity(
        elements: list[BoundingBox],
        fixation: Point2D,
        target_idx: int,
        slope: float,
    ) -> float:
        """Predict search time accounting for retinal eccentricity.

        Each element's inspection cost is scaled by its eccentricity
        from the current fixation.  The observer searches items in order
        of increasing eccentricity (nearest first), consistent with
        optimal foraging in visual search (Najemnik & Geisler, 2005).

        Parameters
        ----------
        elements : list[BoundingBox]
            Display elements.
        fixation : Point2D
            Current gaze position.
        target_idx : int
            Index of the target element in *elements*.
        slope : float
            Base per-item inspection time (seconds).

        Returns
        -------
        float
            Predicted total search time in seconds.
        """
        if not elements:
            return VisualSearchModel.BASE_RT

        n = len(elements)
        target_idx = int(np.clip(target_idx, 0, n - 1))

        # Compute eccentricity cost for every element
        centers = [e.center() for e in elements]
        costs = np.array(
            [
                VisualSearchModel.eccentricity_cost(fixation, c)
                for c in centers
            ],
            dtype=np.float64,
        )

        # Sort by eccentricity (nearest inspected first)
        ecc_distances = np.array(
            [fixation.distance_to(c) for c in centers], dtype=np.float64
        )
        order = np.argsort(ecc_distances)

        # Serial self-terminating: inspect until target found
        total = VisualSearchModel.BASE_RT
        for idx in order:
            total += slope * costs[idx]
            if idx == target_idx:
                break
        return float(total)

    @staticmethod
    def number_of_fixations(
        n_items: int,
        target_present: bool = True,
    ) -> float:
        """Expected number of fixations in a serial search.

        Each fixation inspects approximately one item.  When the target
        is present, on average (n + 1) / 2 fixations are needed; when
        absent, n fixations are required (Treisman & Gelade, 1980).

        Parameters
        ----------
        n_items : int
            Display set size.
        target_present : bool
            Whether the target is in the display.

        Returns
        -------
        float
            Expected number of fixations.
        """
        n_items = max(1, int(n_items))
        if target_present:
            return (n_items + 1) / 2.0
        return float(n_items)
