"""
usability_oracle.bottleneck.perceptual — Perceptual overload detector.

Detects states where the visual information density exceeds the user's
perceptual channel capacity, leading to degraded search performance and
increased error rates.

Information-theoretic model:

    H(visual_field) = -Σᵢ pᵢ log pᵢ

where pᵢ is the salience-weighted probability of fixating element i.  When
H exceeds the visual channel capacity C_visual, perceptual overload occurs:

    overload iff H(visual_field) > C_visual

The effective set size (ESS) determines search time via:

    T_search = a + b · ESS    (linear visual search model)

where ESS = 2^{H} is the effective number of equally-probable targets.

Clutter is measured via pairwise overlap of bounding boxes:

    clutter = (1/n²) Σᵢ Σⱼ overlap_area(BBᵢ, BBⱼ) / min(area(BBᵢ), area(BBⱼ))

References
----------
- Rosenholtz, Li & Nakano (2007). Measuring visual clutter. *JoV*.
- Wolfe & Horowitz (2017). Five factors that guide attention in visual
  search. *Nature Human Behaviour*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import MDP


# ---------------------------------------------------------------------------
# PerceptualOverloadDetector
# ---------------------------------------------------------------------------

@dataclass
class PerceptualOverloadDetector:
    """Detect perceptual overload at UI states.

    A state exhibits perceptual overload when the visual entropy exceeds
    the channel capacity, or when the effective set size (2^H) is too
    large for efficient visual search.

    Parameters
    ----------
    OVERLOAD_THRESHOLD : float
        Entropy threshold (nats) above which overload is flagged.
        Default 3.0 nats ≈ 4.3 bits ≈ ~20 equally-salient items.
    CLUTTER_THRESHOLD : float
        Clutter measure threshold (0–1).  Default 0.3.
    INFO_DENSITY_THRESHOLD : float
        Maximum information density (elements per 100×100 px region).
    VISUAL_CHANNEL_CAPACITY : float
        Assumed visual channel capacity in nats/s.
    """

    OVERLOAD_THRESHOLD: float = 3.0
    CLUTTER_THRESHOLD: float = 0.3
    INFO_DENSITY_THRESHOLD: float = 0.5
    VISUAL_CHANNEL_CAPACITY: float = 40.0

    # ── Public API --------------------------------------------------------

    def detect(
        self,
        state: str,
        mdp: MDP,
        features: dict[str, float],
    ) -> Optional[BottleneckResult]:
        """Detect perceptual overload at *state*.

        Parameters
        ----------
        state : str
            State identifier.
        mdp : MDP
            The MDP (for action/transition context).
        features : dict[str, float]
            State features, expected to include:
            - ``n_elements``: number of UI elements visible
            - ``visual_complexity``: pre-computed complexity score
            - ``element_sizes``: comma-separated sizes (optional)
            - ``bounding_box_overlaps``: overlap count (optional)

        Returns
        -------
        BottleneckResult or None
            A perceptual-overload bottleneck if detected, else None.
        """
        n_elements = features.get("n_elements", 0.0)
        visual_complexity = features.get("visual_complexity", 0.0)

        # Estimate element salience distribution
        elements = self._build_element_list(features, int(n_elements))

        # Compute visual entropy
        entropy = self._visual_entropy(elements)

        # Compute effective set size
        ess = self._effective_set_size(elements, "target")

        # Compute clutter
        bboxes = self._build_bounding_boxes(features, int(n_elements))
        clutter = self._clutter_measure(bboxes)

        # Information density
        viewport_area = features.get("viewport_width", 1920.0) * features.get(
            "viewport_height", 1080.0
        )
        info_density = self._information_density(viewport_area, int(n_elements))

        # Decision logic
        is_overloaded = False
        confidence = 0.0
        evidence: dict[str, float] = {
            "visual_entropy": entropy,
            "effective_set_size": ess,
            "clutter": clutter,
            "information_density": info_density,
            "n_elements": n_elements,
        }

        if entropy > self.OVERLOAD_THRESHOLD:
            is_overloaded = True
            confidence = max(confidence, min(1.0, entropy / (2.0 * self.OVERLOAD_THRESHOLD)))

        if clutter > self.CLUTTER_THRESHOLD:
            is_overloaded = True
            confidence = max(confidence, min(1.0, clutter / self.CLUTTER_THRESHOLD * 0.8))

        if ess > 20:
            is_overloaded = True
            confidence = max(confidence, min(1.0, ess / 40.0))

        if not is_overloaded:
            return None

        severity = self._severity_from_entropy(entropy, clutter)

        return BottleneckResult(
            bottleneck_type=BottleneckType.PERCEPTUAL_OVERLOAD,
            severity=severity,
            confidence=confidence,
            affected_states=[state],
            affected_actions=mdp.get_actions(state),
            cognitive_law=CognitiveLaw.VISUAL_SEARCH,
            channel="visual",
            evidence=evidence,
            description=(
                f"Perceptual overload at state {state!r}: "
                f"H={entropy:.2f} nats (ESS={ess:.0f}), "
                f"clutter={clutter:.2f}"
            ),
            recommendation=(
                "Reduce visual complexity by grouping related elements, "
                "removing non-essential items, or improving visual hierarchy."
            ),
            repair_hints=[
                "Group related elements with whitespace or borders",
                "Reduce total number of visible elements",
                "Increase visual salience of primary actions",
                "Use progressive disclosure for secondary options",
            ],
        )

    # ── Information-theoretic measures ------------------------------------

    def _visual_entropy(self, elements: list[dict[str, float]]) -> float:
        """Compute the visual entropy H(visual field).

        H = -Σᵢ pᵢ log(pᵢ)

        where pᵢ is the salience-normalised probability of element i
        attracting a fixation.

        Parameters
        ----------
        elements : list[dict]
            Each element has a ``"salience"`` key.

        Returns
        -------
        float
            Entropy in nats.
        """
        if not elements:
            return 0.0

        saliences = np.array([e.get("salience", 1.0) for e in elements])
        total = np.sum(saliences)
        if total <= 0:
            return 0.0

        probs = saliences / total
        # Filter out zeros to avoid log(0)
        probs = probs[probs > 0]
        return -float(np.sum(probs * np.log(probs)))

    def _effective_set_size(
        self,
        elements: list[dict[str, float]],
        target_role: str,
    ) -> int:
        """Compute the effective set size: ESS = 2^{H}.

        This represents the number of equally-probable items that would
        produce the same entropy.

        Parameters
        ----------
        elements : list[dict]
        target_role : str
            The role to search for (used for weighting).

        Returns
        -------
        int
            Effective set size (rounded up).
        """
        entropy = self._visual_entropy(elements)
        return max(1, int(math.ceil(math.exp(entropy))))

    def _clutter_measure(self, bounding_boxes: list[dict[str, float]]) -> float:
        """Compute overlap-based clutter measure.

        clutter = (2 / n(n-1)) Σᵢ<ⱼ overlap_area(i,j) / min(area_i, area_j)

        Parameters
        ----------
        bounding_boxes : list[dict]
            Each box has keys ``x``, ``y``, ``width``, ``height``.

        Returns
        -------
        float
            Clutter score ∈ [0, 1].
        """
        n = len(bounding_boxes)
        if n < 2:
            return 0.0

        total_overlap = 0.0
        n_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                bi = bounding_boxes[i]
                bj = bounding_boxes[j]

                # Compute overlap
                x_overlap = max(
                    0.0,
                    min(bi["x"] + bi["width"], bj["x"] + bj["width"])
                    - max(bi["x"], bj["x"]),
                )
                y_overlap = max(
                    0.0,
                    min(bi["y"] + bi["height"], bj["y"] + bj["height"])
                    - max(bi["y"], bj["y"]),
                )
                overlap_area = x_overlap * y_overlap

                area_i = bi["width"] * bi["height"]
                area_j = bj["width"] * bj["height"]
                min_area = min(area_i, area_j)

                if min_area > 0:
                    total_overlap += overlap_area / min_area
                n_pairs += 1

        if n_pairs == 0:
            return 0.0
        return min(1.0, total_overlap / n_pairs)

    def _information_density(
        self,
        region_area: float,
        n_elements: int,
    ) -> float:
        """Compute information density: elements per normalised area.

        density = n_elements / (region_area / 10000)

        where 10000 = 100×100 pixels is the reference area.

        Parameters
        ----------
        region_area : float
            Area in square pixels.
        n_elements : int
            Number of interactive elements.

        Returns
        -------
        float
        """
        if region_area <= 0:
            return float("inf") if n_elements > 0 else 0.0
        normalised_area = region_area / 10000.0
        return n_elements / normalised_area

    # ── Helpers -----------------------------------------------------------

    def _build_element_list(
        self,
        features: dict[str, float],
        n_elements: int,
    ) -> list[dict[str, float]]:
        """Build a synthetic element list from state features."""
        if n_elements <= 0:
            return []
        elements = []
        for i in range(int(n_elements)):
            salience = features.get(f"element_{i}_salience", 1.0)
            elements.append({"salience": salience, "index": float(i)})
        # If individual saliences aren't available, use uniform
        if not any(f"element_{i}_salience" in features for i in range(min(3, n_elements))):
            elements = [{"salience": 1.0, "index": float(i)} for i in range(n_elements)]
        return elements

    def _build_bounding_boxes(
        self,
        features: dict[str, float],
        n_elements: int,
    ) -> list[dict[str, float]]:
        """Build bounding boxes from features or synthesise defaults."""
        boxes = []
        for i in range(n_elements):
            x = features.get(f"element_{i}_x", float(i * 50 % 800))
            y = features.get(f"element_{i}_y", float(i * 40 // 800 * 40))
            w = features.get(f"element_{i}_width", 80.0)
            h = features.get(f"element_{i}_height", 30.0)
            boxes.append({"x": x, "y": y, "width": w, "height": h})
        return boxes

    def _severity_from_entropy(self, entropy: float, clutter: float) -> Severity:
        """Map entropy and clutter to severity level."""
        score = entropy / self.OVERLOAD_THRESHOLD + clutter / self.CLUTTER_THRESHOLD
        if score > 3.0:
            return Severity.CRITICAL
        elif score > 2.0:
            return Severity.HIGH
        elif score > 1.2:
            return Severity.MEDIUM
        elif score > 0.8:
            return Severity.LOW
        return Severity.INFO
