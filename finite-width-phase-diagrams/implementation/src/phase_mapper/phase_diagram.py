"""Phase diagram data structure and operations.

Provides:
  - RegimeType: enum of possible regime labels
  - RegimeRegion: a labelled region of parameter space
  - PhaseDiagram: full phase diagram with boundaries, regions, queries,
    serialisation, and comparison
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .boundary import BoundaryCurve


# ======================================================================
# Enums and data structures
# ======================================================================


class RegimeType(enum.Enum):
    """Possible regime labels for a region of the phase diagram."""

    LAZY = "lazy"
    RICH = "rich"
    CHAOTIC = "chaotic"
    ORDERED = "ordered"
    UNKNOWN = "unknown"


@dataclass
class RegimeRegion:
    """A labelled region of parameter space.

    Parameters
    ----------
    label : RegimeType
        Regime classification.
    boundary_curves : list of BoundaryCurve
        Curves forming the region boundary.
    interior_point : np.ndarray or None
        A representative point known to lie inside this region.
    area_estimate : float
        Estimated area of the region in parameter space.
    """

    label: RegimeType = RegimeType.UNKNOWN
    boundary_curves: List[BoundaryCurve] = field(default_factory=list)
    interior_point: Optional[np.ndarray] = None
    area_estimate: float = 0.0


@dataclass
class PhaseDiagram:
    """Full phase diagram with boundaries, regions, and query methods.

    Parameters
    ----------
    boundary_curves : list of BoundaryCurve
        All detected phase boundaries.
    regime_regions : list of RegimeRegion
        Labelled regions of parameter space.
    parameter_names : tuple of str
        Names of the two axes.
    parameter_ranges : dict
        ``{name: (min, max)}`` for each axis.
    confidence_level : float
        Overall confidence level (0–1).
    metadata : dict
        Arbitrary extra information.
    """

    boundary_curves: List[BoundaryCurve] = field(default_factory=list)
    regime_regions: List[RegimeRegion] = field(default_factory=list)
    parameter_names: Tuple[str, str] = ("param_0", "param_1")
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_regime(self, point: np.ndarray) -> RegimeType:
        """Determine the regime at a given parameter-space point.

        Uses a simple closest-interior-point test: the point is assigned
        to the region whose interior point is nearest.

        Parameters
        ----------
        point : np.ndarray
            2-D parameter-space coordinate.

        Returns
        -------
        RegimeType
        """
        point = np.asarray(point, dtype=np.float64)
        best_label = RegimeType.UNKNOWN
        best_dist = np.inf

        for region in self.regime_regions:
            if region.interior_point is None:
                continue
            d = float(np.linalg.norm(point - region.interior_point))
            if d < best_dist:
                best_dist = d
                best_label = region.label

        return best_label

    def boundary_distance(self, point: np.ndarray) -> float:
        """Minimum distance from *point* to the nearest boundary.

        Parameters
        ----------
        point : np.ndarray
            2-D parameter-space coordinate.

        Returns
        -------
        float
        """
        point = np.asarray(point, dtype=np.float64)
        min_dist = np.inf
        for curve in self.boundary_curves:
            coords = curve.effective_coords()
            if len(coords) == 0:
                continue
            dists = np.linalg.norm(coords - point, axis=1)
            d = float(np.min(dists))
            if d < min_dist:
                min_dist = d
        return min_dist

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, other: PhaseDiagram) -> Dict[str, Any]:
        """Compare this phase diagram against *other*.

        Parameters
        ----------
        other : PhaseDiagram

        Returns
        -------
        dict
            Comparison metrics including boundary distances and regime
            agreement at sampled points.
        """
        from .boundary import BoundaryExtractor

        extractor = BoundaryExtractor()
        boundary_metrics = extractor.compare_boundaries(
            self.boundary_curves, other.boundary_curves,
        )

        # Sample grid and compare regime labels
        n_samples = 50
        agreement = self._regime_agreement(other, n_samples)

        return {
            **boundary_metrics,
            "regime_agreement": agreement,
        }

    def _regime_agreement(
        self, other: PhaseDiagram, n_per_axis: int,
    ) -> float:
        """Fraction of sample points where regime labels agree.

        Parameters
        ----------
        other : PhaseDiagram
        n_per_axis : int

        Returns
        -------
        float
        """
        if not self.parameter_ranges or not other.parameter_ranges:
            return 0.0

        name0, name1 = self.parameter_names
        r0 = self.parameter_ranges.get(name0, (0.0, 1.0))
        r1 = self.parameter_ranges.get(name1, (0.0, 1.0))
        xs = np.linspace(r0[0], r0[1], n_per_axis)
        ys = np.linspace(r1[0], r1[1], n_per_axis)

        agree = 0
        total = 0
        for x in xs:
            for y in ys:
                pt = np.array([x, y])
                if self.query_regime(pt) == other.query_regime(pt):
                    agree += 1
                total += 1

        return agree / max(total, 1)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the phase diagram to a plain dictionary.

        Returns
        -------
        dict
        """
        return {
            "parameter_names": list(self.parameter_names),
            "parameter_ranges": {
                k: list(v) for k, v in self.parameter_ranges.items()
            },
            "confidence_level": self.confidence_level,
            "metadata": self.metadata,
            "boundary_curves": [
                {
                    "coords": c.effective_coords().tolist(),
                    "length": c.length,
                    "closed": c.closed,
                }
                for c in self.boundary_curves
            ],
            "regime_regions": [
                {
                    "label": r.label.value,
                    "interior_point": (
                        r.interior_point.tolist()
                        if r.interior_point is not None
                        else None
                    ),
                    "area_estimate": r.area_estimate,
                }
                for r in self.regime_regions
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PhaseDiagram:
        """Deserialise a phase diagram from a dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        PhaseDiagram
        """
        from .boundary import BoundaryCurve, BoundaryPoint

        curves: List[BoundaryCurve] = []
        for cd in d.get("boundary_curves", []):
            coords = np.array(cd["coords"])
            points = [
                BoundaryPoint(coordinates=row) for row in coords
            ]
            curves.append(
                BoundaryCurve(
                    points=points,
                    smoothed_coords=coords,
                    length=cd.get("length", 0.0),
                    closed=cd.get("closed", False),
                )
            )

        regions: List[RegimeRegion] = []
        for rd in d.get("regime_regions", []):
            ip = rd.get("interior_point")
            regions.append(
                RegimeRegion(
                    label=RegimeType(rd["label"]),
                    interior_point=np.array(ip) if ip is not None else None,
                    area_estimate=rd.get("area_estimate", 0.0),
                )
            )

        return cls(
            boundary_curves=curves,
            regime_regions=regions,
            parameter_names=tuple(d.get("parameter_names", ("param_0", "param_1"))),
            parameter_ranges={
                k: tuple(v) for k, v in d.get("parameter_ranges", {}).items()
            },
            confidence_level=d.get("confidence_level", 0.95),
            metadata=d.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def merge(diagrams: List[PhaseDiagram]) -> PhaseDiagram:
        """Merge multiple phase diagrams into one.

        Boundary curves and regions are concatenated.  Parameter ranges
        are expanded to cover all inputs.

        Parameters
        ----------
        diagrams : list of PhaseDiagram

        Returns
        -------
        PhaseDiagram
        """
        if not diagrams:
            return PhaseDiagram()

        all_curves: List[BoundaryCurve] = []
        all_regions: List[RegimeRegion] = []
        merged_ranges: Dict[str, Tuple[float, float]] = {}

        for diag in diagrams:
            all_curves.extend(diag.boundary_curves)
            all_regions.extend(diag.regime_regions)
            for name, (lo, hi) in diag.parameter_ranges.items():
                if name in merged_ranges:
                    prev_lo, prev_hi = merged_ranges[name]
                    merged_ranges[name] = (min(prev_lo, lo), max(prev_hi, hi))
                else:
                    merged_ranges[name] = (lo, hi)

        return PhaseDiagram(
            boundary_curves=all_curves,
            regime_regions=all_regions,
            parameter_names=diagrams[0].parameter_names,
            parameter_ranges=merged_ranges,
            confidence_level=min(d.confidence_level for d in diagrams),
            metadata={"merged_from": len(diagrams)},
        )
