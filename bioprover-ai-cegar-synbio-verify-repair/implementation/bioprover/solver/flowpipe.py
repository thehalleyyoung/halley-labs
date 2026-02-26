"""
Flowpipe representation and operations.

A flowpipe is a sequence of time-stamped state enclosures covering a
continuous time interval. Each segment provides a rigorous over-approximation
of all reachable states during its time interval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bioprover.solver.interval import (
    Interval,
    IntervalVector,
    hausdorff_distance,
    hull,
)
from bioprover.solver.taylor_model import TaylorModel


# ---------------------------------------------------------------------------
# FlowpipeSegment
# ---------------------------------------------------------------------------

@dataclass
class FlowpipeSegment:
    """
    A single segment of a flowpipe: a time interval paired with a state enclosure.

    The enclosure may be a box (IntervalVector) or a Taylor model.
    """

    time: Interval
    box: IntervalVector
    taylor_models: Optional[List[TaylorModel]] = None

    @property
    def dim(self) -> int:
        return self.box.dim

    @property
    def t_lo(self) -> float:
        return float(self.time.lo)

    @property
    def t_hi(self) -> float:
        return float(self.time.hi)

    @property
    def duration(self) -> float:
        return self.time.width()

    def midpoint(self) -> np.ndarray:
        return self.box.midpoint()

    def widths(self) -> np.ndarray:
        return self.box.widths()

    def max_width(self) -> float:
        return self.box.max_width()

    def contains_state(self, x: np.ndarray) -> bool:
        """Check if a point state is contained in this segment's box."""
        if len(x) != self.dim:
            return False
        for i in range(self.dim):
            if not self.box[i].contains(x[i]):
                return False
        return True

    def contains_box(self, other: IntervalVector) -> bool:
        return self.box.contains(other)

    def overlaps_box(self, other: IntervalVector) -> bool:
        return self.box.overlaps(other)

    def project(self, indices: List[int]) -> "FlowpipeSegment":
        """Project onto a subspace."""
        return FlowpipeSegment(
            time=Interval(self.time.lo, self.time.hi),
            box=self.box.project(indices),
        )

    def bloat(self, eps: float) -> "FlowpipeSegment":
        """Enlarge the enclosure by eps in every dimension."""
        return FlowpipeSegment(
            time=Interval(self.time.lo, self.time.hi),
            box=self.box.bloat(eps),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "time_lo": float(self.time.lo),
            "time_hi": float(self.time.hi),
            "box_lo": self.box.lo_array().tolist(),
            "box_hi": self.box.hi_array().tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowpipeSegment":
        """Deserialize from a dictionary."""
        lo = np.array(d["box_lo"])
        hi = np.array(d["box_hi"])
        return cls(
            time=Interval(d["time_lo"], d["time_hi"]),
            box=IntervalVector.from_bounds(lo, hi),
        )


# ---------------------------------------------------------------------------
# Flowpipe
# ---------------------------------------------------------------------------

class Flowpipe:
    """
    A flowpipe: a sequence of FlowpipeSegments covering [t0, tf].

    Provides operations for union, intersection, containment checking,
    projection, bloating, and serialization.
    """

    def __init__(self, segments: Optional[List[FlowpipeSegment]] = None) -> None:
        self._segments: List[FlowpipeSegment] = segments or []

    @classmethod
    def from_integration_result(cls, result) -> "Flowpipe":
        """Build a Flowpipe from an IntegrationResult (from ode_integrator)."""
        segments = []
        for step in result.steps:
            seg = FlowpipeSegment(
                time=step.t_interval,
                box=step.enclosure,
            )
            segments.append(seg)
        return cls(segments)

    # -- accessors -----------------------------------------------------------

    @property
    def segments(self) -> List[FlowpipeSegment]:
        return self._segments

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, i: int) -> FlowpipeSegment:
        return self._segments[i]

    def __iter__(self):
        return iter(self._segments)

    @property
    def dim(self) -> int:
        if not self._segments:
            return 0
        return self._segments[0].dim

    @property
    def t0(self) -> Optional[float]:
        if not self._segments:
            return None
        return self._segments[0].t_lo

    @property
    def tf(self) -> Optional[float]:
        if not self._segments:
            return None
        return self._segments[-1].t_hi

    @property
    def time_span(self) -> Optional[Interval]:
        if not self._segments:
            return None
        return Interval(self._segments[0].t_lo, self._segments[-1].t_hi)

    def append(self, segment: FlowpipeSegment) -> None:
        self._segments.append(segment)

    # -- reachable set -------------------------------------------------------

    def reachable_set(self) -> Optional[IntervalVector]:
        """Over-approximation of the entire reachable set (hull of all boxes)."""
        if not self._segments:
            return None
        result = self._segments[0].box.copy()
        for seg in self._segments[1:]:
            result = result.hull(seg.box)
        return result

    def reachable_set_at(self, t: float) -> Optional[IntervalVector]:
        """Find the enclosure at a specific time point."""
        for seg in self._segments:
            if seg.time.contains(t):
                return seg.box.copy()
        return None

    def reachable_set_over(self, t_interval: Interval) -> Optional[IntervalVector]:
        """Hull of enclosures overlapping the given time interval."""
        result = None
        for seg in self._segments:
            if seg.time.overlaps(t_interval):
                if result is None:
                    result = seg.box.copy()
                else:
                    result = result.hull(seg.box)
        return result

    # -- containment and intersection ----------------------------------------

    def contains_flowpipe(self, other: "Flowpipe") -> bool:
        """Check if this flowpipe contains another (segment by segment)."""
        for other_seg in other._segments:
            t_mid = other_seg.time.mid()
            my_enc = self.reachable_set_over(other_seg.time)
            if my_enc is None:
                return False
            if not my_enc.contains(other_seg.box):
                return False
        return True

    def contains_trajectory(self, times: np.ndarray, states: np.ndarray) -> bool:
        """Check if a sampled trajectory is contained in the flowpipe."""
        for i, t in enumerate(times):
            enc = self.reachable_set_at(float(t))
            if enc is None:
                return False
            for j in range(states.shape[1]):
                if not enc[j].contains(states[i, j]):
                    return False
        return True

    def union(self, other: "Flowpipe") -> "Flowpipe":
        """
        Compute the union of two flowpipes as a new flowpipe.
        For overlapping time segments, takes the hull.
        """
        all_segments = sorted(
            self._segments + other._segments,
            key=lambda s: s.t_lo,
        )
        if not all_segments:
            return Flowpipe()

        merged: List[FlowpipeSegment] = [all_segments[0]]
        for seg in all_segments[1:]:
            last = merged[-1]
            if seg.time.overlaps(last.time):
                # Merge: hull of boxes, hull of time intervals
                new_time = hull(last.time, seg.time)
                new_box = last.box.hull(seg.box)
                merged[-1] = FlowpipeSegment(time=new_time, box=new_box)
            else:
                merged.append(seg)
        return Flowpipe(merged)

    def intersection(self, other: "Flowpipe") -> Optional["Flowpipe"]:
        """
        Intersection of two flowpipes.
        Returns None if any segment pair has empty intersection.
        """
        result_segments: List[FlowpipeSegment] = []
        for seg_a in self._segments:
            for seg_b in other._segments:
                t_inter = None
                try:
                    from bioprover.solver.interval import intersection as iv_inter
                    t_inter = iv_inter(seg_a.time, seg_b.time)
                except Exception:
                    pass
                if t_inter is None:
                    continue
                box_inter = seg_a.box.intersection(seg_b.box)
                if box_inter is None:
                    return None
                result_segments.append(FlowpipeSegment(time=t_inter, box=box_inter))
        if not result_segments:
            return None
        return Flowpipe(result_segments)

    # -- projection ----------------------------------------------------------

    def project(self, indices: List[int]) -> "Flowpipe":
        """Project all segments onto a subspace."""
        return Flowpipe([seg.project(indices) for seg in self._segments])

    # -- bloating ------------------------------------------------------------

    def bloat(self, eps: float) -> "Flowpipe":
        """Enlarge all enclosures by eps."""
        return Flowpipe([seg.bloat(eps) for seg in self._segments])

    # -- metrics -------------------------------------------------------------

    def max_width(self) -> float:
        """Maximum enclosure width across all segments and dimensions."""
        if not self._segments:
            return 0.0
        return max(seg.max_width() for seg in self._segments)

    def total_volume(self) -> float:
        """Sum of hyper-volumes of all segment boxes (rough measure)."""
        vol = 0.0
        for seg in self._segments:
            seg_vol = 1.0
            for i in range(seg.dim):
                seg_vol *= seg.box[i].width()
            vol += seg_vol
        return vol

    def hausdorff_distance_to(self, other: "Flowpipe") -> float:
        """
        Approximate Hausdorff distance between two flowpipes.
        Computed as the max over corresponding segments.
        """
        d = 0.0
        # Match segments by time overlap
        for seg_a in self._segments:
            best_match = float("inf")
            for seg_b in other._segments:
                if seg_a.time.overlaps(seg_b.time):
                    hd = hausdorff_distance(seg_a.box, seg_b.box)
                    best_match = min(best_match, hd)
            if best_match < float("inf"):
                d = max(d, best_match)
        return d

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "num_segments": len(self._segments),
            "segments": [seg.to_dict() for seg in self._segments],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Flowpipe":
        segments = [FlowpipeSegment.from_dict(s) for s in d["segments"]]
        return cls(segments)

    @classmethod
    def from_json(cls, s: str) -> "Flowpipe":
        return cls.from_dict(json.loads(s))

    # -- visualization data export -------------------------------------------

    def to_plot_data(
        self, var_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Export data for plotting.

        Returns dict with arrays suitable for matplotlib fill_between / patches.
        """
        if not self._segments:
            return {"times_lo": [], "times_hi": [], "variables": {}}

        if var_indices is None:
            var_indices = list(range(self.dim))

        times_lo = [seg.t_lo for seg in self._segments]
        times_hi = [seg.t_hi for seg in self._segments]
        t_mid = [seg.time.mid() for seg in self._segments]

        variables: Dict[str, Dict[str, List[float]]] = {}
        for vi in var_indices:
            lo_vals = [float(seg.box[vi].lo) for seg in self._segments]
            hi_vals = [float(seg.box[vi].hi) for seg in self._segments]
            mid_vals = [seg.box[vi].mid() for seg in self._segments]
            variables[f"x{vi}"] = {
                "lo": lo_vals,
                "hi": hi_vals,
                "mid": mid_vals,
            }

        return {
            "times_lo": times_lo,
            "times_hi": times_hi,
            "times_mid": t_mid,
            "variables": variables,
        }

    def __repr__(self) -> str:
        if not self._segments:
            return "Flowpipe(empty)"
        return (
            f"Flowpipe({len(self._segments)} segments, "
            f"t=[{self.t0:.6g}, {self.tf:.6g}], dim={self.dim})"
        )
