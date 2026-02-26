"""
Utility functions for interval arithmetic operations.
"""

from typing import List, Tuple, Optional, Sequence
from .interval import Interval
import numpy as np


def hull(*intervals) -> Interval:
    """Compute the hull (convex hull) of multiple intervals."""
    if len(intervals) == 1 and hasattr(intervals[0], '__iter__'):
        intervals = list(intervals[0])
    lo = min(iv.lo if isinstance(iv, Interval) else float(iv) for iv in intervals)
    hi = max(iv.hi if isinstance(iv, Interval) else float(iv) for iv in intervals)
    return Interval(lo, hi)


def intersection(a: Interval, b: Interval) -> Interval:
    """Compute intersection of two intervals."""
    return a.intersection(b)


def midpoint(iv: Interval) -> float:
    """Return midpoint of interval."""
    return iv.mid


def radius(iv: Interval) -> float:
    """Return radius of interval."""
    return iv.rad


def width(iv: Interval) -> float:
    """Return width of interval."""
    return iv.width


def contains(outer: Interval, inner) -> bool:
    """Check containment."""
    return outer.contains(inner)


def is_empty(iv: Interval) -> bool:
    """Check if interval is empty."""
    return iv.is_empty()


def is_subset(inner: Interval, outer: Interval) -> bool:
    """Check if inner is a subset of outer."""
    return outer.contains(inner)


def split_interval(iv: Interval, n: int = 2) -> List[Interval]:
    """Split interval into n equal parts."""
    if n <= 0:
        raise ValueError("Number of splits must be positive")
    if n == 1:
        return [iv]
    points = np.linspace(iv.lo, iv.hi, n + 1)
    return [Interval(points[i], points[i + 1]) for i in range(n)]


def bisect_widest(box: List[Interval]) -> Tuple[List[Interval], List[Interval]]:
    """Bisect a box along its widest dimension."""
    if not box:
        raise ValueError("Empty box")
    widths = [iv.width for iv in box]
    dim = int(np.argmax(widths))
    left_iv, right_iv = box[dim].split()
    left_box = list(box)
    left_box[dim] = left_iv
    right_box = list(box)
    right_box[dim] = right_iv
    return left_box, right_box


def box_volume(box: List[Interval]) -> float:
    """Compute volume of an interval box."""
    vol = 1.0
    for iv in box:
        vol *= iv.width
    return vol


def box_contains(outer: List[Interval], inner: List[Interval]) -> bool:
    """Check if outer box contains inner box."""
    if len(outer) != len(inner):
        return False
    return all(o.contains(i) for o, i in zip(outer, inner))


def box_overlaps(a: List[Interval], b: List[Interval]) -> bool:
    """Check if two boxes overlap."""
    if len(a) != len(b):
        return False
    return all(ai.overlaps(bi) for ai, bi in zip(a, b))


def box_intersection(a: List[Interval], b: List[Interval]) -> Optional[List[Interval]]:
    """Compute intersection of two boxes."""
    if len(a) != len(b):
        return None
    result = []
    for ai, bi in zip(a, b):
        inter = ai.intersection(bi)
        if inter.is_empty():
            return None
        result.append(inter)
    return result


def box_hull(a: List[Interval], b: List[Interval]) -> List[Interval]:
    """Compute hull of two boxes."""
    if len(a) != len(b):
        raise ValueError("Box dimensions must match")
    return [ai.hull(bi) for ai, bi in zip(a, b)]


def box_midpoint(box: List[Interval]) -> np.ndarray:
    """Return midpoint of a box as numpy array."""
    return np.array([iv.mid for iv in box])


def box_radius(box: List[Interval]) -> np.ndarray:
    """Return radius vector of a box."""
    return np.array([iv.rad for iv in box])


def box_width(box: List[Interval]) -> np.ndarray:
    """Return width vector of a box."""
    return np.array([iv.width for iv in box])


def box_diameter(box: List[Interval]) -> float:
    """Return diameter (max width) of a box."""
    return max(iv.width for iv in box)


def box_inflate(box: List[Interval], eps: float) -> List[Interval]:
    """Inflate each component of a box by eps."""
    return [iv.inflate(eps) for iv in box]


def box_from_numpy(center: np.ndarray, radius: np.ndarray) -> List[Interval]:
    """Create interval box from center and radius arrays."""
    return [Interval(c - r, c + r) for c, r in zip(center, radius)]


def box_to_numpy(box: List[Interval]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert box to center and radius numpy arrays."""
    center = np.array([iv.mid for iv in box])
    rad = np.array([iv.rad for iv in box])
    return center, rad


def grid_partition(box: List[Interval], divisions: List[int]) -> List[List[Interval]]:
    """Partition a box into a grid of sub-boxes."""
    if len(box) != len(divisions):
        raise ValueError("Dimension mismatch between box and divisions")
    sub_intervals = []
    for iv, n in zip(box, divisions):
        sub_intervals.append(split_interval(iv, n))
    result = []
    _grid_recurse(sub_intervals, [], 0, result)
    return result


def _grid_recurse(sub_intervals, current, dim, result):
    """Recursive helper for grid partition."""
    if dim == len(sub_intervals):
        result.append(list(current))
        return
    for iv in sub_intervals[dim]:
        current.append(iv)
        _grid_recurse(sub_intervals, current, dim + 1, result)
        current.pop()


def hausdorff_distance(a: Interval, b: Interval) -> float:
    """Compute Hausdorff distance between two intervals."""
    return max(abs(a.lo - b.lo), abs(a.hi - b.hi))


def excess_width(enclosure: Interval, true_range: Interval) -> float:
    """Compute excess width (overestimation) of an enclosure."""
    return enclosure.width - true_range.width


def relative_excess(enclosure: Interval, true_range: Interval) -> float:
    """Compute relative excess width."""
    if true_range.width == 0:
        return enclosure.width
    return excess_width(enclosure, true_range) / true_range.width
