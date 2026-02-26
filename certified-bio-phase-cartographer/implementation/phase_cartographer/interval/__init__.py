"""
Interval arithmetic engine for validated numerics.

Provides rigorous interval arithmetic with directed rounding,
interval matrix operations, affine arithmetic, and Taylor model arithmetic.
"""

from .interval import Interval, iv
from .rounding import RoundingMode, set_rounding, get_rounding
from .matrix import IntervalMatrix, IntervalVector
from .affine import AffineForm
from .taylor_model import TaylorModel, TaylorModelVector
from .utils import (
    hull, intersection, midpoint, radius, width,
    contains, is_empty, is_subset, split_interval
)

__all__ = [
    "Interval", "iv",
    "RoundingMode", "set_rounding", "get_rounding",
    "IntervalMatrix", "IntervalVector",
    "AffineForm",
    "TaylorModel", "TaylorModelVector",
    "hull", "intersection", "midpoint", "radius", "width",
    "contains", "is_empty", "is_subset", "split_interval",
]
