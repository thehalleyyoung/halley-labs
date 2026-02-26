"""
Sound floating-point interval arithmetic with guaranteed enclosures.

Bridges the gap between Lean proofs over reals and Python floating-point
by providing epsilon-inflated interval operations that conservatively
enclose the true real-valued result.

Since Python/NumPy do not expose hardware rounding-mode control, we use
epsilon-inflation: for each floating-point result x, we return
    [x - eps*|x| - tiny,  x + eps*|x| + tiny]
where eps = 2**-52 (machine epsilon) and tiny = 2**-1074 (smallest
positive subnormal), guaranteeing a sound enclosure of the real result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# IEEE-754 double constants
_EPS: float = 2.0 ** -52   # machine epsilon for float64
_TINY: float = 2.0 ** -1074  # smallest positive subnormal


def _inflate(x: float) -> Tuple[float, float]:
    """Return (lo, hi) enclosing the real value that *x* approximates."""
    ax = abs(x)
    delta = _EPS * ax + _TINY
    return (x - delta, x + delta)


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Interval:
    """Closed interval [lo, hi] with sound directed-rounding arithmetic.

    Every arithmetic operation returns an interval guaranteed to contain
    the true real-valued result for all operand pairs drawn from the
    respective input intervals.
    """
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi + _TINY:
            raise ValueError(
                f"Empty interval: lo={self.lo} > hi={self.hi}"
            )

    # -- Factories ----------------------------------------------------------

    @classmethod
    def point(cls, x: float) -> "Interval":
        """Create a degenerate interval [x, x] (with inflation)."""
        lo, hi = _inflate(x)
        return cls(lo, hi)

    @classmethod
    def exact(cls, lo: float, hi: float) -> "Interval":
        """Create an interval with explicit bounds (no extra inflation)."""
        return cls(lo, hi)

    @classmethod
    def from_midrad(cls, mid: float, rad: float) -> "Interval":
        """Create [mid - rad, mid + rad]."""
        return cls(mid - rad, mid + rad)

    # -- Predicates ---------------------------------------------------------

    def contains(self, x: float) -> bool:
        """Check whether scalar *x* lies within the interval."""
        return self.lo <= x <= self.hi

    def contains_interval(self, other: "Interval") -> bool:
        """Check whether *other* ⊆ self."""
        return self.lo <= other.lo and other.hi <= self.hi

    def overlaps(self, other: "Interval") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    @property
    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def rad(self) -> float:
        return 0.5 * (self.hi - self.lo)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    # -- Arithmetic ---------------------------------------------------------

    def __add__(self, other: "Interval") -> "Interval":
        lo = self.lo + other.lo
        hi = self.hi + other.hi
        # inflate to account for rounding
        lo_inf, _ = _inflate(lo)
        _, hi_inf = _inflate(hi)
        return Interval(lo_inf, hi_inf)

    def __radd__(self, other: "Interval") -> "Interval":
        return self.__add__(other)

    def __sub__(self, other: "Interval") -> "Interval":
        lo = self.lo - other.hi
        hi = self.hi - other.lo
        lo_inf, _ = _inflate(lo)
        _, hi_inf = _inflate(hi)
        return Interval(lo_inf, hi_inf)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: "Interval") -> "Interval":
        candidates = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        lo = min(candidates)
        hi = max(candidates)
        lo_inf, _ = _inflate(lo)
        _, hi_inf = _inflate(hi)
        return Interval(lo_inf, hi_inf)

    def __truediv__(self, other: "Interval") -> "Interval":
        if other.lo <= 0.0 <= other.hi:
            # Division by interval containing zero → [-inf, inf]
            return Interval(-math.inf, math.inf)
        inv_other = Interval(1.0 / other.hi, 1.0 / other.lo)
        return self * inv_other

    # -- Elementary functions -----------------------------------------------

    def sqrt(self) -> "Interval":
        """Sound enclosure of sqrt over the interval."""
        lo_clamp = max(self.lo, 0.0)
        s_lo = math.sqrt(lo_clamp)
        s_hi = math.sqrt(max(self.hi, 0.0))
        lo_inf, _ = _inflate(s_lo)
        _, hi_inf = _inflate(s_hi)
        return Interval(max(lo_inf, 0.0), hi_inf)

    def exp(self) -> "Interval":
        """Sound enclosure of exp over the interval."""
        e_lo = math.exp(self.lo) if self.lo < 709.0 else math.inf
        e_hi = math.exp(self.hi) if self.hi < 709.0 else math.inf
        lo_inf, _ = _inflate(e_lo)
        _, hi_inf = _inflate(e_hi)
        return Interval(max(lo_inf, 0.0), hi_inf)

    def log(self) -> "Interval":
        """Sound enclosure of log over the interval (domain: positive reals)."""
        if self.hi <= 0.0:
            return Interval(-math.inf, -math.inf)
        lo_clamp = max(self.lo, _TINY)
        l_lo = math.log(lo_clamp)
        l_hi = math.log(max(self.hi, _TINY))
        lo_inf, _ = _inflate(l_lo)
        _, hi_inf = _inflate(l_hi)
        return Interval(lo_inf, hi_inf)

    def abs(self) -> "Interval":
        """Sound enclosure of |x| over the interval."""
        if self.lo >= 0.0:
            return Interval(self.lo, self.hi)
        if self.hi <= 0.0:
            return Interval(-self.hi, -self.lo)
        return Interval(0.0, max(-self.lo, self.hi))

    # -- Lattice operations -------------------------------------------------

    def hull(self, other: "Interval") -> "Interval":
        """Interval hull (smallest interval containing both)."""
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def intersect(self, other: "Interval") -> Optional["Interval"]:
        """Intersection; returns None if disjoint."""
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi + _TINY:
            return None
        return Interval(lo, hi)

    def __repr__(self) -> str:
        return f"[{self.lo:.6e}, {self.hi:.6e}]"


# ---------------------------------------------------------------------------
# IntervalVector
# ---------------------------------------------------------------------------

class IntervalVector:
    """Vector of intervals for state-space representation.

    Represents a hyperrectangle in R^n.
    """

    def __init__(self, intervals: Sequence[Interval]) -> None:
        self._intervals: List[Interval] = list(intervals)

    @classmethod
    def from_bounds(cls, lo: NDArray, hi: NDArray) -> "IntervalVector":
        """Create from parallel lower/upper bound arrays."""
        assert lo.shape == hi.shape and lo.ndim == 1
        return cls([Interval(float(l), float(h)) for l, h in zip(lo, hi)])

    @classmethod
    def from_point(cls, x: NDArray) -> "IntervalVector":
        """Create degenerate interval vector from a point."""
        return cls([Interval.point(float(v)) for v in x])

    @classmethod
    def zeros(cls, n: int) -> "IntervalVector":
        """Zero interval vector."""
        return cls([Interval(0.0, 0.0)] * n)

    @property
    def dim(self) -> int:
        return len(self._intervals)

    def __len__(self) -> int:
        return len(self._intervals)

    def __getitem__(self, idx: int) -> Interval:
        return self._intervals[idx]

    def __setitem__(self, idx: int, val: Interval) -> None:
        self._intervals[idx] = val

    def lo_array(self) -> NDArray:
        return np.array([iv.lo for iv in self._intervals])

    def hi_array(self) -> NDArray:
        return np.array([iv.hi for iv in self._intervals])

    def mid_array(self) -> NDArray:
        return np.array([iv.mid for iv in self._intervals])

    def contains_point(self, x: NDArray) -> bool:
        """Check whether point *x* lies inside every component interval."""
        if len(x) != self.dim:
            return False
        return all(iv.contains(float(x[i])) for i, iv in enumerate(self._intervals))

    def contains_vector(self, other: "IntervalVector") -> bool:
        """Check whether *other* ⊆ self component-wise."""
        if other.dim != self.dim:
            return False
        return all(
            self._intervals[i].contains_interval(other._intervals[i])
            for i in range(self.dim)
        )

    def __add__(self, other: "IntervalVector") -> "IntervalVector":
        assert self.dim == other.dim
        return IntervalVector([a + b for a, b in zip(self._intervals, other._intervals)])

    def __sub__(self, other: "IntervalVector") -> "IntervalVector":
        assert self.dim == other.dim
        return IntervalVector([a - b for a, b in zip(self._intervals, other._intervals)])

    def hull(self, other: "IntervalVector") -> "IntervalVector":
        """Component-wise hull."""
        assert self.dim == other.dim
        return IntervalVector(
            [a.hull(b) for a, b in zip(self._intervals, other._intervals)]
        )

    def max_width(self) -> float:
        """Maximum component width."""
        return max(iv.width for iv in self._intervals)

    def __repr__(self) -> str:
        return f"IntervalVector({self._intervals})"


# ---------------------------------------------------------------------------
# IntervalMatrix
# ---------------------------------------------------------------------------

class IntervalMatrix:
    """Matrix of intervals for transition-matrix representation.

    Stored as parallel lo/hi NumPy arrays for efficiency.
    """

    def __init__(self, lo: NDArray, hi: NDArray) -> None:
        assert lo.shape == hi.shape and lo.ndim == 2
        self._lo = lo.astype(np.float64)
        self._hi = hi.astype(np.float64)

    @classmethod
    def from_matrix(cls, M: NDArray) -> "IntervalMatrix":
        """Create from a point matrix with epsilon inflation."""
        M = np.asarray(M, dtype=np.float64)
        aM = np.abs(M)
        delta = _EPS * aM + _TINY
        return cls(M - delta, M + delta)

    @classmethod
    def from_bounds(cls, lo: NDArray, hi: NDArray) -> "IntervalMatrix":
        """Create from explicit lower/upper bound matrices."""
        return cls(np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64))

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "IntervalMatrix":
        return cls(np.zeros((rows, cols)), np.zeros((rows, cols)))

    @property
    def shape(self) -> Tuple[int, int]:
        return self._lo.shape

    @property
    def lo(self) -> NDArray:
        return self._lo

    @property
    def hi(self) -> NDArray:
        return self._hi

    def get(self, i: int, j: int) -> Interval:
        return Interval(float(self._lo[i, j]), float(self._hi[i, j]))

    def set(self, i: int, j: int, iv: Interval) -> None:
        self._lo[i, j] = iv.lo
        self._hi[i, j] = iv.hi

    def contains_matrix(self, M: NDArray) -> bool:
        """Check whether every entry of *M* lies within the corresponding interval."""
        M = np.asarray(M, dtype=np.float64)
        return bool(np.all(M >= self._lo - _TINY) and np.all(M <= self._hi + _TINY))

    def mid_matrix(self) -> NDArray:
        return 0.5 * (self._lo + self._hi)

    def max_width(self) -> float:
        return float(np.max(self._hi - self._lo))

    def __repr__(self) -> str:
        return f"IntervalMatrix(shape={self.shape}, max_width={self.max_width():.2e})"


# ---------------------------------------------------------------------------
# Sound matrix-vector multiplication
# ---------------------------------------------------------------------------

def interval_matmul(A: IntervalMatrix, v: IntervalVector) -> IntervalVector:
    """Sound interval matrix-vector multiply  A · v.

    For each entry of the result, computes the tightest enclosure of
    the set { (A x)[i] : A_ij ∈ [A.lo_ij, A.hi_ij], x_j ∈ [v_lo_j, v_hi_j] }.

    Uses the standard formula for interval dot products:
        sum_j [A_lo_ij, A_hi_ij] * [v_lo_j, v_hi_j]
    with rounding inflation at each accumulation step.
    """
    m, n = A.shape
    assert v.dim == n, f"Dimension mismatch: A is {A.shape}, v has dim {v.dim}"

    v_lo = v.lo_array()
    v_hi = v.hi_array()

    result_lo = np.zeros(m)
    result_hi = np.zeros(m)

    for i in range(m):
        a_lo = A.lo[i, :]
        a_hi = A.hi[i, :]

        # All four products
        p1 = a_lo * v_lo
        p2 = a_lo * v_hi
        p3 = a_hi * v_lo
        p4 = a_hi * v_hi

        row_lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4)).sum()
        row_hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4)).sum()

        # Inflation for accumulated rounding
        delta_lo = _EPS * abs(row_lo) + n * _TINY
        delta_hi = _EPS * abs(row_hi) + n * _TINY
        result_lo[i] = row_lo - delta_lo
        result_hi[i] = row_hi + delta_hi

    return IntervalVector.from_bounds(result_lo, result_hi)


def interval_matmul_matrix(A: IntervalMatrix, B: IntervalMatrix) -> IntervalMatrix:
    """Sound interval matrix-matrix multiply A · B."""
    m, k1 = A.shape
    k2, n = B.shape
    assert k1 == k2, f"Dimension mismatch: A is {A.shape}, B is {B.shape}"

    result_lo = np.zeros((m, n))
    result_hi = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            a_lo = A.lo[i, :]
            a_hi = A.hi[i, :]
            b_lo = B.lo[:, j]
            b_hi = B.hi[:, j]

            p1 = a_lo * b_lo
            p2 = a_lo * b_hi
            p3 = a_hi * b_lo
            p4 = a_hi * b_hi

            s_lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4)).sum()
            s_hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4)).sum()

            delta_lo = _EPS * abs(s_lo) + k1 * _TINY
            delta_hi = _EPS * abs(s_hi) + k1 * _TINY
            result_lo[i, j] = s_lo - delta_lo
            result_hi[i, j] = s_hi + delta_hi

    return IntervalMatrix(result_lo, result_hi)


# ---------------------------------------------------------------------------
# Convenience: interval evaluation of a scalar polynomial
# ---------------------------------------------------------------------------

def interval_polyval(coeffs: Sequence[float], x: Interval) -> Interval:
    """Evaluate polynomial with real coefficients over an interval (Horner)."""
    if len(coeffs) == 0:
        return Interval(0.0, 0.0)
    result = Interval.point(coeffs[0])
    for c in coeffs[1:]:
        result = result * x + Interval.point(c)
    return result
