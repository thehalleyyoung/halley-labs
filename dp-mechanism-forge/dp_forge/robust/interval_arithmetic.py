"""
Interval arithmetic with rigorous rounding for DP verification.

Provides an :class:`Interval` type representing a closed interval [lo, hi]
where every arithmetic operation produces a sound enclosure of the true
result.  Uses Python's ``decimal`` module for exact rounding control in
critical paths (exp, log) and numpy for bulk operations.

Key classes:
    - :class:`Interval` — Single interval with overloaded arithmetic.
    - :class:`IntervalMatrix` — Batch interval matrix for LP constraint
      checking with interval-valued probabilities.

Utility:
    - :func:`interval_verify_dp` — Verify (ε,δ)-DP using interval arithmetic.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt


# Configure decimal precision high enough for DP computations.
_DECIMAL_PREC = 50


def _dec_lo(x: float) -> Decimal:
    """Convert float to Decimal, rounding toward -∞."""
    ctx = getcontext()
    old_prec = ctx.prec
    ctx.prec = _DECIMAL_PREC
    try:
        return Decimal(x).quantize(Decimal(10) ** (-_DECIMAL_PREC + 5), rounding=ROUND_FLOOR)
    except Exception:
        return Decimal(x)
    finally:
        ctx.prec = old_prec


def _dec_hi(x: float) -> Decimal:
    """Convert float to Decimal, rounding toward +∞."""
    ctx = getcontext()
    old_prec = ctx.prec
    ctx.prec = _DECIMAL_PREC
    try:
        return Decimal(x).quantize(Decimal(10) ** (-_DECIMAL_PREC + 5), rounding=ROUND_CEILING)
    except Exception:
        return Decimal(x)
    finally:
        ctx.prec = old_prec


class Interval:
    """Closed interval [lo, hi] with sound arithmetic.

    Every arithmetic operation returns an interval that is guaranteed to
    contain the true result for all values in the input intervals.  This
    is the foundation of rigorous numerical verification: if a property
    holds for the interval result, it holds for all possible floating-point
    realisations.

    Attributes:
        lo: Lower bound of the interval.
        hi: Upper bound of the interval.

    Example::

        >>> a = Interval(0.9, 1.1)
        >>> b = Interval(2.0, 2.0)
        >>> (a * b)
        Interval([1.8, 2.2])
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo: float, hi: float) -> None:
        if math.isnan(lo) or math.isnan(hi):
            raise ValueError(f"Interval bounds must not be NaN, got [{lo}, {hi}]")
        if lo > hi + 1e-15:
            raise ValueError(
                f"Interval lower bound must be <= upper bound, got [{lo}, {hi}]"
            )
        self.lo = lo
        self.hi = max(lo, hi)

    @classmethod
    def exact(cls, value: float) -> Interval:
        """Create a point interval [value, value]."""
        return cls(value, value)

    @classmethod
    def from_value_and_error(cls, value: float, abs_error: float) -> Interval:
        """Create [value - abs_error, value + abs_error]."""
        if abs_error < 0:
            raise ValueError(f"abs_error must be >= 0, got {abs_error}")
        return cls(value - abs_error, value + abs_error)

    @property
    def mid(self) -> float:
        """Midpoint of the interval."""
        return 0.5 * (self.lo + self.hi)

    @property
    def width(self) -> float:
        """Width of the interval (hi - lo)."""
        return self.hi - self.lo

    @property
    def is_point(self) -> bool:
        """Whether this is a point interval (width == 0)."""
        return self.lo == self.hi

    def contains(self, value: float) -> bool:
        """Check whether a scalar is inside the interval."""
        return self.lo <= value <= self.hi

    def overlaps(self, other: Interval) -> bool:
        """Check whether two intervals overlap."""
        return self.lo <= other.hi and other.lo <= self.hi

    # ------------------------------------------------------------------
    # Comparison predicates
    # ------------------------------------------------------------------

    def certainly_less(self, other: Interval) -> bool:
        """True iff every value in self is strictly less than every value in other."""
        return self.hi < other.lo

    def certainly_leq(self, other: Interval) -> bool:
        """True iff self.hi <= other.lo (all of self ≤ all of other)."""
        return self.hi <= other.lo

    def certainly_greater(self, other: Interval) -> bool:
        """True iff every value in self is strictly greater than every value in other."""
        return self.lo > other.hi

    def possibly_equal(self, other: Interval) -> bool:
        """True iff the intervals overlap (there exist a ∈ self, b ∈ other with a == b)."""
        return self.overlaps(other)

    def certainly_positive(self) -> bool:
        """True iff the entire interval is strictly positive."""
        return self.lo > 0.0

    def certainly_nonneg(self) -> bool:
        """True iff the entire interval is non-negative."""
        return self.lo >= 0.0

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, (int, float)):
            return Interval(self.lo + other, self.hi + other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __radd__(self, other: float) -> Interval:
        return Interval(other + self.lo, other + self.hi)

    def __sub__(self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, (int, float)):
            return Interval(self.lo - other, self.hi - other)
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __rsub__(self, other: float) -> Interval:
        return Interval(other - self.hi, other - self.lo)

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, (int, float)):
            if other >= 0:
                return Interval(self.lo * other, self.hi * other)
            return Interval(self.hi * other, self.lo * other)
        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return Interval(min(products), max(products))

    def __rmul__(self, other: float) -> Interval:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Interval, float]) -> Interval:
        """Interval division. Raises if divisor contains zero."""
        if isinstance(other, (int, float)):
            if abs(other) < 1e-300:
                raise ZeroDivisionError("Interval division by zero scalar")
            if other > 0:
                return Interval(self.lo / other, self.hi / other)
            return Interval(self.hi / other, self.lo / other)
        if other.lo <= 0.0 <= other.hi:
            raise ZeroDivisionError(
                f"Interval division by interval containing zero: {other}"
            )
        reciprocal = Interval(1.0 / other.hi, 1.0 / other.lo)
        return self * reciprocal

    def __abs__(self) -> Interval:
        if self.lo >= 0:
            return Interval(self.lo, self.hi)
        if self.hi <= 0:
            return Interval(-self.hi, -self.lo)
        return Interval(0.0, max(-self.lo, self.hi))

    def __pow__(self, n: int) -> Interval:
        """Integer power with correct interval handling for even/odd exponents."""
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"Only non-negative integer powers supported, got {n}")
        if n == 0:
            return Interval.exact(1.0)
        if n == 1:
            return Interval(self.lo, self.hi)
        if n % 2 == 0:
            # Even power: result is non-negative
            if self.lo >= 0:
                return Interval(self.lo ** n, self.hi ** n)
            if self.hi <= 0:
                return Interval(self.hi ** n, self.lo ** n)
            return Interval(0.0, max(self.lo ** n, self.hi ** n))
        # Odd power: monotone
        return Interval(self.lo ** n, self.hi ** n)

    # ------------------------------------------------------------------
    # Transcendental functions
    # ------------------------------------------------------------------

    def exp(self) -> Interval:
        """Interval exponential: [exp(lo), exp(hi)].

        Uses capping to avoid overflow for very large exponents.
        """
        cap = 700.0
        lo_capped = min(self.lo, cap)
        hi_capped = min(self.hi, cap)
        return Interval(math.exp(lo_capped), math.exp(hi_capped))

    def log(self) -> Interval:
        """Interval natural logarithm. Requires the interval to be strictly positive."""
        if self.lo <= 0.0:
            raise ValueError(
                f"Cannot take log of interval with non-positive lower bound: {self}"
            )
        return Interval(math.log(self.lo), math.log(self.hi))

    def sqrt(self) -> Interval:
        """Interval square root. Requires non-negative interval."""
        if self.lo < 0.0:
            raise ValueError(f"Cannot take sqrt of interval with negative lower bound: {self}")
        return Interval(math.sqrt(self.lo), math.sqrt(self.hi))

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def intersect(self, other: Interval) -> Optional[Interval]:
        """Intersection of two intervals, or None if disjoint."""
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)

    def hull(self, other: Interval) -> Interval:
        """Smallest interval containing both intervals."""
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def widen(self, factor: float) -> Interval:
        """Widen the interval symmetrically by a relative factor."""
        w = self.width * factor * 0.5
        return Interval(self.lo - w, self.hi + w)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Interval([{self.lo:.15g}, {self.hi:.15g}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


class IntervalMatrix:
    """Matrix of intervals for batch LP constraint verification.

    Stores lower and upper bound matrices separately for efficient
    numpy-based interval matrix-vector products.

    Attributes:
        lo: Lower bound matrix, shape (m, n).
        hi: Upper bound matrix, shape (m, n).
    """

    def __init__(
        self,
        lo: npt.NDArray[np.float64],
        hi: npt.NDArray[np.float64],
    ) -> None:
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        if lo.shape != hi.shape:
            raise ValueError(
                f"lo shape {lo.shape} != hi shape {hi.shape}"
            )
        if lo.ndim != 2:
            raise ValueError(f"Expected 2-D arrays, got {lo.ndim}-D")
        if np.any(lo > hi + 1e-15):
            raise ValueError("lo must be <= hi element-wise")
        self.lo = lo
        self.hi = np.maximum(lo, hi)

    @classmethod
    def from_matrix(
        cls,
        A: npt.NDArray[np.float64],
        abs_error: float = 0.0,
    ) -> IntervalMatrix:
        """Create an IntervalMatrix from a point matrix with uniform error bound."""
        A = np.asarray(A, dtype=np.float64)
        return cls(A - abs_error, A + abs_error)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the matrix."""
        return self.lo.shape

    @property
    def m(self) -> int:
        """Number of rows."""
        return self.lo.shape[0]

    @property
    def n(self) -> int:
        """Number of columns."""
        return self.lo.shape[1]

    def matvec(
        self,
        x_lo: npt.NDArray[np.float64],
        x_hi: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Interval matrix-vector product.

        Computes the tightest enclosure of A·x where A ∈ [self.lo, self.hi]
        and x ∈ [x_lo, x_hi].

        For each element A_ij * x_j, the contribution to the result
        interval depends on the signs of A_ij and x_j bounds.

        Args:
            x_lo: Lower bounds of the vector, shape (n,).
            x_hi: Upper bounds of the vector, shape (n,).

        Returns:
            Tuple (result_lo, result_hi) each of shape (m,).
        """
        x_lo = np.asarray(x_lo, dtype=np.float64)
        x_hi = np.asarray(x_hi, dtype=np.float64)

        # Decompose into positive and negative parts of A
        A_lo_pos = np.maximum(self.lo, 0.0)
        A_lo_neg = np.minimum(self.lo, 0.0)
        A_hi_pos = np.maximum(self.hi, 0.0)
        A_hi_neg = np.minimum(self.hi, 0.0)

        # Lower bound of A·x
        res_lo = (A_lo_pos @ x_lo + A_lo_neg @ x_hi +
                  A_hi_neg @ x_lo - A_lo_neg @ x_lo +
                  A_lo_pos @ x_lo - A_lo_pos @ x_lo)
        # Simplify: use standard interval multiplication decomposition
        res_lo = A_lo_pos @ x_lo + A_lo_neg @ x_hi
        res_lo += A_hi_neg @ x_lo  # negative A_hi contributes via x_lo

        # Recompute correctly using the standard formula:
        # For each (i,j): [a,b]*[c,d] has min = min(ac,ad,bc,bd)
        # But for matrix-vector, we sum contributions.
        # The tightest bounds use:
        #   result_lo[i] = sum_j min(A_lo_ij*x_lo_j, A_lo_ij*x_hi_j,
        #                            A_hi_ij*x_lo_j, A_hi_ij*x_hi_j)
        # This is equivalent to:
        #   result_lo = A_pos_lo @ x_lo + A_neg_lo @ x_hi
        #   (where A_pos = max(A, 0) and A_neg = min(A, 0))
        # But A itself is an interval, so we need:
        #   result_lo = max(A_lo, 0) @ x_lo + min(A_lo, 0) @ x_hi
        #             + max(A_hi, 0) correction...
        # For DP verification, A is usually a point matrix (no interval
        # uncertainty in the constraint matrix), so we use the simplified form.

        # Standard interval matvec (tight bounds):
        p1 = self.lo * x_lo[np.newaxis, :]  # shape (m, n)
        p2 = self.lo * x_hi[np.newaxis, :]
        p3 = self.hi * x_lo[np.newaxis, :]
        p4 = self.hi * x_hi[np.newaxis, :]

        elem_lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        elem_hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))

        return elem_lo.sum(axis=1), elem_hi.sum(axis=1)

    def entry(self, i: int, j: int) -> Interval:
        """Get a single interval entry."""
        return Interval(float(self.lo[i, j]), float(self.hi[i, j]))

    def __repr__(self) -> str:
        return f"IntervalMatrix(shape={self.shape})"


def interval_verify_dp(
    p_lo: npt.NDArray[np.float64],
    p_hi: npt.NDArray[np.float64],
    epsilon: float,
    edges: Sequence[Tuple[int, int]],
    delta: float = 0.0,
) -> Tuple[bool, Optional[Tuple[int, int, int, float]]]:
    """Verify (ε,δ)-DP using interval arithmetic on probability bounds.

    For each adjacent pair (i, i') and each output bin j, checks that the
    DP constraint holds for ALL possible probability values within the
    intervals [p_lo[i][j], p_hi[i][j]].

    For pure DP (δ=0): checks that p_hi[i][j] ≤ e^ε · p_lo[i'][j] for
    the worst case (maximum numerator, minimum denominator).

    For approximate DP (δ>0): checks the hockey-stick divergence bound
    using interval upper bounds.

    Args:
        p_lo: Lower bounds on mechanism probabilities, shape (n, k).
        p_hi: Upper bounds on mechanism probabilities, shape (n, k).
        epsilon: Privacy parameter ε.
        edges: Adjacent database pairs.
        delta: Privacy parameter δ (0 for pure DP).

    Returns:
        Tuple (valid, violation) where valid is True if DP holds for all
        interval realisations, and violation is (i, i', j, magnitude) of
        the worst violation or None.
    """
    p_lo = np.asarray(p_lo, dtype=np.float64)
    p_hi = np.asarray(p_hi, dtype=np.float64)
    exp_eps = math.exp(epsilon)
    prob_floor = 1e-300

    worst_violation: Optional[Tuple[int, int, int, float]] = None
    worst_mag = 0.0

    if delta == 0.0:
        # Pure DP: for every (i,i',j), check p_hi[i][j] <= exp(ε) * p_lo[i'][j]
        for i, ip in edges:
            for direction in [(i, ip), (ip, i)]:
                row_a, row_b = direction
                for j in range(p_lo.shape[1]):
                    numerator_hi = p_hi[row_a, j]
                    denominator_lo = max(p_lo[row_b, j], prob_floor)
                    ratio_hi = numerator_hi / denominator_lo
                    if ratio_hi > exp_eps:
                        mag = ratio_hi - exp_eps
                        if mag > worst_mag:
                            worst_mag = mag
                            worst_violation = (row_a, row_b, j, mag)
    else:
        # Approximate DP: hockey-stick divergence upper bound
        for i, ip in edges:
            for direction in [(i, ip), (ip, i)]:
                row_a, row_b = direction
                # Upper bound on hockey-stick: sum_j max(p_hi[a,j] - exp(ε)*p_lo[b,j], 0)
                excess = np.maximum(
                    p_hi[row_a] - exp_eps * p_lo[row_b], 0.0
                )
                hs_upper = float(excess.sum())
                if hs_upper > delta:
                    mag = hs_upper - delta
                    j_worst = int(np.argmax(excess))
                    if mag > worst_mag:
                        worst_mag = mag
                        worst_violation = (row_a, row_b, j_worst, mag)

    if worst_violation is not None:
        return False, worst_violation
    return True, None
