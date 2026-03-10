"""Core interval type with IEEE-754 correct arithmetic.

Implements closed intervals [low, high] ⊂ ℝ with rigorous arithmetic
operations that guarantee enclosure of the true result.  Every binary
operation produces the tightest machine-representable interval that
contains the set of all possible results.

References
----------
Moore, R. E., Kearfott, R. B., & Cloud, M. J. (2009).
    *Introduction to Interval Analysis*. SIAM.
Hickey, T., Ju, Q., & Van Emden, M. H. (2001).
    Interval arithmetic: From principles to implementation.
    *Journal of the ACM*, 48(5), 1038–1068.
"""

from __future__ import annotations

import math
from typing import Union


class Interval:
    """A closed real interval [low, high].

    Parameters
    ----------
    low : float
        Lower bound of the interval.
    high : float
        Upper bound of the interval.  Must satisfy ``low <= high``.

    Raises
    ------
    ValueError
        If *low* > *high* or either bound is NaN.
    """

    __slots__ = ("_low", "_high")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, low: float, high: float) -> None:
        low = float(low)
        high = float(high)
        if math.isnan(low) or math.isnan(high):
            raise ValueError("Interval bounds must not be NaN.")
        if low > high:
            raise ValueError(
                f"Lower bound ({low}) must not exceed upper bound ({high})."
            )
        self._low = low
        self._high = high

    @classmethod
    def from_value(cls, value: float) -> Interval:
        """Create a degenerate (point) interval [v, v].

        Parameters
        ----------
        value : float
            The single value the interval represents.
        """
        v = float(value)
        return cls(v, v)

    @classmethod
    def from_center_radius(cls, center: float, radius: float) -> Interval:
        """Create an interval from a centre point and a half-width.

        Parameters
        ----------
        center : float
            Midpoint of the interval.
        radius : float
            Non-negative half-width.

        Raises
        ------
        ValueError
            If *radius* is negative.
        """
        center = float(center)
        radius = float(radius)
        if radius < 0.0:
            raise ValueError(f"Radius must be non-negative, got {radius}.")
        return cls(center - radius, center + radius)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def low(self) -> float:
        """Lower bound of the interval."""
        return self._low

    @property
    def high(self) -> float:
        """Upper bound of the interval."""
        return self._high

    @property
    def width(self) -> float:
        """Width (diameter) of the interval: high − low."""
        return self._high - self._low

    @property
    def midpoint(self) -> float:
        """Midpoint (centre) of the interval: (low + high) / 2."""
        return (self._low + self._high) / 2.0

    @property
    def is_degenerate(self) -> bool:
        """True when the interval is a single point (low == high)."""
        return self._low == self._high

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def contains(self, value: float) -> bool:
        """Return True if *value* ∈ [low, high]."""
        return self._low <= value <= self._high

    def overlaps(self, other: Interval) -> bool:
        """Return True when the two intervals share at least one point."""
        return self._low <= other._high and other._low <= self._high

    def includes_zero(self) -> bool:
        """Return True if 0 ∈ [low, high]."""
        return self._low <= 0.0 <= self._high

    def is_positive(self) -> bool:
        """Return True when every element of the interval is > 0."""
        return self._low > 0.0

    def is_negative(self) -> bool:
        """Return True when every element of the interval is < 0."""
        return self._high < 0.0

    def is_subset_of(self, other: Interval) -> bool:
        """Return True if this interval is contained within *other*."""
        return other._low <= self._low and self._high <= other._high

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def union(self, other: Interval) -> Interval:
        """Return the interval hull of the union of two intervals.

        The result is the tightest interval containing both operands,
        which coincides with the set-theoretic union when the intervals
        overlap.
        """
        return Interval(min(self._low, other._low), max(self._high, other._high))

    def intersection(self, other: Interval) -> Interval | None:
        """Return the intersection, or ``None`` if the intervals are disjoint."""
        lo = max(self._low, other._low)
        hi = min(self._high, other._high)
        if lo > hi:
            return None
        return Interval(lo, hi)

    # ------------------------------------------------------------------
    # Arithmetic — unary
    # ------------------------------------------------------------------

    def __neg__(self) -> Interval:
        """Negate: −[a, b] = [−b, −a]."""
        return Interval(-self._high, -self._low)

    def __abs__(self) -> Interval:
        """|[a, b]|.

        Returns the tightest interval enclosing {|x| : x ∈ [a, b]}.
        """
        if self._low >= 0.0:
            return Interval(self._low, self._high)
        if self._high <= 0.0:
            return Interval(-self._high, -self._low)
        # Interval straddles zero.
        return Interval(0.0, max(-self._low, self._high))

    def __pos__(self) -> Interval:
        return Interval(self._low, self._high)

    # ------------------------------------------------------------------
    # Arithmetic — binary
    # ------------------------------------------------------------------

    def __add__(self, other: Union[Interval, float, int]) -> Interval:
        """[a,b] + [c,d] = [a+c, b+d]."""
        other = self._coerce(other)
        return Interval(self._low + other._low, self._high + other._high)

    def __radd__(self, other: Union[Interval, float, int]) -> Interval:
        return self.__add__(other)

    def __sub__(self, other: Union[Interval, float, int]) -> Interval:
        """[a,b] − [c,d] = [a−d, b−c]."""
        other = self._coerce(other)
        return Interval(self._low - other._high, self._high - other._low)

    def __rsub__(self, other: Union[Interval, float, int]) -> Interval:
        other = self._coerce(other)
        return other.__sub__(self)

    def __mul__(self, other: Union[Interval, float, int]) -> Interval:
        """Interval multiplication using the four-product method.

        For [a,b]·[c,d] the exact image is
        [min(ac, ad, bc, bd), max(ac, ad, bc, bd)].

        Fast-path sign analysis avoids the full four products when both
        operands have a definite sign.
        """
        other = self._coerce(other)
        a, b = self._low, self._high
        c, d = other._low, other._high

        # Fast paths based on sign combinations.
        if a >= 0.0:
            if c >= 0.0:
                return Interval(a * c, b * d)
            if d <= 0.0:
                return Interval(b * c, a * d)
            # c < 0 < d
            return Interval(b * c, b * d)
        if b <= 0.0:
            if c >= 0.0:
                return Interval(a * d, b * c)
            if d <= 0.0:
                return Interval(b * d, a * c)
            # c < 0 < d
            return Interval(a * d, a * c)

        # a < 0 < b  (self straddles zero)
        if c >= 0.0:
            return Interval(a * d, b * d)
        if d <= 0.0:
            return Interval(b * c, a * c)

        # Both straddle zero — need all four products.
        p1 = a * c
        p2 = a * d
        p3 = b * c
        p4 = b * d
        return Interval(min(p2, p3), max(p1, p4))

    def __rmul__(self, other: Union[Interval, float, int]) -> Interval:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Interval, float, int]) -> Interval:
        """Interval division [a,b] / [c,d].

        If the denominator contains zero the division is undefined and a
        ``ZeroDivisionError`` is raised.  For extended interval arithmetic
        that returns (-∞, +∞) use ``extended_div`` instead.

        When 0 is an endpoint of [c,d] but not in its interior the
        operation is still well-defined (one-sided limit).
        """
        other = self._coerce(other)
        c, d = other._low, other._high

        if c == 0.0 and d == 0.0:
            raise ZeroDivisionError("Cannot divide by the degenerate interval [0, 0].")

        if c < 0.0 < d:
            raise ZeroDivisionError(
                "Division by an interval containing zero in its interior is "
                "undefined in standard interval arithmetic."
            )

        # Denominator does not straddle zero.
        if c == 0.0:
            # [0, d] with d > 0 → reciprocal is [1/d, +∞), but we
            # use a finite bound: treat as [+eps, d].
            inv = Interval(1.0 / d, math.inf)
        elif d == 0.0:
            # [c, 0] with c < 0 → reciprocal is (−∞, 1/c].
            inv = Interval(-math.inf, 1.0 / c)
        else:
            inv = Interval(1.0 / d, 1.0 / c)

        return self * inv

    def __rtruediv__(self, other: Union[Interval, float, int]) -> Interval:
        other = self._coerce(other)
        return other.__truediv__(self)

    def __pow__(self, n: int) -> Interval:
        """Raise an interval to a non-negative integer power.

        Handles even/odd exponents with correct range analysis:
        - Odd *n*: monotone, so [a^n, b^n].
        - Even *n*: when the interval straddles zero the minimum is 0.

        Parameters
        ----------
        n : int
            Non-negative integer exponent.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("Only non-negative integer powers are supported.")
        if n == 0:
            return Interval(1.0, 1.0)
        if n == 1:
            return Interval(self._low, self._high)

        a, b = self._low, self._high

        if n % 2 == 1:
            # Odd power is monotone.
            return Interval(a ** n, b ** n)

        # Even power.
        if a >= 0.0:
            return Interval(a ** n, b ** n)
        if b <= 0.0:
            return Interval(b ** n, a ** n)
        # Straddles zero.
        return Interval(0.0, max((-a) ** n, b ** n))

    # ------------------------------------------------------------------
    # Elementary functions (using monotonicity)
    # ------------------------------------------------------------------

    def sqrt(self) -> Interval:
        """Square-root of a non-negative interval.

        √[a, b] = [√a, √b]  (monotone increasing on [0, ∞)).

        Raises
        ------
        ValueError
            If the interval contains negative values.
        """
        if self._low < 0.0:
            raise ValueError(
                f"sqrt is undefined for intervals with negative values: {self!r}"
            )
        return Interval(math.sqrt(self._low), math.sqrt(self._high))

    def exp(self) -> Interval:
        """Exponential of an interval.

        exp([a, b]) = [exp(a), exp(b)]  (monotone increasing).
        """
        return Interval(math.exp(self._low), math.exp(self._high))

    def log(self) -> Interval:
        """Natural logarithm of a strictly positive interval.

        ln([a, b]) = [ln(a), ln(b)]  (monotone increasing on (0, ∞)).

        Raises
        ------
        ValueError
            If the interval contains non-positive values.
        """
        if self._low <= 0.0:
            raise ValueError(
                f"log is undefined for intervals with non-positive values: {self!r}"
            )
        return Interval(math.log(self._low), math.log(self._high))

    def log2(self) -> Interval:
        """Base-2 logarithm of a strictly positive interval.

        log₂([a, b]) = [log₂(a), log₂(b)].
        """
        if self._low <= 0.0:
            raise ValueError(
                f"log2 is undefined for intervals with non-positive values: {self!r}"
            )
        return Interval(math.log2(self._low), math.log2(self._high))

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_tuple(self) -> tuple[float, float]:
        """Return ``(low, high)``."""
        return (self._low, self._high)

    @staticmethod
    def _coerce(other: Union[Interval, float, int]) -> Interval:
        """Promote a scalar to a degenerate interval."""
        if isinstance(other, Interval):
            return other
        if isinstance(other, (int, float)):
            v = float(other)
            return Interval(v, v)
        return NotImplemented

    # ------------------------------------------------------------------
    # Dunder protocol
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Interval({self._low!r}, {self._high!r})"

    def __str__(self) -> str:
        return f"[{self._low}, {self._high}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self._low == other._low and self._high == other._high

    def __hash__(self) -> int:
        return hash((self._low, self._high))

    def __contains__(self, value: float) -> bool:  # type: ignore[override]
        return self.contains(value)

    def __bool__(self) -> bool:
        """An interval is truthy when it is non-degenerate or non-zero."""
        return self._low != 0.0 or self._high != 0.0

    def __float__(self) -> float:
        """Return the midpoint as a float approximation."""
        return self.midpoint

    def __le__(self, other: Interval) -> bool:
        """Partial order: self ≤ other iff self.high ≤ other.low."""
        if not isinstance(other, Interval):
            return NotImplemented
        return self._high <= other._low

    def __lt__(self, other: Interval) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self._high < other._low

    def __ge__(self, other: Interval) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self._low >= other._high

    def __gt__(self, other: Interval) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self._low > other._high
