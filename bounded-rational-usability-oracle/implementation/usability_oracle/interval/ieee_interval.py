"""IEEE 1788-2015 compliant interval arithmetic.

Provides an :class:`IEEEInterval` type that conforms to the IEEE
Standard for Interval Arithmetic (IEEE 1788-2015).  Key additions over
the base :class:`~usability_oracle.interval.interval.Interval`:

* **Directed rounding** — ``add_rd``/``add_ru``, ``mul_rd``/``mul_ru``,
  ``div_rd``/``div_ru`` perform arithmetic with directed rounding
  toward −∞ / +∞ using platform facilities when available.
* **Decoration system** — each interval carries a *decoration* from the
  lattice ``com > dac > def > trv > ill`` that tracks constraint-
  satisfaction properties through the computation.
* **Empty interval** — a distinguished empty set ∅ (not representable
  in the base ``Interval`` type).
* **Entire interval** — [−∞, +∞].
* **Set predicates** — ``is_empty``, ``is_entire``, ``is_singleton``.

Where platform-level directed rounding is unavailable (most CPython
builds), the module falls back to a conservative software emulation
using ``math.nextafter``.

References
----------
IEEE Std 1788-2015 — IEEE Standard for Interval Arithmetic.
Revol, N., & Rouillier, F. (2005).
    Motivations for an arbitrary precision interval arithmetic and the
    MPFI library. *Reliable Computing*, 11(4), 275–290.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Union

from usability_oracle.interval.interval import Interval


# ---------------------------------------------------------------------------
# Decoration lattice
# ---------------------------------------------------------------------------

@unique
class Decoration(Enum):
    """IEEE 1788-2015 interval decorations (§4.3).

    The decorations form a lattice:  com > dac > def > trv > ill.

    * **com** (common) — the function is continuous and the interval
      is bounded and non-empty.
    * **dac** (defined and continuous) — the function is continuous on
      the interval but the result may be unbounded.
    * **def** (defined) — the function is defined on the interval but
      may not be continuous.
    * **trv** (trivial) — no information beyond non-emptiness.
    * **ill** (ill-formed) — the result is meaningless (e.g. NaN input).
    """

    COM = "com"
    DAC = "dac"
    DEF = "def"
    TRV = "trv"
    ILL = "ill"


_DECORATION_ORDER = {
    Decoration.COM: 4,
    Decoration.DAC: 3,
    Decoration.DEF: 2,
    Decoration.TRV: 1,
    Decoration.ILL: 0,
}


def _meet_decoration(a: Decoration, b: Decoration) -> Decoration:
    """Return the lattice meet (greatest lower bound) of two decorations."""
    oa = _DECORATION_ORDER[a]
    ob = _DECORATION_ORDER[b]
    target = min(oa, ob)
    for dec, order in _DECORATION_ORDER.items():
        if order == target:
            return dec
    return Decoration.ILL  # pragma: no cover


# ---------------------------------------------------------------------------
# Directed rounding helpers
# ---------------------------------------------------------------------------

def _round_down(value: float) -> float:
    """Return *value* rounded toward −∞.

    On platforms without hardware rounding-mode control we use
    ``math.nextafter`` toward −∞ as a conservative bound.
    """
    if math.isinf(value) or math.isnan(value):
        return value
    return math.nextafter(value, -math.inf)


def _round_up(value: float) -> float:
    """Return *value* rounded toward +∞."""
    if math.isinf(value) or math.isnan(value):
        return value
    return math.nextafter(value, math.inf)


# ---------------------------------------------------------------------------
# IEEEInterval
# ---------------------------------------------------------------------------

_EMPTY_SENTINEL = float("nan")


@dataclass(slots=True)
class IEEEInterval:
    """An IEEE 1788-2015 decorated interval.

    Parameters
    ----------
    low : float
        Lower bound (−∞ allowed).
    high : float
        Upper bound (+∞ allowed).
    decoration : Decoration
        Decoration tracking constraint-satisfaction status.
    empty : bool
        If True the interval is the empty set ∅ (bounds are ignored).
    """

    low: float
    high: float
    decoration: Decoration = Decoration.COM
    empty: bool = False

    def __post_init__(self) -> None:
        if self.empty:
            self.low = _EMPTY_SENTINEL
            self.high = _EMPTY_SENTINEL
            self.decoration = Decoration.TRV
            return
        if math.isnan(self.low) or math.isnan(self.high):
            self.empty = True
            self.low = _EMPTY_SENTINEL
            self.high = _EMPTY_SENTINEL
            self.decoration = Decoration.ILL
            return
        if self.low > self.high:
            raise ValueError(
                f"Lower bound ({self.low}) must not exceed upper bound ({self.high})."
            )
        # Update decoration based on boundedness
        if math.isinf(self.low) or math.isinf(self.high):
            if _DECORATION_ORDER[self.decoration] > _DECORATION_ORDER[Decoration.DAC]:
                self.decoration = Decoration.DAC

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def make_empty(cls) -> IEEEInterval:
        """Create the empty interval ∅."""
        return cls(low=0.0, high=0.0, empty=True)

    @classmethod
    def entire(cls) -> IEEEInterval:
        """Create the entire real line [−∞, +∞]."""
        return cls(low=-math.inf, high=math.inf, decoration=Decoration.DAC)

    @classmethod
    def from_interval(cls, iv: Interval) -> IEEEInterval:
        """Promote a standard :class:`Interval` to an IEEE interval."""
        return cls(low=iv.low, high=iv.high, decoration=Decoration.COM)

    @classmethod
    def from_value(cls, value: float) -> IEEEInterval:
        """Create a point (degenerate) interval [v, v]."""
        v = float(value)
        return cls(low=v, high=v, decoration=Decoration.COM)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_interval(self) -> Optional[Interval]:
        """Convert to a standard :class:`Interval`, or None if empty."""
        if self.empty:
            return None
        return Interval(self.low, self.high)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def midpoint(self) -> float:
        """Midpoint of the interval.  NaN for empty intervals."""
        if self.empty:
            return float("nan")
        if self.low == -math.inf and self.high == math.inf:
            return 0.0
        if self.low == -math.inf:
            return -sys.float_info.max
        if self.high == math.inf:
            return sys.float_info.max
        return (self.low + self.high) / 2.0

    def radius(self) -> float:
        """Radius (half-width).  +∞ for unbounded, NaN for empty."""
        if self.empty:
            return float("nan")
        return (self.high - self.low) / 2.0

    def width(self) -> float:
        """Width (diameter).  +∞ for unbounded, NaN for empty."""
        if self.empty:
            return float("nan")
        return self.high - self.low

    def magnitude(self) -> float:
        """max(|low|, |high|).  NaN for empty."""
        if self.empty:
            return float("nan")
        return max(abs(self.low), abs(self.high))

    def mignitude(self) -> float:
        """Minimum absolute value contained.  NaN for empty."""
        if self.empty:
            return float("nan")
        if self.low <= 0.0 <= self.high:
            return 0.0
        return min(abs(self.low), abs(self.high))

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """True if this is the empty interval."""
        return self.empty

    def is_entire(self) -> bool:
        """True if this is [−∞, +∞]."""
        return (not self.empty
                and self.low == -math.inf
                and self.high == math.inf)

    def is_singleton(self) -> bool:
        """True if this is a point interval [v, v]."""
        return not self.empty and self.low == self.high

    def contains(self, value: float) -> bool:
        """True if *value* ∈ this interval."""
        if self.empty:
            return False
        return self.low <= value <= self.high

    def is_subset_of(self, other: IEEEInterval) -> bool:
        """True if this interval ⊆ *other*."""
        if self.empty:
            return True
        if other.empty:
            return False
        return other.low <= self.low and self.high <= other.high

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.empty:
            return "IEEEInterval(∅)"
        return f"IEEEInterval([{self.low}, {self.high}], {self.decoration.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IEEEInterval):
            return NotImplemented
        if self.empty and other.empty:
            return True
        return (
            not self.empty and not other.empty
            and self.low == other.low
            and self.high == other.high
        )


# ═══════════════════════════════════════════════════════════════════════════
# Directed-rounding arithmetic
# ═══════════════════════════════════════════════════════════════════════════

def add_rd(a: float, b: float) -> float:
    """Add with rounding toward −∞."""
    return _round_down(a + b)


def add_ru(a: float, b: float) -> float:
    """Add with rounding toward +∞."""
    return _round_up(a + b)


def mul_rd(a: float, b: float) -> float:
    """Multiply with rounding toward −∞."""
    return _round_down(a * b)


def mul_ru(a: float, b: float) -> float:
    """Multiply with rounding toward +∞."""
    return _round_up(a * b)


def div_rd(a: float, b: float) -> float:
    """Divide with rounding toward −∞.

    Raises
    ------
    ZeroDivisionError
        If *b* is zero.
    """
    if b == 0.0:
        raise ZeroDivisionError("Division by zero in div_rd.")
    return _round_down(a / b)


def div_ru(a: float, b: float) -> float:
    """Divide with rounding toward +∞.

    Raises
    ------
    ZeroDivisionError
        If *b* is zero.
    """
    if b == 0.0:
        raise ZeroDivisionError("Division by zero in div_ru.")
    return _round_up(a / b)


# ═══════════════════════════════════════════════════════════════════════════
# IEEE interval operations
# ═══════════════════════════════════════════════════════════════════════════

def ieee_add(a: IEEEInterval, b: IEEEInterval) -> IEEEInterval:
    """IEEE 1788 interval addition with directed rounding."""
    if a.empty or b.empty:
        return IEEEInterval.make_empty()
    lo = add_rd(a.low, b.low)
    hi = add_ru(a.high, b.high)
    dec = _meet_decoration(a.decoration, b.decoration)
    return IEEEInterval(low=lo, high=hi, decoration=dec)


def ieee_sub(a: IEEEInterval, b: IEEEInterval) -> IEEEInterval:
    """IEEE 1788 interval subtraction with directed rounding."""
    if a.empty or b.empty:
        return IEEEInterval.make_empty()
    lo = add_rd(a.low, -b.high)
    hi = add_ru(a.high, -b.low)
    dec = _meet_decoration(a.decoration, b.decoration)
    return IEEEInterval(low=lo, high=hi, decoration=dec)


def ieee_mul(a: IEEEInterval, b: IEEEInterval) -> IEEEInterval:
    """IEEE 1788 interval multiplication with directed rounding."""
    if a.empty or b.empty:
        return IEEEInterval.make_empty()

    candidates_lo = [
        mul_rd(a.low, b.low),
        mul_rd(a.low, b.high),
        mul_rd(a.high, b.low),
        mul_rd(a.high, b.high),
    ]
    candidates_hi = [
        mul_ru(a.low, b.low),
        mul_ru(a.low, b.high),
        mul_ru(a.high, b.low),
        mul_ru(a.high, b.high),
    ]

    lo = min(candidates_lo)
    hi = max(candidates_hi)
    dec = _meet_decoration(a.decoration, b.decoration)
    return IEEEInterval(low=lo, high=hi, decoration=dec)


def ieee_div(a: IEEEInterval, b: IEEEInterval) -> IEEEInterval:
    """IEEE 1788 interval division with directed rounding.

    Raises
    ------
    ZeroDivisionError
        If *b* contains zero in its interior.
    """
    if a.empty or b.empty:
        return IEEEInterval.make_empty()

    if b.low < 0.0 < b.high:
        raise ZeroDivisionError(
            "Division by an interval containing zero in its interior."
        )
    if b.low == 0.0 and b.high == 0.0:
        raise ZeroDivisionError("Division by the degenerate interval [0, 0].")

    # Compute 1/b then multiply
    if b.low == 0.0:
        inv_b = IEEEInterval(
            low=div_rd(1.0, b.high), high=math.inf,
            decoration=Decoration.TRV,
        )
    elif b.high == 0.0:
        inv_b = IEEEInterval(
            low=-math.inf, high=div_ru(1.0, b.low),
            decoration=Decoration.TRV,
        )
    else:
        inv_lo = div_rd(1.0, b.high)
        inv_hi = div_ru(1.0, b.low)
        inv_b = IEEEInterval(low=inv_lo, high=inv_hi, decoration=b.decoration)

    return ieee_mul(a, inv_b)


def sqrt_interval(x: IEEEInterval) -> IEEEInterval:
    """Square root of an IEEE interval with proper rounding.

    Parameters
    ----------
    x : IEEEInterval
        Must be non-negative (or the negative part is clamped to 0).

    Returns
    -------
    IEEEInterval
    """
    if x.empty:
        return IEEEInterval.make_empty()
    if x.high < 0.0:
        return IEEEInterval.make_empty()

    lo = max(x.low, 0.0)
    lo_sqrt = _round_down(math.sqrt(lo))
    hi_sqrt = _round_up(math.sqrt(x.high))

    dec = x.decoration
    if x.low < 0.0:
        dec = _meet_decoration(dec, Decoration.TRV)

    return IEEEInterval(low=lo_sqrt, high=hi_sqrt, decoration=dec)


def pow_interval(x: IEEEInterval, n: int) -> IEEEInterval:
    """Integer power of an IEEE interval with proper rounding.

    Parameters
    ----------
    x : IEEEInterval
        Base interval.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    IEEEInterval
    """
    if x.empty:
        return IEEEInterval.make_empty()
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"Exponent must be a non-negative integer, got {n}.")
    if n == 0:
        return IEEEInterval.from_value(1.0)
    if n == 1:
        return IEEEInterval(low=x.low, high=x.high, decoration=x.decoration)

    a, b = x.low, x.high

    if n % 2 == 1:
        lo = _round_down(a ** n)
        hi = _round_up(b ** n)
    else:
        if a >= 0.0:
            lo = _round_down(a ** n)
            hi = _round_up(b ** n)
        elif b <= 0.0:
            lo = _round_down(b ** n)
            hi = _round_up(a ** n)
        else:
            lo = 0.0
            hi = _round_up(max((-a) ** n, b ** n))

    return IEEEInterval(low=lo, high=hi, decoration=x.decoration)


# ═══════════════════════════════════════════════════════════════════════════
# Set operations
# ═══════════════════════════════════════════════════════════════════════════

def intersection(a: IEEEInterval, b: IEEEInterval) -> IEEEInterval:
    """Interval intersection.  Returns empty if disjoint."""
    if a.empty or b.empty:
        return IEEEInterval.make_empty()
    lo = max(a.low, b.low)
    hi = min(a.high, b.high)
    if lo > hi:
        return IEEEInterval.make_empty()
    dec = _meet_decoration(a.decoration, b.decoration)
    return IEEEInterval(low=lo, high=hi, decoration=dec)


def hull(a: IEEEInterval, b: IEEEInterval) -> IEEEInterval:
    """Interval hull — smallest interval enclosing both operands."""
    if a.empty:
        return IEEEInterval(low=b.low, high=b.high, decoration=b.decoration,
                            empty=b.empty)
    if b.empty:
        return IEEEInterval(low=a.low, high=a.high, decoration=a.decoration,
                            empty=a.empty)
    lo = min(a.low, b.low)
    hi = max(a.high, b.high)
    dec = _meet_decoration(a.decoration, b.decoration)
    return IEEEInterval(low=lo, high=hi, decoration=dec)
