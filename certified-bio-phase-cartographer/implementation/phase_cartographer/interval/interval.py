"""
Core interval arithmetic class with rigorous rounding.

Implements IEEE 754 interval arithmetic with proper outward rounding
for all basic operations, elementary functions, and set operations.
"""

import numpy as np
import math
from typing import Union, Optional, Tuple
from . import rounding as rd


class Interval:
    """
    Rigorous interval [lo, hi] with outward-rounded arithmetic.
    
    All arithmetic operations produce guaranteed enclosures:
    the true result is always contained in the returned interval.
    """
    
    __slots__ = ['lo', 'hi']
    
    def __init__(self, lo: float = 0.0, hi: Optional[float] = None):
        if hi is None:
            hi = lo
        if isinstance(lo, Interval):
            self.lo = lo.lo
            self.hi = lo.hi
            return
        self.lo = float(lo)
        self.hi = float(hi)
        if self.lo > self.hi:
            raise ValueError(f"Invalid interval: [{self.lo}, {self.hi}]")
    
    @classmethod
    def entire(cls) -> 'Interval':
        """Return the entire real line interval."""
        return cls(float('-inf'), float('inf'))
    
    @classmethod
    def empty(cls) -> 'Interval':
        """Return the empty interval."""
        result = cls.__new__(cls)
        result.lo = float('nan')
        result.hi = float('nan')
        return result
    
    @classmethod
    def from_midpoint_radius(cls, mid: float, rad: float) -> 'Interval':
        """Create interval from midpoint and radius."""
        return cls(mid - abs(rad), mid + abs(rad))
    
    @classmethod
    def hull_of(cls, intervals) -> 'Interval':
        """Compute hull of a collection of intervals."""
        lo = float('inf')
        hi = float('-inf')
        for iv in intervals:
            if isinstance(iv, (int, float)):
                iv = cls(iv)
            lo = min(lo, iv.lo)
            hi = max(hi, iv.hi)
        return cls(lo, hi)
    
    @property
    def mid(self) -> float:
        """Midpoint of the interval."""
        if math.isinf(self.lo) or math.isinf(self.hi):
            if math.isinf(self.lo) and math.isinf(self.hi):
                return 0.0
            if math.isinf(self.lo):
                return -1e308
            return 1e308
        return (self.lo + self.hi) / 2.0
    
    @property
    def rad(self) -> float:
        """Radius (half-width) of the interval."""
        return (self.hi - self.lo) / 2.0
    
    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.hi - self.lo
    
    @property
    def mag(self) -> float:
        """Magnitude: max(|lo|, |hi|)."""
        return max(abs(self.lo), abs(self.hi))
    
    @property
    def mig(self) -> float:
        """Mignitude: min distance from zero."""
        if self.lo <= 0 <= self.hi:
            return 0.0
        return min(abs(self.lo), abs(self.hi))
    
    def is_empty(self) -> bool:
        """Check if interval is empty."""
        return math.isnan(self.lo)
    
    def is_entire(self) -> bool:
        """Check if interval is the entire real line."""
        return math.isinf(self.lo) and self.lo < 0 and math.isinf(self.hi) and self.hi > 0
    
    def is_thin(self, tol: float = 1e-15) -> bool:
        """Check if interval is thin (nearly a point)."""
        return self.width <= tol
    
    def contains(self, x: Union[float, 'Interval']) -> bool:
        """Check if x is contained in this interval."""
        if isinstance(x, Interval):
            return self.lo <= x.lo and x.hi <= self.hi
        return self.lo <= x <= self.hi
    
    def overlaps(self, other: 'Interval') -> bool:
        """Check if intervals overlap."""
        return self.lo <= other.hi and other.lo <= self.hi
    
    def intersection(self, other: 'Interval') -> 'Interval':
        """Compute intersection of two intervals."""
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return Interval.empty()
        return Interval(lo, hi)
    
    def hull(self, other: 'Interval') -> 'Interval':
        """Compute hull (convex hull) of two intervals."""
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))
    
    def split(self) -> Tuple['Interval', 'Interval']:
        """Split interval at midpoint."""
        m = self.mid
        return Interval(self.lo, m), Interval(m, self.hi)
    
    def blow_up(self, factor: float) -> 'Interval':
        """Inflate interval by a factor around its midpoint."""
        m = self.mid
        r = self.rad * factor
        return Interval(m - r, m + r)
    
    def inflate(self, eps: float) -> 'Interval':
        """Add eps to both sides."""
        return Interval(self.lo - eps, self.hi + eps)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        if not isinstance(other, Interval):
            return NotImplemented
        lo = rd.add_down(self.lo, other.lo)
        hi = rd.add_up(self.hi, other.hi)
        return Interval(lo, hi)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return Interval(-self.hi, -self.lo)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        if not isinstance(other, Interval):
            return NotImplemented
        lo = rd.sub_down(self.lo, other.hi)
        hi = rd.sub_up(self.hi, other.lo)
        return Interval(lo, hi)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        return other.__sub__(self)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        if not isinstance(other, Interval):
            return NotImplemented
        candidates = [
            rd.mul_down(self.lo, other.lo),
            rd.mul_down(self.lo, other.hi),
            rd.mul_down(self.hi, other.lo),
            rd.mul_down(self.hi, other.hi),
        ]
        candidates_up = [
            rd.mul_up(self.lo, other.lo),
            rd.mul_up(self.lo, other.hi),
            rd.mul_up(self.hi, other.lo),
            rd.mul_up(self.hi, other.hi),
        ]
        return Interval(min(candidates), max(candidates_up))
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        if not isinstance(other, Interval):
            return NotImplemented
        if other.contains(0.0):
            if other.lo == 0.0 and other.hi == 0.0:
                raise ZeroDivisionError("Division by zero interval [0,0]")
            if other.lo == 0.0:
                return self * Interval(1.0 / other.hi, float('inf'))
            if other.hi == 0.0:
                return self * Interval(float('-inf'), 1.0 / other.lo)
            return Interval.entire()
        recip = Interval(rd.div_down(1.0, other.hi), rd.div_up(1.0, other.lo))
        return self * recip
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        return other.__truediv__(self)
    
    def __pow__(self, n):
        if isinstance(n, int):
            if n == 0:
                return Interval(1.0, 1.0)
            if n == 1:
                return Interval(self.lo, self.hi)
            if n < 0:
                return Interval(1.0) / (self ** (-n))
            lo, hi = rd.power_enclosure(self.lo, self.hi, n)
            return Interval(lo, hi)
        if isinstance(n, float):
            if self.lo < 0:
                raise ValueError("Cannot raise negative interval to non-integer power")
            if self.lo == 0 and n < 0:
                raise ZeroDivisionError("Zero interval to negative power")
            lo = rd.exp_down(n * rd.log_down(max(self.lo, 1e-300)))
            hi = rd.exp_up(n * rd.log_up(max(self.hi, 1e-300)))
            return Interval(lo, hi)
        return NotImplemented
    
    def __abs__(self):
        if self.lo >= 0:
            return Interval(self.lo, self.hi)
        if self.hi <= 0:
            return Interval(-self.hi, -self.lo)
        return Interval(0.0, max(-self.lo, self.hi))
    
    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.lo == other.lo and self.hi == other.hi
        if isinstance(other, (int, float)):
            return self.lo == other and self.hi == other
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.hi < other
        if isinstance(other, Interval):
            return self.hi < other.lo
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, (int, float)):
            return self.hi <= other
        if isinstance(other, Interval):
            return self.hi <= other.lo
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.lo > other
        if isinstance(other, Interval):
            return self.lo > other.hi
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.lo >= other
        if isinstance(other, Interval):
            return self.lo >= other.hi
        return NotImplemented
    
    def __repr__(self):
        return f"Interval({self.lo}, {self.hi})"
    
    def __str__(self):
        return f"[{self.lo:.6e}, {self.hi:.6e}]"
    
    def __hash__(self):
        return hash((self.lo, self.hi))
    
    def __float__(self):
        return self.mid
    
    def __bool__(self):
        return not self.is_empty()
    
    def exp(self) -> 'Interval':
        """Interval exponential."""
        lo = rd.exp_down(self.lo)
        hi = rd.exp_up(self.hi)
        return Interval(lo, hi)
    
    def log(self) -> 'Interval':
        """Interval natural logarithm."""
        if self.lo <= 0:
            raise ValueError("Logarithm requires strictly positive interval")
        lo = rd.log_down(self.lo)
        hi = rd.log_up(self.hi)
        return Interval(lo, hi)
    
    def sqrt(self) -> 'Interval':
        """Interval square root."""
        if self.lo < 0:
            raise ValueError("Square root of negative interval")
        lo = rd.sqrt_down(max(self.lo, 0.0))
        hi = rd.sqrt_up(self.hi)
        return Interval(lo, hi)
    
    def sin(self) -> 'Interval':
        """Interval sine."""
        lo, hi = rd.sin_enclosure(self.lo, self.hi)
        return Interval(lo, hi)
    
    def cos(self) -> 'Interval':
        """Interval cosine."""
        lo, hi = rd.cos_enclosure(self.lo, self.hi)
        return Interval(lo, hi)
    
    def reciprocal(self) -> 'Interval':
        """Validated reciprocal for rational functions."""
        if self.contains(0.0):
            raise ZeroDivisionError("Cannot compute reciprocal of interval containing zero")
        lo = rd.div_down(1.0, self.hi)
        hi = rd.div_up(1.0, self.lo)
        return Interval(lo, hi)
    
    def hill_function(self, K: 'Interval', n_hill: int) -> 'Interval':
        """Compute Hill function x^n / (K^n + x^n) with validated bounds."""
        if self.lo < 0:
            raise ValueError("Hill function requires non-negative state")
        xn = self ** n_hill
        Kn = K ** n_hill
        denom = Kn + xn
        if denom.contains(0.0):
            raise ValueError("Hill function denominator contains zero")
        return xn / denom
    
    def michaelis_menten(self, Km: 'Interval', Vmax: 'Interval') -> 'Interval':
        """Compute Michaelis-Menten kinetics Vmax * x / (Km + x)."""
        if self.lo < 0:
            raise ValueError("Michaelis-Menten requires non-negative substrate")
        denom = Km + self
        if denom.contains(0.0):
            raise ValueError("Michaelis-Menten denominator contains zero")
        return Vmax * self / denom
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (lo, hi) tuple."""
        return (self.lo, self.hi)
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> 'Interval':
        """Create from (lo, hi) tuple."""
        return cls(t[0], t[1])


def iv(lo: float, hi: Optional[float] = None) -> Interval:
    """Convenience constructor for Interval."""
    return Interval(lo, hi)


def iv_pi() -> Interval:
    """Return rigorous enclosure of pi."""
    import mpmath
    with mpmath.workdps(30):
        pi_val = mpmath.pi
        lo = float(pi_val - mpmath.mpf('1e-16'))
        hi = float(pi_val + mpmath.mpf('1e-16'))
    return Interval(lo, hi)


def iv_e() -> Interval:
    """Return rigorous enclosure of e."""
    import mpmath
    with mpmath.workdps(30):
        e_val = mpmath.e
        lo = float(e_val - mpmath.mpf('1e-16'))
        hi = float(e_val + mpmath.mpf('1e-16'))
    return Interval(lo, hi)


def iv_zero() -> Interval:
    """Return the zero interval."""
    return Interval(0.0, 0.0)


def iv_one() -> Interval:
    """Return the unit interval [1,1]."""
    return Interval(1.0, 1.0)


def iv_pos(lo: float, hi: float) -> Interval:
    """Create strictly positive interval, raising error if bounds are non-positive."""
    if lo <= 0 or hi <= 0:
        raise ValueError(f"Expected positive interval, got [{lo}, {hi}]")
    return Interval(lo, hi)
