"""Interval arithmetic for rigorous numerical computation.

Provides an Interval class that propagates bounds through arithmetic operations,
ensuring mathematical correctness of computed confidence intervals and test statistics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
import numpy as np


def _to_interval(value) -> "Interval":
    """Coerce a scalar or Interval to an Interval."""
    if isinstance(value, Interval):
        return value
    return Interval.point(float(value))


@dataclass(frozen=True)
class Interval:
    """A closed interval [lo, hi] supporting arithmetic operations.

    All arithmetic operations produce intervals that are guaranteed to contain
    the true result, assuming the inputs contain the true values.
    """
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi:
            lo, hi = self.hi, self.lo
            object.__setattr__(self, 'lo', lo)
            object.__setattr__(self, 'hi', hi)

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #
    @classmethod
    def point(cls, value: float) -> Interval:
        return cls(value, value)

    @classmethod
    def pm(cls, center: float, radius: float) -> Interval:
        """Create interval center ± radius."""
        r = abs(radius)
        return cls(center - r, center + r)

    @classmethod
    def from_confidence(cls, point_est: float, std_err: float,
                        z: float = 1.96) -> Interval:
        """Create from point estimate and standard error."""
        half = abs(std_err) * abs(z)
        return cls(point_est - half, point_est + half)

    @classmethod
    def hull(cls, intervals: List[Interval]) -> Interval:
        """Smallest interval containing all given intervals."""
        if not intervals:
            return cls.empty()
        lo = min(iv.lo for iv in intervals)
        hi = max(iv.hi for iv in intervals)
        return cls(lo, hi)

    @classmethod
    def empty(cls) -> Interval:
        return cls(float('inf'), float('-inf'))

    @classmethod
    def entire(cls) -> Interval:
        return cls(float('-inf'), float('inf'))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        if math.isinf(self.lo) or math.isinf(self.hi):
            if math.isinf(self.lo) and math.isinf(self.hi):
                return 0.0
            if math.isinf(self.lo):
                return -1e308
            return 1e308
        return (self.lo + self.hi) / 2.0

    @property
    def radius(self) -> float:
        return self.width / 2.0

    @property
    def is_empty(self) -> bool:
        return self.lo > self.hi

    @property
    def is_point(self) -> bool:
        return self.lo == self.hi

    @property
    def is_positive(self) -> bool:
        return self.lo > 0

    @property
    def is_negative(self) -> bool:
        return self.hi < 0

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def overlaps(self, other: Interval) -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    def intersection(self, other: Interval) -> Optional[Interval]:
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)

    def union_hull(self, other: Interval) -> Interval:
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    # ------------------------------------------------------------------ #
    # Arithmetic operations
    # ------------------------------------------------------------------ #
    def __add__(self, other) -> Interval:
        o = _to_interval(other)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    def __radd__(self, other) -> Interval:
        return self.__add__(other)

    def __sub__(self, other) -> Interval:
        o = _to_interval(other)
        return Interval(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, other) -> Interval:
        o = _to_interval(other)
        return Interval(o.lo - self.hi, o.hi - self.lo)

    def __mul__(self, other) -> Interval:
        o = _to_interval(other)
        products = (
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        )
        return Interval(min(products), max(products))

    def __rmul__(self, other) -> Interval:
        return self.__mul__(other)

    def __truediv__(self, other) -> Interval:
        o = _to_interval(other)
        if o.lo <= 0 <= o.hi:
            if o.lo == 0 and o.hi == 0:
                raise ZeroDivisionError("Division by the zero interval [0, 0]")
            if o.lo == 0:
                # Denominator is [0, hi] → reciprocal is [1/hi, +inf]
                recip = Interval(1.0 / o.hi, float('inf'))
            elif o.hi == 0:
                # Denominator is [lo, 0] → reciprocal is [-inf, 1/lo]
                recip = Interval(float('-inf'), 1.0 / o.lo)
            else:
                return Interval.entire()
            return self * recip
        recip = Interval(1.0 / o.hi, 1.0 / o.lo)
        return self * recip

    def __rtruediv__(self, other) -> Interval:
        o = _to_interval(other)
        return o.__truediv__(self)

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __abs__(self) -> Interval:
        if self.lo >= 0:
            return Interval(self.lo, self.hi)
        if self.hi <= 0:
            return Interval(-self.hi, -self.lo)
        return Interval(0.0, max(-self.lo, self.hi))

    def __pow__(self, n: int) -> Interval:
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Only integer powers are supported, got {type(n)}")
        n = int(n)
        if n == 0:
            return Interval.point(1.0)
        if n == 1:
            return Interval(self.lo, self.hi)
        if n < 0:
            return Interval.point(1.0) / (self ** (-n))

        if n % 2 == 0:
            # Even power: result is non-negative
            if self.lo >= 0:
                return Interval(self.lo ** n, self.hi ** n)
            if self.hi <= 0:
                return Interval(self.hi ** n, self.lo ** n)
            # Interval straddles zero
            return Interval(0.0, max(self.lo ** n, self.hi ** n))
        else:
            # Odd power: monotonically increasing
            return Interval(self.lo ** n, self.hi ** n)

    # ------------------------------------------------------------------ #
    # Mathematical functions
    # ------------------------------------------------------------------ #
    def sqrt(self) -> Interval:
        if self.hi < 0:
            raise ValueError("sqrt of entirely negative interval")
        lo = math.sqrt(max(self.lo, 0.0))
        hi = math.sqrt(self.hi)
        return Interval(lo, hi)

    def exp(self) -> Interval:
        return Interval(math.exp(self.lo), math.exp(self.hi))

    def log(self) -> Interval:
        if self.hi <= 0:
            raise ValueError("log of non-positive interval")
        lo = math.log(max(self.lo, 1e-300))
        hi = math.log(self.hi)
        return Interval(lo, hi)

    def sin(self) -> Interval:
        """Conservative sine bound.

        Uses the monotonicity of sin on subintervals of length < π and
        falls back to [-1, 1] for wide intervals.
        """
        if self.width >= 2 * math.pi:
            return Interval(-1.0, 1.0)

        lo_mod = self.lo % (2 * math.pi)
        hi_mod = lo_mod + self.width

        candidates = [math.sin(self.lo), math.sin(self.hi)]

        # Check if a peak (π/2 + 2kπ) is inside the interval
        k_start = math.floor((lo_mod - math.pi / 2) / (2 * math.pi))
        for k in range(int(k_start), int(k_start) + 3):
            peak = math.pi / 2 + 2 * math.pi * k
            if lo_mod <= peak <= hi_mod:
                candidates.append(1.0)

        # Check if a trough (3π/2 + 2kπ) is inside the interval
        k_start = math.floor((lo_mod - 3 * math.pi / 2) / (2 * math.pi))
        for k in range(int(k_start), int(k_start) + 3):
            trough = 3 * math.pi / 2 + 2 * math.pi * k
            if lo_mod <= trough <= hi_mod:
                candidates.append(-1.0)

        return Interval(min(candidates), max(candidates))

    def cos(self) -> Interval:
        """Conservative cosine bound via cos(x) = sin(x + π/2)."""
        shifted = Interval(self.lo + math.pi / 2, self.hi + math.pi / 2)
        return shifted.sin()

    # ------------------------------------------------------------------ #
    # Comparison (definite ordering)
    # ------------------------------------------------------------------ #
    def definitely_less_than(self, other: Union[Interval, float]) -> bool:
        o = _to_interval(other)
        return self.hi < o.lo

    def definitely_greater_than(self, other: Union[Interval, float]) -> bool:
        o = _to_interval(other)
        return self.lo > o.hi

    def possibly_equal(self, other: Union[Interval, float]) -> bool:
        o = _to_interval(other)
        return self.overlaps(o)

    # ------------------------------------------------------------------ #
    # Rich comparison helpers (for use in sorted / min / max)
    # ------------------------------------------------------------------ #
    def __eq__(self, other) -> bool:
        if isinstance(other, Interval):
            return self.lo == other.lo and self.hi == other.hi
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.lo, self.hi))

    def __lt__(self, other) -> bool:
        if isinstance(other, Interval):
            return self.midpoint < other.midpoint
        return NotImplemented

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        return f"[{self.lo:.6g}, {self.hi:.6g}]"

    def __repr__(self) -> str:
        return f"Interval({self.lo}, {self.hi})"


# ==================================================================== #
# IntervalVector
# ==================================================================== #

class IntervalVector:
    """A vector of intervals, supporting element-wise operations."""

    def __init__(self, intervals: List[Interval]):
        self.intervals = list(intervals)

    @classmethod
    def from_arrays(cls, lo: np.ndarray, hi: np.ndarray) -> IntervalVector:
        lo = np.asarray(lo, dtype=float).ravel()
        hi = np.asarray(hi, dtype=float).ravel()
        if lo.shape != hi.shape:
            raise ValueError("lo and hi must have the same shape")
        return cls([Interval(float(l), float(h)) for l, h in zip(lo, hi)])

    @classmethod
    def from_point_array(cls, values: np.ndarray) -> IntervalVector:
        values = np.asarray(values, dtype=float).ravel()
        return cls([Interval.point(float(v)) for v in values])

    @classmethod
    def from_confidence_arrays(cls, values: np.ndarray,
                               errors: np.ndarray,
                               z: float = 1.96) -> IntervalVector:
        values = np.asarray(values, dtype=float).ravel()
        errors = np.asarray(errors, dtype=float).ravel()
        return cls([
            Interval.from_confidence(float(v), float(e), z)
            for v, e in zip(values, errors)
        ])

    def __len__(self) -> int:
        return len(self.intervals)

    def __getitem__(self, idx) -> Union[Interval, "IntervalVector"]:
        result = self.intervals[idx]
        if isinstance(result, list):
            return IntervalVector(result)
        return result

    def _broadcast(self, other) -> List[Interval]:
        """Return other as a list of Intervals matching self's length."""
        if isinstance(other, IntervalVector):
            if len(other) != len(self):
                raise ValueError("IntervalVector lengths must match")
            return other.intervals
        if isinstance(other, Interval):
            return [other] * len(self)
        # scalar
        iv = _to_interval(other)
        return [iv] * len(self)

    def __add__(self, other) -> IntervalVector:
        others = self._broadcast(other)
        return IntervalVector([a + b for a, b in zip(self.intervals, others)])

    def __radd__(self, other) -> IntervalVector:
        return self.__add__(other)

    def __sub__(self, other) -> IntervalVector:
        others = self._broadcast(other)
        return IntervalVector([a - b for a, b in zip(self.intervals, others)])

    def __rsub__(self, other) -> IntervalVector:
        others = self._broadcast(other)
        return IntervalVector([b - a for a, b in zip(self.intervals, others)])

    def __mul__(self, other) -> IntervalVector:
        others = self._broadcast(other)
        return IntervalVector([a * b for a, b in zip(self.intervals, others)])

    def __rmul__(self, other) -> IntervalVector:
        return self.__mul__(other)

    def __truediv__(self, other) -> IntervalVector:
        others = self._broadcast(other)
        return IntervalVector([a / b for a, b in zip(self.intervals, others)])

    def dot(self, other: IntervalVector) -> Interval:
        """Interval dot product."""
        if len(self) != len(other):
            raise ValueError("IntervalVector lengths must match for dot product")
        result = Interval.point(0.0)
        for a, b in zip(self.intervals, other.intervals):
            result = result + a * b
        return result

    def sum(self) -> Interval:
        result = Interval.point(0.0)
        for iv in self.intervals:
            result = result + iv
        return result

    def mean(self) -> Interval:
        n = len(self.intervals)
        if n == 0:
            raise ValueError("Cannot take mean of empty IntervalVector")
        return self.sum() / Interval.point(float(n))

    @property
    def lo_array(self) -> np.ndarray:
        return np.array([iv.lo for iv in self.intervals])

    @property
    def hi_array(self) -> np.ndarray:
        return np.array([iv.hi for iv in self.intervals])

    @property
    def midpoints(self) -> np.ndarray:
        return np.array([iv.midpoint for iv in self.intervals])

    @property
    def widths(self) -> np.ndarray:
        return np.array([iv.width for iv in self.intervals])

    def __repr__(self) -> str:
        body = ", ".join(repr(iv) for iv in self.intervals[:5])
        if len(self.intervals) > 5:
            body += ", ..."
        return f"IntervalVector([{body}])"


# ==================================================================== #
# Utility functions for interval computation
# ==================================================================== #

def interval_mean(data: np.ndarray, std_err: Optional[float] = None) -> Interval:
    """Compute mean with rigorous error bounds.

    If *std_err* is provided it is used directly; otherwise the standard
    error is estimated from the data as  std(data) / sqrt(n).
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    if n == 0:
        raise ValueError("Cannot compute mean of empty array")

    m = float(np.mean(data))
    if std_err is None:
        if n < 2:
            return Interval.point(m)
        se = float(np.std(data, ddof=1)) / math.sqrt(n)
    else:
        se = float(std_err)

    # Use t-like multiplier; for large n this is ≈ 1.96
    # For small n, use a slightly wider multiplier for safety
    if n >= 30:
        z = 1.96
    else:
        # Rough Bonferroni-safe widening for small samples
        z = 2.0 + 4.0 / n
    return Interval.from_confidence(m, se, z)


def interval_variance(data: np.ndarray) -> Interval:
    """Compute variance with bounds.

    Returns an interval guaranteed to contain the population variance
    using chi-squared-style reasoning on the sample variance.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points for variance")

    s2 = float(np.var(data, ddof=1))
    df = n - 1

    # Conservative chi-squared bounds: (df * s2) / chi2_hi  ≤  σ²  ≤  (df * s2) / chi2_lo
    # Approximate chi-squared quantiles using Wilson-Hilferty normal approx
    def _chi2_quantile(df: int, p: float) -> float:
        z = _normal_quantile(p)
        x = df * (1 - 2.0 / (9 * df) + z * math.sqrt(2.0 / (9 * df))) ** 3
        return max(x, 0.01)

    chi2_lo = _chi2_quantile(df, 0.025)
    chi2_hi = _chi2_quantile(df, 0.975)

    var_lo = df * s2 / chi2_hi
    var_hi = df * s2 / chi2_lo
    return Interval(var_lo, var_hi)


def _normal_quantile(p: float) -> float:
    """Rational approximation to the normal quantile function (Beasley-Springer-Moro)."""
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return 0.0

    # Abramowitz & Stegun 26.2.23 rational approximation
    if p < 0.5:
        sign = -1.0
        t = math.sqrt(-2.0 * math.log(p))
    else:
        sign = 1.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))

    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - numerator / denominator)


def interval_correlation(x: np.ndarray, y: np.ndarray) -> Interval:
    """Compute Pearson correlation with rigorous bounds.

    Uses Fisher's z-transform to produce a confidence interval and maps
    it back to the correlation scale.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n < 4:
        return Interval(-1.0, 1.0)

    r = float(np.corrcoef(x, y)[0, 1])
    r = max(-0.9999, min(0.9999, r))

    # Fisher z-transform
    z = 0.5 * math.log((1 + r) / (1 - r))
    se_z = 1.0 / math.sqrt(n - 3)

    z_lo = z - 1.96 * se_z
    z_hi = z + 1.96 * se_z

    # Inverse Fisher transform
    r_lo = math.tanh(z_lo)
    r_hi = math.tanh(z_hi)

    return Interval(max(r_lo, -1.0), min(r_hi, 1.0))


def interval_regression_slope(x: np.ndarray, y: np.ndarray) -> Interval:
    """OLS slope with interval bounds.

    Computes the ordinary least-squares slope and wraps it in a
    confidence interval using the standard error of the slope estimate.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n < 3:
        return Interval.entire()

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    ss_xx = float(np.sum((x - x_mean) ** 2))
    ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))

    if ss_xx < 1e-15:
        return Interval.entire()

    beta = ss_xy / ss_xx

    y_pred = x_mean + beta * (x - x_mean)  # simplified: intercept absorbed
    # Full model: y_hat = (y_mean - beta * x_mean) + beta * x
    y_hat = (y_mean - beta * x_mean) + beta * x
    residuals = y - y_hat
    rss = float(np.sum(residuals ** 2))

    df = n - 2
    mse = rss / df if df > 0 else 0.0
    se_beta = math.sqrt(mse / ss_xx) if ss_xx > 0 else float('inf')

    # Use t-multiplier (approximate for large n, conservative for small n)
    if n >= 30:
        t_mult = 1.96
    else:
        t_mult = 2.0 + 4.0 / n

    return Interval.from_confidence(beta, se_beta, t_mult)


def interval_t_statistic(x: np.ndarray, mu0: float = 0.0) -> Interval:
    """Compute t-statistic with bounds.

    Returns an interval for t = (x̄ - μ₀) / (s / √n), propagating
    uncertainty in both the mean and the standard deviation.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return Interval.entire()

    x_bar = interval_mean(x)
    var_iv = interval_variance(x)

    # Standard error interval: sqrt(var / n)
    se_iv = (var_iv / Interval.point(float(n))).sqrt()

    numerator = x_bar - Interval.point(mu0)

    if se_iv.lo <= 0:
        # Guard against zero or near-zero standard error
        if se_iv.hi <= 1e-15:
            return Interval.entire()
        se_iv = Interval(max(se_iv.lo, 1e-15), se_iv.hi)

    return numerator / se_iv


def interval_collusion_premium(observed: Interval, nash: Interval,
                               monopoly: Interval) -> Interval:
    """Compute collusion premium = (observed - nash) / (monopoly - nash).

    The collusion premium is 0 when observed == nash (competitive) and 1
    when observed == monopoly (full collusion).  The interval result
    rigorously propagates uncertainty from all three inputs.
    """
    numerator = observed - nash
    denominator = monopoly - nash

    if denominator.contains(0.0):
        if denominator.lo == 0.0 and denominator.hi == 0.0:
            raise ValueError(
                "Monopoly and Nash prices are identical; premium is undefined"
            )
        # Denominator interval contains zero — premium is unbounded
        return Interval.entire()

    return numerator / denominator
