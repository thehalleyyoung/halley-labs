"""
Interval arithmetic with proper outward rounding.

All arithmetic operations round outward to guarantee enclosure:
lower bounds round toward -inf, upper bounds round toward +inf.
This ensures that the true result is always contained in the
computed interval, providing mathematically rigorous enclosures.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import mpmath
    mpmath.mp.prec = 113  # quad precision for interval bounds
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False


# ---------------------------------------------------------------------------
# Module-level flag for validated arithmetic
# ---------------------------------------------------------------------------

_USE_VALIDATED = False


def use_validated_arithmetic(enabled: bool = True) -> None:
    """Set global flag to prefer ValidatedInterval for critical computations."""
    global _USE_VALIDATED
    _USE_VALIDATED = enabled


# ---------------------------------------------------------------------------
# Rounding helpers – NumPy does not expose fesetround on all platforms,
# so we use nextafter-based outward rounding as a portable fallback.
# ---------------------------------------------------------------------------

_NEG_INF = -np.inf
_POS_INF = np.inf
_FLOAT64 = np.float64


def _round_down(x: float) -> float:
    """Round toward -inf by one ULP."""
    v = _FLOAT64(x)
    if np.isnan(v) or np.isinf(v):
        return float(v)
    return float(np.nextafter(v, _FLOAT64(_NEG_INF)))


def _round_up(x: float) -> float:
    """Round toward +inf by one ULP."""
    v = _FLOAT64(x)
    if np.isnan(v) or np.isinf(v):
        return float(v)
    return float(np.nextafter(v, _FLOAT64(_POS_INF)))


# ---------------------------------------------------------------------------
# Interval class
# ---------------------------------------------------------------------------

class Interval:
    """A closed interval [lo, hi] with outward-rounded arithmetic."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: float, hi: Optional[float] = None) -> None:
        if hi is None:
            hi = lo
        self.lo = _FLOAT64(lo)
        self.hi = _FLOAT64(hi)
        if self.lo > self.hi:
            raise ValueError(f"Empty interval: [{self.lo}, {self.hi}]")

    # -- predicates ----------------------------------------------------------

    def is_empty(self) -> bool:
        return False  # construction prevents empty intervals

    def is_thin(self, tol: float = 0.0) -> bool:
        return (self.hi - self.lo) <= tol

    def contains_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    def is_positive(self) -> bool:
        return self.lo > 0.0

    def is_negative(self) -> bool:
        return self.hi < 0.0

    def contains(self, other: Union[float, "Interval"]) -> bool:
        if isinstance(other, Interval):
            return self.lo <= other.lo and other.hi <= self.hi
        return self.lo <= other <= self.hi

    def overlaps(self, other: "Interval") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    def strictly_contains(self, other: "Interval") -> bool:
        return self.lo < other.lo and other.hi < self.hi

    # -- scalar queries ------------------------------------------------------

    def mid(self) -> float:
        return float(0.5 * (self.lo + self.hi))

    def width(self) -> float:
        return float(self.hi - self.lo)

    def radius(self) -> float:
        return float(0.5 * (self.hi - self.lo))

    def mignitude(self) -> float:
        if self.contains_zero():
            return 0.0
        return float(min(abs(self.lo), abs(self.hi)))

    def magnitude(self) -> float:
        return float(max(abs(self.lo), abs(self.hi)))

    # -- arithmetic ----------------------------------------------------------

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __abs__(self) -> "Interval":
        if self.lo >= 0:
            return Interval(self.lo, self.hi)
        if self.hi <= 0:
            return Interval(-self.hi, -self.lo)
        return Interval(0.0, max(-self.lo, self.hi))

    def __add__(self, other: Union[float, "Interval"]) -> "Interval":
        if not isinstance(other, Interval):
            other = Interval(other)
        return Interval(
            _round_down(float(self.lo + other.lo)),
            _round_up(float(self.hi + other.hi)),
        )

    def __radd__(self, other: float) -> "Interval":
        return self.__add__(other)

    def __sub__(self, other: Union[float, "Interval"]) -> "Interval":
        if not isinstance(other, Interval):
            other = Interval(other)
        return Interval(
            _round_down(float(self.lo - other.hi)),
            _round_up(float(self.hi - other.lo)),
        )

    def __rsub__(self, other: float) -> "Interval":
        return Interval(other).__sub__(self)

    def __mul__(self, other: Union[float, "Interval"]) -> "Interval":
        if not isinstance(other, Interval):
            other = Interval(other)
        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return Interval(
            _round_down(float(min(products))),
            _round_up(float(max(products))),
        )

    def __rmul__(self, other: float) -> "Interval":
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, "Interval"]) -> "Interval":
        if not isinstance(other, Interval):
            other = Interval(other)
        if other.contains_zero():
            if other.lo == 0 and other.hi == 0:
                raise ZeroDivisionError("Division by zero interval [0,0]")
            # Extended division
            if other.lo == 0:
                inv = Interval(_round_down(1.0 / float(other.hi)), _POS_INF)
            elif other.hi == 0:
                inv = Interval(_NEG_INF, _round_up(1.0 / float(other.lo)))
            else:
                inv = Interval(_NEG_INF, _POS_INF)
            return self * inv
        inv = Interval(
            _round_down(1.0 / float(other.hi)),
            _round_up(1.0 / float(other.lo)),
        )
        return self * inv

    def __rtruediv__(self, other: float) -> "Interval":
        return Interval(other).__truediv__(self)

    def __pow__(self, n: int) -> "Interval":
        if not isinstance(n, (int, np.integer)):
            return pow_interval(self, Interval(n))
        if n == 0:
            return Interval(1.0)
        if n == 1:
            return Interval(self.lo, self.hi)
        if n < 0:
            return Interval(1.0) / (self ** (-n))
        if n % 2 == 0:
            if self.lo >= 0:
                return Interval(
                    _round_down(float(self.lo ** n)),
                    _round_up(float(self.hi ** n)),
                )
            if self.hi <= 0:
                return Interval(
                    _round_down(float(self.hi ** n)),
                    _round_up(float(self.lo ** n)),
                )
            # contains zero – even power
            upper = max(abs(self.lo), abs(self.hi))
            return Interval(0.0, _round_up(float(upper ** n)))
        # odd power
        return Interval(
            _round_down(float(self.lo ** n)),
            _round_up(float(self.hi ** n)),
        )

    # -- elementary functions ------------------------------------------------

    def sqrt(self) -> "Interval":
        if self.hi < 0:
            raise ValueError("sqrt of negative interval")
        lo = max(self.lo, 0.0)
        return Interval(
            _round_down(math.sqrt(lo)),
            _round_up(math.sqrt(float(self.hi))),
        )

    def exp(self) -> "Interval":
        return Interval(
            _round_down(math.exp(float(self.lo))),
            _round_up(math.exp(float(self.hi))),
        )

    def log(self) -> "Interval":
        if self.hi <= 0:
            raise ValueError("log of non-positive interval")
        lo = max(self.lo, np.finfo(np.float64).tiny)
        return Interval(
            _round_down(math.log(lo)),
            _round_up(math.log(float(self.hi))),
        )

    def sin(self) -> "Interval":
        return _monotone_trig(self, math.sin, period=2 * math.pi, kind="sin")

    def cos(self) -> "Interval":
        return _monotone_trig(self, math.cos, period=2 * math.pi, kind="cos")

    # -- comparison helpers (not ordering on intervals) ----------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Interval):
            return self.lo == other.lo and self.hi == other.hi
        return NotImplemented

    def __repr__(self) -> str:
        return f"Interval([{self.lo}, {self.hi}])"

    def __str__(self) -> str:
        return f"[{self.lo:.15g}, {self.hi:.15g}]"

    def __format__(self, spec: str) -> str:
        if not spec:
            return str(self)
        return f"[{self.lo:{spec}}, {self.hi:{spec}}]"

    # -- subdivision ---------------------------------------------------------

    def bisect(self) -> Tuple["Interval", "Interval"]:
        m = self.mid()
        return Interval(self.lo, m), Interval(m, self.hi)

    def subdivide(self, n: int) -> List["Interval"]:
        pts = np.linspace(float(self.lo), float(self.hi), n + 1)
        return [Interval(float(pts[i]), float(pts[i + 1])) for i in range(n)]

    def to_validated(self) -> "ValidatedInterval":
        """Convert to a ValidatedInterval backed by mpmath.iv."""
        return ValidatedInterval(float(self.lo), float(self.hi))


# ---------------------------------------------------------------------------
# ValidatedInterval — mpmath-backed rigorous interval arithmetic
# ---------------------------------------------------------------------------

class ValidatedInterval:
    """Interval arithmetic backed by mpmath.iv for rigorous rounding.

    Unlike Interval which uses nextafter-based outward rounding (sound for
    basic operations but may accumulate errors for transcendental functions),
    ValidatedInterval uses mpmath's arbitrary-precision interval arithmetic
    which provides mathematically rigorous enclosures at configurable precision.
    """
    __slots__ = ("_iv",)

    def __init__(self, lo, hi=None):
        if not _HAS_MPMATH:
            raise ImportError("mpmath required for ValidatedInterval")
        if hi is None:
            self._iv = mpmath.iv.mpf([lo, lo])
        else:
            self._iv = mpmath.iv.mpf([lo, hi])

    @property
    def lo(self):
        return float(self._iv.a)

    @property
    def hi(self):
        return float(self._iv.b)

    def to_interval(self):
        """Convert back to a standard Interval."""
        return Interval(self.lo, self.hi)

    def width(self):
        return self.hi - self.lo

    def mid(self):
        return (self.lo + self.hi) / 2.0

    def contains(self, other):
        if isinstance(other, ValidatedInterval):
            return self.lo <= other.lo and other.hi <= self.hi
        return self.lo <= float(other) <= self.hi

    # -- arithmetic operations using mpmath.iv ------------------------------

    def __add__(self, other):
        if isinstance(other, ValidatedInterval):
            return ValidatedInterval._from_iv(self._iv + other._iv)
        return ValidatedInterval._from_iv(self._iv + mpmath.iv.mpf(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ValidatedInterval):
            return ValidatedInterval._from_iv(self._iv - other._iv)
        return ValidatedInterval._from_iv(self._iv - mpmath.iv.mpf(other))

    def __rsub__(self, other):
        return ValidatedInterval._from_iv(mpmath.iv.mpf(other) - self._iv)

    def __mul__(self, other):
        if isinstance(other, ValidatedInterval):
            return ValidatedInterval._from_iv(self._iv * other._iv)
        return ValidatedInterval._from_iv(self._iv * mpmath.iv.mpf(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ValidatedInterval):
            return ValidatedInterval._from_iv(self._iv / other._iv)
        return ValidatedInterval._from_iv(self._iv / mpmath.iv.mpf(other))

    def __rtruediv__(self, other):
        return ValidatedInterval._from_iv(mpmath.iv.mpf(other) / self._iv)

    def __neg__(self):
        return ValidatedInterval._from_iv(-self._iv)

    def __pow__(self, n):
        return ValidatedInterval._from_iv(self._iv ** n)

    def sqrt(self):
        return ValidatedInterval._from_iv(mpmath.iv.sqrt(self._iv))

    def exp(self):
        return ValidatedInterval._from_iv(mpmath.iv.exp(self._iv))

    def log(self):
        return ValidatedInterval._from_iv(mpmath.iv.log(self._iv))

    def sin(self):
        return ValidatedInterval._from_iv(mpmath.iv.sin(self._iv))

    def cos(self):
        return ValidatedInterval._from_iv(mpmath.iv.cos(self._iv))

    @classmethod
    def _from_iv(cls, iv_val):
        """Construct from an mpmath interval value."""
        obj = object.__new__(cls)
        obj._iv = iv_val
        return obj

    def __eq__(self, other):
        if isinstance(other, ValidatedInterval):
            return self.lo == other.lo and self.hi == other.hi
        return NotImplemented

    def __repr__(self):
        return f"ValidatedInterval([{self.lo}, {self.hi}])"

    def __str__(self):
        return f"[{self.lo:.15g}, {self.hi:.15g}]"


# ---------------------------------------------------------------------------
# Free-standing interval functions
# ---------------------------------------------------------------------------

def hull(a: Interval, b: Interval) -> Interval:
    """Interval hull (smallest interval containing both)."""
    return Interval(min(a.lo, b.lo), max(a.hi, b.hi))


def intersection(a: Interval, b: Interval) -> Optional[Interval]:
    """Intersection of two intervals, or None if disjoint."""
    lo = max(a.lo, b.lo)
    hi = min(a.hi, b.hi)
    if lo > hi:
        return None
    return Interval(lo, hi)


def midpoint(iv: Interval) -> float:
    return iv.mid()


def width(iv: Interval) -> float:
    return iv.width()


def radius(iv: Interval) -> float:
    return iv.radius()


def pow_interval(base: Interval, exp: Interval) -> Interval:
    """base^exp for positive base via exp(exp*log(base))."""
    if base.lo <= 0:
        raise ValueError("pow requires positive base interval")
    return (exp * base.log()).exp()


# ---------------------------------------------------------------------------
# Trigonometric helper
# ---------------------------------------------------------------------------

def _monotone_trig(iv: Interval, fn, period: float, kind: str) -> Interval:
    """Rigorous enclosure of sin/cos over an interval."""
    w = iv.width()
    if w >= period:
        return Interval(-1.0, 1.0)

    vals = [fn(float(iv.lo)), fn(float(iv.hi))]
    lo_val = _round_down(min(vals))
    hi_val = _round_up(max(vals))

    # Check for extrema within the interval
    if kind == "sin":
        # extrema at pi/2 + k*pi
        _check_extrema = [math.pi / 2, 3 * math.pi / 2]
    else:
        _check_extrema = [0.0, math.pi]

    for base in _check_extrema:
        k_start = int(math.floor((float(iv.lo) - base) / period))
        for k in range(k_start - 1, k_start + 3):
            x = base + k * period
            if float(iv.lo) <= x <= float(iv.hi):
                v = fn(x)
                lo_val = min(lo_val, _round_down(v))
                hi_val = max(hi_val, _round_up(v))

    return Interval(max(lo_val, -1.0), min(hi_val, 1.0))


# ---------------------------------------------------------------------------
# IntervalVector
# ---------------------------------------------------------------------------

class IntervalVector:
    """Vector of intervals representing an axis-aligned box."""

    __slots__ = ("_intervals",)

    def __init__(self, intervals: Sequence[Interval]) -> None:
        self._intervals = list(intervals)

    @classmethod
    def from_midpoint_radius(cls, mid: np.ndarray, rad: np.ndarray) -> "IntervalVector":
        n = len(mid)
        ivs = []
        for i in range(n):
            ivs.append(Interval(
                _round_down(mid[i] - rad[i]),
                _round_up(mid[i] + rad[i]),
            ))
        return cls(ivs)

    @classmethod
    def from_bounds(cls, lo: np.ndarray, hi: np.ndarray) -> "IntervalVector":
        return cls([Interval(float(lo[i]), float(hi[i])) for i in range(len(lo))])

    @classmethod
    def constant(cls, values: np.ndarray) -> "IntervalVector":
        return cls([Interval(float(v)) for v in values])

    @property
    def dim(self) -> int:
        return len(self._intervals)

    def __len__(self) -> int:
        return len(self._intervals)

    def __getitem__(self, i: int) -> Interval:
        return self._intervals[i]

    def __setitem__(self, i: int, val: Interval) -> None:
        self._intervals[i] = val

    def __iter__(self):
        return iter(self._intervals)

    def midpoint(self) -> np.ndarray:
        return np.array([iv.mid() for iv in self._intervals])

    def widths(self) -> np.ndarray:
        return np.array([iv.width() for iv in self._intervals])

    def radii(self) -> np.ndarray:
        return np.array([iv.radius() for iv in self._intervals])

    def max_width(self) -> float:
        return float(max(iv.width() for iv in self._intervals))

    def lo_array(self) -> np.ndarray:
        return np.array([float(iv.lo) for iv in self._intervals])

    def hi_array(self) -> np.ndarray:
        return np.array([float(iv.hi) for iv in self._intervals])

    def contains(self, other: "IntervalVector") -> bool:
        if self.dim != other.dim:
            return False
        return all(s.contains(o) for s, o in zip(self._intervals, other._intervals))

    def overlaps(self, other: "IntervalVector") -> bool:
        if self.dim != other.dim:
            return False
        return all(s.overlaps(o) for s, o in zip(self._intervals, other._intervals))

    def hull(self, other: "IntervalVector") -> "IntervalVector":
        assert self.dim == other.dim
        return IntervalVector([hull(a, b) for a, b in zip(self._intervals, other._intervals)])

    def intersection(self, other: "IntervalVector") -> Optional["IntervalVector"]:
        assert self.dim == other.dim
        result = []
        for a, b in zip(self._intervals, other._intervals):
            inter = intersection(a, b)
            if inter is None:
                return None
            result.append(inter)
        return IntervalVector(result)

    def bloat(self, eps: float) -> "IntervalVector":
        """Enlarge each component by eps."""
        return IntervalVector([
            Interval(_round_down(float(iv.lo) - eps), _round_up(float(iv.hi) + eps))
            for iv in self._intervals
        ])

    def project(self, indices: List[int]) -> "IntervalVector":
        return IntervalVector([self._intervals[i] for i in indices])

    def __add__(self, other: "IntervalVector") -> "IntervalVector":
        assert self.dim == other.dim
        return IntervalVector([a + b for a, b in zip(self._intervals, other._intervals)])

    def __sub__(self, other: "IntervalVector") -> "IntervalVector":
        assert self.dim == other.dim
        return IntervalVector([a - b for a, b in zip(self._intervals, other._intervals)])

    def __neg__(self) -> "IntervalVector":
        return IntervalVector([-iv for iv in self._intervals])

    def scalar_mul(self, s: Interval) -> "IntervalVector":
        return IntervalVector([s * iv for iv in self._intervals])

    def copy(self) -> "IntervalVector":
        return IntervalVector([Interval(iv.lo, iv.hi) for iv in self._intervals])

    def vertices(self) -> List[np.ndarray]:
        """Enumerate all 2^n vertices of the box."""
        n = self.dim
        count = 1 << n
        verts = []
        for mask in range(count):
            v = np.empty(n)
            for j in range(n):
                v[j] = float(self._intervals[j].hi if (mask >> j) & 1 else self._intervals[j].lo)
            verts.append(v)
        return verts

    def __repr__(self) -> str:
        inner = ", ".join(str(iv) for iv in self._intervals)
        return f"IntervalVector([{inner}])"

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# IntervalMatrix
# ---------------------------------------------------------------------------

class IntervalMatrix:
    """Matrix of intervals for interval linear algebra."""

    __slots__ = ("_data", "_rows", "_cols")

    def __init__(self, data: List[List[Interval]]) -> None:
        self._data = data
        self._rows = len(data)
        self._cols = len(data[0]) if data else 0

    @classmethod
    def from_numpy(cls, mat: np.ndarray) -> "IntervalMatrix":
        rows, cols = mat.shape
        data = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(Interval(float(mat[i, j])))
            data.append(row)
        return cls(data)

    @classmethod
    def identity(cls, n: int) -> "IntervalMatrix":
        data = []
        for i in range(n):
            row = [Interval(1.0) if i == j else Interval(0.0) for j in range(n)]
            data.append(row)
        return cls(data)

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "IntervalMatrix":
        return cls([[Interval(0.0) for _ in range(cols)] for _ in range(rows)])

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    def __getitem__(self, idx: Tuple[int, int]) -> Interval:
        return self._data[idx[0]][idx[1]]

    def __setitem__(self, idx: Tuple[int, int], val: Interval) -> None:
        self._data[idx[0]][idx[1]] = val

    def midpoint_matrix(self) -> np.ndarray:
        m = np.empty((self._rows, self._cols))
        for i in range(self._rows):
            for j in range(self._cols):
                m[i, j] = self._data[i][j].mid()
        return m

    def __add__(self, other: "IntervalMatrix") -> "IntervalMatrix":
        assert self.shape == other.shape
        return IntervalMatrix([
            [self._data[i][j] + other._data[i][j] for j in range(self._cols)]
            for i in range(self._rows)
        ])

    def __sub__(self, other: "IntervalMatrix") -> "IntervalMatrix":
        assert self.shape == other.shape
        return IntervalMatrix([
            [self._data[i][j] - other._data[i][j] for j in range(self._cols)]
            for i in range(self._rows)
        ])

    def __mul__(self, other: "IntervalMatrix") -> "IntervalMatrix":
        """Matrix multiplication."""
        assert self._cols == other._rows
        result = IntervalMatrix.zeros(self._rows, other._cols)
        for i in range(self._rows):
            for j in range(other._cols):
                s = Interval(0.0)
                for k in range(self._cols):
                    s = s + self._data[i][k] * other._data[k][j]
                result._data[i][j] = s
        return result

    def mat_vec(self, v: IntervalVector) -> IntervalVector:
        """Matrix-vector product."""
        assert self._cols == v.dim
        result = []
        for i in range(self._rows):
            s = Interval(0.0)
            for j in range(self._cols):
                s = s + self._data[i][j] * v[j]
            result.append(s)
        return IntervalVector(result)

    def scalar_mul(self, s: Interval) -> "IntervalMatrix":
        return IntervalMatrix([
            [s * self._data[i][j] for j in range(self._cols)]
            for i in range(self._rows)
        ])

    def transpose(self) -> "IntervalMatrix":
        return IntervalMatrix([
            [self._data[j][i] for j in range(self._rows)]
            for i in range(self._cols)
        ])

    def spectral_radius_bound(self) -> float:
        """Upper bound on spectral radius via Gershgorin circles."""
        rho = 0.0
        for i in range(self._rows):
            center = self._data[i][i].magnitude()
            row_sum = sum(
                self._data[i][j].magnitude()
                for j in range(self._cols) if j != i
            )
            rho = max(rho, center + row_sum)
        return rho

    def max_width(self) -> float:
        w = 0.0
        for i in range(self._rows):
            for j in range(self._cols):
                w = max(w, self._data[i][j].width())
        return w

    def __repr__(self) -> str:
        rows_str = []
        for row in self._data:
            rows_str.append("[" + ", ".join(str(iv) for iv in row) + "]")
        return "IntervalMatrix([\n  " + ",\n  ".join(rows_str) + "\n])"


# ---------------------------------------------------------------------------
# Distance and subdivision utilities
# ---------------------------------------------------------------------------

def hausdorff_distance(a: IntervalVector, b: IntervalVector) -> float:
    """Hausdorff distance between two interval vectors (boxes)."""
    assert a.dim == b.dim
    d = 0.0
    for i in range(a.dim):
        d = max(d, abs(float(a[i].lo) - float(b[i].lo)))
        d = max(d, abs(float(a[i].hi) - float(b[i].hi)))
    return d


def subdivision(box: IntervalVector, n_per_dim: int = 2) -> List[IntervalVector]:
    """Subdivide a box into sub-boxes (n_per_dim divisions per dimension)."""
    dim = box.dim
    sub_intervals = [box[i].subdivide(n_per_dim) for i in range(dim)]

    # Cartesian product
    result: List[IntervalVector] = []
    counts = [n_per_dim] * dim

    def _recurse(depth: int, current: List[Interval]) -> None:
        if depth == dim:
            result.append(IntervalVector(list(current)))
            return
        for k in range(counts[depth]):
            current.append(sub_intervals[depth][k])
            _recurse(depth + 1, current)
            current.pop()

    _recurse(0, [])
    return result
