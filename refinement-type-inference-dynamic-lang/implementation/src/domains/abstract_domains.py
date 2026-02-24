"""
Abstract Interpretation Framework with Multiple Domains.

Implements a suite of abstract domains for refinement type inference:
- Interval domain with proper widening/narrowing
- Octagon domain (+-x_i +- x_j <= c constraints)
- Polyhedra domain (general linear constraints)
- String abstract domain (prefix, suffix, regex patterns)
- Container abstract domain (list/dict/set shapes)
- Reduced product of multiple domains
- Cofibered domain construction
- Abstract transformers for Python operations
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)


# ---------------------------------------------------------------------------
# Section 1: Base types and interfaces (~80 lines)
# ---------------------------------------------------------------------------

T = TypeVar("T")
D = TypeVar("D", bound="AbstractElement")


class AnalysisDirection(Enum):
    """Direction of dataflow analysis."""
    FORWARD = auto()
    BACKWARD = auto()


@runtime_checkable
class AbstractElement(Protocol):
    """Protocol for abstract domain elements."""

    def is_bottom(self) -> bool:
        """Return True if this element is the bottom (empty) element."""
        ...

    def is_top(self) -> bool:
        """Return True if this element is the top (universal) element."""
        ...

    def leq(self, other: AbstractElement) -> bool:
        """Partial order: self ⊑ other."""
        ...


@runtime_checkable
class AbstractDomain(Protocol[D]):
    """Protocol for an abstract domain with lattice operations."""

    def bottom(self) -> D:
        """Return the least element."""
        ...

    def top(self) -> D:
        """Return the greatest element."""
        ...

    def join(self, a: D, b: D) -> D:
        """Least upper bound: a ⊔ b."""
        ...

    def meet(self, a: D, b: D) -> D:
        """Greatest lower bound: a ⊓ b."""
        ...

    def widen(self, a: D, b: D) -> D:
        """Widening operator: a ▽ b."""
        ...

    def narrow(self, a: D, b: D) -> D:
        """Narrowing operator: a △ b."""
        ...

    def leq(self, a: D, b: D) -> bool:
        """Partial order check: a ⊑ b."""
        ...


@dataclass(frozen=True)
class TransferFunction:
    """Describes a transfer function applied at a program point."""
    name: str
    direction: AnalysisDirection = AnalysisDirection.FORWARD
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FixpointResult(Generic[T]):
    """Result of a fixpoint computation."""
    pre_states: Dict[int, T] = field(default_factory=dict)
    post_states: Dict[int, T] = field(default_factory=dict)
    iterations: int = 0
    converged: bool = False
    widening_points: FrozenSet[int] = field(default_factory=frozenset)

    def state_at(self, point: int, pre: bool = True) -> Optional[T]:
        """Retrieve abstract state at a program point."""
        store = self.pre_states if pre else self.post_states
        return store.get(point)


# Positive / negative infinity sentinels for interval bounds.
INF = float("inf")
NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Section 2: IntervalDomain (~180 lines)
# ---------------------------------------------------------------------------

# Standard widening thresholds.
_WIDEN_THRESHOLDS: List[float] = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]


@dataclass(frozen=True)
class Interval:
    """A closed interval [lo, hi] with ±∞ support."""
    lo: float
    hi: float

    # -- special constructors ------------------------------------------------

    @staticmethod
    def bottom() -> Interval:
        return Interval(INF, NEG_INF)

    @staticmethod
    def top() -> Interval:
        return Interval(NEG_INF, INF)

    @staticmethod
    def const(v: float) -> Interval:
        return Interval(v, v)

    @staticmethod
    def from_range(lo: float, hi: float) -> Interval:
        if lo > hi:
            return Interval.bottom()
        return Interval(lo, hi)

    # -- predicates ----------------------------------------------------------

    def is_bottom(self) -> bool:
        return self.lo > self.hi

    def is_top(self) -> bool:
        return self.lo == NEG_INF and self.hi == INF

    def is_const(self) -> bool:
        return (not self.is_bottom()) and self.lo == self.hi

    def contains(self, v: float) -> bool:
        return self.lo <= v <= self.hi

    def leq(self, other: Interval) -> bool:
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        return other.lo <= self.lo and self.hi <= other.hi

    # -- lattice ops ---------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_bottom():
            return "⊥"
        lo_s = "-∞" if self.lo == NEG_INF else str(self.lo)
        hi_s = "+∞" if self.hi == INF else str(self.hi)
        return f"[{lo_s}, {hi_s}]"


class IntervalDomain:
    """Interval abstract domain with widening and narrowing."""

    thresholds: List[float]

    def __init__(self, thresholds: Optional[List[float]] = None) -> None:
        self.thresholds = sorted(thresholds if thresholds is not None else _WIDEN_THRESHOLDS)

    # -- lattice interface ---------------------------------------------------

    def bottom(self) -> Interval:
        return Interval.bottom()

    def top(self) -> Interval:
        return Interval.top()

    def join(self, a: Interval, b: Interval) -> Interval:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        return Interval(min(a.lo, b.lo), max(a.hi, b.hi))

    def meet(self, a: Interval, b: Interval) -> Interval:
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom()
        lo = max(a.lo, b.lo)
        hi = min(a.hi, b.hi)
        return Interval.from_range(lo, hi)

    def leq(self, a: Interval, b: Interval) -> bool:
        return a.leq(b)

    def widen(self, a: Interval, b: Interval) -> Interval:
        """Widening with thresholds."""
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        new_lo = a.lo
        if b.lo < a.lo:
            new_lo = NEG_INF
            for t in self.thresholds:
                if t <= b.lo:
                    new_lo = t
                else:
                    break
            if new_lo > b.lo:
                new_lo = NEG_INF
        new_hi = a.hi
        if b.hi > a.hi:
            new_hi = INF
            for t in reversed(self.thresholds):
                if t >= b.hi:
                    new_hi = t
                else:
                    break
            if new_hi < b.hi:
                new_hi = INF
        return Interval(new_lo, new_hi)

    def narrow(self, a: Interval, b: Interval) -> Interval:
        """Narrowing: refine infinite bounds using b, also tighten finite bounds."""
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom()
        lo = a.lo if a.lo != NEG_INF else b.lo
        if lo != NEG_INF and b.lo != NEG_INF and b.lo > lo:
            lo = b.lo
        hi = a.hi if a.hi != INF else b.hi
        if hi != INF and b.hi != INF and b.hi < hi:
            hi = b.hi
        return Interval.from_range(lo, hi)

    # -- arithmetic transfer functions ---------------------------------------

    def add(self, a: Interval, b: Interval) -> Interval:
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom()
        return Interval(a.lo + b.lo, a.hi + b.hi)

    def sub(self, a: Interval, b: Interval) -> Interval:
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom()
        return Interval(a.lo - b.hi, a.hi - b.lo)

    def mul(self, a: Interval, b: Interval) -> Interval:
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom()
        products = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        finite = [p for p in products if not math.isnan(p)]
        if not finite:
            return Interval.top()
        return Interval(min(finite), max(finite))

    def div(self, a: Interval, b: Interval) -> Interval:
        """Integer-style division.  Division by an interval containing 0 yields ⊤."""
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom()
        if b.contains(0):
            return Interval.top()
        quotients: List[float] = []
        for x in (a.lo, a.hi):
            for y in (b.lo, b.hi):
                if y != 0:
                    q = x / y
                    if not math.isnan(q):
                        quotients.append(q)
        if not quotients:
            return Interval.top()
        return Interval(min(quotients), max(quotients))

    def mod(self, a: Interval, b: Interval) -> Interval:
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom()
        if b.contains(0):
            return Interval.top()
        abs_b = max(abs(b.lo), abs(b.hi))
        return Interval(0, abs_b - 1) if a.lo >= 0 else Interval(-(abs_b - 1), abs_b - 1)

    def neg(self, a: Interval) -> Interval:
        if a.is_bottom():
            return Interval.bottom()
        return Interval(-a.hi, -a.lo)

    # -- forward comparison guards -------------------------------------------

    def guard_lt(self, a: Interval, b: Interval) -> Tuple[Interval, Interval]:
        """Refine a and b under the guard a < b."""
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom(), Interval.bottom()
        a2 = Interval.from_range(a.lo, min(a.hi, b.hi - 1))
        b2 = Interval.from_range(max(b.lo, a.lo + 1), b.hi)
        return a2, b2

    def guard_le(self, a: Interval, b: Interval) -> Tuple[Interval, Interval]:
        if a.is_bottom() or b.is_bottom():
            return Interval.bottom(), Interval.bottom()
        a2 = Interval.from_range(a.lo, min(a.hi, b.hi))
        b2 = Interval.from_range(max(b.lo, a.lo), b.hi)
        return a2, b2

    def guard_eq(self, a: Interval, b: Interval) -> Tuple[Interval, Interval]:
        m = self.meet(a, b)
        return m, m

    # -- backward transfer functions -----------------------------------------

    def backward_add(self, result: Interval, a: Interval, b: Interval) -> Tuple[Interval, Interval]:
        """Given result = a + b, refine a and b."""
        a2 = self.meet(a, self.sub(result, b))
        b2 = self.meet(b, self.sub(result, a))
        return a2, b2

    def backward_sub(self, result: Interval, a: Interval, b: Interval) -> Tuple[Interval, Interval]:
        a2 = self.meet(a, self.add(result, b))
        b2 = self.meet(b, self.sub(a, result))
        return a2, b2

    def backward_mul(self, result: Interval, a: Interval, b: Interval) -> Tuple[Interval, Interval]:
        if not b.contains(0):
            a2 = self.meet(a, self.div(result, b))
        else:
            a2 = a
        if not a.contains(0):
            b2 = self.meet(b, self.div(result, a))
        else:
            b2 = b
        return a2, b2


# ---------------------------------------------------------------------------
# Section 3: OctagonDomain (~200 lines)
# ---------------------------------------------------------------------------

@dataclass
class OctagonMatrix:
    """Difference-bound matrix for octagonal constraints.

    Variables are indexed 0..n-1.  Each variable x_i has two forms:
      v+(i) = 2*i      representing +x_i
      v-(i) = 2*i + 1  representing -x_i
    The matrix m has dimension 2n x 2n.
    Entry m[a][b] encodes the constraint  v(b) - v(a) <= m[a][b].
    """
    n: int
    m: List[List[float]]
    _closed: bool = False

    @staticmethod
    def make(n: int, fill: float = INF) -> OctagonMatrix:
        dim = 2 * n
        mat = [[fill] * dim for _ in range(dim)]
        for i in range(dim):
            mat[i][i] = 0.0
        return OctagonMatrix(n=n, m=mat, _closed=False)

    @staticmethod
    def bottom(n: int) -> OctagonMatrix:
        o = OctagonMatrix.make(n, fill=INF)
        o.m[0][0] = -1.0  # intentionally inconsistent
        return o

    def is_bottom(self) -> bool:
        dim = 2 * self.n
        for i in range(dim):
            if self.m[i][i] < 0:
                return True
        return False

    def is_top(self) -> bool:
        dim = 2 * self.n
        for i in range(dim):
            for j in range(dim):
                if i != j and self.m[i][j] != INF:
                    return False
        return True

    def leq(self, other: OctagonMatrix) -> bool:
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        dim = 2 * self.n
        for i in range(dim):
            for j in range(dim):
                if self.m[i][j] > other.m[i][j]:
                    return False
        return True

    def copy(self) -> OctagonMatrix:
        dim = 2 * self.n
        new_m = [row[:] for row in self.m]
        return OctagonMatrix(n=self.n, m=new_m, _closed=self._closed)


class OctagonDomain:
    """Octagon abstract domain using difference-bound matrices."""

    def __init__(self, num_vars: int) -> None:
        self.num_vars = num_vars

    def _dim(self) -> int:
        return 2 * self.num_vars

    def bottom(self) -> OctagonMatrix:
        return OctagonMatrix.bottom(self.num_vars)

    def top(self) -> OctagonMatrix:
        return OctagonMatrix.make(self.num_vars, INF)

    # -- Floyd-Warshall strong closure ----------------------------------------

    def close(self, o: OctagonMatrix) -> OctagonMatrix:
        """Compute strong closure via Floyd-Warshall + tightening."""
        if o.is_bottom():
            return o
        r = o.copy()
        dim = self._dim()
        # Standard shortest-path closure
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    candidate = r.m[i][k] + r.m[k][j]
                    if candidate < r.m[i][j]:
                        r.m[i][j] = candidate
        # Tightening (strengthening) for octagonal constraints
        for i in range(dim):
            for j in range(dim):
                bar_i = i ^ 1  # complementary index
                bar_j = j ^ 1
                via = (r.m[i][bar_i] + r.m[bar_j][j]) / 2.0
                if via < r.m[i][j]:
                    r.m[i][j] = via
        # Check consistency
        for i in range(dim):
            if r.m[i][i] < 0:
                return self.bottom()
            r.m[i][i] = 0.0
        r._closed = True
        return r

    # -- lattice operations --------------------------------------------------

    def join(self, a: OctagonMatrix, b: OctagonMatrix) -> OctagonMatrix:
        if a.is_bottom():
            return b.copy()
        if b.is_bottom():
            return a.copy()
        r = OctagonMatrix.make(self.num_vars)
        dim = self._dim()
        for i in range(dim):
            for j in range(dim):
                r.m[i][j] = max(a.m[i][j], b.m[i][j])
        return r

    def meet(self, a: OctagonMatrix, b: OctagonMatrix) -> OctagonMatrix:
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        r = OctagonMatrix.make(self.num_vars)
        dim = self._dim()
        for i in range(dim):
            for j in range(dim):
                r.m[i][j] = min(a.m[i][j], b.m[i][j])
        return self.close(r)

    def widen(self, a: OctagonMatrix, b: OctagonMatrix) -> OctagonMatrix:
        if a.is_bottom():
            return b.copy()
        if b.is_bottom():
            return a.copy()
        r = OctagonMatrix.make(self.num_vars)
        dim = self._dim()
        for i in range(dim):
            for j in range(dim):
                if b.m[i][j] <= a.m[i][j]:
                    r.m[i][j] = a.m[i][j]
                else:
                    r.m[i][j] = INF
        return r

    def narrow(self, a: OctagonMatrix, b: OctagonMatrix) -> OctagonMatrix:
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        r = OctagonMatrix.make(self.num_vars)
        dim = self._dim()
        for i in range(dim):
            for j in range(dim):
                if a.m[i][j] == INF:
                    r.m[i][j] = b.m[i][j]
                else:
                    r.m[i][j] = a.m[i][j]
        return r

    def leq(self, a: OctagonMatrix, b: OctagonMatrix) -> bool:
        return a.leq(b)

    # -- transfer functions --------------------------------------------------

    def _forget_var(self, o: OctagonMatrix, var: int) -> OctagonMatrix:
        """Remove all constraints involving *var*."""
        r = o.copy()
        dim = self._dim()
        vi, vj = 2 * var, 2 * var + 1
        for k in range(dim):
            if k in (vi, vj):
                continue
            r.m[vi][k] = INF
            r.m[vj][k] = INF
            r.m[k][vi] = INF
            r.m[k][vj] = INF
        r.m[vi][vj] = INF
        r.m[vj][vi] = INF
        r.m[vi][vi] = 0.0
        r.m[vj][vj] = 0.0
        r._closed = False
        return r

    def assign_const(self, o: OctagonMatrix, var: int, val: float) -> OctagonMatrix:
        """x_var := val."""
        r = self._forget_var(o, var)
        vp, vn = 2 * var, 2 * var + 1
        r.m[vp][vn] = 2 * val    # x_var - (-x_var) <= 2*val  =>  x_var <= val
        r.m[vn][vp] = -2 * val   # -x_var - x_var <= -2*val => x_var >= val
        return self.close(r)

    def assign_var(self, o: OctagonMatrix, dst: int, src: int) -> OctagonMatrix:
        """x_dst := x_src."""
        r = self._forget_var(o, dst)
        dp, dn = 2 * dst, 2 * dst + 1
        sp, sn = 2 * src, 2 * src + 1
        r.m[dp][sp] = 0.0   # x_dst - x_src <= 0
        r.m[sp][dp] = 0.0   # x_src - x_dst <= 0
        r.m[dn][sn] = 0.0
        r.m[sn][dn] = 0.0
        return self.close(r)

    def guard_leq(self, o: OctagonMatrix, i: int, j: int) -> OctagonMatrix:
        """Add guard x_i <= x_j."""
        if o.is_bottom():
            return o
        r = o.copy()
        ip, jn = 2 * i, 2 * j + 1
        jp = 2 * j
        # x_i - x_j <= 0
        r.m[jp][ip] = min(r.m[jp][ip], 0.0)
        r.m[ip][jp] = min(r.m[ip][jp], 0.0)  # not standard; keep symmetric
        return self.close(r)

    def guard_lt(self, o: OctagonMatrix, i: int, j: int) -> OctagonMatrix:
        """Add guard x_i < x_j  (encoded as x_i - x_j <= -1 for integers)."""
        if o.is_bottom():
            return o
        r = o.copy()
        ip, jp = 2 * i, 2 * j
        r.m[jp][ip] = min(r.m[jp][ip], -1.0)
        return self.close(r)

    def project(self, o: OctagonMatrix, var: int) -> Interval:
        """Project octagon onto a single variable, yielding an interval."""
        if o.is_bottom():
            return Interval.bottom()
        vp, vn = 2 * var, 2 * var + 1
        hi = o.m[vp][vn] / 2.0 if o.m[vp][vn] != INF else INF
        lo = -o.m[vn][vp] / 2.0 if o.m[vn][vp] != INF else NEG_INF
        return Interval.from_range(lo, hi)

    def add_variable(self) -> int:
        """Increase dimensionality by one and return the new variable index."""
        new_var = self.num_vars
        self.num_vars += 1
        return new_var

    def remove_variable(self, o: OctagonMatrix, var: int) -> OctagonMatrix:
        """Remove variable *var* from the octagon (project and shrink)."""
        return self._forget_var(o, var)


# ---------------------------------------------------------------------------
# Section 4: PolyhedraDomain (~200 lines)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearConstraint:
    """Represents a1*x1 + a2*x2 + ... + an*xn <= b.

    *coeffs* maps variable index to coefficient; *bound* is the RHS constant.
    """
    coeffs: Dict[int, float]
    bound: float

    def evaluate(self, point: Dict[int, float]) -> float:
        return sum(c * point.get(v, 0.0) for v, c in self.coeffs.items())

    def satisfied(self, point: Dict[int, float]) -> bool:
        return self.evaluate(point) <= self.bound + 1e-9

    def negate(self) -> LinearConstraint:
        """Return the constraint representing -(lhs) <= -bound, i.e., lhs >= bound."""
        return LinearConstraint(
            coeffs={v: -c for v, c in self.coeffs.items()},
            bound=-self.bound,
        )

    def substitute(self, var: int, replacement_coeffs: Dict[int, float], replacement_const: float) -> LinearConstraint:
        """Replace variable *var* with an affine expression."""
        if var not in self.coeffs:
            return self
        a = self.coeffs[var]
        new_coeffs: Dict[int, float] = {}
        for v, c in self.coeffs.items():
            if v == var:
                continue
            new_coeffs[v] = new_coeffs.get(v, 0.0) + c
        for v, c in replacement_coeffs.items():
            new_coeffs[v] = new_coeffs.get(v, 0.0) + a * c
        new_bound = self.bound - a * replacement_const
        # Remove zero coefficients
        new_coeffs = {v: c for v, c in new_coeffs.items() if abs(c) > 1e-12}
        return LinearConstraint(coeffs=new_coeffs, bound=new_bound)

    def __repr__(self) -> str:
        terms = []
        for v in sorted(self.coeffs):
            c = self.coeffs[v]
            terms.append(f"{c}*x{v}")
        lhs = " + ".join(terms) if terms else "0"
        return f"{lhs} <= {self.bound}"


@dataclass
class Polyhedron:
    """A convex polyhedron represented as a conjunction of linear constraints."""
    constraints: List[LinearConstraint]
    _is_bottom: bool = False
    num_vars: int = 0

    @staticmethod
    def bottom(num_vars: int = 0) -> Polyhedron:
        return Polyhedron(constraints=[], _is_bottom=True, num_vars=num_vars)

    @staticmethod
    def top(num_vars: int = 0) -> Polyhedron:
        return Polyhedron(constraints=[], _is_bottom=False, num_vars=num_vars)

    def is_bottom(self) -> bool:
        return self._is_bottom

    def is_top(self) -> bool:
        return (not self._is_bottom) and len(self.constraints) == 0

    def leq(self, other: Polyhedron) -> bool:
        if self._is_bottom:
            return True
        if other._is_bottom:
            return False
        if other.is_top():
            return True
        # Approximate: every constraint in other must be implied.
        # We do a simple redundancy check, not a full LP.
        return all(c in self.constraints for c in other.constraints)

    def copy(self) -> Polyhedron:
        return Polyhedron(
            constraints=list(self.constraints),
            _is_bottom=self._is_bottom,
            num_vars=self.num_vars,
        )


class PolyhedraDomain:
    """Polyhedra abstract domain with Fourier-Motzkin projection."""

    def __init__(self, num_vars: int) -> None:
        self.num_vars = num_vars

    def bottom(self) -> Polyhedron:
        return Polyhedron.bottom(self.num_vars)

    def top(self) -> Polyhedron:
        return Polyhedron.top(self.num_vars)

    def leq(self, a: Polyhedron, b: Polyhedron) -> bool:
        return a.leq(b)

    # -- Fourier-Motzkin elimination -----------------------------------------

    def _fourier_motzkin(self, constraints: List[LinearConstraint], var: int) -> List[LinearConstraint]:
        """Project out *var* via Fourier-Motzkin elimination."""
        pos: List[LinearConstraint] = []
        neg: List[LinearConstraint] = []
        zero: List[LinearConstraint] = []
        for c in constraints:
            coeff = c.coeffs.get(var, 0.0)
            if abs(coeff) < 1e-12:
                zero.append(c)
            elif coeff > 0:
                pos.append(c)
            else:
                neg.append(c)
        result = list(zero)
        for p in pos:
            for n in neg:
                cp = p.coeffs[var]
                cn = -n.coeffs[var]  # make positive
                new_coeffs: Dict[int, float] = {}
                for v, c in p.coeffs.items():
                    if v == var:
                        continue
                    new_coeffs[v] = new_coeffs.get(v, 0.0) + c * cn
                for v, c in n.coeffs.items():
                    if v == var:
                        continue
                    new_coeffs[v] = new_coeffs.get(v, 0.0) + c * cp
                new_bound = p.bound * cn + n.bound * cp
                # Normalise
                divisor = cp * cn
                if abs(divisor) > 1e-12:
                    new_coeffs = {v: c / divisor for v, c in new_coeffs.items()}
                    new_bound /= divisor
                new_coeffs = {v: c for v, c in new_coeffs.items() if abs(c) > 1e-12}
                result.append(LinearConstraint(coeffs=new_coeffs, bound=new_bound))
        return result

    def project(self, p: Polyhedron, var: int) -> Polyhedron:
        if p.is_bottom():
            return self.bottom()
        new_cs = self._fourier_motzkin(p.constraints, var)
        return Polyhedron(constraints=new_cs, num_vars=self.num_vars)

    # -- lattice operations --------------------------------------------------

    def meet(self, a: Polyhedron, b: Polyhedron) -> Polyhedron:
        """Meet = intersection = union of constraint sets."""
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        combined = list(a.constraints) + [c for c in b.constraints if c not in a.constraints]
        return Polyhedron(constraints=combined, num_vars=self.num_vars)

    def join(self, a: Polyhedron, b: Polyhedron) -> Polyhedron:
        """Join = convex hull.  Approximated by keeping constraints valid in both."""
        if a.is_bottom():
            return b.copy()
        if b.is_bottom():
            return a.copy()
        # Keep constraints from *a* that are redundant w.r.t. *b* (i.e., valid in b too).
        # This is an over-approximation of the true convex hull.
        kept: List[LinearConstraint] = []
        for c in a.constraints:
            # Heuristic: we keep the constraint with relaxed bound
            if c in b.constraints:
                kept.append(c)
            else:
                # Relax bound to accommodate b — we'd need an LP; here we just drop.
                pass
        for c in b.constraints:
            if c not in a.constraints and c not in kept:
                pass  # similarly drop
        return Polyhedron(constraints=kept, num_vars=self.num_vars)

    def widen(self, a: Polyhedron, b: Polyhedron) -> Polyhedron:
        """Widening: keep constraints of *a* that are still satisfied by *b*."""
        if a.is_bottom():
            return b.copy()
        if b.is_bottom():
            return a.copy()
        kept: List[LinearConstraint] = []
        for c in a.constraints:
            if c in b.constraints:
                kept.append(c)
        return Polyhedron(constraints=kept, num_vars=self.num_vars)

    def narrow(self, a: Polyhedron, b: Polyhedron) -> Polyhedron:
        """Narrowing: add constraints from *b* that don't contradict *a*."""
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        added = list(a.constraints)
        for c in b.constraints:
            if c not in added:
                added.append(c)
        return Polyhedron(constraints=added, num_vars=self.num_vars)

    # -- transfer functions --------------------------------------------------

    def assign_linear(
        self,
        p: Polyhedron,
        var: int,
        coeffs: Dict[int, float],
        const: float,
    ) -> Polyhedron:
        """x_var := coeffs · x + const.  Substitute in constraints then close."""
        if p.is_bottom():
            return self.bottom()
        new_cs = [c.substitute(var, coeffs, const) for c in p.constraints]
        return Polyhedron(constraints=new_cs, num_vars=self.num_vars)

    def add_constraint(self, p: Polyhedron, c: LinearConstraint) -> Polyhedron:
        if p.is_bottom():
            return self.bottom()
        return Polyhedron(constraints=p.constraints + [c], num_vars=self.num_vars)

    def bound_variable(self, p: Polyhedron, var: int) -> Interval:
        """Extract interval bounds for *var* from constraints."""
        if p.is_bottom():
            return Interval.bottom()
        lo: float = NEG_INF
        hi: float = INF
        for c in p.constraints:
            keys = set(c.coeffs.keys())
            if keys == {var}:
                a = c.coeffs[var]
                if a > 0:
                    hi = min(hi, c.bound / a)
                elif a < 0:
                    lo = max(lo, c.bound / a)
        return Interval.from_range(lo, hi)


# ---------------------------------------------------------------------------
# Section 5: StringDomain (~150 lines)
# ---------------------------------------------------------------------------

@dataclass
class StringAbstract:
    """Abstract representation of a string value."""
    prefix: Optional[str] = None          # known prefix
    suffix: Optional[str] = None          # known suffix
    contained: FrozenSet[str] = field(default_factory=frozenset)  # known substrings
    length: Interval = field(default_factory=Interval.top)
    charset: Optional[FrozenSet[str]] = None  # set of possible characters
    pattern: Optional[str] = None         # regex pattern that all concrete values match
    _is_bottom: bool = False

    @staticmethod
    def bottom() -> StringAbstract:
        return StringAbstract(_is_bottom=True)

    @staticmethod
    def top() -> StringAbstract:
        return StringAbstract()

    @staticmethod
    def const(s: str) -> StringAbstract:
        return StringAbstract(
            prefix=s,
            suffix=s,
            contained=frozenset({s}),
            length=Interval.const(len(s)),
            charset=frozenset(s),
            pattern=re.escape(s),
        )

    def is_bottom(self) -> bool:
        return self._is_bottom

    def is_top(self) -> bool:
        return (
            not self._is_bottom
            and self.prefix is None
            and self.suffix is None
            and len(self.contained) == 0
            and self.length.is_top()
            and self.charset is None
            and self.pattern is None
        )

    def leq(self, other: StringAbstract) -> bool:
        if self._is_bottom:
            return True
        if other._is_bottom:
            return False
        if other.is_top():
            return True
        return False  # conservative


def _common_prefix(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None or b is None:
        return None
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i] if i > 0 else None


def _common_suffix(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None or b is None:
        return None
    i = 0
    while i < len(a) and i < len(b) and a[-(i + 1)] == b[-(i + 1)]:
        i += 1
    return a[-i:] if i > 0 else None


class StringDomain:
    """Abstract domain for string values."""

    _iv = IntervalDomain()

    def bottom(self) -> StringAbstract:
        return StringAbstract.bottom()

    def top(self) -> StringAbstract:
        return StringAbstract.top()

    def join(self, a: StringAbstract, b: StringAbstract) -> StringAbstract:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        return StringAbstract(
            prefix=_common_prefix(a.prefix, b.prefix),
            suffix=_common_suffix(a.suffix, b.suffix),
            contained=a.contained & b.contained,
            length=self._iv.join(a.length, b.length),
            charset=(a.charset | b.charset) if a.charset is not None and b.charset is not None else None,
            pattern=None,  # conservative
        )

    def meet(self, a: StringAbstract, b: StringAbstract) -> StringAbstract:
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        prefix = a.prefix if a.prefix and (b.prefix is None or a.prefix.startswith(b.prefix)) else b.prefix
        suffix = a.suffix if a.suffix and (b.suffix is None or a.suffix.endswith(b.suffix)) else b.suffix
        return StringAbstract(
            prefix=prefix,
            suffix=suffix,
            contained=a.contained | b.contained,
            length=self._iv.meet(a.length, b.length),
            charset=(a.charset & b.charset) if a.charset is not None and b.charset is not None else (a.charset or b.charset),
            pattern=a.pattern if a.pattern else b.pattern,
        )

    def widen(self, a: StringAbstract, b: StringAbstract) -> StringAbstract:
        if a.is_bottom():
            return b
        return StringAbstract(
            prefix=_common_prefix(a.prefix, b.prefix),
            suffix=_common_suffix(a.suffix, b.suffix),
            contained=a.contained & b.contained,
            length=self._iv.widen(a.length, b.length),
            charset=None,
            pattern=None,
        )

    def narrow(self, a: StringAbstract, b: StringAbstract) -> StringAbstract:
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        return StringAbstract(
            prefix=a.prefix if a.prefix else b.prefix,
            suffix=a.suffix if a.suffix else b.suffix,
            contained=a.contained | b.contained,
            length=self._iv.narrow(a.length, b.length),
            charset=b.charset if a.charset is None else a.charset,
            pattern=b.pattern if a.pattern is None else a.pattern,
        )

    def leq(self, a: StringAbstract, b: StringAbstract) -> bool:
        return a.leq(b)

    # -- string operations ---------------------------------------------------

    def concat(self, a: StringAbstract, b: StringAbstract) -> StringAbstract:
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        prefix = a.prefix  # prefix of concat is prefix of a (if a has known prefix)
        suffix = b.suffix  # suffix of concat is suffix of b
        new_contained = a.contained | b.contained
        new_length = self._iv.add(a.length, b.length)
        charset: Optional[FrozenSet[str]] = None
        if a.charset is not None and b.charset is not None:
            charset = a.charset | b.charset
        return StringAbstract(
            prefix=prefix, suffix=suffix, contained=new_contained,
            length=new_length, charset=charset,
        )

    def slice(self, s: StringAbstract, lo_idx: Interval, hi_idx: Interval) -> StringAbstract:
        if s.is_bottom():
            return self.bottom()
        new_len = self._iv.sub(hi_idx, lo_idx)
        new_len = self._iv.meet(new_len, Interval.from_range(0, INF))
        return StringAbstract(length=new_len, charset=s.charset)

    def upper(self, s: StringAbstract) -> StringAbstract:
        if s.is_bottom():
            return self.bottom()
        return StringAbstract(
            prefix=s.prefix.upper() if s.prefix else None,
            suffix=s.suffix.upper() if s.suffix else None,
            length=s.length,
            charset=frozenset(c.upper() for c in s.charset) if s.charset else None,
        )

    def lower(self, s: StringAbstract) -> StringAbstract:
        if s.is_bottom():
            return self.bottom()
        return StringAbstract(
            prefix=s.prefix.lower() if s.prefix else None,
            suffix=s.suffix.lower() if s.suffix else None,
            length=s.length,
            charset=frozenset(c.lower() for c in s.charset) if s.charset else None,
        )

    def strip(self, s: StringAbstract) -> StringAbstract:
        if s.is_bottom():
            return self.bottom()
        return StringAbstract(
            contained=s.contained,
            length=Interval.from_range(0, s.length.hi),
            charset=s.charset,
        )

    def split(self, s: StringAbstract, _sep: Optional[StringAbstract] = None) -> StringAbstract:
        """Returns abstract description of an *element* of the split result."""
        if s.is_bottom():
            return self.bottom()
        return StringAbstract(
            length=Interval.from_range(0, s.length.hi),
            charset=s.charset,
        )

    def replace(self, s: StringAbstract, _old: StringAbstract, _new: StringAbstract) -> StringAbstract:
        if s.is_bottom():
            return self.bottom()
        return StringAbstract(
            length=Interval.from_range(0, INF),
            charset=(s.charset | _new.charset) if s.charset and _new.charset else None,
        )

    def format_str(self, fmt: StringAbstract, args: Sequence[StringAbstract]) -> StringAbstract:
        """Approximate result of str.format / f-string."""
        if fmt.is_bottom():
            return self.bottom()
        all_charset: Optional[FrozenSet[str]] = fmt.charset
        for a in args:
            if a.charset is not None and all_charset is not None:
                all_charset = all_charset | a.charset
            else:
                all_charset = None
                break
        return StringAbstract(length=Interval.from_range(0, INF), charset=all_charset)


# ---------------------------------------------------------------------------
# Section 6: ContainerDomain (~150 lines)
# ---------------------------------------------------------------------------

class ContainerKind(Enum):
    LIST = auto()
    DICT = auto()
    SET = auto()
    TUPLE = auto()


@dataclass
class ContainerAbstract:
    """Abstract representation of a container (list, dict, set, tuple)."""
    kind: ContainerKind
    element_type: Optional[str] = None          # abstract type tag for elements
    key_type: Optional[str] = None              # for dicts
    value_type: Optional[str] = None            # for dicts
    length: Interval = field(default_factory=Interval.top)
    known_indices: Dict[int, Any] = field(default_factory=dict)     # for lists/tuples
    known_keys: Dict[str, Any] = field(default_factory=dict)        # for dicts
    _is_bottom: bool = False

    @staticmethod
    def bottom(kind: ContainerKind = ContainerKind.LIST) -> ContainerAbstract:
        return ContainerAbstract(kind=kind, _is_bottom=True)

    @staticmethod
    def top_list() -> ContainerAbstract:
        return ContainerAbstract(kind=ContainerKind.LIST)

    @staticmethod
    def top_dict() -> ContainerAbstract:
        return ContainerAbstract(kind=ContainerKind.DICT)

    @staticmethod
    def top_set() -> ContainerAbstract:
        return ContainerAbstract(kind=ContainerKind.SET)

    def is_bottom(self) -> bool:
        return self._is_bottom

    def is_top(self) -> bool:
        return (
            not self._is_bottom
            and self.element_type is None
            and self.key_type is None
            and self.value_type is None
            and self.length.is_top()
            and len(self.known_indices) == 0
            and len(self.known_keys) == 0
        )

    def leq(self, other: ContainerAbstract) -> bool:
        if self._is_bottom:
            return True
        if other._is_bottom:
            return False
        if other.is_top():
            return True
        return self.length.leq(other.length)


class ContainerDomain:
    """Abstract domain for Python container values."""

    _iv = IntervalDomain()

    def bottom(self, kind: ContainerKind = ContainerKind.LIST) -> ContainerAbstract:
        return ContainerAbstract.bottom(kind)

    def top(self, kind: ContainerKind = ContainerKind.LIST) -> ContainerAbstract:
        return ContainerAbstract(kind=kind)

    def join(self, a: ContainerAbstract, b: ContainerAbstract) -> ContainerAbstract:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        if a.kind != b.kind:
            return ContainerAbstract(kind=a.kind)  # lose precision on kind mismatch
        # Merge element types
        elem = a.element_type if a.element_type == b.element_type else None
        kt = a.key_type if a.key_type == b.key_type else None
        vt = a.value_type if a.value_type == b.value_type else None
        # Keep only common known indices
        common_idx = {k: a.known_indices[k] for k in a.known_indices if k in b.known_indices and a.known_indices[k] == b.known_indices[k]}
        common_keys = {k: a.known_keys[k] for k in a.known_keys if k in b.known_keys and a.known_keys[k] == b.known_keys[k]}
        return ContainerAbstract(
            kind=a.kind,
            element_type=elem,
            key_type=kt,
            value_type=vt,
            length=self._iv.join(a.length, b.length),
            known_indices=common_idx,
            known_keys=common_keys,
        )

    def meet(self, a: ContainerAbstract, b: ContainerAbstract) -> ContainerAbstract:
        if a.is_bottom() or b.is_bottom():
            return self.bottom(a.kind)
        if a.kind != b.kind:
            return self.bottom(a.kind)
        elem = a.element_type or b.element_type
        kt = a.key_type or b.key_type
        vt = a.value_type or b.value_type
        merged_idx = {**a.known_indices, **b.known_indices}
        merged_keys = {**a.known_keys, **b.known_keys}
        return ContainerAbstract(
            kind=a.kind,
            element_type=elem,
            key_type=kt,
            value_type=vt,
            length=self._iv.meet(a.length, b.length),
            known_indices=merged_idx,
            known_keys=merged_keys,
        )

    def widen(self, a: ContainerAbstract, b: ContainerAbstract) -> ContainerAbstract:
        if a.is_bottom():
            return b
        j = self.join(a, b)
        return ContainerAbstract(
            kind=j.kind,
            element_type=j.element_type,
            key_type=j.key_type,
            value_type=j.value_type,
            length=self._iv.widen(a.length, b.length),
            known_indices={},
            known_keys={},
        )

    def narrow(self, a: ContainerAbstract, b: ContainerAbstract) -> ContainerAbstract:
        if a.is_bottom() or b.is_bottom():
            return self.bottom(a.kind)
        return ContainerAbstract(
            kind=a.kind,
            element_type=a.element_type or b.element_type,
            key_type=a.key_type or b.key_type,
            value_type=a.value_type or b.value_type,
            length=self._iv.narrow(a.length, b.length),
            known_indices={**a.known_indices, **b.known_indices},
            known_keys={**a.known_keys, **b.known_keys},
        )

    def leq(self, a: ContainerAbstract, b: ContainerAbstract) -> bool:
        return a.leq(b)

    # -- container operations -----------------------------------------------

    def append(self, c: ContainerAbstract, elem_type: Optional[str] = None) -> ContainerAbstract:
        if c.is_bottom():
            return c
        new_len = self._iv.add(c.length, Interval.const(1))
        et = c.element_type if c.element_type == elem_type else None
        return ContainerAbstract(
            kind=c.kind, element_type=et, length=new_len,
            known_indices=dict(c.known_indices), known_keys=dict(c.known_keys),
        )

    def insert(self, c: ContainerAbstract, idx: int, elem_type: Optional[str] = None) -> ContainerAbstract:
        if c.is_bottom():
            return c
        new_len = self._iv.add(c.length, Interval.const(1))
        # Shift known indices after idx
        new_known: Dict[int, Any] = {}
        for k, v in c.known_indices.items():
            if k < idx:
                new_known[k] = v
            else:
                new_known[k + 1] = v
        et = c.element_type if c.element_type == elem_type else None
        return ContainerAbstract(
            kind=c.kind, element_type=et, length=new_len,
            known_indices=new_known, known_keys=dict(c.known_keys),
        )

    def pop(self, c: ContainerAbstract, idx: int = -1) -> Tuple[ContainerAbstract, Optional[str]]:
        """Pop element; return (new container, element type)."""
        if c.is_bottom():
            return c, None
        new_len = self._iv.sub(c.length, Interval.const(1))
        new_len = self._iv.meet(new_len, Interval.from_range(0, INF))
        popped_type = c.element_type
        new_known = {k: v for k, v in c.known_indices.items() if k != idx and k != (idx % int(c.length.hi) if c.length.hi != INF else idx)}
        return ContainerAbstract(
            kind=c.kind, element_type=c.element_type, length=new_len,
            known_indices=new_known, known_keys=dict(c.known_keys),
        ), popped_type

    def getitem(self, c: ContainerAbstract, key: Any) -> Optional[str]:
        """Return the abstract type of the element at *key*."""
        if c.is_bottom():
            return None
        if c.kind == ContainerKind.DICT:
            if isinstance(key, str) and key in c.known_keys:
                return str(c.known_keys[key])
            return c.value_type
        if isinstance(key, int) and key in c.known_indices:
            return str(c.known_indices[key])
        return c.element_type

    def setitem(self, c: ContainerAbstract, key: Any, val_type: Optional[str] = None) -> ContainerAbstract:
        if c.is_bottom():
            return c
        new_c = ContainerAbstract(
            kind=c.kind,
            element_type=c.element_type,
            key_type=c.key_type,
            value_type=c.value_type,
            length=c.length,
            known_indices=dict(c.known_indices),
            known_keys=dict(c.known_keys),
        )
        if c.kind == ContainerKind.DICT and isinstance(key, str):
            new_c.known_keys[key] = val_type
        elif isinstance(key, int):
            new_c.known_indices[key] = val_type
        return new_c

    def delitem(self, c: ContainerAbstract, key: Any) -> ContainerAbstract:
        if c.is_bottom():
            return c
        new_len = self._iv.sub(c.length, Interval.const(1))
        new_len = self._iv.meet(new_len, Interval.from_range(0, INF))
        new_known_idx = {k: v for k, v in c.known_indices.items() if k != key}
        new_known_keys = {k: v for k, v in c.known_keys.items() if k != key}
        return ContainerAbstract(
            kind=c.kind, element_type=c.element_type,
            key_type=c.key_type, value_type=c.value_type,
            length=new_len,
            known_indices=new_known_idx, known_keys=new_known_keys,
        )

    def update(self, c: ContainerAbstract, other: ContainerAbstract) -> ContainerAbstract:
        """Merge another container into this one (like dict.update or set union)."""
        if c.is_bottom():
            return other
        if other.is_bottom():
            return c
        merged_keys = {**c.known_keys, **other.known_keys}
        return ContainerAbstract(
            kind=c.kind,
            element_type=c.element_type if c.element_type == other.element_type else None,
            key_type=c.key_type if c.key_type == other.key_type else None,
            value_type=c.value_type if c.value_type == other.value_type else None,
            length=self._iv.add(c.length, other.length),
            known_indices={},
            known_keys=merged_keys,
        )


# ---------------------------------------------------------------------------
# Section 7: ReducedProduct (~120 lines)
# ---------------------------------------------------------------------------

@dataclass
class ProductElement:
    """Element of a reduced product of abstract domains."""
    components: Dict[str, Any]  # domain-name → abstract element
    _is_bottom: bool = False

    @staticmethod
    def bottom(names: Sequence[str]) -> ProductElement:
        return ProductElement(components={n: None for n in names}, _is_bottom=True)

    def is_bottom(self) -> bool:
        if self._is_bottom:
            return True
        return any(
            hasattr(v, "is_bottom") and v.is_bottom()
            for v in self.components.values()
            if v is not None
        )

    def is_top(self) -> bool:
        return all(
            hasattr(v, "is_top") and v.is_top()
            for v in self.components.values()
            if v is not None
        )

    def leq(self, other: ProductElement) -> bool:
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        for name, val in self.components.items():
            o_val = other.components.get(name)
            if val is not None and o_val is not None and hasattr(val, "leq"):
                if not val.leq(o_val):
                    return False
        return True

    def get(self, name: str) -> Any:
        return self.components.get(name)

    def with_component(self, name: str, value: Any) -> ProductElement:
        new_comps = dict(self.components)
        new_comps[name] = value
        return ProductElement(components=new_comps)


ReductionFunc = Callable[[ProductElement], ProductElement]


class ReducedProduct:
    """Reduced product of multiple abstract domains.

    *domains* maps a name to an ``AbstractDomain``-like object.
    *reductions* is a list of reduction functions that propagate information
    between domain components after each lattice operation.
    """

    def __init__(
        self,
        domains: Dict[str, Any],
        reductions: Optional[List[ReductionFunc]] = None,
    ) -> None:
        self.domains = domains
        self.names = list(domains.keys())
        self.reductions: List[ReductionFunc] = reductions or []

    def _reduce(self, elem: ProductElement) -> ProductElement:
        """Apply all reduction rules until stable."""
        prev = elem
        for _ in range(10):  # bounded iteration
            for r in self.reductions:
                elem = r(elem)
            if elem.components == prev.components:
                break
            prev = elem
        return elem

    def bottom(self) -> ProductElement:
        return ProductElement.bottom(self.names)

    def top(self) -> ProductElement:
        comps: Dict[str, Any] = {}
        for name, dom in self.domains.items():
            comps[name] = dom.top() if hasattr(dom, "top") else None
        return ProductElement(components=comps)

    def _apply_pointwise(self, op: str, a: ProductElement, b: ProductElement) -> ProductElement:
        if a.is_bottom() and op in ("join",):
            return b
        if b.is_bottom() and op in ("join",):
            return a
        if a.is_bottom() or b.is_bottom():
            if op in ("meet", "narrow"):
                return self.bottom()
        comps: Dict[str, Any] = {}
        for name, dom in self.domains.items():
            av = a.components.get(name)
            bv = b.components.get(name)
            if av is None or bv is None:
                comps[name] = av if bv is None else bv
            elif hasattr(dom, op):
                comps[name] = getattr(dom, op)(av, bv)
            else:
                comps[name] = av
        return self._reduce(ProductElement(components=comps))

    def join(self, a: ProductElement, b: ProductElement) -> ProductElement:
        return self._apply_pointwise("join", a, b)

    def meet(self, a: ProductElement, b: ProductElement) -> ProductElement:
        return self._apply_pointwise("meet", a, b)

    def widen(self, a: ProductElement, b: ProductElement) -> ProductElement:
        return self._apply_pointwise("widen", a, b)

    def narrow(self, a: ProductElement, b: ProductElement) -> ProductElement:
        return self._apply_pointwise("narrow", a, b)

    def leq(self, a: ProductElement, b: ProductElement) -> bool:
        return a.leq(b)


# ---------------------------------------------------------------------------
# Section 8: CofiberedDomain (~80 lines)
# ---------------------------------------------------------------------------

@dataclass
class CofiberedElement:
    """Element of a cofibered domain: a union of (type-tag, abstract value) pairs."""
    entries: Dict[str, Any]  # type-tag → abstract value in corresponding domain
    _is_bottom: bool = False

    @staticmethod
    def bottom() -> CofiberedElement:
        return CofiberedElement(entries={}, _is_bottom=True)

    @staticmethod
    def singleton(tag: str, value: Any) -> CofiberedElement:
        return CofiberedElement(entries={tag: value})

    def is_bottom(self) -> bool:
        if self._is_bottom:
            return True
        return len(self.entries) == 0

    def is_top(self) -> bool:
        return all(
            hasattr(v, "is_top") and v.is_top()
            for v in self.entries.values()
        )

    def leq(self, other: CofiberedElement) -> bool:
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        for tag, val in self.entries.items():
            o_val = other.entries.get(tag)
            if o_val is None:
                return False
            if hasattr(val, "leq") and not val.leq(o_val):
                return False
        return True

    def tags(self) -> Set[str]:
        return set(self.entries.keys())

    def get(self, tag: str) -> Any:
        return self.entries.get(tag)


class CofiberedDomain:
    """Domain parameterised by type tag.

    Each type tag (e.g., ``"int"``, ``"str"``, ``"list"``) is mapped to a
    dedicated abstract domain.  Join across different tags produces a union.
    """

    def __init__(self, fiber_domains: Dict[str, Any]) -> None:
        self.fiber_domains: Dict[str, Any] = fiber_domains

    def bottom(self) -> CofiberedElement:
        return CofiberedElement.bottom()

    def top(self) -> CofiberedElement:
        entries: Dict[str, Any] = {}
        for tag, dom in self.fiber_domains.items():
            entries[tag] = dom.top() if hasattr(dom, "top") else None
        return CofiberedElement(entries=entries)

    def join(self, a: CofiberedElement, b: CofiberedElement) -> CofiberedElement:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        merged: Dict[str, Any] = {}
        all_tags = a.tags() | b.tags()
        for tag in all_tags:
            av = a.entries.get(tag)
            bv = b.entries.get(tag)
            dom = self.fiber_domains.get(tag)
            if av is None:
                merged[tag] = bv
            elif bv is None:
                merged[tag] = av
            elif dom is not None and hasattr(dom, "join"):
                merged[tag] = dom.join(av, bv)
            else:
                merged[tag] = av
        return CofiberedElement(entries=merged)

    def meet(self, a: CofiberedElement, b: CofiberedElement) -> CofiberedElement:
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        merged: Dict[str, Any] = {}
        common_tags = a.tags() & b.tags()
        for tag in common_tags:
            dom = self.fiber_domains.get(tag)
            av, bv = a.entries[tag], b.entries[tag]
            if dom is not None and hasattr(dom, "meet"):
                merged[tag] = dom.meet(av, bv)
            else:
                merged[tag] = av
        if not merged:
            return self.bottom()
        return CofiberedElement(entries=merged)

    def widen(self, a: CofiberedElement, b: CofiberedElement) -> CofiberedElement:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        merged: Dict[str, Any] = {}
        all_tags = a.tags() | b.tags()
        for tag in all_tags:
            av = a.entries.get(tag)
            bv = b.entries.get(tag)
            dom = self.fiber_domains.get(tag)
            if av is None:
                merged[tag] = bv
            elif bv is None:
                merged[tag] = av
            elif dom is not None and hasattr(dom, "widen"):
                merged[tag] = dom.widen(av, bv)
            else:
                merged[tag] = av
        return CofiberedElement(entries=merged)

    def narrow(self, a: CofiberedElement, b: CofiberedElement) -> CofiberedElement:
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        merged: Dict[str, Any] = {}
        for tag in a.tags():
            av = a.entries[tag]
            bv = b.entries.get(tag)
            dom = self.fiber_domains.get(tag)
            if bv is not None and dom is not None and hasattr(dom, "narrow"):
                merged[tag] = dom.narrow(av, bv)
            else:
                merged[tag] = av
        return CofiberedElement(entries=merged)

    def leq(self, a: CofiberedElement, b: CofiberedElement) -> bool:
        return a.leq(b)

    def dispatch(self, tag: str, op: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch an operation to the domain for *tag*."""
        dom = self.fiber_domains.get(tag)
        if dom is None:
            return None
        fn = getattr(dom, op, None)
        if fn is None:
            return None
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Section 9: PythonTransferFunctions (~150 lines)
# ---------------------------------------------------------------------------

@dataclass
class AbstractState:
    """Maps variable names to cofibered abstract values."""
    env: Dict[str, CofiberedElement] = field(default_factory=dict)
    _is_bottom: bool = False

    @staticmethod
    def bottom() -> AbstractState:
        return AbstractState(_is_bottom=True)

    def is_bottom(self) -> bool:
        return self._is_bottom

    def get(self, var: str) -> CofiberedElement:
        return self.env.get(var, CofiberedElement.bottom())

    def set(self, var: str, val: CofiberedElement) -> AbstractState:
        new_env = dict(self.env)
        new_env[var] = val
        return AbstractState(env=new_env)

    def copy(self) -> AbstractState:
        return AbstractState(env=dict(self.env))


class PythonTransferFunctions:
    """Transfer functions for Python operations on the combined domain."""

    def __init__(self) -> None:
        self.iv = IntervalDomain()
        self.sd = StringDomain()
        self.cd = ContainerDomain()
        self.cof = CofiberedDomain({
            "int": self.iv,
            "float": self.iv,
            "str": self.sd,
            "list": self.cd,
            "dict": self.cd,
            "set": self.cd,
        })

    # -- helpers -------------------------------------------------------------

    def _int_elem(self, itv: Interval) -> CofiberedElement:
        return CofiberedElement.singleton("int", itv)

    def _str_elem(self, sa: StringAbstract) -> CofiberedElement:
        return CofiberedElement.singleton("str", sa)

    def _bool_elem(self) -> CofiberedElement:
        return self._int_elem(Interval.from_range(0, 1))

    def _none_elem(self) -> CofiberedElement:
        return CofiberedElement.singleton("NoneType", None)

    # -- arithmetic ----------------------------------------------------------

    def binary_arith(self, op: str, lhs: CofiberedElement, rhs: CofiberedElement) -> CofiberedElement:
        """Handle +, -, *, /, //, %, **."""
        a_int = lhs.get("int") or lhs.get("float")
        b_int = rhs.get("int") or rhs.get("float")
        if a_int is not None and b_int is not None:
            if not isinstance(a_int, Interval):
                a_int = Interval.top()
            if not isinstance(b_int, Interval):
                b_int = Interval.top()
            if op == "+":
                return self._int_elem(self.iv.add(a_int, b_int))
            if op == "-":
                return self._int_elem(self.iv.sub(a_int, b_int))
            if op == "*":
                return self._int_elem(self.iv.mul(a_int, b_int))
            if op in ("/", "//"):
                return self._int_elem(self.iv.div(a_int, b_int))
            if op == "%":
                return self._int_elem(self.iv.mod(a_int, b_int))
            if op == "**":
                # Conservative: return top for exponentiation
                return self._int_elem(Interval.top())
        # String concatenation
        a_str = lhs.get("str")
        b_str = rhs.get("str")
        if a_str is not None and b_str is not None and op == "+":
            return self._str_elem(self.sd.concat(a_str, b_str))
        return CofiberedElement(entries={})

    def unary_op(self, op: str, operand: CofiberedElement) -> CofiberedElement:
        val = operand.get("int") or operand.get("float")
        if val is not None and isinstance(val, Interval):
            if op == "-":
                return self._int_elem(self.iv.neg(val))
            if op == "+":
                return self._int_elem(val)
            if op == "~":
                neg = self.iv.neg(val)
                return self._int_elem(self.iv.sub(neg, Interval.const(1)))
        return CofiberedElement(entries={})

    # -- comparisons ---------------------------------------------------------

    def compare(self, op: str, lhs: CofiberedElement, rhs: CofiberedElement) -> CofiberedElement:
        """Return abstract bool for a comparison."""
        a = lhs.get("int") or lhs.get("float")
        b = rhs.get("int") or rhs.get("float")
        if a is not None and b is not None and isinstance(a, Interval) and isinstance(b, Interval):
            if op == "<" and a.hi < b.lo:
                return self._int_elem(Interval.const(1))
            if op == "<" and a.lo >= b.hi:
                return self._int_elem(Interval.const(0))
            if op == ">" and a.lo > b.hi:
                return self._int_elem(Interval.const(1))
            if op == ">" and a.hi <= b.lo:
                return self._int_elem(Interval.const(0))
            if op == "==" and a.is_const() and b.is_const() and a.lo == b.lo:
                return self._int_elem(Interval.const(1))
            if op == "!=" and a.is_const() and b.is_const() and a.lo == b.lo:
                return self._int_elem(Interval.const(0))
        return self._bool_elem()

    def guard_compare(
        self,
        op: str,
        state: AbstractState,
        lhs_var: str,
        rhs_var: str,
    ) -> AbstractState:
        """Refine *state* under a comparison guard."""
        lhs = state.get(lhs_var)
        rhs = state.get(rhs_var)
        a = lhs.get("int")
        b = rhs.get("int")
        if a is None or b is None or not isinstance(a, Interval) or not isinstance(b, Interval):
            return state
        if op == "<":
            a2, b2 = self.iv.guard_lt(a, b)
        elif op == "<=":
            a2, b2 = self.iv.guard_le(a, b)
        elif op == ">":
            b2, a2 = self.iv.guard_lt(b, a)
        elif op == ">=":
            b2, a2 = self.iv.guard_le(b, a)
        elif op == "==":
            a2, b2 = self.iv.guard_eq(a, b)
        else:
            return state
        s = state.set(lhs_var, self._int_elem(a2))
        s = s.set(rhs_var, self._int_elem(b2))
        return s

    # -- attribute access ----------------------------------------------------

    def attr_access(self, obj: CofiberedElement, attr: str) -> CofiberedElement:
        """Approximate attribute access on abstract objects."""
        s = obj.get("str")
        if s is not None and isinstance(s, StringAbstract):
            if attr == "upper":
                return self._str_elem(self.sd.upper(s))
            if attr == "lower":
                return self._str_elem(self.sd.lower(s))
            if attr == "strip":
                return self._str_elem(self.sd.strip(s))
        return CofiberedElement(entries={})

    # -- indexing ------------------------------------------------------------

    def index_access(self, obj: CofiberedElement, idx: CofiberedElement) -> CofiberedElement:
        """Approximate obj[idx]."""
        for tag in ("list", "dict", "set"):
            c = obj.get(tag)
            if c is not None and isinstance(c, ContainerAbstract):
                et = self.cd.getitem(c, 0)
                if et is not None:
                    return CofiberedElement.singleton(et, Interval.top())
                return CofiberedElement(entries={})
        s = obj.get("str")
        if s is not None:
            return self._str_elem(StringAbstract(length=Interval.const(1), charset=s.charset if isinstance(s, StringAbstract) else None))
        return CofiberedElement(entries={})

    # -- function calls (builtins) -------------------------------------------

    def builtin_call(self, name: str, args: List[CofiberedElement]) -> CofiberedElement:
        """Transfer for built-in function calls."""
        if name == "len" and len(args) == 1:
            for tag in ("list", "dict", "set", "str"):
                v = args[0].get(tag)
                if v is not None:
                    if isinstance(v, ContainerAbstract):
                        return self._int_elem(v.length)
                    if isinstance(v, StringAbstract):
                        return self._int_elem(v.length)
            return self._int_elem(Interval.from_range(0, INF))
        if name == "int" and len(args) == 1:
            return self._int_elem(Interval.top())
        if name == "str" and len(args) == 1:
            return self._str_elem(StringAbstract.top())
        if name == "float" and len(args) == 1:
            return CofiberedElement.singleton("float", Interval.top())
        if name == "bool" and len(args) == 1:
            return self._bool_elem()
        if name == "abs" and len(args) == 1:
            v = args[0].get("int")
            if v is not None and isinstance(v, Interval):
                lo = 0.0 if v.lo <= 0 <= v.hi else min(abs(v.lo), abs(v.hi))
                hi = max(abs(v.lo), abs(v.hi)) if v.lo != NEG_INF and v.hi != INF else INF
                return self._int_elem(Interval.from_range(lo, hi))
            return self._int_elem(Interval.from_range(0, INF))
        if name == "min" and len(args) == 2:
            a = args[0].get("int")
            b = args[1].get("int")
            if isinstance(a, Interval) and isinstance(b, Interval):
                return self._int_elem(Interval.from_range(min(a.lo, b.lo), min(a.hi, b.hi)))
        if name == "max" and len(args) == 2:
            a = args[0].get("int")
            b = args[1].get("int")
            if isinstance(a, Interval) and isinstance(b, Interval):
                return self._int_elem(Interval.from_range(max(a.lo, b.lo), max(a.hi, b.hi)))
        if name == "range":
            return CofiberedElement.singleton("list", ContainerAbstract(
                kind=ContainerKind.LIST, element_type="int",
                length=Interval.from_range(0, INF),
            ))
        if name == "list" and len(args) <= 1:
            return CofiberedElement.singleton("list", ContainerAbstract.top_list())
        if name == "dict" and len(args) <= 1:
            return CofiberedElement.singleton("dict", ContainerAbstract.top_dict())
        if name == "set" and len(args) <= 1:
            return CofiberedElement.singleton("set", ContainerAbstract.top_set())
        if name == "sorted" and len(args) >= 1:
            return CofiberedElement.singleton("list", ContainerAbstract.top_list())
        if name == "reversed" and len(args) == 1:
            return args[0]
        if name == "enumerate" and len(args) >= 1:
            return CofiberedElement.singleton("list", ContainerAbstract(
                kind=ContainerKind.LIST, element_type="tuple",
                length=Interval.from_range(0, INF),
            ))
        if name == "zip" and len(args) >= 1:
            return CofiberedElement.singleton("list", ContainerAbstract(
                kind=ContainerKind.LIST, element_type="tuple",
                length=Interval.from_range(0, INF),
            ))
        if name == "isinstance" and len(args) == 2:
            return self._bool_elem()
        if name == "type" and len(args) == 1:
            return CofiberedElement.singleton("type", None)
        if name == "print":
            return self._none_elem()
        # Unknown call — return top-like element
        return CofiberedElement(entries={})

    # -- type tests ----------------------------------------------------------

    def isinstance_guard(
        self,
        state: AbstractState,
        var: str,
        type_name: str,
        positive: bool = True,
    ) -> AbstractState:
        """Refine *state* given ``isinstance(var, type_name)`` is True/False."""
        val = state.get(var)
        if positive:
            kept: Dict[str, Any] = {}
            if type_name in val.entries:
                kept[type_name] = val.entries[type_name]
            elif type_name in self.cof.fiber_domains:
                dom = self.cof.fiber_domains[type_name]
                kept[type_name] = dom.top() if hasattr(dom, "top") else None
            if not kept:
                return AbstractState.bottom()
            return state.set(var, CofiberedElement(entries=kept))
        else:
            kept = {t: v for t, v in val.entries.items() if t != type_name}
            if not kept:
                return AbstractState.bottom()
            return state.set(var, CofiberedElement(entries=kept))

    # -- None checks ---------------------------------------------------------

    def none_check(self, state: AbstractState, var: str, is_none: bool) -> AbstractState:
        """Refine *state* for ``var is None`` or ``var is not None``."""
        val = state.get(var)
        if is_none:
            return state.set(var, self._none_elem())
        else:
            kept = {t: v for t, v in val.entries.items() if t != "NoneType"}
            if not kept and val.entries:
                return state  # no info to refine with
            return state.set(var, CofiberedElement(entries=kept)) if kept else state

    # -- truthiness ----------------------------------------------------------

    def truthiness(self, val: CofiberedElement) -> Tuple[bool, bool]:
        """Return (can_be_true, can_be_false) for *val*."""
        can_true = False
        can_false = False
        # int/float: 0 is falsy
        for tag in ("int", "float"):
            v = val.get(tag)
            if v is not None and isinstance(v, Interval):
                if v.contains(0):
                    can_false = True
                if v.lo < 0 or v.hi > 0:
                    can_true = True
        # str: empty string is falsy
        s = val.get("str")
        if s is not None and isinstance(s, StringAbstract):
            if s.length.contains(0):
                can_false = True
            if s.length.hi > 0:
                can_true = True
        # containers: empty is falsy
        for tag in ("list", "dict", "set"):
            c = val.get(tag)
            if c is not None and isinstance(c, ContainerAbstract):
                if c.length.contains(0):
                    can_false = True
                if c.length.hi > 0:
                    can_true = True
        # NoneType is always falsy
        if val.get("NoneType") is not None:
            can_false = True
        # If we have no info, assume both possible
        if not can_true and not can_false:
            can_true = True
            can_false = True
        return can_true, can_false

    def guard_truthy(self, state: AbstractState, var: str, truthy: bool) -> AbstractState:
        """Refine state assuming ``var`` is truthy or falsy."""
        val = state.get(var)
        if val.is_bottom():
            return AbstractState.bottom()
        # Refine int component: truthy => !=0
        iv = val.get("int")
        if iv is not None and isinstance(iv, Interval):
            if truthy:
                # Remove 0 — split into two intervals; we approximate
                if iv.lo == 0 and iv.hi == 0:
                    new_val = CofiberedElement(entries={t: v for t, v in val.entries.items() if t != "int"})
                    if new_val.is_bottom():
                        return AbstractState.bottom()
                    return state.set(var, new_val)
            else:
                return state.set(var, self._int_elem(Interval.const(0)))
        # Refine containers: falsy => length == 0
        for tag in ("list", "dict", "set"):
            c = val.get(tag)
            if c is not None and isinstance(c, ContainerAbstract):
                if not truthy:
                    refined = ContainerAbstract(
                        kind=c.kind,
                        element_type=c.element_type,
                        length=Interval.const(0),
                    )
                    return state.set(var, CofiberedElement.singleton(tag, refined))
        return state

    # -- assignment ----------------------------------------------------------

    def assign_const(self, state: AbstractState, var: str, value: Any) -> AbstractState:
        """Assign a concrete constant to *var*."""
        if isinstance(value, bool):
            elem = self._int_elem(Interval.const(int(value)))
        elif isinstance(value, int):
            elem = self._int_elem(Interval.const(float(value)))
        elif isinstance(value, float):
            elem = CofiberedElement.singleton("float", Interval.const(value))
        elif isinstance(value, str):
            elem = self._str_elem(StringAbstract.const(value))
        elif value is None:
            elem = self._none_elem()
        elif isinstance(value, list):
            elem = CofiberedElement.singleton("list", ContainerAbstract(
                kind=ContainerKind.LIST,
                length=Interval.const(len(value)),
            ))
        elif isinstance(value, dict):
            elem = CofiberedElement.singleton("dict", ContainerAbstract(
                kind=ContainerKind.DICT,
                length=Interval.const(len(value)),
                known_keys={str(k): str(type(v).__name__) for k, v in value.items()},
            ))
        else:
            elem = CofiberedElement(entries={})
        return state.set(var, elem)

    def assign_var(self, state: AbstractState, dst: str, src: str) -> AbstractState:
        """dst := src."""
        return state.set(dst, state.get(src))

    def assign_binop(self, state: AbstractState, dst: str, op: str, lhs: str, rhs: str) -> AbstractState:
        result = self.binary_arith(op, state.get(lhs), state.get(rhs))
        return state.set(dst, result)

    # -- join states ---------------------------------------------------------

    def join_states(self, a: AbstractState, b: AbstractState) -> AbstractState:
        """Join two abstract states."""
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        all_vars = set(a.env.keys()) | set(b.env.keys())
        env: Dict[str, CofiberedElement] = {}
        for v in all_vars:
            env[v] = self.cof.join(a.get(v), b.get(v))
        return AbstractState(env=env)

    def widen_states(self, a: AbstractState, b: AbstractState) -> AbstractState:
        """Widen two abstract states."""
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        all_vars = set(a.env.keys()) | set(b.env.keys())
        env: Dict[str, CofiberedElement] = {}
        for v in all_vars:
            env[v] = self.cof.widen(a.get(v), b.get(v))
        return AbstractState(env=env)
