"""
Taylor model enclosures for validated ODE integration.

A Taylor model T = (p, I) represents the set {p(x) + e : e in I}
where p is a multivariate polynomial and I is an interval remainder.
This provides tight enclosures that combine polynomial approximation
with rigorous error bounding.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from bioprover.solver.interval import (
    Interval,
    IntervalVector,
    _round_down,
    _round_up,
)

# Multi-index type: tuple of non-negative integers
MultiIndex = Tuple[int, ...]


def _multi_index_order(idx: MultiIndex) -> int:
    return sum(idx)


def _multi_index_add(a: MultiIndex, b: MultiIndex) -> MultiIndex:
    return tuple(ai + bi for ai, bi in zip(a, b))


def _zero_index(n: int) -> MultiIndex:
    return (0,) * n


def _unit_index(n: int, i: int) -> MultiIndex:
    idx = [0] * n
    idx[i] = 1
    return tuple(idx)


def _multinomial_coeff(idx: MultiIndex) -> int:
    """Multinomial coefficient: (sum(idx))! / prod(idx_i!)."""
    total = sum(idx)
    result = math.factorial(total)
    for k in idx:
        result //= math.factorial(k)
    return result


# ---------------------------------------------------------------------------
# TaylorModel class
# ---------------------------------------------------------------------------

class TaylorModel:
    """
    Taylor model: polynomial p(x) + interval remainder I.

    The polynomial is stored as a dict mapping multi-indices to float coefficients.
    The domain is an IntervalVector centered at some expansion point.
    """

    __slots__ = ("_coeffs", "_remainder", "_nvars", "_order")

    def __init__(
        self,
        coeffs: Dict[MultiIndex, float],
        remainder: Interval,
        nvars: int,
        order: int,
    ) -> None:
        self._coeffs = {k: v for k, v in coeffs.items() if v != 0.0}
        self._remainder = remainder
        self._nvars = nvars
        self._order = order

    @classmethod
    def constant(cls, value: float, nvars: int, order: int) -> "TaylorModel":
        idx = _zero_index(nvars)
        return cls({idx: value}, Interval(0.0), nvars, order)

    @classmethod
    def variable(cls, var_index: int, nvars: int, order: int) -> "TaylorModel":
        """Create a TM representing the variable x_i."""
        idx = _unit_index(nvars, var_index)
        return cls({idx: 1.0}, Interval(0.0), nvars, order)

    @classmethod
    def from_interval(cls, iv: Interval, nvars: int, order: int) -> "TaylorModel":
        """Create a TM enclosing an interval (constant mid + remainder)."""
        m = iv.mid()
        r = iv.radius()
        return cls({_zero_index(nvars): m}, Interval(-r, r), nvars, order)

    @property
    def nvars(self) -> int:
        return self._nvars

    @property
    def order(self) -> int:
        return self._order

    @property
    def remainder(self) -> Interval:
        return self._remainder

    @property
    def coeffs(self) -> Dict[MultiIndex, float]:
        return dict(self._coeffs)

    def get_coeff(self, idx: MultiIndex) -> float:
        return self._coeffs.get(idx, 0.0)

    def constant_term(self) -> float:
        return self.get_coeff(_zero_index(self._nvars))

    # -- arithmetic ----------------------------------------------------------

    def __neg__(self) -> "TaylorModel":
        new_coeffs = {k: -v for k, v in self._coeffs.items()}
        return TaylorModel(new_coeffs, -self._remainder, self._nvars, self._order)

    def __add__(self, other: Union[float, "TaylorModel"]) -> "TaylorModel":
        if isinstance(other, (int, float)):
            new_coeffs = dict(self._coeffs)
            z = _zero_index(self._nvars)
            new_coeffs[z] = new_coeffs.get(z, 0.0) + other
            return TaylorModel(new_coeffs, self._remainder, self._nvars, self._order)
        assert self._nvars == other._nvars
        order = min(self._order, other._order)
        new_coeffs: Dict[MultiIndex, float] = {}
        all_keys = set(self._coeffs.keys()) | set(other._coeffs.keys())
        remainder = self._remainder + other._remainder
        for k in all_keys:
            val = self._coeffs.get(k, 0.0) + other._coeffs.get(k, 0.0)
            if _multi_index_order(k) > order:
                # Absorb into remainder
                remainder = remainder + Interval(-abs(val), abs(val))
            elif val != 0.0:
                new_coeffs[k] = val
        return TaylorModel(new_coeffs, remainder, self._nvars, order)

    def __radd__(self, other: float) -> "TaylorModel":
        return self.__add__(other)

    def __sub__(self, other: Union[float, "TaylorModel"]) -> "TaylorModel":
        if isinstance(other, (int, float)):
            return self.__add__(-other)
        return self.__add__(-other)

    def __rsub__(self, other: float) -> "TaylorModel":
        return (-self).__add__(other)

    def __mul__(self, other: Union[float, "TaylorModel"]) -> "TaylorModel":
        if isinstance(other, (int, float)):
            new_coeffs = {k: v * other for k, v in self._coeffs.items()}
            return TaylorModel(
                new_coeffs, self._remainder * other, self._nvars, self._order
            )
        assert self._nvars == other._nvars
        order = min(self._order, other._order)
        new_coeffs: Dict[MultiIndex, float] = {}
        remainder = Interval(0.0)

        # Polynomial × polynomial
        for k1, v1 in self._coeffs.items():
            for k2, v2 in other._coeffs.items():
                idx = _multi_index_add(k1, k2)
                if _multi_index_order(idx) > order:
                    remainder = remainder + Interval(-abs(v1 * v2), abs(v1 * v2))
                else:
                    new_coeffs[idx] = new_coeffs.get(idx, 0.0) + v1 * v2

        # Cross terms: polynomial × remainder
        p_self_bound = self._bound_polynomial_abs()
        p_other_bound = other._bound_polynomial_abs()
        remainder = remainder + p_self_bound * other._remainder
        remainder = remainder + self._remainder * p_other_bound
        remainder = remainder + self._remainder * other._remainder

        return TaylorModel(new_coeffs, remainder, self._nvars, order)

    def __rmul__(self, other: float) -> "TaylorModel":
        return self.__mul__(other)

    def _bound_polynomial_abs(self) -> Interval:
        """Bound |p(x)| over the unit domain [-1,1]^n."""
        total = 0.0
        for v in self._coeffs.values():
            total += abs(v)
        return Interval(-total, total)

    # -- order reduction -----------------------------------------------------

    def reduce_order(self, new_order: int) -> "TaylorModel":
        """Truncate terms above new_order into the remainder."""
        if new_order >= self._order:
            return TaylorModel(dict(self._coeffs), self._remainder, self._nvars, self._order)
        new_coeffs: Dict[MultiIndex, float] = {}
        remainder = self._remainder
        for k, v in self._coeffs.items():
            if _multi_index_order(k) > new_order:
                remainder = remainder + Interval(-abs(v), abs(v))
            else:
                new_coeffs[k] = v
        return TaylorModel(new_coeffs, remainder, self._nvars, new_order)

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, point: np.ndarray) -> Interval:
        """Evaluate the Taylor model at a point, returning an interval."""
        val = 0.0
        for idx, coeff in self._coeffs.items():
            term = coeff
            for i, exp in enumerate(idx):
                if exp > 0:
                    term *= point[i] ** exp
            val += term
        return Interval(_round_down(val), _round_up(val)) + self._remainder

    def bound(self, domain: Optional[IntervalVector] = None) -> Interval:
        """Bound the TM over a domain using natural interval extension."""
        if domain is None:
            # Default: unit domain
            domain = IntervalVector([Interval(-1.0, 1.0)] * self._nvars)
        result = Interval(0.0)
        for idx, coeff in self._coeffs.items():
            term = Interval(coeff)
            for i, exp in enumerate(idx):
                if exp > 0:
                    term = term * (domain[i] ** exp)
            result = result + term
        return result + self._remainder

    def bound_horner(self, domain: Optional[IntervalVector] = None) -> Interval:
        """
        Bound using a Horner-like scheme for univariate TMs.
        Falls back to natural extension for multivariate.
        """
        if self._nvars != 1:
            return self.bound(domain)
        if domain is None:
            domain = IntervalVector([Interval(-1.0, 1.0)])
        x = domain[0]
        # Collect coefficients by degree
        max_deg = 0
        coeffs_by_deg: Dict[int, float] = {}
        for idx, coeff in self._coeffs.items():
            d = idx[0]
            coeffs_by_deg[d] = coeffs_by_deg.get(d, 0.0) + coeff
            max_deg = max(max_deg, d)
        # Horner evaluation
        result = Interval(coeffs_by_deg.get(max_deg, 0.0))
        for d in range(max_deg - 1, -1, -1):
            result = result * x + coeffs_by_deg.get(d, 0.0)
        return result + self._remainder

    def to_interval(self, domain: Optional[IntervalVector] = None) -> Interval:
        """Convert to a single interval over the given domain."""
        return self.bound(domain)

    # -- elementary function TMs ---------------------------------------------

    def tm_exp(self) -> "TaylorModel":
        """Taylor model for exp(self) via Taylor expansion of exp."""
        c = self.constant_term()
        # f = self - c (centered part)
        f = self - c
        exp_c = math.exp(c)
        result = TaylorModel.constant(exp_c, self._nvars, self._order)
        f_power = TaylorModel.constant(1.0, self._nvars, self._order)
        factorial = 1.0
        for k in range(1, self._order + 1):
            f_power = f_power * f
            factorial *= k
            result = result + f_power * (exp_c / factorial)
        # Remainder bound: exp is monotone increasing
        f_bound = f.bound()
        exp_bound = (Interval(c) + f_bound).exp()
        poly_bound = result.bound()
        rem = Interval(
            _round_down(float(exp_bound.lo) - float(poly_bound.hi)),
            _round_up(float(exp_bound.hi) - float(poly_bound.lo)),
        )
        result = TaylorModel(result._coeffs, result._remainder + rem, self._nvars, self._order)
        return result

    def tm_log(self) -> "TaylorModel":
        """Taylor model for log(self) via Taylor expansion around constant term."""
        c = self.constant_term()
        if c <= 0:
            raise ValueError("log requires positive constant term")
        f = self - c  # centered
        u = f * (1.0 / c)  # u = (self - c)/c
        log_c = math.log(c)
        result = TaylorModel.constant(log_c, self._nvars, self._order)
        u_power = TaylorModel.constant(1.0, self._nvars, self._order)
        for k in range(1, self._order + 1):
            u_power = u_power * u
            sign = (-1.0) ** (k + 1)
            result = result + u_power * (sign / k)
        # Remainder: bound via interval evaluation
        self_bound = self.bound()
        if self_bound.lo <= 0:
            raise ValueError("log: interval may contain non-positive values")
        log_bound = self_bound.log()
        poly_bound = result.bound()
        rem = Interval(
            _round_down(float(log_bound.lo) - float(poly_bound.hi)),
            _round_up(float(log_bound.hi) - float(poly_bound.lo)),
        )
        result = TaylorModel(result._coeffs, result._remainder + rem, self._nvars, self._order)
        return result

    def tm_sin(self) -> "TaylorModel":
        """Taylor model for sin(self)."""
        c = self.constant_term()
        f = self - c
        sin_c = math.sin(c)
        cos_c = math.cos(c)
        result = TaylorModel.constant(sin_c, self._nvars, self._order)
        f_power = TaylorModel.constant(1.0, self._nvars, self._order)
        factorial = 1.0
        # sin(c + f) = sum_{k=0}^{order} sin^(k)(c) / k! * f^k
        for k in range(1, self._order + 1):
            f_power = f_power * f
            factorial *= k
            phase = k % 4
            if phase == 0:
                deriv_c = sin_c
            elif phase == 1:
                deriv_c = cos_c
            elif phase == 2:
                deriv_c = -sin_c
            else:
                deriv_c = -cos_c
            result = result + f_power * (deriv_c / factorial)
        # Rigorous remainder
        result = TaylorModel(
            result._coeffs,
            result._remainder + Interval(-1.0, 1.0) * _taylor_remainder_bound(f, self._order),
            self._nvars,
            self._order,
        )
        return result

    def tm_cos(self) -> "TaylorModel":
        """Taylor model for cos(self)."""
        c = self.constant_term()
        f = self - c
        sin_c = math.sin(c)
        cos_c = math.cos(c)
        result = TaylorModel.constant(cos_c, self._nvars, self._order)
        f_power = TaylorModel.constant(1.0, self._nvars, self._order)
        factorial = 1.0
        for k in range(1, self._order + 1):
            f_power = f_power * f
            factorial *= k
            phase = k % 4
            if phase == 0:
                deriv_c = cos_c
            elif phase == 1:
                deriv_c = -sin_c
            elif phase == 2:
                deriv_c = -cos_c
            else:
                deriv_c = sin_c
            result = result + f_power * (deriv_c / factorial)
        result = TaylorModel(
            result._coeffs,
            result._remainder + Interval(-1.0, 1.0) * _taylor_remainder_bound(f, self._order),
            self._nvars,
            self._order,
        )
        return result

    def __repr__(self) -> str:
        terms = []
        for idx in sorted(self._coeffs.keys()):
            c = self._coeffs[idx]
            if abs(c) < 1e-300:
                continue
            parts = []
            for i, exp in enumerate(idx):
                if exp == 1:
                    parts.append(f"x{i}")
                elif exp > 1:
                    parts.append(f"x{i}^{exp}")
            mono = "*".join(parts) if parts else "1"
            terms.append(f"{c:+.6g}*{mono}")
        poly = " ".join(terms) if terms else "0"
        return f"TM({poly} + {self._remainder})"


def _taylor_remainder_bound(f: TaylorModel, order: int) -> Interval:
    """Bound on |f|^(order+1)/(order+1)! for remainder estimation."""
    f_bound = f.bound()
    mag = f_bound.magnitude()
    if mag == 0:
        return Interval(0.0)
    rem = mag ** (order + 1) / math.factorial(order + 1)
    return Interval(-rem, rem)


# ---------------------------------------------------------------------------
# Picard-Lindelof operator for ODE enclosures
# ---------------------------------------------------------------------------

def picard_lindelof(
    f: Callable[[List[TaylorModel], TaylorModel], List[TaylorModel]],
    x0: IntervalVector,
    t_interval: Interval,
    order: int,
    max_iterations: int = 20,
    tol: float = 1e-10,
) -> Optional[List[TaylorModel]]:
    """
    Picard-Lindelöf iteration to find a-priori enclosure for ODE x' = f(x, t).

    Starting from a rough enclosure, iterates T[x](t) = x0 + int_0^t f(x(s), s) ds
    until a fixed point (self-validating enclosure) is found.

    Args:
        f: right-hand side, takes (state TMs, time TM) -> derivative TMs
        x0: initial condition box
        t_interval: time step interval [0, h]
        order: Taylor model order
        max_iterations: maximum Picard iterations
        tol: convergence tolerance on remainder width

    Returns:
        List of TaylorModels enclosing the solution, or None if no convergence.
    """
    nvars = x0.dim
    n = nvars + 1  # state vars + time

    # Initial rough enclosure: x0 bloated
    rough = x0.bloat(0.1 * x0.max_width() + 1e-8)

    # Taylor models for initial state
    x_tms = []
    for i in range(nvars):
        tm = TaylorModel.from_interval(rough[i], n, order)
        x_tms.append(tm)

    t_tm = TaylorModel.variable(nvars, n, order)
    h = Interval(0.0, t_interval.hi)
    t_tm = TaylorModel(
        {_unit_index(n, nvars): float(h.hi)},
        Interval(0.0),
        n,
        order,
    )

    for iteration in range(max_iterations):
        # Evaluate f at current enclosure
        derivatives = f(x_tms, t_tm)

        # Integrate: x0 + int_0^t f ds ≈ x0 + h * f(enclosure)
        new_tms = []
        for i in range(nvars):
            x0_tm = TaylorModel.from_interval(x0[i], n, order)
            # Rough integration: multiply derivative bound by time interval
            deriv_bound = derivatives[i].bound()
            integral_bound = h * deriv_bound
            integrated = TaylorModel(
                x0_tm._coeffs,
                x0_tm._remainder + integral_bound,
                n,
                order,
            )
            new_tms.append(integrated)

        # Check if new enclosure is contained in old enclosure
        converged = True
        for i in range(nvars):
            old_bound = x_tms[i].bound()
            new_bound = new_tms[i].bound()
            if not old_bound.contains(new_bound):
                converged = False
                break
            if new_bound.width() > tol and new_bound.width() >= old_bound.width() * 0.99:
                converged = False
                break

        x_tms = new_tms

        if converged:
            return x_tms

    return None


# ---------------------------------------------------------------------------
# Shrink wrapping
# ---------------------------------------------------------------------------

def shrink_wrap(tms: List[TaylorModel], factor: float = 0.1) -> List[TaylorModel]:
    """
    Shrink wrapping: reduce overestimation by absorbing part of the
    polynomial into the remainder in a controlled way.

    Rescales the polynomial part and adjusts the remainder to maintain
    enclosure while potentially reducing the total width.
    """
    result = []
    for tm in tms:
        poly_bound = Interval(0.0)
        for idx, coeff in tm._coeffs.items():
            if _multi_index_order(idx) > 0:
                poly_bound = poly_bound + Interval(-abs(coeff), abs(coeff))

        if poly_bound.width() < 1e-300:
            result.append(tm)
            continue

        # Shrink polynomial, grow remainder
        alpha = 1.0 - factor
        new_coeffs: Dict[MultiIndex, float] = {}
        absorbed = Interval(0.0)
        for idx, coeff in tm._coeffs.items():
            if _multi_index_order(idx) == 0:
                new_coeffs[idx] = coeff
            else:
                new_coeffs[idx] = coeff * alpha
                absorbed = absorbed + Interval(
                    -abs(coeff) * factor, abs(coeff) * factor
                )
        new_remainder = tm._remainder + absorbed
        result.append(TaylorModel(new_coeffs, new_remainder, tm._nvars, tm._order))
    return result
