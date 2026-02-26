"""
Taylor model arithmetic for validated ODE integration.

Taylor models represent functions as polynomial + interval remainder,
providing tight enclosures with reduced dependency problem.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from itertools import product
from .interval import Interval
from .matrix import IntervalVector, IntervalMatrix


class MultiIndex:
    """Multi-index for multivariate polynomial terms."""
    
    __slots__ = ['indices']
    
    def __init__(self, indices: Tuple[int, ...]):
        self.indices = tuple(indices)
    
    @property
    def total_degree(self) -> int:
        return sum(self.indices)
    
    @property
    def dim(self) -> int:
        return len(self.indices)
    
    def __hash__(self):
        return hash(self.indices)
    
    def __eq__(self, other):
        return isinstance(other, MultiIndex) and self.indices == other.indices
    
    def __add__(self, other):
        return MultiIndex(tuple(a + b for a, b in zip(self.indices, other.indices)))
    
    def __repr__(self):
        return f"MI{self.indices}"
    
    def __lt__(self, other):
        return self.indices < other.indices


def multi_indices(dim: int, max_degree: int) -> List[MultiIndex]:
    """Generate all multi-indices up to given total degree."""
    result = []
    _gen_multi_indices(dim, max_degree, [], result)
    return result


def _gen_multi_indices(dim: int, remaining: int, current: list, result: list):
    """Recursive helper for multi-index generation."""
    if len(current) == dim:
        result.append(MultiIndex(tuple(current)))
        return
    for k in range(remaining + 1):
        current.append(k)
        _gen_multi_indices(dim, remaining - k, current, result)
        current.pop()


class TaylorModel:
    """
    Taylor model: polynomial approximation + interval remainder.
    
    Represents f(x) ∈ p(x) + I where p is a multivariate polynomial
    and I is a rigorous interval remainder bound.
    """
    
    def __init__(self, dim: int, order: int,
                 coefficients: Optional[Dict[MultiIndex, float]] = None,
                 remainder: Optional[Interval] = None,
                 domain: Optional[List[Interval]] = None):
        self.dim = dim
        self.order = order
        self.coefficients = coefficients if coefficients is not None else {}
        self.remainder = remainder if remainder is not None else Interval(0.0)
        self.domain = domain if domain is not None else [Interval(-1, 1)] * dim
    
    @classmethod
    def constant(cls, dim: int, order: int, value: float,
                 domain: Optional[List[Interval]] = None) -> 'TaylorModel':
        """Create constant Taylor model."""
        zero_idx = MultiIndex(tuple(0 for _ in range(dim)))
        return cls(dim, order, {zero_idx: value}, Interval(0.0), domain)
    
    @classmethod
    def variable(cls, dim: int, order: int, var_index: int,
                 center: float = 0.0,
                 domain: Optional[List[Interval]] = None) -> 'TaylorModel':
        """Create Taylor model for a single variable x_i."""
        zero_idx = MultiIndex(tuple(0 for _ in range(dim)))
        var_idx = MultiIndex(tuple(1 if j == var_index else 0 for j in range(dim)))
        coeffs = {zero_idx: center, var_idx: 1.0}
        return cls(dim, order, coeffs, Interval(0.0), domain)
    
    @classmethod
    def identity(cls, dim: int, order: int,
                 centers: Optional[np.ndarray] = None,
                 domain: Optional[List[Interval]] = None) -> List['TaylorModel']:
        """Create identity Taylor model vector."""
        if centers is None:
            centers = np.zeros(dim)
        return [cls.variable(dim, order, i, centers[i], domain) for i in range(dim)]
    
    @classmethod
    def from_interval(cls, dim: int, order: int, iv: Interval,
                      domain: Optional[List[Interval]] = None) -> 'TaylorModel':
        """Create Taylor model from interval."""
        zero_idx = MultiIndex(tuple(0 for _ in range(dim)))
        return cls(dim, order, {zero_idx: iv.mid}, Interval(-iv.rad, iv.rad), domain)
    
    def get_coeff(self, idx: MultiIndex) -> float:
        """Get coefficient for multi-index."""
        return self.coefficients.get(idx, 0.0)
    
    def set_coeff(self, idx: MultiIndex, value: float):
        """Set coefficient for multi-index."""
        if value != 0.0:
            self.coefficients[idx] = value
        elif idx in self.coefficients:
            del self.coefficients[idx]
    
    @property
    def constant_term(self) -> float:
        """Get constant term."""
        return self.get_coeff(MultiIndex(tuple(0 for _ in range(self.dim))))
    
    def evaluate(self, point: np.ndarray) -> Interval:
        """Evaluate Taylor model at a point, returning interval enclosure."""
        value = 0.0
        for idx, coeff in self.coefficients.items():
            term = coeff
            for j in range(self.dim):
                term *= point[j] ** idx.indices[j]
            value += term
        return Interval(value) + self.remainder
    
    def evaluate_interval(self, box: List[Interval]) -> Interval:
        """Evaluate Taylor model over an interval box."""
        result = Interval(0.0)
        for idx, coeff in self.coefficients.items():
            term = Interval(coeff)
            for j in range(self.dim):
                if idx.indices[j] > 0:
                    term = term * (box[j] ** idx.indices[j])
            result = result + term
        return result + self.remainder
    
    def bound(self) -> Interval:
        """Compute bound over the domain."""
        return self.evaluate_interval(self.domain)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = TaylorModel(self.dim, self.order,
                               dict(self.coefficients),
                               Interval(self.remainder.lo, self.remainder.hi),
                               list(self.domain))
            zero_idx = MultiIndex(tuple(0 for _ in range(self.dim)))
            result.coefficients[zero_idx] = result.get_coeff(zero_idx) + other
            return result
        if isinstance(other, TaylorModel):
            new_coeffs = dict(self.coefficients)
            for idx, c in other.coefficients.items():
                new_coeffs[idx] = new_coeffs.get(idx, 0.0) + c
            new_rem = self.remainder + other.remainder
            return TaylorModel(self.dim, self.order, new_coeffs, new_rem,
                             list(self.domain))
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        new_coeffs = {idx: -c for idx, c in self.coefficients.items()}
        return TaylorModel(self.dim, self.order, new_coeffs,
                         -self.remainder, list(self.domain))
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self + (-other)
        if isinstance(other, TaylorModel):
            return self + (-other)
        return NotImplemented
    
    def __rsub__(self, other):
        return (-self).__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_coeffs = {idx: c * other for idx, c in self.coefficients.items()}
            new_rem = self.remainder * Interval(other)
            return TaylorModel(self.dim, self.order, new_coeffs, new_rem,
                             list(self.domain))
        if isinstance(other, TaylorModel):
            return self._multiply_tm(other)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def _multiply_tm(self, other: 'TaylorModel') -> 'TaylorModel':
        """Multiply two Taylor models with remainder tracking."""
        new_coeffs = {}
        remainder_contrib = Interval(0.0)
        for idx1, c1 in self.coefficients.items():
            for idx2, c2 in other.coefficients.items():
                new_idx = idx1 + idx2
                if new_idx.total_degree <= self.order:
                    new_coeffs[new_idx] = new_coeffs.get(new_idx, 0.0) + c1 * c2
                else:
                    term = Interval(c1 * c2)
                    for j in range(self.dim):
                        if new_idx.indices[j] > 0:
                            term = term * (self.domain[j] ** new_idx.indices[j])
                    remainder_contrib = remainder_contrib + term
        p_bound = Interval(0.0)
        for idx, c in self.coefficients.items():
            term = Interval(abs(c))
            for j in range(self.dim):
                if idx.indices[j] > 0:
                    term = term * Interval(0.0, self.domain[j].mag ** idx.indices[j])
            p_bound = p_bound + term
        q_bound = Interval(0.0)
        for idx, c in other.coefficients.items():
            term = Interval(abs(c))
            for j in range(self.dim):
                if idx.indices[j] > 0:
                    term = term * Interval(0.0, self.domain[j].mag ** idx.indices[j])
            q_bound = q_bound + term
        new_rem = (self.remainder * other.remainder +
                   self.remainder * q_bound +
                   other.remainder * p_bound +
                   remainder_contrib)
        new_coeffs = {k: v for k, v in new_coeffs.items() if abs(v) > 1e-300}
        return TaylorModel(self.dim, self.order, new_coeffs, new_rem,
                         list(self.domain))
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0.0:
                raise ZeroDivisionError("Division by zero")
            return self * (1.0 / other)
        if isinstance(other, TaylorModel):
            return self * other.reciprocal()
        return NotImplemented
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return TaylorModel.constant(self.dim, self.order, other) / self
        return NotImplemented
    
    def __pow__(self, n: int):
        if n == 0:
            return TaylorModel.constant(self.dim, self.order, 1.0, self.domain)
        if n == 1:
            return TaylorModel(self.dim, self.order,
                             dict(self.coefficients),
                             Interval(self.remainder.lo, self.remainder.hi),
                             list(self.domain))
        if n < 0:
            return TaylorModel.constant(self.dim, self.order, 1.0, self.domain) / (self ** (-n))
        result = TaylorModel.constant(self.dim, self.order, 1.0, self.domain)
        base = self
        exp = n
        while exp > 0:
            if exp % 2 == 1:
                result = result * base
            base = base * base
            exp //= 2
        return result
    
    def reciprocal(self) -> 'TaylorModel':
        """Compute 1/f using Newton iteration on Taylor models."""
        b = self.bound()
        if b.contains(0.0):
            raise ZeroDivisionError("Taylor model contains zero")
        c0 = self.constant_term
        if c0 == 0.0:
            raise ZeroDivisionError("Taylor model has zero constant term")
        g = self - c0
        result = TaylorModel.constant(self.dim, self.order, 1.0 / c0, self.domain)
        ratio = g / c0
        sign = -1.0
        for k in range(1, self.order + 1):
            correction = ratio ** k
            result = result + correction * (sign / c0)
            sign = -sign
        ratio_bound = ratio.bound()
        r_mag = max(abs(ratio_bound.lo), abs(ratio_bound.hi))
        if r_mag < 1.0:
            geo_rem = r_mag ** (self.order + 1) / (1.0 - r_mag) / abs(c0)
            result.remainder = result.remainder + Interval(-geo_rem, geo_rem)
        else:
            iv_recip = Interval(1.0) / b
            result.remainder = result.remainder + iv_recip
        return result
    
    def sqrt(self) -> 'TaylorModel':
        """Compute sqrt via Taylor expansion."""
        b = self.bound()
        if b.lo < 0:
            raise ValueError("Square root of negative TM")
        c0 = self.constant_term
        if c0 <= 0:
            return TaylorModel.from_interval(self.dim, self.order,
                                            b.sqrt(), self.domain)
        sqrt_c0 = np.sqrt(c0)
        g = (self - c0) / c0
        result = TaylorModel.constant(self.dim, self.order, sqrt_c0, self.domain)
        coeff = 1.0
        for k in range(1, min(self.order + 1, 10)):
            coeff *= (0.5 - (k - 1)) / k
            result = result + TaylorModel.constant(
                self.dim, self.order, sqrt_c0 * coeff, self.domain) * (g ** k)
        g_bound = g.bound()
        g_mag = max(abs(g_bound.lo), abs(g_bound.hi))
        if g_mag < 1.0:
            rem_bound = sqrt_c0 * g_mag ** (self.order + 1) * 2
            result.remainder = result.remainder + Interval(-rem_bound, rem_bound)
        else:
            result = TaylorModel.from_interval(self.dim, self.order,
                                              b.sqrt(), self.domain)
        return result
    
    def exp(self) -> 'TaylorModel':
        """Compute exp via Taylor expansion."""
        c0 = self.constant_term
        exp_c0 = np.exp(c0)
        g = self - c0
        result = TaylorModel.constant(self.dim, self.order, exp_c0, self.domain)
        factorial = 1.0
        for k in range(1, self.order + 1):
            factorial *= k
            result = result + (g ** k) * (exp_c0 / factorial)
        g_bound = g.bound()
        g_mag = max(abs(g_bound.lo), abs(g_bound.hi))
        exp_g_mag = np.exp(g_mag)
        rem_bound = exp_c0 * exp_g_mag * g_mag ** (self.order + 1)
        for i in range(1, self.order + 2):
            rem_bound /= i
        result.remainder = result.remainder + Interval(-rem_bound, rem_bound)
        return result
    
    def log(self) -> 'TaylorModel':
        """Compute log via Taylor expansion."""
        b = self.bound()
        if b.lo <= 0:
            raise ValueError("Log of non-positive TM")
        c0 = self.constant_term
        if c0 <= 0:
            return TaylorModel.from_interval(self.dim, self.order,
                                            b.log(), self.domain)
        g = (self - c0) / c0
        result = TaylorModel.constant(self.dim, self.order, np.log(c0), self.domain)
        for k in range(1, self.order + 1):
            sign = (-1.0) ** (k + 1)
            result = result + (g ** k) * (sign / k)
        g_bound = g.bound()
        g_mag = max(abs(g_bound.lo), abs(g_bound.hi))
        if g_mag < 1.0:
            rem_bound = g_mag ** (self.order + 1) / (self.order + 1) / (1 - g_mag)
            result.remainder = result.remainder + Interval(-rem_bound, rem_bound)
        else:
            result = TaylorModel.from_interval(self.dim, self.order,
                                              b.log(), self.domain)
        return result
    
    def sin(self) -> 'TaylorModel':
        """Compute sin via Taylor expansion."""
        c0 = self.constant_term
        sin_c0, cos_c0 = np.sin(c0), np.cos(c0)
        g = self - c0
        result = TaylorModel.constant(self.dim, self.order, sin_c0, self.domain)
        factorial = 1.0
        for k in range(1, self.order + 1):
            factorial *= k
            if k % 4 == 1:
                c = cos_c0
            elif k % 4 == 2:
                c = -sin_c0
            elif k % 4 == 3:
                c = -cos_c0
            else:
                c = sin_c0
            result = result + (g ** k) * (c / factorial)
        g_bound = g.bound()
        g_mag = max(abs(g_bound.lo), abs(g_bound.hi))
        rem_bound = g_mag ** (self.order + 1)
        for i in range(1, self.order + 2):
            rem_bound /= i
        result.remainder = result.remainder + Interval(-rem_bound, rem_bound)
        return result
    
    def cos(self) -> 'TaylorModel':
        """Compute cos via Taylor expansion."""
        c0 = self.constant_term
        sin_c0, cos_c0 = np.sin(c0), np.cos(c0)
        g = self - c0
        result = TaylorModel.constant(self.dim, self.order, cos_c0, self.domain)
        factorial = 1.0
        for k in range(1, self.order + 1):
            factorial *= k
            if k % 4 == 1:
                c = -sin_c0
            elif k % 4 == 2:
                c = -cos_c0
            elif k % 4 == 3:
                c = sin_c0
            else:
                c = cos_c0
            result = result + (g ** k) * (c / factorial)
        g_bound = g.bound()
        g_mag = max(abs(g_bound.lo), abs(g_bound.hi))
        rem_bound = g_mag ** (self.order + 1)
        for i in range(1, self.order + 2):
            rem_bound /= i
        result.remainder = result.remainder + Interval(-rem_bound, rem_bound)
        return result
    
    def hill_function(self, K: 'TaylorModel', n: int) -> 'TaylorModel':
        """Compute Hill function x^n / (K^n + x^n)."""
        xn = self ** n
        Kn = K ** n
        return xn / (Kn + xn)
    
    def truncate(self, new_order: int) -> 'TaylorModel':
        """Truncate to lower order, absorbing high-order terms into remainder."""
        new_coeffs = {}
        extra_rem = Interval(0.0)
        for idx, c in self.coefficients.items():
            if idx.total_degree <= new_order:
                new_coeffs[idx] = c
            else:
                term = Interval(c)
                for j in range(self.dim):
                    if idx.indices[j] > 0:
                        term = term * (self.domain[j] ** idx.indices[j])
                extra_rem = extra_rem + term
        return TaylorModel(self.dim, new_order, new_coeffs,
                         self.remainder + extra_rem, list(self.domain))
    
    def shrink_wrap(self, factor: float = 1.1) -> 'TaylorModel':
        """Shrink-wrap: absorb remainder into polynomial coefficients."""
        if abs(self.remainder.mid) < 1e-300 and self.remainder.rad < 1e-300:
            return TaylorModel(self.dim, self.order,
                             dict(self.coefficients),
                             Interval(0.0), list(self.domain))
        new_coeffs = dict(self.coefficients)
        for idx in list(new_coeffs.keys()):
            if idx.total_degree > 0:
                new_coeffs[idx] *= factor
        zero_idx = MultiIndex(tuple(0 for _ in range(self.dim)))
        new_coeffs[zero_idx] = new_coeffs.get(zero_idx, 0.0) + self.remainder.mid
        absorbed = self.remainder.rad * (factor - 1.0)
        new_rem = Interval(-absorbed, absorbed) if absorbed > 0 else Interval(0.0)
        return TaylorModel(self.dim, self.order, new_coeffs, new_rem,
                         list(self.domain))
    
    def __repr__(self):
        n_terms = len(self.coefficients)
        return f"TaylorModel(dim={self.dim}, order={self.order}, terms={n_terms}, rem={self.remainder})"


class TaylorModelVector:
    """Vector of Taylor models for system-level operations."""
    
    def __init__(self, components: List[TaylorModel]):
        self.components = list(components)
    
    @classmethod
    def identity(cls, dim: int, order: int,
                 centers: Optional[np.ndarray] = None,
                 domain: Optional[List[Interval]] = None) -> 'TaylorModelVector':
        """Create identity TMV."""
        return cls(TaylorModel.identity(dim, order, centers, domain))
    
    @classmethod
    def constant(cls, dim: int, order: int, values: np.ndarray,
                 domain: Optional[List[Interval]] = None) -> 'TaylorModelVector':
        """Create constant TMV."""
        return cls([TaylorModel.constant(dim, order, float(v), domain) for v in values])
    
    @property
    def dim(self) -> int:
        return len(self.components)
    
    def __getitem__(self, i) -> TaylorModel:
        return self.components[i]
    
    def __setitem__(self, i, val: TaylorModel):
        self.components[i] = val
    
    def __len__(self):
        return len(self.components)
    
    def __add__(self, other):
        if isinstance(other, TaylorModelVector):
            return TaylorModelVector([a + b for a, b in
                                    zip(self.components, other.components)])
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, TaylorModelVector):
            return TaylorModelVector([a - b for a, b in
                                    zip(self.components, other.components)])
        return NotImplemented
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return TaylorModelVector([c * scalar for c in self.components])
        return NotImplemented
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def evaluate(self, point: np.ndarray) -> IntervalVector:
        """Evaluate TMV at point."""
        return IntervalVector([c.evaluate(point) for c in self.components])
    
    def evaluate_interval(self, box: List[Interval]) -> IntervalVector:
        """Evaluate TMV over interval box."""
        return IntervalVector([c.evaluate_interval(box) for c in self.components])
    
    def bound(self) -> IntervalVector:
        """Compute bounds over domain."""
        return IntervalVector([c.bound() for c in self.components])
    
    def remainder_vector(self) -> IntervalVector:
        """Get remainder vector."""
        return IntervalVector([c.remainder for c in self.components])
    
    def max_remainder(self) -> float:
        """Maximum remainder magnitude."""
        return max(c.remainder.mag for c in self.components)
    
    def truncate(self, new_order: int) -> 'TaylorModelVector':
        """Truncate all components."""
        return TaylorModelVector([c.truncate(new_order) for c in self.components])
    
    def shrink_wrap(self, factor: float = 1.1) -> 'TaylorModelVector':
        """Apply shrink-wrapping to all components."""
        return TaylorModelVector([c.shrink_wrap(factor) for c in self.components])
    
    def midpoint(self) -> np.ndarray:
        """Return midpoint array."""
        return np.array([c.constant_term for c in self.components])
    
    def __repr__(self):
        return f"TaylorModelVector(dim={self.dim})"
