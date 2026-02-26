"""
Affine arithmetic for reduced overestimation.

Affine arithmetic tracks linear dependencies between variables,
providing tighter enclosures than standard interval arithmetic
for expressions with shared variables.
"""

import numpy as np
from typing import List, Optional, Dict
from .interval import Interval


_next_noise_id = 0


def _fresh_noise_id() -> int:
    """Generate fresh noise symbol identifier."""
    global _next_noise_id
    _next_noise_id += 1
    return _next_noise_id


def reset_noise_counter():
    """Reset the noise symbol counter."""
    global _next_noise_id
    _next_noise_id = 0


class AffineForm:
    """
    Affine form: x0 + sum_i(xi * eps_i) + delta
    
    where x0 is the central value, xi are noise coefficients,
    eps_i are noise symbols in [-1, 1], and delta is the
    accumulated rounding error.
    """
    
    def __init__(self, center: float = 0.0,
                 noise_coeffs: Optional[Dict[int, float]] = None,
                 delta: float = 0.0):
        self.center = center
        self.noise_coeffs = noise_coeffs if noise_coeffs is not None else {}
        self.delta = abs(delta)
    
    @classmethod
    def from_interval(cls, iv: Interval) -> 'AffineForm':
        """Create affine form from interval, introducing a new noise symbol."""
        center = iv.mid
        radius = iv.rad
        noise_id = _fresh_noise_id()
        if radius > 0:
            return cls(center, {noise_id: radius}, 0.0)
        return cls(center, {}, 0.0)
    
    @classmethod
    def constant(cls, value: float) -> 'AffineForm':
        """Create constant affine form."""
        return cls(value, {}, 0.0)
    
    @classmethod
    def from_point(cls, value: float) -> 'AffineForm':
        """Create point (degenerate) affine form."""
        return cls(value, {}, 0.0)
    
    def to_interval(self) -> Interval:
        """Convert to interval enclosure."""
        total_noise = sum(abs(c) for c in self.noise_coeffs.values())
        radius = total_noise + self.delta
        return Interval(self.center - radius, self.center + radius)
    
    @property
    def radius(self) -> float:
        """Total radius of the affine form."""
        return sum(abs(c) for c in self.noise_coeffs.values()) + self.delta
    
    def _all_noise_ids(self, other: 'AffineForm') -> set:
        """Get union of noise symbol IDs."""
        return set(self.noise_coeffs.keys()) | set(other.noise_coeffs.keys())
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return AffineForm(self.center + other, dict(self.noise_coeffs), self.delta)
        if isinstance(other, AffineForm):
            ids = self._all_noise_ids(other)
            new_coeffs = {}
            for i in ids:
                c1 = self.noise_coeffs.get(i, 0.0)
                c2 = other.noise_coeffs.get(i, 0.0)
                if c1 + c2 != 0.0:
                    new_coeffs[i] = c1 + c2
            return AffineForm(self.center + other.center, new_coeffs,
                            self.delta + other.delta)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return AffineForm(-self.center,
                         {k: -v for k, v in self.noise_coeffs.items()},
                         self.delta)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return AffineForm(self.center - other, dict(self.noise_coeffs), self.delta)
        if isinstance(other, AffineForm):
            return self + (-other)
        return NotImplemented
    
    def __rsub__(self, other):
        return (-self).__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_coeffs = {k: v * other for k, v in self.noise_coeffs.items()}
            return AffineForm(self.center * other, new_coeffs, self.delta * abs(other))
        if isinstance(other, AffineForm):
            ids = self._all_noise_ids(other)
            new_center = self.center * other.center
            new_coeffs = {}
            for i in ids:
                c1 = self.noise_coeffs.get(i, 0.0)
                c2 = other.noise_coeffs.get(i, 0.0)
                val = self.center * c2 + other.center * c1
                if val != 0.0:
                    new_coeffs[i] = val
            r1 = self.radius
            r2 = other.radius
            quadratic_error = r1 * r2
            new_delta = self.delta * abs(other.center) + other.delta * abs(self.center)
            new_delta += quadratic_error
            return AffineForm(new_center, new_coeffs, new_delta)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0.0:
                raise ZeroDivisionError("Division by zero")
            return self * (1.0 / other)
        if isinstance(other, AffineForm):
            other_iv = other.to_interval()
            if other_iv.contains(0.0):
                raise ZeroDivisionError("Division by affine form containing zero")
            recip = other.reciprocal()
            return self * recip
        return NotImplemented
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return AffineForm.constant(other) / self
        return NotImplemented
    
    def __pow__(self, n: int):
        if not isinstance(n, int):
            return self.to_interval() ** n
        if n == 0:
            return AffineForm.constant(1.0)
        if n == 1:
            return AffineForm(self.center, dict(self.noise_coeffs), self.delta)
        if n < 0:
            return AffineForm.constant(1.0) / (self ** (-n))
        result = AffineForm.constant(1.0)
        base = self
        exp = n
        while exp > 0:
            if exp % 2 == 1:
                result = result * base
            base = base * base
            exp //= 2
        return result
    
    def reciprocal(self) -> 'AffineForm':
        """Compute reciprocal 1/x using min-range approximation."""
        iv = self.to_interval()
        if iv.contains(0.0):
            raise ZeroDivisionError("Reciprocal of affine form containing zero")
        a, b = iv.lo, iv.hi
        fa = 1.0 / a
        fb = 1.0 / b
        alpha = -1.0 / (a * b)
        u = max(abs(fa - alpha * a), abs(fb - alpha * b))
        zeta = (fa + fb - alpha * (a + b)) / 2.0
        new_center = alpha * self.center + zeta
        new_coeffs = {k: alpha * v for k, v in self.noise_coeffs.items()}
        new_delta = self.delta * abs(alpha) + u / 2.0
        return AffineForm(new_center, new_coeffs, new_delta)
    
    def sqrt(self) -> 'AffineForm':
        """Compute sqrt using Chebyshev approximation."""
        iv = self.to_interval()
        if iv.lo < 0:
            raise ValueError("Square root of negative affine form")
        a, b = max(iv.lo, 1e-300), iv.hi
        if a == b:
            return AffineForm.constant(np.sqrt(a))
        sa, sb = np.sqrt(a), np.sqrt(b)
        alpha = (sb - sa) / (b - a)
        zeta_a = sa - alpha * a
        zeta_b = sb - alpha * b
        zeta = (zeta_a + zeta_b) / 2.0
        err = abs(zeta_a - zeta_b) / 2.0
        new_center = alpha * self.center + zeta
        new_coeffs = {k: alpha * v for k, v in self.noise_coeffs.items()}
        new_delta = self.delta * abs(alpha) + err
        return AffineForm(new_center, new_coeffs, new_delta)
    
    def exp(self) -> 'AffineForm':
        """Compute exp using Chebyshev approximation."""
        iv = self.to_interval()
        a, b = iv.lo, iv.hi
        ea, eb = np.exp(a), np.exp(b)
        alpha = (eb - ea) / (b - a) if b > a else ea
        zeta_a = ea - alpha * a
        zeta_b = eb - alpha * b
        zeta = (zeta_a + zeta_b) / 2.0
        err = abs(zeta_a - zeta_b) / 2.0
        new_center = alpha * self.center + zeta
        new_coeffs = {k: alpha * v for k, v in self.noise_coeffs.items()}
        new_delta = self.delta * abs(alpha) + err
        return AffineForm(new_center, new_coeffs, new_delta)
    
    def log(self) -> 'AffineForm':
        """Compute log using Chebyshev approximation."""
        iv = self.to_interval()
        if iv.lo <= 0:
            raise ValueError("Log of non-positive affine form")
        a, b = iv.lo, iv.hi
        la, lb = np.log(a), np.log(b)
        alpha = (lb - la) / (b - a) if b > a else 1.0 / a
        zeta_a = la - alpha * a
        zeta_b = lb - alpha * b
        zeta = (zeta_a + zeta_b) / 2.0
        err = abs(zeta_a - zeta_b) / 2.0
        new_center = alpha * self.center + zeta
        new_coeffs = {k: alpha * v for k, v in self.noise_coeffs.items()}
        new_delta = self.delta * abs(alpha) + err
        return AffineForm(new_center, new_coeffs, new_delta)
    
    def sin(self) -> 'AffineForm':
        """Compute sin using Chebyshev approximation."""
        iv = self.to_interval()
        a, b = iv.lo, iv.hi
        sa, sb = np.sin(a), np.sin(b)
        ca = np.cos((a + b) / 2.0)
        alpha = ca if b > a else np.cos(a)
        if b - a > np.pi:
            return AffineForm.from_interval(Interval(-1.0, 1.0))
        zeta = (sa + sb) / 2.0 - alpha * (a + b) / 2.0
        err = max(abs(sa - alpha * a - zeta), abs(sb - alpha * b - zeta))
        new_center = alpha * self.center + zeta
        new_coeffs = {k: alpha * v for k, v in self.noise_coeffs.items()}
        new_delta = self.delta * abs(alpha) + err
        return AffineForm(new_center, new_coeffs, new_delta)
    
    def cos(self) -> 'AffineForm':
        """Compute cos using Chebyshev approximation."""
        iv = self.to_interval()
        a, b = iv.lo, iv.hi
        ca, cb = np.cos(a), np.cos(b)
        neg_sa = -np.sin((a + b) / 2.0)
        alpha = neg_sa if b > a else -np.sin(a)
        if b - a > np.pi:
            return AffineForm.from_interval(Interval(-1.0, 1.0))
        zeta = (ca + cb) / 2.0 - alpha * (a + b) / 2.0
        err = max(abs(ca - alpha * a - zeta), abs(cb - alpha * b - zeta))
        new_center = alpha * self.center + zeta
        new_coeffs = {k: alpha * v for k, v in self.noise_coeffs.items()}
        new_delta = self.delta * abs(alpha) + err
        return AffineForm(new_center, new_coeffs, new_delta)
    
    def hill_function(self, K: 'AffineForm', n: int) -> 'AffineForm':
        """Compute Hill function x^n / (K^n + x^n)."""
        xn = self ** n
        Kn = K ** n
        return xn / (Kn + xn)
    
    def michaelis_menten(self, Km: 'AffineForm', Vmax: 'AffineForm') -> 'AffineForm':
        """Compute Michaelis-Menten: Vmax * x / (Km + x)."""
        return Vmax * self / (Km + self)
    
    def __repr__(self):
        terms = [f"{self.center:.6e}"]
        for k, v in sorted(self.noise_coeffs.items()):
            terms.append(f"{v:+.4e}*eps_{k}")
        if self.delta > 0:
            terms.append(f"±{self.delta:.4e}")
        return "AffineForm(" + " ".join(terms) + ")"
    
    def __str__(self):
        iv = self.to_interval()
        return f"Affine~{iv}"


class AffineVector:
    """Vector of affine forms."""
    
    def __init__(self, components: List[AffineForm]):
        self.components = list(components)
    
    @classmethod
    def from_interval_vector(cls, iv_vec) -> 'AffineVector':
        """Create from IntervalVector."""
        return cls([AffineForm.from_interval(iv) for iv in iv_vec.components])
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'AffineVector':
        """Create from numpy array."""
        return cls([AffineForm.constant(float(x)) for x in arr])
    
    @property
    def n(self) -> int:
        return len(self.components)
    
    def __getitem__(self, i) -> AffineForm:
        return self.components[i]
    
    def __setitem__(self, i, val: AffineForm):
        self.components[i] = val
    
    def __len__(self):
        return len(self.components)
    
    def __add__(self, other):
        if isinstance(other, AffineVector):
            return AffineVector([a + b for a, b in zip(self.components, other.components)])
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, AffineVector):
            return AffineVector([a - b for a, b in zip(self.components, other.components)])
        return NotImplemented
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return AffineVector([a * scalar for a in self.components])
        if isinstance(scalar, AffineForm):
            return AffineVector([a * scalar for a in self.components])
        return NotImplemented
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def to_interval_vector(self):
        """Convert to IntervalVector."""
        from .matrix import IntervalVector
        return IntervalVector([af.to_interval() for af in self.components])
    
    def midpoint(self) -> np.ndarray:
        """Return midpoint array."""
        return np.array([af.center for af in self.components])
    
    def max_radius(self) -> float:
        """Maximum component radius."""
        return max(af.radius for af in self.components)
