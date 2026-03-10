"""Affine arithmetic for tighter interval bounds.

Affine arithmetic represents each quantity as an affine form:

    x̂ = x₀ + x₁ε₁ + x₂ε₂ + ⋯ + xₙεₙ

where εᵢ ∈ [−1, 1] are noise symbols shared across correlated
quantities.  This eliminates the *dependency problem* that plagues
standard interval arithmetic when a variable appears more than once in
an expression.

Affine forms track first-order correlations exactly; nonlinear
operations introduce a new independent noise term whose magnitude is
an upper bound on the linearisation error (Chebyshev approximation).

References
----------
Stolfi, J., & de Figueiredo, L. H. (2003).
    An introduction to affine arithmetic. *TEMA — Tendências em
    Matemática Aplicada e Computacional*, 4(3), 297–312.
de Figueiredo, L. H., & Stolfi, J. (2004).
    Affine arithmetic: Concepts and applications.
    *Numerical Algorithms*, 37, 147–158.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from usability_oracle.interval.interval import Interval


# ---------------------------------------------------------------------------
# Global noise-symbol counter
# ---------------------------------------------------------------------------

_next_noise_id: int = 0


def _new_noise_id() -> int:
    """Allocate a fresh, globally unique noise-symbol index."""
    global _next_noise_id
    nid = _next_noise_id
    _next_noise_id += 1
    return nid


def reset_noise_counter() -> None:
    """Reset the global noise-symbol counter (useful in tests)."""
    global _next_noise_id
    _next_noise_id = 0


# ---------------------------------------------------------------------------
# AffineForm
# ---------------------------------------------------------------------------

@dataclass
class AffineForm:
    """An affine form x̂ = x₀ + Σ xᵢεᵢ with εᵢ ∈ [−1, 1].

    Attributes
    ----------
    center : float
        Central value x₀.
    terms : Dict[int, float]
        Mapping from noise-symbol index *i* to coefficient xᵢ.
        Only non-zero coefficients are stored.
    """

    center: float = 0.0
    terms: Dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def radius(self) -> float:
        """Total deviation: Σ |xᵢ|."""
        return sum(abs(v) for v in self.terms.values())

    @property
    def num_terms(self) -> int:
        """Number of active noise symbols."""
        return len(self.terms)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_value(cls, value: float) -> AffineForm:
        """Create a degenerate affine form with no noise (exact value)."""
        return cls(center=float(value), terms={})

    @classmethod
    def from_interval(cls, interval: Interval) -> AffineForm:
        """Create an affine form from a standard interval.

        The interval [a, b] is represented as:

            x̂ = (a + b)/2 + ((b − a)/2) · ε_k

        where ε_k is a fresh noise symbol.

        Parameters
        ----------
        interval : Interval
            Source interval.

        Returns
        -------
        AffineForm
        """
        mid = (interval.low + interval.high) / 2.0
        rad = (interval.high - interval.low) / 2.0
        if rad == 0.0:
            return cls(center=mid, terms={})
        nid = _new_noise_id()
        return cls(center=mid, terms={nid: rad})

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_interval(self) -> Interval:
        """Convert this affine form to a standard interval.

        The interval is [x₀ − r, x₀ + r] where r = Σ |xᵢ|.

        Returns
        -------
        Interval
        """
        r = self.radius
        return Interval(self.center - r, self.center + r)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        terms_str = ", ".join(f"ε{k}: {v}" for k, v in sorted(self.terms.items()))
        return f"AffineForm(center={self.center}, terms={{{terms_str}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineForm):
            return NotImplemented
        return (
            self.center == other.center
            and self.terms == other.terms
        )


# ---------------------------------------------------------------------------
# Helper: merge noise terms
# ---------------------------------------------------------------------------

def _merge_keys(*dicts: Dict[int, float]) -> set[int]:
    """Return the union of all keys in the given dictionaries."""
    keys: set[int] = set()
    for d in dicts:
        keys.update(d.keys())
    return keys


def _clean_terms(terms: Dict[int, float], eps: float = 0.0) -> Dict[int, float]:
    """Remove terms whose absolute value is at most *eps*."""
    return {k: v for k, v in terms.items() if abs(v) > eps}


# ═══════════════════════════════════════════════════════════════════════════
# Arithmetic operations
# ═══════════════════════════════════════════════════════════════════════════

def add(a: AffineForm, b: AffineForm) -> AffineForm:
    """Affine addition: â + b̂.

    Because both operands share the same noise symbols, the addition
    preserves correlations exactly:

        â + b̂ = (a₀ + b₀) + Σ (aᵢ + bᵢ)εᵢ

    Parameters
    ----------
    a, b : AffineForm

    Returns
    -------
    AffineForm
    """
    new_center = a.center + b.center
    keys = _merge_keys(a.terms, b.terms)
    new_terms: Dict[int, float] = {}
    for k in keys:
        val = a.terms.get(k, 0.0) + b.terms.get(k, 0.0)
        if val != 0.0:
            new_terms[k] = val
    return AffineForm(center=new_center, terms=new_terms)


def subtract(a: AffineForm, b: AffineForm) -> AffineForm:
    """Affine subtraction: â − b̂.

    Parameters
    ----------
    a, b : AffineForm

    Returns
    -------
    AffineForm
    """
    new_center = a.center - b.center
    keys = _merge_keys(a.terms, b.terms)
    new_terms: Dict[int, float] = {}
    for k in keys:
        val = a.terms.get(k, 0.0) - b.terms.get(k, 0.0)
        if val != 0.0:
            new_terms[k] = val
    return AffineForm(center=new_center, terms=new_terms)


def negate(a: AffineForm) -> AffineForm:
    """Negate an affine form: −â.

    Parameters
    ----------
    a : AffineForm

    Returns
    -------
    AffineForm
    """
    return AffineForm(
        center=-a.center,
        terms={k: -v for k, v in a.terms.items()},
    )


def scale(a: AffineForm, alpha: float) -> AffineForm:
    """Scale an affine form by a scalar: α · â.

    Parameters
    ----------
    a : AffineForm
    alpha : float

    Returns
    -------
    AffineForm
    """
    alpha = float(alpha)
    if alpha == 0.0:
        return AffineForm(center=0.0, terms={})
    return AffineForm(
        center=alpha * a.center,
        terms={k: alpha * v for k, v in a.terms.items()},
    )


def multiply(a: AffineForm, b: AffineForm) -> AffineForm:
    """Affine multiplication: â · b̂.

    The product of two affine forms is not affine.  We linearise by
    keeping only first-order cross-terms and bounding the quadratic
    residual with a fresh noise symbol:

        â · b̂ ≈ a₀b₀ + Σ (a₀bᵢ + b₀aᵢ)εᵢ + δεₖ

    where δ = Σ|aᵢ| · Σ|bᵢ| bounds the quadratic error.

    Parameters
    ----------
    a, b : AffineForm

    Returns
    -------
    AffineForm
    """
    # Constant part
    new_center = a.center * b.center

    # First-order terms
    keys = _merge_keys(a.terms, b.terms)
    new_terms: Dict[int, float] = {}
    for k in keys:
        val = a.center * b.terms.get(k, 0.0) + b.center * a.terms.get(k, 0.0)
        if val != 0.0:
            new_terms[k] = val

    # Quadratic error bound
    ra = sum(abs(v) for v in a.terms.values())
    rb = sum(abs(v) for v in b.terms.values())
    delta = ra * rb

    if delta > 0.0:
        nid = _new_noise_id()
        new_terms[nid] = delta

    return AffineForm(center=new_center, terms=_clean_terms(new_terms))


def divide(a: AffineForm, b: AffineForm) -> AffineForm:
    """Affine division: â / b̂.

    Converts to â · (1/b̂) using the Chebyshev approximation for 1/x
    on the range of b̂.

    Parameters
    ----------
    a, b : AffineForm

    Returns
    -------
    AffineForm

    Raises
    ------
    ZeroDivisionError
        If the range of *b* contains zero.
    """
    b_interval = b.to_interval()
    if b_interval.low <= 0.0 <= b_interval.high:
        raise ZeroDivisionError(
            f"Cannot divide by an affine form whose range {b_interval} "
            "contains zero."
        )
    inv_b = _reciprocal_chebyshev(b)
    return multiply(a, inv_b)


def _reciprocal_chebyshev(a: AffineForm) -> AffineForm:
    """Chebyshev approximation of 1/x on the range of *a*.

    Computes the min-range affine approximation of f(x) = 1/x on
    [lo, hi], producing an affine form that encloses the true
    reciprocal.

    Parameters
    ----------
    a : AffineForm

    Returns
    -------
    AffineForm
    """
    iv = a.to_interval()
    lo, hi = iv.low, iv.high

    # Chebyshev slope for 1/x on [lo, hi]: α = −1/(lo·hi)
    alpha = -1.0 / (lo * hi)

    # Chebyshev intercept:
    # minimise max |1/x − (α·x + β)| over [lo, hi]
    # The Chebyshev approximation has equal ripple at endpoints and
    # interior extremum.
    f_lo = 1.0 / lo
    f_hi = 1.0 / hi
    p_lo = alpha * lo
    p_hi = alpha * hi

    # Error at endpoints
    e_lo = f_lo - p_lo
    e_hi = f_hi - p_hi

    beta = (e_lo + e_hi) / 2.0
    delta = abs(e_lo - e_hi) / 2.0

    # Interior extremum of 1/x − αx − β at x* = sqrt(−1/α) if in range
    if alpha < 0.0:
        x_star = math.sqrt(-1.0 / alpha)
        if lo <= x_star <= hi:
            f_star = 1.0 / x_star
            p_star = alpha * x_star + beta
            e_star = abs(f_star - p_star)
            delta = max(delta, e_star)

    # Build the affine form: α · â + β + δ·ε_new
    result_center = alpha * a.center + beta
    result_terms: Dict[int, float] = {k: alpha * v for k, v in a.terms.items()}

    if delta > 0.0:
        nid = _new_noise_id()
        result_terms[nid] = delta

    return AffineForm(center=result_center, terms=_clean_terms(result_terms))


def power(a: AffineForm, n: int) -> AffineForm:
    """Raise an affine form to a non-negative integer power.

    Uses repeated squaring for efficiency.  Each multiplication step
    introduces a linearisation error bounded by a fresh noise symbol.

    Parameters
    ----------
    a : AffineForm
        Base.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    AffineForm

    Raises
    ------
    ValueError
        If *n* < 0.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"Exponent must be a non-negative integer, got {n}.")
    if n == 0:
        return AffineForm.from_value(1.0)
    if n == 1:
        return AffineForm(center=a.center, terms=dict(a.terms))

    # Binary exponentiation
    result = AffineForm.from_value(1.0)
    base = AffineForm(center=a.center, terms=dict(a.terms))
    exp = n
    while exp > 0:
        if exp % 2 == 1:
            result = multiply(result, base)
        base = multiply(base, base)
        exp //= 2
    return result


def exp(a: AffineForm) -> AffineForm:
    """Exponential of an affine form using Chebyshev approximation.

    Approximates e^x by the best affine function on the range [lo, hi]
    of *a*, bounding the approximation error with a fresh noise symbol.

    Parameters
    ----------
    a : AffineForm

    Returns
    -------
    AffineForm
    """
    iv = a.to_interval()
    lo, hi = iv.low, iv.high

    if lo == hi:
        return AffineForm.from_value(math.exp(lo))

    f_lo = math.exp(lo)
    f_hi = math.exp(hi)

    # Secant slope
    alpha = (f_hi - f_lo) / (hi - lo)

    # The tangent point for min-max is where exp'(x) = alpha, i.e. x* = ln(alpha)
    if alpha > 0.0:
        x_star = math.log(alpha)
        f_star = alpha  # exp(ln(alpha)) = alpha
    else:
        x_star = lo
        f_star = f_lo

    # Affine approximation: alpha * x + beta
    # Choose beta so errors at endpoints are balanced
    e_lo = f_lo - alpha * lo
    e_hi = f_hi - alpha * hi

    beta = (e_lo + e_hi) / 2.0
    delta = abs(e_lo - e_hi) / 2.0

    # Check interior extremum
    if lo < x_star < hi:
        e_star = abs(f_star - (alpha * x_star + beta))
        delta = max(delta, e_star)

    # Build result: alpha * â + beta ± delta
    result_center = alpha * a.center + beta
    result_terms: Dict[int, float] = {k: alpha * v for k, v in a.terms.items()}

    if delta > 0.0:
        nid = _new_noise_id()
        result_terms[nid] = delta

    return AffineForm(center=result_center, terms=_clean_terms(result_terms))


def log(a: AffineForm) -> AffineForm:
    """Natural logarithm of an affine form using Chebyshev approximation.

    Approximates ln(x) by the best affine function on the range
    [lo, hi] of *a*.

    Parameters
    ----------
    a : AffineForm

    Returns
    -------
    AffineForm

    Raises
    ------
    ValueError
        If the range of *a* includes non-positive values.
    """
    iv = a.to_interval()
    lo, hi = iv.low, iv.high

    if lo <= 0.0:
        raise ValueError(
            f"log is undefined for affine forms with non-positive range: {iv}"
        )

    if lo == hi:
        return AffineForm.from_value(math.log(lo))

    f_lo = math.log(lo)
    f_hi = math.log(hi)

    # Secant slope
    alpha = (f_hi - f_lo) / (hi - lo)

    # Tangent point: ln'(x) = alpha ⟹ x* = 1/alpha
    x_star = 1.0 / alpha if alpha > 0.0 else lo

    # Affine approximation with balanced endpoint errors
    e_lo = f_lo - alpha * lo
    e_hi = f_hi - alpha * hi

    beta = (e_lo + e_hi) / 2.0
    delta = abs(e_lo - e_hi) / 2.0

    # Check interior extremum
    if lo < x_star < hi:
        f_star = math.log(x_star)
        e_star = abs(f_star - (alpha * x_star + beta))
        delta = max(delta, e_star)

    result_center = alpha * a.center + beta
    result_terms: Dict[int, float] = {k: alpha * v for k, v in a.terms.items()}

    if delta > 0.0:
        nid = _new_noise_id()
        result_terms[nid] = delta

    return AffineForm(center=result_center, terms=_clean_terms(result_terms))


def sqrt_affine(a: AffineForm) -> AffineForm:
    """Square root of an affine form using Chebyshev approximation.

    Parameters
    ----------
    a : AffineForm

    Returns
    -------
    AffineForm

    Raises
    ------
    ValueError
        If the range of *a* includes negative values.
    """
    iv = a.to_interval()
    lo, hi = iv.low, iv.high

    if lo < 0.0:
        raise ValueError(
            f"sqrt is undefined for affine forms with negative range: {iv}"
        )

    if lo == hi:
        return AffineForm.from_value(math.sqrt(lo))

    f_lo = math.sqrt(lo)
    f_hi = math.sqrt(hi)

    # Secant slope
    alpha = (f_hi - f_lo) / (hi - lo)

    # Tangent point: 1/(2√x) = alpha ⟹ x* = 1/(4α²)
    x_star = 1.0 / (4.0 * alpha * alpha) if alpha > 0.0 else lo

    e_lo = f_lo - alpha * lo
    e_hi = f_hi - alpha * hi

    beta = (e_lo + e_hi) / 2.0
    delta = abs(e_lo - e_hi) / 2.0

    if lo < x_star < hi:
        f_star = math.sqrt(x_star)
        e_star = abs(f_star - (alpha * x_star + beta))
        delta = max(delta, e_star)

    result_center = alpha * a.center + beta
    result_terms: Dict[int, float] = {k: alpha * v for k, v in a.terms.items()}

    if delta > 0.0:
        nid = _new_noise_id()
        result_terms[nid] = delta

    return AffineForm(center=result_center, terms=_clean_terms(result_terms))


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: interval ↔ affine
# ═══════════════════════════════════════════════════════════════════════════

def to_interval(a: AffineForm) -> Interval:
    """Convert an affine form to a standard interval.

    Convenience wrapper around :meth:`AffineForm.to_interval`.

    Parameters
    ----------
    a : AffineForm

    Returns
    -------
    Interval
    """
    return a.to_interval()


def from_interval(interval: Interval) -> AffineForm:
    """Convert a standard interval to an affine form.

    Convenience wrapper around :meth:`AffineForm.from_interval`.

    Parameters
    ----------
    interval : Interval

    Returns
    -------
    AffineForm
    """
    return AffineForm.from_interval(interval)
