"""
usability_oracle.algebra.differential — Differential cost algebra.

Implements **automatic differentiation** (AD) through cost computations,
enabling exact sensitivity analysis of usability verdicts with respect to
model parameters.

Two modes are provided:

* **Forward mode** (dual numbers): efficient for few inputs, many outputs.
* **Reverse mode** (tape-based): efficient for many inputs, few outputs.

Higher-order derivatives are supported via **hyperdual numbers**.

Mathematical Basis
------------------
A *dual number* is ``a + bε`` where ``ε² = 0``.  Evaluating a function
``f`` on dual numbers yields ``f(a + bε) = f(a) + f'(a)·b·ε``, giving
the derivative for free.

A *hyperdual number* ``a + bε₁ + cε₂ + dε₁ε₂`` carries second-order
derivative information (``d = f''(a)·b·c``).

Application
~~~~~~~~~~~
* Sensitivity of μ_{composed} to individual step parameters.
* Gradient of the usability verdict w.r.t. coupling/interference.
* Jacobian of the full cost vector ``(μ, σ², κ, λ)`` w.r.t. parameter vector.

References
----------
* Griewank & Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008.
* Fike & Alonso, *The Development of Hyper-Dual Numbers for Exact
  Second-Derivative Calculations*, AIAA 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.algebra.models import CostElement

# ---------------------------------------------------------------------------
# Dual numbers (forward mode AD)
# ---------------------------------------------------------------------------


@dataclass
class DualNumber:
    r"""A dual number ``a + b·ε`` with ``ε² = 0``.

    Parameters
    ----------
    real : float
        The primal value.
    dual : float
        The tangent (derivative) value.
    """

    real: float = 0.0
    dual: float = 0.0

    def __add__(self, other: "DualNumber | float") -> "DualNumber":
        if isinstance(other, (int, float)):
            return DualNumber(self.real + other, self.dual)
        return DualNumber(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other: float) -> "DualNumber":
        return self.__add__(other)

    def __sub__(self, other: "DualNumber | float") -> "DualNumber":
        if isinstance(other, (int, float)):
            return DualNumber(self.real - other, self.dual)
        return DualNumber(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other: float) -> "DualNumber":
        return DualNumber(other - self.real, -self.dual)

    def __mul__(self, other: "DualNumber | float") -> "DualNumber":
        if isinstance(other, (int, float)):
            return DualNumber(self.real * other, self.dual * other)
        # (a + bε)(c + dε) = ac + (ad + bc)ε
        return DualNumber(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real,
        )

    def __rmul__(self, other: float) -> "DualNumber":
        return self.__mul__(other)

    def __truediv__(self, other: "DualNumber | float") -> "DualNumber":
        if isinstance(other, (int, float)):
            return DualNumber(self.real / other, self.dual / other)
        # (a + bε)/(c + dε) = a/c + (bc - ad)/c² · ε
        return DualNumber(
            self.real / other.real,
            (self.dual * other.real - self.real * other.dual) / (other.real ** 2),
        )

    def __pow__(self, n: float) -> "DualNumber":
        # (a + bε)^n = a^n + n·a^(n-1)·b·ε
        if self.real == 0 and n < 1:
            return DualNumber(0.0, 0.0)
        return DualNumber(
            self.real ** n,
            n * self.real ** (n - 1) * self.dual,
        )

    def __neg__(self) -> "DualNumber":
        return DualNumber(-self.real, -self.dual)

    def __abs__(self) -> "DualNumber":
        if self.real >= 0:
            return DualNumber(self.real, self.dual)
        return DualNumber(-self.real, -self.dual)

    def __repr__(self) -> str:
        return f"Dual({self.real:.6f} + {self.dual:.6f}ε)"


def dual_sqrt(x: DualNumber) -> DualNumber:
    """Square root of a dual number: ``√(a + bε) = √a + b/(2√a)·ε``."""
    if x.real <= 0:
        return DualNumber(0.0, 0.0)
    s = math.sqrt(x.real)
    return DualNumber(s, x.dual / (2.0 * s))


def dual_max(a: DualNumber, b: DualNumber) -> DualNumber:
    """Max of two dual numbers (derivative follows the argmax)."""
    if a.real >= b.real:
        return a
    return b


def dual_min(a: DualNumber, b: DualNumber) -> DualNumber:
    """Min of two dual numbers (derivative follows the argmin)."""
    if a.real <= b.real:
        return a
    return b


def dual_log(x: DualNumber) -> DualNumber:
    """Natural log of a dual number."""
    if x.real <= 0:
        return DualNumber(float("-inf"), 0.0)
    return DualNumber(math.log(x.real), x.dual / x.real)


def dual_exp(x: DualNumber) -> DualNumber:
    """Exponential of a dual number."""
    e = math.exp(x.real)
    return DualNumber(e, x.dual * e)


# ---------------------------------------------------------------------------
# Dual cost element
# ---------------------------------------------------------------------------


@dataclass
class DualCostElement:
    r"""A cost element with dual-number entries for forward-mode AD.

    Each component ``(μ, σ², κ, λ)`` is a :class:`DualNumber`, carrying
    both the primal value and a directional derivative.
    """

    mu: DualNumber = field(default_factory=DualNumber)
    sigma_sq: DualNumber = field(default_factory=DualNumber)
    kappa: DualNumber = field(default_factory=DualNumber)
    lambda_: DualNumber = field(default_factory=DualNumber)

    def primal(self) -> CostElement:
        """Extract the primal (value) part."""
        return CostElement(
            mu=self.mu.real,
            sigma_sq=self.sigma_sq.real,
            kappa=self.kappa.real,
            lambda_=max(0.0, min(1.0, self.lambda_.real)),
        )

    def tangent(self) -> CostElement:
        """Extract the tangent (derivative) part."""
        return CostElement(
            mu=self.mu.dual,
            sigma_sq=self.sigma_sq.dual,
            kappa=self.kappa.dual,
            lambda_=self.lambda_.dual,
        )

    @classmethod
    def from_cost_element(
        cls,
        ce: CostElement,
        seed: Optional[CostElement] = None,
    ) -> "DualCostElement":
        """Lift a :class:`CostElement` to a dual cost element.

        Parameters
        ----------
        ce : CostElement
            The primal value.
        seed : CostElement | None
            The tangent seed. If None, defaults to zero tangent.
        """
        s = seed or CostElement.zero()
        return cls(
            mu=DualNumber(ce.mu, s.mu),
            sigma_sq=DualNumber(ce.sigma_sq, s.sigma_sq),
            kappa=DualNumber(ce.kappa, s.kappa),
            lambda_=DualNumber(ce.lambda_, s.lambda_),
        )

    def __repr__(self) -> str:
        return f"DualCost(μ={self.mu}, σ²={self.sigma_sq})"


# ---------------------------------------------------------------------------
# Forward mode: sequential and parallel composition on dual cost elements
# ---------------------------------------------------------------------------


def dual_sequential_compose(
    a: DualCostElement,
    b: DualCostElement,
    coupling: DualNumber,
) -> DualCostElement:
    r"""Sequential composition lifted to dual numbers.

    Implements the exact same formulas as :class:`SequentialComposer`
    but with dual arithmetic, so that derivatives propagate automatically.
    """
    sqrt_cross = dual_sqrt(a.sigma_sq * b.sigma_sq)

    mu = a.mu + b.mu + coupling * sqrt_cross
    sigma_sq = a.sigma_sq + b.sigma_sq + DualNumber(2.0) * coupling * sqrt_cross

    # Skewness
    total_var = sigma_sq
    if total_var.real > 1e-15:
        kappa = (a.kappa * a.sigma_sq ** 1.5 + b.kappa * b.sigma_sq ** 1.5) / total_var ** 1.5
    else:
        kappa = DualNumber(0.0, 0.0)

    # Tail risk
    lambda_ = dual_max(a.lambda_, b.lambda_) + coupling * dual_min(a.lambda_, b.lambda_)

    return DualCostElement(mu=mu, sigma_sq=sigma_sq, kappa=kappa, lambda_=lambda_)


def dual_parallel_compose(
    a: DualCostElement,
    b: DualCostElement,
    interference: DualNumber,
) -> DualCostElement:
    r"""Parallel composition lifted to dual numbers."""
    mu = dual_max(a.mu, b.mu) + interference * dual_min(a.mu, b.mu)

    sigma_sq = dual_max(a.sigma_sq, b.sigma_sq) + (
        interference * interference * dual_min(a.sigma_sq, b.sigma_sq)
    )

    if a.sigma_sq.real >= b.sigma_sq.real:
        kappa = a.kappa
    else:
        kappa = b.kappa

    lambda_ = a.lambda_ + b.lambda_ + interference * a.lambda_ * b.lambda_

    return DualCostElement(mu=mu, sigma_sq=sigma_sq, kappa=kappa, lambda_=lambda_)


# ---------------------------------------------------------------------------
# Hyperdual numbers (second-order derivatives)
# ---------------------------------------------------------------------------


@dataclass
class HyperDualNumber:
    r"""A hyperdual number ``a + b·ε₁ + c·ε₂ + d·ε₁ε₂``.

    Carries both first and second derivative information:
    * ``real``: function value ``f(x)``
    * ``eps1``: first derivative ``f'(x)``
    * ``eps2``: first derivative (second direction) ``f'(x)``
    * ``eps12``: second derivative ``f''(x)``
    """

    real: float = 0.0
    eps1: float = 0.0
    eps2: float = 0.0
    eps12: float = 0.0

    def __add__(self, other: "HyperDualNumber | float") -> "HyperDualNumber":
        if isinstance(other, (int, float)):
            return HyperDualNumber(self.real + other, self.eps1, self.eps2, self.eps12)
        return HyperDualNumber(
            self.real + other.real,
            self.eps1 + other.eps1,
            self.eps2 + other.eps2,
            self.eps12 + other.eps12,
        )

    def __radd__(self, other: float) -> "HyperDualNumber":
        return self.__add__(other)

    def __sub__(self, other: "HyperDualNumber | float") -> "HyperDualNumber":
        if isinstance(other, (int, float)):
            return HyperDualNumber(self.real - other, self.eps1, self.eps2, self.eps12)
        return HyperDualNumber(
            self.real - other.real,
            self.eps1 - other.eps1,
            self.eps2 - other.eps2,
            self.eps12 - other.eps12,
        )

    def __mul__(self, other: "HyperDualNumber | float") -> "HyperDualNumber":
        if isinstance(other, (int, float)):
            return HyperDualNumber(
                self.real * other, self.eps1 * other, self.eps2 * other, self.eps12 * other
            )
        return HyperDualNumber(
            self.real * other.real,
            self.real * other.eps1 + self.eps1 * other.real,
            self.real * other.eps2 + self.eps2 * other.real,
            (self.real * other.eps12 + self.eps1 * other.eps2
             + self.eps2 * other.eps1 + self.eps12 * other.real),
        )

    def __rmul__(self, other: float) -> "HyperDualNumber":
        return self.__mul__(other)

    def __truediv__(self, other: "HyperDualNumber | float") -> "HyperDualNumber":
        if isinstance(other, (int, float)):
            return HyperDualNumber(
                self.real / other, self.eps1 / other, self.eps2 / other, self.eps12 / other
            )
        inv = 1.0 / other.real
        inv2 = inv * inv
        return HyperDualNumber(
            self.real * inv,
            (self.eps1 * other.real - self.real * other.eps1) * inv2,
            (self.eps2 * other.real - self.real * other.eps2) * inv2,
            (self.eps12 * other.real - self.real * other.eps12
             - self.eps1 * other.eps2 - self.eps2 * other.eps1
             + 2.0 * self.real * other.eps1 * other.eps2 * inv) * inv2,
        )

    def __pow__(self, n: float) -> "HyperDualNumber":
        if self.real == 0 and n < 1:
            return HyperDualNumber()
        f = self.real ** n
        fp = n * self.real ** (n - 1)
        fpp = n * (n - 1) * self.real ** (n - 2) if n >= 2 else 0.0
        return HyperDualNumber(
            f,
            fp * self.eps1,
            fp * self.eps2,
            fpp * self.eps1 * self.eps2 + fp * self.eps12,
        )

    def __repr__(self) -> str:
        return (
            f"HDual({self.real:.4f} + {self.eps1:.4f}ε₁ "
            f"+ {self.eps2:.4f}ε₂ + {self.eps12:.4f}ε₁ε₂)"
        )


def hyperdual_sqrt(x: HyperDualNumber) -> HyperDualNumber:
    """Square root lifted to hyperdual numbers."""
    if x.real <= 0:
        return HyperDualNumber()
    s = math.sqrt(x.real)
    ds = 0.5 / s
    dds = -0.25 / (x.real * s)
    return HyperDualNumber(
        s,
        ds * x.eps1,
        ds * x.eps2,
        dds * x.eps1 * x.eps2 + ds * x.eps12,
    )


# ---------------------------------------------------------------------------
# Reverse mode AD (tape-based)
# ---------------------------------------------------------------------------


class _TapeEntry:
    """An entry in the reverse-mode AD tape."""

    __slots__ = ("value", "adjoint", "parents", "name")

    def __init__(self, value: float, parents: List[Tuple[float, "_TapeEntry"]],
                 name: str = "") -> None:
        self.value = value
        self.adjoint = 0.0
        self.parents = parents
        self.name = name

    def __repr__(self) -> str:
        return f"Tape({self.name}={self.value:.6f}, adj={self.adjoint:.6f})"


class ReverseModeAD:
    r"""Tape-based reverse-mode automatic differentiation.

    Records a computation trace, then backpropagates to compute gradients
    of a scalar output w.r.t. all inputs.

    Usage::

        ad = ReverseModeAD()
        x = ad.variable(3.0, "x")
        y = ad.variable(2.0, "y")
        z = ad.add(ad.mul(x, y), x)  # z = x*y + x
        grads = ad.backward(z)
        # grads["x"] == y + 1 == 3.0
        # grads["y"] == x == 3.0
    """

    def __init__(self) -> None:
        self._tape: List[_TapeEntry] = []

    def variable(self, value: float, name: str = "") -> _TapeEntry:
        """Create an input variable on the tape."""
        entry = _TapeEntry(value, [], name=name)
        self._tape.append(entry)
        return entry

    def constant(self, value: float) -> _TapeEntry:
        """Create a constant (no gradient flows through it)."""
        entry = _TapeEntry(value, [], name="const")
        self._tape.append(entry)
        return entry

    def add(self, a: _TapeEntry, b: _TapeEntry) -> _TapeEntry:
        """``a + b``."""
        entry = _TapeEntry(a.value + b.value, [(1.0, a), (1.0, b)])
        self._tape.append(entry)
        return entry

    def sub(self, a: _TapeEntry, b: _TapeEntry) -> _TapeEntry:
        """``a - b``."""
        entry = _TapeEntry(a.value - b.value, [(1.0, a), (-1.0, b)])
        self._tape.append(entry)
        return entry

    def mul(self, a: _TapeEntry, b: _TapeEntry) -> _TapeEntry:
        """``a * b``."""
        entry = _TapeEntry(a.value * b.value, [(b.value, a), (a.value, b)])
        self._tape.append(entry)
        return entry

    def div(self, a: _TapeEntry, b: _TapeEntry) -> _TapeEntry:
        """``a / b``."""
        val = a.value / b.value if b.value != 0 else 0.0
        da = 1.0 / b.value if b.value != 0 else 0.0
        db = -a.value / (b.value ** 2) if b.value != 0 else 0.0
        entry = _TapeEntry(val, [(da, a), (db, b)])
        self._tape.append(entry)
        return entry

    def sqrt(self, a: _TapeEntry) -> _TapeEntry:
        """``√a``."""
        if a.value <= 0:
            entry = _TapeEntry(0.0, [])
            self._tape.append(entry)
            return entry
        s = math.sqrt(a.value)
        entry = _TapeEntry(s, [(0.5 / s, a)])
        self._tape.append(entry)
        return entry

    def max(self, a: _TapeEntry, b: _TapeEntry) -> _TapeEntry:
        """``max(a, b)`` with subgradient."""
        if a.value >= b.value:
            entry = _TapeEntry(a.value, [(1.0, a), (0.0, b)])
        else:
            entry = _TapeEntry(b.value, [(0.0, a), (1.0, b)])
        self._tape.append(entry)
        return entry

    def min(self, a: _TapeEntry, b: _TapeEntry) -> _TapeEntry:
        """``min(a, b)`` with subgradient."""
        if a.value <= b.value:
            entry = _TapeEntry(a.value, [(1.0, a), (0.0, b)])
        else:
            entry = _TapeEntry(b.value, [(0.0, a), (1.0, b)])
        self._tape.append(entry)
        return entry

    def scale(self, a: _TapeEntry, c: float) -> _TapeEntry:
        """``c * a``."""
        entry = _TapeEntry(c * a.value, [(c, a)])
        self._tape.append(entry)
        return entry

    def backward(self, output: _TapeEntry) -> Dict[str, float]:
        r"""Backpropagate from ``output`` to compute all gradients.

        Returns
        -------
        dict[str, float]
            Map from variable name to gradient value.
        """
        # Reset all adjoints
        for entry in self._tape:
            entry.adjoint = 0.0
        output.adjoint = 1.0

        # Reverse sweep
        for entry in reversed(self._tape):
            for weight, parent in entry.parents:
                parent.adjoint += weight * entry.adjoint

        # Collect named variable gradients
        grads: Dict[str, float] = {}
        for entry in self._tape:
            if entry.name and entry.name != "const":
                grads[entry.name] = entry.adjoint
        return grads


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------


def cost_jacobian(
    compose_fn: Callable[..., CostElement],
    params: Dict[str, float],
    delta: float = 1e-7,
) -> Dict[str, np.ndarray]:
    r"""Compute the Jacobian of a cost composition function.

    Uses central finite differences as a fallback when AD is not wired
    through the full composition pipeline.

    Parameters
    ----------
    compose_fn : callable
        A function ``(**params) → CostElement`` that computes a cost.
    params : dict[str, float]
        Parameter name → value map.
    delta : float
        Finite difference step size.

    Returns
    -------
    dict[str, np.ndarray]
        Map from parameter name to a 4-vector of partial derivatives
        ``[∂μ/∂p, ∂σ²/∂p, ∂κ/∂p, ∂λ/∂p]``.
    """
    base = compose_fn(**params)
    base_arr = base.as_array

    jacobian: Dict[str, np.ndarray] = {}
    for name, val in params.items():
        params_plus = dict(params)
        params_plus[name] = val + delta
        params_minus = dict(params)
        params_minus[name] = val - delta

        f_plus = compose_fn(**params_plus).as_array
        f_minus = compose_fn(**params_minus).as_array
        jacobian[name] = (f_plus - f_minus) / (2.0 * delta)

    return jacobian


def sensitivity_report(
    compose_fn: Callable[..., CostElement],
    params: Dict[str, float],
    delta: float = 1e-7,
) -> Dict[str, Dict[str, float]]:
    r"""Generate a sensitivity report: normalised elasticities.

    Elasticity: ``E_{p}(μ) = (∂μ/∂p) · (p/μ)`` — percentage change in
    output per percentage change in parameter.

    Parameters
    ----------
    compose_fn : callable
        Cost composition function.
    params : dict[str, float]
        Parameter values.

    Returns
    -------
    dict[str, dict[str, float]]
        Map from parameter name to ``{"mu": elasticity, "sigma_sq": …, …}``.
    """
    jac = cost_jacobian(compose_fn, params, delta=delta)
    base = compose_fn(**params)
    fields = ["mu", "sigma_sq", "kappa", "lambda_"]
    base_vals = [base.mu, base.sigma_sq, base.kappa, base.lambda_]

    report: Dict[str, Dict[str, float]] = {}
    for name, grad in jac.items():
        p = params[name]
        elasticities: Dict[str, float] = {}
        for i, field_name in enumerate(fields):
            bv = base_vals[i]
            if abs(bv) > 1e-15 and abs(p) > 1e-15:
                elasticities[field_name] = grad[i] * p / bv
            else:
                elasticities[field_name] = 0.0
        report[name] = elasticities
    return report
