"""
usability_oracle.algebra.lattice — Cost lattice structure.

Defines a **complete lattice** over cognitive cost elements, enabling:

* Partial ordering (⊑) of costs
* Join (⊔) and meet (⊓) operations
* Fixed-point computation via Kleene iteration and Tarski's theorem
* Galois connections for sound cost abstraction
* Widening / narrowing operators for convergence acceleration

Mathematical Structure
----------------------
The lattice ordering on :class:`CostElement` is component-wise:

.. math::

    (μ_1, σ²_1, κ_1, λ_1) ⊑ (μ_2, σ²_2, κ_2, λ_2)
    \\iff μ_1 ≤ μ_2 ∧ σ²_1 ≤ σ²_2 ∧ |κ_1| ≤ |κ_2| ∧ λ_1 ≤ λ_2

Join is the component-wise maximum (upper bound); meet is the
component-wise minimum (lower bound).

Application
~~~~~~~~~~~
* **Sound approximation**: abstract costs over-approximate concrete costs.
* **Fixed-point analysis**: iterative cost propagation in cyclic task graphs.
* **Widening**: accelerate convergence of iterative analyses.

References
----------
* Davey & Priestley, *Introduction to Lattices and Order*, Cambridge, 2002.
* Cousot & Cousot, *Abstract Interpretation*, POPL 1977.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from usability_oracle.algebra.models import CostElement

# ---------------------------------------------------------------------------
# Partial order
# ---------------------------------------------------------------------------

_COST_TOL = 1e-12


def cost_leq(a: CostElement, b: CostElement, tol: float = _COST_TOL) -> bool:
    r"""Partial order: ``a ⊑ b`` iff a is "cheaper or equal" in every dimension.

    .. math::

        a ⊑ b ⟺ μ_a ≤ μ_b ∧ σ²_a ≤ σ²_b ∧ |κ_a| ≤ |κ_b| ∧ λ_a ≤ λ_b
    """
    return (
        a.mu <= b.mu + tol
        and a.sigma_sq <= b.sigma_sq + tol
        and abs(a.kappa) <= abs(b.kappa) + tol
        and a.lambda_ <= b.lambda_ + tol
    )


def cost_lt(a: CostElement, b: CostElement, tol: float = _COST_TOL) -> bool:
    """Strict partial order: ``a ⊏ b`` iff ``a ⊑ b`` and ``a ≠ b``."""
    return cost_leq(a, b, tol) and not cost_eq(a, b, tol)


def cost_eq(a: CostElement, b: CostElement, tol: float = _COST_TOL) -> bool:
    """Equality under the lattice ordering."""
    return (
        abs(a.mu - b.mu) <= tol
        and abs(a.sigma_sq - b.sigma_sq) <= tol
        and abs(abs(a.kappa) - abs(b.kappa)) <= tol
        and abs(a.lambda_ - b.lambda_) <= tol
    )


def cost_comparable(a: CostElement, b: CostElement, tol: float = _COST_TOL) -> bool:
    """True if ``a`` and ``b`` are comparable (one ⊑ the other)."""
    return cost_leq(a, b, tol) or cost_leq(b, a, tol)


# ---------------------------------------------------------------------------
# Join and meet
# ---------------------------------------------------------------------------


def cost_join(a: CostElement, b: CostElement) -> CostElement:
    r"""Join (least upper bound): ``a ⊔ b``.

    .. math::

        a ⊔ b = (\max(μ_a, μ_b),\, \max(σ²_a, σ²_b),\,
                  \text{sign} · \max(|κ_a|, |κ_b|),\, \max(λ_a, λ_b))
    """
    kappa_abs = max(abs(a.kappa), abs(b.kappa))
    # Preserve sign from the element with larger absolute kappa
    kappa_sign = a.kappa if abs(a.kappa) >= abs(b.kappa) else b.kappa
    kappa = math.copysign(kappa_abs, kappa_sign) if kappa_abs > 0 else 0.0

    return CostElement(
        mu=max(a.mu, b.mu),
        sigma_sq=max(a.sigma_sq, b.sigma_sq),
        kappa=kappa,
        lambda_=max(a.lambda_, b.lambda_),
    )


def cost_meet(a: CostElement, b: CostElement) -> CostElement:
    r"""Meet (greatest lower bound): ``a ⊓ b``.

    .. math::

        a ⊓ b = (\min(μ_a, μ_b),\, \min(σ²_a, σ²_b),\,
                  \text{sign} · \min(|κ_a|, |κ_b|),\, \min(λ_a, λ_b))
    """
    kappa_abs = min(abs(a.kappa), abs(b.kappa))
    kappa_sign = a.kappa if abs(a.kappa) <= abs(b.kappa) else b.kappa
    kappa = math.copysign(kappa_abs, kappa_sign) if kappa_abs > 0 else 0.0

    return CostElement(
        mu=min(a.mu, b.mu),
        sigma_sq=min(a.sigma_sq, b.sigma_sq),
        kappa=kappa,
        lambda_=min(a.lambda_, b.lambda_),
    )


def cost_join_many(elements: Sequence[CostElement]) -> CostElement:
    """Join over a sequence: ``⊔ {e₁, e₂, …}``."""
    if not elements:
        return CostElement.zero()  # bottom element
    result = elements[0]
    for e in elements[1:]:
        result = cost_join(result, e)
    return result


def cost_meet_many(elements: Sequence[CostElement]) -> CostElement:
    """Meet over a sequence: ``⊓ {e₁, e₂, …}``."""
    if not elements:
        return cost_top()
    result = elements[0]
    for e in elements[1:]:
        result = cost_meet(result, e)
    return result


def cost_bottom() -> CostElement:
    """Bottom element ⊥ of the cost lattice (least element)."""
    return CostElement(mu=0.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)


def cost_top() -> CostElement:
    """Top element ⊤ of the cost lattice (greatest element)."""
    return CostElement(mu=float("inf"), sigma_sq=float("inf"), kappa=float("inf"), lambda_=1.0)


# ---------------------------------------------------------------------------
# Fixed-point computation
# ---------------------------------------------------------------------------


def kleene_fixpoint(
    f: Callable[[CostElement], CostElement],
    bottom: Optional[CostElement] = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> Tuple[CostElement, int]:
    r"""Compute the least fixed point of ``f`` via Kleene iteration.

    Starting from ⊥, iterates ``⊥, f(⊥), f(f(⊥)), …`` until convergence.

    The Kleene fixed-point theorem guarantees convergence when ``f`` is
    monotone and the lattice has no infinite strictly ascending chains
    (or with widening applied).

    Parameters
    ----------
    f : callable
        A monotone function ``CostElement → CostElement``.
    bottom : CostElement | None
        Starting point (default: ⊥).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    (fixpoint, iterations) : tuple[CostElement, int]
    """
    x = bottom if bottom is not None else cost_bottom()
    for i in range(max_iter):
        x_new = f(x)
        if cost_eq(x, x_new, tol):
            return x_new, i + 1
        x = x_new
    return x, max_iter


def tarski_fixpoint(
    f: Callable[[CostElement], CostElement],
    lattice_elements: Sequence[CostElement],
) -> CostElement:
    r"""Compute the least fixed point via Tarski's theorem.

    For a finite lattice, finds the least ``x`` such that ``f(x) = x``
    by testing all elements.  (Use Kleene iteration for infinite lattices.)

    Parameters
    ----------
    f : callable
        A monotone function.
    lattice_elements : sequence[CostElement]
        All elements of the finite lattice.

    Returns
    -------
    CostElement
        The least fixed point.

    Raises
    ------
    ValueError
        If no fixed point is found.
    """
    # Sort by cost_leq (topological sort of the partial order)
    sorted_elts = sorted(
        lattice_elements,
        key=lambda e: (e.mu, e.sigma_sq, abs(e.kappa), e.lambda_),
    )
    for x in sorted_elts:
        fx = f(x)
        if cost_eq(x, fx):
            return x
    raise ValueError("No fixed point found in the provided lattice elements.")


# ---------------------------------------------------------------------------
# Widening and narrowing
# ---------------------------------------------------------------------------


def widen(a: CostElement, b: CostElement, threshold: float = 10.0) -> CostElement:
    r"""Widening operator ``a ∇ b`` for convergence acceleration.

    For each component, if ``b`` exceeds ``a``, jump to ``∞`` (or a
    threshold-scaled bound); otherwise keep ``a``.

    This ensures termination of ascending chains at the cost of precision.

    Parameters
    ----------
    a : CostElement
        Previous iterate.
    b : CostElement
        Current iterate (``f(a)``).
    threshold : float
        Scale factor for widening jumps.
    """
    mu = a.mu if b.mu <= a.mu + _COST_TOL else max(b.mu, a.mu * threshold)
    sigma_sq = (a.sigma_sq if b.sigma_sq <= a.sigma_sq + _COST_TOL
                else max(b.sigma_sq, a.sigma_sq * threshold))
    kappa = (a.kappa if abs(b.kappa) <= abs(a.kappa) + _COST_TOL
             else math.copysign(max(abs(b.kappa), abs(a.kappa) * threshold), b.kappa))
    lambda_ = (a.lambda_ if b.lambda_ <= a.lambda_ + _COST_TOL
               else min(1.0, max(b.lambda_, a.lambda_ * threshold)))

    return CostElement(mu=mu, sigma_sq=sigma_sq, kappa=kappa, lambda_=lambda_)


def narrow(a: CostElement, b: CostElement) -> CostElement:
    r"""Narrowing operator ``a Δ b`` for precision recovery.

    After widening has found an over-approximation, narrowing refines
    it downward.  Applied as ``a Δ f(a)`` until convergence.

    Parameters
    ----------
    a : CostElement
        Widened (over-approximate) value.
    b : CostElement
        Refined iterate (``f(a)``).
    """
    mu = b.mu if a.mu == float("inf") else min(a.mu, b.mu) if b.mu < a.mu else a.mu
    sigma_sq = (b.sigma_sq if a.sigma_sq == float("inf")
                else min(a.sigma_sq, b.sigma_sq) if b.sigma_sq < a.sigma_sq else a.sigma_sq)
    kappa = b.kappa if abs(b.kappa) < abs(a.kappa) else a.kappa
    lambda_ = b.lambda_ if b.lambda_ < a.lambda_ else a.lambda_

    return CostElement(mu=mu, sigma_sq=sigma_sq, kappa=kappa, lambda_=lambda_)


def widened_fixpoint(
    f: Callable[[CostElement], CostElement],
    bottom: Optional[CostElement] = None,
    max_widen_iter: int = 100,
    max_narrow_iter: int = 50,
    tol: float = 1e-10,
    threshold: float = 10.0,
) -> Tuple[CostElement, int]:
    r"""Compute a fixed point with widening then narrowing.

    Phase 1 (ascending): apply widening until stable.
    Phase 2 (descending): apply narrowing to refine.

    Parameters
    ----------
    f : callable
        Monotone function.
    bottom : CostElement | None
        Start from ⊥.
    max_widen_iter, max_narrow_iter : int
        Iteration limits.
    tol : float
        Convergence tolerance.
    threshold : float
        Widening threshold.

    Returns
    -------
    (fixpoint, total_iterations)
    """
    x = bottom if bottom is not None else cost_bottom()
    total = 0

    # Ascending phase (widening)
    for i in range(max_widen_iter):
        total += 1
        fx = f(x)
        x_new = widen(x, fx, threshold=threshold)
        if cost_eq(x, x_new, tol):
            break
        x = x_new

    # Descending phase (narrowing)
    for i in range(max_narrow_iter):
        total += 1
        fx = f(x)
        x_new = narrow(x, fx)
        if cost_eq(x, x_new, tol):
            break
        x = x_new

    return x, total


# ---------------------------------------------------------------------------
# Galois connection (abstraction / concretisation)
# ---------------------------------------------------------------------------


@dataclass
class GaloisConnection:
    r"""A Galois connection ``(α, γ)`` between two cost lattices.

    A pair of functions ``α : C → A`` (abstraction) and ``γ : A → C``
    (concretisation) such that for all ``c ∈ C, a ∈ A``:

    .. math::

        α(c) ⊑_A a  ⟺  c ⊑_C γ(a)

    Equivalently: ``c ⊑ γ(α(c))`` and ``α(γ(a)) ⊑ a``.

    Parameters
    ----------
    alpha : callable
        Abstraction function ``C → A``.
    gamma : callable
        Concretisation function ``A → C``.
    name : str
        Human-readable name for this connection.
    """

    alpha: Callable[[CostElement], CostElement]
    gamma: Callable[[CostElement], CostElement]
    name: str = ""

    def abstract(self, c: CostElement) -> CostElement:
        """Apply abstraction: ``α(c)``."""
        return self.alpha(c)

    def concretise(self, a: CostElement) -> CostElement:
        """Apply concretisation: ``γ(a)``."""
        return self.gamma(a)

    def verify(self, c: CostElement, tol: float = 1e-10) -> bool:
        r"""Verify the Galois property: ``c ⊑ γ(α(c))`` (soundness)."""
        return cost_leq(c, self.gamma(self.alpha(c)), tol)

    def verify_reductive(self, a: CostElement, tol: float = 1e-10) -> bool:
        r"""Verify: ``α(γ(a)) ⊑ a``."""
        return cost_leq(self.alpha(self.gamma(a)), a, tol)


def variance_abstraction() -> GaloisConnection:
    r"""Galois connection that abstracts away variance.

    * ``α(μ, σ², κ, λ) = (μ + 2√σ², 0, 0, λ)`` — shift mean to cover 2σ.
    * ``γ(μ', 0, 0, λ') = (μ'/3, (μ'/3)², 0, λ')`` — redistribute into variance.
    """
    def alpha(c: CostElement) -> CostElement:
        return CostElement(
            mu=c.mu + 2.0 * c.std_dev(),
            sigma_sq=0.0,
            kappa=0.0,
            lambda_=c.lambda_,
        )

    def gamma(a: CostElement) -> CostElement:
        third = a.mu / 3.0 if a.mu > 0 else 0.0
        return CostElement(
            mu=third,
            sigma_sq=third ** 2,
            kappa=0.0,
            lambda_=a.lambda_,
        )

    return GaloisConnection(alpha=alpha, gamma=gamma, name="variance_abstraction")


def tail_risk_abstraction(threshold: float = 0.1) -> GaloisConnection:
    r"""Galois connection that abstracts tail risk into mean cost.

    * ``α(μ, σ², κ, λ) = (μ · (1 + λ), σ², κ, 0)`` — fold λ into μ.
    * ``γ(μ', σ², κ, 0) = (μ' / (1 + threshold), σ², κ, threshold)``.
    """
    def alpha(c: CostElement) -> CostElement:
        return CostElement(
            mu=c.mu * (1.0 + c.lambda_),
            sigma_sq=c.sigma_sq,
            kappa=c.kappa,
            lambda_=0.0,
        )

    def gamma(a: CostElement) -> CostElement:
        return CostElement(
            mu=a.mu / (1.0 + threshold),
            sigma_sq=a.sigma_sq,
            kappa=a.kappa,
            lambda_=threshold,
        )

    return GaloisConnection(alpha=alpha, gamma=gamma, name="tail_risk_abstraction")


# ---------------------------------------------------------------------------
# Abstract interpretation of costs
# ---------------------------------------------------------------------------


def abstract_interpret_chain(
    costs: Sequence[CostElement],
    abstraction: GaloisConnection,
) -> CostElement:
    r"""Apply abstract interpretation to a chain of costs.

    1. Abstract each cost via ``α``.
    2. Join all abstractions.
    3. Concretise the result via ``γ``.

    This yields a sound over-approximation of the join of the
    original costs.
    """
    if not costs:
        return cost_bottom()
    abstract_costs = [abstraction.abstract(c) for c in costs]
    joined = cost_join_many(abstract_costs)
    return abstraction.concretise(joined)
