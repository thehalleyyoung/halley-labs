"""
usability_oracle.algebra.sequential — Sequential composition operator ⊕.

Implements the composition of two cognitive cost elements that occur *in
sequence*: the user completes task A, then begins task B.

Mathematical Model
------------------
Given cost elements ``a = (μ_a, σ²_a, κ_a, λ_a)`` and
``b = (μ_b, σ²_b, κ_b, λ_b)`` with coupling parameter ``ρ ∈ [0, 1]``:

.. math::

    μ_{a⊕b}  &= μ_a + μ_b + ρ · \\sqrt{σ²_a · σ²_b}
    σ²_{a⊕b} &= σ²_a + σ²_b + 2ρ · \\sqrt{σ²_a · σ²_b}
    κ_{a⊕b}  &= \\frac{κ_a · (σ²_a)^{3/2} + κ_b · (σ²_b)^{3/2}}{(σ²_{a⊕b})^{3/2}}
    λ_{a⊕b}  &= \\max(λ_a, λ_b) + ρ · \\min(λ_a, λ_b)

Coupling ``ρ = 0`` yields independent summation; ``ρ > 0`` encodes positive
correlation between successive step costs (e.g., a slow first step predicts
a slow second step due to shared cognitive load or UI latency).

Soundness Properties
--------------------
1. **Monotonicity**: ``μ_{a⊕b} ≥ max(μ_a, μ_b)``
2. **Associativity** (when coupling is uniform):
   ``(a ⊕ b) ⊕ c ≈ a ⊕ (b ⊕ c)``
3. **Identity**: ``a ⊕ 0 = a`` where ``0 = (0, 0, 0, 0)``
4. **Variance lower bound**: ``σ²_{a⊕b} ≥ max(σ²_a, σ²_b)``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from usability_oracle.algebra.models import CostElement


# ---------------------------------------------------------------------------
# SequentialComposer
# ---------------------------------------------------------------------------


class SequentialComposer:
    """Compose cost elements sequentially (⊕ operator).

    The coupling parameter ``ρ`` models the statistical dependence between
    successive steps.  When ``ρ = 0``, steps are independent; when ``ρ = 1``,
    there is perfect positive correlation.

    Usage::

        composer = SequentialComposer()
        c = composer.compose(a, b, coupling=0.3)
        chain = composer.compose_chain([a, b, c], couplings=[0.1, 0.2])
    """

    # -- single composition --------------------------------------------------

    def compose(
        self,
        a: CostElement,
        b: CostElement,
        coupling: float = 0.0,
    ) -> CostElement:
        r"""Sequentially compose two cost elements.

        Parameters
        ----------
        a, b : CostElement
            The two cost elements to compose (a first, then b).
        coupling : float
            Correlation coefficient ``ρ ∈ [0, 1]`` between the cost
            distributions of *a* and *b*.

        Returns
        -------
        CostElement
            The composed cost element ``a ⊕ b``.

        Raises
        ------
        ValueError
            If ``coupling`` is outside ``[0, 1]``.

        Mathematical formulation
        ------------------------
        .. math::

            μ_{a⊕b}  = μ_a + μ_b + ρ · \sqrt{σ²_a · σ²_b}
            σ²_{a⊕b} = σ²_a + σ²_b + 2ρ · \sqrt{σ²_a · σ²_b}
            κ_{a⊕b}  = \frac{κ_a · (σ²_a)^{3/2} + κ_b · (σ²_b)^{3/2}}
                              {(σ²_{a⊕b})^{3/2}}
            λ_{a⊕b}  = \max(λ_a, λ_b) + ρ · \min(λ_a, λ_b)

        The coupling term in ``μ`` accounts for *carry-over cost*: when the
        first step is unexpectedly expensive, the user's cognitive state
        (working-memory load, fatigue) increases the expected cost of the
        second step.

        Soundness proof sketch
        ----------------------
        * **Positivity**: all terms are non-negative → ``μ_{a⊕b} ≥ 0`` ✓
        * **Monotonicity**: ``μ_{a⊕b} = μ_a + μ_b + ρ·√(σ²_a·σ²_b)
          ≥ μ_a + μ_b ≥ max(μ_a, μ_b)`` ✓
        * **Identity**: if ``b = 0 = (0,0,0,0)`` then ``μ_{a⊕0} = μ_a``,
          ``σ²_{a⊕0} = σ²_a``, ``κ_{a⊕0} = κ_a``, ``λ_{a⊕0} = λ_a`` ✓
        """
        self._validate_coupling(coupling)

        sqrt_cross = math.sqrt(max(0.0, a.sigma_sq * b.sigma_sq))

        mu = a.mu + b.mu + coupling * sqrt_cross

        sigma_sq = a.sigma_sq + b.sigma_sq + 2.0 * coupling * sqrt_cross

        # Skewness combination
        if sigma_sq > 1e-15:
            kappa = (
                a.kappa * a.sigma_sq ** 1.5
                + b.kappa * b.sigma_sq ** 1.5
            ) / sigma_sq ** 1.5
        else:
            kappa = 0.0

        # Tail risk: worst-case plus coupling-scaled secondary risk
        lambda_ = max(a.lambda_, b.lambda_) + coupling * min(a.lambda_, b.lambda_)
        lambda_ = min(lambda_, 1.0)  # clamp to probability range

        return CostElement(mu=mu, sigma_sq=sigma_sq, kappa=kappa, lambda_=lambda_)

    # -- chain composition ---------------------------------------------------

    def compose_chain(
        self,
        elements: List[CostElement],
        couplings: Optional[List[float]] = None,
    ) -> CostElement:
        """Compose a chain of elements: ``e₁ ⊕ e₂ ⊕ … ⊕ eₙ``.

        Parameters
        ----------
        elements : list[CostElement]
            Ordered sequence of cost elements.
        couplings : list[float] | None
            Coupling coefficients between adjacent pairs.
            Length must be ``len(elements) - 1``.  Defaults to 0.0 for
            all pairs.

        Returns
        -------
        CostElement
            The fully composed cost.
        """
        if not elements:
            return CostElement.zero()

        if len(elements) == 1:
            return elements[0]

        n = len(elements)
        if couplings is None:
            couplings = [0.0] * (n - 1)
        elif len(couplings) != n - 1:
            raise ValueError(
                f"Expected {n - 1} coupling values, got {len(couplings)}."
            )

        result = elements[0]
        for i in range(1, n):
            result = self.compose(result, elements[i], coupling=couplings[i - 1])
        return result

    # -- interval composition ------------------------------------------------

    def compose_interval(
        self,
        a: CostElement,
        b: CostElement,
        coupling_interval: Tuple[float, float],
    ) -> Tuple[CostElement, CostElement]:
        """Compose with an interval-valued coupling parameter.

        Returns conservative (lower, upper) bounds on the composed cost.

        Parameters
        ----------
        a, b : CostElement
            The two cost elements.
        coupling_interval : (float, float)
            Lower and upper bounds on the coupling parameter.

        Returns
        -------
        (lower, upper) : tuple[CostElement, CostElement]
            The lower-bound and upper-bound composed costs.

        Notes
        -----
        Since all partial derivatives of the composition formulas with respect
        to ``ρ`` are non-negative, the lower bound occurs at
        ``ρ = coupling_interval[0]`` and the upper bound at
        ``ρ = coupling_interval[1]``.

        Proof: ``∂μ/∂ρ = √(σ²_a·σ²_b) ≥ 0``,
        ``∂σ²/∂ρ = 2√(σ²_a·σ²_b) ≥ 0``, etc.
        """
        lo, hi = coupling_interval
        if lo > hi:
            lo, hi = hi, lo
        self._validate_coupling(lo)
        self._validate_coupling(hi)

        return (
            self.compose(a, b, coupling=lo),
            self.compose(a, b, coupling=hi),
        )

    # -- n-ary composition with pairwise couplings ---------------------------

    def compose_matrix(
        self,
        elements: List[CostElement],
        coupling_matrix: np.ndarray,
    ) -> CostElement:
        """Compose with a full pairwise coupling matrix.

        Uses sequential folding with effective coupling derived from
        the matrix.

        Parameters
        ----------
        elements : list[CostElement]
            Ordered sequence of cost elements.
        coupling_matrix : np.ndarray
            Symmetric ``n × n`` matrix where entry ``[i, j]`` gives the
            coupling between elements ``i`` and ``j``.

        Returns
        -------
        CostElement
        """
        n = len(elements)
        if coupling_matrix.shape != (n, n):
            raise ValueError(
                f"Coupling matrix shape {coupling_matrix.shape} does not match "
                f"{n} elements."
            )

        if n == 0:
            return CostElement.zero()

        result = elements[0]
        for i in range(1, n):
            # Use the coupling between the accumulated result and element i.
            # Approximate by using the coupling between the last element
            # and the current one.
            rho = float(coupling_matrix[i - 1, i])
            result = self.compose(result, elements[i], coupling=rho)
        return result

    # -- sensitivity analysis ------------------------------------------------

    def sensitivity(
        self,
        a: CostElement,
        b: CostElement,
        coupling: float = 0.0,
        delta: float = 1e-6,
    ) -> dict:
        """Compute partial derivatives of composed cost w.r.t. inputs.

        Returns
        -------
        dict
            Keys: ``"d_mu_a"``, ``"d_mu_b"``, ``"d_sigma_sq_a"``,
            ``"d_sigma_sq_b"``, ``"d_coupling"``, each mapping to
            a :class:`CostElement` of partial derivatives.
        """
        base = self.compose(a, b, coupling=coupling)
        sensitivities = {}

        # ∂/∂μ_a
        perturbed = self.compose(
            CostElement(a.mu + delta, a.sigma_sq, a.kappa, a.lambda_),
            b, coupling=coupling,
        )
        sensitivities["d_mu_a"] = CostElement(
            mu=(perturbed.mu - base.mu) / delta,
            sigma_sq=(perturbed.sigma_sq - base.sigma_sq) / delta,
            kappa=(perturbed.kappa - base.kappa) / delta,
            lambda_=(perturbed.lambda_ - base.lambda_) / delta,
        )

        # ∂/∂μ_b
        perturbed = self.compose(
            a,
            CostElement(b.mu + delta, b.sigma_sq, b.kappa, b.lambda_),
            coupling=coupling,
        )
        sensitivities["d_mu_b"] = CostElement(
            mu=(perturbed.mu - base.mu) / delta,
            sigma_sq=(perturbed.sigma_sq - base.sigma_sq) / delta,
            kappa=(perturbed.kappa - base.kappa) / delta,
            lambda_=(perturbed.lambda_ - base.lambda_) / delta,
        )

        # ∂/∂σ²_a
        perturbed = self.compose(
            CostElement(a.mu, a.sigma_sq + delta, a.kappa, a.lambda_),
            b, coupling=coupling,
        )
        sensitivities["d_sigma_sq_a"] = CostElement(
            mu=(perturbed.mu - base.mu) / delta,
            sigma_sq=(perturbed.sigma_sq - base.sigma_sq) / delta,
            kappa=(perturbed.kappa - base.kappa) / delta,
            lambda_=(perturbed.lambda_ - base.lambda_) / delta,
        )

        # ∂/∂σ²_b
        perturbed = self.compose(
            a,
            CostElement(b.mu, b.sigma_sq + delta, b.kappa, b.lambda_),
            coupling=coupling,
        )
        sensitivities["d_sigma_sq_b"] = CostElement(
            mu=(perturbed.mu - base.mu) / delta,
            sigma_sq=(perturbed.sigma_sq - base.sigma_sq) / delta,
            kappa=(perturbed.kappa - base.kappa) / delta,
            lambda_=(perturbed.lambda_ - base.lambda_) / delta,
        )

        # ∂/∂ρ
        if coupling + delta <= 1.0:
            perturbed = self.compose(a, b, coupling=coupling + delta)
        else:
            perturbed = self.compose(a, b, coupling=coupling - delta)
            delta = -delta
        sensitivities["d_coupling"] = CostElement(
            mu=(perturbed.mu - base.mu) / delta,
            sigma_sq=(perturbed.sigma_sq - base.sigma_sq) / delta,
            kappa=(perturbed.kappa - base.kappa) / delta,
            lambda_=(perturbed.lambda_ - base.lambda_) / delta,
        )

        return sensitivities

    # -- validation ----------------------------------------------------------

    @staticmethod
    def _validate_coupling(coupling: float) -> None:
        if not (0.0 <= coupling <= 1.0):
            raise ValueError(
                f"Coupling must be in [0, 1], got {coupling}."
            )
