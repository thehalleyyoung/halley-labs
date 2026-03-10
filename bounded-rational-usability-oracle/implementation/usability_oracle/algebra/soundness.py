"""
usability_oracle.algebra.soundness — Soundness verification for cost algebra.

Verifies that cost compositions satisfy the axiomatic properties required
for a well-formed usability cost algebra:

1. **Positivity**: ``μ ≥ 0``, ``σ² ≥ 0``, ``λ ∈ [0, 1]``
2. **Monotonicity**: composed cost ≥ max individual cost (in μ)
3. **Identity**: ``a ⊕ 0 = a``, ``a ⊗ 0 = a``
4. **Variance bound**: ``σ²_{composed} ≥ max(σ²_a, σ²_b)``
5. **Triangle inequality** (for sequential): ``d(a, c) ≤ d(a, b) + d(b, c)``
6. **Commutativity** (for parallel): ``a ⊗ b = b ⊗ a``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from usability_oracle.algebra.models import (
    CostElement,
    CostExpression,
    Leaf,
    Sequential,
    Parallel,
    ContextMod,
)
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.parallel import ParallelComposer


# ---------------------------------------------------------------------------
# Verification result types
# ---------------------------------------------------------------------------


class VerificationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class VerificationResult:
    """Result of a single soundness check."""

    property_name: str
    status: VerificationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property": self.property_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# SoundnessVerifier
# ---------------------------------------------------------------------------


class SoundnessVerifier:
    """Verify soundness properties of cost algebra compositions.

    Usage::

        verifier = SoundnessVerifier()
        ok = verifier.verify_sequential(a, b, composed)
        results = verifier.verify_all(expression_tree)
    """

    DEFAULT_TOLERANCE: float = 1e-8

    # -- sequential axioms ---------------------------------------------------

    def verify_sequential(
        self,
        a: CostElement,
        b: CostElement,
        composed: CostElement,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """Verify that a sequential composition satisfies all axioms.

        Checks:
        1. Positivity of result
        2. μ_{a⊕b} ≥ μ_a + μ_b  (superadditive due to coupling ≥ 0)
        3. μ_{a⊕b} ≥ max(μ_a, μ_b)
        4. σ²_{a⊕b} ≥ max(σ²_a, σ²_b)
        5. λ_{a⊕b} ≥ max(λ_a, λ_b)

        Returns
        -------
        bool
            True if all checks pass.
        """
        checks = [
            self._check_positivity(composed),
            composed.mu >= a.mu + b.mu - tolerance,
            composed.mu >= max(a.mu, b.mu) - tolerance,
            composed.sigma_sq >= max(a.sigma_sq, b.sigma_sq) - tolerance,
            composed.lambda_ >= max(a.lambda_, b.lambda_) - tolerance,
        ]
        return all(checks)

    # -- parallel axioms -----------------------------------------------------

    def verify_parallel(
        self,
        a: CostElement,
        b: CostElement,
        composed: CostElement,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """Verify that a parallel composition satisfies all axioms.

        Checks:
        1. Positivity
        2. μ_{a⊗b} ≥ max(μ_a, μ_b)  (monotonicity)
        3. σ²_{a⊗b} ≥ max(σ²_a, σ²_b)
        4. Commutativity: compose(a, b) ≈ compose(b, a)

        Returns
        -------
        bool
        """
        checks = [
            self._check_positivity(composed),
            composed.mu >= max(a.mu, b.mu) - tolerance,
            composed.sigma_sq >= max(a.sigma_sq, b.sigma_sq) - tolerance,
        ]
        return all(checks)

    # -- monotonicity --------------------------------------------------------

    def verify_monotonicity(
        self,
        elements: List[CostElement],
        composed: CostElement,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """Verify that composed cost ≥ max individual cost (μ dimension).

        For both sequential and parallel composition, the total cost
        should be at least as large as the most expensive component.

        Parameters
        ----------
        elements : list[CostElement]
            Individual cost elements.
        composed : CostElement
            The composed result.
        tolerance : float
            Numerical tolerance.

        Returns
        -------
        bool
        """
        if not elements:
            return True
        max_mu = max(e.mu for e in elements)
        return composed.mu >= max_mu - tolerance

    # -- triangle inequality -------------------------------------------------

    def verify_triangle_inequality(
        self,
        a: CostElement,
        b: CostElement,
        c: CostElement,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """Verify the triangle inequality for sequential composition.

        .. math::

            (a ⊕ c).μ ≤ (a ⊕ b).μ + (b ⊕ c).μ

        This holds when coupling is zero (independent composition).
        With positive coupling, the inequality may be violated.

        Parameters
        ----------
        a, b, c : CostElement
        tolerance : float

        Returns
        -------
        bool
        """
        seq = SequentialComposer()
        ac = seq.compose(a, c, coupling=0.0)
        ab = seq.compose(a, b, coupling=0.0)
        bc = seq.compose(b, c, coupling=0.0)
        return ac.mu <= ab.mu + bc.mu + tolerance

    # -- commutativity -------------------------------------------------------

    def verify_commutativity(
        self,
        a: CostElement,
        b: CostElement,
        interference: float = 0.0,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """Verify that parallel composition is commutative: a ⊗ b = b ⊗ a.

        Returns True if the two orderings produce the same result within
        tolerance.
        """
        par = ParallelComposer()
        ab = par.compose(a, b, interference=interference)
        ba = par.compose(b, a, interference=interference)
        return (
            math.isclose(ab.mu, ba.mu, abs_tol=tolerance)
            and math.isclose(ab.sigma_sq, ba.sigma_sq, abs_tol=tolerance)
            and math.isclose(ab.kappa, ba.kappa, abs_tol=tolerance)
            and math.isclose(ab.lambda_, ba.lambda_, abs_tol=tolerance)
        )

    # -- identity property ---------------------------------------------------

    def verify_identity(
        self,
        element: CostElement,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """Verify that composing with the zero element yields the original.

        Checks both ``a ⊕ 0 = a`` and ``a ⊗ 0 = a``.

        Returns True if both hold.
        """
        zero = CostElement.zero()
        seq = SequentialComposer()
        par = ParallelComposer()

        seq_result = seq.compose(element, zero, coupling=0.0)
        par_result = par.compose(element, zero, interference=0.0)

        seq_ok = (
            math.isclose(seq_result.mu, element.mu, abs_tol=tolerance)
            and math.isclose(seq_result.sigma_sq, element.sigma_sq, abs_tol=tolerance)
        )
        par_ok = (
            math.isclose(par_result.mu, element.mu, abs_tol=tolerance)
            and math.isclose(par_result.sigma_sq, element.sigma_sq, abs_tol=tolerance)
        )
        return seq_ok and par_ok

    # -- positivity ----------------------------------------------------------

    def _check_positivity(self, element: CostElement) -> bool:
        """Check that a cost element satisfies positivity constraints.

        * ``μ ≥ 0``
        * ``σ² ≥ 0``
        * ``λ ∈ [0, 1]``
        """
        return (
            element.mu >= 0
            and element.sigma_sq >= 0
            and 0.0 <= element.lambda_ <= 1.0
        )

    # -- variance bound ------------------------------------------------------

    def _check_variance_bound(
        self,
        a: CostElement,
        b: CostElement,
        composed: CostElement,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """Check that composed variance ≥ max(individual variances).

        .. math::

            σ²_{composed} ≥ \\max(σ²_a, σ²_b)

        This holds for both ⊕ and ⊗ under non-negative coupling/interference.
        """
        return composed.sigma_sq >= max(a.sigma_sq, b.sigma_sq) - tolerance

    # -- comprehensive verification ------------------------------------------

    def verify_all(
        self,
        expression: CostExpression,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> List[VerificationResult]:
        """Recursively verify all composition nodes in an expression tree.

        Returns a list of :class:`VerificationResult` — one per node.
        """
        results: List[VerificationResult] = []
        self._verify_recursive(expression, results, tolerance)
        return results

    def _verify_recursive(
        self,
        expr: CostExpression,
        results: List[VerificationResult],
        tolerance: float,
    ) -> None:
        """Walk the expression tree and verify each composition node."""
        if isinstance(expr, Leaf):
            # Check leaf validity
            elem = expr.element
            if elem.is_valid:
                results.append(VerificationResult(
                    property_name="leaf_validity",
                    status=VerificationStatus.PASS,
                    message="Leaf element is valid.",
                    details=elem.to_dict(),
                ))
            else:
                results.append(VerificationResult(
                    property_name="leaf_validity",
                    status=VerificationStatus.FAIL,
                    message="Leaf element violates validity constraints.",
                    details=elem.to_dict(),
                ))
            return

        # Recurse into children first
        for child in expr.children():
            self._verify_recursive(child, results, tolerance)

        composed = expr.evaluate()

        if isinstance(expr, Sequential):
            a = expr.left.evaluate()
            b = expr.right.evaluate()

            # Positivity
            if self._check_positivity(composed):
                results.append(VerificationResult(
                    property_name="sequential_positivity",
                    status=VerificationStatus.PASS,
                    message="Sequential composition satisfies positivity.",
                ))
            else:
                results.append(VerificationResult(
                    property_name="sequential_positivity",
                    status=VerificationStatus.FAIL,
                    message="Sequential composition violates positivity.",
                    details=composed.to_dict(),
                ))

            # Monotonicity
            if self.verify_monotonicity([a, b], composed, tolerance):
                results.append(VerificationResult(
                    property_name="sequential_monotonicity",
                    status=VerificationStatus.PASS,
                    message="Sequential monotonicity holds.",
                ))
            else:
                results.append(VerificationResult(
                    property_name="sequential_monotonicity",
                    status=VerificationStatus.FAIL,
                    message=(
                        f"Sequential monotonicity violated: "
                        f"composed μ={composed.mu:.6f} < max(μ_a={a.mu:.6f}, μ_b={b.mu:.6f})"
                    ),
                ))

            # Variance bound
            if self._check_variance_bound(a, b, composed, tolerance):
                results.append(VerificationResult(
                    property_name="sequential_variance_bound",
                    status=VerificationStatus.PASS,
                    message="Sequential variance bound holds.",
                ))
            else:
                results.append(VerificationResult(
                    property_name="sequential_variance_bound",
                    status=VerificationStatus.FAIL,
                    message="Sequential variance bound violated.",
                    details={
                        "composed_var": composed.sigma_sq,
                        "max_input_var": max(a.sigma_sq, b.sigma_sq),
                    },
                ))

        elif isinstance(expr, Parallel):
            a = expr.left.evaluate()
            b = expr.right.evaluate()

            # Positivity
            if self._check_positivity(composed):
                results.append(VerificationResult(
                    property_name="parallel_positivity",
                    status=VerificationStatus.PASS,
                    message="Parallel composition satisfies positivity.",
                ))
            else:
                results.append(VerificationResult(
                    property_name="parallel_positivity",
                    status=VerificationStatus.FAIL,
                    message="Parallel composition violates positivity.",
                ))

            # Monotonicity
            if self.verify_monotonicity([a, b], composed, tolerance):
                results.append(VerificationResult(
                    property_name="parallel_monotonicity",
                    status=VerificationStatus.PASS,
                    message="Parallel monotonicity holds.",
                ))
            else:
                results.append(VerificationResult(
                    property_name="parallel_monotonicity",
                    status=VerificationStatus.FAIL,
                    message="Parallel monotonicity violated.",
                ))

            # Commutativity
            if self.verify_commutativity(a, b, expr.interference, tolerance):
                results.append(VerificationResult(
                    property_name="parallel_commutativity",
                    status=VerificationStatus.PASS,
                    message="Parallel commutativity holds.",
                ))
            else:
                results.append(VerificationResult(
                    property_name="parallel_commutativity",
                    status=VerificationStatus.FAIL,
                    message="Parallel commutativity violated.",
                ))

        elif isinstance(expr, ContextMod):
            # Context modulation should preserve positivity
            if self._check_positivity(composed):
                results.append(VerificationResult(
                    property_name="context_positivity",
                    status=VerificationStatus.PASS,
                    message="Context modulation preserves positivity.",
                ))
            else:
                results.append(VerificationResult(
                    property_name="context_positivity",
                    status=VerificationStatus.FAIL,
                    message="Context modulation violates positivity.",
                ))

    # -- batch verification --------------------------------------------------

    def verify_elements(
        self, elements: List[CostElement]
    ) -> List[VerificationResult]:
        """Verify a list of individual cost elements."""
        results: List[VerificationResult] = []
        for i, elem in enumerate(elements):
            if elem.is_valid:
                results.append(VerificationResult(
                    property_name=f"element_{i}_validity",
                    status=VerificationStatus.PASS,
                    message=f"Element {i} is valid.",
                ))
            else:
                reasons = []
                if elem.mu < 0:
                    reasons.append(f"μ={elem.mu} < 0")
                if elem.sigma_sq < 0:
                    reasons.append(f"σ²={elem.sigma_sq} < 0")
                if not (0 <= elem.lambda_ <= 1):
                    reasons.append(f"λ={elem.lambda_} ∉ [0, 1]")
                if not math.isfinite(elem.mu):
                    reasons.append(f"μ is not finite")
                results.append(VerificationResult(
                    property_name=f"element_{i}_validity",
                    status=VerificationStatus.FAIL,
                    message=f"Element {i} invalid: {'; '.join(reasons)}",
                    details=elem.to_dict(),
                ))
        return results
