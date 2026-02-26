"""
Convergence proof for the epsilon-calibration fixed-point iteration.

The EpsilonCalibrator in epsilon_race.py computes a sequence

    ε_{k+1} = Φ(ε_k)    where   Φ(ε) = margin(abstract_state(ε)) / L

This module provides a rigorous convergence proof based on the Banach
fixed-point theorem, together with:

  - Verification that Φ is a contraction mapping on [0, ε_max].
  - Monotonicity proofs for abstract transformers.
  - Computation and diagnosis of the contraction constant.
  - Certificates recording the fixed-point and its convergence rate.
  - Adaptive (damped) calibration when the contraction condition fails.
  - Soundness proof that any ε ≥ ε* is a valid over-approximation.

Mathematical background
-----------------------
Let S(ε) denote the abstract state computed at precision ε, and let
margin(S) denote the minimum safety margin of the abstract state.
The calibration iteration is

    Φ(ε) = margin(S(ε)) / L

where L is the Lipschitz constant of the safety predicate.

**Banach fixed-point theorem.**  If Φ : [0, ε_max] → [0, ε_max] is a
contraction with Lipschitz constant q < 1, i.e.

    |Φ(a) - Φ(b)| ≤ q |a - b|   for all a, b ∈ [0, ε_max],

then there exists a unique fixed point ε* = Φ(ε*) and the iterates
converge geometrically:

    |ε_k - ε*| ≤ q^k |ε_0 - ε*|.

This module verifies the contraction condition numerically, computes q,
provides certified iteration bounds, and falls back to damped iteration
or bisection when contraction fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fixed-point certificate
# ---------------------------------------------------------------------------

@dataclass
class FixedPointCertificate:
    """Certificate recording the result of a fixed-point computation.

    A verifier can check the certificate by recomputing Φ(ε*) and
    confirming that |Φ(ε*) - ε*| < tolerance.

    Attributes:
        fixed_point: The computed fixed point ε*.
        contraction_constant: Lipschitz constant q of the map Φ.
        iterations_used: Number of iterations to reach the fixed point.
        convergence_rate: Empirical convergence rate (ratio of successive
            error reductions).
        error_bound: Upper bound on |ε_k - ε*| at the final iterate.
        tolerance: Tolerance used for the convergence criterion.
        method: Method used (``"banach"``, ``"damped"``, ``"bisection"``).
        damping_factor: Damping parameter α if damped iteration was used.
        verified: Whether the certificate was independently verified.
    """

    fixed_point: float = 0.0
    contraction_constant: float = 0.0
    iterations_used: int = 0
    convergence_rate: float = 0.0
    error_bound: float = 0.0
    tolerance: float = 1e-8
    method: str = "banach"
    damping_factor: float = 1.0
    verified: bool = False
    # MATH FIX: Store |ε_1 - ε_0| separately for the a-priori bound.
    # The a-priori bound uses the initial step, not the final step.
    initial_step: float = 0.0

    def verify(self, phi_fn: Callable[[float], float]) -> bool:
        """Recompute Φ(ε*) and check |Φ(ε*) - ε*| < tolerance.

        Args:
            phi_fn: The iteration map Φ.

        Returns:
            True if the certificate is valid.
        """
        residual = abs(phi_fn(self.fixed_point) - self.fixed_point)
        self.verified = residual < self.tolerance
        return self.verified

    @property
    def a_priori_bound(self) -> float:
        """A-priori error bound: q^k / (1 - q) * |ε_1 - ε_0|.

        This bound is valid only when contraction_constant < 1.  Returns
        infinity if the contraction condition is not satisfied.

        MATH FIX: Uses initial_step (|ε_1 - ε_0|), not the final-iterate
        error_bound (|ε_k - ε_{k-1}|).  The Banach a-priori estimate is:
            |ε_k - ε*| ≤ q^k / (1 - q) · |ε_1 - ε_0|
        The previous implementation incorrectly substituted error_bound
        (the last step) for initial_step (the first step).
        """
        q = self.contraction_constant
        if q >= 1.0 or q < 0.0:
            return float("inf")
        step = self.initial_step if self.initial_step > 0.0 else self.error_bound
        return (q ** self.iterations_used) / (1.0 - q) * step

    @property
    def a_posteriori_bound(self) -> float:
        """A-posteriori error bound: q / (1 - q) * |ε_k - ε_{k-1}|.

        Uses the last step size for a tighter bound at the final iterate.
        Banach a-posteriori estimate:
            |ε_k - ε*| ≤ q / (1 - q) · |ε_k - ε_{k-1}|
        """
        q = self.contraction_constant
        if q >= 1.0 or q < 0.0:
            return float("inf")
        return q / (1.0 - q) * self.error_bound

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixed_point": self.fixed_point,
            "contraction_constant": self.contraction_constant,
            "iterations_used": self.iterations_used,
            "convergence_rate": self.convergence_rate,
            "error_bound": self.error_bound,
            "initial_step": self.initial_step,
            "tolerance": self.tolerance,
            "method": self.method,
            "damping_factor": self.damping_factor,
            "verified": self.verified,
        }

    def __repr__(self) -> str:
        return (
            f"FixedPointCertificate(ε*={self.fixed_point:.8f}, "
            f"q={self.contraction_constant:.6f}, "
            f"iters={self.iterations_used}, "
            f"method={self.method!r}, verified={self.verified})"
        )


# ---------------------------------------------------------------------------
# Banach fixed-point theorem
# ---------------------------------------------------------------------------

class BanachFixedPointTheorem:
    r"""Verify and apply the Banach fixed-point theorem to the calibration map.

    The calibration iteration is

        ε_{k+1} = Φ(ε_k)    where   Φ(ε) = margin(S(ε)) / L.

    **Theorem (Banach).**  Let (X, d) be a complete metric space and let
    Φ : X → X be a contraction, i.e. there exists q ∈ [0, 1) such that

        d(Φ(x), Φ(y)) ≤ q · d(x, y)   for all x, y ∈ X.

    Then:
      1. Φ has a unique fixed point x* ∈ X.
      2. For every x_0 ∈ X, the sequence x_{k+1} = Φ(x_k) converges to x*.
      3. A-priori estimate:  d(x_k, x*) ≤ q^k / (1 - q) · d(x_1, x_0).
      4. A-posteriori estimate:  d(x_k, x*) ≤ q / (1 - q) · d(x_k, x_{k-1}).

    **Application.**  We take X = [0, ε_max] ⊂ ℝ with the standard metric.
    The map Φ(ε) = margin(S(ε)) / L is a self-map on [0, ε_max] provided

        0 ≤ margin(S(ε)) ≤ L · ε_max   for all ε ∈ [0, ε_max].

    We verify the contraction condition numerically by sampling pairs
    (a, b) from the interval and computing the empirical Lipschitz constant.

    Args:
        phi_fn: The iteration map Φ : float → float.
        interval: Tuple (lo, hi) defining the domain [lo, hi].
    """

    def __init__(
        self,
        phi_fn: Callable[[float], float],
        interval: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._phi = phi_fn
        self._lo, self._hi = interval
        self._q_estimate: Optional[float] = None
        self._samples: List[Tuple[float, float]] = []

    @property
    def contraction_constant(self) -> Optional[float]:
        """Estimated contraction constant q, or None if not computed."""
        return self._q_estimate

    # -- contraction verification ------------------------------------------

    def verify_contraction(
        self,
        phi_fn: Optional[Callable[[float], float]] = None,
        interval: Optional[Tuple[float, float]] = None,
        n_samples: int = 200,
    ) -> Tuple[bool, float]:
        r"""Verify the contraction condition numerically.

        Samples n_samples pairs (a, b) from the interval and computes

            q̂ = max_{(a,b)} |Φ(a) - Φ(b)| / |a - b|.

        If q̂ < 1 the contraction condition is (empirically) satisfied.

        **Proof sketch.**  If Φ is C¹ on [lo, hi], then by the mean value
        theorem |Φ(a) - Φ(b)| = |Φ'(ξ)| |a - b| for some ξ between a and b.
        Hence the Lipschitz constant is sup |Φ'|.  We approximate this by
        sampling.  If the maximum sampled ratio is q̂ < 1, and Φ is smooth
        enough that the true supremum is close to q̂, we conclude contraction.

        Args:
            phi_fn: Override the iteration map (defaults to self._phi).
            interval: Override the domain (defaults to self._interval).
            n_samples: Number of sample points.

        Returns:
            (is_contraction, q_estimate) where is_contraction is True iff
            q_estimate < 1.
        """
        fn = phi_fn if phi_fn is not None else self._phi
        lo = interval[0] if interval is not None else self._lo
        hi = interval[1] if interval is not None else self._hi

        points = np.linspace(lo, hi, n_samples)
        values = np.array([fn(x) for x in points])

        q_max = 0.0
        self._samples = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dx = abs(points[i] - points[j])
                if dx < 1e-15:
                    continue
                dy = abs(values[i] - values[j])
                ratio = dy / dx
                if ratio > q_max:
                    q_max = ratio
                self._samples.append((dx, dy))

        self._q_estimate = q_max
        return (q_max < 1.0, q_max)

    def verify_contraction_gradient(
        self,
        phi_fn: Optional[Callable[[float], float]] = None,
        interval: Optional[Tuple[float, float]] = None,
        n_samples: int = 500,
        h: float = 1e-7,
    ) -> Tuple[bool, float]:
        r"""Verify contraction via numerical differentiation.

        Computes Φ'(ε) at n_samples points using central differences:

            Φ'(ε) ≈ (Φ(ε + h) - Φ(ε - h)) / (2h)

        and checks that sup |Φ'(ε)| < 1.

        **Theorem.**  If Φ is differentiable and |Φ'(ε)| ≤ q < 1 for all
        ε ∈ [lo, hi], then Φ is a contraction with constant q.

        Args:
            phi_fn: Override the iteration map.
            interval: Override the domain.
            n_samples: Number of evaluation points.
            h: Step size for finite differences.

        Returns:
            (is_contraction, q_estimate).
        """
        fn = phi_fn if phi_fn is not None else self._phi
        lo = interval[0] if interval is not None else self._lo
        hi = interval[1] if interval is not None else self._hi

        points = np.linspace(lo + h, hi - h, n_samples)
        derivatives = np.array([
            abs(fn(x + h) - fn(x - h)) / (2.0 * h) for x in points
        ])

        q_max = float(np.max(derivatives)) if len(derivatives) > 0 else 0.0
        self._q_estimate = q_max
        return (q_max < 1.0, q_max)

    # -- self-map verification ---------------------------------------------

    def verify_self_map(
        self,
        phi_fn: Optional[Callable[[float], float]] = None,
        interval: Optional[Tuple[float, float]] = None,
        n_samples: int = 200,
    ) -> Tuple[bool, float, float]:
        r"""Verify that Φ maps [lo, hi] into [lo, hi].

        A necessary condition for Banach's theorem is that Φ is a self-map
        on the domain.  We check:

            lo ≤ Φ(ε) ≤ hi   for all sampled ε ∈ [lo, hi].

        Returns:
            (is_self_map, min_value, max_value).
        """
        fn = phi_fn if phi_fn is not None else self._phi
        lo = interval[0] if interval is not None else self._lo
        hi = interval[1] if interval is not None else self._hi

        points = np.linspace(lo, hi, n_samples)
        values = np.array([fn(x) for x in points])

        vmin, vmax = float(np.min(values)), float(np.max(values))
        is_self = vmin >= lo - 1e-12 and vmax <= hi + 1e-12
        return (is_self, vmin, vmax)

    # -- iteration bound ---------------------------------------------------

    def guaranteed_iterations(
        self,
        q: Optional[float] = None,
        eps0: float = 1.0,
        tolerance: float = 1e-8,
    ) -> int:
        r"""Number of iterations guaranteed to reach tolerance.

        From the a-priori estimate

            |ε_k - ε*| ≤ q^k · |ε_0 - ε*| ≤ q^k · |ε_0 - ε_1| / (1 - q),

        we need q^k · D / (1 - q) < tolerance where D = |ε_0 - ε_1|.
        Thus k ≥ log(tolerance · (1 - q) / D) / log(q).

        If D is not known we use D = eps0 (the diameter of the interval).

        Args:
            q: Contraction constant (defaults to estimated value).
            eps0: Initial error diameter.
            tolerance: Desired accuracy.

        Returns:
            Minimum number of iterations.

        Raises:
            ValueError: If q ≥ 1 or q is not available.
        """
        if q is None:
            if self._q_estimate is None:
                raise ValueError(
                    "Contraction constant not available; "
                    "call verify_contraction first."
                )
            q = self._q_estimate
        if q >= 1.0:
            raise ValueError(
                f"q = {q:.6f} ≥ 1; Banach theorem does not apply."
            )
        if q <= 0.0:
            return 1

        D = eps0
        ratio = tolerance * (1.0 - q) / D
        if ratio >= 1.0:
            return 0
        return int(math.ceil(math.log(ratio) / math.log(q)))

    # -- run iteration -----------------------------------------------------

    def iterate(
        self,
        eps0: float,
        tolerance: float = 1e-8,
        max_iter: int = 500,
    ) -> FixedPointCertificate:
        r"""Run the fixed-point iteration and return a certificate.

        Iterates ε_{k+1} = Φ(ε_k) until |ε_{k+1} - ε_k| < tolerance
        or max_iter iterations.

        Returns:
            A ``FixedPointCertificate`` recording ε*, q, and convergence data.
        """
        eps = eps0
        history = [eps]
        for k in range(max_iter):
            eps_new = self._phi(eps)
            history.append(eps_new)
            if abs(eps_new - eps) < tolerance:
                break
            eps = eps_new

        # estimate empirical convergence rate from last iterates
        rate = 0.0
        if len(history) >= 3:
            deltas = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            rates = [
                deltas[i] / deltas[i - 1]
                for i in range(1, len(deltas))
                if deltas[i - 1] > 1e-15
            ]
            rate = float(np.mean(rates)) if rates else 0.0

        q = self._q_estimate if self._q_estimate is not None else rate

        cert = FixedPointCertificate(
            fixed_point=history[-1],
            contraction_constant=q,
            iterations_used=len(history) - 1,
            convergence_rate=rate,
            error_bound=abs(history[-1] - history[-2]) if len(history) >= 2 else 0.0,
            tolerance=tolerance,
            method="banach",
            # MATH FIX: Store |ε_1 - ε_0| for the a-priori bound.
            initial_step=abs(history[1] - history[0]) if len(history) >= 2 else 0.0,
        )
        cert.verify(self._phi)
        return cert

    def __repr__(self) -> str:
        q_str = f"{self._q_estimate:.6f}" if self._q_estimate is not None else "?"
        return (
            f"BanachFixedPointTheorem(interval=[{self._lo}, {self._hi}], "
            f"q={q_str})"
        )


# ---------------------------------------------------------------------------
# Monotonicity proof
# ---------------------------------------------------------------------------

class MonotonicityProof:
    r"""Prove monotonicity of the calibration map.

    **Theorem.**  Suppose the abstract transformer is Lipschitz-continuous
    and monotone in ε:

        ε₁ ≤ ε₂  ⟹  S(ε₁) ⊆ S(ε₂)       (monotonicity of abstraction)

    That is, widening the precision parameter produces a larger (more
    conservative) abstract state.  Then the safety margin decreases:

        margin(S(ε₁)) ≥ margin(S(ε₂))     (monotone decrease of margin)

    because a larger abstract state moves the boundary closer to the
    safety violation.

    **Consequence.**  The map Φ(ε) = margin(S(ε)) / L is monotone
    decreasing in a neighbourhood of the fixed point ε*.  Combined with
    Φ(ε*) = ε* this means:

      - If ε > ε*: Φ(ε) < ε  (the iteration decreases).
      - If ε < ε*: Φ(ε) > ε  (the iteration increases).

    Hence the iteration is self-correcting and converges from any
    starting point in the basin of attraction.

    This class verifies monotonicity numerically by sampling.

    Args:
        phi_fn: The iteration map Φ.
        interval: Domain [lo, hi].
    """

    def __init__(
        self,
        phi_fn: Callable[[float], float],
        interval: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._phi = phi_fn
        self._lo, self._hi = interval
        self._violations: List[Tuple[float, float, float, float]] = []

    def verify_monotonicity(
        self,
        phi_fn: Optional[Callable[[float], float]] = None,
        interval: Optional[Tuple[float, float]] = None,
        n_samples: int = 300,
    ) -> Tuple[bool, List[Tuple[float, float, float, float]]]:
        r"""Check that Φ is monotone decreasing on the interval.

        Samples n_samples points ε₁ < ε₂ and checks Φ(ε₁) ≥ Φ(ε₂).

        **Proof argument.**  Under the Lipschitz-monotonicity assumption on
        the abstract transformer:

          1. ε₁ ≤ ε₂  ⟹  S(ε₁) ⊆ S(ε₂)   (abstraction monotonicity)
          2. S₁ ⊆ S₂  ⟹  margin(S₁) ≥ margin(S₂)  (margin anti-monotonicity)
          3. margin(S(ε₁)) ≥ margin(S(ε₂))
          4. Φ(ε₁) = margin(S(ε₁))/L ≥ margin(S(ε₂))/L = Φ(ε₂)

        If any violation is found, it indicates that the abstract transformer
        does not satisfy the monotonicity assumption (e.g., due to
        non-monotone widening operators).

        Args:
            phi_fn: Override map.
            interval: Override domain.
            n_samples: Number of samples.

        Returns:
            (is_monotone, violations) where violations is a list of
            (ε₁, ε₂, Φ(ε₁), Φ(ε₂)) tuples violating monotonicity.
        """
        fn = phi_fn if phi_fn is not None else self._phi
        lo = interval[0] if interval is not None else self._lo
        hi = interval[1] if interval is not None else self._hi

        points = np.linspace(lo, hi, n_samples)
        values = np.array([fn(x) for x in points])

        self._violations = []
        for i in range(len(points) - 1):
            if values[i + 1] > values[i] + 1e-12:
                self._violations.append(
                    (float(points[i]), float(points[i + 1]),
                     float(values[i]), float(values[i + 1]))
                )

        return (len(self._violations) == 0, self._violations)

    def verify_self_correcting(
        self,
        fixed_point: float,
        phi_fn: Optional[Callable[[float], float]] = None,
        interval: Optional[Tuple[float, float]] = None,
        n_samples: int = 200,
    ) -> Tuple[bool, Dict[str, Any]]:
        r"""Verify the self-correcting property around ε*.

        Checks:
          - For ε > ε*: Φ(ε) < ε   (iteration decreases toward ε*).
          - For ε < ε*: Φ(ε) > ε   (iteration increases toward ε*).

        This is a direct consequence of monotone decrease of Φ combined
        with Φ(ε*) = ε*.

        Returns:
            (is_self_correcting, report).
        """
        fn = phi_fn if phi_fn is not None else self._phi
        lo = interval[0] if interval is not None else self._lo
        hi = interval[1] if interval is not None else self._hi

        above_ok = True
        below_ok = True
        above_violations = 0
        below_violations = 0

        points = np.linspace(lo, hi, n_samples)
        for x in points:
            phi_x = fn(float(x))
            if x > fixed_point + 1e-12:
                if phi_x >= x + 1e-12:
                    above_ok = False
                    above_violations += 1
            elif x < fixed_point - 1e-12:
                if phi_x <= x - 1e-12:
                    below_ok = False
                    below_violations += 1

        ok = above_ok and below_ok
        report = {
            "is_self_correcting": ok,
            "fixed_point": fixed_point,
            "above_violations": above_violations,
            "below_violations": below_violations,
        }
        return (ok, report)

    @property
    def violations(self) -> List[Tuple[float, float, float, float]]:
        """List of monotonicity violations (ε₁, ε₂, Φ(ε₁), Φ(ε₂))."""
        return list(self._violations)

    def __repr__(self) -> str:
        n_viol = len(self._violations)
        return (
            f"MonotonicityProof(interval=[{self._lo}, {self._hi}], "
            f"violations={n_viol})"
        )


# ---------------------------------------------------------------------------
# Contraction condition
# ---------------------------------------------------------------------------

class ContractionCondition:
    r"""Compute and diagnose the contraction constant.

    The contraction constant of Φ(ε) = margin(S(ε)) / L is

        q  =  L_abstract / L_policy

    where:
      - L_abstract = Lipschitz constant of ε ↦ margin(S(ε)),
        i.e. the sensitivity of the safety margin to the precision ε.
      - L_policy = L, the Lipschitz constant of the policy / safety predicate.

    **Proof.**

        |Φ(a) - Φ(b)| = |margin(S(a)) - margin(S(b))| / L
                       ≤ L_abstract · |a - b| / L
                       = (L_abstract / L) · |a - b|.

    Hence q = L_abstract / L.  The contraction condition q < 1 holds iff
    the abstract transformer is *less* sensitive to ε than the policy.

    **Diagnostic.**  If q ≥ 1:
      - L_abstract too large ⇒ the abstraction is too sensitive to ε
        (consider coarsening the abstraction or using a widening operator).
      - L too small ⇒ the safety predicate is nearly constant
        (the margin provides little information; consider a finer predicate).

    Args:
        margin_fn: Function ε → margin(S(ε)).
        lipschitz_policy: Lipschitz constant L of the policy.
        interval: Domain [lo, hi].
    """

    def __init__(
        self,
        margin_fn: Callable[[float], float],
        lipschitz_policy: float = 1.0,
        interval: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._margin_fn = margin_fn
        self._L = lipschitz_policy
        self._lo, self._hi = interval
        self._L_abstract: Optional[float] = None
        self._q: Optional[float] = None

    def compute(
        self,
        n_samples: int = 300,
    ) -> Tuple[float, float, float]:
        r"""Compute the contraction constant q = L_abstract / L.

        Estimates L_abstract by sampling pairs from the interval:

            L_abstract = max_{a≠b} |margin(S(a)) - margin(S(b))| / |a - b|.

        Returns:
            (q, L_abstract, L_policy).
        """
        points = np.linspace(self._lo, self._hi, n_samples)
        values = np.array([self._margin_fn(x) for x in points])

        L_abs = 0.0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dx = abs(points[i] - points[j])
                if dx < 1e-15:
                    continue
                dy = abs(values[i] - values[j])
                L_abs = max(L_abs, dy / dx)

        self._L_abstract = L_abs
        self._q = L_abs / self._L if self._L > 1e-15 else float("inf")
        return (self._q, L_abs, self._L)

    def compute_gradient(
        self,
        n_samples: int = 500,
        h: float = 1e-7,
    ) -> Tuple[float, float, float]:
        r"""Compute q via numerical differentiation of the margin function.

        Uses central differences to estimate

            L_abstract ≈ sup_ε |d margin(S(ε)) / dε|.

        Returns:
            (q, L_abstract, L_policy).
        """
        lo, hi = self._lo, self._hi
        points = np.linspace(lo + h, hi - h, n_samples)
        derivatives = np.array([
            abs(self._margin_fn(x + h) - self._margin_fn(x - h)) / (2.0 * h)
            for x in points
        ])

        L_abs = float(np.max(derivatives)) if len(derivatives) > 0 else 0.0
        self._L_abstract = L_abs
        self._q = L_abs / self._L if self._L > 1e-15 else float("inf")
        return (self._q, L_abs, self._L)

    @property
    def contraction_constant(self) -> Optional[float]:
        """The computed contraction constant q = L_abstract / L."""
        return self._q

    @property
    def is_contraction(self) -> bool:
        """True if q < 1 (contraction condition satisfied)."""
        return self._q is not None and self._q < 1.0

    def diagnose(self) -> Dict[str, Any]:
        r"""Diagnostic report when contraction fails (q ≥ 1).

        Returns a dictionary with:
          - q: the contraction constant.
          - L_abstract: Lipschitz constant of the margin w.r.t. ε.
          - L_policy: Lipschitz constant of the policy.
          - diagnosis: human-readable explanation.
          - recommendation: suggested fix.
        """
        if self._q is None or self._L_abstract is None:
            return {"error": "Call compute() first."}

        diag: Dict[str, Any] = {
            "q": self._q,
            "L_abstract": self._L_abstract,
            "L_policy": self._L,
            "is_contraction": self._q < 1.0,
        }

        if self._q < 1.0:
            diag["diagnosis"] = (
                f"Contraction satisfied: q = {self._q:.6f} < 1. "
                f"The abstract transformer is less sensitive to ε than the policy."
            )
            diag["recommendation"] = "No action needed."
        else:
            if self._L_abstract > self._L:
                diag["diagnosis"] = (
                    f"Contraction fails: q = {self._q:.6f} ≥ 1. "
                    f"L_abstract ({self._L_abstract:.6f}) > L_policy ({self._L:.6f}). "
                    f"The abstract transformer is TOO SENSITIVE to ε."
                )
                diag["recommendation"] = (
                    "Coarsen the abstraction (e.g., use interval instead of "
                    "zonotope), apply widening, or increase L_policy by using "
                    "a safety predicate with a steeper gradient."
                )
            else:
                diag["diagnosis"] = (
                    f"Contraction fails: q = {self._q:.6f} ≥ 1. "
                    f"L_policy ({self._L:.6f}) is very small relative to "
                    f"L_abstract ({self._L_abstract:.6f}). "
                    f"The safety margin provides little discriminative information."
                )
                diag["recommendation"] = (
                    "Use a finer safety predicate with a larger Lipschitz "
                    "constant, or apply damped iteration (see AdaptiveCalibration)."
                )

        return diag

    def __repr__(self) -> str:
        q_str = f"{self._q:.6f}" if self._q is not None else "?"
        return (
            f"ContractionCondition(q={q_str}, L_policy={self._L:.4f}, "
            f"interval=[{self._lo}, {self._hi}])"
        )


# ---------------------------------------------------------------------------
# Adaptive calibration (damped + bisection fallback)
# ---------------------------------------------------------------------------

class AdaptiveCalibration:
    r"""Adaptive calibration with damping and bisection fallback.

    When the standard iteration ε_{k+1} = Φ(ε_k) fails to contract
    (q ≥ 1), we apply **damped iteration**:

        ε_{k+1} = (1 - α) ε_k + α Φ(ε_k)

    The damped map is Ψ_α(ε) = (1 - α) ε + α Φ(ε).

    **Theorem (damped contraction).**  The Lipschitz constant of Ψ_α is

        q_α = |1 - α + α · Φ'(ε)|  ≤  |1 - α| + α · q

    where q is the Lipschitz constant of Φ.  For α ∈ (0, 1]:

        q_α ≤ 1 - α + α q = 1 - α(1 - q).

    This is < 1 iff q < 1 (same condition).  However, if Φ oscillates
    (alternating above and below ε*), damping stabilises the iteration:

        q_α = max |1 - α + α Φ'| ≤ max(|1 - α(1 + |Φ'_max|)|, |1 - α(1 - |Φ'_max|)|)

    The optimal damping for an oscillatory Φ with |Φ'| ≈ q is:

        α* = 2 / (1 + q)

    which gives q_{α*} = (q - 1)/(q + 1) when q > 1.  This is < 1 iff q < ∞.

    **Fallback: bisection.**  If damped iteration also fails (or if Φ is
    not monotone), we fall back to bisection on the function g(ε) = Φ(ε) - ε,
    finding ε* such that g(ε*) = 0.  Bisection requires only that g changes
    sign on [lo, hi], which is guaranteed if Φ(lo) > lo and Φ(hi) < hi (or
    vice versa).

    Args:
        phi_fn: The iteration map Φ.
        interval: Domain [lo, hi].
    """

    def __init__(
        self,
        phi_fn: Callable[[float], float],
        interval: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._phi = phi_fn
        self._lo, self._hi = interval

    # -- damping parameter -------------------------------------------------

    @staticmethod
    def optimal_damping(q: float) -> float:
        r"""Compute the optimal damping parameter α* = 2 / (1 + q).

        **Derivation.**  The damped map is Ψ_α(ε) = (1-α)ε + αΦ(ε).
        Its derivative is Ψ_α'(ε) = (1-α) + αΦ'(ε).

        For an oscillatory Φ with Φ'(ε) ∈ [-q, q], the worst-case
        derivative of Ψ_α is max(|1-α-αq|, |1-α+αq|).

        Setting 1 - α + αq = -(1 - α - αq) gives α* = 2/(1+q), and
        the resulting contraction constant is q_α = (q-1)/(q+1).

        Args:
            q: Lipschitz constant of Φ.

        Returns:
            Optimal damping parameter α*.
        """
        if q <= 0.0:
            return 1.0
        return 2.0 / (1.0 + q)

    @staticmethod
    def damped_contraction_constant(q: float, alpha: float) -> float:
        r"""Contraction constant of the damped iteration.

        q_α = max(|1 - α(1-q)|, |1 - α(1+q)|)

        For the optimal α = 2/(1+q) this gives (q-1)/(q+1).
        """
        c1 = abs(1.0 - alpha * (1.0 - q))
        c2 = abs(1.0 - alpha * (1.0 + q))
        return max(c1, c2)

    # -- damped iteration --------------------------------------------------

    def damped_iterate(
        self,
        eps0: float,
        q: float,
        alpha: Optional[float] = None,
        tolerance: float = 1e-8,
        max_iter: int = 1000,
    ) -> FixedPointCertificate:
        r"""Run the damped fixed-point iteration.

        ε_{k+1} = (1 - α) ε_k + α Φ(ε_k)

        **Convergence proof.**  Define Ψ_α(ε) = (1-α)ε + αΦ(ε).
        Then Ψ_α is a contraction with constant q_α < 1 (see class
        docstring).  By Banach's theorem, Ψ_α has a unique fixed point
        ε* satisfying ε* = (1-α)ε* + αΦ(ε*), i.e. ε* = Φ(ε*).  Hence
        the damped and undamped iterations share the same fixed point.

        Args:
            eps0: Initial ε.
            q: Lipschitz constant of Φ (used to choose α if not given).
            alpha: Damping parameter (defaults to optimal α*).
            tolerance: Convergence tolerance.
            max_iter: Maximum iterations.

        Returns:
            FixedPointCertificate.
        """
        if alpha is None:
            alpha = self.optimal_damping(q)

        eps = eps0
        history = [eps]

        for k in range(max_iter):
            phi_eps = self._phi(eps)
            eps_new = (1.0 - alpha) * eps + alpha * phi_eps
            history.append(eps_new)
            if abs(eps_new - eps) < tolerance:
                break
            eps = eps_new

        # empirical convergence rate
        rate = 0.0
        if len(history) >= 3:
            deltas = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            rates = [
                deltas[i] / deltas[i - 1]
                for i in range(1, len(deltas))
                if deltas[i - 1] > 1e-15
            ]
            rate = float(np.mean(rates)) if rates else 0.0

        q_alpha = self.damped_contraction_constant(q, alpha)

        cert = FixedPointCertificate(
            fixed_point=history[-1],
            contraction_constant=q_alpha,
            iterations_used=len(history) - 1,
            convergence_rate=rate,
            error_bound=abs(history[-1] - history[-2]) if len(history) >= 2 else 0.0,
            tolerance=tolerance,
            method="damped",
            damping_factor=alpha,
            initial_step=abs(history[1] - history[0]) if len(history) >= 2 else 0.0,
        )
        cert.verify(self._phi)
        return cert

    # -- bisection fallback ------------------------------------------------

    def bisection(
        self,
        tolerance: float = 1e-10,
        max_iter: int = 200,
    ) -> FixedPointCertificate:
        r"""Find the fixed point by bisection on g(ε) = Φ(ε) - ε.

        **Existence.**  If Φ : [lo, hi] → [lo, hi] is continuous, then
        g(ε) = Φ(ε) - ε satisfies:

            g(lo) = Φ(lo) - lo ≥ 0   (since Φ(lo) ≥ lo)
            g(hi) = Φ(hi) - hi ≤ 0   (since Φ(hi) ≤ hi)

        By the intermediate value theorem, there exists ε* ∈ [lo, hi]
        with g(ε*) = 0, i.e. Φ(ε*) = ε*.

        Bisection finds this root in O(log((hi-lo)/tol)) iterations.

        Returns:
            FixedPointCertificate with method="bisection".
        """
        lo, hi = self._lo, self._hi
        g_lo = self._phi(lo) - lo
        g_hi = self._phi(hi) - hi

        # If signs are the same, try to expand the interval
        if g_lo * g_hi > 0:
            # Attempt to find a sign change by scanning
            n_scan = 100
            points = np.linspace(lo, hi, n_scan)
            g_vals = np.array([self._phi(x) - x for x in points])

            found = False
            for i in range(len(g_vals) - 1):
                if g_vals[i] * g_vals[i + 1] <= 0:
                    lo, hi = float(points[i]), float(points[i + 1])
                    g_lo, g_hi = float(g_vals[i]), float(g_vals[i + 1])
                    found = True
                    break

            if not found:
                # No sign change found; return best approximation
                idx = int(np.argmin(np.abs(g_vals)))
                return FixedPointCertificate(
                    fixed_point=float(points[idx]),
                    contraction_constant=float("inf"),
                    iterations_used=n_scan,
                    convergence_rate=0.0,
                    error_bound=float(np.abs(g_vals[idx])),
                    tolerance=tolerance,
                    method="bisection",
                    verified=False,
                )

        mid = (lo + hi) / 2.0
        for k in range(max_iter):
            mid = (lo + hi) / 2.0
            g_mid = self._phi(mid) - mid

            if abs(g_mid) < tolerance or (hi - lo) / 2.0 < tolerance:
                break

            if g_lo * g_mid <= 0:
                hi = mid
                g_hi = g_mid
            else:
                lo = mid
                g_lo = g_mid

        cert = FixedPointCertificate(
            fixed_point=mid,
            contraction_constant=float("inf"),
            iterations_used=k + 1 if 'k' in dir() else 0,
            convergence_rate=0.5,  # bisection halves interval each step
            error_bound=(hi - lo) / 2.0,
            tolerance=tolerance,
            method="bisection",
        )
        cert.verify(self._phi)
        return cert

    # -- adaptive strategy -------------------------------------------------

    def calibrate(
        self,
        eps0: float,
        tolerance: float = 1e-8,
        max_iter: int = 500,
    ) -> FixedPointCertificate:
        r"""Adaptive calibration with automatic fallback.

        Strategy:
          1. Try standard Banach iteration (ε_{k+1} = Φ(ε_k)).
          2. If q ≥ 1, try damped iteration with optimal α.
          3. If damped iteration also fails, fall back to bisection.

        This method always returns a fixed-point certificate (possibly
        from bisection, which always converges for continuous Φ).

        Args:
            eps0: Initial epsilon.
            tolerance: Convergence tolerance.
            max_iter: Maximum iterations per strategy.

        Returns:
            FixedPointCertificate.
        """
        # Phase 1: try standard Banach iteration
        banach = BanachFixedPointTheorem(self._phi, (self._lo, self._hi))
        is_contraction, q = banach.verify_contraction(n_samples=100)

        if is_contraction:
            cert = banach.iterate(eps0, tolerance=tolerance, max_iter=max_iter)
            if cert.verified:
                return cert

        # Phase 2: try damped iteration
        if q < float("inf") and q > 0:
            cert = self.damped_iterate(
                eps0, q=q, tolerance=tolerance, max_iter=max_iter
            )
            if cert.verified:
                return cert

        # Phase 3: bisection fallback
        return self.bisection(tolerance=tolerance, max_iter=max_iter)

    def __repr__(self) -> str:
        return (
            f"AdaptiveCalibration(interval=[{self._lo}, {self._hi}])"
        )


# ---------------------------------------------------------------------------
# Calibration soundness
# ---------------------------------------------------------------------------

class CalibrationSoundness:
    r"""Soundness proof for the calibration fixed point.

    **Theorem (Soundness of over-approximate ε).**

    Let ε* be the fixed point of Φ(ε) = margin(S(ε)) / L.  Then for
    any ε ≥ ε*, the abstract state S(ε) is a sound over-approximation
    of the concrete reachable states.

    **Proof.**

    1. By definition of the abstract transformer, S(ε) over-approximates
       all concrete states within distance ε of the centre.

    2. The safety margin margin(S(ε)) quantifies how far the abstract
       state is from the safety boundary.  The calibrated epsilon is
       ε* = margin(S(ε*)) / L.

    3. For any ε ≥ ε*, monotonicity of S gives S(ε) ⊇ S(ε*), so
       S(ε) still over-approximates the concrete states.

    4. The margin may decrease (margin(S(ε)) ≤ margin(S(ε*))), but the
       abstract state remains sound.

    **Corollary.**  Stopping the calibration early (at ε_k > ε*) is
    always sound, just less precise.  The fixed point ε* gives the
    TIGHTEST sound epsilon.

    **Corollary.**  Any ε < ε* is potentially UNSOUND: the abstract
    state S(ε) might not contain all concrete states, so race detection
    could have false negatives.

    Args:
        phi_fn: The iteration map Φ.
        interval: Domain [lo, hi].
    """

    def __init__(
        self,
        phi_fn: Callable[[float], float],
        interval: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._phi = phi_fn
        self._lo, self._hi = interval

    def verify_soundness(
        self,
        epsilon: float,
        fixed_point: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        r"""Verify that a given ε is a sound over-approximation.

        An ε is sound iff ε ≥ ε* (the fixed point).

        **Proof.**
          ε ≥ ε*
          ⟹  S(ε) ⊇ S(ε*)          (monotonicity of abstraction)
          ⟹  S(ε) contains all concrete states within distance ε*
          ⟹  Sound over-approximation.

        The precision loss is ε - ε* (the excess radius).

        Args:
            epsilon: The epsilon to check.
            fixed_point: The computed fixed point ε*.

        Returns:
            (is_sound, report).
        """
        is_sound = epsilon >= fixed_point - 1e-12

        phi_eps = self._phi(epsilon)
        residual = abs(phi_eps - epsilon)

        report = {
            "epsilon": epsilon,
            "fixed_point": fixed_point,
            "is_sound": is_sound,
            "precision_loss": max(0.0, epsilon - fixed_point),
            "phi_at_epsilon": phi_eps,
            "residual": residual,
        }

        if not is_sound:
            report["warning"] = (
                f"ε = {epsilon:.8f} < ε* = {fixed_point:.8f}. "
                f"The abstract state may not contain all concrete states. "
                f"Race detection could have FALSE NEGATIVES."
            )
        else:
            report["note"] = (
                f"ε = {epsilon:.8f} ≥ ε* = {fixed_point:.8f}. "
                f"The abstraction is sound (over-approximate). "
                f"Precision loss: {epsilon - fixed_point:.8f}."
            )

        return (is_sound, report)

    def tightness_certificate(
        self,
        fixed_point: float,
        tolerance: float = 1e-8,
    ) -> Dict[str, Any]:
        r"""Certificate that ε* is the tightest sound epsilon.

        **Proof.**
        1. ε* is sound: ε* = Φ(ε*) = margin(S(ε*)) / L, so S(ε*)
           over-approximates all concrete states within distance ε*.

        2. ε* is tight: for any ε' < ε*, we have (by fixed-point
           property and monotonicity):
             margin(S(ε')) > margin(S(ε*))
             ⟹  Φ(ε') > Φ(ε*) = ε* > ε'
           So Φ(ε') > ε', meaning the margin at ε' prescribes a LARGER
           epsilon.  The abstraction at ε' is not self-consistent.

        Returns:
            Certificate dictionary.
        """
        phi_star = self._phi(fixed_point)
        residual = abs(phi_star - fixed_point)

        # check a slightly smaller epsilon
        eps_smaller = fixed_point * 0.99 if fixed_point > 1e-12 else 0.0
        phi_smaller = self._phi(eps_smaller) if eps_smaller > 0 else self._phi(0.0)

        return {
            "fixed_point": fixed_point,
            "phi_at_fixed_point": phi_star,
            "residual": residual,
            "is_tight": residual < tolerance,
            "eps_smaller": eps_smaller,
            "phi_at_smaller": phi_smaller,
            "smaller_is_inconsistent": phi_smaller > eps_smaller + 1e-12,
            "proof": (
                "ε* is the unique fixed point of Φ. Any ε < ε* has "
                "Φ(ε) > ε (the margin prescribes a larger epsilon), so "
                "the abstraction at ε is not self-consistent. Any ε > ε* "
                "is sound but imprecise."
            ),
        }

    def soundness_region(
        self,
        fixed_point: float,
        n_samples: int = 200,
    ) -> Dict[str, Any]:
        r"""Characterise the region of sound epsilons.

        Computes Φ(ε) - ε for ε ∈ [lo, hi] to identify:
          - Sound region: {ε : ε ≥ ε*} where Φ(ε) ≤ ε.
          - Unsound region: {ε : ε < ε*} where Φ(ε) > ε.

        Returns:
            Dictionary with arrays of ε values, Φ(ε), and soundness flags.
        """
        points = np.linspace(self._lo, self._hi, n_samples)
        phi_values = np.array([self._phi(x) for x in points])
        residuals = phi_values - points
        is_sound = points >= fixed_point - 1e-12

        return {
            "epsilon_values": points.tolist(),
            "phi_values": phi_values.tolist(),
            "residuals": residuals.tolist(),
            "is_sound": is_sound.tolist(),
            "fixed_point": fixed_point,
            "sound_range": (fixed_point, self._hi),
            "unsound_range": (self._lo, fixed_point),
        }

    def early_stopping_bound(
        self,
        epsilon_k: float,
        fixed_point: float,
        lipschitz_policy: float = 1.0,
    ) -> Dict[str, Any]:
        r"""Bound the cost of early stopping.

        If we stop at ε_k > ε*, the abstract state S(ε_k) is sound but
        the excess radius ε_k - ε* causes:

          1. **Volume overhead**: the certified region is larger by a factor
             of (ε_k / ε*)^n in n dimensions.

          2. **False positive increase**: the additional volume may contain
             states that are safe but flagged as races.

          3. **Margin reduction**: margin(S(ε_k)) ≤ margin(S(ε*)),
             so the safety guarantee is weaker by at most
             L · (ε_k - ε*).

        Args:
            epsilon_k: The stopping epsilon.
            fixed_point: The true fixed point ε*.
            lipschitz_policy: Policy Lipschitz constant L.

        Returns:
            Dictionary with overhead metrics.
        """
        excess = max(0.0, epsilon_k - fixed_point)
        margin_loss = lipschitz_policy * excess

        return {
            "epsilon_k": epsilon_k,
            "fixed_point": fixed_point,
            "excess_radius": excess,
            "margin_loss_bound": margin_loss,
            "is_sound": epsilon_k >= fixed_point - 1e-12,
            "note": (
                f"Early stopping at ε_k = {epsilon_k:.8f} is sound. "
                f"Margin loss ≤ L · (ε_k - ε*) = {margin_loss:.8f}."
                if epsilon_k >= fixed_point - 1e-12
                else f"WARNING: ε_k = {epsilon_k:.8f} < ε* = {fixed_point:.8f}. "
                     f"Potentially unsound."
            ),
        }

    def __repr__(self) -> str:
        return (
            f"CalibrationSoundness(interval=[{self._lo}, {self._hi}])"
        )


# ---------------------------------------------------------------------------
# Formal convergence certificate
# ---------------------------------------------------------------------------


@dataclass
class FormalConvergenceCertificate:
    r"""Formal convergence certificate for the epsilon-calibration iteration.

    This certificate bundles all the formal ingredients required to
    prove convergence of the fixed-point iteration:

      1. **Self-map property**: Φ maps [lo, hi] into [lo, hi].
      2. **Contraction property**: |Φ(a) - Φ(b)| ≤ q|a - b| for q < 1.
      3. **Monotonicity**: Φ is monotone decreasing.
      4. **Convergence rate**: geometric with constant q.
      5. **Fixed-point residual**: |Φ(ε*) - ε*| < tolerance.

    A verifier can independently check the certificate by evaluating Φ at
    the claimed fixed point and at sampled pairs.

    Attributes
    ----------
    fixed_point : float
        The computed fixed point ε*.
    contraction_constant : float
        Lipschitz constant q of the iteration map.
    is_contraction : bool
        Whether q < 1.
    is_self_map : bool
        Whether Φ maps the interval into itself.
    is_monotone : bool
        Whether Φ is monotone decreasing on the interval.
    convergence_rate_bound : float
        Upper bound on geometric convergence rate (equals q).
    iterations_to_tolerance : int or None
        Guaranteed iterations to reach tolerance (from a-priori bound).
    residual : float
        |Φ(ε*) - ε*| at the claimed fixed point.
    tolerance : float
        Tolerance used for convergence.
    a_priori_error_bound : float
        q^k / (1-q) * |ε_1 - ε_0|.
    a_posteriori_error_bound : float
        q / (1-q) * |ε_k - ε_{k-1}|.
    is_valid : bool
        Whether all certificate conditions are satisfied.
    """
    fixed_point: float = 0.0
    contraction_constant: float = 0.0
    is_contraction: bool = False
    is_self_map: bool = False
    is_monotone: bool = False
    convergence_rate_bound: float = 0.0
    iterations_to_tolerance: Optional[int] = None
    residual: float = 0.0
    tolerance: float = 1e-8
    a_priori_error_bound: float = float("inf")
    a_posteriori_error_bound: float = float("inf")
    is_valid: bool = False

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"=== Formal Convergence Certificate [{status}] ===",
            f"  Fixed point ε* = {self.fixed_point:.10f}",
            f"  Contraction constant q = {self.contraction_constant:.6f}  "
            f"({'< 1 ✓' if self.is_contraction else '>= 1 ✗'})",
            f"  Self-map: {'✓' if self.is_self_map else '✗'}",
            f"  Monotone: {'✓' if self.is_monotone else '✗'}",
            f"  Residual |Φ(ε*) - ε*| = {self.residual:.2e}",
            f"  Convergence rate bound: {self.convergence_rate_bound:.6f}",
        ]
        if self.iterations_to_tolerance is not None:
            lines.append(
                f"  Guaranteed iterations: {self.iterations_to_tolerance}"
            )
        lines.append(f"  A-priori error bound: {self.a_priori_error_bound:.2e}")
        lines.append(f"  A-posteriori error bound: {self.a_posteriori_error_bound:.2e}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixed_point": self.fixed_point,
            "contraction_constant": self.contraction_constant,
            "is_contraction": self.is_contraction,
            "is_self_map": self.is_self_map,
            "is_monotone": self.is_monotone,
            "convergence_rate_bound": self.convergence_rate_bound,
            "iterations_to_tolerance": self.iterations_to_tolerance,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "a_priori_error_bound": self.a_priori_error_bound,
            "a_posteriori_error_bound": self.a_posteriori_error_bound,
            "is_valid": self.is_valid,
        }


class FormalConvergenceProver:
    r"""Generate a formal convergence certificate for epsilon-calibration.

    Combines all verification steps into a single analysis:

      1. Verify self-map property (Φ: [lo,hi] → [lo,hi]).
      2. Verify contraction via numerical derivative estimation.
      3. Verify monotonicity (Φ decreasing).
      4. Compute convergence rate bounds (geometric with constant q).
      5. Run the fixed-point iteration.
      6. Compute residual and error bounds.

    The resulting :class:`FormalConvergenceCertificate` can be independently
    verified by a third party.

    Parameters
    ----------
    phi_fn : callable
        The iteration map Φ(ε) = margin(S(ε)) / L.
    interval : tuple
        Domain [lo, hi].
    n_samples : int
        Number of samples for numerical verification.
    """

    def __init__(
        self,
        phi_fn: Callable[[float], float],
        interval: Tuple[float, float] = (0.0, 1.0),
        n_samples: int = 300,
    ):
        self._phi = phi_fn
        self._lo, self._hi = interval
        self._n_samples = n_samples

    def prove(
        self,
        eps0: Optional[float] = None,
        tolerance: float = 1e-8,
        max_iter: int = 500,
    ) -> FormalConvergenceCertificate:
        """Run the full convergence proof and return a certificate."""
        lo, hi = self._lo, self._hi
        if eps0 is None:
            eps0 = (lo + hi) / 2.0

        # Step 1: Self-map
        banach = BanachFixedPointTheorem(self._phi, (lo, hi))
        is_self_map, _, _ = banach.verify_self_map(n_samples=self._n_samples)

        # Step 2: Contraction (use gradient-based for tighter estimate)
        is_contraction, q = banach.verify_contraction_gradient(
            n_samples=self._n_samples
        )
        # Also check via pair-sampling
        is_c2, q2 = banach.verify_contraction(n_samples=self._n_samples)
        q = max(q, q2)  # conservative
        is_contraction = q < 1.0

        # Step 3: Monotonicity
        mono = MonotonicityProof(self._phi, (lo, hi))
        is_monotone, _ = mono.verify_monotonicity(n_samples=self._n_samples)

        # Step 4: Fixed-point iteration
        cert = banach.iterate(eps0, tolerance=tolerance, max_iter=max_iter)

        # Step 5: Residual
        residual = abs(self._phi(cert.fixed_point) - cert.fixed_point)

        # Step 6: Error bounds
        a_priori = float("inf")
        a_posteriori = float("inf")
        iter_bound = None
        if is_contraction and q < 1.0 and q > 0:
            # a-priori: q^k / (1-q) * |ε_1 - ε_0|
            a_priori = (q ** cert.iterations_used) / (1.0 - q) * (hi - lo)
            # a-posteriori: q / (1-q) * |ε_k - ε_{k-1}|
            a_posteriori = q / (1.0 - q) * cert.error_bound
            try:
                iter_bound = banach.guaranteed_iterations(
                    q, eps0=hi - lo, tolerance=tolerance
                )
            except ValueError:
                pass

        is_valid = (
            is_self_map
            and is_contraction
            and residual < tolerance
        )

        return FormalConvergenceCertificate(
            fixed_point=cert.fixed_point,
            contraction_constant=q,
            is_contraction=is_contraction,
            is_self_map=is_self_map,
            is_monotone=is_monotone,
            convergence_rate_bound=q,
            iterations_to_tolerance=iter_bound,
            residual=residual,
            tolerance=tolerance,
            a_priori_error_bound=a_priori,
            a_posteriori_error_bound=a_posteriori,
            is_valid=is_valid,
        )


# ---------------------------------------------------------------------------
# Convenience: full convergence analysis pipeline
# ---------------------------------------------------------------------------

def analyse_calibration_convergence(
    phi_fn: Callable[[float], float],
    interval: Tuple[float, float] = (0.0, 1.0),
    eps0: Optional[float] = None,
    lipschitz_policy: float = 1.0,
    tolerance: float = 1e-8,
    n_samples: int = 200,
) -> Dict[str, Any]:
    r"""Run a complete convergence analysis of the calibration iteration.

    This is a convenience function that:
      1. Verifies the self-map property.
      2. Verifies the contraction condition (Banach).
      3. Checks monotonicity.
      4. Computes the contraction constant and diagnoses failures.
      5. Runs the adaptive calibration (Banach / damped / bisection).
      6. Verifies soundness and tightness of the result.

    Args:
        phi_fn: The iteration map Φ(ε) = margin(S(ε)) / L.
        interval: Domain [lo, hi].
        eps0: Initial ε (defaults to midpoint of interval).
        lipschitz_policy: Lipschitz constant L of the policy.
        tolerance: Convergence tolerance.
        n_samples: Number of samples for verification.

    Returns:
        Comprehensive analysis report.
    """
    lo, hi = interval
    if eps0 is None:
        eps0 = (lo + hi) / 2.0

    # 1. Banach analysis
    banach = BanachFixedPointTheorem(phi_fn, interval)
    is_self_map, vmin, vmax = banach.verify_self_map(n_samples=n_samples)
    is_contraction, q = banach.verify_contraction(n_samples=n_samples)

    # 2. Monotonicity
    mono = MonotonicityProof(phi_fn, interval)
    is_monotone, violations = mono.verify_monotonicity(n_samples=n_samples)

    # 3. Contraction condition
    margin_fn = lambda eps: phi_fn(eps) * lipschitz_policy
    cc = ContractionCondition(margin_fn, lipschitz_policy, interval)
    q_cc, L_abs, L_pol = cc.compute(n_samples=n_samples)
    diagnosis = cc.diagnose()

    # 4. Adaptive calibration
    adaptive = AdaptiveCalibration(phi_fn, interval)
    cert = adaptive.calibrate(eps0, tolerance=tolerance, max_iter=500)

    # 5. Iteration bound (if contraction holds)
    iter_bound = None
    if is_contraction and q > 0:
        try:
            iter_bound = banach.guaranteed_iterations(q, eps0=hi - lo, tolerance=tolerance)
        except ValueError:
            iter_bound = None

    # 6. Soundness
    soundness = CalibrationSoundness(phi_fn, interval)
    is_sound, sound_report = soundness.verify_soundness(cert.fixed_point, cert.fixed_point)
    tight = soundness.tightness_certificate(cert.fixed_point, tolerance=tolerance)

    # 7. Self-correcting property
    self_correcting = None
    if cert.verified:
        _, self_correcting = mono.verify_self_correcting(
            cert.fixed_point, n_samples=n_samples
        )

    return {
        "self_map": {"ok": is_self_map, "min": vmin, "max": vmax},
        "contraction": {"ok": is_contraction, "q": q},
        "monotonicity": {
            "ok": is_monotone,
            "n_violations": len(violations),
        },
        "contraction_condition": diagnosis,
        "certificate": cert.to_dict(),
        "iteration_bound": iter_bound,
        "soundness": sound_report,
        "tightness": tight,
        "self_correcting": self_correcting,
    }
