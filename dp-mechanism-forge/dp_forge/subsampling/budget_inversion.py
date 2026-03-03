"""
Numerical inversion of privacy amplification curves for DP-Forge.

Given a target amplified privacy (ε, δ) and a subsampling rate q, this
module finds the base privacy parameters (ε₀, δ₀) such that applying the
amplification theorem to (ε₀, δ₀) at rate q yields the target guarantee.

The inversion is needed for the SubsampledCEGIS workflow: the user
specifies the *final* privacy they want after subsampling, and we must
determine what base-level privacy to request from the CEGIS synthesiser.

Approach:
    - **Monotonicity-based bisection**: The Poisson amplification function
      ``ε → log(1 + q(e^ε - 1))`` is strictly increasing in ε, so
      bisection is guaranteed to converge.
    - **Bracket initialisation**: Uses asymptotic formulas to set tight
      initial brackets, reducing iterations needed.
    - **Multiple solutions**: When q is very small, multiple (ε₀, δ₀)
      pairs can yield the same amplified guarantee.  We select the one
      giving the best utility (smallest ε₀) since that produces the
      most accurate mechanism.

Key Class:
    - :class:`BudgetInverter` — Numerical inversion engine.

Functions:
    - :func:`invert_poisson` — One-shot Poisson inversion.
    - :func:`invert_replacement` — One-shot without-replacement inversion.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from dp_forge.exceptions import ConfigurationError, ConvergenceError
from dp_forge.types import PrivacyBudget

from dp_forge.subsampling.amplification import (
    AmplificationBound,
    AmplificationResult,
    _stable_log_poisson_amplification,
    _validate_base_params,
    poisson_amplify,
    replacement_amplify,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InversionResult:
    """Result of inverting the amplification curve.

    Attributes:
        base_eps: The computed base ε₀ such that amplification yields target ε.
        base_delta: The computed base δ₀ such that amplification yields target δ.
        target_eps: The target amplified ε.
        target_delta: The target amplified δ.
        q_rate: Subsampling rate used.
        n_iterations: Number of bisection iterations used.
        residual: Absolute error |f(ε₀) - ε_target| at convergence.
        bound_type: Which amplification bound was inverted.
    """

    base_eps: float
    base_delta: float
    target_eps: float
    target_delta: float
    q_rate: float
    n_iterations: int
    residual: float
    bound_type: AmplificationBound = AmplificationBound.POISSON_BASIC

    def __post_init__(self) -> None:
        if self.base_eps < 0:
            raise ValueError(f"base_eps must be >= 0, got {self.base_eps}")
        if self.base_delta < 0.0 or self.base_delta >= 1.0:
            raise ValueError(f"base_delta must be in [0, 1), got {self.base_delta}")

    @property
    def base_budget(self) -> PrivacyBudget:
        """Return the computed base privacy as a PrivacyBudget."""
        return PrivacyBudget(epsilon=max(self.base_eps, 1e-15), delta=self.base_delta)

    @property
    def target_budget(self) -> PrivacyBudget:
        """Return the target amplified privacy as a PrivacyBudget."""
        return PrivacyBudget(epsilon=max(self.target_eps, 1e-15), delta=self.target_delta)

    def verify(self, tol: float = 1e-6) -> bool:
        """Verify that amplification of base params meets the target.

        Args:
            tol: Tolerance for the verification.

        Returns:
            True if the amplified (ε, δ) is within tolerance of the target.
        """
        amplified = poisson_amplify(self.base_eps, self.base_delta, self.q_rate)
        eps_ok = abs(amplified.eps - self.target_eps) <= tol
        delta_ok = abs(amplified.delta - self.target_delta) <= tol
        return eps_ok and delta_ok

    def __repr__(self) -> str:
        return (
            f"InversionResult(base_ε={self.base_eps:.6f}, base_δ={self.base_delta:.2e}, "
            f"target_ε={self.target_eps:.6f}, q={self.q_rate:.4f}, "
            f"residual={self.residual:.2e})"
        )


# ---------------------------------------------------------------------------
# Asymptotic bracket initialisation
# ---------------------------------------------------------------------------


def _asymptotic_upper_bound(target_eps: float, q_rate: float) -> float:
    """Compute an asymptotic upper bound on ε₀ for bracket initialisation.

    For the Poisson formula ε = log(1 + q(e^ε₀ - 1)), we need ε₀ such
    that the formula yields target_eps.

    An upper bound: since log(1 + q·x) ≤ q·x for x ≥ 0, and e^ε₀ - 1 ≥ ε₀,
    we have ε ≤ q·(e^ε₀ - 1).  Thus e^ε₀ - 1 ≥ ε/q, giving
    ε₀ ≤ log(1 + ε/q).  But this is only a rough bound.

    A tighter approach: since ε = log(1 + q(e^ε₀ - 1)), we get
    e^ε - 1 = q(e^ε₀ - 1), thus e^ε₀ = 1 + (e^ε - 1)/q, giving
    ε₀ = log(1 + (e^ε - 1)/q).

    For the upper bracket, we use ε₀ = log(1 + expm1(ε)/q) + margin.

    Args:
        target_eps: Target amplified ε.
        q_rate: Subsampling rate q.

    Returns:
        Upper bound on ε₀.
    """
    if q_rate >= 1.0:
        return target_eps

    # Exact inversion of the Poisson formula
    expm1_target = math.expm1(target_eps)
    ratio = expm1_target / q_rate
    eps_0_exact = math.log1p(ratio)

    # Add a 10% margin for numerical safety
    return eps_0_exact * 1.1 + 1e-10


def _asymptotic_lower_bound(target_eps: float, q_rate: float) -> float:
    """Compute an asymptotic lower bound on ε₀ for bracket initialisation.

    Since ε = log(1 + q(e^ε₀ - 1)) ≤ q·ε₀ for small ε₀ (by log(1+x) ≤ x
    and e^x - 1 ≥ x), we have ε₀ ≥ ε/q as a lower bound for small ε₀.

    For the actual lower bracket, we use the exact inversion minus a margin.

    Args:
        target_eps: Target amplified ε.
        q_rate: Subsampling rate q.

    Returns:
        Lower bound on ε₀.
    """
    if q_rate >= 1.0:
        return target_eps

    # The function is increasing, and ε₀ ≥ target_eps always
    # (since amplification can only reduce ε)
    return target_eps


# ---------------------------------------------------------------------------
# Core bisection
# ---------------------------------------------------------------------------


def _bisect_poisson_eps(
    target_eps: float,
    q_rate: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[float, int, float]:
    """Find ε₀ such that log(1 + q(e^ε₀ - 1)) = target_eps via bisection.

    The function f(ε₀) = log(1 + q(e^ε₀ - 1)) is strictly increasing in ε₀
    for q > 0, so bisection is guaranteed to converge.

    Args:
        target_eps: Target amplified ε > 0.
        q_rate: Subsampling rate q ∈ (0, 1).
        tol: Convergence tolerance on |f(ε₀) - target_eps|.
        max_iter: Maximum bisection iterations.

    Returns:
        (base_eps, n_iterations, residual).

    Raises:
        ConvergenceError: If bisection fails to converge.
    """
    if target_eps == 0.0:
        return 0.0, 0, 0.0

    if q_rate >= 1.0:
        return target_eps, 0, 0.0

    # Initialise brackets
    lo = _asymptotic_lower_bound(target_eps, q_rate)
    hi = _asymptotic_upper_bound(target_eps, q_rate)

    # Verify bracket validity
    f_lo = _stable_log_poisson_amplification(lo, q_rate)
    f_hi = _stable_log_poisson_amplification(hi, q_rate)

    # Widen brackets if needed
    widen_count = 0
    while f_lo > target_eps and widen_count < 50:
        lo = lo / 2.0
        f_lo = _stable_log_poisson_amplification(lo, q_rate)
        widen_count += 1

    widen_count = 0
    while f_hi < target_eps and widen_count < 50:
        hi = hi * 2.0
        f_hi = _stable_log_poisson_amplification(hi, q_rate)
        widen_count += 1

    if f_lo > target_eps or f_hi < target_eps:
        raise ConvergenceError(
            f"Failed to bracket ε₀ for target_eps={target_eps}, q={q_rate}: "
            f"f({lo})={f_lo}, f({hi})={f_hi}",
            iterations=0,
            max_iter=max_iter,
        )

    # Bisection
    for iteration in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = _stable_log_poisson_amplification(mid, q_rate)
        residual = abs(f_mid - target_eps)

        if residual < tol:
            return mid, iteration + 1, residual

        if f_mid < target_eps:
            lo = mid
        else:
            hi = mid

        # Also check if the interval is tiny
        if (hi - lo) < tol * 1e-3:
            return mid, iteration + 1, residual

    # Return best estimate even if not fully converged
    mid = (lo + hi) / 2.0
    f_mid = _stable_log_poisson_amplification(mid, q_rate)
    residual = abs(f_mid - target_eps)
    return mid, max_iter, residual


def _bisect_replacement_eps(
    target_eps: float,
    q_rate: float,
    n_total: Optional[int],
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[float, int, float]:
    """Find ε₀ for without-replacement amplification via bisection.

    Similar to Poisson bisection but uses the without-replacement
    amplification function which includes a finite-population correction.

    Args:
        target_eps: Target amplified ε.
        q_rate: Subsampling rate q.
        n_total: Total dataset size (or None for Poisson fallback).
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        (base_eps, n_iterations, residual).
    """
    if target_eps == 0.0:
        return 0.0, 0, 0.0

    if q_rate >= 1.0:
        return target_eps, 0, 0.0

    def f(eps0: float) -> float:
        result = replacement_amplify(eps0, 0.0, q_rate, n_total=n_total)
        return result.eps

    # Initialise brackets (slightly wider than Poisson since WOR has correction)
    lo = target_eps * 0.9
    hi = _asymptotic_upper_bound(target_eps, q_rate) * 1.2

    f_lo = f(lo)
    f_hi = f(hi)

    # Widen if needed
    widen_count = 0
    while f_lo > target_eps and widen_count < 50:
        lo = lo / 2.0
        f_lo = f(lo)
        widen_count += 1

    widen_count = 0
    while f_hi < target_eps and widen_count < 50:
        hi = hi * 2.0
        f_hi = f(hi)
        widen_count += 1

    if f_lo > target_eps or f_hi < target_eps:
        raise ConvergenceError(
            f"Failed to bracket ε₀ for WOR inversion: "
            f"target_eps={target_eps}, q={q_rate}",
            iterations=0,
            max_iter=max_iter,
        )

    for iteration in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        residual = abs(f_mid - target_eps)

        if residual < tol:
            return mid, iteration + 1, residual

        if f_mid < target_eps:
            lo = mid
        else:
            hi = mid

        if (hi - lo) < tol * 1e-3:
            return mid, iteration + 1, residual

    mid = (lo + hi) / 2.0
    f_mid = f(mid)
    return mid, max_iter, abs(f_mid - target_eps)


# =========================================================================
# BudgetInverter class
# =========================================================================


class BudgetInverter:
    """Numerical engine for inverting privacy amplification curves.

    Given a target amplified privacy (ε, δ) and subsampling rate q,
    finds the base (ε₀, δ₀) such that amplification of (ε₀, δ₀) at
    rate q yields the target guarantee.

    The inversion is performed via monotonicity-based bisection with
    tight bracket initialisation using asymptotic formulas.

    Args:
        tol: Convergence tolerance for bisection.
        max_iter: Maximum bisection iterations.

    Example::

        inverter = BudgetInverter(tol=1e-10)
        result = inverter.invert(target_eps=0.5, target_delta=1e-5, q_rate=0.01)
        print(result.base_eps)  # Much larger than 0.5
    """

    def __init__(
        self,
        tol: float = 1e-12,
        max_iter: int = 200,
    ) -> None:
        if tol <= 0:
            raise ConfigurationError(
                f"tol must be > 0, got {tol}",
                parameter="tol",
                value=tol,
                constraint="> 0",
            )
        if max_iter < 1:
            raise ConfigurationError(
                f"max_iter must be >= 1, got {max_iter}",
                parameter="max_iter",
                value=max_iter,
                constraint=">= 1",
            )
        self._tol = tol
        self._max_iter = max_iter

    def invert(
        self,
        target_eps: float,
        target_delta: float,
        q_rate: float,
        *,
        bound_type: AmplificationBound = AmplificationBound.POISSON_BASIC,
        n_total: Optional[int] = None,
    ) -> InversionResult:
        """Invert the amplification curve to find base privacy parameters.

        Given target amplified (ε, δ) and rate q, finds (ε₀, δ₀) such that
        amplifying (ε₀, δ₀) at rate q yields (ε, δ).

        The ε inversion is done via bisection on the monotone amplification
        function.  The δ inversion is straightforward: δ₀ = δ / q.

        Args:
            target_eps: Target amplified ε > 0.
            target_delta: Target amplified δ ∈ [0, 1).
            q_rate: Subsampling rate q ∈ (0, 1].
            bound_type: Which amplification bound to invert.
            n_total: Dataset size (for WITHOUT_REPLACEMENT bounds).

        Returns:
            InversionResult with the computed base (ε₀, δ₀).

        Raises:
            ConfigurationError: If parameters are invalid.
            ConvergenceError: If bisection fails to converge.
        """
        if target_eps <= 0:
            raise ConfigurationError(
                f"target_eps must be > 0, got {target_eps}",
                parameter="target_eps",
                value=target_eps,
                constraint="> 0",
            )
        if not math.isfinite(target_eps):
            raise ConfigurationError(
                f"target_eps must be finite, got {target_eps}",
                parameter="target_eps",
                value=target_eps,
                constraint="finite",
            )
        if target_delta < 0.0 or target_delta >= 1.0:
            raise ConfigurationError(
                f"target_delta must be in [0, 1), got {target_delta}",
                parameter="target_delta",
                value=target_delta,
                constraint="[0, 1)",
            )
        if not (0.0 < q_rate <= 1.0):
            raise ConfigurationError(
                f"q_rate must be in (0, 1], got {q_rate}",
                parameter="q_rate",
                value=q_rate,
                constraint="(0, 1]",
            )

        # Trivial case: q = 1 means no subsampling
        if q_rate == 1.0:
            return InversionResult(
                base_eps=target_eps,
                base_delta=target_delta,
                target_eps=target_eps,
                target_delta=target_delta,
                q_rate=q_rate,
                n_iterations=0,
                residual=0.0,
                bound_type=bound_type,
            )

        # Invert δ: δ₀ = δ / q (since amplified δ = q · δ₀)
        base_delta = target_delta / q_rate
        if base_delta >= 1.0:
            # δ₀ would exceed 1; clamp and warn
            warnings.warn(
                f"Inverted base_delta={base_delta:.4f} >= 1.0; "
                f"clamping to 0.999. The target (ε={target_eps}, "
                f"δ={target_delta}) may be unachievable at q={q_rate}.",
                stacklevel=2,
            )
            base_delta = 1.0 - 1e-10

        # Invert ε via bisection
        if bound_type in (
            AmplificationBound.POISSON_BASIC,
            AmplificationBound.POISSON_TIGHT,
        ):
            base_eps, n_iter, residual = _bisect_poisson_eps(
                target_eps, q_rate, tol=self._tol, max_iter=self._max_iter,
            )
        elif bound_type == AmplificationBound.WITHOUT_REPLACEMENT:
            base_eps, n_iter, residual = _bisect_replacement_eps(
                target_eps, q_rate, n_total,
                tol=self._tol, max_iter=self._max_iter,
            )
        else:
            # Default to Poisson inversion
            base_eps, n_iter, residual = _bisect_poisson_eps(
                target_eps, q_rate, tol=self._tol, max_iter=self._max_iter,
            )

        return InversionResult(
            base_eps=base_eps,
            base_delta=base_delta,
            target_eps=target_eps,
            target_delta=target_delta,
            q_rate=q_rate,
            n_iterations=n_iter,
            residual=residual,
            bound_type=bound_type,
        )

    def invert_for_best_utility(
        self,
        target_eps: float,
        target_delta: float,
        q_rate: float,
    ) -> InversionResult:
        """Invert using the bound that gives the best utility (smallest ε₀).

        Tries all applicable Poisson bounds and returns the one yielding
        the smallest base ε₀, since smaller ε₀ means the CEGIS engine
        can find a more accurate mechanism.

        Args:
            target_eps: Target amplified ε.
            target_delta: Target amplified δ.
            q_rate: Subsampling rate q.

        Returns:
            InversionResult from the bound giving smallest base ε₀.
        """
        bounds_to_try = [
            AmplificationBound.POISSON_BASIC,
            AmplificationBound.POISSON_TIGHT,
        ]

        best: Optional[InversionResult] = None
        for bt in bounds_to_try:
            try:
                result = self.invert(
                    target_eps, target_delta, q_rate, bound_type=bt,
                )
                if best is None or result.base_eps < best.base_eps:
                    best = result
            except (ConvergenceError, ConfigurationError):
                continue

        if best is None:
            raise ConvergenceError(
                f"All inversion methods failed for target_eps={target_eps}, "
                f"target_delta={target_delta}, q={q_rate}",
                iterations=0,
                max_iter=self._max_iter,
            )

        return best


# =========================================================================
# One-shot convenience functions
# =========================================================================


def invert_poisson(
    target_eps: float,
    target_delta: float,
    q_rate: float,
    *,
    tol: float = 1e-12,
) -> InversionResult:
    """One-shot inversion of the Poisson subsampling amplification.

    Convenience wrapper around :class:`BudgetInverter` for the common
    case of Poisson subsampling.

    Args:
        target_eps: Target amplified ε.
        target_delta: Target amplified δ.
        q_rate: Subsampling rate q.
        tol: Bisection tolerance.

    Returns:
        InversionResult with base (ε₀, δ₀).
    """
    inverter = BudgetInverter(tol=tol)
    return inverter.invert(
        target_eps, target_delta, q_rate,
        bound_type=AmplificationBound.POISSON_BASIC,
    )


def invert_replacement(
    target_eps: float,
    target_delta: float,
    q_rate: float,
    *,
    n_total: Optional[int] = None,
    tol: float = 1e-12,
) -> InversionResult:
    """One-shot inversion of without-replacement amplification.

    Args:
        target_eps: Target amplified ε.
        target_delta: Target amplified δ.
        q_rate: Subsampling rate q.
        n_total: Dataset size for hypergeometric correction.
        tol: Bisection tolerance.

    Returns:
        InversionResult with base (ε₀, δ₀).
    """
    inverter = BudgetInverter(tol=tol)
    return inverter.invert(
        target_eps, target_delta, q_rate,
        bound_type=AmplificationBound.WITHOUT_REPLACEMENT,
        n_total=n_total,
    )
