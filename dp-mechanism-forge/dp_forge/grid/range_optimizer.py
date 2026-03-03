"""
Optimal output range computation for DP-Forge.

Given privacy parameter ε and significance level α, this module computes
the smallest range ``B*`` such that the tail probability of the optimal
mechanism is at most α.  A larger range means more grid points are needed
(costing LP size), while a smaller range truncates the mechanism tails
(losing optimality).

Supported mechanism families
----------------------------
- **Laplace**: closed-form ``B* = -(1/ε) · log(α/2)``.
- **Staircase**: tighter bound using the step structure.
- **General**: bisection search on the tail CDF.

Key class
---------
- :class:`RangeOptimizer` with method
  ``compute_optimal_range(epsilon, alpha, mechanism_family)``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError, ConvergenceError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mechanism family enum for range computation
# ---------------------------------------------------------------------------


class RangeMechanismFamily(Enum):
    """Mechanism family hint for range computation."""

    LAPLACE = auto()
    STAIRCASE = auto()
    GENERAL = auto()

    def __repr__(self) -> str:
        return f"RangeMechanismFamily.{self.name}"


# ---------------------------------------------------------------------------
# RangeResult
# ---------------------------------------------------------------------------


@dataclass
class RangeResult:
    """Result of optimal range computation.

    Attributes:
        B_star: Optimal half-range (the mechanism is supported on
            ``[-B*, B*]`` centred at the query output).
        tail_probability: Actual tail probability at ``B*``.
        mechanism_family: Which family was used.
        n_bisection_steps: Number of bisection steps (0 for closed-form).
    """

    B_star: float
    tail_probability: float
    mechanism_family: RangeMechanismFamily
    n_bisection_steps: int = 0

    def __repr__(self) -> str:
        return (
            f"RangeResult(B*={self.B_star:.4f}, tail={self.tail_probability:.2e}, "
            f"family={self.mechanism_family.name})"
        )


# ---------------------------------------------------------------------------
# RangeOptimizer
# ---------------------------------------------------------------------------


class RangeOptimizer:
    """Compute optimal output range for a given privacy level.

    The optimizer selects the range ``B*`` such that the probability
    mass outside ``[-B*, B*]`` is at most ``alpha``, while minimising
    ``B*`` to keep the LP small.

    Parameters
    ----------
    max_bisection_iter : int
        Maximum number of bisection iterations for the general case.
    bisection_tol : float
        Bisection convergence tolerance on ``B*``.

    Example::

        opt = RangeOptimizer()
        result = opt.compute_optimal_range(epsilon=1.0, alpha=1e-6)
        print(f"Use range B = {result.B_star:.2f}")
    """

    def __init__(
        self,
        max_bisection_iter: int = 100,
        bisection_tol: float = 1e-8,
    ) -> None:
        if max_bisection_iter < 1:
            raise ConfigurationError(
                f"max_bisection_iter must be >= 1, got {max_bisection_iter}",
                parameter="max_bisection_iter",
                value=max_bisection_iter,
                constraint=">= 1",
            )
        if bisection_tol <= 0:
            raise ConfigurationError(
                f"bisection_tol must be > 0, got {bisection_tol}",
                parameter="bisection_tol",
                value=bisection_tol,
                constraint="> 0",
            )
        self._max_iter = max_bisection_iter
        self._tol = bisection_tol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_optimal_range(
        self,
        epsilon: float,
        alpha: float,
        mechanism_family: RangeMechanismFamily = RangeMechanismFamily.LAPLACE,
        *,
        sensitivity: float = 1.0,
        tail_cdf: Optional[Callable[[float], float]] = None,
    ) -> RangeResult:
        """Compute the optimal output half-range B*.

        Parameters
        ----------
        epsilon : float
            Privacy parameter ε > 0.
        alpha : float
            Significance level (tail probability threshold), in (0, 1).
        mechanism_family : RangeMechanismFamily
            Which mechanism family to use for the tail bound.
        sensitivity : float
            Query sensitivity (scales the range).
        tail_cdf : callable, optional
            For ``GENERAL`` family: a function ``B -> tail_prob`` giving
            the tail probability at half-range ``B``.  Required for
            ``GENERAL``.

        Returns
        -------
        RangeResult with ``B_star`` and metadata.

        Raises
        ------
        ConfigurationError
            If parameters are invalid.
        ConvergenceError
            If bisection does not converge (general case only).
        """
        self._validate_params(epsilon, alpha, sensitivity)

        if mechanism_family == RangeMechanismFamily.LAPLACE:
            return self._laplace_range(epsilon, alpha, sensitivity)
        elif mechanism_family == RangeMechanismFamily.STAIRCASE:
            return self._staircase_range(epsilon, alpha, sensitivity)
        elif mechanism_family == RangeMechanismFamily.GENERAL:
            if tail_cdf is None:
                raise ConfigurationError(
                    "tail_cdf callable required for GENERAL mechanism family",
                    parameter="tail_cdf",
                    constraint="must be provided",
                )
            return self._general_range(epsilon, alpha, sensitivity, tail_cdf)
        else:
            raise ConfigurationError(
                f"Unknown mechanism family: {mechanism_family}",
                parameter="mechanism_family",
                value=mechanism_family,
            )

    # ------------------------------------------------------------------
    # Laplace range
    # ------------------------------------------------------------------

    def _laplace_range(
        self, epsilon: float, alpha: float, sensitivity: float
    ) -> RangeResult:
        """Closed-form range for Laplace mechanism.

        The Laplace(0, Δ/ε) mechanism has CDF ``F(x) = 0.5 · exp(ε·x/Δ)``
        for ``x < 0``.  The tail probability at ``B`` is
        ``2 · (1 - F(B)) = exp(-ε·B/Δ)``.

        Solving ``exp(-ε·B/Δ) = α`` gives ``B = -(Δ/ε) · log(α/2)``.
        But we need ``α/2`` for two tails, not ``α``.
        """
        scale = sensitivity / epsilon  # b = Δ/ε
        # Two-tailed: P(|X| > B) = exp(-B/b) = α
        # => B = -b * log(α)
        # But the standard Laplace tail is P(|X|>B) = exp(-B/b),
        # so B = b * log(1/α) = -(1/ε) * Δ * log(α)
        # More precisely: P(|X|>B) = exp(-εB/Δ), so B = -(Δ/ε)*ln(α)
        # However, the correct formula is:
        # P(|Lap(0,b)|>B) = exp(-B/b)
        # We want this ≤ α, so B ≥ -b*ln(α)
        B_star = -scale * math.log(alpha)

        # Verify tail probability
        tail_prob = math.exp(-B_star / scale)

        return RangeResult(
            B_star=B_star,
            tail_probability=tail_prob,
            mechanism_family=RangeMechanismFamily.LAPLACE,
            n_bisection_steps=0,
        )

    # ------------------------------------------------------------------
    # Staircase range
    # ------------------------------------------------------------------

    def _staircase_range(
        self, epsilon: float, alpha: float, sensitivity: float
    ) -> RangeResult:
        """Range for staircase mechanism.

        The staircase mechanism has a piecewise-constant PDF that decays
        in steps of width ``Δ`` with ratio ``exp(-ε)`` per step.  The
        tail probability after ``m`` steps is::

            tail(m) = exp(-m·ε) / (1 - exp(-ε))   (geometric sum)

        We find the smallest integer ``m`` such that ``tail(m) ≤ α``
        and set ``B* = m · Δ``.
        """
        ratio = math.exp(-epsilon)
        if ratio >= 1.0:
            # ε ≤ 0 should not happen, but handle gracefully
            B_star = sensitivity * 100  # large fallback
            return RangeResult(
                B_star=B_star,
                tail_probability=alpha,
                mechanism_family=RangeMechanismFamily.STAIRCASE,
            )

        # geometric tail: sum_{j=m}^∞ r^j = r^m / (1-r)
        # We want r^m / (1-r) ≤ α  →  m ≥ log(α(1-r)) / log(r)
        # but log(r) < 0, so m ≥ log(α(1-r)) / log(r)
        # since log(r) = -ε, this is m ≥ -log(α(1-r)) / ε
        arg = alpha * (1.0 - ratio)
        if arg <= 0 or arg >= 1.0:
            # Fallback: use Laplace bound
            return self._laplace_range(epsilon, alpha, sensitivity)

        m = math.ceil(-math.log(arg) / epsilon)
        m = max(1, m)
        B_star = m * sensitivity

        # Actual tail probability
        tail_prob = ratio ** m / (1.0 - ratio)

        return RangeResult(
            B_star=B_star,
            tail_probability=tail_prob,
            mechanism_family=RangeMechanismFamily.STAIRCASE,
            n_bisection_steps=0,
        )

    # ------------------------------------------------------------------
    # General bisection
    # ------------------------------------------------------------------

    def _general_range(
        self,
        epsilon: float,
        alpha: float,
        sensitivity: float,
        tail_cdf: Callable[[float], float],
    ) -> RangeResult:
        """Bisection search for the optimal range.

        Parameters
        ----------
        tail_cdf : callable
            ``B -> tail_probability``.  Must be monotonically decreasing
            in ``B``.
        """
        # Bracket: find an upper bound where tail < alpha
        B_lo = 0.0
        B_hi = sensitivity / epsilon  # Laplace scale as starting point
        # Expand B_hi until tail < alpha
        for _ in range(50):
            if tail_cdf(B_hi) <= alpha:
                break
            B_hi *= 2.0
        else:
            raise ConvergenceError(
                "Could not bracket the optimal range: tail_cdf does not "
                f"decrease below alpha={alpha} within B={B_hi}",
                iterations=50,
                max_iter=50,
            )

        # Bisection
        n_steps = 0
        for n_steps in range(1, self._max_iter + 1):
            B_mid = (B_lo + B_hi) / 2.0
            tail_mid = tail_cdf(B_mid)

            if abs(B_hi - B_lo) < self._tol:
                break

            if tail_mid > alpha:
                B_lo = B_mid
            else:
                B_hi = B_mid

        B_star = (B_lo + B_hi) / 2.0
        tail_final = tail_cdf(B_star)

        if abs(B_hi - B_lo) >= self._tol and n_steps >= self._max_iter:
            raise ConvergenceError(
                f"Bisection did not converge after {self._max_iter} iterations "
                f"(B in [{B_lo:.6f}, {B_hi:.6f}], gap={B_hi - B_lo:.2e})",
                iterations=self._max_iter,
                max_iter=self._max_iter,
            )

        return RangeResult(
            B_star=B_star,
            tail_probability=tail_final,
            mechanism_family=RangeMechanismFamily.GENERAL,
            n_bisection_steps=n_steps,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_params(
        epsilon: float, alpha: float, sensitivity: float
    ) -> None:
        """Validate input parameters."""
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ConfigurationError(
                f"epsilon must be finite and > 0, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
                constraint="> 0",
            )
        if not (0.0 < alpha < 1.0):
            raise ConfigurationError(
                f"alpha must be in (0, 1), got {alpha}",
                parameter="alpha",
                value=alpha,
                constraint="in (0, 1)",
            )
        if sensitivity <= 0 or not math.isfinite(sensitivity):
            raise ConfigurationError(
                f"sensitivity must be finite and > 0, got {sensitivity}",
                parameter="sensitivity",
                value=sensitivity,
                constraint="> 0",
            )

    @staticmethod
    def suggested_k(
        B: float,
        epsilon: float,
        target_error: float = 1e-3,
    ) -> int:
        """Suggest a grid size k for a target discretisation error.

        Uses the uniform-grid bound ``error ≤ B/k`` to compute the
        minimum k needed.

        Parameters
        ----------
        B : float
            Output half-range.
        epsilon : float
            Privacy parameter (unused directly but retained for API
            symmetry).
        target_error : float
            Target L1 discretisation error.

        Returns
        -------
        k : int
            Suggested number of grid points (at least 2).
        """
        if target_error <= 0:
            raise ConfigurationError(
                f"target_error must be > 0, got {target_error}",
                parameter="target_error",
                value=target_error,
                constraint="> 0",
            )
        k = max(2, math.ceil(B / target_error))
        return k

    def __repr__(self) -> str:
        return (
            f"RangeOptimizer(max_iter={self._max_iter}, "
            f"tol={self._tol:.0e})"
        )
