"""
(ε, δ)-Differential Privacy verification with counterexample generation.

This module implements the **Verify** step of the DP-Forge CEGIS loop.
Given a mechanism's probability table ``p[i][j] = Pr[M(x_i) = y_j]``,
it checks whether the mechanism satisfies (ε, δ)-DP over all adjacent
database pairs and, when it does not, returns the most-violating pair
for use as a CEGIS counterexample.

Theory Summary
--------------
**Pure DP** (δ = 0):
    For every adjacent pair (i, i') and every output j:
        p[i][j] / p[i'][j]  ≤  e^ε
    A single bin j where the ratio exceeds e^ε + tol is a violation.

**Approximate DP** (δ > 0):
    For every adjacent pair (i, i'), the hockey-stick divergence satisfies:
        H_ε(p[i] ‖ p[i'])  =  Σ_j max(p[i][j] − e^ε · p[i'][j], 0)  ≤  δ
    Individual bin ratios CAN exceed e^ε — that is expected.  Only the
    aggregate hockey-stick divergence matters.

Invariant I4
    The verification tolerance must satisfy:
        tol  ≥  exp(ε) × solver_primal_tol
    This ensures that solver numerical noise does not cause false
    verification failures.

Divergence Measures
-------------------
The module provides numerically stable implementations of:

- Hockey-stick divergence  H_ε(P ‖ Q)
- KL divergence  D_KL(P ‖ Q)
- Rényi divergence  D_α(P ‖ Q)
- Max divergence  D_∞(P ‖ Q)
- Total variation  TV(P, Q)

All divergence computations use log-domain arithmetic and
epsilon-smoothing to handle near-zero probabilities without NaN.

Classes
-------
- :class:`VerificationReport` — Structured report of verification findings.
- :class:`PrivacyVerifier` — Full-featured deterministic verifier.
- :class:`MonteCarloVerifier` — Statistical privacy auditing via sampling.

Functions
---------
- :func:`verify` — Main entry point for CEGIS loop verification.
- :func:`hockey_stick_divergence` — Core divergence computation.
- :func:`compute_safe_tolerance` — Auto-compute I4-safe tolerance.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
    VerificationError,
)
from dp_forge.types import (
    AdjacencyRelation,
    ExtractedMechanism,
    NumericalConfig,
    PrivacyBudget,
    QuerySpec,
    VerifyResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum probability floor to avoid division by zero / log(0).
# Values below this are clipped before ratio / log computations.
_PROB_FLOOR: float = 1e-300

# Default verification tolerance when none is specified.
_DEFAULT_TOL: float = 1e-9

# Default solver primal tolerance for I4 computation.
_DEFAULT_SOLVER_TOL: float = 1e-8

# Maximum number of violations to track in detailed mode.
_MAX_VIOLATIONS_DETAIL: int = 10_000


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VerificationMode(Enum):
    """Verification mode controlling thoroughness vs. speed.

    ``FAST``
        Return as soon as the first violation is found.  Suitable for
        CEGIS inner loops where only one counterexample is needed.

    ``MOST_VIOLATING``
        Scan all pairs and return the pair with the largest violation.
        Theory shows this reduces CEGIS iterations by ~40%.

    ``EXHAUSTIVE``
        Record every violating pair.  Useful for diagnostics and
        verification reports.
    """

    FAST = auto()
    MOST_VIOLATING = auto()
    EXHAUSTIVE = auto()

    def __repr__(self) -> str:
        return f"VerificationMode.{self.name}"


class ViolationType(Enum):
    """Classification of a DP violation."""

    PURE_DP_RATIO = auto()
    APPROX_DP_HOCKEY_STICK = auto()

    def __repr__(self) -> str:
        return f"ViolationType.{self.name}"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ViolationRecord:
    """A single DP violation found during verification.

    Attributes:
        i: First database index in the adjacent pair.
        i_prime: Second database index in the adjacent pair.
        j_worst: Output bin index where the worst ratio occurs (pure DP),
            or -1 for approximate DP (where the divergence is aggregate).
        magnitude: Violation magnitude — amount by which the DP bound is
            exceeded.  For pure DP: ``ratio - e^ε``.  For approx DP:
            ``hockey_stick - δ``.
        ratio: The worst probability ratio ``p[i][j] / p[i'][j]`` (pure DP)
            or the hockey-stick divergence value (approx DP).
        violation_type: Whether this is a pure or approximate DP violation.
        direction: ``'forward'`` if the violation is p[i]→p[i'], ``'reverse'``
            if p[i']→p[i].
    """

    i: int
    i_prime: int
    j_worst: int
    magnitude: float
    ratio: float
    violation_type: ViolationType
    direction: str = "forward"

    def to_tuple(self) -> Tuple[int, int, int, float]:
        """Convert to the (i, i', j, magnitude) format used by VerifyResult."""
        return (self.i, self.i_prime, self.j_worst, self.magnitude)

    def __repr__(self) -> str:
        return (
            f"ViolationRecord(({self.i},{self.i_prime}), j={self.j_worst}, "
            f"mag={self.magnitude:.2e}, {self.violation_type.name})"
        )


@dataclass
class PairAnalysis:
    """Per-pair analysis result for verification reports.

    Attributes:
        i: First database index.
        i_prime: Second database index.
        max_ratio_forward: Worst ratio p[i][j]/p[i'][j] across all j.
        max_ratio_reverse: Worst ratio p[i'][j]/p[i][j] across all j.
        j_worst_forward: Bin achieving max_ratio_forward.
        j_worst_reverse: Bin achieving max_ratio_reverse.
        hockey_stick_forward: H_ε(p[i] ‖ p[i']).
        hockey_stick_reverse: H_ε(p[i'] ‖ p[i]).
        is_violating: Whether this pair violates DP.
    """

    i: int
    i_prime: int
    max_ratio_forward: float
    max_ratio_reverse: float
    j_worst_forward: int
    j_worst_reverse: int
    hockey_stick_forward: float
    hockey_stick_reverse: float
    is_violating: bool

    def __repr__(self) -> str:
        status = "VIOLATING" if self.is_violating else "ok"
        return (
            f"PairAnalysis(({self.i},{self.i_prime}), "
            f"max_ratio={max(self.max_ratio_forward, self.max_ratio_reverse):.4f}, "
            f"HS_fwd={self.hockey_stick_forward:.2e}, "
            f"HS_rev={self.hockey_stick_reverse:.2e}, {status})"
        )


@dataclass
class VerificationReport:
    """Structured report of a full verification run.

    Attributes:
        is_valid: Whether the mechanism satisfies (ε, δ)-DP within tolerance.
        epsilon: Target privacy parameter ε.
        delta: Target privacy parameter δ.
        tolerance: Verification tolerance used.
        n_pairs_checked: Total number of adjacent pairs checked.
        n_violations: Number of violating pairs found.
        worst_violation: The single worst violation, if any.
        all_violations: All violations found (only in EXHAUSTIVE mode).
        pair_analyses: Per-pair analysis (only in EXHAUSTIVE mode).
        actual_epsilon: Tightest ε the mechanism achieves (if computed).
        actual_delta: Tightest δ at the target ε (if computed).
        verification_time_s: Wall-clock time for verification in seconds.
        recommendations: Human-readable recommendations.
        metadata: Additional metadata.
    """

    is_valid: bool
    epsilon: float
    delta: float
    tolerance: float
    n_pairs_checked: int
    n_violations: int
    worst_violation: Optional[ViolationRecord] = None
    all_violations: List[ViolationRecord] = field(default_factory=list)
    pair_analyses: List[PairAnalysis] = field(default_factory=list)
    actual_epsilon: Optional[float] = None
    actual_delta: Optional[float] = None
    verification_time_s: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a concise human-readable summary."""
        status = "PASS ✓" if self.is_valid else "FAIL ✗"
        lines = [
            f"DP Verification Report: {status}",
            f"  Target: (ε={self.epsilon}, δ={self.delta})-DP",
            f"  Tolerance: {self.tolerance:.2e}",
            f"  Pairs checked: {self.n_pairs_checked}",
            f"  Violations: {self.n_violations}",
        ]
        if self.worst_violation is not None:
            v = self.worst_violation
            lines.append(
                f"  Worst violation: pair ({v.i},{v.i_prime}), "
                f"magnitude={v.magnitude:.2e}"
            )
        if self.actual_epsilon is not None:
            lines.append(f"  Actual ε: {self.actual_epsilon:.6f}")
        if self.actual_delta is not None:
            lines.append(f"  Actual δ: {self.actual_delta:.2e}")
        lines.append(f"  Time: {self.verification_time_s:.3f}s")
        if self.recommendations:
            lines.append("  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    • {rec}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "valid" if self.is_valid else "INVALID"
        return (
            f"VerificationReport({status}, ε={self.epsilon}, δ={self.delta}, "
            f"violations={self.n_violations})"
        )


# ============================================================================
# Tolerance Management
# ============================================================================


def compute_safe_tolerance(
    epsilon: float,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
    safety_factor: float = 2.0,
) -> float:
    """Compute a verification tolerance satisfying Invariant I4.

    Invariant I4 requires::

        tol  ≥  exp(ε) × solver_primal_tol

    This function returns ``safety_factor × exp(ε) × solver_tol`` to
    provide margin above the theoretical minimum.

    Args:
        epsilon: Privacy parameter ε.
        solver_tol: Solver primal feasibility tolerance.
        safety_factor: Multiplicative safety margin (default 2×).

    Returns:
        A safe verification tolerance.

    Raises:
        ConfigurationError: If epsilon or solver_tol is invalid.
    """
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ConfigurationError(
            f"epsilon must be finite and > 0, got {epsilon}",
            parameter="epsilon",
            value=epsilon,
            constraint="epsilon > 0 and finite",
        )
    if solver_tol <= 0 or not math.isfinite(solver_tol):
        raise ConfigurationError(
            f"solver_tol must be finite and > 0, got {solver_tol}",
            parameter="solver_tol",
            value=solver_tol,
            constraint="solver_tol > 0 and finite",
        )
    return safety_factor * math.exp(epsilon) * solver_tol


def validate_tolerance(
    tol: float,
    epsilon: float,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
) -> bool:
    """Check whether a tolerance satisfies Invariant I4.

    Args:
        tol: Verification tolerance to check.
        epsilon: Privacy parameter ε.
        solver_tol: Solver primal feasibility tolerance.

    Returns:
        ``True`` if ``tol >= exp(ε) × solver_tol``.
    """
    required = math.exp(epsilon) * solver_tol
    return tol >= required


def warn_tolerance_violation(
    tol: float,
    epsilon: float,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
) -> None:
    """Emit a warning if the tolerance violates Invariant I4.

    This does not raise an exception — it simply warns, since the caller
    may intentionally use a tight tolerance for exploratory work.

    Args:
        tol: Verification tolerance.
        epsilon: Privacy parameter ε.
        solver_tol: Solver primal feasibility tolerance.
    """
    required = math.exp(epsilon) * solver_tol
    if tol < required:
        warnings.warn(
            f"Invariant I4 violated: tol ({tol:.2e}) < exp(ε)·solver_tol "
            f"({required:.2e}).  Verification may produce false positives.  "
            f"Recommended minimum: {compute_safe_tolerance(epsilon, solver_tol):.2e}",
            stacklevel=2,
        )
        logger.warning(
            "I4 violation: tol=%.2e < required=%.2e (eps=%.4f, solver_tol=%.2e)",
            tol,
            required,
            epsilon,
            solver_tol,
        )


# ============================================================================
# Mechanism Validation Helpers
# ============================================================================


def _validate_probability_table(
    p: npt.NDArray[np.float64],
    name: str = "p",
    stochastic_tol: float = 1e-6,
) -> None:
    """Validate that ``p`` is a valid probability table (row-stochastic).

    Checks:
    - 2-D array
    - All entries non-negative (within tolerance)
    - Rows sum to 1 (within tolerance)

    Args:
        p: Probability table of shape (n, k).
        name: Name for error messages.
        stochastic_tol: Tolerance for row-sum check.

    Raises:
        InvalidMechanismError: If any check fails.
    """
    if p.ndim != 2:
        raise InvalidMechanismError(
            f"{name} must be 2-D, got shape {p.shape}",
            reason="wrong_dimensionality",
            actual_shape=p.shape if p.ndim == 2 else None,
        )

    n, k = p.shape
    if n == 0 or k == 0:
        raise InvalidMechanismError(
            f"{name} must have non-zero dimensions, got shape ({n}, {k})",
            reason="empty_table",
            actual_shape=(n, k),
        )

    # Check non-negativity (allow tiny numerical noise)
    min_val = float(np.min(p))
    if min_val < -stochastic_tol:
        raise InvalidMechanismError(
            f"{name} contains negative probabilities (min: {min_val:.2e})",
            reason="negative_probabilities",
            actual_shape=(n, k),
        )

    # Check row sums
    row_sums = p.sum(axis=1)
    max_deviation = float(np.max(np.abs(row_sums - 1.0)))
    if max_deviation > stochastic_tol:
        worst_row = int(np.argmax(np.abs(row_sums - 1.0)))
        raise InvalidMechanismError(
            f"{name} rows must sum to 1 (max deviation: {max_deviation:.2e} "
            f"at row {worst_row}, sum={row_sums[worst_row]:.8f})",
            reason="non_stochastic",
            actual_shape=(n, k),
        )

    # Check for NaN / Inf
    if not np.all(np.isfinite(p)):
        n_nonfinite = int(np.sum(~np.isfinite(p)))
        raise InvalidMechanismError(
            f"{name} contains {n_nonfinite} non-finite values (NaN or Inf)",
            reason="non_finite_values",
            actual_shape=(n, k),
        )


def _validate_edges(
    edges: List[Tuple[int, int]],
    n: int,
    symmetric: bool = True,
) -> List[Tuple[int, int]]:
    """Validate and optionally expand edge list.

    Args:
        edges: List of (i, i') pairs.
        n: Number of database inputs.
        symmetric: If True, edges are treated as undirected (checked both
            directions).

    Returns:
        Validated edge list (expanded with reverse edges if symmetric).

    Raises:
        ConfigurationError: If edges are invalid.
    """
    if not edges:
        raise ConfigurationError(
            "edges list must be non-empty",
            parameter="edges",
            constraint="at least one adjacent pair",
        )

    validated = []
    seen = set()
    for i, ip in edges:
        if not (0 <= i < n):
            raise ConfigurationError(
                f"Edge ({i}, {ip}): index {i} out of range [0, {n})",
                parameter="edges",
                value=(i, ip),
                constraint=f"0 <= i < {n}",
            )
        if not (0 <= ip < n):
            raise ConfigurationError(
                f"Edge ({i}, {ip}): index {ip} out of range [0, {n})",
                parameter="edges",
                value=(i, ip),
                constraint=f"0 <= i' < {n}",
            )
        if i == ip:
            raise ConfigurationError(
                f"Self-loop ({i}, {ip}) is not a valid adjacency edge",
                parameter="edges",
                value=(i, ip),
                constraint="i != i'",
            )
        pair_fwd = (i, ip)
        if pair_fwd not in seen:
            validated.append(pair_fwd)
            seen.add(pair_fwd)
        if symmetric:
            pair_rev = (ip, i)
            if pair_rev not in seen:
                validated.append(pair_rev)
                seen.add(pair_rev)
    return validated


# ============================================================================
# Divergence Computations
# ============================================================================


def hockey_stick_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    epsilon: float,
) -> float:
    """Compute the hockey-stick divergence H_ε(P ‖ Q).

    The hockey-stick divergence (also called the E_γ divergence with
    γ = e^ε) is defined as::

        H_ε(P ‖ Q) = Σ_j max(P_j − e^ε · Q_j, 0)

    This is the key quantity for (ε, δ)-DP verification: a mechanism
    satisfies (ε, δ)-DP if and only if H_ε(P ‖ Q) ≤ δ for all adjacent
    pairs (in both directions).

    The computation is vectorized and numerically stable for near-zero
    probabilities.

    Args:
        p: Distribution P, shape (k,).  Must be non-negative and sum to ~1.
        q: Distribution Q, shape (k,).  Must be non-negative and sum to ~1.
        epsilon: Privacy parameter ε.

    Returns:
        The hockey-stick divergence value, a non-negative float.

    Raises:
        ValueError: If p and q have different shapes.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )

    exp_eps = math.exp(epsilon)
    # Vectorized: max(p_j - e^eps * q_j, 0), then sum
    diff = p - exp_eps * q
    return float(np.sum(np.maximum(diff, 0.0)))


def hockey_stick_divergence_detailed(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    epsilon: float,
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Compute hockey-stick divergence with per-bin contributions.

    Like :func:`hockey_stick_divergence`, but also returns the per-bin
    contributions ``max(P_j − e^ε · Q_j, 0)`` for diagnostic analysis.

    Args:
        p: Distribution P, shape (k,).
        q: Distribution Q, shape (k,).
        epsilon: Privacy parameter ε.

    Returns:
        Tuple of (total divergence, per-bin contributions array).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )

    exp_eps = math.exp(epsilon)
    diff = p - exp_eps * q
    contributions = np.maximum(diff, 0.0)
    return float(np.sum(contributions)), contributions


def kl_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Compute KL divergence D_KL(P ‖ Q) in nats.

    Uses log-domain computation with epsilon-smoothing for numerical
    stability::

        D_KL(P ‖ Q) = Σ_j P_j · log(P_j / Q_j)

    Bins where P_j = 0 contribute 0 (by convention 0·log(0) = 0).
    Bins where Q_j = 0 but P_j > 0 yield +∞.

    Args:
        p: Distribution P, shape (k,).
        q: Distribution Q, shape (k,).

    Returns:
        KL divergence in nats (non-negative, possibly +inf).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )

    # Mask where p > 0
    mask = p > 0
    if not np.any(mask):
        return 0.0

    p_pos = p[mask]
    q_pos = q[mask]

    # If any q_j = 0 where p_j > 0, KL is infinite
    if np.any(q_pos <= 0):
        return float("inf")

    # Log-domain computation: p * (log(p) - log(q))
    log_ratio = np.log(np.maximum(p_pos, _PROB_FLOOR)) - np.log(
        np.maximum(q_pos, _PROB_FLOOR)
    )
    return float(np.sum(p_pos * log_ratio))


def renyi_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    alpha: float,
) -> float:
    """Compute Rényi divergence D_α(P ‖ Q).

    The Rényi divergence of order α is::

        D_α(P ‖ Q) = 1/(α−1) · log(Σ_j P_j^α · Q_j^(1−α))

    Special cases:
    - α → 1: Returns KL divergence.
    - α → ∞: Returns max-divergence D_∞.
    - α = 0.5: Bhattacharyya divergence (× -2).

    Uses log-domain computation for stability.

    Args:
        p: Distribution P, shape (k,).
        q: Distribution Q, shape (k,).
        alpha: Rényi order, must be > 0 and ≠ 1 (use α close to 1 for KL
            approximation, or call :func:`kl_divergence` directly).

    Returns:
        Rényi divergence in nats (non-negative, possibly +inf).

    Raises:
        ValueError: If alpha ≤ 0 or alpha = 1.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    # Special case: α → 1 is KL divergence
    if abs(alpha - 1.0) < 1e-12:
        return kl_divergence(p, q)

    # Special case: large α → max divergence
    if alpha > 1e6:
        return max_divergence(p, q)

    # Smooth to avoid 0^x issues
    p_safe = np.maximum(p, _PROB_FLOOR)
    q_safe = np.maximum(q, _PROB_FLOOR)

    # Log-domain computation:
    # log(Σ p^α q^(1-α)) = log(Σ exp(α·log(p) + (1-α)·log(q)))
    log_p = np.log(p_safe)
    log_q = np.log(q_safe)

    # Only include terms where original p > 0 (since 0^α = 0 for α > 0)
    mask = p > 0
    if not np.any(mask):
        return 0.0

    # Check for q = 0 with p > 0 when α > 1
    if alpha > 1.0 and np.any((q[mask] <= 0)):
        return float("inf")

    log_terms = alpha * log_p[mask] + (1.0 - alpha) * log_q[mask]

    # Use log-sum-exp for numerical stability
    max_log = np.max(log_terms)
    if not np.isfinite(max_log):
        return float("inf")

    log_sum = max_log + np.log(np.sum(np.exp(log_terms - max_log)))

    result = log_sum / (alpha - 1.0)
    return max(float(result), 0.0)


def max_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Compute max-divergence (infinity-Rényi) D_∞(P ‖ Q).

    The max-divergence is::

        D_∞(P ‖ Q) = max_j log(P_j / Q_j)

    where the max is over bins j where P_j > 0.

    This is the log of the worst-case likelihood ratio and directly
    corresponds to pure ε-DP: a mechanism satisfies ε-DP if and only if
    D_∞(P ‖ Q) ≤ ε for all adjacent pairs (both directions).

    Args:
        p: Distribution P, shape (k,).
        q: Distribution Q, shape (k,).

    Returns:
        Max-divergence in nats (non-negative, possibly +inf).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )

    mask = p > 0
    if not np.any(mask):
        return 0.0

    p_pos = p[mask]
    q_pos = q[mask]

    # If any q_j = 0 where p_j > 0, divergence is infinite
    if np.any(q_pos <= 0):
        return float("inf")

    log_ratios = np.log(p_pos) - np.log(q_pos)
    return max(float(np.max(log_ratios)), 0.0)


def total_variation(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Compute total variation distance TV(P, Q).

    The total variation distance is::

        TV(P, Q) = 0.5 · Σ_j |P_j − Q_j|

    This equals the hockey-stick divergence at ε = 0::

        TV(P, Q) = H_0(P ‖ Q) = Σ_j max(P_j − Q_j, 0)

    (assuming both sum to 1).

    Args:
        p: Distribution P, shape (k,).
        q: Distribution Q, shape (k,).

    Returns:
        Total variation distance in [0, 1].
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )

    return 0.5 * float(np.sum(np.abs(p - q)))


def _all_divergences(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    epsilon: float,
) -> Dict[str, float]:
    """Compute all divergence measures between two distributions.

    Convenience function for diagnostics and reporting.

    Args:
        p: Distribution P, shape (k,).
        q: Distribution Q, shape (k,).
        epsilon: Privacy parameter ε (for hockey-stick).

    Returns:
        Dict mapping divergence name to value.
    """
    return {
        "hockey_stick": hockey_stick_divergence(p, q, epsilon),
        "kl": kl_divergence(p, q),
        "max": max_divergence(p, q),
        "total_variation": total_variation(p, q),
        "renyi_2": renyi_divergence(p, q, 2.0),
    }


# ============================================================================
# Core Verification Functions
# ============================================================================


def _verify_pure_dp_pair(
    p_i: npt.NDArray[np.float64],
    p_ip: npt.NDArray[np.float64],
    exp_eps: float,
    tol: float,
) -> Tuple[bool, int, float, float]:
    """Check pure DP for a single pair (i, i') in BOTH directions.

    For pure DP, we check that for every output bin j:
        p[i][j] / p[i'][j]  ≤  e^ε  (forward)
        p[i'][j] / p[i][j]  ≤  e^ε  (reverse)

    Uses vectorized operations for speed.

    Args:
        p_i: Row i of the probability table, shape (k,).
        p_ip: Row i' of the probability table, shape (k,).
        exp_eps: Pre-computed e^ε.
        tol: Verification tolerance.

    Returns:
        Tuple of (is_valid, j_worst, worst_ratio, violation_magnitude).
        j_worst and worst_ratio refer to the bin with the largest ratio.
        violation_magnitude is 0 if valid, else ratio - (exp_eps + tol).
    """
    # Floor probabilities to avoid division by zero
    p_i_safe = np.maximum(p_i, _PROB_FLOOR)
    p_ip_safe = np.maximum(p_ip, _PROB_FLOOR)

    # Forward ratios: p[i][j] / p[i'][j]
    ratios_fwd = p_i_safe / p_ip_safe
    # Reverse ratios: p[i'][j] / p[i][j]
    ratios_rev = p_ip_safe / p_i_safe

    # Find worst ratio across both directions
    max_fwd = float(np.max(ratios_fwd))
    max_rev = float(np.max(ratios_rev))

    if max_fwd >= max_rev:
        worst_ratio = max_fwd
        j_worst = int(np.argmax(ratios_fwd))
    else:
        worst_ratio = max_rev
        j_worst = int(np.argmax(ratios_rev))

    threshold = exp_eps + tol
    if worst_ratio > threshold:
        magnitude = worst_ratio - exp_eps
        return False, j_worst, worst_ratio, magnitude
    else:
        return True, j_worst, worst_ratio, 0.0


def _verify_approx_dp_pair(
    p_i: npt.NDArray[np.float64],
    p_ip: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    tol: float,
) -> Tuple[bool, float, float]:
    """Check approximate DP for a single pair (i, i') in BOTH directions.

    For approximate DP, we check the hockey-stick divergence:
        H_ε(p[i] ‖ p[i'])  ≤  δ  (forward)
        H_ε(p[i'] ‖ p[i])  ≤  δ  (reverse)

    CRITICAL: We do NOT check per-bin ratios.  A mechanism can have
    individual ratios exceeding e^ε and still satisfy (ε, δ)-DP.  This
    is fundamental to how approximate DP works — the δ parameter absorbs
    a small probability mass of "bad" events.

    Args:
        p_i: Row i of the probability table, shape (k,).
        p_ip: Row i' of the probability table, shape (k,).
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        tol: Verification tolerance.

    Returns:
        Tuple of (is_valid, worst_hs, violation_magnitude).
        worst_hs is the larger hockey-stick divergence (fwd or rev).
        violation_magnitude is 0 if valid, else worst_hs - (delta + tol).
    """
    hs_fwd = hockey_stick_divergence(p_i, p_ip, epsilon)
    hs_rev = hockey_stick_divergence(p_ip, p_i, epsilon)

    worst_hs = max(hs_fwd, hs_rev)
    threshold = delta + tol

    if worst_hs > threshold:
        magnitude = worst_hs - delta
        return False, worst_hs, magnitude
    else:
        return True, worst_hs, 0.0


def verify(
    p: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    edges: Union[List[Tuple[int, int]], AdjacencyRelation],
    tol: float = _DEFAULT_TOL,
    *,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
    mode: VerificationMode = VerificationMode.MOST_VIOLATING,
) -> VerifyResult:
    """Main verification function for the CEGIS loop.

    This is the primary entry point for DP verification.  Given a mechanism
    probability table and privacy parameters, it checks (ε, δ)-DP over all
    specified adjacent pairs.

    **Pure DP** (δ = 0):
        For each edge (i, i') and each output bin j, computes the ratios
        p[i][j] / p[i'][j] and p[i'][j] / p[i][j].  Tracks the worst
        ratio exceeding e^ε + tol.  Returns the most-violating
        (i, i', j, ratio) tuple.

    **Approximate DP** (δ > 0):
        SKIPS per-bin ratio checking entirely.  For each edge (i, i'),
        computes hockey-stick divergence H_ε(p[i] ‖ p[i']) in both
        directions.  Returns the most-violating pair if the max divergence
        exceeds δ + tol.

    Args:
        p: Mechanism probability table, shape (n, k).
            ``p[i][j] = Pr[M(x_i) = y_j]``.
        epsilon: Privacy parameter ε > 0.
        delta: Privacy parameter δ ≥ 0.  Use 0 for pure DP.
        edges: List of adjacent pairs (i, i'), or an AdjacencyRelation.
            For symmetric adjacency, both directions are checked
            automatically.
        tol: Verification tolerance.  Must satisfy Invariant I4:
            ``tol ≥ exp(ε) × solver_tol``.
        solver_tol: Solver primal tolerance (for I4 warning).
        mode: Verification mode (FAST, MOST_VIOLATING, or EXHAUSTIVE).

    Returns:
        A :class:`VerifyResult` with ``valid=True`` if the mechanism
        satisfies (ε, δ)-DP, or ``valid=False`` with the worst violation.

    Raises:
        InvalidMechanismError: If ``p`` is not a valid probability table.
        ConfigurationError: If parameters are invalid.
    """
    p = np.asarray(p, dtype=np.float64)
    _validate_probability_table(p, name="p")

    n, k = p.shape

    # Parameter validation
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ConfigurationError(
            f"epsilon must be finite and > 0, got {epsilon}",
            parameter="epsilon",
            value=epsilon,
            constraint="epsilon > 0",
        )
    if delta < 0 or delta >= 1.0:
        raise ConfigurationError(
            f"delta must be in [0, 1), got {delta}",
            parameter="delta",
            value=delta,
            constraint="0 <= delta < 1",
        )
    if tol < 0:
        raise ConfigurationError(
            f"tol must be >= 0, got {tol}",
            parameter="tol",
            value=tol,
            constraint="tol >= 0",
        )

    # Extract edge list
    if isinstance(edges, AdjacencyRelation):
        symmetric = edges.symmetric
        edge_list = edges.edges
    else:
        symmetric = True
        edge_list = edges

    directed_edges = _validate_edges(edge_list, n, symmetric=symmetric)

    # Warn on I4 violations
    warn_tolerance_violation(tol, epsilon, solver_tol)

    is_pure = delta == 0.0
    exp_eps = math.exp(epsilon)

    # Track worst violation across all pairs
    worst_violation: Optional[ViolationRecord] = None

    logger.debug(
        "Verifying %s-DP (ε=%.4f, δ=%.2e) over %d directed edges, "
        "tol=%.2e, mode=%s",
        "pure" if is_pure else "approx",
        epsilon,
        delta,
        len(directed_edges),
        tol,
        mode.name,
    )

    if is_pure:
        worst_violation = _verify_pure_dp_all_pairs(
            p, directed_edges, exp_eps, tol, mode
        )
    else:
        worst_violation = _verify_approx_dp_all_pairs(
            p, directed_edges, epsilon, delta, tol, mode
        )

    if worst_violation is None:
        return VerifyResult(valid=True)
    else:
        return VerifyResult(
            valid=False,
            violation=worst_violation.to_tuple(),
        )


def _verify_pure_dp_all_pairs(
    p: npt.NDArray[np.float64],
    directed_edges: List[Tuple[int, int]],
    exp_eps: float,
    tol: float,
    mode: VerificationMode,
) -> Optional[ViolationRecord]:
    """Check pure DP across all pairs, tracking worst violation.

    For pure DP, we check per-bin ratios: for every adjacent pair (i, i')
    and every output bin j, the ratio p[i][j] / p[i'][j] must be ≤ e^ε.

    Args:
        p: Probability table, shape (n, k).
        directed_edges: List of directed (i, i') pairs.
        exp_eps: Pre-computed e^ε.
        tol: Verification tolerance.
        mode: Controls early-exit vs. exhaustive scanning.

    Returns:
        The worst ViolationRecord, or None if DP holds.
    """
    worst: Optional[ViolationRecord] = None
    worst_magnitude = 0.0

    for i, ip in directed_edges:
        valid, j_worst, ratio, magnitude = _verify_pure_dp_pair(
            p[i], p[ip], exp_eps, tol
        )

        if not valid:
            # Determine direction
            p_i_safe = np.maximum(p[i], _PROB_FLOOR)
            p_ip_safe = np.maximum(p[ip], _PROB_FLOOR)
            fwd_ratio = p_i_safe[j_worst] / p_ip_safe[j_worst]
            rev_ratio = p_ip_safe[j_worst] / p_i_safe[j_worst]
            direction = "forward" if fwd_ratio >= rev_ratio else "reverse"

            record = ViolationRecord(
                i=i,
                i_prime=ip,
                j_worst=j_worst,
                magnitude=magnitude,
                ratio=ratio,
                violation_type=ViolationType.PURE_DP_RATIO,
                direction=direction,
            )

            if mode == VerificationMode.FAST:
                return record

            if magnitude > worst_magnitude:
                worst_magnitude = magnitude
                worst = record

    return worst


def _verify_approx_dp_all_pairs(
    p: npt.NDArray[np.float64],
    directed_edges: List[Tuple[int, int]],
    epsilon: float,
    delta: float,
    tol: float,
    mode: VerificationMode,
) -> Optional[ViolationRecord]:
    """Check approximate DP across all pairs via hockey-stick divergence.

    For approximate DP (δ > 0), we do NOT check per-bin ratios.
    Instead, for each adjacent pair we compute the hockey-stick
    divergence H_ε(p[i] ‖ p[i']) and check that it is ≤ δ.

    A mechanism can have individual ratios > e^ε and still satisfy
    (ε, δ)-DP.  Only the aggregate hockey-stick divergence matters.

    Args:
        p: Probability table, shape (n, k).
        directed_edges: List of directed (i, i') pairs.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        tol: Verification tolerance.
        mode: Controls early-exit vs. exhaustive scanning.

    Returns:
        The worst ViolationRecord, or None if DP holds.
    """
    worst: Optional[ViolationRecord] = None
    worst_magnitude = 0.0

    # De-duplicate edges for hockey-stick check: for directed edges
    # (i, i') and (i', i), we compute both H_ε(p[i]‖p[i']) and
    # H_ε(p[i']‖p[i]) but attribute the violation to the pair with
    # the worse direction.
    checked_pairs: set = set()

    for i, ip in directed_edges:
        # Avoid redundant computation for (i,i') and (i',i)
        pair_key = (min(i, ip), max(i, ip))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)

        hs_fwd = hockey_stick_divergence(p[i], p[ip], epsilon)
        hs_rev = hockey_stick_divergence(p[ip], p[i], epsilon)

        if hs_fwd >= hs_rev:
            worst_hs = hs_fwd
            direction = "forward"
            vi, vip = i, ip
        else:
            worst_hs = hs_rev
            direction = "reverse"
            vi, vip = ip, i

        threshold = delta + tol

        if worst_hs > threshold:
            magnitude = worst_hs - delta

            # For approx DP, j_worst = -1 (violation is aggregate)
            record = ViolationRecord(
                i=vi,
                i_prime=vip,
                j_worst=-1,
                magnitude=magnitude,
                ratio=worst_hs,
                violation_type=ViolationType.APPROX_DP_HOCKEY_STICK,
                direction=direction,
            )

            if mode == VerificationMode.FAST:
                return record

            if magnitude > worst_magnitude:
                worst_magnitude = magnitude
                worst = record

    return worst


# ============================================================================
# PrivacyVerifier Class
# ============================================================================


class PrivacyVerifier:
    """Full-featured deterministic DP verifier.

    Provides methods for pure and approximate DP verification, worst-case
    finding, actual privacy parameter computation, and report generation.

    The verifier maintains state about the last verification for
    diagnostic purposes (worst pairs, all violations, timing).

    Args:
        numerical_config: Numerical precision configuration.
            If None, uses sensible defaults.

    Example::

        verifier = PrivacyVerifier()
        result = verifier.verify_mechanism(mechanism, spec)
        if not result.valid:
            print(f"Violation at pair {result.violation_pair}")
            report = verifier.generate_report()
            print(report.summary())
    """

    def __init__(
        self,
        numerical_config: Optional[NumericalConfig] = None,
    ) -> None:
        self._config = numerical_config or NumericalConfig()
        self._last_violations: List[ViolationRecord] = []
        self._last_pair_analyses: List[PairAnalysis] = []
        self._last_epsilon: float = 0.0
        self._last_delta: float = 0.0
        self._last_tol: float = 0.0
        self._last_n_pairs: int = 0
        self._last_time: float = 0.0
        self._last_p: Optional[npt.NDArray[np.float64]] = None
        self._last_edges: Optional[List[Tuple[int, int]]] = None

    @property
    def config(self) -> NumericalConfig:
        """The numerical configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Pure DP verification
    # ------------------------------------------------------------------

    def verify_pure_dp(
        self,
        p: npt.NDArray[np.float64],
        epsilon: float,
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        tol: Optional[float] = None,
        *,
        mode: VerificationMode = VerificationMode.MOST_VIOLATING,
    ) -> VerifyResult:
        """Verify pure ε-DP (δ = 0) over all adjacent pairs.

        For each adjacent pair (i, i') and each output bin j, checks:
            p[i][j] / p[i'][j]  ≤  e^ε + tol
        in both directions.

        The ``MOST_VIOLATING`` mode (default) scans all pairs and returns
        the one with the largest ratio, rather than stopping at the first
        violation.  Theory shows this yields ~40% fewer CEGIS iterations.

        Args:
            p: Mechanism probability table, shape (n, k).
            epsilon: Privacy parameter ε > 0.
            edges: Adjacent pairs or AdjacencyRelation.
            tol: Verification tolerance.  If None, auto-computed from
                solver tolerance via Invariant I4.
            mode: Verification mode.

        Returns:
            VerifyResult with the worst violation, if any.
        """
        if tol is None:
            tol = compute_safe_tolerance(epsilon, self._config.solver_tol)

        start = time.monotonic()
        result = verify(
            p, epsilon, delta=0.0, edges=edges, tol=tol,
            solver_tol=self._config.solver_tol, mode=mode,
        )
        elapsed = time.monotonic() - start

        self._update_state(p, epsilon, 0.0, tol, edges, elapsed, result)
        return result

    # ------------------------------------------------------------------
    # Approximate DP verification
    # ------------------------------------------------------------------

    def verify_approx_dp(
        self,
        p: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        tol: Optional[float] = None,
        *,
        mode: VerificationMode = VerificationMode.MOST_VIOLATING,
    ) -> VerifyResult:
        """Verify (ε, δ)-DP with δ > 0 over all adjacent pairs.

        Uses hockey-stick divergence — does NOT check per-bin ratios.
        A mechanism can have individual bin ratios exceeding e^ε and
        still satisfy (ε, δ)-DP.  The δ parameter absorbs a small
        probability mass of "bad" events.

        For each adjacent pair (i, i'), checks both directions:
            H_ε(p[i] ‖ p[i'])  ≤  δ + tol
            H_ε(p[i'] ‖ p[i])  ≤  δ + tol

        Args:
            p: Mechanism probability table, shape (n, k).
            epsilon: Privacy parameter ε > 0.
            delta: Privacy parameter δ > 0.
            edges: Adjacent pairs or AdjacencyRelation.
            tol: Verification tolerance.  If None, auto-computed.
            mode: Verification mode.

        Returns:
            VerifyResult with the worst violation, if any.
        """
        if delta <= 0:
            raise ConfigurationError(
                "delta must be > 0 for approximate DP verification; "
                "use verify_pure_dp for delta=0",
                parameter="delta",
                value=delta,
                constraint="delta > 0",
            )

        if tol is None:
            tol = compute_safe_tolerance(epsilon, self._config.solver_tol)

        start = time.monotonic()
        result = verify(
            p, epsilon, delta=delta, edges=edges, tol=tol,
            solver_tol=self._config.solver_tol, mode=mode,
        )
        elapsed = time.monotonic() - start

        self._update_state(p, epsilon, delta, tol, edges, elapsed, result)
        return result

    # ------------------------------------------------------------------
    # High-level verification
    # ------------------------------------------------------------------

    def verify_mechanism(
        self,
        mechanism: Union[ExtractedMechanism, npt.NDArray[np.float64]],
        spec: Union[QuerySpec, PrivacyBudget],
        edges: Optional[Union[List[Tuple[int, int]], AdjacencyRelation]] = None,
        tol: Optional[float] = None,
    ) -> VerifyResult:
        """High-level verification of a mechanism against a specification.

        Dispatches to :meth:`verify_pure_dp` or :meth:`verify_approx_dp`
        based on the specification's δ value.

        Args:
            mechanism: An ExtractedMechanism or raw probability table.
            spec: A QuerySpec or PrivacyBudget defining the target privacy.
            edges: Adjacent pairs.  If None and spec is a QuerySpec, uses
                the spec's adjacency relation.
            tol: Verification tolerance.  If None, auto-computed.

        Returns:
            VerifyResult.

        Raises:
            ConfigurationError: If edges cannot be determined.
        """
        # Extract probability table
        if isinstance(mechanism, ExtractedMechanism):
            p = mechanism.p_final
        else:
            p = np.asarray(mechanism, dtype=np.float64)

        # Extract privacy parameters
        if isinstance(spec, QuerySpec):
            epsilon = spec.epsilon
            delta = spec.delta
            if edges is None:
                edges = spec.edges
        elif isinstance(spec, PrivacyBudget):
            epsilon = spec.epsilon
            delta = spec.delta
        else:
            raise ConfigurationError(
                f"spec must be QuerySpec or PrivacyBudget, got {type(spec).__name__}",
                parameter="spec",
                value=type(spec).__name__,
            )

        if edges is None:
            n = p.shape[0]
            edges = AdjacencyRelation.hamming_distance_1(n)

        if delta == 0.0:
            return self.verify_pure_dp(p, epsilon, edges, tol)
        else:
            return self.verify_approx_dp(p, epsilon, delta, edges, tol)

    # ------------------------------------------------------------------
    # Worst-case finding
    # ------------------------------------------------------------------

    def find_most_violating_pair(
        self,
        p: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        tol: float = 0.0,
    ) -> Optional[ViolationRecord]:
        """Find the adjacent pair with the maximum privacy violation.

        Scans ALL pairs and returns the one with the largest violation
        magnitude, not just the first-found violation.  Theory shows
        this gives ~40% fewer CEGIS iterations compared to first-found.

        Args:
            p: Mechanism probability table, shape (n, k).
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            edges: Adjacent pairs.
            tol: Only report violations exceeding tol.

        Returns:
            The ViolationRecord with maximum magnitude, or None.
        """
        p = np.asarray(p, dtype=np.float64)
        _validate_probability_table(p)
        n, k = p.shape

        if isinstance(edges, AdjacencyRelation):
            symmetric = edges.symmetric
            edge_list = edges.edges
        else:
            symmetric = True
            edge_list = edges

        directed_edges = _validate_edges(edge_list, n, symmetric=symmetric)
        is_pure = delta == 0.0
        exp_eps = math.exp(epsilon)

        worst: Optional[ViolationRecord] = None
        worst_magnitude = 0.0

        if is_pure:
            for i, ip in directed_edges:
                valid, j_worst, ratio, magnitude = _verify_pure_dp_pair(
                    p[i], p[ip], exp_eps, tol
                )
                if not valid and magnitude > worst_magnitude:
                    p_i_safe = np.maximum(p[i], _PROB_FLOOR)
                    p_ip_safe = np.maximum(p[ip], _PROB_FLOOR)
                    fwd_r = p_i_safe[j_worst] / p_ip_safe[j_worst]
                    rev_r = p_ip_safe[j_worst] / p_i_safe[j_worst]
                    direction = "forward" if fwd_r >= rev_r else "reverse"
                    worst = ViolationRecord(
                        i=i, i_prime=ip, j_worst=j_worst,
                        magnitude=magnitude, ratio=ratio,
                        violation_type=ViolationType.PURE_DP_RATIO,
                        direction=direction,
                    )
                    worst_magnitude = magnitude
        else:
            checked = set()
            for i, ip in directed_edges:
                pair_key = (min(i, ip), max(i, ip))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                hs_fwd = hockey_stick_divergence(p[i], p[ip], epsilon)
                hs_rev = hockey_stick_divergence(p[ip], p[i], epsilon)

                if hs_fwd >= hs_rev:
                    worst_hs = hs_fwd
                    direction = "forward"
                    vi, vip = i, ip
                else:
                    worst_hs = hs_rev
                    direction = "reverse"
                    vi, vip = ip, i

                if worst_hs > delta + tol:
                    magnitude = worst_hs - delta
                    if magnitude > worst_magnitude:
                        worst = ViolationRecord(
                            i=vi, i_prime=vip, j_worst=-1,
                            magnitude=magnitude, ratio=worst_hs,
                            violation_type=ViolationType.APPROX_DP_HOCKEY_STICK,
                            direction=direction,
                        )
                        worst_magnitude = magnitude

        return worst

    def find_all_violations(
        self,
        p: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        tol: float = 0.0,
        max_violations: int = _MAX_VIOLATIONS_DETAIL,
    ) -> List[ViolationRecord]:
        """Find ALL violating pairs, sorted by magnitude (descending).

        Args:
            p: Mechanism probability table, shape (n, k).
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            edges: Adjacent pairs.
            tol: Only report violations exceeding tol.
            max_violations: Maximum number of violations to return.

        Returns:
            List of ViolationRecords sorted by descending magnitude.
        """
        p = np.asarray(p, dtype=np.float64)
        _validate_probability_table(p)
        n, k = p.shape

        if isinstance(edges, AdjacencyRelation):
            symmetric = edges.symmetric
            edge_list = edges.edges
        else:
            symmetric = True
            edge_list = edges

        directed_edges = _validate_edges(edge_list, n, symmetric=symmetric)
        is_pure = delta == 0.0
        exp_eps = math.exp(epsilon)

        violations: List[ViolationRecord] = []

        if is_pure:
            for i, ip in directed_edges:
                valid, j_worst, ratio, magnitude = _verify_pure_dp_pair(
                    p[i], p[ip], exp_eps, tol
                )
                if not valid:
                    p_i_safe = np.maximum(p[i], _PROB_FLOOR)
                    p_ip_safe = np.maximum(p[ip], _PROB_FLOOR)
                    fwd_r = p_i_safe[j_worst] / p_ip_safe[j_worst]
                    rev_r = p_ip_safe[j_worst] / p_i_safe[j_worst]
                    direction = "forward" if fwd_r >= rev_r else "reverse"
                    violations.append(ViolationRecord(
                        i=i, i_prime=ip, j_worst=j_worst,
                        magnitude=magnitude, ratio=ratio,
                        violation_type=ViolationType.PURE_DP_RATIO,
                        direction=direction,
                    ))
                    if len(violations) >= max_violations:
                        break
        else:
            checked = set()
            for i, ip in directed_edges:
                pair_key = (min(i, ip), max(i, ip))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                hs_fwd = hockey_stick_divergence(p[i], p[ip], epsilon)
                hs_rev = hockey_stick_divergence(p[ip], p[i], epsilon)

                if hs_fwd >= hs_rev:
                    worst_hs = hs_fwd
                    direction = "forward"
                    vi, vip = i, ip
                else:
                    worst_hs = hs_rev
                    direction = "reverse"
                    vi, vip = ip, i

                if worst_hs > delta + tol:
                    magnitude = worst_hs - delta
                    violations.append(ViolationRecord(
                        i=vi, i_prime=vip, j_worst=-1,
                        magnitude=magnitude, ratio=worst_hs,
                        violation_type=ViolationType.APPROX_DP_HOCKEY_STICK,
                        direction=direction,
                    ))
                    if len(violations) >= max_violations:
                        break

        # Sort by descending magnitude
        violations.sort(key=lambda v: v.magnitude, reverse=True)
        return violations

    # ------------------------------------------------------------------
    # Actual privacy parameter computation
    # ------------------------------------------------------------------

    def compute_actual_epsilon(
        self,
        p: npt.NDArray[np.float64],
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
    ) -> float:
        """Compute the tightest ε the mechanism satisfies for pure DP.

        This is the max-divergence D_∞ over all adjacent pairs:
            ε* = max_{(i,i')∈E} D_∞(p[i] ‖ p[i'])

        For approximate DP, this gives the ε such that (ε, 0)-DP holds,
        which is always at least as large as the target ε for (ε, δ)-DP.

        Args:
            p: Mechanism probability table, shape (n, k).
            edges: Adjacent pairs or AdjacencyRelation.

        Returns:
            The smallest ε such that the mechanism satisfies ε-DP.
        """
        p = np.asarray(p, dtype=np.float64)
        _validate_probability_table(p)
        n, k = p.shape

        if isinstance(edges, AdjacencyRelation):
            symmetric = edges.symmetric
            edge_list = edges.edges
        else:
            symmetric = True
            edge_list = edges

        directed_edges = _validate_edges(edge_list, n, symmetric=symmetric)

        max_eps = 0.0
        for i, ip in directed_edges:
            d_inf = max_divergence(p[i], p[ip])
            if d_inf > max_eps:
                max_eps = d_inf

        logger.debug("Actual epsilon: %.6f (over %d edges)", max_eps, len(directed_edges))
        return max_eps

    def compute_actual_delta(
        self,
        p: npt.NDArray[np.float64],
        epsilon: float,
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
    ) -> float:
        """Compute the tightest δ at a given ε for approximate DP.

        This is the maximum hockey-stick divergence over all adjacent
        pairs:
            δ* = max_{(i,i')∈E} H_ε(p[i] ‖ p[i'])

        Args:
            p: Mechanism probability table, shape (n, k).
            epsilon: Privacy parameter ε.
            edges: Adjacent pairs or AdjacencyRelation.

        Returns:
            The smallest δ such that the mechanism satisfies (ε, δ)-DP.
        """
        p = np.asarray(p, dtype=np.float64)
        _validate_probability_table(p)
        n, k = p.shape

        if isinstance(edges, AdjacencyRelation):
            symmetric = edges.symmetric
            edge_list = edges.edges
        else:
            symmetric = True
            edge_list = edges

        directed_edges = _validate_edges(edge_list, n, symmetric=symmetric)

        max_delta = 0.0
        for i, ip in directed_edges:
            hs = hockey_stick_divergence(p[i], p[ip], epsilon)
            if hs > max_delta:
                max_delta = hs

        logger.debug(
            "Actual delta at eps=%.4f: %.2e (over %d edges)",
            epsilon, max_delta, len(directed_edges),
        )
        return max_delta

    def compute_privacy_curve(
        self,
        p: npt.NDArray[np.float64],
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        epsilon_range: Optional[npt.NDArray[np.float64]] = None,
        n_points: int = 100,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the (ε, δ) trade-off curve for the mechanism.

        For each ε in the range, computes the tightest δ such that the
        mechanism satisfies (ε, δ)-DP.

        Args:
            p: Mechanism probability table, shape (n, k).
            edges: Adjacent pairs.
            epsilon_range: Array of ε values to evaluate.  If None, uses
                a logarithmic range from 0.01 to 10.
            n_points: Number of points if epsilon_range is None.

        Returns:
            Tuple of (epsilon_values, delta_values) arrays.
        """
        if epsilon_range is None:
            epsilon_range = np.logspace(-2, 1, n_points)
        else:
            epsilon_range = np.asarray(epsilon_range, dtype=np.float64)

        delta_values = np.empty_like(epsilon_range)
        for idx, eps in enumerate(epsilon_range):
            delta_values[idx] = self.compute_actual_delta(p, float(eps), edges)

        return epsilon_range, delta_values

    # ------------------------------------------------------------------
    # Per-pair analysis
    # ------------------------------------------------------------------

    def analyze_pair(
        self,
        p: npt.NDArray[np.float64],
        i: int,
        i_prime: int,
        epsilon: float,
        delta: float = 0.0,
        tol: float = 0.0,
    ) -> PairAnalysis:
        """Perform detailed analysis of a single adjacent pair.

        Computes max ratios (both directions), hockey-stick divergences
        (both directions), and determines whether the pair violates DP.

        Args:
            p: Mechanism probability table, shape (n, k).
            i: First database index.
            i_prime: Second database index.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            tol: Verification tolerance.

        Returns:
            PairAnalysis with full diagnostic information.
        """
        p = np.asarray(p, dtype=np.float64)
        n, k = p.shape

        p_i = p[i]
        p_ip = p[i_prime]

        # Floor for ratio computation
        p_i_safe = np.maximum(p_i, _PROB_FLOOR)
        p_ip_safe = np.maximum(p_ip, _PROB_FLOOR)

        # Max ratios
        ratios_fwd = p_i_safe / p_ip_safe
        ratios_rev = p_ip_safe / p_i_safe

        max_ratio_fwd = float(np.max(ratios_fwd))
        max_ratio_rev = float(np.max(ratios_rev))
        j_worst_fwd = int(np.argmax(ratios_fwd))
        j_worst_rev = int(np.argmax(ratios_rev))

        # Hockey-stick divergences
        hs_fwd = hockey_stick_divergence(p_i, p_ip, epsilon)
        hs_rev = hockey_stick_divergence(p_ip, p_i, epsilon)

        # Determine violation
        exp_eps = math.exp(epsilon)
        is_pure = delta == 0.0

        if is_pure:
            is_violating = (
                max(max_ratio_fwd, max_ratio_rev) > exp_eps + tol
            )
        else:
            is_violating = max(hs_fwd, hs_rev) > delta + tol

        return PairAnalysis(
            i=i,
            i_prime=i_prime,
            max_ratio_forward=max_ratio_fwd,
            max_ratio_reverse=max_ratio_rev,
            j_worst_forward=j_worst_fwd,
            j_worst_reverse=j_worst_rev,
            hockey_stick_forward=hs_fwd,
            hockey_stick_reverse=hs_rev,
            is_violating=is_violating,
        )

    def analyze_all_pairs(
        self,
        p: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        tol: float = 0.0,
    ) -> List[PairAnalysis]:
        """Analyze all adjacent pairs and return per-pair diagnostics.

        Args:
            p: Mechanism probability table, shape (n, k).
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            edges: Adjacent pairs.
            tol: Verification tolerance.

        Returns:
            List of PairAnalysis objects, one per undirected pair.
        """
        p = np.asarray(p, dtype=np.float64)
        n, k = p.shape

        if isinstance(edges, AdjacencyRelation):
            edge_list = edges.edges
        else:
            edge_list = edges

        # Analyze each undirected pair once
        seen = set()
        analyses = []
        for i, ip in edge_list:
            pair_key = (min(i, ip), max(i, ip))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            analyses.append(self.analyze_pair(p, i, ip, epsilon, delta, tol))

        return analyses

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        p: Optional[npt.NDArray[np.float64]] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        edges: Optional[Union[List[Tuple[int, int]], AdjacencyRelation]] = None,
        tol: Optional[float] = None,
        compute_actual: bool = True,
    ) -> VerificationReport:
        """Generate a comprehensive verification report.

        If arguments are not provided, uses values from the last
        verification call.

        Args:
            p: Mechanism probability table.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            edges: Adjacent pairs.
            tol: Verification tolerance.
            compute_actual: Whether to compute actual ε and δ values.

        Returns:
            VerificationReport with full analysis.
        """
        # Use cached values if not provided
        if p is None:
            p = self._last_p
        if epsilon is None:
            epsilon = self._last_epsilon
        if delta is None:
            delta = self._last_delta
        if tol is None:
            tol = self._last_tol if self._last_tol > 0 else _DEFAULT_TOL
        if edges is None:
            edges = self._last_edges

        if p is None or edges is None:
            raise ConfigurationError(
                "No cached verification state.  Provide p, epsilon, delta, "
                "edges explicitly, or call verify_* first.",
                parameter="p",
            )

        start = time.monotonic()

        # Full verification
        violations = self.find_all_violations(p, epsilon, delta, edges, tol=0.0)
        pair_analyses = self.analyze_all_pairs(p, epsilon, delta, edges, tol)

        n_pairs = len(pair_analyses)
        n_violations = len(violations)
        is_valid = n_violations == 0
        worst = violations[0] if violations else None

        # Actual privacy parameters
        actual_eps = None
        actual_delta = None
        if compute_actual:
            actual_eps = self.compute_actual_epsilon(p, edges)
            actual_delta = self.compute_actual_delta(p, epsilon, edges)

        elapsed = time.monotonic() - start

        # Generate recommendations
        recommendations = self._generate_recommendations(
            is_valid, epsilon, delta, actual_eps, actual_delta, violations,
        )

        return VerificationReport(
            is_valid=is_valid,
            epsilon=epsilon,
            delta=delta,
            tolerance=tol,
            n_pairs_checked=n_pairs,
            n_violations=n_violations,
            worst_violation=worst,
            all_violations=violations,
            pair_analyses=pair_analyses,
            actual_epsilon=actual_eps,
            actual_delta=actual_delta,
            verification_time_s=elapsed,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        is_valid: bool,
        epsilon: float,
        delta: float,
        actual_eps: Optional[float],
        actual_delta: Optional[float],
        violations: List[ViolationRecord],
    ) -> List[str]:
        """Generate human-readable recommendations based on verification."""
        recs: List[str] = []

        if is_valid:
            if actual_eps is not None:
                slack = epsilon - actual_eps
                if slack > 0.5 * epsilon:
                    recs.append(
                        f"Mechanism has significant ε slack ({slack:.4f}). "
                        f"Consider tightening ε from {epsilon:.4f} to "
                        f"{actual_eps:.4f} for better utility."
                    )
            if actual_delta is not None and delta > 0:
                if actual_delta < 0.1 * delta:
                    recs.append(
                        f"Actual δ ({actual_delta:.2e}) is much smaller than "
                        f"target ({delta:.2e}).  The mechanism may be "
                        f"over-privatized."
                    )
        else:
            if violations:
                worst = violations[0]
                if worst.violation_type == ViolationType.PURE_DP_RATIO:
                    recs.append(
                        f"Worst pure DP violation at pair ({worst.i}, "
                        f"{worst.i_prime}), bin {worst.j_worst}: ratio "
                        f"{worst.ratio:.4f} vs bound e^ε = {math.exp(epsilon):.4f}."
                    )
                    if actual_eps is not None:
                        recs.append(
                            f"Mechanism actually satisfies ε = {actual_eps:.4f}. "
                            f"Increase target ε to at least {actual_eps:.4f}, or "
                            f"re-run CEGIS with this counterexample."
                        )
                else:
                    recs.append(
                        f"Worst hockey-stick violation at pair ({worst.i}, "
                        f"{worst.i_prime}): H_ε = {worst.ratio:.2e} vs "
                        f"bound δ = {delta:.2e}."
                    )
                    if actual_delta is not None:
                        recs.append(
                            f"Mechanism actually satisfies δ = {actual_delta:.2e} "
                            f"at ε = {epsilon:.4f}.  Increase target δ or re-run "
                            f"CEGIS."
                        )

            if len(violations) > 10:
                recs.append(
                    f"Found {len(violations)} violating pairs — mechanism may "
                    f"need full re-synthesis rather than incremental CEGIS."
                )

        return recs

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _update_state(
        self,
        p: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
        tol: float,
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        elapsed: float,
        result: VerifyResult,
    ) -> None:
        """Update internal state after a verification call."""
        self._last_p = p
        self._last_epsilon = epsilon
        self._last_delta = delta
        self._last_tol = tol
        self._last_time = elapsed

        if isinstance(edges, AdjacencyRelation):
            self._last_edges = edges
            self._last_n_pairs = len(edges.edges)
        else:
            self._last_edges = edges
            self._last_n_pairs = len(edges)


# ============================================================================
# MonteCarloVerifier Class
# ============================================================================


class MonteCarloVerifier:
    """Statistical DP verification via Monte Carlo sampling.

    Provides empirical privacy auditing by drawing samples from the
    mechanism and estimating privacy parameters statistically.  This
    complements the deterministic :class:`PrivacyVerifier` with
    practical tests that scale to continuous mechanisms.

    Args:
        seed: Random seed for reproducibility.

    Example::

        auditor = MonteCarloVerifier(seed=42)
        result = auditor.audit_dp(mechanism, n_samples=10000)
        print(f"Empirical ε ≤ {result['epsilon_upper_bound']:.4f}")
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def audit_dp(
        self,
        p: npt.NDArray[np.float64],
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        epsilon: float,
        delta: float = 0.0,
        n_samples: int = 10_000,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Run a statistical privacy audit.

        Samples from each row of the mechanism and estimates the empirical
        privacy parameters.  Returns a dict with audit results including
        upper bounds on ε and δ at the specified confidence level.

        The audit constructs empirical distributions from samples and
        computes divergences on the histograms.  This is a lower bound
        on the true privacy loss (adversary's best attack is at least
        as powerful), so finding a violation here is a definitive failure.

        Args:
            p: Mechanism probability table, shape (n, k).
            edges: Adjacent pairs.
            epsilon: Target ε for comparison.
            delta: Target δ for comparison.
            n_samples: Number of samples per row.
            confidence: Confidence level for upper bounds.

        Returns:
            Dict with keys:
            - ``'pass'``: Whether the audit passed.
            - ``'epsilon_empirical'``: Estimated ε from samples.
            - ``'epsilon_upper_bound'``: Upper confidence bound on ε.
            - ``'delta_empirical'``: Estimated δ at target ε.
            - ``'n_samples'``: Samples used.
            - ``'worst_pair'``: Pair with worst empirical violation.
        """
        p = np.asarray(p, dtype=np.float64)
        _validate_probability_table(p)
        n, k = p.shape

        if isinstance(edges, AdjacencyRelation):
            edge_list = edges.edges
            symmetric = edges.symmetric
        else:
            edge_list = edges
            symmetric = True

        directed_edges = _validate_edges(edge_list, n, symmetric=symmetric)

        # Sample from each row
        samples = self._sample_mechanism(p, n_samples)

        # Build empirical distributions
        emp_distributions = np.zeros_like(p)
        for i in range(n):
            counts = np.bincount(samples[i], minlength=k)
            emp_distributions[i] = counts / n_samples

        # Compute empirical divergences
        max_eps_empirical = 0.0
        max_delta_empirical = 0.0
        worst_pair = (0, 0)

        checked = set()
        for i, ip in directed_edges:
            pair_key = (min(i, ip), max(i, ip))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            emp_i = emp_distributions[i]
            emp_ip = emp_distributions[ip]

            # Empirical max divergence (ε estimate)
            d_inf_fwd = max_divergence(emp_i, emp_ip)
            d_inf_rev = max_divergence(emp_ip, emp_i)
            d_inf = max(d_inf_fwd, d_inf_rev)

            if d_inf > max_eps_empirical:
                max_eps_empirical = d_inf
                worst_pair = (i, ip)

            # Empirical hockey-stick (δ estimate)
            hs_fwd = hockey_stick_divergence(emp_i, emp_ip, epsilon)
            hs_rev = hockey_stick_divergence(emp_ip, emp_i, epsilon)
            hs = max(hs_fwd, hs_rev)

            if hs > max_delta_empirical:
                max_delta_empirical = hs

        # Confidence interval (Hoeffding-style bound)
        # For n_samples samples, the deviation is O(1/sqrt(n_samples))
        hoeffding_bound = math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2 * n_samples))
        eps_upper = max_eps_empirical + hoeffding_bound

        is_pure = delta == 0.0
        if is_pure:
            passed = max_eps_empirical <= epsilon + hoeffding_bound
        else:
            passed = max_delta_empirical <= delta + hoeffding_bound

        return {
            "pass": passed,
            "epsilon_empirical": max_eps_empirical,
            "epsilon_upper_bound": eps_upper,
            "delta_empirical": max_delta_empirical,
            "n_samples": n_samples,
            "confidence": confidence,
            "worst_pair": worst_pair,
            "hoeffding_bound": hoeffding_bound,
        }

    def estimate_epsilon_empirical(
        self,
        p: npt.NDArray[np.float64],
        edges: Union[List[Tuple[int, int]], AdjacencyRelation],
        n_samples: int = 100_000,
    ) -> float:
        """Estimate the actual ε from samples (lower bound on true ε).

        Draws samples from each mechanism row, constructs empirical
        distributions, and computes the max-divergence.

        Note: this is a lower bound — the true ε is at least this large.
        Use :meth:`PrivacyVerifier.compute_actual_epsilon` for the exact
        value when the mechanism table is available.

        Args:
            p: Mechanism probability table, shape (n, k).
            edges: Adjacent pairs.
            n_samples: Samples per row.

        Returns:
            Estimated ε (lower bound).
        """
        p = np.asarray(p, dtype=np.float64)
        _validate_probability_table(p)
        n, k = p.shape

        if isinstance(edges, AdjacencyRelation):
            edge_list = edges.edges
            symmetric = edges.symmetric
        else:
            edge_list = edges
            symmetric = True

        directed_edges = _validate_edges(edge_list, n, symmetric=symmetric)

        samples = self._sample_mechanism(p, n_samples)

        emp = np.zeros_like(p)
        for i in range(n):
            counts = np.bincount(samples[i], minlength=k)
            emp[i] = counts / n_samples

        max_eps = 0.0
        for i, ip in directed_edges:
            d_fwd = max_divergence(emp[i], emp[ip])
            d_rev = max_divergence(emp[ip], emp[i])
            max_eps = max(max_eps, d_fwd, d_rev)

        return max_eps

    def membership_inference_test(
        self,
        p: npt.NDArray[np.float64],
        i: int,
        i_prime: int,
        n_samples: int = 10_000,
    ) -> Dict[str, Any]:
        """Run a membership inference attack between two adjacent databases.

        Simulates an adversary who observes mechanism outputs and tries
        to distinguish whether the input was x_i or x_{i'}.  The
        adversary uses the likelihood ratio test, which is the optimal
        distinguisher.

        Args:
            p: Mechanism probability table, shape (n, k).
            i: First database index.
            i_prime: Second database index.
            n_samples: Number of test samples.

        Returns:
            Dict with keys:
            - ``'accuracy'``: Adversary's classification accuracy.
            - ``'advantage'``: Adversary's advantage over random guessing.
            - ``'auc'``: Area under the ROC curve.
            - ``'empirical_epsilon'``: Implied ε from the advantage.
        """
        p = np.asarray(p, dtype=np.float64)
        _validate_probability_table(p)
        n, k = p.shape

        if not (0 <= i < n and 0 <= i_prime < n):
            raise ConfigurationError(
                f"Indices ({i}, {i_prime}) out of range for n={n}",
                parameter="i, i_prime",
            )

        # Sample from both distributions
        n_half = n_samples // 2
        samples_i = self._rng.choice(k, size=n_half, p=p[i])
        samples_ip = self._rng.choice(k, size=n_half, p=p[i_prime])

        # Likelihood ratio test
        p_i_safe = np.maximum(p[i], _PROB_FLOOR)
        p_ip_safe = np.maximum(p[i_prime], _PROB_FLOOR)
        log_ratio = np.log(p_i_safe) - np.log(p_ip_safe)

        # Compute log-likelihood ratios for all samples
        lr_i = log_ratio[samples_i]  # Should be positive on average
        lr_ip = log_ratio[samples_ip]  # Should be negative on average

        # Adversary predicts i if log-ratio > 0
        correct_i = np.sum(lr_i > 0)
        correct_ip = np.sum(lr_ip <= 0)
        accuracy = (correct_i + correct_ip) / n_samples
        advantage = 2.0 * accuracy - 1.0  # Advantage over random guessing

        # Compute AUC via the Wilcoxon-Mann-Whitney statistic
        # (fraction of pairs where lr_i > lr_ip)
        all_lr = np.concatenate([lr_i, lr_ip])
        all_labels = np.concatenate([np.ones(n_half), np.zeros(n_half)])
        sorted_idx = np.argsort(all_lr)
        sorted_labels = all_labels[sorted_idx]
        # Rank-sum approach
        positive_ranks = np.where(sorted_labels == 1)[0]
        rank_sum = np.sum(positive_ranks) + n_half  # 1-based adjustment
        auc = (rank_sum - n_half * (n_half + 1) / 2) / (n_half * n_half)

        # Implied epsilon from advantage: advantage ≤ e^ε - 1 (approx)
        if advantage > 0:
            empirical_eps = math.log(1.0 + advantage)
        else:
            empirical_eps = 0.0

        return {
            "accuracy": accuracy,
            "advantage": advantage,
            "auc": auc,
            "empirical_epsilon": empirical_eps,
            "n_samples": n_samples,
        }

    def _sample_mechanism(
        self,
        p: npt.NDArray[np.float64],
        n_samples: int,
    ) -> npt.NDArray[np.int64]:
        """Draw samples from each row of the mechanism.

        Args:
            p: Probability table, shape (n, k).
            n_samples: Samples per row.

        Returns:
            Array of shape (n, n_samples) with bin indices.
        """
        n, k = p.shape
        samples = np.empty((n, n_samples), dtype=np.int64)

        for i in range(n):
            row = p[i]
            # Ensure exact stochasticity for sampling
            row_safe = np.maximum(row, 0.0)
            row_safe = row_safe / row_safe.sum()
            samples[i] = self._rng.choice(k, size=n_samples, p=row_safe)

        return samples


# ============================================================================
# Vectorized Batch Verification
# ============================================================================


def verify_batch_pure_dp(
    p: npt.NDArray[np.float64],
    edges: npt.NDArray[np.int64],
    exp_eps: float,
    tol: float,
) -> Tuple[bool, Optional[Tuple[int, int, int, float]]]:
    """Vectorized pure DP verification for batch edges.

    Processes all edges simultaneously using numpy broadcasting for
    maximum throughput on large mechanisms.

    Args:
        p: Probability table, shape (n, k).
        edges: Edge array, shape (m, 2), dtype int64.
        exp_eps: Pre-computed e^ε.
        tol: Verification tolerance.

    Returns:
        Tuple of (is_valid, violation) where violation is
        (i, i', j_worst, magnitude) or None.
    """
    p = np.asarray(p, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be shape (m, 2), got {edges.shape}")

    m = edges.shape[0]
    if m == 0:
        return True, None

    # Extract rows for all edges: shape (m, k)
    p_i = p[edges[:, 0]]
    p_ip = p[edges[:, 1]]

    # Floor to avoid division by zero
    p_i_safe = np.maximum(p_i, _PROB_FLOOR)
    p_ip_safe = np.maximum(p_ip, _PROB_FLOOR)

    # Compute ratios: shape (m, k)
    ratios = p_i_safe / p_ip_safe

    # Max ratio per edge: shape (m,)
    max_ratios = np.max(ratios, axis=1)
    j_worst_per_edge = np.argmax(ratios, axis=1)

    threshold = exp_eps + tol

    # Find violations
    violating_mask = max_ratios > threshold
    if not np.any(violating_mask):
        return True, None

    # Find worst violation
    violating_indices = np.where(violating_mask)[0]
    worst_idx = violating_indices[np.argmax(max_ratios[violating_mask])]

    i = int(edges[worst_idx, 0])
    ip = int(edges[worst_idx, 1])
    j = int(j_worst_per_edge[worst_idx])
    magnitude = float(max_ratios[worst_idx]) - exp_eps

    return False, (i, ip, j, magnitude)


def verify_batch_approx_dp(
    p: npt.NDArray[np.float64],
    edges: npt.NDArray[np.int64],
    epsilon: float,
    delta: float,
    tol: float,
) -> Tuple[bool, Optional[Tuple[int, int, int, float]]]:
    """Vectorized approximate DP verification for batch edges.

    Computes hockey-stick divergences for all edges simultaneously.

    Args:
        p: Probability table, shape (n, k).
        edges: Edge array, shape (m, 2), dtype int64.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        tol: Verification tolerance.

    Returns:
        Tuple of (is_valid, violation) where violation is
        (i, i', -1, magnitude) or None.
    """
    p = np.asarray(p, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be shape (m, 2), got {edges.shape}")

    m = edges.shape[0]
    if m == 0:
        return True, None

    exp_eps = math.exp(epsilon)

    # Extract rows: shape (m, k)
    p_i = p[edges[:, 0]]
    p_ip = p[edges[:, 1]]

    # Hockey-stick: sum_j max(p_i_j - e^eps * p_ip_j, 0) per edge
    diff = p_i - exp_eps * p_ip
    hs_values = np.sum(np.maximum(diff, 0.0), axis=1)

    threshold = delta + tol

    violating_mask = hs_values > threshold
    if not np.any(violating_mask):
        return True, None

    # Worst violation
    violating_indices = np.where(violating_mask)[0]
    worst_idx = violating_indices[np.argmax(hs_values[violating_mask])]

    i = int(edges[worst_idx, 0])
    ip = int(edges[worst_idx, 1])
    magnitude = float(hs_values[worst_idx]) - delta

    return False, (i, ip, -1, magnitude)


# ============================================================================
# Utility: CEGIS Integration Helpers
# ============================================================================


def verify_for_cegis(
    p: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    adjacency: AdjacencyRelation,
    tol: float = _DEFAULT_TOL,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
) -> VerifyResult:
    """Convenience wrapper for CEGIS loop integration.

    Uses MOST_VIOLATING mode to minimize CEGIS iterations (~40% fewer
    iterations compared to FAST mode, per theory).

    Args:
        p: Mechanism probability table.
        epsilon: Target ε.
        delta: Target δ.
        adjacency: Adjacency relation.
        tol: Verification tolerance.
        solver_tol: Solver primal tolerance.

    Returns:
        VerifyResult.
    """
    return verify(
        p, epsilon, delta, adjacency,
        tol=tol,
        solver_tol=solver_tol,
        mode=VerificationMode.MOST_VIOLATING,
    )


def counterexample_from_result(
    result: VerifyResult,
) -> Optional[Tuple[int, int]]:
    """Extract the counterexample pair from a VerifyResult.

    Returns None if the mechanism is valid (no counterexample).

    Args:
        result: Verification result.

    Returns:
        Tuple (i, i') or None.
    """
    if result.valid:
        return None
    assert result.violation is not None
    return (result.violation[0], result.violation[1])


def check_invariant_i4(
    epsilon: float,
    dp_tol: float,
    solver_tol: float,
) -> Tuple[bool, float, float]:
    """Check Invariant I4 and return diagnostic info.

    I4 requires: dp_tol >= exp(ε) × solver_tol

    Args:
        epsilon: Privacy parameter ε.
        dp_tol: DP verification tolerance.
        solver_tol: Solver primal tolerance.

    Returns:
        Tuple of (satisfies_i4, required_tol, ratio).
        ratio = dp_tol / required_tol (>= 1 if I4 satisfied).
    """
    required = math.exp(epsilon) * solver_tol
    ratio = dp_tol / required if required > 0 else float("inf")
    return dp_tol >= required, required, ratio


# ============================================================================
# Module-level convenience
# ============================================================================


def quick_verify(
    p: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    *,
    tol: float = _DEFAULT_TOL,
) -> bool:
    """Quick boolean check: does the mechanism satisfy (ε, δ)-DP?

    Uses consecutive Hamming-1 adjacency and FAST mode for speed.

    Args:
        p: Mechanism probability table, shape (n, k).
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ (default 0 for pure DP).
        tol: Verification tolerance.

    Returns:
        True if the mechanism satisfies (ε, δ)-DP within tolerance.
    """
    p = np.asarray(p, dtype=np.float64)
    n = p.shape[0]
    adjacency = AdjacencyRelation.hamming_distance_1(n)
    result = verify(
        p, epsilon, delta, adjacency,
        tol=tol,
        mode=VerificationMode.FAST,
    )
    return result.valid


def verify_extracted_mechanism(
    mechanism: ExtractedMechanism,
    spec: QuerySpec,
    tol: Optional[float] = None,
) -> VerifyResult:
    """Verify an ExtractedMechanism against its QuerySpec.

    Convenience function that extracts parameters from the spec and
    calls :func:`verify`.

    Args:
        mechanism: The extracted mechanism.
        spec: The query specification.
        tol: Tolerance.  If None, auto-computed from spec.

    Returns:
        VerifyResult.
    """
    if tol is None:
        tol = compute_safe_tolerance(spec.epsilon)

    return verify(
        mechanism.p_final,
        spec.epsilon,
        spec.delta,
        spec.edges,
        tol=tol,
        mode=VerificationMode.MOST_VIOLATING,
    )


# ============================================================================
# __all__ export list
# ============================================================================


__all__ = [
    # Enums
    "VerificationMode",
    "ViolationType",
    # Dataclasses
    "ViolationRecord",
    "PairAnalysis",
    "VerificationReport",
    # Tolerance management
    "compute_safe_tolerance",
    "validate_tolerance",
    "warn_tolerance_violation",
    "check_invariant_i4",
    # Divergence computations
    "hockey_stick_divergence",
    "hockey_stick_divergence_detailed",
    "kl_divergence",
    "renyi_divergence",
    "max_divergence",
    "total_variation",
    # Core verification
    "verify",
    "verify_for_cegis",
    "quick_verify",
    "verify_extracted_mechanism",
    "counterexample_from_result",
    # Batch verification
    "verify_batch_pure_dp",
    "verify_batch_approx_dp",
    # Classes
    "PrivacyVerifier",
    "MonteCarloVerifier",
]
