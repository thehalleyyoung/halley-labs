"""
usability_oracle.variational.convergence — Convergence analysis utilities.

Provides tools for monitoring and analysing the convergence behaviour of
iterative variational solvers.

* :class:`ConvergenceMonitor` — stateful convergence tracker
* :func:`check_convergence` — detect convergence in a value sequence
* :func:`compute_convergence_rate` — estimate asymptotic convergence rate
* :func:`lyapunov_stability` — Lyapunov stability of fixed points
* :func:`detect_oscillation` — detect limit cycles in iterative sequences
* :func:`extrapolate_convergence` — predict iterations to convergence
"""

from __future__ import annotations

import logging
from enum import Enum, unique
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

@unique
class ConvergenceRateType(Enum):
    """Classification of asymptotic convergence rate."""

    SUBLINEAR = "sublinear"
    LINEAR = "linear"
    SUPERLINEAR = "superlinear"
    QUADRATIC = "quadratic"
    NOT_CONVERGING = "not_converging"


# ═══════════════════════════════════════════════════════════════════════════
# Standalone functions
# ═══════════════════════════════════════════════════════════════════════════

def check_convergence(
    values: Sequence[float],
    tolerance: float = 1e-8,
    window: int = 3,
) -> bool:
    """Detect convergence in an iterative sequence.

    The sequence is considered converged when the maximum absolute
    change over the last *window* consecutive values is below *tolerance*.

    Parameters
    ----------
    values : sequence of float
        Iterative sequence of objective values (or similar scalar trace).
    tolerance : float
        Convergence threshold.
    window : int
        Number of consecutive differences to check (≥ 1).

    Returns
    -------
    bool
        ``True`` if the last *window* differences are all below *tolerance*.
    """
    vals = list(values)
    if len(vals) < window + 1:
        return False

    tail = vals[-(window + 1):]
    for i in range(1, len(tail)):
        if abs(tail[i] - tail[i - 1]) >= tolerance:
            return False
    return True


def compute_convergence_rate(
    values: Sequence[float],
) -> Tuple[ConvergenceRateType, float]:
    r"""Estimate the asymptotic convergence rate.

    Given a sequence of errors (or objective differences)
    e_0, e_1, …, e_n, estimates the rate by examining the ratio:

    .. math::

        r_k = \frac{|e_{k+1}|}{|e_k|}

    * If r_k → constant ∈ (0, 1): **linear** (with rate = constant).
    * If r_k → 0: **superlinear** (check for quadratic via |e_{k+1}|/|e_k|²).
    * If r_k → constant ≥ 1: **not converging**.
    * If r_k decreases but remains close to 1: **sublinear**.

    Parameters
    ----------
    values : sequence of float
        Sequence of errors or absolute objective differences.

    Returns
    -------
    rate_type : ConvergenceRateType
        Classification of the convergence rate.
    rate_constant : float
        Estimated rate constant (e.g., contraction factor for linear).
    """
    vals = np.array(values, dtype=np.float64)
    if vals.size < 3:
        return ConvergenceRateType.NOT_CONVERGING, 1.0

    # Compute absolute errors
    errors = np.abs(vals)
    # Remove leading zeros
    nonzero = errors > 0
    if nonzero.sum() < 3:
        return ConvergenceRateType.NOT_CONVERGING, 1.0

    errors = errors[nonzero]

    # Compute ratios r_k = e_{k+1} / e_k
    ratios = errors[1:] / np.maximum(errors[:-1], 1e-300)

    # Use last few ratios for classification
    n_tail = min(5, len(ratios))
    tail_ratios = ratios[-n_tail:]
    mean_ratio = float(np.mean(tail_ratios))

    if mean_ratio >= 1.0:
        return ConvergenceRateType.NOT_CONVERGING, mean_ratio

    if mean_ratio > 0.95:
        return ConvergenceRateType.SUBLINEAR, mean_ratio

    # Check for superlinear: is r_k decreasing toward 0?
    if len(tail_ratios) >= 3 and np.all(np.diff(tail_ratios) < 0) and tail_ratios[-1] < 0.1:
        # Check quadratic: e_{k+1} / e_k^2
        quad_ratios = errors[1:] / np.maximum(errors[:-1] ** 2, 1e-300)
        tail_quad = quad_ratios[-n_tail:]
        if np.std(tail_quad) / (np.mean(tail_quad) + 1e-300) < 0.5:
            return ConvergenceRateType.QUADRATIC, float(np.mean(tail_quad))
        return ConvergenceRateType.SUPERLINEAR, mean_ratio

    return ConvergenceRateType.LINEAR, mean_ratio


def lyapunov_stability(
    jacobian: np.ndarray,
) -> Tuple[bool, np.ndarray]:
    r"""Lyapunov stability analysis of a fixed point.

    A fixed point is stable if all eigenvalues of the Jacobian matrix
    have magnitude < 1 (for discrete-time systems) or have negative
    real part (for continuous-time).  This function checks the
    **discrete-time** condition:

    .. math::

        |\lambda_i(J)| < 1 \quad \forall\, i

    Parameters
    ----------
    jacobian : np.ndarray
        Jacobian matrix at the fixed point, shape ``(n, n)``.

    Returns
    -------
    is_stable : bool
        ``True`` if all eigenvalues have magnitude strictly < 1.
    eigenvalues : np.ndarray
        Complex eigenvalues of the Jacobian.
    """
    J = np.asarray(jacobian, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"Jacobian must be square, got shape {J.shape}")

    eigenvalues = np.linalg.eigvals(J)
    magnitudes = np.abs(eigenvalues)
    is_stable = bool(np.all(magnitudes < 1.0))

    logger.debug(
        "Lyapunov: max |λ|=%.6f, stable=%s",
        float(magnitudes.max()), is_stable,
    )

    return is_stable, eigenvalues


def detect_oscillation(
    values: Sequence[float],
    period_range: Tuple[int, int] = (2, 20),
) -> Optional[int]:
    r"""Detect limit cycles (oscillations) in an iterative sequence.

    Uses autocorrelation to find periodic behaviour.  Returns the detected
    period if a significant oscillation is found, else ``None``.

    Parameters
    ----------
    values : sequence of float
        Iterative sequence of scalar values.
    period_range : tuple of int
        ``(min_period, max_period)`` to search.

    Returns
    -------
    int or None
        Detected period, or ``None`` if no oscillation found.
    """
    vals = np.array(values, dtype=np.float64)
    n = len(vals)
    min_p, max_p = period_range

    if n < 2 * max_p:
        # Not enough data
        max_p = n // 2
    if max_p < min_p:
        return None

    # Remove mean
    vals_centered = vals - vals.mean()
    var = np.dot(vals_centered, vals_centered)
    if var < 1e-30:
        return None

    best_period: Optional[int] = None
    best_corr = 0.5  # threshold for significance

    for p in range(min_p, max_p + 1):
        # Autocorrelation at lag p
        autocorr = np.dot(vals_centered[:n - p], vals_centered[p:]) / var
        if autocorr > best_corr:
            best_corr = autocorr
            best_period = p

    if best_period is not None:
        logger.debug(
            "Oscillation detected: period=%d, autocorr=%.4f",
            best_period, best_corr,
        )

    return best_period


def extrapolate_convergence(
    values: Sequence[float],
    target_tolerance: float = 1e-8,
) -> Optional[int]:
    r"""Estimate iterations remaining until convergence.

    Fits a linear or geometric model to the recent error sequence and
    extrapolates to predict when |e_k| < *target_tolerance*.

    Parameters
    ----------
    values : sequence of float
        Sequence of errors or absolute objective differences.
    target_tolerance : float
        Target convergence criterion.

    Returns
    -------
    int or None
        Estimated additional iterations needed, or ``None`` if the
        sequence does not appear to be converging.
    """
    errors = np.abs(np.array(values, dtype=np.float64))
    if errors.size < 3:
        return None

    # Already converged?
    if errors[-1] < target_tolerance:
        return 0

    # Use last portion of the sequence
    n_tail = min(20, len(errors))
    tail = errors[-n_tail:]

    # Remove zeros
    nonzero_mask = tail > 0
    if nonzero_mask.sum() < 3:
        return None
    tail = tail[nonzero_mask]

    # Fit geometric model: log(e_k) ≈ a + b*k
    log_errors = np.log(tail)
    ks = np.arange(len(tail), dtype=np.float64)
    # Linear regression
    A = np.vstack([ks, np.ones_like(ks)]).T
    try:
        result = np.linalg.lstsq(A, log_errors, rcond=None)
        coeffs = result[0]
    except np.linalg.LinAlgError:
        return None

    slope = coeffs[0]
    intercept = coeffs[1]

    # slope must be negative for convergence
    if slope >= 0:
        return None

    # Predict: log(target) = intercept + slope * k_target
    log_target = np.log(target_tolerance)
    k_target = (log_target - intercept) / slope

    # Additional iterations from current position
    additional = int(np.ceil(k_target - (len(tail) - 1)))
    return max(additional, 1) if additional > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════
# ConvergenceMonitor class
# ═══════════════════════════════════════════════════════════════════════════

class ConvergenceMonitor:
    """Stateful monitor for tracking convergence of iterative algorithms.

    Records objective values and gradient norms, and provides on-line
    convergence detection, rate estimation, and oscillation detection.

    Parameters
    ----------
    tolerance : float
        Convergence threshold on absolute change.
    window : int
        Number of consecutive improvements required.
    max_iterations : int
        Hard iteration budget.
    """

    def __init__(
        self,
        tolerance: float = 1e-8,
        window: int = 3,
        max_iterations: int = 500,
    ) -> None:
        self.tolerance = tolerance
        self.window = window
        self.max_iterations = max_iterations

        self._values: List[float] = []
        self._gradient_norms: List[float] = []
        self._converged = False
        self._diverged = False

    @property
    def values(self) -> List[float]:
        """Recorded objective trace."""
        return list(self._values)

    @property
    def gradient_norms(self) -> List[float]:
        """Recorded gradient norm trace."""
        return list(self._gradient_norms)

    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return len(self._values)

    @property
    def is_converged(self) -> bool:
        """Whether convergence has been detected."""
        return self._converged

    @property
    def is_diverged(self) -> bool:
        """Whether divergence has been detected."""
        return self._diverged

    def record(
        self,
        value: float,
        gradient_norm: Optional[float] = None,
    ) -> bool:
        """Record a new objective value and check convergence.

        Parameters
        ----------
        value : float
            Current objective function value.
        gradient_norm : float, optional
            Current gradient L2 norm.

        Returns
        -------
        bool
            ``True`` if the solver should **stop** (converged, diverged,
            or budget exhausted).
        """
        # Divergence check
        if not np.isfinite(value):
            self._diverged = True
            self._values.append(value)
            if gradient_norm is not None:
                self._gradient_norms.append(gradient_norm)
            return True

        self._values.append(value)
        if gradient_norm is not None:
            self._gradient_norms.append(gradient_norm)

        # Budget exhausted
        if len(self._values) >= self.max_iterations:
            return True

        # Convergence check
        if check_convergence(self._values, self.tolerance, self.window):
            self._converged = True
            return True

        return False

    def get_rate(self) -> Tuple[ConvergenceRateType, float]:
        """Estimate the current convergence rate.

        Returns
        -------
        rate_type : ConvergenceRateType
        rate_constant : float
        """
        if len(self._values) < 3:
            return ConvergenceRateType.NOT_CONVERGING, 1.0

        diffs = [
            abs(self._values[i] - self._values[i - 1])
            for i in range(1, len(self._values))
        ]
        return compute_convergence_rate(diffs)

    def detect_oscillation(
        self,
        period_range: Tuple[int, int] = (2, 20),
    ) -> Optional[int]:
        """Check for oscillatory behaviour in the recorded trace.

        Returns
        -------
        int or None
            Detected period, or ``None``.
        """
        return detect_oscillation(self._values, period_range)

    def extrapolate(self, target_tolerance: Optional[float] = None) -> Optional[int]:
        """Predict iterations remaining to convergence.

        Parameters
        ----------
        target_tolerance : float, optional
            Target tolerance.  Uses the monitor's tolerance if ``None``.

        Returns
        -------
        int or None
        """
        tol = target_tolerance if target_tolerance is not None else self.tolerance
        if len(self._values) < 3:
            return None
        diffs = [
            abs(self._values[i] - self._values[i - 1])
            for i in range(1, len(self._values))
        ]
        return extrapolate_convergence(diffs, tol)
