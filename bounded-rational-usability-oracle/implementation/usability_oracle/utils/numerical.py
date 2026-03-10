"""
usability_oracle.utils.numerical — Numerically stable primitives.

Provides building-block functions for numerical stability in the
bounded-rational usability pipeline: log-sum-exp, stable softmax,
compensated summation, running variance, condition-number monitoring,
precision checking, gradient verification, and integration helpers.

Design principles
-----------------
- Functions operate on NumPy arrays and return NumPy arrays.
- Epsilon guards (``_EPS = 1e-30``) prevent ``log(0)`` and ``0/0``.
- Compensated (Kahan) summation prevents catastrophic cancellation in
  long accumulations.

Performance characteristics
---------------------------
- ``logsumexp_stable``: O(n) with a single pass for the max and a
  single pass for the exp-sum.
- ``kahan_sum``: O(n) with four additions per element.
- ``welford_variance``: O(n) online, O(1) memory.
- ``gradient_check``: O(n) forward evaluations per parameter.

References
----------
Kahan, W. (1965). Further remarks on reducing truncation errors.
    *CACM* 8(1), 40.
Welford, B. P. (1962). Note on a method for calculating corrected sums
    of squares and products. *Technometrics* 4(3).
Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*
    (2nd ed.). SIAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Log-sum-exp
# ---------------------------------------------------------------------------


def logsumexp_stable(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Numerically stable log-sum-exp.

    ``logsumexp(x) = max(x) + log(Σ exp(x − max(x)))``

    Parameters
    ----------
    x : array_like
        Input values.
    axis : int or None
        Axis along which to reduce.  ``None`` reduces over all elements.

    Returns
    -------
    np.ndarray or float
        Reduced log-sum-exp values.

    Complexity
    ----------
    O(n) — two passes (max + sum-exp).
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.max(x, axis=axis, keepdims=True)
    # Handle all -inf case
    finite_mask = np.isfinite(c)
    c_safe = np.where(finite_mask, c, 0.0)
    out = c_safe + np.log(np.sum(np.exp(x - c_safe), axis=axis, keepdims=True) + _EPS)
    out = np.where(finite_mask, out, -np.inf)
    if axis is not None:
        return out.squeeze(axis=axis)
    return float(out.ravel()[0]) if out.size == 1 else out.ravel()


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax.

    ``log_softmax(x)_i = x_i − logsumexp(x)``

    Parameters
    ----------
    x : array_like
        Input logits.
    axis : int
        Axis along which to normalise.

    Returns
    -------
    np.ndarray
        Log-probabilities (same shape as *x*).

    Complexity
    ----------
    O(n) per row.
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.max(x, axis=axis, keepdims=True)
    shifted = x - c
    log_Z = c + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True) + _EPS)
    return x - log_Z


# ---------------------------------------------------------------------------
# Stable softmax
# ---------------------------------------------------------------------------


def softmax_stable(
    x: np.ndarray,
    temperature: float = 1.0,
    axis: int = -1,
) -> np.ndarray:
    """Numerically stable softmax with temperature scaling.

    ``softmax(x / T)_i = exp(x_i / T) / Σ exp(x_j / T)``

    Parameters
    ----------
    x : array_like
        Input logits.
    temperature : float
        Temperature > 0.
    axis : int
        Axis along which to normalise.

    Returns
    -------
    np.ndarray
        Probabilities (same shape as *x*).

    Raises
    ------
    ValueError
        If *temperature* ≤ 0.

    Complexity
    ----------
    O(n) per row.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = np.asarray(x, dtype=np.float64) / temperature
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / (np.sum(exp_vals, axis=axis, keepdims=True) + _EPS)


# ---------------------------------------------------------------------------
# Kahan (compensated) summation
# ---------------------------------------------------------------------------


def kahan_sum(values: np.ndarray) -> float:
    """Kahan compensated summation for reduced floating-point error.

    Achieves O(1) total error for summing *n* values, versus O(n·ε)
    for naive summation.

    Parameters
    ----------
    values : array_like
        Sequence of float values.

    Returns
    -------
    float
        Compensated sum.

    Complexity
    ----------
    O(n) with four additions per element.

    References
    ----------
    Kahan (1965).
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    s = 0.0
    c = 0.0  # compensation
    for v in values:
        y = float(v) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def kahan_cumsum(values: np.ndarray) -> np.ndarray:
    """Kahan compensated cumulative sum.

    Parameters
    ----------
    values : array_like, shape (n,)
        Input values.

    Returns
    -------
    np.ndarray, shape (n,)
        Compensated cumulative sums.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    s = 0.0
    c = 0.0
    for i in range(n):
        y = float(values[i]) - c
        t = s + y
        c = (t - s) - y
        s = t
        result[i] = s
    return result


# ---------------------------------------------------------------------------
# Welford online variance
# ---------------------------------------------------------------------------


@dataclass
class WelfordAccumulator:
    """Online mean and variance via Welford's algorithm.

    Maintains running statistics in a single pass with O(1) memory.

    Attributes
    ----------
    count : int
        Number of observations.
    mean : float
        Running mean.
    m2 : float
        Running sum of squared deviations.

    References
    ----------
    Welford (1962).
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        """Incorporate a new observation *x*."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def update_batch(self, values: np.ndarray) -> None:
        """Incorporate a batch of observations."""
        for v in np.asarray(values, dtype=np.float64).ravel():
            self.update(float(v))

    @property
    def variance(self) -> float:
        """Population variance."""
        return self.m2 / self.count if self.count > 0 else 0.0

    @property
    def sample_variance(self) -> float:
        """Sample variance (Bessel-corrected)."""
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)

    @property
    def sample_std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.sample_variance)

    def merge(self, other: "WelfordAccumulator") -> "WelfordAccumulator":
        """Merge two accumulators (parallel Welford).

        Parameters
        ----------
        other : WelfordAccumulator
            Accumulator to merge with.

        Returns
        -------
        WelfordAccumulator
            Combined accumulator.
        """
        if other.count == 0:
            return WelfordAccumulator(count=self.count, mean=self.mean, m2=self.m2)
        if self.count == 0:
            return WelfordAccumulator(count=other.count, mean=other.mean, m2=other.m2)

        n_a, n_b = self.count, other.count
        n = n_a + n_b
        delta = other.mean - self.mean
        new_mean = (n_a * self.mean + n_b * other.mean) / n
        new_m2 = self.m2 + other.m2 + delta ** 2 * n_a * n_b / n
        return WelfordAccumulator(count=n, mean=new_mean, m2=new_m2)


# ---------------------------------------------------------------------------
# Matrix condition number monitoring
# ---------------------------------------------------------------------------


def condition_number(A: np.ndarray, p: Union[int, float, str] = 2) -> float:
    """Compute the condition number of matrix *A*.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Matrix.
    p : int, float, or 'fro'
        Norm type (default 2 = spectral).

    Returns
    -------
    float
        Condition number.  ``inf`` if *A* is singular.
    """
    A = np.asarray(A, dtype=np.float64)
    try:
        return float(np.linalg.cond(A, p=p))
    except np.linalg.LinAlgError:
        return float("inf")


def check_conditioning(
    A: np.ndarray,
    warn_threshold: float = 1e10,
    error_threshold: float = 1e15,
) -> Tuple[float, str]:
    """Check matrix conditioning and return a diagnostic message.

    Parameters
    ----------
    A : array_like
        Matrix to check.
    warn_threshold : float
        Condition number above which a warning is issued.
    error_threshold : float
        Condition number above which an error is reported.

    Returns
    -------
    cond : float
        Condition number.
    message : str
        Diagnostic: ``"ok"``, ``"warning: ill-conditioned"``, or
        ``"error: near-singular"``.
    """
    cond = condition_number(A)
    if cond >= error_threshold:
        return cond, f"error: near-singular (cond={cond:.2e})"
    if cond >= warn_threshold:
        return cond, f"warning: ill-conditioned (cond={cond:.2e})"
    return cond, "ok"


# ---------------------------------------------------------------------------
# Automatic precision checking
# ---------------------------------------------------------------------------


def relative_error(computed: np.ndarray, reference: np.ndarray) -> float:
    """Relative error between *computed* and *reference*.

    ``||computed − reference|| / ||reference||``

    Parameters
    ----------
    computed, reference : array_like
        Arrays to compare.

    Returns
    -------
    float
        Relative error.  ``0`` if both are zero vectors.
    """
    c = np.asarray(computed, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    r_norm = np.linalg.norm(r)
    if r_norm < _EPS:
        return float(np.linalg.norm(c))
    return float(np.linalg.norm(c - r) / r_norm)


def assert_close(
    computed: np.ndarray,
    reference: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    label: str = "",
) -> None:
    """Assert element-wise closeness, raising on failure.

    Parameters
    ----------
    computed, reference : array_like
        Arrays to compare.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    label : str
        Label for error messages.

    Raises
    ------
    AssertionError
        If any element differs beyond tolerance.
    """
    c = np.asarray(computed, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if not np.allclose(c, r, rtol=rtol, atol=atol):
        max_diff = float(np.max(np.abs(c - r)))
        raise AssertionError(
            f"Precision check failed{' (' + label + ')' if label else ''}: "
            f"max_diff={max_diff:.2e}, rtol={rtol}, atol={atol}"
        )


# ---------------------------------------------------------------------------
# Gradient checking
# ---------------------------------------------------------------------------


def gradient_check(
    fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    grad: np.ndarray,
    epsilon: float = 1e-5,
) -> Tuple[float, np.ndarray]:
    """Numerical gradient check via centred finite differences.

    Verifies that the analytical gradient *grad* at *x* matches the
    numerical approximation ``(f(x+ε) − f(x−ε)) / 2ε``.

    Parameters
    ----------
    fn : callable
        Scalar function ``fn(x) -> float``.
    x : array_like
        Point at which to check.
    grad : array_like
        Analytical gradient at *x*.
    epsilon : float
        Finite-difference step size.

    Returns
    -------
    max_relative_error : float
        Maximum relative error across parameters.
    numerical_grad : np.ndarray
        Numerical gradient for comparison.

    Complexity
    ----------
    O(n) forward evaluations where n is the dimensionality of *x*.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    grad = np.asarray(grad, dtype=np.float64).ravel()
    n = len(x)
    num_grad = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        num_grad[i] = (fn(x_plus) - fn(x_minus)) / (2.0 * epsilon)

    denom = np.maximum(np.abs(grad) + np.abs(num_grad), _EPS)
    rel_errors = np.abs(grad - num_grad) / denom
    return float(np.max(rel_errors)), num_grad


def jacobian_numerical(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Numerical Jacobian via centred finite differences.

    Parameters
    ----------
    fn : callable
        Vector function ``fn(x) -> y``.
    x : array_like, shape (n,)
        Input point.
    epsilon : float
        Step size.

    Returns
    -------
    np.ndarray, shape (m, n)
        Jacobian matrix ``J[i, j] = ∂f_i/∂x_j``.

    Complexity
    ----------
    O(n) forward evaluations.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y0 = np.asarray(fn(x), dtype=np.float64).ravel()
    n = len(x)
    m = len(y0)
    J = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += epsilon
        x_minus[j] -= epsilon
        J[:, j] = (fn(x_plus) - fn(x_minus)) / (2.0 * epsilon)

    return J


# ---------------------------------------------------------------------------
# Numerical integration helpers
# ---------------------------------------------------------------------------


def trapezoid(
    f_values: np.ndarray,
    x_values: np.ndarray,
) -> float:
    """Trapezoidal integration.

    Parameters
    ----------
    f_values : array_like, shape (n,)
        Function values at sample points.
    x_values : array_like, shape (n,)
        Sample point positions (must be sorted).

    Returns
    -------
    float
        Approximate integral.

    Complexity
    ----------
    O(n).
    """
    f = np.asarray(f_values, dtype=np.float64).ravel()
    x = np.asarray(x_values, dtype=np.float64).ravel()
    if len(f) != len(x):
        raise ValueError("f_values and x_values must have the same length")
    if len(f) < 2:
        return 0.0
    dx = np.diff(x)
    return float(np.sum(0.5 * dx * (f[:-1] + f[1:])))


def simpsons(
    f_values: np.ndarray,
    x_values: np.ndarray,
) -> float:
    """Simpson's rule integration (composite).

    Falls back to trapezoidal if the number of points is even.

    Parameters
    ----------
    f_values : array_like, shape (n,)
        Function values.
    x_values : array_like, shape (n,)
        Sample positions (sorted, equally spaced).

    Returns
    -------
    float
        Approximate integral.

    Complexity
    ----------
    O(n).
    """
    f = np.asarray(f_values, dtype=np.float64).ravel()
    x = np.asarray(x_values, dtype=np.float64).ravel()
    n = len(f)
    if n < 3 or n % 2 == 0:
        return trapezoid(f, x)

    h = (x[-1] - x[0]) / (n - 1)
    # Simpson's 1/3 rule
    return float(
        (h / 3.0) * (
            f[0]
            + 4.0 * np.sum(f[1:-1:2])
            + 2.0 * np.sum(f[2:-2:2])
            + f[-1]
        )
    )


def gauss_legendre_quadrature(
    fn: Callable[[float], float],
    a: float,
    b: float,
    n_points: int = 5,
) -> float:
    """Gauss-Legendre quadrature on ``[a, b]``.

    Parameters
    ----------
    fn : callable
        Scalar function ``fn(x) -> float``.
    a, b : float
        Integration bounds.
    n_points : int
        Number of quadrature points (≤ 10 recommended).

    Returns
    -------
    float
        Approximate integral.

    Complexity
    ----------
    O(n_points) function evaluations.
    """
    try:
        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(n_points)
    except ImportError:
        # Fallback for very old numpy
        return trapezoid(
            np.array([fn(x) for x in np.linspace(a, b, n_points)]),
            np.linspace(a, b, n_points),
        )

    # Map from [-1, 1] to [a, b]
    mid = 0.5 * (b + a)
    half = 0.5 * (b - a)
    mapped_nodes = mid + half * nodes
    values = np.array([fn(float(x)) for x in mapped_nodes], dtype=np.float64)
    return float(half * np.sum(weights * values))


__all__ = [
    "logsumexp_stable",
    "log_softmax",
    "softmax_stable",
    "kahan_sum",
    "kahan_cumsum",
    "WelfordAccumulator",
    "condition_number",
    "check_conditioning",
    "relative_error",
    "assert_close",
    "gradient_check",
    "jacobian_numerical",
    "trapezoid",
    "simpsons",
    "gauss_legendre_quadrature",
]
