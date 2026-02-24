"""Numerical utilities for the finite-width phase diagram system.

Numerically stable implementations of common operations: log-sum-exp,
softmax, matrix condition checking, positive-definiteness enforcement,
eigenvalue sorting, Gram matrix regularisation, numerical gradient checking.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg


# ---------------------------------------------------------------------------
# Stable log-sum-exp
# ---------------------------------------------------------------------------

def stable_log_sum_exp(x: NDArray) -> float:
    """Compute log(sum(exp(x))) in a numerically stable way.

    Uses the identity: log(sum(exp(x))) = c + log(sum(exp(x - c)))
    where c = max(x).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -np.inf
    c = np.max(x)
    if not np.isfinite(c):
        return float(c)
    return float(c + np.log(np.sum(np.exp(x - c))))


def stable_log_sum_exp_2d(x: NDArray, axis: int = 1) -> NDArray:
    """Log-sum-exp along an axis for a 2D array."""
    x = np.asarray(x, dtype=np.float64)
    c = np.max(x, axis=axis, keepdims=True)
    mask = np.isfinite(c)
    result = np.where(
        mask,
        c.squeeze(axis) + np.log(np.sum(np.exp(np.where(mask, x - c, 0.0)), axis=axis)),
        c.squeeze(axis),
    )
    return result


# ---------------------------------------------------------------------------
# Stable softmax
# ---------------------------------------------------------------------------

def stable_softmax(x: NDArray, axis: int = -1, temperature: float = 1.0) -> NDArray:
    """Compute softmax in a numerically stable way.

    Parameters
    ----------
    x : array
        Input logits.
    axis : int
        Axis along which to compute softmax.
    temperature : float
        Temperature parameter (>0). Higher → more uniform.
    """
    x = np.asarray(x, dtype=np.float64) / temperature
    c = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - c)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Condition number checking
# ---------------------------------------------------------------------------

def check_condition_number(
    M: NDArray,
    threshold: float = 1e12,
    name: str = "matrix",
) -> Dict[str, Any]:
    """Check the condition number of a matrix.

    Returns
    -------
    dict with keys: cond, log_cond, is_well_conditioned, warning
    """
    M = np.atleast_2d(M)
    try:
        cond = float(np.linalg.cond(M))
    except np.linalg.LinAlgError:
        cond = np.inf
    log_cond = np.log10(max(cond, 1e-300))
    well = cond < threshold
    warning = ""
    if not well:
        warning = (
            f"{name} is ill-conditioned: cond={cond:.2e} "
            f"(log10={log_cond:.1f}, threshold={threshold:.0e})"
        )
    return {
        "cond": cond,
        "log_cond": log_cond,
        "is_well_conditioned": well,
        "warning": warning,
    }


def safe_condition_number(M: NDArray) -> float:
    """Compute condition number, returning inf on failure."""
    try:
        return float(np.linalg.cond(M))
    except (np.linalg.LinAlgError, ValueError):
        return np.inf


# ---------------------------------------------------------------------------
# Positive-definiteness enforcement
# ---------------------------------------------------------------------------

def enforce_psd(
    M: NDArray,
    epsilon: float = 1e-10,
    method: str = "clip",
) -> NDArray:
    """Enforce positive semi-definiteness of a symmetric matrix.

    Parameters
    ----------
    M : array (n, n)
        Input symmetric matrix.
    epsilon : float
        Minimum eigenvalue or regularisation parameter.
    method : str
        ``"clip"`` — clip negative eigenvalues to epsilon.
        ``"shift"`` — add epsilon*I.
        ``"nearest"`` — project to nearest PSD matrix.

    Returns
    -------
    M_psd : array (n, n)
        PSD matrix.
    """
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    # Symmetrise
    M = 0.5 * (M + M.T)

    if method == "shift":
        return M + epsilon * np.eye(M.shape[0])

    if method == "clip":
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, epsilon)
        return (eigvecs * eigvals) @ eigvecs.T

    if method == "nearest":
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0.0)
        return (eigvecs * eigvals) @ eigvecs.T

    raise ValueError(f"Unknown method: {method}")


def is_psd(M: NDArray, tol: float = -1e-10) -> bool:
    """Check if M is positive semi-definite."""
    M = np.atleast_2d(M)
    try:
        eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
        return bool(np.all(eigvals >= tol))
    except np.linalg.LinAlgError:
        return False


def is_symmetric(M: NDArray, rtol: float = 1e-10) -> bool:
    """Check if M is symmetric within tolerance."""
    M = np.atleast_2d(M)
    if M.shape[0] != M.shape[1]:
        return False
    return bool(np.allclose(M, M.T, rtol=rtol))


# ---------------------------------------------------------------------------
# Eigenvalue utilities
# ---------------------------------------------------------------------------

def sorted_eigenvalues(
    M: NDArray,
    ascending: bool = True,
) -> Tuple[NDArray, NDArray]:
    """Compute eigenvalues and eigenvectors, sorted by eigenvalue magnitude.

    Parameters
    ----------
    M : array (n, n)
        Square matrix (assumed symmetric/Hermitian).
    ascending : bool
        Sort ascending (smallest first) if True.

    Returns
    -------
    eigenvalues, eigenvectors : sorted arrays.
    """
    M = np.atleast_2d(M)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (M + M.T))
    if ascending:
        idx = np.argsort(eigvals)
    else:
        idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


def eigenvalue_gap(M: NDArray) -> float:
    """Compute the spectral gap (difference between two largest eigenvalues)."""
    eigvals = np.sort(np.linalg.eigvalsh(M))
    if len(eigvals) < 2:
        return float(eigvals[0]) if len(eigvals) == 1 else 0.0
    return float(eigvals[-1] - eigvals[-2])


def effective_rank(M: NDArray, threshold: float = 1e-10) -> int:
    """Compute effective rank (number of eigenvalues above threshold)."""
    eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
    return int(np.sum(np.abs(eigvals) > threshold))


# ---------------------------------------------------------------------------
# Gram matrix regularisation
# ---------------------------------------------------------------------------

def regularize_gram(
    G: NDArray,
    method: str = "tikhonov",
    alpha: Optional[float] = None,
) -> NDArray:
    """Regularise a Gram matrix for numerical stability.

    Parameters
    ----------
    G : array (n, n)
        Gram matrix.
    method : str
        ``"tikhonov"`` — add alpha*I.
        ``"truncated_svd"`` — zero out small singular values.
        ``"adaptive"`` — choose alpha from condition number.
    alpha : float, optional
        Regularisation strength. Auto-selected if None.

    Returns
    -------
    G_reg : array (n, n)
    """
    G = np.atleast_2d(np.asarray(G, dtype=np.float64))
    n = G.shape[0]

    if method == "tikhonov":
        if alpha is None:
            alpha = max(1e-10, 1e-6 * np.trace(G) / n)
        return G + alpha * np.eye(n)

    if method == "truncated_svd":
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        if alpha is None:
            alpha = 1e-10 * s[0] if len(s) > 0 else 1e-10
        s = np.where(s > alpha, s, 0.0)
        return (U * s) @ Vt

    if method == "adaptive":
        eigvals = np.linalg.eigvalsh(G)
        min_eig = np.min(eigvals)
        if min_eig < 0:
            shift = abs(min_eig) + 1e-10
        else:
            cond = np.max(eigvals) / max(np.min(np.abs(eigvals[eigvals != 0])), 1e-300)
            if cond > 1e10:
                shift = np.max(eigvals) * 1e-10
            else:
                shift = 0.0
        return G + shift * np.eye(n) if shift > 0 else G

    raise ValueError(f"Unknown regularisation method: {method}")


# ---------------------------------------------------------------------------
# Numerical gradient checking
# ---------------------------------------------------------------------------

def numerical_gradient_check(
    f: Callable[[NDArray], float],
    x: NDArray,
    grad: NDArray,
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-7,
) -> Dict[str, Any]:
    """Check analytic gradient against finite-difference approximation.

    Parameters
    ----------
    f : callable
        Scalar-valued function.
    x : array
        Point at which to check.
    grad : array
        Analytic gradient to verify.
    eps : float
        Finite difference step size.
    rtol, atol : float
        Relative and absolute tolerance.

    Returns
    -------
    dict with keys: passed, max_rel_error, max_abs_error, component_errors
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    grad = np.asarray(grad, dtype=np.float64).ravel()
    assert x.shape == grad.shape, "x and grad must have same shape"

    fd_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        fd_grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    abs_err = np.abs(grad - fd_grad)
    denom = np.maximum(np.abs(grad) + np.abs(fd_grad), 1e-300)
    rel_err = abs_err / denom

    max_rel = float(np.max(rel_err))
    max_abs = float(np.max(abs_err))
    passed = bool(np.all((rel_err < rtol) | (abs_err < atol)))

    return {
        "passed": passed,
        "max_rel_error": max_rel,
        "max_abs_error": max_abs,
        "component_errors": rel_err,
        "fd_gradient": fd_grad,
        "analytic_gradient": grad,
    }


# ---------------------------------------------------------------------------
# Matrix function utilities
# ---------------------------------------------------------------------------

def stable_matrix_sqrt(M: NDArray, regularize: bool = True) -> NDArray:
    """Compute matrix square root via eigendecomposition."""
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    if regularize:
        eigvals = np.maximum(eigvals, 0.0)
    sqrt_eig = np.sqrt(eigvals)
    return (eigvecs * sqrt_eig) @ eigvecs.T


def stable_matrix_inv(
    M: NDArray,
    rcond: float = 1e-10,
) -> NDArray:
    """Compute matrix inverse via SVD with condition cutoff."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    threshold = rcond * s[0] if len(s) > 0 else rcond
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    return (Vt.T * s_inv) @ U.T


def stable_matrix_log(M: NDArray) -> NDArray:
    """Compute matrix logarithm via eigendecomposition for SPD matrices."""
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-300)
    log_eig = np.log(eigvals)
    return (eigvecs * log_eig) @ eigvecs.T


def stable_matrix_exp(M: NDArray) -> NDArray:
    """Compute matrix exponential via eigendecomposition for symmetric M."""
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    exp_eig = np.exp(eigvals)
    return (eigvecs * exp_eig) @ eigvecs.T


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def safe_divide(
    numerator: NDArray,
    denominator: NDArray,
    fill: float = 0.0,
) -> NDArray:
    """Element-wise division with fill value for zero denominators."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    mask = np.abs(den) > 1e-300
    result = np.full_like(num, fill)
    result[mask] = num[mask] / den[mask]
    return result


def relative_error(computed: NDArray, reference: NDArray) -> NDArray:
    """Element-wise relative error with safe denominator."""
    ref = np.asarray(reference, dtype=np.float64)
    comp = np.asarray(computed, dtype=np.float64)
    denom = np.maximum(np.abs(ref), 1e-300)
    return np.abs(comp - ref) / denom


def frobenius_relative_error(A: NDArray, B: NDArray) -> float:
    """Relative Frobenius-norm error ||A - B||_F / ||B||_F."""
    norm_b = np.linalg.norm(B, "fro")
    if norm_b < 1e-300:
        return float(np.linalg.norm(A - B, "fro"))
    return float(np.linalg.norm(A - B, "fro") / norm_b)


def clamp(x: NDArray, lo: float, hi: float) -> NDArray:
    """Clamp array values to [lo, hi]."""
    return np.clip(x, lo, hi)


def symmetrize(M: NDArray) -> NDArray:
    """Return (M + M^T) / 2."""
    return 0.5 * (M + M.T)


def trace_normalize(M: NDArray) -> NDArray:
    """Normalise a matrix by its trace."""
    tr = np.trace(M)
    if abs(tr) < 1e-300:
        return M
    return M / tr
