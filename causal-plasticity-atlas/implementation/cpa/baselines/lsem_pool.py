"""Linear SEM pooled estimation baseline (BL7).

Estimates a linear structural equation model (SEM) by pooling data
across all contexts, then fits per-context SEMs to compare coefficients.
The model is  X = B^T X + ε  where B is the weighted adjacency matrix.

Supports no regularization, L1 (lasso), and L2 (ridge) estimation.
Plasticity is classified by comparing per-context coefficient matrices.

References
----------
Shimizu et al. (2006).  A Linear Non-Gaussian Acyclic Model for
Causal Discovery.  *JMLR*, 7, 2003-2030.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from cpa.core.types import PlasticityClass
from cpa.baselines.ind_phc import _collect_edges


# -------------------------------------------------------------------
# Regularized regression utilities
# -------------------------------------------------------------------


def _ols_regression(
    X: NDArray, y: NDArray,
) -> NDArray:
    """Ordinary least squares: return coefficient vector."""
    if X.shape[1] == 0:
        return np.array([], dtype=np.float64)
    X_aug = np.column_stack([X, np.ones(X.shape[0])])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta[:-1]  # exclude intercept


def _lasso_coordinate_descent(
    X: NDArray, y: NDArray, alpha: float,
    max_iter: int = 1000, tol: float = 1e-6,
) -> NDArray:
    """L1-regularized regression via coordinate descent.

    Minimizes  (1/2n) ||y - Xβ||^2 + α ||β||_1
    """
    n, p = X.shape
    if p == 0:
        return np.array([], dtype=np.float64)

    # Standardize
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_c = X - X_mean
    y_c = y - y_mean

    # Precompute
    X_sq = np.sum(X_c ** 2, axis=0)
    beta = np.zeros(p, dtype=np.float64)
    residual = y_c.copy()

    for iteration in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if X_sq[j] < 1e-15:
                beta[j] = 0.0
                continue
            # Partial residual
            residual += X_c[:, j] * beta[j]
            rho = float(np.dot(X_c[:, j], residual)) / n
            # Soft thresholding
            if rho > alpha:
                beta[j] = (rho - alpha) / (X_sq[j] / n)
            elif rho < -alpha:
                beta[j] = (rho + alpha) / (X_sq[j] / n)
            else:
                beta[j] = 0.0
            residual -= X_c[:, j] * beta[j]

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


def _ridge_regression(
    X: NDArray, y: NDArray, alpha: float,
) -> NDArray:
    """L2-regularized regression (ridge).

    Minimizes  (1/2n) ||y - Xβ||^2 + α ||β||^2
    """
    n, p = X.shape
    if p == 0:
        return np.array([], dtype=np.float64)

    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_c = X - X_mean
    y_c = y - y_mean

    # (X^T X + n*alpha*I)^{-1} X^T y
    XtX = X_c.T @ X_c
    Xty = X_c.T @ y_c
    I = np.eye(p)
    beta = np.linalg.solve(XtX + n * alpha * I, Xty)
    return beta


# -------------------------------------------------------------------
# SEM coefficient matrix estimation
# -------------------------------------------------------------------


def _estimate_sem_coefficients(
    data: NDArray,
    regularization: str = "none",
    alpha: float = 0.1,
) -> NDArray:
    """Estimate the B matrix of a linear SEM  X = B^T X + ε.

    For each variable j, regresses j on all other variables.

    Parameters
    ----------
    data : (n, p)
    regularization : "none", "l1", or "l2"
    alpha : regularization strength

    Returns
    -------
    B : (p, p) coefficient matrix.  B[i, j] = coefficient of X_i
        in the equation for X_j.  Diagonal is zero.
    """
    n, p = data.shape
    B = np.zeros((p, p), dtype=np.float64)

    for j in range(p):
        y = data[:, j]
        # Use all other variables as predictors
        predictor_idx = [i for i in range(p) if i != j]
        X = data[:, predictor_idx]

        if regularization == "l1":
            beta = _lasso_coordinate_descent(X, y, alpha)
        elif regularization == "l2":
            beta = _ridge_regression(X, y, alpha)
        else:
            beta = _ols_regression(X, y)

        for k, idx in enumerate(predictor_idx):
            B[idx, j] = beta[k]

    return B


def _threshold_coefficients(
    B: NDArray, threshold: float = 0.1,
) -> NDArray:
    """Zero out coefficients below threshold in absolute value."""
    B_thresh = B.copy()
    B_thresh[np.abs(B_thresh) < threshold] = 0.0
    return B_thresh


def _coefficient_to_adjacency(
    B: NDArray, threshold: float = 0.1,
) -> NDArray:
    """Convert coefficient matrix to binary adjacency."""
    return (np.abs(B) >= threshold).astype(np.float64)


def _compute_residual_variances(
    data: NDArray, B: NDArray,
) -> NDArray:
    """Compute residual variances given B matrix."""
    n, p = data.shape
    variances = np.zeros(p, dtype=np.float64)
    for j in range(p):
        parents = np.where(B[:, j] != 0)[0]
        if len(parents) == 0:
            variances[j] = float(np.var(data[:, j], ddof=1))
        else:
            predicted = data[:, parents] @ B[parents, j]
            residuals = data[:, j] - predicted
            variances[j] = float(np.var(residuals, ddof=1))
    return variances


# -------------------------------------------------------------------
# Cross-context coefficient comparison
# -------------------------------------------------------------------


def _compare_coefficients(
    B_per_ctx: Dict[str, NDArray],
    significance_level: float = 0.05,
) -> Dict[Tuple[int, int], PlasticityClass]:
    """Compare coefficient matrices across contexts.

    For each edge (i, j):
    - If B[i,j] ≈ 0 in all contexts → no edge
    - If B[i,j] ≈ same nonzero value across all → INVARIANT
    - If B[i,j] is nonzero in all but differs → PARAMETRIC_PLASTIC
    - If B[i,j] is nonzero in some, zero in others → STRUCTURAL_PLASTIC
    - If B[i,j] is nonzero in exactly one → EMERGENT
    """
    ctx_keys = sorted(B_per_ctx.keys())
    n_ctx = len(ctx_keys)
    if n_ctx == 0:
        return {}

    p = B_per_ctx[ctx_keys[0]].shape[0]
    threshold = 0.05  # coefficient significance threshold
    classifications: Dict[Tuple[int, int], PlasticityClass] = {}

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if (i, j) in classifications or (j, i) in classifications:
                continue

            coeffs = [float(B_per_ctx[k][i, j]) for k in ctx_keys]
            is_nonzero = [abs(c) > threshold for c in coeffs]
            n_nonzero = sum(is_nonzero)

            if n_nonzero == 0:
                continue  # no edge in any context

            if n_nonzero == n_ctx:
                # Edge present in all contexts – check if coefficients differ
                coeff_arr = np.array(coeffs)
                coeff_range = float(np.max(coeff_arr) - np.min(coeff_arr))
                coeff_mean = float(np.mean(np.abs(coeff_arr)))
                relative_range = (
                    coeff_range / coeff_mean if coeff_mean > 1e-10 else 0.0
                )

                if n_ctx >= 3:
                    # Use F-test equivalent: coefficient of variation
                    cv = float(np.std(coeff_arr) / max(abs(coeff_mean), 1e-10))
                    if cv < 0.15:
                        classifications[(i, j)] = PlasticityClass.INVARIANT
                    else:
                        classifications[(i, j)] = (
                            PlasticityClass.PARAMETRIC_PLASTIC
                        )
                else:
                    if relative_range < 0.3:
                        classifications[(i, j)] = PlasticityClass.INVARIANT
                    else:
                        classifications[(i, j)] = (
                            PlasticityClass.PARAMETRIC_PLASTIC
                        )
            elif n_nonzero == 1:
                classifications[(i, j)] = PlasticityClass.EMERGENT
            else:
                classifications[(i, j)] = PlasticityClass.STRUCTURAL_PLASTIC

    return classifications


# -------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------


class LSEMPooled:
    """Linear SEM pooled estimation baseline (BL7).

    Estimates a linear SEM from pooled data and per-context data,
    then compares coefficients to classify plasticity.

    Parameters
    ----------
    regularization : str
        ``"none"``, ``"l1"`` (lasso), or ``"l2"`` (ridge).
    lambda_reg : float
        Regularization strength.
    threshold : float
        Minimum absolute coefficient to count as an edge.
    """

    def __init__(
        self,
        regularization: str = "none",
        lambda_reg: float = 0.1,
        threshold: float = 0.1,
    ) -> None:
        if regularization not in ("none", "l1", "l2"):
            raise ValueError(
                f"Unsupported regularization: {regularization!r}"
            )
        self._regularization = regularization
        self._lambda_reg = lambda_reg
        self._threshold = threshold
        self._coefficients: Optional[NDArray] = None
        self._residual_vars: Optional[NDArray] = None
        self._per_ctx_coeffs: Dict[str, NDArray] = {}
        self._per_ctx_residual_vars: Dict[str, NDArray] = {}
        self._n_vars: int = 0
        self._datasets: Dict[str, NDArray] = {}
        self._plasticity: Dict[Tuple[int, int], PlasticityClass] = {}
        self._fitted: bool = False

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(
        self,
        datasets: Dict[str, NDArray],
        context_labels: Optional[List[str]] = None,
    ) -> "LSEMPooled":
        """Fit the linear SEM on pooled and per-context data.

        Parameters
        ----------
        datasets : Dict[str, NDArray]
            ``{context_label: (n_samples, n_vars)}`` arrays.
        context_labels : list of str, optional

        Returns
        -------
        self
        """
        if not datasets:
            raise ValueError("datasets must be non-empty")
        if isinstance(datasets, list):
            datasets = {f"ctx_{i}": d for i, d in enumerate(datasets)}
        first = next(iter(datasets.values()))
        self._n_vars = first.shape[1]
        self._datasets = dict(datasets)

        for k, d in datasets.items():
            if d.shape[1] != self._n_vars:
                raise ValueError(
                    f"Context {k!r}: {d.shape[1]} vars, "
                    f"expected {self._n_vars}"
                )

        # Pooled estimation
        pooled = np.vstack([datasets[k] for k in sorted(datasets.keys())])
        self._coefficients = self._pooled_sem_estimation(pooled)
        self._residual_vars = _compute_residual_variances(
            pooled, self._coefficients,
        )

        # Per-context estimation
        for ctx_key, data in datasets.items():
            B_ctx = self._pooled_sem_estimation(data)
            self._per_ctx_coeffs[ctx_key] = B_ctx
            self._per_ctx_residual_vars[ctx_key] = (
                _compute_residual_variances(data, B_ctx)
            )

        # Classify edges
        self._plasticity = self._classify_edges(self._per_ctx_coeffs)

        self._fitted = True
        return self

    def predict_plasticity(self) -> Dict[Tuple[int, int], PlasticityClass]:
        """Return edge plasticity classifications."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._plasticity)

    def coefficient_matrix(self) -> NDArray:
        """Return the pooled coefficient matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._coefficients is not None
        return self._coefficients.copy()

    def residual_variances(self) -> NDArray:
        """Return estimated residual variances (pooled)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._residual_vars is not None
        return self._residual_vars.copy()

    def pooled_graph(self, threshold: Optional[float] = None) -> NDArray:
        """Threshold coefficient matrix to get adjacency."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._coefficients is not None
        t = threshold if threshold is not None else self._threshold
        return _coefficient_to_adjacency(self._coefficients, t)

    def per_context_coefficients(self) -> Dict[str, NDArray]:
        """Return per-context coefficient matrices."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return {k: v.copy() for k, v in self._per_ctx_coeffs.items()}

    def per_context_residual_variances(self) -> Dict[str, NDArray]:
        """Return per-context residual variance vectors."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return {k: v.copy() for k, v in self._per_ctx_residual_vars.items()}

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._coefficients is not None
        adj = self.pooled_graph()
        n_edges = int(np.sum(adj))
        return {
            "n_variables": self._n_vars,
            "n_edges": n_edges,
            "density": n_edges / max(self._n_vars * (self._n_vars - 1), 1),
            "regularization": self._regularization,
            "lambda": self._lambda_reg,
            "max_abs_coeff": float(np.max(np.abs(self._coefficients))),
            "mean_abs_coeff": float(
                np.mean(np.abs(self._coefficients[self._coefficients != 0]))
            ) if np.any(self._coefficients != 0) else 0.0,
        }

    # ---------------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------------

    def _pooled_sem_estimation(self, data: NDArray) -> NDArray:
        """Estimate B matrix via regularized regression."""
        return _estimate_sem_coefficients(
            data, self._regularization, self._lambda_reg,
        )

    def _lasso_estimation(self, data: NDArray, alpha: float) -> NDArray:
        """L1 regularized estimation (convenience wrapper)."""
        return _estimate_sem_coefficients(data, "l1", alpha)

    def _ridge_estimation(self, data: NDArray, alpha: float) -> NDArray:
        """L2 regularized estimation (convenience wrapper)."""
        return _estimate_sem_coefficients(data, "l2", alpha)

    def _threshold_coefficients(
        self, B: NDArray, threshold: Optional[float] = None,
    ) -> NDArray:
        """Threshold small coefficients to zero."""
        t = threshold if threshold is not None else self._threshold
        return _threshold_coefficients(B, t)

    def _classify_edges(
        self, B_per_context: Dict[str, NDArray],
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Compare coefficients across contexts for plasticity."""
        return _compare_coefficients(B_per_context)
