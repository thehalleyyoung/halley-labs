"""
Advanced data transformations for CausalCert.

Provides polynomial features, interaction terms, residualization,
orthogonalization, variable selection, dimensionality reduction,
kernel feature expansion, and random Fourier features.
"""

from __future__ import annotations

import logging
from itertools import combinations, combinations_with_replacement
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polynomial features
# ---------------------------------------------------------------------------


def polynomial_features(
    data: pd.DataFrame,
    degree: int = 2,
    include_bias: bool = False,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Generate polynomial features up to given degree.

    Parameters
    ----------
    data : pd.DataFrame
        Input data (numeric columns only).
    degree : int
        Maximum polynomial degree (default 2).
    include_bias : bool
        Whether to include a constant (bias) column.
    columns : list[str] | None
        Columns to transform. If None, uses all numeric columns.

    Returns
    -------
    pd.DataFrame
        Data with polynomial features appended.
    """
    if columns is None:
        numeric = data.select_dtypes(include=[np.number])
        columns = list(numeric.columns)
    else:
        numeric = data[columns]

    result = data.copy()

    if include_bias:
        result["_bias"] = 1.0

    col_values = {col: numeric[col].values for col in columns}

    for d in range(2, degree + 1):
        for combo in combinations_with_replacement(columns, d):
            name = "_".join(combo) + f"_pow{d}" if len(set(combo)) == 1 else "*".join(combo)
            vals = np.ones(len(data), dtype=float)
            for col in combo:
                vals = vals * col_values[col]
            result[name] = vals

    return result


# ---------------------------------------------------------------------------
# Interaction terms
# ---------------------------------------------------------------------------


def interaction_terms(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    max_order: int = 2,
) -> pd.DataFrame:
    """Generate interaction terms between columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to generate interactions for.
    max_order : int
        Maximum interaction order (2 = pairwise, 3 = three-way, etc.).

    Returns
    -------
    pd.DataFrame
        Data with interaction columns appended.
    """
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)

    result = data.copy()

    for order in range(2, max_order + 1):
        for combo in combinations(columns, order):
            name = ":".join(combo)
            vals = np.ones(len(data), dtype=float)
            for col in combo:
                vals = vals * data[col].values
            result[name] = vals

    return result


# ---------------------------------------------------------------------------
# Residualization
# ---------------------------------------------------------------------------


def residualize(
    data: pd.DataFrame,
    target_cols: list[str],
    control_cols: list[str],
    method: str = "ols",
) -> pd.DataFrame:
    """Residualize target columns with respect to control columns.

    For each target column, regress it on the control columns and return
    the residuals.  This is the Frisch-Waugh-Lovell theorem in action.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    target_cols : list[str]
        Columns to residualize.
    control_cols : list[str]
        Columns to partial out.
    method : str
        'ols' for ordinary least squares.

    Returns
    -------
    pd.DataFrame
        Data with target columns replaced by their residuals.
    """
    result = data.copy()

    if not control_cols:
        return result

    X = data[control_cols].values.astype(float)
    # Add intercept
    X_aug = np.column_stack([np.ones(len(X)), X])

    for col in target_cols:
        y = data[col].values.astype(float)

        # OLS: beta = (X'X)^{-1} X'y
        try:
            XtX = X_aug.T @ X_aug
            reg = np.linalg.regularize if hasattr(np.linalg, "regularize") else None
            XtX_inv = np.linalg.pinv(XtX)
            beta = XtX_inv @ (X_aug.T @ y)
            residuals = y - X_aug @ beta
            result[col] = residuals
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in residualization for column %s", col)

    return result


def partial_residuals(
    data: pd.DataFrame,
    y_col: str,
    x_col: str,
    control_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute partial residuals for Frisch-Waugh-Lovell visualization.

    Returns residualized x and y arrays.
    """
    resid = residualize(data, [y_col, x_col], control_cols)
    return resid[x_col].values, resid[y_col].values


# ---------------------------------------------------------------------------
# Orthogonalization (Gram-Schmidt)
# ---------------------------------------------------------------------------


def orthogonalize(
    data: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Orthogonalize columns using modified Gram-Schmidt.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to orthogonalize (in order). If None, all numeric.

    Returns
    -------
    pd.DataFrame
        Data with specified columns replaced by orthogonalized versions.
    """
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)

    result = data.copy()
    vectors: list[np.ndarray] = []

    for col in columns:
        v = data[col].values.astype(float).copy()

        # Subtract projections onto previous vectors
        for u in vectors:
            norm_u = np.dot(u, u)
            if norm_u > 1e-12:
                v = v - (np.dot(v, u) / norm_u) * u

        # Normalize
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            v = v / norm_v

        vectors.append(v)
        result[col] = v

    return result


def double_orthogonalization(
    data: pd.DataFrame,
    y_col: str,
    treatment_col: str,
    control_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Double/debiased machine learning style orthogonalization.

    Residualizes both treatment and outcome w.r.t. controls, then
    estimates the treatment effect from the residualized regression.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (treatment_residuals, outcome_residuals, estimated_effect)
    """
    resid = residualize(data, [y_col, treatment_col], control_cols)
    t_resid = resid[treatment_col].values
    y_resid = resid[y_col].values

    # Estimate effect from residualized regression
    t_var = np.var(t_resid)
    if t_var < 1e-12:
        effect = 0.0
    else:
        effect = float(np.cov(t_resid, y_resid)[0, 1] / t_var)

    return t_resid, y_resid, effect


# ---------------------------------------------------------------------------
# Variable selection via Lasso
# ---------------------------------------------------------------------------


def lasso_variable_selection(
    data: pd.DataFrame,
    target_col: str,
    candidate_cols: list[str] | None = None,
    lambda_reg: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> list[str]:
    """Select variables using coordinate descent Lasso.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    target_col : str
        Target column name.
    candidate_cols : list[str] | None
        Candidate predictor columns. If None, all numeric except target.
    lambda_reg : float
        Regularization parameter (higher = sparser).
    max_iter : int
        Maximum iterations for coordinate descent.
    tol : float
        Convergence tolerance.

    Returns
    -------
    list[str]
        Selected variable names (non-zero coefficients).
    """
    if candidate_cols is None:
        numeric = data.select_dtypes(include=[np.number])
        candidate_cols = [c for c in numeric.columns if c != target_col]

    y = data[target_col].values.astype(float)
    X = data[candidate_cols].values.astype(float)
    n, p = X.shape

    # Standardize
    y_mean = np.mean(y)
    y_centered = y - y_mean
    X_means = X.mean(axis=0)
    X_stds = X.std(axis=0)
    X_stds[X_stds < 1e-12] = 1.0
    X_scaled = (X - X_means) / X_stds

    # Coordinate descent
    beta = np.zeros(p)
    residuals = y_centered.copy()

    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(p):
            x_j = X_scaled[:, j]
            partial_resid = residuals + beta[j] * x_j
            rho = np.dot(x_j, partial_resid) / n
            # Soft threshold
            if rho > lambda_reg:
                beta[j] = rho - lambda_reg
            elif rho < -lambda_reg:
                beta[j] = rho + lambda_reg
            else:
                beta[j] = 0.0
            residuals = partial_resid - beta[j] * x_j

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    selected = [
        candidate_cols[j] for j in range(p) if abs(beta[j]) > 1e-10
    ]
    return selected


# ---------------------------------------------------------------------------
# Dimensionality reduction for conditioning sets
# ---------------------------------------------------------------------------


def pca_reduce(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> pd.DataFrame:
    """Reduce dimensionality of conditioning variables via PCA.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to reduce. If None, all numeric.
    n_components : int | None
        Number of components. If None, use variance_threshold.
    variance_threshold : float
        Minimum cumulative variance explained.

    Returns
    -------
    pd.DataFrame
        Data with PCA components replacing original columns.
    """
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)

    X = data[columns].values.astype(float)
    n, p = X.shape

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-12] = 1.0
    X_scaled = (X - X_mean) / X_std

    # SVD
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    explained_var = s ** 2 / (n - 1)
    total_var = explained_var.sum()
    cumulative = np.cumsum(explained_var) / total_var

    if n_components is None:
        n_components = int(np.searchsorted(cumulative, variance_threshold)) + 1
        n_components = min(n_components, p)

    # Project
    components = X_scaled @ Vt[:n_components].T
    result = data.drop(columns=columns).copy()
    for i in range(n_components):
        result[f"PC{i+1}"] = components[:, i]

    return result


def sufficient_reduction(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    n_directions: int = 1,
) -> np.ndarray:
    """Sliced inverse regression (SIR) for sufficient dimension reduction.

    Finds directions in X-space that capture the regression relationship
    between Y and X.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y_col : str
        Response column.
    x_cols : list[str]
        Predictor columns.
    n_directions : int
        Number of SIR directions.

    Returns
    -------
    np.ndarray
        Projection matrix (p × n_directions).
    """
    y = data[y_col].values.astype(float)
    X = data[x_cols].values.astype(float)
    n, p = X.shape

    # Standardize X
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-12] = 1.0
    X_scaled = (X - X_mean) / X_std

    # Slice Y into H slices
    H = min(10, n // 5)
    if H < 2:
        return np.eye(p)[:, :n_directions]

    quantiles = np.quantile(y, np.linspace(0, 1, H + 1))
    quantiles[-1] += 1e-10  # ensure last bin captures max

    # Compute slice means
    slice_means = np.zeros((H, p))
    slice_sizes = np.zeros(H)
    for h in range(H):
        mask = (y >= quantiles[h]) & (y < quantiles[h + 1])
        if h == H - 1:
            mask = (y >= quantiles[h]) & (y <= quantiles[h + 1])
        if mask.sum() > 0:
            slice_means[h] = X_scaled[mask].mean(axis=0)
            slice_sizes[h] = mask.sum()

    # Weighted covariance of slice means
    weighted_cov = np.zeros((p, p))
    for h in range(H):
        if slice_sizes[h] > 0:
            m = slice_means[h].reshape(-1, 1)
            weighted_cov += (slice_sizes[h] / n) * (m @ m.T)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    # Take top directions (largest eigenvalues)
    idx = np.argsort(eigvals)[::-1][:n_directions]
    return eigvecs[:, idx]


# ---------------------------------------------------------------------------
# Kernel feature expansion
# ---------------------------------------------------------------------------


def rbf_kernel_features(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    n_landmarks: int = 50,
    gamma: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Explicit kernel feature expansion using RBF kernel landmarks.

    Approximates the RBF kernel by computing distances to random landmarks.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to transform.
    n_landmarks : int
        Number of landmark points.
    gamma : float | None
        RBF kernel bandwidth. If None, uses median heuristic.
    seed : int
        Random seed for landmark selection.

    Returns
    -------
    pd.DataFrame
        Data with kernel features appended.
    """
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)

    X = data[columns].values.astype(float)
    n, p = X.shape

    rng = np.random.default_rng(seed)
    n_lm = min(n_landmarks, n)
    landmark_idx = rng.choice(n, size=n_lm, replace=False)
    landmarks = X[landmark_idx]

    if gamma is None:
        # Median heuristic
        dists = np.linalg.norm(X[:, None, :] - landmarks[None, :, :], axis=2)
        gamma = 1.0 / (2.0 * np.median(dists) ** 2 + 1e-12)

    # Compute features
    features = np.zeros((n, n_lm))
    for j in range(n_lm):
        diff = X - landmarks[j]
        features[:, j] = np.exp(-gamma * np.sum(diff ** 2, axis=1))

    result = data.copy()
    for j in range(n_lm):
        result[f"rbf_{j}"] = features[:, j]

    return result


# ---------------------------------------------------------------------------
# Random Fourier features
# ---------------------------------------------------------------------------


def random_fourier_features(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    n_features: int = 100,
    gamma: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Random Fourier feature approximation to the RBF kernel.

    Implements the Rahimi & Recht (2007) approximation:
        z(x) = sqrt(2/D) * cos(W @ x + b)
    where W ~ N(0, 2*gamma*I) and b ~ Uniform(0, 2*pi).

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to transform.
    n_features : int
        Number of random Fourier features.
    gamma : float | None
        RBF bandwidth. If None, uses median heuristic.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Data with RFF features appended.
    """
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)

    X = data[columns].values.astype(float)
    n, p = X.shape

    rng = np.random.default_rng(seed)

    if gamma is None:
        # Median heuristic on a subsample
        subsample = X[rng.choice(n, size=min(200, n), replace=False)]
        pairwise_dists = np.linalg.norm(
            subsample[:, None, :] - subsample[None, :, :], axis=2
        )
        median_dist = np.median(pairwise_dists[pairwise_dists > 0])
        gamma = 1.0 / (2.0 * median_dist ** 2 + 1e-12)

    # Sample random frequencies and biases
    W = rng.normal(0, np.sqrt(2 * gamma), size=(n_features, p))
    b = rng.uniform(0, 2 * np.pi, size=n_features)

    # Compute features
    projection = X @ W.T + b[None, :]
    features = np.sqrt(2.0 / n_features) * np.cos(projection)

    result = data.copy()
    for j in range(n_features):
        result[f"rff_{j}"] = features[:, j]

    return result


# ---------------------------------------------------------------------------
# Standardization helpers
# ---------------------------------------------------------------------------


def standardize(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "zscore",
) -> pd.DataFrame:
    """Standardize columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to standardize. If None, all numeric.
    method : str
        'zscore' (default), 'minmax', or 'robust' (IQR-based).
    """
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)

    result = data.copy()

    for col in columns:
        vals = data[col].values.astype(float)
        if method == "zscore":
            mu = np.mean(vals)
            sigma = np.std(vals)
            if sigma > 1e-12:
                result[col] = (vals - mu) / sigma
        elif method == "minmax":
            vmin = np.min(vals)
            vmax = np.max(vals)
            rng = vmax - vmin
            if rng > 1e-12:
                result[col] = (vals - vmin) / rng
        elif method == "robust":
            q25 = np.percentile(vals, 25)
            q75 = np.percentile(vals, 75)
            iqr = q75 - q25
            median = np.median(vals)
            if iqr > 1e-12:
                result[col] = (vals - median) / iqr
        else:
            raise ValueError(f"Unknown standardization method: {method}")

    return result


def winsorize(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Winsorize columns at given percentiles.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to winsorize.
    lower_pct, upper_pct : float
        Lower and upper percentile thresholds.
    """
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)

    result = data.copy()
    for col in columns:
        vals = data[col].values.astype(float)
        lo = np.percentile(vals, lower_pct * 100)
        hi = np.percentile(vals, upper_pct * 100)
        result[col] = np.clip(vals, lo, hi)

    return result


def rank_transform(
    data: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Replace values with their ranks (normalized to [0, 1])."""
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)

    result = data.copy()
    for col in columns:
        vals = data[col].values.astype(float)
        n = len(vals)
        order = np.argsort(np.argsort(vals))
        result[col] = order / (n - 1) if n > 1 else 0.0

    return result
