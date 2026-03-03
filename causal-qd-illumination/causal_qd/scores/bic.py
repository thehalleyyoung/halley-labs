"""Bayesian Information Criterion (BIC) score for DAGs.

This module provides a full-featured BIC scoring implementation for
structure learning under a linear-Gaussian model.  It supports:

* Standard OLS-based local scoring.
* Ridge (L2) and Lasso (L1) regularisation of regression coefficients.
* Missing data handling via an Expectation-Maximisation (EM) wrapper that
  iteratively estimates sufficient statistics from incomplete observations.
* Efficient incremental score updates through :meth:`score_diff`.

The implementation relies only on NumPy and mirrors the interface defined by
:class:`~causal_qd.scores.score_base.DecomposableScore`.
"""
from __future__ import annotations

import numpy as np

from causal_qd.scores.score_base import DecomposableScore
from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VARIANCE_FLOOR: float = 1e-300
"""Minimum variance value to guard against log(0)."""

_EM_MAX_ITER: int = 50
"""Default maximum number of EM iterations for missing-data handling."""

_EM_TOL: float = 1e-6
"""Convergence tolerance (relative change in log-likelihood) for EM."""

_LASSO_MAX_ITER: int = 1000
"""Maximum iterations for coordinate-descent Lasso solver."""

_LASSO_TOL: float = 1e-6
"""Convergence tolerance for coordinate-descent Lasso solver."""


class BICScore(DecomposableScore):
    """BIC score under a linear-Gaussian model.

    For each node *j* with parent set *Pa(j)* the local score is the sum
    of a Gaussian log-likelihood and a BIC complexity penalty::

        local_score = log_likelihood + penalty

    where

    * ``log_likelihood = -(m/2)*log(2π) - (m/2)*log(σ²_hat) - m/2``
    * ``penalty        = -(k/2)*log(m)``

    Here *m* is the number of samples, *σ²_hat* is the residual variance
    of *X_j* regressed on its parents, and *k = |Pa(j)| + 1* counts the
    free parameters (intercept + regression coefficients).

    Parameters
    ----------
    penalty_multiplier : float, optional
        Multiplicative scaling factor for the BIC penalty term.  Values
        greater than 1.0 encourage sparser graphs; values less than 1.0
        are more permissive.  Default is ``1.0`` (standard BIC).
    regularization : str, optional
        Regularisation strategy applied to the linear regression.  One of
        ``"none"`` (ordinary least squares), ``"l1"`` (Lasso via coordinate
        descent), or ``"l2"`` (Ridge, closed-form).  Default is ``"none"``.
    reg_lambda : float, optional
        Regularisation strength (λ).  Only used when *regularization* is
        ``"l1"`` or ``"l2"``.  Default is ``0.01``.

    Notes
    -----
    When the data contain ``NaN`` values the scorer automatically falls
    back to an EM procedure that iteratively imputes missing entries and
    re-estimates the sufficient statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_qd.scores.bic import BICScore
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((200, 3))
    >>> scorer = BICScore()
    >>> scorer.local_score(0, [], data)  # score of node 0 with no parents
    -...
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    _VALID_REGULARIZATIONS = frozenset({"none", "l1", "l2"})

    def __init__(
        self,
        penalty_multiplier: float = 1.0,
        regularization: str = "none",
        reg_lambda: float = 0.01,
    ) -> None:
        if regularization not in self._VALID_REGULARIZATIONS:
            raise ValueError(
                f"Unknown regularization '{regularization}'. "
                f"Choose from {sorted(self._VALID_REGULARIZATIONS)}."
            )
        if reg_lambda < 0.0:
            raise ValueError("reg_lambda must be non-negative.")

        self._penalty_multiplier: float = penalty_multiplier
        self._regularization: str = regularization
        self._reg_lambda: float = reg_lambda

    # ------------------------------------------------------------------ #
    #  Public API – required by DecomposableScore
    # ------------------------------------------------------------------ #

    def local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Compute the local BIC score for *node* given *parents*.

        If the data contain ``NaN`` entries in the columns relevant to
        *node* and *parents*, the method transparently delegates to
        :meth:`_em_local_score` for EM-based estimation.

        Parameters
        ----------
        node : int
            Column index of the child variable.
        parents : list[int]
            Column indices of the parent variables.
        data : DataMatrix
            ``(m, p)`` observation matrix.

        Returns
        -------
        float
            The local BIC score contribution of *node*.
        """
        # Determine relevant columns and check for missing values.
        cols = [node] + list(parents)
        relevant = data[:, cols]
        if np.any(np.isnan(relevant)):
            return self._em_local_score(node, parents, data)

        return self._complete_local_score(node, parents, data)

    # score() is inherited from DecomposableScore (sum of local scores).

    def score_diff(
        self,
        dag: AdjacencyMatrix,
        node: int,
        old_parents: list[int],
        new_parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Efficiently compute the change in score when updating parents.

        Instead of re-scoring the entire DAG, only the local score for
        *node* is recomputed under both the old and new parent sets and
        the difference is returned.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Current adjacency matrix (unused beyond documentation; the
            caller is expected to maintain the DAG separately).
        node : int
            The node whose parent set is being modified.
        old_parents : list[int]
            The current parent set of *node*.
        new_parents : list[int]
            The proposed parent set of *node*.
        data : DataMatrix
            ``(m, p)`` observation matrix.

        Returns
        -------
        float
            ``new_local_score - old_local_score``.  A positive value means
            the modification improves the overall score.
        """
        new_local = self.local_score(node, new_parents, data)
        old_local = self.local_score(node, old_parents, data)
        return new_local - old_local

    # ------------------------------------------------------------------ #
    #  Core scoring on complete data
    # ------------------------------------------------------------------ #

    def _complete_local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float:
        """BIC local score assuming no missing values.

        Parameters
        ----------
        node : int
            Column index of the child variable.
        parents : list[int]
            Column indices of the parent variables.
        data : DataMatrix
            Complete ``(m, p)`` observation matrix.

        Returns
        -------
        float
            The local BIC score.
        """
        m = data.shape[0]
        y = data[:, node]

        # --- Fit regression and obtain residual variance ---------------
        coeffs, sigma2 = self._fit_regression(y, parents, data)

        # --- Log-likelihood (Gaussian) ---------------------------------
        #   L = -(m/2)*log(2π) - (m/2)*log(σ²) - m/2
        log_likelihood = (
            -0.5 * m * np.log(2.0 * np.pi)
            - 0.5 * m * np.log(sigma2)
            - 0.5 * m
        )

        # --- BIC penalty ----------------------------------------------
        k = len(parents) + 1  # intercept + regression coefficients
        penalty = -0.5 * self._penalty_multiplier * k * np.log(m)

        return float(log_likelihood + penalty)

    # ------------------------------------------------------------------ #
    #  Regression helpers
    # ------------------------------------------------------------------ #

    def _fit_regression(
        self,
        y: np.ndarray,
        parents: list[int],
        data: DataMatrix,
    ) -> tuple[np.ndarray, float]:
        """Dispatch to the appropriate regression method.

        Parameters
        ----------
        y : np.ndarray
            ``(m,)`` response vector (child node observations).
        parents : list[int]
            Column indices for predictors in *data*.
        data : DataMatrix
            Full ``(m, p)`` observation matrix.

        Returns
        -------
        coeffs : np.ndarray
            Fitted coefficient vector (including intercept as first element).
        sigma2 : float
            Estimated residual variance (MLE, i.e. divided by *m*).
        """
        if self._regularization == "l2":
            return self._fit_ridge(y, parents, data)
        if self._regularization == "l1":
            return self._fit_lasso(y, parents, data)
        return self._fit_ols(y, parents, data)

    # -- OLS -----------------------------------------------------------

    def _fit_ols(
        self,
        y: np.ndarray,
        parents: list[int],
        data: DataMatrix,
    ) -> tuple[np.ndarray, float]:
        """Ordinary least squares regression.

        Fits the model  y = X @ β + ε  where X includes an intercept
        column.  Uses :func:`numpy.linalg.lstsq` for numerical stability.

        Parameters
        ----------
        y : np.ndarray
            ``(m,)`` response vector.
        parents : list[int]
            Column indices for predictors.
        data : DataMatrix
            Full observation matrix.

        Returns
        -------
        coeffs : np.ndarray
            Fitted coefficients ``[intercept, β₁, …, βₚ]``.
        sigma2 : float
            MLE residual variance.
        """
        m = y.shape[0]

        if parents:
            X = np.column_stack([np.ones(m), data[:, parents]])
        else:
            X = np.ones((m, 1))

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback: mean-only model.
            coeffs = np.array([np.mean(y)])
            X = np.ones((m, 1))

        residuals = y - X @ coeffs
        sigma2 = float(np.mean(residuals ** 2))
        sigma2 = max(sigma2, _VARIANCE_FLOOR)
        return coeffs, sigma2

    # -- Ridge (L2) ----------------------------------------------------

    def _fit_ridge(
        self,
        y: np.ndarray,
        parents: list[int],
        data: DataMatrix,
    ) -> tuple[np.ndarray, float]:
        """Ridge regression (L2 penalty) via closed-form solution.

        The ridge estimator is::

            β = (XᵀX + λI')⁻¹ Xᵀy

        where *I'* is the identity with the intercept row/column zeroed
        out (the intercept is **not** penalised).

        Parameters
        ----------
        y : np.ndarray
            ``(m,)`` response vector.
        parents : list[int]
            Column indices for predictors.
        data : DataMatrix
            Full observation matrix.

        Returns
        -------
        coeffs : np.ndarray
            Ridge regression coefficients ``[intercept, β₁, …, βₚ]``.
        sigma2 : float
            MLE residual variance using ridge-fitted values.
        """
        m = y.shape[0]

        if parents:
            X = np.column_stack([np.ones(m), data[:, parents]])
        else:
            X = np.ones((m, 1))

        p = X.shape[1]
        # Penalty matrix: do not penalise the intercept.
        penalty_mat = self._reg_lambda * np.eye(p)
        penalty_mat[0, 0] = 0.0

        try:
            gram = X.T @ X + penalty_mat
            coeffs = np.linalg.solve(gram, X.T @ y)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(p)
            coeffs[0] = np.mean(y)

        residuals = y - X @ coeffs
        sigma2 = float(np.mean(residuals ** 2))
        sigma2 = max(sigma2, _VARIANCE_FLOOR)
        return coeffs, sigma2

    # -- Lasso (L1) ----------------------------------------------------

    def _fit_lasso(
        self,
        y: np.ndarray,
        parents: list[int],
        data: DataMatrix,
    ) -> tuple[np.ndarray, float]:
        """Lasso regression (L1 penalty) via coordinate descent.

        The intercept is **not** penalised.  Predictors are internally
        standardised (zero-mean, unit-variance) for stable convergence and
        then un-standardised before returning.

        Parameters
        ----------
        y : np.ndarray
            ``(m,)`` response vector.
        parents : list[int]
            Column indices for predictors.
        data : DataMatrix
            Full observation matrix.

        Returns
        -------
        coeffs : np.ndarray
            Lasso regression coefficients ``[intercept, β₁, …, βₚ]``.
        sigma2 : float
            MLE residual variance using Lasso-fitted values.
        """
        m = y.shape[0]

        if not parents:
            intercept = float(np.mean(y))
            sigma2 = float(np.mean((y - intercept) ** 2))
            sigma2 = max(sigma2, _VARIANCE_FLOOR)
            return np.array([intercept]), sigma2

        X_raw = data[:, parents].copy()
        n_feats = X_raw.shape[1]

        # Standardise predictors.
        col_means = X_raw.mean(axis=0)
        col_stds = X_raw.std(axis=0)
        col_stds[col_stds < 1e-15] = 1.0  # avoid division by zero
        X_std = (X_raw - col_means) / col_stds

        y_mean = float(np.mean(y))
        y_centered = y - y_mean

        # Precompute column norms.
        col_sq_sums = np.sum(X_std ** 2, axis=0)  # (p,)

        # Initialise coefficients to zero.
        beta = np.zeros(n_feats)
        residual = y_centered.copy()  # residual = y_c - X_std @ beta

        lam = self._reg_lambda * m  # scale penalty by sample size

        for _iteration in range(_LASSO_MAX_ITER):
            beta_old = beta.copy()
            for j in range(n_feats):
                # Partial residual excluding feature j.
                residual += X_std[:, j] * beta[j]
                rho = X_std[:, j] @ residual

                # Soft-thresholding.
                if col_sq_sums[j] < 1e-15:
                    beta[j] = 0.0
                else:
                    beta[j] = (
                        np.sign(rho) * max(abs(rho) - lam, 0.0) / col_sq_sums[j]
                    )
                residual -= X_std[:, j] * beta[j]

            # Check convergence.
            if np.max(np.abs(beta - beta_old)) < _LASSO_TOL:
                break

        # Un-standardise coefficients.
        beta_orig = beta / col_stds
        intercept = y_mean - float(col_means @ beta_orig)

        coeffs = np.empty(n_feats + 1)
        coeffs[0] = intercept
        coeffs[1:] = beta_orig

        X_full = np.column_stack([np.ones(m), X_raw])
        residuals = y - X_full @ coeffs
        sigma2 = float(np.mean(residuals ** 2))
        sigma2 = max(sigma2, _VARIANCE_FLOOR)
        return coeffs, sigma2

    # ------------------------------------------------------------------ #
    #  EM-based scoring for missing data
    # ------------------------------------------------------------------ #

    def _em_local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
        max_iter: int = _EM_MAX_ITER,
        tol: float = _EM_TOL,
    ) -> float:
        """Compute the local BIC score in the presence of missing values.

        Uses an Expectation-Maximisation loop:

        1. **E-step** – impute missing entries using the current estimates
           of the mean vector and covariance matrix restricted to the
           relevant variables (node + parents).
        2. **M-step** – re-estimate sufficient statistics (mean, covariance)
           from the completed data.

        After convergence the local score is computed on the completed
        data using :meth:`_complete_local_score`.

        Parameters
        ----------
        node : int
            Child variable index.
        parents : list[int]
            Parent variable indices.
        data : DataMatrix
            ``(m, p)`` observation matrix (may contain ``NaN``).
        max_iter : int, optional
            Maximum EM iterations.
        tol : float, optional
            Convergence tolerance on relative change in log-likelihood.

        Returns
        -------
        float
            Approximate local BIC score under the EM-estimated parameters.

        Notes
        -----
        The imputation is performed only on the submatrix of columns
        ``[node] + parents``.  Rows that are fully observed contribute
        directly; rows with partial missingness are filled in using
        conditional Gaussian imputation.
        """
        cols = [node] + list(parents)
        sub = data[:, cols].copy()
        m, d = sub.shape

        # ---- Initialise sufficient statistics from observed values ----
        # Column-wise means ignoring NaN.
        mu = np.nanmean(sub, axis=0)
        # Replace NaN with column means for initial covariance estimate.
        init = sub.copy()
        for c in range(d):
            mask_c = np.isnan(init[:, c])
            init[mask_c, c] = mu[c]
        cov = np.cov(init, rowvar=False, bias=True)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        prev_ll = -np.inf

        for _it in range(max_iter):
            # ---------- E-step: impute missing entries -----------------
            imputed = sub.copy()
            for i in range(m):
                nan_mask = np.isnan(sub[i])
                if not np.any(nan_mask):
                    continue
                if np.all(nan_mask):
                    # Entire row missing – fill with mean.
                    imputed[i] = mu
                    continue

                obs_idx = np.where(~nan_mask)[0]
                mis_idx = np.where(nan_mask)[0]

                # Partition covariance.
                cov_oo = cov[np.ix_(obs_idx, obs_idx)]
                cov_mo = cov[np.ix_(mis_idx, obs_idx)]

                try:
                    cov_oo_inv = np.linalg.solve(
                        cov_oo + 1e-10 * np.eye(len(obs_idx)),
                        np.eye(len(obs_idx)),
                    )
                except np.linalg.LinAlgError:
                    cov_oo_inv = np.linalg.pinv(cov_oo)

                # Conditional mean.
                diff_obs = sub[i, obs_idx] - mu[obs_idx]
                imputed[i, mis_idx] = mu[mis_idx] + cov_mo @ cov_oo_inv @ diff_obs

            # ---------- M-step: re-estimate statistics -----------------
            mu = np.mean(imputed, axis=0)
            cov = np.cov(imputed, rowvar=False, bias=True)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])

            # Approximate observed-data log-likelihood (Gaussian).
            var_diag = np.diag(cov)
            var_diag = np.maximum(var_diag, _VARIANCE_FLOOR)
            ll = -0.5 * m * np.sum(np.log(var_diag))

            # Convergence check.
            if prev_ll != -np.inf:
                rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-15)
                if rel_change < tol:
                    break
            prev_ll = ll

        # Build a completed sub-matrix and compute the local score.
        # Construct a temporary data matrix with only the relevant cols.
        data_complete = data.copy()
        data_complete[:, cols] = imputed

        return self._complete_local_score(node, parents, data_complete)

    # ------------------------------------------------------------------ #
    #  Representation helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        parts = [f"penalty_multiplier={self._penalty_multiplier}"]
        if self._regularization != "none":
            parts.append(f"regularization='{self._regularization}'")
            parts.append(f"reg_lambda={self._reg_lambda}")
        return f"BICScore({', '.join(parts)})"
