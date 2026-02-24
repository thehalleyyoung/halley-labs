"""
Multi-width calibration regression for finite-width NTK phase diagrams.

Implements the three-parameter regression model

    Θ(N) = Θ^(0) + Θ^(1) / N + Θ^(2) / N²

where Θ(N) is the empirical NTK at width N, and Θ^(0), Θ^(1), Θ^(2) are
matrix-valued coefficients recovered via (weighted) least squares across
multiple finite widths N_1, …, N_K.

The module provides:
  - DesignMatrixBuilder  – construct and analyse the design matrix
  - CalibrationRegression – unconstrained WLS with full diagnostics
  - ConstrainedRegression – WLS with Θ^(0) fixed from infinite-width theory
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy import stats


# ======================================================================
#  Data containers
# ======================================================================


@dataclass
class RegressionResult:
    """Complete regression output for a single calibration fit.

    Parameters
    ----------
    theta_0 : ndarray
        Estimated infinite-width kernel Θ^(0).  Shape ``(d, d)``.
    theta_1 : ndarray
        First-order finite-width correction Θ^(1).  Shape ``(d, d)``.
    theta_2 : ndarray
        Second-order finite-width correction Θ^(2).  Shape ``(d, d)``.
    residuals : ndarray
        Weighted residual vector (or matrix for element-wise fits).
    r_squared : float
        Coefficient of determination R².
    adjusted_r_squared : float
        Adjusted R² accounting for number of predictors.
    condition_number : float
        Condition number κ(A) of the (possibly weighted) design matrix.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    parameter_std_errors : dict
        Mapping ``{"theta_0": ndarray, "theta_1": ndarray, "theta_2": ndarray}``
        of element-wise standard errors.
    hat_matrix : ndarray
        Hat (projection) matrix H = A (Aᵀ W A)⁻¹ Aᵀ W.
    leverage : ndarray
        Diagonal of H, i.e. h_ii.
    cooks_distance : ndarray
        Cook's distance for each observation.
    loocv_score : float
        Leave-one-out cross-validation mean squared error.
    warnings : list[str]
        Diagnostic warnings (ill-conditioning, influential points, etc.).
    """

    theta_0: NDArray[np.floating]
    theta_1: NDArray[np.floating]
    theta_2: NDArray[np.floating]
    residuals: NDArray[np.floating]
    r_squared: float
    adjusted_r_squared: float
    condition_number: float
    aic: float
    bic: float
    parameter_std_errors: Dict[str, NDArray[np.floating]]
    hat_matrix: NDArray[np.floating]
    leverage: NDArray[np.floating]
    cooks_distance: NDArray[np.floating]
    loocv_score: float
    warnings: List[str] = field(default_factory=list)


# ======================================================================
#  Design-matrix construction and analysis
# ======================================================================


class DesignMatrixBuilder:
    """Build and analyse the Vandermonde-like design matrix in 1/N.

    For a set of widths {N_1, …, N_K} the design matrix is

        A[i, :] = [1,  1/N_i,  1/N_i², …,  1/N_i^p]

    where *p* = ``max_order``.

    Parameters
    ----------
    max_order : int, default 2
        Maximum polynomial order in 1/N.
    """

    def __init__(self, max_order: int = 2) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        self.max_order = max_order

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------

    def build(
        self,
        widths: Sequence[int],
        order: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Build the un-weighted design matrix A.

        Parameters
        ----------
        widths : sequence of int
            Network widths N_1, …, N_K.
        order : int or None
            Polynomial order to use (defaults to ``self.max_order``).

        Returns
        -------
        A : ndarray, shape ``(K, order + 1)``
        """
        p = self.max_order if order is None else order
        widths_arr = np.asarray(widths, dtype=np.float64)
        if np.any(widths_arr <= 0):
            raise ValueError("All widths must be positive")
        inv_widths = 1.0 / widths_arr
        A = np.column_stack([inv_widths ** j for j in range(p + 1)])
        return A

    def build_weighted(
        self,
        widths: Sequence[int],
        weights: NDArray[np.floating],
        order: Optional[int] = None,
    ) -> NDArray[np.floating]:
        r"""Build the weighted design matrix  W^{1/2} A.

        Parameters
        ----------
        widths : sequence of int
            Network widths.
        weights : ndarray, shape ``(K,)``
            Non-negative observation weights.
        order : int or None
            Polynomial order (defaults to ``self.max_order``).

        Returns
        -------
        A_w : ndarray, shape ``(K, order + 1)``
        """
        A = self.build(widths, order=order)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != A.shape[0]:
            raise ValueError("weights length must match number of widths")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        sqrt_w = np.sqrt(w)
        return A * sqrt_w[:, np.newaxis]

    def condition_analysis(
        self,
        A: NDArray[np.floating],
    ) -> Dict[str, Union[float, NDArray[np.floating], bool]]:
        """Analyse conditioning of the design matrix.

        Parameters
        ----------
        A : ndarray, shape ``(K, p+1)``
            Design matrix (possibly weighted).

        Returns
        -------
        info : dict
            ``"condition_number"`` – κ₂(A).
            ``"singular_values"`` – σ_1 ≥ … ≥ σ_{p+1}.
            ``"ill_conditioned"`` – True when κ > 10⁸.
            ``"rank_deficient"`` – True when numerical rank < min(K, p+1).
        """
        sv = la.svdvals(A)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
        numerical_rank = int(np.sum(sv > sv[0] * max(A.shape) * np.finfo(A.dtype).eps))
        expected_rank = min(A.shape)
        return {
            "condition_number": cond,
            "singular_values": sv,
            "ill_conditioned": cond > 1e8,
            "rank_deficient": numerical_rank < expected_rank,
        }

    def suggest_widths(
        self,
        n_widths: int,
        min_width: int,
        max_width: int,
    ) -> NDArray[np.intp]:
        """Suggest calibration widths for a well-conditioned design matrix.

        Uses Chebyshev nodes in the 1/N domain to minimise the maximum
        interpolation error (Runge phenomenon suppression).

        Parameters
        ----------
        n_widths : int
            How many widths to suggest.
        min_width, max_width : int
            Inclusive bounds on the width range.

        Returns
        -------
        widths : ndarray of int, shape ``(n_widths,)``
        """
        if n_widths < 1:
            raise ValueError("n_widths must be >= 1")
        if min_width < 1 or max_width < min_width:
            raise ValueError("Need 1 <= min_width <= max_width")

        # Chebyshev nodes on [1/max_width, 1/min_width]
        a = 1.0 / max_width
        b = 1.0 / min_width
        k = np.arange(1, n_widths + 1)
        nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
            (2 * k - 1) * np.pi / (2 * n_widths)
        )
        # Convert back to widths and round to integers
        widths_float = 1.0 / nodes
        widths_int = np.unique(np.clip(
            np.round(widths_float).astype(np.intp),
            min_width,
            max_width,
        ))

        # If rounding collapsed some nodes, fill in with linspace fallback
        if widths_int.shape[0] < n_widths:
            fallback = np.linspace(min_width, max_width, n_widths, dtype=np.intp)
            widths_int = np.unique(np.concatenate([widths_int, fallback]))[:n_widths]

        return np.sort(widths_int)


# ======================================================================
#  Unconstrained calibration regression
# ======================================================================


class CalibrationRegression:
    r"""Weighted least-squares regression of NTK measurements against 1/N.

    Fits the model

        Θ(N) ≈ Θ^{(0)} + Θ^{(1)} / N + Θ^{(2)} / N²

    via (optionally weighted) least squares.  Full diagnostic statistics
    are produced including leverage, Cook's distance, LOOCV, information
    criteria, and residual-based misspecification tests.

    Parameters
    ----------
    max_order : int, default 2
        Maximum polynomial order in 1/N.
    regularization : float, default 0.0
        Tikhonov (ridge) regularisation parameter λ added to the normal
        equations:  (Aᵀ W A + λ I) x = Aᵀ W b.
    """

    def __init__(
        self,
        max_order: int = 2,
        regularization: float = 0.0,
    ) -> None:
        self.max_order = max_order
        self.regularization = regularization
        self._builder = DesignMatrixBuilder(max_order=max_order)

    # ------------------------------------------------------------------
    #  Main entry points
    # ------------------------------------------------------------------

    def fit(
        self,
        ntk_measurements: Sequence[NDArray[np.floating]],
        widths: Sequence[int],
        weights: Optional[NDArray[np.floating]] = None,
    ) -> RegressionResult:
        """Fit the 3-parameter model to a sequence of NTK matrices.

        Each element of *ntk_measurements* is flattened into a column
        vector and regression is performed column-by-column over the
        observation (width) dimension.

        Parameters
        ----------
        ntk_measurements : list of ndarray, each shape ``(d, d)``
            Empirical NTK matrices at each width.
        widths : sequence of int
            Corresponding network widths.
        weights : ndarray or None
            Observation weights, shape ``(K,)``.

        Returns
        -------
        RegressionResult
        """
        ntk_list = [np.asarray(m, dtype=np.float64) for m in ntk_measurements]
        K = len(ntk_list)
        widths_arr = np.asarray(widths, dtype=np.float64)
        if K != widths_arr.shape[0]:
            raise ValueError("Number of NTK matrices must equal number of widths")

        d = ntk_list[0].shape[0]
        n_entries = d * d

        # Stack flattened NTK matrices: B has shape (K, n_entries)
        B = np.stack([m.ravel() for m in ntk_list], axis=0)

        # Build design matrix
        A = self._builder.build(widths)
        p = A.shape[1]

        # Weight matrix
        W = self._make_weight_matrix(K, weights)

        # Solve for every flattened entry simultaneously
        # X has shape (p, n_entries)
        X, hat_matrix = self._weighted_least_squares(A, B, W)

        # Unpack parameter matrices
        theta_0 = X[0, :].reshape(d, d)
        theta_1 = X[1, :].reshape(d, d) if p > 1 else np.zeros((d, d))
        theta_2 = X[2, :].reshape(d, d) if p > 2 else np.zeros((d, d))

        # Diagnostics
        residuals = self._compute_residuals(A, X, B, W)
        leverage = np.diag(hat_matrix)
        mse = float(np.mean(residuals ** 2))
        cooks = self._compute_cooks_distance(residuals, hat_matrix, p, mse)
        aic, bic = self._compute_information_criteria(residuals, K, p)
        loocv = self._loocv(A, B, W)
        r2, adj_r2 = self._compute_r_squared(B, residuals, K, p, W)
        cond_info = self._builder.condition_analysis(A)
        std_errors = self._compute_std_errors(A, residuals, W, d, p, K)

        warn_list: List[str] = []
        if cond_info["ill_conditioned"]:
            warn_list.append(
                f"Design matrix is ill-conditioned (κ = {cond_info['condition_number']:.2e})"
            )
        if cond_info["rank_deficient"]:
            warn_list.append("Design matrix is numerically rank-deficient")
        if np.any(leverage > 2.0 * p / K):
            idx = np.where(leverage > 2.0 * p / K)[0]
            warn_list.append(f"High-leverage observations at indices {idx.tolist()}")
        if np.any(cooks > 4.0 / K):
            idx = np.where(cooks > 4.0 / K)[0]
            warn_list.append(f"Influential observations (Cook's D) at indices {idx.tolist()}")

        resid_warns = self._residual_analysis(residuals, widths_arr)
        warn_list.extend(resid_warns)

        return RegressionResult(
            theta_0=theta_0,
            theta_1=theta_1,
            theta_2=theta_2,
            residuals=residuals,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            condition_number=cond_info["condition_number"],
            aic=aic,
            bic=bic,
            parameter_std_errors=std_errors,
            hat_matrix=hat_matrix,
            leverage=leverage,
            cooks_distance=cooks,
            loocv_score=loocv,
            warnings=warn_list,
        )

    def fit_elementwise(
        self,
        ntk_measurements: Sequence[NDArray[np.floating]],
        widths: Sequence[int],
        weights: Optional[NDArray[np.floating]] = None,
    ) -> NDArray[np.object_]:
        """Fit each (i, j) entry of the NTK independently.

        Returns a ``(d, d)`` object array of :class:`RegressionResult`,
        one per matrix entry.  This allows per-entry diagnostics at the
        cost of ignoring cross-entry correlations.

        Parameters
        ----------
        ntk_measurements : list of ndarray, each shape ``(d, d)``
        widths : sequence of int
        weights : ndarray or None

        Returns
        -------
        results : ndarray of RegressionResult, shape ``(d, d)``
        """
        ntk_list = [np.asarray(m, dtype=np.float64) for m in ntk_measurements]
        K = len(ntk_list)
        d = ntk_list[0].shape[0]
        widths_arr = np.asarray(widths, dtype=np.float64)

        A = self._builder.build(widths)
        p = A.shape[1]
        W = self._make_weight_matrix(K, weights)

        results = np.empty((d, d), dtype=object)

        for i in range(d):
            for j in range(d):
                b_ij = np.array([m[i, j] for m in ntk_list])[:, np.newaxis]
                X, hat_matrix = self._weighted_least_squares(A, b_ij, W)

                t0 = X[0, 0]
                t1 = X[1, 0] if p > 1 else 0.0
                t2 = X[2, 0] if p > 2 else 0.0

                res = self._compute_residuals(A, X, b_ij, W)
                leverage = np.diag(hat_matrix)
                mse = float(np.mean(res ** 2))
                cooks = self._compute_cooks_distance(res, hat_matrix, p, mse)
                aic, bic = self._compute_information_criteria(res, K, p)
                loocv = self._loocv(A, b_ij, W)
                r2, adj_r2 = self._compute_r_squared(b_ij, res, K, p, W)
                cond_info = self._builder.condition_analysis(A)

                results[i, j] = RegressionResult(
                    theta_0=np.atleast_2d(t0),
                    theta_1=np.atleast_2d(t1),
                    theta_2=np.atleast_2d(t2),
                    residuals=res.ravel(),
                    r_squared=r2,
                    adjusted_r_squared=adj_r2,
                    condition_number=cond_info["condition_number"],
                    aic=aic,
                    bic=bic,
                    parameter_std_errors={
                        "theta_0": np.zeros(1),
                        "theta_1": np.zeros(1),
                        "theta_2": np.zeros(1),
                    },
                    hat_matrix=hat_matrix,
                    leverage=leverage,
                    cooks_distance=cooks,
                    loocv_score=loocv,
                    warnings=[],
                )

        return results

    # ------------------------------------------------------------------
    #  Core linear-algebra routines
    # ------------------------------------------------------------------

    def _weighted_least_squares(
        self,
        A: NDArray[np.floating],
        b: NDArray[np.floating],
        W: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        r"""Solve the weighted least-squares problem.

        Minimises  \|W^{1/2}(Ax - b)\|_2^2 + λ \|x\|_2^2  via the
        normal equations

            (Aᵀ W A + λ I) x = Aᵀ W b.

        Parameters
        ----------
        A : ndarray, shape ``(K, p)``
        b : ndarray, shape ``(K, m)``
        W : ndarray, shape ``(K, K)``
            Diagonal weight matrix.

        Returns
        -------
        x : ndarray, shape ``(p, m)``
            Solution coefficients.
        hat_matrix : ndarray, shape ``(K, K)``
            Hat matrix H = A (Aᵀ W A + λI)⁻¹ Aᵀ W.
        """
        K, p = A.shape
        AtW = A.T @ W                          # (p, K)
        AtWA = AtW @ A                         # (p, p)

        # Tikhonov regularisation
        if self.regularization > 0:
            AtWA += self.regularization * np.eye(p)

        # Use Cholesky when possible; fall back to pseudo-inverse
        try:
            L = la.cholesky(AtWA, lower=True)
            # Solve (L Lᵀ) X = Aᵀ W b
            x = la.cho_solve((L, True), AtW @ b)
            # Inverse for hat matrix: (Aᵀ W A)⁻¹
            AtWA_inv = la.cho_solve((L, True), np.eye(p))
        except la.LinAlgError:
            # Singular or near-singular – use pseudo-inverse
            AtWA_inv = la.pinvh(AtWA)
            x = AtWA_inv @ (AtW @ b)

        hat_matrix = A @ AtWA_inv @ AtW         # (K, K)
        return x, hat_matrix

    def _compute_residuals(
        self,
        A: NDArray[np.floating],
        x: NDArray[np.floating],
        b: NDArray[np.floating],
        W: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute weighted residuals  W^{1/2} (b - A x).

        Parameters
        ----------
        A : (K, p), x : (p, m), b : (K, m), W : (K, K)

        Returns
        -------
        residuals : ndarray, shape ``(K, m)``
        """
        raw = b - A @ x
        sqrt_W = np.sqrt(np.diag(W))[:, np.newaxis]
        return raw * sqrt_W

    def _compute_hat_matrix(
        self,
        A: NDArray[np.floating],
        W: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        r"""Hat matrix  H = A (Aᵀ W A)^{-1} Aᵀ W.

        Parameters
        ----------
        A : (K, p), W : (K, K)

        Returns
        -------
        H : ndarray, shape ``(K, K)``
        """
        p = A.shape[1]
        AtW = A.T @ W
        AtWA = AtW @ A
        if self.regularization > 0:
            AtWA += self.regularization * np.eye(p)
        try:
            AtWA_inv = la.inv(AtWA)
        except la.LinAlgError:
            AtWA_inv = la.pinvh(AtWA)
        return A @ AtWA_inv @ AtW

    def _compute_cooks_distance(
        self,
        residuals: NDArray[np.floating],
        hat_matrix: NDArray[np.floating],
        p: int,
        mse: float,
    ) -> NDArray[np.floating]:
        """Cook's distance for each observation.

        .. math::

            D_i = \\frac{e_i^2}{p \\, \\text{MSE}} \\cdot
                  \\frac{h_{ii}}{(1 - h_{ii})^2}

        Parameters
        ----------
        residuals : (K,) or (K, m)
        hat_matrix : (K, K)
        p : int – number of parameters
        mse : float – mean squared error

        Returns
        -------
        cooks_d : ndarray, shape ``(K,)``
        """
        h = np.diag(hat_matrix)
        if residuals.ndim == 2:
            e2 = np.mean(residuals ** 2, axis=1)
        else:
            e2 = residuals ** 2

        denom = p * mse * (1.0 - h) ** 2
        # Guard against division by zero for high-leverage points
        safe_denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        return e2 * h / safe_denom

    def _compute_information_criteria(
        self,
        residuals: NDArray[np.floating],
        n: int,
        p: int,
    ) -> Tuple[float, float]:
        """Akaike (AIC) and Bayesian (BIC) information criteria.

        Based on the Gaussian log-likelihood with unknown variance:

        .. math::

            \\text{AIC} = n \\ln(\\text{RSS}/n) + 2p

            \\text{BIC} = n \\ln(\\text{RSS}/n) + p \\ln(n)

        Parameters
        ----------
        residuals : ndarray
        n : int – number of observations
        p : int – number of parameters

        Returns
        -------
        aic, bic : float
        """
        rss = float(np.sum(residuals ** 2))
        if rss <= 0:
            rss = 1e-30
        log_lik_term = n * np.log(rss / n)
        aic = float(log_lik_term + 2.0 * p)
        bic = float(log_lik_term + p * np.log(n))
        return aic, bic

    def _loocv(
        self,
        A: NDArray[np.floating],
        b: NDArray[np.floating],
        W: NDArray[np.floating],
    ) -> float:
        r"""Leave-one-out cross-validation via the hat-matrix shortcut.

        The LOOCV prediction error for observation *i* is

        .. math::

            \tilde{e}_i = \frac{e_i}{1 - h_{ii}}

        where e_i is the ordinary residual and h_{ii} the leverage.

        Parameters
        ----------
        A : (K, p), b : (K, m), W : (K, K)

        Returns
        -------
        loocv_mse : float
            Mean squared LOOCV error averaged over observations and entries.
        """
        x, H = self._weighted_least_squares(A, b, W)
        raw_residuals = b - A @ x
        h = np.diag(H)
        # Avoid division by zero for perfect-leverage points
        safe_h = np.clip(h, 0.0, 1.0 - 1e-12)
        loocv_errors = raw_residuals / (1.0 - safe_h)[:, np.newaxis]
        return float(np.mean(loocv_errors ** 2))

    # ------------------------------------------------------------------
    #  R² computation
    # ------------------------------------------------------------------

    def _compute_r_squared(
        self,
        b: NDArray[np.floating],
        residuals: NDArray[np.floating],
        n: int,
        p: int,
        W: NDArray[np.floating],
    ) -> Tuple[float, float]:
        """Compute R² and adjusted R².

        Parameters
        ----------
        b : (K, m) – observed values
        residuals : (K, m) – weighted residuals
        n : int – observations, p : int – parameters
        W : (K, K) – weight matrix

        Returns
        -------
        r2, adj_r2 : float
        """
        w_diag = np.diag(W)[:, np.newaxis]
        b_weighted = b * np.sqrt(w_diag)
        ss_res = float(np.sum(residuals ** 2))
        mean_b = np.average(b, axis=0, weights=np.diag(W))
        ss_tot = float(np.sum(w_diag * (b - mean_b[np.newaxis, :]) ** 2))

        if ss_tot < 1e-30:
            r2 = 1.0 if ss_res < 1e-30 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        if n - p <= 0:
            adj_r2 = r2
        else:
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)

        return float(r2), float(adj_r2)

    # ------------------------------------------------------------------
    #  Standard errors
    # ------------------------------------------------------------------

    def _compute_std_errors(
        self,
        A: NDArray[np.floating],
        residuals: NDArray[np.floating],
        W: NDArray[np.floating],
        d: int,
        p: int,
        K: int,
    ) -> Dict[str, NDArray[np.floating]]:
        """Parameter standard errors from the sandwich estimator.

        Parameters
        ----------
        A : (K, p), residuals : (K, m), W : (K, K), d : int, p : int, K : int

        Returns
        -------
        std_errors : dict with keys ``"theta_0"``, ``"theta_1"``, ``"theta_2"``
        """
        AtW = A.T @ W
        AtWA = AtW @ A
        if self.regularization > 0:
            AtWA += self.regularization * np.eye(p)

        try:
            AtWA_inv = la.inv(AtWA)
        except la.LinAlgError:
            AtWA_inv = la.pinvh(AtWA)

        # Estimate σ² from residuals (per entry, then average)
        dof = max(K - p, 1)
        sigma2 = np.sum(residuals ** 2, axis=0) / dof  # (m,)

        # Variance of each parameter: Var(x_j) = (AtWA_inv)_{jj} * σ²
        se_dict: Dict[str, NDArray[np.floating]] = {}
        names = ["theta_0", "theta_1", "theta_2"]
        for j, name in enumerate(names[:p]):
            var_j = AtWA_inv[j, j] * sigma2  # (m,)
            se_dict[name] = np.sqrt(np.maximum(var_j, 0.0)).reshape(d, d)

        # Pad with zeros if order < 2
        for name in names[p:]:
            se_dict[name] = np.zeros((d, d))

        return se_dict

    # ------------------------------------------------------------------
    #  Residual diagnostics
    # ------------------------------------------------------------------

    def _residual_analysis(
        self,
        residuals: NDArray[np.floating],
        widths: NDArray[np.floating],
    ) -> List[str]:
        """Check residuals for systematic patterns.

        Tests performed:
        1. Breusch-Pagan-style heteroscedasticity check (residual variance
           correlated with 1/N).
        2. Runs test for a non-random sign pattern (trend detection).

        Parameters
        ----------
        residuals : (K, m) or (K,)
        widths : (K,)

        Returns
        -------
        warnings : list of str
        """
        warn_list: List[str] = []
        if residuals.ndim == 2:
            e = np.mean(residuals ** 2, axis=1)
        else:
            e = residuals ** 2

        K = e.shape[0]
        if K < 4:
            return warn_list

        # 1. Heteroscedasticity – rank correlation of e² with 1/N
        inv_w = 1.0 / widths
        corr, pval = stats.spearmanr(inv_w, e)
        if pval < 0.05:
            warn_list.append(
                f"Possible heteroscedasticity (Spearman ρ={corr:.3f}, p={pval:.3f})"
            )

        # 2. Runs test – sign pattern of mean residual
        if residuals.ndim == 2:
            mean_res = np.mean(residuals, axis=1)
        else:
            mean_res = residuals

        signs = np.sign(mean_res)
        signs = signs[signs != 0]
        if signs.shape[0] >= 4:
            n_runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
            n_pos = int(np.sum(signs > 0))
            n_neg = int(np.sum(signs < 0))
            n_total = n_pos + n_neg
            if n_pos > 0 and n_neg > 0:
                expected_runs = 1.0 + 2.0 * n_pos * n_neg / n_total
                var_runs = (
                    2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_total)
                    / (n_total ** 2 * (n_total - 1))
                )
                if var_runs > 0:
                    z_runs = (n_runs - expected_runs) / np.sqrt(var_runs)
                    if abs(z_runs) > 1.96:
                        warn_list.append(
                            f"Non-random residual pattern (runs test z={z_runs:.2f})"
                        )

        return warn_list

    def _detect_misspecification(
        self,
        residuals: NDArray[np.floating],
        widths: NDArray[np.floating],
        ntk_measurements: Sequence[NDArray[np.floating]],
    ) -> Dict[str, Union[float, bool]]:
        """Ramsey RESET-like test for model misspecification.

        Augments the original regression with powers of the fitted values
        and tests whether the augmented model significantly improves fit
        via an F-test.

        Parameters
        ----------
        residuals : (K, m)
        widths : (K,)
        ntk_measurements : list of (d, d) arrays

        Returns
        -------
        result : dict
            ``"f_statistic"`` – F-test statistic.
            ``"p_value"`` – p-value of the test.
            ``"misspecified"`` – True if p < 0.05.
        """
        ntk_list = [np.asarray(m, dtype=np.float64) for m in ntk_measurements]
        K = len(ntk_list)
        B = np.stack([m.ravel() for m in ntk_list], axis=0)  # (K, m)
        m = B.shape[1]

        A = self._builder.build(widths)
        p_orig = A.shape[1]

        W = np.eye(K)
        x_orig, _ = self._weighted_least_squares(A, B, W)
        fitted = A @ x_orig  # (K, m)

        # Augment with squared and cubed fitted values (mean across entries)
        fitted_mean = np.mean(fitted, axis=1, keepdims=True)
        A_aug = np.column_stack([A, fitted_mean ** 2, fitted_mean ** 3])
        p_aug = A_aug.shape[1]

        x_aug, _ = self._weighted_least_squares(A_aug, B, W)
        res_aug = B - A_aug @ x_aug

        rss_orig = float(np.sum(residuals ** 2))
        rss_aug = float(np.sum(res_aug ** 2))

        df_num = p_aug - p_orig
        df_den = K * m - p_aug
        if df_den <= 0 or df_num <= 0:
            return {"f_statistic": 0.0, "p_value": 1.0, "misspecified": False}

        f_stat = ((rss_orig - rss_aug) / df_num) / (rss_aug / df_den)
        p_value = float(1.0 - stats.f.cdf(f_stat, df_num, df_den))

        return {
            "f_statistic": float(f_stat),
            "p_value": p_value,
            "misspecified": p_value < 0.05,
        }

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_weight_matrix(
        K: int,
        weights: Optional[NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Build diagonal weight matrix W.

        Parameters
        ----------
        K : int – number of observations
        weights : (K,) or None

        Returns
        -------
        W : (K, K) diagonal matrix
        """
        if weights is None:
            return np.eye(K)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (K,):
            raise ValueError(f"weights must have shape ({K},), got {w.shape}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        return np.diag(w)


# ======================================================================
#  Constrained regression (Θ^(0) fixed)
# ======================================================================


class ConstrainedRegression:
    r"""Regression with the leading-order term Θ^{(0)} fixed.

    When the infinite-width NTK is known analytically, we can fix
    Θ^{(0)} and estimate only Θ^{(1)} and Θ^{(2)} from finite-width
    measurements.  This reduces the parameter count and can stabilise
    estimation at moderate numbers of calibration widths.

    Parameters
    ----------
    theta_0_fixed : ndarray or None
        If provided, fixes Θ^{(0)} for all subsequent calls to
        :meth:`fit`.
    regularization : float, default 0.0
        Tikhonov regularisation parameter.
    """

    def __init__(
        self,
        theta_0_fixed: Optional[NDArray[np.floating]] = None,
        regularization: float = 0.0,
    ) -> None:
        self.theta_0_fixed = (
            np.asarray(theta_0_fixed, dtype=np.float64)
            if theta_0_fixed is not None
            else None
        )
        self.regularization = regularization
        self._builder = DesignMatrixBuilder(max_order=2)

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        ntk_measurements: Sequence[NDArray[np.floating]],
        widths: Sequence[int],
        theta_0: Optional[NDArray[np.floating]] = None,
        weights: Optional[NDArray[np.floating]] = None,
    ) -> RegressionResult:
        r"""Fit with Θ^{(0)} fixed.

        The model becomes

            Θ(N) - Θ^{(0)} = Θ^{(1)} / N + Θ^{(2)} / N²

        which is a two-parameter regression in 1/N and 1/N².

        Parameters
        ----------
        ntk_measurements : list of ndarray, each (d, d)
        widths : sequence of int
        theta_0 : ndarray or None
            Override for the fixed Θ^{(0)}.  Falls back to
            ``self.theta_0_fixed``.
        weights : ndarray or None

        Returns
        -------
        RegressionResult
        """
        t0 = theta_0 if theta_0 is not None else self.theta_0_fixed
        if t0 is None:
            raise ValueError(
                "theta_0 must be provided either at construction or call time"
            )
        t0 = np.asarray(t0, dtype=np.float64)

        ntk_list = [np.asarray(m, dtype=np.float64) for m in ntk_measurements]
        K = len(ntk_list)
        d = ntk_list[0].shape[0]
        n_entries = d * d
        widths_arr = np.asarray(widths, dtype=np.float64)

        B = np.stack([m.ravel() for m in ntk_list], axis=0)  # (K, m)

        A_reduced, b_reduced = self._build_constrained_system(
            self._builder.build(widths), B, t0,
        )
        p = A_reduced.shape[1]
        W = CalibrationRegression._make_weight_matrix(K, weights)

        # Solve reduced system
        reg = CalibrationRegression(max_order=2, regularization=self.regularization)
        X, hat_matrix = reg._weighted_least_squares(A_reduced, b_reduced, W)

        theta_1 = X[0, :].reshape(d, d)
        theta_2 = X[1, :].reshape(d, d) if p > 1 else np.zeros((d, d))

        residuals = reg._compute_residuals(A_reduced, X, b_reduced, W)
        leverage = np.diag(hat_matrix)
        mse = float(np.mean(residuals ** 2))
        cooks = reg._compute_cooks_distance(residuals, hat_matrix, p, mse)
        aic, bic = reg._compute_information_criteria(residuals, K, p)
        loocv = reg._loocv(A_reduced, b_reduced, W)
        r2, adj_r2 = reg._compute_r_squared(b_reduced, residuals, K, p, W)
        cond_info = self._builder.condition_analysis(A_reduced)

        se_dict = self._compute_constrained_std_errors(
            A_reduced, residuals, W, d, p, K,
        )

        warn_list: List[str] = []
        if cond_info["ill_conditioned"]:
            warn_list.append(
                f"Reduced design matrix ill-conditioned (κ = {cond_info['condition_number']:.2e})"
            )

        return RegressionResult(
            theta_0=t0.copy(),
            theta_1=theta_1,
            theta_2=theta_2,
            residuals=residuals,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            condition_number=cond_info["condition_number"],
            aic=aic,
            bic=bic,
            parameter_std_errors=se_dict,
            hat_matrix=hat_matrix,
            leverage=leverage,
            cooks_distance=cooks,
            loocv_score=loocv,
            warnings=warn_list,
        )

    # ------------------------------------------------------------------
    #  Constrained system builder
    # ------------------------------------------------------------------

    def _build_constrained_system(
        self,
        A: NDArray[np.floating],
        b: NDArray[np.floating],
        theta_0: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        r"""Reformulate the regression with Θ^{(0)} absorbed.

        Original model:  b = A x  where  x = [Θ^{(0)}, Θ^{(1)}, Θ^{(2)}]ᵀ.
        After fixing Θ^{(0)}:

            b - A[:,0] ⊗ Θ^{(0)} = A[:,1:] · [Θ^{(1)}, Θ^{(2)}]ᵀ

        Parameters
        ----------
        A : (K, p) – full design matrix (including intercept column)
        b : (K, m) – flattened NTK observations
        theta_0 : (d, d) – fixed leading-order term

        Returns
        -------
        A_reduced : (K, p-1)
        b_reduced : (K, m)
        """
        t0_flat = theta_0.ravel()[np.newaxis, :]  # (1, m)
        # Subtract the intercept contribution
        b_reduced = b - A[:, 0:1] * t0_flat
        A_reduced = A[:, 1:]
        return A_reduced, b_reduced

    # ------------------------------------------------------------------
    #  Standard errors for constrained fit
    # ------------------------------------------------------------------

    def _compute_constrained_std_errors(
        self,
        A_reduced: NDArray[np.floating],
        residuals: NDArray[np.floating],
        W: NDArray[np.floating],
        d: int,
        p: int,
        K: int,
    ) -> Dict[str, NDArray[np.floating]]:
        """Standard errors for the constrained parameters.

        Parameters
        ----------
        A_reduced : (K, p), residuals : (K, m), W : (K, K)

        Returns
        -------
        se_dict
        """
        AtW = A_reduced.T @ W
        AtWA = AtW @ A_reduced
        if self.regularization > 0:
            AtWA += self.regularization * np.eye(p)

        try:
            AtWA_inv = la.inv(AtWA)
        except la.LinAlgError:
            AtWA_inv = la.pinvh(AtWA)

        dof = max(K - p, 1)
        sigma2 = np.sum(residuals ** 2, axis=0) / dof

        se_dict: Dict[str, NDArray[np.floating]] = {
            "theta_0": np.zeros((d, d)),
        }
        param_names = ["theta_1", "theta_2"]
        for j, name in enumerate(param_names[:p]):
            var_j = AtWA_inv[j, j] * sigma2
            se_dict[name] = np.sqrt(np.maximum(var_j, 0.0)).reshape(d, d)
        for name in param_names[p:]:
            se_dict[name] = np.zeros((d, d))

        return se_dict

    # ------------------------------------------------------------------
    #  Model comparison
    # ------------------------------------------------------------------

    def compare_constrained_unconstrained(
        self,
        ntk_measurements: Sequence[NDArray[np.floating]],
        widths: Sequence[int],
        theta_0: Optional[NDArray[np.floating]] = None,
        weights: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, Union[float, bool, RegressionResult]]:
        r"""F-test comparing constrained vs unconstrained models.

        The test statistic is

        .. math::

            F = \frac{(\text{RSS}_c - \text{RSS}_u) / q}
                {\text{RSS}_u / (n - p_u)}

        where  q = p_u - p_c  is the number of restrictions, and
        RSS_c, RSS_u  are residual sums of squares for the constrained
        and unconstrained fits respectively.

        Parameters
        ----------
        ntk_measurements : list of (d, d) arrays
        widths : sequence of int
        theta_0 : ndarray or None
        weights : ndarray or None

        Returns
        -------
        comparison : dict
            ``"unconstrained"`` – full :class:`RegressionResult`.
            ``"constrained"`` – constrained :class:`RegressionResult`.
            ``"f_statistic"`` – F-test statistic.
            ``"p_value"`` – p-value.
            ``"prefer_constrained"`` – True if the constraint is not
            rejected at the 5 % level.
            ``"rss_unconstrained"`` – RSS of unconstrained model.
            ``"rss_constrained"`` – RSS of constrained model.
        """
        t0 = theta_0 if theta_0 is not None else self.theta_0_fixed
        if t0 is None:
            raise ValueError("theta_0 required for comparison")

        unconstrained_reg = CalibrationRegression(
            max_order=2, regularization=self.regularization,
        )
        result_u = unconstrained_reg.fit(ntk_measurements, widths, weights)
        result_c = self.fit(ntk_measurements, widths, theta_0=t0, weights=weights)

        rss_u = float(np.sum(result_u.residuals ** 2))
        rss_c = float(np.sum(result_c.residuals ** 2))

        ntk_list = [np.asarray(m, dtype=np.float64) for m in ntk_measurements]
        K = len(ntk_list)
        m = ntk_list[0].size
        p_u = self._builder.max_order + 1  # unconstrained parameters
        p_c = self._builder.max_order      # constrained parameters
        q = p_u - p_c

        df_den = K * m - p_u
        if df_den <= 0 or q <= 0:
            return {
                "unconstrained": result_u,
                "constrained": result_c,
                "f_statistic": 0.0,
                "p_value": 1.0,
                "prefer_constrained": True,
                "rss_unconstrained": rss_u,
                "rss_constrained": rss_c,
            }

        f_stat = ((rss_c - rss_u) / q) / (rss_u / df_den)
        f_stat = max(f_stat, 0.0)
        p_value = float(1.0 - stats.f.cdf(f_stat, q, df_den))

        return {
            "unconstrained": result_u,
            "constrained": result_c,
            "f_statistic": float(f_stat),
            "p_value": p_value,
            "prefer_constrained": p_value >= 0.05,
            "rss_unconstrained": rss_u,
            "rss_constrained": rss_c,
        }
