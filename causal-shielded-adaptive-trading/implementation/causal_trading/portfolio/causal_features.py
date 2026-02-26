"""
Causal feature selection for return prediction.

Implements a two-stage pipeline:

1. **Feature pre-selection** – reduce ~500 raw features to ~30 using either:
   - HSIC-Lasso (default): kernel-based nonlinear feature importance,
     consistent with the additive noise model (ANM) causal framework.
   - Linear LASSO (fallback): L1-penalised regression, retained for
     ablation studies and computational comparison.
2. **Invariant feature identification** – keep only features whose
   predictive relationship with the target is *stable* across regimes,
   as determined by results from the Structural Causal Invariance Test
   (SCIT).

The HSIC-Lasso formulation (Yamada et al. 2014) solves:
    min_beta  (1/2) ||HSIC_yy - sum_j beta_j HSIC_jy||^2 + lambda ||beta||_1
    s.t. beta_j >= 0
which identifies features with nonlinear statistical dependence on the
target, consistent with the ANM causal model class.

The selector exposes both the invariant set and regime-specific subsets,
enabling downstream models to trade off robustness vs. regime-conditional
alpha.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureSelectionResult:
    """Output of the full selection pipeline."""
    selected_features: List[int]
    invariant_features: List[int]
    regime_specific_features: Dict[int, List[int]]
    feature_importance: NDArray  # (n_features,)
    lasso_coefficients: NDArray  # (n_features,)  linear LASSO coefficients
    stability_scores: NDArray   # (n_features,)
    cross_val_scores: List[float]
    n_raw: int
    n_after_lasso: int
    n_invariant: int
    hsic_scores: Optional[NDArray] = None  # (n_features,)  HSIC-Lasso scores


@dataclass
class FeatureStabilityReport:
    """Per-feature stability analysis."""
    feature_index: int
    mean_coefficient: float
    std_coefficient: float
    cv_ratio: float  # std / |mean|
    regime_coefficients: Dict[int, float]
    is_invariant: bool


# ---------------------------------------------------------------------------
# LASSO helpers
# ---------------------------------------------------------------------------

def _soft_threshold(x: float, lam: float) -> float:
    """Proximal operator for L1 penalty."""
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    return 0.0


def _coordinate_descent_lasso(
    X: NDArray,
    y: NDArray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    warm_start: Optional[NDArray] = None,
) -> NDArray:
    """Solve LASSO via coordinate descent.

    Minimise  (1/2n) ||y - Xβ||² + α ||β||₁
    """
    n, p = X.shape
    beta = warm_start.copy() if warm_start is not None else np.zeros(p)
    residual = y - X @ beta

    # Pre-compute column norms
    col_norms_sq = np.sum(X ** 2, axis=0) / n

    for iteration in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if col_norms_sq[j] < 1e-15:
                continue
            # Partial residual
            residual += X[:, j] * beta[j]
            rho = X[:, j] @ residual / n
            beta[j] = _soft_threshold(rho, alpha) / col_norms_sq[j]
            residual -= X[:, j] * beta[j]

        if np.max(np.abs(beta - beta_old)) < tol:
            logger.debug("LASSO converged at iteration %d", iteration + 1)
            break
    else:
        logger.debug("LASSO reached max_iter=%d without full convergence", max_iter)

    return beta


def _compute_lasso_path(
    X: NDArray,
    y: NDArray,
    alphas: NDArray,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> NDArray:
    """Compute LASSO solution path over a sequence of regularisation strengths.

    Returns (len(alphas), p) coefficient matrix.
    """
    n, p = X.shape
    coefs = np.zeros((len(alphas), p))
    warm = np.zeros(p)
    for i, alpha in enumerate(alphas):
        warm = _coordinate_descent_lasso(X, y, alpha, max_iter, tol, warm)
        coefs[i] = warm.copy()
    return coefs


# ---------------------------------------------------------------------------
# HSIC-Lasso: nonlinear feature selection (Yamada et al. 2014)
# ---------------------------------------------------------------------------

def _gaussian_kernel_matrix(
    X: NDArray, bandwidth: Optional[float] = None
) -> NDArray:
    """Compute centered Gaussian kernel matrix with median-heuristic bandwidth."""
    from scipy.spatial.distance import cdist, pdist
    X = np.atleast_2d(X)
    sq = cdist(X, X, "sqeuclidean")
    if bandwidth is None:
        pw = pdist(X, "sqeuclidean")
        med = np.median(pw) if len(pw) > 0 else 1.0
        bandwidth = np.sqrt(med / 2.0) if med > 0 else 1.0
    K = np.exp(-sq / (2.0 * bandwidth ** 2))
    # Center the kernel matrix
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic_lasso_scores(
    X: NDArray,
    y: NDArray,
    alpha: float = 0.01,
    max_iter: int = 500,
    tol: float = 1e-5,
    n_subsample: int = 500,
) -> NDArray:
    """Compute HSIC-Lasso feature importance scores.

    Solves:  min_β  (1/2)||L̃ - Σ_j β_j K̃_j||² + α||β||₁
             s.t.  β_j ≥ 0

    where K̃_j is the (normalized, centered) kernel matrix for feature j
    and L̃ is the kernel matrix for the target y.

    Parameters
    ----------
    X : (n, p) array
        Feature matrix.
    y : (n,) array
        Target vector.
    alpha : float
        L1 regularization strength.
    max_iter : int
        Maximum iterations for coordinate descent.
    tol : float
        Convergence tolerance.
    n_subsample : int
        Subsample size for computational tractability.

    Returns
    -------
    scores : (p,) array
        Non-negative importance scores (higher = more important).
    """
    n, p = X.shape

    # Subsample for tractability if needed
    if n > n_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, n_subsample, replace=False)
        X = X[idx]
        y = y[idx]
        n = n_subsample

    # Compute target kernel
    L = _gaussian_kernel_matrix(y.reshape(-1, 1))
    # Vectorize (upper triangle) for regression formulation
    triu_idx = np.triu_indices(n, k=1)
    l_vec = L[triu_idx]
    m = len(l_vec)

    # Compute feature kernels and vectorize
    K_vecs = np.zeros((p, m))
    for j in range(p):
        Kj = _gaussian_kernel_matrix(X[:, j:j+1])
        K_vecs[j] = Kj[triu_idx]

    # Normalize
    for j in range(p):
        norm_j = np.linalg.norm(K_vecs[j])
        if norm_j > 1e-12:
            K_vecs[j] /= norm_j
    l_norm = np.linalg.norm(l_vec)
    if l_norm > 1e-12:
        l_vec = l_vec / l_norm

    # Non-negative LASSO via coordinate descent
    # min (1/2)||l_vec - K_vecs.T @ β||² + α||β||₁, β >= 0
    beta = np.zeros(p)
    KtL = K_vecs @ l_vec  # (p,) correlations
    KtK = K_vecs @ K_vecs.T  # (p, p) Gram matrix
    diag_KtK = np.diag(KtK)

    for iteration in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if diag_KtK[j] < 1e-15:
                continue
            residual_j = KtL[j] - KtK[j] @ beta + diag_KtK[j] * beta[j]
            # Soft thresholding with non-negativity
            beta[j] = max(0.0, _soft_threshold(residual_j, alpha)) / diag_KtK[j]

        if np.max(np.abs(beta - beta_old)) < tol:
            logger.debug("HSIC-Lasso converged at iteration %d", iteration + 1)
            break

    return beta


def _cross_validate_hsic_lasso(
    X: NDArray,
    y: NDArray,
    alphas: NDArray,
    folds: List[Tuple[NDArray, NDArray]],
    n_subsample: int = 500,
) -> Tuple[float, NDArray]:
    """Cross-validate HSIC-Lasso regularization strength.

    Uses kernel ridge regression prediction error as the CV criterion.

    Returns
    -------
    best_alpha : float
    scores_at_best : (p,) array
    """
    n_alphas = len(alphas)
    cv_errors = np.zeros(n_alphas)

    for train_idx, val_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        for i, alpha in enumerate(alphas):
            scores = _hsic_lasso_scores(X_tr, y_tr, alpha=alpha, n_subsample=n_subsample)
            # Select features with nonzero score
            selected = np.where(scores > 1e-10)[0]
            if len(selected) == 0:
                cv_errors[i] += np.mean(y_va ** 2)
                continue
            # Kernel ridge regression with selected features
            X_sel_tr = X_tr[:, selected]
            X_sel_va = X_va[:, selected]
            from scipy.spatial.distance import cdist
            bw = np.median(cdist(X_sel_tr, X_sel_tr, "euclidean")) + 1e-10
            K_tr = np.exp(-cdist(X_sel_tr, X_sel_tr, "sqeuclidean") / (2 * bw ** 2))
            K_va = np.exp(-cdist(X_sel_va, X_sel_tr, "sqeuclidean") / (2 * bw ** 2))
            alpha_reg = 1e-3 * K_tr.shape[0]
            try:
                w = np.linalg.solve(K_tr + alpha_reg * np.eye(K_tr.shape[0]), y_tr)
                pred = K_va @ w
                cv_errors[i] += np.mean((y_va - pred) ** 2)
            except np.linalg.LinAlgError:
                cv_errors[i] += np.mean(y_va ** 2)

    cv_errors /= len(folds)
    best_idx = int(np.argmin(cv_errors))
    best_scores = _hsic_lasso_scores(X, y, alpha=alphas[best_idx], n_subsample=n_subsample)

    logger.info(
        "Best HSIC-Lasso α=%.4e (CV error=%.4f, %d features selected)",
        alphas[best_idx], cv_errors[best_idx],
        int(np.sum(best_scores > 1e-10)),
    )
    return float(alphas[best_idx]), best_scores


# ---------------------------------------------------------------------------
# Regime-aware cross-validation
# ---------------------------------------------------------------------------

def _regime_aware_folds(
    regime_labels: NDArray, n_folds: int = 5
) -> List[Tuple[NDArray, NDArray]]:
    """Stratified folds ensuring each fold has proportional regime coverage.

    Parameters
    ----------
    regime_labels : (T,) int array
        Regime assignment for each time step.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    List of (train_idx, val_idx) tuples.
    """
    unique_regimes = np.unique(regime_labels)
    indices_per_regime: Dict[int, NDArray] = {}
    for r in unique_regimes:
        idx = np.where(regime_labels == r)[0]
        np.random.shuffle(idx)
        indices_per_regime[int(r)] = idx

    folds: List[Tuple[List[int], List[int]]] = [
        ([], []) for _ in range(n_folds)
    ]
    for r, idx in indices_per_regime.items():
        splits = np.array_split(idx, n_folds)
        for fold_i in range(n_folds):
            val_part = splits[fold_i].tolist()
            train_part = np.concatenate(
                [splits[j] for j in range(n_folds) if j != fold_i]
            ).tolist()
            folds[fold_i][0].extend(train_part)
            folds[fold_i][1].extend(val_part)

    return [
        (np.array(sorted(tr)), np.array(sorted(va))) for tr, va in folds
    ]


# ---------------------------------------------------------------------------
# Core selector
# ---------------------------------------------------------------------------

class CausalFeatureSelector:
    """Two-stage causal feature selection.

    Parameters
    ----------
    n_lasso_features : int
        Target number of features after pre-selection.
    stability_threshold : float
        Maximum coefficient-of-variation (σ/|μ|) for a feature to be
        deemed invariant across regimes.
    n_alphas : int
        Number of regularisation strengths in the path.
    n_folds : int
        Number of CV folds.
    alpha_min_ratio : float
        Ratio of smallest to largest α on the regularization path.
    max_iter : int
        Maximum coordinate-descent iterations.
    method : str
        Feature selection method: 'hsic-lasso' (default, nonlinear) or
        'lasso' (linear, for ablation). HSIC-Lasso is consistent with
        the ANM causal model class; linear LASSO is retained for
        computational comparison.
    """

    def __init__(
        self,
        n_lasso_features: int = 30,
        stability_threshold: float = 0.5,
        n_alphas: int = 50,
        n_folds: int = 5,
        alpha_min_ratio: float = 1e-3,
        max_iter: int = 2000,
        method: str = "hsic-lasso",
    ) -> None:
        self.n_lasso_features = n_lasso_features
        self.stability_threshold = stability_threshold
        self.n_alphas = n_alphas
        self.n_folds = n_folds
        self.alpha_min_ratio = alpha_min_ratio
        self.max_iter = max_iter
        if method not in ("hsic-lasso", "lasso"):
            raise ValueError(f"method must be 'hsic-lasso' or 'lasso', got '{method}'")
        self.method = method

        self._result: Optional[FeatureSelectionResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        data: NDArray,
        target: NDArray,
        regime_labels: Optional[NDArray] = None,
    ) -> FeatureSelectionResult:
        """Run the full two-stage selection pipeline.

        Parameters
        ----------
        data : (T, p) array
            Raw feature matrix.
        target : (T,) array
            Prediction target (e.g. forward returns).
        regime_labels : (T,) int array or None
            Regime assignments.  If ``None``, all data is treated as a
            single regime and only feature selection is performed.

        Returns
        -------
        FeatureSelectionResult
        """
        data = np.asarray(data, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        T, p = data.shape
        assert target.shape == (T,), "target length must match data rows"

        # Standardise
        X, mu_x, sigma_x = self._standardise(data)
        y, mu_y, sigma_y = self._standardise_1d(target)

        # Build CV folds
        if regime_labels is not None:
            regime_labels = np.asarray(regime_labels, dtype=int)
            folds = _regime_aware_folds(regime_labels, self.n_folds)
        else:
            folds = self._simple_folds(T)

        # Stage 1: Feature pre-selection
        hsic_scores = None
        if self.method == "hsic-lasso":
            # HSIC-Lasso: nonlinear feature importance (consistent with ANM)
            alphas_hsic = np.geomspace(1e-4, 0.1, self.n_alphas)
            best_alpha_hsic, hsic_scores = _cross_validate_hsic_lasso(
                X, y, alphas_hsic, folds
            )
            abs_coefs = hsic_scores
            # Also compute linear LASSO for comparison/ablation
            alpha_max = np.max(np.abs(X.T @ y)) / T
            alphas_lin = np.geomspace(
                alpha_max, alpha_max * self.alpha_min_ratio, self.n_alphas
            )
            best_alpha, cv_scores = self._cross_validate_lasso(X, y, alphas_lin, folds)
            lasso_coefs = _coordinate_descent_lasso(X, y, best_alpha, self.max_iter)
            logger.info(
                "HSIC-Lasso selected %d features (vs LASSO: %d)",
                int(np.sum(hsic_scores > 1e-10)),
                int(np.sum(np.abs(lasso_coefs) > 1e-10)),
            )
        else:
            # Linear LASSO (for ablation)
            alpha_max = np.max(np.abs(X.T @ y)) / T
            alphas = np.geomspace(
                alpha_max, alpha_max * self.alpha_min_ratio, self.n_alphas
            )
            best_alpha, cv_scores = self._cross_validate_lasso(X, y, alphas, folds)
            lasso_coefs = _coordinate_descent_lasso(X, y, best_alpha, self.max_iter)
            abs_coefs = np.abs(lasso_coefs)

        # Select top features by importance score
        nonzero_mask = abs_coefs > 1e-10
        n_nonzero = int(np.sum(nonzero_mask))
        k = min(self.n_lasso_features, n_nonzero, p)
        top_indices = np.argsort(-abs_coefs)[:k].tolist()

        # Stage 2: invariant feature identification
        if regime_labels is not None:
            stability_scores, regime_coefs = self._compute_stability(
                X, y, regime_labels, top_indices
            )
            invariant = [
                idx
                for idx in top_indices
                if stability_scores[idx] < self.stability_threshold
            ]
            regime_specific = self._regime_specific_features(
                top_indices, invariant, regime_labels, regime_coefs
            )
        else:
            stability_scores = np.zeros(p)
            invariant = list(top_indices)
            regime_coefs = {}
            regime_specific = {}

        importance = self._compute_importance(abs_coefs, stability_scores)

        self._result = FeatureSelectionResult(
            selected_features=top_indices,
            invariant_features=invariant,
            regime_specific_features=regime_specific,
            feature_importance=importance,
            lasso_coefficients=lasso_coefs,
            hsic_scores=hsic_scores,
            stability_scores=stability_scores,
            cross_val_scores=cv_scores,
            n_raw=p,
            n_after_lasso=k,
            n_invariant=len(invariant),
        )
        return self._result

    def get_invariant_features(self) -> List[int]:
        """Return indices of causally invariant features."""
        if self._result is None:
            raise RuntimeError("Call select() first.")
        return list(self._result.invariant_features)

    def get_regime_specific_features(
        self, regime: Optional[int] = None
    ) -> Dict[int, List[int]] | List[int]:
        """Return regime-specific feature sets.

        If *regime* is given, return that regime's list; otherwise return
        the full dictionary.
        """
        if self._result is None:
            raise RuntimeError("Call select() first.")
        if regime is not None:
            return self._result.regime_specific_features.get(regime, [])
        return dict(self._result.regime_specific_features)

    def get_invariant_mask(self, n_features: int) -> NDArray:
        """Return boolean mask of shape (n_features,)."""
        if self._result is None:
            raise RuntimeError("Call select() first.")
        mask = np.zeros(n_features, dtype=bool)
        for idx in self._result.invariant_features:
            if idx < n_features:
                mask[idx] = True
        return mask

    def get_stability_report(self) -> List[FeatureStabilityReport]:
        """Detailed per-feature stability analysis."""
        if self._result is None:
            raise RuntimeError("Call select() first.")
        reports: List[FeatureStabilityReport] = []
        for idx in self._result.selected_features:
            coef = self._result.lasso_coefficients[idx]
            stab = self._result.stability_scores[idx]
            reports.append(
                FeatureStabilityReport(
                    feature_index=idx,
                    mean_coefficient=float(coef),
                    std_coefficient=float(stab * abs(coef)) if abs(coef) > 1e-12 else 0.0,
                    cv_ratio=float(stab),
                    regime_coefficients={},
                    is_invariant=idx in self._result.invariant_features,
                )
            )
        return reports

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _standardise(
        self, X: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0, ddof=1)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return (X - mu) / sigma, mu, sigma

    def _standardise_1d(
        self, y: NDArray
    ) -> Tuple[NDArray, float, float]:
        mu = float(np.mean(y))
        sigma = float(np.std(y, ddof=1))
        if sigma < 1e-12:
            sigma = 1.0
        return (y - mu) / sigma, mu, sigma

    def _simple_folds(
        self, T: int
    ) -> List[Tuple[NDArray, NDArray]]:
        idx = np.arange(T)
        splits = np.array_split(idx, self.n_folds)
        folds = []
        for i in range(self.n_folds):
            val = splits[i]
            train = np.concatenate([splits[j] for j in range(self.n_folds) if j != i])
            folds.append((train, val))
        return folds

    def _cross_validate_lasso(
        self,
        X: NDArray,
        y: NDArray,
        alphas: NDArray,
        folds: List[Tuple[NDArray, NDArray]],
    ) -> Tuple[float, List[float]]:
        """Cross-validated LASSO: select α minimising mean-squared error."""
        n_alphas = len(alphas)
        mse = np.zeros(n_alphas)

        for train_idx, val_idx in folds:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            path = _compute_lasso_path(X_tr, y_tr, alphas, self.max_iter)
            for i in range(n_alphas):
                pred = X_va @ path[i]
                mse[i] += np.mean((y_va - pred) ** 2)

        mse /= len(folds)
        best_idx = int(np.argmin(mse))
        cv_scores = mse.tolist()

        logger.info(
            "Best LASSO α=%.4e  (CV MSE=%.4f)", alphas[best_idx], mse[best_idx]
        )
        return float(alphas[best_idx]), cv_scores

    def _compute_stability(
        self,
        X: NDArray,
        y: NDArray,
        regime_labels: NDArray,
        feature_indices: List[int],
    ) -> Tuple[NDArray, Dict[int, NDArray]]:
        """Compute coefficient stability across regimes.

        For each feature in *feature_indices*, fit a univariate regression
        within each regime and compute the coefficient of variation across
        regimes.

        Returns
        -------
        stability : (p,) array  –  CV ratio for every feature
        regime_coefs : dict mapping regime_id → (p,) coefficient vector
        """
        p = X.shape[1]
        unique_regimes = np.unique(regime_labels)
        regime_coefs: Dict[int, NDArray] = {}

        coef_stack: List[NDArray] = []
        for r in unique_regimes:
            mask_r = regime_labels == r
            X_r = X[mask_r]
            y_r = y[mask_r]
            if len(y_r) < 5:
                continue
            # Simple OLS per feature (diagonal approximation)
            coefs_r = np.zeros(p)
            for j in feature_indices:
                xj = X_r[:, j]
                denom = xj @ xj
                if denom > 1e-12:
                    coefs_r[j] = (xj @ y_r) / denom
            regime_coefs[int(r)] = coefs_r
            coef_stack.append(coefs_r)

        if len(coef_stack) < 2:
            return np.zeros(p), regime_coefs

        coef_matrix = np.stack(coef_stack, axis=0)  # (n_regimes, p)
        means = np.mean(coef_matrix, axis=0)
        stds = np.std(coef_matrix, axis=0, ddof=1)
        stability = np.where(
            np.abs(means) > 1e-10, stds / np.abs(means), np.inf
        )
        return stability, regime_coefs

    def _regime_specific_features(
        self,
        all_selected: List[int],
        invariant: List[int],
        regime_labels: NDArray,
        regime_coefs: Dict[int, NDArray],
    ) -> Dict[int, List[int]]:
        """For each regime, identify features that are significant *only*
        in that regime (not invariant)."""
        non_invariant = [f for f in all_selected if f not in invariant]
        result: Dict[int, List[int]] = {}
        for r, coefs in regime_coefs.items():
            significant = [
                f for f in non_invariant if abs(coefs[f]) > 1e-4
            ]
            if significant:
                result[r] = significant
        return result

    @staticmethod
    def _compute_importance(
        abs_coefs: NDArray, stability_scores: NDArray
    ) -> NDArray:
        """Blend coefficient magnitude and stability into a single score.

        importance = |β| * (1 / (1 + CV))

        Invariant features with large coefficients rank highest.
        """
        damping = 1.0 / (1.0 + np.where(np.isinf(stability_scores), 10.0, stability_scores))
        raw = abs_coefs * damping
        total = np.sum(raw)
        if total > 1e-12:
            return raw / total
        return raw
