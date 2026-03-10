"""
Inverse Probability Weighting (IPW) estimators with stabilization.

Implements Horvitz-Thompson, Hajek (normalised), and stabilized IPW
estimators with weight trimming/truncation strategies, comprehensive
weight diagnostics, augmented IPW comparison, and bootstrap variance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats as sp_stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold


# ===================================================================
# Result data structures
# ===================================================================


@dataclass(frozen=True, slots=True)
class IPWResult:
    """Result of an IPW estimation procedure.

    Attributes
    ----------
    ate : float
        Average treatment effect estimate.
    se : float
        Standard error.
    ci_lower : float
        Lower confidence bound.
    ci_upper : float
        Upper confidence bound.
    p_value : float
        Two-sided p-value.
    method : str
        Estimator variant name.
    n_obs : int
        Number of observations used.
    n_trimmed : int
        Number of observations trimmed.
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str = "ipw"
    n_obs: int = 0
    n_trimmed: int = 0


@dataclass(frozen=True, slots=True)
class WeightDiagnostics:
    """IPW weight diagnostics.

    Attributes
    ----------
    ess_treated : float
        Effective sample size (treated arm).
    ess_control : float
        Effective sample size (control arm).
    max_weight : float
        Maximum weight across all observations.
    cv_weights : float
        Coefficient of variation of weights.
    mean_weight : float
        Mean weight (should be ≈ 1 for stabilized).
    n_extreme : int
        Number of extreme weights (> 10× median).
    weight_ratio : float
        Ratio of max to min weight.
    kl_divergence : float
        KL divergence between weighted and unweighted distributions.
    """

    ess_treated: float
    ess_control: float
    max_weight: float
    cv_weights: float
    mean_weight: float
    n_extreme: int
    weight_ratio: float
    kl_divergence: float


# ===================================================================
# Helper: propensity model builders
# ===================================================================


def _build_ps_model(backend: str, seed: int) -> Any:
    """Instantiate a propensity score classifier."""
    if backend == "logistic":
        return LogisticRegressionCV(
            Cs=10, cv=5, max_iter=5000, solver="lbfgs", random_state=seed,
        )
    if backend == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    if backend == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, subsample=0.8, random_state=seed,
        )
    raise ValueError(f"Unknown propensity backend: {backend!r}")


# ===================================================================
# Weight computation
# ===================================================================


def compute_ipw_weights(
    A: np.ndarray,
    e: np.ndarray,
    *,
    stabilized: bool = False,
) -> np.ndarray:
    """Compute inverse probability weights.

    Parameters
    ----------
    A : np.ndarray
        Treatment assignments (binary), shape ``(n,)``.
    e : np.ndarray
        Propensity scores P(A=1|X), shape ``(n,)``.
    stabilized : bool
        If ``True``, compute stabilized weights SW = P(A) / P(A|X).

    Returns
    -------
    np.ndarray
        IPW weights, shape ``(n,)``.
    """
    if stabilized:
        p_treat = float(np.mean(A))
        w = np.where(A == 1, p_treat / e, (1.0 - p_treat) / (1.0 - e))
    else:
        w = np.where(A == 1, 1.0 / e, 1.0 / (1.0 - e))
    return w


def trim_weights(
    w: np.ndarray,
    *,
    method: str = "truncation",
    threshold: float = 0.99,
    max_weight: float | None = None,
) -> np.ndarray:
    """Apply weight trimming or truncation.

    Parameters
    ----------
    w : np.ndarray
        IPW weights.
    method : str
        ``"truncation"`` clips at a quantile, ``"winsorize"`` replaces
        extremes with boundary values, ``"fixed"`` clips at ``max_weight``.
    threshold : float
        Quantile threshold for truncation/winsorization.
    max_weight : float or None
        Fixed upper bound for ``"fixed"`` method.

    Returns
    -------
    np.ndarray
        Trimmed weights.
    """
    w = w.copy()
    if method == "truncation":
        upper = float(np.quantile(w, threshold))
        w = np.clip(w, 0.0, upper)
    elif method == "winsorize":
        lower_q = float(np.quantile(w, 1.0 - threshold))
        upper_q = float(np.quantile(w, threshold))
        w = np.clip(w, lower_q, upper_q)
    elif method == "fixed":
        if max_weight is None:
            raise ValueError("max_weight must be set for 'fixed' method.")
        w = np.clip(w, 0.0, max_weight)
    else:
        raise ValueError(f"Unknown trimming method: {method!r}")
    return w


def normalize_weights(
    w: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Hajek-normalize weights so they sum to 1 within each arm.

    Parameters
    ----------
    w : np.ndarray
        Unnormalized weights.
    A : np.ndarray
        Treatment assignments.

    Returns
    -------
    np.ndarray
        Normalized weights.
    """
    w = w.copy()
    for arm in (0, 1):
        mask = A == arm
        if mask.any():
            w[mask] = w[mask] / w[mask].sum()
    return w


# ===================================================================
# Weight diagnostics
# ===================================================================


def weight_diagnostics(
    w: np.ndarray,
    A: np.ndarray,
) -> WeightDiagnostics:
    """Compute comprehensive IPW weight diagnostics.

    Parameters
    ----------
    w : np.ndarray
        IPW weights, shape ``(n,)``.
    A : np.ndarray
        Treatment assignments (binary), shape ``(n,)``.

    Returns
    -------
    WeightDiagnostics
    """
    w = np.asarray(w, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64).ravel()

    mask1 = A == 1
    mask0 = A == 0
    w1 = w[mask1]
    w0 = w[mask0]

    # Kish's effective sample size per arm
    ess1 = float(w1.sum() ** 2 / (w1 ** 2).sum()) if len(w1) > 0 else 0.0
    ess0 = float(w0.sum() ** 2 / (w0 ** 2).sum()) if len(w0) > 0 else 0.0

    max_w = float(np.max(w))
    min_w = float(np.min(w[w > 0])) if np.any(w > 0) else 1e-12
    mean_w = float(np.mean(w))
    std_w = float(np.std(w))
    cv = std_w / max(mean_w, 1e-12)

    median_w = float(np.median(w))
    n_extreme = int(np.sum(w > 10.0 * max(median_w, 1e-12)))

    # Approximate KL divergence between weighted and uniform
    w_norm = w / max(w.sum(), 1e-12)
    n = len(w)
    u = np.ones(n) / n
    w_safe = np.clip(w_norm, 1e-12, None)
    kl = float(np.sum(w_norm * np.log(w_safe / u)))

    return WeightDiagnostics(
        ess_treated=ess1,
        ess_control=ess0,
        max_weight=max_w,
        cv_weights=cv,
        mean_weight=mean_w,
        n_extreme=n_extreme,
        weight_ratio=max_w / max(min_w, 1e-12),
        kl_divergence=max(kl, 0.0),
    )


# ===================================================================
# IPW Estimator class
# ===================================================================


class IPWEstimator:
    """Inverse Probability Weighting estimator for the ATE.

    Supports Horvitz-Thompson, Hajek (normalised), and stabilized
    variants with optional weight trimming and bootstrap inference.

    Parameters
    ----------
    propensity_backend : str
        ML backend: ``"logistic"``, ``"rf"``, ``"gbm"``.
    estimator_type : str
        ``"ht"`` (Horvitz-Thompson), ``"hajek"`` (normalised), or
        ``"stabilized"``.
    n_folds : int
        Cross-fitting folds for propensity estimation (0 for no CV).
    seed : int
        Random seed.
    alpha : float
        Significance level.
    trim : float
        Propensity score clipping threshold.
    trim_weights_method : str or None
        Weight trimming method or ``None`` for no trimming.
    trim_weights_threshold : float
        Quantile threshold for weight trimming.
    n_bootstrap : int
        Number of bootstrap replicates for variance estimation.
    """

    def __init__(
        self,
        propensity_backend: str = "logistic",
        estimator_type: str = "hajek",
        n_folds: int = 5,
        seed: int = 42,
        alpha: float = 0.05,
        trim: float = 0.01,
        trim_weights_method: str | None = None,
        trim_weights_threshold: float = 0.99,
        n_bootstrap: int = 500,
    ) -> None:
        if estimator_type not in ("ht", "hajek", "stabilized"):
            raise ValueError(
                f"estimator_type must be 'ht', 'hajek', or 'stabilized', "
                f"got {estimator_type!r}"
            )
        self.propensity_backend = propensity_backend
        self.estimator_type = estimator_type
        self.n_folds = n_folds
        self.seed = seed
        self.alpha = alpha
        self.trim = trim
        self.trim_weights_method = trim_weights_method
        self.trim_weights_threshold = trim_weights_threshold
        self.n_bootstrap = n_bootstrap
        self._weights: np.ndarray | None = None
        self._propensity_scores: np.ndarray | None = None

    # -----------------------------------------------------------------
    # Main estimation
    # -----------------------------------------------------------------

    def estimate(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        X: np.ndarray,
    ) -> IPWResult:
        """Estimate the ATE via IPW.

        Parameters
        ----------
        Y : np.ndarray
            Outcome, shape ``(n,)``.
        A : np.ndarray
            Treatment (binary), shape ``(n,)``.
        X : np.ndarray
            Covariates, shape ``(n, p)``.

        Returns
        -------
        IPWResult
        """
        Y = np.asarray(Y, dtype=np.float64).ravel()
        A = np.asarray(A, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(Y)

        # Propensity scores
        e = self._estimate_propensity(X, A)
        e = np.clip(e, self.trim, 1.0 - self.trim)
        self._propensity_scores = e

        # Weights
        stabilized = self.estimator_type == "stabilized"
        w = compute_ipw_weights(A, e, stabilized=stabilized)

        # Optional trimming
        n_trimmed = 0
        if self.trim_weights_method is not None:
            w_before = w.copy()
            w = trim_weights(
                w, method=self.trim_weights_method,
                threshold=self.trim_weights_threshold,
            )
            n_trimmed = int(np.sum(w != w_before))

        self._weights = w

        # Point estimate
        if self.estimator_type == "hajek":
            ate = self._hajek_ate(Y, A, w)
        else:
            ate = self._ht_ate(Y, A, w)

        # Bootstrap variance
        se, ci_lo, ci_hi = self._bootstrap_inference(Y, A, X, ate)

        p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(ate) / max(se, 1e-12)))

        method = f"ipw_{self.estimator_type}"
        return IPWResult(
            ate=ate,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value=p_value,
            method=method,
            n_obs=n,
            n_trimmed=n_trimmed,
        )

    # -----------------------------------------------------------------
    # Propensity estimation
    # -----------------------------------------------------------------

    def _estimate_propensity(
        self,
        X: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """Estimate propensity scores, optionally cross-fitted."""
        n = len(A)
        if self.n_folds > 1:
            e = np.empty(n, dtype=np.float64)
            kf = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed,
            )
            for train_idx, test_idx in kf.split(X):
                model = _build_ps_model(self.propensity_backend, self.seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X[train_idx], A[train_idx])
                e[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        else:
            model = _build_ps_model(self.propensity_backend, self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, A)
            e = model.predict_proba(X)[:, 1]
        return e

    # -----------------------------------------------------------------
    # ATE computation
    # -----------------------------------------------------------------

    @staticmethod
    def _ht_ate(
        Y: np.ndarray,
        A: np.ndarray,
        w: np.ndarray,
    ) -> float:
        """Horvitz-Thompson ATE: (1/n) Σ [A·w·Y − (1−A)·w·Y]."""
        n = len(Y)
        return float(np.mean(A * w * Y) - np.mean((1.0 - A) * w * Y))

    @staticmethod
    def _hajek_ate(
        Y: np.ndarray,
        A: np.ndarray,
        w: np.ndarray,
    ) -> float:
        """Hajek (normalised IPW) ATE."""
        w1 = A * w
        w0 = (1.0 - A) * w
        sum_w1 = float(np.sum(w1))
        sum_w0 = float(np.sum(w0))
        if sum_w1 < 1e-12 or sum_w0 < 1e-12:
            return 0.0
        mu1 = float(np.sum(w1 * Y) / sum_w1)
        mu0 = float(np.sum(w0 * Y) / sum_w0)
        return mu1 - mu0

    # -----------------------------------------------------------------
    # Bootstrap inference
    # -----------------------------------------------------------------

    def _bootstrap_inference(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        X: np.ndarray,
        ate_obs: float,
    ) -> tuple[float, float, float]:
        """Bootstrap variance and percentile CI.

        Returns
        -------
        tuple[float, float, float]
            ``(se, ci_lower, ci_upper)``.
        """
        rng = np.random.default_rng(self.seed)
        n = len(Y)
        boot_ates = np.empty(self.n_bootstrap, dtype=np.float64)

        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            Y_b, A_b, X_b = Y[idx], A[idx], X[idx]

            # Re-estimate propensity on bootstrap sample
            model = _build_ps_model(self.propensity_backend, self.seed + b)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(X_b, A_b)
                    e_b = model.predict_proba(X_b)[:, 1]
                except Exception:
                    boot_ates[b] = ate_obs
                    continue
            e_b = np.clip(e_b, self.trim, 1.0 - self.trim)

            stabilized = self.estimator_type == "stabilized"
            w_b = compute_ipw_weights(A_b, e_b, stabilized=stabilized)

            if self.trim_weights_method is not None:
                w_b = trim_weights(
                    w_b, method=self.trim_weights_method,
                    threshold=self.trim_weights_threshold,
                )

            if self.estimator_type == "hajek":
                boot_ates[b] = self._hajek_ate(Y_b, A_b, w_b)
            else:
                boot_ates[b] = self._ht_ate(Y_b, A_b, w_b)

        se = float(np.std(boot_ates, ddof=1))
        ci_lo = float(np.percentile(boot_ates, 100 * self.alpha / 2))
        ci_hi = float(np.percentile(boot_ates, 100 * (1.0 - self.alpha / 2)))
        return se, ci_lo, ci_hi

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def get_weight_diagnostics(self, A: np.ndarray) -> WeightDiagnostics | None:
        """Return weight diagnostics from the last estimation.

        Parameters
        ----------
        A : np.ndarray
            Treatment assignments used in the last call.

        Returns
        -------
        WeightDiagnostics or None
        """
        if self._weights is None:
            return None
        return weight_diagnostics(self._weights, A)

    @property
    def weights(self) -> np.ndarray | None:
        """Weights from the last estimation call."""
        return self._weights

    @property
    def propensity_scores(self) -> np.ndarray | None:
        """Propensity scores from the last estimation call."""
        return self._propensity_scores


# ===================================================================
# Augmented IPW comparison
# ===================================================================


def aipw_vs_ipw(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    *,
    propensity_backend: str = "logistic",
    n_folds: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
    trim: float = 0.01,
) -> dict[str, IPWResult]:
    """Compare IPW and augmented IPW for diagnostic purposes.

    Returns both Hajek IPW and a simple AIPW using the same propensity
    model, allowing assessment of the augmentation's impact on efficiency.

    Parameters
    ----------
    Y, A, X : np.ndarray
        Outcome, treatment, covariates.
    propensity_backend : str
        Propensity model backend.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.
    alpha : float
        Significance level.
    trim : float
        Propensity trimming.

    Returns
    -------
    dict[str, IPWResult]
        Keys: ``"ipw"`` and ``"aipw"``.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(Y)

    # IPW estimate
    ipw_est = IPWEstimator(
        propensity_backend=propensity_backend,
        estimator_type="hajek",
        n_folds=n_folds, seed=seed, alpha=alpha, trim=trim,
    )
    ipw_result = ipw_est.estimate(Y, A, X)

    # Simple AIPW using same propensity
    e = ipw_est.propensity_scores
    if e is None:
        return {"ipw": ipw_result, "aipw": ipw_result}

    # Fit simple outcome models per arm
    from sklearn.linear_model import RidgeCV as _Ridge
    mask1 = A == 1
    mask0 = A == 0
    mu1_hat = np.zeros(n, dtype=np.float64)
    mu0_hat = np.zeros(n, dtype=np.float64)

    if mask1.sum() >= 2:
        m1 = _Ridge(alphas=(0.01, 0.1, 1.0, 10.0), cv=min(5, int(mask1.sum())))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1.fit(X[mask1], Y[mask1])
        mu1_hat = m1.predict(X)
    if mask0.sum() >= 2:
        m0 = _Ridge(alphas=(0.01, 0.1, 1.0, 10.0), cv=min(5, int(mask0.sum())))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m0.fit(X[mask0], Y[mask0])
        mu0_hat = m0.predict(X)

    # AIPW scores
    scores = (
        mu1_hat - mu0_hat
        + A * (Y - mu1_hat) / e
        - (1.0 - A) * (Y - mu0_hat) / (1.0 - e)
    )
    ate_aipw = float(np.mean(scores))
    psi = scores - ate_aipw
    se_aipw = float(np.sqrt(np.mean(psi ** 2) / n))

    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    aipw_result = IPWResult(
        ate=ate_aipw,
        se=se_aipw,
        ci_lower=ate_aipw - z * se_aipw,
        ci_upper=ate_aipw + z * se_aipw,
        p_value=2.0 * (1.0 - sp_stats.norm.cdf(abs(ate_aipw) / max(se_aipw, 1e-12))),
        method="aipw_comparison",
        n_obs=n,
    )

    return {"ipw": ipw_result, "aipw": aipw_result}
