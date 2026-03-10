"""
Propensity score models for AIPW estimation.

Wraps scikit-learn classifiers to produce P(T=1|X) estimates with
built-in clipping for positivity.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _build_classifier(model_type: str, seed: int) -> Any:
    """Instantiate an sklearn classifier for the given *model_type*."""
    if model_type == "logistic":
        return LogisticRegressionCV(
            Cs=10, cv=5, max_iter=5000, solver="lbfgs",
            random_state=seed,
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    if model_type == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, subsample=0.8,
            random_state=seed,
        )
    raise ValueError(f"Unknown propensity model type: {model_type!r}")


# ---------------------------------------------------------------------------
# PropensityModel
# ---------------------------------------------------------------------------


class PropensityModel:
    """Propensity score model with positivity clipping.

    Parameters
    ----------
    model_type : str
        Model family: ``"logistic"``, ``"rf"`` (random forest), ``"gbm"``
        (gradient boosted trees).
    clip_bounds : tuple[float, float]
        Bounds for clipping predicted probabilities to ensure positivity.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        model_type: str = "logistic",
        clip_bounds: tuple[float, float] = (0.01, 0.99),
        seed: int = 42,
    ) -> None:
        self.model_type = model_type
        self.clip_bounds = clip_bounds
        self.seed = seed
        self._model: Any = None
        self._fitted: bool = False

    # -- Fitting --------------------------------------------------------------

    def fit(self, X: np.ndarray, t: np.ndarray) -> PropensityModel:
        """Fit the propensity score model.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix, shape ``(n, p)``.
        t : np.ndarray
            Treatment assignment (binary), shape ``(n,)``.

        Returns
        -------
        PropensityModel
            self
        """
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(np.unique(t)) < 2:
            raise ValueError("Treatment vector must have at least two levels.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._model = _build_classifier(self.model_type, self.seed)
            self._model.fit(X, t)
        self._fitted = True
        return self

    # -- Prediction -----------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict propensity scores P(T=1|X).

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix.

        Returns
        -------
        np.ndarray
            Clipped propensity scores in ``clip_bounds``.
        """
        if not self._fitted:
            raise RuntimeError("PropensityModel has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        raw = self._model.predict_proba(X)[:, 1]
        return np.clip(raw, self.clip_bounds[0], self.clip_bounds[1])

    # -- Diagnostics ----------------------------------------------------------

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict *unclipped* propensity scores."""
        if not self._fitted:
            raise RuntimeError("PropensityModel has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._model.predict_proba(X)[:, 1]

    def stabilized_weights(
        self,
        X: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """Compute stabilized inverse-probability weights.

        Stabilized weights are ``P(T=t) / P(T=t|X)`` and have expectation 1
        within each treatment arm, reducing variance relative to raw IPW
        weights.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix, shape ``(n, p)``.
        t : np.ndarray
            Observed treatment assignments (binary), shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Stabilized weights, shape ``(n,)``.
        """
        t = np.asarray(t, dtype=np.float64).ravel()
        e = self.predict(X)
        p_treat = float(np.mean(t))
        p_ctrl = 1.0 - p_treat
        weights = np.where(t == 1, p_treat / e, p_ctrl / (1.0 - e))
        return weights

    def ipw_weights(
        self,
        X: np.ndarray,
        t: np.ndarray,
        stabilized: bool = False,
        normalize: bool = False,
    ) -> np.ndarray:
        """Compute inverse-probability weights for treatment effect estimation.

        Parameters
        ----------
        X : np.ndarray
            Covariates.
        t : np.ndarray
            Treatment assignments (binary).
        stabilized : bool
            If ``True``, return stabilized weights.
        normalize : bool
            If ``True``, Hajek-normalise so weights sum to 1 within each arm.

        Returns
        -------
        np.ndarray
            IPW weights.
        """
        if stabilized:
            w = self.stabilized_weights(X, t)
        else:
            t = np.asarray(t, dtype=np.float64).ravel()
            e = self.predict(X)
            w = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))

        if normalize:
            t = np.asarray(t, dtype=np.float64).ravel()
            for arm in [0, 1]:
                mask = t == arm
                if mask.any():
                    w[mask] = w[mask] / w[mask].sum()
        return w

    def overlap_diagnostics(
        self,
        X: np.ndarray,
        t: np.ndarray,
    ) -> dict[str, Any]:
        """Run propensity score overlap diagnostics.

        Returns a dictionary with:
        - ``"mean_treated"`` / ``"mean_control"``: mean PS per arm
        - ``"std_treated"`` / ``"std_control"``: std PS per arm
        - ``"min"`` / ``"max"``: overall min/max PS
        - ``"n_clipped_low"`` / ``"n_clipped_high"``: counts of clipped scores
        - ``"effective_sample_size_treated"`` / ``"...control"``: ESS per arm

        Parameters
        ----------
        X : np.ndarray
            Covariates.
        t : np.ndarray
            Treatment assignments (binary).

        Returns
        -------
        dict[str, Any]
        """
        t = np.asarray(t, dtype=np.float64).ravel()
        raw = self.predict_raw(X)
        clipped = self.predict(X)

        treated_mask = t == 1
        control_mask = t == 0

        # effective sample size per arm (Kish's ESS)
        w_t = 1.0 / clipped[treated_mask]
        w_c = 1.0 / (1.0 - clipped[control_mask])
        ess_t = float(w_t.sum() ** 2 / (w_t ** 2).sum()) if w_t.size > 0 else 0.0
        ess_c = float(w_c.sum() ** 2 / (w_c ** 2).sum()) if w_c.size > 0 else 0.0

        return {
            "mean_treated": float(np.mean(raw[treated_mask])) if treated_mask.any() else float("nan"),
            "mean_control": float(np.mean(raw[control_mask])) if control_mask.any() else float("nan"),
            "std_treated": float(np.std(raw[treated_mask])) if treated_mask.any() else float("nan"),
            "std_control": float(np.std(raw[control_mask])) if control_mask.any() else float("nan"),
            "min": float(np.min(raw)),
            "max": float(np.max(raw)),
            "n_clipped_low": int(np.sum(raw < self.clip_bounds[0])),
            "n_clipped_high": int(np.sum(raw > self.clip_bounds[1])),
            "effective_sample_size_treated": ess_t,
            "effective_sample_size_control": ess_c,
        }

    def calibration_diagnostics(
        self,
        X: np.ndarray,
        t: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Compute calibration curve for the propensity model.

        Parameters
        ----------
        X : np.ndarray
            Covariates.
        t : np.ndarray
            Treatment assignments.
        n_bins : int
            Number of calibration bins.

        Returns
        -------
        dict[str, Any]
            Contains ``"fraction_of_positives"`` and ``"mean_predicted_value"``.
        """
        t = np.asarray(t, dtype=np.float64).ravel()
        probs = self.predict_raw(X)
        fraction_pos, mean_pred = calibration_curve(t, probs, n_bins=n_bins)
        return {
            "fraction_of_positives": fraction_pos,
            "mean_predicted_value": mean_pred,
        }


# ---------------------------------------------------------------------------
# Cross-validated model selection
# ---------------------------------------------------------------------------


def select_propensity_model(
    X: np.ndarray,
    t: np.ndarray,
    candidates: list[str] | None = None,
    seed: int = 42,
    cv: int = 5,
    clip_bounds: tuple[float, float] = (0.01, 0.99),
) -> PropensityModel:
    """Select the best propensity model via cross-validated log-loss.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix.
    t : np.ndarray
        Treatment vector.
    candidates : list[str] | None
        Model types to compare.  Defaults to ``["logistic", "rf", "gbm"]``.
    seed : int
        Random seed.
    cv : int
        Number of CV folds.
    clip_bounds : tuple[float, float]
        Clipping bounds.

    Returns
    -------
    PropensityModel
        The best model (already fitted on the full data).
    """
    from sklearn.model_selection import cross_val_score

    if candidates is None:
        candidates = ["logistic", "rf", "gbm"]

    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    best_score = -np.inf
    best_type = candidates[0]

    for mtype in candidates:
        clf = _build_classifier(mtype, seed)
        scores = cross_val_score(clf, X, t, cv=cv, scoring="neg_log_loss")
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_type = mtype

    model = PropensityModel(model_type=best_type, clip_bounds=clip_bounds, seed=seed)
    model.fit(X, t)
    return model


# ---------------------------------------------------------------------------
# Trimming utilities
# ---------------------------------------------------------------------------


def trim_by_propensity(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    lower: float = 0.05,
    upper: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trim observations with extreme propensity scores.

    Observations whose estimated propensity lies outside ``[lower, upper]``
    are removed from all arrays.

    Parameters
    ----------
    X : np.ndarray
        Covariates, shape ``(n, p)``.
    t : np.ndarray
        Treatment, shape ``(n,)``.
    y : np.ndarray
        Outcome, shape ``(n,)``.
    e : np.ndarray
        Propensity scores, shape ``(n,)``.
    lower, upper : float
        Trimming thresholds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Trimmed ``(X, t, y, e)``.
    """
    mask = (e >= lower) & (e <= upper)
    return X[mask], t[mask], y[mask], e[mask]


def crump_trimming_rule(
    e: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Compute the Crump et al. (2009) trimming rule.

    The rule keeps observations with propensity score in
    ``[alpha, 1 - alpha]``, which optimises precision of ATE estimation.

    Parameters
    ----------
    e : np.ndarray
        Propensity scores.
    alpha : float
        Threshold (default 0.1).

    Returns
    -------
    np.ndarray
        Boolean mask of observations to *keep*.
    """
    return (e >= alpha) & (e <= 1.0 - alpha)
