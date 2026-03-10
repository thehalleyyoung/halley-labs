"""
Outcome regression models for AIPW estimation.

Wraps scikit-learn regressors to produce E[Y|T=t, X] estimates for
both treatment arms.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _build_regressor(model_type: str, seed: int) -> Any:
    """Instantiate an sklearn regressor for the given *model_type*."""
    if model_type == "linear":
        return RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0), cv=5)
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    if model_type == "gbm":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, subsample=0.8,
            random_state=seed,
        )
    raise ValueError(f"Unknown outcome model type: {model_type!r}")


# ---------------------------------------------------------------------------
# OutcomeModel
# ---------------------------------------------------------------------------


class OutcomeModel:
    """Outcome regression model.

    Fits separate outcome models for control and treated arms:
    ``E[Y | T=0, X]`` and ``E[Y | T=1, X]``.

    Parameters
    ----------
    model_type : str
        Model family: ``"linear"``, ``"rf"``, ``"gbm"``.
    seed : int
        Random seed.
    """

    def __init__(self, model_type: str = "linear", seed: int = 42) -> None:
        self.model_type = model_type
        self.seed = seed
        self._model0: Any = None
        self._model1: Any = None
        self._fitted: bool = False

    # -- Fitting --------------------------------------------------------------

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> OutcomeModel:
        """Fit separate outcome models for each treatment arm.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix, shape ``(n, p)``.
        t : np.ndarray
            Treatment (binary), shape ``(n,)``.
        y : np.ndarray
            Outcome, shape ``(n,)``.

        Returns
        -------
        OutcomeModel
            self
        """
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mask0 = t == 0
        mask1 = t == 1

        if mask0.sum() < 2 or mask1.sum() < 2:
            raise ValueError(
                "Each treatment arm must have at least 2 observations."
            )

        self._model0 = _build_regressor(self.model_type, self.seed)
        self._model1 = _build_regressor(self.model_type, self.seed + 1)

        self._model0.fit(X[mask0], y[mask0])
        self._model1.fit(X[mask1], y[mask1])
        self._fitted = True
        return self

    # -- Prediction -----------------------------------------------------------

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict potential outcomes under both arms.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(mu0, mu1)`` — predicted outcomes under control and treatment.
        """
        if not self._fitted:
            raise RuntimeError("OutcomeModel has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        mu0 = self._model0.predict(X)
        mu1 = self._model1.predict(X)
        return mu0, mu1

    def predict_arm(self, X: np.ndarray, arm: int) -> np.ndarray:
        """Predict outcome under a single treatment arm.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix.
        arm : int
            Treatment arm (0 or 1).

        Returns
        -------
        np.ndarray
            Predicted outcomes.
        """
        if arm == 0:
            return self.predict(X)[0]
        elif arm == 1:
            return self.predict(X)[1]
        raise ValueError(f"arm must be 0 or 1, got {arm}")

    # -- Diagnostics ----------------------------------------------------------

    def outcome_diagnostics(
        self,
        X: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Compute outcome model diagnostics.

        Returns
        -------
        dict[str, Any]
            Contains R² scores, residual statistics per arm, and predicted vs
            actual comparison.
        """
        if not self._fitted:
            raise RuntimeError("OutcomeModel has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mask0, mask1 = t == 0, t == 1
        mu0, mu1 = self.predict(X)

        residuals0 = y[mask0] - mu0[mask0]
        residuals1 = y[mask1] - mu1[mask1]

        ss_res0 = float(np.sum(residuals0 ** 2))
        ss_tot0 = float(np.sum((y[mask0] - np.mean(y[mask0])) ** 2))
        r2_0 = 1.0 - ss_res0 / max(ss_tot0, 1e-12)

        ss_res1 = float(np.sum(residuals1 ** 2))
        ss_tot1 = float(np.sum((y[mask1] - np.mean(y[mask1])) ** 2))
        r2_1 = 1.0 - ss_res1 / max(ss_tot1, 1e-12)

        return {
            "r2_control": r2_0,
            "r2_treated": r2_1,
            "rmse_control": float(np.sqrt(np.mean(residuals0 ** 2))),
            "rmse_treated": float(np.sqrt(np.mean(residuals1 ** 2))),
            "residual_mean_control": float(np.mean(residuals0)),
            "residual_mean_treated": float(np.mean(residuals1)),
            "n_control": int(mask0.sum()),
            "n_treated": int(mask1.sum()),
        }


# ---------------------------------------------------------------------------
# Pooled (single-model) outcome model
# ---------------------------------------------------------------------------


class PooledOutcomeModel:
    """Outcome model that includes treatment as a covariate.

    Instead of fitting separate models per arm, fits a single model
    ``E[Y | T, X]`` and predicts by intervening on T.

    Parameters
    ----------
    model_type : str
        ``"linear"``, ``"rf"``, ``"gbm"``.
    seed : int
        Random seed.
    """

    def __init__(self, model_type: str = "linear", seed: int = 42) -> None:
        self.model_type = model_type
        self.seed = seed
        self._model: Any = None
        self._fitted: bool = False

    def fit(
        self, X: np.ndarray, t: np.ndarray, y: np.ndarray
    ) -> PooledOutcomeModel:
        """Fit the pooled outcome model."""
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xt = np.column_stack([X, t])
        self._model = _build_regressor(self.model_type, self.seed)
        self._model.fit(Xt, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict potential outcomes under both arms."""
        if not self._fitted:
            raise RuntimeError("PooledOutcomeModel has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        X0 = np.column_stack([X, np.zeros(n)])
        X1 = np.column_stack([X, np.ones(n)])
        return self._model.predict(X0), self._model.predict(X1)


# ---------------------------------------------------------------------------
# Cross-validated model selection
# ---------------------------------------------------------------------------


def select_outcome_model(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    candidates: list[str] | None = None,
    seed: int = 42,
    cv: int = 5,
) -> OutcomeModel:
    """Select the best outcome model via cross-validated MSE.

    Evaluates each candidate by fitting per-arm models, computing CV R² on
    the *control* arm, and selecting the best one.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix.
    t : np.ndarray
        Treatment vector.
    y : np.ndarray
        Outcome vector.
    candidates : list[str] | None
        Model types to compare.  Defaults to ``["linear", "rf", "gbm"]``.
    seed : int
        Random seed.
    cv : int
        Number of CV folds.

    Returns
    -------
    OutcomeModel
        The best model (already fitted on the full data).
    """
    if candidates is None:
        candidates = ["linear", "rf", "gbm"]

    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    mask0 = t == 0
    best_score = -np.inf
    best_type = candidates[0]

    for mtype in candidates:
        reg = _build_regressor(mtype, seed)
        if mask0.sum() >= cv + 1:
            scores = cross_val_score(
                reg, X[mask0], y[mask0], cv=min(cv, mask0.sum()), scoring="r2"
            )
            mean_score = float(np.mean(scores))
        else:
            mean_score = -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_type = mtype

    model = OutcomeModel(model_type=best_type, seed=seed)
    model.fit(X, t, y)
    return model
