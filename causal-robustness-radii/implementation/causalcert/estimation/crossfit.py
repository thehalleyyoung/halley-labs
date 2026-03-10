"""
Cross-fitting infrastructure for doubly-robust estimation.

Implements K-fold sample splitting for nuisance parameter estimation,
ensuring that the AIPW estimator achieves root-n rates even with
non-parametric nuisance models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Result from a single cross-fitting fold.

    Attributes
    ----------
    fold_idx : int
        Fold index.
    train_indices : np.ndarray
        Training set row indices.
    test_indices : np.ndarray
        Test (estimation) set row indices.
    propensity_scores : np.ndarray
        Predicted propensity scores on the test set.
    mu0 : np.ndarray
        Predicted E[Y|T=0, X] on the test set.
    mu1 : np.ndarray
        Predicted E[Y|T=1, X] on the test set.
    """

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    propensity_scores: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray


class CrossFitter:
    """K-fold cross-fitting orchestrator.

    Implements the DML-style (Chernozhukov et al., 2018) cross-fitting
    procedure: split data into K folds, for each fold k fit nuisance
    models on the complement of fold k, then predict on fold k.

    Parameters
    ----------
    n_folds : int
        Number of folds.
    seed : int
        Random seed for fold assignment.
    """

    def __init__(self, n_folds: int = 5, seed: int = 42) -> None:
        self.n_folds = n_folds
        self.seed = seed

    # -- Public interface -----------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        treatment_col: int,
        outcome_col: int,
        covariate_cols: list[int],
        propensity_factory: Callable[..., Any],
        outcome_factory: Callable[..., Any],
    ) -> list[FoldResult]:
        """Run cross-fitting and return per-fold results.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset.
        treatment_col : int
            Column index of the treatment variable.
        outcome_col : int
            Column index of the outcome variable.
        covariate_cols : list[int]
            Column indices of covariates (adjustment set).
        propensity_factory : Callable
            Factory for propensity score models.  Must return an object with
            ``.fit(X, t)`` and ``.predict(X)`` methods.
        outcome_factory : Callable
            Factory for outcome regression models.  Must return an object with
            ``.fit(X, t, y)`` and ``.predict(X)`` returning ``(mu0, mu1)``.

        Returns
        -------
        list[FoldResult]
            One result per fold.
        """
        values = data.values
        n = values.shape[0]
        folds = self._create_folds(n)

        T = values[:, treatment_col].astype(np.float64)
        Y = values[:, outcome_col].astype(np.float64)
        X = values[:, covariate_cols].astype(np.float64)

        results: list[FoldResult] = []
        for k, test_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(self.n_folds) if j != k])

            X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
            X_test = X[test_idx]

            # Fit propensity model on training fold
            ps_model = propensity_factory()
            ps_model.fit(X_train, T_train)
            e_test = ps_model.predict(X_test)

            # Fit outcome model on training fold
            out_model = outcome_factory()
            out_model.fit(X_train, T_train, Y_train)
            mu0_test, mu1_test = out_model.predict(X_test)

            results.append(FoldResult(
                fold_idx=k,
                train_indices=train_idx,
                test_indices=test_idx,
                propensity_scores=e_test,
                mu0=mu0_test,
                mu1=mu1_test,
            ))

        return results

    def fit_arrays(
        self,
        X: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
        propensity_factory: Callable[..., Any],
        outcome_factory: Callable[..., Any],
    ) -> list[FoldResult]:
        """Run cross-fitting on numpy arrays directly.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix, shape ``(n, p)``.
        t : np.ndarray
            Treatment, shape ``(n,)``.
        y : np.ndarray
            Outcome, shape ``(n,)``.
        propensity_factory : Callable
            Factory for propensity models.
        outcome_factory : Callable
            Factory for outcome models.

        Returns
        -------
        list[FoldResult]
        """
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]
        folds = self._create_folds(n)
        results: list[FoldResult] = []

        for k, test_idx in enumerate(folds):
            train_idx = np.concatenate(
                [folds[j] for j in range(self.n_folds) if j != k]
            )

            X_train, t_train, y_train = X[train_idx], t[train_idx], y[train_idx]
            X_test = X[test_idx]

            ps_model = propensity_factory()
            ps_model.fit(X_train, t_train)
            e_test = ps_model.predict(X_test)

            out_model = outcome_factory()
            out_model.fit(X_train, t_train, y_train)
            mu0_test, mu1_test = out_model.predict(X_test)

            results.append(FoldResult(
                fold_idx=k,
                train_indices=train_idx,
                test_indices=test_idx,
                propensity_scores=e_test,
                mu0=mu0_test,
                mu1=mu1_test,
            ))

        return results

    # -- Fold creation --------------------------------------------------------

    def _create_folds(self, n: int) -> list[np.ndarray]:
        """Create fold assignments.

        Shuffles indices and splits into *n_folds* roughly equal-sized folds.

        Parameters
        ----------
        n : int
            Number of observations.

        Returns
        -------
        list[np.ndarray]
            List of index arrays, one per fold.
        """
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n)
        return [arr for arr in np.array_split(indices, self.n_folds)]


# ---------------------------------------------------------------------------
# DML-style aggregation
# ---------------------------------------------------------------------------


def aggregate_fold_results(
    fold_results: list[FoldResult],
    y_full: np.ndarray,
    t_full: np.ndarray,
) -> dict[str, np.ndarray]:
    """Combine per-fold nuisance estimates into full-sample arrays.

    Produces arrays of propensity scores, mu0, and mu1 aligned with the
    original data ordering.

    Parameters
    ----------
    fold_results : list[FoldResult]
        Output of :meth:`CrossFitter.fit`.
    y_full : np.ndarray
        Full outcome vector.
    t_full : np.ndarray
        Full treatment vector.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``"e"``, ``"mu0"``, ``"mu1"`` — full-sample nuisance estimates.
    """
    n = len(y_full)
    e = np.empty(n, dtype=np.float64)
    mu0 = np.empty(n, dtype=np.float64)
    mu1 = np.empty(n, dtype=np.float64)

    for fr in fold_results:
        idx = fr.test_indices
        e[idx] = fr.propensity_scores
        mu0[idx] = fr.mu0
        mu1[idx] = fr.mu1

    return {"e": e, "mu0": mu0, "mu1": mu1}


def dml_estimate(
    fold_results: list[FoldResult],
    y_full: np.ndarray,
    t_full: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Compute the DML point estimate and standard error from fold results.

    Uses the cross-fitted AIPW scores.

    Parameters
    ----------
    fold_results : list[FoldResult]
        Cross-fitting fold results.
    y_full : np.ndarray
        Full outcome vector.
    t_full : np.ndarray
        Full treatment vector.

    Returns
    -------
    tuple[float, float, np.ndarray]
        ``(ate, se, psi)`` — point estimate, standard error, and per-observation
        influence function values.
    """
    nuisance = aggregate_fold_results(fold_results, y_full, t_full)
    e = nuisance["e"]
    mu0 = nuisance["mu0"]
    mu1 = nuisance["mu1"]

    y = np.asarray(y_full, dtype=np.float64)
    t = np.asarray(t_full, dtype=np.float64)

    # AIPW scores
    scores = (
        t * (y - mu1) / e
        - (1.0 - t) * (y - mu0) / (1.0 - e)
        + (mu1 - mu0)
    )
    ate = float(np.mean(scores))

    # Influence function (centred)
    psi = scores - ate
    n = len(y)
    se = float(np.sqrt(np.mean(psi ** 2) / n))

    return ate, se, psi


# ---------------------------------------------------------------------------
# Honest sample splitting (2-fold)
# ---------------------------------------------------------------------------


class HonestSplitter:
    """Two-fold honest sample splitting for inference.

    Splits data into an estimation sample and an inference sample. Nuisance
    parameters are fitted on the estimation sample and evaluated on the
    inference sample. Roles are then swapped, and results are averaged.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def split(
        self,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split indices into two halves.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(half_a, half_b)``
        """
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n)
        mid = n // 2
        return indices[:mid], indices[mid:]

    def fit_and_aggregate(
        self,
        X: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
        propensity_factory: Callable[..., Any],
        outcome_factory: Callable[..., Any],
    ) -> tuple[float, float, np.ndarray]:
        """Honest 2-fold cross-fit.

        Returns
        -------
        tuple[float, float, np.ndarray]
            ``(ate, se, psi)``
        """
        cf = CrossFitter(n_folds=2, seed=self.seed)
        folds = cf.fit_arrays(X, t, y, propensity_factory, outcome_factory)
        return dml_estimate(folds, y, t)
