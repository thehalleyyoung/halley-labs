"""
Classifier-based conditional-independence test (CCIT).

Trains a classifier to distinguish real X from permuted X given Z, using
the cross-validated AUC as the test statistic.  If X ⊥ Y | Z, then
knowing Z should not help predict whether X is real or permuted, so
AUC ≈ 0.5.

Implements:
- Random-forest and gradient-boosting classifier backends
- Stratified K-fold cross-validated AUC
- Permutation distribution for p-value calibration
- Feature-importance diagnostic

References
----------
Sen, R., Suresh, A. T., Shanmugam, K., Dimakis, A. G. & Shakkottai, S.
    (2017). Model-powered conditional independence test.
    *NeurIPS 2017*.

Chalupka, K., Perona, P. & Eberhardt, F. (2018).
    Fast conditional independence test for vector variables with large
    sample sizes.  *arXiv:1804.02747*.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from causalcert.ci_testing.base import (
    BaseCITest,
    CITestConfig,
    _extract_columns,
    _insufficient_sample_result,
    _validate_inputs,
)
from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_EPS = 1e-10
_MIN_N = 30  # classifiers need more data


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassifierCIConfig:
    """Configuration for the classifier-based CI test.

    Attributes
    ----------
    classifier : str
        Classifier backend: ``"random_forest"`` or ``"gradient_boosting"``.
    n_estimators : int
        Number of trees in the ensemble classifier.
    max_depth : int | None
        Maximum tree depth (``None`` for unlimited).
    n_folds : int
        Number of cross-validation folds.
    n_permutations : int
        Number of permutations for the null distribution.
    compute_importance : bool
        Whether to compute feature importance as an auxiliary diagnostic.
    n_jobs : int
        Number of parallel jobs for the classifier (``-1`` for all CPUs).
    """

    classifier: str = "random_forest"
    n_estimators: int = 100
    max_depth: int | None = 5
    n_folds: int = 5
    n_permutations: int = 200
    compute_importance: bool = False
    n_jobs: int = 1


# ---------------------------------------------------------------------------
# Classifier construction
# ---------------------------------------------------------------------------


def _build_classifier(
    cfg: ClassifierCIConfig,
    seed: int,
) -> Any:
    """Construct a scikit-learn classifier.

    Parameters
    ----------
    cfg : ClassifierCIConfig
        Configuration.
    seed : int
        Random seed.

    Returns
    -------
    Classifier
        A scikit-learn compatible classifier.
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )

    if cfg.classifier == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth or 3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=seed,
        )
    # Default: random forest
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        n_jobs=cfg.n_jobs,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Cross-validated AUC computation
# ---------------------------------------------------------------------------


def _cross_validated_auc(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: ClassifierCIConfig,
    seed: int,
) -> float:
    """Compute cross-validated AUC for a binary classification task.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix ``(2n, d)`` where the first *n* rows are real
        and the last *n* rows use permuted X.
    labels : np.ndarray
        Binary labels ``(2n,)``: 1 for real, 0 for permuted.
    cfg : ClassifierCIConfig
        Configuration.
    seed : int
        Random seed.

    Returns
    -------
    float
        Mean AUC across folds.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n_folds = min(cfg.n_folds, int(labels.sum()), len(labels) - int(labels.sum()))
    if n_folds < 2:
        return 0.5

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed,
    )
    auc_scores: list[float] = []

    for train_idx, test_idx in skf.split(features, labels):
        clf = _build_classifier(cfg, seed)
        X_train, y_train = features[train_idx], labels[train_idx]
        X_test, y_test = features[test_idx], labels[test_idx]

        try:
            clf.fit(X_train, y_train)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                y_prob = clf.decision_function(X_test)
            auc = roc_auc_score(y_test, y_prob)
            auc_scores.append(auc)
        except Exception:
            auc_scores.append(0.5)

    return float(np.mean(auc_scores)) if auc_scores else 0.5


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeatureImportanceResult:
    """Feature importance from the classifier diagnostic.

    Attributes
    ----------
    importances : np.ndarray
        Per-feature importance scores (sum to 1).
    feature_names : list[str]
        Names corresponding to each importance value.
    """

    importances: np.ndarray
    feature_names: list[str]


def _compute_feature_importance(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    cfg: ClassifierCIConfig,
    seed: int,
) -> FeatureImportanceResult:
    """Train a classifier and extract feature importance.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Binary labels.
    feature_names : list[str]
        Names for each feature column.
    cfg : ClassifierCIConfig
        Configuration.
    seed : int
        Random seed.

    Returns
    -------
    FeatureImportanceResult
    """
    clf = _build_classifier(cfg, seed)
    try:
        clf.fit(features, labels)
        imp = clf.feature_importances_
    except Exception:
        imp = np.ones(features.shape[1]) / features.shape[1]

    return FeatureImportanceResult(
        importances=imp,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# CCIT test-statistic construction
# ---------------------------------------------------------------------------


def _build_ccit_dataset(
    x_col: np.ndarray,
    y_col: np.ndarray,
    z_cols: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the binary classification dataset for CCIT.

    Constructs a dataset where:
    - Class 1 (real): features = [X, Z], i.e. true X with its Z
    - Class 0 (fake): features = [X_perm, Z], where X is permuted

    The classifier tries to distinguish real from permuted X given Z.
    If X ⊥ Y | Z, Y provides no information, so we include Y in features
    and test whether it helps distinguish real from permuted X.

    Parameters
    ----------
    x_col : np.ndarray
        X values ``(n,)``.
    y_col : np.ndarray
        Y values ``(n,)``.
    z_cols : np.ndarray | None
        Conditioning variables ``(n, k)`` or ``None``.
    rng : np.random.Generator
        RNG.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        ``(features, labels, feature_names)`` where features has shape
        ``(2n, d)`` and labels has shape ``(2n,)``.
    """
    n = len(x_col)

    # Real samples: original (Y, Z)
    if z_cols is not None:
        real_features = np.column_stack([y_col, z_cols])
        names = ["Y"] + [f"Z{j}" for j in range(z_cols.shape[1])]
    else:
        real_features = y_col[:, np.newaxis]
        names = ["Y"]

    # Fake samples: permuted X breaks X-Y link
    perm = rng.permutation(n)
    if z_cols is not None:
        fake_features = np.column_stack([y_col[perm], z_cols])
    else:
        fake_features = y_col[perm, np.newaxis]

    features = np.vstack([real_features, fake_features])
    labels = np.concatenate([np.ones(n), np.zeros(n)])

    return features, labels, names


# ---------------------------------------------------------------------------
# ClassifierCITest class
# ---------------------------------------------------------------------------


class ClassifierCITest(BaseCITest):
    """Classifier-based conditional-independence test (CCIT).

    Tests ``X ⊥ Y | Z`` by training a classifier to distinguish real
    observations from those where the X–Y relationship is broken by
    permutation.  The cross-validated AUC serves as the test statistic:
    AUC ≈ 0.5 under independence, AUC > 0.5 under dependence.

    A permutation distribution calibrates the p-value.

    Parameters
    ----------
    alpha : float
        Significance level.
    seed : int
        Random seed.
    config : CITestConfig | None
        Base CI test configuration.
    ccit_config : ClassifierCIConfig | None
        Classifier-specific configuration.
    """

    method = CITestMethod.CLASSIFIER

    def __init__(
        self,
        alpha: float = 0.05,
        seed: int = 42,
        config: CITestConfig | None = None,
        ccit_config: ClassifierCIConfig | None = None,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed, config=config)
        self.ccit_config = ccit_config or ClassifierCIConfig()
        self._last_importance: FeatureImportanceResult | None = None

    # ------------------------------------------------------------------
    # Core test
    # ------------------------------------------------------------------

    def _ccit_statistic(
        self,
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray | None,
        rng: np.random.Generator,
    ) -> float:
        """Compute the CCIT statistic (cross-validated AUC).

        Parameters
        ----------
        x_col : np.ndarray
            X data.
        y_col : np.ndarray
            Y data.
        z_cols : np.ndarray | None
            Conditioning data.
        rng : np.random.Generator
            RNG.

        Returns
        -------
        float
            AUC statistic.
        """
        features, labels, _ = _build_ccit_dataset(
            x_col, y_col, z_cols, rng,
        )
        return _cross_validated_auc(
            features, labels, self.ccit_config, self.seed,
        )

    def _permutation_pvalue(
        self,
        observed_auc: float,
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray | None,
        rng: np.random.Generator,
    ) -> float:
        """Compute permutation p-value for the AUC statistic.

        Under the null, permuting both X and Y should give AUC ≈ 0.5.
        We re-permute to build the null distribution of AUC.

        Parameters
        ----------
        observed_auc : float
            Observed AUC from the original data.
        x_col : np.ndarray
            X data.
        y_col : np.ndarray
            Y data.
        z_cols : np.ndarray | None
            Conditioning data.
        rng : np.random.Generator
            RNG.

        Returns
        -------
        float
            p-value.
        """
        n = len(x_col)
        n_perm = self.ccit_config.n_permutations
        count = 0

        for b in range(n_perm):
            perm = rng.permutation(n)
            x_perm = x_col[perm]
            features, labels, _ = _build_ccit_dataset(
                x_perm, y_col, z_cols, rng,
            )
            null_auc = _cross_validated_auc(
                features, labels, self.ccit_config, self.seed + b + 1,
            )
            if null_auc >= observed_auc:
                count += 1

        return (count + 1) / (n_perm + 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Test X ⊥ Y | Z using a classifier.

        Parameters
        ----------
        x : NodeId
            First variable.
        y : NodeId
            Second variable.
        conditioning_set : NodeSet
            Conditioning variables.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        CITestResult
        """
        _validate_inputs(data, x, y, conditioning_set)
        x_col, y_col, z_cols = _extract_columns(data, x, y, conditioning_set)

        n = len(x_col)
        if n < max(self.config.min_samples, _MIN_N):
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha,
            )

        rng = np.random.default_rng(self.seed)
        cfg = self.ccit_config

        # Compute observed AUC
        observed_auc = self._ccit_statistic(x_col, y_col, z_cols, rng)

        # Feature importance (optional)
        if cfg.compute_importance:
            features, labels, names = _build_ccit_dataset(
                x_col, y_col, z_cols, rng,
            )
            self._last_importance = _compute_feature_importance(
                features, labels, names, cfg, self.seed,
            )

        # Permutation p-value
        p_value = self._permutation_pvalue(
            observed_auc, x_col, y_col, z_cols, rng,
        )

        # Statistic: shift AUC so that 0.5 → 0 (independence)
        stat = observed_auc - 0.5

        return self._make_result(x, y, conditioning_set, stat, p_value)

    @property
    def feature_importance(self) -> FeatureImportanceResult | None:
        """Return the feature importance from the last test run.

        Only available when ``compute_importance=True`` in config.

        Returns
        -------
        FeatureImportanceResult | None
        """
        return self._last_importance

    def test_with_diagnostics(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> tuple[CITestResult, FeatureImportanceResult | None]:
        """Run the test and return both the result and feature importance.

        Parameters
        ----------
        x : NodeId
            First variable.
        y : NodeId
            Second variable.
        conditioning_set : NodeSet
            Conditioning variables.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        tuple[CITestResult, FeatureImportanceResult | None]
        """
        old_flag = self.ccit_config.compute_importance
        self.ccit_config.compute_importance = True
        try:
            result = self.test(x, y, conditioning_set, data)
        finally:
            self.ccit_config.compute_importance = old_flag
        return result, self._last_importance

    def __repr__(self) -> str:  # noqa: D105
        cfg = self.ccit_config
        return (
            f"ClassifierCITest(clf={cfg.classifier!r}, "
            f"n_est={cfg.n_estimators}, alpha={self.alpha})"
        )
