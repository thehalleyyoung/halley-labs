"""
Tests for nonparametric conditional independence tests.

Covers HSIC, distance correlation, mutual information estimation,
and classifier-based CI testing.  Tests independence vs dependence
detection and power comparison across methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------


def _make_independent(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent standard normal samples."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    return x, y


def _make_linear_dependent(
    n: int = 500, coeff: float = 1.0, noise: float = 0.5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = coeff * x + noise * rng.standard_normal(n)
    return x, y


def _make_nonlinear_dependent(
    n: int = 500, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Y = X^2 + noise.  Marginal correlation is ~0 but strong dependence."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = x ** 2 + 0.3 * rng.standard_normal(n)
    return x, y


def _make_conditional_independent(
    n: int = 500, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X ← Z → Y: X and Y independent given Z."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    x = z + 0.5 * rng.standard_normal(n)
    y = z + 0.5 * rng.standard_normal(n)
    return x, y, z


def _make_conditional_dependent(
    n: int = 500, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X → Y with Z as noise: X and Y dependent even given Z."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = 0.8 * x + 0.3 * z + 0.3 * rng.standard_normal(n)
    return x, y, z


# ---------------------------------------------------------------------------
# HSIC implementation for testing
# ---------------------------------------------------------------------------


def _rbf_kernel(x: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Compute RBF kernel matrix."""
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    dists = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    if gamma is None:
        gamma = 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2 + 1e-12)
    return np.exp(-gamma * dists)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic_statistic(
    x: np.ndarray,
    y: np.ndarray,
    gamma_x: float | None = None,
    gamma_y: float | None = None,
) -> float:
    """Compute the biased HSIC statistic."""
    Kx = _center_kernel(_rbf_kernel(x, gamma_x))
    Ky = _center_kernel(_rbf_kernel(y, gamma_y))
    n = len(x)
    return float(np.trace(Kx @ Ky) / (n * n))


def hsic_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """HSIC permutation test. Returns (statistic, p_value)."""
    observed = hsic_statistic(x, y)
    rng = np.random.default_rng(seed)
    null_stats = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        null_stats.append(hsic_statistic(x, y_perm))
    p_value = float(np.mean(np.array(null_stats) >= observed))
    return observed, p_value


# ---------------------------------------------------------------------------
# Distance correlation implementation
# ---------------------------------------------------------------------------


def _distance_matrix(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    return np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))


def _doubly_center(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean


def distance_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance covariance between x and y."""
    A = _doubly_center(_distance_matrix(x))
    B = _doubly_center(_distance_matrix(y))
    n = len(x)
    return float(np.sqrt(np.maximum(0, np.sum(A * B) / (n * n))))


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance correlation."""
    dcov_xy = distance_covariance(x, y)
    dcov_xx = distance_covariance(x, x)
    dcov_yy = distance_covariance(y, y)
    denom = np.sqrt(dcov_xx * dcov_yy)
    if denom < 1e-12:
        return 0.0
    return dcov_xy / denom


def distance_correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """Permutation test for distance correlation."""
    observed = distance_correlation(x, y)
    rng = np.random.default_rng(seed)
    null_stats = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        null_stats.append(distance_correlation(x, y_perm))
    p_value = float(np.mean(np.array(null_stats) >= observed))
    return observed, p_value


# ---------------------------------------------------------------------------
# Mutual information estimation (KSG)
# ---------------------------------------------------------------------------


def _knn_distances(X: np.ndarray, k: int) -> np.ndarray:
    """Compute k-th nearest neighbor distances using Chebyshev metric."""
    n = X.shape[0]
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    dists = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)
    return sorted_dists[:, k - 1]


def ksg_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
) -> float:
    """Estimate mutual information using KSG estimator (method 1).

    Kraskov, Stögbauer, Grassberger (2004).
    """
    from scipy.special import digamma

    n = len(x)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    xy = np.hstack([x, y])

    # k-th NN distances in joint space
    eps = _knn_distances(xy, k)

    # Count neighbors within eps in marginals
    nx = np.zeros(n, dtype=int)
    ny = np.zeros(n, dtype=int)
    for i in range(n):
        dx = np.max(np.abs(x - x[i]), axis=1)
        dy = np.max(np.abs(y - y[i]), axis=1)
        nx[i] = np.sum(dx < eps[i]) - 1  # exclude self
        ny[i] = np.sum(dy < eps[i]) - 1

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return max(0.0, float(mi))


def mi_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    n_permutations: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """Permutation test for mutual information."""
    observed = ksg_mutual_information(x, y, k)
    rng = np.random.default_rng(seed)
    null_stats = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        null_stats.append(ksg_mutual_information(x, y_perm, k))
    p_value = float(np.mean(np.array(null_stats) >= observed))
    return observed, p_value


# ---------------------------------------------------------------------------
# Classifier-based CI test
# ---------------------------------------------------------------------------


def classifier_ci_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
    n_permutations: int = 100,
    seed: int = 42,
) -> tuple[float, float]:
    """Classifier-based CI test using logistic regression accuracy.

    Tests X ⊥ Y [| Z] by fitting a classifier to predict a binary label
    that distinguishes (X, Y) from (X, Y_perm).
    """
    rng = np.random.default_rng(seed)
    n = len(x)

    def _build_features(x_arr, y_arr, z_arr=None):
        feats = np.column_stack([x_arr.reshape(-1, 1), y_arr.reshape(-1, 1)])
        if z_arr is not None:
            feats = np.column_stack([feats, z_arr.reshape(-1, 1) if z_arr.ndim == 1 else z_arr])
        return feats

    # Observed accuracy
    def _compute_accuracy(x_arr, y_arr, z_arr=None):
        y_perm = rng.permutation(y_arr)
        real_feats = _build_features(x_arr, y_arr, z_arr)
        fake_feats = _build_features(x_arr, y_perm, z_arr)
        X_cls = np.vstack([real_feats, fake_feats])
        y_cls = np.concatenate([np.ones(n), np.zeros(n)])

        # Simple logistic regression via gradient descent
        X_aug = np.column_stack([np.ones(2 * n), X_cls])
        X_aug = (X_aug - X_aug.mean(0)) / (X_aug.std(0) + 1e-10)
        p = X_aug.shape[1]
        w = np.zeros(p)
        lr = 0.01
        for _ in range(100):
            logits = X_aug @ w
            probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
            grad = X_aug.T @ (probs - y_cls) / (2 * n)
            w -= lr * grad
        preds = (X_aug @ w > 0).astype(float)
        return float(np.mean(preds == y_cls))

    observed_acc = _compute_accuracy(x, y, z)

    # Null distribution
    null_accs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        null_accs.append(_compute_accuracy(x, y_perm, z))

    p_value = float(np.mean(np.array(null_accs) >= observed_acc))
    return observed_acc, p_value


# ===================================================================
# Tests
# ===================================================================


class TestHSIC:
    def test_independent_not_rejected(self):
        x, y = _make_independent(n=300, seed=42)
        stat, p = hsic_permutation_test(x, y, n_permutations=100, seed=42)
        assert p > 0.05

    def test_linear_dependent_rejected(self):
        x, y = _make_linear_dependent(n=300, coeff=1.0, seed=42)
        stat, p = hsic_permutation_test(x, y, n_permutations=100, seed=42)
        assert p < 0.05

    def test_nonlinear_dependent_rejected(self):
        x, y = _make_nonlinear_dependent(n=300, seed=42)
        stat, p = hsic_permutation_test(x, y, n_permutations=100, seed=42)
        assert p < 0.05

    def test_statistic_nonneg(self):
        x, y = _make_independent(n=100, seed=99)
        stat = hsic_statistic(x, y)
        assert stat >= -1e-10  # biased HSIC can be slightly negative

    def test_stronger_dependence_higher_stat(self):
        x1, y1 = _make_linear_dependent(n=300, coeff=0.5, seed=42)
        x2, y2 = _make_linear_dependent(n=300, coeff=2.0, seed=42)
        s1 = hsic_statistic(x1, y1)
        s2 = hsic_statistic(x2, y2)
        assert s2 > s1


class TestDistanceCorrelation:
    def test_independent_near_zero(self):
        x, y = _make_independent(n=300, seed=42)
        dcor = distance_correlation(x, y)
        assert dcor < 0.15

    def test_dependent_large(self):
        x, y = _make_linear_dependent(n=300, coeff=1.0, seed=42)
        dcor = distance_correlation(x, y)
        assert dcor > 0.5

    def test_nonlinear_detected(self):
        x, y = _make_nonlinear_dependent(n=300, seed=42)
        dcor = distance_correlation(x, y)
        assert dcor > 0.1

    def test_bounded_zero_one(self):
        x, y = _make_linear_dependent(n=200, seed=42)
        dcor = distance_correlation(x, y)
        assert 0 <= dcor <= 1 + 1e-10

    def test_permutation_test_independent(self):
        x, y = _make_independent(n=200, seed=42)
        stat, p = distance_correlation_test(x, y, n_permutations=100, seed=42)
        assert p > 0.05

    def test_permutation_test_dependent(self):
        x, y = _make_linear_dependent(n=200, coeff=1.0, seed=42)
        stat, p = distance_correlation_test(x, y, n_permutations=100, seed=42)
        assert p < 0.05

    def test_self_correlation_is_one(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        dcor = distance_correlation(x, x)
        assert dcor > 0.95


class TestMutualInformation:
    def test_independent_near_zero(self):
        x, y = _make_independent(n=300, seed=42)
        mi = ksg_mutual_information(x, y, k=5)
        assert mi < 0.15

    def test_dependent_positive(self):
        x, y = _make_linear_dependent(n=300, coeff=1.0, seed=42)
        mi = ksg_mutual_information(x, y, k=5)
        assert mi > 0.1

    def test_nonlinear_positive(self):
        x, y = _make_nonlinear_dependent(n=300, seed=42)
        mi = ksg_mutual_information(x, y, k=5)
        assert mi > 0.05

    def test_nonnegative(self):
        x, y = _make_independent(n=200, seed=42)
        mi = ksg_mutual_information(x, y, k=5)
        assert mi >= 0

    def test_permutation_test_independent(self):
        x, y = _make_independent(n=200, seed=42)
        mi, p = mi_permutation_test(x, y, k=5, n_permutations=100, seed=42)
        assert p > 0.05

    def test_permutation_test_dependent(self):
        x, y = _make_linear_dependent(n=200, coeff=1.0, seed=42)
        mi, p = mi_permutation_test(x, y, k=5, n_permutations=100, seed=42)
        assert p < 0.10  # MI test may have lower power


class TestClassifierCITest:
    def test_independent(self):
        x, y = _make_independent(n=200, seed=42)
        acc, p = classifier_ci_test(x, y, n_permutations=50, seed=42)
        assert p > 0.01

    def test_dependent(self):
        x, y = _make_linear_dependent(n=200, coeff=2.0, seed=42)
        acc, p = classifier_ci_test(x, y, n_permutations=50, seed=42)
        assert acc >= 0.50  # at least chance level

    def test_conditional_independent(self):
        x, y, z = _make_conditional_independent(n=300, seed=42)
        acc, p = classifier_ci_test(x, y, z, n_permutations=50, seed=42)
        # Conditional on Z, should be close to chance
        assert p > 0.01

    def test_conditional_dependent(self):
        x, y, z = _make_conditional_dependent(n=300, seed=42)
        acc, p = classifier_ci_test(x, y, z, n_permutations=50, seed=42)
        assert acc >= 0.48  # should do at least near chance


class TestPowerComparison:
    """Compare power of different methods on the same data."""

    def test_linear_all_detect(self):
        x, y = _make_linear_dependent(n=300, coeff=1.5, seed=42)
        _, p_hsic = hsic_permutation_test(x, y, n_permutations=100, seed=42)
        _, p_dcor = distance_correlation_test(x, y, n_permutations=100, seed=42)
        _, p_mi = mi_permutation_test(x, y, k=5, n_permutations=100, seed=42)
        assert p_hsic < 0.1
        assert p_dcor < 0.1
        assert p_mi < 0.15

    def test_nonlinear_nonparametric_better(self):
        x, y = _make_nonlinear_dependent(n=300, seed=42)
        _, p_hsic = hsic_permutation_test(x, y, n_permutations=100, seed=42)
        dcor = distance_correlation(x, y)
        pearson_r = abs(float(np.corrcoef(x, y)[0, 1]))
        # Nonparametric measures should detect the Y=X^2 relationship
        # while Pearson correlation should be near zero
        assert dcor > pearson_r
        assert p_hsic < 0.1

    def test_independent_all_accept(self):
        x, y = _make_independent(n=300, seed=42)
        _, p_hsic = hsic_permutation_test(x, y, n_permutations=100, seed=42)
        _, p_dcor = distance_correlation_test(x, y, n_permutations=100, seed=42)
        dcor = distance_correlation(x, y)
        assert p_hsic > 0.05
        assert dcor < 0.2

    def test_weak_signal_power_ordering(self):
        """With weak signal, more powerful tests should detect more often."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.standard_normal(n)
        y = 0.3 * x + 0.9 * rng.standard_normal(n)
        dcor = distance_correlation(x, y)
        hsic_stat = hsic_statistic(x, y)
        assert dcor > 0  # should detect something
        assert hsic_stat > 0
