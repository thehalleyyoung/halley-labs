"""Nonparametric conditional independence tests.

Implements HSIC, distance correlation, KSG mutual information estimator,
conditional mutual information, maximal information coefficient,
randomised conditional correlation, Friedman-Rafsky adapted CI test,
and classifier-based CI test.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from causalcert.ci_testing.base import BaseCITest
from causalcert.types import CITestResult


# ===================================================================
# Kernel helpers
# ===================================================================

def _rbf_kernel(X: np.ndarray, *, bandwidth: Optional[float] = None) -> np.ndarray:
    """Gaussian RBF kernel matrix with median-heuristic bandwidth."""
    n = X.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    if bandwidth is None:
        med = np.median(sq_dists[sq_dists > 0])
        bandwidth = max(med, 1e-8)
    return np.exp(-sq_dists / (2.0 * bandwidth))


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _distance_matrix(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))


def _double_center_distance(D: np.ndarray) -> np.ndarray:
    """Double-center a distance matrix for dCor computation."""
    n = D.shape[0]
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean


# ===================================================================
# 1.  HSIC (Hilbert-Schmidt Independence Criterion)
# ===================================================================

class HSICTest(BaseCITest):
    """Hilbert-Schmidt Independence Criterion for CI testing.

    Tests X ⊥ Y | Z by regressing out Z from both X and Y, then
    computing the HSIC between the residuals.

    Reference: Gretton et al. (2005, 2008).
    """

    def __init__(
        self,
        *,
        n_permutations: int = 500,
        bandwidth: Optional[float] = None,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._n_perm = n_permutations
        self._bw = bandwidth

    def _regress_out(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Residuals of OLS regression X ~ Z."""
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
        try:
            beta = np.linalg.lstsq(Z_aug, X, rcond=None)[0]
            return X - Z_aug @ beta
        except np.linalg.LinAlgError:
            return X

    def _hsic_statistic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Biased HSIC estimator."""
        n = X.shape[0]
        Kx = _center_kernel(_rbf_kernel(X, bandwidth=self._bw))
        Ky = _center_kernel(_rbf_kernel(Y, bandwidth=self._bw))
        return float(np.trace(Kx @ Ky) / (n * n))

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        if Z is not None and Z.shape[1] > 0:
            X_res = self._regress_out(X, Z)
            Y_res = self._regress_out(Y, Z)
        else:
            X_res = X
            Y_res = Y

        stat = self._hsic_statistic(X_res, Y_res)

        null_stats = np.empty(self._n_perm)
        rng = np.random.RandomState(42)
        for i in range(self._n_perm):
            perm = rng.permutation(n)
            null_stats[i] = self._hsic_statistic(X_res[perm], Y_res)

        p_value = float(np.mean(null_stats >= stat))
        return CITestResult(
            statistic=stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 2.  Distance correlation CI test (dCor)
# ===================================================================

class DistanceCorrelationTest(BaseCITest):
    """Distance correlation test for conditional independence.

    Computes the distance correlation between residuals after
    regressing out the conditioning set.

    Reference: Székely, Rizzo & Bakirov (2007).
    """

    def __init__(
        self,
        *,
        n_permutations: int = 500,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._n_perm = n_permutations

    def _dcor_statistic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute the squared distance correlation."""
        n = X.shape[0]
        Dx = _double_center_distance(_distance_matrix(X))
        Dy = _double_center_distance(_distance_matrix(Y))

        dcov_xy = float(np.sum(Dx * Dy)) / (n * n)
        dcov_xx = float(np.sum(Dx * Dx)) / (n * n)
        dcov_yy = float(np.sum(Dy * Dy)) / (n * n)

        denom = math.sqrt(dcov_xx * dcov_yy)
        if denom < 1e-12:
            return 0.0
        return dcov_xy / denom

    def _regress_out(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
        try:
            beta = np.linalg.lstsq(Z_aug, X, rcond=None)[0]
            return X - Z_aug @ beta
        except np.linalg.LinAlgError:
            return X

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        if Z is not None and Z.shape[1] > 0:
            X_res = self._regress_out(X, Z)
            Y_res = self._regress_out(Y, Z)
        else:
            X_res = X
            Y_res = Y

        stat = self._dcor_statistic(X_res, Y_res)

        null_stats = np.empty(self._n_perm)
        rng = np.random.RandomState(42)
        for i in range(self._n_perm):
            perm = rng.permutation(n)
            null_stats[i] = self._dcor_statistic(X_res[perm], Y_res)

        p_value = float(np.mean(null_stats >= stat))
        return CITestResult(
            statistic=stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 3.  KSG mutual information estimator
# ===================================================================

def _ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """Kraskov-Stögbauer-Grassberger MI estimator (Algorithm 1).

    Reference: Kraskov, Stögbauer, Grassberger (2004).
    """
    from scipy.spatial import cKDTree

    n = X.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    XY = np.column_stack([X, Y])
    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)

    digamma = math.lgamma
    psi = lambda x: float(np.real(sp_stats.digamma(x))) if hasattr(sp_stats, 'digamma') else _digamma(x)

    mi_sum = 0.0
    for i in range(n):
        dists, _ = tree_xy.query(XY[i], k=k + 1)
        eps = dists[-1]
        if eps < 1e-12:
            eps = 1e-12

        nx = len(tree_x.query_ball_point(X[i], eps - 1e-15)) - 1
        ny = len(tree_y.query_ball_point(Y[i], eps - 1e-15)) - 1

        nx = max(nx, 1)
        ny = max(ny, 1)
        mi_sum += _digamma(nx) + _digamma(ny)

    mi = _digamma(k) - mi_sum / n + _digamma(n)
    return max(mi, 0.0)


def _digamma(x: int) -> float:
    """Digamma function for positive integers."""
    if x <= 0:
        return 0.0
    result = -0.5772156649
    for i in range(1, x):
        result += 1.0 / i
    return result


class KSGMutualInformationTest(BaseCITest):
    """CI test based on the KSG mutual information estimator.

    Tests X ⊥ Y | Z by estimating MI(X_res; Y_res) after regressing
    out Z, then using permutation to calibrate.
    """

    def __init__(
        self,
        *,
        k: int = 5,
        n_permutations: int = 500,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._k = k
        self._n_perm = n_permutations

    def _regress_out(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
        try:
            beta = np.linalg.lstsq(Z_aug, X, rcond=None)[0]
            return X - Z_aug @ beta
        except np.linalg.LinAlgError:
            return X

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        if Z is not None and Z.shape[1] > 0:
            X_res = self._regress_out(X, Z)
            Y_res = self._regress_out(Y, Z)
        else:
            X_res = X.ravel()
            Y_res = Y.ravel()

        stat = _ksg_mi(X_res, Y_res, k=self._k)

        null_stats = np.empty(self._n_perm)
        rng = np.random.RandomState(42)
        for i in range(self._n_perm):
            perm = rng.permutation(X_res.shape[0])
            null_stats[i] = _ksg_mi(X_res[perm], Y_res, k=self._k)

        p_value = float(np.mean(null_stats >= stat))
        return CITestResult(
            statistic=stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 4.  Conditional mutual information
# ===================================================================

def _conditional_mi_binned(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Binned estimator of conditional MI: I(X; Y | Z).

    Discretises Z into bins and computes MI(X; Y) within each bin,
    weighted by bin probability.
    """
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = X.shape[0]

    z_bins = np.zeros(n, dtype=int)
    for col in range(Z.shape[1]):
        try:
            _, edges = np.histogram(Z[:, col], bins=n_bins)
            col_bins = np.digitize(Z[:, col], edges[:-1]) - 1
            col_bins = np.clip(col_bins, 0, n_bins - 1)
            z_bins = z_bins * n_bins + col_bins
        except Exception:
            pass

    unique_bins = np.unique(z_bins)
    cmi = 0.0
    for b in unique_bins:
        mask = z_bins == b
        nb = mask.sum()
        if nb < 5:
            continue
        weight = nb / n
        mi_b = _ksg_mi(X[mask], Y[mask], k=min(3, nb - 1))
        cmi += weight * mi_b
    return max(cmi, 0.0)


class ConditionalMITest(BaseCITest):
    """Conditional mutual information based CI test."""

    def __init__(
        self,
        *,
        n_bins: int = 10,
        n_permutations: int = 300,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._n_bins = n_bins
        self._n_perm = n_permutations

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        if Z is None or Z.shape[1] == 0:
            stat = _ksg_mi(X.ravel(), Y.ravel())
        else:
            stat = _conditional_mi_binned(X.ravel(), Y.ravel(), Z, n_bins=self._n_bins)

        rng = np.random.RandomState(42)
        null_stats = np.empty(self._n_perm)
        n = X.shape[0]
        for i in range(self._n_perm):
            perm = rng.permutation(n)
            if Z is None or Z.shape[1] == 0:
                null_stats[i] = _ksg_mi(X.ravel()[perm], Y.ravel())
            else:
                null_stats[i] = _conditional_mi_binned(
                    X.ravel()[perm], Y.ravel(), Z, n_bins=self._n_bins
                )

        p_value = float(np.mean(null_stats >= stat))
        return CITestResult(
            statistic=stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 5.  Maximal information coefficient
# ===================================================================

def _compute_mic(X: np.ndarray, Y: np.ndarray, *, B: float = 0.6) -> float:
    """Approximate MIC via grid-based mutual information maximisation.

    Reference: Reshef et al. (2011).
    """
    n = len(X)
    max_grid = int(n ** B)
    max_grid = max(2, min(max_grid, 30))

    best_mi = 0.0
    for gx in range(2, max_grid + 1):
        for gy in range(2, max_grid + 1):
            if gx * gy > max_grid * 2:
                break
            try:
                x_bins = np.digitize(X, np.linspace(X.min(), X.max(), gx + 1)[:-1]) - 1
                y_bins = np.digitize(Y, np.linspace(Y.min(), Y.max(), gy + 1)[:-1]) - 1
                x_bins = np.clip(x_bins, 0, gx - 1)
                y_bins = np.clip(y_bins, 0, gy - 1)

                joint = np.zeros((gx, gy))
                for i in range(n):
                    joint[x_bins[i], y_bins[i]] += 1
                joint /= n

                px = joint.sum(axis=1)
                py = joint.sum(axis=0)

                mi = 0.0
                for xi in range(gx):
                    for yi in range(gy):
                        if joint[xi, yi] > 0 and px[xi] > 0 and py[yi] > 0:
                            mi += joint[xi, yi] * math.log(
                                joint[xi, yi] / (px[xi] * py[yi])
                            )

                mi_norm = mi / math.log(min(gx, gy)) if min(gx, gy) > 1 else 0.0
                best_mi = max(best_mi, mi_norm)
            except Exception:
                continue
    return best_mi


class MICTest(BaseCITest):
    """Maximal Information Coefficient based CI test."""

    def __init__(
        self,
        *,
        n_permutations: int = 500,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._n_perm = n_permutations

    def _regress_out(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
        try:
            beta = np.linalg.lstsq(Z_aug, X, rcond=None)[0]
            return X - Z_aug @ beta
        except np.linalg.LinAlgError:
            return X

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        if Z is not None and Z.shape[1] > 0:
            X_res = self._regress_out(X.ravel(), Z).ravel()
            Y_res = self._regress_out(Y.ravel(), Z).ravel()
        else:
            X_res = X.ravel()
            Y_res = Y.ravel()

        stat = _compute_mic(X_res, Y_res)

        null_stats = np.empty(self._n_perm)
        rng = np.random.RandomState(42)
        for i in range(self._n_perm):
            perm = rng.permutation(len(X_res))
            null_stats[i] = _compute_mic(X_res[perm], Y_res)

        p_value = float(np.mean(null_stats >= stat))
        return CITestResult(
            statistic=stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 6.  Randomised conditional correlation test
# ===================================================================

class RandomizedConditionalCorrelationTest(BaseCITest):
    """CI test using random nonlinear projections.

    Projects X and Y into random feature space, regresses out Z,
    and tests independence of residuals.

    Reference: Shah & Peters (2020).
    """

    def __init__(
        self,
        *,
        n_features: int = 50,
        n_permutations: int = 300,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._n_feat = n_features
        self._n_perm = n_permutations

    def _random_features(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        d = X.shape[1]
        W = rng.randn(d, self._n_feat) / math.sqrt(d)
        b = rng.uniform(0, 2 * math.pi, self._n_feat)
        return np.cos(X @ W + b) * math.sqrt(2.0 / self._n_feat)

    def _regress_out(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
        try:
            beta = np.linalg.lstsq(Z_aug, X, rcond=None)[0]
            return X - Z_aug @ beta
        except np.linalg.LinAlgError:
            return X

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        rng = np.random.RandomState(42)
        n = X.shape[0]

        X_feat = self._random_features(X, rng)
        Y_feat = self._random_features(Y, rng)

        if Z is not None and Z.shape[1] > 0:
            X_res = self._regress_out(X_feat, Z)
            Y_res = self._regress_out(Y_feat, Z)
        else:
            X_res = X_feat
            Y_res = Y_feat

        corr_matrix = (X_res.T @ Y_res) / n
        stat = float(np.sum(corr_matrix ** 2))

        null_stats = np.empty(self._n_perm)
        for i in range(self._n_perm):
            perm = rng.permutation(n)
            corr_perm = (X_res[perm].T @ Y_res) / n
            null_stats[i] = float(np.sum(corr_perm ** 2))

        p_value = float(np.mean(null_stats >= stat))
        return CITestResult(
            statistic=stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 7.  Friedman-Rafsky adapted CI test
# ===================================================================

class FriedmanRafskyTest(BaseCITest):
    """Graph-based two-sample test adapted for conditional independence.

    Builds a minimum spanning tree on (X, Y) residuals and counts
    the number of edges connecting different labels (above/below
    median of Y residuals).

    Reference: Friedman & Rafsky (1979).
    """

    def __init__(
        self,
        *,
        n_permutations: int = 500,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._n_perm = n_permutations

    def _regress_out(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
        try:
            beta = np.linalg.lstsq(Z_aug, X, rcond=None)[0]
            return X - Z_aug @ beta
        except np.linalg.LinAlgError:
            return X

    def _mst_edge_count(self, points: np.ndarray, labels: np.ndarray) -> int:
        """Count cross-label edges in the MST of *points*."""
        n = len(points)
        if n < 2:
            return 0
        D = _distance_matrix(points)
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        min_dist = D[0].copy()
        min_from = np.zeros(n, dtype=int)
        cross_edges = 0

        for _ in range(n - 1):
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            best = unvisited[np.argmin(min_dist[unvisited])]
            if labels[best] != labels[min_from[best]]:
                cross_edges += 1
            visited[best] = True
            for j in range(n):
                if not visited[j] and D[best, j] < min_dist[j]:
                    min_dist[j] = D[best, j]
                    min_from[j] = best
        return cross_edges

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        if Z is not None and Z.shape[1] > 0:
            X_res = self._regress_out(X.ravel(), Z)
            Y_res = self._regress_out(Y.ravel(), Z)
        else:
            X_res = X.ravel()
            Y_res = Y.ravel()

        labels = (Y_res > np.median(Y_res)).astype(int)
        if X_res.ndim == 1:
            X_res = X_res.reshape(-1, 1)
        stat = self._mst_edge_count(X_res, labels)

        rng = np.random.RandomState(42)
        null_stats = np.empty(self._n_perm)
        n = len(labels)
        for i in range(self._n_perm):
            perm_labels = labels[rng.permutation(n)]
            null_stats[i] = self._mst_edge_count(X_res, perm_labels)

        p_value = float(np.mean(null_stats >= stat))
        return CITestResult(
            statistic=float(stat),
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 8.  Classifier-based CI test
# ===================================================================

class ClassifierCITest(BaseCITest):
    """CI test based on a binary classifier.

    Tests X ⊥ Y | Z by training a classifier to predict Y-label
    from X (after regressing out Z).  If the classifier does no better
    than chance (measured by cross-validated accuracy), independence holds.

    Reference: Sen, Suresh, Shanmugam et al. (2017).
    """

    def __init__(
        self,
        *,
        n_splits: int = 5,
        n_permutations: int = 200,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._n_splits = n_splits
        self._n_perm = n_permutations

    def _regress_out(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
        try:
            beta = np.linalg.lstsq(Z_aug, X, rcond=None)[0]
            return X - Z_aug @ beta
        except np.linalg.LinAlgError:
            return X

    def _cv_accuracy(self, X: np.ndarray, labels: np.ndarray) -> float:
        """K-fold cross-validated accuracy using a simple linear classifier."""
        n = len(labels)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        indices = np.arange(n)
        fold_size = n // self._n_splits
        correct = 0
        total = 0

        for fold in range(self._n_splits):
            start = fold * fold_size
            end = start + fold_size if fold < self._n_splits - 1 else n
            test_mask = np.zeros(n, dtype=bool)
            test_mask[start:end] = True
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = labels[train_mask], labels[test_mask]

            if len(np.unique(y_train)) < 2:
                pred = np.full(len(y_test), y_train[0] if len(y_train) > 0 else 0)
            else:
                X_aug = np.column_stack([np.ones(X_train.shape[0]), X_train])
                try:
                    beta = np.linalg.lstsq(X_aug, y_train.astype(float), rcond=None)[0]
                    X_test_aug = np.column_stack([np.ones(X_test.shape[0]), X_test])
                    scores = X_test_aug @ beta
                    pred = (scores > 0.5).astype(int)
                except np.linalg.LinAlgError:
                    pred = np.zeros(len(y_test), dtype=int)

            correct += np.sum(pred == y_test)
            total += len(y_test)

        return correct / total if total > 0 else 0.5

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        if Z is not None and Z.shape[1] > 0:
            X_res = self._regress_out(X, Z)
            Y_res = self._regress_out(Y.ravel(), Z)
        else:
            X_res = X
            Y_res = Y.ravel()

        labels = (Y_res > np.median(Y_res)).astype(int)
        if X_res.ndim == 1:
            X_res = X_res.reshape(-1, 1)

        stat = self._cv_accuracy(X_res, labels)

        rng = np.random.RandomState(42)
        null_stats = np.empty(self._n_perm)
        n = len(labels)
        for i in range(self._n_perm):
            perm = rng.permutation(n)
            null_stats[i] = self._cv_accuracy(X_res[perm], labels)

        p_value = float(np.mean(null_stats >= stat))
        return CITestResult(
            statistic=stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )
