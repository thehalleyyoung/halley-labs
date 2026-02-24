"""Kernel functions including adaptive and manifold-adaptive variants."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .utils import psd_check, nearest_psd


class Kernel(ABC):
    """Base kernel class."""

    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate kernel K(x, y)."""
        ...

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute Gram matrix K_ij = K(X_i, X_j)."""
        n = X.shape[0]
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = self.evaluate(X[i], X[j])
                G[i, j] = val
                G[j, i] = val
        return G


class RBFKernel(Kernel):
    """RBF (Gaussian) kernel: K(x,y) = exp(-||x-y||^2 / (2*sigma^2))."""

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return float(np.exp(-np.dot(diff, diff) / (2.0 * self.bandwidth ** 2)))


class MaternKernel(Kernel):
    """Matern kernel family.

    For nu=0.5: exponential kernel
    For nu=1.5: once-differentiable Matern
    For nu=2.5: twice-differentiable Matern
    For nu->inf: RBF kernel
    """

    def __init__(self, nu: float = 1.5, length_scale: float = 1.0):
        self.nu = nu
        self.length_scale = length_scale

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        d = np.linalg.norm(x - y) / self.length_scale
        if d < 1e-12:
            return 1.0
        if self.nu == 0.5:
            return float(np.exp(-d))
        elif self.nu == 1.5:
            s = np.sqrt(3.0) * d
            return float((1.0 + s) * np.exp(-s))
        elif self.nu == 2.5:
            s = np.sqrt(5.0) * d
            return float((1.0 + s + s ** 2 / 3.0) * np.exp(-s))
        else:
            # Fallback to RBF approximation for large nu
            return float(np.exp(-0.5 * d ** 2))


class CosineKernel(Kernel):
    """Cosine similarity kernel: K(x,y) = x·y / (||x|| ||y||)."""

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx < 1e-12 or ny < 1e-12:
            return 0.0
        return float(np.dot(x, y) / (nx * ny))

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_norm = X / norms
        G = X_norm @ X_norm.T
        return G


class PolynomialKernel(Kernel):
    """Polynomial kernel: K(x,y) = (x·y + c)^d."""

    def __init__(self, degree: int = 2, c: float = 1.0):
        self.degree = degree
        self.c = c

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((np.dot(x, y) + self.c) ** self.degree)


class SpectralKernel(Kernel):
    """Random Fourier features approximation of RBF kernel.

    Uses Rahimi & Recht (2007) random features.
    """

    def __init__(self, n_components: int = 100, bandwidth: float = 1.0, seed: int = 42):
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.rng = np.random.RandomState(seed)
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None

    def _init_features(self, dim: int) -> None:
        if self._W is None or self._W.shape[1] != dim:
            self._W = self.rng.randn(self.n_components, dim) / self.bandwidth
            self._b = self.rng.uniform(0, 2 * np.pi, self.n_components)

    def _features(self, x: np.ndarray) -> np.ndarray:
        self._init_features(x.shape[0])
        z = np.sqrt(2.0 / self.n_components) * np.cos(self._W @ x + self._b)
        return z

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        zx = self._features(x)
        zy = self._features(y)
        return float(np.dot(zx, zy))

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        self._init_features(X.shape[1])
        Z = np.sqrt(2.0 / self.n_components) * np.cos(X @ self._W.T + self._b)
        return Z @ Z.T


class AdaptiveRBFKernel(Kernel):
    """Adaptive RBF kernel that learns bandwidth from data.

    Uses median heuristic as initialization, then refines via
    leave-one-out cross-validation of the kernel density estimate.
    """

    def __init__(self, initial_bandwidth: float = 1.0):
        self.bandwidth = initial_bandwidth
        self._distances: list = []

    def update(self, new_points: np.ndarray) -> None:
        """Update bandwidth estimate from observed points via LOO-CV."""
        n = new_points.shape[0]
        if n < 2:
            return
        # Collect pairwise distances
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(new_points[i] - new_points[j])
                dists.append(d)
        self._distances.extend(dists)

        if len(self._distances) < 2:
            return

        all_dists = np.array(self._distances)
        median_dist = np.median(all_dists)

        if median_dist < 1e-12:
            return

        # LOO-CV: find bandwidth that maximizes leave-one-out log-likelihood
        candidates = [
            median_dist * f for f in [0.1, 0.25, 0.5, 0.707, 1.0, 1.5, 2.0, 3.0]
        ]

        best_bw = median_dist / np.sqrt(2.0)
        best_score = -float('inf')

        for bw in candidates:
            if bw < 1e-12:
                continue
            # LOO kernel density estimate
            score = 0.0
            for i in range(n):
                loo_log_dens = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    d_sq = np.sum((new_points[i] - new_points[j]) ** 2)
                    loo_log_dens += np.exp(-d_sq / (2.0 * bw ** 2))
                loo_log_dens = max(loo_log_dens / (n - 1), 1e-30)
                score += np.log(loo_log_dens)

            if score > best_score:
                best_score = score
                best_bw = bw

        self.bandwidth = float(best_bw)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return float(np.exp(-np.dot(diff, diff) / (2.0 * self.bandwidth ** 2)))


class ManifoldAdaptiveKernel(Kernel):
    """Manifold-adaptive kernel using local PCA for geodesic distance approximation.

    At each point, estimates the local tangent space via PCA of nearby points,
    then computes geodesic-approximated distances using the local metric tensor.
    K(x,y) = exp(-d_manifold(x,y)^2 / (2*sigma^2))
    """

    def __init__(self, bandwidth: float = 1.0, n_neighbors: int = 10, n_components: int = None):
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self._points: Optional[np.ndarray] = None
        self._local_bases: dict = {}

    def fit(self, X: np.ndarray) -> None:
        """Fit local PCA at each point."""
        self._points = X.copy()
        n = X.shape[0]
        k = min(self.n_neighbors, n - 1)
        if k < 1:
            return

        for i in range(n):
            # Find k nearest neighbors
            dists = np.linalg.norm(X - X[i], axis=1)
            nn_idx = np.argsort(dists)[1: k + 1]
            neighbors = X[nn_idx] - X[i]

            # Local PCA
            if neighbors.shape[0] < 2:
                self._local_bases[i] = np.eye(X.shape[1])
                continue

            cov = neighbors.T @ neighbors / neighbors.shape[0]
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            n_comp = self.n_components or X.shape[1]
            # Metric tensor: inverse of eigenvalues (stretch less-variable directions)
            metric_diag = np.ones(X.shape[1])
            for j in range(min(n_comp, len(eigvals))):
                if eigvals[j] > 1e-10:
                    metric_diag[j] = 1.0 / eigvals[j]
                else:
                    metric_diag[j] = 1.0
            self._local_bases[i] = eigvecs @ np.diag(metric_diag) @ eigvecs.T

    def _manifold_distance(self, x: np.ndarray, y: np.ndarray, idx_x: int = -1) -> float:
        """Approximate geodesic distance using local metric tensor."""
        diff = x - y
        if idx_x >= 0 and idx_x in self._local_bases:
            M = self._local_bases[idx_x]
            return float(np.sqrt(max(0.0, diff @ M @ diff)))
        return float(np.linalg.norm(diff))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        # Use Euclidean distance if not fitted
        if self._points is None:
            d = np.linalg.norm(x - y)
        else:
            # Find closest reference point to x
            dists = np.linalg.norm(self._points - x, axis=1)
            idx = int(np.argmin(dists))
            d = self._manifold_distance(x, y, idx)
        return float(np.exp(-d ** 2 / (2.0 * self.bandwidth ** 2)))

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        n = X.shape[0]
        G = np.zeros((n, n))
        for i in range(n):
            G[i, i] = 1.0
            for j in range(i + 1, n):
                d = self._manifold_distance(X[i], X[j], i)
                val = np.exp(-d ** 2 / (2.0 * self.bandwidth ** 2))
                G[i, j] = val
                G[j, i] = val
        return G


class QualityDiversityKernel(Kernel):
    """Quality-weighted kernel for DPP: L_ij = q_i * q_j * K(phi_i, phi_j)."""

    def __init__(self, base_kernel: Kernel):
        self.base_kernel = base_kernel
        self._qualities: Optional[np.ndarray] = None

    def set_qualities(self, qualities: np.ndarray) -> None:
        self._qualities = qualities

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.base_kernel.evaluate(x, y)

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        G = self.base_kernel.gram_matrix(X)
        if self._qualities is not None:
            q = self._qualities
            G = G * np.outer(q, q)
        return G


class MultiScaleKernel(Kernel):
    """Multi-scale kernel: weighted combination of RBF kernels at different bandwidths.

    K(x,y) = sum_s w_s * exp(-||x-y||^2 / (2*sigma_s^2))

    Learns optimal weights from data via LOO cross-validation,
    capturing structure at multiple length scales simultaneously.
    """

    def __init__(self, bandwidths: Optional[list] = None, n_scales: int = 5):
        if bandwidths is not None:
            self.bandwidths = list(bandwidths)
        else:
            # Log-spaced default bandwidths
            self.bandwidths = [0.1 * (3.0 ** i) for i in range(n_scales)]
        self.weights = np.ones(len(self.bandwidths)) / len(self.bandwidths)
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Learn optimal scale weights from data via LOO cross-validation."""
        n = X.shape[0]
        if n < 3:
            return
        n_scales = len(self.bandwidths)

        # Compute per-scale Gram matrices
        scale_grams = []
        for bw in self.bandwidths:
            rbf = RBFKernel(bandwidth=bw)
            scale_grams.append(rbf.gram_matrix(X))

        # LOO log-likelihood for each weight configuration
        # Use simplex projection of scores
        scores = np.zeros(n_scales)
        for s in range(n_scales):
            G = scale_grams[s]
            # LOO density estimate
            loo_score = 0.0
            for i in range(n):
                loo_density = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    loo_density += G[i, j]
                loo_density /= (n - 1)
                loo_score += np.log(max(loo_density, 1e-30))
            scores[s] = loo_score

        # Convert to weights via softmax
        scores = scores - np.max(scores)  # numerical stability
        self.weights = np.exp(scores) / np.sum(np.exp(scores))
        self._fitted = True

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        d_sq = np.dot(diff, diff)
        val = 0.0
        for w, bw in zip(self.weights, self.bandwidths):
            val += w * np.exp(-d_sq / (2.0 * bw ** 2))
        return float(val)

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            self.fit(X)
        n = X.shape[0]
        G = np.zeros((n, n))
        for s, (w, bw) in enumerate(zip(self.weights, self.bandwidths)):
            rbf = RBFKernel(bandwidth=bw)
            G += w * rbf.gram_matrix(X)
        return G
