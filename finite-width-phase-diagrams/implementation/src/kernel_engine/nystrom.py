"""Nyström approximation for kernel matrices.

Provides low-rank kernel approximation via landmark (inducing) point selection,
adaptive rank determination, error bounds, and integration with NTK computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as sp_linalg


# ======================================================================
# Landmark (inducing) point selection
# ======================================================================

class LandmarkSelector:
    """Select landmark points for Nyström approximation."""

    def __init__(self, method: str = "uniform", seed: int = 42) -> None:
        """
        Parameters
        ----------
        method : one of "uniform", "kmeans", "greedy", "leverage"
        seed : random seed
        """
        self.method = method
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        X: np.ndarray,
        m: int,
        kernel_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """Select m landmark indices from X.

        Parameters
        ----------
        X : (N, D) data matrix
        m : number of landmarks
        kernel_fn : optional kernel function (X_sub, X_sub) -> K for
                    greedy/leverage methods

        Returns
        -------
        indices : (m,) integer array of selected indices
        """
        N = X.shape[0]
        m = min(m, N)

        if self.method == "uniform":
            return self._uniform(N, m)
        elif self.method == "kmeans":
            return self._kmeans(X, m)
        elif self.method == "greedy":
            return self._greedy(X, m, kernel_fn)
        elif self.method == "leverage":
            return self._leverage_score(X, m, kernel_fn)
        else:
            raise ValueError(f"Unknown landmark selection method: {self.method}")

    def _uniform(self, N: int, m: int) -> np.ndarray:
        """Uniform random sampling without replacement."""
        return self.rng.choice(N, size=m, replace=False)

    def _kmeans(self, X: np.ndarray, m: int, max_iter: int = 50) -> np.ndarray:
        """K-means++ initialisation followed by assignment; return nearest-to-centroid indices."""
        N, D = X.shape

        # K-means++ seeding
        centers = np.zeros((m, D), dtype=np.float64)
        first_idx = self.rng.integers(N)
        centers[0] = X[first_idx]
        dists = np.full(N, np.inf)

        for k in range(1, m):
            d = np.sum((X - centers[k - 1]) ** 2, axis=1)
            dists = np.minimum(dists, d)
            probs = dists / (dists.sum() + 1e-30)
            centers[k] = X[self.rng.choice(N, p=probs)]

        # Lloyd's iterations
        for _ in range(max_iter):
            # Assignment
            dist_matrix = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (N, m)
            labels = np.argmin(dist_matrix, axis=1)
            # Update
            new_centers = np.zeros_like(centers)
            for k in range(m):
                mask = labels == k
                if mask.any():
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]
            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers

        # Find nearest data point to each centroid
        dist_matrix = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        indices = np.argmin(dist_matrix, axis=0)
        # Ensure uniqueness
        indices = np.unique(indices)
        while len(indices) < m:
            remaining = np.setdiff1d(np.arange(N), indices)
            extra = self.rng.choice(remaining, size=min(m - len(indices), len(remaining)), replace=False)
            indices = np.concatenate([indices, extra])
        return indices[:m]

    def _greedy(
        self,
        X: np.ndarray,
        m: int,
        kernel_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """Greedy column-pivoted Cholesky selection.

        Selects points that maximally reduce the approximation error.
        """
        N = X.shape[0]
        if kernel_fn is None:
            # Fall back to RBF kernel with median bandwidth
            dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
            median_dist = np.median(dists[dists > 0])
            gamma = 1.0 / max(median_dist, 1e-10)

            def kernel_fn(Xi: np.ndarray, Xj: np.ndarray) -> np.ndarray:
                d = np.sum((Xi[:, None] - Xj[None, :]) ** 2, axis=2)
                return np.exp(-gamma * d)

        # Diagonal of the kernel (used for greedy selection)
        diag = np.array([float(kernel_fn(X[i:i+1], X[i:i+1])[0, 0]) for i in range(N)])
        residual_diag = diag.copy()
        selected: List[int] = []
        L = np.zeros((N, m), dtype=np.float64)  # partial Cholesky factor

        for j in range(m):
            # Select point with largest residual diagonal
            remaining = np.setdiff1d(np.arange(N), selected)
            idx_in_remaining = np.argmax(residual_diag[remaining])
            pivot = remaining[idx_in_remaining]
            selected.append(pivot)

            # Compute column of kernel
            k_col = np.array([
                float(kernel_fn(X[pivot:pivot+1], X[i:i+1])[0, 0]) for i in range(N)
            ])

            # Update Cholesky factor
            if j == 0:
                L[:, 0] = k_col
                L[pivot, 0] = math.sqrt(max(diag[pivot], 1e-15))
                L[:, 0] /= L[pivot, 0]
            else:
                v = k_col - L[:, :j] @ L[pivot, :j]
                denom = math.sqrt(max(residual_diag[pivot], 1e-15))
                L[:, j] = v / denom

            residual_diag -= L[:, j] ** 2
            residual_diag = np.maximum(residual_diag, 0.0)

        return np.array(selected)

    def _leverage_score(
        self,
        X: np.ndarray,
        m: int,
        kernel_fn: Optional[Callable] = None,
        rank: int = 50,
    ) -> np.ndarray:
        """Ridge leverage score sampling."""
        N = X.shape[0]
        if kernel_fn is None:
            K = X @ X.T / X.shape[1]
        else:
            K = kernel_fn(X, X)

        # Compute leverage scores via truncated eigendecomposition
        k = min(rank, N)
        eigvals, eigvecs = np.linalg.eigh(K)
        idx = np.argsort(eigvals)[::-1][:k]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Ridge parameter
        lam = np.sum(eigvals[k-1:]) / max(N - k + 1, 1)

        # Leverage scores: l_i = Σ_j v_{ij}^2 λ_j / (λ_j + λ)
        scores = np.sum(eigvecs ** 2 * (eigvals / (eigvals + lam + 1e-15))[None, :], axis=1)
        scores /= scores.sum() + 1e-15

        return self.rng.choice(N, size=m, replace=False, p=scores)


# ======================================================================
# Adaptive rank selector
# ======================================================================

class AdaptiveRankSelector:
    """Determine the optimal rank for Nyström approximation."""

    def __init__(
        self,
        min_rank: int = 5,
        max_rank: int = 500,
        tol: float = 1e-3,
        criterion: str = "eigenvalue_gap",
    ) -> None:
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.tol = tol
        self.criterion = criterion

    def select_rank(
        self,
        eigenvalues: np.ndarray,
        n: Optional[int] = None,
    ) -> int:
        """Select rank based on eigenvalue distribution.

        Parameters
        ----------
        eigenvalues : sorted descending eigenvalues of the kernel sub-matrix
        n : total dataset size (for scaling)

        Returns
        -------
        k : selected rank
        """
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-15]

        if len(eigenvalues) == 0:
            return self.min_rank

        if self.criterion == "eigenvalue_gap":
            return self._eigenvalue_gap(eigenvalues)
        elif self.criterion == "cumulative_energy":
            return self._cumulative_energy(eigenvalues)
        elif self.criterion == "elbow":
            return self._elbow(eigenvalues)
        elif self.criterion == "aic":
            return self._aic(eigenvalues, n or len(eigenvalues))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _eigenvalue_gap(self, eigvals: np.ndarray) -> int:
        """Find rank where consecutive eigenvalue ratio drops below tolerance."""
        for k in range(1, len(eigvals)):
            ratio = eigvals[k] / (eigvals[k - 1] + 1e-15)
            if ratio < self.tol:
                return max(k, self.min_rank)
        return min(len(eigvals), self.max_rank)

    def _cumulative_energy(self, eigvals: np.ndarray) -> int:
        """Rank capturing (1 - tol) fraction of total energy."""
        total = np.sum(eigvals)
        if total < 1e-15:
            return self.min_rank
        cumsum = np.cumsum(eigvals) / total
        threshold = 1.0 - self.tol
        k = int(np.searchsorted(cumsum, threshold)) + 1
        return max(min(k, self.max_rank), self.min_rank)

    def _elbow(self, eigvals: np.ndarray) -> int:
        """Elbow detection via maximum curvature."""
        if len(eigvals) < 3:
            return self.min_rank
        log_eigs = np.log(eigvals + 1e-15)
        # Discrete second derivative
        d2 = np.diff(log_eigs, 2)
        if len(d2) == 0:
            return self.min_rank
        k = int(np.argmax(np.abs(d2))) + 1
        return max(min(k, self.max_rank), self.min_rank)

    def _aic(self, eigvals: np.ndarray, n: int) -> int:
        """Akaike Information Criterion for rank selection."""
        best_aic = float("inf")
        best_k = self.min_rank
        p = len(eigvals)

        for k in range(self.min_rank, min(p, self.max_rank) + 1):
            # Log-likelihood approximation
            signal = eigvals[:k]
            noise_var = np.mean(eigvals[k:]) if k < p else 1e-10
            ll = -0.5 * n * (
                np.sum(np.log(signal + 1e-15))
                + (p - k) * np.log(noise_var + 1e-15)
            )
            num_params = k * (2 * p - k + 1) / 2
            aic = -2 * ll + 2 * num_params
            if aic < best_aic:
                best_aic = aic
                best_k = k

        return best_k


# ======================================================================
# Nyström approximation
# ======================================================================

@dataclass
class NystromResult:
    """Result of a Nyström approximation."""
    K_approx: np.ndarray
    landmarks: np.ndarray
    rank: int
    reconstruction_error: Optional[float] = None
    relative_error: Optional[float] = None
    error_bound: Optional[float] = None
    eigenvalues_landmark: Optional[np.ndarray] = None


class NystromApproximation:
    """Low-rank Nyström approximation of kernel matrices.

    Given a kernel K and m landmark points, approximates:
        K ≈ K_{nm} K_{mm}^{-1} K_{mn}

    where K_{nm} is the N×m cross-kernel and K_{mm} is the m×m sub-kernel.
    """

    def __init__(
        self,
        landmark_method: str = "uniform",
        rank_method: str = "eigenvalue_gap",
        regularization: float = 1e-8,
        seed: int = 42,
    ) -> None:
        self.selector = LandmarkSelector(method=landmark_method, seed=seed)
        self.rank_selector = AdaptiveRankSelector(criterion=rank_method)
        self.reg = regularization

    def approximate(
        self,
        X: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        m: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> NystromResult:
        """Compute the Nyström approximation.

        Parameters
        ----------
        X : (N, D) data matrix
        kernel_fn : callable (X1, X2) -> K  kernel function
        m : number of landmarks (auto if None)
        landmarks : pre-selected landmark indices

        Returns
        -------
        NystromResult with approximate kernel and diagnostics
        """
        N = X.shape[0]

        # Select landmarks
        if landmarks is not None:
            idx = landmarks
            m = len(idx)
        else:
            if m is None:
                m = min(max(int(math.sqrt(N)), 10), N)
            idx = self.selector.select(X, m, kernel_fn)

        X_m = X[idx]

        # Compute sub-matrices
        K_mm = kernel_fn(X_m, X_m)  # (m, m)
        K_nm = kernel_fn(X, X_m)    # (N, m)

        # Regularise K_mm
        K_mm += self.reg * np.eye(m)

        # Eigendecomposition of K_mm for stable inversion
        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, self.reg)

        # Adaptive rank selection
        rank = self.rank_selector.select_rank(eigvals, n=N)
        rank = min(rank, m)

        # Truncated pseudo-inverse
        idx_top = np.argsort(eigvals)[::-1][:rank]
        U_r = eigvecs[:, idx_top]
        S_r = eigvals[idx_top]
        K_mm_inv_half = U_r @ np.diag(1.0 / np.sqrt(S_r))  # (m, rank)

        # Nyström approximation: K ≈ (K_nm K_mm^{-1/2}) (K_mm^{-1/2})^T K_mn
        C = K_nm @ K_mm_inv_half  # (N, rank)
        K_approx = C @ C.T  # (N, N)

        # Error bounds
        error_bound = self._compute_error_bound(eigvals, rank, N, m)

        return NystromResult(
            K_approx=K_approx,
            landmarks=idx,
            rank=rank,
            error_bound=error_bound,
            eigenvalues_landmark=np.sort(eigvals)[::-1],
        )

    def approximate_with_exact(
        self,
        K_full: np.ndarray,
        m: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> NystromResult:
        """Nyström approximation given the full kernel matrix.

        Useful for benchmarking against exact computation.
        """
        N = K_full.shape[0]

        if landmarks is not None:
            idx = landmarks
            m = len(idx)
        else:
            if m is None:
                m = min(max(int(math.sqrt(N)), 10), N)
            idx = self.selector._uniform(N, m)

        K_mm = K_full[np.ix_(idx, idx)]
        K_nm = K_full[:, idx]

        K_mm += self.reg * np.eye(m)

        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, self.reg)

        rank = self.rank_selector.select_rank(eigvals, n=N)
        rank = min(rank, m)

        idx_top = np.argsort(eigvals)[::-1][:rank]
        U_r = eigvecs[:, idx_top]
        S_r = eigvals[idx_top]
        K_mm_inv_half = U_r @ np.diag(1.0 / np.sqrt(S_r))

        C = K_nm @ K_mm_inv_half
        K_approx = C @ C.T

        # Compute actual errors
        recon_err = float(np.linalg.norm(K_full - K_approx, "fro"))
        K_full_norm = float(np.linalg.norm(K_full, "fro"))
        rel_err = recon_err / max(K_full_norm, 1e-15)
        error_bound = self._compute_error_bound(eigvals, rank, N, m)

        return NystromResult(
            K_approx=K_approx,
            landmarks=idx,
            rank=rank,
            reconstruction_error=recon_err,
            relative_error=rel_err,
            error_bound=error_bound,
            eigenvalues_landmark=np.sort(eigvals)[::-1],
        )

    def _compute_error_bound(
        self,
        eigvals: np.ndarray,
        rank: int,
        N: int,
        m: int,
    ) -> float:
        """Compute theoretical error bound for Nyström approximation.

        Using the result:  ||K - K_nyst||_F <= (N/m) Σ_{i>rank} λ_i
        """
        sorted_eigs = np.sort(eigvals)[::-1]
        tail_sum = np.sum(sorted_eigs[rank:])
        return float((N / max(m, 1)) * tail_sum)

    # ------------------------------------------------------------------
    # Integration with NTK
    # ------------------------------------------------------------------

    def approximate_ntk(
        self,
        ntk_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        m: Optional[int] = None,
    ) -> NystromResult:
        """Approximate the NTK using Nyström.

        Parameters
        ----------
        ntk_fn : callable X -> Theta(X, X) returning NTK matrix
        X : (N, D) data
        m : number of landmarks
        """
        def kernel_fn(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
            # Build combined kernel via full NTK on landmarks
            combined = np.vstack([X1, X2])
            K_full = ntk_fn(combined)
            n1 = X1.shape[0]
            return K_full[:n1, n1:]

        N = X.shape[0]
        if m is None:
            m = min(max(int(math.sqrt(N)), 10), N)

        idx = self.selector.select(X, m)
        X_m = X[idx]

        # Compute K_mm via NTK
        K_mm = ntk_fn(X_m)

        # Compute K_nm: NTK between all points and landmarks
        K_nm = np.zeros((N, m), dtype=np.float64)
        # Compute row-by-row using the NTK of [x_i, X_m]
        for i in range(N):
            combined = np.vstack([X[i:i+1], X_m])
            K_block = ntk_fn(combined)
            K_nm[i, :] = K_block[0, 1:]

        K_mm += self.reg * np.eye(m)
        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, self.reg)

        rank = self.rank_selector.select_rank(eigvals, n=N)
        rank = min(rank, m)

        idx_top = np.argsort(eigvals)[::-1][:rank]
        U_r = eigvecs[:, idx_top]
        S_r = eigvals[idx_top]
        K_mm_inv_half = U_r @ np.diag(1.0 / np.sqrt(S_r))

        C = K_nm @ K_mm_inv_half
        K_approx = C @ C.T

        return NystromResult(
            K_approx=K_approx,
            landmarks=idx,
            rank=rank,
            error_bound=self._compute_error_bound(eigvals, rank, N, m),
            eigenvalues_landmark=np.sort(eigvals)[::-1],
        )

    # ------------------------------------------------------------------
    # Low-rank factor
    # ------------------------------------------------------------------

    def low_rank_factor(
        self,
        X: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        m: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (L, idx) such that K ≈ L L^T with L of shape (N, rank).

        Useful for operations requiring explicit low-rank factors.
        """
        result = self.approximate(X, kernel_fn, m=m)
        N = X.shape[0]
        idx = result.landmarks

        X_m = X[idx]
        K_mm = kernel_fn(X_m, X_m) + self.reg * np.eye(len(idx))
        K_nm = kernel_fn(X, X_m)

        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, self.reg)

        rank = result.rank
        top = np.argsort(eigvals)[::-1][:rank]
        U_r = eigvecs[:, top]
        S_r = eigvals[top]

        L = K_nm @ U_r @ np.diag(1.0 / np.sqrt(S_r))  # (N, rank)
        return L, idx
