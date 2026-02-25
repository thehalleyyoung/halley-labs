"""Exact NTK computation: finite-width and infinite-width.

Provides:
  - EmpiricalNTK: finite-width NTK via Jacobian-vector products
  - AnalyticNTK: infinite-width NTK for common activations using kernel recursion
  - NTKComputer: unified interface
  - NTKTracker: track NTK evolution during training
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as sp_linalg


# ======================================================================
# Empirical (finite-width) NTK
# ======================================================================

class EmpiricalNTK:
    """Compute the finite-width Neural Tangent Kernel via Jacobians.

    For a network f(θ, x) with parameter vector θ ∈ R^P and output in R^C:
        Θ(x, x') = J(x) J(x')^T   where J(x) = ∂f/∂θ ∈ R^{C×P}
    """

    def __init__(self, output_dim: int = 1) -> None:
        self.output_dim = output_dim

    # ------------------------------------------------------------------
    # Core: Jacobian computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_jacobian(
        forward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        params: np.ndarray,
        x: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute the Jacobian ∂f(params, x)/∂params via finite differences.

        Parameters
        ----------
        forward_fn : callable (params, x) -> output  (1-D array of length C)
        params : 1-D array of length P
        x : single input sample
        eps : finite-difference step

        Returns
        -------
        J : ndarray of shape (C, P)
        """
        f0 = np.atleast_1d(forward_fn(params, x))
        C = len(f0)
        P = len(params)
        J = np.zeros((C, P), dtype=np.float64)

        for i in range(P):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = np.atleast_1d(forward_fn(params_plus, x))
            J[:, i] = (f_plus - f0) / eps

        return J

    @staticmethod
    def compute_jacobian_vp(
        jvp_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        params: np.ndarray,
        x: np.ndarray,
        num_params: int,
    ) -> np.ndarray:
        """Compute the Jacobian using Jacobian-vector products.

        Parameters
        ----------
        jvp_fn : callable (params, x, v) -> J @ v   where v ∈ R^P
        params : parameter vector
        x : single input
        num_params : P

        Returns
        -------
        J : ndarray of shape (C, P) built column-by-column
        """
        e = np.zeros(num_params, dtype=np.float64)
        columns: List[np.ndarray] = []
        for i in range(num_params):
            e[i] = 1.0
            col = np.atleast_1d(jvp_fn(params, x, e))
            columns.append(col)
            e[i] = 0.0
        return np.column_stack(columns)

    @staticmethod
    def compute_jacobian_vjp(
        vjp_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        params: np.ndarray,
        x: np.ndarray,
        output_dim: int,
    ) -> np.ndarray:
        """Compute the Jacobian using vector-Jacobian products (more efficient
        when C < P).

        Parameters
        ----------
        vjp_fn : callable (params, x, v) -> v^T @ J  where v ∈ R^C
        params : parameter vector
        x : single input
        output_dim : C

        Returns
        -------
        J : ndarray of shape (C, P) built row-by-row
        """
        e = np.zeros(output_dim, dtype=np.float64)
        rows: List[np.ndarray] = []
        for c in range(output_dim):
            e[c] = 1.0
            row = np.atleast_1d(vjp_fn(params, x, e))
            rows.append(row)
            e[c] = 0.0
        return np.vstack(rows)

    # ------------------------------------------------------------------
    # NTK matrix computation
    # ------------------------------------------------------------------

    def compute_ntk(
        self,
        forward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        params: np.ndarray,
        X: np.ndarray,
        X2: Optional[np.ndarray] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute the empirical NTK matrix.

        Parameters
        ----------
        forward_fn : callable (params, x) -> output  (1-D array)
        params : 1-D parameter vector
        X : (N, D) data matrix
        X2 : optional (M, D); if None, compute Θ(X, X)
        eps : finite-difference step

        Returns
        -------
        Theta : (N*C, M*C) NTK matrix  (or N*C x N*C if X2 is None)
        """
        N = X.shape[0]
        jacobians_1 = [self.compute_jacobian(forward_fn, params, X[i], eps) for i in range(N)]

        if X2 is None:
            jacobians_2 = jacobians_1
            M = N
        else:
            M = X2.shape[0]
            jacobians_2 = [self.compute_jacobian(forward_fn, params, X2[j], eps) for j in range(M)]

        C = jacobians_1[0].shape[0]
        Theta = np.zeros((N * C, M * C), dtype=np.float64)

        for i in range(N):
            Ji = jacobians_1[i]  # (C, P)
            for j in range(M):
                Jj = jacobians_2[j]  # (C, P)
                block = Ji @ Jj.T  # (C, C)
                Theta[i * C:(i + 1) * C, j * C:(j + 1) * C] = block

        return Theta

    def compute_ntk_trace(
        self,
        forward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        params: np.ndarray,
        X: np.ndarray,
        X2: Optional[np.ndarray] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute trace-NTK: Θ_tr(x, x') = Tr(J(x) J(x')^T).

        Returns an (N, M) matrix instead of (NC, MC).
        """
        N = X.shape[0]
        jacobians_1 = [self.compute_jacobian(forward_fn, params, X[i], eps) for i in range(N)]

        if X2 is None:
            jacobians_2 = jacobians_1
            M = N
        else:
            M = X2.shape[0]
            jacobians_2 = [self.compute_jacobian(forward_fn, params, X2[j], eps) for j in range(M)]

        Theta_tr = np.zeros((N, M), dtype=np.float64)
        for i in range(N):
            for j in range(M):
                Theta_tr[i, j] = np.sum(jacobians_1[i] * jacobians_2[j])

        return Theta_tr

    def compute_ntk_batched(
        self,
        forward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        params: np.ndarray,
        X: np.ndarray,
        batch_size: int = 32,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute trace-NTK in batches to manage memory."""
        N = X.shape[0]
        Theta = np.zeros((N, N), dtype=np.float64)

        for i_start in range(0, N, batch_size):
            i_end = min(i_start + batch_size, N)
            batch_i = X[i_start:i_end]
            J_i = [self.compute_jacobian(forward_fn, params, batch_i[k], eps)
                    for k in range(i_end - i_start)]
            for j_start in range(0, N, batch_size):
                j_end = min(j_start + batch_size, N)
                batch_j = X[j_start:j_end]
                J_j = [self.compute_jacobian(forward_fn, params, batch_j[k], eps)
                        for k in range(j_end - j_start)]
                for ii, Ji in enumerate(J_i):
                    for jj, Jj in enumerate(J_j):
                        Theta[i_start + ii, j_start + jj] = np.sum(Ji * Jj)

        return Theta


# ======================================================================
# Analytic (infinite-width) NTK via kernel recursion
# ======================================================================

class AnalyticNTK:
    """Infinite-width NTK via dual-activation kernel recursion.

    For an L-layer fully-connected network with activation sigma:
        K^0(x,x') = x . x' / d
        K^l(x,x') = sigma_w^2 * kappa(K^{l-1}) + sigma_b^2
        Theta^L = Sum_l ( Prod_{l'>l} dot_kappa^{l'} ) . K^{l-1} . sigma_w^2 / n_{l-1}
    """

    def __init__(
        self,
        depth: int,
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
        activation: str = "relu",
    ) -> None:
        self.depth = depth
        self.sigma_w_sq = sigma_w ** 2
        self.sigma_b_sq = sigma_b ** 2
        self.activation = activation
        self._kappa = self._get_kappa(activation)
        self._dot_kappa = self._get_dot_kappa(activation)

    # ------------------------------------------------------------------
    # Activation-specific dual kernels
    # ------------------------------------------------------------------

    @staticmethod
    def _get_kappa(activation: str) -> Callable[[float, float, float], float]:
        """Return κ(k_xx, k_yy, k_xy) = E[σ(u)σ(v)]."""
        if activation == "relu":
            return _kappa_relu
        elif activation == "gelu":
            return _kappa_gelu
        elif activation == "sigmoid":
            return _kappa_sigmoid
        elif activation == "tanh":
            return _kappa_tanh
        elif activation == "identity":
            return lambda kxx, kyy, kxy: kxy
        raise ValueError(f"Unknown activation: {activation}")

    @staticmethod
    def _get_dot_kappa(activation: str) -> Callable[[float, float, float], float]:
        """Return dot_kappa(k_xx, k_yy, k_xy) = E[sigma'(u)sigma'(v)]."""
        if activation == "relu":
            return _dot_kappa_relu
        elif activation == "gelu":
            return _dot_kappa_gelu
        elif activation == "sigmoid":
            return _dot_kappa_sigmoid
        elif activation == "tanh":
            return _dot_kappa_tanh
        elif activation == "identity":
            return lambda kxx, kyy, kxy: 1.0
        raise ValueError(f"Unknown activation: {activation}")

    # ------------------------------------------------------------------
    # NNGP kernel recursion
    # ------------------------------------------------------------------

    def nngp_kernel(self, X: np.ndarray) -> np.ndarray:
        """Compute the NNGP kernel K^L(X, X).

        Parameters
        ----------
        X : (N, D) input data

        Returns
        -------
        K : (N, N) NNGP kernel matrix after L layers
        """
        N, D = X.shape
        # Base kernel
        K = (X @ X.T) / D

        for l in range(self.depth):
            K_new = np.zeros_like(K)
            for i in range(N):
                for j in range(i, N):
                    val = self.sigma_w_sq * self._kappa(K[i, i], K[j, j], K[i, j]) + self.sigma_b_sq
                    K_new[i, j] = val
                    K_new[j, i] = val
            K = K_new

        return K

    def nngp_kernel_single(self, k_xx: float, k_yy: float, k_xy: float) -> Tuple[float, float, float]:
        """Propagate a single triple (k_xx, k_yy, k_xy) through L layers."""
        for l in range(self.depth):
            new_kxx = self.sigma_w_sq * self._kappa(k_xx, k_xx, k_xx) + self.sigma_b_sq
            new_kyy = self.sigma_w_sq * self._kappa(k_yy, k_yy, k_yy) + self.sigma_b_sq
            new_kxy = self.sigma_w_sq * self._kappa(k_xx, k_yy, k_xy) + self.sigma_b_sq
            k_xx, k_yy, k_xy = new_kxx, new_kyy, new_kxy
        return k_xx, k_yy, k_xy

    # ------------------------------------------------------------------
    # Full NTK computation
    # ------------------------------------------------------------------

    def compute_ntk(self, X: np.ndarray) -> np.ndarray:
        """Compute the infinite-width NTK Θ^L(X, X).

        Uses the recursion:
            Theta^L = Sum_{l=0}^{L-1} sigma_w^2 . K^l . Prod_{l'=l+1}^{L-1} dot_kappa^{l'}

        Parameters
        ----------
        X : (N, D) input

        Returns
        -------
        Theta : (N, N) NTK matrix
        """
        N, D = X.shape
        K_base = (X @ X.T) / D

        # Store kernel at each layer
        K_layers: List[np.ndarray] = [K_base.copy()]
        K = K_base.copy()
        for l in range(self.depth):
            K_new = np.zeros_like(K)
            for i in range(N):
                for j in range(i, N):
                    val = self.sigma_w_sq * self._kappa(K[i, i], K[j, j], K[i, j]) + self.sigma_b_sq
                    K_new[i, j] = val
                    K_new[j, i] = val
            K = K_new
            K_layers.append(K.copy())

        # Compute derivative kernels at each layer
        dot_K_layers: List[np.ndarray] = []
        for l in range(self.depth):
            K_prev = K_layers[l]
            dK = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(i, N):
                    val = self._dot_kappa(K_prev[i, i], K_prev[j, j], K_prev[i, j])
                    dK[i, j] = val
                    dK[j, i] = val
            dot_K_layers.append(dK)

        # NTK = sum over layers
        Theta = np.zeros((N, N), dtype=np.float64)
        for l in range(self.depth):
            # Product of dot_kappas from layer l+1 to L-1
            prod = np.ones((N, N), dtype=np.float64)
            for lp in range(l + 1, self.depth):
                prod *= dot_K_layers[lp]
            Theta += self.sigma_w_sq * K_layers[l] * prod

        # Add bias contribution
        Theta += self.sigma_b_sq

        return Theta

    def compute_ntk_single(self, k_xx: float, k_yy: float, k_xy: float) -> float:
        """Compute scalar NTK for a single pair of inputs."""
        k_layers = [(k_xx, k_yy, k_xy)]
        kx, ky, kxy = k_xx, k_yy, k_xy
        for l in range(self.depth):
            new_kx = self.sigma_w_sq * self._kappa(kx, kx, kx) + self.sigma_b_sq
            new_ky = self.sigma_w_sq * self._kappa(ky, ky, ky) + self.sigma_b_sq
            new_kxy = self.sigma_w_sq * self._kappa(kx, ky, kxy) + self.sigma_b_sq
            kx, ky, kxy = new_kx, new_ky, new_kxy
            k_layers.append((kx, ky, kxy))

        # Derivative products
        dot_layers = []
        for l in range(self.depth):
            kx_l, ky_l, kxy_l = k_layers[l]
            dot_layers.append(self._dot_kappa(kx_l, ky_l, kxy_l))

        theta = 0.0
        for l in range(self.depth):
            prod = 1.0
            for lp in range(l + 1, self.depth):
                prod *= dot_layers[lp]
            _, _, kxy_l = k_layers[l]
            theta += self.sigma_w_sq * kxy_l * prod

        theta += self.sigma_b_sq
        return theta

    # ------------------------------------------------------------------
    # Eigendecomposition
    # ------------------------------------------------------------------

    @staticmethod
    def eigendecompose(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition of a kernel matrix.

        Returns
        -------
        eigenvalues : sorted descending
        eigenvectors : columns are eigenvectors
        """
        eigvals, eigvecs = np.linalg.eigh(K)
        idx = np.argsort(eigvals)[::-1]
        return eigvals[idx], eigvecs[:, idx]

    @staticmethod
    def effective_rank(K: np.ndarray, threshold: float = 0.99) -> int:
        """Effective rank: minimum k such that top-k eigenvalues capture
        `threshold` fraction of total spectral mass."""
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(K)))[::-1]
        total = np.sum(eigvals)
        if total < 1e-15:
            return 0
        cumulative = np.cumsum(eigvals) / total
        return int(np.searchsorted(cumulative, threshold)) + 1

    @staticmethod
    def spectral_entropy(K: np.ndarray) -> float:
        """Spectral entropy: -Σ p_i log(p_i), where p_i = λ_i / Σ λ_j."""
        eigvals = np.abs(np.linalg.eigvalsh(K))
        eigvals = eigvals[eigvals > 1e-15]
        p = eigvals / np.sum(eigvals)
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def condition_number(K: np.ndarray) -> float:
        eigvals = np.abs(np.linalg.eigvalsh(K))
        eigvals = eigvals[eigvals > 1e-15]
        if len(eigvals) == 0:
            return float("inf")
        return float(eigvals[0] / eigvals[-1])


# ======================================================================
# NTK alignment
# ======================================================================

def kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """Centered kernel alignment (CKA) between two kernel matrices.

    CKA(K1, K2) = <K1_c, K2_c>_F / (||K1_c||_F · ||K2_c||_F)
    where K_c = H K H with H = I - 11^T/n (centering matrix).
    """
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H
    num = np.sum(K1c * K2c)
    denom = np.sqrt(np.sum(K1c * K1c) * np.sum(K2c * K2c))
    if denom < 1e-15:
        return 0.0
    return float(num / denom)


def kernel_alignment_uncentered(K1: np.ndarray, K2: np.ndarray) -> float:
    """Uncentered kernel alignment: <K1, K2>_F / (||K1||_F · ||K2||_F)."""
    num = np.sum(K1 * K2)
    denom = np.linalg.norm(K1, "fro") * np.linalg.norm(K2, "fro")
    if denom < 1e-15:
        return 0.0
    return float(num / denom)


def target_alignment(
    ntk: np.ndarray, y: np.ndarray
) -> float:
    """Alignment between NTK and target kernel y y^T."""
    if y.ndim == 1:
        y = y[:, None]
    K_target = y @ y.T
    return kernel_alignment(ntk, K_target)


# ======================================================================
# Unified NTK computer
# ======================================================================

class NTKComputer:
    """Unified interface for NTK computation.

    Automatically selects between empirical and analytic computation
    based on available information.
    """

    def __init__(
        self,
        mode: str = "analytic",
        depth: int = 2,
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
        activation: str = "relu",
        output_dim: int = 1,
    ) -> None:
        self.mode = mode
        if mode == "analytic":
            self._analytic = AnalyticNTK(depth, sigma_w, sigma_b, activation)
            self._empirical: Optional[EmpiricalNTK] = None
        else:
            self._analytic = None
            self._empirical = EmpiricalNTK(output_dim)

    def compute(
        self,
        X: np.ndarray,
        forward_fn: Optional[Callable] = None,
        params: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute the NTK matrix."""
        if self.mode == "analytic":
            assert self._analytic is not None
            return self._analytic.compute_ntk(X)
        else:
            assert self._empirical is not None
            assert forward_fn is not None and params is not None
            return self._empirical.compute_ntk_trace(forward_fn, params, X, **kwargs)

    def eigendecompose(self, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return AnalyticNTK.eigendecompose(K)

    def alignment(self, K1: np.ndarray, K2: np.ndarray) -> float:
        return kernel_alignment(K1, K2)

    def target_alignment(self, K: np.ndarray, y: np.ndarray) -> float:
        return target_alignment(K, y)


# ======================================================================
# NTK evolution tracker
# ======================================================================

@dataclass
class NTKSnapshot:
    """Snapshot of NTK properties at a point during training."""
    step: int
    ntk_trace: float
    top_eigenvalue: float
    effective_rank: int
    spectral_entropy: float
    condition_number: float
    alignment_with_init: float
    alignment_with_target: Optional[float] = None
    eigenvalues: Optional[np.ndarray] = None


class NTKTracker:
    """Track NTK evolution during training.

    Records snapshots of the NTK at specified intervals and computes
    diagnostics for lazy-vs-rich regime identification.
    """

    def __init__(
        self,
        compute_fn: Callable[[], np.ndarray],
        y: Optional[np.ndarray] = None,
        record_eigenvalues: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        compute_fn : callable returning the current NTK matrix (N, N)
        y : target labels for alignment tracking
        record_eigenvalues : if True, store full eigenvalue arrays
        """
        self._compute_fn = compute_fn
        self._y = y
        self._record_eigenvalues = record_eigenvalues
        self._snapshots: List[NTKSnapshot] = []
        self._init_ntk: Optional[np.ndarray] = None

    def record(self, step: int) -> NTKSnapshot:
        """Record a snapshot of the current NTK."""
        K = self._compute_fn()
        if self._init_ntk is None:
            self._init_ntk = K.copy()

        eigvals = np.sort(np.abs(np.linalg.eigvalsh(K)))[::-1]
        trace = float(np.trace(K))
        top_eig = float(eigvals[0]) if len(eigvals) > 0 else 0.0
        eff_rank = AnalyticNTK.effective_rank(K)
        entropy = AnalyticNTK.spectral_entropy(K)
        cond = AnalyticNTK.condition_number(K)
        align_init = kernel_alignment(K, self._init_ntk)

        align_target = None
        if self._y is not None:
            align_target = target_alignment(K, self._y)

        snap = NTKSnapshot(
            step=step,
            ntk_trace=trace,
            top_eigenvalue=top_eig,
            effective_rank=eff_rank,
            spectral_entropy=entropy,
            condition_number=cond,
            alignment_with_init=align_init,
            alignment_with_target=align_target,
            eigenvalues=eigvals.copy() if self._record_eigenvalues else None,
        )
        self._snapshots.append(snap)
        return snap

    @property
    def snapshots(self) -> List[NTKSnapshot]:
        return list(self._snapshots)

    def is_lazy_regime(self, threshold: float = 0.95) -> bool:
        """Detect lazy regime: NTK stays close to initialisation."""
        if len(self._snapshots) < 2:
            return True
        return self._snapshots[-1].alignment_with_init >= threshold

    def ntk_change_rate(self) -> List[float]:
        """Fractional change in NTK trace between consecutive snapshots."""
        rates = []
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1].ntk_trace
            curr = self._snapshots[i].ntk_trace
            if abs(prev) > 1e-15:
                rates.append(abs(curr - prev) / abs(prev))
            else:
                rates.append(0.0)
        return rates

    def eigenvalue_trajectory(self, k: int = 5) -> np.ndarray:
        """Return top-k eigenvalues over time as (T, k) array."""
        if not self._record_eigenvalues or not self._snapshots:
            return np.array([])
        rows = []
        for snap in self._snapshots:
            if snap.eigenvalues is not None:
                rows.append(snap.eigenvalues[:k])
        if not rows:
            return np.array([])
        max_len = max(len(r) for r in rows)
        padded = [np.pad(r, (0, max_len - len(r))) for r in rows]
        return np.array(padded)

    def summary(self) -> Dict[str, Any]:
        if not self._snapshots:
            return {"num_snapshots": 0}
        first = self._snapshots[0]
        last = self._snapshots[-1]
        return {
            "num_snapshots": len(self._snapshots),
            "initial_trace": first.ntk_trace,
            "final_trace": last.ntk_trace,
            "trace_change": abs(last.ntk_trace - first.ntk_trace) / max(abs(first.ntk_trace), 1e-15),
            "initial_eff_rank": first.effective_rank,
            "final_eff_rank": last.effective_rank,
            "final_alignment_with_init": last.alignment_with_init,
            "regime": "lazy" if self.is_lazy_regime() else "rich",
        }


# ======================================================================
# Dual activation kernels (closed-form)
# ======================================================================

def _kappa_relu(k_xx: float, k_yy: float, k_xy: float) -> float:
    """Arc-cosine kernel of order 1 (ReLU dual)."""
    denom = math.sqrt(max(k_xx * k_yy, 1e-30))
    c = np.clip(k_xy / denom, -1.0, 1.0)
    theta = math.acos(float(c))
    return (1.0 / (2.0 * math.pi)) * denom * (math.sin(theta) + (math.pi - theta) * float(c))


def _dot_kappa_relu(k_xx: float, k_yy: float, k_xy: float) -> float:
    """Derivative dual for ReLU: arc-cosine kernel of order 0."""
    denom = math.sqrt(max(k_xx * k_yy, 1e-30))
    c = np.clip(k_xy / denom, -1.0, 1.0)
    theta = math.acos(float(c))
    return (math.pi - theta) / (2.0 * math.pi)


def _kappa_gelu(k_xx: float, k_yy: float, k_xy: float) -> float:
    """GELU dual activation via Monte-Carlo integration."""
    return _mc_dual_kernel(k_xx, k_yy, k_xy, _gelu_scalar)


def _dot_kappa_gelu(k_xx: float, k_yy: float, k_xy: float) -> float:
    return _mc_dual_kernel(k_xx, k_yy, k_xy, _gelu_deriv_scalar)


def _kappa_sigmoid(k_xx: float, k_yy: float, k_xy: float) -> float:
    """Sigmoid dual activation.

    Uses the Williams (1997) formula for erf-like activations:
    E[σ(u)σ(v)] ≈ closed-form via arcsine kernel.
    """
    # Approximation via arc-sine kernel (exact for probit ~ sigmoid)
    denom = math.sqrt((1 + 2 * k_xx) * (1 + 2 * k_yy))
    if denom < 1e-15:
        return 0.25
    arg = np.clip(2 * k_xy / denom, -1.0, 1.0)
    return (1.0 / (2.0 * math.pi)) * math.asin(float(arg)) + 0.25


def _dot_kappa_sigmoid(k_xx: float, k_yy: float, k_xy: float) -> float:
    return _mc_dual_kernel(k_xx, k_yy, k_xy, _sigmoid_deriv_scalar)


def _kappa_tanh(k_xx: float, k_yy: float, k_xy: float) -> float:
    """Tanh dual activation via arc-sine kernel."""
    denom = math.sqrt((1 + 2 * k_xx) * (1 + 2 * k_yy))
    if denom < 1e-15:
        return 0.0
    arg = np.clip(2 * k_xy / denom, -1.0, 1.0)
    return (2.0 / math.pi) * math.asin(float(arg))


def _dot_kappa_tanh(k_xx: float, k_yy: float, k_xy: float) -> float:
    return _mc_dual_kernel(k_xx, k_yy, k_xy, _tanh_deriv_scalar)


# ======================================================================
# Scalar activation helpers for MC integration
# ======================================================================

def _gelu_scalar(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def _gelu_deriv_scalar(x: float) -> float:
    cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf


def _sigmoid_deriv_scalar(x: float) -> float:
    s = 1.0 / (1.0 + math.exp(-min(max(x, -500), 500)))
    return s * (1.0 - s)


def _tanh_deriv_scalar(x: float) -> float:
    return 1.0 - math.tanh(x) ** 2


def _mc_dual_kernel(
    k_xx: float,
    k_yy: float,
    k_xy: float,
    func: Callable[[float], float],
    n_samples: int = 40_000,
) -> float:
    """Monte-Carlo estimate of E[f(u)f(v)] with (u,v) ~ N(0, Σ)."""
    rng = np.random.default_rng(42)
    cov = np.array([[k_xx, k_xy], [k_xy, k_yy]], dtype=np.float64)
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < 0:
        cov += (abs(np.min(eigvals)) + 1e-8) * np.eye(2)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += 1e-6 * np.eye(2)
        L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_samples, 2))
    uv = z @ L.T
    fu = np.array([func(float(u)) for u in uv[:, 0]])
    fv = np.array([func(float(v)) for v in uv[:, 1]])
    return float(np.mean(fu * fv))
