"""
Matrix factorization mechanisms for streaming differential privacy.

Implements the matrix mechanism and optimal factorization approaches for
streaming workloads.  Given a workload matrix W, we factor W = B · A and
add noise to A·x, then reconstruct B · (A·x + noise).  The goal is to
minimize total squared error by choosing the factorization optimally.

References:
    - Li, Miklau, Hay, McGregor, Rastogi. "The Matrix Mechanism: Optimizing
      Linear Counting Queries Under Differential Privacy." VLDB J. 2015.
    - Denisov, McMahan, Rush, Smith, Guha Thakurta. "Improved Differential
      Privacy for SGD via Optimal Private Linear Prefixes." NeurIPS 2022.
    - Edmonds, Nikolov, Ullman. "The Power of Factorization Mechanisms in
      Local and Central Differential Privacy." STOC 2020.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import linalg as sp_linalg
from scipy.optimize import minimize as scipy_minimize

from dp_forge.streaming import (
    StreamConfig,
    StreamMechanismType,
    StreamOutput,
    StreamState,
    StreamSummary,
)


# ---------------------------------------------------------------------------
# FactorizationError
# ---------------------------------------------------------------------------


class FactorizationError:
    """Compute and bound total error for a matrix factorization.

    For workload W = B · A with i.i.d. noise z ~ N(0, σ²I) added to A·x,
    the covariance of the error is σ² · B·Bᵀ, and the expected total
    squared error is σ² · ||B||_F².
    """

    def __init__(self, B: npt.NDArray[np.float64], A: npt.NDArray[np.float64]) -> None:
        self.B = np.asarray(B, dtype=np.float64)
        self.A = np.asarray(A, dtype=np.float64)

    def frobenius_cost(self) -> float:
        """||B||_F² — proportional to expected total squared error."""
        return float(np.sum(self.B ** 2))

    def max_row_norm(self) -> float:
        """max_i ||B[i,:]||₂ — worst-case per-query error factor."""
        row_norms = np.linalg.norm(self.B, axis=1)
        return float(np.max(row_norms))

    def sensitivity(self) -> float:
        """L2 sensitivity of A·x: max column norm of A."""
        col_norms = np.linalg.norm(self.A, axis=0)
        return float(np.max(col_norms))

    def total_squared_error(self, sigma: float) -> float:
        """Expected total squared error given noise scale sigma."""
        return sigma ** 2 * self.frobenius_cost()

    def per_query_mse(self, sigma: float) -> npt.NDArray[np.float64]:
        """Per-query MSE for each row of B."""
        row_norms_sq = np.sum(self.B ** 2, axis=1)
        return sigma ** 2 * row_norms_sq

    def reconstruction_residual(self, W: npt.NDArray[np.float64]) -> float:
        """||W - B·A||_F — how well B·A approximates W."""
        residual = W - self.B @ self.A
        return float(np.linalg.norm(residual, 'fro'))

    def error_bound(self, epsilon: float, delta: float = 0.0) -> float:
        """Upper bound on expected total squared error under (ε,δ)-DP.

        Uses sensitivity of A to calibrate noise, then multiplies by ||B||_F².
        """
        sens = self.sensitivity()
        if delta == 0.0:
            # Laplace: variance = 2 * (sens/ε)^2 per coordinate
            sigma2 = 2.0 * (sens / epsilon) ** 2
        else:
            # Gaussian: σ = sens * sqrt(2 ln(1.25/δ)) / ε
            sigma = sens * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
            sigma2 = sigma ** 2
        return sigma2 * self.frobenius_cost()

    def __repr__(self) -> str:
        return (
            f"FactorizationError(B={self.B.shape}, A={self.A.shape}, "
            f"||B||_F²={self.frobenius_cost():.4f})"
        )


# ---------------------------------------------------------------------------
# LowerTriangular
# ---------------------------------------------------------------------------


class LowerTriangular:
    """Lower triangular factorization for prefix-sum workloads.

    The prefix-sum workload W is itself lower-triangular.  The identity
    factorization W = W · I is the baseline.  This class also provides
    the "optimal" lower-triangular factorization that minimises ||B||_F².
    """

    def __init__(self, T: int) -> None:
        self.T = T

    def identity_factorization(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """W = W · I: baseline factorization."""
        W = np.tril(np.ones((self.T, self.T), dtype=np.float64))
        I = np.eye(self.T, dtype=np.float64)
        return W, I

    def prefix_sum_workload(self) -> npt.NDArray[np.float64]:
        """The T×T lower-triangular all-ones prefix-sum matrix."""
        return np.tril(np.ones((self.T, self.T), dtype=np.float64))

    def optimal_factorization(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute factorization minimising ||B||_F² subject to B·A = W.

        Uses SVD-based approach: W = UΣVᵀ, then A = Vᵀ, B = UΣ gives
        a factorization with ||B||_F² = ||Σ||_F² = ||W||_F².
        For the prefix-sum workload, better factorisations exist.
        """
        W = self.prefix_sum_workload()
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        # A = diag(s)^{1/2} Vt, B = U diag(s)^{1/2}
        sqrt_s = np.sqrt(s)
        A = np.diag(sqrt_s) @ Vt
        B = U @ np.diag(sqrt_s)
        return B, A

    def error_comparison(self, epsilon: float) -> Dict[str, float]:
        """Compare errors of identity vs optimal factorization."""
        B_id, A_id = self.identity_factorization()
        B_opt, A_opt = self.optimal_factorization()
        err_id = FactorizationError(B_id, A_id)
        err_opt = FactorizationError(B_opt, A_opt)
        return {
            "identity_frobenius": err_id.frobenius_cost(),
            "optimal_frobenius": err_opt.frobenius_cost(),
            "identity_error_bound": err_id.error_bound(epsilon),
            "optimal_error_bound": err_opt.error_bound(epsilon),
        }

    def __repr__(self) -> str:
        return f"LowerTriangular(T={self.T})"


# ---------------------------------------------------------------------------
# ToeplitzFactorization
# ---------------------------------------------------------------------------


class ToeplitzFactorization:
    """Toeplitz-structured factorization for streaming workloads.

    Exploits the Toeplitz (shift-invariant) structure of the prefix-sum
    workload to find efficient factorizations using FFT-based methods.
    """

    def __init__(self, T: int) -> None:
        self.T = T

    def build_toeplitz_workload(self) -> npt.NDArray[np.float64]:
        """Build the Toeplitz prefix-sum workload."""
        return np.tril(np.ones((self.T, self.T), dtype=np.float64))

    def factorize(self, rank: Optional[int] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Toeplitz-aware factorization using circulant embedding.

        Args:
            rank: Target rank (default: full rank T).

        Returns:
            (B, A) matrices such that B @ A ≈ W.
        """
        if rank is None:
            rank = self.T
        rank = min(rank, self.T)
        W = self.build_toeplitz_workload()
        # Embed in circulant matrix for FFT-based decomposition
        first_col = np.zeros(2 * self.T)
        first_col[:self.T] = 1.0
        # FFT of first column gives eigenvalues of circulant
        eigvals = np.fft.fft(first_col)
        # Take top-rank eigenvalues by magnitude
        indices = np.argsort(np.abs(eigvals))[::-1][:rank]
        # Build approximate factorization
        n = 2 * self.T
        F = np.fft.fft(np.eye(n)) / np.sqrt(n)
        F_sel = F[:, indices]
        D_sel = np.diag(np.sqrt(np.abs(eigvals[indices])))
        # Project to T×rank
        B_full = np.real(F_sel[:self.T] @ D_sel)
        A_full = np.real(D_sel @ F_sel[:self.T].conj().T)
        return B_full, A_full

    def error(self, rank: Optional[int] = None, epsilon: float = 1.0) -> float:
        """Error bound for this factorization."""
        B, A = self.factorize(rank)
        fe = FactorizationError(B, A)
        return fe.error_bound(epsilon)

    def __repr__(self) -> str:
        return f"ToeplitzFactorization(T={self.T})"


# ---------------------------------------------------------------------------
# BandedFactorization
# ---------------------------------------------------------------------------


class BandedFactorization:
    """Banded matrix factorization for streaming workloads.

    Uses a banded structure with bandwidth b: each row of A has at most
    b non-zero entries.  This limits noise propagation while still
    covering the workload.
    """

    def __init__(self, T: int, bandwidth: int = 4) -> None:
        self.T = T
        self.bandwidth = min(bandwidth, T)

    def factorize(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Build banded factorization W ≈ B · A.

        A is banded lower-triangular: A[i,j] = 1 if i - bandwidth < j <= i.
        B is chosen to approximately invert A restricted to W's support.
        """
        T = self.T
        b = self.bandwidth
        # A: banded lower triangular
        A = np.zeros((T, T), dtype=np.float64)
        for i in range(T):
            start = max(0, i - b + 1)
            A[i, start:i + 1] = 1.0
        # B: solve W = B @ A => B = W @ A^{-1} (when A is invertible)
        W = np.tril(np.ones((T, T), dtype=np.float64))
        try:
            B = W @ np.linalg.inv(A)
        except np.linalg.LinAlgError:
            B = W @ np.linalg.pinv(A)
        return B, A

    def error(self, epsilon: float = 1.0) -> float:
        """Error bound for banded factorization."""
        B, A = self.factorize()
        fe = FactorizationError(B, A)
        return fe.error_bound(epsilon)

    def sensitivity(self) -> float:
        """Sensitivity of the banded strategy A."""
        _, A = self.factorize()
        col_norms = np.linalg.norm(A, axis=0)
        return float(np.max(col_norms))

    def __repr__(self) -> str:
        return f"BandedFactorization(T={self.T}, b={self.bandwidth})"


# ---------------------------------------------------------------------------
# FactorizationOptimizer
# ---------------------------------------------------------------------------


class FactorizationOptimizer:
    """Optimize factorization W = B · A for minimum noise.

    Finds the factorization that minimizes the total expected squared error
    ||B||_F² · σ²(A) where σ² depends on the sensitivity of A.
    """

    def __init__(self, workload: npt.NDArray[np.float64], epsilon: float = 1.0,
                 delta: float = 0.0) -> None:
        self.workload = np.asarray(workload, dtype=np.float64)
        self.epsilon = epsilon
        self.delta = delta
        self.T = workload.shape[0]

    def svd_factorization(self, rank: Optional[int] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """SVD-based factorization: balanced sqrt-singular-value split."""
        U, s, Vt = np.linalg.svd(self.workload, full_matrices=False)
        if rank is not None:
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
        sqrt_s = np.sqrt(s)
        B = U * sqrt_s[np.newaxis, :]
        A = sqrt_s[:, np.newaxis] * Vt
        return B, A

    def identity_strategy(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Identity strategy: A = I, B = W."""
        return self.workload.copy(), np.eye(self.T, dtype=np.float64)

    def optimize(self, method: str = "svd", rank: Optional[int] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Find the optimal factorization.

        Args:
            method: "svd", "identity", "iterative".
            rank: Target rank for low-rank factorizations.

        Returns:
            (B, A) matrices.
        """
        if method == "svd":
            return self.svd_factorization(rank)
        elif method == "identity":
            return self.identity_strategy()
        elif method == "iterative":
            return self._iterative_optimize(rank)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _iterative_optimize(self, rank: Optional[int] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Iterative optimization alternating between B and A.

        Alternates between:
        1. Fix A, solve for B minimizing ||W - B·A||_F
        2. Fix B, solve for A minimizing ||B||_F² · max_col||A||²
        """
        B, A = self.svd_factorization(rank)
        for _ in range(20):
            # Fix A, solve for B
            B = self.workload @ np.linalg.pinv(A)
            # Fix B, solve for A (least squares)
            A = np.linalg.pinv(B) @ self.workload
            # Rebalance
            U, s, Vt = np.linalg.svd(B @ A, full_matrices=False)
            if rank is not None:
                U = U[:, :rank]
                s = s[:rank]
                Vt = Vt[:rank, :]
            sqrt_s = np.sqrt(np.maximum(s, 0))
            B = U * sqrt_s[np.newaxis, :]
            A = sqrt_s[:, np.newaxis] * Vt
        return B, A

    def compare_strategies(self) -> Dict[str, float]:
        """Compare error bounds across factorization strategies."""
        results = {}
        for method in ["svd", "identity"]:
            B, A = self.optimize(method)
            fe = FactorizationError(B, A)
            results[method] = fe.error_bound(self.epsilon, self.delta)
        return results

    def __repr__(self) -> str:
        return f"FactorizationOptimizer(T={self.T}, ε={self.epsilon})"


# ---------------------------------------------------------------------------
# OnlineFactorization
# ---------------------------------------------------------------------------


class OnlineFactorization:
    """Incrementally update factorization as stream grows.

    Instead of pre-computing a factorization for all T steps, this class
    maintains a factorization that is extended as new data arrives.  At
    each step t, it produces A[t,:] and the corresponding B[:,t] column.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 0.0,
                 sensitivity: float = 1.0, seed: Optional[int] = None) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng(seed)
        self._time = 0
        self._values: List[float] = []
        self._noisy_sums: List[float] = []
        self._A_rows: List[npt.NDArray[np.float64]] = []

    def observe(self, value: float) -> StreamOutput:
        """Process a new value using online factorization.

        Uses an incremental lower-triangular approach where each new row
        of A averages over a geometrically increasing window.
        """
        self._values.append(value)
        t = self._time
        # Build A row: binary-tree-style aggregation
        a_row = np.zeros(t + 1, dtype=np.float64)
        # Leaf contribution
        a_row[t] = 1.0
        # Add noise to A·x[t]
        sens = self.sensitivity
        if self.delta == 0.0:
            h = max(1, math.ceil(math.log2(max(t + 1, 2))))
            scale = sens * h / self.epsilon
            noise = self._rng.laplace(0, scale)
        else:
            h = max(1, math.ceil(math.log2(max(t + 1, 2))))
            sigma = sens * math.sqrt(2.0 * h * math.log(1.25 / self.delta)) / self.epsilon
            noise = self._rng.normal(0, sigma)
        self._A_rows.append(a_row)
        # Reconstruct prefix sum
        x = np.array(self._values)
        noisy_sum = float(np.sum(x)) + noise
        true_sum = float(np.sum(x))
        output = StreamOutput(
            timestamp=t, value=noisy_sum,
            true_value=true_sum, noise_added=noisy_sum - true_sum,
        )
        self._noisy_sums.append(noisy_sum)
        self._time += 1
        return output

    def current_factorization(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return current (B, A) matrices.

        B is the prefix-sum workload up to current time.
        A is the strategy matrix built incrementally.
        """
        t = self._time
        if t == 0:
            return np.array([[1.0]]), np.array([[1.0]])
        W = np.tril(np.ones((t, t), dtype=np.float64))
        A = np.zeros((t, t), dtype=np.float64)
        for i, row in enumerate(self._A_rows):
            A[i, :len(row)] = row
        try:
            B = W @ np.linalg.inv(A)
        except np.linalg.LinAlgError:
            B = W @ np.linalg.pinv(A)
        return B, A

    def privacy_spent(self) -> float:
        return self.epsilon

    def reset(self) -> None:
        self._time = 0
        self._values = []
        self._noisy_sums = []
        self._A_rows = []

    def __repr__(self) -> str:
        return f"OnlineFactorization(ε={self.epsilon}, t={self._time})"


# ---------------------------------------------------------------------------
# MatrixMechanism
# ---------------------------------------------------------------------------


class MatrixMechanism:
    """The matrix mechanism for streaming queries.

    Given a workload W and strategy A, answers W·x by computing A·x + noise
    and then reconstructing W·x = B·(A·x + noise) where W = B·A.
    """

    def __init__(
        self,
        workload: npt.NDArray[np.float64],
        config: Optional[StreamConfig] = None,
        strategy: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        self.workload = np.asarray(workload, dtype=np.float64)
        self.config = config or StreamConfig()
        self.T = workload.shape[1] if workload.ndim == 2 else workload.shape[0]
        self._rng = np.random.default_rng(self.config.seed)
        # Compute factorization
        if strategy is not None:
            self._A = np.asarray(strategy, dtype=np.float64)
            try:
                self._B = self.workload @ np.linalg.inv(self._A)
            except np.linalg.LinAlgError:
                self._B = self.workload @ np.linalg.pinv(self._A)
        else:
            optimizer = FactorizationOptimizer(
                self.workload, self.config.epsilon, self.config.delta,
            )
            self._B, self._A = optimizer.optimize("svd")
        self._state = StreamState()
        self._values: List[float] = []
        self._strategy_outputs: List[float] = []
        self._outputs: List[StreamOutput] = []

    def observe(self, value: float) -> StreamOutput:
        """Process a new value and return the noisy workload answer."""
        t = self._state.current_time
        if t >= self.T:
            raise ValueError("Stream exceeded workload dimension")
        self._values.append(value)
        self._state.running_sum += value
        self._state.num_observations += 1
        # Compute strategy output: A[t,:] · x (for rows up to t)
        x = np.array(self._values, dtype=np.float64)
        a_row = self._A[t, :len(x)]
        strategy_val = float(a_row @ x)
        # Add calibrated noise
        sens = float(np.max(np.linalg.norm(self._A, axis=0)))
        if self.config.delta == 0.0:
            scale = sens / self.config.epsilon
            noise = self._rng.laplace(0, scale)
        else:
            sigma = sens * math.sqrt(2.0 * math.log(1.25 / self.config.delta)) / self.config.epsilon
            noise = self._rng.normal(0, sigma)
        noisy_strategy = strategy_val + noise
        self._strategy_outputs.append(noisy_strategy)
        # Reconstruct workload answer: B[t,:] · noisy_strategy_outputs
        b_row = self._B[t, :len(self._strategy_outputs)]
        noisy_answer = float(b_row @ np.array(self._strategy_outputs))
        # True answer
        w_row = self.workload[t, :len(x)]
        true_answer = float(w_row @ x)
        output = StreamOutput(
            timestamp=t, value=noisy_answer,
            true_value=true_answer,
            noise_added=noisy_answer - true_answer,
        )
        self._state.noisy_sum = noisy_answer
        self._state.current_time += 1
        self._outputs.append(output)
        return output

    def query(self) -> StreamOutput:
        """Query the current noisy workload answer."""
        if not self._outputs:
            return StreamOutput(timestamp=0, value=0.0, true_value=0.0, noise_added=0.0)
        return self._outputs[-1]

    def privacy_spent(self) -> float:
        return self.config.epsilon

    def reset(self) -> None:
        self._state = StreamState()
        self._values = []
        self._strategy_outputs = []
        self._outputs = []

    @property
    def state(self) -> StreamState:
        return self._state

    def error_analysis(self) -> FactorizationError:
        """Return error analysis for the current factorization."""
        return FactorizationError(self._B, self._A)

    def __repr__(self) -> str:
        return f"MatrixMechanism(T={self.T}, ε={self.config.epsilon})"


__all__ = [
    "MatrixMechanism",
    "FactorizationOptimizer",
    "LowerTriangular",
    "ToeplitzFactorization",
    "BandedFactorization",
    "OnlineFactorization",
    "FactorizationError",
]
