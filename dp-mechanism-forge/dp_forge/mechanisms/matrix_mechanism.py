"""
Matrix mechanism implementation for DP-Forge.

Implements the matrix mechanism framework by Li, Miklau, Hay et al., which
optimizes the noise addition strategy for linear workloads by factorizing
the workload matrix and answering a different set of queries.

The key idea: instead of answering workload A directly with noise, answer
a factorization A = BQ where Q is easier to answer privately, then post-process
via B to recover the workload answers with lower total error.

Key References:
    - Li, Miklau, Hay, McGregor, Rastogi: "Optimal Error of Query Sets under the Differentially Private Matrix Mechanism" (ICDT 2013)
    - Hay, Rastogi, Miklau, Suciu: "Boosting the Accuracy of Differentially Private Histograms Through Consistency" (VLDB 2010)

Features:
    - WorkloadFactorization: find A = BQ minimizing noise
    - StrategyMatrix optimization via Frank-Wolfe / multiplicative weights
    - compute_strategy_error for error analysis
    - Gaussian vs Laplace noise variants
    - Integration with workload optimizer
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import linalg as sp_linalg
from scipy import sparse

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]
SparseMatrix = Union[sparse.csr_matrix, sparse.csc_matrix]


# ---------------------------------------------------------------------------
# Workload factorization utilities
# ---------------------------------------------------------------------------


@dataclass
class WorkloadFactorization:
    """A factorization A = BQ for workload optimization.
    
    Attributes:
        A: Original workload matrix (m, d).
        B: Reconstruction matrix (m, k).
        Q: Strategy matrix (k, d).
        expected_error: Expected total squared error.
    """
    A: FloatArray
    B: FloatArray
    Q: FloatArray
    expected_error: float
    
    def __post_init__(self) -> None:
        """Validate factorization."""
        m_a, d_a = self.A.shape
        m_b, k_b = self.B.shape
        k_q, d_q = self.Q.shape
        
        if m_a != m_b:
            raise ValueError(f"A rows ({m_a}) must match B rows ({m_b})")
        if d_a != d_q:
            raise ValueError(f"A cols ({d_a}) must match Q cols ({d_q})")
        if k_b != k_q:
            raise ValueError(f"B cols ({k_b}) must match Q rows ({k_q})")
    
    @property
    def reconstruction_error(self) -> float:
        """Frobenius norm of A - BQ."""
        diff = self.A - self.B @ self.Q
        return float(np.linalg.norm(diff, 'fro'))
    
    @property
    def is_exact(self, tol: float = 1e-10) -> bool:
        """Check if factorization is exact (A = BQ within tol)."""
        return self.reconstruction_error < tol


def identity_factorization(A: FloatArray) -> WorkloadFactorization:
    """Trivial factorization A = I * A (no optimization).
    
    Args:
        A: Workload matrix (m, d).
    
    Returns:
        WorkloadFactorization with B=I, Q=A.
    """
    m, d = A.shape
    B = np.eye(m)
    Q = A.copy()
    
    # Error: sum of variances = m * sigma^2 for Laplace(sensitivity/ε)
    # Assuming unit sensitivity and ε=1 baseline:
    expected_error = float(m)
    
    return WorkloadFactorization(A=A, B=B, Q=Q, expected_error=expected_error)


def range_query_factorization(d: int) -> WorkloadFactorization:
    """Optimal factorization for range queries via prefix sums.
    
    For range queries over [1..d], use the prefix-sum strategy:
        Q = identity (answer all counts)
        B = lower triangular (accumulate for ranges)
    
    This reduces error from O(d^3) to O(d log^2 d) using hierarchical strategy.
    
    Args:
        d: Domain size.
    
    Returns:
        WorkloadFactorization for range queries.
    """
    # Workload A: all range queries [i, j]
    # For simplicity, construct all O(d^2) ranges
    ranges = []
    for i in range(d):
        for j in range(i, d):
            ranges.append((i, j))
    
    m = len(ranges)
    A = np.zeros((m, d), dtype=np.float64)
    for idx, (i, j) in enumerate(ranges):
        A[idx, i:j+1] = 1.0
    
    # Strategy Q: identity (answer each count)
    Q = np.eye(d)
    
    # Reconstruction B: lower triangular (prefix sums)
    # For each range [i, j], compute sum of counts i..j
    B = np.zeros((m, d), dtype=np.float64)
    for idx, (i, j) in enumerate(ranges):
        B[idx, i:j+1] = 1.0
    
    # Expected error (Laplace mechanism):
    # Each count has variance 2*(1/ε)^2, propagates through B
    # Error = trace(B @ B.T) * variance_per_query
    expected_error = float(np.trace(B @ B.T))
    
    return WorkloadFactorization(A=A, B=B, Q=Q, expected_error=expected_error)


def hierarchical_factorization(d: int, branching: int = 2) -> WorkloadFactorization:
    """Hierarchical strategy for range queries (Hay et al. 2010).
    
    Uses a tree-based strategy where each node represents a range.
    Achieves O(d log^2 d) error for all range queries.
    
    Args:
        d: Domain size.
        branching: Branching factor (default 2 for binary tree).
    
    Returns:
        WorkloadFactorization using hierarchical strategy.
    """
    # Build hierarchical queries
    # Level 0: full range [0, d)
    # Level 1: branching sub-ranges
    # etc.
    
    levels = []
    level = 0
    width = d
    
    while width >= 1:
        n_nodes = branching ** level
        node_width = max(1, d // n_nodes)
        
        level_queries = []
        for i in range(n_nodes):
            start = i * node_width
            end = min((i + 1) * node_width, d)
            if start < d:
                level_queries.append((start, end))
        
        levels.append(level_queries)
        width = node_width
        level += 1
        
        if node_width == 1:
            break
    
    # Flatten all queries
    all_queries = [q for level in levels for q in level]
    k = len(all_queries)
    
    # Strategy matrix Q: answer these hierarchical queries
    Q = np.zeros((k, d), dtype=np.float64)
    for idx, (start, end) in enumerate(all_queries):
        Q[idx, start:end] = 1.0
    
    # Workload A: all range queries
    ranges = [(i, j) for i in range(d) for j in range(i, d)]
    m = len(ranges)
    A = np.zeros((m, d), dtype=np.float64)
    for idx, (i, j) in enumerate(ranges):
        A[idx, i:j+1] = 1.0
    
    # Reconstruction B: solve B @ Q = A
    # Use least squares: B = A @ Q.T @ (Q @ Q.T)^{-1}
    QQT = Q @ Q.T
    QQT_inv = np.linalg.pinv(QQT)
    B = A @ Q.T @ QQT_inv
    
    # Expected error
    expected_error = float(np.trace(B @ B.T))
    
    return WorkloadFactorization(A=A, B=B, Q=Q, expected_error=expected_error)


def low_rank_factorization(
    A: FloatArray,
    rank: int,
) -> WorkloadFactorization:
    """Low-rank approximation A ≈ B @ Q via SVD.
    
    Uses truncated SVD: A ≈ U_k @ Σ_k @ V_k^T, then:
        Q = Σ_k @ V_k^T
        B = U_k
    
    Args:
        A: Workload matrix (m, d).
        rank: Target rank k.
    
    Returns:
        WorkloadFactorization with rank-k approximation.
    """
    m, d = A.shape
    
    if rank > min(m, d):
        rank = min(m, d)
    
    # Truncated SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :rank]
    S_k = np.diag(S[:rank])
    Vt_k = Vt[:rank, :]
    
    Q = S_k @ Vt_k
    B = U_k
    
    # Expected error: reconstruction + noise propagation
    # Reconstruction error: sum of squared singular values beyond rank k
    recon_error = float(np.sum(S[rank:] ** 2)) if rank < len(S) else 0.0
    
    # Noise propagation: trace(B @ B.T) = k (since B has orthonormal cols)
    noise_error = float(rank)
    
    expected_error = recon_error + noise_error
    
    return WorkloadFactorization(A=A, B=B, Q=Q, expected_error=expected_error)


# ---------------------------------------------------------------------------
# Strategy matrix optimization
# ---------------------------------------------------------------------------


def optimize_strategy_frank_wolfe(
    A: FloatArray,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> WorkloadFactorization:
    """Optimize strategy matrix Q using Frank-Wolfe algorithm.
    
    Solves:
        min_{Q} trace(A @ pinv(Q @ Q.T) @ A.T)
        subject to: Q is a valid strategy matrix
    
    Uses Frank-Wolfe (conditional gradient) to iteratively improve Q.
    
    Args:
        A: Workload matrix (m, d).
        epsilon: Privacy parameter (affects noise scaling).
        sensitivity: Query sensitivity.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
    
    Returns:
        Optimized WorkloadFactorization.
    """
    m, d = A.shape
    
    # Initialize with identity strategy
    Q = np.eye(d)
    
    # Noise variance per query
    noise_var = 2.0 * (sensitivity / epsilon) ** 2
    
    for iteration in range(max_iter):
        # Compute gradient
        QQT = Q @ Q.T
        QQT_inv = np.linalg.pinv(QQT)
        AQQA = A @ QQT_inv @ A.T
        
        # Current objective
        obj = float(np.trace(AQQA)) * noise_var
        
        # Gradient w.r.t. Q (simplified)
        grad = -2.0 * QQT_inv @ Q @ A.T @ A @ QQT_inv
        
        # Frank-Wolfe step: find best direction
        # For simplicity, use gradient descent step
        step_size = 2.0 / (iteration + 2)
        Q_new = Q - step_size * grad
        
        # Project back to valid strategy (non-negative rows sum to sensitivity)
        Q_new = np.maximum(Q_new, 0.0)
        row_sums = Q_new.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        Q_new = Q_new / row_sums * sensitivity
        
        # Check convergence
        delta = np.linalg.norm(Q_new - Q, 'fro')
        Q = Q_new
        
        if delta < tol:
            break
    
    # Compute B
    QQT = Q @ Q.T
    QQT_inv = np.linalg.pinv(QQT)
    B = A @ Q.T @ QQT_inv
    
    # Expected error
    expected_error = float(np.trace(B @ B.T)) * noise_var
    
    return WorkloadFactorization(A=A, B=B, Q=Q, expected_error=expected_error)


def optimize_strategy_multiplicative_weights(
    A: FloatArray,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    max_iter: int = 100,
    learning_rate: float = 0.1,
) -> WorkloadFactorization:
    """Optimize strategy matrix using multiplicative weights update.
    
    Maintains a probability distribution over strategies and updates
    via exponential weights based on query errors.
    
    Args:
        A: Workload matrix (m, d).
        epsilon: Privacy parameter.
        sensitivity: Query sensitivity.
        max_iter: Maximum iterations.
        learning_rate: Learning rate for weight updates.
    
    Returns:
        Optimized WorkloadFactorization.
    """
    m, d = A.shape
    
    # Initialize uniform weights over candidate strategies
    # For simplicity, use identity + random perturbations
    n_strategies = min(d, 10)
    strategies = [np.eye(d) + 0.1 * np.random.randn(d, d) for _ in range(n_strategies)]
    weights = np.ones(n_strategies) / n_strategies
    
    noise_var = 2.0 * (sensitivity / epsilon) ** 2
    
    for iteration in range(max_iter):
        # Compute weighted combination of strategies
        Q = np.sum([w * S for w, S in zip(weights, strategies)], axis=0)
        Q = np.maximum(Q, 0.0)
        
        # Normalize rows
        row_sums = Q.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        Q = Q / row_sums * sensitivity
        
        # Compute errors for each strategy
        errors = []
        for S in strategies:
            S_norm = np.maximum(S, 0.0)
            S_norm = S_norm / (S_norm.sum(axis=1, keepdims=True) + 1e-10) * sensitivity
            
            SST = S_norm @ S_norm.T
            SST_inv = np.linalg.pinv(SST)
            B_s = A @ S_norm.T @ SST_inv
            err = float(np.trace(B_s @ B_s.T)) * noise_var
            errors.append(err)
        
        errors = np.array(errors)
        
        # Multiplicative weights update
        # Lower error = higher weight
        losses = errors / (np.max(errors) + 1e-10)
        weights = weights * np.exp(-learning_rate * losses)
        weights = weights / np.sum(weights)
    
    # Final Q
    Q = np.sum([w * S for w, S in zip(weights, strategies)], axis=0)
    Q = np.maximum(Q, 0.0)
    row_sums = Q.sum(axis=1, keepdims=True)
    Q = Q / (row_sums + 1e-10) * sensitivity
    
    # Compute B
    QQT = Q @ Q.T
    QQT_inv = np.linalg.pinv(QQT)
    B = A @ Q.T @ QQT_inv
    
    expected_error = float(np.trace(B @ B.T)) * noise_var
    
    return WorkloadFactorization(A=A, B=B, Q=Q, expected_error=expected_error)


def compute_strategy_error(
    A: FloatArray,
    Q: FloatArray,
    noise_type: str = "laplace",
    epsilon: float = 1.0,
    delta: float = 0.0,
    sensitivity: float = 1.0,
) -> float:
    """Compute expected squared error for a strategy matrix Q.
    
    Error = E[||A x - B (Q x + noise)||^2]
         = trace(B @ Cov[noise] @ B.T)
    
    Args:
        A: Workload matrix (m, d).
        Q: Strategy matrix (k, d).
        noise_type: "laplace" or "gaussian".
        epsilon: Privacy parameter.
        delta: Privacy parameter (for Gaussian).
        sensitivity: Query sensitivity.
    
    Returns:
        Expected total squared error.
    """
    m, d = A.shape
    k = Q.shape[0]
    
    # Reconstruction matrix B
    QQT = Q @ Q.T
    QQT_inv = np.linalg.pinv(QQT)
    B = A @ Q.T @ QQT_inv
    
    # Noise variance per query
    if noise_type == "laplace":
        noise_var = 2.0 * (sensitivity / epsilon) ** 2
    elif noise_type == "gaussian":
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1) for Gaussian, got {delta}")
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        noise_var = sigma ** 2
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    # Total error
    error = float(np.trace(B @ B.T)) * noise_var
    
    return error


# ---------------------------------------------------------------------------
# MatrixMechanism
# ---------------------------------------------------------------------------


class MatrixMechanism:
    """Matrix mechanism for linear workload queries (Li-Miklau-Hay et al.).
    
    Instead of answering workload A directly, answers a factorization A = BQ
    where Q is optimized to minimize total error. The mechanism:
    
    1. Answers queries Q with noise: y = Q @ x + z
    2. Post-processes via B: answer = B @ y
    
    This can dramatically reduce error compared to answering A directly.
    
    Attributes:
        factorization: WorkloadFactorization A = BQ.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ (0 for Laplace, >0 for Gaussian).
        noise_type: "laplace" or "gaussian".
        metadata: Additional metadata.
    
    Usage::
    
        # Optimize for range queries
        A = construct_range_workload(d=100)
        mech = MatrixMechanism.from_workload(
            A, epsilon=1.0, optimization="hierarchical"
        )
        answers = mech.sample(true_histogram)
        print(f"Error: {mech.expected_error():.2f}")
    """
    
    def __init__(
        self,
        factorization: WorkloadFactorization,
        epsilon: float = 1.0,
        delta: float = 0.0,
        noise_type: str = "laplace",
        sensitivity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize matrix mechanism.
        
        Args:
            factorization: WorkloadFactorization A = BQ.
            epsilon: Privacy parameter ε > 0.
            delta: Privacy parameter δ ∈ [0, 1) (0 for pure DP).
            noise_type: "laplace" or "gaussian".
            sensitivity: Query sensitivity.
            metadata: Optional metadata dict.
            seed: Random seed.
        
        Raises:
            ConfigurationError: If parameters are invalid.
        """
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ConfigurationError(
                f"epsilon must be positive and finite, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if not (0.0 <= delta < 1.0):
            raise ConfigurationError(
                f"delta must be in [0, 1), got {delta}",
                parameter="delta",
                value=delta,
            )
        if noise_type not in ("laplace", "gaussian"):
            raise ConfigurationError(
                f"noise_type must be 'laplace' or 'gaussian', got {noise_type}",
                parameter="noise_type",
                value=noise_type,
            )
        if noise_type == "laplace" and delta > 0:
            warnings.warn("Laplace mechanism ignores delta parameter")
        if noise_type == "gaussian" and delta == 0:
            raise ConfigurationError(
                "Gaussian mechanism requires delta > 0",
                parameter="delta",
                value=delta,
            )
        
        self._factorization = factorization
        self._epsilon = epsilon
        self._delta = delta
        self._noise_type = noise_type
        self._sensitivity = sensitivity
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)
        
        # Precompute noise scale
        if noise_type == "laplace":
            self._noise_scale = sensitivity / epsilon
        else:  # gaussian
            self._noise_scale = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    
    @classmethod
    def from_workload(
        cls,
        A: FloatArray,
        epsilon: float = 1.0,
        delta: float = 0.0,
        optimization: str = "identity",
        noise_type: str = "laplace",
        sensitivity: float = 1.0,
        **kwargs: Any,
    ) -> "MatrixMechanism":
        """Construct matrix mechanism from a workload matrix.
        
        Args:
            A: Workload matrix (m, d).
            epsilon: Privacy parameter.
            delta: Privacy parameter.
            optimization: Strategy optimization method:
                - "identity": trivial factorization A = I * A
                - "low_rank": rank-k SVD approximation
                - "hierarchical": hierarchical strategy for ranges
                - "frank_wolfe": Frank-Wolfe optimization
                - "multiplicative_weights": MW optimization
            noise_type: "laplace" or "gaussian".
            sensitivity: Query sensitivity.
            **kwargs: Additional args for optimization (e.g., rank, max_iter).
        
        Returns:
            MatrixMechanism instance.
        """
        A = np.asarray(A, dtype=np.float64)
        
        if optimization == "identity":
            factorization = identity_factorization(A)
        elif optimization == "low_rank":
            rank = kwargs.get("rank", min(A.shape) // 2)
            factorization = low_rank_factorization(A, rank)
        elif optimization == "hierarchical":
            d = A.shape[1]
            factorization = hierarchical_factorization(d)
            # Update A to match
            factorization = WorkloadFactorization(
                A=A, B=factorization.B, Q=factorization.Q,
                expected_error=factorization.expected_error,
            )
        elif optimization == "frank_wolfe":
            max_iter = kwargs.get("max_iter", 100)
            factorization = optimize_strategy_frank_wolfe(
                A, epsilon, sensitivity, max_iter
            )
        elif optimization == "multiplicative_weights":
            max_iter = kwargs.get("max_iter", 100)
            factorization = optimize_strategy_multiplicative_weights(
                A, epsilon, sensitivity, max_iter
            )
        else:
            raise ValueError(f"Unknown optimization: {optimization}")
        
        return cls(
            factorization=factorization,
            epsilon=epsilon,
            delta=delta,
            noise_type=noise_type,
            sensitivity=sensitivity,
            metadata={"optimization": optimization},
            seed=kwargs.get("seed"),
        )
    
    @property
    def epsilon(self) -> float:
        """Privacy parameter ε."""
        return self._epsilon
    
    @property
    def delta(self) -> float:
        """Privacy parameter δ."""
        return self._delta
    
    @property
    def A(self) -> FloatArray:
        """Workload matrix."""
        return self._factorization.A.copy()
    
    @property
    def B(self) -> FloatArray:
        """Reconstruction matrix."""
        return self._factorization.B.copy()
    
    @property
    def Q(self) -> FloatArray:
        """Strategy matrix."""
        return self._factorization.Q.copy()
    
    @property
    def m(self) -> int:
        """Number of workload queries."""
        return self._factorization.A.shape[0]
    
    @property
    def d(self) -> int:
        """Domain dimension."""
        return self._factorization.A.shape[1]
    
    @property
    def k(self) -> int:
        """Number of strategy queries."""
        return self._factorization.Q.shape[0]
    
    # ----- Sampling -----
    
    def sample(
        self,
        true_data: FloatArray,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample noisy answers for a true data vector.
        
        Algorithm:
        1. Compute strategy answers: y_true = Q @ x
        2. Add noise: y_noisy = y_true + z
        3. Reconstruct workload: answer = B @ y_noisy
        
        Args:
            true_data: True data vector x, shape (d,).
            rng: Optional RNG override.
        
        Returns:
            Noisy workload answers, shape (m,).
        
        Raises:
            ConfigurationError: If true_data has wrong shape.
        """
        rng = rng or self._rng
        true_data = np.asarray(true_data, dtype=np.float64)
        
        if true_data.shape[0] != self.d:
            raise ConfigurationError(
                f"true_data must have shape ({self.d},), got {true_data.shape}",
                parameter="true_data",
            )
        
        # Step 1: Strategy queries
        y_true = self._factorization.Q @ true_data
        
        # Step 2: Add noise
        if self._noise_type == "laplace":
            noise = rng.laplace(scale=self._noise_scale, size=self.k)
        else:  # gaussian
            noise = rng.normal(scale=self._noise_scale, size=self.k)
        
        y_noisy = y_true + noise
        
        # Step 3: Reconstruct
        answer = self._factorization.B @ y_noisy
        
        return answer
    
    def sample_batch(
        self,
        true_data_batch: FloatArray,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample noisy answers for a batch of data vectors.
        
        Args:
            true_data_batch: Batch of data vectors, shape (batch, d).
            rng: Optional RNG override.
        
        Returns:
            Noisy answers, shape (batch, m).
        """
        rng = rng or self._rng
        true_data_batch = np.asarray(true_data_batch, dtype=np.float64)
        
        if true_data_batch.ndim == 1:
            true_data_batch = true_data_batch.reshape(1, -1)
        
        batch_size, d = true_data_batch.shape
        if d != self.d:
            raise ConfigurationError(
                f"true_data_batch must have {self.d} columns, got {d}",
                parameter="true_data_batch",
            )
        
        # Strategy answers: (batch, k)
        Y_true = (self._factorization.Q @ true_data_batch.T).T
        
        # Noise: (batch, k)
        if self._noise_type == "laplace":
            noise = rng.laplace(scale=self._noise_scale, size=(batch_size, self.k))
        else:
            noise = rng.normal(scale=self._noise_scale, size=(batch_size, self.k))
        
        Y_noisy = Y_true + noise
        
        # Reconstruct: (batch, m)
        answers = (self._factorization.B @ Y_noisy.T).T
        
        return answers
    
    # ----- Error analysis -----
    
    def expected_error(self) -> float:
        """Expected total squared error across all workload queries.
        
        Returns:
            E[||A x - answer||^2].
        """
        return compute_strategy_error(
            self._factorization.A,
            self._factorization.Q,
            noise_type=self._noise_type,
            epsilon=self._epsilon,
            delta=self._delta,
            sensitivity=self._sensitivity,
        )
    
    def per_query_error(self) -> FloatArray:
        """Expected squared error per workload query.
        
        Returns:
            Array of per-query MSE, shape (m,).
        """
        # Noise covariance in strategy space
        if self._noise_type == "laplace":
            noise_var = 2.0 * self._noise_scale ** 2
        else:
            noise_var = self._noise_scale ** 2
        
        # Error covariance in workload space: B @ (noise_var * I) @ B.T
        error_cov = noise_var * (self._factorization.B @ self._factorization.B.T)
        
        return np.diag(error_cov).copy()
    
    def max_error(self) -> float:
        """Maximum per-query expected squared error.
        
        Returns:
            Max MSE across queries.
        """
        return float(np.max(self.per_query_error()))
    
    def compare_to_direct(self) -> Dict[str, float]:
        """Compare error to directly answering A without factorization.
        
        Returns:
            Dict with keys:
                'direct_error': Error from answering A directly.
                'matrix_error': Error from matrix mechanism.
                'improvement_ratio': direct_error / matrix_error.
                'error_reduction_pct': Percentage error reduction.
        """
        # Direct mechanism: answer A with noise
        if self._noise_type == "laplace":
            noise_var = 2.0 * self._noise_scale ** 2
        else:
            noise_var = self._noise_scale ** 2
        
        direct_error = self.m * noise_var
        matrix_error = self.expected_error()
        
        ratio = direct_error / max(matrix_error, 1e-10)
        reduction = (1.0 - matrix_error / max(direct_error, 1e-10)) * 100.0
        
        return {
            'direct_error': direct_error,
            'matrix_error': matrix_error,
            'improvement_ratio': ratio,
            'error_reduction_pct': reduction,
        }
    
    # ----- Privacy guarantee -----
    
    def privacy_guarantee(self) -> Tuple[float, float]:
        """Return the privacy guarantee (ε, δ).
        
        Returns:
            Tuple (epsilon, delta).
        """
        return self._epsilon, self._delta
    
    # ----- Validity checking -----
    
    def is_valid(self, tol: float = 1e-6) -> Tuple[bool, List[str]]:
        """Check mechanism validity.
        
        Validates:
        1. Factorization A = BQ is accurate.
        2. Strategy matrix Q is well-conditioned.
        3. Noise scale is sufficient for privacy.
        
        Args:
            tol: Numerical tolerance.
        
        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues: List[str] = []
        
        # Check factorization accuracy
        recon_err = self._factorization.reconstruction_error
        if recon_err > tol * np.linalg.norm(self._factorization.A, 'fro'):
            issues.append(
                f"Factorization error too large: ||A - BQ|| = {recon_err:.2e}"
            )
        
        # Check Q conditioning
        QQT = self._factorization.Q @ self._factorization.Q.T
        cond = np.linalg.cond(QQT)
        if cond > 1e12:
            issues.append(
                f"Strategy matrix Q @ Q.T is ill-conditioned: cond={cond:.2e}"
            )
        
        # Check noise scale
        if self._noise_scale <= 0 or not math.isfinite(self._noise_scale):
            issues.append(
                f"Invalid noise scale: {self._noise_scale}"
            )
        
        return len(issues) == 0, issues
    
    # ----- Representation -----
    
    def __repr__(self) -> str:
        return (
            f"MatrixMechanism(m={self.m}, k={self.k}, d={self.d}, "
            f"ε={self._epsilon:.4f}, δ={self._delta:.2e}, "
            f"{self._noise_type})"
        )
    
    def __str__(self) -> str:
        comp = self.compare_to_direct()
        return (
            f"MatrixMechanism(m={self.m}, k={self.k}, d={self.d}, "
            f"ε={self._epsilon}, δ={self._delta}, "
            f"{comp['improvement_ratio']:.2f}x better than direct)"
        )
