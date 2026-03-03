"""
HDMM (High-Dimensional Matrix Mechanism) strategy optimization.

Implements the multiplicative weights algorithm from McKenna et al. (2018)
and extensions from McKenna et al. (2021) for optimizing measurement
strategies for differential privacy workloads.

Reference:
    McKenna, Ryan, et al. "Optimizing error of high-dimensional statistical
    queries under differential privacy." VLDB 2018.
    
    McKenna, Ryan, et al. "HDMM: Optimizing error of high-dimensional
    statistical queries under differential privacy." VLDB 2021.

Core Algorithm:
    Given workload W ∈ R^{m×d}, find strategy A ∈ R^{d×d} minimizing:
        TSE(W, A) = trace(W (AᵀA)⁻¹ Wᵀ)
    
    Multiplicative weights maintains non-negative weights on measurements
    and updates them proportional to their contribution to error reduction.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import linalg as sp_linalg
from scipy import sparse
from scipy.sparse import csr_matrix, linalg as sparse_linalg
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from dp_forge.exceptions import ConfigurationError, NumericalInstabilityError
from dp_forge.types import QueryType, WorkloadSpec

logger = logging.getLogger(__name__)


class DomainTooLargeError(Exception):
    """Raised when domain size exceeds computational limits."""
    pass


@dataclass
class StrategyMatrix:
    """Representation of a measurement strategy for DP workloads.
    
    A strategy defines which measurements to make and how to combine them.
    Can be represented explicitly (dense/sparse matrix) or implicitly
    (linear operator for large domains).
    
    Attributes:
        matrix: Explicit strategy matrix (d×d) if available.
        operator: Implicit strategy as LinearOperator if matrix too large.
        domain_size: Size of the data domain (d).
        epsilon: Privacy parameter for error computation.
        metadata: Additional information about the strategy.
    """
    
    matrix: Optional[npt.NDArray[np.float64]] = None
    operator: Optional[LinearOperator] = None
    domain_size: int = 0
    epsilon: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if self.matrix is None and self.operator is None:
            raise ValueError("Either matrix or operator must be provided")
        
        if self.matrix is not None:
            if self.matrix.shape[0] != self.matrix.shape[1]:
                raise ValueError("Strategy matrix must be square")
            self.domain_size = self.matrix.shape[0]
        elif self.operator is not None:
            if self.operator.shape[0] != self.operator.shape[1]:
                raise ValueError("Strategy operator must be square")
            self.domain_size = self.operator.shape[0]
        
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
    
    def total_squared_error(
        self, 
        workload: npt.NDArray[np.float64],
        epsilon: Optional[float] = None
    ) -> float:
        """Compute total squared error for given workload.
        
        The total squared error for workload W and strategy A is:
            TSE = (2/ε²) · trace(W (AᵀA)⁻¹ Wᵀ)
        
        Args:
            workload: Query workload matrix W (m×d).
            epsilon: Privacy parameter (uses self.epsilon if None).
        
        Returns:
            Total squared error.
        """
        eps = epsilon if epsilon is not None else self.epsilon
        
        if self.matrix is not None:
            # Explicit computation
            A = self.matrix
            # Compute AᵀA
            ATA = A.T @ A
            
            # Add regularization for stability
            reg = 1e-10 * np.trace(ATA) / ATA.shape[0]
            ATA_reg = ATA + reg * np.eye(ATA.shape[0])
            
            try:
                # Solve (AᵀA)⁻¹ Wᵀ = X => AᵀA X = Wᵀ
                X = sp_linalg.solve(ATA_reg, workload.T, assume_a='pos')
                # trace(W X) = trace(W (AᵀA)⁻¹ Wᵀ)
                tse = np.trace(workload @ X)
            except np.linalg.LinAlgError as e:
                logger.warning(f"Stability issue in TSE computation: {e}")
                # Fallback to pseudoinverse
                ATA_inv = np.linalg.pinv(ATA)
                tse = np.trace(workload @ ATA_inv @ workload.T)
        else:
            # Implicit computation via iterative solver
            raise NotImplementedError(
                "TSE computation for implicit operators not yet implemented"
            )
        
        # Scale by privacy parameter
        return (2.0 / (eps * eps)) * tse
    
    def apply(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply strategy to data vector: y = Ax."""
        if self.matrix is not None:
            return self.matrix @ x
        else:
            return self.operator @ x
    
    def to_explicit(self) -> npt.NDArray[np.float64]:
        """Convert to explicit matrix representation."""
        if self.matrix is not None:
            return self.matrix
        else:
            # Materialize operator by applying to basis vectors
            d = self.domain_size
            A = np.zeros((d, d))
            for i in range(d):
                ei = np.zeros(d)
                ei[i] = 1.0
                A[:, i] = self.operator @ ei
            return A


class HDMMOptimizer:
    """HDMM strategy optimizer using multiplicative weights.
    
    Finds optimal measurement strategy for a given workload by iteratively
    updating measurement weights based on their error contribution.
    
    Args:
        max_iterations: Maximum MW iterations.
        tolerance: Convergence tolerance on relative error change.
        learning_rate: MW learning rate (default adaptive).
        domain_size_limit: Maximum domain size for explicit representation.
        timeout_seconds: Maximum optimization time.
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        learning_rate: Optional[float] = None,
        domain_size_limit: int = 10000,
        timeout_seconds: float = 300.0,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.domain_size_limit = domain_size_limit
        self.timeout_seconds = timeout_seconds
        
        self.optimization_history: List[Dict[str, float]] = []
    
    def optimize(
        self,
        workload: npt.NDArray[np.float64],
        epsilon: float = 1.0,
        initial_strategy: Optional[StrategyMatrix] = None,
    ) -> StrategyMatrix:
        """Optimize measurement strategy for given workload.
        
        Args:
            workload: Query workload matrix W (m×d).
            epsilon: Privacy parameter.
            initial_strategy: Optional initial strategy (default: identity).
        
        Returns:
            Optimized strategy matrix.
        
        Raises:
            DomainTooLargeError: If domain size exceeds limit.
        """
        m, d = workload.shape
        
        if d > self.domain_size_limit:
            raise DomainTooLargeError(
                f"Domain size {d} exceeds limit {self.domain_size_limit}. "
                f"Use KroneckerStrategy or reduce domain."
            )
        
        logger.info(f"Optimizing HDMM strategy for workload ({m}×{d}), ε={epsilon}")
        
        # Initialize strategy
        if initial_strategy is not None:
            A = initial_strategy.to_explicit()
        else:
            # Start with identity
            A = np.eye(d, dtype=np.float64)
        
        # Run multiplicative weights
        A_opt = self._multiplicative_weights(workload, A, epsilon)
        
        return StrategyMatrix(
            matrix=A_opt,
            epsilon=epsilon,
            metadata={
                "algorithm": "hdmm_mw",
                "iterations": len(self.optimization_history),
                "final_error": self.optimization_history[-1]["error"] if self.optimization_history else None,
            }
        )
    
    def _multiplicative_weights(
        self,
        W: npt.NDArray[np.float64],
        A_init: npt.NDArray[np.float64],
        epsilon: float,
    ) -> npt.NDArray[np.float64]:
        """Core multiplicative weights algorithm.
        
        Maintains measurement weights and updates based on error gradients.
        """
        m, d = W.shape
        start_time = time.time()
        
        # Initialize weights uniformly
        weights = np.ones(d, dtype=np.float64)
        
        # Learning rate (adaptive if not specified)
        if self.learning_rate is None:
            lr = 0.5 / np.sqrt(self.max_iterations)
        else:
            lr = self.learning_rate
        
        best_A = A_init.copy()
        best_error = float('inf')
        
        self.optimization_history = []
        
        for iteration in range(self.max_iterations):
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                logger.warning(f"HDMM optimization timed out at iteration {iteration}")
                break
            
            # Build strategy from weights
            A = np.diag(np.sqrt(weights))
            
            # Compute error
            try:
                error = self._compute_error(W, A, epsilon)
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.warning(f"Numerical instability at iteration {iteration}: {e}")
                break
            
            self.optimization_history.append({
                "iteration": iteration,
                "error": error,
                "weight_entropy": self._entropy(weights),
            })
            
            # Update best
            if error < best_error:
                best_error = error
                best_A = A.copy()
            
            # Check convergence
            if iteration > 0:
                prev_error = self.optimization_history[-2]["error"]
                rel_change = abs(error - prev_error) / (prev_error + 1e-10)
                if rel_change < self.tolerance:
                    logger.info(f"HDMM converged at iteration {iteration}")
                    break
            
            # Compute error gradient w.r.t. weights
            grad = self._compute_gradient(W, A, weights, epsilon)
            
            # Multiplicative update with clipping for stability
            update = -lr * grad
            update = np.clip(update, -10.0, 10.0)  # Prevent overflow
            weights *= np.exp(update)
            
            # Normalize
            weight_sum = np.sum(weights)
            if weight_sum > 1e-10:
                weights /= weight_sum
            else:
                # Reset to uniform if weights collapsed
                weights = np.ones(d, dtype=np.float64) / d
            
            # Prevent underflow
            weights = np.maximum(weights, 1e-10)
        
        logger.info(f"HDMM optimization complete: {len(self.optimization_history)} iterations, "
                   f"final error={best_error:.4e}")
        
        return best_A
    
    def _compute_error(
        self,
        W: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        epsilon: float,
    ) -> float:
        """Compute total squared error."""
        ATA = A.T @ A
        reg = 1e-10 * np.trace(ATA) / ATA.shape[0]
        ATA_reg = ATA + reg * np.eye(ATA.shape[0])
        
        # Solve (AᵀA)⁻¹ Wᵀ
        X = sp_linalg.solve(ATA_reg, W.T, assume_a='pos')
        tse = np.trace(W @ X)
        
        return (2.0 / (epsilon * epsilon)) * tse
    
    def _compute_gradient(
        self,
        W: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        epsilon: float,
    ) -> npt.NDArray[np.float64]:
        """Compute error gradient w.r.t. measurement weights."""
        d = A.shape[0]
        ATA = A.T @ A
        reg = 1e-10 * np.trace(ATA) / d
        ATA_reg = ATA + reg * np.eye(d)
        
        # Compute (AᵀA)⁻¹
        try:
            ATA_inv = sp_linalg.inv(ATA_reg)
        except np.linalg.LinAlgError:
            ATA_inv = np.linalg.pinv(ATA_reg)
        
        # Gradient: ∂TSE/∂wᵢ ∝ [(AᵀA)⁻¹ Wᵀ W (AᵀA)⁻¹]ᵢᵢ
        WTW = W.T @ W
        G = ATA_inv @ WTW @ ATA_inv
        
        # Extract diagonal (sensitivity to each weight)
        grad = -np.diag(G) / (weights + 1e-10)
        
        return grad
    
    def _entropy(self, weights: npt.NDArray[np.float64]) -> float:
        """Compute entropy of weight distribution."""
        p = weights / (np.sum(weights) + 1e-10)
        return -np.sum(p * np.log(p + 1e-10))


def optimize_strategy(
    workload: npt.NDArray[np.float64],
    epsilon: float = 1.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-4,
) -> StrategyMatrix:
    """Convenience function for HDMM optimization.
    
    Args:
        workload: Query workload matrix W (m×d).
        epsilon: Privacy parameter.
        max_iterations: Maximum MW iterations.
        tolerance: Convergence tolerance.
    
    Returns:
        Optimized strategy matrix.
    
    Examples:
        >>> from dp_forge.workloads import WorkloadGenerator
        >>> W = WorkloadGenerator.prefix_sums(50)
        >>> strategy = optimize_strategy(W, epsilon=1.0)
        >>> error = strategy.total_squared_error(W)
    """
    optimizer = HDMMOptimizer(
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return optimizer.optimize(workload, epsilon)


def multiplicative_weights_update(
    workload: npt.NDArray[np.float64],
    strategy: StrategyMatrix,
    learning_rate: float,
    iterations: int,
) -> StrategyMatrix:
    """Perform MW updates on existing strategy.
    
    Args:
        workload: Query workload matrix.
        strategy: Current strategy.
        learning_rate: MW step size.
        iterations: Number of updates.
    
    Returns:
        Updated strategy.
    """
    optimizer = HDMMOptimizer(
        max_iterations=iterations,
        learning_rate=learning_rate,
    )
    return optimizer.optimize(workload, strategy.epsilon, strategy)


def frank_wolfe_strategy(
    workload: npt.NDArray[np.float64],
    constraints: Optional[Dict[str, Any]] = None,
) -> StrategyMatrix:
    """Optimize strategy using Frank-Wolfe algorithm.
    
    Frank-Wolfe is useful when strategy must satisfy additional constraints
    (e.g., sparsity, low-rank, measurement budget).
    
    Args:
        workload: Query workload matrix.
        constraints: Optional constraints on strategy.
    
    Returns:
        Optimized strategy satisfying constraints.
    """
    m, d = workload.shape
    
    # Default: no constraints (equivalent to HDMM)
    if constraints is None:
        return optimize_strategy(workload)
    
    # Initialize
    A = np.eye(d, dtype=np.float64)
    epsilon = constraints.get("epsilon", 1.0)
    max_iters = constraints.get("max_iterations", 100)
    
    for iteration in range(max_iters):
        # Compute gradient
        ATA = A.T @ A
        reg = 1e-10 * np.trace(ATA) / d
        ATA_reg = ATA + reg * np.eye(d)
        
        try:
            ATA_inv = sp_linalg.inv(ATA_reg)
        except np.linalg.LinAlgError:
            ATA_inv = np.linalg.pinv(ATA_reg)
        
        WTW = workload.T @ workload
        grad = -ATA_inv @ WTW @ ATA_inv
        
        # Linear minimization oracle (LMO)
        # Find direction S that minimizes <grad, S>
        # Subject to constraints
        if "sparsity" in constraints:
            # Sparse strategy: select top-k measurements
            k = constraints["sparsity"]
            eigenvalues, eigenvectors = sp_linalg.eigh(-grad)
            S = eigenvectors[:, :k] @ eigenvectors[:, :k].T
        elif "rank" in constraints:
            # Low-rank strategy
            r = constraints["rank"]
            eigenvalues, eigenvectors = sp_linalg.eigh(-grad)
            S = eigenvectors[:, :r] @ eigenvectors[:, :r].T
        else:
            # Unconstrained: take negative gradient direction
            eigenvalues, eigenvectors = sp_linalg.eigh(-grad)
            S = eigenvectors[:, :1] @ eigenvectors[:, :1].T
        
        # Line search
        step_size = 2.0 / (iteration + 2.0)
        
        # Update
        A = (1 - step_size) * A + step_size * S
    
    return StrategyMatrix(matrix=A, epsilon=epsilon)


def _compute_total_squared_error(
    workload: npt.NDArray[np.float64],
    strategy: npt.NDArray[np.float64],
    epsilon: float,
) -> float:
    """Compute total squared error for workload and strategy.
    
    TSE = (2/ε²) · trace(W (AᵀA)⁻¹ Wᵀ)
    """
    ATA = strategy.T @ strategy
    d = ATA.shape[0]
    reg = 1e-10 * np.trace(ATA) / d
    ATA_reg = ATA + reg * np.eye(d)
    
    try:
        X = sp_linalg.solve(ATA_reg, workload.T, assume_a='pos')
        tse = np.trace(workload @ X)
    except np.linalg.LinAlgError:
        ATA_inv = np.linalg.pinv(ATA_reg)
        tse = np.trace(workload @ ATA_inv @ workload.T)
    
    return (2.0 / (epsilon * epsilon)) * tse


def compute_workload_sensitivity(
    workload: npt.NDArray[np.float64],
    norm: str = "l2",
) -> float:
    """Compute sensitivity of workload queries.
    
    Args:
        workload: Query matrix W (m×d).
        norm: Norm to use ("l1", "l2", or "linf").
    
    Returns:
        Maximum sensitivity across all queries.
    """
    if norm == "l1":
        return np.max(np.sum(np.abs(workload), axis=1))
    elif norm == "l2":
        return np.max(np.linalg.norm(workload, axis=1))
    elif norm == "linf":
        return np.max(np.max(np.abs(workload), axis=1))
    else:
        raise ValueError(f"Unknown norm: {norm}")


def identity_strategy(domain_size: int, epsilon: float = 1.0) -> StrategyMatrix:
    """Create identity strategy (measure each element directly).
    
    Args:
        domain_size: Size of data domain.
        epsilon: Privacy parameter.
    
    Returns:
        Identity strategy matrix.
    """
    return StrategyMatrix(
        matrix=np.eye(domain_size, dtype=np.float64),
        epsilon=epsilon,
        metadata={"type": "identity"},
    )


def uniform_strategy(domain_size: int, epsilon: float = 1.0) -> StrategyMatrix:
    """Create uniform strategy (measure total only).
    
    Args:
        domain_size: Size of data domain.
        epsilon: Privacy parameter.
    
    Returns:
        Uniform measurement strategy.
    """
    A = np.ones((domain_size, 1), dtype=np.float64) / np.sqrt(domain_size)
    return StrategyMatrix(
        matrix=A @ A.T,
        epsilon=epsilon,
        metadata={"type": "uniform"},
    )


def hierarchical_strategy(
    domain_size: int,
    levels: Optional[int] = None,
    epsilon: float = 1.0,
) -> StrategyMatrix:
    """Create hierarchical strategy (binary tree of range queries).
    
    Args:
        domain_size: Size of data domain (must be power of 2).
        levels: Number of tree levels (default: log2(domain_size)).
        epsilon: Privacy parameter.
    
    Returns:
        Hierarchical measurement strategy.
    
    Raises:
        ValueError: If domain_size is not a power of 2.
    """
    if not (domain_size & (domain_size - 1)) == 0:
        raise ValueError(f"domain_size must be power of 2, got {domain_size}")
    
    if levels is None:
        levels = int(np.log2(domain_size))
    else:
        levels = int(levels)  # Ensure integer
    
    # Build hierarchical measurement matrix
    n_measurements = domain_size * levels
    A = np.zeros((n_measurements, domain_size), dtype=np.float64)
    
    row = 0
    for level in range(levels):
        interval_size = 2 ** (level + 1)
        n_intervals = domain_size // interval_size
        
        for interval_idx in range(n_intervals):
            start = interval_idx * interval_size
            end = start + interval_size
            A[row, start:end] = 1.0
            row += 1
    
    # Normalize
    A = A / np.sqrt(levels)
    
    return StrategyMatrix(
        matrix=A.T @ A,
        epsilon=epsilon,
        metadata={"type": "hierarchical", "levels": levels},
    )


def prefix_strategy(domain_size: int, epsilon: float = 1.0) -> StrategyMatrix:
    """Create prefix strategy (optimized for prefix sum queries).
    
    Args:
        domain_size: Size of data domain.
        epsilon: Privacy parameter.
    
    Returns:
        Prefix-optimized strategy.
    """
    # For prefix sums, identity is optimal
    # But we can use a factored representation
    A = np.eye(domain_size, dtype=np.float64)
    
    return StrategyMatrix(
        matrix=A,
        epsilon=epsilon,
        metadata={"type": "prefix"},
    )
