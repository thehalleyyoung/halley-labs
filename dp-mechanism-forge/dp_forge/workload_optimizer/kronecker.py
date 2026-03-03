"""
Kronecker product strategies for high-dimensional workloads.

For multi-dimensional data with separable structure, strategies can be
represented as Kronecker products of per-dimension strategies, reducing
the optimization complexity from O(d³) to O(Σ dᵢ³).

Reference:
    Zhang, Dan, et al. "EKTELO: A framework for defining differentially-
    private computations." SIGMOD 2018.

Mathematical Background:
    For d-dimensional data with domain D = D₁ × D₂ × ... × Dₖ,
    if the workload is separable (Kronecker product of per-dimension workloads),
    the optimal strategy is also separable:
    
        A* = A₁ ⊗ A₂ ⊗ ... ⊗ Aₖ
    
    where Aᵢ is optimized for dimension i independently.
    
    This reduces optimization from O((Πdᵢ)³) to O(Σdᵢ³).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import linalg as sp_linalg

from dp_forge.exceptions import ConfigurationError
from dp_forge.workload_optimizer.hdmm import (
    HDMMOptimizer,
    StrategyMatrix,
    _compute_total_squared_error,
)

logger = logging.getLogger(__name__)


@dataclass
class KroneckerStrategy:
    """Strategy represented as Kronecker product of factor strategies.
    
    For multi-dimensional workloads, this representation dramatically
    reduces memory and computation requirements.
    
    Attributes:
        factors: List of per-dimension strategy matrices.
        dimensions: Domain sizes for each dimension.
        epsilon: Privacy parameter.
        metadata: Additional information.
    """
    
    factors: List[npt.NDArray[np.float64]]
    dimensions: Tuple[int, ...]
    epsilon: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if len(self.factors) != len(self.dimensions):
            raise ValueError(
                f"Number of factors ({len(self.factors)}) must match "
                f"number of dimensions ({len(self.dimensions)})"
            )
        
        for i, (factor, dim) in enumerate(zip(self.factors, self.dimensions)):
            if factor.shape != (dim, dim):
                raise ValueError(
                    f"Factor {i} has shape {factor.shape}, expected ({dim}, {dim})"
                )
        
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
    
    @property
    def total_domain_size(self) -> int:
        """Total size of product domain."""
        return math.prod(self.dimensions)
    
    @property
    def n_dimensions(self) -> int:
        """Number of dimensions."""
        return len(self.dimensions)
    
    def to_explicit(self) -> npt.NDArray[np.float64]:
        """Materialize full Kronecker product.
        
        Warning: This can be very large (d² for d = Π dᵢ).
        Only use for small domains.
        
        Returns:
            Full strategy matrix as Kronecker product of factors.
        """
        result = self.factors[0]
        for factor in self.factors[1:]:
            result = np.kron(result, factor)
        return result
    
    def apply(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply Kronecker product strategy efficiently.
        
        Uses repeated matrix-vector products instead of materializing
        the full Kronecker product.
        
        Args:
            x: Input vector (flattened multi-dimensional array).
        
        Returns:
            Result of strategy applied to x.
        """
        # Reshape to multi-dimensional array
        x_reshaped = x.reshape(self.dimensions)
        
        # Apply each factor along its dimension
        result = x_reshaped
        for dim, factor in enumerate(self.factors):
            # Move dimension to front, apply factor, move back
            result = np.moveaxis(result, dim, 0)
            result = factor @ result.reshape(result.shape[0], -1)
            result = result.reshape(self.dimensions[dim], *self.dimensions[:dim], 
                                   *self.dimensions[dim+1:])
            result = np.moveaxis(result, 0, dim)
        
        return result.ravel()
    
    def total_squared_error(
        self,
        workload: npt.NDArray[np.float64],
        epsilon: Optional[float] = None,
    ) -> float:
        """Compute total squared error.
        
        If workload is also separable, this can be computed efficiently.
        Otherwise, falls back to explicit computation.
        """
        eps = epsilon if epsilon is not None else self.epsilon
        
        # Check if workload is Kronecker-factorizable
        workload_factors = kronecker_decompose(workload, self.dimensions)
        
        if workload_factors is not None:
            # Separable case: TSE is product of per-dimension TSEs
            total_error = 0.0
            for w_factor, a_factor in zip(workload_factors, self.factors):
                factor_error = _compute_total_squared_error(
                    w_factor, a_factor, eps
                )
                total_error += factor_error
            return total_error
        else:
            # Non-separable: must materialize
            logger.warning("Workload not separable, materializing full strategy")
            A_full = self.to_explicit()
            return _compute_total_squared_error(workload, A_full, eps)
    
    def to_strategy_matrix(self) -> StrategyMatrix:
        """Convert to general StrategyMatrix."""
        if self.total_domain_size <= 10000:
            return StrategyMatrix(
                matrix=self.to_explicit(),
                epsilon=self.epsilon,
                metadata=self.metadata,
            )
        else:
            # Use implicit representation
            from scipy.sparse.linalg import LinearOperator
            
            def matvec(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return self.apply(x)
            
            d = self.total_domain_size
            op = LinearOperator((d, d), matvec=matvec)
            
            return StrategyMatrix(
                operator=op,
                domain_size=d,
                epsilon=self.epsilon,
                metadata=self.metadata,
            )


def kronecker_decompose(
    matrix: npt.NDArray[np.float64],
    dimensions: Tuple[int, ...],
) -> Optional[List[npt.NDArray[np.float64]]]:
    """Decompose matrix into Kronecker product factors.
    
    Checks if matrix A can be written as A = A₁ ⊗ A₂ ⊗ ... ⊗ Aₖ
    where Aᵢ has shape (dᵢ, dᵢ).
    
    Args:
        matrix: Matrix to decompose (d×d where d = Π dᵢ).
        dimensions: Target factor dimensions (d₁, d₂, ..., dₖ).
    
    Returns:
        List of factor matrices if decomposable, None otherwise.
    """
    d = matrix.shape[0]
    if matrix.shape[1] != d:
        return None
    
    if math.prod(dimensions) != d:
        return None
    
    if len(dimensions) == 1:
        return [matrix]
    
    # Try to extract factors using SVD-based approach
    factors = _extract_kronecker_factors(matrix, dimensions)
    
    if factors is None:
        return None
    
    # Verify decomposition is accurate
    reconstructed = factors[0]
    for factor in factors[1:]:
        reconstructed = np.kron(reconstructed, factor)
    
    error = np.linalg.norm(matrix - reconstructed, 'fro')
    relative_error = error / (np.linalg.norm(matrix, 'fro') + 1e-10)
    
    if relative_error > 1e-6:
        logger.debug(f"Kronecker decomposition error too large: {relative_error}")
        return None
    
    return factors


def _extract_kronecker_factors(
    matrix: npt.NDArray[np.float64],
    dimensions: Tuple[int, ...],
) -> Optional[List[npt.NDArray[np.float64]]]:
    """Extract Kronecker factors using recursive decomposition."""
    if len(dimensions) == 1:
        return [matrix]
    
    d1 = dimensions[0]
    d_rest = math.prod(dimensions[1:])
    
    # Reshape matrix to (d1, d_rest, d1, d_rest)
    try:
        M_reshaped = matrix.reshape(d1, d_rest, d1, d_rest)
    except ValueError:
        return None
    
    # Extract first factor by averaging over second dimensions
    # A₁[i,j] should be constant ratio to M[i*d_rest:(i+1)*d_rest, j*d_rest:(j+1)*d_rest]
    
    # Use the (0,0) block as reference
    A_rest_ref = M_reshaped[0, :, 0, :]
    if np.linalg.norm(A_rest_ref) < 1e-10:
        return None
    
    # Extract A₁ from scaling factors
    A1 = np.zeros((d1, d1))
    for i in range(d1):
        for j in range(d1):
            block = M_reshaped[i, :, j, :]
            # block ≈ A₁[i,j] * A_rest
            # Find scaling factor
            if np.linalg.norm(A_rest_ref) > 1e-10:
                scale = np.sum(block * A_rest_ref) / np.sum(A_rest_ref * A_rest_ref)
                A1[i, j] = scale
    
    # Reconstruct A_rest
    A_rest = M_reshaped[0, :, 0, :] / (A1[0, 0] + 1e-10)
    
    # Recursively decompose A_rest
    rest_factors = _extract_kronecker_factors(A_rest, dimensions[1:])
    
    if rest_factors is None:
        return None
    
    return [A1] + rest_factors


def optimize_kronecker(
    workload: npt.NDArray[np.float64],
    dimensions: Tuple[int, ...],
    epsilon: float = 1.0,
) -> KroneckerStrategy:
    """Optimize Kronecker product strategy for separable workload.
    
    Decomposes workload into per-dimension factors and optimizes each
    independently using HDMM.
    
    Args:
        workload: Query workload matrix.
        dimensions: Domain sizes for each dimension.
        epsilon: Privacy parameter.
    
    Returns:
        Optimized Kronecker strategy.
    
    Raises:
        ValueError: If workload is not separable.
    """
    # Decompose workload
    workload_factors = kronecker_decompose(workload, dimensions)
    
    if workload_factors is None:
        raise ValueError("Workload is not separable into Kronecker factors")
    
    logger.info(f"Optimizing Kronecker strategy with {len(dimensions)} dimensions: {dimensions}")
    
    # Optimize each factor independently
    strategy_factors = []
    for i, (w_factor, dim) in enumerate(zip(workload_factors, dimensions)):
        logger.debug(f"Optimizing factor {i+1}/{len(dimensions)} (dim={dim})")
        
        optimizer = HDMMOptimizer(
            max_iterations=500,
            tolerance=1e-4,
        )
        
        # Use per-dimension epsilon (split equally)
        eps_factor = epsilon / np.sqrt(len(dimensions))
        
        factor_strategy = optimizer.optimize(w_factor, eps_factor)
        strategy_factors.append(factor_strategy.to_explicit())
    
    return KroneckerStrategy(
        factors=strategy_factors,
        dimensions=dimensions,
        epsilon=epsilon,
        metadata={
            "algorithm": "kronecker_hdmm",
            "n_dimensions": len(dimensions),
        },
    )


def detect_kronecker_structure(
    workload: npt.NDArray[np.float64],
    max_dimensions: int = 10,
) -> Optional[Tuple[int, ...]]:
    """Detect if workload has Kronecker structure.
    
    Attempts to find dimensions (d₁, ..., dₖ) such that workload
    can be decomposed as Kronecker product.
    
    Args:
        workload: Workload matrix to analyze.
        max_dimensions: Maximum number of factors to try.
    
    Returns:
        Tuple of dimensions if structure detected, None otherwise.
    """
    m, d = workload.shape
    
    # Try factorizations of d
    for k in range(2, min(max_dimensions + 1, d + 1)):
        # Try to find k factors
        for dims in _factorize_dimension(d, k):
            factors = kronecker_decompose(workload, dims)
            if factors is not None:
                logger.info(f"Detected Kronecker structure: dimensions={dims}")
                return dims
    
    return None


def _factorize_dimension(n: int, k: int) -> List[Tuple[int, ...]]:
    """Generate all ways to factor n into k positive integers."""
    if k == 1:
        return [(n,)]
    
    result = []
    for d1 in range(2, n):
        if n % d1 == 0:
            for rest in _factorize_dimension(n // d1, k - 1):
                result.append((d1,) + rest)
    
    return result


def marginal_to_kronecker_workload(
    marginal_queries: List[Tuple[int, ...]],
    dimensions: Tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Build workload matrix for marginal queries.
    
    Args:
        marginal_queries: List of marginal coordinate tuples.
        dimensions: Domain sizes for each dimension.
    
    Returns:
        Workload matrix for all marginals.
    """
    d = math.prod(dimensions)
    n_queries = sum(math.prod([dimensions[i] for i in marg]) 
                   for marg in marginal_queries)
    
    workload = np.zeros((n_queries, d), dtype=np.float64)
    
    row = 0
    for marginal in marginal_queries:
        # Build identity for marginal dimensions
        marg_dims = tuple(dimensions[i] for i in marginal)
        marg_size = math.prod(marg_dims)
        
        # Enumerate all cells in marginal
        for cell_idx in range(marg_size):
            # Map cell index to full domain indices
            cell_coords = _unravel_index(cell_idx, marg_dims)
            
            # Find all domain elements matching this marginal cell
            for domain_idx in range(d):
                domain_coords = _unravel_index(domain_idx, dimensions)
                
                # Check if domain element matches marginal cell
                match = True
                for marg_dim, coord in zip(marginal, cell_coords):
                    if domain_coords[marg_dim] != coord:
                        match = False
                        break
                
                if match:
                    workload[row, domain_idx] = 1.0
            
            row += 1
    
    return workload


def _unravel_index(idx: int, dimensions: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert flat index to multi-dimensional coordinates."""
    coords = []
    for dim in reversed(dimensions):
        coords.append(idx % dim)
        idx //= dim
    return tuple(reversed(coords))


def efficient_noise_generation(
    strategy: KroneckerStrategy,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """Generate noise vector efficiently for Kronecker strategy.
    
    Uses the factored structure to generate noise without materializing
    the full covariance matrix.
    
    Args:
        strategy: Kronecker product strategy.
        epsilon: Privacy parameter.
    
    Returns:
        Noise vector with correct covariance.
    """
    # For Laplace mechanism, noise covariance is (2/ε²) * (AᵀA)⁻¹
    # For Kronecker A = A₁ ⊗ ... ⊗ Aₖ, we have
    # (AᵀA)⁻¹ = (A₁ᵀA₁)⁻¹ ⊗ ... ⊗ (AₖᵀAₖ)⁻¹
    
    scale = np.sqrt(2.0) / epsilon
    
    # Generate independent Gaussian noise for each factor
    noise_factors = []
    for factor in strategy.factors:
        d_i = factor.shape[0]
        
        # Compute covariance for this factor
        ATA = factor.T @ factor
        reg = 1e-10 * np.trace(ATA) / d_i
        ATA_reg = ATA + reg * np.eye(d_i)
        
        try:
            # Cholesky decomposition for sampling
            L = sp_linalg.cholesky(ATA_reg, lower=True)
            L_inv = sp_linalg.solve_triangular(L, np.eye(d_i), lower=True)
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition
            eigenvalues, eigenvectors = sp_linalg.eigh(ATA_reg)
            L_inv = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        
        # Generate standard Gaussian and transform
        z = np.random.randn(d_i)
        noise_factors.append(scale * L_inv @ z)
    
    # Combine via Kronecker product
    noise = noise_factors[0]
    for factor_noise in noise_factors[1:]:
        noise = np.kron(noise, factor_noise)
    
    return noise


def kronecker_error_analysis(
    workload: npt.NDArray[np.float64],
    strategy: KroneckerStrategy,
    epsilon: float,
) -> Dict[str, Any]:
    """Analyze error decomposition for Kronecker strategy.
    
    Args:
        workload: Query workload.
        strategy: Kronecker strategy.
        epsilon: Privacy parameter.
    
    Returns:
        Dictionary with error analysis results.
    """
    # Attempt to decompose workload
    workload_factors = kronecker_decompose(workload, strategy.dimensions)
    
    if workload_factors is None:
        # Non-separable workload
        total_error = strategy.to_strategy_matrix().total_squared_error(workload, epsilon)
        return {
            "separable": False,
            "total_error": total_error,
            "per_dimension_error": None,
        }
    
    # Compute per-dimension errors
    per_dim_errors = []
    for w_factor, a_factor in zip(workload_factors, strategy.factors):
        error = _compute_total_squared_error(w_factor, a_factor, epsilon)
        per_dim_errors.append(error)
    
    total_error = sum(per_dim_errors)
    
    return {
        "separable": True,
        "total_error": total_error,
        "per_dimension_error": per_dim_errors,
        "dimensions": strategy.dimensions,
    }


def adaptive_kronecker_optimization(
    workload: npt.NDArray[np.float64],
    dimensions: Tuple[int, ...],
    epsilon: float = 1.0,
    budget_allocation: str = "uniform",
) -> KroneckerStrategy:
    """Optimize Kronecker strategy with adaptive budget allocation.
    
    Instead of splitting epsilon equally across dimensions, allocate
    more budget to dimensions with higher query complexity.
    
    Args:
        workload: Query workload.
        dimensions: Domain dimensions.
        epsilon: Total privacy budget.
        budget_allocation: Method for allocating budget ("uniform", "adaptive").
    
    Returns:
        Optimized Kronecker strategy.
    """
    workload_factors = kronecker_decompose(workload, dimensions)
    
    if workload_factors is None:
        raise ValueError("Workload not separable")
    
    # Compute budget allocation
    if budget_allocation == "uniform":
        epsilons = [epsilon / np.sqrt(len(dimensions))] * len(dimensions)
    elif budget_allocation == "adaptive":
        # Allocate based on per-dimension sensitivity
        sensitivities = []
        for w_factor in workload_factors:
            sens = np.max(np.linalg.norm(w_factor, axis=1))
            sensitivities.append(sens)
        
        # Allocate inversely proportional to sensitivity
        weights = [1.0 / (s + 1e-10) for s in sensitivities]
        total_weight = sum(weights)
        epsilons = [epsilon * w / total_weight for w in weights]
    else:
        raise ValueError(f"Unknown budget allocation: {budget_allocation}")
    
    # Optimize each dimension
    strategy_factors = []
    for w_factor, eps_i in zip(workload_factors, epsilons):
        optimizer = HDMMOptimizer(max_iterations=500, tolerance=1e-4)
        factor_strategy = optimizer.optimize(w_factor, eps_i)
        strategy_factors.append(factor_strategy.to_explicit())
    
    return KroneckerStrategy(
        factors=strategy_factors,
        dimensions=dimensions,
        epsilon=epsilon,
        metadata={
            "algorithm": "adaptive_kronecker",
            "budget_allocation": budget_allocation,
        },
    )
