"""
Automated strategy selection for DP workloads.

Analyzes workload structure and automatically selects the best measurement
strategy from a library of known strategies, or falls back to HDMM optimization.

Strategy Library:
    - Identity: For point queries and histograms
    - Hierarchical: For range queries  
    - Prefix: For prefix sum queries
    - Uniform: For total query only
    - Kronecker: For separable multi-dimensional workloads
    - HDMM: General-purpose optimization for any workload

Classification features:
    - Sparsity pattern
    - Rank and condition number
    - Range query structure
    - Kronecker factorability
    - Query sensitivity distribution
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import linalg as sp_linalg
from scipy import sparse

from dp_forge.exceptions import ConfigurationError
from dp_forge.workload_optimizer.hdmm import (
    HDMMOptimizer,
    StrategyMatrix,
    hierarchical_strategy,
    identity_strategy,
    prefix_strategy,
    uniform_strategy,
)
from dp_forge.workload_optimizer.kronecker import (
    KroneckerStrategy,
    detect_kronecker_structure,
    optimize_kronecker,
)

logger = logging.getLogger(__name__)


class WorkloadClassification(Enum):
    """Classification of workload types."""
    IDENTITY = auto()
    RANGE = auto()
    PREFIX = auto()
    MARGINAL = auto()
    HIERARCHICAL = auto()
    KRONECKER = auto()
    GENERAL = auto()


@dataclass
class WorkloadFeatures:
    """Extracted features from workload matrix.
    
    Attributes:
        shape: Workload shape (m, d).
        sparsity: Fraction of zero entries.
        rank: Effective rank.
        condition_number: Condition number.
        is_identity: Whether workload is identity matrix.
        is_range_structured: Whether queries are range queries.
        is_prefix: Whether queries are prefix sums.
        kronecker_dims: Kronecker factor dimensions if detected.
        max_sensitivity: Maximum query sensitivity.
    """
    shape: Tuple[int, int]
    sparsity: float
    rank: int
    condition_number: float
    is_identity: bool
    is_range_structured: bool
    is_prefix: bool
    kronecker_dims: Optional[Tuple[int, ...]]
    max_sensitivity: float
    
    def classify(self) -> WorkloadClassification:
        """Classify workload based on features."""
        if self.is_identity:
            return WorkloadClassification.IDENTITY
        elif self.kronecker_dims is not None:
            return WorkloadClassification.KRONECKER
        elif self.is_prefix:
            return WorkloadClassification.PREFIX
        elif self.is_range_structured:
            return WorkloadClassification.RANGE
        elif self.sparsity > 0.5 and self.rank < self.shape[1] / 2:
            return WorkloadClassification.HIERARCHICAL
        else:
            return WorkloadClassification.GENERAL


class StrategySelector:
    """Automated strategy selection based on workload analysis.
    
    This is the **mandatory entry point** for strategy optimization.
    Direct instantiation of HDMMOptimizer for large domains is disallowed.
    
    Args:
        max_domain_size: Maximum domain size for explicit HDMM.
        feature_cache: Optional cache of pre-computed workload features.
    """
    
    def __init__(
        self,
        max_domain_size: int = 10000,
        feature_cache: Optional[Dict[int, WorkloadFeatures]] = None,
    ):
        self.max_domain_size = max_domain_size
        self.feature_cache = feature_cache or {}
        
        self.strategy_library = self._build_strategy_library()
    
    def select_strategy(
        self,
        workload: npt.NDArray[np.float64],
        epsilon: float = 1.0,
        prefer_simple: bool = True,
    ) -> StrategyMatrix:
        """Select and return optimal strategy for workload.
        
        Args:
            workload: Query workload matrix (m×d).
            epsilon: Privacy parameter.
            prefer_simple: Whether to prefer simple strategies over optimization.
        
        Returns:
            Selected strategy matrix.
        """
        m, d = workload.shape
        
        logger.info(f"Selecting strategy for workload ({m}×{d}), ε={epsilon}")
        
        # Extract features
        features = self._extract_features(workload)
        
        # Classify workload
        classification = features.classify()
        
        logger.info(f"Workload classified as: {classification.name}")
        
        # Select strategy based on classification
        if classification == WorkloadClassification.IDENTITY:
            strategy = identity_strategy(d, epsilon)
        
        elif classification == WorkloadClassification.KRONECKER:
            # Use Kronecker optimization
            dims = features.kronecker_dims
            logger.info(f"Using Kronecker strategy with dimensions {dims}")
            kronecker_strat = optimize_kronecker(workload, dims, epsilon)
            strategy = kronecker_strat.to_strategy_matrix()
        
        elif classification == WorkloadClassification.PREFIX:
            if prefer_simple:
                strategy = prefix_strategy(d, epsilon)
            else:
                strategy = self._optimize_with_hdmm(workload, epsilon)
        
        elif classification == WorkloadClassification.RANGE:
            if prefer_simple and self._is_power_of_two(d):
                strategy = hierarchical_strategy(d, epsilon=epsilon)
            else:
                strategy = self._optimize_with_hdmm(workload, epsilon)
        
        elif classification == WorkloadClassification.HIERARCHICAL:
            if prefer_simple and self._is_power_of_two(d):
                strategy = hierarchical_strategy(d, epsilon=epsilon)
            else:
                strategy = self._optimize_with_hdmm(workload, epsilon)
        
        else:  # GENERAL
            strategy = self._optimize_with_hdmm(workload, epsilon)
        
        # Evaluate strategy
        error = strategy.total_squared_error(workload, epsilon)
        logger.info(f"Selected strategy has TSE={error:.4e}")
        
        return strategy
    
    def _extract_features(
        self,
        workload: npt.NDArray[np.float64],
    ) -> WorkloadFeatures:
        """Extract features from workload matrix."""
        m, d = workload.shape
        
        # Check cache
        workload_hash = hash(workload.tobytes())
        if workload_hash in self.feature_cache:
            return self.feature_cache[workload_hash]
        
        # Compute sparsity
        sparsity = np.sum(workload == 0) / workload.size
        
        # Compute rank
        if d <= 500:
            singular_values = sp_linalg.svdvals(workload)
            rank = np.sum(singular_values > 1e-10 * singular_values[0])
            condition_number = singular_values[0] / (singular_values[-1] + 1e-10)
        else:
            # Approximate for large matrices
            rank = min(m, d)
            condition_number = 1.0
        
        # Check if identity
        is_identity = (m == d and 
                      np.allclose(workload, np.eye(d)))
        
        # Check if range-structured
        is_range = self._detect_range_structure(workload)
        
        # Check if prefix
        is_prefix = self._detect_prefix_structure(workload)
        
        # Check Kronecker structure
        kronecker_dims = None
        if d > 20 and not is_identity:
            kronecker_dims = detect_kronecker_structure(workload, max_dimensions=5)
        
        # Compute sensitivity
        max_sensitivity = np.max(np.linalg.norm(workload, axis=1))
        
        features = WorkloadFeatures(
            shape=(m, d),
            sparsity=sparsity,
            rank=rank,
            condition_number=condition_number,
            is_identity=is_identity,
            is_range_structured=is_range,
            is_prefix=is_prefix,
            kronecker_dims=kronecker_dims,
            max_sensitivity=max_sensitivity,
        )
        
        # Cache
        self.feature_cache[workload_hash] = features
        
        return features
    
    def _detect_range_structure(
        self,
        workload: npt.NDArray[np.float64],
    ) -> bool:
        """Detect if workload consists of range queries."""
        m, d = workload.shape
        
        # Range queries have consecutive 1s in each row
        for i in range(m):
            row = workload[i, :]
            nonzero = np.where(row != 0)[0]
            
            if len(nonzero) == 0:
                continue
            
            # Check if nonzero indices are consecutive
            if not np.all(np.diff(nonzero) == 1):
                return False
            
            # Check if values are all 1
            if not np.allclose(row[nonzero], 1.0):
                return False
        
        return True
    
    def _detect_prefix_structure(
        self,
        workload: npt.NDArray[np.float64],
    ) -> bool:
        """Detect if workload is prefix sum queries."""
        m, d = workload.shape
        
        if m != d:
            return False
        
        # Prefix sums: lower triangular matrix of ones
        expected = np.tril(np.ones((d, d)))
        return np.allclose(workload, expected)
    
    def _optimize_with_hdmm(
        self,
        workload: npt.NDArray[np.float64],
        epsilon: float,
    ) -> StrategyMatrix:
        """Optimize using HDMM with domain size check."""
        m, d = workload.shape
        
        if d > self.max_domain_size:
            raise ConfigurationError(
                f"Domain size {d} exceeds limit {self.max_domain_size}. "
                f"Workload must be Kronecker-factorizable for large domains.",
                parameter="domain_size",
                value=d,
                constraint=f"d <= {self.max_domain_size}",
            )
        
        optimizer = HDMMOptimizer()
        return optimizer.optimize(workload, epsilon)
    
    def _is_power_of_two(self, n: int) -> bool:
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0
    
    def _build_strategy_library(self) -> Dict[str, Any]:
        """Build library of known strategies."""
        return {
            "identity": identity_strategy,
            "uniform": uniform_strategy,
            "hierarchical": hierarchical_strategy,
            "prefix": prefix_strategy,
        }
    
    def predict_error(
        self,
        workload: npt.NDArray[np.float64],
        strategy_name: str,
        epsilon: float,
    ) -> float:
        """Predict error for a named strategy without optimization.
        
        Args:
            workload: Query workload.
            strategy_name: Name of strategy from library.
            epsilon: Privacy parameter.
        
        Returns:
            Predicted total squared error.
        """
        m, d = workload.shape
        
        if strategy_name not in self.strategy_library:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Get strategy constructor
        strategy_fn = self.strategy_library[strategy_name]
        
        # Construct strategy
        try:
            strategy = strategy_fn(d, epsilon)
        except Exception as e:
            logger.warning(f"Failed to construct {strategy_name}: {e}")
            return float('inf')
        
        # Compute error
        try:
            error = strategy.total_squared_error(workload, epsilon)
        except Exception as e:
            logger.warning(f"Failed to compute error for {strategy_name}: {e}")
            return float('inf')
        
        return error
    
    def compare_strategies(
        self,
        workload: npt.NDArray[np.float64],
        epsilon: float,
    ) -> Dict[str, float]:
        """Compare all applicable strategies for the workload.
        
        Args:
            workload: Query workload.
            epsilon: Privacy parameter.
        
        Returns:
            Dictionary mapping strategy names to predicted errors.
        """
        results = {}
        
        for strategy_name in self.strategy_library:
            error = self.predict_error(workload, strategy_name, epsilon)
            results[strategy_name] = error
        
        # Sort by error
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
        
        return sorted_results


def adaptive_strategy_selection(
    workload: npt.NDArray[np.float64],
    epsilon: float,
    validation_fraction: float = 0.2,
) -> StrategyMatrix:
    """Adaptive strategy selection with validation.
    
    Splits workload into training and validation sets, optimizes on training,
    and validates on held-out queries.
    
    Args:
        workload: Query workload matrix.
        epsilon: Privacy parameter.
        validation_fraction: Fraction of queries for validation.
    
    Returns:
        Best strategy according to validation error.
    """
    m, d = workload.shape
    
    # Split workload
    n_val = int(m * validation_fraction)
    n_train = m - n_val
    
    indices = np.random.permutation(m)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_workload = workload[train_indices, :]
    val_workload = workload[val_indices, :]
    
    # Select strategy on training set
    selector = StrategySelector()
    strategy = selector.select_strategy(train_workload, epsilon)
    
    # Evaluate on validation set
    val_error = strategy.total_squared_error(val_workload, epsilon)
    
    logger.info(f"Validation error: {val_error:.4e}")
    
    return strategy


def workload_complexity_score(
    workload: npt.NDArray[np.float64],
) -> float:
    """Compute complexity score for workload.
    
    Higher score indicates more complex workload requiring sophisticated
    strategy optimization.
    
    Args:
        workload: Query workload matrix.
    
    Returns:
        Complexity score (0 = simple, 1 = complex).
    """
    m, d = workload.shape
    
    # Normalize size
    size_score = np.log(m * d) / 20.0
    
    # Rank complexity
    rank = np.linalg.matrix_rank(workload)
    rank_score = rank / d
    
    # Sensitivity distribution
    sensitivities = np.linalg.norm(workload, axis=1)
    sensitivity_variance = np.std(sensitivities) / (np.mean(sensitivities) + 1e-10)
    
    # Combine scores
    complexity = 0.3 * size_score + 0.4 * rank_score + 0.3 * sensitivity_variance
    
    return min(complexity, 1.0)


def recommend_strategy(
    workload: npt.NDArray[np.float64],
    epsilon: float,
    compute_budget: str = "medium",
) -> Tuple[str, StrategyMatrix]:
    """Recommend strategy based on workload and compute budget.
    
    Args:
        workload: Query workload.
        epsilon: Privacy parameter.
        compute_budget: Compute budget ("low", "medium", "high").
    
    Returns:
        Tuple of (recommendation text, strategy matrix).
    """
    selector = StrategySelector()
    features = selector._extract_features(workload)
    classification = features.classify()
    
    m, d = workload.shape
    
    if compute_budget == "low":
        # Prefer simple strategies
        if classification == WorkloadClassification.IDENTITY:
            rec = "Identity strategy (optimal for point queries)"
            strategy = identity_strategy(d, epsilon)
        elif features.is_prefix:
            rec = "Prefix strategy (optimal for prefix sums)"
            strategy = prefix_strategy(d, epsilon)
        elif features.is_range_structured and selector._is_power_of_two(d):
            rec = "Hierarchical strategy (good for range queries)"
            strategy = hierarchical_strategy(d, epsilon=epsilon)
        else:
            rec = "Identity strategy (low-compute fallback)"
            strategy = identity_strategy(d, epsilon)
    
    elif compute_budget == "medium":
        # Use automatic selection
        rec = f"Automatic selection (classified as {classification.name})"
        strategy = selector.select_strategy(workload, epsilon, prefer_simple=True)
    
    else:  # high
        # Always optimize
        if classification == WorkloadClassification.KRONECKER:
            rec = f"Kronecker HDMM (dimensions {features.kronecker_dims})"
            dims = features.kronecker_dims
            kronecker_strat = optimize_kronecker(workload, dims, epsilon)
            strategy = kronecker_strat.to_strategy_matrix()
        elif d <= selector.max_domain_size:
            rec = "Full HDMM optimization"
            strategy = selector._optimize_with_hdmm(workload, epsilon)
        else:
            rec = f"Domain too large ({d}), using automatic selection"
            strategy = selector.select_strategy(workload, epsilon, prefer_simple=False)
    
    return rec, strategy


def estimate_optimization_time(
    workload_shape: Tuple[int, int],
    method: str = "hdmm",
) -> float:
    """Estimate optimization time in seconds.
    
    Args:
        workload_shape: Shape (m, d) of workload.
        method: Optimization method.
    
    Returns:
        Estimated time in seconds.
    """
    m, d = workload_shape
    
    if method == "hdmm":
        # HDMM is roughly O(d³ * iterations)
        # Empirically: ~0.001 seconds per d³ per 100 iterations
        base_time = 0.001 * (d ** 3) / 1e6  # per iteration
        estimated_iterations = min(1000, 100 * np.log(d + 1))
        return base_time * estimated_iterations
    
    elif method == "kronecker":
        # Sum of per-dimension times
        # Assume equal dimensions for rough estimate
        k = int(np.log2(d)) if d > 1 else 1
        dim_size = int(d ** (1.0 / k)) if k > 0 else d
        return k * estimate_optimization_time((m, dim_size), "hdmm")
    
    elif method == "simple":
        # Simple strategies are instant
        return 0.001
    
    else:
        return 1.0


def select_strategy_with_timeout(
    workload: npt.NDArray[np.float64],
    epsilon: float,
    timeout_seconds: float,
) -> StrategyMatrix:
    """Select strategy with time budget constraint.
    
    Args:
        workload: Query workload.
        epsilon: Privacy parameter.
        timeout_seconds: Maximum time allowed.
    
    Returns:
        Best strategy achievable within timeout.
    """
    m, d = workload.shape
    
    # Estimate time for full optimization
    hdmm_time = estimate_optimization_time((m, d), "hdmm")
    
    if hdmm_time <= timeout_seconds:
        # We have time for full optimization
        selector = StrategySelector()
        return selector.select_strategy(workload, epsilon, prefer_simple=False)
    else:
        # Use simple strategy
        logger.warning(f"HDMM would take ~{hdmm_time:.1f}s, using simple strategy")
        selector = StrategySelector()
        return selector.select_strategy(workload, epsilon, prefer_simple=True)
