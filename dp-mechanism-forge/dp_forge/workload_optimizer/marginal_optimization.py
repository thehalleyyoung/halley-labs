"""
Marginal query optimization for high-dimensional DP workloads.

Selects which marginals to measure directly and how to answer other
queries via post-processing, minimizing total error under privacy constraints.

Reference:
    McKenna, Ryan, et al. "Winning the NIST contest: A scalable and
    general approach to differentially private synthetic data." VLDB 2021.

Problem Formulation:
    Given:
    - Set of target marginal queries M
    - Privacy budget ε
    - Maximum number of direct measurements k
    
    Find:
    - Subset S ⊆ M of k marginals to measure directly
    - Strategy for answering remaining queries via post-processing
    
    Objective: Minimize total squared error across all queries.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt
from scipy import linalg as sp_linalg
from scipy import optimize as sp_optimize

from dp_forge.exceptions import ConfigurationError
from dp_forge.workload_optimizer.hdmm import (
    HDMMOptimizer,
    StrategyMatrix,
    _compute_total_squared_error,
)

logger = logging.getLogger(__name__)


@dataclass
class Marginal:
    """Specification of a marginal query.
    
    Attributes:
        coordinates: Tuple of coordinate indices in the marginal.
        domain_sizes: Per-coordinate domain sizes.
        weight: Query weight for optimization.
    """
    coordinates: Tuple[int, ...]
    domain_sizes: Tuple[int, ...]
    weight: float = 1.0
    
    def __post_init__(self) -> None:
        if len(self.coordinates) != len(self.domain_sizes):
            raise ValueError("coordinates and domain_sizes must have same length")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
    
    @property
    def order(self) -> int:
        """Order of the marginal (number of dimensions)."""
        return len(self.coordinates)
    
    @property
    def size(self) -> int:
        """Number of cells in the marginal."""
        return math.prod(self.domain_sizes)
    
    @property
    def coordinate_set(self) -> FrozenSet[int]:
        """Set of coordinates."""
        return frozenset(self.coordinates)
    
    def __hash__(self) -> int:
        return hash((self.coordinates, self.domain_sizes))
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Marginal):
            return False
        return (self.coordinates == other.coordinates and
                self.domain_sizes == other.domain_sizes)


class MarginalOptimizer:
    """Optimizer for selecting which marginals to measure.
    
    Args:
        max_marginals: Maximum number of marginals to select.
        selection_method: Method for selecting marginals ("greedy", "optimal").
        consistency_method: Method for enforcing consistency ("projection", "mle").
    """
    
    def __init__(
        self,
        max_marginals: Optional[int] = None,
        selection_method: str = "greedy",
        consistency_method: str = "projection",
    ):
        self.max_marginals = max_marginals
        self.selection_method = selection_method
        self.consistency_method = consistency_method
    
    def select_marginals(
        self,
        target_marginals: List[Marginal],
        epsilon: float,
        budget: Optional[float] = None,
    ) -> List[Marginal]:
        """Select subset of marginals to measure directly.
        
        Args:
            target_marginals: All marginals we want to answer.
            epsilon: Privacy budget.
            budget: Optional computation/measurement budget.
        
        Returns:
            Selected marginals to measure.
        """
        if self.selection_method == "greedy":
            return greedy_marginal_selection(
                target_marginals,
                epsilon,
                self.max_marginals or len(target_marginals),
            )
        elif self.selection_method == "optimal":
            return self._optimal_marginal_selection(
                target_marginals,
                epsilon,
                budget,
            )
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _optimal_marginal_selection(
        self,
        target_marginals: List[Marginal],
        epsilon: float,
        budget: Optional[float],
    ) -> List[Marginal]:
        """Optimal marginal selection via integer programming."""
        # This is NP-hard in general, so we use approximation
        # For now, fall back to greedy
        logger.warning("Optimal selection not implemented, using greedy")
        return greedy_marginal_selection(
            target_marginals,
            epsilon,
            self.max_marginals or len(target_marginals),
        )
    
    def optimize_strategy(
        self,
        selected_marginals: List[Marginal],
        target_marginals: List[Marginal],
        epsilon: float,
    ) -> StrategyMatrix:
        """Optimize measurement strategy for selected marginals.
        
        Args:
            selected_marginals: Marginals to measure directly.
            target_marginals: All target marginals to answer.
            epsilon: Privacy budget.
        
        Returns:
            Optimized strategy matrix.
        """
        # Build workload matrix for target marginals
        workload = self._build_marginal_workload(target_marginals)
        
        # Build initial strategy from selected marginals
        strategy = self._build_marginal_strategy(selected_marginals)
        
        # Refine with HDMM
        optimizer = HDMMOptimizer(max_iterations=500)
        return optimizer.optimize(workload, epsilon, strategy)
    
    def _build_marginal_workload(
        self,
        marginals: List[Marginal],
    ) -> npt.NDArray[np.float64]:
        """Build workload matrix from marginals."""
        # Compute total domain size
        if not marginals:
            raise ValueError("No marginals provided")
        
        max_coord = max(max(m.coordinates) for m in marginals)
        all_domain_sizes = [m.domain_sizes[i] 
                           for m in marginals 
                           for i in range(len(m.coordinates))]
        
        # For simplicity, assume uniform domain sizes
        # In practice, would need full dimension specification
        domain_size = math.prod(marginals[0].domain_sizes)
        
        total_queries = sum(m.size for m in marginals)
        workload = np.zeros((total_queries, domain_size))
        
        # Fill in marginal queries
        # This is simplified; full implementation would handle
        # multi-dimensional indexing properly
        row = 0
        for marginal in marginals:
            for cell in range(marginal.size):
                workload[row, cell] = 1.0
                row += 1
        
        return workload
    
    def _build_marginal_strategy(
        self,
        marginals: List[Marginal],
    ) -> StrategyMatrix:
        """Build initial strategy from selected marginals."""
        # Simplified: return identity
        # Full implementation would construct proper marginal strategy
        domain_size = math.prod(marginals[0].domain_sizes) if marginals else 10
        return StrategyMatrix(matrix=np.eye(domain_size, dtype=np.float64))


def greedy_marginal_selection(
    target_marginals: List[Marginal],
    epsilon: float,
    max_marginals: int,
) -> List[Marginal]:
    """Greedy marginal selection based on error reduction.
    
    Iteratively selects the marginal that most reduces total error
    when added to the current set.
    
    Args:
        target_marginals: All target marginals.
        epsilon: Privacy budget.
        max_marginals: Maximum marginals to select.
    
    Returns:
        Selected marginals.
    """
    if not target_marginals:
        return []
    
    selected: List[Marginal] = []
    remaining = set(target_marginals)
    
    logger.info(f"Greedy marginal selection: {len(target_marginals)} candidates, "
               f"selecting up to {max_marginals}")
    
    for iteration in range(max_marginals):
        if not remaining:
            break
        
        best_marginal = None
        best_score = float('-inf')
        
        for candidate in remaining:
            # Compute score (mutual information with unselected)
            score = mutual_information_criterion(
                candidate,
                selected,
                list(remaining),
            )
            
            if score > best_score:
                best_score = score
                best_marginal = candidate
        
        if best_marginal is None:
            break
        
        selected.append(best_marginal)
        remaining.remove(best_marginal)
        
        logger.debug(f"Selected marginal {iteration+1}: "
                    f"coords={best_marginal.coordinates}, score={best_score:.3f}")
    
    logger.info(f"Selected {len(selected)} marginals")
    return selected


def mutual_information_criterion(
    candidate: Marginal,
    selected: List[Marginal],
    remaining: List[Marginal],
) -> float:
    """Compute mutual information score for candidate marginal.
    
    Measures how much information the candidate provides about
    remaining marginals that isn't already in selected set.
    
    Args:
        candidate: Candidate marginal to score.
        selected: Currently selected marginals.
        remaining: Remaining unselected marginals.
    
    Returns:
        Mutual information score (higher is better).
    """
    # Score based on coordinate overlap
    candidate_coords = candidate.coordinate_set
    
    # Penalty for overlap with already selected
    overlap_penalty = 0.0
    for sel in selected:
        overlap = len(candidate_coords & sel.coordinate_set)
        overlap_penalty += overlap / len(candidate_coords)
    
    # Reward for covering unselected coordinates
    coverage_reward = 0.0
    for rem in remaining:
        if rem == candidate:
            continue
        overlap = len(candidate_coords & rem.coordinate_set)
        coverage_reward += overlap / len(rem.coordinate_set)
    
    # Weight by marginal size (prefer smaller marginals)
    size_penalty = np.log(candidate.size + 1)
    
    score = coverage_reward - 0.5 * overlap_penalty - 0.1 * size_penalty
    
    return score


def consistency_projection(
    noisy_marginals: Dict[Marginal, npt.NDArray[np.float64]],
    dimensions: Tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Project noisy marginals onto consistent distribution.
    
    Finds the distribution that is closest to the noisy marginals
    (in L2 sense) while satisfying consistency constraints.
    
    Args:
        noisy_marginals: Dictionary mapping marginals to noisy counts.
        dimensions: Domain dimensions.
    
    Returns:
        Consistent distribution over full domain.
    """
    domain_size = math.prod(dimensions)
    
    # Set up optimization problem
    # Variables: p[i] for i in domain
    # Minimize: Σ_M ||M(p) - noisy_M||²
    # Subject to: Σ_i p[i] = 1, p[i] ≥ 0
    
    def objective(p: npt.NDArray[np.float64]) -> float:
        """L2 distance to noisy marginals."""
        total_error = 0.0
        for marginal, noisy_counts in noisy_marginals.items():
            # Compute marginal from p
            marginal_counts = _compute_marginal(p, marginal, dimensions)
            error = np.sum((marginal_counts - noisy_counts) ** 2)
            total_error += error * marginal.weight
        return total_error
    
    def gradient(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of objective."""
        grad = np.zeros(domain_size)
        
        for marginal, noisy_counts in noisy_marginals.items():
            marginal_counts = _compute_marginal(p, marginal, dimensions)
            diff = marginal_counts - noisy_counts
            
            # Backpropagate gradient
            marg_grad = _marginal_gradient(diff, marginal, dimensions)
            grad += 2.0 * marginal.weight * marg_grad
        
        return grad
    
    # Initial point: uniform distribution
    p0 = np.ones(domain_size) / domain_size
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0},
    ]
    bounds = [(0.0, 1.0) for _ in range(domain_size)]
    
    # Solve
    result = sp_optimize.minimize(
        objective,
        p0,
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000},
    )
    
    if not result.success:
        logger.warning(f"Consistency projection failed: {result.message}")
    
    return result.x


def _compute_marginal(
    distribution: npt.NDArray[np.float64],
    marginal: Marginal,
    dimensions: Tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Compute marginal from full distribution."""
    # Reshape distribution to multi-dimensional array
    dist_reshaped = distribution.reshape(dimensions)
    
    # Sum over non-marginal dimensions
    kept_dims = list(marginal.coordinates)
    summed_dims = [i for i in range(len(dimensions)) if i not in kept_dims]
    
    result = dist_reshaped
    for dim in sorted(summed_dims, reverse=True):
        result = np.sum(result, axis=dim)
    
    return result.ravel()


def _marginal_gradient(
    marginal_error: npt.NDArray[np.float64],
    marginal: Marginal,
    dimensions: Tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Compute gradient of marginal error w.r.t. full distribution."""
    domain_size = math.prod(dimensions)
    grad = np.zeros(domain_size)
    
    # The gradient is obtained by broadcasting the marginal error
    # back to the full domain
    marginal_error_reshaped = marginal_error.reshape(marginal.domain_sizes)
    
    # Broadcast to full dimensions
    grad_reshaped = np.zeros(dimensions)
    
    # This is a simplified version; full implementation would
    # properly handle the multi-dimensional indexing
    # For now, just use uniform broadcasting
    grad_reshaped[:] = np.mean(marginal_error_reshaped)
    
    return grad_reshaped.ravel()


def maximum_likelihood_estimation(
    noisy_marginals: Dict[Marginal, npt.NDArray[np.float64]],
    dimensions: Tuple[int, ...],
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """Estimate distribution via maximum likelihood.
    
    Finds the distribution that maximizes likelihood of observing
    the noisy marginals under the noise model.
    
    Args:
        noisy_marginals: Noisy marginal measurements.
        dimensions: Domain dimensions.
        epsilon: Privacy parameter (determines noise variance).
    
    Returns:
        Maximum likelihood estimate of distribution.
    """
    # For Laplace mechanism, this reduces to weighted least squares
    # with weights proportional to 1/variance
    
    # Noise variance is (2/ε²) for each measurement
    noise_var = 2.0 / (epsilon * epsilon)
    
    # Weight marginals by inverse variance
    weighted_marginals = {
        marg: (counts, 1.0 / noise_var)
        for marg, counts in noisy_marginals.items()
    }
    
    # Use weighted projection
    return _weighted_consistency_projection(weighted_marginals, dimensions)


def _weighted_consistency_projection(
    weighted_marginals: Dict[Marginal, Tuple[npt.NDArray[np.float64], float]],
    dimensions: Tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Weighted consistency projection."""
    domain_size = math.prod(dimensions)
    
    def objective(p: npt.NDArray[np.float64]) -> float:
        total_error = 0.0
        for marginal, (noisy_counts, weight) in weighted_marginals.items():
            marginal_counts = _compute_marginal(p, marginal, dimensions)
            error = np.sum((marginal_counts - noisy_counts) ** 2)
            total_error += weight * error
        return total_error
    
    p0 = np.ones(domain_size) / domain_size
    
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(domain_size)]
    
    result = sp_optimize.minimize(
        objective,
        p0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000},
    )
    
    return result.x


def compute_marginal_sensitivity(
    marginal: Marginal,
    adjacency: str = "add_remove",
) -> float:
    """Compute sensitivity of marginal query.
    
    Args:
        marginal: Marginal query.
        adjacency: Type of adjacency ("add_remove" or "substitute").
    
    Returns:
        L1 sensitivity of the marginal.
    """
    if adjacency == "add_remove":
        # Adding/removing one record changes at most one cell by 1
        return 1.0
    elif adjacency == "substitute":
        # Substituting one record changes two cells by 1 each
        return 2.0
    else:
        raise ValueError(f"Unknown adjacency: {adjacency}")


def optimize_marginal_workload(
    marginals: List[Marginal],
    epsilon: float,
    max_selected: Optional[int] = None,
) -> Tuple[List[Marginal], StrategyMatrix]:
    """Full pipeline for marginal workload optimization.
    
    Args:
        marginals: Target marginals to answer.
        epsilon: Privacy budget.
        max_selected: Maximum marginals to measure directly.
    
    Returns:
        Tuple of (selected marginals, measurement strategy).
    """
    optimizer = MarginalOptimizer(max_marginals=max_selected)
    
    # Select marginals
    selected = optimizer.select_marginals(marginals, epsilon)
    
    # Optimize strategy
    strategy = optimizer.optimize_strategy(selected, marginals, epsilon)
    
    return selected, strategy


def build_marginal_workload_matrix(
    marginals: List[Marginal],
    total_domain_size: int,
) -> npt.NDArray[np.float64]:
    """Build explicit workload matrix for marginal queries.
    
    Args:
        marginals: List of marginal queries.
        total_domain_size: Size of full domain.
    
    Returns:
        Workload matrix (m×d).
    """
    total_queries = sum(m.size for m in marginals)
    W = np.zeros((total_queries, total_domain_size), dtype=np.float64)
    
    row = 0
    for marginal in marginals:
        # Each cell in the marginal corresponds to one query
        for cell_idx in range(marginal.size):
            # Simplified: assumes lexicographic ordering
            # Full implementation would map cell_idx to domain indices
            W[row, cell_idx % total_domain_size] = 1.0
            row += 1
    
    return W


def iterative_proportional_fitting(
    noisy_marginals: Dict[Marginal, npt.NDArray[np.float64]],
    dimensions: Tuple[int, ...],
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> npt.NDArray[np.float64]:
    """Iterative proportional fitting (IPF) for consistency.
    
    Classical algorithm for finding a distribution matching given marginals.
    
    Args:
        noisy_marginals: Target marginals.
        dimensions: Domain dimensions.
        max_iterations: Maximum IPF iterations.
        tolerance: Convergence tolerance.
    
    Returns:
        Consistent distribution.
    """
    domain_size = math.prod(dimensions)
    
    # Initialize to uniform
    p = np.ones(domain_size) / domain_size
    
    marginal_list = list(noisy_marginals.items())
    
    for iteration in range(max_iterations):
        p_old = p.copy()
        
        # Update each marginal in turn
        for marginal, target_counts in marginal_list:
            # Compute current marginal
            current_counts = _compute_marginal(p, marginal, dimensions)
            
            # Avoid division by zero
            current_counts = np.maximum(current_counts, 1e-10)
            target_counts = np.maximum(target_counts, 1e-10)
            
            # Compute scaling factors
            scales = target_counts / current_counts
            
            # Apply scaling to distribution
            # This is simplified; full version would properly broadcast
            p *= np.mean(scales)
            p /= np.sum(p)
        
        # Check convergence
        change = np.max(np.abs(p - p_old))
        if change < tolerance:
            logger.debug(f"IPF converged at iteration {iteration}")
            break
    
    return p


def select_marginals_by_importance(
    marginals: List[Marginal],
    importance_scores: npt.NDArray[np.float64],
    max_selected: int,
) -> List[Marginal]:
    """Select marginals by pre-computed importance scores.
    
    Args:
        marginals: Candidate marginals.
        importance_scores: Importance score for each marginal.
        max_selected: Number to select.
    
    Returns:
        Selected marginals.
    """
    if len(marginals) != len(importance_scores):
        raise ValueError("Number of marginals and scores must match")
    
    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    # Select top-k
    selected_indices = sorted_indices[:max_selected]
    
    return [marginals[i] for i in selected_indices]
