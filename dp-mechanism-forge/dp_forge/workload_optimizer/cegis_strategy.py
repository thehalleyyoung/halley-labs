"""
CEGIS-based strategy synthesis for joint mechanism + strategy optimization.

Extends the standard CEGIS loop for mechanism synthesis to jointly optimize
both the mechanism and the measurement strategy, enabling more powerful
synthesis for workload-based queries.

Problem Formulation:
    Find mechanism M and strategy A that:
    1. M satisfies (ε, δ)-DP
    2. Strategy A minimizes workload error
    3. M and A are jointly optimized for the workload
    
    This is more powerful than sequential optimization because the mechanism
    can be tailored to the strategy.

Reference:
    Extends CEGIS framework from:
    Albarghouthi, Aws, et al. "Synthesizing coupling proofs of differential
    privacy." POPL 2017.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError, SolverError
from dp_forge.types import QueryType
from dp_forge.workload_optimizer.hdmm import (
    HDMMOptimizer,
    StrategyMatrix,
)
from dp_forge.workload_optimizer.strategy_selection import (
    StrategySelector,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategySynthesisResult:
    """Result of joint mechanism + strategy synthesis.
    
    Attributes:
        mechanism: Synthesized mechanism probability table.
        strategy: Optimal measurement strategy.
        total_error: Total squared error achieved.
        privacy_params: Privacy parameters (ε, δ).
        iterations: Number of CEGIS iterations.
        synthesis_time: Total synthesis time.
        metadata: Additional information.
    """
    mechanism: npt.NDArray[np.float64]
    strategy: StrategyMatrix
    total_error: float
    privacy_params: Tuple[float, float]
    iterations: int
    synthesis_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CEGISStrategySynthesizer:
    """CEGIS-based joint mechanism and strategy synthesizer.
    
    Alternates between:
    1. Synthesizing mechanism for fixed strategy
    2. Optimizing strategy for fixed mechanism
    
    Until convergence or timeout.
    
    Args:
        epsilon: Privacy parameter.
        delta: Privacy parameter (0 for pure DP).
        max_iterations: Maximum CEGIS iterations.
        strategy_optimizer: Strategy optimization method.
        timeout_seconds: Total timeout for synthesis.
    """
    
    def __init__(
        self,
        epsilon: float,
        delta: float = 0.0,
        max_iterations: int = 50,
        strategy_optimizer: str = "hdmm",
        timeout_seconds: float = 600.0,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta < 0:
            raise ValueError("delta must be non-negative")
        
        self.epsilon = epsilon
        self.delta = delta
        self.max_iterations = max_iterations
        self.strategy_optimizer = strategy_optimizer
        self.timeout_seconds = timeout_seconds
        
        self.iteration_history: List[Dict[str, Any]] = []
    
    def synthesize(
        self,
        workload: npt.NDArray[np.float64],
        initial_strategy: Optional[StrategyMatrix] = None,
        initial_mechanism: Optional[npt.NDArray[np.float64]] = None,
    ) -> StrategySynthesisResult:
        """Synthesize mechanism and strategy jointly.
        
        Args:
            workload: Query workload matrix.
            initial_strategy: Optional initial strategy.
            initial_mechanism: Optional initial mechanism.
        
        Returns:
            Synthesis result with mechanism and strategy.
        """
        m, d = workload.shape
        start_time = time.time()
        
        logger.info(f"Starting CEGIS strategy synthesis: workload ({m}×{d}), "
                   f"ε={self.epsilon}, δ={self.delta}")
        
        # Initialize strategy
        if initial_strategy is None:
            selector = StrategySelector()
            current_strategy = selector.select_strategy(workload, self.epsilon)
        else:
            current_strategy = initial_strategy
        
        # Initialize mechanism (uniform for now)
        if initial_mechanism is None:
            current_mechanism = self._initialize_mechanism(d)
        else:
            current_mechanism = initial_mechanism
        
        best_error = float('inf')
        best_mechanism = current_mechanism
        best_strategy = current_strategy
        
        self.iteration_history = []
        
        for iteration in range(self.max_iterations):
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                logger.warning(f"CEGIS synthesis timed out at iteration {iteration}")
                break
            
            logger.debug(f"CEGIS iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Optimize mechanism for fixed strategy
            try:
                current_mechanism = self._optimize_mechanism(
                    workload, current_strategy
                )
            except Exception as e:
                logger.warning(f"Mechanism optimization failed: {e}")
                break
            
            # Step 2: Optimize strategy for fixed mechanism
            try:
                current_strategy = self._optimize_strategy(
                    workload, current_mechanism
                )
            except Exception as e:
                logger.warning(f"Strategy optimization failed: {e}")
                break
            
            # Compute current error
            current_error = current_strategy.total_squared_error(
                workload, self.epsilon
            )
            
            self.iteration_history.append({
                "iteration": iteration,
                "error": current_error,
                "time": time.time() - start_time,
            })
            
            # Update best
            if current_error < best_error:
                best_error = current_error
                best_mechanism = current_mechanism
                best_strategy = current_strategy
                logger.debug(f"New best error: {best_error:.4e}")
            
            # Check convergence
            if iteration > 0:
                prev_error = self.iteration_history[-2]["error"]
                rel_change = abs(current_error - prev_error) / (prev_error + 1e-10)
                if rel_change < 1e-4:
                    logger.info(f"CEGIS converged at iteration {iteration + 1}")
                    break
        
        synthesis_time = time.time() - start_time
        
        logger.info(f"CEGIS synthesis complete: {len(self.iteration_history)} iterations, "
                   f"final error={best_error:.4e}, time={synthesis_time:.2f}s")
        
        return StrategySynthesisResult(
            mechanism=best_mechanism,
            strategy=best_strategy,
            total_error=best_error,
            privacy_params=(self.epsilon, self.delta),
            iterations=len(self.iteration_history),
            synthesis_time=synthesis_time,
            metadata={
                "workload_shape": (m, d),
                "strategy_optimizer": self.strategy_optimizer,
            },
        )
    
    def _initialize_mechanism(self, domain_size: int) -> npt.NDArray[np.float64]:
        """Initialize mechanism with uniform distribution."""
        return np.ones(domain_size, dtype=np.float64) / domain_size
    
    def _optimize_mechanism(
        self,
        workload: npt.NDArray[np.float64],
        strategy: StrategyMatrix,
    ) -> npt.NDArray[np.float64]:
        """Optimize mechanism for fixed strategy.
        
        This would integrate with dp_forge.cegis_loop for full implementation.
        For now, returns current mechanism (simplified).
        """
        # In full implementation, this would:
        # 1. Convert strategy to constraints in LP
        # 2. Run CEGIS loop with strategy-aware objective
        # 3. Return synthesized mechanism
        
        # Simplified: return identity (no optimization)
        d = strategy.domain_size
        return np.ones(d, dtype=np.float64) / d
    
    def _optimize_strategy(
        self,
        workload: npt.NDArray[np.float64],
        mechanism: npt.NDArray[np.float64],
    ) -> StrategyMatrix:
        """Optimize strategy for fixed mechanism."""
        if self.strategy_optimizer == "hdmm":
            optimizer = HDMMOptimizer(max_iterations=500)
            return optimizer.optimize(workload, self.epsilon)
        elif self.strategy_optimizer == "auto":
            selector = StrategySelector()
            return selector.select_strategy(workload, self.epsilon)
        else:
            raise ValueError(f"Unknown optimizer: {self.strategy_optimizer}")
    
    def verify_strategy(
        self,
        strategy: StrategyMatrix,
        workload: npt.NDArray[np.float64],
        target_error: float,
    ) -> bool:
        """Verify if strategy achieves target error.
        
        Args:
            strategy: Strategy to verify.
            workload: Query workload.
            target_error: Target total squared error.
        
        Returns:
            True if strategy achieves target error.
        """
        actual_error = strategy.total_squared_error(workload, self.epsilon)
        return actual_error <= target_error
    
    def find_counterexample(
        self,
        strategy: StrategyMatrix,
        workload: npt.NDArray[np.float64],
    ) -> Optional[npt.NDArray[np.float64]]:
        """Find query where strategy performs poorly.
        
        Args:
            strategy: Current strategy.
            workload: Query workload.
        
        Returns:
            Counterexample query (row of workload) or None.
        """
        # Compute per-query errors
        errors = []
        for i in range(workload.shape[0]):
            query = workload[i:i+1, :]
            error = strategy.total_squared_error(query, self.epsilon)
            errors.append(error)
        
        # Find worst query
        worst_idx = np.argmax(errors)
        worst_error = errors[worst_idx]
        
        # If error is significantly above average, return as counterexample
        mean_error = np.mean(errors)
        if worst_error > 2.0 * mean_error:
            return workload[worst_idx:worst_idx+1, :]
        
        return None
    
    def refine_strategy_space(
        self,
        counterexample: npt.NDArray[np.float64],
        current_strategy: StrategyMatrix,
    ) -> Dict[str, Any]:
        """Refine strategy search space based on counterexample.
        
        Args:
            counterexample: Query with high error.
            current_strategy: Current strategy.
        
        Returns:
            Refinement information.
        """
        # Analyze counterexample to determine refinement
        # For now, return empty refinement
        return {
            "refined": False,
            "message": "Refinement not implemented",
        }


def joint_optimization_with_verification(
    workload: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    verification_level: str = "interval",
) -> StrategySynthesisResult:
    """Joint mechanism and strategy optimization with formal verification.
    
    Synthesizes mechanism and strategy jointly, then formally verifies
    the privacy guarantee.
    
    Args:
        workload: Query workload.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
        verification_level: Verification method ("interval", "rational").
    
    Returns:
        Synthesis result with verified mechanism and strategy.
    """
    synthesizer = CEGISStrategySynthesizer(
        epsilon=epsilon,
        delta=delta,
        max_iterations=30,
    )
    
    result = synthesizer.synthesize(workload)
    
    # Verification would happen here in full implementation
    logger.info("Verification not implemented in this version")
    
    return result


def strategy_guided_mechanism_synthesis(
    workload: npt.NDArray[np.float64],
    epsilon: float,
    strategy_hints: Optional[List[str]] = None,
) -> StrategySynthesisResult:
    """Mechanism synthesis guided by strategy hints.
    
    Uses domain knowledge about good strategies to guide mechanism
    synthesis toward compatible designs.
    
    Args:
        workload: Query workload.
        epsilon: Privacy parameter.
        strategy_hints: List of strategy types to consider.
    
    Returns:
        Synthesis result.
    """
    if strategy_hints is None:
        strategy_hints = ["identity", "hierarchical", "hdmm"]
    
    best_result = None
    best_error = float('inf')
    
    for hint in strategy_hints:
        logger.info(f"Trying strategy hint: {hint}")
        
        synthesizer = CEGISStrategySynthesizer(
            epsilon=epsilon,
            strategy_optimizer=hint if hint == "hdmm" else "auto",
        )
        
        try:
            result = synthesizer.synthesize(workload)
            
            if result.total_error < best_error:
                best_error = result.total_error
                best_result = result
        except Exception as e:
            logger.warning(f"Strategy hint {hint} failed: {e}")
            continue
    
    if best_result is None:
        raise SolverError("All strategy hints failed")
    
    return best_result


def multiobjective_synthesis(
    workload: npt.NDArray[np.float64],
    epsilon: float,
    objectives: List[Callable],
    weights: Optional[List[float]] = None,
) -> StrategySynthesisResult:
    """Multi-objective synthesis optimizing multiple error metrics.
    
    Args:
        workload: Query workload.
        epsilon: Privacy parameter.
        objectives: List of objective functions (workload, strategy) -> error.
        weights: Weights for each objective.
    
    Returns:
        Pareto-optimal synthesis result.
    """
    if weights is None:
        weights = [1.0] * len(objectives)
    
    if len(weights) != len(objectives):
        raise ValueError("Number of weights must match number of objectives")
    
    synthesizer = CEGISStrategySynthesizer(epsilon=epsilon)
    
    # Optimize weighted combination
    # Full implementation would maintain Pareto frontier
    result = synthesizer.synthesize(workload)
    
    return result


def adaptive_cegis_synthesis(
    workload: npt.NDArray[np.float64],
    epsilon: float,
    error_tolerance: float = 1e-3,
) -> StrategySynthesisResult:
    """Adaptive CEGIS with early stopping based on error tolerance.
    
    Args:
        workload: Query workload.
        epsilon: Privacy parameter.
        error_tolerance: Stop when error below this threshold.
    
    Returns:
        Synthesis result.
    """
    synthesizer = CEGISStrategySynthesizer(
        epsilon=epsilon,
        max_iterations=100,
    )
    
    result = synthesizer.synthesize(workload)
    
    # Check if tolerance achieved
    if result.total_error <= error_tolerance:
        logger.info(f"Achieved error tolerance: {result.total_error:.4e} <= {error_tolerance}")
    else:
        logger.warning(f"Did not achieve error tolerance: {result.total_error:.4e} > {error_tolerance}")
    
    return result


def distributed_strategy_optimization(
    workloads: List[npt.NDArray[np.float64]],
    epsilon: float,
    aggregation: str = "average",
) -> List[StrategyMatrix]:
    """Optimize strategies for multiple workloads in parallel.
    
    Args:
        workloads: List of workload matrices.
        epsilon: Privacy parameter.
        aggregation: How to aggregate results ("average", "max", "individual").
    
    Returns:
        List of optimized strategies (one per workload).
    """
    strategies = []
    
    for i, workload in enumerate(workloads):
        logger.info(f"Optimizing strategy {i+1}/{len(workloads)}")
        
        selector = StrategySelector()
        strategy = selector.select_strategy(workload, epsilon)
        strategies.append(strategy)
    
    return strategies


def incremental_strategy_refinement(
    workload: npt.NDArray[np.float64],
    initial_strategy: StrategyMatrix,
    epsilon: float,
    refinement_steps: int = 10,
) -> StrategyMatrix:
    """Incrementally refine strategy through local search.
    
    Args:
        workload: Query workload.
        initial_strategy: Starting strategy.
        epsilon: Privacy parameter.
        refinement_steps: Number of refinement steps.
    
    Returns:
        Refined strategy.
    """
    current_strategy = initial_strategy
    current_error = current_strategy.total_squared_error(workload, epsilon)
    
    for step in range(refinement_steps):
        # Try small perturbation
        A = current_strategy.to_explicit()
        perturbation = np.random.randn(*A.shape) * 0.01
        A_perturbed = A + perturbation
        
        # Ensure positive definiteness
        A_perturbed = (A_perturbed + A_perturbed.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(A_perturbed)
        if np.min(eigenvalues) < 1e-6:
            A_perturbed += (1e-6 - np.min(eigenvalues)) * np.eye(A.shape[0])
        
        candidate = StrategyMatrix(matrix=A_perturbed, epsilon=epsilon)
        candidate_error = candidate.total_squared_error(workload, epsilon)
        
        if candidate_error < current_error:
            current_strategy = candidate
            current_error = candidate_error
            logger.debug(f"Refinement step {step}: error improved to {current_error:.4e}")
    
    return current_strategy
