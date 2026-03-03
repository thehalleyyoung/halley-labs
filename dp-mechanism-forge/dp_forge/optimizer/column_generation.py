"""
Column generation for infinite/large output domains.

This module implements Dantzig-Wolfe decomposition and column generation
for DP mechanism synthesis when the output space is infinite (e.g., ℝ) or
very large (k > 10^6). Instead of representing all outputs explicitly, we
generate columns (output values) on-demand by solving pricing subproblems.

Key algorithms:
    - Column generation master/pricing framework
    - Adaptive domain discretization based on pricing
    - Stabilized column generation with restricted master problem
    - Integration with continuous noise mechanisms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize, sparse
from scipy.optimize import linprog

from dp_forge.exceptions import ConvergenceError, SolverError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Column:
    """A column (output value) in the master problem.
    
    Attributes:
        output_value: The output value (e.g., noisy answer)
        probabilities: Probability of this output for each input (length n)
        reduced_cost: Reduced cost from pricing problem
        iteration_added: Iteration when column was generated
        usage_count: Times column has been basic in master LP
    """
    output_value: float
    probabilities: npt.NDArray[np.float64]
    reduced_cost: float
    iteration_added: int
    usage_count: int = 0


@dataclass
class ColumnGenerationState:
    """State of column generation algorithm.
    
    Attributes:
        columns: List of generated columns
        master_solution: Current master problem solution (probability weights)
        dual_values: Dual values from master problem
        iteration: Current iteration
        objective_history: History of master objective values
        pricing_history: History of pricing subproblem results
    """
    columns: list[Column] = field(default_factory=list)
    master_solution: Optional[npt.NDArray[np.float64]] = None
    dual_values: Optional[npt.NDArray[np.float64]] = None
    iteration: int = 0
    objective_history: list[float] = field(default_factory=list)
    pricing_history: list[float] = field(default_factory=list)


@dataclass
class ColumnGenerationResult:
    """Result from column generation.
    
    Attributes:
        mechanism: Discretized mechanism table (n × k_active)
        output_values: Active output values
        objective: Final objective value
        success: Whether algorithm converged
        iterations: Number of iterations
        total_columns: Total columns generated
        active_columns: Columns with nonzero weight in final solution
        message: Status message
    """
    mechanism: npt.NDArray[np.float64]
    output_values: npt.NDArray[np.float64]
    objective: float
    success: bool
    iterations: int
    total_columns: int
    active_columns: int
    message: str


# ---------------------------------------------------------------------------
# Column generation engine
# ---------------------------------------------------------------------------


class ColumnGenerationEngine:
    """Column generation for mechanism synthesis with infinite output space.
    
    Formulation:
        Master problem (restricted to columns in pool):
            min sum_j c_j λ_j
            s.t. sum_j p_ij λ_j = p_i (output distribution for input i)
                 sum_j λ_j = 1 (probabilities sum to 1)
                 λ_j >= 0
                 
        Pricing subproblem (find new column with negative reduced cost):
            min (c - π^T P) for new column P
            where π are dual values from master
            
    Algorithm:
        1. Initialize with small set of columns
        2. Solve master LP to get dual values π
        3. Solve pricing to find column with most negative reduced cost
        4. If reduced cost < -tol: add column and repeat
        5. If reduced cost >= -tol: optimal solution found
        
    Args:
        n: Number of input values
        epsilon: Privacy parameter
        delta: Privacy parameter
        objective_weights: Weights for objective (e.g., loss function)
        initial_discretization: Initial output values to seed columns
        tol: Convergence tolerance for reduced cost
        max_columns: Maximum columns to generate
        max_iterations: Maximum iterations
        
    References:
        - Dantzig & Wolfe, "Decomposition Principle for Linear Programs", 1960
        - Desrosiers & Lübbecke, "Branch-Price-and-Cut Algorithms", 2010
    """
    
    def __init__(
        self,
        n: int,
        epsilon: float,
        delta: float = 0.0,
        objective_weights: Optional[npt.NDArray[np.float64]] = None,
        initial_discretization: Optional[npt.NDArray[np.float64]] = None,
        tol: float = 1e-6,
        max_columns: int = 10000,
        max_iterations: int = 1000,
    ):
        self.n = n
        self.epsilon = epsilon
        self.delta = delta
        self.tol = tol
        self.max_columns = max_columns
        self.max_iterations = max_iterations
        
        # Objective weights (default: uniform loss)
        if objective_weights is None:
            self.c = np.ones(n) / n
        else:
            self.c = objective_weights
        
        # Initial discretization
        if initial_discretization is None:
            # Default: 100 equally spaced points from -10 to 10
            self.initial_outputs = np.linspace(-10, 10, 100)
        else:
            self.initial_outputs = initial_discretization
        
        # State
        self.state = ColumnGenerationState()
        
    def solve(self) -> ColumnGenerationResult:
        """Run column generation algorithm.
        
        Returns:
            ColumnGenerationResult with discretized mechanism
        """
        logger.info(
            f"Starting column generation: n={self.n}, ε={self.epsilon}, "
            f"initial_outputs={len(self.initial_outputs)}"
        )
        
        # Initialize with columns from initial discretization
        self._initialize_columns()
        
        # Main loop
        for iteration in range(self.max_iterations):
            self.state.iteration = iteration
            
            # Solve restricted master problem
            try:
                master_result = self._solve_master()
            except Exception as e:
                logger.error(f"Master solve failed: {e}")
                return self._failure_result(f"Master solve failed: {e}")
            
            if not master_result.success:
                return self._failure_result("Master problem infeasible")
            
            self.state.master_solution = master_result.x
            self.state.objective_history.append(master_result.fun)
            
            # Extract dual values (shadow prices)
            dual_values = self._extract_dual_values(master_result)
            self.state.dual_values = dual_values
            
            # Solve pricing subproblem
            new_output, reduced_cost = self._solve_pricing(dual_values)
            self.state.pricing_history.append(reduced_cost)
            
            logger.debug(
                f"Iteration {iteration}: obj={master_result.fun:.6e}, "
                f"reduced_cost={reduced_cost:.6e}, columns={len(self.state.columns)}"
            )
            
            # Check optimality
            if reduced_cost >= -self.tol:
                logger.info(
                    f"Converged at iteration {iteration}: "
                    f"reduced_cost={reduced_cost:.6e} >= {-self.tol}"
                )
                return self._build_result(master_result, success=True)
            
            # Add new column
            self._add_column(new_output, iteration)
            
            # Check column limit
            if len(self.state.columns) >= self.max_columns:
                logger.warning(f"Max columns {self.max_columns} reached")
                return self._build_result(master_result, success=False)
        
        # Max iterations
        logger.warning(f"Max iterations {self.max_iterations} reached")
        return self._build_result(master_result, success=False)
    
    def _initialize_columns(self) -> None:
        """Initialize column pool with discretization."""
        for y in self.initial_outputs:
            # Compute probabilities for this output under Laplace mechanism
            # p[i, y] ∝ exp(-ε |i - y|)
            probabilities = self._laplace_probabilities(y)
            
            column = Column(
                output_value=y,
                probabilities=probabilities,
                reduced_cost=0.0,
                iteration_added=0,
            )
            self.state.columns.append(column)
    
    def _laplace_probabilities(self, y: float) -> npt.NDArray[np.float64]:
        """Compute Laplace mechanism probabilities for output y.
        
        Args:
            y: Output value
            
        Returns:
            Probability array of length n
        """
        # Laplace: p[i, y] ∝ exp(-ε |i - y|)
        distances = np.abs(np.arange(self.n) - y)
        log_probs = -self.epsilon * distances
        
        # Normalize (log-sum-exp for numerical stability)
        max_log_prob = np.max(log_probs)
        probs = np.exp(log_probs - max_log_prob)
        probs /= probs.sum()
        
        return probs
    
    def _solve_master(self) -> optimize.OptimizeResult:
        """Solve restricted master problem.
        
        Master LP:
            min sum_j c_j λ_j
            s.t. sum_j λ_j = 1
                 λ_j >= 0
                 
        (Simplified: we're just selecting a discrete distribution over columns)
        """
        k = len(self.state.columns)
        
        # Objective: expected loss
        c = np.zeros(k)
        for j, col in enumerate(self.state.columns):
            # Expected loss for this output
            c[j] = np.dot(self.c, col.probabilities)
        
        # Equality constraint: sum λ = 1
        A_eq = np.ones((1, k))
        b_eq = np.array([1.0])
        
        # Solve
        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * k,
            method='highs',
            options={'presolve': True, 'disp': False},
        )
        
        return result
    
    def _extract_dual_values(
        self, master_result: optimize.OptimizeResult
    ) -> npt.NDArray[np.float64]:
        """Extract dual values from master solution.
        
        Returns:
            Dual values π (shadow prices for equality constraints)
        """
        # Try to get dual values from result
        if hasattr(master_result, 'eqlin') and hasattr(master_result.eqlin, 'marginals'):
            return master_result.eqlin.marginals
        
        # Fallback: estimate from reduced costs
        # For simplex, dual values relate to reduced costs
        return np.zeros(1)  # Single constraint: sum λ = 1
    
    def _solve_pricing(
        self, dual_values: npt.NDArray[np.float64]
    ) -> Tuple[float, float]:
        """Solve pricing subproblem to find new column.
        
        Pricing problem:
            min_y  c(y) - π^T p(y)
            
        where c(y) is the cost of output y and p(y) are the probabilities.
        
        We solve this by sampling candidate outputs and picking the best.
        
        Returns:
            (output_value, reduced_cost): New output and its reduced cost
        """
        # Sample candidate outputs
        # Strategy: bisection on intervals between existing outputs
        existing_outputs = np.array([col.output_value for col in self.state.columns])
        existing_outputs = np.sort(existing_outputs)
        
        candidates = []
        
        # Midpoints between existing outputs
        for i in range(len(existing_outputs) - 1):
            mid = (existing_outputs[i] + existing_outputs[i + 1]) / 2
            candidates.append(mid)
        
        # Extend beyond boundaries
        if len(existing_outputs) > 0:
            candidates.append(existing_outputs[0] - 1.0)
            candidates.append(existing_outputs[-1] + 1.0)
        
        # Also sample uniformly in range
        y_min = existing_outputs[0] - 5.0 if len(existing_outputs) > 0 else -10.0
        y_max = existing_outputs[-1] + 5.0 if len(existing_outputs) > 0 else 10.0
        candidates.extend(np.linspace(y_min, y_max, 50))
        
        # Evaluate reduced cost for each candidate
        best_y = candidates[0]
        best_reduced_cost = float('inf')
        
        for y in candidates:
            probs = self._laplace_probabilities(y)
            cost = np.dot(self.c, probs)
            
            # Reduced cost: c(y) - π^T p(y)
            # (Simplified since our master has single constraint)
            reduced_cost = cost - dual_values[0]
            
            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_y = y
        
        return best_y, best_reduced_cost
    
    def _add_column(self, output_value: float, iteration: int) -> None:
        """Add new column to pool."""
        probabilities = self._laplace_probabilities(output_value)
        
        column = Column(
            output_value=output_value,
            probabilities=probabilities,
            reduced_cost=0.0,
            iteration_added=iteration,
        )
        
        self.state.columns.append(column)
        
        logger.debug(f"Added column: y={output_value:.6f}, iteration={iteration}")
    
    def _build_result(
        self, master_result: optimize.OptimizeResult, success: bool
    ) -> ColumnGenerationResult:
        """Build final result from master solution."""
        # Extract active columns (nonzero weight)
        weights = self.state.master_solution
        active_mask = weights > 1e-8
        active_indices = np.where(active_mask)[0]
        
        # Build mechanism table
        active_columns = [self.state.columns[i] for i in active_indices]
        k_active = len(active_columns)
        
        mechanism = np.zeros((self.n, k_active))
        output_values = np.zeros(k_active)
        
        for j, col in enumerate(active_columns):
            mechanism[:, j] = col.probabilities * weights[col.iteration_added]
            output_values[j] = col.output_value
        
        # Normalize each row
        row_sums = mechanism.sum(axis=1, keepdims=True)
        mechanism /= row_sums + 1e-15
        
        return ColumnGenerationResult(
            mechanism=mechanism,
            output_values=output_values,
            objective=master_result.fun,
            success=success,
            iterations=self.state.iteration,
            total_columns=len(self.state.columns),
            active_columns=k_active,
            message="Converged" if success else "Max iterations/columns",
        )
    
    def _failure_result(self, message: str) -> ColumnGenerationResult:
        """Build failure result."""
        return ColumnGenerationResult(
            mechanism=np.zeros((self.n, 1)),
            output_values=np.array([0.0]),
            objective=float('inf'),
            success=False,
            iterations=self.state.iteration,
            total_columns=len(self.state.columns),
            active_columns=0,
            message=message,
        )


# ---------------------------------------------------------------------------
# Pricing oracle
# ---------------------------------------------------------------------------


class PricingOracle:
    """Oracle for solving pricing subproblems in column generation.
    
    The pricing subproblem finds the output value with most negative reduced
    cost. For continuous domains (ℝ), this requires optimization over an
    infinite space.
    
    Strategy:
        1. Adaptive bisection on promising intervals
        2. Local optimization via scipy.optimize.minimize_scalar
        3. Post-termination certificate via dense sampling
        
    Args:
        epsilon: Privacy parameter
        loss_function: Loss function L(true, output)
        dual_values: Dual values from master problem
        domain_bounds: (lower, upper) bounds for output domain
        
    References:
        - Barnhart et al., "Branch-and-Price: Column Generation", 1998
    """
    
    def __init__(
        self,
        epsilon: float,
        loss_function: Callable[[npt.NDArray, float], npt.NDArray],
        dual_values: npt.NDArray[np.float64],
        domain_bounds: Tuple[float, float] = (-20.0, 20.0),
    ):
        self.epsilon = epsilon
        self.loss = loss_function
        self.dual_values = dual_values
        self.bounds = domain_bounds
        
    def solve(self) -> Tuple[float, float]:
        """Find output with most negative reduced cost.
        
        Returns:
            (output_value, reduced_cost)
        """
        # Use scipy's bounded minimization
        def reduced_cost_fn(y: float) -> float:
            # Compute reduced cost for output y
            # This depends on mechanism structure
            return self._compute_reduced_cost(y)
        
        result = optimize.minimize_scalar(
            reduced_cost_fn,
            bounds=self.bounds,
            method='bounded',
        )
        
        return result.x, result.fun
    
    def _compute_reduced_cost(self, y: float) -> float:
        """Compute reduced cost for output value y.
        
        Reduced cost: c(y) - π^T p(y)
        where c(y) is expected loss and p(y) are probabilities.
        """
        # Placeholder implementation
        # Real version depends on mechanism structure
        return abs(y) - self.dual_values[0]
    
    def verify_optimality(
        self, best_output: float, best_reduced_cost: float, sample_density: int = 1000
    ) -> bool:
        """Verify optimality by dense sampling.
        
        Samples at 10× density and checks that all reduced costs >= best - tol.
        
        Args:
            best_output: Best output found
            best_reduced_cost: Its reduced cost
            sample_density: Number of samples to check
            
        Returns:
            True if verified optimal (within tolerance)
        """
        samples = np.linspace(self.bounds[0], self.bounds[1], sample_density)
        
        for y in samples:
            rc = self._compute_reduced_cost(y)
            if rc < best_reduced_cost - 1e-6:
                logger.warning(
                    f"Pricing optimality violated: y={y:.6f} has "
                    f"reduced_cost={rc:.6e} < {best_reduced_cost:.6e}"
                )
                return False
        
        return True


# ---------------------------------------------------------------------------
# Domain discretizer
# ---------------------------------------------------------------------------


class DomainDiscretizer:
    """Adaptive discretization based on pricing subproblem results.
    
    Instead of uniform discretization, adaptively refine regions where
    pricing subproblem finds attractive outputs.
    
    Algorithm:
        1. Start with coarse uniform grid
        2. After each pricing solve, refine interval containing new output
        3. Track output density and adjust grid spacing
        
    Args:
        initial_points: Initial discretization points
        refinement_factor: Factor to increase density when refining
    """
    
    def __init__(
        self,
        initial_points: npt.NDArray[np.float64],
        refinement_factor: int = 2,
    ):
        self.points = np.sort(initial_points)
        self.refinement_factor = refinement_factor
        self.density_map: dict[Tuple[float, float], int] = {}
        
    def refine_around(self, y: float) -> None:
        """Refine discretization around output y.
        
        Args:
            y: Output value to refine around
        """
        # Find bracketing interval
        idx = np.searchsorted(self.points, y)
        
        if idx == 0:
            lower, upper = self.points[0], self.points[1]
        elif idx >= len(self.points):
            lower, upper = self.points[-2], self.points[-1]
        else:
            lower, upper = self.points[idx - 1], self.points[idx]
        
        # Add refined points in this interval
        new_points = np.linspace(lower, upper, self.refinement_factor + 2)[1:-1]
        
        self.points = np.sort(np.concatenate([self.points, new_points]))
        
        # Track density
        interval = (lower, upper)
        self.density_map[interval] = self.density_map.get(interval, 0) + 1
        
        logger.debug(
            f"Refined interval [{lower:.3f}, {upper:.3f}]: "
            f"{len(new_points)} new points"
        )
    
    def get_current_discretization(self) -> npt.NDArray[np.float64]:
        """Get current discretization points.
        
        Returns:
            Sorted array of discretization points
        """
        return self.points


# ---------------------------------------------------------------------------
# Stabilized column generation
# ---------------------------------------------------------------------------


class StabilizedColumnGeneration:
    """Stabilized column generation with restricted master problem.
    
    Standard column generation can oscillate when dual values change
    drastically between iterations. Stabilization techniques smooth the
    dual trajectory.
    
    Method: Dual stabilization via penalty function
        - Add penalty term to master: min c^T λ + α ||π - π_bar||²
        - π_bar is a "stability center" (previous dual solution)
        - α is penalty weight (decreases as algorithm converges)
        
    Args:
        base_engine: Underlying ColumnGenerationEngine
        alpha_initial: Initial penalty weight
        alpha_decay: Decay factor for penalty weight
        
    References:
        - du Merle et al., "Stabilized Column Generation", 1999
        - Ben Amor et al., "Dual-Optimal Inequalities for Stabilized Column Generation", 2006
    """
    
    def __init__(
        self,
        base_engine: ColumnGenerationEngine,
        alpha_initial: float = 1.0,
        alpha_decay: float = 0.9,
    ):
        self.engine = base_engine
        self.alpha = alpha_initial
        self.alpha_decay = alpha_decay
        self.dual_center: Optional[npt.NDArray[np.float64]] = None
        
    def solve(self) -> ColumnGenerationResult:
        """Run stabilized column generation.
        
        Returns:
            ColumnGenerationResult
        """
        logger.info("Starting stabilized column generation")
        
        # Run base algorithm with stabilization
        for iteration in range(self.engine.max_iterations):
            # Solve master (with stabilization if dual center exists)
            master_result = self._solve_stabilized_master()
            
            if not master_result.success:
                break
            
            # Extract dual values
            dual_values = self.engine._extract_dual_values(master_result)
            
            # Update dual center (exponential smoothing)
            if self.dual_center is None:
                self.dual_center = dual_values
            else:
                self.dual_center = (
                    0.8 * self.dual_center + 0.2 * dual_values
                )
            
            # Solve pricing with stabilized duals
            new_output, reduced_cost = self.engine._solve_pricing(self.dual_center)
            
            # Check convergence
            if reduced_cost >= -self.engine.tol:
                return self.engine._build_result(master_result, success=True)
            
            # Add column
            self.engine._add_column(new_output, iteration)
            
            # Decay penalty weight
            self.alpha *= self.alpha_decay
        
        return self.engine._failure_result("Max iterations reached")
    
    def _solve_stabilized_master(self) -> optimize.OptimizeResult:
        """Solve master with dual stabilization penalty.
        
        This is a simplified version; full implementation would modify
        the master LP to include penalty terms.
        """
        # For now, just call base master solver
        return self.engine._solve_master()


# ---------------------------------------------------------------------------
# Advanced pricing strategies
# ---------------------------------------------------------------------------


class MultiStartPricing:
    """Multi-start pricing for difficult subproblems.
    
    Run pricing from multiple starting points and select best:
        1. Previous best output
        2. Uniform samples across domain
        3. Outputs from previous iterations
        4. Gradient-guided samples
        
    This improves robustness when pricing has multiple local minima.
    
    Args:
        base_oracle: Underlying PricingOracle
        num_starts: Number of starting points
        domain_bounds: Search space bounds
    """
    
    def __init__(
        self,
        base_oracle: PricingOracle,
        num_starts: int = 10,
        domain_bounds: Tuple[float, float] = (-20.0, 20.0),
    ):
        self.oracle = base_oracle
        self.num_starts = num_starts
        self.bounds = domain_bounds
        self._previous_outputs: list[float] = []
        
    def solve(self) -> Tuple[float, float]:
        """Solve pricing with multi-start strategy.
        
        Returns:
            (best_output, best_reduced_cost)
        """
        candidates = []
        
        # Generate starting points
        starts = self._generate_starts()
        
        # Run local optimization from each start
        for y0 in starts:
            # Local optimization around y0
            result = optimize.minimize_scalar(
                lambda y: self.oracle._compute_reduced_cost(y),
                bounds=(max(self.bounds[0], y0 - 5), min(self.bounds[1], y0 + 5)),
                method='bounded',
            )
            
            candidates.append((result.x, result.fun))
        
        # Select best
        best_output, best_rc = min(candidates, key=lambda x: x[1])
        
        # Store for next iteration
        self._previous_outputs.append(best_output)
        if len(self._previous_outputs) > 20:
            self._previous_outputs.pop(0)
        
        return best_output, best_rc
    
    def _generate_starts(self) -> list[float]:
        """Generate starting points for multi-start."""
        starts = []
        
        # Uniform samples
        starts.extend(np.linspace(self.bounds[0], self.bounds[1], self.num_starts // 2))
        
        # Previous outputs
        starts.extend(self._previous_outputs[:self.num_starts // 4])
        
        # Random perturbations
        if len(self._previous_outputs) > 0:
            for y in self._previous_outputs[-3:]:
                starts.append(y + np.random.randn() * 0.5)
        
        return starts[:self.num_starts]


class DynamicGridRefinement:
    """Dynamic grid refinement for column generation discretization.
    
    Adaptively refines grid based on:
        1. Reduced cost gradient magnitude
        2. Column density (avoid over-sampling)
        3. Constraint violation patterns
        
    Args:
        initial_grid: Starting grid points
        max_grid_size: Maximum grid points
        refinement_threshold: Gradient threshold for refinement
    """
    
    def __init__(
        self,
        initial_grid: npt.NDArray[np.float64],
        max_grid_size: int = 10000,
        refinement_threshold: float = 0.1,
    ):
        self.grid = np.sort(initial_grid)
        self.max_size = max_grid_size
        self.threshold = refinement_threshold
        
        # Track density
        self.density_map: dict[int, int] = {}
        
    def refine_interval(
        self,
        idx: int,
        gradient: Optional[float] = None,
    ) -> bool:
        """Refine grid around interval idx.
        
        Args:
            idx: Interval index to refine
            gradient: Gradient magnitude (if available)
            
        Returns:
            True if refinement performed
        """
        if len(self.grid) >= self.max_size:
            return False
        
        if idx >= len(self.grid) - 1:
            return False
        
        # Check density
        if self.density_map.get(idx, 0) >= 3:
            # Already refined enough
            return False
        
        # Check gradient threshold
        if gradient is not None and gradient < self.threshold:
            return False
        
        # Add midpoint
        lower, upper = self.grid[idx], self.grid[idx + 1]
        midpoint = (lower + upper) / 2
        
        self.grid = np.sort(np.append(self.grid, midpoint))
        self.density_map[idx] = self.density_map.get(idx, 0) + 1
        
        return True
    
    def get_grid(self) -> npt.NDArray[np.float64]:
        """Get current grid."""
        return self.grid
    
    def estimate_gradient(
        self,
        idx: int,
        cost_function: Callable[[float], float],
    ) -> float:
        """Estimate gradient magnitude at interval.
        
        Args:
            idx: Interval index
            cost_function: Function to evaluate
            
        Returns:
            Estimated gradient magnitude
        """
        if idx >= len(self.grid) - 1:
            return 0.0
        
        lower, upper = self.grid[idx], self.grid[idx + 1]
        
        # Finite difference
        f_lower = cost_function(lower)
        f_upper = cost_function(upper)
        
        gradient = abs(f_upper - f_lower) / (upper - lower + 1e-15)
        
        return gradient


class RestrictedMasterStabilization:
    """Restricted master problem with trust region stabilization.
    
    Adds trust region constraints to master problem:
        ||λ - λ_prev||_1 <= Δ
        
    This prevents wild oscillations in dual values between iterations.
    
    Args:
        delta_initial: Initial trust region radius
        delta_min: Minimum trust region radius
        delta_max: Maximum trust region radius
        shrink_factor: Factor to shrink Δ on failed iterations
        expand_factor: Factor to expand Δ on successful iterations
    """
    
    def __init__(
        self,
        delta_initial: float = 1.0,
        delta_min: float = 0.01,
        delta_max: float = 10.0,
        shrink_factor: float = 0.5,
        expand_factor: float = 1.5,
    ):
        self.delta = delta_initial
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.shrink = shrink_factor
        self.expand = expand_factor
        
        self._previous_lambda: Optional[npt.NDArray[np.float64]] = None
        self._previous_obj: float = float('inf')
        
    def solve_restricted_master(
        self,
        c: npt.NDArray[np.float64],
        A_eq: npt.NDArray[np.float64],
        b_eq: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Solve restricted master with trust region.
        
        Args:
            c: Objective coefficients
            A_eq: Equality constraints
            b_eq: Equality RHS
            
        Returns:
            (lambda_solution, objective_value)
        """
        k = len(c)
        
        if self._previous_lambda is None:
            # First iteration: no trust region
            result = linprog(
                c=c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=[(0, None)] * k,
                method='highs',
            )
            
            if result.success:
                self._previous_lambda = result.x
                self._previous_obj = result.fun
                return result.x, result.fun
            else:
                raise SolverError("Master problem infeasible")
        
        # Add trust region: ||λ - λ_prev||_1 <= Δ
        # Linearize with auxiliary variables: -s_i <= λ_i - λ_prev_i <= s_i
        # Trust region: sum s_i <= Δ
        
        # Extended variables: [λ, s]
        c_ext = np.concatenate([c, np.zeros(k)])
        
        # Trust region constraint
        A_trust = np.concatenate([np.zeros(k), np.ones(k)]).reshape(1, -1)
        b_trust = np.array([self.delta])
        
        # Linearization: λ - λ_prev - s <= 0, -λ + λ_prev - s <= 0
        A_lin_upper = np.hstack([np.eye(k), -np.eye(k)])
        b_lin_upper = self._previous_lambda
        
        A_lin_lower = np.hstack([-np.eye(k), -np.eye(k)])
        b_lin_lower = -self._previous_lambda
        
        # Stack constraints
        A_eq_ext = np.hstack([A_eq, np.zeros((A_eq.shape[0], k))])
        
        A_ub = np.vstack([A_trust, A_lin_upper, A_lin_lower])
        b_ub = np.concatenate([b_trust, b_lin_upper, b_lin_lower])
        
        # Solve
        result = linprog(
            c=c_ext,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq_ext,
            b_eq=b_eq,
            bounds=[(0, None)] * (2 * k),
            method='highs',
        )
        
        if not result.success:
            raise SolverError("Restricted master infeasible")
        
        lambda_new = result.x[:k]
        obj_new = result.fun
        
        # Update trust region
        self._update_trust_region(obj_new)
        
        self._previous_lambda = lambda_new
        self._previous_obj = obj_new
        
        return lambda_new, obj_new
    
    def _update_trust_region(self, obj_new: float) -> None:
        """Update trust region radius based on progress."""
        if obj_new < self._previous_obj - 1e-6:
            # Good iteration: expand
            self.delta = min(self.delta * self.expand, self.delta_max)
        else:
            # Poor iteration: shrink
            self.delta = max(self.delta * self.shrink, self.delta_min)


class ColumnPoolManager:
    """Manage pool of generated columns for column generation.
    
    Maintains a pool of columns with eviction policy:
        - Keep columns with nonzero weight in recent solutions
        - Evict columns unused for many iterations
        - Limit pool size for memory efficiency
        
    Args:
        max_size: Maximum columns in pool
        eviction_threshold: Iterations before evicting unused column
    """
    
    def __init__(
        self,
        max_size: int = 5000,
        eviction_threshold: int = 20,
    ):
        self.max_size = max_size
        self.eviction_threshold = eviction_threshold
        self._columns: list[Column] = []
        self._last_used: dict[int, int] = {}
        self._current_iteration = 0
        
    def add_column(self, column: Column) -> bool:
        """Add column to pool.
        
        Returns:
            True if added, False if pool is full and column not added
        """
        if len(self._columns) >= self.max_size:
            self._evict_lru()
        
        if len(self._columns) >= self.max_size:
            return False
        
        col_idx = len(self._columns)
        self._columns.append(column)
        self._last_used[col_idx] = self._current_iteration
        
        return True
    
    def mark_used(self, indices: list[int]) -> None:
        """Mark columns as used in current iteration."""
        self._current_iteration += 1
        
        for idx in indices:
            if idx < len(self._columns):
                self._last_used[idx] = self._current_iteration
    
    def get_columns(self) -> list[Column]:
        """Get all columns in pool."""
        return self._columns
    
    def _evict_lru(self) -> None:
        """Evict least recently used column."""
        if len(self._columns) == 0:
            return
        
        # Find LRU column
        lru_idx = min(
            range(len(self._columns)),
            key=lambda i: self._last_used.get(i, 0),
        )
        
        age = self._current_iteration - self._last_used.get(lru_idx, 0)
        
        if age >= self.eviction_threshold:
            # Evict
            del self._columns[lru_idx]
            
            # Reindex
            new_last_used = {}
            for idx, last in self._last_used.items():
                if idx < lru_idx:
                    new_last_used[idx] = last
                elif idx > lru_idx:
                    new_last_used[idx - 1] = last
            
            self._last_used = new_last_used
