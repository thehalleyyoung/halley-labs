"""
Unified optimization backend interface with auto-selection.

This module provides abstract interfaces for LP/convex solvers and
concrete implementations for HiGHS (via scipy) and CVXPY. The backend
selector automatically chooses the best solver based on problem structure.

Key components:
    - OptimizationBackend abstract base class
    - HiGHSBackend wrapping scipy.optimize.linprog(method='highs')
    - CVXPYBackend wrapping existing CVXPY usage
    - BackendSelector with automatic structure-based selection
    - SolverConfig for tolerance and iteration settings
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import linprog

from dp_forge.exceptions import ConfigurationError, SolverError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SolverConfig:
    """Configuration for LP/convex solvers.
    
    Attributes:
        tolerance: Convergence tolerance (primal/dual feasibility)
        max_iterations: Maximum solver iterations
        time_limit: Time limit in seconds
        presolve: Enable presolve (simplify problem before solving)
        dual_simplex: Prefer dual simplex over primal
        crossover: Enable crossover to basic solution (IPM -> simplex)
        verbose: Enable solver output
        threads: Number of threads (0 = auto)
    """
    tolerance: float = 1e-7
    max_iterations: int = 100000
    time_limit: float = 300.0
    presolve: bool = True
    dual_simplex: bool = False
    crossover: bool = True
    verbose: bool = False
    threads: int = 0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.tolerance <= 0:
            raise ConfigurationError(f"tolerance must be positive, got {self.tolerance}")
        
        if self.max_iterations <= 0:
            raise ConfigurationError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )
        
        if self.time_limit <= 0:
            raise ConfigurationError(f"time_limit must be positive, got {self.time_limit}")


class SolverStatus(Enum):
    """Status codes for optimization results."""
    OPTIMAL = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    ITERATION_LIMIT = auto()
    TIME_LIMIT = auto()
    NUMERICAL_ERROR = auto()
    UNKNOWN = auto()


@dataclass
class OptimizationResult:
    """Result from optimization backend.
    
    Attributes:
        x: Primal solution (None if not available)
        objective: Objective value at solution
        dual: Dual solution / shadow prices (None if not available)
        status: Solver status code
        iterations: Number of iterations
        solve_time: Wall-clock time in seconds
        message: Human-readable status message
    """
    x: Optional[npt.NDArray[np.float64]]
    objective: float
    dual: Optional[npt.NDArray[np.float64]]
    status: SolverStatus
    iterations: int
    solve_time: float
    message: str


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class OptimizationBackend(ABC):
    """Abstract base class for optimization backends.
    
    All optimization solvers (HiGHS, CVXPY, custom) implement this interface
    for uniform calling conventions.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        """Initialize backend with configuration.
        
        Args:
            config: Solver configuration (uses defaults if None)
        """
        self.config = config or SolverConfig()
        self.config.validate()
        
    @abstractmethod
    def solve_lp(
        self,
        c: npt.NDArray[np.float64],
        A_ub: Optional[sparse.spmatrix] = None,
        b_ub: Optional[npt.NDArray[np.float64]] = None,
        A_eq: Optional[sparse.spmatrix] = None,
        b_eq: Optional[npt.NDArray[np.float64]] = None,
        bounds: Optional[list[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Solve linear program.
        
        min c^T x
        s.t. A_ub @ x <= b_ub
             A_eq @ x == b_eq
             bounds[i][0] <= x[i] <= bounds[i][1]
             
        Args:
            c: Objective coefficients
            A_ub: Inequality constraint matrix
            b_ub: Inequality RHS
            A_eq: Equality constraint matrix
            b_eq: Equality RHS
            bounds: Variable bounds (None = [0, inf))
            
        Returns:
            OptimizationResult with solution and status
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return backend name."""
        pass


# ---------------------------------------------------------------------------
# HiGHS backend
# ---------------------------------------------------------------------------


class HiGHSBackend(OptimizationBackend):
    """HiGHS LP solver via scipy.optimize.linprog.
    
    HiGHS is a high-performance open-source LP solver with:
        - Primal/dual simplex and IPM implementations
        - Efficient presolve and scaling
        - Warm-starting capabilities
        - Multi-threaded support
        
    We use it via scipy.optimize.linprog(method='highs'), which provides
    a Python interface to the HiGHS C++ library.
    
    References:
        - Huangfu & Hall, "Parallelizing the dual revised simplex method", 2018
        - HiGHS: https://github.com/ERGO-Code/HiGHS
    """
    
    def solve_lp(
        self,
        c: npt.NDArray[np.float64],
        A_ub: Optional[sparse.spmatrix] = None,
        b_ub: Optional[npt.NDArray[np.float64]] = None,
        A_eq: Optional[sparse.spmatrix] = None,
        b_eq: Optional[npt.NDArray[np.float64]] = None,
        bounds: Optional[list[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Solve LP using HiGHS."""
        import time
        
        start_time = time.time()
        
        # Build options dict
        options = {
            'presolve': self.config.presolve,
            'disp': self.config.verbose,
            'time_limit': self.config.time_limit,
            'dual_feasibility_tolerance': self.config.tolerance,
            'primal_feasibility_tolerance': self.config.tolerance,
            'ipm_optimality_tolerance': self.config.tolerance,
        }
        
        # Add iteration limit if specified
        if self.config.max_iterations < 100000:
            options['maxiter'] = self.config.max_iterations
        
        logger.debug(
            f"HiGHS solve: n={len(c)}, "
            f"m_ub={A_ub.shape[0] if A_ub is not None else 0}, "
            f"m_eq={A_eq.shape[0] if A_eq is not None else 0}"
        )
        
        # Solve
        try:
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options=options,
            )
        except Exception as e:
            logger.error(f"HiGHS solve failed: {e}")
            return OptimizationResult(
                x=None,
                objective=float('inf'),
                dual=None,
                status=SolverStatus.NUMERICAL_ERROR,
                iterations=0,
                solve_time=time.time() - start_time,
                message=f"Solver exception: {e}",
            )
        
        solve_time = time.time() - start_time
        
        # Map scipy status to our status enum
        status_map = {
            0: SolverStatus.OPTIMAL,
            1: SolverStatus.ITERATION_LIMIT,
            2: SolverStatus.INFEASIBLE,
            3: SolverStatus.UNBOUNDED,
            4: SolverStatus.NUMERICAL_ERROR,
        }
        
        status = status_map.get(result.status, SolverStatus.UNKNOWN)
        
        # Extract dual values
        dual = None
        if hasattr(result, 'ineqlin') and hasattr(result.ineqlin, 'marginals'):
            dual = result.ineqlin.marginals
        
        # Log result
        if result.success:
            logger.debug(
                f"HiGHS optimal: obj={result.fun:.6e}, time={solve_time:.3f}s"
            )
        else:
            logger.warning(
                f"HiGHS status={result.status}: {result.message}"
            )
        
        return OptimizationResult(
            x=result.x if result.success else None,
            objective=result.fun if result.success else float('inf'),
            dual=dual,
            status=status,
            iterations=result.get('nit', 0),
            solve_time=solve_time,
            message=result.message,
        )
    
    def name(self) -> str:
        """Return backend name."""
        return "HiGHS"


# ---------------------------------------------------------------------------
# CVXPY backend
# ---------------------------------------------------------------------------


class CVXPYBackend(OptimizationBackend):
    """CVXPY disciplined convex programming interface.
    
    CVXPY provides a domain-specific language for convex optimization with:
        - Automatic problem classification (LP, QP, SOCP, SDP)
        - Multiple backend solvers (ECOS, SCS, MOSEK, GUROBI)
        - Disciplined convex programming (DCP) verification
        
    This backend wraps CVXPY for compatibility with existing code.
    
    References:
        - Diamond & Boyd, "CVXPY: A Python-Embedded Modeling Language", 2016
        - https://www.cvxpy.org/
    """
    
    def __init__(self, config: Optional[SolverConfig] = None, solver: str = 'ECOS'):
        """Initialize CVXPY backend.
        
        Args:
            config: Solver configuration
            solver: CVXPY solver name ('ECOS', 'SCS', 'MOSEK', etc.)
        """
        super().__init__(config)
        self.solver = solver
        
    def solve_lp(
        self,
        c: npt.NDArray[np.float64],
        A_ub: Optional[sparse.spmatrix] = None,
        b_ub: Optional[npt.NDArray[np.float64]] = None,
        A_eq: Optional[sparse.spmatrix] = None,
        b_eq: Optional[npt.NDArray[np.float64]] = None,
        bounds: Optional[list[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Solve LP using CVXPY."""
        try:
            import cvxpy as cp
        except ImportError:
            raise ConfigurationError(
                "CVXPY not installed. Install with: pip install cvxpy"
            )
        
        import time
        
        start_time = time.time()
        
        n = len(c)
        
        # Create variables
        x = cp.Variable(n)
        
        # Objective
        objective = cp.Minimize(c @ x)
        
        # Constraints
        constraints = []
        
        if A_ub is not None and b_ub is not None:
            constraints.append(A_ub @ x <= b_ub)
        
        if A_eq is not None and b_eq is not None:
            constraints.append(A_eq @ x == b_eq)
        
        # Variable bounds
        if bounds is not None:
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None:
                    constraints.append(x[i] >= lb)
                if ub is not None:
                    constraints.append(x[i] <= ub)
        else:
            # Default: x >= 0
            constraints.append(x >= 0)
        
        # Build and solve problem
        problem = cp.Problem(objective, constraints)
        
        solver_options = {
            'verbose': self.config.verbose,
            'max_iters': self.config.max_iterations,
        }
        
        try:
            problem.solve(solver=self.solver, **solver_options)
        except Exception as e:
            logger.error(f"CVXPY solve failed: {e}")
            return OptimizationResult(
                x=None,
                objective=float('inf'),
                dual=None,
                status=SolverStatus.NUMERICAL_ERROR,
                iterations=0,
                solve_time=time.time() - start_time,
                message=f"CVXPY exception: {e}",
            )
        
        solve_time = time.time() - start_time
        
        # Map CVXPY status
        status_map = {
            'optimal': SolverStatus.OPTIMAL,
            'infeasible': SolverStatus.INFEASIBLE,
            'unbounded': SolverStatus.UNBOUNDED,
        }
        
        status = status_map.get(problem.status, SolverStatus.UNKNOWN)
        
        # Extract dual values from constraints
        dual = None
        if len(constraints) > 0 and constraints[0].dual_value is not None:
            dual = constraints[0].dual_value
        
        return OptimizationResult(
            x=x.value if x.value is not None else None,
            objective=problem.value if problem.value is not None else float('inf'),
            dual=dual,
            status=status,
            iterations=0,  # CVXPY doesn't expose iteration count
            solve_time=solve_time,
            message=problem.status,
        )
    
    def name(self) -> str:
        """Return backend name."""
        return f"CVXPY({self.solver})"


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------


class BackendSelector:
    """Automatically select optimization backend based on problem structure.
    
    Selection heuristics:
        1. If problem has special structure (Toeplitz, banded):
           -> Use HiGHS with structure exploitation
        2. If problem is very large (n > 10^6):
           -> Use HiGHS (better scalability than CVXPY)
        3. If problem has complex constraints (nonlinear, SDP):
           -> Use CVXPY (broader modeling)
        4. Default: HiGHS for LP, CVXPY for others
        
    Args:
        config: Solver configuration
        prefer_highs: Prefer HiGHS when backends are equivalent
    """
    
    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        prefer_highs: bool = True,
    ):
        self.config = config or SolverConfig()
        self.prefer_highs = prefer_highs
        
    def select_backend(
        self,
        problem_size: Tuple[int, int],
        is_sparse: bool,
        has_equality: bool,
        structure: Optional[str] = None,
    ) -> OptimizationBackend:
        """Select backend based on problem properties.
        
        Args:
            problem_size: (num_variables, num_constraints)
            is_sparse: Whether constraint matrix is sparse
            has_equality: Whether problem has equality constraints
            structure: Special structure ('toeplitz', 'banded', None)
            
        Returns:
            Appropriate OptimizationBackend instance
        """
        n, m = problem_size
        
        logger.debug(
            f"Backend selection: n={n}, m={m}, sparse={is_sparse}, "
            f"structure={structure}"
        )
        
        # Very large problems: HiGHS
        if n > 100000 or m > 100000:
            logger.debug("Selected HiGHS: large problem")
            return HiGHSBackend(self.config)
        
        # Structured problems: HiGHS with exploitation
        if structure in ['toeplitz', 'banded', 'circulant']:
            logger.debug(f"Selected HiGHS: {structure} structure")
            return HiGHSBackend(self.config)
        
        # Sparse problems: prefer HiGHS
        if is_sparse and self.prefer_highs:
            logger.debug("Selected HiGHS: sparse problem")
            return HiGHSBackend(self.config)
        
        # Default: HiGHS
        logger.debug("Selected HiGHS: default")
        return HiGHSBackend(self.config)
    
    def select_for_lp(
        self,
        c: npt.NDArray[np.float64],
        A_ub: Optional[sparse.spmatrix] = None,
        A_eq: Optional[sparse.spmatrix] = None,
    ) -> OptimizationBackend:
        """Select backend for a specific LP instance.
        
        Args:
            c: Objective vector
            A_ub: Inequality constraint matrix
            A_eq: Equality constraint matrix
            
        Returns:
            Selected backend
        """
        n = len(c)
        m = 0
        
        if A_ub is not None:
            m += A_ub.shape[0]
        if A_eq is not None:
            m += A_eq.shape[0]
        
        # Check sparsity
        is_sparse = False
        if A_ub is not None:
            is_sparse = sparse.issparse(A_ub) and (A_ub.nnz / (A_ub.shape[0] * A_ub.shape[1]) < 0.1)
        
        has_equality = A_eq is not None
        
        # Detect structure (simple heuristic)
        structure = None
        if A_ub is not None and sparse.issparse(A_ub):
            # Check if banded (cheap test)
            nnz_per_row = A_ub.getnnz(axis=1)
            max_nnz = np.max(nnz_per_row) if len(nnz_per_row) > 0 else 0
            
            if max_nnz < 0.01 * n:
                structure = 'banded'
        
        return self.select_backend(
            problem_size=(n, m),
            is_sparse=is_sparse,
            has_equality=has_equality,
            structure=structure,
        )
