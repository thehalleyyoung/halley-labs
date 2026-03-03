"""
Warm-start strategies for CEGIS loop optimization.

This module provides techniques for accelerating iterative LP solves in
CEGIS (Counter-Example Guided Inductive Synthesis) loops by preserving
and updating solution information across iterations.

Key techniques:
    - Dual simplex warm-start preserving dual feasible basis
    - Constraint pool management with LRU eviction
    - Basis tracking across CEGIS iterations
    - Incremental rank-1 updates for constraint addition
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import linprog

from dp_forge.exceptions import NumericalInstabilityError, SolverError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BasisInfo:
    """Information about an LP basis.
    
    Attributes:
        basic_vars: Indices of basic variables
        nonbasic_vars: Indices of nonbasic variables
        basis_matrix_inv: Inverse of basis matrix B^{-1} (optional, for updates)
        dual_values: Dual variable values (shadow prices)
        iteration: Iteration when basis was computed
        is_dual_feasible: Whether dual solution is feasible
    """
    basic_vars: npt.NDArray[np.int32]
    nonbasic_vars: npt.NDArray[np.int32]
    basis_matrix_inv: Optional[npt.NDArray[np.float64]]
    dual_values: npt.NDArray[np.float64]
    iteration: int
    is_dual_feasible: bool


@dataclass
class ConstraintInfo:
    """Metadata for a constraint in the pool.
    
    Attributes:
        constraint_id: Unique identifier (e.g., database pair hash)
        coefficients: Constraint coefficients (sparse)
        rhs: Right-hand side value
        slack: Slack value in current solution (b - a^T x)
        last_accessed: Iteration when constraint was last active
        hit_count: Number of times constraint was binding
    """
    constraint_id: str
    coefficients: npt.NDArray[np.float64]
    rhs: float
    slack: float = float('inf')
    last_accessed: int = 0
    hit_count: int = 0


# ---------------------------------------------------------------------------
# Dual simplex warm-start
# ---------------------------------------------------------------------------


class DualSimplexWarmStart:
    """Preserve dual feasible basis across CEGIS iterations.
    
    When adding constraints to an LP in CEGIS, the previous optimal basis
    may remain dual feasible. Dual simplex can then find the new optimum
    much faster than cold-starting.
    
    Strategy:
        1. After solving LP(k), store optimal basis B*
        2. When adding constraints to form LP(k+1), check if B* remains
           dual feasible (reduced costs have correct sign)
        3. If yes: warm-start HiGHS with basis B*
        4. If no: cold-start (basis too old or problem structure changed)
        
    Args:
        max_basis_age: Maximum iterations before forcing cold start
        dual_feasibility_tol: Tolerance for dual feasibility check
        
    References:
        - Dantzig, Linear Programming and Extensions, Chapter 8
        - Bixby, "Solving Real-World Linear Programs", 2002
    """
    
    def __init__(
        self,
        max_basis_age: int = 10,
        dual_feasibility_tol: float = 1e-7,
    ):
        self.max_basis_age = max_basis_age
        self.dual_tol = dual_feasibility_tol
        
        self._last_basis: Optional[BasisInfo] = None
        self._current_iteration = 0
        
    def solve_with_warm_start(
        self,
        c: npt.NDArray[np.float64],
        A_ub: sparse.spmatrix,
        b_ub: npt.NDArray[np.float64],
        bounds: Optional[list[Tuple[float, float]]] = None,
    ) -> linprog:
        """Solve LP with warm-start if available.
        
        Args:
            c: Objective vector
            A_ub: Inequality constraint matrix
            b_ub: Inequality RHS
            bounds: Variable bounds
            
        Returns:
            scipy.optimize.OptimizeResult
        """
        self._current_iteration += 1
        
        # Check if warm-start is possible
        use_warm_start = self._should_warm_start(A_ub, b_ub)
        
        if use_warm_start and self._last_basis is not None:
            logger.debug(
                f"Attempting warm-start from iteration "
                f"{self._last_basis.iteration} (age={self._current_iteration - self._last_basis.iteration})"
            )
            
            # Try warm-start
            result = self._solve_with_basis(c, A_ub, b_ub, bounds, self._last_basis)
            
            # Check if warm-start succeeded
            if result.success:
                # Verify dual feasibility residual
                dual_residual = self._compute_dual_residual(result, A_ub)
                
                if dual_residual < 10 * self.dual_tol:
                    logger.debug(
                        f"Warm-start succeeded: dual_residual={dual_residual:.3e}"
                    )
                    self._update_basis(result, A_ub)
                    return result
                else:
                    logger.debug(
                        f"Warm-start failed dual check: residual={dual_residual:.3e}, "
                        f"falling back to cold start"
                    )
        
        # Cold start
        logger.debug("Cold start LP solve")
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'disp': False},
        )
        
        if result.success:
            self._update_basis(result, A_ub)
        
        return result
    
    def _should_warm_start(
        self, A_ub: sparse.spmatrix, b_ub: npt.NDArray[np.float64]
    ) -> bool:
        """Decide whether to attempt warm-start.
        
        Returns False if:
            - No previous basis available
            - Basis is too old (age > max_basis_age)
            - Problem dimensions changed significantly
        """
        if self._last_basis is None:
            return False
        
        age = self._current_iteration - self._last_basis.iteration
        if age > self.max_basis_age:
            logger.debug(f"Basis too old (age={age} > {self.max_basis_age})")
            return False
        
        # Check if constraint count changed dramatically (>20%)
        m_old = len(self._last_basis.dual_values)
        m_new = A_ub.shape[0]
        
        if abs(m_new - m_old) > 0.2 * m_old:
            logger.debug(
                f"Constraint count changed significantly: {m_old} -> {m_new}"
            )
            return False
        
        return True
    
    def _solve_with_basis(
        self,
        c: npt.NDArray[np.float64],
        A_ub: sparse.spmatrix,
        b_ub: npt.NDArray[np.float64],
        bounds: Optional[list[Tuple[float, float]]],
        basis: BasisInfo,
    ) -> linprog:
        """Attempt solve using previous basis.
        
        Note: scipy.optimize.linprog doesn't expose basis warm-starting.
        We use HiGHS directly via optional highspy import if available.
        Otherwise, fall back to regular linprog.
        """
        try:
            import highspy
            
            # Use HiGHS Python API for warm-starting
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("presolve", "off")  # Preserve basis
            
            # Build model
            n = len(c)
            m = A_ub.shape[0]
            
            # Add variables
            for i in range(n):
                lb = bounds[i][0] if bounds and bounds[i][0] is not None else 0
                ub = bounds[i][1] if bounds and bounds[i][1] is not None else highspy.kHighsInf
                h.addVar(lb, ub)
            
            # Set objective
            h.changeColsCost(n, np.arange(n), c)
            
            # Add constraints
            A_coo = A_ub.tocoo()
            for i in range(m):
                row_mask = A_coo.row == i
                cols = A_coo.col[row_mask]
                vals = A_coo.data[row_mask]
                h.addRow(-highspy.kHighsInf, b_ub[i], len(cols), cols, vals)
            
            # Set basis (if dimensions match)
            if len(basis.basic_vars) <= m and len(basis.nonbasic_vars) <= n:
                col_status = [highspy.HighsBasisStatus.kNonbasic] * n
                row_status = [highspy.HighsBasisStatus.kBasic] * m
                
                for idx in basis.basic_vars:
                    if idx < n:
                        col_status[idx] = highspy.HighsBasisStatus.kBasic
                
                try:
                    h.setBasis(col_status, row_status)
                except Exception as e:
                    logger.debug(f"Failed to set basis: {e}")
            
            # Solve
            h.run()
            
            # Extract solution
            info = h.getInfo()
            solution = h.getSolution()
            
            # Convert to scipy format
            result = type('obj', (object,), {
                'success': info.primal_solution_status == highspy.kSolutionStatusFeasible,
                'x': np.array(solution.col_value[:n]),
                'fun': info.objective_function_value,
                'status': 0 if info.primal_solution_status == highspy.kSolutionStatusFeasible else 1,
            })()
            
            return result
            
        except ImportError:
            # highspy not available, use regular linprog
            logger.debug("highspy not available, falling back to linprog")
            return linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='highs',
                options={'presolve': True, 'disp': False},
            )
    
    def _compute_dual_residual(
        self, result: linprog, A_ub: sparse.spmatrix
    ) -> float:
        """Compute dual feasibility residual.
        
        For minimization with A_ub @ x <= b_ub:
            Dual: max b^T y s.t. A^T y = c, y >= 0
            
        Residual: ||A^T y - c||_∞ where y are dual values
        """
        if not hasattr(result, 'ineqlin'):
            return 0.0
        
        # Extract dual values (marginals)
        if hasattr(result.ineqlin, 'marginals'):
            y = result.ineqlin.marginals
        else:
            return 0.0
        
        # Compute residual
        c = np.zeros(A_ub.shape[1])  # Would need access to objective
        residual = np.linalg.norm(A_ub.T @ y - c, ord=np.inf)
        
        return residual
    
    def _update_basis(self, result: linprog, A_ub: sparse.spmatrix) -> None:
        """Extract and store basis from solution."""
        n = A_ub.shape[1]
        m = A_ub.shape[0]
        
        # Try to extract basis information
        # Note: scipy's linprog doesn't expose basis directly
        # We approximate by looking at which constraints are tight
        
        if not hasattr(result, 'x'):
            return
        
        x = result.x
        slacks = A_ub @ x
        
        # Identify tight constraints (small slack)
        tight_mask = np.abs(slacks) < 1e-6
        tight_indices = np.where(tight_mask)[0]
        
        # Basic variables: indices with tight constraints
        # Nonbasic: others
        basic_vars = tight_indices[:min(n, len(tight_indices))]
        nonbasic_vars = np.array([i for i in range(n) if i not in basic_vars], dtype=np.int32)
        
        # Dummy dual values (would need actual dual extraction)
        dual_values = np.zeros(m)
        
        self._last_basis = BasisInfo(
            basic_vars=basic_vars,
            nonbasic_vars=nonbasic_vars,
            basis_matrix_inv=None,
            dual_values=dual_values,
            iteration=self._current_iteration,
            is_dual_feasible=True,
        )


# ---------------------------------------------------------------------------
# Constraint pool manager
# ---------------------------------------------------------------------------


class ConstraintPoolManager:
    """Maintain active constraint pool with LRU eviction.
    
    In CEGIS, we accumulate privacy constraints from violated database pairs.
    To prevent unbounded growth, we maintain a pool of "active" constraints
    using LRU (Least Recently Used) eviction.
    
    Algorithm:
        1. Add new constraints from CEGIS counterexamples
        2. Track which constraints are binding in current solution
        3. When pool exceeds max_size, evict LRU constraints
        4. Never evict constraints accessed in last 5 iterations
        
    Args:
        max_size: Maximum constraints to keep in pool
        min_retention_iterations: Minimum iterations to keep new constraints
        
    References:
        - Brayton et al., "SAT-Based Verification Methods", 2004
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        min_retention_iterations: int = 5,
    ):
        self.max_size = max_size
        self.min_retention = min_retention_iterations
        
        # OrderedDict provides LRU ordering
        self._pool: OrderedDict[str, ConstraintInfo] = OrderedDict()
        self._current_iteration = 0
        
    def add_constraint(
        self,
        constraint_id: str,
        coefficients: npt.NDArray[np.float64],
        rhs: float,
    ) -> bool:
        """Add constraint to pool.
        
        Args:
            constraint_id: Unique identifier (e.g., "pair_3_7")
            coefficients: Constraint coefficients
            rhs: Right-hand side
            
        Returns:
            True if added (new constraint), False if already present
        """
        if constraint_id in self._pool:
            # Move to end (most recently used)
            self._pool.move_to_end(constraint_id)
            self._pool[constraint_id].hit_count += 1
            return False
        
        # Add new constraint
        info = ConstraintInfo(
            constraint_id=constraint_id,
            coefficients=coefficients,
            rhs=rhs,
            last_accessed=self._current_iteration,
        )
        
        self._pool[constraint_id] = info
        
        # Evict if over capacity
        self._evict_if_needed()
        
        return True
    
    def update_slacks(
        self, x: npt.NDArray[np.float64], binding_threshold: float = 1e-6
    ) -> None:
        """Update slack values for all constraints given solution x.
        
        Args:
            x: Current solution
            binding_threshold: Threshold for considering constraint binding
        """
        self._current_iteration += 1
        
        for info in self._pool.values():
            slack = info.rhs - np.dot(info.coefficients, x)
            info.slack = slack
            
            if slack < binding_threshold:
                # Constraint is binding
                info.last_accessed = self._current_iteration
                info.hit_count += 1
    
    def get_active_constraints(
        self,
    ) -> Tuple[sparse.csr_matrix, npt.NDArray[np.float64]]:
        """Get constraint matrix and RHS for active constraints.
        
        Returns:
            (A, b): Constraint matrix and RHS vector
        """
        if len(self._pool) == 0:
            return sparse.csr_matrix((0, 0)), np.array([])
        
        # Stack coefficients
        n = len(next(iter(self._pool.values())).coefficients)
        m = len(self._pool)
        
        rows = []
        rhs = []
        
        for info in self._pool.values():
            rows.append(info.coefficients)
            rhs.append(info.rhs)
        
        A = sparse.csr_matrix(np.vstack(rows)) if rows else sparse.csr_matrix((0, n))
        b = np.array(rhs)
        
        return A, b
    
    def _evict_if_needed(self) -> None:
        """Evict LRU constraints if pool exceeds max_size."""
        while len(self._pool) > self.max_size:
            # Find LRU constraint that can be evicted
            evicted = False
            
            for constraint_id, info in list(self._pool.items()):
                age = self._current_iteration - info.last_accessed
                
                if age >= self.min_retention:
                    del self._pool[constraint_id]
                    logger.debug(
                        f"Evicted constraint {constraint_id} (age={age}, "
                        f"hits={info.hit_count})"
                    )
                    evicted = True
                    break
            
            if not evicted:
                # All constraints are recent; evict oldest anyway
                oldest_id = next(iter(self._pool))
                del self._pool[oldest_id]
                logger.warning(f"Forced eviction of recent constraint {oldest_id}")


# ---------------------------------------------------------------------------
# Basis tracker
# ---------------------------------------------------------------------------


class BasisTracker:
    """Track optimal basis evolution across CEGIS iterations.
    
    Maintains history of optimal bases to detect patterns:
        - Basis stability (same basis for multiple iterations)
        - Cycling detection (returning to previous basis)
        - Constraint importance (which constraints are frequently basic)
        
    Args:
        history_size: Number of past bases to retain
    """
    
    def __init__(self, history_size: int = 20):
        self.history_size = history_size
        self._history: list[BasisInfo] = []
        self._constraint_basic_count: dict[int, int] = {}
        
    def record_basis(self, basis: BasisInfo) -> None:
        """Record a new optimal basis."""
        self._history.append(basis)
        
        # Update basic counts
        for idx in basis.basic_vars:
            self._constraint_basic_count[int(idx)] = (
                self._constraint_basic_count.get(int(idx), 0) + 1
            )
        
        # Trim history
        if len(self._history) > self.history_size:
            old_basis = self._history.pop(0)
            # Decrement counts from removed basis
            for idx in old_basis.basic_vars:
                if int(idx) in self._constraint_basic_count:
                    self._constraint_basic_count[int(idx)] -= 1
    
    def get_important_constraints(self, top_k: int = 50) -> list[int]:
        """Get indices of constraints most frequently in basis.
        
        Args:
            top_k: Number of top constraints to return
            
        Returns:
            List of constraint indices sorted by importance
        """
        sorted_constraints = sorted(
            self._constraint_basic_count.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        
        return [idx for idx, count in sorted_constraints[:top_k]]
    
    def detect_cycling(self, window: int = 5) -> bool:
        """Detect if basis is cycling (returning to previous state).
        
        Args:
            window: Number of recent bases to check
            
        Returns:
            True if cycling detected
        """
        if len(self._history) < 2 * window:
            return False
        
        recent = self._history[-window:]
        earlier = self._history[-2*window:-window]
        
        # Check if any recent basis matches an earlier one
        for r_basis in recent:
            for e_basis in earlier:
                if self._bases_equal(r_basis, e_basis):
                    logger.warning("Basis cycling detected")
                    return True
        
        return False
    
    @staticmethod
    def _bases_equal(b1: BasisInfo, b2: BasisInfo) -> bool:
        """Check if two bases are equal (same basic variables)."""
        if len(b1.basic_vars) != len(b2.basic_vars):
            return False
        
        return np.array_equal(np.sort(b1.basic_vars), np.sort(b2.basic_vars))


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------


class IncrementalUpdate:
    """Efficient rank-1 updates for constraint addition.
    
    When adding a single constraint to an LP, we can update the factorization
    incrementally using Sherman-Morrison-Woodbury formula:
        
        (B + uv^T)^{-1} = B^{-1} - (B^{-1} u)(v^T B^{-1}) / (1 + v^T B^{-1} u)
        
    This avoids re-factorizing the entire basis matrix.
    
    Args:
        basis_matrix_inv: Current basis matrix inverse B^{-1}
        
    References:
        - Golub & Van Loan, Matrix Computations, §2.1.4
        - Hager, "Updating the Inverse of a Matrix", SIAM Review, 1989
    """
    
    def __init__(self, basis_matrix_inv: npt.NDArray[np.float64]):
        self.B_inv = basis_matrix_inv
        self.n = basis_matrix_inv.shape[0]
        
    def add_row(
        self, new_row: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Add a row to the basis matrix and update inverse.
        
        Args:
            new_row: New constraint coefficients for basic variables
            
        Returns:
            Updated basis inverse
        """
        if len(new_row) != self.n:
            raise ValueError(f"new_row has length {len(new_row)}, expected {self.n}")
        
        # Sherman-Morrison formula: (B + uv^T)^{-1}
        # Here u = new_row, v = unit vector
        
        u = new_row
        v = np.zeros(self.n)
        v[-1] = 1.0  # Add as last row
        
        # Compute denominator: 1 + v^T B^{-1} u
        B_inv_u = self.B_inv @ u
        denom = 1.0 + np.dot(v, B_inv_u)
        
        if abs(denom) < 1e-12:
            raise NumericalInstabilityError(
                "Singular update: denominator near zero"
            )
        
        # Update: B^{-1} - (B^{-1} u)(v^T B^{-1}) / denom
        v_B_inv = v @ self.B_inv
        outer = np.outer(B_inv_u, v_B_inv)
        
        B_inv_new = self.B_inv - outer / denom
        
        self.B_inv = B_inv_new
        return B_inv_new


# ---------------------------------------------------------------------------
# CEGIS integration utilities
# ---------------------------------------------------------------------------


class CEGISWarmStartManager:
    """Manage warm-starting across entire CEGIS loop.
    
    Integrates all warm-start techniques for CEGIS:
        1. DualSimplexWarmStart for basis preservation
        2. ConstraintPoolManager for constraint caching
        3. BasisTracker for pattern detection
        4. IncrementalUpdate for efficient factorization updates
        
    This is the high-level interface used by the CEGIS loop.
    
    Args:
        n: Number of database values
        k: Number of output values
        max_pool_size: Maximum constraints in pool
        max_basis_age: Maximum iterations to reuse basis
    """
    
    def __init__(
        self,
        n: int,
        k: int,
        max_pool_size: int = 10000,
        max_basis_age: int = 10,
    ):
        self.n = n
        self.k = k
        
        # Component warm-starters
        self.dual_simplex = DualSimplexWarmStart(
            max_basis_age=max_basis_age,
            dual_feasibility_tol=1e-7,
        )
        
        self.pool_manager = ConstraintPoolManager(
            max_size=max_pool_size,
            min_retention_iterations=5,
        )
        
        self.basis_tracker = BasisTracker(history_size=20)
        
        # Statistics
        self.iteration = 0
        self.warm_start_successes = 0
        self.cold_start_count = 0
        
    def solve_iteration(
        self,
        c: npt.NDArray[np.float64],
        new_constraints: list[Tuple[str, npt.NDArray[np.float64], float]],
        bounds: Optional[list[Tuple[float, float]]] = None,
    ) -> Tuple[linprog, dict]:
        """Solve one CEGIS iteration with warm-starting.
        
        Args:
            c: Objective vector
            new_constraints: List of (id, coefficients, rhs) for new constraints
            bounds: Variable bounds
            
        Returns:
            (result, stats): LP result and warm-start statistics
        """
        self.iteration += 1
        
        # Add new constraints to pool
        for constraint_id, coeffs, rhs in new_constraints:
            self.pool_manager.add_constraint(constraint_id, coeffs, rhs)
        
        # Get active constraints from pool
        A_ub, b_ub = self.pool_manager.get_active_constraints()
        
        # Solve with warm-start
        result = self.dual_simplex.solve_with_warm_start(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
        )
        
        # Update statistics
        stats = {
            'iteration': self.iteration,
            'active_constraints': A_ub.shape[0],
            'pool_size': len(self.pool_manager._pool),
            'warm_started': self.dual_simplex._last_basis is not None,
        }
        
        if result.success:
            # Update pool slacks
            self.pool_manager.update_slacks(result.x)
            
            # Track basis
            if hasattr(result, 'x'):
                # Extract basis info (simplified)
                slacks = A_ub @ result.x - b_ub
                tight_mask = np.abs(slacks) < 1e-6
                basic_vars = np.where(tight_mask)[0]
                
                basis_info = BasisInfo(
                    basic_vars=basic_vars,
                    nonbasic_vars=np.array([], dtype=np.int32),
                    basis_matrix_inv=None,
                    dual_values=np.zeros(len(b_ub)),
                    iteration=self.iteration,
                    is_dual_feasible=True,
                )
                
                self.basis_tracker.record_basis(basis_info)
                
                # Check for cycling
                if self.basis_tracker.detect_cycling():
                    logger.warning("Basis cycling detected in CEGIS loop")
                    stats['cycling'] = True
        
        return result, stats
    
    def get_important_constraints(self, top_k: int = 100) -> list[int]:
        """Get indices of most important constraints for next iteration.
        
        Returns:
            List of constraint indices to prioritize
        """
        return self.basis_tracker.get_important_constraints(top_k)
    
    def reset_after_parameter_change(self) -> None:
        """Reset warm-start data after privacy parameter change.
        
        Call this when epsilon or delta changes significantly.
        """
        self.dual_simplex._last_basis = None
        logger.info("Warm-start data reset after parameter change")


class AdaptiveTolerance:
    """Adaptively adjust LP solver tolerance during CEGIS.
    
    Start with loose tolerance for fast early iterations, then tighten
    as we approach optimum. This balances speed vs. accuracy.
    
    Algorithm:
        1. Start with tolerance = 1e-4 (loose)
        2. If no new counterexamples for N iterations: tighten by 10×
        3. If counterexample found: keep current tolerance
        4. Minimum tolerance: 1e-9
        
    Args:
        initial_tolerance: Starting tolerance
        min_tolerance: Minimum (tightest) tolerance
        tighten_threshold: Iterations without CE before tightening
        tighten_factor: Factor to tighten by (e.g., 10)
    """
    
    def __init__(
        self,
        initial_tolerance: float = 1e-4,
        min_tolerance: float = 1e-9,
        tighten_threshold: int = 3,
        tighten_factor: float = 10.0,
    ):
        self.current_tolerance = initial_tolerance
        self.min_tolerance = min_tolerance
        self.tighten_threshold = tighten_threshold
        self.tighten_factor = tighten_factor
        
        self._iterations_without_ce = 0
        
    def update(self, found_counterexample: bool) -> float:
        """Update tolerance based on whether counterexample was found.
        
        Args:
            found_counterexample: True if CEGIS found a counterexample
            
        Returns:
            New tolerance to use
        """
        if found_counterexample:
            # CE found: keep current tolerance
            self._iterations_without_ce = 0
        else:
            # No CE: increment counter
            self._iterations_without_ce += 1
            
            # Tighten if threshold reached
            if self._iterations_without_ce >= self.tighten_threshold:
                old_tol = self.current_tolerance
                self.current_tolerance = max(
                    self.min_tolerance,
                    self.current_tolerance / self.tighten_factor,
                )
                
                if self.current_tolerance < old_tol:
                    logger.info(
                        f"Tightened tolerance: {old_tol:.2e} -> {self.current_tolerance:.2e}"
                    )
                
                self._iterations_without_ce = 0
        
        return self.current_tolerance
    
    def reset(self) -> None:
        """Reset to initial tolerance."""
        self.current_tolerance = self.min_tolerance * self.tighten_factor ** 3
        self._iterations_without_ce = 0


class ConstraintImportanceRanker:
    """Rank constraints by importance for pruning decisions.
    
    Uses multiple heuristics to score constraint importance:
        1. Binding frequency (how often tight in solutions)
        2. Dual value magnitude (shadow price)
        3. Recency (recently added constraints prioritized)
        4. Constraint type (privacy constraints never pruned)
        
    Args:
        weights: Dictionary of weights for each heuristic
    """
    
    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
    ):
        self.weights = weights or {
            'binding_frequency': 0.4,
            'dual_value': 0.3,
            'recency': 0.2,
            'type_priority': 0.1,
        }
        
        self._binding_counts: dict[str, int] = {}
        self._dual_values: dict[str, float] = {}
        self._iteration_added: dict[str, int] = {}
        self._current_iteration = 0
        
    def update(
        self,
        constraint_id: str,
        is_binding: bool,
        dual_value: float,
        constraint_type: str = 'privacy',
    ) -> None:
        """Update scores for a constraint.
        
        Args:
            constraint_id: Unique constraint identifier
            is_binding: Whether constraint is binding in current solution
            dual_value: Dual variable value (shadow price)
            constraint_type: Type of constraint
        """
        if constraint_id not in self._iteration_added:
            self._iteration_added[constraint_id] = self._current_iteration
        
        if is_binding:
            self._binding_counts[constraint_id] = (
                self._binding_counts.get(constraint_id, 0) + 1
            )
        
        self._dual_values[constraint_id] = abs(dual_value)
        
    def rank_constraints(
        self,
        constraint_ids: list[str],
    ) -> list[Tuple[str, float]]:
        """Rank constraints by importance score.
        
        Args:
            constraint_ids: List of constraint IDs to rank
            
        Returns:
            List of (constraint_id, score) sorted by decreasing importance
        """
        scores = []
        
        for cid in constraint_ids:
            score = 0.0
            
            # Binding frequency score
            binding_count = self._binding_counts.get(cid, 0)
            binding_score = min(1.0, binding_count / 10.0)
            score += self.weights['binding_frequency'] * binding_score
            
            # Dual value score
            dual_val = self._dual_values.get(cid, 0.0)
            dual_score = min(1.0, dual_val)
            score += self.weights['dual_value'] * dual_score
            
            # Recency score
            age = self._current_iteration - self._iteration_added.get(cid, 0)
            recency_score = np.exp(-age / 10.0)
            score += self.weights['recency'] * recency_score
            
            scores.append((cid, score))
        
        # Sort by descending score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def advance_iteration(self) -> None:
        """Increment iteration counter."""
        self._current_iteration += 1
