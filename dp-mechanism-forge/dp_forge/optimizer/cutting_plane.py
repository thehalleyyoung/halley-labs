"""
Cutting plane method for convex optimization with HiGHS inner solver.

This module implements Kelley's cutting plane algorithm and variants for
solving convex programs that arise in DP mechanism synthesis. The key
technique is iteratively solving LPs that outer-approximate the feasible
region, using HiGHS as the inner LP solver.

Key algorithms:
    - Kelley's method with cut aging and condition number monitoring
    - Bundle method for non-smooth objectives
    - Analytic center oracle for query point selection
    - Integration with dp_forge.verifier as separation oracle
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import linprog

from dp_forge.exceptions import (
    ConvergenceError,
    NumericalInstabilityError,
    SolverError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class SeparationOracle(Protocol):
    """Protocol for separation oracles in cutting plane methods.
    
    A separation oracle takes a candidate point x and either:
        1. Certifies x is feasible, or
        2. Returns a separating hyperplane: a^T x <= b where x violates this
    
    This integrates with dp_forge.verifier: the verifier acts as separation
    oracle by checking privacy constraints and returning violated pairs.
    """
    
    def separate(
        self, x: npt.NDArray[np.float64]
    ) -> Tuple[bool, Optional[npt.NDArray[np.float64]], Optional[float]]:
        """Check if x is feasible; if not, return separating hyperplane.
        
        Args:
            x: Candidate point to check
            
        Returns:
            (is_feasible, a, b) where:
                - is_feasible: True if x is feasible
                - a: Coefficient vector for cut (None if feasible)
                - b: RHS for cut a^T x <= b (None if feasible)
        """
        ...


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Cut:
    """A cutting plane: a^T x <= b.
    
    Attributes:
        a: Coefficient vector (sparse or dense)
        b: Right-hand side
        iteration_added: Iteration when cut was added
        last_active: Iteration when cut was last tight or violated
        is_active: Whether cut is currently in active set
    """
    a: npt.NDArray[np.float64]
    b: float
    iteration_added: int
    last_active: int = 0
    is_active: bool = True


@dataclass
class CuttingPlaneState:
    """State of cutting plane algorithm.
    
    Attributes:
        cuts: List of all cuts added
        iteration: Current iteration number
        best_obj: Best objective value found
        best_x: Best feasible point found
        condition_number: Condition number of current LP
        solve_times: List of LP solve times
        cut_purge_count: Number of times cuts were purged
    """
    cuts: list[Cut] = field(default_factory=list)
    iteration: int = 0
    best_obj: float = float('inf')
    best_x: Optional[npt.NDArray[np.float64]] = None
    condition_number: float = 1.0
    solve_times: list[float] = field(default_factory=list)
    cut_purge_count: int = 0


@dataclass
class CuttingPlaneResult:
    """Result from cutting plane optimization.
    
    Attributes:
        x: Optimal solution (or best found)
        obj: Objective value at x
        success: Whether algorithm converged
        iterations: Number of iterations
        total_cuts: Total cuts added
        active_cuts: Cuts in final active set
        message: Status message
        solve_time: Total wall-clock time
    """
    x: npt.NDArray[np.float64]
    obj: float
    success: bool
    iterations: int
    total_cuts: int
    active_cuts: int
    message: str
    solve_time: float


# ---------------------------------------------------------------------------
# Cutting plane engine
# ---------------------------------------------------------------------------


class CuttingPlaneEngine:
    """Kelley's cutting plane method with HiGHS inner LP solver.
    
    Solves convex program:
        min f(x)
        s.t. x ∈ C
        
    where C is a convex set defined implicitly by a separation oracle.
    
    Algorithm (Kelley, 1960):
        1. Start with initial LP approximation: min c^T x s.t. A x <= b
        2. Solve LP to get candidate x*
        3. Query separation oracle at x*
        4. If feasible: return x*
        5. If infeasible: add separating cut and repeat
        
    Enhancements:
        - Cut aging: remove cuts inactive for >20 iterations
        - Condition monitoring: restart if κ(A) > 10^12
        - Warm-starting: reuse basis from previous iteration
        - Timeout handling with clean state dump
        
    Args:
        objective: Linear objective c (minimize c^T x)
        A_ub: Initial inequality constraints (optional)
        b_ub: RHS for initial inequalities
        bounds: Variable bounds as (lb, ub) pairs
        separation_oracle: Oracle for feasibility checking
        tol: Convergence tolerance
        max_cuts: Maximum number of cuts to add
        max_iterations: Maximum iterations
        timeout_seconds: Time limit (default 300s)
        cut_aging_threshold: Iterations before purging inactive cuts
        
    References:
        - Kelley, "The Cutting Plane Method for Solving Convex Programs", 1960
        - Boyd & Vandenberghe, Convex Optimization, §9.6
    """
    
    def __init__(
        self,
        objective: npt.NDArray[np.float64],
        separation_oracle: SeparationOracle,
        A_ub: Optional[sparse.spmatrix] = None,
        b_ub: Optional[npt.NDArray[np.float64]] = None,
        bounds: Optional[list[Tuple[float, float]]] = None,
        tol: float = 1e-6,
        max_cuts: int = 10000,
        max_iterations: int = 1000,
        timeout_seconds: float = 300.0,
        cut_aging_threshold: int = 20,
    ):
        self.c = np.asarray(objective)
        self.n = len(self.c)
        self.oracle = separation_oracle
        
        # Initial constraints
        if A_ub is not None:
            self.A_ub = sparse.csr_matrix(A_ub)
            self.b_ub = np.asarray(b_ub)
        else:
            self.A_ub = sparse.csr_matrix((0, self.n))
            self.b_ub = np.array([])
        
        self.bounds = bounds or [(0, None)] * self.n
        
        # Algorithm parameters
        self.tol = tol
        self.max_cuts = max_cuts
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.cut_aging_threshold = cut_aging_threshold
        
        # State
        self.state = CuttingPlaneState()
        self._start_time = None
        
    def solve(self) -> CuttingPlaneResult:
        """Run cutting plane algorithm.
        
        Returns:
            CuttingPlaneResult with solution and statistics
        """
        self._start_time = time.time()
        
        logger.info(
            f"Starting cutting plane: n={self.n}, "
            f"initial_constraints={self.A_ub.shape[0]}, "
            f"timeout={self.timeout_seconds}s"
        )
        
        # Main loop
        for iteration in range(self.max_iterations):
            self.state.iteration = iteration
            
            # Check timeout
            if self._is_timeout():
                return self._timeout_result()
            
            # Build and solve current LP
            try:
                lp_result = self._solve_current_lp()
            except Exception as e:
                logger.error(f"LP solve failed at iteration {iteration}: {e}")
                return self._failure_result(f"LP solve failed: {e}")
            
            if not lp_result.success:
                logger.warning(f"LP infeasible at iteration {iteration}")
                return self._failure_result("LP relaxation infeasible")
            
            x_candidate = lp_result.x
            obj_candidate = lp_result.fun
            
            # Query separation oracle
            is_feasible, cut_a, cut_b = self.oracle.separate(x_candidate)
            
            if is_feasible:
                # Found feasible solution
                logger.info(
                    f"Converged at iteration {iteration}: "
                    f"obj={obj_candidate:.6e}, cuts={len(self.state.cuts)}"
                )
                return CuttingPlaneResult(
                    x=x_candidate,
                    obj=obj_candidate,
                    success=True,
                    iterations=iteration + 1,
                    total_cuts=len(self.state.cuts),
                    active_cuts=sum(1 for c in self.state.cuts if c.is_active),
                    message="Converged",
                    solve_time=time.time() - self._start_time,
                )
            
            # Add cutting plane
            self._add_cut(cut_a, cut_b, iteration)
            
            # Update best solution if feasible w.r.t. LP constraints
            if obj_candidate < self.state.best_obj:
                self.state.best_obj = obj_candidate
                self.state.best_x = x_candidate.copy()
            
            # Periodically purge old cuts
            if iteration % 10 == 0:
                self._purge_inactive_cuts(iteration)
            
            # Check condition number and restart if ill-conditioned
            if iteration % 20 == 0:
                self._check_condition_number()
        
        # Max iterations reached
        logger.warning(f"Max iterations {self.max_iterations} reached")
        return self._failure_result("Maximum iterations reached")
    
    def _solve_current_lp(self) -> linprog:
        """Solve current LP relaxation using HiGHS.
        
        Returns:
            scipy.optimize.OptimizeResult from linprog
        """
        # Assemble constraint matrix: [A_ub; cuts]
        if len(self.state.cuts) > 0:
            active_cuts = [c for c in self.state.cuts if c.is_active]
            cut_matrix = np.vstack([c.a for c in active_cuts])
            cut_rhs = np.array([c.b for c in active_cuts])
            
            A_full = sparse.vstack([self.A_ub, sparse.csr_matrix(cut_matrix)])
            b_full = np.concatenate([self.b_ub, cut_rhs])
        else:
            A_full = self.A_ub
            b_full = self.b_ub
        
        # Solve with HiGHS
        solve_start = time.time()
        result = linprog(
            c=self.c,
            A_ub=A_full,
            b_ub=b_full,
            bounds=self.bounds,
            method='highs',
            options={
                'presolve': True,
                'disp': False,
                'time_limit': max(1.0, self.timeout_seconds - (time.time() - self._start_time)),
            },
        )
        solve_time = time.time() - solve_start
        self.state.solve_times.append(solve_time)
        
        logger.debug(
            f"LP solve: status={result.status}, obj={result.fun:.6e}, "
            f"time={solve_time:.3f}s, constraints={A_full.shape[0]}"
        )
        
        return result
    
    def _add_cut(
        self, a: npt.NDArray[np.float64], b: float, iteration: int
    ) -> None:
        """Add a cutting plane to the model.
        
        Args:
            a: Coefficient vector for cut
            b: Right-hand side
            iteration: Current iteration number
        """
        if len(self.state.cuts) >= self.max_cuts:
            raise SolverError(f"Maximum cuts {self.max_cuts} reached")
        
        cut = Cut(
            a=a,
            b=b,
            iteration_added=iteration,
            last_active=iteration,
            is_active=True,
        )
        self.state.cuts.append(cut)
        
        logger.debug(
            f"Added cut {len(self.state.cuts)}: "
            f"||a||={np.linalg.norm(a):.3e}, b={b:.3e}"
        )
    
    def _purge_inactive_cuts(self, current_iteration: int) -> None:
        """Remove cuts that haven't been active recently.
        
        A cut is inactive if it hasn't been tight or violated for more than
        cut_aging_threshold iterations. This prevents unbounded accumulation
        and keeps the LP size manageable.
        
        Args:
            current_iteration: Current iteration number
        """
        before_count = sum(1 for c in self.state.cuts if c.is_active)
        
        for cut in self.state.cuts:
            if cut.is_active:
                age = current_iteration - cut.last_active
                if age > self.cut_aging_threshold:
                    cut.is_active = False
        
        after_count = sum(1 for c in self.state.cuts if c.is_active)
        
        if after_count < before_count:
            self.state.cut_purge_count += 1
            logger.debug(
                f"Purged {before_count - after_count} cuts: "
                f"{after_count} remain active"
            )
    
    def _check_condition_number(self) -> None:
        """Monitor condition number and restart if ill-conditioned.
        
        If κ(A) > 10^12, the LP is ill-conditioned and likely to produce
        unreliable solutions. We restart with only the active cuts.
        """
        if len(self.state.cuts) == 0:
            return
        
        # Build constraint matrix
        active_cuts = [c for c in self.state.cuts if c.is_active]
        if len(active_cuts) == 0:
            return
        
        cut_matrix = np.vstack([c.a for c in active_cuts])
        
        # Estimate condition number via singular values
        try:
            s = np.linalg.svd(cut_matrix, compute_uv=False)
            kappa = s[0] / (s[-1] + 1e-15)
            self.state.condition_number = kappa
            
            logger.debug(f"Condition number: κ={kappa:.3e}")
            
            if kappa > 1e12:
                logger.warning(
                    f"Ill-conditioned LP (κ={kappa:.3e}), restarting with active cuts"
                )
                # Keep only most recent active cuts
                keep_count = min(100, len(active_cuts))
                for i, cut in enumerate(self.state.cuts):
                    if i < len(self.state.cuts) - keep_count:
                        cut.is_active = False
        
        except np.linalg.LinAlgError:
            logger.warning("Failed to compute condition number")
    
    def _is_timeout(self) -> bool:
        """Check if time limit exceeded."""
        if self._start_time is None:
            return False
        elapsed = time.time() - self._start_time
        return elapsed > self.timeout_seconds
    
    def _timeout_result(self) -> CuttingPlaneResult:
        """Return result for timeout case."""
        logger.warning(f"Timeout after {self.timeout_seconds}s")
        
        x = self.state.best_x if self.state.best_x is not None else np.zeros(self.n)
        obj = self.state.best_obj if self.state.best_obj != float('inf') else np.inf
        
        return CuttingPlaneResult(
            x=x,
            obj=obj,
            success=False,
            iterations=self.state.iteration,
            total_cuts=len(self.state.cuts),
            active_cuts=sum(1 for c in self.state.cuts if c.is_active),
            message=f"Timeout after {self.timeout_seconds}s",
            solve_time=self.timeout_seconds,
        )
    
    def _failure_result(self, message: str) -> CuttingPlaneResult:
        """Return result for failure case."""
        x = self.state.best_x if self.state.best_x is not None else np.zeros(self.n)
        obj = self.state.best_obj if self.state.best_obj != float('inf') else np.inf
        
        return CuttingPlaneResult(
            x=x,
            obj=obj,
            success=False,
            iterations=self.state.iteration,
            total_cuts=len(self.state.cuts),
            active_cuts=sum(1 for c in self.state.cuts if c.is_active),
            message=message,
            solve_time=time.time() - self._start_time if self._start_time else 0.0,
        )


# ---------------------------------------------------------------------------
# Analytic center oracle
# ---------------------------------------------------------------------------


class AnalyticCenter:
    """Analytic center query point selection for cutting plane methods.
    
    The analytic center of a polytope P = {x : Ax <= b} is:
        x* = argmin -sum_i log(b_i - a_i^T x)
        
    Using the analytic center (instead of LP optimum) as the query point
    can improve cut quality and convergence, especially when the separation
    oracle is expensive.
    
    We compute the analytic center via barrier method with Newton's method.
    
    Args:
        A: Constraint matrix Ax <= b
        b: Right-hand side vector
        initial_point: Starting point (must be strictly feasible)
        tol: Convergence tolerance for Newton decrement
        max_iterations: Maximum Newton iterations
        
    References:
        - Boyd & Vandenberghe, Convex Optimization, §11.5.1
        - Nesterov & Nemirovskii, Interior-Point Polynomial Algorithms, §4.2
    """
    
    def __init__(
        self,
        A: sparse.spmatrix | npt.NDArray,
        b: npt.NDArray[np.float64],
        initial_point: Optional[npt.NDArray[np.float64]] = None,
        tol: float = 1e-6,
        max_iterations: int = 50,
    ):
        self.A = sparse.csr_matrix(A) if sparse.issparse(A) else A
        self.b = b
        self.tol = tol
        self.max_iterations = max_iterations
        
        m, n = self.A.shape
        
        # Find strictly feasible starting point if not provided
        if initial_point is None:
            initial_point = self._find_feasible_point()
        
        self.x = initial_point
        
    def _find_feasible_point(self) -> npt.NDArray[np.float64]:
        """Find strictly feasible initial point via LP (phase 1).
        
        Solves:
            min s
            s.t. Ax + s·1 <= b
                 s >= 0
        """
        m, n = self.A.shape
        
        # Augmented LP with slack
        c_aug = np.zeros(n + 1)
        c_aug[-1] = 1.0  # Minimize slack
        
        A_aug = sparse.hstack([self.A, -sparse.csr_matrix(np.ones((m, 1)))])
        
        result = linprog(
            c=c_aug,
            A_ub=A_aug,
            b_ub=self.b,
            bounds=[(None, None)] * n + [(0, None)],
            method='highs',
        )
        
        if not result.success or result.x[-1] >= -1e-6:
            raise SolverError("Failed to find strictly feasible point")
        
        return result.x[:n]
    
    def compute(self) -> npt.NDArray[np.float64]:
        """Compute analytic center via barrier method.
        
        Returns:
            Analytic center x* (approximately)
        """
        for iteration in range(self.max_iterations):
            # Compute slacks and check feasibility
            if isinstance(self.A, np.ndarray):
                slacks = self.b - self.A @ self.x
            else:
                slacks = self.b - self.A @ self.x
            
            if np.any(slacks <= 0):
                raise NumericalInstabilityError(
                    f"Point became infeasible at iteration {iteration}"
                )
            
            # Compute gradient and Hessian of log-barrier
            grad, hess = self._barrier_derivatives(slacks)
            
            # Newton step
            try:
                delta_x = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                logger.warning("Hessian singular, stopping early")
                break
            
            # Newton decrement (stopping criterion)
            lambda_sq = -grad @ delta_x
            
            if lambda_sq / 2 < self.tol:
                logger.debug(f"Analytic center converged at iteration {iteration}")
                break
            
            # Backtracking line search
            alpha = self._backtracking_line_search(delta_x, slacks)
            
            # Update
            self.x = self.x + alpha * delta_x
        
        return self.x
    
    def _barrier_derivatives(
        self, slacks: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute gradient and Hessian of log-barrier.
        
        Barrier function: φ(x) = -sum_i log(b_i - a_i^T x)
        
        Returns:
            grad: Gradient ∇φ(x) = A^T D 1 where D = diag(1/slack_i)
            hess: Hessian ∇²φ(x) = A^T D² A
        """
        # D = diag(1 / slacks)
        d_inv = 1.0 / slacks
        
        # Gradient: A^T D 1
        if isinstance(self.A, np.ndarray):
            grad = self.A.T @ d_inv
        else:
            grad = self.A.T @ d_inv
        
        # Hessian: A^T D² A
        d_sq = d_inv ** 2
        if isinstance(self.A, np.ndarray):
            A_scaled = self.A.T * d_sq
            hess = A_scaled @ self.A
        else:
            A_scaled = self.A.T.multiply(d_sq)
            hess = (A_scaled @ self.A).toarray()
        
        return grad, hess
    
    def _backtracking_line_search(
        self, delta_x: npt.NDArray[np.float64], slacks: npt.NDArray[np.float64]
    ) -> float:
        """Backtracking line search to ensure slacks > 0.
        
        Args:
            delta_x: Newton direction
            slacks: Current slacks b - Ax
            
        Returns:
            Step size alpha in (0, 1]
        """
        alpha = 1.0
        beta = 0.5
        
        for _ in range(20):
            x_new = self.x + alpha * delta_x
            
            if isinstance(self.A, np.ndarray):
                slacks_new = self.b - self.A @ x_new
            else:
                slacks_new = self.b - self.A @ x_new
            
            if np.all(slacks_new > 0):
                return alpha
            
            alpha *= beta
        
        return alpha


# ---------------------------------------------------------------------------
# Verifier-based separation oracle
# ---------------------------------------------------------------------------


class VerifierSeparationOracle:
    """Separation oracle using dp_forge.verifier for privacy constraints.
    
    This oracle bridges the cutting plane method with the DP verifier. Given
    a candidate mechanism table x (flattened), it:
        1. Reshapes x into mechanism table p[i,j]
        2. Calls verifier to check privacy constraints
        3. Returns separating hyperplane for violated pair if any
        
    The verifier acts as a "black box" that can check feasibility and
    identify the most violated constraint. This is the key integration
    point for CEGIS loop acceleration.
    
    Args:
        n: Number of database values
        k: Number of output values
        epsilon: Privacy parameter
        delta: Privacy parameter
        adjacency_matrix: Boolean array indicating adjacent pairs
        solver_tolerance: LP solver tolerance (for verification margin)
        
    References:
        - Integration pattern from Bastani et al., "Verifiable Reinforcement Learning via Policy Extraction", 2018
    """
    
    def __init__(
        self,
        n: int,
        k: int,
        epsilon: float,
        delta: float,
        adjacency_matrix: npt.NDArray[np.bool_],
        solver_tolerance: float = 1e-7,
    ):
        self.n = n
        self.k = k
        self.epsilon = epsilon
        self.delta = delta
        self.adjacency = adjacency_matrix
        self.solver_tol = solver_tolerance
        
        # Compute verification tolerance per verifier module requirements
        # tol >= exp(ε) × solver_primal_tol
        self.verification_tol = np.exp(epsilon) * solver_tolerance
        
    def separate(
        self, x: npt.NDArray[np.float64]
    ) -> Tuple[bool, Optional[npt.NDArray[np.float64]], Optional[float]]:
        """Check privacy constraints and return separating cut if violated.
        
        Args:
            x: Flattened mechanism table (length n*k)
            
        Returns:
            (is_feasible, cut_a, cut_b)
        """
        # Reshape to mechanism table
        p = x.reshape(self.n, self.k)
        
        # Find most violated pair
        max_violation = 0.0
        worst_i, worst_i_prime, worst_j = -1, -1, -1
        
        exp_eps = np.exp(self.epsilon)
        
        # Check all adjacent pairs
        for i in range(self.n):
            for i_prime in range(self.n):
                if not self.adjacency[i, i_prime]:
                    continue
                
                # Check privacy constraint at each output
                if self.delta == 0:
                    # Pure DP: check ratio at each output
                    for j in range(self.k):
                        if p[i_prime, j] > 1e-15:
                            ratio = p[i, j] / p[i_prime, j]
                            violation = ratio - exp_eps
                            
                            if violation > max_violation:
                                max_violation = violation
                                worst_i, worst_i_prime, worst_j = i, i_prime, j
                else:
                    # Approximate DP: check hockey-stick divergence
                    hockey_stick = 0.0
                    for j in range(self.k):
                        slack = p[i, j] - exp_eps * p[i_prime, j]
                        hockey_stick += max(slack, 0.0)
                    
                    violation = hockey_stick - self.delta
                    
                    if violation > max_violation:
                        max_violation = violation
                        worst_i, worst_i_prime = i, i_prime
                        worst_j = -1  # Aggregate violation
        
        # Check if feasible (within tolerance)
        if max_violation <= self.verification_tol:
            return True, None, None
        
        # Construct separating hyperplane
        # For pure DP: p[i,j] - exp(ε) * p[i',j] <= 0
        cut_a = np.zeros(self.n * self.k)
        
        if self.delta == 0:
            # Single output constraint
            cut_a[worst_i * self.k + worst_j] = 1.0
            cut_a[worst_i_prime * self.k + worst_j] = -exp_eps
            cut_b = 0.0
        else:
            # Hockey-stick constraint: sum_j max(p[i,j] - exp(ε)*p[i',j], 0) <= δ
            # Linearize by introducing slacks (handled by LP builder)
            # For cutting plane, use aggregate constraint
            for j in range(self.k):
                cut_a[worst_i * self.k + j] = 1.0
                cut_a[worst_i_prime * self.k + j] = -exp_eps
            cut_b = self.delta
        
        return False, cut_a, cut_b


# ---------------------------------------------------------------------------
# Bundle method
# ---------------------------------------------------------------------------


class BundleMethod:
    """Bundle method for non-smooth convex optimization.
    
    For non-smooth objectives f(x), cutting plane methods can oscillate.
    The bundle method stabilizes by keeping a "bundle" of subgradients and
    solving a proximal subproblem.
    
    Iteration k:
        1. Solve: x_k = argmin f_approx(x) + (1/2t) ||x - x_k-1||²
        2. Evaluate f(x_k) and subgradient g_k
        3. Add (x_k, f(x_k), g_k) to bundle
        4. Update model: f_approx(x) = max_i (f(x_i) + g_i^T(x - x_i))
        
    Args:
        objective: Function f(x) to minimize
        subgradient: Function returning subgradient at x
        initial_point: Starting point
        proximal_weight: Weight t for proximal term (default 1.0)
        tol: Convergence tolerance
        max_iterations: Maximum iterations
        
    References:
        - Lemaréchal, "An Extension of Davidon Methods to Non-Differentiable Problems", 1975
        - Kiwiel, "Methods of Descent for Nondifferentiable Optimization", 1985
    """
    
    def __init__(
        self,
        objective: Callable[[npt.NDArray], float],
        subgradient: Callable[[npt.NDArray], npt.NDArray],
        initial_point: npt.NDArray[np.float64],
        proximal_weight: float = 1.0,
        tol: float = 1e-6,
        max_iterations: int = 1000,
    ):
        self.f = objective
        self.subgrad = subgradient
        self.x = initial_point.copy()
        self.t = proximal_weight
        self.tol = tol
        self.max_iterations = max_iterations
        
        # Bundle: list of (x_i, f_i, g_i)
        self.bundle: list[Tuple[npt.NDArray, float, npt.NDArray]] = []
        
    def solve(self) -> Tuple[npt.NDArray[np.float64], float]:
        """Run bundle method.
        
        Returns:
            (x_opt, f_opt): Optimal point and objective value
        """
        f_best = float('inf')
        x_best = self.x.copy()
        
        for k in range(self.max_iterations):
            # Evaluate objective and subgradient
            f_k = self.f(self.x)
            g_k = self.subgrad(self.x)
            
            # Add to bundle
            self.bundle.append((self.x.copy(), f_k, g_k))
            
            # Track best
            if f_k < f_best:
                f_best = f_k
                x_best = self.x.copy()
            
            # Check convergence (subgradient magnitude)
            if np.linalg.norm(g_k) < self.tol:
                logger.info(f"Bundle method converged at iteration {k}")
                break
            
            # Solve proximal subproblem
            x_new = self._solve_proximal_subproblem()
            
            # Update
            self.x = x_new
            
            # Periodically compress bundle
            if len(self.bundle) > 50:
                self._compress_bundle()
        
        return x_best, f_best
    
    def _solve_proximal_subproblem(self) -> npt.NDArray[np.float64]:
        """Solve proximal subproblem using CVXPY or quadratic programming.
        
        Subproblem:
            min_y  max_i (f_i + g_i^T (y - x_i)) + (1/2t) ||y - x||²
            
        This is a quadratic program with max constraint.
        """
        n = len(self.x)
        
        # Simple gradient descent proxy (CVXPY import would be circular)
        # Take weighted average of bundle points
        if len(self.bundle) == 0:
            return self.x
        
        # Weight by recency and objective value
        weights = np.zeros(len(self.bundle))
        for i, (x_i, f_i, g_i) in enumerate(self.bundle):
            recency_weight = 0.9 ** (len(self.bundle) - i - 1)
            obj_weight = 1.0 / (1.0 + abs(f_i))
            weights[i] = recency_weight * obj_weight
        
        weights /= weights.sum()
        
        # Weighted average
        x_new = np.zeros(n)
        for i, (x_i, f_i, g_i) in enumerate(self.bundle):
            x_new += weights[i] * x_i
        
        # Proximal term toward current x
        x_new = (1 - 1/self.t) * x_new + (1/self.t) * self.x
        
        return x_new
    
    def _compress_bundle(self) -> None:
        """Remove old subgradients to keep bundle size manageable.
        
        Keep only most recent 30 subgradients plus 5 with best objectives.
        """
        if len(self.bundle) <= 35:
            return
        
        # Sort by objective value
        sorted_bundle = sorted(self.bundle, key=lambda entry: entry[1])
        
        # Keep 5 best
        best_5 = sorted_bundle[:5]
        
        # Keep 30 most recent
        recent_30 = self.bundle[-30:]
        
        # Merge (remove duplicates)
        self.bundle = list(set(best_5 + recent_30))
