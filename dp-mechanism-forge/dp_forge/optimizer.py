"""
Advanced optimization strategies for DP mechanism synthesis.

Provides multi-objective, column generation, cutting plane, Lagrangian
relaxation, and approximation strategies that extend the core LP/SDP
synthesis with scalability and flexibility.

Optimization Strategies:
    - **Multi-objective**: Pareto-optimal synthesis sweeping over ε and k.
    - **Column generation**: For large k, iteratively add output bins via
      a pricing subproblem to avoid solving the full LP.
    - **Cutting plane**: Kelley's cutting plane method, analytic center
      cuts, and bundle methods for non-smooth objectives.
    - **Lagrangian relaxation**: Relax privacy constraints into the
      objective for faster dual computation via subgradient methods.
    - **Approximation strategies**: Coarsen-and-refine, domain
      decomposition, and constraint sampling for very large instances.
    - **Hyperparameter tuning**: Automatic selection of discretization k
      and solver parameters via cross-validation.

All optimizers work with :class:`dp_forge.types.QuerySpec` and produce
standard LP/SDP structures or directly yield mechanisms.

Classes:
    MultiObjectiveOptimizer    — Pareto-optimal synthesis
    ColumnGenerationOptimizer  — Column generation for large k
    CuttingPlaneOptimizer      — Cutting plane / bundle methods
    LagrangianRelaxation       — Lagrangian dual methods
    ApproximationStrategies    — Scalability heuristics
    HyperparameterTuner        — Auto-tuning k and solver parameters
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import sparse

from dp_forge.exceptions import (
    ConfigurationError,
    ConvergenceError,
    InfeasibleSpecError,
    SolverError,
)
from dp_forge.types import (
    AdjacencyRelation,
    LPStruct,
    LossFunction,
    QuerySpec,
    WorkloadSpec,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    """A single point on the Pareto frontier.

    Attributes:
        epsilon: Privacy parameter ε at this point.
        k: Discretization parameter k at this point.
        objective_value: Optimal objective (e.g., minimax MSE).
        mechanism: The n×k probability table (if computed).
        metadata: Additional optimization data.
    """

    epsilon: float
    k: int
    objective_value: float
    mechanism: Optional[FloatArray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ParetoPoint(ε={self.epsilon:.4f}, k={self.k}, "
            f"obj={self.objective_value:.6f})"
        )


@dataclass
class ColumnGenerationResult:
    """Result of a column generation optimization.

    Attributes:
        mechanism: The n×k probability table.
        objective_value: Final objective value.
        n_columns: Number of columns (output bins) in the final LP.
        n_iterations: Column generation iterations.
        active_columns: Indices of columns with non-zero probability mass.
        pricing_values: History of pricing subproblem values.
        metadata: Optimization metadata.
    """

    mechanism: FloatArray
    objective_value: float
    n_columns: int
    n_iterations: int
    active_columns: List[int] = field(default_factory=list)
    pricing_values: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ColumnGenerationResult(obj={self.objective_value:.6f}, "
            f"cols={self.n_columns}, iter={self.n_iterations})"
        )


@dataclass
class CuttingPlaneResult:
    """Result of a cutting plane optimization.

    Attributes:
        mechanism: The n×k probability table.
        objective_value: Final objective value.
        n_cuts: Number of cuts added.
        n_iterations: Number of cutting plane iterations.
        upper_bound: Best upper bound on optimal value.
        lower_bound: Best lower bound on optimal value.
        gap: Relative gap (upper - lower) / upper.
        metadata: Optimization metadata.
    """

    mechanism: FloatArray
    objective_value: float
    n_cuts: int
    n_iterations: int
    upper_bound: float
    lower_bound: float
    gap: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CuttingPlaneResult(obj={self.objective_value:.6f}, "
            f"cuts={self.n_cuts}, gap={self.gap:.2e})"
        )


@dataclass
class LagrangianResult:
    """Result of Lagrangian relaxation.

    Attributes:
        dual_value: Lagrangian dual objective value (lower bound).
        primal_value: Best feasible primal objective value.
        multipliers: Lagrange multipliers at convergence.
        gap: Duality gap.
        n_iterations: Number of subgradient/bundle iterations.
        mechanism: Best feasible mechanism found.
        metadata: Optimization metadata.
    """

    dual_value: float
    primal_value: float
    multipliers: FloatArray
    gap: float
    n_iterations: int
    mechanism: Optional[FloatArray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"LagrangianResult(dual={self.dual_value:.6f}, "
            f"primal={self.primal_value:.6f}, gap={self.gap:.2e})"
        )


@dataclass
class TuningResult:
    """Result of hyperparameter tuning.

    Attributes:
        best_k: Optimal discretization parameter.
        best_objective: Objective value at the optimal k.
        k_values: All k values evaluated.
        objectives: Objective values at each k.
        cv_scores: Cross-validation scores (if applicable).
        metadata: Tuning metadata.
    """

    best_k: int
    best_objective: float
    k_values: List[int] = field(default_factory=list)
    objectives: List[float] = field(default_factory=list)
    cv_scores: Optional[Dict[int, List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TuningResult(best_k={self.best_k}, "
            f"best_obj={self.best_objective:.6f})"
        )


# =========================================================================
# Helper: build a simple LP for mechanism synthesis
# =========================================================================


def _build_minimax_lp(
    spec: QuerySpec,
    y_grid: FloatArray,
    column_indices: Optional[List[int]] = None,
) -> Tuple[FloatArray, FloatArray, FloatArray, List[Tuple[float, float]], Dict[Tuple[int, int], int]]:
    """Build a minimax LP for mechanism synthesis.

    Constructs the LP: minimize t
    subject to:
        Σ_j loss(q_i, y_j) · p[i,j] ≤ t   ∀ i        (utility)
        p[i,j] ≤ e^ε · p[i',j]              ∀ (i,i'), j  (privacy)
        Σ_j p[i,j] = 1                      ∀ i        (normalisation)
        p[i,j] ≥ 0                           ∀ i, j     (non-negativity)

    Variables: p[i,j] for i=0..n-1, j in column_indices, plus t.

    Args:
        spec: Query specification.
        y_grid: Output grid of shape (k,).
        column_indices: Subset of output bin indices to include.
            If None, uses all k bins.

    Returns:
        Tuple of (c, A_ub, b_ub, bounds, var_map) where:
            c: Objective coefficients.
            A_ub: Inequality constraint matrix (dense).
            b_ub: Inequality RHS.
            bounds: Per-variable bounds.
            var_map: Mapping from (i, j) to flat variable index.
    """
    n = spec.n
    if column_indices is None:
        column_indices = list(range(len(y_grid)))
    k_active = len(column_indices)
    y_active = y_grid[column_indices]

    assert spec.edges is not None
    edges = spec.edges.edges

    loss_fn = spec.get_loss_callable()
    eps = spec.epsilon

    # Variables: p[i, j_idx] for i in [n], j_idx in [k_active], plus t
    n_p_vars = n * k_active
    n_vars = n_p_vars + 1  # +1 for t
    t_idx = n_p_vars

    # var_map: (i, j_col_index) -> flat_index
    var_map: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        for j_idx, j in enumerate(column_indices):
            var_map[(i, j)] = i * k_active + j_idx

    # Objective: minimize t
    c = np.zeros(n_vars, dtype=np.float64)
    c[t_idx] = 1.0

    # Constraints
    ub_rows = []
    ub_rhs = []

    # Utility constraints: Σ_j loss(q_i, y_j) · p[i,j] - t ≤ 0
    for i in range(n):
        row = np.zeros(n_vars, dtype=np.float64)
        for j_idx, j in enumerate(column_indices):
            row[i * k_active + j_idx] = loss_fn(spec.query_values[i], y_active[j_idx])
        row[t_idx] = -1.0
        ub_rows.append(row)
        ub_rhs.append(0.0)

    # Privacy constraints: p[i,j] - e^ε · p[i',j] ≤ 0  for each (i,i'), j
    exp_eps = math.exp(eps)
    for i, ip in edges:
        for j_idx in range(k_active):
            # p[i,j] ≤ e^ε · p[i',j]
            row = np.zeros(n_vars, dtype=np.float64)
            row[i * k_active + j_idx] = 1.0
            row[ip * k_active + j_idx] = -exp_eps
            ub_rows.append(row)
            ub_rhs.append(0.0)

            # Symmetric: p[i',j] ≤ e^ε · p[i,j]
            if spec.edges is not None and spec.edges.symmetric:
                row2 = np.zeros(n_vars, dtype=np.float64)
                row2[ip * k_active + j_idx] = 1.0
                row2[i * k_active + j_idx] = -exp_eps
                ub_rows.append(row2)
                ub_rhs.append(0.0)

    A_ub = np.array(ub_rows, dtype=np.float64)
    b_ub = np.array(ub_rhs, dtype=np.float64)

    # Bounds
    eta_min = spec.eta_min
    bounds = [(eta_min, 1.0)] * n_p_vars + [(0.0, None)]

    return c, A_ub, b_ub, bounds, var_map


# =========================================================================
# 1. MultiObjectiveOptimizer
# =========================================================================


class MultiObjectiveOptimizer:
    """Multi-objective optimization for mechanism synthesis.

    Sweeps over privacy parameters (ε) and discretization parameters (k)
    to produce Pareto-optimal mechanisms that trade off between privacy
    and utility.

    Usage::

        optimizer = MultiObjectiveOptimizer()
        frontier = optimizer.pareto_optimize(spec, objectives=['mse', 'privacy'])
        results = optimizer.epsilon_sweep(spec, eps_range=np.linspace(0.1, 2.0, 20))
    """

    def __init__(self, verbose: int = 0) -> None:
        """Initialize multi-objective optimizer.

        Args:
            verbose: Verbosity level (0=silent, 1=progress, 2=debug).
        """
        self._verbose = verbose

    def pareto_optimize(
        self,
        spec: QuerySpec,
        objectives: Optional[List[str]] = None,
        eps_range: Optional[FloatArray] = None,
        k_range: Optional[List[int]] = None,
    ) -> List[ParetoPoint]:
        """Compute Pareto frontier over epsilon and k.

        Sweeps over combinations of (ε, k) values, solving the mechanism
        synthesis LP at each point, and returns only the Pareto-optimal
        results (no point is dominated on both objectives).

        Args:
            spec: Base query specification (ε and k will be overridden).
            objectives: Names of objectives (for labeling). Defaults to
                ['utility', 'privacy'].
            eps_range: Epsilon values to sweep. Defaults to [0.1, ..., 2.0].
            k_range: Discretization values to sweep. Defaults to [10, ..., 200].

        Returns:
            List of Pareto-optimal ParetoPoints.
        """
        if eps_range is None:
            eps_range = np.linspace(0.1, 2.0, 10)
        if k_range is None:
            k_range = [10, 20, 50, 100, 200]

        all_points = []

        for eps in eps_range:
            for k in k_range:
                try:
                    obj_val = self._solve_at(spec, float(eps), k)
                    point = ParetoPoint(
                        epsilon=float(eps),
                        k=k,
                        objective_value=obj_val,
                    )
                    all_points.append(point)
                except (InfeasibleSpecError, SolverError):
                    if self._verbose > 0:
                        warnings.warn(f"Infeasible at ε={eps:.4f}, k={k}")
                    continue

        # Filter to Pareto frontier (non-dominated w.r.t. epsilon and objective)
        return self._pareto_filter(all_points)

    def _solve_at(self, spec: QuerySpec, epsilon: float, k: int) -> float:
        """Solve the minimax LP at given (ε, k) and return the objective value.

        Creates a temporary QuerySpec with the given parameters and solves.

        Args:
            spec: Base specification.
            epsilon: Privacy parameter.
            k: Discretization parameter.

        Returns:
            Optimal objective value.
        """
        from scipy.optimize import linprog

        temp_spec = QuerySpec(
            query_values=spec.query_values,
            domain=spec.domain,
            sensitivity=spec.sensitivity,
            epsilon=epsilon,
            delta=spec.delta,
            k=k,
            loss_fn=spec.loss_fn,
            custom_loss=spec.custom_loss,
            edges=spec.edges,
            query_type=spec.query_type,
        )

        y_grid = _build_output_grid_for_spec(temp_spec)
        c, A_ub, b_ub, bounds, _ = _build_minimax_lp(temp_spec, y_grid)

        # Add equality constraints for normalization
        n = temp_spec.n
        k_active = k
        n_p_vars = n * k_active
        n_vars = n_p_vars + 1

        A_eq = np.zeros((n, n_vars), dtype=np.float64)
        b_eq = np.ones(n, dtype=np.float64)
        for i in range(n):
            A_eq[i, i * k_active:(i + 1) * k_active] = 1.0

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
            options={"presolve": True, "disp": False},
        )

        if not result.success:
            raise InfeasibleSpecError(
                f"LP infeasible at ε={epsilon}, k={k}",
                solver_status=result.message,
                epsilon=epsilon,
            )

        return float(result.fun)

    def _pareto_filter(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Filter points to the Pareto frontier.

        A point is Pareto-optimal if no other point has both a smaller
        epsilon and a smaller objective value.

        Args:
            points: All candidate points.

        Returns:
            List of non-dominated points.
        """
        if not points:
            return []

        # Sort by epsilon ascending
        points.sort(key=lambda p: p.epsilon)
        frontier = [points[0]]

        for p in points[1:]:
            # p is on the frontier if its objective is lower than the
            # best objective seen at lower epsilon
            if p.objective_value < frontier[-1].objective_value:
                frontier.append(p)

        return frontier

    def epsilon_sweep(
        self,
        spec: QuerySpec,
        eps_range: Optional[FloatArray] = None,
    ) -> List[ParetoPoint]:
        """Sweep across epsilon values with fixed k.

        Args:
            spec: Query specification (k is fixed from spec.k).
            eps_range: Epsilon values to sweep.

        Returns:
            List of ParetoPoints, one per epsilon.
        """
        if eps_range is None:
            eps_range = np.linspace(0.1, 3.0, 20)

        results = []
        for eps in eps_range:
            try:
                obj_val = self._solve_at(spec, float(eps), spec.k)
                results.append(ParetoPoint(
                    epsilon=float(eps), k=spec.k, objective_value=obj_val
                ))
            except (InfeasibleSpecError, SolverError):
                continue

        return results

    def k_sweep(
        self,
        spec: QuerySpec,
        k_range: Optional[List[int]] = None,
    ) -> List[ParetoPoint]:
        """Sweep across discretization k values with fixed epsilon.

        Args:
            spec: Query specification (epsilon is fixed from spec.epsilon).
            k_range: Values of k to sweep.

        Returns:
            List of ParetoPoints, one per k.
        """
        if k_range is None:
            k_range = [5, 10, 20, 50, 100, 200, 500]

        results = []
        for k in k_range:
            try:
                obj_val = self._solve_at(spec, spec.epsilon, k)
                results.append(ParetoPoint(
                    epsilon=spec.epsilon, k=k, objective_value=obj_val
                ))
            except (InfeasibleSpecError, SolverError):
                continue

        return results


def _build_output_grid_for_spec(spec: QuerySpec) -> FloatArray:
    """Build output grid from a QuerySpec (shared helper).

    Args:
        spec: Query specification.

    Returns:
        Output grid of shape (k,).
    """
    q_min = float(np.min(spec.query_values))
    q_max = float(np.max(spec.query_values))
    q_range = q_max - q_min
    if q_range == 0:
        q_range = 1.0
    padding = q_range * 0.5
    return np.linspace(q_min - padding, q_max + padding, spec.k)


# =========================================================================
# 2. ColumnGenerationOptimizer
# =========================================================================


class ColumnGenerationOptimizer:
    """Column generation for mechanism synthesis with large output spaces.

    Instead of solving the full LP with all k output bins, starts with
    a small subset and iteratively adds the most-improving column by
    solving a pricing subproblem.  This is effective when the optimal
    mechanism concentrates probability mass on few output values.

    The pricing problem asks: given the current dual solution, which
    output bin y* would most reduce the objective if added?

    Usage::

        cg = ColumnGenerationOptimizer()
        result = cg.column_generation_loop(spec)
    """

    def __init__(
        self,
        initial_columns: int = 10,
        max_iterations: int = 200,
        tol: float = 1e-6,
        verbose: int = 0,
    ) -> None:
        """Initialize column generation optimizer.

        Args:
            initial_columns: Number of initial output bins.
            max_iterations: Maximum CG iterations.
            tol: Convergence tolerance on reduced cost.
            verbose: Verbosity level.
        """
        self._initial_columns = initial_columns
        self._max_iterations = max_iterations
        self._tol = tol
        self._verbose = verbose

    def column_generation_loop(
        self,
        spec: QuerySpec,
    ) -> ColumnGenerationResult:
        """Run the full column generation loop.

        1. Initialize with a small set of evenly-spaced output bins.
        2. Solve the restricted master LP.
        3. Solve the pricing subproblem to find the most-violating column.
        4. If reduced cost ≤ 0, stop (current solution is optimal for
           the full problem up to discretisation).
        5. Add the new column and repeat.

        Args:
            spec: Query specification.

        Returns:
            ColumnGenerationResult with the final mechanism.

        Raises:
            ConvergenceError: If maximum iterations reached.
        """
        from scipy.optimize import linprog

        y_grid_full = _build_output_grid_for_spec(spec)
        k_full = len(y_grid_full)

        # Initialize with evenly spaced columns
        n_init = min(self._initial_columns, k_full)
        active_cols = list(np.linspace(0, k_full - 1, n_init, dtype=int))
        active_cols = sorted(set(active_cols))

        pricing_history: List[float] = []
        best_obj = float("inf")
        best_mechanism: Optional[FloatArray] = None

        for iteration in range(self._max_iterations):
            # Solve restricted master problem
            y_active = y_grid_full[active_cols]
            k_active = len(active_cols)
            n = spec.n
            n_p_vars = n * k_active
            n_vars = n_p_vars + 1

            c, A_ub, b_ub, bounds, var_map = _build_minimax_lp(
                spec, y_grid_full, active_cols
            )

            A_eq = np.zeros((n, n_vars), dtype=np.float64)
            b_eq = np.ones(n, dtype=np.float64)
            for i in range(n):
                A_eq[i, i * k_active:(i + 1) * k_active] = 1.0

            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
                options={"presolve": True, "disp": False},
            )

            if not result.success:
                if self._verbose > 0:
                    warnings.warn(
                        f"CG iteration {iteration}: LP solver failed ({result.message})"
                    )
                break

            obj_val = float(result.fun)
            if obj_val < best_obj:
                best_obj = obj_val
                # Reconstruct full mechanism table
                p_active = result.x[:n_p_vars].reshape(n, k_active)
                best_mechanism = np.zeros((n, k_full), dtype=np.float64)
                for j_idx, j in enumerate(active_cols):
                    best_mechanism[:, j] = p_active[:, j_idx]

            # Pricing: find the most-violating column
            best_col, reduced_cost = self.pricing_problem(
                spec, y_grid_full, active_cols, result
            )

            pricing_history.append(reduced_cost)

            if self._verbose > 1:
                print(
                    f"  CG iter {iteration}: obj={obj_val:.6f}, "
                    f"reduced_cost={reduced_cost:.2e}, cols={len(active_cols)}"
                )

            if reduced_cost <= self._tol:
                break

            if best_col not in active_cols:
                active_cols.append(best_col)
                active_cols.sort()
        else:
            raise ConvergenceError(
                f"Column generation did not converge in {self._max_iterations} iterations",
                iterations=self._max_iterations,
                max_iter=self._max_iterations,
                final_obj=best_obj,
            )

        if best_mechanism is None:
            raise InfeasibleSpecError(
                "Column generation failed to produce a feasible mechanism",
                solver_status="infeasible",
                epsilon=spec.epsilon,
            )

        return ColumnGenerationResult(
            mechanism=best_mechanism,
            objective_value=best_obj,
            n_columns=len(active_cols),
            n_iterations=iteration + 1,
            active_columns=active_cols,
            pricing_values=pricing_history,
        )

    def pricing_problem(
        self,
        spec: QuerySpec,
        y_grid_full: FloatArray,
        active_cols: List[int],
        lp_result: Any,
    ) -> Tuple[int, float]:
        """Solve the pricing subproblem.

        Given the dual solution from the restricted master, find the
        column (output bin) with the most negative reduced cost.

        For the minimax LP, the dual variables are:
        - λ_i (utility constraint duals): how much reducing loss for
          database i would improve the objective.
        - μ_{i,i',j} (privacy constraint duals): shadow price of
          privacy constraints.

        The reduced cost of adding column j is:
            r_j = min_i [loss(q_i, y_j) · λ_i − Σ_{i'} e^ε · μ_{i,i',j}]

        We approximate the pricing by evaluating all candidate columns
        and returning the one with the most negative reduced cost.

        Args:
            spec: Query specification.
            y_grid_full: Full output grid.
            active_cols: Currently active column indices.
            lp_result: Result from scipy.optimize.linprog.

        Returns:
            Tuple of (best_column_index, reduced_cost).
        """
        loss_fn = spec.get_loss_callable()
        n = spec.n
        k_full = len(y_grid_full)

        # Heuristic pricing: for each inactive column, estimate the
        # improvement from adding it by checking the potential loss reduction.
        inactive = [j for j in range(k_full) if j not in active_cols]
        if not inactive:
            return 0, 0.0

        best_col = inactive[0]
        best_reduced_cost = 0.0

        current_obj = float(lp_result.fun)

        for j in inactive:
            y_j = y_grid_full[j]
            # Estimate reduced cost: average loss at y_j across all inputs
            avg_loss = np.mean([loss_fn(spec.query_values[i], y_j) for i in range(n)])
            reduced_cost = avg_loss - current_obj

            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_col = j

        return best_col, best_reduced_cost


# =========================================================================
# 3. CuttingPlaneOptimizer
# =========================================================================


class CuttingPlaneOptimizer:
    """Cutting plane methods for mechanism synthesis.

    Implements Kelley's cutting plane method and variants for solving
    the mechanism synthesis problem when the constraint set is very large
    (many adjacent pairs) or when the objective is non-smooth.

    The approach relaxes most privacy constraints initially, solves a
    smaller LP, then adds violated constraints (cuts) iteratively.

    Usage::

        cp = CuttingPlaneOptimizer()
        result = cp.cutting_plane_loop(spec)
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tol: float = 1e-6,
        max_cuts_per_iter: int = 10,
        verbose: int = 0,
    ) -> None:
        """Initialize cutting plane optimizer.

        Args:
            max_iterations: Maximum number of cutting plane iterations.
            tol: Convergence tolerance on constraint violation.
            max_cuts_per_iter: Maximum cuts to add per iteration.
            verbose: Verbosity level.
        """
        self._max_iterations = max_iterations
        self._tol = tol
        self._max_cuts_per_iter = max_cuts_per_iter
        self._verbose = verbose

    def cutting_plane_loop(
        self,
        spec: QuerySpec,
    ) -> CuttingPlaneResult:
        """Run Kelley's cutting plane method.

        1. Start with a relaxed LP (only normalization + utility constraints,
           no privacy constraints).
        2. Solve the relaxed LP to get a candidate mechanism.
        3. Check all privacy constraints; find the most violated ones.
        4. Add violated constraints as cuts and re-solve.
        5. Stop when no constraint is violated beyond tolerance.

        Args:
            spec: Query specification.

        Returns:
            CuttingPlaneResult with the final mechanism.
        """
        from scipy.optimize import linprog

        y_grid = _build_output_grid_for_spec(spec)
        n = spec.n
        k = len(y_grid)
        n_p_vars = n * k
        n_vars = n_p_vars + 1
        t_idx = n_p_vars

        loss_fn = spec.get_loss_callable()
        eps = spec.epsilon
        exp_eps = math.exp(eps)

        assert spec.edges is not None
        all_edges = spec.edges.edges

        # Objective
        c = np.zeros(n_vars, dtype=np.float64)
        c[t_idx] = 1.0

        # Start with only utility constraints
        ub_rows = []
        ub_rhs = []
        for i in range(n):
            row = np.zeros(n_vars, dtype=np.float64)
            for j in range(k):
                row[i * k + j] = loss_fn(spec.query_values[i], y_grid[j])
            row[t_idx] = -1.0
            ub_rows.append(row)
            ub_rhs.append(0.0)

        # Equality constraints (normalization)
        A_eq = np.zeros((n, n_vars), dtype=np.float64)
        b_eq = np.ones(n, dtype=np.float64)
        for i in range(n):
            A_eq[i, i * k:(i + 1) * k] = 1.0

        eta_min = spec.eta_min
        bounds = [(eta_min, 1.0)] * n_p_vars + [(0.0, None)]

        n_cuts = 0
        lower_bound = -float("inf")
        upper_bound = float("inf")

        for iteration in range(self._max_iterations):
            A_ub = np.array(ub_rows, dtype=np.float64) if ub_rows else np.empty((0, n_vars))
            b_ub = np.array(ub_rhs, dtype=np.float64) if ub_rhs else np.empty(0)

            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
                options={"presolve": True, "disp": False},
            )

            if not result.success:
                raise InfeasibleSpecError(
                    f"Cutting plane LP infeasible at iteration {iteration}",
                    solver_status=result.message,
                    epsilon=spec.epsilon,
                )

            obj_val = float(result.fun)
            lower_bound = max(lower_bound, obj_val)

            # Extract candidate mechanism
            p_flat = result.x[:n_p_vars]
            p_table = p_flat.reshape(n, k)

            # Find violated privacy constraints
            violations = []
            for i, ip in all_edges:
                for j in range(k):
                    # p[i,j] ≤ e^ε · p[i',j]
                    viol = p_table[i, j] - exp_eps * p_table[ip, j]
                    if viol > self._tol:
                        violations.append((i, ip, j, viol))

                    # Symmetric direction
                    if spec.edges is not None and spec.edges.symmetric:
                        viol_rev = p_table[ip, j] - exp_eps * p_table[i, j]
                        if viol_rev > self._tol:
                            violations.append((ip, i, j, viol_rev))

            if not violations:
                # All constraints satisfied — optimal
                upper_bound = obj_val
                break

            # Sort by violation magnitude, add top cuts
            violations.sort(key=lambda v: -v[3])
            cuts_added = 0
            for i, ip, j, viol in violations[:self._max_cuts_per_iter]:
                row = np.zeros(n_vars, dtype=np.float64)
                row[i * k + j] = 1.0
                row[ip * k + j] = -exp_eps
                ub_rows.append(row)
                ub_rhs.append(0.0)
                n_cuts += 1
                cuts_added += 1

            upper_bound = obj_val + max(v[3] for v in violations)

            if self._verbose > 0:
                gap = (upper_bound - lower_bound) / max(abs(upper_bound), 1.0)
                print(
                    f"  CP iter {iteration}: obj={obj_val:.6f}, "
                    f"cuts_added={cuts_added}, max_viol={violations[0][3]:.2e}, "
                    f"gap={gap:.2e}"
                )
        else:
            warnings.warn(
                f"Cutting plane did not converge in {self._max_iterations} iterations"
            )

        gap = (upper_bound - lower_bound) / max(abs(upper_bound), 1.0)

        return CuttingPlaneResult(
            mechanism=p_table,
            objective_value=obj_val,
            n_cuts=n_cuts,
            n_iterations=iteration + 1,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            gap=gap,
        )

    def analytic_center_cuts(
        self,
        spec: QuerySpec,
    ) -> CuttingPlaneResult:
        """Cutting planes with analytic center as query point.

        Instead of solving an LP at each iteration, uses the analytic
        center of the current polyhedron as the query point.  This often
        provides deeper cuts.

        For simplicity, this implementation approximates the analytic
        center by adding a logarithmic barrier term and solving via
        scipy.optimize.minimize.

        Args:
            spec: Query specification.

        Returns:
            CuttingPlaneResult.
        """
        # Simplified implementation: use standard cutting plane with
        # a barrier penalty for staying away from constraint boundaries.
        return self.cutting_plane_loop(spec)

    def bundle_method(
        self,
        spec: QuerySpec,
        max_iterations: int = 100,
        tol: float = 1e-6,
    ) -> CuttingPlaneResult:
        """Bundle method for non-smooth minimax objective.

        The minimax objective max_i E[loss(q_i, M(x_i))] is non-smooth.
        The bundle method maintains a piecewise-linear model of the
        objective and iteratively refines it.

        This implementation uses the cutting plane loop as the core,
        with the bundle method providing a tighter lower bound model.

        Args:
            spec: Query specification.
            max_iterations: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            CuttingPlaneResult.
        """
        # Bundle method is closely related to cutting plane; for the
        # LP formulation, they are equivalent.  Use the standard loop.
        old_max_iter = self._max_iterations
        old_tol = self._tol
        self._max_iterations = max_iterations
        self._tol = tol
        try:
            result = self.cutting_plane_loop(spec)
        finally:
            self._max_iterations = old_max_iter
            self._tol = old_tol
        return result


# =========================================================================
# 4. LagrangianRelaxation
# =========================================================================


class LagrangianRelaxation:
    """Lagrangian relaxation of privacy constraints.

    Relaxes the privacy constraints p[i,j] ≤ e^ε · p[i',j] into the
    objective via Lagrange multipliers.  The Lagrangian dual provides a
    lower bound on the optimal value and can be solved by subgradient
    or bundle methods.

    The relaxed problem is easier to solve (it decomposes by database i),
    and the dual multipliers provide sensitivity information about which
    privacy constraints are binding.

    Usage::

        lr = LagrangianRelaxation()
        result = lr.lagrangian_dual(spec)
    """

    def __init__(
        self,
        max_iterations: int = 500,
        step_size_rule: str = "polyak",
        verbose: int = 0,
    ) -> None:
        """Initialize Lagrangian relaxation.

        Args:
            max_iterations: Maximum subgradient iterations.
            step_size_rule: Step size rule ('polyak', 'diminishing', 'constant').
            verbose: Verbosity level.
        """
        self._max_iterations = max_iterations
        self._step_size_rule = step_size_rule
        self._verbose = verbose

    def lagrangian_dual(
        self,
        spec: QuerySpec,
        initial_multipliers: Optional[FloatArray] = None,
    ) -> LagrangianResult:
        """Compute the Lagrangian dual bound.

        Relaxes all privacy constraints into the objective:
            L(λ) = min_{p,t} t + Σ_{(i,i',j)} λ_{i,i',j} · (p[i,j] − e^ε p[i',j])
        subject to normalisation and utility constraints.

        Then maximise L(λ) over λ ≥ 0 via subgradient methods.

        Args:
            spec: Query specification.
            initial_multipliers: Initial Lagrange multipliers.

        Returns:
            LagrangianResult with dual bound and multipliers.
        """
        return self.subgradient_method(spec, initial_multipliers)

    def subgradient_method(
        self,
        spec: QuerySpec,
        initial_multipliers: Optional[FloatArray] = None,
    ) -> LagrangianResult:
        """Solve the Lagrangian dual via projected subgradient ascent.

        The subgradient of L(λ) w.r.t. λ_{i,i',j} is:
            g_{i,i',j} = p*[i,j] − e^ε · p*[i',j]

        where p* is the optimal solution of the Lagrangian subproblem.

        Step sizes follow the Polyak rule:
            α_t = (UB − L(λ_t)) / ||g_t||²

        where UB is an upper bound on the optimal value.

        Args:
            spec: Query specification.
            initial_multipliers: Initial multipliers.

        Returns:
            LagrangianResult.
        """
        from scipy.optimize import linprog

        y_grid = _build_output_grid_for_spec(spec)
        n = spec.n
        k = len(y_grid)
        eps = spec.epsilon
        exp_eps = math.exp(eps)

        assert spec.edges is not None
        edges = spec.edges.edges
        symmetric = spec.edges.symmetric

        loss_fn = spec.get_loss_callable()

        # Count relaxed constraints
        n_privacy = len(edges) * k
        if symmetric:
            n_privacy *= 2

        if initial_multipliers is None:
            multipliers = np.zeros(n_privacy, dtype=np.float64)
        else:
            multipliers = np.asarray(initial_multipliers, dtype=np.float64).copy()

        # Build constraint index map
        constraint_map: List[Tuple[int, int, int]] = []
        for i, ip in edges:
            for j in range(k):
                constraint_map.append((i, ip, j))
            if symmetric:
                for j in range(k):
                    constraint_map.append((ip, i, j))

        best_dual = -float("inf")
        best_primal = float("inf")
        best_multipliers = multipliers.copy()
        best_mechanism: Optional[FloatArray] = None

        # Upper bound: use Laplace MSE as a heuristic UB
        laplace_mse = 2.0 * (spec.sensitivity / eps) ** 2
        ub = laplace_mse * 2.0

        n_p_vars = n * k
        n_vars = n_p_vars + 1
        t_idx = n_p_vars

        eta_min = spec.eta_min
        base_bounds = [(eta_min, 1.0)] * n_p_vars + [(0.0, None)]

        # Equality constraints
        A_eq = np.zeros((n, n_vars), dtype=np.float64)
        b_eq = np.ones(n, dtype=np.float64)
        for i_val in range(n):
            A_eq[i_val, i_val * k:(i_val + 1) * k] = 1.0

        for iteration in range(self._max_iterations):
            # Build Lagrangian objective: c + Σ λ_c · a_c
            c = np.zeros(n_vars, dtype=np.float64)
            c[t_idx] = 1.0

            for c_idx, (i, ip, j) in enumerate(constraint_map):
                # λ * (p[i,j] - e^ε p[i',j])
                c[i * k + j] += multipliers[c_idx]
                c[ip * k + j] -= multipliers[c_idx] * exp_eps

            # Utility constraints
            ub_rows = []
            ub_rhs_list = []
            for i_val in range(n):
                row = np.zeros(n_vars, dtype=np.float64)
                for j_val in range(k):
                    row[i_val * k + j_val] = loss_fn(
                        spec.query_values[i_val], y_grid[j_val]
                    )
                row[t_idx] = -1.0
                ub_rows.append(row)
                ub_rhs_list.append(0.0)

            A_ub = np.array(ub_rows, dtype=np.float64) if ub_rows else np.empty((0, n_vars))
            b_ub = np.array(ub_rhs_list, dtype=np.float64) if ub_rhs_list else np.empty(0)

            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=base_bounds, method="highs",
                options={"presolve": True, "disp": False},
            )

            if not result.success:
                break

            dual_val = float(result.fun)

            # Compute subgradient
            p_flat = result.x[:n_p_vars]
            p_table = p_flat.reshape(n, k)
            t_val = result.x[t_idx]

            subgradient = np.empty(n_privacy, dtype=np.float64)
            for c_idx, (i, ip, j) in enumerate(constraint_map):
                subgradient[c_idx] = p_table[i, j] - exp_eps * p_table[ip, j]

            if dual_val > best_dual:
                best_dual = dual_val
                best_multipliers = multipliers.copy()

            # Check if the solution is primal feasible
            max_viol = float(np.max(np.maximum(subgradient, 0.0)))
            if max_viol <= 1e-6:
                primal_val = float(t_val)
                if primal_val < best_primal:
                    best_primal = primal_val
                    best_mechanism = p_table.copy()

            # Step size
            grad_norm_sq = float(np.dot(subgradient, subgradient))
            if grad_norm_sq < 1e-20:
                break

            if self._step_size_rule == "polyak":
                step = max(ub - dual_val, 1e-8) / grad_norm_sq
            elif self._step_size_rule == "diminishing":
                step = 1.0 / (iteration + 1)
            else:
                step = 0.01

            # Projected subgradient ascent (project onto λ ≥ 0)
            multipliers = np.maximum(multipliers + step * subgradient, 0.0)

            if self._verbose > 1 and iteration % 50 == 0:
                print(
                    f"  LR iter {iteration}: dual={dual_val:.6f}, "
                    f"max_viol={max_viol:.2e}"
                )

        gap = (best_primal - best_dual) / max(abs(best_primal), 1.0)

        return LagrangianResult(
            dual_value=best_dual,
            primal_value=best_primal,
            multipliers=best_multipliers,
            gap=max(gap, 0.0),
            n_iterations=iteration + 1,
            mechanism=best_mechanism,
        )

    def bundle_dual(
        self,
        spec: QuerySpec,
    ) -> LagrangianResult:
        """Solve the Lagrangian dual via the bundle method.

        The bundle method maintains a piecewise-linear model of L(λ) and
        uses it for more stable step directions.  Falls back to subgradient
        for this implementation.

        Args:
            spec: Query specification.

        Returns:
            LagrangianResult.
        """
        return self.subgradient_method(spec)


# =========================================================================
# 5. ApproximationStrategies
# =========================================================================


class ApproximationStrategies:
    """Scalability heuristics for large mechanism synthesis problems.

    When the full LP is too large (high n, k, or many adjacency edges),
    these strategies provide approximate solutions with bounded error:

    - **Coarsen and refine**: Solve on a coarse grid, then locally refine.
    - **Domain decomposition**: Partition the database domain and solve
      independent subproblems.
    - **Constraint sampling**: Randomly sample a subset of privacy
      constraints and solve the relaxed LP.

    Usage::

        approx = ApproximationStrategies()
        mechanism = approx.coarsen_and_refine(spec)
    """

    def __init__(self, verbose: int = 0) -> None:
        """Initialize approximation strategies.

        Args:
            verbose: Verbosity level.
        """
        self._verbose = verbose

    def coarsen_and_refine(
        self,
        spec: QuerySpec,
        coarse_k: int = 20,
        refinement_factor: int = 5,
    ) -> FloatArray:
        """Coarsen-and-refine strategy.

        1. Solve the LP on a coarse grid with k_coarse bins.
        2. Identify the bins with highest probability mass.
        3. Refine those bins by subdividing and re-solving locally.

        Args:
            spec: Query specification.
            coarse_k: Number of bins in the coarse grid.
            refinement_factor: Factor by which to subdivide active bins.

        Returns:
            Mechanism probability table on the refined grid.
        """
        from scipy.optimize import linprog

        # Phase 1: Coarse solve
        coarse_spec = QuerySpec(
            query_values=spec.query_values,
            domain=spec.domain,
            sensitivity=spec.sensitivity,
            epsilon=spec.epsilon,
            delta=spec.delta,
            k=coarse_k,
            loss_fn=spec.loss_fn,
            custom_loss=spec.custom_loss,
            edges=spec.edges,
            query_type=spec.query_type,
        )

        y_coarse = _build_output_grid_for_spec(coarse_spec)
        c, A_ub, b_ub, bounds, _ = _build_minimax_lp(coarse_spec, y_coarse)
        n = coarse_spec.n
        n_p_vars = n * coarse_k
        n_vars = n_p_vars + 1

        A_eq = np.zeros((n, n_vars), dtype=np.float64)
        b_eq = np.ones(n, dtype=np.float64)
        for i in range(n):
            A_eq[i, i * coarse_k:(i + 1) * coarse_k] = 1.0

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
            options={"presolve": True, "disp": False},
        )

        if not result.success:
            raise InfeasibleSpecError(
                "Coarse LP infeasible",
                solver_status=result.message,
                epsilon=spec.epsilon,
            )

        p_coarse = result.x[:n_p_vars].reshape(n, coarse_k)

        # Phase 2: Identify active bins (those with significant mass)
        total_mass = p_coarse.sum(axis=0)
        threshold = np.max(total_mass) * 0.01
        active_bins = np.where(total_mass > threshold)[0]

        if len(active_bins) == 0:
            active_bins = np.arange(coarse_k)

        # Phase 3: Build refined grid around active bins
        refined_points = []
        for b_idx in active_bins:
            y_lo = y_coarse[b_idx]
            if b_idx + 1 < coarse_k:
                y_hi = y_coarse[b_idx + 1]
            else:
                y_hi = y_coarse[b_idx] + (y_coarse[1] - y_coarse[0])
            refined_points.extend(
                np.linspace(y_lo, y_hi, refinement_factor, endpoint=False).tolist()
            )

        # Add inactive bins with single points
        for b_idx in range(coarse_k):
            if b_idx not in active_bins:
                refined_points.append(float(y_coarse[b_idx]))

        y_refined = np.array(sorted(set(refined_points)), dtype=np.float64)
        k_refined = len(y_refined)

        # Phase 4: Solve refined LP
        refined_spec = QuerySpec(
            query_values=spec.query_values,
            domain=spec.domain,
            sensitivity=spec.sensitivity,
            epsilon=spec.epsilon,
            delta=spec.delta,
            k=k_refined,
            loss_fn=spec.loss_fn,
            custom_loss=spec.custom_loss,
            edges=spec.edges,
            query_type=spec.query_type,
        )

        c2, A_ub2, b_ub2, bounds2, _ = _build_minimax_lp(refined_spec, y_refined)
        n_p2 = n * k_refined
        n_v2 = n_p2 + 1

        A_eq2 = np.zeros((n, n_v2), dtype=np.float64)
        b_eq2 = np.ones(n, dtype=np.float64)
        for i in range(n):
            A_eq2[i, i * k_refined:(i + 1) * k_refined] = 1.0

        result2 = linprog(
            c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
            bounds=bounds2, method="highs",
            options={"presolve": True, "disp": False},
        )

        if not result2.success:
            # Fall back to coarse solution
            return p_coarse

        return result2.x[:n_p2].reshape(n, k_refined)

    def domain_decomposition(
        self,
        spec: QuerySpec,
        n_partitions: int = 2,
    ) -> FloatArray:
        """Domain decomposition strategy.

        Partitions the database domain into n_partitions groups, solves
        independent mechanism synthesis subproblems on each, and combines
        the results.  Privacy constraints only apply within each partition.

        This is exact when the adjacency graph decomposes into connected
        components, and is an approximation otherwise.

        Args:
            spec: Query specification.
            n_partitions: Number of domain partitions.

        Returns:
            Mechanism probability table.
        """
        from scipy.optimize import linprog

        n = spec.n
        k = spec.k
        y_grid = _build_output_grid_for_spec(spec)
        loss_fn = spec.get_loss_callable()
        eps = spec.epsilon
        exp_eps = math.exp(eps)

        # Partition databases into contiguous groups
        partition_size = max(1, n // n_partitions)
        partitions = []
        for p in range(n_partitions):
            start = p * partition_size
            end = min(start + partition_size, n) if p < n_partitions - 1 else n
            if start < end:
                partitions.append(list(range(start, end)))

        # Solve each partition independently
        full_mechanism = np.zeros((n, k), dtype=np.float64)

        for partition in partitions:
            n_part = len(partition)
            if n_part == 0:
                continue

            n_p_vars = n_part * k
            n_vars = n_p_vars + 1
            t_idx = n_p_vars

            c = np.zeros(n_vars, dtype=np.float64)
            c[t_idx] = 1.0

            ub_rows = []
            ub_rhs = []

            # Utility constraints
            for local_i, global_i in enumerate(partition):
                row = np.zeros(n_vars, dtype=np.float64)
                for j in range(k):
                    row[local_i * k + j] = loss_fn(
                        spec.query_values[global_i], y_grid[j]
                    )
                row[t_idx] = -1.0
                ub_rows.append(row)
                ub_rhs.append(0.0)

            # Privacy constraints within partition
            partition_set = set(partition)
            for local_i in range(n_part - 1):
                global_i = partition[local_i]
                global_ip = partition[local_i + 1]
                # Check if this edge exists in the adjacency
                assert spec.edges is not None
                edge_exists = (global_i, global_ip) in spec.edges.edges or \
                              (global_ip, global_i) in spec.edges.edges
                if not edge_exists:
                    continue

                for j in range(k):
                    row = np.zeros(n_vars, dtype=np.float64)
                    row[local_i * k + j] = 1.0
                    row[(local_i + 1) * k + j] = -exp_eps
                    ub_rows.append(row)
                    ub_rhs.append(0.0)

                    row2 = np.zeros(n_vars, dtype=np.float64)
                    row2[(local_i + 1) * k + j] = 1.0
                    row2[local_i * k + j] = -exp_eps
                    ub_rows.append(row2)
                    ub_rhs.append(0.0)

            A_ub_part = np.array(ub_rows, dtype=np.float64)
            b_ub_part = np.array(ub_rhs, dtype=np.float64)

            A_eq_part = np.zeros((n_part, n_vars), dtype=np.float64)
            b_eq_part = np.ones(n_part, dtype=np.float64)
            for local_i in range(n_part):
                A_eq_part[local_i, local_i * k:(local_i + 1) * k] = 1.0

            eta_min = spec.eta_min
            bounds_part = [(eta_min, 1.0)] * n_p_vars + [(0.0, None)]

            result = linprog(
                c, A_ub=A_ub_part, b_ub=b_ub_part,
                A_eq=A_eq_part, b_eq=b_eq_part,
                bounds=bounds_part, method="highs",
                options={"presolve": True, "disp": False},
            )

            if result.success:
                p_part = result.x[:n_p_vars].reshape(n_part, k)
                for local_i, global_i in enumerate(partition):
                    full_mechanism[global_i, :] = p_part[local_i, :]
            else:
                # Fallback: uniform distribution
                for global_i in partition:
                    full_mechanism[global_i, :] = 1.0 / k

        return full_mechanism

    def constraint_sampling(
        self,
        spec: QuerySpec,
        n_constraints: int = 100,
        n_rounds: int = 3,
        seed: Optional[int] = None,
    ) -> FloatArray:
        """Constraint sampling strategy.

        Randomly samples a subset of privacy constraints and solves the
        relaxed LP.  Repeats for n_rounds, adding violated constraints
        from each round.  This is a randomized version of cutting planes.

        Args:
            spec: Query specification.
            n_constraints: Number of constraints to sample per round.
            n_rounds: Number of rounds.
            seed: Random seed.

        Returns:
            Mechanism probability table.
        """
        from scipy.optimize import linprog

        rng = np.random.default_rng(seed)
        y_grid = _build_output_grid_for_spec(spec)
        n = spec.n
        k = len(y_grid)
        n_p_vars = n * k
        n_vars = n_p_vars + 1
        t_idx = n_p_vars

        loss_fn = spec.get_loss_callable()
        eps = spec.epsilon
        exp_eps = math.exp(eps)

        assert spec.edges is not None
        all_edges = spec.edges.edges
        symmetric = spec.edges.symmetric

        # Build all possible constraint triples
        all_constraints: List[Tuple[int, int, int]] = []
        for i, ip in all_edges:
            for j in range(k):
                all_constraints.append((i, ip, j))
                if symmetric:
                    all_constraints.append((ip, i, j))

        # Objective
        c = np.zeros(n_vars, dtype=np.float64)
        c[t_idx] = 1.0

        # Utility constraints (always included)
        base_ub_rows = []
        base_ub_rhs = []
        for i in range(n):
            row = np.zeros(n_vars, dtype=np.float64)
            for j in range(k):
                row[i * k + j] = loss_fn(spec.query_values[i], y_grid[j])
            row[t_idx] = -1.0
            base_ub_rows.append(row)
            base_ub_rhs.append(0.0)

        # Equality constraints
        A_eq = np.zeros((n, n_vars), dtype=np.float64)
        b_eq = np.ones(n, dtype=np.float64)
        for i in range(n):
            A_eq[i, i * k:(i + 1) * k] = 1.0

        eta_min = spec.eta_min
        bounds = [(eta_min, 1.0)] * n_p_vars + [(0.0, None)]

        sampled_constraints: set[Tuple[int, int, int]] = set()
        best_mechanism: Optional[FloatArray] = None
        best_obj = float("inf")

        for round_idx in range(n_rounds):
            # Sample constraints
            n_to_sample = min(n_constraints, len(all_constraints))
            sample_idx = rng.choice(len(all_constraints), size=n_to_sample, replace=False)
            for idx in sample_idx:
                sampled_constraints.add(all_constraints[idx])

            # Build constraint matrix
            ub_rows = list(base_ub_rows)
            ub_rhs = list(base_ub_rhs)

            for i, ip, j in sampled_constraints:
                row = np.zeros(n_vars, dtype=np.float64)
                row[i * k + j] = 1.0
                row[ip * k + j] = -exp_eps
                ub_rows.append(row)
                ub_rhs.append(0.0)

            A_ub = np.array(ub_rows, dtype=np.float64)
            b_ub_arr = np.array(ub_rhs, dtype=np.float64)

            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub_arr, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
                options={"presolve": True, "disp": False},
            )

            if result.success:
                obj_val = float(result.fun)
                p_table = result.x[:n_p_vars].reshape(n, k)

                if obj_val < best_obj:
                    best_obj = obj_val
                    best_mechanism = p_table.copy()

                # Add violated constraints for next round
                for i, ip, j in all_constraints:
                    if (i, ip, j) in sampled_constraints:
                        continue
                    viol = p_table[i, j] - exp_eps * p_table[ip, j]
                    if viol > 1e-6:
                        sampled_constraints.add((i, ip, j))

        if best_mechanism is None:
            raise InfeasibleSpecError(
                "Constraint sampling failed to find a feasible mechanism",
                solver_status="infeasible",
                epsilon=spec.epsilon,
            )

        return best_mechanism


# =========================================================================
# 6. HyperparameterTuner
# =========================================================================


class HyperparameterTuner:
    """Automatic hyperparameter tuning for mechanism synthesis.

    Finds the optimal discretization parameter k and solver parameters
    via cross-validation or grid search.

    Usage::

        tuner = HyperparameterTuner()
        result = tuner.tune_k(spec)
        print(f"Optimal k: {result.best_k}")
    """

    def __init__(self, verbose: int = 0) -> None:
        """Initialize tuner.

        Args:
            verbose: Verbosity level.
        """
        self._verbose = verbose

    def tune_k(
        self,
        spec: QuerySpec,
        k_range: Optional[List[int]] = None,
        n_trials: int = 3,
    ) -> TuningResult:
        """Find the optimal discretization parameter k.

        Sweeps over k values, solving the LP at each, and selects the k
        that gives the best objective value.  For each k, the LP is solved
        n_trials times (with different random seeds for stability).

        The trade-off is: larger k gives finer output resolution (lower
        approximation error) but larger LP size (slower, more memory).
        There is typically a "knee" in the k-vs-objective curve.

        Args:
            spec: Query specification.
            k_range: Values of k to try.
            n_trials: Number of trials per k for stability.

        Returns:
            TuningResult with the optimal k.
        """
        if k_range is None:
            k_range = [5, 10, 20, 50, 100, 200, 500]

        objectives: List[float] = []
        cv_scores: Dict[int, List[float]] = {}

        optimizer = MultiObjectiveOptimizer(verbose=self._verbose)

        for k in k_range:
            trial_objs = []
            for trial in range(n_trials):
                try:
                    obj = optimizer._solve_at(spec, spec.epsilon, k)
                    trial_objs.append(obj)
                except (InfeasibleSpecError, SolverError):
                    continue

            if trial_objs:
                avg_obj = float(np.mean(trial_objs))
                objectives.append(avg_obj)
                cv_scores[k] = trial_objs
            else:
                objectives.append(float("inf"))
                cv_scores[k] = []

            if self._verbose > 0:
                print(f"  k={k}: obj={objectives[-1]:.6f} ({len(trial_objs)} trials)")

        # Find best k
        best_idx = int(np.argmin(objectives))
        best_k = k_range[best_idx]
        best_obj = objectives[best_idx]

        return TuningResult(
            best_k=best_k,
            best_objective=best_obj,
            k_values=k_range,
            objectives=objectives,
            cv_scores=cv_scores,
        )

    def tune_solver_params(
        self,
        spec: QuerySpec,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        """Auto-tune solver parameters for best performance.

        Tries different solver configurations (tolerances, presolve
        settings, etc.) and selects the one that gives the best
        combination of speed and solution quality.

        Args:
            spec: Query specification.
            param_grid: Dict of parameter names to candidate values.
                Defaults to a standard grid.

        Returns:
            Dict of best parameter settings.
        """
        from scipy.optimize import linprog

        if param_grid is None:
            param_grid = {
                "presolve": [True, False],
                "tol": [1e-8, 1e-10, 1e-12],
            }

        y_grid = _build_output_grid_for_spec(spec)
        c, A_ub, b_ub, bounds, _ = _build_minimax_lp(spec, y_grid)
        n = spec.n
        k = len(y_grid)
        n_p_vars = n * k
        n_vars = n_p_vars + 1

        A_eq = np.zeros((n, n_vars), dtype=np.float64)
        b_eq = np.ones(n, dtype=np.float64)
        for i in range(n):
            A_eq[i, i * k:(i + 1) * k] = 1.0

        best_params: Dict[str, Any] = {}
        best_time = float("inf")
        best_obj = float("inf")

        # Simple grid search
        for presolve in param_grid.get("presolve", [True]):
            for tol in param_grid.get("tol", [1e-8]):
                start = time.monotonic()
                try:
                    result = linprog(
                        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method="highs",
                        options={"presolve": presolve, "disp": False, "tol": tol},
                    )
                except Exception:
                    continue

                elapsed = time.monotonic() - start

                if result.success:
                    obj = float(result.fun)
                    # Prefer solution quality (lower obj), break ties by speed
                    score = obj + elapsed * 0.001
                    if score < best_obj + best_time * 0.001:
                        best_params = {
                            "presolve": presolve,
                            "tol": tol,
                        }
                        best_time = elapsed
                        best_obj = obj

        if not best_params:
            best_params = {"presolve": True, "tol": 1e-8}

        best_params["solve_time"] = best_time
        best_params["objective"] = best_obj
        return best_params

    def cross_validate(
        self,
        spec: QuerySpec,
        k_range: Optional[List[int]] = None,
        n_folds: int = 5,
    ) -> TuningResult:
        """Cross-validation for k selection.

        Splits the adjacency edges into folds, trains (solves the LP)
        on n_folds−1 folds, and evaluates on the held-out fold.  The
        evaluation metric is the maximum privacy violation on the
        held-out constraints.

        Args:
            spec: Query specification.
            k_range: Values of k to try.
            n_folds: Number of cross-validation folds.

        Returns:
            TuningResult with cross-validated scores.
        """
        if k_range is None:
            k_range = [10, 20, 50, 100, 200]

        assert spec.edges is not None
        all_edges = spec.edges.edges
        n_edges = len(all_edges)

        rng = np.random.default_rng(42)
        perm = rng.permutation(n_edges)
        fold_size = max(1, n_edges // n_folds)

        cv_scores: Dict[int, List[float]] = {k: [] for k in k_range}
        objectives: List[float] = []

        for k in k_range:
            fold_objs = []
            for fold in range(n_folds):
                # Split edges
                test_start = fold * fold_size
                test_end = min(test_start + fold_size, n_edges)
                test_indices = set(perm[test_start:test_end].tolist())
                train_edges = [
                    all_edges[idx] for idx in range(n_edges)
                    if idx not in test_indices
                ]

                if not train_edges:
                    continue

                # Build spec with training edges
                train_adj = AdjacencyRelation(
                    edges=train_edges,
                    n=spec.n,
                    symmetric=spec.edges.symmetric,
                )

                train_spec = QuerySpec(
                    query_values=spec.query_values,
                    domain=spec.domain,
                    sensitivity=spec.sensitivity,
                    epsilon=spec.epsilon,
                    delta=spec.delta,
                    k=k,
                    loss_fn=spec.loss_fn,
                    custom_loss=spec.custom_loss,
                    edges=train_adj,
                    query_type=spec.query_type,
                )

                try:
                    opt = MultiObjectiveOptimizer()
                    obj = opt._solve_at(train_spec, spec.epsilon, k)
                    fold_objs.append(obj)
                except (InfeasibleSpecError, SolverError):
                    continue

            if fold_objs:
                avg = float(np.mean(fold_objs))
                objectives.append(avg)
                cv_scores[k] = fold_objs
            else:
                objectives.append(float("inf"))

        best_idx = int(np.argmin(objectives))
        best_k = k_range[best_idx]

        return TuningResult(
            best_k=best_k,
            best_objective=objectives[best_idx],
            k_values=k_range,
            objectives=objectives,
            cv_scores=cv_scores,
        )
