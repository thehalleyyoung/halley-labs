"""
Column Generation Solver
=========================

Implements the column-generation loop for solving LP relaxations of the
causal polytope.  The *restricted master problem* (RMP) is solved via
``scipy.optimize.linprog``; columns are added by calling the pricing
sub-problem (see ``pricing.py``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, OptimizeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class Column:
    """A single column (extreme point) in the column-generation scheme."""
    index: int
    coefficients: np.ndarray   # entry of the column in the master LP
    cost: float                # objective coefficient c_j
    reduced_cost: float = 0.0
    age: int = 0               # iterations since last in basis
    in_basis: bool = False
    origin: str = "initial"    # how the column was generated


@dataclass
class ColumnPool:
    """Pool of columns managed during column generation."""
    columns: List[Column] = field(default_factory=list)
    _next_id: int = 0

    def add(self, coefficients: np.ndarray, cost: float, origin: str = "pricing") -> Column:
        col = Column(
            index=self._next_id,
            coefficients=coefficients.copy(),
            cost=cost,
            origin=origin,
        )
        self.columns.append(col)
        self._next_id += 1
        return col

    def remove(self, index: int) -> None:
        self.columns = [c for c in self.columns if c.index != index]

    def age_out(self, max_age: int) -> int:
        """Remove columns not in basis that have aged beyond max_age.  Returns count removed."""
        before = len(self.columns)
        self.columns = [
            c for c in self.columns
            if c.in_basis or c.age < max_age
        ]
        return before - len(self.columns)

    def increment_ages(self) -> None:
        for c in self.columns:
            if not c.in_basis:
                c.age += 1
            else:
                c.age = 0

    def size(self) -> int:
        return len(self.columns)

    def active_count(self) -> int:
        return sum(1 for c in self.columns if c.in_basis)

    def get_cost_vector(self) -> np.ndarray:
        return np.array([c.cost for c in self.columns], dtype=np.float64)

    def get_coefficient_matrix(self, sparse_format: bool = True):
        """Return the column coefficient matrix A_pool (m x n_cols)."""
        if not self.columns:
            return sparse.csc_matrix((0, 0))
        m = self.columns[0].coefficients.shape[0]
        n = len(self.columns)
        if sparse_format:
            rows, cols, data = [], [], []
            for j, col in enumerate(self.columns):
                nz = np.nonzero(col.coefficients)[0]
                rows.extend(nz.tolist())
                cols.extend([j] * len(nz))
                data.extend(col.coefficients[nz].tolist())
            return sparse.csc_matrix((data, (rows, cols)), shape=(m, n))
        else:
            mat = np.zeros((m, n), dtype=np.float64)
            for j, col in enumerate(self.columns):
                mat[:, j] = col.coefficients
            return mat

    def update_basis_status(self, basis_mask: np.ndarray) -> None:
        """Update in_basis flag from an array of booleans."""
        for j, col in enumerate(self.columns):
            if j < len(basis_mask):
                col.in_basis = bool(basis_mask[j])

    def get_reduced_costs(self) -> np.ndarray:
        return np.array([c.reduced_cost for c in self.columns], dtype=np.float64)


@dataclass
class MasterProblem:
    """
    Encapsulates the restricted master problem (RMP).

    The master LP is:
        min   c_pool^T lambda
        s.t.  A_pool lambda = b
              lambda >= 0
    where each column of A_pool is one extreme point of the polytope.
    """
    A_eq: sparse.spmatrix          # constraint matrix (full, not just pool)
    b_eq: np.ndarray               # RHS
    c_full: np.ndarray             # full-dimensional objective
    total_vars: int                # dimension of original (full) problem
    pool: ColumnPool
    _last_result: Optional[OptimizeResult] = None
    _last_duals: Optional[np.ndarray] = None
    _last_obj: float = float("inf")
    _num_solves: int = 0

    def solve(self, warm_start_x: Optional[np.ndarray] = None) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solve the restricted master problem.

        Returns
        -------
        obj_val : float
            Optimal objective value.
        x : ndarray or None
            Optimal primal variables (lambda weights on columns).
        duals : ndarray or None
            Dual variables for equality constraints.
        """
        if self.pool.size() == 0:
            return float("inf"), None, None

        c_pool = self.pool.get_cost_vector()
        A_pool = self.pool.get_coefficient_matrix(sparse_format=True)

        m, n = A_pool.shape
        if m == 0 or n == 0:
            return float("inf"), None, None

        # Add upper bound of 1 on each lambda to help numerics
        bounds = [(0.0, None) for _ in range(n)]

        options: Dict = {"maxiter": 10000, "presolve": True,
                         "dual_feasibility_tolerance": 1e-10,
                         "primal_feasibility_tolerance": 1e-10}

        try:
            if sparse.issparse(A_pool):
                A_dense = A_pool.toarray()
            else:
                A_dense = np.asarray(A_pool)

            result = linprog(
                c=c_pool,
                A_eq=A_dense,
                b_eq=self.b_eq[:m],
                bounds=bounds,
                method="highs",
                options=options,
            )
        except Exception as exc:
            logger.warning("linprog failed: %s", exc)
            return float("inf"), None, None

        self._num_solves += 1
        self._last_result = result

        if result.success:
            self._last_obj = result.fun
            x = result.x

            # Extract duals from the scipy result
            if hasattr(result, "eqlin") and hasattr(result.eqlin, "marginals"):
                duals = np.array(result.eqlin.marginals, dtype=np.float64)
            else:
                duals = self._estimate_duals(A_dense, c_pool, x)

            self._last_duals = duals

            # Update basis status
            basis_mask = np.array([xi > 1e-10 for xi in x])
            self.pool.update_basis_status(basis_mask)

            return result.fun, x, duals
        else:
            logger.warning("Master LP status=%d: %s", result.status, result.message)
            return float("inf"), None, None

    def _estimate_duals(
        self,
        A: np.ndarray,
        c: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate dual variables from primal solution using least-squares
        on the active constraints.  duals satisfy A^T y = c for basic columns.
        """
        basic = x > 1e-10
        n_basic = np.sum(basic)
        if n_basic == 0:
            return np.zeros(A.shape[0], dtype=np.float64)

        A_basic = A[:, basic]
        c_basic = c[basic]

        try:
            # Solve A_basic^T y = c_basic in least-squares sense
            y, _, _, _ = np.linalg.lstsq(A_basic.T, c_basic, rcond=None)
            return y
        except np.linalg.LinAlgError:
            return np.zeros(A.shape[0], dtype=np.float64)

    @property
    def num_solves(self) -> int:
        return self._num_solves

    @property
    def last_objective(self) -> float:
        return self._last_obj


# ---------------------------------------------------------------------------
#  Column generation solver
# ---------------------------------------------------------------------------

@dataclass
class CGDiagnostics:
    """Per-iteration diagnostics for column generation."""
    iterations: int = 0
    total_columns_generated: int = 0
    active_columns: int = 0
    master_lp_solves: int = 0
    pricing_solves: int = 0
    convergence_gap: float = float("inf")
    solve_time_seconds: float = 0.0
    master_times: List[float] = field(default_factory=list)
    pricing_times: List[float] = field(default_factory=list)
    bound_history: List[Tuple[float, float]] = field(default_factory=list)
    reduced_cost_history: List[float] = field(default_factory=list)


class ColumnGenerationSolver:
    """
    Column-generation solver for the causal polytope LP.

    The solver alternates between:
      1. Solving the restricted master problem (RMP).
      2. Solving the pricing sub-problem to find columns with negative
         reduced cost.

    Parameters
    ----------
    c : ndarray
        Full-dimensional objective vector.
    A_eq : sparse matrix
        Equality constraint matrix.
    b_eq : ndarray
        Equality RHS.
    total_vars : int
        Dimension of the full problem.
    config : SolverConfig
        Solver parameters.
    dag : DAGSpec
        The (possibly mutilated) DAG.
    """

    def __init__(
        self,
        c: np.ndarray,
        A_eq: sparse.spmatrix,
        b_eq: np.ndarray,
        total_vars: int,
        config,
        dag,
    ):
        from .pricing import PricingSubproblem

        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.total_vars = total_vars
        self.config = config
        self.dag = dag

        self.pool = ColumnPool()
        self.master = MasterProblem(
            A_eq=A_eq,
            b_eq=b_eq,
            c_full=c,
            total_vars=total_vars,
            pool=self.pool,
        )
        self.pricer = PricingSubproblem(dag=dag, c=c, A_eq=A_eq, b_eq=b_eq)
        self._diagnostics = CGDiagnostics()
        self._best_obj = float("inf")
        self._best_x: Optional[np.ndarray] = None
        self._best_duals: Optional[np.ndarray] = None

        # Stabilisation centre
        self._stab_centre: Optional[np.ndarray] = None
        self._stab_alpha = config.stabilization_alpha if config.stabilization else 0.0

    def solve(
        self,
        max_iterations: Optional[int] = None,
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], Optional[CGDiagnostics]]:
        """
        Run the column-generation loop.

        Returns
        -------
        obj_val : float
        columns : ndarray or None  (column weights)
        duals : ndarray or None
        diagnostics : CGDiagnostics or None
        """
        max_iter = max_iterations or self.config.max_iterations
        start = time.time()

        # Phase 0: generate initial columns
        self._generate_initial_columns()
        logger.debug("Initial column pool size: %d", self.pool.size())

        for it in range(1, max_iter + 1):
            # Phase 1: solve restricted master
            t0 = time.time()
            obj_val, x, duals = self.master.solve(
                warm_start_x=self._best_x if self.config.warm_start else None
            )
            master_time = time.time() - t0
            self._diagnostics.master_times.append(master_time)
            self._diagnostics.master_lp_solves += 1

            if x is None or duals is None:
                logger.warning("Master LP infeasible/failed at iter %d", it)
                if self._best_x is not None:
                    break
                # Try adding more initial columns
                self._generate_random_columns(10)
                continue

            if obj_val < self._best_obj:
                self._best_obj = obj_val
                self._best_x = x.copy()
                self._best_duals = duals.copy()

            # Stabilisation: blend duals with centre
            pricing_duals = duals
            if self.config.stabilization and self._stab_centre is not None:
                alpha = self._stab_alpha
                pricing_duals = alpha * self._stab_centre + (1 - alpha) * duals
            self._stab_centre = duals.copy()

            # Phase 2: pricing
            t1 = time.time()
            new_columns = self._run_pricing(pricing_duals)
            pricing_time = time.time() - t1
            self._diagnostics.pricing_times.append(pricing_time)
            self._diagnostics.pricing_solves += 1

            # Compute best reduced cost
            if new_columns:
                best_rc = min(col.reduced_cost for col in new_columns)
            else:
                best_rc = 0.0
            self._diagnostics.reduced_cost_history.append(best_rc)
            self._diagnostics.bound_history.append((obj_val, best_rc))

            # Check convergence
            gap = abs(best_rc) if new_columns else 0.0
            self._diagnostics.convergence_gap = gap
            self._diagnostics.iterations = it

            if it % self.config.log_interval == 0 or gap < self.config.gap_tolerance:
                logger.debug(
                    "CG iter %4d | obj=%.8f | rc=%.2e | cols=%d | master=%.3fs | pricing=%.3fs",
                    it, obj_val, best_rc, self.pool.size(),
                    master_time, pricing_time,
                )

            if gap < self.config.gap_tolerance:
                logger.info("Column generation converged at iteration %d (gap=%.2e)", it, gap)
                break

            if not new_columns:
                logger.info("No improving columns found at iteration %d", it)
                break

            # Age out old columns
            self.pool.increment_ages()
            removed = self.pool.age_out(self.config.column_age_limit)
            if removed > 0:
                logger.debug("Aged out %d columns", removed)

            elapsed = time.time() - start
            if elapsed > self.config.time_limit:
                logger.warning("Time limit reached (%.1fs)", elapsed)
                break

        self._diagnostics.solve_time_seconds = time.time() - start
        self._diagnostics.total_columns_generated = self.pool._next_id
        self._diagnostics.active_columns = self.pool.active_count()

        return self._best_obj, self._best_x, self._best_duals, self._diagnostics

    # ------------------------------------------------------------------
    # Initial column generation
    # ------------------------------------------------------------------

    def _generate_initial_columns(self) -> None:
        """Generate initial feasible columns (Markov-compatible distributions)."""
        n = self.total_vars
        rng = np.random.default_rng(12345)
        topo = self.dag.topological_order()

        # Strategy 1: generate Markov-compatible distributions by sampling
        # random CPTs and computing the joint via the DAG factorisation
        for trial in range(self.config.num_initial_columns):
            joint = self._sample_markov_compatible(rng, topo)
            if joint is not None:
                self._add_column_if_feasible(joint, origin="markov_sample")

        # Strategy 2: vertex columns (deterministic CPTs)
        for idx in range(min(self.config.num_initial_columns, n)):
            col = np.zeros(n, dtype=np.float64)
            col[idx % n] = 1.0
            self._add_column_if_feasible(col, origin="vertex")

    def _sample_markov_compatible(
        self, rng: np.random.Generator, topo: list
    ) -> Optional[np.ndarray]:
        """
        Sample a joint distribution that factorises according to the DAG:
            P(x) = prod_i P(x_i | pa(x_i))
        by generating random CPTs.
        """
        n = self.total_vars
        strides = _compute_strides(self.dag, topo)

        # Generate random CPTs for each node
        cpts: dict = {}
        for node in topo:
            card = self.dag.card[node]
            parents = self.dag.parents(node)
            if not parents:
                cpt = rng.dirichlet(np.ones(card))
                cpts[node] = cpt.reshape(1, card)
            else:
                pa_size = 1
                for p in parents:
                    pa_size *= self.dag.card[p]
                cpt = np.zeros((pa_size, card), dtype=np.float64)
                for pi in range(pa_size):
                    cpt[pi] = rng.dirichlet(np.ones(card))
                cpts[node] = cpt

        # Compute joint by enumerating all assignments
        joint = np.zeros(n, dtype=np.float64)
        for flat_idx in range(n):
            assign: dict = {}
            rem = flat_idx
            for node in topo:
                card = self.dag.card[node]
                stride = strides[node]
                assign[node] = (rem // stride) % card

            prob = 1.0
            for node in topo:
                parents = self.dag.parents(node)
                card = self.dag.card[node]
                val = assign[node]

                if not parents:
                    prob *= cpts[node][0, val]
                else:
                    # Compute parent config index
                    pa_idx = 0
                    pa_stride = 1
                    for p in reversed(parents):
                        pa_idx += assign[p] * pa_stride
                        pa_stride *= self.dag.card[p]
                    prob *= cpts[node][pa_idx, val]

            joint[flat_idx] = prob

        total = joint.sum()
        if total > 1e-15:
            joint /= total
        return joint

    def _generate_random_columns(self, count: int) -> None:
        """Add random feasible columns to the pool."""
        n = self.total_vars
        rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
        for _ in range(count):
            weights = rng.dirichlet(np.ones(n))
            self._add_column_if_feasible(weights, origin="random_recovery")

    def _add_column_if_feasible(self, col_vec: np.ndarray, origin: str) -> Optional[Column]:
        """Add a column to the pool, computing its constraint coefficients."""
        if self.A_eq is not None:
            if sparse.issparse(self.A_eq):
                a_col = self.A_eq.dot(col_vec)
            else:
                a_col = self.A_eq @ col_vec
        else:
            a_col = col_vec

        cost = float(self.c @ col_vec)
        col = self.pool.add(coefficients=a_col, cost=cost, origin=origin)
        return col

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def _run_pricing(self, duals: np.ndarray) -> List[Column]:
        """Run the pricing sub-problem and add columns with negative reduced cost."""
        from .pricing import PricingStrategy

        strategy = PricingStrategy(self.config.pricing_strategy)
        results = self.pricer.price(duals, strategy=strategy,
                                    max_columns=self.config.max_columns_per_iter)

        new_cols: List[Column] = []
        for pr in results:
            if pr.reduced_cost < -self.config.reduced_cost_tolerance:
                col = self._add_column_if_feasible(pr.column, origin="pricing")
                if col is not None:
                    col.reduced_cost = pr.reduced_cost
                    new_cols.append(col)

        return new_cols

    # ------------------------------------------------------------------
    # Warm start helpers
    # ------------------------------------------------------------------

    def warm_start_from(self, other: "ColumnGenerationSolver") -> None:
        """Import columns from another solver instance for warm-starting."""
        for col in other.pool.columns:
            if col.in_basis:
                self.pool.add(
                    coefficients=col.coefficients,
                    cost=col.cost,
                    origin="warm_start",
                )

    def export_basis(self) -> Dict:
        """Export current basis information for later warm-starting."""
        basis_cols = [
            {
                "coefficients": col.coefficients.tolist(),
                "cost": col.cost,
                "index": col.index,
            }
            for col in self.pool.columns
            if col.in_basis
        ]
        return {
            "basis_columns": basis_cols,
            "best_obj": self._best_obj,
            "iterations": self._diagnostics.iterations,
        }

    def import_basis(self, basis_data: Dict) -> None:
        """Import basis data previously exported."""
        for bc in basis_data.get("basis_columns", []):
            self.pool.add(
                coefficients=np.array(bc["coefficients"], dtype=np.float64),
                cost=bc["cost"],
                origin="imported_basis",
            )


class DantzigWolfeDecomposer:
    """
    Dantzig-Wolfe decomposition for structured LPs arising from
    causal polytopes with block-diagonal subproblems.

    The joint distribution polytope decomposes by variable families
    (each family = a variable and its parents).  Each subproblem is
    a local conditional-distribution polytope.
    """

    def __init__(self, dag, c: np.ndarray, A_eq: sparse.spmatrix, b_eq: np.ndarray):
        self.dag = dag
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self._blocks: List[_SubBlock] = []
        self._linking_rows: Optional[np.ndarray] = None
        self._decompose()

    def _decompose(self) -> None:
        """Identify block structure from DAG families."""
        topo = self.dag.topological_order()
        total_vars = 1
        for n in self.dag.nodes:
            total_vars *= self.dag.card[n]

        col_to_block: Dict[int, int] = {}
        strides = _compute_strides(self.dag, topo)

        for b_idx, node in enumerate(topo):
            family = [node] + self.dag.parents(node)
            family_size = 1
            for f in family:
                family_size *= self.dag.card[f]

            block_cols = list(range(b_idx * family_size, min((b_idx + 1) * family_size, total_vars)))
            for col_idx in block_cols:
                col_to_block[col_idx] = b_idx

            self._blocks.append(_SubBlock(
                block_id=b_idx,
                node=node,
                family=family,
                col_indices=block_cols,
            ))

        if sparse.issparse(self.A_eq):
            m = self.A_eq.shape[0]
        else:
            m = len(self.b_eq)

        linking = []
        for i in range(m):
            if sparse.issparse(self.A_eq):
                row = self.A_eq.getrow(i)
                nz_cols = row.nonzero()[1]
            else:
                nz_cols = np.nonzero(self.A_eq[i])[0]
            blocks_touched = set(col_to_block.get(c, -1) for c in nz_cols)
            blocks_touched.discard(-1)
            if len(blocks_touched) > 1:
                linking.append(i)

        self._linking_rows = np.array(linking, dtype=int)

    def get_blocks(self) -> List["_SubBlock"]:
        return self._blocks

    def get_linking_rows(self) -> np.ndarray:
        return self._linking_rows if self._linking_rows is not None else np.array([], dtype=int)

    def solve_subproblem(self, block_id: int, duals_linking: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Solve a single block subproblem given dual variables for linking constraints.

        Returns (objective, extreme_point).
        """
        block = self._blocks[block_id]
        cols = block.col_indices
        n_cols = len(cols)

        if n_cols == 0:
            return 0.0, np.array([], dtype=np.float64)

        # Local objective: c_block - A_linking^T duals
        c_local = self.c[cols].copy()

        if self._linking_rows is not None and len(self._linking_rows) > 0:
            if sparse.issparse(self.A_eq):
                A_link = self.A_eq[self._linking_rows][:, cols]
                c_local -= A_link.T.dot(duals_linking)
            else:
                A_link = self.A_eq[np.ix_(self._linking_rows, cols)]
                c_local -= A_link.T @ duals_linking

        # Extract local constraints (rows involving only this block)
        if sparse.issparse(self.A_eq):
            m = self.A_eq.shape[0]
        else:
            m = self.A_eq.shape[0]

        local_rows = []
        for i in range(m):
            if self._linking_rows is not None and i in self._linking_rows:
                continue
            if sparse.issparse(self.A_eq):
                row = self.A_eq.getrow(i)
                nz = row.nonzero()[1]
            else:
                nz = np.nonzero(self.A_eq[i])[0]
            if all(c in cols for c in nz) and len(nz) > 0:
                local_rows.append(i)

        if local_rows:
            if sparse.issparse(self.A_eq):
                A_local = self.A_eq[local_rows][:, cols].toarray()
            else:
                A_local = self.A_eq[np.ix_(local_rows, cols)]
            b_local = self.b_eq[local_rows]

            bounds = [(0.0, None)] * n_cols
            result = linprog(
                c=c_local,
                A_eq=A_local,
                b_eq=b_local,
                bounds=bounds,
                method="highs",
            )
            if result.success:
                full_point = np.zeros(len(self.c), dtype=np.float64)
                full_point[cols] = result.x
                return result.fun, full_point
        else:
            # No local constraints; just pick the minimum-cost vertex
            best_idx = np.argmin(c_local)
            full_point = np.zeros(len(self.c), dtype=np.float64)
            full_point[cols[best_idx]] = 1.0
            return float(c_local[best_idx]), full_point

        return float("inf"), np.zeros(len(self.c), dtype=np.float64)


@dataclass
class _SubBlock:
    block_id: int
    node: str
    family: List[str]
    col_indices: List[int]


def _compute_strides(dag, topo: List[str]) -> Dict[str, int]:
    strides: Dict[str, int] = {}
    stride = 1
    for node in reversed(topo):
        strides[node] = stride
        stride *= dag.card[node]
    return strides


class ColumnStabilizer:
    """
    Implements column stabilisation via the du Merle et al. technique.

    Adds boxed penalty variables around a stability centre to prevent
    wild oscillations of dual variables between iterations.
    """

    def __init__(self, num_constraints: int, delta_init: float = 0.1):
        self.num_constraints = num_constraints
        self.delta = delta_init
        self._centre: Optional[np.ndarray] = None
        self._penalty: float = 1e4

    def set_centre(self, duals: np.ndarray) -> None:
        self._centre = duals.copy()

    def augment_master(
        self,
        c_pool: np.ndarray,
        A_pool: np.ndarray,
        b_eq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Augment the master LP with stabilisation variables.

        For each constraint i, add variables s_i^+, s_i^- with:
            A_pool lambda + s^+ - s^- = b
            0 <= s^+ <= delta, 0 <= s^- <= delta
            cost of s^+_i = +penalty * (1 if center_i > 0 else -1)
            cost of s^-_i = -cost of s^+_i
        """
        if self._centre is None:
            return c_pool, A_pool, b_eq

        m = A_pool.shape[0]
        n = A_pool.shape[1]
        n_stab = 2 * m

        # Augmented cost
        stab_costs = np.zeros(n_stab, dtype=np.float64)
        for i in range(m):
            sign = 1.0 if self._centre[i] >= 0 else -1.0
            stab_costs[i] = self._penalty * sign         # s_i^+
            stab_costs[m + i] = -self._penalty * sign    # s_i^-

        c_aug = np.concatenate([c_pool, stab_costs])

        # Augmented A: [A_pool | I | -I]
        I_m = np.eye(m, dtype=np.float64)
        A_aug = np.hstack([A_pool, I_m, -I_m])

        return c_aug, A_aug, b_eq

    def extract_duals(
        self,
        x_aug: np.ndarray,
        n_original: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split augmented solution into original variables and stabilisation.

        Returns (x_original, s_plus_minus).
        """
        x_orig = x_aug[:n_original]
        s = x_aug[n_original:]
        return x_orig, s

    def update_delta(self, improved: bool) -> None:
        """Shrink or grow the stabilisation box."""
        if improved:
            self.delta *= 1.1
        else:
            self.delta *= 0.5
        self.delta = max(self.delta, 1e-8)
        self.delta = min(self.delta, 1e3)

    def update_penalty(self, gap: float) -> None:
        """Adapt penalty based on convergence gap."""
        if gap < 1e-4:
            self._penalty *= 10.0
        elif gap > 1e-1:
            self._penalty *= 0.1
        self._penalty = max(self._penalty, 1.0)
        self._penalty = min(self._penalty, 1e8)
