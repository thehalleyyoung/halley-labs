"""Column generation for large-scale DP mechanism LP.

Implements Dantzig-Wolfe decomposition and branch-and-price for mechanism
design where the full LP has too many columns to enumerate explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp_sparse
from scipy.optimize import linprog

from dp_forge.sparse import (
    Column,
    CutType,
    DecompositionType,
    LagrangianState,
    PricingStrategy,
    SparseConfig,
    SparseResult,
)
from dp_forge.types import (
    AdjacencyRelation,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column pool
# ---------------------------------------------------------------------------


class ColumnPool:
    """Manage candidate mechanism columns efficiently.

    Stores columns (output distributions) indexed by source input,
    supports efficient lookup and pruning of inactive columns.

    Attributes:
        columns: List of all columns in the pool.
        _by_input: Columns grouped by source input index.
        _next_index: Next column index to assign.
    """

    def __init__(self) -> None:
        self.columns: List[Column] = []
        self._by_input: Dict[int, List[int]] = {}
        self._next_index: int = 0

    @property
    def size(self) -> int:
        """Number of columns in the pool."""
        return len(self.columns)

    def add(self, distribution: npt.NDArray[np.float64], source_input: int,
            reduced_cost: float = 0.0) -> Column:
        """Add a column to the pool.

        Args:
            distribution: Probability distribution over output bins.
            source_input: Input database index this column corresponds to.
            reduced_cost: Reduced cost from pricing.

        Returns:
            The newly created Column.
        """
        col = Column(
            index=self._next_index,
            distribution=distribution.copy(),
            reduced_cost=reduced_cost,
            source_input=source_input,
        )
        self.columns.append(col)
        self._by_input.setdefault(source_input, []).append(len(self.columns) - 1)
        self._next_index += 1
        return col

    def get_columns_for_input(self, source_input: int) -> List[Column]:
        """Return all columns corresponding to a given input index."""
        indices = self._by_input.get(source_input, [])
        return [self.columns[i] for i in indices]

    def prune(self, keep_indices: set) -> None:
        """Remove columns not in *keep_indices* (by Column.index)."""
        old = self.columns
        self.columns = [c for c in old if c.index in keep_indices]
        self._by_input.clear()
        for pos, col in enumerate(self.columns):
            self._by_input.setdefault(col.source_input, []).append(pos)

    def get_matrix(self, k: int) -> npt.NDArray[np.float64]:
        """Return the column matrix (k × num_columns).

        Each column j is the output distribution of ``columns[j]``.
        """
        if not self.columns:
            return np.empty((k, 0), dtype=np.float64)
        return np.column_stack([c.distribution for c in self.columns])

    def __repr__(self) -> str:
        return f"ColumnPool(size={self.size})"


# ---------------------------------------------------------------------------
# Reduced cost computation
# ---------------------------------------------------------------------------


class ReducedCostComputation:
    """Compute reduced costs for column selection.

    For the restricted master ``min c^T x  s.t. Ax = b, x >= 0``,
    the reduced cost of a candidate column ``a`` is ``c_a - pi^T a``
    where ``pi`` are the dual variables of the master constraints.

    Attributes:
        objective_coeffs: Per-column objective coefficients template.
    """

    def __init__(self, loss_matrix: npt.NDArray[np.float64]) -> None:
        """
        Args:
            loss_matrix: (n × k) matrix where entry (i, j) is the loss
                incurred when input *i* maps to output bin *j*.
        """
        self.loss_matrix = np.asarray(loss_matrix, dtype=np.float64)

    def compute(
        self,
        column: npt.NDArray[np.float64],
        source_input: int,
        dual_values: npt.NDArray[np.float64],
        constraint_matrix_col: npt.NDArray[np.float64],
    ) -> float:
        """Compute the reduced cost of a candidate column.

        Args:
            column: Candidate output distribution, shape (k,).
            source_input: Input index this column belongs to.
            dual_values: Dual variables of the master constraints.
            constraint_matrix_col: Column of the constraint matrix
                corresponding to this candidate.

        Returns:
            Reduced cost value.
        """
        obj_coeff = float(self.loss_matrix[source_input] @ column)
        return obj_coeff - float(dual_values @ constraint_matrix_col)

    def batch_compute(
        self,
        columns: npt.NDArray[np.float64],
        source_input: int,
        dual_values: npt.NDArray[np.float64],
        constraint_cols: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute reduced costs for a batch of candidate columns.

        Args:
            columns: (k × m) matrix of candidate distributions.
            source_input: Input index.
            dual_values: Dual variables.
            constraint_cols: Constraint matrix columns, shape (n_constraints, m).

        Returns:
            Array of reduced costs, shape (m,).
        """
        obj_coeffs = self.loss_matrix[source_input] @ columns
        dual_part = dual_values @ constraint_cols
        return obj_coeffs - dual_part


# ---------------------------------------------------------------------------
# Pricing oracle
# ---------------------------------------------------------------------------


class PricingOracle:
    """Solve pricing subproblem to find improving columns.

    For each input *i*, the pricing subproblem finds an output distribution
    ``p`` over *k* bins that minimises the reduced cost subject to DP
    constraints involving adjacent inputs.

    Attributes:
        spec: The query specification.
        strategy: Pricing strategy (exact / heuristic / hybrid).
        loss_matrix: Precomputed (n × k) loss matrix.
    """

    def __init__(
        self,
        spec: QuerySpec,
        strategy: PricingStrategy = PricingStrategy.EXACT,
    ) -> None:
        self.spec = spec
        self.strategy = strategy
        n, k = spec.n, spec.k
        self.loss_matrix = self._build_loss_matrix(spec)
        self._rc = ReducedCostComputation(self.loss_matrix)

    # -- public API --

    def solve(
        self,
        dual_values: npt.NDArray[np.float64],
        input_index: int,
        budget: PrivacyBudget,
    ) -> Optional[Column]:
        """Solve pricing for *input_index* given current master duals.

        Returns a Column with negative reduced cost, or None if the current
        restricted master is already optimal for this input.
        """
        if self.strategy == PricingStrategy.EXACT:
            return self._solve_exact(dual_values, input_index, budget)
        elif self.strategy == PricingStrategy.HEURISTIC:
            return self._solve_heuristic(dual_values, input_index, budget)
        else:
            col = self._solve_heuristic(dual_values, input_index, budget)
            if col is None:
                col = self._solve_exact(dual_values, input_index, budget)
            return col

    # -- internal --

    def _solve_exact(
        self,
        dual_values: npt.NDArray[np.float64],
        input_index: int,
        budget: PrivacyBudget,
    ) -> Optional[Column]:
        """Exact pricing via LP."""
        k = self.spec.k
        eps = budget.epsilon
        e_eps = np.exp(eps)

        # Objective: minimise (loss_i - dual contribution) · p
        obj = self.loss_matrix[input_index].copy()
        # Incorporate dual values for the probability-sum constraint
        n = self.spec.n
        if len(dual_values) > 0:
            # Duals for sum-to-one constraints (first n duals)
            if len(dual_values) >= n:
                obj -= dual_values[input_index]
            # Duals for DP constraints
            assert self.spec.edges is not None
            all_edges = self.spec.edges.edges
            dual_offset = n
            for edge_idx, (i, ip) in enumerate(all_edges):
                if i != input_index and ip != input_index:
                    continue
                if dual_offset + edge_idx < len(dual_values):
                    lam = dual_values[dual_offset + edge_idx]
                    if i == input_index:
                        obj -= lam  # p_j contributes to M[i,j] <= e^eps M[i',j]
                    else:
                        obj += lam * e_eps

        # Constraints: p >= 0, sum(p) = 1
        A_eq = np.ones((1, k), dtype=np.float64)
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0)] * k

        res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            return None

        rc = float(res.fun)
        if rc >= -1e-8:
            return None

        dist = np.maximum(res.x, 0.0)
        dist /= dist.sum()
        return Column(
            index=-1,
            distribution=dist,
            reduced_cost=rc,
            source_input=input_index,
        )

    def _solve_heuristic(
        self,
        dual_values: npt.NDArray[np.float64],
        input_index: int,
        budget: PrivacyBudget,
    ) -> Optional[Column]:
        """Heuristic pricing via greedy support selection."""
        k = self.spec.k
        obj = self.loss_matrix[input_index].copy()
        n = self.spec.n
        if len(dual_values) >= n:
            obj -= dual_values[input_index]

        # Pick the k//4 bins with lowest objective contribution
        support_size = max(2, k // 4)
        best_bins = np.argsort(obj)[:support_size]
        dist = np.zeros(k, dtype=np.float64)
        # Assign uniform over selected bins
        dist[best_bins] = 1.0 / support_size

        rc = float(obj @ dist)
        if rc >= -1e-8:
            return None

        return Column(
            index=-1,
            distribution=dist,
            reduced_cost=rc,
            source_input=input_index,
        )

    def _get_adjacent_pairs(self, input_index: int) -> List[Tuple[int, int]]:
        """Return adjacent pairs involving *input_index*."""
        pairs = []
        assert self.spec.edges is not None
        for i, ip in self.spec.edges.edges:
            if i == input_index or ip == input_index:
                pairs.append((i, ip))
        return pairs

    @staticmethod
    def _build_loss_matrix(spec: QuerySpec) -> npt.NDArray[np.float64]:
        """Build (n × k) loss matrix from the query specification."""
        n, k = spec.n, spec.k
        y_min = float(spec.query_values.min()) - spec.sensitivity
        y_max = float(spec.query_values.max()) + spec.sensitivity
        y_grid = np.linspace(y_min, y_max, k)
        loss_fn = spec.get_loss_callable()
        L = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            for j in range(k):
                L[i, j] = loss_fn(spec.query_values[i], y_grid[j])
        return L


# ---------------------------------------------------------------------------
# Master problem
# ---------------------------------------------------------------------------


class MasterProblem:
    """Restricted master LP with privacy constraints.

    The master problem is:
        min  sum_j c_j x_j
        s.t. sum_{j in S_i} x_j = 1  for each input i
             M[i,j] <= e^eps M[i',j]  for adjacent (i,i'), output j
             x_j >= 0

    where S_i is the set of columns for input i.

    Attributes:
        spec: Query specification.
        pool: Column pool backing this master.
    """

    def __init__(self, spec: QuerySpec, pool: ColumnPool) -> None:
        self.spec = spec
        self.pool = pool
        self._last_duals: Optional[npt.NDArray[np.float64]] = None
        self._last_obj: float = np.inf

    def solve(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Solve the restricted master problem.

        Returns:
            Tuple of (primal_solution, dual_values, objective_value).
        """
        n, k = self.spec.n, self.spec.k
        pool = self.pool
        num_cols = pool.size
        if num_cols == 0:
            raise ValueError("Column pool is empty; cannot solve master.")

        # Build objective
        loss_matrix = PricingOracle._build_loss_matrix(self.spec)
        c = np.array([
            float(loss_matrix[col.source_input] @ col.distribution)
            for col in pool.columns
        ], dtype=np.float64)

        # Equality constraints: for each input i, sum of weights = 1
        A_eq_rows, A_eq_cols_idx, A_eq_data = [], [], []
        for i in range(n):
            cols_for_i = [
                pos for pos, col in enumerate(pool.columns)
                if col.source_input == i
            ]
            for pos in cols_for_i:
                A_eq_rows.append(i)
                A_eq_cols_idx.append(pos)
                A_eq_data.append(1.0)
        A_eq = sp_sparse.csr_matrix(
            (A_eq_data, (A_eq_rows, A_eq_cols_idx)),
            shape=(n, num_cols),
        )
        b_eq = np.ones(n, dtype=np.float64)

        # DP inequality constraints: M[i,j] <= e^eps M[i',j]
        # Translates to: for each (i,i') in edges, for each output j:
        #   sum_{col for i} col.dist[j] * x_col
        #     - e^eps * sum_{col for i'} col.dist[j] * x_col' <= 0
        eps = self.spec.epsilon
        e_eps = np.exp(eps)
        assert self.spec.edges is not None
        edges = self.spec.edges.edges
        A_ub_rows, A_ub_cols_idx, A_ub_data = [], [], []
        row_idx = 0
        for i, ip in edges:
            for j in range(k):
                for pos, col in enumerate(pool.columns):
                    if col.source_input == i:
                        A_ub_rows.append(row_idx)
                        A_ub_cols_idx.append(pos)
                        A_ub_data.append(col.distribution[j])
                    elif col.source_input == ip:
                        A_ub_rows.append(row_idx)
                        A_ub_cols_idx.append(pos)
                        A_ub_data.append(-e_eps * col.distribution[j])
                row_idx += 1

        n_ub = row_idx
        if n_ub > 0:
            A_ub = sp_sparse.csr_matrix(
                (A_ub_data, (A_ub_rows, A_ub_cols_idx)),
                shape=(n_ub, num_cols),
            )
            b_ub = np.zeros(n_ub, dtype=np.float64)
        else:
            A_ub = sp_sparse.csr_matrix((0, num_cols))
            b_ub = np.zeros(0, dtype=np.float64)

        bounds = [(0.0, None)] * num_cols

        res = linprog(
            c,
            A_ub=A_ub if n_ub > 0 else None,
            b_ub=b_ub if n_ub > 0 else None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            logger.warning("Master LP failed: %s", res.message)
            duals = np.zeros(n + n_ub, dtype=np.float64)
            return np.zeros(num_cols), duals, np.inf

        primal = res.x
        # Extract duals (equality then inequality)
        eq_duals = np.zeros(n, dtype=np.float64)
        ub_duals = np.zeros(n_ub, dtype=np.float64)
        if hasattr(res, "eqlin") and res.eqlin is not None:
            eq_duals = np.asarray(res.eqlin.marginals, dtype=np.float64)
        if hasattr(res, "ineqlin") and res.ineqlin is not None and n_ub > 0:
            ub_duals = np.asarray(res.ineqlin.marginals, dtype=np.float64)
        dual_values = np.concatenate([eq_duals, ub_duals])

        self._last_duals = dual_values
        self._last_obj = float(res.fun)
        return primal, dual_values, float(res.fun)

    @property
    def last_duals(self) -> Optional[npt.NDArray[np.float64]]:
        """Dual values from the most recent solve."""
        return self._last_duals

    @property
    def last_objective(self) -> float:
        """Objective value from the most recent solve."""
        return self._last_obj


# ---------------------------------------------------------------------------
# Dantzig-Wolfe decomposition
# ---------------------------------------------------------------------------


class DantzigWolfeDecomposition:
    """Dantzig-Wolfe decomposition for block-angular DP mechanism LP.

    Decomposes the mechanism LP into per-input sub-blocks linked by DP
    coupling constraints, then applies column generation on the convexified
    master.

    Attributes:
        spec: Query specification.
        config: Sparse configuration.
        pool: Column pool.
        master: Master problem.
        oracle: Pricing oracle.
    """

    def __init__(self, spec: QuerySpec, config: Optional[SparseConfig] = None) -> None:
        self.spec = spec
        self.config = config or SparseConfig()
        self.pool = ColumnPool()
        self._initialise_pool()
        self.master = MasterProblem(spec, self.pool)
        self.oracle = PricingOracle(spec, self.config.pricing_strategy)
        self.convergence_history: List[float] = []

    def _initialise_pool(self) -> None:
        """Seed the pool with one uniform column per input."""
        k = self.spec.k
        uniform = np.ones(k, dtype=np.float64) / k
        for i in range(self.spec.n):
            self.pool.add(uniform, source_input=i)

    def solve(self, max_iterations: Optional[int] = None) -> SparseResult:
        """Run the Dantzig-Wolfe column generation loop.

        Args:
            max_iterations: Override for max iterations.

        Returns:
            SparseResult with the optimised mechanism.
        """
        max_iter = max_iterations or self.config.max_iterations
        budget = PrivacyBudget(epsilon=self.spec.epsilon, delta=self.spec.delta)
        best_obj = np.inf

        for iteration in range(max_iter):
            # Solve restricted master
            primal, duals, obj_val = self.master.solve()
            self.convergence_history.append(obj_val)
            best_obj = min(best_obj, obj_val)

            if self.config.verbose >= 2:
                logger.info(
                    "DW iter %d: obj=%.6f, pool=%d",
                    iteration, obj_val, self.pool.size,
                )

            # Pricing: find improving columns
            found_improving = False
            for i in range(self.spec.n):
                col = self.oracle.solve(duals, i, budget)
                if col is not None:
                    self.pool.add(col.distribution, col.source_input, col.reduced_cost)
                    found_improving = True

            if not found_improving:
                if self.config.verbose >= 1:
                    logger.info(
                        "DW converged at iteration %d, obj=%.6f", iteration, obj_val
                    )
                break

            # Enforce column limit
            if self.pool.size > self.config.max_columns:
                self._prune_pool(primal)

        mechanism = self._extract_mechanism(primal)
        n, k = self.spec.n, self.spec.k
        return SparseResult(
            mechanism=mechanism,
            active_columns=int(np.sum(primal > 1e-8)) if primal is not None else 0,
            total_possible=n * k,
            iterations=len(self.convergence_history),
            obj_val=best_obj,
            lower_bound=best_obj,
            upper_bound=best_obj,
            convergence_history=self.convergence_history,
        )

    def _prune_pool(self, primal: npt.NDArray[np.float64]) -> None:
        """Remove columns with zero primal weight."""
        keep = {
            self.pool.columns[j].index
            for j in range(len(primal))
            if primal[j] > 1e-10
        }
        # Always keep at least one per input
        for i in range(self.spec.n):
            cols = self.pool.get_columns_for_input(i)
            if cols and not any(c.index in keep for c in cols):
                keep.add(cols[0].index)
        self.pool.prune(keep)

    def _extract_mechanism(self, primal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Build the n × k mechanism matrix from column weights."""
        n, k = self.spec.n, self.spec.k
        M = np.zeros((n, k), dtype=np.float64)
        for j, col in enumerate(self.pool.columns):
            if j < len(primal) and primal[j] > 1e-12:
                M[col.source_input] += primal[j] * col.distribution
        # Normalise rows
        for i in range(n):
            s = M[i].sum()
            if s > 0:
                M[i] /= s
        return M


# ---------------------------------------------------------------------------
# Branch-and-price
# ---------------------------------------------------------------------------


@dataclass
class BranchNode:
    """Node in the branch-and-price tree.

    Attributes:
        lower_bound: LP relaxation bound at this node.
        fixed_zero: Column indices fixed to zero.
        fixed_one: Column indices fixed to one.
        depth: Depth in the branch tree.
    """
    lower_bound: float = np.inf
    fixed_zero: List[int] = field(default_factory=list)
    fixed_one: List[int] = field(default_factory=list)
    depth: int = 0


class BranchAndPrice:
    """Branch-and-price for integer mechanism design.

    Combines column generation at each node with branching on fractional
    column variables to find integer-optimal mechanisms.

    Attributes:
        spec: Query specification.
        config: Sparse configuration.
        best_integer_obj: Best integer objective found so far.
        best_mechanism: Best integer mechanism found.
    """

    def __init__(self, spec: QuerySpec, config: Optional[SparseConfig] = None) -> None:
        self.spec = spec
        self.config = config or SparseConfig()
        self.best_integer_obj: float = np.inf
        self.best_mechanism: Optional[npt.NDArray[np.float64]] = None
        self._nodes_explored: int = 0

    def solve(self, max_nodes: int = 100) -> SparseResult:
        """Run branch-and-price search.

        Args:
            max_nodes: Maximum number of branch tree nodes to explore.

        Returns:
            SparseResult with the best mechanism found.
        """
        root = BranchNode()
        stack: List[BranchNode] = [root]
        convergence: List[float] = []

        while stack and self._nodes_explored < max_nodes:
            node = stack.pop()
            self._nodes_explored += 1

            # Solve LP relaxation at this node
            dw = DantzigWolfeDecomposition(self.spec, self.config)
            result = dw.solve(max_iterations=self.config.max_iterations // 2)

            node.lower_bound = result.obj_val
            convergence.append(result.obj_val)

            # Prune
            if node.lower_bound >= self.best_integer_obj - 1e-8:
                continue

            # Check integrality
            mechanism = result.mechanism
            is_integer = self._check_integrality(mechanism)

            if is_integer:
                if result.obj_val < self.best_integer_obj:
                    self.best_integer_obj = result.obj_val
                    self.best_mechanism = mechanism.copy()
                continue

            # Branch on most fractional variable
            frac_i, frac_j = self._find_branching_variable(mechanism)
            # Left child: fix (i, j) entry towards 0
            left = BranchNode(
                fixed_zero=node.fixed_zero + [(frac_i, frac_j)],
                fixed_one=node.fixed_one.copy(),
                depth=node.depth + 1,
            )
            # Right child: fix (i, j) entry towards 1
            right = BranchNode(
                fixed_zero=node.fixed_zero.copy(),
                fixed_one=node.fixed_one + [(frac_i, frac_j)],
                depth=node.depth + 1,
            )
            stack.extend([left, right])

        if self.best_mechanism is None:
            self.best_mechanism = np.ones((self.spec.n, self.spec.k)) / self.spec.k

        n, k = self.spec.n, self.spec.k
        return SparseResult(
            mechanism=self.best_mechanism,
            active_columns=int(np.count_nonzero(self.best_mechanism > 1e-8)),
            total_possible=n * k,
            iterations=self._nodes_explored,
            obj_val=self.best_integer_obj if np.isfinite(self.best_integer_obj) else 0.0,
            lower_bound=min(convergence) if convergence else 0.0,
            upper_bound=self.best_integer_obj if np.isfinite(self.best_integer_obj) else np.inf,
            convergence_history=convergence,
        )

    def _check_integrality(self, mechanism: npt.NDArray[np.float64], tol: float = 1e-4) -> bool:
        """Check if mechanism entries are near-integer (0 or sparse support)."""
        # In the DP mechanism context, integrality means each row has
        # support on at most a small number of outputs.
        for i in range(mechanism.shape[0]):
            nonzero = np.sum(mechanism[i] > tol)
            if nonzero > 1 and np.any((mechanism[i] > tol) & (mechanism[i] < 1.0 - tol)):
                return False
        return True

    def _find_branching_variable(self, mechanism: npt.NDArray[np.float64]) -> Tuple[int, int]:
        """Find the most fractional entry in the mechanism matrix."""
        frac = np.abs(mechanism - 0.5)
        # Mask out entries already near 0 or 1
        mask = (mechanism > 1e-4) & (mechanism < 1.0 - 1e-4)
        if not np.any(mask):
            return (0, 0)
        frac[~mask] = 1.0
        idx = np.argmin(frac)
        return (int(idx // mechanism.shape[1]), int(idx % mechanism.shape[1]))
