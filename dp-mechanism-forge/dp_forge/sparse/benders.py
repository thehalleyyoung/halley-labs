"""Benders decomposition for DP mechanism synthesis.

Decomposes the mechanism LP into a master problem (mechanism structure
variables) and per-pair subproblems (privacy feasibility checks).
Supports multi-cut and trust-region stabilisation variants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp_sparse
from scipy.optimize import linprog

from dp_forge.sparse import (
    BendersCut,
    CutType,
    DecompositionType,
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
# Helpers
# ---------------------------------------------------------------------------


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
# Feasibility cut
# ---------------------------------------------------------------------------


class FeasibilityCut:
    """Generate Benders feasibility cuts from infeasible subproblems.

    When the master solution violates a privacy constraint for adjacent
    pair (i, i'), we derive a cut from the Farkas dual of the infeasible
    subproblem and add it to the master.

    Attributes:
        spec: Query specification.
    """

    def __init__(self, spec: QuerySpec) -> None:
        self.spec = spec

    def generate(
        self,
        master_solution: npt.NDArray[np.float64],
        pair: Tuple[int, int],
        budget: PrivacyBudget,
    ) -> Optional[BendersCut]:
        """Generate a feasibility cut for the given adjacent pair.

        Args:
            master_solution: Current master variable values, shape (n, k).
            pair: Adjacent pair (i, i').
            budget: Privacy budget.

        Returns:
            A BendersCut of type FEASIBILITY, or None if feasible.
        """
        i, ip = pair
        k = self.spec.k
        e_eps = np.exp(budget.epsilon)
        M = master_solution

        # Check each output bin for DP violation
        violations = np.zeros(k, dtype=np.float64)
        for j in range(k):
            v = M[i, j] - e_eps * M[ip, j]
            violations[j] = max(0.0, v)

        max_violation = float(violations.max())
        if max_violation <= 1e-10:
            return None

        # Farkas-type cut: the direction of maximal violation
        # Coefficient for M[i, j]: +1 for violating j
        # Coefficient for M[i', j]: -e^eps for violating j
        n = self.spec.n
        coeffs = np.zeros((n, k), dtype=np.float64)
        for j in range(k):
            if violations[j] > 1e-12:
                coeffs[i, j] += 1.0
                coeffs[ip, j] -= e_eps

        coeffs_flat = coeffs.ravel()
        rhs = 0.0

        return BendersCut(
            cut_type=CutType.FEASIBILITY,
            coefficients=coeffs_flat,
            rhs=rhs,
            subproblem_pair=pair,
        )


# ---------------------------------------------------------------------------
# Optimality cut
# ---------------------------------------------------------------------------


class OptimalityCut:
    """Generate Benders optimality cuts.

    When the subproblem is feasible, the dual of the subproblem provides
    an optimality cut that tightens the master's approximation of the
    recourse function.

    Attributes:
        spec: Query specification.
        loss_matrix: Precomputed (n × k) loss matrix.
    """

    def __init__(self, spec: QuerySpec) -> None:
        self.spec = spec
        self.loss_matrix = _build_loss_matrix(spec)

    def generate(
        self,
        master_solution: npt.NDArray[np.float64],
        pair: Tuple[int, int],
        budget: PrivacyBudget,
        subproblem_dual: npt.NDArray[np.float64],
    ) -> Optional[BendersCut]:
        """Generate an optimality cut from subproblem dual solution.

        Args:
            master_solution: Current master variable values, shape (n, k).
            pair: Adjacent pair (i, i').
            budget: Privacy budget.
            subproblem_dual: Dual variables from the subproblem.

        Returns:
            A BendersCut of type OPTIMALITY, or None if not useful.
        """
        i, ip = pair
        k = self.spec.k
        e_eps = np.exp(budget.epsilon)
        n = self.spec.n

        # Build optimality cut coefficients from dual information
        coeffs = np.zeros(n * k, dtype=np.float64)

        # The cut approximates the recourse cost as a function of
        # master variables: theta >= dual^T (b - T x)
        for j in range(k):
            dual_j = subproblem_dual[j] if j < len(subproblem_dual) else 0.0
            coeffs[i * k + j] += dual_j
            coeffs[ip * k + j] -= dual_j * e_eps

        rhs = float(subproblem_dual.sum()) if len(subproblem_dual) > 0 else 0.0

        # Only add if it actually cuts
        current_lhs = float(coeffs @ master_solution.ravel())
        if current_lhs <= rhs + 1e-8:
            return None

        return BendersCut(
            cut_type=CutType.OPTIMALITY,
            coefficients=coeffs,
            rhs=rhs,
            subproblem_pair=pair,
        )


# ---------------------------------------------------------------------------
# Benders subproblem
# ---------------------------------------------------------------------------


class BendersSubproblem:
    """Continuous subproblem for privacy feasibility.

    For a given master solution (mechanism matrix M) and adjacent pair
    (i, i'), checks whether the DP constraint M[i,j] <= e^eps M[i',j]
    holds for all output bins j. Returns feasibility/optimality cuts.

    Attributes:
        spec: Query specification.
        feasibility_gen: Feasibility cut generator.
        optimality_gen: Optimality cut generator.
    """

    def __init__(self, spec: QuerySpec) -> None:
        self.spec = spec
        self.feasibility_gen = FeasibilityCut(spec)
        self.optimality_gen = OptimalityCut(spec)

    def solve(
        self,
        master_solution: npt.NDArray[np.float64],
        pair: Tuple[int, int],
        budget: PrivacyBudget,
    ) -> Tuple[bool, Optional[BendersCut]]:
        """Solve the subproblem for an adjacent pair.

        Args:
            master_solution: Mechanism matrix, shape (n, k).
            pair: Adjacent pair (i, i').
            budget: Privacy budget.

        Returns:
            Tuple of (is_feasible, cut_or_none).
        """
        i, ip = pair
        k = self.spec.k
        e_eps = np.exp(budget.epsilon)
        M = master_solution

        # Check feasibility: M[i, j] <= e^eps M[i', j] for all j
        max_violation = 0.0
        for j in range(k):
            violation = M[i, j] - e_eps * M[ip, j]
            max_violation = max(max_violation, violation)

        if max_violation > 1e-8:
            cut = self.feasibility_gen.generate(master_solution, pair, budget)
            return False, cut

        # Feasible: generate optimality cut from pseudo-dual
        # The pseudo-dual captures how tight the constraints are
        slack = np.zeros(k, dtype=np.float64)
        for j in range(k):
            slack[j] = e_eps * M[ip, j] - M[i, j]
        # Dual approximation: inverse of slack (tighter constraints get higher dual)
        pseudo_dual = np.zeros(k, dtype=np.float64)
        tight_mask = slack < 1e-6
        if np.any(tight_mask):
            pseudo_dual[tight_mask] = 1.0 / np.sum(tight_mask)

        cut = self.optimality_gen.generate(
            master_solution, pair, budget, pseudo_dual
        )
        return True, cut


# ---------------------------------------------------------------------------
# Benders master
# ---------------------------------------------------------------------------


class BendersMaster:
    """Master problem with mechanism structure variables.

    The master chooses the mechanism matrix M (n × k) and an epigraph
    variable θ approximating the recourse, subject to row-stochasticity
    and accumulated Benders cuts.

    Attributes:
        spec: Query specification.
        cuts: Accumulated Benders cuts.
        loss_matrix: Precomputed loss matrix.
    """

    def __init__(self, spec: QuerySpec) -> None:
        self.spec = spec
        self.cuts: List[BendersCut] = []
        self.loss_matrix = _build_loss_matrix(spec)
        self._last_obj: float = np.inf

    def add_cut(self, cut: BendersCut) -> None:
        """Add a Benders cut to the master."""
        self.cuts.append(cut)

    def solve(self) -> Tuple[npt.NDArray[np.float64], float]:
        """Solve the master problem.

        Returns:
            Tuple of (mechanism matrix shape (n, k), objective value).
        """
        n, k = self.spec.n, self.spec.k
        n_mech = n * k  # mechanism variables
        # x = [M_flat, theta] where theta is the epigraph variable
        n_vars = n_mech + 1

        # Objective: minimise sum L[i,j] M[i,j] + theta
        c = np.zeros(n_vars, dtype=np.float64)
        c[:n_mech] = self.loss_matrix.ravel()
        c[n_mech] = 1.0  # theta

        # Equality: sum_j M[i,j] = 1 for each i
        A_eq_rows, A_eq_cols, A_eq_data = [], [], []
        for i in range(n):
            for j in range(k):
                A_eq_rows.append(i)
                A_eq_cols.append(i * k + j)
                A_eq_data.append(1.0)
        A_eq = sp_sparse.csr_matrix(
            (A_eq_data, (A_eq_rows, A_eq_cols)),
            shape=(n, n_vars),
        )
        b_eq = np.ones(n, dtype=np.float64)

        # Inequality: Benders cuts
        # Each cut: coeffs^T M_flat <= rhs  or  theta >= coeffs^T M_flat - rhs
        # Reformulate as: -coeffs^T M_flat + theta >= -rhs
        # => coeffs^T M_flat - theta <= rhs
        n_cuts = len(self.cuts)
        A_ub_rows, A_ub_cols, A_ub_data = [], [], []
        b_ub_list = []
        for ci, cut in enumerate(self.cuts):
            coeffs = cut.coefficients
            if len(coeffs) == n_mech:
                for idx in range(n_mech):
                    if abs(coeffs[idx]) > 1e-15:
                        A_ub_rows.append(ci)
                        A_ub_cols.append(idx)
                        A_ub_data.append(coeffs[idx])
                if cut.cut_type == CutType.OPTIMALITY:
                    A_ub_rows.append(ci)
                    A_ub_cols.append(n_mech)  # -theta
                    A_ub_data.append(-1.0)
                b_ub_list.append(cut.rhs)
            else:
                b_ub_list.append(0.0)

        if n_cuts > 0:
            A_ub = sp_sparse.csr_matrix(
                (A_ub_data, (A_ub_rows, A_ub_cols)),
                shape=(n_cuts, n_vars),
            )
            b_ub = np.array(b_ub_list, dtype=np.float64)
        else:
            A_ub = None
            b_ub = None

        bounds = [(0.0, 1.0)] * n_mech + [(0.0, None)]

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            logger.warning("Benders master LP failed: %s", res.message)
            M = np.ones((n, k), dtype=np.float64) / k
            return M, np.inf

        M = res.x[:n_mech].reshape(n, k)
        M = np.maximum(M, 0.0)
        for i in range(n):
            s = M[i].sum()
            if s > 0:
                M[i] /= s
        self._last_obj = float(res.fun)
        return M, float(res.fun)


# ---------------------------------------------------------------------------
# Multi-cut Benders
# ---------------------------------------------------------------------------


class MultiCutBenders:
    """Multi-cut Benders decomposition for faster convergence.

    Instead of aggregating all subproblem information into a single cut
    per iteration, adds one cut per adjacent pair per iteration. This
    provides richer information to the master and typically converges
    in fewer iterations.

    Attributes:
        spec: Query specification.
        config: Sparse configuration.
        master: Benders master problem.
        subproblem: Subproblem solver.
    """

    def __init__(self, spec: QuerySpec, config: Optional[SparseConfig] = None) -> None:
        self.spec = spec
        self.config = config or SparseConfig(decomposition_type=DecompositionType.BENDERS)
        self.master = BendersMaster(spec)
        self.subproblem = BendersSubproblem(spec)
        self.convergence_history: List[float] = []

    def solve(self) -> SparseResult:
        """Run multi-cut Benders decomposition.

        Returns:
            SparseResult with the optimised mechanism.
        """
        max_iter = self.config.max_iterations
        budget = PrivacyBudget(epsilon=self.spec.epsilon, delta=self.spec.delta)
        assert self.spec.edges is not None
        edges = self.spec.edges.edges

        best_ub = np.inf
        best_mechanism = None

        for iteration in range(max_iter):
            # Solve master
            M, master_obj = self.master.solve()
            self.convergence_history.append(master_obj)

            # Solve all subproblems and add cuts
            all_feasible = True
            n_cuts_added = 0

            for pair in edges:
                feasible, cut = self.subproblem.solve(M, pair, budget)
                if not feasible:
                    all_feasible = False
                if cut is not None and n_cuts_added < self.config.max_cuts:
                    self.master.add_cut(cut)
                    n_cuts_added += 1

            if all_feasible:
                obj_val = float(self.master.loss_matrix.ravel() @ M.ravel())
                if obj_val < best_ub:
                    best_ub = obj_val
                    best_mechanism = M.copy()

            if self.config.verbose >= 2:
                logger.info(
                    "Benders iter %d: master=%.6f, feasible=%s, cuts=%d",
                    iteration, master_obj, all_feasible, n_cuts_added,
                )

            # Convergence check
            if all_feasible and n_cuts_added == 0:
                if self.config.verbose >= 1:
                    logger.info(
                        "Benders converged at iteration %d", iteration
                    )
                break

            if len(self.master.cuts) >= self.config.max_cuts:
                if self.config.verbose >= 1:
                    logger.info("Benders: max cuts reached (%d)", self.config.max_cuts)
                break

        if best_mechanism is None:
            best_mechanism = M
            best_ub = float(self.master.loss_matrix.ravel() @ M.ravel())

        n, k = self.spec.n, self.spec.k
        lb = min(self.convergence_history) if self.convergence_history else 0.0
        return SparseResult(
            mechanism=best_mechanism,
            active_columns=int(np.count_nonzero(best_mechanism > 1e-8)),
            total_possible=n * k,
            iterations=len(self.convergence_history),
            obj_val=best_ub,
            lower_bound=lb,
            upper_bound=best_ub,
            convergence_history=self.convergence_history,
        )


# ---------------------------------------------------------------------------
# Trust-region Benders
# ---------------------------------------------------------------------------


class TrustRegionBenders:
    """Stabilised Benders decomposition with trust regions.

    Adds a trust region around the incumbent solution to prevent
    large oscillations in the master solution between iterations.
    The trust region radius is dynamically adjusted based on
    predicted vs. actual improvement.

    Attributes:
        spec: Query specification.
        config: Sparse configuration.
        trust_radius: Current trust region radius.
        incumbent: Best feasible mechanism found so far.
    """

    def __init__(
        self,
        spec: QuerySpec,
        config: Optional[SparseConfig] = None,
        initial_radius: float = 0.5,
    ) -> None:
        self.spec = spec
        self.config = config or SparseConfig(decomposition_type=DecompositionType.BENDERS)
        self.trust_radius = initial_radius
        self.incumbent: Optional[npt.NDArray[np.float64]] = None
        self.master = BendersMaster(spec)
        self.subproblem = BendersSubproblem(spec)
        self.convergence_history: List[float] = []
        self._min_radius = 0.01
        self._max_radius = 2.0

    def solve(self) -> SparseResult:
        """Run trust-region stabilised Benders.

        Returns:
            SparseResult with the optimised mechanism.
        """
        max_iter = self.config.max_iterations
        budget = PrivacyBudget(epsilon=self.spec.epsilon, delta=self.spec.delta)
        assert self.spec.edges is not None
        edges = self.spec.edges.edges

        # Initialise incumbent with uniform mechanism
        n, k = self.spec.n, self.spec.k
        self.incumbent = np.ones((n, k), dtype=np.float64) / k
        best_ub = float(self.master.loss_matrix.ravel() @ self.incumbent.ravel())

        for iteration in range(max_iter):
            # Solve master with trust region
            M, master_obj = self._solve_with_trust_region()
            self.convergence_history.append(master_obj)

            # Evaluate subproblems
            all_feasible = True
            n_cuts_added = 0

            for pair in edges:
                feasible, cut = self.subproblem.solve(M, pair, budget)
                if not feasible:
                    all_feasible = False
                if cut is not None and n_cuts_added < self.config.max_cuts:
                    self.master.add_cut(cut)
                    n_cuts_added += 1

            # Update incumbent and trust region
            if all_feasible:
                obj_val = float(self.master.loss_matrix.ravel() @ M.ravel())
                predicted_improvement = best_ub - master_obj
                actual_improvement = best_ub - obj_val

                if actual_improvement > 0:
                    self.incumbent = M.copy()
                    best_ub = obj_val

                    # Adjust trust radius based on improvement ratio
                    if predicted_improvement > 1e-10:
                        rho = actual_improvement / predicted_improvement
                        if rho > 0.75:
                            self.trust_radius = min(
                                self.trust_radius * 1.5, self._max_radius
                            )
                        elif rho < 0.25:
                            self.trust_radius = max(
                                self.trust_radius * 0.5, self._min_radius
                            )
                else:
                    self.trust_radius = max(
                        self.trust_radius * 0.5, self._min_radius
                    )

            if self.config.verbose >= 2:
                logger.info(
                    "TR-Benders iter %d: obj=%.6f, radius=%.4f, feasible=%s",
                    iteration, master_obj, self.trust_radius, all_feasible,
                )

            # Convergence check
            if all_feasible and n_cuts_added == 0:
                break

        assert self.incumbent is not None
        lb = min(self.convergence_history) if self.convergence_history else 0.0
        return SparseResult(
            mechanism=self.incumbent,
            active_columns=int(np.count_nonzero(self.incumbent > 1e-8)),
            total_possible=n * k,
            iterations=len(self.convergence_history),
            obj_val=best_ub,
            lower_bound=lb,
            upper_bound=best_ub,
            convergence_history=self.convergence_history,
        )

    def _solve_with_trust_region(self) -> Tuple[npt.NDArray[np.float64], float]:
        """Solve the master with added trust-region constraints."""
        n, k = self.spec.n, self.spec.k
        n_mech = n * k
        n_vars = n_mech + 1

        # Build base master LP
        c = np.zeros(n_vars, dtype=np.float64)
        c[:n_mech] = self.master.loss_matrix.ravel()
        c[n_mech] = 1.0

        # Equality: row stochasticity
        A_eq_rows, A_eq_cols, A_eq_data = [], [], []
        for i in range(n):
            for j in range(k):
                A_eq_rows.append(i)
                A_eq_cols.append(i * k + j)
                A_eq_data.append(1.0)
        A_eq = sp_sparse.csr_matrix(
            (A_eq_data, (A_eq_rows, A_eq_cols)),
            shape=(n, n_vars),
        )
        b_eq = np.ones(n, dtype=np.float64)

        # Benders cuts + trust region bounds
        cuts = self.master.cuts
        n_cuts = len(cuts)

        A_ub_rows, A_ub_cols, A_ub_data = [], [], []
        b_ub_list = []
        for ci, cut in enumerate(cuts):
            coeffs = cut.coefficients
            if len(coeffs) == n_mech:
                for idx in range(n_mech):
                    if abs(coeffs[idx]) > 1e-15:
                        A_ub_rows.append(ci)
                        A_ub_cols.append(idx)
                        A_ub_data.append(coeffs[idx])
                if cut.cut_type == CutType.OPTIMALITY:
                    A_ub_rows.append(ci)
                    A_ub_cols.append(n_mech)
                    A_ub_data.append(-1.0)
                b_ub_list.append(cut.rhs)
            else:
                b_ub_list.append(0.0)

        # Trust region: |M - incumbent| <= radius (box constraint)
        assert self.incumbent is not None
        inc_flat = self.incumbent.ravel()
        lb = np.maximum(inc_flat - self.trust_radius, 0.0)
        ub = np.minimum(inc_flat + self.trust_radius, 1.0)
        bounds = [(float(lb[j]), float(ub[j])) for j in range(n_mech)]
        bounds.append((0.0, None))  # theta

        if n_cuts > 0:
            A_ub = sp_sparse.csr_matrix(
                (A_ub_data, (A_ub_rows, A_ub_cols)),
                shape=(n_cuts, n_vars),
            )
            b_ub = np.array(b_ub_list, dtype=np.float64)
        else:
            A_ub = None
            b_ub = None

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            return self.incumbent.copy(), np.inf

        M = res.x[:n_mech].reshape(n, k)
        M = np.maximum(M, 0.0)
        for i in range(n):
            s = M[i].sum()
            if s > 0:
                M[i] /= s
        return M, float(res.fun)
