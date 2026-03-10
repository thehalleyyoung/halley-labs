"""
ILP formulation for exact robustness radius computation (ALG 4).

Encodes the minimum-edit problem as an integer linear program using
*python-mip*.  Decision variables are binary indicators for each candidate
edge edit; constraints enforce DAG acyclicity and CI-consistency of the
perturbed graph.

Variables
---------
* ``x[i,j]``  in {0,1}:  1 iff edge i->j present in the target DAG.
* ``t[i]``  in {0,...,n-1}:  topological-order variable for node i.

Objective
---------
Minimize total edit cost (derived from comparing x[i,j] with the
original adjacency matrix).

Constraints
-----------
1. Acyclicity:  ``t[j] - t[i] >= 1 - n*(1-x[i,j])``
2. Mutual exclusion:  ``x[i,j] + x[j,i] <= 1``
3. Budget:  total edits <= max_k
4. CI consistency (optional):  preserve / break d-separation relations.
5. Conclusion negation (lazy):  no-good cuts added iteratively.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np

from causalcert.exceptions import InfeasibleError, TimeLimitError
from causalcert.types import (
    AdjacencyMatrix,
    CITestResult,
    ConclusionPredicate,
    EditType,
    NodeId,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

logger = logging.getLogger(__name__)

_MIP_AVAILABLE = True
try:
    import mip as _mip
except ImportError:
    _MIP_AVAILABLE = False
    _mip = None  # type: ignore[assignment]


def _require_mip() -> None:
    if not _MIP_AVAILABLE:
        raise ImportError(
            "python-mip is required for ILP solving. Install with: pip install mip"
        )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class ILPSolver:
    """Exact ILP solver for the robustness radius (ALG 4).

    Parameters
    ----------
    time_limit_s : float
        Maximum solver wall-clock time in seconds.
    gap_tolerance : float
        Relative MIP gap tolerance for early termination.
    threads : int
        Number of solver threads.
    verbose : bool
        Whether to print solver logs.
    warm_start : dict | None
        Warm-start values ``{(i,j): 0/1}`` for the edge decision variables.
    """

    def __init__(
        self,
        time_limit_s: float = 300.0,
        gap_tolerance: float = 1e-4,
        threads: int = 1,
        verbose: bool = False,
    ) -> None:
        self.time_limit_s = time_limit_s
        self.gap_tolerance = gap_tolerance
        self.threads = threads
        self.verbose = verbose
        self._warm_start: dict[tuple[int, int], float] | None = None
        self._last_model: Any = None
        self._solution_cache: dict[bytes, bool] = {}

    # ---- warm-start -------------------------------------------------------

    def set_warm_start(self, edge_vals: dict[tuple[int, int], float]) -> None:
        """Set warm-start values for the next solve call.

        Parameters
        ----------
        edge_vals : dict
            Mapping ``(i,j) -> 0/1`` for edge decision variables.
        """
        self._warm_start = dict(edge_vals)

    def set_warm_start_from_adj(self, adj: AdjacencyMatrix) -> None:
        """Set warm-start from an adjacency matrix."""
        n = adj.shape[0]
        ws: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    ws[i, j] = float(adj[i, j])
        self._warm_start = ws

    # ---- main solve -------------------------------------------------------

    def solve(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int = 10,
        ci_results: Sequence[CITestResult] | None = None,
    ) -> RobustnessRadius:
        """Solve the ILP for the exact robustness radius.

        Uses iterative re-solving with lazy no-good cuts:

        1. Build ILP with acyclicity + budget constraints.
        2. Optionally add CI-consistency constraints.
        3. Solve the MIP.
        4. If the predicate still holds on the candidate solution, add a
           no-good cut excluding that DAG and re-solve.
        5. Repeat until the predicate is overturned or infeasibility is proven.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG adjacency matrix.
        predicate : ConclusionPredicate
            Conclusion to stress-test.
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome variables.
        max_k : int
            Maximum edit distance.
        ci_results : Sequence[CITestResult] | None
            Pre-computed CI test results for constraint generation.

        Returns
        -------
        RobustnessRadius
        """
        _require_mip()
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        t0 = time.perf_counter()

        # Trivial case: predicate already fails on original DAG
        if not predicate(adj, data, treatment=treatment, outcome=outcome):
            return RobustnessRadius(
                lower_bound=0,
                upper_bound=0,
                witness_edits=(),
                solver_strategy=SolverStrategy.ILP,
                solver_time_s=time.perf_counter() - t0,
                gap=0.0,
                certified=True,
            )

        # Build base model
        model = self._build_model(adj, max_k)
        self._last_model = model

        # Add CI-consistency constraints if available
        if ci_results:
            from causalcert.solver.constraints import CIConsistencyConstraint

            CIConsistencyConstraint(ci_results, treatment, outcome).add_to_model(
                model, adj
            )

        # Add valid inequalities for small graphs
        if n <= 20:
            from causalcert.solver.constraints import ValidInequalityTightener

            tightener = ValidInequalityTightener(adj)
            tightener.add_clique_cuts(model)
            tightener.add_degree_cuts(model)

        # Set up lazy callback helper
        from causalcert.solver.constraints import (
            ConclusionNegationConstraint,
            LazyConstraintCallback,
        )

        lazy_cb = LazyConstraintCallback(adj, predicate, data, treatment, outcome)

        # Apply warm-start
        if self._warm_start is not None:
            x = model._x_vars
            start_pairs = []
            for (i, j), val in self._warm_start.items():
                if (i, j) in x:
                    start_pairs.append((x[i, j], val))
            if start_pairs:
                model.start = start_pairs

        # Iterative solving with no-good cuts
        best_edits: list[StructuralEdit] | None = None
        best_cost = max_k + 1
        lp_lower_bound = 0
        max_iters = min(2000, 3 ** n)

        for iteration in range(max_iters):
            elapsed = time.perf_counter() - t0
            if elapsed >= self.time_limit_s:
                logger.info("ILP time limit after %d iterations", iteration)
                break

            remaining = max(1.0, self.time_limit_s - elapsed)
            status = model.optimize(max_seconds=remaining)

            # Update lower bound from MIP bound
            if model.objective_bound is not None:
                ilp_lb = int(np.ceil(model.objective_bound - 1e-6))
                lp_lower_bound = max(lp_lower_bound, ilp_lb)

            if status == _mip.OptimizationStatus.INFEASIBLE:
                logger.info(
                    "ILP infeasible after %d no-good cuts (iteration %d)",
                    lazy_cb.n_cuts,
                    iteration,
                )
                break

            if status == _mip.OptimizationStatus.NO_SOLUTION_FOUND:
                logger.info("No ILP solution within time budget")
                break

            if status not in (
                _mip.OptimizationStatus.OPTIMAL,
                _mip.OptimizationStatus.FEASIBLE,
            ):
                logger.warning("ILP unexpected status: %s", status)
                break

            # Check solution against conclusion predicate
            overturned = lazy_cb.check_and_cut(model)

            if overturned:
                from causalcert.solver.constraints import extract_edits_from_model

                edits = extract_edits_from_model(model, adj)
                cost = len(edits)
                if cost < best_cost:
                    best_cost = cost
                    best_edits = edits
                    logger.info(
                        "ILP found witness with %d edits (iteration %d)",
                        cost,
                        iteration,
                    )
                # Try to find a better solution by adding a budget cut
                if cost > lp_lower_bound:
                    from causalcert.solver.constraints import (
                        build_edit_cost_expression,
                    )

                    cost_expr = build_edit_cost_expression(model, adj)
                    model += cost_expr <= cost - 1, f"improve_{iteration}"
                    continue
                break
            # else: no-good cut was added, re-solve

        elapsed = time.perf_counter() - t0

        if best_edits is not None:
            lb = max(lp_lower_bound, best_cost)
            return RobustnessRadius(
                lower_bound=lb,
                upper_bound=best_cost,
                witness_edits=tuple(best_edits),
                solver_strategy=SolverStrategy.ILP,
                solver_time_s=elapsed,
                gap=(best_cost - lb) / max(best_cost, 1),
                certified=(lb == best_cost),
            )
        else:
            lb = max(lp_lower_bound, max_k + 1)
            return RobustnessRadius(
                lower_bound=lb,
                upper_bound=max_k + 1,
                witness_edits=(),
                solver_strategy=SolverStrategy.ILP,
                solver_time_s=elapsed,
                gap=0.0,
                certified=True,
            )

    # ---- model construction -----------------------------------------------

    def _build_model(
        self,
        adj: AdjacencyMatrix,
        max_k: int,
    ) -> Any:
        """Construct the python-mip model with variables and base constraints.

        Creates:
        * Edge decision variables ``x[i,j]`` (binary).
        * Topological-order variables ``t[i]`` (integer, 0..n-1).
        * Acyclicity constraints.
        * Mutual-exclusion constraints.
        * Budget constraint.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        max_k : int
            Maximum number of edits.

        Returns
        -------
        mip.Model
        """
        _require_mip()
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]

        model = _mip.Model(name="robustness_radius", sense=_mip.MINIMIZE)
        model.verbose = int(self.verbose)
        model.threads = self.threads
        model.max_mip_gap = self.gap_tolerance

        # ---- decision variables ----
        x_vars: dict[tuple[int, int], Any] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x_vars[i, j] = model.add_var(
                    name=f"x_{i}_{j}",
                    var_type=_mip.BINARY,
                )
        model._x_vars = x_vars  # type: ignore[attr-defined]

        # ---- objective: minimize total edit cost ----
        from causalcert.solver.constraints import build_edit_cost_expression

        cost_expr = build_edit_cost_expression(model, adj)
        model.objective = _mip.minimize(cost_expr)

        # ---- structural constraints ----
        self._add_acyclicity_constraints(model, adj)

        from causalcert.solver.constraints import (
            BudgetConstraint,
            MutualExclusionConstraint,
        )

        MutualExclusionConstraint(n).add_to_model(model, adj)
        BudgetConstraint(n, max_k).add_to_model(model, adj)

        logger.debug(
            "Built ILP model: %d vars, %d constraints",
            model.num_cols,
            model.num_rows,
        )
        return model

    # ---- acyclicity (convenience wrapper) ---------------------------------

    def _add_acyclicity_constraints(self, model: Any, adj: AdjacencyMatrix) -> None:
        """Add DAG acyclicity constraints to the ILP model.

        Delegates to :class:`AcyclicityConstraint`.

        Parameters
        ----------
        model : mip.Model
        adj : AdjacencyMatrix
        """
        from causalcert.solver.constraints import AcyclicityConstraint

        n = adj.shape[0]
        AcyclicityConstraint(n).add_to_model(model, adj)

    # ---- solution extraction -----------------------------------------------

    def _extract_solution(self, model: Any) -> tuple[list[StructuralEdit], float]:
        """Extract the edit set from a solved ILP model.

        Parameters
        ----------
        model : mip.Model
            Solved model.

        Returns
        -------
        tuple[list[StructuralEdit], float]
            ``(edits, objective_value)``.
        """
        from causalcert.solver.constraints import extract_edits_from_model

        adj_orig = getattr(model, "_adj_orig", None)
        if adj_orig is None:
            raise RuntimeError("Model does not have _adj_orig attached.")

        edits = extract_edits_from_model(model, adj_orig)
        obj = model.objective_value if model.objective_value is not None else float("inf")
        return edits, float(obj)

    # ---- incremental solving -----------------------------------------------

    def solve_incremental(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        k_values: Sequence[int] | None = None,
        ci_results: Sequence[CITestResult] | None = None,
    ) -> RobustnessRadius:
        """Incremental solve: try k = 1, 2, ... up to max_k.

        For each budget k, checks feasibility.  Returns as soon as a
        witness is found.  Uses warm-starting from the previous k.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG.
        predicate : ConclusionPredicate
            Conclusion predicate.
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment/outcome nodes.
        k_values : Sequence[int] | None
            Budgets to try.  Defaults to ``range(1, 11)``.
        ci_results : Sequence[CITestResult] | None
            CI test results.

        Returns
        -------
        RobustnessRadius
        """
        adj = np.asarray(adj, dtype=np.int8)
        if k_values is None:
            k_values = list(range(1, 11))

        t0 = time.perf_counter()
        best_result: RobustnessRadius | None = None

        for k in k_values:
            elapsed = time.perf_counter() - t0
            if elapsed >= self.time_limit_s:
                break

            remaining = self.time_limit_s - elapsed
            sub_solver = ILPSolver(
                time_limit_s=min(remaining, self.time_limit_s / len(k_values)),
                gap_tolerance=self.gap_tolerance,
                threads=self.threads,
                verbose=self.verbose,
            )
            if self._warm_start is not None:
                sub_solver.set_warm_start(self._warm_start)

            result = sub_solver.solve(
                adj, predicate, data, treatment, outcome,
                max_k=k, ci_results=ci_results,
            )

            if result.witness_edits:
                best_result = result
                break

            # Update warm-start from partial solution
            if sub_solver._last_model is not None:
                try:
                    from causalcert.solver.constraints import solution_adjacency

                    sol = solution_adjacency(sub_solver._last_model, adj.shape[0])
                    self.set_warm_start_from_adj(sol)
                except Exception:
                    pass

        total_time = time.perf_counter() - t0

        if best_result is not None:
            return RobustnessRadius(
                lower_bound=best_result.lower_bound,
                upper_bound=best_result.upper_bound,
                witness_edits=best_result.witness_edits,
                solver_strategy=SolverStrategy.ILP,
                solver_time_s=total_time,
                gap=best_result.gap,
                certified=best_result.certified,
            )
        else:
            max_tried = max(k_values) if k_values else 0
            return RobustnessRadius(
                lower_bound=max_tried + 1,
                upper_bound=max_tried + 1,
                witness_edits=(),
                solver_strategy=SolverStrategy.ILP,
                solver_time_s=total_time,
                gap=0.0,
                certified=True,
            )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def build_and_solve_ilp(
    adj: AdjacencyMatrix,
    predicate: ConclusionPredicate,
    data: Any,
    treatment: NodeId,
    outcome: NodeId,
    max_k: int = 10,
    ci_results: Sequence[CITestResult] | None = None,
    time_limit_s: float = 300.0,
    verbose: bool = False,
) -> RobustnessRadius:
    """One-shot convenience wrapper around :class:`ILPSolver`.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original DAG adjacency matrix.
    predicate : ConclusionPredicate
        Conclusion to stress-test.
    data : Any
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome variables.
    max_k : int
        Maximum edit distance.
    ci_results : Sequence[CITestResult] | None
        CI test results.
    time_limit_s : float
        Time limit in seconds.
    verbose : bool
        Print solver logs.

    Returns
    -------
    RobustnessRadius
    """
    solver = ILPSolver(
        time_limit_s=time_limit_s,
        verbose=verbose,
    )
    return solver.solve(
        adj, predicate, data, treatment, outcome,
        max_k=max_k, ci_results=ci_results,
    )
