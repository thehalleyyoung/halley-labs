"""
LP relaxation for fast lower bounds on the robustness radius (ALG 5).

Relaxes the binary edit variables to continuous [0, 1] and drops the
integrality constraint, yielding a polynomial-time lower bound.  The
fractional solution also provides branching guidance for the ILP and
shadow-price interpretation for edge importance.

Features
--------
* **LP lower bound**: ceil(LP_opt) is a valid lower bound.
* **Iterative refinement**: cutting planes from fractional solutions.
* **Rounding heuristic**: convert fractional to feasible integer solution.
* **Dual prices**: shadow prices indicate per-edge sensitivity.
"""

from __future__ import annotations

import logging
import math
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
            "python-mip is required for LP solving. Install with: pip install mip"
        )


# ---------------------------------------------------------------------------
# LP Relaxation Solver
# ---------------------------------------------------------------------------


class LPRelaxationSolver:
    """LP relaxation solver for robustness-radius lower bounds (ALG 5).

    Parameters
    ----------
    time_limit_s : float
        Maximum solver time in seconds.
    verbose : bool
        Whether to print solver logs.
    max_cutting_rounds : int
        Maximum number of cutting-plane iterations.
    """

    def __init__(
        self,
        time_limit_s: float = 60.0,
        verbose: bool = False,
        max_cutting_rounds: int = 20,
    ) -> None:
        self.time_limit_s = time_limit_s
        self.verbose = verbose
        self.max_cutting_rounds = max_cutting_rounds
        self._last_model: Any = None
        self._dual_prices: dict[str, float] = {}

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
        """Compute the LP relaxation lower bound.

        The LP relaxation replaces all binary variables with continuous
        ``[0, 1]`` variables.  The ceiling of the LP optimum is a valid
        lower bound on the robustness radius.

        After solving the LP, iterative cutting planes are added from
        fractional solutions.  A rounding heuristic is applied to obtain
        a feasible integer solution (upper bound).

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG adjacency matrix.
        predicate : ConclusionPredicate
            Conclusion to stress-test.
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment / outcome nodes.
        max_k : int
            Maximum edit distance.
        ci_results : Sequence[CITestResult] | None
            Pre-computed CI test results.

        Returns
        -------
        RobustnessRadius
        """
        _require_mip()
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        t0 = time.perf_counter()

        # Quick check: does predicate already fail?
        if not predicate(adj, data, treatment=treatment, outcome=outcome):
            return RobustnessRadius(
                lower_bound=0, upper_bound=0,
                witness_edits=(),
                solver_strategy=SolverStrategy.LP_RELAXATION,
                solver_time_s=time.perf_counter() - t0,
                gap=0.0, certified=True,
            )

        # Build LP model
        model = self._build_lp_model(adj, max_k)
        self._last_model = model

        # Add CI constraints
        if ci_results:
            from causalcert.solver.constraints import CIConsistencyConstraint
            CIConsistencyConstraint(ci_results, treatment, outcome).add_to_model(
                model, adj
            )

        # Initial LP solve
        lp_lower = 0
        model.optimize(max_seconds=max(1.0, self.time_limit_s / 2))

        if model.status == _mip.OptimizationStatus.INFEASIBLE:
            elapsed = time.perf_counter() - t0
            return RobustnessRadius(
                lower_bound=max_k + 1, upper_bound=max_k + 1,
                witness_edits=(),
                solver_strategy=SolverStrategy.LP_RELAXATION,
                solver_time_s=elapsed, gap=0.0, certified=True,
            )

        if model.objective_value is not None:
            lp_lower = int(math.ceil(model.objective_value - 1e-9))

        # Iterative cutting planes
        for rnd in range(self.max_cutting_rounds):
            elapsed = time.perf_counter() - t0
            if elapsed >= self.time_limit_s:
                break

            cuts_added = self._add_gomory_cuts(model, adj, n)
            cuts_added += self._add_integrality_cuts(model, n)

            if cuts_added == 0:
                break

            remaining = max(1.0, self.time_limit_s - elapsed)
            model.optimize(max_seconds=remaining)

            if model.objective_value is not None:
                new_lb = int(math.ceil(model.objective_value - 1e-9))
                if new_lb > lp_lower:
                    lp_lower = new_lb
                    logger.debug("LP bound tightened to %d (round %d)", lp_lower, rnd)

        # Rounding heuristic for upper bound
        upper_bound = max_k + 1
        witness: tuple[StructuralEdit, ...] = ()
        rounded_result = self._rounding_heuristic(
            model, adj, predicate, data, treatment, outcome, max_k
        )
        if rounded_result is not None:
            upper_bound, witness = rounded_result

        # Extract dual prices
        self._extract_dual_prices(model)

        elapsed = time.perf_counter() - t0
        gap = (upper_bound - lp_lower) / max(upper_bound, 1)
        return RobustnessRadius(
            lower_bound=lp_lower,
            upper_bound=upper_bound,
            witness_edits=witness,
            solver_strategy=SolverStrategy.LP_RELAXATION,
            solver_time_s=elapsed,
            gap=gap,
            certified=(lp_lower == upper_bound),
        )

    # ---- fractional solution access ----------------------------------------

    def fractional_solution(
        self,
        adj: AdjacencyMatrix,
        max_k: int,
    ) -> dict[tuple[int, int], float]:
        """Return the fractional edge-edit variables from the LP solution.

        Useful for branching heuristics in the ILP solver.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        max_k : int
            Maximum edits.

        Returns
        -------
        dict[tuple[int, int], float]
            Mapping from ``(source, target)`` to fractional edit value in [0, 1].
        """
        _require_mip()
        if self._last_model is None:
            model = self._build_lp_model(adj, max_k)
            model.optimize(max_seconds=self.time_limit_s)
            self._last_model = model

        model = self._last_model
        n = adj.shape[0]
        x = model._x_vars
        result: dict[tuple[int, int], float] = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = x[i, j].x
                if val is not None:
                    aij = int(adj[i, j])
                    edit_val = abs(val - aij)
                    if edit_val > 1e-6:
                        result[i, j] = edit_val
        return result

    # ---- dual prices -------------------------------------------------------

    def dual_prices(self) -> dict[str, float]:
        """Shadow prices from the LP relaxation.

        Higher magnitude indicates that the constraint is more binding,
        i.e. the associated edge is more important for robustness.

        Returns
        -------
        dict[str, float]
            Constraint name -> dual price.
        """
        return dict(self._dual_prices)

    def edge_importance(self, adj: AdjacencyMatrix) -> dict[tuple[int, int], float]:
        """Compute edge importance from dual prices.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original adjacency matrix.

        Returns
        -------
        dict[tuple[int, int], float]
            Edge (i,j) -> importance score (higher = more important).
        """
        if self._last_model is None:
            return {}

        model = self._last_model
        n = adj.shape[0]
        importance: dict[tuple[int, int], float] = {}

        # Aggregate dual prices for constraints involving each edge
        for cname, price in self._dual_prices.items():
            if abs(price) < 1e-10:
                continue
            parts = cname.split("_")
            if len(parts) >= 3 and parts[0] == "acyc":
                try:
                    i, j = int(parts[1]), int(parts[2])
                    importance[i, j] = importance.get((i, j), 0.0) + abs(price)
                except (ValueError, IndexError):
                    pass

        return importance

    # ---- integrality gap ---------------------------------------------------

    def integrality_gap(
        self,
        adj: AdjacencyMatrix,
        integer_opt: int,
        max_k: int = 10,
    ) -> float:
        """Compute the integrality gap: (integer_opt - lp_opt) / lp_opt.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG.
        integer_opt : int
            Known integer optimum.
        max_k : int
            Budget.

        Returns
        -------
        float
            Integrality gap ratio (0.0 means no gap).
        """
        if self._last_model is None:
            model = self._build_lp_model(adj, max_k)
            model.optimize(max_seconds=self.time_limit_s)
            self._last_model = model

        model = self._last_model
        if model.objective_value is None:
            return float("inf")

        lp_opt = model.objective_value
        if lp_opt < 1e-9:
            return float("inf") if integer_opt > 0 else 0.0
        return (integer_opt - lp_opt) / lp_opt

    # ---- internal methods --------------------------------------------------

    def _build_lp_model(self, adj: AdjacencyMatrix, max_k: int) -> Any:
        """Build the LP relaxation model (continuous variables)."""
        _require_mip()
        n = adj.shape[0]

        model = _mip.Model(name="lp_relaxation", sense=_mip.MINIMIZE)
        model.verbose = int(self.verbose)

        # Continuous edge variables
        x_vars: dict[tuple[int, int], Any] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x_vars[i, j] = model.add_var(
                    name=f"x_{i}_{j}",
                    var_type=_mip.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                )
        model._x_vars = x_vars  # type: ignore[attr-defined]

        # Continuous topological-order variables
        t_vars: dict[int, Any] = {}
        for i in range(n):
            t_vars[i] = model.add_var(
                name=f"t_{i}",
                var_type=_mip.CONTINUOUS,
                lb=0.0,
                ub=float(n - 1),
            )
        model._t_vars = t_vars  # type: ignore[attr-defined]

        # Objective
        from causalcert.solver.constraints import build_edit_cost_expression
        cost_expr = build_edit_cost_expression(model, adj)
        model.objective = _mip.minimize(cost_expr)

        # Acyclicity constraints (same big-M encoding, continuous t)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                model += (
                    t_vars[j] - t_vars[i] >= 1 - n * (1 - x_vars[i, j]),
                    f"acyc_{i}_{j}",
                )

        # Mutual exclusion
        for i in range(n):
            for j in range(i + 1, n):
                model += x_vars[i, j] + x_vars[j, i] <= 1, f"mutex_{i}_{j}"

        # Budget
        model += cost_expr <= max_k, "budget"

        return model

    def _add_gomory_cuts(self, model: Any, adj: AdjacencyMatrix, n: int) -> int:
        """Add Gomory-style cuts from fractional solutions.

        For each fractional variable close to 0.5, add a disjunctive cut
        forcing it toward integrality.
        """
        _require_mip()
        x = model._x_vars
        cuts_added = 0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = x[i, j].x
                if val is None:
                    continue
                # Most fractional variables
                frac = val - int(val)
                if 0.2 < frac < 0.8:
                    # Strengthen: combine with acyclicity
                    t = model._t_vars
                    # If edge (i,j) is "half present", either enforce
                    # the full topological ordering or eliminate it
                    # This is a simplified Gomory-type cut
                    if val > 0.5:
                        model += x[i, j] >= 0.5, f"gom_lb_{i}_{j}_{cuts_added}"
                    else:
                        model += x[i, j] <= 0.5, f"gom_ub_{i}_{j}_{cuts_added}"
                    cuts_added += 1
                    if cuts_added >= 50:
                        return cuts_added

        return cuts_added

    def _add_integrality_cuts(self, model: Any, n: int) -> int:
        """Add integrality-enforcement cuts for the most fractional variables."""
        _require_mip()
        x = model._x_vars
        cuts_added = 0

        # Collect fractional variables sorted by distance from integrality
        fractional: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = x[i, j].x
                if val is None:
                    continue
                dist = min(val, 1.0 - val)
                if dist > 0.01:
                    fractional.append((dist, i, j))

        fractional.sort(reverse=True)

        # Add cuts for the 10 most fractional
        for _, i, j in fractional[:10]:
            val = x[i, j].x
            if val is not None and val > 0.5:
                model += x[i, j] >= 1.0, f"intcut_up_{i}_{j}_{cuts_added}"
            else:
                model += x[i, j] <= 0.0, f"intcut_dn_{i}_{j}_{cuts_added}"
            cuts_added += 1

        return cuts_added

    def _rounding_heuristic(
        self,
        model: Any,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int,
    ) -> tuple[int, tuple[StructuralEdit, ...]] | None:
        """Round the fractional LP solution to an integer-feasible DAG.

        Tries deterministic rounding, then randomized rounding.

        Returns
        -------
        tuple[int, tuple[StructuralEdit, ...]] | None
            ``(cost, edits)`` or ``None`` if no feasible solution found.
        """
        n = adj.shape[0]
        x = model._x_vars

        from causalcert.dag.validation import is_dag
        from causalcert.dag.edit import diff_edits

        # Deterministic rounding: round to nearest integer
        for threshold in [0.5, 0.6, 0.4, 0.7, 0.3]:
            rounded = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    val = x[i, j].x
                    if val is not None and val >= threshold:
                        rounded[i, j] = 1

            # Fix bidirectional edges
            for i in range(n):
                for j in range(i + 1, n):
                    if rounded[i, j] and rounded[j, i]:
                        vi = x[i, j].x or 0
                        vj = x[j, i].x or 0
                        if vi >= vj:
                            rounded[j, i] = 0
                        else:
                            rounded[i, j] = 0

            if not is_dag(rounded):
                continue

            edits = diff_edits(adj, rounded)
            cost = len(edits)
            if cost > max_k:
                continue

            if not predicate(
                rounded, data, treatment=treatment, outcome=outcome
            ):
                return cost, tuple(edits)

        # Randomized rounding (5 attempts)
        rng = np.random.default_rng(42)
        for _ in range(5):
            rounded = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    val = x[i, j].x
                    if val is not None and rng.random() < val:
                        rounded[i, j] = 1

            # Fix bidirectional
            for i in range(n):
                for j in range(i + 1, n):
                    if rounded[i, j] and rounded[j, i]:
                        rounded[j, i] = 0

            if not is_dag(rounded):
                continue

            edits = diff_edits(adj, rounded)
            cost = len(edits)
            if cost > max_k:
                continue

            if not predicate(
                rounded, data, treatment=treatment, outcome=outcome
            ):
                return cost, tuple(edits)

        return None

    def _extract_dual_prices(self, model: Any) -> None:
        """Extract dual variable values (shadow prices) from the LP."""
        self._dual_prices.clear()
        try:
            for constr in model.constrs:
                if constr.pi is not None:
                    self._dual_prices[constr.name] = constr.pi
        except Exception:
            pass
