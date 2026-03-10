"""
Unified solver interface with automatic strategy selection.

Selects the best solver strategy based on DAG size, treewidth, and available
time budget, then orchestrates multiple solvers to tighten bounds.

Strategy selection rules
------------------------
* **n <= 20  and  tw <= 5**: FPT (exact, fast for small treewidth).
* **n <= 50  and  tw <= max_treewidth**: FPT, then ILP if time remains.
* **n <= 100**: LP relaxation for lower bound, then ILP.
* **n >  100**: LP relaxation + CDCL as fallback.
* **Explicit strategy**: use the user-specified solver directly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np

from causalcert.solver.bounds import BoundManager, run_primal_heuristics
from causalcert.types import (
    AdjacencyMatrix,
    CITestResult,
    ConclusionPredicate,
    NodeId,
    RobustnessRadius,
    SolverStrategy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified solver
# ---------------------------------------------------------------------------


class UnifiedSolver:
    """Unified search interface that dispatches to specialised solvers.

    Strategy selection logic:

    * **Small DAGs** (n <= 20): try FPT first, then ILP.
    * **Low treewidth** (w <= max_treewidth): prefer FPT.
    * **Otherwise**: LP relaxation for a lower bound, then ILP.

    Parameters
    ----------
    strategy : SolverStrategy
        Explicit strategy, or ``AUTO`` for automatic selection.
    time_limit_s : float
        Total time budget for all solver invocations.
    max_treewidth_for_fpt : int
        Treewidth threshold for choosing FPT.
    verbose : bool
        Whether to print progress.
    """

    def __init__(
        self,
        strategy: SolverStrategy = SolverStrategy.AUTO,
        time_limit_s: float = 600.0,
        max_treewidth_for_fpt: int = 8,
        verbose: bool = False,
    ) -> None:
        self.strategy = strategy
        self.time_limit_s = time_limit_s
        self.max_treewidth_for_fpt = max_treewidth_for_fpt
        self.verbose = verbose

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
        """Compute the robustness radius using the selected (or auto) strategy.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG adjacency matrix.
        predicate : ConclusionPredicate
            Conclusion to stress-test.
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        max_k : int
            Maximum edit distance.
        ci_results : Sequence[CITestResult] | None
            Pre-computed CI test results.

        Returns
        -------
        RobustnessRadius
        """
        adj = np.asarray(adj, dtype=np.int8)
        t0 = time.perf_counter()

        # Trivial case
        if not predicate(adj, data, treatment=treatment, outcome=outcome):
            return RobustnessRadius(
                lower_bound=0, upper_bound=0,
                witness_edits=(),
                solver_strategy=self.strategy,
                solver_time_s=time.perf_counter() - t0,
                gap=0.0, certified=True,
            )

        # Determine strategy
        if self.strategy == SolverStrategy.AUTO:
            effective = self._select_strategy(adj)
        else:
            effective = self.strategy

        logger.info("Selected solver strategy: %s", effective.value)

        # Initialize bound manager
        bounds = BoundManager(max_k)

        # Run primal heuristics first for quick upper bounds
        try:
            run_primal_heuristics(
                adj, predicate, data, treatment, outcome,
                max_k=max_k, bound_mgr=bounds,
            )
        except Exception as exc:
            logger.debug("Primal heuristics failed: %s", exc)

        if bounds.is_tight:
            return bounds.to_result(time.perf_counter() - t0)

        # Dispatch to the chosen strategy
        if effective == SolverStrategy.FPT:
            result = self._run_fpt(
                adj, predicate, data, treatment, outcome,
                max_k, bounds, t0,
            )
        elif effective == SolverStrategy.ILP:
            result = self._run_ilp(
                adj, predicate, data, treatment, outcome,
                max_k, ci_results, bounds, t0,
            )
        elif effective == SolverStrategy.LP_RELAXATION:
            result = self._run_lp(
                adj, predicate, data, treatment, outcome,
                max_k, ci_results, bounds, t0,
            )
        elif effective == SolverStrategy.CDCL:
            result = self._run_cdcl(
                adj, predicate, data, treatment, outcome,
                max_k, bounds, t0,
            )
        else:
            result = self._run_fallback_chain(
                adj, predicate, data, treatment, outcome,
                max_k, ci_results, bounds, t0,
            )

        return result

    # ---- strategy selection -----------------------------------------------

    def _select_strategy(self, adj: AdjacencyMatrix) -> SolverStrategy:
        """Auto-select the best solver strategy.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.

        Returns
        -------
        SolverStrategy
        """
        n = adj.shape[0]

        # Compute treewidth
        try:
            from causalcert.dag.moral import treewidth_of_dag
            tw = treewidth_of_dag(adj)
        except Exception:
            tw = n  # assume worst case

        logger.debug("Auto-select: n=%d, treewidth=%d", n, tw)

        # Small + low treewidth: FPT is best
        if n <= 20 and tw <= 5:
            return SolverStrategy.FPT

        # Medium graph with bounded treewidth
        if tw <= self.max_treewidth_for_fpt and n <= 50:
            return SolverStrategy.FPT

        # Medium graph: try ILP
        if n <= 100:
            return SolverStrategy.ILP

        # Large graph: CDCL as heuristic, LP for bounds
        return SolverStrategy.CDCL

    # ---- individual solver runners ----------------------------------------

    def _run_fpt(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int,
        bounds: BoundManager,
        t0: float,
    ) -> RobustnessRadius:
        """Run the FPT solver with fallback to ILP."""
        from causalcert.solver.fpt import FPTSolver

        remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
        fpt = FPTSolver(
            max_treewidth=self.max_treewidth_for_fpt,
            time_limit_s=remaining,
        )

        try:
            result = fpt.solve(
                adj, predicate, data, treatment, outcome, max_k=max_k,
            )
            self._update_bounds(bounds, result)

            if result.certified:
                return bounds.to_result(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("FPT solver failed: %s", exc)

        # Fallback to ILP if time remains
        remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
        if remaining > 5.0:
            return self._run_ilp(
                adj, predicate, data, treatment, outcome,
                max_k, None, bounds, t0,
            )

        return bounds.to_result(time.perf_counter() - t0)

    def _run_ilp(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int,
        ci_results: Sequence[CITestResult] | None,
        bounds: BoundManager,
        t0: float,
    ) -> RobustnessRadius:
        """Run the ILP solver."""
        from causalcert.solver.ilp import ILPSolver

        remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
        ilp = ILPSolver(
            time_limit_s=remaining,
            verbose=self.verbose,
        )

        try:
            result = ilp.solve(
                adj, predicate, data, treatment, outcome,
                max_k=max_k, ci_results=ci_results,
            )
            self._update_bounds(bounds, result)
        except ImportError:
            logger.warning("python-mip not available; falling back to CDCL")
            return self._run_cdcl(
                adj, predicate, data, treatment, outcome,
                max_k, bounds, t0,
            )
        except Exception as exc:
            logger.warning("ILP solver failed: %s", exc)

        return bounds.to_result(time.perf_counter() - t0)

    def _run_lp(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int,
        ci_results: Sequence[CITestResult] | None,
        bounds: BoundManager,
        t0: float,
    ) -> RobustnessRadius:
        """Run LP relaxation for lower bound, then ILP for exact solution."""
        from causalcert.solver.lp_relaxation import LPRelaxationSolver

        remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
        lp_budget = min(remaining * 0.3, 60.0)

        try:
            lp = LPRelaxationSolver(
                time_limit_s=lp_budget,
                verbose=self.verbose,
            )
            lp_result = lp.solve(
                adj, predicate, data, treatment, outcome,
                max_k=max_k, ci_results=ci_results,
            )
            self._update_bounds(bounds, lp_result)
        except ImportError:
            logger.warning("python-mip not available for LP relaxation")
        except Exception as exc:
            logger.debug("LP relaxation failed: %s", exc)

        if bounds.is_tight:
            return bounds.to_result(time.perf_counter() - t0)

        # Follow up with ILP for exact solution
        return self._run_ilp(
            adj, predicate, data, treatment, outcome,
            max_k, ci_results, bounds, t0,
        )

    def _run_cdcl(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int,
        bounds: BoundManager,
        t0: float,
    ) -> RobustnessRadius:
        """Run the CDCL solver."""
        from causalcert.solver.cdcl import CDCLSolver

        remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
        cdcl = CDCLSolver(
            max_conflicts=10_000,
            time_limit_s=remaining,
        )

        try:
            result = cdcl.solve(
                adj, predicate, data, treatment, outcome, max_k=max_k,
            )
            self._update_bounds(bounds, result)
        except Exception as exc:
            logger.warning("CDCL solver failed: %s", exc)

        return bounds.to_result(time.perf_counter() - t0)

    def _run_fallback_chain(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int,
        ci_results: Sequence[CITestResult] | None,
        bounds: BoundManager,
        t0: float,
    ) -> RobustnessRadius:
        """Run the full fallback chain: FPT -> ILP -> CDCL.

        Allocates time proportionally: 30% FPT, 50% ILP, 20% CDCL.
        """
        total_budget = self.time_limit_s

        # Phase 1: FPT (if treewidth is reasonable)
        try:
            from causalcert.dag.moral import treewidth_of_dag
            tw = treewidth_of_dag(adj)
        except Exception:
            tw = adj.shape[0]

        if tw <= self.max_treewidth_for_fpt:
            fpt_budget = total_budget * 0.3
            remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
            if remaining > 5.0 and fpt_budget > 2.0:
                from causalcert.solver.fpt import FPTSolver

                fpt = FPTSolver(
                    max_treewidth=self.max_treewidth_for_fpt,
                    time_limit_s=min(fpt_budget, remaining),
                )
                try:
                    result = fpt.solve(
                        adj, predicate, data, treatment, outcome, max_k=max_k,
                    )
                    self._update_bounds(bounds, result)
                    if bounds.is_tight:
                        return bounds.to_result(time.perf_counter() - t0)
                except Exception as exc:
                    logger.debug("FPT in fallback chain failed: %s", exc)

        # Phase 2: ILP
        remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
        if remaining > 5.0:
            ilp_budget = remaining * 0.7
            try:
                from causalcert.solver.ilp import ILPSolver

                ilp = ILPSolver(
                    time_limit_s=ilp_budget,
                    verbose=self.verbose,
                )
                result = ilp.solve(
                    adj, predicate, data, treatment, outcome,
                    max_k=max_k, ci_results=ci_results,
                )
                self._update_bounds(bounds, result)
                if bounds.is_tight:
                    return bounds.to_result(time.perf_counter() - t0)
            except ImportError:
                logger.debug("ILP not available in fallback chain")
            except Exception as exc:
                logger.debug("ILP in fallback chain failed: %s", exc)

        # Phase 3: CDCL
        remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
        if remaining > 2.0:
            from causalcert.solver.cdcl import CDCLSolver

            cdcl = CDCLSolver(
                max_conflicts=10_000,
                time_limit_s=remaining,
            )
            try:
                result = cdcl.solve(
                    adj, predicate, data, treatment, outcome, max_k=max_k,
                )
                self._update_bounds(bounds, result)
            except Exception as exc:
                logger.debug("CDCL in fallback chain failed: %s", exc)

        return bounds.to_result(time.perf_counter() - t0)

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _update_bounds(bounds: BoundManager, result: RobustnessRadius) -> None:
        """Update the bound manager from a solver result."""
        if result.lower_bound > 0:
            bounds.update_lower(result.lower_bound, result.solver_strategy)
        if result.witness_edits:
            bounds.update_upper(
                result.upper_bound,
                result.witness_edits,
                result.solver_strategy,
            )
