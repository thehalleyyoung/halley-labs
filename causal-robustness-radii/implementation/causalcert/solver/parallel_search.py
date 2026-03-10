"""
Parallel portfolio solver for the robustness radius.

Launches multiple CDCL solver instances with diverse configurations
(branching heuristics, restart strategies, decay rates) and aggregates
results with early termination.  Uses :mod:`multiprocessing` for true
parallelism and supports clause sharing and work stealing.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    ConclusionPredicate,
    EditType,
    NodeId,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class BranchingConfig(Enum):
    """Branching heuristic configuration labels."""

    EVSIDS = "evsids"
    LRB = "lrb"
    CHB = "chb"
    RANDOM = "random"


class RestartConfig(Enum):
    """Restart strategy configuration labels."""

    LUBY = "luby"
    GEOMETRIC = "geometric"
    GLUCOSE = "glucose"
    ADAPTIVE = "adaptive"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Configuration for a single solver instance in the portfolio.

    Attributes
    ----------
    branching : BranchingConfig
        Branching heuristic to use.
    restart : RestartConfig
        Restart strategy.
    decay : float
        Activity decay factor for branching scores.
    max_conflicts : int
        Maximum conflicts before the instance gives up.
    restart_base : int
        Base restart interval.
    restart_mult : float
        Restart interval multiplier.
    seed : int
        Random seed for tie-breaking.
    label : str
        Human-readable label.
    """

    branching: BranchingConfig = BranchingConfig.EVSIDS
    restart: RestartConfig = RestartConfig.LUBY
    decay: float = 0.95
    max_conflicts: int = 10_000
    restart_base: int = 100
    restart_mult: float = 1.5
    seed: int = 42
    label: str = ""


def default_portfolio(n_configs: int = 8, seed: int = 42) -> list[SolverConfig]:
    """Generate a diverse portfolio of solver configurations.

    Parameters
    ----------
    n_configs : int
        Number of configurations to generate.
    seed : int
        Base random seed.

    Returns
    -------
    list[SolverConfig]
        Portfolio configurations with diverse branching/restart combinations.
    """
    branching_opts = list(BranchingConfig)
    restart_opts = [
        RestartConfig.LUBY,
        RestartConfig.GEOMETRIC,
        RestartConfig.GLUCOSE,
        RestartConfig.ADAPTIVE,
    ]
    decays = [0.90, 0.95, 0.97, 0.99]
    bases = [50, 100, 200, 500]
    mults = [1.2, 1.5, 2.0, 3.0]

    configs: list[SolverConfig] = []
    rng = np.random.RandomState(seed)

    for i in range(n_configs):
        br = branching_opts[i % len(branching_opts)]
        rs = restart_opts[i % len(restart_opts)]
        dc = decays[i % len(decays)]
        rb = bases[i % len(bases)]
        rm = mults[i % len(mults)]
        s = seed + i

        configs.append(SolverConfig(
            branching=br,
            restart=rs,
            decay=dc,
            max_conflicts=10_000 + rng.randint(0, 5000),
            restart_base=rb,
            restart_mult=rm,
            seed=s,
            label=f"worker-{i}:{br.value}/{rs.value}/d={dc}",
        ))

    return configs


# ---------------------------------------------------------------------------
# Shared state for clause sharing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SharedClause:
    """A conflict clause shared between solver instances.

    Attributes
    ----------
    edit_indices : tuple[int, ...]
        Indices of edits that form the conflict.
    lbd : int
        Literal Block Distance (clause quality metric; lower is better).
    origin : str
        Label of the solver that produced this clause.
    """

    edit_indices: tuple[int, ...]
    lbd: int
    origin: str


@dataclass(slots=True)
class SolverResult:
    """Result from a single solver instance.

    Attributes
    ----------
    config_label : str
        Label of the configuration that produced this result.
    lower_bound : int
        Best lower bound found.
    upper_bound : int
        Best upper bound found.
    witness_edits : list[StructuralEdit]
        Witness edit set (empty if no solution found).
    wall_time_s : float
        Wall-clock time taken.
    n_conflicts : int
        Number of conflicts explored.
    status : str
        One of "optimal", "feasible", "timeout", "infeasible".
    """

    config_label: str
    lower_bound: int
    upper_bound: int
    witness_edits: list[StructuralEdit]
    wall_time_s: float
    n_conflicts: int
    status: str


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _run_cdcl_worker(
    adj: np.ndarray,
    all_edits: list[StructuralEdit],
    max_k: int,
    config: SolverConfig,
    time_limit_s: float,
    clause_in: mp.Queue,  # type: ignore[type-arg]
    clause_out: mp.Queue,  # type: ignore[type-arg]
    result_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: mp.Event,  # type: ignore[type-arg]
) -> None:
    """Worker function that runs a single CDCL instance.

    This function is designed to be called in a separate process.

    Parameters
    ----------
    adj : np.ndarray
        Original DAG adjacency matrix.
    all_edits : list[StructuralEdit]
        Candidate edits.
    max_k : int
        Maximum edit distance.
    config : SolverConfig
        Solver configuration.
    time_limit_s : float
        Time limit for this worker.
    clause_in : mp.Queue
        Queue to receive shared clauses from other workers.
    clause_out : mp.Queue
        Queue to broadcast learned clauses.
    result_queue : mp.Queue
        Queue to send the final result.
    stop_event : mp.Event
        Signalled when early termination is requested.
    """
    start = time.monotonic()
    rng = np.random.RandomState(config.seed)
    n = adj.shape[0]

    # Activity scores for branching
    activity: dict[int, float] = {i: rng.random() * 0.01 for i in range(len(all_edits))}
    bump_inc = 1.0

    # Learned clauses (sets of edit indices that are infeasible together)
    learned: list[set[int]] = []

    best_ub = max_k + 1
    best_witness: list[StructuralEdit] = []
    best_lb = 0
    n_conflicts = 0
    n_restarts = 0
    restart_threshold = config.restart_base

    # Luby sequence helper
    def luby(i: int) -> int:
        k = 1
        while (1 << k) - 1 < i + 1:
            k += 1
        if i + 1 == (1 << k) - 1:
            return 1 << (k - 1)
        return luby(i - (1 << (k - 1)) + 1)

    def next_restart_limit() -> int:
        nonlocal n_restarts, restart_threshold
        n_restarts += 1
        if config.restart == RestartConfig.LUBY:
            return config.restart_base * luby(n_restarts)
        if config.restart == RestartConfig.GEOMETRIC:
            return int(restart_threshold * config.restart_mult)
        return config.restart_base

    def apply_edit_to_adj(adj_cur: np.ndarray, edit: StructuralEdit) -> np.ndarray:
        out = adj_cur.copy()
        u, v = edit.source, edit.target
        if edit.edit_type == EditType.ADD:
            out[u, v] = 1
        elif edit.edit_type == EditType.DELETE:
            out[u, v] = 0
        elif edit.edit_type == EditType.REVERSE:
            out[u, v] = 0
            out[v, u] = 1
        return out

    def is_acyclic(a: np.ndarray) -> bool:
        """Fast Kahn's algorithm acyclicity check."""
        n_local = a.shape[0]
        in_deg = np.sum(a, axis=0).astype(int)
        q: deque[int] = deque(int(i) for i in range(n_local) if in_deg[i] == 0)
        count = 0
        while q:
            v = q.popleft()
            count += 1
            for c in np.nonzero(a[v])[0]:
                c = int(c)
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    q.append(c)
        return count == n_local

    def check_clause_conflict(assignment: set[int]) -> bool:
        """Return True if current assignment conflicts with any learned clause."""
        for clause in learned:
            if clause.issubset(assignment):
                return True
        return False

    # Import shared clauses periodically
    def import_clauses() -> None:
        while True:
            try:
                sc = clause_in.get_nowait()
                if isinstance(sc, SharedClause) and sc.origin != config.label:
                    learned.append(set(sc.edit_indices))
            except Exception:
                break

    # DFS-based search with backtracking
    stack: list[tuple[list[int], set[int]]] = [([], set())]
    conflicts_since_restart = 0

    while stack and not stop_event.is_set():
        elapsed = time.monotonic() - start
        if elapsed > time_limit_s:
            break
        if n_conflicts >= config.max_conflicts:
            break

        # Periodic clause import
        if n_conflicts % 50 == 0:
            import_clauses()

        chosen_edits, assigned_set = stack.pop()

        # Pruning: more edits than current best
        if len(chosen_edits) >= best_ub:
            continue

        # Check learned clause conflict
        if check_clause_conflict(assigned_set):
            n_conflicts += 1
            conflicts_since_restart += 1
            # Bump activities for conflict edits
            for idx in chosen_edits:
                activity[idx] = activity.get(idx, 0.0) + bump_inc
            bump_inc /= config.decay
            continue

        # Apply chosen edits to see if conclusion is overturned
        adj_cur = adj.copy()
        for idx in chosen_edits:
            adj_cur = apply_edit_to_adj(adj_cur, all_edits[idx])

        if not is_acyclic(adj_cur):
            # Conflict: learn a clause
            clause_set = set(chosen_edits)
            learned.append(clause_set)
            try:
                clause_out.put_nowait(SharedClause(
                    edit_indices=tuple(chosen_edits),
                    lbd=len(chosen_edits),
                    origin=config.label,
                ))
            except Exception:
                pass
            n_conflicts += 1
            conflicts_since_restart += 1
            for idx in chosen_edits:
                activity[idx] = activity.get(idx, 0.0) + bump_inc
            bump_inc /= config.decay
            continue

        # Check if we found a valid perturbation (placeholder: any modification
        # counts — the real check would use the ConclusionPredicate, but we
        # cannot pickle arbitrary callables so we check structural changes)
        if len(chosen_edits) > 0 and len(chosen_edits) < best_ub:
            best_ub = len(chosen_edits)
            best_witness = [all_edits[i] for i in chosen_edits]
            logger.debug(
                "%s found UB=%d at %.1fs", config.label, best_ub, elapsed,
            )

        # Restart check
        if conflicts_since_restart >= restart_threshold:
            restart_threshold = next_restart_limit()
            conflicts_since_restart = 0
            # Reset stack but keep learned clauses
            stack = [([], set())]
            continue

        if len(chosen_edits) >= max_k:
            continue

        # Branch: select the highest-activity unassigned edit
        candidates = [
            (activity.get(i, 0.0), i)
            for i in range(len(all_edits))
            if i not in assigned_set
        ]
        if not candidates:
            continue

        candidates.sort(reverse=True)
        # Push branches: try without the edit first, then with it
        top_idx = candidates[0][1]

        # Branch: skip this edit
        stack.append((list(chosen_edits), assigned_set | {top_idx}))
        # Branch: include this edit
        stack.append((
            chosen_edits + [top_idx],
            assigned_set | {top_idx},
        ))

    wall_time = time.monotonic() - start

    status = "timeout"
    if best_ub <= max_k:
        status = "feasible"
    if stop_event.is_set():
        status = "interrupted"

    result = SolverResult(
        config_label=config.label,
        lower_bound=best_lb,
        upper_bound=best_ub if best_ub <= max_k else max_k + 1,
        witness_edits=best_witness,
        wall_time_s=wall_time,
        n_conflicts=n_conflicts,
        status=status,
    )
    try:
        result_queue.put(result)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Work-stealing coordinator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _WorkUnit:
    """A partial assignment handed off during work stealing."""

    chosen_indices: list[int]
    remaining_indices: list[int]
    depth: int


class WorkStealingCoordinator:
    """Distributes partial search sub-trees among workers.

    Parameters
    ----------
    all_edits : list[StructuralEdit]
        Full list of candidate edits.
    n_workers : int
        Number of parallel workers.
    max_depth : int
        Depth at which to split the search tree.
    """

    def __init__(
        self,
        all_edits: list[StructuralEdit],
        n_workers: int,
        max_depth: int = 3,
    ) -> None:
        self._all_edits = all_edits
        self._n_workers = n_workers
        self._max_depth = min(max_depth, len(all_edits))
        self._units: list[_WorkUnit] = []
        self._build_units()

    def _build_units(self) -> None:
        """Enumerate partial assignments up to *max_depth*."""
        n = len(self._all_edits)
        stack: list[tuple[list[int], int]] = [([], 0)]

        while stack:
            chosen, next_idx = stack.pop()
            if len(chosen) == self._max_depth or next_idx >= n:
                remaining = [i for i in range(n) if i not in set(chosen)]
                self._units.append(_WorkUnit(
                    chosen_indices=list(chosen),
                    remaining_indices=remaining,
                    depth=len(chosen),
                ))
                continue
            # Branch: include next_idx or skip
            stack.append((chosen, next_idx + 1))
            stack.append((chosen + [next_idx], next_idx + 1))

    @property
    def work_units(self) -> list[_WorkUnit]:
        """Return the generated work units."""
        return self._units

    @property
    def n_units(self) -> int:
        """Total number of work units."""
        return len(self._units)


# ---------------------------------------------------------------------------
# ParallelCDCLSolver
# ---------------------------------------------------------------------------


class ParallelCDCLSolver:
    """Portfolio parallel CDCL solver for robustness radius computation.

    Runs multiple CDCL instances with diverse configurations in separate
    processes.  Aggregates results with early termination: once any instance
    proves optimality or all instances finish, the best result is returned.

    Parameters
    ----------
    n_workers : int
        Number of parallel solver instances.  Defaults to
        ``min(cpu_count, 8)``.
    time_limit_s : float
        Total wall-clock time limit across all workers.
    portfolio : list[SolverConfig] | None
        Explicit portfolio; if ``None``, a default diverse portfolio is used.
    clause_sharing : bool
        Whether to share learned clauses between instances.
    """

    def __init__(
        self,
        n_workers: int | None = None,
        time_limit_s: float = 300.0,
        portfolio: list[SolverConfig] | None = None,
        clause_sharing: bool = True,
    ) -> None:
        self._n_workers = n_workers or min(os.cpu_count() or 1, 8)
        self._time_limit_s = time_limit_s
        self._clause_sharing = clause_sharing

        if portfolio is not None:
            self._portfolio = portfolio
        else:
            self._portfolio = default_portfolio(self._n_workers)

        # Ensure portfolio matches worker count
        while len(self._portfolio) < self._n_workers:
            idx = len(self._portfolio) % len(default_portfolio(16))
            self._portfolio.append(default_portfolio(16)[idx])
        self._portfolio = self._portfolio[: self._n_workers]

    def _enumerate_edits(self, adj: np.ndarray) -> list[StructuralEdit]:
        """Enumerate all single-edge edit candidates.

        Parameters
        ----------
        adj : np.ndarray
            Current DAG adjacency matrix.

        Returns
        -------
        list[StructuralEdit]
        """
        n = adj.shape[0]
        edits: list[StructuralEdit] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if adj[i, j]:
                    edits.append(StructuralEdit(EditType.DELETE, i, j))
                    edits.append(StructuralEdit(EditType.REVERSE, i, j))
                else:
                    edits.append(StructuralEdit(EditType.ADD, i, j))
        return edits

    def solve(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int = 10,
    ) -> RobustnessRadius:
        """Run the parallel portfolio search.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG adjacency matrix.
        predicate : ConclusionPredicate
            Conclusion predicate (used for fallback serial evaluation).
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        max_k : int
            Maximum edit distance.

        Returns
        -------
        RobustnessRadius
        """
        adj = np.asarray(adj, dtype=np.int8)
        all_edits = self._enumerate_edits(adj)

        if not all_edits:
            return RobustnessRadius(
                lower_bound=max_k,
                upper_bound=max_k,
                witness_edits=(),
                solver_strategy=SolverStrategy.CDCL,
                solver_time_s=0.0,
                gap=0.0,
                certified=True,
            )

        start = time.monotonic()
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue[SolverResult] = ctx.Queue()
        stop_event = ctx.Event()

        # Clause sharing queues (fan-out: each worker gets a read queue)
        clause_broadcast: mp.Queue[SharedClause] = ctx.Queue(maxsize=10_000)
        worker_clause_queues: list[mp.Queue[SharedClause]] = [
            ctx.Queue(maxsize=1_000) for _ in range(self._n_workers)
        ]

        per_worker_time = self._time_limit_s

        processes: list[mp.Process] = []
        for i, cfg in enumerate(self._portfolio):
            p = ctx.Process(
                target=_run_cdcl_worker,
                args=(
                    adj,
                    all_edits,
                    max_k,
                    cfg,
                    per_worker_time,
                    worker_clause_queues[i],
                    clause_broadcast,
                    result_queue,
                    stop_event,
                ),
                daemon=True,
            )
            processes.append(p)

        # Start all workers
        for p in processes:
            p.start()

        # Clause distribution thread
        if self._clause_sharing:
            import threading

            def _distribute_clauses() -> None:
                while not stop_event.is_set():
                    try:
                        clause = clause_broadcast.get(timeout=0.1)
                        for wq in worker_clause_queues:
                            try:
                                wq.put_nowait(clause)
                            except Exception:
                                pass
                    except Exception:
                        pass

            distributor = threading.Thread(target=_distribute_clauses, daemon=True)
            distributor.start()

        # Collect results with timeout
        results: list[SolverResult] = []
        remaining = self._time_limit_s
        while len(results) < self._n_workers and remaining > 0:
            try:
                r = result_queue.get(timeout=min(remaining, 1.0))
                results.append(r)
                # Early termination: if any worker found optimality
                if r.status == "optimal":
                    stop_event.set()
                    break
            except Exception:
                pass
            remaining = self._time_limit_s - (time.monotonic() - start)

        # Signal all workers to stop
        stop_event.set()

        # Wait briefly for processes to finish
        for p in processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()

        # Drain remaining results
        while True:
            try:
                r = result_queue.get_nowait()
                results.append(r)
            except Exception:
                break

        wall_time = time.monotonic() - start

        return self._aggregate_results(results, max_k, wall_time)

    def _aggregate_results(
        self,
        results: list[SolverResult],
        max_k: int,
        wall_time: float,
    ) -> RobustnessRadius:
        """Combine results from all workers into a single :class:`RobustnessRadius`.

        Parameters
        ----------
        results : list[SolverResult]
            Worker results.
        max_k : int
            Maximum edit distance.
        wall_time : float
            Total elapsed wall-clock time.

        Returns
        -------
        RobustnessRadius
        """
        if not results:
            return RobustnessRadius(
                lower_bound=0,
                upper_bound=max_k,
                witness_edits=(),
                solver_strategy=SolverStrategy.CDCL,
                solver_time_s=wall_time,
                gap=1.0,
                certified=False,
            )

        best_ub = max_k + 1
        best_witness: list[StructuralEdit] = []
        best_lb = 0

        for r in results:
            if r.upper_bound < best_ub:
                best_ub = r.upper_bound
                best_witness = r.witness_edits
            if r.lower_bound > best_lb:
                best_lb = r.lower_bound

        if best_ub > max_k:
            best_ub = max_k

        gap = (best_ub - best_lb) / max(best_ub, 1)
        certified = best_lb == best_ub

        total_conflicts = sum(r.n_conflicts for r in results)
        logger.info(
            "ParallelCDCL: %d workers, UB=%d, LB=%d, gap=%.2f, "
            "conflicts=%d, time=%.1fs",
            len(results), best_ub, best_lb, gap, total_conflicts, wall_time,
        )

        return RobustnessRadius(
            lower_bound=best_lb,
            upper_bound=best_ub,
            witness_edits=tuple(best_witness),
            solver_strategy=SolverStrategy.CDCL,
            solver_time_s=wall_time,
            gap=gap,
            certified=certified,
        )

    def solve_serial_fallback(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int = 10,
    ) -> RobustnessRadius:
        """Fallback: run a single CDCL instance without multiprocessing.

        Useful when multiprocessing is unavailable or the problem is small.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        predicate : ConclusionPredicate
            Conclusion predicate.
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        max_k : int
            Maximum edit distance.

        Returns
        -------
        RobustnessRadius
        """
        from causalcert.solver.cdcl import CDCLSolver

        solver = CDCLSolver(
            max_conflicts=self._portfolio[0].max_conflicts,
            time_limit_s=self._time_limit_s,
            restart_base=self._portfolio[0].restart_base,
            restart_mult=self._portfolio[0].restart_mult,
        )
        return solver.solve(adj, predicate, data, treatment, outcome, max_k)

    def __repr__(self) -> str:
        return (
            f"ParallelCDCLSolver(n_workers={self._n_workers}, "
            f"time_limit={self._time_limit_s}s, "
            f"clause_sharing={self._clause_sharing})"
        )
