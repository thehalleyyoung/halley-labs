"""
Upper / lower bound management for the robustness radius.

Maintains incumbent (best-known) upper and lower bounds across multiple
solver invocations and supports bound tightening callbacks, primal
heuristics, and convergence monitoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

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
# Bound state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BoundState:
    """Snapshot of current bounds.

    Attributes
    ----------
    lower : int
        Best lower bound.
    upper : int
        Best upper bound.
    witness : tuple[StructuralEdit, ...]
        Edit set achieving the upper bound.
    strategy : SolverStrategy
        Solver that produced the tightest bounds.
    """

    lower: int = 0
    upper: int = 999
    witness: tuple[StructuralEdit, ...] = ()
    strategy: SolverStrategy = SolverStrategy.AUTO


# ---------------------------------------------------------------------------
# Bound history entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundEvent:
    """A single bound-update event for convergence tracking.

    Attributes
    ----------
    timestamp : float
        Wall-clock time (seconds since solver start).
    bound_type : str
        ``"lower"`` or ``"upper"``.
    value : int
        New bound value.
    strategy : SolverStrategy
        Solver that produced the bound.
    """

    timestamp: float
    bound_type: str
    value: int
    strategy: SolverStrategy


# ---------------------------------------------------------------------------
# Bound manager
# ---------------------------------------------------------------------------


class BoundManager:
    """Manages and tightens upper/lower bounds across solver runs.

    Parameters
    ----------
    max_k : int
        A priori upper bound on the edit distance.
    """

    def __init__(self, max_k: int = 10) -> None:
        self._state = BoundState(lower=0, upper=max_k)
        self._callbacks: list[Callable[[BoundState], None]] = []
        self._history: list[BoundEvent] = []
        self._start_time: float = time.perf_counter()

    # ---- properties -------------------------------------------------------

    @property
    def state(self) -> BoundState:
        """Current bound state."""
        return self._state

    @property
    def is_tight(self) -> bool:
        """Whether the bounds have converged (lower == upper)."""
        return self._state.lower == self._state.upper

    @property
    def gap(self) -> float:
        """Current relative optimality gap ``(UB - LB) / max(UB, 1)``."""
        s = self._state
        return (s.upper - s.lower) / max(s.upper, 1)

    @property
    def history(self) -> list[BoundEvent]:
        """Full history of bound updates."""
        return list(self._history)

    # ---- bound updates ----------------------------------------------------

    def update_lower(self, lb: int, strategy: SolverStrategy) -> bool:
        """Update the lower bound if *lb* improves on the incumbent.

        Parameters
        ----------
        lb : int
            New lower bound.
        strategy : SolverStrategy
            Solver that produced this bound.

        Returns
        -------
        bool
            ``True`` if the bound was tightened.
        """
        if lb > self._state.lower:
            self._state.lower = lb
            self._state.strategy = strategy
            self._history.append(BoundEvent(
                time.perf_counter() - self._start_time,
                "lower", lb, strategy,
            ))
            self._notify()
            logger.debug("Lower bound tightened to %d (by %s)", lb, strategy.value)
            return True
        return False

    def update_upper(
        self,
        ub: int,
        witness: tuple[StructuralEdit, ...],
        strategy: SolverStrategy,
    ) -> bool:
        """Update the upper bound if *ub* improves on the incumbent.

        Parameters
        ----------
        ub : int
            New upper bound.
        witness : tuple[StructuralEdit, ...]
            Edit set achieving the bound.
        strategy : SolverStrategy
            Solver that produced this bound.

        Returns
        -------
        bool
            ``True`` if the bound was tightened.
        """
        if ub < self._state.upper:
            self._state.upper = ub
            self._state.witness = witness
            self._state.strategy = strategy
            self._history.append(BoundEvent(
                time.perf_counter() - self._start_time,
                "upper", ub, strategy,
            ))
            self._notify()
            logger.debug("Upper bound tightened to %d (by %s)", ub, strategy.value)
            return True
        return False

    # ---- conversion -------------------------------------------------------

    def to_result(self, solver_time_s: float = 0.0) -> RobustnessRadius:
        """Convert the current bounds to a :class:`RobustnessRadius`.

        Parameters
        ----------
        solver_time_s : float
            Total solver time.

        Returns
        -------
        RobustnessRadius
        """
        s = self._state
        gap = (s.upper - s.lower) / max(s.upper, 1)
        return RobustnessRadius(
            lower_bound=s.lower,
            upper_bound=s.upper,
            witness_edits=s.witness,
            solver_strategy=s.strategy,
            solver_time_s=solver_time_s,
            gap=gap,
            certified=s.lower == s.upper,
        )

    # ---- callbacks --------------------------------------------------------

    def register_callback(self, callback: Callable[[BoundState], None]) -> None:
        """Register a callback invoked whenever bounds change.

        Parameters
        ----------
        callback : Callable[[BoundState], None]
        """
        self._callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self._callbacks:
            cb(self._state)

    # ---- convergence monitoring -------------------------------------------

    def convergence_rate(self) -> float:
        """Estimated convergence rate (gap reduction per second).

        Returns
        -------
        float
            Rate of gap closure. Zero if no history.
        """
        if len(self._history) < 2:
            return 0.0

        first = self._history[0]
        last = self._history[-1]
        dt = last.timestamp - first.timestamp
        if dt < 1e-6:
            return 0.0

        initial_gap = self._state.upper  # rough
        final_gap = self._state.upper - self._state.lower
        return (initial_gap - final_gap) / dt

    def estimated_time_to_close(self) -> float:
        """Estimate remaining time to close the gap.

        Returns
        -------
        float
            Estimated seconds, or ``float('inf')`` if convergence is stalled.
        """
        rate = self.convergence_rate()
        if rate < 1e-9:
            return float("inf")
        remaining_gap = self._state.upper - self._state.lower
        return remaining_gap / rate

    # ---- anytime reporting ------------------------------------------------

    def anytime_report(self) -> str:
        """Generate a human-readable status report.

        Returns
        -------
        str
            Multi-line summary of current bounds and convergence.
        """
        s = self._state
        elapsed = time.perf_counter() - self._start_time
        lines = [
            f"Bound Status (elapsed: {elapsed:.1f}s)",
            f"  Lower bound: {s.lower}",
            f"  Upper bound: {s.upper}",
            f"  Gap: {self.gap:.1%}",
            f"  Witness edits: {len(s.witness)}",
            f"  Strategy: {s.strategy.value}",
            f"  Bound events: {len(self._history)}",
            f"  Converged: {self.is_tight}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primal heuristics
# ---------------------------------------------------------------------------


class GreedyPrimalHeuristic:
    """Greedy heuristic for computing upper bounds.

    Iteratively applies the single edit that brings the conclusion predicate
    closest to being overturned (measured by predicate evaluation on the
    perturbed graph).

    Parameters
    ----------
    max_edits : int
        Maximum number of greedy edits to try.
    """

    def __init__(self, max_edits: int = 10) -> None:
        self.max_edits = max_edits

    def run(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
    ) -> tuple[int, tuple[StructuralEdit, ...]] | None:
        """Run the greedy heuristic.

        Returns
        -------
        tuple[int, tuple[StructuralEdit, ...]] | None
            ``(cost, edits)`` if a witness is found, else ``None``.
        """
        from causalcert.dag.edit import all_single_edits, apply_edit

        current = np.asarray(adj, dtype=np.int8).copy()
        edits_applied: list[StructuralEdit] = []

        for _ in range(self.max_edits):
            candidates = all_single_edits(current)
            if not candidates:
                break

            found = False
            for edit in candidates:
                trial = apply_edit(current, edit)
                if not predicate(
                    trial, data, treatment=treatment, outcome=outcome
                ):
                    edits_applied.append(edit)
                    return len(edits_applied), tuple(edits_applied)

            # No single edit overturns — pick one that changes most
            # (heuristic: prefer deletions, then reversals, then additions)
            priority = {EditType.DELETE: 0, EditType.REVERSE: 1, EditType.ADD: 2}
            candidates.sort(key=lambda e: priority.get(e.edit_type, 3))

            if candidates:
                edit = candidates[0]
                current = apply_edit(current, edit)
                edits_applied.append(edit)
            else:
                break

        return None


class RandomSamplingHeuristic:
    """Random sampling heuristic for upper bounds.

    Randomly selects subsets of edits and checks if any overturn
    the conclusion.

    Parameters
    ----------
    n_samples : int
        Number of random edit sets to try.
    seed : int
        Random seed.
    """

    def __init__(self, n_samples: int = 100, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def run(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int = 5,
    ) -> tuple[int, tuple[StructuralEdit, ...]] | None:
        """Run random sampling.

        Returns
        -------
        tuple[int, tuple[StructuralEdit, ...]] | None
            ``(cost, edits)`` if a witness is found, else ``None``.
        """
        from causalcert.dag.edit import all_single_edits, apply_edits

        candidates = all_single_edits(adj)
        if not candidates:
            return None

        rng = np.random.default_rng(self.seed)
        best: tuple[int, tuple[StructuralEdit, ...]] | None = None

        for _ in range(self.n_samples):
            k = rng.integers(1, min(max_k, len(candidates)) + 1)
            indices = rng.choice(len(candidates), size=k, replace=False)
            edits = [candidates[i] for i in indices]

            trial = apply_edits(adj, edits)
            from causalcert.dag.validation import is_dag
            if not is_dag(trial):
                continue

            if not predicate(
                trial, data, treatment=treatment, outcome=outcome
            ):
                cost = len(edits)
                if best is None or cost < best[0]:
                    best = (cost, tuple(edits))

        return best


# ---------------------------------------------------------------------------
# Convenience: run all primal heuristics
# ---------------------------------------------------------------------------


def run_primal_heuristics(
    adj: AdjacencyMatrix,
    predicate: ConclusionPredicate,
    data: Any,
    treatment: NodeId,
    outcome: NodeId,
    max_k: int = 10,
    bound_mgr: BoundManager | None = None,
) -> BoundManager:
    """Run all primal heuristics and update bounds.

    Parameters
    ----------
    adj : AdjacencyMatrix
    predicate : ConclusionPredicate
    data : Any
    treatment, outcome : NodeId
    max_k : int
    bound_mgr : BoundManager | None
        Existing manager or create new.

    Returns
    -------
    BoundManager
    """
    if bound_mgr is None:
        bound_mgr = BoundManager(max_k)

    # Greedy heuristic
    greedy = GreedyPrimalHeuristic(max_edits=max_k)
    result = greedy.run(adj, predicate, data, treatment, outcome)
    if result is not None:
        cost, edits = result
        bound_mgr.update_upper(cost, edits, SolverStrategy.CDCL)

    # Random sampling
    sampler = RandomSamplingHeuristic(n_samples=200, seed=42)
    result = sampler.run(adj, predicate, data, treatment, outcome, max_k=max_k)
    if result is not None:
        cost, edits = result
        bound_mgr.update_upper(cost, edits, SolverStrategy.CDCL)

    return bound_mgr
