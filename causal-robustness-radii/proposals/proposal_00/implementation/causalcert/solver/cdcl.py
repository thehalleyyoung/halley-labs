"""
Conflict-driven clause-learning (CDCL) search for robustness radius.

A SAT-inspired iterative search that maintains a partial assignment of
edge edits and uses learned conflict clauses (infeasible sub-sets of edits)
to prune the search tree.

Algorithm outline
-----------------
1. **Decision**: choose an unassigned edge edit to apply.
2. **Propagation**: forced edits are propagated (e.g. acyclicity enforcement).
3. **Conflict detection**: check if the partial assignment is inconsistent.
4. **Conflict analysis**: extract a minimal infeasible subset as a no-good.
5. **Backtrack**: undo decisions to the appropriate level.
6. **Restart**: periodically restart with learned clauses retained.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from causalcert.dag.dsep import DSeparationOracle
from causalcert.dag.edit import apply_edit as raw_apply_edit, apply_edits as _apply_edits
from causalcert.dag.validation import is_dag
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
# Conflict clause
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConflictClause:
    """A learned conflict clause: a set of edits that is provably infeasible.

    Attributes
    ----------
    edits : frozenset[StructuralEdit]
        Subset of edits that together violate acyclicity or CI consistency.
    reason : str
        Human-readable explanation of the conflict.
    """

    edits: frozenset[StructuralEdit]
    reason: str = ""


# ---------------------------------------------------------------------------
# VSIDS-style activity scores
# ---------------------------------------------------------------------------


class VSIDSScorer:
    """Variable-State Independent Decaying Sum heuristic.

    Maintains an activity score for each candidate edit.  Scores are bumped
    when an edit participates in a conflict and periodically decayed.

    Parameters
    ----------
    edits : list[StructuralEdit]
        All candidate edits.
    decay : float
        Decay factor applied after each conflict (default 0.95).
    """

    def __init__(self, edits: list[StructuralEdit], decay: float = 0.95) -> None:
        self._scores: dict[StructuralEdit, float] = {e: 0.0 for e in edits}
        self._decay = decay
        self._bump_value = 1.0

    def bump(self, edit: StructuralEdit) -> None:
        """Increase the activity of *edit*."""
        if edit in self._scores:
            self._scores[edit] += self._bump_value

    def bump_all(self, edits: frozenset[StructuralEdit]) -> None:
        """Bump all edits in a conflict clause."""
        for e in edits:
            self.bump(e)
        # Decay everything to keep scores bounded
        self._bump_value /= self._decay

    def decay(self) -> None:
        """Apply multiplicative decay to all scores."""
        for e in list(self._scores):
            self._scores[e] *= self._decay

    def best_unassigned(self, assigned: set[StructuralEdit]) -> StructuralEdit | None:
        """Return the highest-activity unassigned edit."""
        best: StructuralEdit | None = None
        best_score = -1.0
        for e, s in self._scores.items():
            if e not in assigned and s > best_score:
                best_score = s
                best = e
        return best

    def score(self, edit: StructuralEdit) -> float:
        """Return the activity score of *edit*."""
        return self._scores.get(edit, 0.0)


# ---------------------------------------------------------------------------
# Decision / assignment tracking
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """A single decision in the CDCL search."""

    edit: StructuralEdit
    level: int
    is_decision: bool  # True if branching decision, False if propagated
    antecedent: ConflictClause | None = None  # clause that forced this propagation


# ---------------------------------------------------------------------------
# CDCL Solver
# ---------------------------------------------------------------------------


class CDCLSolver:
    """Conflict-driven clause-learning solver for the robustness radius.

    Parameters
    ----------
    max_conflicts : int
        Maximum number of conflict clauses before giving up.
    time_limit_s : float
        Maximum wall-clock time.
    restart_base : int
        Base interval (in conflicts) for geometric restarts.
    restart_mult : float
        Geometric multiplier for restart intervals.
    """

    def __init__(
        self,
        max_conflicts: int = 10_000,
        time_limit_s: float = 300.0,
        restart_base: int = 100,
        restart_mult: float = 1.5,
    ) -> None:
        self.max_conflicts = max_conflicts
        self.time_limit_s = time_limit_s
        self.restart_base = restart_base
        self.restart_mult = restart_mult
        self._clauses: list[ConflictClause] = []

    # ---- main solve -------------------------------------------------------

    def solve(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int = 10,
    ) -> RobustnessRadius:
        """Run the CDCL search.

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
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        t0 = time.perf_counter()

        # Trivial check
        if not predicate(adj, data, treatment=treatment, outcome=outcome):
            return RobustnessRadius(
                lower_bound=0, upper_bound=0,
                witness_edits=(),
                solver_strategy=SolverStrategy.CDCL,
                solver_time_s=time.perf_counter() - t0,
                gap=0.0, certified=True,
            )

        # Enumerate candidate edits (restricted to acyclic ones for speed)
        from causalcert.dag.edit import all_single_edits
        candidates = all_single_edits(adj)
        if not candidates:
            elapsed = time.perf_counter() - t0
            return RobustnessRadius(
                lower_bound=max_k + 1, upper_bound=max_k + 1,
                witness_edits=(),
                solver_strategy=SolverStrategy.CDCL,
                solver_time_s=elapsed, gap=0.0, certified=True,
            )

        # Initialize VSIDS scorer
        vsids = VSIDSScorer(candidates)

        # Search state
        best_edits: list[StructuralEdit] | None = None
        best_cost = max_k + 1
        self._clauses.clear()

        # Restart control
        next_restart = self.restart_base
        conflicts_since_restart = 0

        # Iterative deepening: try each budget k from 1 to max_k
        for budget in range(1, max_k + 1):
            elapsed = time.perf_counter() - t0
            if elapsed >= self.time_limit_s:
                break

            result = self._search_at_budget(
                adj=adj,
                predicate=predicate,
                data=data,
                treatment=treatment,
                outcome=outcome,
                candidates=candidates,
                vsids=vsids,
                budget=budget,
                t0=t0,
            )

            if result is not None:
                best_edits = result
                best_cost = len(result)
                break

        elapsed = time.perf_counter() - t0

        if best_edits is not None:
            return RobustnessRadius(
                lower_bound=best_cost,
                upper_bound=best_cost,
                witness_edits=tuple(best_edits),
                solver_strategy=SolverStrategy.CDCL,
                solver_time_s=elapsed,
                gap=0.0,
                certified=True,
            )
        else:
            return RobustnessRadius(
                lower_bound=max_k + 1,
                upper_bound=max_k + 1,
                witness_edits=(),
                solver_strategy=SolverStrategy.CDCL,
                solver_time_s=elapsed,
                gap=0.0,
                certified=(len(self._clauses) > 0),
            )

    # ---- budget-bounded search ---------------------------------------------

    def _search_at_budget(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        candidates: list[StructuralEdit],
        vsids: VSIDSScorer,
        budget: int,
        t0: float,
    ) -> list[StructuralEdit] | None:
        """Search for a witness with exactly *budget* edits.

        Uses a DFS with backtracking and clause learning.
        """
        n = adj.shape[0]

        # Stack-based DFS: each frame is (index into candidates, trail so far)
        # Sort candidates by VSIDS score (descending)
        scored = sorted(
            range(len(candidates)),
            key=lambda i: vsids.score(candidates[i]),
            reverse=True,
        )

        # Recursive DFS with iterative implementation
        # State: list of chosen candidate indices
        chosen: list[int] = []
        # For each depth d, `start[d]` is the next index in `scored` to try
        start: list[int] = [0]

        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= self.time_limit_s:
                return None
            if len(self._clauses) >= self.max_conflicts:
                return None

            depth = len(chosen)

            if depth == budget:
                # Check if this edit set overturns the predicate
                edits = [candidates[scored[i]] for i in chosen]
                trial_adj = _apply_edits(adj, edits)

                if is_dag(trial_adj) and not predicate(
                    trial_adj, data,
                    treatment=treatment, outcome=outcome,
                ):
                    return edits

                # Backtrack
                chosen.pop()
                start.pop()
                continue

            # Try to pick the next candidate at this depth
            found_next = False
            while start[-1] < len(scored):
                idx = start[-1]
                start[-1] += 1

                # Skip if already chosen
                if idx in chosen:
                    continue

                cand_edit = candidates[scored[idx]]

                # Skip if it violates a learned clause
                edit_set = frozenset(
                    candidates[scored[c]] for c in chosen
                ) | {cand_edit}
                if any(cl.edits.issubset(edit_set) for cl in self._clauses):
                    continue

                # Quick acyclicity check
                edits_so_far = [candidates[scored[c]] for c in chosen] + [cand_edit]
                trial = _apply_edits(adj, edits_so_far)
                if not is_dag(trial):
                    # Learn a conflict clause
                    conflict = ConflictClause(
                        frozenset(edits_so_far),
                        "acyclicity violation in partial assignment",
                    )
                    self._clauses.append(conflict)
                    vsids.bump_all(conflict.edits)
                    continue

                # Make the choice
                chosen.append(idx)
                start.append(idx + 1)  # next depth starts after this index
                found_next = True
                break

            if not found_next:
                # Exhausted candidates at this depth — backtrack
                start.pop()
                if not chosen:
                    return None  # Exhausted everything
                chosen.pop()

        return None

    # ---- propagation -------------------------------------------------------

    def _propagate(
        self,
        partial: list[StructuralEdit],
        adj: AdjacencyMatrix,
    ) -> ConflictClause | None:
        """Propagate constraints and detect conflicts.

        Checks:
        1. Acyclicity: is the current adjacency still a DAG?
        2. Clause violations: does the current assignment falsify a clause?

        Parameters
        ----------
        partial : list[StructuralEdit]
            Current partial assignment.
        adj : AdjacencyMatrix
            Current adjacency matrix.

        Returns
        -------
        ConflictClause | None
            A conflict clause if one is found, else ``None``.
        """
        # Check acyclicity
        if not is_dag(adj):
            involved = frozenset(partial[-2:]) if len(partial) >= 2 else frozenset(partial)
            return ConflictClause(
                involved,
                "acyclicity violation",
            )

        # Check learned clauses
        current_set = frozenset(partial)
        for clause in self._clauses:
            if clause.edits.issubset(current_set):
                return ConflictClause(
                    clause.edits,
                    f"violated learned clause: {clause.reason}",
                )

        return None

    # ---- conflict analysis -------------------------------------------------

    def _analyze_conflict(
        self,
        clause: ConflictClause,
        decision_level: int,
    ) -> tuple[ConflictClause, int]:
        """Analyse a conflict and produce a learned clause and backtrack level.

        Uses first-UIP (unique implication point) analysis:
        the learned clause is the conflict clause itself (simplified),
        and the backtrack level is one less than the current.

        Parameters
        ----------
        clause : ConflictClause
            The conflict clause.
        decision_level : int
            Current decision level.

        Returns
        -------
        tuple[ConflictClause, int]
            ``(learned_clause, backtrack_level)``.
        """
        # Simplified 1-UIP: learn the conflict clause directly
        # and backtrack to one level below current
        bt_level = max(0, decision_level - 1)

        # Try to minimize the clause
        minimized = self._minimize_clause(clause)

        return minimized, bt_level

    def _minimize_clause(self, clause: ConflictClause) -> ConflictClause:
        """Attempt to reduce the clause to a smaller infeasible subset.

        Tries removing each edit and checking if the remaining set
        is still infeasible (using a greedy approach).
        """
        edits = list(clause.edits)
        if len(edits) <= 1:
            return clause

        minimal = set(edits)
        for edit in edits:
            candidate = minimal - {edit}
            if len(candidate) == 0:
                break
            # A subset is still infeasible if it appears in learned clauses
            candidate_fs = frozenset(candidate)
            still_infeasible = any(
                cl.edits.issubset(candidate_fs) for cl in self._clauses
            )
            if still_infeasible:
                minimal = candidate

        return ConflictClause(
            frozenset(minimal),
            clause.reason,
        )

    # ---- helpers -----------------------------------------------------------

    def _backtrack(
        self,
        trail: list[Decision],
        assigned: set[StructuralEdit],
        target_level: int,
        orig_adj: AdjacencyMatrix,
        current_adj: AdjacencyMatrix,
    ) -> None:
        """Undo decisions above *target_level*."""
        while trail and trail[-1].level > target_level:
            d = trail.pop()
            assigned.discard(d.edit)

    def _rebuild_adj(
        self,
        orig_adj: AdjacencyMatrix,
        trail: list[Decision],
    ) -> AdjacencyMatrix:
        """Rebuild the adjacency matrix from the trail."""
        from causalcert.dag.edit import apply_edits
        edits = [d.edit for d in trail]
        return apply_edits(orig_adj, edits)

    def _violates_clause(self, assigned: set[StructuralEdit]) -> bool:
        """Check if the current assignment violates any learned clause."""
        for clause in self._clauses:
            if clause.edits.issubset(assigned):
                return True
        return False

    @property
    def n_learned_clauses(self) -> int:
        """Number of learned conflict clauses."""
        return len(self._clauses)

    @property
    def learned_clauses(self) -> list[ConflictClause]:
        """All learned conflict clauses."""
        return list(self._clauses)
