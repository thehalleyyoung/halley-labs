"""
Two-watched-literal scheme for constraint propagation in CDCL search.

Adapts the classic two-watched-literal data structure from SAT solvers to
the DAG edit search context.  Each clause (constraint) has two designated
*watched* literals.  When an assignment falsifies a watched literal the
engine scans the clause for a replacement; if none exists and only one
literal remains unassigned the clause is *unit* and that literal is propagated.

This yields amortised O(1) per-assignment cost for propagation instead of
scanning every clause.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from causalcert.solver.clause_database import Clause, ClauseDatabase, EditLiteral
from causalcert.types import StructuralEdit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assignment value
# ---------------------------------------------------------------------------

class LiteralValue:
    """Tri-state assignment for an edit literal."""

    TRUE = 1
    FALSE = -1
    UNDEF = 0


# ---------------------------------------------------------------------------
# Propagation result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PropagationResult:
    """Outcome of a propagation round.

    Attributes
    ----------
    conflict : Clause | None
        The clause that became empty (conflict), or ``None``.
    propagated : list[EditLiteral]
        Literals forced to true by unit propagation during this round.
    """

    conflict: Clause | None = None
    propagated: list[EditLiteral] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Implication record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ImplicationRecord:
    """Record of why a literal was propagated.

    Attributes
    ----------
    literal : EditLiteral
        The propagated literal.
    reason : Clause | None
        The antecedent clause that forced this propagation, or ``None`` for
        decision literals.
    decision_level : int
        The decision level at which this literal was assigned.
    trail_index : int
        Position of this literal on the assignment trail.
    """

    literal: EditLiteral
    reason: Clause | None
    decision_level: int
    trail_index: int


# ---------------------------------------------------------------------------
# Watched literal engine
# ---------------------------------------------------------------------------

class WatchedLiteralEngine:
    """Two-watched-literal propagation engine.

    Parameters
    ----------
    clause_db : ClauseDatabase
        The clause database to draw constraints from.

    Notes
    -----
    The engine maintains:

    * **Watch lists**: for each literal *l*, the set of clause ids where *l*
      is one of the two watched literals.  When *l* is falsified the engine
      visits those clauses to find a replacement watch.
    * **Trail**: an ordered sequence of assigned literals with their antecedent
      clauses, supporting efficient backtracking.
    * **Assignment map**: edit → ``LiteralValue`` for fast lookup.
    """

    def __init__(self, clause_db: ClauseDatabase) -> None:
        self._db = clause_db

        # watch_list[lit] -> list of clause ids where lit is watched
        self._watch_list: dict[EditLiteral, list[int]] = defaultdict(list)

        # Two watched literals per clause: clause_id -> (lit0, lit1)
        self._watches: dict[int, list[EditLiteral]] = {}

        # Assignment: edit -> LiteralValue
        self._assignment: dict[StructuralEdit, int] = {}

        # Trail of assigned literals (in order)
        self._trail: list[ImplicationRecord] = []

        # Decision level boundaries: _trail_lim[k] = trail index where
        # decision level k+1 starts
        self._trail_lim: list[int] = []

        # Current decision level
        self._decision_level: int = 0

        # Propagation queue
        self._prop_queue: list[EditLiteral] = []

        self._init_watches()

    # -- initialisation -----------------------------------------------------

    def _init_watches(self) -> None:
        """Set up initial watches for all clauses in the database."""
        for clause in self._db.all_clauses:
            self._attach_clause(clause)

    def attach_clause(self, clause: Clause) -> None:
        """Register watches for a newly added clause.

        Parameters
        ----------
        clause : Clause
            The clause to watch.
        """
        self._attach_clause(clause)

    def _attach_clause(self, clause: Clause) -> None:
        """Internal: pick two watched literals and install watch-list entries."""
        if clause.size == 0:
            return

        lits = list(clause.literals)

        # Try to pick two non-false literals
        watchers: list[EditLiteral] = []
        for lit in lits:
            val = self._eval_literal(lit)
            if val != LiteralValue.FALSE:
                watchers.append(lit)
            if len(watchers) == 2:
                break
        # Fill with remaining if needed
        for lit in lits:
            if lit not in watchers:
                watchers.append(lit)
            if len(watchers) == 2:
                break
        if len(watchers) == 1:
            watchers.append(watchers[0])  # degenerate unit clause

        self._watches[clause.cid] = watchers[:2]
        self._watch_list[watchers[0]].append(clause.cid)
        if len(watchers) >= 2 and watchers[1] != watchers[0]:
            self._watch_list[watchers[1]].append(clause.cid)

    # -- assignment ---------------------------------------------------------

    def assign_decision(self, literal: EditLiteral) -> None:
        """Record a branching decision.

        Parameters
        ----------
        literal : EditLiteral
            The literal chosen as the next decision.
        """
        self._decision_level += 1
        self._trail_lim.append(len(self._trail))
        self._enqueue(literal, reason=None)

    def _enqueue(self, literal: EditLiteral, reason: Clause | None) -> None:
        """Add a literal to the trail and assignment map."""
        edit = literal.edit
        val = LiteralValue.TRUE if literal.positive else LiteralValue.FALSE
        self._assignment[edit] = val
        rec = ImplicationRecord(
            literal=literal,
            reason=reason,
            decision_level=self._decision_level,
            trail_index=len(self._trail),
        )
        self._trail.append(rec)
        self._prop_queue.append(literal)

    def _eval_literal(self, lit: EditLiteral) -> int:
        """Evaluate a literal under the current assignment.

        Returns
        -------
        int
            ``LiteralValue.TRUE``, ``FALSE``, or ``UNDEF``.
        """
        val = self._assignment.get(lit.edit, LiteralValue.UNDEF)
        if val == LiteralValue.UNDEF:
            return LiteralValue.UNDEF
        if lit.positive:
            return val
        return -val

    # -- propagation --------------------------------------------------------

    def propagate(self) -> PropagationResult:
        """Run BCP (Boolean Constraint Propagation) to a fixpoint.

        Processes every literal in the propagation queue.  For each falsified
        watched literal the corresponding clauses are scanned for a
        replacement watch.  If none exists and only one literal remains
        un-falsified, that literal is propagated.  If all literals are falsified
        a conflict is returned.

        Returns
        -------
        PropagationResult
            Propagated literals and an optional conflict clause.
        """
        result = PropagationResult()

        while self._prop_queue:
            literal = self._prop_queue.pop(0)
            # The negation of the assigned literal is the one that becomes false
            false_lit = literal.negated()
            conflict = self._propagate_literal(false_lit, result)
            if conflict is not None:
                result.conflict = conflict
                self._prop_queue.clear()
                return result

        return result

    def _propagate_literal(
        self,
        false_lit: EditLiteral,
        result: PropagationResult,
    ) -> Clause | None:
        """Process all clauses watching *false_lit*.

        Returns a conflict clause if one is detected, else ``None``.
        """
        # Iterate over clauses watching false_lit
        watch_ids = self._watch_list.get(false_lit, [])
        new_watch_list: list[int] = []

        i = 0
        conflict: Clause | None = None

        while i < len(watch_ids):
            cid = watch_ids[i]
            i += 1

            clause = self._db.get(cid)
            if clause is None:
                continue

            watchers = self._watches.get(cid)
            if watchers is None:
                continue

            # Make false_lit the first watcher for convenience
            if len(watchers) >= 2 and watchers[1] == false_lit:
                watchers[0], watchers[1] = watchers[1], watchers[0]

            if watchers[0] != false_lit:
                # This watcher was already replaced; keep in list
                new_watch_list.append(cid)
                continue

            # Check if the other watcher is satisfied
            if len(watchers) >= 2:
                other = watchers[1]
                if self._eval_literal(other) == LiteralValue.TRUE:
                    new_watch_list.append(cid)
                    continue

            # Look for a replacement watcher
            found_replacement = False
            for lit in clause.literals:
                if len(watchers) >= 2 and lit == watchers[1]:
                    continue
                if lit == false_lit:
                    continue
                if self._eval_literal(lit) != LiteralValue.FALSE:
                    # Found a replacement
                    watchers[0] = lit
                    self._watch_list[lit].append(cid)
                    found_replacement = True
                    break

            if found_replacement:
                continue

            # No replacement found — clause is unit or conflicting
            new_watch_list.append(cid)

            if len(watchers) < 2:
                # Unit clause that is now empty
                conflict = clause
                break

            other = watchers[1]
            other_val = self._eval_literal(other)

            if other_val == LiteralValue.FALSE:
                # Conflict: all literals are false
                conflict = clause
                break
            elif other_val == LiteralValue.UNDEF:
                # Unit propagation
                self._enqueue(other, reason=clause)
                result.propagated.append(other)

        # Rebuild the watch list for false_lit
        remaining = watch_ids[i:]
        self._watch_list[false_lit] = new_watch_list + remaining

        return conflict

    # -- backtracking -------------------------------------------------------

    def backtrack(self, target_level: int) -> list[EditLiteral]:
        """Undo assignments above *target_level*.

        Parameters
        ----------
        target_level : int
            The decision level to backtrack to (assignments at this level
            are retained).

        Returns
        -------
        list[EditLiteral]
            Literals that were unassigned.
        """
        unassigned: list[EditLiteral] = []

        while self._decision_level > target_level:
            if not self._trail_lim:
                break
            bt_point = self._trail_lim.pop()
            while len(self._trail) > bt_point:
                rec = self._trail.pop()
                del self._assignment[rec.literal.edit]
                unassigned.append(rec.literal)
            self._decision_level -= 1

        self._prop_queue.clear()
        return unassigned

    def backtrack_to_root(self) -> list[EditLiteral]:
        """Undo all assignments, returning to decision level 0."""
        return self.backtrack(0)

    # -- queries ------------------------------------------------------------

    @property
    def decision_level(self) -> int:
        """Current decision level."""
        return self._decision_level

    @property
    def trail(self) -> list[ImplicationRecord]:
        """Current assignment trail (read-only copy)."""
        return list(self._trail)

    @property
    def trail_size(self) -> int:
        """Number of currently assigned literals."""
        return len(self._trail)

    def value(self, edit: StructuralEdit) -> int:
        """Return the assignment value of *edit*."""
        return self._assignment.get(edit, LiteralValue.UNDEF)

    def literal_value(self, lit: EditLiteral) -> int:
        """Return the truth value of *lit* under current assignment."""
        return self._eval_literal(lit)

    def is_assigned(self, edit: StructuralEdit) -> bool:
        """Return ``True`` if *edit* has been assigned."""
        return edit in self._assignment

    def assigned_edits(self) -> set[StructuralEdit]:
        """Return the set of all currently assigned edits."""
        return set(self._assignment.keys())

    def active_edits(self) -> list[StructuralEdit]:
        """Return edits currently assigned to ``True``."""
        return [
            edit for edit, val in self._assignment.items()
            if val == LiteralValue.TRUE
        ]

    def level_of(self, edit: StructuralEdit) -> int:
        """Return the decision level at which *edit* was assigned, or -1."""
        for rec in self._trail:
            if rec.literal.edit == edit:
                return rec.decision_level
        return -1

    def reason_of(self, edit: StructuralEdit) -> Clause | None:
        """Return the antecedent clause of *edit*, or ``None`` if decision."""
        for rec in self._trail:
            if rec.literal.edit == edit:
                return rec.reason
        return None

    def trail_record(self, edit: StructuralEdit) -> ImplicationRecord | None:
        """Return the full trail record for *edit*."""
        for rec in self._trail:
            if rec.literal.edit == edit:
                return rec
        return None

    def literals_at_level(self, level: int) -> list[EditLiteral]:
        """Return all literals assigned at *level*."""
        return [rec.literal for rec in self._trail if rec.decision_level == level]

    def n_vars_at_level(self, level: int) -> int:
        """Number of variables assigned at *level*."""
        return sum(1 for rec in self._trail if rec.decision_level == level)

    # -- detach / re-attach for clause deletion -----------------------------

    def detach_clause(self, clause: Clause) -> None:
        """Remove watches for a clause (prior to deletion)."""
        watchers = self._watches.pop(clause.cid, None)
        if watchers is None:
            return
        for w in watchers:
            wl = self._watch_list.get(w, [])
            try:
                wl.remove(clause.cid)
            except ValueError:
                pass

    def reattach_clause(self, clause: Clause) -> None:
        """Re-attach watches for a clause (after database restore)."""
        self._attach_clause(clause)

    # -- stats --------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"WatchedLiteralEngine(level={self._decision_level}, "
            f"trail={len(self._trail)}, queue={len(self._prop_queue)})"
        )
