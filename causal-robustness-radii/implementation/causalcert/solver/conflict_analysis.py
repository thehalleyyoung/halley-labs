"""
Advanced conflict analysis for CDCL-based robustness radius search.

Implements first-UIP (Unique Implication Point) learning, conflict clause
minimisation (recursive and local), backtrack level computation, and
decision-level tracking via the implication graph.

The key adaptation for DAG edit search is that "variables" are structural
edit operations (add / delete / reverse an edge) and "clauses" are
constraints requiring at least one edit from a set.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Sequence

from causalcert.solver.clause_database import Clause, ClauseDatabase, EditLiteral
from causalcert.solver.watched_literals import (
    ImplicationRecord,
    LiteralValue,
    WatchedLiteralEngine,
)
from causalcert.types import StructuralEdit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conflict analysis result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ConflictResult:
    """Result of conflict analysis.

    Attributes
    ----------
    learned_clause : Clause
        The 1-UIP conflict clause to add to the database.
    backtrack_level : int
        The decision level to backtrack to.
    lbd : int
        Literal Block Distance of the learned clause.
    asserting_literal : EditLiteral | None
        The literal that will be propagated after backtracking (the UIP).
    involved_edits : frozenset[StructuralEdit]
        All edits that participated in the derivation (for VSIDS bumping).
    """

    learned_clause: Clause
    backtrack_level: int
    lbd: int
    asserting_literal: EditLiteral | None = None
    involved_edits: frozenset[StructuralEdit] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Implication graph
# ---------------------------------------------------------------------------

class ImplicationGraph:
    """Sparse implication graph built from the assignment trail.

    Each node is an assigned literal.  Edges point from the literals in the
    antecedent clause to the implied literal.  The graph is used to
    compute the 1-UIP cut.

    Parameters
    ----------
    engine : WatchedLiteralEngine
        The propagation engine whose trail defines the graph.
    """

    def __init__(self, engine: WatchedLiteralEngine) -> None:
        self._engine = engine

    def antecedent(self, edit: StructuralEdit) -> Clause | None:
        """Return the clause that implied *edit*, or ``None`` for decisions."""
        return self._engine.reason_of(edit)

    def decision_level(self, edit: StructuralEdit) -> int:
        """Return the decision level at which *edit* was assigned."""
        return self._engine.level_of(edit)

    def is_decision(self, edit: StructuralEdit) -> bool:
        """Return ``True`` if *edit* was a branching decision."""
        return self._engine.reason_of(edit) is None

    def predecessors(self, edit: StructuralEdit) -> list[EditLiteral]:
        """Return the literals in the antecedent of *edit*.

        These are the nodes with edges pointing *into* the node for *edit*
        in the implication graph.
        """
        ante = self.antecedent(edit)
        if ante is None:
            return []
        return [lit for lit in ante.literals if lit.edit != edit]

    def trail_order(self) -> list[ImplicationRecord]:
        """Return the trail in assignment order."""
        return self._engine.trail


# ---------------------------------------------------------------------------
# Conflict analyser
# ---------------------------------------------------------------------------

class ConflictAnalyzer:
    """First-UIP conflict analysis with clause minimisation.

    Parameters
    ----------
    engine : WatchedLiteralEngine
        Propagation engine (provides the assignment trail).
    clause_db : ClauseDatabase
        Clause database for clause operations.
    minimize_mode : str
        Minimisation mode: ``"recursive"``, ``"local"``, or ``"none"``.
    """

    def __init__(
        self,
        engine: WatchedLiteralEngine,
        clause_db: ClauseDatabase,
        minimize_mode: str = "recursive",
    ) -> None:
        self._engine = engine
        self._db = clause_db
        self._minimize_mode = minimize_mode

        # Temporary state for a single analysis run
        self._seen: set[StructuralEdit] = set()
        self._involved: set[StructuralEdit] = set()
        self._learnt_lits: list[EditLiteral] = []
        self._n_analyses: int = 0

    # -- main entry ---------------------------------------------------------

    def analyze(self, conflict_clause: Clause) -> ConflictResult:
        """Perform 1-UIP conflict analysis.

        Starting from the *conflict_clause* (the clause whose all literals are
        falsified), resolve backward along the implication graph until exactly
        one literal from the current decision level remains — the first
        Unique Implication Point (UIP).

        Parameters
        ----------
        conflict_clause : Clause
            The clause that triggered the conflict.

        Returns
        -------
        ConflictResult
            Learned clause, backtrack level, LBD, and metadata.
        """
        self._n_analyses += 1
        self._seen.clear()
        self._involved.clear()
        self._learnt_lits.clear()

        current_level = self._engine.decision_level
        if current_level == 0:
            # Conflict at root — problem is UNSAT
            return ConflictResult(
                learned_clause=Clause(literals=()),
                backtrack_level=-1,
                lbd=0,
                asserting_literal=None,
                involved_edits=frozenset(),
            )

        # Counter of literals at the current decision level that we still
        # need to resolve away
        n_at_current = 0

        # Seed from the conflict clause
        for lit in conflict_clause.literals:
            edit = lit.edit
            if edit not in self._seen:
                self._seen.add(edit)
                self._involved.add(edit)
                lvl = self._engine.level_of(edit)
                if lvl == current_level:
                    n_at_current += 1
                else:
                    self._learnt_lits.append(lit)

        # Walk the trail backwards, resolving literals at the current level
        trail = self._engine.trail
        trail_idx = len(trail) - 1
        asserting_lit: EditLiteral | None = None

        while n_at_current > 1 and trail_idx >= 0:
            rec = trail[trail_idx]
            trail_idx -= 1
            edit = rec.literal.edit

            if edit not in self._seen:
                continue

            reason = rec.reason
            if reason is None:
                # Decision variable — cannot resolve further
                continue

            # This literal is at the current level and has an antecedent:
            # resolve it away
            n_at_current -= 1

            for lit in reason.literals:
                other = lit.edit
                if other not in self._seen:
                    self._seen.add(other)
                    self._involved.add(other)
                    lvl = self._engine.level_of(other)
                    if lvl == current_level:
                        n_at_current += 1
                    elif lvl > 0:
                        self._learnt_lits.append(lit)

        # The remaining literal at the current level is the UIP
        # Walk trail again to find it
        for idx in range(len(trail) - 1, -1, -1):
            rec = trail[idx]
            if rec.literal.edit in self._seen and rec.decision_level == current_level:
                asserting_lit = rec.literal.negated()
                break

        if asserting_lit is not None:
            # Place the asserting literal first in the clause
            all_lits = [asserting_lit] + self._learnt_lits
        else:
            all_lits = list(self._learnt_lits)

        # Minimise
        if self._minimize_mode == "recursive":
            all_lits = self._minimize_recursive(all_lits, current_level)
        elif self._minimize_mode == "local":
            all_lits = self._minimize_local(all_lits, current_level)

        # Compute backtrack level: second-highest decision level in clause
        bt_level = self._compute_backtrack_level(all_lits)

        # Compute LBD
        lbd = self._compute_lbd(all_lits)

        learned = Clause(
            literals=tuple(all_lits),
            lbd=lbd,
            reason="1-UIP conflict clause",
        )

        return ConflictResult(
            learned_clause=learned,
            backtrack_level=bt_level,
            lbd=lbd,
            asserting_literal=asserting_lit,
            involved_edits=frozenset(self._involved),
        )

    # -- backtrack level ----------------------------------------------------

    def _compute_backtrack_level(self, lits: list[EditLiteral]) -> int:
        """Compute the backtrack level: second-highest decision level.

        If the clause is unit the backtrack level is 0.
        """
        if len(lits) <= 1:
            return 0

        levels = sorted(
            {self._engine.level_of(lit.edit) for lit in lits},
            reverse=True,
        )
        return levels[1] if len(levels) >= 2 else 0

    # -- LBD computation ----------------------------------------------------

    def _compute_lbd(self, lits: list[EditLiteral]) -> int:
        """Compute the Literal Block Distance of a clause.

        LBD is the number of distinct decision levels among the clause's
        literals.  Lower LBD indicates a higher-quality (more "glue") clause.
        """
        return len({self._engine.level_of(lit.edit) for lit in lits})

    # -- recursive minimisation --------------------------------------------

    def _minimize_recursive(
        self,
        lits: list[EditLiteral],
        current_level: int,
    ) -> list[EditLiteral]:
        """Recursive clause minimisation.

        A literal *l* is redundant if every path from *l* in the implication
        graph leads to another literal already in the clause (or to the
        decision variable of *l*'s level, which is implicitly in the clause).

        This follows the MiniSat approach to recursive self-subsuming
        resolution.
        """
        clause_edits = {lit.edit for lit in lits}
        clause_levels = {self._engine.level_of(lit.edit) for lit in lits}
        result: list[EditLiteral] = []
        cache: dict[StructuralEdit, bool] = {}

        for i, lit in enumerate(lits):
            # Never remove the asserting literal (first position)
            if i == 0:
                result.append(lit)
                continue
            if self._is_redundant_recursive(lit.edit, clause_edits, clause_levels, cache):
                continue
            result.append(lit)

        return result

    def _is_redundant_recursive(
        self,
        edit: StructuralEdit,
        clause_edits: set[StructuralEdit],
        clause_levels: set[int],
        cache: dict[StructuralEdit, bool],
    ) -> bool:
        """Check if *edit* is redundant via recursive analysis."""
        if edit in cache:
            return cache[edit]

        reason = self._engine.reason_of(edit)
        if reason is None:
            # Decision variable — not redundant unless it's already in clause
            cache[edit] = edit in clause_edits
            return cache[edit]

        # Check all antecedent literals
        for lit in reason.literals:
            other = lit.edit
            if other == edit:
                continue
            if other in clause_edits:
                continue
            lvl = self._engine.level_of(other)
            if lvl == 0:
                # Root-level assignments are always safe
                continue
            if lvl not in clause_levels:
                # Level not represented in clause — can't resolve away
                cache[edit] = False
                return False
            if not self._is_redundant_recursive(other, clause_edits, clause_levels, cache):
                cache[edit] = False
                return False

        cache[edit] = True
        return True

    # -- local minimisation ------------------------------------------------

    def _minimize_local(
        self,
        lits: list[EditLiteral],
        current_level: int,
    ) -> list[EditLiteral]:
        """Local (non-recursive) clause minimisation.

        A literal *l* is removed if its antecedent clause has all non-*l*
        literals already present in the learned clause.  This is cheaper
        than the recursive version but removes fewer literals.
        """
        clause_edits = {lit.edit for lit in lits}
        result: list[EditLiteral] = []

        for i, lit in enumerate(lits):
            if i == 0:
                result.append(lit)
                continue

            reason = self._engine.reason_of(lit.edit)
            if reason is None:
                result.append(lit)
                continue

            # Check: are all other literals in the antecedent in the clause?
            all_in_clause = True
            for ante_lit in reason.literals:
                if ante_lit.edit == lit.edit:
                    continue
                if self._engine.level_of(ante_lit.edit) == 0:
                    continue
                if ante_lit.edit not in clause_edits:
                    all_in_clause = False
                    break

            if not all_in_clause:
                result.append(lit)
            # else: lit is redundant — skip it

        return result

    # -- on-the-fly self-subsumption ----------------------------------------

    def on_the_fly_subsumption(
        self,
        learned: Clause,
        conflict_clause: Clause,
    ) -> Clause | None:
        """Attempt on-the-fly subsumption of the original conflict clause.

        If the learned clause subsumes *conflict_clause* after removing one
        literal, that literal can be removed from the original clause.

        Parameters
        ----------
        learned : Clause
            The just-learned clause.
        conflict_clause : Clause
            The original clause that triggered the conflict.

        Returns
        -------
        Clause | None
            Strengthened conflict clause, or ``None`` if no subsumption.
        """
        learned_set = set(learned.literals)
        conflict_set = set(conflict_clause.literals)

        diff = conflict_set - learned_set
        if len(diff) == 1:
            # Can strengthen conflict_clause by removing the extra literal
            new_lits = tuple(lit for lit in conflict_clause.literals if lit not in diff)
            return Clause(
                literals=new_lits,
                activity=conflict_clause.activity,
                lbd=conflict_clause.lbd,
                is_learned=conflict_clause.is_learned,
                reason=conflict_clause.reason + " (strengthened by OTF subsumption)",
                cid=conflict_clause.cid,
            )
        return None

    # -- clause-level analysis helpers --------------------------------------

    def decision_levels_in_clause(self, clause: Clause) -> set[int]:
        """Return the set of decision levels represented in *clause*."""
        return {self._engine.level_of(lit.edit) for lit in clause.literals}

    def n_literals_at_level(self, clause: Clause, level: int) -> int:
        """Count literals in *clause* assigned at *level*."""
        return sum(
            1 for lit in clause.literals
            if self._engine.level_of(lit.edit) == level
        )

    # -- implication graph traversal ----------------------------------------

    def collect_reasons(self, edit: StructuralEdit) -> list[Clause]:
        """Collect all antecedent clauses in the implication chain of *edit*.

        Traverses the implication graph backwards from *edit* to the
        decision variables, collecting every antecedent clause encountered.

        Parameters
        ----------
        edit : StructuralEdit
            The edit to trace.

        Returns
        -------
        list[Clause]
            Antecedent clauses in reverse trail order.
        """
        visited: set[StructuralEdit] = set()
        clauses: list[Clause] = []
        queue: deque[StructuralEdit] = deque([edit])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            reason = self._engine.reason_of(current)
            if reason is None:
                continue
            clauses.append(reason)
            for lit in reason.literals:
                if lit.edit != current and lit.edit not in visited:
                    queue.append(lit.edit)

        return clauses

    def compute_1uip_cut(self, conflict_clause: Clause) -> set[EditLiteral]:
        """Compute the 1-UIP cut of the implication graph.

        The 1-UIP cut separates the conflict node from the most recent
        decision on the trail such that there is a single path from the
        UIP to the conflict.

        Parameters
        ----------
        conflict_clause : Clause
            The conflicting clause.

        Returns
        -------
        set[EditLiteral]
            Literals on the reason side of the 1-UIP cut.
        """
        result = self.analyze(conflict_clause)
        return set(result.learned_clause.literals)

    # -- statistics ---------------------------------------------------------

    @property
    def n_analyses(self) -> int:
        """Total number of conflict analyses performed."""
        return self._n_analyses

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ConflictAnalyzer(mode={self._minimize_mode!r}, "
            f"analyses={self._n_analyses})"
        )
