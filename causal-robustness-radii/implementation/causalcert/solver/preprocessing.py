"""
Constraint preprocessing for CDCL-based robustness radius search.

Applies simplification passes to the clause database before (and during)
search.  Includes unit propagation at the root level, pure literal
elimination, blocked edit detection, variable elimination via bounded
resolution, and subsumption checking.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from causalcert.solver.clause_database import (
    Clause,
    ClauseDatabase,
    ClauseStats,
    EditLiteral,
)
from causalcert.types import EditType, StructuralEdit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing statistics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PreprocessStats:
    """Statistics gathered during preprocessing.

    Attributes
    ----------
    n_unit_props : int
        Literals fixed by root-level unit propagation.
    n_pure_lits : int
        Variables eliminated by pure-literal rule.
    n_blocked_edits : int
        Variables eliminated by blocked-edit detection.
    n_var_elim : int
        Variables eliminated by bounded resolution.
    n_subsumed : int
        Clauses removed by subsumption.
    n_strengthened : int
        Clauses strengthened by self-subsumption.
    n_clauses_removed : int
        Total clauses removed.
    n_rounds : int
        Preprocessing rounds executed.
    """

    n_unit_props: int = 0
    n_pure_lits: int = 0
    n_blocked_edits: int = 0
    n_var_elim: int = 0
    n_subsumed: int = 0
    n_strengthened: int = 0
    n_clauses_removed: int = 0
    n_rounds: int = 0


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """Constraint preprocessor for the edit-search clause database.

    Applies a sequence of simplification techniques to reduce the size
    of the problem before CDCL search begins.

    Parameters
    ----------
    clause_db : ClauseDatabase
        The clause database to simplify.
    max_resolution_size : int
        Maximum clause size produced by variable elimination (bounded
        resolution).  Larger resolvents are rejected.
    max_rounds : int
        Maximum preprocessing iterations.
    enable_var_elim : bool
        Whether to enable variable elimination via bounded resolution.
    """

    def __init__(
        self,
        clause_db: ClauseDatabase,
        max_resolution_size: int = 20,
        max_rounds: int = 5,
        enable_var_elim: bool = True,
    ) -> None:
        self._db = clause_db
        self._max_res_size = max_resolution_size
        self._max_rounds = max_rounds
        self._enable_var_elim = enable_var_elim

        self._fixed: dict[StructuralEdit, bool] = {}
        self._eliminated: set[StructuralEdit] = set()
        self._stats = PreprocessStats()

    # -- main entry ---------------------------------------------------------

    def preprocess(self) -> PreprocessStats:
        """Run all preprocessing passes.

        Returns
        -------
        PreprocessStats
            Summary of work done.
        """
        for round_idx in range(self._max_rounds):
            self._stats.n_rounds += 1
            changed = False

            # 1. Root-level unit propagation
            n_up = self._unit_propagate()
            if n_up > 0:
                changed = True

            # 2. Pure literal elimination
            n_pure = self._pure_literal_elimination()
            if n_pure > 0:
                changed = True

            # 3. Subsumption checking
            n_sub = self._subsumption_pass()
            if n_sub > 0:
                changed = True

            # 4. Blocked edit detection
            n_blocked = self._blocked_edit_detection()
            if n_blocked > 0:
                changed = True

            # 5. Variable elimination (bounded resolution)
            if self._enable_var_elim:
                n_elim = self._variable_elimination()
                if n_elim > 0:
                    changed = True

            if not changed:
                break

        logger.debug(
            "Preprocessing done: %d rounds, %d unit props, %d pure lits, "
            "%d blocked, %d var elim, %d subsumed",
            self._stats.n_rounds,
            self._stats.n_unit_props,
            self._stats.n_pure_lits,
            self._stats.n_blocked_edits,
            self._stats.n_var_elim,
            self._stats.n_subsumed,
        )

        return self._stats

    # -- unit propagation at root level ------------------------------------

    def _unit_propagate(self) -> int:
        """Propagate unit clauses at the root level.

        A unit clause ``(l)`` forces literal *l* to be true.  All clauses
        containing *l* are satisfied (removed).  All clauses containing
        ``¬l`` have that literal removed.

        Returns
        -------
        int
            Number of literals fixed.
        """
        fixed = 0
        changed = True

        while changed:
            changed = False
            units = self._db.unit_clauses()

            for clause in units:
                if clause.is_empty:
                    continue
                lit = clause.literals[0]
                if lit.edit in self._fixed:
                    continue

                self._fixed[lit.edit] = lit.positive
                self._stats.n_unit_props += 1
                fixed += 1
                changed = True

                # Remove satisfied clauses
                self._remove_satisfied(lit)

                # Shorten clauses containing ¬lit
                self._remove_literal_from_clauses(lit.negated())

        return fixed

    def _remove_satisfied(self, lit: EditLiteral) -> None:
        """Remove all clauses containing *lit* (they are satisfied)."""
        clauses = self._db.clauses_with_literal(lit)
        for clause in clauses:
            if clause.is_learned:
                self._db._remove_clause(clause.cid)
                self._stats.n_clauses_removed += 1

    def _remove_literal_from_clauses(self, lit: EditLiteral) -> None:
        """Remove *lit* from all clauses (it is falsified)."""
        clauses = self._db.clauses_with_literal(lit)
        for clause in clauses:
            new_lits = tuple(l for l in clause.literals if l != lit)
            if len(new_lits) == len(clause.literals):
                continue
            replacement = Clause(
                literals=new_lits,
                activity=clause.activity,
                lbd=clause.lbd,
                is_learned=clause.is_learned,
                reason=clause.reason,
                cid=clause.cid,
            )
            self._db._by_id[clause.cid] = replacement
            # Update occurrence lists
            occ = self._db._occurrence.get(lit, [])
            try:
                occ.remove(clause.cid)
            except ValueError:
                pass
            # Update in original/learned lists
            if clause.is_learned:
                for i, c in enumerate(self._db._learned):
                    if c.cid == clause.cid:
                        self._db._learned[i] = replacement
                        break
            else:
                for i, c in enumerate(self._db._original):
                    if c.cid == clause.cid:
                        self._db._original[i] = replacement
                        break

    # -- pure literal elimination ------------------------------------------

    def _pure_literal_elimination(self) -> int:
        """Eliminate pure literals.

        A literal *l* is *pure* if it appears in clauses but ``¬l`` does not.
        Pure literals can be fixed without affecting satisfiability.

        Returns
        -------
        int
            Number of pure literals eliminated.
        """
        # Collect all edits and their polarities across active clauses
        pos_edits: set[StructuralEdit] = set()
        neg_edits: set[StructuralEdit] = set()

        for clause in self._db.all_clauses:
            for lit in clause.literals:
                if lit.edit in self._fixed or lit.edit in self._eliminated:
                    continue
                if lit.positive:
                    pos_edits.add(lit.edit)
                else:
                    neg_edits.add(lit.edit)

        # Pure positive: appears only positive
        pure_pos = pos_edits - neg_edits
        # Pure negative: appears only negative
        pure_neg = neg_edits - pos_edits

        eliminated = 0

        for edit in pure_pos:
            lit = EditLiteral(edit, True)
            self._fixed[edit] = True
            self._remove_satisfied(lit)
            eliminated += 1

        for edit in pure_neg:
            lit = EditLiteral(edit, False)
            self._fixed[edit] = False
            self._remove_satisfied(lit)
            eliminated += 1

        self._stats.n_pure_lits += eliminated
        return eliminated

    # -- blocked edit detection --------------------------------------------

    def _blocked_edit_detection(self) -> int:
        """Detect and remove blocked edits.

        A literal *l* is *blocked* in clause *C* if for every clause *D*
        containing ``¬l``, the resolvent of *C* and *D* on the variable
        of *l* is a tautology (contains both *p* and ``¬p`` for some *p*).

        If *l* is blocked in every clause containing it, all those clauses
        can be removed.

        Returns
        -------
        int
            Number of blocked edits removed.
        """
        blocked_count = 0

        # Collect all edits that appear in active clauses
        all_edits: set[StructuralEdit] = set()
        for clause in self._db.all_clauses:
            for lit in clause.literals:
                if lit.edit not in self._fixed and lit.edit not in self._eliminated:
                    all_edits.add(lit.edit)

        for edit in all_edits:
            pos_lit = EditLiteral(edit, True)
            neg_lit = EditLiteral(edit, False)

            pos_clauses = self._db.clauses_with_literal(pos_lit)
            neg_clauses = self._db.clauses_with_literal(neg_lit)

            if not pos_clauses or not neg_clauses:
                continue

            # Check if positive literal is blocked in ALL its clauses
            all_blocked = True
            for p_clause in pos_clauses:
                blocked_in_clause = True
                for n_clause in neg_clauses:
                    if not self._resolvent_is_tautology(p_clause, n_clause, edit):
                        blocked_in_clause = False
                        break
                if not blocked_in_clause:
                    all_blocked = False
                    break

            if all_blocked and pos_clauses:
                for clause in pos_clauses:
                    if clause.is_learned:
                        self._db._remove_clause(clause.cid)
                        self._stats.n_clauses_removed += 1
                self._eliminated.add(edit)
                blocked_count += 1

        self._stats.n_blocked_edits += blocked_count
        return blocked_count

    def _resolvent_is_tautology(
        self,
        clause_a: Clause,
        clause_b: Clause,
        pivot_edit: StructuralEdit,
    ) -> bool:
        """Check if the resolvent of *clause_a* and *clause_b* on *pivot_edit* is a tautology."""
        # Collect literals from both clauses, excluding the pivot
        lits_a = {lit for lit in clause_a.literals if lit.edit != pivot_edit}
        lits_b = {lit for lit in clause_b.literals if lit.edit != pivot_edit}

        # Check for complementary pair
        for lit in lits_a:
            if lit.negated() in lits_b:
                return True
        return False

    # -- variable elimination (bounded resolution) -------------------------

    def _variable_elimination(self) -> int:
        """Eliminate variables via bounded resolution.

        For each variable *x*, if the number of resolvents of all positive
        and negative clauses on *x* is no larger than the total number of
        clauses containing *x*, then *x* can be eliminated by replacing
        those clauses with the (non-tautological) resolvents.

        Returns
        -------
        int
            Number of variables eliminated.
        """
        eliminated = 0

        # Score candidates by occurrence count
        var_counts: dict[StructuralEdit, int] = defaultdict(int)
        for clause in self._db.all_clauses:
            for lit in clause.literals:
                if lit.edit not in self._fixed and lit.edit not in self._eliminated:
                    var_counts[lit.edit] += 1

        # Sort by occurrence count (prefer low-occurrence variables)
        candidates = sorted(var_counts.keys(), key=lambda e: var_counts[e])

        for edit in candidates:
            if edit in self._eliminated or edit in self._fixed:
                continue

            pos_lit = EditLiteral(edit, True)
            neg_lit = EditLiteral(edit, False)
            pos_clauses = [
                c for c in self._db.clauses_with_literal(pos_lit)
                if c.cid in self._db._by_id
            ]
            neg_clauses = [
                c for c in self._db.clauses_with_literal(neg_lit)
                if c.cid in self._db._by_id
            ]

            original_count = len(pos_clauses) + len(neg_clauses)
            if original_count == 0:
                continue

            # Generate all non-tautological resolvents
            resolvents: list[Clause] = []
            too_large = False

            for pc in pos_clauses:
                for nc in neg_clauses:
                    resolvent = self._resolve(pc, nc, edit)
                    if resolvent is None:
                        continue  # tautology
                    if resolvent.size > self._max_res_size:
                        too_large = True
                        break
                    resolvents.append(resolvent)
                if too_large:
                    break

            if too_large:
                continue

            # Check bounded resolution criterion
            if len(resolvents) > original_count:
                continue

            # Perform elimination: remove old clauses, add resolvents
            for clause in pos_clauses + neg_clauses:
                self._db._remove_clause(clause.cid)
                self._stats.n_clauses_removed += 1

            # Rebuild learned/original lists
            removed_ids = {c.cid for c in pos_clauses + neg_clauses}
            self._db._learned = [
                c for c in self._db._learned if c.cid not in removed_ids
            ]
            self._db._original = [
                c for c in self._db._original if c.cid not in removed_ids
            ]

            for resolvent in resolvents:
                self._db.add_original(resolvent)

            self._eliminated.add(edit)
            eliminated += 1

        self._stats.n_var_elim += eliminated
        return eliminated

    def _resolve(
        self,
        clause_a: Clause,
        clause_b: Clause,
        pivot_edit: StructuralEdit,
    ) -> Clause | None:
        """Compute the resolvent of *clause_a* and *clause_b* on *pivot_edit*.

        Returns ``None`` if the resolvent is a tautology.
        """
        lits: set[EditLiteral] = set()
        for lit in clause_a.literals:
            if lit.edit != pivot_edit:
                lits.add(lit)
        for lit in clause_b.literals:
            if lit.edit != pivot_edit:
                if lit.negated() in lits:
                    return None  # tautology
                lits.add(lit)

        return Clause(
            literals=tuple(sorted(lits, key=lambda l: (l.edit.source, l.edit.target))),
            reason=f"resolvent on ({pivot_edit.source},{pivot_edit.target})",
        )

    # -- subsumption pass --------------------------------------------------

    def _subsumption_pass(self) -> int:
        """Full forward + backward subsumption pass.

        Returns
        -------
        int
            Number of clauses removed or strengthened.
        """
        removed = 0
        strengthened = 0

        # Sort clauses by size (smaller clauses can subsume larger ones)
        all_clauses = sorted(
            self._db.frozen_snapshot(),
            key=lambda c: c.size,
        )

        for clause in all_clauses:
            if clause.cid not in self._db._by_id:
                continue
            n_removed = self._db.backward_subsumption(clause)
            removed += n_removed

            # Self-subsumption strengthening
            n_strengthened = self._self_subsumption_strengthen(clause)
            strengthened += n_strengthened

        self._stats.n_subsumed += removed
        self._stats.n_strengthened += strengthened
        return removed + strengthened

    def _self_subsumption_strengthen(self, clause: Clause) -> int:
        """Strengthen clauses by self-subsumption resolution.

        If clause *C* = ``(l ∨ rest)`` and there exists clause *D* = ``(¬l ∨ rest)``,
        then *D* can be strengthened to just ``rest``.

        Returns
        -------
        int
            Number of clauses strengthened.
        """
        if clause.size < 2:
            return 0

        strengthened = 0
        clause_set = set(clause.literals)

        for lit in clause.literals:
            neg = lit.negated()
            candidates = self._db.clauses_with_literal(neg)
            for cand in candidates:
                if cand.cid == clause.cid:
                    continue
                if cand.cid not in self._db._by_id:
                    continue
                # Check: is clause \ {lit} ⊆ cand?
                subset = clause_set - {lit}
                if subset.issubset(set(cand.literals)):
                    # Strengthen cand by removing neg
                    new_lits = tuple(l for l in cand.literals if l != neg)
                    replacement = Clause(
                        literals=new_lits,
                        activity=cand.activity,
                        lbd=cand.lbd,
                        is_learned=cand.is_learned,
                        reason=cand.reason + " (self-subsumption strengthened)",
                        cid=cand.cid,
                    )
                    self._db._by_id[cand.cid] = replacement
                    # Update occurrence
                    occ = self._db._occurrence.get(neg, [])
                    try:
                        occ.remove(cand.cid)
                    except ValueError:
                        pass
                    if cand.is_learned:
                        for j, c in enumerate(self._db._learned):
                            if c.cid == cand.cid:
                                self._db._learned[j] = replacement
                                break
                    else:
                        for j, c in enumerate(self._db._original):
                            if c.cid == cand.cid:
                                self._db._original[j] = replacement
                                break
                    strengthened += 1

        return strengthened

    # -- queries ------------------------------------------------------------

    @property
    def fixed_variables(self) -> dict[StructuralEdit, bool]:
        """Variables fixed during preprocessing."""
        return dict(self._fixed)

    @property
    def eliminated_variables(self) -> set[StructuralEdit]:
        """Variables eliminated during preprocessing."""
        return set(self._eliminated)

    @property
    def stats(self) -> PreprocessStats:
        """Preprocessing statistics."""
        return self._stats

    def is_fixed(self, edit: StructuralEdit) -> bool:
        """Return ``True`` if *edit* was fixed during preprocessing."""
        return edit in self._fixed

    def fixed_value(self, edit: StructuralEdit) -> bool | None:
        """Return the fixed value of *edit*, or ``None`` if not fixed."""
        return self._fixed.get(edit)

    def is_eliminated(self, edit: StructuralEdit) -> bool:
        """Return ``True`` if *edit* was eliminated during preprocessing."""
        return edit in self._eliminated

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Preprocessor(fixed={len(self._fixed)}, "
            f"eliminated={len(self._eliminated)}, "
            f"rounds={self._stats.n_rounds})"
        )
