"""
Clause database management for CDCL-based robustness radius search.

Provides clause storage, retrieval, subsumption checking, activity scoring,
learned clause minimisation via self-subsumption, binary clause handling,
and clause statistics tracking.  Clauses are sets of *edit literals*:
each literal represents the inclusion or exclusion of a
:class:`~causalcert.types.StructuralEdit`.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from causalcert.types import EditType, StructuralEdit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Literal representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EditLiteral:
    """A positive or negative edit literal.

    Parameters
    ----------
    edit : StructuralEdit
        The underlying edit operation.
    positive : bool
        ``True`` means *apply this edit*; ``False`` means *do not apply*.
    """

    edit: StructuralEdit
    positive: bool = True

    def negated(self) -> EditLiteral:
        """Return the literal with flipped polarity."""
        return EditLiteral(self.edit, not self.positive)

    def __repr__(self) -> str:  # pragma: no cover
        sign = "" if self.positive else "¬"
        return f"{sign}{self.edit.edit_type.value}({self.edit.source},{self.edit.target})"


# ---------------------------------------------------------------------------
# Clause
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Clause:
    """A disjunction of edit literals.

    At least one literal in the clause must be satisfied.

    Attributes
    ----------
    literals : tuple[EditLiteral, ...]
        Immutable sequence of edit literals.
    activity : float
        Activity score for clause deletion heuristic.
    lbd : int
        Literal Block Distance — number of distinct decision levels in the clause
        at the time it was learned (lower is better).
    is_learned : bool
        Whether this clause was derived during search (vs. an original constraint).
    reason : str
        Human-readable reason for this clause.
    cid : int
        Unique clause identifier assigned by the database.
    """

    literals: tuple[EditLiteral, ...]
    activity: float = 0.0
    lbd: int = 0
    is_learned: bool = False
    reason: str = ""
    cid: int = -1

    # derived helpers -------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of literals in the clause."""
        return len(self.literals)

    @property
    def is_unit(self) -> bool:
        """True if the clause contains exactly one literal."""
        return len(self.literals) == 1

    @property
    def is_binary(self) -> bool:
        """True if the clause has exactly two literals."""
        return len(self.literals) == 2

    @property
    def is_empty(self) -> bool:
        """True if the clause is the empty clause (always falsified)."""
        return len(self.literals) == 0

    @property
    def edits(self) -> frozenset[StructuralEdit]:
        """Set of edits mentioned in the clause (ignoring polarity)."""
        return frozenset(lit.edit for lit in self.literals)

    def subsumes(self, other: Clause) -> bool:
        """Return ``True`` if *self* subsumes *other*.

        A clause *C* subsumes *D* when every literal in *C* appears in *D*.
        If *C* subsumes *D*, then *D* is redundant.
        """
        own = set(self.literals)
        return own.issubset(set(other.literals))

    def __hash__(self) -> int:
        return hash(self.literals)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Clause):
            return NotImplemented
        return self.literals == other.literals


# ---------------------------------------------------------------------------
# Clause statistics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ClauseStats:
    """Aggregate clause-database statistics.

    Attributes
    ----------
    n_original : int
        Original (problem) clauses.
    n_learned : int
        Learned clauses currently in the database.
    n_deleted : int
        Total clauses removed by garbage collection.
    n_subsumptions : int
        Clauses removed by subsumption.
    n_minimisations : int
        Learned clauses successfully minimised.
    avg_lbd : float
        Average LBD of learned clauses (0.0 if none).
    avg_size : float
        Average literal count of learned clauses.
    peak_learned : int
        High-water mark of learned clause count.
    """

    n_original: int = 0
    n_learned: int = 0
    n_deleted: int = 0
    n_subsumptions: int = 0
    n_minimisations: int = 0
    avg_lbd: float = 0.0
    avg_size: float = 0.0
    peak_learned: int = 0


# ---------------------------------------------------------------------------
# Clause database
# ---------------------------------------------------------------------------

class ClauseDatabase:
    """Clause database with activity scoring, subsumption, and garbage collection.

    Parameters
    ----------
    max_learned : int
        Soft cap on learned clauses before garbage collection triggers.
    activity_decay : float
        Multiplicative decay applied to clause activities after each conflict.
    lbd_threshold : int
        Clauses with LBD ≤ this are protected from deletion.
    gc_fraction : float
        Fraction of learned clauses to keep after garbage collection.
    """

    def __init__(
        self,
        max_learned: int = 20_000,
        activity_decay: float = 0.999,
        lbd_threshold: int = 6,
        gc_fraction: float = 0.5,
    ) -> None:
        self._max_learned = max_learned
        self._activity_decay = activity_decay
        self._lbd_threshold = lbd_threshold
        self._gc_fraction = gc_fraction

        self._next_id: int = 0
        self._original: list[Clause] = []
        self._learned: list[Clause] = []
        self._by_id: dict[int, Clause] = {}
        self._bump_increment: float = 1.0

        # Occurrence lists: literal -> list of clause ids containing it
        self._occurrence: dict[EditLiteral, list[int]] = defaultdict(list)

        # Binary clause index for fast binary propagation
        self._binary_partner: dict[EditLiteral, list[EditLiteral]] = defaultdict(list)

        self._stats = ClauseStats()

    # -- clause insertion ---------------------------------------------------

    def _assign_id(self, clause: Clause) -> Clause:
        """Give the clause a unique id and register it."""
        clause.cid = self._next_id
        self._next_id += 1
        self._by_id[clause.cid] = clause
        for lit in clause.literals:
            self._occurrence[lit].append(clause.cid)
        if clause.is_binary:
            a, b = clause.literals
            self._binary_partner[a.negated()].append(b)
            self._binary_partner[b.negated()].append(a)
        return clause

    def add_original(self, clause: Clause) -> Clause:
        """Add a problem (original) clause.

        Parameters
        ----------
        clause : Clause
            The constraint clause to add.

        Returns
        -------
        Clause
            The stored clause (with assigned id).
        """
        clause.is_learned = False
        clause = self._assign_id(clause)
        self._original.append(clause)
        self._stats.n_original += 1
        return clause

    def add_learned(self, clause: Clause) -> Clause:
        """Add a learned clause (conflict-derived).

        Triggers garbage collection when the soft cap is exceeded.

        Parameters
        ----------
        clause : Clause
            Learned clause.

        Returns
        -------
        Clause
            The stored clause.
        """
        clause.is_learned = True
        clause.activity = self._bump_increment
        clause = self._assign_id(clause)
        self._learned.append(clause)
        self._stats.n_learned += 1
        if self._stats.n_learned > self._stats.peak_learned:
            self._stats.peak_learned = self._stats.n_learned
        self._update_learned_stats()
        if len(self._learned) > self._max_learned:
            self._garbage_collect()
        return clause

    # -- retrieval ----------------------------------------------------------

    def get(self, cid: int) -> Clause | None:
        """Retrieve a clause by id, or ``None`` if deleted."""
        return self._by_id.get(cid)

    def clauses_with_literal(self, lit: EditLiteral) -> list[Clause]:
        """Return all active clauses containing *lit*."""
        return [
            self._by_id[cid]
            for cid in self._occurrence.get(lit, [])
            if cid in self._by_id
        ]

    def binary_partners(self, lit: EditLiteral) -> list[EditLiteral]:
        """Return partner literals from binary clauses containing ``¬lit``.

        If ``(¬lit ∨ p)`` is in the database, setting *lit* to false forces *p*.
        """
        return list(self._binary_partner.get(lit, []))

    @property
    def original_clauses(self) -> list[Clause]:
        """All original (problem) clauses."""
        return list(self._original)

    @property
    def learned_clauses(self) -> list[Clause]:
        """All currently stored learned clauses."""
        return list(self._learned)

    @property
    def all_clauses(self) -> Iterator[Clause]:
        """Iterate over every clause (original + learned)."""
        yield from self._original
        yield from self._learned

    @property
    def n_clauses(self) -> int:
        """Total number of active clauses."""
        return len(self._original) + len(self._learned)

    # -- unit clauses -------------------------------------------------------

    def unit_clauses(self) -> list[Clause]:
        """Return all unit clauses in the database."""
        return [c for c in self.all_clauses if c.is_unit]

    # -- activity scoring ---------------------------------------------------

    def bump_clause(self, clause: Clause) -> None:
        """Increase the activity of *clause*."""
        clause.activity += self._bump_increment

    def decay_activities(self) -> None:
        """Apply multiplicative decay to the bump increment.

        This is equivalent to decaying all activities but more numerically stable.
        """
        self._bump_increment /= self._activity_decay

    def rescale_activities(self) -> None:
        """Rescale all activities to prevent overflow."""
        if self._bump_increment > 1e100:
            factor = 1.0 / self._bump_increment
            for c in self._learned:
                c.activity *= factor
            self._bump_increment = 1.0

    # -- subsumption --------------------------------------------------------

    def forward_subsumption(self, clause: Clause) -> bool:
        """Check if *clause* is subsumed by any existing clause.

        Parameters
        ----------
        clause : Clause
            Candidate clause.

        Returns
        -------
        bool
            ``True`` if an existing clause subsumes the candidate (redundant).
        """
        if clause.size == 0:
            return False
        # Use occurrence list on the rarest literal
        best_lit = min(clause.literals, key=lambda l: len(self._occurrence.get(l, [])))
        for cid in self._occurrence.get(best_lit, []):
            existing = self._by_id.get(cid)
            if existing is not None and existing.subsumes(clause):
                return True
        return False

    def backward_subsumption(self, clause: Clause) -> int:
        """Remove clauses subsumed by *clause*.

        Parameters
        ----------
        clause : Clause
            Newly added clause.

        Returns
        -------
        int
            Number of clauses removed.
        """
        if clause.size == 0:
            return 0
        removed = 0
        best_lit = min(clause.literals, key=lambda l: len(self._occurrence.get(l, [])))
        to_remove: list[int] = []
        for cid in self._occurrence.get(best_lit, []):
            existing = self._by_id.get(cid)
            if existing is not None and existing.cid != clause.cid:
                if clause.subsumes(existing) and existing.is_learned:
                    to_remove.append(cid)
        for cid in to_remove:
            self._remove_clause(cid)
            removed += 1
        self._stats.n_subsumptions += removed
        return removed

    # -- learned clause minimisation ----------------------------------------

    def minimize_learned(self, clause: Clause) -> Clause:
        """Attempt to shrink *clause* via self-subsumption resolution.

        For each literal *l* in *clause*, if there exists a binary clause
        ``(¬l ∨ m)`` and *m* is also in *clause*, then *l* is redundant.

        Parameters
        ----------
        clause : Clause
            Learned clause to minimise.

        Returns
        -------
        Clause
            Possibly smaller clause (same id).
        """
        lits = set(clause.literals)
        changed = False

        for lit in list(lits):
            if lit not in lits:
                continue
            for partner in self._binary_partner.get(lit.negated(), []):
                if partner in lits and partner != lit:
                    # l is redundant: resolving (¬l ∨ partner) with clause
                    # removes l while keeping partner
                    lits.discard(lit)
                    changed = True
                    break

        if changed:
            new_clause = Clause(
                literals=tuple(sorted(lits, key=lambda l: (l.edit.source, l.edit.target))),
                activity=clause.activity,
                lbd=clause.lbd,
                is_learned=clause.is_learned,
                reason=clause.reason,
                cid=clause.cid,
            )
            self._by_id[clause.cid] = new_clause
            self._stats.n_minimisations += 1
            # Update occurrence lists
            for lit in clause.literals:
                occ = self._occurrence.get(lit, [])
                if clause.cid in occ:
                    occ.remove(clause.cid)
            for lit in new_clause.literals:
                self._occurrence[lit].append(new_clause.cid)
            # Replace in learned list
            for i, c in enumerate(self._learned):
                if c.cid == clause.cid:
                    self._learned[i] = new_clause
                    break
            return new_clause
        return clause

    # -- garbage collection -------------------------------------------------

    def _garbage_collect(self) -> None:
        """Remove low-activity learned clauses to stay within limits."""
        # Protect clauses with low LBD (high quality)
        protected = [c for c in self._learned if c.lbd <= self._lbd_threshold]
        removable = [c for c in self._learned if c.lbd > self._lbd_threshold]

        if not removable:
            return

        # Sort by activity ascending; keep the top fraction
        removable.sort(key=lambda c: c.activity)
        n_keep = max(1, int(len(removable) * self._gc_fraction))
        to_delete = removable[:len(removable) - n_keep]

        for clause in to_delete:
            self._remove_clause(clause.cid)

        # Rebuild learned list
        self._learned = [c for c in self._learned if c.cid in self._by_id]
        self._stats.n_deleted += len(to_delete)
        self._stats.n_learned = len(self._learned)

        logger.debug(
            "Clause GC: deleted %d learned clauses, %d remaining",
            len(to_delete),
            len(self._learned),
        )

    def _remove_clause(self, cid: int) -> None:
        """Remove a clause from all indices."""
        clause = self._by_id.pop(cid, None)
        if clause is None:
            return
        for lit in clause.literals:
            occ = self._occurrence.get(lit, [])
            try:
                occ.remove(cid)
            except ValueError:
                pass
        if clause.is_binary:
            a, b = clause.literals
            partners_a = self._binary_partner.get(a.negated(), [])
            try:
                partners_a.remove(b)
            except ValueError:
                pass
            partners_b = self._binary_partner.get(b.negated(), [])
            try:
                partners_b.remove(a)
            except ValueError:
                pass

    # -- statistics ---------------------------------------------------------

    def _update_learned_stats(self) -> None:
        """Recompute running average statistics for learned clauses."""
        if not self._learned:
            self._stats.avg_lbd = 0.0
            self._stats.avg_size = 0.0
            return
        self._stats.avg_lbd = sum(c.lbd for c in self._learned) / len(self._learned)
        self._stats.avg_size = sum(c.size for c in self._learned) / len(self._learned)

    @property
    def stats(self) -> ClauseStats:
        """Current clause-database statistics."""
        self._update_learned_stats()
        return self._stats

    # -- bulk operations ----------------------------------------------------

    def clear_learned(self) -> None:
        """Remove all learned clauses (used on full restart)."""
        for c in list(self._learned):
            self._remove_clause(c.cid)
        self._learned.clear()
        self._stats.n_learned = 0

    def protect_clause(self, clause: Clause) -> None:
        """Mark a clause as protected (low LBD) to survive GC."""
        clause.lbd = min(clause.lbd, self._lbd_threshold)

    def frozen_snapshot(self) -> list[Clause]:
        """Return a snapshot of all clauses for external consumers."""
        return list(self.all_clauses)

    def __len__(self) -> int:
        return self.n_clauses

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ClauseDatabase(original={len(self._original)}, "
            f"learned={len(self._learned)})"
        )
