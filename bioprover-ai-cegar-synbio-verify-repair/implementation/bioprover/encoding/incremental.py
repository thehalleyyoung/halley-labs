"""Incremental encoding manager for CEGAR-style iterative solving.

Manages push/pop state, clause databases, assumption literals,
and delta computation across CEGAR iterations.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TextIO, Tuple

from .expression import (
    And,
    Const,
    ExprNode,
    Implies,
    Not,
    Or,
    Var,
)
from .smtlib_serializer import (
    SerializerConfig,
    _SerCtx,
    emit_assert,
    emit_check_sat,
    emit_check_sat_assuming,
    emit_get_model,
    emit_pop,
    emit_push,
    expr_to_smtlib,
)


# ---------------------------------------------------------------------------
# Clause with reference counting
# ---------------------------------------------------------------------------

@dataclass
class _ManagedClause:
    """A clause in the database with reference counting."""
    clause_id: int
    expr: ExprNode
    ref_count: int = 1
    active: bool = True
    assumption_lit: Optional[str] = None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class IncrementalStats:
    """Track incremental encoding statistics."""
    clauses_added: int = 0
    clauses_removed: int = 0
    push_count: int = 0
    pop_count: int = 0
    iterations: int = 0
    assumption_lits_created: int = 0

    def __repr__(self) -> str:
        return (f"IncrementalStats(added={self.clauses_added}, "
                f"removed={self.clauses_removed}, "
                f"pushes={self.push_count}, pops={self.pop_count}, "
                f"iterations={self.iterations})")


# ---------------------------------------------------------------------------
# IncrementalEncoder
# ---------------------------------------------------------------------------

class IncrementalEncoder:
    """Manage encoding state across CEGAR iterations.

    Tracks which constraints are active, manages push/pop scope,
    and provides assumption-literal-based activation/deactivation.
    """

    def __init__(self, config: Optional[SerializerConfig] = None) -> None:
        self._config = config or SerializerConfig()
        self._clauses: Dict[int, _ManagedClause] = {}
        self._next_id: int = 0
        self._scope_stack: List[Set[int]] = []  # clause IDs per push level
        self._assumption_lits: Dict[str, int] = {}  # lit_name -> clause_id
        self._active_assumptions: Set[str] = set()
        self.stats = IncrementalStats()
        self._ctx = _SerCtx()

    # -- clause management --------------------------------------------------

    def add_clause(
        self,
        expr: ExprNode,
        assumption_lit: Optional[str] = None,
    ) -> int:
        """Add a clause to the database.  Returns the clause ID."""
        cid = self._next_id
        self._next_id += 1

        clause = _ManagedClause(
            clause_id=cid,
            expr=expr,
            assumption_lit=assumption_lit,
        )
        self._clauses[cid] = clause
        self.stats.clauses_added += 1

        if assumption_lit:
            self._assumption_lits[assumption_lit] = cid
            self.stats.assumption_lits_created += 1

        # Record in current scope
        if self._scope_stack:
            self._scope_stack[-1].add(cid)

        return cid

    def remove_clause(self, clause_id: int) -> None:
        """Deactivate a clause (mark as inactive)."""
        if clause_id in self._clauses:
            self._clauses[clause_id].active = False
            self.stats.clauses_removed += 1

    def get_clause(self, clause_id: int) -> Optional[ExprNode]:
        """Get a clause expression by ID."""
        clause = self._clauses.get(clause_id)
        if clause and clause.active:
            return clause.expr
        return None

    def incref(self, clause_id: int) -> None:
        """Increment reference count for a clause."""
        if clause_id in self._clauses:
            self._clauses[clause_id].ref_count += 1

    def decref(self, clause_id: int) -> None:
        """Decrement reference count; deactivate if it reaches zero."""
        if clause_id in self._clauses:
            self._clauses[clause_id].ref_count -= 1
            if self._clauses[clause_id].ref_count <= 0:
                self.remove_clause(clause_id)

    # -- push / pop ---------------------------------------------------------

    def push(self) -> None:
        """Push a new scope level."""
        self._scope_stack.append(set())
        self.stats.push_count += 1

    def pop(self) -> List[int]:
        """Pop the most recent scope, deactivating its clauses.

        Returns the IDs of deactivated clauses.
        """
        if not self._scope_stack:
            raise RuntimeError("Cannot pop: no scope to pop")
        scope = self._scope_stack.pop()
        for cid in scope:
            self.remove_clause(cid)
        self.stats.pop_count += 1
        return list(scope)

    @property
    def scope_depth(self) -> int:
        return len(self._scope_stack)

    # -- assumption literal management --------------------------------------

    def activate_assumption(self, lit_name: str) -> None:
        """Activate an assumption literal."""
        self._active_assumptions.add(lit_name)

    def deactivate_assumption(self, lit_name: str) -> None:
        """Deactivate an assumption literal."""
        self._active_assumptions.discard(lit_name)

    def get_active_assumptions(self) -> List[str]:
        """Return currently active assumption literals."""
        return sorted(self._active_assumptions)

    # -- active clauses -----------------------------------------------------

    def active_clauses(self) -> List[Tuple[int, ExprNode]]:
        """Return all currently active clauses."""
        result: List[Tuple[int, ExprNode]] = []
        for cid, clause in self._clauses.items():
            if clause.active:
                result.append((cid, clause.expr))
        return result

    def active_clause_count(self) -> int:
        return sum(1 for c in self._clauses.values() if c.active)

    # -- delta computation --------------------------------------------------

    def compute_delta(
        self,
        new_clauses: List[ExprNode],
    ) -> Tuple[List[ExprNode], List[int]]:
        """Compute the delta between current state and *new_clauses*.

        Returns (clauses_to_add, clause_ids_to_remove).
        """
        # Current active clause expressions (by structural equality)
        current_set: Dict[ExprNode, int] = {}
        for cid, clause in self._clauses.items():
            if clause.active:
                current_set[clause.expr] = cid

        new_set = set(new_clauses)

        to_add = [c for c in new_clauses if c not in current_set]
        to_remove = [cid for expr, cid in current_set.items()
                     if expr not in new_set]

        return to_add, to_remove

    def apply_delta(
        self,
        to_add: List[ExprNode],
        to_remove: List[int],
    ) -> List[int]:
        """Apply a delta: remove old clauses and add new ones.

        Returns IDs of newly added clauses.
        """
        for cid in to_remove:
            self.remove_clause(cid)

        new_ids: List[int] = []
        for expr in to_add:
            cid = self.add_clause(expr)
            new_ids.append(cid)

        return new_ids

    # -- iteration tracking -------------------------------------------------

    def begin_iteration(self) -> None:
        """Mark the start of a new CEGAR iteration."""
        self.stats.iterations += 1

    # -- SMT-LIB emission ---------------------------------------------------

    def emit_all(self, output: TextIO) -> None:
        """Emit all active clauses as SMT-LIB assertions."""
        for cid, clause in self._clauses.items():
            if not clause.active:
                continue
            if clause.assumption_lit:
                guarded = Implies(Var(clause.assumption_lit), clause.expr)
                emit_assert(output, guarded, self._ctx)
            else:
                emit_assert(output, clause.expr, self._ctx)

    def emit_check(
        self,
        output: TextIO,
        use_assumptions: bool = True,
    ) -> None:
        """Emit check-sat (or check-sat-assuming if assumptions active)."""
        if use_assumptions and self._active_assumptions:
            emit_check_sat_assuming(output, self.get_active_assumptions())
        else:
            emit_check_sat(output)

    def emit_incremental_step(
        self,
        output: TextIO,
        new_clauses: List[ExprNode],
    ) -> None:
        """Emit an incremental step: push, add new clauses, check-sat."""
        self.begin_iteration()
        emit_push(output)
        self.push()

        for expr in new_clauses:
            cid = self.add_clause(expr)
            emit_assert(output, expr, self._ctx)

        self.emit_check(output)
        emit_get_model(output)

    def emit_pop_step(self, output: TextIO) -> None:
        """Pop the last incremental step."""
        self.pop()
        emit_pop(output)

    # -- snapshot / restore -------------------------------------------------

    def snapshot(self) -> Dict[int, ExprNode]:
        """Take a snapshot of all active clauses."""
        return {cid: c.expr for cid, c in self._clauses.items() if c.active}

    def restore(self, snapshot: Dict[int, ExprNode]) -> None:
        """Restore state from a snapshot (deactivate everything not in snapshot)."""
        for cid in list(self._clauses.keys()):
            if cid not in snapshot:
                self.remove_clause(cid)
        # Re-activate snapshot clauses
        for cid, expr in snapshot.items():
            if cid in self._clauses:
                self._clauses[cid].active = True
            else:
                # Re-add
                clause = _ManagedClause(clause_id=cid, expr=expr)
                self._clauses[cid] = clause

    def __repr__(self) -> str:
        return (f"IncrementalEncoder(active={self.active_clause_count()}, "
                f"depth={self.scope_depth}, stats={self.stats})")
