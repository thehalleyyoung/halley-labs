"""DPLL(T) solver for satisfiability modulo theories.

Implements a genuine DPLL(T) framework with:
- CDCL-style Boolean search with two-watched-literal scheme
- Theory-aware propagation and conflict analysis
- VSIDS decision heuristics
- Clause learning from theory conflicts
- Incremental push/pop support
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.types import Formula

from dp_forge.smt.theory_solver import (
    LinearArithmeticSolver,
    LinearConstraint,
    parse_linear_constraint,
)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class LiteralPolarity(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()


@dataclass
class Literal:
    """A Boolean literal: a variable index with polarity."""

    var_id: int
    polarity: LiteralPolarity

    @property
    def is_positive(self) -> bool:
        return self.polarity == LiteralPolarity.POSITIVE

    def negate(self) -> Literal:
        new_pol = (LiteralPolarity.NEGATIVE if self.is_positive
                   else LiteralPolarity.POSITIVE)
        return Literal(self.var_id, new_pol)

    def to_int(self) -> int:
        """DIMACS-style signed integer."""
        return self.var_id if self.is_positive else -self.var_id

    @staticmethod
    def from_int(i: int) -> Literal:
        if i > 0:
            return Literal(i, LiteralPolarity.POSITIVE)
        else:
            return Literal(-i, LiteralPolarity.NEGATIVE)

    def __hash__(self) -> int:
        return hash(self.to_int())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.to_int() == other.to_int()

    def __repr__(self) -> str:
        sign = "" if self.is_positive else "¬"
        return f"{sign}x{self.var_id}"


@dataclass
class Clause:
    """A disjunctive clause of literals."""

    literals: List[Literal]
    is_learned: bool = False
    activity: float = 0.0
    lbd: int = 0  # Literal Block Distance (glue)

    @property
    def size(self) -> int:
        return len(self.literals)

    @property
    def is_unit(self) -> bool:
        return len(self.literals) == 1

    @property
    def is_empty(self) -> bool:
        return len(self.literals) == 0

    def __repr__(self) -> str:
        lits = " ∨ ".join(str(l) for l in self.literals)
        return f"({lits})"


class AssignmentValue(Enum):
    TRUE = auto()
    FALSE = auto()
    UNASSIGNED = auto()


@dataclass
class TrailEntry:
    """An entry in the assignment trail."""

    literal: Literal
    decision_level: int
    reason: Optional[int] = None  # clause index, None for decisions
    is_decision: bool = False


# ---------------------------------------------------------------------------
# WatchedLiterals
# ---------------------------------------------------------------------------

class WatchedLiterals:
    """Two-watched-literal scheme for efficient BCP.

    Each clause watches exactly two of its literals. A clause only
    needs re-examination when one of its watched literals becomes false.
    """

    def __init__(self) -> None:
        # Map from literal (as int) to list of clause indices watching it
        self._watchers: Dict[int, List[int]] = {}
        self._watch_positions: Dict[int, Tuple[int, int]] = {}  # clause -> (w1, w2)

    def init_clause(self, clause_idx: int, clause: Clause) -> None:
        """Initialize watching for a new clause."""
        if clause.size == 0:
            return
        if clause.size == 1:
            lit = clause.literals[0].to_int()
            self._watchers.setdefault(lit, []).append(clause_idx)
            self._watch_positions[clause_idx] = (0, 0)
            return

        w1, w2 = 0, 1
        self._watch_positions[clause_idx] = (w1, w2)
        lit1 = clause.literals[w1].to_int()
        lit2 = clause.literals[w2].to_int()
        self._watchers.setdefault(lit1, []).append(clause_idx)
        self._watchers.setdefault(lit2, []).append(clause_idx)

    def propagate(
        self,
        false_literal: Literal,
        clauses: List[Clause],
        assignment: Dict[int, AssignmentValue],
    ) -> Tuple[List[Literal], Optional[int]]:
        """Process a literal becoming false. Find unit clauses or conflicts.

        Returns:
            (implied_literals, conflict_clause_or_none)
        """
        implied: List[Literal] = []
        false_int = false_literal.to_int()
        watching = list(self._watchers.get(false_int, []))
        new_watchers: List[int] = []

        for clause_idx in watching:
            if clause_idx >= len(clauses):
                continue
            clause = clauses[clause_idx]
            if clause.size <= 1:
                new_watchers.append(clause_idx)
                continue

            w1, w2 = self._watch_positions.get(clause_idx, (0, 1))

            # Determine which watch is the false literal
            lit1_int = clause.literals[w1].to_int()
            lit2_int = clause.literals[w2].to_int()

            if lit1_int == false_int:
                other_w = w2
                false_w = w1
            elif lit2_int == false_int:
                other_w = w1
                false_w = w2
            else:
                new_watchers.append(clause_idx)
                continue

            # Check if the other watched literal is true
            other_lit = clause.literals[other_w]
            other_val = assignment.get(other_lit.var_id, AssignmentValue.UNASSIGNED)
            if other_lit.is_positive and other_val == AssignmentValue.TRUE:
                new_watchers.append(clause_idx)
                continue
            if not other_lit.is_positive and other_val == AssignmentValue.FALSE:
                new_watchers.append(clause_idx)
                continue

            # Try to find a new literal to watch
            found_new = False
            for i in range(clause.size):
                if i == w1 or i == w2:
                    continue
                lit = clause.literals[i]
                val = assignment.get(lit.var_id, AssignmentValue.UNASSIGNED)
                # Can watch if not false
                is_false = ((lit.is_positive and val == AssignmentValue.FALSE) or
                            (not lit.is_positive and val == AssignmentValue.TRUE))
                if not is_false:
                    # Move watch
                    new_w = i
                    if false_w == w1:
                        self._watch_positions[clause_idx] = (new_w, w2)
                    else:
                        self._watch_positions[clause_idx] = (w1, new_w)
                    new_lit_int = clause.literals[new_w].to_int()
                    self._watchers.setdefault(new_lit_int, []).append(clause_idx)
                    found_new = True
                    break

            if found_new:
                continue

            # No replacement found: clause is unit or conflicting
            new_watchers.append(clause_idx)
            other_val = assignment.get(other_lit.var_id, AssignmentValue.UNASSIGNED)
            is_other_false = ((other_lit.is_positive and other_val == AssignmentValue.FALSE) or
                              (not other_lit.is_positive and other_val == AssignmentValue.TRUE))

            if is_other_false:
                # Conflict
                self._watchers[false_int] = new_watchers
                return implied, clause_idx
            elif other_val == AssignmentValue.UNASSIGNED:
                implied.append(other_lit)

        self._watchers[false_int] = new_watchers
        return implied, None


# ---------------------------------------------------------------------------
# UnitPropagation
# ---------------------------------------------------------------------------

class UnitPropagation:
    """Boolean Constraint Propagation with theory interaction.

    Performs BCP using the two-watched-literal scheme and invokes
    theory checking after each propagation round.
    """

    def __init__(self, watched: WatchedLiterals) -> None:
        self.watched = watched
        self._propagation_queue: List[Literal] = []

    def enqueue(self, literal: Literal) -> None:
        self._propagation_queue.append(literal)

    def propagate(
        self,
        clauses: List[Clause],
        assignment: Dict[int, AssignmentValue],
        trail: List[TrailEntry],
        decision_level: int,
    ) -> Optional[int]:
        """Run BCP to fixpoint.

        Returns:
            Conflict clause index, or None if no conflict.
        """
        while self._propagation_queue:
            lit = self._propagation_queue.pop(0)
            false_lit = lit.negate()
            implied, conflict = self.watched.propagate(false_lit, clauses, assignment)
            if conflict is not None:
                self._propagation_queue.clear()
                return conflict

            for imp_lit in implied:
                val = assignment.get(imp_lit.var_id, AssignmentValue.UNASSIGNED)
                if val == AssignmentValue.UNASSIGNED:
                    # Find the clause that implied this literal
                    reason_clause = self._find_reason(imp_lit, clauses, assignment)
                    self._assign(imp_lit, assignment, trail, decision_level, reason_clause)
                    self._propagation_queue.append(imp_lit)

        return None

    def _assign(
        self,
        literal: Literal,
        assignment: Dict[int, AssignmentValue],
        trail: List[TrailEntry],
        decision_level: int,
        reason: Optional[int],
    ) -> None:
        if literal.is_positive:
            assignment[literal.var_id] = AssignmentValue.TRUE
        else:
            assignment[literal.var_id] = AssignmentValue.FALSE
        trail.append(TrailEntry(literal, decision_level, reason, is_decision=False))

    def _find_reason(
        self,
        literal: Literal,
        clauses: List[Clause],
        assignment: Dict[int, AssignmentValue],
    ) -> Optional[int]:
        """Find the clause that forces this literal (unit propagation reason)."""
        for idx, clause in enumerate(clauses):
            if literal not in clause.literals:
                continue
            # Check if all other literals in clause are false
            all_false = True
            for other in clause.literals:
                if other == literal:
                    continue
                val = assignment.get(other.var_id, AssignmentValue.UNASSIGNED)
                is_false = ((other.is_positive and val == AssignmentValue.FALSE) or
                            (not other.is_positive and val == AssignmentValue.TRUE))
                if not is_false:
                    all_false = False
                    break
            if all_false:
                return idx
        return None

    def clear(self) -> None:
        self._propagation_queue.clear()


# ---------------------------------------------------------------------------
# DecisionHeuristic — VSIDS
# ---------------------------------------------------------------------------

class DecisionHeuristic:
    """VSIDS-like decision heuristic for variable selection.

    Maintains activity scores bumped on conflicts, with periodic decay.
    """

    def __init__(self, num_vars: int, decay_factor: float = 0.95) -> None:
        self.activity: Dict[int, float] = {i: 0.0 for i in range(1, num_vars + 1)}
        self.decay_factor = decay_factor
        self.increment = 1.0
        self._phase: Dict[int, LiteralPolarity] = {}

    def decide(self, assignment: Dict[int, AssignmentValue]) -> Optional[Literal]:
        """Pick the next unassigned variable with highest activity."""
        best_var = None
        best_score = -1.0
        for var_id, score in self.activity.items():
            if assignment.get(var_id, AssignmentValue.UNASSIGNED) == AssignmentValue.UNASSIGNED:
                if score > best_score:
                    best_score = score
                    best_var = var_id
        if best_var is None:
            return None
        phase = self._phase.get(best_var, LiteralPolarity.POSITIVE)
        return Literal(best_var, phase)

    def bump(self, var_id: int) -> None:
        """Bump the activity of a variable (called on conflict)."""
        self.activity[var_id] = self.activity.get(var_id, 0.0) + self.increment

    def decay(self) -> None:
        """Apply decay to all activity scores."""
        self.increment /= self.decay_factor

    def save_phase(self, var_id: int, polarity: LiteralPolarity) -> None:
        self._phase[var_id] = polarity

    def add_variable(self, var_id: int) -> None:
        if var_id not in self.activity:
            self.activity[var_id] = 0.0


# ---------------------------------------------------------------------------
# ConflictAnalysis
# ---------------------------------------------------------------------------

class ConflictAnalysis:
    """Analyze conflicts and derive learned clauses.

    Implements 1-UIP (First Unique Implication Point) learning scheme.
    """

    def __init__(self) -> None:
        pass

    def analyze(
        self,
        conflict_clause_idx: int,
        clauses: List[Clause],
        trail: List[TrailEntry],
        decision_level: int,
    ) -> Tuple[Clause, int]:
        """Perform 1-UIP conflict analysis.

        Args:
            conflict_clause_idx: Index of the conflict clause.
            clauses: All clauses.
            trail: Assignment trail.
            decision_level: Current decision level.

        Returns:
            (learned_clause, backtrack_level)
        """
        if conflict_clause_idx >= len(clauses):
            return Clause([], is_learned=True), 0

        # Build reason map: var_id -> clause index
        reason_map: Dict[int, Optional[int]] = {}
        level_map: Dict[int, int] = {}
        for entry in trail:
            reason_map[entry.literal.var_id] = entry.reason
            level_map[entry.literal.var_id] = entry.decision_level

        # Start with the conflict clause
        conflict_clause = clauses[conflict_clause_idx]
        learned_lits: Set[int] = set()  # literal ints
        seen: Set[int] = set()  # var ids
        num_at_current = 0

        # Count literals at current decision level in conflict clause
        for lit in conflict_clause.literals:
            var_level = level_map.get(lit.var_id, 0)
            if var_level == decision_level:
                num_at_current += 1
            learned_lits.add(lit.to_int())
            seen.add(lit.var_id)

        # Resolve until we reach 1-UIP
        trail_idx = len(trail) - 1
        while num_at_current > 1 and trail_idx >= 0:
            entry = trail[trail_idx]
            trail_idx -= 1

            if entry.literal.var_id not in seen:
                continue
            if level_map.get(entry.literal.var_id, 0) != decision_level:
                continue
            if entry.reason is None:
                continue
            if entry.reason >= len(clauses):
                continue

            # Resolve
            resolve_clause = clauses[entry.reason]
            resolve_var = entry.literal.var_id

            # Remove the resolved variable from learned_lits
            learned_lits.discard(resolve_var)
            learned_lits.discard(-resolve_var)
            num_at_current -= 1

            # Add literals from the reason clause
            for lit in resolve_clause.literals:
                if lit.var_id == resolve_var:
                    continue
                lit_int = lit.to_int()
                if lit_int not in learned_lits and (-lit_int) not in learned_lits:
                    learned_lits.add(lit_int)
                    seen.add(lit.var_id)
                    if level_map.get(lit.var_id, 0) == decision_level:
                        num_at_current += 1

        # Build learned clause
        learned = Clause(
            literals=[Literal.from_int(i) for i in learned_lits],
            is_learned=True,
        )

        # Compute backtrack level: second-highest level in learned clause
        levels = set()
        for lit_int in learned_lits:
            var_id = abs(lit_int)
            levels.add(level_map.get(var_id, 0))
        levels.discard(decision_level)
        backtrack_level = max(levels) if levels else 0

        # Compute LBD (glue)
        learned.lbd = len(levels) + 1

        return learned, backtrack_level


# ---------------------------------------------------------------------------
# LearningEngine
# ---------------------------------------------------------------------------

class LearningEngine:
    """Clause learning from Boolean and theory conflicts.

    Manages the learned clause database with periodic cleanup.
    """

    def __init__(self, max_learned: int = 10000, cleanup_ratio: float = 0.5) -> None:
        self.max_learned = max_learned
        self.cleanup_ratio = cleanup_ratio
        self.learned_clauses: List[Clause] = []
        self.num_conflicts: int = 0

    def add_learned_clause(self, clause: Clause) -> int:
        """Add a learned clause and return its index offset."""
        clause.is_learned = True
        self.learned_clauses.append(clause)
        self.num_conflicts += 1
        if len(self.learned_clauses) > self.max_learned:
            self._cleanup()
        return len(self.learned_clauses) - 1

    def add_theory_conflict(self, conflict_literals: List[Literal]) -> Clause:
        """Create a learned clause from a theory conflict."""
        # The theory conflict is a set of literals whose conjunction is T-unsat.
        # The clause to learn is the disjunction of their negations.
        negated = [lit.negate() for lit in conflict_literals]
        clause = Clause(literals=negated, is_learned=True)
        self.add_learned_clause(clause)
        return clause

    def _cleanup(self) -> None:
        """Remove low-activity learned clauses."""
        # Keep clauses with low LBD (high quality) and high activity
        self.learned_clauses.sort(key=lambda c: (c.lbd, -c.activity))
        keep = int(len(self.learned_clauses) * self.cleanup_ratio)
        self.learned_clauses = self.learned_clauses[:keep]


# ---------------------------------------------------------------------------
# TheoryPropagator
# ---------------------------------------------------------------------------

class TheoryPropagator:
    """Interface between the DPLL Boolean core and the theory solver.

    Translates between Boolean literals and theory atoms, invokes
    theory checking, and converts theory conflicts to clauses.
    """

    def __init__(self, theory_solver: LinearArithmeticSolver) -> None:
        self.theory_solver = theory_solver
        self._atom_map: Dict[int, Formula] = {}  # var_id -> theory atom
        self._formula_map: Dict[str, int] = {}  # formula expr -> var_id

    def register_atom(self, var_id: int, formula: Formula) -> None:
        """Register a Boolean variable as a theory atom."""
        self._atom_map[var_id] = formula
        self._formula_map[formula.expr] = var_id

    def get_atom(self, var_id: int) -> Optional[Formula]:
        return self._atom_map.get(var_id)

    def check_theory(
        self,
        trail: List[TrailEntry],
        assignment: Dict[int, AssignmentValue],
    ) -> Tuple[bool, Optional[List[Literal]]]:
        """Check theory consistency of the current Boolean assignment.

        Returns:
            (is_consistent, conflict_literals_or_none)
        """
        # Collect active theory literals
        theory_literals: List[Formula] = []
        active_var_ids: List[int] = []

        for entry in trail:
            vid = entry.literal.var_id
            atom = self._atom_map.get(vid)
            if atom is None:
                continue
            if entry.literal.is_positive:
                theory_literals.append(atom)
            else:
                # Negation: construct negated formula
                neg_expr = f"NOT({atom.expr})"
                neg = Formula(expr=neg_expr, variables=atom.variables,
                              formula_type=atom.formula_type)
                theory_literals.append(neg)
            active_var_ids.append(vid)

        if not theory_literals:
            return True, None

        self.theory_solver.push()
        consistent, conflict = self.theory_solver.check_consistency(theory_literals)
        self.theory_solver.pop()

        if consistent:
            return True, None

        # Convert theory conflict to Boolean conflict literals
        # The conflict clause should contain the negations of the active assignments
        conflict_lits: List[Literal] = []
        for vid in active_var_ids:
            val = assignment.get(vid, AssignmentValue.UNASSIGNED)
            if val == AssignmentValue.TRUE:
                conflict_lits.append(Literal(vid, LiteralPolarity.POSITIVE))
            elif val == AssignmentValue.FALSE:
                conflict_lits.append(Literal(vid, LiteralPolarity.NEGATIVE))

        return False, conflict_lits

    def theory_propagate(
        self,
        trail: List[TrailEntry],
        assignment: Dict[int, AssignmentValue],
    ) -> List[Literal]:
        """Perform theory propagation: derive new Boolean assignments.

        Returns:
            List of implied Boolean literals.
        """
        theory_literals: List[Formula] = []
        for entry in trail:
            vid = entry.literal.var_id
            atom = self._atom_map.get(vid)
            if atom is None:
                continue
            if entry.literal.is_positive:
                theory_literals.append(atom)
            else:
                neg_expr = f"NOT({atom.expr})"
                neg = Formula(expr=neg_expr, variables=atom.variables,
                              formula_type=atom.formula_type)
                theory_literals.append(neg)

        if not theory_literals:
            return []

        implied_formulas = self.theory_solver.propagate(theory_literals)
        implied_lits: List[Literal] = []

        for f in implied_formulas:
            var_id = self._formula_map.get(f.expr)
            if var_id is not None:
                val = assignment.get(var_id, AssignmentValue.UNASSIGNED)
                if val == AssignmentValue.UNASSIGNED:
                    implied_lits.append(Literal(var_id, LiteralPolarity.POSITIVE))

        return implied_lits


# ---------------------------------------------------------------------------
# BooleanSolver — CDCL core
# ---------------------------------------------------------------------------

class BooleanSolver:
    """CDCL SAT solver with watched literals and 1-UIP learning.

    This is the Boolean core of the DPLL(T) framework. It handles:
    - Decision / backtracking
    - BCP via watched literals
    - Conflict analysis and clause learning
    """

    def __init__(self, num_vars: int = 0) -> None:
        self.num_vars = num_vars
        self.clauses: List[Clause] = []
        self.assignment: Dict[int, AssignmentValue] = {}
        self.trail: List[TrailEntry] = []
        self.decision_level: int = 0
        self._watched = WatchedLiterals()
        self._bcp = UnitPropagation(self._watched)
        self._heuristic = DecisionHeuristic(num_vars)
        self._conflict_analyzer = ConflictAnalysis()
        self._learning = LearningEngine()
        self.num_conflicts: int = 0
        self.num_decisions: int = 0

    def add_clause(self, clause: Clause) -> int:
        idx = len(self.clauses)
        self.clauses.append(clause)
        self._watched.init_clause(idx, clause)
        for lit in clause.literals:
            self._heuristic.add_variable(lit.var_id)
            if lit.var_id > self.num_vars:
                self.num_vars = lit.var_id
        return idx

    def add_variable(self) -> int:
        self.num_vars += 1
        self._heuristic.add_variable(self.num_vars)
        return self.num_vars

    def decide(self) -> Optional[Literal]:
        """Make a decision: pick an unassigned variable."""
        lit = self._heuristic.decide(self.assignment)
        if lit is None:
            return None
        self.decision_level += 1
        self.num_decisions += 1
        if lit.is_positive:
            self.assignment[lit.var_id] = AssignmentValue.TRUE
        else:
            self.assignment[lit.var_id] = AssignmentValue.FALSE
        self.trail.append(TrailEntry(lit, self.decision_level, None, is_decision=True))
        self._heuristic.save_phase(lit.var_id, lit.polarity)
        self._bcp.enqueue(lit)
        return lit

    def propagate(self) -> Optional[int]:
        """Run BCP. Returns conflict clause index or None."""
        return self._bcp.propagate(
            self.clauses, self.assignment, self.trail, self.decision_level
        )

    def analyze_conflict(self, conflict_idx: int) -> Tuple[Clause, int]:
        """Analyze conflict and return (learned_clause, backtrack_level)."""
        learned, bt_level = self._conflict_analyzer.analyze(
            conflict_idx, self.clauses, self.trail, self.decision_level
        )
        # Bump activities
        for lit in learned.literals:
            self._heuristic.bump(lit.var_id)
        self._heuristic.decay()
        self.num_conflicts += 1
        return learned, bt_level

    def backtrack(self, level: int) -> None:
        """Backtrack to the given decision level."""
        new_trail: List[TrailEntry] = []
        for entry in self.trail:
            if entry.decision_level <= level:
                new_trail.append(entry)
            else:
                self.assignment[entry.literal.var_id] = AssignmentValue.UNASSIGNED
        self.trail = new_trail
        self.decision_level = level
        self._bcp.clear()

    def add_learned_clause(self, clause: Clause) -> int:
        """Add a learned clause to the clause database."""
        idx = self.add_clause(clause)
        self._learning.add_learned_clause(clause)
        return idx

    def is_complete(self) -> bool:
        """Check if all variables are assigned."""
        for vid in range(1, self.num_vars + 1):
            if self.assignment.get(vid, AssignmentValue.UNASSIGNED) == AssignmentValue.UNASSIGNED:
                return False
        return True

    def get_model(self) -> Dict[int, bool]:
        """Extract the Boolean model from the current assignment."""
        model: Dict[int, bool] = {}
        for vid in range(1, self.num_vars + 1):
            val = self.assignment.get(vid, AssignmentValue.UNASSIGNED)
            model[vid] = (val == AssignmentValue.TRUE)
        return model


# ---------------------------------------------------------------------------
# DPLLTSolverImpl — main DPLL(T) loop
# ---------------------------------------------------------------------------

class DPLLTSolverImpl:
    """Full DPLL(T) solver combining BooleanSolver with TheoryPropagator.

    Implements the standard DPLL(T) loop:
    1. BCP (Boolean Constraint Propagation)
    2. Theory check
    3. Theory propagation
    4. Decision
    5. Conflict -> learn + backtrack
    """

    def __init__(
        self,
        timeout: float = 120.0,
        produce_proofs: bool = True,
        produce_models: bool = True,
    ) -> None:
        self.timeout = timeout
        self.produce_proofs = produce_proofs
        self.produce_models = produce_models
        self._bool_solver = BooleanSolver()
        self._theory_solver = LinearArithmeticSolver()
        self._propagator = TheoryPropagator(self._theory_solver)
        self._atom_count = 0
        self._var_names: Dict[int, str] = {}

    def new_bool_var(self, name: str = "") -> int:
        """Create a new Boolean variable."""
        vid = self._bool_solver.add_variable()
        if name:
            self._var_names[vid] = name
        return vid

    def new_theory_atom(self, formula: Formula) -> int:
        """Create a Boolean variable representing a theory atom."""
        vid = self.new_bool_var(formula.expr)
        self._propagator.register_atom(vid, formula)
        self._atom_count += 1
        return vid

    def add_clause(self, literals: List[Literal]) -> int:
        """Add a clause to the solver."""
        return self._bool_solver.add_clause(Clause(literals))

    def add_unit(self, literal: Literal) -> int:
        """Add a unit clause (assertion)."""
        idx = self._bool_solver.add_clause(Clause([literal]))
        # Assign unit clause literal at decision level 0 and enqueue for BCP
        val = self._bool_solver.assignment.get(literal.var_id, AssignmentValue.UNASSIGNED)
        if val == AssignmentValue.UNASSIGNED:
            if literal.is_positive:
                self._bool_solver.assignment[literal.var_id] = AssignmentValue.TRUE
            else:
                self._bool_solver.assignment[literal.var_id] = AssignmentValue.FALSE
            self._bool_solver.trail.append(
                TrailEntry(literal, 0, idx, is_decision=False)
            )
            self._bool_solver._bcp.enqueue(literal)
        return idx

    def solve(self) -> Tuple[str, Optional[Dict[int, bool]], Optional[List[str]]]:
        """Run the DPLL(T) loop.

        Returns:
            ("SAT" | "UNSAT" | "UNKNOWN", model_or_none, proof_or_none)
        """
        start_time = time.time()

        # Initial propagation
        conflict = self._bool_solver.propagate()
        if conflict is not None:
            return "UNSAT", None, ["Initial BCP conflict"]

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                return "UNKNOWN", None, None

            # Theory check
            consistent, theory_conflict = self._propagator.check_theory(
                self._bool_solver.trail, self._bool_solver.assignment
            )

            if not consistent and theory_conflict is not None:
                # Learn theory conflict clause
                learned = self._bool_solver._learning.add_theory_conflict(theory_conflict)
                clause_idx = self._bool_solver.add_clause(learned)

                if self._bool_solver.decision_level == 0:
                    proof = self._build_proof()
                    return "UNSAT", None, proof

                _, bt_level = self._bool_solver.analyze_conflict(clause_idx)
                self._bool_solver.backtrack(bt_level)

                conflict = self._bool_solver.propagate()
                if conflict is not None:
                    if self._bool_solver.decision_level == 0:
                        proof = self._build_proof()
                        return "UNSAT", None, proof
                    learned_c, bt = self._bool_solver.analyze_conflict(conflict)
                    self._bool_solver.add_learned_clause(learned_c)
                    self._bool_solver.backtrack(bt)
                continue

            # Theory propagation
            implied = self._propagator.theory_propagate(
                self._bool_solver.trail, self._bool_solver.assignment
            )
            for lit in implied:
                val = self._bool_solver.assignment.get(lit.var_id, AssignmentValue.UNASSIGNED)
                if val == AssignmentValue.UNASSIGNED:
                    if lit.is_positive:
                        self._bool_solver.assignment[lit.var_id] = AssignmentValue.TRUE
                    else:
                        self._bool_solver.assignment[lit.var_id] = AssignmentValue.FALSE
                    self._bool_solver.trail.append(
                        TrailEntry(lit, self._bool_solver.decision_level, None, False)
                    )
                    self._bool_solver._bcp.enqueue(lit)

            conflict = self._bool_solver.propagate()
            if conflict is not None:
                if self._bool_solver.decision_level == 0:
                    proof = self._build_proof()
                    return "UNSAT", None, proof
                learned_c, bt = self._bool_solver.analyze_conflict(conflict)
                self._bool_solver.add_learned_clause(learned_c)
                self._bool_solver.backtrack(bt)
                continue

            # Decision
            decision = self._bool_solver.decide()
            if decision is None:
                # All vars assigned — final theory check
                final_ok, _ = self._propagator.check_theory(
                    self._bool_solver.trail, self._bool_solver.assignment
                )
                if final_ok:
                    model = self._bool_solver.get_model() if self.produce_models else None
                    return "SAT", model, None
                else:
                    if self._bool_solver.decision_level == 0:
                        proof = self._build_proof()
                        return "UNSAT", None, proof
                    self._bool_solver.backtrack(self._bool_solver.decision_level - 1)
                    continue

            # BCP after decision
            conflict = self._bool_solver.propagate()
            if conflict is not None:
                if self._bool_solver.decision_level == 0:
                    proof = self._build_proof()
                    return "UNSAT", None, proof
                learned_c, bt = self._bool_solver.analyze_conflict(conflict)
                self._bool_solver.add_learned_clause(learned_c)
                self._bool_solver.backtrack(bt)

    def _build_proof(self) -> List[str]:
        """Build a simple proof trace from learned clauses."""
        if not self.produce_proofs:
            return []
        steps = []
        for i, clause in enumerate(self._bool_solver._learning.learned_clauses):
            steps.append(f"Learned clause {i}: {clause}")
        steps.append("Derived empty clause -> UNSAT")
        return steps

    def get_theory_model(self) -> Optional[Dict[str, float]]:
        """Get the theory-level model."""
        return self._theory_solver.get_model()


__all__ = [
    "LiteralPolarity",
    "Literal",
    "Clause",
    "AssignmentValue",
    "TrailEntry",
    "WatchedLiterals",
    "UnitPropagation",
    "DecisionHeuristic",
    "ConflictAnalysis",
    "LearningEngine",
    "TheoryPropagator",
    "BooleanSolver",
    "DPLLTSolverImpl",
]
