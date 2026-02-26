"""Specification validation for BioProver.

Validates Bio-STL specifications for syntactic well-formedness, semantic
consistency with a BioModel, satisfiability, redundancy, and coverage.
Generates actionable warnings for common issues.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from bioprover.models.bio_model import BioModel
from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
    Expression,
    Interval,
    Predicate,
    STLAnd,
    STLFormula,
    STLImplies,
    STLNot,
    STLOr,
    Until,
)


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Severity levels for validation messages."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class ValidationMessage:
    """A single validation finding.

    Attributes:
        severity:    ERROR, WARNING, or INFO.
        category:    Short category tag (e.g. ``"syntax"``, ``"semantics"``).
        message:     Human-readable description.
        location:    Optional STL sub-formula string identifying the issue.
    """

    severity: Severity
    category: str
    message: str
    location: str = ""

    def __str__(self) -> str:
        loc = f" at [{self.location}]" if self.location else ""
        return f"[{self.severity.name}] ({self.category}){loc}: {self.message}"


@dataclass
class ValidationResult:
    """Aggregated validation output.

    Attributes:
        messages: All validation messages.
        is_valid: ``True`` if no ERROR-level messages exist.
    """

    messages: List[ValidationMessage] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(m.severity == Severity.ERROR for m in self.messages)

    @property
    def errors(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.severity == Severity.WARNING]

    @property
    def infos(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.severity == Severity.INFO]

    def summary(self) -> str:
        n_err = len(self.errors)
        n_warn = len(self.warnings)
        n_info = len(self.infos)
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"Validation {status}: {n_err} error(s), "
            f"{n_warn} warning(s), {n_info} info(s)"
        )

    def __str__(self) -> str:
        lines = [self.summary()]
        for m in self.messages:
            lines.append(f"  {m}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SpecValidator
# ---------------------------------------------------------------------------

class SpecValidator:
    """Validates Bio-STL specifications against a BioModel.

    Parameters
    ----------
    model:
        The BioModel to validate against.  If ``None``, semantic checks
        that depend on the model are skipped.
    """

    def __init__(self, model: Optional[BioModel] = None) -> None:
        self._model = model

    # -- main entry point ---------------------------------------------------

    def validate(self, formula: STLFormula) -> ValidationResult:
        """Run all validation checks and return a :class:`ValidationResult`."""
        result = ValidationResult()
        self._check_syntactic(formula, result)
        self._check_semantic(formula, result)
        self._check_vacuity(formula, result)
        self._check_satisfiability_heuristic(formula, result)
        self._check_consistency(formula, result)
        self._check_common_issues(formula, result)
        return result

    # -- syntactic validation -----------------------------------------------

    def _check_syntactic(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Check syntactic well-formedness."""
        self._check_intervals(formula, result)
        self._check_predicates(formula, result)

    def _check_intervals(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Validate time intervals in temporal operators."""
        if isinstance(formula, (Always, Eventually)):
            iv = formula.interval
            if iv.lo < 0:
                result.messages.append(ValidationMessage(
                    Severity.ERROR, "syntax",
                    f"Negative interval lower bound: {iv}",
                    str(formula),
                ))
            if iv.hi < iv.lo:
                result.messages.append(ValidationMessage(
                    Severity.ERROR, "syntax",
                    f"Interval upper bound < lower bound: {iv}",
                    str(formula),
                ))
            if iv.lo == iv.hi:
                result.messages.append(ValidationMessage(
                    Severity.WARNING, "syntax",
                    f"Point interval {iv} — temporal operator collapses "
                    "to evaluation at a single time point",
                    str(formula),
                ))
            if iv.hi > 1e6:
                result.messages.append(ValidationMessage(
                    Severity.WARNING, "syntax",
                    f"Very large time bound {iv.hi} — may cause "
                    "computational issues",
                    str(formula),
                ))
        elif isinstance(formula, Until):
            iv = formula.interval
            if iv.lo < 0:
                result.messages.append(ValidationMessage(
                    Severity.ERROR, "syntax",
                    f"Negative interval lower bound in Until: {iv}",
                    str(formula),
                ))
            if iv.hi < iv.lo:
                result.messages.append(ValidationMessage(
                    Severity.ERROR, "syntax",
                    f"Until interval upper < lower: {iv}",
                    str(formula),
                ))

        for child in formula.children():
            self._check_intervals(child, result)

    def _check_predicates(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Check atomic predicates for basic validity."""
        for atom in formula.atoms():
            if math.isnan(atom.threshold):
                result.messages.append(ValidationMessage(
                    Severity.ERROR, "syntax",
                    f"NaN threshold in predicate: {atom}",
                    str(atom),
                ))
            if math.isinf(atom.threshold):
                result.messages.append(ValidationMessage(
                    Severity.ERROR, "syntax",
                    f"Infinite threshold in predicate: {atom}",
                    str(atom),
                ))

    # -- semantic validation ------------------------------------------------

    def _check_semantic(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Check semantic consistency with the attached BioModel."""
        if self._model is None:
            return

        model_species = {s.name for s in self._model.species}
        spec_vars = formula.free_variables()

        # Variables must correspond to model species
        for var in spec_vars:
            if var not in model_species:
                result.messages.append(ValidationMessage(
                    Severity.ERROR, "semantics",
                    f"Variable '{var}' not found in model species "
                    f"{sorted(model_species)}",
                ))

        # Check threshold realism against initial concentrations
        for atom in formula.atoms():
            if atom.expr.variable and atom.expr.variable in model_species:
                sp = self._model.get_species(atom.expr.variable)
                if sp is not None:
                    ic = sp.initial_concentration
                    # Warn if threshold is >100× initial concentration
                    if atom.threshold > 0 and ic > 0:
                        ratio = atom.threshold / ic
                        if ratio > 100:
                            result.messages.append(ValidationMessage(
                                Severity.WARNING, "semantics",
                                f"Threshold {atom.threshold} for '{atom.expr.variable}' "
                                f"is {ratio:.0f}× the initial concentration ({ic})",
                                str(atom),
                            ))

        # Negative thresholds for concentrations
        for atom in formula.atoms():
            if atom.threshold < 0 and atom.expr.variable:
                result.messages.append(ValidationMessage(
                    Severity.WARNING, "semantics",
                    f"Negative threshold {atom.threshold} for "
                    f"'{atom.expr.variable}' — concentrations are "
                    "typically non-negative",
                    str(atom),
                ))

    # -- vacuity checking ---------------------------------------------------

    def _check_vacuity(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Detect specifications that are vacuously true or false."""
        # Check for contradictory conjuncts at the atom level
        atoms = formula.atoms()
        for i, a1 in enumerate(atoms):
            for a2 in atoms[i + 1:]:
                if (a1.expr.variable == a2.expr.variable
                        and a1.expr.variable is not None):
                    contradiction = self._atoms_contradict(a1, a2)
                    if contradiction:
                        result.messages.append(ValidationMessage(
                            Severity.WARNING, "vacuity",
                            f"Potentially contradictory predicates: "
                            f"({a1}) and ({a2})",
                        ))

        # G[a,b](x > T) where T is impossibly high
        self._check_always_impossible(formula, result)

    def _atoms_contradict(self, a1: Predicate, a2: Predicate) -> bool:
        """Heuristic: check if two predicates on the same variable contradict.

        For example, x > 10 and x < 5 are contradictory.
        """
        if a1.expr.variable != a2.expr.variable:
            return False
        # x > T1 and x < T2 where T2 <= T1
        if (a1.op in (ComparisonOp.GT, ComparisonOp.GE)
                and a2.op in (ComparisonOp.LT, ComparisonOp.LE)):
            return a2.threshold <= a1.threshold
        if (a2.op in (ComparisonOp.GT, ComparisonOp.GE)
                and a1.op in (ComparisonOp.LT, ComparisonOp.LE)):
            return a1.threshold <= a2.threshold
        return False

    def _check_always_impossible(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Warn about Always(impossible_predicate) patterns."""
        if isinstance(formula, Always):
            child = formula.child
            if isinstance(child, Predicate):
                # G(x > very_large) is likely vacuously false
                if (child.op in (ComparisonOp.GT, ComparisonOp.GE)
                        and child.threshold > 1e5):
                    result.messages.append(ValidationMessage(
                        Severity.WARNING, "vacuity",
                        f"Always({child}) with very large threshold "
                        f"{child.threshold} — likely vacuously false",
                        str(formula),
                    ))
        for ch in formula.children():
            self._check_always_impossible(ch, result)

    # -- satisfiability heuristic -------------------------------------------

    def _check_satisfiability_heuristic(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Heuristic satisfiability checking.

        Detects obviously unsatisfiable patterns without full SMT solving.
        """
        # Collect all direct conjuncts
        conjuncts = self._collect_conjuncts(formula)
        if len(conjuncts) < 2:
            return

        # Check for directly contradictory conjuncts
        for i, c1 in enumerate(conjuncts):
            for c2 in conjuncts[i + 1:]:
                if self._formulas_contradict(c1, c2):
                    result.messages.append(ValidationMessage(
                        Severity.ERROR, "satisfiability",
                        "Specification contains contradictory conjuncts — "
                        "it is unsatisfiable",
                    ))
                    return

    def _collect_conjuncts(self, formula: STLFormula) -> List[STLFormula]:
        """Flatten nested AND into a list of conjuncts."""
        if isinstance(formula, STLAnd):
            return (self._collect_conjuncts(formula.left)
                    + self._collect_conjuncts(formula.right))
        return [formula]

    def _formulas_contradict(self, f1: STLFormula, f2: STLFormula) -> bool:
        """Heuristic: check if f1 and f2 are direct contradictions.

        Covers: Always(x > T1, I) ∧ Always(x < T2, I) where T2 ≤ T1
        on the same interval.
        """
        if not (isinstance(f1, Always) and isinstance(f2, Always)):
            return False
        if f1.interval != f2.interval:
            return False
        c1, c2 = f1.child, f2.child
        if isinstance(c1, Predicate) and isinstance(c2, Predicate):
            return self._atoms_contradict(c1, c2)
        return False

    # -- consistency checking -----------------------------------------------

    def _check_consistency(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Check for internally inconsistent requirements."""
        # Check for conflicting temporal operators on the same variable
        temporal_ops = formula.temporal_operators()
        for i, t1 in enumerate(temporal_ops):
            for t2 in temporal_ops[i + 1:]:
                if self._temporal_conflict(t1, t2):
                    result.messages.append(ValidationMessage(
                        Severity.WARNING, "consistency",
                        f"Potentially conflicting temporal requirements: "
                        f"({t1}) vs ({t2})",
                    ))

    def _temporal_conflict(
        self, t1: STLFormula, t2: STLFormula
    ) -> bool:
        """Heuristic: check for conflicting temporal operators."""
        # Always(x > T) and Eventually(x < T) with overlapping intervals
        if isinstance(t1, Always) and isinstance(t2, Eventually):
            if (isinstance(t1.child, Predicate) and isinstance(t2.child, Predicate)):
                if self._atoms_contradict(t1.child, t2.child):
                    overlap = t1.interval.intersect(t2.interval)
                    return overlap is not None
        if isinstance(t2, Always) and isinstance(t1, Eventually):
            return self._temporal_conflict(t2, t1)
        return False

    # -- common issue detection ---------------------------------------------

    def _check_common_issues(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Detect common specification authoring mistakes."""
        # No temporal operators
        if formula.is_boolean():
            result.messages.append(ValidationMessage(
                Severity.WARNING, "common_issue",
                "Specification contains no temporal operators — it only "
                "constrains the initial state",
            ))

        # Very deep nesting
        if formula.temporal_depth > 5:
            result.messages.append(ValidationMessage(
                Severity.WARNING, "common_issue",
                f"Temporal nesting depth {formula.temporal_depth} is high — "
                "this may cause slow verification",
            ))

        # Very large AST
        if formula.size > 100:
            result.messages.append(ValidationMessage(
                Severity.INFO, "common_issue",
                f"Specification has {formula.size} AST nodes — "
                "consider simplification",
            ))

        # Zero-length intervals
        self._check_zero_intervals(formula, result)

    def _check_zero_intervals(
        self, formula: STLFormula, result: ValidationResult
    ) -> None:
        """Warn about zero-length temporal intervals."""
        if isinstance(formula, (Always, Eventually)):
            if formula.interval.length == 0:
                result.messages.append(ValidationMessage(
                    Severity.WARNING, "common_issue",
                    f"Zero-length interval {formula.interval} in "
                    f"{type(formula).__name__}",
                    str(formula),
                ))
        elif isinstance(formula, Until):
            if formula.interval.length == 0:
                result.messages.append(ValidationMessage(
                    Severity.WARNING, "common_issue",
                    f"Zero-length interval {formula.interval} in Until",
                    str(formula),
                ))
        for ch in formula.children():
            self._check_zero_intervals(ch, result)

    # -- coverage analysis --------------------------------------------------

    def coverage_analysis(self, formula: STLFormula) -> Dict[str, Any]:
        """Analyse which model behaviours are constrained by the spec.

        Returns a dict with:
          - ``constrained_species``: species with predicates
          - ``unconstrained_species``: model species without predicates
          - ``constrained_time_range``: (min_lo, max_hi) of intervals
          - ``num_constraints``: total number of atomic predicates
        """
        constrained: Set[str] = set()
        for atom in formula.atoms():
            constrained.update(atom.free_variables())

        unconstrained: Set[str] = set()
        if self._model is not None:
            model_sp = {s.name for s in self._model.species}
            unconstrained = model_sp - constrained

        # Collect time ranges
        intervals = self._collect_all_intervals(formula)
        if intervals:
            min_lo = min(iv.lo for iv in intervals)
            max_hi = max(iv.hi for iv in intervals)
            time_range = (min_lo, max_hi)
        else:
            time_range = (0.0, 0.0)

        return {
            "constrained_species": sorted(constrained),
            "unconstrained_species": sorted(unconstrained),
            "constrained_time_range": time_range,
            "num_constraints": len(formula.atoms()),
        }

    def _collect_all_intervals(self, formula: STLFormula) -> List[Interval]:
        """Recursively collect all temporal intervals."""
        intervals: List[Interval] = []
        if isinstance(formula, (Always, Eventually)):
            intervals.append(formula.interval)
        elif isinstance(formula, Until):
            intervals.append(formula.interval)
        for ch in formula.children():
            intervals.extend(self._collect_all_intervals(ch))
        return intervals

    # -- simplification -----------------------------------------------------

    def simplify(self, formula: STLFormula) -> STLFormula:
        """Simplify a specification by removing redundant conjuncts.

        Applies basic simplification rules:
          - Remove duplicate conjuncts
          - Remove conjuncts that are implied by others (heuristic)
          - Flatten nested AND
        """
        # Flatten nested AND
        conjuncts = self._collect_conjuncts(formula)
        if len(conjuncts) <= 1:
            return formula

        # Remove exact duplicates (by string representation)
        seen: Dict[str, STLFormula] = {}
        unique: List[STLFormula] = []
        for c in conjuncts:
            key = str(c)
            if key not in seen:
                seen[key] = c
                unique.append(c)

        if len(unique) < len(conjuncts):
            pass  # duplicates removed

        # Remove dominated predicates (heuristic):
        # If G[0,100](x > 5) and G[0,100](x > 3), drop the weaker one
        filtered = self._remove_dominated(unique)

        if not filtered:
            return formula

        result = filtered[0]
        for f in filtered[1:]:
            result = STLAnd(result, f)
        return result

    def _remove_dominated(
        self, conjuncts: List[STLFormula]
    ) -> List[STLFormula]:
        """Remove dominated Always/predicate conjuncts."""
        # Only handle simple Always(predicate) cases
        always_preds: Dict[int, Tuple[Always, Predicate]] = {}
        for i, c in enumerate(conjuncts):
            if isinstance(c, Always) and isinstance(c.child, Predicate):
                always_preds[i] = (c, c.child)

        dominated: Set[int] = set()
        for i, (a1, p1) in always_preds.items():
            for j, (a2, p2) in always_preds.items():
                if i == j or j in dominated:
                    continue
                # Same variable, same interval, same direction
                if (p1.expr.variable == p2.expr.variable
                        and a1.interval == a2.interval
                        and p1.op == p2.op):
                    # x > 5 dominates x > 3 (stronger constraint)
                    if p1.op in (ComparisonOp.GT, ComparisonOp.GE):
                        if p1.threshold > p2.threshold:
                            dominated.add(j)
                        elif p2.threshold > p1.threshold:
                            dominated.add(i)
                    elif p1.op in (ComparisonOp.LT, ComparisonOp.LE):
                        if p1.threshold < p2.threshold:
                            dominated.add(j)
                        elif p2.threshold < p1.threshold:
                            dominated.add(i)

        return [c for i, c in enumerate(conjuncts) if i not in dominated]
