"""
usability_oracle.smt_repair.types — Data types for SMT-backed repair synthesis.

Provides immutable value types for representing UI repair constraints,
mutation candidates, and solver results.  The repair synthesis pipeline
encodes usability bottlenecks as SMT constraints over UI properties
(position, size, label, role) and searches for minimal mutations that
eliminate the bottleneck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, List, NewType, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# NewType aliases
# ---------------------------------------------------------------------------

VariableId = NewType("VariableId", str)
"""Unique identifier for an SMT variable in the constraint system."""

ConstraintId = NewType("ConstraintId", str)
"""Unique identifier for a single constraint clause."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class VariableSort(Enum):
    """SMT sort (type) for a UI variable."""

    BOOL = "bool"
    """Boolean property (e.g. ``aria-hidden``)."""

    INT = "int"
    """Integer property (e.g. pixel position, tab index)."""

    REAL = "real"
    """Real-valued property (e.g. font size in pt, opacity)."""

    STRING = "string"
    """Enumerated string property (e.g. role, label text)."""

    BITVEC = "bitvec"
    """Bit-vector for colour values (e.g. 24-bit RGB)."""


@unique
class ConstraintKind(Enum):
    """Classification of a repair constraint."""

    ACCESSIBILITY = "accessibility"
    """WCAG / ARIA conformance requirement."""

    LAYOUT = "layout"
    """Spatial layout constraint (overlap, alignment, containment)."""

    COGNITIVE = "cognitive"
    """Cognitive cost constraint (Fitts' law target size, Hick–Hyman
    choice count, working-memory load)."""

    CONSISTENCY = "consistency"
    """Cross-element consistency (same role ⇒ same affordance)."""

    PRESERVATION = "preservation"
    """Change-minimisation: preserve original values unless necessary."""


@unique
class MutationType(Enum):
    """Kind of UI property mutation."""

    PROPERTY_CHANGE = "property_change"
    """Modify a single scalar property."""

    ELEMENT_ADD = "element_add"
    """Insert a new accessibility node."""

    ELEMENT_REMOVE = "element_remove"
    """Remove an accessibility node."""

    REORDER = "reorder"
    """Change the DOM / tree order of siblings."""

    REPARENT = "reparent"
    """Move a node to a different parent."""


@unique
class SolverStatus(Enum):
    """Result status from the SMT solver."""

    SAT = "sat"
    """Satisfiable — a repair was found."""

    UNSAT = "unsat"
    """Unsatisfiable — no repair exists under the given constraints."""

    UNKNOWN = "unknown"
    """Solver timed out or returned indeterminate."""

    TIMEOUT = "timeout"
    """Explicit timeout before a result was produced."""


# ═══════════════════════════════════════════════════════════════════════════
# UIVariable
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class UIVariable:
    """A single SMT variable representing a mutable UI property.

    Maps a UI tree property (e.g. ``button.width``, ``link.label``)
    to an SMT variable with a declared sort and domain bounds.

    Attributes:
        variable_id: Unique SMT variable name.
        node_id: Accessibility-tree node this variable belongs to.
        property_name: Name of the UI property (e.g. ``"width"``).
        sort: SMT sort of the variable.
        current_value: Current value in the baseline UI.
        lower_bound: Optional lower bound (for INT / REAL sorts).
        upper_bound: Optional upper bound.
        allowed_values: For STRING sort, the finite set of legal values.
    """

    variable_id: str
    node_id: str
    property_name: str
    sort: VariableSort
    current_value: Union[bool, int, float, str]
    lower_bound: Optional[Union[int, float]] = None
    upper_bound: Optional[Union[int, float]] = None
    allowed_values: Optional[FrozenSet[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "variable_id": self.variable_id,
            "node_id": self.node_id,
            "property_name": self.property_name,
            "sort": self.sort.value,
            "current_value": self.current_value,
        }
        if self.lower_bound is not None:
            d["lower_bound"] = self.lower_bound
        if self.upper_bound is not None:
            d["upper_bound"] = self.upper_bound
        if self.allowed_values is not None:
            d["allowed_values"] = sorted(self.allowed_values)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> UIVariable:
        av = d.get("allowed_values")
        return cls(
            variable_id=str(d["variable_id"]),
            node_id=str(d["node_id"]),
            property_name=str(d["property_name"]),
            sort=VariableSort(d["sort"]),
            current_value=d["current_value"],
            lower_bound=d.get("lower_bound"),
            upper_bound=d.get("upper_bound"),
            allowed_values=frozenset(av) if av is not None else None,
        )


# ═══════════════════════════════════════════════════════════════════════════
# RepairConstraint
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class RepairConstraint:
    """A single constraint in the repair SMT problem.

    Each constraint encodes a requirement that the repaired UI must
    satisfy — for example "button width ≥ 44px" (WCAG 2.5.5) or
    "Fitts' index of difficulty ≤ 4 bits".

    Attributes:
        constraint_id: Unique identifier.
        kind: Classification of the constraint.
        description: Human-readable description.
        expression: SMT-LIB 2.6 expression string (S-expression).
        variables: Identifiers of UI variables referenced by this
            constraint.
        is_hard: ``True`` for inviolable constraints (accessibility);
            ``False`` for soft/optimisation objectives.
        weight: Priority weight for soft constraints (ignored if hard).
    """

    constraint_id: str
    kind: ConstraintKind
    description: str
    expression: str
    variables: Tuple[str, ...]
    is_hard: bool = True
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "description": self.description,
            "expression": self.expression,
            "variables": list(self.variables),
            "is_hard": self.is_hard,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RepairConstraint:
        return cls(
            constraint_id=str(d["constraint_id"]),
            kind=ConstraintKind(d["kind"]),
            description=str(d["description"]),
            expression=str(d["expression"]),
            variables=tuple(d["variables"]),
            is_hard=bool(d.get("is_hard", True)),
            weight=float(d.get("weight", 1.0)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ConstraintSystem
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ConstraintSystem:
    """Complete SMT constraint system for a repair problem.

    Bundles all variables and constraints into a single, serialisable
    structure that can be handed to a :class:`RepairSolver`.

    Attributes:
        variables: All UI variables in the problem.
        constraints: All repair constraints (hard and soft).
        objective_expression: Optional SMT expression for the
            minimisation objective (e.g. total edit distance).
        timeout_seconds: Solver timeout.
    """

    variables: Tuple[UIVariable, ...]
    constraints: Tuple[RepairConstraint, ...]
    objective_expression: Optional[str] = None
    timeout_seconds: float = 30.0

    @property
    def num_hard(self) -> int:
        """Number of hard (inviolable) constraints."""
        return sum(1 for c in self.constraints if c.is_hard)

    @property
    def num_soft(self) -> int:
        """Number of soft (optimisation) constraints."""
        return sum(1 for c in self.constraints if not c.is_hard)

    @property
    def variable_ids(self) -> FrozenSet[str]:
        """Set of all variable identifiers."""
        return frozenset(v.variable_id for v in self.variables)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variables": [v.to_dict() for v in self.variables],
            "constraints": [c.to_dict() for c in self.constraints],
            "objective_expression": self.objective_expression,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConstraintSystem:
        return cls(
            variables=tuple(UIVariable.from_dict(v) for v in d["variables"]),
            constraints=tuple(
                RepairConstraint.from_dict(c) for c in d["constraints"]
            ),
            objective_expression=d.get("objective_expression"),
            timeout_seconds=float(d.get("timeout_seconds", 30.0)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# MutationCandidate
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MutationCandidate:
    """A single proposed mutation to the UI.

    Represents one atomic change extracted from the SMT model that,
    combined with other mutations, resolves the usability bottleneck.

    Attributes:
        node_id: Accessibility-tree node affected.
        mutation_type: Kind of mutation.
        property_name: Property being changed (``None`` for structural
            mutations like add/remove).
        old_value: Value before mutation.
        new_value: Value after mutation.
        cost_delta: Estimated change in cognitive cost (negative = improvement).
        confidence: Solver confidence in [0, 1], based on slack analysis.
    """

    node_id: str
    mutation_type: MutationType
    property_name: Optional[str]
    old_value: Union[bool, int, float, str, None]
    new_value: Union[bool, int, float, str, None]
    cost_delta: float = 0.0
    confidence: float = 1.0

    @property
    def is_structural(self) -> bool:
        """Whether this mutation adds, removes, or moves nodes."""
        return self.mutation_type in {
            MutationType.ELEMENT_ADD,
            MutationType.ELEMENT_REMOVE,
            MutationType.REPARENT,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "mutation_type": self.mutation_type.value,
            "property_name": self.property_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "cost_delta": self.cost_delta,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MutationCandidate:
        return cls(
            node_id=str(d["node_id"]),
            mutation_type=MutationType(d["mutation_type"]),
            property_name=d.get("property_name"),
            old_value=d.get("old_value"),
            new_value=d.get("new_value"),
            cost_delta=float(d.get("cost_delta", 0.0)),
            confidence=float(d.get("confidence", 1.0)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# RepairResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class RepairResult:
    """Complete result of an SMT repair synthesis.

    Attributes:
        status: Solver outcome (SAT, UNSAT, UNKNOWN, TIMEOUT).
        mutations: Ordered sequence of proposed mutations (empty if UNSAT).
        total_cost_delta: Aggregate change in expected cognitive cost.
        unsat_core: If UNSAT, the minimal subset of hard constraints
            responsible (empty otherwise).
        solver_time_seconds: Wall-clock time spent in the SMT solver.
        constraint_system: The constraint system that was solved.
    """

    status: SolverStatus
    mutations: Tuple[MutationCandidate, ...]
    total_cost_delta: float
    unsat_core: Tuple[str, ...]
    solver_time_seconds: float
    constraint_system: Optional[ConstraintSystem] = None

    @property
    def is_feasible(self) -> bool:
        """Whether a repair was found."""
        return self.status == SolverStatus.SAT

    @property
    def num_mutations(self) -> int:
        """Number of proposed mutations."""
        return len(self.mutations)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status.value,
            "mutations": [m.to_dict() for m in self.mutations],
            "total_cost_delta": self.total_cost_delta,
            "unsat_core": list(self.unsat_core),
            "solver_time_seconds": self.solver_time_seconds,
        }
        if self.constraint_system is not None:
            d["constraint_system"] = self.constraint_system.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RepairResult:
        cs_raw = d.get("constraint_system")
        return cls(
            status=SolverStatus(d["status"]),
            mutations=tuple(
                MutationCandidate.from_dict(m) for m in d["mutations"]
            ),
            total_cost_delta=float(d["total_cost_delta"]),
            unsat_core=tuple(d["unsat_core"]),
            solver_time_seconds=float(d["solver_time_seconds"]),
            constraint_system=(
                ConstraintSystem.from_dict(cs_raw) if cs_raw else None
            ),
        )
