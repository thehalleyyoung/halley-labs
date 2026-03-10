"""
usability_oracle.repair.models — Data models for the repair synthesis pipeline.

Defines :class:`UIMutation`, :class:`RepairCandidate`, :class:`RepairResult`,
and :class:`RepairConstraint` used to describe, rank, and validate proposed
UI modifications produced by the SMT-backed synthesiser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# Mutation type constants
# ---------------------------------------------------------------------------

class MutationType(str, Enum):
    """Canonical mutation types that the synthesiser can propose."""

    RESIZE = "resize"
    REPOSITION = "reposition"
    REGROUP = "regroup"
    RELABEL = "relabel"
    REMOVE = "remove"
    ADD_SHORTCUT = "add_shortcut"
    SIMPLIFY_MENU = "simplify_menu"
    ADD_LANDMARK = "add_landmark"

    @classmethod
    def all_types(cls) -> frozenset[str]:
        return frozenset(m.value for m in cls)


# ---------------------------------------------------------------------------
# UIMutation
# ---------------------------------------------------------------------------

@dataclass
class UIMutation:
    """A single atomic UI modification proposed by the synthesiser.

    Each mutation targets exactly one accessibility-tree node (or a small set
    for grouping operations) and carries a *parameters* dict that the
    :class:`MutationOperator` uses to apply the change.

    Parameters
    ----------
    mutation_type : str
        One of :pyclass:`MutationType` values.
    target_node_id : str
        Accessibility node that this mutation acts on.
    parameters : dict[str, Any]
        Type-specific parameters.  Examples:
        - resize: ``{"width": 64, "height": 48}``
        - reposition: ``{"x": 120, "y": 300}``
        - relabel: ``{"new_name": "Submit Order"}``
        - add_shortcut: ``{"shortcut_key": "Ctrl+S"}``
    description : str
        Human-readable explanation of what the mutation does.
    priority : float
        Estimated importance (higher ⇒ more impactful).  Default 0.
    """

    mutation_type: str
    target_node_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    priority: float = 0.0

    # -- validation --------------------------------------------------------

    _VALID_TYPES: frozenset[str] = MutationType.all_types()

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if self.mutation_type not in self._VALID_TYPES:
            errors.append(
                f"Unknown mutation_type {self.mutation_type!r}; "
                f"expected one of {sorted(self._VALID_TYPES)}"
            )
        if not self.target_node_id:
            errors.append("target_node_id must be non-empty")
        # Type-specific checks
        if self.mutation_type == MutationType.RESIZE:
            w = self.parameters.get("width")
            h = self.parameters.get("height")
            if w is not None and w <= 0:
                errors.append(f"resize width must be positive, got {w}")
            if h is not None and h <= 0:
                errors.append(f"resize height must be positive, got {h}")
        if self.mutation_type == MutationType.REPOSITION:
            for coord in ("x", "y"):
                v = self.parameters.get(coord)
                if v is not None and v < 0:
                    errors.append(f"reposition {coord} must be non-negative, got {v}")
        if self.mutation_type == MutationType.ADD_SHORTCUT:
            if not self.parameters.get("shortcut_key"):
                errors.append("add_shortcut requires a 'shortcut_key' parameter")
        if self.mutation_type == MutationType.RELABEL:
            if not self.parameters.get("new_name"):
                errors.append("relabel requires a 'new_name' parameter")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_type": self.mutation_type,
            "target_node_id": self.target_node_id,
            "parameters": dict(self.parameters),
            "description": self.description,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UIMutation:
        return cls(
            mutation_type=data["mutation_type"],
            target_node_id=data["target_node_id"],
            parameters=data.get("parameters", {}),
            description=data.get("description", ""),
            priority=data.get("priority", 0.0),
        )

    def __repr__(self) -> str:
        return (
            f"UIMutation(type={self.mutation_type!r}, "
            f"target={self.target_node_id!r})"
        )


# ---------------------------------------------------------------------------
# RepairCandidate
# ---------------------------------------------------------------------------

@dataclass
class RepairCandidate:
    """A proposed UI repair consisting of one or more mutations.

    The synthesiser generates multiple candidates, ranks them by expected
    cost reduction, and returns the Pareto-optimal set.

    Attributes
    ----------
    mutations : list[UIMutation]
        Ordered sequence of mutations to apply.
    expected_cost_reduction : float
        Estimated reduction in total cognitive cost (bits or seconds).
    confidence : float
        Posterior confidence that the repair improves usability ∈ [0, 1].
    bottleneck_addressed : str
        The :class:`BottleneckType` value this repair targets.
    feasible : bool
        Whether the SMT solver confirmed feasibility.
    verification_status : str
        One of ``"verified"``, ``"unverified"``, ``"failed"``.
    description : str
        Human-readable summary of the repair.
    code_suggestion : str | None
        Optional CSS / HTML snippet implementing the repair.
    estimated_effort : float
        Developer effort estimate in hours.
    """

    mutations: list[UIMutation] = field(default_factory=list)
    expected_cost_reduction: float = 0.0
    confidence: float = 0.0
    bottleneck_addressed: str = ""
    feasible: bool = False
    verification_status: str = "unverified"
    description: str = ""
    code_suggestion: Optional[str] = None
    estimated_effort: float = 0.0

    # -- derived properties ------------------------------------------------

    @property
    def n_mutations(self) -> int:
        return len(self.mutations)

    @property
    def mutation_types(self) -> list[str]:
        return [m.mutation_type for m in self.mutations]

    @property
    def is_verified(self) -> bool:
        return self.verification_status == "verified"

    # -- scoring -----------------------------------------------------------

    def score(self, alpha: float = 0.7, beta: float = 0.3) -> float:
        """Weighted score combining cost reduction and confidence.

        Parameters
        ----------
        alpha : float
            Weight on expected cost reduction (normalised).
        beta : float
            Weight on confidence.

        Returns
        -------
        float
            Non-negative composite score.
        """
        if not self.feasible:
            return 0.0
        cost_component = max(0.0, self.expected_cost_reduction)
        conf_component = max(0.0, min(1.0, self.confidence))
        effort_penalty = 1.0 / (1.0 + self.estimated_effort)
        return (alpha * cost_component + beta * conf_component) * effort_penalty

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutations": [m.to_dict() for m in self.mutations],
            "expected_cost_reduction": self.expected_cost_reduction,
            "confidence": self.confidence,
            "bottleneck_addressed": self.bottleneck_addressed,
            "feasible": self.feasible,
            "verification_status": self.verification_status,
            "description": self.description,
            "code_suggestion": self.code_suggestion,
            "estimated_effort": self.estimated_effort,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairCandidate:
        return cls(
            mutations=[UIMutation.from_dict(m) for m in data.get("mutations", [])],
            expected_cost_reduction=data.get("expected_cost_reduction", 0.0),
            confidence=data.get("confidence", 0.0),
            bottleneck_addressed=data.get("bottleneck_addressed", ""),
            feasible=data.get("feasible", False),
            verification_status=data.get("verification_status", "unverified"),
            description=data.get("description", ""),
            code_suggestion=data.get("code_suggestion"),
            estimated_effort=data.get("estimated_effort", 0.0),
        )

    def __repr__(self) -> str:
        return (
            f"RepairCandidate(mutations={self.n_mutations}, "
            f"cost_reduction={self.expected_cost_reduction:.3f}, "
            f"confidence={self.confidence:.2f}, "
            f"feasible={self.feasible})"
        )


# ---------------------------------------------------------------------------
# RepairResult
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """Aggregated result of the repair synthesis process.

    Attributes
    ----------
    candidates : list[RepairCandidate]
        All generated repair candidates, sorted by score.
    best : RepairCandidate | None
        Highest-scoring feasible candidate, or None.
    synthesis_time : float
        Wall-clock time for synthesis (seconds).
    solver_status : str
        SMT solver result: ``"sat"``, ``"unsat"``, ``"timeout"``, ``"unknown"``.
    n_candidates_explored : int
        Total candidates evaluated during search.
    """

    candidates: list[RepairCandidate] = field(default_factory=list)
    best: Optional[RepairCandidate] = None
    synthesis_time: float = 0.0
    solver_status: str = "unknown"
    n_candidates_explored: int = 0

    @property
    def has_repair(self) -> bool:
        return self.best is not None and self.best.feasible

    @property
    def n_feasible(self) -> int:
        return sum(1 for c in self.candidates if c.feasible)

    def top_k(self, k: int = 5) -> list[RepairCandidate]:
        """Return the top-k candidates by score."""
        scored = sorted(self.candidates, key=lambda c: c.score(), reverse=True)
        return scored[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "best": self.best.to_dict() if self.best else None,
            "synthesis_time": self.synthesis_time,
            "solver_status": self.solver_status,
            "n_candidates_explored": self.n_candidates_explored,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairResult:
        candidates = [RepairCandidate.from_dict(c) for c in data.get("candidates", [])]
        best_data = data.get("best")
        best = RepairCandidate.from_dict(best_data) if best_data else None
        return cls(
            candidates=candidates,
            best=best,
            synthesis_time=data.get("synthesis_time", 0.0),
            solver_status=data.get("solver_status", "unknown"),
            n_candidates_explored=data.get("n_candidates_explored", 0),
        )

    def __repr__(self) -> str:
        return (
            f"RepairResult(candidates={len(self.candidates)}, "
            f"feasible={self.n_feasible}, "
            f"solver={self.solver_status!r}, "
            f"time={self.synthesis_time:.2f}s)"
        )


# ---------------------------------------------------------------------------
# RepairConstraint
# ---------------------------------------------------------------------------

class ConstraintDirection(str, Enum):
    """Direction for bound constraints."""
    UPPER = "upper"
    LOWER = "lower"
    EQUAL = "equal"


@dataclass
class RepairConstraint:
    """A single constraint on the repair search space.

    Used to express hard limits that any valid repair must satisfy,
    e.g. "Fitts index of difficulty for button X must be ≤ 4.0 bits".

    Attributes
    ----------
    constraint_type : str
        One of ``"fitts"``, ``"hick"``, ``"memory"``, ``"target_size"``,
        ``"cost"``, ``"structural"``, ``"layout"``.
    target : str
        Node ID or variable name the constraint applies to.
    bound : float
        Numerical bound value.
    direction : str
        One of ``"upper"``, ``"lower"``, ``"equal"``.
    weight : float
        Soft-constraint weight (0 = hard constraint).
    description : str
        Human-readable explanation.
    """

    constraint_type: str
    target: str
    bound: float
    direction: str = "upper"
    weight: float = 0.0
    description: str = ""

    _VALID_TYPES: frozenset[str] = frozenset({
        "fitts", "hick", "memory", "target_size",
        "cost", "structural", "layout", "hierarchy",
    })

    _VALID_DIRECTIONS: frozenset[str] = frozenset({"upper", "lower", "equal"})

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.constraint_type not in self._VALID_TYPES:
            errors.append(
                f"Unknown constraint_type {self.constraint_type!r}; "
                f"expected one of {sorted(self._VALID_TYPES)}"
            )
        if self.direction not in self._VALID_DIRECTIONS:
            errors.append(
                f"Unknown direction {self.direction!r}; "
                f"expected one of {sorted(self._VALID_DIRECTIONS)}"
            )
        if not self.target:
            errors.append("target must be non-empty")
        if math.isnan(self.bound) or math.isinf(self.bound):
            errors.append(f"bound must be finite, got {self.bound}")
        return errors

    @property
    def is_hard(self) -> bool:
        return self.weight == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "target": self.target,
            "bound": self.bound,
            "direction": self.direction,
            "weight": self.weight,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairConstraint:
        return cls(
            constraint_type=data["constraint_type"],
            target=data["target"],
            bound=float(data["bound"]),
            direction=data.get("direction", "upper"),
            weight=data.get("weight", 0.0),
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:
        op = {"upper": "≤", "lower": "≥", "equal": "="}[self.direction]
        return (
            f"RepairConstraint({self.constraint_type}[{self.target}] "
            f"{op} {self.bound:.2f})"
        )
