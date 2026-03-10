"""
usability_oracle.goms.types — GOMS / KLM-GOMS cognitive architecture types.

Models the Goals, Operators, Methods, Selection-rules hierarchy (Card, Moran
& Newell 1983) and the Keystroke-Level Model (Card, Moran & Newell 1980)
with calibrated operator durations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from usability_oracle.core.types import BoundingBox, Point2D


# ═══════════════════════════════════════════════════════════════════════════
# OperatorType — KLM primitive operator taxonomy
# ═══════════════════════════════════════════════════════════════════════════

@unique
class OperatorType(Enum):
    """KLM operator taxonomy (Card, Moran & Newell 1980).

    Each member carries a default duration from the literature.
    """

    K = "keystroke"
    """Keystroke or button press — default 0.28 s (average typist)."""

    P = "pointing"
    """Point to a target (Fitts' Law governed) — varies by ID."""

    H = "homing"
    """Hand movement between devices — default 0.40 s."""

    D = "drawing"
    """Drawing gesture — duration depends on path length."""

    M = "mental_preparation"
    """Mental preparation operator — default 1.35 s."""

    R = "system_response"
    """System response wait — depends on latency."""

    W = "waiting"
    """Waiting for external event — variable duration."""

    B = "mouse_button"
    """Mouse button press or release — default 0.10 s."""

    @property
    def default_duration_s(self) -> float:
        """Default operator duration in seconds from KLM literature."""
        return _OPERATOR_DEFAULTS[self]

    @property
    def is_motor(self) -> bool:
        return self in (OperatorType.K, OperatorType.P, OperatorType.H,
                        OperatorType.D, OperatorType.B)

    @property
    def is_cognitive(self) -> bool:
        return self == OperatorType.M

    @property
    def is_system(self) -> bool:
        return self in (OperatorType.R, OperatorType.W)

    def __str__(self) -> str:
        return self.value


_OPERATOR_DEFAULTS: Dict[OperatorType, float] = {
    OperatorType.K: 0.28,
    OperatorType.P: 1.10,
    OperatorType.H: 0.40,
    OperatorType.D: 0.90,
    OperatorType.M: 1.35,
    OperatorType.R: 0.00,
    OperatorType.W: 0.00,
    OperatorType.B: 0.10,
}


# ═══════════════════════════════════════════════════════════════════════════
# GomsOperator — instantiated operator with parametric duration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GomsOperator:
    """A single KLM/GOMS operator instance.

    Attributes
    ----------
    op_type : OperatorType
        The KLM operator category.
    duration_s : float
        Predicted duration in seconds (may differ from default via Fitts etc.).
    target_id : str
        Accessibility node id of the interaction target, if applicable.
    target_bounds : Optional[BoundingBox]
        Bounding box of the target element.
    description : str
        Human-readable description of this operator instance.
    parameters : Mapping[str, Any]
        Extra parameters (e.g. Fitts ID, key character, response latency).
    """

    op_type: OperatorType
    duration_s: float
    target_id: str = ""
    target_bounds: Optional[BoundingBox] = None
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_motor(self) -> bool:
        return self.op_type.is_motor

    @property
    def is_cognitive(self) -> bool:
        return self.op_type.is_cognitive

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "op_type": self.op_type.value,
            "duration_s": self.duration_s,
            "target_id": self.target_id,
            "description": self.description,
            "parameters": dict(self.parameters),
        }
        if self.target_bounds is not None:
            d["target_bounds"] = self.target_bounds.to_dict()
        return d


# ═══════════════════════════════════════════════════════════════════════════
# GomsGoal — hierarchical task goal
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GomsGoal:
    """A goal in the GOMS goal hierarchy.

    Attributes
    ----------
    goal_id : str
        Unique identifier for this goal.
    description : str
        Human-readable description.
    parent_id : str
        Id of the parent goal (empty string for top-level goals).
    subgoal_ids : tuple[str, ...]
        Ids of immediate sub-goals.
    """

    goal_id: str
    description: str
    parent_id: str = ""
    subgoal_ids: Tuple[str, ...] = ()

    @property
    def is_leaf(self) -> bool:
        """True if this goal has no sub-goals (unit task)."""
        return len(self.subgoal_ids) == 0

    @property
    def depth(self) -> int:
        """Depth is inferred externally; returns 0 for root placeholder."""
        return 0 if not self.parent_id else -1  # sentinel; real depth from model


# ═══════════════════════════════════════════════════════════════════════════
# GomsMethod — sequence of operators achieving a goal
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GomsMethod:
    """A method that achieves a goal via a sequence of operators.

    Attributes
    ----------
    method_id : str
        Unique identifier.
    goal_id : str
        The goal this method achieves.
    name : str
        Short name (e.g. ``"click_button_method"``).
    operators : tuple[GomsOperator, ...]
        Ordered sequence of KLM operators.
    selection_rule : str
        Free-text predicate describing when this method is chosen.
    """

    method_id: str
    goal_id: str
    name: str
    operators: Tuple[GomsOperator, ...] = ()
    selection_rule: str = ""

    @property
    def total_duration_s(self) -> float:
        """Sum of all operator durations."""
        return sum(op.duration_s for op in self.operators)

    @property
    def operator_count(self) -> int:
        return len(self.operators)

    @property
    def motor_time_s(self) -> float:
        """Total motor operator time."""
        return sum(op.duration_s for op in self.operators if op.is_motor)

    @property
    def cognitive_time_s(self) -> float:
        """Total cognitive (M) operator time."""
        return sum(op.duration_s for op in self.operators if op.is_cognitive)


# ═══════════════════════════════════════════════════════════════════════════
# KLMSequence — flat keystroke-level model sequence
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class KLMSequence:
    """A flat KLM operator sequence with aggregate timing.

    This is the simplified KLM analysis (no goal hierarchy) suitable
    for quick task-time estimation.

    Attributes
    ----------
    task_name : str
        Name of the task being modelled.
    operators : tuple[GomsOperator, ...]
        Ordered operator sequence.
    mental_prep_placed : bool
        Whether M-operator placement heuristics have been applied.
    """

    task_name: str
    operators: Tuple[GomsOperator, ...] = ()
    mental_prep_placed: bool = False

    @property
    def total_time_s(self) -> float:
        return sum(op.duration_s for op in self.operators)

    @property
    def operator_string(self) -> str:
        """Compact string representation, e.g. ``"M P K K M P B"``."""
        return " ".join(op.op_type.name for op in self.operators)

    @property
    def operator_count_by_type(self) -> Mapping[OperatorType, int]:
        counts: Dict[OperatorType, int] = {}
        for op in self.operators:
            counts[op.op_type] = counts.get(op.op_type, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════════════
# GomsTrace — execution trace for analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GomsTrace:
    """Record of a GOMS model execution for a specific task.

    Attributes
    ----------
    trace_id : str
        Unique trace identifier.
    task_name : str
        Task being traced.
    goals : tuple[GomsGoal, ...]
        Goal hierarchy used.
    methods_selected : tuple[GomsMethod, ...]
        Methods chosen by selection rules.
    total_time_s : float
        Predicted total execution time.
    critical_path_time_s : float
        Time along the critical path (accounting for parallelism in CPM-GOMS).
    metadata : Mapping[str, Any]
        Additional trace metadata.
    """

    trace_id: str
    task_name: str
    goals: Tuple[GomsGoal, ...] = ()
    methods_selected: Tuple[GomsMethod, ...] = ()
    total_time_s: float = 0.0
    critical_path_time_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_operator_count(self) -> int:
        return sum(m.operator_count for m in self.methods_selected)

    @property
    def speedup_from_parallelism(self) -> float:
        """Ratio of serial time to critical-path time (>1 ⟹ parallelism helps)."""
        if self.critical_path_time_s <= 0:
            return 1.0
        return self.total_time_s / self.critical_path_time_s


# ═══════════════════════════════════════════════════════════════════════════
# GomsModel — full GOMS model for a task domain
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GomsModel:
    """Complete GOMS model comprising goals, methods, and selection rules.

    Attributes
    ----------
    model_id : str
        Unique model identifier.
    name : str
        Descriptive name.
    goals : tuple[GomsGoal, ...]
        Complete goal hierarchy.
    methods : tuple[GomsMethod, ...]
        All available methods.
    top_level_goal_id : str
        Id of the root goal.
    """

    model_id: str
    name: str
    goals: Tuple[GomsGoal, ...] = ()
    methods: Tuple[GomsMethod, ...] = ()
    top_level_goal_id: str = ""

    @property
    def goal_count(self) -> int:
        return len(self.goals)

    @property
    def method_count(self) -> int:
        return len(self.methods)

    def methods_for_goal(self, goal_id: str) -> Tuple[GomsMethod, ...]:
        """Return all methods that achieve the given goal."""
        return tuple(m for m in self.methods if m.goal_id == goal_id)


__all__ = [
    "GomsGoal",
    "GomsMethod",
    "GomsModel",
    "GomsOperator",
    "GomsTrace",
    "KLMSequence",
    "OperatorType",
]
