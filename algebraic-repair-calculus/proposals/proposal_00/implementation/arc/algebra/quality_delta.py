"""
Quality Delta Lattice (Δ_Q)
============================

Implements the quality delta lattice from the three-sorted delta algebra.
Quality deltas represent changes in data quality state: violations introduced,
improvements made, constraints added or removed, and distribution shifts.

Algebraic properties:
- Lattice: (Δ_Q, ⊔, ⊓, ⊥, ⊤) with join, meet, bottom, and top
- Partial order: δ₁ ≤ δ₂ iff δ₁ ⊔ δ₂ = δ₂
- Severity is monotone with respect to the lattice order
- Bottom (⊥) = no quality change; Top (⊤) = complete quality failure

Operations: QUALITY_VIOLATION, QUALITY_IMPROVEMENT, CONSTRAINT_ADDED,
            CONSTRAINT_REMOVED, DISTRIBUTION_SHIFT
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Local type definitions
# ---------------------------------------------------------------------------

class ViolationType(Enum):
    """Types of quality violations."""
    NULL_IN_NON_NULL = "NULL_IN_NON_NULL"
    UNIQUENESS_VIOLATION = "UNIQUENESS_VIOLATION"
    FOREIGN_KEY_VIOLATION = "FOREIGN_KEY_VIOLATION"
    CHECK_VIOLATION = "CHECK_VIOLATION"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    RANGE_VIOLATION = "RANGE_VIOLATION"
    PATTERN_VIOLATION = "PATTERN_VIOLATION"
    CUSTOM_RULE_VIOLATION = "CUSTOM_RULE_VIOLATION"
    REFERENTIAL_INTEGRITY = "REFERENTIAL_INTEGRITY"
    DOMAIN_VIOLATION = "DOMAIN_VIOLATION"
    STATISTICAL_OUTLIER = "STATISTICAL_OUTLIER"
    COMPLETENESS_VIOLATION = "COMPLETENESS_VIOLATION"
    TIMELINESS_VIOLATION = "TIMELINESS_VIOLATION"
    CONSISTENCY_VIOLATION = "CONSISTENCY_VIOLATION"


class SeverityLevel(Enum):
    """Discrete severity levels."""
    NONE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    FATAL = 5

    def to_float(self) -> float:
        return self.value / 5.0

    @staticmethod
    def from_float(f: float) -> SeverityLevel:
        idx = max(0, min(5, round(f * 5)))
        for level in SeverityLevel:
            if level.value == idx:
                return level
        return SeverityLevel.NONE


class ConstraintType(Enum):
    """Types of quality constraints."""
    NOT_NULL = "NOT_NULL"
    UNIQUE = "UNIQUE"
    PRIMARY_KEY = "PRIMARY_KEY"
    FOREIGN_KEY = "FOREIGN_KEY"
    CHECK = "CHECK"
    EXCLUSION = "EXCLUSION"
    STATISTICAL = "STATISTICAL"
    DISTRIBUTION = "DISTRIBUTION"
    COMPLETENESS = "COMPLETENESS"
    TIMELINESS = "TIMELINESS"
    CUSTOM = "CUSTOM"


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics of a column's distribution."""
    mean: Optional[float] = None
    stddev: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    null_fraction: float = 0.0
    distinct_count: Optional[int] = None
    histogram_buckets: Optional[Tuple[float, ...]] = None
    most_common_values: Optional[Tuple[Tuple[Any, float], ...]] = None

    def distance_to(self, other: DistributionSummary) -> float:
        """Compute a distance metric between two distributions."""
        d = 0.0
        if self.mean is not None and other.mean is not None:
            if self.stddev and self.stddev > 0:
                d += abs(self.mean - other.mean) / self.stddev
            else:
                d += abs(self.mean - other.mean) if self.mean != other.mean else 0.0
        d += abs(self.null_fraction - other.null_fraction) * 2.0
        if self.distinct_count is not None and other.distinct_count is not None:
            max_dc = max(self.distinct_count, other.distinct_count, 1)
            d += abs(self.distinct_count - other.distinct_count) / max_dc
        return min(d / 4.0, 1.0)


@dataclass
class QualityState:
    """
    Represents the current quality state of a dataset.

    Tracks active violations, constraint statuses, and quality scores.
    """
    active_violations: Dict[str, List[QualityViolation]] = field(default_factory=dict)
    constraint_statuses: Dict[str, ConstraintStatus] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 1.0
    column_distributions: Dict[str, DistributionSummary] = field(default_factory=dict)

    def copy(self) -> QualityState:
        return QualityState(
            active_violations={
                k: list(v) for k, v in self.active_violations.items()
            },
            constraint_statuses=dict(self.constraint_statuses),
            quality_scores=dict(self.quality_scores),
            overall_score=self.overall_score,
            column_distributions=dict(self.column_distributions),
        )

    def add_violation(self, violation: QualityViolation) -> None:
        self.active_violations.setdefault(violation.constraint_id, []).append(violation)
        self._recompute_score()

    def remove_violations(self, constraint_id: str) -> List[QualityViolation]:
        removed = self.active_violations.pop(constraint_id, [])
        self._recompute_score()
        return removed

    def set_constraint_status(self, constraint_id: str, status: ConstraintStatus) -> None:
        self.constraint_statuses[constraint_id] = status

    def remove_constraint(self, constraint_id: str) -> None:
        self.constraint_statuses.pop(constraint_id, None)
        self.active_violations.pop(constraint_id, None)
        self._recompute_score()

    def violation_count(self) -> int:
        return sum(len(v) for v in self.active_violations.values())

    def has_violations(self) -> bool:
        return self.violation_count() > 0

    def max_severity(self) -> SeverityLevel:
        max_sev = SeverityLevel.NONE
        for violations in self.active_violations.values():
            for v in violations:
                if v.severity.value > max_sev.value:
                    max_sev = v.severity
        return max_sev

    def _recompute_score(self) -> None:
        if not self.active_violations:
            self.overall_score = 1.0
            return
        total_penalty = 0.0
        for violations in self.active_violations.values():
            for v in violations:
                total_penalty += v.severity.to_float() * 0.1
        self.overall_score = max(0.0, 1.0 - total_penalty)

    def __repr__(self) -> str:
        vc = self.violation_count()
        return f"QualityState(score={self.overall_score:.2f}, violations={vc})"


@dataclass(frozen=True)
class ConstraintStatus:
    """Status of a quality constraint."""
    constraint_id: str
    constraint_type: ConstraintType
    is_active: bool = True
    is_satisfied: bool = True
    violation_count: int = 0
    columns: Tuple[str, ...] = ()
    predicate: Optional[str] = None


# ---------------------------------------------------------------------------
# Quality Operations
# ---------------------------------------------------------------------------

class QualityOperation(ABC):
    """Base class for all quality operations in Δ_Q."""

    @abstractmethod
    def inverse(self) -> QualityOperation:
        """Return the inverse operation."""

    @abstractmethod
    def severity_score(self) -> float:
        """Return a severity score in [0, 1]."""

    @abstractmethod
    def affected_constraints(self) -> Set[str]:
        """Return constraint IDs affected."""

    @abstractmethod
    def affected_columns(self) -> Set[str]:
        """Return column names affected."""

    @abstractmethod
    def apply(self, state: QualityState) -> QualityState:
        """Apply this operation to a quality state."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @abstractmethod
    def _key(self) -> tuple:
        """Return a hashable key."""

    @abstractmethod
    def dominates(self, other: QualityOperation) -> bool:
        """Check if this operation dominates (is worse than) another."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QualityOperation):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))


@dataclass(frozen=True)
class QualityViolation(QualityOperation):
    """A quality violation: a constraint was violated."""
    constraint_id: str
    severity: SeverityLevel
    affected_tuples: int
    violation_type: ViolationType
    columns: Tuple[str, ...] = ()
    message: Optional[str] = None

    def inverse(self) -> QualityOperation:
        return QualityImprovement(
            constraint_id=self.constraint_id,
            old_severity=self.severity,
            new_severity=SeverityLevel.NONE,
            fixed_tuples=self.affected_tuples,
            columns=self.columns,
        )

    def severity_score(self) -> float:
        base = self.severity.to_float()
        scale = min(self.affected_tuples / 1000.0, 1.0) if self.affected_tuples > 0 else 0.0
        return min(base * (0.5 + 0.5 * scale), 1.0)

    def affected_constraints(self) -> Set[str]:
        return {self.constraint_id}

    def affected_columns(self) -> Set[str]:
        return set(self.columns)

    def apply(self, state: QualityState) -> QualityState:
        s = state.copy()
        s.add_violation(self)
        s.set_constraint_status(
            self.constraint_id,
            ConstraintStatus(
                constraint_id=self.constraint_id,
                constraint_type=_violation_type_to_constraint_type(self.violation_type),
                is_active=True,
                is_satisfied=False,
                violation_count=self.affected_tuples,
                columns=self.columns,
            ),
        )
        return s

    def dominates(self, other: QualityOperation) -> bool:
        if not isinstance(other, QualityViolation):
            return False
        if self.constraint_id != other.constraint_id:
            return False
        return (
            self.severity.value >= other.severity.value
            and self.affected_tuples >= other.affected_tuples
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "QUALITY_VIOLATION",
            "constraint_id": self.constraint_id,
            "severity": self.severity.name,
            "affected_tuples": self.affected_tuples,
            "violation_type": self.violation_type.value,
            "columns": list(self.columns),
            "message": self.message,
        }

    def _key(self) -> tuple:
        return (
            self.constraint_id,
            self.severity,
            self.affected_tuples,
            self.violation_type,
            self.columns,
        )

    def __repr__(self) -> str:
        return (
            f"VIOLATION({self.constraint_id}, {self.severity.name}, "
            f"{self.affected_tuples} tuples, {self.violation_type.value})"
        )


@dataclass(frozen=True)
class QualityImprovement(QualityOperation):
    """A quality improvement: violations were fixed."""
    constraint_id: str
    old_severity: SeverityLevel
    new_severity: SeverityLevel
    fixed_tuples: int
    columns: Tuple[str, ...] = ()
    message: Optional[str] = None

    def inverse(self) -> QualityOperation:
        return QualityViolation(
            constraint_id=self.constraint_id,
            severity=self.old_severity,
            affected_tuples=self.fixed_tuples,
            violation_type=ViolationType.CUSTOM_RULE_VIOLATION,
            columns=self.columns,
        )

    def severity_score(self) -> float:
        old_s = self.old_severity.to_float()
        new_s = self.new_severity.to_float()
        improvement = max(0.0, old_s - new_s)
        return -improvement

    def affected_constraints(self) -> Set[str]:
        return {self.constraint_id}

    def affected_columns(self) -> Set[str]:
        return set(self.columns)

    def apply(self, state: QualityState) -> QualityState:
        s = state.copy()
        if self.new_severity == SeverityLevel.NONE:
            s.remove_violations(self.constraint_id)
            s.set_constraint_status(
                self.constraint_id,
                ConstraintStatus(
                    constraint_id=self.constraint_id,
                    constraint_type=ConstraintType.CUSTOM,
                    is_active=True,
                    is_satisfied=True,
                    violation_count=0,
                    columns=self.columns,
                ),
            )
        else:
            existing = s.active_violations.get(self.constraint_id, [])
            remaining = max(0, len(existing) - self.fixed_tuples)
            if remaining == 0:
                s.remove_violations(self.constraint_id)
            else:
                s.active_violations[self.constraint_id] = existing[:remaining]
            s._recompute_score()
        return s

    def dominates(self, other: QualityOperation) -> bool:
        if not isinstance(other, QualityImprovement):
            return False
        if self.constraint_id != other.constraint_id:
            return False
        return (
            self.new_severity.value <= other.new_severity.value
            and self.fixed_tuples >= other.fixed_tuples
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "QUALITY_IMPROVEMENT",
            "constraint_id": self.constraint_id,
            "old_severity": self.old_severity.name,
            "new_severity": self.new_severity.name,
            "fixed_tuples": self.fixed_tuples,
            "columns": list(self.columns),
            "message": self.message,
        }

    def _key(self) -> tuple:
        return (
            self.constraint_id,
            self.old_severity,
            self.new_severity,
            self.fixed_tuples,
            self.columns,
        )

    def __repr__(self) -> str:
        return (
            f"IMPROVEMENT({self.constraint_id}, "
            f"{self.old_severity.name}->{self.new_severity.name}, "
            f"{self.fixed_tuples} fixed)"
        )


@dataclass(frozen=True)
class ConstraintAdded(QualityOperation):
    """A new quality constraint was added."""
    constraint_id: str
    constraint_type: ConstraintType
    predicate: Optional[str] = None
    columns: Tuple[str, ...] = ()

    def inverse(self) -> QualityOperation:
        return ConstraintRemoved(
            constraint_id=self.constraint_id,
            reason="inverse of addition",
        )

    def severity_score(self) -> float:
        return 0.0

    def affected_constraints(self) -> Set[str]:
        return {self.constraint_id}

    def affected_columns(self) -> Set[str]:
        return set(self.columns)

    def apply(self, state: QualityState) -> QualityState:
        s = state.copy()
        s.set_constraint_status(
            self.constraint_id,
            ConstraintStatus(
                constraint_id=self.constraint_id,
                constraint_type=self.constraint_type,
                is_active=True,
                is_satisfied=True,
                violation_count=0,
                columns=self.columns,
                predicate=self.predicate,
            ),
        )
        return s

    def dominates(self, other: QualityOperation) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "CONSTRAINT_ADDED",
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type.value,
            "predicate": self.predicate,
            "columns": list(self.columns),
        }

    def _key(self) -> tuple:
        return (self.constraint_id, self.constraint_type, self.predicate, self.columns)

    def __repr__(self) -> str:
        return (
            f"CONSTRAINT_ADDED({self.constraint_id}, "
            f"{self.constraint_type.value})"
        )


@dataclass(frozen=True)
class ConstraintRemoved(QualityOperation):
    """A quality constraint was removed."""
    constraint_id: str
    reason: str = ""
    _preserved_type: Optional[ConstraintType] = None
    _preserved_predicate: Optional[str] = None
    _preserved_columns: Optional[Tuple[str, ...]] = None

    def inverse(self) -> QualityOperation:
        return ConstraintAdded(
            constraint_id=self.constraint_id,
            constraint_type=self._preserved_type or ConstraintType.CUSTOM,
            predicate=self._preserved_predicate,
            columns=self._preserved_columns or (),
        )

    def severity_score(self) -> float:
        return 0.1

    def affected_constraints(self) -> Set[str]:
        return {self.constraint_id}

    def affected_columns(self) -> Set[str]:
        if self._preserved_columns:
            return set(self._preserved_columns)
        return set()

    def apply(self, state: QualityState) -> QualityState:
        s = state.copy()
        s.remove_constraint(self.constraint_id)
        return s

    def dominates(self, other: QualityOperation) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "CONSTRAINT_REMOVED",
            "constraint_id": self.constraint_id,
            "reason": self.reason,
        }

    def _key(self) -> tuple:
        return (self.constraint_id, self.reason)

    def __repr__(self) -> str:
        return f"CONSTRAINT_REMOVED({self.constraint_id}, reason={self.reason!r})"


@dataclass(frozen=True)
class DistributionShift(QualityOperation):
    """A statistical distribution shift was detected on a column."""
    column: str
    old_dist: DistributionSummary
    new_dist: DistributionSummary
    psi_score: float = 0.0
    ks_statistic: float = 0.0

    def inverse(self) -> QualityOperation:
        return DistributionShift(
            column=self.column,
            old_dist=self.new_dist,
            new_dist=self.old_dist,
            psi_score=self.psi_score,
            ks_statistic=self.ks_statistic,
        )

    def severity_score(self) -> float:
        psi_severity = min(self.psi_score / 0.25, 1.0) if self.psi_score > 0 else 0.0
        ks_severity = min(self.ks_statistic / 0.1, 1.0) if self.ks_statistic > 0 else 0.0
        return max(psi_severity, ks_severity)

    def affected_constraints(self) -> Set[str]:
        return {f"dist_{self.column}"}

    def affected_columns(self) -> Set[str]:
        return {self.column}

    def apply(self, state: QualityState) -> QualityState:
        s = state.copy()
        s.column_distributions[self.column] = self.new_dist
        severity = self.severity_score()
        if severity > 0.5:
            s.quality_scores[f"dist_{self.column}"] = 1.0 - severity
        else:
            s.quality_scores[f"dist_{self.column}"] = 1.0
        s._recompute_score()
        return s

    def dominates(self, other: QualityOperation) -> bool:
        if not isinstance(other, DistributionShift):
            return False
        if self.column != other.column:
            return False
        return self.severity_score() >= other.severity_score()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "DISTRIBUTION_SHIFT",
            "column": self.column,
            "psi_score": self.psi_score,
            "ks_statistic": self.ks_statistic,
        }

    def _key(self) -> tuple:
        return (self.column, self.psi_score, self.ks_statistic)

    def __repr__(self) -> str:
        return (
            f"DIST_SHIFT({self.column}, "
            f"PSI={self.psi_score:.4f}, KS={self.ks_statistic:.4f})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _violation_type_to_constraint_type(vt: ViolationType) -> ConstraintType:
    mapping = {
        ViolationType.NULL_IN_NON_NULL: ConstraintType.NOT_NULL,
        ViolationType.UNIQUENESS_VIOLATION: ConstraintType.UNIQUE,
        ViolationType.FOREIGN_KEY_VIOLATION: ConstraintType.FOREIGN_KEY,
        ViolationType.CHECK_VIOLATION: ConstraintType.CHECK,
        ViolationType.TYPE_MISMATCH: ConstraintType.CHECK,
        ViolationType.RANGE_VIOLATION: ConstraintType.CHECK,
        ViolationType.PATTERN_VIOLATION: ConstraintType.CHECK,
        ViolationType.REFERENTIAL_INTEGRITY: ConstraintType.FOREIGN_KEY,
        ViolationType.DOMAIN_VIOLATION: ConstraintType.CHECK,
        ViolationType.STATISTICAL_OUTLIER: ConstraintType.STATISTICAL,
        ViolationType.COMPLETENESS_VIOLATION: ConstraintType.COMPLETENESS,
        ViolationType.TIMELINESS_VIOLATION: ConstraintType.TIMELINESS,
        ViolationType.CONSISTENCY_VIOLATION: ConstraintType.CHECK,
        ViolationType.CUSTOM_RULE_VIOLATION: ConstraintType.CUSTOM,
    }
    return mapping.get(vt, ConstraintType.CUSTOM)


def _merge_violations(
    ops1: List[QualityOperation], ops2: List[QualityOperation]
) -> List[QualityOperation]:
    """
    Merge two sets of quality operations for lattice join.
    For same constraint, keep the more severe violation.
    """
    by_constraint: Dict[str, List[QualityOperation]] = defaultdict(list)
    for op in ops1 + ops2:
        for cid in op.affected_constraints():
            by_constraint[cid].append(op)

    result: List[QualityOperation] = []
    seen_constraints: Set[str] = set()

    for cid, ops in by_constraint.items():
        if cid in seen_constraints:
            continue

        violations = [o for o in ops if isinstance(o, QualityViolation)]
        improvements = [o for o in ops if isinstance(o, QualityImprovement)]
        added = [o for o in ops if isinstance(o, ConstraintAdded)]
        removed = [o for o in ops if isinstance(o, ConstraintRemoved)]
        shifts = [o for o in ops if isinstance(o, DistributionShift)]

        if violations:
            best = max(violations, key=lambda v: (v.severity.value, v.affected_tuples))
            result.append(best)
        elif improvements:
            best = max(improvements, key=lambda v: v.fixed_tuples)
            result.append(best)
        elif removed:
            result.append(removed[0])
        elif added:
            result.append(added[0])

        if shifts:
            worst = max(shifts, key=lambda s: s.severity_score())
            result.append(worst)

        seen_constraints.add(cid)

    ops1_unique_cids = set()
    for op in ops1:
        ops1_unique_cids |= op.affected_constraints()
    ops2_unique_cids = set()
    for op in ops2:
        ops2_unique_cids |= op.affected_constraints()

    return result


def _intersect_operations(
    ops1: List[QualityOperation], ops2: List[QualityOperation]
) -> List[QualityOperation]:
    """
    Intersect two sets of quality operations for lattice meet.
    For same constraint, keep the less severe option.
    """
    cids1: Dict[str, List[QualityOperation]] = defaultdict(list)
    cids2: Dict[str, List[QualityOperation]] = defaultdict(list)

    for op in ops1:
        for cid in op.affected_constraints():
            cids1[cid].append(op)
    for op in ops2:
        for cid in op.affected_constraints():
            cids2[cid].append(op)

    common = set(cids1.keys()) & set(cids2.keys())
    result: List[QualityOperation] = []

    for cid in common:
        ops_a = cids1[cid]
        ops_b = cids2[cid]
        all_ops = ops_a + ops_b

        violations = [o for o in all_ops if isinstance(o, QualityViolation)]
        improvements = [o for o in all_ops if isinstance(o, QualityImprovement)]
        shifts = [o for o in all_ops if isinstance(o, DistributionShift)]

        if violations:
            least = min(violations, key=lambda v: (v.severity.value, v.affected_tuples))
            result.append(least)
        elif improvements:
            least = min(improvements, key=lambda v: v.fixed_tuples)
            result.append(least)
        elif shifts:
            least = min(shifts, key=lambda s: s.severity_score())
            result.append(least)
        else:
            result.append(all_ops[0])

    return result


# ---------------------------------------------------------------------------
# Quality Delta Lattice
# ---------------------------------------------------------------------------

class QualityDelta:
    """
    Represents an element of the quality delta lattice Δ_Q.

    The lattice operations are join (⊔ = least upper bound) and
    meet (⊓ = greatest lower bound). Bottom (⊥) is no quality change,
    and top (⊤) is complete quality failure.
    """

    __slots__ = ("_operations", "_hash_cache")

    def __init__(self, operations: Optional[List[QualityOperation]] = None) -> None:
        self._operations: List[QualityOperation] = list(operations) if operations else []
        self._hash_cache: Optional[int] = None

    @property
    def operations(self) -> List[QualityOperation]:
        return list(self._operations)

    @staticmethod
    def bottom() -> QualityDelta:
        """Return the bottom element (⊥): no quality change."""
        return QualityDelta([])

    @staticmethod
    def top() -> QualityDelta:
        """Return the top element (⊤): complete quality failure."""
        return QualityDelta([
            QualityViolation(
                constraint_id="__TOP__",
                severity=SeverityLevel.FATAL,
                affected_tuples=2**31,
                violation_type=ViolationType.CUSTOM_RULE_VIOLATION,
                columns=(),
                message="Top element: complete quality failure",
            )
        ])

    @staticmethod
    def from_operation(op: QualityOperation) -> QualityDelta:
        return QualityDelta([op])

    @staticmethod
    def from_operations(ops: Sequence[QualityOperation]) -> QualityDelta:
        return QualityDelta(list(ops))

    @staticmethod
    def violation(
        constraint_id: str,
        severity: SeverityLevel,
        affected_tuples: int,
        violation_type: ViolationType,
        columns: Tuple[str, ...] = (),
    ) -> QualityDelta:
        """Create a delta with a single violation."""
        return QualityDelta([
            QualityViolation(
                constraint_id=constraint_id,
                severity=severity,
                affected_tuples=affected_tuples,
                violation_type=violation_type,
                columns=columns,
            )
        ])

    @staticmethod
    def improvement(
        constraint_id: str,
        old_severity: SeverityLevel,
        new_severity: SeverityLevel,
        fixed_tuples: int,
        columns: Tuple[str, ...] = (),
    ) -> QualityDelta:
        """Create a delta with a single improvement."""
        return QualityDelta([
            QualityImprovement(
                constraint_id=constraint_id,
                old_severity=old_severity,
                new_severity=new_severity,
                fixed_tuples=fixed_tuples,
                columns=columns,
            )
        ])

    def join(self, other: QualityDelta) -> QualityDelta:
        """
        Lattice join (⊔): least upper bound.

        Combines both quality deltas, keeping the more severe violation
        for each constraint. For quality improvements, keeps the larger
        improvement.
        """
        if self.is_top() or other.is_top():
            return QualityDelta.top()
        if self.is_bottom():
            return QualityDelta(list(other._operations))
        if other.is_bottom():
            return QualityDelta(list(self._operations))

        merged = _merge_violations(self._operations, other._operations)
        return QualityDelta(merged)

    def meet(self, other: QualityDelta) -> QualityDelta:
        """
        Lattice meet (⊓): greatest lower bound.

        Intersection of quality deltas, keeping the less severe option
        for each constraint.
        """
        if self.is_bottom() or other.is_bottom():
            return QualityDelta.bottom()
        if self.is_top():
            return QualityDelta(list(other._operations))
        if other.is_top():
            return QualityDelta(list(self._operations))

        intersected = _intersect_operations(self._operations, other._operations)
        return QualityDelta(intersected)

    def is_bottom(self) -> bool:
        """Check if this is the bottom element (no quality change)."""
        return len(self._operations) == 0

    def is_top(self) -> bool:
        """Check if this is the top element (complete quality failure)."""
        for op in self._operations:
            if isinstance(op, QualityViolation):
                if op.constraint_id == "__TOP__" and op.severity == SeverityLevel.FATAL:
                    return True
        return False

    def severity(self) -> float:
        """
        Overall severity score in [0, 1].

        0 = no quality impact, 1 = complete quality failure.
        Computed as the max of individual operation severities,
        with a penalty for multiple violations.
        """
        if not self._operations:
            return 0.0
        if self.is_top():
            return 1.0

        scores = [op.severity_score() for op in self._operations]
        positive_scores = [s for s in scores if s > 0]
        negative_scores = [s for s in scores if s < 0]

        if not positive_scores:
            return max(0.0, sum(scores))

        max_severity = max(positive_scores)
        multi_penalty = min(0.1 * (len(positive_scores) - 1), 0.3)
        improvement_credit = sum(negative_scores) * 0.5 if negative_scores else 0.0

        return max(0.0, min(1.0, max_severity + multi_penalty + improvement_credit))

    def affected_constraints(self) -> Set[str]:
        """Return all constraint IDs affected."""
        result: Set[str] = set()
        for op in self._operations:
            result |= op.affected_constraints()
        return result

    def affected_columns(self) -> Set[str]:
        """Return all column names affected."""
        result: Set[str] = set()
        for op in self._operations:
            result |= op.affected_columns()
        return result

    def apply_to_quality_state(self, state: QualityState) -> QualityState:
        """Apply this delta to a quality state."""
        result = state.copy()
        for op in self._operations:
            result = op.apply(result)
        return result

    def violation_count(self) -> int:
        """Count the number of violation operations."""
        return sum(
            1 for op in self._operations if isinstance(op, QualityViolation)
        )

    def improvement_count(self) -> int:
        """Count the number of improvement operations."""
        return sum(
            1 for op in self._operations if isinstance(op, QualityImprovement)
        )

    def has_violations(self) -> bool:
        return any(isinstance(op, QualityViolation) for op in self._operations)

    def has_improvements(self) -> bool:
        return any(isinstance(op, QualityImprovement) for op in self._operations)

    def has_distribution_shifts(self) -> bool:
        return any(isinstance(op, DistributionShift) for op in self._operations)

    def get_violations(self) -> List[QualityViolation]:
        return [op for op in self._operations if isinstance(op, QualityViolation)]

    def get_improvements(self) -> List[QualityImprovement]:
        return [op for op in self._operations if isinstance(op, QualityImprovement)]

    def get_distribution_shifts(self) -> List[DistributionShift]:
        return [op for op in self._operations if isinstance(op, DistributionShift)]

    def get_constraint_additions(self) -> List[ConstraintAdded]:
        return [op for op in self._operations if isinstance(op, ConstraintAdded)]

    def get_constraint_removals(self) -> List[ConstraintRemoved]:
        return [op for op in self._operations if isinstance(op, ConstraintRemoved)]

    def filter_by_constraint(self, constraint_id: str) -> QualityDelta:
        """Return a sub-delta with operations for the given constraint."""
        filtered = [
            op for op in self._operations
            if constraint_id in op.affected_constraints()
        ]
        return QualityDelta(filtered)

    def filter_by_column(self, column: str) -> QualityDelta:
        """Return a sub-delta with operations affecting the given column."""
        filtered = [
            op for op in self._operations
            if column in op.affected_columns()
        ]
        return QualityDelta(filtered)

    def filter_by_severity(self, min_severity: SeverityLevel) -> QualityDelta:
        """Return a sub-delta with operations at or above the given severity."""
        filtered = [
            op for op in self._operations
            if op.severity_score() >= min_severity.to_float()
        ]
        return QualityDelta(filtered)

    def rename_column(self, old_name: str, new_name: str) -> QualityDelta:
        """Return a new QualityDelta with column references renamed."""
        new_ops: List[QualityOperation] = []
        for op in self._operations:
            if isinstance(op, QualityViolation) and old_name in op.columns:
                new_cols = tuple(new_name if c == old_name else c for c in op.columns)
                new_ops.append(QualityViolation(
                    constraint_id=op.constraint_id,
                    severity=op.severity,
                    affected_tuples=op.affected_tuples,
                    violation_type=op.violation_type,
                    columns=new_cols,
                    message=op.message,
                ))
            elif isinstance(op, QualityImprovement) and old_name in op.columns:
                new_cols = tuple(new_name if c == old_name else c for c in op.columns)
                new_ops.append(QualityImprovement(
                    constraint_id=op.constraint_id,
                    old_severity=op.old_severity,
                    new_severity=op.new_severity,
                    fixed_tuples=op.fixed_tuples,
                    columns=new_cols,
                    message=op.message,
                ))
            elif isinstance(op, ConstraintAdded) and old_name in op.columns:
                new_cols = tuple(new_name if c == old_name else c for c in op.columns)
                new_ops.append(ConstraintAdded(
                    constraint_id=op.constraint_id,
                    constraint_type=op.constraint_type,
                    predicate=op.predicate,
                    columns=new_cols,
                ))
            elif isinstance(op, DistributionShift) and op.column == old_name:
                new_ops.append(DistributionShift(
                    column=new_name,
                    old_dist=op.old_dist,
                    new_dist=op.new_dist,
                    psi_score=op.psi_score,
                    ks_statistic=op.ks_statistic,
                ))
            else:
                new_ops.append(op)
        return QualityDelta(new_ops)

    def drop_column(self, column: str) -> QualityDelta:
        """Return a new QualityDelta with operations on the given column removed."""
        filtered = [
            op for op in self._operations
            if column not in op.affected_columns()
        ]
        return QualityDelta(filtered)

    def operation_count(self) -> int:
        return len(self._operations)

    def to_dict(self) -> Dict[str, Any]:
        return {"operations": [op.to_dict() for op in self._operations]}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> QualityDelta:
        op_map: Dict[str, Callable] = {
            "QUALITY_VIOLATION": lambda d: QualityViolation(
                constraint_id=d["constraint_id"],
                severity=SeverityLevel[d["severity"]],
                affected_tuples=d["affected_tuples"],
                violation_type=ViolationType(d["violation_type"]),
                columns=tuple(d.get("columns", [])),
                message=d.get("message"),
            ),
            "QUALITY_IMPROVEMENT": lambda d: QualityImprovement(
                constraint_id=d["constraint_id"],
                old_severity=SeverityLevel[d["old_severity"]],
                new_severity=SeverityLevel[d["new_severity"]],
                fixed_tuples=d["fixed_tuples"],
                columns=tuple(d.get("columns", [])),
                message=d.get("message"),
            ),
            "CONSTRAINT_ADDED": lambda d: ConstraintAdded(
                constraint_id=d["constraint_id"],
                constraint_type=ConstraintType(d["constraint_type"]),
                predicate=d.get("predicate"),
                columns=tuple(d.get("columns", [])),
            ),
            "CONSTRAINT_REMOVED": lambda d: ConstraintRemoved(
                constraint_id=d["constraint_id"],
                reason=d.get("reason", ""),
            ),
            "DISTRIBUTION_SHIFT": lambda d: DistributionShift(
                column=d["column"],
                old_dist=DistributionSummary(),
                new_dist=DistributionSummary(),
                psi_score=d.get("psi_score", 0.0),
                ks_statistic=d.get("ks_statistic", 0.0),
            ),
        }
        ops: List[QualityOperation] = []
        for od in data.get("operations", []):
            factory = op_map.get(od["op"])
            if factory:
                ops.append(factory(od))
        return QualityDelta(ops)

    def __le__(self, other: QualityDelta) -> bool:
        """Partial order: self ≤ other iff self ⊔ other = other."""
        joined = self.join(other)
        return joined == other

    def __lt__(self, other: QualityDelta) -> bool:
        return self <= other and self != other

    def __ge__(self, other: QualityDelta) -> bool:
        return other <= self

    def __gt__(self, other: QualityDelta) -> bool:
        return other < self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QualityDelta):
            return NotImplemented
        if self.is_bottom() and other.is_bottom():
            return True
        if self.is_top() and other.is_top():
            return True
        s1 = set(self._operations)
        s2 = set(other._operations)
        return s1 == s2

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(frozenset(self._operations))
        return self._hash_cache

    def __repr__(self) -> str:
        if self.is_bottom():
            return "QualityDelta(⊥)"
        if self.is_top():
            return "QualityDelta(⊤)"
        ops_str = ", ".join(repr(op) for op in self._operations)
        return f"QualityDelta([{ops_str}])"

    def __len__(self) -> int:
        return len(self._operations)

    def __bool__(self) -> bool:
        return not self.is_bottom()

    def __iter__(self):
        return iter(self._operations)
