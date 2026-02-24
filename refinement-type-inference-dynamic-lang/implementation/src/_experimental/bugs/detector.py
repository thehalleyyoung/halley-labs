"""
Bug Detection Engine for Refinement-Type-Based Analysis.

Detects four classes of bugs from the theory:
  - Array OOB:      Access a[i] where ¬(0 ≤ i < len(a)) is satisfiable
  - Null deref:     Access x.attr where is_none(x) is satisfiable
  - Div-by-zero:    Division a / b where b = 0 is satisfiable
  - Type confusion:  Call x.method() where Tag(x) excludes types with method
"""

from __future__ import annotations

import hashlib
import html as html_mod
import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Core enumerations
# ---------------------------------------------------------------------------

class BugClass(Enum):
    """The four bug classes from the refinement-type theory."""
    ArrayOutOfBounds = "array_out_of_bounds"
    NullDereference = "null_dereference"
    DivisionByZero = "division_by_zero"
    TypeConfusion = "type_confusion"


class Confidence(Enum):
    """Discrete confidence levels."""
    High = "high"        # > 0.9
    Medium = "medium"    # 0.7 – 0.9
    Low = "low"          # 0.5 – 0.7
    VeryLow = "very_low" # < 0.5

    @classmethod
    def from_score(cls, score: float) -> "Confidence":
        if score > 0.9:
            return cls.High
        if score > 0.7:
            return cls.Medium
        if score > 0.5:
            return cls.Low
        return cls.VeryLow


class Severity(Enum):
    """Severity of the reported bug."""
    Error = "error"
    Warning = "warning"
    Info = "info"


# ---------------------------------------------------------------------------
# Lightweight protocol types for IR / analysis objects
# ---------------------------------------------------------------------------

class SourceLocation(Protocol):
    file: str
    line: int
    column: int
    end_line: Optional[int]
    end_column: Optional[int]


@dataclass(frozen=True)
class Loc:
    """Concrete source-location value object."""
    file: str = ""
    line: int = 0
    column: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class Interval:
    """Integer interval [lo, hi] (inclusive).  None means unbounded."""
    lo: Optional[int] = None
    hi: Optional[int] = None

    def contains(self, v: int) -> bool:
        if self.lo is not None and v < self.lo:
            return False
        if self.hi is not None and v > self.hi:
            return False
        return True

    def may_be_negative(self) -> bool:
        return self.lo is None or self.lo < 0

    def may_be_zero(self) -> bool:
        return self.contains(0)

    def guaranteed_nonnegative(self) -> bool:
        return self.lo is not None and self.lo >= 0

    def guaranteed_less_than(self, other: "Interval") -> bool:
        if self.hi is None or other.lo is None:
            return False
        return self.hi < other.lo

    @property
    def is_bounded(self) -> bool:
        return self.lo is not None and self.hi is not None

    def __repr__(self) -> str:
        lo_s = str(self.lo) if self.lo is not None else "-∞"
        hi_s = str(self.hi) if self.hi is not None else "+∞"
        return f"[{lo_s}, {hi_s}]"


class NullityValue(Enum):
    """Abstract nullity domain."""
    NotNull = "not_null"
    Null = "null"
    MaybeNull = "maybe_null"
    Unknown = "unknown"


@dataclass
class TypeTag:
    """A type tag from the refinement-type lattice."""
    name: str
    methods: FrozenSet[str] = field(default_factory=frozenset)
    attributes: FrozenSet[str] = field(default_factory=frozenset)

    def has_method(self, method: str) -> bool:
        return method in self.methods

    def has_attribute(self, attr: str) -> bool:
        return attr in self.attributes


@dataclass
class AbstractState:
    """Simplified abstract state for a program point."""
    intervals: Dict[str, Interval] = field(default_factory=dict)
    nullity: Dict[str, NullityValue] = field(default_factory=dict)
    type_tags: Dict[str, Set[str]] = field(default_factory=dict)

    def get_interval(self, var: str) -> Interval:
        return self.intervals.get(var, Interval())

    def get_nullity(self, var: str) -> NullityValue:
        return self.nullity.get(var, NullityValue.Unknown)

    def get_type_tags(self, var: str) -> Set[str]:
        return self.type_tags.get(var, set())


@dataclass
class IRInstruction:
    """A single IR instruction."""
    opcode: str
    operands: List[str] = field(default_factory=list)
    result: Optional[str] = None
    location: Loc = field(default_factory=Loc)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRFunction:
    """An IR function."""
    name: str
    params: List[str] = field(default_factory=list)
    instructions: List[IRInstruction] = field(default_factory=list)
    location: Loc = field(default_factory=Loc)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRModule:
    """An IR module (translation unit)."""
    name: str
    functions: List[IRFunction] = field(default_factory=list)
    source_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionAnalysisResult:
    """Analysis results for a single function."""
    function_name: str
    abstract_states: Dict[int, AbstractState] = field(default_factory=dict)
    method_registry: Dict[str, TypeTag] = field(default_factory=dict)
    smt_context: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def state_at(self, instruction_index: int) -> AbstractState:
        return self.abstract_states.get(instruction_index, AbstractState())


@dataclass
class AnalysisResults:
    """Analysis results for an entire module."""
    module_name: str
    function_results: Dict[str, FunctionAnalysisResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorConfig:
    """Configuration for the bug detector."""
    confidence_threshold: float = 0.5
    enabled_classes: Set[BugClass] = field(
        default_factory=lambda: set(BugClass)
    )
    max_reports: int = 10000
    deduplicate: bool = True
    include_counterexamples: bool = True
    include_fix_suggestions: bool = True
    sarif_output: Optional[str] = None
    html_output: Optional[str] = None
    language: str = "python"
    path_sensitivity: bool = True
    false_positive_filtering: bool = True


# ---------------------------------------------------------------------------
# BugReport
# ---------------------------------------------------------------------------

@dataclass
class BugReport:
    """A single bug report."""
    bug_class: BugClass
    source_location: Loc
    message: str
    confidence: float = 0.5
    severity: Severity = Severity.Warning
    counterexample: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None
    related_guards: List[Loc] = field(default_factory=list)
    sarif_data: Optional[Dict[str, Any]] = None
    function_name: str = ""
    instruction_index: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_level(self) -> Confidence:
        return Confidence.from_score(self.confidence)

    @property
    def fingerprint(self) -> str:
        raw = (
            f"{self.bug_class.value}|{self.source_location.file}|"
            f"{self.source_location.line}|{self.source_location.column}|"
            f"{self.message}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "bug_class": self.bug_class.value,
            "source_location": {
                "file": self.source_location.file,
                "line": self.source_location.line,
                "column": self.source_location.column,
            },
            "message": self.message,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "severity": self.severity.value,
            "function_name": self.function_name,
            "fingerprint": self.fingerprint,
        }
        if self.counterexample is not None:
            d["counterexample"] = self.counterexample
        if self.fix_suggestion is not None:
            d["fix_suggestion"] = self.fix_suggestion
        if self.related_guards:
            d["related_guards"] = [str(g) for g in self.related_guards]
        return d


# ---------------------------------------------------------------------------
# Individual checkers
# ---------------------------------------------------------------------------

class ArrayBoundsChecker:
    """Detects array out-of-bounds accesses.

    Theory: access a[i] is safe iff the refinement 0 ≤ i < len(a) holds.
    If ¬(0 ≤ i < len(a)) is satisfiable, the access may be out of bounds.
    """

    INDEX_OPCODES = frozenset({"index", "subscript", "getitem", "load_element"})

    def find_index_accesses(
        self, ir_func: IRFunction
    ) -> List[Tuple[str, str, Loc, int]]:
        """Return (array_var, index_var, location, instr_index) tuples."""
        results: List[Tuple[str, str, Loc, int]] = []
        for idx, instr in enumerate(ir_func.instructions):
            if instr.opcode in self.INDEX_OPCODES and len(instr.operands) >= 2:
                array_var = instr.operands[0]
                index_var = instr.operands[1]
                results.append((array_var, index_var, instr.location, idx))
        return results

    def check_bounds(
        self,
        array_var: str,
        index_var: str,
        abstract_state: AbstractState,
        smt: Any = None,
        location: Optional[Loc] = None,
        func_name: str = "",
        instr_index: int = -1,
    ) -> Optional[BugReport]:
        """Check whether the index is provably in-bounds."""
        index_interval = abstract_state.get_interval(index_var)
        len_key = f"len({array_var})"
        array_len_interval = abstract_state.get_interval(len_key)

        lower_ok = self.check_lower_bound(index_interval)
        upper_ok = self.check_upper_bound(index_interval, array_len_interval)

        if lower_ok and upper_ok:
            return None

        parts: List[str] = []
        if not lower_ok:
            parts.append(f"index '{index_var}' may be negative ({index_interval})")
        if not upper_ok:
            parts.append(
                f"index '{index_var}' ({index_interval}) may exceed "
                f"array '{array_var}' length ({array_len_interval})"
            )
        message = f"Potential array out-of-bounds: {'; '.join(parts)}"

        confidence = self._compute_confidence(
            lower_ok, upper_ok, index_interval, array_len_interval
        )
        severity = Severity.Error if confidence > 0.8 else Severity.Warning

        counterexample: Optional[Dict[str, Any]] = None
        if smt is not None:
            counterexample = self.generate_counterexample(
                array_var, index_var, smt
            )
        elif not lower_ok and index_interval.lo is not None:
            counterexample = self.generate_counterexample_from_interval(
                array_var, index_var, index_interval, array_len_interval
            )

        bug = BugReport(
            bug_class=BugClass.ArrayOutOfBounds,
            source_location=location or Loc(),
            message=message,
            confidence=confidence,
            severity=severity,
            counterexample=counterexample,
            fix_suggestion=self.suggest_fix_detail(
                array_var, index_var, lower_ok, upper_ok
            ),
            function_name=func_name,
            instruction_index=instr_index,
        )
        return bug

    # ---- helpers -----------------------------------------------------------

    def check_lower_bound(self, index_interval: Interval) -> bool:
        """Is 0 ≤ i guaranteed?"""
        return index_interval.guaranteed_nonnegative()

    def check_upper_bound(
        self, index_interval: Interval, array_len_interval: Interval
    ) -> bool:
        """Is i < len(a) guaranteed?"""
        return index_interval.guaranteed_less_than(array_len_interval)

    def generate_counterexample(
        self, array_var: str, index_var: str, model: Any
    ) -> Dict[str, Any]:
        """Build a counterexample dict from an SMT model."""
        try:
            idx_val = model.get(index_var, -1)
            arr_len = model.get(f"len({array_var})", 0)
        except Exception:
            idx_val = -1
            arr_len = 0
        return {
            "index_value": idx_val,
            "array_length": arr_len,
            "array_var": array_var,
            "index_var": index_var,
            "trigger": f"{array_var}[{idx_val}] with len({array_var})={arr_len}",
        }

    def generate_counterexample_from_interval(
        self,
        array_var: str,
        index_var: str,
        index_interval: Interval,
        array_len_interval: Interval,
    ) -> Dict[str, Any]:
        """Heuristic counterexample from intervals (no SMT solver needed)."""
        if index_interval.lo is not None and index_interval.lo < 0:
            idx_val = index_interval.lo
        elif (
            index_interval.hi is not None
            and array_len_interval.lo is not None
            and index_interval.hi >= array_len_interval.lo
        ):
            idx_val = index_interval.hi
        else:
            idx_val = -1
        arr_len = array_len_interval.lo if array_len_interval.lo is not None else 0
        return {
            "index_value": idx_val,
            "array_length": arr_len,
            "array_var": array_var,
            "index_var": index_var,
            "trigger": f"{array_var}[{idx_val}] with len({array_var})={arr_len}",
        }

    def suggest_fix(self, bug: BugReport) -> str:
        return "Add bounds check before array access."

    def suggest_fix_detail(
        self,
        array_var: str,
        index_var: str,
        lower_ok: bool,
        upper_ok: bool,
    ) -> str:
        parts: List[str] = []
        if not lower_ok and not upper_ok:
            parts.append(
                f"if 0 <= {index_var} < len({array_var}):"
            )
        elif not lower_ok:
            parts.append(f"if {index_var} >= 0:")
        else:
            parts.append(f"if {index_var} < len({array_var}):")
        parts.append(f"    ... # access {array_var}[{index_var}]")
        return "\n".join(parts)

    def _compute_confidence(
        self,
        lower_ok: bool,
        upper_ok: bool,
        index_interval: Interval,
        array_len_interval: Interval,
    ) -> float:
        base = 0.6
        if not lower_ok and not upper_ok:
            base = 0.85
        if index_interval.is_bounded and array_len_interval.is_bounded:
            base += 0.1
        return min(base, 1.0)


class NullDerefChecker:
    """Detects null/None dereferences.

    Theory: access x.attr is safe iff ¬is_none(x) holds.  If is_none(x) is
    satisfiable the access may raise AttributeError / TypeError / NPE.
    """

    DEREF_OPCODES = frozenset({
        "getattr", "load_attr", "call_method", "deref",
        "load_field", "invoke",
    })

    def find_dereferences(
        self, ir_func: IRFunction
    ) -> List[Tuple[str, str, Loc, int]]:
        """Return (var, access_kind, location, instr_index) tuples."""
        results: List[Tuple[str, str, Loc, int]] = []
        for idx, instr in enumerate(ir_func.instructions):
            if instr.opcode in self.DEREF_OPCODES and instr.operands:
                var = instr.operands[0]
                access_kind = instr.opcode
                results.append((var, access_kind, instr.location, idx))
        return results

    def check_nullity(
        self,
        var: str,
        abstract_state: AbstractState,
        access_kind: str = "getattr",
        location: Optional[Loc] = None,
        func_name: str = "",
        instr_index: int = -1,
        attr_name: Optional[str] = None,
    ) -> Optional[BugReport]:
        """Check whether *var* may be None at the given program point."""
        nv = abstract_state.get_nullity(var)
        if not self.check_optional_access(var, nv):
            return None

        attr_part = f".{attr_name}" if attr_name else ""
        if nv == NullityValue.Null:
            message = (
                f"Definite null dereference: '{var}' is always None "
                f"when accessed via {access_kind}{attr_part}"
            )
            confidence = 0.98
            severity = Severity.Error
        else:
            message = (
                f"Potential null dereference: '{var}' may be None "
                f"when accessed via {access_kind}{attr_part}"
            )
            confidence = 0.75
            severity = Severity.Warning

        return BugReport(
            bug_class=BugClass.NullDereference,
            source_location=location or Loc(),
            message=message,
            confidence=confidence,
            severity=severity,
            counterexample=self.generate_counterexample(var, {}),
            fix_suggestion=self.suggest_fix_detail(var, access_kind, attr_name),
            function_name=func_name,
            instruction_index=instr_index,
        )

    def check_optional_access(self, var: str, nullity_value: NullityValue) -> bool:
        """Return True if a null deref is *possible*."""
        return nullity_value in (NullityValue.Null, NullityValue.MaybeNull)

    def handle_optional_chain(
        self,
        var: str,
        chain: List[str],
        abstract_state: AbstractState,
        location: Optional[Loc] = None,
        func_name: str = "",
    ) -> List[BugReport]:
        """Check a chain ``x.a.b.c`` for intermediate nulls."""
        reports: List[BugReport] = []
        current_var = var
        for part in chain:
            nv = abstract_state.get_nullity(current_var)
            if self.check_optional_access(current_var, nv):
                msg = (
                    f"Potential null in optional chain: '{current_var}' "
                    f"may be None before accessing '.{part}'"
                )
                reports.append(
                    BugReport(
                        bug_class=BugClass.NullDereference,
                        source_location=location or Loc(),
                        message=msg,
                        confidence=0.7,
                        severity=Severity.Warning,
                        fix_suggestion=self.suggest_fix_detail(
                            current_var, "getattr", part
                        ),
                        function_name=func_name,
                    )
                )
            current_var = f"{current_var}.{part}"
        return reports

    def generate_counterexample(
        self, var: str, model: Any
    ) -> Dict[str, Any]:
        return {
            "variable": var,
            "value": None,
            "trigger": f"{var} is None when dereferenced",
        }

    def suggest_fix(self, bug: BugReport) -> str:
        return "Add null check before access."

    def suggest_fix_detail(
        self,
        var: str,
        access_kind: str,
        attr_name: Optional[str] = None,
    ) -> str:
        attr_part = f".{attr_name}" if attr_name else ""
        return (
            f"if {var} is not None:\n"
            f"    ... # access {var}{attr_part}"
        )


class DivByZeroChecker:
    """Detects division-by-zero bugs.

    Theory: ``a / b`` is safe iff b ≠ 0 is guaranteed.  If b = 0 is
    satisfiable the operation may raise ZeroDivisionError.
    """

    DIV_OPCODES = frozenset({
        "div", "truediv", "floordiv", "mod", "divmod",
        "binary_divide", "binary_modulo",
    })

    def find_divisions(
        self, ir_func: IRFunction
    ) -> List[Tuple[str, str, Loc, int]]:
        """Return (dividend, divisor, location, instr_index)."""
        results: List[Tuple[str, str, Loc, int]] = []
        for idx, instr in enumerate(ir_func.instructions):
            if instr.opcode in self.DIV_OPCODES and len(instr.operands) >= 2:
                dividend = instr.operands[0]
                divisor = instr.operands[1]
                results.append((dividend, divisor, instr.location, idx))
        return results

    def check_divisor(
        self,
        divisor_var: str,
        abstract_state: AbstractState,
        smt: Any = None,
        location: Optional[Loc] = None,
        func_name: str = "",
        dividend_var: str = "",
        instr_index: int = -1,
    ) -> Optional[BugReport]:
        """Check whether the divisor may be zero."""
        interval = abstract_state.get_interval(divisor_var)
        if not interval.may_be_zero():
            return None

        if interval.lo == 0 and interval.hi == 0:
            message = (
                f"Definite division by zero: '{divisor_var}' is always 0"
            )
            confidence = 0.98
            severity = Severity.Error
        else:
            message = (
                f"Potential division by zero: '{divisor_var}' may be 0 "
                f"(interval {interval})"
            )
            confidence = 0.7
            severity = Severity.Warning

        counterexample: Optional[Dict[str, Any]] = None
        if smt is not None:
            counterexample = self.generate_counterexample(divisor_var, smt)
        else:
            counterexample = {
                "divisor_var": divisor_var,
                "divisor_value": 0,
                "trigger": f"{dividend_var} / {divisor_var} with {divisor_var}=0",
            }

        return BugReport(
            bug_class=BugClass.DivisionByZero,
            source_location=location or Loc(),
            message=message,
            confidence=confidence,
            severity=severity,
            counterexample=counterexample,
            fix_suggestion=self.suggest_fix_detail(divisor_var, dividend_var),
            function_name=func_name,
            instruction_index=instr_index,
        )

    def check_modulo(
        self,
        divisor_var: str,
        abstract_state: AbstractState,
        location: Optional[Loc] = None,
        func_name: str = "",
        dividend_var: str = "",
        instr_index: int = -1,
    ) -> Optional[BugReport]:
        """Check modulo operations (same zero-check applies)."""
        return self.check_divisor(
            divisor_var,
            abstract_state,
            smt=None,
            location=location,
            func_name=func_name,
            dividend_var=dividend_var,
            instr_index=instr_index,
        )

    def generate_counterexample(
        self, divisor_var: str, model: Any
    ) -> Dict[str, Any]:
        try:
            val = model.get(divisor_var, 0)
        except Exception:
            val = 0
        return {
            "divisor_var": divisor_var,
            "divisor_value": val,
            "trigger": f"division with {divisor_var}={val}",
        }

    def suggest_fix(self, bug: BugReport) -> str:
        return "Add zero check before division."

    def suggest_fix_detail(self, divisor_var: str, dividend_var: str) -> str:
        return (
            f"if {divisor_var} != 0:\n"
            f"    result = {dividend_var} / {divisor_var}\n"
            f"else:\n"
            f"    # handle zero divisor"
        )


class TypeConfusionChecker:
    """Detects type-tag confusion.

    Theory: calling x.method() is safe iff for every tag t ∈ Tag(x),
    t has *method* in its method set.  If ∃ t ∈ Tag(x) without *method*,
    a type error may occur at runtime.
    """

    CALL_OPCODES = frozenset({
        "call_method", "invoke", "method_call",
    })
    ATTR_OPCODES = frozenset({
        "getattr", "load_attr", "load_field",
    })

    def find_method_calls(
        self, ir_func: IRFunction
    ) -> List[Tuple[str, str, Loc, int]]:
        """Return (receiver, method, location, instr_index)."""
        results: List[Tuple[str, str, Loc, int]] = []
        for idx, instr in enumerate(ir_func.instructions):
            if instr.opcode in self.CALL_OPCODES and len(instr.operands) >= 2:
                receiver = instr.operands[0]
                method = instr.operands[1]
                results.append((receiver, method, instr.location, idx))
        return results

    def find_attribute_accesses(
        self, ir_func: IRFunction
    ) -> List[Tuple[str, str, Loc, int]]:
        """Return (var, attr, location, instr_index)."""
        results: List[Tuple[str, str, Loc, int]] = []
        for idx, instr in enumerate(ir_func.instructions):
            if instr.opcode in self.ATTR_OPCODES and len(instr.operands) >= 2:
                var = instr.operands[0]
                attr = instr.operands[1]
                results.append((var, attr, instr.location, idx))
        return results

    def check_type_compatibility(
        self,
        var: str,
        method_or_attr: str,
        type_tags: Set[str],
        method_registry: Dict[str, TypeTag],
        is_method: bool = True,
        location: Optional[Loc] = None,
        func_name: str = "",
        instr_index: int = -1,
    ) -> Optional[BugReport]:
        """Check that every possible type tag supports the given method/attr."""
        if not type_tags:
            return None

        if is_method:
            available = self.resolve_available_methods(type_tags, method_registry)
            missing = type_tags - {
                t for t in type_tags
                if t in method_registry and method_registry[t].has_method(method_or_attr)
            }
        else:
            available = self.resolve_available_attributes(type_tags, method_registry)
            missing = type_tags - {
                t for t in type_tags
                if t in method_registry and method_registry[t].has_attribute(method_or_attr)
            }

        if not missing:
            return None

        kind = "method" if is_method else "attribute"
        msg = (
            f"Type confusion: '{var}' may have tag(s) {missing} "
            f"which lack {kind} '{method_or_attr}'"
        )

        total = len(type_tags)
        fraction_bad = len(missing) / total if total > 0 else 0.0
        confidence = 0.5 + 0.4 * fraction_bad
        severity = Severity.Error if fraction_bad > 0.5 else Severity.Warning

        return BugReport(
            bug_class=BugClass.TypeConfusion,
            source_location=location or Loc(),
            message=msg,
            confidence=confidence,
            severity=severity,
            counterexample=self.generate_counterexample(
                var, type_tags, method_or_attr
            ),
            fix_suggestion=self.suggest_fix_detail(
                var, method_or_attr, missing, is_method
            ),
            function_name=func_name,
            instruction_index=instr_index,
        )

    def resolve_available_methods(
        self, tag_set: Set[str], registry: Dict[str, TypeTag]
    ) -> Set[str]:
        """Compute the intersection of methods across all tags."""
        if not tag_set:
            return set()
        method_sets = [
            set(registry[t].methods)
            for t in tag_set
            if t in registry
        ]
        if not method_sets:
            return set()
        return set.intersection(*method_sets)

    def resolve_available_attributes(
        self, tag_set: Set[str], registry: Dict[str, TypeTag]
    ) -> Set[str]:
        """Compute the intersection of attributes across all tags."""
        if not tag_set:
            return set()
        attr_sets = [
            set(registry[t].attributes)
            for t in tag_set
            if t in registry
        ]
        if not attr_sets:
            return set()
        return set.intersection(*attr_sets)

    def generate_counterexample(
        self, var: str, tag_set: Set[str], method: str
    ) -> Dict[str, Any]:
        return {
            "variable": var,
            "possible_tags": sorted(tag_set),
            "missing_method_or_attr": method,
            "trigger": (
                f"{var}.{method}() called when {var} could be "
                + " | ".join(sorted(tag_set))
            ),
        }

    def suggest_fix(self, bug: BugReport) -> str:
        return "Add isinstance check before method call."

    def suggest_fix_detail(
        self,
        var: str,
        method_or_attr: str,
        missing_tags: Set[str],
        is_method: bool,
    ) -> str:
        safe_tags = sorted(missing_tags)
        tag_list = ", ".join(safe_tags)
        kind = "method" if is_method else "attribute"
        lines = [
            f"# Guard against types lacking {kind} '{method_or_attr}':",
            f"if not isinstance({var}, ({tag_list},)):",
            f"    raise TypeError(f\"{{type({var}).__name__}} has no {kind} '{method_or_attr}'\")",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Assigns and refines confidence scores for bug reports.

    Factors:
      * path sensitivity boost  – bug is on a common execution path
      * guard proximity penalty – a guard exists nearby but doesn't cover
      * counterexample quality  – concrete witness raises confidence
      * false-positive heuristics – detect common FP patterns
    """

    PATH_SENSITIVITY_BOOST: float = 0.05
    GUARD_PROXIMITY_PENALTY: float = 0.10
    COUNTEREXAMPLE_BOOST: float = 0.10
    FP_PENALTY: float = 0.15

    def score(self, bug_report: BugReport) -> float:
        """Compute a refined confidence score."""
        base = bug_report.confidence

        base = self.path_sensitivity_boost(base, bug_report)
        base = self.guard_proximity_penalty(base, bug_report)
        base = self.counterexample_quality(base, bug_report)
        base = self.false_positive_heuristics(base, bug_report)

        return max(0.0, min(1.0, base))

    def path_sensitivity_boost(
        self, current: float, bug: BugReport
    ) -> float:
        """Increase confidence if bug is on a dominant execution path."""
        if bug.metadata.get("on_dominant_path", False):
            return current + self.PATH_SENSITIVITY_BOOST
        return current

    def guard_proximity_penalty(
        self, current: float, bug: BugReport
    ) -> float:
        """Decrease confidence if a guard is nearby but incomplete."""
        if bug.related_guards:
            for guard_loc in bug.related_guards:
                distance = abs(bug.source_location.line - guard_loc.line)
                if 0 < distance <= 5:
                    return current - self.GUARD_PROXIMITY_PENALTY
        return current

    def counterexample_quality(
        self, current: float, bug: BugReport
    ) -> float:
        """Boost confidence if a concrete counterexample is present."""
        if bug.counterexample is not None and bug.counterexample:
            if "trigger" in bug.counterexample:
                return current + self.COUNTEREXAMPLE_BOOST
            return current + self.COUNTEREXAMPLE_BOOST * 0.5
        return current

    def false_positive_heuristics(
        self, current: float, bug: BugReport
    ) -> float:
        """Apply heuristic penalties for common false positive patterns."""
        if self._is_test_code(bug):
            current -= self.FP_PENALTY
        if self._is_assertion_guarded(bug):
            current -= self.FP_PENALTY
        if self._is_dead_code(bug):
            current -= self.FP_PENALTY * 2
        return current

    # -- private heuristics --------------------------------------------------

    @staticmethod
    def _is_test_code(bug: BugReport) -> bool:
        f = bug.source_location.file.lower()
        return "test" in f or "spec" in f or "mock" in f

    @staticmethod
    def _is_assertion_guarded(bug: BugReport) -> bool:
        return bug.metadata.get("assertion_guarded", False)

    @staticmethod
    def _is_dead_code(bug: BugReport) -> bool:
        return bug.metadata.get("unreachable", False)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class BugDeduplicator:
    """Deduplicates related bug reports."""

    def is_duplicate(self, a: BugReport, b: BugReport) -> bool:
        """Two reports are duplicates if they share the same fingerprint."""
        return a.fingerprint == b.fingerprint

    def same_root_cause(self, a: BugReport, b: BugReport) -> bool:
        """Heuristic: same bug class, same file, within 3 lines."""
        if a.bug_class != b.bug_class:
            return False
        if a.source_location.file != b.source_location.file:
            return False
        return abs(a.source_location.line - b.source_location.line) <= 3

    def merge_reports(self, reports: List[BugReport]) -> List[BugReport]:
        """Remove duplicates and merge same-root-cause reports."""
        if not reports:
            return []

        seen_fps: Set[str] = set()
        unique: List[BugReport] = []
        for r in reports:
            if r.fingerprint not in seen_fps:
                seen_fps.add(r.fingerprint)
                unique.append(r)

        merged: List[BugReport] = []
        used: Set[int] = set()
        for i, a in enumerate(unique):
            if i in used:
                continue
            group = [a]
            for j in range(i + 1, len(unique)):
                if j in used:
                    continue
                if self.same_root_cause(a, unique[j]):
                    group.append(unique[j])
                    used.add(j)
            best = max(group, key=lambda r: r.confidence)
            if len(group) > 1:
                best.metadata["merged_count"] = len(group)
                best.metadata["merged_fingerprints"] = [
                    r.fingerprint for r in group
                ]
            merged.append(best)
            used.add(i)

        return merged


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class BugStatistics:
    """Aggregate statistics about detected bugs."""

    def summary(self, reports: List[BugReport]) -> Dict[str, Any]:
        return {
            "total": len(reports),
            "by_class": self.by_class(reports),
            "by_severity": self.by_severity(reports),
            "avg_confidence": (
                sum(r.confidence for r in reports) / len(reports)
                if reports
                else 0.0
            ),
            "high_confidence_count": sum(
                1 for r in reports if r.confidence > 0.9
            ),
            "false_positive_estimate": self.false_positive_estimate(reports),
        }

    def by_class(self, reports: List[BugReport]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in reports:
            key = r.bug_class.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_severity(self, reports: List[BugReport]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in reports:
            key = r.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_file(self, reports: List[BugReport]) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for r in reports:
            f = r.source_location.file
            result.setdefault(f, []).append(r.to_dict())
        return result

    def false_positive_estimate(self, reports: List[BugReport]) -> float:
        """Estimate false-positive rate from low-confidence reports."""
        if not reports:
            return 0.0
        low_conf = sum(1 for r in reports if r.confidence < 0.5)
        return low_conf / len(reports)


# ---------------------------------------------------------------------------
# SARIF 2.1.0 reporter
# ---------------------------------------------------------------------------

_SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/"
    "main/sarif-2.1/schema/sarif-schema-2.1.0.json"
)
_SARIF_VERSION = "2.1.0"
_TOOL_NAME = "refinement-type-bug-detector"
_TOOL_VERSION = "0.1.0"
_TOOL_INFO_URI = "https://github.com/example/refinement-types"

_BUG_RULE_INDEX: Dict[BugClass, int] = {
    BugClass.ArrayOutOfBounds: 0,
    BugClass.NullDereference: 1,
    BugClass.DivisionByZero: 2,
    BugClass.TypeConfusion: 3,
}

_BUG_RULE_IDS: Dict[BugClass, str] = {
    BugClass.ArrayOutOfBounds: "RT001",
    BugClass.NullDereference: "RT002",
    BugClass.DivisionByZero: "RT003",
    BugClass.TypeConfusion: "RT004",
}

_BUG_RULE_DESCRIPTIONS: Dict[BugClass, str] = {
    BugClass.ArrayOutOfBounds: (
        "Array index may be outside valid bounds [0, len(a))."
    ),
    BugClass.NullDereference: (
        "Variable may be None/null when dereferenced."
    ),
    BugClass.DivisionByZero: (
        "Divisor may be zero, causing ZeroDivisionError."
    ),
    BugClass.TypeConfusion: (
        "Receiver may have a type tag that lacks the called method/attribute."
    ),
}

_SEVERITY_TO_SARIF: Dict[Severity, str] = {
    Severity.Error: "error",
    Severity.Warning: "warning",
    Severity.Info: "note",
}


class SARIFReporter:
    """Generate SARIF 2.1.0 JSON reports."""

    def generate_sarif(self, reports: List[BugReport]) -> Dict[str, Any]:
        rules = self._build_rules()
        results = [self.sarif_result(r) for r in reports]
        return {
            "$schema": _SARIF_SCHEMA,
            "version": _SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": _TOOL_NAME,
                            "version": _TOOL_VERSION,
                            "informationUri": _TOOL_INFO_URI,
                            "rules": rules,
                        }
                    },
                    "results": results,
                }
            ],
        }

    def sarif_result(self, report: BugReport) -> Dict[str, Any]:
        rule_id = _BUG_RULE_IDS.get(report.bug_class, "RT000")
        rule_index = _BUG_RULE_INDEX.get(report.bug_class, -1)
        result: Dict[str, Any] = {
            "ruleId": rule_id,
            "ruleIndex": rule_index,
            "level": _SEVERITY_TO_SARIF.get(report.severity, "warning"),
            "message": self.sarif_message(report),
            "locations": [self.sarif_location(report.source_location)],
        }
        if report.fingerprint:
            result["fingerprints"] = {"primaryLocationLineHash": report.fingerprint}
        if report.fix_suggestion:
            result["fixes"] = [
                {
                    "description": {"text": report.fix_suggestion},
                    "artifactChanges": [],
                }
            ]
        props: Dict[str, Any] = {
            "confidence": report.confidence,
            "confidenceLevel": report.confidence_level.value,
        }
        if report.counterexample:
            props["counterexample"] = report.counterexample
        result["properties"] = props
        return result

    def sarif_location(self, loc: Loc) -> Dict[str, Any]:
        region: Dict[str, Any] = {
            "startLine": max(loc.line, 1),
            "startColumn": max(loc.column, 1),
        }
        if loc.end_line is not None:
            region["endLine"] = loc.end_line
        if loc.end_column is not None:
            region["endColumn"] = loc.end_column
        return {
            "physicalLocation": {
                "artifactLocation": {"uri": loc.file},
                "region": region,
            }
        }

    def sarif_message(self, report: BugReport) -> Dict[str, str]:
        return {"text": report.message}

    def write_sarif(
        self, reports: List[BugReport], output_path: Union[str, Path]
    ) -> None:
        sarif = self.generate_sarif(reports)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sarif, indent=2), encoding="utf-8")

    # -- private -------------------------------------------------------------

    @staticmethod
    def _build_rules() -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        for bc in BugClass:
            rules.append(
                {
                    "id": _BUG_RULE_IDS[bc],
                    "name": bc.value,
                    "shortDescription": {"text": _BUG_RULE_DESCRIPTIONS[bc]},
                    "fullDescription": {"text": _BUG_RULE_DESCRIPTIONS[bc]},
                    "defaultConfiguration": {"level": "warning"},
                }
            )
        return rules


# ---------------------------------------------------------------------------
# HTML reporter
# ---------------------------------------------------------------------------

_SEVERITY_COLORS: Dict[Severity, str] = {
    Severity.Error: "#e74c3c",
    Severity.Warning: "#f39c12",
    Severity.Info: "#3498db",
}

_BUGCLASS_ICONS: Dict[BugClass, str] = {
    BugClass.ArrayOutOfBounds: "📐",
    BugClass.NullDereference: "💀",
    BugClass.DivisionByZero: "➗",
    BugClass.TypeConfusion: "🏷️",
}

_HTML_TEMPLATE_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Bug Detection Report</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 2em; background: #f9f9f9; }}
h1 {{ color: #2c3e50; }}
.summary {{ background: #fff; border-radius: 8px; padding: 1em; margin-bottom: 2em; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.bug-card {{ background: #fff; border-left: 4px solid {border_color}; border-radius: 4px; padding: 1em; margin-bottom: 1em; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
.bug-card h3 {{ margin-top: 0; }}
.severity {{ font-weight: bold; text-transform: uppercase; font-size: 0.85em; }}
.code-snippet {{ background: #2d2d2d; color: #f8f8f2; padding: 1em; border-radius: 4px; font-family: 'Fira Code', 'Consolas', monospace; overflow-x: auto; white-space: pre; }}
.confidence-bar {{ height: 6px; border-radius: 3px; background: #ecf0f1; margin-top: 0.5em; }}
.confidence-fill {{ height: 6px; border-radius: 3px; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 4px; }}
.counterexample {{ background: #fdf2e9; border-radius: 4px; padding: 0.8em; margin-top: 0.5em; font-size: 0.9em; }}
.fix-suggestion {{ background: #eafaf1; border-radius: 4px; padding: 0.8em; margin-top: 0.5em; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>🔍 Bug Detection Report</h1>
"""

_HTML_TEMPLATE_TAIL = """\
</body>
</html>
"""


class HTMLReporter:
    """Generates an HTML report with code snippets and annotations."""

    def generate_html(
        self,
        reports: List[BugReport],
        source_files: Optional[Dict[str, str]] = None,
    ) -> str:
        parts: List[str] = []
        parts.append(
            _HTML_TEMPLATE_HEAD.format(border_color="#3498db")
        )

        stats = BugStatistics()
        summary = stats.summary(reports)
        parts.append(self._render_summary(summary))

        for report in reports:
            parts.append(self.render_bug_card(report, source_files))

        parts.append(_HTML_TEMPLATE_TAIL)
        return "".join(parts)

    def render_bug_card(
        self,
        report: BugReport,
        source_files: Optional[Dict[str, str]] = None,
    ) -> str:
        color = _SEVERITY_COLORS.get(report.severity, "#95a5a6")
        icon = _BUGCLASS_ICONS.get(report.bug_class, "🐛")
        conf_pct = int(report.confidence * 100)
        conf_color = self._confidence_color(report.confidence)

        parts: List[str] = [
            f'<div class="bug-card" style="border-left-color:{color};">',
            f"<h3>{icon} {html_mod.escape(report.bug_class.value)}</h3>",
            f'<span class="severity" style="color:{color};">'
            f"{html_mod.escape(report.severity.value)}</span>",
            f" &mdash; {html_mod.escape(str(report.source_location))}",
            f"<p>{html_mod.escape(report.message)}</p>",
            f'<div class="confidence-bar">'
            f'<div class="confidence-fill" style="width:{conf_pct}%;'
            f'background:{conf_color};"></div></div>',
            f"<small>Confidence: {conf_pct}% "
            f"({html_mod.escape(report.confidence_level.value)})</small>",
        ]

        snippet = self.render_code_snippet(report, source_files)
        if snippet:
            parts.append(snippet)

        if report.counterexample:
            parts.append(
                '<div class="counterexample"><strong>Counterexample:</strong> '
                + html_mod.escape(json.dumps(report.counterexample, indent=2))
                + "</div>"
            )

        if report.fix_suggestion:
            parts.append(
                '<div class="fix-suggestion"><strong>Suggested fix:</strong>'
                f'<pre>{html_mod.escape(report.fix_suggestion)}</pre></div>'
            )

        parts.append("</div>")
        return "\n".join(parts)

    def render_code_snippet(
        self,
        report: BugReport,
        source_files: Optional[Dict[str, str]] = None,
    ) -> str:
        if source_files is None:
            return ""
        source = source_files.get(report.source_location.file)
        if source is None:
            return ""

        lines = source.splitlines()
        target_line = report.source_location.line - 1
        start = max(0, target_line - 3)
        end = min(len(lines), target_line + 4)
        snippet_lines: List[str] = []
        for i in range(start, end):
            marker = ">>>" if i == target_line else "   "
            snippet_lines.append(
                f"{marker} {i + 1:4d} | {html_mod.escape(lines[i])}"
            )
        return f'<div class="code-snippet">{"chr(10)".join(snippet_lines)}</div>'

    def write_html(
        self,
        reports: List[BugReport],
        output_path: Union[str, Path],
        source_files: Optional[Dict[str, str]] = None,
    ) -> None:
        content = self.generate_html(reports, source_files)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _render_summary(summary: Dict[str, Any]) -> str:
        total = summary.get("total", 0)
        by_sev = summary.get("by_severity", {})
        errors = by_sev.get("error", 0)
        warnings = by_sev.get("warning", 0)
        infos = by_sev.get("info", 0)
        fp_est = summary.get("false_positive_estimate", 0.0)
        return (
            '<div class="summary">'
            f"<h2>Summary</h2>"
            f"<p><strong>Total bugs:</strong> {total} &nbsp;"
            f'<span class="tag" style="background:#e74c3c;color:#fff;">'
            f"{errors} errors</span>"
            f'<span class="tag" style="background:#f39c12;color:#fff;">'
            f"{warnings} warnings</span>"
            f'<span class="tag" style="background:#3498db;color:#fff;">'
            f"{infos} info</span></p>"
            f"<p><strong>Estimated false-positive rate:</strong> "
            f"{fp_est:.0%}</p>"
            "</div>"
        )

    @staticmethod
    def _confidence_color(score: float) -> str:
        if score > 0.9:
            return "#27ae60"
        if score > 0.7:
            return "#f39c12"
        if score > 0.5:
            return "#e67e22"
        return "#e74c3c"


# ---------------------------------------------------------------------------
# Fix-suggestion engine
# ---------------------------------------------------------------------------

class FixSuggestionEngine:
    """Generates language-aware fix suggestions."""

    def suggest_bounds_check(self, bug: BugReport) -> str:
        ce = bug.counterexample or {}
        arr = ce.get("array_var", "arr")
        idx = ce.get("index_var", "i")
        return (
            f"if 0 <= {idx} < len({arr}):\n"
            f"    value = {arr}[{idx}]\n"
            f"else:\n"
            f"    # handle out-of-bounds"
        )

    def suggest_null_check(self, bug: BugReport) -> str:
        ce = bug.counterexample or {}
        var = ce.get("variable", "x")
        return (
            f"if {var} is not None:\n"
            f"    # safe to access {var}\n"
            f"else:\n"
            f"    # handle None case"
        )

    def suggest_zero_check(self, bug: BugReport) -> str:
        ce = bug.counterexample or {}
        dv = ce.get("divisor_var", "d")
        return (
            f"if {dv} != 0:\n"
            f"    result = numerator / {dv}\n"
            f"else:\n"
            f"    # handle zero divisor"
        )

    def suggest_type_check(self, bug: BugReport) -> str:
        ce = bug.counterexample or {}
        var = ce.get("variable", "x")
        method = ce.get("missing_method_or_attr", "method")
        return (
            f"if hasattr({var}, '{method}'):\n"
            f"    {var}.{method}()\n"
            f"else:\n"
            f"    raise TypeError(f'{{type({var}).__name__}} lacks {method}')"
        )

    def suggest_for_language(
        self, bug: BugReport, language: str = "python"
    ) -> str:
        """Return a fix snippet in the requested language syntax."""
        dispatch = {
            BugClass.ArrayOutOfBounds: self._bounds_for_lang,
            BugClass.NullDereference: self._null_for_lang,
            BugClass.DivisionByZero: self._zero_for_lang,
            BugClass.TypeConfusion: self._type_for_lang,
        }
        handler = dispatch.get(bug.bug_class)
        if handler is None:
            return "// no suggestion available"
        return handler(bug, language)

    # -- per-language helpers ------------------------------------------------

    def _bounds_for_lang(self, bug: BugReport, lang: str) -> str:
        ce = bug.counterexample or {}
        arr = ce.get("array_var", "arr")
        idx = ce.get("index_var", "i")
        if lang in ("typescript", "ts", "javascript", "js"):
            return (
                f"if ({idx} >= 0 && {idx} < {arr}.length) {{\n"
                f"    const value = {arr}[{idx}];\n"
                f"}} else {{\n"
                f"    // handle out-of-bounds\n"
                f"}}"
            )
        return self.suggest_bounds_check(bug)

    def _null_for_lang(self, bug: BugReport, lang: str) -> str:
        ce = bug.counterexample or {}
        var = ce.get("variable", "x")
        if lang in ("typescript", "ts", "javascript", "js"):
            return (
                f"if ({var} != null) {{\n"
                f"    // safe to access {var}\n"
                f"}} else {{\n"
                f"    // handle null/undefined\n"
                f"}}"
            )
        return self.suggest_null_check(bug)

    def _zero_for_lang(self, bug: BugReport, lang: str) -> str:
        ce = bug.counterexample or {}
        dv = ce.get("divisor_var", "d")
        if lang in ("typescript", "ts", "javascript", "js"):
            return (
                f"if ({dv} !== 0) {{\n"
                f"    const result = numerator / {dv};\n"
                f"}} else {{\n"
                f"    // handle zero divisor\n"
                f"}}"
            )
        return self.suggest_zero_check(bug)

    def _type_for_lang(self, bug: BugReport, lang: str) -> str:
        ce = bug.counterexample or {}
        var = ce.get("variable", "x")
        method = ce.get("missing_method_or_attr", "method")
        if lang in ("typescript", "ts", "javascript", "js"):
            return (
                f"if (typeof ({var} as any).{method} === 'function') {{\n"
                f"    ({var} as any).{method}();\n"
                f"}} else {{\n"
                f"    throw new TypeError("
                f"`${{typeof {var}}} lacks {method}`);\n"
                f"}}"
            )
        return self.suggest_type_check(bug)


# ---------------------------------------------------------------------------
# Main BugDetector
# ---------------------------------------------------------------------------

class BugDetector:
    """Orchestrates all individual checkers to produce a list of BugReports."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self.array_checker = ArrayBoundsChecker()
        self.null_checker = NullDerefChecker()
        self.div_checker = DivByZeroChecker()
        self.type_checker = TypeConfusionChecker()
        self.scorer = ConfidenceScorer()
        self.deduplicator = BugDeduplicator()
        self.fix_engine = FixSuggestionEngine()
        self.stats = BugStatistics()
        self.sarif_reporter = SARIFReporter()
        self.html_reporter = HTMLReporter()

    # ---- public API --------------------------------------------------------

    def detect_all(
        self,
        ir_module: IRModule,
        analysis_results: AnalysisResults,
        config: Optional[DetectorConfig] = None,
    ) -> List[BugReport]:
        """Run all enabled checkers on every function in the module."""
        cfg = config or self.config
        all_reports: List[BugReport] = []

        for ir_func in ir_module.functions:
            func_result = analysis_results.function_results.get(ir_func.name)
            if func_result is None:
                func_result = FunctionAnalysisResult(function_name=ir_func.name)
            reports = self.detect_in_function(ir_func, func_result, cfg)
            all_reports.extend(reports)

        # Rescore
        for r in all_reports:
            r.confidence = self.scorer.score(r)

        # Enrich fix suggestions
        if cfg.include_fix_suggestions:
            for r in all_reports:
                if r.fix_suggestion is None:
                    r.fix_suggestion = self.fix_engine.suggest_for_language(
                        r, cfg.language
                    )

        # Filter + deduplicate + rank
        all_reports = self.filter_by_confidence(
            all_reports, cfg.confidence_threshold
        )
        if cfg.deduplicate:
            all_reports = self.deduplicate(all_reports)
        all_reports = self.rank_by_severity(all_reports)

        # Limit
        if len(all_reports) > cfg.max_reports:
            all_reports = all_reports[: cfg.max_reports]

        # SARIF output
        if cfg.sarif_output:
            self.sarif_reporter.write_sarif(all_reports, cfg.sarif_output)

        # HTML output
        if cfg.html_output:
            self.html_reporter.write_html(all_reports, cfg.html_output)

        return all_reports

    def detect_in_function(
        self,
        ir_func: IRFunction,
        func_result: FunctionAnalysisResult,
        config: Optional[DetectorConfig] = None,
    ) -> List[BugReport]:
        """Run all enabled checkers on a single function."""
        cfg = config or self.config
        reports: List[BugReport] = []

        if BugClass.ArrayOutOfBounds in cfg.enabled_classes:
            reports.extend(
                self._check_array_bounds(ir_func, func_result)
            )
        if BugClass.NullDereference in cfg.enabled_classes:
            reports.extend(
                self._check_null_derefs(ir_func, func_result)
            )
        if BugClass.DivisionByZero in cfg.enabled_classes:
            reports.extend(
                self._check_div_by_zero(ir_func, func_result)
            )
        if BugClass.TypeConfusion in cfg.enabled_classes:
            reports.extend(
                self._check_type_confusion(ir_func, func_result)
            )

        return reports

    def filter_by_confidence(
        self, reports: List[BugReport], threshold: float
    ) -> List[BugReport]:
        return [r for r in reports if r.confidence >= threshold]

    def deduplicate(self, reports: List[BugReport]) -> List[BugReport]:
        return self.deduplicator.merge_reports(reports)

    def rank_by_severity(self, reports: List[BugReport]) -> List[BugReport]:
        severity_order = {Severity.Error: 0, Severity.Warning: 1, Severity.Info: 2}
        return sorted(
            reports,
            key=lambda r: (severity_order.get(r.severity, 9), -r.confidence),
        )

    # ---- per-checker orchestration -----------------------------------------

    def _check_array_bounds(
        self, ir_func: IRFunction, func_result: FunctionAnalysisResult
    ) -> List[BugReport]:
        reports: List[BugReport] = []
        accesses = self.array_checker.find_index_accesses(ir_func)
        for array_var, index_var, loc, idx in accesses:
            state = func_result.state_at(idx)
            bug = self.array_checker.check_bounds(
                array_var,
                index_var,
                state,
                smt=func_result.smt_context,
                location=loc,
                func_name=ir_func.name,
                instr_index=idx,
            )
            if bug is not None:
                reports.append(bug)
        return reports

    def _check_null_derefs(
        self, ir_func: IRFunction, func_result: FunctionAnalysisResult
    ) -> List[BugReport]:
        reports: List[BugReport] = []
        derefs = self.null_checker.find_dereferences(ir_func)
        for var, access_kind, loc, idx in derefs:
            state = func_result.state_at(idx)
            attr_name = None
            instr = ir_func.instructions[idx] if idx < len(ir_func.instructions) else None
            if instr and len(instr.operands) >= 2:
                attr_name = instr.operands[1]
            bug = self.null_checker.check_nullity(
                var,
                state,
                access_kind=access_kind,
                location=loc,
                func_name=ir_func.name,
                instr_index=idx,
                attr_name=attr_name,
            )
            if bug is not None:
                reports.append(bug)
        return reports

    def _check_div_by_zero(
        self, ir_func: IRFunction, func_result: FunctionAnalysisResult
    ) -> List[BugReport]:
        reports: List[BugReport] = []
        divisions = self.div_checker.find_divisions(ir_func)
        for dividend, divisor, loc, idx in divisions:
            state = func_result.state_at(idx)
            bug = self.div_checker.check_divisor(
                divisor,
                state,
                smt=func_result.smt_context,
                location=loc,
                func_name=ir_func.name,
                dividend_var=dividend,
                instr_index=idx,
            )
            if bug is not None:
                reports.append(bug)
        return reports

    def _check_type_confusion(
        self, ir_func: IRFunction, func_result: FunctionAnalysisResult
    ) -> List[BugReport]:
        reports: List[BugReport] = []

        method_calls = self.type_checker.find_method_calls(ir_func)
        for receiver, method, loc, idx in method_calls:
            state = func_result.state_at(idx)
            tags = state.get_type_tags(receiver)
            if tags:
                bug = self.type_checker.check_type_compatibility(
                    receiver,
                    method,
                    tags,
                    func_result.method_registry,
                    is_method=True,
                    location=loc,
                    func_name=ir_func.name,
                    instr_index=idx,
                )
                if bug is not None:
                    reports.append(bug)

        attr_accesses = self.type_checker.find_attribute_accesses(ir_func)
        for var, attr, loc, idx in attr_accesses:
            state = func_result.state_at(idx)
            tags = state.get_type_tags(var)
            if tags:
                bug = self.type_checker.check_type_compatibility(
                    var,
                    attr,
                    tags,
                    func_result.method_registry,
                    is_method=False,
                    location=loc,
                    func_name=ir_func.name,
                    instr_index=idx,
                )
                if bug is not None:
                    reports.append(bug)

        return reports


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def detect_bugs(
    ir_module: IRModule,
    analysis_results: AnalysisResults,
    config: Optional[DetectorConfig] = None,
) -> List[BugReport]:
    """Top-level convenience function."""
    detector = BugDetector(config)
    return detector.detect_all(ir_module, analysis_results, config)


def detect_bugs_json(
    ir_module: IRModule,
    analysis_results: AnalysisResults,
    config: Optional[DetectorConfig] = None,
) -> str:
    """Return bug reports as a JSON string."""
    reports = detect_bugs(ir_module, analysis_results, config)
    return json.dumps([r.to_dict() for r in reports], indent=2)


def detect_bugs_sarif(
    ir_module: IRModule,
    analysis_results: AnalysisResults,
    config: Optional[DetectorConfig] = None,
) -> Dict[str, Any]:
    """Return bug reports as a SARIF 2.1.0 dict."""
    reports = detect_bugs(ir_module, analysis_results, config)
    return SARIFReporter().generate_sarif(reports)


def detect_bugs_html(
    ir_module: IRModule,
    analysis_results: AnalysisResults,
    source_files: Optional[Dict[str, str]] = None,
    config: Optional[DetectorConfig] = None,
) -> str:
    """Return bug reports as an HTML string."""
    reports = detect_bugs(ir_module, analysis_results, config)
    return HTMLReporter().generate_html(reports, source_files)


# ---------------------------------------------------------------------------
# Self-test / smoke-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Quick smoke test exercising every checker."""

    # Build a small IR module
    instrs = [
        IRInstruction(
            opcode="index",
            operands=["data", "idx"],
            location=Loc("example.py", 10, 4),
        ),
        IRInstruction(
            opcode="getattr",
            operands=["result", "value"],
            location=Loc("example.py", 15, 4),
        ),
        IRInstruction(
            opcode="div",
            operands=["total", "count"],
            location=Loc("example.py", 20, 4),
        ),
        IRInstruction(
            opcode="call_method",
            operands=["obj", "serialize"],
            location=Loc("example.py", 25, 4),
        ),
    ]
    func = IRFunction(name="process", instructions=instrs)
    module = IRModule(name="test_module", functions=[func], source_file="example.py")

    # Build analysis results
    state = AbstractState(
        intervals={
            "idx": Interval(-1, 10),
            "len(data)": Interval(5, 5),
            "count": Interval(0, 100),
        },
        nullity={"result": NullityValue.MaybeNull},
        type_tags={"obj": {"DictLike", "ListLike"}},
    )
    registry = {
        "DictLike": TypeTag(
            "DictLike",
            methods=frozenset({"serialize", "keys"}),
            attributes=frozenset({"data"}),
        ),
        "ListLike": TypeTag(
            "ListLike",
            methods=frozenset({"append", "sort"}),
            attributes=frozenset({"length"}),
        ),
    }
    func_result = FunctionAnalysisResult(
        function_name="process",
        abstract_states={0: state, 1: state, 2: state, 3: state},
        method_registry=registry,
    )
    analysis = AnalysisResults(
        module_name="test_module",
        function_results={"process": func_result},
    )

    # Detect
    config = DetectorConfig(confidence_threshold=0.0)
    reports = detect_bugs(module, analysis, config)

    assert len(reports) > 0, "Expected at least one bug report"

    # Check all four bug classes are represented
    classes_found = {r.bug_class for r in reports}
    assert BugClass.ArrayOutOfBounds in classes_found, "Missing array OOB"
    assert BugClass.NullDereference in classes_found, "Missing null deref"
    assert BugClass.DivisionByZero in classes_found, "Missing div-by-zero"
    assert BugClass.TypeConfusion in classes_found, "Missing type confusion"

    # Check JSON serialization
    j = detect_bugs_json(module, analysis, config)
    parsed = json.loads(j)
    assert isinstance(parsed, list)

    # Check SARIF
    sarif = detect_bugs_sarif(module, analysis, config)
    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"][0]["results"]) == len(reports)

    # Check HTML
    html_out = detect_bugs_html(module, analysis, config=config)
    assert "<html" in html_out

    # Statistics
    stats = BugStatistics()
    s = stats.summary(reports)
    assert s["total"] == len(reports)

    # Deduplication idempotence
    dedup = BugDeduplicator()
    merged = dedup.merge_reports(reports)
    merged2 = dedup.merge_reports(merged)
    assert len(merged2) == len(merged)

    # Fix suggestion engine
    fse = FixSuggestionEngine()
    for r in reports:
        py_fix = fse.suggest_for_language(r, "python")
        ts_fix = fse.suggest_for_language(r, "typescript")
        assert py_fix
        assert ts_fix

    print(f"Self-test passed: {len(reports)} bugs detected across all 4 classes.")


if __name__ == "__main__":
    _self_test()
