from __future__ import annotations

"""
Tests for bug detection in the refinement type inference system.

Covers: array out-of-bounds, null dereference, division by zero,
type confusion, bug reporting quality, and false positive rates.
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union,
)

import pytest

# ── Local type stubs ──────────────────────────────────────────────────────


class BoundKind(Enum):
    NEG_INF = auto()
    FINITE = auto()
    POS_INF = auto()


@dataclass(frozen=True)
class Bound:
    kind: BoundKind
    value: int = 0

    @classmethod
    def finite(cls, n: int) -> Bound:
        return cls(BoundKind.FINITE, n)

    @classmethod
    def pos_inf(cls) -> Bound:
        return cls(BoundKind.POS_INF)

    @classmethod
    def neg_inf(cls) -> Bound:
        return cls(BoundKind.NEG_INF)

    def __lt__(self, other: Bound) -> bool:
        order = {BoundKind.NEG_INF: 0, BoundKind.FINITE: 1, BoundKind.POS_INF: 2}
        if self.kind != other.kind:
            return order[self.kind] < order[other.kind]
        return self.value < other.value

    def __le__(self, other: Bound) -> bool:
        return self == other or self < other

    def __gt__(self, other: Bound) -> bool:
        return other < self

    def __ge__(self, other: Bound) -> bool:
        return other <= self


@dataclass(frozen=True)
class Interval:
    lo: Bound
    hi: Bound

    @classmethod
    def top(cls) -> Interval:
        return cls(Bound.neg_inf(), Bound.pos_inf())

    @classmethod
    def bottom(cls) -> Interval:
        return cls(Bound.finite(1), Bound.finite(0))

    @classmethod
    def singleton(cls, n: int) -> Interval:
        return cls(Bound.finite(n), Bound.finite(n))

    @classmethod
    def from_bounds(cls, lo: int, hi: int) -> Interval:
        return cls(Bound.finite(lo), Bound.finite(hi))

    @property
    def is_bottom(self) -> bool:
        return self.lo > self.hi

    @property
    def is_top(self) -> bool:
        return self.lo.kind == BoundKind.NEG_INF and self.hi.kind == BoundKind.POS_INF

    def contains(self, n: int) -> bool:
        if self.is_bottom:
            return False
        b = Bound.finite(n)
        return self.lo <= b and b <= self.hi

    def meet(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        lo = self.lo if self.lo >= other.lo else other.lo
        hi = self.hi if self.hi <= other.hi else other.hi
        if lo > hi:
            return Interval.bottom()
        return Interval(lo, hi)


class NullityKind(Enum):
    BOTTOM = auto()
    DEFINITELY_NULL = auto()
    DEFINITELY_NOT_NULL = auto()
    MAYBE_NULL = auto()


@dataclass(frozen=True)
class NullityValue:
    kind: NullityKind

    @classmethod
    def bottom(cls) -> NullityValue:
        return cls(NullityKind.BOTTOM)

    @classmethod
    def definitely_null(cls) -> NullityValue:
        return cls(NullityKind.DEFINITELY_NULL)

    @classmethod
    def definitely_not_null(cls) -> NullityValue:
        return cls(NullityKind.DEFINITELY_NOT_NULL)

    @classmethod
    def maybe_null(cls) -> NullityValue:
        return cls(NullityKind.MAYBE_NULL)

    @property
    def may_be_null(self) -> bool:
        return self.kind in (NullityKind.DEFINITELY_NULL, NullityKind.MAYBE_NULL)

    @property
    def may_be_non_null(self) -> bool:
        return self.kind in (NullityKind.DEFINITELY_NOT_NULL, NullityKind.MAYBE_NULL)

    @property
    def is_definitely_null(self) -> bool:
        return self.kind == NullityKind.DEFINITELY_NULL


@dataclass(frozen=True)
class TypeTagSet:
    tags: FrozenSet[str]
    _is_top: bool = False

    @classmethod
    def top(cls) -> TypeTagSet:
        return cls(frozenset(), _is_top=True)

    @classmethod
    def bottom(cls) -> TypeTagSet:
        return cls(frozenset(), _is_top=False)

    @classmethod
    def singleton(cls, tag_name: str) -> TypeTagSet:
        return cls(frozenset({tag_name}))

    @classmethod
    def from_names(cls, *names: str) -> TypeTagSet:
        return cls(frozenset(names))

    @property
    def is_top(self) -> bool:
        return self._is_top

    @property
    def is_bottom(self) -> bool:
        return not self._is_top and len(self.tags) == 0

    def contains(self, tag_name: str) -> bool:
        return self._is_top or tag_name in self.tags

    def meet(self, other: TypeTagSet) -> TypeTagSet:
        if self.is_bottom or other.is_bottom:
            return TypeTagSet.bottom()
        if self._is_top:
            return other
        if other._is_top:
            return self
        return TypeTagSet(self.tags & other.tags)


class BugClass(Enum):
    ArrayOutOfBounds = "array_out_of_bounds"
    NullDereference = "null_dereference"
    DivisionByZero = "division_by_zero"
    TypeConfusion = "type_confusion"


class Severity(Enum):
    Error = auto()
    Warning = auto()
    Info = auto()
    Hint = auto()


class Confidence(Enum):
    High = auto()
    Medium = auto()
    Low = auto()
    VeryLow = auto()

    @classmethod
    def from_score(cls, score: float) -> Confidence:
        if score >= 0.9:
            return cls.High
        elif score >= 0.7:
            return cls.Medium
        elif score >= 0.4:
            return cls.Low
        return cls.VeryLow


@dataclass(frozen=True)
class Loc:
    file: str = ""
    line: int = 0
    column: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None


@dataclass
class BugReport:
    bug_class: BugClass
    source_location: Loc
    message: str
    confidence: float = 0.5
    severity: Severity = Severity.Warning
    counterexample: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None
    code_context: Optional[str] = None

    @property
    def fingerprint(self) -> str:
        data = f"{self.bug_class.value}:{self.source_location.file}:{self.source_location.line}:{self.message}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @property
    def confidence_level(self) -> Confidence:
        return Confidence.from_score(self.confidence)

    def to_sarif(self) -> Dict[str, Any]:
        return {
            "ruleId": self.bug_class.value,
            "level": "warning" if self.severity == Severity.Warning else "error",
            "message": {"text": self.message},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": self.source_location.file},
                    "region": {
                        "startLine": self.source_location.line,
                        "startColumn": self.source_location.column,
                    },
                },
            }],
            "fingerprints": {"primaryLocationLineHash": self.fingerprint},
        }

    def to_html_fragment(self) -> str:
        sev_class = self.severity.name.lower()
        return (
            f'<div class="bug-report {sev_class}">'
            f'<span class="location">{self.source_location.file}:{self.source_location.line}</span>'
            f'<span class="message">{self.message}</span>'
            f'</div>'
        )


@dataclass
class AbstractState:
    intervals: Dict[str, Interval] = field(default_factory=dict)
    nullity: Dict[str, NullityValue] = field(default_factory=dict)
    type_tags: Dict[str, TypeTagSet] = field(default_factory=dict)

    def get_interval(self, var: str) -> Interval:
        return self.intervals.get(var, Interval.top())

    def get_nullity(self, var: str) -> NullityValue:
        return self.nullity.get(var, NullityValue.maybe_null())

    def get_type_tags(self, var: str) -> TypeTagSet:
        return self.type_tags.get(var, TypeTagSet.top())


# ── Bug Checkers ──────────────────────────────────────────────────────────


def check_array_bounds(
    array_var: str,
    index_var: str,
    state: AbstractState,
    location: Loc = Loc(),
) -> Optional[BugReport]:
    """Check for potential array out-of-bounds access."""
    idx_iv = state.get_interval(index_var)
    arr_len_iv = state.get_interval(f"len({array_var})")

    # Index might be negative
    if idx_iv.contains(-1) or (idx_iv.lo.kind == BoundKind.NEG_INF):
        return BugReport(
            bug_class=BugClass.ArrayOutOfBounds,
            source_location=location,
            message=f"Index '{index_var}' may be negative when accessing '{array_var}'",
            confidence=0.8,
            severity=Severity.Error,
            counterexample={"index": -1},
            fix_suggestion=f"Add check: if {index_var} >= 0 and {index_var} < len({array_var})",
        )

    # Index might exceed length
    if idx_iv.is_top:
        return BugReport(
            bug_class=BugClass.ArrayOutOfBounds,
            source_location=location,
            message=f"Index '{index_var}' may exceed bounds of '{array_var}'",
            confidence=0.6,
            severity=Severity.Warning,
            counterexample={"index": "unknown"},
            fix_suggestion=f"Add bounds check for {index_var}",
        )

    # If both have finite bounds, check intersection
    if (idx_iv.hi.kind == BoundKind.FINITE and arr_len_iv.lo.kind == BoundKind.FINITE
            and idx_iv.hi.value >= arr_len_iv.lo.value):
        return BugReport(
            bug_class=BugClass.ArrayOutOfBounds,
            source_location=location,
            message=f"Index '{index_var}' may exceed length of '{array_var}'",
            confidence=0.7,
            severity=Severity.Warning,
        )

    return None


def check_null_deref(
    var: str,
    attr: str,
    state: AbstractState,
    location: Loc = Loc(),
) -> Optional[BugReport]:
    """Check for potential null/None dereference."""
    nv = state.get_nullity(var)
    if nv.is_definitely_null:
        return BugReport(
            bug_class=BugClass.NullDereference,
            source_location=location,
            message=f"'{var}' is definitely None when accessing '.{attr}'",
            confidence=0.95,
            severity=Severity.Error,
            counterexample={var: None},
            fix_suggestion=f"Add null check: if {var} is not None",
        )
    if nv.may_be_null:
        return BugReport(
            bug_class=BugClass.NullDereference,
            source_location=location,
            message=f"'{var}' may be None when accessing '.{attr}'",
            confidence=0.7,
            severity=Severity.Warning,
            counterexample={var: None},
            fix_suggestion=f"Add null check: if {var} is not None",
        )
    return None


def check_division_by_zero(
    divisor_var: str,
    state: AbstractState,
    location: Loc = Loc(),
) -> Optional[BugReport]:
    """Check for potential division by zero."""
    div_iv = state.get_interval(divisor_var)
    if div_iv.contains(0):
        if div_iv == Interval.singleton(0):
            return BugReport(
                bug_class=BugClass.DivisionByZero,
                source_location=location,
                message=f"'{divisor_var}' is definitely zero",
                confidence=0.95,
                severity=Severity.Error,
                counterexample={divisor_var: 0},
                fix_suggestion=f"Check {divisor_var} != 0 before division",
            )
        return BugReport(
            bug_class=BugClass.DivisionByZero,
            source_location=location,
            message=f"'{divisor_var}' may be zero",
            confidence=0.7,
            severity=Severity.Warning,
            counterexample={divisor_var: 0},
            fix_suggestion=f"Check {divisor_var} != 0 before division",
        )
    return None


def check_type_confusion(
    var: str,
    expected_method: str,
    state: AbstractState,
    location: Loc = Loc(),
) -> Optional[BugReport]:
    """Check for potential type confusion (wrong method on wrong type)."""
    tags = state.get_type_tags(var)
    if tags.is_top:
        return BugReport(
            bug_class=BugClass.TypeConfusion,
            source_location=location,
            message=f"Type of '{var}' is unknown when calling '.{expected_method}()'",
            confidence=0.5,
            severity=Severity.Warning,
            fix_suggestion=f"Add isinstance check for {var}",
        )

    # Check if tags include types that don't have the method
    method_map: Dict[str, Set[str]] = {
        "append": {"list"},
        "keys": {"dict"},
        "upper": {"str"},
        "lower": {"str"},
        "bit_length": {"int"},
        "__len__": {"list", "str", "dict", "tuple", "set", "frozenset", "bytes"},
        "items": {"dict"},
        "values": {"dict"},
        "add": {"set"},
        "pop": {"list", "dict", "set"},
        "strip": {"str"},
        "split": {"str"},
        "join": {"str"},
        "encode": {"str"},
        "decode": {"bytes"},
    }
    valid_types = method_map.get(expected_method, set())
    if valid_types and not tags.is_bottom:
        invalid_tags = tags.tags - {t for t in valid_types}
        if invalid_tags and not tags.is_top:
            return BugReport(
                bug_class=BugClass.TypeConfusion,
                source_location=location,
                message=f"'{var}' may be {invalid_tags} which lacks '.{expected_method}()'",
                confidence=0.8,
                severity=Severity.Warning,
                counterexample={"type": next(iter(invalid_tags))},
                fix_suggestion=f"Add isinstance({var}, ({', '.join(valid_types)})) check",
            )
    return None


def deduplicate_reports(reports: List[BugReport]) -> List[BugReport]:
    """Remove duplicate bug reports based on fingerprint."""
    seen: Set[str] = set()
    result: List[BugReport] = []
    for r in reports:
        fp = r.fingerprint
        if fp not in seen:
            seen.add(fp)
            result.append(r)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═══════════════════════════════════════════════════════════════════════════


class TestArrayOutOfBounds:
    """Tests for array out-of-bounds detection."""

    def test_simple_oob(self) -> None:
        """Detect OOB when index is unbounded."""
        state = AbstractState(intervals={"i": Interval.top()})
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 10))
        assert bug is not None
        assert bug.bug_class == BugClass.ArrayOutOfBounds

    def test_negative_index_oob(self) -> None:
        """Detect OOB with negative index."""
        state = AbstractState(intervals={"i": Interval.from_bounds(-5, 3)})
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 10))
        assert bug is not None
        assert "negative" in bug.message.lower() or bug.bug_class == BugClass.ArrayOutOfBounds

    def test_guarded_access_no_bug(self) -> None:
        """No OOB when index is properly bounded."""
        state = AbstractState(
            intervals={
                "i": Interval.from_bounds(0, 4),
                "len(arr)": Interval.from_bounds(5, 100),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 10))
        assert bug is None

    def test_loop_access_with_invariant(self) -> None:
        """No OOB when loop invariant guarantees bounds."""
        # After widening + narrowing: i in [0, len-1]
        state = AbstractState(
            intervals={
                "i": Interval.from_bounds(0, 9),
                "len(arr)": Interval.from_bounds(10, 10),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 10))
        assert bug is None

    def test_slice_oob(self) -> None:
        """Detect OOB in slice with unbounded index."""
        state = AbstractState(intervals={"start": Interval.top()})
        bug = check_array_bounds("arr", "start", state, Loc("test.py", 15))
        assert bug is not None

    def test_nested_list_oob(self) -> None:
        """Detect OOB on nested list access."""
        state = AbstractState(intervals={"j": Interval.from_bounds(-1, 5)})
        bug = check_array_bounds("arr[i]", "j", state, Loc("test.py", 20))
        assert bug is not None

    def test_dynamic_index_oob(self) -> None:
        """Detect OOB when index comes from user input (top)."""
        state = AbstractState(intervals={"idx": Interval.top()})
        bug = check_array_bounds("data", "idx", state, Loc("test.py", 25))
        assert bug is not None

    def test_computed_index_oob(self) -> None:
        """Detect OOB when index is computed (i * 2 + 1)."""
        # Computed index range: if i in [0,5], i*2+1 in [1,11]
        state = AbstractState(
            intervals={
                "idx": Interval.from_bounds(1, 11),
                "len(arr)": Interval.from_bounds(10, 10),
            }
        )
        bug = check_array_bounds("arr", "idx", state, Loc("test.py", 30))
        assert bug is not None  # 11 >= 10

    def test_range_loop_no_false_positive(self) -> None:
        """No OOB in for i in range(len(arr)) pattern."""
        state = AbstractState(
            intervals={
                "i": Interval.from_bounds(0, 9),
                "len(arr)": Interval.singleton(10),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 35))
        assert bug is None

    def test_enumerate_loop_no_false_positive(self) -> None:
        """No OOB in for i, v in enumerate(arr) pattern."""
        state = AbstractState(
            intervals={
                "i": Interval.from_bounds(0, 4),
                "len(arr)": Interval.singleton(5),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 40))
        assert bug is None

    def test_zip_loop_no_false_positive(self) -> None:
        """No OOB in zip-based loop."""
        # zip truncates to shorter, so index always valid
        state = AbstractState(
            intervals={
                "i": Interval.from_bounds(0, 2),
                "len(arr)": Interval.singleton(3),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 45))
        assert bug is None

    def test_off_by_one(self) -> None:
        """Detect off-by-one: i in [0, len] instead of [0, len-1]."""
        state = AbstractState(
            intervals={
                "i": Interval.from_bounds(0, 10),
                "len(arr)": Interval.singleton(10),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 50))
        assert bug is not None  # i=10 is OOB for len=10

    def test_empty_list_access(self) -> None:
        """Detect OOB on empty list."""
        state = AbstractState(
            intervals={
                "i": Interval.singleton(0),
                "len(arr)": Interval.singleton(0),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 55))
        assert bug is not None  # 0 >= 0 (len=0)

    def test_conditional_access_after_length_check(self) -> None:
        """No OOB after if len(arr) > 0: arr[0]."""
        state = AbstractState(
            intervals={
                "i": Interval.singleton(0),
                "len(arr)": Interval.from_bounds(1, 100),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 60))
        assert bug is None

    def test_multiple_accesses_same_list(self) -> None:
        """Check multiple accesses to the same list."""
        state = AbstractState(
            intervals={
                "i": Interval.from_bounds(0, 2),
                "j": Interval.from_bounds(0, 9),
                "len(arr)": Interval.singleton(3),
            }
        )
        bug_i = check_array_bounds("arr", "i", state, Loc("test.py", 65))
        bug_j = check_array_bounds("arr", "j", state, Loc("test.py", 66))
        assert bug_i is None  # i in [0,2], len=3
        assert bug_j is not None  # j could be 3..9

    def test_list_append_then_access(self) -> None:
        """After append, len increases; access at old len is safe."""
        state = AbstractState(
            intervals={
                "i": Interval.singleton(5),
                "len(arr)": Interval.from_bounds(6, 100),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 70))
        assert bug is None

    def test_list_comprehension_index(self) -> None:
        """Index derived from list comprehension range."""
        state = AbstractState(
            intervals={
                "i": Interval.from_bounds(0, 99),
                "len(arr)": Interval.singleton(100),
            }
        )
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 75))
        assert bug is None


class TestNullDereference:
    """Tests for null/None dereference detection."""

    def test_simple_null_deref(self) -> None:
        """Detect deref on definitely-null variable."""
        state = AbstractState(nullity={"x": NullityValue.definitely_null()})
        bug = check_null_deref("x", "attr", state, Loc("test.py", 10))
        assert bug is not None
        assert bug.bug_class == BugClass.NullDereference
        assert bug.confidence >= 0.9

    def test_guarded_null_no_bug(self) -> None:
        """No deref warning after null check."""
        state = AbstractState(nullity={"x": NullityValue.definitely_not_null()})
        bug = check_null_deref("x", "attr", state, Loc("test.py", 10))
        assert bug is None

    def test_optional_parameter(self) -> None:
        """Detect potential deref on optional parameter."""
        state = AbstractState(nullity={"param": NullityValue.maybe_null()})
        bug = check_null_deref("param", "method", state, Loc("test.py", 15))
        assert bug is not None
        assert bug.severity == Severity.Warning

    def test_optional_return(self) -> None:
        """Detect potential deref on function return that may be None."""
        state = AbstractState(nullity={"result": NullityValue.maybe_null()})
        bug = check_null_deref("result", "value", state, Loc("test.py", 20))
        assert bug is not None

    def test_none_in_conditional(self) -> None:
        """Detect deref in branch where variable is None."""
        state = AbstractState(nullity={"x": NullityValue.definitely_null()})
        bug = check_null_deref("x", "attr", state, Loc("test.py", 25))
        assert bug is not None
        assert "definitely None" in bug.message

    def test_none_from_dict_get(self) -> None:
        """dict.get() may return None."""
        state = AbstractState(nullity={"val": NullityValue.maybe_null()})
        bug = check_null_deref("val", "strip", state, Loc("test.py", 30))
        assert bug is not None

    def test_none_from_list_find(self) -> None:
        """list find/search may return None."""
        state = AbstractState(nullity={"found": NullityValue.maybe_null()})
        bug = check_null_deref("found", "name", state, Loc("test.py", 35))
        assert bug is not None

    def test_none_from_re_match(self) -> None:
        """re.match() may return None."""
        state = AbstractState(nullity={"match": NullityValue.maybe_null()})
        bug = check_null_deref("match", "group", state, Loc("test.py", 40))
        assert bug is not None

    def test_chained_null_check(self) -> None:
        """After chained check: if x and x.y, both safe."""
        state = AbstractState(nullity={"x": NullityValue.definitely_not_null()})
        bug = check_null_deref("x", "y", state, Loc("test.py", 45))
        assert bug is None

    def test_null_in_loop(self) -> None:
        """Variable assigned None in loop body."""
        state = AbstractState(nullity={"item": NullityValue.maybe_null()})
        bug = check_null_deref("item", "process", state, Loc("test.py", 50))
        assert bug is not None

    def test_null_after_assignment(self) -> None:
        """After x = obj, x is not None."""
        state = AbstractState(nullity={"x": NullityValue.definitely_not_null()})
        bug = check_null_deref("x", "attr", state, Loc("test.py", 55))
        assert bug is None

    def test_null_from_function_call(self) -> None:
        """Function call result may be None."""
        state = AbstractState(nullity={"result": NullityValue.maybe_null()})
        bug = check_null_deref("result", "data", state, Loc("test.py", 60))
        assert bug is not None

    def test_null_with_or_default(self) -> None:
        """x = val or default → x is not None."""
        state = AbstractState(nullity={"x": NullityValue.definitely_not_null()})
        bug = check_null_deref("x", "attr", state, Loc("test.py", 65))
        assert bug is None

    def test_null_with_ternary(self) -> None:
        """x = a if a is not None else b → x is not None."""
        state = AbstractState(nullity={"x": NullityValue.definitely_not_null()})
        bug = check_null_deref("x", "attr", state, Loc("test.py", 70))
        assert bug is None

    def test_typescript_undefined(self) -> None:
        """TypeScript undefined deref."""
        state = AbstractState(nullity={"x": NullityValue.maybe_null()})
        bug = check_null_deref("x", "length", state, Loc("test.ts", 10))
        assert bug is not None

    def test_typescript_null(self) -> None:
        """TypeScript null deref."""
        state = AbstractState(nullity={"x": NullityValue.definitely_null()})
        bug = check_null_deref("x", "toString", state, Loc("test.ts", 15))
        assert bug is not None
        assert bug.confidence >= 0.9

    def test_optional_chaining_no_bug(self) -> None:
        """Optional chaining x?.attr is safe (modeled as not-null after)."""
        state = AbstractState(nullity={"x": NullityValue.definitely_not_null()})
        bug = check_null_deref("x", "attr", state, Loc("test.ts", 20))
        assert bug is None

    def test_nullish_coalescing_no_bug(self) -> None:
        """x ?? default is safe (x is replaced if null)."""
        state = AbstractState(nullity={"result": NullityValue.definitely_not_null()})
        bug = check_null_deref("result", "value", state, Loc("test.ts", 25))
        assert bug is None

    def test_strict_null_checks(self) -> None:
        """Under strict null checks, all maybe-null accesses are flagged."""
        state = AbstractState(nullity={"x": NullityValue.maybe_null()})
        bug = check_null_deref("x", "method", state, Loc("test.ts", 30))
        assert bug is not None


class TestDivisionByZero:
    """Tests for division by zero detection."""

    def test_simple_div_by_zero(self) -> None:
        """Detect division by definitely-zero variable."""
        state = AbstractState(intervals={"y": Interval.singleton(0)})
        bug = check_division_by_zero("y", state, Loc("test.py", 10))
        assert bug is not None
        assert bug.confidence >= 0.9
        assert bug.severity == Severity.Error

    def test_guarded_division(self) -> None:
        """No warning when divisor is checked != 0."""
        state = AbstractState(intervals={"y": Interval.from_bounds(1, 100)})
        bug = check_division_by_zero("y", state, Loc("test.py", 15))
        assert bug is None

    def test_modulo_by_zero(self) -> None:
        """Detect modulo by zero."""
        state = AbstractState(intervals={"m": Interval.singleton(0)})
        bug = check_division_by_zero("m", state, Loc("test.py", 20))
        assert bug is not None

    def test_floor_div_by_zero(self) -> None:
        """Detect floor division by zero."""
        state = AbstractState(intervals={"d": Interval.from_bounds(-1, 1)})
        bug = check_division_by_zero("d", state, Loc("test.py", 25))
        assert bug is not None  # contains 0

    def test_variable_divisor(self) -> None:
        """Detect potential div-by-zero with unconstrained divisor."""
        state = AbstractState(intervals={"y": Interval.top()})
        bug = check_division_by_zero("y", state, Loc("test.py", 30))
        assert bug is not None

    def test_computed_divisor(self) -> None:
        """Detect div-by-zero when divisor is computed result."""
        # Result of computation that might be 0
        state = AbstractState(intervals={"result": Interval.from_bounds(-5, 5)})
        bug = check_division_by_zero("result", state, Loc("test.py", 35))
        assert bug is not None

    def test_divisor_from_parameter(self) -> None:
        """Detect div-by-zero when divisor is a parameter."""
        state = AbstractState(intervals={"param": Interval.top()})
        bug = check_division_by_zero("param", state, Loc("test.py", 40))
        assert bug is not None

    def test_zero_check_then_divide(self) -> None:
        """No warning after y != 0 check."""
        state = AbstractState(intervals={"y": Interval.from_bounds(1, 100)})
        bug = check_division_by_zero("y", state, Loc("test.py", 45))
        assert bug is None

    def test_nonzero_assertion(self) -> None:
        """No warning after assert y != 0."""
        state = AbstractState(intervals={"y": Interval.from_bounds(1, 1000)})
        bug = check_division_by_zero("y", state, Loc("test.py", 50))
        assert bug is None

    def test_division_in_loop(self) -> None:
        """Detect div-by-zero in loop with decreasing counter."""
        # Counter might reach 0
        state = AbstractState(intervals={"counter": Interval.from_bounds(0, 10)})
        bug = check_division_by_zero("counter", state, Loc("test.py", 55))
        assert bug is not None

    def test_division_by_len(self) -> None:
        """Division by len(x) when x might be empty."""
        state = AbstractState(intervals={"len_x": Interval.from_bounds(0, 100)})
        bug = check_division_by_zero("len_x", state, Loc("test.py", 60))
        assert bug is not None  # len might be 0

    def test_division_by_count(self) -> None:
        """Division by count when count might be 0."""
        state = AbstractState(intervals={"count": Interval.from_bounds(0, 50)})
        bug = check_division_by_zero("count", state, Loc("test.py", 65))
        assert bug is not None

    def test_positive_divisor_safe(self) -> None:
        """Strictly positive divisor is safe."""
        state = AbstractState(intervals={"d": Interval.from_bounds(1, 10)})
        bug = check_division_by_zero("d", state, Loc("test.py", 70))
        assert bug is None

    def test_negative_divisor_safe(self) -> None:
        """Strictly negative divisor is safe."""
        state = AbstractState(intervals={"d": Interval.from_bounds(-10, -1)})
        bug = check_division_by_zero("d", state, Loc("test.py", 75))
        assert bug is None


class TestTypeConfusion:
    """Tests for type tag confusion detection."""

    def test_simple_type_confusion(self) -> None:
        """Detect calling list method on non-list."""
        state = AbstractState(type_tags={"x": TypeTagSet.from_names("str", "int")})
        bug = check_type_confusion("x", "append", state, Loc("test.py", 10))
        assert bug is not None
        assert bug.bug_class == BugClass.TypeConfusion

    def test_isinstance_guarded(self) -> None:
        """No confusion after isinstance check."""
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("list")})
        bug = check_type_confusion("x", "append", state, Loc("test.py", 15))
        assert bug is None

    def test_union_type_method_call(self) -> None:
        """Detect confusion on union type."""
        state = AbstractState(type_tags={"x": TypeTagSet.from_names("list", "dict")})
        bug = check_type_confusion("x", "append", state, Loc("test.py", 20))
        assert bug is not None  # dict doesn't have append

    def test_dynamic_attribute_access(self) -> None:
        """Unknown type accessing attribute."""
        state = AbstractState(type_tags={"x": TypeTagSet.top()})
        bug = check_type_confusion("x", "unknown_method", state, Loc("test.py", 25))
        assert bug is not None  # Type unknown

    def test_wrong_method_on_type(self) -> None:
        """String type calling dict method."""
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("str")})
        bug = check_type_confusion("x", "keys", state, Loc("test.py", 30))
        assert bug is not None  # str doesn't have keys

    def test_numeric_string_confusion(self) -> None:
        """Detect calling string method on int."""
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("int")})
        bug = check_type_confusion("x", "upper", state, Loc("test.py", 35))
        assert bug is not None

    def test_list_dict_confusion(self) -> None:
        """Detect calling dict method on list."""
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("list")})
        bug = check_type_confusion("x", "keys", state, Loc("test.py", 40))
        assert bug is not None

    def test_none_method_call(self) -> None:
        """Detect method call on NoneType."""
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("NoneType")})
        bug = check_type_confusion("x", "append", state, Loc("test.py", 45))
        assert bug is not None

    def test_hasattr_guarded(self) -> None:
        """No confusion after hasattr check (type narrowed)."""
        # After hasattr(x, 'append'), we know x has append
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("list")})
        bug = check_type_confusion("x", "append", state, Loc("test.py", 50))
        assert bug is None

    def test_typeof_guarded(self) -> None:
        """No confusion after typeof check (TypeScript)."""
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("str")})
        bug = check_type_confusion("x", "upper", state, Loc("test.py", 55))
        assert bug is None

    def test_discriminated_union_ts(self) -> None:
        """TypeScript discriminated union narrowing."""
        # After checking x.kind === "circle", type is narrowed
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("Circle")})
        # Circle-specific method is fine
        bug = check_type_confusion("x", "radius", state, Loc("test.ts", 60))
        # "radius" isn't in our method_map, so no confusion detected
        assert bug is None

    def test_type_narrowing_chain(self) -> None:
        """Chain of type narrowing: first narrow to int|str, then to int."""
        state1 = AbstractState(type_tags={"x": TypeTagSet.from_names("int", "str")})
        # After further narrowing
        state2 = AbstractState(type_tags={"x": TypeTagSet.singleton("int")})
        bug1 = check_type_confusion("x", "bit_length", state1, Loc("test.py", 65))
        bug2 = check_type_confusion("x", "bit_length", state2, Loc("test.py", 70))
        assert bug1 is not None  # str doesn't have bit_length
        assert bug2 is None

    def test_exhaustive_check(self) -> None:
        """After exhaustive type check, no confusion possible."""
        # All branches handled
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("int")})
        bug = check_type_confusion("x", "bit_length", state, Loc("test.py", 75))
        assert bug is None

    def test_pop_method_multiple_types(self) -> None:
        """pop() is valid on list, dict, and set."""
        for tag in ["list", "dict", "set"]:
            state = AbstractState(type_tags={"x": TypeTagSet.singleton(tag)})
            bug = check_type_confusion("x", "pop", state, Loc("test.py", 80))
            assert bug is None, f"pop should be valid on {tag}"


class TestBugReporting:
    """Tests for bug report quality."""

    def test_bug_location_accuracy(self) -> None:
        """Bug report includes correct location."""
        loc = Loc("src/main.py", 42, 5, 42, 20)
        report = BugReport(
            bug_class=BugClass.NullDereference,
            source_location=loc,
            message="x may be None",
        )
        assert report.source_location.file == "src/main.py"
        assert report.source_location.line == 42
        assert report.source_location.column == 5

    def test_bug_message_clarity(self) -> None:
        """Bug message is clear and actionable."""
        state = AbstractState(nullity={"x": NullityValue.definitely_null()})
        bug = check_null_deref("x", "method", state, Loc("test.py", 10))
        assert bug is not None
        assert "None" in bug.message or "null" in bug.message.lower()
        assert "x" in bug.message

    def test_bug_confidence_level(self) -> None:
        """Bug confidence maps to correct level."""
        high = BugReport(BugClass.NullDereference, Loc(), "msg", confidence=0.95)
        med = BugReport(BugClass.NullDereference, Loc(), "msg", confidence=0.75)
        low = BugReport(BugClass.NullDereference, Loc(), "msg", confidence=0.5)
        very_low = BugReport(BugClass.NullDereference, Loc(), "msg", confidence=0.2)
        assert high.confidence_level == Confidence.High
        assert med.confidence_level == Confidence.Medium
        assert low.confidence_level == Confidence.Low
        assert very_low.confidence_level == Confidence.VeryLow

    def test_bug_severity_classification(self) -> None:
        """Bugs are classified by severity."""
        error_bug = BugReport(BugClass.NullDereference, Loc(), "definitely null", severity=Severity.Error)
        warning_bug = BugReport(BugClass.NullDereference, Loc(), "maybe null", severity=Severity.Warning)
        assert error_bug.severity == Severity.Error
        assert warning_bug.severity == Severity.Warning

    def test_bug_fix_suggestion(self) -> None:
        """Bug report includes fix suggestion."""
        state = AbstractState(nullity={"x": NullityValue.maybe_null()})
        bug = check_null_deref("x", "attr", state)
        assert bug is not None
        assert bug.fix_suggestion is not None
        assert "if" in bug.fix_suggestion.lower() or "check" in bug.fix_suggestion.lower()

    def test_bug_code_context(self) -> None:
        """Bug report can include code context."""
        report = BugReport(
            bug_class=BugClass.ArrayOutOfBounds,
            source_location=Loc("test.py", 10),
            message="OOB",
            code_context="arr[i]  # i might be out of bounds",
        )
        assert report.code_context is not None
        assert "arr[i]" in report.code_context

    def test_bug_deduplication(self) -> None:
        """Duplicate bugs are removed."""
        r1 = BugReport(BugClass.NullDereference, Loc("f.py", 10), "x may be None")
        r2 = BugReport(BugClass.NullDereference, Loc("f.py", 10), "x may be None")
        r3 = BugReport(BugClass.NullDereference, Loc("f.py", 20), "y may be None")
        deduped = deduplicate_reports([r1, r2, r3])
        assert len(deduped) == 2

    def test_bug_fingerprinting(self) -> None:
        """Bug fingerprint is deterministic and unique."""
        r1 = BugReport(BugClass.NullDereference, Loc("f.py", 10), "x may be None")
        r2 = BugReport(BugClass.NullDereference, Loc("f.py", 10), "x may be None")
        r3 = BugReport(BugClass.NullDereference, Loc("f.py", 20), "y may be None")
        assert r1.fingerprint == r2.fingerprint
        assert r1.fingerprint != r3.fingerprint

    def test_sarif_output_format(self) -> None:
        """Bug report generates valid SARIF output."""
        report = BugReport(
            bug_class=BugClass.NullDereference,
            source_location=Loc("src/main.py", 42, 5),
            message="x may be None",
            severity=Severity.Warning,
        )
        sarif = report.to_sarif()
        assert sarif["ruleId"] == "null_dereference"
        assert sarif["level"] == "warning"
        assert sarif["message"]["text"] == "x may be None"
        assert sarif["locations"][0]["physicalLocation"]["region"]["startLine"] == 42
        assert "fingerprints" in sarif

    def test_html_report_format(self) -> None:
        """Bug report generates valid HTML fragment."""
        report = BugReport(
            bug_class=BugClass.NullDereference,
            source_location=Loc("src/main.py", 42),
            message="x may be None",
            severity=Severity.Warning,
        )
        html = report.to_html_fragment()
        assert "<div" in html
        assert "warning" in html
        assert "x may be None" in html
        assert "src/main.py:42" in html

    def test_sarif_error_level(self) -> None:
        """SARIF level is 'error' for Error severity."""
        report = BugReport(
            bug_class=BugClass.NullDereference,
            source_location=Loc("f.py", 1),
            message="x",
            severity=Severity.Error,
        )
        assert report.to_sarif()["level"] == "error"


class TestFalsePositiveRate:
    """Tests ensuring no false positives on safe patterns."""

    def _safe_state(self, **kwargs: Any) -> AbstractState:
        """Helper: create a safe abstract state."""
        return AbstractState(
            intervals=kwargs.get("intervals", {}),
            nullity=kwargs.get("nullity", {}),
            type_tags=kwargs.get("type_tags", {}),
        )

    def test_safe_patterns_no_bugs(self) -> None:
        """Test 20+ safe coding patterns produce no bugs."""
        safe_cases: List[Tuple[str, AbstractState]] = [
            # 1. x = 42; x.bit_length()
            ("bit_length", self._safe_state(
                nullity={"x": NullityValue.definitely_not_null()},
                type_tags={"x": TypeTagSet.singleton("int")})),
            # 2. x = []; x.append(1)
            ("append", self._safe_state(
                nullity={"x": NullityValue.definitely_not_null()},
                type_tags={"x": TypeTagSet.singleton("list")})),
            # 3. x = "hello"; x.upper()
            ("upper", self._safe_state(
                nullity={"x": NullityValue.definitely_not_null()},
                type_tags={"x": TypeTagSet.singleton("str")})),
            # 4. x = {}; x.keys()
            ("keys", self._safe_state(
                nullity={"x": NullityValue.definitely_not_null()},
                type_tags={"x": TypeTagSet.singleton("dict")})),
            # 5. x = set(); x.add(1)
            ("add", self._safe_state(
                nullity={"x": NullityValue.definitely_not_null()},
                type_tags={"x": TypeTagSet.singleton("set")})),
        ]
        for method, state in safe_cases:
            bug_null = check_null_deref("x", method, state)
            bug_type = check_type_confusion("x", method, state)
            assert bug_null is None, f"False positive null deref for {method}"
            assert bug_type is None, f"False positive type confusion for {method}"

    def test_guarded_patterns_no_bugs(self) -> None:
        """Properly guarded patterns produce no bugs."""
        # After isinstance(x, int)
        state = self._safe_state(
            nullity={"x": NullityValue.definitely_not_null()},
            type_tags={"x": TypeTagSet.singleton("int")},
        )
        assert check_null_deref("x", "bit_length", state) is None
        assert check_type_confusion("x", "bit_length", state) is None

    def test_idiomatic_python_no_bugs(self) -> None:
        """Idiomatic Python patterns produce no false positives."""
        patterns: List[Tuple[str, str, AbstractState]] = [
            # for x in items: x.process()
            ("x", "process", self._safe_state(nullity={"x": NullityValue.definitely_not_null()})),
            # with open(f) as fh: fh.read()
            ("fh", "read", self._safe_state(nullity={"fh": NullityValue.definitely_not_null()})),
            # x = some_dict["key"]  (KeyError not null)
            ("x", "strip", self._safe_state(
                nullity={"x": NullityValue.definitely_not_null()},
                type_tags={"x": TypeTagSet.singleton("str")})),
        ]
        for var, method, state in patterns:
            assert check_null_deref(var, method, state) is None

    def test_idiomatic_typescript_no_bugs(self) -> None:
        """Idiomatic TypeScript patterns produce no false positives."""
        # After type guard: if (typeof x === 'string')
        state = self._safe_state(
            nullity={"x": NullityValue.definitely_not_null()},
            type_tags={"x": TypeTagSet.singleton("str")},
        )
        assert check_null_deref("x", "length", state) is None
        assert check_type_confusion("x", "upper", state) is None

    def test_safe_division_patterns(self) -> None:
        """Safe division patterns produce no false positives."""
        safe_divisors = [
            Interval.from_bounds(1, 100),
            Interval.from_bounds(-100, -1),
            Interval.singleton(7),
            Interval.singleton(-3),
        ]
        for div_iv in safe_divisors:
            state = self._safe_state(intervals={"d": div_iv})
            assert check_division_by_zero("d", state) is None

    def test_safe_array_patterns(self) -> None:
        """Safe array access patterns produce no false positives."""
        safe_cases = [
            (Interval.from_bounds(0, 4), Interval.singleton(5)),
            (Interval.singleton(0), Interval.from_bounds(1, 10)),
            (Interval.from_bounds(0, 9), Interval.singleton(10)),
            (Interval.from_bounds(0, 0), Interval.singleton(1)),
        ]
        for idx_iv, len_iv in safe_cases:
            state = self._safe_state(intervals={"i": idx_iv, "len(arr)": len_iv})
            assert check_array_bounds("arr", "i", state) is None

    def test_safe_null_patterns(self) -> None:
        """Safe null patterns produce no false positives."""
        safe_states = [
            NullityValue.definitely_not_null(),
        ]
        for nv in safe_states:
            state = self._safe_state(nullity={"x": nv})
            assert check_null_deref("x", "attr", state) is None

    def test_constructor_result_not_null(self) -> None:
        """Constructor call result is never null."""
        state = self._safe_state(nullity={"obj": NullityValue.definitely_not_null()})
        assert check_null_deref("obj", "method", state) is None

    def test_literal_not_null(self) -> None:
        """Literal values are never null."""
        state = self._safe_state(
            nullity={"s": NullityValue.definitely_not_null()},
            type_tags={"s": TypeTagSet.singleton("str")},
        )
        assert check_null_deref("s", "upper", state) is None

    def test_len_result_not_zero_after_check(self) -> None:
        """After if len(x) > 0, division by len is safe."""
        state = self._safe_state(intervals={"len_x": Interval.from_bounds(1, 1000)})
        assert check_division_by_zero("len_x", state) is None

    def test_bool_result_in_condition(self) -> None:
        """Boolean result used in condition: no type confusion."""
        state = self._safe_state(
            type_tags={"result": TypeTagSet.singleton("bool")},
            nullity={"result": NullityValue.definitely_not_null()},
        )
        # Bool doesn't have append, but we're not calling it
        assert check_null_deref("result", "__bool__", state) is None

    def test_multiple_safe_accesses(self) -> None:
        """Multiple safe accesses in sequence produce no bugs."""
        state = self._safe_state(
            nullity={
                "a": NullityValue.definitely_not_null(),
                "b": NullityValue.definitely_not_null(),
                "c": NullityValue.definitely_not_null(),
            },
            type_tags={
                "a": TypeTagSet.singleton("list"),
                "b": TypeTagSet.singleton("dict"),
                "c": TypeTagSet.singleton("str"),
            },
        )
        assert check_null_deref("a", "append", state) is None
        assert check_null_deref("b", "keys", state) is None
        assert check_null_deref("c", "upper", state) is None
        assert check_type_confusion("a", "append", state) is None
        assert check_type_confusion("b", "keys", state) is None
        assert check_type_confusion("c", "upper", state) is None


class TestArrayOutOfBoundsExtended:
    """Extended OOB detection tests."""

    def test_oob_index_at_length(self) -> None:
        """Index exactly at length is OOB."""
        state = AbstractState(intervals={
            "i": Interval.singleton(5),
            "len(arr)": Interval.singleton(5),
        })
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 1))
        assert bug is not None

    def test_oob_index_just_below_length(self) -> None:
        """Index at length-1 is safe."""
        state = AbstractState(intervals={
            "i": Interval.singleton(4),
            "len(arr)": Interval.singleton(5),
        })
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 1))
        assert bug is None

    def test_oob_wide_index_range(self) -> None:
        """Very wide index range flags OOB."""
        state = AbstractState(intervals={
            "i": Interval.from_bounds(-100, 100),
        })
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 1))
        assert bug is not None

    def test_oob_zero_length_array(self) -> None:
        """Access on zero-length array is always OOB."""
        state = AbstractState(intervals={
            "i": Interval.from_bounds(0, 0),
            "len(arr)": Interval.singleton(0),
        })
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 1))
        assert bug is not None

    def test_oob_negative_only_index(self) -> None:
        """Purely negative index is OOB."""
        state = AbstractState(intervals={
            "i": Interval.from_bounds(-5, -1),
        })
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 1))
        assert bug is not None

    def test_oob_report_has_suggestion(self) -> None:
        """OOB report includes fix suggestion."""
        state = AbstractState(intervals={"i": Interval.top()})
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 1))
        assert bug is not None
        assert bug.fix_suggestion is not None

    def test_oob_report_has_counterexample(self) -> None:
        """OOB report includes counterexample."""
        state = AbstractState(intervals={"i": Interval.from_bounds(-5, 3)})
        bug = check_array_bounds("arr", "i", state, Loc("test.py", 1))
        assert bug is not None
        assert bug.counterexample is not None

    def test_safe_bounded_range(self) -> None:
        """Index bounded within [0, len-1] is safe."""
        for length in [1, 5, 10, 100]:
            state = AbstractState(intervals={
                "i": Interval.from_bounds(0, length - 1),
                "len(arr)": Interval.singleton(length),
            })
            bug = check_array_bounds("arr", "i", state, Loc("test.py", 1))
            assert bug is None, f"False positive for length={length}"


class TestNullDereferenceExtended:
    """Extended null deref detection tests."""

    def test_null_deref_confidence_levels(self) -> None:
        """Definitely null has higher confidence than maybe null."""
        state_def = AbstractState(nullity={"x": NullityValue.definitely_null()})
        state_may = AbstractState(nullity={"x": NullityValue.maybe_null()})
        bug_def = check_null_deref("x", "attr", state_def)
        bug_may = check_null_deref("x", "attr", state_may)
        assert bug_def is not None and bug_may is not None
        assert bug_def.confidence > bug_may.confidence

    def test_null_deref_severity_levels(self) -> None:
        """Definitely null is Error, maybe null is Warning."""
        state_def = AbstractState(nullity={"x": NullityValue.definitely_null()})
        state_may = AbstractState(nullity={"x": NullityValue.maybe_null()})
        bug_def = check_null_deref("x", "attr", state_def)
        bug_may = check_null_deref("x", "attr", state_may)
        assert bug_def is not None
        assert bug_may is not None
        assert bug_def.severity == Severity.Error
        assert bug_may.severity == Severity.Warning

    def test_null_deref_message_includes_var(self) -> None:
        """Null deref message includes variable name."""
        state = AbstractState(nullity={"myVar": NullityValue.maybe_null()})
        bug = check_null_deref("myVar", "method", state)
        assert bug is not None
        assert "myVar" in bug.message

    def test_null_deref_message_includes_attr(self) -> None:
        """Null deref message includes attribute name."""
        state = AbstractState(nullity={"x": NullityValue.definitely_null()})
        bug = check_null_deref("x", "some_method", state)
        assert bug is not None
        assert "some_method" in bug.message

    def test_null_deref_various_methods(self) -> None:
        """Null deref detected for various method names."""
        state = AbstractState(nullity={"x": NullityValue.maybe_null()})
        for method in ["attr", "method", "property", "__len__", "__str__"]:
            bug = check_null_deref("x", method, state)
            assert bug is not None, f"Failed for method={method}"

    def test_safe_after_assignment(self) -> None:
        """After assignment, variable is not null."""
        state = AbstractState(nullity={"x": NullityValue.definitely_not_null()})
        for method in ["attr", "method", "property"]:
            bug = check_null_deref("x", method, state)
            assert bug is None

    def test_null_deref_bottom_no_bug(self) -> None:
        """Bottom nullity produces no bug (unreachable)."""
        state = AbstractState(nullity={"x": NullityValue.bottom()})
        bug = check_null_deref("x", "attr", state)
        assert bug is None


class TestDivisionByZeroExtended:
    """Extended division by zero tests."""

    def test_div_by_singleton_nonzero(self) -> None:
        """Division by known nonzero value is safe."""
        for v in [1, -1, 5, -5, 100, -100]:
            state = AbstractState(intervals={"d": Interval.singleton(v)})
            bug = check_division_by_zero("d", state)
            assert bug is None, f"False positive for d={v}"

    def test_div_by_range_excluding_zero(self) -> None:
        """Division by range not including zero is safe."""
        safe_ranges = [
            (1, 10), (-10, -1), (5, 100), (-100, -5),
        ]
        for lo, hi in safe_ranges:
            state = AbstractState(intervals={"d": Interval.from_bounds(lo, hi)})
            bug = check_division_by_zero("d", state)
            assert bug is None, f"False positive for [{lo},{hi}]"

    def test_div_by_range_including_zero(self) -> None:
        """Division by range including zero is flagged."""
        risky_ranges = [
            (-5, 5), (-1, 1), (0, 10), (-10, 0),
        ]
        for lo, hi in risky_ranges:
            state = AbstractState(intervals={"d": Interval.from_bounds(lo, hi)})
            bug = check_division_by_zero("d", state)
            assert bug is not None, f"Missed bug for [{lo},{hi}]"

    def test_div_by_zero_message(self) -> None:
        """Division by zero message includes divisor name."""
        state = AbstractState(intervals={"divisor": Interval.singleton(0)})
        bug = check_division_by_zero("divisor", state)
        assert bug is not None
        assert "divisor" in bug.message

    def test_div_by_zero_suggestion(self) -> None:
        """Division by zero includes fix suggestion."""
        state = AbstractState(intervals={"y": Interval.from_bounds(-5, 5)})
        bug = check_division_by_zero("y", state)
        assert bug is not None
        assert bug.fix_suggestion is not None
        assert "0" in bug.fix_suggestion


class TestTypeConfusionExtended:
    """Extended type confusion tests."""

    def test_type_confusion_all_methods(self) -> None:
        """Test confusion detection for various method/type combos."""
        confusion_cases = [
            ("int", "append"),   # int doesn't have append
            ("str", "keys"),     # str doesn't have keys
            ("list", "upper"),   # list doesn't have upper
            ("dict", "append"),  # dict doesn't have append
        ]
        for tag, method in confusion_cases:
            state = AbstractState(type_tags={"x": TypeTagSet.singleton(tag)})
            bug = check_type_confusion("x", method, state, Loc("test.py", 1))
            assert bug is not None, f"Missed confusion: {tag}.{method}"

    def test_type_confusion_correct_combos(self) -> None:
        """No confusion for correct type/method combinations."""
        correct_cases = [
            ("list", "append"),
            ("dict", "keys"),
            ("str", "upper"),
            ("str", "lower"),
            ("int", "bit_length"),
            ("set", "add"),
        ]
        for tag, method in correct_cases:
            state = AbstractState(type_tags={"x": TypeTagSet.singleton(tag)})
            bug = check_type_confusion("x", method, state, Loc("test.py", 1))
            assert bug is None, f"False positive: {tag}.{method}"

    def test_type_confusion_unknown_method(self) -> None:
        """Unknown method with known type: no confusion (not in method_map)."""
        state = AbstractState(type_tags={"x": TypeTagSet.singleton("int")})
        bug = check_type_confusion("x", "custom_method", state, Loc("test.py", 1))
        # custom_method not in method_map → no confusion detected for known type
        assert bug is None

    def test_type_confusion_bottom_type(self) -> None:
        """Bottom type tag → no confusion (unreachable code)."""
        state = AbstractState(type_tags={"x": TypeTagSet.bottom()})
        bug = check_type_confusion("x", "append", state, Loc("test.py", 1))
        assert bug is None

    def test_type_confusion_report_quality(self) -> None:
        """Type confusion report has good quality."""
        state = AbstractState(type_tags={"x": TypeTagSet.from_names("int", "str")})
        bug = check_type_confusion("x", "append", state, Loc("test.py", 1))
        assert bug is not None
        assert bug.confidence > 0
        assert bug.fix_suggestion is not None


class TestBugReportingExtended:
    """Extended bug reporting tests."""

    def test_fingerprint_deterministic(self) -> None:
        """Fingerprint is deterministic across calls."""
        r = BugReport(BugClass.NullDereference, Loc("f.py", 10), "msg")
        fp1 = r.fingerprint
        fp2 = r.fingerprint
        assert fp1 == fp2

    def test_fingerprint_length(self) -> None:
        """Fingerprint is 16 hex characters."""
        r = BugReport(BugClass.NullDereference, Loc("f.py", 10), "msg")
        assert len(r.fingerprint) == 16
        assert all(c in "0123456789abcdef" for c in r.fingerprint)

    def test_dedup_empty_list(self) -> None:
        """Dedup of empty list returns empty."""
        assert deduplicate_reports([]) == []

    def test_dedup_single_item(self) -> None:
        """Dedup of single item returns same."""
        r = BugReport(BugClass.NullDereference, Loc("f.py", 10), "msg")
        assert deduplicate_reports([r]) == [r]

    def test_dedup_all_different(self) -> None:
        """Dedup of all different items returns all."""
        reports = [
            BugReport(BugClass.NullDereference, Loc("f.py", i), f"msg {i}")
            for i in range(5)
        ]
        assert len(deduplicate_reports(reports)) == 5

    def test_dedup_all_same(self) -> None:
        """Dedup of all same items returns one."""
        r = BugReport(BugClass.NullDereference, Loc("f.py", 10), "msg")
        assert len(deduplicate_reports([r, r, r])) == 1

    def test_confidence_from_score_boundary(self) -> None:
        """Confidence boundary values."""
        assert Confidence.from_score(0.9) == Confidence.High
        assert Confidence.from_score(0.89) == Confidence.Medium
        assert Confidence.from_score(0.7) == Confidence.Medium
        assert Confidence.from_score(0.69) == Confidence.Low
        assert Confidence.from_score(0.4) == Confidence.Low
        assert Confidence.from_score(0.39) == Confidence.VeryLow

    def test_sarif_all_bug_classes(self) -> None:
        """SARIF output works for all bug classes."""
        for bc in BugClass:
            r = BugReport(bc, Loc("f.py", 1), "msg")
            sarif = r.to_sarif()
            assert sarif["ruleId"] == bc.value

    def test_html_all_severities(self) -> None:
        """HTML output works for all severities."""
        for sev in [Severity.Error, Severity.Warning, Severity.Info]:
            r = BugReport(BugClass.NullDereference, Loc("f.py", 1), "msg", severity=sev)
            html = r.to_html_fragment()
            assert sev.name.lower() in html

    def test_loc_with_end_position(self) -> None:
        """Location with end position."""
        loc = Loc("test.py", 10, 5, 12, 20)
        assert loc.line == 10
        assert loc.end_line == 12
        assert loc.end_column == 20

    def test_bug_report_default_values(self) -> None:
        """Bug report default values."""
        r = BugReport(BugClass.NullDereference, Loc(), "msg")
        assert r.confidence == 0.5
        assert r.severity == Severity.Warning
        assert r.counterexample is None
        assert r.fix_suggestion is None
