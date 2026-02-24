from __future__ import annotations

import ast
import re
import enum
import hashlib
from dataclasses import dataclass, field
from typing import (
    Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union, Callable,
    Iterator, Sequence, Mapping, Type as TypingType,
)


# ---------------------------------------------------------------------------
# Local lightweight IR / abstract-state types (no cross-module imports)
# ---------------------------------------------------------------------------

@dataclass
class Location:
    """Source location."""
    file: str = "<unknown>"
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass
class AbstractType:
    """Simplified abstract type for pattern matching."""
    base: str = "unknown"
    args: List[AbstractType] = field(default_factory=list)
    nullable: bool = False
    literal_value: Any = None
    refinements: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_numeric(self) -> bool:
        return self.base in ("int", "float", "complex", "number")

    @property
    def is_string(self) -> bool:
        return self.base in ("str", "bytes", "bytearray")

    @property
    def is_sequence(self) -> bool:
        return self.base in ("list", "tuple", "str", "bytes", "bytearray", "array", "deque")

    @property
    def is_mapping(self) -> bool:
        return self.base in ("dict", "defaultdict", "OrderedDict", "Counter")

    @property
    def is_set(self) -> bool:
        return self.base in ("set", "frozenset")

    @property
    def is_callable(self) -> bool:
        return self.base in ("function", "method", "callable", "lambda", "builtin_function")

    @property
    def is_none(self) -> bool:
        return self.base == "NoneType" or (self.literal_value is None and self.base == "literal")

    @property
    def is_union(self) -> bool:
        return self.base == "Union"

    def union_members(self) -> List[AbstractType]:
        if self.is_union:
            return list(self.args)
        return [self]

    def contains_none(self) -> bool:
        if self.is_none:
            return True
        if self.nullable:
            return True
        if self.is_union:
            return any(m.contains_none() for m in self.args)
        return False


@dataclass
class IRNode:
    """Minimal IR node used by pattern matchers."""
    kind: str = ""
    name: str = ""
    location: Location = field(default_factory=Location)
    children: List[IRNode] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[IRNode] = None
    type_info: Optional[AbstractType] = None

    def child(self, idx: int) -> Optional[IRNode]:
        if 0 <= idx < len(self.children):
            return self.children[idx]
        return None

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def has_attr(self, key: str) -> bool:
        return key in self.attributes

    def iter_descendants(self) -> Iterator[IRNode]:
        for c in self.children:
            yield c
            yield from c.iter_descendants()

    def find(self, kind: str) -> List[IRNode]:
        return [n for n in self.iter_descendants() if n.kind == kind]

    def ancestor(self, kind: str) -> Optional[IRNode]:
        p = self.parent
        while p is not None:
            if p.kind == kind:
                return p
            p = p.parent
        return None


@dataclass
class AbstractValue:
    """Abstract value carrying range / constraint info."""
    type_info: AbstractType = field(default_factory=AbstractType)
    lower_bound: Optional[Union[int, float]] = None
    upper_bound: Optional[Union[int, float]] = None
    possible_values: Optional[FrozenSet[Any]] = None
    is_sorted: Optional[bool] = None
    length_lower: Optional[int] = None
    length_upper: Optional[int] = None
    is_closed: Optional[bool] = None
    is_locked: Optional[bool] = None
    is_constant: bool = False
    constant_value: Any = None


@dataclass
class AbstractState:
    """Abstract state mapping names to abstract values."""
    bindings: Dict[str, AbstractValue] = field(default_factory=dict)
    reachable: bool = True
    path_conditions: List[str] = field(default_factory=list)
    loop_depth: int = 0
    in_try: bool = False
    in_except: bool = False
    in_finally: bool = False
    in_with: bool = False
    in_async: bool = False
    current_function: Optional[str] = None
    locks_held: List[str] = field(default_factory=list)

    def get(self, name: str) -> Optional[AbstractValue]:
        return self.bindings.get(name)

    def get_type(self, name: str) -> Optional[AbstractType]:
        v = self.bindings.get(name)
        return v.type_info if v else None


# ---------------------------------------------------------------------------
# Severity & Confidence
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"

    def __lt__(self, other: Severity) -> bool:
        order = {Severity.HINT: 0, Severity.INFO: 1, Severity.WARNING: 2, Severity.ERROR: 3}
        return order[self] < order[other]


class Confidence(enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# PatternMatch
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """Result of a pattern match."""
    location: Location
    pattern_name: str
    confidence: Confidence
    severity: Severity
    message: str
    fix_suggestion: Optional[str] = None
    category: str = ""
    related_locations: List[Location] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        raw = f"{self.pattern_name}:{self.location}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# PatternMatcher (base)
# ---------------------------------------------------------------------------

class PatternMatcher:
    """Base class for pattern matchers."""

    name: str = "base"
    category: str = "general"
    description: str = ""
    severity: Severity = Severity.WARNING
    enabled: bool = True

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        raise NotImplementedError

    def _make_match(
        self,
        node: IRNode,
        pattern_name: str,
        message: str,
        confidence: Confidence = Confidence.MEDIUM,
        severity: Optional[Severity] = None,
        fix_suggestion: Optional[str] = None,
        related: Optional[List[Location]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> PatternMatch:
        return PatternMatch(
            location=node.location,
            pattern_name=pattern_name,
            confidence=confidence,
            severity=severity or self.severity,
            message=message,
            fix_suggestion=fix_suggestion,
            category=self.category,
            related_locations=related or [],
            metadata=meta or {},
        )

    def _is_name(self, node: IRNode, name: str) -> bool:
        return node.kind == "Name" and node.name == name

    def _is_call(self, node: IRNode, func_name: Optional[str] = None) -> bool:
        if node.kind != "Call":
            return False
        if func_name is None:
            return True
        fn = node.child(0)
        if fn is None:
            return False
        if fn.kind == "Name" and fn.name == func_name:
            return True
        if fn.kind == "Attribute" and fn.attr("attr") == func_name:
            return True
        return False

    def _is_attribute(self, node: IRNode, attr_name: Optional[str] = None) -> bool:
        if node.kind != "Attribute":
            return False
        if attr_name is None:
            return True
        return node.attr("attr") == attr_name

    def _get_call_name(self, node: IRNode) -> Optional[str]:
        if node.kind != "Call":
            return None
        fn = node.child(0)
        if fn is None:
            return None
        if fn.kind == "Name":
            return fn.name
        if fn.kind == "Attribute":
            return fn.attr("attr")
        return None

    def _node_is_literal(self, node: IRNode) -> bool:
        return node.kind in ("Constant", "Num", "Str", "Bytes", "NameConstant")

    def _get_literal_value(self, node: IRNode) -> Any:
        if node.kind in ("Constant", "Num", "Str", "Bytes"):
            return node.attr("value")
        if node.kind == "NameConstant":
            return node.attr("value")
        return None

    def _type_might_be_none(self, node: IRNode, state: AbstractState) -> bool:
        if node.kind == "Name":
            t = state.get_type(node.name)
            if t is not None:
                return t.contains_none()
        if node.type_info is not None:
            return node.type_info.contains_none()
        return False

    def _find_enclosing_loop(self, node: IRNode) -> Optional[IRNode]:
        return node.ancestor("For") or node.ancestor("While") or node.ancestor("AsyncFor")

    def _find_enclosing_function(self, node: IRNode) -> Optional[IRNode]:
        return node.ancestor("FunctionDef") or node.ancestor("AsyncFunctionDef")


# ===================================================================
# ARRAY PATTERNS
# ===================================================================

class OffByOneAccess(PatternMatcher):
    """Detects arr[len(arr)] which should be arr[len(arr)-1]."""
    name = "off-by-one-access"
    category = "array"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Subscript":
            return results
        arr_node = ir_node.child(0)
        idx_node = ir_node.child(1)
        if arr_node is None or idx_node is None:
            return results
        # Pattern: arr[len(arr)]
        if self._is_call(idx_node, "len"):
            len_arg = idx_node.child(0)
            if len_arg is None:
                len_arg_children = [c for c in idx_node.children if c.kind != "Name" or c.name != "len"]
                if len_arg_children:
                    len_arg = len_arg_children[0]
            if len_arg is not None and self._same_name(arr_node, len_arg):
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Off-by-one: indexing with len() gives IndexError. "
                    f"Use len({arr_node.name})-1 for last element.",
                    confidence=Confidence.HIGH,
                    fix_suggestion=f"{arr_node.name}[len({arr_node.name}) - 1]",
                ))
        # Pattern: arr[len(arr) + k] for k > 0
        if idx_node.kind == "BinOp" and idx_node.attr("op") == "Add":
            left = idx_node.child(0)
            right = idx_node.child(1)
            if left is not None and self._is_call(left, "len"):
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Off-by-one: index beyond array length.",
                    confidence=Confidence.HIGH,
                    fix_suggestion=f"Use index < len({arr_node.name})",
                ))
        # Pattern from abstract state: index == length
        if arr_node.kind == "Name" and idx_node.kind == "Name":
            arr_val = abstract_state.get(arr_node.name)
            idx_val = abstract_state.get(idx_node.name)
            if arr_val and idx_val:
                if (arr_val.length_upper is not None and
                        idx_val.lower_bound is not None and
                        idx_val.lower_bound >= arr_val.length_upper):
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Potential IndexError: index {idx_node.name} (>= {idx_val.lower_bound}) "
                        f"may exceed array length (<= {arr_val.length_upper}).",
                        confidence=Confidence.MEDIUM,
                    ))
        return results

    def _same_name(self, a: IRNode, b: IRNode) -> bool:
        if a.kind == "Name" and b.kind == "Name":
            return a.name == b.name
        return False


class NegativeIndexUnintended(PatternMatcher):
    """Detects arr[-1] when arr might be empty."""
    name = "negative-index-empty"
    category = "array"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Subscript":
            return results
        arr_node = ir_node.child(0)
        idx_node = ir_node.child(1)
        if arr_node is None or idx_node is None:
            return results
        idx_val = self._get_literal_value(idx_node) if self._node_is_literal(idx_node) else None
        if idx_val is not None and isinstance(idx_val, int) and idx_val < 0:
            if arr_node.kind == "Name":
                av = abstract_state.get(arr_node.name)
                if av is not None and av.length_lower is not None and av.length_lower == 0:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Negative index {idx_val} on '{arr_node.name}' which might be empty, "
                        f"causing IndexError.",
                        confidence=Confidence.MEDIUM,
                        fix_suggestion=f"if {arr_node.name}: {arr_node.name}[{idx_val}]",
                    ))
                elif av is not None and av.length_lower is not None and abs(idx_val) > av.length_lower:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Negative index {idx_val} may exceed array length "
                        f"(min length {av.length_lower}).",
                        confidence=Confidence.LOW,
                    ))
        if idx_node.kind == "Name":
            iv = abstract_state.get(idx_node.name)
            if iv is not None and iv.upper_bound is not None and iv.upper_bound < 0:
                if arr_node.kind == "Name":
                    av = abstract_state.get(arr_node.name)
                    if av is not None and av.length_lower == 0:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Negative index variable '{idx_node.name}' used on "
                            f"possibly-empty '{arr_node.name}'.",
                            confidence=Confidence.LOW,
                        ))
        return results


class SliceOverflow(PatternMatcher):
    """Detects arr[i:j] where j > len(arr)."""
    name = "slice-overflow"
    category = "array"
    severity = Severity.INFO

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Subscript":
            return results
        slice_node = ir_node.child(1)
        if slice_node is None or slice_node.kind != "Slice":
            return results
        arr_node = ir_node.child(0)
        if arr_node is None or arr_node.kind != "Name":
            return results
        upper = slice_node.attr("upper")
        if upper is None:
            return results
        arr_val = abstract_state.get(arr_node.name)
        if arr_val is None or arr_val.length_upper is None:
            return results
        if isinstance(upper, (int, float)):
            if upper > arr_val.length_upper:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Slice upper bound {upper} exceeds maximum array length "
                    f"{arr_val.length_upper} for '{arr_node.name}'. "
                    f"Python silently truncates, but this may indicate a logic error.",
                    confidence=Confidence.LOW,
                    fix_suggestion=f"{arr_node.name}[:{arr_val.length_upper}]",
                ))
        if isinstance(upper, str):
            uv = abstract_state.get(upper)
            if uv and uv.lower_bound is not None and uv.lower_bound > arr_val.length_upper:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Slice upper bound '{upper}' (>= {uv.lower_bound}) may exceed "
                    f"array length (<= {arr_val.length_upper}).",
                    confidence=Confidence.LOW,
                ))
        return results


class EmptyArrayIteration(PatternMatcher):
    """Detects iteration over a possibly-empty array."""
    name = "empty-array-iteration"
    category = "array"
    severity = Severity.INFO

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("For", "AsyncFor"):
            return results
        iter_node = ir_node.attr("iter")
        if iter_node is None and len(ir_node.children) >= 2:
            iter_node = ir_node.child(1)
        if iter_node is None:
            return results
        if iter_node.kind == "Name":
            av = abstract_state.get(iter_node.name)
            if av and av.length_lower == 0 and av.length_upper == 0:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Iterating over '{iter_node.name}' which is always empty. "
                    f"Loop body will never execute.",
                    confidence=Confidence.HIGH,
                    severity=Severity.WARNING,
                ))
            elif av and av.length_lower is not None and av.length_lower == 0:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Iterating over '{iter_node.name}' which might be empty. "
                    f"Consider handling the empty case.",
                    confidence=Confidence.LOW,
                ))
        return results


class ArrayLengthMismatch(PatternMatcher):
    """Detects zip(a, b) where len(a) != len(b)."""
    name = "array-length-mismatch"
    category = "array"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if not self._is_call(ir_node, "zip"):
            return results
        args = [c for c in ir_node.children[1:] if c.kind != "keyword"]
        if len(args) < 2:
            return results
        lengths: List[Tuple[str, Optional[int], Optional[int]]] = []
        for a in args:
            if a.kind == "Name":
                av = abstract_state.get(a.name)
                if av:
                    lengths.append((a.name, av.length_lower, av.length_upper))
                else:
                    lengths.append((a.name, None, None))
            else:
                lengths.append(("<expr>", None, None))
        for i in range(len(lengths)):
            for j in range(i + 1, len(lengths)):
                n1, lo1, hi1 = lengths[i]
                n2, lo2, hi2 = lengths[j]
                if lo1 is not None and hi1 is not None and lo2 is not None and hi2 is not None:
                    if hi1 < lo2 or hi2 < lo1:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"zip({n1}, {n2}): lengths definitely differ "
                            f"({n1}: [{lo1},{hi1}], {n2}: [{lo2},{hi2}]). "
                            f"Shorter sequence truncates. Use itertools.zip_longest if needed.",
                            confidence=Confidence.HIGH,
                            fix_suggestion="import itertools; itertools.zip_longest("
                                           + ", ".join(l[0] for l in lengths) + ")",
                        ))
                    elif lo1 != hi1 or lo2 != hi2 or lo1 != lo2:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"zip({n1}, {n2}): lengths might differ. "
                            f"Consider itertools.zip_longest or explicit length check.",
                            confidence=Confidence.LOW,
                            severity=Severity.INFO,
                        ))
        return results


class AppendInLoop(PatternMatcher):
    """Detects repeated append in loop that could be a list comprehension."""
    name = "append-in-loop"
    category = "array"
    severity = Severity.INFO

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("For", "AsyncFor"):
            return results
        append_targets: Dict[str, int] = {}
        for desc in ir_node.iter_descendants():
            if desc.kind == "Expr":
                call = desc.child(0)
                if call and call.kind == "Call":
                    fn = call.child(0)
                    if fn and fn.kind == "Attribute" and fn.attr("attr") == "append":
                        obj = fn.child(0)
                        if obj and obj.kind == "Name":
                            append_targets[obj.name] = append_targets.get(obj.name, 0) + 1
        for target, count in append_targets.items():
            if count >= 1:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Repeated .append() to '{target}' in loop. "
                    f"Consider using a list comprehension for better performance.",
                    confidence=Confidence.LOW,
                    fix_suggestion=f"{target} = [<expr> for <var> in <iterable>]",
                ))
        return results


class SortedSearchUnsorted(PatternMatcher):
    """Detects binary search on unsorted array."""
    name = "sorted-search-unsorted"
    category = "array"
    severity = Severity.WARNING

    _SEARCH_FUNCS = {"bisect", "bisect_left", "bisect_right", "insort",
                     "insort_left", "insort_right"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn_name = self._get_call_name(ir_node)
        if fn_name not in self._SEARCH_FUNCS:
            return results
        arr_arg = ir_node.child(1) if len(ir_node.children) > 1 else None
        if arr_arg is None:
            return results
        if arr_arg.kind == "Name":
            av = abstract_state.get(arr_arg.name)
            if av and av.is_sorted is False:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Binary search function '{fn_name}' called on '{arr_arg.name}' "
                    f"which may not be sorted. Results will be incorrect.",
                    confidence=Confidence.HIGH,
                    fix_suggestion=f"{arr_arg.name}.sort()  # or sorted({arr_arg.name})",
                ))
            elif av and av.is_sorted is None:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Binary search '{fn_name}' on '{arr_arg.name}': "
                    f"cannot verify array is sorted.",
                    confidence=Confidence.LOW,
                    severity=Severity.INFO,
                ))
        return results


class ConcurrentModification(PatternMatcher):
    """Detects modifying a list during iteration."""
    name = "concurrent-modification"
    category = "array"
    severity = Severity.ERROR

    _MUTATING_METHODS = {"append", "extend", "insert", "remove", "pop", "clear",
                         "sort", "reverse", "__setitem__", "__delitem__",
                         "add", "discard", "update"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("For", "AsyncFor"):
            return results
        iter_node = ir_node.child(1) if len(ir_node.children) >= 2 else None
        if iter_node is None:
            return results
        iter_name: Optional[str] = None
        if iter_node.kind == "Name":
            iter_name = iter_node.name
        elif iter_node.kind == "Call":
            fn = iter_node.child(0)
            if fn and fn.kind == "Attribute":
                obj = fn.child(0)
                if obj and obj.kind == "Name":
                    iter_name = obj.name
        if iter_name is None:
            return results
        body_nodes = list(ir_node.iter_descendants())
        for node in body_nodes:
            if node.kind == "Call":
                fn = node.child(0)
                if fn and fn.kind == "Attribute" and fn.attr("attr") in self._MUTATING_METHODS:
                    obj = fn.child(0)
                    if obj and obj.kind == "Name" and obj.name == iter_name:
                        results.append(self._make_match(
                            node, self.name,
                            f"Modifying '{iter_name}' (via .{fn.attr('attr')}()) while "
                            f"iterating over it. This can cause skipped elements or RuntimeError.",
                            confidence=Confidence.HIGH,
                            fix_suggestion=f"Iterate over a copy: for x in list({iter_name}):",
                        ))
            if node.kind == "Delete":
                for target in node.children:
                    if target.kind == "Subscript":
                        obj = target.child(0)
                        if obj and obj.kind == "Name" and obj.name == iter_name:
                            results.append(self._make_match(
                                node, self.name,
                                f"Deleting from '{iter_name}' while iterating over it.",
                                confidence=Confidence.HIGH,
                                fix_suggestion=f"Iterate over a copy: for x in list({iter_name}):",
                            ))
        return results


class IndexFromDifferentArray(PatternMatcher):
    """Detects arr1[idx] where idx was computed from arr2."""
    name = "index-from-different-array"
    category = "array"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Subscript":
            return results
        arr_node = ir_node.child(0)
        idx_node = ir_node.child(1)
        if arr_node is None or idx_node is None:
            return results
        if arr_node.kind != "Name" or idx_node.kind != "Name":
            return results
        idx_val = abstract_state.get(idx_node.name)
        if idx_val is None:
            return results
        source_array = idx_val.type_info.refinements.get("source_array")
        if source_array and source_array != arr_node.name:
            arr_val = abstract_state.get(arr_node.name)
            src_val = abstract_state.get(source_array)
            detail = ""
            if arr_val and src_val:
                if (arr_val.length_upper is not None and src_val.length_upper is not None
                        and arr_val.length_upper != src_val.length_upper):
                    detail = (f" ({arr_node.name} has max length {arr_val.length_upper}, "
                              f"{source_array} has max length {src_val.length_upper})")
            results.append(self._make_match(
                ir_node, self.name,
                f"Index '{idx_node.name}' was computed from '{source_array}' "
                f"but used to index '{arr_node.name}'.{detail}",
                confidence=Confidence.MEDIUM,
            ))
        return results


class RepeatedLinearSearch(PatternMatcher):
    """Detects O(n) lookup in a loop (suggest dict)."""
    name = "repeated-linear-search"
    category = "array"
    severity = Severity.INFO

    _SEARCH_METHODS = {"index", "count"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("For", "AsyncFor"):
            return results
        body_nodes = list(ir_node.iter_descendants())
        linear_ops: Dict[str, List[str]] = {}
        for node in body_nodes:
            if node.kind == "Compare" and node.attr("op") in ("In", "NotIn"):
                right = node.child(1)
                if right and right.kind == "Name":
                    rt = abstract_state.get_type(right.name)
                    if rt and rt.is_sequence:
                        linear_ops.setdefault(right.name, []).append("in")
            if node.kind == "Call":
                fn = node.child(0)
                if fn and fn.kind == "Attribute" and fn.attr("attr") in self._SEARCH_METHODS:
                    obj = fn.child(0)
                    if obj and obj.kind == "Name":
                        linear_ops.setdefault(obj.name, []).append(fn.attr("attr"))
        for name, ops in linear_ops.items():
            if len(ops) >= 1:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Linear search on '{name}' inside loop ({', '.join(ops)}). "
                    f"Consider using a set or dict for O(1) lookup.",
                    confidence=Confidence.MEDIUM,
                    fix_suggestion=f"{name}_set = set({name})  # precompute outside loop",
                ))
        return results


class ArrayPatterns(PatternMatcher):
    """Aggregator for all array-related pattern matchers."""
    name = "array-patterns"
    category = "array"

    def __init__(self) -> None:
        self._matchers: List[PatternMatcher] = [
            OffByOneAccess(),
            NegativeIndexUnintended(),
            SliceOverflow(),
            EmptyArrayIteration(),
            ArrayLengthMismatch(),
            AppendInLoop(),
            SortedSearchUnsorted(),
            ConcurrentModification(),
            IndexFromDifferentArray(),
            RepeatedLinearSearch(),
        ]

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        for m in self._matchers:
            if m.enabled:
                results.extend(m.match(ir_node, abstract_state))
        return results


# ===================================================================
# NULL PATTERNS
# ===================================================================

class OptionalFieldAccess(PatternMatcher):
    """Detects obj.field where field might be None."""
    name = "optional-field-access"
    category = "null"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Attribute":
            return results
        obj = ir_node.child(0)
        if obj is None:
            return results
        attr_name = ir_node.attr("attr")
        if obj.kind == "Name":
            val = abstract_state.get(obj.name)
            if val and val.type_info.contains_none():
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Accessing attribute '.{attr_name}' on '{obj.name}' "
                    f"which might be None, causing AttributeError.",
                    confidence=Confidence.HIGH,
                    fix_suggestion=f"if {obj.name} is not None: {obj.name}.{attr_name}",
                ))
        if obj.kind == "Attribute":
            inner = obj.child(0)
            if inner and inner.kind == "Name":
                val = abstract_state.get(inner.name)
                if val and val.type_info.contains_none():
                    inner_attr = obj.attr("attr")
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Chained access '{inner.name}.{inner_attr}.{attr_name}' "
                        f"but '{inner.name}' might be None.",
                        confidence=Confidence.MEDIUM,
                    ))
        return results


class NullInCollection(PatternMatcher):
    """Detects None in list/dict values."""
    name = "null-in-collection"
    category = "null"
    severity = Severity.INFO

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind == "List":
            for i, child in enumerate(ir_node.children):
                if child.kind in ("Constant", "NameConstant") and self._get_literal_value(child) is None:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"None value at index {i} in list literal. "
                        f"Downstream code may not expect None.",
                        confidence=Confidence.LOW,
                    ))
                elif child.kind == "Name":
                    t = abstract_state.get_type(child.name)
                    if t and t.contains_none():
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Variable '{child.name}' (possibly None) used in list literal.",
                            confidence=Confidence.LOW,
                        ))
        if ir_node.kind == "Dict":
            values = ir_node.attr("values") or ir_node.children
            for v in values:
                if isinstance(v, IRNode):
                    if v.kind in ("Constant", "NameConstant") and self._get_literal_value(v) is None:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"None value in dict literal.",
                            confidence=Confidence.LOW,
                        ))
        return results


class ComparisonWithNone(PatternMatcher):
    """Detects x == None instead of x is None."""
    name = "comparison-with-none"
    category = "null"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Compare":
            return results
        ops = ir_node.attr("ops") or []
        comparators = ir_node.attr("comparators") or ir_node.children[1:]
        left = ir_node.child(0)
        if not isinstance(ops, list):
            ops = [ops]
        for i, op in enumerate(ops):
            if op in ("Eq", "NotEq", "==", "!="):
                comp = comparators[i] if i < len(comparators) else None
                if comp is None and i < len(ir_node.children) - 1:
                    comp = ir_node.children[i + 1]
                if comp is not None:
                    comp_val = self._get_literal_value(comp) if self._node_is_literal(comp) else "NOTNONE"
                    left_val = self._get_literal_value(left) if left and self._node_is_literal(left) else "NOTNONE"
                    if comp_val is None or left_val is None:
                        identity_op = "is" if op in ("Eq", "==") else "is not"
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Use '{identity_op} None' instead of '{op} None'. "
                            f"Equality comparison can be overridden by __eq__.",
                            confidence=Confidence.HIGH,
                            fix_suggestion=f"x {identity_op} None",
                        ))
        return results


class NullReturnUnchecked(PatternMatcher):
    """Detects ignoring a return value that might be None."""
    name = "null-return-unchecked"
    category = "null"
    severity = Severity.WARNING

    _NULLABLE_BUILTINS = {"dict.get", "re.search", "re.match", "re.fullmatch",
                          "list.pop", "next", "getattr", "os.environ.get",
                          "os.getenv", "importlib.import_module"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Assign":
            return results
        value = ir_node.child(1) if len(ir_node.children) >= 2 else ir_node.attr("value")
        if value is None:
            return results
        if not isinstance(value, IRNode) or value.kind != "Call":
            return results
        fn_name = self._get_call_name(value)
        if fn_name is None:
            return results
        fn = value.child(0)
        full_name = fn_name
        if fn and fn.kind == "Attribute":
            obj = fn.child(0)
            if obj and obj.kind == "Name":
                full_name = f"{obj.name}.{fn_name}"
                obj_type = abstract_state.get_type(obj.name)
                if obj_type:
                    full_name = f"{obj_type.base}.{fn_name}"
        is_nullable = full_name in self._NULLABLE_BUILTINS or fn_name in self._NULLABLE_BUILTINS
        if not is_nullable and value.type_info and value.type_info.contains_none():
            is_nullable = True
        if not is_nullable:
            return results
        target = ir_node.child(0)
        if target is None:
            return results
        target_name = target.name if target.kind == "Name" else None
        if target_name is None:
            return results
        parent = ir_node.parent
        if parent is None:
            return results
        found_self = False
        for sibling in parent.children:
            if sibling is ir_node:
                found_self = True
                continue
            if not found_self:
                continue
            if sibling.kind == "If":
                cond = sibling.child(0)
                if cond and self._is_none_check(cond, target_name):
                    return results
            for desc in sibling.iter_descendants():
                if desc.kind == "Attribute" and desc.child(0) and desc.child(0).kind == "Name":
                    if desc.child(0).name == target_name:
                        results.append(self._make_match(
                            desc, self.name,
                            f"'{target_name}' from '{full_name}()' might be None "
                            f"but is used without a None check.",
                            confidence=Confidence.MEDIUM,
                            fix_suggestion=f"if {target_name} is not None: ...",
                        ))
                        return results
            break
        return results

    def _is_none_check(self, node: IRNode, name: str) -> bool:
        if node.kind == "Compare":
            left = node.child(0)
            if left and left.kind == "Name" and left.name == name:
                ops = node.attr("ops") or []
                if any(op in ("Is", "IsNot", "is", "is not") for op in
                       (ops if isinstance(ops, list) else [ops])):
                    return True
        if node.kind == "Name" and node.name == name:
            return True
        return False


class NullAfterExceptionHandler(PatternMatcher):
    """Detects variable that might be None if exception path taken."""
    name = "null-after-exception"
    category = "null"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Try":
            return results
        try_body = ir_node.attr("body") or (ir_node.children[:1] if ir_node.children else [])
        assigned_in_try: Set[str] = set()
        if isinstance(try_body, list):
            for stmt in try_body:
                if isinstance(stmt, IRNode):
                    for desc in [stmt] + list(stmt.iter_descendants()):
                        if desc.kind == "Assign":
                            tgt = desc.child(0)
                            if tgt and tgt.kind == "Name":
                                assigned_in_try.add(tgt.name)
        parent = ir_node.parent
        if parent is None:
            return results
        found_self = False
        for sibling in parent.children:
            if sibling is ir_node:
                found_self = True
                continue
            if not found_self:
                continue
            for desc in [sibling] + list(sibling.iter_descendants()):
                if desc.kind == "Name" and desc.name in assigned_in_try:
                    val = abstract_state.get(desc.name)
                    if val is None or val.type_info.contains_none():
                        results.append(self._make_match(
                            desc, self.name,
                            f"Variable '{desc.name}' assigned in try block might be "
                            f"uninitialized if exception occurred before assignment.",
                            confidence=Confidence.MEDIUM,
                            fix_suggestion=f"{desc.name} = None  # initialize before try",
                        ))
                        assigned_in_try.discard(desc.name)
        return results


class OptionalChainMissing(PatternMatcher):
    """Detects obj.a.b.c where any intermediate might be None."""
    name = "optional-chain-missing"
    category = "null"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Attribute":
            return results
        chain = self._collect_chain(ir_node)
        if len(chain) < 2:
            return results
        root = chain[0]
        if root.kind != "Name":
            return results
        val = abstract_state.get(root.name)
        if val is None:
            return results
        nullable_at: List[str] = []
        current_path = root.name
        if val.type_info.contains_none():
            nullable_at.append(current_path)
        for attr_node in chain[1:]:
            attr_name = attr_node.attr("attr") or attr_node.name
            current_path += f".{attr_name}"
        if nullable_at:
            results.append(self._make_match(
                ir_node, self.name,
                f"Chained attribute access '{current_path}' but "
                f"{', '.join(nullable_at)} might be None.",
                confidence=Confidence.MEDIUM,
                fix_suggestion="Use getattr() with default or check each level for None.",
            ))
        return results

    def _collect_chain(self, node: IRNode) -> List[IRNode]:
        chain: List[IRNode] = []
        current: Optional[IRNode] = node
        while current is not None:
            chain.append(current)
            if current.kind == "Attribute":
                current = current.child(0)
            else:
                break
        chain.reverse()
        return chain


class DanglingReference(PatternMatcher):
    """Detects variable used after being set to None."""
    name = "dangling-reference"
    category = "null"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Attribute":
            return results
        obj = ir_node.child(0)
        if obj is None or obj.kind != "Name":
            return results
        val = abstract_state.get(obj.name)
        if val is None:
            return results
        attr_name = ir_node.attr("attr")
        if val.type_info.is_none and val.is_constant:
            results.append(self._make_match(
                ir_node, self.name,
                f"'{obj.name}' is None (was explicitly set) but "
                f"'.{attr_name}' is accessed.",
                confidence=Confidence.HIGH,
                fix_suggestion=f"Remove the None assignment or guard the access.",
            ))
        return results


class UnintentionalNone(PatternMatcher):
    """Detects function missing return statement (returns None)."""
    name = "unintentional-none"
    category = "null"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("FunctionDef", "AsyncFunctionDef"):
            return results
        has_return_value = False
        for desc in ir_node.iter_descendants():
            if desc.kind == "Return":
                val = desc.child(0)
                if val is not None:
                    has_return_value = True
            if desc.kind in ("FunctionDef", "AsyncFunctionDef"):
                continue
        if has_return_value:
            body = ir_node.children
            if body and not self._all_paths_return(body):
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Function '{ir_node.name}' has explicit return with value "
                    f"but not all code paths return a value (implicit None).",
                    confidence=Confidence.MEDIUM,
                    fix_suggestion="Add return statement to all code paths.",
                ))
        return results

    def _all_paths_return(self, stmts: List[IRNode]) -> bool:
        if not stmts:
            return False
        last = stmts[-1]
        if last.kind == "Return":
            return last.child(0) is not None
        if last.kind == "Raise":
            return True
        if last.kind == "If":
            body = last.attr("body") or last.children[1:2]
            orelse = last.attr("orelse") or last.children[2:3]
            if isinstance(body, list) and isinstance(orelse, list) and orelse:
                return self._all_paths_return(body) and self._all_paths_return(orelse)
            return False
        return False


class NoneAsDefaultMutable(PatternMatcher):
    """Detects def f(x=None): x.append(...) without guard."""
    name = "none-as-default-mutable"
    category = "null"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("FunctionDef", "AsyncFunctionDef"):
            return results
        defaults = ir_node.attr("defaults") or []
        args = ir_node.attr("args") or []
        none_params: Set[str] = set()
        if isinstance(args, list) and isinstance(defaults, list):
            offset = len(args) - len(defaults)
            for i, d in enumerate(defaults):
                if isinstance(d, IRNode) and self._node_is_literal(d) and self._get_literal_value(d) is None:
                    param_idx = offset + i
                    if param_idx < len(args):
                        p = args[param_idx]
                        if isinstance(p, IRNode) and p.kind in ("arg", "Name"):
                            none_params.add(p.name if p.name else p.attr("arg", ""))
                        elif isinstance(p, str):
                            none_params.add(p)
        if not none_params:
            return results
        mutating_methods = {"append", "extend", "insert", "remove", "pop", "add",
                            "update", "clear", "sort", "reverse"}
        for desc in ir_node.iter_descendants():
            if desc.kind == "Call":
                fn = desc.child(0)
                if fn and fn.kind == "Attribute" and fn.attr("attr") in mutating_methods:
                    obj = fn.child(0)
                    if obj and obj.kind == "Name" and obj.name in none_params:
                        if_ancestor = desc.ancestor("If")
                        guarded = False
                        if if_ancestor:
                            cond = if_ancestor.child(0)
                            if cond and cond.kind == "Compare":
                                left = cond.child(0)
                                if left and left.kind == "Name" and left.name == obj.name:
                                    guarded = True
                            if cond and cond.kind == "Name" and cond.name == obj.name:
                                guarded = True
                        if not guarded:
                            method_name = fn.attr("attr")
                            results.append(self._make_match(
                                desc, self.name,
                                f"Parameter '{obj.name}' has default None but "
                                f"'.{method_name}()' called without None check.",
                                confidence=Confidence.HIGH,
                                fix_suggestion=f"if {obj.name} is None: {obj.name} = []",
                            ))
        return results


class AssertNotNone(PatternMatcher):
    """Detects assert x is not None but x might be None later."""
    name = "assert-not-none"
    category = "null"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Assert":
            return results
        test = ir_node.child(0)
        if test is None:
            return results
        if test.kind == "Compare":
            left = test.child(0)
            ops = test.attr("ops") or []
            if left and left.kind == "Name":
                is_none_check = any(op in ("IsNot", "is not") for op in
                                    (ops if isinstance(ops, list) else [ops]))
                if is_none_check:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"assert {left.name} is not None: assertions can be "
                        f"disabled with -O flag. Use explicit if-check for safety.",
                        confidence=Confidence.MEDIUM,
                        severity=Severity.INFO,
                        fix_suggestion=(
                            f"if {left.name} is None: "
                            f"raise ValueError('{left.name} must not be None')"
                        ),
                    ))
        return results


class NullPatterns(PatternMatcher):
    """Aggregator for null-related patterns."""
    name = "null-patterns"
    category = "null"

    def __init__(self) -> None:
        self._matchers: List[PatternMatcher] = [
            OptionalFieldAccess(),
            NullInCollection(),
            ComparisonWithNone(),
            NullReturnUnchecked(),
            NullAfterExceptionHandler(),
            OptionalChainMissing(),
            DanglingReference(),
            UnintentionalNone(),
            NoneAsDefaultMutable(),
            AssertNotNone(),
        ]

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        for m in self._matchers:
            if m.enabled:
                results.extend(m.match(ir_node, abstract_state))
        return results


# ===================================================================
# TYPE PATTERNS
# ===================================================================

class UnionMethodCall(PatternMatcher):
    """Detects calling a method only available on some union members."""
    name = "union-method-call"
    category = "type"
    severity = Severity.ERROR

    _TYPE_METHODS: Dict[str, Set[str]] = {
        "str": {"upper", "lower", "strip", "split", "join", "replace", "find",
                "startswith", "endswith", "format", "encode", "isdigit", "isalpha"},
        "int": {"bit_length", "to_bytes", "from_bytes", "conjugate"},
        "float": {"is_integer", "hex", "fromhex", "conjugate"},
        "list": {"append", "extend", "insert", "remove", "pop", "sort", "reverse",
                 "index", "count", "copy", "clear"},
        "dict": {"keys", "values", "items", "get", "pop", "update", "setdefault",
                 "clear", "copy", "fromkeys"},
        "set": {"add", "remove", "discard", "pop", "clear", "union", "intersection",
                "difference", "symmetric_difference", "issubset", "issuperset"},
        "tuple": {"index", "count"},
        "bytes": {"decode", "hex", "upper", "lower", "strip", "split", "find"},
    }

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None or fn.kind != "Attribute":
            return results
        method_name = fn.attr("attr")
        if method_name is None:
            return results
        obj = fn.child(0)
        if obj is None or obj.kind != "Name":
            return results
        t = abstract_state.get_type(obj.name)
        if t is None or not t.is_union:
            return results
        members_with: List[str] = []
        members_without: List[str] = []
        for member in t.union_members():
            type_methods = self._TYPE_METHODS.get(member.base, set())
            if method_name in type_methods:
                members_with.append(member.base)
            else:
                members_without.append(member.base)
        if members_without and members_with:
            type_str = ", ".join(m.base for m in t.union_members())
            results.append(self._make_match(
                ir_node, self.name,
                f"Method '.{method_name}()' called on '{obj.name}' "
                f"(Union[{type_str}]) "
                f"but not available on: {', '.join(members_without)}.",
                confidence=Confidence.HIGH,
                fix_suggestion=(
                    f"if isinstance({obj.name}, ({', '.join(members_with)})): "
                    f"{obj.name}.{method_name}(...)"
                ),
            ))
        return results


class ImplicitConversion(PatternMatcher):
    """Detects int + str without explicit conversion."""
    name = "implicit-conversion"
    category = "type"
    severity = Severity.ERROR

    _INCOMPATIBLE_OPS: Dict[str, Set[FrozenSet[str]]] = {
        "Add": {frozenset({"str", "int"}), frozenset({"str", "float"}),
                frozenset({"str", "list"}), frozenset({"bytes", "str"})},
        "Sub": {frozenset({"str", "int"}), frozenset({"str", "float"}),
                frozenset({"str", "str"})},
        "Mult": set(),
        "Div": {frozenset({"str", "int"}), frozenset({"str", "float"}),
                frozenset({"str", "str"})},
    }

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "BinOp":
            return results
        op = ir_node.attr("op")
        if op is None:
            return results
        left = ir_node.child(0)
        right = ir_node.child(1)
        if left is None or right is None:
            return results
        lt = self._resolve_type(left, abstract_state)
        rt = self._resolve_type(right, abstract_state)
        if lt is None or rt is None:
            return results
        pair = frozenset({lt, rt})
        incompatible = self._INCOMPATIBLE_OPS.get(op, set())
        if pair in incompatible:
            results.append(self._make_match(
                ir_node, self.name,
                f"Implicit conversion: {lt} {op} {rt} will raise TypeError.",
                confidence=Confidence.HIGH,
                fix_suggestion=f"Use explicit conversion: str() or int()",
            ))
        return results

    def _resolve_type(self, node: IRNode, state: AbstractState) -> Optional[str]:
        if node.kind == "Name":
            t = state.get_type(node.name)
            if t:
                return t.base
        if self._node_is_literal(node):
            v = self._get_literal_value(node)
            if isinstance(v, str):
                return "str"
            if isinstance(v, int):
                return "int"
            if isinstance(v, float):
                return "float"
        if node.type_info:
            return node.type_info.base
        return None


class WrongContainerMethod(PatternMatcher):
    """Detects dict.append() or list.get()."""
    name = "wrong-container-method"
    category = "type"
    severity = Severity.ERROR

    _WRONG_METHODS: Dict[str, Dict[str, str]] = {
        "dict": {
            "append": "Use dict[key] = value or dict.setdefault()",
            "extend": "Use dict.update()",
            "insert": "Dicts don't support insert",
            "sort": "Use sorted(dict.keys()) or dict(sorted(d.items()))",
            "reverse": "Dicts don't support reverse",
        },
        "list": {
            "get": "Use list[i] with bounds check or try/except",
            "keys": "Lists don't have keys",
            "values": "Lists don't have values",
            "items": "Use enumerate(list) instead",
        },
        "set": {
            "append": "Use set.add()",
            "extend": "Use set.update()",
            "insert": "Sets don't support insert",
            "index": "Sets don't support index",
            "sort": "Use sorted(set_var)",
        },
        "tuple": {
            "append": "Tuples are immutable, convert to list first",
            "extend": "Tuples are immutable",
            "insert": "Tuples are immutable",
            "remove": "Tuples are immutable",
            "pop": "Tuples are immutable",
            "sort": "Use sorted(tuple_var)",
        },
    }

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None or fn.kind != "Attribute":
            return results
        method = fn.attr("attr")
        obj = fn.child(0)
        if obj is None or obj.kind != "Name" or method is None:
            return results
        t = abstract_state.get_type(obj.name)
        if t is None:
            return results
        wrong = self._WRONG_METHODS.get(t.base, {})
        if method in wrong:
            results.append(self._make_match(
                ir_node, self.name,
                f"'{t.base}' object '{obj.name}' has no method '.{method}()'. "
                f"{wrong[method]}.",
                confidence=Confidence.HIGH,
                fix_suggestion=wrong[method],
            ))
        return results


class NonCallable(PatternMatcher):
    """Detects calling a non-callable value."""
    name = "non-callable"
    category = "type"
    severity = Severity.ERROR

    _NON_CALLABLE = {"int", "float", "str", "bytes", "NoneType", "bool"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None or fn.kind != "Name":
            return results
        t = abstract_state.get_type(fn.name)
        if t is None:
            return results
        if t.base in self._NON_CALLABLE and not t.is_callable:
            results.append(self._make_match(
                ir_node, self.name,
                f"'{fn.name}' is of type '{t.base}' which is not callable.",
                confidence=Confidence.HIGH,
            ))
        return results


class WrongArgumentType(PatternMatcher):
    """Detects passing int where str expected, etc."""
    name = "wrong-argument-type"
    category = "type"
    severity = Severity.ERROR

    _BUILTIN_SIGS: Dict[str, List[str]] = {
        "len": ["Sized"],
        "int": ["str|float|int|bool"],
        "str": ["Any"],
        "float": ["str|float|int"],
        "abs": ["int|float|complex"],
        "round": ["int|float"],
        "sorted": ["Iterable"],
        "reversed": ["Sequence"],
        "sum": ["Iterable"],
        "min": ["Iterable"],
        "max": ["Iterable"],
        "chr": ["int"],
        "ord": ["str"],
        "hex": ["int"],
        "oct": ["int"],
        "bin": ["int"],
    }

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None or fn.kind != "Name":
            return results
        sig = self._BUILTIN_SIGS.get(fn.name)
        if sig is None:
            return results
        args = [c for c in ir_node.children[1:] if c.kind != "keyword"]
        for i, (arg, expected) in enumerate(zip(args, sig)):
            if expected == "Any":
                continue
            if arg.kind == "Name":
                t = abstract_state.get_type(arg.name)
                if t is None:
                    continue
                allowed = set(expected.split("|"))
                if t.base not in allowed and "Any" not in allowed:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Argument {i} to '{fn.name}()': expected {expected}, "
                        f"got '{t.base}' (from '{arg.name}').",
                        confidence=Confidence.MEDIUM,
                    ))
        return results


class WrongReturnType(PatternMatcher):
    """Detects returning wrong type from annotated function."""
    name = "wrong-return-type"
    category = "type"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Return":
            return results
        val = ir_node.child(0)
        if val is None:
            return results
        func = self._find_enclosing_function(ir_node)
        if func is None:
            return results
        ret_annotation = func.attr("return_annotation")
        if ret_annotation is None:
            return results
        expected_type: Optional[str] = None
        if isinstance(ret_annotation, str):
            expected_type = ret_annotation
        elif isinstance(ret_annotation, IRNode) and ret_annotation.kind == "Name":
            expected_type = ret_annotation.name
        if expected_type is None:
            return results
        actual_type: Optional[str] = None
        if val.kind == "Name":
            t = abstract_state.get_type(val.name)
            if t:
                actual_type = t.base
        elif self._node_is_literal(val):
            v = self._get_literal_value(val)
            actual_type = type(v).__name__ if v is not None else "NoneType"
        if actual_type and expected_type and actual_type != expected_type:
            if not (expected_type == "float" and actual_type == "int"):
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Return type mismatch: expected '{expected_type}', "
                    f"returning '{actual_type}'.",
                    confidence=Confidence.MEDIUM,
                ))
        return results


class IncompatibleComparison(PatternMatcher):
    """Detects comparing incompatible types."""
    name = "incompatible-comparison"
    category = "type"
    severity = Severity.WARNING

    _INCOMPATIBLE_PAIRS = {
        frozenset({"str", "int"}),
        frozenset({"str", "list"}),
        frozenset({"str", "dict"}),
        frozenset({"list", "dict"}),
        frozenset({"list", "set"}),
        frozenset({"dict", "set"}),
        frozenset({"int", "list"}),
        frozenset({"float", "str"}),
    }

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Compare":
            return results
        left = ir_node.child(0)
        comparators = ir_node.children[1:]
        if left is None or not comparators:
            return results
        lt = self._resolve_type_name(left, abstract_state)
        for comp in comparators:
            rt = self._resolve_type_name(comp, abstract_state)
            if lt and rt:
                pair = frozenset({lt, rt})
                if pair in self._INCOMPATIBLE_PAIRS:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Comparing incompatible types: {lt} and {rt}. "
                        f"Result is always False for == and always True for !=.",
                        confidence=Confidence.HIGH,
                    ))
        return results

    def _resolve_type_name(self, node: IRNode, state: AbstractState) -> Optional[str]:
        if node.kind == "Name":
            t = state.get_type(node.name)
            return t.base if t else None
        if self._node_is_literal(node):
            v = self._get_literal_value(node)
            return type(v).__name__ if v is not None else "NoneType"
        return None


class MissingMethodInProtocol(PatternMatcher):
    """Detects object not implementing required protocol method."""
    name = "missing-protocol-method"
    category = "type"
    severity = Severity.ERROR

    _PROTOCOLS: Dict[str, List[str]] = {
        "Iterator": ["__iter__", "__next__"],
        "Iterable": ["__iter__"],
        "Sized": ["__len__"],
        "Container": ["__contains__"],
        "Hashable": ["__hash__"],
        "Comparable": ["__lt__", "__le__", "__gt__", "__ge__"],
        "ContextManager": ["__enter__", "__exit__"],
        "AsyncContextManager": ["__aenter__", "__aexit__"],
        "Callable": ["__call__"],
    }

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind == "For":
            iter_node = ir_node.child(1)
            if iter_node and iter_node.kind == "Name":
                t = abstract_state.get_type(iter_node.name)
                if (t and not t.is_sequence
                        and t.base not in ("range", "generator", "map", "filter",
                                           "zip", "enumerate")):
                    required = self._PROTOCOLS.get("Iterable", [])
                    has_methods = t.refinements.get("methods", set())
                    missing = [m for m in required if m not in has_methods]
                    if missing and isinstance(has_methods, set) and len(has_methods) > 0:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"'{iter_node.name}' of type '{t.base}' used in for loop "
                            f"but missing Iterable methods: {', '.join(missing)}.",
                            confidence=Confidence.MEDIUM,
                        ))
        if ir_node.kind == "With":
            ctx = ir_node.child(0)
            if ctx and ctx.kind == "Name":
                t = abstract_state.get_type(ctx.name)
                if t:
                    has_methods = t.refinements.get("methods", set())
                    required = self._PROTOCOLS.get("ContextManager", [])
                    missing = [m for m in required
                               if isinstance(has_methods, set) and m not in has_methods]
                    if missing and isinstance(has_methods, set) and len(has_methods) > 0:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"'{ctx.name}' of type '{t.base}' used as context manager "
                            f"but missing: {', '.join(missing)}.",
                            confidence=Confidence.MEDIUM,
                        ))
        return results


class VariableTypeChange(PatternMatcher):
    """Detects variable changing type across branches."""
    name = "variable-type-change"
    category = "type"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "If":
            return results
        body = ir_node.attr("body") or ir_node.children[1:2]
        orelse = ir_node.attr("orelse") or ir_node.children[2:3]
        if not isinstance(body, list) or not isinstance(orelse, list):
            return results
        body_assigns = self._collect_assigns(body)
        else_assigns = self._collect_assigns(orelse)
        common = set(body_assigns.keys()) & set(else_assigns.keys())
        for name in common:
            bt = body_assigns[name]
            et = else_assigns[name]
            if bt and et and bt != et:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Variable '{name}' has type '{bt}' in if-branch "
                    f"but '{et}' in else-branch. May cause type errors downstream.",
                    confidence=Confidence.MEDIUM,
                ))
        return results

    def _collect_assigns(self, stmts: List[Any]) -> Dict[str, Optional[str]]:
        assigns: Dict[str, Optional[str]] = {}
        for stmt in stmts:
            if not isinstance(stmt, IRNode):
                continue
            if stmt.kind == "Assign":
                tgt = stmt.child(0)
                val = stmt.child(1)
                if tgt and tgt.kind == "Name":
                    vtype = None
                    if val and val.type_info:
                        vtype = val.type_info.base
                    elif val and self._node_is_literal(val):
                        v = self._get_literal_value(val)
                        vtype = type(v).__name__ if v is not None else "NoneType"
                    assigns[tgt.name] = vtype
        return assigns


class UnsafeCoercion(PatternMatcher):
    """Detects implicit type coercion that might fail."""
    name = "unsafe-coercion"
    category = "type"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None or fn.kind != "Name":
            return results
        if fn.name not in ("int", "float"):
            return results
        arg = ir_node.child(1) if len(ir_node.children) > 1 else None
        if arg is None:
            return results
        if arg.kind == "Name":
            t = abstract_state.get_type(arg.name)
            if t and t.base == "str":
                results.append(self._make_match(
                    ir_node, self.name,
                    f"{fn.name}('{arg.name}') where '{arg.name}' is str: "
                    f"will raise ValueError if string is not a valid number.",
                    confidence=Confidence.MEDIUM,
                    fix_suggestion=f"try: {fn.name}({arg.name}) except ValueError: ...",
                ))
        return results


class TypePatterns(PatternMatcher):
    """Aggregator for type-related patterns."""
    name = "type-patterns"
    category = "type"

    def __init__(self) -> None:
        self._matchers: List[PatternMatcher] = [
            UnionMethodCall(),
            ImplicitConversion(),
            WrongContainerMethod(),
            NonCallable(),
            WrongArgumentType(),
            WrongReturnType(),
            IncompatibleComparison(),
            MissingMethodInProtocol(),
            VariableTypeChange(),
            UnsafeCoercion(),
        ]

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        for m in self._matchers:
            if m.enabled:
                results.extend(m.match(ir_node, abstract_state))
        return results


# ===================================================================
# ARITHMETIC PATTERNS
# ===================================================================

class IntegerOverflow(PatternMatcher):
    """Detects potential integer overflow for fixed-width integer interfaces."""
    name = "integer-overflow"
    category = "arithmetic"
    severity = Severity.WARNING

    _LIMITS = {
        "int8": (-128, 127),
        "int16": (-32768, 32767),
        "int32": (-2147483648, 2147483647),
        "int64": (-9223372036854775808, 9223372036854775807),
        "uint8": (0, 255),
        "uint16": (0, 65535),
        "uint32": (0, 4294967295),
        "uint64": (0, 18446744073709551615),
    }

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "BinOp":
            return results
        op = ir_node.attr("op")
        if op not in ("Add", "Sub", "Mult", "Pow", "LShift"):
            return results
        left = ir_node.child(0)
        right = ir_node.child(1)
        if left is None or right is None:
            return results
        lv = self._get_bounds(left, abstract_state)
        rv = self._get_bounds(right, abstract_state)
        if lv is None or rv is None:
            return results
        result_range = self._estimate_result(op, lv, rv)
        if result_range is None:
            return results
        lo, hi = result_range
        i32_lo, i32_hi = self._LIMITS["int32"]
        if lo < i32_lo or hi > i32_hi:
            results.append(self._make_match(
                ir_node, self.name,
                f"Result of {op} might exceed 32-bit integer range "
                f"[{lo}, {hi}]. If interfacing with C/numpy, this may overflow.",
                confidence=Confidence.LOW,
                severity=Severity.INFO,
            ))
        return results

    def _get_bounds(
        self, node: IRNode, state: AbstractState
    ) -> Optional[Tuple[int, int]]:
        if node.kind == "Name":
            v = state.get(node.name)
            if v and v.lower_bound is not None and v.upper_bound is not None:
                return (int(v.lower_bound), int(v.upper_bound))
        if self._node_is_literal(node):
            val = self._get_literal_value(node)
            if isinstance(val, (int, float)):
                return (int(val), int(val))
        return None

    def _estimate_result(
        self, op: str,
        lv: Tuple[int, int],
        rv: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        l_lo, l_hi = lv
        r_lo, r_hi = rv
        if op == "Add":
            return (l_lo + r_lo, l_hi + r_hi)
        if op == "Sub":
            return (l_lo - r_hi, l_hi - r_lo)
        if op == "Mult":
            products = [l_lo * r_lo, l_lo * r_hi, l_hi * r_lo, l_hi * r_hi]
            return (min(products), max(products))
        if op == "Pow":
            if r_hi <= 64 and l_hi <= 1000:
                try:
                    vals = [l_lo ** r_lo, l_lo ** r_hi, l_hi ** r_lo, l_hi ** r_hi]
                    return (min(vals), max(vals))
                except (OverflowError, ValueError):
                    return (-(2**63), 2**63)
        if op == "LShift":
            if r_hi <= 64:
                return (l_lo << max(0, r_lo), l_hi << r_hi)
        return None


class FloatingPointComparison(PatternMatcher):
    """Detects floating point equality comparison (e.g. 0.1 + 0.2 == 0.3)."""
    name = "floating-point-comparison"
    category = "arithmetic"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Compare":
            return results
        ops = ir_node.attr("ops") or []
        if not isinstance(ops, list):
            ops = [ops]
        if not any(op in ("Eq", "NotEq", "==", "!=") for op in ops):
            return results
        left = ir_node.child(0)
        comparators = ir_node.children[1:]
        all_nodes = ([left] + comparators) if left else comparators
        has_float_expr = False
        for node in all_nodes:
            if node is None:
                continue
            if self._involves_float(node, abstract_state):
                has_float_expr = True
                break
        if has_float_expr:
            results.append(self._make_match(
                ir_node, self.name,
                "Floating point equality comparison. Due to precision, "
                "0.1 + 0.2 != 0.3 in IEEE 754. Use math.isclose() or abs(a-b) < epsilon.",
                confidence=Confidence.MEDIUM,
                fix_suggestion="import math; math.isclose(a, b, rel_tol=1e-9)",
            ))
        return results

    def _involves_float(self, node: IRNode, state: AbstractState) -> bool:
        if self._node_is_literal(node):
            v = self._get_literal_value(node)
            return isinstance(v, float)
        if node.kind == "Name":
            t = state.get_type(node.name)
            if t and t.base == "float":
                return True
        if node.kind == "BinOp":
            for child in node.children:
                if self._involves_float(child, state):
                    return True
        return False


class DivisionTruncation(PatternMatcher):
    """Detects integer division where float might be expected."""
    name = "division-truncation"
    category = "arithmetic"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "BinOp":
            return results
        op = ir_node.attr("op")
        if op != "FloorDiv":
            return results
        parent = ir_node.parent
        if parent and parent.kind == "BinOp":
            sibling = parent.child(1) if parent.child(0) is ir_node else parent.child(0)
            if sibling:
                st = self._resolve_type(sibling, abstract_state)
                if st == "float":
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Floor division (//) used but result combined with float. "
                        f"Did you mean true division (/) ?",
                        confidence=Confidence.MEDIUM,
                        fix_suggestion="Use / instead of // for true division",
                    ))
        if parent and parent.kind == "Assign":
            target = parent.child(0)
            if target and target.kind == "Name":
                t = abstract_state.get_type(target.name)
                if t and t.base == "float":
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Floor division (//) result assigned to float variable. "
                        f"Integer truncation may not be intended.",
                        confidence=Confidence.MEDIUM,
                        fix_suggestion="Use / instead of //",
                    ))
        return results

    def _resolve_type(self, node: IRNode, state: AbstractState) -> Optional[str]:
        if node.kind == "Name":
            t = state.get_type(node.name)
            return t.base if t else None
        if self._node_is_literal(node):
            v = self._get_literal_value(node)
            if isinstance(v, float):
                return "float"
            if isinstance(v, int):
                return "int"
        return None


class ModuloNegative(PatternMatcher):
    """Detects x % n where x might be negative."""
    name = "modulo-negative"
    category = "arithmetic"
    severity = Severity.INFO

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "BinOp" or ir_node.attr("op") != "Mod":
            return results
        left = ir_node.child(0)
        if left is None:
            return results
        if left.kind == "Name":
            v = abstract_state.get(left.name)
            if v and v.lower_bound is not None and v.lower_bound < 0:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"'{left.name}' might be negative in modulo operation. "
                    f"Python's % always returns non-negative for positive divisor, "
                    f"but behavior differs in C/Java.",
                    confidence=Confidence.LOW,
                ))
        return results


class AccumulatorOverflow(PatternMatcher):
    """Detects sum growing without bound in loop."""
    name = "accumulator-overflow"
    category = "arithmetic"
    severity = Severity.INFO

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("For", "While"):
            return results
        accumulators: Dict[str, str] = {}
        for desc in ir_node.iter_descendants():
            if desc.kind == "AugAssign":
                op = desc.attr("op")
                target = desc.child(0)
                if target and target.kind == "Name" and op in ("Add", "Mult", "+=", "*="):
                    accumulators[target.name] = op
        for name, op in accumulators.items():
            v = abstract_state.get(name)
            if v and v.upper_bound is not None and v.upper_bound > 1e15:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Accumulator '{name}' ({op}) in loop might grow very large "
                    f"(current upper bound: {v.upper_bound}). "
                    f"Consider using math.fsum() for float sums.",
                    confidence=Confidence.LOW,
                ))
        return results


class PrecisionLoss(PatternMatcher):
    """Detects large int to float conversion losing precision."""
    name = "precision-loss"
    category = "arithmetic"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None or fn.kind != "Name" or fn.name != "float":
            return results
        arg = ir_node.child(1) if len(ir_node.children) > 1 else None
        if arg is None:
            return results
        if arg.kind == "Name":
            v = abstract_state.get(arg.name)
            if v and v.type_info.base == "int":
                if v.lower_bound is not None and v.upper_bound is not None:
                    if abs(v.lower_bound) > 2**53 or abs(v.upper_bound) > 2**53:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Converting large int '{arg.name}' to float may lose precision. "
                            f"float has 53 bits of mantissa, value range "
                            f"[{v.lower_bound}, {v.upper_bound}] exceeds 2^53.",
                            confidence=Confidence.HIGH,
                            fix_suggestion="Use Decimal for arbitrary precision.",
                        ))
        return results


class OffByOneLoop(PatternMatcher):
    """Detects off-by-one in loop range."""
    name = "off-by-one-loop"
    category = "arithmetic"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "For":
            return results
        iter_node = ir_node.child(1) if len(ir_node.children) >= 2 else None
        if iter_node is None or not self._is_call(iter_node, "range"):
            return results
        range_args = [c for c in iter_node.children if c is not iter_node.child(0)]
        if not range_args:
            return results
        target = ir_node.child(0)
        if target is None or target.kind != "Name":
            return results
        idx_name = target.name
        for desc in ir_node.iter_descendants():
            if desc.kind == "Subscript":
                arr = desc.child(0)
                idx = desc.child(1)
                if arr and idx and arr.kind == "Name" and idx.kind == "Name":
                    if idx.name == idx_name:
                        av = abstract_state.get(arr.name)
                        if av and av.length_upper is not None:
                            range_upper = None
                            if len(range_args) == 1:
                                ru = range_args[0]
                                if self._node_is_literal(ru):
                                    range_upper = self._get_literal_value(ru)
                            elif len(range_args) >= 2:
                                ru = range_args[1]
                                if self._node_is_literal(ru):
                                    range_upper = self._get_literal_value(ru)
                            if range_upper is not None and isinstance(range_upper, int):
                                if range_upper > av.length_upper:
                                    results.append(self._make_match(
                                        desc, self.name,
                                        f"Loop range upper bound ({range_upper}) exceeds "
                                        f"'{arr.name}' length ({av.length_upper}). "
                                        f"IndexError on last iteration.",
                                        confidence=Confidence.HIGH,
                                        severity=Severity.ERROR,
                                        fix_suggestion=f"range(len({arr.name}))",
                                    ))
                                elif (av.length_lower is not None
                                      and range_upper < av.length_lower):
                                    results.append(self._make_match(
                                        desc, self.name,
                                        f"Loop range ({range_upper}) is less than "
                                        f"'{arr.name}' length ({av.length_lower}). "
                                        f"Not all elements processed.",
                                        confidence=Confidence.LOW,
                                        severity=Severity.INFO,
                                    ))
        return results


class InfiniteLoop(PatternMatcher):
    """Detects loop condition that never becomes false."""
    name = "infinite-loop"
    category = "arithmetic"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "While":
            return results
        cond = ir_node.child(0)
        if cond is None:
            return results
        if self._node_is_literal(cond):
            val = self._get_literal_value(cond)
            if val is True or val == 1:
                has_break = any(d.kind == "Break" for d in ir_node.iter_descendants())
                has_return = any(d.kind == "Return" for d in ir_node.iter_descendants())
                has_raise = any(d.kind == "Raise" for d in ir_node.iter_descendants())
                has_exit = any(
                    d.kind == "Call" and self._get_call_name(d) in (
                        "exit", "sys.exit", "quit", "os._exit")
                    for d in ir_node.iter_descendants()
                )
                if not has_break and not has_return and not has_raise and not has_exit:
                    results.append(self._make_match(
                        ir_node, self.name,
                        "while True loop with no break, return, raise, or exit. "
                        "This loop will run forever.",
                        confidence=Confidence.HIGH,
                    ))
        if cond.kind == "Compare":
            left = cond.child(0)
            if left and left.kind == "Name":
                modified = False
                for desc in ir_node.iter_descendants():
                    if desc.kind == "AugAssign":
                        t = desc.child(0)
                        if t and t.kind == "Name" and t.name == left.name:
                            modified = True
                    if desc.kind == "Assign":
                        t = desc.child(0)
                        if t and t.kind == "Name" and t.name == left.name:
                            modified = True
                if not modified:
                    has_break = any(d.kind == "Break" for d in ir_node.iter_descendants())
                    if not has_break:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Loop variable '{left.name}' in while condition is never "
                            f"modified in loop body. Loop may be infinite.",
                            confidence=Confidence.MEDIUM,
                        ))
        return results


class UnreachableCode(PatternMatcher):
    """Detects code after unconditional return/raise."""
    name = "unreachable-code"
    category = "arithmetic"
    severity = Severity.WARNING

    _TERMINAL = {"Return", "Raise", "Break", "Continue"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("FunctionDef", "AsyncFunctionDef", "If",
                                "For", "While", "Try"):
            return results
        body: List[Any] = ir_node.attr("body") or ir_node.children
        if not isinstance(body, list):
            return results
        found_terminal = False
        terminal_node: Optional[IRNode] = None
        for stmt in body:
            if not isinstance(stmt, IRNode):
                continue
            if found_terminal:
                t_kind = terminal_node.kind if terminal_node else "terminal"
                t_line = terminal_node.location.line if terminal_node else "?"
                results.append(self._make_match(
                    stmt, self.name,
                    f"Unreachable code after {t_kind} statement at line {t_line}.",
                    confidence=Confidence.HIGH,
                    fix_suggestion="Remove unreachable code or restructure control flow.",
                ))
                break
            if stmt.kind in self._TERMINAL:
                found_terminal = True
                terminal_node = stmt
        return results


class ArithmeticPatterns(PatternMatcher):
    """Aggregator for arithmetic patterns."""
    name = "arithmetic-patterns"
    category = "arithmetic"

    def __init__(self) -> None:
        self._matchers: List[PatternMatcher] = [
            IntegerOverflow(),
            FloatingPointComparison(),
            DivisionTruncation(),
            ModuloNegative(),
            AccumulatorOverflow(),
            PrecisionLoss(),
            OffByOneLoop(),
            InfiniteLoop(),
            UnreachableCode(),
        ]

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        for m in self._matchers:
            if m.enabled:
                results.extend(m.match(ir_node, abstract_state))
        return results


# ===================================================================
# STRING PATTERNS
# ===================================================================

class FormatStringMismatch(PatternMatcher):
    """Detects format string args don't match placeholders."""
    name = "format-string-mismatch"
    category = "string"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        if fn.kind == "Attribute" and fn.attr("attr") == "format":
            obj = fn.child(0)
            if obj and self._node_is_literal(obj):
                fmt_str = self._get_literal_value(obj)
                if isinstance(fmt_str, str):
                    placeholders = self._count_format_placeholders(fmt_str)
                    args = [c for c in ir_node.children[1:] if c.kind != "keyword"]
                    kwargs = [c for c in ir_node.children[1:] if c.kind == "keyword"]
                    positional_count = len(args)
                    if placeholders.get("positional", 0) > positional_count:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Format string expects {placeholders['positional']} positional "
                            f"args but {positional_count} provided.",
                            confidence=Confidence.HIGH,
                        ))
                    for pname in placeholders.get("named", []):
                        found = any(c.attr("arg") == pname for c in kwargs)
                        if not found:
                            results.append(self._make_match(
                                ir_node, self.name,
                                f"Format string references '{{{pname}}}' "
                                f"but no keyword argument '{pname}' provided.",
                                confidence=Confidence.HIGH,
                            ))
        return results

    def _count_format_placeholders(self, fmt: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"positional": 0, "named": []}
        i = 0
        auto_idx = 0
        while i < len(fmt):
            if fmt[i] == "{":
                if i + 1 < len(fmt) and fmt[i + 1] == "{":
                    i += 2
                    continue
                j = fmt.find("}", i)
                if j == -1:
                    break
                field_spec = fmt[i + 1:j].split(":")[0].split("!")[0]
                if field_spec == "" or field_spec.isdigit():
                    result["positional"] = max(
                        result["positional"],
                        int(field_spec) + 1 if field_spec.isdigit() else auto_idx + 1,
                    )
                    auto_idx += 1
                else:
                    result["named"].append(field_spec)
                i = j + 1
            else:
                i += 1
        return result


class EncodingMismatch(PatternMatcher):
    """Detects mixing bytes and str."""
    name = "encoding-mismatch"
    category = "string"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "BinOp":
            return results
        op = ir_node.attr("op")
        if op != "Add":
            return results
        left = ir_node.child(0)
        right = ir_node.child(1)
        if left is None or right is None:
            return results
        lt = self._resolve_type(left, abstract_state)
        rt = self._resolve_type(right, abstract_state)
        if (lt == "str" and rt == "bytes") or (lt == "bytes" and rt == "str"):
            results.append(self._make_match(
                ir_node, self.name,
                f"Concatenating str and bytes: TypeError. "
                f"Use .encode() or .decode() to convert.",
                confidence=Confidence.HIGH,
                fix_suggestion="bytes_var.decode('utf-8') or str_var.encode('utf-8')",
            ))
        return results

    def _resolve_type(self, node: IRNode, state: AbstractState) -> Optional[str]:
        if node.kind == "Name":
            t = state.get_type(node.name)
            return t.base if t else None
        if self._node_is_literal(node):
            v = self._get_literal_value(node)
            if isinstance(v, str):
                return "str"
            if isinstance(v, bytes):
                return "bytes"
        return None


class SqlInjection(PatternMatcher):
    """Detects string formatting in SQL queries."""
    name = "sql-injection"
    category = "string"
    severity = Severity.ERROR

    _SQL_KEYWORDS = {"SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
                     "ALTER", "EXEC", "EXECUTE", "UNION", "WHERE"}
    _EXECUTE_METHODS = {"execute", "executemany", "executescript", "raw",
                        "execute_sql"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        is_db_execute = (fn.kind == "Attribute"
                         and fn.attr("attr") in self._EXECUTE_METHODS)
        if not is_db_execute:
            return results
        arg = ir_node.child(1) if len(ir_node.children) > 1 else None
        if arg is None:
            return results
        if arg.kind == "JoinedStr":
            results.append(self._make_match(
                ir_node, self.name,
                "SQL injection risk: f-string used in SQL query. "
                "Use parameterized queries instead.",
                confidence=Confidence.HIGH,
                fix_suggestion='cursor.execute("SELECT * FROM t WHERE id = ?", (user_id,))',
            ))
        if arg.kind == "BinOp" and arg.attr("op") in ("Add", "Mod"):
            left = arg.child(0)
            if left and self._node_is_literal(left):
                val = self._get_literal_value(left)
                if isinstance(val, str):
                    upper_val = val.upper()
                    if any(kw in upper_val for kw in self._SQL_KEYWORDS):
                        results.append(self._make_match(
                            ir_node, self.name,
                            "SQL injection risk: string concatenation in SQL query. "
                            "Use parameterized queries.",
                            confidence=Confidence.HIGH,
                            fix_suggestion='cursor.execute("... WHERE id = ?", (param,))',
                        ))
        if arg.kind == "Call":
            arg_fn = arg.child(0)
            if arg_fn and arg_fn.kind == "Attribute" and arg_fn.attr("attr") == "format":
                obj = arg_fn.child(0)
                if obj and self._node_is_literal(obj):
                    val = self._get_literal_value(obj)
                    if isinstance(val, str) and any(
                        kw in val.upper() for kw in self._SQL_KEYWORDS
                    ):
                        results.append(self._make_match(
                            ir_node, self.name,
                            "SQL injection risk: .format() on SQL string. "
                            "Use parameterized queries.",
                            confidence=Confidence.HIGH,
                        ))
        return results


class PathInjection(PatternMatcher):
    """Detects user input in file path."""
    name = "path-injection"
    category = "string"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn_name = self._get_call_name(ir_node)
        if fn_name not in ("open",):
            return results
        arg = ir_node.child(1) if len(ir_node.children) > 1 else None
        if arg is None:
            return results
        if arg.kind == "Name":
            v = abstract_state.get(arg.name)
            if v and v.type_info.refinements.get("tainted"):
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Path injection risk: '{arg.name}' (tainted) used as file path. "
                    f"Sanitize with os.path.realpath() and check against allowed paths.",
                    confidence=Confidence.HIGH,
                    fix_suggestion=(
                        "import os; safe = os.path.realpath(path); "
                        "assert safe.startswith(ALLOWED_DIR)"
                    ),
                ))
        if arg.kind == "JoinedStr":
            results.append(self._make_match(
                ir_node, self.name,
                "Potential path injection: f-string used as file path in open(). "
                "Ensure components are sanitized.",
                confidence=Confidence.MEDIUM,
            ))
        return results


class RegexError(PatternMatcher):
    """Detects invalid regex patterns."""
    name = "regex-error"
    category = "string"
    severity = Severity.ERROR

    _RE_FUNCS = {"search", "match", "fullmatch", "findall", "finditer",
                 "sub", "subn", "split", "compile"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        is_re_call = False
        if fn.kind == "Attribute" and fn.attr("attr") in self._RE_FUNCS:
            obj = fn.child(0)
            if obj and obj.kind == "Name" and obj.name == "re":
                is_re_call = True
        if not is_re_call:
            return results
        pattern_arg = ir_node.child(1) if len(ir_node.children) > 1 else None
        if pattern_arg is None:
            return results
        if self._node_is_literal(pattern_arg):
            pattern = self._get_literal_value(pattern_arg)
            if isinstance(pattern, str):
                try:
                    re.compile(pattern)
                except re.error as e:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Invalid regex pattern: {e}",
                        confidence=Confidence.HIGH,
                    ))
                if pattern.endswith("\\"):
                    results.append(self._make_match(
                        ir_node, self.name,
                        "Regex pattern ends with single backslash. "
                        "Use raw string r'...' to avoid escaping issues.",
                        confidence=Confidence.MEDIUM,
                        fix_suggestion=f"r'{pattern}'",
                    ))
                if "(?P<" in pattern:
                    group_names = re.findall(r"\(\?P<(\w+)>", pattern)
                    if len(group_names) != len(set(group_names)):
                        results.append(self._make_match(
                            ir_node, self.name,
                            "Duplicate named groups in regex pattern.",
                            confidence=Confidence.HIGH,
                        ))
        return results


class EmptyStringCheck(PatternMatcher):
    """Detects 'if s' instead of 'if len(s) > 0' (informational)."""
    name = "empty-string-check"
    category = "string"
    severity = Severity.INFO

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind not in ("If", "While", "IfExp"):
            return results
        cond = ir_node.child(0)
        if cond is None:
            return results
        check_name: Optional[str] = None
        negated = False
        if cond.kind == "Name":
            check_name = cond.name
        elif cond.kind == "UnaryOp" and cond.attr("op") == "Not":
            inner = cond.child(0)
            if inner and inner.kind == "Name":
                check_name = inner.name
                negated = True
        if check_name:
            t = abstract_state.get_type(check_name)
            if t and t.base == "str":
                prefix = "not " if negated else ""
                results.append(self._make_match(
                    ir_node, self.name,
                    f"'if {prefix}{check_name}' checks for empty/falsy string. "
                    f"This is Pythonic but also catches None if variable is Optional[str].",
                    confidence=Confidence.LOW,
                ))
        return results


class UnicodeError(PatternMatcher):
    """Detects non-ASCII in ASCII context."""
    name = "unicode-error"
    category = "string"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        if fn.kind == "Attribute" and fn.attr("attr") == "encode":
            obj = fn.child(0)
            encoding_arg = ir_node.child(1) if len(ir_node.children) > 1 else None
            if encoding_arg and self._node_is_literal(encoding_arg):
                enc = self._get_literal_value(encoding_arg)
                if isinstance(enc, str) and enc.lower() == "ascii":
                    if obj and obj.kind == "Name":
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Encoding '{obj.name}' as ASCII may fail with "
                            f"UnicodeEncodeError if it contains non-ASCII characters.",
                            confidence=Confidence.MEDIUM,
                            fix_suggestion=(
                                f"{obj.name}.encode('utf-8') or "
                                f".encode('ascii', errors='replace')"
                            ),
                        ))
        return results


class StringPatterns(PatternMatcher):
    """Aggregator for string-related patterns."""
    name = "string-patterns"
    category = "string"

    def __init__(self) -> None:
        self._matchers: List[PatternMatcher] = [
            FormatStringMismatch(),
            EncodingMismatch(),
            SqlInjection(),
            PathInjection(),
            RegexError(),
            EmptyStringCheck(),
            UnicodeError(),
        ]

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        for m in self._matchers:
            if m.enabled:
                results.extend(m.match(ir_node, abstract_state))
        return results


# ===================================================================
# RESOURCE PATTERNS
# ===================================================================

class UnclosedFile(PatternMatcher):
    """Detects open() without close() or with statement."""
    name = "unclosed-file"
    category = "resource"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Assign":
            return results
        value = ir_node.child(1) if len(ir_node.children) >= 2 else ir_node.attr("value")
        if not isinstance(value, IRNode):
            return results
        if not self._is_call(value, "open"):
            return results
        if ir_node.ancestor("With") or ir_node.ancestor("AsyncWith"):
            return results
        target = ir_node.child(0)
        if target is None or target.kind != "Name":
            return results
        file_var = target.name
        func = self._find_enclosing_function(ir_node) or ir_node.parent
        if func is None:
            return results
        has_close = False
        for desc in func.iter_descendants():
            if desc.kind == "Call":
                fn = desc.child(0)
                if fn and fn.kind == "Attribute" and fn.attr("attr") == "close":
                    obj = fn.child(0)
                    if obj and obj.kind == "Name" and obj.name == file_var:
                        has_close = True
        if not has_close:
            results.append(self._make_match(
                ir_node, self.name,
                f"File '{file_var}' opened but never closed. "
                f"Use 'with open(...) as f:' for automatic cleanup.",
                confidence=Confidence.MEDIUM,
                fix_suggestion=f"with open(...) as {file_var}:",
            ))
        return results


class UnclosedConnection(PatternMatcher):
    """Detects database/network connection not closed."""
    name = "unclosed-connection"
    category = "resource"
    severity = Severity.WARNING

    _CONNECT_FUNCS = {"connect", "create_connection", "urlopen", "socket",
                      "create_engine", "Connection", "Session"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Assign":
            return results
        value = ir_node.child(1) if len(ir_node.children) >= 2 else ir_node.attr("value")
        if not isinstance(value, IRNode) or value.kind != "Call":
            return results
        fn_name = self._get_call_name(value)
        if fn_name not in self._CONNECT_FUNCS:
            return results
        if ir_node.ancestor("With") or ir_node.ancestor("AsyncWith"):
            return results
        target = ir_node.child(0)
        if target is None or target.kind != "Name":
            return results
        conn_var = target.name
        func = self._find_enclosing_function(ir_node) or ir_node.parent
        if func is None:
            return results
        has_close = False
        for desc in func.iter_descendants():
            if desc.kind == "Call":
                fn = desc.child(0)
                if (fn and fn.kind == "Attribute"
                        and fn.attr("attr") in ("close", "disconnect", "shutdown")):
                    obj = fn.child(0)
                    if obj and obj.kind == "Name" and obj.name == conn_var:
                        has_close = True
        if not has_close:
            results.append(self._make_match(
                ir_node, self.name,
                f"Connection '{conn_var}' from '{fn_name}()' may not be closed. "
                f"Use context manager or explicit close().",
                confidence=Confidence.MEDIUM,
                fix_suggestion=f"with {fn_name}(...) as {conn_var}:",
            ))
        return results


class ResourceLeakInException(PatternMatcher):
    """Detects resource not closed on exception path."""
    name = "resource-leak-exception"
    category = "resource"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Try":
            return results
        try_body = ir_node.attr("body") or ir_node.children[:1]
        if not isinstance(try_body, list):
            return results
        resources: List[str] = []
        for stmt in try_body:
            if isinstance(stmt, IRNode) and stmt.kind == "Assign":
                val = stmt.child(1)
                if val and isinstance(val, IRNode) and val.kind == "Call":
                    fn_name = self._get_call_name(val)
                    if fn_name in ("open", "connect", "socket", "urlopen"):
                        tgt = stmt.child(0)
                        if tgt and tgt.kind == "Name":
                            resources.append(tgt.name)
        if not resources:
            return results
        finalbody = ir_node.attr("finalbody") or []
        for res in resources:
            closed_in_handler = False
            if isinstance(finalbody, list):
                for stmt in finalbody:
                    if isinstance(stmt, IRNode):
                        for desc in [stmt] + list(stmt.iter_descendants()):
                            if desc.kind == "Call":
                                fn = desc.child(0)
                                if (fn and fn.kind == "Attribute"
                                        and fn.attr("attr") == "close"):
                                    obj = fn.child(0)
                                    if obj and obj.kind == "Name" and obj.name == res:
                                        closed_in_handler = True
            if not closed_in_handler:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Resource '{res}' acquired in try block but not closed "
                    f"in finally block. Will leak if exception occurs.",
                    confidence=Confidence.MEDIUM,
                    fix_suggestion=f"Use 'with' statement or add finally: {res}.close()",
                ))
        return results


class DoubleFree(PatternMatcher):
    """Detects closing an already-closed resource."""
    name = "double-free"
    category = "resource"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None or fn.kind != "Attribute" or fn.attr("attr") != "close":
            return results
        obj = fn.child(0)
        if obj is None or obj.kind != "Name":
            return results
        val = abstract_state.get(obj.name)
        if val and val.is_closed is True:
            results.append(self._make_match(
                ir_node, self.name,
                f"'{obj.name}' is already closed. Double close may raise ValueError.",
                confidence=Confidence.HIGH,
            ))
        return results


class UseAfterClose(PatternMatcher):
    """Detects using file/resource after close()."""
    name = "use-after-close"
    category = "resource"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind == "Call":
            fn = ir_node.child(0)
            if fn and fn.kind == "Attribute":
                method = fn.attr("attr")
                if method and method != "close":
                    obj = fn.child(0)
                    if obj and obj.kind == "Name":
                        val = abstract_state.get(obj.name)
                        if val and val.is_closed is True:
                            results.append(self._make_match(
                                ir_node, self.name,
                                f"'{obj.name}.{method}()' called after close(). "
                                f"ValueError: I/O operation on closed file.",
                                confidence=Confidence.HIGH,
                            ))
        if ir_node.kind == "Attribute":
            obj = ir_node.child(0)
            attr_name = ir_node.attr("attr")
            if obj and obj.kind == "Name":
                val = abstract_state.get(obj.name)
                if val and val.is_closed is True:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Accessing '{obj.name}.{attr_name}' after close().",
                        confidence=Confidence.HIGH,
                    ))
        return results


class DeadlockPotential(PatternMatcher):
    """Detects nested locks in different order."""
    name = "deadlock-potential"
    category = "resource"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "With":
            return results
        lock_order: List[str] = []
        current: Optional[IRNode] = ir_node
        visited: Set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if current.kind == "With":
                items = current.attr("items") or current.children[:1]
                for item in (items if isinstance(items, list) else [items]):
                    if isinstance(item, IRNode) and item.kind == "Call":
                        fn = item.child(0)
                        if fn and fn.kind == "Name":
                            lock_order.append(fn.name)
                    elif isinstance(item, IRNode) and item.kind == "Name":
                        val = abstract_state.get(item.name)
                        if val and val.is_locked is not None:
                            lock_order.append(item.name)
            current = current.ancestor("With")
        if len(lock_order) >= 2 and abstract_state.locks_held:
            for held in abstract_state.locks_held:
                for acquiring in lock_order:
                    if held != acquiring:
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Potential deadlock: acquiring '{acquiring}' while "
                            f"holding '{held}'. Ensure consistent lock ordering.",
                            confidence=Confidence.MEDIUM,
                        ))
        return results


class ThreadSafetyViolation(PatternMatcher):
    """Detects shared mutable state without synchronization."""
    name = "thread-safety-violation"
    category = "resource"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn_name = self._get_call_name(ir_node)
        if fn_name not in ("Thread", "threading.Thread"):
            return results
        target_arg = None
        for child in ir_node.children:
            if child.kind == "keyword" and child.attr("arg") == "target":
                target_arg = child.child(0) if child.children else None
        if target_arg and target_arg.kind == "Name":
            val = abstract_state.get(target_arg.name)
            if val and val.type_info.is_callable:
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Thread target '{target_arg.name}' may access shared mutable state "
                    f"without synchronization. Consider using threading.Lock.",
                    confidence=Confidence.LOW,
                    severity=Severity.INFO,
                ))
        return results


class ResourcePatterns(PatternMatcher):
    """Aggregator for resource patterns."""
    name = "resource-patterns"
    category = "resource"

    def __init__(self) -> None:
        self._matchers: List[PatternMatcher] = [
            UnclosedFile(),
            UnclosedConnection(),
            ResourceLeakInException(),
            DoubleFree(),
            UseAfterClose(),
            DeadlockPotential(),
            ThreadSafetyViolation(),
        ]

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        for m in self._matchers:
            if m.enabled:
                results.extend(m.match(ir_node, abstract_state))
        return results


# ===================================================================
# CONCURRENCY PATTERNS
# ===================================================================

class AwaitMissing(PatternMatcher):
    """Detects calling async function without await."""
    name = "await-missing"
    category = "concurrency"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Expr":
            return results
        call = ir_node.child(0)
        if call is None or call.kind != "Call":
            return results
        fn = call.child(0)
        if fn is None:
            return results
        fn_name = (fn.name if fn.kind == "Name"
                   else fn.attr("attr") if fn.kind == "Attribute"
                   else None)
        if fn_name is None:
            return results
        if fn.kind == "Name":
            t = abstract_state.get_type(fn.name)
            if t and t.refinements.get("is_coroutine"):
                results.append(self._make_match(
                    ir_node, self.name,
                    f"Async function '{fn_name}()' called without await. "
                    f"Coroutine will not execute.",
                    confidence=Confidence.HIGH,
                    fix_suggestion=f"await {fn_name}(...)",
                ))
        return results


class RaceCondition(PatternMatcher):
    """Detects shared state access without synchronization."""
    name = "race-condition"
    category = "concurrency"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Assign":
            return results
        target = ir_node.child(0)
        if target is None:
            return results
        if target.kind == "Attribute":
            obj = target.child(0)
            attr_name = target.attr("attr")
            if obj and obj.kind == "Name" and obj.name == "self":
                if abstract_state.in_async or abstract_state.current_function:
                    func = ir_node.ancestor("AsyncFunctionDef")
                    if func:
                        if not abstract_state.locks_held:
                            results.append(self._make_match(
                                ir_node, self.name,
                                f"Assigning to 'self.{attr_name}' in async context "
                                f"without lock. Concurrent tasks may cause race conditions.",
                                confidence=Confidence.LOW,
                                severity=Severity.INFO,
                            ))
        return results


class DeadlockDetection(PatternMatcher):
    """Detects lock ordering violations."""
    name = "deadlock-detection"
    category = "concurrency"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        if fn.kind == "Attribute" and fn.attr("attr") == "acquire":
            obj = fn.child(0)
            if obj and obj.kind == "Name":
                lock_name = obj.name
                if lock_name in abstract_state.locks_held:
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"Re-acquiring lock '{lock_name}' which is already held. "
                        f"This will deadlock unless it's a RLock.",
                        confidence=Confidence.HIGH,
                        fix_suggestion="Use threading.RLock() for re-entrant locking.",
                    ))
        return results


class UnhandledRejection(PatternMatcher):
    """Detects async operations without error handling."""
    name = "unhandled-rejection"
    category = "concurrency"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn_name = self._get_call_name(ir_node)
        if fn_name not in ("create_task", "ensure_future", "gather"):
            return results
        if ir_node.ancestor("Try"):
            return results
        parent = ir_node.parent
        if parent and parent.kind == "Await":
            return results
        results.append(self._make_match(
            ir_node, self.name,
            f"asyncio.{fn_name}() without error handling. "
            f"Exceptions in the task will be silently swallowed.",
            confidence=Confidence.MEDIUM,
            fix_suggestion="try: result = await task except Exception as e: handle(e)",
        ))
        return results


class AsyncGeneratorLeak(PatternMatcher):
    """Detects async generator not properly closed."""
    name = "async-generator-leak"
    category = "concurrency"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Assign":
            return results
        value = ir_node.child(1) if len(ir_node.children) >= 2 else ir_node.attr("value")
        if not isinstance(value, IRNode) or value.kind != "Call":
            return results
        fn = value.child(0)
        if fn is None:
            return results
        fn_name = fn.name if fn.kind == "Name" else None
        if fn_name is None:
            return results
        t = abstract_state.get_type(fn_name)
        if t is None or not t.refinements.get("is_async_generator"):
            return results
        target = ir_node.child(0)
        if target is None or target.kind != "Name":
            return results
        gen_var = target.name
        func = self._find_enclosing_function(ir_node) or ir_node.parent
        if func is None:
            return results
        properly_used = False
        for desc in func.iter_descendants():
            if desc.kind == "AsyncFor":
                iter_node = desc.child(1)
                if iter_node and iter_node.kind == "Name" and iter_node.name == gen_var:
                    properly_used = True
            if desc.kind == "Call":
                fn_d = desc.child(0)
                if fn_d and fn_d.kind == "Attribute" and fn_d.attr("attr") == "aclose":
                    obj = fn_d.child(0)
                    if obj and obj.kind == "Name" and obj.name == gen_var:
                        properly_used = True
        if not properly_used:
            results.append(self._make_match(
                ir_node, self.name,
                f"Async generator '{gen_var}' may not be properly closed. "
                f"Use 'async for' or call .aclose().",
                confidence=Confidence.MEDIUM,
                fix_suggestion=(
                    f"async for item in {gen_var}: ... "
                    f"# or await {gen_var}.aclose()"
                ),
            ))
        return results


class ConcurrencyPatterns(PatternMatcher):
    """Aggregator for concurrency patterns."""
    name = "concurrency-patterns"
    category = "concurrency"

    def __init__(self) -> None:
        self._matchers: List[PatternMatcher] = [
            AwaitMissing(),
            RaceCondition(),
            DeadlockDetection(),
            UnhandledRejection(),
            AsyncGeneratorLeak(),
        ]

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        for m in self._matchers:
            if m.enabled:
                results.extend(m.match(ir_node, abstract_state))
        return results


# ===================================================================
# SECURITY PATTERNS
# ===================================================================

class HardcodedSecret(PatternMatcher):
    """Detects hardcoded passwords, API keys, etc."""
    name = "hardcoded-secret"
    category = "security"
    severity = Severity.ERROR

    _SECRET_NAMES = {"password", "passwd", "pwd", "secret", "api_key", "apikey",
                     "secret_key", "access_token", "auth_token", "private_key",
                     "db_password", "database_password", "token", "credentials"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Assign":
            return results
        target = ir_node.child(0)
        value = ir_node.child(1) if len(ir_node.children) >= 2 else ir_node.attr("value")
        if target is None or not isinstance(value, IRNode):
            return results
        target_name: Optional[str] = None
        if target.kind == "Name":
            target_name = target.name
        elif target.kind == "Attribute":
            target_name = target.attr("attr")
        if target_name is None:
            return results
        if target_name.lower() in self._SECRET_NAMES:
            if self._node_is_literal(value):
                val = self._get_literal_value(value)
                if isinstance(val, str) and len(val) > 0:
                    placeholders = {"", "changeme", "xxx", "TODO", "FIXME",
                                    "placeholder", "your_password_here",
                                    "your_api_key_here"}
                    if val.lower() not in placeholders:
                        display = val[:4] + "..." if len(val) > 4 else val
                        results.append(self._make_match(
                            ir_node, self.name,
                            f"Hardcoded secret: '{target_name}' = '{display}'. "
                            f"Use environment variables or secret management.",
                            confidence=Confidence.HIGH,
                            fix_suggestion=(
                                f"{target_name} = os.environ['{target_name.upper()}']"
                            ),
                        ))
        return results


class InsecureRandom(PatternMatcher):
    """Detects using random instead of secrets for security."""
    name = "insecure-random"
    category = "security"
    severity = Severity.WARNING

    _INSECURE_FUNCS = {"random", "randint", "randrange", "choice", "shuffle",
                       "sample", "getrandbits", "uniform"}

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        is_random_call = False
        func_name = ""
        if fn.kind == "Attribute" and fn.attr("attr") in self._INSECURE_FUNCS:
            obj = fn.child(0)
            if obj and obj.kind == "Name" and obj.name == "random":
                is_random_call = True
                func_name = f"random.{fn.attr('attr')}"
        if not is_random_call:
            return results
        parent = ir_node.parent
        if parent and parent.kind == "Assign":
            tgt = parent.child(0)
            if tgt and tgt.kind == "Name":
                name_lower = tgt.name.lower()
                security_names = {"token", "key", "secret", "password", "nonce",
                                  "salt", "iv", "session_id", "csrf", "otp"}
                if any(s in name_lower for s in security_names):
                    results.append(self._make_match(
                        ir_node, self.name,
                        f"'{func_name}' used for security-sensitive '{tgt.name}'. "
                        f"Use 'secrets' module instead.",
                        confidence=Confidence.HIGH,
                        severity=Severity.ERROR,
                        fix_suggestion=(
                            f"import secrets; {tgt.name} = secrets.token_hex(32)"
                        ),
                    ))
                    return results
        results.append(self._make_match(
            ir_node, self.name,
            f"'{func_name}' is not cryptographically secure. "
            f"Use 'secrets' module for security-sensitive randomness.",
            confidence=Confidence.LOW,
            severity=Severity.INFO,
        ))
        return results


class EvalUsage(PatternMatcher):
    """Detects eval() or exec() usage."""
    name = "eval-usage"
    category = "security"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None or fn.kind != "Name":
            return results
        if fn.name in ("eval", "exec"):
            arg = ir_node.child(1) if len(ir_node.children) > 1 else None
            if arg is None:
                return results
            is_tainted = False
            if arg.kind == "Name":
                v = abstract_state.get(arg.name)
                if v and v.type_info.refinements.get("tainted"):
                    is_tainted = True
            confidence = Confidence.HIGH if is_tainted else Confidence.MEDIUM
            msg_detail = ("User input is used — critical vulnerability!"
                          if is_tainted
                          else "Potential code injection risk.")
            fix = ("import ast; ast.literal_eval(expr)"
                   if fn.name == "eval" else None)
            results.append(self._make_match(
                ir_node, self.name,
                f"{fn.name}() usage detected. {msg_detail} "
                f"Use ast.literal_eval() for safe evaluation of literals.",
                confidence=confidence,
                fix_suggestion=fix,
            ))
        return results


class PickleUntrusted(PatternMatcher):
    """Detects unpickling untrusted data."""
    name = "pickle-untrusted"
    category = "security"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        is_pickle = False
        if fn.kind == "Attribute" and fn.attr("attr") in ("load", "loads"):
            obj = fn.child(0)
            if (obj and obj.kind == "Name"
                    and obj.name in ("pickle", "cPickle", "shelve", "dill")):
                is_pickle = True
        if is_pickle:
            arg = ir_node.child(1) if len(ir_node.children) > 1 else None
            is_tainted = False
            if arg and arg.kind == "Name":
                v = abstract_state.get(arg.name)
                if v and v.type_info.refinements.get("tainted"):
                    is_tainted = True
            detail = ("Data source appears tainted — critical vulnerability!"
                      if is_tainted
                      else "Ensure data is from trusted source.")
            results.append(self._make_match(
                ir_node, self.name,
                f"pickle.load() can execute arbitrary code. {detail} "
                f"Use json or msgpack for untrusted data.",
                confidence=Confidence.HIGH if is_tainted else Confidence.MEDIUM,
                fix_suggestion="import json; json.loads(data)",
            ))
        return results


class YamlUnsafeLoad(PatternMatcher):
    """Detects yaml.load() without safe Loader."""
    name = "yaml-unsafe-load"
    category = "security"
    severity = Severity.ERROR

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        if fn.kind == "Attribute" and fn.attr("attr") == "load":
            obj = fn.child(0)
            if obj and obj.kind == "Name" and obj.name == "yaml":
                has_safe_loader = False
                for child in ir_node.children:
                    if child.kind == "keyword" and child.attr("arg") == "Loader":
                        loader_val = child.child(0) if child.children else None
                        if loader_val:
                            loader_name = ""
                            if loader_val.kind == "Name":
                                loader_name = loader_val.name
                            elif loader_val.kind == "Attribute":
                                loader_name = loader_val.attr("attr", "")
                            if "Safe" in loader_name or "Base" in loader_name:
                                has_safe_loader = True
                if not has_safe_loader:
                    results.append(self._make_match(
                        ir_node, self.name,
                        "yaml.load() without SafeLoader can execute arbitrary code. "
                        "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader).",
                        confidence=Confidence.HIGH,
                        fix_suggestion="yaml.safe_load(data)",
                    ))
        if fn.kind == "Attribute" and fn.attr("attr") == "unsafe_load":
            obj = fn.child(0)
            if obj and obj.kind == "Name" and obj.name == "yaml":
                results.append(self._make_match(
                    ir_node, self.name,
                    "yaml.unsafe_load() can execute arbitrary code.",
                    confidence=Confidence.HIGH,
                ))
        return results


class SubprocessShell(PatternMatcher):
    """Detects subprocess with shell=True."""
    name = "subprocess-shell"
    category = "security"
    severity = Severity.WARNING

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        if ir_node.kind != "Call":
            return results
        fn = ir_node.child(0)
        if fn is None:
            return results
        is_subprocess = False
        func_name = ""
        if fn.kind == "Attribute" and fn.attr("attr") in (
            "call", "run", "Popen", "check_output",
            "check_call", "getoutput", "getstatusoutput",
        ):
            obj = fn.child(0)
            if obj and obj.kind == "Name" and obj.name in ("subprocess", "os"):
                is_subprocess = True
                func_name = f"{obj.name}.{fn.attr('attr')}"
        if fn.kind == "Name" and fn.name == "system":
            is_subprocess = True
            func_name = "os.system"
        if not is_subprocess:
            return results
        has_shell_true = False
        for child in ir_node.children:
            if child.kind == "keyword" and child.attr("arg") == "shell":
                val = child.child(0) if child.children else None
                if (val and self._node_is_literal(val)
                        and self._get_literal_value(val) is True):
                    has_shell_true = True
        if func_name == "os.system":
            has_shell_true = True
        if has_shell_true:
            cmd_arg = ir_node.child(1) if len(ir_node.children) > 1 else None
            is_tainted = False
            if cmd_arg:
                if cmd_arg.kind in ("JoinedStr", "BinOp"):
                    is_tainted = True
                if cmd_arg.kind == "Name":
                    v = abstract_state.get(cmd_arg.name)
                    if v and v.type_info.refinements.get("tainted"):
                        is_tainted = True
            sev = Severity.ERROR if is_tainted else Severity.WARNING
            detail = ("Command includes user input — shell injection vulnerability!"
                      if is_tainted
                      else "Prefer shell=False with list of args.")
            results.append(self._make_match(
                ir_node, self.name,
                f"{func_name}() with shell=True. {detail}",
                confidence=Confidence.HIGH if is_tainted else Confidence.MEDIUM,
                severity=sev,
                fix_suggestion="subprocess.run(['cmd', 'arg1', 'arg2'], shell=False)",
            ))
        return results


class SecurityPatterns(PatternMatcher):
    """Aggregator for security patterns."""
    name = "security-patterns"
    category = "security"

    def __init__(self) -> None:
        self._matchers: List[PatternMatcher] = [
            HardcodedSecret(),
            InsecureRandom(),
            EvalUsage(),
            PickleUntrusted(),
            YamlUnsafeLoad(),
            SubprocessShell(),
        ]

    def match(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        results: List[PatternMatch] = []
        for m in self._matchers:
            if m.enabled:
                results.extend(m.match(ir_node, abstract_state))
        return results


# ===================================================================
# PATTERN DATABASE
# ===================================================================

class PatternDatabase:
    """Central database of all registered patterns."""

    def __init__(self) -> None:
        self._patterns: Dict[str, PatternMatcher] = {}
        self._categories: Dict[str, List[PatternMatcher]] = {}
        self._node_type_index: Dict[str, List[PatternMatcher]] = {}
        self._severity_filter: Optional[Severity] = None
        self._disabled_patterns: Set[str] = set()
        self._disabled_categories: Set[str] = set()
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all built-in pattern matchers."""
        builtins: List[PatternMatcher] = [
            # Array patterns
            OffByOneAccess(), NegativeIndexUnintended(), SliceOverflow(),
            EmptyArrayIteration(), ArrayLengthMismatch(), AppendInLoop(),
            SortedSearchUnsorted(), ConcurrentModification(),
            IndexFromDifferentArray(), RepeatedLinearSearch(),
            # Null patterns
            OptionalFieldAccess(), NullInCollection(), ComparisonWithNone(),
            NullReturnUnchecked(), NullAfterExceptionHandler(),
            OptionalChainMissing(), DanglingReference(), UnintentionalNone(),
            NoneAsDefaultMutable(), AssertNotNone(),
            # Type patterns
            UnionMethodCall(), ImplicitConversion(), WrongContainerMethod(),
            NonCallable(), WrongArgumentType(), WrongReturnType(),
            IncompatibleComparison(), MissingMethodInProtocol(),
            VariableTypeChange(), UnsafeCoercion(),
            # Arithmetic patterns
            IntegerOverflow(), FloatingPointComparison(), DivisionTruncation(),
            ModuloNegative(), AccumulatorOverflow(), PrecisionLoss(),
            OffByOneLoop(), InfiniteLoop(), UnreachableCode(),
            # String patterns
            FormatStringMismatch(), EncodingMismatch(), SqlInjection(),
            PathInjection(), RegexError(), EmptyStringCheck(), UnicodeError(),
            # Resource patterns
            UnclosedFile(), UnclosedConnection(), ResourceLeakInException(),
            DoubleFree(), UseAfterClose(), DeadlockPotential(),
            ThreadSafetyViolation(),
            # Concurrency patterns
            AwaitMissing(), RaceCondition(), DeadlockDetection(),
            UnhandledRejection(), AsyncGeneratorLeak(),
            # Security patterns
            HardcodedSecret(), InsecureRandom(), EvalUsage(),
            PickleUntrusted(), YamlUnsafeLoad(), SubprocessShell(),
        ]
        for p in builtins:
            self.register_pattern(p)

    def register_pattern(self, pattern: PatternMatcher) -> None:
        """Register a pattern matcher."""
        self._patterns[pattern.name] = pattern
        cat = pattern.category
        if cat not in self._categories:
            self._categories[cat] = []
        self._categories[cat].append(pattern)

    def unregister_pattern(self, name: str) -> bool:
        """Remove a pattern by name."""
        p = self._patterns.pop(name, None)
        if p is None:
            return False
        cat_list = self._categories.get(p.category, [])
        self._categories[p.category] = [x for x in cat_list if x.name != name]
        return True

    def get_pattern(self, name: str) -> Optional[PatternMatcher]:
        """Get pattern by name."""
        return self._patterns.get(name)

    def get_patterns_for_category(self, category: str) -> List[PatternMatcher]:
        """Get all patterns in a category."""
        return list(self._categories.get(category, []))

    def get_patterns_for_node_type(self, node_kind: str) -> List[PatternMatcher]:
        """Get patterns relevant to a given IR node type."""
        if node_kind in self._node_type_index:
            return self._node_type_index[node_kind]
        return [p for p in self._patterns.values()
                if p.enabled and p.name not in self._disabled_patterns
                and p.category not in self._disabled_categories]

    def get_all_patterns(self) -> List[PatternMatcher]:
        """Get all registered patterns."""
        return list(self._patterns.values())

    def get_categories(self) -> List[str]:
        """Get all category names."""
        return list(self._categories.keys())

    def enable_pattern(self, name: str) -> None:
        """Enable a specific pattern."""
        self._disabled_patterns.discard(name)
        p = self._patterns.get(name)
        if p:
            p.enabled = True

    def disable_pattern(self, name: str) -> None:
        """Disable a specific pattern."""
        self._disabled_patterns.add(name)
        p = self._patterns.get(name)
        if p:
            p.enabled = False

    def enable_category(self, category: str) -> None:
        """Enable all patterns in a category."""
        self._disabled_categories.discard(category)
        for p in self._categories.get(category, []):
            p.enabled = True

    def disable_category(self, category: str) -> None:
        """Disable all patterns in a category."""
        self._disabled_categories.add(category)
        for p in self._categories.get(category, []):
            p.enabled = False

    def set_severity_filter(self, min_severity: Severity) -> None:
        """Only return patterns at or above this severity."""
        self._severity_filter = min_severity

    def clear_severity_filter(self) -> None:
        """Remove severity filter."""
        self._severity_filter = None

    def match_all(self, ir_node: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        """Run all enabled patterns against a node."""
        results: List[PatternMatch] = []
        for p in self._patterns.values():
            if not p.enabled:
                continue
            if p.name in self._disabled_patterns:
                continue
            if p.category in self._disabled_categories:
                continue
            try:
                matches = p.match(ir_node, abstract_state)
                results.extend(matches)
            except Exception:
                pass
        if self._severity_filter is not None:
            results = [m for m in results if not (m.severity < self._severity_filter)]
        return results

    def match_tree(self, root: IRNode, abstract_state: AbstractState) -> List[PatternMatch]:
        """Run all patterns on every node in the tree."""
        results: List[PatternMatch] = []
        results.extend(self.match_all(root, abstract_state))
        for child in root.iter_descendants():
            results.extend(self.match_all(child, abstract_state))
        return results

    def get_documentation(self) -> Dict[str, Dict[str, str]]:
        """Get documentation for all patterns grouped by category."""
        docs: Dict[str, Dict[str, str]] = {}
        for cat, patterns in self._categories.items():
            cat_docs: Dict[str, str] = {}
            for p in patterns:
                cat_docs[p.name] = (p.description
                                    or p.__class__.__doc__
                                    or "No description")
            docs[cat] = cat_docs
        return docs

    def summary(self) -> str:
        """Summary of the pattern database."""
        total = len(self._patterns)
        enabled = sum(1 for p in self._patterns.values() if p.enabled)
        cats = len(self._categories)
        lines = [
            f"PatternDatabase: {total} patterns ({enabled} enabled) "
            f"in {cats} categories"
        ]
        for cat in sorted(self._categories.keys()):
            pats = self._categories[cat]
            en = sum(1 for p in pats if p.enabled)
            lines.append(f"  {cat}: {len(pats)} patterns ({en} enabled)")
        return "\n".join(lines)


# ===================================================================
# PATTERN STATISTICS
# ===================================================================

class PatternStatistics:
    """Tracks statistics about pattern matching results."""

    def __init__(self) -> None:
        self._hit_counts: Dict[str, int] = {}
        self._category_counts: Dict[str, int] = {}
        self._severity_counts: Dict[Severity, int] = {s: 0 for s in Severity}
        self._confidence_counts: Dict[Confidence, int] = {c: 0 for c in Confidence}
        self._false_positives: Dict[str, int] = {}
        self._true_positives: Dict[str, int] = {}
        self._fix_suggested: Dict[str, int] = {}
        self._fix_accepted: Dict[str, int] = {}
        self._total_matches: int = 0
        self._total_nodes_scanned: int = 0
        self._file_counts: Dict[str, int] = {}
        self._matches: List[PatternMatch] = []

    def record_match(self, match: PatternMatch) -> None:
        """Record a pattern match."""
        self._total_matches += 1
        self._hit_counts[match.pattern_name] = (
            self._hit_counts.get(match.pattern_name, 0) + 1
        )
        self._category_counts[match.category] = (
            self._category_counts.get(match.category, 0) + 1
        )
        self._severity_counts[match.severity] = (
            self._severity_counts.get(match.severity, 0) + 1
        )
        self._confidence_counts[match.confidence] = (
            self._confidence_counts.get(match.confidence, 0) + 1
        )
        if match.fix_suggestion:
            self._fix_suggested[match.pattern_name] = (
                self._fix_suggested.get(match.pattern_name, 0) + 1
            )
        self._file_counts[match.location.file] = (
            self._file_counts.get(match.location.file, 0) + 1
        )
        self._matches.append(match)

    def record_matches(self, matches: List[PatternMatch]) -> None:
        """Record multiple matches at once."""
        for m in matches:
            self.record_match(m)

    def record_scan(self, node_count: int = 1) -> None:
        """Record nodes scanned."""
        self._total_nodes_scanned += node_count

    def mark_false_positive(self, pattern_name: str, count: int = 1) -> None:
        """Mark matches as false positives."""
        self._false_positives[pattern_name] = (
            self._false_positives.get(pattern_name, 0) + count
        )

    def mark_true_positive(self, pattern_name: str, count: int = 1) -> None:
        """Mark matches as true positives."""
        self._true_positives[pattern_name] = (
            self._true_positives.get(pattern_name, 0) + count
        )

    def mark_fix_accepted(self, pattern_name: str, count: int = 1) -> None:
        """Record that a fix suggestion was accepted."""
        self._fix_accepted[pattern_name] = (
            self._fix_accepted.get(pattern_name, 0) + count
        )

    @property
    def total_matches(self) -> int:
        return self._total_matches

    @property
    def total_nodes_scanned(self) -> int:
        return self._total_nodes_scanned

    def per_pattern_counts(self) -> Dict[str, int]:
        """Get hit counts per pattern."""
        return dict(self._hit_counts)

    def per_category_counts(self) -> Dict[str, int]:
        """Get hit counts per category."""
        return dict(self._category_counts)

    def per_severity_counts(self) -> Dict[Severity, int]:
        """Get hit counts per severity."""
        return dict(self._severity_counts)

    def per_confidence_counts(self) -> Dict[Confidence, int]:
        """Get hit counts per confidence."""
        return dict(self._confidence_counts)

    def per_file_counts(self) -> Dict[str, int]:
        """Get match counts per file."""
        return dict(self._file_counts)

    def false_positive_rate(self, pattern_name: str) -> float:
        """Estimated false positive rate for a pattern."""
        fp = self._false_positives.get(pattern_name, 0)
        tp = self._true_positives.get(pattern_name, 0)
        total = fp + tp
        if total == 0:
            return 0.0
        return fp / total

    def overall_false_positive_rate(self) -> float:
        """Overall false positive rate across all patterns."""
        total_fp = sum(self._false_positives.values())
        total_tp = sum(self._true_positives.values())
        total = total_fp + total_tp
        if total == 0:
            return 0.0
        return total_fp / total

    def fix_suggestion_acceptance_rate(
        self, pattern_name: Optional[str] = None
    ) -> float:
        """Rate at which fix suggestions are accepted."""
        if pattern_name:
            suggested = self._fix_suggested.get(pattern_name, 0)
            accepted = self._fix_accepted.get(pattern_name, 0)
        else:
            suggested = sum(self._fix_suggested.values())
            accepted = sum(self._fix_accepted.values())
        if suggested == 0:
            return 0.0
        return accepted / suggested

    def most_common_patterns(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the N most common patterns."""
        sorted_patterns = sorted(
            self._hit_counts.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_patterns[:n]

    def least_common_patterns(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the N least common patterns."""
        sorted_patterns = sorted(self._hit_counts.items(), key=lambda x: x[1])
        return sorted_patterns[:n]

    def hotspot_files(self, n: int = 10) -> List[Tuple[str, int]]:
        """Files with the most pattern matches."""
        sorted_files = sorted(
            self._file_counts.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_files[:n]

    def matches_by_severity(self, severity: Severity) -> List[PatternMatch]:
        """Get all matches of a given severity."""
        return [m for m in self._matches if m.severity == severity]

    def matches_by_category(self, category: str) -> List[PatternMatch]:
        """Get all matches of a given category."""
        return [m for m in self._matches if m.category == category]

    def matches_by_file(self, file_path: str) -> List[PatternMatch]:
        """Get all matches in a given file."""
        return [m for m in self._matches if m.location.file == file_path]

    def precision(self, pattern_name: str) -> float:
        """Precision = TP / (TP + FP)."""
        tp = self._true_positives.get(pattern_name, 0)
        fp = self._false_positives.get(pattern_name, 0)
        total = tp + fp
        if total == 0:
            return 0.0
        return tp / total

    def summary(self) -> str:
        """Summary statistics."""
        match_rate = self._total_matches / max(1, self._total_nodes_scanned)
        lines = [
            f"Pattern Matching Statistics:",
            f"  Total matches: {self._total_matches}",
            f"  Total nodes scanned: {self._total_nodes_scanned}",
            f"  Match rate: {match_rate:.4f}",
            f"  Unique patterns matched: {len(self._hit_counts)}",
            f"  Files with matches: {len(self._file_counts)}",
            f"",
            f"  By severity:",
        ]
        for sev in Severity:
            count = self._severity_counts.get(sev, 0)
            lines.append(f"    {sev.value}: {count}")
        lines.append(f"")
        lines.append(f"  By confidence:")
        for conf in Confidence:
            count = self._confidence_counts.get(conf, 0)
            lines.append(f"    {conf.value}: {count}")
        lines.append(f"")
        lines.append(f"  By category:")
        for cat, count in sorted(
            self._category_counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"    {cat}: {count}")
        top = self.most_common_patterns(5)
        if top:
            lines.append(f"")
            lines.append(f"  Top 5 patterns:")
            for name, count in top:
                lines.append(f"    {name}: {count}")
        fpr = self.overall_false_positive_rate()
        if fpr > 0:
            lines.append(f"")
            lines.append(f"  Overall false positive rate: {fpr:.2%}")
        far = self.fix_suggestion_acceptance_rate()
        if sum(self._fix_suggested.values()) > 0:
            lines.append(f"  Fix suggestion acceptance rate: {far:.2%}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self._hit_counts.clear()
        self._category_counts.clear()
        self._severity_counts = {s: 0 for s in Severity}
        self._confidence_counts = {c: 0 for c in Confidence}
        self._false_positives.clear()
        self._true_positives.clear()
        self._fix_suggested.clear()
        self._fix_accepted.clear()
        self._total_matches = 0
        self._total_nodes_scanned = 0
        self._file_counts.clear()
        self._matches.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize statistics to dict."""
        return {
            "total_matches": self._total_matches,
            "total_nodes_scanned": self._total_nodes_scanned,
            "per_pattern": dict(self._hit_counts),
            "per_category": dict(self._category_counts),
            "per_severity": {s.value: c for s, c in self._severity_counts.items()},
            "per_confidence": {c.value: n for c, n in self._confidence_counts.items()},
            "false_positives": dict(self._false_positives),
            "true_positives": dict(self._true_positives),
            "per_file": dict(self._file_counts),
            "fix_suggested": dict(self._fix_suggested),
            "fix_accepted": dict(self._fix_accepted),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PatternStatistics:
        """Deserialize from dict."""
        stats = cls()
        stats._total_matches = data.get("total_matches", 0)
        stats._total_nodes_scanned = data.get("total_nodes_scanned", 0)
        stats._hit_counts = data.get("per_pattern", {})
        stats._category_counts = data.get("per_category", {})
        for sev_str, count in data.get("per_severity", {}).items():
            try:
                stats._severity_counts[Severity(sev_str)] = count
            except ValueError:
                pass
        for conf_str, count in data.get("per_confidence", {}).items():
            try:
                stats._confidence_counts[Confidence(conf_str)] = count
            except ValueError:
                pass
        stats._false_positives = data.get("false_positives", {})
        stats._true_positives = data.get("true_positives", {})
        stats._file_counts = data.get("per_file", {})
        stats._fix_suggested = data.get("fix_suggested", {})
        stats._fix_accepted = data.get("fix_accepted", {})
        return stats


# ===================================================================
# CONVENIENCE: run_all_patterns
# ===================================================================

def run_all_patterns(
    root: IRNode,
    abstract_state: AbstractState,
    *,
    categories: Optional[Set[str]] = None,
    min_severity: Optional[Severity] = None,
    min_confidence: Optional[Confidence] = None,
    disabled_patterns: Optional[Set[str]] = None,
) -> Tuple[List[PatternMatch], PatternStatistics]:
    """Run all pattern matchers on an IR tree and return matches + stats."""
    db = PatternDatabase()
    stats = PatternStatistics()

    if disabled_patterns:
        for p in disabled_patterns:
            db.disable_pattern(p)

    if categories:
        for cat in db.get_categories():
            if cat not in categories:
                db.disable_category(cat)

    if min_severity:
        db.set_severity_filter(min_severity)

    all_nodes = [root] + list(root.iter_descendants())
    stats.record_scan(len(all_nodes))

    all_matches: List[PatternMatch] = []
    for node in all_nodes:
        matches = db.match_all(node, abstract_state)
        all_matches.extend(matches)

    if min_confidence:
        conf_order = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
        min_conf_val = conf_order.get(min_confidence, 0)
        all_matches = [
            m for m in all_matches
            if conf_order.get(m.confidence, 0) >= min_conf_val
        ]

    # Deduplicate by uid
    seen: Set[str] = set()
    deduped: List[PatternMatch] = []
    for m in all_matches:
        if m.uid not in seen:
            seen.add(m.uid)
            deduped.append(m)

    stats.record_matches(deduped)
    return deduped, stats
