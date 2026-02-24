from __future__ import annotations

"""
conftest.py – Pytest configuration and fixtures for the refinement-type
inference system targeting dynamically-typed languages via CEGAR.

Every type used by the test-suite is defined locally so this module has
zero coupling to the rest of the project.
"""

import enum
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pytest

# ---------------------------------------------------------------------------
# 1. Locally-defined dataclasses / classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceLocation:
    """A location in source code (file, line, column)."""

    file: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"

    def offset(self, lines: int = 0, cols: int = 0) -> SourceLocation:
        """Return a new location offset by the given lines/cols."""
        return SourceLocation(self.file, self.line + lines, self.col + cols)


@dataclass(frozen=True)
class SSAVar:
    """A variable in SSA form."""

    name: str
    version: int
    source_loc: Optional[SourceLocation] = None

    def __str__(self) -> str:
        return f"{self.name}_{self.version}"

    def next_version(self) -> SSAVar:
        """Return a new SSAVar with an incremented version."""
        return SSAVar(self.name, self.version + 1, self.source_loc)


@dataclass
class SSAInstruction:
    """A single SSA instruction."""

    opcode: str
    operands: List[SSAVar]
    result: Optional[SSAVar]
    source_loc: Optional[SourceLocation] = None

    def __str__(self) -> str:
        ops = ", ".join(str(o) for o in self.operands)
        res = f"{self.result} = " if self.result else ""
        return f"{res}{self.opcode}({ops})"

    def uses(self) -> List[SSAVar]:
        """Variables read by this instruction."""
        return list(self.operands)

    def defs(self) -> List[SSAVar]:
        """Variables defined by this instruction."""
        return [self.result] if self.result else []


@dataclass
class BasicBlock:
    """A basic block in a control-flow graph."""

    label: str
    instructions: List[SSAInstruction] = field(default_factory=list)
    terminator: Optional[SSAInstruction] = None
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    def add_instruction(self, inst: SSAInstruction) -> None:
        self.instructions.append(inst)

    def variables_defined(self) -> Set[str]:
        """All variable names defined in this block."""
        result: Set[str] = set()
        for inst in self.instructions:
            for d in inst.defs():
                result.add(d.name)
        return result

    def variables_used(self) -> Set[str]:
        """All variable names used in this block."""
        result: Set[str] = set()
        for inst in self.instructions:
            for u in inst.uses():
                result.add(u.name)
        return result

    def is_empty(self) -> bool:
        return len(self.instructions) == 0 and self.terminator is None


@dataclass
class CFG:
    """Control-flow graph."""

    entry: str
    exit: str
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)

    def add_block(self, block: BasicBlock) -> None:
        self.blocks[block.label] = block

    def get_block(self, label: str) -> BasicBlock:
        return self.blocks[label]

    def predecessors(self, label: str) -> List[str]:
        return self.blocks[label].predecessors

    def successors(self, label: str) -> List[str]:
        return self.blocks[label].successors

    def postorder(self) -> List[str]:
        """Return labels in reverse-postorder (useful for dataflow)."""
        visited: Set[str] = set()
        order: List[str] = []

        def _dfs(lbl: str) -> None:
            if lbl in visited:
                return
            visited.add(lbl)
            for s in self.successors(lbl):
                _dfs(s)
            order.append(lbl)

        _dfs(self.entry)
        return order

    def reverse_postorder(self) -> List[str]:
        return list(reversed(self.postorder()))

    def dominators(self) -> Dict[str, Set[str]]:
        """Compute dominator sets for every block."""
        all_labels = set(self.blocks.keys())
        dom: Dict[str, Set[str]] = {lbl: set(all_labels) for lbl in all_labels}
        dom[self.entry] = {self.entry}
        changed = True
        while changed:
            changed = False
            for lbl in self.reverse_postorder():
                if lbl == self.entry:
                    continue
                preds = self.predecessors(lbl)
                if not preds:
                    new_dom = {lbl}
                else:
                    new_dom = set.intersection(*(dom[p] for p in preds)) | {lbl}
                if new_dom != dom[lbl]:
                    dom[lbl] = new_dom
                    changed = True
        return dom


@dataclass
class Function:
    """A function in the intermediate representation."""

    name: str
    params: List[SSAVar]
    cfg: CFG
    return_type: Optional[str] = None

    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    def all_vars(self) -> Set[str]:
        """Collect every variable name mentioned anywhere in the function."""
        result: Set[str] = set()
        for p in self.params:
            result.add(p.name)
        for blk in self.cfg.blocks.values():
            result |= blk.variables_defined()
            result |= blk.variables_used()
        return result


@dataclass
class Module:
    """A translation unit / module."""

    name: str
    functions: List[Function] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def function_names(self) -> List[str]:
        return [f.name for f in self.functions]

    def lookup(self, name: str) -> Optional[Function]:
        for fn in self.functions:
            if fn.name == name:
                return fn
        return None


# -- abstract domains -------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """Integer interval abstract domain element [low, high].

    ``low > high`` represents the bottom (empty) element.
    ``low == -inf, high == +inf`` represents top.
    """

    low: float
    high: float

    # sentinels
    NEG_INF: float = field(default=float("-inf"), init=False, repr=False)
    POS_INF: float = field(default=float("inf"), init=False, repr=False)

    def __post_init__(self) -> None:
        # We allow construction of bottom via low>high.
        pass

    @property
    def is_bottom(self) -> bool:
        return self.low > self.high

    @property
    def is_top(self) -> bool:
        return self.low == float("-inf") and self.high == float("inf")

    def contains(self, value: float) -> bool:
        """Check whether *value* is contained in this interval."""
        if self.is_bottom:
            return False
        return self.low <= value <= self.high

    def join(self, other: Interval) -> Interval:
        """Least upper bound."""
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return Interval(min(self.low, other.low), max(self.high, other.high))

    def meet(self, other: Interval) -> Interval:
        """Greatest lower bound."""
        if self.is_bottom or other.is_bottom:
            return Interval(1, 0)  # bottom
        lo = max(self.low, other.low)
        hi = min(self.high, other.high)
        return Interval(lo, hi)

    def widen(self, other: Interval) -> Interval:
        """Standard widening operator for intervals."""
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        lo = self.low if other.low >= self.low else float("-inf")
        hi = self.high if other.high <= self.high else float("inf")
        return Interval(lo, hi)

    def narrow(self, other: Interval) -> Interval:
        """Narrowing operator (dual of widen)."""
        if self.is_bottom:
            return self
        if other.is_bottom:
            return other
        lo = other.low if self.low == float("-inf") else self.low
        hi = other.high if self.high == float("inf") else self.high
        return Interval(lo, hi)

    def add(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval(1, 0)
        return Interval(self.low + other.low, self.high + other.high)

    def sub(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval(1, 0)
        return Interval(self.low - other.high, self.high - other.low)

    def __str__(self) -> str:
        if self.is_bottom:
            return "⊥"
        lo = "-∞" if self.low == float("-inf") else str(int(self.low))
        hi = "+∞" if self.high == float("inf") else str(int(self.high))
        return f"[{lo}, {hi}]"


class TypeTag(enum.Enum):
    """Enumeration of runtime type tags for a dynamically-typed language."""

    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    NONE = "none"
    LIST = "list"
    DICT = "dict"
    TUPLE = "tuple"
    SET = "set"
    OBJECT = "object"
    CALLABLE = "callable"
    ANY = "any"


@dataclass(frozen=True)
class TypeTagSet:
    """A set of possible runtime type-tags (powerset domain)."""

    tags: FrozenSet[TypeTag]

    def union(self, other: TypeTagSet) -> TypeTagSet:
        return TypeTagSet(self.tags | other.tags)

    def intersect(self, other: TypeTagSet) -> TypeTagSet:
        return TypeTagSet(self.tags & other.tags)

    def subtract(self, other: TypeTagSet) -> TypeTagSet:
        return TypeTagSet(self.tags - other.tags)

    def is_bottom(self) -> bool:
        return len(self.tags) == 0

    def is_top(self) -> bool:
        return TypeTag.ANY in self.tags or self.tags == frozenset(TypeTag)

    def is_singleton(self) -> bool:
        return len(self.tags) == 1

    def contains(self, tag: TypeTag) -> bool:
        if TypeTag.ANY in self.tags:
            return True
        return tag in self.tags

    def __len__(self) -> int:
        return len(self.tags)

    def __iter__(self) -> Iterator[TypeTag]:
        return iter(self.tags)


class NullityState(enum.Enum):
    """Nullity (None-ness) abstract domain."""

    NULL = "null"
    NOT_NULL = "not_null"
    MAYBE_NULL = "maybe_null"
    BOTTOM = "bottom"

    def join(self, other: NullityState) -> NullityState:
        if self == NullityState.BOTTOM:
            return other
        if other == NullityState.BOTTOM:
            return self
        if self == other:
            return self
        return NullityState.MAYBE_NULL

    def meet(self, other: NullityState) -> NullityState:
        if self == NullityState.MAYBE_NULL:
            return other
        if other == NullityState.MAYBE_NULL:
            return self
        if self == other:
            return self
        return NullityState.BOTTOM


class PredicateKind(enum.Enum):
    """Kinds of refinement predicates."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    ISINSTANCE = "isinstance"
    HASATTR = "hasattr"
    IS_NONE = "is_none"
    IS_NOT_NONE = "is_not_none"
    TRUTHINESS = "truthiness"


@dataclass(frozen=True)
class Predicate:
    """A refinement predicate over SSA variables."""

    kind: PredicateKind
    lhs: str
    rhs: Optional[str] = None
    negated: bool = False

    def negate(self) -> Predicate:
        """Return the negation of this predicate."""
        return Predicate(self.kind, self.lhs, self.rhs, not self.negated)

    def substitute(self, mapping: Dict[str, str]) -> Predicate:
        """Apply a variable-name substitution."""
        new_lhs = mapping.get(self.lhs, self.lhs)
        new_rhs = mapping.get(self.rhs, self.rhs) if self.rhs else self.rhs
        return Predicate(self.kind, new_lhs, new_rhs, self.negated)

    def __str__(self) -> str:
        neg = "¬" if self.negated else ""
        rhs_str = f", {self.rhs}" if self.rhs else ""
        return f"{neg}{self.kind.value}({self.lhs}{rhs_str})"

    def implies(self, other: Predicate) -> bool:
        """Conservative check: does *self* logically imply *other*?"""
        if self == other:
            return True
        if self.kind == other.kind and self.lhs == other.lhs and self.rhs == other.rhs:
            if self.negated and not other.negated:
                return False
            if not self.negated and other.negated:
                return False
        # LT implies LE
        if (
            self.kind == PredicateKind.LT
            and other.kind == PredicateKind.LE
            and self.lhs == other.lhs
            and self.rhs == other.rhs
            and self.negated == other.negated
        ):
            return True
        # IS_NONE implies not IS_NOT_NONE (and vice-versa)
        if (
            self.kind == PredicateKind.IS_NONE
            and other.kind == PredicateKind.IS_NOT_NONE
            and self.lhs == other.lhs
            and self.negated != other.negated
        ):
            return True
        return False


@dataclass
class FunctionSummary:
    """Summary for inter-procedural analysis."""

    name: str
    preconditions: List[Predicate] = field(default_factory=list)
    postconditions: List[Predicate] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)

    def is_pure(self) -> bool:
        return len(self.effects) == 0

    def satisfied_by(self, state: Dict[str, Any]) -> bool:
        """Check whether *state* satisfies all preconditions (very rough)."""
        for pre in self.preconditions:
            val = state.get(pre.lhs)
            if val is None:
                return False
        return True


@dataclass
class Contract:
    """Design-by-contract specification."""

    requires: List[Predicate] = field(default_factory=list)
    ensures: List[Predicate] = field(default_factory=list)
    invariants: List[Predicate] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.requires and not self.ensures and not self.invariants

    def merge(self, other: Contract) -> Contract:
        return Contract(
            requires=self.requires + other.requires,
            ensures=self.ensures + other.ensures,
            invariants=self.invariants + other.invariants,
        )


class BugKind(enum.Enum):
    """Classification of detected bugs."""

    TYPE_ERROR = "type_error"
    NULL_DEREF = "null_deref"
    INDEX_OOB = "index_oob"
    DIV_ZERO = "div_zero"
    ATTR_ERROR = "attr_error"
    UNREACHABLE = "unreachable"


@dataclass
class Bug:
    """A single bug report."""

    location: SourceLocation
    kind: BugKind
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.kind.value} at {self.location}: {self.message}"


@dataclass
class AnalysisResult:
    """Aggregate result of running the analysis."""

    bugs: List[Bug] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    inferred_types: Dict[str, TypeTagSet] = field(default_factory=dict)
    contracts: Dict[str, Contract] = field(default_factory=dict)

    def has_bugs(self) -> bool:
        return len(self.bugs) > 0

    def bugs_of_kind(self, kind: BugKind) -> List[Bug]:
        return [b for b in self.bugs if b.kind == kind]

    def type_of(self, var: str) -> Optional[TypeTagSet]:
        return self.inferred_types.get(var)


# ---------------------------------------------------------------------------
# 2. Pytest fixtures – IR nodes
# ---------------------------------------------------------------------------


@pytest.fixture
def source_loc() -> SourceLocation:
    """A default source location for tests."""
    return SourceLocation("test.py", 1, 0)


@pytest.fixture
def ssa_var(source_loc: SourceLocation) -> SSAVar:
    """A default SSA variable ``x_0``."""
    return SSAVar("x", 0, source_loc)


@pytest.fixture
def basic_block(source_loc: SourceLocation) -> BasicBlock:
    """A basic block with one assignment instruction."""
    var_x = SSAVar("x", 0, source_loc)
    var_y = SSAVar("y", 0, source_loc)
    inst = SSAInstruction("add", [var_x, var_y], SSAVar("z", 0, source_loc), source_loc)
    blk = BasicBlock(label="bb0", instructions=[inst])
    return blk


@pytest.fixture
def simple_cfg(source_loc: SourceLocation) -> CFG:
    """A minimal CFG with entry → body → exit."""
    entry = BasicBlock("entry", successors=["body"])
    body_var = SSAVar("x", 0, source_loc)
    body_inst = SSAInstruction("const", [], body_var, source_loc)
    body = BasicBlock("body", instructions=[body_inst], predecessors=["entry"], successors=["exit"])
    exit_blk = BasicBlock("exit", predecessors=["body"])
    cfg = CFG(entry="entry", exit="exit")
    cfg.add_block(entry)
    cfg.add_block(body)
    cfg.add_block(exit_blk)
    return cfg


@pytest.fixture
def sample_function(simple_cfg: CFG, source_loc: SourceLocation) -> Function:
    """A sample function wrapping the simple CFG."""
    param = SSAVar("arg", 0, source_loc)
    return Function(name="sample_fn", params=[param], cfg=simple_cfg, return_type="int")


@pytest.fixture
def sample_module(sample_function: Function) -> Module:
    """A module containing the sample function."""
    return Module(name="test_module", functions=[sample_function], imports=["builtins"])


# ---------------------------------------------------------------------------
# 3. Fixtures – abstract domain values
# ---------------------------------------------------------------------------


@pytest.fixture
def zero_interval() -> Interval:
    """The interval [0, 0]."""
    return Interval(0, 0)


@pytest.fixture
def positive_interval() -> Interval:
    """The interval [1, +∞)."""
    return Interval(1, float("inf"))


@pytest.fixture
def negative_interval() -> Interval:
    """The interval (-∞, -1]."""
    return Interval(float("-inf"), -1)


@pytest.fixture
def full_interval() -> Interval:
    """The interval (-∞, +∞) (top)."""
    return Interval(float("-inf"), float("inf"))


@pytest.fixture
def empty_interval() -> Interval:
    """The bottom interval (empty set)."""
    return Interval(1, 0)


@pytest.fixture
def int_type_tag() -> TypeTagSet:
    """A singleton {INT}."""
    return TypeTagSet(frozenset({TypeTag.INT}))


@pytest.fixture
def str_type_tag() -> TypeTagSet:
    """A singleton {STR}."""
    return TypeTagSet(frozenset({TypeTag.STR}))


@pytest.fixture
def numeric_type_tags() -> TypeTagSet:
    """The set {INT, FLOAT}."""
    return TypeTagSet(frozenset({TypeTag.INT, TypeTag.FLOAT}))


@pytest.fixture
def all_type_tags() -> TypeTagSet:
    """All concrete type tags (excluding ANY)."""
    concrete = frozenset(t for t in TypeTag if t != TypeTag.ANY)
    return TypeTagSet(concrete)


@pytest.fixture
def null_state() -> NullityState:
    return NullityState.NULL


@pytest.fixture
def not_null_state() -> NullityState:
    return NullityState.NOT_NULL


@pytest.fixture
def maybe_null_state() -> NullityState:
    return NullityState.MAYBE_NULL


# ---------------------------------------------------------------------------
# 4. Fixtures – predicate templates
# ---------------------------------------------------------------------------


@pytest.fixture
def eq_predicate() -> Predicate:
    """x == y"""
    return Predicate(PredicateKind.EQ, "x", "y")


@pytest.fixture
def lt_predicate() -> Predicate:
    """x < y"""
    return Predicate(PredicateKind.LT, "x", "y")


@pytest.fixture
def isinstance_predicate() -> Predicate:
    """isinstance(x, int)"""
    return Predicate(PredicateKind.ISINSTANCE, "x", "int")


@pytest.fixture
def is_none_predicate() -> Predicate:
    """x is None"""
    return Predicate(PredicateKind.IS_NONE, "x")


@pytest.fixture
def hasattr_predicate() -> Predicate:
    """hasattr(x, 'foo')"""
    return Predicate(PredicateKind.HASATTR, "x", "foo")


# ---------------------------------------------------------------------------
# 5. Fixtures – function summaries & contracts
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_summary() -> FunctionSummary:
    """A pure function summary with no preconditions."""
    return FunctionSummary(
        name="identity",
        preconditions=[],
        postconditions=[Predicate(PredicateKind.EQ, "result", "arg")],
        effects=[],
    )


@pytest.fixture
def guarded_summary() -> FunctionSummary:
    """A summary with a type-guard precondition."""
    return FunctionSummary(
        name="safe_div",
        preconditions=[
            Predicate(PredicateKind.ISINSTANCE, "a", "int"),
            Predicate(PredicateKind.ISINSTANCE, "b", "int"),
            Predicate(PredicateKind.NE, "b", "0"),
        ],
        postconditions=[Predicate(PredicateKind.ISINSTANCE, "result", "int")],
        effects=[],
    )


@pytest.fixture
def sample_contract() -> Contract:
    """A contract for a list-index operation."""
    return Contract(
        requires=[
            Predicate(PredicateKind.ISINSTANCE, "lst", "list"),
            Predicate(PredicateKind.GE, "idx", "0"),
        ],
        ensures=[Predicate(PredicateKind.IS_NOT_NONE, "result")],
        invariants=[Predicate(PredicateKind.GE, "len(lst)", "1")],
    )


# ---------------------------------------------------------------------------
# 6. Fixtures – sample Python source-code strings
# ---------------------------------------------------------------------------


@pytest.fixture
def python_simple_function() -> str:
    """A trivial Python function."""
    return textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            return a + b
    """)


@pytest.fixture
def python_guard_function() -> str:
    """Python function with an isinstance guard."""
    return textwrap.dedent("""\
        def safe_len(x):
            if isinstance(x, (list, tuple, str)):
                return len(x)
            return -1
    """)


@pytest.fixture
def python_class() -> str:
    """A simple Python class."""
    return textwrap.dedent("""\
        class Point:
            def __init__(self, x: float, y: float) -> None:
                self.x = x
                self.y = y

            def distance(self, other: 'Point') -> float:
                dx = self.x - other.x
                dy = self.y - other.y
                return (dx * dx + dy * dy) ** 0.5
    """)


@pytest.fixture
def python_loop() -> str:
    """Python code with a while loop and invariant."""
    return textwrap.dedent("""\
        def sum_to(n: int) -> int:
            total = 0
            i = 0
            while i < n:
                total += i
                i += 1
            return total
    """)


# ---------------------------------------------------------------------------
# 7. Fixtures – sample TypeScript source-code strings
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_simple_function() -> str:
    return textwrap.dedent("""\
        function add(a: number, b: number): number {
            return a + b;
        }
    """)


@pytest.fixture
def ts_interface() -> str:
    return textwrap.dedent("""\
        interface Shape {
            kind: string;
            area(): number;
        }

        interface Circle extends Shape {
            kind: "circle";
            radius: number;
        }

        interface Rectangle extends Shape {
            kind: "rectangle";
            width: number;
            height: number;
        }
    """)


@pytest.fixture
def ts_generic() -> str:
    return textwrap.dedent("""\
        function identity<T>(arg: T): T {
            return arg;
        }

        function map<T, U>(arr: T[], fn: (x: T) => U): U[] {
            const result: U[] = [];
            for (const item of arr) {
                result.push(fn(item));
            }
            return result;
        }
    """)


@pytest.fixture
def ts_guard() -> str:
    return textwrap.dedent("""\
        function isString(x: unknown): x is string {
            return typeof x === "string";
        }

        function process(input: string | number) {
            if (isString(input)) {
                console.log(input.toUpperCase());
            } else {
                console.log(input.toFixed(2));
            }
        }
    """)


# ---------------------------------------------------------------------------
# 8. SamplePrograms – static methods returning source strings
# ---------------------------------------------------------------------------


class SamplePrograms:
    """Library of sample programs used across the test suite."""

    # -- Python samples -----------------------------------------------------

    @staticmethod
    def simple_function() -> str:
        return textwrap.dedent("""\
            def greet(name: str) -> str:
                return "Hello, " + name
        """)

    @staticmethod
    def function_with_guard() -> str:
        return textwrap.dedent("""\
            def process(x):
                if isinstance(x, int):
                    return x + 1
                elif isinstance(x, str):
                    return len(x)
                else:
                    raise TypeError("unsupported")
        """)

    @staticmethod
    def function_with_loop() -> str:
        return textwrap.dedent("""\
            def factorial(n: int) -> int:
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result
        """)

    @staticmethod
    def null_check_pattern() -> str:
        return textwrap.dedent("""\
            def safe_access(obj, key):
                val = obj.get(key)
                if val is None:
                    return "default"
                return val.strip()
        """)

    @staticmethod
    def isinstance_pattern() -> str:
        return textwrap.dedent("""\
            def describe(value):
                if isinstance(value, bool):
                    return "boolean"
                elif isinstance(value, int):
                    return "integer"
                elif isinstance(value, float):
                    return "float"
                elif isinstance(value, str):
                    return "string"
                return "unknown"
        """)

    @staticmethod
    def hasattr_pattern() -> str:
        return textwrap.dedent("""\
            def call_if_possible(obj, method_name):
                if hasattr(obj, method_name):
                    func = getattr(obj, method_name)
                    if callable(func):
                        return func()
                return None
        """)

    @staticmethod
    def list_index_pattern() -> str:
        return textwrap.dedent("""\
            def safe_get(lst, idx):
                if 0 <= idx < len(lst):
                    return lst[idx]
                return None
        """)

    @staticmethod
    def dict_access_pattern() -> str:
        return textwrap.dedent("""\
            def get_nested(data, *keys):
                current = data
                for key in keys:
                    if not isinstance(current, dict):
                        return None
                    current = current.get(key)
                    if current is None:
                        return None
                return current
        """)

    @staticmethod
    def division_pattern() -> str:
        return textwrap.dedent("""\
            def safe_divide(a, b):
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                return a / b
        """)

    @staticmethod
    def nested_conditions() -> str:
        return textwrap.dedent("""\
            def classify(x, y):
                if x > 0:
                    if y > 0:
                        return "first_quadrant"
                    elif y < 0:
                        return "fourth_quadrant"
                    else:
                        return "positive_x_axis"
                elif x < 0:
                    if y > 0:
                        return "second_quadrant"
                    elif y < 0:
                        return "third_quadrant"
                    else:
                        return "negative_x_axis"
                else:
                    if y > 0:
                        return "positive_y_axis"
                    elif y < 0:
                        return "negative_y_axis"
                    else:
                        return "origin"
        """)

    @staticmethod
    def loop_with_invariant() -> str:
        return textwrap.dedent("""\
            def binary_search(arr, target):
                lo, hi = 0, len(arr) - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return -1
        """)

    @staticmethod
    def recursive_function() -> str:
        return textwrap.dedent("""\
            def gcd(a: int, b: int) -> int:
                if b == 0:
                    return a
                return gcd(b, a % b)
        """)

    @staticmethod
    def class_with_methods() -> str:
        return textwrap.dedent("""\
            class Stack:
                def __init__(self):
                    self._items = []

                def push(self, item):
                    self._items.append(item)

                def pop(self):
                    if not self._items:
                        raise IndexError("pop from empty stack")
                    return self._items.pop()

                def peek(self):
                    if not self._items:
                        return None
                    return self._items[-1]

                def is_empty(self) -> bool:
                    return len(self._items) == 0

                def size(self) -> int:
                    return len(self._items)
        """)

    @staticmethod
    def closure_pattern() -> str:
        return textwrap.dedent("""\
            def make_adder(n):
                def adder(x):
                    return x + n
                return adder

            def make_counter(start=0):
                count = start
                def increment():
                    nonlocal count
                    count += 1
                    return count
                return increment
        """)

    @staticmethod
    def generator_pattern() -> str:
        return textwrap.dedent("""\
            def fibonacci():
                a, b = 0, 1
                while True:
                    yield a
                    a, b = b, a + b

            def take(n, gen):
                result = []
                for _ in range(n):
                    result.append(next(gen))
                return result
        """)

    @staticmethod
    def async_function() -> str:
        return textwrap.dedent("""\
            import asyncio

            async def fetch_data(url: str) -> dict:
                await asyncio.sleep(0.1)
                return {"url": url, "status": 200}

            async def fetch_all(urls: list) -> list:
                tasks = [fetch_data(u) for u in urls]
                return await asyncio.gather(*tasks)
        """)

    @staticmethod
    def context_manager() -> str:
        return textwrap.dedent("""\
            class ManagedResource:
                def __init__(self, name):
                    self.name = name
                    self.opened = False

                def __enter__(self):
                    self.opened = True
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.opened = False
                    return False

            def use_resource(name):
                with ManagedResource(name) as r:
                    assert r.opened
                    return r.name
        """)

    @staticmethod
    def exception_handling() -> str:
        return textwrap.dedent("""\
            def parse_int(s):
                try:
                    return int(s)
                except ValueError:
                    return None
                except TypeError:
                    return None

            def safe_compute(a, b, op):
                try:
                    if op == "+":
                        return a + b
                    elif op == "/":
                        return a / b
                    else:
                        raise ValueError(f"unknown op: {op}")
                except ZeroDivisionError:
                    return float('inf')
                except TypeError as e:
                    raise RuntimeError(f"type error: {e}") from e
        """)

    @staticmethod
    def type_narrowing_chain() -> str:
        return textwrap.dedent("""\
            def narrow(x):
                if x is None:
                    return "none"
                if isinstance(x, bool):
                    return "bool"
                if isinstance(x, int):
                    if x > 0:
                        return "positive_int"
                    elif x < 0:
                        return "negative_int"
                    return "zero"
                if isinstance(x, str):
                    if len(x) == 0:
                        return "empty_str"
                    return "nonempty_str"
                return "other"
        """)

    @staticmethod
    def multiple_returns() -> str:
        return textwrap.dedent("""\
            def find_first(lst, predicate):
                for idx, item in enumerate(lst):
                    if predicate(item):
                        return idx, item
                return -1, None
        """)

    @staticmethod
    def default_arguments() -> str:
        return textwrap.dedent("""\
            def connect(host, port=8080, timeout=30, retries=3):
                attempts = 0
                while attempts < retries:
                    try:
                        return {"host": host, "port": port, "connected": True}
                    except Exception:
                        attempts += 1
                return {"host": host, "port": port, "connected": False}
        """)

    @staticmethod
    def property_access() -> str:
        return textwrap.dedent("""\
            class User:
                def __init__(self, name, email=None):
                    self._name = name
                    self._email = email

                @property
                def name(self):
                    return self._name

                @property
                def email(self):
                    return self._email

                @email.setter
                def email(self, value):
                    if not isinstance(value, str) or "@" not in value:
                        raise ValueError("invalid email")
                    self._email = value
        """)

    @staticmethod
    def method_chaining() -> str:
        return textwrap.dedent("""\
            class QueryBuilder:
                def __init__(self):
                    self._table = None
                    self._conditions = []
                    self._limit = None

                def from_table(self, table):
                    self._table = table
                    return self

                def where(self, condition):
                    self._conditions.append(condition)
                    return self

                def limit(self, n):
                    self._limit = n
                    return self

                def build(self):
                    if self._table is None:
                        raise ValueError("no table specified")
                    q = f"SELECT * FROM {self._table}"
                    if self._conditions:
                        q += " WHERE " + " AND ".join(self._conditions)
                    if self._limit is not None:
                        q += f" LIMIT {self._limit}"
                    return q
        """)

    @staticmethod
    def comprehension_patterns() -> str:
        return textwrap.dedent("""\
            def transform(data):
                squares = [x * x for x in data if isinstance(x, (int, float))]
                unique = {x for x in squares}
                indexed = {i: v for i, v in enumerate(squares)}
                gen = (x for x in squares if x > 10)
                return squares, unique, indexed, list(gen)
        """)

    @staticmethod
    def decorator_pattern() -> str:
        return textwrap.dedent("""\
            import functools

            def retry(max_attempts=3):
                def decorator(func):
                    @functools.wraps(func)
                    def wrapper(*args, **kwargs):
                        last_exc = None
                        for attempt in range(max_attempts):
                            try:
                                return func(*args, **kwargs)
                            except Exception as e:
                                last_exc = e
                        raise last_exc
                    return wrapper
                return decorator
        """)

    @staticmethod
    def metaclass_pattern() -> str:
        return textwrap.dedent("""\
            class SingletonMeta(type):
                _instances = {}

                def __call__(cls, *args, **kwargs):
                    if cls not in cls._instances:
                        instance = super().__call__(*args, **kwargs)
                        cls._instances[cls] = instance
                    return cls._instances[cls]

            class Database(metaclass=SingletonMeta):
                def __init__(self, url="sqlite://"):
                    self.url = url
                    self.connected = False
        """)

    @staticmethod
    def mixin_pattern() -> str:
        return textwrap.dedent("""\
            class JsonMixin:
                def to_json(self):
                    import json
                    return json.dumps(self.__dict__)

            class LogMixin:
                def log(self, msg):
                    print(f"[{self.__class__.__name__}] {msg}")

            class Service(JsonMixin, LogMixin):
                def __init__(self, name):
                    self.name = name

                def run(self):
                    self.log("starting")
                    return self.to_json()
        """)

    @staticmethod
    def overloaded_function() -> str:
        return textwrap.dedent("""\
            from typing import overload, Union

            @overload
            def process(x: int) -> int: ...
            @overload
            def process(x: str) -> str: ...

            def process(x):
                if isinstance(x, int):
                    return x * 2
                elif isinstance(x, str):
                    return x.upper()
                raise TypeError
        """)

    @staticmethod
    def variadic_function() -> str:
        return textwrap.dedent("""\
            def merge(*dicts):
                result = {}
                for d in dicts:
                    if not isinstance(d, dict):
                        raise TypeError(f"expected dict, got {type(d).__name__}")
                    result.update(d)
                return result
        """)

    @staticmethod
    def keyword_only() -> str:
        return textwrap.dedent("""\
            def create_user(name, *, email=None, admin=False, active=True):
                user = {"name": name, "admin": admin, "active": active}
                if email is not None:
                    user["email"] = email
                return user
        """)

    @staticmethod
    def walrus_operator() -> str:
        return textwrap.dedent("""\
            def find_match(items, predicate):
                results = []
                for item in items:
                    if (processed := predicate(item)) is not None:
                        results.append(processed)
                return results
        """)

    @staticmethod
    def match_statement() -> str:
        return textwrap.dedent("""\
            def handle_command(command):
                match command.split():
                    case ["quit"]:
                        return "quitting"
                    case ["go", direction]:
                        return f"going {direction}"
                    case ["get", item] if item != "nothing":
                        return f"getting {item}"
                    case _:
                        return "unknown command"
        """)

    @staticmethod
    def f_string_complex() -> str:
        return textwrap.dedent("""\
            def format_table(headers, rows):
                widths = [max(len(str(r[i])) for r in rows + [headers])
                          for i in range(len(headers))]
                fmt = " | ".join(f"{{:<{w}}}" for w in widths)
                lines = [fmt.format(*headers)]
                lines.append("-+-".join("-" * w for w in widths))
                for row in rows:
                    lines.append(fmt.format(*row))
                return "\\n".join(lines)
        """)

    @staticmethod
    def dataclass_usage() -> str:
        return textwrap.dedent("""\
            from dataclasses import dataclass, field

            @dataclass
            class Config:
                host: str = "localhost"
                port: int = 8080
                debug: bool = False
                tags: list = field(default_factory=list)

                def url(self) -> str:
                    return f"http://{self.host}:{self.port}"
        """)

    @staticmethod
    def enum_usage() -> str:
        return textwrap.dedent("""\
            from enum import Enum, auto

            class Color(Enum):
                RED = auto()
                GREEN = auto()
                BLUE = auto()

            def mix(a: Color, b: Color) -> str:
                if {a, b} == {Color.RED, Color.BLUE}:
                    return "purple"
                elif {a, b} == {Color.RED, Color.GREEN}:
                    return "yellow"
                elif {a, b} == {Color.GREEN, Color.BLUE}:
                    return "cyan"
                return a.name.lower()
        """)

    @staticmethod
    def protocol_usage() -> str:
        return textwrap.dedent("""\
            from typing import Protocol, runtime_checkable

            @runtime_checkable
            class Drawable(Protocol):
                def draw(self) -> str: ...

            class Circle:
                def __init__(self, radius: float):
                    self.radius = radius
                def draw(self) -> str:
                    return f"Circle(r={self.radius})"

            def render(shape: Drawable) -> str:
                return shape.draw()
        """)

    # -- TypeScript samples -------------------------------------------------

    @staticmethod
    def interface_pattern() -> str:
        return textwrap.dedent("""\
            interface Animal {
                name: string;
                sound(): string;
            }

            interface Dog extends Animal {
                breed: string;
                fetch(item: string): boolean;
            }

            function greetAnimal(a: Animal): string {
                return `Hello, ${a.name}! You say ${a.sound()}`;
            }
        """)

    @staticmethod
    def generic_function() -> str:
        return textwrap.dedent("""\
            function first<T>(arr: T[]): T | undefined {
                return arr.length > 0 ? arr[0] : undefined;
            }

            function zip<A, B>(as: A[], bs: B[]): [A, B][] {
                const len = Math.min(as.length, bs.length);
                const result: [A, B][] = [];
                for (let i = 0; i < len; i++) {
                    result.push([as[i], bs[i]]);
                }
                return result;
            }
        """)

    @staticmethod
    def discriminated_union() -> str:
        return textwrap.dedent("""\
            type Shape =
                | { kind: "circle"; radius: number }
                | { kind: "rectangle"; width: number; height: number }
                | { kind: "triangle"; base: number; height: number };

            function area(shape: Shape): number {
                switch (shape.kind) {
                    case "circle":
                        return Math.PI * shape.radius ** 2;
                    case "rectangle":
                        return shape.width * shape.height;
                    case "triangle":
                        return 0.5 * shape.base * shape.height;
                }
            }
        """)

    @staticmethod
    def optional_chaining() -> str:
        return textwrap.dedent("""\
            interface Config {
                db?: {
                    host?: string;
                    port?: number;
                    options?: {
                        timeout?: number;
                    };
                };
            }

            function getTimeout(config: Config): number {
                return config.db?.options?.timeout ?? 30;
            }
        """)

    @staticmethod
    def nullish_coalescing() -> str:
        return textwrap.dedent("""\
            function getOrDefault<T>(value: T | null | undefined, fallback: T): T {
                return value ?? fallback;
            }

            function processInput(input?: string): string {
                const normalized = input?.trim() ?? "";
                return normalized.length > 0 ? normalized : "empty";
            }
        """)

    @staticmethod
    def type_guard_function() -> str:
        return textwrap.dedent("""\
            function isNumber(x: unknown): x is number {
                return typeof x === "number" && !isNaN(x);
            }

            function isNonEmptyArray<T>(arr: T[] | undefined): arr is [T, ...T[]] {
                return arr !== undefined && arr.length > 0;
            }

            function safeDivide(a: unknown, b: unknown): number | null {
                if (isNumber(a) && isNumber(b) && b !== 0) {
                    return a / b;
                }
                return null;
            }
        """)

    @staticmethod
    def assertion_function() -> str:
        return textwrap.dedent("""\
            function assertDefined<T>(val: T | undefined, msg?: string): asserts val is T {
                if (val === undefined) {
                    throw new Error(msg ?? "value is undefined");
                }
            }

            function assertNonNull<T>(val: T | null, msg?: string): asserts val is T {
                if (val === null) {
                    throw new Error(msg ?? "value is null");
                }
            }
        """)

    @staticmethod
    def mapped_type() -> str:
        return textwrap.dedent("""\
            type Readonly<T> = { readonly [K in keyof T]: T[K] };
            type Partial<T> = { [K in keyof T]?: T[K] };
            type Required<T> = { [K in keyof T]-?: T[K] };

            type Getters<T> = {
                [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
            };

            interface Person {
                name: string;
                age: number;
            }

            // PersonGetters = { getName: () => string; getAge: () => number; }
            type PersonGetters = Getters<Person>;
        """)

    @staticmethod
    def conditional_type() -> str:
        return textwrap.dedent("""\
            type IsString<T> = T extends string ? true : false;
            type Flatten<T> = T extends Array<infer U> ? U : T;

            type NonNullable<T> = T extends null | undefined ? never : T;

            function process<T>(val: T): NonNullable<T> {
                if (val === null || val === undefined) {
                    throw new Error("null or undefined");
                }
                return val as NonNullable<T>;
            }
        """)

    @staticmethod
    def template_literal() -> str:
        return textwrap.dedent("""\
            type EventName = "click" | "focus" | "blur";
            type Handler<E extends EventName> = `on${Capitalize<E>}`;

            type CSSProperty = "margin" | "padding";
            type CSSDirection = "top" | "right" | "bottom" | "left";
            type CSSKey = `${CSSProperty}-${CSSDirection}`;

            function setStyle(el: HTMLElement, key: CSSKey, value: string): void {
                (el.style as any)[key] = value;
            }
        """)


# ---------------------------------------------------------------------------
# 9. Helper functions (standalone, not fixtures)
# ---------------------------------------------------------------------------


def make_ssa_var(
    name: str = "v",
    version: int = 0,
    file: str = "test.py",
    line: int = 1,
    col: int = 0,
) -> SSAVar:
    """Convenience constructor for SSAVar with inline SourceLocation."""
    return SSAVar(name, version, SourceLocation(file, line, col))


def make_interval(low: float, high: float) -> Interval:
    """Create an Interval, allowing symbolic shorthands for ±∞."""
    return Interval(low, high)


def make_type_tag_set(*tags: TypeTag) -> TypeTagSet:
    """Create a TypeTagSet from variadic TypeTag arguments."""
    return TypeTagSet(frozenset(tags))


def make_nullity(state: str) -> NullityState:
    """Create a NullityState from its string name."""
    return NullityState(state)


def make_predicate(
    kind: str,
    lhs: str,
    rhs: Optional[str] = None,
    negated: bool = False,
) -> Predicate:
    """Create a Predicate from string names."""
    return Predicate(PredicateKind(kind), lhs, rhs, negated)


def make_cfg(
    blocks: Dict[str, List[Tuple[str, List[str], Optional[str]]]],
    entry: str = "entry",
    exit_label: str = "exit",
    edges: Optional[List[Tuple[str, str]]] = None,
) -> CFG:
    """Build a CFG from a compact specification.

    *blocks* maps label → list of (opcode, operand_names, result_name) tuples.
    *edges*  is an optional list of (src, dst) label pairs; if omitted, blocks
    are chained in insertion order.
    """
    cfg = CFG(entry=entry, exit=exit_label)
    labels = list(blocks.keys())
    for lbl, insts_spec in blocks.items():
        instructions: List[SSAInstruction] = []
        for opcode, operand_names, result_name in insts_spec:
            operands = [SSAVar(n, 0) for n in operand_names]
            result = SSAVar(result_name, 0) if result_name else None
            instructions.append(SSAInstruction(opcode, operands, result))
        cfg.add_block(BasicBlock(label=lbl, instructions=instructions))

    if edges is not None:
        for src, dst in edges:
            cfg.blocks[src].successors.append(dst)
            cfg.blocks[dst].predecessors.append(src)
    else:
        for i in range(len(labels) - 1):
            cfg.blocks[labels[i]].successors.append(labels[i + 1])
            cfg.blocks[labels[i + 1]].predecessors.append(labels[i])
    return cfg


def make_function(
    name: str = "f",
    param_names: Optional[List[str]] = None,
    cfg: Optional[CFG] = None,
    return_type: Optional[str] = None,
) -> Function:
    """Convenience constructor for Function."""
    if param_names is None:
        param_names = []
    params = [SSAVar(p, 0) for p in param_names]
    if cfg is None:
        entry = BasicBlock("entry", successors=["exit"])
        exit_blk = BasicBlock("exit", predecessors=["entry"])
        cfg = CFG("entry", "exit")
        cfg.add_block(entry)
        cfg.add_block(exit_blk)
    return Function(name=name, params=params, cfg=cfg, return_type=return_type)


def make_module(
    name: str = "mod",
    functions: Optional[List[Function]] = None,
    classes: Optional[List[str]] = None,
    imports: Optional[List[str]] = None,
) -> Module:
    """Convenience constructor for Module."""
    return Module(
        name=name,
        functions=functions or [],
        classes=classes or [],
        imports=imports or [],
    )


# ---------------------------------------------------------------------------
# 10. Assert helpers
# ---------------------------------------------------------------------------


def assert_interval_eq(a: Interval, b: Interval) -> None:
    """Assert two intervals are equal, with a helpful message on failure."""
    assert a.low == b.low and a.high == b.high, (
        f"Interval mismatch: {a} != {b}"
    )


def assert_type_tags_eq(a: TypeTagSet, b: TypeTagSet) -> None:
    """Assert two TypeTagSets contain exactly the same tags."""
    assert a.tags == b.tags, (
        f"TypeTagSet mismatch: {sorted(t.value for t in a.tags)} "
        f"!= {sorted(t.value for t in b.tags)}"
    )


def assert_nullity_eq(a: NullityState, b: NullityState) -> None:
    """Assert two nullity states are equal."""
    assert a == b, f"NullityState mismatch: {a.value} != {b.value}"


def assert_predicate_implies(p: Predicate, q: Predicate) -> None:
    """Assert that predicate *p* logically implies *q*."""
    assert p.implies(q), f"Expected {p} to imply {q}"


def assert_no_bugs(result: AnalysisResult) -> None:
    """Assert that the analysis result has no bugs."""
    assert not result.has_bugs(), (
        f"Expected no bugs but found {len(result.bugs)}: "
        + "; ".join(str(b) for b in result.bugs)
    )


def assert_has_bug(
    result: AnalysisResult,
    kind: BugKind,
    *,
    message_contains: Optional[str] = None,
) -> None:
    """Assert the result contains at least one bug of the given kind."""
    matching = result.bugs_of_kind(kind)
    assert matching, (
        f"Expected bug of kind {kind.value}, found: "
        + ", ".join(b.kind.value for b in result.bugs)
    )
    if message_contains is not None:
        assert any(message_contains in b.message for b in matching), (
            f"No bug of kind {kind.value} has message containing "
            f"'{message_contains}'. Messages: "
            + "; ".join(b.message for b in matching)
        )


def assert_contract_matches(
    contract: Contract,
    *,
    min_requires: int = 0,
    min_ensures: int = 0,
    min_invariants: int = 0,
) -> None:
    """Assert a contract has at least the expected number of clauses."""
    assert len(contract.requires) >= min_requires, (
        f"Expected >= {min_requires} requires, got {len(contract.requires)}"
    )
    assert len(contract.ensures) >= min_ensures, (
        f"Expected >= {min_ensures} ensures, got {len(contract.ensures)}"
    )
    assert len(contract.invariants) >= min_invariants, (
        f"Expected >= {min_invariants} invariants, got {len(contract.invariants)}"
    )


# ---------------------------------------------------------------------------
# 11. MockSMTSolver
# ---------------------------------------------------------------------------


class MockSMTSolver:
    """A mock SMT solver that records assertions and returns configurable
    satisfiability results.

    Useful for testing CEGAR loops without depending on Z3.
    """

    def __init__(self, *, default_result: str = "sat") -> None:
        """Initialise with a default satisfiability answer.

        Args:
            default_result: One of ``"sat"``, ``"unsat"``, ``"unknown"``.
        """
        self._assertions: List[str] = []
        self._default_result = default_result
        self._scripted_results: List[str] = []
        self._check_count: int = 0
        self._models: List[Dict[str, Any]] = []
        self._push_stack: List[int] = []

    def assert_formula(self, formula: str) -> None:
        """Record an assertion."""
        self._assertions.append(formula)

    def push(self) -> None:
        """Save the current assertion-stack depth."""
        self._push_stack.append(len(self._assertions))

    def pop(self) -> None:
        """Restore the assertion stack to the last push point."""
        if not self._push_stack:
            raise RuntimeError("pop without matching push")
        depth = self._push_stack.pop()
        self._assertions = self._assertions[:depth]

    def check(self) -> str:
        """Return the next scripted result, or the default."""
        self._check_count += 1
        if self._scripted_results:
            return self._scripted_results.pop(0)
        return self._default_result

    def get_model(self) -> Dict[str, Any]:
        """Return the next scripted model, or an empty dict."""
        if self._models:
            return self._models.pop(0)
        return {}

    def script_results(self, *results: str) -> None:
        """Pre-load results to be returned by successive ``check()`` calls."""
        self._scripted_results.extend(results)

    def script_models(self, *models: Dict[str, Any]) -> None:
        """Pre-load models to be returned by successive ``get_model()`` calls."""
        self._models.extend(models)

    def reset(self) -> None:
        """Clear all state."""
        self._assertions.clear()
        self._scripted_results.clear()
        self._check_count = 0
        self._models.clear()
        self._push_stack.clear()

    @property
    def assertions(self) -> List[str]:
        return list(self._assertions)

    @property
    def check_count(self) -> int:
        return self._check_count


# ---------------------------------------------------------------------------
# 12. MockFileSystem
# ---------------------------------------------------------------------------


class MockFileSystem:
    """Virtual in-memory filesystem backed by a ``dict``.

    Paths are normalised with ``/`` separators and no leading ``./``.
    """

    def __init__(self) -> None:
        self._files: Dict[str, str] = {}

    def _norm(self, path: str) -> str:
        return path.replace("\\", "/").lstrip("./")

    def write(self, path: str, content: str) -> None:
        """Create or overwrite a file."""
        self._files[self._norm(path)] = content

    def read(self, path: str) -> str:
        """Read a file; raises ``FileNotFoundError`` if missing."""
        key = self._norm(path)
        if key not in self._files:
            raise FileNotFoundError(f"No such file: {path}")
        return self._files[key]

    def exists(self, path: str) -> bool:
        """Return whether a file exists."""
        key = self._norm(path)
        if key in self._files:
            return True
        # Check if it's a directory prefix
        prefix = key.rstrip("/") + "/"
        return any(k.startswith(prefix) for k in self._files)

    def listdir(self, path: str) -> List[str]:
        """List immediate children of *path*."""
        prefix = self._norm(path).rstrip("/") + "/"
        children: Set[str] = set()
        for k in self._files:
            if k.startswith(prefix):
                remainder = k[len(prefix):]
                child = remainder.split("/")[0]
                children.add(child)
        return sorted(children)

    def delete(self, path: str) -> None:
        """Delete a file."""
        key = self._norm(path)
        if key in self._files:
            del self._files[key]

    def walk(self, root: str = "") -> Iterator[Tuple[str, List[str], List[str]]]:
        """Walk the virtual filesystem, yielding (dirpath, dirs, files)."""
        root_norm = self._norm(root).rstrip("/")
        all_dirs: Set[str] = set()
        dir_files: Dict[str, List[str]] = {}
        for k in self._files:
            if root_norm and not k.startswith(root_norm + "/") and k != root_norm:
                continue
            parts = k.split("/")
            for i in range(len(parts) - 1):
                d = "/".join(parts[: i + 1])
                all_dirs.add(d)
            parent = "/".join(parts[:-1]) if len(parts) > 1 else ""
            dir_files.setdefault(parent, []).append(parts[-1])

        visited = sorted(all_dirs | {root_norm} if root_norm else all_dirs | {""})
        for d in visited:
            subdirs = sorted(
                c for c in all_dirs if c.rsplit("/", 1)[0] == d and c != d
            )
            subdir_names = [s.rsplit("/", 1)[-1] for s in subdirs]
            files = dir_files.get(d, [])
            yield d, subdir_names, sorted(files)

    @property
    def file_count(self) -> int:
        return len(self._files)


# ---------------------------------------------------------------------------
# 13. TestDataBuilder
# ---------------------------------------------------------------------------


class TestDataBuilder:
    """Builder pattern for constructing complex test scenarios.

    Usage::

        data = (TestDataBuilder()
                .with_function("foo", params=["a", "b"])
                .with_predicate("isinstance", "a", "int")
                .with_cfg(entry="bb0", exit_label="bb2",
                          edges=[("bb0","bb1"),("bb1","bb2")])
                .build())
    """

    def __init__(self) -> None:
        self._functions: List[Dict[str, Any]] = []
        self._predicates: List[Predicate] = []
        self._cfg_spec: Optional[Dict[str, Any]] = None
        self._bugs: List[Bug] = []
        self._contracts: List[Contract] = []
        self._module_name: str = "test_module"
        self._imports: List[str] = []
        self._classes: List[str] = []

    def with_function(
        self,
        name: str,
        params: Optional[List[str]] = None,
        return_type: Optional[str] = None,
    ) -> TestDataBuilder:
        """Add a function spec to the builder."""
        self._functions.append(
            {"name": name, "params": params or [], "return_type": return_type}
        )
        return self

    def with_predicate(
        self,
        kind: str,
        lhs: str,
        rhs: Optional[str] = None,
        negated: bool = False,
    ) -> TestDataBuilder:
        """Add a predicate to the builder."""
        self._predicates.append(make_predicate(kind, lhs, rhs, negated))
        return self

    def with_cfg(
        self,
        entry: str = "entry",
        exit_label: str = "exit",
        edges: Optional[List[Tuple[str, str]]] = None,
    ) -> TestDataBuilder:
        """Specify a CFG skeleton."""
        self._cfg_spec = {"entry": entry, "exit": exit_label, "edges": edges or []}
        return self

    def with_bug(
        self,
        kind: BugKind,
        message: str = "",
        file: str = "test.py",
        line: int = 1,
    ) -> TestDataBuilder:
        """Add an expected bug to the result."""
        loc = SourceLocation(file, line, 0)
        self._bugs.append(Bug(loc, kind, message))
        return self

    def with_contract(
        self,
        requires: Optional[List[Predicate]] = None,
        ensures: Optional[List[Predicate]] = None,
        invariants: Optional[List[Predicate]] = None,
    ) -> TestDataBuilder:
        """Add a contract."""
        self._contracts.append(
            Contract(
                requires=requires or [],
                ensures=ensures or [],
                invariants=invariants or [],
            )
        )
        return self

    def with_module_name(self, name: str) -> TestDataBuilder:
        self._module_name = name
        return self

    def with_import(self, imp: str) -> TestDataBuilder:
        self._imports.append(imp)
        return self

    def with_class(self, cls: str) -> TestDataBuilder:
        self._classes.append(cls)
        return self

    def build(self) -> Dict[str, Any]:
        """Materialise the test data into concrete IR objects."""
        functions: List[Function] = []
        for fspec in self._functions:
            cfg = self._build_cfg()
            fn = make_function(
                name=fspec["name"],
                param_names=fspec["params"],
                cfg=cfg,
                return_type=fspec["return_type"],
            )
            functions.append(fn)

        module = make_module(
            name=self._module_name,
            functions=functions,
            classes=self._classes,
            imports=self._imports,
        )

        result = AnalysisResult(bugs=list(self._bugs))
        for i, c in enumerate(self._contracts):
            result.contracts[f"contract_{i}"] = c

        return {
            "module": module,
            "predicates": list(self._predicates),
            "analysis_result": result,
        }

    def _build_cfg(self) -> CFG:
        """Internal: build a CFG from the stored spec or a default."""
        if self._cfg_spec is None:
            entry = BasicBlock("entry", successors=["exit"])
            exit_blk = BasicBlock("exit", predecessors=["entry"])
            cfg = CFG("entry", "exit")
            cfg.add_block(entry)
            cfg.add_block(exit_blk)
            return cfg

        spec = self._cfg_spec
        label_set: Set[str] = {spec["entry"], spec["exit"]}
        for src, dst in spec["edges"]:
            label_set.add(src)
            label_set.add(dst)
        cfg = CFG(entry=spec["entry"], exit=spec["exit"])
        for lbl in label_set:
            cfg.add_block(BasicBlock(label=lbl))
        for src, dst in spec["edges"]:
            cfg.blocks[src].successors.append(dst)
            cfg.blocks[dst].predecessors.append(src)
        return cfg


# ---------------------------------------------------------------------------
# 14. PerformanceBenchmark
# ---------------------------------------------------------------------------


class PerformanceBenchmark:
    """Context manager for measuring elapsed wall-clock time.

    Usage::

        with PerformanceBenchmark("analysis", threshold_ms=500) as bm:
            run_analysis()
        assert bm.passed, f"Too slow: {bm.elapsed_ms:.1f}ms"
    """

    def __init__(self, name: str, *, threshold_ms: float = 1000.0) -> None:
        self.name = name
        self.threshold_ms = threshold_ms
        self._start: float = 0.0
        self._end: float = 0.0
        self.elapsed_ms: float = 0.0
        self.passed: bool = True

    def __enter__(self) -> PerformanceBenchmark:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        self._end = time.perf_counter()
        self.elapsed_ms = (self._end - self._start) * 1000.0
        self.passed = self.elapsed_ms <= self.threshold_ms
        return False  # don't suppress exceptions

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"<PerformanceBenchmark {self.name!r} "
            f"{self.elapsed_ms:.1f}ms / {self.threshold_ms:.1f}ms [{status}]>"
        )

    def assert_passed(self, msg: Optional[str] = None) -> None:
        """Raise ``AssertionError`` if the threshold was exceeded."""
        if not self.passed:
            default = (
                f"Performance benchmark {self.name!r} failed: "
                f"{self.elapsed_ms:.1f}ms > {self.threshold_ms:.1f}ms"
            )
            raise AssertionError(msg or default)


# ---------------------------------------------------------------------------
# 15. SnapshotTesting
# ---------------------------------------------------------------------------


class SnapshotTesting:
    """Snapshot testing utilities that serialise analysis results to JSON
    and compare them against previously-stored snapshots.

    Snapshots are stored in a configurable directory (default: a temp dir).
    """

    def __init__(self, snapshot_dir: Optional[str] = None) -> None:
        if snapshot_dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="snapshots_"))
            self._own_dir = True
        else:
            self._dir = Path(snapshot_dir)
            self._dir.mkdir(parents=True, exist_ok=True)
            self._own_dir = False

    @property
    def snapshot_dir(self) -> Path:
        return self._dir

    def _snapshot_path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.snapshot.json"

    def _serialise_result(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert an AnalysisResult to a JSON-serialisable dict."""
        return {
            "bugs": [
                {
                    "location": str(b.location),
                    "kind": b.kind.value,
                    "message": b.message,
                    "severity": b.severity,
                }
                for b in result.bugs
            ],
            "warnings": result.warnings,
            "inferred_types": {
                var: sorted(t.value for t in ts.tags)
                for var, ts in result.inferred_types.items()
            },
            "contracts": {
                name: {
                    "requires": [str(p) for p in c.requires],
                    "ensures": [str(p) for p in c.ensures],
                    "invariants": [str(p) for p in c.invariants],
                }
                for name, c in result.contracts.items()
            },
        }

    def _serialise_any(self, obj: Any) -> str:
        """Best-effort JSON serialisation of arbitrary objects."""
        if isinstance(obj, AnalysisResult):
            return json.dumps(self._serialise_result(obj), indent=2, sort_keys=True)
        return json.dumps(obj, indent=2, sort_keys=True, default=str)

    def update(self, name: str, data: Any) -> Path:
        """Write (or overwrite) a snapshot."""
        path = self._snapshot_path(name)
        path.write_text(self._serialise_any(data), encoding="utf-8")
        return path

    def compare(self, name: str, data: Any) -> bool:
        """Compare *data* against the stored snapshot.

        Returns ``True`` when they match.  If no snapshot exists yet the
        comparison always fails (the caller should call ``update`` first).
        """
        path = self._snapshot_path(name)
        if not path.exists():
            return False
        stored = path.read_text(encoding="utf-8")
        current = self._serialise_any(data)
        return stored == current

    def assert_match(self, name: str, data: Any, *, update: bool = False) -> None:
        """Assert that *data* matches the stored snapshot.

        If *update* is ``True``, a mismatching snapshot is overwritten instead
        of raising an assertion error (useful for ``--snapshot-update`` flags).
        """
        path = self._snapshot_path(name)
        current = self._serialise_any(data)
        if not path.exists() or update:
            path.write_text(current, encoding="utf-8")
            return
        stored = path.read_text(encoding="utf-8")
        if stored != current:
            # Produce a human-readable diff
            import difflib

            diff = "\n".join(
                difflib.unified_diff(
                    stored.splitlines(),
                    current.splitlines(),
                    fromfile=f"snapshot/{name}",
                    tofile="actual",
                    lineterm="",
                )
            )
            raise AssertionError(f"Snapshot mismatch for {name!r}:\n{diff}")

    def fingerprint(self, data: Any) -> str:
        """Return a SHA-256 hex digest of the serialised data."""
        raw = self._serialise_any(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def cleanup(self) -> None:
        """Remove the snapshot directory if we created it."""
        if self._own_dir and self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 16. ParametrizedTestData
# ---------------------------------------------------------------------------


class ParametrizedTestData:
    """Generates randomised test data for property-based-style testing.

    Not a full property-based testing framework – just a deterministic
    data generator seeded for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def random_interval(
        self, lo_range: Tuple[float, float] = (-100, 100), max_width: float = 200
    ) -> Interval:
        """Generate a random non-bottom interval."""
        lo = self._rng.uniform(*lo_range)
        width = self._rng.uniform(0, max_width)
        return Interval(lo, lo + width)

    def random_intervals(self, n: int) -> List[Interval]:
        """Generate *n* random intervals."""
        return [self.random_interval() for _ in range(n)]

    def random_type_tag_set(self, max_size: int = 5) -> TypeTagSet:
        """Generate a random TypeTagSet (excluding ANY)."""
        concrete = [t for t in TypeTag if t != TypeTag.ANY]
        k = self._rng.randint(1, min(max_size, len(concrete)))
        chosen = self._rng.sample(concrete, k)
        return TypeTagSet(frozenset(chosen))

    def random_type_tag_sets(self, n: int) -> List[TypeTagSet]:
        return [self.random_type_tag_set() for _ in range(n)]

    def random_predicate(self) -> Predicate:
        """Generate a random predicate."""
        kind = self._rng.choice(list(PredicateKind))
        lhs = self._rng.choice(["x", "y", "z", "a", "b", "obj", "val"])
        rhs_options = ["int", "str", "0", "y", "None", "foo", None]
        rhs = self._rng.choice(rhs_options)
        negated = self._rng.choice([True, False])
        return Predicate(kind, lhs, rhs, negated)

    def random_predicates(self, n: int) -> List[Predicate]:
        return [self.random_predicate() for _ in range(n)]

    def random_nullity(self) -> NullityState:
        return self._rng.choice(list(NullityState))

    def random_bug(self) -> Bug:
        """Generate a random bug report."""
        kind = self._rng.choice(list(BugKind))
        line = self._rng.randint(1, 500)
        loc = SourceLocation("test.py", line, 0)
        messages = {
            BugKind.TYPE_ERROR: "unexpected type",
            BugKind.NULL_DEREF: "dereference of None",
            BugKind.INDEX_OOB: "index out of bounds",
            BugKind.DIV_ZERO: "division by zero",
            BugKind.ATTR_ERROR: "missing attribute",
            BugKind.UNREACHABLE: "unreachable code",
        }
        return Bug(loc, kind, messages.get(kind, "unknown error"))

    def random_analysis_result(self, max_bugs: int = 5) -> AnalysisResult:
        """Generate a random AnalysisResult."""
        n_bugs = self._rng.randint(0, max_bugs)
        bugs = [self.random_bug() for _ in range(n_bugs)]
        n_types = self._rng.randint(0, 5)
        inferred: Dict[str, TypeTagSet] = {}
        for _ in range(n_types):
            var = self._rng.choice(["x", "y", "z", "result", "tmp"])
            inferred[var] = self.random_type_tag_set()
        return AnalysisResult(bugs=bugs, inferred_types=inferred)

    def random_contract(self, max_clauses: int = 3) -> Contract:
        n_req = self._rng.randint(0, max_clauses)
        n_ens = self._rng.randint(0, max_clauses)
        n_inv = self._rng.randint(0, max_clauses)
        return Contract(
            requires=self.random_predicates(n_req),
            ensures=self.random_predicates(n_ens),
            invariants=self.random_predicates(n_inv),
        )

    def pairs(
        self, generator: Callable[[], Any], n: int = 10
    ) -> List[Tuple[Any, Any]]:
        """Generate *n* pairs using the given generator (for commutativity etc.)."""
        return [(generator(), generator()) for _ in range(n)]


# ---------------------------------------------------------------------------
# 17. TempProject
# ---------------------------------------------------------------------------


class TempProject:
    """Creates a temporary project directory structure on disk for
    integration tests.  The directory is removed on ``cleanup()`` or
    when used as a context manager.

    Usage::

        with TempProject(lang="python") as proj:
            proj.add_file("main.py", "print('hi')")
            result = run_analysis(proj.root)
    """

    def __init__(
        self,
        lang: str = "python",
        *,
        prefix: str = "refinement_test_",
    ) -> None:
        self._lang = lang
        self._root = Path(tempfile.mkdtemp(prefix=prefix))
        self._files: Dict[str, Path] = {}
        self._setup_project()

    def _setup_project(self) -> None:
        """Create the skeleton appropriate for the language."""
        if self._lang == "python":
            self._create_python_skeleton()
        elif self._lang in ("typescript", "ts"):
            self._create_typescript_skeleton()
        else:
            # Generic: just an empty src/ directory
            (self._root / "src").mkdir(exist_ok=True)

    def _create_python_skeleton(self) -> None:
        """Create a minimal Python project layout."""
        src = self._root / "src"
        src.mkdir(exist_ok=True)
        init = src / "__init__.py"
        init.write_text("", encoding="utf-8")
        self._files["src/__init__.py"] = init

        setup_cfg = self._root / "setup.cfg"
        setup_cfg.write_text(
            textwrap.dedent("""\
                [metadata]
                name = test_project
                version = 0.0.1

                [options]
                packages = find:
            """),
            encoding="utf-8",
        )
        self._files["setup.cfg"] = setup_cfg

        pyproject = self._root / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
                [build-system]
                requires = ["setuptools"]
                build-backend = "setuptools.backends._legacy:_Backend"
            """),
            encoding="utf-8",
        )
        self._files["pyproject.toml"] = pyproject

    def _create_typescript_skeleton(self) -> None:
        """Create a minimal TypeScript project layout."""
        src = self._root / "src"
        src.mkdir(exist_ok=True)

        tsconfig = self._root / "tsconfig.json"
        tsconfig.write_text(
            json.dumps(
                {
                    "compilerOptions": {
                        "target": "ES2020",
                        "module": "commonjs",
                        "strict": True,
                        "outDir": "./dist",
                        "rootDir": "./src",
                    },
                    "include": ["src/**/*"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._files["tsconfig.json"] = tsconfig

        pkg = self._root / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "version": "0.0.1",
                    "private": True,
                    "scripts": {"build": "tsc"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._files["package.json"] = pkg

    @property
    def root(self) -> Path:
        """The root directory of the temporary project."""
        return self._root

    @property
    def lang(self) -> str:
        return self._lang

    def add_file(self, relative_path: str, content: str) -> Path:
        """Create a file inside the project."""
        target = self._root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content), encoding="utf-8")
        self._files[relative_path] = target
        return target

    def read_file(self, relative_path: str) -> str:
        """Read a file from the project."""
        target = self._root / relative_path
        return target.read_text(encoding="utf-8")

    def file_exists(self, relative_path: str) -> bool:
        return (self._root / relative_path).exists()

    def list_files(self, subdir: str = "") -> List[str]:
        """List all files under *subdir* (relative paths)."""
        base = self._root / subdir
        result: List[str] = []
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    result.append(str(p.relative_to(self._root)))
        return result

    def cleanup(self) -> None:
        """Remove the temporary project directory."""
        if self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> TempProject:
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        self.cleanup()
        return False

    def __repr__(self) -> str:
        return f"<TempProject lang={self._lang!r} root={self._root}>"


# ---------------------------------------------------------------------------
# Extra fixtures wiring up the utility classes
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_solver() -> MockSMTSolver:
    """A fresh MockSMTSolver defaulting to sat."""
    return MockSMTSolver(default_result="sat")


@pytest.fixture
def mock_fs() -> MockFileSystem:
    """An empty MockFileSystem."""
    return MockFileSystem()


@pytest.fixture
def builder() -> TestDataBuilder:
    """A fresh TestDataBuilder."""
    return TestDataBuilder()


@pytest.fixture
def snapshot(tmp_path: Path) -> Iterator[SnapshotTesting]:
    """SnapshotTesting instance writing to pytest's tmp_path."""
    st = SnapshotTesting(snapshot_dir=str(tmp_path / "snapshots"))
    yield st
    st.cleanup()


@pytest.fixture
def parametrised_data() -> ParametrizedTestData:
    """A seeded ParametrizedTestData generator."""
    return ParametrizedTestData(seed=12345)


@pytest.fixture
def temp_python_project() -> Iterator[TempProject]:
    """A temporary Python project, cleaned up after the test."""
    proj = TempProject(lang="python")
    yield proj
    proj.cleanup()


@pytest.fixture
def temp_ts_project() -> Iterator[TempProject]:
    """A temporary TypeScript project, cleaned up after the test."""
    proj = TempProject(lang="typescript")
    yield proj
    proj.cleanup()


@pytest.fixture
def perf_benchmark() -> Callable[..., PerformanceBenchmark]:
    """Factory fixture returning PerformanceBenchmark instances."""

    def _make(name: str = "test", threshold_ms: float = 1000.0) -> PerformanceBenchmark:
        return PerformanceBenchmark(name, threshold_ms=threshold_ms)

    return _make


@pytest.fixture
def sample_programs() -> type:
    """Return the SamplePrograms class itself for easy access."""
    return SamplePrograms
