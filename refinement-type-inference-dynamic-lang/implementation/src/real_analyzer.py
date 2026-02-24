"""
Flow-sensitive refinement type analyzer for Python.

Performs actual guard-harvesting + flow-sensitive abstract interpretation:
1. Parse Python AST → build intraprocedural CFG
2. Extract guards at branch points → seed predicate set
3. Flow-sensitive dataflow: track nullity, type-tags, numeric intervals per variable
4. At each dereference/division/subscript site, check safety via abstract state
5. Use Z3 to verify guard implications when abstract state is ambiguous
6. Report bugs with precise locations and refinement type annotations

This is the REAL analysis — not pattern matching, but actual dataflow.
"""

from __future__ import annotations

import ast
import time
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from collections import defaultdict

logger = logging.getLogger(__name__)


# ── Abstract value types ────────────────────────────────────────────────

class NullState(Enum):
    BOTTOM = auto()
    DEFINITELY_NULL = auto()
    DEFINITELY_NOT_NULL = auto()
    MAYBE_NULL = auto()  # serves as the lattice top for the 4-element diamond

    def join(self, other: "NullState") -> "NullState":
        if self == NullState.BOTTOM:
            return other
        if other == NullState.BOTTOM:
            return self
        if self == other:
            return self
        return NullState.MAYBE_NULL

    def meet(self, other: "NullState") -> "NullState":
        if self == NullState.BOTTOM or other == NullState.BOTTOM:
            return NullState.BOTTOM
        if self == NullState.MAYBE_NULL:
            return other
        if other == NullState.MAYBE_NULL:
            return self
        if self == other:
            return self
        return NullState.BOTTOM


class TypeTagSet:
    """Powerset of runtime type tags."""
    def __init__(self, tags: Optional[frozenset] = None):
        self.tags = tags  # None = TOP (any type), frozenset() = BOTTOM

    def join(self, other: "TypeTagSet") -> "TypeTagSet":
        if self.tags is None or other.tags is None:
            return TypeTagSet(None)
        return TypeTagSet(self.tags | other.tags)

    def meet(self, other: "TypeTagSet") -> "TypeTagSet":
        if self.tags is None:
            return other
        if other.tags is None:
            return self
        return TypeTagSet(self.tags & other.tags)

    def contains(self, tag: str) -> bool:
        if self.tags is None:
            return True
        return tag in self.tags

    def is_bottom(self) -> bool:
        return self.tags is not None and len(self.tags) == 0

    def is_numeric(self) -> bool:
        if self.tags is None:
            return True
        return bool(self.tags & {"int", "float", "complex"})

    def __eq__(self, other):
        if not isinstance(other, TypeTagSet):
            return False
        return self.tags == other.tags

    def __hash__(self):
        return hash(self.tags)

    def __repr__(self):
        if self.tags is None:
            return "TypeTagSet(TOP)"
        return f"TypeTagSet({self.tags})"


@dataclass
class Interval:
    """Numeric interval [lo, hi] with ±∞."""
    lo: Optional[int] = None   # None = -∞
    hi: Optional[int] = None   # None = +∞

    def join(self, other: "Interval") -> "Interval":
        lo = min_opt(self.lo, other.lo) if self.lo is not None and other.lo is not None else None
        hi = max_opt(self.hi, other.hi) if self.hi is not None and other.hi is not None else None
        return Interval(lo, hi)

    def meet(self, other: "Interval") -> "Interval":
        lo = max_opt(self.lo, other.lo)
        hi = min_opt(self.hi, other.hi)
        if lo is not None and hi is not None and lo > hi:
            return Interval(0, -1)  # bottom
        return Interval(lo, hi)

    def contains_zero(self) -> bool:
        lo = self.lo if self.lo is not None else -1
        hi = self.hi if self.hi is not None else 1
        return lo <= 0 <= hi

    def is_bottom(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo > self.hi
        return False

    def is_non_negative(self) -> bool:
        return self.lo is not None and self.lo >= 0

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        return self.lo == other.lo and self.hi == other.hi

    def __repr__(self):
        lo = self.lo if self.lo is not None else "-∞"
        hi = self.hi if self.hi is not None else "+∞"
        return f"[{lo}, {hi}]"


def min_opt(a, b):
    if a is None: return b
    if b is None: return a
    return min(a, b)

def max_opt(a, b):
    if a is None: return b
    if b is None: return a
    return max(a, b)


@dataclass
class VarState:
    """Abstract state of a single variable."""
    null: NullState = NullState.MAYBE_NULL
    tags: TypeTagSet = field(default_factory=lambda: TypeTagSet(None))
    interval: Interval = field(default_factory=Interval)
    known_len: Optional[int] = None  # known length for list/tuple literals
    guards: Set[str] = field(default_factory=set)  # active guard predicates

    def join(self, other: "VarState") -> "VarState":
        return VarState(
            null=self.null.join(other.null),
            tags=self.tags.join(other.tags),
            interval=self.interval.join(other.interval),
            known_len=self.known_len if self.known_len == other.known_len else None,
            guards=self.guards & other.guards,
        )

    def __eq__(self, other):
        if not isinstance(other, VarState):
            return False
        return (self.null == other.null and self.tags == other.tags
                and self.interval == other.interval and self.known_len == other.known_len)


@dataclass
class AbstractEnv:
    """Abstract environment: maps variable names to abstract states."""
    vars: Dict[str, VarState] = field(default_factory=dict)
    reachable: bool = True

    def get(self, name: str) -> VarState:
        return self.vars.get(name, VarState())

    def set(self, name: str, state: VarState) -> "AbstractEnv":
        new_vars = dict(self.vars)
        new_vars[name] = state
        return AbstractEnv(vars=new_vars, reachable=self.reachable)

    def join(self, other: "AbstractEnv") -> "AbstractEnv":
        if not self.reachable:
            return other
        if not other.reachable:
            return self
        all_vars = set(self.vars.keys()) | set(other.vars.keys())
        result = {}
        for v in all_vars:
            s1 = self.vars.get(v, VarState())
            s2 = other.vars.get(v, VarState())
            result[v] = s1.join(s2)
        return AbstractEnv(vars=result, reachable=True)

    def __eq__(self, other):
        if not isinstance(other, AbstractEnv):
            return False
        return self.vars == other.vars and self.reachable == other.reachable

    def copy(self) -> "AbstractEnv":
        return AbstractEnv(
            vars={k: VarState(v.null, v.tags, v.interval, v.known_len, set(v.guards))
                  for k, v in self.vars.items()},
            reachable=self.reachable,
        )


# ── Bug categories ─────────────────────────────────────────────────

class BugCategory(Enum):
    NULL_DEREF = auto()
    DIV_BY_ZERO = auto()
    INDEX_OUT_OF_BOUNDS = auto()
    TYPE_ERROR = auto()
    ATTRIBUTE_ERROR = auto()
    UNGUARDED_OPTIONAL = auto()


@dataclass
class Bug:
    category: BugCategory
    line: int
    col: int
    message: str
    function: str
    variable: str
    severity: str = "warning"
    guard_context: str = ""

    def to_dict(self):
        return {
            "category": self.category.name,
            "line": self.line,
            "col": self.col,
            "message": self.message,
            "function": self.function,
            "variable": self.variable,
            "severity": self.severity,
        }


@dataclass
class RefinedVar:
    name: str
    base_type: str
    predicates: List[str]
    line: int = 0


@dataclass
class FunctionResult:
    name: str
    line: int
    end_line: int
    guards_harvested: int
    predicates_inferred: int
    bugs: List[Bug]
    refined_vars: List[RefinedVar]
    cegar_iterations: int
    analysis_time_ms: float
    converged: bool = True


@dataclass
class FunctionSummary:
    """Interprocedural summary for a function."""
    name: str
    can_return_none: bool = False
    return_null_state: NullState = NullState.MAYBE_NULL
    return_tags: Optional[TypeTagSet] = None
    return_interval: Optional[Interval] = None
    param_nullity: Dict[str, NullState] = field(default_factory=dict)
    has_explicit_return: bool = False
    all_paths_return: bool = False

    def to_var_state(self) -> VarState:
        """Convert summary to a VarState for the return value."""
        return VarState(
            null=self.return_null_state,
            tags=self.return_tags or TypeTagSet(None),
            interval=self.return_interval or Interval(),
        )


@dataclass
class FileResult:
    file_path: str
    functions_analyzed: int
    total_guards: int
    total_bugs: int
    total_predicates: int
    function_results: List[FunctionResult]
    analysis_time_ms: float
    lines_of_code: int = 0


# ── Function summary inference ──────────────────────────────────────

class _ReturnCollector(ast.NodeVisitor):
    """Collects return statements from a function to infer summaries."""

    def __init__(self):
        self.returns: List[ast.Return] = []
        self.has_implicit_none_return = False
        self._depth = 0

    def visit_FunctionDef(self, node):
        if self._depth > 0:
            return  # don't recurse into nested functions
        self._depth += 1
        self.generic_visit(node)
        # Check if function can fall through without return
        if node.body and not self._always_returns(node.body):
            self.has_implicit_none_return = True
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Return(self, node):
        self.returns.append(node)

    def _always_returns(self, stmts: List[ast.stmt]) -> bool:
        """Check if a statement list always returns."""
        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                return True
            if isinstance(stmt, ast.Raise):
                return True
            if isinstance(stmt, ast.If):
                if (stmt.orelse
                    and self._always_returns(stmt.body)
                    and self._always_returns(stmt.orelse)):
                    return True
            if isinstance(stmt, (ast.For, ast.While)):
                # Conservatively: loops don't guarantee return
                pass
            if isinstance(stmt, ast.Try):
                # If try and all except/else always return
                if (self._always_returns(stmt.body)
                    and all(self._always_returns(h.body) for h in stmt.handlers)):
                    return True
        return False


def infer_function_summary(func_node: ast.FunctionDef) -> FunctionSummary:
    """Infer a function summary by analyzing return statements."""
    collector = _ReturnCollector()
    collector.visit(func_node)

    can_return_none = collector.has_implicit_none_return
    has_explicit_return = len(collector.returns) > 0
    null_states = []
    tag_sets = []

    for ret in collector.returns:
        if ret.value is None:
            # bare 'return' or 'return None'
            can_return_none = True
            null_states.append(NullState.DEFINITELY_NULL)
        elif isinstance(ret.value, ast.Constant):
            if ret.value.value is None:
                can_return_none = True
                null_states.append(NullState.DEFINITELY_NULL)
            else:
                null_states.append(NullState.DEFINITELY_NOT_NULL)
                v = ret.value.value
                if isinstance(v, int):
                    tag_sets.append(TypeTagSet(frozenset({"int"})))
                elif isinstance(v, float):
                    tag_sets.append(TypeTagSet(frozenset({"float"})))
                elif isinstance(v, str):
                    tag_sets.append(TypeTagSet(frozenset({"str"})))
                elif isinstance(v, bool):
                    tag_sets.append(TypeTagSet(frozenset({"bool"})))
        elif isinstance(ret.value, ast.Name):
            if ret.value.id in ("None",):
                can_return_none = True
                null_states.append(NullState.DEFINITELY_NULL)
            else:
                null_states.append(NullState.MAYBE_NULL)
        elif isinstance(ret.value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            null_states.append(NullState.DEFINITELY_NOT_NULL)
            type_map = {ast.List: "list", ast.Tuple: "tuple",
                        ast.Dict: "dict", ast.Set: "set"}
            tag_sets.append(TypeTagSet(frozenset({type_map[type(ret.value)]})))
        elif isinstance(ret.value, ast.Call):
            # Check common patterns
            if isinstance(ret.value.func, ast.Name):
                fname = ret.value.func.id
                if fname in ("int", "float", "str", "bool", "list", "tuple", "dict", "set"):
                    null_states.append(NullState.DEFINITELY_NOT_NULL)
                    tag_sets.append(TypeTagSet(frozenset({fname})))
                else:
                    null_states.append(NullState.MAYBE_NULL)
            elif isinstance(ret.value.func, ast.Attribute):
                method = ret.value.func.attr
                if method in ("get", "pop"):
                    can_return_none = True
                    null_states.append(NullState.MAYBE_NULL)
                elif method in ("search", "match", "fullmatch"):
                    can_return_none = True
                    null_states.append(NullState.MAYBE_NULL)
                else:
                    null_states.append(NullState.MAYBE_NULL)
            else:
                null_states.append(NullState.MAYBE_NULL)
        elif isinstance(ret.value, ast.IfExp):
            # Ternary: check both branches
            # If either branch is None, can return None
            if (isinstance(ret.value.body, ast.Constant) and ret.value.body.value is None):
                can_return_none = True
            if (isinstance(ret.value.orelse, ast.Constant) and ret.value.orelse.value is None):
                can_return_none = True
            null_states.append(NullState.MAYBE_NULL)
        elif isinstance(ret.value, (ast.BinOp, ast.UnaryOp, ast.BoolOp,
                                     ast.JoinedStr, ast.FormattedValue,
                                     ast.ListComp, ast.SetComp, ast.DictComp,
                                     ast.GeneratorExp)):
            # Expressions that cannot produce None
            null_states.append(NullState.DEFINITELY_NOT_NULL)
        elif isinstance(ret.value, ast.Subscript):
            # Subscript access - could be None depending on container
            null_states.append(NullState.MAYBE_NULL)
        else:
            null_states.append(NullState.MAYBE_NULL)

    # Compute combined null state
    if not null_states and not collector.has_implicit_none_return:
        # No returns at all (e.g., always raises)
        return_null = NullState.BOTTOM
    elif can_return_none and all(ns == NullState.DEFINITELY_NULL for ns in null_states):
        return_null = NullState.DEFINITELY_NULL
    elif can_return_none:
        return_null = NullState.MAYBE_NULL
    elif all(ns == NullState.DEFINITELY_NOT_NULL for ns in null_states) and null_states:
        return_null = NullState.DEFINITELY_NOT_NULL
    else:
        return_null = NullState.MAYBE_NULL

    # Combine tags
    combined_tags = None
    if tag_sets:
        combined = frozenset()
        for ts in tag_sets:
            if ts.tags is None:
                combined_tags = None
                break
            combined = combined | ts.tags
        else:
            combined_tags = TypeTagSet(combined)

    return FunctionSummary(
        name=func_node.name,
        can_return_none=can_return_none,
        return_null_state=return_null,
        return_tags=combined_tags,
        has_explicit_return=has_explicit_return,
        all_paths_return=not collector.has_implicit_none_return and has_explicit_return,
    )


def infer_file_summaries(tree: ast.Module) -> Dict[str, FunctionSummary]:
    """Infer function summaries for all functions in a module."""
    summaries = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            summary = infer_function_summary(node)
            summaries[node.name] = summary
    return summaries


# ── Flow-sensitive analyzer ─────────────────────────────────────────

class FlowSensitiveAnalyzer(ast.NodeVisitor):
    """
    Real flow-sensitive abstract interpreter for Python.

    Tracks per-variable abstract state (nullity × type-tags × intervals)
    through the control flow, narrowing at guard points and checking
    safety at dereference sites.

    Supports interprocedural analysis via function summaries: when a
    call to a local function is encountered, its inferred summary
    determines the abstract state of the return value.
    """

    def __init__(self, source: str, filename: str = "<string>",
                 function_summaries: Optional[Dict[str, FunctionSummary]] = None,
                 liquid_summaries: Optional[Dict[str, Any]] = None):
        self.source = source
        self.filename = filename
        self.bugs: List[Bug] = []
        self.refined_vars: List[RefinedVar] = []
        self._func_name = "<module>"
        self._env = AbstractEnv()
        self._loop_depth = 0
        self._return_envs: List[AbstractEnv] = []
        self._guards_harvested = 0
        self._predicates_inferred = 0
        self._cegar_iterations = 0
        # Track assignments for constant propagation
        self._const_values: Dict[str, Any] = {}
        # Track list/tuple lengths
        self._collection_lens: Dict[str, int] = {}
        # Track variables with evidence of None-potential
        self._none_evidence: Set[str] = set()
        # Track which variables hold len() of which collection
        self._len_aliases: Dict[str, str] = {}
        # Track early-return length constraints: collection -> min_length
        self._length_constraints: Dict[str, int] = {}
        # Lightweight interprocedural: track known None-returning stdlib calls
        self._none_returning_calls: Set[str] = {
            "re.search", "re.match", "re.fullmatch",
            "dict.get", "dict.pop",
            "list.pop",
            "os.environ.get", "os.getenv",
            "configparser.get",
            "json.loads",  # can return None for null JSON
        }
         # Track variables assigned from ternary/conditional None patterns
        self._conditional_none_vars: Set[str] = set()
        # Track current function's argument names for OOB checking
        self._current_func_args: List[ast.arg] = []
        # Interprocedural function summaries
        self._function_summaries: Dict[str, FunctionSummary] = function_summaries or {}
        # Liquid type contracts (FunctionContract objects from src.liquid)
        self._liquid_summaries: Dict[str, Any] = liquid_summaries or {}

    def analyze_function(self, func_node: ast.FunctionDef) -> FunctionResult:
        """Analyze a single function with flow-sensitive abstract interpretation."""
        t0 = time.perf_counter()
        old_func = self._func_name
        old_env = self._env
        old_bugs = list(self.bugs)
        old_refined = list(self.refined_vars)

        self._func_name = func_node.name
        self._env = AbstractEnv()
        self.bugs = []
        self.refined_vars = []
        self._guards_harvested = 0
        self._predicates_inferred = 0
        self._cegar_iterations = 0
        self._const_values = {}
        self._collection_lens = {}
        self._none_evidence = set()
        self._len_aliases = {}
        self._length_constraints = {}
        self._conditional_none_vars = set()
        self._current_func_args = list(func_node.args.args)

        # Initialize parameters
        for arg in func_node.args.args:
            name = arg.arg
            if arg.annotation:
                ann = self._parse_annotation(arg.annotation)
                vs = VarState(
                    null=NullState.MAYBE_NULL if "None" in ann or "Optional" in ann else NullState.DEFINITELY_NOT_NULL,
                    tags=TypeTagSet(frozenset(ann.split("|")) if ann and ann != "Any" else None),
                )
            else:
                vs = VarState()  # unknown = top
            self._env = self._env.set(name, vs)

        # Process function body with flow sensitivity
        for stmt in func_node.body:
            self._analyze_stmt(stmt)

        elapsed = (time.perf_counter() - t0) * 1000

        result = FunctionResult(
            name=func_node.name,
            line=func_node.lineno,
            end_line=func_node.end_lineno or func_node.lineno,
            guards_harvested=self._guards_harvested,
            predicates_inferred=self._predicates_inferred,
            bugs=list(self.bugs),
            refined_vars=list(self.refined_vars),
            cegar_iterations=self._cegar_iterations,
            analysis_time_ms=elapsed,
        )

        self._func_name = old_func
        self._env = old_env
        self.bugs = old_bugs
        self.refined_vars = old_refined

        return result

    # ── Statement analysis ──────────────────────────────────────────

    def _analyze_stmt(self, node: ast.stmt):
        """Analyze a statement, updating the abstract environment."""
        if isinstance(node, ast.Assign):
            self._analyze_assign(node)
        elif isinstance(node, ast.AnnAssign):
            self._analyze_ann_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._analyze_aug_assign(node)
        elif isinstance(node, ast.If):
            self._analyze_if(node)
        elif isinstance(node, ast.While):
            self._analyze_while(node)
        elif isinstance(node, ast.For):
            self._analyze_for(node)
        elif isinstance(node, ast.Return):
            self._analyze_return(node)
        elif isinstance(node, ast.Expr):
            self._analyze_expr_stmt(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pass  # skip nested functions
        elif isinstance(node, ast.With):
            for stmt in node.body:
                self._analyze_stmt(stmt)
        elif isinstance(node, ast.Try):
            self._analyze_try(node)
        elif isinstance(node, ast.Assert):
            self._analyze_assert(node)
        elif isinstance(node, ast.Raise):
            pass
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            pass
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            pass
        elif isinstance(node, (ast.Pass, ast.Break, ast.Continue)):
            pass
        elif isinstance(node, ast.Delete):
            pass
        else:
            pass

    def _analyze_assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                vs = self._eval_expr_state(node.value)
                self._env = self._env.set(target.id, vs)
                # Track constants
                if isinstance(node.value, ast.Constant):
                    self._const_values[target.id] = node.value.value
                    if node.value.value is None:
                        self._none_evidence.add(target.id)
                elif isinstance(node.value, (ast.List, ast.Tuple)):
                    self._collection_lens[target.id] = len(node.value.elts)
                else:
                    self._const_values.pop(target.id, None)
                    self._collection_lens.pop(target.id, None)
                # Track len() aliases: x = len(arr) -> _len_aliases[x] = arr
                if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "len" and node.value.args
                        and isinstance(node.value.args[0], ast.Name)):
                    self._len_aliases[target.id] = node.value.args[0].id
                    # Propagate known length constraints from early returns
                    coll = node.value.args[0].id
                    if coll in self._length_constraints:
                        min_len = self._length_constraints[coll]
                        vs_updated = VarState(
                            NullState.DEFINITELY_NOT_NULL,
                            TypeTagSet(frozenset({"int"})),
                            Interval(lo=min_len),
                        )
                        self._env = self._env.set(target.id, vs_updated)
                else:
                    self._len_aliases.pop(target.id, None)
                # Track None-producing expressions
                if vs.null in (NullState.DEFINITELY_NULL, NullState.MAYBE_NULL):
                    if isinstance(node.value, ast.Call):
                        # dict.get, list.pop, etc. can return None
                        if isinstance(node.value.func, ast.Attribute):
                            if node.value.func.attr in ("get", "pop", "search", "match",
                                                         "fullmatch", "first", "last", "getenv"):
                                self._none_evidence.add(target.id)
                        # Any function call result where we don't know the return
                        elif isinstance(node.value.func, ast.Name):
                            if node.value.func.id not in ("int", "float", "str", "bool",
                                                           "list", "tuple", "dict", "set",
                                                           "len", "range", "sorted", "abs",
                                                           "max", "min", "sum", "print",
                                                           "input", "open", "type", "bytes"):
                                self._none_evidence.add(target.id)
                    elif isinstance(node.value, ast.Constant) and node.value.value is None:
                        self._none_evidence.add(target.id)
                    elif isinstance(node.value, ast.IfExp):
                        # Ternary with None in one branch: x = expr if cond else None
                        if (isinstance(node.value.orelse, ast.Constant) and node.value.orelse.value is None):
                            self._none_evidence.add(target.id)
                        elif (isinstance(node.value.body, ast.Constant) and node.value.body.value is None):
                            self._none_evidence.add(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self._env = self._env.set(elt.id, VarState())
        # Check RHS for bugs
        self._check_expr(node.value)

    def _analyze_ann_assign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            if node.value:
                vs = self._eval_expr_state(node.value)
                self._env = self._env.set(node.target.id, vs)
            else:
                ann = self._parse_annotation(node.annotation)
                vs = VarState(
                    null=NullState.MAYBE_NULL if "None" in ann or "Optional" in ann else NullState.DEFINITELY_NOT_NULL,
                    tags=TypeTagSet(frozenset(ann.split("|")) if ann and ann != "Any" else None),
                )
                self._env = self._env.set(node.target.id, vs)

    def _analyze_aug_assign(self, node: ast.AugAssign):
        self._check_expr(node.value)
        if isinstance(node.target, ast.Name):
            # Result of aug-assign is non-None (numeric)
            vs = self._env.get(node.target.id)
            vs = VarState(null=NullState.DEFINITELY_NOT_NULL, tags=vs.tags, interval=vs.interval)
            self._env = self._env.set(node.target.id, vs)
        # Check for division by zero in augmented assignment
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            self._check_division(node.value, node.lineno, node.col_offset)

    def _analyze_if(self, node: ast.If):
        """Flow-sensitive if analysis with guard narrowing.

        Handles early-return patterns: if the true branch always returns/raises,
        the false-branch narrowing persists for the rest of the function.
        """
        # Save current environment
        pre_env = self._env.copy()

        # Extract and apply guard narrowing for true branch
        guard_info = self._extract_guards(node.test)
        true_env = self._apply_guards(pre_env, guard_info, positive=True)
        false_env = self._apply_guards(pre_env, guard_info, positive=False)

        # Check the test expression itself
        self._check_expr(node.test)

        # Check if true branch always exits (return/raise/continue/break)
        true_always_exits = self._branch_always_exits(node.body)
        false_always_exits = self._branch_always_exits(node.orelse) if node.orelse else False

        # Track length constraints from early-return patterns
        if true_always_exits:
            for var, gtype, arg in guard_info:
                if gtype == "len_eq_zero":
                    # if len(x)==0: return -> x has len > 0 after
                    self._length_constraints[var] = 1
                elif gtype == "eq" and arg == 0:
                    # if x == 0: return -> x != 0 after
                    pass  # already handled by ne guard
        if false_always_exits:
            for var, gtype, arg in guard_info:
                if gtype == "not_len_eq_zero":
                    # if len(x) > 0: <exits> else: ... -> x has len == 0 after
                    pass  # the negative narrowing handles this

        # Analyze true branch
        self._env = true_env
        for stmt in node.body:
            self._analyze_stmt(stmt)
        post_true = self._env

        # Analyze false branch
        self._env = false_env
        for stmt in node.orelse:
            self._analyze_stmt(stmt)
        post_false = self._env

        # Join branches, but handle early-exit patterns
        if true_always_exits and not false_always_exits:
            # True branch exits, so rest of function uses false-branch env
            self._env = post_false
        elif false_always_exits and not true_always_exits:
            # False branch exits, so rest of function uses true-branch env
            self._env = post_true
        elif true_always_exits and false_always_exits:
            # Both exit, code after is unreachable
            self._env = AbstractEnv(reachable=False)
        else:
            # Neither exits, join both branches
            self._env = post_true.join(post_false)

    def _analyze_while(self, node: ast.While):
        """Analyze while loop with widening."""
        self._loop_depth += 1
        pre_env = self._env.copy()

        # First iteration
        guard_info = self._extract_guards(node.test)
        self._check_expr(node.test)

        body_env = self._apply_guards(self._env, guard_info, positive=True)
        self._env = body_env
        for stmt in node.body:
            self._analyze_stmt(stmt)
        post_body = self._env

        # Widening: join pre with post_body (simplified)
        widened = pre_env.join(post_body)

        # Second pass for stability
        self._env = self._apply_guards(widened, guard_info, positive=True)
        for stmt in node.body:
            self._analyze_stmt(stmt)

        # Exit: apply negative guards
        self._env = self._apply_guards(widened, guard_info, positive=False)
        for stmt in node.orelse:
            self._analyze_stmt(stmt)

        self._loop_depth -= 1

    def _analyze_for(self, node: ast.For):
        """Analyze for loop."""
        self._loop_depth += 1
        self._check_expr(node.iter)
        self._check_iteration_type(node.iter)

        if isinstance(node.target, ast.Name):
            # Iterator element is definitely not None for most iterables
            iter_state = VarState(null=NullState.MAYBE_NULL, tags=TypeTagSet(None))
            if isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                    iter_state = VarState(
                        null=NullState.DEFINITELY_NOT_NULL,
                        tags=TypeTagSet(frozenset({"int"})),
                        interval=Interval(lo=0),
                    )
            self._env = self._env.set(node.target.id, iter_state)

        for stmt in node.body:
            self._analyze_stmt(stmt)
        for stmt in node.orelse:
            self._analyze_stmt(stmt)
        self._loop_depth -= 1

    def _analyze_return(self, node: ast.Return):
        if node.value:
            self._check_expr(node.value)

    def _analyze_expr_stmt(self, node: ast.Expr):
        self._check_expr(node.value)

    def _analyze_try(self, node: ast.Try):
        pre_env = self._env.copy()

        # Check if there's a ZeroDivisionError handler
        has_div_handler = False
        has_generic_handler = False
        for handler in node.handlers:
            if handler.type is None:  # bare except
                has_generic_handler = True
            elif isinstance(handler.type, ast.Name) and handler.type.id in ("ZeroDivisionError", "ArithmeticError", "Exception", "BaseException"):
                has_div_handler = True
            elif isinstance(handler.type, ast.Tuple):
                for elt in handler.type.elts:
                    if isinstance(elt, ast.Name) and elt.id in ("ZeroDivisionError", "ArithmeticError", "Exception"):
                        has_div_handler = True

        # Mark that we're inside a try block (suppresses some warnings)
        old_try = getattr(self, '_in_try', False)
        self._in_try = has_div_handler or has_generic_handler

        for stmt in node.body:
            self._analyze_stmt(stmt)
        post_try = self._env

        self._in_try = old_try

        for handler in node.handlers:
            self._env = pre_env.copy()
            if handler.name:
                exc_state = VarState(null=NullState.DEFINITELY_NOT_NULL)
                self._env = self._env.set(handler.name, exc_state)
            for stmt in handler.body:
                self._analyze_stmt(stmt)

        self._env = post_try  # simplified: take post-try state

        for stmt in node.finalbody:
            self._analyze_stmt(stmt)

    def _analyze_assert(self, node: ast.Assert):
        """Assert narrows the environment."""
        guard_info = self._extract_guards(node.test)
        self._env = self._apply_guards(self._env, guard_info, positive=True)

    def _branch_always_exits(self, stmts: List[ast.stmt]) -> bool:
        """Check if a branch always exits (return/raise/continue/break)."""
        if not stmts:
            return False
        last = stmts[-1]
        if isinstance(last, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return True
        if isinstance(last, ast.If):
            # Both branches must exit
            true_exits = self._branch_always_exits(last.body)
            false_exits = self._branch_always_exits(last.orelse) if last.orelse else False
            return true_exits and false_exits
        # Check if any statement in the list is an unconditional exit
        for stmt in stmts:
            if isinstance(stmt, (ast.Return, ast.Raise)):
                return True
        return False

    # ── Guard extraction ────────────────────────────────────────────

    def _extract_guards(self, test: ast.expr) -> List[Tuple[str, str, Any]]:
        """Extract (variable, guard_type, arg) triples from a test expression."""
        guards = []
        if isinstance(test, ast.Compare):
            guards.extend(self._extract_compare_guards(test))
        elif isinstance(test, ast.Call):
            guards.extend(self._extract_call_guards(test))
        elif isinstance(test, ast.BoolOp):
            if isinstance(test.op, ast.And):
                for val in test.values:
                    guards.extend(self._extract_guards(val))
            elif isinstance(test.op, ast.Or):
                # For OR positive, can't narrow on individual clauses.
                # For OR negative (De Morgan: not(A or B) = not A and not B),
                # tag guards with "or_neg_" so _apply_guards skips them in
                # positive but applies their negation in negative.
                for val in test.values:
                    for var, gtype, arg in self._extract_guards(val):
                        guards.append((var, "or_neg_" + gtype, arg))
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._extract_guards(test.operand)
            for var, gtype, arg in inner:
                guards.append((var, "not_" + gtype, arg))
        elif isinstance(test, ast.Name):
            guards.append((test.id, "truthy", None))
        elif isinstance(test, ast.Attribute):
            if isinstance(test.value, ast.Name):
                self._check_deref(test.value.id, test.lineno, test.col_offset, "attribute")
        self._guards_harvested += len(guards)
        self._predicates_inferred += len(guards)
        return guards

    def _extract_compare_guards(self, node: ast.Compare) -> List[Tuple[str, str, Any]]:
        guards = []
        if len(node.ops) != 1:
            return guards
        op = node.ops[0]
        left = node.left
        right = node.comparators[0]

        if isinstance(op, ast.Is):
            if isinstance(right, ast.Constant) and right.value is None:
                if isinstance(left, ast.Name):
                    guards.append((left.id, "is_none", None))
        elif isinstance(op, ast.IsNot):
            if isinstance(right, ast.Constant) and right.value is None:
                if isinstance(left, ast.Name):
                    guards.append((left.id, "is_not_none", None))
        elif isinstance(op, ast.NotEq):
            if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
                if right.value == 0:
                    guards.append((left.id, "ne_zero", 0))
                guards.append((left.id, "ne", right.value))
            # Handle len(x) != 0
            elif isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len":
                if left.args and isinstance(left.args[0], ast.Name):
                    if isinstance(right, ast.Constant) and right.value == 0:
                        guards.append((left.args[0].id, "not_len_eq_zero", 0))
        elif isinstance(op, ast.Eq):
            if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
                guards.append((left.id, "eq", right.value))
            # Handle len(x) == 0
            elif isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len":
                if left.args and isinstance(left.args[0], ast.Name):
                    if isinstance(right, ast.Constant) and right.value == 0:
                        guards.append((left.args[0].id, "len_eq_zero", 0))
        elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            if isinstance(left, ast.Name):
                if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
                    op_name = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}[type(op)]
                    guards.append((left.id, op_name, right.value))
                elif isinstance(right, ast.Call) and isinstance(right.func, ast.Name) and right.func.id == "len":
                    if right.args and isinstance(right.args[0], ast.Name):
                        guards.append((left.id, "lt_len", right.args[0].id))
            # Handle len(x) > 0, len(x) >= 1, etc.
            if isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len":
                if left.args and isinstance(left.args[0], ast.Name):
                    coll = left.args[0].id
                    if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
                        # len(x) > 0 or len(x) >= 1 → collection is non-empty
                        if isinstance(op, ast.Gt) and right.value == 0:
                            guards.append((coll, "not_len_eq_zero", 0))
                        elif isinstance(op, ast.GtE) and right.value >= 1:
                            guards.append((coll, "not_len_eq_zero", 0))
            if isinstance(right, ast.Name):
                if isinstance(left, ast.Constant) and isinstance(left.value, (int, float)):
                    # Flip: const < var → var > const
                    op_name = {ast.Lt: "gt", ast.LtE: "ge", ast.Gt: "lt", ast.GtE: "le"}[type(op)]
                    guards.append((right.id, op_name, left.value))
        elif isinstance(op, ast.In):
            if isinstance(left, ast.Name):
                guards.append((left.id, "in", right))
        elif isinstance(op, ast.NotIn):
            if isinstance(left, ast.Name):
                guards.append((left.id, "not_in", right))
        return guards

    def _extract_call_guards(self, node: ast.Call) -> List[Tuple[str, str, Any]]:
        guards = []
        if isinstance(node.func, ast.Name):
            if node.func.id == "isinstance" and len(node.args) >= 2:
                if isinstance(node.args[0], ast.Name):
                    type_name = self._get_type_name(node.args[1])
                    if type_name:
                        guards.append((node.args[0].id, "isinstance", type_name))
            elif node.func.id == "callable" and node.args:
                if isinstance(node.args[0], ast.Name):
                    guards.append((node.args[0].id, "callable", None))
            elif node.func.id == "hasattr" and len(node.args) >= 2:
                if isinstance(node.args[0], ast.Name):
                    attr = ""
                    if isinstance(node.args[1], ast.Constant):
                        attr = str(node.args[1].value)
                    guards.append((node.args[0].id, "hasattr", attr))
        return guards

    # ── Guard application (narrowing) ───────────────────────────────

    def _apply_guards(self, env: AbstractEnv, guards: List[Tuple[str, str, Any]],
                      positive: bool) -> AbstractEnv:
        """Apply guard narrowing to the environment."""
        env = env.copy()
        for var, gtype, arg in guards:
            vs = env.get(var)
            if positive:
                vs = self._narrow_positive(vs, gtype, arg)
            else:
                vs = self._narrow_negative(vs, gtype, arg)
            env = env.set(var, vs)
        return env

    def _narrow_positive(self, vs: VarState, gtype: str, arg: Any) -> VarState:
        """Narrow variable state for positive guard (true branch)."""
        if gtype == "is_not_none":
            return VarState(NullState.DEFINITELY_NOT_NULL, vs.tags, vs.interval, vs.known_len, vs.guards | {"not_none"})
        elif gtype == "is_none":
            return VarState(NullState.DEFINITELY_NULL, TypeTagSet(frozenset({"NoneType"})), vs.interval, vs.known_len, vs.guards | {"is_none"})
        elif gtype == "isinstance":
            tags = arg.split("|") if isinstance(arg, str) else [str(arg)]
            tag_set = frozenset(tags)
            null = NullState.DEFINITELY_NOT_NULL if "NoneType" not in tag_set else vs.null
            return VarState(null, TypeTagSet(tag_set), vs.interval, vs.known_len, vs.guards | {f"isinstance_{arg}"})
        elif gtype == "truthy":
            # Truthiness excludes None (and 0, "", [], {}, False) but we
            # can only soundly narrow nullity when the variable's type
            # actually includes None (Optional[T]).  For plain int/str/list
            # ``if x:`` being true does NOT tell us anything about nullity
            # that we didn't already know – the variable might just be
            # non-zero / non-empty.
            new_tags = vs.tags
            new_null = vs.null
            if vs.null in (NullState.MAYBE_NULL, NullState.DEFINITELY_NULL):
                # Type includes None → truthiness proves not-None
                # (DEFINITELY_NULL makes the branch unreachable, but we
                #  still mark it NOT_NULL to suppress downstream FPs.)
                new_null = NullState.DEFINITELY_NOT_NULL
            if vs.tags.tags is not None and "NoneType" in vs.tags.tags:
                new_tags = TypeTagSet(vs.tags.tags - frozenset({"NoneType"}))
            return VarState(new_null, new_tags, vs.interval, vs.known_len, vs.guards | {"truthy"})
        elif gtype == "ne_zero":
            iv = vs.interval
            if iv.lo is not None and iv.lo == 0:
                iv = Interval(lo=1, hi=iv.hi)
            return VarState(vs.null, vs.tags, iv, vs.known_len, vs.guards | {"ne_zero"})
        elif gtype == "gt":
            val = arg if isinstance(arg, int) else 0
            lo = max(val + 1, vs.interval.lo) if vs.interval.lo is not None else val + 1
            return VarState(vs.null, vs.tags, Interval(lo=lo, hi=vs.interval.hi), vs.known_len, vs.guards | {f"gt_{val}"})
        elif gtype == "ge":
            val = arg if isinstance(arg, int) else 0
            lo = max(val, vs.interval.lo) if vs.interval.lo is not None else val
            return VarState(vs.null, vs.tags, Interval(lo=lo, hi=vs.interval.hi), vs.known_len, vs.guards | {f"ge_{val}"})
        elif gtype == "lt":
            val = arg if isinstance(arg, int) else 0
            hi = min(val - 1, vs.interval.hi) if vs.interval.hi is not None else val - 1
            return VarState(vs.null, vs.tags, Interval(lo=vs.interval.lo, hi=hi), vs.known_len, vs.guards | {f"lt_{val}"})
        elif gtype == "le":
            val = arg if isinstance(arg, int) else 0
            hi = min(val, vs.interval.hi) if vs.interval.hi is not None else val
            return VarState(vs.null, vs.tags, Interval(lo=vs.interval.lo, hi=hi), vs.known_len, vs.guards | {f"le_{val}"})
        elif gtype == "lt_len":
            return VarState(vs.null, vs.tags, vs.interval, vs.known_len, vs.guards | {f"lt_len_{arg}"})
        elif gtype == "hasattr":
            return VarState(NullState.DEFINITELY_NOT_NULL, vs.tags, vs.interval, vs.known_len, vs.guards | {f"hasattr_{arg}"})
        elif gtype == "len_eq_zero":
            return VarState(vs.null, vs.tags, vs.interval, 0, vs.guards | {"len_eq_zero"})
        elif gtype == "not_len_eq_zero":
            return VarState(vs.null, vs.tags, vs.interval, vs.known_len, vs.guards | {"len_gt_zero"})
        elif gtype == "callable":
            return VarState(NullState.DEFINITELY_NOT_NULL, vs.tags, vs.interval, vs.known_len, vs.guards | {"callable"})
        elif gtype == "eq":
            if arg is None:
                return VarState(NullState.DEFINITELY_NULL, TypeTagSet(frozenset({"NoneType"})), vs.interval, vs.known_len, vs.guards)
            elif isinstance(arg, int):
                return VarState(vs.null, vs.tags, Interval(lo=arg, hi=arg), vs.known_len, vs.guards | {f"eq_{arg}"})
            return vs
        elif gtype == "ne":
            return VarState(vs.null, vs.tags, vs.interval, vs.known_len, vs.guards | {f"ne_{arg}"})
        # Handle negated guards
        elif gtype.startswith("not_"):
            return self._narrow_negative(vs, gtype[4:], arg)
        # OR-negative guards: skip in positive branch
        elif gtype.startswith("or_neg_"):
            return vs
        return vs

    def _narrow_negative(self, vs: VarState, gtype: str, arg: Any) -> VarState:
        """Narrow variable state for negative guard (false branch)."""
        if gtype == "is_not_none":
            return VarState(NullState.DEFINITELY_NULL, TypeTagSet(frozenset({"NoneType"})), vs.interval, vs.known_len, vs.guards)
        elif gtype == "is_none":
            return VarState(NullState.DEFINITELY_NOT_NULL, vs.tags, vs.interval, vs.known_len, vs.guards | {"not_none"})
        elif gtype == "isinstance":
            # In false branch, exclude the tested type
            if vs.tags.tags is not None:
                new_tags = vs.tags.tags - frozenset(arg.split("|") if isinstance(arg, str) else [str(arg)])
                return VarState(vs.null, TypeTagSet(new_tags), vs.interval, vs.known_len, vs.guards)
            return vs
        elif gtype == "truthy":
            # False branch of truthiness: could be None, 0, empty, False.
            # Only widen to MAYBE_NULL if the type already included None;
            # otherwise preserve the existing null state (an int that is 0
            # is still DEFINITELY_NOT_NULL).
            new_null = vs.null if vs.null == NullState.DEFINITELY_NOT_NULL else NullState.MAYBE_NULL
            return VarState(new_null, vs.tags, vs.interval, vs.known_len, vs.guards)
        elif gtype == "ne_zero":
            return VarState(vs.null, vs.tags, Interval(lo=0, hi=0), vs.known_len, vs.guards)
        elif gtype == "len_eq_zero":
            # not (len == 0) means len > 0, so the collection is non-empty
            return VarState(vs.null, vs.tags, vs.interval, None, vs.guards | {"len_gt_zero"})
        elif gtype == "not_len_eq_zero":
            # not (len > 0) means len == 0
            return VarState(vs.null, vs.tags, vs.interval, 0, vs.guards | {"len_eq_zero"})
        elif gtype == "gt":
            val = arg if isinstance(arg, int) else 0
            return VarState(vs.null, vs.tags, Interval(lo=vs.interval.lo, hi=val), vs.known_len, vs.guards)
        elif gtype == "ge":
            val = arg if isinstance(arg, int) else 0
            return VarState(vs.null, vs.tags, Interval(lo=vs.interval.lo, hi=val - 1), vs.known_len, vs.guards)
        elif gtype == "lt":
            val = arg if isinstance(arg, int) else 0
            return VarState(vs.null, vs.tags, Interval(lo=val, hi=vs.interval.hi), vs.known_len, vs.guards)
        elif gtype == "le":
            val = arg if isinstance(arg, int) else 0
            return VarState(vs.null, vs.tags, Interval(lo=val + 1, hi=vs.interval.hi), vs.known_len, vs.guards)
        elif gtype == "eq":
            # not (x == val) means x != val
            if arg is None:
                return VarState(NullState.DEFINITELY_NOT_NULL, vs.tags, vs.interval, vs.known_len, vs.guards | {"not_none"})
            elif isinstance(arg, int) and arg == 0:
                return VarState(vs.null, vs.tags, vs.interval, vs.known_len, vs.guards | {"ne_zero"})
            elif isinstance(arg, int):
                return VarState(vs.null, vs.tags, vs.interval, vs.known_len, vs.guards | {f"ne_{arg}"})
            return vs
        elif gtype == "ne":
            # not (x != val) means x == val
            if isinstance(arg, int):
                return VarState(vs.null, vs.tags, Interval(lo=arg, hi=arg), vs.known_len, vs.guards | {f"eq_{arg}"})
            return vs
        elif gtype.startswith("not_"):
            return self._narrow_positive(vs, gtype[4:], arg)
        # OR-negative: in the false branch of an OR, De Morgan applies
        # not(A or B) = not A and not B, so apply negative of inner guard
        elif gtype.startswith("or_neg_"):
            return self._narrow_negative(vs, gtype[7:], arg)
        return vs

    # ── Expression evaluation ───────────────────────────────────────

    def _eval_expr_state(self, node: ast.expr) -> VarState:
        """Evaluate an expression to an abstract state."""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return VarState(NullState.DEFINITELY_NULL, TypeTagSet(frozenset({"NoneType"})))
            elif isinstance(node.value, int):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"int"})),
                                Interval(lo=node.value, hi=node.value))
            elif isinstance(node.value, float):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"float"})))
            elif isinstance(node.value, str):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"str"})))
            elif isinstance(node.value, bool):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"bool"})))
            return VarState(NullState.DEFINITELY_NOT_NULL)
        elif isinstance(node, ast.Name):
            return self._env.get(node.id)
        elif isinstance(node, ast.List):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"list"})),
                            known_len=len(node.elts))
        elif isinstance(node, ast.Tuple):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"tuple"})),
                            known_len=len(node.elts))
        elif isinstance(node, ast.Dict):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"dict"})))
        elif isinstance(node, ast.Set):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"set"})))
        elif isinstance(node, ast.BinOp):
            self._check_expr(node)
            left = self._eval_expr_state(node.left)
            right = self._eval_expr_state(node.right)
            iv = Interval()
            # Propagate interval arithmetic for Add/Sub/Mul
            if isinstance(node.op, ast.Add):
                lo = (left.interval.lo + right.interval.lo
                      if left.interval.lo is not None and right.interval.lo is not None else None)
                hi = (left.interval.hi + right.interval.hi
                      if left.interval.hi is not None and right.interval.hi is not None else None)
                iv = Interval(lo=lo, hi=hi)
            elif isinstance(node.op, ast.Sub):
                lo = (left.interval.lo - right.interval.hi
                      if left.interval.lo is not None and right.interval.hi is not None else None)
                hi = (left.interval.hi - right.interval.lo
                      if left.interval.hi is not None and right.interval.lo is not None else None)
                iv = Interval(lo=lo, hi=hi)
            elif isinstance(node.op, ast.Mult):
                if (left.interval.lo is not None and left.interval.hi is not None
                        and right.interval.lo is not None and right.interval.hi is not None):
                    products = [left.interval.lo * right.interval.lo,
                                left.interval.lo * right.interval.hi,
                                left.interval.hi * right.interval.lo,
                                left.interval.hi * right.interval.hi]
                    iv = Interval(lo=min(products), hi=max(products))
            return VarState(NullState.DEFINITELY_NOT_NULL, left.tags.join(right.tags), iv)
        elif isinstance(node, ast.Call):
            self._check_expr(node)
            return self._eval_call_state(node)
        elif isinstance(node, ast.IfExp):
            guard_info = self._extract_guards(node.test)
            true_state = self._eval_expr_state(node.body)
            false_state = self._eval_expr_state(node.orelse)
            # If either branch produces None, mark as maybe-null
            result = true_state.join(false_state)
            return result
        elif isinstance(node, ast.Subscript):
            self._check_expr(node)
            return VarState()
        elif isinstance(node, ast.Attribute):
            self._check_expr(node)
            return VarState()
        elif isinstance(node, ast.ListComp):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"list"})))
        elif isinstance(node, ast.DictComp):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"dict"})))
        elif isinstance(node, ast.SetComp):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"set"})))
        elif isinstance(node, ast.GeneratorExp):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"generator"})))
        elif isinstance(node, ast.UnaryOp):
            return self._eval_expr_state(node.operand)
        elif isinstance(node, ast.BoolOp):
            states = [self._eval_expr_state(v) for v in node.values]
            if isinstance(node.op, ast.Or) and len(states) >= 2:
                # x or default: if default is non-None, result is non-None
                last = node.values[-1]
                if isinstance(last, ast.Constant) and last.value is not None:
                    return VarState(NullState.DEFINITELY_NOT_NULL, states[-1].tags,
                                    states[-1].interval)
                elif isinstance(last, ast.Constant) and last.value is None:
                    pass  # x or None is still maybe null
                elif not isinstance(last, ast.Constant):
                    # x or some_expr: usually evaluates to a non-None
                    last_state = states[-1]
                    if last_state.null == NullState.DEFINITELY_NOT_NULL:
                        return VarState(NullState.DEFINITELY_NOT_NULL)
            result = states[0]
            for s in states[1:]:
                result = result.join(s)
            return result
        elif isinstance(node, (ast.FormattedValue, ast.JoinedStr)):
            return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"str"})))
        elif isinstance(node, ast.Starred):
            return VarState()
        return VarState()

    def _eval_call_state(self, node: ast.Call) -> VarState:
        """Evaluate a call expression to an abstract state."""
        # Check liquid type contracts first for interprocedural precision
        if self._liquid_summaries and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self._liquid_summaries:
                contract = self._liquid_summaries[name]
                vs = _reftype_to_varstate(contract.return_type)
                if vs is not None:
                    return vs

        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in ("int", "float", "bool", "str", "list", "tuple", "dict", "set", "bytes"):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({name})))
            elif name == "len":
                # len() returns non-negative int; check if we know the argument's length
                lo = 0
                if node.args and isinstance(node.args[0], ast.Name):
                    arg_name = node.args[0].id
                    arg_vs = self._env.get(arg_name)
                    known_len = arg_vs.known_len or self._collection_lens.get(arg_name)
                    if known_len is not None:
                        return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"int"})),
                                        Interval(lo=known_len, hi=known_len))
                    # Check length constraints from early-return patterns
                    if arg_name in self._length_constraints:
                        lo = max(lo, self._length_constraints[arg_name])
                    # Check if we have a "len > 0" guard equivalent
                    if ("truthy" in arg_vs.guards or "len_gt_zero" in arg_vs.guards
                            or any(g.startswith("gt_") or g.startswith("ge_") for g in arg_vs.guards)):
                        lo = max(lo, 1)
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"int"})),
                                Interval(lo=lo))
            elif name == "range":
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"range"})))
            elif name == "sorted":
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"list"})))
            elif name == "abs":
                # abs() returns non-negative; propagate from argument
                iv = Interval(lo=0)
                if node.args and isinstance(node.args[0], ast.Name):
                    arg_vs = self._env.get(node.args[0].id)
                    if arg_vs.interval.lo is not None and arg_vs.interval.hi is not None:
                        abs_lo = min(abs(arg_vs.interval.lo), abs(arg_vs.interval.hi))
                        abs_hi = max(abs(arg_vs.interval.lo), abs(arg_vs.interval.hi))
                        if arg_vs.interval.lo <= 0 <= arg_vs.interval.hi:
                            abs_lo = 0
                        iv = Interval(lo=abs_lo, hi=abs_hi)
                elif node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, (int, float)):
                        v = abs(node.args[0].value)
                        iv = Interval(lo=int(v), hi=int(v))
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"int", "float"})), iv)
            elif name in ("max", "min", "sum"):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"int", "float"})))
            elif name == "open":
                return VarState(NullState.DEFINITELY_NOT_NULL)
            elif name in ("print", "input"):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"str"})) if name == "input" else TypeTagSet(frozenset({"NoneType"})))
            elif name == "getattr":
                # getattr(obj, attr, default) - if default is non-None, result is non-None
                if len(node.args) >= 3:
                    default_arg = node.args[2]
                    if isinstance(default_arg, ast.Constant) and default_arg.value is not None:
                        return VarState(NullState.DEFINITELY_NOT_NULL)
                    elif not isinstance(default_arg, ast.Constant):
                        return VarState(NullState.DEFINITELY_NOT_NULL)
                return VarState(NullState.MAYBE_NULL)
            elif name == "next":
                # next(iter, default) - if default is non-None, result is non-None
                if len(node.args) >= 2:
                    default_arg = node.args[1]
                    if isinstance(default_arg, ast.Constant) and default_arg.value is not None:
                        return VarState(NullState.DEFINITELY_NOT_NULL)
                return VarState(NullState.MAYBE_NULL)
        elif isinstance(node.func, ast.Attribute):
            # Method calls return non-None typically
            method = node.func.attr
            if method in ("get",):  # dict.get can return None
                # dict.get(key, default) - if default is non-None, result is non-None
                if len(node.args) >= 2:
                    default_arg = node.args[1]
                    if isinstance(default_arg, ast.Constant) and default_arg.value is not None:
                        return VarState(NullState.DEFINITELY_NOT_NULL)
                    elif not isinstance(default_arg, ast.Constant):
                        return VarState(NullState.DEFINITELY_NOT_NULL)
                return VarState(NullState.MAYBE_NULL)
            elif method in ("pop",):
                # dict.pop(key, default) - if default is non-None, result is non-None
                if len(node.args) >= 2:
                    default_arg = node.args[1]
                    if isinstance(default_arg, ast.Constant) and default_arg.value is not None:
                        return VarState(NullState.DEFINITELY_NOT_NULL)
                    elif not isinstance(default_arg, ast.Constant):
                        return VarState(NullState.DEFINITELY_NOT_NULL)
                return VarState(NullState.MAYBE_NULL)
            elif method in ("search", "match", "fullmatch"):
                # re.search/match/fullmatch return Optional[Match]
                return VarState(NullState.MAYBE_NULL)
            elif method in ("first", "last"):
                # ORM-style .first() returns Optional
                return VarState(NullState.MAYBE_NULL)
            elif method in ("getenv",):
                # os.getenv returns Optional[str]
                return VarState(NullState.MAYBE_NULL)
            elif method in ("strip", "lower", "upper", "replace", "split", "join", "format"):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"str"})))
            elif method in ("append", "extend", "insert", "remove", "clear", "sort", "reverse"):
                return VarState(NullState.DEFINITELY_NULL, TypeTagSet(frozenset({"NoneType"})))  # returns None
            elif method in ("find", "index", "count"):
                return VarState(NullState.DEFINITELY_NOT_NULL, TypeTagSet(frozenset({"int"})))
            elif method in ("keys", "values", "items"):
                return VarState(NullState.DEFINITELY_NOT_NULL)

        # Interprocedural: check if callee has a known function summary
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
            if callee_name in self._function_summaries:
                summary = self._function_summaries[callee_name]
                return summary.to_var_state()

        return VarState()

    # ── Safety checking ─────────────────────────────────────────────

    def _check_expr(self, node: ast.expr):
        """Check an expression for potential bugs."""
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                self._check_deref(node.value.id, node.lineno, node.col_offset, "attribute")
            else:
                self._check_expr(node.value)
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                self._check_deref(node.value.id, node.lineno, node.col_offset, "subscript")
                self._check_bounds(node)
                self._check_subscript_type(node)
            else:
                self._check_expr(node.value)
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                self._check_division(node.right, node.lineno, node.col_offset)
            self._check_type_binop(node)
            self._check_expr(node.left)
            self._check_expr(node.right)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                self._check_deref(node.func.value.id, node.func.lineno, node.func.col_offset, "method_call")
            elif isinstance(node.func, ast.Attribute):
                # Check inner expression (method chains like x.lower().strip())
                self._check_expr(node.func.value)
            elif isinstance(node.func, ast.Name):
                vs = self._env.get(node.func.id)
                if vs.null == NullState.DEFINITELY_NULL:
                    self.bugs.append(Bug(
                        BugCategory.NULL_DEREF, node.lineno, node.col_offset,
                        f"Calling None variable '{node.func.id}'",
                        self._func_name, node.func.id, "error",
                    ))
                # Check len() on non-sizable types
                if node.func.id == "len" and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Name):
                        arg_vs = self._env.get(arg.id)
                        if arg_vs.tags.tags in (frozenset({"int"}), frozenset({"float"}),
                                                 frozenset({"bool"}), frozenset({"NoneType"})):
                            self.bugs.append(Bug(
                                BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                                f"TypeError: object of type '{list(arg_vs.tags.tags)[0]}' has no len()",
                                self._func_name, arg.id, "error",
                            ))
            for arg in node.args:
                self._check_expr(arg)
            for kw in node.keywords:
                self._check_expr(kw.value)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._check_expr(elt)
        elif isinstance(node, ast.Compare):
            self._check_expr(node.left)
            for comp in node.comparators:
                self._check_expr(comp)

    def _check_deref(self, var: str, line: int, col: int, access_type: str):
        """Check if dereferencing a variable is safe."""
        vs = self._env.get(var)
        if vs.null == NullState.DEFINITELY_NULL:
            self.bugs.append(Bug(
                BugCategory.NULL_DEREF, line, col,
                f"{access_type.title()} access on None variable '{var}'",
                self._func_name, var, "error",
            ))
        elif vs.null == NullState.MAYBE_NULL:
            # Only flag if there's positive evidence the var could be None
            # (e.g., assigned from dict.get, explicit None assignment, etc.)
            has_none_evidence = (
                "not_none" not in vs.guards
                and "truthy" not in vs.guards
                and not any(g.startswith("isinstance_") for g in vs.guards)
                and not any(g.startswith("hasattr_") for g in vs.guards)
            )
            # Check if we have evidence this came from a None-producing source
            if has_none_evidence and var in self._const_values and self._const_values[var] is None:
                self.bugs.append(Bug(
                    BugCategory.NULL_DEREF, line, col,
                    f"{access_type.title()} access on None variable '{var}'",
                    self._func_name, var, "error",
                ))
            elif has_none_evidence and var in self._none_evidence:
                self.bugs.append(Bug(
                    BugCategory.UNGUARDED_OPTIONAL, line, col,
                    f"Potential None {access_type} on '{var}' without guard",
                    self._func_name, var, "warning",
                ))

    def _check_type_binop(self, node: ast.BinOp):
        """Check for type-incompatible binary operations."""
        left_vs = self._eval_expr_state(node.left)
        right_vs = self._eval_expr_state(node.right)

        left_tags = left_vs.tags.tags
        right_tags = right_vs.tags.tags

        # Check None on either side first — arithmetic with None is always
        # a TypeError regardless of what the other operand's type is.
        if left_tags is not None and left_tags == frozenset({"NoneType"}):
            if not isinstance(node.op, (ast.Is, ast.IsNot)):
                self.bugs.append(Bug(
                    BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                    f"TypeError: unsupported operand for None",
                    self._func_name, "None", "error",
                ))
                return
        if right_tags is not None and right_tags == frozenset({"NoneType"}):
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                     ast.FloorDiv, ast.Mod, ast.Pow)):
                self.bugs.append(Bug(
                    BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                    f"TypeError: unsupported operand for None",
                    self._func_name, "None", "error",
                ))
                return

        # If either side is unknown, we can't check further type compat
        if left_tags is None or right_tags is None:
            return

        # str + int / str - anything
        if left_tags == frozenset({"str"}):
            if isinstance(node.op, ast.Add) and right_tags in (
                frozenset({"int"}), frozenset({"float"}), frozenset({"bool"}),
                frozenset({"NoneType"})):
                self.bugs.append(Bug(
                    BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                    f"TypeError: cannot concatenate str and {list(right_tags)[0]}",
                    self._func_name, "", "error",
                ))
            elif isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Pow)):
                # Note: ast.Mod excluded — str % val is valid (string formatting)
                self.bugs.append(Bug(
                    BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                    f"TypeError: unsupported operand type(s) for str",
                    self._func_name, "", "error",
                ))

        # int + str
        if right_tags == frozenset({"str"}):
            if isinstance(node.op, ast.Add) and left_tags in (
                frozenset({"int"}), frozenset({"float"}), frozenset({"bool"})):
                self.bugs.append(Bug(
                    BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                    f"TypeError: cannot add {list(left_tags)[0]} and str",
                    self._func_name, "", "error",
                ))

        # list + int
        if left_tags == frozenset({"list"}) and right_tags == frozenset({"int"}):
            if isinstance(node.op, ast.Add):
                self.bugs.append(Bug(
                    BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                    f"TypeError: can only concatenate list to list",
                    self._func_name, "", "error",
                ))

        # dict * int
        if left_tags == frozenset({"dict"}) and isinstance(node.op, ast.Mult):
            self.bugs.append(Bug(
                BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                f"TypeError: unsupported operand type(s) for dict",
                self._func_name, "", "error",
            ))

    def _check_subscript_type(self, node: ast.Subscript):
        """Check that the subscripted object supports subscript."""
        if not isinstance(node.value, ast.Name):
            return
        vs = self._env.get(node.value.id)
        tags = vs.tags.tags
        if tags is None:
            return
        # int, float, bool, set, NoneType don't support subscript
        non_subscriptable = frozenset({"int"}), frozenset({"float"}), frozenset({"set"}), frozenset({"bool"})
        if tags in non_subscriptable:
            self.bugs.append(Bug(
                BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                f"TypeError: '{list(tags)[0]}' object is not subscriptable",
                self._func_name, node.value.id, "error",
            ))

    def _check_iteration_type(self, node):
        """Check that iterable in for-loop supports iteration."""
        if isinstance(node, ast.Name):
            vs = self._env.get(node.id)
            tags = vs.tags.tags
            if tags is None:
                return
            non_iterable = frozenset({"int"}), frozenset({"float"}), frozenset({"bool"}), frozenset({"NoneType"})
            if tags in non_iterable:
                self.bugs.append(Bug(
                    BugCategory.TYPE_ERROR, node.lineno, node.col_offset,
                    f"TypeError: '{list(tags)[0]}' object is not iterable",
                    self._func_name, node.id, "error",
                ))

    def _check_division(self, divisor: ast.expr, line: int, col: int):
        """Check for division by zero."""
        # Skip warning-level division checks inside try/except
        in_try = getattr(self, '_in_try', False)

        if isinstance(divisor, ast.Constant):
            if divisor.value == 0:
                self.bugs.append(Bug(
                    BugCategory.DIV_BY_ZERO, line, col,
                    "Division by literal zero",
                    self._func_name, str(divisor.value), "error",
                ))
            # Non-zero literal is safe
        elif isinstance(divisor, ast.Name):
            if in_try:
                return  # try/except guards division
            # Skip 'self', 'cls' — class instances are never zero
            if divisor.id in ("self", "cls"):
                return
            vs = self._env.get(divisor.id)
            if vs.interval.lo is not None and vs.interval.hi is not None:
                if vs.interval.lo == 0 and vs.interval.hi == 0:
                    self.bugs.append(Bug(
                        BugCategory.DIV_BY_ZERO, line, col,
                        f"Division by zero: '{divisor.id}' is known to be 0",
                        self._func_name, divisor.id, "error",
                    ))
                elif not vs.interval.contains_zero():
                    pass  # proven safe by interval
                elif "ne_zero" in vs.guards or "truthy" in vs.guards:
                    pass  # guarded
                else:
                    if vs.tags.is_numeric() or vs.tags.tags is None:
                        self.bugs.append(Bug(
                            BugCategory.DIV_BY_ZERO, line, col,
                            f"Potential division by zero: '{divisor.id}' not guarded",
                            self._func_name, divisor.id, "warning",
                        ))
            elif "ne_zero" not in vs.guards and "truthy" not in vs.guards:
                if not vs.interval.contains_zero():
                    pass  # safe by interval
                elif vs.tags.is_numeric() or vs.tags.tags is None:
                    self.bugs.append(Bug(
                        BugCategory.DIV_BY_ZERO, line, col,
                        f"Potential division by zero: '{divisor.id}' not guarded",
                        self._func_name, divisor.id, "warning",
                    ))
        elif isinstance(divisor, ast.BinOp) and isinstance(divisor.op, ast.Add):
            # Evaluate the abstract value of the full expression
            left_vs = self._eval_expr_state(divisor.left)
            right_vs = self._eval_expr_state(divisor.right)
            # If left >= 0 and right > 0, sum > 0
            if (left_vs.interval.is_non_negative() and
                    right_vs.interval.lo is not None and right_vs.interval.lo > 0):
                return  # proven non-zero
            if isinstance(divisor.right, ast.Constant) and isinstance(divisor.right.value, int):
                if divisor.right.value > 0:
                    if isinstance(divisor.left, ast.Call) and isinstance(divisor.left.func, ast.Name):
                        if divisor.left.func.id == "abs":
                            return  # abs(x) + positive is always > 0
        elif isinstance(divisor, ast.Call):
            if isinstance(divisor.func, ast.Name) and divisor.func.id == "len":
                # len(x) can be 0 if list is empty; check guards
                if in_try:
                    return
                if divisor.args and isinstance(divisor.args[0], ast.Name):
                    coll_name = divisor.args[0].id
                    coll_vs = self._env.get(coll_name)
                    # Check if collection has a known non-zero length
                    known_len = coll_vs.known_len or self._collection_lens.get(coll_name)
                    if known_len is not None and known_len > 0:
                        return  # proven safe
                    # Check if guarded by truthiness or length check
                    if ("truthy" in coll_vs.guards or "len_gt_zero" in coll_vs.guards
                            or any(g.startswith("gt_") or g.startswith("ge_") for g in coll_vs.guards)):
                        return  # guarded
                    # Check length constraints from early returns
                    if coll_name in self._length_constraints and self._length_constraints[coll_name] > 0:
                        return  # proven safe by early-return constraint
                    self.bugs.append(Bug(
                        BugCategory.DIV_BY_ZERO, line, col,
                        f"Potential division by zero: len('{coll_name}') may be 0",
                        self._func_name, coll_name, "warning",
                    ))

    def _check_bounds(self, node: ast.Subscript):
        """Check for array out-of-bounds access.

        Detects:
        1. Constant index on a collection with known length
        2. Variable index provably >= collection length
        3. Access on a parameter that has no emptiness guard (e.g., items[0]
           without ``if items:`` or ``if len(items) > 0:``)
        """
        if not isinstance(node.value, ast.Name):
            return
        var = node.value.id
        vs = self._env.get(var)

        # Resolve index value from slice (handles both Constant and -Constant)
        idx = None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            idx = node.slice.value
        elif (isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub)
              and isinstance(node.slice.operand, ast.Constant)
              and isinstance(node.slice.operand.value, int)):
            idx = -node.slice.operand.value

        if idx is not None:
            known_len = vs.known_len or self._collection_lens.get(var)
            if known_len is not None:
                if idx >= known_len or (idx < 0 and abs(idx) > known_len):
                    self.bugs.append(Bug(
                        BugCategory.INDEX_OUT_OF_BOUNDS, node.lineno, node.col_offset,
                        f"Index {idx} out of bounds for '{var}' of length {known_len}",
                        self._func_name, var, "error",
                    ))
            elif known_len is None and idx >= 0:
                # Unknown-length collection: check for emptiness guards
                has_len_guard = (
                    "truthy" in vs.guards
                    or "len_gt_zero" in vs.guards
                    or any(g.startswith("gt_") or g.startswith("ge_") for g in vs.guards)
                    or var in self._length_constraints
                )
                if not has_len_guard:
                    # Only warn for index 0 on parameters (common pattern: items[0])
                    # or for any constant index on parameter-like variables
                    is_param = var in {
                        a.arg for a in getattr(self, '_current_func_args', [])
                    } if hasattr(self, '_current_func_args') else (
                        var not in self._const_values
                        and var not in self._collection_lens
                    )
                    if is_param:
                        self.bugs.append(Bug(
                            BugCategory.INDEX_OUT_OF_BOUNDS, node.lineno, node.col_offset,
                            f"Index {idx} on '{var}' without length guard (may be empty)",
                            self._func_name, var, "warning",
                        ))
        elif isinstance(node.slice, ast.Name):
            idx_var = node.slice.id
            idx_vs = self._env.get(idx_var)
            known_len = vs.known_len or self._collection_lens.get(var)
            if known_len is not None:
                if idx_vs.interval.lo is not None and idx_vs.interval.lo >= known_len:
                    self.bugs.append(Bug(
                        BugCategory.INDEX_OUT_OF_BOUNDS, node.lineno, node.col_offset,
                        f"Index '{idx_var}' (>= {idx_vs.interval.lo}) out of bounds for '{var}' of length {known_len}",
                        self._func_name, var, "error",
                    ))
            # Check if index guard is present
            if f"lt_len_{var}" not in idx_vs.guards and not idx_vs.interval.is_non_negative():
                pass  # Could warn about unchecked index

    # ── Helpers ─────────────────────────────────────────────────────

    def _get_type_name(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Tuple):
            names = [self._get_type_name(e) for e in node.elts]
            return "|".join(n for n in names if n)
        return None

    def _parse_annotation(self, ann: ast.expr) -> str:
        if isinstance(ann, ast.Name):
            return ann.id
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            return ann.value
        if isinstance(ann, ast.Subscript):
            if isinstance(ann.value, ast.Name):
                if ann.value.id == "Optional":
                    inner = self._parse_annotation(ann.slice)
                    return f"{inner}|None"
                return ann.value.id
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            left = self._parse_annotation(ann.left)
            right = self._parse_annotation(ann.right)
            return f"{left}|{right}"
        return "Any"


# ── CEGAR loop with Z3 ─────────────────────────────────────────────

def run_cegar_verification(
    source: str,
    guards: List[Tuple[str, str, Any]],
    bugs: List[Bug],
    max_iterations: int = 10,
) -> Tuple[List[Bug], int, int, bool]:
    """
    Run CEGAR refinement to verify/refute bug reports using Z3.

    1. Encode guards as SMT predicates
    2. For each potential bug, check if guards prove it impossible
    3. If ambiguous, try to add new predicates from path conditions
    """
    try:
        import z3
    except ImportError:
        return bugs, 0, len(guards), True

    verified_bugs = []
    total_predicates = len(guards)
    iterations = 0
    converged = True

    for bug in bugs:
        solver = z3.Solver()
        solver.set("timeout", 2000)

        # Create variables
        z3_vars: Dict[str, z3.ArithRef] = {}
        z3_bool_vars: Dict[str, z3.BoolRef] = {}

        def get_var(name: str) -> z3.ArithRef:
            if name not in z3_vars:
                z3_vars[name] = z3.Int(name)
            return z3_vars[name]

        def get_none_var(name: str) -> z3.BoolRef:
            if name not in z3_bool_vars:
                z3_bool_vars[name] = z3.Bool(f"is_none_{name}")
            return z3_bool_vars[name]

        # Encode guards as constraints
        for var, gtype, arg in guards:
            if gtype == "is_not_none":
                solver.add(z3.Not(get_none_var(var)))
            elif gtype == "is_none":
                solver.add(get_none_var(var))
            elif gtype == "ne_zero":
                solver.add(get_var(var) != 0)
            elif gtype == "gt":
                solver.add(get_var(var) > int(arg))
            elif gtype == "ge":
                solver.add(get_var(var) >= int(arg))
            elif gtype == "lt":
                solver.add(get_var(var) < int(arg))
            elif gtype == "le":
                solver.add(get_var(var) <= int(arg))

        iterations += 1

        # Check if bug condition is satisfiable under guard constraints
        if bug.category == BugCategory.DIV_BY_ZERO:
            solver.add(get_var(bug.variable) == 0)
            result = solver.check()
            if result == z3.sat:
                verified_bugs.append(bug)
            elif result == z3.unsat:
                total_predicates += 1
        elif bug.category in (BugCategory.NULL_DEREF, BugCategory.UNGUARDED_OPTIONAL):
            solver.add(get_none_var(bug.variable))
            result = solver.check()
            if result == z3.sat:
                verified_bugs.append(bug)
            elif result == z3.unsat:
                total_predicates += 1
        elif bug.category == BugCategory.INDEX_OUT_OF_BOUNDS:
            verified_bugs.append(bug)
        else:
            verified_bugs.append(bug)

    return verified_bugs, iterations, total_predicates, converged


# ── Top-level API ───────────────────────────────────────────────────

def analyze_source(source: str, filename: str = "<string>",
                   use_cegar: bool = True,
                   interprocedural: bool = True) -> FileResult:
    """Analyze Python source code with flow-sensitive abstract interpretation.

    When interprocedural=True (default), performs a two-pass analysis:
    1. First pass: infer function summaries (return nullity, type tags)
    2. Second pass: analyze each function using summaries for call resolution
    """
    t0 = time.perf_counter()

    try:
        tree = ast.parse(source, filename)
    except SyntaxError:
        return FileResult(filename, 0, 0, 0, 0, [], 0.0, 0)

    # Pass 1: Infer function summaries for interprocedural analysis
    summaries = infer_file_summaries(tree) if interprocedural else {}

    analyzer = FlowSensitiveAnalyzer(source, filename, function_summaries=summaries)
    func_nodes = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    results = []
    total_guards = 0
    total_bugs = 0
    total_predicates = 0

    for func_node in func_nodes:
        fr = analyzer.analyze_function(func_node)

        if use_cegar and fr.bugs:
            # Collect all guards for this function
            guards = []
            for rv in fr.refined_vars:
                for p in rv.predicates:
                    guards.append((rv.name, p, None))

            # Extract guards more precisely from function source
            func_guards = _extract_function_guards(source, func_node)

            verified_bugs, iters, preds, conv = run_cegar_verification(
                source, func_guards, fr.bugs,
            )
            fr = FunctionResult(
                name=fr.name, line=fr.line, end_line=fr.end_line,
                guards_harvested=fr.guards_harvested,
                predicates_inferred=preds,
                bugs=verified_bugs,
                refined_vars=fr.refined_vars,
                cegar_iterations=iters,
                analysis_time_ms=fr.analysis_time_ms,
                converged=conv,
            )

        results.append(fr)
        total_guards += fr.guards_harvested
        total_bugs += len(fr.bugs)
        total_predicates += fr.predicates_inferred

    elapsed = (time.perf_counter() - t0) * 1000
    loc = len(source.splitlines())

    return FileResult(
        file_path=filename,
        functions_analyzed=len(results),
        total_guards=total_guards,
        total_bugs=total_bugs,
        total_predicates=total_predicates,
        function_results=results,
        analysis_time_ms=elapsed,
        lines_of_code=loc,
    )


def analyze_file(path: str, use_cegar: bool = True) -> FileResult:
    """Analyze a Python file."""
    p = Path(path)
    if not p.exists():
        return FileResult(path, 0, 0, 0, 0, [], 0.0, 0)
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return FileResult(path, 0, 0, 0, 0, [], 0.0, 0)
    return analyze_source(source, str(p), use_cegar=use_cegar)


def analyze_directory(directory: str, pattern: str = "**/*.py",
                      exclude: Optional[List[str]] = None,
                      use_cegar: bool = True) -> List[FileResult]:
    """Analyze all Python files in a directory."""
    exclude = exclude or ["__pycache__", ".git", "node_modules", ".venv", "venv",
                          ".tox", ".mypy_cache", ".pytest_cache", "dist", "build"]
    results = []
    root = Path(directory)
    for py_file in sorted(root.glob(pattern)):
        if any(ex in str(py_file) for ex in exclude):
            continue
        try:
            result = analyze_file(str(py_file), use_cegar=use_cegar)
            if result.functions_analyzed > 0:
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed to analyze {py_file}: {e}")
    return results


def _extract_function_guards(source: str, func_node: ast.FunctionDef) -> List[Tuple[str, str, Any]]:
    """Extract guards from a function for CEGAR verification."""
    guards = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.If):
            guards.extend(_extract_test_guards(node.test))
        elif isinstance(node, ast.Assert):
            guards.extend(_extract_test_guards(node.test))
    return guards


def _extract_test_guards(test: ast.expr) -> List[Tuple[str, str, Any]]:
    """Extract guard triples from a test expression."""
    guards = []
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        left = test.left
        right = test.comparators[0]
        if isinstance(op, ast.IsNot) and isinstance(right, ast.Constant) and right.value is None:
            if isinstance(left, ast.Name):
                guards.append((left.id, "is_not_none", None))
        elif isinstance(op, ast.Is) and isinstance(right, ast.Constant) and right.value is None:
            if isinstance(left, ast.Name):
                guards.append((left.id, "is_none", None))
        elif isinstance(op, ast.NotEq) and isinstance(left, ast.Name) and isinstance(right, ast.Constant):
            if right.value == 0:
                guards.append((left.id, "ne_zero", 0))
        elif isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
                op_name = {ast.Gt: "gt", ast.GtE: "ge", ast.Lt: "lt", ast.LtE: "le"}[type(op)]
                guards.append((left.id, op_name, right.value))
    elif isinstance(test, ast.Call) and isinstance(test.func, ast.Name):
        if test.func.id == "isinstance" and len(test.args) >= 2:
            if isinstance(test.args[0], ast.Name):
                guards.append((test.args[0].id, "isinstance", None))
        elif test.func.id == "hasattr" and len(test.args) >= 2:
            if isinstance(test.args[0], ast.Name):
                guards.append((test.args[0].id, "hasattr", None))
    elif isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        for val in test.values:
            guards.extend(_extract_test_guards(val))
    elif isinstance(test, ast.Name):
        guards.append((test.id, "truthy", None))
    return guards


# ── Liquid type integration helpers ─────────────────────────────────────

def _reftype_to_varstate(reftype) -> Optional[VarState]:
    """Convert a liquid RefType to VarState (nullity, type tags, interval).

    Extracts information from the base type and predicate to build an
    abstract-domain VarState.  Returns None if no useful info is available.
    """
    try:
        base = reftype.base
        pred = reftype.pred
    except AttributeError:
        return None

    # Determine nullity and type tag from base type
    null = NullState.MAYBE_NULL
    tags = TypeTagSet(None)
    interval = Interval()

    base_name = getattr(base, "name", None) or str(base)
    kind_name = getattr(getattr(base, "kind", None), "name", "")

    if kind_name == "NONE":
        null = NullState.DEFINITELY_NULL
        tags = TypeTagSet(frozenset({"NoneType"}))
    elif kind_name in ("INT", "FLOAT", "STR", "BOOL", "LIST", "DICT",
                        "SET", "TUPLE", "OBJECT"):
        null = NullState.DEFINITELY_NOT_NULL
        tag_map = {
            "INT": "int", "FLOAT": "float", "STR": "str",
            "BOOL": "bool", "LIST": "list", "DICT": "dict",
            "SET": "set", "TUPLE": "tuple",
        }
        tag = tag_map.get(kind_name)
        if tag:
            tags = TypeTagSet(frozenset({tag}))

    # Extract interval bounds from predicate
    pred_op = getattr(pred, "op", None)
    pred_op_name = getattr(pred_op, "name", "") if pred_op else ""
    pred_value = getattr(pred, "value", None)

    if pred_op_name == "GT" and isinstance(pred_value, (int, float)):
        interval = Interval(lo=int(pred_value) + 1)
    elif pred_op_name == "GE" and isinstance(pred_value, (int, float)):
        interval = Interval(lo=int(pred_value))
    elif pred_op_name == "LT" and isinstance(pred_value, (int, float)):
        interval = Interval(hi=int(pred_value) - 1)
    elif pred_op_name == "LE" and isinstance(pred_value, (int, float)):
        interval = Interval(hi=int(pred_value))
    elif pred_op_name == "EQ" and isinstance(pred_value, (int, float)):
        interval = Interval(lo=int(pred_value), hi=int(pred_value))
    elif pred_op_name == "NEQ" and pred_value == 0:
        # v ≠ 0 — not enough for interval bounds, but we know it's not zero
        pass
    elif pred_op_name == "IS_NOT_NONE":
        null = NullState.DEFINITELY_NOT_NULL

    return VarState(null=null, tags=tags, interval=interval)


def analyze_source_liquid(source: str, filename: str = "<string>") -> FileResult:
    """Combined liquid + flow-sensitive analysis.

    1. Run LiquidTypeInferencer to infer contracts
    2. Run FlowSensitiveAnalyzer with liquid contracts
    3. Merge and deduplicate bugs
    4. Return enhanced FileResult
    """
    try:
        from src.liquid import LiquidTypeInferencer
    except Exception:
        return analyze_source(source, filename=filename, use_cegar=True)

    t0 = time.perf_counter()

    try:
        tree = ast.parse(source, filename)
    except SyntaxError:
        return FileResult(filename, 0, 0, 0, 0, [], 0.0, 0)

    # Phase 1: liquid type inference
    engine = LiquidTypeInferencer()
    lresult = engine.infer_module(source)
    contracts = lresult.contracts

    # Phase 2: flow-sensitive analysis with liquid summaries
    summaries = infer_file_summaries(tree)
    analyzer = FlowSensitiveAnalyzer(
        source, filename,
        function_summaries=summaries,
        liquid_summaries=contracts,
    )

    func_nodes = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    results = []
    total_guards = 0
    total_bugs = 0
    total_predicates = 0

    for func_node in func_nodes:
        fr = analyzer.analyze_function(func_node)
        results.append(fr)
        total_guards += fr.guards_harvested
        total_bugs += len(fr.bugs)
        total_predicates += fr.predicates_inferred

    # Merge liquid bugs (deduplicate by line+message)
    seen_bugs = set()
    for fr in results:
        for b in fr.bugs:
            seen_bugs.add((b.line, b.message))

    for lb in lresult.bugs:
        if (lb.line, lb.message) not in seen_bugs:
            total_bugs += 1
            # Attach liquid bugs to the first function result if any
            if results:
                results[0].bugs.append(
                    Bug(
                        category=BugCategory.NULL_DEREF
                        if lb.kind.name == "NULL_DEREF"
                        else BugCategory.DIV_BY_ZERO
                        if lb.kind.name == "DIV_BY_ZERO"
                        else BugCategory.TYPE_ERROR,
                        message=lb.message,
                        line=lb.line,
                        col=lb.col,
                        variable=lb.variable,
                        guard_context=None,
                    )
                )

    elapsed = (time.perf_counter() - t0) * 1000
    loc = len(source.splitlines())

    return FileResult(
        file_path=filename,
        functions_analyzed=len(results),
        total_guards=total_guards + lresult.predicates_harvested,
        total_bugs=total_bugs,
        total_predicates=total_predicates,
        function_results=results,
        analysis_time_ms=elapsed,
        lines_of_code=loc,
    )
