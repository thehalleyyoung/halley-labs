"""
Heap-sensitive abstract interpreter for Python using refinement types.

Performs flow-sensitive, heap-aware analysis by combining:
  - A refinement type environment (variable → PyRefinementType)
  - An abstract heap with recency abstraction
  - Alias tracking and mutation invalidation
  - Guard interpretation for type narrowing at branches

The main entry point is :class:`ModuleAnalyzer`, which produces
:class:`ModuleAnalysisResult` containing refined function summaries,
class types, and diagnostics.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.heap.heap_model import (
    HeapAddress,
    AbstractValue,
    AbstractHeap,
    HeapObject,
    HeapState,
    HeapTransformer,
    RecencyFlag,
    AddressValue,
    PrimitiveValue,
    TopValue,
    BottomValue,
    NoneValue,
)
from src.heap.alias_analysis import (
    AliasSet,
    PointsToGraph,
    FlowSensitivePointsTo,
    FieldPath,
)
from src.heap.class_model import (
    PythonClass,
    ClassRegistry,
    MROComputer,
    AttributeResolver,
    DescriptorKind,
)
from src.heap.mutation_tracking import (
    MutationTracker,
    MutationKind,
    FrameCondition,
    RefinementRef,
)
from src.refinement.python_refinements import (
    HeapPredicate,
    HeapPredKind,
    PyRefinementType,
    PyType,
    AnyType,
    NeverType,
    NoneType as NoneTypeP,
    IntPyType,
    FloatPyType,
    BoolPyType,
    StrPyType,
    BytesPyType,
    ClassType,
    ProtocolType,
    PyUnionType,
    PyIntersectionType,
    OptionalType,
    ListPyType,
    DictPyType,
    SetPyType,
    TuplePyType,
    FunctionPyType,
    TypeNarrower,
    RefinementSubtyping,
)
from src.refinement.guard_to_refinement import (
    PythonGuardInterpreter,
    GuardExtractor,
    RefinementPropagator,
)
from src.refinement.container_refinements import (
    ListRefinement,
    DictRefinement,
    SetRefinement,
    ContainerRefinementTracker,
)
from src.refinement.protocol_refinements import (
    FunctionRefinement,
    ProtocolRefinement,
    BuiltinProtocolRegistry,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  WarningKind + AnalysisWarning
# ═══════════════════════════════════════════════════════════════════════════


class WarningKind(Enum):
    """Categories of analysis warnings."""
    NPE = auto()
    TYPE_ERROR = auto()
    ATTRIBUTE_ERROR = auto()
    KEY_ERROR = auto()
    MUTATION_DURING_ITERATION = auto()
    PROTOCOL_VIOLATION = auto()
    RESOURCE_LEAK = auto()
    UNREACHABLE = auto()
    DEAD_CODE = auto()


@dataclass
class AnalysisWarning:
    """A single diagnostic produced during analysis."""
    kind: WarningKind
    message: str
    line: int
    column: int
    severity: str = "warning"          # 'error' | 'warning' | 'info'
    variable: Optional[str] = None

    def __repr__(self) -> str:
        tag = self.kind.name
        return f"[{self.severity}:{tag}] {self.message} (L{self.line}:{self.column})"


# ═══════════════════════════════════════════════════════════════════════════
# 2.  AnalysisState — combined abstract state at a program point
# ═══════════════════════════════════════════════════════════════════════════

_BOTTOM_SENTINEL = "__bottom__"


@dataclass
class AnalysisState:
    """Combined analysis state at a single program point."""
    heap: AbstractHeap = field(default_factory=AbstractHeap)
    var_env: Dict[str, PyRefinementType] = field(default_factory=dict)
    var_addrs: Dict[str, AbstractValue] = field(default_factory=dict)
    alias_set: AliasSet = field(default_factory=AliasSet)
    active_predicates: Set[HeapPredicate] = field(default_factory=set)
    mutation_tracker: MutationTracker = field(default=None)  # type: ignore[assignment]
    container_tracker: ContainerRefinementTracker = field(
        default_factory=ContainerRefinementTracker
    )
    _is_bottom: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.mutation_tracker is None:
            self.mutation_tracker = MutationTracker(self.alias_set)

    # -- queries -----------------------------------------------------------

    def lookup(self, var: str) -> PyRefinementType:
        """Return the refined type for *var*, or ``{Any}`` if unknown."""
        if var in self.var_env:
            return self.var_env[var]
        return PyRefinementType.simple(AnyType())

    def is_bottom(self) -> bool:
        return self._is_bottom

    # -- transformers (return new state) -----------------------------------

    def bind(
        self,
        var: str,
        typ: PyRefinementType,
        addr_val: Optional[AbstractValue] = None,
    ) -> AnalysisState:
        """Bind *var* to *typ* (and optionally an abstract value)."""
        new_env = dict(self.var_env)
        new_env[var] = typ
        new_addrs = dict(self.var_addrs)
        if addr_val is not None:
            new_addrs[var] = addr_val
        else:
            new_addrs[var] = PrimitiveValue(kind=None, constraint=None)
        new_alias = self.alias_set
        if addr_val is not None:
            addrs = addr_val.as_addresses()
            if addrs:
                new_alias = AliasSet(var_to_pts=dict(self.alias_set.var_to_pts))
                new_alias.var_to_pts[var] = set(addrs)
        return AnalysisState(
            heap=self.heap,
            var_env=new_env,
            var_addrs=new_addrs,
            alias_set=new_alias,
            active_predicates=set(self.active_predicates),
            mutation_tracker=self.mutation_tracker,
            container_tracker=self.container_tracker,
            _is_bottom=self._is_bottom,
        )

    def narrow(self, var: str, pred: HeapPredicate) -> AnalysisState:
        """Narrow *var*'s type with *pred*."""
        new_env = dict(self.var_env)
        current = self.lookup(var)
        new_env[var] = current.narrow(pred)
        new_preds = set(self.active_predicates)
        new_preds.add(pred)
        return AnalysisState(
            heap=self.heap,
            var_env=new_env,
            var_addrs=dict(self.var_addrs),
            alias_set=self.alias_set,
            active_predicates=new_preds,
            mutation_tracker=self.mutation_tracker,
            container_tracker=self.container_tracker,
            _is_bottom=self._is_bottom,
        )

    def invalidate(self, refs: Set[RefinementRef]) -> AnalysisState:
        """Remove invalidated refinements from the environment."""
        if not refs:
            return self
        new_env = dict(self.var_env)
        new_preds = set(self.active_predicates)
        for ref in refs:
            var = ref.variable
            if var in new_env:
                rt = new_env[var]
                kept = tuple(
                    p for p in rt.predicates
                    if not _pred_matches_ref(p, ref)
                )
                new_env[var] = PyRefinementType(rt.base, kept)
            new_preds = {
                p for p in new_preds
                if not (p.variable == var and _path_matches(p.path, ref.field_path))
            }
        return AnalysisState(
            heap=self.heap,
            var_env=new_env,
            var_addrs=dict(self.var_addrs),
            alias_set=self.alias_set,
            active_predicates=new_preds,
            mutation_tracker=self.mutation_tracker,
            container_tracker=self.container_tracker,
            _is_bottom=self._is_bottom,
        )

    def join(self, other: AnalysisState) -> AnalysisState:
        """Lattice join (upper bound)."""
        if self._is_bottom:
            return other.deep_copy()
        if other._is_bottom:
            return self.deep_copy()

        joined_heap = self.heap.join(other.heap)
        all_vars = set(self.var_env) | set(other.var_env)
        joined_env: Dict[str, PyRefinementType] = {}
        for v in all_vars:
            t1 = self.var_env.get(v)
            t2 = other.var_env.get(v)
            if t1 is not None and t2 is not None:
                joined_env[v] = t1.join(t2)
            elif t1 is not None:
                joined_env[v] = t1
            else:
                assert t2 is not None
                joined_env[v] = t2

        joined_addrs: Dict[str, AbstractValue] = {}
        for v in set(self.var_addrs) | set(other.var_addrs):
            a1 = self.var_addrs.get(v)
            a2 = other.var_addrs.get(v)
            if a1 is not None and a2 is not None:
                joined_addrs[v] = a1.join(a2)
            elif a1 is not None:
                joined_addrs[v] = a1
            else:
                assert a2 is not None
                joined_addrs[v] = a2

        joined_alias = self.alias_set.join(other.alias_set)
        joined_preds = self.active_predicates & other.active_predicates
        joined_container = self.container_tracker.join(other.container_tracker)

        return AnalysisState(
            heap=joined_heap,
            var_env=joined_env,
            var_addrs=joined_addrs,
            alias_set=joined_alias,
            active_predicates=joined_preds,
            mutation_tracker=MutationTracker(joined_alias),
            container_tracker=joined_container,
        )

    def widen(self, other: AnalysisState) -> AnalysisState:
        """Widening: join with predicate dropping for convergence."""
        if self._is_bottom:
            return other.deep_copy()
        if other._is_bottom:
            return self.deep_copy()

        widened_heap = self.heap.widen(other.heap)
        all_vars = set(self.var_env) | set(other.var_env)
        widened_env: Dict[str, PyRefinementType] = {}
        for v in all_vars:
            t1 = self.var_env.get(v)
            t2 = other.var_env.get(v)
            if t1 is not None and t2 is not None:
                widened_env[v] = RefinementSubtyping.widen_type(t1, t2)
            elif t1 is not None:
                widened_env[v] = t1.widen()
            else:
                assert t2 is not None
                widened_env[v] = t2.widen()

        widened_addrs: Dict[str, AbstractValue] = {}
        for v in set(self.var_addrs) | set(other.var_addrs):
            a1 = self.var_addrs.get(v)
            a2 = other.var_addrs.get(v)
            if a1 is not None and a2 is not None:
                widened_addrs[v] = a1.widen(a2)
            elif a1 is not None:
                widened_addrs[v] = a1
            else:
                assert a2 is not None
                widened_addrs[v] = a2

        widened_alias = self.alias_set.widen(other.alias_set)
        widened_preds = self.active_predicates & other.active_predicates
        widened_container = self.container_tracker.join(other.container_tracker)

        return AnalysisState(
            heap=widened_heap,
            var_env=widened_env,
            var_addrs=widened_addrs,
            alias_set=widened_alias,
            active_predicates=widened_preds,
            mutation_tracker=MutationTracker(widened_alias),
            container_tracker=widened_container,
        )

    def deep_copy(self) -> AnalysisState:
        """Return a deep copy of this state."""
        return AnalysisState(
            heap=self.heap.deep_copy(),
            var_env={k: v for k, v in self.var_env.items()},
            var_addrs={k: v for k, v in self.var_addrs.items()},
            alias_set=AliasSet(
                var_to_pts={k: set(v) for k, v in self.alias_set.var_to_pts.items()}
            ),
            active_predicates=set(self.active_predicates),
            mutation_tracker=MutationTracker(self.alias_set),
            container_tracker=self.container_tracker,
            _is_bottom=self._is_bottom,
        )

    def with_heap(self, new_heap: AbstractHeap) -> AnalysisState:
        """Return a copy with a replaced heap."""
        return AnalysisState(
            heap=new_heap,
            var_env=dict(self.var_env),
            var_addrs=dict(self.var_addrs),
            alias_set=self.alias_set,
            active_predicates=set(self.active_predicates),
            mutation_tracker=self.mutation_tracker,
            container_tracker=self.container_tracker,
            _is_bottom=self._is_bottom,
        )

    @staticmethod
    def bottom() -> AnalysisState:
        """The unreachable state."""
        return AnalysisState(_is_bottom=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _pred_matches_ref(pred: HeapPredicate, ref: RefinementRef) -> bool:
    """Check if *pred* is tracked by *ref*."""
    if pred.variable != ref.variable:
        return False
    return _path_matches(pred.path, ref.field_path)


def _path_matches(p1: Tuple[str, ...], p2: Tuple[str, ...]) -> bool:
    """True if p1 is a prefix of p2, p2 is a prefix of p1, or they are equal."""
    shorter = min(len(p1), len(p2))
    return p1[:shorter] == p2[:shorter]


def _annotation_to_type(ann: Optional[ast.expr]) -> PyType:
    """Convert a simple annotation AST node to a PyType (best-effort)."""
    if ann is None:
        return AnyType()
    if isinstance(ann, ast.Constant):
        if ann.value is None:
            return NoneTypeP()
        return AnyType()
    if isinstance(ann, ast.Name):
        mapping: Dict[str, PyType] = {
            "int": IntPyType(),
            "float": FloatPyType(),
            "bool": BoolPyType(),
            "str": StrPyType(),
            "bytes": BytesPyType(),
            "None": NoneTypeP(),
            "object": AnyType(),
        }
        if ann.id in mapping:
            return mapping[ann.id]
        return ClassType(name=ann.id, address=HeapAddress(site=ann.id, context=()))
    if isinstance(ann, ast.Subscript):
        base = ann.value
        if isinstance(base, ast.Name):
            if base.id == "Optional":
                inner = _annotation_to_type(ann.slice)
                return OptionalType(inner=inner)
            if base.id == "List" or base.id == "list":
                elem = _annotation_to_type(ann.slice)
                return ListPyType(element=elem)
            if base.id in ("Dict", "dict"):
                if isinstance(ann.slice, ast.Tuple) and len(ann.slice.elts) == 2:
                    kt = _annotation_to_type(ann.slice.elts[0])
                    vt = _annotation_to_type(ann.slice.elts[1])
                    return DictPyType(key=kt, value=vt)
                return DictPyType(key=AnyType(), value=AnyType())
            if base.id in ("Set", "set"):
                elem = _annotation_to_type(ann.slice)
                return SetPyType(element=elem)
            if base.id in ("Tuple", "tuple"):
                if isinstance(ann.slice, ast.Tuple):
                    elems = tuple(
                        _annotation_to_type(e) for e in ann.slice.elts
                    )
                    return TuplePyType(elements=elems)
                e = _annotation_to_type(ann.slice)
                return TuplePyType(elements=(e,))
            if base.id == "Union":
                if isinstance(ann.slice, ast.Tuple):
                    members = frozenset(
                        _annotation_to_type(e) for e in ann.slice.elts
                    )
                    return PyUnionType(members=members)
                return _annotation_to_type(ann.slice)
        return AnyType()
    if isinstance(ann, ast.Attribute):
        return AnyType()
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        left = _annotation_to_type(ann.left)
        right = _annotation_to_type(ann.right)
        return PyUnionType(members=frozenset({left, right}))
    return AnyType()


def _constant_to_type(value: Any) -> PyType:
    """Map a Python constant to a PyType."""
    if value is None:
        return NoneTypeP()
    if isinstance(value, bool):
        return BoolPyType()
    if isinstance(value, int):
        return IntPyType()
    if isinstance(value, float):
        return FloatPyType()
    if isinstance(value, str):
        return StrPyType()
    if isinstance(value, bytes):
        return BytesPyType()
    return AnyType()


def _binop_result_type(left: PyType, right: PyType, op: ast.operator) -> PyType:
    """Compute the result type of a binary operation."""
    if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)):
        if isinstance(left, IntPyType) and isinstance(right, IntPyType):
            return IntPyType()
        if isinstance(left, (IntPyType, FloatPyType)) and isinstance(
            right, (IntPyType, FloatPyType)
        ):
            return FloatPyType()
        if isinstance(op, ast.Add):
            if isinstance(left, StrPyType) and isinstance(right, StrPyType):
                return StrPyType()
            if isinstance(left, ListPyType) and isinstance(right, ListPyType):
                elem = left.element.join(right.element)
                return ListPyType(element=elem)
            if isinstance(left, TuplePyType) and isinstance(right, TuplePyType):
                return TuplePyType(elements=left.elements + right.elements)
        if isinstance(op, ast.Mult):
            if isinstance(left, StrPyType) and isinstance(right, IntPyType):
                return StrPyType()
            if isinstance(left, IntPyType) and isinstance(right, StrPyType):
                return StrPyType()
            if isinstance(left, ListPyType) and isinstance(right, IntPyType):
                return ListPyType(element=left.element)
        return AnyType()
    if isinstance(op, ast.Div):
        if isinstance(left, (IntPyType, FloatPyType)) and isinstance(
            right, (IntPyType, FloatPyType)
        ):
            return FloatPyType()
        return AnyType()
    if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
        if isinstance(left, IntPyType) and isinstance(right, IntPyType):
            return IntPyType()
        if isinstance(left, BoolPyType) and isinstance(right, BoolPyType):
            return BoolPyType()
        if isinstance(left, SetPyType) and isinstance(right, SetPyType):
            elem = left.element.join(right.element)
            return SetPyType(element=elem)
        return AnyType()
    if isinstance(op, ast.MatMult):
        return AnyType()
    return AnyType()


def _cmp_op_str(op: ast.cmpop) -> str:
    """Convert an AST comparison operator to its string representation."""
    table: Dict[type, str] = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.In: "in",
        ast.NotIn: "not in",
    }
    return table.get(type(op), "==")


def _extract_name(node: ast.expr) -> Optional[str]:
    """Extract a simple name from a Name node."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _make_heap_address(node: ast.AST) -> HeapAddress:
    """Create a heap address from an AST node's source location."""
    line = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0)
    return HeapAddress(site=f"alloc_{line}_{col}", context=())


# ═══════════════════════════════════════════════════════════════════════════
# 3.  HeapSensitiveAnalyzer — main abstract interpreter
# ═══════════════════════════════════════════════════════════════════════════


class HeapSensitiveAnalyzer:
    """Flow-sensitive, heap-aware abstract interpreter for Python.

    Walks the AST of each function body, maintaining an :class:`AnalysisState`
    that tracks the abstract heap, variable types, alias information, and
    active refinements.  At control-flow joins the states are merged via
    lattice join; loops use widening after *widening_threshold* iterations.
    """

    def __init__(
        self,
        class_registry: Optional[ClassRegistry] = None,
        widening_threshold: int = 3,
    ) -> None:
        self.class_registry: ClassRegistry = (
            class_registry if class_registry is not None else ClassRegistry()
        )
        self.mro_computer: MROComputer = MROComputer()
        self.protocol_registry: BuiltinProtocolRegistry = BuiltinProtocolRegistry()
        self.guard_interpreter: PythonGuardInterpreter = PythonGuardInterpreter(
            registry=self.class_registry,
            heap=AbstractHeap(),
        )
        self.guard_extractor: GuardExtractor = GuardExtractor()
        self.propagator: RefinementPropagator = RefinementPropagator()
        self.narrower: TypeNarrower = TypeNarrower()
        self.subtyping: RefinementSubtyping = RefinementSubtyping()
        self.warnings: List[AnalysisWarning] = []
        self.function_summaries: Dict[str, FunctionRefinement] = {}
        self.widening_threshold: int = widening_threshold
        self._return_types: List[PyRefinementType] = []
        self._current_function: Optional[str] = None
        self._loop_iteration_vars: Set[str] = set()

    # ── module-level ──────────────────────────────────────────────────

    def analyze_module(self, module: ast.Module) -> Dict[str, FunctionRefinement]:
        """Analyze all top-level function/class definitions in *module*."""
        state = AnalysisState()
        for node in module.body:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                summary = self.analyze_function(node)
                self.function_summaries[node.name] = summary
                func_type = self._function_def_to_type(node)
                state = state.bind(
                    node.name,
                    PyRefinementType.simple(func_type),
                )
            elif isinstance(node, ast.ClassDef):
                state = self._analyze_class_def(node, state)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                state = self.analyze_stmt(node, state)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                state = self._analyze_import(node, state)
            else:
                state = self.analyze_stmt(node, state)
        return dict(self.function_summaries)

    # ── function-level ────────────────────────────────────────────────

    def analyze_function(
        self, func: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> FunctionRefinement:
        """Analyze a single function definition and build a summary."""
        self._return_types = []
        prev_func = self._current_function
        self._current_function = func.name

        state = AnalysisState()
        self.guard_interpreter = PythonGuardInterpreter(
            registry=self.class_registry,
            heap=state.heap,
        )

        params: List[Tuple[str, PyRefinementType]] = []
        args = func.args
        # positional args
        for i, arg in enumerate(args.args):
            ann_type = _annotation_to_type(arg.annotation)
            ref_type = PyRefinementType.simple(ann_type)
            params.append((arg.arg, ref_type))
            addr = _make_heap_address(arg)
            addr_val: AbstractValue = PrimitiveValue(kind=None, constraint=None)
            if isinstance(ann_type, ClassType):
                obj = HeapObject(
                    cls_addr=addr,
                    fields={},
                    recency=RecencyFlag.RECENT,
                )
                state.heap.objects[addr] = obj
                addr_val = AddressValue(addresses=frozenset({addr}))
            state = state.bind(arg.arg, ref_type, addr_val)

        # *args
        varargs_type: Optional[PyRefinementType] = None
        if args.vararg:
            ann = _annotation_to_type(args.vararg.annotation)
            varargs_rt = PyRefinementType.simple(TuplePyType(elements=(ann,)))
            varargs_type = varargs_rt
            state = state.bind(args.vararg.arg, varargs_rt)

        # **kwargs
        kwargs_type: Optional[PyRefinementType] = None
        if args.kwarg:
            ann = _annotation_to_type(args.kwarg.annotation)
            kwargs_rt = PyRefinementType.simple(DictPyType(key=StrPyType(), value=ann))
            kwargs_type = kwargs_rt
            state = state.bind(args.kwarg.arg, kwargs_rt)

        # keyword-only args
        for arg in args.kwonlyargs:
            ann_type = _annotation_to_type(arg.annotation)
            ref_type = PyRefinementType.simple(ann_type)
            params.append((arg.arg, ref_type))
            state = state.bind(arg.arg, ref_type)

        # defaults with non-None knowledge
        for i, default in enumerate(args.defaults):
            arg_index = len(args.args) - len(args.defaults) + i
            if arg_index >= 0 and arg_index < len(args.args):
                arg_name = args.args[arg_index].arg
                if isinstance(default, ast.Constant) and default.value is not None:
                    pass  # keep existing binding

        # analyze body
        for stmt in func.body:
            if state.is_bottom():
                self._warn(WarningKind.DEAD_CODE, "Unreachable code", stmt)
                break
            state = self.analyze_stmt(stmt, state)

        # build return type
        if self._return_types:
            ret = self._return_types[0]
            for rt in self._return_types[1:]:
                ret = ret.join(rt)
        else:
            ret = PyRefinementType.simple(NoneTypeP())

        # check declared return annotation
        if func.returns is not None:
            declared = _annotation_to_type(func.returns)
            declared_rt = PyRefinementType.simple(declared)
            if not ret.is_subtype_of(declared_rt):
                if not isinstance(declared, AnyType):
                    self._warn(
                        WarningKind.TYPE_ERROR,
                        f"Return type {ret.pretty()} is not compatible "
                        f"with declared {declared_rt.pretty()}",
                        func,
                    )

        is_async = isinstance(func, ast.AsyncFunctionDef)
        is_gen = any(
            isinstance(n, (ast.Yield, ast.YieldFrom))
            for n in ast.walk(func)
        )

        summary = FunctionRefinement(
            params=params,
            varargs=varargs_type,
            kwargs=kwargs_type,
            return_type=ret,
            raises=frozenset(),
            pre_conditions=[],
            post_conditions=[],
            frame=frozenset(),
            is_pure=False,
            is_generator=is_gen,
            is_async=is_async,
        )

        self._current_function = prev_func
        return summary

    # ── statement dispatch ────────────────────────────────────────────

    def analyze_stmt(self, stmt: ast.stmt, state: AnalysisState) -> AnalysisState:
        """Dispatch to the appropriate statement analyzer."""
        if state.is_bottom():
            return state

        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            return self._analyze_assign_stmt(stmt, state)
        if isinstance(stmt, ast.AugAssign):
            return self._analyze_aug_assign(stmt, state)
        if isinstance(stmt, ast.If):
            return self.analyze_if(stmt, state)
        if isinstance(stmt, ast.While):
            return self.analyze_while(stmt, state)
        if isinstance(stmt, ast.For):
            return self.analyze_for(stmt, state)
        if isinstance(stmt, ast.With):
            return self.analyze_with(stmt, state)
        if isinstance(stmt, ast.Try):
            return self.analyze_try_except(stmt, state)
        if isinstance(stmt, ast.Return):
            return self.analyze_return(stmt, state)
        if isinstance(stmt, ast.Expr):
            return self.analyze_expr_stmt(stmt, state)
        if isinstance(stmt, ast.Assert):
            return self._analyze_assert(stmt, state)
        if isinstance(stmt, ast.Delete):
            return self._analyze_delete(stmt, state)
        if isinstance(stmt, ast.Raise):
            return self._analyze_raise(stmt, state)
        if isinstance(stmt, ast.Pass) or isinstance(stmt, ast.Break):
            return state
        if isinstance(stmt, ast.Continue):
            return state
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            summary = self.analyze_function(stmt)
            self.function_summaries[stmt.name] = summary
            ftype = self._function_def_to_type(stmt)
            return state.bind(stmt.name, PyRefinementType.simple(ftype))
        if isinstance(stmt, ast.ClassDef):
            return self._analyze_class_def(stmt, state)
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return self._analyze_import(stmt, state)
        if isinstance(stmt, ast.Global) or isinstance(stmt, ast.Nonlocal):
            return state
        return state

    # ── assignment ────────────────────────────────────────────────────

    def _analyze_assign_stmt(
        self, stmt: Union[ast.Assign, ast.AnnAssign], state: AnalysisState
    ) -> AnalysisState:
        """Handle Assign and AnnAssign statements."""
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                ann_type = _annotation_to_type(stmt.annotation)
                if stmt.target and isinstance(stmt.target, ast.Name):
                    return state.bind(
                        stmt.target.id,
                        PyRefinementType.simple(ann_type),
                    )
                return state
            val_type, val_addr, state = self.eval_expr(stmt.value, state)
            ann_type = _annotation_to_type(stmt.annotation)
            ann_rt = PyRefinementType.simple(ann_type)
            if not val_type.is_subtype_of(ann_rt) and not isinstance(ann_type, AnyType):
                self._warn(
                    WarningKind.TYPE_ERROR,
                    f"Assigned value type {val_type.pretty()} incompatible "
                    f"with annotation {ann_rt.pretty()}",
                    stmt,
                )
            target_type = val_type.meet(ann_rt) if not isinstance(ann_type, AnyType) else val_type
            if isinstance(stmt.target, ast.Name):
                return state.bind(stmt.target.id, target_type, val_addr)
            return self.analyze_assignment(stmt.target, target_type, val_addr, state)

        # ast.Assign — may have multiple targets
        val_type, val_addr, state = self.eval_expr(stmt.value, state)
        for target in stmt.targets:
            state = self.analyze_assignment(target, val_type, val_addr, state)
        return state

    def analyze_assignment(
        self,
        target: ast.expr,
        val_type: PyRefinementType,
        val_addr: AbstractValue,
        state: AnalysisState,
    ) -> AnalysisState:
        """Handle a single assignment target."""
        if isinstance(target, ast.Name):
            return state.bind(target.id, val_type, val_addr)

        if isinstance(target, ast.Attribute):
            # x.attr = value
            obj_type, obj_addr, state = self.eval_expr(target.value, state)
            self._check_none_safety(obj_type, f".{target.attr} =", target)
            addrs = obj_addr.as_addresses()
            for addr in addrs:
                state.heap.write_attr(addr, target.attr, val_addr)
            # invalidate refinements that mention this field
            obj_name = _extract_name(target.value)
            if obj_name:
                invalidated = state.mutation_tracker.on_setattr(
                    target_addr=list(addrs)[0] if addrs else HeapAddress("unknown", ()),
                    field_name=target.attr,
                    new_value=val_addr,
                    program_point=getattr(target, "lineno", 0),
                )
                state = state.invalidate(invalidated)
            return state

        if isinstance(target, ast.Subscript):
            # x[k] = value
            obj_type, obj_addr, state = self.eval_expr(target.value, state)
            self._check_none_safety(obj_type, "[...] =", target)
            idx_type, idx_addr, state = self.eval_expr(target.slice, state)
            addrs = obj_addr.as_addresses()
            obj_name = _extract_name(target.value)
            if obj_name:
                for addr in addrs:
                    invalidated = state.mutation_tracker.on_store_subscript(
                        target_addr=addr,
                        key_value=idx_addr,
                        new_value=val_addr,
                        program_point=getattr(target, "lineno", 0),
                    )
                    state = state.invalidate(invalidated)
                # update container tracker
                base = obj_type.base
                if isinstance(base, ListPyType):
                    state = AnalysisState(
                        heap=state.heap,
                        var_env=state.var_env,
                        var_addrs=state.var_addrs,
                        alias_set=state.alias_set,
                        active_predicates=state.active_predicates,
                        mutation_tracker=state.mutation_tracker,
                        container_tracker=state.container_tracker.track_list_op(
                            obj_name, "__setitem__", [idx_type, val_type]
                        ),
                    )
                elif isinstance(base, DictPyType):
                    state = AnalysisState(
                        heap=state.heap,
                        var_env=state.var_env,
                        var_addrs=state.var_addrs,
                        alias_set=state.alias_set,
                        active_predicates=state.active_predicates,
                        mutation_tracker=state.mutation_tracker,
                        container_tracker=state.container_tracker.track_dict_op(
                            obj_name, "__setitem__", [idx_type, val_type]
                        ),
                    )
            return state

        if isinstance(target, (ast.Tuple, ast.List)):
            # unpacking: a, b = expr  or  [a, b] = expr
            base = val_type.base
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Starred):
                    inner_name = _extract_name(elt.value)
                    if inner_name:
                        if isinstance(base, TuplePyType):
                            remaining = base.elements[i:]
                            star_elem = AnyType()
                            for e in remaining:
                                star_elem = star_elem.join(e)
                            star_type = PyRefinementType.simple(
                                ListPyType(element=star_elem)
                            )
                        elif isinstance(base, ListPyType):
                            star_type = val_type
                        else:
                            star_type = PyRefinementType.simple(
                                ListPyType(element=AnyType())
                            )
                        state = state.bind(inner_name, star_type)
                else:
                    elem_name = _extract_name(elt)
                    if isinstance(base, TuplePyType) and i < len(base.elements):
                        elem_type = PyRefinementType.simple(base.elements[i])
                    elif isinstance(base, ListPyType):
                        elem_type = PyRefinementType.simple(base.element)
                    else:
                        elem_type = PyRefinementType.simple(AnyType())
                    if elem_name:
                        state = state.bind(elem_name, elem_type)
                    else:
                        state = self.analyze_assignment(elt, elem_type, val_addr, state)
            return state

        return state

    def _analyze_aug_assign(
        self, stmt: ast.AugAssign, state: AnalysisState
    ) -> AnalysisState:
        """Handle augmented assignment: x += 1, etc."""
        lhs_type, lhs_addr, state = self.eval_expr(stmt.target, state)
        rhs_type, rhs_addr, state = self.eval_expr(stmt.value, state)
        result_base = _binop_result_type(lhs_type.base, rhs_type.base, stmt.op)
        result_type = PyRefinementType.simple(result_base)

        if isinstance(stmt.target, ast.Name):
            return state.bind(stmt.target.id, result_type)
        if isinstance(stmt.target, ast.Attribute):
            obj_type, obj_addr, state = self.eval_expr(stmt.target.value, state)
            addrs = obj_addr.as_addresses()
            for addr in addrs:
                state.heap.write_attr(addr, stmt.target.attr, rhs_addr)
            return state
        if isinstance(stmt.target, ast.Subscript):
            return state
        return state

    # ── if ────────────────────────────────────────────────────────────

    def analyze_if(self, node: ast.If, state: AnalysisState) -> AnalysisState:
        """Analyze an if/elif/else block with guard-based narrowing."""
        true_pred, false_pred = self.guard_interpreter.interpret(node.test)

        true_env = self.propagator.propagate_true_branch(true_pred, state.var_env)
        true_state = AnalysisState(
            heap=state.heap.deep_copy(),
            var_env=true_env,
            var_addrs=dict(state.var_addrs),
            alias_set=state.alias_set,
            active_predicates=state.active_predicates | {true_pred},
            mutation_tracker=state.mutation_tracker,
            container_tracker=state.container_tracker,
        )

        false_env = self.propagator.propagate_false_branch(true_pred, state.var_env)
        false_state = AnalysisState(
            heap=state.heap.deep_copy(),
            var_env=false_env,
            var_addrs=dict(state.var_addrs),
            alias_set=state.alias_set,
            active_predicates=state.active_predicates | {false_pred},
            mutation_tracker=state.mutation_tracker,
            container_tracker=state.container_tracker,
        )

        # check if either branch is unreachable
        if true_pred.kind == HeapPredKind.FALSE:
            self._warn(WarningKind.UNREACHABLE, "True branch is unreachable", node)
        if false_pred.kind == HeapPredKind.FALSE:
            if node.orelse:
                self._warn(WarningKind.UNREACHABLE, "Else branch is unreachable", node)

        # analyze true body
        for stmt in node.body:
            if true_state.is_bottom():
                self._warn(WarningKind.DEAD_CODE, "Dead code in true branch", stmt)
                break
            true_state = self.analyze_stmt(stmt, true_state)

        # analyze else / elif body
        if node.orelse:
            for stmt in node.orelse:
                if false_state.is_bottom():
                    self._warn(WarningKind.DEAD_CODE, "Dead code in else branch", stmt)
                    break
                false_state = self.analyze_stmt(stmt, false_state)

        # if one branch is bottom (e.g. always returns), use the other
        if true_state.is_bottom() and false_state.is_bottom():
            return AnalysisState.bottom()
        if true_state.is_bottom():
            return false_state
        if false_state.is_bottom():
            return true_state

        return true_state.join(false_state)

    # ── while ─────────────────────────────────────────────────────────

    def analyze_while(self, node: ast.While, state: AnalysisState) -> AnalysisState:
        """Analyze a while loop with fixed-point iteration and widening."""
        iteration = 0
        prev_state = state

        while True:
            true_pred, false_pred = self.guard_interpreter.interpret(node.test)
            loop_env = self.propagator.propagate_true_branch(true_pred, prev_state.var_env)
            loop_state = AnalysisState(
                heap=prev_state.heap.deep_copy(),
                var_env=loop_env,
                var_addrs=dict(prev_state.var_addrs),
                alias_set=prev_state.alias_set,
                active_predicates=prev_state.active_predicates | {true_pred},
                mutation_tracker=prev_state.mutation_tracker,
                container_tracker=prev_state.container_tracker,
            )

            for stmt in node.body:
                if loop_state.is_bottom():
                    break
                loop_state = self.analyze_stmt(stmt, loop_state)

            # merge back-edge state with entry state
            if iteration >= self.widening_threshold:
                merged = prev_state.widen(loop_state)
            else:
                merged = prev_state.join(loop_state)

            # check convergence: compare environments
            converged = True
            for v in set(merged.var_env) | set(prev_state.var_env):
                m_type = merged.var_env.get(v)
                p_type = prev_state.var_env.get(v)
                if m_type is None or p_type is None:
                    converged = False
                    break
                if m_type.pretty() != p_type.pretty():
                    converged = False
                    break

            if converged or iteration > self.widening_threshold + 2:
                break

            prev_state = merged
            iteration += 1

        # after loop: apply false predicate (guard failed)
        exit_env = self.propagator.propagate_false_branch(true_pred, merged.var_env)
        exit_state = AnalysisState(
            heap=merged.heap,
            var_env=exit_env,
            var_addrs=merged.var_addrs,
            alias_set=merged.alias_set,
            active_predicates=merged.active_predicates | {false_pred},
            mutation_tracker=merged.mutation_tracker,
            container_tracker=merged.container_tracker,
        )

        # else clause
        if node.orelse:
            for stmt in node.orelse:
                exit_state = self.analyze_stmt(stmt, exit_state)

        return exit_state

    # ── for ───────────────────────────────────────────────────────────

    def analyze_for(self, node: ast.For, state: AnalysisState) -> AnalysisState:
        """Analyze a for loop: extract element type, detect mutation."""
        iter_type, iter_addr, state = self.eval_expr(node.iter, state)
        iter_base = iter_type.base

        # determine element type from iterable
        if isinstance(iter_base, ListPyType):
            elem_type = PyRefinementType.simple(iter_base.element)
        elif isinstance(iter_base, SetPyType):
            elem_type = PyRefinementType.simple(iter_base.element)
        elif isinstance(iter_base, DictPyType):
            elem_type = PyRefinementType.simple(iter_base.key)
        elif isinstance(iter_base, TuplePyType):
            if iter_base.elements:
                combined = iter_base.elements[0]
                for e in iter_base.elements[1:]:
                    combined = combined.join(e)
                elem_type = PyRefinementType.simple(combined)
            else:
                elem_type = PyRefinementType.simple(AnyType())
        elif isinstance(iter_base, StrPyType):
            elem_type = PyRefinementType.simple(StrPyType())
        elif isinstance(iter_base, BytesPyType):
            elem_type = PyRefinementType.simple(IntPyType())
        else:
            elem_type = PyRefinementType.simple(AnyType())

        # record the iterable variable for mutation-during-iteration detection
        iter_var = _extract_name(node.iter)
        if iter_var:
            self._loop_iteration_vars.add(iter_var)

        # bind loop variable
        target_name = _extract_name(node.target)
        loop_state = state
        if target_name:
            loop_state = loop_state.bind(target_name, elem_type)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            loop_state = self.analyze_assignment(
                node.target, elem_type, PrimitiveValue(kind=None, constraint=None), loop_state
            )

        # iterate body with widening
        prev_state = loop_state
        for iteration in range(self.widening_threshold + 2):
            body_state = prev_state
            if target_name:
                body_state = body_state.bind(target_name, elem_type)

            for stmt in node.body:
                if body_state.is_bottom():
                    break
                body_state = self.analyze_stmt(stmt, body_state)

            if iteration >= self.widening_threshold:
                merged = prev_state.widen(body_state)
            else:
                merged = prev_state.join(body_state)

            converged = True
            for v in set(merged.var_env) | set(prev_state.var_env):
                m = merged.var_env.get(v)
                p = prev_state.var_env.get(v)
                if m is None or p is None:
                    converged = False
                    break
                if m.pretty() != p.pretty():
                    converged = False
                    break
            if converged:
                break
            prev_state = merged

        # detect mutation during iteration
        if iter_var and iter_var in self._loop_iteration_vars:
            self._loop_iteration_vars.discard(iter_var)

        # else clause
        exit_state = merged if 'merged' in dir() else loop_state  # noqa: F821
        if node.orelse:
            for stmt in node.orelse:
                exit_state = self.analyze_stmt(stmt, exit_state)

        return exit_state

    # ── with ──────────────────────────────────────────────────────────

    def analyze_with(self, node: ast.With, state: AnalysisState) -> AnalysisState:
        """Analyze a with statement: __enter__/__exit__ protocol."""
        for item in node.items:
            ctx_type, ctx_addr, state = self.eval_expr(item.context_expr, state)
            self._check_attr_safety(ctx_type, "__enter__", item.context_expr)
            self._check_attr_safety(ctx_type, "__exit__", item.context_expr)

            # resolve __enter__ return type
            enter_type = self._resolve_method_return(ctx_type, "__enter__", state)

            if item.optional_vars is not None:
                var_name = _extract_name(item.optional_vars)
                if var_name:
                    state = state.bind(var_name, enter_type)
                else:
                    state = self.analyze_assignment(
                        item.optional_vars, enter_type,
                        PrimitiveValue(kind=None, constraint=None), state,
                    )

        # analyze body
        for stmt in node.body:
            if state.is_bottom():
                break
            state = self.analyze_stmt(stmt, state)

        # check resource leaks: if user didn't use `as`, warn
        for item in node.items:
            if item.optional_vars is None:
                ctx_base = None
                try:
                    ct, _, _ = self.eval_expr(item.context_expr, state)
                    ctx_base = ct.base
                except Exception:
                    pass
                if ctx_base is not None:
                    has_close = self._type_has_attr(ctx_base, "close")
                    if has_close and not self._type_has_attr(ctx_base, "__exit__"):
                        self._warn(
                            WarningKind.RESOURCE_LEAK,
                            "Context manager may leak resources",
                            item.context_expr,
                        )

        return state

    # ── try/except ────────────────────────────────────────────────────

    def analyze_try_except(self, node: ast.Try, state: AnalysisState) -> AnalysisState:
        """Analyze try/except/else/finally."""
        # analyze try body
        try_state = state.deep_copy()
        for stmt in node.body:
            if try_state.is_bottom():
                break
            try_state = self.analyze_stmt(stmt, try_state)

        # for each handler, start from the entry state (exception may occur
        # at any point in the try body, so we use a join of entry + try_state)
        handler_entry = state.join(try_state)
        handler_states: List[AnalysisState] = []

        for handler in node.handlers:
            h_state = handler_entry.deep_copy()
            if handler.type is not None:
                exc_type = self._resolve_exception_type(handler.type)
            else:
                exc_type = PyRefinementType.simple(
                    ClassType(
                        name="BaseException",
                        address=HeapAddress(site="BaseException", context=()),
                    )
                )

            if handler.name:
                h_state = h_state.bind(handler.name, exc_type)

            for stmt in handler.body:
                if h_state.is_bottom():
                    break
                h_state = self.analyze_stmt(stmt, h_state)

            handler_states.append(h_state)

        # else body (runs if no exception)
        else_state = try_state
        if node.orelse:
            for stmt in node.orelse:
                if else_state.is_bottom():
                    break
                else_state = self.analyze_stmt(stmt, else_state)

        # join all paths
        all_paths = [else_state] + handler_states
        result = all_paths[0]
        for p in all_paths[1:]:
            result = result.join(p)

        # finally body (always runs)
        if node.finalbody:
            for stmt in node.finalbody:
                if result.is_bottom():
                    break
                result = self.analyze_stmt(stmt, result)

        return result

    # ── return ────────────────────────────────────────────────────────

    def analyze_return(self, node: ast.Return, state: AnalysisState) -> AnalysisState:
        """Record the return type and mark state as bottom (unreachable after)."""
        if node.value is not None:
            ret_type, ret_addr, state = self.eval_expr(node.value, state)
        else:
            ret_type = PyRefinementType.simple(NoneTypeP())
        self._return_types.append(ret_type)
        return AnalysisState.bottom()

    # ── expression statement ──────────────────────────────────────────

    def analyze_expr_stmt(self, node: ast.Expr, state: AnalysisState) -> AnalysisState:
        """Handle bare expression statements (typically calls)."""
        _, _, state = self.eval_expr(node.value, state)
        return state

    # ── method call ───────────────────────────────────────────────────

    def analyze_method_call(
        self,
        obj_var: str,
        method: str,
        args: List[ast.expr],
        state: AnalysisState,
    ) -> Tuple[PyRefinementType, AnalysisState]:
        """Analyze a method call obj.method(args)."""
        obj_type = state.lookup(obj_var)
        obj_addr_val = state.var_addrs.get(obj_var, TopValue())

        # check none safety
        self._check_none_safety(obj_type, f".{method}()", None)

        # check attr safety
        self._check_attr_safety(obj_type, method, None)

        # resolve method via class registry
        result_type = self._resolve_method_return(obj_type, method, state)

        # evaluate arguments
        arg_types: List[PyRefinementType] = []
        for arg_node in args:
            at, aa, state = self.eval_expr(arg_node, state)
            arg_types.append(at)

        # handle container mutations
        base = obj_type.base
        if isinstance(base, ListPyType):
            # mutation during iteration check
            if obj_var in self._loop_iteration_vars and method in (
                "append", "insert", "remove", "pop", "extend", "clear",
                "sort", "reverse", "__setitem__", "__delitem__",
            ):
                self._warn(
                    WarningKind.MUTATION_DURING_ITERATION,
                    f"Mutating list '{obj_var}' during iteration via .{method}()",
                    None,
                )
            ct = state.container_tracker.track_list_op(
                obj_var, method, [at.base for at in arg_types]
            )
            state = AnalysisState(
                heap=state.heap, var_env=state.var_env, var_addrs=state.var_addrs,
                alias_set=state.alias_set, active_predicates=state.active_predicates,
                mutation_tracker=state.mutation_tracker, container_tracker=ct,
            )
            result_type = self._list_method_result(method, base, arg_types)
        elif isinstance(base, DictPyType):
            if obj_var in self._loop_iteration_vars and method in (
                "pop", "update", "clear", "__setitem__", "__delitem__", "setdefault",
            ):
                self._warn(
                    WarningKind.MUTATION_DURING_ITERATION,
                    f"Mutating dict '{obj_var}' during iteration via .{method}()",
                    None,
                )
            ct = state.container_tracker.track_dict_op(
                obj_var, method, [at.base for at in arg_types]
            )
            state = AnalysisState(
                heap=state.heap, var_env=state.var_env, var_addrs=state.var_addrs,
                alias_set=state.alias_set, active_predicates=state.active_predicates,
                mutation_tracker=state.mutation_tracker, container_tracker=ct,
            )
            result_type = self._dict_method_result(method, base, arg_types)
        elif isinstance(base, SetPyType):
            if obj_var in self._loop_iteration_vars and method in (
                "add", "discard", "remove", "pop", "clear", "update",
            ):
                self._warn(
                    WarningKind.MUTATION_DURING_ITERATION,
                    f"Mutating set '{obj_var}' during iteration via .{method}()",
                    None,
                )
            ct = state.container_tracker.track_set_op(
                obj_var, method, [at.base for at in arg_types]
            )
            state = AnalysisState(
                heap=state.heap, var_env=state.var_env, var_addrs=state.var_addrs,
                alias_set=state.alias_set, active_predicates=state.active_predicates,
                mutation_tracker=state.mutation_tracker, container_tracker=ct,
            )
            result_type = self._set_method_result(method, base, arg_types)

        # invalidate refinements affected by the mutation
        addrs = obj_addr_val.as_addresses()
        for addr in addrs:
            invalidated = state.mutation_tracker.on_call(
                callee_addr=addr,
                arg_addrs=[
                    state.var_addrs.get(_extract_name(a) or "", TopValue())
                    for a in args
                ],
                program_point=0,
            )
            state = state.invalidate(invalidated)

        return result_type, state

    # ── function call ─────────────────────────────────────────────────

    def analyze_function_call(
        self,
        func_var: str,
        args: List[ast.expr],
        state: AnalysisState,
    ) -> Tuple[PyRefinementType, AnalysisState]:
        """Analyze a function call."""
        # evaluate arguments
        arg_types: List[PyRefinementType] = []
        for arg_node in args:
            at, aa, state = self.eval_expr(arg_node, state)
            arg_types.append(at)

        # check for known summary
        if func_var in self.function_summaries:
            summary = self.function_summaries[func_var]
            # check preconditions
            for i, (pname, ptype) in enumerate(summary.params):
                if i < len(arg_types):
                    if not arg_types[i].is_subtype_of(ptype):
                        if not isinstance(ptype.base, AnyType):
                            self._warn(
                                WarningKind.TYPE_ERROR,
                                f"Argument {pname}: expected {ptype.pretty()}, "
                                f"got {arg_types[i].pretty()}",
                                None,
                            )
            return summary.return_type, state

        # built-in function handling
        result = self._builtin_call_result(func_var, arg_types, state)
        if result is not None:
            return result, state

        # unknown function: conservative
        func_type = state.lookup(func_var)
        if isinstance(func_type.base, FunctionPyType):
            ft = func_type.base
            # check argument count
            if len(arg_types) < len(ft.param_types) - len(ft.defaults or []):
                self._warn(
                    WarningKind.TYPE_ERROR,
                    f"Too few arguments for {func_var}",
                    None,
                )
            return PyRefinementType.simple(ft.return_type), state

        if isinstance(func_type.base, ClassType):
            # constructor call
            return PyRefinementType.simple(func_type.base), state

        return PyRefinementType.simple(AnyType()), state

    # ── expression evaluator ──────────────────────────────────────────

    def eval_expr(
        self, node: ast.expr, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate *node* and return (type, abstract_value, updated_state)."""
        if isinstance(node, ast.Constant):
            base = _constant_to_type(node.value)
            rt = PyRefinementType.simple(base)
            if node.value is None:
                return rt, NoneValue(), state
            return rt, PrimitiveValue(kind=None, constraint=None), state

        if isinstance(node, ast.Name):
            rt = state.lookup(node.id)
            av = state.var_addrs.get(node.id, PrimitiveValue(kind=None, constraint=None))
            return rt, av, state

        if isinstance(node, ast.Attribute):
            obj_type, obj_addr, state = self.eval_expr(node.value, state)
            self._check_none_safety(obj_type, f".{node.attr}", node)
            self._check_attr_safety(obj_type, node.attr, node)
            # resolve attribute type
            attr_type = self._resolve_attr_type(obj_type, node.attr, obj_addr, state)
            attr_addr: AbstractValue = PrimitiveValue(kind=None, constraint=None)
            addrs = obj_addr.as_addresses()
            for addr in addrs:
                val = state.heap.read_attr(addr, node.attr)
                if not val.is_bottom():
                    attr_addr = val
                    break
            return attr_type, attr_addr, state

        if isinstance(node, ast.Subscript):
            obj_type, obj_addr, state = self.eval_expr(node.value, state)
            idx_type, idx_addr, state = self.eval_expr(node.slice, state)
            self._check_none_safety(obj_type, "[...]", node)
            elem_type = self._subscript_result_type(obj_type, idx_type, node)
            return elem_type, PrimitiveValue(kind=None, constraint=None), state

        if isinstance(node, ast.Call):
            return self._eval_call(node, state)

        if isinstance(node, ast.BinOp):
            left_type, left_addr, state = self.eval_expr(node.left, state)
            right_type, right_addr, state = self.eval_expr(node.right, state)
            result_base = _binop_result_type(left_type.base, right_type.base, node.op)
            return (
                PyRefinementType.simple(result_base),
                PrimitiveValue(kind=None, constraint=None),
                state,
            )

        if isinstance(node, ast.Compare):
            return self._eval_compare(node, state)

        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node, state)

        if isinstance(node, ast.IfExp):
            return self._eval_ifexp(node, state)

        if isinstance(node, ast.Lambda):
            return self._eval_lambda(node, state)

        if isinstance(node, ast.ListComp):
            return self._eval_list_comp(node, state)

        if isinstance(node, ast.DictComp):
            return self._eval_dict_comp(node, state)

        if isinstance(node, ast.SetComp):
            return self._eval_set_comp(node, state)

        if isinstance(node, ast.List):
            return self._eval_list_literal(node, state)

        if isinstance(node, ast.Dict):
            return self._eval_dict_literal(node, state)

        if isinstance(node, ast.Set):
            return self._eval_set_literal(node, state)

        if isinstance(node, ast.Tuple):
            return self._eval_tuple_literal(node, state)

        if isinstance(node, ast.JoinedStr):
            return (
                PyRefinementType.simple(StrPyType()),
                PrimitiveValue(kind=None, constraint=None),
                state,
            )

        if isinstance(node, ast.FormattedValue):
            _, _, state = self.eval_expr(node.value, state)
            return (
                PyRefinementType.simple(StrPyType()),
                PrimitiveValue(kind=None, constraint=None),
                state,
            )

        if isinstance(node, ast.UnaryOp):
            return self._eval_unaryop(node, state)

        if isinstance(node, ast.Starred):
            inner_type, inner_addr, state = self.eval_expr(node.value, state)
            return inner_type, inner_addr, state

        if isinstance(node, ast.Await):
            inner_type, inner_addr, state = self.eval_expr(node.value, state)
            return inner_type, inner_addr, state

        if isinstance(node, ast.Yield):
            if node.value:
                val_type, val_addr, state = self.eval_expr(node.value, state)
                return val_type, val_addr, state
            return PyRefinementType.simple(NoneTypeP()), NoneValue(), state

        if isinstance(node, ast.YieldFrom):
            val_type, val_addr, state = self.eval_expr(node.value, state)
            base = val_type.base
            if isinstance(base, ListPyType):
                return PyRefinementType.simple(base.element), val_addr, state
            return PyRefinementType.simple(AnyType()), val_addr, state

        if isinstance(node, ast.Slice):
            return (
                PyRefinementType.simple(AnyType()),
                PrimitiveValue(kind=None, constraint=None),
                state,
            )

        if isinstance(node, ast.NamedExpr):
            val_type, val_addr, state = self.eval_expr(node.value, state)
            if isinstance(node.target, ast.Name):
                state = state.bind(node.target.id, val_type, val_addr)
            return val_type, val_addr, state

        # fallback
        return (
            PyRefinementType.simple(AnyType()),
            PrimitiveValue(kind=None, constraint=None),
            state,
        )

    # ── expression sub-evaluators ─────────────────────────────────────

    def _eval_call(
        self, node: ast.Call, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a Call expression."""
        if isinstance(node.func, ast.Attribute):
            # method call: obj.method(args)
            obj_type, obj_addr, state = self.eval_expr(node.func.value, state)
            obj_name = _extract_name(node.func.value)
            if obj_name:
                result_type, state = self.analyze_method_call(
                    obj_name, node.func.attr, node.args, state
                )
                return result_type, PrimitiveValue(kind=None, constraint=None), state
            # method on a complex expression
            result_type = self._resolve_method_return(
                obj_type, node.func.attr, state
            )
            for arg in node.args:
                _, _, state = self.eval_expr(arg, state)
            return result_type, PrimitiveValue(kind=None, constraint=None), state

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # isinstance / hasattr / type / len — special forms
            if func_name == "isinstance" and len(node.args) == 2:
                return (
                    PyRefinementType.simple(BoolPyType()),
                    PrimitiveValue(kind=None, constraint=None),
                    state,
                )
            if func_name == "hasattr" and len(node.args) == 2:
                return (
                    PyRefinementType.simple(BoolPyType()),
                    PrimitiveValue(kind=None, constraint=None),
                    state,
                )
            if func_name == "len" and len(node.args) == 1:
                _, _, state = self.eval_expr(node.args[0], state)
                return (
                    PyRefinementType.simple(IntPyType()),
                    PrimitiveValue(kind=None, constraint=None),
                    state,
                )
            if func_name == "type" and len(node.args) == 1:
                arg_type, _, state = self.eval_expr(node.args[0], state)
                return (
                    PyRefinementType.simple(
                        ClassType(
                            name="type",
                            address=HeapAddress(site="type", context=()),
                        )
                    ),
                    PrimitiveValue(kind=None, constraint=None),
                    state,
                )
            if func_name in ("print", "repr", "str", "int", "float", "bool",
                             "list", "dict", "set", "tuple", "sorted", "reversed",
                             "enumerate", "zip", "map", "filter", "range",
                             "iter", "next", "abs", "min", "max", "sum",
                             "any", "all", "id", "hash", "hex", "oct", "bin",
                             "chr", "ord", "callable", "getattr", "setattr",
                             "delattr", "vars", "dir", "input", "open",
                             "super"):
                result_type, state = self.analyze_function_call(
                    func_name, node.args, state
                )
                return result_type, PrimitiveValue(kind=None, constraint=None), state

            result_type, state = self.analyze_function_call(
                func_name, node.args, state
            )
            return result_type, PrimitiveValue(kind=None, constraint=None), state

        # call on complex expression (e.g. f(x)(y))
        func_type, func_addr, state = self.eval_expr(node.func, state)
        for arg in node.args:
            _, _, state = self.eval_expr(arg, state)
        if isinstance(func_type.base, FunctionPyType):
            return (
                PyRefinementType.simple(func_type.base.return_type),
                PrimitiveValue(kind=None, constraint=None),
                state,
            )
        return (
            PyRefinementType.simple(AnyType()),
            PrimitiveValue(kind=None, constraint=None),
            state,
        )

    def _eval_compare(
        self, node: ast.Compare, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a comparison expression."""
        left_type, left_addr, state = self.eval_expr(node.left, state)
        for comparator in node.comparators:
            _, _, state = self.eval_expr(comparator, state)
        return (
            PyRefinementType.simple(BoolPyType()),
            PrimitiveValue(kind=None, constraint=None),
            state,
        )

    def _eval_boolop(
        self, node: ast.BoolOp, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate and/or expressions."""
        result_type = PyRefinementType.simple(NeverType())
        for value in node.values:
            val_type, val_addr, state = self.eval_expr(value, state)
            result_type = result_type.join(val_type)
        if isinstance(node.op, ast.And):
            # result type is the last value's type (if all truthy) or first falsy
            _, last_addr, state2 = self.eval_expr(node.values[-1], state)
            return result_type, last_addr, state
        # Or: result is first truthy value
        return result_type, PrimitiveValue(kind=None, constraint=None), state

    def _eval_ifexp(
        self, node: ast.IfExp, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a ternary if-expression."""
        true_pred, false_pred = self.guard_interpreter.interpret(node.test)

        true_env = self.propagator.propagate_true_branch(true_pred, state.var_env)
        true_state = AnalysisState(
            heap=state.heap.deep_copy(),
            var_env=true_env,
            var_addrs=dict(state.var_addrs),
            alias_set=state.alias_set,
            active_predicates=state.active_predicates,
            mutation_tracker=state.mutation_tracker,
            container_tracker=state.container_tracker,
        )
        true_type, true_addr, true_state = self.eval_expr(node.body, true_state)

        false_env = self.propagator.propagate_false_branch(true_pred, state.var_env)
        false_state = AnalysisState(
            heap=state.heap.deep_copy(),
            var_env=false_env,
            var_addrs=dict(state.var_addrs),
            alias_set=state.alias_set,
            active_predicates=state.active_predicates,
            mutation_tracker=state.mutation_tracker,
            container_tracker=state.container_tracker,
        )
        false_type, false_addr, false_state = self.eval_expr(node.orelse, false_state)

        joined = true_state.join(false_state)
        result_type = true_type.join(false_type)
        result_addr = true_addr.join(false_addr)
        return result_type, result_addr, joined

    def _eval_lambda(
        self, node: ast.Lambda, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a lambda expression."""
        param_types: List[PyType] = []
        for arg in node.args.args:
            param_types.append(_annotation_to_type(arg.annotation))
        func_type = FunctionPyType(
            param_types=tuple(param_types),
            return_type=AnyType(),
            defaults=(),
        )
        addr = _make_heap_address(node)
        return (
            PyRefinementType.simple(func_type),
            AddressValue(addresses=frozenset({addr})),
            state,
        )

    def _eval_list_comp(
        self, node: ast.ListComp, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a list comprehension."""
        inner_state = state.deep_copy()
        for gen in node.generators:
            iter_type, _, inner_state = self.eval_expr(gen.iter, inner_state)
            iter_base = iter_type.base
            if isinstance(iter_base, ListPyType):
                elem = iter_base.element
            elif isinstance(iter_base, SetPyType):
                elem = iter_base.element
            elif isinstance(iter_base, DictPyType):
                elem = iter_base.key
            elif isinstance(iter_base, StrPyType):
                elem = StrPyType()
            else:
                elem = AnyType()
            target_name = _extract_name(gen.target)
            if target_name:
                inner_state = inner_state.bind(
                    target_name, PyRefinementType.simple(elem)
                )
            for if_clause in gen.ifs:
                _, _, inner_state = self.eval_expr(if_clause, inner_state)

        elt_type, _, inner_state = self.eval_expr(node.elt, inner_state)
        addr = _make_heap_address(node)
        return (
            PyRefinementType.simple(ListPyType(element=elt_type.base)),
            AddressValue(addresses=frozenset({addr})),
            state,
        )

    def _eval_dict_comp(
        self, node: ast.DictComp, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a dict comprehension."""
        inner_state = state.deep_copy()
        for gen in node.generators:
            iter_type, _, inner_state = self.eval_expr(gen.iter, inner_state)
            iter_base = iter_type.base
            if isinstance(iter_base, DictPyType):
                elem = iter_base.key
            elif isinstance(iter_base, ListPyType):
                elem = iter_base.element
            else:
                elem = AnyType()
            target_name = _extract_name(gen.target)
            if target_name:
                inner_state = inner_state.bind(
                    target_name, PyRefinementType.simple(elem)
                )
            for if_clause in gen.ifs:
                _, _, inner_state = self.eval_expr(if_clause, inner_state)

        key_type, _, inner_state = self.eval_expr(node.key, inner_state)
        val_type, _, inner_state = self.eval_expr(node.value, inner_state)
        addr = _make_heap_address(node)
        return (
            PyRefinementType.simple(DictPyType(key=key_type.base, value=val_type.base)),
            AddressValue(addresses=frozenset({addr})),
            state,
        )

    def _eval_set_comp(
        self, node: ast.SetComp, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a set comprehension."""
        inner_state = state.deep_copy()
        for gen in node.generators:
            iter_type, _, inner_state = self.eval_expr(gen.iter, inner_state)
            iter_base = iter_type.base
            if isinstance(iter_base, (ListPyType, SetPyType)):
                elem = getattr(iter_base, "element", AnyType())
            else:
                elem = AnyType()
            target_name = _extract_name(gen.target)
            if target_name:
                inner_state = inner_state.bind(
                    target_name, PyRefinementType.simple(elem)
                )
            for if_clause in gen.ifs:
                _, _, inner_state = self.eval_expr(if_clause, inner_state)

        elt_type, _, inner_state = self.eval_expr(node.elt, inner_state)
        addr = _make_heap_address(node)
        return (
            PyRefinementType.simple(SetPyType(element=elt_type.base)),
            AddressValue(addresses=frozenset({addr})),
            state,
        )

    def _eval_list_literal(
        self, node: ast.List, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a list literal [a, b, ...]."""
        elem_types: List[PyType] = []
        for elt in node.elts:
            et, _, state = self.eval_expr(elt, state)
            elem_types.append(et.base)
        if elem_types:
            combined = elem_types[0]
            for e in elem_types[1:]:
                combined = combined.join(e)
        else:
            combined = AnyType()
        addr = _make_heap_address(node)
        obj = HeapObject(
            cls_addr=HeapAddress(site="list", context=()),
            fields={"__len__": PrimitiveValue(kind=None, constraint=None)},
            recency=RecencyFlag.RECENT,
        )
        state.heap.objects[addr] = obj
        rt = PyRefinementType.simple(ListPyType(element=combined))
        if elem_types:
            rt = rt.with_predicate(
                HeapPredicate.container_len("v", "==", len(elem_types))
            )
        return rt, AddressValue(addresses=frozenset({addr})), state

    def _eval_dict_literal(
        self, node: ast.Dict, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a dict literal {k: v, ...}."""
        key_types: List[PyType] = []
        val_types: List[PyType] = []
        for k, v in zip(node.keys, node.values):
            if k is not None:
                kt, _, state = self.eval_expr(k, state)
                key_types.append(kt.base)
            else:
                key_types.append(AnyType())
            vt, _, state = self.eval_expr(v, state)
            val_types.append(vt.base)

        key_t = AnyType()
        val_t = AnyType()
        if key_types:
            key_t = key_types[0]
            for k in key_types[1:]:
                key_t = key_t.join(k)
        if val_types:
            val_t = val_types[0]
            for v in val_types[1:]:
                val_t = val_t.join(v)

        addr = _make_heap_address(node)
        obj = HeapObject(
            cls_addr=HeapAddress(site="dict", context=()),
            fields={},
            recency=RecencyFlag.RECENT,
        )
        state.heap.objects[addr] = obj
        rt = PyRefinementType.simple(DictPyType(key=key_t, value=val_t))
        if node.keys:
            rt = rt.with_predicate(
                HeapPredicate.container_len("v", "==", len(node.keys))
            )
        return rt, AddressValue(addresses=frozenset({addr})), state

    def _eval_set_literal(
        self, node: ast.Set, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a set literal {a, b, ...}."""
        elem_types: List[PyType] = []
        for elt in node.elts:
            et, _, state = self.eval_expr(elt, state)
            elem_types.append(et.base)
        if elem_types:
            combined = elem_types[0]
            for e in elem_types[1:]:
                combined = combined.join(e)
        else:
            combined = AnyType()
        addr = _make_heap_address(node)
        return (
            PyRefinementType.simple(SetPyType(element=combined)),
            AddressValue(addresses=frozenset({addr})),
            state,
        )

    def _eval_tuple_literal(
        self, node: ast.Tuple, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a tuple literal (a, b, ...)."""
        elem_types: List[PyType] = []
        for elt in node.elts:
            et, _, state = self.eval_expr(elt, state)
            elem_types.append(et.base)
        return (
            PyRefinementType.simple(TuplePyType(elements=tuple(elem_types))),
            PrimitiveValue(kind=None, constraint=None),
            state,
        )

    def _eval_unaryop(
        self, node: ast.UnaryOp, state: AnalysisState
    ) -> Tuple[PyRefinementType, AbstractValue, AnalysisState]:
        """Evaluate a unary operation."""
        operand_type, operand_addr, state = self.eval_expr(node.operand, state)
        if isinstance(node.op, ast.Not):
            return (
                PyRefinementType.simple(BoolPyType()),
                PrimitiveValue(kind=None, constraint=None),
                state,
            )
        if isinstance(node.op, ast.USub):
            base = operand_type.base
            if isinstance(base, IntPyType):
                return PyRefinementType.simple(IntPyType()), operand_addr, state
            if isinstance(base, FloatPyType):
                return PyRefinementType.simple(FloatPyType()), operand_addr, state
            return PyRefinementType.simple(AnyType()), operand_addr, state
        if isinstance(node.op, ast.UAdd):
            return operand_type, operand_addr, state
        if isinstance(node.op, ast.Invert):
            if isinstance(operand_type.base, (IntPyType, BoolPyType)):
                return PyRefinementType.simple(IntPyType()), operand_addr, state
            return PyRefinementType.simple(AnyType()), operand_addr, state
        return operand_type, operand_addr, state

    # ── diagnostics ───────────────────────────────────────────────────

    def _warn(
        self, kind: WarningKind, message: str, node: Optional[ast.AST]
    ) -> None:
        """Record an analysis warning."""
        line = getattr(node, "lineno", 0) if node else 0
        col = getattr(node, "col_offset", 0) if node else 0
        severity = "error" if kind in (
            WarningKind.NPE, WarningKind.TYPE_ERROR,
        ) else "warning"
        self.warnings.append(
            AnalysisWarning(
                kind=kind, message=message,
                line=line, column=col, severity=severity,
            )
        )

    def _check_none_safety(
        self,
        value_type: PyRefinementType,
        operation: str,
        node: Optional[ast.AST],
    ) -> None:
        """Check if a None dereference is possible."""
        base = value_type.base
        if isinstance(base, NoneTypeP):
            self._warn(
                WarningKind.NPE,
                f"Definite None dereference on {operation}",
                node,
            )
            return
        if isinstance(base, OptionalType):
            # check if NOT_NONE predicate is present
            has_not_none = any(
                p.kind == HeapPredKind.NOT_NONE for p in value_type.predicates
            )
            if not has_not_none:
                self._warn(
                    WarningKind.NPE,
                    f"Possible None dereference on {operation}",
                    node,
                )
            return
        if isinstance(base, PyUnionType):
            has_none = any(isinstance(m, NoneTypeP) for m in base.members)
            if has_none:
                has_not_none = any(
                    p.kind == HeapPredKind.NOT_NONE for p in value_type.predicates
                )
                if not has_not_none:
                    self._warn(
                        WarningKind.NPE,
                        f"Possible None dereference on {operation}",
                        node,
                    )

    def _check_attr_safety(
        self,
        obj_type: PyRefinementType,
        attr_name: str,
        node: Optional[ast.AST],
    ) -> None:
        """Check if attribute access is safe."""
        base = obj_type.base
        if isinstance(base, NoneTypeP):
            self._warn(
                WarningKind.ATTRIBUTE_ERROR,
                f"NoneType has no attribute '{attr_name}'",
                node,
            )
            return
        if isinstance(base, ClassType):
            cls = self.class_registry.lookup_by_name(base.name)
            if cls is not None:
                has_attr = (
                    cls.has_class_attr(attr_name)
                    or attr_name in cls.instance_attrs
                )
                if not has_attr:
                    self._warn(
                        WarningKind.ATTRIBUTE_ERROR,
                        f"'{base.name}' object has no attribute '{attr_name}'",
                        node,
                    )
            return
        if isinstance(base, ProtocolType):
            if attr_name not in base.required_attrs and attr_name not in base.required_methods:
                # protocols don't forbid extra attrs, so this is a soft check
                pass

    # ── private helpers ───────────────────────────────────────────────

    def _resolve_method_return(
        self,
        obj_type: PyRefinementType,
        method_name: str,
        state: AnalysisState,
    ) -> PyRefinementType:
        """Resolve the return type of a method call on *obj_type*."""
        base = obj_type.base
        # string methods
        if isinstance(base, StrPyType):
            str_methods: Dict[str, PyType] = {
                "upper": StrPyType(), "lower": StrPyType(),
                "strip": StrPyType(), "lstrip": StrPyType(), "rstrip": StrPyType(),
                "replace": StrPyType(), "join": StrPyType(),
                "format": StrPyType(), "encode": BytesPyType(),
                "split": ListPyType(element=StrPyType()),
                "rsplit": ListPyType(element=StrPyType()),
                "splitlines": ListPyType(element=StrPyType()),
                "find": IntPyType(), "rfind": IntPyType(),
                "index": IntPyType(), "rindex": IntPyType(),
                "count": IntPyType(),
                "startswith": BoolPyType(), "endswith": BoolPyType(),
                "isdigit": BoolPyType(), "isalpha": BoolPyType(),
                "isalnum": BoolPyType(), "isspace": BoolPyType(),
                "isupper": BoolPyType(), "islower": BoolPyType(),
                "title": StrPyType(), "capitalize": StrPyType(),
                "swapcase": StrPyType(), "center": StrPyType(),
                "ljust": StrPyType(), "rjust": StrPyType(),
                "zfill": StrPyType(), "expandtabs": StrPyType(),
                "partition": TuplePyType(elements=(StrPyType(), StrPyType(), StrPyType())),
                "rpartition": TuplePyType(elements=(StrPyType(), StrPyType(), StrPyType())),
                "__contains__": BoolPyType(),
                "__len__": IntPyType(),
            }
            if method_name in str_methods:
                return PyRefinementType.simple(str_methods[method_name])

        # list methods
        if isinstance(base, ListPyType):
            return self._list_method_result(method_name, base, [])

        # dict methods
        if isinstance(base, DictPyType):
            return self._dict_method_result(method_name, base, [])

        # set methods
        if isinstance(base, SetPyType):
            return self._set_method_result(method_name, base, [])

        # class instance methods via registry
        if isinstance(base, ClassType):
            cls = self.class_registry.lookup_by_name(base.name)
            if cls is not None:
                method_val = cls.get_method(method_name)
                if method_val is not None:
                    return PyRefinementType.simple(AnyType())
                # __enter__ for context managers
                if method_name == "__enter__":
                    return PyRefinementType.simple(base)
                if method_name == "__exit__":
                    return PyRefinementType.simple(NoneTypeP())

        # protocol methods
        if isinstance(base, ProtocolType):
            if method_name in base.required_methods:
                ft = base.required_methods[method_name]
                return PyRefinementType.simple(ft.return_type)

        return PyRefinementType.simple(AnyType())

    def _list_method_result(
        self,
        method: str,
        base: ListPyType,
        arg_types: List[PyRefinementType],
    ) -> PyRefinementType:
        """Return type of a list method call."""
        method_types: Dict[str, PyType] = {
            "append": NoneTypeP(),
            "extend": NoneTypeP(),
            "insert": NoneTypeP(),
            "remove": NoneTypeP(),
            "clear": NoneTypeP(),
            "sort": NoneTypeP(),
            "reverse": NoneTypeP(),
            "copy": ListPyType(element=base.element),
            "pop": base.element,
            "index": IntPyType(),
            "count": IntPyType(),
            "__len__": IntPyType(),
            "__contains__": BoolPyType(),
            "__iter__": AnyType(),
            "__getitem__": base.element,
        }
        if method in method_types:
            return PyRefinementType.simple(method_types[method])
        return PyRefinementType.simple(AnyType())

    def _dict_method_result(
        self,
        method: str,
        base: DictPyType,
        arg_types: List[PyRefinementType],
    ) -> PyRefinementType:
        """Return type of a dict method call."""
        method_types: Dict[str, PyType] = {
            "get": OptionalType(inner=base.value),
            "pop": base.value,
            "setdefault": base.value,
            "update": NoneTypeP(),
            "clear": NoneTypeP(),
            "copy": DictPyType(key=base.key, value=base.value),
            "keys": AnyType(),
            "values": AnyType(),
            "items": AnyType(),
            "__len__": IntPyType(),
            "__contains__": BoolPyType(),
            "__getitem__": base.value,
            "__setitem__": NoneTypeP(),
            "__delitem__": NoneTypeP(),
        }
        if method in method_types:
            return PyRefinementType.simple(method_types[method])
        return PyRefinementType.simple(AnyType())

    def _set_method_result(
        self,
        method: str,
        base: SetPyType,
        arg_types: List[PyRefinementType],
    ) -> PyRefinementType:
        """Return type of a set method call."""
        method_types: Dict[str, PyType] = {
            "add": NoneTypeP(),
            "discard": NoneTypeP(),
            "remove": NoneTypeP(),
            "pop": base.element,
            "clear": NoneTypeP(),
            "copy": SetPyType(element=base.element),
            "update": NoneTypeP(),
            "union": SetPyType(element=base.element),
            "intersection": SetPyType(element=base.element),
            "difference": SetPyType(element=base.element),
            "symmetric_difference": SetPyType(element=base.element),
            "issubset": BoolPyType(),
            "issuperset": BoolPyType(),
            "isdisjoint": BoolPyType(),
            "__len__": IntPyType(),
            "__contains__": BoolPyType(),
        }
        if method in method_types:
            return PyRefinementType.simple(method_types[method])
        return PyRefinementType.simple(AnyType())

    def _resolve_attr_type(
        self,
        obj_type: PyRefinementType,
        attr_name: str,
        obj_addr: AbstractValue,
        state: AnalysisState,
    ) -> PyRefinementType:
        """Resolve the type of an attribute access."""
        base = obj_type.base
        if isinstance(base, ClassType):
            cls = self.class_registry.lookup_by_name(base.name)
            if cls is not None:
                if cls.has_class_attr(attr_name):
                    attr_val = cls.get_class_attr(attr_name)
                    if attr_val is not None:
                        return PyRefinementType.simple(AnyType())
                if attr_name in cls.instance_attrs:
                    addrs = obj_addr.as_addresses()
                    for addr in addrs:
                        val = state.heap.read_attr(addr, attr_name)
                        if not val.is_bottom():
                            return PyRefinementType.simple(AnyType())
        if isinstance(base, ProtocolType):
            if attr_name in base.required_attrs:
                return PyRefinementType.simple(base.required_attrs[attr_name])
        # module-level / fallback
        return PyRefinementType.simple(AnyType())

    def _subscript_result_type(
        self,
        obj_type: PyRefinementType,
        idx_type: PyRefinementType,
        node: ast.AST,
    ) -> PyRefinementType:
        """Determine the result type of obj[idx]."""
        base = obj_type.base
        if isinstance(base, ListPyType):
            return PyRefinementType.simple(base.element)
        if isinstance(base, DictPyType):
            # check for key error
            if isinstance(idx_type.base, StrPyType):
                pass  # could check known keys
            return PyRefinementType.simple(base.value)
        if isinstance(base, TuplePyType):
            if isinstance(idx_type.base, IntPyType):
                # could resolve constant index
                pass
            if base.elements:
                combined = base.elements[0]
                for e in base.elements[1:]:
                    combined = combined.join(e)
                return PyRefinementType.simple(combined)
            return PyRefinementType.simple(AnyType())
        if isinstance(base, StrPyType):
            return PyRefinementType.simple(StrPyType())
        if isinstance(base, BytesPyType):
            return PyRefinementType.simple(IntPyType())
        return PyRefinementType.simple(AnyType())

    def _builtin_call_result(
        self,
        func_name: str,
        arg_types: List[PyRefinementType],
        state: AnalysisState,
    ) -> Optional[PyRefinementType]:
        """Return the result type of a built-in function call, or None."""
        builtins: Dict[str, PyType] = {
            "print": NoneTypeP(),
            "repr": StrPyType(),
            "str": StrPyType(),
            "int": IntPyType(),
            "float": FloatPyType(),
            "bool": BoolPyType(),
            "abs": IntPyType(),
            "len": IntPyType(),
            "hash": IntPyType(),
            "id": IntPyType(),
            "hex": StrPyType(),
            "oct": StrPyType(),
            "bin": StrPyType(),
            "chr": StrPyType(),
            "ord": IntPyType(),
            "callable": BoolPyType(),
            "isinstance": BoolPyType(),
            "issubclass": BoolPyType(),
            "hasattr": BoolPyType(),
            "any": BoolPyType(),
            "all": BoolPyType(),
            "input": StrPyType(),
            "vars": DictPyType(key=StrPyType(), value=AnyType()),
            "dir": ListPyType(element=StrPyType()),
        }
        if func_name in builtins:
            return PyRefinementType.simple(builtins[func_name])

        # container constructors
        if func_name == "list":
            if arg_types:
                inner = arg_types[0].base
                if isinstance(inner, ListPyType):
                    return PyRefinementType.simple(inner)
                if isinstance(inner, SetPyType):
                    return PyRefinementType.simple(ListPyType(element=inner.element))
                if isinstance(inner, DictPyType):
                    return PyRefinementType.simple(ListPyType(element=inner.key))
                if isinstance(inner, StrPyType):
                    return PyRefinementType.simple(ListPyType(element=StrPyType()))
            return PyRefinementType.simple(ListPyType(element=AnyType()))

        if func_name == "dict":
            return PyRefinementType.simple(DictPyType(key=AnyType(), value=AnyType()))

        if func_name == "set":
            if arg_types:
                inner = arg_types[0].base
                if isinstance(inner, (ListPyType, SetPyType)):
                    elem = getattr(inner, "element", AnyType())
                    return PyRefinementType.simple(SetPyType(element=elem))
                if isinstance(inner, StrPyType):
                    return PyRefinementType.simple(SetPyType(element=StrPyType()))
            return PyRefinementType.simple(SetPyType(element=AnyType()))

        if func_name == "tuple":
            if arg_types:
                inner = arg_types[0].base
                if isinstance(inner, TuplePyType):
                    return PyRefinementType.simple(inner)
                if isinstance(inner, ListPyType):
                    return PyRefinementType.simple(
                        TuplePyType(elements=(inner.element,))
                    )
            return PyRefinementType.simple(TuplePyType(elements=()))

        if func_name == "sorted":
            if arg_types:
                inner = arg_types[0].base
                if isinstance(inner, ListPyType):
                    return PyRefinementType.simple(inner)
                if isinstance(inner, SetPyType):
                    return PyRefinementType.simple(ListPyType(element=inner.element))
            return PyRefinementType.simple(ListPyType(element=AnyType()))

        if func_name == "reversed":
            if arg_types:
                return PyRefinementType.simple(arg_types[0].base)
            return PyRefinementType.simple(AnyType())

        if func_name == "enumerate":
            if arg_types:
                inner = arg_types[0].base
                elem = AnyType()
                if isinstance(inner, ListPyType):
                    elem = inner.element
                return PyRefinementType.simple(
                    ListPyType(
                        element=TuplePyType(elements=(IntPyType(), elem))
                    )
                )
            return PyRefinementType.simple(AnyType())

        if func_name == "zip":
            elems: List[PyType] = []
            for at in arg_types:
                inner = at.base
                if isinstance(inner, ListPyType):
                    elems.append(inner.element)
                elif isinstance(inner, TuplePyType) and inner.elements:
                    combined = inner.elements[0]
                    for e in inner.elements[1:]:
                        combined = combined.join(e)
                    elems.append(combined)
                else:
                    elems.append(AnyType())
            if elems:
                return PyRefinementType.simple(
                    ListPyType(element=TuplePyType(elements=tuple(elems)))
                )
            return PyRefinementType.simple(ListPyType(element=AnyType()))

        if func_name == "map":
            return PyRefinementType.simple(ListPyType(element=AnyType()))

        if func_name == "filter":
            if len(arg_types) >= 2:
                return PyRefinementType.simple(ListPyType(element=arg_types[1].base))
            return PyRefinementType.simple(ListPyType(element=AnyType()))

        if func_name == "range":
            return PyRefinementType.simple(ListPyType(element=IntPyType()))

        if func_name == "iter":
            if arg_types:
                return PyRefinementType.simple(arg_types[0].base)
            return PyRefinementType.simple(AnyType())

        if func_name == "next":
            if arg_types:
                inner = arg_types[0].base
                if isinstance(inner, ListPyType):
                    return PyRefinementType.simple(inner.element)
            return PyRefinementType.simple(AnyType())

        if func_name == "min" or func_name == "max":
            if arg_types:
                if len(arg_types) == 1:
                    inner = arg_types[0].base
                    if isinstance(inner, ListPyType):
                        return PyRefinementType.simple(inner.element)
                    return PyRefinementType.simple(AnyType())
                result = arg_types[0]
                for at in arg_types[1:]:
                    result = result.join(at)
                return result
            return PyRefinementType.simple(AnyType())

        if func_name == "sum":
            if arg_types:
                inner = arg_types[0].base
                if isinstance(inner, ListPyType):
                    elem = inner.element
                    if isinstance(elem, (IntPyType, FloatPyType)):
                        return PyRefinementType.simple(elem)
                return PyRefinementType.simple(IntPyType())
            return PyRefinementType.simple(IntPyType())

        if func_name == "open":
            return PyRefinementType.simple(
                ClassType(
                    name="TextIOWrapper",
                    address=HeapAddress(site="io.TextIOWrapper", context=()),
                )
            )

        if func_name == "super":
            return PyRefinementType.simple(AnyType())

        if func_name == "getattr":
            if len(arg_types) >= 3:
                # has default
                return arg_types[2].join(PyRefinementType.simple(AnyType()))
            return PyRefinementType.simple(AnyType())

        if func_name == "setattr" or func_name == "delattr":
            return PyRefinementType.simple(NoneTypeP())

        return None

    def _resolve_exception_type(self, node: ast.expr) -> PyRefinementType:
        """Resolve exception type from except handler type annotation."""
        if isinstance(node, ast.Name):
            return PyRefinementType.simple(
                ClassType(
                    name=node.id,
                    address=HeapAddress(site=node.id, context=()),
                )
            )
        if isinstance(node, ast.Tuple):
            members: Set[PyType] = set()
            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    members.add(
                        ClassType(
                            name=elt.id,
                            address=HeapAddress(site=elt.id, context=()),
                        )
                    )
            if members:
                return PyRefinementType.simple(PyUnionType(members=frozenset(members)))
        return PyRefinementType.simple(
            ClassType(
                name="BaseException",
                address=HeapAddress(site="BaseException", context=()),
            )
        )

    def _function_def_to_type(
        self, func: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> FunctionPyType:
        """Convert a FunctionDef to a FunctionPyType."""
        param_types: List[PyType] = []
        for arg in func.args.args:
            param_types.append(_annotation_to_type(arg.annotation))
        ret = _annotation_to_type(func.returns)
        defaults: List[PyType] = []
        for d in func.args.defaults:
            if isinstance(d, ast.Constant):
                defaults.append(_constant_to_type(d.value))
            else:
                defaults.append(AnyType())
        return FunctionPyType(
            param_types=tuple(param_types),
            return_type=ret,
            defaults=tuple(defaults),
        )

    def _analyze_class_def(
        self, node: ast.ClassDef, state: AnalysisState
    ) -> AnalysisState:
        """Analyze a class definition and register it."""
        addr = HeapAddress(site=f"class_{node.name}", context=())
        bases: List[HeapAddress] = []
        for base_node in node.bases:
            if isinstance(base_node, ast.Name):
                bases.append(HeapAddress(site=base_node.id, context=()))

        instance_attrs: Set[str] = set()
        class_attrs: Dict[str, AbstractValue] = {}

        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_addr = HeapAddress(site=f"{node.name}.{stmt.name}", context=())
                class_attrs[stmt.name] = AddressValue(
                    addresses=frozenset({method_addr})
                )
                # scan __init__ for instance attrs
                if stmt.name == "__init__":
                    for s in ast.walk(stmt):
                        if isinstance(s, ast.Attribute):
                            if isinstance(s.value, ast.Name) and s.value.id == "self":
                                instance_attrs.add(s.attr)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        class_attrs[target.id] = PrimitiveValue(kind=None, constraint=None)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                class_attrs[stmt.target.id] = PrimitiveValue(kind=None, constraint=None)

        mro: List[HeapAddress] = [addr] + bases
        try:
            computed_mro = self.mro_computer.compute_mro(addr, {
                addr: bases,
                **{b: [] for b in bases},
            })
            mro = computed_mro
        except Exception:
            pass

        py_class = PythonClass(
            name=node.name,
            address=addr,
            bases=bases,
            mro=mro,
            class_attrs=class_attrs,
            instance_attrs=instance_attrs,
            descriptors={},
        )
        self.class_registry.register(py_class)

        class_type = ClassType(name=node.name, address=addr)
        return state.bind(node.name, PyRefinementType.simple(class_type))

    def _analyze_import(
        self,
        node: Union[ast.Import, ast.ImportFrom],
        state: AnalysisState,
    ) -> AnalysisState:
        """Handle import statements conservatively."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                state = state.bind(name, PyRefinementType.simple(AnyType()))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                state = state.bind(name, PyRefinementType.simple(AnyType()))
        return state

    def _analyze_assert(
        self, stmt: ast.Assert, state: AnalysisState
    ) -> AnalysisState:
        """Handle assert statements by narrowing the environment."""
        pred = self.guard_interpreter.interpret_assert(stmt)
        narrowed_env = self.propagator.propagate_assert(pred, state.var_env)
        return AnalysisState(
            heap=state.heap,
            var_env=narrowed_env,
            var_addrs=state.var_addrs,
            alias_set=state.alias_set,
            active_predicates=state.active_predicates | {pred},
            mutation_tracker=state.mutation_tracker,
            container_tracker=state.container_tracker,
        )

    def _analyze_delete(
        self, stmt: ast.Delete, state: AnalysisState
    ) -> AnalysisState:
        """Handle del statements."""
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                new_env = dict(state.var_env)
                new_env.pop(target.id, None)
                new_addrs = dict(state.var_addrs)
                new_addrs.pop(target.id, None)
                state = AnalysisState(
                    heap=state.heap,
                    var_env=new_env,
                    var_addrs=new_addrs,
                    alias_set=state.alias_set,
                    active_predicates=state.active_predicates,
                    mutation_tracker=state.mutation_tracker,
                    container_tracker=state.container_tracker,
                )
            elif isinstance(target, ast.Attribute):
                obj_type, obj_addr, state = self.eval_expr(target.value, state)
                addrs = obj_addr.as_addresses()
                obj_name = _extract_name(target.value)
                if obj_name:
                    for addr in addrs:
                        invalidated = state.mutation_tracker.on_delete_attr(
                            target_addr=addr,
                            field_name=target.attr,
                            program_point=getattr(target, "lineno", 0),
                        )
                        state = state.invalidate(invalidated)
            elif isinstance(target, ast.Subscript):
                obj_type, obj_addr, state = self.eval_expr(target.value, state)
                idx_type, idx_addr, state = self.eval_expr(target.slice, state)
                addrs = obj_addr.as_addresses()
                obj_name = _extract_name(target.value)
                if obj_name:
                    for addr in addrs:
                        invalidated = state.mutation_tracker.on_delete_subscript(
                            target_addr=addr,
                            key_value=idx_addr,
                            program_point=getattr(target, "lineno", 0),
                        )
                        state = state.invalidate(invalidated)
        return state

    def _analyze_raise(
        self, stmt: ast.Raise, state: AnalysisState
    ) -> AnalysisState:
        """Handle raise statements (makes control flow unreachable)."""
        if stmt.exc:
            _, _, state = self.eval_expr(stmt.exc, state)
        return AnalysisState.bottom()

    def _type_has_attr(self, base: PyType, attr: str) -> bool:
        """Check if a type has a particular attribute."""
        if isinstance(base, ClassType):
            cls = self.class_registry.lookup_by_name(base.name)
            if cls is not None:
                return cls.has_class_attr(attr) or attr in cls.instance_attrs
        if isinstance(base, ProtocolType):
            return attr in base.required_attrs or attr in base.required_methods
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 4.  ModuleAnalysisResult
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModuleAnalysisResult:
    """Result of analyzing a complete Python module."""
    function_types: Dict[str, FunctionRefinement] = field(default_factory=dict)
    class_types: Dict[str, PythonClass] = field(default_factory=dict)
    global_types: Dict[str, PyRefinementType] = field(default_factory=dict)
    warnings: List[AnalysisWarning] = field(default_factory=list)
    heap_snapshot: AbstractHeap = field(default_factory=AbstractHeap)

    def get_warnings_for_line(self, line: int) -> List[AnalysisWarning]:
        """Return all warnings at a given line number."""
        return [w for w in self.warnings if w.line == line]

    def get_function_type(self, name: str) -> Optional[FunctionRefinement]:
        """Return the inferred function refinement for *name*, or None."""
        return self.function_types.get(name)

    def summary(self) -> str:
        """Produce a human-readable summary of the analysis results."""
        lines: List[str] = []
        lines.append(f"Module analysis: {len(self.function_types)} functions, "
                      f"{len(self.class_types)} classes, "
                      f"{len(self.warnings)} warnings")
        lines.append("")

        if self.function_types:
            lines.append("Functions:")
            for name, fref in sorted(self.function_types.items()):
                lines.append(f"  {name}: {fref.pretty()}")
            lines.append("")

        if self.class_types:
            lines.append("Classes:")
            for name, cls in sorted(self.class_types.items()):
                attrs = sorted(cls.instance_attrs)
                methods = sorted(
                    k for k in cls.class_attrs if k not in ("__module__", "__qualname__")
                )
                lines.append(f"  {name}: attrs={attrs}, methods={methods}")
            lines.append("")

        if self.global_types:
            lines.append("Globals:")
            for name, rt in sorted(self.global_types.items()):
                lines.append(f"  {name}: {rt.pretty()}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for w in sorted(self.warnings, key=lambda w: (w.line, w.column)):
                lines.append(f"  L{w.line}:{w.column} [{w.severity}] {w.message}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ModuleAnalyzer — top-level entry point
# ═══════════════════════════════════════════════════════════════════════════


class ModuleAnalyzer:
    """Top-level analyzer that takes source code and produces results."""

    def __init__(
        self,
        class_registry: Optional[ClassRegistry] = None,
        widening_threshold: int = 3,
    ) -> None:
        self._registry = class_registry or ClassRegistry()
        self._widening_threshold = widening_threshold

    def analyze(self, source: str) -> ModuleAnalysisResult:
        """Parse and analyze a Python source string."""
        tree = ast.parse(source)
        return self.analyze_ast(tree)

    def analyze_ast(self, tree: ast.Module) -> ModuleAnalysisResult:
        """Analyze an already-parsed AST module."""
        analyzer = HeapSensitiveAnalyzer(
            class_registry=self._registry,
            widening_threshold=self._widening_threshold,
        )

        # first pass: register classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                addr = HeapAddress(site=f"class_{node.name}", context=())
                bases: List[HeapAddress] = []
                for base_node in node.bases:
                    if isinstance(base_node, ast.Name):
                        bases.append(HeapAddress(site=base_node.id, context=()))
                instance_attrs: Set[str] = set()
                class_attrs: Dict[str, AbstractValue] = {}
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_addr = HeapAddress(
                            site=f"{node.name}.{stmt.name}", context=()
                        )
                        class_attrs[stmt.name] = AddressValue(
                            addresses=frozenset({method_addr})
                        )
                        if stmt.name == "__init__":
                            for s in ast.walk(stmt):
                                if isinstance(s, ast.Attribute):
                                    if (
                                        isinstance(s.value, ast.Name)
                                        and s.value.id == "self"
                                    ):
                                        instance_attrs.add(s.attr)
                    elif isinstance(stmt, ast.Assign):
                        for t in stmt.targets:
                            if isinstance(t, ast.Name):
                                class_attrs[t.id] = PrimitiveValue(
                                    kind=None, constraint=None
                                )
                    elif isinstance(stmt, ast.AnnAssign):
                        if isinstance(stmt.target, ast.Name):
                            class_attrs[stmt.target.id] = PrimitiveValue(
                                kind=None, constraint=None
                            )

                mro = [addr] + bases
                py_class = PythonClass(
                    name=node.name,
                    address=addr,
                    bases=bases,
                    mro=mro,
                    class_attrs=class_attrs,
                    instance_attrs=instance_attrs,
                    descriptors={},
                )
                self._registry.register(py_class)

        # second pass: analyze module
        func_summaries = analyzer.analyze_module(tree)

        # collect class types from registry
        class_types: Dict[str, PythonClass] = {}
        for addr, cls in self._registry.classes.items():
            class_types[cls.name] = cls

        # collect global types from the module-level state
        global_types: Dict[str, PyRefinementType] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in func_summaries and target.id not in class_types:
                            global_types[target.id] = PyRefinementType.simple(AnyType())
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    if name not in func_summaries and name not in class_types:
                        global_types[name] = PyRefinementType.simple(
                            _annotation_to_type(node.annotation)
                        )

        return ModuleAnalysisResult(
            function_types=func_summaries,
            class_types=class_types,
            global_types=global_types,
            warnings=list(analyzer.warnings),
            heap_snapshot=analyzer.guard_interpreter.heap,
        )
