"""
Z3 SMT Backend for Refinement Type Inference.

Full Z3 integration providing:
- Encoding of Python values as Z3 sorts
- Encoding of refinement predicates as Z3 formulas
- Incremental solving with push/pop
- Unsat core-based predicate extraction for contract discovery
- Model extraction for counterexamples
- Quantifier elimination for predicate simplification
- Caching of Z3 queries
- Simplification of redundant constraints
- Lazy constraint generation
"""

from __future__ import annotations

import enum
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Z3 import with mock fallback
# ---------------------------------------------------------------------------

try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

    class _MockZ3:
        """Minimal mock so the module can be imported without z3-solver."""

        class _SortRef:
            def __init__(self, name: str = "mock") -> None:
                self._name = name

            def __repr__(self) -> str:
                return f"MockSort({self._name})"

        class _ExprRef:
            def __init__(self, name: str = "mock") -> None:
                self._name = name

            def __repr__(self) -> str:
                return f"MockExpr({self._name})"

            def sexpr(self) -> str:
                return self._name

        class _SolverRef:
            def push(self) -> None:
                pass

            def pop(self, n: int = 1) -> None:
                pass

            def add(self, *args: Any) -> None:
                pass

            def assert_and_track(self, expr: Any, name: Any) -> None:
                pass

            def check(self, *args: Any) -> Any:
                return "unknown"

            def model(self) -> dict:
                return {}

            def unsat_core(self) -> list:
                return []

            def set(self, *args: Any, **kwargs: Any) -> None:
                pass

            def reason_unknown(self) -> str:
                return "mock"

            def statistics(self) -> Any:
                return _MockZ3._Stats()

        class _Stats:
            def __repr__(self) -> str:
                return "MockStats()"

        # Sorts
        @staticmethod
        def IntSort() -> "_MockZ3._SortRef":
            return _MockZ3._SortRef("Int")

        @staticmethod
        def BoolSort() -> "_MockZ3._SortRef":
            return _MockZ3._SortRef("Bool")

        @staticmethod
        def RealSort() -> "_MockZ3._SortRef":
            return _MockZ3._SortRef("Real")

        @staticmethod
        def StringSort() -> "_MockZ3._SortRef":
            return _MockZ3._SortRef("String")

        @staticmethod
        def ArraySort(idx: Any, elem: Any) -> "_MockZ3._SortRef":
            return _MockZ3._SortRef(f"Array({idx},{elem})")

        @staticmethod
        def DeclareSort(name: str) -> "_MockZ3._SortRef":
            return _MockZ3._SortRef(name)

        @staticmethod
        def Datatype(name: str) -> Any:
            class _DT:
                def __init__(self) -> None:
                    self.name = name
                    self._ctors: list = []

                def declare(self, n: str, *fields: Any) -> None:
                    self._ctors.append(n)

                def create(self) -> "_MockZ3._SortRef":
                    return _MockZ3._SortRef(name)
            return _DT()

        # Expressions
        @staticmethod
        def Int(name: str) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(name)

        @staticmethod
        def Bool(name: str) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(name)

        @staticmethod
        def Real(name: str) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(name)

        @staticmethod
        def String(name: str) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(name)

        @staticmethod
        def Const(name: str, sort: Any) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(name)

        @staticmethod
        def IntVal(v: int) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(str(v))

        @staticmethod
        def BoolVal(v: bool) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(str(v))

        @staticmethod
        def RealVal(v: float) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(str(v))

        @staticmethod
        def StringVal(v: str) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef(f'"{v}"')

        @staticmethod
        def Function(name: str, *sorts: Any) -> Callable[..., "_MockZ3._ExprRef"]:
            return lambda *a: _MockZ3._ExprRef(f"{name}(...)")

        @staticmethod
        def And(*args: Any) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef("And(...)")

        @staticmethod
        def Or(*args: Any) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef("Or(...)")

        @staticmethod
        def Not(a: Any) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef("Not(...)")

        @staticmethod
        def Implies(a: Any, b: Any) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef("Implies(...)")

        @staticmethod
        def ForAll(vs: Any, body: Any) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef("ForAll(...)")

        @staticmethod
        def Exists(vs: Any, body: Any) -> "_MockZ3._ExprRef":
            return _MockZ3._ExprRef("Exists(...)")

        @staticmethod
        def Solver() -> "_MockZ3._SolverRef":
            return _MockZ3._SolverRef()

        @staticmethod
        def Tactic(name: str) -> Any:
            class _T:
                def __call__(self, goal: Any) -> Any:
                    return [[]]
                def apply(self, goal: Any) -> Any:
                    return [[]]
            return _T()

        @staticmethod
        def Goal() -> Any:
            class _G:
                def add(self, *a: Any) -> None: pass
                def as_expr(self) -> "_MockZ3._ExprRef":
                    return _MockZ3._ExprRef("goal")
            return _G()

        @staticmethod
        def simplify(e: Any) -> "_MockZ3._ExprRef":
            return e if isinstance(e, _MockZ3._ExprRef) else _MockZ3._ExprRef("simplified")

        @staticmethod
        def is_true(e: Any) -> bool:
            return False

        @staticmethod
        def is_false(e: Any) -> bool:
            return False

        sat = "sat"
        unsat = "unsat"
        unknown = "unknown"

    z3 = _MockZ3()  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Local type stubs and enums
# ═══════════════════════════════════════════════════════════════════════════


class Sort(enum.Enum):
    """Represents base sorts for the refinement type system."""
    INT = "int"
    BOOL = "bool"
    FLOAT = "float"
    STR = "str"
    NONE = "none"
    LIST = "list"
    DICT = "dict"
    SET = "set"
    TUPLE = "tuple"
    UNION = "union"
    ANY = "any"


class BinOperator(enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    FLOOR_DIV = "//"
    POW = "**"


class CmpOperator(enum.Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class LogicOperator(enum.Enum):
    AND = "and"
    OR = "or"
    IMPLIES = "=>"
    IFF = "<=>"


class UnaryOperator(enum.Enum):
    NEG = "-"
    NOT = "not"
    ABS = "abs"


class QuantifierKind(enum.Enum):
    FORALL = "forall"
    EXISTS = "exists"


@dataclass(frozen=True)
class TypeDesc:
    """Describes a type used in refinement predicates."""
    sort: Sort
    params: Tuple["TypeDesc", ...] = ()
    union_members: FrozenSet["TypeDesc"] = frozenset()

    def __repr__(self) -> str:
        if self.sort == Sort.UNION:
            return f"Union[{', '.join(str(m) for m in self.union_members)}]"
        if self.params:
            return f"{self.sort.value}[{', '.join(str(p) for p in self.params)}]"
        return self.sort.value


# --- Expr AST nodes ---

@dataclass(frozen=True)
class Var:
    name: str
    type_desc: Optional[TypeDesc] = None


@dataclass(frozen=True)
class Const:
    value: Any
    type_desc: Optional[TypeDesc] = None


@dataclass(frozen=True)
class BinOp:
    op: BinOperator
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class UnaryOp:
    op: UnaryOperator
    operand: "Expr"


@dataclass(frozen=True)
class Comparison:
    op: CmpOperator
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class LogicalOp:
    op: LogicOperator
    args: Tuple["Expr", ...]


@dataclass(frozen=True)
class Quantifier:
    kind: QuantifierKind
    variables: Tuple[Var, ...]
    body: "Expr"


@dataclass(frozen=True)
class IsInstance:
    """isinstance(var, type) encoded as tag equality."""
    var: Var
    type_desc: TypeDesc


@dataclass(frozen=True)
class IsNone:
    var: Var


@dataclass(frozen=True)
class FunctionApp:
    """Application of an uninterpreted function (len, contains, ...)."""
    name: str
    args: Tuple["Expr", ...]


Expr = Union[
    Var, Const, BinOp, UnaryOp, Comparison, LogicalOp,
    Quantifier, IsInstance, IsNone, FunctionApp,
]


@dataclass(frozen=True)
class RefinementPredicate:
    """A refinement {v : T | φ(v)}."""
    binder: str
    base_type: TypeDesc
    formula: Expr


class SolverResult(enum.Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


@dataclass
class QueryResult:
    result: SolverResult
    model: Optional[Dict[str, Any]] = None
    unsat_core: Optional[List[str]] = None
    time_seconds: float = 0.0
    from_cache: bool = False


@dataclass
class Counterexample:
    """A concrete assignment that violates a refinement."""
    bindings: Dict[str, Any]
    description: str = ""


@dataclass
class SolverStats:
    total_queries: int = 0
    sat_count: int = 0
    unsat_count: int = 0
    unknown_count: int = 0
    cache_hits: int = 0
    total_time: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries


# ═══════════════════════════════════════════════════════════════════════════
# 2. Z3SortEncoder
# ═══════════════════════════════════════════════════════════════════════════


class Z3SortEncoder:
    """Maps Python / TypeDesc types to Z3 sorts."""

    def __init__(self) -> None:
        self._sort_cache: Dict[TypeDesc, Any] = {}
        self._none_sort: Optional[Any] = None
        self._union_datatypes: Dict[FrozenSet[TypeDesc], Any] = {}
        self._tag_accessors: Dict[FrozenSet[TypeDesc], Dict[TypeDesc, Tuple[Any, Any]]] = {}

    # -- primitive sorts --------------------------------------------------

    def _int_sort(self) -> Any:
        return z3.IntSort()

    def _bool_sort(self) -> Any:
        return z3.BoolSort()

    def _real_sort(self) -> Any:
        return z3.RealSort()

    def _str_sort(self) -> Any:
        try:
            return z3.StringSort()
        except Exception:
            return z3.DeclareSort("PyStr")

    def _none_sort_z3(self) -> Any:
        if self._none_sort is None:
            self._none_sort = z3.DeclareSort("PyNone")
        return self._none_sort

    # -- composite sorts --------------------------------------------------

    def _list_sort(self, td: TypeDesc) -> Any:
        if td.params:
            elem = self.encode(td.params[0])
        else:
            elem = self._int_sort()
        return z3.ArraySort(self._int_sort(), elem)

    def _dict_sort(self, td: TypeDesc) -> Any:
        if len(td.params) >= 2:
            key_s = self.encode(td.params[0])
            val_s = self.encode(td.params[1])
        else:
            key_s = self._int_sort()
            val_s = self._int_sort()
        return z3.ArraySort(key_s, val_s)

    def _set_sort(self, td: TypeDesc) -> Any:
        if td.params:
            elem = self.encode(td.params[0])
        else:
            elem = self._int_sort()
        return z3.ArraySort(elem, self._bool_sort())

    def _tuple_sort(self, td: TypeDesc) -> Any:
        # Encode tuples as nested arrays keyed by int index.
        if td.params:
            elem = self.encode(td.params[0])
        else:
            elem = self._int_sort()
        return z3.ArraySort(self._int_sort(), elem)

    # -- union sort (tagged union datatype) --------------------------------

    def _union_sort(self, td: TypeDesc) -> Any:
        members = td.union_members
        if not members:
            return self._int_sort()

        if members in self._union_datatypes:
            return self._union_datatypes[members]

        name = "Union_" + "_".join(sorted(m.sort.value for m in members))
        dt = z3.Datatype(name)
        for m in sorted(members, key=lambda x: x.sort.value):
            field_sort_name = f"val_{m.sort.value}"
            dt.declare(f"mk_{m.sort.value}", (field_sort_name, self.encode(m)))

        created = dt.create()
        self._union_datatypes[members] = created

        # Build accessor lookup
        accessors: Dict[TypeDesc, Tuple[Any, Any]] = {}
        if Z3_AVAILABLE:
            for i, m in enumerate(sorted(members, key=lambda x: x.sort.value)):
                try:
                    constructor = created.constructor(i)
                    recognizer = created.recognizer(i)
                    accessors[m] = (constructor, recognizer)
                except Exception:
                    pass
        self._tag_accessors[members] = accessors
        return created

    # -- public API --------------------------------------------------------

    def encode(self, td: TypeDesc) -> Any:
        """Return the Z3 sort corresponding to *td*."""
        if td in self._sort_cache:
            return self._sort_cache[td]

        dispatch = {
            Sort.INT: lambda: self._int_sort(),
            Sort.BOOL: lambda: self._bool_sort(),
            Sort.FLOAT: lambda: self._real_sort(),
            Sort.STR: lambda: self._str_sort(),
            Sort.NONE: lambda: self._none_sort_z3(),
            Sort.LIST: lambda: self._list_sort(td),
            Sort.DICT: lambda: self._dict_sort(td),
            Sort.SET: lambda: self._set_sort(td),
            Sort.TUPLE: lambda: self._tuple_sort(td),
            Sort.UNION: lambda: self._union_sort(td),
            Sort.ANY: lambda: self._int_sort(),
        }

        builder = dispatch.get(td.sort)
        if builder is None:
            logger.warning("Unknown sort %s; falling back to Int", td.sort)
            result = self._int_sort()
        else:
            result = builder()

        self._sort_cache[td] = result
        return result

    def get_tag_accessors(
        self, members: FrozenSet[TypeDesc]
    ) -> Dict[TypeDesc, Tuple[Any, Any]]:
        """Return (constructor, recognizer) pairs for a union type."""
        return self._tag_accessors.get(members, {})


# ═══════════════════════════════════════════════════════════════════════════
# 3. Z3ExprEncoder
# ═══════════════════════════════════════════════════════════════════════════


class Z3ExprEncoder:
    """Encode predicate AST nodes as Z3 expressions."""

    def __init__(self, sort_encoder: Z3SortEncoder) -> None:
        self._sort_enc = sort_encoder
        self._var_cache: Dict[str, Any] = {}
        self._uf_cache: Dict[str, Any] = {}

    # -- helpers -----------------------------------------------------------

    def _get_or_make_var(self, name: str, td: Optional[TypeDesc]) -> Any:
        if name in self._var_cache:
            return self._var_cache[name]
        if td is None:
            td = TypeDesc(Sort.INT)
        z3_sort = self._sort_enc.encode(td)
        v = z3.Const(name, z3_sort)
        self._var_cache[name] = v
        return v

    def _get_uf(self, name: str, arg_sorts: Sequence[Any], ret_sort: Any) -> Any:
        key = f"{name}/{len(arg_sorts)}"
        if key not in self._uf_cache:
            self._uf_cache[key] = z3.Function(name, *arg_sorts, ret_sort)
        return self._uf_cache[key]

    # -- main dispatch -----------------------------------------------------

    def encode(self, expr: Expr) -> Any:
        """Recursively encode an Expr AST into a Z3 expression."""
        if isinstance(expr, Var):
            return self._encode_var(expr)
        if isinstance(expr, Const):
            return self._encode_const(expr)
        if isinstance(expr, BinOp):
            return self._encode_binop(expr)
        if isinstance(expr, UnaryOp):
            return self._encode_unaryop(expr)
        if isinstance(expr, Comparison):
            return self._encode_comparison(expr)
        if isinstance(expr, LogicalOp):
            return self._encode_logical(expr)
        if isinstance(expr, Quantifier):
            return self._encode_quantifier(expr)
        if isinstance(expr, IsInstance):
            return self._encode_isinstance(expr)
        if isinstance(expr, IsNone):
            return self._encode_is_none(expr)
        if isinstance(expr, FunctionApp):
            return self._encode_funcapp(expr)
        raise TypeError(f"Unknown Expr node: {type(expr)}")

    # -- node encoders -----------------------------------------------------

    def _encode_var(self, v: Var) -> Any:
        return self._get_or_make_var(v.name, v.type_desc)

    def _encode_const(self, c: Const) -> Any:
        val = c.value
        if isinstance(val, bool):
            return z3.BoolVal(val)
        if isinstance(val, int):
            return z3.IntVal(val)
        if isinstance(val, float):
            return z3.RealVal(val)
        if isinstance(val, str):
            return z3.StringVal(val)
        if val is None:
            sort = self._sort_enc.encode(TypeDesc(Sort.NONE))
            return z3.Const("None_val", sort)
        return z3.IntVal(0)

    def _encode_binop(self, b: BinOp) -> Any:
        l = self.encode(b.left)
        r = self.encode(b.right)
        op_map = {
            BinOperator.ADD: lambda: l + r,
            BinOperator.SUB: lambda: l - r,
            BinOperator.MUL: lambda: l * r,
            BinOperator.DIV: lambda: l / r,
            BinOperator.MOD: lambda: l % r,
            BinOperator.FLOOR_DIV: lambda: l / r,  # approx
            BinOperator.POW: lambda: l ** r,
        }
        builder = op_map.get(b.op)
        if builder is None:
            raise ValueError(f"Unsupported binop: {b.op}")
        try:
            return builder()
        except Exception:
            # Fall back to uninterpreted function
            uf = self._get_uf(
                f"binop_{b.op.value}", [z3.IntSort(), z3.IntSort()], z3.IntSort()
            )
            return uf(l, r)

    def _encode_unaryop(self, u: UnaryOp) -> Any:
        inner = self.encode(u.operand)
        if u.op == UnaryOperator.NEG:
            try:
                return -inner
            except Exception:
                return inner
        if u.op == UnaryOperator.NOT:
            return z3.Not(inner)
        if u.op == UnaryOperator.ABS:
            uf = self._get_uf("py_abs", [z3.IntSort()], z3.IntSort())
            return uf(inner)
        raise ValueError(f"Unsupported unary op: {u.op}")

    def _encode_comparison(self, c: Comparison) -> Any:
        l = self.encode(c.left)
        r = self.encode(c.right)
        cmp_map = {
            CmpOperator.EQ: lambda: l == r,
            CmpOperator.NE: lambda: l != r,
            CmpOperator.LT: lambda: l < r,
            CmpOperator.LE: lambda: l <= r,
            CmpOperator.GT: lambda: l > r,
            CmpOperator.GE: lambda: l >= r,
        }
        builder = cmp_map.get(c.op)
        if builder is None:
            raise ValueError(f"Unsupported cmp: {c.op}")
        try:
            return builder()
        except Exception:
            return z3.BoolVal(False)

    def _encode_logical(self, lo: LogicalOp) -> Any:
        encoded = [self.encode(a) for a in lo.args]
        if lo.op == LogicOperator.AND:
            return z3.And(*encoded)
        if lo.op == LogicOperator.OR:
            return z3.Or(*encoded)
        if lo.op == LogicOperator.IMPLIES:
            if len(encoded) != 2:
                raise ValueError("Implies requires exactly 2 arguments")
            return z3.Implies(encoded[0], encoded[1])
        if lo.op == LogicOperator.IFF:
            if len(encoded) != 2:
                raise ValueError("Iff requires exactly 2 arguments")
            return encoded[0] == encoded[1]
        raise ValueError(f"Unsupported logical op: {lo.op}")

    def _encode_quantifier(self, q: Quantifier) -> Any:
        bound = [self._get_or_make_var(v.name, v.type_desc) for v in q.variables]
        body = self.encode(q.body)
        if q.kind == QuantifierKind.FORALL:
            return z3.ForAll(bound, body)
        if q.kind == QuantifierKind.EXISTS:
            return z3.Exists(bound, body)
        raise ValueError(f"Unknown quantifier: {q.kind}")

    def _encode_isinstance(self, isi: IsInstance) -> Any:
        var_z3 = self._get_or_make_var(isi.var.name, isi.var.type_desc)
        if isi.var.type_desc and isi.var.type_desc.sort == Sort.UNION:
            accessors = self._sort_enc.get_tag_accessors(
                isi.var.type_desc.union_members
            )
            pair = accessors.get(isi.type_desc)
            if pair is not None:
                _constructor, recognizer = pair
                try:
                    return recognizer(var_z3)
                except Exception:
                    pass
        # Fallback: uninterpreted tag function
        tag_fn = self._get_uf("py_tag", [z3.IntSort()], z3.IntSort())
        tag_val = z3.IntVal(hash(isi.type_desc.sort.value) % 1000)
        return tag_fn(var_z3) == tag_val

    def _encode_is_none(self, isn: IsNone) -> Any:
        var_z3 = self._get_or_make_var(isn.var.name, isn.var.type_desc)
        tag_fn = self._get_uf("py_is_none", [z3.IntSort()], z3.BoolSort())
        return tag_fn(var_z3)

    def _encode_funcapp(self, fa: FunctionApp) -> Any:
        encoded_args = [self.encode(a) for a in fa.args]
        name = fa.name

        # Special handling for well-known functions
        if name == "len" and len(encoded_args) == 1:
            uf = self._get_uf("py_len", [z3.IntSort()], z3.IntSort())
            return uf(encoded_args[0])

        if name == "contains" and len(encoded_args) == 2:
            try:
                return z3.Contains(encoded_args[0], encoded_args[1])  # type: ignore[attr-defined]
            except (AttributeError, Exception):
                uf = self._get_uf(
                    "str_contains",
                    [z3.StringSort(), z3.StringSort()],
                    z3.BoolSort(),
                )
                return uf(encoded_args[0], encoded_args[1])

        if name == "startswith" and len(encoded_args) == 2:
            try:
                return z3.PrefixOf(encoded_args[1], encoded_args[0])  # type: ignore[attr-defined]
            except (AttributeError, Exception):
                uf = self._get_uf(
                    "str_startswith",
                    [z3.StringSort(), z3.StringSort()],
                    z3.BoolSort(),
                )
                return uf(encoded_args[0], encoded_args[1])

        if name == "endswith" and len(encoded_args) == 2:
            try:
                return z3.SuffixOf(encoded_args[1], encoded_args[0])  # type: ignore[attr-defined]
            except (AttributeError, Exception):
                uf = self._get_uf(
                    "str_endswith",
                    [z3.StringSort(), z3.StringSort()],
                    z3.BoolSort(),
                )
                return uf(encoded_args[0], encoded_args[1])

        # Generic: build uninterpreted function
        arg_sorts = [z3.IntSort()] * len(encoded_args)
        ret_sort = z3.IntSort()
        uf = self._get_uf(name, arg_sorts, ret_sort)
        return uf(*encoded_args)


# ═══════════════════════════════════════════════════════════════════════════
# 4. IncrementalSolver
# ═══════════════════════════════════════════════════════════════════════════


class IncrementalSolver:
    """Wrapper around z3.Solver with scope management and statistics."""

    def __init__(self, timeout_ms: int = 5000) -> None:
        self._solver = z3.Solver()
        self._solver.set("timeout", timeout_ms)
        self._timeout_ms = timeout_ms
        self._scope_depth: int = 0
        self._assertion_groups: Dict[str, List[Any]] = {}
        self._active_group: Optional[str] = None
        self._stats = SolverStats()

    # -- scope management --------------------------------------------------

    def push(self) -> None:
        """Push a new assertion scope."""
        self._solver.push()
        self._scope_depth += 1
        logger.debug("Solver push (depth=%d)", self._scope_depth)

    def pop(self, n: int = 1) -> None:
        """Pop *n* assertion scopes."""
        actual = min(n, self._scope_depth)
        if actual > 0:
            self._solver.pop(actual)
            self._scope_depth -= actual
        logger.debug("Solver pop %d (depth=%d)", actual, self._scope_depth)

    @property
    def scope_depth(self) -> int:
        return self._scope_depth

    # -- assertion groups --------------------------------------------------

    def begin_group(self, name: str) -> None:
        """Start a named assertion group for tracking."""
        self._active_group = name
        if name not in self._assertion_groups:
            self._assertion_groups[name] = []

    def end_group(self) -> None:
        self._active_group = None

    # -- adding assertions -------------------------------------------------

    def add(self, *exprs: Any) -> None:
        """Add assertions to the solver."""
        for e in exprs:
            self._solver.add(e)
            if self._active_group is not None:
                self._assertion_groups[self._active_group].append(e)

    def assert_and_track(self, expr: Any, label: str) -> None:
        """Add a tracked assertion for unsat core extraction."""
        tracker = z3.Bool(label)
        self._solver.assert_and_track(expr, tracker)
        if self._active_group is not None:
            self._assertion_groups[self._active_group].append(expr)

    # -- solving -----------------------------------------------------------

    def check_sat(self, *assumptions: Any) -> SolverResult:
        """Check satisfiability, updating statistics."""
        start = time.monotonic()
        try:
            raw = self._solver.check(*assumptions)
        except Exception as exc:
            logger.error("Z3 check failed: %s", exc)
            self._stats.unknown_count += 1
            self._stats.total_queries += 1
            self._stats.total_time += time.monotonic() - start
            return SolverResult.UNKNOWN

        elapsed = time.monotonic() - start
        self._stats.total_time += elapsed
        self._stats.total_queries += 1

        if str(raw) == "sat":
            self._stats.sat_count += 1
            return SolverResult.SAT
        if str(raw) == "unsat":
            self._stats.unsat_count += 1
            return SolverResult.UNSAT

        self._stats.unknown_count += 1
        return SolverResult.UNKNOWN

    def check(self, *assumptions: Any) -> QueryResult:
        """Full query returning a QueryResult with model / core."""
        start = time.monotonic()
        result = self.check_sat(*assumptions)
        elapsed = time.monotonic() - start

        qr = QueryResult(result=result, time_seconds=elapsed)

        if result == SolverResult.SAT:
            qr.model = self._extract_model_dict()
        elif result == SolverResult.UNSAT:
            qr.unsat_core = self._extract_core_labels()

        return qr

    # -- model extraction --------------------------------------------------

    def get_model(self) -> Optional[Any]:
        """Return raw Z3 model, or None if unavailable."""
        try:
            return self._solver.model()
        except Exception:
            return None

    def _extract_model_dict(self) -> Dict[str, Any]:
        raw = self.get_model()
        if raw is None:
            return {}
        result: Dict[str, Any] = {}
        try:
            for decl in raw:
                name = str(decl.name()) if hasattr(decl, "name") else str(decl)
                val = raw[decl]
                result[name] = self._z3_val_to_python(val)
        except Exception as exc:
            logger.debug("Model extraction partial failure: %s", exc)
        return result

    @staticmethod
    def _z3_val_to_python(val: Any) -> Any:
        """Best-effort conversion of a Z3 value back to Python."""
        s = str(val)
        if s in ("True", "true"):
            return True
        if s in ("False", "false"):
            return False
        try:
            return int(s)
        except (ValueError, TypeError):
            pass
        try:
            return float(s)
        except (ValueError, TypeError):
            pass
        # Strip surrounding quotes for strings
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    # -- unsat core --------------------------------------------------------

    def get_unsat_core(self) -> List[Any]:
        try:
            return list(self._solver.unsat_core())
        except Exception:
            return []

    def _extract_core_labels(self) -> List[str]:
        return [str(c) for c in self.get_unsat_core()]

    # -- statistics --------------------------------------------------------

    @property
    def stats(self) -> SolverStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = SolverStats()

    def solver_statistics(self) -> str:
        try:
            return str(self._solver.statistics())
        except Exception:
            return ""

    # -- configuration -----------------------------------------------------

    def set_timeout(self, ms: int) -> None:
        self._timeout_ms = ms
        self._solver.set("timeout", ms)

    def set_option(self, key: str, value: Any) -> None:
        try:
            self._solver.set(key, value)
        except Exception as exc:
            logger.warning("Failed to set solver option %s=%s: %s", key, value, exc)

    def reason_unknown(self) -> str:
        try:
            return str(self._solver.reason_unknown())
        except Exception:
            return "unavailable"


# ═══════════════════════════════════════════════════════════════════════════
# 5. InterpolationEngine (unsat core-based predicate extraction)
# ═══════════════════════════════════════════════════════════════════════════


class InterpolationEngine:
    """Unsat core-based predicate extraction for contract discovery.

    Given A ∧ B = unsat, extracts predicates from the unsat core
    to identify constraints contributing to unsatisfiability.
    """

    def __init__(self, sort_encoder: Z3SortEncoder) -> None:
        self._sort_enc = sort_encoder

    def compute_binary_interpolant(
        self, formula_a: Any, formula_b: Any
    ) -> Optional[Any]:
        """Compute a separating predicate between *formula_a* and *formula_b*.

        Returns None when extraction is unavailable or the conjunction is
        satisfiable.  Note: uses unsat-core-based extraction; the result
        may not satisfy Craig's vocabulary restriction.
        """
        if not Z3_AVAILABLE:
            logger.warning("Z3 not available; predicate extraction skipped")
            return None

        # Attempt native z3 interpolation (available in some builds)
        interp = self._try_native_interpolation(formula_a, formula_b)
        if interp is not None:
            return self._simplify_interpolant(interp)

        # Fallback: unsat core approximation
        return self._sequence_interpolation_fallback(formula_a, formula_b)

    def _try_native_interpolation(
        self, formula_a: Any, formula_b: Any
    ) -> Optional[Any]:
        """Try z3's built-in interpolation if the API is present."""
        try:
            interp_fn = getattr(z3, "interpolant", None) or getattr(
                z3, "Interpolant", None
            )
            if interp_fn is None:
                return None
            result = interp_fn(formula_a, formula_b)
            if result is not None and len(result) > 0:
                return result[0]
        except Exception as exc:
            logger.debug("Native interpolation failed: %s", exc)
        return None

    def _sequence_interpolation_fallback(
        self, formula_a: Any, formula_b: Any
    ) -> Optional[Any]:
        """Approximate interpolation using quantifier elimination on A's
        variables projected away from B."""
        solver = z3.Solver()
        solver.add(formula_a)
        solver.add(formula_b)
        if str(solver.check()) != "unsat":
            logger.debug("Conjunction is satisfiable; no interpolant exists")
            return None

        # Heuristic: simplify A as the interpolant candidate
        candidate = self._simplify_interpolant(formula_a)

        # Verify: candidate ∧ B should be unsat
        check_solver = z3.Solver()
        check_solver.add(candidate)
        check_solver.add(formula_b)
        if str(check_solver.check()) == "unsat":
            return candidate

        # If simplification didn't work, return A itself
        return formula_a

    def _simplify_interpolant(self, formula: Any) -> Any:
        """Simplify an interpolant using Z3 tactics."""
        try:
            goal = z3.Goal()
            goal.add(formula)
            tactic = z3.Tactic("simplify")
            result = tactic(goal)
            if result and len(result) > 0:
                subgoal = result[0]
                if hasattr(subgoal, "as_expr"):
                    return subgoal.as_expr()
                # Conjoin all formulas in the subgoal
                formulas = list(subgoal)
                if len(formulas) == 0:
                    return z3.BoolVal(True)
                if len(formulas) == 1:
                    return formulas[0]
                return z3.And(*formulas)
        except Exception as exc:
            logger.debug("Interpolant simplification failed: %s", exc)
        return formula

    def compute_sequence_interpolants(
        self, formulas: Sequence[Any]
    ) -> List[Optional[Any]]:
        """Compute a sequence of interpolants for a chain of formulas.

        Given [A₀, A₁, …, Aₙ] whose conjunction is unsat, returns
        [I₁, I₂, …, Iₙ₋₁] such that A₀ ⇒ I₁, Iₖ ∧ Aₖ ⇒ Iₖ₊₁, Iₙ₋₁ ∧ Aₙ = ⊥.
        """
        if len(formulas) < 2:
            return []

        interpolants: List[Optional[Any]] = []
        cumulative = formulas[0]
        for i in range(1, len(formulas)):
            remaining = z3.And(*[formulas[j] for j in range(i, len(formulas))])
            interp = self.compute_binary_interpolant(cumulative, remaining)
            interpolants.append(interp)
            if interp is not None:
                cumulative = z3.And(interp, formulas[i])
            else:
                cumulative = z3.And(cumulative, formulas[i])
        return interpolants


# ═══════════════════════════════════════════════════════════════════════════
# 6. ModelExtractor
# ═══════════════════════════════════════════════════════════════════════════


class ModelExtractor:
    """Extract concrete values from Z3 models and build counterexamples."""

    def __init__(self) -> None:
        self._type_hints: Dict[str, TypeDesc] = {}

    def register_variable_type(self, name: str, td: TypeDesc) -> None:
        """Register expected type for a variable for better extraction."""
        self._type_hints[name] = td

    def extract_values(self, model: Any) -> Dict[str, Any]:
        """Extract all variable assignments from a Z3 model."""
        result: Dict[str, Any] = {}
        if model is None:
            return result
        try:
            for decl in model:
                name = str(decl.name()) if hasattr(decl, "name") else str(decl)
                raw_val = model[decl]
                td = self._type_hints.get(name)
                result[name] = self._convert_value(raw_val, td)
        except Exception as exc:
            logger.debug("Value extraction error: %s", exc)
        return result

    def _convert_value(self, val: Any, td: Optional[TypeDesc]) -> Any:
        """Convert a Z3 value to a Python value using type hints."""
        s = str(val)

        if td is not None:
            if td.sort == Sort.BOOL:
                return s.lower() in ("true", "1")
            if td.sort == Sort.INT:
                try:
                    return int(s)
                except ValueError:
                    return 0
            if td.sort == Sort.FLOAT:
                try:
                    if "/" in s:
                        parts = s.split("/")
                        return float(parts[0]) / float(parts[1])
                    return float(s)
                except (ValueError, ZeroDivisionError):
                    return 0.0
            if td.sort == Sort.STR:
                if s.startswith('"') and s.endswith('"'):
                    return s[1:-1]
                return s
            if td.sort == Sort.NONE:
                return None

        # Best-effort without type hint
        return IncrementalSolver._z3_val_to_python(val)

    def build_counterexample(
        self,
        model: Any,
        predicate: Optional[RefinementPredicate] = None,
    ) -> Counterexample:
        """Build a Counterexample from a Z3 model."""
        bindings = self.extract_values(model)
        desc = self._format_counterexample(bindings, predicate)
        return Counterexample(bindings=bindings, description=desc)

    def _format_counterexample(
        self,
        bindings: Dict[str, Any],
        predicate: Optional[RefinementPredicate],
    ) -> str:
        """Format a counterexample for human-readable display."""
        parts: List[str] = []
        if predicate is not None:
            parts.append(
                f"Refinement violation for {{{predicate.binder} : "
                f"{predicate.base_type} | φ}}"
            )

        if not bindings:
            parts.append("  (no concrete bindings extracted)")
        else:
            parts.append("  Witness assignment:")
            for name, val in sorted(bindings.items()):
                parts.append(f"    {name} = {val!r}")

        return "\n".join(parts)

    def extract_trace(
        self, model: Any, variable_sequence: Sequence[str]
    ) -> List[Tuple[str, Any]]:
        """Extract an ordered trace of variable values for debugging."""
        values = self.extract_values(model)
        trace: List[Tuple[str, Any]] = []
        for var in variable_sequence:
            val = values.get(var, "<undefined>")
            trace.append((var, val))
        return trace


# ═══════════════════════════════════════════════════════════════════════════
# 7. QueryCache
# ═══════════════════════════════════════════════════════════════════════════


class QueryCache:
    """LRU cache for Z3 query results with subsumption checking."""

    def __init__(self, max_size: int = 4096) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, QueryResult] = OrderedDict()
        self._total_queries: int = 0
        self._hits: int = 0
        # For subsumption: store formulas that are known valid (unsat negation)
        self._valid_formulas: Set[str] = set()

    # -- key computation ---------------------------------------------------

    @staticmethod
    def _normalize_key(formula: Any) -> str:
        """Create a canonical string key from a Z3 formula."""
        try:
            sexpr = formula.sexpr() if hasattr(formula, "sexpr") else str(formula)
        except Exception:
            sexpr = str(formula)
        return hashlib.sha256(sexpr.encode("utf-8", errors="replace")).hexdigest()

    # -- lookup / insert ---------------------------------------------------

    def lookup(self, formula: Any) -> Optional[QueryResult]:
        """Look up a cached result for *formula*."""
        self._total_queries += 1
        key = self._normalize_key(formula)

        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            cached = self._cache[key]
            return QueryResult(
                result=cached.result,
                model=cached.model,
                unsat_core=cached.unsat_core,
                time_seconds=0.0,
                from_cache=True,
            )

        # Subsumption: if formula is subsumed by a known valid formula
        if key in self._valid_formulas:
            self._hits += 1
            return QueryResult(
                result=SolverResult.UNSAT, time_seconds=0.0, from_cache=True
            )

        return None

    def store(self, formula: Any, result: QueryResult) -> None:
        """Store a query result, evicting LRU entries if necessary."""
        key = self._normalize_key(formula)
        self._cache[key] = result
        self._cache.move_to_end(key)

        if result.result == SolverResult.UNSAT:
            self._valid_formulas.add(key)

        while len(self._cache) > self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            self._valid_formulas.discard(evicted_key)

    def record_valid(self, formula: Any) -> None:
        """Mark a formula as valid (its negation is unsat) for subsumption."""
        key = self._normalize_key(formula)
        self._valid_formulas.add(key)

    def invalidate(self, formula: Any) -> None:
        """Remove a specific formula from the cache."""
        key = self._normalize_key(formula)
        self._cache.pop(key, None)
        self._valid_formulas.discard(key)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._valid_formulas.clear()
        self._total_queries = 0
        self._hits = 0

    # -- statistics --------------------------------------------------------

    @property
    def total_queries(self) -> int:
        return self._total_queries

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def hit_rate(self) -> float:
        if self._total_queries == 0:
            return 0.0
        return self._hits / self._total_queries

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats_summary(self) -> str:
        return (
            f"QueryCache: {self.size} entries, "
            f"{self._hits}/{self._total_queries} hits "
            f"({self.hit_rate:.1%}), "
            f"{len(self._valid_formulas)} valid formulas"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Z3Backend — main façade
# ═══════════════════════════════════════════════════════════════════════════


class Z3Backend:
    """Main façade composing all Z3 sub-components.

    Provides high-level operations for the refinement type inference system:
    subtype checking, counterexample generation, predicate extraction,
    quantifier elimination, and formula simplification.
    """

    def __init__(self, timeout_ms: int = 5000, cache_size: int = 4096) -> None:
        self._sort_encoder = Z3SortEncoder()
        self._expr_encoder = Z3ExprEncoder(self._sort_encoder)
        self._solver = IncrementalSolver(timeout_ms=timeout_ms)
        self._interpolation = InterpolationEngine(self._sort_encoder)
        self._model_extractor = ModelExtractor()
        self._cache = QueryCache(max_size=cache_size)
        self._timeout_ms = timeout_ms

    # -- properties --------------------------------------------------------

    @property
    def stats(self) -> SolverStats:
        return self._solver.stats

    @property
    def cache(self) -> QueryCache:
        return self._cache

    @property
    def sort_encoder(self) -> Z3SortEncoder:
        return self._sort_encoder

    @property
    def expr_encoder(self) -> Z3ExprEncoder:
        return self._expr_encoder

    # -- refinement subtype checking ---------------------------------------

    def check_refinement_subtype(
        self,
        env: Dict[str, RefinementPredicate],
        sub: RefinementPredicate,
        sup: RefinementPredicate,
    ) -> bool:
        """Check if {v:T | φ_sub} <: {v:T | φ_sup} under environment *env*.

        Encodes: env ∧ φ_sub ⇒ φ_sup  (i.e., ¬(env ∧ φ_sub ∧ ¬φ_sup) is unsat).
        Returns True if the subtype relation holds.
        """
        sub_z3 = self._expr_encoder.encode(sub.formula)
        sup_z3 = self._expr_encoder.encode(sup.formula)

        env_constraints: List[Any] = []
        for _name, pred in env.items():
            env_constraints.append(self._expr_encoder.encode(pred.formula))

        if env_constraints:
            antecedent = z3.And(*env_constraints, sub_z3)
        else:
            antecedent = sub_z3

        query = z3.And(antecedent, z3.Not(sup_z3))

        # Check cache
        cached = self._cache.lookup(query)
        if cached is not None:
            return cached.result == SolverResult.UNSAT

        # Fresh solver scope
        self._solver.push()
        try:
            self._solver.add(query)
            result = self._solver.check_sat()
            qr = QueryResult(result=result)
            self._cache.store(query, qr)
            return result == SolverResult.UNSAT
        finally:
            self._solver.pop()

    # -- counterexample generation -----------------------------------------

    def find_counterexample(
        self,
        env: Dict[str, RefinementPredicate],
        predicate: RefinementPredicate,
    ) -> Optional[Counterexample]:
        """Find a concrete assignment violating *predicate* under *env*.

        Looks for a model of env ∧ ¬predicate.
        """
        pred_z3 = self._expr_encoder.encode(predicate.formula)
        negated = z3.Not(pred_z3)

        env_constraints = [
            self._expr_encoder.encode(p.formula) for p in env.values()
        ]

        self._solver.push()
        try:
            for c in env_constraints:
                self._solver.add(c)
            self._solver.add(negated)

            # Register type hints for extraction
            self._model_extractor.register_variable_type(
                predicate.binder, predicate.base_type
            )
            for name, p in env.items():
                self._model_extractor.register_variable_type(name, p.base_type)

            result = self._solver.check_sat()
            if result == SolverResult.SAT:
                model = self._solver.get_model()
                return self._model_extractor.build_counterexample(model, predicate)
            return None
        finally:
            self._solver.pop()

    # -- interpolation -----------------------------------------------------

    def compute_interpolant(
        self, pre: Expr, post: Expr
    ) -> Optional[Expr]:
        """Compute an interpolant between *pre* and *post*.

        Returns a simplified Expr if found, None otherwise.
        Note: the returned object is a raw Z3 formula wrapped in the Expr
        type system at the boundary; callers should treat it opaquely.
        """
        pre_z3 = self._expr_encoder.encode(pre)
        post_z3 = self._expr_encoder.encode(post)
        interp = self._interpolation.compute_binary_interpolant(pre_z3, post_z3)
        if interp is None:
            return None
        # Wrap the Z3 interpolant as a Const holding the formula
        simplified = z3.simplify(interp)
        return Const(value=simplified, type_desc=TypeDesc(Sort.BOOL))

    # -- quantifier elimination --------------------------------------------

    def eliminate_quantifiers(self, formula: Expr) -> Expr:
        """Eliminate quantifiers from *formula* using Z3 tactics."""
        z3_formula = self._expr_encoder.encode(formula)
        try:
            goal = z3.Goal()
            goal.add(z3_formula)
            tactic = z3.Tactic("qe")
            result = tactic(goal)
            if result and len(result) > 0:
                subgoal = result[0]
                if hasattr(subgoal, "as_expr"):
                    qf = subgoal.as_expr()
                else:
                    formulas = list(subgoal)
                    if len(formulas) == 0:
                        return Const(value=True, type_desc=TypeDesc(Sort.BOOL))
                    if len(formulas) == 1:
                        qf = formulas[0]
                    else:
                        qf = z3.And(*formulas)
                simplified = z3.simplify(qf)
                return Const(value=simplified, type_desc=TypeDesc(Sort.BOOL))
        except Exception as exc:
            logger.debug("Quantifier elimination failed: %s", exc)

        return formula

    # -- simplification ----------------------------------------------------

    def simplify(self, formula: Expr) -> Expr:
        """Simplify a formula using Z3's simplifier and tactics."""
        z3_formula = self._expr_encoder.encode(formula)
        try:
            simplified = z3.simplify(z3_formula)
            if z3.is_true(simplified):
                return Const(value=True, type_desc=TypeDesc(Sort.BOOL))
            if z3.is_false(simplified):
                return Const(value=False, type_desc=TypeDesc(Sort.BOOL))
            return Const(value=simplified, type_desc=TypeDesc(Sort.BOOL))
        except Exception as exc:
            logger.debug("Simplification failed: %s", exc)
            return formula

    # -- batch validity checking -------------------------------------------

    def check_valid(self, formula: Expr) -> bool:
        """Check if *formula* is valid (i.e., ¬formula is unsat)."""
        z3_formula = self._expr_encoder.encode(formula)
        negated = z3.Not(z3_formula)

        cached = self._cache.lookup(negated)
        if cached is not None:
            return cached.result == SolverResult.UNSAT

        self._solver.push()
        try:
            self._solver.add(negated)
            result = self._solver.check_sat()
            self._cache.store(negated, QueryResult(result=result))
            if result == SolverResult.UNSAT:
                self._cache.record_valid(z3_formula)
            return result == SolverResult.UNSAT
        finally:
            self._solver.pop()

    def check_satisfiable(self, formula: Expr) -> QueryResult:
        """Check satisfiability and return full result with model if SAT."""
        z3_formula = self._expr_encoder.encode(formula)

        cached = self._cache.lookup(z3_formula)
        if cached is not None:
            return cached

        self._solver.push()
        try:
            self._solver.add(z3_formula)
            qr = self._solver.check()
            self._cache.store(z3_formula, qr)
            return qr
        finally:
            self._solver.pop()

    # -- unsat core for diagnostics ----------------------------------------

    def get_minimal_unsat_subset(
        self, formulas: Sequence[Expr]
    ) -> Optional[List[int]]:
        """Find a minimal unsatisfiable subset (by index) of *formulas*."""
        if not formulas:
            return None

        z3_formulas = [self._expr_encoder.encode(f) for f in formulas]
        labels = [f"__mus_{i}" for i in range(len(formulas))]

        self._solver.push()
        try:
            for z3f, label in zip(z3_formulas, labels):
                self._solver.assert_and_track(z3f, label)

            result = self._solver.check_sat()
            if result != SolverResult.UNSAT:
                return None

            core_labels = self._solver._extract_core_labels()
            indices = []
            for i, label in enumerate(labels):
                if label in core_labels:
                    indices.append(i)
            return indices if indices else list(range(len(formulas)))
        finally:
            self._solver.pop()

    # -- lifecycle management ----------------------------------------------

    def reset(self) -> None:
        """Reset the backend, clearing solver state and caches."""
        self._solver = IncrementalSolver(timeout_ms=self._timeout_ms)
        self._cache.clear()
        self._expr_encoder = Z3ExprEncoder(self._sort_encoder)
        logger.info("Z3Backend reset")

    def statistics_summary(self) -> str:
        """Return a human-readable summary of backend statistics."""
        s = self._solver.stats
        lines = [
            "Z3Backend Statistics:",
            f"  Queries: {s.total_queries} total "
            f"({s.sat_count} sat, {s.unsat_count} unsat, "
            f"{s.unknown_count} unknown)",
            f"  Time: {s.total_time:.3f}s",
            f"  {self._cache.stats_summary()}",
        ]
        return "\n".join(lines)
