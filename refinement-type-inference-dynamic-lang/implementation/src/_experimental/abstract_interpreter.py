"""Abstract interpretation framework for Python programs.

Implements multiple abstract domains (interval, sign, type, nullness) with
widening/narrowing operators, forward and backward analysis, and alarm reporting
for potential runtime errors.
"""
from __future__ import annotations
import ast, copy, textwrap, math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import numpy as np

NEG_INF, POS_INF = float("-inf"), float("inf")
WIDENING_THRESHOLDS: List[float] = [NEG_INF, -1000, -100, -10, 0, 10, 100, 1000, POS_INF]

# ---------------------------------------------------------------------------
# Interval Domain
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    lo: float; hi: float
    @staticmethod
    def bottom() -> Interval: return Interval(POS_INF, NEG_INF)
    @staticmethod
    def top() -> Interval: return Interval(NEG_INF, POS_INF)
    @staticmethod
    def const(v: float) -> Interval: return Interval(v, v)
    def is_bottom(self) -> bool: return self.lo > self.hi
    def is_top(self) -> bool: return self.lo == NEG_INF and self.hi == POS_INF
    def contains(self, v: float) -> bool: return not self.is_bottom() and self.lo <= v <= self.hi
    def __le__(self, other: Interval) -> bool:
        if self.is_bottom(): return True
        if other.is_bottom(): return False
        return other.lo <= self.lo and self.hi <= other.hi
    def add(self, o: Interval) -> Interval:
        if self.is_bottom() or o.is_bottom(): return Interval.bottom()
        return Interval(self.lo + o.lo, self.hi + o.hi)
    def sub(self, o: Interval) -> Interval:
        if self.is_bottom() or o.is_bottom(): return Interval.bottom()
        return Interval(self.lo - o.hi, self.hi - o.lo)
    def mul(self, o: Interval) -> Interval:
        if self.is_bottom() or o.is_bottom(): return Interval.bottom()
        ps = [self.lo*o.lo, self.lo*o.hi, self.hi*o.lo, self.hi*o.hi]
        f = [p for p in ps if not math.isnan(p)]
        return Interval(min(f), max(f)) if f else Interval.top()
    def div(self, o: Interval) -> Interval:
        if self.is_bottom() or o.is_bottom(): return Interval.bottom()
        if o.contains(0): return Interval.top()
        cs = [a/b for a in (self.lo,self.hi) for b in (o.lo,o.hi) if b != 0]
        return Interval(math.floor(min(cs)), math.ceil(max(cs))) if cs else Interval.top()
    def join(self, o: Interval) -> Interval:
        if self.is_bottom(): return o
        if o.is_bottom(): return self
        return Interval(min(self.lo, o.lo), max(self.hi, o.hi))
    def meet(self, o: Interval) -> Interval:
        if self.is_bottom() or o.is_bottom(): return Interval.bottom()
        lo, hi = max(self.lo, o.lo), min(self.hi, o.hi)
        return Interval.bottom() if lo > hi else Interval(lo, hi)
    def widen(self, o: Interval) -> Interval:
        if self.is_bottom(): return o
        if o.is_bottom(): return self
        new_lo = self.lo
        if o.lo < self.lo:
            c = [t for t in WIDENING_THRESHOLDS if t <= o.lo]
            new_lo = max(c) if c else NEG_INF
        new_hi = self.hi
        if o.hi > self.hi:
            c = [t for t in WIDENING_THRESHOLDS if t >= o.hi]
            new_hi = min(c) if c else POS_INF
        return Interval(new_lo, new_hi)
    def narrow(self, o: Interval) -> Interval:
        if self.is_bottom() or o.is_bottom(): return Interval.bottom()
        new_lo = o.lo if self.lo == NEG_INF else self.lo
        new_hi = o.hi if self.hi == POS_INF else self.hi
        return Interval(new_lo, new_hi)

# ---------------------------------------------------------------------------
# Sign Domain
# ---------------------------------------------------------------------------
class Sign(Enum):
    BOTTOM = auto(); POSITIVE = auto(); NEGATIVE = auto(); ZERO = auto()
    NON_NEG = auto(); NON_POS = auto(); TOP = auto()
    def is_bottom(self) -> bool: return self == Sign.BOTTOM
    def is_top(self) -> bool: return self == Sign.TOP
    @staticmethod
    def _as_set(s: Sign) -> frozenset:
        return {Sign.BOTTOM: frozenset(), Sign.POSITIVE: frozenset({"p"}),
                Sign.NEGATIVE: frozenset({"n"}), Sign.ZERO: frozenset({"z"}),
                Sign.NON_NEG: frozenset({"z","p"}), Sign.NON_POS: frozenset({"z","n"}),
                Sign.TOP: frozenset({"n","z","p"})}[s]
    @staticmethod
    def _from_set(fs: frozenset) -> Sign:
        return {frozenset(): Sign.BOTTOM, frozenset({"p"}): Sign.POSITIVE,
                frozenset({"n"}): Sign.NEGATIVE, frozenset({"z"}): Sign.ZERO,
                frozenset({"z","p"}): Sign.NON_NEG, frozenset({"z","n"}): Sign.NON_POS,
                frozenset({"n","z","p"}): Sign.TOP}.get(fs, Sign.TOP)
    def join(self, o: Sign) -> Sign: return Sign._from_set(Sign._as_set(self)|Sign._as_set(o))
    def meet(self, o: Sign) -> Sign: return Sign._from_set(Sign._as_set(self)&Sign._as_set(o))
    def widen(self, o: Sign) -> Sign: return self.join(o)
    def narrow(self, o: Sign) -> Sign: return self.meet(o)
    def negate(self) -> Sign:
        return {Sign.BOTTOM: Sign.BOTTOM, Sign.POSITIVE: Sign.NEGATIVE,
                Sign.NEGATIVE: Sign.POSITIVE, Sign.ZERO: Sign.ZERO,
                Sign.NON_NEG: Sign.NON_POS, Sign.NON_POS: Sign.NON_NEG,
                Sign.TOP: Sign.TOP}[self]
    def add(self, o: Sign) -> Sign:
        if self.is_bottom() or o.is_bottom(): return Sign.BOTTOM
        if self == Sign.ZERO: return o
        if o == Sign.ZERO: return self
        if self == Sign.POSITIVE and o == Sign.POSITIVE: return Sign.POSITIVE
        if self == Sign.NEGATIVE and o == Sign.NEGATIVE: return Sign.NEGATIVE
        return Sign.TOP
    def sub(self, o: Sign) -> Sign: return self.add(o.negate())
    def mul(self, o: Sign) -> Sign:
        if self.is_bottom() or o.is_bottom(): return Sign.BOTTOM
        if self == Sign.ZERO or o == Sign.ZERO: return Sign.ZERO
        if self == Sign.POSITIVE: return o
        if o == Sign.POSITIVE: return self
        if self == Sign.NEGATIVE and o == Sign.NEGATIVE: return Sign.POSITIVE
        if self == Sign.NEGATIVE: return o.negate()
        if o == Sign.NEGATIVE: return self.negate()
        return Sign.TOP
    def div(self, o: Sign) -> Sign:
        if self.is_bottom() or o.is_bottom(): return Sign.BOTTOM
        if o == Sign.ZERO: return Sign.BOTTOM
        if self == Sign.ZERO: return Sign.ZERO
        return Sign.TOP

# ---------------------------------------------------------------------------
# Type Domain
# ---------------------------------------------------------------------------
ALL_TYPES: FrozenSet[str] = frozenset(
    {"int","float","str","list","dict","set","bool","NoneType","tuple","bytes"})

@dataclass(frozen=True)
class TypeDomain:
    types: FrozenSet[str]
    @staticmethod
    def bottom() -> TypeDomain: return TypeDomain(frozenset())
    @staticmethod
    def top() -> TypeDomain: return TypeDomain(ALL_TYPES)
    @staticmethod
    def single(t: str) -> TypeDomain: return TypeDomain(frozenset({t}))
    def is_bottom(self) -> bool: return len(self.types) == 0
    def is_top(self) -> bool: return self.types == ALL_TYPES
    def join(self, o: TypeDomain) -> TypeDomain: return TypeDomain(self.types | o.types)
    def meet(self, o: TypeDomain) -> TypeDomain: return TypeDomain(self.types & o.types)
    def widen(self, o: TypeDomain) -> TypeDomain: return self.join(o)
    def narrow(self, o: TypeDomain) -> TypeDomain: return self.meet(o)
    def __contains__(self, t: str) -> bool: return t in self.types

# ---------------------------------------------------------------------------
# Null Domain
# ---------------------------------------------------------------------------
class NullDomain(Enum):
    BOTTOM = auto(); NULL = auto(); NOT_NULL = auto(); MAYBE_NULL = auto()
    def is_bottom(self) -> bool: return self == NullDomain.BOTTOM
    def join(self, o: NullDomain) -> NullDomain:
        if self == NullDomain.BOTTOM: return o
        if o == NullDomain.BOTTOM: return self
        return self if self == o else NullDomain.MAYBE_NULL
    def meet(self, o: NullDomain) -> NullDomain:
        if self == NullDomain.BOTTOM or o == NullDomain.BOTTOM: return NullDomain.BOTTOM
        if self == NullDomain.MAYBE_NULL: return o
        if o == NullDomain.MAYBE_NULL: return self
        return self if self == o else NullDomain.BOTTOM
    def widen(self, o: NullDomain) -> NullDomain: return self.join(o)
    def narrow(self, o: NullDomain) -> NullDomain: return self.meet(o)

# ---------------------------------------------------------------------------
# Domain Product
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainProduct:
    interval: Interval; sign: Sign; type_dom: TypeDomain; null_dom: NullDomain
    @staticmethod
    def bottom() -> DomainProduct:
        return DomainProduct(Interval.bottom(), Sign.BOTTOM, TypeDomain.bottom(), NullDomain.BOTTOM)
    @staticmethod
    def top() -> DomainProduct:
        return DomainProduct(Interval.top(), Sign.TOP, TypeDomain.top(), NullDomain.MAYBE_NULL)
    def is_bottom(self) -> bool:
        return self.interval.is_bottom() or self.sign.is_bottom() or self.type_dom.is_bottom() or self.null_dom.is_bottom()
    def join(self, o: DomainProduct) -> DomainProduct:
        return DomainProduct(self.interval.join(o.interval), self.sign.join(o.sign),
                             self.type_dom.join(o.type_dom), self.null_dom.join(o.null_dom))
    def meet(self, o: DomainProduct) -> DomainProduct:
        return DomainProduct(self.interval.meet(o.interval), self.sign.meet(o.sign),
                             self.type_dom.meet(o.type_dom), self.null_dom.meet(o.null_dom))
    def widen(self, o: DomainProduct) -> DomainProduct:
        return DomainProduct(self.interval.widen(o.interval), self.sign.widen(o.sign),
                             self.type_dom.widen(o.type_dom), self.null_dom.widen(o.null_dom))
    def narrow(self, o: DomainProduct) -> DomainProduct:
        return DomainProduct(self.interval.narrow(o.interval), self.sign.narrow(o.sign),
                             self.type_dom.narrow(o.type_dom), self.null_dom.narrow(o.null_dom))

# ---------------------------------------------------------------------------
# Abstract State
# ---------------------------------------------------------------------------
class AbstractState:
    def __init__(self, mapping: Optional[Dict[str, DomainProduct]] = None, _is_bottom: bool = False):
        self._map: Dict[str, DomainProduct] = dict(mapping) if mapping else {}
        self._bottom = _is_bottom
    @staticmethod
    def bottom_state() -> AbstractState: return AbstractState(_is_bottom=True)
    @staticmethod
    def top_state() -> AbstractState: return AbstractState()
    def copy(self) -> AbstractState: return AbstractState(dict(self._map), self._bottom)
    def is_bottom(self) -> bool:
        return self._bottom or any(v.is_bottom() for v in self._map.values())
    def is_top(self) -> bool: return not self._bottom and len(self._map) == 0
    def get(self, var: str) -> DomainProduct:
        return DomainProduct.bottom() if self._bottom else self._map.get(var, DomainProduct.top())
    def set(self, var: str, val: DomainProduct) -> AbstractState:
        s = self.copy(); s._map[var] = val
        if val.is_bottom(): s._bottom = True
        return s
    def variables(self) -> Set[str]: return set(self._map.keys())
    def join(self, o: AbstractState) -> AbstractState:
        if self.is_bottom(): return o.copy()
        if o.is_bottom(): return self.copy()
        vs = self.variables() | o.variables()
        return AbstractState({v: self.get(v).join(o.get(v)) for v in vs})
    def meet(self, o: AbstractState) -> AbstractState:
        if self.is_bottom() or o.is_bottom(): return AbstractState.bottom_state()
        vs = self.variables() | o.variables()
        return AbstractState({v: self.get(v).meet(o.get(v)) for v in vs})
    def widen(self, o: AbstractState, iteration: int = 0) -> AbstractState:
        if self.is_bottom(): return o.copy()
        if o.is_bottom(): return self.copy()
        vs = self.variables() | o.variables()
        return AbstractState({v: self.get(v).widen(o.get(v)) for v in vs})
    def narrow(self, o: AbstractState) -> AbstractState:
        if self.is_bottom() or o.is_bottom(): return AbstractState.bottom_state()
        vs = self.variables() | o.variables()
        return AbstractState({v: self.get(v).narrow(o.get(v)) for v in vs})
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, AbstractState): return NotImplemented
        if self._bottom and o._bottom: return True
        if self._bottom != o._bottom: return False
        return all(self.get(v) == o.get(v) for v in self.variables() | o.variables())

# ---------------------------------------------------------------------------
# Alarm types and result
# ---------------------------------------------------------------------------
class AlarmKind(Enum):
    DIVISION_BY_ZERO = auto(); NULL_DEREFERENCE = auto()
    TYPE_ERROR = auto(); INDEX_OUT_OF_BOUNDS = auto()

@dataclass
class Alarm:
    kind: AlarmKind; line: int; col: int; message: str; severity: str = "warning"

@dataclass
class PrecisionStats:
    total_variables: int = 0; variables_with_finite_interval: int = 0
    variables_with_known_sign: int = 0; variables_with_single_type: int = 0
    variables_with_known_nullness: int = 0
    estimated_true_alarms: int = 0; estimated_false_alarms: int = 0

@dataclass
class AnalysisResult:
    abstract_states: Dict[int, AbstractState] = field(default_factory=dict)
    alarms: List[Alarm] = field(default_factory=list)
    precision_stats: PrecisionStats = field(default_factory=PrecisionStats)
    return_types: TypeDomain = field(default_factory=TypeDomain.bottom)

# ---------------------------------------------------------------------------
# Alarm Reporter
# ---------------------------------------------------------------------------
class AlarmReporter:
    def __init__(self) -> None: self.alarms: List[Alarm] = []
    def check_division(self, divisor: DomainProduct, line: int, col: int) -> None:
        if divisor.interval.contains(0):
            self.alarms.append(Alarm(AlarmKind.DIVISION_BY_ZERO, line, col,
                                     "Potential division by zero: divisor interval contains 0"))
        elif divisor.sign in (Sign.ZERO, Sign.TOP, Sign.NON_NEG, Sign.NON_POS):
            self.alarms.append(Alarm(AlarmKind.DIVISION_BY_ZERO, line, col,
                                     "Potential division by zero: sign domain includes zero"))
    def check_null_deref(self, value: DomainProduct, line: int, col: int) -> None:
        if value.null_dom in (NullDomain.NULL, NullDomain.MAYBE_NULL):
            self.alarms.append(Alarm(AlarmKind.NULL_DEREFERENCE, line, col,
                                     "Potential null dereference: value may be None"))
    def check_type_error(self, op: str, left: DomainProduct, right: DomainProduct,
                         line: int, col: int) -> None:
        numeric = frozenset({"int","float","bool"})
        if op in ("+","-","*","/","//","%","**") and left.type_dom.types and right.type_dom.types:
            if not (left.type_dom.types & numeric) and not (right.type_dom.types & numeric):
                if op == "+" and ("str" in left.type_dom and "str" in right.type_dom): return
                if op == "+" and ("list" in left.type_dom and "list" in right.type_dom): return
                self.alarms.append(Alarm(AlarmKind.TYPE_ERROR, line, col,
                    f"Type error: '{op}' on {left.type_dom.types} and {right.type_dom.types}"))
    def check_index_bounds(self, container: DomainProduct, index: DomainProduct,
                           line: int, col: int) -> None:
        if index.interval.lo < 0 or index.interval.hi == POS_INF:
            self.alarms.append(Alarm(AlarmKind.INDEX_OUT_OF_BOUNDS, line, col,
                f"Potential index out of bounds: [{index.interval.lo}, {index.interval.hi}]"))

# ---------------------------------------------------------------------------
# Precision Analyzer
# ---------------------------------------------------------------------------
class PrecisionAnalyzer:
    def compute_stats(self, states: Dict[int, AbstractState], alarms: List[Alarm]) -> PrecisionStats:
        stats = PrecisionStats()
        all_vars: Set[str] = set()
        for st in states.values(): all_vars |= st.variables()
        stats.total_variables = len(all_vars)
        fin, ks, st_set, kn = set(), set(), set(), set()
        for st in states.values():
            for v in st.variables():
                dp = st.get(v)
                if not dp.interval.is_top() and not dp.interval.is_bottom(): fin.add(v)
                if dp.sign not in (Sign.TOP, Sign.BOTTOM): ks.add(v)
                if len(dp.type_dom.types) == 1: st_set.add(v)
                if dp.null_dom in (NullDomain.NULL, NullDomain.NOT_NULL): kn.add(v)
        stats.variables_with_finite_interval = len(fin)
        stats.variables_with_known_sign = len(ks)
        stats.variables_with_single_type = len(st_set)
        stats.variables_with_known_nullness = len(kn)
        ratio = (len(fin)+len(ks)+len(st_set)) / (3*len(all_vars)) if all_vars else 0.0
        for _ in alarms:
            if ratio > 0.6: stats.estimated_true_alarms += 1
            else: stats.estimated_false_alarms += 1
        return stats

# ---------------------------------------------------------------------------
# Abstract Interpreter
# ---------------------------------------------------------------------------
_WIDENING_DELAY, _MAX_ITERATIONS, _NARROW_PASSES = 3, 100, 3

class AbstractInterpreter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reporter = AlarmReporter()
        self.precision = PrecisionAnalyzer()
        self._states: Dict[int, AbstractState] = {}
        self._return_types = TypeDomain.bottom()

    def analyze(self, source_code: str) -> AnalysisResult:
        tree = ast.parse(textwrap.dedent(source_code))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._forward_analysis(node); self._backward_analysis(node); break
        else:
            self._analyse_body(tree.body, AbstractState.top_state())
        stats = self.precision.compute_stats(self._states, self.reporter.alarms)
        return AnalysisResult(dict(self._states), list(self.reporter.alarms), stats, self._return_types)

    def _forward_analysis(self, func: ast.FunctionDef) -> AbstractState:
        init = AbstractState.top_state()
        for arg in func.args.args: init = init.set(arg.arg, DomainProduct.top())
        return self._analyse_body(func.body, init)

    def _analyse_body(self, stmts: List[ast.stmt], state: AbstractState) -> AbstractState:
        for stmt in stmts:
            if state.is_bottom(): break
            state = self._transfer(stmt, state)
            self._states[getattr(stmt, "lineno", 0)] = state
        return state

    def _transfer(self, stmt: ast.stmt, state: AbstractState) -> AbstractState:
        if isinstance(stmt, ast.Assign): return self._transfer_assign(stmt, state)
        if isinstance(stmt, ast.AugAssign): return self._transfer_aug_assign(stmt, state)
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value and isinstance(stmt.target, ast.Name):
                return state.set(stmt.target.id, self._eval_expr(stmt.value, state))
            return state
        if isinstance(stmt, ast.If): return self._transfer_if(stmt, state)
        if isinstance(stmt, ast.While): return self._transfer_while(stmt, state)
        if isinstance(stmt, ast.For): return self._transfer_for(stmt, state)
        if isinstance(stmt, ast.Return): return self._transfer_return(stmt, state)
        if isinstance(stmt, ast.Expr): self._eval_expr(stmt.value, state); return state
        if isinstance(stmt, ast.Assert): return self._apply_condition(stmt.test, state, True)
        return state

    def _transfer_assign(self, stmt: ast.Assign, state: AbstractState) -> AbstractState:
        val = self._eval_expr(stmt.value, state)
        for target in stmt.targets:
            if isinstance(target, ast.Name): state = state.set(target.id, val)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name): state = state.set(elt.id, DomainProduct.top())
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                idx = self._eval_expr(target.slice, state)
                self.reporter.check_index_bounds(state.get(target.value.id), idx, stmt.lineno, 0)
        return state

    def _transfer_aug_assign(self, stmt: ast.AugAssign, state: AbstractState) -> AbstractState:
        if isinstance(stmt.target, ast.Name):
            left, right = state.get(stmt.target.id), self._eval_expr(stmt.value, state)
            state = state.set(stmt.target.id, self._apply_binop(stmt.op, left, right, stmt.lineno, 0))
        return state

    def _transfer_if(self, stmt: ast.If, state: AbstractState) -> AbstractState:
        self._eval_expr(stmt.test, state)
        t_out = self._analyse_body(stmt.body, self._apply_condition(stmt.test, state, True))
        f_out = self._analyse_body(stmt.orelse, self._apply_condition(stmt.test, state, False)) if stmt.orelse else self._apply_condition(stmt.test, state, False)
        return t_out.join(f_out)

    def _transfer_while(self, stmt: ast.While, state: AbstractState) -> AbstractState:
        current = state
        for iteration in range(_MAX_ITERATIONS):
            loop_entry = self._apply_condition(stmt.test, current, True)
            loop_exit = self._analyse_body(stmt.body, loop_entry)
            next_st = state.join(loop_exit)
            if iteration >= _WIDENING_DELAY: next_st = current.widen(next_st, iteration)
            if next_st == current: break
            current = next_st
        for _ in range(_NARROW_PASSES):
            loop_entry = self._apply_condition(stmt.test, current, True)
            loop_exit = self._analyse_body(stmt.body, loop_entry)
            narrowed = current.narrow(state.join(loop_exit))
            if narrowed == current: break
            current = narrowed
        exit_st = self._apply_condition(stmt.test, current, False)
        if stmt.orelse: exit_st = exit_st.join(self._analyse_body(stmt.orelse, exit_st))
        return exit_st

    def _transfer_for(self, stmt: ast.For, state: AbstractState) -> AbstractState:
        self._eval_expr(stmt.iter, state)
        if isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == "range":
            lv = self._abstract_range(stmt.iter.args, state)
        else:
            lv = DomainProduct.top()
        tname = stmt.target.id if isinstance(stmt.target, ast.Name) else None
        current = state
        for iteration in range(_MAX_ITERATIONS):
            ls = current.copy()
            if tname: ls = ls.set(tname, lv)
            loop_out = self._analyse_body(stmt.body, ls)
            next_st = state.join(loop_out)
            if iteration >= _WIDENING_DELAY: next_st = current.widen(next_st, iteration)
            if next_st == current: break
            current = next_st
        for _ in range(_NARROW_PASSES):
            ls = current.copy()
            if tname: ls = ls.set(tname, lv)
            narrowed = current.narrow(state.join(self._analyse_body(stmt.body, ls)))
            if narrowed == current: break
            current = narrowed
        if stmt.orelse: current = self._analyse_body(stmt.orelse, current)
        return current

    def _transfer_return(self, stmt: ast.Return, state: AbstractState) -> AbstractState:
        if stmt.value:
            self._return_types = self._return_types.join(self._eval_expr(stmt.value, state).type_dom)
        else:
            self._return_types = self._return_types.join(TypeDomain.single("NoneType"))
        return AbstractState.bottom_state()

    def _eval_expr(self, expr: ast.expr, state: AbstractState) -> DomainProduct:
        if isinstance(expr, ast.Constant): return self._eval_constant(expr)
        if isinstance(expr, ast.Name):
            dp = state.get(expr.id)
            self.reporter.check_null_deref(dp, getattr(expr,"lineno",0), getattr(expr,"col_offset",0))
            return dp
        if isinstance(expr, ast.BinOp):
            l, r = self._eval_expr(expr.left, state), self._eval_expr(expr.right, state)
            return self._apply_binop(expr.op, l, r, getattr(expr,"lineno",0), getattr(expr,"col_offset",0))
        if isinstance(expr, ast.UnaryOp):
            op_val = self._eval_expr(expr.operand, state)
            if isinstance(expr.op, ast.USub):
                return DomainProduct(Interval(-op_val.interval.hi, -op_val.interval.lo),
                                     op_val.sign.negate(), op_val.type_dom, op_val.null_dom)
            if isinstance(expr.op, ast.Not):
                return DomainProduct(Interval(0,1), Sign.NON_NEG, TypeDomain.single("bool"), NullDomain.NOT_NULL)
            if isinstance(expr.op, ast.Invert):
                return DomainProduct(Interval.top(), Sign.TOP, TypeDomain.single("int"), NullDomain.NOT_NULL)
            return op_val
        if isinstance(expr, ast.BoolOp):
            vals = [self._eval_expr(v, state) for v in expr.values]
            r = vals[0]
            for v in vals[1:]: r = r.join(v)
            return r
        if isinstance(expr, ast.Compare):
            self._eval_expr(expr.left, state)
            for c in expr.comparators: self._eval_expr(c, state)
            return DomainProduct(Interval(0,1), Sign.NON_NEG, TypeDomain.single("bool"), NullDomain.NOT_NULL)
        if isinstance(expr, ast.Call): return self._eval_call(expr, state)
        if isinstance(expr, ast.Subscript):
            cont = self._eval_expr(expr.value, state); idx = self._eval_expr(expr.slice, state)
            self.reporter.check_index_bounds(cont, idx, getattr(expr,"lineno",0), getattr(expr,"col_offset",0))
            return DomainProduct.top()
        if isinstance(expr, ast.Attribute):
            obj = self._eval_expr(expr.value, state)
            self.reporter.check_null_deref(obj, getattr(expr,"lineno",0), getattr(expr,"col_offset",0))
            return DomainProduct.top()
        if isinstance(expr, ast.List):
            for e in expr.elts: self._eval_expr(e, state)
            return DomainProduct(Interval.const(len(expr.elts)), Sign.NON_NEG, TypeDomain.single("list"), NullDomain.NOT_NULL)
        if isinstance(expr, ast.Dict):
            for k in expr.keys:
                if k: self._eval_expr(k, state)
            for v in expr.values: self._eval_expr(v, state)
            return DomainProduct(Interval.const(len(expr.keys)), Sign.NON_NEG, TypeDomain.single("dict"), NullDomain.NOT_NULL)
        if isinstance(expr, ast.Set):
            for e in expr.elts: self._eval_expr(e, state)
            return DomainProduct(Interval.const(len(expr.elts)), Sign.NON_NEG, TypeDomain.single("set"), NullDomain.NOT_NULL)
        if isinstance(expr, ast.Tuple):
            for e in expr.elts: self._eval_expr(e, state)
            return DomainProduct(Interval.const(len(expr.elts)), Sign.NON_NEG, TypeDomain.single("tuple"), NullDomain.NOT_NULL)
        if isinstance(expr, ast.IfExp):
            return self._eval_expr(expr.body, state).join(self._eval_expr(expr.orelse, state))
        return DomainProduct.top()

    def _eval_constant(self, node: ast.Constant) -> DomainProduct:
        v = node.value
        if v is None:
            return DomainProduct(Interval.bottom(), Sign.BOTTOM, TypeDomain.single("NoneType"), NullDomain.NULL)
        if isinstance(v, bool):
            return DomainProduct(Interval.const(int(v)), Sign.POSITIVE if v else Sign.ZERO,
                                 TypeDomain.single("bool"), NullDomain.NOT_NULL)
        if isinstance(v, int):
            s = Sign.POSITIVE if v > 0 else (Sign.NEGATIVE if v < 0 else Sign.ZERO)
            return DomainProduct(Interval.const(v), s, TypeDomain.single("int"), NullDomain.NOT_NULL)
        if isinstance(v, float):
            s = Sign.POSITIVE if v > 0 else (Sign.NEGATIVE if v < 0 else (Sign.ZERO if v == 0.0 else Sign.TOP))
            return DomainProduct(Interval(v, v), s, TypeDomain.single("float"), NullDomain.NOT_NULL)
        if isinstance(v, str):
            return DomainProduct(Interval.const(len(v)), Sign.NON_NEG, TypeDomain.single("str"), NullDomain.NOT_NULL)
        if isinstance(v, bytes):
            return DomainProduct(Interval.const(len(v)), Sign.NON_NEG, TypeDomain.single("bytes"), NullDomain.NOT_NULL)
        return DomainProduct.top()

    def _eval_call(self, call: ast.Call, state: AbstractState) -> DomainProduct:
        arg_vals = [self._eval_expr(a, state) for a in call.args]
        if isinstance(call.func, ast.Name):
            n = call.func.id
            if n == "len":
                return DomainProduct(Interval(0, POS_INF), Sign.NON_NEG, TypeDomain.single("int"), NullDomain.NOT_NULL)
            if n == "int":
                return DomainProduct(Interval.top(), Sign.TOP, TypeDomain.single("int"), NullDomain.NOT_NULL)
            if n == "float":
                return DomainProduct(Interval.top(), Sign.TOP, TypeDomain.single("float"), NullDomain.NOT_NULL)
            if n == "str":
                return DomainProduct(Interval(0, POS_INF), Sign.NON_NEG, TypeDomain.single("str"), NullDomain.NOT_NULL)
            if n == "bool":
                return DomainProduct(Interval(0, 1), Sign.NON_NEG, TypeDomain.single("bool"), NullDomain.NOT_NULL)
            if n == "abs" and arg_vals:
                inner = arg_vals[0]
                hi = max(abs(inner.interval.lo) if inner.interval.lo != NEG_INF else POS_INF,
                         abs(inner.interval.hi) if inner.interval.hi != POS_INF else POS_INF)
                return DomainProduct(Interval(0, hi), Sign.NON_NEG, inner.type_dom, NullDomain.NOT_NULL)
            if n == "range":
                return DomainProduct(Interval(0, POS_INF), Sign.NON_NEG, TypeDomain.single("list"), NullDomain.NOT_NULL)
            if n in ("list","dict","set"):
                return DomainProduct(Interval(0, POS_INF), Sign.NON_NEG, TypeDomain.single(n), NullDomain.NOT_NULL)
            if n == "print":
                return DomainProduct(Interval.bottom(), Sign.BOTTOM, TypeDomain.single("NoneType"), NullDomain.NULL)
            if n in ("min","max") and arg_vals:
                result = arg_vals[0]
                for av in arg_vals[1:]: result = result.join(av)
                if n == "min":
                    niv = Interval(min(a.interval.lo for a in arg_vals), min(a.interval.hi for a in arg_vals))
                else:
                    niv = Interval(max(a.interval.lo for a in arg_vals), max(a.interval.hi for a in arg_vals))
                return DomainProduct(niv, result.sign, result.type_dom, NullDomain.NOT_NULL)
        if isinstance(call.func, ast.Attribute): self._eval_expr(call.func.value, state)
        return DomainProduct.top()

    def _abstract_range(self, args: List[ast.expr], state: AbstractState) -> DomainProduct:
        if len(args) == 1:
            stop = self._eval_expr(args[0], state)
            hi = stop.interval.hi - 1 if stop.interval.hi != POS_INF else POS_INF
            return DomainProduct(Interval(0, hi), Sign.NON_NEG, TypeDomain.single("int"), NullDomain.NOT_NULL)
        if len(args) >= 2:
            start, stop = self._eval_expr(args[0], state), self._eval_expr(args[1], state)
            lo = start.interval.lo
            hi = stop.interval.hi - 1 if stop.interval.hi != POS_INF else POS_INF
            s = Sign.NON_NEG if lo >= 0 else (Sign.NEGATIVE if hi < 0 else Sign.TOP)
            return DomainProduct(Interval(lo, hi), s, TypeDomain.single("int"), NullDomain.NOT_NULL)
        return DomainProduct(Interval.top(), Sign.TOP, TypeDomain.single("int"), NullDomain.NOT_NULL)

    def _apply_binop(self, op: ast.operator, left: DomainProduct, right: DomainProduct,
                     line: int, col: int) -> DomainProduct:
        self.reporter.check_type_error(self._op_symbol(op), left, right, line, col)
        if isinstance(op, ast.Add):
            return DomainProduct(left.interval.add(right.interval), left.sign.add(right.sign),
                                 self._binop_type(left.type_dom, right.type_dom, "+"), NullDomain.NOT_NULL)
        if isinstance(op, ast.Sub):
            return DomainProduct(left.interval.sub(right.interval), left.sign.sub(right.sign),
                                 self._binop_type(left.type_dom, right.type_dom, "-"), NullDomain.NOT_NULL)
        if isinstance(op, ast.Mult):
            return DomainProduct(left.interval.mul(right.interval), left.sign.mul(right.sign),
                                 self._binop_type(left.type_dom, right.type_dom, "*"), NullDomain.NOT_NULL)
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            self.reporter.check_division(right, line, col)
            return DomainProduct(left.interval.div(right.interval), left.sign.div(right.sign),
                                 self._binop_type(left.type_dom, right.type_dom, "/"), NullDomain.NOT_NULL)
        if isinstance(op, ast.Mod):
            self.reporter.check_division(right, line, col)
            if right.interval.is_bottom(): return DomainProduct.bottom()
            hi = max(abs(right.interval.lo), abs(right.interval.hi)) if right.interval.hi != POS_INF else POS_INF
            return DomainProduct(Interval(0, hi), Sign.NON_NEG,
                                 self._binop_type(left.type_dom, right.type_dom, "%"), NullDomain.NOT_NULL)
        if isinstance(op, ast.Pow):
            return DomainProduct(Interval.top(), Sign.TOP,
                                 self._binop_type(left.type_dom, right.type_dom, "**"), NullDomain.NOT_NULL)
        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            return DomainProduct(Interval.top(), Sign.TOP, TypeDomain.single("int"), NullDomain.NOT_NULL)
        return DomainProduct.top()

    @staticmethod
    def _op_symbol(op: ast.operator) -> str:
        return {ast.Add:"+", ast.Sub:"-", ast.Mult:"*", ast.Div:"/", ast.FloorDiv:"//",
                ast.Mod:"%", ast.Pow:"**", ast.BitAnd:"&", ast.BitOr:"|", ast.BitXor:"^",
                ast.LShift:"<<", ast.RShift:">>"}.get(type(op), "?")

    @staticmethod
    def _binop_type(left: TypeDomain, right: TypeDomain, op: str) -> TypeDomain:
        numeric = frozenset({"int","float","bool"})
        result: Set[str] = set()
        if left.types & numeric and right.types & numeric:
            if "float" in left.types or "float" in right.types: result.add("float")
            if ("int" in left.types or "bool" in left.types) and ("int" in right.types or "bool" in right.types):
                result.add("float" if op == "/" else "int")
        if op == "+":
            if "str" in left.types and "str" in right.types: result.add("str")
            if "list" in left.types and "list" in right.types: result.add("list")
            if "tuple" in left.types and "tuple" in right.types: result.add("tuple")
        if op == "*":
            if "str" in left.types and ("int" in right.types or "bool" in right.types): result.add("str")
            if "list" in left.types and ("int" in right.types or "bool" in right.types): result.add("list")
        return TypeDomain(frozenset(result)) if result else TypeDomain.top()

    # --- condition refinement ------------------------------------------------
    def _apply_condition(self, cond: ast.expr, state: AbstractState, polarity: bool) -> AbstractState:
        if isinstance(cond, ast.Compare) and len(cond.ops) == 1:
            return self._refine_comparison(cond.left, cond.ops[0], cond.comparators[0], state, polarity)
        if isinstance(cond, ast.UnaryOp) and isinstance(cond.op, ast.Not):
            return self._apply_condition(cond.operand, state, not polarity)
        if isinstance(cond, ast.BoolOp):
            if isinstance(cond.op, ast.And):
                if polarity:
                    r = state
                    for v in cond.values: r = self._apply_condition(v, r, True)
                    return r
                else:
                    r = AbstractState.bottom_state()
                    for v in cond.values: r = r.join(self._apply_condition(v, state, False))
                    return r
            if isinstance(cond.op, ast.Or):
                if polarity:
                    r = AbstractState.bottom_state()
                    for v in cond.values: r = r.join(self._apply_condition(v, state, True))
                    return r
                else:
                    r = state
                    for v in cond.values: r = self._apply_condition(v, r, False)
                    return r
        if isinstance(cond, ast.Name) and polarity:
            dp = state.get(cond.id)
            return state.set(cond.id, DomainProduct(dp.interval, dp.sign, dp.type_dom, dp.null_dom.meet(NullDomain.NOT_NULL)))
        return state

    def _refine_comparison(self, left: ast.expr, op: ast.cmpop, right: ast.expr,
                           state: AbstractState, polarity: bool) -> AbstractState:
        left_val = self._eval_expr(left, state)
        right_val = self._eval_expr(right, state)
        if not isinstance(left, ast.Name): return state
        var, dp, iv = left.id, state.get(left.id), state.get(left.id).interval
        # None checks
        if isinstance(op, ast.Is) and isinstance(right, ast.Constant) and right.value is None:
            if polarity:
                return state.set(var, DomainProduct(Interval.bottom(), Sign.BOTTOM, TypeDomain.single("NoneType"), NullDomain.NULL))
            return state.set(var, DomainProduct(dp.interval, dp.sign, dp.type_dom, NullDomain.NOT_NULL))
        if isinstance(op, ast.IsNot) and isinstance(right, ast.Constant) and right.value is None:
            if polarity:
                return state.set(var, DomainProduct(dp.interval, dp.sign, dp.type_dom, NullDomain.NOT_NULL))
            return state.set(var, DomainProduct(Interval.bottom(), Sign.BOTTOM, TypeDomain.single("NoneType"), NullDomain.NULL))
        # Numeric refinement
        riv = right_val.interval
        if isinstance(op, ast.Lt):
            new_iv = iv.meet(Interval(NEG_INF, riv.hi - 1)) if polarity else iv.meet(Interval(riv.lo, POS_INF))
        elif isinstance(op, ast.LtE):
            new_iv = iv.meet(Interval(NEG_INF, riv.hi)) if polarity else iv.meet(Interval(riv.lo + 1, POS_INF))
        elif isinstance(op, ast.Gt):
            new_iv = iv.meet(Interval(riv.lo + 1, POS_INF)) if polarity else iv.meet(Interval(NEG_INF, riv.hi))
        elif isinstance(op, ast.GtE):
            new_iv = iv.meet(Interval(riv.lo, POS_INF)) if polarity else iv.meet(Interval(NEG_INF, riv.hi - 1))
        elif isinstance(op, ast.Eq):
            new_iv = iv.meet(riv) if polarity else iv
        elif isinstance(op, ast.NotEq):
            new_iv = iv if polarity else iv.meet(riv)
        else:
            new_iv = iv
        new_sign = dp.sign
        if not new_iv.is_bottom():
            if new_iv.lo >= 0: new_sign = new_sign.meet(Sign.NON_NEG)
            elif new_iv.hi < 0: new_sign = new_sign.meet(Sign.NEGATIVE)
        return state.set(var, DomainProduct(new_iv, new_sign, dp.type_dom, dp.null_dom))

    # --- backward analysis ---------------------------------------------------
    def _backward_analysis(self, func: ast.FunctionDef) -> None:
        post = AbstractState.top_state()
        for stmt in reversed(func.body):
            post = self._backward_transfer(stmt, post)
            line = getattr(stmt, "lineno", 0)
            if line in self._states: self._states[line] = self._states[line].meet(post)

    def _backward_transfer(self, stmt: ast.stmt, post: AbstractState) -> AbstractState:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    pre = post.copy(); pre._map.pop(target.id, None)
                    for v, dp in self._backward_expr(stmt.value, post).items():
                        pre = pre.set(v, pre.get(v).meet(dp))
                    return pre
            return post
        if isinstance(stmt, ast.Return): return AbstractState.top_state()
        return post

    def _backward_expr(self, expr: ast.expr, state: AbstractState) -> Dict[str, DomainProduct]:
        reqs: Dict[str, DomainProduct] = {}
        if isinstance(expr, ast.Name):
            reqs[expr.id] = state.get(expr.id)
        elif isinstance(expr, ast.BinOp):
            for k, v in self._backward_expr(expr.left, state).items(): reqs[k] = v
            for k, v in self._backward_expr(expr.right, state).items():
                reqs[k] = reqs[k].meet(v) if k in reqs else v
            if isinstance(expr.op, (ast.Div, ast.FloorDiv, ast.Mod)) and isinstance(expr.right, ast.Name):
                dp = state.get(expr.right.id)
                if dp.interval.lo == 0: ref = Interval(1, dp.interval.hi)
                elif dp.interval.hi == 0: ref = Interval(dp.interval.lo, -1)
                else: ref = dp.interval
                reqs[expr.right.id] = DomainProduct(ref, dp.sign, dp.type_dom, NullDomain.NOT_NULL)
        elif isinstance(expr, ast.UnaryOp):
            reqs.update(self._backward_expr(expr.operand, state))
        return reqs

# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def analyze_code(source: str) -> AnalysisResult:
    return AbstractInterpreter().analyze(source)

def _self_test() -> None:
    sample = textwrap.dedent("""\
        def foo(n):
            x = 0
            y = 1
            i = 0
            while i < n:
                x = x + y
                y = y * 2
                i = i + 1
            if x > 0:
                z = 100 / x
            else:
                z = None
            if z is not None:
                return z
            return -1
    """)
    result = analyze_code(sample)
    assert isinstance(result, AnalysisResult) and len(result.abstract_states) > 0
    a, b = Interval(1, 5), Interval(3, 10)
    assert a.join(b) == Interval(1, 10) and a.meet(b) == Interval(3, 5)
    assert a.add(b) == Interval(4, 15) and a.sub(b) == Interval(-9, 2)
    assert Sign.POSITIVE.add(Sign.POSITIVE) == Sign.POSITIVE
    assert Sign.NEGATIVE.mul(Sign.NEGATIVE) == Sign.POSITIVE
    assert Sign.POSITIVE.negate() == Sign.NEGATIVE
    assert Interval(0, 5).widen(Interval(0, 15)).hi >= 15
    assert TypeDomain.single("int").join(TypeDomain.single("float")) == TypeDomain(frozenset({"int","float"}))
    assert NullDomain.NULL.join(NullDomain.NOT_NULL) == NullDomain.MAYBE_NULL
    assert NullDomain.MAYBE_NULL.meet(NullDomain.NOT_NULL) == NullDomain.NOT_NULL
    thresholds = np.array(WIDENING_THRESHOLDS[1:-1], dtype=np.float64)
    assert len(thresholds) == 7

if __name__ == "__main__":
    _self_test(); print("All self-tests passed.")
