"""SMT-LIB 2.6 serialization for ExprNode trees.

Generates standard SMT-LIB output with support for multiple logics,
incremental solving (push/pop), dReal extensions, and response parsing.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, TextIO, Tuple, Union

from .expression import (
    Abs,
    Add,
    And,
    Const,
    Cos,
    Div,
    Eq,
    ExprNode,
    Exists,
    Exp,
    ForAll,
    Ge,
    Gt,
    HillAct,
    HillRep,
    Implies,
    Ite,
    Le,
    Log,
    Lt,
    Max,
    Min,
    Mul,
    Neg,
    Not,
    Or,
    Pow,
    Sin,
    Sqrt,
    Var,
    ZERO,
    const_value,
)


# ---------------------------------------------------------------------------
# Logic selection
# ---------------------------------------------------------------------------

class SMTLogic(Enum):
    """Supported SMT-LIB logics."""
    QF_NRA = "QF_NRA"    # quantifier-free nonlinear real arithmetic
    QF_LRA = "QF_LRA"    # quantifier-free linear real arithmetic
    NRA = "NRA"           # nonlinear real arithmetic (with quantifiers)
    QF_NIA = "QF_NIA"    # quantifier-free nonlinear integer arithmetic
    ALL = "ALL"           # all theories


def auto_select_logic(expr: ExprNode, has_quantifiers: bool = False) -> SMTLogic:
    """Choose the tightest SMT-LIB logic for *expr*."""
    has_nonlinear = False
    for node in expr.iter_preorder():
        if isinstance(node, (Mul, Div, Pow, Exp, Log, Sin, Cos, Sqrt,
                             HillAct, HillRep)):
            if isinstance(node, Mul):
                lv = const_value(node.left)
                rv = const_value(node.right)
                if lv is None and rv is None:
                    has_nonlinear = True
            else:
                has_nonlinear = True
        if isinstance(node, (ForAll, Exists)):
            has_quantifiers = True

    if has_quantifiers:
        return SMTLogic.NRA
    return SMTLogic.QF_NRA if has_nonlinear else SMTLogic.QF_LRA


# ---------------------------------------------------------------------------
# Serialization context
# ---------------------------------------------------------------------------

@dataclass
class SerializerConfig:
    """Configuration for SMT-LIB output."""
    logic: Optional[SMTLogic] = None
    produce_models: bool = True
    produce_unsat_cores: bool = False
    dreal_precision: Optional[float] = None
    indent: str = "  "
    comments: bool = True
    cse_threshold: int = 3  # CSE for subexpressions used >= this many times
    line_width: int = 120


@dataclass
class _SerCtx:
    """Internal serialization context for CSE and variable tracking."""
    declared_vars: Set[str] = field(default_factory=set)
    cse_map: Dict[int, str] = field(default_factory=dict)  # id(node) -> name
    cse_defs: List[Tuple[str, str]] = field(default_factory=list)  # (name, smt_expr)
    _cse_counter: int = 0

    def fresh_cse(self) -> str:
        name = f"_cse_{self._cse_counter}"
        self._cse_counter += 1
        return name


# ---------------------------------------------------------------------------
# ExprNode → SMT-LIB string
# ---------------------------------------------------------------------------

def _fmt_const(v: float) -> str:
    """Format a constant for SMT-LIB."""
    if v == int(v) and not (v == float("inf") or v == float("-inf")):
        iv = int(v)
        return f"(- {-iv})" if iv < 0 else str(iv)
    if v < 0:
        return f"(- {_fmt_const(-v)})"
    # Real literal
    s = f"{v:.15g}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


def expr_to_smtlib(e: ExprNode, ctx: Optional[_SerCtx] = None) -> str:
    """Convert an ExprNode to SMT-LIB 2.6 string."""
    if ctx and id(e) in ctx.cse_map:
        return ctx.cse_map[id(e)]

    if isinstance(e, Const):
        return _fmt_const(e.value)

    if isinstance(e, Var):
        return e.name

    if isinstance(e, Add):
        return f"(+ {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"

    if isinstance(e, Mul):
        return f"(* {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"

    if isinstance(e, Div):
        return f"(/ {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"

    if isinstance(e, Neg):
        return f"(- {expr_to_smtlib(e.operand, ctx)})"

    if isinstance(e, Pow):
        rv = const_value(e.right)
        if rv is not None and float(rv).is_integer() and rv >= 0:
            n = int(rv)
            if n == 0:
                return "1"
            if n == 1:
                return expr_to_smtlib(e.left, ctx)
            if n == 2:
                base = expr_to_smtlib(e.left, ctx)
                return f"(* {base} {base})"
            # General integer power via nested multiplication
            base = expr_to_smtlib(e.left, ctx)
            result = base
            for _ in range(n - 1):
                result = f"(* {result} {base})"
            return result
        # Non-integer: use pow (dReal extension)
        return f"(^ {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"

    if isinstance(e, Exp):
        return f"(exp {expr_to_smtlib(e.operand, ctx)})"

    if isinstance(e, Log):
        return f"(log {expr_to_smtlib(e.operand, ctx)})"

    if isinstance(e, Sin):
        return f"(sin {expr_to_smtlib(e.operand, ctx)})"

    if isinstance(e, Cos):
        return f"(cos {expr_to_smtlib(e.operand, ctx)})"

    if isinstance(e, Sqrt):
        return f"(^ {expr_to_smtlib(e.operand, ctx)} 0.5)"

    if isinstance(e, Abs):
        inner = expr_to_smtlib(e.operand, ctx)
        return f"(ite (>= {inner} 0) {inner} (- {inner}))"

    if isinstance(e, Min):
        l = expr_to_smtlib(e.left, ctx)
        r = expr_to_smtlib(e.right, ctx)
        return f"(ite (<= {l} {r}) {l} {r})"

    if isinstance(e, Max):
        l = expr_to_smtlib(e.left, ctx)
        r = expr_to_smtlib(e.right, ctx)
        return f"(ite (>= {l} {r}) {l} {r})"

    # Comparisons
    if isinstance(e, Lt):
        return f"(< {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"
    if isinstance(e, Le):
        return f"(<= {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"
    if isinstance(e, Eq):
        return f"(= {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"
    if isinstance(e, Ge):
        return f"(>= {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"
    if isinstance(e, Gt):
        return f"(> {expr_to_smtlib(e.left, ctx)} {expr_to_smtlib(e.right, ctx)})"

    # Boolean
    if isinstance(e, And):
        args = " ".join(expr_to_smtlib(a, ctx) for a in e.args)
        return f"(and {args})"
    if isinstance(e, Or):
        args = " ".join(expr_to_smtlib(a, ctx) for a in e.args)
        return f"(or {args})"
    if isinstance(e, Not):
        return f"(not {expr_to_smtlib(e.operand, ctx)})"
    if isinstance(e, Implies):
        return f"(=> {expr_to_smtlib(e.antecedent, ctx)} {expr_to_smtlib(e.consequent, ctx)})"
    if isinstance(e, Ite):
        return (f"(ite {expr_to_smtlib(e.cond, ctx)} "
                f"{expr_to_smtlib(e.then_expr, ctx)} "
                f"{expr_to_smtlib(e.else_expr, ctx)})")

    # Quantifiers
    if isinstance(e, ForAll):
        body = expr_to_smtlib(e.body, ctx)
        decl = f"(({e.var} Real))"
        result = f"(forall {decl} {body})"
        if e.domain is not None:
            lo = _fmt_const(e.domain.lo)
            hi = _fmt_const(e.domain.hi)
            bound = f"(and (>= {e.var} {lo}) (<= {e.var} {hi}))"
            result = f"(forall {decl} (=> {bound} {body}))"
        return result

    if isinstance(e, Exists):
        body = expr_to_smtlib(e.body, ctx)
        decl = f"(({e.var} Real))"
        result = f"(exists {decl} {body})"
        if e.domain is not None:
            lo = _fmt_const(e.domain.lo)
            hi = _fmt_const(e.domain.hi)
            bound = f"(and (>= {e.var} {lo}) (<= {e.var} {hi}))"
            result = f"(exists {decl} (and {bound} {body}))"
        return result

    # Hill functions: expand inline
    if isinstance(e, HillAct):
        x = expr_to_smtlib(e.x, ctx)
        K = expr_to_smtlib(e.K, ctx)
        n = expr_to_smtlib(e.n, ctx)
        xn = f"(^ {x} {n})"
        kn = f"(^ {K} {n})"
        return f"(/ {xn} (+ {kn} {xn}))"

    if isinstance(e, HillRep):
        x = expr_to_smtlib(e.x, ctx)
        K = expr_to_smtlib(e.K, ctx)
        n = expr_to_smtlib(e.n, ctx)
        xn = f"(^ {x} {n})"
        kn = f"(^ {K} {n})"
        return f"(/ {kn} (+ {kn} {xn}))"

    raise ValueError(f"Cannot serialize ExprNode type: {type(e).__name__}")


# ---------------------------------------------------------------------------
# CSE analysis
# ---------------------------------------------------------------------------

def _compute_cse(expr: ExprNode, threshold: int = 3) -> _SerCtx:
    """Find repeated subexpressions and assign define-fun names."""
    counts: Dict[int, int] = {}
    nodes: Dict[int, ExprNode] = {}

    for node in expr.iter_postorder():
        nid = id(node)
        counts[nid] = counts.get(nid, 0) + 1
        nodes[nid] = node

    ctx = _SerCtx()
    for nid, count in counts.items():
        node = nodes[nid]
        if count >= threshold and node.size() > 2:
            name = ctx.fresh_cse()
            ctx.cse_map[nid] = name
            smt_str = expr_to_smtlib(node, None)  # no CSE for the def itself
            ctx.cse_defs.append((name, smt_str))

    return ctx


# ---------------------------------------------------------------------------
# Full SMT-LIB file generation
# ---------------------------------------------------------------------------

def serialize_smtlib(
    assertions: List[ExprNode],
    variables: Optional[List[Var]] = None,
    config: Optional[SerializerConfig] = None,
    output: Optional[TextIO] = None,
) -> str:
    """Generate a complete SMT-LIB 2.6 file.

    If *output* is provided, writes to it and returns "".
    Otherwise returns the SMT-LIB string.
    """
    cfg = config or SerializerConfig()
    buf = output or io.StringIO()

    # Collect all free variables if not provided
    if variables is None:
        all_vars: Set[str] = set()
        for a in assertions:
            all_vars |= a.free_vars()
        var_names = sorted(all_vars)
    else:
        var_names = [v.name for v in variables]

    # Auto-select logic
    logic = cfg.logic
    if logic is None:
        has_q = any(isinstance(n, (ForAll, Exists))
                    for a in assertions for n in a.iter_preorder())
        combined = And(*assertions) if len(assertions) > 1 else assertions[0] if assertions else Const(1.0)
        logic = auto_select_logic(combined, has_q)

    # CSE
    ctx = _SerCtx()

    # Header
    if cfg.comments:
        buf.write("; SMT-LIB 2.6 encoding generated by BioProver\n")
        buf.write(f"; Logic: {logic.value}\n")
        buf.write(f"; Variables: {len(var_names)}\n")
        buf.write(f"; Assertions: {len(assertions)}\n\n")

    buf.write(f"(set-logic {logic.value})\n")

    if cfg.produce_models:
        buf.write("(set-option :produce-models true)\n")
    if cfg.produce_unsat_cores:
        buf.write("(set-option :produce-unsat-cores true)\n")

    # dReal precision
    if cfg.dreal_precision is not None:
        buf.write(f"(set-option :precision {cfg.dreal_precision})\n")

    buf.write("\n")

    # Variable declarations
    if cfg.comments:
        buf.write("; Variable declarations\n")
    for name in var_names:
        buf.write(f"(declare-const {name} Real)\n")
    buf.write("\n")

    # CSE define-funs
    if ctx.cse_defs:
        if cfg.comments:
            buf.write("; Common subexpression definitions\n")
        for name, smt_expr in ctx.cse_defs:
            buf.write(f"(define-fun {name} () Real {smt_expr})\n")
        buf.write("\n")

    # Assertions
    if cfg.comments:
        buf.write("; Assertions\n")
    for i, assertion in enumerate(assertions):
        smt_str = expr_to_smtlib(assertion, ctx)
        if cfg.comments and i > 0 and i % 10 == 0:
            buf.write(f"; --- assertion {i} ---\n")
        buf.write(f"(assert {smt_str})\n")
    buf.write("\n")

    # Check-sat
    buf.write("(check-sat)\n")
    if cfg.produce_models:
        buf.write("(get-model)\n")

    if output is not None:
        return ""
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Push / pop and incremental commands
# ---------------------------------------------------------------------------

def emit_push(output: TextIO, n: int = 1) -> None:
    """Write a push command."""
    output.write(f"(push {n})\n")


def emit_pop(output: TextIO, n: int = 1) -> None:
    """Write a pop command."""
    output.write(f"(pop {n})\n")


def emit_assert(output: TextIO, expr: ExprNode, ctx: Optional[_SerCtx] = None) -> None:
    """Write a single assertion."""
    output.write(f"(assert {expr_to_smtlib(expr, ctx)})\n")


def emit_check_sat(output: TextIO) -> None:
    output.write("(check-sat)\n")


def emit_check_sat_assuming(
    output: TextIO,
    assumptions: List[str],
) -> None:
    """Write check-sat-assuming with assumption literals."""
    lits = " ".join(assumptions)
    output.write(f"(check-sat-assuming ({lits}))\n")


def emit_get_model(output: TextIO) -> None:
    output.write("(get-model)\n")


def emit_get_value(output: TextIO, variables: List[str]) -> None:
    """Write a get-value command for specific variables."""
    vars_str = " ".join(variables)
    output.write(f"(get-value ({vars_str}))\n")


# ---------------------------------------------------------------------------
# Optimization objectives (Z3 / dReal)
# ---------------------------------------------------------------------------

def emit_minimize(output: TextIO, expr: ExprNode, ctx: Optional[_SerCtx] = None) -> None:
    """Emit a minimize objective (Z3 optimization)."""
    output.write(f"(minimize {expr_to_smtlib(expr, ctx)})\n")


def emit_maximize(output: TextIO, expr: ExprNode, ctx: Optional[_SerCtx] = None) -> None:
    """Emit a maximize objective (Z3 optimization)."""
    output.write(f"(maximize {expr_to_smtlib(expr, ctx)})\n")


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def write_smtlib_file(
    path: str,
    assertions: List[ExprNode],
    variables: Optional[List[Var]] = None,
    config: Optional[SerializerConfig] = None,
) -> None:
    """Write SMT-LIB to a file."""
    with open(path, "w") as f:
        serialize_smtlib(assertions, variables, config, output=f)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

@dataclass
class SMTResult:
    """Parsed SMT solver result."""
    status: str  # "sat", "unsat", "unknown", "delta-sat"
    model: Optional[Dict[str, float]] = None
    unsat_core: Optional[List[str]] = None
    raw: str = ""


def parse_smt_response(response: str) -> SMTResult:
    """Parse an SMT-LIB solver response string."""
    lines = response.strip().splitlines()
    if not lines:
        return SMTResult(status="unknown", raw=response)

    status_line = lines[0].strip().lower()
    if status_line == "sat":
        status = "sat"
    elif status_line == "unsat":
        status = "unsat"
    elif status_line.startswith("delta-sat"):
        status = "delta-sat"
    elif status_line == "unknown":
        status = "unknown"
    else:
        return SMTResult(status="unknown", raw=response)

    model: Optional[Dict[str, float]] = None
    if status in ("sat", "delta-sat"):
        model = _parse_model(response)

    return SMTResult(status=status, model=model, raw=response)


def _parse_model(response: str) -> Dict[str, float]:
    """Extract variable assignments from SMT-LIB model output."""
    model: Dict[str, float] = {}
    # Match patterns like (define-fun x () Real 1.5) or (= x 1.5)
    define_pat = re.compile(
        r"\(define-fun\s+(\w+)\s+\(\)\s+Real\s+([^)]+)\)"
    )
    for m in define_pat.finditer(response):
        name = m.group(1)
        val_str = m.group(2).strip()
        val = _parse_numeral(val_str)
        if val is not None:
            model[name] = val

    # Also try (= var val) patterns
    eq_pat = re.compile(r"\(\s*=\s+(\w+)\s+([^)]+)\)")
    for m in eq_pat.finditer(response):
        name = m.group(1)
        if name not in model:
            val = _parse_numeral(m.group(2).strip())
            if val is not None:
                model[name] = val

    return model


def _parse_numeral(s: str) -> Optional[float]:
    """Parse an SMT-LIB numeral, including (- x) and (/ a b)."""
    s = s.strip()
    if not s:
        return None

    # Negative: (- X)
    neg_match = re.match(r"^\(\s*-\s+(.+)\s*\)$", s)
    if neg_match:
        inner = _parse_numeral(neg_match.group(1))
        return -inner if inner is not None else None

    # Fraction: (/ a b)
    div_match = re.match(r"^\(\s*/\s+(.+?)\s+(.+?)\s*\)$", s)
    if div_match:
        num = _parse_numeral(div_match.group(1))
        den = _parse_numeral(div_match.group(2))
        if num is not None and den is not None and den != 0:
            return num / den
        return None

    # Plain number
    try:
        return float(s)
    except ValueError:
        return None
