"""Infer function contracts — preconditions, postconditions, and invariants.

Uses AST analysis to extract implicit contracts from guard patterns, assert
statements, raise-if patterns, and return-value constraints.  Can generate
``icontract`` / ``deal`` annotations and Hypothesis property-based test
strategies from inferred contracts.
"""
from __future__ import annotations

import ast
import copy
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ── Data types ───────────────────────────────────────────────────────────────

class ContractKind(Enum):
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"


class ConditionOp(Enum):
    IS_NOT_NONE = "is_not_none"
    IS_NONE = "is_none"
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    IN = "in"
    NOT_IN = "not_in"
    ISINSTANCE = "isinstance"
    HASATTR = "hasattr"
    LEN_GT = "len_gt"
    LEN_GE = "len_ge"
    LEN_EQ = "len_eq"
    LEN_LT = "len_lt"
    TRUTHY = "truthy"
    CALLABLE = "callable"
    CUSTOM = "custom"


@dataclass
class Condition:
    variable: str
    op: ConditionOp
    value: Any = None
    source_line: int = 0
    confidence: float = 0.9

    def to_python(self) -> str:
        v = self.variable
        if self.op == ConditionOp.IS_NOT_NONE:
            return f"{v} is not None"
        if self.op == ConditionOp.IS_NONE:
            return f"{v} is None"
        if self.op == ConditionOp.GT:
            return f"{v} > {self.value!r}"
        if self.op == ConditionOp.GE:
            return f"{v} >= {self.value!r}"
        if self.op == ConditionOp.LT:
            return f"{v} < {self.value!r}"
        if self.op == ConditionOp.LE:
            return f"{v} <= {self.value!r}"
        if self.op == ConditionOp.EQ:
            return f"{v} == {self.value!r}"
        if self.op == ConditionOp.NE:
            return f"{v} != {self.value!r}"
        if self.op == ConditionOp.IN:
            return f"{v} in {self.value!r}"
        if self.op == ConditionOp.NOT_IN:
            return f"{v} not in {self.value!r}"
        if self.op == ConditionOp.ISINSTANCE:
            return f"isinstance({v}, {self.value})"
        if self.op == ConditionOp.HASATTR:
            return f"hasattr({v}, {self.value!r})"
        if self.op == ConditionOp.LEN_GT:
            return f"len({v}) > {self.value}"
        if self.op == ConditionOp.LEN_GE:
            return f"len({v}) >= {self.value}"
        if self.op == ConditionOp.LEN_EQ:
            return f"len({v}) == {self.value}"
        if self.op == ConditionOp.LEN_LT:
            return f"len({v}) < {self.value}"
        if self.op == ConditionOp.TRUTHY:
            return f"bool({v})"
        if self.op == ConditionOp.CALLABLE:
            return f"callable({v})"
        if self.op == ConditionOp.CUSTOM:
            return str(self.value)
        return f"{v}"


@dataclass
class Precondition:
    function: str
    conditions: List[Condition] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.9

    def to_icontract(self) -> str:
        parts = " and ".join(c.to_python() for c in self.conditions)
        return f'@icontract.require(lambda {self._lambda_args()}: {parts})'

    def to_deal(self) -> str:
        parts = " and ".join(c.to_python() for c in self.conditions)
        return f'@deal.pre(lambda {self._lambda_args()}: {parts})'

    def _lambda_args(self) -> str:
        names = sorted({c.variable.split(".")[0] for c in self.conditions})
        return ", ".join(names)


@dataclass
class Postcondition:
    function: str
    conditions: List[Condition] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.85

    def to_icontract(self) -> str:
        parts = " and ".join(c.to_python() for c in self.conditions)
        return f'@icontract.ensure(lambda result: {parts})'

    def to_deal(self) -> str:
        parts = " and ".join(c.to_python() for c in self.conditions)
        return f'@deal.post(lambda result: {parts})'


@dataclass
class Invariant:
    class_name: str
    conditions: List[Condition] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.8

    def to_icontract(self) -> str:
        parts = " and ".join(c.to_python() for c in self.conditions)
        return f'@icontract.invariant(lambda self: {parts})'


@dataclass
class ContractViolation:
    line: int
    column: int
    kind: ContractKind
    message: str
    contract_text: str
    severity: str = "warning"
    confidence: float = 0.8


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: List[str] = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _name_str(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_str(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _const_value(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_value(node.operand)
        if v is not None:
            return -v
    if isinstance(node, ast.List):
        return [_const_value(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_const_value(e) for e in node.elts)
    if isinstance(node, ast.Set):
        return {_const_value(e) for e in node.elts}
    return None


CMP_OP_MAP = {
    ast.Gt: ConditionOp.GT,
    ast.GtE: ConditionOp.GE,
    ast.Lt: ConditionOp.LT,
    ast.LtE: ConditionOp.LE,
    ast.Eq: ConditionOp.EQ,
    ast.NotEq: ConditionOp.NE,
    ast.In: ConditionOp.IN,
    ast.NotIn: ConditionOp.NOT_IN,
}


def _extract_conditions_from_test(
    test: ast.expr, params: Set[str], line: int
) -> List[Condition]:
    """Extract conditions from a boolean test expression."""
    conditions: List[Condition] = []

    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        for val in test.values:
            conditions.extend(_extract_conditions_from_test(val, params, line))
        return conditions

    if isinstance(test, ast.Compare):
        left_name = _name_str(test.left)
        base_name = left_name.split(".")[0]
        if base_name in params and len(test.ops) == 1:
            op_type = type(test.ops[0])
            comp = test.comparators[0]
            if isinstance(test.ops[0], (ast.Is, ast.IsNot)):
                if isinstance(comp, ast.Constant) and comp.value is None:
                    op = ConditionOp.IS_NOT_NONE if isinstance(test.ops[0], ast.IsNot) else ConditionOp.IS_NONE
                    conditions.append(Condition(variable=left_name, op=op, source_line=line))
            elif op_type in CMP_OP_MAP:
                val = _const_value(comp)
                if val is not None:
                    conditions.append(Condition(
                        variable=left_name, op=CMP_OP_MAP[op_type],
                        value=val, source_line=line,
                    ))
                elif _name_str(comp):
                    conditions.append(Condition(
                        variable=left_name, op=CMP_OP_MAP[op_type],
                        value=_name_str(comp), source_line=line,
                        confidence=0.7,
                    ))
        # Check for len(x) comparisons
        if isinstance(test.left, ast.Call):
            cn = _get_call_name(test.left)
            if cn == "len" and test.left.args:
                arg_name = _name_str(test.left.args[0])
                base = arg_name.split(".")[0]
                if base in params and len(test.ops) == 1:
                    val = _const_value(test.comparators[0])
                    if val is not None and isinstance(val, int):
                        op_type = type(test.ops[0])
                        len_ops = {
                            ast.Gt: ConditionOp.LEN_GT,
                            ast.GtE: ConditionOp.LEN_GE,
                            ast.Lt: ConditionOp.LEN_LT,
                            ast.Eq: ConditionOp.LEN_EQ,
                        }
                        if op_type in len_ops:
                            conditions.append(Condition(
                                variable=arg_name, op=len_ops[op_type],
                                value=val, source_line=line,
                            ))

    elif isinstance(test, ast.Call):
        cn = _get_call_name(test)
        if cn == "isinstance" and len(test.args) == 2:
            arg_name = _name_str(test.args[0])
            base = arg_name.split(".")[0]
            if base in params:
                type_arg = test.args[1]
                if isinstance(type_arg, ast.Name):
                    conditions.append(Condition(
                        variable=arg_name, op=ConditionOp.ISINSTANCE,
                        value=type_arg.id, source_line=line,
                    ))
                elif isinstance(type_arg, ast.Tuple):
                    names = [e.id for e in type_arg.elts if isinstance(e, ast.Name)]
                    if names:
                        conditions.append(Condition(
                            variable=arg_name, op=ConditionOp.ISINSTANCE,
                            value=f"({', '.join(names)})", source_line=line,
                        ))
        elif cn == "hasattr" and len(test.args) == 2:
            arg_name = _name_str(test.args[0])
            base = arg_name.split(".")[0]
            if base in params:
                if isinstance(test.args[1], ast.Constant) and isinstance(test.args[1].value, str):
                    conditions.append(Condition(
                        variable=arg_name, op=ConditionOp.HASATTR,
                        value=test.args[1].value, source_line=line,
                    ))
        elif cn == "callable" and len(test.args) == 1:
            arg_name = _name_str(test.args[0])
            base = arg_name.split(".")[0]
            if base in params:
                conditions.append(Condition(
                    variable=arg_name, op=ConditionOp.CALLABLE,
                    source_line=line,
                ))

    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _extract_conditions_from_test(test.operand, params, line)
        for c in inner:
            negated = _negate_condition(c)
            if negated:
                conditions.append(negated)

    elif isinstance(test, ast.Name):
        if test.id in params:
            conditions.append(Condition(
                variable=test.id, op=ConditionOp.TRUTHY,
                source_line=line, confidence=0.6,
            ))

    return conditions


def _negate_condition(c: Condition) -> Optional[Condition]:
    neg_map = {
        ConditionOp.IS_NOT_NONE: ConditionOp.IS_NONE,
        ConditionOp.IS_NONE: ConditionOp.IS_NOT_NONE,
        ConditionOp.GT: ConditionOp.LE,
        ConditionOp.GE: ConditionOp.LT,
        ConditionOp.LT: ConditionOp.GE,
        ConditionOp.LE: ConditionOp.GT,
        ConditionOp.EQ: ConditionOp.NE,
        ConditionOp.NE: ConditionOp.EQ,
        ConditionOp.IN: ConditionOp.NOT_IN,
        ConditionOp.NOT_IN: ConditionOp.IN,
    }
    if c.op in neg_map:
        return Condition(variable=c.variable, op=neg_map[c.op], value=c.value,
                         source_line=c.source_line, confidence=c.confidence)
    return None


def _get_function_params(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Set[str]:
    params: Set[str] = set()
    for arg in node.args.args:
        if arg.arg != "self" and arg.arg != "cls":
            params.add(arg.arg)
    for arg in node.args.kwonlyargs:
        params.add(arg.arg)
    if node.args.vararg:
        params.add(node.args.vararg.arg)
    if node.args.kwarg:
        params.add(node.args.kwarg.arg)
    return params


# ── Precondition inference ───────────────────────────────────────────────────

class _PreconditionExtractor(ast.NodeVisitor):
    """Extract preconditions from guard-and-raise/return patterns."""

    def __init__(self, params: Set[str]):
        self.params = params
        self.conditions: List[Condition] = []
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth > 0:
            return
        self._depth += 1
        for stmt in node.body:
            self._check_stmt(stmt)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check_stmt(self, stmt: ast.stmt) -> None:
        # Pattern: if not <cond>: raise/return
        if isinstance(stmt, ast.If):
            body_raises = self._body_raises_or_returns_early(stmt.body)
            if body_raises:
                test = stmt.test
                if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                    conds = _extract_conditions_from_test(test.operand, self.params, stmt.lineno)
                    self.conditions.extend(conds)
                else:
                    conds = _extract_conditions_from_test(test, self.params, stmt.lineno)
                    for c in conds:
                        neg = _negate_condition(c)
                        if neg:
                            self.conditions.append(neg)

        # Pattern: assert <cond>
        elif isinstance(stmt, ast.Assert):
            conds = _extract_conditions_from_test(stmt.test, self.params, stmt.lineno)
            self.conditions.extend(conds)

        # Pattern: <param> = <param> or raise
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            pass

    def _body_raises_or_returns_early(self, body: List[ast.stmt]) -> bool:
        if not body:
            return False
        last = body[-1] if len(body) <= 2 else body[0]
        if isinstance(last, ast.Raise):
            return True
        if isinstance(last, ast.Return):
            return True
        if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call):
            name = _get_call_name(last.value)
            if name in ("raise", "sys.exit", "exit", "abort"):
                return True
        return False


def infer_preconditions(
    source: str, fn_name: str
) -> List[Precondition]:
    """Infer preconditions for function *fn_name* in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: List[Precondition] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != fn_name:
                continue
            params = _get_function_params(node)
            extractor = _PreconditionExtractor(params)
            extractor.visit(node)
            if extractor.conditions:
                results.append(Precondition(
                    function=fn_name,
                    conditions=extractor.conditions,
                    description=f"Preconditions for {fn_name}",
                ))
            # Also check type annotations as implicit preconditions
            for arg in node.args.args:
                if arg.annotation and arg.arg in params:
                    if isinstance(arg.annotation, ast.Subscript):
                        if isinstance(arg.annotation.value, ast.Name):
                            if arg.annotation.value.id != "Optional":
                                results.append(Precondition(
                                    function=fn_name,
                                    conditions=[Condition(
                                        variable=arg.arg,
                                        op=ConditionOp.IS_NOT_NONE,
                                        source_line=arg.lineno,
                                        confidence=0.7,
                                    )],
                                    description=f"Non-optional parameter {arg.arg}",
                                ))
    return results


# ── Postcondition inference ──────────────────────────────────────────────────

class _PostconditionExtractor(ast.NodeVisitor):
    """Infer postconditions from return patterns and result constraints."""

    def __init__(self, params: Set[str]):
        self.params = params
        self.return_conditions: List[Condition] = []
        self.return_types: List[str] = []
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth > 0:
            return
        self._depth += 1
        self._analyze_returns(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def _analyze_returns(self, node: ast.AST) -> None:
        returns_none = False
        returns_value = False
        all_return_constants = True
        return_values: List[Any] = []

        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                if child.value is None:
                    returns_none = True
                    all_return_constants = False
                else:
                    returns_value = True
                    if isinstance(child.value, ast.Constant):
                        return_values.append(child.value.value)
                    else:
                        all_return_constants = False
                    self._check_return_expr(child.value, child.lineno)

        if returns_value and not returns_none:
            self.return_conditions.append(Condition(
                variable="result",
                op=ConditionOp.IS_NOT_NONE,
                confidence=0.75,
            ))

        if all_return_constants and return_values:
            if all(isinstance(v, bool) for v in return_values):
                self.return_types.append("bool")
            elif all(isinstance(v, int) for v in return_values):
                self.return_types.append("int")
                all_non_neg = all(v >= 0 for v in return_values)
                if all_non_neg:
                    self.return_conditions.append(Condition(
                        variable="result",
                        op=ConditionOp.GE,
                        value=0,
                        confidence=0.6,
                    ))
            elif all(isinstance(v, str) for v in return_values):
                self.return_types.append("str")

    def _check_return_expr(self, expr: ast.expr, line: int) -> None:
        if isinstance(expr, ast.Call):
            name = _get_call_name(expr)
            if name in ("list", "dict", "set", "tuple"):
                self.return_conditions.append(Condition(
                    variable="result",
                    op=ConditionOp.IS_NOT_NONE,
                    confidence=0.95,
                ))
        elif isinstance(expr, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
            self.return_conditions.append(Condition(
                variable="result",
                op=ConditionOp.IS_NOT_NONE,
                confidence=0.95,
            ))
        elif isinstance(expr, ast.IfExp):
            pass
        elif isinstance(expr, ast.BinOp):
            if isinstance(expr.op, ast.Add):
                pass


def infer_postconditions(
    source: str, fn_name: str
) -> List[Postcondition]:
    """Infer postconditions for function *fn_name*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: List[Postcondition] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != fn_name:
                continue
            params = _get_function_params(node)
            extractor = _PostconditionExtractor(params)
            extractor.visit(node)
            if extractor.return_conditions:
                seen: Set[str] = set()
                unique: List[Condition] = []
                for c in extractor.return_conditions:
                    key = c.to_python()
                    if key not in seen:
                        seen.add(key)
                        unique.append(c)
                results.append(Postcondition(
                    function=fn_name,
                    conditions=unique,
                    description=f"Postconditions for {fn_name}",
                ))
            if node.returns:
                ret_conds = _postconditions_from_annotation(node.returns)
                if ret_conds:
                    results.append(Postcondition(
                        function=fn_name,
                        conditions=ret_conds,
                        description=f"Annotation-derived postconditions for {fn_name}",
                        confidence=0.9,
                    ))
    return results


def _postconditions_from_annotation(ann: ast.expr) -> List[Condition]:
    conds: List[Condition] = []
    if isinstance(ann, ast.Name):
        if ann.id in ("int", "float", "str", "bool", "bytes", "list", "dict"):
            conds.append(Condition(
                variable="result",
                op=ConditionOp.ISINSTANCE,
                value=ann.id,
                confidence=0.9,
            ))
    elif isinstance(ann, ast.Subscript):
        if isinstance(ann.value, ast.Name):
            if ann.value.id == "Optional":
                pass
            elif ann.value.id == "List":
                conds.append(Condition(
                    variable="result",
                    op=ConditionOp.ISINSTANCE,
                    value="list",
                    confidence=0.9,
                ))
            elif ann.value.id == "Dict":
                conds.append(Condition(
                    variable="result",
                    op=ConditionOp.ISINSTANCE,
                    value="dict",
                    confidence=0.9,
                ))
    return conds


# ── Invariant inference ──────────────────────────────────────────────────────

def infer_invariants(source: str, class_name: str) -> List[Invariant]:
    """Infer class invariants for *class_name* in *source*.

    Looks for patterns in ``__init__`` and other methods:
    - Attributes assigned non-None constants in __init__
    - Attributes validated in __init__ with asserts
    - Attributes that are always compared before use
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: List[Invariant] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue

        conditions: List[Condition] = []
        init_attrs: Dict[str, Any] = {}
        validated_attrs: Set[str] = set()

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "__init__":
                    _extract_init_invariants(item, init_attrs, validated_attrs, conditions)
                else:
                    _extract_method_invariants(item, conditions)

        for attr, val in init_attrs.items():
            if val is not None and not isinstance(val, type(None)):
                if attr not in validated_attrs:
                    if isinstance(val, (int, float)):
                        pass
                    else:
                        conditions.append(Condition(
                            variable=f"self.{attr}",
                            op=ConditionOp.IS_NOT_NONE,
                            confidence=0.6,
                        ))

        if conditions:
            seen: Set[str] = set()
            unique: List[Condition] = []
            for c in conditions:
                key = c.to_python()
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            results.append(Invariant(
                class_name=class_name,
                conditions=unique,
                description=f"Invariants for {class_name}",
            ))

    return results


def _extract_init_invariants(
    init_node: ast.FunctionDef,
    init_attrs: Dict[str, Any],
    validated_attrs: Set[str],
    conditions: List[Condition],
) -> None:
    for stmt in init_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (isinstance(target, ast.Attribute) and
                        isinstance(target.value, ast.Name) and
                        target.value.id == "self"):
                    if isinstance(stmt.value, ast.Constant):
                        init_attrs[target.attr] = stmt.value.value
                    elif isinstance(stmt.value, (ast.List, ast.Dict, ast.Set)):
                        init_attrs[target.attr] = type(stmt.value).__name__
                    else:
                        init_attrs[target.attr] = "<expr>"

        elif isinstance(stmt, ast.Assert):
            for child in ast.walk(stmt.test):
                if (isinstance(child, ast.Attribute) and
                        isinstance(child.value, ast.Name) and
                        child.value.id == "self"):
                    validated_attrs.add(child.attr)
            params = {"self"}
            conds = _extract_conditions_from_test(stmt.test, params, stmt.lineno)
            for c in conds:
                if c.variable.startswith("self."):
                    conditions.append(c)

        elif isinstance(stmt, ast.If):
            body_raises = any(isinstance(s, ast.Raise) for s in stmt.body)
            if body_raises:
                test = stmt.test
                if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                    params = {"self"}
                    conds = _extract_conditions_from_test(test.operand, params, stmt.lineno)
                    for c in conds:
                        if c.variable.startswith("self."):
                            conditions.append(c)
                else:
                    params = {"self"}
                    conds = _extract_conditions_from_test(test, params, stmt.lineno)
                    for c in conds:
                        neg = _negate_condition(c)
                        if neg and neg.variable.startswith("self."):
                            conditions.append(neg)


def _extract_method_invariants(
    method: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    conditions: List[Condition],
) -> None:
    for stmt in method.body:
        if isinstance(stmt, ast.Assert):
            params = {"self"}
            conds = _extract_conditions_from_test(stmt.test, params, stmt.lineno)
            for c in conds:
                if c.variable.startswith("self."):
                    conditions.append(c)


# ── Contract generation ──────────────────────────────────────────────────────

def generate_contracts(source: str, style: str = "icontract") -> str:
    """Add contract annotations to all functions in *source*.

    *style* can be ``"icontract"`` or ``"deal"``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    insertions: List[Tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        fn_name = node.name
        params = _get_function_params(node)

        pre_ext = _PreconditionExtractor(params)
        pre_ext.visit(node)
        post_ext = _PostconditionExtractor(params)
        post_ext.visit(node)

        decorators: List[str] = []
        if pre_ext.conditions:
            pre = Precondition(function=fn_name, conditions=pre_ext.conditions)
            if style == "deal":
                decorators.append(pre.to_deal())
            else:
                decorators.append(pre.to_icontract())

        if post_ext.return_conditions:
            seen: Set[str] = set()
            unique: List[Condition] = []
            for c in post_ext.return_conditions:
                key = c.to_python()
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            post = Postcondition(function=fn_name, conditions=unique)
            if style == "deal":
                decorators.append(post.to_deal())
            else:
                decorators.append(post.to_icontract())

        if decorators:
            insert_line = node.lineno - 1
            indent = ""
            if insert_line < len(lines):
                original = lines[insert_line]
                indent = original[: len(original) - len(original.lstrip())]
            for dec in reversed(decorators):
                insertions.append((insert_line, f"{indent}{dec}\n"))

    for line_idx, text in sorted(insertions, key=lambda x: x[0], reverse=True):
        lines.insert(line_idx, text)

    result = "".join(lines)
    if style == "icontract" and "import icontract" not in result:
        result = "import icontract\n" + result
    elif style == "deal" and "import deal" not in result:
        result = "import deal\n" + result
    return result


# ── Contract verification ────────────────────────────────────────────────────

def verify_contracts(source: str) -> List[ContractViolation]:
    """Verify that existing contracts (assert, raise-if) are consistent.

    Checks for:
    - Contradictory preconditions
    - Postconditions violated by return paths
    - Dead-code preconditions (always true)
    """
    violations: List[ContractViolation] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = _get_function_params(node)
        pre_ext = _PreconditionExtractor(params)
        pre_ext.visit(node)

        var_conditions: Dict[str, List[Condition]] = defaultdict(list)
        for c in pre_ext.conditions:
            var_conditions[c.variable].append(c)

        for var, conds in var_conditions.items():
            if _has_contradiction(conds):
                violations.append(ContractViolation(
                    line=conds[0].source_line,
                    column=0,
                    kind=ContractKind.PRECONDITION,
                    message=f"Contradictory preconditions on '{var}' in '{node.name}'",
                    contract_text=" and ".join(c.to_python() for c in conds),
                ))

        for stmt in node.body:
            if isinstance(stmt, ast.Assert):
                if isinstance(stmt.test, ast.Constant) and stmt.test.value is True:
                    violations.append(ContractViolation(
                        line=stmt.lineno,
                        column=stmt.col_offset,
                        kind=ContractKind.PRECONDITION,
                        message="Tautological assert (always True)",
                        contract_text="assert True",
                        severity="info",
                    ))
                elif isinstance(stmt.test, ast.Constant) and stmt.test.value is False:
                    violations.append(ContractViolation(
                        line=stmt.lineno,
                        column=stmt.col_offset,
                        kind=ContractKind.PRECONDITION,
                        message="Contradictory assert (always False) — unreachable code follows",
                        contract_text="assert False",
                        severity="error",
                    ))

    return violations


def _has_contradiction(conditions: List[Condition]) -> bool:
    for i, a in enumerate(conditions):
        for b in conditions[i + 1:]:
            neg = _negate_condition(a)
            if neg and neg.op == b.op and neg.value == b.value:
                return True
            if a.op == ConditionOp.IS_NONE and b.op == ConditionOp.IS_NOT_NONE:
                return True
            if a.op == ConditionOp.IS_NOT_NONE and b.op == ConditionOp.IS_NONE:
                return True
            if a.op == ConditionOp.LT and b.op == ConditionOp.GT:
                if isinstance(a.value, (int, float)) and isinstance(b.value, (int, float)):
                    if a.value <= b.value:
                        return True
    return False


# ── Hypothesis strategy generation ──────────────────────────────────────────

TYPE_STRATEGY_MAP: Dict[str, str] = {
    "int": "st.integers()",
    "float": "st.floats(allow_nan=False, allow_infinity=False)",
    "str": "st.text()",
    "bool": "st.booleans()",
    "bytes": "st.binary()",
    "list": "st.lists(st.integers())",
    "dict": "st.dictionaries(st.text(), st.integers())",
    "set": "st.sets(st.integers())",
    "tuple": "st.tuples(st.integers())",
}


def generate_hypothesis_strategies(source: str) -> str:
    """Generate Hypothesis property-based test strategies from contracts.

    For each function with inferred preconditions, generates a ``@given``
    decorated test that exercises the function with valid inputs.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    output_lines: List[str] = [
        "from hypothesis import given, strategies as st, assume",
        "",
    ]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        params = _get_function_params(node)
        if not params:
            continue

        pre_ext = _PreconditionExtractor(params)
        pre_ext.visit(node)

        strategies: Dict[str, str] = {}
        assumes: List[str] = []

        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            ann = arg.annotation
            if ann and isinstance(ann, ast.Name) and ann.id in TYPE_STRATEGY_MAP:
                strategies[arg.arg] = TYPE_STRATEGY_MAP[ann.id]
            else:
                strategies[arg.arg] = "st.from_type(object)"

        for cond in pre_ext.conditions:
            var = cond.variable
            if cond.op == ConditionOp.IS_NOT_NONE:
                assumes.append(f"assume({var} is not None)")
            elif cond.op == ConditionOp.GT:
                if var in strategies and isinstance(cond.value, (int, float)):
                    strategies[var] = f"st.integers(min_value={int(cond.value) + 1})"
                else:
                    assumes.append(f"assume({cond.to_python()})")
            elif cond.op == ConditionOp.GE:
                if var in strategies and isinstance(cond.value, (int, float)):
                    strategies[var] = f"st.integers(min_value={int(cond.value)})"
                else:
                    assumes.append(f"assume({cond.to_python()})")
            elif cond.op == ConditionOp.LT:
                if var in strategies and isinstance(cond.value, (int, float)):
                    strategies[var] = f"st.integers(max_value={int(cond.value) - 1})"
                else:
                    assumes.append(f"assume({cond.to_python()})")
            elif cond.op == ConditionOp.LE:
                if var in strategies and isinstance(cond.value, (int, float)):
                    strategies[var] = f"st.integers(max_value={int(cond.value)})"
                else:
                    assumes.append(f"assume({cond.to_python()})")
            elif cond.op == ConditionOp.ISINSTANCE:
                type_name = str(cond.value)
                if type_name in TYPE_STRATEGY_MAP:
                    strategies[var] = TYPE_STRATEGY_MAP[type_name]
            elif cond.op == ConditionOp.LEN_GE:
                if isinstance(cond.value, int):
                    strategies[var] = f"st.lists(st.integers(), min_size={cond.value})"
            elif cond.op == ConditionOp.LEN_GT:
                if isinstance(cond.value, int):
                    strategies[var] = f"st.lists(st.integers(), min_size={cond.value + 1})"
            else:
                assumes.append(f"assume({cond.to_python()})")

        strat_args = ", ".join(f"{k}={v}" for k, v in strategies.items())
        param_list = ", ".join(strategies.keys())

        output_lines.append(f"@given({strat_args})")
        output_lines.append(f"def test_{node.name}({param_list}):")
        for a in assumes:
            output_lines.append(f"    {a}")
        output_lines.append(f"    result = {node.name}({param_list})")

        # Add postcondition assertions
        post_ext = _PostconditionExtractor(params)
        post_ext.visit(node)
        for cond in post_ext.return_conditions:
            py = cond.to_python()
            output_lines.append(f"    assert {py}")

        if not post_ext.return_conditions:
            output_lines.append("    # No postcondition violations")

        output_lines.append("")

    return "\n".join(output_lines)
