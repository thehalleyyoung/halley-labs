"""Pretty-printer: AST → TLA-lite source text.

Traverses the AST using the visitor pattern and emits properly formatted,
valid TLA-lite syntax.  Parenthesises only when required by operator
precedence.
"""

from __future__ import annotations

from io import StringIO
from typing import Dict, List, Optional, Set

from .ast_nodes import (
    AlwaysExpr,
    ASTNode,
    ASTVisitor,
    Assumption,
    BoolLiteral,
    CaseArm,
    CaseExpr,
    ChooseExpr,
    ConstantDecl,
    Definition,
    DomainExpr,
    EventuallyExpr,
    ExceptExpr,
    Expression,
    FairnessExpr,
    FunctionApplication,
    FunctionConstruction,
    FunctionDef,
    Identifier,
    IfThenElse,
    InstanceDef,
    IntLiteral,
    LeadsToExpr,
    LetIn,
    Module,
    Operator,
    OperatorApplication,
    OperatorDef,
    PrimedIdentifier,
    QuantifiedExpr,
    RecordAccess,
    RecordConstruction,
    SequenceLiteral,
    SetComprehension,
    SetEnumeration,
    StringLiteral,
    StutteringAction,
    TemporalExistsExpr,
    TemporalForallExpr,
    Theorem,
    TupleLiteral,
    UnchangedExpr,
    VariableDecl,
)

# Precedence table for deciding when to parenthesise
_OP_PREC: Dict[Operator, int] = {
    Operator.EQUIV: 1,
    Operator.IMPLIES: 2,
    Operator.LEADS_TO: 3,
    Operator.LOR: 4,
    Operator.LAND: 5,
    Operator.EQ: 6, Operator.NEQ: 6,
    Operator.LT: 6, Operator.GT: 6,
    Operator.LEQ: 6, Operator.GEQ: 6,
    Operator.IN: 6, Operator.NOTIN: 6,
    Operator.SUBSETEQ: 6,
    Operator.UNION: 9, Operator.INTERSECT: 9, Operator.SETDIFF: 9,
    Operator.RANGE: 10,
    Operator.PLUS: 11, Operator.MINUS: 11,
    Operator.TIMES: 13, Operator.DIV: 13, Operator.MOD: 13,
    Operator.CROSS: 14,
    Operator.COLON_GT: 15, Operator.AT_AT: 15,
}

_BINARY_SYM: Dict[Operator, str] = {
    Operator.LAND: "/\\",
    Operator.LOR: "\\/",
    Operator.IMPLIES: "=>",
    Operator.EQUIV: "<=>",
    Operator.EQ: "=",
    Operator.NEQ: "/=",
    Operator.LT: "<",
    Operator.GT: ">",
    Operator.LEQ: "<=",
    Operator.GEQ: ">=",
    Operator.PLUS: "+",
    Operator.MINUS: "-",
    Operator.TIMES: "*",
    Operator.DIV: "\\div",
    Operator.MOD: "%",
    Operator.RANGE: "..",
    Operator.IN: "\\in",
    Operator.NOTIN: "\\notin",
    Operator.UNION: "\\union",
    Operator.INTERSECT: "\\intersect",
    Operator.SETDIFF: "\\",
    Operator.SUBSETEQ: "\\subseteq",
    Operator.CROSS: "\\X",
    Operator.COLON_GT: ":>",
    Operator.AT_AT: "@@",
    Operator.LEADS_TO: "~>",
}

_UNARY_SYM: Dict[Operator, str] = {
    Operator.LNOT: "~",
    Operator.UMINUS: "-",
    Operator.POWERSET: "SUBSET ",
    Operator.UNION_ALL: "UNION ",
    Operator.ENABLED_OP: "ENABLED ",
    Operator.UNCHANGED_OP: "UNCHANGED ",
    Operator.ALWAYS: "[]",
    Operator.EVENTUALLY: "<>",
    Operator.PRIME: "'",
}


def _needs_parens(child: Expression, parent_prec: int) -> bool:
    """Return True if *child* needs parentheses in a context with *parent_prec*."""
    if isinstance(child, OperatorApplication) and child.operator in _OP_PREC:
        return _OP_PREC[child.operator] < parent_prec
    return False


class PrettyPrinter(ASTVisitor[str]):
    """Visitor that converts an AST back into TLA-lite text."""

    def __init__(self, indent_width: int = 2) -> None:
        self._indent = 0
        self._indent_width = indent_width

    def pretty(self, node: ASTNode) -> str:
        return node.accept(self)

    def _ind(self) -> str:
        return " " * (self._indent * self._indent_width)

    # ── Module ──────────────────────────────────────────────────────

    def visit_Module(self, node: Module) -> str:
        parts: List[str] = []
        sep = "-" * max(4, len(node.name) + 12)
        parts.append(f"{sep} MODULE {node.name} {sep}")

        if node.extends:
            parts.append(f"EXTENDS {', '.join(node.extends)}")
        parts.append("")

        for c in node.constants:
            parts.append(c.accept(self))
        for v in node.variables:
            parts.append(v.accept(self))
        if node.constants or node.variables:
            parts.append("")

        for inst in node.instances:
            parts.append(inst.accept(self))

        for a in node.assumptions:
            parts.append(a.accept(self))

        for d in node.definitions:
            parts.append(d.accept(self))
            parts.append("")

        for t in node.theorems:
            parts.append(t.accept(self))

        parts.append("=" * len(sep) * 2)
        return "\n".join(parts)

    # ── Declarations ────────────────────────────────────────────────

    def visit_VariableDecl(self, node: VariableDecl) -> str:
        kw = "VARIABLES" if len(node.names) > 1 else "VARIABLE"
        return f"{kw} {', '.join(node.names)}"

    def visit_ConstantDecl(self, node: ConstantDecl) -> str:
        kw = "CONSTANTS" if len(node.names) > 1 else "CONSTANT"
        return f"{kw} {', '.join(node.names)}"

    def visit_Assumption(self, node: Assumption) -> str:
        return f"ASSUME {node.expr.accept(self)}" if node.expr else "ASSUME ???"

    def visit_Theorem(self, node: Theorem) -> str:
        return f"THEOREM {node.expr.accept(self)}" if node.expr else "THEOREM ???"

    def visit_InstanceDef(self, node: InstanceDef) -> str:
        prefix = "LOCAL " if node.is_local else ""
        s = f"{prefix}INSTANCE {node.module_name}"
        if node.substitutions:
            subs = ", ".join(
                f"{n} <- {e.accept(self)}" for n, e in node.substitutions
            )
            s += f" WITH {subs}"
        return s

    # ── Definitions ─────────────────────────────────────────────────

    def visit_OperatorDef(self, node: OperatorDef) -> str:
        prefix = "LOCAL " if node.is_local else ""
        params = f"({', '.join(node.params)})" if node.params else ""
        body = node.body.accept(self) if node.body else "???"
        return f"{prefix}{node.name}{params} == {body}"

    def visit_FunctionDef(self, node: FunctionDef) -> str:
        prefix = "LOCAL " if node.is_local else ""
        s_set = node.set_expr.accept(self) if node.set_expr else "???"
        body = node.body.accept(self) if node.body else "???"
        return f"{prefix}{node.name}[{node.variable} \\in {s_set}] == {body}"

    # ── Literals ────────────────────────────────────────────────────

    def visit_IntLiteral(self, node: IntLiteral) -> str:
        return str(node.value)

    def visit_BoolLiteral(self, node: BoolLiteral) -> str:
        return "TRUE" if node.value else "FALSE"

    def visit_StringLiteral(self, node: StringLiteral) -> str:
        escaped = (
            node.value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'

    def visit_Identifier(self, node: Identifier) -> str:
        return node.name

    def visit_PrimedIdentifier(self, node: PrimedIdentifier) -> str:
        return f"{node.name}'"

    # ── Operator application ────────────────────────────────────────

    def visit_OperatorApplication(self, node: OperatorApplication) -> str:
        op = node.operator

        # User-defined operator call
        if node.operator_name and op == Operator.FUNC_APPLY:
            args = ", ".join(a.accept(self) for a in node.operands)
            return f"{node.operator_name}({args})"

        # Built-in function-style calls
        if node.operator_name and op in (
            Operator.APPEND, Operator.HEAD, Operator.TAIL,
            Operator.LEN, Operator.SUBSEQ, Operator.SEQ,
        ):
            args = ", ".join(a.accept(self) for a in node.operands)
            return f"{node.operator_name}({args})"

        # Unary prefix
        if len(node.operands) == 1 and op in _UNARY_SYM:
            sym = _UNARY_SYM[op]
            operand_str = node.operands[0].accept(self)
            if op == Operator.PRIME:
                return f"{operand_str}'"
            return f"{sym}{operand_str}"

        # Binary infix
        if len(node.operands) == 2 and op in _BINARY_SYM:
            sym = _BINARY_SYM[op]
            prec = _OP_PREC.get(op, 0)
            left = node.operands[0]
            right = node.operands[1]
            ls = left.accept(self)
            rs = right.accept(self)
            if _needs_parens(left, prec):
                ls = f"({ls})"
            if _needs_parens(right, prec + 1):
                rs = f"({rs})"
            return f"{ls} {sym} {rs}"

        # Fallback
        args = ", ".join(a.accept(self) for a in node.operands)
        return f"{op.name}({args})"

    # ── Set expressions ─────────────────────────────────────────────

    def visit_SetEnumeration(self, node: SetEnumeration) -> str:
        elems = ", ".join(e.accept(self) for e in node.elements)
        return f"{{{elems}}}"

    def visit_SetComprehension(self, node: SetComprehension) -> str:
        if node.map_expr:
            e = node.map_expr.accept(self)
            s = node.set_expr.accept(self) if node.set_expr else "???"
            return f"{{{e} : {node.variable} \\in {s}}}"
        s = node.set_expr.accept(self) if node.set_expr else "???"
        p = node.predicate.accept(self) if node.predicate else "???"
        return f"{{{node.variable} \\in {s} : {p}}}"

    # ── Function / record ───────────────────────────────────────────

    def visit_FunctionConstruction(self, node: FunctionConstruction) -> str:
        s = node.set_expr.accept(self) if node.set_expr else "???"
        b = node.body.accept(self) if node.body else "???"
        return f"[{node.variable} \\in {s} |-> {b}]"

    def visit_FunctionApplication(self, node: FunctionApplication) -> str:
        f = node.function.accept(self) if node.function else "???"
        a = node.argument.accept(self) if node.argument else ""
        if a:
            return f"{f}[{a}]"
        return f"{f}[]"

    def visit_RecordConstruction(self, node: RecordConstruction) -> str:
        fields = ", ".join(
            f"{name} |-> {val.accept(self)}" for name, val in node.fields
        )
        return f"[{fields}]"

    def visit_RecordAccess(self, node: RecordAccess) -> str:
        r = node.record.accept(self) if node.record else "???"
        return f"{r}.{node.field_name}"

    # ── Tuple / Sequence ────────────────────────────────────────────

    def visit_TupleLiteral(self, node: TupleLiteral) -> str:
        elems = ", ".join(e.accept(self) for e in node.elements)
        return f"<<{elems}>>"

    def visit_SequenceLiteral(self, node: SequenceLiteral) -> str:
        elems = ", ".join(e.accept(self) for e in node.elements)
        return f"<<{elems}>>"

    # ── Quantified ──────────────────────────────────────────────────

    def visit_QuantifiedExpr(self, node: QuantifiedExpr) -> str:
        q = "\\A" if node.quantifier == "forall" else "\\E"
        bounds = ", ".join(
            f"{v} \\in {s.accept(self)}" for v, s in node.variables
        )
        body = node.body.accept(self) if node.body else "???"
        return f"{q} {bounds} : {body}"

    # ── Control flow ────────────────────────────────────────────────

    def visit_IfThenElse(self, node: IfThenElse) -> str:
        c = node.condition.accept(self) if node.condition else "???"
        t = node.then_expr.accept(self) if node.then_expr else "???"
        e = node.else_expr.accept(self) if node.else_expr else "???"
        return f"IF {c} THEN {t} ELSE {e}"

    def visit_LetIn(self, node: LetIn) -> str:
        self._indent += 1
        defs = ("\n" + self._ind()).join(d.accept(self) for d in node.definitions)
        self._indent -= 1
        body = node.body.accept(self) if node.body else "???"
        return f"LET {defs}\n{self._ind()}IN  {body}"

    def visit_CaseExpr(self, node: CaseExpr) -> str:
        parts: List[str] = []
        for i, arm in enumerate(node.arms):
            c = arm.condition.accept(self) if arm.condition else "???"
            v = arm.value.accept(self) if arm.value else "???"
            prefix = "CASE " if i == 0 else "  [] "
            parts.append(f"{prefix}{c} -> {v}")
        if node.other is not None:
            o = node.other.accept(self)
            parts.append(f"  [] OTHER -> {o}")
        return "\n".join(parts)

    def visit_ChooseExpr(self, node: ChooseExpr) -> str:
        s = ""
        if node.set_expr:
            s = f" \\in {node.set_expr.accept(self)}"
        p = node.predicate.accept(self) if node.predicate else "???"
        return f"CHOOSE {node.variable}{s} : {p}"

    # ── UNCHANGED / EXCEPT ──────────────────────────────────────────

    def visit_UnchangedExpr(self, node: UnchangedExpr) -> str:
        if len(node.variables) == 1:
            return f"UNCHANGED {node.variables[0].accept(self)}"
        elems = ", ".join(v.accept(self) for v in node.variables)
        return f"UNCHANGED <<{elems}>>"

    def visit_ExceptExpr(self, node: ExceptExpr) -> str:
        base = node.base.accept(self) if node.base else "???"
        subs: List[str] = []
        for path, val in node.substitutions:
            path_str = "".join(f"[{p.accept(self)}]" for p in path)
            subs.append(f"!{path_str} = {val.accept(self)}")
        return f"[{base} EXCEPT {', '.join(subs)}]"

    def visit_DomainExpr(self, node: DomainExpr) -> str:
        e = node.expr.accept(self) if node.expr else "???"
        return f"DOMAIN {e}"

    # ── Temporal / Action ───────────────────────────────────────────

    def visit_AlwaysExpr(self, node: AlwaysExpr) -> str:
        e = node.expr.accept(self) if node.expr else "???"
        return f"[]{e}"

    def visit_EventuallyExpr(self, node: EventuallyExpr) -> str:
        e = node.expr.accept(self) if node.expr else "???"
        return f"<>{e}"

    def visit_LeadsToExpr(self, node: LeadsToExpr) -> str:
        l = node.left.accept(self) if node.left else "???"
        r = node.right.accept(self) if node.right else "???"
        return f"{l} ~> {r}"

    def visit_TemporalForallExpr(self, node: TemporalForallExpr) -> str:
        b = node.body.accept(self) if node.body else "???"
        return f"\\AA {node.variable} : {b}"

    def visit_TemporalExistsExpr(self, node: TemporalExistsExpr) -> str:
        b = node.body.accept(self) if node.body else "???"
        return f"\\EE {node.variable} : {b}"

    def visit_StutteringAction(self, node: StutteringAction) -> str:
        a = node.action.accept(self) if node.action else "???"
        v = node.variables.accept(self) if node.variables else "???"
        if node.is_angle:
            return f"<<{a}>>_{v}"
        return f"[{a}]_{v}"

    def visit_FairnessExpr(self, node: FairnessExpr) -> str:
        v = node.variables.accept(self) if node.variables else "???"
        a = node.action.accept(self) if node.action else "???"
        return f"{node.kind}_{v}({a})"


# ============================================================================
# Convenience
# ============================================================================

def pretty_print(node: ASTNode, indent_width: int = 2) -> str:
    """Convert an AST node to a TLA-lite string."""
    return PrettyPrinter(indent_width=indent_width).pretty(node)
