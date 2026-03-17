"""
Pretty-printer and formatter for the RegSynth DSL.

Takes AST nodes produced by the parser and emits well-formatted DSL source
code with configurable indentation, alignment, and line-wrapping.
"""

from __future__ import annotations

import difflib
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from regsynth_py.dsl.ast_nodes import (
    Program,
    ObligationDecl,
    JurisdictionDecl,
    StrategyDecl,
    CompositionDecl,
    ConstraintDecl,
    BinaryOp,
    UnaryOp,
    Identifier,
    Literal,
    TemporalExpr,
    RiskLevel,
    ObligationType,
    FrameworkType,
    ComposeMode,
    ASTVisitor,
)


# ---------------------------------------------------------------------------
# Operator precedence (lower number = tighter binding)
# ---------------------------------------------------------------------------

_PRECEDENCE: dict[str, int] = {
    "OR": 10,
    "AND": 20,
    "IMPLIES": 15,
    "EQ": 30, "NEQ": 30,
    "LT": 40, "GT": 40, "LTE": 40, "GTE": 40,
    "PLUS": 50, "MINUS": 50,
    "STAR": 60, "SLASH": 60, "PERCENT": 60,
}

_OP_SYMBOLS: dict[str, str] = {
    "AND": "and",
    "OR": "or",
    "IMPLIES": "=>",
    "EQ": "==",
    "NEQ": "!=",
    "LT": "<",
    "GT": ">",
    "LTE": "<=",
    "GTE": ">=",
    "PLUS": "+",
    "MINUS": "-",
    "STAR": "*",
    "SLASH": "/",
    "PERCENT": "%",
    "NOT": "not",
    "REQUIRED": "required",
}

_DECL_SORT_ORDER: dict[type, int] = {
    JurisdictionDecl: 0,
    ObligationDecl: 1,
    StrategyDecl: 2,
    CompositionDecl: 3,
    ConstraintDecl: 4,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FormatterConfig:
    """Knobs that control the formatter's output style."""

    indent_size: int = 4
    max_line_width: int = 100
    blank_lines_between_decls: int = 1
    align_colons: bool = True
    sort_declarations: bool = False
    include_comments: bool = True


# ---------------------------------------------------------------------------
# Formatter (ASTVisitor)
# ---------------------------------------------------------------------------

class Formatter(ASTVisitor):
    """Converts an AST back into well-formatted DSL source text."""

    def __init__(self, config: Optional[FormatterConfig] = None) -> None:
        self.config = config or FormatterConfig()
        self._indent_str = " " * self.config.indent_size

    # -- public API ----------------------------------------------------------

    def format(self, program: Program) -> str:
        """Format a full *Program* AST into DSL source."""
        decls = list(program.declarations)
        if self.config.sort_declarations:
            decls.sort(key=lambda d: (
                _DECL_SORT_ORDER.get(type(d), 99),
                getattr(d, "name", ""),
            ))

        separator = "\n" * (1 + self.config.blank_lines_between_decls)
        parts = [self.format_declaration(d) for d in decls]
        return separator.join(parts) + "\n"

    def format_declaration(self, decl: Any) -> str:
        """Dispatch to the correct declaration formatter."""
        if isinstance(decl, ObligationDecl):
            return self.format_obligation(decl)
        if isinstance(decl, JurisdictionDecl):
            return self.format_jurisdiction(decl)
        if isinstance(decl, StrategyDecl):
            return self.format_strategy(decl)
        if isinstance(decl, CompositionDecl):
            return self.format_composition(decl)
        if isinstance(decl, ConstraintDecl):
            return self.format_constraint(decl)
        return repr(decl)

    # -- declaration formatters ----------------------------------------------

    def format_obligation(self, decl: ObligationDecl) -> str:
        entries: list[tuple[str, str]] = []
        if decl.jurisdiction is not None:
            entries.append(("jurisdiction", decl.jurisdiction))
        if decl.risk_level is not None:
            entries.append(("risk", decl.risk_level.value))
        if decl.category is not None:
            entries.append(("category", f'"{decl.category.value}"'))
        if decl.articles:
            entries.append(("articles", self.format_list(
                [f'"{a}"' for a in decl.articles])))
        if decl.temporal is not None:
            entries.append(("temporal", self.format_temporal(decl.temporal)))
        if decl.requirements:
            formatted_reqs = [self.format_expression(r) for r in decl.requirements]
            entries.append(("requirements", self.format_list(formatted_reqs)))
        for key, val in decl.metadata.items():
            entries.append((key, self._format_meta_value(val)))
        body = self.format_block(entries, align=self.config.align_colons)
        return f"obligation {decl.name} {{\n{body}\n}}"

    def format_jurisdiction(self, decl: JurisdictionDecl) -> str:
        entries: list[tuple[str, str]] = []
        if decl.framework_type is not None:
            entries.append(("framework", decl.framework_type.value))
        if decl.region is not None:
            entries.append(("region", f'"{decl.region}"'))
        if decl.enforcement_date is not None:
            entries.append(("enforcement_date", decl.enforcement_date))
        if decl.penalties:
            formatted = [self.format_expression(p) for p in decl.penalties]
            entries.append(("penalties", self.format_list(formatted)))
        body = self.format_block(entries, align=self.config.align_colons)
        return f"jurisdiction {decl.name} {{\n{body}\n}}"

    def format_strategy(self, decl: StrategyDecl) -> str:
        entries: list[tuple[str, str]] = []
        if decl.obligations:
            entries.append(("obligations", self.format_list(decl.obligations)))
        if decl.cost is not None:
            entries.append(("cost", self.format_expression(decl.cost)))
        if decl.timeline is not None:
            entries.append(("timeline", self.format_temporal(decl.timeline)))
        body = self.format_block(entries, align=self.config.align_colons)
        return f"strategy {decl.name} {{\n{body}\n}}"

    def format_composition(self, decl: CompositionDecl) -> str:
        entries: list[tuple[str, str]] = []
        if decl.strategies:
            entries.append(("strategies", self.format_list(decl.strategies)))
        entries.append(("mode", decl.mode.value))
        body = self.format_block(entries, align=self.config.align_colons)
        return f"composition {decl.name} {{\n{body}\n}}"

    def format_constraint(self, decl: ConstraintDecl) -> str:
        entries: list[tuple[str, str]] = []
        if decl.constraint_type is not None:
            entries.append(("type", f'"{decl.constraint_type}"'))
        for key, val in decl.parameters.items():
            entries.append((key, self.format_expression(val)))
        body = self.format_block(entries, align=self.config.align_colons)
        return f"constraint {decl.name} {{\n{body}\n}}"

    # -- expression formatters -----------------------------------------------

    def format_expression(self, expr: Any) -> str:
        """Format an arbitrary expression node."""
        if isinstance(expr, BinaryOp):
            return self.format_binary(expr)
        if isinstance(expr, UnaryOp):
            return self.format_unary(expr)
        if isinstance(expr, TemporalExpr):
            return self.format_temporal(expr)
        if isinstance(expr, Literal):
            return self.format_literal(expr)
        if isinstance(expr, Identifier):
            return expr.name
        if isinstance(expr, str):
            return expr
        return repr(expr)

    def format_binary(self, expr: BinaryOp) -> str:
        """Format a binary operation with minimal parenthesisation."""
        symbol = _OP_SYMBOLS.get(expr.op, expr.op)
        prec = _PRECEDENCE.get(expr.op, 0)

        left_str = self.format_expression(expr.left)
        if isinstance(expr.left, BinaryOp):
            left_prec = _PRECEDENCE.get(expr.left.op, 0)
            if left_prec < prec:
                left_str = f"({left_str})"

        right_str = self.format_expression(expr.right)
        if isinstance(expr.right, BinaryOp):
            right_prec = _PRECEDENCE.get(expr.right.op, 0)
            if right_prec <= prec:
                right_str = f"({right_str})"

        return f"{left_str} {symbol} {right_str}"

    def format_unary(self, expr: UnaryOp) -> str:
        """Format a unary operation."""
        symbol = _OP_SYMBOLS.get(expr.op, expr.op)
        operand_str = self.format_expression(expr.operand)
        if isinstance(expr.operand, BinaryOp):
            operand_str = f"({operand_str})"
        if expr.op == "MINUS":
            return f"-{operand_str}"
        return f"{symbol} {operand_str}"

    def format_temporal(self, expr: TemporalExpr) -> str:
        """Format a temporal expression."""
        parts = [expr.operator.lower()]
        if expr.deadline is not None:
            parts.append(expr.deadline)
        if expr.phase is not None:
            parts.append(f"phase {expr.phase}")
        if expr.recurrence is not None:
            parts.append(f"every {expr.recurrence}")
        return " ".join(parts)

    def format_literal(self, lit: Literal) -> str:
        """Format a literal with proper quoting."""
        if lit.literal_type == "string":
            escaped = str(lit.value).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if lit.literal_type == "bool":
            return "true" if lit.value else "false"
        if lit.literal_type == "date":
            return str(lit.value)
        return str(lit.value)

    # -- structural helpers --------------------------------------------------

    def format_list(self, items: Sequence[str],
                    multiline_threshold: int = 3) -> str:
        """Format a list, going multi-line when it exceeds *multiline_threshold*."""
        if not items:
            return "[]"

        single_line = "[" + ", ".join(items) + "]"
        if (len(items) <= multiline_threshold
                and len(single_line) <= self.config.max_line_width):
            return single_line

        inner = ",\n".join(self.indent(item, 1) for item in items)
        return f"[\n{inner}\n]"

    def format_block(self, entries: list[tuple[str, str]],
                     align: bool = True) -> str:
        """Format key-value pairs as indented block lines.

        When *align* is ``True`` colons are vertically aligned.
        """
        if not entries:
            return ""

        if align:
            max_key_len = max(len(k) for k, _ in entries)
        else:
            max_key_len = 0

        lines: list[str] = []
        for key, value in entries:
            if align:
                padded_key = key.ljust(max_key_len)
            else:
                padded_key = key
            raw_line = f"{padded_key}: {value}"
            wrapped = self.wrap_line(raw_line, self.config.max_line_width
                                     - self.config.indent_size)
            lines.append(self.indent(wrapped, 1))
        return "\n".join(lines)

    def indent(self, text: str, level: int) -> str:
        """Indent every line of *text* by *level* tab-stops."""
        prefix = self._indent_str * level
        indented_lines: list[str] = []
        for line in text.split("\n"):
            if line.strip():
                indented_lines.append(prefix + line)
            else:
                indented_lines.append("")
        return "\n".join(indented_lines)

    def wrap_line(self, text: str, width: int) -> str:
        """Soft-wrap a long line at word boundaries.

        Short lines are returned unchanged.  Subsequent continuation lines
        are indented by one extra indent level.
        """
        if len(text) <= width or width <= 0:
            return text

        continuation_indent = self._indent_str
        wrapped = textwrap.fill(
            text,
            width=width,
            subsequent_indent=continuation_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        return wrapped

    # -- visitor protocol (delegates to format helpers) ----------------------

    def visit_Program(self, node: Program) -> str:
        return self.format(node)

    def visit_ObligationDecl(self, node: ObligationDecl) -> str:
        return self.format_obligation(node)

    def visit_JurisdictionDecl(self, node: JurisdictionDecl) -> str:
        return self.format_jurisdiction(node)

    def visit_StrategyDecl(self, node: StrategyDecl) -> str:
        return self.format_strategy(node)

    def visit_CompositionDecl(self, node: CompositionDecl) -> str:
        return self.format_composition(node)

    def visit_ConstraintDecl(self, node: ConstraintDecl) -> str:
        return self.format_constraint(node)

    def visit_BinaryOp(self, node: BinaryOp) -> str:
        return self.format_binary(node)

    def visit_UnaryOp(self, node: UnaryOp) -> str:
        return self.format_unary(node)

    def visit_Identifier(self, node: Identifier) -> str:
        return node.name

    def visit_Literal(self, node: Literal) -> str:
        return self.format_literal(node)

    def visit_TemporalExpr(self, node: TemporalExpr) -> str:
        return self.format_temporal(node)

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _format_meta_value(value: Any) -> str:
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple)):
            inner = ", ".join(
                Formatter._format_meta_value(v) for v in value
            )
            return f"[{inner}]"
        return str(value)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def format_source(source: str, config: Optional[FormatterConfig] = None) -> str:
    """Parse *source* then pretty-print it back.

    This is a one-call convenience wrapper around the parser and formatter.
    """
    from regsynth_py.dsl.parser import parse  # local to avoid circular import
    program = parse(source)
    return Formatter(config).format(program)


def diff_format(original: str, formatted: str) -> str:
    """Return a unified-diff between *original* and *formatted* source."""
    orig_lines = original.splitlines(keepends=True)
    fmt_lines = formatted.splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        orig_lines,
        fmt_lines,
        fromfile="original",
        tofile="formatted",
        lineterm="",
    )
    return "".join(diff_lines)
