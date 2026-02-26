"""Recursive-descent parser for the TLA-lite fragment.

Converts a token stream produced by :mod:`lexer` into the typed AST defined
in :mod:`ast_nodes`.  Implements full operator-precedence climbing with
17 precedence levels matching TLA+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .ast_nodes import (
    AlwaysExpr,
    ASTNode,
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
    Property,
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
from .lexer import Lexer, LexerError
from .source_map import SourceLocation, UNKNOWN_LOCATION
from .tokens import Token, TokenKind


# ============================================================================
# Errors
# ============================================================================

class ParseError(Exception):
    """Raised when the parser encounters a syntax error."""

    def __init__(self, message: str, location: SourceLocation):
        self.location = location
        super().__init__(f"{location}: {message}")


# ============================================================================
# Operator precedence table  (higher number = tighter binding)
# ============================================================================

# TLA+ precedence levels (1 = lowest, 17 = highest)
_BINARY_PREC: Dict[TokenKind, Tuple[int, str]] = {
    # level 1  — equivalence
    TokenKind.EQUIV: (1, "left"),
    # level 2  — implication
    TokenKind.IMPLIES: (2, "right"),
    # level 3  — leads-to (temporal)
    TokenKind.LEADS_TO: (3, "left"),
    # level 4  — disjunction
    TokenKind.LOR: (4, "left"),
    # level 5  — conjunction
    TokenKind.LAND: (5, "left"),
    # level 6–8 — comparisons
    TokenKind.EQ: (6, "none"),
    TokenKind.NEQ: (6, "none"),
    TokenKind.LT: (6, "none"),
    TokenKind.GT: (6, "none"),
    TokenKind.LEQ: (6, "none"),
    TokenKind.GEQ: (6, "none"),
    TokenKind.SET_IN: (6, "none"),
    TokenKind.SET_NOTIN: (6, "none"),
    TokenKind.SET_SUBSETEQ: (6, "none"),
    # level 9  — set union / diff
    TokenKind.SET_UNION: (9, "left"),
    TokenKind.SET_INTER: (9, "left"),
    TokenKind.SET_DIFF: (9, "left"),
    # level 10 — range
    TokenKind.DOTDOT: (10, "none"),
    # level 11 — addition
    TokenKind.PLUS: (11, "left"),
    TokenKind.MINUS: (11, "left"),
    # level 13 — multiplication
    TokenKind.STAR: (13, "left"),
    TokenKind.DIV: (13, "left"),
    TokenKind.PERCENT: (13, "left"),
    # level 14 — cross product
    TokenKind.CROSS: (14, "none"),
    # level 15 — function operators
    TokenKind.COLON_GT: (15, "left"),
    TokenKind.AT_AT: (15, "left"),
}

_TOKEN_TO_OP: Dict[TokenKind, Operator] = {
    TokenKind.LAND: Operator.LAND,
    TokenKind.LOR: Operator.LOR,
    TokenKind.IMPLIES: Operator.IMPLIES,
    TokenKind.EQUIV: Operator.EQUIV,
    TokenKind.EQ: Operator.EQ,
    TokenKind.NEQ: Operator.NEQ,
    TokenKind.LT: Operator.LT,
    TokenKind.GT: Operator.GT,
    TokenKind.LEQ: Operator.LEQ,
    TokenKind.GEQ: Operator.GEQ,
    TokenKind.PLUS: Operator.PLUS,
    TokenKind.MINUS: Operator.MINUS,
    TokenKind.STAR: Operator.TIMES,
    TokenKind.DIV: Operator.DIV,
    TokenKind.PERCENT: Operator.MOD,
    TokenKind.DOTDOT: Operator.RANGE,
    TokenKind.SET_IN: Operator.IN,
    TokenKind.SET_NOTIN: Operator.NOTIN,
    TokenKind.SET_UNION: Operator.UNION,
    TokenKind.SET_INTER: Operator.INTERSECT,
    TokenKind.SET_DIFF: Operator.SETDIFF,
    TokenKind.SET_SUBSETEQ: Operator.SUBSETEQ,
    TokenKind.CROSS: Operator.CROSS,
    TokenKind.COLON_GT: Operator.COLON_GT,
    TokenKind.AT_AT: Operator.AT_AT,
    TokenKind.LEADS_TO: Operator.LEADS_TO,
}


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Recursive-descent parser for TLA-lite.

    Usage::

        parser = Parser(source_text, file="Spec.tla")
        module = parser.parse_module()
    """

    def __init__(self, source: str, file: str = "<string>") -> None:
        lexer = Lexer(source, file)
        self._tokens = lexer.tokenize()
        self._pos = 0
        self._file = file
        self._errors: List[ParseError] = []

    # ── Token helpers ───────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return self._tokens[-1]  # EOF

    def _at(self, *kinds: TokenKind) -> bool:
        return self._peek().kind in kinds

    def _advance(self) -> Token:
        tok = self._peek()
        if tok.kind != TokenKind.EOF:
            self._pos += 1
        return tok

    def _expect(self, kind: TokenKind, context: str = "") -> Token:
        tok = self._peek()
        if tok.kind != kind:
            ctx = f" in {context}" if context else ""
            self._raise(f"Expected {kind.name} but got {tok.kind.name}{ctx}", tok.location)
        return self._advance()

    def _match(self, *kinds: TokenKind) -> Optional[Token]:
        if self._peek().kind in kinds:
            return self._advance()
        return None

    def _raise(self, message: str, location: Optional[SourceLocation] = None) -> None:
        loc = location or self._peek().location
        raise ParseError(message, loc)

    def _loc(self) -> SourceLocation:
        return self._peek().location

    def _skip_newlines(self) -> None:
        while self._peek().kind == TokenKind.NEWLINE:
            self._advance()

    def _skip_separators(self) -> None:
        while self._peek().kind in (TokenKind.SEPARATOR, TokenKind.NEWLINE):
            self._advance()

    # ================================================================
    # Module parsing
    # ================================================================

    def parse_module(self) -> Module:
        """Parse a complete TLA-lite module."""
        self._skip_separators()

        # ---- MODULE Name ----
        start_loc = self._loc()
        if self._at(TokenKind.SEPARATOR):
            self._advance()
            self._skip_newlines()

        self._expect(TokenKind.MODULE, "module header")
        name_tok = self._expect(TokenKind.IDENTIFIER, "module name")
        module_name = name_tok.value

        self._skip_separators()

        module = Module(name=module_name, source_location=start_loc)

        # EXTENDS
        if self._at(TokenKind.EXTENDS):
            self._advance()
            module.extends = self._parse_identifier_list()

        # Body
        self._parse_module_body(module)

        module.source_location = start_loc.merge(self._loc())
        return module

    def _parse_identifier_list(self) -> List[str]:
        names: List[str] = []
        tok = self._expect(TokenKind.IDENTIFIER, "identifier list")
        names.append(tok.value)
        while self._match(TokenKind.COMMA):
            tok = self._expect(TokenKind.IDENTIFIER, "identifier list")
            names.append(tok.value)
        return names

    def _parse_module_body(self, module: Module) -> None:
        """Parse the body of a module until EOF or end separator."""
        while not self._at(TokenKind.EOF):
            self._skip_separators()
            if self._at(TokenKind.EOF):
                break

            tok = self._peek()
            kind = tok.kind

            if kind == TokenKind.SEPARATOR:
                self._advance()
                continue

            if kind in (TokenKind.VARIABLE, TokenKind.VARIABLES):
                module.variables.append(self._parse_variable_decl())
            elif kind in (TokenKind.CONSTANT, TokenKind.CONSTANTS):
                module.constants.append(self._parse_constant_decl())
            elif kind == TokenKind.ASSUME:
                module.assumptions.append(self._parse_assumption())
            elif kind == TokenKind.THEOREM:
                module.theorems.append(self._parse_theorem())
            elif kind == TokenKind.INSTANCE:
                module.instances.append(self._parse_instance())
            elif kind == TokenKind.LOCAL:
                defn = self._parse_local_definition()
                module.definitions.append(defn)
            elif kind == TokenKind.IDENTIFIER:
                defn = self._parse_definition()
                module.definitions.append(defn)
            else:
                # Skip unrecognised tokens
                self._advance()

    # ================================================================
    # Declaration parsing
    # ================================================================

    def _parse_variable_decl(self) -> VariableDecl:
        loc = self._loc()
        self._advance()  # VARIABLE(S)
        names = self._parse_identifier_list()
        return VariableDecl(names=names, source_location=loc)

    def _parse_constant_decl(self) -> ConstantDecl:
        loc = self._loc()
        self._advance()  # CONSTANT(S)
        names = self._parse_identifier_list()
        return ConstantDecl(names=names, source_location=loc)

    def _parse_assumption(self) -> Assumption:
        loc = self._loc()
        self._advance()  # ASSUME
        expr = self.parse_expression()
        return Assumption(expr=expr, source_location=loc)

    def _parse_theorem(self) -> Theorem:
        loc = self._loc()
        self._advance()  # THEOREM
        expr = self.parse_expression()
        return Theorem(expr=expr, source_location=loc)

    def _parse_instance(self) -> InstanceDef:
        loc = self._loc()
        self._advance()  # INSTANCE
        name = self._expect(TokenKind.IDENTIFIER, "INSTANCE module name").value
        subs: List[tuple[str, Expression]] = []
        if self._match(TokenKind.WITH):
            subs = self._parse_substitution_list()
        return InstanceDef(module_name=name, substitutions=subs, source_location=loc)

    def _parse_substitution_list(self) -> List[tuple[str, Expression]]:
        subs: List[tuple[str, Expression]] = []
        while True:
            name = self._expect(TokenKind.IDENTIFIER, "substitution").value
            self._expect(TokenKind.ASSIGN, "substitution '<-'")
            expr = self.parse_expression()
            subs.append((name, expr))
            if not self._match(TokenKind.COMMA):
                break
        return subs

    def _parse_local_definition(self) -> Definition:
        self._advance()  # LOCAL
        if self._at(TokenKind.INSTANCE):
            inst = self._parse_instance()
            inst.is_local = True
            return inst
        defn = self._parse_definition()
        if isinstance(defn, (OperatorDef, FunctionDef)):
            defn.is_local = True
        return defn

    # ================================================================
    # Definition parsing
    # ================================================================

    def parse_definition(self) -> Definition:
        """Parse a top-level definition."""
        return self._parse_definition()

    def _parse_definition(self) -> Definition:
        loc = self._loc()
        name_tok = self._expect(TokenKind.IDENTIFIER, "definition name")
        name = name_tok.value

        # Function definition:  f[x \in S] == body
        if self._at(TokenKind.LBRACKET):
            return self._parse_function_def(name, loc)

        # Operator definition:  Op(p1, p2) == body  or  Op == body
        params: List[str] = []
        if self._match(TokenKind.LPAREN):
            if not self._at(TokenKind.RPAREN):
                params = self._parse_identifier_list()
            self._expect(TokenKind.RPAREN, "operator parameters")

        self._expect(TokenKind.DEF_EQ, "definition '=='")
        body = self.parse_expression()
        return OperatorDef(
            name=name, params=params, body=body, source_location=loc
        )

    def _parse_function_def(self, name: str, loc: SourceLocation) -> FunctionDef:
        self._advance()  # [
        var = self._expect(TokenKind.IDENTIFIER, "function parameter").value
        self._expect(TokenKind.SET_IN, "function parameter '\\in'")
        set_expr = self.parse_expression()
        self._expect(TokenKind.RBRACKET, "function definition ']'")
        self._expect(TokenKind.DEF_EQ, "function definition '=='")
        body = self.parse_expression()
        return FunctionDef(
            name=name, variable=var, set_expr=set_expr, body=body,
            source_location=loc,
        )

    # ================================================================
    # Expression parsing — precedence climbing
    # ================================================================

    def parse_expression(self, min_prec: int = 0) -> Expression:
        """Parse an expression using operator-precedence climbing."""
        left = self._parse_unary()

        while True:
            tok = self._peek()
            if tok.kind not in _BINARY_PREC:
                break
            prec, assoc = _BINARY_PREC[tok.kind]
            if prec < min_prec:
                break

            op_tok = self._advance()
            op = _TOKEN_TO_OP[op_tok.kind]

            if assoc == "right":
                right = self.parse_expression(prec)
            elif assoc == "none":
                right = self.parse_expression(prec + 1)
            else:  # left
                right = self.parse_expression(prec + 1)

            left = OperatorApplication(
                operator=op,
                operands=[left, right],
                source_location=left.source_location.merge(right.source_location),
            )

        return left

    # ── Unary / prefix ──────────────────────────────────────────────

    def _parse_unary(self) -> Expression:
        tok = self._peek()

        if tok.kind == TokenKind.LNOT:
            return self._parse_prefix_op(Operator.LNOT)
        if tok.kind == TokenKind.MINUS:
            return self._parse_prefix_op(Operator.UMINUS)
        if tok.kind == TokenKind.SET_SUBSET:
            return self._parse_prefix_op(Operator.POWERSET)
        if tok.kind == TokenKind.SET_UNION_KW:
            return self._parse_prefix_op(Operator.UNION_ALL)
        if tok.kind == TokenKind.DOMAIN:
            return self._parse_domain()
        if tok.kind == TokenKind.ENABLED:
            return self._parse_enabled()
        if tok.kind == TokenKind.BOX:
            return self._parse_box()
        if tok.kind == TokenKind.DIAMOND:
            return self._parse_diamond()

        return self._parse_postfix()

    def _parse_prefix_op(self, op: Operator) -> Expression:
        loc = self._loc()
        self._advance()
        operand = self._parse_unary()
        return OperatorApplication(
            operator=op, operands=[operand],
            source_location=loc.merge(operand.source_location),
        )

    def _parse_domain(self) -> DomainExpr:
        loc = self._loc()
        self._advance()  # DOMAIN
        expr = self._parse_unary()
        return DomainExpr(expr=expr, source_location=loc.merge(expr.source_location))

    def _parse_enabled(self) -> OperatorApplication:
        loc = self._loc()
        self._advance()
        expr = self._parse_unary()
        return OperatorApplication(
            operator=Operator.ENABLED_OP, operands=[expr],
            source_location=loc.merge(expr.source_location),
        )

    def _parse_box(self) -> Expression:
        loc = self._loc()
        self._advance()  # []

        # [A]_v  pattern
        if self._at(TokenKind.LBRACKET):
            self._advance()
            action = self.parse_expression()
            self._expect(TokenKind.RBRACKET, "box action ']'")
            self._expect(TokenKind.UNDERSCORE, "box action '_'")
            var = self._parse_primary()
            return StutteringAction(
                action=action, variables=var, is_angle=False,
                source_location=loc.merge(var.source_location),
            )

        expr = self._parse_unary()
        return AlwaysExpr(expr=expr, source_location=loc.merge(expr.source_location))

    def _parse_diamond(self) -> Expression:
        loc = self._loc()
        self._advance()  # <>

        # <A>_v  pattern
        if self._at(TokenKind.LBRACKET):
            self._advance()
            action = self.parse_expression()
            self._expect(TokenKind.RBRACKET, "diamond action ']'")
            self._expect(TokenKind.UNDERSCORE, "diamond action '_'")
            var = self._parse_primary()
            return StutteringAction(
                action=action, variables=var, is_angle=True,
                source_location=loc.merge(var.source_location),
            )

        expr = self._parse_unary()
        return EventuallyExpr(
            expr=expr, source_location=loc.merge(expr.source_location),
        )

    # ── Postfix (prime, function application, record access) ────────

    def _parse_postfix(self) -> Expression:
        expr = self._parse_primary()

        while True:
            if self._at(TokenKind.PRIME):
                loc = self._loc()
                self._advance()
                if isinstance(expr, Identifier):
                    expr = PrimedIdentifier(
                        name=expr.name,
                        source_location=expr.source_location.merge(loc),
                    )
                else:
                    expr = OperatorApplication(
                        operator=Operator.PRIME, operands=[expr],
                        source_location=expr.source_location.merge(loc),
                    )
            elif self._at(TokenKind.LBRACKET):
                expr = self._parse_func_application(expr)
            elif self._at(TokenKind.DOT):
                expr = self._parse_record_access(expr)
            else:
                break
        return expr

    def _parse_func_application(self, func: Expression) -> FunctionApplication:
        self._advance()  # [
        arg = self.parse_expression()
        end = self._expect(TokenKind.RBRACKET, "function application ']'")
        return FunctionApplication(
            function=func, argument=arg,
            source_location=func.source_location.merge(end.location),
        )

    def _parse_record_access(self, record: Expression) -> RecordAccess:
        self._advance()  # .
        name_tok = self._expect(TokenKind.IDENTIFIER, "record field name")
        return RecordAccess(
            record=record, field_name=name_tok.value,
            source_location=record.source_location.merge(name_tok.location),
        )

    # ── Primary expressions ─────────────────────────────────────────

    def _parse_primary(self) -> Expression:
        tok = self._peek()

        if tok.kind == TokenKind.INTEGER:
            self._advance()
            return IntLiteral(value=tok.value, source_location=tok.location)

        if tok.kind == TokenKind.BOOL_TRUE:
            self._advance()
            return BoolLiteral(value=True, source_location=tok.location)

        if tok.kind == TokenKind.BOOL_FALSE:
            self._advance()
            return BoolLiteral(value=False, source_location=tok.location)

        if tok.kind == TokenKind.STRING:
            self._advance()
            return StringLiteral(value=tok.value, source_location=tok.location)

        if tok.kind == TokenKind.IDENTIFIER:
            return self._parse_identifier_or_call()

        if tok.kind == TokenKind.KW_BOOLEAN:
            self._advance()
            return Identifier(name="BOOLEAN", source_location=tok.location)

        if tok.kind == TokenKind.KW_STRING:
            self._advance()
            return Identifier(name="STRING", source_location=tok.location)

        if tok.kind == TokenKind.KW_NAT:
            self._advance()
            return Identifier(name="Nat", source_location=tok.location)

        if tok.kind == TokenKind.KW_INT:
            self._advance()
            return Identifier(name="Int", source_location=tok.location)

        if tok.kind == TokenKind.LPAREN:
            return self._parse_parenthesized()

        if tok.kind == TokenKind.LBRACE:
            return self._parse_set_expression()

        if tok.kind == TokenKind.LBRACKET:
            return self._parse_bracket_expression()

        if tok.kind == TokenKind.LANGLE:
            return self._parse_tuple()

        if tok.kind == TokenKind.IF:
            return self._parse_if_then_else()

        if tok.kind == TokenKind.LET:
            return self._parse_let_in()

        if tok.kind == TokenKind.CASE:
            return self._parse_case()

        if tok.kind == TokenKind.CHOOSE:
            return self._parse_choose()

        if tok.kind == TokenKind.UNCHANGED:
            return self._parse_unchanged()

        if tok.kind in (TokenKind.FORALL, TokenKind.EXISTS):
            return self._parse_quantified()

        if tok.kind in (TokenKind.TEMPORAL_FORALL, TokenKind.TEMPORAL_EXISTS):
            return self._parse_temporal_quantified()

        if tok.kind in (TokenKind.WF, TokenKind.SF):
            return self._parse_fairness()

        if tok.kind == TokenKind.AT:
            self._advance()
            return Identifier(name="@", source_location=tok.location)

        self._raise(f"Unexpected token {tok.kind.name}", tok.location)
        return Identifier(name="<error>", source_location=tok.location)  # unreachable

    def _parse_identifier_or_call(self) -> Expression:
        tok = self._advance()
        name = tok.value

        # Sequence built-in operators used as functions: Head(s), Tail(s), etc.
        seq_ops = {
            "Append": Operator.APPEND,
            "Head": Operator.HEAD,
            "Tail": Operator.TAIL,
            "Len": Operator.LEN,
            "SubSeq": Operator.SUBSEQ,
            "Seq": Operator.SEQ,
        }
        if name in seq_ops and self._at(TokenKind.LPAREN):
            return self._parse_builtin_call(name, seq_ops[name], tok.location)

        # User-defined operator call:  Op(e1, e2, ...)
        if self._at(TokenKind.LPAREN):
            return self._parse_op_call(name, tok.location)

        return Identifier(name=name, source_location=tok.location)

    def _parse_builtin_call(
        self, name: str, op: Operator, loc: SourceLocation
    ) -> OperatorApplication:
        self._advance()  # (
        args: List[Expression] = []
        if not self._at(TokenKind.RPAREN):
            args.append(self.parse_expression())
            while self._match(TokenKind.COMMA):
                args.append(self.parse_expression())
        end = self._expect(TokenKind.RPAREN, f"{name}(...)")
        return OperatorApplication(
            operator=op, operands=args, operator_name=name,
            source_location=loc.merge(end.location),
        )

    def _parse_op_call(self, name: str, loc: SourceLocation) -> Expression:
        self._advance()  # (
        args: List[Expression] = []
        if not self._at(TokenKind.RPAREN):
            args.append(self.parse_expression())
            while self._match(TokenKind.COMMA):
                args.append(self.parse_expression())
        end = self._expect(TokenKind.RPAREN, f"{name}(...)")
        return OperatorApplication(
            operator=Operator.FUNC_APPLY,
            operands=args,
            operator_name=name,
            source_location=loc.merge(end.location),
        )

    def _parse_parenthesized(self) -> Expression:
        self._advance()  # (
        expr = self.parse_expression()
        self._expect(TokenKind.RPAREN, "parenthesized expression")
        return expr

    # ── Set expressions ─────────────────────────────────────────────

    def _parse_set_expression(self) -> Expression:
        """Parse {}, {e1, e2}, {x \\in S : P}, or {e : x \\in S}."""
        loc = self._loc()
        self._advance()  # {

        if self._at(TokenKind.RBRACE):
            end = self._advance()
            return SetEnumeration(
                elements=[], source_location=loc.merge(end.location)
            )

        first = self.parse_expression()

        # {x \in S : P(x)} — set comprehension (filter)
        if self._at(TokenKind.SET_IN) and isinstance(first, Identifier):
            return self._parse_set_comprehension_filter(first.name, loc)

        # {e : x \in S} — set comprehension (map)
        if self._at(TokenKind.COLON):
            return self._parse_set_comprehension_map(first, loc)

        # {e1, e2, ...} — enumeration
        elements = [first]
        while self._match(TokenKind.COMMA):
            elements.append(self.parse_expression())
        end = self._expect(TokenKind.RBRACE, "set enumeration '}'")
        return SetEnumeration(
            elements=elements, source_location=loc.merge(end.location)
        )

    def _parse_set_comprehension_filter(
        self, var: str, loc: SourceLocation
    ) -> SetComprehension:
        self._advance()  # \in
        set_expr = self.parse_expression()
        self._expect(TokenKind.COLON, "set comprehension ':'")
        predicate = self.parse_expression()
        end = self._expect(TokenKind.RBRACE, "set comprehension '}'")
        return SetComprehension(
            variable=var, set_expr=set_expr, predicate=predicate,
            source_location=loc.merge(end.location),
        )

    def _parse_set_comprehension_map(
        self, map_expr: Expression, loc: SourceLocation
    ) -> SetComprehension:
        self._advance()  # :
        var = self._expect(TokenKind.IDENTIFIER, "set comprehension variable").value
        self._expect(TokenKind.SET_IN, "set comprehension '\\in'")
        set_expr = self.parse_expression()
        end = self._expect(TokenKind.RBRACE, "set comprehension '}'")
        return SetComprehension(
            variable=var, set_expr=set_expr, map_expr=map_expr,
            source_location=loc.merge(end.location),
        )

    # ── Bracket expressions (function/record construction, EXCEPT) ─

    def _parse_bracket_expression(self) -> Expression:
        """Dispatch [x \\in S |-> e], [f1 |-> v1, …], [f EXCEPT …], or func app."""
        loc = self._loc()
        self._advance()  # [

        # Empty brackets
        if self._at(TokenKind.RBRACKET):
            end = self._advance()
            return FunctionConstruction(source_location=loc.merge(end.location))

        # Look-ahead to distinguish cases
        # Case 1: identifier followed by \in => function construction
        if self._at(TokenKind.IDENTIFIER) and self._peek(1).kind == TokenKind.SET_IN:
            return self._parse_function_construction(loc)

        # Case 2: identifier followed by |-> => record construction
        if self._at(TokenKind.IDENTIFIER) and self._peek(1).kind == TokenKind.MAPS_TO:
            return self._parse_record_construction(loc)

        # Case 3: starts with expression followed by EXCEPT => except expression
        first = self.parse_expression()

        if self._at(TokenKind.EXCEPT):
            return self._parse_except_expr(first, loc)

        # Otherwise it's a record type like [key: Type, ...]
        # or something we don't recognise — wrap as function application
        end = self._expect(TokenKind.RBRACKET, "bracket expression ']'")
        return FunctionApplication(
            function=first, argument=None,
            source_location=loc.merge(end.location),
        )

    def _parse_function_construction(self, loc: SourceLocation) -> FunctionConstruction:
        var = self._advance().value  # identifier
        self._advance()  # \in
        set_expr = self.parse_expression()
        self._expect(TokenKind.MAPS_TO, "function construction '|->'")
        body = self.parse_expression()
        end = self._expect(TokenKind.RBRACKET, "function construction ']'")
        return FunctionConstruction(
            variable=var, set_expr=set_expr, body=body,
            source_location=loc.merge(end.location),
        )

    def _parse_record_construction(self, loc: SourceLocation) -> RecordConstruction:
        fields: List[tuple[str, Expression]] = []
        while True:
            name = self._expect(TokenKind.IDENTIFIER, "record field name").value
            self._expect(TokenKind.MAPS_TO, "record field '|->'")
            val = self.parse_expression()
            fields.append((name, val))
            if not self._match(TokenKind.COMMA):
                break
        end = self._expect(TokenKind.RBRACKET, "record construction ']'")
        return RecordConstruction(
            fields=fields, source_location=loc.merge(end.location)
        )

    def _parse_except_expr(
        self, base: Expression, loc: SourceLocation
    ) -> ExceptExpr:
        self._advance()  # EXCEPT
        subs: List[tuple[List[Expression], Expression]] = []
        while True:
            self._expect(TokenKind.BANG, "EXCEPT '!'")
            path: List[Expression] = []
            # Parse path: ![a][b] = val  or  !.field = val
            while self._at(TokenKind.LBRACKET):
                self._advance()
                path.append(self.parse_expression())
                self._expect(TokenKind.RBRACKET, "EXCEPT path ']'")
            while self._at(TokenKind.DOT):
                self._advance()
                name_tok = self._expect(TokenKind.IDENTIFIER, "EXCEPT field")
                path.append(StringLiteral(value=name_tok.value, source_location=name_tok.location))
            self._expect(TokenKind.EQ, "EXCEPT '='")
            val = self.parse_expression()
            subs.append((path, val))
            if not self._match(TokenKind.COMMA):
                break
        end = self._expect(TokenKind.RBRACKET, "EXCEPT ']'")
        return ExceptExpr(
            base=base, substitutions=subs,
            source_location=loc.merge(end.location),
        )

    # ── Tuple / Sequence ────────────────────────────────────────────

    def _parse_tuple(self) -> Expression:
        """Parse <<e1, e2, …>>."""
        loc = self._loc()
        self._advance()  # <<

        if self._at(TokenKind.RANGLE):
            end = self._advance()
            return TupleLiteral(elements=[], source_location=loc.merge(end.location))

        elements: List[Expression] = [self.parse_expression()]
        while self._match(TokenKind.COMMA):
            elements.append(self.parse_expression())
        end = self._expect(TokenKind.RANGLE, "tuple '>>'")
        return TupleLiteral(
            elements=elements, source_location=loc.merge(end.location)
        )

    # ── IF / THEN / ELSE ────────────────────────────────────────────

    def _parse_if_then_else(self) -> IfThenElse:
        loc = self._loc()
        self._advance()  # IF
        cond = self.parse_expression()
        self._expect(TokenKind.THEN, "IF-THEN-ELSE")
        then_e = self.parse_expression()
        self._expect(TokenKind.ELSE, "IF-THEN-ELSE")
        else_e = self.parse_expression()
        return IfThenElse(
            condition=cond, then_expr=then_e, else_expr=else_e,
            source_location=loc.merge(else_e.source_location),
        )

    # ── LET / IN ────────────────────────────────────────────────────

    def _parse_let_in(self) -> LetIn:
        loc = self._loc()
        self._advance()  # LET
        defs: List[Definition] = []
        while not self._at(TokenKind.IN) and not self._at(TokenKind.EOF):
            defs.append(self._parse_definition())
        self._expect(TokenKind.IN, "LET-IN")
        body = self.parse_expression()
        return LetIn(
            definitions=defs, body=body,
            source_location=loc.merge(body.source_location),
        )

    # ── CASE ────────────────────────────────────────────────────────

    def _parse_case(self) -> CaseExpr:
        loc = self._loc()
        self._advance()  # CASE
        arms: List[CaseArm] = []
        other: Optional[Expression] = None

        # First arm (no leading [])
        arm = self._parse_case_arm()
        if arm is None:
            # OTHER -> e
            if self._match(TokenKind.OTHER):
                self._expect(TokenKind.IMPLIES, "CASE OTHER '->'")
                other = self.parse_expression()
                return CaseExpr(arms=arms, other=other, source_location=loc)
        else:
            arms.append(arm)

        # Subsequent arms: [] guard -> expr
        while self._at(TokenKind.BOX):
            self._advance()  # []
            if self._match(TokenKind.OTHER):
                self._expect(TokenKind.IMPLIES, "CASE OTHER '->'")
                other = self.parse_expression()
                break
            arm = self._parse_case_arm()
            if arm:
                arms.append(arm)

        return CaseExpr(arms=arms, other=other, source_location=loc)

    def _parse_case_arm(self) -> Optional[CaseArm]:
        if self._at(TokenKind.OTHER):
            return None
        loc = self._loc()
        cond = self.parse_expression()
        self._expect(TokenKind.IMPLIES, "CASE arm '->'")
        val = self.parse_expression()
        return CaseArm(
            condition=cond, value=val, source_location=loc.merge(val.source_location)
        )

    # ── CHOOSE ──────────────────────────────────────────────────────

    def _parse_choose(self) -> ChooseExpr:
        loc = self._loc()
        self._advance()  # CHOOSE
        var = self._expect(TokenKind.IDENTIFIER, "CHOOSE variable").value
        set_expr: Optional[Expression] = None
        if self._match(TokenKind.SET_IN):
            set_expr = self.parse_expression()
        self._expect(TokenKind.COLON, "CHOOSE ':'")
        pred = self.parse_expression()
        return ChooseExpr(
            variable=var, set_expr=set_expr, predicate=pred,
            source_location=loc.merge(pred.source_location),
        )

    # ── UNCHANGED ───────────────────────────────────────────────────

    def _parse_unchanged(self) -> UnchangedExpr:
        loc = self._loc()
        self._advance()  # UNCHANGED

        variables: List[Expression] = []
        if self._at(TokenKind.LANGLE):
            self._advance()
            if not self._at(TokenKind.RANGLE):
                variables.append(self.parse_expression())
                while self._match(TokenKind.COMMA):
                    variables.append(self.parse_expression())
            end = self._expect(TokenKind.RANGLE, "UNCHANGED '>>'")
            return UnchangedExpr(
                variables=variables, source_location=loc.merge(end.location)
            )

        # Single variable
        expr = self._parse_primary()
        variables.append(expr)
        return UnchangedExpr(
            variables=variables, source_location=loc.merge(expr.source_location)
        )

    # ── Quantified expressions ──────────────────────────────────────

    def _parse_quantified(self) -> QuantifiedExpr:
        loc = self._loc()
        tok = self._advance()
        kind = "forall" if tok.kind == TokenKind.FORALL else "exists"

        bounds: List[tuple[str, Expression]] = []
        while True:
            var = self._expect(TokenKind.IDENTIFIER, "quantifier variable").value
            self._expect(TokenKind.SET_IN, "quantifier '\\in'")
            set_expr = self.parse_expression()
            bounds.append((var, set_expr))
            if not self._match(TokenKind.COMMA):
                break

        self._expect(TokenKind.COLON, "quantifier ':'")
        body = self.parse_expression()
        return QuantifiedExpr(
            quantifier=kind, variables=bounds, body=body,
            source_location=loc.merge(body.source_location),
        )

    def _parse_temporal_quantified(self) -> Expression:
        loc = self._loc()
        tok = self._advance()
        var = self._expect(TokenKind.IDENTIFIER, "temporal quantifier variable").value
        self._expect(TokenKind.COLON, "temporal quantifier ':'")
        body = self.parse_expression()
        if tok.kind == TokenKind.TEMPORAL_FORALL:
            return TemporalForallExpr(
                variable=var, body=body,
                source_location=loc.merge(body.source_location),
            )
        return TemporalExistsExpr(
            variable=var, body=body,
            source_location=loc.merge(body.source_location),
        )

    # ── Fairness ────────────────────────────────────────────────────

    def _parse_fairness(self) -> FairnessExpr:
        loc = self._loc()
        tok = self._advance()  # WF_ or SF_
        kind = "WF" if tok.kind == TokenKind.WF else "SF"
        var_expr = self._parse_primary()
        self._expect(TokenKind.LPAREN, f"{kind}_ '('")
        action = self.parse_expression()
        end = self._expect(TokenKind.RPAREN, f"{kind}_ ')'")
        return FairnessExpr(
            kind=kind, variables=var_expr, action=action,
            source_location=loc.merge(end.location),
        )

    # ── Action / temporal parsing (called from module level) ────────

    def parse_action(self) -> Expression:
        """Parse an action-level expression."""
        return self.parse_expression()

    def parse_temporal(self) -> Expression:
        """Parse a temporal-level expression."""
        return self.parse_expression()


# ============================================================================
# Convenience functions
# ============================================================================

def parse(source: str, file: str = "<string>") -> Module:
    """Parse TLA-lite source text and return the Module AST."""
    parser = Parser(source, file)
    return parser.parse_module()


def parse_expression(source: str, file: str = "<string>") -> Expression:
    """Parse a single TLA-lite expression."""
    parser = Parser(source, file)
    return parser.parse_expression()
