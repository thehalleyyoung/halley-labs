"""Recursive-descent parser for the RegSynth regulatory DSL.

Consumes a token stream produced by the tokenizer and builds a typed AST.
The grammar is LL(1)-friendly: every decision point can be resolved by
inspecting at most one token of look-ahead.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, TypeVar

from regsynth_py.dsl.tokenizer import Token, TokenType, tokenize
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
    SourceLocation,
    Expression,
    Declaration,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """Raised when the parser encounters an unexpected token or structure."""

    def __init__(self, message: str, line: int = 0, column: int = 0) -> None:
        self.line = line
        self.column = column
        loc = f" at line {line}, column {column}" if line or column else ""
        super().__init__(f"ParseError{loc}: {message}")


# ---------------------------------------------------------------------------
# Operator look-up tables
# ---------------------------------------------------------------------------

_COMPARISON_OPS = {
    TokenType.EQ,
    TokenType.NEQ,
    TokenType.LT,
    TokenType.GT,
    TokenType.LTE,
    TokenType.GTE,
}

_ADDITIVE_OPS = {
    TokenType.PLUS,
    TokenType.MINUS,
}

_MULTIPLICATIVE_OPS = {
    TokenType.STAR,
    TokenType.SLASH,
    TokenType.PERCENT,
}

_RISK_LEVEL_MAP = {
    "UNACCEPTABLE": RiskLevel.UNACCEPTABLE,
    "HIGH": RiskLevel.HIGH,
    "LIMITED": RiskLevel.LIMITED,
    "MINIMAL": RiskLevel.MINIMAL,
}

_OBLIGATION_TYPE_MAP = {
    "MANDATORY": ObligationType.MANDATORY,
    "RECOMMENDED": ObligationType.RECOMMENDED,
    "OPTIONAL": ObligationType.OPTIONAL,
    "CONDITIONAL": ObligationType.CONDITIONAL,
}

_FRAMEWORK_TYPE_MAP = {
    "BINDING": FrameworkType.BINDING,
    "VOLUNTARY": FrameworkType.VOLUNTARY,
    "HYBRID": FrameworkType.HYBRID,
}

_COMPOSE_MODE_MAP = {
    "UNION": ComposeMode.UNION,
    "INTERSECT": ComposeMode.INTERSECT,
    "OVERRIDE": ComposeMode.OVERRIDE,
    "SEQUENCE": ComposeMode.SEQUENCE,
}

_TEMPORAL_KEYWORDS = {"BEFORE", "AFTER", "WITHIN", "EVERY"}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    """Recursive-descent parser that transforms a token list into an AST."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens: list[Token] = tokens
        self._pos: int = 0

    # -- helpers ------------------------------------------------------------

    def peek(self) -> Token:
        """Return the current token without consuming it."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        # Synthesise an EOF token when we run past the end.
        last = self._tokens[-1] if self._tokens else Token(TokenType.EOF, "", 1, 1)
        return Token(TokenType.EOF, "", last.line, last.column)

    def advance(self) -> Token:
        """Consume and return the current token."""
        token = self.peek()
        if token.type != TokenType.EOF:
            self._pos += 1
        return token

    def check(self, token_type: TokenType) -> bool:
        """Return *True* if the current token has the given type."""
        return self.peek().type == token_type

    def match(self, *token_types: TokenType) -> bool:
        """If the current token matches any of *token_types*, advance and
        return ``True``; otherwise return ``False``."""
        for tt in token_types:
            if self.check(tt):
                self.advance()
                return True
        return False

    def expect(self, token_type: TokenType, message: str) -> Token:
        """Consume a token of the expected type, or raise :class:`ParseError`."""
        tok = self.peek()
        if tok.type == token_type:
            return self.advance()
        raise ParseError(message, tok.line, tok.column)

    def is_at_end(self) -> bool:
        """Return *True* when there are no more meaningful tokens."""
        return self.peek().type == TokenType.EOF

    def skip_newlines(self) -> None:
        """Skip over any NEWLINE tokens at the current position."""
        while self.check(TokenType.NEWLINE):
            self.advance()

    def location(self) -> SourceLocation:
        """Build a :class:`SourceLocation` from the current token."""
        tok = self.peek()
        return SourceLocation(line=tok.line, column=tok.column)

    # -- top-level ----------------------------------------------------------

    def parse(self) -> Program:
        """Parse the full token stream into a :class:`Program` node."""
        declarations: list[Declaration] = []
        self.skip_newlines()
        while not self.is_at_end():
            decl = self.parse_declaration()
            declarations.append(decl)
            self.skip_newlines()
        return Program(declarations=declarations, location=SourceLocation(line=1, column=1))

    def parse_declaration(self) -> Declaration:
        """Dispatch to the correct declaration parser based on the current
        keyword token."""
        self.skip_newlines()
        tok = self.peek()

        if tok.type == TokenType.KEYWORD:
            kw = tok.value.lower()
            if kw == "obligation":
                return self.parse_obligation()
            if kw == "jurisdiction":
                return self.parse_jurisdiction()
            if kw == "strategy":
                return self.parse_strategy()
            if kw == "compose":
                return self.parse_composition()
            if kw == "constraint":
                return self.parse_constraint()
            raise ParseError(
                f"Unknown declaration keyword '{tok.value}'",
                tok.line,
                tok.column,
            )

        if tok.type == TokenType.IDENTIFIER:
            ident_val = tok.value.lower()
            if ident_val == "obligation":
                return self.parse_obligation()
            if ident_val == "jurisdiction":
                return self.parse_jurisdiction()
            if ident_val == "strategy":
                return self.parse_strategy()
            if ident_val == "compose":
                return self.parse_composition()
            if ident_val == "constraint":
                return self.parse_constraint()

        raise ParseError(
            f"Expected declaration keyword, got {tok.type.name} '{tok.value}'",
            tok.line,
            tok.column,
        )

    # -- declaration parsers ------------------------------------------------

    def parse_obligation(self) -> ObligationDecl:
        """Parse an ``obligation`` declaration.

        ::

            obligation NAME {
                jurisdiction: X,
                risk: HIGH,
                category: "cat",
                articles: [a1, a2],
                temporal: BEFORE 2025-08-01,
                requirements: [r1, r2]
            }
        """
        loc = self.location()
        self._expect_keyword("obligation")
        name = self.parse_identifier()
        self.skip_newlines()
        self.expect(TokenType.LBRACE, "Expected '{' after obligation name")
        self.skip_newlines()

        jurisdiction: Optional[Identifier] = None
        risk: Optional[RiskLevel] = None
        category: Optional[str] = None
        articles: list[Expression] = []
        temporal: Optional[TemporalExpr] = None
        requirements: list[Expression] = []
        obligation_type: Optional[ObligationType] = None

        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            key, key_tok = self._parse_field_key()
            kl = key.lower()

            if kl == "jurisdiction":
                jurisdiction = self.parse_identifier()
            elif kl == "risk":
                risk = self.parse_risk_level()
            elif kl == "category":
                category = self._consume_string_value()
            elif kl == "articles":
                articles = self.parse_list(self._parse_list_item)
            elif kl == "temporal":
                temporal = self.parse_temporal()
            elif kl == "requirements":
                requirements = self.parse_list(self._parse_list_item)
            elif kl == "type":
                obligation_type = self.parse_obligation_type()
            else:
                raise ParseError(
                    f"Unknown obligation field '{key}'",
                    key_tok.line,
                    key_tok.column,
                )

            self._skip_comma_or_newline()

        self.expect(TokenType.RBRACE, "Expected '}' to close obligation block")

        return ObligationDecl(
            name=name,
            jurisdiction=jurisdiction,
            risk=risk,
            category=category,
            articles=articles,
            temporal=temporal,
            requirements=requirements,
            obligation_type=obligation_type,
            location=loc,
        )

    def parse_jurisdiction(self) -> JurisdictionDecl:
        """Parse a ``jurisdiction`` declaration.

        ::

            jurisdiction NAME {
                framework: BINDING,
                region: "EU",
                enforcement: 2025-08-01,
                penalties: { min: 1000, max: 35000000 }
            }
        """
        loc = self.location()
        self._expect_keyword("jurisdiction")
        name = self.parse_identifier()
        self.skip_newlines()
        self.expect(TokenType.LBRACE, "Expected '{' after jurisdiction name")
        self.skip_newlines()

        framework: Optional[FrameworkType] = None
        region: Optional[str] = None
        enforcement: Optional[str] = None
        penalties_min: Optional[float] = None
        penalties_max: Optional[float] = None

        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            key, key_tok = self._parse_field_key()
            kl = key.lower()

            if kl == "framework":
                framework = self._parse_framework_type()
            elif kl == "region":
                region = self._consume_string_value()
            elif kl == "enforcement":
                enforcement = self._consume_date_or_string()
            elif kl == "penalties":
                penalties_min, penalties_max = self._parse_penalty_block()
            else:
                raise ParseError(
                    f"Unknown jurisdiction field '{key}'",
                    key_tok.line,
                    key_tok.column,
                )

            self._skip_comma_or_newline()

        self.expect(TokenType.RBRACE, "Expected '}' to close jurisdiction block")

        return JurisdictionDecl(
            name=name,
            framework=framework,
            region=region,
            enforcement=enforcement,
            penalties_min=penalties_min,
            penalties_max=penalties_max,
            location=loc,
        )

    def parse_strategy(self) -> StrategyDecl:
        """Parse a ``strategy`` declaration.

        ::

            strategy NAME {
                obligations: [ob1, ob2],
                cost: 50000,
                timeline: 12
            }
        """
        loc = self.location()
        self._expect_keyword("strategy")
        name = self.parse_identifier()
        self.skip_newlines()
        self.expect(TokenType.LBRACE, "Expected '{' after strategy name")
        self.skip_newlines()

        obligations: list[Identifier] = []
        cost: Optional[float] = None
        timeline: Optional[int] = None

        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            key, key_tok = self._parse_field_key()
            kl = key.lower()

            if kl == "obligations":
                obligations = self.parse_list(self.parse_identifier)
            elif kl == "cost":
                cost = self._consume_number_value()
            elif kl == "timeline":
                timeline = int(self._consume_number_value())
            else:
                raise ParseError(
                    f"Unknown strategy field '{key}'",
                    key_tok.line,
                    key_tok.column,
                )

            self._skip_comma_or_newline()

        self.expect(TokenType.RBRACE, "Expected '}' to close strategy block")

        return StrategyDecl(
            name=name,
            obligations=obligations,
            cost=cost,
            timeline=timeline,
            location=loc,
        )

    def parse_composition(self) -> CompositionDecl:
        """Parse a ``compose`` declaration.

        ::

            compose NAME = strategy1 UNION strategy2
        """
        loc = self.location()
        self._expect_keyword("compose")
        name = self.parse_identifier()
        self.expect(TokenType.EQ, "Expected '=' after compose name")

        left = self.parse_identifier()
        mode = self._parse_compose_mode()
        right = self.parse_identifier()

        return CompositionDecl(
            name=name,
            left=left,
            mode=mode,
            right=right,
            location=loc,
        )

    def parse_constraint(self) -> ConstraintDecl:
        """Parse a ``constraint`` declaration.

        ::

            constraint NAME {
                type: "budget",
                max: 100000
            }
        """
        loc = self.location()
        self._expect_keyword("constraint")
        name = self.parse_identifier()
        self.skip_newlines()
        self.expect(TokenType.LBRACE, "Expected '{' after constraint name")
        self.skip_newlines()

        constraint_type: Optional[str] = None
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        expression: Optional[Expression] = None

        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            key, key_tok = self._parse_field_key()
            kl = key.lower()

            if kl == "type":
                constraint_type = self._consume_string_value()
            elif kl == "min":
                min_val = self._consume_number_value()
            elif kl == "max":
                max_val = self._consume_number_value()
            elif kl == "expr" or kl == "expression":
                expression = self.parse_expression()
            else:
                raise ParseError(
                    f"Unknown constraint field '{key}'",
                    key_tok.line,
                    key_tok.column,
                )

            self._skip_comma_or_newline()

        self.expect(TokenType.RBRACE, "Expected '}' to close constraint block")

        return ConstraintDecl(
            name=name,
            constraint_type=constraint_type,
            min_val=min_val,
            max_val=max_val,
            expression=expression,
            location=loc,
        )

    # -- expression parsers (precedence climbing) ---------------------------

    def parse_expression(self) -> Expression:
        """Entry point for expression parsing — delegates to the lowest
        precedence level."""
        return self.parse_or()

    def parse_or(self) -> Expression:
        """Parse an OR expression (lowest precedence binary op)."""
        left = self.parse_and()
        while self._match_keyword("OR") or self.match(TokenType.OR):
            loc = self.location()
            right = self.parse_and()
            left = BinaryOp(op="OR", left=left, right=right, location=loc)
        return left

    def parse_and(self) -> Expression:
        """Parse an AND expression."""
        left = self.parse_implies()
        while self._match_keyword("AND") or self.match(TokenType.AND):
            loc = self.location()
            right = self.parse_implies()
            left = BinaryOp(op="AND", left=left, right=right, location=loc)
        return left

    def parse_implies(self) -> Expression:
        """Parse an IMPLIES expression (right-associative)."""
        left = self.parse_comparison()
        if self._match_keyword("IMPLIES") or self.match(TokenType.IMPLIES):
            loc = self.location()
            right = self.parse_implies()  # right-associative
            left = BinaryOp(op="IMPLIES", left=left, right=right, location=loc)
        return left

    def parse_comparison(self) -> Expression:
        """Parse comparison operators: ==  !=  <  >  <=  >="""
        left = self.parse_addition()
        while self.peek().type in _COMPARISON_OPS:
            op_tok = self.advance()
            loc = SourceLocation(line=op_tok.line, column=op_tok.column)
            right = self.parse_addition()
            left = BinaryOp(op=op_tok.value, left=left, right=right, location=loc)
        return left

    def parse_addition(self) -> Expression:
        """Parse additive operators: +  -"""
        left = self.parse_multiplication()
        while self.peek().type in _ADDITIVE_OPS:
            op_tok = self.advance()
            loc = SourceLocation(line=op_tok.line, column=op_tok.column)
            right = self.parse_multiplication()
            left = BinaryOp(op=op_tok.value, left=left, right=right, location=loc)
        return left

    def parse_multiplication(self) -> Expression:
        """Parse multiplicative operators: *  /  %"""
        left = self.parse_unary()
        while self.peek().type in _MULTIPLICATIVE_OPS:
            op_tok = self.advance()
            loc = SourceLocation(line=op_tok.line, column=op_tok.column)
            right = self.parse_unary()
            left = BinaryOp(op=op_tok.value, left=left, right=right, location=loc)
        return left

    def parse_unary(self) -> Expression:
        """Parse unary prefix operators: NOT  -"""
        tok = self.peek()

        if tok.type == TokenType.NOT or (
            tok.type in (TokenType.KEYWORD, TokenType.IDENTIFIER)
            and tok.value.upper() == "NOT"
        ):
            self.advance()
            loc = SourceLocation(line=tok.line, column=tok.column)
            operand = self.parse_unary()
            return UnaryOp(op="NOT", operand=operand, location=loc)

        if tok.type == TokenType.MINUS:
            self.advance()
            loc = SourceLocation(line=tok.line, column=tok.column)
            operand = self.parse_unary()
            return UnaryOp(op="-", operand=operand, location=loc)

        return self.parse_primary()

    def parse_primary(self) -> Expression:
        """Parse a primary expression: literal, identifier, grouped ``(…)``,
        or temporal keyword."""
        tok = self.peek()

        # Numeric literal
        if tok.type == TokenType.NUMBER:
            return self.parse_literal()

        # String literal
        if tok.type == TokenType.STRING:
            return self.parse_literal()

        # Boolean literals
        if tok.type == TokenType.TRUE or tok.type == TokenType.FALSE:
            return self.parse_literal()

        # Grouped expression
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN, "Expected ')' after grouped expression")
            return expr

        # Temporal expression (BEFORE / AFTER / WITHIN / EVERY)
        if self._is_temporal_keyword(tok):
            return self.parse_temporal()

        # Identifier (or keyword used in identifier position)
        if tok.type in (TokenType.IDENTIFIER, TokenType.KEYWORD):
            return self.parse_identifier()

        raise ParseError(
            f"Unexpected token {tok.type.name} '{tok.value}' in expression",
            tok.line,
            tok.column,
        )

    def parse_temporal(self) -> TemporalExpr:
        """Parse a temporal expression.

        ::

            BEFORE 2025-08-01
            AFTER  2025-01-01
            WITHIN 90
            EVERY  30
        """
        tok = self.peek()
        if not self._is_temporal_keyword(tok):
            raise ParseError(
                f"Expected temporal keyword (BEFORE/AFTER/WITHIN/EVERY), got '{tok.value}'",
                tok.line,
                tok.column,
            )

        loc = SourceLocation(line=tok.line, column=tok.column)
        kind = self.advance().value.upper()
        deadline = self._consume_date_or_number_or_string()

        return TemporalExpr(kind=kind, deadline=deadline, location=loc)

    # -- utility parsers ----------------------------------------------------

    def parse_list(self, parse_fn: Callable[[], T]) -> list[T]:
        """Parse a comma-separated list enclosed in brackets.

        ::

            [ item1, item2, item3 ]
        """
        self.expect(TokenType.LBRACKET, "Expected '[' to open list")
        self.skip_newlines()
        items: list[T] = []

        if not self.check(TokenType.RBRACKET):
            items.append(parse_fn())
            while self.match(TokenType.COMMA):
                self.skip_newlines()
                if self.check(TokenType.RBRACKET):
                    break  # trailing comma
                items.append(parse_fn())

        self.skip_newlines()
        self.expect(TokenType.RBRACKET, "Expected ']' to close list")
        return items

    def parse_block(self, parse_fn: Callable[[], T]) -> list[T]:
        """Parse items inside a brace-delimited block.

        ::

            { item1, item2, item3 }
        """
        self.expect(TokenType.LBRACE, "Expected '{' to open block")
        self.skip_newlines()
        items: list[T] = []

        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            items.append(parse_fn())
            self._skip_comma_or_newline()

        self.expect(TokenType.RBRACE, "Expected '}' to close block")
        return items

    def parse_key_value(self) -> tuple[str, Expression]:
        """Parse a ``key: value`` pair and return ``(key, expression)``."""
        key, _ = self._parse_field_key()
        value = self.parse_expression()
        return (key, value)

    def parse_identifier(self) -> Identifier:
        """Parse an identifier token and wrap it in an :class:`Identifier`
        AST node."""
        tok = self.peek()
        if tok.type == TokenType.IDENTIFIER:
            self.advance()
            return Identifier(
                name=tok.value,
                location=SourceLocation(line=tok.line, column=tok.column),
            )
        # Allow keywords in identifier position (e.g. ``risk`` as a name).
        if tok.type == TokenType.KEYWORD:
            self.advance()
            return Identifier(
                name=tok.value,
                location=SourceLocation(line=tok.line, column=tok.column),
            )
        raise ParseError(
            f"Expected identifier, got {tok.type.name} '{tok.value}'",
            tok.line,
            tok.column,
        )

    def parse_literal(self) -> Literal:
        """Parse a literal value (number, string, boolean)."""
        tok = self.peek()

        if tok.type == TokenType.NUMBER:
            self.advance()
            value = self._coerce_number(tok.value)
            return Literal(
                value=value,
                location=SourceLocation(line=tok.line, column=tok.column),
            )

        if tok.type == TokenType.STRING:
            self.advance()
            return Literal(
                value=tok.value,
                location=SourceLocation(line=tok.line, column=tok.column),
            )

        if tok.type == TokenType.TRUE:
            self.advance()
            return Literal(
                value=True,
                location=SourceLocation(line=tok.line, column=tok.column),
            )

        if tok.type == TokenType.FALSE:
            self.advance()
            return Literal(
                value=False,
                location=SourceLocation(line=tok.line, column=tok.column),
            )

        raise ParseError(
            f"Expected literal, got {tok.type.name} '{tok.value}'",
            tok.line,
            tok.column,
        )

    def parse_risk_level(self) -> RiskLevel:
        """Parse one of the predefined risk-level keywords."""
        tok = self.peek()
        key = tok.value.upper() if tok.value else ""
        if key in _RISK_LEVEL_MAP:
            self.advance()
            return _RISK_LEVEL_MAP[key]
        raise ParseError(
            f"Expected risk level (LOW/MEDIUM/HIGH/CRITICAL), got '{tok.value}'",
            tok.line,
            tok.column,
        )

    def parse_obligation_type(self) -> ObligationType:
        """Parse one of the predefined obligation-type keywords."""
        tok = self.peek()
        key = tok.value.upper() if tok.value else ""
        if key in _OBLIGATION_TYPE_MAP:
            self.advance()
            return _OBLIGATION_TYPE_MAP[key]
        raise ParseError(
            f"Expected obligation type (DISCLOSURE/ASSESSMENT/MONITORING/"
            f"REPORTING/DOCUMENTATION), got '{tok.value}'",
            tok.line,
            tok.column,
        )

    # -- private helpers ----------------------------------------------------

    def _expect_keyword(self, keyword: str) -> Token:
        """Consume a keyword or identifier that matches *keyword*
        (case-insensitive)."""
        tok = self.peek()
        if tok.type in (TokenType.KEYWORD, TokenType.IDENTIFIER) and tok.value.lower() == keyword.lower():
            return self.advance()
        raise ParseError(
            f"Expected keyword '{keyword}', got {tok.type.name} '{tok.value}'",
            tok.line,
            tok.column,
        )

    def _match_keyword(self, keyword: str) -> bool:
        """If the current token is a keyword/identifier matching *keyword*,
        advance and return ``True``."""
        tok = self.peek()
        if tok.type in (TokenType.KEYWORD, TokenType.IDENTIFIER) and tok.value.upper() == keyword.upper():
            self.advance()
            return True
        return False

    def _is_temporal_keyword(self, tok: Token) -> bool:
        """Check whether *tok* is a temporal keyword."""
        return (
            tok.type in (TokenType.KEYWORD, TokenType.IDENTIFIER)
            and tok.value.upper() in _TEMPORAL_KEYWORDS
        )

    def _parse_field_key(self) -> Tuple[str, Token]:
        """Parse a field key followed by a colon and return ``(key, token)``."""
        tok = self.peek()
        if tok.type not in (TokenType.IDENTIFIER, TokenType.KEYWORD, TokenType.STRING):
            raise ParseError(
                f"Expected field name, got {tok.type.name} '{tok.value}'",
                tok.line,
                tok.column,
            )
        self.advance()
        key = tok.value
        self.expect(TokenType.COLON, f"Expected ':' after field name '{key}'")
        self.skip_newlines()
        return (key, tok)

    def _skip_comma_or_newline(self) -> None:
        """Consume an optional comma and/or newlines between fields."""
        self.match(TokenType.COMMA)
        self.skip_newlines()

    def _consume_string_value(self) -> str:
        """Consume and return a string literal value."""
        tok = self.peek()
        if tok.type == TokenType.STRING:
            self.advance()
            return tok.value
        # Allow bare identifiers where a string is expected.
        if tok.type in (TokenType.IDENTIFIER, TokenType.KEYWORD):
            self.advance()
            return tok.value
        raise ParseError(
            f"Expected string value, got {tok.type.name} '{tok.value}'",
            tok.line,
            tok.column,
        )

    def _consume_number_value(self) -> float:
        """Consume and return a numeric literal value."""
        tok = self.peek()
        if tok.type == TokenType.NUMBER:
            self.advance()
            return self._coerce_number(tok.value)
        # Allow negative numbers
        if tok.type == TokenType.MINUS:
            self.advance()
            inner = self.peek()
            if inner.type == TokenType.NUMBER:
                self.advance()
                return -self._coerce_number(inner.value)
            raise ParseError(
                f"Expected number after '-', got {inner.type.name} '{inner.value}'",
                inner.line,
                inner.column,
            )
        raise ParseError(
            f"Expected numeric value, got {tok.type.name} '{tok.value}'",
            tok.line,
            tok.column,
        )

    def _consume_date_or_string(self) -> str:
        """Consume a date literal (``YYYY-MM-DD``), a string, or an
        identifier and return the raw text."""
        tok = self.peek()
        if tok.type == TokenType.STRING:
            self.advance()
            return tok.value
        if tok.type == TokenType.DATE:
            self.advance()
            return tok.value
        # Fall back: accumulate identifier-minus-number pattern for dates
        # represented as bare tokens  (e.g. ``2025-08-01`` without quotes).
        if tok.type in (TokenType.NUMBER, TokenType.IDENTIFIER):
            return self._consume_bare_date_or_value()
        raise ParseError(
            f"Expected date or string, got {tok.type.name} '{tok.value}'",
            tok.line,
            tok.column,
        )

    def _consume_bare_date_or_value(self) -> str:
        """Try to consume a bare date ``YYYY-MM-DD`` that was tokenised as
        separate NUMBER, MINUS, NUMBER, MINUS, NUMBER tokens.  Falls back to
        a single token if the pattern does not match."""
        first = self.advance()
        parts: list[str] = [first.value]

        # Try to collect up to two  ``- NUMBER`` segments.
        for _ in range(2):
            if self.check(TokenType.MINUS):
                saved = self._pos
                self.advance()  # consume '-'
                nxt = self.peek()
                if nxt.type == TokenType.NUMBER:
                    self.advance()
                    parts.append(nxt.value)
                else:
                    # Not a date pattern — rewind the minus.
                    self._pos = saved
                    break
            else:
                break

        return "-".join(parts)

    def _consume_date_or_number_or_string(self) -> str:
        """Consume a deadline value that may be a date, a number, or a
        string and return it as a string."""
        tok = self.peek()
        if tok.type == TokenType.STRING:
            self.advance()
            return tok.value
        if tok.type == TokenType.DATE:
            self.advance()
            return tok.value
        if tok.type == TokenType.NUMBER:
            return self._consume_bare_date_or_value()
        if tok.type in (TokenType.IDENTIFIER, TokenType.KEYWORD):
            self.advance()
            return tok.value
        raise ParseError(
            f"Expected deadline value, got {tok.type.name} '{tok.value}'",
            tok.line,
            tok.column,
        )

    def _parse_framework_type(self) -> FrameworkType:
        """Parse one of the framework-type keywords."""
        tok = self.peek()
        key = tok.value.upper() if tok.value else ""
        if key in _FRAMEWORK_TYPE_MAP:
            self.advance()
            return _FRAMEWORK_TYPE_MAP[key]
        raise ParseError(
            f"Expected framework type (BINDING/VOLUNTARY/HYBRID), got '{tok.value}'",
            tok.line,
            tok.column,
        )

    def _parse_compose_mode(self) -> ComposeMode:
        """Parse a composition mode keyword."""
        tok = self.peek()
        key = tok.value.upper() if tok.value else ""
        if key in _COMPOSE_MODE_MAP:
            self.advance()
            return _COMPOSE_MODE_MAP[key]
        raise ParseError(
            f"Expected compose mode (UNION/INTERSECT/OVERRIDE/SEQUENCE), got '{tok.value}'",
            tok.line,
            tok.column,
        )

    def _parse_penalty_block(self) -> Tuple[Optional[float], Optional[float]]:
        """Parse a penalties block: ``{ min: N, max: M }``."""
        self.expect(TokenType.LBRACE, "Expected '{' to open penalties block")
        self.skip_newlines()
        pmin: Optional[float] = None
        pmax: Optional[float] = None

        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            key, key_tok = self._parse_field_key()
            kl = key.lower()
            if kl == "min":
                pmin = self._consume_number_value()
            elif kl == "max":
                pmax = self._consume_number_value()
            else:
                raise ParseError(
                    f"Unknown penalties field '{key}'",
                    key_tok.line,
                    key_tok.column,
                )
            self._skip_comma_or_newline()

        self.expect(TokenType.RBRACE, "Expected '}' to close penalties block")
        return (pmin, pmax)

    def _parse_list_item(self) -> Expression:
        """Parse a single item inside a list — may be an identifier, literal,
        or full expression."""
        return self.parse_expression()

    @staticmethod
    def _coerce_number(raw: str) -> float | int:
        """Convert a raw numeric string to ``int`` or ``float``."""
        if "." in raw:
            return float(raw)
        try:
            return int(raw)
        except ValueError:
            return float(raw)


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def parse(source: str) -> Program:
    """Tokenize *source* and parse it into a :class:`Program` AST node.

    This is the primary public entry point and is re-exported by the package
    ``__init__``.
    """
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()
