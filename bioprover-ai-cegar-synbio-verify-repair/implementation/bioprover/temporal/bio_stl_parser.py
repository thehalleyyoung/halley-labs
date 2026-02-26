"""Bio-STL Parser with domain-specific macros for synthetic biology.

Implements a recursive descent parser for Bio-STL, an extension of STL with
biology-specific macros such as oscillates, bistable, adapts, etc.
Macros are expanded into base STL during parsing.

Grammar (informal):
    formula := atomic | '!' formula | formula '&&' formula | formula '||' formula
             | formula '->' formula
             | 'G' interval '(' formula ')' | 'F' interval '(' formula ')'
             | formula 'U' interval formula
             | macro_call | '(' formula ')'
    interval := '[' number ',' number ']'
    atomic := expr op expr
    op := '<' | '<=' | '>' | '>='
    expr := identifier | number | identifier ('+' | '-') number
    macro_call := macro_name '(' args ')'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
    Expression,
    Interval,
    Predicate,
    STLAnd,
    STLFormula,
    STLImplies,
    STLNot,
    STLOr,
    Until,
    make_const_expr,
    make_var_expr,
)


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(Enum):
    IDENT = auto()
    NUMBER = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    AND = auto()       # &&
    OR = auto()        # ||
    NOT = auto()       # !
    IMPLIES = auto()   # ->
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    G = auto()         # G (globally)
    F = auto()         # F (eventually/finally)
    U = auto()         # U (until)
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


class ParseError(Exception):
    """Raised when the parser encounters a syntax error."""

    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Parse error at L{line}:{col}: {message}")


# ---------------------------------------------------------------------------
# Lexer / Tokenizer
# ---------------------------------------------------------------------------

_KEYWORDS = {"G": TokenType.G, "F": TokenType.F, "U": TokenType.U}

_TOKEN_PATTERNS = [
    (r"&&", TokenType.AND),
    (r"\|\|", TokenType.OR),
    (r"->", TokenType.IMPLIES),
    (r"<=", TokenType.LE),
    (r">=", TokenType.GE),
    (r"<", TokenType.LT),
    (r">", TokenType.GT),
    (r"!", TokenType.NOT),
    (r"\(", TokenType.LPAREN),
    (r"\)", TokenType.RPAREN),
    (r"\[", TokenType.LBRACKET),
    (r"\]", TokenType.RBRACKET),
    (r",", TokenType.COMMA),
    (r"\+", TokenType.PLUS),
    (r"-(?!\d)", TokenType.MINUS),
    (r"\*", TokenType.STAR),
    (r"[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?", TokenType.NUMBER),
    (r"[a-zA-Z_][a-zA-Z0-9_]*", TokenType.IDENT),
]

_COMPILED_PATTERNS = [(re.compile(p), tt) for p, tt in _TOKEN_PATTERNS]


def tokenize(text: str) -> List[Token]:
    """Tokenize a Bio-STL formula string."""
    tokens: List[Token] = []
    line = 1
    col = 1
    i = 0
    while i < len(text):
        if text[i] == "\n":
            line += 1
            col = 1
            i += 1
            continue
        if text[i] in " \t\r":
            col += 1
            i += 1
            continue
        # Try negative number (only after operator tokens or start)
        if text[i] == "-" and i + 1 < len(text) and text[i + 1].isdigit():
            if not tokens or tokens[-1].type in (
                TokenType.LPAREN, TokenType.LBRACKET, TokenType.COMMA,
                TokenType.AND, TokenType.OR, TokenType.NOT, TokenType.IMPLIES,
                TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE,
            ):
                m = re.match(r"-[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?", text[i:])
                if m:
                    tokens.append(Token(TokenType.NUMBER, m.group(), line, col))
                    advance = len(m.group())
                    col += advance
                    i += advance
                    continue

        matched = False
        for pattern, tt in _COMPILED_PATTERNS:
            m = pattern.match(text, i)
            if m:
                val = m.group()
                if tt == TokenType.IDENT and val in _KEYWORDS:
                    tt = _KEYWORDS[val]
                tokens.append(Token(tt, val, line, col))
                advance = len(val)
                col += advance
                i += advance
                matched = True
                break
        if not matched:
            raise ParseError(f"Unexpected character: {text[i]!r}", line, col)

    tokens.append(Token(TokenType.EOF, "", line, col))
    return tokens


# ---------------------------------------------------------------------------
# Bio-STL Macros
# ---------------------------------------------------------------------------

_MACRO_REGISTRY: Dict[str, "MacroDef"] = {}


@dataclass
class MacroDef:
    """Definition of a Bio-STL macro."""
    name: str
    param_names: List[str]
    expand: "MacroExpander"


MacroExpander = type(lambda args: None)  # Callable[[List[str]], STLFormula]


def _register_macro(name: str, param_names: List[str],
                     expand_fn: "MacroExpander") -> None:
    _MACRO_REGISTRY[name] = MacroDef(name, param_names, expand_fn)


def _expand_oscillates(args: List[str]) -> STLFormula:
    """oscillates(species, period, amplitude) -> periodic peaks and troughs."""
    species, period, amplitude = args[0], float(args[1]), float(args[2])
    half = period / 2.0
    peak = Predicate(make_var_expr(species), ComparisonOp.GT, amplitude)
    trough = Predicate(make_var_expr(species), ComparisonOp.LT, -amplitude)
    return Always(
        STLAnd(
            Eventually(peak, Interval(0, half)),
            Eventually(trough, Interval(0, half)),
        ),
        Interval(0, period * 3),
    )


def _expand_bistable(args: List[str]) -> STLFormula:
    """bistable(species, low_state, high_state) -> eventually in one of two states."""
    species = args[0]
    low = float(args[1])
    high = float(args[2])
    margin = (high - low) * 0.1
    near_low = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, low - margin),
        Predicate(make_var_expr(species), ComparisonOp.LE, low + margin),
    )
    near_high = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, high - margin),
        Predicate(make_var_expr(species), ComparisonOp.LE, high + margin),
    )
    return Eventually(STLOr(near_low, near_high), Interval(0, 100))


def _expand_adapts(args: List[str]) -> STLFormula:
    """adapts(species, stimulus, adaptation_time) -> returns to baseline after stimulus."""
    species = args[0]
    stimulus = float(args[1])
    adapt_time = float(args[2])
    baseline_tol = stimulus * 0.1
    # After a transient response, species returns near baseline (0)
    responds = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.GT, stimulus * 0.5),
        Interval(0, adapt_time * 0.3),
    )
    returns = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.LT, baseline_tol),
        Interval(adapt_time * 0.3, adapt_time),
    )
    return STLAnd(responds, returns)


def _expand_monotone_response(args: List[str]) -> STLFormula:
    """monotone_response(input, output, delay) -> output follows input monotonically."""
    inp, out = args[0], args[1]
    delay = float(args[2])
    # If input high, eventually output high
    inp_high = Predicate(make_var_expr(inp), ComparisonOp.GT, 0.5)
    out_high = Predicate(make_var_expr(out), ComparisonOp.GT, 0.5)
    return Always(
        STLImplies(inp_high, Eventually(out_high, Interval(0, delay))),
        Interval(0, delay * 5),
    )


def _expand_pulse(args: List[str]) -> STLFormula:
    """pulse(species, duration, amplitude) -> transient spike then return."""
    species = args[0]
    duration = float(args[1])
    amplitude = float(args[2])
    spike = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.GT, amplitude),
        Interval(0, duration),
    )
    returns = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.LT, amplitude * 0.1),
        Interval(duration, duration * 2),
    )
    return STLAnd(spike, returns)


def _expand_reaches_steady_state(args: List[str]) -> STLFormula:
    """reaches_steady_state(species, value, tolerance, time) -> convergence."""
    species = args[0]
    value = float(args[1])
    tol = float(args[2])
    time = float(args[3])
    in_band = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, value - tol),
        Predicate(make_var_expr(species), ComparisonOp.LE, value + tol),
    )
    return Eventually(Always(in_band, Interval(0, time * 0.5)), Interval(0, time))


def _expand_switches(args: List[str]) -> STLFormula:
    """switches(species, from_val, to_val, time) -> switching behavior."""
    species = args[0]
    from_val = float(args[1])
    to_val = float(args[2])
    time = float(args[3])
    tol = abs(to_val - from_val) * 0.1
    initially_near = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, from_val - tol),
        Predicate(make_var_expr(species), ComparisonOp.LE, from_val + tol),
    )
    finally_near = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, to_val - tol),
        Predicate(make_var_expr(species), ComparisonOp.LE, to_val + tol),
    )
    return STLAnd(
        initially_near,
        Eventually(Always(finally_near, Interval(0, time * 0.3)), Interval(0, time)),
    )


# Register all macros
_register_macro("oscillates", ["species", "period", "amplitude"], _expand_oscillates)
_register_macro("bistable", ["species", "low_state", "high_state"], _expand_bistable)
_register_macro("adapts", ["species", "stimulus", "adaptation_time"], _expand_adapts)
_register_macro("monotone_response", ["input", "output", "delay"], _expand_monotone_response)
_register_macro("pulse", ["species", "duration", "amplitude"], _expand_pulse)
_register_macro("reaches_steady_state", ["species", "value", "tolerance", "time"],
                _expand_reaches_steady_state)
_register_macro("switches", ["species", "from_val", "to_val", "time"], _expand_switches)


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# ---------------------------------------------------------------------------

class BioSTLParser:
    """Recursive descent parser for Bio-STL formulas.

    Usage::

        parser = BioSTLParser()
        formula = parser.parse("G[0,10](x > 5)")
        formula = parser.parse("oscillates(GFP, 20, 0.5)")
    """

    def __init__(self, custom_macros: Optional[Dict[str, MacroDef]] = None) -> None:
        self._macros = dict(_MACRO_REGISTRY)
        if custom_macros:
            self._macros.update(custom_macros)
        self._tokens: List[Token] = []
        self._pos: int = 0

    def parse(self, text: str) -> STLFormula:
        """Parse a Bio-STL formula string into an STL AST."""
        self._tokens = tokenize(text)
        self._pos = 0
        formula = self._parse_implies()
        if self._peek().type != TokenType.EOF:
            tok = self._peek()
            raise ParseError(
                f"Unexpected token {tok.value!r} after formula",
                tok.line, tok.col,
            )
        return formula

    # --- Token helpers ---

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._advance()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})",
                tok.line, tok.col,
            )
        return tok

    def _match(self, tt: TokenType) -> Optional[Token]:
        if self._peek().type == tt:
            return self._advance()
        return None

    # --- Recursive descent ---

    def _parse_implies(self) -> STLFormula:
        left = self._parse_or()
        while self._peek().type == TokenType.IMPLIES:
            self._advance()
            right = self._parse_or()
            left = STLImplies(left, right)
        return left

    def _parse_or(self) -> STLFormula:
        left = self._parse_and()
        while self._peek().type == TokenType.OR:
            self._advance()
            right = self._parse_and()
            left = STLOr(left, right)
        return left

    def _parse_and(self) -> STLFormula:
        left = self._parse_until()
        while self._peek().type == TokenType.AND:
            self._advance()
            right = self._parse_until()
            left = STLAnd(left, right)
        return left

    def _parse_until(self) -> STLFormula:
        left = self._parse_unary()
        if self._peek().type == TokenType.U:
            self._advance()
            interval = self._parse_interval()
            right = self._parse_unary()
            left = Until(left, right, interval)
        return left

    def _parse_unary(self) -> STLFormula:
        tok = self._peek()

        # Negation
        if tok.type == TokenType.NOT:
            self._advance()
            child = self._parse_unary()
            return STLNot(child)

        # Globally
        if tok.type == TokenType.G:
            self._advance()
            interval = self._parse_interval()
            self._expect(TokenType.LPAREN)
            child = self._parse_implies()
            self._expect(TokenType.RPAREN)
            return Always(child, interval)

        # Eventually
        if tok.type == TokenType.F:
            self._advance()
            interval = self._parse_interval()
            self._expect(TokenType.LPAREN)
            child = self._parse_implies()
            self._expect(TokenType.RPAREN)
            return Eventually(child, interval)

        return self._parse_primary()

    def _parse_primary(self) -> STLFormula:
        tok = self._peek()

        # Parenthesized formula
        if tok.type == TokenType.LPAREN:
            self._advance()
            formula = self._parse_implies()
            self._expect(TokenType.RPAREN)
            return formula

        # Identifier: either a macro call or start of atomic predicate
        if tok.type == TokenType.IDENT:
            # Check for macro
            if tok.value in self._macros and self._lookahead_is_macro_call():
                return self._parse_macro_call()
            # Atomic predicate: expr op expr
            return self._parse_atomic()

        # Number starting an atomic predicate
        if tok.type == TokenType.NUMBER:
            return self._parse_atomic()

        raise ParseError(
            f"Unexpected token {tok.value!r} in formula",
            tok.line, tok.col,
        )

    def _lookahead_is_macro_call(self) -> bool:
        """Check if the current identifier is followed by '(' for a macro call."""
        if self._pos + 1 < len(self._tokens):
            return self._tokens[self._pos + 1].type == TokenType.LPAREN
        return False

    def _parse_macro_call(self) -> STLFormula:
        """Parse a macro invocation: name(arg1, arg2, ...)."""
        name_tok = self._advance()
        macro_def = self._macros[name_tok.value]
        self._expect(TokenType.LPAREN)
        args: List[str] = []
        if self._peek().type != TokenType.RPAREN:
            args.append(self._parse_macro_arg())
            while self._match(TokenType.COMMA):
                args.append(self._parse_macro_arg())
        self._expect(TokenType.RPAREN)

        if len(args) != len(macro_def.param_names):
            raise ParseError(
                f"Macro {name_tok.value} expects {len(macro_def.param_names)} "
                f"arguments ({', '.join(macro_def.param_names)}), got {len(args)}",
                name_tok.line, name_tok.col,
            )
        return macro_def.expand(args)

    def _parse_macro_arg(self) -> str:
        """Parse a single macro argument (identifier or number)."""
        tok = self._peek()
        if tok.type == TokenType.IDENT:
            self._advance()
            return tok.value
        if tok.type == TokenType.NUMBER:
            self._advance()
            return tok.value
        # Allow negative numbers
        if tok.type == TokenType.MINUS:
            self._advance()
            num = self._expect(TokenType.NUMBER)
            return f"-{num.value}"
        raise ParseError(
            f"Expected macro argument (identifier or number), got {tok.value!r}",
            tok.line, tok.col,
        )

    def _parse_atomic(self) -> Predicate:
        """Parse an atomic predicate: expr op threshold."""
        expr = self._parse_expression()
        op = self._parse_comparison_op()
        threshold_tok = self._advance()
        if threshold_tok.type == TokenType.NUMBER:
            threshold = float(threshold_tok.value)
        elif threshold_tok.type == TokenType.MINUS:
            num_tok = self._expect(TokenType.NUMBER)
            threshold = -float(num_tok.value)
        else:
            raise ParseError(
                f"Expected number after comparison operator, got {threshold_tok.value!r}",
                threshold_tok.line, threshold_tok.col,
            )
        return Predicate(expr, op, threshold)

    def _parse_expression(self) -> Expression:
        """Parse a simple signal expression: var, number, var +/- number, scale*var."""
        tok = self._peek()

        if tok.type == TokenType.NUMBER:
            self._advance()
            val = float(tok.value)
            # Check for scale * var
            if self._peek().type == TokenType.STAR:
                self._advance()
                var_tok = self._expect(TokenType.IDENT)
                expr = Expression(variable=var_tok.value, scale=val)
                return self._parse_expr_offset(expr)
            return Expression(constant=val)

        if tok.type == TokenType.IDENT:
            self._advance()
            expr = Expression(variable=tok.value)
            return self._parse_expr_offset(expr)

        raise ParseError(
            f"Expected expression, got {tok.value!r}",
            tok.line, tok.col,
        )

    def _parse_expr_offset(self, expr: Expression) -> Expression:
        """Optionally parse + or - offset after a variable expression."""
        if self._peek().type == TokenType.PLUS:
            self._advance()
            num = self._expect(TokenType.NUMBER)
            expr.offset = float(num.value)
        elif self._peek().type == TokenType.MINUS:
            self._advance()
            num = self._expect(TokenType.NUMBER)
            expr.offset = -float(num.value)
        return expr

    def _parse_comparison_op(self) -> ComparisonOp:
        tok = self._advance()
        mapping = {
            TokenType.LT: ComparisonOp.LT,
            TokenType.LE: ComparisonOp.LE,
            TokenType.GT: ComparisonOp.GT,
            TokenType.GE: ComparisonOp.GE,
        }
        if tok.type not in mapping:
            raise ParseError(
                f"Expected comparison operator (<, <=, >, >=), got {tok.value!r}",
                tok.line, tok.col,
            )
        return mapping[tok.type]

    def _parse_interval(self) -> Interval:
        """Parse a time interval [lo, hi]."""
        self._expect(TokenType.LBRACKET)
        lo_tok = self._advance()
        if lo_tok.type != TokenType.NUMBER:
            raise ParseError(
                f"Expected number for interval lower bound, got {lo_tok.value!r}",
                lo_tok.line, lo_tok.col,
            )
        self._expect(TokenType.COMMA)
        hi_tok = self._advance()
        if hi_tok.type != TokenType.NUMBER:
            raise ParseError(
                f"Expected number for interval upper bound, got {hi_tok.value!r}",
                hi_tok.line, hi_tok.col,
            )
        self._expect(TokenType.RBRACKET)
        return Interval(float(lo_tok.value), float(hi_tok.value))

    # --- Utility ---

    @staticmethod
    def available_macros() -> List[str]:
        """Return list of available Bio-STL macros and their signatures."""
        result = []
        for name, mdef in sorted(_MACRO_REGISTRY.items()):
            sig = f"{name}({', '.join(mdef.param_names)})"
            result.append(sig)
        return result
