"""Token types and Token class for the TLA-lite lexer.

Defines every token kind produced by the lexer and a Token value object
carrying the kind, lexeme, and source position.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from .source_map import SourceLocation


class TokenKind(Enum):
    """Enumeration of every token type in TLA-lite."""

    # ── Literals ──────────────────────────────────────────────────
    INTEGER = auto()
    STRING = auto()
    BOOL_TRUE = auto()
    BOOL_FALSE = auto()
    IDENTIFIER = auto()

    # ── Keywords ──────────────────────────────────────────────────
    MODULE = auto()
    EXTENDS = auto()
    VARIABLE = auto()
    VARIABLES = auto()
    CONSTANT = auto()
    CONSTANTS = auto()
    ASSUME = auto()
    THEOREM = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    LET = auto()
    IN = auto()
    CASE = auto()
    OTHER = auto()
    CHOOSE = auto()
    UNCHANGED = auto()
    ENABLED = auto()
    EXCEPT = auto()
    WITH = auto()
    LOCAL = auto()
    INSTANCE = auto()
    KW_BOOLEAN = auto()  # BOOLEAN (the set)
    KW_STRING = auto()   # STRING (the set)
    KW_NAT = auto()      # Nat
    KW_INT = auto()       # Int
    DOMAIN = auto()

    # ── Quantifiers ───────────────────────────────────────────────
    FORALL = auto()       # \A
    EXISTS = auto()       # \E
    TEMPORAL_FORALL = auto()  # \AA
    TEMPORAL_EXISTS = auto()  # \EE

    # ── Logical operators ─────────────────────────────────────────
    LAND = auto()         # /\ or \land
    LOR = auto()          # \/ or \lor
    LNOT = auto()         # ~ or \lnot or \neg
    IMPLIES = auto()      # =>
    EQUIV = auto()        # <=>

    # ── Comparison operators ──────────────────────────────────────
    EQ = auto()           # =
    DEF_EQ = auto()       # ==
    NEQ = auto()          # /= or #
    LT = auto()           # <
    GT = auto()           # >
    LEQ = auto()          # <= or \leq
    GEQ = auto()          # >= or \geq

    # ── Arithmetic operators ──────────────────────────────────────
    PLUS = auto()
    MINUS = auto()
    STAR = auto()         # *
    DIV = auto()          # \div
    PERCENT = auto()      # %
    DOTDOT = auto()       # ..

    # ── Set operators ─────────────────────────────────────────────
    SET_IN = auto()       # \in
    SET_NOTIN = auto()    # \notin
    SET_UNION = auto()    # \union or \cup
    SET_INTER = auto()    # \inter or \cap or \intersect
    SET_DIFF = auto()     # \ (set difference)
    SET_SUBSETEQ = auto() # \subseteq
    SET_SUBSET = auto()   # SUBSET (powerset)
    SET_UNION_KW = auto() # UNION (generalized union)
    CROSS = auto()        # \X or \times

    # ── Function / record operators ───────────────────────────────
    COLON_GT = auto()     # :>
    AT_AT = auto()        # @@
    MAPS_TO = auto()      # |->
    BANG = auto()         # !

    # ── Temporal operators ────────────────────────────────────────
    PRIME = auto()        # '
    BOX = auto()          # []
    DIAMOND = auto()      # <>
    LEADS_TO = auto()     # ~>
    BOX_ACTION = auto()   # [][...]_
    DIAMOND_ACTION = auto() # <>[...]_
    WF = auto()           # WF_
    SF = auto()           # SF_

    # ── Delimiters ────────────────────────────────────────────────
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    LANGLE = auto()       # <<
    RANGLE = auto()       # >>
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    DOT = auto()
    UNDERSCORE = auto()   # _
    AT = auto()           # @
    HASH = auto()         # # (also NEQ synonym)

    # ── Structural ────────────────────────────────────────────────
    SEPARATOR = auto()    # ---- (module separator line)
    ASSIGN = auto()       # <-

    # ── Special ───────────────────────────────────────────────────
    NEWLINE = auto()
    EOF = auto()
    ERROR = auto()

    def __repr__(self) -> str:
        return f"TokenKind.{self.name}"


# Sets used during lexing / parsing for quick membership tests
KEYWORDS: dict[str, TokenKind] = {
    "MODULE": TokenKind.MODULE,
    "EXTENDS": TokenKind.EXTENDS,
    "VARIABLE": TokenKind.VARIABLE,
    "VARIABLES": TokenKind.VARIABLES,
    "CONSTANT": TokenKind.CONSTANT,
    "CONSTANTS": TokenKind.CONSTANTS,
    "ASSUME": TokenKind.ASSUME,
    "THEOREM": TokenKind.THEOREM,
    "IF": TokenKind.IF,
    "THEN": TokenKind.THEN,
    "ELSE": TokenKind.ELSE,
    "LET": TokenKind.LET,
    "IN": TokenKind.IN,
    "CASE": TokenKind.CASE,
    "OTHER": TokenKind.OTHER,
    "CHOOSE": TokenKind.CHOOSE,
    "UNCHANGED": TokenKind.UNCHANGED,
    "ENABLED": TokenKind.ENABLED,
    "EXCEPT": TokenKind.EXCEPT,
    "WITH": TokenKind.WITH,
    "LOCAL": TokenKind.LOCAL,
    "INSTANCE": TokenKind.INSTANCE,
    "BOOLEAN": TokenKind.KW_BOOLEAN,
    "STRING": TokenKind.KW_STRING,
    "Nat": TokenKind.KW_NAT,
    "Int": TokenKind.KW_INT,
    "TRUE": TokenKind.BOOL_TRUE,
    "FALSE": TokenKind.BOOL_FALSE,
    "DOMAIN": TokenKind.DOMAIN,
    "SUBSET": TokenKind.SET_SUBSET,
    "UNION": TokenKind.SET_UNION_KW,
    "WF_": TokenKind.WF,
    "SF_": TokenKind.SF,
}

BINARY_OP_TOKENS = frozenset({
    TokenKind.LAND, TokenKind.LOR, TokenKind.IMPLIES, TokenKind.EQUIV,
    TokenKind.EQ, TokenKind.DEF_EQ, TokenKind.NEQ, TokenKind.LT,
    TokenKind.GT, TokenKind.LEQ, TokenKind.GEQ, TokenKind.PLUS,
    TokenKind.MINUS, TokenKind.STAR, TokenKind.DIV, TokenKind.PERCENT,
    TokenKind.DOTDOT, TokenKind.SET_IN, TokenKind.SET_NOTIN,
    TokenKind.SET_UNION, TokenKind.SET_INTER, TokenKind.SET_DIFF,
    TokenKind.SET_SUBSETEQ, TokenKind.CROSS, TokenKind.COLON_GT,
    TokenKind.AT_AT, TokenKind.LEADS_TO,
})

UNARY_PREFIX_TOKENS = frozenset({
    TokenKind.LNOT, TokenKind.MINUS, TokenKind.SET_SUBSET,
    TokenKind.SET_UNION_KW, TokenKind.DOMAIN, TokenKind.ENABLED,
    TokenKind.UNCHANGED, TokenKind.BOX, TokenKind.DIAMOND,
})


@dataclass(frozen=True, slots=True)
class Token:
    """A single lexical token.

    Attributes:
        kind: The token type.
        value: The lexeme text (or parsed value for literals).
        location: Source position information.
    """
    kind: TokenKind
    value: Any = None
    location: SourceLocation = SourceLocation()

    @property
    def line(self) -> int:
        return self.location.line

    @property
    def column(self) -> int:
        return self.location.column

    def is_keyword(self) -> bool:
        return self.kind in KEYWORDS.values()

    def is_operator(self) -> bool:
        return self.kind in BINARY_OP_TOKENS or self.kind in UNARY_PREFIX_TOKENS

    def is_literal(self) -> bool:
        return self.kind in (
            TokenKind.INTEGER, TokenKind.STRING,
            TokenKind.BOOL_TRUE, TokenKind.BOOL_FALSE,
        )

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.kind.name}({self.value!r})"
        return self.kind.name

    def __repr__(self) -> str:
        return (
            f"Token(kind={self.kind!r}, value={self.value!r}, "
            f"loc={self.location})"
        )
