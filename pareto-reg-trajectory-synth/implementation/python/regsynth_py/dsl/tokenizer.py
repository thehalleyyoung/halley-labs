"""
Tokenizer for the RegSynth regulatory compliance DSL.

Produces a token stream compatible with what the Rust lexer emits, including
keyword recognition, date literals (YYYY-MM-DD), string escapes, and
single/multi-line comments.  All keywords are matched case-insensitively.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(enum.Enum):
    """Every token the RegSynth lexer can emit."""

    # -- keywords (sorted alphabetically) -----------------------------------
    OBLIGATION = "OBLIGATION"
    JURISDICTION = "JURISDICTION"
    STRATEGY = "STRATEGY"
    COMPOSE = "COMPOSE"
    CONSTRAINT = "CONSTRAINT"
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"
    IF = "IF"
    THEN = "THEN"
    ELSE = "ELSE"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    REQUIRES = "REQUIRES"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    WITHIN = "WITHIN"
    EVERY = "EVERY"
    RISK = "RISK"
    HIGH = "HIGH"
    LIMITED = "LIMITED"
    MINIMAL = "MINIMAL"
    UNACCEPTABLE = "UNACCEPTABLE"
    MANDATORY = "MANDATORY"
    RECOMMENDED = "RECOMMENDED"
    OPTIONAL = "OPTIONAL"
    CONDITIONAL = "CONDITIONAL"
    BINDING = "BINDING"
    VOLUNTARY = "VOLUNTARY"
    HYBRID = "HYBRID"
    UNION = "UNION"
    INTERSECT = "INTERSECT"
    SEQUENCE = "SEQUENCE"
    OVERRIDE = "OVERRIDE"
    TRUE = "TRUE"
    FALSE = "FALSE"
    DEFINE = "DEFINE"
    WHERE = "WHERE"
    WITH = "WITH"
    AS = "AS"
    FOR = "FOR"
    IN = "IN"
    PHASE = "PHASE"
    DEADLINE = "DEADLINE"
    PENALTY = "PENALTY"
    ARTICLE = "ARTICLE"
    CATEGORY = "CATEGORY"
    REGION = "REGION"
    FRAMEWORK = "FRAMEWORK"

    # -- literals -----------------------------------------------------------
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    STRING = "STRING"
    DATE = "DATE"

    # -- identifier ---------------------------------------------------------
    IDENT = "IDENT"

    # -- punctuation --------------------------------------------------------
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    COLON = "COLON"
    DOT = "DOT"
    ARROW = "ARROW"
    FAT_ARROW = "FAT_ARROW"
    ASSIGN = "ASSIGN"
    PIPE = "PIPE"

    # -- operators ----------------------------------------------------------
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    GT = "GT"
    LTE = "LTE"
    GTE = "GTE"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    PERCENT = "PERCENT"

    # -- special ------------------------------------------------------------
    COMMENT = "COMMENT"
    NEWLINE = "NEWLINE"
    EOF = "EOF"
    INDENT = "INDENT"
    DEDENT = "DEDENT"


# Keyword lookup — maps UPPER-CASED word to its TokenType
_KEYWORDS: dict[str, TokenType] = {
    tt.value: tt
    for tt in TokenType
    if tt.value.isalpha() and tt not in {
        TokenType.INTEGER, TokenType.FLOAT, TokenType.STRING,
        TokenType.DATE, TokenType.IDENT,
        TokenType.LPAREN, TokenType.RPAREN, TokenType.LBRACE,
        TokenType.RBRACE, TokenType.LBRACKET, TokenType.RBRACKET,
        TokenType.COMMA, TokenType.SEMICOLON, TokenType.COLON,
        TokenType.DOT, TokenType.ARROW, TokenType.FAT_ARROW,
        TokenType.ASSIGN, TokenType.PIPE,
        TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT,
        TokenType.LTE, TokenType.GTE, TokenType.PLUS, TokenType.MINUS,
        TokenType.STAR, TokenType.SLASH, TokenType.PERCENT,
        TokenType.COMMENT, TokenType.NEWLINE, TokenType.EOF,
        TokenType.INDENT, TokenType.DEDENT,
    }
}


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Token:
    """A single lexical token."""
    token_type: TokenType
    value: str
    line: int
    column: int
    offset: int

    def __repr__(self) -> str:
        return (
            f"Token({self.token_type.name}, {self.value!r}, "
            f"line={self.line}, col={self.column})"
        )


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class TokenError(Exception):
    """Raised when the tokenizer encounters invalid input."""

    def __init__(self, message: str, line: int, column: int) -> None:
        self.line = line
        self.column = column
        super().__init__(f"TokenError at {line}:{column}: {message}")


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_IDENT_START = re.compile(r"[A-Za-z_]")
_IDENT_CONT = re.compile(r"[A-Za-z0-9_]")
_DIGIT = re.compile(r"[0-9]")

# String escape map
_ESCAPE_MAP: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    '"': '"',
    "'": "'",
    "0": "\0",
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[Token]:
    """
    Lex *source* into a list of ``Token`` objects.

    Raises ``TokenError`` on invalid input.
    """
    tokens: list[Token] = []
    length = len(source)
    pos = 0
    line = 1
    col = 1

    def _peek(offset: int = 0) -> Optional[str]:
        idx = pos + offset
        return source[idx] if idx < length else None

    def _advance(n: int = 1) -> str:
        nonlocal pos, col
        text = source[pos : pos + n]
        pos += n
        col += n
        return text

    def _emit(tt: TokenType, value: str, start_line: int, start_col: int, start_offset: int) -> None:
        tokens.append(Token(tt, value, start_line, start_col, start_offset))

    while pos < length:
        start_line = line
        start_col = col
        start_offset = pos
        ch = source[pos]

        # -- newlines -------------------------------------------------------
        if ch == "\n":
            _emit(TokenType.NEWLINE, "\n", start_line, start_col, start_offset)
            _advance()
            line += 1
            col = 1
            continue

        if ch == "\r":
            _advance()
            if _peek() == "\n":
                _advance()
            _emit(TokenType.NEWLINE, "\n", start_line, start_col, start_offset)
            line += 1
            col = 1
            continue

        # -- whitespace (skip, not newline) ---------------------------------
        if ch in (" ", "\t"):
            _advance()
            continue

        # -- single-line comment: // or # -----------------------------------
        if ch == "/" and _peek(1) == "/":
            buf = ""
            while pos < length and source[pos] != "\n":
                buf += _advance()
            _emit(TokenType.COMMENT, buf, start_line, start_col, start_offset)
            continue

        if ch == "#":
            buf = ""
            while pos < length and source[pos] != "\n":
                buf += _advance()
            _emit(TokenType.COMMENT, buf, start_line, start_col, start_offset)
            continue

        # -- multi-line comment: /* ... */ ----------------------------------
        if ch == "/" and _peek(1) == "*":
            buf = _advance(2)  # consume /*
            depth = 1
            while pos < length and depth > 0:
                c = source[pos]
                if c == "/" and _peek(1) == "*":
                    buf += _advance(2)
                    depth += 1
                elif c == "*" and _peek(1) == "/":
                    buf += _advance(2)
                    depth -= 1
                else:
                    if c == "\n":
                        line += 1
                        col = 0  # _advance will set to 1
                    buf += _advance()
            if depth != 0:
                raise TokenError("Unterminated block comment", start_line, start_col)
            _emit(TokenType.COMMENT, buf, start_line, start_col, start_offset)
            continue

        # -- string literal -------------------------------------------------
        if ch in ('"', "'"):
            quote = ch
            _advance()  # opening quote
            buf = ""
            while pos < length:
                c = source[pos]
                if c == "\n":
                    raise TokenError("Unterminated string literal", start_line, start_col)
                if c == "\\":
                    _advance()  # backslash
                    esc = _peek()
                    if esc is None:
                        raise TokenError("Unterminated escape in string", start_line, start_col)
                    if esc in _ESCAPE_MAP:
                        buf += _ESCAPE_MAP[esc]
                        _advance()
                    elif esc == "u":
                        _advance()  # consume 'u'
                        hex_chars = ""
                        for _ in range(4):
                            hc = _peek()
                            if hc is None or hc not in "0123456789abcdefABCDEF":
                                raise TokenError(
                                    f"Invalid unicode escape \\u{hex_chars}",
                                    start_line, start_col,
                                )
                            hex_chars += _advance()
                        buf += chr(int(hex_chars, 16))
                    else:
                        raise TokenError(
                            f"Unknown escape sequence \\{esc}",
                            start_line, start_col,
                        )
                elif c == quote:
                    _advance()  # closing quote
                    break
                else:
                    buf += _advance()
            else:
                raise TokenError("Unterminated string literal", start_line, start_col)
            _emit(TokenType.STRING, buf, start_line, start_col, start_offset)
            continue

        # -- numeric / date literals ----------------------------------------
        if ch.isdigit():
            # Try date literal first: exactly YYYY-MM-DD
            if pos + 10 <= length:
                candidate = source[pos : pos + 10]
                after_date = source[pos + 10] if pos + 10 < length else None
                if (
                    _DATE_RE.fullmatch(candidate)
                    and (after_date is None or not _IDENT_CONT.match(after_date))
                ):
                    _advance(10)
                    _emit(TokenType.DATE, candidate, start_line, start_col, start_offset)
                    continue

            # Integer or float
            num = ""
            while pos < length and source[pos].isdigit():
                num += _advance()

            if _peek() == "." and pos + 1 < length and source[pos + 1].isdigit():
                num += _advance()  # consume '.'
                while pos < length and source[pos].isdigit():
                    num += _advance()
                _emit(TokenType.FLOAT, num, start_line, start_col, start_offset)
            else:
                _emit(TokenType.INTEGER, num, start_line, start_col, start_offset)
            continue

        # -- identifiers / keywords -----------------------------------------
        if _IDENT_START.match(ch):
            buf = ""
            while pos < length and _IDENT_CONT.match(source[pos]):
                buf += _advance()
            upper = buf.upper()
            tt = _KEYWORDS.get(upper, TokenType.IDENT)
            _emit(tt, buf, start_line, start_col, start_offset)
            continue

        # -- two-character punctuation / operators --------------------------
        two = source[pos : pos + 2] if pos + 1 < length else None

        if two == "=>":
            _advance(2)
            _emit(TokenType.FAT_ARROW, "=>", start_line, start_col, start_offset)
            continue
        if two == "->":
            _advance(2)
            _emit(TokenType.ARROW, "->", start_line, start_col, start_offset)
            continue
        if two == "==":
            _advance(2)
            _emit(TokenType.EQ, "==", start_line, start_col, start_offset)
            continue
        if two == "!=":
            _advance(2)
            _emit(TokenType.NEQ, "!=", start_line, start_col, start_offset)
            continue
        if two == "<=":
            _advance(2)
            _emit(TokenType.LTE, "<=", start_line, start_col, start_offset)
            continue
        if two == ">=":
            _advance(2)
            _emit(TokenType.GTE, ">=", start_line, start_col, start_offset)
            continue

        # -- single-character punctuation / operators -----------------------
        _SINGLE: dict[str, TokenType] = {
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ",": TokenType.COMMA,
            ";": TokenType.SEMICOLON,
            ":": TokenType.COLON,
            ".": TokenType.DOT,
            "=": TokenType.ASSIGN,
            "|": TokenType.PIPE,
            "<": TokenType.LT,
            ">": TokenType.GT,
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
        }

        if ch in _SINGLE:
            _advance()
            _emit(_SINGLE[ch], ch, start_line, start_col, start_offset)
            continue

        # -- unknown character ----------------------------------------------
        raise TokenError(f"Unexpected character {ch!r}", start_line, start_col)

    # Append EOF
    _emit(TokenType.EOF, "", line, col, pos)
    return tokens
