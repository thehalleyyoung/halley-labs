"""Lexer for the TLA-lite fragment.

Converts raw source text into a stream of Token objects. Handles all TLA+
operators, keywords, literals, comments, and whitespace according to the
TLA-lite subset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from .source_map import SourceLocation
from .tokens import KEYWORDS, Token, TokenKind


class LexerError(Exception):
    """Raised on unrecoverable lexer errors."""

    def __init__(self, message: str, line: int, column: int, file: str = "<string>"):
        self.message = message
        self.line = line
        self.column = column
        self.file = file
        super().__init__(f"{file}:{line}:{column}: {message}")


@dataclass
class _LexerState:
    """Mutable state carried through lexing."""
    source: str
    file: str
    pos: int = 0
    line: int = 1
    column: int = 1
    tokens: List[Token] = field(default_factory=list)
    errors: List[LexerError] = field(default_factory=list)

    @property
    def at_end(self) -> bool:
        return self.pos >= len(self.source)

    def peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return "\0"

    def peek_str(self, length: int) -> str:
        return self.source[self.pos: self.pos + length]

    def advance(self, n: int = 1) -> str:
        text = self.source[self.pos: self.pos + n]
        for ch in text:
            if ch == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        self.pos += n
        return text

    def make_loc(self, start_line: int, start_col: int, start_offset: int) -> SourceLocation:
        return SourceLocation(
            file=self.file,
            line=start_line,
            column=start_col,
            end_line=self.line,
            end_column=self.column,
            offset=start_offset,
            length=self.pos - start_offset,
        )


# ── Backslash operator table ────────────────────────────────────────────────
_BACKSLASH_OPS: dict[str, TokenKind] = {
    "in": TokenKind.SET_IN,
    "notin": TokenKind.SET_NOTIN,
    "union": TokenKind.SET_UNION,
    "cup": TokenKind.SET_UNION,
    "intersect": TokenKind.SET_INTER,
    "inter": TokenKind.SET_INTER,
    "cap": TokenKind.SET_INTER,
    "subseteq": TokenKind.SET_SUBSETEQ,
    "X": TokenKind.CROSS,
    "times": TokenKind.CROSS,
    "div": TokenKind.DIV,
    "A": TokenKind.FORALL,
    "E": TokenKind.EXISTS,
    "AA": TokenKind.TEMPORAL_FORALL,
    "EE": TokenKind.TEMPORAL_EXISTS,
    "land": TokenKind.LAND,
    "lor": TokenKind.LOR,
    "lnot": TokenKind.LNOT,
    "neg": TokenKind.LNOT,
    "leq": TokenKind.LEQ,
    "geq": TokenKind.GEQ,
    "o": TokenKind.AT_AT,  # \o (composition) — mapped to @@
}

# Simple two-char operators
_TWO_CHAR_OPS: dict[str, TokenKind] = {
    "==": TokenKind.DEF_EQ,
    "/=": TokenKind.NEQ,
    "=>": TokenKind.IMPLIES,
    "<=": TokenKind.LEQ,
    ">=": TokenKind.GEQ,
    "..": TokenKind.DOTDOT,
    ":>": TokenKind.COLON_GT,
    "@@": TokenKind.AT_AT,
    "~>": TokenKind.LEADS_TO,
    "<<": TokenKind.LANGLE,
    ">>": TokenKind.RANGLE,
    "<-": TokenKind.ASSIGN,
    "|->": TokenKind.MAPS_TO,
}

_SINGLE_CHAR_OPS: dict[str, TokenKind] = {
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    ",": TokenKind.COMMA,
    ":": TokenKind.COLON,
    ";": TokenKind.SEMICOLON,
    ".": TokenKind.DOT,
    "~": TokenKind.LNOT,
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "*": TokenKind.STAR,
    "%": TokenKind.PERCENT,
    "'": TokenKind.PRIME,
    "_": TokenKind.UNDERSCORE,
    "@": TokenKind.AT,
    "#": TokenKind.NEQ,
    "!": TokenKind.BANG,
    "=": TokenKind.EQ,
    "<": TokenKind.LT,
    ">": TokenKind.GT,
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INTEGER_RE = re.compile(r"(0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+|[0-9]+)")


class Lexer:
    """Tokeniser for TLA-lite source text.

    Usage::

        lexer = Lexer(source_text, file="Spec.tla")
        tokens = lexer.tokenize()
    """

    def __init__(self, source: str, file: str = "<string>") -> None:
        self._state = _LexerState(source=source, file=file)

    # ── Public API ──────────────────────────────────────────────────

    def tokenize(self) -> List[Token]:
        """Lex the entire source and return the token list (including EOF)."""
        st = self._state
        while not st.at_end:
            self._scan_token()
        st.tokens.append(Token(
            kind=TokenKind.EOF,
            value=None,
            location=st.make_loc(st.line, st.column, st.pos),
        ))
        return st.tokens

    @property
    def errors(self) -> List[LexerError]:
        return self._state.errors

    # ── Main dispatch ───────────────────────────────────────────────

    def _scan_token(self) -> None:
        st = self._state
        ch = st.peek()

        # Skip whitespace (not newlines — they can be structurally significant)
        if ch in (" ", "\t", "\r"):
            st.advance()
            return

        if ch == "\n":
            st.advance()
            return

        # Line continuation
        if ch == "\\" and st.peek(1) == "\n":
            st.advance(2)
            return

        start_line, start_col, start_off = st.line, st.column, st.pos

        # ── Comments ────────────────────────────────────────────────
        if ch == "\\" and st.peek(1) == "*":
            self._scan_line_comment()
            return

        if ch == "(" and st.peek(1) == "*":
            self._scan_block_comment()
            return

        # ── Separator lines  ---- ... ---- ─────────────────────────
        if ch == "-" and st.peek_str(4) == "----":
            self._scan_separator()
            return

        # ── Equality separator  ==== ... ==== ──────────────────────
        if ch == "=" and st.peek_str(4) == "====":
            self._scan_eq_separator()
            return

        # ── String literal ──────────────────────────────────────────
        if ch == '"':
            self._scan_string()
            return

        # ── Integer literal ─────────────────────────────────────────
        if ch.isdigit():
            self._scan_integer()
            return

        # ── Backslash operators / quantifiers ───────────────────────
        if ch == "\\":
            self._scan_backslash_op()
            return

        # ── Logical connectives with slash ──────────────────────────
        # /\ (conjunction)
        if ch == "/" and st.peek(1) == "\\":
            st.advance(2)
            self._emit(TokenKind.LAND, "/\\", start_line, start_col, start_off)
            return

        # \/ (disjunction) — handled in _scan_backslash_op via
        # the initial '\' char.  But if we see a bare / followed by
        # something that isn't '\' or '=', it's an error (not in TLA-lite).
        if ch == "/":
            if st.peek(1) == "=":
                st.advance(2)
                self._emit(TokenKind.NEQ, "/=", start_line, start_col, start_off)
                return
            st.advance()
            self._error(f"Unexpected character '/'", start_line, start_col)
            return

        # ── Multi-char operators (three-char first) ─────────────────
        three = st.peek_str(3)
        if three == "|->":
            st.advance(3)
            self._emit(TokenKind.MAPS_TO, "|->", start_line, start_col, start_off)
            return

        if three == "<=>":
            st.advance(3)
            self._emit(TokenKind.EQUIV, "<=>", start_line, start_col, start_off)
            return

        # ── Two-char operators ──────────────────────────────────────
        two = st.peek_str(2)
        if two == "[]":
            st.advance(2)
            self._emit(TokenKind.BOX, "[]", start_line, start_col, start_off)
            return

        if two == "<>":
            st.advance(2)
            self._emit(TokenKind.DIAMOND, "<>", start_line, start_col, start_off)
            return

        if two in _TWO_CHAR_OPS:
            st.advance(2)
            self._emit(_TWO_CHAR_OPS[two], two, start_line, start_col, start_off)
            return

        # ── Identifiers / keywords ──────────────────────────────────
        if ch.isalpha() or ch == "_":
            self._scan_identifier()
            return

        # ── Single-char operators / delimiters ──────────────────────
        if ch in _SINGLE_CHAR_OPS:
            st.advance()
            self._emit(_SINGLE_CHAR_OPS[ch], ch, start_line, start_col, start_off)
            return

        # ── Unknown character ───────────────────────────────────────
        st.advance()
        self._error(f"Unexpected character {ch!r}", start_line, start_col)

    # ── Sub-scanners ────────────────────────────────────────────────

    def _scan_line_comment(self) -> None:
        """Consume a \\* ... EOL comment."""
        st = self._state
        while not st.at_end and st.peek() != "\n":
            st.advance()

    def _scan_block_comment(self) -> None:
        """Consume a (* ... *) comment, supporting nesting."""
        st = self._state
        st.advance(2)  # skip (*
        depth = 1
        start_line, start_col = st.line, st.column
        while not st.at_end and depth > 0:
            if st.peek() == "(" and st.peek(1) == "*":
                depth += 1
                st.advance(2)
            elif st.peek() == "*" and st.peek(1) == ")":
                depth -= 1
                st.advance(2)
            else:
                st.advance()
        if depth > 0:
            self._error("Unterminated block comment", start_line, start_col)

    def _scan_separator(self) -> None:
        """Consume ---- ... separator lines."""
        st = self._state
        start_line, start_col, start_off = st.line, st.column, st.pos
        while not st.at_end and st.peek() == "-":
            st.advance()
        self._emit(TokenKind.SEPARATOR, "----", start_line, start_col, start_off)

    def _scan_eq_separator(self) -> None:
        """Consume ==== ... end-of-module lines."""
        st = self._state
        while not st.at_end and st.peek() == "=":
            st.advance()

    def _scan_string(self) -> None:
        st = self._state
        start_line, start_col, start_off = st.line, st.column, st.pos
        st.advance()  # skip opening "
        chars: list[str] = []
        while not st.at_end:
            ch = st.peek()
            if ch == '"':
                st.advance()
                self._emit(TokenKind.STRING, "".join(chars), start_line, start_col, start_off)
                return
            if ch == "\\":
                st.advance()
                esc = st.peek()
                if esc == "n":
                    chars.append("\n")
                elif esc == "t":
                    chars.append("\t")
                elif esc == "\\":
                    chars.append("\\")
                elif esc == '"':
                    chars.append('"')
                else:
                    chars.append(esc)
                st.advance()
            else:
                chars.append(ch)
                st.advance()
        self._error("Unterminated string literal", start_line, start_col)

    def _scan_integer(self) -> None:
        st = self._state
        start_line, start_col, start_off = st.line, st.column, st.pos
        m = _INTEGER_RE.match(st.source, st.pos)
        if m:
            text = m.group(0)
            st.advance(len(text))
            if text.startswith(("0x", "0X")):
                val = int(text, 16)
            elif text.startswith(("0o", "0O")):
                val = int(text, 8)
            elif text.startswith(("0b", "0B")):
                val = int(text, 2)
            else:
                val = int(text)
            self._emit(TokenKind.INTEGER, val, start_line, start_col, start_off)
        else:
            st.advance()
            self._error("Invalid integer literal", start_line, start_col)

    def _scan_identifier(self) -> None:
        st = self._state
        start_line, start_col, start_off = st.line, st.column, st.pos
        m = _IDENT_RE.match(st.source, st.pos)
        if not m:
            st.advance()
            self._error("Invalid identifier", start_line, start_col)
            return
        text = m.group(0)
        st.advance(len(text))

        # Bare underscore is a special token, not an identifier
        if text == "_":
            self._emit(TokenKind.UNDERSCORE, "_", start_line, start_col, start_off)
            return

        # Check for WF_ and SF_ which include the underscore
        if text in ("WF", "SF") and not st.at_end and st.peek() == "_":
            st.advance()  # consume _
            kind = TokenKind.WF if text == "WF" else TokenKind.SF
            self._emit(kind, text + "_", start_line, start_col, start_off)
            return

        kind = KEYWORDS.get(text, TokenKind.IDENTIFIER)
        self._emit(kind, text, start_line, start_col, start_off)

    def _scan_backslash_op(self) -> None:
        """Handle operators/quantifiers starting with backslash."""
        st = self._state
        start_line, start_col, start_off = st.line, st.column, st.pos
        st.advance()  # skip '\'

        if st.at_end:
            self._error("Unexpected end of input after '\\'", start_line, start_col)
            return

        ch = st.peek()

        # \/ (disjunction)
        if ch == "/":
            st.advance()
            self._emit(TokenKind.LOR, "\\/", start_line, start_col, start_off)
            return

        # \* is a line comment start — but normally caught earlier;
        # handle defensively
        if ch == "*":
            self._scan_line_comment()
            return

        # Alphabetic backslash operator: \in, \union, \A, etc.
        if ch.isalpha():
            m = _IDENT_RE.match(st.source, st.pos)
            if m:
                word = m.group(0)
                if word in _BACKSLASH_OPS:
                    st.advance(len(word))
                    self._emit(_BACKSLASH_OPS[word], "\\" + word, start_line, start_col, start_off)
                    return
                # Unknown backslash word — emit as error
                st.advance(len(word))
                self._error(f"Unknown operator \\{word}", start_line, start_col)
                return

        # Fallback: line continuation (\<newline>) already handled.
        self._error(f"Unexpected character after backslash: {ch!r}", start_line, start_col)

    # ── Helpers ─────────────────────────────────────────────────────

    def _emit(
        self,
        kind: TokenKind,
        value: object,
        start_line: int,
        start_col: int,
        start_off: int,
    ) -> None:
        loc = self._state.make_loc(start_line, start_col, start_off)
        self._state.tokens.append(Token(kind=kind, value=value, location=loc))

    def _error(self, message: str, line: int, column: int) -> None:
        err = LexerError(message, line, column, self._state.file)
        self._state.errors.append(err)
        self._state.tokens.append(Token(
            kind=TokenKind.ERROR,
            value=message,
            location=SourceLocation(
                file=self._state.file, line=line, column=column,
                end_line=line, end_column=column + 1,
            ),
        ))


# ── Convenience function ────────────────────────────────────────────────────

def tokenize(source: str, file: str = "<string>") -> List[Token]:
    """Tokenize *source* and return the token list.

    Raises :class:`LexerError` if unrecoverable problems are found.
    """
    lexer = Lexer(source, file)
    tokens = lexer.tokenize()
    if lexer.errors:
        raise lexer.errors[0]
    return tokens


def token_stream(source: str, file: str = "<string>") -> Iterator[Token]:
    """Yield tokens lazily (still lexes eagerly under the hood)."""
    yield from tokenize(source, file)
