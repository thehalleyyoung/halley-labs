"""
Specification parser for the MARACE specification language.

Parses human-readable specification strings into ``TemporalFormula``
objects, supports an assume-guarantee contract DSL, and validates
specifications against environment metadata.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from marace.spec.predicates import (
    ConjunctivePredicate,
    CollisionPredicate,
    CustomPredicate,
    DisjunctivePredicate,
    DistancePredicate,
    LinearPredicate,
    NegationPredicate,
    Predicate,
    RegionPredicate,
    RelativeVelocityPredicate,
)
from marace.spec.temporal import (
    Always,
    BoundedResponse,
    Eventually,
    Next,
    TemporalFormula,
    Until,
)


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(enum.Enum):
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    OP_GT = "OP_GT"
    OP_LT = "OP_LT"
    OP_GE = "OP_GE"
    OP_LE = "OP_LE"
    OP_EQ = "OP_EQ"
    OP_AND = "OP_AND"
    OP_OR = "OP_OR"
    OP_NOT = "OP_NOT"
    OP_ARROW = "OP_ARROW"
    ASSIGN = "ASSIGN"
    SEMICOLON = "SEMICOLON"
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_PATTERNS: List[Tuple[str, Optional[TokenType]]] = [
    (r"\s+", None),  # skip whitespace
    (r"->", TokenType.OP_ARROW),
    (r">=", TokenType.OP_GE),
    (r"<=", TokenType.OP_LE),
    (r"==", TokenType.OP_EQ),
    (r"&&", TokenType.OP_AND),
    (r"\|\|", TokenType.OP_OR),
    (r"!", TokenType.OP_NOT),
    (r">", TokenType.OP_GT),
    (r"<", TokenType.OP_LT),
    (r"=", TokenType.ASSIGN),
    (r";", TokenType.SEMICOLON),
    (r"\(", TokenType.LPAREN),
    (r"\)", TokenType.RPAREN),
    (r"\[", TokenType.LBRACKET),
    (r"\]", TokenType.RBRACKET),
    (r",", TokenType.COMMA),
    (r"-?[0-9]+(\.[0-9]+)?", TokenType.NUMBER),
    (r"[a-zA-Z_][a-zA-Z_0-9]*", TokenType.IDENT),
]


class Tokenizer:
    """Tokenizer for the specification language."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._patterns = [(re.compile(p), t) for p, t in _TOKEN_PATTERNS]

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self._pos < len(self._text):
            matched = False
            for pattern, tok_type in self._patterns:
                m = pattern.match(self._text, self._pos)
                if m:
                    if tok_type is not None:
                        tokens.append(Token(tok_type, m.group(), self._pos))
                    self._pos = m.end()
                    matched = True
                    break
            if not matched:
                raise SpecParseError(
                    f"Unexpected character {self._text[self._pos]!r} at position {self._pos}",
                    self._pos,
                )
        tokens.append(Token(TokenType.EOF, "", self._pos))
        return tokens


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

class SpecParseError(Exception):
    """Parse error with position information."""

    def __init__(self, message: str, pos: int = -1, hint: str = "") -> None:
        self.pos = pos
        self.hint = hint
        full = message
        if pos >= 0:
            full += f" (at position {pos})"
        if hint:
            full += f"\n  Hint: {hint}"
        super().__init__(full)


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class SpecParser:
    """Parse specification strings into ``TemporalFormula`` objects.

    Supported grammar (informal)::

        spec     ::= temporal
        temporal ::= 'always' '(' expr [',' kwargs] ')'
                   | 'eventually' '(' expr [',' kwargs] ')'
                   | 'until' '(' expr ',' expr [',' kwargs] ')'
                   | 'next' '(' expr ')'
                   | 'bounded_response' '(' expr ',' expr ',' kwargs ')'
                   | expr
        expr     ::= pred ( '&&' pred | '||' pred )*
        pred     ::= 'distance' '(' ident ',' ident ')' cmp NUMBER
                   | 'collision' '(' ident ',' ident ')'
                   | 'region' '(' ident ',' '[' num ',' num ']' ',' '[' num ',' num ']' ')'
                   | 'relvel' '(' ident ',' ident ')' cmp NUMBER
                   | '!' pred
                   | '(' expr ')'
                   | ident '(' ... ')'

    Example::

        parser = SpecParser()
        f = parser.parse("always(distance(agent_0, agent_1) > 2.0, horizon=100)")
    """

    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0
        self._text: str = ""

    def parse(self, text: str) -> TemporalFormula:
        """Parse a specification string into a ``TemporalFormula``."""
        self._text = text
        self._tokens = Tokenizer(text).tokenize()
        self._pos = 0
        result = self._parse_temporal()
        if self._current().type != TokenType.EOF:
            self._error("Expected end of input")
        return result

    # -- token helpers -------------------------------------------------------

    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _peek(self) -> Token:
        return self._tokens[min(self._pos + 1, len(self._tokens) - 1)]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tok_type: TokenType) -> Token:
        tok = self._current()
        if tok.type != tok_type:
            self._error(f"Expected {tok_type.value}, got {tok.type.value}")
        return self._advance()

    def _match(self, tok_type: TokenType) -> Optional[Token]:
        if self._current().type == tok_type:
            return self._advance()
        return None

    def _error(self, msg: str) -> None:
        tok = self._current()
        raise SpecParseError(msg, tok.pos)

    # -- keyword arguments ---------------------------------------------------

    def _parse_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        while self._current().type == TokenType.COMMA:
            self._advance()  # skip comma
            if self._current().type != TokenType.IDENT:
                break
            key = self._advance().value
            self._expect(TokenType.ASSIGN)
            val_tok = self._expect(TokenType.NUMBER)
            kwargs[key] = float(val_tok.value)
        return kwargs

    # -- temporal ------------------------------------------------------------

    def _parse_temporal(self) -> TemporalFormula:
        tok = self._current()
        if tok.type == TokenType.IDENT:
            if tok.value == "always":
                return self._parse_always()
            if tok.value == "eventually":
                return self._parse_eventually()
            if tok.value == "until":
                return self._parse_until()
            if tok.value == "next":
                return self._parse_next()
            if tok.value == "bounded_response":
                return self._parse_bounded_response()
        # Fall through to predicate expression
        from marace.spec.temporal import PredicateLift
        pred = self._parse_expr()
        return PredicateLift(pred)

    def _parse_always(self) -> Always:
        self._advance()  # skip 'always'
        self._expect(TokenType.LPAREN)
        inner = self._parse_expr()
        kwargs = self._parse_kwargs()
        self._expect(TokenType.RPAREN)
        horizon = int(kwargs.get("horizon", 0)) or None
        return Always(inner, horizon=horizon)

    def _parse_eventually(self) -> Eventually:
        self._advance()
        self._expect(TokenType.LPAREN)
        inner = self._parse_expr()
        kwargs = self._parse_kwargs()
        self._expect(TokenType.RPAREN)
        horizon = int(kwargs.get("horizon", 0)) or None
        return Eventually(inner, horizon=horizon)

    def _parse_until(self) -> Until:
        self._advance()
        self._expect(TokenType.LPAREN)
        p1 = self._parse_expr()
        self._expect(TokenType.COMMA)
        p2 = self._parse_expr()
        kwargs = self._parse_kwargs()
        self._expect(TokenType.RPAREN)
        horizon = int(kwargs.get("horizon", 0)) or None
        return Until(p1, p2, horizon=horizon)

    def _parse_next(self) -> Next:
        self._advance()
        self._expect(TokenType.LPAREN)
        inner = self._parse_expr()
        self._expect(TokenType.RPAREN)
        return Next(inner)

    def _parse_bounded_response(self) -> BoundedResponse:
        self._advance()
        self._expect(TokenType.LPAREN)
        trigger = self._parse_expr()
        self._expect(TokenType.COMMA)
        response = self._parse_expr()
        kwargs = self._parse_kwargs()
        self._expect(TokenType.RPAREN)
        deadline = int(kwargs.get("deadline", 10))
        return BoundedResponse(trigger, response, deadline=deadline)

    # -- expressions ---------------------------------------------------------

    def _parse_expr(self) -> Predicate:
        left = self._parse_unary()
        while True:
            if self._match(TokenType.OP_AND):
                right = self._parse_unary()
                left = ConjunctivePredicate([left, right])
            elif self._match(TokenType.OP_OR):
                right = self._parse_unary()
                left = DisjunctivePredicate([left, right])
            else:
                break
        return left

    def _parse_unary(self) -> Predicate:
        if self._match(TokenType.OP_NOT):
            inner = self._parse_unary()
            return NegationPredicate(inner)
        return self._parse_primary()

    def _parse_primary(self) -> Predicate:
        tok = self._current()

        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return expr

        if tok.type == TokenType.IDENT:
            if tok.value == "distance":
                return self._parse_distance_pred()
            if tok.value == "collision":
                return self._parse_collision_pred()
            if tok.value == "region":
                return self._parse_region_pred()
            if tok.value == "relvel":
                return self._parse_relvel_pred()
            # Generic identifier — treat as a named boolean predicate
            name = self._advance().value
            return CustomPredicate(lambda s, _n=name: True, name=name)

        self._error(f"Unexpected token {tok.value!r}")
        raise AssertionError("unreachable")

    # -- predicate parsers ---------------------------------------------------

    def _parse_distance_pred(self) -> Predicate:
        self._advance()  # skip 'distance'
        self._expect(TokenType.LPAREN)
        agent_i = self._expect(TokenType.IDENT).value
        self._expect(TokenType.COMMA)
        agent_j = self._expect(TokenType.IDENT).value
        self._expect(TokenType.RPAREN)
        cmp = self._parse_cmp_op()
        threshold = float(self._expect(TokenType.NUMBER).value)
        greater = cmp in (">", ">=")
        return DistancePredicate(agent_i, agent_j, threshold, greater=greater)

    def _parse_collision_pred(self) -> Predicate:
        self._advance()
        self._expect(TokenType.LPAREN)
        agent_i = self._expect(TokenType.IDENT).value
        self._expect(TokenType.COMMA)
        agent_j = self._expect(TokenType.IDENT).value
        self._expect(TokenType.RPAREN)
        return CollisionPredicate(agent_i, agent_j)

    def _parse_region_pred(self) -> Predicate:
        self._advance()
        self._expect(TokenType.LPAREN)
        agent = self._expect(TokenType.IDENT).value
        self._expect(TokenType.COMMA)
        self._expect(TokenType.LBRACKET)
        lx = float(self._expect(TokenType.NUMBER).value)
        self._expect(TokenType.COMMA)
        ly = float(self._expect(TokenType.NUMBER).value)
        self._expect(TokenType.RBRACKET)
        self._expect(TokenType.COMMA)
        self._expect(TokenType.LBRACKET)
        hx = float(self._expect(TokenType.NUMBER).value)
        self._expect(TokenType.COMMA)
        hy = float(self._expect(TokenType.NUMBER).value)
        self._expect(TokenType.RBRACKET)
        self._expect(TokenType.RPAREN)
        return RegionPredicate(agent, low=[lx, ly], high=[hx, hy])

    def _parse_relvel_pred(self) -> Predicate:
        self._advance()
        self._expect(TokenType.LPAREN)
        agent_i = self._expect(TokenType.IDENT).value
        self._expect(TokenType.COMMA)
        agent_j = self._expect(TokenType.IDENT).value
        self._expect(TokenType.RPAREN)
        self._parse_cmp_op()
        threshold = float(self._expect(TokenType.NUMBER).value)
        return RelativeVelocityPredicate(agent_i, agent_j, threshold)

    def _parse_cmp_op(self) -> str:
        tok = self._current()
        if tok.type in (TokenType.OP_GT, TokenType.OP_LT,
                        TokenType.OP_GE, TokenType.OP_LE,
                        TokenType.OP_EQ):
            self._advance()
            return tok.value
        self._error("Expected comparison operator")
        raise AssertionError("unreachable")

    # -- convenience ---------------------------------------------------------

    def parse_many(self, text: str) -> List[TemporalFormula]:
        """Parse multiple specifications separated by semicolons."""
        parts = [p.strip() for p in text.split(";") if p.strip()]
        return [self.parse(p) for p in parts]


# ---------------------------------------------------------------------------
# Contract DSL
# ---------------------------------------------------------------------------

@dataclass
class AssumeGuaranteeContract:
    """An assume-guarantee contract for a group of agents.

    Attributes:
        name: Contract name.
        agents: Agents covered by this contract.
        assumptions: Predicates/formulas assumed true.
        guarantees: Predicates/formulas guaranteed under assumptions.
    """
    name: str
    agents: List[str]
    assumptions: List[TemporalFormula]
    guarantees: List[TemporalFormula]

    def check(self, trace: List[Dict[str, np.ndarray]]) -> Dict[str, bool]:
        """Evaluate the contract on a trace."""
        assumptions_hold = all(a.evaluate(trace) for a in self.assumptions)
        results: Dict[str, bool] = {"assumptions_hold": assumptions_hold}
        if assumptions_hold:
            for g in self.guarantees:
                results[f"guarantee_{g.name}"] = g.evaluate(trace)
        else:
            for g in self.guarantees:
                results[f"guarantee_{g.name}"] = True  # vacuously true
        return results


class ContractDSL:
    """Parse assume-guarantee contracts.

    Grammar::

        contract ::= 'contract' ident 'for' ident (',' ident)* ':'
                      'assume' '{' spec (';' spec)* '}'
                      'guarantee' '{' spec (';' spec)* '}'

    Example::

        dsl = ContractDSL()
        contract = dsl.parse('''
            contract safe_merge for agent_0, agent_1:
            assume { always(distance(agent_0, agent_1) > 10.0, horizon=50) }
            guarantee { always(distance(agent_0, agent_1) > 2.0, horizon=100) }
        ''')
    """

    def __init__(self) -> None:
        self._parser = SpecParser()

    def parse(self, text: str) -> AssumeGuaranteeContract:
        text = text.strip()
        # Parse header
        header_match = re.match(
            r"contract\s+(\w+)\s+for\s+([\w\s,]+):", text
        )
        if not header_match:
            raise SpecParseError("Invalid contract header", 0,
                                 "Expected: contract <name> for <agents>:")
        name = header_match.group(1)
        agents = [a.strip() for a in header_match.group(2).split(",")]
        rest = text[header_match.end():].strip()

        # Parse assume block
        assume_match = re.search(r"assume\s*\{([^}]*)\}", rest)
        if not assume_match:
            raise SpecParseError("Missing assume block", hint="Expected: assume { ... }")
        assume_specs = self._parser.parse_many(assume_match.group(1))

        # Parse guarantee block
        guarantee_match = re.search(r"guarantee\s*\{([^}]*)\}", rest)
        if not guarantee_match:
            raise SpecParseError("Missing guarantee block",
                                 hint="Expected: guarantee { ... }")
        guarantee_specs = self._parser.parse_many(guarantee_match.group(1))

        return AssumeGuaranteeContract(
            name=name,
            agents=agents,
            assumptions=assume_specs,
            guarantees=guarantee_specs,
        )


# ---------------------------------------------------------------------------
# Specification validator
# ---------------------------------------------------------------------------

class SpecValidator:
    """Validate specifications against environment metadata.

    Checks that agent identifiers referenced in a specification exist
    in the environment and that predicate dimensions are compatible
    with the observation/action spaces.
    """

    def __init__(
        self,
        agent_ids: List[str],
        state_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        self._agent_ids = set(agent_ids)
        self._state_dims = state_dims or {}

    def validate(self, formula: TemporalFormula) -> List[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: List[str] = []
        self._validate_formula(formula, errors)
        return errors

    def _validate_formula(
        self, formula: TemporalFormula, errors: List[str]
    ) -> None:
        if isinstance(formula, Always):
            self._validate_formula(formula.inner, errors)
        elif isinstance(formula, Eventually):
            self._validate_formula(formula.inner, errors)
        elif isinstance(formula, Until):
            self._validate_formula(formula.f1, errors)
            self._validate_formula(formula.f2, errors)
        elif isinstance(formula, Next):
            self._validate_formula(formula.inner, errors)
        elif isinstance(formula, BoundedResponse):
            self._validate_formula(formula.trigger, errors)
            self._validate_formula(formula.response, errors)
        else:
            # Try to extract predicate
            pred = getattr(formula, "predicate", None)
            if pred is not None:
                self._validate_predicate(pred, errors)

    def _validate_predicate(self, pred: Predicate, errors: List[str]) -> None:
        if isinstance(pred, DistancePredicate):
            self._check_agent(pred.agent_i, errors)
            self._check_agent(pred.agent_j, errors)
        elif isinstance(pred, CollisionPredicate):
            self._check_agent(pred.agent_i, errors)
            self._check_agent(pred.agent_j, errors)
        elif isinstance(pred, RegionPredicate):
            self._check_agent(pred.agent_id, errors)
        elif isinstance(pred, RelativeVelocityPredicate):
            self._check_agent(pred.agent_i, errors)
            self._check_agent(pred.agent_j, errors)
        elif isinstance(pred, (ConjunctivePredicate, DisjunctivePredicate)):
            for p in pred.predicates:
                self._validate_predicate(p, errors)
        elif isinstance(pred, NegationPredicate):
            self._validate_predicate(pred.predicate, errors)
        elif isinstance(pred, LinearPredicate):
            if pred.agent_ids:
                for aid in pred.agent_ids:
                    self._check_agent(aid, errors)

    def _check_agent(self, agent_id: str, errors: List[str]) -> None:
        if agent_id not in self._agent_ids:
            errors.append(
                f"Unknown agent '{agent_id}'. "
                f"Available: {sorted(self._agent_ids)}"
            )
