"""
Formal BNF grammar for the MARACE specification language.

Provides a machine-readable grammar definition, LL(1) analysis utilities,
denotational semantics documentation, and well-formedness checking for
MARACE specifications.

The grammar mirrors the recursive-descent parser in ``parser.py`` but
expresses it declaratively so that it can be used for documentation
generation, grammar validation, and alternative parser back-ends.

BNF Grammar
------------

.. productionlist::
   spec            : formula | contract
   contract        : 'assume' '{' formula_list '}' 'guarantee' '{' formula_list '}'
   formula_list    : formula (';' formula)*
   formula         : temporal_formula | predicate_formula | boolean_formula
   temporal_formula: 'always' '(' formula ')' |
                   : 'eventually' '(' formula ')' |
                   : formula 'until' formula |
                   : 'next' '(' formula ')' |
                   : 'bounded_response' '(' formula ',' formula ',' NUMBER ')'
   boolean_formula : formula 'and' formula |
                   : formula 'or' formula |
                   : 'not' '(' formula ')'
   predicate_formula: distance_pred | collision_pred | region_pred |
                    : relvel_pred | linear_pred
   distance_pred   : 'distance' '(' IDENT ',' IDENT ')' comp_op NUMBER
   collision_pred  : 'collision' '(' IDENT ',' IDENT ')'
   region_pred     : 'region' '(' IDENT ',' '[' number_list ']' ')'
   relvel_pred     : 'relvel' '(' IDENT ',' IDENT ')' comp_op NUMBER
   linear_pred     : '[' number_list ']' comp_op NUMBER
   comp_op         : '<' | '>' | '<=' | '>=' | '='
   number_list     : NUMBER (',' NUMBER)*
"""

from __future__ import annotations

import copy
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from marace.spec.parser import TokenType


# ───────────────────────────────────────────────────────────────────────────
# Grammar rule data structure
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GrammarRule:
    """A single production in the MARACE BNF grammar.

    Attributes:
        name: Non-terminal on the left-hand side (e.g. ``"formula"``).
        production: Right-hand side written as a string of terminals and
            non-terminals separated by spaces.  Terminals are quoted
            (e.g. ``"'always'"``); non-terminals are bare identifiers.
            The empty production is represented by ``"ε"``.
        semantic_action: Free-text description of the semantic action
            executed when this production is reduced.  Uses mathematical
            notation where appropriate.
    """

    name: str
    production: str
    semantic_action: str = ""

    def terminals(self) -> List[str]:
        """Return the terminal symbols appearing in this production."""
        return [
            tok.strip("'\"")
            for tok in self.production.split()
            if tok.startswith("'") or tok.startswith('"')
        ]

    def non_terminals(self) -> List[str]:
        """Return the non-terminal symbols appearing in this production."""
        return [
            tok
            for tok in self.production.split()
            if not tok.startswith("'")
            and not tok.startswith('"')
            and tok not in ("ε", "|", "(", ")", "*", "+", "?")
        ]

    def is_epsilon(self) -> bool:
        """Return ``True`` if this is the empty production."""
        return self.production.strip() == "ε"

    def __str__(self) -> str:
        return f"{self.name} ::= {self.production}"


# ───────────────────────────────────────────────────────────────────────────
# Complete BNF grammar definition
# ───────────────────────────────────────────────────────────────────────────

#: Mapping from non-terminal name to a list of alternative productions.
#: Each alternative is a :class:`GrammarRule`.
MARACE_BNF: Dict[str, List[GrammarRule]] = {
    "spec": [
        GrammarRule(
            "spec",
            "contract",
            "Parse an assume-guarantee contract.",
        ),
        GrammarRule(
            "spec",
            "formula",
            "Parse a standalone temporal/predicate formula.",
        ),
    ],
    "contract": [
        GrammarRule(
            "contract",
            "'assume' '{' formula_list '}' 'guarantee' '{' formula_list '}'",
            (
                "Construct AssumeGuaranteeContract.  Semantics: if all "
                "assumption formulas hold on a trace σ, then every guarantee "
                "formula must hold on σ.  Formally: "
                "(∧ᵢ Aᵢ(σ)) ⟹ (∧ⱼ Gⱼ(σ))."
            ),
        ),
    ],
    "formula_list": [
        GrammarRule(
            "formula_list",
            "formula formula_list_tail",
            "Build a list of formulas separated by semicolons.",
        ),
    ],
    "formula_list_tail": [
        GrammarRule(
            "formula_list_tail",
            "';' formula formula_list_tail",
            "Append another formula to the list.",
        ),
        GrammarRule(
            "formula_list_tail",
            "ε",
            "End of formula list.",
        ),
    ],
    "formula": [
        GrammarRule(
            "formula",
            "temporal_formula",
            "A formula whose outermost connective is a temporal operator.",
        ),
        GrammarRule(
            "formula",
            "boolean_formula",
            "A formula built from Boolean combinators.",
        ),
        GrammarRule(
            "formula",
            "predicate_formula",
            "An atomic predicate over joint state.",
        ),
    ],
    "temporal_formula": [
        GrammarRule(
            "temporal_formula",
            "'always' '(' formula ')'",
            (
                "□ φ  (Always).  Semantics on trace σ at time t:\n"
                "  ⟦□φ⟧(σ, t) = ∀ t′ ≥ t . ⟦φ⟧(σ, t′)\n"
                "Robustness: ρ(□φ, σ, t) = inf_{t′≥t} ρ(φ, σ, t′)."
            ),
        ),
        GrammarRule(
            "temporal_formula",
            "'eventually' '(' formula ')'",
            (
                "◇ φ  (Eventually).  Semantics on trace σ at time t:\n"
                "  ⟦◇φ⟧(σ, t) = ∃ t′ ≥ t . ⟦φ⟧(σ, t′)\n"
                "Robustness: ρ(◇φ, σ, t) = sup_{t′≥t} ρ(φ, σ, t′)."
            ),
        ),
        GrammarRule(
            "temporal_formula",
            "formula 'until' formula",
            (
                "φ₁ U φ₂  (Until).  Semantics on trace σ at time t:\n"
                "  ⟦φ₁ U φ₂⟧(σ, t) = ∃ t′ ≥ t . (⟦φ₂⟧(σ, t′) ∧\n"
                "    ∀ t″ ∈ [t, t′) . ⟦φ₁⟧(σ, t″))\n"
                "Robustness: ρ(φ₁ U φ₂, σ, t) = "
                "sup_{t′≥t} min(ρ(φ₂, σ, t′), inf_{t″∈[t,t′)} ρ(φ₁, σ, t″))."
            ),
        ),
        GrammarRule(
            "temporal_formula",
            "'next' '(' formula ')'",
            (
                "○ φ  (Next).  Semantics on trace σ at time t:\n"
                "  ⟦○φ⟧(σ, t) = ⟦φ⟧(σ, t+1)\n"
                "Robustness: ρ(○φ, σ, t) = ρ(φ, σ, t+1)."
            ),
        ),
        GrammarRule(
            "temporal_formula",
            "'bounded_response' '(' formula ',' formula ',' NUMBER ')'",
            (
                "□(trigger → ◇[0,d] response).  Semantics:\n"
                "  ⟦BR(φ_t, φ_r, d)⟧(σ, t) = ∀ t′ ≥ t .\n"
                "    (⟦φ_t⟧(σ, t′) ⟹ ∃ t″ ∈ [t′, t′+d] . ⟦φ_r⟧(σ, t″))\n"
                "Robustness: ρ = inf_{t′} max(-ρ(φ_t, σ, t′), "
                "sup_{t″∈[t′,t′+d]} ρ(φ_r, σ, t″))."
            ),
        ),
    ],
    "boolean_formula": [
        GrammarRule(
            "boolean_formula",
            "formula 'and' formula",
            (
                "φ₁ ∧ φ₂  (Conjunction).\n"
                "  ⟦φ₁ ∧ φ₂⟧(σ, t) = ⟦φ₁⟧(σ, t) ∧ ⟦φ₂⟧(σ, t)\n"
                "Robustness: ρ(φ₁ ∧ φ₂, σ, t) = min(ρ(φ₁, σ, t), ρ(φ₂, σ, t))."
            ),
        ),
        GrammarRule(
            "boolean_formula",
            "formula 'or' formula",
            (
                "φ₁ ∨ φ₂  (Disjunction).\n"
                "  ⟦φ₁ ∨ φ₂⟧(σ, t) = ⟦φ₁⟧(σ, t) ∨ ⟦φ₂⟧(σ, t)\n"
                "Robustness: ρ(φ₁ ∨ φ₂, σ, t) = max(ρ(φ₁, σ, t), ρ(φ₂, σ, t))."
            ),
        ),
        GrammarRule(
            "boolean_formula",
            "'not' '(' formula ')'",
            (
                "¬φ  (Negation).\n"
                "  ⟦¬φ⟧(σ, t) = ¬⟦φ⟧(σ, t)\n"
                "Robustness: ρ(¬φ, σ, t) = −ρ(φ, σ, t)."
            ),
        ),
    ],
    "predicate_formula": [
        GrammarRule(
            "predicate_formula",
            "distance_pred",
            "Delegate to the distance predicate sub-rule.",
        ),
        GrammarRule(
            "predicate_formula",
            "collision_pred",
            "Delegate to the collision predicate sub-rule.",
        ),
        GrammarRule(
            "predicate_formula",
            "region_pred",
            "Delegate to the region predicate sub-rule.",
        ),
        GrammarRule(
            "predicate_formula",
            "relvel_pred",
            "Delegate to the relative-velocity predicate sub-rule.",
        ),
        GrammarRule(
            "predicate_formula",
            "linear_pred",
            "Delegate to the linear half-space predicate sub-rule.",
        ),
    ],
    "distance_pred": [
        GrammarRule(
            "distance_pred",
            "'distance' '(' IDENT ',' IDENT ')' comp_op NUMBER",
            (
                "Evaluate ‖pos(a₁) − pos(a₂)‖ ⊳ d where ⊳ is the "
                "comparison operator.  Constructs a DistancePredicate.\n"
                "Robustness: ρ = d − ‖pos(a₁) − pos(a₂)‖  (for ≤),\n"
                "            ρ = ‖pos(a₁) − pos(a₂)‖ − d  (for ≥)."
            ),
        ),
    ],
    "collision_pred": [
        GrammarRule(
            "collision_pred",
            "'collision' '(' IDENT ',' IDENT ')'",
            (
                "Axis-aligned bounding-box overlap test for agents a₁, a₂.\n"
                "Constructs a CollisionPredicate.  Evaluates to true iff the "
                "two agents' AABBs overlap."
            ),
        ),
    ],
    "region_pred": [
        GrammarRule(
            "region_pred",
            "'region' '(' IDENT ',' '[' number_list ']' ')'",
            (
                "Check pos(a) ∈ [low, high].  The number_list encodes the "
                "bounding box as alternating low/high pairs.\n"
                "Robustness: ρ = min_d min(pos_d − low_d, high_d − pos_d)."
            ),
        ),
    ],
    "relvel_pred": [
        GrammarRule(
            "relvel_pred",
            "'relvel' '(' IDENT ',' IDENT ')' comp_op NUMBER",
            (
                "Evaluate ‖vel(a₁) − vel(a₂)‖ ⊳ v where ⊳ is the "
                "comparison operator.  Constructs a RelativeVelocityPredicate.\n"
                "Robustness: ρ = v − ‖vel(a₁) − vel(a₂)‖."
            ),
        ),
    ],
    "linear_pred": [
        GrammarRule(
            "linear_pred",
            "'[' number_list ']' comp_op NUMBER",
            (
                "Half-space predicate aᵀx ≤ b.  The number_list gives the "
                "normal vector a; NUMBER gives the offset b.\n"
                "Constructs a LinearPredicate.\n"
                "Robustness: ρ = b − aᵀx."
            ),
        ),
    ],
    "comp_op": [
        GrammarRule("comp_op", "'<'", "Less-than."),
        GrammarRule("comp_op", "'>'", "Greater-than."),
        GrammarRule("comp_op", "'<='", "Less-than-or-equal."),
        GrammarRule("comp_op", "'>='", "Greater-than-or-equal."),
        GrammarRule("comp_op", "'='", "Equality."),
    ],
    "number_list": [
        GrammarRule(
            "number_list",
            "NUMBER number_list_tail",
            "A comma-separated list of numeric literals.",
        ),
    ],
    "number_list_tail": [
        GrammarRule(
            "number_list_tail",
            "',' NUMBER number_list_tail",
            "Append another number to the list.",
        ),
        GrammarRule(
            "number_list_tail",
            "ε",
            "End of number list.",
        ),
    ],
}


#: Set of terminal symbols in the grammar.
TERMINALS: FrozenSet[str] = frozenset(
    {
        "always",
        "eventually",
        "until",
        "next",
        "bounded_response",
        "and",
        "or",
        "not",
        "assume",
        "guarantee",
        "distance",
        "collision",
        "region",
        "relvel",
        "IDENT",
        "NUMBER",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        ",",
        ";",
        "<",
        ">",
        "<=",
        ">=",
        "=",
        "EOF",
    }
)

#: Mapping from terminal symbols to ``TokenType`` enum members used by
#: the recursive-descent parser in ``parser.py``.
TERMINAL_TO_TOKEN: Dict[str, TokenType] = {
    "IDENT": TokenType.IDENT,
    "NUMBER": TokenType.NUMBER,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    ";": TokenType.SEMICOLON,
    "<": TokenType.OP_LT,
    ">": TokenType.OP_GT,
    "<=": TokenType.OP_LE,
    ">=": TokenType.OP_GE,
    "=": TokenType.OP_EQ,
    "and": TokenType.OP_AND,
    "or": TokenType.OP_OR,
    "not": TokenType.OP_NOT,
}


# ───────────────────────────────────────────────────────────────────────────
# Grammar specification class
# ───────────────────────────────────────────────────────────────────────────

class GrammarSpec:
    """Machine-readable representation of the MARACE BNF grammar.

    Stores the complete grammar and exposes methods for LL(1) analysis
    (FIRST / FOLLOW set computation) and membership checking.

    Parameters:
        rules: Mapping from non-terminal name to its production
            alternatives.  Defaults to :data:`MARACE_BNF`.

    Example::

        g = GrammarSpec()
        g.validate_grammar()
        firsts = g.first_sets()
        follows = g.follow_sets()
        g.pretty_print()
    """

    _EPSILON = "ε"
    _EOF = "EOF"

    def __init__(
        self,
        rules: Optional[Dict[str, List[GrammarRule]]] = None,
    ) -> None:
        self._rules: Dict[str, List[GrammarRule]] = (
            copy.deepcopy(rules) if rules is not None else copy.deepcopy(MARACE_BNF)
        )
        self._start: str = "spec"
        self._non_terminals: Set[str] = set(self._rules.keys())
        self._terminals: Set[str] = self._collect_terminals()

    # ── internal helpers ──────────────────────────────────────────────

    def _collect_terminals(self) -> Set[str]:
        """Collect every terminal symbol referenced in the grammar."""
        terms: Set[str] = set()
        for alts in self._rules.values():
            for rule in alts:
                for sym in self._rhs_symbols(rule):
                    if sym not in self._non_terminals and sym != self._EPSILON:
                        terms.add(sym)
        terms.add(self._EOF)
        return terms

    @staticmethod
    def _rhs_symbols(rule: GrammarRule) -> List[str]:
        """Parse the right-hand side of a production into a symbol list.

        Quoted tokens like ``'always'`` are unquoted.  The special
        symbol ``ε`` is preserved.
        """
        if rule.is_epsilon():
            return [GrammarSpec._EPSILON]
        tokens: List[str] = []
        for tok in rule.production.split():
            tok = tok.strip("'\"")
            tokens.append(tok)
        return tokens

    # ── public API ────────────────────────────────────────────────────

    @property
    def start_symbol(self) -> str:
        """The start symbol of the grammar."""
        return self._start

    @property
    def non_terminals(self) -> Set[str]:
        """All non-terminal symbols."""
        return set(self._non_terminals)

    @property
    def terminals(self) -> Set[str]:
        """All terminal symbols (including EOF)."""
        return set(self._terminals)

    @property
    def rules(self) -> Dict[str, List[GrammarRule]]:
        """Mapping from non-terminal to its alternative productions."""
        return dict(self._rules)

    # ── grammar validation ────────────────────────────────────────────

    def validate_grammar(self) -> List[str]:
        """Check the grammar for structural errors.

        Returns a list of error messages.  An empty list means the
        grammar is well-formed.

        Checks performed:

        1. Every non-terminal referenced on a RHS is defined.
        2. No duplicate productions for the same non-terminal.
        3. The start symbol is defined.
        4. Every non-terminal is reachable from the start symbol.
        5. Every non-terminal can derive at least one terminal string.
        """
        errors: List[str] = []

        # 1. Undefined non-terminals
        for nt, alts in self._rules.items():
            for rule in alts:
                for sym in rule.non_terminals():
                    if sym not in self._non_terminals and sym not in self._terminals:
                        errors.append(
                            f"Non-terminal '{sym}' used in production "
                            f"'{rule}' is never defined."
                        )

        # 2. Duplicate productions
        for nt, alts in self._rules.items():
            seen: Set[str] = set()
            for rule in alts:
                if rule.production in seen:
                    errors.append(
                        f"Duplicate production for '{nt}': {rule.production}"
                    )
                seen.add(rule.production)

        # 3. Start symbol
        if self._start not in self._rules:
            errors.append(f"Start symbol '{self._start}' is not defined.")

        # 4. Reachability
        reachable = self._reachable_nonterminals()
        for nt in self._non_terminals:
            if nt not in reachable:
                errors.append(
                    f"Non-terminal '{nt}' is not reachable from "
                    f"start symbol '{self._start}'."
                )

        # 5. Productivity (can derive a terminal string)
        productive = self._productive_nonterminals()
        for nt in self._non_terminals:
            if nt not in productive:
                errors.append(
                    f"Non-terminal '{nt}' cannot derive any terminal string."
                )

        return errors

    def _reachable_nonterminals(self) -> Set[str]:
        """Return set of non-terminals reachable from the start symbol."""
        visited: Set[str] = set()
        stack: List[str] = [self._start]
        while stack:
            nt = stack.pop()
            if nt in visited:
                continue
            visited.add(nt)
            for rule in self._rules.get(nt, []):
                for sym in rule.non_terminals():
                    if sym in self._non_terminals:
                        stack.append(sym)
        return visited

    def _productive_nonterminals(self) -> Set[str]:
        """Return set of non-terminals that can derive a terminal string.

        Uses a fixed-point iteration: a non-terminal is productive if at
        least one of its productions consists entirely of terminals and/or
        already-productive non-terminals.
        """
        productive: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for nt, alts in self._rules.items():
                if nt in productive:
                    continue
                for rule in alts:
                    syms = self._rhs_symbols(rule)
                    if all(
                        s in self._terminals
                        or s in productive
                        or s == self._EPSILON
                        for s in syms
                    ):
                        productive.add(nt)
                        changed = True
                        break
        return productive

    # ── FIRST / FOLLOW sets ──────────────────────────────────────────

    def first_sets(self) -> Dict[str, Set[str]]:
        """Compute FIRST sets for all non-terminals.

        FIRST(A) is the set of terminals that can appear as the first
        symbol of a string derived from A.  If A ⇒* ε then ε ∈ FIRST(A).

        Uses the standard fixed-point algorithm:

        1. For terminal *a*: FIRST(a) = {a}.
        2. For production A → Y₁ Y₂ … Yₖ:
           - Add FIRST(Y₁) \\ {ε} to FIRST(A).
           - If ε ∈ FIRST(Y₁), also add FIRST(Y₂) \\ {ε}, and so on.
           - If ε ∈ FIRST(Yᵢ) for all i, add ε to FIRST(A).
        """
        first: Dict[str, Set[str]] = defaultdict(set)
        # Every terminal's FIRST set is itself.
        for t in self._terminals:
            first[t] = {t}
        first[self._EPSILON] = {self._EPSILON}

        changed = True
        while changed:
            changed = False
            for nt, alts in self._rules.items():
                for rule in alts:
                    syms = self._rhs_symbols(rule)
                    before = len(first[nt])
                    self._add_first_of_string(syms, first, nt)
                    if len(first[nt]) != before:
                        changed = True
        return dict(first)

    def _add_first_of_string(
        self,
        symbols: List[str],
        first: Dict[str, Set[str]],
        target: str,
    ) -> None:
        """Add FIRST(symbols) to first[target]."""
        for sym in symbols:
            sym_first = first.get(sym, set())
            first[target] |= sym_first - {self._EPSILON}
            if self._EPSILON not in sym_first:
                return
        # All symbols can derive ε
        first[target].add(self._EPSILON)

    def follow_sets(self) -> Dict[str, Set[str]]:
        """Compute FOLLOW sets for all non-terminals.

        FOLLOW(A) is the set of terminals that can appear immediately
        after A in some sentential form derived from the start symbol.

        Algorithm:

        1. Place EOF in FOLLOW(S) where S is the start symbol.
        2. For each production A → α B β:
           - Add FIRST(β) \\ {ε} to FOLLOW(B).
           - If ε ∈ FIRST(β) (or β is empty), add FOLLOW(A) to FOLLOW(B).
        """
        first = self.first_sets()
        follow: Dict[str, Set[str]] = defaultdict(set)
        follow[self._start].add(self._EOF)

        changed = True
        while changed:
            changed = False
            for nt, alts in self._rules.items():
                for rule in alts:
                    syms = self._rhs_symbols(rule)
                    for i, sym in enumerate(syms):
                        if sym not in self._non_terminals:
                            continue
                        beta = syms[i + 1:]
                        before = len(follow[sym])
                        # Add FIRST(β) \ {ε}
                        beta_first = self._first_of_string(beta, first)
                        follow[sym] |= beta_first - {self._EPSILON}
                        # If β ⇒* ε, add FOLLOW(A)
                        if self._EPSILON in beta_first or not beta:
                            follow[sym] |= follow[nt]
                        if len(follow[sym]) != before:
                            changed = True

        return dict(follow)

    def _first_of_string(
        self,
        symbols: List[str],
        first: Dict[str, Set[str]],
    ) -> Set[str]:
        """Compute FIRST of a sequence of symbols."""
        result: Set[str] = set()
        for sym in symbols:
            sym_first = first.get(sym, {sym} if sym in self._terminals else set())
            result |= sym_first - {self._EPSILON}
            if self._EPSILON not in sym_first:
                return result
        result.add(self._EPSILON)
        return result

    # ── membership checking ──────────────────────────────────────────

    def accepts(self, token_types: Sequence[TokenType]) -> bool:
        """Check whether a sequence of token types is in the language.

        Uses an Earley-style top-down recogniser with memoisation.
        This is intentionally simple and not designed for production
        throughput — use the recursive-descent ``SpecParser`` for real
        parsing.

        Parameters:
            token_types: Sequence of :class:`TokenType` values ending
                with ``TokenType.EOF``.

        Returns:
            ``True`` if the token sequence can be derived from the
            start symbol.
        """
        token_strs = [self._token_type_to_terminal(t) for t in token_types]
        memo: Dict[Tuple[str, int], Optional[int]] = {}
        result = self._match_nt(self._start, token_strs, 0, memo)
        return result is not None and result == len(token_strs) - 1

    def _token_type_to_terminal(self, tt: TokenType) -> str:
        """Map a ``TokenType`` to its grammar terminal name."""
        _reverse: Dict[TokenType, str] = {v: k for k, v in TERMINAL_TO_TOKEN.items()}
        return _reverse.get(tt, tt.value)

    def _match_nt(
        self,
        nt: str,
        tokens: List[str],
        pos: int,
        memo: Dict[Tuple[str, int], Optional[int]],
    ) -> Optional[int]:
        """Try to match non-terminal *nt* starting at *pos*.

        Returns the position after matching, or ``None`` on failure.
        """
        key = (nt, pos)
        if key in memo:
            return memo[key]
        memo[key] = None  # prevent infinite recursion

        for rule in self._rules.get(nt, []):
            result = self._match_rule(rule, tokens, pos, memo)
            if result is not None:
                memo[key] = result
                return result
        return None

    def _match_rule(
        self,
        rule: GrammarRule,
        tokens: List[str],
        pos: int,
        memo: Dict[Tuple[str, int], Optional[int]],
    ) -> Optional[int]:
        """Try to match a single production rule at *pos*."""
        syms = self._rhs_symbols(rule)
        cur = pos
        for sym in syms:
            if sym == self._EPSILON:
                continue
            if sym in self._non_terminals:
                result = self._match_nt(sym, tokens, cur, memo)
                if result is None:
                    return None
                cur = result
            else:
                # Terminal
                if cur >= len(tokens):
                    return None
                if tokens[cur] != sym:
                    return None
                cur += 1
        return cur

    # ── pretty printing ──────────────────────────────────────────────

    def pretty_print(self) -> str:
        """Return a human-readable multi-line representation of the grammar.

        Each non-terminal is printed with all its alternatives,
        accompanied by their semantic actions.
        """
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("MARACE Specification Language — Formal BNF Grammar")
        lines.append("=" * 72)
        lines.append("")
        for nt in self._ordered_nonterminals():
            alts = self._rules[nt]
            for i, rule in enumerate(alts):
                prefix = f"{nt:<22} ::= " if i == 0 else f"{'':22}  |  "
                lines.append(f"{prefix}{rule.production}")
                if rule.semantic_action:
                    for action_line in rule.semantic_action.splitlines():
                        lines.append(f"{'':26}  ;; {action_line}")
            lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def _ordered_nonterminals(self) -> List[str]:
        """Return non-terminals in a canonical documentation order.

        The start symbol comes first, then non-terminals are ordered
        roughly by their first appearance in the grammar.
        """
        seen: List[str] = []
        stack: List[str] = [self._start]
        visited: Set[str] = set()
        while stack:
            nt = stack.pop(0)
            if nt in visited or nt not in self._rules:
                continue
            visited.add(nt)
            seen.append(nt)
            for rule in self._rules[nt]:
                for sym in rule.non_terminals():
                    if sym in self._non_terminals and sym not in visited:
                        stack.append(sym)
        # Append any remaining non-terminals not reachable from start
        for nt in sorted(self._non_terminals - visited):
            seen.append(nt)
        return seen


# ───────────────────────────────────────────────────────────────────────────
# Formal (denotational) semantics documentation
# ───────────────────────────────────────────────────────────────────────────

class FormalSemantics:
    """Denotational semantics for the MARACE specification language.

    This class is primarily documentary: it maps each syntactic construct
    to its mathematical meaning (trace predicates, robustness functions,
    and contract semantics) and exposes those definitions as structured
    data for documentation generators and analysis tools.

    Trace model
    -----------
    A *trace* is a finite sequence  σ = s₀ s₁ … s_{T−1}  where each
    sₜ ∈ S = (Agent → ℝⁿ)  is a *joint state* mapping agent identifiers
    to real-valued state vectors.

    Boolean semantics
    -----------------
    Each formula φ defines a function ⟦φ⟧ : Σ × ℕ → {⊤, ⊥} where
    Σ is the set of all traces and the second argument is a time index.

    Quantitative (robustness) semantics
    ------------------------------------
    Each formula φ defines a function ρ(φ, ·, ·) : Σ × ℕ → ℝ̄ that
    is *sound*: ρ(φ, σ, t) > 0  ⟹  ⟦φ⟧(σ, t) = ⊤  and
                 ρ(φ, σ, t) < 0  ⟹  ⟦φ⟧(σ, t) = ⊥.

    Contract semantics
    ------------------
    An assume-guarantee contract C = (A, G) is satisfied on a trace σ
    iff (∧ᵢ ⟦Aᵢ⟧(σ, 0)) ⟹ (∧ⱼ ⟦Gⱼ⟧(σ, 0)).  Contract composition
    follows Benveniste et al.'s *contract algebra*.
    """

    # ── temporal operator semantics ───────────────────────────────────

    TEMPORAL_SEMANTICS: Dict[str, Dict[str, str]] = {
        "always": {
            "symbol": "□",
            "boolean": "⟦□φ⟧(σ, t) = ∀ t′ ∈ [t, T) . ⟦φ⟧(σ, t′)",
            "robustness": "ρ(□φ, σ, t) = inf_{t′ ∈ [t,T)} ρ(φ, σ, t′)",
            "bounded": (
                "⟦□[0,H]φ⟧(σ, t) = ∀ t′ ∈ [t, t+H) . ⟦φ⟧(σ, t′)"
            ),
            "monitor": (
                "Online: maintain running min of ρ(φ, σ, t) over a "
                "sliding window of size H."
            ),
        },
        "eventually": {
            "symbol": "◇",
            "boolean": "⟦◇φ⟧(σ, t) = ∃ t′ ∈ [t, T) . ⟦φ⟧(σ, t′)",
            "robustness": "ρ(◇φ, σ, t) = sup_{t′ ∈ [t,T)} ρ(φ, σ, t′)",
            "bounded": (
                "⟦◇[0,H]φ⟧(σ, t) = ∃ t′ ∈ [t, t+H) . ⟦φ⟧(σ, t′)"
            ),
            "monitor": (
                "Online: maintain running max of ρ(φ, σ, t) over a "
                "sliding window of size H."
            ),
        },
        "until": {
            "symbol": "U",
            "boolean": (
                "⟦φ₁ U φ₂⟧(σ, t) = ∃ t′ ≥ t . "
                "(⟦φ₂⟧(σ, t′) ∧ ∀ t″ ∈ [t, t′) . ⟦φ₁⟧(σ, t″))"
            ),
            "robustness": (
                "ρ(φ₁ U φ₂, σ, t) = sup_{t′≥t} "
                "min(ρ(φ₂, σ, t′), inf_{t″∈[t,t′)} ρ(φ₁, σ, t″))"
            ),
            "bounded": (
                "⟦φ₁ U[0,H] φ₂⟧(σ, t) = ∃ t′ ∈ [t, t+H) . "
                "(⟦φ₂⟧(σ, t′) ∧ ∀ t″ ∈ [t, t′) . ⟦φ₁⟧(σ, t″))"
            ),
            "monitor": (
                "Online: maintain a buffer of ρ(φ₁) values and check "
                "ρ(φ₂) at each step; report satisfaction as soon as "
                "φ₂ holds."
            ),
        },
        "next": {
            "symbol": "○",
            "boolean": "⟦○φ⟧(σ, t) = ⟦φ⟧(σ, t+1)",
            "robustness": "ρ(○φ, σ, t) = ρ(φ, σ, t+1)",
            "bounded": "N/A (single-step lookahead).",
            "monitor": (
                "Online: delay evaluation by one step."
            ),
        },
        "bounded_response": {
            "symbol": "□(→◇)",
            "boolean": (
                "⟦BR(φ_t, φ_r, d)⟧(σ, t) = ∀ t′ ≥ t . "
                "(⟦φ_t⟧(σ, t′) ⟹ ∃ t″ ∈ [t′, t′+d] . ⟦φ_r⟧(σ, t″))"
            ),
            "robustness": (
                "ρ = inf_{t′} max(−ρ(φ_t, σ, t′), "
                "sup_{t″∈[t′,t′+d]} ρ(φ_r, σ, t″))"
            ),
            "bounded": "Intrinsically bounded by deadline d.",
            "monitor": (
                "Online: track pending triggers in a queue; for each "
                "trigger, check response within deadline steps."
            ),
        },
    }

    # ── predicate semantics ───────────────────────────────────────────

    PREDICATE_SEMANTICS: Dict[str, Dict[str, str]] = {
        "distance": {
            "domain": "Two agent identifiers a₁, a₂; threshold d ∈ ℝ₊",
            "boolean": "⟦distance(a₁,a₂) ⊳ d⟧(s) ⟺ ‖pos(a₁)−pos(a₂)‖ ⊳ d",
            "robustness_le": "ρ = d − ‖pos(a₁) − pos(a₂)‖",
            "robustness_ge": "ρ = ‖pos(a₁) − pos(a₂)‖ − d",
        },
        "collision": {
            "domain": "Two agent identifiers a₁, a₂",
            "boolean": (
                "⟦collision(a₁,a₂)⟧(s) ⟺ AABB(a₁) ∩ AABB(a₂) ≠ ∅"
            ),
            "robustness": "ρ = −min(margin_x, margin_y)",
        },
        "region": {
            "domain": "Agent identifier a; bounds [low, high] ⊂ ℝⁿ",
            "boolean": "⟦region(a, [l,h])⟧(s) ⟺ pos(a) ∈ [l, h]",
            "robustness": "ρ = min_d min(pos_d(a) − l_d, h_d − pos_d(a))",
        },
        "relvel": {
            "domain": "Two agent identifiers a₁, a₂; threshold v ∈ ℝ₊",
            "boolean": "⟦relvel(a₁,a₂) ⊳ v⟧(s) ⟺ ‖vel(a₁)−vel(a₂)‖ ⊳ v",
            "robustness": "ρ = v − ‖vel(a₁) − vel(a₂)‖",
        },
        "linear": {
            "domain": "Normal vector a ∈ ℝⁿ; offset b ∈ ℝ",
            "boolean": "⟦[a] ≤ b⟧(s) ⟺ aᵀx(s) ≤ b",
            "robustness": "ρ = b − aᵀx(s)",
        },
    }

    # ── contract semantics ────────────────────────────────────────────

    CONTRACT_SEMANTICS: Dict[str, str] = {
        "satisfaction": (
            "A contract C = (A, G) is satisfied on trace σ iff:\n"
            "  (∧ᵢ ⟦Aᵢ⟧(σ, 0)) ⟹ (∧ⱼ ⟦Gⱼ⟧(σ, 0)).\n"
            "If assumptions do not hold, guarantees are vacuously true."
        ),
        "robustness": (
            "Quantitative contract robustness:\n"
            "  ρ(C, σ) = max(−min_i ρ(Aᵢ, σ), min_j ρ(Gⱼ, σ)).\n"
            "This follows the implication encoding: "
            "ρ(A → G) = max(−ρ(A), ρ(G))."
        ),
        "composition": (
            "Parallel composition of contracts C₁ = (A₁, G₁) and "
            "C₂ = (A₂, G₂):\n"
            "  C₁ ⊗ C₂ = (A₁ ∧ A₂, G₁ ∧ G₂).\n"
            "Compositional reasoning requires checking that the "
            "assumptions of one component are discharged by the "
            "guarantees of the other (circular dependency check)."
        ),
        "refinement": (
            "Contract C₁ refines C₂  (C₁ ≤ C₂) iff:\n"
            "  A₂ ⊆ A₁  and  G₁ ⊆ G₂.\n"
            "In other words, C₁ assumes less and guarantees more."
        ),
    }

    # ── query methods ─────────────────────────────────────────────────

    @classmethod
    def get_temporal_semantics(cls, operator: str) -> Dict[str, str]:
        """Return the semantics dictionary for a temporal operator.

        Parameters:
            operator: One of ``"always"``, ``"eventually"``,
                ``"until"``, ``"next"``, ``"bounded_response"``.

        Raises:
            KeyError: If the operator is not recognised.
        """
        return dict(cls.TEMPORAL_SEMANTICS[operator])

    @classmethod
    def get_predicate_semantics(cls, predicate: str) -> Dict[str, str]:
        """Return the semantics dictionary for a predicate kind.

        Parameters:
            predicate: One of ``"distance"``, ``"collision"``,
                ``"region"``, ``"relvel"``, ``"linear"``.

        Raises:
            KeyError: If the predicate kind is not recognised.
        """
        return dict(cls.PREDICATE_SEMANTICS[predicate])

    @classmethod
    def get_contract_semantics(cls) -> Dict[str, str]:
        """Return the full contract semantics documentation."""
        return dict(cls.CONTRACT_SEMANTICS)

    @classmethod
    def summary(cls) -> str:
        """Return a multi-line summary of the denotational semantics."""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("MARACE Denotational Semantics Summary")
        lines.append("=" * 72)
        lines.append("")

        lines.append("Temporal Operators")
        lines.append("-" * 40)
        for op, sem in cls.TEMPORAL_SEMANTICS.items():
            lines.append(f"  {sem['symbol']}  ({op})")
            lines.append(f"    Boolean:    {sem['boolean']}")
            lines.append(f"    Robustness: {sem['robustness']}")
            lines.append("")

        lines.append("Predicates")
        lines.append("-" * 40)
        for pred, sem in cls.PREDICATE_SEMANTICS.items():
            lines.append(f"  {pred}")
            lines.append(f"    Domain:  {sem['domain']}")
            lines.append(f"    Boolean: {sem['boolean']}")
            lines.append("")

        lines.append("Contract Semantics")
        lines.append("-" * 40)
        for key, desc in cls.CONTRACT_SEMANTICS.items():
            lines.append(f"  {key}:")
            for dl in desc.splitlines():
                lines.append(f"    {dl}")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────────────
# Well-formedness checker
# ───────────────────────────────────────────────────────────────────────────

# Expected arity (number of agent-id arguments) for each predicate keyword.
_PREDICATE_ARITY: Dict[str, int] = {
    "distance": 2,
    "collision": 2,
    "region": 1,
    "relvel": 2,
}


class WellFormednessChecker:
    """Validate well-formedness of parsed MARACE specifications.

    Performs the following checks on AST-level objects produced by the
    ``SpecParser`` or ``ContractDSL``:

    1. **Agent existence** — every agent identifier referenced by a
       predicate must be present in the declared agent set.
    2. **Predicate arity** — each predicate must receive the expected
       number of agent-id arguments (e.g. ``distance`` needs exactly 2).
    3. **Temporal nesting validity** — certain nesting patterns are
       flagged as warnings (e.g. ``next(always(...))`` with an unbounded
       inner ``always``).
    4. **Contract acyclicity** — in a system of contracts, the
       assumption/guarantee dependency graph must be a DAG to avoid
       circular reasoning.

    Parameters:
        agent_ids: Known agent identifiers in the environment.
    """

    def __init__(self, agent_ids: Sequence[str]) -> None:
        self._agent_ids: Set[str] = set(agent_ids)

    # ── public interface ──────────────────────────────────────────────

    def check_formula(self, formula: Any) -> List[str]:
        """Check a single formula for well-formedness.

        Parameters:
            formula: A :class:`~marace.spec.temporal.TemporalFormula`
                or :class:`~marace.spec.predicates.Predicate`.

        Returns:
            List of error/warning messages (empty ⟹ well-formed).
        """
        errors: List[str] = []
        self._check_node(formula, errors, nesting_depth=0)
        return errors

    def check_contract(
        self,
        contract: Any,
    ) -> List[str]:
        """Check an :class:`~marace.spec.parser.AssumeGuaranteeContract`.

        Validates all assumptions and guarantees, and checks that the
        agents declared in the contract exist.

        Parameters:
            contract: An ``AssumeGuaranteeContract`` instance.

        Returns:
            List of error/warning messages.
        """
        errors: List[str] = []

        # Check declared agents
        for aid in getattr(contract, "agents", []):
            if aid not in self._agent_ids:
                errors.append(
                    f"Contract '{getattr(contract, 'name', '?')}' declares "
                    f"unknown agent '{aid}'. "
                    f"Known agents: {sorted(self._agent_ids)}."
                )

        # Check each formula
        for assumption in getattr(contract, "assumptions", []):
            errors.extend(self.check_formula(assumption))
        for guarantee in getattr(contract, "guarantees", []):
            errors.extend(self.check_formula(guarantee))

        return errors

    def check_contract_system(
        self,
        contracts: Sequence[Any],
    ) -> List[str]:
        """Check a system of contracts for acyclicity.

        Builds a dependency graph where contract *C₁* depends on *C₂*
        if *C₁*'s assumptions mention agents that appear in *C₂*'s
        guarantees, and checks that this graph is a DAG.

        Parameters:
            contracts: Sequence of ``AssumeGuaranteeContract`` instances.

        Returns:
            List of error messages (empty ⟹ acyclic).
        """
        errors: List[str] = []

        # Validate each contract individually
        for c in contracts:
            errors.extend(self.check_contract(c))

        # Build agent → contract-guarantee mapping
        guarantee_agents: Dict[str, List[str]] = defaultdict(list)
        for c in contracts:
            name = getattr(c, "name", "?")
            for g in getattr(c, "guarantees", []):
                for aid in self._extract_agent_ids(g):
                    guarantee_agents[aid].append(name)

        # Build dependency graph:
        # contract C depends on contract D if C's assumptions reference
        # agents that D guarantees.
        adj: Dict[str, Set[str]] = defaultdict(set)
        contract_names = {getattr(c, "name", f"anon_{i}") for i, c in enumerate(contracts)}
        for c in contracts:
            c_name = getattr(c, "name", "?")
            for a in getattr(c, "assumptions", []):
                for aid in self._extract_agent_ids(a):
                    for dep_name in guarantee_agents.get(aid, []):
                        if dep_name != c_name:
                            adj[c_name].add(dep_name)

        # Cycle detection via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in contract_names}

        def _dfs(node: str, path: List[str]) -> None:
            color[node] = GRAY
            path.append(node)
            for nb in adj.get(node, set()):
                if nb not in color:
                    continue
                if color[nb] == GRAY:
                    cycle_start = path.index(nb)
                    cycle = path[cycle_start:] + [nb]
                    errors.append(
                        f"Circular contract dependency detected: "
                        f"{' → '.join(cycle)}.  Assumptions cannot "
                        f"circularly depend on guarantees."
                    )
                elif color[nb] == WHITE:
                    _dfs(nb, path)
            path.pop()
            color[node] = BLACK

        for n in contract_names:
            if color[n] == WHITE:
                _dfs(n, [])

        return errors

    # ── internal helpers ──────────────────────────────────────────────

    def _check_node(
        self,
        node: Any,
        errors: List[str],
        nesting_depth: int,
    ) -> None:
        """Recursively check a formula/predicate AST node."""
        from marace.spec.temporal import (
            Always,
            BoundedResponse,
            Eventually,
            Next,
            PredicateLift,
            Until,
        )
        from marace.spec.predicates import (
            ConjunctivePredicate,
            CollisionPredicate,
            DisjunctivePredicate,
            DistancePredicate,
            LinearPredicate,
            NegationPredicate,
            Predicate,
            RegionPredicate,
            RelativeVelocityPredicate,
        )

        # ── Temporal operators ────────────────────────────────────
        if isinstance(node, Always):
            self._check_nesting(node, "always", nesting_depth, errors)
            self._check_node(node.inner, errors, nesting_depth + 1)
            return

        if isinstance(node, Eventually):
            self._check_nesting(node, "eventually", nesting_depth, errors)
            self._check_node(node.inner, errors, nesting_depth + 1)
            return

        if isinstance(node, Until):
            self._check_nesting(node, "until", nesting_depth, errors)
            self._check_node(node.f1, errors, nesting_depth + 1)
            self._check_node(node.f2, errors, nesting_depth + 1)
            return

        if isinstance(node, Next):
            self._check_nesting(node, "next", nesting_depth, errors)
            self._check_node(node.inner, errors, nesting_depth + 1)
            return

        if isinstance(node, BoundedResponse):
            self._check_nesting(node, "bounded_response", nesting_depth, errors)
            if node.deadline <= 0:
                errors.append(
                    f"BoundedResponse deadline must be positive, "
                    f"got {node.deadline}."
                )
            self._check_node(node.trigger, errors, nesting_depth + 1)
            self._check_node(node.response, errors, nesting_depth + 1)
            return

        if isinstance(node, PredicateLift):
            self._check_node(node.predicate, errors, nesting_depth)
            return

        # ── Predicates ────────────────────────────────────────────
        if isinstance(node, DistancePredicate):
            self._check_agent(node.agent_i, "distance", errors)
            self._check_agent(node.agent_j, "distance", errors)
            self._check_arity("distance", 2, [node.agent_i, node.agent_j], errors)
            return

        if isinstance(node, CollisionPredicate):
            self._check_agent(node.agent_i, "collision", errors)
            self._check_agent(node.agent_j, "collision", errors)
            self._check_arity("collision", 2, [node.agent_i, node.agent_j], errors)
            return

        if isinstance(node, RegionPredicate):
            self._check_agent(node.agent_id, "region", errors)
            self._check_arity("region", 1, [node.agent_id], errors)
            return

        if isinstance(node, RelativeVelocityPredicate):
            self._check_agent(node.agent_i, "relvel", errors)
            self._check_agent(node.agent_j, "relvel", errors)
            self._check_arity("relvel", 2, [node.agent_i, node.agent_j], errors)
            return

        if isinstance(node, LinearPredicate):
            for aid in (node.agent_ids or []):
                self._check_agent(aid, "linear", errors)
            return

        if isinstance(node, ConjunctivePredicate):
            for p in node.predicates:
                self._check_node(p, errors, nesting_depth)
            return

        if isinstance(node, DisjunctivePredicate):
            for p in node.predicates:
                self._check_node(p, errors, nesting_depth)
            return

        if isinstance(node, NegationPredicate):
            self._check_node(node.predicate, errors, nesting_depth)
            return

    def _check_agent(
        self,
        agent_id: str,
        pred_name: str,
        errors: List[str],
    ) -> None:
        """Check that an agent identifier is declared."""
        if agent_id not in self._agent_ids:
            errors.append(
                f"Unknown agent '{agent_id}' in {pred_name} predicate.  "
                f"Known agents: {sorted(self._agent_ids)}."
            )

    def _check_arity(
        self,
        pred_name: str,
        expected: int,
        agents: List[str],
        errors: List[str],
    ) -> None:
        """Check that a predicate received the expected number of agents."""
        if len(agents) != expected:
            errors.append(
                f"Predicate '{pred_name}' expects {expected} agent "
                f"argument(s) but received {len(agents)}: {agents}."
            )

    def _check_nesting(
        self,
        node: Any,
        operator: str,
        depth: int,
        errors: List[str],
    ) -> None:
        """Warn about dubious temporal nesting patterns.

        Current heuristics:

        * ``next`` wrapping an unbounded ``always`` or ``eventually``
          is suspicious because the outer ``next`` shifts by one step
          but the inner operator ranges over the entire remaining trace.
        * Nesting depth > 5 is flagged as a readability warning.
        """
        from marace.spec.temporal import Always, Eventually

        if depth > 5:
            errors.append(
                f"Warning: temporal nesting depth {depth} at "
                f"'{operator}' — consider simplifying."
            )

        if operator == "next":
            inner = getattr(node, "inner", None)
            if isinstance(inner, Always) and inner.horizon is None:
                errors.append(
                    "Warning: next(always(...)) with unbounded inner "
                    "always — did you mean always(next(...))?"
                )
            if isinstance(inner, Eventually) and inner.horizon is None:
                errors.append(
                    "Warning: next(eventually(...)) with unbounded "
                    "inner eventually — did you mean "
                    "eventually(next(...))?"
                )

    def _extract_agent_ids(self, node: Any) -> Set[str]:
        """Recursively collect agent identifiers from a formula/predicate."""
        from marace.spec.temporal import (
            Always,
            BoundedResponse,
            Eventually,
            Next,
            PredicateLift,
            Until,
        )
        from marace.spec.predicates import (
            ConjunctivePredicate,
            CollisionPredicate,
            DisjunctivePredicate,
            DistancePredicate,
            LinearPredicate,
            NegationPredicate,
            RegionPredicate,
            RelativeVelocityPredicate,
        )

        ids: Set[str] = set()

        if isinstance(node, Always):
            ids |= self._extract_agent_ids(node.inner)
        elif isinstance(node, Eventually):
            ids |= self._extract_agent_ids(node.inner)
        elif isinstance(node, Until):
            ids |= self._extract_agent_ids(node.f1)
            ids |= self._extract_agent_ids(node.f2)
        elif isinstance(node, Next):
            ids |= self._extract_agent_ids(node.inner)
        elif isinstance(node, BoundedResponse):
            ids |= self._extract_agent_ids(node.trigger)
            ids |= self._extract_agent_ids(node.response)
        elif isinstance(node, PredicateLift):
            ids |= self._extract_agent_ids(node.predicate)
        elif isinstance(node, DistancePredicate):
            ids.update([node.agent_i, node.agent_j])
        elif isinstance(node, CollisionPredicate):
            ids.update([node.agent_i, node.agent_j])
        elif isinstance(node, RegionPredicate):
            ids.add(node.agent_id)
        elif isinstance(node, RelativeVelocityPredicate):
            ids.update([node.agent_i, node.agent_j])
        elif isinstance(node, LinearPredicate):
            ids.update(node.agent_ids or [])
        elif isinstance(node, ConjunctivePredicate):
            for p in node.predicates:
                ids |= self._extract_agent_ids(p)
        elif isinstance(node, DisjunctivePredicate):
            for p in node.predicates:
                ids |= self._extract_agent_ids(p)
        elif isinstance(node, NegationPredicate):
            ids |= self._extract_agent_ids(node.predicate)

        return ids
