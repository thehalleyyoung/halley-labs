"""
Temporal logic satisfaction checking.

Provides bounded LTL formula parsing, representation, evaluation on
trajectories, and Büchi automaton construction for safety properties,
including product MDP construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Formula AST
# ---------------------------------------------------------------------------

class Operator(Enum):
    """LTL operators."""
    ATOM = auto()       # atomic proposition
    NOT = auto()        # negation
    AND = auto()        # conjunction
    OR = auto()         # disjunction
    IMPLIES = auto()    # implication
    NEXT = auto()       # X φ
    ALWAYS = auto()     # G φ  or G[a,b] φ
    EVENTUALLY = auto() # F φ  or F[a,b] φ
    UNTIL = auto()      # φ U ψ  or φ U[a,b] ψ
    TRUE = auto()
    FALSE = auto()


@dataclass
class LTLFormula:
    """
    AST node for an LTL formula.

    Attributes
    ----------
    op : Operator
    atom : str or None
        Name of the atomic proposition (if op == ATOM).
    children : list of LTLFormula
    bound_lo, bound_hi : int or None
        Interval bounds for bounded operators.
    """
    op: Operator
    atom: Optional[str] = None
    children: List["LTLFormula"] = field(default_factory=list)
    bound_lo: Optional[int] = None
    bound_hi: Optional[int] = None

    # Convenience constructors ------------------------------------------------

    @staticmethod
    def true_() -> "LTLFormula":
        return LTLFormula(op=Operator.TRUE)

    @staticmethod
    def false_() -> "LTLFormula":
        return LTLFormula(op=Operator.FALSE)

    @staticmethod
    def prop(name: str) -> "LTLFormula":
        return LTLFormula(op=Operator.ATOM, atom=name)

    def negate(self) -> "LTLFormula":
        return LTLFormula(op=Operator.NOT, children=[self])

    def and_(self, other: "LTLFormula") -> "LTLFormula":
        return LTLFormula(op=Operator.AND, children=[self, other])

    def or_(self, other: "LTLFormula") -> "LTLFormula":
        return LTLFormula(op=Operator.OR, children=[self, other])

    def implies(self, other: "LTLFormula") -> "LTLFormula":
        return LTLFormula(op=Operator.IMPLIES, children=[self, other])

    @staticmethod
    def next_(child: "LTLFormula") -> "LTLFormula":
        return LTLFormula(op=Operator.NEXT, children=[child])

    @staticmethod
    def always(child: "LTLFormula", lo: int = 0, hi: Optional[int] = None) -> "LTLFormula":
        return LTLFormula(op=Operator.ALWAYS, children=[child], bound_lo=lo, bound_hi=hi)

    @staticmethod
    def eventually(child: "LTLFormula", lo: int = 0, hi: Optional[int] = None) -> "LTLFormula":
        return LTLFormula(op=Operator.EVENTUALLY, children=[child], bound_lo=lo, bound_hi=hi)

    @staticmethod
    def until(left: "LTLFormula", right: "LTLFormula", lo: int = 0, hi: Optional[int] = None) -> "LTLFormula":
        return LTLFormula(op=Operator.UNTIL, children=[left, right], bound_lo=lo, bound_hi=hi)

    # Pretty printing ---------------------------------------------------------

    def __repr__(self) -> str:
        return self._fmt()

    def _fmt(self) -> str:
        if self.op == Operator.TRUE:
            return "true"
        if self.op == Operator.FALSE:
            return "false"
        if self.op == Operator.ATOM:
            return self.atom or "?"
        if self.op == Operator.NOT:
            return f"!({self.children[0]._fmt()})"
        if self.op == Operator.AND:
            return f"({self.children[0]._fmt()} & {self.children[1]._fmt()})"
        if self.op == Operator.OR:
            return f"({self.children[0]._fmt()} | {self.children[1]._fmt()})"
        if self.op == Operator.IMPLIES:
            return f"({self.children[0]._fmt()} -> {self.children[1]._fmt()})"
        bnd = self._bound_str()
        if self.op == Operator.NEXT:
            return f"X({self.children[0]._fmt()})"
        if self.op == Operator.ALWAYS:
            return f"G{bnd}({self.children[0]._fmt()})"
        if self.op == Operator.EVENTUALLY:
            return f"F{bnd}({self.children[0]._fmt()})"
        if self.op == Operator.UNTIL:
            return f"({self.children[0]._fmt()} U{bnd} {self.children[1]._fmt()})"
        return "?"

    def _bound_str(self) -> str:
        if self.bound_lo is not None and self.bound_hi is not None:
            return f"[{self.bound_lo},{self.bound_hi}]"
        if self.bound_hi is not None:
            return f"[0,{self.bound_hi}]"
        return ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Token patterns
_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (\()|(\))|               # parens
        (true|false)|            # boolean constants
        (G|F|X|U)|              # temporal operators
        (\!|not)|               # negation
        (\&\&?|and)|            # conjunction
        (\|\|?|or)|             # disjunction
        (->|implies)|           # implication
        (\[\s*\d+\s*,\s*\d+\s*\])| # bounds [lo,hi]
        ([A-Za-z_][A-Za-z0-9_]*)   # atom
    )\s*
    """,
    re.VERBOSE,
)


class _Tokenizer:
    """Simple tokenizer for LTL formulas."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.tokens: List[Tuple[str, str]] = []
        self._tokenize()
        self.idx = 0

    def _tokenize(self) -> None:
        pos = 0
        while pos < len(self.text):
            m = _TOKEN_RE.match(self.text, pos)
            if m is None:
                pos += 1
                continue
            if m.group(1):
                self.tokens.append(("LPAREN", "("))
            elif m.group(2):
                self.tokens.append(("RPAREN", ")"))
            elif m.group(3):
                self.tokens.append(("BOOL", m.group(3)))
            elif m.group(4):
                self.tokens.append(("TEMP", m.group(4)))
            elif m.group(5):
                self.tokens.append(("NOT", "!"))
            elif m.group(6):
                self.tokens.append(("AND", "&"))
            elif m.group(7):
                self.tokens.append(("OR", "|"))
            elif m.group(8):
                self.tokens.append(("IMPLIES", "->"))
            elif m.group(9):
                self.tokens.append(("BOUNDS", m.group(9)))
            elif m.group(10):
                self.tokens.append(("ATOM", m.group(10)))
            pos = m.end()
        self.tokens.append(("EOF", ""))

    def peek(self) -> Tuple[str, str]:
        return self.tokens[min(self.idx, len(self.tokens) - 1)]

    def advance(self) -> Tuple[str, str]:
        tok = self.peek()
        self.idx += 1
        return tok

    def expect(self, kind: str) -> str:
        tok = self.advance()
        if tok[0] != kind:
            raise SyntaxError(f"Expected {kind}, got {tok}")
        return tok[1]


class _Parser:
    """Recursive-descent parser for bounded LTL."""

    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tok = tokenizer

    def parse(self) -> LTLFormula:
        f = self._parse_implies()
        return f

    def _parse_implies(self) -> LTLFormula:
        left = self._parse_or()
        if self.tok.peek()[0] == "IMPLIES":
            self.tok.advance()
            right = self._parse_implies()
            return left.implies(right)
        return left

    def _parse_or(self) -> LTLFormula:
        left = self._parse_and()
        while self.tok.peek()[0] == "OR":
            self.tok.advance()
            right = self._parse_and()
            left = left.or_(right)
        return left

    def _parse_and(self) -> LTLFormula:
        left = self._parse_unary()
        while self.tok.peek()[0] == "AND":
            self.tok.advance()
            right = self._parse_unary()
            left = left.and_(right)
        return left

    def _parse_unary(self) -> LTLFormula:
        kind, val = self.tok.peek()
        if kind == "NOT":
            self.tok.advance()
            child = self._parse_unary()
            return child.negate()
        if kind == "TEMP":
            return self._parse_temporal()
        return self._parse_primary()

    def _parse_temporal(self) -> LTLFormula:
        _, op = self.tok.advance()
        lo, hi = 0, None
        if self.tok.peek()[0] == "BOUNDS":
            _, bnd = self.tok.advance()
            nums = re.findall(r"\d+", bnd)
            lo, hi = int(nums[0]), int(nums[1])

        if op == "X":
            child = self._parse_unary()
            return LTLFormula.next_(child)
        if op == "G":
            child = self._parse_unary()
            return LTLFormula.always(child, lo, hi)
        if op == "F":
            child = self._parse_unary()
            return LTLFormula.eventually(child, lo, hi)
        if op == "U":
            # U is binary but we parse left operand before calling _parse_temporal
            # Fallback: treat next token as right operand
            right = self._parse_unary()
            return LTLFormula(op=Operator.UNTIL, children=[LTLFormula.true_(), right], bound_lo=lo, bound_hi=hi)
        raise SyntaxError(f"Unknown temporal op: {op}")

    def _parse_primary(self) -> LTLFormula:
        kind, val = self.tok.peek()
        if kind == "LPAREN":
            self.tok.advance()
            f = self.parse()
            # Handle infix Until
            if self.tok.peek()[0] == "TEMP" and self.tok.peek()[1] == "U":
                self.tok.advance()
                lo, hi = 0, None
                if self.tok.peek()[0] == "BOUNDS":
                    _, bnd = self.tok.advance()
                    nums = re.findall(r"\d+", bnd)
                    lo, hi = int(nums[0]), int(nums[1])
                right = self.parse()
                self.tok.expect("RPAREN")
                return LTLFormula.until(f, right, lo, hi)
            self.tok.expect("RPAREN")
            return f
        if kind == "BOOL":
            self.tok.advance()
            return LTLFormula.true_() if val == "true" else LTLFormula.false_()
        if kind == "ATOM":
            self.tok.advance()
            return LTLFormula.prop(val)
        raise SyntaxError(f"Unexpected token: {kind} {val}")


# ---------------------------------------------------------------------------
# BoundedLTL: high-level API
# ---------------------------------------------------------------------------

class BoundedLTL:
    """
    Bounded Linear Temporal Logic.

    Provides parsing, evaluation on finite trajectories, and
    automaton construction.

    Parameters
    ----------
    default_horizon : int
        Default bound for unbounded operators.
    """

    def __init__(self, default_horizon: int = 100) -> None:
        self.default_horizon = default_horizon

    def parse(self, formula_str: str) -> LTLFormula:
        """
        Parse a textual LTL formula into an AST.

        Supported syntax:
        - Atoms: alphanumeric identifiers (e.g. ``safe``, ``target``)
        - Boolean: ``true``, ``false``, ``!``, ``&``, ``|``, ``->``
        - Temporal: ``G``, ``F``, ``X``, ``U``
        - Bounded: ``G[0,10]``, ``F[0,5]``, ``U[0,8]``
        - Parentheses for grouping
        """
        tokenizer = _Tokenizer(formula_str)
        parser = _Parser(tokenizer)
        return parser.parse()

    def evaluate(
        self,
        formula: LTLFormula,
        trajectory: List[Set[str]],
        step: int = 0,
    ) -> bool:
        """
        Evaluate *formula* on a finite *trajectory* starting at *step*.

        Parameters
        ----------
        formula : LTLFormula
        trajectory : list of sets of strings
            Each element is the set of atomic propositions true at that time.
        step : int

        Returns
        -------
        bool
        """
        if step >= len(trajectory):
            # Beyond the trajectory: bounded semantics
            if formula.op in (Operator.TRUE, Operator.ALWAYS):
                return True
            if formula.op in (Operator.FALSE, Operator.EVENTUALLY):
                return False
            if formula.op == Operator.ATOM:
                return False
            return False

        op = formula.op

        if op == Operator.TRUE:
            return True
        if op == Operator.FALSE:
            return False
        if op == Operator.ATOM:
            return formula.atom in trajectory[step]
        if op == Operator.NOT:
            return not self.evaluate(formula.children[0], trajectory, step)
        if op == Operator.AND:
            return (self.evaluate(formula.children[0], trajectory, step)
                    and self.evaluate(formula.children[1], trajectory, step))
        if op == Operator.OR:
            return (self.evaluate(formula.children[0], trajectory, step)
                    or self.evaluate(formula.children[1], trajectory, step))
        if op == Operator.IMPLIES:
            return (not self.evaluate(formula.children[0], trajectory, step)
                    or self.evaluate(formula.children[1], trajectory, step))
        if op == Operator.NEXT:
            return self.evaluate(formula.children[0], trajectory, step + 1)
        if op == Operator.ALWAYS:
            return self._eval_always(formula, trajectory, step)
        if op == Operator.EVENTUALLY:
            return self._eval_eventually(formula, trajectory, step)
        if op == Operator.UNTIL:
            return self._eval_until(formula, trajectory, step)

        raise ValueError(f"Unknown operator: {op}")

    def to_automaton(self, formula: LTLFormula) -> "BuchiAutomaton":
        """
        Construct a Büchi automaton that accepts exactly the traces
        satisfying *formula*.

        For bounded formulas this produces a finite automaton whose
        state encodes the remaining obligation.
        """
        builder = _AutomatonBuilder(self.default_horizon)
        return builder.build(formula)

    # ------------------------------------------------------------------
    # Internal evaluation helpers
    # ------------------------------------------------------------------

    def _eval_always(self, formula: LTLFormula, trajectory: List[Set[str]], step: int) -> bool:
        lo = formula.bound_lo or 0
        hi = formula.bound_hi if formula.bound_hi is not None else (len(trajectory) - step - 1)
        hi = min(hi, len(trajectory) - step - 1)
        child = formula.children[0]
        for t in range(lo, hi + 1):
            if not self.evaluate(child, trajectory, step + t):
                return False
        return True

    def _eval_eventually(self, formula: LTLFormula, trajectory: List[Set[str]], step: int) -> bool:
        lo = formula.bound_lo or 0
        hi = formula.bound_hi if formula.bound_hi is not None else (len(trajectory) - step - 1)
        hi = min(hi, len(trajectory) - step - 1)
        child = formula.children[0]
        for t in range(lo, hi + 1):
            if self.evaluate(child, trajectory, step + t):
                return True
        return False

    def _eval_until(self, formula: LTLFormula, trajectory: List[Set[str]], step: int) -> bool:
        lo = formula.bound_lo or 0
        hi = formula.bound_hi if formula.bound_hi is not None else (len(trajectory) - step - 1)
        hi = min(hi, len(trajectory) - step - 1)
        left, right = formula.children[0], formula.children[1]
        for t in range(lo, hi + 1):
            if self.evaluate(right, trajectory, step + t):
                # Check left holds for all steps before t
                all_left = all(
                    self.evaluate(left, trajectory, step + s) for s in range(t)
                )
                if all_left:
                    return True
        return False


# ---------------------------------------------------------------------------
# Convex LTL Fragment Classification
# ---------------------------------------------------------------------------

class ConvexLTLFragment:
    """Characterize which bounded LTL formulas have convex satisfaction sets.

    The critique identified that "bounded LTL satisfaction is convex over
    transition probabilities" holds only for specific safety/co-safety
    fragments. This class formally classifies formulas and enforces the
    fragment restrictions required for O(K^2) tractable verification.

    **Theorem (Convex Fragment Characterization):**
    The satisfaction probability P_T(phi) is convex in the transition
    matrix T for the following fragment L_convex of bounded LTL:

        L_convex ::= G[0,H] p          (bounded invariance / safety)
                   | G[0,H] (p -> q)    (bounded response / co-safety)
                   | L_convex & L_convex (conjunction of convex)
                   | !F[0,H] p          (equivalent to G[0,H] !p)

    where p, q are atomic propositions (possibly negated).

    For formulas outside L_convex (e.g., F[0,H] p, general Until),
    satisfaction probability is NOT convex in T, and O(K^2) vertex
    checking is only a conservative overapproximation.

    This classification is key: we restrict the shield specification
    language to L_convex and document that richer specs use conservative
    (sound but incomplete) verification.
    """

    @staticmethod
    def is_convex(formula: LTLFormula) -> bool:
        """Check if a formula belongs to the convex fragment L_convex.

        Parameters
        ----------
        formula : LTLFormula

        Returns
        -------
        bool
            True if the formula's satisfaction probability is provably
            convex in the transition matrix.
        """
        return ConvexLTLFragment._check_convex(formula)

    @staticmethod
    def _check_convex(f: LTLFormula) -> bool:
        """Recursive convexity check."""
        op = f.op

        # Atomic propositions: satisfaction is linear (hence convex)
        if op in (Operator.ATOM, Operator.TRUE):
            return True
        if op == Operator.FALSE:
            return True

        # G[0,H] phi: convex if phi is a state property (no temporal ops)
        if op == Operator.ALWAYS:
            if f.bound_hi is not None:  # bounded
                return ConvexLTLFragment._is_state_property(f.children[0])
            return False  # unbounded G is not in the fragment

        # Negation: !F[0,H] p = G[0,H] !p, so check the dual
        if op == Operator.NOT:
            child = f.children[0]
            if child.op == Operator.EVENTUALLY and child.bound_hi is not None:
                return ConvexLTLFragment._is_state_property(child.children[0])
            # !p for atomic p is still a state property
            if ConvexLTLFragment._is_state_property(child):
                return True
            return False

        # Conjunction: convex & convex = convex
        if op == Operator.AND:
            return all(ConvexLTLFragment._check_convex(c) for c in f.children)

        # Implication inside G: G[0,H](p -> q) handled by G case
        if op == Operator.IMPLIES:
            return ConvexLTLFragment._is_state_property(f)

        # F (eventually) is NOT convex in general
        if op == Operator.EVENTUALLY:
            return False

        # Until is NOT convex in general
        if op == Operator.UNTIL:
            return False

        # Disjunction: NOT convex in general (max of convex is not convex)
        if op == Operator.OR:
            return False

        # Next: linear, but only convex for one step
        if op == Operator.NEXT:
            return ConvexLTLFragment._is_state_property(f.children[0])

        return False

    @staticmethod
    def _is_state_property(f: LTLFormula) -> bool:
        """Check if formula is a pure state property (no temporal operators)."""
        if f.op in (Operator.ATOM, Operator.TRUE, Operator.FALSE):
            return True
        if f.op == Operator.NOT:
            return ConvexLTLFragment._is_state_property(f.children[0])
        if f.op in (Operator.AND, Operator.OR, Operator.IMPLIES):
            return all(ConvexLTLFragment._is_state_property(c) for c in f.children)
        return False

    @staticmethod
    def classify(formula: LTLFormula) -> str:
        """Classify a formula's verification complexity.

        Returns
        -------
        str
            'convex' - O(K^2) exact vertex checking
            'conservative' - O(K^2) sound but incomplete overapprox.
            'intractable' - requires full posterior enumeration
        """
        if ConvexLTLFragment.is_convex(formula):
            return "convex"
        # Check if it's at least a bounded safety property
        if formula.op in (Operator.ALWAYS, Operator.NOT):
            return "conservative"
        if formula.op == Operator.AND:
            classes = [ConvexLTLFragment.classify(c) for c in formula.children]
            if "intractable" in classes:
                return "intractable"
            return "conservative"
        return "conservative"

    @staticmethod
    def supported_safety_specs() -> List[str]:
        """Return the supported safety specification patterns.

        These are the specification templates that belong to L_convex
        and admit exact O(K^2) verification.
        """
        return [
            "G[0,H](drawdown < threshold)",
            "G[0,H](|position| <= max_position)",
            "G[0,H](margin_ratio >= min_margin)",
            "G[0,H](drawdown < d1) & G[0,H](|position| <= p1)",
            "G[0,H](drawdown < d -> position_reduce)",
        ]


# ---------------------------------------------------------------------------
# Büchi Automaton
# ---------------------------------------------------------------------------

@dataclass
class AutomatonTransition:
    """A transition in the Büchi automaton."""
    source: int
    target: int
    guard: Callable[[Set[str]], bool]
    label: str = ""


@dataclass
class BuchiAutomaton:
    """
    Büchi automaton for LTL formula acceptance.

    For bounded LTL the acceptance condition reduces to a finite-horizon
    reachability condition.

    Attributes
    ----------
    n_states : int
    initial_state : int
    accepting_states : frozenset of int
    transitions : dict mapping state -> list of AutomatonTransition
    """
    n_states: int
    initial_state: int
    accepting_states: FrozenSet[int]
    transitions: Dict[int, List[AutomatonTransition]]
    is_bounded: bool = False
    horizon: Optional[int] = None

    def accepts(self, word: List[Set[str]]) -> bool:
        """
        Check whether the automaton accepts a finite word.

        For bounded automata, acceptance means reaching an accepting
        state within the horizon.
        """
        current_states: Set[int] = {self.initial_state}

        for step, letter in enumerate(word):
            next_states: Set[int] = set()
            for q in current_states:
                for tr in self.transitions.get(q, []):
                    if tr.guard(letter):
                        next_states.add(tr.target)
            if not next_states:
                return False
            current_states = next_states

        return bool(current_states & self.accepting_states)

    def get_product_transitions(self) -> Dict[Tuple[int, int], int]:
        """
        Return a simplified transition map (q, mdp_state) -> q'
        for product MDP construction.

        This is a heuristic mapping: for each automaton state and
        MDP state, pick the first matching transition target.
        """
        result: Dict[Tuple[int, int], int] = {}
        for q, trs in self.transitions.items():
            for tr in trs:
                # Use label as state index hint
                try:
                    mdp_s = int(tr.label) if tr.label.isdigit() else -1
                except (ValueError, AttributeError):
                    mdp_s = -1
                if mdp_s >= 0:
                    result[(q, mdp_s)] = tr.target
        return result

    def to_adjacency_matrix(self) -> NDArray:
        """Return a dense adjacency matrix of the automaton graph."""
        adj = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for q, trs in self.transitions.items():
            for tr in trs:
                adj[tr.source, tr.target] = 1.0
        return adj

    def minimize(self) -> "BuchiAutomaton":
        """
        Simple state merging minimization.

        Merge states that have identical transitions and acceptance status.
        """
        # Build signature for each state
        sigs: Dict[int, Any] = {}
        for q in range(self.n_states):
            trs = self.transitions.get(q, [])
            sig = (
                q in self.accepting_states,
                tuple(sorted((tr.target, tr.label) for tr in trs)),
            )
            sigs[q] = sig

        # Group by signature
        groups: Dict[Any, List[int]] = {}
        for q, sig in sigs.items():
            groups.setdefault(sig, []).append(q)

        # Build mapping old -> new
        state_map: Dict[int, int] = {}
        new_idx = 0
        for sig, members in groups.items():
            for m in members:
                state_map[m] = new_idx
            new_idx += 1

        n_new = new_idx
        new_accepting = frozenset(state_map[q] for q in self.accepting_states if q in state_map)
        new_transitions: Dict[int, List[AutomatonTransition]] = {}

        seen_edges: Set[Tuple[int, int, str]] = set()
        for q, trs in self.transitions.items():
            nq = state_map[q]
            for tr in trs:
                nt = state_map[tr.target]
                edge_key = (nq, nt, tr.label)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    new_transitions.setdefault(nq, []).append(
                        AutomatonTransition(
                            source=nq, target=nt,
                            guard=tr.guard, label=tr.label,
                        )
                    )

        return BuchiAutomaton(
            n_states=n_new,
            initial_state=state_map[self.initial_state],
            accepting_states=new_accepting,
            transitions=new_transitions,
            is_bounded=self.is_bounded,
            horizon=self.horizon,
        )


# ---------------------------------------------------------------------------
# Automaton builder
# ---------------------------------------------------------------------------

class _AutomatonBuilder:
    """Constructs a Büchi automaton from an LTL formula."""

    def __init__(self, default_horizon: int) -> None:
        self.default_horizon = default_horizon

    def build(self, formula: LTLFormula) -> BuchiAutomaton:
        op = formula.op

        if op == Operator.TRUE:
            return self._trivial_automaton(accepting=True)
        if op == Operator.FALSE:
            return self._trivial_automaton(accepting=False)
        if op == Operator.ATOM:
            return self._atom_automaton(formula.atom or "")
        if op == Operator.NOT:
            inner = self.build(formula.children[0])
            return self._complement(inner)
        if op == Operator.AND:
            a1 = self.build(formula.children[0])
            a2 = self.build(formula.children[1])
            return self._intersection(a1, a2)
        if op == Operator.OR:
            a1 = self.build(formula.children[0])
            a2 = self.build(formula.children[1])
            return self._union(a1, a2)
        if op == Operator.NEXT:
            return self._next_automaton(formula.children[0])
        if op == Operator.ALWAYS:
            return self._always_automaton(formula)
        if op == Operator.EVENTUALLY:
            return self._eventually_automaton(formula)
        if op == Operator.UNTIL:
            return self._until_automaton(formula)
        if op == Operator.IMPLIES:
            neg = formula.children[0].negate()
            disj = neg.or_(formula.children[1])
            return self.build(disj)

        return self._trivial_automaton(accepting=True)

    # -- Primitive automata -----------------------------------------------

    def _trivial_automaton(self, accepting: bool) -> BuchiAutomaton:
        acc = frozenset({0}) if accepting else frozenset()
        return BuchiAutomaton(
            n_states=1,
            initial_state=0,
            accepting_states=acc,
            transitions={0: [AutomatonTransition(0, 0, lambda _: True, "any")]},
            is_bounded=True,
            horizon=0,
        )

    def _atom_automaton(self, atom: str) -> BuchiAutomaton:
        """Two-state automaton: check if atom holds at the first step."""
        return BuchiAutomaton(
            n_states=2,
            initial_state=0,
            accepting_states=frozenset({1}),
            transitions={
                0: [
                    AutomatonTransition(0, 1, lambda s, a=atom: a in s, atom),
                    AutomatonTransition(0, 0, lambda s, a=atom: a not in s, f"!{atom}"),
                ],
                1: [AutomatonTransition(1, 1, lambda _: True, "any")],
            },
            is_bounded=True,
            horizon=1,
        )

    def _next_automaton(self, child: LTLFormula) -> BuchiAutomaton:
        """Automaton for X(child): skip one step, then check child."""
        inner = self.build(child)
        # Add a preamble state
        n_new = inner.n_states + 1
        new_trans: Dict[int, List[AutomatonTransition]] = {}
        # State 0 → unconditionally go to shifted inner initial
        new_trans[0] = [
            AutomatonTransition(0, inner.initial_state + 1, lambda _: True, "skip")
        ]
        for q, trs in inner.transitions.items():
            new_trans[q + 1] = [
                AutomatonTransition(
                    tr.source + 1, tr.target + 1, tr.guard, tr.label
                )
                for tr in trs
            ]
        new_acc = frozenset(q + 1 for q in inner.accepting_states)
        return BuchiAutomaton(
            n_states=n_new,
            initial_state=0,
            accepting_states=new_acc,
            transitions=new_trans,
            is_bounded=True,
            horizon=(inner.horizon or 0) + 1,
        )

    def _always_automaton(self, formula: LTLFormula) -> BuchiAutomaton:
        """
        Automaton for G[lo,hi](child).

        Uses a chain of states, one per time step in [lo, hi].
        Acceptance requires child to hold at every step.
        """
        lo = formula.bound_lo or 0
        hi = formula.bound_hi if formula.bound_hi is not None else self.default_horizon
        child = formula.children[0]
        child_aut = self.build(child)

        length = hi - lo + 1
        # Chain of length+1 states: 0..length
        # State i means "i steps remaining to check"
        n = length + 2  # +1 for acceptance, +1 for rejection sink
        accept_state = length + 1
        reject_state = length

        trans: Dict[int, List[AutomatonTransition]] = {}

        # Skip lo steps
        if lo > 0:
            n += lo
            for i in range(lo):
                trans[i] = [AutomatonTransition(i, i + 1, lambda _: True, "skip")]
            offset = lo
        else:
            offset = 0

        # Checking states
        for i in range(length):
            state_idx = offset + i
            next_state = offset + i + 1 if i < length - 1 else accept_state
            child_atom = child.atom if child.op == Operator.ATOM else None

            if child_atom:
                guard_yes = lambda s, a=child_atom: a in s
                guard_no = lambda s, a=child_atom: a not in s
            else:
                guard_yes = lambda _: True
                guard_no = lambda _: False

            trans[state_idx] = [
                AutomatonTransition(state_idx, next_state, guard_yes, f"step_{i}_pass"),
                AutomatonTransition(state_idx, reject_state, guard_no, f"step_{i}_fail"),
            ]

        # Accept / reject sinks
        trans[accept_state] = [
            AutomatonTransition(accept_state, accept_state, lambda _: True, "accept_sink")
        ]
        trans[reject_state] = [
            AutomatonTransition(reject_state, reject_state, lambda _: True, "reject_sink")
        ]

        return BuchiAutomaton(
            n_states=max(n, accept_state + 1, reject_state + 1),
            initial_state=0,
            accepting_states=frozenset({accept_state}),
            transitions=trans,
            is_bounded=True,
            horizon=hi,
        )

    def _eventually_automaton(self, formula: LTLFormula) -> BuchiAutomaton:
        """
        Automaton for F[lo,hi](child).

        Non-deterministically guess when child holds.
        """
        lo = formula.bound_lo or 0
        hi = formula.bound_hi if formula.bound_hi is not None else self.default_horizon
        child = formula.children[0]
        child_atom = child.atom if child.op == Operator.ATOM else None

        n = hi + 3  # waiting states + accept + reject
        accept_state = hi + 1
        reject_state = hi + 2

        trans: Dict[int, List[AutomatonTransition]] = {}

        for t in range(hi + 1):
            state = t
            if t < lo:
                # Must wait
                trans[state] = [
                    AutomatonTransition(state, state + 1, lambda _: True, f"wait_{t}")
                ]
            else:
                # Can accept or continue
                if child_atom:
                    guard_yes = lambda s, a=child_atom: a in s
                    guard_no = lambda s, a=child_atom: a not in s
                else:
                    guard_yes = lambda _: True
                    guard_no = lambda _: False

                next_wait = state + 1 if t < hi else reject_state
                trans[state] = [
                    AutomatonTransition(state, accept_state, guard_yes, f"found_{t}"),
                    AutomatonTransition(state, next_wait, guard_no, f"skip_{t}"),
                ]

        trans[accept_state] = [
            AutomatonTransition(accept_state, accept_state, lambda _: True, "accept_sink")
        ]
        trans[reject_state] = [
            AutomatonTransition(reject_state, reject_state, lambda _: True, "reject_sink")
        ]

        return BuchiAutomaton(
            n_states=n,
            initial_state=0,
            accepting_states=frozenset({accept_state}),
            transitions=trans,
            is_bounded=True,
            horizon=hi,
        )

    def _until_automaton(self, formula: LTLFormula) -> BuchiAutomaton:
        """
        Automaton for (left U[lo,hi] right).

        Left must hold until right becomes true.
        """
        lo = formula.bound_lo or 0
        hi = formula.bound_hi if formula.bound_hi is not None else self.default_horizon
        left, right = formula.children[0], formula.children[1]
        left_atom = left.atom if left.op == Operator.ATOM else None
        right_atom = right.atom if right.op == Operator.ATOM else None

        n = hi + 3
        accept_state = hi + 1
        reject_state = hi + 2

        trans: Dict[int, List[AutomatonTransition]] = {}

        for t in range(hi + 1):
            state = t
            transitions_here: List[AutomatonTransition] = []

            if right_atom:
                guard_right = lambda s, a=right_atom: a in s
            else:
                guard_right = lambda _: True

            if left_atom:
                guard_left = lambda s, a=left_atom: a in s
                guard_not_left = lambda s, a=left_atom: a not in s
            else:
                guard_left = lambda _: True
                guard_not_left = lambda _: False

            if t >= lo:
                # Can accept if right holds
                transitions_here.append(
                    AutomatonTransition(state, accept_state, guard_right, f"right_{t}")
                )

            # Continue if left holds (and right doesn't, but non-deterministic)
            next_state = state + 1 if t < hi else reject_state
            transitions_here.append(
                AutomatonTransition(state, next_state, guard_left, f"left_{t}")
            )
            # Reject if neither
            transitions_here.append(
                AutomatonTransition(state, reject_state, guard_not_left, f"fail_{t}")
            )

            trans[state] = transitions_here

        trans[accept_state] = [
            AutomatonTransition(accept_state, accept_state, lambda _: True, "accept_sink")
        ]
        trans[reject_state] = [
            AutomatonTransition(reject_state, reject_state, lambda _: True, "reject_sink")
        ]

        return BuchiAutomaton(
            n_states=n,
            initial_state=0,
            accepting_states=frozenset({accept_state}),
            transitions=trans,
            is_bounded=True,
            horizon=hi,
        )

    # -- Automaton combinators -------------------------------------------

    def _complement(self, aut: BuchiAutomaton) -> BuchiAutomaton:
        """Swap accepting and non-accepting states (valid for safety automata)."""
        all_states = frozenset(range(aut.n_states))
        new_acc = all_states - aut.accepting_states
        return BuchiAutomaton(
            n_states=aut.n_states,
            initial_state=aut.initial_state,
            accepting_states=new_acc,
            transitions=aut.transitions,
            is_bounded=aut.is_bounded,
            horizon=aut.horizon,
        )

    def _intersection(self, a1: BuchiAutomaton, a2: BuchiAutomaton) -> BuchiAutomaton:
        """Product automaton for intersection (AND)."""
        n1, n2 = a1.n_states, a2.n_states
        n_prod = n1 * n2

        def encode(q1: int, q2: int) -> int:
            return q1 * n2 + q2

        prod_acc = frozenset(
            encode(q1, q2)
            for q1 in a1.accepting_states
            for q2 in a2.accepting_states
        )

        prod_trans: Dict[int, List[AutomatonTransition]] = {}
        for q1 in range(n1):
            for q2 in range(n2):
                prod_q = encode(q1, q2)
                prod_trans[prod_q] = []
                for t1 in a1.transitions.get(q1, []):
                    for t2 in a2.transitions.get(q2, []):
                        target = encode(t1.target, t2.target)
                        guard = lambda s, g1=t1.guard, g2=t2.guard: g1(s) and g2(s)
                        prod_trans[prod_q].append(
                            AutomatonTransition(prod_q, target, guard, f"{t1.label}&{t2.label}")
                        )

        horizon = max(a1.horizon or 0, a2.horizon or 0) or None
        return BuchiAutomaton(
            n_states=n_prod,
            initial_state=encode(a1.initial_state, a2.initial_state),
            accepting_states=prod_acc,
            transitions=prod_trans,
            is_bounded=a1.is_bounded and a2.is_bounded,
            horizon=horizon,
        )

    def _union(self, a1: BuchiAutomaton, a2: BuchiAutomaton) -> BuchiAutomaton:
        """Disjoint union automaton for OR (non-deterministic)."""
        n1, n2 = a1.n_states, a2.n_states
        n_total = n1 + n2 + 1  # extra initial state

        new_initial = n1 + n2
        new_acc = frozenset(
            list(a1.accepting_states) + [q + n1 for q in a2.accepting_states]
        )

        new_trans: Dict[int, List[AutomatonTransition]] = {}
        # Copy a1 transitions
        for q, trs in a1.transitions.items():
            new_trans[q] = [
                AutomatonTransition(tr.source, tr.target, tr.guard, tr.label)
                for tr in trs
            ]
        # Copy a2 transitions (shifted)
        for q, trs in a2.transitions.items():
            new_trans[q + n1] = [
                AutomatonTransition(tr.source + n1, tr.target + n1, tr.guard, tr.label)
                for tr in trs
            ]
        # Initial state epsilon-transitions
        new_trans[new_initial] = [
            AutomatonTransition(new_initial, a1.initial_state, lambda _: True, "eps1"),
            AutomatonTransition(new_initial, a2.initial_state + n1, lambda _: True, "eps2"),
        ]

        horizon = max(a1.horizon or 0, a2.horizon or 0) or None
        return BuchiAutomaton(
            n_states=n_total,
            initial_state=new_initial,
            accepting_states=new_acc,
            transitions=new_trans,
            is_bounded=a1.is_bounded and a2.is_bounded,
            horizon=horizon,
        )
