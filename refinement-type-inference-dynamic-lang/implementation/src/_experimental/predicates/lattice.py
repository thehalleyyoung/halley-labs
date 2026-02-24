"""
Predicate Abstraction Lattice.

From theory: The sequence of predicate abstractions forms a strictly ascending
chain in a lattice of finite height, bounded by
|Vars| × |Constants| × |AtomicPredicateSchemas|.

Implements the predicate lattice with join, meet, widening, narrowing,
fixed-point computation, and entailment checking.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .templates import (
    AtomicPredicate,
    ComparisonOp,
    ComparisonPredicate,
    Conjunction,
    Disjunction,
    FalsePredicate,
    HasAttrPredicate,
    LinearExpression,
    Negation,
    NullityPredicate,
    PredicateEvaluator,
    PredicateFactory,
    PredicateNormalizer,
    PredicateSimplifier,
    PredicateSort,
    PredicateTemplate,
    TruePredicate,
    TruthinessPredicate,
    TypeTagPredicate,
)


# ---------------------------------------------------------------------------
# Three-valued logic
# ---------------------------------------------------------------------------

class ThreeValuedBool(Enum):
    """Three-valued Boolean: True, False, Unknown."""
    TRUE = auto()
    FALSE = auto()
    UNKNOWN = auto()

    def and_(self, other: "ThreeValuedBool") -> "ThreeValuedBool":
        if self == ThreeValuedBool.FALSE or other == ThreeValuedBool.FALSE:
            return ThreeValuedBool.FALSE
        if self == ThreeValuedBool.TRUE and other == ThreeValuedBool.TRUE:
            return ThreeValuedBool.TRUE
        return ThreeValuedBool.UNKNOWN

    def or_(self, other: "ThreeValuedBool") -> "ThreeValuedBool":
        if self == ThreeValuedBool.TRUE or other == ThreeValuedBool.TRUE:
            return ThreeValuedBool.TRUE
        if self == ThreeValuedBool.FALSE and other == ThreeValuedBool.FALSE:
            return ThreeValuedBool.FALSE
        return ThreeValuedBool.UNKNOWN

    def not_(self) -> "ThreeValuedBool":
        if self == ThreeValuedBool.TRUE:
            return ThreeValuedBool.FALSE
        if self == ThreeValuedBool.FALSE:
            return ThreeValuedBool.TRUE
        return ThreeValuedBool.UNKNOWN

    def is_definite(self) -> bool:
        return self != ThreeValuedBool.UNKNOWN

    def to_bool(self) -> Optional[bool]:
        if self == ThreeValuedBool.TRUE:
            return True
        if self == ThreeValuedBool.FALSE:
            return False
        return None

    @staticmethod
    def from_bool(b: bool) -> "ThreeValuedBool":
        return ThreeValuedBool.TRUE if b else ThreeValuedBool.FALSE

    @staticmethod
    def join(a: "ThreeValuedBool", b: "ThreeValuedBool") -> "ThreeValuedBool":
        if a == b:
            return a
        return ThreeValuedBool.UNKNOWN

    @staticmethod
    def meet(a: "ThreeValuedBool", b: "ThreeValuedBool") -> "ThreeValuedBool":
        if a == b:
            return a
        if a == ThreeValuedBool.UNKNOWN:
            return b
        if b == ThreeValuedBool.UNKNOWN:
            return a
        # TRUE meet FALSE => should not happen in consistent abstractions
        return ThreeValuedBool.UNKNOWN

    def __repr__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# LatticeElement ABC
# ---------------------------------------------------------------------------

class LatticeElement(ABC):
    """Abstract base for lattice elements."""

    @abstractmethod
    def join(self, other: "LatticeElement") -> "LatticeElement":
        """Least upper bound (⊔)."""
        ...

    @abstractmethod
    def meet(self, other: "LatticeElement") -> "LatticeElement":
        """Greatest lower bound (⊓)."""
        ...

    @abstractmethod
    def widen(self, other: "LatticeElement") -> "LatticeElement":
        """Widening operator (▽)."""
        ...

    @abstractmethod
    def narrow(self, other: "LatticeElement") -> "LatticeElement":
        """Narrowing operator (△)."""
        ...

    @abstractmethod
    def is_top(self) -> bool:
        """Check if this is the top element."""
        ...

    @abstractmethod
    def is_bottom(self) -> bool:
        """Check if this is the bottom element."""
        ...

    @abstractmethod
    def leq(self, other: "LatticeElement") -> bool:
        """Partial order (⊑)."""
        ...

    def geq(self, other: "LatticeElement") -> bool:
        return other.leq(self)

    def lt(self, other: "LatticeElement") -> bool:
        return self.leq(other) and not other.leq(self)

    def gt(self, other: "LatticeElement") -> bool:
        return other.lt(self)

    def equivalent(self, other: "LatticeElement") -> bool:
        return self.leq(other) and other.leq(self)


# ---------------------------------------------------------------------------
# PredicateSet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredicateSet:
    """Finite set of predicates with Boolean algebra operations."""
    predicates: FrozenSet[PredicateTemplate]

    @staticmethod
    def empty() -> "PredicateSet":
        return PredicateSet(frozenset())

    @staticmethod
    def singleton(p: PredicateTemplate) -> "PredicateSet":
        return PredicateSet(frozenset({p}))

    @staticmethod
    def from_iterable(preds: Iterable[PredicateTemplate]) -> "PredicateSet":
        return PredicateSet(frozenset(preds))

    def add(self, pred: PredicateTemplate) -> "PredicateSet":
        return PredicateSet(self.predicates | frozenset({pred}))

    def remove(self, pred: PredicateTemplate) -> "PredicateSet":
        return PredicateSet(self.predicates - frozenset({pred}))

    def union(self, other: "PredicateSet") -> "PredicateSet":
        return PredicateSet(self.predicates | other.predicates)

    def intersection(self, other: "PredicateSet") -> "PredicateSet":
        return PredicateSet(self.predicates & other.predicates)

    def difference(self, other: "PredicateSet") -> "PredicateSet":
        return PredicateSet(self.predicates - other.predicates)

    def symmetric_difference(self, other: "PredicateSet") -> "PredicateSet":
        return PredicateSet(self.predicates ^ other.predicates)

    def is_subset(self, other: "PredicateSet") -> bool:
        return self.predicates.issubset(other.predicates)

    def is_superset(self, other: "PredicateSet") -> bool:
        return self.predicates.issuperset(other.predicates)

    def contains(self, pred: PredicateTemplate) -> bool:
        return pred in self.predicates

    def size(self) -> int:
        return len(self.predicates)

    def is_empty(self) -> bool:
        return len(self.predicates) == 0

    def variables(self) -> FrozenSet[str]:
        result: Set[str] = set()
        for p in self.predicates:
            result |= p.variables()
        return frozenset(result)

    def atoms(self) -> Set[AtomicPredicate]:
        result: Set[AtomicPredicate] = set()
        for p in self.predicates:
            result |= p.atoms()
        return result

    def as_conjunction(self) -> PredicateTemplate:
        if not self.predicates:
            return TruePredicate()
        return Conjunction.create(*self.predicates)

    def as_disjunction(self) -> PredicateTemplate:
        if not self.predicates:
            return FalsePredicate()
        return Disjunction.create(*self.predicates)

    def negate_all(self) -> "PredicateSet":
        return PredicateSet(frozenset(p.negate() for p in self.predicates))

    def simplify_all(self) -> "PredicateSet":
        return PredicateSet(frozenset(p.simplify() for p in self.predicates))

    def filter_by_variable(self, var: str) -> "PredicateSet":
        return PredicateSet(frozenset(p for p in self.predicates if var in p.variables()))

    def filter_by_type(self, pred_type: type) -> "PredicateSet":
        return PredicateSet(frozenset(p for p in self.predicates if isinstance(p, pred_type)))

    def sorted_list(self) -> List[PredicateTemplate]:
        return sorted(self.predicates, key=lambda p: p.pretty_print())

    def __iter__(self) -> Iterator[PredicateTemplate]:
        return iter(self.predicates)

    def __len__(self) -> int:
        return len(self.predicates)

    def __contains__(self, item: PredicateTemplate) -> bool:
        return item in self.predicates

    def __repr__(self) -> str:
        if self.is_empty():
            return "PredicateSet(∅)"
        items = ", ".join(p.pretty_print() for p in self.sorted_list())
        return f"PredicateSet({{{items}}})"


# ---------------------------------------------------------------------------
# AbstractPredicateState
# ---------------------------------------------------------------------------

@dataclass
class AbstractPredicateState:
    """Maps each predicate to True/False/Unknown (three-valued logic)."""
    _valuation: Dict[PredicateTemplate, ThreeValuedBool] = field(default_factory=dict)

    @staticmethod
    def top(predicates: Iterable[PredicateTemplate]) -> "AbstractPredicateState":
        return AbstractPredicateState({p: ThreeValuedBool.UNKNOWN for p in predicates})

    @staticmethod
    def bottom(predicates: Iterable[PredicateTemplate]) -> "AbstractPredicateState":
        return AbstractPredicateState({p: ThreeValuedBool.FALSE for p in predicates})

    @staticmethod
    def from_concrete(predicates: Iterable[PredicateTemplate], env: Mapping[str, Any]) -> "AbstractPredicateState":
        evaluator = PredicateEvaluator(env)
        valuation: Dict[PredicateTemplate, ThreeValuedBool] = {}
        for p in predicates:
            try:
                val = evaluator.evaluate(p)
                valuation[p] = ThreeValuedBool.from_bool(val)
            except (KeyError, TypeError):
                valuation[p] = ThreeValuedBool.UNKNOWN
        return AbstractPredicateState(valuation)

    def get(self, pred: PredicateTemplate) -> ThreeValuedBool:
        return self._valuation.get(pred, ThreeValuedBool.UNKNOWN)

    def set(self, pred: PredicateTemplate, value: ThreeValuedBool) -> "AbstractPredicateState":
        new_val = dict(self._valuation)
        new_val[pred] = value
        return AbstractPredicateState(new_val)

    def set_true(self, pred: PredicateTemplate) -> "AbstractPredicateState":
        return self.set(pred, ThreeValuedBool.TRUE)

    def set_false(self, pred: PredicateTemplate) -> "AbstractPredicateState":
        return self.set(pred, ThreeValuedBool.FALSE)

    def set_unknown(self, pred: PredicateTemplate) -> "AbstractPredicateState":
        return self.set(pred, ThreeValuedBool.UNKNOWN)

    def predicates(self) -> FrozenSet[PredicateTemplate]:
        return frozenset(self._valuation.keys())

    def true_predicates(self) -> PredicateSet:
        return PredicateSet(frozenset(
            p for p, v in self._valuation.items() if v == ThreeValuedBool.TRUE
        ))

    def false_predicates(self) -> PredicateSet:
        return PredicateSet(frozenset(
            p for p, v in self._valuation.items() if v == ThreeValuedBool.FALSE
        ))

    def unknown_predicates(self) -> PredicateSet:
        return PredicateSet(frozenset(
            p for p, v in self._valuation.items() if v == ThreeValuedBool.UNKNOWN
        ))

    def definite_predicates(self) -> PredicateSet:
        return PredicateSet(frozenset(
            p for p, v in self._valuation.items() if v.is_definite()
        ))

    def is_consistent(self) -> bool:
        """Check if no predicate and its negation are both true."""
        true_set = self.true_predicates()
        for p in true_set:
            neg = p.negate()
            if self.get(neg) == ThreeValuedBool.TRUE:
                return False
        return True

    def as_formula(self) -> PredicateTemplate:
        """Convert to a formula: conjunction of true predicates and negation of false ones."""
        parts: List[PredicateTemplate] = []
        for p, v in self._valuation.items():
            if v == ThreeValuedBool.TRUE:
                parts.append(p)
            elif v == ThreeValuedBool.FALSE:
                parts.append(p.negate())
        if not parts:
            return TruePredicate()
        return Conjunction.create(*parts)

    def join(self, other: "AbstractPredicateState") -> "AbstractPredicateState":
        all_preds = set(self._valuation.keys()) | set(other._valuation.keys())
        result: Dict[PredicateTemplate, ThreeValuedBool] = {}
        for p in all_preds:
            v1 = self.get(p)
            v2 = other.get(p)
            result[p] = ThreeValuedBool.join(v1, v2)
        return AbstractPredicateState(result)

    def meet(self, other: "AbstractPredicateState") -> "AbstractPredicateState":
        all_preds = set(self._valuation.keys()) | set(other._valuation.keys())
        result: Dict[PredicateTemplate, ThreeValuedBool] = {}
        for p in all_preds:
            v1 = self.get(p)
            v2 = other.get(p)
            result[p] = ThreeValuedBool.meet(v1, v2)
        return AbstractPredicateState(result)

    def leq(self, other: "AbstractPredicateState") -> bool:
        for p, v in self._valuation.items():
            ov = other.get(p)
            if v.is_definite() and ov == ThreeValuedBool.UNKNOWN:
                continue
            if v != ov and ov != ThreeValuedBool.UNKNOWN:
                return False
        return True

    def is_top(self) -> bool:
        return all(v == ThreeValuedBool.UNKNOWN for v in self._valuation.values())

    def is_bottom(self) -> bool:
        return len(self._valuation) > 0 and all(
            v == ThreeValuedBool.FALSE for v in self._valuation.values()
        )

    def refine(self, pred: PredicateTemplate, value: bool) -> "AbstractPredicateState":
        """Refine state by assuming pred has given truth value."""
        new_state = self.set(pred, ThreeValuedBool.from_bool(value))
        negated = pred.negate()
        if negated in self._valuation:
            new_state = new_state.set(negated, ThreeValuedBool.from_bool(not value))
        return new_state

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractPredicateState):
            return NotImplemented
        return self._valuation == other._valuation

    def __hash__(self) -> int:
        return hash(frozenset(self._valuation.items()))

    def __repr__(self) -> str:
        parts: List[str] = []
        for p in sorted(self._valuation.keys(), key=lambda x: x.pretty_print()):
            v = self._valuation[p]
            parts.append(f"{p.pretty_print()}: {v.name}")
        return f"AbstractPredicateState({{{', '.join(parts)}}})"


# ---------------------------------------------------------------------------
# PredicateAbstraction
# ---------------------------------------------------------------------------

class PredicateAbstraction:
    """Maps concrete states to predicate valuations."""

    def __init__(self, predicates: PredicateSet) -> None:
        self._predicates = predicates

    @property
    def predicates(self) -> PredicateSet:
        return self._predicates

    def abstract(self, env: Mapping[str, Any]) -> AbstractPredicateState:
        """Abstract a concrete state to a predicate state."""
        return AbstractPredicateState.from_concrete(self._predicates, env)

    def abstract_many(self, envs: Iterable[Mapping[str, Any]]) -> AbstractPredicateState:
        """Abstract multiple concrete states by joining."""
        states = [self.abstract(env) for env in envs]
        if not states:
            return AbstractPredicateState.top(self._predicates)
        result = states[0]
        for s in states[1:]:
            result = result.join(s)
        return result

    def concretizes(self, state: AbstractPredicateState, env: Mapping[str, Any]) -> bool:
        """Check if a concrete env is in the concretization of the abstract state."""
        concrete = self.abstract(env)
        return state.leq(concrete) or concrete.leq(state)

    def gamma_check(self, state: AbstractPredicateState, env: Mapping[str, Any]) -> bool:
        """Check if env ∈ γ(state): all definite predicates match."""
        evaluator = PredicateEvaluator(env)
        for p in self._predicates:
            v = state.get(p)
            if v.is_definite():
                try:
                    actual = evaluator.evaluate(p)
                    if ThreeValuedBool.from_bool(actual) != v:
                        return False
                except (KeyError, TypeError):
                    pass
        return True

    def add_predicate(self, pred: PredicateTemplate) -> "PredicateAbstraction":
        return PredicateAbstraction(self._predicates.add(pred))

    def remove_predicate(self, pred: PredicateTemplate) -> "PredicateAbstraction":
        return PredicateAbstraction(self._predicates.remove(pred))

    def __repr__(self) -> str:
        return f"PredicateAbstraction({self._predicates})"


# ---------------------------------------------------------------------------
# PredicateLatticeElement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredicateLatticeElement(LatticeElement):
    """An element in the predicate lattice.

    Represents an abstract state as a set of predicates known to hold (true_preds)
    and predicates known not to hold (false_preds). Predicates in neither set are unknown.
    """
    true_preds: FrozenSet[PredicateTemplate]
    false_preds: FrozenSet[PredicateTemplate]
    universe: FrozenSet[PredicateTemplate]

    @staticmethod
    def top(universe: FrozenSet[PredicateTemplate]) -> "PredicateLatticeElement":
        return PredicateLatticeElement(
            true_preds=frozenset(),
            false_preds=frozenset(),
            universe=universe,
        )

    @staticmethod
    def bottom(universe: FrozenSet[PredicateTemplate]) -> "PredicateLatticeElement":
        return PredicateLatticeElement(
            true_preds=frozenset(),
            false_preds=universe,
            universe=universe,
        )

    @staticmethod
    def from_state(state: AbstractPredicateState) -> "PredicateLatticeElement":
        true_p = frozenset(
            p for p, v in state._valuation.items() if v == ThreeValuedBool.TRUE
        )
        false_p = frozenset(
            p for p, v in state._valuation.items() if v == ThreeValuedBool.FALSE
        )
        universe = frozenset(state._valuation.keys())
        return PredicateLatticeElement(true_preds=true_p, false_preds=false_p, universe=universe)

    def to_state(self) -> AbstractPredicateState:
        valuation: Dict[PredicateTemplate, ThreeValuedBool] = {}
        for p in self.universe:
            if p in self.true_preds:
                valuation[p] = ThreeValuedBool.TRUE
            elif p in self.false_preds:
                valuation[p] = ThreeValuedBool.FALSE
            else:
                valuation[p] = ThreeValuedBool.UNKNOWN
        return AbstractPredicateState(valuation)

    def join(self, other: "LatticeElement") -> "PredicateLatticeElement":
        if not isinstance(other, PredicateLatticeElement):
            raise TypeError(f"Cannot join with {type(other)}")
        new_true = self.true_preds & other.true_preds
        new_false = self.false_preds & other.false_preds
        new_universe = self.universe | other.universe
        return PredicateLatticeElement(true_preds=new_true, false_preds=new_false, universe=new_universe)

    def meet(self, other: "LatticeElement") -> "PredicateLatticeElement":
        if not isinstance(other, PredicateLatticeElement):
            raise TypeError(f"Cannot meet with {type(other)}")
        new_true = self.true_preds | other.true_preds
        new_false = self.false_preds | other.false_preds
        conflict = new_true & new_false
        new_true = new_true - conflict
        new_false = new_false - conflict
        new_universe = self.universe | other.universe
        return PredicateLatticeElement(true_preds=new_true, false_preds=new_false, universe=new_universe)

    def widen(self, other: "LatticeElement") -> "PredicateLatticeElement":
        if not isinstance(other, PredicateLatticeElement):
            raise TypeError(f"Cannot widen with {type(other)}")
        stable_true = self.true_preds & other.true_preds
        stable_false = self.false_preds & other.false_preds
        return PredicateLatticeElement(
            true_preds=stable_true,
            false_preds=stable_false,
            universe=self.universe | other.universe,
        )

    def narrow(self, other: "LatticeElement") -> "PredicateLatticeElement":
        if not isinstance(other, PredicateLatticeElement):
            raise TypeError(f"Cannot narrow with {type(other)}")
        new_true = self.true_preds | (other.true_preds - self.false_preds)
        new_false = self.false_preds | (other.false_preds - self.true_preds)
        conflict = new_true & new_false
        new_true = new_true - conflict
        new_false = new_false - conflict
        return PredicateLatticeElement(
            true_preds=new_true,
            false_preds=new_false,
            universe=self.universe | other.universe,
        )

    def is_top(self) -> bool:
        return len(self.true_preds) == 0 and len(self.false_preds) == 0

    def is_bottom(self) -> bool:
        return self.false_preds == self.universe and len(self.universe) > 0

    def leq(self, other: "LatticeElement") -> bool:
        if not isinstance(other, PredicateLatticeElement):
            raise TypeError(f"Cannot compare with {type(other)}")
        return self.true_preds.issubset(other.true_preds) and self.false_preds.issubset(other.false_preds)

    def known_count(self) -> int:
        return len(self.true_preds) + len(self.false_preds)

    def unknown_count(self) -> int:
        return len(self.universe) - self.known_count()

    def information_content(self) -> float:
        if len(self.universe) == 0:
            return 0.0
        return self.known_count() / len(self.universe)

    def as_formula(self) -> PredicateTemplate:
        parts: List[PredicateTemplate] = []
        for p in sorted(self.true_preds, key=lambda x: x.pretty_print()):
            parts.append(p)
        for p in sorted(self.false_preds, key=lambda x: x.pretty_print()):
            parts.append(p.negate())
        if not parts:
            return TruePredicate()
        return Conjunction.create(*parts)

    def __repr__(self) -> str:
        true_s = ", ".join(p.pretty_print() for p in sorted(self.true_preds, key=lambda x: x.pretty_print()))
        false_s = ", ".join(p.pretty_print() for p in sorted(self.false_preds, key=lambda x: x.pretty_print()))
        return f"PLE(true={{{true_s}}}, false={{{false_s}}})"


# ---------------------------------------------------------------------------
# CubeElement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CubeElement:
    """A cube: conjunction of literals (predicate or its negation).

    Each literal is (predicate, polarity) where polarity=True means the
    predicate holds and polarity=False means its negation holds.
    """
    literals: FrozenSet[Tuple[PredicateTemplate, bool]]

    @staticmethod
    def empty() -> "CubeElement":
        return CubeElement(frozenset())

    @staticmethod
    def from_predicate(pred: PredicateTemplate, positive: bool = True) -> "CubeElement":
        return CubeElement(frozenset({(pred, positive)}))

    @staticmethod
    def from_state(state: AbstractPredicateState) -> "CubeElement":
        lits: Set[Tuple[PredicateTemplate, bool]] = set()
        for p, v in state._valuation.items():
            if v == ThreeValuedBool.TRUE:
                lits.add((p, True))
            elif v == ThreeValuedBool.FALSE:
                lits.add((p, False))
        return CubeElement(frozenset(lits))

    def add_literal(self, pred: PredicateTemplate, positive: bool) -> "CubeElement":
        return CubeElement(self.literals | frozenset({(pred, positive)}))

    def remove_literal(self, pred: PredicateTemplate) -> "CubeElement":
        return CubeElement(frozenset((p, pol) for p, pol in self.literals if p != pred))

    def predicates(self) -> FrozenSet[PredicateTemplate]:
        return frozenset(p for p, _ in self.literals)

    def positive_literals(self) -> FrozenSet[PredicateTemplate]:
        return frozenset(p for p, pol in self.literals if pol)

    def negative_literals(self) -> FrozenSet[PredicateTemplate]:
        return frozenset(p for p, pol in self.literals if not pol)

    def is_consistent(self) -> bool:
        preds = {}
        for p, pol in self.literals:
            if p in preds and preds[p] != pol:
                return False
            preds[p] = pol
        return True

    def subsumes(self, other: "CubeElement") -> bool:
        """Check if self subsumes other (self's literals are a subset of other's)."""
        return self.literals.issubset(other.literals)

    def as_formula(self) -> PredicateTemplate:
        parts: List[PredicateTemplate] = []
        for p, pol in sorted(self.literals, key=lambda x: x[0].pretty_print()):
            parts.append(p if pol else p.negate())
        if not parts:
            return TruePredicate()
        return Conjunction.create(*parts)

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        evaluator = PredicateEvaluator(env)
        for p, pol in self.literals:
            val = evaluator.evaluate(p)
            if val != pol:
                return False
        return True

    def intersection(self, other: "CubeElement") -> "CubeElement":
        return CubeElement(self.literals & other.literals)

    def union(self, other: "CubeElement") -> "CubeElement":
        combined = dict(self.literals)
        for p, pol in other.literals:
            if p in {pp for pp, _ in self.literals}:
                existing = next(pl for pp, pl in self.literals if pp == p)
                if existing != pol:
                    continue
            combined[(p, pol)] = None
        return CubeElement(frozenset(combined.keys()))

    def size(self) -> int:
        return len(self.literals)

    def __repr__(self) -> str:
        if not self.literals:
            return "Cube(⊤)"
        parts = []
        for p, pol in sorted(self.literals, key=lambda x: x[0].pretty_print()):
            parts.append(p.pretty_print() if pol else f"¬{p.pretty_print()}")
        return f"Cube({' ∧ '.join(parts)})"


# ---------------------------------------------------------------------------
# CartesianAbstraction
# ---------------------------------------------------------------------------

class CartesianAbstraction:
    """Predicate abstraction via Cartesian product.

    Each predicate is tracked independently; the abstract state is the
    Cartesian product of individual predicate valuations.
    """

    def __init__(self, predicates: PredicateSet) -> None:
        self._predicates = predicates

    @property
    def predicates(self) -> PredicateSet:
        return self._predicates

    def abstract(self, env: Mapping[str, Any]) -> AbstractPredicateState:
        evaluator = PredicateEvaluator(env)
        valuation: Dict[PredicateTemplate, ThreeValuedBool] = {}
        for p in self._predicates:
            try:
                val = evaluator.evaluate(p)
                valuation[p] = ThreeValuedBool.from_bool(val)
            except (KeyError, TypeError):
                valuation[p] = ThreeValuedBool.UNKNOWN
        return AbstractPredicateState(valuation)

    def join_abstract(self, s1: AbstractPredicateState, s2: AbstractPredicateState) -> AbstractPredicateState:
        return s1.join(s2)

    def meet_abstract(self, s1: AbstractPredicateState, s2: AbstractPredicateState) -> AbstractPredicateState:
        return s1.meet(s2)

    def transfer(self, state: AbstractPredicateState, transform: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 sample_envs: Sequence[Mapping[str, Any]]) -> AbstractPredicateState:
        """Approximate transfer function via sampling."""
        results: List[AbstractPredicateState] = []
        for env in sample_envs:
            if state.as_formula().evaluate(env):
                new_env = transform(env)
                results.append(self.abstract(new_env))
        if not results:
            return AbstractPredicateState.top(self._predicates)
        result = results[0]
        for r in results[1:]:
            result = result.join(r)
        return result

    def strongest_postcondition(self, pre: AbstractPredicateState,
                                guard: PredicateTemplate) -> AbstractPredicateState:
        """Refine pre-state by assuming guard is true."""
        result = AbstractPredicateState(dict(pre._valuation))
        for p in self._predicates:
            if p == guard:
                result = result.set_true(p)
            neg = p.negate()
            if neg == guard:
                result = result.set_false(p)
        return result

    def weakest_precondition(self, post: AbstractPredicateState,
                             guard: PredicateTemplate) -> AbstractPredicateState:
        """Compute weakest precondition assuming guard must hold for post."""
        result = AbstractPredicateState(dict(post._valuation))
        if guard in self._predicates.predicates:
            result = result.set_true(guard)
        return result

    def __repr__(self) -> str:
        return f"CartesianAbstraction({self._predicates})"


# ---------------------------------------------------------------------------
# BooleanAbstraction
# ---------------------------------------------------------------------------

class BooleanAbstraction:
    """Full Boolean predicate abstraction.

    Tracks all Boolean combinations of predicates, represented as a set
    of cubes (DNF).
    """

    def __init__(self, predicates: PredicateSet) -> None:
        self._predicates = predicates

    @property
    def predicates(self) -> PredicateSet:
        return self._predicates

    def abstract(self, env: Mapping[str, Any]) -> CubeElement:
        evaluator = PredicateEvaluator(env)
        lits: Set[Tuple[PredicateTemplate, bool]] = set()
        for p in self._predicates:
            try:
                val = evaluator.evaluate(p)
                lits.add((p, val))
            except (KeyError, TypeError):
                pass
        return CubeElement(frozenset(lits))

    def abstract_many(self, envs: Sequence[Mapping[str, Any]]) -> List[CubeElement]:
        return [self.abstract(env) for env in envs]

    def join_cubes(self, cubes: Sequence[CubeElement]) -> CubeElement:
        """Join cubes by keeping only common literals."""
        if not cubes:
            return CubeElement.empty()
        result = cubes[0]
        for c in cubes[1:]:
            result = result.intersection(c)
        return result

    def all_cubes(self) -> List[CubeElement]:
        """Enumerate all possible cubes (exponential in |predicates|)."""
        preds = list(self._predicates)
        cubes: List[CubeElement] = []
        for assignment in itertools.product([True, False], repeat=len(preds)):
            lits = frozenset((p, v) for p, v in zip(preds, assignment))
            cube = CubeElement(lits)
            if cube.is_consistent():
                cubes.append(cube)
        return cubes

    def strongest_cube(self, env: Mapping[str, Any]) -> CubeElement:
        """The strongest cube consistent with the environment."""
        return self.abstract(env)

    def __repr__(self) -> str:
        return f"BooleanAbstraction({self._predicates})"


# ---------------------------------------------------------------------------
# LatticeHeight
# ---------------------------------------------------------------------------

class LatticeHeight:
    """Computes the height bound of the predicate lattice.

    Height ≤ |Vars| × |Constants| × |AtomicPredicateSchemas|.
    """

    def __init__(self, variables: Set[str], constants: Set[Any],
                 schemas: Set[str]) -> None:
        self._variables = variables
        self._constants = constants
        self._schemas = schemas

    @property
    def num_variables(self) -> int:
        return len(self._variables)

    @property
    def num_constants(self) -> int:
        return len(self._constants)

    @property
    def num_schemas(self) -> int:
        return len(self._schemas)

    def height_bound(self) -> int:
        """Upper bound on the lattice height."""
        return self.num_variables * max(self.num_constants, 1) * max(self.num_schemas, 1)

    def max_predicates(self) -> int:
        """Maximum number of atomic predicates."""
        return self.height_bound()

    @staticmethod
    def from_predicates(predicates: PredicateSet) -> "LatticeHeight":
        variables: Set[str] = set()
        constants: Set[Any] = set()
        schemas: Set[str] = set()
        for p in predicates:
            variables |= p.variables()
            if isinstance(p, ComparisonPredicate):
                schemas.add("comparison")
                cv = p.right.constant_value()
                if cv is not None:
                    constants.add(cv)
                cv = p.left.constant_value()
                if cv is not None:
                    constants.add(cv)
            elif isinstance(p, TypeTagPredicate):
                schemas.add("isinstance")
                constants.add(p.type_tag)
            elif isinstance(p, NullityPredicate):
                schemas.add("is_none")
            elif isinstance(p, TruthinessPredicate):
                schemas.add("is_truthy")
            elif isinstance(p, HasAttrPredicate):
                schemas.add("hasattr")
                constants.add(p.attribute)
        return LatticeHeight(variables=variables, constants=constants, schemas=schemas)

    def summary(self) -> str:
        return (
            f"Lattice Height Analysis:\n"
            f"  |Vars| = {self.num_variables}\n"
            f"  |Constants| = {self.num_constants}\n"
            f"  |Schemas| = {self.num_schemas}\n"
            f"  Height bound = {self.height_bound()}"
        )

    def __repr__(self) -> str:
        return f"LatticeHeight(vars={self.num_variables}, consts={self.num_constants}, schemas={self.num_schemas})"


# ---------------------------------------------------------------------------
# PredicateManager
# ---------------------------------------------------------------------------

class PredicateManager:
    """Manages the active predicate set, adding new predicates from contract discovery."""

    def __init__(self, initial: Optional[PredicateSet] = None) -> None:
        self._predicates = initial or PredicateSet.empty()
        self._history: List[PredicateSet] = [self._predicates]
        self._generation: int = 0

    @property
    def predicates(self) -> PredicateSet:
        return self._predicates

    @property
    def generation(self) -> int:
        return self._generation

    def add_predicate(self, pred: PredicateTemplate) -> bool:
        """Add a predicate. Returns True if it was new."""
        if pred in self._predicates:
            return False
        self._predicates = self._predicates.add(pred)
        self._generation += 1
        self._history.append(self._predicates)
        return True

    def add_predicates(self, preds: Iterable[PredicateTemplate]) -> int:
        """Add multiple predicates. Returns count of new predicates added."""
        count = 0
        for p in preds:
            if self.add_predicate(p):
                count += 1
        return count

    def remove_predicate(self, pred: PredicateTemplate) -> bool:
        if pred not in self._predicates:
            return False
        self._predicates = self._predicates.remove(pred)
        self._generation += 1
        self._history.append(self._predicates)
        return True

    def add_from_counterexample(self, cex_env: Mapping[str, Any],
                                candidate_preds: Iterable[PredicateTemplate]) -> List[PredicateTemplate]:
        """Contract discovery: add predicates that distinguish the counterexample."""
        evaluator = PredicateEvaluator(cex_env)
        added: List[PredicateTemplate] = []
        for p in candidate_preds:
            if p in self._predicates:
                continue
            try:
                evaluator.evaluate(p)
                if self.add_predicate(p):
                    added.append(p)
            except (KeyError, TypeError):
                pass
        return added

    def add_from_interpolant(self, interpolant: PredicateTemplate) -> List[PredicateTemplate]:
        """Add atomic predicates from an interpolant."""
        atoms = interpolant.atoms()
        added: List[PredicateTemplate] = []
        for a in atoms:
            if self.add_predicate(a):
                added.append(a)
        return added

    def history(self) -> List[PredicateSet]:
        return list(self._history)

    def height_analysis(self) -> LatticeHeight:
        return LatticeHeight.from_predicates(self._predicates)

    def predicate_count(self) -> int:
        return self._predicates.size()

    def variables(self) -> FrozenSet[str]:
        return self._predicates.variables()

    def partition_by_variable(self) -> "PredicatePartition":
        return PredicatePartition.from_predicates(self._predicates)

    def __repr__(self) -> str:
        return f"PredicateManager(gen={self._generation}, count={self.predicate_count()})"


# ---------------------------------------------------------------------------
# PredicatePartition
# ---------------------------------------------------------------------------

class PredicatePartition:
    """Partitions predicates by variable for efficient lookup."""

    def __init__(self) -> None:
        self._by_variable: Dict[str, PredicateSet] = {}
        self._shared: PredicateSet = PredicateSet.empty()

    @staticmethod
    def from_predicates(preds: PredicateSet) -> "PredicatePartition":
        partition = PredicatePartition()
        for p in preds:
            vs = p.variables()
            if len(vs) == 1:
                var = next(iter(vs))
                if var not in partition._by_variable:
                    partition._by_variable[var] = PredicateSet.empty()
                partition._by_variable[var] = partition._by_variable[var].add(p)
            elif len(vs) > 1:
                partition._shared = partition._shared.add(p)
                for v in vs:
                    if v not in partition._by_variable:
                        partition._by_variable[v] = PredicateSet.empty()
                    partition._by_variable[v] = partition._by_variable[v].add(p)
            else:
                partition._shared = partition._shared.add(p)
        return partition

    def get_for_variable(self, var: str) -> PredicateSet:
        return self._by_variable.get(var, PredicateSet.empty())

    def get_shared(self) -> PredicateSet:
        return self._shared

    def get_all_variables(self) -> Set[str]:
        return set(self._by_variable.keys())

    def get_independent_groups(self) -> List[PredicateSet]:
        """Find groups of predicates that share no variables (independent)."""
        visited: Set[str] = set()
        groups: List[PredicateSet] = []
        for var in self._by_variable:
            if var in visited:
                continue
            group_preds = PredicateSet.empty()
            stack = [var]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                for p in self._by_variable.get(v, PredicateSet.empty()):
                    group_preds = group_preds.add(p)
                    for vv in p.variables():
                        if vv not in visited:
                            stack.append(vv)
            if not group_preds.is_empty():
                groups.append(group_preds)
        return groups

    def __repr__(self) -> str:
        parts = []
        for v in sorted(self._by_variable.keys()):
            parts.append(f"{v}: {self._by_variable[v].size()} preds")
        return f"PredicatePartition({', '.join(parts)})"


# ---------------------------------------------------------------------------
# PredicateLattice
# ---------------------------------------------------------------------------

class PredicateLattice:
    """The lattice of predicate sets with join, meet, and ordering."""

    def __init__(self, universe: PredicateSet) -> None:
        self._universe = universe

    @property
    def universe(self) -> PredicateSet:
        return self._universe

    def top(self) -> PredicateLatticeElement:
        return PredicateLatticeElement.top(self._universe.predicates)

    def bottom(self) -> PredicateLatticeElement:
        return PredicateLatticeElement.bottom(self._universe.predicates)

    def join(self, a: PredicateLatticeElement, b: PredicateLatticeElement) -> PredicateLatticeElement:
        return a.join(b)

    def meet(self, a: PredicateLatticeElement, b: PredicateLatticeElement) -> PredicateLatticeElement:
        return a.meet(b)

    def leq(self, a: PredicateLatticeElement, b: PredicateLatticeElement) -> bool:
        return a.leq(b)

    def height(self) -> int:
        return LatticeHeight.from_predicates(self._universe).height_bound()

    def widen(self, a: PredicateLatticeElement, b: PredicateLatticeElement) -> PredicateLatticeElement:
        return a.widen(b)

    def narrow(self, a: PredicateLatticeElement, b: PredicateLatticeElement) -> PredicateLatticeElement:
        return a.narrow(b)

    def join_all(self, elements: Sequence[PredicateLatticeElement]) -> PredicateLatticeElement:
        if not elements:
            return self.bottom()
        result = elements[0]
        for e in elements[1:]:
            result = result.join(e)
        return result

    def meet_all(self, elements: Sequence[PredicateLatticeElement]) -> PredicateLatticeElement:
        if not elements:
            return self.top()
        result = elements[0]
        for e in elements[1:]:
            result = result.meet(e)
        return result

    def from_concrete(self, env: Mapping[str, Any]) -> PredicateLatticeElement:
        evaluator = PredicateEvaluator(env)
        true_p: Set[PredicateTemplate] = set()
        false_p: Set[PredicateTemplate] = set()
        for p in self._universe:
            try:
                val = evaluator.evaluate(p)
                if val:
                    true_p.add(p)
                else:
                    false_p.add(p)
            except (KeyError, TypeError):
                pass
        return PredicateLatticeElement(
            true_preds=frozenset(true_p),
            false_preds=frozenset(false_p),
            universe=self._universe.predicates,
        )

    def __repr__(self) -> str:
        return f"PredicateLattice(universe_size={self._universe.size()})"


# ---------------------------------------------------------------------------
# FixedPointComputer
# ---------------------------------------------------------------------------

class FixedPointComputer:
    """Computes fixed points over the predicate lattice using Kleene iteration."""

    def __init__(self, lattice: PredicateLattice, max_iterations: int = 1000) -> None:
        self._lattice = lattice
        self._max_iterations = max_iterations

    def kleene_ascending(
        self,
        transfer: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        initial: Optional[PredicateLatticeElement] = None,
    ) -> Tuple[PredicateLatticeElement, int]:
        """Compute least fixed point by ascending Kleene iteration.

        Returns (fixed_point, iteration_count).
        """
        current = initial if initial is not None else self._lattice.bottom()
        for i in range(self._max_iterations):
            next_val = transfer(current)
            joined = self._lattice.join(current, next_val)
            if joined.leq(current) and current.leq(joined):
                return current, i + 1
            current = joined
        return current, self._max_iterations

    def kleene_descending(
        self,
        transfer: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        initial: Optional[PredicateLatticeElement] = None,
    ) -> Tuple[PredicateLatticeElement, int]:
        """Compute greatest fixed point by descending Kleene iteration."""
        current = initial if initial is not None else self._lattice.top()
        for i in range(self._max_iterations):
            next_val = transfer(current)
            met = self._lattice.meet(current, next_val)
            if met.leq(current) and current.leq(met):
                return current, i + 1
            current = met
        return current, self._max_iterations

    def widened_ascending(
        self,
        transfer: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        initial: Optional[PredicateLatticeElement] = None,
        widening_delay: int = 3,
    ) -> Tuple[PredicateLatticeElement, int]:
        """Ascending iteration with widening after delay iterations."""
        current = initial if initial is not None else self._lattice.bottom()
        for i in range(self._max_iterations):
            next_val = transfer(current)
            if i < widening_delay:
                joined = self._lattice.join(current, next_val)
            else:
                joined = self._lattice.widen(current, next_val)
            if joined.leq(current) and current.leq(joined):
                return current, i + 1
            current = joined
        return current, self._max_iterations

    def narrowed_descending(
        self,
        transfer: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        initial: PredicateLatticeElement,
        max_narrow_steps: int = 5,
    ) -> Tuple[PredicateLatticeElement, int]:
        """Descending narrowing from a post-widening result."""
        current = initial
        for i in range(max_narrow_steps):
            next_val = transfer(current)
            narrowed = self._lattice.narrow(current, next_val)
            if narrowed.leq(current) and current.leq(narrowed):
                return current, i + 1
            current = narrowed
        return current, max_narrow_steps

    def widened_then_narrowed(
        self,
        transfer: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        initial: Optional[PredicateLatticeElement] = None,
        widening_delay: int = 3,
        max_narrow_steps: int = 5,
    ) -> Tuple[PredicateLatticeElement, int]:
        """Widen-then-narrow strategy for precise fixed points."""
        widened, w_iters = self.widened_ascending(transfer, initial, widening_delay)
        narrowed, n_iters = self.narrowed_descending(transfer, widened, max_narrow_steps)
        return narrowed, w_iters + n_iters

    def chaotic_iteration(
        self,
        nodes: List[str],
        transfer_map: Dict[str, Callable[[Dict[str, PredicateLatticeElement]], PredicateLatticeElement]],
        initial: Optional[Dict[str, PredicateLatticeElement]] = None,
    ) -> Tuple[Dict[str, PredicateLatticeElement], int]:
        """Chaotic iteration over a CFG with multiple nodes."""
        state: Dict[str, PredicateLatticeElement] = {}
        for n in nodes:
            state[n] = initial.get(n, self._lattice.bottom()) if initial else self._lattice.bottom()
        total_iters = 0
        for _ in range(self._max_iterations):
            changed = False
            total_iters += 1
            for n in nodes:
                if n not in transfer_map:
                    continue
                new_val = transfer_map[n](state)
                joined = self._lattice.join(state[n], new_val)
                if not (joined.leq(state[n]) and state[n].leq(joined)):
                    state[n] = joined
                    changed = True
            if not changed:
                break
        return state, total_iters

    def __repr__(self) -> str:
        return f"FixedPointComputer(max_iters={self._max_iterations})"


# ---------------------------------------------------------------------------
# WideningOperator
# ---------------------------------------------------------------------------

class WideningOperator:
    """Widening for the predicate lattice.

    Strategy: drop predicates that are not stable between iterations
    (i.e., drop least-informative predicates).
    """

    def __init__(self, threshold: int = 0) -> None:
        self._threshold = threshold
        self._stability_count: Dict[PredicateTemplate, int] = {}

    def widen(self, old: PredicateLatticeElement, new: PredicateLatticeElement) -> PredicateLatticeElement:
        """Apply widening: keep only predicates stable in both old and new."""
        stable_true = old.true_preds & new.true_preds
        stable_false = old.false_preds & new.false_preds
        for p in stable_true | stable_false:
            self._stability_count[p] = self._stability_count.get(p, 0) + 1
        unstable_true = old.true_preds - new.true_preds
        unstable_false = old.false_preds - new.false_preds
        for p in unstable_true | unstable_false:
            self._stability_count[p] = 0
        if self._threshold > 0:
            filtered_true: Set[PredicateTemplate] = set()
            filtered_false: Set[PredicateTemplate] = set()
            for p in stable_true:
                if self._stability_count.get(p, 0) >= self._threshold:
                    filtered_true.add(p)
            for p in stable_false:
                if self._stability_count.get(p, 0) >= self._threshold:
                    filtered_false.add(p)
            return PredicateLatticeElement(
                true_preds=frozenset(filtered_true),
                false_preds=frozenset(filtered_false),
                universe=old.universe | new.universe,
            )
        return PredicateLatticeElement(
            true_preds=stable_true,
            false_preds=stable_false,
            universe=old.universe | new.universe,
        )

    def reset(self) -> None:
        self._stability_count.clear()

    def stability_info(self) -> Dict[PredicateTemplate, int]:
        return dict(self._stability_count)

    def __repr__(self) -> str:
        return f"WideningOperator(threshold={self._threshold})"


# ---------------------------------------------------------------------------
# NarrowingOperator
# ---------------------------------------------------------------------------

class NarrowingOperator:
    """Narrowing after widening to recover precision."""

    def narrow(self, old: PredicateLatticeElement, new: PredicateLatticeElement) -> PredicateLatticeElement:
        """Apply narrowing: incorporate definite information from new."""
        extra_true = new.true_preds - old.false_preds
        extra_false = new.false_preds - old.true_preds
        result_true = old.true_preds | extra_true
        result_false = old.false_preds | extra_false
        conflict = result_true & result_false
        result_true = result_true - conflict
        result_false = result_false - conflict
        return PredicateLatticeElement(
            true_preds=result_true,
            false_preds=result_false,
            universe=old.universe | new.universe,
        )

    def __repr__(self) -> str:
        return "NarrowingOperator()"


# ---------------------------------------------------------------------------
# PredicateEntailment
# ---------------------------------------------------------------------------

class PredicateEntailment:
    """Checks entailment between predicate sets.

    Uses syntactic checks and optional SMT queries.
    """

    @staticmethod
    def entails_syntactic(premise: PredicateSet, conclusion: PredicateTemplate) -> Optional[bool]:
        """Syntactic entailment check (returns None if undecidable syntactically)."""
        if isinstance(conclusion, TruePredicate):
            return True
        if isinstance(conclusion, FalsePredicate):
            if premise.is_empty():
                return False
            return None
        if conclusion in premise:
            return True
        if isinstance(conclusion, Disjunction):
            for child in conclusion.children:
                if child in premise:
                    return True
        if isinstance(conclusion, Conjunction):
            if all(
                PredicateEntailment.entails_syntactic(premise, child) is True
                for child in conclusion.children
            ):
                return True
        return None

    @staticmethod
    def entails_state(state: AbstractPredicateState, pred: PredicateTemplate) -> ThreeValuedBool:
        """Check if an abstract state entails a predicate."""
        v = state.get(pred)
        if v == ThreeValuedBool.TRUE:
            return ThreeValuedBool.TRUE
        neg = pred.negate()
        vn = state.get(neg)
        if vn == ThreeValuedBool.TRUE:
            return ThreeValuedBool.FALSE
        if isinstance(pred, Conjunction):
            all_true = True
            for child in pred.children:
                cv = PredicateEntailment.entails_state(state, child)
                if cv == ThreeValuedBool.FALSE:
                    return ThreeValuedBool.FALSE
                if cv != ThreeValuedBool.TRUE:
                    all_true = False
            return ThreeValuedBool.TRUE if all_true else ThreeValuedBool.UNKNOWN
        if isinstance(pred, Disjunction):
            for child in pred.children:
                cv = PredicateEntailment.entails_state(state, child)
                if cv == ThreeValuedBool.TRUE:
                    return ThreeValuedBool.TRUE
            all_false = all(
                PredicateEntailment.entails_state(state, child) == ThreeValuedBool.FALSE
                for child in pred.children
            )
            return ThreeValuedBool.FALSE if all_false else ThreeValuedBool.UNKNOWN
        return ThreeValuedBool.UNKNOWN

    @staticmethod
    def generate_entailment_smt(premise: PredicateSet, conclusion: PredicateTemplate) -> str:
        """Generate SMT-LIB query to check premise ⊨ conclusion.

        Returns unsat iff entailment holds.
        """
        premise_formula = premise.as_conjunction()
        negated_conclusion = conclusion.negate()
        check_formula = Conjunction.create(premise_formula, negated_conclusion)
        all_vars = check_formula.variables()
        lines: List[str] = []
        for v in sorted(all_vars):
            lines.append(f"(declare-const {v} Int)")
        lines.append(f"(assert {check_formula.to_smt()})")
        lines.append("(check-sat)")
        return "\n".join(lines)

    @staticmethod
    def is_tautology(pred: PredicateTemplate) -> Optional[bool]:
        """Check if a predicate is a tautology (syntactically)."""
        simplified = PredicateSimplifier.full_simplify(pred)
        if isinstance(simplified, TruePredicate):
            return True
        if isinstance(simplified, FalsePredicate):
            return False
        return None

    @staticmethod
    def is_contradiction(pred: PredicateTemplate) -> Optional[bool]:
        """Check if a predicate is a contradiction (syntactically)."""
        simplified = PredicateSimplifier.full_simplify(pred)
        if isinstance(simplified, FalsePredicate):
            return True
        if isinstance(simplified, TruePredicate):
            return False
        return None

    @staticmethod
    def are_equivalent(p1: PredicateTemplate, p2: PredicateTemplate) -> Optional[bool]:
        """Syntactic equivalence check."""
        n1 = PredicateNormalizer.normalize(p1)
        n2 = PredicateNormalizer.normalize(p2)
        if n1 == n2:
            return True
        return None


# ---------------------------------------------------------------------------
# PredicateInterpolation
# ---------------------------------------------------------------------------

class PredicateInterpolation:
    """Extracts relevant predicates from interpolants and formulas."""

    @staticmethod
    def extract_atoms(formula: PredicateTemplate) -> List[AtomicPredicate]:
        """Extract all atomic predicates from a formula."""
        return sorted(formula.atoms(), key=lambda a: a.pretty_print())

    @staticmethod
    def extract_relevant(formula: PredicateTemplate,
                         relevant_vars: FrozenSet[str]) -> List[AtomicPredicate]:
        """Extract atoms whose variables intersect with relevant_vars."""
        atoms = formula.atoms()
        return [a for a in atoms if a.variables() & relevant_vars]

    @staticmethod
    def interpolant_predicates(pre: PredicateTemplate,
                               post: PredicateTemplate) -> List[AtomicPredicate]:
        """Extract candidate interpolant predicates from pre and post.

        The interpolant predicates should use only variables common to both.
        """
        common_vars = pre.variables() & post.variables()
        pre_atoms = PredicateInterpolation.extract_relevant(pre, common_vars)
        post_atoms = PredicateInterpolation.extract_relevant(post, common_vars)
        all_atoms: Set[AtomicPredicate] = set(pre_atoms) | set(post_atoms)
        return sorted(all_atoms, key=lambda a: a.pretty_print())

    @staticmethod
    def generate_candidate_predicates(variables: Set[str],
                                      constants: Set[int],
                                      type_tags: Set[str]) -> List[AtomicPredicate]:
        """Generate candidate predicates from variables, constants, and type tags."""
        candidates: List[AtomicPredicate] = []
        for v in variables:
            candidates.append(NullityPredicate(variable=v, is_null=True))
            candidates.append(TruthinessPredicate(variable=v, is_truthy=True))
            for tag in type_tags:
                candidates.append(TypeTagPredicate(variable=v, type_tag=tag, positive=True))
            for c in constants:
                v_expr = LinearExpression.from_variable(v)
                c_expr = LinearExpression.from_int(c)
                candidates.append(ComparisonPredicate(left=v_expr, op=ComparisonOp.Eq, right=c_expr))
                candidates.append(ComparisonPredicate(left=v_expr, op=ComparisonOp.Lt, right=c_expr))
                candidates.append(ComparisonPredicate(left=v_expr, op=ComparisonOp.Le, right=c_expr))
                candidates.append(ComparisonPredicate(left=v_expr, op=ComparisonOp.Ge, right=c_expr))
                candidates.append(ComparisonPredicate(left=v_expr, op=ComparisonOp.Gt, right=c_expr))
        for v1, v2 in itertools.combinations(sorted(variables), 2):
            v1_expr = LinearExpression.from_variable(v1)
            v2_expr = LinearExpression.from_variable(v2)
            candidates.append(ComparisonPredicate(left=v1_expr, op=ComparisonOp.Eq, right=v2_expr))
            candidates.append(ComparisonPredicate(left=v1_expr, op=ComparisonOp.Lt, right=v2_expr))
            candidates.append(ComparisonPredicate(left=v1_expr, op=ComparisonOp.Le, right=v2_expr))
        return candidates

    @staticmethod
    def refine_from_counterexample(
        current_preds: PredicateSet,
        cex_env: Mapping[str, Any],
        candidate_preds: Sequence[AtomicPredicate],
    ) -> List[AtomicPredicate]:
        """Contract discovery: find predicates that distinguish the counterexample state."""
        evaluator = PredicateEvaluator(cex_env)
        distinguishing: List[AtomicPredicate] = []
        for p in candidate_preds:
            if p in current_preds:
                continue
            try:
                val = evaluator.evaluate(p)
                distinguishing.append(p)
            except (KeyError, TypeError):
                pass
        return distinguishing

    def __repr__(self) -> str:
        return "PredicateInterpolation()"


# ---------------------------------------------------------------------------
# MonotoneMaps
# ---------------------------------------------------------------------------

class MonotoneMaps:
    """Monotone maps between predicate lattices.

    Provides utilities for constructing and composing monotone functions
    on predicate lattice elements.
    """

    @staticmethod
    def identity() -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """The identity monotone map."""
        return lambda x: x

    @staticmethod
    def constant(value: PredicateLatticeElement) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """A constant monotone map."""
        return lambda _: value

    @staticmethod
    def compose(
        f: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        g: Callable[[PredicateLatticeElement], PredicateLatticeElement],
    ) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Compose two monotone maps: (f ∘ g)(x) = f(g(x))."""
        return lambda x: f(g(x))

    @staticmethod
    def join_map(
        f: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        g: Callable[[PredicateLatticeElement], PredicateLatticeElement],
    ) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Pointwise join of two monotone maps."""
        return lambda x: f(x).join(g(x))

    @staticmethod
    def meet_map(
        f: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        g: Callable[[PredicateLatticeElement], PredicateLatticeElement],
    ) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Pointwise meet of two monotone maps."""
        return lambda x: f(x).meet(g(x))

    @staticmethod
    def assume_true(pred: PredicateTemplate) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Monotone map that assumes a predicate is true."""
        def _assume(elem: PredicateLatticeElement) -> PredicateLatticeElement:
            new_true = elem.true_preds | frozenset({pred})
            new_false = elem.false_preds - frozenset({pred})
            return PredicateLatticeElement(
                true_preds=new_true,
                false_preds=new_false,
                universe=elem.universe,
            )
        return _assume

    @staticmethod
    def assume_false(pred: PredicateTemplate) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Monotone map that assumes a predicate is false."""
        def _assume(elem: PredicateLatticeElement) -> PredicateLatticeElement:
            new_true = elem.true_preds - frozenset({pred})
            new_false = elem.false_preds | frozenset({pred})
            return PredicateLatticeElement(
                true_preds=new_true,
                false_preds=new_false,
                universe=elem.universe,
            )
        return _assume

    @staticmethod
    def project(keep_preds: FrozenSet[PredicateTemplate]) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Project a lattice element onto a subset of predicates."""
        def _project(elem: PredicateLatticeElement) -> PredicateLatticeElement:
            return PredicateLatticeElement(
                true_preds=elem.true_preds & keep_preds,
                false_preds=elem.false_preds & keep_preds,
                universe=keep_preds,
            )
        return _project

    @staticmethod
    def extend(extra_preds: FrozenSet[PredicateTemplate]) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Extend a lattice element with additional unknown predicates."""
        def _extend(elem: PredicateLatticeElement) -> PredicateLatticeElement:
            return PredicateLatticeElement(
                true_preds=elem.true_preds,
                false_preds=elem.false_preds,
                universe=elem.universe | extra_preds,
            )
        return _extend

    @staticmethod
    def transfer_guard(
        guard: PredicateTemplate,
        branch: bool,
    ) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Transfer function for a conditional branch."""
        if branch:
            return MonotoneMaps.assume_true(guard)
        else:
            return MonotoneMaps.assume_false(guard)

    @staticmethod
    def sequential(
        *maps: Callable[[PredicateLatticeElement], PredicateLatticeElement],
    ) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Sequential composition of multiple monotone maps."""
        def _seq(elem: PredicateLatticeElement) -> PredicateLatticeElement:
            result = elem
            for m in maps:
                result = m(result)
            return result
        return _seq

    @staticmethod
    def parallel_join(
        *maps: Callable[[PredicateLatticeElement], PredicateLatticeElement],
    ) -> Callable[[PredicateLatticeElement], PredicateLatticeElement]:
        """Parallel composition with join (for merge points)."""
        def _par(elem: PredicateLatticeElement) -> PredicateLatticeElement:
            results = [m(elem) for m in maps]
            if not results:
                return elem
            result = results[0]
            for r in results[1:]:
                result = result.join(r)
            return result
        return _par

    @staticmethod
    def is_monotone_sample(
        f: Callable[[PredicateLatticeElement], PredicateLatticeElement],
        samples: Sequence[Tuple[PredicateLatticeElement, PredicateLatticeElement]],
    ) -> bool:
        """Check monotonicity on sample pairs: a ⊑ b → f(a) ⊑ f(b)."""
        for a, b in samples:
            if a.leq(b):
                fa = f(a)
                fb = f(b)
                if not fa.leq(fb):
                    return False
        return True

    def __repr__(self) -> str:
        return "MonotoneMaps()"
