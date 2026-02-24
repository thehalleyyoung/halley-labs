"""
Predicate Template Language P.

Quantifier-free predicate language over four sorts (Int, Bool, Tag, Str)
with function symbols for len, isinstance, is_none, is_truthy, hasattr,
arithmetic, and comparisons.
"""

from src.predicates.templates import (
    PredicateTemplate,
    AtomicPredicate,
    ComparisonPredicate,
    TypeTagPredicate,
    NullityPredicate,
    TruthinessPredicate,
    HasAttrPredicate,
    Conjunction,
    Disjunction,
    Negation,
)
from src.predicates.matching import GuardPatternMatcher
from src.predicates.lattice import PredicateLattice

__all__ = [
    "PredicateTemplate",
    "AtomicPredicate",
    "ComparisonPredicate",
    "TypeTagPredicate",
    "NullityPredicate",
    "TruthinessPredicate",
    "HasAttrPredicate",
    "Conjunction",
    "Disjunction",
    "Negation",
    "GuardPatternMatcher",
    "PredicateLattice",
]
