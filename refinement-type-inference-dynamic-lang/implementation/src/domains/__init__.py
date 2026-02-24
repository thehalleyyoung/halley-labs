"""
Abstract Domains — Reduced product of numeric, type-tag, nullity, and string domains.

Each domain forms a Galois connection with the concrete powerset domain,
ensuring soundness of abstract transformers.
"""

from src.domains.base import AbstractDomain, AbstractValue, Lattice
from src.domains.intervals import IntervalDomain, Interval, Bound
from src.domains.typetags import TypeTagDomain, TypeTagSet
from src.domains.nullity import NullityDomain, NullityValue
from src.domains.product import ReducedProductDomain, ProductValue

__all__ = [
    "AbstractDomain",
    "AbstractValue",
    "Lattice",
    "IntervalDomain",
    "Interval",
    "Bound",
    "TypeTagDomain",
    "TypeTagSet",
    "NullityDomain",
    "NullityValue",
    "ReducedProductDomain",
    "ProductValue",
]
