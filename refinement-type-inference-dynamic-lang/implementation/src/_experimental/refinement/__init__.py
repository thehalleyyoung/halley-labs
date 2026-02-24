"""
Specialized refinement modules for Python-specific idioms and patterns.

The core guard-based system handles flat guards (isinstance, is None, comparisons).
These modules extend it to handle richer Python patterns:

- exception_refinements: EAFP patterns, try/except flow control
- comprehension_refinements: list/dict/set comprehension filter narrowing
- unpacking_refinements: structural refinements from tuple/star unpacking
- decorator_refinements: function type transformations via decorators
- async_refinements: async/await, async for, async with patterns
- string_refinements: string method predicates, regex, taint tracking
- numeric_refinements: arithmetic refinement propagation
"""

from src.refinement.exception_refinements import ExceptionRefinementAnalyzer
from src.refinement.comprehension_refinements import ComprehensionAnalyzer
from src.refinement.unpacking_refinements import UnpackingAnalyzer
from src.refinement.decorator_refinements import DecoratorAnalyzer
from src.refinement.async_refinements import AsyncAnalyzer
from src.refinement.string_refinements import StringRefinementAnalyzer
from src.refinement.numeric_refinements import NumericRefinementAnalyzer

__all__ = [
    "ExceptionRefinementAnalyzer",
    "ComprehensionAnalyzer",
    "UnpackingAnalyzer",
    "DecoratorAnalyzer",
    "AsyncAnalyzer",
    "StringRefinementAnalyzer",
    "NumericRefinementAnalyzer",
]
