"""
Bug Detection Engine.

Queries inferred refinement types to detect concrete bug instances:
array out-of-bounds, null/None dereference, division by zero,
and type-tag confusion.
"""

from src.bugs.detector import (
    BugDetector,
    BugReport,
    BugClass,
    Confidence,
)

__all__ = [
    "BugDetector",
    "BugReport",
    "BugClass",
    "Confidence",
]
