"""
Incremental Dependency Tracker.

Maintains function-level dependency graph and implements predicate-sensitive
invalidation for efficient re-analysis of changed code.
"""

from src.incremental.tracker import (
    IncrementalTracker,
    DependencyGraph,
    AnalysisCache,
    ChangeSet,
)

__all__ = [
    "IncrementalTracker",
    "DependencyGraph",
    "AnalysisCache",
    "ChangeSet",
]
