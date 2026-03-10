"""
usability_oracle.alignment — Semantic tree alignment module.

Computes a structural diff between two UI accessibility-tree versions using
a 3-pass pipeline:

1. **Exact matching** — pair nodes by semantic hash, id, and tree path.
2. **Fuzzy matching** — bipartite optimal assignment via the Hungarian
   algorithm on a multi-dimensional similarity matrix.
3. **Residual classification** — classify unmatched nodes as moves, renames,
   retypes, additions, or removals.

Quick start::

    from usability_oracle.alignment import SemanticDiffer, AlignmentConfig

    differ = SemanticDiffer(AlignmentConfig())
    result = differ.diff(source_tree, target_tree)
    print(result.summary())
"""

from __future__ import annotations

# --- models -----------------------------------------------------------------
from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityRole,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentContext,
    AlignmentPass,
    AlignmentResult,
    BoundingBox,
    EditOperation,
    EditOperationType,
    NodeMapping,
    Point2D,
)

# --- passes -----------------------------------------------------------------
from usability_oracle.alignment.exact_match import ExactMatcher
from usability_oracle.alignment.fuzzy_match import FuzzyMatcher
from usability_oracle.alignment.classifier import ResidualClassifier

# --- cost model & differ ----------------------------------------------------
from usability_oracle.alignment.cost_model import AlignmentCostModel
from usability_oracle.alignment.differ import SemanticDiffer

# --- visualisation ----------------------------------------------------------
from usability_oracle.alignment.visualizer import AlignmentVisualizer

__all__ = [
    # models / data classes
    "AccessibilityNode",
    "AccessibilityRole",
    "AccessibilityTree",
    "AlignmentConfig",
    "AlignmentContext",
    "AlignmentPass",
    "AlignmentResult",
    "BoundingBox",
    "EditOperation",
    "EditOperationType",
    "NodeMapping",
    "Point2D",
    # pass implementations
    "ExactMatcher",
    "FuzzyMatcher",
    "ResidualClassifier",
    # cost model & orchestration
    "AlignmentCostModel",
    "SemanticDiffer",
    # visualisation
    "AlignmentVisualizer",
]
