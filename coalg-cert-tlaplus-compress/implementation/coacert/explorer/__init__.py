"""
Explicit-state explorer module for CoaCert-TLA.

Provides state-space exploration, symmetry reduction, fairness tracking,
high-performance hashing, and execution trace management for TLA+ models.
"""

from .graph import TransitionGraph, StateNode, TransitionEdge
from .explorer import ExplicitStateExplorer, ExplorationStats
from .symmetry import SymmetryDetector, PermutationGroup, Orbit
from .fairness import FairnessTracker, FairnessConstraint, AcceptancePair
from .hash_table import StateHashTable, ZobristHasher
from .traces import ExecutionTrace, LassoTrace, TraceManager

__all__ = [
    "TransitionGraph",
    "StateNode",
    "TransitionEdge",
    "ExplicitStateExplorer",
    "ExplorationStats",
    "SymmetryDetector",
    "PermutationGroup",
    "Orbit",
    "FairnessTracker",
    "FairnessConstraint",
    "AcceptancePair",
    "StateHashTable",
    "ZobristHasher",
    "ExecutionTrace",
    "LassoTrace",
    "TraceManager",
]
