"""CausalQD: Quality-Diversity Illumination for Causal Discovery.

This package implements MAP-Elites for causal structure learning,
exploring the space of Directed Acyclic Graphs (DAGs) while
maintaining diversity across Markov Equivalence Classes (MECs).
"""

from __future__ import annotations

__version__ = "0.1.0"

from causal_qd.types import (
    AdjacencyMatrix,
    BehavioralDescriptor,
    CellIndex,
    DataMatrix,
    EdgeList,
    EdgeMask,
    GraphHash,
    QualityScore,
    TopologicalOrder,
)

__all__ = [
    "__version__",
    "AdjacencyMatrix",
    "BehavioralDescriptor",
    "CellIndex",
    "DataMatrix",
    "EdgeList",
    "EdgeMask",
    "GraphHash",
    "QualityScore",
    "TopologicalOrder",
]
