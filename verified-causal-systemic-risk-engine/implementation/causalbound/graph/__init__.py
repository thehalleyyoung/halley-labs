"""
CausalBound Graph Decomposition Module.

Provides tree decomposition of financial networks into bounded-treewidth
subgraphs, with causal-aware partitioning that preserves interventional
semantics across partition boundaries.

Main classes:
    - TreeDecomposer: Tree decomposition via elimination orderings
    - SeparatorExtractor: Separator set extraction and enumeration
    - CausalPartitioner: Causal-structure-aware DAG partitioning
    - TreewidthEstimator: Upper/lower bounds on treewidth
    - SubgraphExtractor: Induced subgraph extraction with boundary handling
    - MoralGraphConstructor: Moral/ancestral graph construction and triangulation
"""

from causalbound.graph.decomposition import TreeDecomposer
from causalbound.graph.separator import SeparatorExtractor
from causalbound.graph.causal_partition import CausalPartitioner
from causalbound.graph.treewidth import TreewidthEstimator
from causalbound.graph.subgraph import SubgraphExtractor
from causalbound.graph.moral import MoralGraphConstructor

__all__ = [
    "TreeDecomposer",
    "SeparatorExtractor",
    "CausalPartitioner",
    "TreewidthEstimator",
    "SubgraphExtractor",
    "MoralGraphConstructor",
]
