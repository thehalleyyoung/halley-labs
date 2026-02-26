"""
CausalBound: Verified Worst-Case Systemic Risk Bounds via Decomposed
Causal Polytope Inference with Adversarial Search.

This package implements the full CausalBound pipeline:
1. Graph decomposition into bounded-treewidth subgraphs
2. Causal polytope LP solving with column generation
3. Junction-tree exact inference with do-operator support
4. Streaming SMT verification of inference steps
5. MCTS adversarial search with causal UCB pruning
6. Financial contagion models (DebtRank, cascades, fire-sales)
7. Bound composition theorem for global risk aggregation
"""

__version__ = "1.0.0"
__author__ = "CausalBound Team"

from causalbound.graph.decomposition import TreeDecomposer
from causalbound.polytope.causal_polytope import CausalPolytopeSolver
from causalbound.composition.composer import BoundComposer
from causalbound.junction.engine import JunctionTreeEngine
from causalbound.smt.verifier import SMTVerifier
from causalbound.mcts.search import MCTSSearch
from causalbound.scm.builder import SCMBuilder
from causalbound.contagion.debtrank import DebtRankModel

__all__ = [
    "TreeDecomposer",
    "CausalPolytopeSolver",
    "BoundComposer",
    "JunctionTreeEngine",
    "SMTVerifier",
    "MCTSSearch",
    "SCMBuilder",
    "DebtRankModel",
]
