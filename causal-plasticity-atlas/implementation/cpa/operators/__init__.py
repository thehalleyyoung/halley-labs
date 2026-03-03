"""CPA genetic operators subpackage.

Advanced genetic operators for quality-diversity search over DAG space
including crossover, mutation, constraint-respecting operators, and
local search operators.

Modules
-------
crossover
    Markov blanket and edge-preserving crossover.
mutation
    Targeted mutation operators.
constrained
    Constraint-respecting operators.
local_search
    Local search operators (GES-like).
"""

from cpa.operators.crossover import (
    MarkovBlanketCrossover,
    EdgePreservingCrossover,
    SubgraphCrossover,
    UniformCrossover,
)
from cpa.operators.mutation import (
    TargetedMutation,
    DAGMutation,
    EdgeMutation,
    WeightMutation,
    StructuralMutation,
    AdaptiveMutation,
)
from cpa.operators.constrained import (
    ConstrainedOperator,
    EdgeConstraints,
    ConstrainedMutation,
    ConstrainedCrossover,
)
from cpa.operators.local_search import (
    ForwardOperator,
    BackwardOperator,
    TurnOperator,
    GESLocalSearch,
    ForwardStep,
    BackwardStep,
    TurnStep,
    GreedyLocalSearch,
)

__all__ = [
    # crossover.py
    "MarkovBlanketCrossover",
    "EdgePreservingCrossover",
    "SubgraphCrossover",
    "UniformCrossover",
    # mutation.py
    "TargetedMutation",
    "DAGMutation",
    "EdgeMutation",
    "WeightMutation",
    "StructuralMutation",
    "AdaptiveMutation",
    # constrained.py
    "ConstrainedOperator",
    "EdgeConstraints",
    "ConstrainedMutation",
    "ConstrainedCrossover",
    # local_search.py
    "ForwardOperator",
    "BackwardOperator",
    "TurnOperator",
    "GESLocalSearch",
    "ForwardStep",
    "BackwardStep",
    "TurnStep",
    "GreedyLocalSearch",
]
