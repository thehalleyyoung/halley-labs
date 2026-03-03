"""CPA scalability subpackage.

Scalability infrastructure for large-scale causal discovery including
parent-set score caching, approximate scoring, sparse matrix operations,
and distributed computation primitives.

Modules
-------
parent_set_cache
    Parent-set score caching with LRU eviction.
approximate_scores
    Approximate scoring for large graphs.
sparse_operations
    Sparse matrix operations for large DAGs.
distributed
    Distributed computation primitives.
"""

from cpa.scalability.parent_set_cache import (
    ParentSetCache,
    CacheStats,
    TieredCache,
)
from cpa.scalability.approximate_scores import (
    ApproximateScorer,
    ScoreApproximation,
    ApproximateBIC,
    ScreeningScore,
    RandomizedScore,
)
from cpa.scalability.sparse_operations import (
    SparseDAG,
    SparseAlignmentMatrix,
    sparse_dag_operations,
)
from cpa.scalability.distributed import (
    DistributedEngine,
    WorkPartitioner,
    TaskResult,
    DAGParallelEvaluator,
    SharedScoreCache,
)

__all__ = [
    # parent_set_cache.py
    "ParentSetCache",
    "CacheStats",
    "TieredCache",
    # approximate_scores.py
    "ApproximateScorer",
    "ScoreApproximation",
    "ApproximateBIC",
    "ScreeningScore",
    "RandomizedScore",
    # sparse_operations.py
    "SparseDAG",
    "SparseAlignmentMatrix",
    "sparse_dag_operations",
    # distributed.py
    "DistributedEngine",
    "WorkPartitioner",
    "TaskResult",
    "DAGParallelEvaluator",
    "SharedScoreCache",
]
