"""Core type definitions for the CausalQD framework.

These type aliases provide semantic meaning to numpy arrays and primitive types
used throughout the causal discovery pipeline.  Also includes enums and
dataclasses shared across modules.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Graph representation types
# ---------------------------------------------------------------------------

AdjacencyMatrix = npt.NDArray[np.int8]
"""n × n binary matrix where A[i,j] = 1 iff edge i → j exists."""

WeightedAdjacencyMatrix = npt.NDArray[np.float64]
"""n × n real-valued matrix of edge weights."""

TopologicalOrder = List[int]
"""A valid topological ordering of node indices."""

EdgeList = List[Tuple[int, int]]
"""List of directed edges as (source, target) pairs."""

EdgeMask = npt.NDArray[np.bool_]
"""n × n boolean mask over edges."""

NodeSet = FrozenSet[int]
"""Immutable set of node indices."""

# ---------------------------------------------------------------------------
# MAP-Elites / Quality-Diversity types
# ---------------------------------------------------------------------------

BehavioralDescriptor = npt.NDArray[np.float64]
"""d-dimensional real vector characterizing a solution's behaviour in descriptor space."""

QualityScore = float
"""Scalar fitness/quality value (higher is better by convention)."""

CellIndex = Tuple[int, ...]
"""Multi-dimensional index into the archive grid."""

# ---------------------------------------------------------------------------
# Statistical types
# ---------------------------------------------------------------------------

PValue = float
"""p-value in [0, 1] from a conditional independence test."""

ConfidenceInterval = Tuple[float, float]
"""Lower and upper bounds of a confidence interval."""

CorrelationMatrix = npt.NDArray[np.float64]
"""n × n symmetric positive-definite matrix of pairwise correlations."""

DataMatrix = npt.NDArray[np.float64]
"""N × p data matrix with N observations and p variables."""

SampleWeight = npt.NDArray[np.float64]
"""N-length vector of observation weights."""

# ---------------------------------------------------------------------------
# Certificate types
# ---------------------------------------------------------------------------

CertificateVector = npt.NDArray[np.float64]
"""Vector of per-edge or per-path certificate values."""

BootstrapSample = npt.NDArray[np.int64]
"""Index array for a bootstrap resample."""

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EdgeType(enum.Enum):
    """Classification of an edge in a (partially) directed graph."""
    DIRECTED = "directed"       # i -> j
    UNDIRECTED = "undirected"   # i -- j (stored as both i->j and j->i)
    BIDIRECTED = "bidirected"   # i <-> j (latent common cause)
    NONE = "none"               # no edge


class GraphType(enum.Enum):
    """Type of graph structure."""
    DAG = "dag"
    CPDAG = "cpdag"
    PDAG = "pdag"
    MAG = "mag"
    PAG = "pag"
    UNDIRECTED = "undirected"


class MutationType(enum.Enum):
    """Types of DAG mutation operations."""
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    REVERSE_EDGE = "reverse_edge"
    COMPOSITE = "composite"


class ScoreType(enum.Enum):
    """Scoring criterion for structure learning."""
    BIC = "bic"
    BDeu = "bdeu"
    BGe = "bge"
    AIC = "aic"
    LOG_LIKELIHOOD = "log_likelihood"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """A single directed or undirected edge."""
    source: int
    target: int
    edge_type: EdgeType = EdgeType.DIRECTED
    weight: float = 1.0


@dataclass
class MutationRecord:
    """Record of a mutation applied to a DAG."""
    mutation_type: MutationType
    edge: Optional[Tuple[int, int]] = None
    success: bool = True
    created_cycle: bool = False


@dataclass
class CrossoverRecord:
    """Record of a crossover operation."""
    parent1_edges: int = 0
    parent2_edges: int = 0
    child_edges: int = 0
    edges_from_parent1: int = 0
    edges_from_parent2: int = 0
    edges_removed_for_acyclicity: int = 0


@dataclass
class GraphProperties:
    """Cached structural properties of a graph."""
    num_nodes: int = 0
    num_edges: int = 0
    max_in_degree: int = 0
    max_out_degree: int = 0
    num_v_structures: int = 0
    num_connected_components: int = 0
    longest_path_length: int = 0
    density: float = 0.0

    @classmethod
    def from_adjacency(cls, adj: AdjacencyMatrix) -> "GraphProperties":
        """Compute properties from an adjacency matrix."""
        n = adj.shape[0]
        ne = int(adj.sum())
        max_possible = n * (n - 1) if n > 1 else 1
        return cls(
            num_nodes=n,
            num_edges=ne,
            max_in_degree=int(adj.sum(axis=0).max()) if n > 0 else 0,
            max_out_degree=int(adj.sum(axis=1).max()) if n > 0 else 0,
            density=ne / max_possible if max_possible > 0 else 0.0,
        )


@dataclass
class ArchiveEntry:
    """An entry in the MAP-Elites archive."""
    adjacency: AdjacencyMatrix
    quality: QualityScore
    descriptor: BehavioralDescriptor
    cell_index: CellIndex
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MutationConfig:
    """Configuration for mutation operators."""
    add_prob: float = 0.4
    remove_prob: float = 0.3
    reverse_prob: float = 0.3
    max_parents: int = -1  # -1 means no limit

    def __post_init__(self) -> None:
        total = self.add_prob + self.remove_prob + self.reverse_prob
        if abs(total - 1.0) > 1e-9:
            self.add_prob /= total
            self.remove_prob /= total
            self.reverse_prob /= total


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

T = TypeVar("T")
GraphHash = int
"""Canonical hash of a graph structure (e.g., via nauty)."""

ParamDict = Dict[str, Union[int, float, str, bool, None]]
"""Generic parameter dictionary."""
