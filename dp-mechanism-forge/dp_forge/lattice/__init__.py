"""
Lattice-based mechanism synthesis via enumeration and branch-and-bound.

This package synthesises DP mechanisms by exploring a lattice of candidate
mechanism structures. The lattice is ordered by a refinement relation
(e.g., finer discretisation or more support points), and the search
exploits monotonicity properties of the DP feasibility criterion.

Key algorithms:
- **Lattice enumeration**: Systematic enumeration of mechanism structures
  ordered by complexity (number of support points, grid resolution).
- **Branch and bound (B&B)**: Prune subtrees of the lattice by computing
  bounds on the best achievable objective and feasibility.
- **Anti-chain search**: Find Pareto-optimal mechanisms that trade off
  utility vs. mechanism complexity.
- **Lattice join/meet**: Combine mechanisms via lattice operations to
  create new candidates.

Architecture:
    1. **MechanismLattice** — Defines the lattice structure over mechanism
       designs (partial order, successors, predecessors).
    2. **LatticeEnumerator** — Enumerates lattice points in a specified order.
    3. **BranchAndBound** — B&B search with bounding and pruning.
    4. **BoundComputer** — Computes upper and lower bounds for B&B.
    5. **LatticeAnalyzer** — Post-hoc analysis of the lattice search.

Example::

    from dp_forge.lattice import BranchAndBound, LatticeConfig

    config = LatticeConfig(max_nodes=10000, pruning_strategy="dominance")
    bnb = BranchAndBound(config=config)
    result = bnb.search(query_spec)
    print(f"Optimal found at depth {result.optimal_depth}")
    print(f"Nodes explored: {result.nodes_explored}/{result.nodes_total}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    LatticePoint,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TraversalOrder(Enum):
    """Order for traversing the mechanism lattice."""

    BREADTH_FIRST = auto()
    DEPTH_FIRST = auto()
    BEST_FIRST = auto()
    BOTTOM_UP = auto()
    TOP_DOWN = auto()

    def __repr__(self) -> str:
        return f"TraversalOrder.{self.name}"


class PruningStrategy(Enum):
    """Pruning strategy for branch-and-bound."""

    BOUND_BASED = auto()
    DOMINANCE = auto()
    SYMMETRY = auto()
    FEASIBILITY = auto()
    COMBINED = auto()

    def __repr__(self) -> str:
        return f"PruningStrategy.{self.name}"


class BoundType(Enum):
    """Type of bound used in branch-and-bound."""

    LP_RELAXATION = auto()
    LAGRANGIAN = auto()
    COMBINATORIAL = auto()
    DUAL = auto()

    def __repr__(self) -> str:
        return f"BoundType.{self.name}"


class NodeStatus(Enum):
    """Status of a node in the B&B search tree."""

    UNEXPLORED = auto()
    ACTIVE = auto()
    PRUNED = auto()
    FEASIBLE = auto()
    OPTIMAL = auto()
    INFEASIBLE = auto()

    def __repr__(self) -> str:
        return f"NodeStatus.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LatticeConfig:
    """Configuration for lattice-based mechanism synthesis.

    Attributes:
        traversal_order: Order for lattice traversal.
        pruning_strategy: Pruning strategy for B&B.
        bound_type: Type of bounds for B&B.
        max_nodes: Maximum number of nodes to explore.
        max_depth: Maximum depth in the lattice.
        convergence_tol: Tolerance for optimality gap.
        timeout_seconds: Maximum wall-clock time.
        min_support: Minimum support size for mechanisms.
        max_support: Maximum support size for mechanisms.
        symmetry_breaking: Whether to use symmetry breaking.
        verbose: Verbosity level.
    """

    traversal_order: TraversalOrder = TraversalOrder.BEST_FIRST
    pruning_strategy: PruningStrategy = PruningStrategy.COMBINED
    bound_type: BoundType = BoundType.LP_RELAXATION
    max_nodes: int = 10000
    max_depth: int = 50
    convergence_tol: float = 1e-6
    timeout_seconds: float = 300.0
    min_support: int = 2
    max_support: int = 1000
    symmetry_breaking: bool = True
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {self.max_nodes}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.convergence_tol <= 0:
            raise ValueError(f"convergence_tol must be > 0, got {self.convergence_tol}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.min_support < 1:
            raise ValueError(f"min_support must be >= 1, got {self.min_support}")
        if self.max_support < self.min_support:
            raise ValueError(
                f"max_support ({self.max_support}) must be >= "
                f"min_support ({self.min_support})"
            )

    def __repr__(self) -> str:
        return (
            f"LatticeConfig(order={self.traversal_order.name}, "
            f"pruning={self.pruning_strategy.name}, max_nodes={self.max_nodes})"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class LatticeNode:
    """A node in the mechanism design lattice / B&B search tree.

    Attributes:
        node_id: Unique identifier.
        point: The lattice point (mechanism parameters).
        depth: Depth in the lattice/tree.
        status: Current status of this node.
        lower_bound: Lower bound on objective at this node.
        upper_bound: Upper bound on objective at this node.
        parent_id: ID of the parent node (None for root).
        children_ids: IDs of child nodes.
    """

    node_id: int
    point: LatticePoint
    depth: int
    status: NodeStatus = NodeStatus.UNEXPLORED
    lower_bound: float = float("-inf")
    upper_bound: float = float("inf")
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    @property
    def gap(self) -> float:
        """Gap between upper and lower bounds."""
        if self.upper_bound == float("inf") or self.lower_bound == float("-inf"):
            return float("inf")
        return self.upper_bound - self.lower_bound

    @property
    def is_leaf(self) -> bool:
        """Whether this is a leaf node."""
        return len(self.children_ids) == 0

    def __repr__(self) -> str:
        return (
            f"LatticeNode(id={self.node_id}, depth={self.depth}, "
            f"status={self.status.name}, gap={self.gap:.2e})"
        )


@dataclass
class BranchDecision:
    """A branching decision in the B&B tree.

    Attributes:
        parent_id: Node being branched.
        variable_index: Index of the variable to branch on.
        branch_value: Value at which to branch.
        left_bound: New bound for the left child.
        right_bound: New bound for the right child.
    """

    parent_id: int
    variable_index: int
    branch_value: float
    left_bound: float
    right_bound: float

    def __repr__(self) -> str:
        return (
            f"BranchDecision(parent={self.parent_id}, var={self.variable_index}, "
            f"val={self.branch_value:.4f})"
        )


@dataclass
class LatticeSearchResult:
    """Result of a lattice-based mechanism search.

    Attributes:
        mechanism: Best mechanism found (n × k probability table).
        objective_value: Objective value of the best mechanism.
        optimal_depth: Depth at which the optimal was found.
        nodes_explored: Number of nodes explored.
        nodes_pruned: Number of nodes pruned.
        nodes_total: Total nodes in the lattice (estimated).
        lower_bound: Global lower bound.
        upper_bound: Global upper bound (= objective_value if optimal).
        optimality_certificate: Duality-based certificate.
        search_time: Total search time in seconds.
        convergence_history: Best objective at each iteration.
    """

    mechanism: npt.NDArray[np.float64]
    objective_value: float
    optimal_depth: int
    nodes_explored: int
    nodes_pruned: int
    nodes_total: int
    lower_bound: float
    upper_bound: float
    optimality_certificate: Optional[OptimalityCertificate] = None
    search_time: float = 0.0
    convergence_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mechanism = np.asarray(self.mechanism, dtype=np.float64)
        if self.mechanism.ndim != 2:
            raise ValueError(f"mechanism must be 2-D, got shape {self.mechanism.shape}")

    @property
    def gap(self) -> float:
        """Relative optimality gap."""
        return (self.upper_bound - self.lower_bound) / max(abs(self.upper_bound), 1.0)

    @property
    def pruning_ratio(self) -> float:
        """Fraction of nodes pruned."""
        total = self.nodes_explored + self.nodes_pruned
        if total == 0:
            return 0.0
        return self.nodes_pruned / total

    @property
    def is_optimal(self) -> bool:
        """Whether the solution is provably optimal."""
        return self.gap < 1e-6

    def __repr__(self) -> str:
        opt = "optimal" if self.is_optimal else f"gap={self.gap:.2e}"
        return (
            f"LatticeSearchResult(obj={self.objective_value:.6f}, {opt}, "
            f"explored={self.nodes_explored}, pruned={self.nodes_pruned}, "
            f"time={self.search_time:.2f}s)"
        )


@dataclass
class ParetoFront:
    """Pareto-optimal mechanisms trading off utility vs. complexity.

    Attributes:
        mechanisms: List of Pareto-optimal mechanisms.
        objective_values: Objective (utility loss) for each mechanism.
        complexities: Complexity measure (e.g., support size) for each.
    """

    mechanisms: List[npt.NDArray[np.float64]]
    objective_values: List[float]
    complexities: List[int]

    def __post_init__(self) -> None:
        if not (len(self.mechanisms) == len(self.objective_values) == len(self.complexities)):
            raise ValueError("All lists must have the same length")

    @property
    def size(self) -> int:
        """Number of Pareto-optimal solutions."""
        return len(self.mechanisms)

    def __repr__(self) -> str:
        if self.size == 0:
            return "ParetoFront(empty)"
        return (
            f"ParetoFront(size={self.size}, "
            f"obj_range=[{min(self.objective_values):.4f}, "
            f"{max(self.objective_values):.4f}])"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class BoundComputer(Protocol):
    """Protocol for computing bounds in branch-and-bound."""

    def lower_bound(self, node: LatticeNode, spec: QuerySpec) -> float:
        """Compute a lower bound on the objective at this node."""
        ...

    def upper_bound(self, node: LatticeNode, spec: QuerySpec) -> float:
        """Compute an upper bound on the objective at this node."""
        ...


@runtime_checkable
class BranchingRule(Protocol):
    """Protocol for selecting branching variables in B&B."""

    def select(self, node: LatticeNode, spec: QuerySpec) -> BranchDecision:
        """Select a variable to branch on."""
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class MechanismLattice:
    """Defines the lattice structure over mechanism designs.

    The lattice is ordered by refinement: mechanism A ⊑ B if B uses
    a finer discretisation or more support points than A.
    """

    def __init__(self, spec: QuerySpec, config: Optional[LatticeConfig] = None) -> None:
        self.spec = spec
        self.config = config or LatticeConfig()

    def root(self) -> LatticeNode:
        """Return the root (coarsest) node of the lattice."""
        raise NotImplementedError("MechanismLattice.root")

    def successors(self, node: LatticeNode) -> List[LatticeNode]:
        """Return the successors (refinements) of a node."""
        raise NotImplementedError("MechanismLattice.successors")

    def predecessors(self, node: LatticeNode) -> List[LatticeNode]:
        """Return the predecessors (coarsenings) of a node."""
        raise NotImplementedError("MechanismLattice.predecessors")

    def join(self, a: LatticeNode, b: LatticeNode) -> LatticeNode:
        """Compute the join (least upper bound) of two nodes."""
        raise NotImplementedError("MechanismLattice.join")

    def meet(self, a: LatticeNode, b: LatticeNode) -> LatticeNode:
        """Compute the meet (greatest lower bound) of two nodes."""
        raise NotImplementedError("MechanismLattice.meet")

    def is_feasible(self, node: LatticeNode) -> bool:
        """Check if a lattice point yields a DP-feasible mechanism."""
        raise NotImplementedError("MechanismLattice.is_feasible")


class LatticeEnumerator:
    """Enumerate mechanism lattice points in a specified order."""

    def __init__(self, config: Optional[LatticeConfig] = None) -> None:
        self.config = config or LatticeConfig()

    def enumerate(
        self,
        spec: QuerySpec,
        *,
        max_points: Optional[int] = None,
    ) -> List[LatticePoint]:
        """Enumerate lattice points up to a limit.

        Args:
            spec: Query specification.
            max_points: Maximum number of points to enumerate.

        Returns:
            List of lattice points in the specified order.
        """
        raise NotImplementedError("LatticeEnumerator.enumerate")

    def enumerate_feasible(
        self,
        spec: QuerySpec,
        budget: PrivacyBudget,
        *,
        max_points: Optional[int] = None,
    ) -> List[LatticePoint]:
        """Enumerate only DP-feasible lattice points.

        Args:
            spec: Query specification.
            budget: Privacy budget for feasibility checking.
            max_points: Maximum number of feasible points.

        Returns:
            List of feasible lattice points.
        """
        raise NotImplementedError("LatticeEnumerator.enumerate_feasible")


class BranchAndBound:
    """Branch-and-bound search over the mechanism lattice.

    Systematically explores the lattice with pruning based on
    computed bounds. Guarantees finding the optimal mechanism
    (up to convergence tolerance).
    """

    def __init__(self, config: Optional[LatticeConfig] = None) -> None:
        self.config = config or LatticeConfig()

    def search(self, spec: QuerySpec) -> LatticeSearchResult:
        """Run B&B search for the optimal mechanism.

        Args:
            spec: Query specification.

        Returns:
            LatticeSearchResult with the best mechanism found.
        """
        raise NotImplementedError("BranchAndBound.search")

    def search_with_initial(
        self,
        spec: QuerySpec,
        initial_mechanism: npt.NDArray[np.float64],
    ) -> LatticeSearchResult:
        """Run B&B with an initial feasible mechanism for warm-starting.

        Args:
            spec: Query specification.
            initial_mechanism: Initial feasible mechanism.

        Returns:
            LatticeSearchResult with the improved mechanism.
        """
        raise NotImplementedError("BranchAndBound.search_with_initial")

    def pareto_search(
        self,
        spec: QuerySpec,
        *,
        complexity_budget: int = 100,
    ) -> ParetoFront:
        """Search for the Pareto front of utility vs. complexity.

        Args:
            spec: Query specification.
            complexity_budget: Maximum mechanism complexity.

        Returns:
            ParetoFront with Pareto-optimal mechanisms.
        """
        raise NotImplementedError("BranchAndBound.pareto_search")


class LatticeAnalyzer:
    """Post-hoc analysis of lattice search results."""

    def sensitivity_analysis(
        self,
        result: LatticeSearchResult,
        spec: QuerySpec,
    ) -> Dict[str, float]:
        """Analyse sensitivity of the optimal mechanism to parameters.

        Args:
            result: Search result to analyse.
            spec: Query specification.

        Returns:
            Dict of parameter sensitivities.
        """
        raise NotImplementedError("LatticeAnalyzer.sensitivity_analysis")

    def compare_with_baseline(
        self,
        result: LatticeSearchResult,
        baseline_mechanism: npt.NDArray[np.float64],
    ) -> Dict[str, float]:
        """Compare lattice-optimal mechanism against a baseline.

        Args:
            result: Lattice search result.
            baseline_mechanism: Baseline mechanism for comparison.

        Returns:
            Dict of comparison metrics.
        """
        raise NotImplementedError("LatticeAnalyzer.compare_with_baseline")


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def lattice_synthesize(
    spec: QuerySpec,
    *,
    config: Optional[LatticeConfig] = None,
) -> LatticeSearchResult:
    """Convenience function for lattice-based mechanism synthesis.

    Args:
        spec: Query specification.
        config: Optional lattice configuration.

    Returns:
        LatticeSearchResult with the optimal mechanism.
    """
    raise NotImplementedError("lattice_synthesize")


def estimate_lattice_size(spec: QuerySpec) -> int:
    """Estimate the size of the mechanism lattice for a given spec.

    Args:
        spec: Query specification.

    Returns:
        Estimated number of lattice points.
    """
    raise NotImplementedError("estimate_lattice_size")


from dp_forge.lattice.reduction import (  # noqa: E402
    GramSchmidt,
    GramSchmidtResult,
    LLLReduction,
    BKZReduction,
    HermiteNormalForm,
    ShortVectorProblem,
    ClosestVectorProblem,
)
from dp_forge.lattice.enumeration import (  # noqa: E402
    PruningType,
    PruningBounds,
    PruningStrategy as EnumPruningStrategy,
    LatticeEnumerator as PointEnumerator,
    KannanEnumeration,
    FinckePohlstEnumeration,
    BoundedEnumeration,
    EnumerationNode,
    EnumerationTree,
)
from dp_forge.lattice.branch_and_bound import (  # noqa: E402
    SelectionStrategy,
    BranchingHeuristic,
    BBNode,
    BBResult,
    BoundComputation,
    BranchingStrategy,
    NodeSelection,
    SymmetryBreaking,
    CuttingPlanes,
    BranchAndBound as BBSolver,
)
from dp_forge.lattice.mechanism_lattice import (  # noqa: E402
    FeasibilityChecker,
    DominanceRelation,
    ParetoFront as MechanismParetoFront,
    LatticeProjection,
    MechanismLattice as MechLattice,
    DiscreteOptimizer,
)

__all__ = [
    # Enums
    "TraversalOrder",
    "PruningStrategy",
    "BoundType",
    "NodeStatus",
    # Config
    "LatticeConfig",
    # Data types
    "LatticeNode",
    "BranchDecision",
    "LatticeSearchResult",
    "ParetoFront",
    # Protocols
    "BoundComputer",
    "BranchingRule",
    # Classes
    "MechanismLattice",
    "LatticeEnumerator",
    "BranchAndBound",
    "LatticeAnalyzer",
    # Functions
    "lattice_synthesize",
    "estimate_lattice_size",
    # --- reduction ---
    "GramSchmidt",
    "GramSchmidtResult",
    "LLLReduction",
    "BKZReduction",
    "HermiteNormalForm",
    "ShortVectorProblem",
    "ClosestVectorProblem",
    # --- enumeration ---
    "PruningType",
    "PruningBounds",
    "EnumPruningStrategy",
    "PointEnumerator",
    "KannanEnumeration",
    "FinckePohlstEnumeration",
    "BoundedEnumeration",
    "EnumerationNode",
    "EnumerationTree",
    # --- branch_and_bound ---
    "SelectionStrategy",
    "BranchingHeuristic",
    "BBNode",
    "BBResult",
    "BoundComputation",
    "BranchingStrategy",
    "NodeSelection",
    "SymmetryBreaking",
    "CuttingPlanes",
    "BBSolver",
    # --- mechanism_lattice ---
    "FeasibilityChecker",
    "DominanceRelation",
    "MechanismParetoFront",
    "LatticeProjection",
    "MechLattice",
    "DiscreteOptimizer",
]
