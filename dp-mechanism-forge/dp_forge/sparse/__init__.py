"""
Sparse mechanism design via column generation, Benders decomposition,
and Lagrangian relaxation.

This package addresses the scalability challenge of DP mechanism synthesis
when the number of output discretisation bins k or the number of inputs n
is large. Instead of constructing the full n·k LP, sparse methods
iteratively identify and add only the columns (variables) or constraints
that are needed for optimality.

Architecture:
    1. **ColumnGenerator** — Generates mechanism columns (output distributions)
       on demand by solving pricing subproblems. Only columns with negative
       reduced cost are added to the restricted master problem.
    2. **BendersDecomposer** — Decomposes the mechanism LP into a master
       problem (mechanism structure) and subproblems (per-pair privacy checks).
       Adds optimality/feasibility cuts from subproblem duals.
    3. **LagrangianRelaxer** — Relaxes privacy constraints into the objective
       with Lagrange multipliers. Iteratively updates multipliers via
       subgradient or bundle methods.
    4. **SparseLP** — Sparse LP formulation that tracks active columns and
       supports dynamic column/constraint management.
    5. **PricingSolver** — Solves the pricing subproblem for column generation.

Example::

    from dp_forge.sparse import ColumnGenerator, SparseConfig

    config = SparseConfig(pricing_strategy="exact", max_columns=1000)
    cg = ColumnGenerator(config=config)
    result = cg.solve(query_spec)
    print(f"Optimal with {result.active_columns} columns out of {result.total_possible}")
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
from scipy import sparse

from dp_forge.types import (
    AdjacencyRelation,
    GameMatrix,
    LPStruct,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
    SolverBackend,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PricingStrategy(Enum):
    """Strategy for solving the column generation pricing subproblem."""

    EXACT = auto()
    HEURISTIC = auto()
    HYBRID = auto()

    def __repr__(self) -> str:
        return f"PricingStrategy.{self.name}"


class DecompositionType(Enum):
    """Type of decomposition used for large-scale mechanism design."""

    COLUMN_GENERATION = auto()
    BENDERS = auto()
    LAGRANGIAN = auto()
    DANTZIG_WOLFE = auto()

    def __repr__(self) -> str:
        return f"DecompositionType.{self.name}"


class CutType(Enum):
    """Types of cuts added during Benders decomposition."""

    FEASIBILITY = auto()
    OPTIMALITY = auto()

    def __repr__(self) -> str:
        return f"CutType.{self.name}"


class MultiplierUpdate(Enum):
    """Methods for updating Lagrange multipliers."""

    SUBGRADIENT = auto()
    BUNDLE = auto()
    CUTTING_PLANE = auto()
    VOLUME = auto()

    def __repr__(self) -> str:
        return f"MultiplierUpdate.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SparseConfig:
    """Configuration for sparse mechanism design methods.

    Attributes:
        decomposition_type: Decomposition method to use.
        pricing_strategy: Strategy for column generation pricing.
        max_columns: Maximum number of columns in the restricted master.
        max_cuts: Maximum number of Benders cuts.
        max_iterations: Maximum number of outer iterations.
        multiplier_update: Method for Lagrangian multiplier updates.
        step_size_init: Initial step size for subgradient methods.
        convergence_tol: Convergence tolerance on the duality gap.
        solver: LP solver backend.
        verbose: Verbosity level.
    """

    decomposition_type: DecompositionType = DecompositionType.COLUMN_GENERATION
    pricing_strategy: PricingStrategy = PricingStrategy.EXACT
    max_columns: int = 5000
    max_cuts: int = 1000
    max_iterations: int = 500
    multiplier_update: MultiplierUpdate = MultiplierUpdate.SUBGRADIENT
    step_size_init: float = 1.0
    convergence_tol: float = 1e-6
    solver: SolverBackend = SolverBackend.AUTO
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.max_columns < 1:
            raise ValueError(f"max_columns must be >= 1, got {self.max_columns}")
        if self.max_cuts < 1:
            raise ValueError(f"max_cuts must be >= 1, got {self.max_cuts}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.step_size_init <= 0:
            raise ValueError(f"step_size_init must be > 0, got {self.step_size_init}")
        if self.convergence_tol <= 0:
            raise ValueError(f"convergence_tol must be > 0, got {self.convergence_tol}")

    def __repr__(self) -> str:
        return (
            f"SparseConfig(type={self.decomposition_type.name}, "
            f"pricing={self.pricing_strategy.name}, max_cols={self.max_columns})"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class Column:
    """A column (output distribution) in the restricted master problem.

    Attributes:
        index: Column index in the master problem.
        distribution: Probability distribution over output bins for this column.
        reduced_cost: Reduced cost from the pricing subproblem.
        source_input: Which input database this column corresponds to.
    """

    index: int
    distribution: npt.NDArray[np.float64]
    reduced_cost: float
    source_input: int

    def __post_init__(self) -> None:
        self.distribution = np.asarray(self.distribution, dtype=np.float64)
        if np.any(self.distribution < -1e-12):
            raise ValueError("Column distribution contains negative probabilities")

    def __repr__(self) -> str:
        return (
            f"Column(idx={self.index}, input={self.source_input}, "
            f"rc={self.reduced_cost:.2e})"
        )


@dataclass
class BendersCut:
    """A Benders cut (optimality or feasibility) for the master problem.

    Attributes:
        cut_type: Whether this is a feasibility or optimality cut.
        coefficients: Cut coefficients in the master variable space.
        rhs: Right-hand side of the cut.
        subproblem_pair: The adjacent pair (i, i') that generated this cut.
    """

    cut_type: CutType
    coefficients: npt.NDArray[np.float64]
    rhs: float
    subproblem_pair: Tuple[int, int]

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"BendersCut(type={self.cut_type.name}, pair={self.subproblem_pair}, "
            f"rhs={self.rhs:.4f})"
        )


@dataclass
class LagrangianState:
    """Current state of the Lagrangian relaxation.

    Attributes:
        multipliers: Current Lagrange multiplier values.
        lower_bound: Current Lagrangian lower bound.
        upper_bound: Current best feasible upper bound.
        step_size: Current step size for multiplier updates.
        iteration: Current iteration number.
        subgradient: Most recent subgradient.
    """

    multipliers: npt.NDArray[np.float64]
    lower_bound: float
    upper_bound: float
    step_size: float
    iteration: int = 0
    subgradient: Optional[npt.NDArray[np.float64]] = None

    @property
    def gap(self) -> float:
        """Relative duality gap."""
        if abs(self.upper_bound) < 1e-12:
            return abs(self.upper_bound - self.lower_bound)
        return (self.upper_bound - self.lower_bound) / max(abs(self.upper_bound), 1.0)

    def __repr__(self) -> str:
        return (
            f"LagrangianState(iter={self.iteration}, LB={self.lower_bound:.6f}, "
            f"UB={self.upper_bound:.6f}, gap={self.gap:.2e})"
        )


@dataclass
class SparseResult:
    """Result of sparse mechanism synthesis.

    Attributes:
        mechanism: The n × k probability table.
        active_columns: Number of active columns used.
        total_possible: Total number of possible columns.
        iterations: Number of outer iterations.
        obj_val: Final objective value.
        lower_bound: Lower bound on the optimal objective.
        upper_bound: Upper bound on the optimal objective.
        optimality_certificate: Duality-based certificate.
        convergence_history: Objective value at each iteration.
    """

    mechanism: npt.NDArray[np.float64]
    active_columns: int
    total_possible: int
    iterations: int
    obj_val: float
    lower_bound: float
    upper_bound: float
    optimality_certificate: Optional[OptimalityCertificate] = None
    convergence_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mechanism = np.asarray(self.mechanism, dtype=np.float64)
        if self.mechanism.ndim != 2:
            raise ValueError(
                f"mechanism must be 2-D, got shape {self.mechanism.shape}"
            )

    @property
    def compression_ratio(self) -> float:
        """Ratio of active columns to total possible columns."""
        if self.total_possible == 0:
            return 0.0
        return self.active_columns / self.total_possible

    @property
    def gap(self) -> float:
        """Relative optimality gap."""
        return (self.upper_bound - self.lower_bound) / max(abs(self.upper_bound), 1.0)

    def __repr__(self) -> str:
        return (
            f"SparseResult(cols={self.active_columns}/{self.total_possible}, "
            f"iter={self.iterations}, obj={self.obj_val:.6f}, gap={self.gap:.2e})"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class PricingSolver(Protocol):
    """Protocol for solving column generation pricing subproblems."""

    def solve(
        self,
        dual_values: npt.NDArray[np.float64],
        input_index: int,
        budget: PrivacyBudget,
    ) -> Optional[Column]:
        """Solve the pricing subproblem for a given input.

        Returns a column with negative reduced cost, or None if no
        improving column exists (optimality of the restricted master).
        """
        ...


@runtime_checkable
class SubproblemSolver(Protocol):
    """Protocol for solving Benders subproblems."""

    def solve(
        self,
        master_solution: npt.NDArray[np.float64],
        adjacent_pair: Tuple[int, int],
        budget: PrivacyBudget,
    ) -> Tuple[bool, Optional[BendersCut]]:
        """Solve the subproblem for an adjacent pair.

        Returns:
            Tuple of (feasible, cut_or_none).
        """
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class ColumnGenerator:
    """Column generation solver for large-scale DP mechanism synthesis.

    Iteratively adds columns to a restricted master problem by solving
    pricing subproblems. Terminates when no column with negative reduced
    cost exists.
    """

    def __init__(self, config: Optional[SparseConfig] = None) -> None:
        self.config = config or SparseConfig(decomposition_type=DecompositionType.COLUMN_GENERATION)

    def solve(self, spec: QuerySpec) -> SparseResult:
        """Synthesise a mechanism via column generation.

        Args:
            spec: Query specification.

        Returns:
            SparseResult with the optimised mechanism.
        """
        from dp_forge.sparse.column_generation import DantzigWolfeDecomposition

        dw = DantzigWolfeDecomposition(spec, self.config)
        return dw.solve()

    def solve_with_initial_columns(
        self,
        spec: QuerySpec,
        initial_columns: List[Column],
    ) -> SparseResult:
        """Solve with user-provided initial columns.

        Args:
            spec: Query specification.
            initial_columns: Initial columns for warm-starting.

        Returns:
            SparseResult with the optimised mechanism.
        """
        from dp_forge.sparse.column_generation import (
            ColumnPool,
            DantzigWolfeDecomposition,
        )

        dw = DantzigWolfeDecomposition(spec, self.config)
        for col in initial_columns:
            dw.pool.add(col.distribution, col.source_input, col.reduced_cost)
        return dw.solve()


class BendersDecomposer:
    """Benders decomposition for DP mechanism synthesis.

    Decomposes the mechanism LP into a master (mechanism structure) and
    per-pair subproblems (privacy verification). Adds cuts from subproblem
    duals to the master.
    """

    def __init__(self, config: Optional[SparseConfig] = None) -> None:
        self.config = config or SparseConfig(decomposition_type=DecompositionType.BENDERS)

    def solve(self, spec: QuerySpec) -> SparseResult:
        """Synthesise a mechanism via Benders decomposition.

        Args:
            spec: Query specification.

        Returns:
            SparseResult with the optimised mechanism.
        """
        from dp_forge.sparse.benders import MultiCutBenders

        benders = MultiCutBenders(spec, self.config)
        return benders.solve()


class LagrangianRelaxer:
    """Lagrangian relaxation for DP mechanism synthesis.

    Relaxes DP constraints into the objective with multipliers and
    iteratively tightens the bound via subgradient or bundle methods.
    """

    def __init__(self, config: Optional[SparseConfig] = None) -> None:
        self.config = config or SparseConfig(decomposition_type=DecompositionType.LAGRANGIAN)
        self._state: Optional[LagrangianState] = None

    def solve(self, spec: QuerySpec) -> SparseResult:
        """Synthesise a mechanism via Lagrangian relaxation.

        Args:
            spec: Query specification.

        Returns:
            SparseResult with the optimised mechanism.
        """
        from dp_forge.sparse.lagrangian import (
            LagrangianRelaxation as LagRelax,
            SubgradientOptimizer,
        )

        relaxation = LagRelax(spec)
        optimizer = SubgradientOptimizer(relaxation, self.config)
        state, mechanism = optimizer.solve()
        self._state = state
        n, k = spec.n, spec.k
        return SparseResult(
            mechanism=mechanism,
            active_columns=int(np.count_nonzero(mechanism > 1e-8)),
            total_possible=n * k,
            iterations=state.iteration,
            obj_val=state.upper_bound,
            lower_bound=state.lower_bound,
            upper_bound=state.upper_bound,
        )

    def get_state(self) -> Optional[LagrangianState]:
        """Return the current Lagrangian state (multipliers, bounds)."""
        return self._state


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def sparse_synthesize(
    spec: QuerySpec,
    *,
    method: DecompositionType = DecompositionType.COLUMN_GENERATION,
    config: Optional[SparseConfig] = None,
) -> SparseResult:
    """Convenience function for sparse mechanism synthesis.

    Selects the appropriate solver based on the decomposition type.

    Args:
        spec: Query specification.
        method: Decomposition method.
        config: Optional configuration.

    Returns:
        SparseResult with the optimised mechanism.
    """
    if method == DecompositionType.COLUMN_GENERATION:
        solver = ColumnGenerator(config)
        return solver.solve(spec)
    elif method == DecompositionType.BENDERS:
        solver_b = BendersDecomposer(config)
        return solver_b.solve(spec)
    elif method == DecompositionType.LAGRANGIAN:
        solver_l = LagrangianRelaxer(config)
        return solver_l.solve(spec)
    elif method == DecompositionType.DANTZIG_WOLFE:
        solver_dw = ColumnGenerator(config)
        return solver_dw.solve(spec)
    else:
        raise ValueError(f"Unknown decomposition type: {method}")


def estimate_sparsity(spec: QuerySpec) -> float:
    """Estimate the sparsity of the optimal mechanism.

    Heuristic estimate of what fraction of mechanism columns will be
    nonzero in the optimal solution. Useful for deciding whether sparse
    methods will be beneficial.

    Args:
        spec: Query specification.

    Returns:
        Estimated sparsity ratio in [0, 1].
    """
    # Heuristic: higher epsilon => sparser mechanism (more peaked distributions)
    # Larger k => sparser (fewer bins needed relative to total)
    eps_factor = 1.0 / (1.0 + spec.epsilon)
    k_factor = min(1.0, 2.0 * spec.n / spec.k) if spec.k > 0 else 1.0
    return float(np.clip(eps_factor * k_factor, 0.0, 1.0))


__all__ = [
    # Enums
    "PricingStrategy",
    "DecompositionType",
    "CutType",
    "MultiplierUpdate",
    # Config
    "SparseConfig",
    # Data types
    "Column",
    "BendersCut",
    "LagrangianState",
    "SparseResult",
    # Protocols
    "PricingSolver",
    "SubproblemSolver",
    # Classes
    "ColumnGenerator",
    "BendersDecomposer",
    "LagrangianRelaxer",
    # Functions
    "sparse_synthesize",
    "estimate_sparsity",
]
