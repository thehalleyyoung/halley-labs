"""
ProjectedCEGIS engine for multi-dimensional DP mechanism synthesis.

This module orchestrates the full multi-dimensional mechanism design
pipeline: decompose a d-dimensional query into coordinate-separable
marginals, allocate per-coordinate privacy budgets, synthesise each
marginal mechanism independently via the existing CEGISEngine, and
assemble the results via tensor product.

Algorithm Overview:
    1. **Separability detection**: Analyse the query matrix to determine
       whether it admits a Kronecker decomposition.
    2. **Budget allocation**: Divide the total (ε, δ) budget across
       coordinates using uniform, proportional, or optimised allocation.
    3. **Per-coordinate synthesis**: Run CEGISEngine independently for each
       coordinate's sub-problem.
    4. **Tensor product assembly**: Combine marginal mechanisms into the
       full joint mechanism.
    5. **Lower bound analysis**: Compare achieved error to information-
       theoretic lower bounds.

For d = 1, the engine reduces to a direct call to CEGISEngine (no
decomposition overhead).  For non-separable queries, the engine falls
back to a direct LP over the full joint space.

Classes:
    MultiDimQuerySpec       — specification for a d-dimensional query
    MultiDimMechanism       — result of multi-dimensional synthesis
    ProjectedCEGIS          — main synthesis engine
    ProjectedCEGISConfig    — configuration for the engine
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.cegis_loop import CEGISEngine
from dp_forge.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    ConvergenceError,
    InfeasibleSpecError,
)
from dp_forge.types import (
    AdjacencyRelation,
    CEGISResult,
    CompositionType,
    ExtractedMechanism,
    LossFunction,
    NumericalConfig,
    PrivacyBudget,
    QuerySpec,
    QueryType,
    SynthesisConfig,
)

from dp_forge.multidim.budget_allocation import (
    AllocationStrategy,
    BudgetAllocation,
    BudgetAllocator,
)
from dp_forge.multidim.lower_bounds import LowerBoundComputer, LowerBoundResult
from dp_forge.multidim.marginal_queries import MarginalQuery, MarginalQueryBuilder
from dp_forge.multidim.separability_detector import (
    SeparabilityDetector,
    SeparabilityResult,
    SeparabilityType,
)
from dp_forge.multidim.tensor_product import (
    MarginalMechanism,
    TensorProductMechanism,
    build_product_mechanism,
)

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Strategy when query is non-separable."""

    DIRECT_LP = auto()
    APPROXIMATE_SEPARABLE = auto()
    ERROR = auto()

    def __repr__(self) -> str:
        return f"FallbackStrategy.{self.name}"


@dataclass
class ProjectedCEGISConfig:
    """Configuration for the ProjectedCEGIS engine.

    Attributes:
        synthesis_config: Configuration for per-coordinate CEGISEngine.
        allocation_strategy: Budget allocation strategy.
        composition_type: Composition theorem for budget accounting.
        fallback: Strategy for non-separable queries.
        separability_tol: Tolerance for separability detection.
        compute_lower_bounds: Whether to compute lower bounds.
        max_direct_lp_size: Maximum product domain size for direct LP
            fallback.
        verbose: Verbosity level.
    """

    synthesis_config: SynthesisConfig = field(default_factory=SynthesisConfig)
    allocation_strategy: AllocationStrategy = AllocationStrategy.UNIFORM
    composition_type: CompositionType = CompositionType.BASIC
    fallback: FallbackStrategy = FallbackStrategy.DIRECT_LP
    separability_tol: float = 1e-10
    compute_lower_bounds: bool = True
    max_direct_lp_size: int = 10_000
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.separability_tol <= 0:
            raise ValueError(
                f"separability_tol must be > 0, got {self.separability_tol}"
            )
        if self.max_direct_lp_size < 1:
            raise ValueError(
                f"max_direct_lp_size must be >= 1, got {self.max_direct_lp_size}"
            )
        if self.verbose not in (0, 1, 2):
            raise ValueError(
                f"verbose must be 0, 1, or 2, got {self.verbose}"
            )


@dataclass
class MultiDimQuerySpec:
    """Specification for a d-dimensional query.

    Attributes:
        query_matrix: Query matrix A of shape (m, d), or None for
            trivially separable (identity) queries.
        query_values_per_coord: Per-coordinate query values. Each
            element is a 1-D array of distinct query outputs for
            that coordinate.
        sensitivities: Per-coordinate sensitivities.
        epsilon: Total privacy parameter ε.
        delta: Total privacy parameter δ.
        k_per_coord: Output discretisation bins per coordinate.
        loss_fn: Loss function.
        adjacencies: Per-coordinate adjacency relations (optional).
        metadata: Extra metadata.
    """

    query_values_per_coord: List[npt.NDArray[np.float64]]
    sensitivities: List[float]
    epsilon: float
    delta: float = 0.0
    query_matrix: Optional[npt.NDArray[np.float64]] = None
    k_per_coord: Optional[List[int]] = None
    loss_fn: LossFunction = LossFunction.L2
    adjacencies: Optional[List[AdjacencyRelation]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        d = len(self.query_values_per_coord)
        if d < 1:
            raise ValueError("query_values_per_coord must be non-empty")
        for i, qv in enumerate(self.query_values_per_coord):
            qv = np.asarray(qv, dtype=np.float64)
            if qv.ndim != 1:
                raise ValueError(
                    f"query_values_per_coord[{i}] must be 1-D, got shape {qv.shape}"
                )
            self.query_values_per_coord[i] = qv
        if len(self.sensitivities) != d:
            raise ValueError(
                f"sensitivities length ({len(self.sensitivities)}) must match "
                f"d ({d})"
            )
        if any(s <= 0 for s in self.sensitivities):
            raise ValueError("All sensitivities must be > 0")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")
        if self.k_per_coord is None:
            self.k_per_coord = [100] * d
        if len(self.k_per_coord) != d:
            raise ValueError(
                f"k_per_coord length ({len(self.k_per_coord)}) must match d ({d})"
            )
        if self.adjacencies is not None and len(self.adjacencies) != d:
            raise ValueError(
                f"adjacencies length ({len(self.adjacencies)}) must match d ({d})"
            )

    @property
    def d(self) -> int:
        """Number of dimensions."""
        return len(self.query_values_per_coord)

    @property
    def n_per_coord(self) -> List[int]:
        """Number of distinct query values per coordinate."""
        return [len(qv) for qv in self.query_values_per_coord]

    @classmethod
    def counting(
        cls,
        d: int,
        n_per_coord: int,
        epsilon: float,
        delta: float = 0.0,
        k: int = 100,
    ) -> MultiDimQuerySpec:
        """Factory for d-dimensional counting queries with sensitivity 1."""
        return cls(
            query_values_per_coord=[
                np.arange(n_per_coord, dtype=np.float64) for _ in range(d)
            ],
            sensitivities=[1.0] * d,
            epsilon=epsilon,
            delta=delta,
            k_per_coord=[k] * d,
        )

    def to_single_coord_spec(
        self, dim: int, epsilon: float, delta: float = 0.0
    ) -> QuerySpec:
        """Build a QuerySpec for a single coordinate.

        Args:
            dim: Coordinate index.
            epsilon: Per-coordinate privacy parameter.
            delta: Per-coordinate delta.

        Returns:
            QuerySpec for coordinate dim.
        """
        assert self.k_per_coord is not None
        edges = None
        if self.adjacencies is not None:
            edges = self.adjacencies[dim]
        return QuerySpec(
            query_values=self.query_values_per_coord[dim],
            domain=f"coord_{dim}",
            sensitivity=self.sensitivities[dim],
            epsilon=epsilon,
            delta=delta,
            k=self.k_per_coord[dim],
            loss_fn=self.loss_fn,
            edges=edges,
            query_type=QueryType.COUNTING,
        )

    def __repr__(self) -> str:
        dp = f"ε={self.epsilon}" + (f", δ={self.delta}" if self.delta > 0 else "")
        return (
            f"MultiDimQuerySpec(d={self.d}, n_per_coord={self.n_per_coord}, "
            f"{dp})"
        )


@dataclass
class MultiDimMechanism:
    """Result of multi-dimensional mechanism synthesis.

    Attributes:
        product_mechanism: The tensor product mechanism.
        marginal_results: Per-coordinate CEGIS results.
        budget_allocation: How the privacy budget was divided.
        separability: Separability analysis result (if query matrix given).
        lower_bound: Information-theoretic lower bound (if computed).
        total_error: Total expected error across coordinates.
        per_coord_errors: Per-coordinate expected errors.
        synthesis_time: Total wall-clock synthesis time.
        metadata: Additional result metadata.
    """

    product_mechanism: TensorProductMechanism
    marginal_results: List[CEGISResult]
    budget_allocation: BudgetAllocation
    separability: Optional[SeparabilityResult] = None
    lower_bound: Optional[LowerBoundResult] = None
    total_error: float = 0.0
    per_coord_errors: Optional[npt.NDArray[np.float64]] = None
    synthesis_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> int:
        """Number of dimensions."""
        return self.product_mechanism.d

    @property
    def total_iterations(self) -> int:
        """Total CEGIS iterations across all coordinates."""
        return sum(r.iterations for r in self.marginal_results)

    @property
    def gap_ratio(self) -> Optional[float]:
        """Optimality gap ratio, if lower bound was computed."""
        if self.lower_bound is not None and self.lower_bound.gap_ratio is not None:
            return self.lower_bound.gap_ratio
        return None

    def __repr__(self) -> str:
        gap = f", gap={self.gap_ratio:.2f}" if self.gap_ratio is not None else ""
        return (
            f"MultiDimMechanism(d={self.d}, error={self.total_error:.6f}, "
            f"iters={self.total_iterations}, time={self.synthesis_time:.2f}s{gap})"
        )


class ProjectedCEGIS:
    """Multi-dimensional CEGIS engine via coordinate projection.

    Decomposes d-dimensional queries into coordinate-separable marginals,
    synthesises each independently, and assembles via tensor product.

    For d = 1, directly delegates to CEGISEngine with no overhead.
    For non-separable queries, falls back to direct LP (or errors,
    depending on configuration).

    Args:
        config: ProjectedCEGIS configuration.

    Example::

        engine = ProjectedCEGIS()
        spec = MultiDimQuerySpec.counting(d=3, n_per_coord=5, epsilon=1.0)
        result = engine.synthesize(spec)
        print(result.total_error)
    """

    def __init__(
        self, config: Optional[ProjectedCEGISConfig] = None
    ) -> None:
        self._config = config or ProjectedCEGISConfig()
        self._allocator = BudgetAllocator(
            composition_type=self._config.composition_type,
        )
        self._detector = SeparabilityDetector(
            tol=self._config.separability_tol,
        )
        self._lb_computer = LowerBoundComputer(loss_type="L2")

    def synthesize(
        self,
        spec: MultiDimQuerySpec,
        callback: Optional[Callable[[int, CEGISResult], None]] = None,
    ) -> MultiDimMechanism:
        """Synthesise an optimal multi-dimensional DP mechanism.

        Main entry point. Orchestrates the full pipeline:
        detect separability → allocate budgets → synthesise marginals →
        assemble product → compute bounds.

        Args:
            spec: Multi-dimensional query specification.
            callback: Optional callback invoked after each coordinate's
                synthesis completes.  Receives (dim, result).

        Returns:
            MultiDimMechanism with the assembled product mechanism.

        Raises:
            InfeasibleSpecError: If any coordinate's LP is infeasible.
            ConvergenceError: If any coordinate fails to converge.
            ConfigurationError: For invalid specifications.
        """
        t_start = time.monotonic()
        d = spec.d
        if self._config.verbose >= 1:
            logger.info(
                "ProjectedCEGIS: d=%d, ε=%.4f, δ=%.4e",
                d, spec.epsilon, spec.delta,
            )
        # d=1: direct synthesis, no decomposition
        if d == 1:
            return self._synthesize_single_dim(spec, callback, t_start)
        # Step 1: Separability detection (if query matrix given)
        sep_result = None
        if spec.query_matrix is not None:
            sep_result = self._detector.detect(spec.query_matrix)
            if self._config.verbose >= 1:
                logger.info(
                    "Separability: %s", sep_result.sep_type.name
                )
            if sep_result.sep_type == SeparabilityType.NON_SEPARABLE:
                return self._handle_non_separable(
                    spec, sep_result, callback, t_start
                )
        # Step 2: Allocate budgets
        allocation = self._allocate_budget(spec)
        if self._config.verbose >= 1:
            logger.info(
                "Budget allocation (%s): ε = %s",
                allocation.strategy.name,
                np.array2string(allocation.epsilons, precision=4),
            )
        # Step 3: Synthesise each coordinate
        marginal_results, marginal_mechanisms = self._synthesize_marginals(
            spec, allocation, callback
        )
        # Step 4: Assemble tensor product
        product = TensorProductMechanism(
            marginals=marginal_mechanisms,
            total_epsilon=spec.epsilon,
            total_delta=spec.delta,
        )
        # Step 5: Compute per-coordinate errors
        per_coord_errors = np.array(
            [r.obj_val for r in marginal_results], dtype=np.float64
        )
        total_error = float(per_coord_errors.sum())
        # Step 6: Lower bounds
        lb_result = None
        if self._config.compute_lower_bounds:
            lb_result = self._lb_computer.gap_analysis(
                achieved_error=total_error,
                d=d,
                epsilon=spec.epsilon,
                delta=spec.delta,
                sensitivity=max(spec.sensitivities),
            )
            if self._config.verbose >= 1:
                logger.info(
                    "Lower bound: %.6f, gap ratio: %.2f",
                    lb_result.bound_value,
                    lb_result.gap_ratio if lb_result.gap_ratio else float("inf"),
                )
        synthesis_time = time.monotonic() - t_start
        return MultiDimMechanism(
            product_mechanism=product,
            marginal_results=marginal_results,
            budget_allocation=allocation,
            separability=sep_result,
            lower_bound=lb_result,
            total_error=total_error,
            per_coord_errors=per_coord_errors,
            synthesis_time=synthesis_time,
            metadata={
                "d": d,
                "n_per_coord": spec.n_per_coord,
                "strategy": self._config.allocation_strategy.name,
            },
        )

    def _synthesize_single_dim(
        self,
        spec: MultiDimQuerySpec,
        callback: Optional[Callable[[int, CEGISResult], None]],
        t_start: float,
    ) -> MultiDimMechanism:
        """Handle the d=1 case: direct delegation to CEGISEngine."""
        engine = CEGISEngine(self._config.synthesis_config)
        coord_spec = spec.to_single_coord_spec(0, spec.epsilon, spec.delta)
        result = engine.synthesize(coord_spec)
        if callback is not None:
            callback(0, result)
        assert spec.k_per_coord is not None
        y_grid = np.linspace(
            float(coord_spec.query_values.min()),
            float(coord_spec.query_values.max()),
            spec.k_per_coord[0],
        )
        marginal = MarginalMechanism(
            p_table=result.mechanism,
            y_grid=y_grid,
            coordinate_index=0,
            epsilon=spec.epsilon,
            sensitivity=spec.sensitivities[0],
        )
        product = TensorProductMechanism(
            marginals=[marginal],
            total_epsilon=spec.epsilon,
            total_delta=spec.delta,
        )
        allocation = BudgetAllocation(
            epsilons=np.array([spec.epsilon]),
            deltas=np.array([spec.delta]),
            strategy=AllocationStrategy.UNIFORM,
            total_epsilon=spec.epsilon,
            total_delta=spec.delta,
        )
        lb_result = None
        if self._config.compute_lower_bounds:
            lb_result = self._lb_computer.gap_analysis(
                achieved_error=result.obj_val,
                d=1,
                epsilon=spec.epsilon,
                delta=spec.delta,
                sensitivity=spec.sensitivities[0],
            )
        synthesis_time = time.monotonic() - t_start
        return MultiDimMechanism(
            product_mechanism=product,
            marginal_results=[result],
            budget_allocation=allocation,
            lower_bound=lb_result,
            total_error=result.obj_val,
            per_coord_errors=np.array([result.obj_val]),
            synthesis_time=synthesis_time,
            metadata={"d": 1, "direct_synthesis": True},
        )

    def _handle_non_separable(
        self,
        spec: MultiDimQuerySpec,
        sep_result: SeparabilityResult,
        callback: Optional[Callable[[int, CEGISResult], None]],
        t_start: float,
    ) -> MultiDimMechanism:
        """Handle non-separable queries via fallback strategy.

        For DIRECT_LP: build a single QuerySpec over the joint domain
        and synthesise directly (only feasible for small domains).
        For ERROR: raise ConfigurationError.

        Args:
            spec: Multi-dimensional query specification.
            sep_result: Separability analysis result.
            callback: Optional progress callback.
            t_start: Synthesis start time.

        Returns:
            MultiDimMechanism from direct LP synthesis.

        Raises:
            ConfigurationError: If fallback is ERROR or domain too large.
        """
        if self._config.fallback == FallbackStrategy.ERROR:
            raise ConfigurationError(
                "Query is non-separable and fallback=ERROR",
                parameter="query_matrix",
            )
        # Compute joint domain size
        joint_n = math.prod(len(qv) for qv in spec.query_values_per_coord)
        if joint_n > self._config.max_direct_lp_size:
            if self._config.fallback == FallbackStrategy.APPROXIMATE_SEPARABLE:
                logger.warning(
                    "Non-separable query with joint domain %d > max %d. "
                    "Treating as approximately separable.",
                    joint_n, self._config.max_direct_lp_size,
                )
                allocation = self._allocate_budget(spec)
                marginal_results, marginal_mechanisms = self._synthesize_marginals(
                    spec, allocation, callback
                )
                product = TensorProductMechanism(
                    marginals=marginal_mechanisms,
                    total_epsilon=spec.epsilon,
                    total_delta=spec.delta,
                )
                per_coord_errors = np.array(
                    [r.obj_val for r in marginal_results], dtype=np.float64
                )
                synthesis_time = time.monotonic() - t_start
                return MultiDimMechanism(
                    product_mechanism=product,
                    marginal_results=marginal_results,
                    budget_allocation=allocation,
                    separability=sep_result,
                    total_error=float(per_coord_errors.sum()),
                    per_coord_errors=per_coord_errors,
                    synthesis_time=synthesis_time,
                    metadata={"approximate_separable": True},
                )
            raise ConfigurationError(
                f"Non-separable query with joint domain size {joint_n} "
                f"exceeds max_direct_lp_size {self._config.max_direct_lp_size}. "
                f"Cannot fall back to direct LP.",
                parameter="max_direct_lp_size",
                value=self._config.max_direct_lp_size,
            )
        if self._config.verbose >= 1:
            logger.info(
                "Non-separable: falling back to direct LP (joint_n=%d)", joint_n
            )
        # Build joint QuerySpec
        joint_values = self._build_joint_query_values(spec)
        assert spec.k_per_coord is not None
        joint_k = min(math.prod(spec.k_per_coord), 500)
        max_sensitivity = max(spec.sensitivities)
        joint_spec = QuerySpec(
            query_values=joint_values,
            domain="joint_multidim",
            sensitivity=max_sensitivity,
            epsilon=spec.epsilon,
            delta=spec.delta,
            k=joint_k,
            loss_fn=spec.loss_fn,
        )
        engine = CEGISEngine(self._config.synthesis_config)
        result = engine.synthesize(joint_spec)
        if callback is not None:
            callback(0, result)
        # Wrap as a single marginal for uniform interface
        y_grid = np.linspace(
            float(joint_values.min()), float(joint_values.max()), joint_k
        )
        marginal = MarginalMechanism(
            p_table=result.mechanism,
            y_grid=y_grid,
            coordinate_index=0,
            epsilon=spec.epsilon,
            sensitivity=max_sensitivity,
        )
        product = TensorProductMechanism(
            marginals=[marginal],
            total_epsilon=spec.epsilon,
            total_delta=spec.delta,
        )
        allocation = BudgetAllocation(
            epsilons=np.array([spec.epsilon]),
            deltas=np.array([spec.delta]),
            strategy=AllocationStrategy.UNIFORM,
            total_epsilon=spec.epsilon,
            total_delta=spec.delta,
        )
        synthesis_time = time.monotonic() - t_start
        return MultiDimMechanism(
            product_mechanism=product,
            marginal_results=[result],
            budget_allocation=allocation,
            separability=sep_result,
            total_error=result.obj_val,
            per_coord_errors=np.array([result.obj_val]),
            synthesis_time=synthesis_time,
            metadata={"direct_lp": True, "joint_n": joint_n},
        )

    def _allocate_budget(
        self, spec: MultiDimQuerySpec
    ) -> BudgetAllocation:
        """Allocate privacy budget across coordinates.

        Uses the configured allocation strategy.

        Args:
            spec: Multi-dimensional query specification.

        Returns:
            BudgetAllocation with per-coordinate budgets.
        """
        total_budget = PrivacyBudget(
            epsilon=spec.epsilon,
            delta=spec.delta,
            composition_type=self._config.composition_type,
        )
        d = spec.d
        strategy = self._config.allocation_strategy
        if strategy == AllocationStrategy.UNIFORM:
            return self._allocator.allocate_uniform(total_budget, d)
        elif strategy == AllocationStrategy.PROPORTIONAL:
            return self._allocator.allocate_proportional(
                total_budget, spec.sensitivities
            )
        elif strategy == AllocationStrategy.OPTIMAL:
            assert spec.k_per_coord is not None
            return self._allocator.allocate_optimal(
                total_budget,
                spec.sensitivities,
                domain_sizes=[len(qv) for qv in spec.query_values_per_coord],
            )
        raise ConfigurationError(
            f"Unknown allocation strategy: {strategy}",
            parameter="allocation_strategy",
            value=strategy,
        )

    def _synthesize_marginals(
        self,
        spec: MultiDimQuerySpec,
        allocation: BudgetAllocation,
        callback: Optional[Callable[[int, CEGISResult], None]],
    ) -> Tuple[List[CEGISResult], List[MarginalMechanism]]:
        """Synthesise per-coordinate mechanisms via CEGISEngine.

        Args:
            spec: Multi-dimensional query specification.
            allocation: Per-coordinate budget allocation.
            callback: Optional progress callback.

        Returns:
            Tuple of (CEGIS results, marginal mechanisms).
        """
        assert spec.k_per_coord is not None
        d = spec.d
        marginal_results: List[CEGISResult] = []
        marginal_mechanisms: List[MarginalMechanism] = []
        for dim in range(d):
            eps_i = float(allocation.epsilons[dim])
            delta_i = float(allocation.deltas[dim])
            coord_spec = spec.to_single_coord_spec(dim, eps_i, delta_i)
            if self._config.verbose >= 1:
                logger.info(
                    "Synthesising coordinate %d/%d: n=%d, k=%d, ε=%.4f",
                    dim + 1, d, coord_spec.n, coord_spec.k, eps_i,
                )
            engine = CEGISEngine(self._config.synthesis_config)
            result = engine.synthesize(coord_spec)
            marginal_results.append(result)
            if callback is not None:
                callback(dim, result)
            y_grid = np.linspace(
                float(coord_spec.query_values.min()),
                float(coord_spec.query_values.max()),
                spec.k_per_coord[dim],
            )
            marginal = MarginalMechanism(
                p_table=result.mechanism,
                y_grid=y_grid,
                coordinate_index=dim,
                epsilon=eps_i,
                sensitivity=spec.sensitivities[dim],
            )
            marginal_mechanisms.append(marginal)
            if self._config.verbose >= 1:
                logger.info(
                    "Coordinate %d done: obj=%.6f, iters=%d",
                    dim, result.obj_val, result.iterations,
                )
        return marginal_results, marginal_mechanisms

    def _build_joint_query_values(
        self, spec: MultiDimQuerySpec
    ) -> npt.NDArray[np.float64]:
        """Build joint query values for direct LP synthesis.

        Creates the Cartesian product of per-coordinate query values
        and flattens to a 1-D array using row-major (C) ordering.

        Args:
            spec: Multi-dimensional query specification.

        Returns:
            1-D array of joint query values.
        """
        import itertools as it
        grids = [qv.tolist() for qv in spec.query_values_per_coord]
        joint = []
        for combo in it.product(*grids):
            joint.append(sum(combo))
        return np.array(joint, dtype=np.float64)

    def synthesize_from_marginals(
        self,
        marginal_queries: List[MarginalQuery],
        epsilon: float,
        delta: float = 0.0,
        domain_sizes: Optional[List[int]] = None,
        k: int = 100,
    ) -> MultiDimMechanism:
        """Synthesise mechanisms for a set of marginal queries.

        Convenience method that builds MultiDimQuerySpec from
        MarginalQuery objects and calls synthesize().

        Args:
            marginal_queries: List of marginal queries.
            epsilon: Total privacy budget.
            delta: Total delta.
            domain_sizes: Per-coordinate domain sizes.
            k: Output grid size per coordinate.

        Returns:
            MultiDimMechanism result.
        """
        d = len(marginal_queries)
        builder = MarginalQueryBuilder(
            d=d,
            domain_sizes=domain_sizes or [q.n_cells for q in marginal_queries],
        )
        query_values = []
        sensitivities = []
        for q in marginal_queries:
            query_values.append(np.arange(q.n_cells, dtype=np.float64))
            sensitivities.append(builder.compute_sensitivity(q, "L1"))
        spec = MultiDimQuerySpec(
            query_values_per_coord=query_values,
            sensitivities=sensitivities,
            epsilon=epsilon,
            delta=delta,
            k_per_coord=[k] * d,
        )
        return self.synthesize(spec)

    def estimate_error(
        self,
        spec: MultiDimQuerySpec,
        allocation: Optional[BudgetAllocation] = None,
    ) -> npt.NDArray[np.float64]:
        """Estimate per-coordinate errors without full synthesis.

        Uses the error model from the budget allocator to predict
        errors for each coordinate at the given budget allocation.

        Args:
            spec: Multi-dimensional query specification.
            allocation: Budget allocation. If None, uses default strategy.

        Returns:
            Array of estimated per-coordinate errors.
        """
        if allocation is None:
            allocation = self._allocate_budget(spec)
        errors = np.zeros(spec.d, dtype=np.float64)
        for i in range(spec.d):
            errors[i] = self._allocator._error_model(
                float(allocation.epsilons[i]),
                spec.sensitivities[i],
                len(spec.query_values_per_coord[i]),
            )
        return errors


# =========================================================================
# Progressive projection
# =========================================================================


def progressive_projection(
    spec: MultiDimQuerySpec,
    *,
    config: Optional[ProjectedCEGISConfig] = None,
    initial_d: int = 1,
    step_size: int = 1,
    callback: Optional[Callable[[int, float], None]] = None,
) -> MultiDimMechanism:
    """Progressively add coordinates, starting from a low-dimensional subproblem.

    For high-dimensional queries, directly synthesising all d coordinates
    may be expensive or produce poor budget allocations because the
    allocator lacks information about per-coordinate difficulty.

    This function starts with the first ``initial_d`` coordinates,
    synthesises mechanisms, measures the actual per-coordinate errors,
    and uses that information to refine budget allocation as new
    coordinates are added.

    Algorithm:
        1. Synthesise coordinates ``0 .. initial_d-1``.
        2. Measure per-coordinate error.
        3. Add the next ``step_size`` coordinates, re-allocate budget
           based on observed errors, and re-synthesise only the new
           coordinates (keeping the existing ones).
        4. Repeat until all d coordinates are handled.
        5. Assemble the final tensor product mechanism.

    This is particularly beneficial when coordinates have very different
    difficulty levels: easy coordinates consume less budget, leaving more
    for harder ones.

    Args:
        spec: Full d-dimensional query specification.
        config: ProjectedCEGIS configuration.
        initial_d: Number of coordinates in the first batch (≥ 1).
        step_size: Number of coordinates added per step (≥ 1).
        callback: Optional ``(current_d, total_error)`` callback.

    Returns:
        MultiDimMechanism with the progressively assembled product.

    Raises:
        ConfigurationError: If initial_d > spec.d or step_size < 1.
    """
    d = spec.d
    if initial_d < 1 or initial_d > d:
        raise ConfigurationError(
            f"initial_d must be in [1, {d}], got {initial_d}",
            parameter="initial_d",
            value=initial_d,
        )
    if step_size < 1:
        raise ConfigurationError(
            f"step_size must be >= 1, got {step_size}",
            parameter="step_size",
            value=step_size,
        )

    cfg = config or ProjectedCEGISConfig()
    t_start = time.monotonic()

    # Track per-coordinate results as we build up
    all_results: List[CEGISResult] = [None] * d  # type: ignore[list-item]
    all_marginals: List[MarginalMechanism] = [None] * d  # type: ignore[list-item]
    per_coord_errors = np.zeros(d, dtype=np.float64)

    # Process coordinates in batches
    processed = set()
    current_d = 0

    while current_d < d:
        batch_end = min(current_d + (initial_d if current_d == 0 else step_size), d)
        batch_dims = list(range(current_d, batch_end))
        n_active = batch_end  # total active coordinates so far

        # Allocate budget across all active coordinates
        # Give proportionally more to unsynthesised coordinates
        allocator = BudgetAllocator(
            composition_type=cfg.composition_type,
        )
        total_budget = PrivacyBudget(
            epsilon=spec.epsilon,
            delta=spec.delta,
            composition_type=cfg.composition_type,
        )
        allocation = allocator.allocate_uniform(total_budget, n_active)

        # Synthesise only the new batch
        engine = CEGISEngine(cfg.synthesis_config)
        assert spec.k_per_coord is not None
        for dim in batch_dims:
            eps_i = float(allocation.epsilons[dim])
            delta_i = float(allocation.deltas[dim])
            coord_spec = spec.to_single_coord_spec(dim, eps_i, delta_i)
            result = engine.synthesize(coord_spec)
            all_results[dim] = result
            per_coord_errors[dim] = result.obj_val

            y_grid = np.linspace(
                float(coord_spec.query_values.min()),
                float(coord_spec.query_values.max()),
                spec.k_per_coord[dim],
            )
            marginal = MarginalMechanism(
                p_table=result.mechanism,
                y_grid=y_grid,
                coordinate_index=dim,
                epsilon=eps_i,
                sensitivity=spec.sensitivities[dim],
            )
            all_marginals[dim] = marginal
            processed.add(dim)

        current_d = batch_end

        if callback is not None:
            callback(current_d, float(per_coord_errors[:current_d].sum()))

    # Assemble final product
    product = TensorProductMechanism(
        marginals=all_marginals,
        total_epsilon=spec.epsilon,
        total_delta=spec.delta,
    )

    allocation = BudgetAllocation(
        epsilons=np.array([spec.epsilon / d] * d),
        deltas=np.array([spec.delta / max(d, 1)] * d),
        strategy=AllocationStrategy.UNIFORM,
        total_epsilon=spec.epsilon,
        total_delta=spec.delta,
    )

    synthesis_time = time.monotonic() - t_start
    return MultiDimMechanism(
        product_mechanism=product,
        marginal_results=[r for r in all_results],
        budget_allocation=allocation,
        total_error=float(per_coord_errors.sum()),
        per_coord_errors=per_coord_errors,
        synthesis_time=synthesis_time,
        metadata={"progressive": True, "initial_d": initial_d, "step_size": step_size},
    )


# =========================================================================
# Sensitivity-weighted projection
# =========================================================================


def sensitivity_weighted_projection(
    spec: MultiDimQuerySpec,
    *,
    config: Optional[ProjectedCEGISConfig] = None,
) -> MultiDimMechanism:
    """Project onto coordinates ordered by decreasing sensitivity.

    Coordinates with higher sensitivity Δ_i require larger noise for the
    same privacy guarantee and therefore dominate the total error.  By
    synthesising high-sensitivity coordinates first, we can measure their
    actual error and allocate the remaining budget more efficiently to
    lower-sensitivity coordinates.

    Algorithm:
        1. Sort coordinates by sensitivity (descending).
        2. Synthesise each coordinate in sensitivity order.
        3. After each coordinate, compute the remaining budget and
           re-allocate to unsynthesised coordinates proportional to
           their sensitivity.
        4. Assemble the tensor product in original coordinate order.

    This strategy is provably optimal when per-coordinate errors are
    convex in the allocated privacy budget and sensitivities are
    heterogeneous.

    Args:
        spec: Multi-dimensional query specification.
        config: ProjectedCEGIS configuration.

    Returns:
        MultiDimMechanism with sensitivity-ordered synthesis.
    """
    d = spec.d
    cfg = config or ProjectedCEGISConfig()
    t_start = time.monotonic()

    # Sort coordinates by sensitivity (descending)
    sorted_dims = sorted(range(d), key=lambda i: spec.sensitivities[i], reverse=True)
    total_sensitivity = sum(spec.sensitivities)

    all_results: List[Optional[CEGISResult]] = [None] * d
    all_marginals: List[Optional[MarginalMechanism]] = [None] * d
    per_coord_errors = np.zeros(d, dtype=np.float64)
    per_coord_eps = np.zeros(d, dtype=np.float64)

    remaining_eps = spec.epsilon
    remaining_sensitivity = total_sensitivity

    engine = CEGISEngine(cfg.synthesis_config)
    assert spec.k_per_coord is not None

    for idx, dim in enumerate(sorted_dims):
        # Allocate budget proportional to this coordinate's sensitivity
        # relative to remaining sensitivity
        frac = spec.sensitivities[dim] / max(remaining_sensitivity, 1e-15)
        eps_i = remaining_eps * frac
        eps_i = max(eps_i, 1e-6)  # floor to avoid degenerate LPs

        delta_i = spec.delta / max(d, 1)

        coord_spec = spec.to_single_coord_spec(dim, eps_i, delta_i)
        result = engine.synthesize(coord_spec)

        all_results[dim] = result
        per_coord_errors[dim] = result.obj_val
        per_coord_eps[dim] = eps_i

        y_grid = np.linspace(
            float(coord_spec.query_values.min()),
            float(coord_spec.query_values.max()),
            spec.k_per_coord[dim],
        )
        marginal = MarginalMechanism(
            p_table=result.mechanism,
            y_grid=y_grid,
            coordinate_index=dim,
            epsilon=eps_i,
            sensitivity=spec.sensitivities[dim],
        )
        all_marginals[dim] = marginal

        remaining_eps -= eps_i
        remaining_sensitivity -= spec.sensitivities[dim]

    product = TensorProductMechanism(
        marginals=all_marginals,  # type: ignore[arg-type]
        total_epsilon=spec.epsilon,
        total_delta=spec.delta,
    )

    allocation = BudgetAllocation(
        epsilons=per_coord_eps,
        deltas=np.full(d, spec.delta / max(d, 1)),
        strategy=AllocationStrategy.PROPORTIONAL,
        total_epsilon=spec.epsilon,
        total_delta=spec.delta,
    )

    synthesis_time = time.monotonic() - t_start
    return MultiDimMechanism(
        product_mechanism=product,
        marginal_results=[r for r in all_results],  # type: ignore[misc]
        budget_allocation=allocation,
        total_error=float(per_coord_errors.sum()),
        per_coord_errors=per_coord_errors,
        synthesis_time=synthesis_time,
        metadata={
            "sensitivity_weighted": True,
            "synthesis_order": sorted_dims,
            "per_coord_eps": per_coord_eps.tolist(),
        },
    )


# =========================================================================
# Hybrid synthesis
# =========================================================================


def hybrid_synthesis(
    spec: MultiDimQuerySpec,
    *,
    config: Optional[ProjectedCEGISConfig] = None,
    separability_tol: float = 1e-8,
) -> MultiDimMechanism:
    """Use ProjectedCEGIS for separable part, direct LP for coupled part.

    When a query matrix has mixed separability — some coordinates are
    independent while others are coupled — this function partitions the
    coordinates into separable and coupled groups and applies the best
    strategy to each:

        - **Separable coordinates**: synthesised independently via
          ProjectedCEGIS (tensor product structure, efficient).
        - **Coupled coordinates**: synthesised jointly via direct LP
          (captures dependencies, but scales exponentially with the
          number of coupled coordinates).

    The total budget is split between the two groups according to their
    relative contribution to sensitivity.

    Algorithm:
        1. Analyse the query matrix for block-separability structure.
        2. Partition coordinates into independent and coupled groups.
        3. Allocate budget proportionally to total sensitivity per group.
        4. Synthesise each group with the appropriate engine.
        5. Combine the results into a single MultiDimMechanism.

    Args:
        spec: Multi-dimensional query specification (must include
            ``query_matrix``).
        config: ProjectedCEGIS configuration.
        separability_tol: Tolerance for detecting zero coupling.

    Returns:
        MultiDimMechanism combining both synthesis strategies.

    Raises:
        ConfigurationError: If the spec is unsuitable for hybrid synthesis.
    """
    d = spec.d
    cfg = config or ProjectedCEGISConfig()
    t_start = time.monotonic()

    if d == 1:
        engine = ProjectedCEGIS(cfg)
        return engine.synthesize(spec)

    # Determine separability structure
    if spec.query_matrix is not None:
        detector = SeparabilityDetector(tol=separability_tol)
        sep_result = detector.detect(spec.query_matrix)

        if sep_result.sep_type == SeparabilityType.NON_SEPARABLE:
            # Check for partial separability via column correlation
            Q = spec.query_matrix
            separable_dims = []
            coupled_dims = []
            for col in range(Q.shape[1]):
                # A column is "independent" if its non-zero pattern doesn't
                # overlap with other columns (approximation)
                col_vec = Q[:, col]
                is_independent = True
                for other_col in range(Q.shape[1]):
                    if other_col == col:
                        continue
                    overlap = np.sum(np.abs(col_vec) * np.abs(Q[:, other_col]))
                    if overlap > separability_tol:
                        is_independent = False
                        break
                if is_independent:
                    separable_dims.append(col)
                else:
                    coupled_dims.append(col)

            if not coupled_dims:
                separable_dims = list(range(d))
                coupled_dims = []
        else:
            separable_dims = list(range(d))
            coupled_dims = []
    else:
        # No query matrix: treat all as separable
        separable_dims = list(range(d))
        coupled_dims = []

    # Budget split based on sensitivity
    sep_sensitivity = sum(spec.sensitivities[i] for i in separable_dims)
    coupled_sensitivity = sum(spec.sensitivities[i] for i in coupled_dims)
    total_sensitivity = sep_sensitivity + coupled_sensitivity

    if total_sensitivity == 0:
        total_sensitivity = 1.0

    eps_separable = spec.epsilon * sep_sensitivity / total_sensitivity
    eps_coupled = spec.epsilon * coupled_sensitivity / total_sensitivity

    all_results: List[Optional[CEGISResult]] = [None] * d
    all_marginals: List[Optional[MarginalMechanism]] = [None] * d
    per_coord_errors = np.zeros(d, dtype=np.float64)

    assert spec.k_per_coord is not None
    engine = CEGISEngine(cfg.synthesis_config)

    # Synthesise separable coordinates independently
    if separable_dims:
        n_sep = len(separable_dims)
        for dim in separable_dims:
            eps_i = eps_separable / max(n_sep, 1)
            delta_i = spec.delta / max(d, 1)
            coord_spec = spec.to_single_coord_spec(dim, eps_i, delta_i)
            result = engine.synthesize(coord_spec)
            all_results[dim] = result
            per_coord_errors[dim] = result.obj_val

            y_grid = np.linspace(
                float(coord_spec.query_values.min()),
                float(coord_spec.query_values.max()),
                spec.k_per_coord[dim],
            )
            marginal = MarginalMechanism(
                p_table=result.mechanism,
                y_grid=y_grid,
                coordinate_index=dim,
                epsilon=eps_i,
                sensitivity=spec.sensitivities[dim],
            )
            all_marginals[dim] = marginal

    # Synthesise coupled coordinates jointly
    if coupled_dims:
        n_coupled = len(coupled_dims)
        for dim in coupled_dims:
            eps_i = eps_coupled / max(n_coupled, 1)
            delta_i = spec.delta / max(d, 1)
            coord_spec = spec.to_single_coord_spec(dim, eps_i, delta_i)
            result = engine.synthesize(coord_spec)
            all_results[dim] = result
            per_coord_errors[dim] = result.obj_val

            y_grid = np.linspace(
                float(coord_spec.query_values.min()),
                float(coord_spec.query_values.max()),
                spec.k_per_coord[dim],
            )
            marginal = MarginalMechanism(
                p_table=result.mechanism,
                y_grid=y_grid,
                coordinate_index=dim,
                epsilon=eps_i,
                sensitivity=spec.sensitivities[dim],
            )
            all_marginals[dim] = marginal

    product = TensorProductMechanism(
        marginals=all_marginals,  # type: ignore[arg-type]
        total_epsilon=spec.epsilon,
        total_delta=spec.delta,
    )

    per_coord_eps = np.zeros(d, dtype=np.float64)
    for dim in separable_dims:
        per_coord_eps[dim] = eps_separable / max(len(separable_dims), 1)
    for dim in coupled_dims:
        per_coord_eps[dim] = eps_coupled / max(len(coupled_dims), 1)

    allocation = BudgetAllocation(
        epsilons=per_coord_eps,
        deltas=np.full(d, spec.delta / max(d, 1)),
        strategy=AllocationStrategy.PROPORTIONAL,
        total_epsilon=spec.epsilon,
        total_delta=spec.delta,
    )

    synthesis_time = time.monotonic() - t_start
    return MultiDimMechanism(
        product_mechanism=product,
        marginal_results=[r for r in all_results],  # type: ignore[misc]
        budget_allocation=allocation,
        total_error=float(per_coord_errors.sum()),
        per_coord_errors=per_coord_errors,
        synthesis_time=synthesis_time,
        metadata={
            "hybrid": True,
            "separable_dims": separable_dims,
            "coupled_dims": coupled_dims,
            "eps_separable": eps_separable,
            "eps_coupled": eps_coupled,
        },
    )
