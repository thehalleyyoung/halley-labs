"""
Core type system for DP-Forge.

This module defines all structured data types used throughout the DP-Forge
pipeline: query specifications, LP/SDP problem structures, verification
results, CEGIS loop outputs, and configuration containers.

Design Principles:
    - Every type is a frozen or validated dataclass with ``__post_init__``
      checks to catch invalid states early.
    - Sparse matrix representations (``scipy.sparse``) are used wherever
      constraint matrices appear, since LP constraint matrices are typically
      >95% sparse for realistic mechanism sizes.
    - Enums carry metadata (e.g., ``LossFunction`` knows its callable form)
      so downstream code never needs switch statements on enum values.
    - Factory methods (``QuerySpec.counting``, ``WorkloadSpec.histogram``, etc.)
      provide ergonomic construction for common cases.

All types are re-exported from ``dp_forge.__init__``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from scipy import sparse


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QueryType(Enum):
    """Supported query families.

    Each variant describes the algebraic structure of the query, which the
    sensitivity module and LP builder exploit for tighter bounds and
    structural reductions.
    """

    COUNTING = auto()
    HISTOGRAM = auto()
    RANGE = auto()
    LINEAR_WORKLOAD = auto()
    MARGINAL = auto()
    CUSTOM = auto()

    def __repr__(self) -> str:
        return f"QueryType.{self.name}"


class MechanismFamily(Enum):
    """Mechanism families that can be synthesised.

    ``PIECEWISE_CONST``
        Discrete mechanism with piecewise-constant noise PDF over a grid.
        Synthesised via LP.

    ``PIECEWISE_LINEAR``
        Discrete mechanism with piecewise-linear interpolation between grid
        points.  Synthesised via LP with additional slope constraints.

    ``GAUSSIAN_WORKLOAD``
        Optimal Gaussian mechanism for a linear workload matrix.
        Synthesised via SDP.
    """

    PIECEWISE_CONST = auto()
    PIECEWISE_LINEAR = auto()
    GAUSSIAN_WORKLOAD = auto()

    def __repr__(self) -> str:
        return f"MechanismFamily.{self.name}"


class LossFunction(Enum):
    """Loss functions for measuring mechanism utility.

    Each variant stores a callable ``(true_val, noisy_val) -> loss`` in its
    ``fn`` attribute.  ``CUSTOM`` stores ``None``; the caller must supply a
    callable via :attr:`QuerySpec.loss_fn`.
    """

    L1 = auto()
    L2 = auto()
    LINF = auto()
    CUSTOM = auto()

    @property
    def fn(self) -> Optional[Callable[[float, float], float]]:
        """Return the built-in loss callable, or ``None`` for CUSTOM."""
        _builtins: dict[LossFunction, Callable[[float, float], float]] = {
            LossFunction.L1: lambda t, n: abs(t - n),
            LossFunction.L2: lambda t, n: (t - n) ** 2,
            LossFunction.LINF: lambda t, n: abs(t - n),
        }
        return _builtins.get(self)

    def __repr__(self) -> str:
        return f"LossFunction.{self.name}"


class SamplingMethod(Enum):
    """Sampling strategy for drawing from a synthesised mechanism."""

    ALIAS = auto()
    CDF = auto()
    REJECTION = auto()

    def __repr__(self) -> str:
        return f"SamplingMethod.{self.name}"


class CompositionType(Enum):
    """Privacy composition theorem to use when aggregating budgets."""

    BASIC = auto()
    ADVANCED = auto()
    RDP = auto()
    ZERO_CDP = auto()

    def __repr__(self) -> str:
        return f"CompositionType.{self.name}"


class SolverBackend(Enum):
    """LP/SDP solver backends, in preference order."""

    HIGHS = "highs"
    GLPK = "glpk"
    MOSEK = "mosek"
    SCS = "scs"
    SCIPY = "scipy"
    AUTO = "auto"

    def __repr__(self) -> str:
        return f"SolverBackend.{self.name}"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdjacencyRelation:
    """Defines which database pairs are considered "adjacent" for DP.

    Attributes:
        edges: Explicit list of adjacent pairs ``(i, i')``.
        n: Number of databases / input values.
        symmetric: Whether adjacency is symmetric (``(i,j)`` implies ``(j,i)``).
        description: Human-readable description of the adjacency notion.
    """

    edges: List[Tuple[int, int]]
    n: int
    symmetric: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        for i, j in self.edges:
            if not (0 <= i < self.n and 0 <= j < self.n):
                raise ValueError(
                    f"Edge ({i}, {j}) out of range for n={self.n}"
                )
            if i == j:
                raise ValueError(f"Self-loop ({i}, {j}) is not a valid adjacency edge")

    @classmethod
    def hamming_distance_1(cls, n: int) -> AdjacencyRelation:
        """Create adjacency for consecutive integers (Hamming distance 1)."""
        edges = [(i, i + 1) for i in range(n - 1)]
        return cls(edges=edges, n=n, symmetric=True, description="Hamming-1 (consecutive)")

    @classmethod
    def complete(cls, n: int) -> AdjacencyRelation:
        """Every pair is adjacent (global sensitivity setting)."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return cls(edges=edges, n=n, symmetric=True, description="Complete adjacency")

    @property
    def num_edges(self) -> int:
        """Number of directed adjacency edges (accounting for symmetry)."""
        return len(self.edges) * (2 if self.symmetric else 1)

    def __repr__(self) -> str:
        return (
            f"AdjacencyRelation(n={self.n}, edges={len(self.edges)}, "
            f"symmetric={self.symmetric})"
        )


@dataclass
class PrivacyBudget:
    """Privacy budget for (ε, δ)-differential privacy.

    Attributes:
        epsilon: Privacy parameter ε > 0.
        delta: Privacy parameter δ ∈ [0, 1). Use 0 for pure DP.
        composition_type: Composition theorem for multi-query budgeting.
    """

    epsilon: float
    delta: float = 0.0
    composition_type: CompositionType = CompositionType.BASIC

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not math.isfinite(self.epsilon):
            raise ValueError(f"epsilon must be finite, got {self.epsilon}")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")

    @property
    def is_pure(self) -> bool:
        """Whether this is a pure DP budget (δ = 0)."""
        return self.delta == 0.0

    def __repr__(self) -> str:
        if self.is_pure:
            return f"PrivacyBudget(ε={self.epsilon})"
        return f"PrivacyBudget(ε={self.epsilon}, δ={self.delta})"


@dataclass
class QuerySpec:
    """Full specification of a query for mechanism synthesis.

    This is the primary input to the CEGIS pipeline. It describes what
    query is being answered, the privacy requirements, the discretization
    resolution, and the loss function.

    Attributes:
        query_values: Array of n distinct query output values f(x_1), ..., f(x_n).
        domain: Description or enumeration of the query input domain.
        sensitivity: Global sensitivity of the query (pre-computed or certified).
        epsilon: Privacy parameter ε > 0.
        delta: Privacy parameter δ ≥ 0.
        k: Number of discretization bins for the output grid.
        loss_fn: Loss function variant.
        custom_loss: Callable for CUSTOM loss (required iff ``loss_fn == CUSTOM``).
        edges: Adjacency relation between database indices.
        query_type: Structural type of the query.
        metadata: Arbitrary extra metadata.
    """

    query_values: npt.NDArray[np.float64]
    domain: Any
    sensitivity: float
    epsilon: float
    delta: float = 0.0
    k: int = 100
    loss_fn: LossFunction = LossFunction.L2
    custom_loss: Optional[Callable[[float, float], float]] = None
    edges: Optional[AdjacencyRelation] = None
    query_type: QueryType = QueryType.CUSTOM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query_values = np.asarray(self.query_values, dtype=np.float64)
        if self.query_values.ndim != 1:
            raise ValueError(
                f"query_values must be 1-D, got shape {self.query_values.shape}"
            )
        n = len(self.query_values)
        if n < 1:
            raise ValueError("query_values must be non-empty")
        if self.sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {self.sensitivity}")
        if not math.isfinite(self.sensitivity):
            raise ValueError(f"sensitivity must be finite, got {self.sensitivity}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not math.isfinite(self.epsilon):
            raise ValueError(f"epsilon must be finite, got {self.epsilon}")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")
        if self.loss_fn == LossFunction.CUSTOM and self.custom_loss is None:
            raise ValueError("custom_loss callable required when loss_fn is CUSTOM")
        if self.edges is not None and self.edges.n != n:
            raise ValueError(
                f"edges.n ({self.edges.n}) must match len(query_values) ({n})"
            )
        # Auto-build default adjacency (consecutive Hamming-1) if not given
        if self.edges is None:
            self.edges = AdjacencyRelation.hamming_distance_1(n)

    @property
    def n(self) -> int:
        """Number of distinct database inputs."""
        return len(self.query_values)

    @property
    def is_pure_dp(self) -> bool:
        """Whether this is a pure DP specification (δ = 0)."""
        return self.delta == 0.0

    @property
    def eta_min(self) -> float:
        """Minimum probability floor: exp(-ε) × solver_tol."""
        return math.exp(-self.epsilon) * 1e-10

    def get_loss_callable(self) -> Callable[[float, float], float]:
        """Return the loss callable for this spec."""
        if self.loss_fn == LossFunction.CUSTOM:
            assert self.custom_loss is not None
            return self.custom_loss
        fn = self.loss_fn.fn
        assert fn is not None
        return fn

    @classmethod
    def counting(
        cls,
        n: int,
        epsilon: float,
        delta: float = 0.0,
        k: int = 100,
    ) -> QuerySpec:
        """Factory for a counting query with sensitivity 1."""
        return cls(
            query_values=np.arange(n, dtype=np.float64),
            domain=f"counting({n})",
            sensitivity=1.0,
            epsilon=epsilon,
            delta=delta,
            k=k,
            query_type=QueryType.COUNTING,
        )

    @classmethod
    def histogram(
        cls,
        n_bins: int,
        epsilon: float,
        delta: float = 0.0,
        k: int = 100,
    ) -> QuerySpec:
        """Factory for a histogram query with sensitivity 1 per bin."""
        return cls(
            query_values=np.arange(n_bins, dtype=np.float64),
            domain=f"histogram({n_bins})",
            sensitivity=1.0,
            epsilon=epsilon,
            delta=delta,
            k=k,
            query_type=QueryType.HISTOGRAM,
        )

    def __repr__(self) -> str:
        dp = f"ε={self.epsilon}" + (f", δ={self.delta}" if self.delta > 0 else "")
        return (
            f"QuerySpec(n={self.n}, k={self.k}, sensitivity={self.sensitivity}, "
            f"{dp}, loss={self.loss_fn.name})"
        )


@dataclass
class WorkloadSpec:
    """Specification for a linear workload mechanism (SDP path).

    Attributes:
        matrix: Workload matrix A of shape (m, d).
        query_type: Structural type of the workload.
        structural_hint: Hint for structural optimizations (e.g., 'toeplitz').
    """

    matrix: npt.NDArray[np.float64]
    query_type: QueryType = QueryType.LINEAR_WORKLOAD
    structural_hint: Optional[str] = None

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.ndim != 2:
            raise ValueError(
                f"workload matrix must be 2-D, got shape {self.matrix.shape}"
            )
        if not np.all(np.isfinite(self.matrix)):
            raise ValueError("workload matrix contains non-finite values")

    @property
    def m(self) -> int:
        """Number of queries in the workload."""
        return self.matrix.shape[0]

    @property
    def d(self) -> int:
        """Dimension of the data domain."""
        return self.matrix.shape[1]

    @classmethod
    def identity(cls, d: int) -> WorkloadSpec:
        """Identity workload (answer all unit queries)."""
        return cls(matrix=np.eye(d), query_type=QueryType.LINEAR_WORKLOAD)

    @classmethod
    def all_range(cls, d: int) -> WorkloadSpec:
        """All prefix-sum (range) queries over d elements."""
        A = np.tril(np.ones((d, d)))
        return cls(
            matrix=A,
            query_type=QueryType.RANGE,
            structural_hint="toeplitz",
        )

    def __repr__(self) -> str:
        hint = f", hint={self.structural_hint!r}" if self.structural_hint else ""
        return f"WorkloadSpec(m={self.m}, d={self.d}{hint})"


# ---------------------------------------------------------------------------
# LP / SDP problem structures
# ---------------------------------------------------------------------------


@dataclass
class LPStruct:
    """Sparse LP structure suitable for HiGHS / GLPK / SciPy linprog.

    Represents: minimize c^T x  subject to  A_ub x <= b_ub, A_eq x = b_eq,
    bounds[i][0] <= x[i] <= bounds[i][1].

    Attributes:
        c: Objective coefficients, shape (n_vars,).
        A_ub: Inequality constraint matrix (sparse CSR).
        b_ub: Inequality RHS, shape (n_ub,).
        A_eq: Equality constraint matrix (sparse CSR), or None.
        b_eq: Equality RHS, shape (n_eq,), or None.
        bounds: Per-variable (lower, upper) bounds.
        var_map: Mapping from (i, j) mechanism indices to flat variable index.
        y_grid: Output discretization grid, shape (k,).
    """

    c: npt.NDArray[np.float64]
    A_ub: sparse.spmatrix
    b_ub: npt.NDArray[np.float64]
    A_eq: Optional[sparse.spmatrix]
    b_eq: Optional[npt.NDArray[np.float64]]
    bounds: List[Tuple[float, float]]
    var_map: Dict[Tuple[int, int], int]
    y_grid: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        n_vars = len(self.c)
        if self.A_ub.shape[1] != n_vars:
            raise ValueError(
                f"A_ub has {self.A_ub.shape[1]} columns but c has {n_vars} elements"
            )
        if len(self.b_ub) != self.A_ub.shape[0]:
            raise ValueError(
                f"b_ub length {len(self.b_ub)} != A_ub rows {self.A_ub.shape[0]}"
            )
        if self.A_eq is not None:
            if self.A_eq.shape[1] != n_vars:
                raise ValueError(
                    f"A_eq has {self.A_eq.shape[1]} columns but c has {n_vars} elements"
                )
            if self.b_eq is None:
                raise ValueError("b_eq must be provided when A_eq is not None")
            if len(self.b_eq) != self.A_eq.shape[0]:
                raise ValueError(
                    f"b_eq length {len(self.b_eq)} != A_eq rows {self.A_eq.shape[0]}"
                )
        if len(self.bounds) != n_vars:
            raise ValueError(
                f"bounds has {len(self.bounds)} entries but c has {n_vars} elements"
            )

    @property
    def n_vars(self) -> int:
        """Number of decision variables."""
        return len(self.c)

    @property
    def n_ub(self) -> int:
        """Number of inequality constraints."""
        return self.A_ub.shape[0]

    @property
    def n_eq(self) -> int:
        """Number of equality constraints."""
        return self.A_eq.shape[0] if self.A_eq is not None else 0

    @property
    def sparsity(self) -> float:
        """Fraction of zero entries in A_ub."""
        total = self.A_ub.shape[0] * self.A_ub.shape[1]
        if total == 0:
            return 1.0
        nnz = self.A_ub.nnz
        return 1.0 - nnz / total

    def __repr__(self) -> str:
        return (
            f"LPStruct(vars={self.n_vars}, ub={self.n_ub}, eq={self.n_eq}, "
            f"k={len(self.y_grid)}, sparsity={self.sparsity:.2%})"
        )


@dataclass
class SDPStruct:
    """SDP problem structure for Gaussian workload mechanism synthesis.

    Wraps a CVXPY problem with the key decision variable and metadata
    needed by the CEGIS loop and extractor.

    Attributes:
        problem: CVXPY Problem instance.
        sigma_var: CVXPY Variable representing the covariance matrix Σ.
        objective: CVXPY objective expression.
        constraints: List of CVXPY constraint expressions.
        workload: The workload specification this SDP was built from.
    """

    problem: Any  # cvxpy.Problem — not typed to avoid hard import
    sigma_var: Any  # cvxpy.Variable
    objective: Any  # cvxpy.Expression
    constraints: List[Any]
    workload: Optional[WorkloadSpec] = None

    def __repr__(self) -> str:
        status = getattr(self.problem, "status", "not_solved")
        return f"SDPStruct(status={status!r})"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class OptimalityCertificate:
    """Certificate of (near-)optimality from LP/SDP duality.

    A small duality gap certifies that the synthesised mechanism is close to
    the information-theoretic optimum for the given specification.

    Attributes:
        dual_vars: Dual variable values (solver-specific format).
        duality_gap: Absolute gap ``primal_obj - dual_obj``.
        primal_obj: Primal objective value.
        dual_obj: Dual objective value.
    """

    dual_vars: Optional[Any]
    duality_gap: float
    primal_obj: float
    dual_obj: float

    def __post_init__(self) -> None:
        if self.duality_gap < -1e-8:
            raise ValueError(
                f"duality_gap should be non-negative, got {self.duality_gap}"
            )
        if not math.isfinite(self.primal_obj):
            raise ValueError(f"primal_obj must be finite, got {self.primal_obj}")
        if not math.isfinite(self.dual_obj):
            raise ValueError(f"dual_obj must be finite, got {self.dual_obj}")

    @property
    def relative_gap(self) -> float:
        """Relative duality gap: ``gap / max(|primal|, 1)``."""
        denom = max(abs(self.primal_obj), 1.0)
        return self.duality_gap / denom

    @property
    def is_tight(self, tol: float = 1e-6) -> bool:
        """Whether the gap is within tolerance."""
        return self.relative_gap <= tol

    def __repr__(self) -> str:
        return (
            f"OptimalityCertificate(gap={self.duality_gap:.2e}, "
            f"primal={self.primal_obj:.6f}, dual={self.dual_obj:.6f})"
        )


@dataclass
class VerifyResult:
    """Result of DP verification on a mechanism.

    Attributes:
        valid: ``True`` if the mechanism satisfies (ε, δ)-DP within tolerance.
        violation: If ``valid`` is ``False``, the worst-violating tuple
            ``(i, i', j_worst, magnitude)``.  ``None`` if valid.
    """

    valid: bool
    violation: Optional[Tuple[int, int, int, float]] = None

    def __post_init__(self) -> None:
        if not self.valid and self.violation is None:
            raise ValueError("violation must be provided when valid is False")
        if self.valid and self.violation is not None:
            raise ValueError("violation must be None when valid is True")

    @property
    def violation_pair(self) -> Optional[Tuple[int, int]]:
        """The violating (i, i') pair, or None."""
        if self.violation is None:
            return None
        return (self.violation[0], self.violation[1])

    @property
    def violation_magnitude(self) -> float:
        """Magnitude of the worst violation, or 0.0 if valid."""
        if self.violation is None:
            return 0.0
        return self.violation[3]

    def __repr__(self) -> str:
        if self.valid:
            return "VerifyResult(valid=True)"
        assert self.violation is not None
        i, ip, j, mag = self.violation
        return f"VerifyResult(valid=False, pair=({i},{ip}), j={j}, mag={mag:.2e})"


@dataclass
class CEGISResult:
    """Output of the CEGIS synthesis loop.

    Attributes:
        mechanism: The n × k probability table ``p[i][j] = Pr[M(x_i) = y_j]``.
        iterations: Number of CEGIS iterations completed.
        obj_val: Final minimax objective value.
        optimality_certificate: Duality-based optimality certificate.
        convergence_history: Objective value at each iteration.
    """

    mechanism: npt.NDArray[np.float64]
    iterations: int
    obj_val: float
    optimality_certificate: Optional[OptimalityCertificate] = None
    convergence_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mechanism = np.asarray(self.mechanism, dtype=np.float64)
        if self.mechanism.ndim != 2:
            raise ValueError(
                f"mechanism must be 2-D (n × k), got shape {self.mechanism.shape}"
            )
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        # Validate rows sum to ~1
        row_sums = self.mechanism.sum(axis=1)
        max_deviation = float(np.max(np.abs(row_sums - 1.0)))
        if max_deviation > 1e-4:
            raise ValueError(
                f"mechanism rows must sum to 1 (max deviation: {max_deviation:.2e})"
            )

    @property
    def n(self) -> int:
        """Number of database inputs."""
        return self.mechanism.shape[0]

    @property
    def k(self) -> int:
        """Number of output discretization bins."""
        return self.mechanism.shape[1]

    @property
    def converged(self) -> bool:
        """Whether the convergence history shows stabilisation."""
        if len(self.convergence_history) < 2:
            return True
        last_two = self.convergence_history[-2:]
        return abs(last_two[1] - last_two[0]) < 1e-8

    def __repr__(self) -> str:
        cert = "with_certificate" if self.optimality_certificate else "no_certificate"
        return (
            f"CEGISResult(n={self.n}, k={self.k}, iter={self.iterations}, "
            f"obj={self.obj_val:.6f}, {cert})"
        )


@dataclass
class ExtractedMechanism:
    """A deployable mechanism extracted and post-processed from a CEGIS result.

    This is the final artifact: a probability table that has been projected
    onto the DP-feasible set and equipped with efficient sampling structures.

    Attributes:
        p_final: The n × k probability table after DP-preserving projection.
        cdf_tables: Per-row CDF tables for CDF-based sampling.
        alias_tables: Per-row alias tables for O(1) sampling.
        optimality_certificate: Optimality certificate from the solver.
        metadata: Extraction metadata (solver info, timings, etc.).
    """

    p_final: npt.NDArray[np.float64]
    cdf_tables: Optional[npt.NDArray[np.float64]] = None
    alias_tables: Optional[List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]]] = None
    optimality_certificate: Optional[OptimalityCertificate] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p_final = np.asarray(self.p_final, dtype=np.float64)
        if self.p_final.ndim != 2:
            raise ValueError(
                f"p_final must be 2-D (n × k), got shape {self.p_final.shape}"
            )
        # All probabilities must be non-negative
        if np.any(self.p_final < -1e-12):
            min_val = float(np.min(self.p_final))
            raise ValueError(
                f"p_final contains negative probabilities (min: {min_val:.2e})"
            )
        # Rows must sum to ~1
        row_sums = self.p_final.sum(axis=1)
        max_deviation = float(np.max(np.abs(row_sums - 1.0)))
        if max_deviation > 1e-6:
            raise ValueError(
                f"p_final rows must sum to 1 (max deviation: {max_deviation:.2e})"
            )

    @property
    def n(self) -> int:
        """Number of database inputs."""
        return self.p_final.shape[0]

    @property
    def k(self) -> int:
        """Number of output bins."""
        return self.p_final.shape[1]

    def __repr__(self) -> str:
        cert = "certified" if self.optimality_certificate else "uncertified"
        return f"ExtractedMechanism(n={self.n}, k={self.k}, {cert})"


@dataclass
class BenchmarkResult:
    """Result of benchmarking a mechanism on a query.

    Attributes:
        mse: Mean squared error of the mechanism.
        mae: Mean absolute error of the mechanism.
        synthesis_time: Time to synthesise the mechanism, in seconds.
        iterations: Number of CEGIS iterations.
        privacy_verified: Whether the mechanism passed DP verification.
    """

    mse: float
    mae: float
    synthesis_time: float
    iterations: int
    privacy_verified: bool

    def __post_init__(self) -> None:
        if self.mse < 0:
            raise ValueError(f"mse must be >= 0, got {self.mse}")
        if self.mae < 0:
            raise ValueError(f"mae must be >= 0, got {self.mae}")
        if self.synthesis_time < 0:
            raise ValueError(f"synthesis_time must be >= 0, got {self.synthesis_time}")
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")

    def __repr__(self) -> str:
        verified = "✓" if self.privacy_verified else "✗"
        return (
            f"BenchmarkResult(mse={self.mse:.6f}, mae={self.mae:.6f}, "
            f"time={self.synthesis_time:.2f}s, iter={self.iterations}, dp={verified})"
        )


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Configuration for sampling from a synthesised mechanism.

    Attributes:
        method: Sampling method to use.
        seed: Random seed for reproducibility. None for non-deterministic.
    """

    method: SamplingMethod = SamplingMethod.ALIAS
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    def get_rng(self) -> np.random.Generator:
        """Create a numpy random Generator with the configured seed."""
        return np.random.default_rng(self.seed)

    def __repr__(self) -> str:
        seed_str = f", seed={self.seed}" if self.seed is not None else ""
        return f"SamplingConfig(method={self.method.name}{seed_str})"


@dataclass
class NumericalConfig:
    """Numerical precision and stability configuration.

    Attributes:
        solver_tol: Solver feasibility / optimality tolerance.
        dp_tol: Tolerance for DP verification. Must satisfy
            ``dp_tol >= exp(ε) × solver_tol`` (Invariant I4).
        eta_min_scale: Scale factor for η_min = exp(-ε) × eta_min_scale.
        max_condition_number: Maximum acceptable condition number for
            constraint matrices before raising NumericalInstabilityError.
    """

    solver_tol: float = 1e-8
    dp_tol: float = 1e-6
    eta_min_scale: float = 1e-10
    max_condition_number: float = 1e12

    def __post_init__(self) -> None:
        if self.solver_tol <= 0:
            raise ValueError(f"solver_tol must be > 0, got {self.solver_tol}")
        if self.dp_tol <= 0:
            raise ValueError(f"dp_tol must be > 0, got {self.dp_tol}")
        if self.eta_min_scale <= 0:
            raise ValueError(f"eta_min_scale must be > 0, got {self.eta_min_scale}")
        if self.max_condition_number <= 0:
            raise ValueError(
                f"max_condition_number must be > 0, got {self.max_condition_number}"
            )

    def eta_min(self, epsilon: float) -> float:
        """Compute η_min = exp(-ε) × eta_min_scale."""
        return math.exp(-epsilon) * self.eta_min_scale

    def validate_dp_tol(self, epsilon: float) -> bool:
        """Check Invariant I4: dp_tol >= exp(ε) × solver_tol."""
        return self.dp_tol >= math.exp(epsilon) * self.solver_tol

    def __repr__(self) -> str:
        return (
            f"NumericalConfig(solver_tol={self.solver_tol:.0e}, "
            f"dp_tol={self.dp_tol:.0e}, max_cond={self.max_condition_number:.0e})"
        )


@dataclass
class SynthesisConfig:
    """Configuration for the CEGIS synthesis loop.

    Attributes:
        max_iter: Maximum CEGIS iterations before ConvergenceError.
        tol: Convergence tolerance on objective improvement.
        warm_start: Whether to use Laplace warm-start for the first LP.
        solver: LP/SDP solver backend.
        verbose: Verbosity level (0=silent, 1=progress, 2=debug).
        eta_min: Minimum probability floor (overrides NumericalConfig if set).
        symmetry_detection: Whether to detect and exploit query symmetries.
        numerical: Numerical precision configuration.
        sampling: Sampling configuration for the final mechanism.
    """

    max_iter: int = 50
    tol: float = 1e-8
    warm_start: bool = True
    solver: SolverBackend = SolverBackend.AUTO
    verbose: int = 1
    eta_min: Optional[float] = None
    symmetry_detection: bool = True
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if self.verbose not in (0, 1, 2):
            raise ValueError(f"verbose must be 0, 1, or 2, got {self.verbose}")
        if self.eta_min is not None and self.eta_min <= 0:
            raise ValueError(f"eta_min must be > 0, got {self.eta_min}")

    def effective_eta_min(self, epsilon: float) -> float:
        """Return the effective η_min, using the override or the formula."""
        if self.eta_min is not None:
            return self.eta_min
        return self.numerical.eta_min(epsilon)

    def __repr__(self) -> str:
        return (
            f"SynthesisConfig(max_iter={self.max_iter}, tol={self.tol:.0e}, "
            f"solver={self.solver.name}, warm_start={self.warm_start})"
        )


# ---------------------------------------------------------------------------
# Abstract domains (used by CEGAR, interpolation, SMT, lattice modules)
# ---------------------------------------------------------------------------


class AbstractDomainType(Enum):
    """Types of abstract domains for program analysis."""

    INTERVAL = auto()
    OCTAGON = auto()
    POLYHEDRA = auto()
    ZONE = auto()
    PREDICATE = auto()

    def __repr__(self) -> str:
        return f"AbstractDomainType.{self.name}"


@dataclass
class AbstractValue:
    """A value in an abstract domain, used for abstract interpretation.

    Attributes:
        domain_type: The abstract domain this value belongs to.
        lower: Lower bound (for interval-like domains).
        upper: Upper bound (for interval-like domains).
        constraints: Additional constraints (for polyhedra/predicate domains).
    """

    domain_type: AbstractDomainType
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    constraints: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"lower shape {self.lower.shape} != upper shape {self.upper.shape}"
            )
        if np.any(self.lower > self.upper + 1e-12):
            raise ValueError("lower bound exceeds upper bound")

    @property
    def ndim(self) -> int:
        """Dimensionality of the abstract value."""
        return self.lower.shape[0] if self.lower.ndim == 1 else int(np.prod(self.lower.shape))

    def __repr__(self) -> str:
        return f"AbstractValue(domain={self.domain_type.name}, ndim={self.ndim})"


# ---------------------------------------------------------------------------
# Game-theoretic types (used by game_theory, sparse modules)
# ---------------------------------------------------------------------------


@dataclass
class GameMatrix:
    """Payoff matrix for a two-player game.

    Rows correspond to the mechanism designer's strategies, columns to the
    adversary's strategies. Entry (i, j) is the payoff (e.g., privacy loss
    or utility) when designer plays i and adversary plays j.

    Attributes:
        payoffs: The m × n payoff matrix.
        row_labels: Optional labels for row strategies.
        col_labels: Optional labels for column strategies.
    """

    payoffs: npt.NDArray[np.float64]
    row_labels: Optional[List[str]] = None
    col_labels: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.payoffs = np.asarray(self.payoffs, dtype=np.float64)
        if self.payoffs.ndim != 2:
            raise ValueError(
                f"payoffs must be 2-D, got shape {self.payoffs.shape}"
            )
        if self.row_labels is not None and len(self.row_labels) != self.payoffs.shape[0]:
            raise ValueError("row_labels length must match payoffs rows")
        if self.col_labels is not None and len(self.col_labels) != self.payoffs.shape[1]:
            raise ValueError("col_labels length must match payoffs columns")

    @property
    def m(self) -> int:
        """Number of row strategies (designer)."""
        return self.payoffs.shape[0]

    @property
    def n(self) -> int:
        """Number of column strategies (adversary)."""
        return self.payoffs.shape[1]

    def __repr__(self) -> str:
        return f"GameMatrix(m={self.m}, n={self.n})"


# ---------------------------------------------------------------------------
# Interpolant types (used by interpolation, CEGAR, SMT modules)
# ---------------------------------------------------------------------------


class InterpolantType(Enum):
    """Types of Craig interpolants for synthesis refinement."""

    LINEAR_ARITHMETIC = auto()
    BOOLEAN = auto()
    QUANTIFIER_FREE = auto()
    TREE = auto()
    SEQUENCE = auto()

    def __repr__(self) -> str:
        return f"InterpolantType.{self.name}"


@dataclass
class Formula:
    """A logical formula representation for SMT/interpolation.

    Attributes:
        expr: String representation of the formula.
        variables: Set of variable names appearing in the formula.
        formula_type: Type classification (e.g., 'linear_arithmetic', 'boolean').
        metadata: Additional metadata.
    """

    expr: str
    variables: FrozenSet[str]
    formula_type: str = "linear_arithmetic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.expr:
            raise ValueError("Formula expression must be non-empty")
        if not isinstance(self.variables, frozenset):
            self.variables = frozenset(self.variables)

    def __repr__(self) -> str:
        return f"Formula({self.expr!r}, vars={len(self.variables)})"


@dataclass
class Predicate:
    """A predicate used in predicate abstraction and CEGAR.

    Attributes:
        name: Human-readable name for the predicate.
        formula: The underlying logical formula.
        is_atomic: Whether this is an atomic predicate.
    """

    name: str
    formula: Formula
    is_atomic: bool = True

    def __repr__(self) -> str:
        return f"Predicate({self.name!r})"


# ---------------------------------------------------------------------------
# Streaming DP types
# ---------------------------------------------------------------------------


class StreamEventType(Enum):
    """Types of events in a data stream for continual observation."""

    INSERTION = auto()
    DELETION = auto()
    UPDATE = auto()
    QUERY = auto()

    def __repr__(self) -> str:
        return f"StreamEventType.{self.name}"


@dataclass
class StreamEvent:
    """A single event in a data stream.

    Attributes:
        timestamp: Monotonically increasing timestamp.
        event_type: Type of stream event.
        value: The data value associated with the event.
        weight: Optional weight for weighted streams.
    """

    timestamp: int
    event_type: StreamEventType
    value: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be >= 0, got {self.timestamp}")
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value}")

    def __repr__(self) -> str:
        return (
            f"StreamEvent(t={self.timestamp}, type={self.event_type.name}, "
            f"value={self.value})"
        )


# ---------------------------------------------------------------------------
# Privacy definition variants
# ---------------------------------------------------------------------------


class PrivacyNotion(Enum):
    """Supported privacy definitions across the system."""

    PURE_DP = auto()
    APPROXIMATE_DP = auto()
    RENYI_DP = auto()
    ZERO_CONCENTRATED_DP = auto()
    LOCAL_DP = auto()
    SHUFFLE_DP = auto()
    F_DP = auto()

    def __repr__(self) -> str:
        return f"PrivacyNotion.{self.name}"


@dataclass
class ZCDPBudget:
    """Budget for zero-concentrated differential privacy (ρ-zCDP).

    Attributes:
        rho: Concentration parameter ρ > 0.
        xi: Optional offset ξ for (ξ, ρ)-zCDP.
    """

    rho: float
    xi: float = 0.0

    def __post_init__(self) -> None:
        if self.rho <= 0:
            raise ValueError(f"rho must be > 0, got {self.rho}")
        if not math.isfinite(self.rho):
            raise ValueError(f"rho must be finite, got {self.rho}")
        if self.xi < 0:
            raise ValueError(f"xi must be >= 0, got {self.xi}")

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert to (ε, δ)-DP using optimal conversion.

        Uses ε = ρ + 2√(ρ·ln(1/δ)) (Bun & Steinke 2016).
        """
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        epsilon = self.xi + self.rho + 2.0 * math.sqrt(self.rho * math.log(1.0 / delta))
        return PrivacyBudget(epsilon=epsilon, delta=delta)

    @property
    def is_pure_zcdp(self) -> bool:
        """Whether this is pure ρ-zCDP (ξ = 0)."""
        return self.xi == 0.0

    def __repr__(self) -> str:
        if self.is_pure_zcdp:
            return f"ZCDPBudget(ρ={self.rho})"
        return f"ZCDPBudget(ξ={self.xi}, ρ={self.rho})"


@dataclass
class LocalDPBudget:
    """Budget for local differential privacy.

    Attributes:
        epsilon: Privacy parameter ε > 0.
        protocol: Name of the LDP protocol (e.g., 'randomized_response', 'rappor').
    """

    epsilon: float
    protocol: str = "randomized_response"

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not math.isfinite(self.epsilon):
            raise ValueError(f"epsilon must be finite, got {self.epsilon}")

    @property
    def flip_probability(self) -> float:
        """Probability of truthful response in randomized response."""
        return math.exp(self.epsilon) / (math.exp(self.epsilon) + 1.0)

    def __repr__(self) -> str:
        return f"LocalDPBudget(ε={self.epsilon}, protocol={self.protocol!r})"


# ---------------------------------------------------------------------------
# Lattice types (used by lattice module)
# ---------------------------------------------------------------------------


@dataclass
class LatticePoint:
    """A point in a mechanism design lattice.

    Attributes:
        coordinates: The coordinates of the lattice point.
        objective_value: The objective value at this point (if evaluated).
        feasible: Whether this point is DP-feasible.
    """

    coordinates: npt.NDArray[np.float64]
    objective_value: Optional[float] = None
    feasible: Optional[bool] = None

    def __post_init__(self) -> None:
        self.coordinates = np.asarray(self.coordinates, dtype=np.float64)

    @property
    def ndim(self) -> int:
        """Dimensionality of the lattice point."""
        return len(self.coordinates)

    def __repr__(self) -> str:
        feas = "?" if self.feasible is None else ("✓" if self.feasible else "✗")
        obj = f", obj={self.objective_value:.6f}" if self.objective_value is not None else ""
        return f"LatticePoint(ndim={self.ndim}{obj}, feasible={feas})"


# ---------------------------------------------------------------------------
# Gradient / autodiff types
# ---------------------------------------------------------------------------


@dataclass
class GradientInfo:
    """Gradient information from automatic differentiation.

    Attributes:
        value: Function value at the evaluation point.
        gradient: First-order gradient vector.
        hessian: Optional second-order Hessian matrix.
        computation_graph: Optional reference to the computation graph.
    """

    value: float
    gradient: npt.NDArray[np.float64]
    hessian: Optional[npt.NDArray[np.float64]] = None
    computation_graph: Optional[Any] = None

    def __post_init__(self) -> None:
        self.gradient = np.asarray(self.gradient, dtype=np.float64)
        if self.hessian is not None:
            self.hessian = np.asarray(self.hessian, dtype=np.float64)
            n = len(self.gradient)
            if self.hessian.shape != (n, n):
                raise ValueError(
                    f"Hessian shape {self.hessian.shape} doesn't match "
                    f"gradient dimension {n}"
                )

    @property
    def ndim(self) -> int:
        """Dimensionality of the gradient."""
        return len(self.gradient)

    @property
    def gradient_norm(self) -> float:
        """L2 norm of the gradient."""
        return float(np.linalg.norm(self.gradient))

    def __repr__(self) -> str:
        hess = ", with_hessian" if self.hessian is not None else ""
        return f"GradientInfo(value={self.value:.6f}, |∇|={self.gradient_norm:.2e}{hess})"
