"""
Core type definitions for CausalCert.

Every public data structure flowing between modules is defined here so that
sub-packages depend only on ``causalcert.types`` rather than on each other.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NodeId = int
"""Nodes are identified by non-negative integer indices into the adjacency matrix."""

AdjacencyMatrix = NDArray[np.int8]
"""Binary square matrix where ``A[i, j] == 1`` iff edge *i → j* exists."""

NodeSet = frozenset[NodeId]
"""Immutable set of node ids, used for conditioning sets and ancestral sets."""

EdgeTuple = tuple[NodeId, NodeId]
"""Directed edge represented as ``(source, target)``."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EditType(enum.Enum):
    """Kind of structural perturbation applied to a DAG edge."""

    ADD = "add"
    """Insert a directed edge that was absent."""

    DELETE = "delete"
    """Remove a directed edge that was present."""

    REVERSE = "reverse"
    """Reverse the orientation of an existing directed edge."""


class FragilityChannel(enum.Enum):
    """Source of fragility for a single edge.

    Each channel corresponds to one mechanism by which an edge edit
    can alter a causal conclusion.
    """

    D_SEPARATION = "d_separation"
    """Edit changes a d-separation / d-connection relation used by CI tests."""

    IDENTIFICATION = "identification"
    """Edit invalidates (or creates) a valid adjustment set for identification."""

    ESTIMATION = "estimation"
    """Edit changes the numerical causal effect estimate significantly."""


class SolverStrategy(enum.Enum):
    """Which solver back-end to use for the robustness-radius computation."""

    ILP = "ilp"
    """Exact integer linear program (ALG 4)."""

    LP_RELAXATION = "lp_relaxation"
    """LP relaxation for fast lower bounds (ALG 5)."""

    FPT = "fpt"
    """Fixed-parameter tractable DP on a tree decomposition (ALG 7)."""

    CDCL = "cdcl"
    """Conflict-driven clause-learning search."""

    AUTO = "auto"
    """Automatically select the best strategy based on DAG size and treewidth."""


class VariableType(enum.Enum):
    """Statistical type of a variable column in the dataset."""

    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    BINARY = "binary"


class CITestMethod(enum.Enum):
    """Available conditional-independence test methods."""

    KERNEL = "kernel"
    """Kernel CI test with Nyström approximation."""

    PARTIAL_CORRELATION = "partial_correlation"
    """Fisher-z partial-correlation test."""

    RANK = "rank"
    """Rank-based (Spearman) partial-correlation test."""

    CRT = "crt"
    """Conditional randomization test."""

    ENSEMBLE = "ensemble"
    """Cauchy combination across multiple test methods (ALG 6)."""

    HSIC = "hsic"
    """Hilbert-Schmidt Independence Criterion test."""

    MUTUAL_INFO = "mutual_info"
    """Mutual-information based CI test (KSG estimator)."""

    CLASSIFIER = "classifier"
    """Classifier-based CI test (CCIT)."""

    ADAPTIVE = "adaptive"
    """Adaptive meta-learning test selection."""


# ---------------------------------------------------------------------------
# Core data-classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructuralEdit:
    """A single edge edit applied to a DAG.

    Attributes
    ----------
    edit_type : EditType
        Kind of edit.
    source : NodeId
        Tail of the directed edge (or the edge being added/removed).
    target : NodeId
        Head of the directed edge.
    """

    edit_type: EditType
    source: NodeId
    target: NodeId

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("Self-loops are not permitted in a DAG.")

    @property
    def edge(self) -> EdgeTuple:
        """Return the ``(source, target)`` tuple."""
        return (self.source, self.target)

    @property
    def cost(self) -> int:
        """Edit cost (each edit has unit cost by default)."""
        return 1


@dataclass(frozen=True, slots=True)
class CITestResult:
    """Result of a single conditional-independence test.

    Attributes
    ----------
    x : NodeId
        First variable.
    y : NodeId
        Second variable.
    conditioning_set : NodeSet
        Set of conditioning variables.
    statistic : float
        Test statistic value.
    p_value : float
        p-value for the null hypothesis X ⊥ Y | S.
    method : CITestMethod
        Which CI test method produced this result.
    reject : bool
        Whether the null (independence) was rejected at the prescribed α.
    alpha : float
        Significance level used.
    """

    x: NodeId
    y: NodeId
    conditioning_set: NodeSet
    statistic: float
    p_value: float
    method: CITestMethod
    reject: bool
    alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class FragilityScore:
    """Fragility annotation for a single existing or potential edge.

    Attributes
    ----------
    edge : EdgeTuple
        The directed edge ``(i, j)``.
    total_score : float
        Aggregated fragility score in [0, 1]; higher ⇒ more fragile.
    channel_scores : dict[FragilityChannel, float]
        Per-channel contributions to the total.
    witness_ci : CITestResult | None
        The CI test result most affected by this edge, if applicable.
    """

    edge: EdgeTuple
    total_score: float
    channel_scores: dict[FragilityChannel, float] = field(default_factory=dict)
    witness_ci: CITestResult | None = None


@dataclass(frozen=True, slots=True)
class RobustnessRadius:
    """Result of the robustness-radius computation.

    Attributes
    ----------
    lower_bound : int
        Certified lower bound on the minimum edit distance.
    upper_bound : int
        Upper bound (equals ``lower_bound`` when the solver proves optimality).
    witness_edits : tuple[StructuralEdit, ...]
        An explicit edit set attaining the upper bound.
    solver_strategy : SolverStrategy
        Which solver produced this result.
    solver_time_s : float
        Wall-clock time in seconds.
    gap : float
        Relative optimality gap ``(UB − LB) / UB``.
    certified : bool
        ``True`` when ``lower_bound == upper_bound`` (exact).
    """

    lower_bound: int
    upper_bound: int
    witness_edits: tuple[StructuralEdit, ...] = ()
    solver_strategy: SolverStrategy = SolverStrategy.AUTO
    solver_time_s: float = 0.0
    gap: float = 0.0
    certified: bool = False

    def __post_init__(self) -> None:
        if self.lower_bound > self.upper_bound:
            raise ValueError("Lower bound exceeds upper bound.")


@dataclass(frozen=True, slots=True)
class EstimationResult:
    """Causal effect estimate under a particular DAG.

    Attributes
    ----------
    ate : float
        Average treatment effect point estimate.
    se : float
        Standard error of the ATE estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    adjustment_set : NodeSet
        The valid adjustment set used for identification.
    method : str
        Name of the estimation method (e.g. ``"aipw"``).
    n_obs : int
        Number of observations used.
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    adjustment_set: NodeSet
    method: str = "aipw"
    n_obs: int = 0


@dataclass(slots=True)
class AuditReport:
    """Full structural-robustness audit produced by the pipeline.

    This is the top-level output object returned by
    :pyclass:`causalcert.pipeline.orchestrator.CausalCertPipeline`.

    Attributes
    ----------
    treatment : NodeId
        Treatment variable index.
    outcome : NodeId
        Outcome variable index.
    n_nodes : int
        Number of nodes in the DAG.
    n_edges : int
        Number of edges in the original DAG.
    radius : RobustnessRadius
        The computed robustness radius (with bounds and witness).
    fragility_ranking : list[FragilityScore]
        Edges ranked by decreasing fragility score.
    baseline_estimate : EstimationResult | None
        Causal effect estimate under the original DAG.
    perturbed_estimates : list[EstimationResult]
        Effect estimates under each witness perturbation.
    ci_results : list[CITestResult]
        All CI test results evaluated during the audit.
    metadata : dict[str, Any]
        Free-form provenance metadata (timings, seeds, versions).
    """

    treatment: NodeId
    outcome: NodeId
    n_nodes: int
    n_edges: int
    radius: RobustnessRadius
    fragility_ranking: list[FragilityScore] = field(default_factory=list)
    baseline_estimate: EstimationResult | None = None
    perturbed_estimates: list[EstimationResult] = field(default_factory=list)
    ci_results: list[CITestResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline / run configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineConfig:
    """Global configuration governing a CausalCert run.

    Attributes
    ----------
    treatment : NodeId
        Treatment variable index.
    outcome : NodeId
        Outcome variable index.
    alpha : float
        Significance level for CI tests.
    ci_method : CITestMethod
        Default CI test method.
    solver_strategy : SolverStrategy
        Which solver back-end to use.
    max_k : int
        Maximum edit distance to search.
    n_folds : int
        Number of cross-fitting folds for estimation.
    fdr_method : str
        FDR control method (``"by"`` = Benjamini–Yekutieli).
    n_jobs : int
        Number of parallel workers (``-1`` for all CPUs).
    seed : int
        Random seed for reproducibility.
    cache_dir : str | None
        Directory for caching intermediate results.
    """

    treatment: NodeId = 0
    outcome: NodeId = 1
    alpha: float = 0.05
    ci_method: CITestMethod = CITestMethod.ENSEMBLE
    solver_strategy: SolverStrategy = SolverStrategy.AUTO
    max_k: int = 10
    n_folds: int = 5
    fdr_method: str = "by"
    n_jobs: int = 1
    seed: int = 42
    cache_dir: str | None = None


# ---------------------------------------------------------------------------
# Protocols (structural sub-typing interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class ConclusionPredicate(Protocol):
    """A causal conclusion that can be evaluated on a DAG + data.

    Implementations must define ``__call__`` which returns ``True`` when the
    conclusion holds under the given DAG and data.
    """

    def __call__(
        self,
        adj: AdjacencyMatrix,
        data: Any,
        *,
        treatment: NodeId,
        outcome: NodeId,
    ) -> bool:
        """Evaluate the conclusion predicate.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix of the (possibly perturbed) DAG.
        data : Any
            Observational dataset.
        treatment : NodeId
            Treatment variable.
        outcome : NodeId
            Outcome variable.

        Returns
        -------
        bool
            ``True`` if the conclusion holds.
        """
        ...


@runtime_checkable
class CIOracle(Protocol):
    """Oracle that answers conditional-independence queries on a DAG."""

    def is_d_separated(
        self,
        x: NodeId,
        y: NodeId,
        conditioning: NodeSet,
    ) -> bool:
        """Return ``True`` if *x* and *y* are d-separated given *conditioning*."""
        ...


@runtime_checkable
class CITester(Protocol):
    """Statistical conditional-independence tester."""

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: Any,
    ) -> CITestResult:
        """Test X ⊥ Y | S and return a :class:`CITestResult`."""
        ...


@runtime_checkable
class CausalEstimator(Protocol):
    """Estimator for causal effects under a given DAG."""

    def estimate(
        self,
        adj: AdjacencyMatrix,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        adjustment_set: NodeSet,
    ) -> EstimationResult:
        """Estimate the causal effect.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Current DAG adjacency matrix.
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome variables.
        adjustment_set : NodeSet
            Valid adjustment set for the back-door criterion.

        Returns
        -------
        EstimationResult
        """
        ...


@runtime_checkable
class RobustnessSolver(Protocol):
    """Solver for the minimum-edit robustness radius."""

    def solve(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int,
    ) -> RobustnessRadius:
        """Compute the robustness radius.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG adjacency matrix.
        predicate : ConclusionPredicate
            Conclusion to be stress-tested.
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        max_k : int
            Maximum edit distance to explore.

        Returns
        -------
        RobustnessRadius
        """
        ...


@runtime_checkable
class FragilityScorer(Protocol):
    """Assigns per-edge fragility scores."""

    def score(
        self,
        adj: AdjacencyMatrix,
        treatment: NodeId,
        outcome: NodeId,
    ) -> Sequence[FragilityScore]:
        """Score all edges (and candidate additions) for fragility.

        Returns
        -------
        Sequence[FragilityScore]
            Scores sorted by decreasing fragility.
        """
        ...


# ---------------------------------------------------------------------------
# Extended enumerations
# ---------------------------------------------------------------------------


class SolverStatus(enum.Enum):
    """Termination status reported by a robustness-radius solver.

    Used by all solver back-ends (ILP, LP relaxation, FPT-DP, CDCL) to
    communicate whether the returned solution is optimal, feasible but
    sub-optimal, or no solution was found.
    """

    OPTIMAL = "optimal"
    """Solver proved global optimality of the returned solution."""

    FEASIBLE = "feasible"
    """A feasible solution was found, but optimality is not guaranteed."""

    INFEASIBLE = "infeasible"
    """The problem is infeasible (no overturning edit set exists up to max_k)."""

    TIMEOUT = "timeout"
    """Solver hit the time limit before proving optimality or infeasibility."""

    UNKNOWN = "unknown"
    """Solver status could not be determined (e.g. numerical issues)."""


class FormatType(enum.Enum):
    """Supported file formats for DAG serialization and deserialization.

    Each variant corresponds to a format parser / writer pair registered
    with :class:`causalcert.formats.registry.FormatRegistry`.
    """

    DOT = "dot"
    """Graphviz DOT language."""

    DAGITTY = "dagitty"
    """DAGitty browser-tool format."""

    BIF = "bif"
    """Bayesian Interchange Format (Bayes net repository)."""

    JSON = "json"
    """CausalCert native JSON format."""

    CSV = "csv"
    """Adjacency-list CSV (edge list or adjacency matrix)."""

    TETRAD = "tetrad"
    """TETRAD project XML format."""

    PCALG = "pcalg"
    """R ``pcalg`` package adjacency CSV convention."""

    GML = "gml"
    """Graph Modeling Language format."""


# ---------------------------------------------------------------------------
# Extended data-classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TreewidthBound:
    """Lower and upper bounds on the treewidth of a graph.

    Used by decomposition algorithms to report certified bounds and by the
    solver auto-selection logic to decide whether to invoke the FPT-DP
    solver (which is efficient only when treewidth is small).

    Attributes
    ----------
    lower : int
        Certified lower bound on the treewidth.
    upper : int
        Upper bound obtained from a heuristic elimination ordering.
    exact : bool
        ``True`` when ``lower == upper``, meaning the treewidth is known
        exactly.
    """

    lower: int
    upper: int
    exact: bool = False

    def __post_init__(self) -> None:
        if self.lower < 0:
            raise ValueError("Lower bound must be non-negative.")
        if self.lower > self.upper:
            raise ValueError("Lower bound exceeds upper bound.")


@dataclass(slots=True)
class SimulationConfig:
    """Configuration for a synthetic-data simulation experiment.

    Bundles all parameters controlling data generation so that experiments
    are reproducible and can be serialized for logging.

    Attributes
    ----------
    n_samples : int
        Number of observations to generate per replicate.
    n_replicates : int
        Number of independent datasets to generate.
    n_nodes : int
        Number of variables in the simulated DAG.
    edge_density : float
        Expected edge density in ``[0, 1]`` for random DAG generation.
    noise_type : str
        Noise distribution family (``"gaussian"``, ``"uniform"``, ``"t"``).
    noise_scale : float
        Scale parameter for the noise distribution.
    functional_form : str
        Structural equation type (``"linear"``, ``"additive_nonlinear"``).
    coefficient_range : tuple[float, float]
        ``(min, max)`` range for randomly drawn edge coefficients.
    treatment : NodeId
        Index of the treatment variable.
    outcome : NodeId
        Index of the outcome variable.
    seed : int
        Master random seed (child seeds derived deterministically).
    """

    n_samples: int = 1000
    n_replicates: int = 100
    n_nodes: int = 10
    edge_density: float = 0.3
    noise_type: str = "gaussian"
    noise_scale: float = 1.0
    functional_form: str = "linear"
    coefficient_range: tuple[float, float] = (0.5, 2.0)
    treatment: NodeId = 0
    outcome: NodeId = 1
    seed: int = 42


@dataclass(frozen=True, slots=True)
class PowerEnvelope:
    """Statistical power envelope over a range of sample sizes.

    Records how the power to detect a non-zero robustness radius varies
    with sample size, enabling analysts to plan adequately powered studies.

    Attributes
    ----------
    sample_sizes : tuple[int, ...]
        Sample sizes at which power was evaluated.
    power_values : tuple[float, ...]
        Estimated power at each sample size (values in ``[0, 1]``).
    target_power : float
        Desired power level (e.g. ``0.80``).
    required_n : int | None
        Smallest sample size achieving ``target_power``, or ``None`` if
        the target was not reached within the evaluated range.
    alpha : float
        Significance level used in the power calculation.
    effect_size : float
        True robustness radius (or surrogate) used for the power curve.
    """

    sample_sizes: tuple[int, ...]
    power_values: tuple[float, ...]
    target_power: float = 0.80
    required_n: int | None = None
    alpha: float = 0.05
    effect_size: float = 1.0

    def __post_init__(self) -> None:
        if len(self.sample_sizes) != len(self.power_values):
            raise ValueError(
                "sample_sizes and power_values must have the same length."
            )
